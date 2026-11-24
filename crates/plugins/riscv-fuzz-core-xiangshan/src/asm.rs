use riscv_fuzz_plugin_api::{ConfigError, ISABase};

pub fn linker_script() -> &'static str {
    r#"OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
  . = 0x80000000;

  .text.init : {
    KEEP(*(.text.init))
    *(.text)
    *(.text.*)
  }

  . = ALIGN(0x1000);
  .tohost : {
    KEEP(*(.tohost))
  }

  /* Keep the fuzz data pool at a fixed early-RAM address so the plugin's
   * program_mem_range stays in sync even as .text size grows for larger dual-hart
   * sync tests. */
  . = MAX(., 0x80010000);
  .data : {
    *(.data)
    *(.data.*)
  }

  .bss (NOLOAD) : ALIGN(16) {
    *(.bss)
    *(.bss.*)
    *(COMMON)
  }

  PROVIDE(_end = .);
  PROVIDE(__stack = stack_top);
}"#
}

/// Build the assembly file for a XiangShan run.
///
/// `per_hart` is an explicit per-hart slice of asm lines: each tuple is
/// `(hart_id, lines_for_that_hart)`. The list determines the hart layout —
/// if it contains a single entry the single-hart template is emitted,
/// otherwise the dual-hart dispatch template is emitted. No marker parsing
/// is performed.
pub fn build_asm(per_hart: &[(u32, &[String])], isa_base: ISABase) -> Result<String, ConfigError> {
    if isa_base != ISABase::Rv64 {
        return Err(ConfigError::UnsupportedIsaBase {
            core_id: "XiangShan".to_string(),
            isa_base: format!("{isa_base:?}"),
        });
    }

    let hart0_insts: Vec<String> = per_hart
        .iter()
        .find(|(id, _)| *id == 0)
        .map(|(_, lines)| lines.to_vec())
        .unwrap_or_default();
    let hart1_insts: Vec<String> = per_hart
        .iter()
        .find(|(id, _)| *id == 1)
        .map(|(_, lines)| lines.to_vec())
        .unwrap_or_default();
    let harts = per_hart.len() as u32;
    let merged_lines = {
        let mut merged = hart0_insts.clone();
        merged.extend(hart1_insts.clone());
        merged
    };
    let instructions_string = if merged_lines.is_empty() {
        "nop".to_string()
    } else {
        merged_lines.join("\n")
    };
    let hart0_insts_string = if hart0_insts.is_empty() {
        "nop".to_string()
    } else {
        hart0_insts.join("\n")
    };
    let hart1_insts_string = if hart1_insts.is_empty() {
        "nop".to_string()
    } else {
        hart1_insts.join("\n")
    };

    let trap_install = "    la      t0, trap_handler\n    csrw    mtvec, t0\n";
    // Do not emit XiangShan-private CSR 0x5c4 here. When the emulator runs with
    // its Spike DiffTest reference enabled to produce commit logs, that private
    // CSR aborts before testcase_code and surfaces as an init mismatch.
    let disable_fusion = "";

    let perf_start_h0 = "";
    let perf_end_h0 = "";
    let perf_start_h1 = "";
    let perf_end_h1 = "";
    let perf_data = "";

    if harts == 1 {
        return Ok(format!(
            r#"    .section .text
    .globl  _start

_start:
{trap_install}
    csrr    t0, mstatus
    li      t1, 0x00007600       # enable FPU/vector state (FS/VS = Dirty)
    or      t0, t0, t1
    csrw    mstatus, t0
    csrw    fcsr, x0
{disable_fusion}
    j       testcase_code

testcase_code:
{perf_start_h0}
{instructions_string}
{perf_end_h0}
testcase_code_end:
    nop

exit:
    li      t0, 1
    la      t1, skiptrap_store_buf
    sd      t0, 0(t1)

    # DiffTest STATE_GOODTRAP
    li      t0, 0
    .insn   i 0x6b, 0, x0, t0, 0
1:
    j       1b

    .align  2
trap_handler:
    csrr    t0, mepc
    csrr    t1, mcause
    csrr    t4, mtval

    slli    t5, t1, 1
    srli    t1, t5, 1

    li      t2, 2
    li      t3, 2
    beq     t1, t3, use_mtval
    li      t3, 1
    beq     t1, t3, fetch_inst
    li      t3, 12
    beq     t1, t3, fetch_inst

fetch_inst:
    lhu     t4, 0(t0)
    j       decode_length

use_mtval:
    j       decode_length

decode_length:
    andi    t4, t4, 3
    li      t3, 3
    bne     t4, t3, compressed_len
    li      t2, 4
    j       update_mepc

compressed_len:
    li      t2, 2

update_mepc:
    add     t0, t0, t2
    csrw    mepc, t0
    csrw    mcause, x0
    csrw    mtval, x0
    mret

    .section .data
    .align  3
skiptrap_store_buf:
    .dword  0

    .align  6
fuzz_mem_pool:
    .space 4096
{perf_data}
"#,
            trap_install = trap_install,
            disable_fusion = disable_fusion,
            perf_start_h0 = perf_start_h0,
            perf_end_h0 = perf_end_h0,
            instructions_string = indent_testcase_code(&instructions_string),
            perf_data = perf_data,
        ));
    }

    // In XiangShan dual-hart mode we prefer WFI for parked harts.
    // Tight spin loops can consume the global instruction budget (`--max-instr`) before the
    // exiting hart reaches GOODTRAP, which breaks strict+perf collection.
    let park_insn = "wfi";

    // XiangShan 2-hart runs must not let whichever hart reaches `exit` first terminate the
    // whole simulation. Doing so can cut off the other hart while it is still retiring the
    // final testcase instruction or trap-return sequence, which leaves the strict trace missing
    // reg/mem/exception/timing data for the tail instruction.
    //
    // Keep the rendezvous explicit: hart0 waits for hart1's done flag, hart1 parks after
    // publishing completion, and a bounded timeout escalates to BADTRAP instead of silently
    // spinning until the emulator-level instruction/cycle caps fire.
    let barrier_data = "    .section .data\n    .align  3\nstart_flag:\n    .dword 0\nskiptrap_store_buf:\n    .dword 0\nskiptrap_hart0_done:\n    .dword 0\nskiptrap_hart1_done:\n    .dword 0\n\n    .align  6\nfuzz_mem_pool:\n    .space 4096\n"
        .to_string();

    Ok(format!(
        r#"    .section .text
    .globl  _start

 _start:
{trap_install}
    csrr    t0, mstatus
    li      t1, 0x00007600
    or      t0, t0, t1
    csrw    mstatus, t0
    csrw    fcsr, x0
{disable_fusion}
     csrr    a0, mhartid
     bnez    a0, hart1

   hart0:
       la      t0, start_flag
       sd      x0, 0(t0)
       fence   rw, rw
       la      t0, skiptrap_hart0_done
       sd      x0, 0(t0)
       fence   rw, rw
       la      t0, skiptrap_hart1_done
       sd      x0, 0(t0)
       fence   rw, rw
       li      t1, 1
       la      t0, start_flag
       sd      t1, 0(t0)
       fence   rw, rw

   release_testcase_code:
       /* Let hart1 leave the start barrier before entering the measured payload. */
       .rept 32
       nop
       .endr

   testcase_code:
   {perf_start_h0}
   {hart0_insts_string}
   {perf_end_h0}
   testcase_code_end:
    nop
       li      t0, 1
       la      t1, skiptrap_hart0_done
       sd      t0, 0(t1)
       fence   rw, rw

       li      t6, 4096
 wait_hart1_done:
       fence   rw, rw
       la      t1, skiptrap_hart1_done
       ld      t2, 0(t1)
       bnez    t2, exit
       addi    t6, t6, -1
       bnez    t6, wait_hart1_done
       j       timeout_fail

   hart1:
       li      t6, 4096
 wait_start:
       la      t0, start_flag
       fence   rw, rw
       ld      t2, 0(t0)
       fence   rw, rw
       bnez    t2, hart1_go
       addi    t6, t6, -1
       bnez    t6, wait_start
       j       park
 hart1_go:
   hart1_testcase_code:
   {perf_start_h1}
   {hart1_insts_string}
   {perf_end_h1}
   hart1_testcase_code_end:
    nop
       li      t0, 1
       la      t1, skiptrap_hart1_done
       sd      t0, 0(t1)
       fence   rw, rw
       j       park

  exit:
     li      t0, 1
     la      t1, skiptrap_store_buf
     sd      t0, 0(t1)

     # Good trap
     li      t0, 0
     .insn   i 0x6b, 0, x0, t0, 0
 1:
     j       1b

 timeout_fail:
     # DiffTest STATE_BADTRAP
     li      t0, 1
     .insn   i 0x6b, 0, x0, t0, 0
 2:
     j       2b

  park:
  1:
      {park_insn}
      j       1b

    .align  2
trap_handler:
    csrr    t0, mepc
    csrr    t1, mcause
    csrr    t4, mtval

    slli    t5, t1, 1
    srli    t1, t5, 1

    li      t2, 2
    li      t3, 2
    beq     t1, t3, use_mtval
    li      t3, 1
    beq     t1, t3, fetch_inst
    li      t3, 12
    beq     t1, t3, fetch_inst

fetch_inst:
    lhu     t4, 0(t0)
    j       decode_length

use_mtval:
    j       decode_length

decode_length:
    andi    t4, t4, 3
    li      t3, 3
    bne     t4, t3, compressed_len
    li      t2, 4
    j       update_mepc

compressed_len:
    li      t2, 2

update_mepc:
    add     t0, t0, t2
    csrw    mepc, t0
    csrw    mcause, x0
    csrw    mtval, x0
    mret

 {barrier_data}
 {perf_data}
"#,
        trap_install = trap_install,
        disable_fusion = disable_fusion,
        hart0_insts_string = indent_testcase_code(&hart0_insts_string),
        hart1_insts_string = indent_testcase_code(&hart1_insts_string),
        perf_start_h0 = perf_start_h0,
        perf_end_h0 = perf_end_h0,
        perf_start_h1 = perf_start_h1,
        perf_end_h1 = perf_end_h1,
        barrier_data = barrier_data,
        park_insn = park_insn,
        perf_data = perf_data,
    ))
}

fn indent_testcase_code(code: &str) -> String {
    code.lines()
        .map(|line| {
            if line.trim().is_empty() {
                "    ".to_string()
            } else {
                format!("    {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_hart_program_avoids_xiangshan_private_csr_before_payload() {
        let hart0: Vec<String> = vec!["addi x1, x1, 1".to_string()];
        let per_hart: [(u32, &[String]); 1] = [(0, hart0.as_slice())];
        let asm = build_asm(&per_hart, ISABase::Rv64).expect("single-hart asm");

        let private_csr = "csrr    t2, 0x5c4";
        let payload = "testcase_code:";
        assert!(!asm.contains(private_csr));
        assert!(asm.contains(payload));
    }

    #[test]
    fn dual_hart_program_waits_for_other_hart_before_goodtrap() {
        let hart0: Vec<String> = vec!["addi x1, x1, 1".to_string()];
        let hart1: Vec<String> = vec!["addi x2, x2, 1".to_string()];
        let per_hart: [(u32, &[String]); 2] = [(0, hart0.as_slice()), (1, hart1.as_slice())];
        let asm = build_asm(&per_hart, ISABase::Rv64).expect("dual-hart asm");

        assert!(asm.contains("start_flag:"));
        assert!(asm.contains("wait_start:"));
        assert!(asm.contains("release_testcase_code:"));
        assert!(asm.contains(".rept 32"));
        assert!(asm.contains("wait_hart1_done:"));
        assert!(asm.contains("ld      t2, 0(t1)"));
        assert!(asm.contains("bnez    t2, exit"));
        assert!(asm.contains("timeout_fail:"));
        assert!(asm.contains("# DiffTest STATE_BADTRAP"));
        assert!(asm.contains("j       park"));
        assert!(!asm.contains("# opencode:hart"));
    }
}
