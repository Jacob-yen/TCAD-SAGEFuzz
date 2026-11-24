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

/// Build the assembly file for a Spike run.
///
/// `per_hart` is an explicit per-hart slice of asm lines: each tuple is
/// `(hart_id, lines_for_that_hart)`. The list determines the hart layout —
/// if it contains a single entry the single-hart template is emitted,
/// otherwise the dual-hart dispatch template is emitted. No marker parsing
/// is performed.
pub fn build_asm(per_hart: &[(u32, &[String])], isa_base: ISABase) -> Result<String, ConfigError> {
    build_asm_with_vector_state(per_hart, isa_base, false)
}

pub fn build_asm_with_vector_state(
    per_hart: &[(u32, &[String])],
    isa_base: ISABase,
    enable_vector_state: bool,
) -> Result<String, ConfigError> {
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

    let (store_instr, tohost_line) = match isa_base {
        ISABase::Rv32 => ("sw", ".word   0"),
        ISABase::Rv64 => ("sd", ".dword  0"),
    };
    let load_instr = match isa_base {
        ISABase::Rv32 => "lw",
        ISABase::Rv64 => "ld",
    };

    let perf_start_h0 = "";
    let perf_end_h0 = "";
    let perf_start_h1 = "";
    let perf_end_h1 = "";
    let perf_data = "";
    let (mstatus_dirty_bits, mstatus_dirty_comment) = if enable_vector_state {
        (
            "0x00007600",
            "set FS/VS fields to Dirty so FP/vector regs usable",
        )
    } else {
        ("0x00003000", "set FS field to Dirty so FP regs usable")
    };

    if harts <= 1 {
        return Ok(format!(
            r#"    .section .text
    .globl  _start

_start:
    la      t0, trap_handler
    csrw    mtvec, t0

    csrr    t0, mstatus
    li      t1, {mstatus_dirty_bits}       # {mstatus_dirty_comment}
    or      t0, t0, t1
    csrw    mstatus, t0
    csrw    fcsr, x0             # clear floating-point status

    j       testcase_code

testcase_code:
{perf_start_h0}
{instructions_string}
{perf_end_h0}

exit:
    li      t0, 1                # report success
    la      t1, tohost
    {store_instr}      t0, 0(t1)
1:
    j       1b

    .align  2
trap_handler:
    csrr    t0, mepc             # faulting PC
    csrr    t1, mcause           # trap cause
    csrr    t4, mtval

    slli    t5, t1, 1            # strip interrupt bit
    srli    t1, t5, 1

    li      t2, 2                # default length assumes compressed
    li      t3, 2                # illegal instruction -> mtval holds encoding
    beq     t1, t3, use_mtval
    li      t3, 1                # instruction access fault
    beq     t1, t3, update_mepc
    li      t3, 12               # instruction page fault
    beq     t1, t3, update_mepc

    lhu     t4, 0(t0)
    j       decode_length

use_mtval:
    j       decode_length

decode_length:
    andi    t4, t4, 3
    li      t3, 3
    bne     t4, t3, compressed_len
    li      t2, 4                # standard 32-bit instruction
    j       update_mepc

compressed_len:
    li      t2, 2                # compressed instruction

update_mepc:
    add     t0, t0, t2           # skip offending instruction
    csrw    mepc, t0
    csrw    mcause, x0
    csrw    mtval, x0
    mret

    .section .tohost,"aw",@progbits
    .align  6
    .globl  tohost
    .globl  fromhost
tohost:
    {tohost_line}
fromhost:
    {tohost_line}
{perf_data}"#,
            store_instr = store_instr,
            tohost_line = tohost_line,
            perf_start_h0 = perf_start_h0,
            perf_end_h0 = perf_end_h0,
            perf_data = perf_data,
            mstatus_dirty_bits = mstatus_dirty_bits,
            mstatus_dirty_comment = mstatus_dirty_comment,
            instructions_string = indent_testcase_code(&instructions_string),
        ));
    }

    Ok(format!(
        r#"    .section .text
    .globl  _start

_start:
    la      t0, trap_handler
    csrw    mtvec, t0

    csrr    t0, mstatus
    li      t1, {mstatus_dirty_bits}       # {mstatus_dirty_comment}
    or      t0, t0, t1
    csrw    mstatus, t0
    csrw    fcsr, x0             # clear floating-point status

    csrr    a0, mhartid
    bnez    a0, hart1

hart0:
    la      t0, start_flag
    {store_instr}      x0, 0(t0)
    la      t0, done0
    {store_instr}      x0, 0(t0)
    la      t0, done1
    {store_instr}      x0, 0(t0)
    fence   rw, rw
    li      t1, 1
    la      t0, start_flag
    {store_instr}      t1, 0(t0)
    fence   rw, rw

hart0_testcase_code:
{perf_start_h0}
{hart0_insts_string}
{perf_end_h0}
    li      t1, 1
    la      t0, done0
    {store_instr}      t1, 0(t0)
    fence   rw, rw

    li      t6, 200000
wait_done1:
    la      t0, done1
    {load_instr}      t2, 0(t0)
    bnez    t2, hart0_post_wait
    addi    t6, t6, -1
    bnez    t6, wait_done1
    j       timeout_fail

hart0_post_wait:
    li      t3, {exit_hart}
    beqz    t3, exit
    j       park0

hart1:
    li      t6, 200000
wait_start:
    la      t0, start_flag
    {load_instr}      t2, 0(t0)
    bnez    t2, hart1_testcase_code
    addi    t6, t6, -1
    bnez    t6, wait_start
    j       park1

hart1_testcase_code:
{perf_start_h1}
{hart1_insts_string}
{perf_end_h1}
    li      t1, 1
    la      t0, done1
    {store_instr}      t1, 0(t0)
    fence   rw, rw

    li      t6, 200000
wait_done0:
    la      t0, done0
    {load_instr}      t2, 0(t0)
    bnez    t2, hart1_post_wait
    addi    t6, t6, -1
    bnez    t6, wait_done0
    j       timeout_fail

hart1_post_wait:
    li      t3, {exit_hart}
    li      t4, 1
    beq     t3, t4, exit
    j       park1

timeout_fail:
    li      t0, 2
    la      t1, tohost
    {store_instr}      t0, 0(t1)
3:
    j       3b

exit:
    li      t0, 1                # report success
    la      t1, tohost
    {store_instr}      t0, 0(t1)
1:
    j       1b

park0:
2:
    wfi
    j       2b

park1:
4:
    wfi
    j       4b

    .align  2
trap_handler:
    csrr    t0, mepc             # faulting PC
    csrr    t1, mcause           # trap cause
    csrr    t4, mtval

    slli    t5, t1, 1            # strip interrupt bit
    srli    t1, t5, 1

    li      t2, 2                # default length assumes compressed
    li      t3, 2                # illegal instruction -> mtval holds encoding
    beq     t1, t3, use_mtval
    li      t3, 1                # instruction access fault
    beq     t1, t3, update_mepc
    li      t3, 12               # instruction page fault
    beq     t1, t3, update_mepc

    lhu     t4, 0(t0)
    j       decode_length

use_mtval:
    j       decode_length

decode_length:
    andi    t4, t4, 3
    li      t3, 3
    bne     t4, t3, compressed_len
    li      t2, 4                # standard 32-bit instruction
    j       update_mepc

compressed_len:
    li      t2, 2                # compressed instruction

update_mepc:
    add     t0, t0, t2           # skip offending instruction
    csrw    mepc, t0
    csrw    mcause, x0
    csrw    mtval, x0
    mret

    .section .tohost,"aw",@progbits
    .align  6
    .globl  tohost
    .globl  fromhost
tohost:
    {tohost_line}
fromhost:
    {tohost_line}
{perf_data}

    .section .data
    .align  3
start_flag:
    {tohost_line}
done0:
    {tohost_line}
done1:
    {tohost_line}"#,
        store_instr = store_instr,
        load_instr = load_instr,
        exit_hart = 0,
        tohost_line = tohost_line,
        perf_start_h0 = perf_start_h0,
        perf_end_h0 = perf_end_h0,
        perf_start_h1 = perf_start_h1,
        perf_end_h1 = perf_end_h1,
        perf_data = perf_data,
        mstatus_dirty_bits = mstatus_dirty_bits,
        mstatus_dirty_comment = mstatus_dirty_comment,
        hart0_insts_string = indent_testcase_code(&hart0_insts_string),
        hart1_insts_string = indent_testcase_code(&hart1_insts_string),
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
    fn multicore_template_includes_hart_dispatch() {
        let hart0: Vec<String> = vec!["addi x5, x0, 1".to_string()];
        let hart1: Vec<String> = vec!["addi x6, x0, 2".to_string()];
        let per_hart: [(u32, &[String]); 2] = [(0, hart0.as_slice()), (1, hart1.as_slice())];

        let asm = build_asm(&per_hart, ISABase::Rv64)
            .expect("spike multicore template should be generated");

        assert!(asm.contains("csrr    a0, mhartid"));
        assert!(asm.contains("hart1_testcase_code:"));
        assert!(asm.contains("addi x5, x0, 1"));
        assert!(asm.contains("addi x6, x0, 2"));
        assert!(!asm.contains("# opencode:hart"));
    }

    #[test]
    fn singlecore_template_omits_hart_dispatch() {
        let hart0: Vec<String> = vec!["addi x5, x0, 1".to_string()];
        let per_hart: [(u32, &[String]); 1] = [(0, hart0.as_slice())];

        let asm = build_asm(&per_hart, ISABase::Rv64).expect("singlecore asm should build");

        assert!(!asm.contains("csrr    a0, mhartid"));
        assert!(!asm.contains("hart1_testcase_code:"));
        assert!(asm.contains("addi x5, x0, 1"));
    }

    #[test]
    fn vector_state_enable_is_explicit() {
        let hart0: Vec<String> = vec!["vsetvli x0, x0, 24".to_string()];
        let per_hart: [(u32, &[String]); 1] = [(0, hart0.as_slice())];

        let default_asm =
            build_asm(&per_hart, ISABase::Rv64).expect("default spike asm should build");
        let vector_asm = build_asm_with_vector_state(&per_hart, ISABase::Rv64, true)
            .expect("vector-enabled spike asm should build");

        assert!(default_asm.contains("li      t1, 0x00003000"));
        assert!(!default_asm.contains("0x00007600"));
        assert!(vector_asm.contains("li      t1, 0x00007600"));
    }
}
