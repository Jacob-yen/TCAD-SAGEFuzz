use std::env;

use riscv_fuzz_plugin_api::{ConfigError, ISABase};

const SIMPLE_TRAP_ENV: &str = "RISCV_FUZZ_ARA_SIMPLE_TRAP";

fn env_flag_enabled(name: &str) -> bool {
    matches!(
        env::var(name)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on")
    )
}

fn ara_trap_handler() -> &'static str {
    if env_flag_enabled(SIMPLE_TRAP_ENV) {
        SIMPLE_TRAP_HANDLER
    } else {
        DEFAULT_TRAP_HANDLER
    }
}

const SIMPLE_TRAP_HANDLER: &str = r#"    .align   2
 trap_handler:
    csrr     t0, mepc               # offending PC
    addi     t0, t0, 4              # generated RVV tests use 32-bit instructions
    csrw     mepc, t0
    csrw     mcause, x0
    csrw     mtval, x0
    csrw     mip, x0
    mret
"#;

const DEFAULT_TRAP_HANDLER: &str = r#"    .align   2
 trap_handler:
    csrr     t0, mepc               # offending PC
    csrr     t1, mcause             # trap cause
    slli     t5, t1, 1              # clear interrupt bit -> synchronous cause only
    srli     t1, t5, 1

    li       t2, 2                  # default length (compressed)
    li       t3, 2                  # illegal instruction -> mtval holds encoding
    beq      t1, t3, use_mtval
    li       t3, 1                  # instruction access fault
    beq      t1, t3, update_mepc
    li       t3, 12                 # instruction page fault
    beq      t1, t3, update_mepc

    lhu      t4, 0(t0)              # fetch lower half-word of instruction
    j        decode_length

 use_mtval:
    csrr     t4, mtval              # mtval contains the faulting instruction

 decode_length:
    andi     t4, t4, 3
    li       t3, 3
    bne      t4, t3, compressed_len
    li       t2, 4                  # standard 32-bit instruction
    j        update_mepc

 compressed_len:
    li       t2, 2                  # compressed instruction

 update_mepc:
    add      t0, t0, t2             # skip offending instruction
    csrw     mepc, t0
    csrw     mcause, x0
    csrw     mtval, x0
    csrw     mip, x0
    mret
"#;

pub fn linker_script() -> &'static str {
    r#"OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS {
  . = 0x80000000;
  .text : { *(.text*) }

  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }

  . = ALIGN(0x1000);
  .data : { *(.data*) }
  .bss  : { *(.bss*) *(COMMON) }
}"#
}

pub fn build_asm(per_hart: &[(u32, &[String])], isa_base: ISABase) -> Result<String, ConfigError> {
    if isa_base != ISABase::Rv64 {
        return Err(ConfigError::UnsupportedIsaBase {
            core_id: "ara".to_string(),
            isa_base: isa_base.to_string(),
        });
    }
    if per_hart.len() != 1 || per_hart.first().map(|(hart, _)| *hart) != Some(0) {
        return Err(ConfigError::InvalidConfiguration {
            message: "ARA plugin currently supports one RV64 hart only".to_string(),
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
    let barrier_is_amo_counter = harts == 2;
    let barrier_is_mem_flag = false;
    let barrier_is_clint_msip = false;
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

    let trap_install = "    la       t0, trap_handler\n    csrw     mtvec, t0\n";

    let perf_start_h0 = "";
    let perf_end_h0 = "";
    let perf_start_h1 = "";
    let perf_end_h1 = "";
    let perf_data = "";
    let trap_handler_body = ara_trap_handler();

    if harts == 1 {
        return Ok(format!(
            "    .section .text\n    .globl   _start\n\n _start:\n{trap_install}\n    csrr     t0, mstatus\n    li       t1, 0x00007600         # enable FPU/vector state (FS/VS = Dirty)\n    or       t0, t0, t1\n    csrw     mstatus, t0\n    csrw     fcsr, x0               # clear floating-point status\n\n    j        testcase_code\n\n testcase_code:\n{perf_start_h0}\n{instructions_string}\n{perf_end_h0}\n\n exit:\n    li       t0, 0xD0000000         # ARA control register base\n    {store_instr}       x0, 0(t0)   # exit code 0\n 1:\n    j        1b\n\n{trap_handler_body}\n    .section .tohost,\"aw\",@progbits\n    .align   6\n    .globl   tohost\n    .globl   fromhost\n tohost:\n    {tohost_line}\n fromhost:\n    {tohost_line}\n {perf_data}\n ",
            trap_install = trap_install,
            perf_start_h0 = perf_start_h0,
            perf_end_h0 = perf_end_h0,
            instructions_string = indent_testcase_code(&instructions_string),
            store_instr = store_instr,
            trap_handler_body = trap_handler_body,
            tohost_line = tohost_line,
            perf_data = perf_data,
        ));
    }

    let park_insn = "j park";

    if barrier_is_amo_counter {
        let amo_instr = match isa_base {
            ISABase::Rv32 => "amoadd.w",
            ISABase::Rv64 => "amoadd.d",
        };
        let poll_instr = match isa_base {
            ISABase::Rv32 => "amoadd.w.aq",
            ISABase::Rv64 => "amoadd.d.aq",
        };
        return Ok(format!(
            r#"    .section .text
    .globl   _start

 _start:
{trap_install}
    csrr     t0, mstatus
    li       t1, 0x00007600         # enable FPU/vector state (FS/VS = Dirty)
    or       t0, t0, t1
    csrw     mstatus, t0
    csrw     fcsr, x0

    csrr     a0, mhartid
    bnez     a0, hart1

hart0:
testcase_code:
{perf_start_h0}
{hart0_insts_string}
{perf_end_h0}

    li       t1, 1
    la       t0, done_count
    {amo_instr}    x0, t1, (t0)
    fence    rw, rw

    li       t6, 200000
wait_done_count:
    la       t0, done_count
    {poll_instr}    t2, x0, (t0)
    li       t3, 2
    bge      t2, t3, exit
    addi     t6, t6, -1
    bnez     t6, wait_done_count
    j        timeout_fail

exit:
    li       t0, 1
    la       t1, tohost
    {store_instr}       t0, 0(t1)
1:
    j        1b

timeout_fail:
    li       t0, 3
    la       t1, tohost
    {store_instr}       t0, 0(t1)
2:
    j        2b

hart1:
    li       t4, 256
hart1_stagger:
    addi     t4, t4, -1
    bnez     t4, hart1_stagger

hart1_testcase_code:
{perf_start_h1}
{hart1_insts_string}
{perf_end_h1}

    li       t1, 1
    la       t0, done_count
    {amo_instr}    x0, t1, (t0)
    fence    rw, rw
    j        park

park:
    {park_insn}

    .align   2
trap_handler:
    csrr     t0, mepc
    csrr     t1, mcause
    slli     t5, t1, 1
    srli     t1, t5, 1

    li       t2, 2
    li       t3, 2
    beq      t1, t3, use_mtval
    li       t3, 1
    beq      t1, t3, update_mepc
    li       t3, 12
    beq      t1, t3, update_mepc
    lhu      t4, 0(t0)
    j        decode_length

use_mtval:
    csrr     t4, mtval

decode_length:
    andi     t4, t4, 3
    li       t3, 3
    bne      t4, t3, compressed_len
    li       t2, 4
    j        update_mepc

compressed_len:
    li       t2, 2

update_mepc:
    add      t0, t0, t2
    csrw     mepc, t0
    csrw     mcause, x0
    csrw     mtval, x0
    csrw     mip, x0
    mret

    .section .data
    .align   3
done_count:
    {tohost_line}

    .section .tohost,"aw",@progbits
    .align   6
    .globl   tohost
    .globl   fromhost
tohost:
    {tohost_line}
fromhost:
    {tohost_line}
{perf_data}"#,
            trap_install = trap_install,
            hart0_insts_string = indent_testcase_code(&hart0_insts_string),
            hart1_insts_string = indent_testcase_code(&hart1_insts_string),
            perf_start_h0 = perf_start_h0,
            perf_end_h0 = perf_end_h0,
            perf_start_h1 = perf_start_h1,
            perf_end_h1 = perf_end_h1,
            amo_instr = amo_instr,
            poll_instr = poll_instr,
            store_instr = store_instr,
            tohost_line = tohost_line,
            park_insn = park_insn,
            perf_data = perf_data,
        ));
    }

    if barrier_is_mem_flag {
        return Ok(format!(
            r#"    .section .text
    .globl   _start

 _start:
{trap_install}
    csrr     t0, mstatus
    li       t1, 0x00007600         # enable FPU/vector state (FS/VS = Dirty)
    or       t0, t0, t1
    csrw     mstatus, t0
    csrw     fcsr, x0

    csrr     a0, mhartid
    bnez     a0, hart1

hart0:
    la       t0, start_flag
    {store_instr}       x0, 0(t0)
    fence    rw, rw
    la       t0, done1
    {store_instr}       x0, 0(t0)
    fence    rw, rw

    li       t1, 1
    la       t0, start_flag
    {store_instr}       t1, 0(t0)
    fence    rw, rw

testcase_code:
{perf_start_h0}
{hart0_insts_string}
{perf_end_h0}

    # Wait for hart1 done-flag in RAM.
    li       t6, 200000
wait_done1:
    la       t0, done1
    {load_instr}       t2, 0(t0)
    bnez     t2, exit
    addi     t6, t6, -1
    bnez     t6, wait_done1
    j        timeout_fail

exit:
    li       t0, 1
    la       t1, tohost
    {store_instr}       t0, 0(t1)
1:
    j        1b

timeout_fail:
    li       t0, 3
    la       t1, tohost
    {store_instr}       t0, 0(t1)
2:
    j        2b

hart1:
    li       t6, 200000
wait_start:
    la       t0, start_flag
    {load_instr}       t2, 0(t0)
    bnez     t2, hart1_testcase_code
    addi     t6, t6, -1
    bnez     t6, wait_start
    j        park

hart1_testcase_code:
{perf_start_h1}
{hart1_insts_string}
{perf_end_h1}

    li       t1, 1
    la       t0, done1
    {store_instr}       t1, 0(t0)
    fence    rw, rw
    j        park

park:
    {park_insn}

    .align   2
trap_handler:
    csrr     t0, mepc
    csrr     t1, mcause
    slli     t5, t1, 1
    srli     t1, t5, 1

    li       t2, 2
    li       t3, 2
    beq      t1, t3, use_mtval
    li       t3, 1
    beq      t1, t3, update_mepc
    li       t3, 12
    beq      t1, t3, update_mepc
    lhu      t4, 0(t0)
    j        decode_length

use_mtval:
    csrr     t4, mtval

decode_length:
    andi     t4, t4, 3
    li       t3, 3
    bne      t4, t3, compressed_len
    li       t2, 4
    j        update_mepc

compressed_len:
    li       t2, 2

update_mepc:
    add      t0, t0, t2
    csrw     mepc, t0
    csrw     mcause, x0
    csrw     mtval, x0
    csrw     mip, x0
    mret

    .section .data
    .align   3
start_flag:
    {tohost_line}
done1:
    {tohost_line}

    .section .tohost,"aw",@progbits
    .align   6
    .globl   tohost
    .globl   fromhost
tohost:
    {tohost_line}
fromhost:
    {tohost_line}
{perf_data}"#,
            trap_install = trap_install,
            hart0_insts_string = indent_testcase_code(&hart0_insts_string),
            hart1_insts_string = indent_testcase_code(&hart1_insts_string),
            perf_start_h0 = perf_start_h0,
            perf_end_h0 = perf_end_h0,
            perf_start_h1 = perf_start_h1,
            perf_end_h1 = perf_end_h1,
            store_instr = store_instr,
            load_instr = match isa_base {
                ISABase::Rv32 => "lw",
                ISABase::Rv64 => "ld",
            },
            tohost_line = tohost_line,
            park_insn = park_insn,
            perf_data = perf_data,
        ));
    }

    if !barrier_is_clint_msip {
        return Err(ConfigError::InvalidConfiguration {
            message: "ARA multicore template supports only AmoCounter/MemFlag/ClintMsip barriers"
                .to_string(),
        });
    }

    Ok(format!(
        r#"    .section .text
    .globl   _start

    .equ CLINT_BASE, 0x02000000
    .equ MSIP0, (CLINT_BASE + 0x0)
    .equ MSIP1, (CLINT_BASE + 0x4)

 _start:
{trap_install}
    csrr     t0, mstatus
    li       t1, 0x00007600         # enable FPU/vector state (FS/VS = Dirty)
    or       t0, t0, t1
    csrw     mstatus, t0
    csrw     fcsr, x0

    csrr     a0, mhartid
    bnez     a0, hart1

hart0:
    # Reset MSIP state and publish ready (via MSIP1).
    li       t0, MSIP0
    sw       x0, 0(t0)
    li       t0, MSIP1
    sw       x0, 0(t0)
    fence    iorw, iorw
    li       t1, 1
    li       t0, MSIP1
    sw       t1, 0(t0)
    fence    iorw, iorw

testcase_code:
{perf_start_h0}
{hart0_insts_string}
{perf_end_h0}

    # Wait for hart1 signal on MSIP0.
    li       t6, 200000
wait_msip0:
    li       t0, MSIP0
    lw       t2, 0(t0)
    bnez     t2, exit
    addi     t6, t6, -1
    bnez     t6, wait_msip0
    j        timeout_fail

exit:
    li       t0, 1
    la       t1, tohost
    {store_instr}       t0, 0(t1)
1:
    j        1b

timeout_fail:
    li       t0, 3
    la       t1, tohost
    {store_instr}       t0, 0(t1)
2:
    j        2b

hart1:
    # Wait until hart0 publishes ready (MSIP1).
    li       t6, 200000
wait_msip1:
    li       t0, MSIP1
    lw       t2, 0(t0)
    bnez     t2, hart1_testcase_code
    addi     t6, t6, -1
    bnez     t6, wait_msip1
    j        park

hart1_testcase_code:
{perf_start_h1}
{hart1_insts_string}
{perf_end_h1}
    j        signal

signal:
    li       t0, MSIP0
    li       t1, 1
    sw       t1, 0(t0)
    fence    iorw, iorw

park:
    {park_insn}

    .align   2
trap_handler:
    csrr     t0, mepc
    csrr     t1, mcause
    slli     t5, t1, 1
    srli     t1, t5, 1

    li       t2, 2
    li       t3, 2
    beq      t1, t3, use_mtval
    li       t3, 1
    beq      t1, t3, update_mepc
    li       t3, 12
    beq      t1, t3, update_mepc
    lhu      t4, 0(t0)
    j        decode_length

use_mtval:
    csrr     t4, mtval

decode_length:
    andi     t4, t4, 3
    li       t3, 3
    bne      t4, t3, compressed_len
    li       t2, 4
    j        update_mepc

compressed_len:
    li       t2, 2

update_mepc:
    add      t0, t0, t2
    csrw     mepc, t0
    csrw     mcause, x0
    csrw     mtval, x0
    csrw     mip, x0
    mret

    .section .tohost,"aw",@progbits
    .align   6
    .globl   tohost
    .globl   fromhost
tohost:
    {tohost_line}
fromhost:
    {tohost_line}
{perf_data}"#,
        trap_install = trap_install,
        hart0_insts_string = indent_testcase_code(&hart0_insts_string),
        hart1_insts_string = indent_testcase_code(&hart1_insts_string),
        perf_start_h0 = perf_start_h0,
        perf_end_h0 = perf_end_h0,
        perf_start_h1 = perf_start_h1,
        perf_end_h1 = perf_end_h1,
        store_instr = store_instr,
        tohost_line = tohost_line,
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
    use std::sync::Mutex;

    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        name: &'static str,
        previous: Option<String>,
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => unsafe { env::set_var(self.name, value) },
                None => unsafe { env::remove_var(self.name) },
            }
        }
    }

    fn set_env(name: &'static str, value: &str) -> EnvGuard {
        let previous = env::var(name).ok();
        unsafe { env::set_var(name, value) };
        EnvGuard { name, previous }
    }

    fn clear_env(name: &'static str) -> EnvGuard {
        let previous = env::var(name).ok();
        unsafe { env::remove_var(name) };
        EnvGuard { name, previous }
    }

    #[test]
    fn default_ara_trap_handler_keeps_full_decode_logic() {
        let _lock = ENV_LOCK.lock().expect("lock env");
        let _guard = clear_env(SIMPLE_TRAP_ENV);
        let asm = build_asm(&[(0, &["nop".to_string()])], ISABase::Rv64).expect("build Ara asm");
        assert!(asm.contains("decode_length:"));
        assert!(asm.contains("beq      t1, t3, use_mtval"));
    }

    #[test]
    fn simple_ara_trap_handler_removes_wrapper_branches() {
        let _lock = ENV_LOCK.lock().expect("lock env");
        let _guard = set_env(SIMPLE_TRAP_ENV, "1");
        let asm = build_asm(&[(0, &["nop".to_string()])], ISABase::Rv64).expect("build Ara asm");
        assert!(asm.contains("addi     t0, t0, 4"));
        assert!(!asm.contains("decode_length:"));
        assert!(!asm.contains("use_mtval:"));
        assert!(!asm.contains("beq      t1, t3, use_mtval"));
    }
}
