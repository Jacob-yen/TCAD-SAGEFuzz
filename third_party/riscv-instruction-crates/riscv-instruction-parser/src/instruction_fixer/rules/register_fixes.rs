//! Register-placeholder repair rules.
//!
//! unified-db occasionally uses the wrong register family in operand names,
//! e.g. integer placeholders where the ISA semantics clearly require floating
//! point or vector registers. These fixes keep the IR semantically aligned.

use riscv_instruction_schema::*;
use std::collections::HashSet;

use super::super::naming::{fix_operand_name, update_syntax_operands};

/// `vsetvli` sometimes mislabels the 11-bit immediate-like operand as `xs1`.
pub(super) fn fix_vsetvli_operand_names(instruction: &mut Instruction) {
    if instruction.name == "vsetvli" {
        instruction
            .operands
            .iter_mut()
            .find(|op| op.name == "xs1" && op.bit_lengths.get(&ISABase::RV32) == Some(&11))
            .map(|op| op.name = "vtypei".to_string());
    }
}

/// Small vector-specific register fixes that do not justify a larger table.
pub(super) fn fix_vector_register_errors_wrapper(
    instruction: &mut Instruction,
    operands_part: &mut String,
    final_syntax_operands: &mut HashSet<String>,
) {
    match instruction.name.as_str() {
        "vmv.x.s" | "vcpop.m" => {
            fix_operand_name(instruction, "vd", "xd");
            update_syntax_operands(operands_part, final_syntax_operands, "vd", "xd");
        }
        _ => {}
    }
}

/// Repair instructions whose operand names use integer register placeholders
/// even though the ISA semantics clearly require floating-point placeholders.
///
/// This is intentionally explicit and table-driven. These are upstream data
/// issues, not inference-heavy heuristics.
pub(super) fn fix_floating_point_register_errors(
    instruction: &mut Instruction,
    operands_part: &mut String,
    final_syntax_operands: &mut HashSet<String>,
) {
    match instruction.name.as_str() {
        "fcvt.d.q" => {
            fix_operand_name(instruction, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
        }
        "fcvt.h.q" => {
            fix_operand_name(instruction, "xd", "hd");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "hd");
        }
        "fcvt.lu.q" => {
            fix_operand_name(instruction, "qd", "xd");
            update_syntax_operands(operands_part, final_syntax_operands, "qd", "xd");
        }
        "fcvt.wu.q" => {
            fix_operand_name(instruction, "xs1", "qs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "qs1");
        }
        "fmin.q" => {
            fix_operand_name(instruction, "xd", "qd");
            fix_operand_name(instruction, "xs1", "qs1");
            fix_operand_name(instruction, "xs2", "qs2");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "qd");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "qs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs2", "qs2");
        }
        "fmaxm.s" | "fmin.s" => {
            fix_operand_name(instruction, "xd", "fd");
            fix_operand_name(instruction, "xs1", "fs1");
            fix_operand_name(instruction, "xs2", "fs2");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs2", "fs2");
        }
        "fnmsub.s" => {
            fix_operand_name(instruction, "xd", "fd");
            fix_operand_name(instruction, "xs1", "fs1");
            fix_operand_name(instruction, "xs2", "fs2");
            fix_operand_name(instruction, "xs3", "fs3");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs2", "fs2");
            update_syntax_operands(operands_part, final_syntax_operands, "xs3", "fs3");
        }
        "fround.s" | "froundnx.s" => {
            fix_operand_name(instruction, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
        }
        "fadd.d" | "fsub.d" | "fmul.d" | "fdiv.d" => {
            fix_operand_name(instruction, "xd", "fd");
            fix_operand_name(instruction, "xs1", "fs1");
            fix_operand_name(instruction, "xs2", "fs2");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs2", "fs2");
        }
        "fsqrt.d" | "fround.d" | "froundnx.d" => {
            fix_operand_name(instruction, "xd", "fd");
            fix_operand_name(instruction, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
        }
        "fmadd.d" | "fmsub.d" | "fnmadd.d" | "fnmsub.d" => {
            fix_operand_name(instruction, "xd", "fd");
            fix_operand_name(instruction, "xs1", "fs1");
            fix_operand_name(instruction, "xs2", "fs2");
            fix_operand_name(instruction, "xs3", "fs3");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs2", "fs2");
            update_syntax_operands(operands_part, final_syntax_operands, "xs3", "fs3");
        }
        "fli.d" => {
            fix_operand_name(instruction, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
        }
        "fcvt.d.l" | "fcvt.d.lu" | "fcvt.d.w" | "fcvt.d.wu" => {
            fix_operand_name(instruction, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
        }
        "fcvt.l.d" | "fcvt.lu.d" | "fcvt.w.d" | "fcvt.wu.d" | "fcvtmod.w.d" => {
            fix_operand_name(instruction, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
        }
        "fcvt.d.s" | "fcvt.s.d" => {
            fix_operand_name(instruction, "xd", "fd");
            fix_operand_name(instruction, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
        }
        "feq.d" | "fle.d" | "fleq.d" | "flt.d" | "fltq.d" => {
            fix_operand_name(instruction, "xs1", "fs1");
            fix_operand_name(instruction, "xs2", "fs2");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs2", "fs2");
        }
        "fclass.d" => {
            fix_operand_name(instruction, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
        }
        "fmax.d" | "fmaxm.d" | "fmin.d" | "fminm.d" | "fsgnj.d" | "fsgnjn.d" | "fsgnjx.d" => {
            fix_operand_name(instruction, "xd", "fd");
            fix_operand_name(instruction, "xs1", "fs1");
            fix_operand_name(instruction, "xs2", "fs2");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs2", "fs2");
        }
        "fmv.d.x" => {
            fix_operand_name(instruction, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
        }
        "fmv.x.d" | "fmvh.x.d" => {
            fix_operand_name(instruction, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
        }
        "fmvp.d.x" => {
            fix_operand_name(instruction, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
        }
        "c.fld" | "c.flw" => {
            fix_operand_name(instruction, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
        }
        "c.fsd" | "c.fsw" => {
            fix_operand_name(instruction, "xs2", "fs2");
            update_syntax_operands(operands_part, final_syntax_operands, "xs2", "fs2");
        }
        "fcvt.bf16.s" | "fcvt.s.bf16" => {
            fix_operand_name(instruction, "xd", "fd");
            fix_operand_name(instruction, "xs1", "fs1");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xs1", "fs1");
        }
        "fclass.h" | "fcvt.l.h" | "fcvt.lu.h" | "fcvt.w.h" | "fcvt.wu.h" | "feq.h" | "fle.h"
        | "fleq.h" | "flt.h" | "fltq.h" => {
            fix_operand_name(instruction, "fd", "xd");
            update_syntax_operands(operands_part, final_syntax_operands, "fd", "xd");
        }
        "flh" | "fcvt.h.wu" | "fcvt.h.w" | "fcvt.h.lu" | "fcvt.h.l" | "fmv.h.x" => {
            fix_operand_name(instruction, "xd", "fd");
            update_syntax_operands(operands_part, final_syntax_operands, "xd", "fd");
        }
        "fmv.x.h" | "fltq.q" => {
            fix_operand_name(instruction, "fd", "xd");
            update_syntax_operands(operands_part, final_syntax_operands, "fd", "xd");
        }
        _ => {}
    }
}

/// Latest unified-db already split most Zalrsc aq/rl encodings into dedicated
/// instruction aliases. Keeping `aq`/`rl` as random operands would append the
/// suffix twice during assembly generation.
///
/// `ssamoswap.*` is still modeled as a single instruction definition, so its
/// `aq`/`rl` operands must stay.
pub(super) fn fix_fixed_aqrl_operands(instruction: &mut Instruction) {
    if matches!(instruction.name.as_str(), "ssamoswap.d" | "ssamoswap.w") {
        return;
    }

    if !instruction
        .operands
        .iter()
        .any(|operand| operand.name == "aq" || operand.name == "rl")
    {
        return;
    }

    instruction
        .operands
        .retain(|operand| operand.name != "aq" && operand.name != "rl");
}
