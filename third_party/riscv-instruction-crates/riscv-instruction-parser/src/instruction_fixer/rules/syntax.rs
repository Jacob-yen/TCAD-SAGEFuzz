//! Syntax-shape repair rules.
//!
//! This module directly edits the intermediate assembly representation:
//! `operands_part` and `final_syntax_operands`. These rules are about how the
//! instruction should be printed, not about operand typing.

use riscv_instruction_schema::*;
use std::collections::{HashMap, HashSet};

/// Remove the hard-coded stack pointer placeholder from the final syntax set.
///
/// Instructions such as `c.lwsp` print `sp` literally. Keeping it in the
/// placeholder set would incorrectly turn it into a random operand.
pub(super) fn fix_sp_should_not_in_final_syntax_operands(
    final_syntax_operands: &mut HashSet<String>,
) {
    final_syntax_operands.remove("sp");
}

/// `fence` is rendered with named `pred`/`succ` modes instead of the raw YAML
/// placeholder state that sometimes arrives incomplete.
pub(super) fn fix_fence_instruction(
    operands_part: &mut String,
    instruction: &mut Instruction,
    final_syntax_operands: &mut HashSet<String>,
) {
    if instruction.name != "fence" {
        return;
    }

    *operands_part = "pred, succ".to_string();
    final_syntax_operands.insert("pred".to_string());
    final_syntax_operands.insert("succ".to_string());
}

/// `c.lui` has a special immediate restriction and prints its immediate through
/// custom runtime formatting, so the normal operands segment is cleared here.
pub(super) fn fix_clui_instruction(
    operands_part: &mut String,
    instruction: &mut Instruction,
    final_syntax_operands: &mut HashSet<String>,
) {
    if instruction.name != "c.lui" {
        return;
    }

    instruction
        .operands
        .iter_mut()
        .find(|op| op.name == "imm")
        .map(|op| {
            op.restrictions = Some(OperandRestriction {
                multiple_of: None,
                min_max: Some((-32, 31)),
                forbidden_values: vec![0],
                odd_only: false,
            });
        });

    *operands_part = String::new();
    final_syntax_operands.clear();
}

/// Some conversion instructions hard-code their rounding mode spelling and
/// therefore should not expose `rm`/`frm` as explicit operands.
pub(super) fn fix_round_mode_removal(
    operands_part: &mut String,
    instruction: &mut Instruction,
    final_syntax_operands: &mut HashSet<String>,
) {
    if !matches!(
        instruction.name.as_str(),
        "fcvt.d.w"
            | "fcvt.d.wu"
            | "fcvt.d.s"
            | "fcvt.q.w"
            | "fcvt.q.wu"
            | "fcvt.q.s"
            | "fcvt.q.d"
            | "fcvt.q.h"
            | "fcvt.s.bf16"
            | "fcvt.s.h"
            | "fcvt.d.h"
            | "fcvtmod.w.d"
    ) {
        return;
    }

    instruction
        .operands
        .retain(|op| op.name != "rm" && op.name != "frm");
    *operands_part = operands_part.replace(", rm", "").replace(", frm", "");
    final_syntax_operands.remove("rm");
    final_syntax_operands.remove("frm");
}

/// `fli.*` uses a constrained literal-constant operand, not a floating-point
/// source register even if unified-db names it like one.
pub(super) fn fix_fli_instruction_operands(
    operands_part: &mut String,
    instruction: &mut Instruction,
    final_syntax_operands: &mut HashSet<String>,
) {
    if !matches!(
        instruction.name.as_str(),
        "fli.d" | "fli.s" | "fli.h" | "fli.q"
    ) {
        return;
    }

    if let Some(operand) = instruction
        .operands
        .iter_mut()
        .find(|op| matches!(op.name.as_str(), "qs1" | "fs1" | "xs1" | "imm"))
    {
        let old_name = std::mem::replace(&mut operand.name, "uimm".to_string());
        operand.kind = OperandKind::FliConstant;

        if final_syntax_operands.remove(&old_name) {
            final_syntax_operands.insert("uimm".to_string());
            *operands_part = operands_part.replace(&old_name, "uimm");
        }
    }
}

/// Rename signed `imm` placeholders to `uimm` for instructions whose encoding
/// and downstream random generation both treat the field as unsigned.
pub(super) fn fix_uimm_instead_of_imm(
    operands_part: &mut String,
    instruction: &mut Instruction,
    final_syntax_operands: &mut HashSet<String>,
) {
    if !matches!(
        instruction.name.as_str(),
        "lui"
            | "auipc"
            | "csrrsi"
            | "csrrwi"
            | "csrrci"
            | "c.fswsp"
            | "c.sdsp"
            | "c.lwsp"
            | "c.ldsp"
            | "c.fsdsp"
            | "c.fldsp"
            | "c.swsp"
            | "c.flwsp"
            | "c.fld"
            | "c.fsd"
            | "c.ld"
            | "c.sd"
            | "c.sw"
            | "c.lw"
            | "c.flw"
            | "c.fsw"
            | "c.addi4spn"
            | "c.srai"
            | "c.srli"
            | "c.slli"
            | "c.lbu"
            | "c.lh"
            | "c.lhu"
            | "c.sb"
            | "c.sh"
            | "vslidedown.vi"
            | "vslideup.vi"
            | "vrgather.vi"
            | "vsll.vi"
            | "vsrl.vi"
            | "vsra.vi"
            | "vnclipu.wi"
            | "vnsra.wi"
            | "vnsrl.wi"
            | "vssra.wi"
            | "vssrl.wi"
            | "vssra.vi"
            | "vssrl.vi"
            | "vror.vi"
            | "vwsll.vi"
            | "vnclip.wi"
            | "lpad"
            | "vaeskf1.vi"
            | "vaeskf2.vi"
            | "vsm3c.vi"
            | "vsm4k.vi"
    ) {
        return;
    }

    for operand in instruction.operands.iter_mut() {
        if operand.name == "imm" {
            operand.name = "uimm".to_string();
        }
    }

    if final_syntax_operands.contains("imm") {
        final_syntax_operands.remove("imm");
        final_syntax_operands.insert("uimm".to_string());
        *operands_part = operands_part.replace("imm", "uimm");
    }
}

/// Normalize floating-point load/store addressing into the canonical
/// `reg, imm(base)` textual form.
pub(super) fn fix_floating_point_load_store_addressing(
    operands_part: &mut String,
    instruction: &mut Instruction,
    final_syntax_operands: &mut HashSet<String>,
) {
    if !matches!(instruction.name.as_str(), "flw" | "fsw" | "fld" | "flq") {
        return;
    }

    let (expected_first_reg, expected_base_reg, expected_offset_imm) =
        match instruction.name.as_str() {
            "flw" => ("fd", "xs1", "imm"),
            "fsw" => ("fs2", "xs1", "imm"),
            "fld" => ("fd", "xs1", "imm"),
            "flq" => ("fd", "xs1", "imm"),
            _ => return,
        };

    let current_operands_str = operands_part.trim();
    let parts: Vec<&str> = current_operands_str.split(',').map(|s| s.trim()).collect();

    if parts.len() == 3
        && parts[0] == expected_first_reg
        && parts[1] == expected_base_reg
        && parts[2] == expected_offset_imm
        && final_syntax_operands.contains(expected_first_reg)
        && final_syntax_operands.contains(expected_base_reg)
        && final_syntax_operands.contains(expected_offset_imm)
    {
        *operands_part = format!(
            "{}, {}({})",
            expected_first_reg, expected_offset_imm, expected_base_reg
        );
    }
}

/// Re-insert `rm` into syntax when the operand list still contains it but the
/// raw assembly text forgot to print it.
pub(super) fn fix_missing_round_mode(
    operands_part: &mut String,
    instruction: &mut Instruction,
    final_syntax_operands: &mut HashSet<String>,
) {
    if !instruction.operands.iter().any(|o| o.name == "rm") || final_syntax_operands.contains("rm")
    {
        return;
    }

    if operands_part.trim().is_empty() {
        *operands_part = "rm".to_string();
    } else {
        *operands_part = format!(
            "{}, rm",
            operands_part.trim_end_matches(|c: char| c.is_whitespace() || c == ',')
        );
    }

    final_syntax_operands.insert("rm".to_string());
}

/// Compressed stack-pointer instructions print `sp` as a literal base register.
pub(super) fn fix_compressed_stack_pointer_instructions(
    operands_part: &mut String,
    instruction: &mut Instruction,
    final_syntax_operands: &mut HashSet<String>,
) {
    if !matches!(
        instruction.name.as_str(),
        "c.flwsp" | "c.fswsp" | "c.ldsp" | "c.lwsp" | "c.sdsp" | "c.swsp" | "c.fsdsp" | "c.fldsp"
    ) {
        return;
    }

    let expected_pattern = match instruction.name.as_str() {
        "c.flwsp" => ("fd", "uimm"),
        "c.fswsp" => ("fs2", "uimm"),
        "c.ldsp" => ("xd", "uimm"),
        "c.lwsp" => ("xd", "uimm"),
        "c.sdsp" => ("xs2", "uimm"),
        "c.swsp" => ("xs2", "uimm"),
        "c.fsdsp" => ("fs2", "uimm"),
        "c.fldsp" => ("fd", "uimm"),
        _ => return,
    };

    let current_operands_str = operands_part.trim();
    let parts: Vec<&str> = current_operands_str.split(',').map(|s| s.trim()).collect();

    if parts.len() == 2
        && parts[0] == expected_pattern.0
        && parts[1] == expected_pattern.1
        && final_syntax_operands.contains(expected_pattern.0)
        && final_syntax_operands.contains(expected_pattern.1)
    {
        *operands_part = format!("{}, {}(sp)", expected_pattern.0, expected_pattern.1);
    }
}

/// Normalize compressed stack-adjustment mnemonics whose textual forms are
/// partially implicit in unified-db.
pub(super) fn fix_compressed_stack_adjustment_instructions(
    operands_part: &mut String,
    instruction: &mut Instruction,
) {
    if instruction.name == "c.addi4spn" {
        *operands_part = "xd, sp, uimm".to_string();
    } else if instruction.name == "c.addi16sp" {
        *operands_part = "sp, imm".to_string();
    }
}

/// `v0` is a hard-coded mask register in printed syntax, not a random operand.
pub(super) fn fix_vector_v0_hardcoded_operands(final_syntax_operands: &mut HashSet<String>) {
    final_syntax_operands.remove("v0");
}

/// Hypervisor load/store instructions print a fixed zero displacement.
pub(super) fn fix_hypervisor_load_store_instructions(
    operands_part: &mut String,
    instruction: &mut Instruction,
) {
    match instruction.name.as_str() {
        "hlv.b" | "hlv.bu" | "hlv.d" | "hlv.h" | "hlv.hu" | "hlv.w" | "hlv.wu" | "hlvx.hu"
        | "hlvx.wu" => {
            *operands_part = "xd, 0(xs1)".to_string();
        }
        "hsv.b" | "hsv.d" | "hsv.h" | "hsv.w" => {
            *operands_part = "xs2, 0(xs1)".to_string();
        }
        _ => {}
    }
}

/// Merge `reg_list` and `stack_adj` into the synthetic operand that downstream
/// codegen and random generation can reason about coherently.
pub(super) fn fix_cm_instructions_reg_list_stack_adj(
    instruction: &mut Instruction,
    operands_part: &mut String,
    final_syntax_operands: &mut HashSet<String>,
) {
    if !matches!(
        instruction.name.as_str(),
        "cm.pop" | "cm.push" | "cm.popretz" | "cm.popret"
    ) {
        return;
    }

    let has_reg_list = instruction.operands.iter().any(|op| op.name == "reg_list");
    let has_stack_adj = instruction.operands.iter().any(|op| op.name == "stack_adj");

    if has_reg_list && has_stack_adj {
        // Both fields are encoded together semantically even though unified-db
        // models them as separate raw operands.
        let reg_list_operand = instruction
            .operands
            .iter()
            .find(|op| op.name == "reg_list")
            .cloned();
        let stack_adj_operand = instruction
            .operands
            .iter()
            .find(|op| op.name == "stack_adj")
            .cloned();

        if let (Some(reg_list), Some(stack_adj)) = (reg_list_operand, stack_adj_operand) {
            let mut combined_bit_lengths = HashMap::new();
            for (&isa_base, &reg_bits) in &reg_list.bit_lengths {
                if let Some(&stack_bits) = stack_adj.bit_lengths.get(&isa_base) {
                    combined_bit_lengths.insert(isa_base, reg_bits + stack_bits);
                }
            }

            let combined_operand = Operand {
                name: "saved_reg_list_with_stack_adj".to_string(),
                kind: OperandKind::SavedRegListWithStackAdj {
                    xlen_dependent: true,
                },
                bit_lengths: combined_bit_lengths,
                restrictions: None,
                per_isa_restrictions: HashMap::new(),
            };

            instruction.operands = vec![combined_operand];
            final_syntax_operands.clear();
            final_syntax_operands.insert("saved_reg_list_with_stack_adj".to_string());
            *operands_part = String::new();
        }
    }
}

/// `cm.mvsa01` requires the saved-register pair to be modeled as a single
/// "not equal" operand so random generation can enforce the cross-field rule.
pub(super) fn fix_cm_mvsa01_instruction(
    instruction: &mut Instruction,
    operands_part: &mut String,
    final_syntax_operands: &mut HashSet<String>,
) {
    if instruction.name != "cm.mvsa01" {
        return;
    }

    let has_r1s = instruction.operands.iter().any(|op| op.name == "r1s");
    let has_r2s = instruction.operands.iter().any(|op| op.name == "r2s");

    if has_r1s && has_r2s {
        let mut combined_bit_lengths = HashMap::new();
        let r1s_op = instruction.operands.iter().find(|op| op.name == "r1s");
        let r2s_op = instruction.operands.iter().find(|op| op.name == "r2s");

        if let (Some(op1), Some(op2)) = (r1s_op, r2s_op) {
            for (base, len1) in &op1.bit_lengths {
                if let Some(len2) = op2.bit_lengths.get(base) {
                    combined_bit_lengths.insert(*base, len1 + len2);
                }
            }
        } else {
            combined_bit_lengths.insert(ISABase::RV32, 6);
            combined_bit_lengths.insert(ISABase::RV64, 6);
        }

        let merged_operand = Operand {
            name: "ne_r1s_r2s".to_string(),
            kind: OperandKind::RegisterPair(RegisterPairSpec {
                bank: RegisterBank::Integer,
                distinct: true,
            }),
            bit_lengths: combined_bit_lengths,
            restrictions: None,
            per_isa_restrictions: HashMap::new(),
        };

        instruction
            .operands
            .retain(|op| op.name != "r1s" && op.name != "r2s");
        instruction.operands.push(merged_operand);

        if operands_part.contains("r1s") && operands_part.contains("r2s") {
            *operands_part = operands_part
                .replace("r1s, r2s", "ne_r1s_r2s")
                .replace("r1s,r2s", "ne_r1s_r2s");
        } else {
            *operands_part = "ne_r1s_r2s".to_string();
        }

        final_syntax_operands.remove("r1s");
        final_syntax_operands.remove("r2s");
        final_syntax_operands.insert("ne_r1s_r2s".to_string());
    }
}
