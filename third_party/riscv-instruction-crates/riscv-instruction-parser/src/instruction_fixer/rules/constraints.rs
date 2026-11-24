//! Constraint and metadata repair rules.
//!
//! These rules do not mainly change syntax spelling. Instead they repair
//! operand restrictions, ISA bases, missing operands, and final operand types.

use crate::operand_inference::infer_operand_kind;
use riscv_instruction_schema::*;
use std::collections::HashMap;

/// `c.mop.n` requires odd-only `n`, which is easier to express as a restriction
/// than by baking the rule into every downstream consumer.
pub(super) fn fix_c_mop_n_instruction(instruction: &mut Instruction) {
    if instruction.name == "c.mop.n" {
        instruction
            .operands
            .iter_mut()
            .find(|op| op.name == "n")
            .map(|op| {
                op.restrictions = Some(OperandRestriction {
                    multiple_of: None,
                    min_max: None,
                    odd_only: true,
                    forbidden_values: Vec::with_capacity(0),
                });
            });
    }
}

/// Some upstream entries for `add.uw` arrive without any operand list at all.
pub(super) fn fix_add_uw_missing_operands(instruction: &mut Instruction) {
    if instruction.name == "add.uw" && instruction.operands.is_empty() {
        instruction.operands.push(Operand {
            name: "xd".to_string(),
            kind: OperandKind::Register(RegisterSpec {
                bank: RegisterBank::Integer,
            }),
            bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
            restrictions: None,
            per_isa_restrictions: HashMap::new(),
        });

        instruction.operands.push(Operand {
            name: "xs1".to_string(),
            kind: OperandKind::Register(RegisterSpec {
                bank: RegisterBank::Integer,
            }),
            bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
            restrictions: None,
            per_isa_restrictions: HashMap::new(),
        });

        instruction.operands.push(Operand {
            name: "xs2".to_string(),
            kind: OperandKind::Register(RegisterSpec {
                bank: RegisterBank::Integer,
            }),
            bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
            restrictions: None,
            per_isa_restrictions: HashMap::new(),
        });
    }
}

/// Fill in missing operand kinds after all name repairs have stabilized.
pub(crate) fn assign_operand_kinds(instruction: &mut Instruction) {
    for operand in instruction.operands.iter_mut() {
        if matches!(operand.kind, OperandKind::Unknown) {
            operand.kind = infer_operand_kind(&operand.name);
        }
    }
}

/// Repair known ISA-base mistakes that would otherwise leak into bit widths and
/// downstream codegen.
pub(super) fn fix_isabase_error_instructions(instruction: &mut Instruction) {
    if instruction.name == "ssamoswap.d" {
        instruction.isa_bases = vec![ISABase::RV64];
        instruction.operands.iter_mut().for_each(|op| {
            op.bit_lengths.remove(&ISABase::RV32);
        });
    }

    if instruction.name == "ssamoswap.w" {
        instruction.isa_bases = vec![ISABase::RV32];
        instruction.operands.iter_mut().for_each(|op| {
            op.bit_lengths.remove(&ISABase::RV64);
        });
    }
}

/// Prefetch instructions require 32-byte alignment on their immediate offset.
pub(super) fn fix_prefetch_offset_alignment(instruction: &mut Instruction) {
    if !matches!(
        instruction.name.as_str(),
        "prefetch.i" | "prefetch.r" | "prefetch.w"
    ) {
        return;
    }

    if let Some(imm_operand) = instruction.operands.iter_mut().find(|op| op.name == "imm") {
        let restriction = imm_operand
            .restrictions
            .get_or_insert_with(OperandRestriction::default);
        restriction.multiple_of = Some(32);
    }
}

/// `c.ldsp`/`c.sdsp` xd/xs2 register pair alignment is RV32-only (Zilsd pair requirement).
/// On RV64, any register is valid. Store the constraint in `per_isa_restrictions[RV32]` so the
/// code-gen generates the correct field type per ISA base.
pub(super) fn fix_zclsd_register_restrictions(instruction: &mut Instruction) {
    let operand_name = match instruction.name.as_str() {
        "c.ldsp" => "xd",
        "c.sdsp" => "xs2",
        _ => return,
    };

    if let Some(operand) = instruction
        .operands
        .iter_mut()
        .find(|op| op.name == operand_name)
    {
        // Clear any shared restrictions (YAML RV32-only artifacts)
        operand.restrictions = None;
        operand.per_isa_restrictions.clear();
        // Set RV32-only even-register constraint
        operand.per_isa_restrictions.insert(
            ISABase::RV32,
            Some(OperandRestriction {
                multiple_of: Some(2),
                min_max: Some((0, 30)),
                forbidden_values: Vec::new(),
                odd_only: false,
            }),
        );
    }
}

/// `amocas.d*` and `amocas.q*` xd/xs2 must be even-aligned — GNU as enforces this because:
/// - `amocas.d*` on RV32 is a 64-bit pair (two 32-bit registers, must be consecutive even+odd)
/// - `amocas.q*` on RV64 is a 128-bit pair (two 64-bit registers, must be consecutive even+odd)
/// (`amocas.d*` on RV64 does NOT need alignment, but we use a shared type, so we enforce it.)
pub(super) fn fix_amocas_q_pair_register_constraints(instruction: &mut Instruction) {
    if !instruction.name.starts_with("amocas.q")
        && !instruction.name.starts_with("amocas.d")
    {
        return;
    }
    for operand_name in ["xd", "xs2"] {
        if let Some(operand) = instruction.operands.iter_mut().find(|op| op.name == operand_name) {
            let restriction = operand.restrictions.get_or_insert_with(OperandRestriction::default);
            restriction.multiple_of = Some(2);
            restriction.min_max = Some((0, 30));
        }
    }
}

/// `aes64ks1i` accepts only round numbers 0..=10.
pub(super) fn fix_aes64ks1i_rnum_range(instruction: &mut Instruction) {
    if instruction.name != "aes64ks1i" {
        return;
    }

    instruction
        .operands
        .iter_mut()
        .find(|op| op.name == "rnum")
        .map(|op| {
            op.restrictions = Some(OperandRestriction {
                multiple_of: None,
                min_max: Some((0, 10)),
                forbidden_values: vec![],
                odd_only: false,
            });
        });
}
