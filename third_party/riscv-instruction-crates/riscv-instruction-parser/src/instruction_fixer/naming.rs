//! Operand naming reconciliation helpers.
//!
//! unified-db often lets the canonical operand list and the assembly
//! placeholders drift slightly in naming, for example `rd` vs `xd` or `rm`
//! vs `frm`. This module performs a conservative first-pass alignment so later
//! rules do not have to keep guessing between two naming schemes.

use riscv_instruction_schema::*;
use std::collections::{HashMap, HashSet};

/// Reconcile the YAML-derived operand list with the operand placeholders that
/// actually appear in the assembly syntax.
///
/// This stage is intentionally conservative: it only renames an operand when a
/// placeholder match is available, so later rules start from a synchronized
/// baseline instead of inventing new structure too early.
pub(crate) fn reconcile_operand_names(
    instruction: &mut Instruction,
    syntax_operands: &HashSet<String>,
) {
    if syntax_operands.is_empty() && instruction.operands.is_empty() {
        return;
    }

    let name_mapping = create_operand_name_mapping();
    let mut available_syntax_ops: Vec<String> = syntax_operands.iter().cloned().collect();
    let mut updated_operands = Vec::with_capacity(instruction.operands.len());
    let original_operands = instruction.operands.clone();

    for operand in &original_operands {
        let new_name =
            find_matching_operand_name(&operand.name, &mut available_syntax_ops, &name_mapping);

        if new_name != operand.name {
            rename_instruction_operand_references(instruction, &operand.name, &new_name);
        }

        updated_operands.push(Operand {
            name: new_name,
            kind: operand.kind.clone(),
            bit_lengths: operand.bit_lengths.clone(),
            restrictions: operand.restrictions.clone(),
            per_isa_restrictions: operand.per_isa_restrictions.clone(),
        });
    }

    instruction.operands = updated_operands;
}

fn create_operand_name_mapping() -> HashMap<&'static str, Vec<&'static str>> {
    [
        ("rd", vec!["xd", "fd", "qd"]),
        ("rs1", vec!["xs1", "fs1", "qs1"]),
        ("rs2", vec!["xs2", "fs2", "qs2"]),
        ("rs3", vec!["xs3", "fs3"]),
        ("fd", vec!["xd"]),
        ("fs1", vec!["xs1"]),
        ("fs2", vec!["xs2"]),
        ("fs3", vec!["xs3"]),
        ("rm", vec!["frm"]),
        ("rlist", vec!["reg_list"]),
        ("spimm", vec!["stack_adj"]),
        ("imm", vec!["offset", "imm12", "uimm"]),
        ("uimm", vec!["imm", "nzuimm"]),
        ("nzuimm", vec!["uimm", "imm"]),
        ("zimm5", vec!["imm"]),
    ]
    .iter()
    .map(|(k, v)| (*k, v.to_vec()))
    .collect()
}

/// Find the best syntax placeholder for one operand and consume it once used.
///
/// The destructive removal is important: it prevents multiple operands from
/// accidentally claiming the same placeholder alias.
fn find_matching_operand_name(
    current_name: &str,
    available_syntax_ops: &mut Vec<String>,
    name_mapping: &HashMap<&str, Vec<&str>>,
) -> String {
    if let Some(pos) = available_syntax_ops.iter().position(|x| x == current_name) {
        return available_syntax_ops.remove(pos);
    }

    if let Some(possible_names) = name_mapping.get(current_name) {
        for &mapped_name in possible_names {
            if let Some(pos) = available_syntax_ops.iter().position(|x| x == mapped_name) {
                return available_syntax_ops.remove(pos);
            }
        }
    }

    current_name.to_string()
}

/// Rename an operand in the canonical operand list.
pub(crate) fn fix_operand_name(instruction: &mut Instruction, old_name: &str, new_name: &str) {
    for operand in instruction.operands.iter_mut() {
        if operand.name == old_name {
            operand.name = new_name.to_string();
        }
    }

    rename_instruction_operand_references(instruction, old_name, new_name);
}

/// Rename the same operand in the assembly placeholder set/string.
pub(crate) fn update_syntax_operands(
    operands_part: &mut String,
    final_syntax_operands: &mut HashSet<String>,
    old_name: &str,
    new_name: &str,
) {
    if final_syntax_operands.contains(old_name) {
        final_syntax_operands.remove(old_name);
        final_syntax_operands.insert(new_name.to_string());
        *operands_part = operands_part.replace(old_name, new_name);
    }
}

fn rename_instruction_operand_references(
    instruction: &mut Instruction,
    old_name: &str,
    new_name: &str,
) {
    if let Some(csr_semantics) = instruction.semantics.csr.as_mut() {
        if csr_semantics.address_operand == old_name {
            csr_semantics.address_operand = new_name.to_string();
        }
        if csr_semantics.source_register_operand.as_deref() == Some(old_name) {
            csr_semantics.source_register_operand = Some(new_name.to_string());
        }
    }

    if let Some(control_flow_semantics) = instruction.semantics.control_flow.as_mut() {
        if control_flow_semantics.target_register_operand.as_deref() == Some(old_name) {
            control_flow_semantics.target_register_operand = Some(new_name.to_string());
        }
    }
}
