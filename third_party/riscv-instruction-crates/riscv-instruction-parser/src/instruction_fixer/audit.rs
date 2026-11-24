//! Structured audit output for the fixer pipeline.
//!
//! The fixer is allowed to be aggressive internally, but external debugging
//! becomes tractable only if we can answer "which rule changed what".
//! This module stores exactly that information in a serializable form.

use riscv_instruction_schema::*;
use serde::Serialize;
use std::collections::HashMap;

/// Top-level audit report returned by `fix_instructions_with_audit`.
#[derive(Debug, Clone, Serialize)]
pub struct FixerAuditReport {
    pub total_instructions: usize,
    pub changed_instructions: usize,
    pub unchanged_instructions: usize,
    pub rule_stats: Vec<FixerRuleStat>,
    pub instruction_audits: Vec<InstructionFixAudit>,
}

/// Aggregate statistics for one fixer step across all instructions.
#[derive(Debug, Clone, Serialize)]
pub struct FixerRuleStat {
    pub step: String,
    pub changed_instructions: usize,
    pub instruction_names: Vec<String>,
}

/// Full before/after audit for one instruction.
#[derive(Debug, Clone, Serialize)]
pub struct InstructionFixAudit {
    pub instruction_name: String,
    pub changed: bool,
    pub initial_instruction: Instruction,
    pub initial_operands_part: String,
    pub initial_syntax_operands: Vec<String>,
    pub steps: Vec<FixerStepAudit>,
    pub final_instruction: Instruction,
}

/// One recorded step inside a single instruction audit.
#[derive(Debug, Clone, Serialize)]
pub struct FixerStepAudit {
    pub step: String,
    pub changes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_operands_part: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_operands_part: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_syntax_operands: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_syntax_operands: Option<Vec<String>>,
}

/// Record one fixer step if it actually changed any observable state.
///
/// This keeps the report readable: steps that ran but produced no semantic
/// delta are intentionally omitted.
pub(crate) fn record_step(
    audit: &mut Option<InstructionFixAudit>,
    step: &str,
    before_instruction: &Instruction,
    after_instruction: &Instruction,
    before_operands_part: &str,
    after_operands_part: &str,
    before_syntax_operands: Vec<String>,
    after_syntax_operands: Vec<String>,
) {
    let Some(audit) = audit.as_mut() else {
        return;
    };

    let mut changes = Vec::new();

    if before_instruction.extension != after_instruction.extension {
        changes.push(format!(
            "extension: {} -> {}",
            before_instruction.extension, after_instruction.extension
        ));
    }
    if before_instruction.isa_bases != after_instruction.isa_bases {
        changes.push(format!(
            "isa_bases: {} -> {}",
            summarize_isa_bases(&before_instruction.isa_bases),
            summarize_isa_bases(&after_instruction.isa_bases)
        ));
    }
    if before_instruction.operands != after_instruction.operands {
        changes.push(format!(
            "operands: {} -> {}",
            summarize_operands(&before_instruction.operands),
            summarize_operands(&after_instruction.operands)
        ));
    }
    if before_instruction.assembly_syntax != after_instruction.assembly_syntax {
        changes.push(format!(
            "assembly_syntax: {} -> {}",
            summarize_assembly_syntax(&before_instruction.assembly_syntax),
            summarize_assembly_syntax(&after_instruction.assembly_syntax)
        ));
    }
    if before_instruction.memory_access != after_instruction.memory_access {
        changes.push(format!(
            "memory_access: {} -> {}",
            summarize_memory_access(&before_instruction.memory_access),
            summarize_memory_access(&after_instruction.memory_access)
        ));
    }

    let operands_part_changed = before_operands_part != after_operands_part;
    if operands_part_changed {
        changes.push(format!(
            "operands_part: {:?} -> {:?}",
            before_operands_part, after_operands_part
        ));
    }

    let syntax_operands_changed = before_syntax_operands != after_syntax_operands;
    if syntax_operands_changed {
        changes.push(format!(
            "syntax_operands: {} -> {}",
            summarize_string_vec(&before_syntax_operands),
            summarize_string_vec(&after_syntax_operands)
        ));
    }

    if changes.is_empty() {
        return;
    }

    audit.steps.push(FixerStepAudit {
        step: step.to_string(),
        changes,
        before_operands_part: operands_part_changed.then(|| before_operands_part.to_string()),
        after_operands_part: operands_part_changed.then(|| after_operands_part.to_string()),
        before_syntax_operands: syntax_operands_changed.then_some(before_syntax_operands),
        after_syntax_operands: syntax_operands_changed.then_some(after_syntax_operands),
    });
}

/// Convert the unordered placeholder set into a stable sorted vector.
pub(crate) fn sorted_syntax_operands(
    syntax_operands: &std::collections::HashSet<String>,
) -> Vec<String> {
    let mut operands = syntax_operands.iter().cloned().collect::<Vec<_>>();
    operands.sort();
    operands
}

fn summarize_isa_bases(isa_bases: &[ISABase]) -> String {
    isa_bases
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

fn summarize_operands(operands: &[Operand]) -> String {
    let parts = operands.iter().map(summarize_operand).collect::<Vec<_>>();
    format!("[{}]", parts.join("; "))
}

fn summarize_operand(operand: &Operand) -> String {
    let operand_kind = operand.kind.to_string();
    let bit_lengths = summarize_bit_lengths(&operand.bit_lengths);
    let restrictions = summarize_restrictions(&operand.restrictions);
    format!(
        "{}:{} bits{{{}}}{}",
        operand.name, operand_kind, bit_lengths, restrictions
    )
}

fn summarize_bit_lengths(bit_lengths: &HashMap<ISABase, u8>) -> String {
    let mut ordered = bit_lengths
        .iter()
        .map(|(isa, bits)| (isa.to_string(), *bits))
        .collect::<Vec<_>>();
    ordered.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    ordered
        .into_iter()
        .map(|(isa, bits)| format!("{}={}", isa, bits))
        .collect::<Vec<_>>()
        .join(",")
}

fn summarize_restrictions(restrictions: &Option<OperandRestriction>) -> String {
    let Some(restrictions) = restrictions else {
        return String::new();
    };

    let mut parts = Vec::new();
    if let Some(multiple_of) = restrictions.multiple_of {
        parts.push(format!("multiple_of={}", multiple_of));
    }
    if let Some((min, max)) = restrictions.min_max {
        parts.push(format!("range={}..{}", min, max));
    }
    if !restrictions.forbidden_values.is_empty() {
        parts.push(format!(
            "forbidden={}",
            summarize_i64_vec(&restrictions.forbidden_values)
        ));
    }
    if restrictions.odd_only {
        parts.push("odd_only=true".to_string());
    }

    if parts.is_empty() {
        String::new()
    } else {
        format!(" restr{{{}}}", parts.join(","))
    }
}

fn summarize_memory_access(memory_access: &Option<MemoryAddressInfo>) -> String {
    let Some(memory_access) = memory_access else {
        return "None".to_string();
    };

    format!(
        "base_operand={:?}, base_fixed={:?}, offset={:?}",
        memory_access.base_register_operand,
        memory_access.base_register_fixed,
        memory_access.offset_operand
    )
}

fn summarize_assembly_syntax(assembly_syntax: &AssemblySyntax) -> String {
    match assembly_syntax {
        AssemblySyntax::Format(format) => format!("Format({:?})", format),
        AssemblySyntax::RustCode(code) => format!("RustCode({:?})", code),
    }
}

fn summarize_string_vec(values: &[String]) -> String {
    format!("[{}]", values.join(", "))
}

fn summarize_i64_vec(values: &[i64]) -> String {
    let mut ordered = values.to_vec();
    ordered.sort();
    format!(
        "[{}]",
        ordered
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}
