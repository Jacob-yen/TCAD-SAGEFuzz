//! Instruction fixer pipeline.
//!
//! The parser intentionally keeps the YAML-to-IR step close to unified-db.
//! This module is the normalization layer that repairs known upstream
//! inconsistencies before the data is exported to JSON and code-generated
//! into the final crates.
//!
//! High-level flow for each instruction:
//! 1. Read the raw operand placeholders from the assembly syntax.
//! 2. Reconcile `instruction.operands` names with the syntax placeholders.
//! 3. Apply a deterministic sequence of repair rules.
//! 4. Rebuild the final `AssemblySyntax`.
//! 5. Fill any still-missing operand kinds.

use crate::operand_extractor::extract_operands_from_asm_without_name;
use riscv_instruction_schema::*;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

mod assembly;
mod audit;
mod naming;
mod report;
mod rules;

pub use audit::{FixerAuditReport, FixerRuleStat, FixerStepAudit, InstructionFixAudit};
pub use report::GeneratedFixerReportPaths;

use assembly::generate_assembly_syntax;
use naming::reconcile_operand_names;
use report::write_timestamped_audit_reports;
use rules::{apply_special_rules, assign_operand_kinds};

pub(crate) use audit::{record_step, sorted_syntax_operands};

/// Capture the before/after state of one rule execution.
///
/// The macro intentionally records only observable changes, so "no-op" rules
/// do not pollute the audit trail.
macro_rules! audited_step {
    (
        $audit:expr,
        $instruction:expr,
        $operands_part:expr,
        $syntax_operands:expr,
        $step_name:expr,
        $body:block
    ) => {{
        let before_instruction = $instruction.clone();
        let before_operands_part = $operands_part.clone();
        let before_syntax_operands =
            $crate::instruction_fixer::sorted_syntax_operands(&$syntax_operands);
        $body
        let after_syntax_operands =
            $crate::instruction_fixer::sorted_syntax_operands(&$syntax_operands);
        $crate::instruction_fixer::record_step(
            $audit,
            $step_name,
            &before_instruction,
            $instruction,
            &before_operands_part,
            &$operands_part,
            before_syntax_operands,
            after_syntax_operands,
        );
    }};
}

pub(crate) use audited_step;

/// Apply the fixer pipeline without recording a per-step audit trail.
pub fn fix_instructions(instructions: &mut [Instruction]) {
    for instruction in instructions.iter_mut() {
        fix_instruction_internal(instruction, false);
    }
}

/// Apply the fixer pipeline and collect a structured report of every change.
pub fn fix_instructions_with_audit(instructions: &mut [Instruction]) -> FixerAuditReport {
    let mut instruction_audits = Vec::with_capacity(instructions.len());
    let mut rule_to_instructions: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for instruction in instructions.iter_mut() {
        if let Some(audit) = fix_instruction_internal(instruction, true) {
            for step in &audit.steps {
                rule_to_instructions
                    .entry(step.step.clone())
                    .or_default()
                    .insert(audit.instruction_name.clone());
            }
            instruction_audits.push(audit);
        }
    }

    let changed_instructions = instruction_audits
        .iter()
        .filter(|audit| audit.changed)
        .count();
    let total_instructions = instruction_audits.len();
    let unchanged_instructions = total_instructions.saturating_sub(changed_instructions);
    let rule_stats = rule_to_instructions
        .into_iter()
        .map(|(step, instruction_names)| FixerRuleStat {
            step,
            changed_instructions: instruction_names.len(),
            instruction_names: instruction_names.into_iter().collect(),
        })
        .collect();

    FixerAuditReport {
        total_instructions,
        changed_instructions,
        unchanged_instructions,
        rule_stats,
        instruction_audits,
    }
}

/// Apply the fixer pipeline and immediately write timestamped audit artifacts.
pub fn fix_instructions_and_write_audit_reports<P: AsRef<Path>, Q: AsRef<Path>>(
    instructions: &mut [Instruction],
    input_path: P,
    output_root: Q,
) -> Result<(FixerAuditReport, GeneratedFixerReportPaths), Box<dyn std::error::Error>> {
    let report = fix_instructions_with_audit(instructions);
    let report_paths =
        write_timestamped_audit_reports(&report, input_path.as_ref(), output_root.as_ref())?;
    Ok((report, report_paths))
}

/// Run the full repair pipeline for a single instruction.
fn fix_instruction_internal(
    instruction: &mut Instruction,
    capture_audit: bool,
) -> Option<InstructionFixAudit> {
    // Before this stage every instruction should still come directly from
    // unified-db conversion, so `Format` is the only expected syntax form.
    // By this stage each instruction should still be in the direct unified-db
    // conversion form, so only `Format` is valid here. Seeing `RustCode`
    // means an unexpected second repair pass or premature code generation.
    if matches!(instruction.assembly_syntax, AssemblySyntax::RustCode(_)) {
        panic!(
            "Unexpected RustCode syntax for instruction: {}",
            instruction.name
        );
    }

    let mut assembly_syntax = match &instruction.assembly_syntax {
        AssemblySyntax::Format(s) => s.clone(),
        AssemblySyntax::RustCode(_) => unreachable!(),
    };

    let mut syntax_operands = extract_operands_from_asm_without_name(&assembly_syntax);
    let initial_instruction = instruction.clone();
    let mut audit = capture_audit.then(|| InstructionFixAudit {
        instruction_name: instruction.name.clone(),
        changed: false,
        initial_instruction: initial_instruction.clone(),
        initial_operands_part: assembly_syntax.clone(),
        initial_syntax_operands: sorted_syntax_operands(&syntax_operands),
        steps: Vec::new(),
        final_instruction: initial_instruction,
    });

    // Step 1: align operand names with the placeholders already present in the
    // assembly syntax. Later rules assume these names are in sync.
    // Step 1 aligns operand names first. Most later rules assume operand names
    // and syntax placeholders already refer to the same semantic objects.
    audited_step!(
        &mut audit,
        instruction,
        assembly_syntax,
        syntax_operands,
        "reconcile_operand_names",
        {
            reconcile_operand_names(instruction, &syntax_operands);
        }
    );

    // Step 2: run ordered special-case rules. The order is intentional and
    // documented in `rules::apply_special_rules`.
    // Step 2 runs the ordered special-case rules. This is an intentionally
    // stateful pipeline, not an arbitrary set of independent rewrites.
    apply_special_rules(
        &mut assembly_syntax,
        instruction,
        &mut syntax_operands,
        &mut audit,
    );

    // Step 3: after all mutations to operands/syntax placeholders are done,
    // rebuild the final exported assembly syntax.
    // Step 3 rebuilds the exported assembly syntax only after earlier rules
    // finish mutating operands, placeholder sets, and special-formatting cases.
    audited_step!(
        &mut audit,
        instruction,
        assembly_syntax,
        syntax_operands,
        "generate_assembly_syntax",
        {
            instruction.assembly_syntax = generate_assembly_syntax(
                &instruction.name,
                &assembly_syntax,
                &syntax_operands,
                instruction,
            );
        }
    );

    // Step 4: some rules only repair names/ranges. Finish by assigning any
    // still-missing operand kinds for codegen and random generation.
    // Step 4 closes the pipeline by filling in any remaining operand kinds.
    // Some earlier rules fix names, ranges, or ISA bases without assigning types.
    audited_step!(
        &mut audit,
        instruction,
        assembly_syntax,
        syntax_operands,
        "assign_operand_kinds",
        {
            assign_operand_kinds(instruction);
        }
    );

    if let Some(audit) = audit.as_mut() {
        audit.changed = !audit.steps.is_empty();
        audit.final_instruction = instruction.clone();
    }

    audit
}
