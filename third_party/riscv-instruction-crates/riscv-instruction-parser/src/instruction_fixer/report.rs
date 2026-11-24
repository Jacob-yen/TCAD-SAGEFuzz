//! Fixer audit report writer.
//!
//! The fixer itself should stay focused on repairing instruction IR.
//! This module turns the structured audit data into timestamped artifacts under
//! one stable output directory so every normal parse run leaves a trace.

use super::audit::{FixerAuditReport, FixerStepAudit, InstructionFixAudit};
use chrono::{DateTime, Local};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const FIXER_REPORT_DIR_NAME: &str = "fixer_audit";

/// Paths of the timestamped fixer audit artifacts generated for one parse run.
#[derive(Debug, Clone)]
pub struct GeneratedFixerReportPaths {
    pub output_dir: PathBuf,
    pub json_path: PathBuf,
    pub repair_markdown_path: PathBuf,
}

struct RepairSummary {
    repair_touched_instructions: usize,
    special_fix_touched_instructions: usize,
    reconcile_only_instructions: usize,
    derivation_only_instructions: usize,
    rule_summaries: Vec<RepairRuleSummary>,
}

struct RepairRuleSummary {
    step: String,
    description: &'static str,
    changed_instructions: usize,
    changed_fields: Vec<String>,
    instruction_names: Vec<String>,
    examples: Vec<RepairExample>,
}

struct RepairExample {
    instruction_name: String,
    changes: Vec<String>,
    before_operands_part: Option<String>,
    after_operands_part: Option<String>,
    before_syntax_operands: Option<Vec<String>>,
    after_syntax_operands: Option<Vec<String>>,
}

/// Write timestamped fixer audit artifacts under `<output_root>/fixer_audit/`.
///
/// The directory name stays stable so CI and humans know where to look, while
/// the file names carry a timestamp suffix to preserve run history.
pub fn write_timestamped_audit_reports(
    report: &FixerAuditReport,
    input_path: &Path,
    output_root: &Path,
) -> Result<GeneratedFixerReportPaths, Box<dyn std::error::Error>> {
    let generated_at = Local::now();
    let timestamp_suffix = generated_at.format("%Y%m%d_%H%M%S_%3f").to_string();
    let output_dir = output_root.join(FIXER_REPORT_DIR_NAME);
    let json_path = output_dir.join(format!("fixer_audit_report_{timestamp_suffix}.json"));
    let repair_markdown_path =
        output_dir.join(format!("fixer_repair_report_{timestamp_suffix}.md"));

    fs::create_dir_all(&output_dir)?;

    let repair_summary = build_repair_summary(report);

    fs::write(&json_path, serde_json::to_string_pretty(report)?)?;
    fs::write(
        &repair_markdown_path,
        build_repair_markdown_report(
            report,
            &repair_summary,
            input_path,
            &json_path,
            &generated_at,
        ),
    )?;

    Ok(GeneratedFixerReportPaths {
        output_dir,
        json_path,
        repair_markdown_path,
    })
}

fn build_repair_markdown_report(
    report: &FixerAuditReport,
    repair_summary: &RepairSummary,
    input_path: &Path,
    json_path: &Path,
    generated_at: &DateTime<Local>,
) -> String {
    let mut out = String::new();
    out.push_str("# Fixer 修复报告\n\n");
    out.push_str(&format!(
        "- 生成时间: `{}`\n",
        generated_at.format("%Y-%m-%d %H:%M:%S")
    ));
    out.push_str(&format!("- 输入路径: `{}`\n", input_path.display()));
    out.push_str(&format!("- 完整 JSON 明细: `{}`\n", json_path.display()));
    out.push_str("\n");

    out.push_str("## 报告说明\n\n");
    out.push_str("- `generate_assembly_syntax` 和 `assign_operand_kinds` 视为“派生步骤”。它们负责补出最终导出的汇编语法和操作数 kind，但单独出现并不意味着 unified-db 上游数据本身有错误。\n");
    out.push_str("- `reconcile_operand_names` 以及所有 `fix_*` 规则都视为“修复步骤”。这部分才是在导出前真正纠正、规整问题指令定义的逻辑。\n\n");

    out.push_str("## 修复概览\n\n");
    out.push_str(&format!(
        "- 解析到的指令总数: `{}`\n",
        report.total_instructions
    ));
    out.push_str(&format!(
        "- 被任意 fixer 步骤触及的指令数: `{}`\n",
        report.changed_instructions
    ));
    out.push_str(&format!(
        "- 被修复步骤触及的指令数: `{}`\n",
        repair_summary.repair_touched_instructions
    ));
    out.push_str(&format!(
        "- 被至少一条硬编码 `fix_*` 规则触及的指令数: `{}`\n",
        repair_summary.special_fix_touched_instructions
    ));
    out.push_str(&format!(
        "- 仅被 `reconcile_operand_names` 触及的指令数: `{}`\n",
        repair_summary.reconcile_only_instructions
    ));
    out.push_str(&format!(
        "- 仅被派生步骤触及的指令数: `{}`\n",
        repair_summary.derivation_only_instructions
    ));
    out.push('\n');

    out.push_str("## 修复规则汇总\n\n");
    out.push_str("| 规则 | 含义 | 变更指令数 | 变更字段 |\n");
    out.push_str("|------|------|------------|----------|\n");
    for rule in &repair_summary.rule_summaries {
        out.push_str(&format!(
            "| `{}` | {} | {} | {} |\n",
            rule.step,
            escape_table_cell(rule.description),
            rule.changed_instructions,
            escape_table_cell(&rule.changed_fields.join(", "))
        ));
    }
    out.push('\n');

    out.push_str("## 修复规则详情\n\n");
    for rule in &repair_summary.rule_summaries {
        out.push_str(&format!("### `{}`\n\n", rule.step));
        out.push_str(&format!("- 修复意图: {}\n", rule.description));
        out.push_str(&format!(
            "- 受影响指令数: `{}`\n",
            rule.changed_instructions
        ));
        out.push_str(&format!(
            "- 变更字段: {}\n\n",
            format_inline_code_list(&rule.changed_fields)
        ));
        out.push_str("受影响的指令名称:\n\n");
        out.push_str(&format!(
            "{}\n",
            format_instruction_name_blocks(&rule.instruction_names)
        ));

        if !rule.examples.is_empty() {
            out.push_str("\n代表性变化示例:\n\n");
            for example in &rule.examples {
                append_repair_example(&mut out, example);
            }
        }
    }

    out.push_str("## 已修复指令详情\n\n");
    let mut repaired_instruction_audits = report
        .instruction_audits
        .iter()
        .filter(|audit| audit.steps.iter().any(|step| is_repair_step(&step.step)))
        .collect::<Vec<_>>();
    repaired_instruction_audits.sort_by(|left, right| {
        left.instruction_name
            .cmp(&right.instruction_name)
            .then(left.steps.len().cmp(&right.steps.len()))
    });

    for audit in repaired_instruction_audits {
        append_repair_instruction_audit(&mut out, audit);
    }

    out
}

fn append_repair_instruction_audit(out: &mut String, audit: &InstructionFixAudit) {
    let repair_steps = audit
        .steps
        .iter()
        .filter(|step| is_repair_step(&step.step))
        .collect::<Vec<_>>();
    if repair_steps.is_empty() {
        return;
    }

    out.push_str(&format!("### `{}`\n\n", audit.instruction_name));
    out.push_str(&format!(
        "- 初始汇编语法: `{}`\n",
        escape_inline_code(&format!("{:?}", audit.initial_instruction.assembly_syntax))
    ));
    out.push_str(&format!(
        "- 初始语法操作数: `{}`\n",
        escape_inline_code(&audit.initial_syntax_operands.join(", "))
    ));
    out.push_str(&format!("- 修复步骤数: `{}`\n\n", repair_steps.len()));

    for step in repair_steps {
        append_step_audit(out, step);
    }
}

fn append_step_audit(out: &mut String, step: &FixerStepAudit) {
    out.push_str(&format!("#### `{}`\n\n", step.step));
    for change in &step.changes {
        out.push_str(&format!("- {}\n", change));
    }
    if let (Some(before), Some(after)) = (&step.before_operands_part, &step.after_operands_part) {
        out.push_str("\n");
        out.push_str("修复前的 operands_part:\n");
        out.push_str("```text\n");
        out.push_str(before);
        out.push_str("\n```\n\n");
        out.push_str("修复后的 operands_part:\n");
        out.push_str("```text\n");
        out.push_str(after);
        out.push_str("\n```\n\n");
    }
    if let (Some(before), Some(after)) = (&step.before_syntax_operands, &step.after_syntax_operands)
    {
        out.push_str(&format!(
            "修复前的语法操作数: `{}`\n",
            escape_inline_code(&before.join(", "))
        ));
        out.push_str(&format!(
            "修复后的语法操作数: `{}`\n\n",
            escape_inline_code(&after.join(", "))
        ));
    } else {
        out.push('\n');
    }
}

fn append_repair_example(out: &mut String, example: &RepairExample) {
    out.push_str(&format!("#### `{}`\n\n", example.instruction_name));
    for change in &example.changes {
        out.push_str(&format!("- {}\n", change));
    }
    if let (Some(before), Some(after)) =
        (&example.before_operands_part, &example.after_operands_part)
    {
        out.push_str("\n");
        out.push_str("修复前的 operands_part:\n");
        out.push_str("```text\n");
        out.push_str(before);
        out.push_str("\n```\n\n");
        out.push_str("修复后的 operands_part:\n");
        out.push_str("```text\n");
        out.push_str(after);
        out.push_str("\n```\n\n");
    }
    if let (Some(before), Some(after)) = (
        &example.before_syntax_operands,
        &example.after_syntax_operands,
    ) {
        out.push_str(&format!(
            "修复前的语法操作数: `{}`\n",
            escape_inline_code(&before.join(", "))
        ));
        out.push_str(&format!(
            "修复后的语法操作数: `{}`\n\n",
            escape_inline_code(&after.join(", "))
        ));
    } else {
        out.push('\n');
    }
}

fn format_instruction_name_blocks(names: &[String]) -> String {
    if names.is_empty() {
        return "`<无>`\n".to_string();
    }

    let mut out = String::new();
    let mut idx = 0usize;
    while idx < names.len() {
        let end = (idx + 12).min(names.len());
        out.push('`');
        out.push_str(&names[idx..end].join("`, `"));
        out.push_str("`\n");
        idx = end;
    }
    out
}

fn build_repair_summary(report: &FixerAuditReport) -> RepairSummary {
    let mut repair_instruction_names = BTreeSet::new();
    let mut special_fix_instruction_names = BTreeSet::new();
    let mut derivation_only_instruction_names = BTreeSet::new();
    let mut rule_to_fields: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut rule_to_examples: BTreeMap<String, Vec<RepairExample>> = BTreeMap::new();

    for audit in &report.instruction_audits {
        let mut touched_by_repair = false;
        let mut touched_by_special_fix = false;
        let mut touched_by_derivation = false;

        for step in &audit.steps {
            if is_repair_step(&step.step) {
                touched_by_repair = true;
                if is_special_fix_step(&step.step) {
                    touched_by_special_fix = true;
                }

                for field in step
                    .changes
                    .iter()
                    .filter_map(|change| extract_changed_field(change))
                {
                    rule_to_fields
                        .entry(step.step.clone())
                        .or_default()
                        .insert(field);
                }

                let examples = rule_to_examples.entry(step.step.clone()).or_default();
                if examples.len() < 3 {
                    examples.push(RepairExample {
                        instruction_name: audit.instruction_name.clone(),
                        changes: step.changes.clone(),
                        before_operands_part: step.before_operands_part.clone(),
                        after_operands_part: step.after_operands_part.clone(),
                        before_syntax_operands: step.before_syntax_operands.clone(),
                        after_syntax_operands: step.after_syntax_operands.clone(),
                    });
                }
            } else {
                touched_by_derivation = true;
            }
        }

        if touched_by_repair {
            repair_instruction_names.insert(audit.instruction_name.clone());
        }
        if touched_by_special_fix {
            special_fix_instruction_names.insert(audit.instruction_name.clone());
        }
        if touched_by_derivation && !touched_by_repair {
            derivation_only_instruction_names.insert(audit.instruction_name.clone());
        }
    }

    let mut rule_summaries = Vec::new();
    for stat in &report.rule_stats {
        if !is_repair_step(&stat.step) {
            continue;
        }

        let changed_fields = rule_to_fields
            .remove(&stat.step)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let examples = rule_to_examples.remove(&stat.step).unwrap_or_default();

        rule_summaries.push(RepairRuleSummary {
            step: stat.step.clone(),
            description: repair_step_description(&stat.step),
            changed_instructions: stat.changed_instructions,
            changed_fields,
            instruction_names: stat.instruction_names.clone(),
            examples,
        });
    }

    RepairSummary {
        repair_touched_instructions: repair_instruction_names.len(),
        special_fix_touched_instructions: special_fix_instruction_names.len(),
        reconcile_only_instructions: repair_instruction_names
            .difference(&special_fix_instruction_names)
            .count(),
        derivation_only_instructions: derivation_only_instruction_names.len(),
        rule_summaries,
    }
}

fn is_special_fix_step(step: &str) -> bool {
    step.starts_with("fix_")
}

fn is_derivation_step(step: &str) -> bool {
    matches!(step, "generate_assembly_syntax" | "assign_operand_kinds")
}

fn is_repair_step(step: &str) -> bool {
    !is_derivation_step(step)
}

fn extract_changed_field(change: &str) -> Option<String> {
    change
        .split_once(':')
        .map(|(field, _)| field.trim().to_string())
}

fn repair_step_description(step: &str) -> &'static str {
    match step {
        "reconcile_operand_names" => {
            "重命名操作数字段，使解析得到的 operand 列表与汇编语法占位符对齐。"
        }
        "fix_add_uw_missing_operands" => "补回 `add.uw` 缺失的操作数列表。",
        "fix_aes64ks1i_rnum_range" => "为 `aes64ks1i` 补上缺失的 `rnum` 取值范围约束。",
        "fix_amocas_pair_register_constraints" => {
            "为 AMOCAS 成对寄存器补上合法的偶数对齐寄存器约束。"
        }
        "fix_c_mop_n_instruction" => "为 `c.mop.n` 补上缺失的“仅奇数”约束。",
        "fix_clui_instruction" => "应用 `c.lui` 的特殊立即数约束，并移除通用占位符形式。",
        "fix_cm_instructions_reg_list_stack_adj" => {
            "把 `reg_list` 与 `stack_adj` 融合成 `cm.push/pop*` 使用的组合保存寄存器操作数。"
        }
        "fix_cm_mvsa01_instruction" => "把拆开的保存寄存器对替换为组合操作数 `ne_r1s_r2s`。",
        "fix_compressed_stack_adjustment_instructions" => {
            "规范化压缩栈调整指令的语法形式，确保硬编码的 `sp` 基址表达正确。"
        }
        "fix_compressed_stack_pointer_instructions" => {
            "规范化压缩栈指针访存语法，确保固定的 `sp` 基址被显式打印出来。"
        }
        "fix_fence_instruction" => "修复 `fence` 语法，使其暴露规范的 `pred, succ` 操作数。",
        "fix_fixed_aqrl_operands" => "移除 unified-db 已拆成独立别名后仍残留的固定 aq/rl 操作数。",
        "fix_fli_instruction_operands" => {
            "把 `fli.*` 中错误的寄存器源操作数替换为编码后的 FLI 常量操作数。"
        }
        "fix_floating_point_load_store_addressing" => {
            "修复浮点访存地址语法，使 offset 以 `imm(base)` 形式输出。"
        }
        "fix_floating_point_register_errors" => {
            "在 unified-db 使用了错误寄存器类别的地方，把整数寄存器占位符改成浮点寄存器操作数。"
        }
        "fix_hypervisor_load_store_instructions" => {
            "为 hypervisor load/store 指令补上缺失的 `0(base)` 地址形式。"
        }
        "fix_isabase_error_instructions" => {
            "修正本应只属于 RV32 或 RV64 的指令的 ISA base 与位宽元数据。"
        }
        "fix_missing_round_mode" => "当语义上仍存在 `rm` 时，把漏掉的 `rm` 重新插回打印语法。",
        "fix_prefetch_offset_alignment" => "补上 prefetch offset 缺失的对齐约束。",
        "fix_round_mode_removal" => "当指令使用固定或隐式舍入模式时，移除对外暴露的 `rm` 操作数。",
        "fix_sp_should_not_in_final_syntax_operands" => {
            "把硬编码的 `sp` 从最终语法操作数列表中移除，避免把它当成用户可提供操作数。"
        }
        "fix_uimm_instead_of_imm" => "把语义上无符号的立即数从 `imm` 重命名为 `uimm`。",
        "fix_vector_register_errors_wrapper" => "修复本应为标量整数寄存器的向量指令结果操作数。",
        "fix_vector_v0_hardcoded_operands" => {
            "把硬编码的 `v0` mask 操作数从用户可见的语法操作数列表中移除。"
        }
        "fix_vsetvli_operand_names" => {
            "把 `vsetvli` 中编码出来的 `vtypei` 操作数从错误的 `xs1` 占位符重命名回来。"
        }
        "fix_zclsd_pair_register_constraints" => "为 Zclsd 成对寄存器补上合法的偶数对齐取值约束。",
        _ => "未分类的修复步骤。",
    }
}

fn format_inline_code_list(values: &[String]) -> String {
    if values.is_empty() {
        return "`<无>`".to_string();
    }

    values
        .iter()
        .map(|value| format!("`{}`", escape_inline_code(value)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn escape_inline_code(value: &str) -> String {
    value.replace('`', "\\`")
}

fn escape_table_cell(value: &str) -> String {
    value.replace('|', "\\|")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn write_timestamped_audit_reports_creates_expected_files()
    -> Result<(), Box<dyn std::error::Error>> {
        let unique_suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "riscv-fixer-report-{}-{}",
            std::process::id(),
            unique_suffix
        ));

        let report = FixerAuditReport {
            total_instructions: 0,
            changed_instructions: 0,
            unchanged_instructions: 0,
            rule_stats: Vec::new(),
            instruction_audits: Vec::new(),
        };

        let paths =
            write_timestamped_audit_reports(&report, Path::new("assets/riscv-unified-db"), &root)?;

        assert_eq!(paths.output_dir, root.join(FIXER_REPORT_DIR_NAME));
        assert!(paths.json_path.exists());
        assert!(paths.repair_markdown_path.exists());

        let json_file_name = paths.json_path.file_name().and_then(|name| name.to_str());
        assert!(matches!(
            json_file_name,
            Some(name) if name.starts_with("fixer_audit_report_") && name.ends_with(".json")
        ));

        let markdown_file_name = paths
            .repair_markdown_path
            .file_name()
            .and_then(|name| name.to_str());
        assert!(matches!(
            markdown_file_name,
            Some(name) if name.starts_with("fixer_repair_report_") && name.ends_with(".md")
        ));

        fs::remove_dir_all(&root)?;
        Ok(())
    }
}
