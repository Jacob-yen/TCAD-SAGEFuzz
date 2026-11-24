//! diff-analysis 流水线的 Markdown 报告生成器。
//! Markdown report builders for the diff-analysis pipeline.
//!
//! 报告生成与归约逻辑分离，
//! Report generation is kept separate from reduction logic,
//! 这样 runner 可以专注控制流，而这个模块专注把中间状态转成便于人工排查的工件。
//! so the runner can stay focused on control flow while this module concentrates on turning
//! intermediate state into artifacts that are readable during manual bug triage.

use std::{
    collections::{BTreeMap, HashMap},
    fmt::Write as _,
    path::Path,
    time::Duration,
};

use riscv_fuzz_plugin_api::{
    ExecutionOutput,
    context::{MemoryContext, RegisterContext, format_memory_context, format_register_context},
};
use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;

use super::view::DiffExecutionView;

fn required_output<'a>(
    outputs: &'a HashMap<String, DiffExecutionView>,
    impl_ref: &str,
) -> Result<&'a DiffExecutionView, DiffAnalysisError> {
    outputs
        .get(impl_ref)
        .ok_or_else(|| DiffAnalysisError::MissingInstructions {
            core_id: impl_ref.to_string(),
        })
}

fn required_raw_output<'a>(
    outputs: &'a HashMap<String, ExecutionOutput>,
    impl_ref: &str,
) -> Result<&'a ExecutionOutput, DiffAnalysisError> {
    outputs
        .get(impl_ref)
        .ok_or_else(|| DiffAnalysisError::MissingInstructions {
            core_id: impl_ref.to_string(),
        })
}

fn required_test_instruction(
    testcase: &GeneratedTestCase,
    impl_ref: &str,
    test_index: usize,
) -> Result<String, DiffAnalysisError> {
    testcase
        .traceable_test_instruction_at(test_index)
        .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "missing traceable test instruction {} in {}",
                test_index, impl_ref
            ),
        })
}

fn required_register_context_display(
    contexts: &BTreeMap<String, RegisterContext>,
    impl_ref: &str,
    global_index: usize,
) -> Result<String, DiffAnalysisError> {
    let context =
        contexts
            .get(impl_ref)
            .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing register context for instruction {} in {}",
                    global_index, impl_ref
                ),
            })?;
    Ok(format_register_context(context))
}

fn required_memory_context_display(
    contexts: &BTreeMap<String, MemoryContext>,
    impl_ref: &str,
    global_index: usize,
) -> Result<String, DiffAnalysisError> {
    let context =
        contexts
            .get(impl_ref)
            .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing memory context for instruction {} in {}",
                    global_index, impl_ref
                ),
            })?;
    Ok(format_memory_context(context))
}

fn required_exception_cause_display(
    diff: &ExceptionDifference,
    impl_ref: &str,
) -> Result<String, DiffAnalysisError> {
    let causes = diff.per_impl.get(impl_ref).ok_or_else(|| {
        DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "missing exception cause entry for instruction {} in {}",
                diff.global_index, impl_ref
            ),
        }
    })?;
    Ok(if causes.is_empty() {
        "-".to_string()
    } else {
        causes
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    })
}

fn required_write_register_list(
    diff: &WriteDifference,
    impl_ref: &str,
) -> Result<String, DiffAnalysisError> {
    let entries = diff.register_changes.get(impl_ref).ok_or_else(|| {
        DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "missing register write entry for instruction {} in {}",
                diff.global_index, impl_ref
            ),
        }
    })?;
    Ok(format_register_list(entries))
}

fn required_write_memory_list(
    diff: &WriteDifference,
    impl_ref: &str,
) -> Result<String, DiffAnalysisError> {
    let entries = diff.memory_changes.get(impl_ref).ok_or_else(|| {
        DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "missing memory write entry for instruction {} in {}",
                diff.global_index, impl_ref
            ),
        }
    })?;
    Ok(format_memory_list(entries))
}

fn write_exception_table_header(out: &mut String) {
    writeln!(
        out,
        "| 实现 | 指令 | 寄存器上文 | 内存上文 | 异常 |\n|------|------|------------|------------|------|"
    )
    .unwrap();
}

fn write_write_table_header(out: &mut String) {
    writeln!(
        out,
        "| 实现 | 指令 | 寄存器上文 | 内存上文 | 寄存器写入 | 内存写入 |\n|------|------|------------|------------|------------|----------|"
    )
    .unwrap();
}

fn write_exception_rows(
    out: &mut String,
    testcase: &GeneratedTestCase,
    diff: &ExceptionDifference,
    impl_order: &[String],
) -> Result<(), DiffAnalysisError> {
    for impl_ref in impl_order {
        let inst = required_test_instruction(testcase, impl_ref, diff.test_index)?;
        let register_context = required_register_context_display(
            &diff.register_contexts,
            impl_ref,
            diff.global_index,
        )?;
        let memory_context =
            required_memory_context_display(&diff.memory_contexts, impl_ref, diff.global_index)?;
        let causes = required_exception_cause_display(diff, impl_ref)?;
        writeln!(
            out,
            "| {} | `{}` | {} | {} | {} |",
            impl_ref, inst, register_context, memory_context, causes
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    Ok(())
}

fn write_write_rows(
    out: &mut String,
    testcase: &GeneratedTestCase,
    diff: &WriteDifference,
    impl_order: &[String],
) -> Result<(), DiffAnalysisError> {
    for impl_ref in impl_order {
        let inst = required_test_instruction(testcase, impl_ref, diff.test_index)?;
        let register_context = required_register_context_display(
            &diff.register_contexts,
            impl_ref,
            diff.global_index,
        )?;
        let memory_context =
            required_memory_context_display(&diff.memory_contexts, impl_ref, diff.global_index)?;
        let regs = required_write_register_list(diff, impl_ref)?;
        let mems = required_write_memory_list(diff, impl_ref)?;
        writeln!(
            out,
            "| {} | `{}` | {} | {} | {} | {} |",
            impl_ref, inst, register_context, memory_context, regs, mems
        )
        .unwrap();
    }
    writeln!(out).unwrap();
    Ok(())
}

use super::{
    dedup::{DiffDedupOutcome, DiffDedupStatus},
    detection::{ExceptionDifference, WriteDifference},
    normalize::{WriteEffect, determine_test_length, test_instruction_location},
    report_io::write_string_report,
    types::DiffAnalysisError,
    verification::{
        InitializationMismatchData, InitializationSummary, classify_initialization_mismatch,
    },
};

pub(crate) enum ExecutionSummaryInitialization<'a> {
    Verified(&'a InitializationSummary),
    Failed {
        mismatch: &'a InitializationMismatchData,
        dedup_outcome: Option<&'a DiffDedupOutcome>,
    },
}

fn write_initialization_failure_details(
    out: &mut String,
    testcase: &GeneratedTestCase,
    impl_order: &[String],
    mismatch: &InitializationMismatchData,
    dedup_outcome: Option<&DiffDedupOutcome>,
) {
    let marker = classify_initialization_mismatch(mismatch);
    writeln!(
        out,
        "## Initialization Failure\n\n- Failing init instruction index: {}\n- BUG category: `{}`\n- BUG fingerprint: `{}`\n- Reason: {}\n",
        mismatch.instruction_index, marker.tag, marker.fingerprint, mismatch.reason
    )
    .unwrap();
    if let Some(outcome) = dedup_outcome {
        write_dedup_outcome(out, outcome);
    } else {
        writeln!(out, "- 自动去重：未启用\n").unwrap();
    }
    writeln!(
        out,
        "| Implementation | Init Instruction | Register Writes | Memory Writes |"
    )
    .unwrap();
    writeln!(
        out,
        "|------|------------------|-----------------|---------------|"
    )
    .unwrap();

    for impl_ref in impl_order {
        let instruction = testcase
            .init_rendered_line_at_of(impl_ref, mismatch.instruction_index)
            .unwrap_or_else(|| "-".to_string());
        let regs = mismatch
            .per_impl_registers
            .get(impl_ref)
            .map(|entries| format_register_list(entries))
            .unwrap_or_else(|| "-".to_string());
        let mems = mismatch
            .per_impl_memory
            .get(impl_ref)
            .map(|entries| format_memory_list(entries))
            .unwrap_or_else(|| "-".to_string());
        writeln!(
            out,
            "| {} | `{}` | {} | {} |",
            impl_ref, instruction, regs, mems
        )
        .unwrap();
    }
}

pub(crate) fn write_exception_report(
    path: &Path,
    previous_testcase: &GeneratedTestCase,
    current_testcase: &GeneratedTestCase,
    differences: &[(ExceptionDifference, Option<DiffDedupOutcome>)],
    impl_order: &[String],
) -> Result<(), DiffAnalysisError> {
    let mut content = String::new();
    let old_len = determine_test_length(impl_order, previous_testcase)?;
    let new_len = determine_test_length(impl_order, current_testcase)?;
    let removed_count = old_len.saturating_sub(new_len);

    writeln!(
        &mut content,
        "# 异常差异移除报告\n\n- 原测试指令数量：{}\n- 移除数量：{}\n- 移除后测试指令数量：{}\n",
        old_len, removed_count, new_len
    )
    .unwrap();

    for (diff, dedup_outcome) in differences {
        writeln!(
            &mut content,
            "## 测试指令 #{} （全局索引 {}）\n",
            diff.test_index, diff.global_index
        )
        .unwrap();
        if let Some(outcome) = dedup_outcome {
            write_dedup_outcome(&mut content, outcome);
        } else {
            writeln!(&mut content, "- 自动去重：未启用\n").unwrap();
        }
        write_exception_table_header(&mut content);
        write_exception_rows(&mut content, previous_testcase, diff, impl_order)?;
    }

    write_string_report(path, content)
}

pub(crate) fn write_exception_cause_report(
    path: &Path,
    testcase: &GeneratedTestCase,
    differences: &[(ExceptionDifference, Option<DiffDedupOutcome>)],
    impl_order: &[String],
) -> Result<(), DiffAnalysisError> {
    let test_len = determine_test_length(impl_order, testcase)?;
    let mut content = String::new();
    writeln!(
        &mut content,
        "# 异常原因差异报告\n\n- 测试指令数量：{}\n- 异常原因差异数量：{}\n",
        test_len,
        differences.len()
    )
    .unwrap();

    for (diff, dedup_outcome) in differences {
        writeln!(
            &mut content,
            "## 测试指令 #{} （全局索引 {}）\n",
            diff.test_index, diff.global_index
        )
        .unwrap();
        if let Some(outcome) = dedup_outcome {
            write_dedup_outcome(&mut content, outcome);
        } else {
            writeln!(&mut content, "- 自动去重：未启用\n").unwrap();
        }
        write_exception_table_header(&mut content);
        write_exception_rows(&mut content, testcase, diff, impl_order)?;
    }

    write_string_report(path, content)
}

pub(crate) fn write_write_report(
    path: &Path,
    previous_testcase: &GeneratedTestCase,
    current_testcase: &GeneratedTestCase,
    differences: &[(WriteDifference, String, Option<DiffDedupOutcome>)],
    impl_order: &[String],
) -> Result<(), DiffAnalysisError> {
    let mut content = String::new();
    let old_len = determine_test_length(impl_order, previous_testcase)?;
    let new_len = determine_test_length(impl_order, current_testcase)?;
    let removed_count = old_len.saturating_sub(new_len);
    let extra_removed = removed_count.saturating_sub(differences.len());

    writeln!(
        &mut content,
        "# 写入差异移除报告\n\n- 原测试指令数量：{}\n- 移除数量：{}\n- 移除后测试指令数量：{}\n",
        old_len, removed_count, new_len
    )
    .unwrap();
    if extra_removed > 0 {
        writeln!(&mut content, "- 其中历史指令额外移除：{}\n", extra_removed).unwrap();
    }

    for (diff, cause_detail, dedup_outcome) in differences.iter() {
        writeln!(
            &mut content,
            "## 测试指令 #{}, 全局索引 {}\n",
            diff.test_index, diff.global_index
        )
        .unwrap();
        let classification = describe_write_difference_pattern(diff, impl_order);
        let classification = classification?;
        writeln!(&mut content, "- 分类：{}\n", classification).unwrap();
        if let Some(outcome) = dedup_outcome {
            write_dedup_outcome(&mut content, outcome);
        } else {
            writeln!(&mut content, "- 自动去重：未启用\n").unwrap();
        }
        writeln!(&mut content, "- {}\n", cause_detail).unwrap();
        write_write_table_header(&mut content);
        write_write_rows(&mut content, previous_testcase, diff, impl_order)?;
    }

    write_string_report(path, content)
}

fn write_dedup_outcome(out: &mut String, outcome: &DiffDedupOutcome) {
    let status = match &outcome.status {
        DiffDedupStatus::FirstSeen => "首次出现".to_string(),
        DiffDedupStatus::Duplicate {
            previous_count,
            first_seen_run_root,
            first_seen_report_path,
        } => {
            let mut text = format!("重复；此前已出现 {} 次", previous_count);
            if let Some(path) = first_seen_run_root {
                text.push_str(&format!("；首次 run `{}`", path));
            }
            if let Some(path) = first_seen_report_path {
                text.push_str(&format!("；首次报告 `{}`", path));
            }
            text
        }
    };

    writeln!(out, "- 自动标签：`{}`", outcome.event.auto_tag).unwrap();
    writeln!(out, "- 去重 family：`{}`", outcome.event.family_fingerprint).unwrap();
    writeln!(out, "- 去重 strict：`{}`", outcome.event.strict_fingerprint).unwrap();
    writeln!(out, "- 去重状态：{}", status).unwrap();
    writeln!(out, "- 去重库：`{}`", outcome.database_path.display()).unwrap();
    writeln!(out, "- 碰撞保护：{}", outcome.event.collision_guard).unwrap();
    if outcome.malformed_records_ignored > 0 {
        writeln!(
            out,
            "- 去重库警告：忽略了 {} 条无法解析的历史记录",
            outcome.malformed_records_ignored
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

pub(crate) fn write_replay_write_report(
    path: &Path,
    testcase: &GeneratedTestCase,
    difference: &WriteDifference,
    impl_order: &[String],
    cause_detail: &str,
) -> Result<(), DiffAnalysisError> {
    let mut content = String::new();
    writeln!(
        &mut content,
        "# 写入差异重放报告\n\n- 差异来源：{}\n- 测试指令索引：{}\n- 全局索引：{}\n",
        cause_detail, difference.test_index, difference.global_index
    )
    .unwrap();

    write_write_table_header(&mut content);
    write_write_rows(&mut content, testcase, difference, impl_order)?;

    write_string_report(path, content)
}

pub(crate) fn write_replay_write_failure_report(
    path: &Path,
    testcase: &GeneratedTestCase,
    difference: &WriteDifference,
    impl_order: &[String],
    expected_index: usize,
    failure_reason: &str,
) -> Result<(), DiffAnalysisError> {
    let actual_global = difference.global_index;
    let actual_index = difference.test_index;
    let expected_global =
        test_instruction_location(impl_order, testcase, expected_index)?.global_index;

    let mut content = String::new();
    writeln!(
        &mut content,
        "# 写入差异重放报告（失败）\n\n- 目标测试指令索引：{}\n- 实际首个差异索引：{}\n- 目标全局索引：{}\n- 实际全局索引：{}\n- 失败原因：{}\n",
        expected_index,
        actual_index,
        expected_global,
        actual_global,
        failure_reason
    )
    .unwrap();

    writeln!(
        &mut content,
        "- 后续动作：由于写入重放失败，分析流程回退为“单条测试指令引发差异”处理，将直接移除目标指令。\n"
    )
    .unwrap();

    write_string_report(path, content)
}

pub(crate) fn write_history_minimization_error_report(
    path: &Path,
    difference: &WriteDifference,
    reason: &str,
) -> Result<(), DiffAnalysisError> {
    let mut content = String::new();
    writeln!(
        &mut content,
        "# 历史指令最小化报告（失败）\n\n- 测试指令索引：{}\n- 全局索引：{}\n- 失败原因：{}\n",
        difference.test_index, difference.global_index, reason
    )
    .unwrap();

    writeln!(
        &mut content,
        "| 实现 | 寄存器写入 | 内存写入 |\n|------|------------|----------|"
    )
    .unwrap();

    for (impl_ref, regs) in &difference.register_changes {
        let mems = required_write_memory_list(difference, impl_ref)?;
        let regs_fmt = format_register_list(regs);
        writeln!(
            &mut content,
            "| {} | {} | {} |",
            impl_ref,
            if regs_fmt.is_empty() {
                "-".to_string()
            } else {
                regs_fmt
            },
            if mems.is_empty() {
                "-".to_string()
            } else {
                mems
            }
        )
        .unwrap();
    }
    writeln!(&mut content).unwrap();

    write_string_report(path, content)
}

pub(crate) fn write_history_minimization_report(
    path: &Path,
    original_difference: &WriteDifference,
    minimized_start: usize,
    original_target_index: usize,
    minimized_testcase: &GeneratedTestCase,
    minimized_difference: Option<&WriteDifference>,
    impl_order: &[String],
) -> Result<(), DiffAnalysisError> {
    let mut content = String::new();
    let new_target_index = original_target_index
        .checked_sub(minimized_start)
        .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "history minimization produced invalid index: start={}, target={}",
                minimized_start, original_target_index
            ),
        })?;

    writeln!(
        &mut content,
        "# 历史差异最小化报告\n\n- 原始全局索引：{}\n- 原始测试指令索引：{}\n- 最小化后保留的历史起点：{}\n- 历史指令数量：{}\n",
        original_difference.global_index,
        original_target_index,
        minimized_start,
        original_target_index.saturating_sub(minimized_start)
    )
    .unwrap();

    if let Some(diff) = minimized_difference {
        let classification = describe_write_difference_pattern(diff, impl_order);
        writeln!(&mut content, "- 分类：{}\n", classification?).unwrap();
        write_write_table_header(&mut content);
        write_write_rows(&mut content, minimized_testcase, diff, impl_order)?;
    } else {
        writeln!(
            &mut content,
            "> 警告：最小化后的用例未在索引 #{} 检测到写入差异，可能需要手动检查。",
            new_target_index
        )
        .unwrap();
    }

    write_string_report(path, content)
}

pub(crate) fn write_history_candidate_report(
    path: &Path,
    original_start: usize,
    original_target_index: usize,
    difference: Option<&WriteDifference>,
    testcase: &GeneratedTestCase,
    impl_order: &[String],
    cause_detail: &str,
) -> Result<(), DiffAnalysisError> {
    let mut content = String::new();
    let display_global =
        test_instruction_location(impl_order, testcase, original_target_index)?.global_index;

    match difference {
        Some(diff) => {
            let display_index = original_start + diff.test_index;
            writeln!(
                &mut content,
                "## 测试指令 #{}, 全局索引 {}\n",
                display_index, display_global
            )
            .unwrap();
            let classification = describe_write_difference_pattern(diff, impl_order)?;
            writeln!(&mut content, "- 分类：{}\n", classification).unwrap();
            writeln!(&mut content, "- {}\n", cause_detail).unwrap();
            write_write_table_header(&mut content);
            write_write_rows(&mut content, testcase, diff, impl_order)?;
        }
        None => {
            writeln!(
                &mut content,
                "## 测试指令 #{}, 全局索引 {}\n",
                original_target_index, display_global
            )
            .unwrap();
            if !cause_detail.is_empty() {
                writeln!(&mut content, "- {}\n", cause_detail).unwrap();
            }
            writeln!(&mut content, "> 在该候选用例中未检测到历史差异复现。").unwrap();
        }
    }

    write_string_report(path, content)
}

pub(crate) fn write_execution_state_report(
    path: &Path,
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
    impl_order: &[String],
    timings: Option<&HashMap<String, Duration>>,
    initialization: Option<ExecutionSummaryInitialization<'_>>,
) -> Result<(), DiffAnalysisError> {
    let test_len = determine_test_length(impl_order, testcase)?;

    let mut content = String::new();
    let impl_list = if impl_order.is_empty() {
        "-".to_string()
    } else {
        impl_order.join(", ")
    };
    let (init_status, init_detail) = match &initialization {
        Some(ExecutionSummaryInitialization::Verified(summary)) => (
            "PASS",
            format!(
                "verified {} initialization instructions",
                summary.init_instruction_count
            ),
        ),
        Some(ExecutionSummaryInitialization::Failed { mismatch, .. }) => {
            ("FAIL", mismatch.reason.clone())
        }
        None => (
            "not rechecked",
            "this phase reuses the shared summary format".to_string(),
        ),
    };

    writeln!(
        &mut content,
        "# Execution Summary\n\n| Field | Value |\n|------|-------|\n| Test instructions | {} |\n| Implementations | {} |\n| Initialization | {} |\n| Initialization detail | {} |\n",
        test_len, impl_list, init_status, init_detail
    )
    .unwrap();

    writeln!(&mut content, "## Implementation Overview\n").unwrap();
    writeln!(
        &mut content,
        "| Implementation | Duration (s) | Exception Count | Instructions With Writes |\n|------|--------------:|----------------:|-------------------------:|"
    )
    .unwrap();
    for impl_ref in impl_order {
        let ctx = required_output(outputs, impl_ref)?;
        let exception_count = ctx
            .logical_view
            .iter()
            .filter(|instruction| !instruction.exceptions.is_empty())
            .count();
        let write_count = ctx
            .logical_view
            .iter()
            .filter(|instruction| {
                !instruction.register_writes.is_empty() || !instruction.memory_writes.is_empty()
            })
            .count();
        let duration = timings
            .and_then(|per_impl| per_impl.get(impl_ref))
            .map(|elapsed| format!("{:.3}", elapsed.as_secs_f64()))
            .unwrap_or_else(|| "-".to_string());
        writeln!(
            &mut content,
            "| {} | {} | {} | {} |",
            impl_ref, duration, exception_count, write_count
        )
        .unwrap();
    }

    if let Some(ExecutionSummaryInitialization::Failed {
        mismatch,
        dedup_outcome,
    }) = initialization
    {
        writeln!(&mut content).unwrap();
        write_initialization_failure_details(
            &mut content,
            testcase,
            impl_order,
            mismatch,
            dedup_outcome,
        );
    }

    write_string_report(path, content)
}

pub(crate) fn write_final_summary(
    path: &Path,
    initial_testcase: &GeneratedTestCase,
    final_testcase: &GeneratedTestCase,
    final_outputs: &HashMap<String, ExecutionOutput>,
    impl_order: &[String],
    exception_rounds: usize,
    write_rounds: usize,
) -> Result<(), DiffAnalysisError> {
    let initial_len = determine_test_length(impl_order, initial_testcase)?;
    let final_len = determine_test_length(impl_order, final_testcase)?;

    let mut content = String::new();
    writeln!(
        &mut content,
        "# Diff Analysis 最终报告\n\n- 初始测试指令数量：{}\n- 最终测试指令数量：{}\n- 异常差异移除轮次：{}\n- 写入差异移除轮次：{}\n",
        initial_len, final_len, exception_rounds, write_rounds
    )
    .unwrap();

    // 这里是 best-effort 统计：某些实现可能完全没有 perf 指标。
    // This section is best-effort: metrics may be missing for some implementations.
    writeln!(&mut content, "## 性能指标（best-effort）\n").unwrap();

    // 这些 baseline 用于可选的减法列，取所有实现中的最小值。
    // These baselines are used for optional subtraction columns and take the minimum across
    // implementations.
    let mut global_cycle_baseline: Option<u64> = None;
    for impl_ref in impl_order {
        let output = required_raw_output(final_outputs, impl_ref)?;
        if let Some(value) = output.perf.global_cycles {
            global_cycle_baseline = Some(
                global_cycle_baseline
                    .map(|baseline| baseline.min(value))
                    .unwrap_or(value),
            );
        }
    }

    // 分 hart 收集 mcycle/minstret 的 baseline。
    // Collect per-hart baselines for mcycle/minstret deltas.
    let mut hart0_mcycle_baseline: Option<u64> = None;
    let mut hart0_minstret_baseline: Option<u64> = None;
    let mut hart1_mcycle_baseline: Option<u64> = None;
    let mut hart1_minstret_baseline: Option<u64> = None;
    for impl_ref in impl_order {
        let output = required_raw_output(final_outputs, impl_ref)?;
        if let Some(h0) = output.perf.per_hart.get(&0) {
            if let Some(v) = h0.mcycle_delta() {
                hart0_mcycle_baseline = Some(hart0_mcycle_baseline.map(|b| b.min(v)).unwrap_or(v));
            }
            if let Some(v) = h0.minstret_delta() {
                hart0_minstret_baseline =
                    Some(hart0_minstret_baseline.map(|b| b.min(v)).unwrap_or(v));
            }
        }
        if let Some(h1) = output.perf.per_hart.get(&1) {
            if let Some(v) = h1.mcycle_delta() {
                hart1_mcycle_baseline = Some(hart1_mcycle_baseline.map(|b| b.min(v)).unwrap_or(v));
            }
            if let Some(v) = h1.minstret_delta() {
                hart1_minstret_baseline =
                    Some(hart1_minstret_baseline.map(|b| b.min(v)).unwrap_or(v));
            }
        }
    }

    fn fmt_opt_u64(v: Option<u64>) -> String {
        v.map(|x| x.to_string()).unwrap_or_else(|| "-".to_string())
    }

    fn fmt_delta(v: Option<u64>, baseline: Option<u64>) -> String {
        match (v, baseline) {
            (Some(value), Some(base)) => value.saturating_sub(base).to_string(),
            _ => "-".to_string(),
        }
    }

    fn fmt_opt_f64(v: Option<f64>) -> String {
        v.map(|x| format!("{:.4}", x))
            .unwrap_or_else(|| "-".to_string())
    }

    writeln!(
        &mut content,
        "| 实现 | global_cycles | global_cycles - min | hart | mcycle | mcycle - min | minstret | minstret - min | CPI |\n|------|-------------:|-------------------:|-----:|------:|------------:|---------:|---------------:|-----:|"
    )
    .unwrap();

    for impl_ref in impl_order {
        let output = required_raw_output(final_outputs, impl_ref)?;
        let global_cycles = output.perf.global_cycles;
        let global_cycles_delta = fmt_delta(global_cycles, global_cycle_baseline);

        // 即使某个实现完全没有 perf 数据，也仍然输出一行，避免整张表缺项。
        // If perf is missing entirely, still emit a single row.
        let mut emitted_any = false;
        for hart in [0u32, 1u32] {
            let hart_metrics = output.perf.per_hart.get(&hart);
            let mcycle = hart_metrics.and_then(|h| h.mcycle_delta());
            let minstret = hart_metrics.and_then(|h| h.minstret_delta());
            let cpi = hart_metrics.and_then(|h| h.cpi());

            let (mcycle_baseline, minstret_baseline) = match hart {
                0 => (hart0_mcycle_baseline, hart0_minstret_baseline),
                1 => (hart1_mcycle_baseline, hart1_minstret_baseline),
                _ => (None, None),
            };

            if global_cycles.is_none() && mcycle.is_none() && minstret.is_none() && cpi.is_none() {
                continue;
            }
            emitted_any = true;
            writeln!(
                &mut content,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                impl_ref,
                fmt_opt_u64(global_cycles),
                global_cycles_delta,
                hart,
                fmt_opt_u64(mcycle),
                fmt_delta(mcycle, mcycle_baseline),
                fmt_opt_u64(minstret),
                fmt_delta(minstret, minstret_baseline),
                fmt_opt_f64(cpi),
            )
            .unwrap();
        }

        if !emitted_any {
            writeln!(
                &mut content,
                "| {} | {} | {} | - | - | - | - | - | - |",
                impl_ref,
                fmt_opt_u64(global_cycles),
                global_cycles_delta,
            )
            .unwrap();
        }
    }

    write_string_report(path, content)
}

fn describe_write_difference_pattern(
    diff: &WriteDifference,
    impl_order: &[String],
) -> Result<&'static str, DiffAnalysisError> {
    let mut signature_counts: HashMap<WriteEffect, usize> = HashMap::new();

    for impl_ref in impl_order {
        let register_changes = diff
            .register_changes
            .get(impl_ref)
            .cloned()
            .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing register write entry for instruction {} in {}",
                    diff.global_index, impl_ref
                ),
            })?;
        let memory_changes = diff.memory_changes.get(impl_ref).cloned().ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing memory write entry for instruction {} in {}",
                    diff.global_index, impl_ref
                ),
            }
        })?;
        *signature_counts
            .entry((register_changes, memory_changes))
            .or_insert(0) += 1;
    }

    if signature_counts.values().all(|count| *count == 1) {
        Ok("每个impl的write都不相同")
    } else {
        Ok("多数派对少数派")
    }
}

pub(crate) fn format_register_list(entries: &[(String, u64)]) -> String {
    if entries.is_empty() {
        "-".to_string()
    } else {
        entries
            .iter()
            .map(|(name, value)| format!("{}=0x{:x}", name, value))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

pub(crate) fn format_memory_list(entries: &[(u64, u8)]) -> String {
    if entries.is_empty() {
        "-".to_string()
    } else {
        entries
            .iter()
            .map(|(addr, value)| format!("0x{:x}=0x{:02x}", addr, value))
            .collect::<Vec<_>>()
            .join(", ")
    }
}
