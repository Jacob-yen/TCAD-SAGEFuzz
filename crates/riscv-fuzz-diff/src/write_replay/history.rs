use std::{
    collections::{BTreeSet, HashMap},
    fs,
    io::ErrorKind,
    path::Path,
};

use riscv_fuzz_plugin_api::CoverageMode;
use riscv_fuzz_plugin_api::ExecutionOutput;
use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;

use super::{
    DiffAnalysisError, HISTORY_MIN_START_CANDIDATE_LOSS_REASON, WriteDifference,
    detect_unique_exceptions, detect_unique_writes, determine_test_length, iteration_summary_path,
    persist_iteration_impl_artifacts, persist_iteration_testcase, run_iteration,
    write_execution_state_report, write_history_candidate_report,
    write_history_minimization_report,
};
use crate::normalize::{logical_instruction_at_location, test_instruction_location};

#[derive(Clone)]
struct HistoryEvaluationSummary {
    start_index: usize,
    status: HistoryCandidateStatus,
    difference: Option<WriteDifference>,
}

#[derive(Clone)]
struct HistoryEvaluation {
    summary: HistoryEvaluationSummary,
    testcase: GeneratedTestCase,
    raw_outputs: HashMap<String, ExecutionOutput>,
}

impl HistoryEvaluation {
    fn summary(&self) -> &HistoryEvaluationSummary {
        &self.summary
    }

    fn start_index(&self) -> usize {
        self.summary.start_index
    }

    fn status(&self) -> &HistoryCandidateStatus {
        &self.summary.status
    }

    fn difference(&self) -> Option<&WriteDifference> {
        self.summary.difference.as_ref()
    }
}

#[derive(Clone)]
enum HistoryCandidateStatus {
    TargetHasDifference,
    TargetExceptionMismatch,
    TargetExceptionAll,
    TargetNoDifference,
    NonTargetDifference { test_index: usize },
    NonTargetException { test_index: usize },
}

#[derive(Clone, Copy)]
pub(super) struct HistoryMinimizationConfig {
    pub(super) verification_round: usize,
    pub(super) history_min_threshold: Option<usize>,
    pub(super) coverage_mode: CoverageMode,
    pub(super) emit_execution_output_json: bool,
    pub(super) emit_execution_report_md: bool,
    pub(super) cleanup_history_candidates: bool,
}

pub(super) fn minimize_history_write_difference(
    diff: &WriteDifference,
    testcase: &GeneratedTestCase,
    impl_order: &[String],
    iteration_dir: &Path,
    config: HistoryMinimizationConfig,
) -> Result<(), DiffAnalysisError> {
    if diff.test_index == 0 {
        return Ok(());
    }

    if let Some(threshold) = config.history_min_threshold {
        let current_len = determine_test_length(impl_order, testcase)?;
        if current_len <= threshold {
            return Ok(());
        }
    }

    let target_index = diff.test_index;
    let minimize_root = iteration_dir.join(format!(
        "write_history_min_{:03}",
        config.verification_round
    ));
    fs::create_dir_all(&minimize_root)?;

    let mut cache: HashMap<usize, HistoryEvaluationSummary> = HashMap::new();
    let mut evaluated_starts = BTreeSet::new();

    let initial_eval = evaluate_history_candidate(
        testcase,
        impl_order,
        &minimize_root,
        0,
        target_index,
        config,
    )?;
    cache.insert(0, initial_eval.summary().clone());
    evaluated_starts.insert(0);
    let mut best_start = 0usize;
    let mut best_eval = initial_eval;

    match best_eval.status() {
        HistoryCandidateStatus::TargetHasDifference => {}
        HistoryCandidateStatus::TargetExceptionMismatch
        | HistoryCandidateStatus::TargetExceptionAll => {
            return conclude_history_minimization(
                diff,
                &best_eval,
                best_start,
                target_index,
                impl_order,
                &minimize_root,
                config,
                &evaluated_starts,
            );
        }
        HistoryCandidateStatus::TargetNoDifference => {
            return Err(DiffAnalysisError::HistoryMinimizationFailure {
                reason: HISTORY_MIN_START_CANDIDATE_LOSS_REASON.to_string(),
            });
        }
        HistoryCandidateStatus::NonTargetDifference { test_index } => {
            let original_index = best_eval.start_index() + test_index;
            return Err(DiffAnalysisError::HistoryMinimizationFailure {
                reason: format!(
                    "历史最小化起始候选在指令 #{} 上仍存在写入差异",
                    original_index
                ),
            });
        }
        HistoryCandidateStatus::NonTargetException { test_index } => {
            let original_index = best_eval.start_index() + test_index;
            return Err(DiffAnalysisError::HistoryMinimizationFailure {
                reason: format!(
                    "历史最小化起始候选在指令 #{} 上出现异常差异",
                    original_index
                ),
            });
        }
    }

    let mut good = 0usize;
    let mut bad = target_index + 1usize;

    // 这里对“还能保留目标差异的最晚前缀起点”做二分搜索：
    // Binary-search the latest removable prefix start:
    // - `good` 这一侧仍然能复现目标差异。
    //   `good` still reproduces the target difference.
    // - `bad` 这一侧已经丢失目标差异。
    //   `bad` already loses it.
    while good + 1 < bad {
        let mid = (good + bad) / 2;
        let cached_summary = if let Some(existing) = cache.get(&mid) {
            evaluated_starts.insert(mid);
            Some(existing.clone())
        } else {
            None
        };

        if let Some(summary) = cached_summary {
            match &summary.status {
                HistoryCandidateStatus::TargetHasDifference => {
                    good = mid;
                }
                HistoryCandidateStatus::TargetNoDifference => {
                    bad = mid;
                }
                HistoryCandidateStatus::TargetExceptionMismatch
                | HistoryCandidateStatus::TargetExceptionAll => {
                    break;
                }
                HistoryCandidateStatus::NonTargetDifference { test_index } => {
                    if best_eval.difference().is_some() {
                        return conclude_history_minimization(
                            diff,
                            &best_eval,
                            best_start,
                            target_index,
                            impl_order,
                            &minimize_root,
                            config,
                            &evaluated_starts,
                        );
                    } else {
                        let original_index = summary.start_index + test_index;
                        return Err(DiffAnalysisError::HistoryMinimizationFailure {
                            reason: format!(
                                "历史最小化失败：指令 #{} 出现额外写入差异",
                                original_index
                            ),
                        });
                    }
                }
                HistoryCandidateStatus::NonTargetException { test_index } => {
                    if best_eval.difference().is_some() {
                        return conclude_history_minimization(
                            diff,
                            &best_eval,
                            best_start,
                            target_index,
                            impl_order,
                            &minimize_root,
                            config,
                            &evaluated_starts,
                        );
                    } else {
                        let original_index = summary.start_index + test_index;
                        return Err(DiffAnalysisError::HistoryMinimizationFailure {
                            reason: format!(
                                "历史最小化失败：指令 #{} 出现异常差异",
                                original_index
                            ),
                        });
                    }
                }
            }
            continue;
        }

        let eval = evaluate_history_candidate(
            testcase,
            impl_order,
            &minimize_root,
            mid,
            target_index,
            config,
        )?;
        cache.insert(mid, eval.summary().clone());
        evaluated_starts.insert(mid);

        match eval.status() {
            HistoryCandidateStatus::TargetHasDifference => {
                good = mid;
                best_start = mid;
                best_eval = eval;
            }
            HistoryCandidateStatus::TargetNoDifference => {
                bad = mid;
            }
            HistoryCandidateStatus::TargetExceptionMismatch
            | HistoryCandidateStatus::TargetExceptionAll => {
                if eval.difference().is_some() {
                    best_start = mid;
                    best_eval = eval;
                }
                break;
            }
            HistoryCandidateStatus::NonTargetDifference { test_index } => {
                if best_eval.difference().is_some() {
                    return conclude_history_minimization(
                        diff,
                        &best_eval,
                        best_start,
                        target_index,
                        impl_order,
                        &minimize_root,
                        config,
                        &evaluated_starts,
                    );
                } else {
                    let original_index = eval.start_index() + test_index;
                    return Err(DiffAnalysisError::HistoryMinimizationFailure {
                        reason: format!(
                            "历史最小化失败：指令 #{} 出现额外写入差异",
                            original_index
                        ),
                    });
                }
            }
            HistoryCandidateStatus::NonTargetException { test_index } => {
                if best_eval.difference().is_some() {
                    return conclude_history_minimization(
                        diff,
                        &best_eval,
                        best_start,
                        target_index,
                        impl_order,
                        &minimize_root,
                        config,
                        &evaluated_starts,
                    );
                } else {
                    let original_index = eval.start_index() + test_index;
                    return Err(DiffAnalysisError::HistoryMinimizationFailure {
                        reason: format!("历史最小化失败：指令 #{} 出现异常差异", original_index),
                    });
                }
            }
        }
    }

    if best_eval.difference().is_none() {
        return Err(DiffAnalysisError::HistoryMinimizationFailure {
            reason: "无法定位目标指令写入差异最小化结果".to_string(),
        });
    }

    conclude_history_minimization(
        diff,
        &best_eval,
        best_start,
        target_index,
        impl_order,
        &minimize_root,
        config,
        &evaluated_starts,
    )
}

fn finalize_history_minimization(
    diff: &WriteDifference,
    best_eval: &HistoryEvaluation,
    best_start: usize,
    target_index: usize,
    impl_order: &[String],
    minimize_root: &Path,
    config: HistoryMinimizationConfig,
) -> Result<(), DiffAnalysisError> {
    let minimized_dir = minimize_root.join("final");
    fs::create_dir_all(&minimized_dir)?;
    persist_iteration_testcase(&best_eval.testcase, &minimized_dir)?;
    persist_iteration_impl_artifacts(
        impl_order,
        &minimized_dir,
        &best_eval.raw_outputs,
        config.emit_execution_output_json,
        config.emit_execution_report_md,
    )?;

    let report_path = minimize_root.join("report_history_minimization.md");
    write_history_minimization_report(
        &report_path,
        diff,
        best_start,
        target_index,
        &best_eval.testcase,
        best_eval.difference(),
        impl_order,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn conclude_history_minimization(
    diff: &WriteDifference,
    best_eval: &HistoryEvaluation,
    best_start: usize,
    target_index: usize,
    impl_order: &[String],
    minimize_root: &Path,
    config: HistoryMinimizationConfig,
    evaluated_starts: &BTreeSet<usize>,
) -> Result<(), DiffAnalysisError> {
    finalize_history_minimization(
        diff,
        best_eval,
        best_start,
        target_index,
        impl_order,
        minimize_root,
        config,
    )?;

    if config.cleanup_history_candidates {
        cleanup_history_minimization_attempts(minimize_root, evaluated_starts, best_start)
            .map_err(DiffAnalysisError::Io)?;
    }

    Ok(())
}

fn cleanup_history_minimization_attempts(
    minimize_root: &Path,
    evaluated_starts: &BTreeSet<usize>,
    best_start: usize,
) -> std::io::Result<()> {
    for start in evaluated_starts {
        if *start == best_start {
            continue;
        }
        let candidate_dir = minimize_root.join(format!("candidate_{:03}", start));
        match fs::remove_dir_all(&candidate_dir) {
            Ok(_) => {}
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_history_candidate(
    testcase: &GeneratedTestCase,
    impl_order: &[String],
    minimize_root: &Path,
    start_index: usize,
    target_index: usize,
    config: HistoryMinimizationConfig,
) -> Result<HistoryEvaluation, DiffAnalysisError> {
    if start_index > target_index {
        return Err(DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "invalid history minimization range: start {} > target {}",
                start_index, target_index
            ),
        });
    }

    let trimmed = build_history_trimmed_testcase(testcase, impl_order, start_index, target_index)?;
    let candidate_dir = minimize_root.join(format!("candidate_{:03}", start_index));
    let execution = run_iteration(
        &trimmed,
        impl_order,
        &candidate_dir,
        config.coverage_mode,
        config.emit_execution_output_json,
        config.emit_execution_report_md,
    )?;
    let outputs = execution.outputs;
    let raw_outputs = execution.raw_outputs;
    let summary_path = iteration_summary_path(&candidate_dir);
    write_execution_state_report(
        &summary_path,
        &trimmed,
        &outputs,
        impl_order,
        Some(&execution.timings),
        None,
    )?;
    let new_target = target_index.checked_sub(start_index).ok_or_else(|| {
        DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "history minimization computed negative target: start={}, target={}",
                start_index, target_index
            ),
        }
    })?;
    let differences = detect_unique_writes(impl_order, &trimmed, &outputs)?;
    let mut target_difference: Option<WriteDifference> = None;
    let mut other_difference: Option<WriteDifference> = None;
    for diff in &differences {
        if diff.test_index == new_target && target_difference.is_none() {
            target_difference = Some(diff.clone());
        } else if other_difference.is_none() {
            other_difference = Some(diff.clone());
        }
    }

    let exceptions = detect_unique_exceptions(impl_order, &trimmed, &outputs)?;
    let mut target_exception_mismatch = false;
    let mut other_exception_index: Option<usize> = None;
    for exc in exceptions
        .trigger_mismatches
        .iter()
        .chain(exceptions.cause_mismatches.iter())
    {
        if exc.test_index == new_target {
            target_exception_mismatch = true;
        } else if other_exception_index.is_none() {
            other_exception_index = Some(exc.test_index);
        }
    }

    let target_location = test_instruction_location(impl_order, &trimmed, new_target)?;
    let mut target_exception_count = 0usize;
    for impl_ref in impl_order {
        let output =
            outputs
                .get(impl_ref)
                .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                    core_id: impl_ref.to_string(),
                })?;
        if logical_instruction_at_location(output, target_location)
            .map(|instruction| !instruction.exceptions.is_empty())
            .unwrap_or(false)
        {
            target_exception_count += 1;
        }
    }
    let target_exception_all = target_exception_count == impl_order.len();

    let status = if let Some(diff) = other_difference.as_ref() {
        HistoryCandidateStatus::NonTargetDifference {
            test_index: diff.test_index,
        }
    } else if let Some(idx) = other_exception_index {
        HistoryCandidateStatus::NonTargetException { test_index: idx }
    } else if target_difference.is_some() {
        if target_exception_mismatch {
            HistoryCandidateStatus::TargetExceptionMismatch
        } else if target_exception_all {
            HistoryCandidateStatus::TargetExceptionAll
        } else {
            HistoryCandidateStatus::TargetHasDifference
        }
    } else {
        HistoryCandidateStatus::TargetNoDifference
    };

    // 每个被评估过的候选都会生成报告，即便它失败了也一样。
    // Every evaluated candidate gets a report, even failed ones.
    // 这样事后可以检查最小化过程，而不是把它变成黑盒。
    // This makes the minimization process inspectable after the fact instead of a black box.
    let cause_detail = match &status {
        HistoryCandidateStatus::TargetHasDifference => {
            "差异来源：历史测试指令与当前指令共同引发差异".to_string()
        }
        HistoryCandidateStatus::TargetExceptionMismatch => {
            "目标指令在不同实现上的异常触发情况不一致，停止最小化。".to_string()
        }
        HistoryCandidateStatus::TargetExceptionAll => {
            "目标指令在所有实现上均触发异常，停止最小化。".to_string()
        }
        HistoryCandidateStatus::TargetNoDifference => "目标指令不再出现写入差异。".to_string(),
        HistoryCandidateStatus::NonTargetDifference { test_index } => {
            let original_index = start_index + test_index;
            format!(
                "非目标指令 #{} 出现写入差异，已终止历史最小化。",
                original_index
            )
        }
        HistoryCandidateStatus::NonTargetException { test_index } => {
            let original_index = start_index + test_index;
            format!(
                "非目标指令 #{} 出现异常差异，已终止历史最小化。",
                original_index
            )
        }
    };

    let report_difference = match &status {
        HistoryCandidateStatus::TargetHasDifference
        | HistoryCandidateStatus::TargetExceptionMismatch
        | HistoryCandidateStatus::TargetExceptionAll => target_difference.as_ref(),
        HistoryCandidateStatus::NonTargetDifference { .. } => other_difference.as_ref(),
        _ => None,
    };

    let report_path = candidate_dir.join("report_01_history_candidate.md");
    write_history_candidate_report(
        &report_path,
        start_index,
        target_index,
        report_difference,
        &trimmed,
        impl_order,
        &cause_detail,
    )?;

    Ok(HistoryEvaluation {
        summary: HistoryEvaluationSummary {
            start_index,
            status,
            difference: target_difference,
        },
        testcase: trimmed,
        raw_outputs,
    })
}

fn build_history_trimmed_testcase(
    testcase: &GeneratedTestCase,
    impl_order: &[String],
    start_index: usize,
    target_index: usize,
) -> Result<GeneratedTestCase, DiffAnalysisError> {
    if start_index > target_index {
        return Err(DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "invalid history minimization range: start {} > target {}",
                start_index, target_index
            ),
        });
    }

    let total_len = determine_test_length(impl_order, testcase)?;
    if target_index >= total_len {
        return Err(DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "target index {} out of range for test length {}",
                target_index, total_len
            ),
        });
    }
    let remove = (0..total_len)
        .filter(|idx| *idx < start_index || *idx > target_index)
        .collect::<BTreeSet<_>>();
    Ok(testcase.without_test_indices(&remove))
}
