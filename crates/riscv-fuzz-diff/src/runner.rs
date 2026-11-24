//! diff-analysis 的顶层收敛循环。
//! Top-level convergence loop for diff-analysis.
//!
//! 这条流水线是严格模式：
//! The pipeline is strict:
//! - 先验证初始化写入。
//!   First, verify initialization writes.
//! - 持续移除触发异常差异的指令。
//!   Repeatedly remove instructions that trigger exception differences.
//! - 再定位第一个剩余的写差异。
//!   Then localize the first remaining write difference.
//! - 不允许弱轨迹或静默跳过；任何失败都必须向上传播为错误。
//!   The runner does not tolerate weak traces or silent skips; failures propagate as errors.

use std::{
    collections::{BTreeSet, HashMap},
    fs,
    io::ErrorKind,
    path::Path,
    time::Duration,
};

use riscv_fuzz_plugin_api::{ExecutionOutput, IsaExtensions};
use riscv_fuzz_plugin_manager::PluginManager;
use riscv_fuzz_testcase_engine::testcase::{GeneratedTestCase, HartPhase};

use super::{
    dedup::{
        ExceptionDedupKind, record_exception_difference, record_initialization_mismatch,
        record_write_difference,
    },
    detection::{
        detect_unique_exceptions, detect_unique_exceptions_at_location, detect_unique_writes,
        detect_unique_writes_at_location,
    },
    iteration::{
        cleanup_successful_iteration_outputs, iteration_directory, iteration_summary_path,
        next_report_path, reserve_report_slot, run_iteration_detailed,
    },
    localization::record_first_exception_localization,
    normalize::{PhaseInstructionLocation, phase_instruction_location_by_hart_local},
    reporting::{
        ExecutionSummaryInitialization, write_exception_cause_report, write_exception_report,
        write_execution_state_report, write_final_summary, write_write_report,
    },
    types::{DiffAnalysisConfig, DiffAnalysisError, DiffAnalysisResult},
    verification::{
        InitializationMismatchData, classify_initialization_mismatch, verify_initialization,
        verify_initialization_at_location,
    },
    view::{DiffExecutionView, build_diff_execution_view_with_weak_fallback},
    write_replay::{WriteDifferenceCause, classify_write_difference},
};

enum StreamingDivergencePhase {
    Init(PhaseInstructionLocation),
    Test(PhaseInstructionLocation),
}

fn streaming_divergence_phase(
    testcase: &GeneratedTestCase,
    divergence: Option<super::streaming_execution::StreamDivergencePoint>,
) -> Result<Option<StreamingDivergencePhase>, DiffAnalysisError> {
    let Some(divergence) = divergence else {
        return Ok(None);
    };
    if let Some(location) = phase_instruction_location_by_hart_local(
        testcase,
        HartPhase::Test,
        divergence.hart_id,
        divergence.local_index,
    ) {
        return Ok(Some(StreamingDivergencePhase::Test(location)));
    }
    if let Some(location) = phase_instruction_location_by_hart_local(
        testcase,
        HartPhase::Init,
        divergence.hart_id,
        divergence.local_index,
    ) {
        return Ok(Some(StreamingDivergencePhase::Init(location)));
    }
    Err(DiffAnalysisError::TestInstructionCountMismatch {
        details: format!(
            "streaming divergence points at missing testcase instruction hart {} local {}",
            divergence.hart_id, divergence.local_index
        ),
    })
}

fn validate_unaligned_requirement(impl_order: &[String], requirement: bool) {
    let plugin_manager = PluginManager::global();
    // diff-analysis 运行时只接受一个明确的非对齐访问要求。
    // Diff-analysis is configured with one explicit unaligned-access requirement.
    // 每个被选中的实现都必须满足它，否则这次运行在执行前就是无效的。
    // Each selected implementation must support that requirement or the run is invalid before
    // execution starts.
    for impl_ref in impl_order {
        assert!(
            plugin_manager
                .supported_unaligned_access_modes_with_overrides(impl_ref, &HashMap::new())
                .contains(&requirement),
            "implementation {impl_ref:?} does not satisfy the configured unaligned requirement",
        );
    }
}

fn initialization_mismatch_error(mismatch: &InitializationMismatchData) -> DiffAnalysisError {
    let marker = classify_initialization_mismatch(mismatch);
    DiffAnalysisError::InitializationMismatch {
        instruction_index: mismatch.instruction_index,
        details: format!(
            "[BUGCAT:{}][BUGFP:{}] {}",
            marker.tag, marker.fingerprint, mismatch.reason
        ),
    }
}

fn resolve_streaming_init_divergence(
    config: &DiffAnalysisConfig,
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
    timings: &HashMap<String, Duration>,
    summary_path: &Path,
    location: PhaseInstructionLocation,
) -> Result<bool, DiffAnalysisError> {
    match verify_initialization_at_location(impl_order, testcase, outputs, location) {
        Err(mismatch) => {
            let dedup_outcome = record_initialization_mismatch(
                &config.dedup,
                &config.run_root,
                summary_path,
                &mismatch,
                impl_order,
                testcase,
            )?;
            write_execution_state_report(
                summary_path,
                testcase,
                outputs,
                impl_order,
                Some(timings),
                Some(ExecutionSummaryInitialization::Failed {
                    mismatch: &mismatch,
                    dedup_outcome: dedup_outcome.as_ref(),
                }),
            )?;
            Err(initialization_mismatch_error(&mismatch))
        }
        Ok(()) => Ok(true),
    }
}

/// Verify that all selected implementations agree on the effects of the testcase initialization
/// prefix.
///
/// The returned value is the number of traceable initialization instructions that were checked.
/// Any mismatch is surfaced as a strict `InitializationMismatch` error with a stable bug marker.
pub fn verify_initialization_consistency(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, ExecutionOutput>,
) -> Result<usize, DiffAnalysisError> {
    let mut views = HashMap::new();
    for impl_ref in impl_order {
        let output =
            outputs
                .get(impl_ref)
                .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                    core_id: impl_ref.clone(),
                })?;
        let mem_range = testcase.mem_range.get(impl_ref).copied().ok_or_else(|| {
            DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            }
        })?;
        let per_hart_typed = testcase
            .combined_typed_sequences_per_hart_of(impl_ref)
            .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            })?;
        let view = build_diff_execution_view_with_weak_fallback(
            output.clone(),
            mem_range,
            &per_hart_typed,
        )
        .map_err(|err| DiffAnalysisError::Execution(Box::new(err)))?;
        views.insert(impl_ref.clone(), view);
    }

    match verify_initialization(impl_order, testcase, &views) {
        Ok(summary) => Ok(summary.init_instruction_count),
        Err(mismatch) => Err(initialization_mismatch_error(&mismatch)),
    }
}

#[cfg(all(test, feature = "internal-tests"))]
mod tests {
    #[test]
    #[cfg(feature = "plugin-tests")]
    fn resolve_support_map_reflects_requirement() {
        let impl_order = vec!["rocket".to_string(), "vex".to_string(), "spike".to_string()];

        super::validate_unaligned_requirement(&impl_order, false);
        super::validate_unaligned_requirement(&["spike".to_string()], true);
    }
}

fn apply_execution_extensions_override(
    testcase: &mut GeneratedTestCase,
    execution_extensions_override: Option<&IsaExtensions>,
) {
    if let Some(extensions) = execution_extensions_override {
        testcase.extensions = extensions.clone();
    }
}

fn finish_diff_analysis(
    config: &DiffAnalysisConfig,
    initial_testcase: &GeneratedTestCase,
    final_testcase: &GeneratedTestCase,
    final_outputs: &HashMap<String, ExecutionOutput>,
    impl_order: &[String],
    exception_round: usize,
    write_round: usize,
    cleanup_if_no_reduction: bool,
) -> Result<DiffAnalysisResult, DiffAnalysisError> {
    let final_summary_path = config.run_root.join("final_report.md");
    write_final_summary(
        &final_summary_path,
        initial_testcase,
        final_testcase,
        final_outputs,
        impl_order,
        exception_round,
        write_round,
    )?;

    let result = DiffAnalysisResult {
        initial_testcase: initial_testcase.clone(),
        final_testcase: final_testcase.clone(),
        final_outputs: final_outputs.clone(),
        exception_removal_rounds: exception_round,
        write_removal_rounds: write_round,
        output_root: config.run_root.clone(),
    };

    if cleanup_if_no_reduction && exception_round == 0 && write_round == 0 {
        match fs::remove_dir_all(&config.run_root) {
            Ok(_) => {}
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => return Err(DiffAnalysisError::Io(err)),
        }
    }

    Ok(result)
}

pub fn run_diff_analysis(
    core_ids: &[String],
    config: DiffAnalysisConfig,
) -> Result<DiffAnalysisResult, DiffAnalysisError> {
    let mut initial_testcase = match &config.generation_extensions_override {
        Some(extensions) => {
            riscv_fuzz_testcase_engine::impls::generate_random_testcase_with_extensions(
                core_ids,
                config.testcase_config.clone(),
                extensions.clone(),
            )?
        }
        None => riscv_fuzz_testcase_engine::impls::generate_random_testcase(
            core_ids,
            config.testcase_config.clone(),
        )?,
    };
    apply_execution_extensions_override(
        &mut initial_testcase,
        config.execution_extensions_override.as_ref(),
    );
    run_diff_analysis_with_testcase(core_ids, config, initial_testcase)
}

pub fn run_diff_analysis_with_testcase(
    core_ids: &[String],
    config: DiffAnalysisConfig,
    initial_testcase: GeneratedTestCase,
) -> Result<DiffAnalysisResult, DiffAnalysisError> {
    fs::create_dir_all(&config.run_root)?;

    let impl_order: Vec<String> = core_ids.to_vec();
    if impl_order.is_empty() {
        return Err(DiffAnalysisError::TestInstructionCountMismatch {
            details: "no RISC-V implementations available".to_string(),
        });
    }

    let unaligned_required = config
        .testcase_config
        .unaligned_access_required
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "testcase_config.unaligned_access_required must be explicitly set",
            )
        })?;

    validate_unaligned_requirement(&impl_order, unaligned_required);

    let mut testcase = initial_testcase.clone();

    let mut iteration_index = 0usize;
    let mut report_counters: HashMap<usize, usize> = HashMap::new();
    let emit_json = config.emit_execution_output_json;
    let emit_markdown = config.emit_execution_report_md;

    let mut iteration_dir = iteration_directory(&config.run_root, iteration_index);
    let mut iteration = run_iteration_detailed(
        &testcase,
        &impl_order,
        &iteration_dir,
        config.coverage_mode,
        emit_json,
        emit_markdown,
    )?;
    let mut outputs = iteration.outputs;
    let mut raw_outputs = iteration.raw_outputs;
    let mut streaming_divergence =
        streaming_divergence_phase(&testcase, iteration.first_divergence)?;
    let summary_path = iteration_summary_path(&iteration_dir);

    match streaming_divergence {
        Some(StreamingDivergencePhase::Init(location)) => {
            if resolve_streaming_init_divergence(
                &config,
                &impl_order,
                &testcase,
                &outputs,
                &iteration.timings,
                &summary_path,
                location,
            )? {
                streaming_divergence = None;
            }
        }
        Some(StreamingDivergencePhase::Test(_)) | None => {}
    }

    let init_summary = match verify_initialization(&impl_order, &testcase, &outputs) {
        Ok(summary) => summary,
        Err(mismatch) => {
            let dedup_outcome = record_initialization_mismatch(
                &config.dedup,
                &config.run_root,
                &summary_path,
                &mismatch,
                &impl_order,
                &testcase,
            )?;
            write_execution_state_report(
                &summary_path,
                &testcase,
                &outputs,
                &impl_order,
                Some(&iteration.timings),
                Some(ExecutionSummaryInitialization::Failed {
                    mismatch: &mismatch,
                    dedup_outcome: dedup_outcome.as_ref(),
                }),
            )?;
            return Err(initialization_mismatch_error(&mismatch));
        }
    };
    reserve_report_slot(&mut report_counters, iteration_index);
    write_execution_state_report(
        &summary_path,
        &testcase,
        &outputs,
        &impl_order,
        Some(&iteration.timings),
        Some(ExecutionSummaryInitialization::Verified(&init_summary)),
    )?;

    let mut exception_round = 0usize;
    let mut write_round = 0usize;

    loop {
        // 异常触发不一致优先于写差异分析，
        // Exception-trigger disagreement takes priority over write analysis,
        // 因为在 trap 行为已经分叉之后看到的写差异并不是稳定信号。
        // because a write diff observed after divergent trap behavior is not a stable signal.
        let exception_detections =
            if let Some(StreamingDivergencePhase::Test(location)) = streaming_divergence {
                detect_unique_exceptions_at_location(&impl_order, &testcase, &outputs, location)?
            } else {
                detect_unique_exceptions(&impl_order, &testcase, &outputs)?
            };
        if !exception_detections.trigger_mismatches.is_empty() {
            let removal_indices: BTreeSet<usize> = exception_detections
                .trigger_mismatches
                .iter()
                .map(|diff| diff.test_index)
                .collect();

            let report_path = next_report_path(
                &mut report_counters,
                iteration_index,
                &iteration_dir,
                if config.enable_reduction {
                    "exception_removal"
                } else {
                    "exception_diff"
                },
            );
            let reported_differences = exception_detections
                .trigger_mismatches
                .iter()
                .map(|diff| {
                    record_exception_difference(
                        &config.dedup,
                        &config.run_root,
                        &report_path,
                        diff,
                        &impl_order,
                        &testcase,
                        ExceptionDedupKind::Trigger,
                    )
                    .map(|outcome| (diff.clone(), outcome))
                })
                .collect::<Result<Vec<_>, DiffAnalysisError>>()?;
            if !config.enable_reduction {
                write_exception_report(
                    &report_path,
                    &testcase,
                    &testcase,
                    &reported_differences,
                    &impl_order,
                )?;
                return finish_diff_analysis(
                    &config,
                    &initial_testcase,
                    &testcase,
                    &raw_outputs,
                    &impl_order,
                    exception_round,
                    write_round,
                    false,
                );
            }
            exception_round += 1;
            let trimmed_testcase = testcase.without_test_indices(&removal_indices);
            write_exception_report(
                &report_path,
                &testcase,
                &trimmed_testcase,
                &reported_differences,
                &impl_order,
            )?;
            testcase = trimmed_testcase;

            if let Some(limit) = config.max_iterations
                && exception_round + write_round >= limit
            {
                return Err(DiffAnalysisError::IterationLimitReached(limit));
            }
            iteration_index += 1;
            iteration_dir = iteration_directory(&config.run_root, iteration_index);
            iteration = run_iteration_detailed(
                &testcase,
                &impl_order,
                &iteration_dir,
                config.coverage_mode,
                emit_json,
                emit_markdown,
            )?;
            outputs = iteration.outputs;
            raw_outputs = iteration.raw_outputs;
            streaming_divergence =
                streaming_divergence_phase(&testcase, iteration.first_divergence)?;
            let summary_path = iteration_summary_path(&iteration_dir);
            if let Some(StreamingDivergencePhase::Init(location)) = streaming_divergence {
                if resolve_streaming_init_divergence(
                    &config,
                    &impl_order,
                    &testcase,
                    &outputs,
                    &iteration.timings,
                    &summary_path,
                    location,
                )? {
                    streaming_divergence = None;
                }
            }

            let init_summary = match verify_initialization(&impl_order, &testcase, &outputs) {
                Ok(summary) => summary,
                Err(mismatch) => {
                    let dedup_outcome = record_initialization_mismatch(
                        &config.dedup,
                        &config.run_root,
                        &summary_path,
                        &mismatch,
                        &impl_order,
                        &testcase,
                    )?;
                    write_execution_state_report(
                        &summary_path,
                        &testcase,
                        &outputs,
                        &impl_order,
                        Some(&iteration.timings),
                        Some(ExecutionSummaryInitialization::Failed {
                            mismatch: &mismatch,
                            dedup_outcome: dedup_outcome.as_ref(),
                        }),
                    )?;
                    return Err(initialization_mismatch_error(&mismatch));
                }
            };
            write_execution_state_report(
                &summary_path,
                &testcase,
                &outputs,
                &impl_order,
                Some(&iteration.timings),
                Some(ExecutionSummaryInitialization::Verified(&init_summary)),
            )?;

            if testcase.test_traceable_len() == 0 {
                break;
            }

            record_first_exception_localization();
            continue;
        } else if !exception_detections.cause_mismatches.is_empty() {
            // 仅异常原因不一致的情况会保留在报告里，
            // Cause-only mismatches are kept for reporting,
            // 但不会改变 testcase 结构，因为所有实现仍然一致认为该指令会触发异常。
            // but they do not change testcase structure because all implementations still agree
            // that the instruction traps.
            let report_path = next_report_path(
                &mut report_counters,
                iteration_index,
                &iteration_dir,
                "exception_cause_diff",
            );
            let reported_differences = exception_detections
                .cause_mismatches
                .iter()
                .map(|diff| {
                    record_exception_difference(
                        &config.dedup,
                        &config.run_root,
                        &report_path,
                        diff,
                        &impl_order,
                        &testcase,
                        ExceptionDedupKind::Cause,
                    )
                    .map(|outcome| (diff.clone(), outcome))
                })
                .collect::<Result<Vec<_>, DiffAnalysisError>>()?;
            write_exception_cause_report(
                &report_path,
                &testcase,
                &reported_differences,
                &impl_order,
            )?;
        }

        let write_differences =
            if let Some(StreamingDivergencePhase::Test(location)) = streaming_divergence {
                detect_unique_writes_at_location(&impl_order, &testcase, &outputs, location)?
            } else {
                detect_unique_writes(&impl_order, &testcase, &outputs)?
            };
        if matches!(
            streaming_divergence,
            Some(StreamingDivergencePhase::Test(_))
        ) && exception_detections.trigger_mismatches.is_empty()
            && exception_detections.cause_mismatches.is_empty()
            && write_differences.is_empty()
        {
            return Err(DiffAnalysisError::TestInstructionCountMismatch {
                details: "streaming compare reported an early test divergence, but no exception or write mismatch could be reconstructed at that point".to_string(),
            });
        }
        if write_differences.is_empty() {
            if config.cleanup_successful_iteration_artifacts {
                cleanup_successful_iteration_outputs(&iteration_dir, &impl_order)
                    .map_err(DiffAnalysisError::Io)?;
            }
            break;
        }

        let first_difference = write_differences
            .into_iter()
            .next()
            .expect("write_differences is non-empty; qed");

        // 一次只定位一个写差异，
        // Localize one write difference at a time,
        // 这样 replay 工件可以保持聚焦，归约循环也更容易维持确定性。
        // so replay artifacts remain focused and the reduction loop stays deterministic.
        if !config.enable_reduction {
            let report_path = next_report_path(
                &mut report_counters,
                iteration_index,
                &iteration_dir,
                "write_diff",
            );
            let dedup_outcome = record_write_difference(
                &config.dedup,
                &config.run_root,
                &report_path,
                &first_difference,
                &impl_order,
                &testcase,
            )?;
            let diff_report = vec![(
                first_difference.clone(),
                "差异来源：已关闭测试用例约减；保留原始 testcase，不执行 replay/history minimization/probe。"
                    .to_string(),
                dedup_outcome,
            )];
            write_write_report(
                &report_path,
                &testcase,
                &testcase,
                &diff_report,
                &impl_order,
            )?;
            return finish_diff_analysis(
                &config,
                &initial_testcase,
                &testcase,
                &raw_outputs,
                &impl_order,
                exception_round,
                write_round,
                false,
            );
        }

        let (cause, probe_outcome) = classify_write_difference(
            &first_difference,
            &testcase,
            &outputs,
            &impl_order,
            &iteration_dir,
            write_round + 1,
            config.history_min_test_threshold,
            config.coverage_mode,
            emit_json,
            emit_markdown,
            config.cleanup_successful_iteration_artifacts,
            config.verify_write_with_probes,
            config.write_probe_max_registers,
            config.write_probe_max_memory_bytes,
        )?;
        let mut cause_description = format!("差异来源：{}", cause);
        if let Some(probe) = &probe_outcome {
            cause_description.push_str("；探针验证：");
            cause_description.push_str(&probe.summary());
        }
        let mut removal_indices = BTreeSet::new();
        let mut consumes_iteration_budget = false;
        match cause {
            WriteDifferenceCause::SingleInstruction => {
                removal_indices.insert(first_difference.test_index);
                write_round += 1;
                consumes_iteration_budget = true;
            }
            WriteDifferenceCause::DependsOnHistory => {
                // 如果目标指令只有在带着某段前缀历史时才会分叉，
                // If the target instruction only diverges with some prefix present,
                // 就一次性把这段前缀连同目标指令一起移除，直接消掉依赖链。
                // remove the whole prefix through the target to eliminate the dependency chain
                // in one step.
                for idx in 0..=first_difference.test_index {
                    removal_indices.insert(idx);
                }
                write_round += 1;
                consumes_iteration_budget = true;
            }
            WriteDifferenceCause::LikelyFalsePositive => {
                // replay/history-min 都无法保留该差异时，说明这更像日志/解析噪声。
                // 仍然删掉目标指令让当前 run 前进，但不要把它记成真实 write 差异轮次。
                removal_indices.insert(first_difference.test_index);
            }
        }

        let report_path = next_report_path(
            &mut report_counters,
            iteration_index,
            &iteration_dir,
            "write_diff",
        );
        let dedup_outcome = record_write_difference(
            &config.dedup,
            &config.run_root,
            &report_path,
            &first_difference,
            &impl_order,
            &testcase,
        )?;
        let diff_report = vec![(first_difference.clone(), cause_description, dedup_outcome)];

        let trimmed_testcase = testcase.without_test_indices(&removal_indices);
        write_write_report(
            &report_path,
            &testcase,
            &trimmed_testcase,
            &diff_report,
            &impl_order,
        )?;
        testcase = trimmed_testcase;

        if let Some(limit) = config.max_iterations
            && consumes_iteration_budget
            && exception_round + write_round >= limit
        {
            return Err(DiffAnalysisError::IterationLimitReached(limit));
        }
        iteration_index += 1;
        iteration_dir = iteration_directory(&config.run_root, iteration_index);
        iteration = run_iteration_detailed(
            &testcase,
            &impl_order,
            &iteration_dir,
            config.coverage_mode,
            emit_json,
            emit_markdown,
        )?;
        outputs = iteration.outputs;
        raw_outputs = iteration.raw_outputs;
        streaming_divergence = streaming_divergence_phase(&testcase, iteration.first_divergence)?;
        let summary_path = iteration_summary_path(&iteration_dir);
        if let Some(StreamingDivergencePhase::Init(location)) = streaming_divergence {
            if resolve_streaming_init_divergence(
                &config,
                &impl_order,
                &testcase,
                &outputs,
                &iteration.timings,
                &summary_path,
                location,
            )? {
                streaming_divergence = None;
            }
        }

        let init_summary = match verify_initialization(&impl_order, &testcase, &outputs) {
            Ok(summary) => summary,
            Err(mismatch) => {
                let dedup_outcome = record_initialization_mismatch(
                    &config.dedup,
                    &config.run_root,
                    &summary_path,
                    &mismatch,
                    &impl_order,
                    &testcase,
                )?;
                write_execution_state_report(
                    &summary_path,
                    &testcase,
                    &outputs,
                    &impl_order,
                    Some(&iteration.timings),
                    Some(ExecutionSummaryInitialization::Failed {
                        mismatch: &mismatch,
                        dedup_outcome: dedup_outcome.as_ref(),
                    }),
                )?;
                return Err(initialization_mismatch_error(&mismatch));
            }
        };
        write_execution_state_report(
            &summary_path,
            &testcase,
            &outputs,
            &impl_order,
            Some(&iteration.timings),
            Some(ExecutionSummaryInitialization::Verified(&init_summary)),
        )?;

        if testcase.test_traceable_len() == 0 {
            break;
        }
    }

    finish_diff_analysis(
        &config,
        &initial_testcase,
        &testcase,
        &raw_outputs,
        &impl_order,
        exception_round,
        write_round,
        config.cleanup_successful_diff_run,
    )
}

#[cfg(test)]
mod override_tests {
    use std::collections::HashMap;

    use riscv_fuzz_plugin_api::IsaExtensions;
    use riscv_fuzz_testcase_engine::testcase::{GeneratedTestCase, TestCaseConfig};
    use riscv_instruction::RV64Extensions;

    use super::apply_execution_extensions_override;

    #[test]
    fn apply_execution_extensions_override_replaces_testcase_extensions() {
        let original = IsaExtensions::rv64(vec![RV64Extensions::I, RV64Extensions::Zicsr]);
        let override_extensions = IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::Zaamo,
            RV64Extensions::Zalrsc,
            RV64Extensions::Zicsr,
        ]);
        let mut testcase = GeneratedTestCase {
            config: TestCaseConfig::default(),
            harts: Vec::new(),
            mem_range: HashMap::new(),
            extensions: original,
            multicore_sync_metrics: None,
        };

        apply_execution_extensions_override(&mut testcase, Some(&override_extensions));

        assert_eq!(testcase.extensions, override_extensions);
    }

    #[test]
    fn apply_execution_extensions_override_keeps_original_when_absent() {
        let original = IsaExtensions::rv64(vec![RV64Extensions::I, RV64Extensions::Zicsr]);
        let mut testcase = GeneratedTestCase {
            config: TestCaseConfig::default(),
            harts: Vec::new(),
            mem_range: HashMap::new(),
            extensions: original.clone(),
            multicore_sync_metrics: None,
        };

        apply_execution_extensions_override(&mut testcase, None);

        assert_eq!(testcase.extensions, original);
    }
}
