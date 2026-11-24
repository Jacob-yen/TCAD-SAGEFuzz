//! 单轮执行辅助逻辑。
//! Single-iteration execution helpers.
//!
//! runner 负责收敛和 testcase 裁剪；
//! The runner owns convergence and testcase trimming;
//! 这个模块只负责持久化单个 testcase、跨实现执行它，并产出该轮的工件。
//! this module only knows how to persist one testcase, execute it across implementations, and
//! emit the per-iteration artifacts.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use riscv_fuzz_plugin_api::{CoverageMode, ExecutionOutput};
use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;
use riscv_fuzz_toolchain::{ArtifactIoError, emit_raw_execution_artifacts, write_json_pretty};

use super::{
    streaming_execution::{
        StreamDivergencePoint, execute_per_impl_testcases_streaming_diff,
        execute_testcase_streaming_diff_detailed,
    },
    types::DiffAnalysisError,
    view::DiffExecutionView,
};

pub(super) struct IterationExecutionResult {
    pub(super) outputs: HashMap<String, DiffExecutionView>,
    pub(super) raw_outputs: HashMap<String, ExecutionOutput>,
    pub(super) first_divergence: Option<StreamDivergencePoint>,
    pub(super) timings: HashMap<String, Duration>,
}

pub(super) fn persist_iteration_testcase_at(
    testcase: &GeneratedTestCase,
    testcase_path: &Path,
) -> Result<(), DiffAnalysisError> {
    write_json_pretty(testcase_path, testcase).map_err(|err| match err {
        ArtifactIoError::Io(source) => DiffAnalysisError::Io(source),
        ArtifactIoError::Serialization(source) => DiffAnalysisError::Serialization(source),
    })?;
    Ok(())
}

pub(super) fn persist_iteration_testcase(
    testcase: &GeneratedTestCase,
    iteration_dir: &Path,
) -> Result<(), DiffAnalysisError> {
    persist_iteration_testcase_at(testcase, &iteration_dir.join("testcase.json"))
}

pub(super) fn persist_iteration_impl_artifacts(
    impl_order: &[String],
    iteration_dir: &Path,
    outputs: &HashMap<String, ExecutionOutput>,
    emit_execution_output_json: bool,
    emit_execution_report_md: bool,
) -> Result<(), DiffAnalysisError> {
    for impl_ref in impl_order {
        let output =
            outputs
                .get(impl_ref)
                .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                    core_id: impl_ref.to_string(),
                })?;
        let impl_dir = iteration_dir.join(impl_ref);

        emit_raw_execution_artifacts(
            output,
            &impl_dir,
            emit_execution_output_json,
            emit_execution_report_md,
            emit_execution_report_md,
        )
        .map_err(|err| DiffAnalysisError::Report(Box::new(err)))?;
    }

    Ok(())
}

pub(super) fn run_iteration(
    testcase: &GeneratedTestCase,
    impl_order: &[String],
    iteration_dir: &Path,
    coverage_mode: CoverageMode,
    emit_execution_output_json: bool,
    emit_execution_report_md: bool,
) -> Result<IterationExecutionResult, DiffAnalysisError> {
    run_iteration_detailed(
        testcase,
        impl_order,
        iteration_dir,
        coverage_mode,
        emit_execution_output_json,
        emit_execution_report_md,
    )
}

pub(super) fn run_iteration_detailed(
    testcase: &GeneratedTestCase,
    impl_order: &[String],
    iteration_dir: &Path,
    coverage_mode: CoverageMode,
    emit_execution_output_json: bool,
    emit_execution_report_md: bool,
) -> Result<IterationExecutionResult, DiffAnalysisError> {
    fs::create_dir_all(iteration_dir)?;
    persist_iteration_testcase(testcase, iteration_dir)?;
    let execution = execute_testcase_streaming_diff_detailed(
        testcase,
        impl_order,
        iteration_dir,
        coverage_mode,
    )?;
    persist_iteration_impl_artifacts(
        impl_order,
        iteration_dir,
        &execution.raw_outputs,
        emit_execution_output_json,
        emit_execution_report_md,
    )?;
    Ok(IterationExecutionResult {
        outputs: execution.outputs,
        raw_outputs: execution.raw_outputs,
        first_divergence: execution.first_divergence,
        timings: execution.timings,
    })
}

pub(super) fn run_iteration_with_per_impl_testcases(
    canonical_testcase: &GeneratedTestCase,
    per_impl_testcases: &HashMap<String, GeneratedTestCase>,
    impl_order: &[String],
    iteration_dir: &Path,
    coverage_mode: CoverageMode,
    emit_execution_output_json: bool,
    emit_execution_report_md: bool,
) -> Result<IterationExecutionResult, DiffAnalysisError> {
    fs::create_dir_all(iteration_dir)?;
    persist_iteration_testcase(canonical_testcase, iteration_dir)?;

    for impl_ref in impl_order {
        let testcase = per_impl_testcases.get(impl_ref).ok_or_else(|| {
            DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            }
        })?;
        let testcase_filename = format!("testcase_{impl_ref}.json");
        persist_iteration_testcase_at(testcase, &iteration_dir.join(testcase_filename))?;
    }

    let execution = execute_per_impl_testcases_streaming_diff(
        canonical_testcase,
        per_impl_testcases,
        impl_order,
        iteration_dir,
        coverage_mode,
    )?;

    for impl_ref in impl_order {
        let output = execution.raw_outputs.get(impl_ref).ok_or_else(|| {
            DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            }
        })?;
        emit_raw_execution_artifacts(
            output,
            iteration_dir.join(impl_ref),
            emit_execution_output_json,
            emit_execution_report_md,
            true,
        )
        .map_err(|err| DiffAnalysisError::Report(Box::new(err)))?;
    }

    Ok(IterationExecutionResult {
        outputs: execution.outputs,
        raw_outputs: execution.raw_outputs,
        first_divergence: execution.first_divergence,
        timings: execution.timings,
    })
}

pub(super) fn iteration_directory(run_root: &Path, iteration_index: usize) -> PathBuf {
    run_root.join(format!("iter_{:03}", iteration_index))
}

pub(super) fn iteration_summary_path(iteration_dir: &Path) -> PathBuf {
    iteration_dir.join("report_00_summary.md")
}

pub(super) fn reserve_report_slot(
    report_counters: &mut HashMap<usize, usize>,
    iteration_index: usize,
) {
    let counter = report_counters.entry(iteration_index).or_insert(1);
    *counter += 1;
}

pub(super) fn next_report_path(
    report_counters: &mut HashMap<usize, usize>,
    iteration_index: usize,
    iteration_dir: &Path,
    label: &str,
) -> PathBuf {
    let counter = report_counters.entry(iteration_index).or_insert(1);
    let filename = format!("report_{:02}_{}.md", counter, label);
    *counter += 1;
    iteration_dir.join(filename)
}

pub(super) fn cleanup_successful_iteration_outputs(
    iteration_dir: &Path,
    impl_order: &[String],
) -> std::io::Result<()> {
    for impl_ref in impl_order {
        let impl_dir = iteration_dir.join(impl_ref);
        match fs::remove_dir_all(&impl_dir) {
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
    }

    Ok(())
}
