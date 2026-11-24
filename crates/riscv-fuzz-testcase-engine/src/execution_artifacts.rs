use std::{collections::HashMap, fs, path::Path};

use riscv_fuzz_plugin_api::{CoverageMode, ExecutionOutput};
use riscv_fuzz_toolchain::{
    ArtifactIoError, ExecutionArtifactError, emit_raw_execution_artifacts, write_json_pretty,
};
use thiserror::Error;

use crate::{
    impls::{MulticoreSyncMetricError, compute_runtime_multicore_sync_metrics, execute},
    testcase::{GeneratedTestCase, MulticoreSyncMetrics, write_multicore_sync_metric_artifacts},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MulticoreSyncMetricFailureMode {
    Strict,
    BestEffort,
}

#[derive(Debug, Clone)]
pub struct ExecutionArtifactsRunResult {
    pub outputs: HashMap<String, ExecutionOutput>,
    pub multicore_sync_metrics: HashMap<String, MulticoreSyncMetrics>,
}

#[derive(Debug, Error)]
pub enum ExecutionArtifactsRunError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("execution failed: {0}")]
    Execution(#[source] Box<dyn std::error::Error>),
    #[error("missing instructions for implementation {core_id}")]
    MissingInstructions { core_id: String },
    #[error("artifact generation failed: {0}")]
    Artifact(#[source] Box<dyn std::error::Error>),
    #[error("multicore sync metric generation failed for {core_id}: {source}")]
    MulticoreSyncMetric {
        core_id: String,
        #[source]
        source: MulticoreSyncMetricError,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct ExecutionArtifactOptions<'a> {
    pub testcase_filename: &'a str,
    pub emit_execution_output_json: bool,
    pub emit_execution_report_md: bool,
    pub emit_execution_timeline_svg: bool,
    pub multicore_sync_metric_failure_mode: MulticoreSyncMetricFailureMode,
}

pub fn execute_testcase_with_artifacts(
    testcase: &GeneratedTestCase,
    core_ids: &[String],
    run_dir: &Path,
    coverage_mode: CoverageMode,
    options: ExecutionArtifactOptions<'_>,
) -> Result<ExecutionArtifactsRunResult, ExecutionArtifactsRunError> {
    fs::create_dir_all(run_dir)?;
    write_json_pretty(run_dir.join(options.testcase_filename), testcase)
        .map_err(map_artifact_io)?;

    let outputs =
        execute(testcase, run_dir, coverage_mode).map_err(ExecutionArtifactsRunError::Execution)?;
    let mut multicore_sync_metrics = HashMap::new();

    for core_id in core_ids {
        let output = outputs.get(core_id).ok_or_else(|| {
            ExecutionArtifactsRunError::MissingInstructions {
                core_id: core_id.clone(),
            }
        })?;
        let core_dir = run_dir.join(core_id);

        emit_raw_execution_artifacts(
            output,
            &core_dir,
            options.emit_execution_output_json,
            options.emit_execution_report_md,
            options.emit_execution_timeline_svg,
        )
        .map_err(map_execution_artifact_error)?;

        if testcase.config.is_dual_hart() {
            match compute_runtime_multicore_sync_metrics(testcase, core_id, output) {
                Ok(metrics) => {
                    write_multicore_sync_metric_artifacts(&core_dir, Some(&metrics))?;
                    multicore_sync_metrics.insert(core_id.clone(), metrics);
                }
                Err(source) => match options.multicore_sync_metric_failure_mode {
                    MulticoreSyncMetricFailureMode::Strict => {
                        return Err(ExecutionArtifactsRunError::MulticoreSyncMetric {
                            core_id: core_id.clone(),
                            source,
                        });
                    }
                    MulticoreSyncMetricFailureMode::BestEffort => {
                        eprintln!(
                            "skipping cycle-based multicore sync metrics for {} in {}: {}",
                            core_id,
                            run_dir.display(),
                            source
                        );
                    }
                },
            }
        }
    }

    Ok(ExecutionArtifactsRunResult {
        outputs,
        multicore_sync_metrics,
    })
}

fn map_artifact_io(error: ArtifactIoError) -> ExecutionArtifactsRunError {
    match error {
        ArtifactIoError::Io(source) => ExecutionArtifactsRunError::Io(source),
        ArtifactIoError::Serialization(source) => ExecutionArtifactsRunError::Serialization(source),
    }
}

fn map_execution_artifact_error(error: ExecutionArtifactError) -> ExecutionArtifactsRunError {
    match error {
        ExecutionArtifactError::Io(source) => map_artifact_io(source),
        ExecutionArtifactError::Report(source) => ExecutionArtifactsRunError::Artifact(source),
    }
}
