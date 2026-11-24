use std::{
    env,
    fmt::Write as _,
    fs, io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use log::{error, info, warn};
use riscv_fuzz_config::{ExperimentConfig, RunnerMode};
use riscv_fuzz_diff::{
    DiffAnalysisError, DiffAnalysisSettings, DiffDedupSettings, run_diff_analysis,
    run_diff_analysis_with_testcase,
};
use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
use riscv_fuzz_plugin_manager::PluginManager;
use riscv_fuzz_testcase_engine::{GenerationError, testcase::GeneratedTestCase};
use riscv_fuzz_toolchain::write_string_file;
use riscv_instruction::{RV32Extensions, RV64Extensions};
use serde_json::from_reader;

use super::{
    CliError, Result, ResultExt, diff_settings_from_experiment, format_rejected_cores,
    select_compatible_cores,
};

macro_rules! bail {
    ($($arg:tt)*) => {
        return Err(CliError::new(format!($($arg)*)));
    };
}

const SPIKE_CORE_ID: &str = "spike";
const CORE_ENV_PREFIX: &str = "CX_RISCV_CORES_";

fn strip_unstable_single_hart_generation_extensions(extensions: &mut IsaExtensions) -> bool {
    match extensions {
        IsaExtensions::Rv32(values) => {
            let before = values.len();
            values.retain(|ext| {
                *ext != RV32Extensions::B
                    && *ext != RV32Extensions::Zaamo
                    && *ext != RV32Extensions::Zabha
                    && *ext != RV32Extensions::Zacas
                    && *ext != RV32Extensions::Zalrsc
                    && *ext != RV32Extensions::Zawrs
                    && *ext != RV32Extensions::Zimop
                    && *ext != RV32Extensions::Zcmop
                    && *ext != RV32Extensions::Zcb
                    && *ext != RV32Extensions::Zba
                    && *ext != RV32Extensions::Zbb
                    && *ext != RV32Extensions::Zbc
                    && *ext != RV32Extensions::Zbkb
                    && *ext != RV32Extensions::Zbkx
                    && *ext != RV32Extensions::Zbs
                    && *ext != RV32Extensions::Zknd
                    && *ext != RV32Extensions::Zkne
                    && *ext != RV32Extensions::Zknh
                    && *ext != RV32Extensions::Zksed
                    && *ext != RV32Extensions::Zksh
            });
            values.len() != before
        }
        IsaExtensions::Rv64(values) => {
            let before = values.len();
            values.retain(|ext| {
                *ext != RV64Extensions::B
                    && *ext != RV64Extensions::Zaamo
                    && *ext != RV64Extensions::Zabha
                    && *ext != RV64Extensions::Zacas
                    && *ext != RV64Extensions::Zalrsc
                    && *ext != RV64Extensions::Zawrs
                    && *ext != RV64Extensions::Zimop
                    && *ext != RV64Extensions::Zcmop
                    && *ext != RV64Extensions::Zcb
                    && *ext != RV64Extensions::Zba
                    && *ext != RV64Extensions::Zbb
                    && *ext != RV64Extensions::Zbc
                    && *ext != RV64Extensions::Zbkb
                    && *ext != RV64Extensions::Zbkx
                    && *ext != RV64Extensions::Zbs
                    && *ext != RV64Extensions::Zkn
                    && *ext != RV64Extensions::Zknd
                    && *ext != RV64Extensions::Zkne
                    && *ext != RV64Extensions::Zknh
                    && *ext != RV64Extensions::Zksed
                    && *ext != RV64Extensions::Zksh
            });
            values.len() != before
        }
    }
}

fn spike_pair_target(core_ids: &[String]) -> Option<&str> {
    if core_ids.len() != 2 {
        return None;
    }
    if core_ids[0] == SPIKE_CORE_ID && core_ids[1] != SPIKE_CORE_ID {
        return Some(core_ids[1].as_str());
    }
    if core_ids[1] == SPIKE_CORE_ID && core_ids[0] != SPIKE_CORE_ID {
        return Some(core_ids[0].as_str());
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StaleCoreArtifactWarning {
    env_var: String,
    configured_path: PathBuf,
    local_path: PathBuf,
}

fn workspace_local_core_artifact_dir() -> Option<PathBuf> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent()?.parent()?;
    let local_artifact_dir = workspace_root
        .parent()?
        .join("cx-riscv-cores")
        .join("artifacts");
    local_artifact_dir.is_dir().then_some(local_artifact_dir)
}

fn stale_core_artifact_warnings_for_bindings<I>(
    local_artifact_dir: &Path,
    bindings: I,
) -> io::Result<Vec<StaleCoreArtifactWarning>>
where
    I: IntoIterator<Item = (String, PathBuf)>,
{
    let mut warnings = Vec::new();
    for (env_var, configured_path) in bindings {
        let Some(file_name) = configured_path.file_name() else {
            continue;
        };
        let local_path = local_artifact_dir.join(file_name);
        if !configured_path.is_file() || !local_path.is_file() {
            continue;
        }

        let same_path = match (
            fs::canonicalize(&configured_path),
            fs::canonicalize(&local_path),
        ) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        if same_path {
            continue;
        }

        let configured_modified = fs::metadata(&configured_path)?.modified()?;
        let local_modified = fs::metadata(&local_path)?.modified()?;
        if local_modified > configured_modified {
            warnings.push(StaleCoreArtifactWarning {
                env_var,
                configured_path,
                local_path,
            });
        }
    }

    warnings.sort_by(|lhs, rhs| lhs.env_var.cmp(&rhs.env_var));
    Ok(warnings)
}

fn warn_if_stale_core_artifacts_configured() {
    let Some(local_artifact_dir) = workspace_local_core_artifact_dir() else {
        return;
    };

    let bindings = env::vars_os().filter_map(|(key, value)| {
        let key = key.into_string().ok()?;
        if !key.starts_with(CORE_ENV_PREFIX) {
            return None;
        }
        if matches!(
            key.as_str(),
            "CX_RISCV_CORES_ROOT" | "CX_RISCV_CORES_ARTIFACT_DIR"
        ) {
            return None;
        }
        Some((key, PathBuf::from(value)))
    });

    let warnings = match stale_core_artifact_warnings_for_bindings(&local_artifact_dir, bindings) {
        Ok(warnings) => warnings,
        Err(err) => {
            warn!(
                "Failed to compare configured core artifacts with local wrappers under {}: {}",
                local_artifact_dir.display(),
                err
            );
            return;
        }
    };

    if warnings.is_empty() {
        return;
    }

    warn!(
        "Detected {} configured core wrapper binaries older than same-name files under {}. Stale wrappers can cause diff false positives.",
        warnings.len(),
        local_artifact_dir.display()
    );
    for warning in warnings.iter().take(8) {
        warn!(
            "{} -> {} | newer local wrapper: {}",
            warning.env_var,
            warning.configured_path.display(),
            warning.local_path.display()
        );
    }
    if warnings.len() > 8 {
        warn!(
            "... and {} more stale wrapper bindings. Refresh the core env vars before trusting diff results.",
            warnings.len() - 8
        );
    } else {
        warn!("Refresh the core env vars before trusting diff results.");
    }
}

/// Random-diff runner (validator R9 ensures `[runner.random_diff]` exists).
pub(super) fn run_random_diff_unified(cfg: &ExperimentConfig, run_dir: PathBuf) -> Result<()> {
    debug_assert!(matches!(cfg.runner.mode, RunnerMode::RandomDiff));
    warn_if_stale_core_artifacts_configured();

    let plugin_manager = PluginManager::global();
    let explicit_selection = !cfg.cores.selected.is_empty();
    let base_cores = if explicit_selection {
        PluginManager::normalize_impls(cfg.cores.selected.clone())
    } else {
        plugin_manager.supported_impls_for_isa(cfg.testcase_config.isa_base)
    };

    let mut settings = diff_settings_from_experiment(cfg);
    let selected = select_compatible_cores(
        plugin_manager,
        &base_cores,
        &settings.testcase_config,
        settings.coverage_mode,
    );

    if explicit_selection && !selected.rejected_cores.is_empty() {
        bail!(
            "selected cores are incompatible with workload: {}",
            format_rejected_cores(&selected.rejected_cores)
        );
    }
    for (core_ref, reason) in &selected.rejected_cores {
        warn!("Skipping {core_ref}: {reason}");
    }
    if selected.compatible_cores.is_empty() {
        bail!(
            "no cores satisfy workload (hart_count={}, unaligned_access_required={:?}, coverage={:?})",
            settings.testcase_config.hart_count(),
            settings.testcase_config.unaligned_access_required,
            settings.coverage_mode
        );
    }
    if explicit_selection && let Some(target) = spike_pair_target(&selected.compatible_cores) {
        let execution_extensions = riscv_fuzz_testcase_engine::impls::extensions_for_config(
            &selected.compatible_cores,
            &settings.testcase_config,
        )
        .map_err(|err| {
            CliError::new(format!(
                "workload configuration invalid for Spike vs {target}: {err}"
            ))
        })?;
        let mut pair_extensions = execution_extensions.clone();
        if strip_unstable_single_hart_generation_extensions(&mut pair_extensions) {
            warn!(
                "Spike vs {target} pairwise diff disables aggregate B/bitmanip/crypto plus AMO/LRSC/CAS/WRS and MayBeOp extensions during testcase generation/execution so this run stays assembler-compatible and focused on stable scalar/vector signals."
            );
            settings.generation_extensions_override = Some(pair_extensions.clone());
            settings.execution_extensions_override = Some(pair_extensions);
        }
    }

    let max_runs = cfg
        .runner
        .random_diff
        .as_ref()
        .map(|c| c.max_iterations)
        .filter(|n| *n > 0);
    run_diff_with_cores(
        selected.compatible_cores,
        settings,
        cfg.cores.concurrency,
        run_dir,
        max_runs,
        "random_diff",
    )
}

/// Diff-against-Spike runner (R9 ensures the subtable exists).
pub(super) fn run_diff_against_spike_unified(
    cfg: &ExperimentConfig,
    run_dir: PathBuf,
) -> Result<()> {
    debug_assert!(matches!(cfg.runner.mode, RunnerMode::DiffAgainstSpike));
    warn_if_stale_core_artifacts_configured();
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create base run directory {}", run_dir.display()))?;

    let plugin_manager = PluginManager::global();
    let configured_isa = cfg.testcase_config.isa_base;

    let candidates: Vec<String> = if cfg.cores.selected.is_empty() {
        plugin_manager.registered_impls()
    } else {
        PluginManager::normalize_impls(cfg.cores.selected.clone())
    };
    let targets: Vec<String> = candidates
        .into_iter()
        .filter(|c| c != SPIKE_CORE_ID)
        .collect();
    if targets.is_empty() {
        bail!("diff_against_spike requires at least one non-Spike core in cores.selected");
    }

    let settings = diff_settings_from_experiment(cfg);
    let max_runs = cfg
        .runner
        .diff_against_spike
        .as_ref()
        .map(|c| c.max_iterations)
        .filter(|n| *n > 0);

    let mut executed_pairs = 0usize;
    for target in targets {
        let mut pair_settings = DiffAnalysisSettings {
            testcase_config: settings.testcase_config.clone(),
            ..settings.clone()
        };
        let pair_run_dir = run_dir.join(format!(
            "spike_{}_{}",
            target.to_ascii_lowercase(),
            configured_isa.as_ref()
        ));
        let pair_selection = select_compatible_cores(
            plugin_manager,
            &PluginManager::normalize_impls(vec![SPIKE_CORE_ID.to_string(), target.clone()]),
            &pair_settings.testcase_config,
            pair_settings.coverage_mode,
        );
        if !pair_selection.rejected_cores.is_empty() {
            warn!(
                "Skipping Spike vs {target}: {}",
                format_rejected_cores(&pair_selection.rejected_cores)
            );
            continue;
        }
        if !pair_settings.testcase_config.is_dual_hart() {
            let execution_extensions = riscv_fuzz_testcase_engine::impls::extensions_for_config(
                &pair_selection.compatible_cores,
                &pair_settings.testcase_config,
            )
            .map_err(|err| {
                CliError::new(format!(
                    "workload configuration invalid for Spike vs {target}: {err}"
                ))
            })?;
            let mut pair_extensions = execution_extensions.clone();
            if strip_unstable_single_hart_generation_extensions(&mut pair_extensions) {
                warn!(
                    "Spike vs {target} single-hart diff disables aggregate B/bitmanip/crypto plus AMO/LRSC/CAS/WRS and MayBeOp extensions during testcase generation/execution so this run stays assembler-compatible and focused on stable scalar/vector signals."
                );
            }
            pair_settings.generation_extensions_override = Some(pair_extensions.clone());
            pair_settings.execution_extensions_override = Some(pair_extensions);
        }
        let label = format!("Spike vs {target} ({})", configured_isa);
        run_diff_with_cores(
            pair_selection.compatible_cores,
            pair_settings,
            cfg.cores.concurrency,
            pair_run_dir.clone(),
            max_runs,
            &label,
        )
        .with_context(|| {
            format!(
                "diff analysis failed for Spike vs {target} ({}) in {}",
                configured_isa,
                pair_run_dir.display()
            )
        })?;
        executed_pairs += 1;
    }
    if executed_pairs == 0 {
        bail!(
            "no diff_against_spike pairings ran for ISA {configured_isa}; check core compatibility"
        );
    }
    info!(
        "Completed {executed_pairs} Spike diff pairings; artifacts under {}",
        run_dir.display()
    );
    Ok(())
}

/// Replay runner (R9 ensures `[runner.replay]` exists). Loads the testcase
/// JSON, then drives the diff pipeline against whichever cores it references.
pub(super) fn run_replay_unified(cfg: &ExperimentConfig, run_dir: PathBuf) -> Result<()> {
    debug_assert!(matches!(cfg.runner.mode, RunnerMode::Replay));
    warn_if_stale_core_artifacts_configured();
    let replay = cfg
        .runner
        .replay
        .as_ref()
        .expect("validator R9 ensures replay subtable is present");

    let testcase_path = replay.testcase_path.clone();
    let file = fs::File::open(&testcase_path).with_context(|| {
        format!(
            "failed to open testcase definition {}",
            testcase_path.display()
        )
    })?;
    let testcase: GeneratedTestCase = from_reader(file)
        .with_context(|| format!("failed to parse testcase {}", testcase_path.display()))?;

    if testcase.config.unaligned_access_required.is_none() {
        bail!(
            "testcase {} must include testcase_config.unaligned_access_required",
            testcase_path.display()
        );
    }
    let testcase_hart_count = testcase.config.hart_count();
    if !(1..=2).contains(&testcase_hart_count) {
        bail!(
            "testcase {} must include testcase_config.hart_count = 1 or 2 (found {})",
            testcase_path.display(),
            testcase_hart_count
        );
    }

    let cores = testcase.impl_refs();
    if cores.is_empty() {
        bail!(
            "testcase {} does not reference any cores",
            testcase_path.display()
        );
    }

    let plugin_manager = PluginManager::global();
    let core_ids = PluginManager::normalize_impls(cores);
    let experiment_testcase_config = &cfg.testcase_config;
    let replay_testcase_config = &testcase.config;
    if experiment_testcase_config.isa_base != replay_testcase_config.isa_base {
        bail!(
            "replay testcase_config.isa_base mismatch: experiment has {}, testcase has {}",
            experiment_testcase_config.isa_base,
            replay_testcase_config.isa_base
        );
    }
    if experiment_testcase_config.hart_config != replay_testcase_config.hart_config {
        bail!(
            "replay testcase_config.hart_config mismatch between experiment config and testcase {}",
            testcase_path.display()
        );
    }
    if experiment_testcase_config.unaligned_access_required
        != replay_testcase_config.unaligned_access_required
    {
        bail!(
            "replay testcase_config.unaligned_access_required mismatch: experiment has {:?}, testcase has {:?}",
            experiment_testcase_config.unaligned_access_required,
            replay_testcase_config.unaligned_access_required
        );
    }
    if experiment_testcase_config.float_extension != replay_testcase_config.float_extension {
        bail!(
            "replay testcase_config.float_extension mismatch: experiment has {:?}, testcase has {:?}",
            experiment_testcase_config.float_extension,
            replay_testcase_config.float_extension
        );
    }

    // Reporting & probe knobs come from the experiment config so the caller
    // can toggle them per-replay without editing the testcase.
    let replay_settings = DiffAnalysisSettings {
        testcase_config: testcase.config.clone(),
        coverage_mode: cfg.reporting.coverage_collection,
        execution_timeout_seconds: cfg.cores.execution_timeout_seconds,
        max_iterations: None,
        enable_reduction: true,
        history_min_test_threshold: None,
        emit_execution_output_json: cfg.reporting.emit_execution_json,
        emit_execution_report_md: cfg.reporting.emit_execution_markdown,
        cleanup_successful_iteration_artifacts: cfg.reporting.delete_successful_iteration_artifacts,
        cleanup_successful_diff_run: cfg.reporting.delete_successful_run_artifacts,
        verify_write_with_probes: cfg.reporting.write_probe_verification.enabled,
        write_probe_max_registers: cfg.reporting.write_probe_verification.max_registers,
        write_probe_max_memory_bytes: cfg.reporting.write_probe_verification.max_memory_bytes,
        dedup: DiffDedupSettings {
            enabled: cfg.reporting.dedup.enabled,
            database_path: cfg.reporting.dedup.database_path.clone(),
        },
        generation_extensions_override: None,
        execution_extensions_override: None,
    };
    let diff_config = replay_settings.with_run_root(run_dir.clone());

    let selected = select_compatible_cores(
        plugin_manager,
        &core_ids,
        &diff_config.testcase_config,
        diff_config.coverage_mode,
    );
    if !selected.rejected_cores.is_empty() {
        bail!(
            "testcase {} references incompatible cores: {}",
            testcase_path.display(),
            format_rejected_cores(&selected.rejected_cores)
        );
    }
    if selected.compatible_cores.is_empty() {
        bail!(
            "testcase {} has no compatible cores after applying experiment config",
            testcase_path.display()
        );
    }

    run_diff_analysis_with_testcase(&selected.compatible_cores, diff_config, testcase).map_err(
        |err| {
            CliError::new(format!(
                "diff analysis failed when replaying testcase {}: {err}",
                testcase_path.display()
            ))
        },
    )?;

    info!(
        "Replayed diff testcase {}; artifacts written to {}",
        testcase_path.display(),
        run_dir.display()
    );
    Ok(())
}

fn run_diff_with_cores(
    core_ids: Vec<String>,
    settings: DiffAnalysisSettings,
    concurrency: usize,
    run_dir: PathBuf,
    max_runs: Option<usize>,
    label: &str,
) -> Result<()> {
    if concurrency == 0 {
        bail!("cores.concurrency must be greater than zero");
    }
    if matches!(max_runs, Some(0)) {
        info!("No diff-analysis runs requested for {label}; exiting.");
        return Ok(());
    }
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create base run directory {}", run_dir.display()))?;
    if core_ids.is_empty() {
        bail!("diff-analysis requires at least one core to execute");
    }

    if let Err(err) = riscv_fuzz_testcase_engine::impls::extensions_for_config(
        &core_ids,
        &settings.testcase_config,
    ) {
        let msg = format!("workload configuration invalid for selected cores: {err}");
        return Err(match err {
            GenerationError::RequestedFloatUnsupported { .. }
            | GenerationError::RequestedDoubleUnsupported { .. }
            | GenerationError::RequestedFloatToggleUnsupported { .. }
            | GenerationError::InvalidMemAccessOffset { .. }
            | GenerationError::InsufficientTempRegisters { .. }
            | GenerationError::MemRangeNotFound { .. }
            | GenerationError::MemoryAddressOutOfRange { .. } => CliError::new(msg),
            other => CliError::with_source(msg, other),
        });
    }

    let shared_settings = Arc::new(settings);
    let shared_cores = Arc::new(core_ids);
    let run_counter = Arc::new(AtomicU64::new(0));
    let max_runs_limit = max_runs.map(|v| v as u64);

    info!(
        "Starting diff-analysis runner ({label}) with concurrency {concurrency} in {}",
        run_dir.display()
    );

    let running = Arc::new(AtomicBool::new(true));
    let mut handles = Vec::with_capacity(concurrency);
    for worker_id in 0..concurrency {
        let worker_cores = Arc::clone(&shared_cores);
        let worker_settings = Arc::clone(&shared_settings);
        let worker_base_dir = run_dir.clone();
        let worker_counter = Arc::clone(&run_counter);
        let worker_flag = Arc::clone(&running);

        handles.push(thread::spawn(move || {
            diff_worker_loop(
                worker_id,
                worker_cores,
                worker_settings,
                worker_base_dir,
                worker_counter,
                worker_flag,
                max_runs_limit,
            )
        }));
    }

    let mut first_error: Option<CliError> = None;
    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                running.store(false, Ordering::SeqCst);
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
            Err(panic) => {
                running.store(false, Ordering::SeqCst);
                if first_error.is_none() {
                    first_error = Some(CliError::new(format!(
                        "diff analysis worker panicked: {:?}",
                        panic
                    )));
                }
            }
        }
    }

    if let Some(err) = first_error {
        Err(err)
    } else {
        info!("All diff-analysis workers ({label}) exited cleanly.");
        Ok(())
    }
}

fn diff_worker_loop(
    worker_id: usize,
    core_ids: Arc<Vec<String>>,
    settings: Arc<DiffAnalysisSettings>,
    base_run_dir: PathBuf,
    counter: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
    max_runs: Option<u64>,
) -> Result<()> {
    while running.load(Ordering::SeqCst) {
        let maybe_run_root = next_run_directory(
            &base_run_dir,
            settings.testcase_config.isa_base,
            &counter,
            worker_id,
            max_runs,
        )?;
        let run_root = match maybe_run_root {
            Some(path) => path,
            None => {
                running.store(false, Ordering::SeqCst);
                break;
            }
        };
        fs::create_dir_all(&run_root).with_context(|| {
            format!(
                "failed to create diff-analysis directory {}",
                run_root.display()
            )
        })?;

        let diff_config = settings.with_run_root(run_root.clone());

        info!(
            "[worker {:02}] starting diff analysis in {}",
            worker_id,
            run_root.display()
        );

        match run_diff_analysis(&core_ids, diff_config) {
            Ok(result) => {
                info!(
                    "[worker {:02}] completed diff analysis: {} exception rounds, {} write rounds; output at {}",
                    worker_id,
                    result.exception_removal_rounds,
                    result.write_removal_rounds,
                    result.output_root.display()
                );
            }
            Err(err) => {
                error!(
                    "[worker {:02}] diff analysis failed in {}: {err}",
                    worker_id,
                    run_root.display()
                );
                if let Err(report_err) =
                    write_diff_analysis_failure_report(&run_root, worker_id, &err)
                {
                    error!(
                        "[worker {:02}] failed to write diff-analysis failure report in {}: {}",
                        worker_id,
                        run_root.display(),
                        report_err
                    );
                }
                continue;
            }
        }

        if !running.load(Ordering::SeqCst) {
            break;
        }
    }

    info!("[worker {:02}] shutting down", worker_id);
    Ok(())
}

fn write_diff_analysis_failure_report(
    run_root: &Path,
    worker_id: usize,
    error: &DiffAnalysisError,
) -> Result<()> {
    let report_path = run_root.join("diff_analysis_failure.md");

    let mut content = String::new();
    writeln!(
        &mut content,
        "# Diff Analysis Run Failed\n\n- Worker ID: {:02}\n- Error: {}\n",
        worker_id, error
    )
    .unwrap();
    writeln!(&mut content, "```text\n{:#?}\n```", error).unwrap();

    write_string_file(&report_path, content).with_context(|| {
        format!(
            "failed to write diff-analysis failure report {}",
            report_path.display()
        )
    })?;
    Ok(())
}

fn next_run_directory(
    base_run_dir: &Path,
    isa_base: ISABase,
    counter: &AtomicU64,
    worker_id: usize,
    max_runs: Option<u64>,
) -> Result<Option<PathBuf>> {
    let seq = if let Some(limit) = max_runs {
        match counter.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
            if current >= limit {
                None
            } else {
                Some(current + 1)
            }
        }) {
            Ok(prev) => prev,
            Err(_) => return Ok(None),
        }
    } else {
        counter.fetch_add(1, Ordering::Relaxed)
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX_EPOCH")?;
    let dir_name = format!(
        "{}_{:09}_{}_{}_w{:02}",
        now.as_secs(),
        now.subsec_nanos(),
        isa_base.as_ref(),
        seq,
        worker_id
    );
    Ok(Some(base_run_dir.join(dir_name)))
}

#[cfg(test)]
mod tests {
    use super::{
        SPIKE_CORE_ID, spike_pair_target, stale_core_artifact_warnings_for_bindings,
        strip_unstable_single_hart_generation_extensions,
    };
    use std::{fs, thread, time::Duration};

    use riscv_fuzz_plugin_api::IsaExtensions;
    use riscv_instruction::{RV32Extensions, RV64Extensions};
    use tempfile::tempdir;

    #[test]
    fn detects_newer_local_wrapper_with_same_basename() {
        let tmp = tempdir().expect("tempdir");
        let env_dir = tmp.path().join("env");
        let local_dir = tmp.path().join("local");
        fs::create_dir_all(&env_dir).expect("create env dir");
        fs::create_dir_all(&local_dir).expect("create local dir");

        let env_file = env_dir.join("cva6_rv32_1c");
        fs::write(&env_file, b"old wrapper").expect("write env file");
        thread::sleep(Duration::from_millis(1100));
        let local_file = local_dir.join("cva6_rv32_1c");
        fs::write(&local_file, b"new wrapper").expect("write local file");

        let warnings = stale_core_artifact_warnings_for_bindings(
            &local_dir,
            vec![("CX_RISCV_CORES_CVA6_RV32_1C".to_string(), env_file.clone())],
        )
        .expect("compute warnings");

        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].env_var, "CX_RISCV_CORES_CVA6_RV32_1C");
        assert_eq!(warnings[0].configured_path, env_file);
        assert_eq!(warnings[0].local_path, local_file);
    }

    #[test]
    fn skips_warning_when_configured_path_already_points_to_local_wrapper() {
        let tmp = tempdir().expect("tempdir");
        let local_dir = tmp.path().join("local");
        fs::create_dir_all(&local_dir).expect("create local dir");

        let local_file = local_dir.join("rocket-chip_rv32_1c");
        fs::write(&local_file, b"wrapper").expect("write local file");

        let warnings = stale_core_artifact_warnings_for_bindings(
            &local_dir,
            vec![("CX_RISCV_CORES_ROCKET_CHIP_RV32_1C".to_string(), local_file)],
        )
        .expect("compute warnings");

        assert!(warnings.is_empty());
    }

    #[test]
    fn strip_unstable_single_hart_extensions_remove_amo_lrsc_and_mop() {
        let mut rv32 = IsaExtensions::rv32(vec![
            RV32Extensions::I,
            RV32Extensions::Zaamo,
            RV32Extensions::Zalrsc,
            RV32Extensions::Zimop,
            RV32Extensions::Zcmop,
            RV32Extensions::Zicsr,
        ]);
        assert!(strip_unstable_single_hart_generation_extensions(&mut rv32));
        assert_eq!(
            rv32.as_rv32().expect("rv32 extensions"),
            &[RV32Extensions::I, RV32Extensions::Zicsr]
        );

        let mut rv64 = IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::Zaamo,
            RV64Extensions::Zalrsc,
            RV64Extensions::Zimop,
            RV64Extensions::Zcmop,
            RV64Extensions::Zicsr,
        ]);
        assert!(strip_unstable_single_hart_generation_extensions(&mut rv64));
        assert_eq!(
            rv64.as_rv64().expect("rv64 extensions"),
            &[RV64Extensions::I, RV64Extensions::Zicsr]
        );
    }

    #[test]
    fn spike_pair_target_detects_exact_pairings() {
        assert_eq!(
            spike_pair_target(&[SPIKE_CORE_ID.to_string(), "rocket".to_string()]),
            Some("rocket")
        );
        assert_eq!(
            spike_pair_target(&["cva6".to_string(), SPIKE_CORE_ID.to_string()]),
            Some("cva6")
        );
        assert_eq!(
            spike_pair_target(&[SPIKE_CORE_ID.to_string(), SPIKE_CORE_ID.to_string()]),
            None
        );
        assert_eq!(
            spike_pair_target(&[
                SPIKE_CORE_ID.to_string(),
                "rocket".to_string(),
                "cva6".to_string()
            ]),
            None
        );
    }
}
