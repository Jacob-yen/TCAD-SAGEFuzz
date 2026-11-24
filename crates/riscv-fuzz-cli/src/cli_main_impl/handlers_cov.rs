use std::path::PathBuf;

use riscv_fuzz_config::{ExperimentConfig, RunnerMode};
use riscv_fuzz_testcase_engine::cov_guided::run_cov_guided;

use super::{CliError, Result, cov_settings_from_experiment};

/// Run the coverage-guided runner. Validation has already happened in the
/// loader, so we only need to pull the `[runner.coverage_guided]` subtable
/// and the relevant workload/reporting/cores fields.
pub(super) fn run_cov_guided_unified(cfg: &ExperimentConfig, run_dir: PathBuf) -> Result<()> {
    debug_assert!(matches!(cfg.runner.mode, RunnerMode::CoverageGuided));
    cfg.runner
        .coverage_guided
        .as_ref()
        .expect("validator R9 ensures coverage_guided subtable is present");

    // Coverage-guided historically targets a single core. For now we keep
    // that behaviour: the first selected core wins; CoresConfig::selected
    // must contain at least one entry (validator surfaces a clear error
    // upstream if not).
    let core_id = cfg.cores.selected.first().cloned().ok_or_else(|| {
        CliError::new(
            "runner.mode = coverage_guided requires cores.selected to name at least one core"
                .to_string(),
        )
    })?;

    let label = core_id.clone();
    let settings = cov_settings_from_experiment(cfg, run_dir, core_id);

    run_cov_guided(settings)
        .map_err(|err| CliError::new(format!("cov-guided failed (core={label}): {err}")))?;

    Ok(())
}
