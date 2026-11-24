//! Command-line entry point for riscv-fuzz.
//!
//! After the configuration refactor, the CLI surface is intentionally small:
//! every nontrivial invocation goes through `riscv-fuzz run`, which loads a
//! single `experiment.toml` (optionally layered with overlay files) and
//! dispatches to the runner mode declared inside it. Only a handful of
//! high-frequency knobs remain on the command line; everything else lives in
//! the file so configurations are reproducible and reviewable.
//!
//! Subcommands:
//!
//! - `run`              — load an experiment config and dispatch by `runner.mode`
//! - `generate`         — produce program.S/.elf for one core, no execution
//! - `measure-sync`     — dual-hart cycle metrics (kept separate; different output schema)
//! - `render-timeline`  — pure post-processing; no config file needed
//! - `plugin {catalog,doctor}` — plugin diagnostics

use std::env;
use std::error::Error as StdError;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use riscv_fuzz_config::{
    CliOverrides, ExperimentConfig, RunnerMode, ValidationError, load_experiment_config,
    validate_core_runtime_requirements,
};
use riscv_fuzz_diff::{DiffAnalysisSettings, DiffDedupSettings};
use riscv_fuzz_plugin_api::{CoverageMode, ISABase};
use riscv_fuzz_plugin_manager::PluginManager;
use riscv_fuzz_testcase_engine::{cov_guided::CovGuidedSettings, testcase::TestCaseConfig};

mod handlers_cov;
mod handlers_diff;
mod handlers_exec_generate;
mod handlers_measure_sync;
mod handlers_plugins;
mod handlers_visualize;
mod paths;

use handlers_cov::run_cov_guided_unified;
use handlers_diff::{run_diff_against_spike_unified, run_random_diff_unified, run_replay_unified};
use handlers_exec_generate::{
    ExecGeneratedGenerator, run_exec_generated_batch_unified, run_exec_generated_unified,
    run_generate_unified,
};
use handlers_measure_sync::run_measure_multicore_sync_mode;
use handlers_plugins::{run_plugin_catalog_mode, run_plugin_doctor_mode};
use handlers_visualize::run_render_timeline_mode;
use paths::{absolutize_optional_path, absolutize_path};

pub(crate) type Result<T> = std::result::Result<T, CliError>;
const EXEC_TIMEOUT_ENV: &str = "RISCV_FUZZ_EXEC_TIMEOUT_SECS";

#[derive(Debug)]
pub(crate) struct CliError {
    message: String,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl CliError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    pub(crate) fn with_source<E>(message: impl Into<String>, source: E) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl StdError for CliError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_ref()
            .map(|src| src.as_ref() as &(dyn StdError + 'static))
    }
}

trait ResultExt<T> {
    fn with_context<F, M>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> M,
        M: Into<String>;

    fn context<M>(self, message: M) -> Result<T>
    where
        M: Into<String>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: StdError + Send + Sync + 'static,
{
    fn with_context<F, M>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> M,
        M: Into<String>,
    {
        self.map_err(|err| CliError::with_source(f().into(), err))
    }

    fn context<M>(self, message: M) -> Result<T>
    where
        M: Into<String>,
    {
        self.map_err(|err| CliError::with_source(message.into(), err))
    }
}

#[derive(Parser)]
#[command(
    version,
    about = "Generate, execute, and differential-test RISC-V programs",
    long_about = "Drive the runner via a single experiment.toml: workload, runner mode, reporting, and core selection are all declared in the file. Use overlays for per-experiment tweaks and CLI flags only for high-frequency overrides."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(
        name = "run",
        about = "Load an experiment config and dispatch by runner.mode",
        long_about = "Loads --config (with optional overlays merged on top), validates, and runs whichever mode the file declares: replay, random_diff, diff_against_spike, or coverage_guided. CLI overrides apply on top."
    )]
    Run {
        /// Path to the base `experiment.toml`.
        #[arg(long)]
        config: PathBuf,

        /// Optional overlay TOML files. Each overlay may declare any of
        /// [testcase_config] / [runner] / [reporting] / [cores] and replaces
        /// the matching section wholesale (no deep merge).
        #[arg(long = "overlay", num_args = 0.., value_delimiter = ',')]
        overlays: Vec<PathBuf>,

        /// Directory under which all run artifacts are written.
        #[arg(long = "run-dir")]
        run_dir: PathBuf,

        /// Override `cores.selected`. Pass comma-separated ids.
        #[arg(long = "core", num_args = 0.., value_delimiter = ',')]
        cores: Vec<String>,

        /// Override `cores.concurrency`.
        #[arg(long)]
        concurrency: Option<usize>,

        /// Override the active runner subtable's `max_iterations`. Has no
        /// effect on `replay`.
        #[arg(long = "max-iterations")]
        max_iterations: Option<usize>,

        /// Override `runner.seed`.
        #[arg(long)]
        seed: Option<u64>,
    },

    #[command(
        name = "generate",
        about = "Generate program.S/.elf for one core without executing",
        long_about = "Reads the testcase_config section of --config, generates a program targeted at --core, and writes program.S, ELF binaries, and (unless --minimal-artifacts) testcase.json under --run-dir."
    )]
    Generate {
        /// Path to the base `experiment.toml`. Only the [testcase_config] section is consumed.
        #[arg(long)]
        config: PathBuf,

        /// Optional overlay files (same semantics as `run`).
        #[arg(long = "overlay", num_args = 0.., value_delimiter = ',')]
        overlays: Vec<PathBuf>,

        /// Target core id (e.g. `spike`, `rocket`).
        #[arg(long)]
        core: String,

        /// Optional override for workload.isa_base.
        #[arg(long, value_enum)]
        isa: Option<ISABase>,

        /// Remove Zacas from the generated/executed ISA extension set.
        #[arg(long = "disable-zacas")]
        disable_zacas: bool,

        /// Testcase generator flavor, including `hardware-vc-sf-only-negative` for VC/SF plus sampled negative tests.
        #[arg(long = "generator", value_enum, default_value = "hardware-tcad")]
        generator: ExecGeneratedGenerator,

        /// Output directory for generated artifacts.
        #[arg(long = "run-dir")]
        run_dir: PathBuf,

        /// Emit only program.S/.s and ELF; skip testcase.json + instruction snapshot.
        #[arg(long = "minimal-artifacts")]
        minimal_artifacts: bool,
    },

    #[command(
        name = "exec-generated",
        about = "Generate one testcase and directly execute it on one core",
        long_about = "Reads the testcase_config section of --config, generates one testcase for --core, writes user_instructions.asm/testcase.json under --run-dir, and directly executes the core plugin. This bypasses diff-analysis reduction, exception removal, and write localization."
    )]
    ExecGenerated {
        /// Path to the base `experiment.toml`. Only the [testcase_config] section is consumed.
        #[arg(long)]
        config: PathBuf,

        /// Optional overlay files (same semantics as `run`).
        #[arg(long = "overlay", num_args = 0.., value_delimiter = ',')]
        overlays: Vec<PathBuf>,

        /// Target core id (e.g. `xiangshan`).
        #[arg(long)]
        core: String,

        /// Optional override for workload.isa_base.
        #[arg(long, value_enum)]
        isa: Option<ISABase>,

        /// Remove Zacas from the generated/executed ISA extension set.
        #[arg(long = "disable-zacas")]
        disable_zacas: bool,

        /// Testcase generator flavor, including `hardware-vc-sf-only-negative` for VC/SF plus sampled negative tests.
        #[arg(long = "generator", value_enum, default_value = "hardware-tcad")]
        generator: ExecGeneratedGenerator,

        /// Output directory for generated and execution artifacts.
        #[arg(long = "run-dir")]
        run_dir: PathBuf,
    },

    #[command(
        name = "exec-generated-batch",
        about = "Generate many testcases and directly execute them on one core",
        long_about = "Reads the testcase_config section of --config, generates --cases independent testcases for --core, executes them directly with up to --workers parallel workers, and writes per-case DiffTest summaries plus batch_summary.json/md. This bypasses diff-analysis reduction, exception removal, write localization, and deduplication."
    )]
    ExecGeneratedBatch {
        /// Path to the base `experiment.toml`. Only the [testcase_config] section is consumed.
        #[arg(long)]
        config: PathBuf,

        /// Optional overlay files (same semantics as `run`).
        #[arg(long = "overlay", num_args = 0.., value_delimiter = ',')]
        overlays: Vec<PathBuf>,

        /// Target core id (e.g. `xiangshan`).
        #[arg(long)]
        core: String,

        /// Optional override for workload.isa_base.
        #[arg(long, value_enum)]
        isa: Option<ISABase>,

        /// Remove Zacas from the generated/executed ISA extension set.
        #[arg(long = "disable-zacas")]
        disable_zacas: bool,

        /// Testcase generator flavor, including `hardware-vc-sf-only-negative` for VC/SF plus sampled negative tests.
        #[arg(long = "generator", value_enum, default_value = "hardware-tcad")]
        generator: ExecGeneratedGenerator,

        /// Output directory for generated and execution artifacts.
        #[arg(long = "run-dir")]
        run_dir: PathBuf,

        /// Number of generated/executed testcases.
        #[arg(long, default_value_t = 100)]
        cases: usize,

        /// Number of parallel workers.
        #[arg(long, default_value_t = 10)]
        workers: usize,

        /// First seed to use; case N uses seed_start + N.
        #[arg(long = "seed-start")]
        seed_start: Option<u64>,
    },

    #[command(
        name = "measure-sync",
        about = "Batch-generate dual-hart workloads and summarize cycle overlap metrics",
        long_about = "Generates many dual-hart programs from --config's workload, executes each on --core, computes multicore sync metrics from per-instruction cycle windows, and writes aggregate JSON/Markdown summaries under --run-dir."
    )]
    MeasureSync {
        #[arg(long)]
        config: PathBuf,
        #[arg(long = "overlay", num_args = 0.., value_delimiter = ',')]
        overlays: Vec<PathBuf>,
        /// Target core id; must support dual-hart and the workload's ISA base.
        #[arg(long)]
        core: String,
        #[arg(long, value_enum)]
        isa: Option<ISABase>,
        #[arg(long = "run-dir")]
        run_dir: PathBuf,
        #[arg(long, default_value_t = 100)]
        samples: usize,
        #[arg(long = "seed-start")]
        seed_start: Option<u64>,
    },

    #[command(
        name = "render-timeline",
        about = "Render an execution timeline SVG from an existing execution JSON",
        long_about = "Reads execution JSON produced by an earlier run and renders execution_timeline.svg without re-executing the simulator."
    )]
    RenderTimeline {
        #[arg(long = "execution-output")]
        execution_output: PathBuf,

        #[arg(long = "output")]
        output: PathBuf,
    },

    #[command(
        name = "plugin",
        alias = "plugins",
        about = "Inspect plugin discovery, loaded plugins, and plugin metadata",
        long_about = "Diagnose RISC-V core plugin discovery (env vars, search paths, candidate libraries, registry contents)."
    )]
    Plugin {
        #[command(subcommand)]
        command: PluginCommand,
    },
}

#[derive(Debug, Subcommand)]
enum PluginCommand {
    #[command(
        name = "catalog",
        visible_alias = "list",
        alias = "plugin-catalog",
        about = "Render a Markdown catalog of every dynamically loaded plugin"
    )]
    Catalog {
        /// Optional output path for the generated Markdown catalog.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    #[command(
        name = "doctor",
        about = "Diagnose plugin environment variables, discovery paths, and load results"
    )]
    Doctor {
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Debug)]
pub(crate) struct CoreSelectionResult {
    pub(crate) compatible_cores: Vec<String>,
    pub(crate) rejected_cores: Vec<(String, String)>,
}

pub(crate) fn select_compatible_cores(
    plugin_manager: &PluginManager,
    cores: &[String],
    testcase_config: &TestCaseConfig,
    coverage_mode: CoverageMode,
) -> CoreSelectionResult {
    let mut compatible_cores = Vec::new();
    let mut rejected_cores = Vec::new();

    for core_ref in cores {
        let plugin = match plugin_manager.plugin(core_ref) {
            Ok(plugin) => plugin,
            Err(_) => {
                rejected_cores.push((
                    core_ref.clone(),
                    ValidationError::CoreNotDiscovered {
                        core: core_ref.clone(),
                    }
                    .to_string(),
                ));
                continue;
            }
        };

        match validate_core_runtime_requirements(
            core_ref,
            plugin.metadata(),
            testcase_config.isa_base,
            testcase_config.hart_count(),
            testcase_config.unaligned_access_required,
            testcase_config.float_extension,
            coverage_mode,
        ) {
            Ok(()) => compatible_cores.push(core_ref.clone()),
            Err(ValidationError::CoreIncompatibleWithWorkload { reason, .. }) => {
                rejected_cores.push((core_ref.clone(), reason));
            }
            Err(err) => rejected_cores.push((core_ref.clone(), err.to_string())),
        }
    }

    CoreSelectionResult {
        compatible_cores,
        rejected_cores,
    }
}

pub(crate) fn format_rejected_cores(rejected_cores: &[(String, String)]) -> String {
    rejected_cores
        .iter()
        .map(|(core_ref, reason)| format!("{core_ref}: {reason}"))
        .collect::<Vec<_>>()
        .join("; ")
}

pub(crate) fn diff_settings_from_experiment(cfg: &ExperimentConfig) -> DiffAnalysisSettings {
    let testcase_config = cfg.testcase_config.clone();
    let max_iterations = cfg.diff_max_iterations();
    let history = cfg.diff_history_min_threshold();

    DiffAnalysisSettings {
        testcase_config,
        coverage_mode: cfg.reporting.coverage_collection,
        execution_timeout_seconds: cfg.cores.execution_timeout_seconds,
        max_iterations,
        enable_reduction: cfg.diff_reduction_enabled().unwrap_or(true),
        history_min_test_threshold: history,
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
    }
}

pub(crate) fn cov_settings_from_experiment(
    cfg: &ExperimentConfig,
    run_dir: PathBuf,
    core_id: String,
) -> CovGuidedSettings {
    let cov = cfg
        .runner
        .coverage_guided
        .as_ref()
        .expect("validator R9 ensures coverage_guided subtable is present");
    let max_runs = if cov.max_iterations == 0 {
        None
    } else {
        Some(cov.max_iterations as u64)
    };

    CovGuidedSettings {
        testcase_config: cfg.testcase_config.clone(),
        core_id,
        run_dir,
        max_runs,
        coverage_mode: cfg.reporting.coverage_collection,
        coverage_profile: cov.coverage_profile.clone(),
        coverage_types: cov.coverage_types.clone(),
        seed: cfg.effective_seed(),
        mutation_steps: cov.mutations_per_candidate,
        fresh_probability: cov.fresh_program_probability,
        spice_probability: cov.chaos_block_injection_probability,
        min_sequences: cov.instruction_block_count_range.min,
        max_sequences: cov.instruction_block_count_range.max,
        max_corpus: cov.corpus_max_entries,
        stats_every: cov.stats_print_interval,
        keep_noninteresting: cfg.reporting.keep_uninteresting_workdirs,
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct CliApp;

impl CliApp {
    pub(crate) fn run(self) -> Result<()> {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init()
            .map_err(|err| CliError::new(format!("failed to initialize logger: {err}")))?;

        let cli = Cli::parse();
        self.dispatch(cli.command)
    }

    fn dispatch(&self, command: Command) -> Result<()> {
        match command {
            Command::Plugin { command } => match command {
                PluginCommand::Catalog { output } => {
                    run_plugin_catalog_mode(absolutize_optional_path(output)?)
                }
                PluginCommand::Doctor { output } => {
                    run_plugin_doctor_mode(absolutize_optional_path(output)?)
                }
            },
            Command::Run {
                config,
                overlays,
                run_dir,
                cores,
                concurrency,
                max_iterations,
                seed,
            } => {
                let cfg = load_with_overrides(
                    config,
                    overlays,
                    CliOverrides {
                        cores: if cores.is_empty() { None } else { Some(cores) },
                        concurrency,
                        max_iterations,
                        seed,
                        run_dir: Some(run_dir.clone()),
                    },
                )?;
                dispatch_run(&cfg, absolutize_path(run_dir)?)
            }
            Command::Generate {
                config,
                overlays,
                core,
                isa,
                disable_zacas,
                generator,
                run_dir,
                minimal_artifacts,
            } => {
                let cfg = load_with_overrides(config, overlays, CliOverrides::default())?;
                run_generate_unified(
                    &cfg,
                    absolutize_path(run_dir)?,
                    core,
                    isa,
                    disable_zacas,
                    minimal_artifacts,
                    generator,
                )
            }
            Command::ExecGenerated {
                config,
                overlays,
                core,
                isa,
                disable_zacas,
                generator,
                run_dir,
            } => {
                let cfg = load_with_overrides(config, overlays, CliOverrides::default())?;
                let _timeout_guard = set_env_var(
                    EXEC_TIMEOUT_ENV,
                    &cfg.cores.execution_timeout_seconds.to_string(),
                );
                run_exec_generated_unified(
                    &cfg,
                    absolutize_path(run_dir)?,
                    core,
                    isa,
                    disable_zacas,
                    generator,
                )
            }
            Command::ExecGeneratedBatch {
                config,
                overlays,
                core,
                isa,
                disable_zacas,
                generator,
                run_dir,
                cases,
                workers,
                seed_start,
            } => {
                let cfg = load_with_overrides(config, overlays, CliOverrides::default())?;
                let _timeout_guard = set_env_var(
                    EXEC_TIMEOUT_ENV,
                    &cfg.cores.execution_timeout_seconds.to_string(),
                );
                run_exec_generated_batch_unified(
                    &cfg,
                    absolutize_path(run_dir)?,
                    core,
                    isa,
                    cases,
                    workers,
                    seed_start,
                    disable_zacas,
                    generator,
                )
            }
            Command::MeasureSync {
                config,
                overlays,
                core,
                isa,
                run_dir,
                samples,
                seed_start,
            } => {
                let cfg = load_with_overrides(config, overlays, CliOverrides::default())?;
                run_measure_multicore_sync_mode(
                    &cfg,
                    absolutize_path(run_dir)?,
                    core,
                    isa,
                    samples,
                    seed_start,
                )
            }
            Command::RenderTimeline {
                execution_output,
                output,
            } => run_render_timeline_mode(
                absolutize_path(execution_output)?,
                absolutize_path(output)?,
            ),
        }
    }
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => unsafe { env::set_var(self.key, value) },
            None => unsafe { env::remove_var(self.key) },
        }
    }
}

fn set_env_var(key: &'static str, value: &str) -> EnvVarGuard {
    let previous = env::var_os(key);
    unsafe { env::set_var(key, value) };
    EnvVarGuard { key, previous }
}

fn load_with_overrides(
    config: PathBuf,
    overlays: Vec<PathBuf>,
    overrides: CliOverrides,
) -> Result<ExperimentConfig> {
    let config = absolutize_path(config)?;
    let overlays = overlays
        .into_iter()
        .map(absolutize_path)
        .collect::<Result<Vec<_>>>()?;
    load_experiment_config(&config, &overlays, &overrides)
        .map_err(|err| CliError::new(err.to_string()))
}

fn dispatch_run(cfg: &ExperimentConfig, run_dir: PathBuf) -> Result<()> {
    let _timeout_guard = set_env_var(
        EXEC_TIMEOUT_ENV,
        &cfg.cores.execution_timeout_seconds.to_string(),
    );
    match cfg.runner.mode {
        RunnerMode::Replay => run_replay_unified(cfg, run_dir),
        RunnerMode::RandomDiff => run_random_diff_unified(cfg, run_dir),
        RunnerMode::DiffAgainstSpike => run_diff_against_spike_unified(cfg, run_dir),
        RunnerMode::CoverageGuided => run_cov_guided_unified(cfg, run_dir),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riscv_fuzz_config::ExperimentConfig;

    #[test]
    fn plugin_catalog_subcommand_parses() {
        let cli = Cli::try_parse_from(["riscv_fuzz_test", "plugin", "catalog"])
            .expect("plugin catalog should parse");
        match cli.command {
            Command::Plugin {
                command: PluginCommand::Catalog { output },
            } => assert!(output.is_none()),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn plugin_doctor_subcommand_parses() {
        let cli = Cli::try_parse_from([
            "riscv_fuzz_test",
            "plugin",
            "doctor",
            "--output",
            "doctor.md",
        ])
        .expect("plugin doctor should parse");
        match cli.command {
            Command::Plugin {
                command: PluginCommand::Doctor { output },
            } => assert_eq!(output, Some(PathBuf::from("doctor.md"))),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn plugin_catalog_list_alias_parses() {
        let cli = Cli::try_parse_from(["riscv_fuzz_test", "plugin", "list"])
            .expect("plugin list alias should parse");
        match cli.command {
            Command::Plugin {
                command: PluginCommand::Catalog { .. },
            } => {}
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn run_subcommand_parses_minimal_form() {
        let cli = Cli::try_parse_from([
            "riscv_fuzz_test",
            "run",
            "--config",
            "experiment.toml",
            "--run-dir",
            "/tmp/run",
        ])
        .expect("run should parse");
        match cli.command {
            Command::Run {
                config,
                overlays,
                run_dir,
                cores,
                concurrency,
                max_iterations,
                seed,
            } => {
                assert_eq!(config, PathBuf::from("experiment.toml"));
                assert!(overlays.is_empty());
                assert_eq!(run_dir, PathBuf::from("/tmp/run"));
                assert!(cores.is_empty());
                assert!(concurrency.is_none());
                assert!(max_iterations.is_none());
                assert!(seed.is_none());
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn run_subcommand_parses_overrides() {
        let cli = Cli::try_parse_from([
            "riscv_fuzz_test",
            "run",
            "--config",
            "x.toml",
            "--overlay",
            "a.toml,b.toml",
            "--run-dir",
            "/tmp/r",
            "--core",
            "spike,rocket",
            "--concurrency",
            "8",
            "--max-iterations",
            "100",
            "--seed",
            "42",
        ])
        .expect("run with overrides should parse");
        match cli.command {
            Command::Run {
                overlays,
                cores,
                concurrency,
                max_iterations,
                seed,
                ..
            } => {
                assert_eq!(overlays.len(), 2);
                assert_eq!(cores, vec!["spike", "rocket"]);
                assert_eq!(concurrency, Some(8));
                assert_eq!(max_iterations, Some(100));
                assert_eq!(seed, Some(42));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn render_timeline_subcommand_parses() {
        let cli = Cli::try_parse_from([
            "riscv_fuzz_test",
            "render-timeline",
            "--execution-output",
            "execution_output.json",
            "--output",
            "execution_timeline.svg",
        ])
        .expect("render-timeline should parse");
        match cli.command {
            Command::RenderTimeline {
                execution_output,
                output,
            } => {
                assert_eq!(execution_output, PathBuf::from("execution_output.json"));
                assert_eq!(output, PathBuf::from("execution_timeline.svg"));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn diff_settings_default_to_no_generation_override() {
        let cfg = ExperimentConfig::default();
        let settings = diff_settings_from_experiment(&cfg);
        assert!(settings.generation_extensions_override.is_none());
        assert!(settings.execution_extensions_override.is_none());
        assert_eq!(
            settings.execution_timeout_seconds,
            cfg.cores.execution_timeout_seconds
        );
    }

    #[test]
    fn set_env_var_guard_restores_previous_value() {
        let previous = env::var_os(EXEC_TIMEOUT_ENV);
        unsafe { env::remove_var(EXEC_TIMEOUT_ENV) };

        {
            let _guard = set_env_var(EXEC_TIMEOUT_ENV, "123");
            assert_eq!(env::var(EXEC_TIMEOUT_ENV).as_deref(), Ok("123"));
        }
        assert!(env::var_os(EXEC_TIMEOUT_ENV).is_none());

        match previous {
            Some(value) => unsafe { env::set_var(EXEC_TIMEOUT_ENV, value) },
            None => unsafe { env::remove_var(EXEC_TIMEOUT_ENV) },
        }
    }
}
