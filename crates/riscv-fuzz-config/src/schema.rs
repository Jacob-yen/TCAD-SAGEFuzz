//! Unified experiment configuration schema.
//!
//! All TOML the operator writes deserializes into [`ExperimentConfig`] and its
//! sub-tables. Field names are explicit about units (`region_size_kib`,
//! `execution_timeout_seconds`, `access_offset_range_bytes`) and avoid project
//! jargon. Cross-field validation lives in [`crate::validate_structural`]; this module
//! only defines the data shape and per-field defaults.
//!
//! Every section has a stable, snake_case TOML key. Subtables under
//! `[runner]` are exclusive — exactly one matching `runner.mode` must be
//! present (validator rule R9). All optional fields fall back to compile-time
//! defaults from [`crate::defaults`], so a minimal experiment file only needs
//! to set the values it actually wants to deviate from.

use std::path::PathBuf;

use riscv_fuzz_plugin_api::coverage::CoverageMode;
use riscv_fuzz_testcase_engine::testcase::TestCaseConfig;
use serde::{Deserialize, Serialize};

use crate::defaults;

// =====================================================================
// Range helpers — TOML accepts `[min, max]`; we store as a struct for
// easier field access and good Display.
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct U8Range {
    pub min: u8,
    pub max: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct UsizeRange {
    pub min: usize,
    pub max: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct I64Range {
    pub min: i64,
    pub max: i64,
}

macro_rules! impl_pair_deserialize {
    ($t:ident, $inner:ty) => {
        impl<'de> Deserialize<'de> for $t {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                #[derive(Deserialize)]
                #[serde(untagged)]
                enum Repr {
                    Pair([$inner; 2]),
                    Single($inner),
                }
                let repr = Repr::deserialize(deserializer)?;
                let (min, max) = match repr {
                    Repr::Pair([a, b]) => (a, b),
                    Repr::Single(v) => (v, v),
                };
                Ok($t { min, max })
            }
        }
    };
}
impl_pair_deserialize!(U8Range, u8);
impl_pair_deserialize!(UsizeRange, usize);
impl_pair_deserialize!(I64Range, i64);

// =====================================================================
// Top-level experiment
// =====================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    #[serde(default)]
    pub testcase_config: TestCaseConfig,
    #[serde(default)]
    pub runner: RunnerConfig,
    #[serde(default)]
    pub reporting: ReportingConfig,
    #[serde(default)]
    pub cores: CoresConfig,
}

impl ExperimentConfig {
    /// Resolve the effective RNG seed for runners that need one.
    ///
    /// Runner-level `seed` overrides the testcase seed so users can replay a
    /// specific run without disturbing the testcase-shaped default.
    pub fn effective_seed(&self) -> Option<u64> {
        self.runner.seed.or(self.testcase_config.seed)
    }

    /// Maximum iteration budget reported by the active runner subtable, if any.
    ///
    /// `random_diff` and `diff_against_spike` both expose `max_iterations`;
    /// other runner modes do not, in which case this returns `None`.
    pub fn diff_max_iterations(&self) -> Option<usize> {
        self.runner
            .random_diff
            .as_ref()
            .map(|c| c.max_iterations)
            .or_else(|| {
                self.runner
                    .diff_against_spike
                    .as_ref()
                    .map(|c| c.max_iterations)
            })
    }

    /// Whether the active diff runner should reduce failing testcases.
    pub fn diff_reduction_enabled(&self) -> Option<bool> {
        self.runner
            .random_diff
            .as_ref()
            .map(|c| c.enable_reduction)
            .or_else(|| {
                self.runner
                    .diff_against_spike
                    .as_ref()
                    .map(|c| c.enable_reduction)
            })
    }

    /// Minimum history-iterations threshold for the active diff runner, if any.
    pub fn diff_history_min_threshold(&self) -> Option<usize> {
        self.runner
            .random_diff
            .as_ref()
            .map(|c| c.minimization_min_history_iterations)
            .or_else(|| {
                self.runner
                    .diff_against_spike
                    .as_ref()
                    .map(|c| c.minimization_min_history_iterations)
            })
    }
}

// =====================================================================
// Runner — drives iteration. Mode-exclusive subtables.
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunnerMode {
    Replay,
    RandomDiff,
    DiffAgainstSpike,
    CoverageGuided,
}

impl Default for RunnerMode {
    fn default() -> Self {
        defaults::runner_mode()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerConfig {
    #[serde(default)]
    pub mode: RunnerMode,
    /// RNG seed for the runner driver (corpus selection, mutation choice,
    /// worker partitioning). Testcase generation has its own seed under
    /// `[testcase_config].seed`.
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub replay: Option<ReplayConfig>,
    #[serde(default)]
    pub random_diff: Option<RandomDiffConfig>,
    #[serde(default)]
    pub diff_against_spike: Option<DiffAgainstSpikeConfig>,
    #[serde(default)]
    pub coverage_guided: Option<CoverageGuidedConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConfig {
    /// Path to a previously generated `testcase.json` to replay.
    pub testcase_path: PathBuf,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomDiffConfig {
    #[serde(default = "defaults::diff_max_iterations")]
    pub max_iterations: usize,
    /// During minimization, require at least this many history iterations
    /// before concluding the failure is reproducible.
    #[serde(default = "defaults::diff_enable_reduction")]
    pub enable_reduction: bool,
    /// During minimization, require at least this many history iterations
    /// before concluding the failure is reproducible.
    #[serde(default = "defaults::diff_minimization_min_history_iterations")]
    pub minimization_min_history_iterations: usize,
}

impl Default for RandomDiffConfig {
    fn default() -> Self {
        Self {
            max_iterations: defaults::diff_max_iterations(),
            enable_reduction: defaults::diff_enable_reduction(),
            minimization_min_history_iterations: defaults::diff_minimization_min_history_iterations(
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffAgainstSpikeConfig {
    #[serde(default = "defaults::diff_max_iterations")]
    pub max_iterations: usize,
    #[serde(default = "defaults::diff_enable_reduction")]
    pub enable_reduction: bool,
    #[serde(default = "defaults::diff_minimization_min_history_iterations")]
    pub minimization_min_history_iterations: usize,
}

impl Default for DiffAgainstSpikeConfig {
    fn default() -> Self {
        Self {
            max_iterations: defaults::diff_max_iterations(),
            enable_reduction: defaults::diff_enable_reduction(),
            minimization_min_history_iterations: defaults::diff_minimization_min_history_iterations(
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageGuidedConfig {
    /// Maximum total iterations across all workers. `0` means unbounded.
    #[serde(default = "defaults::cov_max_iterations")]
    pub max_iterations: usize,
    /// Number of mutation operations applied per selected candidate before
    /// re-evaluating coverage.
    #[serde(default = "defaults::cov_mutations_per_candidate")]
    pub mutations_per_candidate: u32,
    /// Probability of starting from a freshly generated program instead of
    /// mutating an existing corpus entry.
    #[serde(default = "defaults::cov_fresh_program_probability")]
    pub fresh_program_probability: f64,
    /// Probability of injecting a chaos block (illegal ops, misaligned
    /// memory, etc.) during mutation.
    #[serde(default = "defaults::cov_chaos_block_injection_probability")]
    pub chaos_block_injection_probability: f64,
    /// Inclusive range for the per-program instruction-block count used by
    /// the generator.
    #[serde(default = "defaults::cov_instruction_block_count_range")]
    pub instruction_block_count_range: UsizeRange,
    /// Maximum number of corpus entries kept on disk.
    #[serde(default = "defaults::cov_corpus_max_entries")]
    pub corpus_max_entries: usize,
    /// Print fuzzing stats every N runs.
    #[serde(default = "defaults::cov_stats_print_interval")]
    pub stats_print_interval: u64,
    /// Optional named coverage profile (e.g. `"line"`, `"line-branch"`).
    /// Mutually overrides plugin defaults; ignored when `coverage_types` is
    /// also set.
    #[serde(default)]
    pub coverage_profile: Option<String>,
    /// Optional explicit list of coverage types to enable. Overrides
    /// `coverage_profile` when present.
    #[serde(default)]
    pub coverage_types: Option<Vec<String>>,
}

impl Default for CoverageGuidedConfig {
    fn default() -> Self {
        Self {
            max_iterations: defaults::cov_max_iterations(),
            mutations_per_candidate: defaults::cov_mutations_per_candidate(),
            fresh_program_probability: defaults::cov_fresh_program_probability(),
            chaos_block_injection_probability: defaults::cov_chaos_block_injection_probability(),
            instruction_block_count_range: defaults::cov_instruction_block_count_range(),
            corpus_max_entries: defaults::cov_corpus_max_entries(),
            stats_print_interval: defaults::cov_stats_print_interval(),
            coverage_profile: None,
            coverage_types: None,
        }
    }
}

// =====================================================================
// Reporting
// =====================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReportingConfig {
    /// Coverage collection level. Must be non-`off` when
    /// `runner.mode == coverage_guided` (R1).
    #[serde(default = "defaults::reporting_coverage_collection")]
    pub coverage_collection: CoverageMode,
    #[serde(default = "defaults::reporting_emit_execution_json")]
    pub emit_execution_json: bool,
    #[serde(default = "defaults::reporting_emit_execution_markdown")]
    pub emit_execution_markdown: bool,
    #[serde(default = "defaults::reporting_delete_successful_iteration_artifacts")]
    pub delete_successful_iteration_artifacts: bool,
    #[serde(default = "defaults::reporting_delete_successful_run_artifacts")]
    pub delete_successful_run_artifacts: bool,
    /// Coverage-guided fuzzer keeps work directories of uninteresting runs
    /// when set; otherwise they are removed.
    #[serde(default = "defaults::reporting_keep_uninteresting_workdirs")]
    pub keep_uninteresting_workdirs: bool,
    #[serde(default)]
    pub write_probe_verification: WriteProbeVerificationConfig,
    #[serde(default)]
    pub dedup: DiffDedupConfig,
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            coverage_collection: defaults::reporting_coverage_collection(),
            emit_execution_json: defaults::reporting_emit_execution_json(),
            emit_execution_markdown: defaults::reporting_emit_execution_markdown(),
            delete_successful_iteration_artifacts:
                defaults::reporting_delete_successful_iteration_artifacts(),
            delete_successful_run_artifacts: defaults::reporting_delete_successful_run_artifacts(),
            keep_uninteresting_workdirs: defaults::reporting_keep_uninteresting_workdirs(),
            write_probe_verification: WriteProbeVerificationConfig::default(),
            dedup: DiffDedupConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WriteProbeVerificationConfig {
    /// When enabled, write differences between cores are re-verified by
    /// replaying the program with extra register/memory probes to filter
    /// false positives.
    #[serde(default = "defaults::write_probe_enabled")]
    pub enabled: bool,
    /// Maximum number of register probes injected per replay.
    #[serde(default = "defaults::write_probe_max_registers")]
    pub max_registers: usize,
    /// Maximum number of memory probe bytes injected per replay.
    #[serde(default = "defaults::write_probe_max_memory_bytes")]
    pub max_memory_bytes: usize,
}

impl Default for WriteProbeVerificationConfig {
    fn default() -> Self {
        Self {
            enabled: defaults::write_probe_enabled(),
            max_registers: defaults::write_probe_max_registers(),
            max_memory_bytes: defaults::write_probe_max_memory_bytes(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffDedupConfig {
    /// Enable automatic diff labels and local persistent de-duplication.
    #[serde(default = "defaults::diff_dedup_enabled")]
    pub enabled: bool,
    /// Optional JSONL store path. When omitted, diff runs place the store next
    /// to the directory containing per-run artifacts.
    #[serde(default)]
    pub database_path: Option<PathBuf>,
}

impl Default for DiffDedupConfig {
    fn default() -> Self {
        Self {
            enabled: defaults::diff_dedup_enabled(),
            database_path: None,
        }
    }
}

// =====================================================================
// Cores — selection and discovery of RISC-V core implementations.
// =====================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoresConfig {
    /// Core ids to include (e.g. `["spike", "rocket"]`). Empty means
    /// "every core whose metadata is compatible with the workload".
    #[serde(default)]
    pub selected: Vec<String>,
    /// Number of parallel workers. Must be 1 when
    /// The active runner may still require this to be `1` depending on its
    /// implementation, but the schema no longer reserves a dedicated
    /// single-shot mode.
    #[serde(default = "defaults::cores_concurrency")]
    pub concurrency: usize,
    /// Per-execution wall-clock timeout, in seconds.
    #[serde(default = "defaults::cores_execution_timeout_seconds")]
    pub execution_timeout_seconds: u64,
    #[serde(default)]
    pub discovery: CoresDiscovery,
}

impl Default for CoresConfig {
    fn default() -> Self {
        Self {
            selected: Vec::new(),
            concurrency: defaults::cores_concurrency(),
            execution_timeout_seconds: defaults::cores_execution_timeout_seconds(),
            discovery: CoresDiscovery::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoresDiscovery {
    /// Directories scanned for core plugin shared libraries. Falls back to
    /// the built-in defaults when empty.
    #[serde(default = "defaults::cores_default_search_directories")]
    pub search_directories: Vec<PathBuf>,
    /// Explicit shared-library paths to load in addition to the directory
    /// scan. Always loaded if present.
    #[serde(default)]
    pub explicit_plugin_files: Vec<PathBuf>,
}

impl Default for CoresDiscovery {
    fn default() -> Self {
        Self {
            search_directories: defaults::cores_default_search_directories(),
            explicit_plugin_files: Vec::new(),
        }
    }
}
