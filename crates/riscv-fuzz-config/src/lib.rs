//! Unified configuration for riscv-fuzz experiments.
//!
//! This crate owns the operator-facing configuration schema (`ExperimentConfig`),
//! its default values, cross-field validation, and the file/overlay/CLI
//! merge logic. Engine and CLI crates consume the validated `ExperimentConfig`
//! and never parse TOML themselves.

pub mod defaults;
pub mod loader;
pub mod schema;
pub mod validate;

pub use loader::{CliOverrides, ConfigError, load_experiment_config, parse_experiment_config};
pub use riscv_fuzz_testcase_engine::testcase::{
    ExtensionInstScaling, HartConfig, MulticoreSyncConfig, TestCaseConfig,
};
pub use schema::{
    CoresConfig, CoresDiscovery, CoverageGuidedConfig, DiffAgainstSpikeConfig, ExperimentConfig,
    I64Range, RandomDiffConfig, ReplayConfig, ReportingConfig, RunnerConfig, RunnerMode, U8Range,
    UsizeRange, WriteProbeVerificationConfig,
};
pub use validate::{
    ValidationError, validate_against_runtime_metadata, validate_core_runtime_requirements,
    validate_structural,
};

// Re-export the upstream enums so consumers don't need to depend on
// riscv-fuzz-plugin-api just to name the values used in the schema.
pub use riscv_fuzz_plugin_api::{ISABase, coverage::CoverageMode, metadata::FloatExtension};
