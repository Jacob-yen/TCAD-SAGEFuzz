//! File parsing, overlay merging, and CLI scalar overrides.
//!
//! # Precedence
//!
//! Lowest → highest:
//!
//! 1. Compile-time defaults from [`crate::defaults`]
//! 2. The base file passed to [`load_experiment_config`] as `base`
//! 3. Each overlay in `overlays` (later overlays win section-by-section)
//! 4. CLI scalar overrides in `overrides`
//!
//! Overlays are merged at the *section* level: if an overlay contains
//! `[testcase_config]`, the base's `[testcase_config]` is replaced wholesale.
//! There is no deep merging within a section. Anything an overlay does not
//! mention is taken from the previous layer. This rule is intentionally simple
//! — when an operator wants to tweak two unrelated knobs they should put them in the
//! same overlay file rather than relying on field-level merging that is hard to
//! reason about.
//!
//! # Validation
//!
//! After all layers are applied, [`crate::validate_structural`] runs the
//! I/O-free rules R1–R10/R12. R11 (workload ↔ core compatibility) requires
//! plugin discovery and is not run here; callers should invoke
//! [`crate::validate::validate_against_runtime_metadata`] once cores have been
//! loaded.

use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;
use toml::Value;

use crate::schema::ExperimentConfig;
use crate::validate::{ValidationError, validate_structural};

/// Whitelist of top-level sections an overlay may declare. Any other key in
/// an overlay file is rejected so a typo doesn't silently get ignored.
const TOP_LEVEL_SECTIONS: &[&str] = &["testcase_config", "runner", "reporting", "cores"];

/// CLI scalar overrides applied last, on top of base + overlays.
///
/// Only high-frequency knobs that are awkward to keep in a file (typically
/// changed per-invocation) are exposed here. Everything else must live in a
/// TOML file so configurations are reproducible and reviewable.
#[derive(Debug, Default, Clone)]
pub struct CliOverrides {
    /// Override `cores.selected`. When `Some(..)`, replaces the file value.
    pub cores: Option<Vec<String>>,
    /// Override `cores.concurrency`.
    pub concurrency: Option<usize>,
    /// Override the active runner mode's `max_iterations`. Only meaningful
    /// for `random_diff`, `diff_against_spike`, and `coverage_guided` modes.
    /// For other modes this override is silently ignored.
    pub max_iterations: Option<usize>,
    /// Override `runner.seed`.
    pub seed: Option<u64>,
    /// Working directory for run artifacts. Not part of the schema (it would
    /// not be meaningful to commit to a file); kept here so callers can
    /// thread it through with the rest of the override surface.
    pub run_dir: Option<PathBuf>,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse TOML {path}: {source}")]
    ParseToml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error(
        "config {path}: top-level section `{section}` is not recognized; expected one of {expected:?}"
    )]
    UnknownTopLevelSection {
        path: PathBuf,
        section: String,
        expected: &'static [&'static str],
    },
    #[error("config {path}: top-level value `{key}` is not a TOML table")]
    NonTableTopLevel { path: PathBuf, key: String },
    #[error("failed to materialize config from merged TOML: {source}")]
    Materialize {
        #[source]
        source: toml::de::Error,
    },
    #[error(transparent)]
    Validation(#[from] ValidationError),
}

/// Load and validate an experiment configuration.
pub fn load_experiment_config(
    base: &Path,
    overlays: &[PathBuf],
    overrides: &CliOverrides,
) -> Result<ExperimentConfig, ConfigError> {
    let mut merged = read_toml_table(base)?;
    enforce_top_level_keys(&merged, base)?;

    for overlay in overlays {
        let layer = read_toml_table(overlay)?;
        enforce_top_level_keys(&layer, overlay)?;
        for (key, value) in layer {
            merged.insert(key, value);
        }
    }

    let mut cfg: ExperimentConfig = Value::Table(merged)
        .try_into()
        .map_err(|source| ConfigError::Materialize { source })?;

    apply_overrides(&mut cfg, overrides);
    validate_structural(&cfg)?;
    Ok(cfg)
}

/// Parse a stand-alone configuration string (no overlays, no overrides). Useful
/// in tests and for snapshot fixtures.
pub fn parse_experiment_config(content: &str) -> Result<ExperimentConfig, ConfigError> {
    let value: Value = toml::from_str(content).map_err(|source| ConfigError::ParseToml {
        path: PathBuf::from("<inline>"),
        source,
    })?;
    let table = match value {
        Value::Table(t) => t,
        _ => {
            return Err(ConfigError::NonTableTopLevel {
                path: PathBuf::from("<inline>"),
                key: "<root>".to_string(),
            });
        }
    };
    enforce_top_level_keys(&table, Path::new("<inline>"))?;
    let cfg: ExperimentConfig = Value::Table(table)
        .try_into()
        .map_err(|source| ConfigError::Materialize { source })?;
    validate_structural(&cfg)?;
    Ok(cfg)
}

fn read_toml_table(path: &Path) -> Result<toml::Table, ConfigError> {
    let raw = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let value: Value = toml::from_str(&raw).map_err(|source| ConfigError::ParseToml {
        path: path.to_path_buf(),
        source,
    })?;
    match value {
        Value::Table(t) => Ok(t),
        _ => Err(ConfigError::NonTableTopLevel {
            path: path.to_path_buf(),
            key: "<root>".to_string(),
        }),
    }
}

fn enforce_top_level_keys(table: &toml::Table, path: &Path) -> Result<(), ConfigError> {
    for (key, value) in table {
        if !TOP_LEVEL_SECTIONS.contains(&key.as_str()) {
            return Err(ConfigError::UnknownTopLevelSection {
                path: path.to_path_buf(),
                section: key.clone(),
                expected: TOP_LEVEL_SECTIONS,
            });
        }
        if !matches!(value, Value::Table(_)) {
            return Err(ConfigError::NonTableTopLevel {
                path: path.to_path_buf(),
                key: key.clone(),
            });
        }
    }
    Ok(())
}

fn apply_overrides(cfg: &mut ExperimentConfig, overrides: &CliOverrides) {
    if let Some(selected) = &overrides.cores {
        cfg.cores.selected = selected.clone();
    }
    if let Some(concurrency) = overrides.concurrency {
        cfg.cores.concurrency = concurrency;
    }
    if let Some(seed) = overrides.seed {
        cfg.runner.seed = Some(seed);
    }
    if let Some(max_iters) = overrides.max_iterations {
        if let Some(rd) = cfg.runner.random_diff.as_mut() {
            rd.max_iterations = max_iters;
        }
        if let Some(ds) = cfg.runner.diff_against_spike.as_mut() {
            ds.max_iterations = max_iters;
        }
        if let Some(cov) = cfg.runner.coverage_guided.as_mut() {
            cov.max_iterations = max_iters;
        }
    }
    // `run_dir` is consumed by callers directly; we keep it on `CliOverrides`
    // as a carrier and do not write it into the config struct.
    let _ = &overrides.run_dir;
}
