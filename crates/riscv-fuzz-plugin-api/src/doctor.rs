use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Duration,
};

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{CoverageMode, ISABase, IsaExecutionProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PluginDoctorStatus {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginDoctorCheck {
    pub label: String,
    pub env_var: Option<String>,
    pub status: PluginDoctorStatus,
    pub value: String,
    pub note: String,
}

impl PluginDoctorCheck {
    pub fn pass_env(
        label: impl Into<String>,
        env_var: impl Into<String>,
        value: impl Into<String>,
        note: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            env_var: Some(env_var.into()),
            status: PluginDoctorStatus::Pass,
            value: value.into(),
            note: note.into(),
        }
    }

    pub fn warn_env(
        label: impl Into<String>,
        env_var: impl Into<String>,
        value: impl Into<String>,
        note: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            env_var: Some(env_var.into()),
            status: PluginDoctorStatus::Warn,
            value: value.into(),
            note: note.into(),
        }
    }

    pub fn fail_env(
        label: impl Into<String>,
        env_var: impl Into<String>,
        value: impl Into<String>,
        note: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            env_var: Some(env_var.into()),
            status: PluginDoctorStatus::Fail,
            value: value.into(),
            note: note.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginDoctorReport {
    pub plugin_id: String,
    pub checks: Vec<PluginDoctorCheck>,
}

impl PluginDoctorReport {
    pub fn new(plugin_id: impl Into<String>, checks: Vec<PluginDoctorCheck>) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            checks,
        }
    }

    pub fn empty(plugin_id: impl Into<String>) -> Self {
        Self::new(plugin_id, Vec::new())
    }

    pub fn overall_status(&self) -> PluginDoctorStatus {
        self.checks
            .iter()
            .map(|check| check.status)
            .max()
            .unwrap_or(PluginDoctorStatus::Pass)
    }
}

pub fn supported_coverage_modes(profile: &IsaExecutionProfile) -> Vec<CoverageMode> {
    let mut modes = Vec::new();
    for mode in [CoverageMode::Off, CoverageMode::Full, CoverageMode::Light] {
        if profile.supports_coverage_mode(mode) {
            modes.push(mode);
        }
    }
    modes
}

pub fn coverage_requirement_label(mode: CoverageMode) -> &'static str {
    match mode {
        CoverageMode::Off => "coverage=off",
        CoverageMode::Full => "coverage=full",
        CoverageMode::Light => "coverage=light",
    }
}

pub fn isa_label(isa: ISABase) -> &'static str {
    match isa {
        ISABase::Rv32 => "rv32",
        ISABase::Rv64 => "rv64",
    }
}

pub fn push_coverage_mode_checks<F>(
    checks: &mut Vec<PluginDoctorCheck>,
    profile: &IsaExecutionProfile,
    mut build: F,
) where
    F: FnMut(CoverageMode) -> Vec<PluginDoctorCheck>,
{
    for mode in supported_coverage_modes(profile) {
        checks.extend(build(mode));
    }
}

pub fn push_single_and_optional_dual_hart_binary_checks(
    checks: &mut Vec<PluginDoctorCheck>,
    profile: &IsaExecutionProfile,
    isa: ISABase,
    single_hart_env_var: impl Fn(CoverageMode) -> &'static str,
    single_hart_note: &'static str,
    dual_hart_env_var: impl Fn(CoverageMode) -> &'static str,
    dual_hart_note: &'static str,
) {
    push_coverage_mode_checks(checks, profile, |mode| {
        let mut mode_checks = vec![required_executable_path_check(
            format!(
                "{} 1-hart {}",
                isa_label(isa),
                coverage_requirement_label(mode)
            ),
            single_hart_env_var(mode),
            single_hart_note,
        )];
        if profile.supports_dual_hart {
            mode_checks.push(required_executable_path_check(
                format!(
                    "{} 2-hart {}",
                    isa_label(isa),
                    coverage_requirement_label(mode)
                ),
                dual_hart_env_var(mode),
                dual_hart_note,
            ));
        }
        mode_checks
    });
}

pub fn push_labeled_binary_triplet_checks(
    checks: &mut Vec<PluginDoctorCheck>,
    profile: &IsaExecutionProfile,
    hart_label: &'static str,
    env_triplet: impl Fn(CoverageMode) -> [(&'static str, &'static str); 3],
    note: &'static str,
) {
    push_coverage_mode_checks(checks, profile, |mode| {
        env_triplet(mode)
            .into_iter()
            .map(|(label, env_var)| {
                required_executable_path_check(
                    format!("{label} {hart_label} {}", coverage_requirement_label(mode)),
                    env_var,
                    note,
                )
            })
            .collect()
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloatProfileBinaryCheckSpec<Family> {
    pub family: Family,
    pub label_suffix: &'static str,
    pub single_hart_note: &'static str,
    pub dual_hart_note: &'static str,
}

pub fn float_profile_requirement_label(
    isa: ISABase,
    hart_label: &'static str,
    mode: CoverageMode,
    label_suffix: &'static str,
) -> String {
    format!(
        "{} {hart_label} {} {label_suffix}",
        isa_label(isa),
        coverage_requirement_label(mode)
    )
}

pub fn push_float_profile_binary_checks<Family: Copy>(
    checks: &mut Vec<PluginDoctorCheck>,
    profile: &IsaExecutionProfile,
    isa: ISABase,
    specs: &[FloatProfileBinaryCheckSpec<Family>],
    family_supported: impl Fn(&IsaExecutionProfile, Family) -> bool,
    mut single_hart_check: impl FnMut(
        ISABase,
        CoverageMode,
        FloatProfileBinaryCheckSpec<Family>,
    ) -> PluginDoctorCheck,
    mut dual_hart_check: impl FnMut(
        ISABase,
        CoverageMode,
        FloatProfileBinaryCheckSpec<Family>,
    ) -> Option<PluginDoctorCheck>,
) {
    push_coverage_mode_checks(checks, profile, |mode| {
        let mut mode_checks = Vec::new();
        for spec in specs.iter().copied() {
            if !family_supported(profile, spec.family) {
                continue;
            }
            mode_checks.push(single_hart_check(isa, mode, spec));
            if profile.supports_dual_hart
                && let Some(check) = dual_hart_check(isa, mode, spec)
            {
                mode_checks.push(check);
            }
        }
        mode_checks
    });
}

pub fn required_executable_path_check(
    label: impl Into<String>,
    env_var: &'static str,
    note: impl Into<String>,
) -> PluginDoctorCheck {
    required_executable_path_check_with_validation(label, env_var, note, |_| Ok(()))
}

pub fn required_executable_path_check_with_validation<F>(
    label: impl Into<String>,
    env_var: &'static str,
    note: impl Into<String>,
    validate: F,
) -> PluginDoctorCheck
where
    F: FnOnce(&Path) -> Result<(), String>,
{
    let label = label.into();
    let note = note.into();
    match env::var_os(env_var) {
        None => PluginDoctorCheck::fail_env(
            label,
            env_var,
            "_unset_",
            format!("{note}; environment variable is not set"),
        ),
        Some(raw) if raw.is_empty() => PluginDoctorCheck::fail_env(
            label,
            env_var,
            "",
            format!("{note}; environment variable is empty"),
        ),
        Some(raw) => {
            let value = raw.to_string_lossy().into_owned();
            let path = PathBuf::from(&raw);
            if !path.exists() {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!("{note}; path does not exist"),
                );
            }
            if !path.is_file() {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!("{note}; path exists but is not a file"),
                );
            }
            if !is_executable_file(&path) {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!("{note}; file exists but is not executable"),
                );
            }
            if let Err(error) = validate(&path) {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!("{note}; {error}"),
                );
            }

            PluginDoctorCheck::pass_env(
                label,
                env_var,
                value,
                format!("{note}; verified executable {}", canonical_or_raw(&path)),
            )
        }
    }
}

pub fn required_executable_path_or_command_check(
    label: impl Into<String>,
    env_var: &'static str,
    note: impl Into<String>,
) -> PluginDoctorCheck {
    let label = label.into();
    let note = note.into();
    match env::var_os(env_var) {
        None => PluginDoctorCheck::fail_env(
            label,
            env_var,
            "_unset_",
            format!("{note}; environment variable is not set"),
        ),
        Some(raw) if raw.is_empty() => PluginDoctorCheck::fail_env(
            label,
            env_var,
            "",
            format!("{note}; environment variable is empty"),
        ),
        Some(raw) => {
            let value = raw.to_string_lossy().into_owned();
            let candidate = PathBuf::from(&raw);
            if looks_like_command_name(&candidate) {
                if let Some(resolved) = resolve_command_in_path(&candidate) {
                    if !is_executable_file(&resolved) {
                        return PluginDoctorCheck::fail_env(
                            label,
                            env_var,
                            value,
                            format!(
                                "{note}; command resolved to {} but the file is not executable",
                                resolved.display()
                            ),
                        );
                    }
                    return PluginDoctorCheck::pass_env(
                        label,
                        env_var,
                        value,
                        format!("{note}; resolved via PATH to {}", resolved.display()),
                    );
                }
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!("{note}; command name was not found in PATH"),
                );
            }

            required_executable_path_check(label, env_var, note)
        }
    }
}

pub fn required_file_check(
    label: impl Into<String>,
    env_var: &'static str,
    note: impl Into<String>,
) -> PluginDoctorCheck {
    let label = label.into();
    let note = note.into();
    match env::var_os(env_var) {
        None => PluginDoctorCheck::fail_env(
            label,
            env_var,
            "_unset_",
            format!("{note}; environment variable is not set"),
        ),
        Some(raw) if raw.is_empty() => PluginDoctorCheck::fail_env(
            label,
            env_var,
            "",
            format!("{note}; environment variable is empty"),
        ),
        Some(raw) => {
            let value = raw.to_string_lossy().into_owned();
            let path = PathBuf::from(&raw);
            if !path.exists() {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!("{note}; path does not exist"),
                );
            }
            if !path.is_file() {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!("{note}; path exists but is not a file"),
                );
            }
            PluginDoctorCheck::pass_env(
                label,
                env_var,
                value,
                format!("{note}; verified file {}", canonical_or_raw(&path)),
            )
        }
    }
}

pub fn required_directory_with_executable_check(
    label: impl Into<String>,
    env_var: &'static str,
    relative_executable: &'static str,
    note: impl Into<String>,
) -> PluginDoctorCheck {
    let label = label.into();
    let note = note.into();
    match env::var_os(env_var) {
        None => PluginDoctorCheck::fail_env(
            label,
            env_var,
            "_unset_",
            format!("{note}; environment variable is not set"),
        ),
        Some(raw) if raw.is_empty() => PluginDoctorCheck::fail_env(
            label,
            env_var,
            "",
            format!("{note}; environment variable is empty"),
        ),
        Some(raw) => {
            let value = raw.to_string_lossy().into_owned();
            let root = PathBuf::from(&raw);
            if !root.exists() {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!("{note}; directory does not exist"),
                );
            }
            if !root.is_dir() {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!("{note}; path exists but is not a directory"),
                );
            }
            let executable = root.join(relative_executable);
            if !executable.is_file() {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!(
                        "{note}; required tool {} is missing under the configured root",
                        executable.display()
                    ),
                );
            }
            if !is_executable_file(&executable) {
                return PluginDoctorCheck::fail_env(
                    label,
                    env_var,
                    value,
                    format!(
                        "{note}; required tool {} exists but is not executable",
                        executable.display()
                    ),
                );
            }
            PluginDoctorCheck::pass_env(
                label,
                env_var,
                value,
                format!("{note}; verified tool {}", canonical_or_raw(&executable)),
            )
        }
    }
}

pub fn parse_global_cycles_from_text(text: &str) -> Option<u64> {
    static RE_LIST: Lazy<Vec<Regex>> = Lazy::new(|| {
        vec![
            Regex::new(r"Simulation terminated after\s+([0-9][0-9_,]*)\s+cycles").unwrap(),
            Regex::new(r"Completed after\s+([0-9][0-9_,]*)\s+cycles").unwrap(),
            Regex::new(r"\bafter\s+([0-9][0-9_,]*)\s+cycles\b").unwrap(),
            Regex::new(r"Executed cycles\s*[:=]\s*([0-9][0-9_,]*)").unwrap(),
            Regex::new(r"Had simulate\s+.*?\s+([0-9][0-9_,]*)\s+clock cycles").unwrap(),
            Regex::new(r"cycleCnt\s*[:=]\s*([0-9][0-9_,]*)").unwrap(),
            Regex::new(r"(?m)(?:^|\s)Cycles\s*[:=]\s*([0-9][0-9_,]*)").unwrap(),
            Regex::new(r"(?m)(?:^|\s)cycles\s*[:=]\s*([0-9][0-9_,]*)").unwrap(),
        ]
    });

    let mut best: Option<u64> = None;
    for re in RE_LIST.iter() {
        for cap in re.captures_iter(text) {
            if let Some(raw) = cap.get(1) {
                let normalized: String = raw
                    .as_str()
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect();
                if let Ok(value) = normalized.parse::<u64>() {
                    best = Some(best.map(|b| b.max(value)).unwrap_or(value));
                }
            }
        }
    }
    best
}

pub fn optional_positive_u64_check(
    label: impl Into<String>,
    env_var: &'static str,
    ignored_because: Option<&'static str>,
    note: impl Into<String>,
) -> PluginDoctorCheck {
    let label = label.into();
    let note = note.into();
    let raw = env::var_os(env_var);
    let value = raw
        .as_ref()
        .map(|raw| raw.to_string_lossy().into_owned())
        .unwrap_or_else(|| "_unset_".to_string());

    if let Some(reason) = ignored_because {
        return PluginDoctorCheck::pass_env(label, env_var, value, format!("{note}; {reason}"));
    }

    let Some(raw) = raw else {
        return PluginDoctorCheck::pass_env(
            label,
            env_var,
            value,
            format!("{note}; unset, so the plugin default timeout will be used"),
        );
    };

    if raw.is_empty() {
        return PluginDoctorCheck::fail_env(
            label,
            env_var,
            "",
            format!("{note}; environment variable is empty"),
        );
    }

    let trimmed = raw.to_string_lossy().trim().to_string();
    match trimmed.parse::<u64>() {
        Ok(secs) if secs > 0 => PluginDoctorCheck::pass_env(
            label,
            env_var,
            trimmed,
            format!("{note}; parsed effective timeout override of {secs} second(s)"),
        ),
        Ok(_) => PluginDoctorCheck::fail_env(
            label,
            env_var,
            trimmed,
            format!("{note}; value must be greater than zero"),
        ),
        Err(error) => PluginDoctorCheck::fail_env(
            label,
            env_var,
            trimmed,
            format!("{note}; failed to parse as u64: {error}"),
        ),
    }
}

pub fn execution_timeout_override_check(env_var: &'static str) -> PluginDoctorCheck {
    optional_positive_u64_check(
        "timeout override",
        env_var,
        None,
        "Overrides the plugin timeout when set; must be a non-zero integer number of seconds",
    )
}

pub fn push_execution_timeout_override_check(
    checks: &mut Vec<PluginDoctorCheck>,
    env_var: &'static str,
) {
    checks.push(execution_timeout_override_check(env_var));
}

pub fn parse_optional_nonzero_timeout_secs_env(
    env_var: &'static str,
) -> Result<Option<Duration>, String> {
    let Some(raw) = env::var_os(env_var) else {
        return Ok(None);
    };
    if raw.is_empty() {
        return Err(format!("{env_var} must not be empty"));
    }

    let trimmed = raw.to_string_lossy().trim().to_string();
    if trimmed.is_empty() {
        return Err(format!("{env_var} must not be empty"));
    }

    let secs = trimmed
        .parse::<u64>()
        .map_err(|err| format!("failed to parse {env_var}='{trimmed}' as u64: {err}"))?;
    if secs == 0 {
        return Err(format!("{env_var} must be greater than zero"));
    }

    Ok(Some(Duration::from_secs(secs)))
}

pub fn is_executable_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path)
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }

    #[cfg(not(unix))]
    {
        true
    }
}

pub fn resolve_command_in_path(command: &Path) -> Option<PathBuf> {
    let path_env = env::var_os("PATH")?;
    for directory in env::split_paths(&path_env) {
        let candidate = directory.join(command);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

pub fn canonical_or_raw(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn looks_like_command_name(path: &Path) -> bool {
    !path.is_absolute() && path.components().count() == 1
}
