use std::{
    collections::HashSet,
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginDiscoverySource {
    ExplicitFileEnv,
    SearchDirectory,
}

#[derive(Debug, Clone)]
pub struct PluginDirectoryScan {
    pub path: PathBuf,
    pub exists: bool,
    pub read_error: Option<String>,
    pub candidate_entries: usize,
}

#[derive(Debug, Clone)]
pub struct PluginDiscoveryCandidate {
    pub path: PathBuf,
    pub source: PluginDiscoverySource,
    pub exists: bool,
    pub is_file: bool,
    pub matches_directory_naming_heuristic: bool,
    pub stem: Option<String>,
    pub selected: bool,
    pub skip_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PluginDiscoveryReport {
    pub plugin_files_env: Option<String>,
    pub plugin_dirs_env: Option<String>,
    pub configured_dirs: Vec<PluginDirectoryScan>,
    pub candidates: Vec<PluginDiscoveryCandidate>,
}

impl PluginDiscoveryReport {
    pub fn selected_candidate_paths(&self) -> Vec<PathBuf> {
        self.candidates
            .iter()
            .filter(|candidate| candidate.selected)
            .map(|candidate| candidate.path.clone())
            .collect()
    }
}

pub fn plugin_discovery_report() -> PluginDiscoveryReport {
    let mut seen_paths = HashSet::new();
    let mut seen_stems = HashSet::new();
    let mut candidates = Vec::new();
    let plugin_files_env = env_var_lossy("RISCV_FUZZ_PLUGIN_FILES");
    let plugin_dirs_env = env_var_lossy("RISCV_FUZZ_PLUGIN_DIRS");

    for file in split_path_var("RISCV_FUZZ_PLUGIN_FILES") {
        candidates.push(classify_explicit_file_candidate(
            file,
            &mut seen_paths,
            &mut seen_stems,
        ));
    }

    let mut configured_dirs = Vec::new();
    for dir in configured_plugin_dirs() {
        let mut scan = PluginDirectoryScan {
            path: dir.clone(),
            exists: dir.exists(),
            read_error: None,
            candidate_entries: 0,
        };

        if !dir.exists() {
            configured_dirs.push(scan);
            continue;
        }

        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) => {
                scan.read_error = Some(error.to_string());
                configured_dirs.push(scan);
                continue;
            }
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    scan.read_error = Some(format!(
                        "failed to read directory entry under {}: {error}",
                        dir.display()
                    ));
                    continue;
                }
            };
            let path = entry.path();

            if path.is_file() && is_candidate_plugin_library(&path) {
                scan.candidate_entries += 1;
                candidates.push(classify_directory_candidate(
                    path,
                    &mut seen_paths,
                    &mut seen_stems,
                ));
            }
        }

        configured_dirs.push(scan);
    }

    PluginDiscoveryReport {
        plugin_files_env,
        plugin_dirs_env,
        configured_dirs,
        candidates,
    }
}

fn classify_explicit_file_candidate(
    path: PathBuf,
    seen_paths: &mut HashSet<PathBuf>,
    seen_stems: &mut HashSet<String>,
) -> PluginDiscoveryCandidate {
    let exists = path.exists();
    let is_file = path.is_file();
    let matches_directory_naming_heuristic = is_candidate_plugin_library(&path);

    let (stem, selected, skip_reason) =
        select_candidate(&path, is_file, seen_paths, seen_stems, true);

    PluginDiscoveryCandidate {
        path,
        source: PluginDiscoverySource::ExplicitFileEnv,
        exists,
        is_file,
        matches_directory_naming_heuristic,
        stem,
        selected,
        skip_reason,
    }
}

fn classify_directory_candidate(
    path: PathBuf,
    seen_paths: &mut HashSet<PathBuf>,
    seen_stems: &mut HashSet<String>,
) -> PluginDiscoveryCandidate {
    let exists = path.exists();
    let is_file = path.is_file();
    let matches_directory_naming_heuristic = is_candidate_plugin_library(&path);

    let (stem, selected, skip_reason) =
        select_candidate(&path, is_file, seen_paths, seen_stems, false);

    PluginDiscoveryCandidate {
        path,
        source: PluginDiscoverySource::SearchDirectory,
        exists,
        is_file,
        matches_directory_naming_heuristic,
        stem,
        selected,
        skip_reason,
    }
}

fn select_candidate(
    path: &Path,
    is_file: bool,
    seen_paths: &mut HashSet<PathBuf>,
    seen_stems: &mut HashSet<String>,
    explicit_env_file: bool,
) -> (Option<String>, bool, Option<String>) {
    if !is_file {
        return (None, false, Some("path is not a file".to_string()));
    }

    let stem = match normalized_plugin_stem(path) {
        Ok(stem) => stem,
        Err(error) => return (None, false, Some(error)),
    };

    if !seen_paths.insert(path.to_path_buf()) {
        return (
            Some(stem),
            false,
            Some("duplicate path ignored after an earlier match".to_string()),
        );
    }

    if !seen_stems.insert(stem.clone()) {
        return (
            Some(stem.clone()),
            true,
            Some(format!(
                "duplicate plugin stem `{stem}` retained as a fallback candidate after an earlier match"
            )),
        );
    }

    let skip_reason = if explicit_env_file {
        None
    } else {
        debug_assert!(is_candidate_plugin_library(path));
        None
    };

    (Some(stem), true, skip_reason)
}

fn env_var_lossy(name: &str) -> Option<String> {
    env::var_os(name).map(|value| value.to_string_lossy().into_owned())
}

fn configured_plugin_dirs() -> Vec<PathBuf> {
    let configured = split_path_var("RISCV_FUZZ_PLUGIN_DIRS");
    if !configured.is_empty() {
        return configured;
    }

    riscv_fuzz_plugin_api::default_plugin_search_directories()
}

fn split_path_var(name: &str) -> Vec<PathBuf> {
    env::var_os(name)
        .map(|value| env::split_paths(&value).collect())
        .unwrap_or_default()
}

fn is_candidate_plugin_library(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };

    let Some(extension) = path.extension().and_then(OsStr::to_str) else {
        return false;
    };

    file_name.contains("riscv_fuzz_core_")
        && extension.eq_ignore_ascii_case(dynamic_lib_extension())
}

pub(super) fn normalized_plugin_stem(path: &Path) -> Result<String, String> {
    let stem = path.file_stem().and_then(OsStr::to_str).ok_or_else(|| {
        format!(
            "failed to extract plugin library stem from {}",
            path.display()
        )
    })?;
    Ok(stem.strip_prefix("lib").unwrap_or(stem).to_string())
}

fn dynamic_lib_extension() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "dll"
    }

    #[cfg(target_os = "macos")]
    {
        "dylib"
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        "so"
    }
}
