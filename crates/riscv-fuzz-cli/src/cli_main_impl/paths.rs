use std::path::PathBuf;

use super::{Result, ResultExt};

pub(super) fn absolutize_path(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path);
    }

    let cwd = std::env::current_dir().context("failed to get current working directory")?;
    let joined = cwd.join(&path);
    if joined.exists() {
        joined
            .canonicalize()
            .with_context(|| format!("failed to canonicalize {}", joined.display()))
    } else {
        Ok(joined)
    }
}

pub(super) fn absolutize_optional_path(path: Option<PathBuf>) -> Result<Option<PathBuf>> {
    path.map(absolutize_path).transpose()
}
