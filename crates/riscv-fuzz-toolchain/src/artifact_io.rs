use std::{fs, io, path::Path};

use serde::Serialize;

pub fn ensure_parent_dir(path: impl AsRef<Path>) -> io::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub fn write_string_file(path: impl AsRef<Path>, content: impl AsRef<str>) -> io::Result<()> {
    ensure_parent_dir(&path)?;
    fs::write(path, content.as_ref())
}

pub fn write_json_pretty<T: Serialize + ?Sized>(
    path: impl AsRef<Path>,
    value: &T,
) -> Result<(), ArtifactIoError> {
    ensure_parent_dir(&path)?;
    let file = fs::File::create(path)?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum ArtifactIoError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}
