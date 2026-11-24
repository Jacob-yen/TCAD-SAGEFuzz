use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BuildMabiError {
    #[error("unsupported ISA base")]
    UnsupportedIsaBase,

    #[error("missing base ABI option")]
    MissingBase,

    #[error("conflicting base ABI options: `{existing}` vs `{requested}`")]
    ConflictingBase { existing: String, requested: String },

    #[error("extension `{ext}` requires `{required}` for ABI selection")]
    FloatRequires {
        ext: &'static str,
        required: &'static str,
    },
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BuildMarchError {
    #[error("unsupported ISA base")]
    UnsupportedIsaBase,

    #[error("missing base ISA extension (expected I or E)")]
    MissingBaseIsa,

    #[error("conflicting base ISA extensions (I and E)")]
    ConflictingBaseIsa,

    #[error("extension `{ext}` requires `{required}`")]
    ExtensionRequires { ext: String, required: String },

    #[error("extensions `{left}` and `{right}` conflict")]
    ExtensionsConflict { left: String, right: String },

    #[error("extension `{ext}` is only valid for rv{required_xlen}")]
    ExtensionOnlyForXlen { ext: String, required_xlen: u32 },

    #[error("extension `{ext}` is not supported by the GNU toolchain")]
    UnsupportedExtension { ext: String },
}

#[derive(Debug, Error)]
pub enum BuildElfError {
    #[error("failed to write linker script at {path}: {source}")]
    LinkerScriptWrite {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to execute `{command}` during {stage}: {source}")]
    CommandSpawn {
        stage: &'static str,
        command: String,
        #[source]
        source: std::io::Error,
    },
    #[error("`{command}` failed during {stage}: {stderr}")]
    CommandFailure {
        stage: &'static str,
        command: String,
        stderr: String,
    },
    #[error("failed to write disassembly output at {path}: {source}")]
    DisassemblyWrite {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to build -march string: {source}")]
    MarchBuild {
        #[source]
        source: BuildMarchError,
    },
    #[error("failed to build -mabi string: {source}")]
    MabiBuild {
        #[source]
        source: BuildMabiError,
    },
}
