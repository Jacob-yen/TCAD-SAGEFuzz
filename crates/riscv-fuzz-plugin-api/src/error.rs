use thiserror::Error;

pub type BoxedExecutionError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("empty string provided where character expected")]
    EmptyString,

    #[error("invalid register name format: {name}")]
    InvalidRegisterName { name: String },

    #[error("failed to parse hex value '{value}': {source}")]
    HexParseError {
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("failed to parse integer value '{value}': {source}")]
    IntParseError {
        value: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("failed to parse program counter from '{text}'")]
    PcParseError { text: String },

    #[error("failed to parse register number from '{text}'")]
    RegisterNumberParseError { text: String },

    #[error("failed to parse memory address from '{text}'")]
    AddressParseError { text: String },

    #[error("failed to parse value from '{text}'")]
    ValueParseError { text: String },
}

#[derive(Debug, Error)]
pub enum LogParseError {
    #[error("failed to compile regex pattern '{pattern}': {source}")]
    RegexCompilationFailed {
        pattern: String,
        #[source]
        source: regex::Error,
    },

    #[error("failed to match pattern in line: {line}")]
    PatternMatchFailed { line: String },

    #[error("missing required capture group '{group}' in line: {line}")]
    MissingCaptureGroup { group: String, line: String },

    #[error("invalid line format: {line}")]
    InvalidLineFormat { line: String },

    #[error("failed to parse value from capture: {source}")]
    CaptureParseError {
        #[source]
        source: ParseError,
    },

    #[error("failed to read log file at {path}: {source}")]
    LogFileReadError {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("no PC mapping found for trace instruction local index {index}")]
    NoPcMappingFound { index: usize },
}

#[derive(Debug, Error)]
pub enum ProcessError {
    #[error("failed to spawn process '{command}': {source}")]
    SpawnFailed {
        command: String,
        #[source]
        source: std::io::Error,
    },

    #[error("process '{command}' failed with stderr: {stderr}")]
    ProcessFailed { command: String, stderr: String },

    #[error("process '{command}' timed out after {timeout:?}")]
    TimedOut {
        command: String,
        timeout: std::time::Duration,
    },

    #[error("failed to create directory at {path}: {source}")]
    DirectoryCreationFailed {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to write file at {path}: {source}")]
    FileWriteFailed {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to open log file at {path}: {source}")]
    LogFileOpenFailed {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

#[derive(Debug, Error)]
pub enum ElfError {
    #[error("failed to load ELF dump from {path}: {source}")]
    DumpLoadFailed {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to build ELF: {source}")]
    BuildFailed {
        #[source]
        source: BoxedExecutionError,
    },
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("ISA base {isa_base} is not supported by {core_id}")]
    UnsupportedIsaBase { core_id: String, isa_base: String },

    #[error("invalid configuration: {message}")]
    InvalidConfiguration { message: String },
}

#[derive(Debug, Error)]
pub enum ImplExecutionError {
    #[error(transparent)]
    Process(#[from] ProcessError),

    #[error(transparent)]
    LogParse(#[from] LogParseError),

    #[error(transparent)]
    Elf(#[from] ElfError),

    #[error(transparent)]
    Parse(#[from] ParseError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] BoxedExecutionError),

    #[error("generic error: {0}")]
    Generic(String),
}

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("implementation plugin {core_id} is not available")]
    LookupFailed { core_id: String },

    #[error("implementation {core_id} does not support ISA base {isa_base}")]
    UnsupportedIsaBase { core_id: String, isa_base: String },

    #[error(
        "implementation {core_id} does not support the requested alignment mode ({alignment_mode:?})"
    )]
    UnsupportedAlignmentMode {
        core_id: String,
        alignment_mode: crate::AlignmentMode,
    },

    #[error("execution failed for implementation {core_id} with ISA base {isa_base}: {source}")]
    ImplementationFailed {
        core_id: String,
        isa_base: String,
        #[source]
        source: ImplExecutionError,
    },
}
