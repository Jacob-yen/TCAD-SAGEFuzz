use thiserror::Error;

use riscv_fuzz_plugin_api::{AlignmentMode, CoverageMode, ISABase, ImplExecutionError};

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum LookupError {
    #[error("RISC-V implementation plugin `{core_id}` is not available")]
    PluginNotFound { core_id: String },
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum CapabilityError {
    #[error(transparent)]
    Lookup(#[from] LookupError),

    #[error("implementation {core_id} does not support ISA base {isa_base:?}")]
    UnsupportedIsaBase { core_id: String, isa_base: ISABase },

    #[error(
        "implementation {core_id} does not support coverage mode {coverage_mode:?} for ISA base {isa_base:?}"
    )]
    UnsupportedCoverageMode {
        core_id: String,
        isa_base: ISABase,
        coverage_mode: CoverageMode,
    },

    #[error(
        "implementation {core_id} does not support hart count {hart_count} for ISA base {isa_base:?}"
    )]
    UnsupportedHartCount {
        core_id: String,
        isa_base: ISABase,
        hart_count: usize,
    },

    #[error(
        "implementation {core_id} does not support the requested alignment mode ({alignment_mode:?}) for ISA base {isa_base:?}"
    )]
    UnsupportedAlignmentMode {
        core_id: String,
        isa_base: ISABase,
        alignment_mode: AlignmentMode,
    },

    #[error("implementation {core_id} rejected the execute request: {message}")]
    InvalidRequest { core_id: String, message: String },
}

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error(transparent)]
    Lookup(#[from] LookupError),

    #[error(transparent)]
    Capability(#[from] CapabilityError),

    #[error("execution failed for implementation {core_id} with ISA base {isa_base:?}: {source}")]
    ImplementationFailed {
        core_id: String,
        isa_base: ISABase,
        #[source]
        source: ImplExecutionError,
    },
}
