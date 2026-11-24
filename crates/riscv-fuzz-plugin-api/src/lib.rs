pub mod abi;
pub mod context;
pub mod coverage;
pub mod doctor;
pub mod error;
pub mod execution;
pub mod isa;
pub mod metadata;
pub mod plugin;
pub mod process_tree;
mod protocol;
pub mod register_name;
pub mod streaming;

use std::path::PathBuf;

pub fn default_plugin_search_directories() -> Vec<PathBuf> {
    vec![PathBuf::from("target/release")]
}

pub use context::{
    AsmInstructionContextOutput, ExecutionContextOutput, HartExecutionContextOutput,
    InstructionContext, LogicalInstructionContextOutput, MemoryContext, MemoryContextEntry,
    MemoryValueWidth, RegisterContext, RegisterContextEntry,
};
pub use coverage::CoverageMode;
pub use doctor::{
    PluginDoctorCheck, PluginDoctorReport, PluginDoctorStatus, parse_global_cycles_from_text,
};
pub use error::{
    BoxedExecutionError, ConfigError, ElfError, ImplExecutionError, LogParseError, ParseError,
    ProcessError,
};
pub use execution::{
    AsmInstructionExecution, ExceptionCause, ExecutionOutput, HartExecutionOutput, HartPerfMetrics,
    InstructionTiming, MemValue, NormalizeError, PerfMetrics, RegisterValue, TraceInstId,
    UNKNOWN_EXCEPTION_CODE,
};
pub use isa::{ISABase, IsaExtensions, ParseISABaseError};
pub use metadata::{
    AlignmentMode, CorePluginMetadata, CoverageModeSupport, FloatExtension, IsaExecutionProfile,
    IsaExecutionProfileMap, UnalignedAccessSupport,
};
pub use plugin::{
    ExecuteRequest, PerHartTypedSequences, RiscvCorePlugin, build_per_hart_asm,
    execute_with_config, flattened_sequence_instruction_count, typed_sequences_for_hart,
    validate_execute_request_with_config, validate_typed_execution_output,
};
pub use process_tree::{configure_command_process_group, kill_child_process_group};
pub use protocol::PluginMetadataRecord;
pub use register_name::{ParsedRegisterName, canonicalize_register_name, parse_register_name};
pub use streaming::{
    InstructionEvent, StreamingCollector, StreamingOutputDelta, complete_line_prefix_len,
    filter_hart_output_by_timing, filter_per_hart_output_by_timing, read_complete_utf8_prefix,
};
