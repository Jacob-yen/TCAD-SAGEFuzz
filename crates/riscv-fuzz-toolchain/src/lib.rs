pub mod artifact_io;
pub mod build_elf;
pub mod context_extract;
pub mod coverage;
pub mod error;
pub mod execution_context;
pub mod mabi;
pub mod march;
pub mod trace_instruction;
pub mod tracer;

pub use artifact_io::{ArtifactIoError, ensure_parent_dir, write_json_pretty, write_string_file};
pub use context_extract::{ContextExtractionError, RegexError};
pub use execution_context::{
    ContextBuildError, ExecutionArtifactError, TimelineRenderError, build_execution_context_output,
    build_partial_execution_context_output, emit_raw_execution_artifacts,
    generate_execution_report, generate_execution_timeline_svg_from_output,
    write_raw_execution_output_json,
};
pub use tracer::{
    Tracer, filter_hart_output_for_streaming_by_tracer,
    filter_per_hart_output_for_streaming_by_tracer,
};
