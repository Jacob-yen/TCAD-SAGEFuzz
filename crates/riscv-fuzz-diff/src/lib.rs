//! 多实现 RISC-V testcase 归约的差分分析流水线。
//! Differential-analysis reduction pipeline for multi-implementation RISC-V testcases.
//!
//! 对外 API 故意保持很小：
//! The public API is intentionally small:
//! - `compare` 暴露给测试和调用方使用的精确输出/性能对比器。
//!   `compare` exposes exact output/perf comparators used by tests and callers.
//! - 其余模块实现内部归约循环，持续裁剪 testcase，直到定位到异常触发差异和写差异。
//!   The remaining modules implement the internal reduction loop that trims a testcase until
//!   exception-trigger and write differences are localized.

mod compare;
mod dedup;
mod detection;
mod iteration;
mod localization;
mod normalize;
mod report_io;
mod reporting;
mod runner;
mod streaming_execution;
mod types;
mod verification;
mod view;
mod write_replay;

pub use compare::{
    OutputMismatch, collect_perf_collection_issue_notes, compare_execution_views,
    compare_perf_metrics,
};
pub use localization::{LocalizationTiming, localization_probe_get, localization_probe_reset};
pub use runner::{
    run_diff_analysis, run_diff_analysis_with_testcase, verify_initialization_consistency,
};
pub use types::{
    DiffAnalysisConfig, DiffAnalysisError, DiffAnalysisResult, DiffAnalysisSettings,
    DiffDedupSettings,
};
pub use view::DiffExecutionView;
