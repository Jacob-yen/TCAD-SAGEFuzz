//! diff-analysis 的公共类型与错误分类。
//! Public types and error taxonomy for diff-analysis.
//!
//! 这些定义会被 CLI、多核编排逻辑以及归约器内部共同使用，
//! These definitions are shared by CLI, multicore orchestration, and the reducer internals,
//! 因此这里的注释更强调行为契约，而不是实现细节。
//! so the comments here focus on behavior contracts rather than implementation detail.

use std::{collections::HashMap, ops::Deref, path::PathBuf};
use thiserror::Error;

use riscv_fuzz_plugin_api::{CoverageMode, ExecutionOutput, IsaExtensions};
use riscv_fuzz_testcase_engine::{
    GenerationError,
    testcase::{GeneratedTestCase, TestCaseConfig},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiffBugCategory {
    /// 在任何实现开始执行之前，随机 testcase 生成就失败了。
    /// Random testcase generation failed before any implementation was executed.
    TestcaseGenerationFailure,
    /// 某个实现的 wrapper 或执行步骤失败了。
    /// An implementation wrapper or execution step failed.
    ExecutionFailure,
    /// 生成工件时发生了文件系统或其他 I/O 失败。
    /// Filesystem or other I/O failed while producing artifacts.
    IoFailure,
    /// testcase 或输出的 JSON 序列化/反序列化失败。
    /// JSON serialization/deserialization failed for testcase or outputs.
    SerializationFailure,
    /// Markdown 或其他派生报告生成失败。
    /// Markdown or derived report emission failed.
    ReportGenerationFailure,
    /// 缺少某个实现必需的指令或执行输出数据。
    /// Required per-implementation instruction/output data was missing.
    MissingInstructions,
    /// 不同实现对逻辑指令数或上下文数量存在分歧。
    /// Implementations disagreed on the number of logical instructions/contexts.
    InstructionCountMismatch,
    /// 写历史最小化无法保留一个有效的目标差异。
    /// Write-history minimization could not preserve a valid target difference.
    HistoryMinimizationFailure,
    /// 测试指令尚未开始前，初始化写入就已经分叉。
    /// Initialization writes diverged before test instructions even started.
    InitializationMismatch,
    /// 归约在配置的迭代预算内没有收敛。
    /// Reduction did not converge within the configured iteration budget.
    IterationLimitReached,
}

impl std::fmt::Display for DiffBugCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DiffBugCategory::TestcaseGenerationFailure => "TESTCASE_GENERATION_FAILURE",
            DiffBugCategory::ExecutionFailure => "EXECUTION_FAILURE",
            DiffBugCategory::IoFailure => "IO_FAILURE",
            DiffBugCategory::SerializationFailure => "SERIALIZATION_FAILURE",
            DiffBugCategory::ReportGenerationFailure => "REPORT_GENERATION_FAILURE",
            DiffBugCategory::MissingInstructions => "MISSING_INSTRUCTIONS",
            DiffBugCategory::InstructionCountMismatch => "INSTRUCTION_COUNT_MISMATCH",
            DiffBugCategory::HistoryMinimizationFailure => "HISTORY_MINIMIZATION_FAILURE",
            DiffBugCategory::InitializationMismatch => "INITIALIZATION_MISMATCH",
            DiffBugCategory::IterationLimitReached => "ITERATION_LIMIT_REACHED",
        })
    }
}

#[derive(Debug, Clone)]
pub struct DiffAnalysisConfig {
    /// 共享的 diff 分析设置（不含输出目录）。
    /// Shared diff-analysis settings (no output directory).
    pub settings: DiffAnalysisSettings,
    /// 写入迭代工件和最终报告的根目录。
    /// Root directory where iteration artifacts and final reports are written.
    pub run_root: PathBuf,
}

impl Deref for DiffAnalysisConfig {
    type Target = DiffAnalysisSettings;
    fn deref(&self) -> &Self::Target {
        &self.settings
    }
}

/// Local persistent de-duplication settings for diff findings.
///
/// A finding still goes through the normal reducer. These settings only control
/// whether the finalized finding signature is written to and matched against a
/// local store so repeated runs can tell "new signal" from "already seen".
#[derive(Debug, Clone)]
pub struct DiffDedupSettings {
    /// Enable automatic finding signatures and local store recording.
    pub enabled: bool,
    /// JSONL database path. When absent, the runner stores records next to the
    /// parent directory that contains per-run workdirs.
    pub database_path: Option<PathBuf>,
}

impl Default for DiffDedupSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            database_path: None,
        }
    }
}

/// 不包含输出目录的 diff-analysis 配置。
/// Diff-analysis settings that do not include the output directory.
///
/// 这是 diff 流程的规范配置形式：CLI 在多个 run 共享同一套逻辑配置、
/// This is the canonical form for diff configuration: CLI shares one logical settings
/// 但每次 run 需要不同工件目录时会用 [`DiffAnalysisSettings::with_run_root`] 把它和 run 目录组合。
/// across multiple runs and uses [`DiffAnalysisSettings::with_run_root`] to attach a per-run directory.
#[derive(Debug, Clone)]
pub struct DiffAnalysisSettings {
    /// 一次 run 内所有迭代共享的 testcase 生成/执行参数。
    /// Testcase generation/execution parameters shared by all iterations in one run.
    pub testcase_config: TestCaseConfig,
    /// 透传给 testcase 执行阶段的 coverage 模式。
    /// Coverage mode passed through to testcase execution.
    pub coverage_mode: CoverageMode,
    /// Per-execution wall-clock timeout in seconds, forwarded to core plugins via
    /// the shared execution-timeout environment contract.
    pub execution_timeout_seconds: u64,
    /// 归约器允许执行的真实归约轮次硬上限。
    /// Hard cap on exception/write reduction rounds performed by the reducer.
    ///
    /// 已被 replay/probe 证伪、仅用于滤除日志/解析噪声的
    /// `LikelyFalsePositive` 不计入这个预算，否则少量误报就会让
    /// 烟雾测试在尚未完成最终 clean rerun 前错误触发
    /// `IterationLimitReached`。
    /// `LikelyFalsePositive` removals that are only filtering logging/parsing
    /// noise do not consume this budget; otherwise a small number of false
    /// positives can make a smoke run hit `IterationLimitReached` before the
    /// final clean rerun completes.
    pub max_iterations: Option<usize>,
    /// Whether diff analysis should remove/localize mismatching instructions.
    /// When false, the runner reports the first observed exception/write mismatch and
    /// preserves the original generated testcase.
    pub enable_reduction: bool,
    /// 可选下界；当 testcase 长度不高于它时，跳过历史最小化。
    /// Optional lower bound below which history minimization is skipped.
    pub history_min_test_threshold: Option<usize>,
    /// 是否为每个实现/每轮迭代保留原始执行输出。
    /// Persist raw execution outputs for each implementation/iteration.
    pub emit_execution_output_json: bool,
    /// 是否为每个实现/每轮迭代保留 Markdown 执行报告。
    /// Persist Markdown execution reports for each implementation/iteration.
    pub emit_execution_report_md: bool,
    /// 当某轮迭代干净收敛后，是否删除按实现拆分的迭代工件。
    /// Remove per-implementation iteration artifacts when an iteration converges cleanly.
    pub cleanup_successful_iteration_artifacts: bool,
    /// 如果整个 diff run 没发生任何归约，是否删除整次 run 的目录。
    /// Remove the whole run directory if the diff run succeeds without any reductions.
    pub cleanup_successful_diff_run: bool,

    /// 启用后，会为每个写差异额外生成一个"probe" replay testcase。
    /// When enabled, generate a second "probe" replay testcase for each write difference.
    pub verify_write_with_probes: bool,
    /// probe replay 最多实体化多少个发生分叉的寄存器。
    /// Maximum number of divergent registers that probe replay should materialize.
    pub write_probe_max_registers: usize,
    /// probe replay 最多实体化多少个发生分叉的内存字节。
    /// Maximum number of divergent memory bytes that probe replay should materialize.
    pub write_probe_max_memory_bytes: usize,

    /// Automatic local de-duplication for diff findings.
    pub dedup: DiffDedupSettings,

    /// 可选的 testcase 生成扩展覆盖；设置后会绕过基于实现交集的默认扩展推导。
    /// Optional testcase-generation extension override; when set, this bypasses the default
    /// extension intersection derived from the selected implementations.
    pub generation_extensions_override: Option<IsaExtensions>,

    /// 可选的 testcase 执行扩展覆盖；设置后会在生成后替换 testcase.extensions，
    /// 供各实现组装/执行共享模板时使用。
    /// Optional testcase-execution extension override; when set, this replaces
    /// testcase.extensions after generation so implementation wrappers can still
    /// assemble/execute any framework instructions they require.
    pub execution_extensions_override: Option<IsaExtensions>,
}

impl DiffAnalysisSettings {
    /// 把共享配置与输出根目录合并，生成一次具体 run 的配置。
    /// Materialize a concrete run config by attaching an output root to shared settings.
    pub fn with_run_root(&self, run_root: PathBuf) -> DiffAnalysisConfig {
        DiffAnalysisConfig {
            settings: self.clone(),
            run_root,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiffAnalysisResult {
    /// 发生任何归约之前的初始 testcase。
    /// The testcase before any reduction happened.
    pub initial_testcase: GeneratedTestCase,
    /// 异常/写差异归约收敛之后得到的最终 testcase。
    /// The final testcase after exception/write reduction converged.
    pub final_testcase: GeneratedTestCase,
    /// `final_testcase` 对应的执行输出。
    /// Outputs for `final_testcase`.
    pub final_outputs: HashMap<String, ExecutionOutput>,
    /// 移除了异常触发指令的轮次数。
    /// Number of rounds that removed exception-triggering instructions.
    pub exception_removal_rounds: usize,
    /// 移除了写差异指令或其依赖历史的轮次数。
    /// Number of rounds that removed write-difference instructions/history.
    pub write_removal_rounds: usize,
    /// 本次 run 所有工件所在的根目录。
    /// Root directory that contains all artifacts for this run.
    pub output_root: PathBuf,
}

#[derive(Debug, Error)]
pub enum DiffAnalysisError {
    #[error("testcase generation failed: {0}")]
    TestcaseGeneration(#[from] GenerationError),
    #[error("execution failed: {0}")]
    Execution(#[source] Box<dyn std::error::Error>),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("report generation failed: {0}")]
    Report(#[source] Box<dyn std::error::Error>),
    #[error("missing instructions for implementation {core_id}")]
    MissingInstructions { core_id: String },
    #[error("test instruction count mismatch: {details}")]
    TestInstructionCountMismatch { details: String },
    #[error("history minimization failed: {reason}")]
    HistoryMinimizationFailure { reason: String },
    #[error("initialization mismatch at instruction {instruction_index}: {details}")]
    InitializationMismatch {
        instruction_index: usize,
        details: String,
    },
    #[error("maximum iteration limit {0} reached without convergence")]
    /// 严格模式下的硬错误：归约器耗尽了迭代预算后被迫停止。
    /// Strict-mode hard error: the reducer stopped because it exhausted its iteration budget.
    IterationLimitReached(usize),
}

impl DiffAnalysisError {
    /// 把错误映射到稳定的 bug category，供调用方和报告使用。
    /// Classify an error into a stable bug category string used by callers and reports.
    pub fn bug_category(&self) -> DiffBugCategory {
        match self {
            DiffAnalysisError::TestcaseGeneration(_) => DiffBugCategory::TestcaseGenerationFailure,
            DiffAnalysisError::Execution(_) => DiffBugCategory::ExecutionFailure,
            DiffAnalysisError::Io(_) => DiffBugCategory::IoFailure,
            DiffAnalysisError::Serialization(_) => DiffBugCategory::SerializationFailure,
            DiffAnalysisError::Report(_) => DiffBugCategory::ReportGenerationFailure,
            DiffAnalysisError::MissingInstructions { .. } => DiffBugCategory::MissingInstructions,
            DiffAnalysisError::TestInstructionCountMismatch { .. } => {
                DiffBugCategory::InstructionCountMismatch
            }
            DiffAnalysisError::HistoryMinimizationFailure { .. } => {
                DiffBugCategory::HistoryMinimizationFailure
            }
            DiffAnalysisError::InitializationMismatch { .. } => {
                DiffBugCategory::InitializationMismatch
            }
            DiffAnalysisError::IterationLimitReached(_) => DiffBugCategory::IterationLimitReached,
        }
    }
}
