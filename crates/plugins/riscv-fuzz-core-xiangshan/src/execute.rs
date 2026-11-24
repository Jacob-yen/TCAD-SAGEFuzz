use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use log::{debug, info};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Serialize;
use thiserror::Error;

use super::{
    asm::{build_asm, linker_script},
    metadata::isa_profiles,
};
use riscv_fuzz_plugin_api::coverage;
use riscv_fuzz_plugin_api::doctor::parse_global_cycles_from_text;
use riscv_fuzz_plugin_api::streaming::{InstructionEvent, StreamingOutputDelta};
use riscv_fuzz_plugin_api::{
    AsmInstructionExecution, ConfigError, ElfError, ExceptionCause, ExecutionOutput,
    HartExecutionOutput, ISABase, ImplExecutionError, InstructionTiming, IsaExtensions,
    MemValue as TraitMemValue, ParseError, PerHartTypedSequences, PerfMetrics, ProcessError,
    RegisterValue as TraitRegisterValue, UNKNOWN_EXCEPTION_CODE, canonicalize_register_name,
    configure_command_process_group, kill_child_process_group, parse_register_name,
    typed_sequences_for_hart,
};
use riscv_fuzz_toolchain::{
    build_elf::{ElfBuildResult, build_elf_with_extensions},
    error::BuildElfError,
    tracer::{TracedInstruction, Tracer, build_raw_hart_execution_output},
};
use riscv_instruction::{
    AsmDecodeWithContext, RegisterBank, RuntimeOperandAccess, RuntimeOperandUsage,
};

pub(crate) type PerHartAsm<'a> = &'a [(u32, &'a [String])];

const XIANGSHAN_DIFF_ENV_1C: &str = "CX_RISCV_CORES_XIANGSHAN_DIFFTEST_RV64_1C_SO";
const XIANGSHAN_DIFF_ENV_2C: &str = "CX_RISCV_CORES_XIANGSHAN_DIFFTEST_RV64_2C_SO";
const XIANGSHAN_EMU_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C";
const XIANGSHAN_EMU_COV_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C_COV";
const XIANGSHAN_EMU_COV_LIGHT_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C_COV_LIGHT";
const XIANGSHAN_EMU_NO_UNALIGNED_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C";
const XIANGSHAN_EMU_NO_UNALIGNED_COV_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C_COV";
const XIANGSHAN_EMU_NO_UNALIGNED_COV_LIGHT_ENV: &str =
    "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C_COV_LIGHT";
const XIANGSHAN_EMU_2HART_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C";
const XIANGSHAN_EMU_2HART_COV_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C_COV";
const XIANGSHAN_EMU_2HART_COV_LIGHT_ENV: &str =
    "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C_COV_LIGHT";
const XIANGSHAN_EMU_NO_UNALIGNED_2HART_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C";
const XIANGSHAN_EMU_NO_UNALIGNED_2HART_COV_ENV: &str =
    "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C_COV";
const XIANGSHAN_EMU_NO_UNALIGNED_2HART_COV_LIGHT_ENV: &str =
    "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C_COV_LIGHT";

#[derive(Debug, Error)]
pub enum XiangShanError {
    #[error("environment variable {var} is not set")]
    EnvVarNotSet { var: String },

    #[error("XiangShan emulator not found: {path}")]
    BinaryNotFound { path: String },

    #[error("DiffTest reference library not found: {path}")]
    DiffSoNotFound { path: String },

    #[error("invalid environment variable {var}: {reason}")]
    InvalidEnvVar { var: String, reason: String },

    #[error(transparent)]
    ProcessError(#[from] ProcessError),

    #[error(transparent)]
    ParseError(#[from] ParseError),

    #[error(transparent)]
    ElfError(#[from] ElfError),

    #[error(transparent)]
    IoError(#[from] std::io::Error),

    #[error(transparent)]
    BuildElf(#[from] BuildElfError),

    #[error(transparent)]
    Config(#[from] ConfigError),

    #[error("failed to build ELF: {0}")]
    BuildElfError(String),

    #[error("XiangShan commit trace is empty")]
    EmptyTrace,

    #[error("source PC mapping missing for instruction index {index}")]
    NoPcMapping { index: usize },

    #[error(
        "XiangShan trace contains duplicate {kind} record for hart {hart} pc 0x{pc:016x}: {detail}"
    )]
    DuplicateTraceRecord {
        kind: &'static str,
        hart: u32,
        pc: u64,
        detail: String,
    },

    #[error(
        "XiangShan trace contains conflicting {kind} records for hart {hart} pc 0x{pc:016x}: {detail}"
    )]
    ConflictingTraceRecord {
        kind: &'static str,
        hart: u32,
        pc: u64,
        detail: String,
    },

    #[error("XiangShan trace is incomplete for trace instruction {source_index}: {detail}")]
    IncompleteTrace { source_index: usize, detail: String },

    #[error("XiangShan emulator exited with status {status:?}. See logs for details.")]
    NonZeroExit {
        status: Option<i32>,
        stdout: String,
        stderr: String,
    },
}

impl From<XiangShanError> for ImplExecutionError {
    fn from(value: XiangShanError) -> Self {
        match value {
            XiangShanError::ProcessError(err) => err.into(),
            XiangShanError::ParseError(err) => err.into(),
            XiangShanError::ElfError(err) => err.into(),
            XiangShanError::IoError(err) => err.into(),
            other => ImplExecutionError::Other(Box::new(other)),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RegisterWrite {
    pub pc: u64,
    pub register: String,
    pub value: u64,
    pub timing: InstructionTiming,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryWrite {
    pub pc: u64,
    pub addr: u64,
    pub bytes: Vec<(u64, u8)>,
    pub timing: InstructionTiming,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExceptionEvent {
    pub pc: u64,
    pub cause: u64,
    pub timing: InstructionTiming,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstructionTrace {
    pub pc: u64,
    pub timing: InstructionTiming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
struct TraceKey {
    hart: u32,
    pc: u64,
}

impl TraceKey {
    fn new(hart: u32, pc: u64) -> Self {
        Self { hart, pc }
    }
}

#[derive(Debug, Clone, Copy)]
struct PendingStandaloneMemLine {
    pc: u64,
    addr: u64,
    data: u64,
    mask: u8,
}

#[derive(Debug, Clone, Default)]
pub struct XiangShanTrace {
    registers: HashMap<TraceKey, Vec<RegisterWrite>>,
    memory: HashMap<TraceKey, Vec<MemoryWrite>>,
    exceptions: HashMap<TraceKey, ExceptionEvent>,
    instruction_timings: HashMap<TraceKey, Vec<InstructionTrace>>,
    committed_pcs: HashSet<TraceKey>,
}

impl XiangShanTrace {
    fn has_content(&self) -> bool {
        !self.registers.is_empty()
            || !self.memory.is_empty()
            || !self.exceptions.is_empty()
            || !self.instruction_timings.is_empty()
            || !self.committed_pcs.is_empty()
    }

    fn max_cycle_for_hart(&self, hart_id: u32) -> Option<u64> {
        self.instruction_timings
            .iter()
            .filter(|(key, _)| key.hart == hart_id)
            .flat_map(|(_, entries)| entries.iter())
            .filter_map(|entry| entry.timing.clk_start.or(entry.timing.clk_end))
            .max()
    }

    #[cfg(test)]
    fn register_writes(&self, hart: u32, pc: u64) -> Option<&Vec<RegisterWrite>> {
        self.registers.get(&TraceKey::new(hart, pc))
    }

    #[cfg(test)]
    fn memory_writes(&self, hart: u32, pc: u64) -> Option<&Vec<MemoryWrite>> {
        self.memory.get(&TraceKey::new(hart, pc))
    }

    fn exception(&self, hart: u32, pc: u64) -> Option<&ExceptionEvent> {
        self.exceptions.get(&TraceKey::new(hart, pc))
    }

    fn instruction_timing_entries(&self, hart: u32, pc: u64) -> Option<&Vec<InstructionTrace>> {
        self.instruction_timings.get(&TraceKey::new(hart, pc))
    }

    #[cfg(test)]
    fn registers_contain_pc(&self, hart: u32, pc: u64) -> bool {
        self.registers.contains_key(&TraceKey::new(hart, pc))
    }

    #[cfg(test)]
    fn memory_contains_pc(&self, hart: u32, pc: u64) -> bool {
        self.memory.contains_key(&TraceKey::new(hart, pc))
    }

    #[cfg(test)]
    fn committed_contains_pc(&self, hart: u32, pc: u64) -> bool {
        self.committed_pcs.contains(&TraceKey::new(hart, pc))
    }

    fn exceptions_contain_pc(&self, hart: u32, pc: u64) -> bool {
        self.exceptions.contains_key(&TraceKey::new(hart, pc))
    }
}

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub work_dir: PathBuf,
    pub isa_base: ISABase,
    pub core_id: String,
    pub timeout: Option<Duration>,
    pub timeout_overridden: bool,
    pub extension_override: Option<IsaExtensions>,
    pub diff_so_path: Option<PathBuf>,
    pub extra_emu_args: Vec<String>,
    pub allow_unaligned: bool,
    pub coverage_mode: coverage::CoverageMode,
}

fn internal_difftest_enabled() -> bool {
    !matches!(
        env::var("CX_RISCV_CORES_XIANGSHAN_INTERNAL_DIFF")
            .or_else(|_| env::var("CX_RISCV_CORES_XIANGSHAN_ENABLE_INTERNAL_DIFF")),
        Ok(value)
            if matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
    )
}

fn difftest_log_only(internal_diff: bool) -> bool {
    !internal_diff
        || matches!(
            env::var("CX_RISCV_CORES_XIANGSHAN_DIFFTEST_LOG_ONLY")
                .or_else(|_| env::var("DIFFTEST_LOG_ONLY")),
            Ok(value)
                if matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
        )
}

impl ExecutorConfig {
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);
    pub const DEFAULT_MULTICORE_TIMEOUT: Duration = Duration::from_secs(300);

    pub fn new(work_dir: PathBuf, isa_base: ISABase, core_id: String) -> Self {
        Self {
            work_dir,
            isa_base,
            core_id,
            timeout: Some(Self::DEFAULT_TIMEOUT),
            timeout_overridden: false,
            extension_override: None,
            diff_so_path: None,
            extra_emu_args: Vec::new(),
            allow_unaligned: true,
            coverage_mode: coverage::CoverageMode::Off,
        }
    }

    fn effective_timeout(&self, multicore: bool) -> Option<Duration> {
        if multicore {
            return match self.timeout {
                Some(timeout) => Some(timeout.max(Self::DEFAULT_MULTICORE_TIMEOUT)),
                None => Some(Self::DEFAULT_MULTICORE_TIMEOUT),
            };
        }
        self.timeout
    }

    pub fn set_extension_override(&mut self, extensions: IsaExtensions) {
        self.extension_override = Some(extensions);
    }

    pub fn resolved_extensions(&self) -> IsaExtensions {
        if let Some(map) = &self.extension_override {
            map.clone()
        } else {
            isa_profiles().for_isa(self.isa_base).extensions.clone()
        }
    }

    fn emulator_env_var(&self, multicore: bool) -> &'static str {
        if self.allow_unaligned {
            match (multicore, self.coverage_mode) {
                (false, coverage::CoverageMode::Off) => XIANGSHAN_EMU_ENV,
                (false, coverage::CoverageMode::Full) => XIANGSHAN_EMU_COV_ENV,
                (false, coverage::CoverageMode::Light) => XIANGSHAN_EMU_COV_LIGHT_ENV,
                (true, coverage::CoverageMode::Off) => XIANGSHAN_EMU_2HART_ENV,
                (true, coverage::CoverageMode::Full) => XIANGSHAN_EMU_2HART_COV_ENV,
                (true, coverage::CoverageMode::Light) => XIANGSHAN_EMU_2HART_COV_LIGHT_ENV,
            }
        } else {
            match (multicore, self.coverage_mode) {
                (false, coverage::CoverageMode::Off) => XIANGSHAN_EMU_NO_UNALIGNED_ENV,
                (false, coverage::CoverageMode::Full) => XIANGSHAN_EMU_NO_UNALIGNED_COV_ENV,
                (false, coverage::CoverageMode::Light) => XIANGSHAN_EMU_NO_UNALIGNED_COV_LIGHT_ENV,
                (true, coverage::CoverageMode::Off) => XIANGSHAN_EMU_NO_UNALIGNED_2HART_ENV,
                (true, coverage::CoverageMode::Full) => XIANGSHAN_EMU_NO_UNALIGNED_2HART_COV_ENV,
                (true, coverage::CoverageMode::Light) => {
                    XIANGSHAN_EMU_NO_UNALIGNED_2HART_COV_LIGHT_ENV
                }
            }
        }
    }

    fn emulator_path(&self, multicore: bool) -> Result<PathBuf, XiangShanError> {
        let env_var = self.emulator_env_var(multicore);
        let path = env::var(env_var).map_err(|_| XiangShanError::EnvVarNotSet {
            var: env_var.to_string(),
        })?;

        if path.is_empty() {
            return Err(XiangShanError::EnvVarNotSet {
                var: env_var.to_string(),
            });
        }

        let path_buf = PathBuf::from(&path);
        if !path_buf.exists() {
            return Err(XiangShanError::BinaryNotFound {
                path: path_buf.display().to_string(),
            });
        }

        Ok(path_buf)
    }

    fn diff_so_path(&self, multicore: bool) -> Result<PathBuf, XiangShanError> {
        if let Some(path) = &self.diff_so_path {
            return Ok(path.clone());
        }

        let env_var = if multicore {
            XIANGSHAN_DIFF_ENV_2C
        } else {
            XIANGSHAN_DIFF_ENV_1C
        };
        let candidate = path_from_env_value(env_var, env::var_os(env_var))?.ok_or_else(|| {
            XiangShanError::EnvVarNotSet {
                var: env_var.to_string(),
            }
        })?;

        if !candidate.is_file() {
            return Err(XiangShanError::DiffSoNotFound {
                path: candidate.display().to_string(),
            });
        }

        Ok(candidate)
    }
}

fn path_from_env_value(
    var: &str,
    value: Option<OsString>,
) -> Result<Option<PathBuf>, XiangShanError> {
    let Some(value) = value else {
        return Ok(None);
    };

    if value.is_empty() {
        return Err(XiangShanError::InvalidEnvVar {
            var: var.to_string(),
            reason: "must not be empty".to_string(),
        });
    }

    Ok(Some(PathBuf::from(value)))
}

fn resolve_wrapper_elf_path(elf_path: &Path, run_dir: &Path) -> Result<PathBuf, XiangShanError> {
    if elf_path.is_absolute() {
        if elf_path.is_file() {
            return Ok(elf_path.to_path_buf());
        }
        return Err(XiangShanError::BuildElfError(format!(
            "built ELF path does not exist: {}",
            elf_path.display()
        )));
    }

    if elf_path.exists() {
        return fs::canonicalize(elf_path).map_err(|err| {
            XiangShanError::BuildElfError(format!(
                "failed to canonicalize built ELF {}: {err}",
                elf_path.display()
            ))
        });
    }

    let Some(file_name) = elf_path.file_name() else {
        return Err(XiangShanError::BuildElfError(format!(
            "built ELF path has no file name: {}",
            elf_path.display()
        )));
    };

    let candidate = run_dir.join(file_name);
    if !candidate.is_file() {
        return Err(XiangShanError::BuildElfError(format!(
            "built ELF path does not exist: {}; run-dir candidate also missing: {}",
            elf_path.display(),
            candidate.display()
        )));
    }

    fs::canonicalize(&candidate).map_err(|err| {
        XiangShanError::BuildElfError(format!(
            "failed to canonicalize run-dir ELF candidate {}: {err}",
            candidate.display()
        ))
    })
}

fn extract_xs_log_payload(line: &str) -> Option<&str> {
    static RE_COMPACT_COMMIT: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\[\d+\](?:\[\d+\])?\s+commit pc [^\r\n]*").unwrap());
    static RE_COMPACT_EXCEPTION: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\[\d+\](?:\[\d+\])?\s+exception pc [^\r\n]*").unwrap());

    if line.trim().is_empty() {
        return None;
    }

    for marker in [
        "Usage: ",
        "unrecognized option",
        "HIT GOOD TRAP",
        "HIT BAD TRAP",
        "Refill test failed!",
        "retired hart",
        "store commit hart",
        "trap hart",
        "mem pc ",
        "instrCnt",
        "cycleCnt",
        "Cycles: ",
    ] {
        if let Some(start) = line.find(marker) {
            return Some(line[start..].trim());
        }
    }

    if let Some(matched) = RE_COMPACT_COMMIT.find(line) {
        return Some(line[matched.start()..matched.end()].trim());
    }
    if let Some(matched) = RE_COMPACT_EXCEPTION.find(line) {
        return Some(line[matched.start()..matched.end()].trim());
    }

    None
}

#[cfg(test)]
fn should_keep_xs_log_line(line: &str) -> bool {
    extract_xs_log_payload(line).is_some()
}

fn filtered_log_write_error(path: &Path, source: std::io::Error) -> XiangShanError {
    XiangShanError::ProcessError(ProcessError::FileWriteFailed {
        path: path.to_path_buf(),
        source,
    })
}

fn drain_xs_log_stream<R: Read>(
    reader: R,
    writer: Arc<Mutex<fs::File>>,
    path: PathBuf,
) -> Result<(), XiangShanError> {
    let reader = BufReader::new(reader);

    for line in reader.lines() {
        let line = line.map_err(XiangShanError::IoError)?;
        let mut writer = writer.lock().map_err(|err| {
            filtered_log_write_error(
                &path,
                std::io::Error::other(format!("filtered XiangShan log writer poisoned: {err}")),
            )
        })?;
        writeln!(writer, "{line}").map_err(|source| filtered_log_write_error(&path, source))?;
    }

    Ok(())
}

fn join_xs_log_drain(
    handle: thread::JoinHandle<Result<(), XiangShanError>>,
    stream_name: &str,
) -> Result<(), XiangShanError> {
    handle.join().map_err(|payload| {
        let detail = if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_string()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "unknown panic payload".to_string()
        };
        XiangShanError::IoError(std::io::Error::other(format!(
            "XiangShan {stream_name} log drain thread panicked: {detail}"
        )))
    })?
}

#[derive(Debug)]
pub struct XiangShanExecutor {
    config: ExecutorConfig,
}

impl XiangShanExecutor {
    pub fn new(config: ExecutorConfig) -> Self {
        Self { config }
    }

    /// Compile the program and run XiangShan, but skip trace parsing / ExecutionOutput construction.
    ///
    /// Intended for fast outer-loop workflows such as coverage-guided generation where only the
    /// side-effects (e.g. coverage.dat) are needed.
    pub fn compile_and_run<P: AsRef<Path>>(
        &self,
        run_root: P,
        per_hart: PerHartAsm<'_>,
    ) -> Result<(), XiangShanError> {
        if self.config.isa_base != ISABase::Rv64 {
            return Err(XiangShanError::Config(ConfigError::UnsupportedIsaBase {
                core_id: "XiangShan".to_string(),
                isa_base: format!("{:?}", self.config.isa_base),
            }));
        }

        let run_dir = run_root.as_ref();
        fs::create_dir_all(run_dir).map_err(|e| {
            XiangShanError::ProcessError(ProcessError::DirectoryCreationFailed {
                path: run_dir.to_path_buf(),
                source: e,
            })
        })?;

        let extensions = self.config.resolved_extensions();
        let program = build_asm(per_hart, self.config.isa_base)?;
        let asm_path = run_dir.join("program.S");
        fs::write(&asm_path, &program).map_err(|e| {
            XiangShanError::ProcessError(ProcessError::FileWriteFailed {
                path: asm_path.clone(),
                source: e,
            })
        })?;

        let build = build_elf_with_extensions(&asm_path, &extensions, linker_script())?;

        let harts = per_hart.len() as u32;
        let _ = self.run_emu(run_dir, &build, harts > 1)?;
        Ok(())
    }

    pub fn execute<P: AsRef<Path>>(
        &self,
        run_root: P,
        per_hart: PerHartAsm<'_>,
    ) -> Result<ExecutionOutput, XiangShanError> {
        if self.config.isa_base != ISABase::Rv64 {
            return Err(XiangShanError::Config(ConfigError::UnsupportedIsaBase {
                core_id: "XiangShan".to_string(),
                isa_base: format!("{:?}", self.config.isa_base),
            }));
        }

        let run_dir = run_root.as_ref();
        fs::create_dir_all(run_dir).map_err(|e| {
            XiangShanError::ProcessError(ProcessError::DirectoryCreationFailed {
                path: run_dir.to_path_buf(),
                source: e,
            })
        })?;

        let extensions = self.config.resolved_extensions();
        let program = build_asm(per_hart, self.config.isa_base)?;
        let asm_path = run_dir.join("program.S");
        fs::write(&asm_path, &program).map_err(|e| {
            XiangShanError::ProcessError(ProcessError::FileWriteFailed {
                path: asm_path.clone(),
                source: e,
            })
        })?;

        let build = build_elf_with_extensions(&asm_path, &extensions, linker_script())?;

        let harts = per_hart.len() as u32;
        let run_result = self.run_emu(run_dir, &build, harts > 1)?;

        let trace = parse_xs_commit_trace(&run_result.commit_log_path)?;

        let perf = extract_perf_metrics_from_xs_trace(
            self.config.isa_base,
            self.config.core_id.clone(),
            harts,
            false,
            &build.disassembly_file,
            &trace,
            &run_result.commit_log_path,
        )?;

        let mut per_hart_out = BTreeMap::new();
        let final_snapshot = run_result.exit_disposition.requires_final_snapshot();
        let partial_stop_reason = run_result.exit_disposition.partial_stop_reason();
        for (hart_id, hart_insts) in per_hart.iter().copied() {
            let tracer = Tracer::new_with_section_labels(
                hart_insts,
                &build.disassembly_file,
                xiangshan_section_labels_for_hart(harts, hart_id),
                self.config.isa_base,
            )
            .map_err(|e| {
                XiangShanError::ElfError(ElfError::DumpLoadFailed {
                    path: build.disassembly_file.clone(),
                    source: e,
                })
            })?;
            let mut hart_output = build_execution_output(&trace, hart_id, &tracer, final_snapshot)?;
            if let Some(reason) = partial_stop_reason {
                annotate_partial_trace_termination(&mut hart_output, reason);
            }
            per_hart_out.insert(hart_id, hart_output);
        }

        let mut out = ExecutionOutput::empty(self.config.core_id.clone(), self.config.isa_base);
        out.per_hart = per_hart_out;
        out.perf = perf;
        Ok(out)
    }

    pub fn execute_streaming<P: AsRef<Path>>(
        &self,
        run_root: P,
        per_hart: PerHartAsm<'_>,
        typed_sequences: Option<PerHartTypedSequences<'_>>,
        emit: &mut dyn FnMut(InstructionEvent) -> bool,
    ) {
        match self.execute_streaming_inner(run_root.as_ref(), per_hart, typed_sequences, emit) {
            Ok(Some(perf)) => {
                emit(InstructionEvent::Completed { perf });
            }
            Ok(None) => {}
            Err(error) => {
                emit(InstructionEvent::Failed {
                    error: ImplExecutionError::from(error),
                });
            }
        }
    }

    fn execute_streaming_inner(
        &self,
        run_dir: &Path,
        per_hart: PerHartAsm<'_>,
        typed_sequences: Option<PerHartTypedSequences<'_>>,
        emit: &mut dyn FnMut(InstructionEvent) -> bool,
    ) -> Result<Option<PerfMetrics>, XiangShanError> {
        if self.config.isa_base != ISABase::Rv64 {
            return Err(XiangShanError::Config(ConfigError::UnsupportedIsaBase {
                core_id: "XiangShan".to_string(),
                isa_base: format!("{:?}", self.config.isa_base),
            }));
        }

        fs::create_dir_all(run_dir).map_err(|source| {
            XiangShanError::ProcessError(ProcessError::DirectoryCreationFailed {
                path: run_dir.to_path_buf(),
                source,
            })
        })?;
        let extensions = self.config.resolved_extensions();
        let program = build_asm(per_hart, self.config.isa_base)?;
        let asm_path = run_dir.join("program.S");
        fs::write(&asm_path, &program).map_err(|source| {
            XiangShanError::ProcessError(ProcessError::FileWriteFailed {
                path: asm_path.clone(),
                source,
            })
        })?;
        let build = build_elf_with_extensions(&asm_path, &extensions, linker_script())?;
        let harts = per_hart.len() as u32;

        let mut tracers = BTreeMap::new();
        for (hart_id, hart_insts) in per_hart.iter().copied() {
            let labels = xiangshan_section_labels_for_hart(harts, hart_id);
            let tracer =
                if let Some(sequences) = typed_sequences_for_hart(typed_sequences, hart_id) {
                    Tracer::new_for_sequences_with_section_labels(
                        sequences,
                        &build.disassembly_file,
                        labels,
                        self.config.isa_base,
                    )
                } else {
                    Tracer::new_with_section_labels(
                        hart_insts,
                        &build.disassembly_file,
                        labels,
                        self.config.isa_base,
                    )
                }
                .map_err(|source| {
                    XiangShanError::ElfError(ElfError::DumpLoadFailed {
                        path: build.disassembly_file.clone(),
                        source,
                    })
                })?;
            tracers.insert(hart_id, tracer);
        }

        let emu_path_raw = self.config.emulator_path(harts > 1)?;
        let emu_path = emu_path_raw.canonicalize().map_err(|err| {
            if emu_path_raw.exists() {
                XiangShanError::IoError(err)
            } else {
                XiangShanError::BinaryNotFound {
                    path: emu_path_raw.display().to_string(),
                }
            }
        })?;
        let internal_diff = internal_difftest_enabled();
        let diff_so = if internal_diff {
            let diff_so_raw = self.config.diff_so_path(harts > 1)?;
            Some(diff_so_raw.canonicalize().map_err(|err| {
                if diff_so_raw.exists() {
                    XiangShanError::IoError(err)
                } else {
                    XiangShanError::DiffSoNotFound {
                        path: diff_so_raw.display().to_string(),
                    }
                }
            })?)
        } else {
            None
        };

        let commit_log_path = run_dir.join("skiptrap.commit.log");
        let coverage_path = if self.config.coverage_mode.is_enabled() {
            Some(run_dir.join("coverage.dat"))
        } else {
            None
        };
        let mut cmd = Command::new(&emu_path);
        cmd.current_dir(run_dir);
        cmd.env(
            "DIFFTEST_LOG_ONLY",
            if difftest_log_only(internal_diff) {
                "1"
            } else {
                "0"
            },
        );
        cmd.env(
            "CX_RISCV_CORES_XIANGSHAN_ENABLE_INTERNAL_DIFF",
            if internal_diff { "1" } else { "0" },
        );

        let has_ram_size_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--ram-size" || arg.starts_with("--ram-size="));
        let has_max_cycles_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--max-cycles" || arg.starts_with("--max-cycles="));
        let has_max_instr_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--max-instr" || arg.starts_with("--max-instr="));
        let has_log_begin_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--log-begin" || arg.starts_with("--log-begin="));
        let has_log_end_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--log-end" || arg.starts_with("--log-end="));
        let has_dump_commit_trace_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--dump-commit-trace");
        let elf_arg = resolve_wrapper_elf_path(&build.executable_file, run_dir)?;

        cmd.arg("--image").arg(&elf_arg);
        if let Some(diff_so) = &diff_so {
            cmd.arg("--diff").arg(diff_so);
        } else {
            cmd.arg("--no-diff");
        }
        if !has_ram_size_arg {
            cmd.arg("--ram-size=64GB");
        }
        if !has_max_cycles_arg {
            cmd.arg("--max-cycles=2000000");
        }
        if !has_max_instr_arg {
            cmd.arg("--max-instr=50000");
        }
        if !has_dump_commit_trace_arg {
            cmd.arg("--dump-commit-trace");
        }
        if !has_log_begin_arg {
            cmd.arg("--log-begin=0");
        }
        if !has_log_end_arg {
            cmd.arg("--log-end=200000");
        }
        if coverage_path.is_some() {
            cmd.arg("--dump-coverage");
        }
        if let Some(ref cov) = coverage_path {
            cmd.arg("--coverage-file").arg(cov);
        }
        for arg in &self.config.extra_emu_args {
            cmd.arg(arg);
        }

        let filtered_log = Arc::new(Mutex::new(fs::File::create(&commit_log_path).map_err(
            |source| {
                XiangShanError::ProcessError(ProcessError::FileWriteFailed {
                    path: commit_log_path.clone(),
                    source,
                })
            },
        )?));
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        configure_command_process_group(&mut cmd);
        let mut command_parts = Vec::new();
        command_parts.push(quote_os_str(emu_path.as_os_str()));
        command_parts.extend(cmd.get_args().map(quote_os_str));
        let command_display = format!(
            "cd \"{}\" && {}",
            run_dir.display(),
            command_parts.join(" ")
        );
        info!("Launching XiangShan streaming command: {}", command_display);

        let mut child = cmd.spawn().map_err(|source| {
            XiangShanError::ProcessError(ProcessError::SpawnFailed {
                command: command_display.clone(),
                source,
            })
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            XiangShanError::ProcessError(ProcessError::SpawnFailed {
                command: command_display.clone(),
                source: std::io::Error::other("child stdout pipe missing"),
            })
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            XiangShanError::ProcessError(ProcessError::SpawnFailed {
                command: command_display.clone(),
                source: std::io::Error::other("child stderr pipe missing"),
            })
        })?;
        let stdout_log = Arc::clone(&filtered_log);
        let stdout_path = commit_log_path.clone();
        let stdout_drain =
            thread::spawn(move || drain_xs_log_stream(stdout, stdout_log, stdout_path));
        let stderr_log = Arc::clone(&filtered_log);
        let stderr_path = commit_log_path.clone();
        let stderr_drain =
            thread::spawn(move || drain_xs_log_stream(stderr, stderr_log, stderr_path));

        let mut delta = StreamingOutputDelta::new();
        let mut last_complete_log_len = 0usize;
        let mut last_event_at = Instant::now();
        let mut timed_out = None;
        let timeout = self.config.effective_timeout(harts > 1);
        let status = loop {
            if let Some((log_text, complete_len)) =
                riscv_fuzz_plugin_api::read_complete_utf8_prefix(&commit_log_path, false)?
            {
                if complete_len == last_complete_log_len {
                    // No new complete line is available yet. The child may be
                    // in the middle of writing the next log row.
                } else {
                    last_complete_log_len = complete_len;
                    match parse_xs_commit_trace_reader(log_text.as_bytes()) {
                        Ok(trace) if trace.has_content() => {
                            let ready =
                                streaming_snapshot_from_xiangshan_trace(&trace, &tracers, false)?;
                            let mut emitted = 0usize;
                            let keep_going =
                                delta.emit_per_hart_delta(&ready, false, &mut |event| {
                                    emitted += 1;
                                    emit(event)
                                });
                            if emitted > 0 {
                                last_event_at = Instant::now();
                            }
                            if !keep_going {
                                let _ = kill_child_process_group(&mut child);
                                let _ = child.wait();
                                join_xs_log_drain(stdout_drain, "stdout")?;
                                join_xs_log_drain(stderr_drain, "stderr")?;
                                return Ok(None);
                            }
                        }
                        Ok(_) => {}
                        Err(error) => return Err(error),
                    }
                };
            }

            if let Some(exit_status) = child.try_wait().map_err(|source| {
                XiangShanError::ProcessError(ProcessError::SpawnFailed {
                    command: command_display.clone(),
                    source,
                })
            })? {
                break exit_status;
            }
            if let Some(limit) = timeout
                && last_event_at.elapsed() >= limit
            {
                timed_out = Some(limit);
                let _ = kill_child_process_group(&mut child);
                break child.wait()?;
            }
            thread::sleep(Duration::from_millis(25));
        };

        join_xs_log_drain(stdout_drain, "stdout")?;
        join_xs_log_drain(stderr_drain, "stderr")?;

        if let Some(limit) = timed_out {
            return Err(XiangShanError::ProcessError(ProcessError::TimedOut {
                command: command_display,
                timeout: limit,
            }));
        }

        let exit_disposition =
            classify_xiangshan_exit_status(status, &command_display, &commit_log_path)?;

        if let Some(cov_path) = &coverage_path {
            if !cov_path.exists() {
                return Err(XiangShanError::ProcessError(ProcessError::ProcessFailed {
                    command: command_display.clone(),
                    stderr: format!(
                        "coverage enabled but coverage file is missing: {}",
                        cov_path.display()
                    ),
                }));
            }
            let summary =
                riscv_fuzz_toolchain::coverage::parse_verilator_dat(cov_path).map_err(|err| {
                    XiangShanError::ProcessError(ProcessError::ProcessFailed {
                        command: command_display.clone(),
                        stderr: format!(
                            "coverage enabled but failed to parse {}: {err}",
                            cov_path.display()
                        ),
                    })
                })?;
            info!("Coverage file written to {}", cov_path.display());
            for line in summary.render_lines() {
                info!("  coverage: {line}");
            }
        }

        let trace = parse_xs_commit_trace(&commit_log_path)?;
        let mut ready = if exit_disposition.requires_final_snapshot() {
            streaming_snapshot_from_xiangshan_trace(&trace, &tracers, true)?
        } else {
            tracers
                .iter()
                .map(|(hart_id, tracer)| {
                    build_execution_output(&trace, *hart_id, tracer, false)
                        .map(|output| (*hart_id, output))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?
        };
        if let Some(reason) = exit_disposition.partial_stop_reason() {
            for output in ready.values_mut() {
                annotate_partial_trace_termination(output, reason);
            }
        }
        if !delta.emit_per_hart_delta(&ready, true, emit) {
            return Ok(None);
        }
        let perf = extract_perf_metrics_from_xs_trace(
            self.config.isa_base,
            self.config.core_id.clone(),
            harts,
            false,
            &build.disassembly_file,
            &trace,
            &commit_log_path,
        )?;
        Ok(Some(perf))
    }

    fn run_emu(
        &self,
        run_dir: &Path,
        build: &ElfBuildResult,
        multicore: bool,
    ) -> Result<XiangShanRunResult, XiangShanError> {
        let emu_path_raw = self.config.emulator_path(multicore)?;
        let emu_path = emu_path_raw.canonicalize().map_err(|err| {
            if emu_path_raw.exists() {
                XiangShanError::IoError(err)
            } else {
                XiangShanError::BinaryNotFound {
                    path: emu_path_raw.display().to_string(),
                }
            }
        })?;

        let internal_diff = internal_difftest_enabled();
        let diff_so = if internal_diff {
            let diff_so_raw = self.config.diff_so_path(multicore)?;
            Some(diff_so_raw.canonicalize().map_err(|err| {
                if diff_so_raw.exists() {
                    XiangShanError::IoError(err)
                } else {
                    XiangShanError::DiffSoNotFound {
                        path: diff_so_raw.display().to_string(),
                    }
                }
            })?)
        } else {
            None
        };

        let commit_log_path = run_dir.join("skiptrap.commit.log");
        let coverage_path = if self.config.coverage_mode.is_enabled() {
            Some(run_dir.join("coverage.dat"))
        } else {
            None
        };

        let mut cmd = Command::new(&emu_path);
        cmd.current_dir(run_dir);

        cmd.env(
            "DIFFTEST_LOG_ONLY",
            if difftest_log_only(internal_diff) {
                "1"
            } else {
                "0"
            },
        );
        cmd.env(
            "CX_RISCV_CORES_XIANGSHAN_ENABLE_INTERNAL_DIFF",
            if internal_diff { "1" } else { "0" },
        );

        let has_ram_size_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--ram-size" || arg.starts_with("--ram-size="));
        let has_max_cycles_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--max-cycles" || arg.starts_with("--max-cycles="));
        let has_max_instr_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--max-instr" || arg.starts_with("--max-instr="));
        let has_log_begin_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--log-begin" || arg.starts_with("--log-begin="));
        let has_log_end_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--log-end" || arg.starts_with("--log-end="));
        let has_dump_commit_trace_arg = self
            .config
            .extra_emu_args
            .iter()
            .any(|arg| arg == "--dump-commit-trace");

        // The XiangShan wrapper runs with `current_dir(run_dir)`. Resolve the built ELF
        // explicitly so missing or ambiguous paths fail immediately instead of silently degrading.
        let elf_arg = resolve_wrapper_elf_path(&build.executable_file, run_dir)?;

        cmd.arg("--image").arg(&elf_arg);
        if let Some(diff_so) = &diff_so {
            cmd.arg("--diff").arg(diff_so);
        } else {
            cmd.arg("--no-diff");
        }

        // Keep enough DRAM mapped for high-address RVV boundary probes near 0x9000_0000.
        if !has_ram_size_arg {
            cmd.arg("--ram-size=64GB");
        }

        // XiangShan can hang in the post-exit spin loop when GOODTRAP is not recognized
        // (e.g. difftest not loaded / misconfigured). Put conservative caps in place so
        // higher-level runners can rely on timeouts as the last resort.
        if !has_max_cycles_arg {
            // Large enough for normal testcases, small enough to bound worst-case.
            cmd.arg("--max-cycles=2000000");
        }
        if !has_max_instr_arg {
            cmd.arg("--max-instr=50000");
        }

        // Always request compact difftest commit trace so architectural register / exception
        // events do not depend on the heavyweight global debug stream.
        if !has_dump_commit_trace_arg {
            cmd.arg("--dump-commit-trace");
        }

        // Bound the runtime logging window so runaway cases stay finite even when the provider
        // binary was built with extra logging enabled.
        if !has_log_begin_arg {
            cmd.arg("--log-begin=0");
        }
        if !has_log_end_arg {
            cmd.arg("--log-end=200000");
        }

        if coverage_path.is_some() {
            cmd.arg("--dump-coverage");
        }
        if let Some(ref cov) = coverage_path {
            cmd.arg("--coverage-file").arg(cov);
        }
        for arg in &self.config.extra_emu_args {
            cmd.arg(arg);
        }
        let filtered_log = Arc::new(Mutex::new(fs::File::create(&commit_log_path).map_err(
            |e| {
                XiangShanError::ProcessError(ProcessError::FileWriteFailed {
                    path: commit_log_path.clone(),
                    source: e,
                })
            },
        )?));
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        configure_command_process_group(&mut cmd);

        let mut command_parts = Vec::new();
        command_parts.push(quote_os_str(emu_path.as_os_str()));
        command_parts.extend(cmd.get_args().map(quote_os_str));

        let command_display = format!(
            "cd \"{}\" && {}",
            run_dir.display(),
            command_parts.join(" ")
        );
        info!("Launching XiangShan command: {}", command_display);

        let mut child = cmd.spawn().map_err(|err| {
            XiangShanError::ProcessError(ProcessError::SpawnFailed {
                command: command_display.clone(),
                source: err,
            })
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            XiangShanError::ProcessError(ProcessError::SpawnFailed {
                command: command_display.clone(),
                source: std::io::Error::other("child stdout pipe missing"),
            })
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            XiangShanError::ProcessError(ProcessError::SpawnFailed {
                command: command_display.clone(),
                source: std::io::Error::other("child stderr pipe missing"),
            })
        })?;
        let stdout_log = Arc::clone(&filtered_log);
        let stdout_path = commit_log_path.clone();
        let stdout_drain =
            thread::spawn(move || drain_xs_log_stream(stdout, stdout_log, stdout_path));
        let stderr_log = Arc::clone(&filtered_log);
        let stderr_path = commit_log_path.clone();
        let stderr_drain =
            thread::spawn(move || drain_xs_log_stream(stderr, stderr_log, stderr_path));

        let timeout = self.config.effective_timeout(multicore);
        let start = Instant::now();
        let wait_result = loop {
            if let Some(status) = child.try_wait().map_err(|e| {
                XiangShanError::ProcessError(ProcessError::SpawnFailed {
                    command: command_display.clone(),
                    source: e,
                })
            })? {
                break Ok(status);
            }
            if let Some(limit) = timeout
                && start.elapsed() >= limit
            {
                let _ = kill_child_process_group(&mut child);
                let _ = child.wait();
                break Err(XiangShanError::ProcessError(ProcessError::TimedOut {
                    command: command_display.clone(),
                    timeout: limit,
                }));
            }
            // avoid busy spin
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        join_xs_log_drain(stdout_drain, "stdout")?;
        join_xs_log_drain(stderr_drain, "stderr")?;
        let status = wait_result?;

        let exit_disposition =
            classify_xiangshan_exit_status(status, &command_display, &commit_log_path)?;

        if let Some(cov_path) = &coverage_path {
            if !cov_path.exists() {
                return Err(XiangShanError::ProcessError(ProcessError::ProcessFailed {
                    command: command_display.clone(),
                    stderr: format!(
                        "coverage enabled but coverage file is missing: {}",
                        cov_path.display()
                    ),
                }));
            }

            let summary =
                riscv_fuzz_toolchain::coverage::parse_verilator_dat(cov_path).map_err(|err| {
                    XiangShanError::ProcessError(ProcessError::ProcessFailed {
                        command: command_display.clone(),
                        stderr: format!(
                            "coverage enabled but failed to parse {}: {err}",
                            cov_path.display()
                        ),
                    })
                })?;

            info!("Coverage file written to {}", cov_path.display());
            for line in summary.render_lines() {
                info!("  coverage: {line}");
            }
        }

        Ok(XiangShanRunResult {
            build: build.clone(),
            commit_log_path,
            exit_disposition,
        })
    }
}

fn extract_perf_metrics_from_xs_trace(
    _isa_base: ISABase,
    _riscv_impl: String,
    _harts: u32,
    _perf_enabled: bool,
    _dump_path: &Path,
    _trace: &XiangShanTrace,
    commit_log_path: &Path,
) -> Result<PerfMetrics, XiangShanError> {
    let log_text = fs::read_to_string(commit_log_path).map_err(|err| {
        XiangShanError::IoError(std::io::Error::new(
            err.kind(),
            format!(
                "failed to read XiangShan commit log {}: {err}",
                commit_log_path.display()
            ),
        ))
    })?;

    Ok(PerfMetrics {
        global_cycles: parse_global_cycles_from_text(&log_text),
        ..Default::default()
    })
}

fn xs_commit_log_has_commit_records(log_text: &str) -> bool {
    log_text.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with("retired hart")
            || trimmed.starts_with("commit pc")
            || (trimmed.starts_with('[') && trimmed.contains("] commit pc"))
    })
}

fn xs_commit_log_has_difftest_abort(log_text: &str) -> bool {
    log_text.contains(" different at pc = ")
        || log_text.contains("ABORT at pc =")
        || log_text.contains("HIT BAD TRAP")
}

fn read_xs_commit_log_for_error(commit_log_path: &Path) -> Result<String, XiangShanError> {
    fs::read_to_string(commit_log_path).map_err(|err| {
        XiangShanError::IoError(std::io::Error::new(
            err.kind(),
            format!(
                "failed to read XiangShan commit log {}: {err}",
                commit_log_path.display()
            ),
        ))
    })
}

fn xiangshan_unsuccessful_status_error(
    code: Option<i32>,
    log_text: &str,
    command_display: &str,
    commit_log_path: &Path,
) -> XiangShanError {
    if code == Some(1) && log_text.contains("Refill test failed!") {
        return XiangShanError::ProcessError(ProcessError::ProcessFailed {
            command: command_display.to_string(),
            stderr: format!(
                "XiangShan refill self-check aborted. See log: {}",
                commit_log_path.display()
            ),
        });
    }

    let stderr = if xs_commit_log_has_commit_records(log_text) {
        format!(
            "XiangShan emulator exited with status {:?} after producing commit records. Treating the commit trace as incomplete/truncated instead of returning partial per-instruction results. See log: {}",
            code,
            commit_log_path.display()
        )
    } else {
        format!(
            "XiangShan emulator exited with status {:?} before producing any commit records. See log: {}",
            code,
            commit_log_path.display()
        )
    };

    XiangShanError::ProcessError(ProcessError::ProcessFailed {
        command: command_display.to_string(),
        stderr,
    })
}

fn classify_xiangshan_exit_status(
    status: std::process::ExitStatus,
    command_display: &str,
    commit_log_path: &Path,
) -> Result<XiangShanExitDisposition, XiangShanError> {
    if status.success() {
        return Ok(XiangShanExitDisposition::Complete);
    }

    let log_text = read_xs_commit_log_for_error(commit_log_path)?;
    if status.code() == Some(1)
        && xs_commit_log_has_commit_records(&log_text)
        && xs_commit_log_has_difftest_abort(&log_text)
    {
        return Ok(XiangShanExitDisposition::PartialDifftestAbort);
    }

    Err(xiangshan_unsuccessful_status_error(
        status.code(),
        &log_text,
        command_display,
        commit_log_path,
    ))
}

#[derive(Debug)]
pub struct XiangShanRunResult {
    #[allow(dead_code)]
    pub build: ElfBuildResult,
    pub commit_log_path: PathBuf,
    pub exit_disposition: XiangShanExitDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XiangShanExitDisposition {
    Complete,
    PartialDifftestAbort,
}

impl XiangShanExitDisposition {
    fn requires_final_snapshot(self) -> bool {
        matches!(self, Self::Complete)
    }

    fn partial_stop_reason(self) -> Option<&'static str> {
        match self {
            Self::Complete => None,
            Self::PartialDifftestAbort => {
                Some("xiangshan difftest aborted after committing a partial trace")
            }
        }
    }
}

fn streaming_snapshot_from_xiangshan_trace(
    trace: &XiangShanTrace,
    tracers: &BTreeMap<u32, Tracer>,
    final_snapshot: bool,
) -> Result<BTreeMap<u32, HartExecutionOutput>, XiangShanError> {
    let per_hart = tracers
        .iter()
        .map(|(hart_id, tracer)| {
            build_execution_output(trace, *hart_id, tracer, final_snapshot)
                .map(|output| (*hart_id, output))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    if final_snapshot {
        validate_final_pc_completeness(trace, tracers)?;
        Ok(per_hart)
    } else {
        let stable_before_cycle_by_hart = tracers
            .keys()
            .map(|hart_id| (*hart_id, trace.max_cycle_for_hart(*hart_id)))
            .collect::<BTreeMap<_, _>>();
        Ok(filter_per_hart_output_for_streaming(
            &per_hart,
            tracers,
            &stable_before_cycle_by_hart,
        ))
    }
}

fn filter_per_hart_output_for_streaming(
    per_hart: &BTreeMap<u32, HartExecutionOutput>,
    tracers: &BTreeMap<u32, Tracer>,
    stable_before_cycle_by_hart: &BTreeMap<u32, Option<u64>>,
) -> BTreeMap<u32, HartExecutionOutput> {
    per_hart
        .iter()
        .map(|(hart_id, hart)| {
            let Some(tracer) = tracers.get(hart_id) else {
                return (
                    *hart_id,
                    HartExecutionOutput {
                        hart_id: hart.hart_id,
                        instructions: Vec::new(),
                    },
                );
            };
            (
                *hart_id,
                filter_hart_output_for_streaming(
                    hart,
                    tracer,
                    stable_before_cycle_by_hart.get(hart_id).copied().flatten(),
                ),
            )
        })
        .collect()
}

fn filter_hart_output_for_streaming(
    hart: &HartExecutionOutput,
    tracer: &Tracer,
    stable_before_cycle: Option<u64>,
) -> HartExecutionOutput {
    let mut instructions = Vec::new();
    for instruction in &hart.instructions {
        if instruction.trace_inst_id.is_none() {
            continue;
        }
        if !instruction_ready_for_streaming(instruction, tracer, stable_before_cycle) {
            break;
        }
        instructions.push(instruction.clone());
    }

    HartExecutionOutput {
        hart_id: hart.hart_id,
        instructions,
    }
}

fn instruction_ready_for_streaming(
    instruction: &AsmInstructionExecution,
    tracer: &Tracer,
    stable_before_cycle: Option<u64>,
) -> bool {
    let Some(limit) = stable_before_cycle else {
        return false;
    };
    let Some(cycle) = instruction.timing.clk_start.or(instruction.timing.clk_end) else {
        return false;
    };
    if cycle >= limit {
        return false;
    }
    if !instruction.exceptions.is_empty() {
        return true;
    }

    let Some(trace_instruction) = tracer.instruction_for_pc(&instruction.pc) else {
        return true;
    };
    let Some(op) = opcode(&trace_instruction.disassembly) else {
        return true;
    };
    if scalar_store_width(op).is_some() || float_store_width(op).is_some() {
        return !instruction.memory_writes.is_empty();
    }
    let expected = expected_destination_registers_for_traced_instruction(trace_instruction);
    if expected.is_empty() {
        return true;
    }
    instruction
        .register_writes
        .iter()
        .filter_map(|write| canonicalize_register_name(&write.name))
        .collect::<HashSet<_>>()
        .is_superset(&expected)
}

fn expected_source_pcs(tracer: &Tracer, source_index: usize) -> Vec<u64> {
    tracer
        .get_pcs_for_trace_inst(source_index)
        .cloned()
        .unwrap_or_default()
}

fn normalize_disassembly_for_decode(disassembly: &str) -> String {
    disassembly
        .replace(',', ", ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn destination_registers_for_instruction(
    instruction: &riscv_instruction::RiscvInstruction,
) -> HashSet<String> {
    instruction
        .register_operands()
        .into_iter()
        .filter(|operand| operand.usage == RuntimeOperandUsage::Destination)
        .filter_map(|operand| match operand.register.bank {
            RegisterBank::Integer if operand.register.physical_number == 0 => None,
            RegisterBank::Integer => Some(format!("x{}", operand.register.physical_number)),
            RegisterBank::FloatingPoint => Some(format!("f{}", operand.register.physical_number)),
            RegisterBank::Vector => Some(format!("v{}", operand.register.physical_number)),
        })
        .collect()
}

fn expected_destination_registers(disassembly: &str) -> HashSet<String> {
    let normalized = normalize_disassembly_for_decode(disassembly);
    let instruction = riscv_instruction::RiscvInstruction::decode_asm_with_context(
        &normalized,
        riscv_instruction::IsaDecodeContext::RV64,
    )
    .ok();
    let Some(instruction) = instruction else {
        return HashSet::new();
    };
    destination_registers_for_instruction(&instruction)
}

fn expected_destination_registers_for_traced_instruction(
    trace_instruction: &TracedInstruction,
) -> HashSet<String> {
    if let Some(instruction) = trace_instruction
        .parsed_disassembly
        .as_ref()
        .or(trace_instruction.source_instruction.as_ref())
    {
        return destination_registers_for_instruction(instruction);
    }
    expected_destination_registers(&trace_instruction.disassembly)
}

fn build_execution_output(
    trace: &XiangShanTrace,
    hart_id: u32,
    tracer: &Tracer,
    final_snapshot: bool,
) -> Result<HartExecutionOutput, XiangShanError> {
    validate_trace_records(trace)?;
    let associated_memory = associate_memory_writes(trace, hart_id, tracer);
    let register_writes =
        associate_missing_register_writes(trace, hart_id, tracer, &associated_memory);
    let observed_pcs = trace
        .registers
        .keys()
        .filter(|key| key.hart == hart_id)
        .map(|key| key.pc)
        .chain(register_writes.keys().copied())
        .chain(associated_memory.keys().copied())
        .chain(
            trace
                .exceptions
                .keys()
                .filter(|key| key.hart == hart_id)
                .map(|key| key.pc),
        )
        .chain(
            trace
                .committed_pcs
                .iter()
                .filter(|key| key.hart == hart_id)
                .map(|key| key.pc),
        )
        .chain(
            trace
                .instruction_timings
                .keys()
                .filter(|key| key.hart == hart_id)
                .map(|key| key.pc),
        )
        .collect::<Vec<_>>();

    let observed_pc_set = observed_pcs.iter().copied().collect::<HashSet<_>>();
    if final_snapshot {
        validate_final_observed_pcs_complete(tracer, hart_id, &observed_pc_set, trace)?;
    }
    let output = build_raw_hart_execution_output(
        tracer,
        hart_id,
        observed_pcs,
        |pc| {
            let mut regs = Vec::new();
            if let Some(writes) = register_writes.get(&pc) {
                for write in writes {
                    regs.push(TraitRegisterValue {
                        name: write.register.clone(),
                        value: write.value,
                    });
                }
            }
            regs.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.value.cmp(&b.value)));
            regs
        },
        |pc| {
            let mut mems = associated_memory
                .get(&pc)
                .into_iter()
                .flat_map(|writes| writes.iter())
                .flat_map(|write| write.bytes.iter().copied())
                .map(|(addr, value)| TraitMemValue { addr, value })
                .collect::<Vec<_>>();
            mems.sort_by_key(|mem| mem.addr);
            mems
        },
        |pc| {
            trace
                .exception(hart_id, pc)
                .map(|event| {
                    if event.cause == UNKNOWN_EXCEPTION_CODE {
                        vec![ExceptionCause::unknown("unknown XiangShan exceptionVec")]
                    } else {
                        vec![ExceptionCause::from_code(
                            event.cause,
                            format!("0x{:x}", event.cause),
                        )]
                    }
                })
                .unwrap_or_default()
        },
        |pc| {
            let mut timing_opt: Option<(u64, InstructionTiming)> = None;
            if let Some(entries) = trace.instruction_timing_entries(hart_id, pc) {
                for entry in entries {
                    let key = entry
                        .timing
                        .clk_end
                        .or(entry.timing.clk_start)
                        .or(entry.timing.clk_span)
                        .unwrap_or(0);
                    match &mut timing_opt {
                        Some((best_key, best_timing)) if key >= *best_key => {
                            *best_key = key;
                            *best_timing = entry.timing.clone();
                        }
                        None => timing_opt = Some((key, entry.timing.clone())),
                        _ => {}
                    }
                }
            }
            timing_opt.map(|(_, timing)| timing).unwrap_or_default()
        },
    );
    Ok(filter_unobserved_output(output, &observed_pc_set))
}

fn validate_final_observed_pcs_complete(
    tracer: &Tracer,
    hart_id: u32,
    observed_pcs: &HashSet<u64>,
    trace: &XiangShanTrace,
) -> Result<(), XiangShanError> {
    let mut checked = HashSet::new();
    for instruction in tracer.instructions() {
        let Some(source_index) = instruction.trace_inst_local_index else {
            continue;
        };
        if !checked.insert(source_index) {
            continue;
        }
        let Some(expected) = tracer.get_pcs_for_trace_inst(source_index) else {
            continue;
        };
        if expected
            .iter()
            .any(|pc| trace.exceptions_contain_pc(hart_id, *pc))
        {
            continue;
        }
        if !expected.iter().any(|pc| observed_pcs.contains(pc)) {
            let missing = expected.first().copied().unwrap_or_default();
            return Err(XiangShanError::IncompleteTrace {
                source_index,
                detail: format!("missing pc=0x{missing:x} before output construction"),
            });
        }
        // XiangShan can retire fused scalar pairs as one commit record at the
        // first physical PC while the ELF mapping still contains every PC from
        // the source pseudo-instruction expansion. Diffing happens at logical
        // source-instruction granularity, so require that the logical instruction
        // is observed at least once and let later context construction aggregate
        // the observed writes. A fully missing logical instruction remains a hard
        // incomplete-trace error above.
    }
    Ok(())
}

fn filter_unobserved_output(
    mut output: HartExecutionOutput,
    observed_pcs: &HashSet<u64>,
) -> HartExecutionOutput {
    output
        .instructions
        .retain(|row| observed_pcs.contains(&row.pc));
    output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OutputRowLocation {
    row_index: usize,
}

fn row_order_key(row: &AsmInstructionExecution) -> (u64, u64) {
    (
        row.timing
            .clk_end
            .or(row.timing.clk_start)
            .or(row.timing.clk_span)
            .unwrap_or(0),
        row.pc,
    )
}

fn terminal_output_row_location(output: &HartExecutionOutput) -> Option<OutputRowLocation> {
    output
        .instructions
        .iter()
        .enumerate()
        .max_by_key(|(_, row)| row_order_key(row))
        .map(|(row_index, _)| OutputRowLocation { row_index })
}

fn annotate_partial_trace_termination(output: &mut HartExecutionOutput, reason: &str) {
    let Some(location) = terminal_output_row_location(output) else {
        return;
    };

    let marker = ExceptionCause::unknown(reason);
    if let Some(row) = output.instructions.get_mut(location.row_index)
        && row.exceptions.is_empty()
    {
        row.exceptions.push(marker);
    }
}

fn validate_trace_records(trace: &XiangShanTrace) -> Result<(), XiangShanError> {
    for (key, writes) in &trace.memory {
        let mut seen = HashMap::<u64, u8>::new();
        for write in writes {
            for (addr, value) in &write.bytes {
                if let Some(previous) = seen.insert(*addr, *value) {
                    let detail = format!(
                        "memory byte 0x{addr:016x} appears more than once (0x{previous:02x}, 0x{value:02x})"
                    );
                    return Err(if previous == *value {
                        XiangShanError::DuplicateTraceRecord {
                            kind: "memory",
                            hart: key.hart,
                            pc: key.pc,
                            detail,
                        }
                    } else {
                        XiangShanError::ConflictingTraceRecord {
                            kind: "memory",
                            hart: key.hart,
                            pc: key.pc,
                            detail,
                        }
                    });
                }
            }
        }
    }

    Ok(())
}

fn validate_final_pc_completeness(
    trace: &XiangShanTrace,
    tracers: &BTreeMap<u32, Tracer>,
) -> Result<(), XiangShanError> {
    for (&hart_id, tracer) in tracers {
        let observed = observed_trace_pcs(trace, hart_id);
        for source_index in source_indices_for_tracer(tracer) {
            let expected = expected_source_pcs(tracer, source_index);
            if expected.is_empty() {
                continue;
            }
            let observed_expected = expected
                .iter()
                .copied()
                .filter(|pc| observed.contains(pc))
                .collect::<Vec<_>>();
            if observed_expected.is_empty() {
                continue;
            }
            if let Some(exception_pc) = expected
                .iter()
                .copied()
                .find(|pc| trace.exceptions_contain_pc(hart_id, *pc))
            {
                let mut required = Vec::new();
                for pc in expected.iter().copied() {
                    required.push(pc);
                    if pc == exception_pc {
                        break;
                    }
                }
                let missing = required
                    .iter()
                    .copied()
                    .filter(|pc| !observed.contains(pc))
                    .collect::<Vec<_>>();
                if !missing.is_empty() {
                    return Err(XiangShanError::IncompleteTrace {
                        source_index,
                        detail: format!(
                            "missing PCs before exception: {}",
                            format_pc_list(&missing)
                        ),
                    });
                }
                let after_exception = expected
                    .iter()
                    .copied()
                    .skip_while(|pc| *pc != exception_pc)
                    .skip(1)
                    .filter(|pc| observed.contains(pc))
                    .collect::<Vec<_>>();
                if !after_exception.is_empty() {
                    return Err(XiangShanError::ConflictingTraceRecord {
                        kind: "source instruction",
                        hart: hart_id,
                        pc: exception_pc,
                        detail: format!(
                            "instruction has exception and later observed PCs: {}",
                            format_pc_list(&after_exception)
                        ),
                    });
                }
            } else {
                // Non-exception source instructions are compared after collapsing
                // all physical PCs that belong to the same logical input
                // instruction. XiangShan fused commits may omit intermediate PCs
                // from pseudo-instruction expansions, but the source instruction
                // is still traceable when at least one expected PC was observed
                // (checked by observed_expected above).
            }
        }
    }
    Ok(())
}

fn observed_trace_pcs(trace: &XiangShanTrace, hart_id: u32) -> HashSet<u64> {
    trace
        .registers
        .keys()
        .chain(trace.memory.keys())
        .chain(trace.exceptions.keys())
        .chain(trace.instruction_timings.keys())
        .filter(|key| key.hart == hart_id)
        .map(|key| key.pc)
        .collect()
}

fn source_indices_for_tracer(tracer: &Tracer) -> Vec<usize> {
    let mut indices = tracer
        .instructions()
        .iter()
        .filter_map(|inst| inst.trace_inst_local_index)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices
}

fn format_pc_list(pcs: &[u64]) -> String {
    pcs.iter()
        .map(|pc| format!("0x{pc:016x}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn associate_memory_writes(
    trace: &XiangShanTrace,
    hart_id: u32,
    _tracer: &Tracer,
) -> HashMap<u64, Vec<MemoryWrite>> {
    trace
        .memory
        .iter()
        .filter(|(key, _)| key.hart == hart_id)
        .map(|(key, writes)| (key.pc, writes.clone()))
        .collect()
}

fn associate_missing_register_writes(
    trace: &XiangShanTrace,
    hart_id: u32,
    _tracer: &Tracer,
    _associated_memory: &HashMap<u64, Vec<MemoryWrite>>,
) -> HashMap<u64, Vec<RegisterWrite>> {
    trace
        .registers
        .iter()
        .filter(|(key, _)| key.hart == hart_id)
        .map(|(key, writes)| (key.pc, writes.clone()))
        .collect()
}

fn opcode(disassembly: &str) -> Option<&str> {
    disassembly.split_whitespace().next()
}

fn scalar_store_width(opcode: &str) -> Option<usize> {
    match opcode {
        "sb" | "c.sb" => Some(1),
        "sh" | "c.sh" => Some(2),
        "sw" | "c.sw" | "c.swsp" => Some(4),
        "sd" | "c.sd" | "c.sdsp" => Some(8),
        _ => None,
    }
}

fn float_store_width(opcode: &str) -> Option<usize> {
    match opcode {
        "fsh" => Some(2),
        "fsw" | "c.fsw" | "c.fswsp" => Some(4),
        "fsd" | "c.fsd" | "c.fsdsp" => Some(8),
        _ => None,
    }
}

fn xiangshan_section_labels_for_hart(
    harts: u32,
    hart_id: u32,
) -> &'static [(&'static str, &'static str)] {
    match (harts > 1, hart_id) {
        (true, 1) => &[("hart1_testcase_code:", "hart1_testcase_code_end:")],
        _ => &[("testcase_code:", "testcase_code_end:")],
    }
}

fn register_number(register: &str) -> Option<u32> {
    parse_register_name(register).map(|parsed| parsed.index as u32)
}

fn commit_disassembly_text(rest: &str) -> &str {
    let trimmed = rest.trim_start();
    if trimmed.starts_with('(')
        && let Some(close_idx) = trimmed.find(')')
    {
        return trimmed[close_idx + 1..].trim_start();
    }
    trimmed
}

fn commit_first_operand(rest: &str) -> Option<&str> {
    let asm = commit_disassembly_text(rest);
    let space_idx = asm.find(char::is_whitespace)?;
    let operands = asm[space_idx..].trim_start();
    operands.split(',').next().map(str::trim)
}

fn commit_destination_register(
    dst_num: u32,
    dst_prefix: Option<&str>,
    rest: &str,
) -> Option<String> {
    if dst_num > 31 {
        if dst_prefix.is_none()
            && let Some(first_operand) = commit_first_operand(rest)
            && let Some(parsed) = parse_register_name(first_operand)
            && parsed.bank == RegisterBank::Vector
        {
            return Some(parsed.canonical_name());
        }
        return None;
    }

    if let Some(prefix) = dst_prefix {
        return Some(match prefix {
            "f" | "F" => format!("f{}", dst_num),
            "v" | "V" => format!("v{}", dst_num),
            _ => format!("x{}", dst_num),
        });
    }

    if let Some(first_operand) = commit_first_operand(rest)
        && let Some(parsed) = parse_register_name(first_operand)
        && u32::from(parsed.index) == dst_num
    {
        return Some(
            first_operand
                .trim()
                .trim_end_matches(',')
                .to_ascii_lowercase(),
        );
    }

    Some(format!("x{}", dst_num))
}

fn parse_xs_commit_trace<P: AsRef<Path>>(path: P) -> Result<XiangShanTrace, XiangShanError> {
    let file = fs::File::open(path.as_ref()).map_err(|e| {
        XiangShanError::ProcessError(ProcessError::LogFileOpenFailed {
            path: path.as_ref().to_path_buf(),
            source: e,
        })
    })?;
    let reader = BufReader::new(file);
    parse_xs_commit_trace_reader(reader)
}

fn parse_xs_commit_trace_reader<R: BufRead>(reader: R) -> Result<XiangShanTrace, XiangShanError> {
    static RE_COMMIT: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^\[(?P<log_a>\d+)\](?:\[(?P<log_b>\d+)\])?\s+commit pc (?P<pc>[0-9a-fA-F]{16}) inst (?P<inst>[0-9a-fA-F]{8}) wen (?P<wen>\d+) dst (?:(?P<dstprefix>[xXfFvV]))?(?P<dst>\d{1,2}) data (?P<data>[0-9a-fA-F]{16}) idx (?P<rob>[0-9a-fA-F]{3})(?P<rest>.*)$",
        )
        .unwrap()
    });
    static RE_EXCEPTION: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^\[(?P<log_a>\d+)\](?:\[(?P<log_b>\d+)\])?\s+exception pc (?P<pc>[0-9a-fA-F]{16}) inst (?P<inst>[0-9a-fA-F]{8}) cause (?P<cause>[0-9a-fA-F]{16})(?P<rest>.*)$",
        )
        .unwrap()
    });
    static RE_MEM_LINE: Lazy<Regex> = Lazy::new(|| {
        // log-only mode standalone mem line: "mem pc <pc> addr <addr> data <data> mask 0x.."
        Regex::new(r"^mem pc (?P<pc>[0-9a-fA-F]{16}) addr (?P<addr>[0-9a-fA-F]{16}) data (?P<data>[0-9a-fA-F]{16}) mask 0x(?P<mask>[0-9a-fA-F]{2})$").unwrap()
    });
    static RE_INLINE_MEM_IN_COMMIT: Lazy<Regex> = Lazy::new(|| {
        // Inline memory fragment on commit line: "... addr <addr> data <data> mask 0x.. <asm>"
        Regex::new(
            r"addr (?P<addr>[0-9a-fA-F]{16}) data (?P<data>[0-9a-fA-F]{16}) mask 0x(?P<mask>[0-9a-fA-F]{2})",
        )
        .unwrap()
    });
    static RE_RETIRE_TIMING: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^.*?\bretired\s+hart\s+(?P<hart>\d+)\s+pc\s+(?P<pc>[0-9a-fA-F]+)\s+wen\s+(?P<wen>\d+)\s+ldest\s+(?P<ldest>\d+)\s+pdest\s+(?P<pdest>[0-9a-fA-F]+)\s+data\s+(?P<data>[0-9a-fA-F]+)\s+fflags:\s+\S+\s+vxsat:\s+\S+\s+clk_start\s+(?P<clk_start>\d+)\s+clk_end\s+(?P<clk_end>\d+)\s+clk_span\s+(?P<clk_span>\d+)$",
        )
        .unwrap()
    });
    static RE_STORE_COMMIT_TIMING: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^.*?\bstore\s+commit\s+hart\s+(?P<hart>\d+)\s+pc\s+0x(?P<pc>[0-9a-fA-F]+)\s+robidx\s+\d+\s+addr\s+(?P<addr>[0-9a-fA-F]+)\s+data_lo\s+(?P<data_lo>[0-9a-fA-F]+)\s+data_hi\s+(?P<data_hi>[0-9a-fA-F]+)\s+mask\s+(?P<mask>[0-9a-fA-F]+)\s+wline\s+[0-9a-fA-F]+\s+vecsplit\s+[0-9a-fA-F]+\s+clk_start\s+(?P<clk_start>\d+)\s+clk_end\s+(?P<clk_end>\d+)\s+clk_span\s+(?P<clk_span>\d+)$",
        )
        .unwrap()
    });
    static RE_TRAP_TIMING: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"^.*?\btrap\s+hart\s+(?P<hart>\d+)\s+pc\s+(?P<pc>[0-9a-fA-F]+)\s+intr\s+(?P<intr>\d+)\s+excp\s+(?P<excp>\d+)\s+exceptionVec\s+(?P<exception_vec>[0-9a-fA-F]+)(?:\s+clk_start\s+(?P<clk_start>\d+)\s+clk_end\s+(?P<clk_end>\d+)\s+clk_span\s+(?P<clk_span>\d+))?.*$",
        )
        .unwrap()
    });

    let mut trace = XiangShanTrace::default();
    let mut harts_for_pc = HashMap::<u64, BTreeSet<u32>>::new();
    let mut pending_mem_lines = Vec::<PendingStandaloneMemLine>::new();
    let mut recognized_trace_line_count = 0usize;
    let mut saw_usage_header = false;
    let mut usage_error_line: Option<String> = None;
    let parse_u32_dec = |value: &str| -> Result<u32, XiangShanError> {
        value.parse().map_err(|source| {
            XiangShanError::ParseError(ParseError::IntParseError {
                value: value.to_string(),
                source,
            })
        })
    };
    let parse_u64_hex = |value: &str| -> Result<u64, XiangShanError> {
        u64::from_str_radix(value, 16).map_err(|source| {
            XiangShanError::ParseError(ParseError::HexParseError {
                value: value.to_string(),
                source,
            })
        })
    };
    let parse_timing = |caps: &regex::Captures<'_>| -> Result<InstructionTiming, XiangShanError> {
        let parse_component = |name: &str| -> Result<Option<u64>, XiangShanError> {
            caps.name(name)
                .map(|m| {
                    m.as_str().parse::<u64>().map_err(|source| {
                        XiangShanError::ParseError(ParseError::IntParseError {
                            value: m.as_str().to_string(),
                            source,
                        })
                    })
                })
                .transpose()
        };
        Ok(InstructionTiming::new(
            parse_component("clk_start")?,
            parse_component("clk_end")?,
            parse_component("clk_span")?,
        ))
    };
    let parse_compact_hart = |caps: &regex::Captures<'_>| -> Result<u32, XiangShanError> {
        if caps.name("log_b").is_some() {
            parse_u32_dec(caps.name("log_a").unwrap().as_str())
        } else {
            Ok(0)
        }
    };
    for line in reader.lines() {
        let line = line.map_err(XiangShanError::IoError)?;
        let raw_trimmed = line.trim();
        if raw_trimmed.is_empty() {
            continue;
        }
        let trimmed = extract_xs_log_payload(raw_trimmed).unwrap_or(raw_trimmed);
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("Usage: ") {
            saw_usage_header = true;
            continue;
        }
        if trimmed.contains("unrecognized option") {
            usage_error_line = Some(trimmed.to_string());
            continue;
        }

        if let Some(cap) = RE_RETIRE_TIMING.captures(trimmed) {
            recognized_trace_line_count += 1;
            let hart = parse_u32_dec(cap.name("hart").unwrap().as_str())?;
            let pc = parse_u64_hex(cap.name("pc").unwrap().as_str())?;
            let timing = parse_timing(&cap)?;
            harts_for_pc.entry(pc).or_default().insert(hart);
            trace
                .instruction_timings
                .entry(TraceKey::new(hart, pc))
                .or_default()
                .push(InstructionTrace {
                    pc,
                    timing: timing.clone(),
                });
            let ldest = parse_u32_dec(cap.name("ldest").unwrap().as_str())?;
            if ldest != 0 {
                merge_timing_into_existing_register_write(&mut trace, hart, pc, ldest, &timing);
            }
            continue;
        }

        if let Some(cap) = RE_STORE_COMMIT_TIMING.captures(trimmed) {
            recognized_trace_line_count += 1;
            let hart = parse_u32_dec(cap.name("hart").unwrap().as_str())?;
            let pc = parse_u64_hex(cap.name("pc").unwrap().as_str())?;
            let addr = parse_u64_hex(cap.name("addr").unwrap().as_str())?;
            let data_lo = cap.name("data_lo").unwrap().as_str();
            let data_hi = cap.name("data_hi").unwrap().as_str();
            let mask = cap.name("mask").unwrap().as_str();
            let timing = parse_timing(&cap)?;
            harts_for_pc.entry(pc).or_default().insert(hart);
            trace
                .instruction_timings
                .entry(TraceKey::new(hart, pc))
                .or_default()
                .push(InstructionTrace {
                    pc,
                    timing: timing.clone(),
                });
            let bytes = expand_store_bytes_wide(addr, data_lo, data_hi, mask)?;
            if !merge_timing_into_existing_memory_write(&mut trace, hart, pc, addr, &bytes, &timing)
            {
                upsert_memory_write(&mut trace, hart, pc, addr, bytes, timing)?;
            }
            continue;
        }

        if let Some(cap) = RE_TRAP_TIMING.captures(trimmed) {
            recognized_trace_line_count += 1;
            let hart = parse_u32_dec(cap.name("hart").unwrap().as_str())?;
            let pc = parse_u64_hex(cap.name("pc").unwrap().as_str())?;
            let timing = parse_timing(&cap)?;
            harts_for_pc.entry(pc).or_default().insert(hart);
            trace
                .instruction_timings
                .entry(TraceKey::new(hart, pc))
                .or_default()
                .push(InstructionTrace {
                    pc,
                    timing: timing.clone(),
                });
            if let Some(existing) = trace.exceptions.get_mut(&TraceKey::new(hart, pc)) {
                merge_instruction_timing(&mut existing.timing, &timing);
            }
            continue;
        }

        if let Some(cap) = RE_COMMIT.captures(trimmed) {
            recognized_trace_line_count += 1;
            let hart = parse_compact_hart(&cap)?;
            let pc = parse_u64_hex(cap.name("pc").unwrap().as_str())?;
            harts_for_pc.entry(pc).or_default().insert(hart);
            trace.committed_pcs.insert(TraceKey::new(hart, pc));
            let wen = cap.name("wen").unwrap().as_str() == "1";
            let rest = cap
                .name("rest")
                .ok_or_else(|| {
                    XiangShanError::ParseError(ParseError::ValueParseError {
                        text: trimmed.to_string(),
                    })
                })?
                .as_str();
            if wen {
                let dst_str = cap.name("dst").unwrap().as_str();
                // Normalize register index to remove any leading zeros (e.g., x03 -> x3)
                let dst_num = parse_u32_dec(dst_str)?;
                let data = parse_u64_hex(cap.name("data").unwrap().as_str())?;

                if let Some(register) = commit_destination_register(
                    dst_num,
                    cap.name("dstprefix").map(|m| m.as_str()),
                    rest,
                ) {
                    let timing = aggregated_instruction_timing(&trace, hart, pc);
                    upsert_register_write(&mut trace, hart, pc, register, data, timing)?;
                }
            }

            // Parse inline memory write fragments on commit lines (newer XiangShan logs).
            // Older logs also emit standalone "mem pc ..." lines; we deduplicate here.
            if let Some(mem_cap) = RE_INLINE_MEM_IN_COMMIT.captures(rest) {
                let addr = parse_u64_hex(mem_cap.name("addr").unwrap().as_str())?;
                let data = parse_u64_hex(mem_cap.name("data").unwrap().as_str())?;
                let mask = u8::from_str_radix(mem_cap.name("mask").unwrap().as_str(), 16).map_err(
                    |source| {
                        XiangShanError::ParseError(ParseError::HexParseError {
                            value: mem_cap.name("mask").unwrap().as_str().to_string(),
                            source,
                        })
                    },
                )?;
                let bytes = expand_store_bytes(addr, data, mask);
                let timing = aggregated_instruction_timing(&trace, hart, pc);
                upsert_memory_write(&mut trace, hart, pc, addr, bytes, timing)?;
            }

            continue;
        }

        if let Some(cap) = RE_EXCEPTION.captures(trimmed) {
            recognized_trace_line_count += 1;
            let hart = parse_compact_hart(&cap)?;
            let pc = parse_u64_hex(cap.name("pc").unwrap().as_str())?;
            let cause = parse_u64_hex(cap.name("cause").unwrap().as_str())?;
            harts_for_pc.entry(pc).or_default().insert(hart);
            let key = TraceKey::new(hart, pc);
            let timing = aggregated_instruction_timing(&trace, hart, pc);
            match trace.exceptions.get_mut(&key) {
                Some(existing) if existing.cause == cause => {
                    merge_instruction_timing(&mut existing.timing, &timing);
                }
                Some(existing) => {
                    return Err(XiangShanError::ConflictingTraceRecord {
                        kind: "exception",
                        hart,
                        pc,
                        detail: format!(
                            "exception cause appears more than once (0x{:016x}, 0x{cause:016x})",
                            existing.cause
                        ),
                    });
                }
                None => {
                    trace
                        .exceptions
                        .insert(key, ExceptionEvent { pc, cause, timing });
                }
            }
            continue;
        }

        if let Some(cap) = RE_MEM_LINE.captures(trimmed) {
            recognized_trace_line_count += 1;
            let pc = parse_u64_hex(cap.name("pc").unwrap().as_str())?;
            let addr = parse_u64_hex(cap.name("addr").unwrap().as_str())?;
            let data = parse_u64_hex(cap.name("data").unwrap().as_str())?;
            let mask =
                u8::from_str_radix(cap.name("mask").unwrap().as_str(), 16).map_err(|source| {
                    XiangShanError::ParseError(ParseError::HexParseError {
                        value: cap.name("mask").unwrap().as_str().to_string(),
                        source,
                    })
                })?;
            pending_mem_lines.push(PendingStandaloneMemLine {
                pc,
                addr,
                data,
                mask,
            });
            continue;
        }

        debug!("Unrecognized XiangShan commit trace line: {}", trimmed);
    }

    if recognized_trace_line_count == 0 && saw_usage_header {
        let detail = usage_error_line.unwrap_or_else(|| {
            "XiangShan emulator printed usage/help instead of commit trace".to_string()
        });
        return Err(XiangShanError::ParseError(ParseError::ValueParseError {
            text: detail,
        }));
    }

    for pending in pending_mem_lines {
        let harts = harts_for_pc.get(&pending.pc).ok_or_else(|| {
            XiangShanError::ParseError(ParseError::ValueParseError {
                text: format!(
                    "mem line for pc 0x{:016x} has no explicit hart context anywhere in the trace",
                    pending.pc
                ),
            })
        })?;
        if harts.len() != 1 {
            let observed = harts
                .iter()
                .map(|hart| hart.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(XiangShanError::ParseError(ParseError::ValueParseError {
                text: format!(
                    "mem line for pc 0x{:016x} is ambiguous across harts: observed explicit contexts {{{observed}}}",
                    pending.pc
                ),
            }));
        }
        let hart = *harts.iter().next().expect("checked len == 1");
        let bytes = expand_store_bytes(pending.addr, pending.data, pending.mask);
        let timing = aggregated_instruction_timing(&trace, hart, pending.pc);
        if merge_timing_into_existing_memory_write(
            &mut trace,
            hart,
            pending.pc,
            pending.addr,
            &bytes,
            &timing,
        ) {
            continue;
        }
        upsert_memory_write(&mut trace, hart, pending.pc, pending.addr, bytes, timing)?;
    }

    Ok(trace)
}

fn expand_store_bytes(addr: u64, data: u64, mask: u8) -> Vec<(u64, u8)> {
    let mut result = Vec::new();
    for i in 0..8 {
        if (mask & (1 << i)) != 0 {
            let byte = ((data >> (i * 8)) & 0xFF) as u8;
            result.push((addr + i as u64, byte));
        }
    }
    result
}

fn parse_hex_bytes_le(value: &str) -> Result<Vec<u8>, XiangShanError> {
    let normalized = if value.len().is_multiple_of(2) {
        value.to_string()
    } else {
        format!("0{value}")
    };

    let mut result = Vec::with_capacity(normalized.len() / 2);
    for idx in (0..normalized.len()).step_by(2).rev() {
        let byte = u8::from_str_radix(&normalized[idx..idx + 2], 16).map_err(|source| {
            XiangShanError::ParseError(ParseError::HexParseError {
                value: value.to_string(),
                source,
            })
        })?;
        result.push(byte);
    }
    Ok(result)
}

fn expand_store_bytes_wide(
    addr: u64,
    data_lo: &str,
    data_hi: &str,
    mask: &str,
) -> Result<Vec<(u64, u8)>, XiangShanError> {
    let mut result = Vec::new();
    let mut data_bytes = parse_hex_bytes_le(data_lo)?;
    data_bytes.extend(parse_hex_bytes_le(data_hi)?);
    let mask_bytes = parse_hex_bytes_le(mask)?;

    for lane in 0..(mask_bytes.len() * 8) {
        let mask_byte = mask_bytes[lane / 8];
        if (mask_byte & (1u8 << (lane % 8))) != 0 {
            let byte = data_bytes.get(lane).copied().ok_or_else(|| {
                XiangShanError::ParseError(ParseError::ValueParseError {
                    text: format!(
                        "wide store mask selects byte lane {lane}, but data has only {} byte(s): data_lo={data_lo} data_hi={data_hi} mask={mask}",
                        data_bytes.len()
                    ),
                })
            })?;
            result.push((addr + lane as u64, byte));
        }
    }
    Ok(result)
}

fn merge_instruction_timing(existing: &mut InstructionTiming, candidate: &InstructionTiming) {
    if !candidate.is_recorded() {
        return;
    }
    if !existing.is_recorded() {
        *existing = candidate.clone();
        return;
    }

    if existing.clk_start.is_none() {
        existing.clk_start = candidate.clk_start;
    }
    if existing.clk_end.is_none() {
        existing.clk_end = candidate.clk_end;
    }
    if existing.clk_span.is_none() {
        existing.clk_span = candidate.clk_span;
    }
    *existing = existing.clone().normalized();
}

fn aggregated_instruction_timing(trace: &XiangShanTrace, hart: u32, pc: u64) -> InstructionTiming {
    let mut merged = InstructionTiming::default();
    let Some(entries) = trace.instruction_timings.get(&TraceKey::new(hart, pc)) else {
        return merged;
    };
    for entry in entries {
        merge_instruction_timing(&mut merged, &entry.timing);
    }
    merged
}

fn merge_timing_into_existing_register_write(
    trace: &mut XiangShanTrace,
    hart: u32,
    pc: u64,
    logical_dest: u32,
    timing: &InstructionTiming,
) -> bool {
    let Some(entry) = trace.registers.get_mut(&TraceKey::new(hart, pc)) else {
        return false;
    };
    let Some(existing) = entry
        .iter_mut()
        .find(|write| register_number(&write.register) == Some(logical_dest))
    else {
        return false;
    };
    merge_instruction_timing(&mut existing.timing, timing);
    true
}

fn merge_timing_into_existing_memory_write(
    trace: &mut XiangShanTrace,
    hart: u32,
    pc: u64,
    addr: u64,
    bytes: &[(u64, u8)],
    timing: &InstructionTiming,
) -> bool {
    let Some(entry) = trace.memory.get_mut(&TraceKey::new(hart, pc)) else {
        return false;
    };
    let Some(existing) = entry
        .iter_mut()
        .find(|write| write.addr == addr && write.bytes.as_slice() == bytes)
    else {
        return false;
    };
    merge_instruction_timing(&mut existing.timing, timing);
    true
}

fn upsert_register_write(
    trace: &mut XiangShanTrace,
    hart: u32,
    pc: u64,
    register: String,
    value: u64,
    timing: InstructionTiming,
) -> Result<(), XiangShanError> {
    upsert_register_write_map(&mut trace.registers, hart, pc, register, value, timing)
}

fn upsert_register_write_map(
    register_writes: &mut HashMap<TraceKey, Vec<RegisterWrite>>,
    hart: u32,
    pc: u64,
    register: String,
    value: u64,
    timing: InstructionTiming,
) -> Result<(), XiangShanError> {
    let entry = register_writes.entry(TraceKey::new(hart, pc)).or_default();
    let parsed_register = parse_register_name(&register);
    if let Some(existing) = entry.iter_mut().find(|write| {
        write.register == register
            || match (parse_register_name(&write.register), parsed_register) {
                (Some(left), Some(right)) => left == right,
                _ => false,
            }
    }) {
        if existing.register != register {
            existing.register = register.clone();
        }
        existing.value = value;
        merge_instruction_timing(&mut existing.timing, &timing);
        return Ok(());
    }
    entry.push(RegisterWrite {
        pc,
        register,
        value,
        timing,
    });
    Ok(())
}

fn upsert_memory_write(
    trace: &mut XiangShanTrace,
    hart: u32,
    pc: u64,
    addr: u64,
    bytes: Vec<(u64, u8)>,
    timing: InstructionTiming,
) -> Result<(), XiangShanError> {
    if bytes.is_empty() {
        return Ok(());
    }

    let entry = trace.memory.entry(TraceKey::new(hart, pc)).or_default();
    if let Some(existing) = entry.iter().find(|write| write.addr == addr) {
        let detail = format!(
            "memory write at addr 0x{addr:016x} appears more than once ({:?}, {:?})",
            existing.bytes, bytes
        );
        return Err(if existing.bytes == bytes {
            XiangShanError::DuplicateTraceRecord {
                kind: "memory",
                hart,
                pc,
                detail,
            }
        } else {
            XiangShanError::ConflictingTraceRecord {
                kind: "memory",
                hart,
                pc,
                detail,
            }
        });
    }

    entry.push(MemoryWrite {
        pc,
        addr,
        bytes,
        timing,
    });
    Ok(())
}

fn quote_os_str(arg: &OsStr) -> String {
    let value = arg.to_string_lossy();
    if value.is_empty() {
        "\"\"".to_string()
    } else if value.chars().all(|c| !c.is_whitespace() && c != '"') {
        value.to_string()
    } else {
        format!("\"{}\"", value.replace('"', "\\\""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn key(hart: u32, pc: u64) -> TraceKey {
        TraceKey::new(hart, pc)
    }

    #[test]
    fn parse_commit_without_side_effect_marks_pc_observed() {
        let log = "[1957] commit pc 0000000080001698 inst 07c0000f wen 0 dst x00 data 0000000000000000 idx 01c fence orw,io\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        assert!(trace.committed_contains_pc(0, 0x0000_0000_8000_1698u64));
        assert!(
            trace
                .instruction_timing_entries(0, 0x0000_0000_8000_1698u64)
                .is_none(),
            "commit observation should not synthesize timing"
        );
    }

    #[test]
    fn parse_commit_with_inline_mem_only() {
        let log = "[571] commit pc 0000000080000792 inst 01151423 wen 0 dst x08 data 0000000000000000 idx 02b (0c) addr 000000008fffffe0 data 0000000045000000 mask 0x08 sh      a7, 8(a0)\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x0000_0000_8000_0792u64;
        let mem = trace.memory_writes(0, pc).expect("memory entry missing");
        assert_eq!(mem.len(), 1);
        assert_eq!(mem[0].addr, 0x0000_0000_8fff_ffe0u64);
        assert_eq!(mem[0].bytes.len(), 1);
        assert_eq!(mem[0].bytes[0].0, 0x0000_0000_8fff_ffe3u64);
        assert_eq!(mem[0].bytes[0].1, 0x45u8);
    }

    #[test]
    fn parse_commit_inline_mem_duplicate_mem_line_is_deduplicated() {
        let log = "[571] commit pc 0000000080000792 inst 01151423 wen 0 dst x08 data 0000000000000000 idx 02b (0c) addr 000000008fffffe0 data 0000000045000000 mask 0x08 sh      a7, 8(a0)\nmem pc 0000000080000792 addr 000000008fffffe0 data 0000000045000000 mask 0x08\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("identical inline and standalone memory payloads should be deduplicated");

        let mem = trace
            .memory_writes(0, 0x0000_0000_8000_0792u64)
            .expect("memory entry missing");
        assert_eq!(mem.len(), 1);
        assert_eq!(mem[0].addr, 0x0000_0000_8fff_ffe0u64);
        assert_eq!(mem[0].bytes, vec![(0x0000_0000_8fff_ffe3u64, 0x45u8)]);
    }

    #[test]
    fn parse_dual_hart_commit_without_dst_prefix() {
        let log = "[0][11] commit pc 0000000080000028 inst 80000693 wen 1 dst 13 data fffffffffffff800 idx 00e li      a3, -2048\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x0000_0000_8000_0028u64;
        let regs = trace
            .register_writes(0, pc)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "a3");
        assert_eq!(regs[0].value, 0xffff_ffff_ffff_f800u64);
    }

    #[test]
    fn parse_commit_without_dst_prefix_keeps_float_alias_from_asm_operand() {
        let log = "[0][72] commit pc 00000000800000e6 inst f20406d3 wen 1 dst 13 data 4004000000000000 idx 04b fmv.d.x fa3, s0\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("missing float dstprefix should keep asm operand register alias");
        let regs = trace
            .register_writes(0, 0x0000_0000_8000_00e6)
            .expect("register entry missing");
        assert_eq!(regs[0].register, "fa3");
    }

    #[test]
    fn parse_commit_without_dst_prefix_keeps_log_register_name_for_float_dest() {
        let log = "[219] commit pc 00000000800002be inst f4180a53 wen 1 dst 20 data ffffffffffff3c00 idx 0de fli.h   s4, 1.0\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("missing float dstprefix should keep the log register name");
        let regs = trace
            .register_writes(0, 0x0000_0000_8000_02be)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "s4");
    }

    #[test]
    fn parse_commit_without_dst_prefix_keeps_log_register_name_for_fli_d_s3() {
        let log = "[179] commit pc 0000000080000234 inst f21e89d3 wen 1 dst 19 data 40f0000000000000 idx 0b6 fli.d   s3, 65536.0\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("missing float dstprefix should keep fli.d log register name");
        let regs = trace
            .register_writes(0, 0x0000_0000_8000_0234)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "s3");
        assert_eq!(regs[0].value, 0x40f0_0000_0000_0000);
    }

    #[test]
    fn parse_commit_without_dst_prefix_keeps_log_register_name_for_fli_d_a6() {
        let log = "[222] commit pc 00000000800002ce inst f2110853 wen 1 dst 16 data 3ef0000000000000 idx 0e1 fli.d   a6, 1.52587890625e-05\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("missing float dstprefix should keep fli.d log register name");
        let regs = trace
            .register_writes(0, 0x0000_0000_8000_02ce)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "a6");
        assert_eq!(regs[0].value, 0x3ef0_0000_0000_0000);
    }

    #[test]
    fn parse_commit_without_dst_prefix_keeps_log_register_name_for_fli_s_a4() {
        let log = "[233] commit pc 00000000800002d8 inst f0108753 wen 1 dst 14 data ffffffff00800000 idx 0ec fli.s   a4, min\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("missing float dstprefix should keep fli.s log register name");
        let regs = trace
            .register_writes(0, 0x0000_0000_8000_02d8)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "a4");
        assert_eq!(regs[0].value, 0xffff_ffff_0080_0000);
    }

    #[test]
    fn parse_commit_without_dst_prefix_accepts_vector_dest_from_asm_operand() {
        let log = "[5] commit pc 0000000080000100 inst 02058557 wen 1 dst 10 data 0000000000000001 idx 015 vredsum.vs v10, v11, v12\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("missing vector dstprefix should keep vector operand name");
        let regs = trace
            .register_writes(0, 0x0000_0000_8000_0100)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "v10");
        assert_eq!(regs[0].value, 1);
    }

    #[test]
    fn parse_commit_dst32_vector_sentinel_uses_vector_asm_dest() {
        let log = "[99] commit pc 0000000080000160 inst 5e0446d7 wen 1 dst 32 data 0005555555555555 idx 063 vmv.v.x v13, x8\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("vector dst sentinel should parse from asm destination");
        let regs = trace
            .register_writes(0, 0x0000_0000_8000_0160)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "v13");
        assert_eq!(regs[0].value, 0x0005_5555_5555_5555);
    }

    #[test]
    fn parse_commit_dst32_without_vector_dest_drops_non_architectural_write() {
        let log = "[99] commit pc 0000000080000160 inst 5e0446d7 wen 1 dst 32 data 0005555555555555 idx 063 addi x1, x0, 1\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("unknown dst sentinel should not synthesize x32");
        assert!(!trace.registers_contain_pc(0, 0x0000_0000_8000_0160));
        assert!(trace.committed_contains_pc(0, 0x0000_0000_8000_0160));
    }

    #[test]
    fn parse_retired_timing_merges_into_explicit_float_commit() {
        let log = concat!(
            "[219] commit pc 00000000800002be inst f4180a53 wen 1 dst f20 data ffffffffffff3c00 idx 0de fli.h   s4, 1.0\n",
            "retired hart  0 pc 00000800002be wen 0 ldest 20 pdest 37 data 0000000000000000 fflags: 00000 vxsat: 0 clk_start                 8483 clk_end                 8494 clk_span                   12\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x0000_0000_8000_02beu64;
        let regs = trace
            .register_writes(0, pc)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "f20");
        assert_eq!(regs[0].value, 0xffff_ffff_ffff_3c00u64);
        assert_eq!(regs[0].timing.clk_start, Some(8483));
        assert_eq!(regs[0].timing.clk_end, Some(8494));
        assert_eq!(regs[0].timing.clk_span, Some(12));
    }

    #[test]
    fn parse_commit_without_dst_prefix_keeps_integer_dest_for_float_compare() {
        let log = "[1][130] commit pc 000000008000062e inst a4f827d3 wen 1 dst 15 data 0000000000000000 idx 085 feq.h   a5, fa6, fa5\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x0000_0000_8000_062eu64;
        let regs = trace
            .register_writes(1, pc)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "a5");
        assert_eq!(regs[0].value, 0);
    }

    #[test]
    fn parse_retired_wen0_atomic_commit_keeps_timing_without_reconstructing_write() {
        let log = concat!(
            "[109] commit pc 0000000080000156 inst 291426af wen 0 dst 13 data 0000000000000001 idx 070 amocas.w a3, a7, (s0)\n",
            "retired hart  0 pc 0000080000156 wen 0 ldest 13 pdest 00 data 000000006a3ee1cf fflags: 00000 vxsat: 0 clk_start 0 clk_end 8475 clk_span 8476\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        assert!(!trace.registers_contain_pc(0, 0x8000_0156));
        let timings = trace
            .instruction_timing_entries(0, 0x8000_0156)
            .expect("instruction timing missing");
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].timing.clk_end, Some(8475));
    }

    #[test]
    fn parse_retired_wen0_plain_store_does_not_create_register_write() {
        let log = concat!(
            "[60] commit pc 00000000800000bc inst 00a4bc23 wen 0 dst 24 data 5bf177506994aa7f idx 03f (03) sd      a0, 24(s1)\n",
            "retired hart  0 pc 00000800000bc wen 0 ldest 24 pdest 00 data 0000000000000000 fflags: 00000 vxsat: 0 clk_start 8341 clk_end 8356 clk_span 16\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        assert!(!trace.registers_contain_pc(0, 0x8000_00bc));
    }

    #[test]
    fn associate_memory_writes_does_not_reconstruct_missing_scalar_store() {
        let tracer = Tracer::from_traced_instructions(vec![
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_1498,
                disassembly: "addi x2, x0, 0".to_string(),
                parsed_disassembly: None,
                source_line: None,
                trace_inst_local_index: None,
                source_instruction: None,
            },
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_14a6,
                disassembly: "addi x13, x0, 0".to_string(),
                parsed_disassembly: None,
                source_line: None,
                trace_inst_local_index: None,
                source_instruction: None,
            },
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_14ba,
                disassembly: "c.swsp x13, 0(sp)".to_string(),
                parsed_disassembly: None,
                source_line: Some("c.swsp x13, 0(sp)".to_string()),
                trace_inst_local_index: Some(0),
                source_instruction: None,
            },
        ]);

        let mut trace = XiangShanTrace::default();
        trace.registers.insert(
            key(0, 0x8000_1498),
            vec![RegisterWrite {
                pc: 0x8000_1498,
                register: "x2".to_string(),
                value: 0x8001_1000,
                timing: InstructionTiming::default(),
            }],
        );
        trace.registers.insert(
            key(0, 0x8000_14a6),
            vec![RegisterWrite {
                pc: 0x8000_14a6,
                register: "x13".to_string(),
                value: 0,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_1498),
            vec![InstructionTrace {
                pc: 0x8000_1498,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_14a6),
            vec![InstructionTrace {
                pc: 0x8000_14a6,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_14ba),
            vec![InstructionTrace {
                pc: 0x8000_14ba,
                timing: InstructionTiming {
                    clk_start: Some(11267),
                    clk_end: Some(11274),
                    clk_span: Some(8),
                },
            }],
        );

        let memory = associate_memory_writes(&trace, 0, &tracer);
        assert!(
            !memory.contains_key(&0x8000_14ba),
            "missing store payload must not be reconstructed from disassembly/register state"
        );
    }

    #[test]
    fn associate_memory_writes_does_not_reconstruct_missing_float_store() {
        let tracer = Tracer::from_traced_instructions(vec![
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_1456,
                disassembly: "fli.h f18, 65536.0".to_string(),
                parsed_disassembly: None,
                source_line: Some("fli.h f18, 65536.0".to_string()),
                trace_inst_local_index: Some(0),
                source_instruction: None,
            },
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_145c,
                disassembly: "c.addi x12, -6".to_string(),
                parsed_disassembly: None,
                source_line: Some("li x12, 2147553274".to_string()),
                trace_inst_local_index: Some(1),
                source_instruction: None,
            },
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_145e,
                disassembly: "fsd f18, 6(x12)".to_string(),
                parsed_disassembly: None,
                source_line: Some("fsd f18, 6(x12)".to_string()),
                trace_inst_local_index: Some(2),
                source_instruction: None,
            },
        ]);

        let mut trace = XiangShanTrace::default();
        trace.registers.insert(
            key(0, 0x8000_1456),
            vec![RegisterWrite {
                pc: 0x8000_1456,
                register: "f18".to_string(),
                value: 0xffff_ffff_ffff_7c00,
                timing: InstructionTiming::default(),
            }],
        );
        trace.registers.insert(
            key(0, 0x8000_145c),
            vec![RegisterWrite {
                pc: 0x8000_145c,
                register: "x12".to_string(),
                value: 0x8001_0ffa,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_1456),
            vec![InstructionTrace {
                pc: 0x8000_1456,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_145c),
            vec![InstructionTrace {
                pc: 0x8000_145c,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_145e),
            vec![InstructionTrace {
                pc: 0x8000_145e,
                timing: InstructionTiming {
                    clk_start: Some(11004),
                    clk_end: Some(11026),
                    clk_span: Some(23),
                },
            }],
        );

        let memory = associate_memory_writes(&trace, 0, &tracer);
        assert!(
            !memory.contains_key(&0x8000_145e),
            "missing float store payload must not be reconstructed from disassembly/register state"
        );
    }

    #[test]
    fn associate_memory_writes_skips_scalar_store_reconstruction_on_exception() {
        let tracer = Tracer::from_traced_instructions(vec![
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_1498,
                disassembly: "addi x2, x0, 0".to_string(),
                parsed_disassembly: None,
                source_line: None,
                trace_inst_local_index: None,
                source_instruction: None,
            },
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_14a6,
                disassembly: "addi x13, x0, 0".to_string(),
                parsed_disassembly: None,
                source_line: None,
                trace_inst_local_index: None,
                source_instruction: None,
            },
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_14ba,
                disassembly: "c.swsp x13, 0(sp)".to_string(),
                parsed_disassembly: None,
                source_line: Some("c.swsp x13, 0(sp)".to_string()),
                trace_inst_local_index: Some(0),
                source_instruction: None,
            },
        ]);

        let mut trace = XiangShanTrace::default();
        trace.registers.insert(
            key(0, 0x8000_1498),
            vec![RegisterWrite {
                pc: 0x8000_1498,
                register: "x2".to_string(),
                value: 0x8001_1000,
                timing: InstructionTiming::default(),
            }],
        );
        trace.registers.insert(
            key(0, 0x8000_14a6),
            vec![RegisterWrite {
                pc: 0x8000_14a6,
                register: "x13".to_string(),
                value: 0,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_1498),
            vec![InstructionTrace {
                pc: 0x8000_1498,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_14a6),
            vec![InstructionTrace {
                pc: 0x8000_14a6,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_14ba),
            vec![InstructionTrace {
                pc: 0x8000_14ba,
                timing: InstructionTiming {
                    clk_start: Some(11267),
                    clk_end: Some(11274),
                    clk_span: Some(8),
                },
            }],
        );
        trace.exceptions.insert(
            key(0, 0x8000_14ba),
            ExceptionEvent {
                pc: 0x8000_14ba,
                cause: 7,
                timing: InstructionTiming {
                    clk_start: Some(11267),
                    clk_end: Some(11274),
                    clk_span: Some(8),
                },
            },
        );

        let memory = associate_memory_writes(&trace, 0, &tracer);
        assert!(
            !memory.contains_key(&0x8000_14ba),
            "store reconstruction must not run for excepting instructions"
        );
    }

    #[test]
    fn parse_retired_timing_without_commit_only_records_instruction_timing() {
        let log = "retired hart 1 pc 80000080 wen 1 ldest 19 pdest 1f data fffffffffffff7ff fflags: 0 vxsat: 0 clk_start 94 clk_end 96 clk_span 3\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x8000_0080u64;
        assert!(!trace.registers_contain_pc(1, pc));

        let timings = trace
            .instruction_timing_entries(1, pc)
            .expect("timing entry missing");
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].timing.clk_span, Some(3));
    }

    #[test]
    fn parse_retired_timing_merges_with_existing_commit_write_without_duplicates() {
        let log = "[1][11] commit pc 0000000080000080 inst 79fd0000 wen 1 dst 19 data fffffffffffff7ff idx 00e c.lui s3,0xfffff\nretired hart 1 pc 80000080 wen 1 ldest 19 pdest 1f data fffffffffffff7ff fflags: 0 vxsat: 0 clk_start 94 clk_end 96 clk_span 3\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x8000_0080u64;
        let regs = trace
            .register_writes(1, pc)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "s3");
        assert_eq!(regs[0].timing.clk_start, Some(94));
        assert_eq!(regs[0].timing.clk_end, Some(96));
        assert_eq!(regs[0].timing.clk_span, Some(3));
    }

    #[test]
    fn parse_retired_before_commit_keeps_compact_commit_value() {
        let log = concat!(
            "retired hart  0 pc 0000080000bec wen 1 ldest 20 pdest 2b data 0000000000080000 fflags: 00000 vxsat: 0 clk_start 9980 clk_end 9994 clk_span 15\n",
            "[927] commit pc 0000000080000bec inst 00080a13 wen 1 dst 20 data 000000005bd58e5b idx 0e2 mv      s4, a6\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let regs = trace
            .register_writes(0, 0x8000_0bec)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "s4");
        assert_eq!(regs[0].value, 0x5bd5_8e5b);
        assert_eq!(regs[0].timing.clk_start, Some(9980));
        assert_eq!(regs[0].timing.clk_end, Some(9994));
        assert_eq!(regs[0].timing.clk_span, Some(15));
    }

    #[test]
    fn parse_retired_before_atomic_commit_does_not_reconstruct_write() {
        let log = concat!(
            "retired hart  0 pc 0000080000156 wen 0 ldest 13 pdest 00 data 000000006a3ee1cf fflags: 00000 vxsat: 0 clk_start 0 clk_end 8475 clk_span 8476\n",
            "[109] commit pc 0000000080000156 inst 291426af wen 0 dst 13 data 0000000000000001 idx 070 amocas.w a3, a7, (s0)\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        assert!(!trace.registers_contain_pc(0, 0x8000_0156));
        let timings = trace
            .instruction_timing_entries(0, 0x8000_0156)
            .expect("instruction timing missing");
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].timing.clk_end, Some(8475));
    }

    #[test]
    fn parse_store_commit_timing_without_mem_payload_records_memory_write() {
        let log = "store commit hart 1 pc 0x800000a0 robidx 9 addr 90000000 data_lo 1122334455667788 data_hi 0 mask 0f wline 0 vecsplit 0 clk_start 101 clk_end 104 clk_span 4\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x8000_00a0u64;
        let mem = trace.memory_writes(1, pc).expect("memory entry missing");
        assert_eq!(mem.len(), 1);
        assert_eq!(mem[0].addr, 0x9000_0000u64);
        assert_eq!(
            mem[0].bytes,
            vec![
                (0x9000_0000, 0x88),
                (0x9000_0001, 0x77),
                (0x9000_0002, 0x66),
                (0x9000_0003, 0x55),
            ]
        );
        assert_eq!(mem[0].timing.clk_start, Some(101));
        assert_eq!(mem[0].timing.clk_end, Some(104));
        assert_eq!(mem[0].timing.clk_span, Some(4));
        let timings = trace
            .instruction_timing_entries(1, pc)
            .expect("instruction timing missing");
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].timing.clk_start, Some(101));
        assert_eq!(timings[0].timing.clk_end, Some(104));
        assert_eq!(timings[0].timing.clk_span, Some(4));
    }

    #[test]
    fn parse_real_spacing_and_wide_store_commit_timing_records_authoritative_payload() {
        let log = concat!(
            "retired hart  0 pc 000008000003e wen 1 ldest 13 pdest 0f data d5b2d89674fb1fe1 fflags: 00000 vxsat: 0 clk_start                  556 clk_end                  560 clk_span                    5\n",
            "store commit hart  0 pc 0x0000080000228 robidx 21 addr 0000000080001f78 data_lo 0000000000000000bce55adfd3b3356d data_hi 0000000000000000 mask 00ff wline 0 vecsplit 0 clk_start                  828 clk_end                  835 clk_span                    8\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        assert!(!trace.registers_contain_pc(0, 0x8000_003e));
        let mem = trace
            .memory_writes(0, 0x8000_0228)
            .expect("memory entry missing");
        assert_eq!(mem.len(), 1);
        assert_eq!(mem[0].addr, 0x8000_1f78u64);
        assert_eq!(mem[0].timing.clk_end, Some(835));
        assert_eq!(mem[0].timing.clk_span, Some(8));
        let retire_timing = trace
            .instruction_timing_entries(0, 0x8000_003e)
            .expect("missing retired timing");
        assert_eq!(retire_timing[0].timing.clk_start, Some(556));
        assert_eq!(retire_timing[0].timing.clk_span, Some(5));
        let store_timing = trace
            .instruction_timing_entries(0, 0x8000_0228)
            .expect("missing store timing");
        assert_eq!(store_timing[0].timing.clk_end, Some(835));
        assert_eq!(store_timing[0].timing.clk_span, Some(8));
    }

    #[test]
    fn store_commit_timing_merges_into_explicit_mem_payload() {
        let log = concat!(
            "[40] commit pc 0000000080000082 inst 000404b7 wen 1 dst 09 data 0000000000040000 idx 02b lui     s1, 0x40\n",
            "[41] commit pc 0000000080000086 inst 0014849b wen 1 dst 09 data 0000000000040001 idx 02c addiw   s1, s1, 1\n",
            "[42] commit pc 0000000080000088 inst 00d49493 wen 1 dst 09 data 0000000080002000 idx 02d slli    s1, s1, 13\n",
            "[43] commit pc 000000008000008a inst fd848493 wen 1 dst 09 data 0000000080001fd8 idx 02e addi    s1, s1, -40\n",
            "retired hart  0 pc 00000800000c0 wen 1 ldest  9 pdest 00 data a94f3c537afe7c34 fflags: 00000 vxsat: 0 clk_start 8337 clk_end 8354 clk_span 18\n",
            "[60] commit pc 00000000800000bc inst 00a4bc23 wen 0 dst 24 data 5bf177506994aa7f idx 03f (03) sd      a0, 24(s1)\n",
            "store commit hart  0 pc 0x00000800000bc robidx 63 addr 0000000080001ff0 data_lo 00000000000000005bf177506994aa7f data_hi 0000000000000000 mask 00ff wline 0 vecsplit 0 clk_start 8341 clk_end 8356 clk_span 16\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x8000_00bcu64;
        let mem = trace.memory_writes(0, pc).expect("memory entry missing");
        assert_eq!(mem.len(), 1);
        assert_eq!(mem[0].addr, 0x8000_1ff0u64);
        assert_eq!(mem[0].timing.clk_start, Some(8341));
        assert_eq!(mem[0].timing.clk_end, Some(8356));
        assert_eq!(mem[0].timing.clk_span, Some(16));
    }

    #[test]
    fn commit_store_without_explicit_store_payload_is_not_reconstructed() {
        let log = concat!(
            "[39] commit pc 0000000080000080 inst 00000013 wen 1 dst 10 data 5bf177506994aa7f idx 02a addi    a0, zero, 0\n",
            "[40] commit pc 0000000080000082 inst 000404b7 wen 1 dst 09 data 0000000000040000 idx 02b lui     s1, 0x40\n",
            "[41] commit pc 0000000080000086 inst 0014849b wen 1 dst 09 data 0000000000040001 idx 02c addiw   s1, s1, 1\n",
            "[42] commit pc 0000000080000088 inst 00d49493 wen 1 dst 09 data 0000000080002000 idx 02d slli    s1, s1, 13\n",
            "[43] commit pc 000000008000008a inst fd848493 wen 1 dst 09 data 0000000080001fd8 idx 02e addi    s1, s1, -40\n",
            "retired hart  0 pc 00000800000c0 wen 1 ldest  9 pdest 00 data a94f3c537afe7c34 fflags: 00000 vxsat: 0 clk_start 8337 clk_end 8354 clk_span 18\n",
            "[60] commit pc 00000000800000bc inst 00a4bc23 wen 0 dst 24 data 5bf177506994aa7f idx 03f (03) sd      a0, 24(s1)\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x8000_00bcu64;
        assert!(
            !trace.memory_contains_pc(0, pc),
            "plain commit-only stores must not be reconstructed"
        );
    }

    #[test]
    fn debug_dump_low_memory_entries_from_env_log() {
        let Ok(path) = std::env::var("XS_DEBUG_LOG_PATH") else {
            return;
        };
        let mem_start = std::env::var("XS_DEBUG_MEM_START")
            .ok()
            .and_then(|raw| u64::from_str_radix(raw.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);
        let trace = parse_xs_commit_trace(Path::new(&path)).expect("parse failed");
        for (trace_key, writes) in trace.memory.iter() {
            for write in writes {
                if write.addr < mem_start {
                    eprintln!(
                        "LOW hart={} pc=0x{:016x} addr=0x{:016x} bytes={:?}",
                        trace_key.hart, trace_key.pc, write.addr, write.bytes
                    );
                }
            }
        }
    }

    #[test]
    fn parse_retired_zero_pdest_write_only_records_timing() {
        let log = "retired hart  0 pc 000008000007e wen 1 ldest 18 pdest 00 data e70dc6492fdf346d fflags: 00000 vxsat: 0 clk_start 1190 clk_end 1202 clk_span 13\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        assert!(!trace.registers_contain_pc(0, 0x8000_007e));
        let timings = trace
            .instruction_timing_entries(0, 0x8000_007e)
            .expect("missing instruction timing");
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].timing.clk_span, Some(13));
    }

    #[test]
    fn parse_trap_timing_merges_into_existing_exception() {
        let log = "[1][11] exception pc 000000008000008a inst 00009002 cause 0000000000000002\ntrap hart 1 pc 8000008a intr 0 excp 1 exceptionVec 4 clk_start 47 clk_end 49 clk_span 3\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let pc = 0x8000_008au64;
        let exc = trace.exception(1, pc).expect("exception entry missing");
        assert_eq!(exc.cause, 2);
        assert_eq!(exc.timing.clk_start, Some(47));
        assert_eq!(exc.timing.clk_end, Some(49));
        assert_eq!(exc.timing.clk_span, Some(3));
    }

    #[test]
    fn parse_trap_timing_without_exception_line_only_records_instruction_timing() {
        let log = "trap hart 0 pc 000008000034e intr 0 excp 1 exceptionVec 000800 clk_start 8846 clk_end 8856 clk_span 11\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        assert!(!trace.exceptions_contain_pc(0, 0x8000_034e));
        let timings = trace
            .instruction_timing_entries(0, 0x8000_034e)
            .expect("instruction timing missing");
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].timing.clk_start, Some(8846));
        assert_eq!(timings[0].timing.clk_end, Some(8856));
        assert_eq!(timings[0].timing.clk_span, Some(11));
    }

    #[test]
    fn parse_multi_hart_same_pc_commit_and_retired_timing_stay_isolated() {
        let log = concat!(
            "[0][11] commit pc 0000000080000080 inst 00000513 wen 1 dst 10 data 0000000000000001 idx 00e addi a0, zero, 1\n",
            "retired hart 0 pc 80000080 wen 1 ldest 10 pdest 1f data 0000000000000001 fflags: 0 vxsat: 0 clk_start 11 clk_end 13 clk_span 3\n",
            "[1][12] commit pc 0000000080000080 inst 00000513 wen 1 dst 10 data 0000000000000002 idx 00f addi a0, zero, 2\n",
            "retired hart 1 pc 80000080 wen 1 ldest 10 pdest 1f data 0000000000000002 fflags: 0 vxsat: 0 clk_start 21 clk_end 24 clk_span 4\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let hart0 = trace
            .register_writes(0, 0x8000_0080)
            .expect("hart 0 write missing");
        let hart1 = trace
            .register_writes(1, 0x8000_0080)
            .expect("hart 1 write missing");
        assert_eq!(hart0[0].value, 1);
        assert_eq!(hart0[0].timing.clk_start, Some(11));
        assert_eq!(hart1[0].value, 2);
        assert_eq!(hart1[0].timing.clk_start, Some(21));

        let timing0 = trace
            .instruction_timing_entries(0, 0x8000_0080)
            .expect("hart 0 timing missing");
        let timing1 = trace
            .instruction_timing_entries(1, 0x8000_0080)
            .expect("hart 1 timing missing");
        assert_eq!(timing0[0].timing.clk_end, Some(13));
        assert_eq!(timing1[0].timing.clk_end, Some(24));
    }

    #[test]
    fn parse_multi_hart_same_pc_exception_and_trap_timing_stay_isolated() {
        let log = concat!(
            "[0][11] exception pc 000000008000008a inst 00009002 cause 0000000000000002\n",
            "trap hart 0 pc 8000008a intr 0 excp 1 exceptionVec 2 clk_start 47 clk_end 49 clk_span 3\n",
            "[1][12] exception pc 000000008000008a inst 00009002 cause 0000000000000004\n",
            "trap hart 1 pc 8000008a intr 0 excp 1 exceptionVec 4 clk_start 57 clk_end 60 clk_span 4\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let exc0 = trace
            .exception(0, 0x8000_008a)
            .expect("hart 0 exception missing");
        let exc1 = trace
            .exception(1, 0x8000_008a)
            .expect("hart 1 exception missing");
        assert_eq!(exc0.cause, 2);
        assert_eq!(exc0.timing.clk_start, Some(47));
        assert_eq!(exc1.cause, 4);
        assert_eq!(exc1.timing.clk_start, Some(57));

        let timing0 = trace
            .instruction_timing_entries(0, 0x8000_008a)
            .expect("hart 0 timing missing");
        let timing1 = trace
            .instruction_timing_entries(1, 0x8000_008a)
            .expect("hart 1 timing missing");
        assert_eq!(timing0[0].timing.clk_span, Some(3));
        assert_eq!(timing1[0].timing.clk_span, Some(4));
    }

    #[test]
    fn parse_multi_hart_same_pc_mem_and_store_timing_stay_isolated() {
        let log = concat!(
            "[0][41] commit pc 00000000800000bc inst 00a4bc23 wen 0 dst 24 data 1111111111111111 idx 03f sd a0, 24(s1)\n",
            "store commit hart 0 pc 0x00000800000bc robidx 63 addr 0000000080001ff0 data_lo 00000000000000001111111111111111 data_hi 0000000000000000 mask 00ff wline 0 vecsplit 0 clk_start 8341 clk_end 8356 clk_span 16\n",
            "[1][42] commit pc 00000000800000bc inst 00a4bc23 wen 0 dst 24 data 2222222222222222 idx 040 sd a0, 24(s1)\n",
            "store commit hart 1 pc 0x00000800000bc robidx 64 addr 0000000080002ff0 data_lo 00000000000000002222222222222222 data_hi 0000000000000000 mask 00ff wline 0 vecsplit 0 clk_start 9341 clk_end 9356 clk_span 16\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        let timing0 = trace
            .instruction_timing_entries(0, 0x8000_00bc)
            .expect("hart 0 timing missing");
        let timing1 = trace
            .instruction_timing_entries(1, 0x8000_00bc)
            .expect("hart 1 timing missing");
        assert_eq!(timing0[0].timing.clk_start, Some(8341));
        assert_eq!(timing1[0].timing.clk_start, Some(9341));
        let mem0 = trace
            .memory_writes(0, 0x8000_00bc)
            .expect("hart 0 memory missing");
        let mem1 = trace
            .memory_writes(1, 0x8000_00bc)
            .expect("hart 1 memory missing");
        assert_eq!(mem0[0].addr, 0x8000_1ff0u64);
        assert_eq!(mem1[0].addr, 0x8000_2ff0u64);
        assert_eq!(mem0[0].timing.clk_span, Some(16));
        assert_eq!(mem1[0].timing.clk_span, Some(16));
    }

    #[test]
    fn integer_destination_row_is_not_stream_ready_before_register_write_arrives() {
        let tracer = Tracer::from_traced_instructions(vec![
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_0036,
                disassembly: "addi x10, x0, 1".to_string(),
                parsed_disassembly: None,
                source_line: Some("addi x10, x0, 1".to_string()),
                trace_inst_local_index: Some(0),
                source_instruction: None,
            },
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_0040,
                disassembly: "addi x11, x0, 2".to_string(),
                parsed_disassembly: None,
                source_line: Some("addi x11, x0, 2".to_string()),
                trace_inst_local_index: Some(1),
                source_instruction: None,
            },
        ]);

        let row_without_write = AsmInstructionExecution {
            pc: 0x8000_0036,
            instruction: riscv_instruction::RiscvInstruction::decode_asm_with_context(
                "addi x10, x0, 1",
                riscv_instruction::IsaDecodeContext::RV64,
            )
            .expect("decode asm"),
            source_annotation: None,
            trace_inst_id: None,
            exceptions: Vec::new(),
            register_writes: Vec::new(),
            memory_writes: Vec::new(),
            timing: InstructionTiming {
                clk_start: Some(10),
                clk_end: Some(12),
                clk_span: Some(3),
            },
        };
        assert!(
            !instruction_ready_for_streaming(&row_without_write, &tracer, Some(20)),
            "integer destination rows must wait for their architectural writeback"
        );

        let mut row_with_write = row_without_write.clone();
        row_with_write.register_writes.push(TraitRegisterValue {
            name: "a0".to_string(),
            value: 1,
        });
        assert!(
            instruction_ready_for_streaming(&row_with_write, &tracer, Some(20)),
            "row becomes ready once the integer destination write is present"
        );
    }

    #[test]
    fn streaming_filter_skips_framework_spin_rows_and_keeps_ready_user_prefix() {
        let tracer = Tracer::from_traced_instructions(vec![
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_0040,
                disassembly: "fence rw, rw".to_string(),
                parsed_disassembly: riscv_instruction::RiscvInstruction::decode_asm_with_context(
                    "fence rw, rw",
                    riscv_instruction::IsaDecodeContext::RV64,
                )
                .ok(),
                source_line: Some("wait_start".to_string()),
                trace_inst_local_index: None,
                source_instruction: None,
            },
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_0044,
                disassembly: "addi x10, x0, 1".to_string(),
                parsed_disassembly: None,
                source_line: Some("addi x10, x0, 1".to_string()),
                trace_inst_local_index: Some(0),
                source_instruction: Some(
                    riscv_instruction::RiscvInstruction::decode_asm_with_context(
                        "addi x10, x0, 1",
                        riscv_instruction::IsaDecodeContext::RV64,
                    )
                    .expect("decode user row"),
                ),
            },
        ]);
        let hart = HartExecutionOutput {
            hart_id: 1,
            instructions: vec![
                AsmInstructionExecution {
                    pc: 0x8000_0040,
                    instruction: riscv_instruction::RiscvInstruction::decode_asm_with_context(
                        "fence rw, rw",
                        riscv_instruction::IsaDecodeContext::RV64,
                    )
                    .expect("decode framework output"),
                    source_annotation: None,
                    trace_inst_id: None,
                    exceptions: Vec::new(),
                    register_writes: Vec::new(),
                    memory_writes: Vec::new(),
                    timing: InstructionTiming::new(Some(10), Some(12), Some(3)),
                },
                AsmInstructionExecution {
                    pc: 0x8000_0044,
                    instruction: riscv_instruction::RiscvInstruction::decode_asm_with_context(
                        "addi x10, x0, 1",
                        riscv_instruction::IsaDecodeContext::RV64,
                    )
                    .expect("decode user output"),
                    source_annotation: None,
                    trace_inst_id: Some(riscv_fuzz_plugin_api::TraceInstId::new(1, 0)),
                    exceptions: Vec::new(),
                    register_writes: vec![TraitRegisterValue {
                        name: "a0".to_string(),
                        value: 1,
                    }],
                    memory_writes: Vec::new(),
                    timing: InstructionTiming::new(Some(13), Some(15), Some(3)),
                },
            ],
        };

        let filtered = filter_hart_output_for_streaming(&hart, &tracer, Some(20));
        assert_eq!(filtered.instructions.len(), 1);
        assert_eq!(filtered.instructions[0].pc, 0x8000_0044);
        assert!(filtered.instructions[0].trace_inst_id.is_some());
    }

    #[test]
    fn expected_destination_registers_accepts_compact_dump_spacing() {
        assert_eq!(
            expected_destination_registers("addi x13,x13,1621"),
            HashSet::from([String::from("x13")])
        );
        assert_eq!(
            expected_destination_registers("c.slli x14,0xc"),
            HashSet::from([String::from("x14")])
        );
    }

    #[test]
    fn expected_destination_registers_uses_source_instruction_when_dump_parse_is_missing() {
        let source_instruction = riscv_instruction::RiscvInstruction::decode_asm_with_context(
            "li x16, 917456",
            riscv_instruction::IsaDecodeContext::RV64,
        )
        .expect("source instruction should decode");
        let traced = TracedInstruction {
            pc: 0x8000_0044,
            disassembly: "lui x16,0x7bf".to_string(),
            parsed_disassembly: None,
            source_line: Some("li x16, 917456".to_string()),
            trace_inst_local_index: Some(0),
            source_instruction: Some(source_instruction),
        };

        assert_eq!(
            expected_destination_registers_for_traced_instruction(&traced),
            HashSet::from([String::from("x16")])
        );
    }

    #[test]
    fn parse_duplicate_exception_for_same_hart_pc_is_deduplicated() {
        let log = concat!(
            "[1][11] exception pc 000000008000008a inst 00009002 cause 0000000000000002\n",
            "[1][12] exception pc 000000008000008a inst 00009002 cause 0000000000000002\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("duplicate exception record with same cause should be deduplicated");
        assert_eq!(trace.exceptions.len(), 1);
        let event = trace
            .exceptions
            .get(&TraceKey::new(1, 0x8000_008a))
            .expect("deduplicated exception must remain in trace");
        assert_eq!(event.cause, 0x2);
    }

    #[test]
    fn parse_conflicting_register_write_for_same_hart_pc_reg_keeps_last_value() {
        let log = concat!(
            "[0][11] commit pc 0000000080000080 inst 00000513 wen 1 dst 10 data 0000000000000001 idx 00e addi a0, zero, 1\n",
            "[0][12] commit pc 0000000080000080 inst 00000513 wen 1 dst 10 data 0000000000000002 idx 00f addi a0, zero, 2\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("conflicting register records should be deterministically merged");

        let regs = trace
            .register_writes(0, 0x8000_0080)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "a0");
        assert_eq!(regs[0].value, 2);
    }

    #[test]
    fn parse_duplicate_register_write_for_same_hart_pc_reg_keeps_last_value() {
        let log = concat!(
            "[0][11] commit pc 0000000080000080 inst 00000513 wen 1 dst 10 data 0000000000000001 idx 00e addi a0, zero, 1\n",
            "[0][12] commit pc 0000000080000080 inst 00000513 wen 1 dst 10 data 0000000000000001 idx 00f addi a0, zero, 1\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("duplicate register records should be deterministically merged");

        let regs = trace
            .register_writes(0, 0x8000_0080)
            .expect("register entry missing");
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].register, "a0");
        assert_eq!(regs[0].value, 1);
    }

    #[test]
    fn parse_conflicting_memory_write_for_same_hart_pc_addr_is_rejected() {
        let log = concat!(
            "[1][41] commit pc 00000000800000bc inst 00a4bc23 wen 0 dst 24 data 1111111111111111 idx 03f sd a0, 24(s1)\n",
            "mem pc 00000000800000bc addr 0000000080001ff0 data 1111111111111111 mask 0xff\n",
            "mem pc 00000000800000bc addr 0000000080001ff0 data 2222222222222222 mask 0xff\n",
        );
        let err = parse_xs_commit_trace_reader(log.as_bytes())
            .expect_err("conflicting memory record must be rejected");
        assert!(matches!(
            err,
            XiangShanError::ConflictingTraceRecord {
                kind: "memory",
                hart: 1,
                pc: 0x8000_00bc,
                ..
            }
        ));
    }

    #[test]
    fn parse_mem_line_without_prior_hart_context_is_rejected() {
        let log = "mem pc 00000000800000bc addr 0000000080001ff0 data 1111111111111111 mask 0xff\n";
        let err = parse_xs_commit_trace_reader(log.as_bytes())
            .expect_err("mem lines without hart context must be rejected");
        assert!(format!("{err}").contains("no explicit hart context anywhere"));
    }

    #[test]
    fn parse_mem_line_with_multi_hart_same_pc_is_rejected_as_ambiguous() {
        let log = concat!(
            "[0][41] commit pc 00000000800000bc inst 00a4bc23 wen 0 dst 24 data 1111111111111111 idx 03f sd a0, 24(s1)\n",
            "[1][42] commit pc 00000000800000bc inst 00a4bc23 wen 0 dst 24 data 2222222222222222 idx 040 sd a0, 24(s1)\n",
            "mem pc 00000000800000bc addr 0000000080001ff0 data 1111111111111111 mask 0xff\n",
        );
        let err = parse_xs_commit_trace_reader(log.as_bytes())
            .expect_err("multi-hart mem lines for one pc must be rejected as ambiguous");
        assert!(format!("{err}").contains("ambiguous across harts"));
    }

    #[test]
    fn resolve_diff_so_env_path_rejects_empty_value() {
        let err = path_from_env_value(XIANGSHAN_DIFF_ENV_1C, Some("".into())).unwrap_err();
        assert!(matches!(err, XiangShanError::InvalidEnvVar { .. }));
    }

    #[test]
    fn resolve_diff_so_env_path_accepts_set_value() {
        let resolved =
            path_from_env_value(XIANGSHAN_DIFF_ENV_1C, Some("/tmp/shared-ref.so".into()))
                .expect("set env values should resolve");
        assert_eq!(resolved, Some(PathBuf::from("/tmp/shared-ref.so")));
    }

    #[test]
    fn emulator_env_var_tracks_alignment_multicore_and_coverage_mode() {
        let mut config = ExecutorConfig::new(
            PathBuf::from("/tmp/xiangshan"),
            ISABase::Rv64,
            "xiangshan".to_string(),
        );

        assert_eq!(config.emulator_env_var(false), XIANGSHAN_EMU_ENV);

        config.coverage_mode = coverage::CoverageMode::Full;
        assert_eq!(config.emulator_env_var(false), XIANGSHAN_EMU_COV_ENV);

        config.coverage_mode = coverage::CoverageMode::Light;
        assert_eq!(
            config.emulator_env_var(true),
            XIANGSHAN_EMU_2HART_COV_LIGHT_ENV
        );

        config.allow_unaligned = false;
        config.coverage_mode = coverage::CoverageMode::Off;
        assert_eq!(
            config.emulator_env_var(false),
            XIANGSHAN_EMU_NO_UNALIGNED_ENV
        );
        assert_eq!(
            config.emulator_env_var(true),
            XIANGSHAN_EMU_NO_UNALIGNED_2HART_ENV
        );
    }

    #[test]
    fn effective_timeout_keeps_single_hart_default() {
        let config = ExecutorConfig::new(
            PathBuf::from("/tmp/xiangshan"),
            ISABase::Rv64,
            "xiangshan".to_string(),
        );

        assert_eq!(
            config.effective_timeout(false),
            Some(ExecutorConfig::DEFAULT_TIMEOUT)
        );
    }

    #[test]
    fn effective_timeout_raises_multicore_default_when_not_overridden() {
        let config = ExecutorConfig::new(
            PathBuf::from("/tmp/xiangshan"),
            ISABase::Rv64,
            "xiangshan".to_string(),
        );

        assert_eq!(
            config.effective_timeout(true),
            Some(ExecutorConfig::DEFAULT_MULTICORE_TIMEOUT)
        );
    }

    #[test]
    fn effective_timeout_respects_explicit_override_for_multicore() {
        let mut config = ExecutorConfig::new(
            PathBuf::from("/tmp/xiangshan"),
            ISABase::Rv64,
            "xiangshan".to_string(),
        );
        config.timeout = Some(Duration::from_secs(123));
        config.timeout_overridden = true;

        assert_eq!(
            config.effective_timeout(true),
            Some(ExecutorConfig::DEFAULT_MULTICORE_TIMEOUT)
        );
    }

    #[test]
    fn effective_timeout_preserves_larger_multicore_override() {
        let mut config = ExecutorConfig::new(
            PathBuf::from("/tmp/xiangshan"),
            ISABase::Rv64,
            "xiangshan".to_string(),
        );
        config.timeout = Some(Duration::from_secs(600));
        config.timeout_overridden = true;

        assert_eq!(
            config.effective_timeout(true),
            Some(Duration::from_secs(600))
        );
    }

    #[test]
    fn parse_commit_with_overflowing_store_immediate_is_ignored_without_reconstruction() {
        let log = "[571] commit pc 0000000080000792 inst 01151423 wen 0 dst x08 data 0000000000000000 idx 02b (0c) sd a2, 999999999999999999999999999999(a1)\n";
        let trace = parse_xs_commit_trace_reader(log.as_bytes())
            .expect("commit-only store should not fail without reconstruction");
        assert!(trace.registers.is_empty());
        assert!(trace.memory.is_empty());
        assert!(trace.exceptions.is_empty());
    }

    #[test]
    fn parse_usage_output_rejected() {
        let log = "emu compiled at Apr  7 2026, 11:37:14\nUsage: /tmp/emu [OPTION...]\n/tmp/emu: unrecognized option '--enable-log'\n";
        let err = parse_xs_commit_trace_reader(log.as_bytes())
            .expect_err("usage output should not be accepted as an empty trace");
        assert!(format!("{err}").contains("unrecognized option '--enable-log'"));
    }

    #[test]
    fn unsuccessful_exit_with_commit_records_is_rejected_as_truncated_trace() {
        let log = concat!(
            "[07] commit pc 000000008000001a inst 00301073 wen 0 dst 00 data 0000000000000000 idx 007 csrw fcsr, zero\n",
            "retired hart  0 pc 000008000010a wen 1 ldest  8 pdest 4d data ba"
        );
        let err = xiangshan_unsuccessful_status_error(
            Some(1),
            log,
            "xiangshan --image program.elf",
            Path::new("/tmp/skiptrap.commit.log"),
        );

        let message = err.to_string();
        assert!(message.contains("incomplete/truncated"));
        assert!(message.contains("/tmp/skiptrap.commit.log"));
    }

    #[test]
    fn xs_commit_log_detects_difftest_abort_markers() {
        let log = concat!(
            "[1][19] commit pc 00000000800083f4 inst ffff8f93 wen 1 dst 31 data 0000000000030d3f idx 016 addi t6, t6, -1\n",
            "a3 different at pc = 0x008000857a, right = 0x000000007b9d95f2, wrong = 0x0000000000000010\n",
            "Core 0: \u{1b}[31mABORT at pc = 0xffffe182a4b541f5\n"
        );
        assert!(xs_commit_log_has_difftest_abort(log));
    }

    #[cfg(unix)]
    #[test]
    fn unsuccessful_exit_with_difftest_abort_is_classified_as_partial() {
        use std::fs;
        use std::os::unix::process::ExitStatusExt;

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("xs_abort_{unique}.log"));
        let log = concat!(
            "[07] commit pc 000000008000001a inst 00301073 wen 0 dst 00 data 0000000000000000 idx 007 csrw fcsr, zero\n",
            "a3 different at pc = 0x008000857a, right = 0x000000007b9d95f2, wrong = 0x0000000000000010\n",
            "Core 0: \u{1b}[31mABORT at pc = 0xffffe182a4b541f5\n"
        );
        fs::write(&path, log).expect("write temp xiangshan log");
        let status = std::process::ExitStatus::from_raw(1 << 8);

        let disposition =
            classify_xiangshan_exit_status(status, "xiangshan --image program.elf", &path)
                .expect("difftest abort should be classified as partial");
        assert_eq!(disposition, XiangShanExitDisposition::PartialDifftestAbort);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn annotate_partial_trace_termination_marks_latest_row() {
        let decode = |text: &str| {
            riscv_instruction::RiscvInstruction::decode_asm_with_context(
                text,
                riscv_instruction::IsaDecodeContext::RV64,
            )
            .expect("decode asm")
        };
        let mut output = HartExecutionOutput {
            hart_id: 0,
            instructions: vec![
                AsmInstructionExecution {
                    pc: 0x8000_0000,
                    instruction: decode("addi x1, x0, 1"),
                    source_annotation: None,
                    trace_inst_id: Some(riscv_fuzz_plugin_api::TraceInstId::new(0, 0)),
                    exceptions: Vec::new(),
                    register_writes: Vec::new(),
                    memory_writes: Vec::new(),
                    timing: InstructionTiming {
                        clk_start: Some(10),
                        clk_end: Some(11),
                        clk_span: Some(2),
                    },
                },
                AsmInstructionExecution {
                    pc: 0x8000_0004,
                    instruction: decode("addi x2, x0, 2"),
                    source_annotation: None,
                    trace_inst_id: Some(riscv_fuzz_plugin_api::TraceInstId::new(0, 1)),
                    exceptions: Vec::new(),
                    register_writes: Vec::new(),
                    memory_writes: Vec::new(),
                    timing: InstructionTiming {
                        clk_start: Some(20),
                        clk_end: Some(21),
                        clk_span: Some(2),
                    },
                },
            ],
        };

        annotate_partial_trace_termination(&mut output, "xiangshan partial stop");
        let main = &output.instructions[1];
        assert_eq!(main.exceptions.len(), 1);
        assert_eq!(main.exceptions[0].raw, "xiangshan partial stop");
    }

    #[test]
    fn xs_log_filter_keeps_required_trace_lines() {
        assert!(should_keep_xs_log_line(
            "[0][11] commit pc 0000000080000080 inst 79fd0000 wen 1 dst 19 data fffffffffffff7ff idx 00e c.lui s3,0xfffff"
        ));
        assert!(should_keep_xs_log_line(
            "[0][11] exception pc 000000008000008a inst 00009002 cause 0000000000000002"
        ));
        assert!(should_keep_xs_log_line(
            "retired hart 0 pc 80000000 wen 1 ldest 13 pdest 1f data 80 fflags: 0 vxsat: 0 clk_start 90 clk_end 92 clk_span 3"
        ));
        assert!(should_keep_xs_log_line(
            "store commit hart 0 pc 0x80000010 robidx 4 addr 80001ff0 data_lo 1122334455667788 data_hi 0 mask ff wline 0 vecsplit 0 clk_start 93 clk_end 97 clk_span 5"
        ));
        assert!(should_keep_xs_log_line(
            "trap hart 0 pc 80000014 intr 0 excp 1 exceptionVec b clk_start 98 clk_end 99 clk_span 2"
        ));
        assert!(should_keep_xs_log_line(
            "mem pc 0000000080000792 addr 000000008fffffe0 data 0000000045000000 mask 0x08"
        ));
        assert!(should_keep_xs_log_line("HIT GOOD TRAP"));
        assert!(should_keep_xs_log_line("cycleCnt = 77"));
    }

    #[test]
    fn xs_log_filter_drops_generic_noise() {
        assert!(!should_keep_xs_log_line(
            "[PERF ][time=1] SimTop.foo: some periodic debug message"
        ));
        assert!(!should_keep_xs_log_line(
            "[DEBUG][time=2] SimTop.bar: verbose internal state dump"
        ));
        assert!(!should_keep_xs_log_line(
            "random banner without trace payload"
        ));
    }

    #[test]
    fn xs_log_filter_extracts_payload_from_prefixed_noise() {
        let line = "[DEBUG][time=569] SimTop.foo: [DEBUG][time=569] SimTop.bar: [INFO ][time=569] SimTop.baz: retired hart  0 pc 0000010000000 wen 1 ldest  5 pdest 01 data 0000000000000001 fflags: 00000 vxsat: 0 clk_start 562 clk_end 566 clk_span 5";
        assert_eq!(
            extract_xs_log_payload(line),
            Some(
                "retired hart  0 pc 0000010000000 wen 1 ldest  5 pdest 01 data 0000000000000001 fflags: 00000 vxsat: 0 clk_start 562 clk_end 566 clk_span 5"
            )
        );
    }

    #[test]
    fn parse_prefixed_real_log_timing_lines() {
        let log = concat!(
            "[INFO ][time=4242] SimTop.cpu.l_soc.core_with_l2.core.backend.inner.ctrlBlock.rob: retired hart 0 pc 80000000 wen 1 ldest 13 pdest 1f data 80 fflags: 0 vxsat: 0 clk_start 90 clk_end 92 clk_span 3\n",
            "[INFO ][time=4243] SimTop.cpu.l_soc.core_with_l2.core.memBlock.inner.sbuffer: store commit hart 0 pc 0x80000010 robidx 4 addr 80001ff0 data_lo 1122334455667788 data_hi 0000000000000000 mask ff wline 0 vecsplit 0 clk_start 93 clk_end 97 clk_span 5\n",
            "[000] exception pc 0000000080000014 inst 00000073 cause 000000000000000b\n",
            "[INFO ][time=4244] SimTop.cpu.l_soc.core_with_l2.core.backend.inner.ctrlBlock.rob: trap hart 0 pc 80000014 intr 0 excp 1 exceptionVec b clk_start 98 clk_end 99 clk_span 2\n",
        );
        let trace = parse_xs_commit_trace_reader(log.as_bytes()).expect("parse failed");

        assert!(!trace.registers_contain_pc(0, 0x8000_0000));
        let mem = trace
            .memory_writes(0, 0x8000_0010)
            .expect("prefixed store payload should produce memory write");
        assert_eq!(mem.len(), 1);
        assert_eq!(mem[0].addr, 0x8000_1ff0u64);
        assert_eq!(mem[0].timing.clk_end, Some(97));
        let retire_timing = trace
            .instruction_timing_entries(0, 0x8000_0000)
            .expect("missing prefixed retire timing");
        assert_eq!(retire_timing[0].timing.clk_span, Some(3));
        let store_timing = trace
            .instruction_timing_entries(0, 0x8000_0010)
            .expect("missing prefixed store timing");
        assert_eq!(store_timing[0].timing.clk_end, Some(97));

        let trap = trace
            .exception(0, 0x8000_0014)
            .expect("missing prefixed trap");
        assert_eq!(trap.cause, 0xb);
        assert_eq!(trap.timing.clk_start, Some(98));
        assert_eq!(trap.timing.clk_span, Some(2));
    }

    #[test]
    fn archived_case_1778510663_accepts_missing_float_dstprefix_commit_iter_001() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("repo root");
        let case_dir = repo_root.join(
            "artifacts/singlecore_spike_audit/rv64_baseline/spike_xiangshan_rv64/1778510663_702404874_rv64_7_w00/iter_001/xiangshan",
        );
        let trace_path = case_dir.join("skiptrap.commit.log");
        if !trace_path.exists() {
            eprintln!(
                "skipping XiangShan artifact replay; fixture is not present: {}",
                trace_path.display()
            );
            return;
        }
        let trace = parse_xs_commit_trace(&trace_path)
            .expect("missing float dstprefix should now be accepted");
        assert!(trace.has_content());
    }

    #[test]
    fn archived_case_1778510663_accepts_missing_float_dstprefix_commit_iter_000() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("repo root");
        let case_dir = repo_root.join(
            "artifacts/singlecore_spike_audit/rv64_baseline/spike_xiangshan_rv64/1778510663_702404874_rv64_7_w00/iter_000/xiangshan",
        );
        let trace_path = case_dir.join("skiptrap.commit.log");
        if !trace_path.exists() {
            eprintln!(
                "skipping XiangShan artifact replay; fixture is not present: {}",
                trace_path.display()
            );
            return;
        }
        let trace = parse_xs_commit_trace(&trace_path)
            .expect("missing float dstprefix should now be accepted");
        assert!(trace.has_content());
    }

    #[test]
    fn archived_case_1778510663_accepts_missing_float_dstprefix_commit_iter_002() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("repo root");
        let case_dir = repo_root.join(
            "artifacts/singlecore_spike_audit/rv64_baseline/spike_xiangshan_rv64/1778510663_702404874_rv64_7_w00/iter_002/xiangshan",
        );
        let trace_path = case_dir.join("skiptrap.commit.log");
        if !trace_path.exists() {
            eprintln!(
                "skipping XiangShan artifact replay; fixture is not present: {}",
                trace_path.display()
            );
            return;
        }
        let trace = parse_xs_commit_trace(&trace_path)
            .expect("missing float dstprefix should now be accepted");
        assert!(trace.has_content());
    }

    #[test]
    fn live_case_1778521970_accepts_missing_float_dstprefix_commit() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("repo root");
        let case_dir = repo_root.join(
            "artifacts/live_single_core_smoke_rv64/spike_xiangshan_rv64/1778521970_453620778_rv64_0_w00/iter_000/xiangshan",
        );
        let trace_path = case_dir.join("skiptrap.commit.log");
        if !trace_path.exists() {
            eprintln!(
                "skipping XiangShan artifact replay; fixture is not present: {}",
                trace_path.display()
            );
            return;
        }
        let trace = parse_xs_commit_trace(&trace_path)
            .expect("missing float dstprefix should now be accepted");
        assert!(trace.has_content());
    }

    #[test]
    fn associate_missing_register_writes_keeps_logged_register_class() {
        let tracer = Tracer::from_traced_instructions(vec![
            riscv_fuzz_toolchain::tracer::TracedInstruction {
                pc: 0x8000_14dc,
                disassembly: "fsgnjn.d f20, f16, f19".to_string(),
                parsed_disassembly: None,
                source_line: Some("fsgnjn.d f20, f16, f19".to_string()),
                trace_inst_local_index: Some(0),
                source_instruction: None,
            },
        ]);

        let mut trace = XiangShanTrace::default();
        trace.registers.insert(
            key(0, 0x8000_14dc),
            vec![RegisterWrite {
                pc: 0x8000_14dc,
                register: "x20".to_string(),
                value: 0x7fff_ffff_7fff_e8dc,
                timing: InstructionTiming::default(),
            }],
        );
        trace.instruction_timings.insert(
            key(0, 0x8000_14dc),
            vec![InstructionTrace {
                pc: 0x8000_14dc,
                timing: InstructionTiming::default(),
            }],
        );

        let register_writes =
            associate_missing_register_writes(&trace, 0, &tracer, &HashMap::new());
        let writes = register_writes
            .get(&0x8000_14dc)
            .expect("register write missing");

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].register, "x20");
        assert_eq!(writes[0].value, 0x7fff_ffff_7fff_e8dc);
    }

    #[test]
    fn archived_case_1778507411_accepts_missing_float_dstprefix_commit() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("repo root");
        let case_dir = repo_root.join(
            "artifacts/singlecore_spike_audit/rv64_baseline/spike_xiangshan_rv64/1778507411_175642835_rv64_0_w00/iter_000/xiangshan",
        );
        let trace_path = case_dir.join("skiptrap.commit.log");
        if !trace_path.exists() {
            eprintln!(
                "skipping XiangShan artifact replay; fixture is not present: {}",
                trace_path.display()
            );
            return;
        }
        let trace = parse_xs_commit_trace(&trace_path)
            .expect("missing float dstprefix should now be accepted");
        assert!(trace.has_content());
    }

    #[test]
    fn archived_case_1778507839_accepts_missing_float_dstprefix_commit() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("repo root");
        let case_dir = repo_root.join(
            "artifacts/singlecore_spike_audit/rv64_baseline/spike_xiangshan_rv64/1778507839_510158935_rv64_1_w00/iter_000/xiangshan",
        );
        let trace_path = case_dir.join("skiptrap.commit.log");
        if !trace_path.exists() {
            eprintln!(
                "skipping XiangShan artifact replay; fixture is not present: {}",
                trace_path.display()
            );
            return;
        }
        let trace = parse_xs_commit_trace(&trace_path)
            .expect("missing float dstprefix should now be accepted");
        assert!(trace.has_content());
    }

    #[test]
    fn archived_case_1778508283_accepts_missing_float_dstprefix_commit() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("repo root");
        let case_dir = repo_root.join(
            "artifacts/singlecore_spike_audit/rv64_baseline/spike_xiangshan_rv64/1778508283_545928146_rv64_3_w00/iter_001/xiangshan",
        );
        let trace_path = case_dir.join("skiptrap.commit.log");
        if !trace_path.exists() {
            eprintln!(
                "skipping XiangShan artifact replay; fixture is not present: {}",
                trace_path.display()
            );
            return;
        }
        let trace = parse_xs_commit_trace(&trace_path)
            .expect("missing float dstprefix should now be accepted");
        assert!(trace.has_content());
    }

    #[test]
    fn archived_case_1778510143_accepts_missing_float_dstprefix_commit() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .expect("repo root");
        let case_dir = repo_root.join(
            "artifacts/singlecore_spike_audit/rv64_baseline/spike_xiangshan_rv64/1778510143_991957263_rv64_5_w00/iter_000/xiangshan",
        );
        let trace_path = case_dir.join("skiptrap.commit.log");
        if !trace_path.exists() {
            eprintln!(
                "skipping XiangShan artifact replay; fixture is not present: {}",
                trace_path.display()
            );
            return;
        }
        let trace = parse_xs_commit_trace(&trace_path)
            .expect("missing float dstprefix should now be accepted");
        assert!(trace.has_content());
    }

    #[test]
    fn perf_cycles_only_use_commit_log() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let run_dir = std::env::temp_dir().join(format!("xiangshan_perf_{unique}"));
        fs::create_dir_all(&run_dir).expect("create run dir");
        let commit_log_path = run_dir.join("skiptrap.commit.log");
        fs::write(&commit_log_path, "").expect("write commit log");

        let perf = extract_perf_metrics_from_xs_trace(
            ISABase::Rv64,
            "xiangshan".to_string(),
            1,
            false,
            Path::new("/tmp"),
            &XiangShanTrace::default(),
            &commit_log_path,
        )
        .expect("perf extraction should read commit log");
        assert_eq!(perf.global_cycles, None);

        fs::write(&commit_log_path, "Cycles: 77").expect("rewrite commit log");
        let perf = extract_perf_metrics_from_xs_trace(
            ISABase::Rv64,
            "xiangshan".to_string(),
            1,
            false,
            Path::new("/tmp"),
            &XiangShanTrace::default(),
            &commit_log_path,
        )
        .expect("perf extraction should read commit log");
        let _ = fs::remove_dir_all(&run_dir);

        assert_eq!(perf.global_cycles, Some(77));
    }

    #[test]
    fn perf_requires_commit_log() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let run_dir = std::env::temp_dir().join(format!("xiangshan_missing_{unique}"));
        fs::create_dir_all(&run_dir).expect("create run dir");
        let commit_log_path = run_dir.join("skiptrap.commit.log");

        let err = extract_perf_metrics_from_xs_trace(
            ISABase::Rv64,
            "xiangshan".to_string(),
            1,
            false,
            Path::new("/tmp"),
            &XiangShanTrace::default(),
            &commit_log_path,
        )
        .expect_err("missing commit log should fail");
        let _ = fs::remove_dir_all(&run_dir);

        assert!(format!("{err}").contains("skiptrap.commit.log"));
    }
}
