use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};
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
use riscv_fuzz_plugin_api::metadata::FloatExtension;
use riscv_fuzz_plugin_api::streaming::{
    InstructionEvent, StreamingOutputDelta, read_complete_utf8_prefix,
};
use riscv_fuzz_plugin_api::{
    ConfigError, ElfError, ExceptionCause, ExecutionOutput, HartExecutionOutput, ISABase,
    ImplExecutionError, InstructionTiming, IsaExtensions, LogParseError, MemValue as TraitMemValue,
    ParseError, PerHartTypedSequences, PerfMetrics, ProcessError,
    RegisterValue as TraitRegisterValue, configure_command_process_group, kill_child_process_group,
    typed_sequences_for_hart,
};
use riscv_fuzz_toolchain::{
    build_elf::{ElfBuildResult, build_elf_with_extensions},
    error::BuildElfError,
    tracer::{
        Tracer, build_raw_hart_execution_output, filter_per_hart_output_for_streaming_by_tracer,
    },
};

pub(crate) type PerHartAsm<'a> = &'a [(u32, &'a [String])];

const ARA_RV32_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32_1C";
const ARA_RV32F_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32F_1C";
const ARA_RV64_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV64_1C";
const ARA_RV32_COV_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32_1C_COV";
const ARA_RV32F_COV_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32F_1C_COV";
const ARA_RV64_COV_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV64_1C_COV";
const ARA_RV32_COV_LIGHT_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32_1C_COV_LIGHT";
const ARA_RV32F_COV_LIGHT_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32F_1C_COV_LIGHT";
const ARA_RV64_COV_LIGHT_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV64_1C_COV_LIGHT";

const ARA_RV32_2HART_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32_2C";
const ARA_RV32F_2HART_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32F_2C";
const ARA_RV64_2HART_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV64_2C";
const ARA_RV32_2HART_COV_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32_2C_COV";
const ARA_RV32F_2HART_COV_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32F_2C_COV";
const ARA_RV64_2HART_COV_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV64_2C_COV";
const ARA_RV32_2HART_COV_LIGHT_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32_2C_COV_LIGHT";
const ARA_RV32F_2HART_COV_LIGHT_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV32F_2C_COV_LIGHT";
const ARA_RV64_2HART_COV_LIGHT_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV64_2C_COV_LIGHT";
const ARA_EXIT_LOOP_REPEAT_THRESHOLD: usize = 32;
const ARA_TRACE_TAIL_BYTES: u64 = 64 * 1024;

#[derive(Debug, Error)]
pub enum ARAError {
    #[error("environment variable {var} is not set")]
    EnvVarNotSet { var: String },

    #[error("ARA binary not found: {path}")]
    BinaryNotFound { path: String },

    #[error(transparent)]
    ProcessError(#[from] ProcessError),

    #[error(transparent)]
    LogParseError(#[from] LogParseError),

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
}

impl From<ARAError> for ImplExecutionError {
    fn from(value: ARAError) -> Self {
        match value {
            ARAError::ProcessError(err) => err.into(),
            ARAError::LogParseError(err) => err.into(),
            ARAError::ParseError(err) => err.into(),
            ARAError::ElfError(err) => err.into(),
            ARAError::IoError(err) => err.into(),
            other => ImplExecutionError::Other(Box::new(other)),
        }
    }
}

/// Check if a command is available, either as a file path or in PATH
fn is_command_available(path_or_command: &Path) -> bool {
    // If it looks like a file path (contains directory separator or is absolute), check if file exists
    if path_or_command.components().count() > 1 || path_or_command.is_absolute() {
        return path_or_command.exists();
    }

    // Otherwise, search in PATH
    if let Some(path_env) = env::var_os("PATH") {
        for dir in env::split_paths(&path_env) {
            let candidate = dir.join(path_or_command);
            if candidate.exists() {
                return true;
            }
        }
    }

    false
}

/// Resolve ARA binary path by checking environment variables.
/// Returns error if none of the environment variables are set.
fn get_binary_from_env(env_var: &str) -> Result<PathBuf, ARAError> {
    let path = env::var(env_var).map_err(|_| ARAError::EnvVarNotSet {
        var: env_var.to_string(),
    })?;

    if path.is_empty() {
        return Err(ARAError::EnvVarNotSet {
            var: env_var.to_string(),
        });
    }

    let path_buf = PathBuf::from(&path);
    if !path_buf.exists() {
        return Err(ARAError::BinaryNotFound {
            path: path_buf.display().to_string(),
        });
    }

    Ok(path_buf)
}

fn resolve_wrapper_elf_path(elf_path: &Path, run_dir: &Path) -> Result<PathBuf, ARAError> {
    if elf_path.is_absolute() {
        if elf_path.is_file() {
            return Ok(elf_path.to_path_buf());
        }
        return Err(ARAError::BuildElfError(format!(
            "built ELF path does not exist: {}",
            elf_path.display()
        )));
    }

    if elf_path.exists() {
        return fs::canonicalize(elf_path).map_err(|err| {
            ARAError::BuildElfError(format!(
                "failed to canonicalize built ELF {}: {err}",
                elf_path.display()
            ))
        });
    }

    let Some(file_name) = elf_path.file_name() else {
        return Err(ARAError::BuildElfError(format!(
            "built ELF path has no file name: {}",
            elf_path.display()
        )));
    };

    let candidate = run_dir.join(file_name);
    if !candidate.is_file() {
        return Err(ARAError::BuildElfError(format!(
            "built ELF path does not exist: {}; run-dir candidate also missing: {}",
            elf_path.display(),
            candidate.display()
        )));
    }

    fs::canonicalize(&candidate).map_err(|err| {
        ARAError::BuildElfError(format!(
            "failed to canonicalize run-dir ELF candidate {}: {err}",
            candidate.display()
        ))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
struct TraceKey {
    hart_id: u32,
    pc: u64,
}

impl TraceKey {
    fn new(hart_id: u32, pc: u64) -> Self {
        Self { hart_id, pc }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ExceptionEvent {
    pub source: ExceptionSource,
    pub kind: String,
    pub pc: u64,
    pub code: Option<u64>,
    pub timing: InstructionTiming,
}

#[derive(Debug, Clone, Serialize)]
pub enum ExceptionSource {
    Rvfi,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegisterWrite {
    pub index: u8,
    pub pc: u64,
    pub register: String,
    pub value: u64,
    pub instruction: Option<u32>,
    pub timing: InstructionTiming,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemoryWriteByte {
    pub lane: u8,
    pub address: u64,
    pub value: u8,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryWrite {
    pub pc: u64,
    pub address: u64,
    pub value: u64,
    pub width: usize,
    pub mask: u64,
    pub instruction: Option<u32>,
    pub byte_writes: Vec<MemoryWriteByte>,
    pub timing: InstructionTiming,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstructionTrace {
    pub pc: u64,
    pub timing: InstructionTiming,
}

/// ARA RVFI trace organized by PC (matching Rocket's structure)
#[derive(Debug, Clone, Serialize, Default)]
pub struct RvfiTrace {
    register_writes: HashMap<TraceKey, Vec<RegisterWrite>>,
    memory_writes: HashMap<TraceKey, Vec<MemoryWrite>>,
    exceptions: HashMap<TraceKey, ExceptionEvent>,
    instruction_timings: HashMap<TraceKey, InstructionTrace>,
}

impl RvfiTrace {
    fn has_content(&self) -> bool {
        !self.register_writes.is_empty()
            || !self.memory_writes.is_empty()
            || !self.exceptions.is_empty()
            || !self.instruction_timings.is_empty()
    }

    fn observed_pcs_for_hart(&self, hart_id: u32) -> Vec<u64> {
        self.register_writes
            .keys()
            .chain(self.memory_writes.keys())
            .chain(self.exceptions.keys())
            .chain(self.instruction_timings.keys())
            .filter(|key| key.hart_id == hart_id)
            .map(|key| key.pc)
            .collect()
    }

    fn register_writes_for_hart_pc(&self, hart_id: u32, pc: u64) -> Option<&[RegisterWrite]> {
        self.register_writes
            .get(&TraceKey::new(hart_id, pc))
            .map(Vec::as_slice)
    }

    fn memory_writes_for_hart_pc(&self, hart_id: u32, pc: u64) -> Option<&[MemoryWrite]> {
        self.memory_writes
            .get(&TraceKey::new(hart_id, pc))
            .map(Vec::as_slice)
    }

    fn exception_for_hart_pc(&self, hart_id: u32, pc: u64) -> Option<&ExceptionEvent> {
        self.exceptions.get(&TraceKey::new(hart_id, pc))
    }

    fn instruction_timing_for_hart_pc(&self, hart_id: u32, pc: u64) -> Option<&InstructionTrace> {
        self.instruction_timings.get(&TraceKey::new(hart_id, pc))
    }
}

#[derive(Debug, Clone)]
pub struct TraceConfig {
    pub rvfi_paths: Vec<PathBuf>,
}

impl TraceConfig {
    pub fn new(rvfi_path: impl Into<PathBuf>) -> Self {
        Self {
            rvfi_paths: vec![rvfi_path.into()],
        }
    }

    pub fn from_run_directory(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            rvfi_paths: vec![dir.join("trace_rvfi_hart_00000000.dasm")],
        }
    }

    pub fn from_run_directory_with_harts(
        dir: impl AsRef<Path>,
        expected_harts: u32,
    ) -> Result<Self, ARAError> {
        let dir = dir.as_ref();
        let mut rvfi_paths = Vec::new();
        let harts = expected_harts.max(1);

        for hart in 0u32..harts {
            let candidates = ara_trace_path_candidates(dir, hart);
            let Some(path) = candidates.iter().find(|path| path.is_file()) else {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "missing ARA RVFI trace for hart{hart}: tried {}",
                        candidates
                            .iter()
                            .map(|p| p.display().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                )
                .into());
            };
            rvfi_paths.push(path.to_path_buf());
        }

        Ok(Self { rvfi_paths })
    }
}

pub fn analyze_traces(config: &TraceConfig) -> Result<RvfiTrace, ARAError> {
    if config.rvfi_paths.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no ARA RVFI traces configured for analysis",
        )
        .into());
    }

    let mut traces = Vec::with_capacity(config.rvfi_paths.len());
    for (index, path) in config.rvfi_paths.iter().enumerate() {
        info!("Analyzing RVFI trace from {:?}", path);
        let hart_id = trace_hart_id_from_path(path).unwrap_or(index as u32);
        traces.push(parse_rvfi_file(path, hart_id)?);
    }

    let rvfi = merge_rvfi_traces(traces)?;

    let total_reg_writes: usize = rvfi.register_writes.values().map(|v| v.len()).sum();
    let total_mem_writes: usize = rvfi.memory_writes.values().map(|v| v.len()).sum();
    let total_timing_records = rvfi.instruction_timings.len();
    debug!(
        "RVFI trace parsed: {} PCs with register writes (total {}), {} PCs with memory writes (total {}), {} PCs with exceptions, {} PCs with timing-only entries (total {})",
        rvfi.register_writes.len(),
        total_reg_writes,
        rvfi.memory_writes.len(),
        total_mem_writes,
        rvfi.exceptions.len(),
        rvfi.instruction_timings.len(),
        total_timing_records
    );

    info!("Trace analysis complete");
    Ok(rvfi)
}

fn merge_rvfi_traces(traces: Vec<RvfiTrace>) -> Result<RvfiTrace, ARAError> {
    let mut merged = RvfiTrace::default();

    for trace in traces {
        for (key, writes) in trace.register_writes {
            let merged_writes = merged.register_writes.entry(key).or_default();
            for write in writes {
                if let Some(existing) = merged_writes
                    .iter_mut()
                    .find(|existing| existing.register == write.register)
                {
                    if register_write_should_replace(existing, &write) {
                        *existing = write;
                    }
                    continue;
                }
                merged_writes.push(write);
            }
        }
        for (key, writes) in trace.memory_writes {
            let merged_writes = merged.memory_writes.entry(key).or_default();
            for write in writes {
                if let Some(existing) = merged_writes
                    .iter_mut()
                    .find(|existing| existing.address == write.address)
                {
                    if memory_write_should_replace(existing, &write) {
                        *existing = write;
                    }
                    continue;
                }
                merged_writes.push(write);
            }
        }
        for (key, event) in trace.exceptions {
            if merged.exceptions.insert(key, event).is_some() {
                return Err(invalid_ara_trace(format!(
                    "duplicate ARA exception at hart={} pc=0x{:x}",
                    key.hart_id, key.pc
                )));
            }
        }
        for (key, timing) in trace.instruction_timings {
            if let Some(existing) = merged.instruction_timings.get_mut(&key) {
                merge_instruction_timing(&mut existing.timing, &timing.timing);
            } else {
                merged.instruction_timings.insert(key, timing);
            }
        }
    }

    Ok(merged)
}

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub work_dir: PathBuf,
    pub isa_base: ISABase,
    pub core_id: String,
    pub timeout: Option<Duration>,
    pub term_after_cycles: u64,
    pub coverage_mode: coverage::CoverageMode,
    pub extension_override: Option<IsaExtensions>,
}

impl ExecutorConfig {
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(15);
    pub const DEFAULT_TERM_AFTER_CYCLES: u64 = 10_000_000;

    pub fn new(work_dir: PathBuf, isa_base: ISABase, core_id: String) -> Result<Self, ARAError> {
        Ok(Self {
            work_dir,
            isa_base,
            core_id,
            timeout: Some(Self::DEFAULT_TIMEOUT),
            term_after_cycles: Self::DEFAULT_TERM_AFTER_CYCLES,
            coverage_mode: coverage::CoverageMode::Off,
            extension_override: None,
        })
    }

    pub fn set_extension_override(&mut self, extensions: IsaExtensions) {
        self.extension_override = Some(extensions);
    }

    pub fn resolved_extensions(&self) -> IsaExtensions {
        if let Some(extensions) = &self.extension_override {
            return extensions.clone();
        }

        isa_profiles().for_isa(self.isa_base).extensions.clone()
    }
}

pub struct AraExecutor {
    config: ExecutorConfig,
}

impl AraExecutor {
    pub fn new(config: ExecutorConfig) -> Self {
        debug!("Instantiating ARA executor with config: {:?}", config);
        Self { config }
    }

    fn get_ara_binary(&self, multicore: bool) -> Result<PathBuf, ARAError> {
        let cov_mode = self.config.coverage_mode;
        let profiles = isa_profiles();
        let profile = profiles.for_isa(self.config.isa_base);
        let float_choice = profile.active_float_extension(&self.config.resolved_extensions());
        debug!("ARA: Detected coverage mode: {:?}", cov_mode);

        let env_var = match (multicore, self.config.isa_base, cov_mode, float_choice) {
            (false, ISABase::Rv32, coverage::CoverageMode::Off, FloatExtension::None) => {
                ARA_RV32_BINARY_ENV
            }
            (false, ISABase::Rv32, coverage::CoverageMode::Full, FloatExtension::None) => {
                ARA_RV32_COV_BINARY_ENV
            }
            (false, ISABase::Rv32, coverage::CoverageMode::Light, FloatExtension::None) => {
                ARA_RV32_COV_LIGHT_BINARY_ENV
            }
            (false, ISABase::Rv32, coverage::CoverageMode::Off, FloatExtension::F) => {
                ARA_RV32F_BINARY_ENV
            }
            (false, ISABase::Rv32, coverage::CoverageMode::Full, FloatExtension::F) => {
                ARA_RV32F_COV_BINARY_ENV
            }
            (false, ISABase::Rv32, coverage::CoverageMode::Light, FloatExtension::F) => {
                ARA_RV32F_COV_LIGHT_BINARY_ENV
            }
            (false, ISABase::Rv64, coverage::CoverageMode::Off, _) => ARA_RV64_BINARY_ENV,
            (false, ISABase::Rv64, coverage::CoverageMode::Full, _) => ARA_RV64_COV_BINARY_ENV,
            (false, ISABase::Rv64, coverage::CoverageMode::Light, _) => {
                ARA_RV64_COV_LIGHT_BINARY_ENV
            }
            (true, ISABase::Rv32, coverage::CoverageMode::Off, FloatExtension::None) => {
                ARA_RV32_2HART_BINARY_ENV
            }
            (true, ISABase::Rv32, coverage::CoverageMode::Full, FloatExtension::None) => {
                ARA_RV32_2HART_COV_BINARY_ENV
            }
            (true, ISABase::Rv32, coverage::CoverageMode::Light, FloatExtension::None) => {
                ARA_RV32_2HART_COV_LIGHT_BINARY_ENV
            }
            (true, ISABase::Rv32, coverage::CoverageMode::Off, FloatExtension::F) => {
                ARA_RV32F_2HART_BINARY_ENV
            }
            (true, ISABase::Rv32, coverage::CoverageMode::Full, FloatExtension::F) => {
                ARA_RV32F_2HART_COV_BINARY_ENV
            }
            (true, ISABase::Rv32, coverage::CoverageMode::Light, FloatExtension::F) => {
                ARA_RV32F_2HART_COV_LIGHT_BINARY_ENV
            }
            (true, ISABase::Rv64, coverage::CoverageMode::Off, _) => ARA_RV64_2HART_BINARY_ENV,
            (true, ISABase::Rv64, coverage::CoverageMode::Full, _) => ARA_RV64_2HART_COV_BINARY_ENV,
            (true, ISABase::Rv64, coverage::CoverageMode::Light, _) => {
                ARA_RV64_2HART_COV_LIGHT_BINARY_ENV
            }
            (multicore, ISABase::Rv32, _, FloatExtension::FD) => {
                return Err(ARAError::BuildElfError(format!(
                    "ARA RV32 does not provide an FD-capable wrapper (multicore={multicore})"
                )));
            }
        };

        debug!("ARA: Using environment variable: {}", env_var);

        get_binary_from_env(env_var)
    }

    fn ensure_work_dir(&self, dir: &Path) -> Result<(), ARAError> {
        debug!("Ensuring work directory exists at {}", dir.display());
        fs::create_dir_all(dir)?;
        debug!("Work directory ready at {}", dir.display());
        Ok(())
    }

    fn cleanup_trace_files(&self, trace_dir: &Path) {
        for hart in 0u32..2u32 {
            for path in ara_trace_path_candidates(trace_dir, hart) {
                if !path.exists() {
                    continue;
                }
                match fs::remove_file(&path) {
                    Ok(_) => debug!("Removed stale trace file {}", path.display()),
                    Err(err) => warn!(
                        "Failed to remove stale trace file {}: {}",
                        path.display(),
                        err
                    ),
                }
            }
        }
    }

    fn build_program(&self, per_hart: PerHartAsm<'_>) -> Result<String, ARAError> {
        let total_insts: usize = per_hart.iter().map(|(_, lines)| lines.len()).sum();
        debug!(
            "Building assembly program for {} testcase instructions",
            total_insts
        );
        Ok(build_asm(per_hart, self.config.isa_base)?)
    }

    fn compile_and_run_with_dir(
        &self,
        run_dir: &Path,
        per_hart: PerHartAsm<'_>,
    ) -> Result<RunArtifacts, ARAError> {
        let total_insts: usize = per_hart.iter().map(|(_, lines)| lines.len()).sum();
        info!(
            "Starting ARA compile-and-run flow with {} testcase instructions in {}",
            total_insts,
            run_dir.display()
        );
        self.ensure_work_dir(run_dir)?;
        self.cleanup_trace_files(run_dir);
        let harts = per_hart.len() as u32;

        let asm_path = run_dir.join("program.S");
        let program = self.build_program(per_hart)?;
        fs::write(&asm_path, &program)?;
        debug!("Assembly written to {}", asm_path.display());

        let extensions = self.config.resolved_extensions();
        let build_result = build_elf_with_extensions(&asm_path, &extensions, linker_script())
            .map_err(|err| {
                error!(
                    "Failed to build ELF for ARA execution (asm={}, isa={:?}, impl={:?}): {:#}",
                    asm_path.display(),
                    self.config.isa_base,
                    self.config.core_id.clone(),
                    err
                );
                err
            })?;
        info!(
            "ELF build complete: exe={}, disasm={}",
            build_result.executable_file.display(),
            build_result.disassembly_file.display()
        );

        self.invoke_ara(run_dir, &build_result, harts)?;
        info!("ARA execution finished successfully");

        let trace_config = TraceConfig::from_run_directory_with_harts(run_dir, harts)?;
        debug!("Collecting traces using {:?}", trace_config);
        let summary = analyze_traces(&trace_config)?;
        info!("Trace parsing succeeded");

        Ok(RunArtifacts {
            rvfi: summary,
            disassembly_file: build_result.disassembly_file,
        })
    }

    /// Compile the program and run ARA, but skip trace parsing / ExecutionOutput construction.
    ///
    /// Intended for fast outer-loop workflows such as coverage-guided generation where only the
    /// side-effects (e.g. coverage.dat) are needed.
    pub fn compile_and_run_only_with_dir(
        &self,
        run_dir: &Path,
        per_hart: PerHartAsm<'_>,
    ) -> Result<(), ARAError> {
        let total_insts: usize = per_hart.iter().map(|(_, lines)| lines.len()).sum();
        info!(
            "Starting ARA compile-and-run (no trace analysis) with {} testcase instructions in {}",
            total_insts,
            run_dir.display()
        );
        self.ensure_work_dir(run_dir)?;
        self.cleanup_trace_files(run_dir);
        let harts = per_hart.len() as u32;

        let asm_path = run_dir.join("program.S");
        let program = self.build_program(per_hart)?;
        fs::write(&asm_path, &program)?;

        let extensions = self.config.resolved_extensions();
        let build_result = build_elf_with_extensions(&asm_path, &extensions, linker_script())
            .map_err(|err| {
                error!(
                    "Failed to build ELF for ARA execution (asm={}, isa={:?}, impl={:?}): {:#}",
                    asm_path.display(),
                    self.config.isa_base,
                    self.config.core_id.clone(),
                    err
                );
                err
            })?;

        self.invoke_ara(run_dir, &build_result, harts)?;
        Ok(())
    }

    fn invoke_ara(
        &self,
        run_dir: &Path,
        build_result: &ElfBuildResult,
        harts: u32,
    ) -> Result<(), ARAError> {
        let ara_binary = self.get_ara_binary(harts > 1)?;
        info!(
            "Invoking ARA binary {} with executable {}",
            ara_binary.display(),
            build_result.executable_file.display()
        );
        if !is_command_available(&ara_binary) {
            error!(
                "ARA binary not available at {} (not found as file or in PATH)",
                ara_binary.display()
            );
            return Err(ARAError::BinaryNotFound {
                path: ara_binary.display().to_string(),
            });
        }

        let use_coverage = self.config.coverage_mode.is_enabled();
        let coverage_path = if use_coverage {
            let path = run_dir.join("coverage.dat");
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            Some(path)
        } else {
            None
        };

        let mut cmd = Command::new(&ara_binary);

        // ARA is executed with `current_dir(run_dir)`. Resolve the built ELF so
        // path issues fail immediately instead of silently degrading.
        let elf_path = resolve_wrapper_elf_path(&build_result.executable_file, run_dir)?;
        cmd.arg(format!(
            "--term-after-cycles={}",
            self.config.term_after_cycles
        ))
        .arg("-l")
        .arg(format!("ram,{},elf", elf_path.display()));

        if let Some(ref cov) = coverage_path {
            cmd.arg(format!("+covfile={}", cov.display()));
        }

        cmd.current_dir(run_dir);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        configure_command_process_group(&mut cmd);
        let command_display = format!("{:?}", cmd);
        info!("Launching ARA: {}", command_display);

        let mut child = cmd.spawn().map_err(|err| {
            ARAError::from(ProcessError::SpawnFailed {
                command: command_display.clone(),
                source: err,
            })
        })?;

        let stdout_reader = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("failed to capture ARA stdout"))?;
        let stderr_reader = child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("failed to capture ARA stderr"))?;

        let stdout_handle = thread::spawn(move || -> Result<Vec<u8>, io::Error> {
            let mut buffer = Vec::new();
            let mut reader = BufReader::new(stdout_reader);
            reader.read_to_end(&mut buffer)?;
            Ok(buffer)
        });
        let stderr_handle = thread::spawn(move || -> Result<Vec<u8>, io::Error> {
            let mut buffer = Vec::new();
            let mut reader = BufReader::new(stderr_reader);
            reader.read_to_end(&mut buffer)?;
            Ok(buffer)
        });

        let timeout = self.config.timeout;
        let exit_loop_pc = exit_loop_pc_from_dump(&build_result.disassembly_file);
        let start = Instant::now();
        let mut timed_out = false;
        let mut completed_by_exit_loop = false;
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if let Some(pc) = exit_loop_pc {
                let repeat_count = ara_exit_loop_repetition_count(run_dir, harts, pc);
                if repeat_count >= ARA_EXIT_LOOP_REPEAT_THRESHOLD {
                    info!(
                        "ARA reached exit loop at PC {pc:#x} ({repeat_count} repeated trace entries); stopping simulator for trace analysis"
                    );
                    completed_by_exit_loop = true;
                    let _ = kill_child_process_group(&mut child);
                    break child.wait()?;
                }
            }
            if let Some(limit) = timeout
                && start.elapsed() >= limit
            {
                timed_out = true;
                let _ = kill_child_process_group(&mut child);
                break child.wait()?;
            }
            thread::sleep(Duration::from_millis(50));
        };

        let stdout_bytes = stdout_handle
            .join()
            .map_err(|_| io::Error::other("ARA stdout reader thread panicked"))??;
        let stderr_bytes = stderr_handle
            .join()
            .map_err(|_| io::Error::other("ARA stderr reader thread panicked"))??;

        let stdout_path = run_dir.join("ara_stdout.log");
        let stderr_path = run_dir.join("ara_stderr.log");
        fs::write(&stdout_path, &stdout_bytes).map_err(|err| {
            error!(
                "Failed to persist ARA stdout to {}: {}",
                stdout_path.display(),
                err
            );
            ARAError::from(ProcessError::FileWriteFailed {
                path: stdout_path.clone(),
                source: err,
            })
        })?;
        fs::write(&stderr_path, &stderr_bytes).map_err(|err| {
            error!(
                "Failed to persist ARA stderr to {}: {}",
                stderr_path.display(),
                err
            );
            ARAError::from(ProcessError::FileWriteFailed {
                path: stderr_path.clone(),
                source: err,
            })
        })?;

        debug!(
            "ARA stdout saved to {} | stderr saved to {}",
            stdout_path.display(),
            stderr_path.display()
        );

        if timed_out {
            if let Some(limit) = timeout {
                error!(
                    "ARA execution timed out after {:?}. Logs saved to {} and {}",
                    limit,
                    stdout_path.display(),
                    stderr_path.display()
                );
                return Err(ARAError::from(ProcessError::TimedOut {
                    command: command_display,
                    timeout: limit,
                }));
            } else {
                unreachable!("timed_out implies timeout was set");
            }
        }

        if !completed_by_exit_loop && !status.success() {
            let stdout_text = String::from_utf8_lossy(&stdout_bytes).to_string();
            let stderr_text = String::from_utf8_lossy(&stderr_bytes).to_string();
            let combined_error = if stdout_text.trim().is_empty() {
                stderr_text.clone()
            } else if stderr_text.trim().is_empty() {
                stdout_text.clone()
            } else {
                format!("{stderr_text}\n--- stdout ---\n{stdout_text}")
            };
            error!(
                "ARA execution failed with status {:?}. Logs saved to {} and {}",
                status.code(),
                stdout_path.display(),
                stderr_path.display()
            );
            return Err(ARAError::from(ProcessError::ProcessFailed {
                command: command_display,
                stderr: combined_error,
            }));
        }

        if let Some(cov_path) = coverage_path {
            if !cov_path.exists() {
                return Err(ARAError::from(ProcessError::ProcessFailed {
                    command: command_display.clone(),
                    stderr: format!(
                        "coverage enabled but coverage file is missing: {}",
                        cov_path.display()
                    ),
                }));
            }

            let summary =
                riscv_fuzz_toolchain::coverage::parse_verilator_dat(&cov_path).map_err(|err| {
                    ARAError::from(ProcessError::ProcessFailed {
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

        if completed_by_exit_loop {
            debug!("ARA process stopped after reaching wrapper exit loop");
        } else {
            debug!("ARA process exited successfully");
        }
        Ok(())
    }
}

struct RunArtifacts {
    rvfi: RvfiTrace,
    disassembly_file: PathBuf,
}

impl AraExecutor {
    pub fn execute<T: AsRef<Path>>(
        &self,
        run_folder: T,
        per_hart: PerHartAsm<'_>,
    ) -> Result<ExecutionOutput, ARAError> {
        let total_insts: usize = per_hart.iter().map(|(_, lines)| lines.len()).sum();
        info!(
            "Executing ARA run for {} testcase instructions",
            total_insts
        );
        let run_dir = run_folder.as_ref().to_owned();
        info!("Using ARA run directory {}", run_dir.display());

        let artifacts = self.compile_and_run_with_dir(&run_dir, per_hart)?;

        let RunArtifacts {
            rvfi,
            disassembly_file,
        } = artifacts;
        let harts = per_hart.len() as u32;

        let exception_count = rvfi.exceptions.len();
        info!("ARA run complete: {} exceptions (by PC)", exception_count);

        let perf = extract_perf_metrics_from_ara_rvfi(
            self.config.isa_base,
            self.config.core_id.clone(),
            harts,
            false,
            &run_dir,
            &disassembly_file,
            &rvfi,
        )?;

        let mut per_hart_out = BTreeMap::new();
        for (hart_id, hart_insts) in per_hart.iter().copied() {
            let tracer = Tracer::new_with_section_labels(
                hart_insts,
                &disassembly_file,
                ara_section_labels_for_hart(harts, hart_id),
                self.config.isa_base,
            )
            .map_err(|e| {
                ARAError::from(ElfError::DumpLoadFailed {
                    path: disassembly_file.clone(),
                    source: e,
                })
            })?;
            let hart_output = build_execution_output(&rvfi, hart_id, &tracer);
            per_hart_out.insert(hart_id, hart_output);
        }

        let mut out = ExecutionOutput::empty(self.config.core_id.clone(), self.config.isa_base);
        out.per_hart = per_hart_out;
        out.perf = perf;
        Ok(out)
    }

    pub fn execute_streaming<T: AsRef<Path>>(
        &self,
        run_folder: T,
        per_hart: PerHartAsm<'_>,
        typed_sequences: Option<PerHartTypedSequences<'_>>,
        emit: &mut dyn FnMut(InstructionEvent) -> bool,
    ) {
        match self.execute_streaming_inner(run_folder.as_ref(), per_hart, typed_sequences, emit) {
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
    ) -> Result<Option<PerfMetrics>, ARAError> {
        if self.config.isa_base != ISABase::Rv32 && self.config.isa_base != ISABase::Rv64 {
            return Err(ARAError::Config(ConfigError::UnsupportedIsaBase {
                core_id: "ara".to_string(),
                isa_base: self.config.isa_base.to_string(),
            }));
        }
        self.ensure_work_dir(run_dir)?;
        self.cleanup_trace_files(run_dir);
        let harts = per_hart.len() as u32;
        let asm_path = run_dir.join("program.S");
        let program = self.build_program(per_hart)?;
        fs::write(&asm_path, &program)?;
        let extensions = self.config.resolved_extensions();
        let build =
            build_elf_with_extensions(&asm_path, &extensions, linker_script()).map_err(|err| {
                error!(
                    "Failed to build ELF for ARA streaming (asm={}, isa={:?}, impl={:?}): {:#}",
                    asm_path.display(),
                    self.config.isa_base,
                    self.config.core_id.clone(),
                    err
                );
                err
            })?;

        let mut tracers = BTreeMap::new();
        for (hart_id, hart_insts) in per_hart.iter().copied() {
            let section_labels = ara_section_labels_for_hart(harts, hart_id);
            let tracer =
                if let Some(sequences) = typed_sequences_for_hart(typed_sequences, hart_id) {
                    Tracer::new_for_sequences_with_section_labels(
                        sequences,
                        &build.disassembly_file,
                        section_labels,
                        self.config.isa_base,
                    )
                } else {
                    Tracer::new_with_section_labels(
                        hart_insts,
                        &build.disassembly_file,
                        section_labels,
                        self.config.isa_base,
                    )
                }
                .map_err(|source| {
                    ARAError::from(ElfError::DumpLoadFailed {
                        path: build.disassembly_file.clone(),
                        source,
                    })
                })?;
            tracers.insert(hart_id, tracer);
        }

        let ara_binary = self.get_ara_binary(harts > 1)?;
        if !is_command_available(&ara_binary) {
            return Err(ARAError::BinaryNotFound {
                path: ara_binary.display().to_string(),
            });
        }
        let coverage_path = if self.config.coverage_mode.is_enabled() {
            let path = run_dir.join("coverage.dat");
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            Some(path)
        } else {
            None
        };
        let elf_path = resolve_wrapper_elf_path(&build.executable_file, run_dir)?;
        let mut cmd = Command::new(&ara_binary);
        cmd.arg(format!(
            "--term-after-cycles={}",
            self.config.term_after_cycles
        ))
        .arg("-l")
        .arg(format!("ram,{},elf", elf_path.display()));
        if let Some(ref cov) = coverage_path {
            cmd.arg(format!("+covfile={}", cov.display()));
        }
        cmd.current_dir(run_dir);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        configure_command_process_group(&mut cmd);
        let command_display = format!("{:?}", cmd);
        info!("Launching ARA streaming: {command_display}");

        let mut child = cmd.spawn().map_err(|source| {
            ARAError::from(ProcessError::SpawnFailed {
                command: command_display.clone(),
                source,
            })
        })?;
        let stdout_reader = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("failed to capture ARA stdout"))?;
        let stderr_reader = child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("failed to capture ARA stderr"))?;
        let stdout_handle = thread::spawn(move || -> Result<Vec<u8>, io::Error> {
            let mut buffer = Vec::new();
            BufReader::new(stdout_reader).read_to_end(&mut buffer)?;
            Ok(buffer)
        });
        let stderr_handle = thread::spawn(move || -> Result<Vec<u8>, io::Error> {
            let mut buffer = Vec::new();
            BufReader::new(stderr_reader).read_to_end(&mut buffer)?;
            Ok(buffer)
        });

        let mut delta = StreamingOutputDelta::new();
        let mut last_trace_size = 0u64;
        let mut last_event_at = Instant::now();
        let mut timed_out = None;
        let status = loop {
            let current_size = ara_trace_size(run_dir, harts);
            if current_size != last_trace_size {
                last_trace_size = current_size;
                if let Some(rvfi) = parse_existing_ara_traces(run_dir, harts)?
                    && rvfi.has_content()
                {
                    let ready = streaming_snapshot_from_ara_trace(&rvfi, &tracers, false);
                    let mut emitted = 0usize;
                    let keep_going = delta.emit_per_hart_delta(&ready, false, &mut |event| {
                        emitted += 1;
                        emit(event)
                    });
                    if emitted > 0 {
                        last_event_at = Instant::now();
                    }
                    if !keep_going {
                        let _ = kill_child_process_group(&mut child);
                        let _ = child.wait();
                        let _ = join_ara_log_thread(stdout_handle, "stdout")?;
                        let _ = join_ara_log_thread(stderr_handle, "stderr")?;
                        return Ok(None);
                    }
                }
            }

            if let Some(exit_status) = child.try_wait()? {
                break exit_status;
            }
            if let Some(limit) = self.config.timeout
                && last_event_at.elapsed() >= limit
            {
                timed_out = Some(limit);
                let _ = kill_child_process_group(&mut child);
                break child.wait()?;
            }
            thread::sleep(Duration::from_millis(25));
        };

        let stdout_bytes = join_ara_log_thread(stdout_handle, "stdout")?;
        let stderr_bytes = join_ara_log_thread(stderr_handle, "stderr")?;
        let stdout_path = run_dir.join("ara_stdout.log");
        let stderr_path = run_dir.join("ara_stderr.log");
        fs::write(&stdout_path, &stdout_bytes).map_err(|err| {
            ARAError::from(ProcessError::FileWriteFailed {
                path: stdout_path.clone(),
                source: err,
            })
        })?;
        fs::write(&stderr_path, &stderr_bytes).map_err(|err| {
            ARAError::from(ProcessError::FileWriteFailed {
                path: stderr_path.clone(),
                source: err,
            })
        })?;

        if let Some(limit) = timed_out {
            return Err(ARAError::from(ProcessError::TimedOut {
                command: command_display,
                timeout: limit,
            }));
        }
        if !status.success() {
            let stdout_text = String::from_utf8_lossy(&stdout_bytes).to_string();
            let stderr_text = String::from_utf8_lossy(&stderr_bytes).to_string();
            let combined_error = if stdout_text.trim().is_empty() {
                stderr_text
            } else if stderr_text.trim().is_empty() {
                stdout_text
            } else {
                format!("{stderr_text}\n--- stdout ---\n{stdout_text}")
            };
            return Err(ARAError::from(ProcessError::ProcessFailed {
                command: command_display,
                stderr: combined_error,
            }));
        }

        let trace_config = TraceConfig::from_run_directory_with_harts(run_dir, harts)?;
        let rvfi = analyze_traces(&trace_config)?;
        let ready = streaming_snapshot_from_ara_trace(&rvfi, &tracers, true);
        if !delta.emit_per_hart_delta(&ready, true, emit) {
            return Ok(None);
        }

        if let Some(cov_path) = coverage_path {
            if !cov_path.exists() {
                return Err(ARAError::from(ProcessError::ProcessFailed {
                    command: command_display.clone(),
                    stderr: format!(
                        "coverage enabled but coverage file is missing: {}",
                        cov_path.display()
                    ),
                }));
            }
            let summary =
                riscv_fuzz_toolchain::coverage::parse_verilator_dat(&cov_path).map_err(|err| {
                    ARAError::from(ProcessError::ProcessFailed {
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

        let stdout = String::from_utf8_lossy(&stdout_bytes).to_string();
        Ok(Some(PerfMetrics {
            global_cycles: parse_global_cycles_from_text(&stdout),
            ..Default::default()
        }))
    }
}

fn ara_trace_path_candidates(run_dir: &Path, hart: u32) -> [PathBuf; 4] {
    [
        run_dir.join(format!("trace_rvfi_hart_{hart:08}.dasm")),
        run_dir.join(format!("trace_rvfi_hart_{hart:02x}.dasm")),
        run_dir.join(format!("trace_rvfi_hart_{hart:x}.dasm")),
        run_dir.join(format!("trace_hart_{hart}.dasm")),
    ]
}

fn ara_trace_size(run_dir: &Path, harts: u32) -> u64 {
    (0..harts.max(1))
        .flat_map(|hart| ara_trace_path_candidates(run_dir, hart))
        .map(|path| {
            fs::metadata(path)
                .map(|metadata| metadata.len())
                .unwrap_or(0)
        })
        .sum()
}

fn objdump_instruction_pc(line: &str) -> Option<u64> {
    let (pc, _) = line.trim_start().split_once(":")?;
    if pc.is_empty() || !pc.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(pc, 16).ok()
}

fn is_objdump_label(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.ends_with(":") && trimmed.contains(" <") && trimmed.contains(">:")
}

fn exit_loop_pc_from_dump(path: &Path) -> Option<u64> {
    let text = fs::read_to_string(path).ok()?;
    let mut in_exit = false;
    let mut last_pc = None;
    for line in text.lines() {
        if line.contains(" <exit>:") {
            in_exit = true;
            last_pc = None;
            continue;
        }
        if in_exit && is_objdump_label(line) {
            break;
        }
        if in_exit && let Some(pc) = objdump_instruction_pc(line) {
            last_pc = Some(pc);
        }
    }
    last_pc
}

fn first_hex_pc_token(line: &str) -> Option<u64> {
    line.split_whitespace()
        .find_map(|token| token.strip_prefix("0x"))
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
}

fn read_trace_tail(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let start = len.saturating_sub(ARA_TRACE_TAIL_BYTES);
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn recent_trace_pc_repeat_count(path: &Path, pc: u64) -> usize {
    let Ok(text) = read_trace_tail(path) else {
        return 0;
    };
    let mut count = 0usize;
    for line in text.lines().rev() {
        match first_hex_pc_token(line) {
            Some(line_pc) if line_pc == pc => count = count.saturating_add(1),
            Some(_) if count > 0 => break,
            _ => {}
        }
    }
    count
}

fn ara_exit_loop_repetition_count(run_dir: &Path, harts: u32, exit_loop_pc: u64) -> usize {
    (0..harts.max(1))
        .flat_map(|hart| ara_trace_path_candidates(run_dir, hart))
        .filter(|path| path.is_file())
        .map(|path| recent_trace_pc_repeat_count(&path, exit_loop_pc))
        .max()
        .unwrap_or(0)
}

fn parse_existing_ara_traces(run_dir: &Path, harts: u32) -> Result<Option<RvfiTrace>, ARAError> {
    let mut traces = Vec::new();
    for hart in 0..harts.max(1) {
        if let Some(path) = ara_trace_path_candidates(run_dir, hart)
            .into_iter()
            .find(|path| path.is_file())
            && let Some((text, _complete_len)) = read_complete_utf8_prefix(&path, false)?
        {
            traces.push(parse_rvfi(Cursor::new(text.into_bytes()), hart)?);
        }
    }
    if traces.is_empty() {
        return Ok(None);
    }
    Ok(Some(merge_rvfi_traces(traces)?))
}

fn streaming_snapshot_from_ara_trace(
    rvfi: &RvfiTrace,
    tracers: &BTreeMap<u32, Tracer>,
    final_snapshot: bool,
) -> BTreeMap<u32, HartExecutionOutput> {
    let per_hart = tracers
        .iter()
        .map(|(hart_id, tracer)| (*hart_id, build_execution_output(rvfi, *hart_id, tracer)))
        .collect::<BTreeMap<_, _>>();
    if final_snapshot {
        per_hart
    } else {
        let stable_before_cycle_by_hart = tracers
            .keys()
            .map(|hart_id| {
                let stable = rvfi
                    .instruction_timings
                    .iter()
                    .filter(|(key, _)| key.hart_id == *hart_id)
                    .filter_map(|(_, entry)| entry.timing.clk_start.or(entry.timing.clk_end))
                    .max();
                (*hart_id, stable)
            })
            .collect::<BTreeMap<_, _>>();
        filter_per_hart_output_for_streaming_by_tracer(
            &per_hart,
            tracers,
            &stable_before_cycle_by_hart,
            false,
        )
    }
}

fn join_ara_log_thread(
    handle: thread::JoinHandle<Result<Vec<u8>, io::Error>>,
    stream_name: &str,
) -> Result<Vec<u8>, ARAError> {
    handle
        .join()
        .map_err(|_| {
            ARAError::from(io::Error::other(format!(
                "ARA {stream_name} reader thread panicked"
            )))
        })?
        .map_err(ARAError::from)
}

fn extract_perf_metrics_from_ara_rvfi(
    _isa_base: ISABase,
    _riscv_impl: String,
    _harts: u32,
    _perf_enabled: bool,
    run_dir: &Path,
    _dump_path: &Path,
    _rvfi: &RvfiTrace,
) -> Result<PerfMetrics, ARAError> {
    let stdout_path = run_dir.join("ara_stdout.log");
    let stdout = fs::read_to_string(&stdout_path).map_err(|err| {
        ARAError::IoError(io::Error::new(
            err.kind(),
            format!(
                "failed to read ARA stdout log {}: {err}",
                stdout_path.display()
            ),
        ))
    })?;

    Ok(PerfMetrics {
        global_cycles: parse_global_cycles_from_text(&stdout),
        ..Default::default()
    })
}

/// Build ExecutionOutput from ARA trace (matching Rocket's approach)
fn traced_instruction_mnemonic(tracer: &Tracer, pc: u64) -> Option<String> {
    let traced = tracer.instruction_for_pc(&pc)?;
    let text = traced
        .source_instruction
        .as_ref()
        .or(traced.parsed_disassembly.as_ref())
        .map(|instruction| instruction.to_string())
        .or_else(|| Some(traced.disassembly.clone()))?;
    Some(
        text.split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase(),
    )
}

fn is_spurious_vector_f0_zero_write(tracer: &Tracer, pc: u64, write: &RegisterWrite) -> bool {
    if write.register != "f0" || write.value != 0 {
        return false;
    }
    let Some(mnemonic) = traced_instruction_mnemonic(tracer, pc) else {
        return false;
    };
    mnemonic.starts_with('v') && mnemonic != "vfmv.f.s"
}

fn semantic_ara_register_name(tracer: &Tracer, pc: u64, write: &RegisterWrite) -> String {
    if let Some(mnemonic) = traced_instruction_mnemonic(tracer, pc)
        && mnemonic == "vfmv.f.s"
        && let Some(index) = write.register.strip_prefix('x')
        && !index.is_empty()
        && index.chars().all(|ch| ch.is_ascii_digit())
    {
        return format!("f{index}");
    }
    write.register.clone()
}

fn build_execution_output(rvfi: &RvfiTrace, hart_id: u32, tracer: &Tracer) -> HartExecutionOutput {
    let observed_pcs = rvfi.observed_pcs_for_hart(hart_id);

    build_raw_hart_execution_output(
        tracer,
        hart_id,
        observed_pcs,
        |pc| {
            let mut regs = rvfi
                .register_writes_for_hart_pc(hart_id, pc)
                .into_iter()
                .flat_map(|writes| writes.iter())
                .filter(|write| !is_spurious_vector_f0_zero_write(tracer, pc, write))
                .map(|write| TraitRegisterValue {
                    name: semantic_ara_register_name(tracer, pc, write),
                    value: write.value,
                })
                .collect::<Vec<_>>();
            regs.sort_by(|a, b| compare_register_names(&a.name, &b.name));
            regs
        },
        |pc| {
            let mut mems = rvfi
                .memory_writes_for_hart_pc(hart_id, pc)
                .into_iter()
                .flat_map(|writes| writes.iter())
                .flat_map(store_bytes_from_write)
                .map(|(addr, value)| TraitMemValue { addr, value })
                .collect::<Vec<_>>();
            mems.sort_by_key(|mem| mem.addr);
            mems
        },
        |pc| {
            rvfi.exception_for_hart_pc(hart_id, pc)
                .map(|event| {
                    vec![if let Some(code) = event.code {
                        ExceptionCause::from_code(code, event.kind.clone())
                    } else {
                        warn!(
                            "ARA: unknown exception kind '{}' at PC 0x{:x}",
                            event.kind, event.pc
                        );
                        ExceptionCause::unknown(event.kind.clone())
                    }]
                })
                .unwrap_or_default()
        },
        |pc| {
            let mut timing_opt: Option<(u64, u64, InstructionTiming)> = None;
            if let Some(writes) = rvfi.register_writes_for_hart_pc(hart_id, pc) {
                for write in writes {
                    if write.timing.is_recorded() {
                        let timing_end = write
                            .timing
                            .clk_end
                            .unwrap_or(write.timing.clk_start.unwrap_or(0));
                        if timing_opt
                            .as_ref()
                            .map(|(existing_pc, existing_end, _)| {
                                pc > *existing_pc
                                    || (pc == *existing_pc && timing_end >= *existing_end)
                            })
                            .unwrap_or(true)
                        {
                            timing_opt = Some((pc, timing_end, write.timing.clone()));
                        }
                    }
                }
            }
            if let Some(writes) = rvfi.memory_writes_for_hart_pc(hart_id, pc) {
                for write in writes {
                    if write.timing.is_recorded() {
                        let timing_end = write
                            .timing
                            .clk_end
                            .unwrap_or(write.timing.clk_start.unwrap_or(0));
                        if timing_opt
                            .as_ref()
                            .map(|(existing_pc, existing_end, _)| {
                                pc > *existing_pc
                                    || (pc == *existing_pc && timing_end >= *existing_end)
                            })
                            .unwrap_or(true)
                        {
                            timing_opt = Some((pc, timing_end, write.timing.clone()));
                        }
                    }
                }
            }
            if let Some(event) = rvfi.exception_for_hart_pc(hart_id, pc)
                && event.timing.is_recorded()
            {
                let timing_end = event
                    .timing
                    .clk_end
                    .unwrap_or(event.timing.clk_start.unwrap_or(0));
                if timing_opt
                    .as_ref()
                    .map(|(existing_pc, existing_end, _)| {
                        pc > *existing_pc || (pc == *existing_pc && timing_end >= *existing_end)
                    })
                    .unwrap_or(true)
                {
                    timing_opt = Some((pc, timing_end, event.timing.clone()));
                }
            }
            if let Some(entry) = rvfi.instruction_timing_for_hart_pc(hart_id, pc)
                && entry.timing.is_recorded()
            {
                let timing_end = entry
                    .timing
                    .clk_end
                    .unwrap_or(entry.timing.clk_start.unwrap_or(0));
                if timing_opt
                    .as_ref()
                    .map(|(existing_pc, existing_end, _)| {
                        pc > *existing_pc || (pc == *existing_pc && timing_end >= *existing_end)
                    })
                    .unwrap_or(true)
                {
                    timing_opt = Some((pc, timing_end, entry.timing.clone()));
                }
            }
            timing_opt.map(|(_, _, timing)| timing).unwrap_or_default()
        },
    )
}

fn ara_section_labels_for_hart(
    harts: u32,
    hart_id: u32,
) -> &'static [(&'static str, &'static str)] {
    match (harts > 1, hart_id) {
        (true, 1) => &[("hart1_testcase_code:", "park:")],
        _ => &[("testcase_code:", "exit:")],
    }
}

/// Extract bytes from ARA memory write using mask
/// The mask indicates which bytes in an 8-byte aligned block are accessed.
/// Each bit in the mask corresponds to a byte offset within the aligned block.
fn store_bytes_from_write(write: &MemoryWrite) -> Vec<(u64, u8)> {
    debug_assert!(
        write.mask & !0xff == 0,
        "ARA parser must reject masks wider than one 8-byte data field"
    );
    let mask = write.mask;
    if mask == 0 {
        return Vec::new();
    }

    if !write.byte_writes.is_empty() {
        let mut bytes = write
            .byte_writes
            .iter()
            .map(|lane| (lane.address, lane.value))
            .collect::<Vec<_>>();
        bytes.sort_by_key(|(addr, _)| *addr);
        debug_assert!(
            bytes.len() == write.width || bytes.len() == mask.count_ones() as usize,
            "mask/width mismatch: mask=0x{mask:x}, width={}, bytes={}",
            write.width,
            bytes.len()
        );
        return bytes;
    }

    // Determine the start of the block covered by the mask. The RVFI log records the effective
    // address (base + offset), so recover the block base by subtracting the lowest asserted bit.
    let lowest_bit = mask.trailing_zeros() as u64;
    let base_addr = write.address.saturating_sub(lowest_bit);

    let mut bytes = Vec::with_capacity(write.width.max(1));
    let mut remaining = mask;
    while remaining != 0 {
        let bit_idx = remaining.trailing_zeros() as u64;
        let addr = base_addr.wrapping_add(bit_idx);
        let shift = (bit_idx * 8) as u32;
        let value = ((write.value >> shift) & 0xFF) as u8;
        bytes.push((addr, value));
        remaining &= remaining - 1;
    }

    debug_assert!(
        bytes.len() == write.width || bytes.len() == mask.count_ones() as usize,
        "mask/width mismatch: mask=0x{mask:x}, width={}, bytes={}",
        write.width,
        bytes.len()
    );

    bytes
}

fn parse_rvfi_file<P: AsRef<Path>>(path: P, hart_id: u32) -> Result<RvfiTrace, ARAError> {
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    parse_rvfi(reader, hart_id)
}

/// Convert memory mask to byte count
/// The mask is a bitmap where each bit represents one byte in an 8-byte aligned block.
/// 0x01 (0b00000001) -> 1 byte at offset 0
/// 0x03 (0b00000011) -> 2 bytes at offset 0-1
/// 0x30 (0b00110000) -> 2 bytes at offset 4-5
/// 0x0f (0b00001111) -> 4 bytes at offset 0-3
/// 0xff (0b11111111) -> 8 bytes at offset 0-7
/// We count the total number of 1 bits to determine the access width.
fn mask_to_width(mask: u64) -> usize {
    if mask == 0 {
        return 0;
    }
    // Count total number of 1 bits in the mask
    mask.count_ones() as usize
}

fn validate_memory_mask(mask: u64, pc: u64, addr: u64) -> Result<(), ARAError> {
    if (mask & !0xff) != 0 {
        return Err(ARAError::from(LogParseError::InvalidLineFormat {
            line: format!(
                "ARA RVFI memory mask 0x{mask:x} selects lanes beyond the 8-byte data field at PC 0x{pc:x} addr=0x{addr:x}"
            ),
        }));
    }
    Ok(())
}

fn infer_store_mask_from_instruction(instr: u32, _addr: u64) -> Option<(u64, usize)> {
    let width = if (instr & 0x3) != 0x3 {
        let quadrant = instr & 0x3;
        let funct3 = (instr >> 13) & 0x7;
        match (quadrant, funct3) {
            (0, 5) | (2, 5) => 8,
            (0, 6) | (2, 6) => 4,
            (0, 7) | (2, 7) => 8,
            _ => return None,
        }
    } else {
        let opcode = instr & 0x7f;
        let funct3 = (instr >> 12) & 0x7;
        match (opcode, funct3) {
            (0x23, 0) => 1,
            (0x23, 1) => 2,
            (0x23, 2) => 4,
            (0x23, 3) => 8,
            (0x27, 1) => 2,
            (0x27, 2) => 4,
            (0x27, 3) => 8,
            (0x2f, 2) => 4,
            (0x2f, 3) => 8,
            _ => return None,
        }
    };
    // When ARA omits an explicit wmask for scalar stores, the logged value is the
    // store payload at the effective address, not an 8-byte-lane-aligned bus word.
    // Keep the inferred mask normalized to byte 0 so unaligned scalar stores do
    // not look like a wider-than-64-bit RVFI memory data field.
    let raw_mask = if width == 8 {
        0xff
    } else {
        (1u64 << width) - 1
    };
    Some((raw_mask, width as usize))
}

fn invalid_ara_trace(message: impl Into<String>) -> ARAError {
    ARAError::from(LogParseError::InvalidLineFormat {
        line: message.into(),
    })
}

fn instruction_timing_bounds(timing: &InstructionTiming) -> Option<(u64, u64)> {
    if !timing.is_recorded() {
        return None;
    }

    if let (Some(start), Some(end)) = (timing.clk_start, timing.clk_end) {
        return (end >= start).then_some((start, end));
    }
    if let (Some(start), Some(span)) = (timing.clk_start, timing.clk_span) {
        return (span > 0)
            .then(|| start.checked_add(span - 1))
            .flatten()
            .map(|end| (start, end));
    }
    if let (Some(end), Some(span)) = (timing.clk_end, timing.clk_span) {
        return (span > 0)
            .then(|| end.checked_sub(span - 1))
            .flatten()
            .map(|start| (start, end));
    }
    if let Some(start) = timing.clk_start {
        return Some((start, start));
    }
    timing.clk_end.map(|end| (end, end))
}

fn timing_preference_key(timing: &InstructionTiming) -> Option<(u64, u64)> {
    instruction_timing_bounds(timing).map(|(start, end)| (end, start))
}

fn register_write_should_replace(existing: &RegisterWrite, candidate: &RegisterWrite) -> bool {
    match (
        timing_preference_key(&existing.timing),
        timing_preference_key(&candidate.timing),
    ) {
        (Some(existing_key), Some(candidate_key)) => candidate_key >= existing_key,
        (None, Some(_)) => true,
        (Some(_), None) => false,
        (None, None) => true,
    }
}

fn memory_write_payload_matches(existing: &MemoryWrite, candidate: &MemoryWrite) -> bool {
    existing.address == candidate.address
        && existing.value == candidate.value
        && existing.width == candidate.width
        && existing.mask == candidate.mask
        && existing.instruction == candidate.instruction
        && existing.byte_writes == candidate.byte_writes
}

fn memory_write_should_replace(existing: &MemoryWrite, candidate: &MemoryWrite) -> bool {
    match (
        timing_preference_key(&existing.timing),
        timing_preference_key(&candidate.timing),
    ) {
        (Some(existing_key), Some(candidate_key)) => candidate_key > existing_key,
        (None, Some(_)) => true,
        (Some(_), None) => false,
        (None, None) => true,
    }
}

fn merge_instruction_timing(existing: &mut InstructionTiming, candidate: &InstructionTiming) {
    if !candidate.is_recorded() {
        return;
    }
    if !existing.is_recorded() {
        *existing = candidate.clone();
        return;
    }

    let start = [existing as &InstructionTiming, candidate]
        .into_iter()
        .filter_map(|timing| instruction_timing_bounds(timing).map(|(start, _)| start))
        .min();
    let end = [existing as &InstructionTiming, candidate]
        .into_iter()
        .filter_map(|timing| instruction_timing_bounds(timing).map(|(_, end)| end))
        .max();

    if let (Some(start), Some(end)) = (start, end) {
        *existing = InstructionTiming::new(Some(start), Some(end), None);
        return;
    }

    let clk_start = [existing.clk_start, candidate.clk_start]
        .into_iter()
        .flatten()
        .min();
    let clk_end = [existing.clk_end, candidate.clk_end]
        .into_iter()
        .flatten()
        .max();

    if let (Some(start), Some(end)) = (clk_start.or(clk_end), clk_end.or(clk_start)) {
        *existing = InstructionTiming::new(Some(start), Some(end), None);
        return;
    }

    let clk_span = [existing.clk_span, candidate.clk_span]
        .into_iter()
        .flatten()
        .max();
    *existing = InstructionTiming::new(None, None, clk_span);
}

fn parse_instruction_timing_from_text(text: &str) -> Result<InstructionTiming, ARAError> {
    static TIMING_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\bclk_(start|end|span)\s*[:=]\s*([0-9][0-9_]*)").unwrap());

    let mut clk_start = None;
    let mut clk_end = None;
    let mut clk_span = None;

    for cap in TIMING_RE.captures_iter(text) {
        let key = cap.get(1).map(|m| m.as_str()).unwrap_or_default();
        let raw = cap.get(2).map(|m| m.as_str()).unwrap_or_default();
        let normalized: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
        let value = normalized.parse::<u64>().map_err(|source| {
            ARAError::from(ParseError::IntParseError {
                value: raw.to_string(),
                source,
            })
        })?;
        match key {
            "start" => clk_start = Some(value),
            "end" => clk_end = Some(value),
            "span" => clk_span = Some(value),
            _ => {}
        }
    }

    Ok(InstructionTiming::new(clk_start, clk_end, clk_span))
}

fn trace_hart_id_from_path(path: &Path) -> Option<u32> {
    let name = path.file_name()?.to_str()?;
    if let Some(raw) = name
        .strip_prefix("trace_rvfi_hart_")
        .and_then(|value| value.strip_suffix(".dasm"))
    {
        return raw.parse().ok();
    }
    if let Some(raw) = name
        .strip_prefix("trace_hart_")
        .and_then(|value| value.strip_suffix(".dasm"))
    {
        return raw.parse().ok();
    }
    None
}

fn parse_rvfi<R: std::io::BufRead>(reader: R, hart_id: u32) -> Result<RvfiTrace, ARAError> {
    // ARA RVFI log patterns (with mandatory masks):
    // Register write (no load): "3 0x<PC> (0x<instr>) x<reg> 0x<value>" or "f<reg>"
    // Load instruction: "3 0x<PC> (0x<instr>) x<reg> 0x<value> mem 0x<addr> 0x<rmask>"
    // Store instruction: "3 0x<PC> (0x<instr>) mem 0x<addr> 0x<value> 0x<wmask> [lane]0x<byte>@0x<addr> ..."
    // Atomic/complex instruction: "3 0x<PC> (0x<instr>) x<reg> 0x<value> mem 0x<addr> 0x<mem_val> 0x<wmask> [lane]0x<byte>@0x<addr> ..."
    // Exception: "<KIND> exception @ 0x<PC> (0x<instr>)"

    // Register write with optional load address and MANDATORY rmask, OR optional store with wmask and segments
    const REG_PATTERN: &str = r"^\s*3\s+0x(?P<pc>[0-9a-fA-F]+)\s+\(0x(?P<instr>[0-9a-fA-F]+)\)\s+(?P<reg_type>[xf])\s*(?P<reg>\d+)\s+0x(?P<val>[0-9a-fA-F]+)(?:\s+mem\s+0x(?P<addr>[0-9a-fA-F]+)\s+0x(?P<mem_val>[0-9a-fA-F]+)(?:\s+0x(?P<wmask>[0-9a-fA-F]+)(?P<segments>(?:\s+\[\s*\d+\]0x[0-9a-fA-F]+@0x[0-9a-fA-F]+)*))?)?";
    static REG_RE: Lazy<Result<Regex, regex::Error>> = Lazy::new(|| Regex::new(REG_PATTERN));

    // Store instruction with MANDATORY wmask (no register write at the beginning)
    const MEM_PATTERN: &str = r"^\s*3\s+0x(?P<pc>[0-9a-fA-F]+)\s+\(0x(?P<instr>[0-9a-fA-F]+)\)\s+mem\s+0x(?P<addr>[0-9a-fA-F]+)\s+0x(?P<val>[0-9a-fA-F]+)(?:\s+0x(?P<wmask>[0-9a-fA-F]+)(?P<segments>(?:\s+\[\s*\d+\]0x[0-9a-fA-F]+@0x[0-9a-fA-F]+)*))?(?:\s+[A-Za-z_][A-Za-z0-9_]*=\S+)*\s*$";
    static MEM_RE: Lazy<Result<Regex, regex::Error>> = Lazy::new(|| Regex::new(MEM_PATTERN));
    const MEM_SEGMENT_PATTERN: &str =
        r"\[\s*(?P<lane>\d+)\]0x(?P<val>[0-9a-fA-F]+)@0x(?P<byte_addr>[0-9a-fA-F]+)";
    static MEM_SEG_RE: Lazy<Result<Regex, regex::Error>> =
        Lazy::new(|| Regex::new(MEM_SEGMENT_PATTERN));

    const EXC_PATTERN: &str =
        r"^(?P<kind>\w+)\s+exception\s+@\s+0x(?P<pc>[0-9a-fA-F]+)\s+\(0x(?P<instr>[0-9a-fA-F]+)\)";
    static EXC_RE: Lazy<Result<Regex, regex::Error>> = Lazy::new(|| Regex::new(EXC_PATTERN));
    const INST_PATTERN: &str = r"^\s*3\s+0x(?P<pc>[0-9a-fA-F]+)\s+\(0x(?P<instr>[0-9a-fA-F]+)\)";
    static INST_RE: Lazy<Result<Regex, regex::Error>> = Lazy::new(|| Regex::new(INST_PATTERN));

    let reg_re = compiled_regex(&REG_RE, REG_PATTERN)?;
    let mem_re = compiled_regex(&MEM_RE, MEM_PATTERN)?;
    let mem_segment_re = compiled_regex(&MEM_SEG_RE, MEM_SEGMENT_PATTERN)?;
    let exc_re = compiled_regex(&EXC_RE, EXC_PATTERN)?;
    let inst_re = compiled_regex(&INST_RE, INST_PATTERN)?;

    // Collapse repeated executions of the same static PC down to the latest
    // side-effect per register/address instead of treating loop iterations as
    // hard conflicts.
    let mut temp_reg_writes: HashMap<u64, HashMap<String, RegisterWrite>> = HashMap::new();
    let mut temp_mem_writes: HashMap<u64, HashMap<u64, MemoryWrite>> = HashMap::new();
    let mut exceptions: HashMap<u64, ExceptionEvent> = HashMap::new();
    let mut instruction_timings: HashMap<u64, InstructionTrace> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        let timing = parse_instruction_timing_from_text(trimmed)?;

        // Try to match exception first
        if let Some(cap) = exc_re.captures(trimmed) {
            let kind = cap
                .name("kind")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "kind".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str()
                .to_string();
            let pc_str = cap
                .name("pc")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "pc".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let pc = u64::from_str_radix(pc_str, 16).map_err(|e| {
                ARAError::from(ParseError::HexParseError {
                    value: pc_str.to_string(),
                    source: e,
                })
            })?;

            if let Some(previous) = exceptions.get(&pc) {
                if previous.kind != kind {
                    return Err(invalid_ara_trace(format!(
                        "conflicting ARA exception at pc=0x{pc:x}: {} vs {kind}",
                        previous.kind
                    )));
                }
                return Err(invalid_ara_trace(format!(
                    "duplicate ARA exception at pc=0x{pc:x}: {kind}"
                )));
            }
            exceptions.insert(
                pc,
                ExceptionEvent {
                    source: ExceptionSource::Rvfi,
                    pc,
                    code: None,
                    kind,
                    timing,
                },
            );
            continue;
        }

        // Try to match register write
        if let Some(cap) = reg_re.captures(trimmed) {
            let pc_str = cap
                .name("pc")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "pc".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let pc = u64::from_str_radix(pc_str, 16).map_err(|e| {
                ARAError::from(ParseError::HexParseError {
                    value: pc_str.to_string(),
                    source: e,
                })
            })?;
            let instr_str = cap
                .name("instr")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "instr".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let instr = u32::from_str_radix(instr_str, 16).map_err(|e| {
                ARAError::from(ParseError::HexParseError {
                    value: instr_str.to_string(),
                    source: e,
                })
            })?;
            let reg_type = cap
                .name("reg_type")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "reg_type".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let reg_num_str = cap
                .name("reg")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "reg".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let reg_num: usize = reg_num_str.parse().map_err(|e| {
                ARAError::from(ParseError::IntParseError {
                    value: reg_num_str.to_string(),
                    source: e,
                })
            })?;
            let val_str = cap
                .name("val")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "val".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let value = u64::from_str_radix(val_str, 16).map_err(|e| {
                ARAError::from(ParseError::HexParseError {
                    value: val_str.to_string(),
                    source: e,
                })
            })?;

            if reg_num < 32 {
                let register_name = format!("{}{}", reg_type, reg_num);
                let writes = temp_reg_writes.entry(pc).or_default();
                let candidate = RegisterWrite {
                    index: reg_num as u8,
                    pc,
                    register: register_name.clone(),
                    value,
                    instruction: Some(instr),
                    timing: timing.clone(),
                };
                match writes.entry(register_name) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        if register_write_should_replace(entry.get(), &candidate) {
                            entry.insert(candidate);
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(candidate);
                    }
                }
            }

            // Check if this line also contains a memory write (atomic/complex instruction)
            // Pattern: x<reg> 0x<val> mem 0x<addr> 0x<mem_val> 0x<wmask> [segments]
            if let Some(wmask_str) = cap.name("wmask") {
                let addr_str = cap
                    .name("addr")
                    .ok_or_else(|| {
                        ARAError::from(LogParseError::MissingCaptureGroup {
                            group: "addr".to_string(),
                            line: trimmed.to_string(),
                        })
                    })?
                    .as_str();
                let addr = u64::from_str_radix(addr_str, 16).map_err(|e| {
                    ARAError::from(ParseError::HexParseError {
                        value: addr_str.to_string(),
                        source: e,
                    })
                })?;

                let mem_val_str = cap
                    .name("mem_val")
                    .ok_or_else(|| {
                        ARAError::from(LogParseError::MissingCaptureGroup {
                            group: "mem_val".to_string(),
                            line: trimmed.to_string(),
                        })
                    })?
                    .as_str();
                let mem_value = u64::from_str_radix(mem_val_str, 16).map_err(|e| {
                    ARAError::from(ParseError::HexParseError {
                        value: mem_val_str.to_string(),
                        source: e,
                    })
                })?;

                let wmask = u64::from_str_radix(wmask_str.as_str(), 16).map_err(|e| {
                    ARAError::from(ParseError::HexParseError {
                        value: wmask_str.as_str().to_string(),
                        source: e,
                    })
                })?;
                validate_memory_mask(wmask, pc, addr)?;

                // Parse optional lane/value annotations
                let segments_str = cap.name("segments").map(|m| m.as_str()).unwrap_or("");
                let mut byte_writes: Vec<MemoryWriteByte> = Vec::new();
                let mut remaining = segments_str.trim();
                while !remaining.is_empty() {
                    let captures = mem_segment_re.captures(remaining).ok_or_else(|| {
                        ARAError::from(LogParseError::InvalidLineFormat {
                            line: trimmed.to_string(),
                        })
                    })?;
                    let matched = captures.get(0).ok_or_else(|| {
                        ARAError::from(LogParseError::InvalidLineFormat {
                            line: trimmed.to_string(),
                        })
                    })?;
                    if matched.start() != 0 {
                        let prefix = &remaining[..matched.start()];
                        if !prefix.trim().is_empty() {
                            return Err(ARAError::from(LogParseError::InvalidLineFormat {
                                line: trimmed.to_string(),
                            }));
                        }
                    }

                    let lane_str = captures
                        .name("lane")
                        .ok_or_else(|| {
                            ARAError::from(LogParseError::MissingCaptureGroup {
                                group: "lane".to_string(),
                                line: trimmed.to_string(),
                            })
                        })?
                        .as_str();
                    let lane = lane_str.parse::<u8>().map_err(|e| {
                        ARAError::from(ParseError::IntParseError {
                            value: lane_str.to_string(),
                            source: e,
                        })
                    })?;

                    let byte_str = captures
                        .name("val")
                        .ok_or_else(|| {
                            ARAError::from(LogParseError::MissingCaptureGroup {
                                group: "val".to_string(),
                                line: trimmed.to_string(),
                            })
                        })?
                        .as_str();
                    let byte_value = u8::from_str_radix(byte_str, 16).map_err(|e| {
                        ARAError::from(ParseError::HexParseError {
                            value: format!("0x{}", byte_str),
                            source: e,
                        })
                    })?;

                    let byte_addr_str = captures
                        .name("byte_addr")
                        .ok_or_else(|| {
                            ARAError::from(LogParseError::MissingCaptureGroup {
                                group: "byte_addr".to_string(),
                                line: trimmed.to_string(),
                            })
                        })?
                        .as_str();
                    let byte_addr = u64::from_str_radix(byte_addr_str, 16).map_err(|e| {
                        ARAError::from(ParseError::HexParseError {
                            value: byte_addr_str.to_string(),
                            source: e,
                        })
                    })?;

                    byte_writes.push(MemoryWriteByte {
                        lane,
                        address: byte_addr,
                        value: byte_value,
                    });

                    remaining = remaining[matched.end()..].trim_start();
                }

                byte_writes.sort_by_key(|entry| entry.lane);

                let mask_width = mask_to_width(wmask);
                let width = if !byte_writes.is_empty() {
                    if byte_writes.len() != mask_width {
                        return Err(ARAError::LogParseError(LogParseError::InvalidLineFormat {
                            line: format!(
                                "ARA RVFI mem-write width mismatch at PC 0x{pc:x}: mask_width={} byte_writes={} (addr=0x{addr:x}, mask=0x{wmask:x})",
                                mask_width,
                                byte_writes.len()
                            ),
                        }));
                    }
                    byte_writes.len()
                } else {
                    mask_width
                };

                let writes = temp_mem_writes.entry(pc).or_default();
                let candidate = MemoryWrite {
                    pc,
                    address: addr,
                    value: mem_value,
                    width,
                    mask: wmask,
                    instruction: Some(instr),
                    byte_writes,
                    timing: timing.clone(),
                };
                match writes.entry(addr) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        let existing = entry.get();
                        if !memory_write_payload_matches(existing, &candidate)
                            && !memory_write_should_replace(existing, &candidate)
                        {
                            return Err(invalid_ara_trace(format!(
                                "conflicting ARA memory write at pc=0x{pc:x} addr=0x{addr:x}"
                            )));
                        }
                        if memory_write_should_replace(existing, &candidate) {
                            entry.insert(candidate);
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(candidate);
                    }
                }
            }

            // If there's a "mem" field but no wmask, it's a load instruction
            // We don't record loads as memory writes
            continue;
        }

        // Try to match memory write (store)
        if let Some(cap) = mem_re.captures(trimmed) {
            let pc_str = cap
                .name("pc")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "pc".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let pc = u64::from_str_radix(pc_str, 16).map_err(|e| {
                ARAError::from(ParseError::HexParseError {
                    value: pc_str.to_string(),
                    source: e,
                })
            })?;
            let instr_str = cap
                .name("instr")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "instr".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let instr = u32::from_str_radix(instr_str, 16).map_err(|e| {
                ARAError::from(ParseError::HexParseError {
                    value: instr_str.to_string(),
                    source: e,
                })
            })?;
            let addr_str = cap
                .name("addr")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "addr".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let addr = u64::from_str_radix(addr_str, 16).map_err(|e| {
                ARAError::from(ParseError::HexParseError {
                    value: addr_str.to_string(),
                    source: e,
                })
            })?;
            let value_str = cap
                .name("val")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "val".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let value = u64::from_str_radix(value_str, 16).map_err(|e| {
                ARAError::from(ParseError::HexParseError {
                    value: value_str.to_string(),
                    source: e,
                })
            })?;

            let (wmask, inferred_width) = if let Some(wmask_match) = cap.name("wmask") {
                let parsed = u64::from_str_radix(wmask_match.as_str(), 16).map_err(|e| {
                    ARAError::from(ParseError::HexParseError {
                        value: wmask_match.as_str().to_string(),
                        source: e,
                    })
                })?;
                (parsed, None)
            } else {
                let Some((mask, width)) = infer_store_mask_from_instruction(instr, addr) else {
                    return Err(ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "wmask".to_string(),
                        line: trimmed.to_string(),
                    }));
                };
                (mask, Some(width))
            };
            validate_memory_mask(wmask, pc, addr)?;

            // Parse optional lane/value annotations (e.g. "[0]0x12@0x1000")
            let segments_str = cap.name("segments").map(|m| m.as_str()).unwrap_or("");
            let mut byte_writes: Vec<MemoryWriteByte> = Vec::new();
            let mut remaining = segments_str.trim();
            while !remaining.is_empty() {
                let captures = mem_segment_re.captures(remaining).ok_or_else(|| {
                    ARAError::from(LogParseError::InvalidLineFormat {
                        line: trimmed.to_string(),
                    })
                })?;
                let matched = captures.get(0).ok_or_else(|| {
                    ARAError::from(LogParseError::InvalidLineFormat {
                        line: trimmed.to_string(),
                    })
                })?;
                if matched.start() != 0 {
                    let prefix = &remaining[..matched.start()];
                    if !prefix.trim().is_empty() {
                        return Err(ARAError::from(LogParseError::InvalidLineFormat {
                            line: trimmed.to_string(),
                        }));
                    }
                }

                let lane_str = captures
                    .name("lane")
                    .ok_or_else(|| {
                        ARAError::from(LogParseError::MissingCaptureGroup {
                            group: "lane".to_string(),
                            line: trimmed.to_string(),
                        })
                    })?
                    .as_str();
                let lane = lane_str.parse::<u8>().map_err(|e| {
                    ARAError::from(ParseError::IntParseError {
                        value: lane_str.to_string(),
                        source: e,
                    })
                })?;

                let byte_str = captures
                    .name("val")
                    .ok_or_else(|| {
                        ARAError::from(LogParseError::MissingCaptureGroup {
                            group: "val".to_string(),
                            line: trimmed.to_string(),
                        })
                    })?
                    .as_str();
                let byte_value = u8::from_str_radix(byte_str, 16).map_err(|e| {
                    ARAError::from(ParseError::HexParseError {
                        value: format!("0x{}", byte_str),
                        source: e,
                    })
                })?;

                let byte_addr_str = captures
                    .name("byte_addr")
                    .ok_or_else(|| {
                        ARAError::from(LogParseError::MissingCaptureGroup {
                            group: "byte_addr".to_string(),
                            line: trimmed.to_string(),
                        })
                    })?
                    .as_str();
                let byte_addr = u64::from_str_radix(byte_addr_str, 16).map_err(|e| {
                    ARAError::from(ParseError::HexParseError {
                        value: byte_addr_str.to_string(),
                        source: e,
                    })
                })?;

                byte_writes.push(MemoryWriteByte {
                    lane,
                    address: byte_addr,
                    value: byte_value,
                });

                remaining = remaining[matched.end()..].trim_start();
            }

            byte_writes.sort_by_key(|entry| entry.lane);

            let mask_width = mask_to_width(wmask);
            let width = if !byte_writes.is_empty() {
                if byte_writes.len() != mask_width {
                    return Err(ARAError::LogParseError(LogParseError::InvalidLineFormat {
                        line: format!(
                            "ARA RVFI mem-write width mismatch at PC 0x{pc:x}: mask_width={} byte_writes={} (addr=0x{addr:x}, mask=0x{wmask:x})",
                            mask_width,
                            byte_writes.len()
                        ),
                    }));
                }
                byte_writes.len()
            } else {
                inferred_width.unwrap_or(mask_width)
            };

            let writes = temp_mem_writes.entry(pc).or_default();
            let candidate = MemoryWrite {
                pc,
                address: addr,
                value,
                width,
                mask: wmask,
                instruction: Some(instr),
                byte_writes,
                timing,
            };
            match writes.entry(addr) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get();
                    if !memory_write_payload_matches(existing, &candidate)
                        && !memory_write_should_replace(existing, &candidate)
                    {
                        return Err(invalid_ara_trace(format!(
                            "conflicting ARA memory write at pc=0x{pc:x} addr=0x{addr:x}"
                        )));
                    }
                    if memory_write_should_replace(existing, &candidate) {
                        entry.insert(candidate);
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(candidate);
                }
            }
            continue;
        }

        if timing.is_recorded()
            && let Some(cap) = inst_re.captures(trimmed)
        {
            let pc_str = cap
                .name("pc")
                .ok_or_else(|| {
                    ARAError::from(LogParseError::MissingCaptureGroup {
                        group: "pc".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let pc = u64::from_str_radix(pc_str, 16).map_err(|e| {
                ARAError::from(ParseError::HexParseError {
                    value: pc_str.to_string(),
                    source: e,
                })
            })?;
            if let Some(existing) = instruction_timings.get_mut(&pc) {
                merge_instruction_timing(&mut existing.timing, &timing);
            } else {
                instruction_timings.insert(
                    pc,
                    InstructionTrace {
                        pc,
                        timing: timing.clone(),
                    },
                );
            }
        }
    }

    // Convert intermediate HashMaps to final Vec format
    let register_writes: HashMap<TraceKey, Vec<RegisterWrite>> = temp_reg_writes
        .into_iter()
        .map(|(pc, reg_map)| (TraceKey::new(hart_id, pc), reg_map.into_values().collect()))
        .collect();

    let memory_writes: HashMap<TraceKey, Vec<MemoryWrite>> = temp_mem_writes
        .into_iter()
        .map(|(pc, mem_map)| (TraceKey::new(hart_id, pc), mem_map.into_values().collect()))
        .collect();

    let exceptions: HashMap<TraceKey, ExceptionEvent> = exceptions
        .into_iter()
        .map(|(pc, event)| (TraceKey::new(hart_id, pc), event))
        .collect();

    let instruction_timings: HashMap<TraceKey, InstructionTrace> = instruction_timings
        .into_iter()
        .map(|(pc, timing)| (TraceKey::new(hart_id, pc), timing))
        .collect();

    let total_reg_writes: usize = register_writes.values().map(|v| v.len()).sum();
    let total_mem_writes: usize = memory_writes.values().map(|v| v.len()).sum();
    let total_exceptions: usize = exceptions.len();
    let total_timing_records = instruction_timings.len();

    debug!(
        "ARA RVFI trace parsed: {} PCs with register writes (total {}), {} PCs with memory writes (total {}), {} PCs with exceptions, {} PCs with timing-only entries (total {})",
        register_writes.len(),
        total_reg_writes,
        memory_writes.len(),
        total_mem_writes,
        total_exceptions,
        instruction_timings.len(),
        total_timing_records
    );

    Ok(RvfiTrace {
        register_writes,
        memory_writes,
        exceptions,
        instruction_timings,
    })
}

fn register_sort_key(name: &str) -> Option<(char, &str)> {
    let mut chars = name.chars();
    let prefix = chars.next()?;
    let number = chars.as_str();
    if number.is_empty() {
        return None;
    }
    if !number.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let normalized = number.trim_start_matches('0');
    let normalized = if normalized.is_empty() {
        "0"
    } else {
        normalized
    };
    Some((prefix, normalized))
}

fn compare_register_names(left: &str, right: &str) -> Ordering {
    match (register_sort_key(left), register_sort_key(right)) {
        (Some((left_prefix, left_num)), Some((right_prefix, right_num))) => left_prefix
            .cmp(&right_prefix)
            .then_with(|| left_num.len().cmp(&right_num.len()))
            .then_with(|| left_num.cmp(right_num)),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left.cmp(right),
    }
}

fn compiled_regex<'a>(
    lazy: &'a Lazy<Result<Regex, regex::Error>>,
    pattern: &'static str,
) -> Result<&'a Regex, ARAError> {
    lazy.as_ref().map_err(|err| {
        ARAError::from(LogParseError::RegexCompilationFailed {
            pattern: pattern.to_string(),
            source: err.clone(),
        })
    })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Mutex;
    use std::{env, fs};

    use tempfile::tempdir;

    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => unsafe { env::set_var(self.key, v) },
                None => unsafe { env::remove_var(self.key) },
            }
        }
    }

    fn set_env_var(key: &'static str, value: &str) -> EnvVarGuard {
        let prev = env::var(key).ok();
        unsafe { env::set_var(key, value) };
        EnvVarGuard { key, prev }
    }

    #[test]
    fn exit_loop_pc_is_extracted_from_objdump_exit_section() {
        let tmp = tempdir().expect("create tempdir");
        let dump = tmp.path().join("program.dump");
        fs::write(
            &dump,
            "0000000080000100 <exit>:\n    80000100:\t00d0029b\taddiw x5,x0,13\n    80000104:\ta001\tc.j 80000104 <exit+0x4>\n0000000080000106 <trap_handler>:\n    80000106:\t341022f3\tcsrrs x5,mepc,x0\n",
        )
        .expect("write dump");

        assert_eq!(exit_loop_pc_from_dump(&dump), Some(0x8000_0104));
    }

    #[test]
    fn recent_trace_pc_repeat_count_tracks_exit_loop_tail() {
        let tmp = tempdir().expect("create tempdir");
        let trace = tmp.path().join("trace_rvfi_hart_00.dasm");
        let mut text = String::from(
            "core   0: 3 0x00000000800000fe (0x0001)\n3 0x0000000080000100 (0x0002) x5 0x0\n",
        );
        for _ in 0..40 {
            text.push_str("core   0: 3 0x0000000080000104 (0xa001)\n");
        }
        fs::write(&trace, text).expect("write trace");

        assert_eq!(recent_trace_pc_repeat_count(&trace, 0x8000_0104), 40);
        assert_eq!(
            ara_exit_loop_repetition_count(tmp.path(), 1, 0x8000_0104),
            40
        );
    }

    #[test]
    fn selects_ara_binary_by_coverage_mode() {
        let _lock = ENV_LOCK.lock().expect("lock env mutex");

        let tmp = tempdir().expect("create tempdir");
        let base = tmp.path().join("ara_rv64");
        let cov = tmp.path().join("ara_rv64_cov");
        let cov_light = tmp.path().join("ara_rv64_cov_light");
        std::fs::write(&base, b"").expect("create base stub");
        std::fs::write(&cov, b"").expect("create cov stub");
        std::fs::write(&cov_light, b"").expect("create cov_light stub");

        let _g_base = set_env_var(ARA_RV64_BINARY_ENV, base.to_str().expect("base path utf-8"));
        let _g_cov = set_env_var(
            ARA_RV64_COV_BINARY_ENV,
            cov.to_str().expect("cov path utf-8"),
        );
        let _g_light = set_env_var(
            ARA_RV64_COV_LIGHT_BINARY_ENV,
            cov_light.to_str().expect("light path utf-8"),
        );

        let mut cfg =
            ExecutorConfig::new(tmp.path().to_path_buf(), ISABase::Rv64, "ara".to_string())
                .unwrap();

        cfg.coverage_mode = coverage::CoverageMode::Off;
        assert_eq!(
            AraExecutor::new(cfg.clone()).get_ara_binary(false).unwrap(),
            base
        );

        cfg.coverage_mode = coverage::CoverageMode::Full;
        assert_eq!(
            AraExecutor::new(cfg.clone()).get_ara_binary(false).unwrap(),
            cov
        );

        cfg.coverage_mode = coverage::CoverageMode::Light;
        assert_eq!(
            AraExecutor::new(cfg).get_ara_binary(false).unwrap(),
            cov_light
        );
    }

    #[test]
    fn trace_config_collects_all_expected_harts() {
        let tmp = tempdir().expect("create tempdir");
        let h0 = tmp.path().join("trace_rvfi_hart_00000000.dasm");
        let h1 = tmp.path().join("trace_rvfi_hart_00000001.dasm");
        std::fs::write(&h0, b"3 0x0 (0x00000013) x1 0x1\n").expect("write hart0 trace");
        std::fs::write(&h1, b"3 0x4 (0x00000013) x2 0x2\n").expect("write hart1 trace");

        let cfg = TraceConfig::from_run_directory_with_harts(tmp.path(), 2)
            .expect("collect multicore traces");
        assert_eq!(cfg.rvfi_paths.len(), 2);
        assert_eq!(cfg.rvfi_paths[0], h0);
        assert_eq!(cfg.rvfi_paths[1], h1);
    }

    #[test]
    fn trace_config_accepts_short_hex_hart_trace_name() {
        let tmp = tempdir().expect("create tempdir");
        let h0 = tmp.path().join("trace_rvfi_hart_00.dasm");
        std::fs::write(&h0, b"3 0x0 (0x00000013) x1 0x1\n").expect("write hart0 trace");

        let cfg = TraceConfig::from_run_directory_with_harts(tmp.path(), 1)
            .expect("short hex ARA trace name should be accepted");
        assert_eq!(cfg.rvfi_paths, vec![h0]);
    }

    #[test]
    fn parse_rvfi_infers_mask_for_ara_store_without_wmask() {
        let rvfi = parse_rvfi(
            Cursor::new(
                "3 0x00000000800000bc (0x00a4b023) mem 0x0000000080010040 0x0000000000000080\n",
            ),
            0,
        )
        .expect("parse rvfi");

        let writes = rvfi
            .memory_writes_for_hart_pc(0, 0x0000_0000_8000_00bc)
            .expect("store should be recorded");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].address, 0x0000_0000_8001_0040);
        assert_eq!(writes[0].width, 8);
        assert_eq!(writes[0].mask, 0xff);

        let rvfi = parse_rvfi(
            Cursor::new(
                "3 0x00000000800000c0 (0xe088) mem 0x0000000080010048 0x0000000000000081\n",
            ),
            0,
        )
        .expect("parse compressed store rvfi");
        let writes = rvfi
            .memory_writes_for_hart_pc(0, 0x0000_0000_8000_00c0)
            .expect("compressed store should be recorded");
        assert_eq!(writes[0].width, 8);
        assert_eq!(writes[0].mask, 0xff);
    }

    #[test]
    fn parse_rvfi_normalizes_inferred_mask_for_unaligned_ara_store() {
        let rvfi = parse_rvfi(
            Cursor::new(
                "3 0x000000008000097e (0xe036) mem 0x00000000800effbf 0x0000000000007fff\n",
            ),
            0,
        )
        .expect("parse unaligned ARA scalar store without explicit wmask");

        let writes = rvfi
            .memory_writes_for_hart_pc(0, 0x0000_0000_8000_097e)
            .expect("store should be recorded");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].address, 0x0000_0000_800e_ffbf);
        assert_eq!(writes[0].width, 8);
        assert_eq!(writes[0].mask, 0xff);
    }

    #[test]
    fn store_bytes_from_write_wraps_address_space_instead_of_panicking() {
        let write = MemoryWrite {
            pc: 0x8000_0000,
            address: u64::MAX,
            value: 0x1122,
            width: 2,
            mask: 0x03,
            instruction: None,
            byte_writes: Vec::new(),
            timing: InstructionTiming::default(),
        };

        let bytes = store_bytes_from_write(&write);
        assert_eq!(bytes, vec![(u64::MAX, 0x22), (0, 0x11)]);
    }

    #[test]
    fn trace_config_errors_when_hart_trace_missing() {
        let tmp = tempdir().expect("create tempdir");
        let h0 = tmp.path().join("trace_rvfi_hart_00000000.dasm");
        std::fs::write(&h0, b"3 0x0 (0x00000013) x1 0x1\n").expect("write hart0 trace");

        let err = TraceConfig::from_run_directory_with_harts(tmp.path(), 2)
            .expect_err("missing hart1 trace should fail");
        let text = format!("{err}");
        assert!(text.contains("hart1"), "unexpected error text: {text}");
    }

    #[test]
    fn parse_rvfi_preserves_exception_text_and_leaves_cause_unknown() {
        let rvfi = parse_rvfi(
            Cursor::new("LD_ADDR_MISALIGNED exception @ 0x0000000080000000 (0x00000073)\n"),
            0,
        )
        .expect("parse rvfi");

        let event = rvfi
            .exception_for_hart_pc(0, 0x0000_0000_8000_0000)
            .expect("exception should be recorded");
        assert_eq!(event.kind, "LD_ADDR_MISALIGNED");
        assert_eq!(event.code, None);
    }

    #[test]
    fn analyze_traces_keeps_same_pc_separate_per_hart() {
        let tmp = tempdir().expect("create tempdir");
        let h0 = tmp.path().join("trace_rvfi_hart_00000000.dasm");
        let h1 = tmp.path().join("trace_rvfi_hart_00000001.dasm");
        std::fs::write(
            &h0,
            b"3 0x0000000080000000 (0x00000013) x1 0x0000000000000001\n",
        )
        .expect("write hart0 trace");
        std::fs::write(
            &h1,
            b"3 0x0000000080000000 (0x00000013) x2 0x0000000000000002\n",
        )
        .expect("write hart1 trace");

        let cfg = TraceConfig::from_run_directory_with_harts(tmp.path(), 2)
            .expect("collect multicore traces");
        let merged = analyze_traces(&cfg).expect("analyze merged traces");

        assert!(
            merged
                .register_writes
                .contains_key(&TraceKey::new(0, 0x0000_0000_8000_0000))
        );
        assert!(
            merged
                .register_writes
                .contains_key(&TraceKey::new(1, 0x0000_0000_8000_0000))
        );
        let total_reg_writes: usize = merged.register_writes.values().map(|w| w.len()).sum();
        assert_eq!(total_reg_writes, 2);
    }

    #[test]
    fn perf_cycles_only_use_stdout() {
        let tmp = tempdir().expect("create tempdir");
        fs::write(tmp.path().join("ara_stdout.log"), "").expect("write stdout log");
        fs::write(tmp.path().join("ara_stderr.log"), "Cycles: 99").expect("write stderr log");

        let perf = extract_perf_metrics_from_ara_rvfi(
            ISABase::Rv64,
            "ara".to_string(),
            1,
            false,
            tmp.path(),
            Path::new("/tmp"),
            &RvfiTrace::default(),
        )
        .expect("perf extraction should read stdout log");
        assert_eq!(perf.global_cycles, None);

        fs::write(tmp.path().join("ara_stdout.log"), "Cycles: 77").expect("rewrite stdout log");
        let perf = extract_perf_metrics_from_ara_rvfi(
            ISABase::Rv64,
            "ara".to_string(),
            1,
            false,
            tmp.path(),
            Path::new("/tmp"),
            &RvfiTrace::default(),
        )
        .expect("perf extraction should read stdout log");

        assert_eq!(perf.global_cycles, Some(77));
    }

    #[test]
    fn perf_requires_ara_stdout_log() {
        let tmp = tempdir().expect("create tempdir");

        let err = extract_perf_metrics_from_ara_rvfi(
            ISABase::Rv64,
            "ara".to_string(),
            1,
            false,
            tmp.path(),
            Path::new("/tmp"),
            &RvfiTrace::default(),
        )
        .expect_err("missing stdout log should fail");

        assert!(format!("{err}").contains("ara_stdout.log"));
    }

    #[test]
    fn build_execution_output_keeps_pure_store_memory_writes() {
        let instructions = vec![
            "li x9, 0x8ffffff0".to_string(),
            "li x10, 0x80".to_string(),
            "sd x10, 0(x9)".to_string(),
            "li x8, 0".to_string(),
        ];
        let dump = r#"
    li x9, 0x8ffffff0
    800000b0:	0090049b          	addiw	x9,x0,9
    800000b4:	04f2                	c.slli	x9,0x1c
    800000b6:	14c1                	c.addi	x9,-16
    li x10, 0x80
    800000b8:	08000513          	addi	x10,x0,128
    sd x10, 0(x9)
    800000bc:	e088                	c.sd	x10,0(x9)
    li x8, 0
    800000be:	4401                	c.li	x8,0
"#;
        let rvfi_text = r#"
3 0x00000000800000b0 (0x0090049b) x 9 0x0000000000000009
3 0x00000000800000b4 (0x04f2) x 9 0x0000000090000000
3 0x00000000800000b6 (0x14c1) x 9 0x000000008ffffff0
3 0x00000000800000b8 (0x08000513) x10 0x0000000000000080
3 0x00000000800000bc (0xe088) mem 0x000000008ffffff0 0x0000000000000080 0xff [0]0x80@0x000000008ffffff0 [1]0x00@0x000000008ffffff1 [2]0x00@0x000000008ffffff2 [3]0x00@0x000000008ffffff3 [4]0x00@0x000000008ffffff4 [5]0x00@0x000000008ffffff5 [6]0x00@0x000000008ffffff6 [7]0x00@0x000000008ffffff7
3 0x00000000800000be (0x4401) x 8 0x0000000000000000
"#;

        let tmp = tempdir().expect("create tempdir");
        let dump_path = tmp.path().join("program.dump");
        fs::write(&dump_path, dump).expect("write dump");

        let tracer = Tracer::new(&instructions, &dump_path, ISABase::Rv64).expect("parse dump");
        let rvfi = parse_rvfi(Cursor::new(rvfi_text), 0).expect("parse rvfi");

        let output = build_execution_output(&rvfi, 0, &tracer);

        let sd_inst = output
            .instructions
            .iter()
            .find(|instruction| {
                instruction
                    .trace_inst_id
                    .is_some_and(|trace_inst_id| trace_inst_id.local_index == 2)
            })
            .expect("sd instruction not found in output");

        let store_bytes = sd_inst
            .memory_writes
            .iter()
            .map(|entry| (entry.addr, entry.value))
            .collect::<Vec<_>>();
        assert_eq!(
            store_bytes,
            vec![
                (0x0000_0000_8fff_fff0, 0x80),
                (0x0000_0000_8fff_fff1, 0x00),
                (0x0000_0000_8fff_fff2, 0x00),
                (0x0000_0000_8fff_fff3, 0x00),
                (0x0000_0000_8fff_fff4, 0x00),
                (0x0000_0000_8fff_fff5, 0x00),
                (0x0000_0000_8fff_fff6, 0x00),
                (0x0000_0000_8fff_fff7, 0x00),
            ]
        );
    }

    #[test]
    fn build_execution_output_maps_vfmv_f_s_x_write_to_float_register() {
        let instructions = vec!["vfmv.f.s f16, v18".to_string()];
        let dump = r#"
    vfmv.f.s f16, v18
    80000000:\t43201857          \tvfmv.f.s\tf16,v18
"#;
        let rvfi_text = r#"
3 0x0000000080000000 (0x43201857) x16 0x052d326d2a577976
"#;

        let tmp = tempdir().expect("create tempdir");
        let dump_path = tmp.path().join("program.dump");
        fs::write(&dump_path, dump).expect("write dump");

        let tracer = Tracer::new(&instructions, &dump_path, ISABase::Rv64).expect("parse dump");
        let rvfi = parse_rvfi(Cursor::new(rvfi_text), 0).expect("parse rvfi");

        let output = build_execution_output(&rvfi, 0, &tracer);
        let inst = output
            .instructions
            .iter()
            .find(|instruction| {
                instruction
                    .trace_inst_id
                    .is_some_and(|trace_inst_id| trace_inst_id.local_index == 0)
            })
            .expect("vfmv.f.s instruction not found in output");

        assert_eq!(inst.register_writes.len(), 1);
        assert_eq!(inst.register_writes[0].name, "f16");
        assert_eq!(inst.register_writes[0].value, 0x052d_326d_2a57_7976);
    }

    #[test]
    fn build_execution_output_filters_side_effects_to_target_hart() {
        let instructions = vec!["addi x1, x0, 1".to_string()];
        let dump = r#"
    addi x1, x0, 1
    80000000:	00100093          	addi	x1,x0,1
"#;
        let tmp = tempdir().expect("create tempdir");
        let h0 = tmp.path().join("trace_rvfi_hart_00000000.dasm");
        let h1 = tmp.path().join("trace_rvfi_hart_00000001.dasm");
        fs::write(
            &h0,
            "3 0x0000000080000000 (0x00100093) x1 0x0000000000000001\n",
        )
        .expect("write hart0 trace");
        fs::write(
            &h1,
            "\
3 0x0000000080000000 (0x00100093) x2 0x0000000000000002 mem 0x0000000000001000 0x00000000000000aa 0x01 [0]0xaa@0x0000000000001000\n\
illegal exception @ 0x0000000080000000 (0x00100093)\n",
        )
        .expect("write hart1 trace");

        let cfg = TraceConfig::from_run_directory_with_harts(tmp.path(), 2)
            .expect("collect multicore traces");
        let rvfi = analyze_traces(&cfg).expect("analyze merged traces");
        let dump_path = tmp.path().join("program.dump");
        fs::write(&dump_path, dump).expect("write dump");
        let tracer = Tracer::new(&instructions, &dump_path, ISABase::Rv64).expect("parse dump");

        let hart0 = build_execution_output(&rvfi, 0, &tracer);
        let hart1 = build_execution_output(&rvfi, 1, &tracer);

        let hart0_main = hart0.instructions.first().expect("hart0 main instruction");
        assert_eq!(hart0_main.register_writes.len(), 1);
        assert_eq!(hart0_main.register_writes[0].name, "x1");
        assert_eq!(hart0_main.register_writes[0].value, 1);
        assert!(hart0_main.memory_writes.is_empty());
        assert!(hart0_main.exceptions.is_empty());

        let hart1_main = hart1.instructions.first().expect("hart1 main instruction");
        assert_eq!(hart1_main.register_writes.len(), 1);
        assert_eq!(hart1_main.register_writes[0].name, "x2");
        assert_eq!(hart1_main.register_writes[0].value, 2);
        assert_eq!(hart1_main.memory_writes.len(), 1);
        assert_eq!(hart1_main.memory_writes[0].addr, 0x1000);
        assert_eq!(hart1_main.memory_writes[0].value, 0xaa);
        assert_eq!(
            hart1_main.exceptions,
            vec![ExceptionCause::unknown("illegal")]
        );
    }

    #[test]
    fn parse_rvfi_rejects_conflicting_register_write_for_same_hart_pc_reg() {
        let rvfi = parse_rvfi(
            Cursor::new(
                "\
3 0x0000000080000000 (0x00000013) x1 0x0000000000000001\n\
3 0x0000000080000000 (0x00000013) x1 0x0000000000000002\n",
            ),
            0,
        )
        .expect("repeated static pc register write should keep the latest value");

        let writes = rvfi
            .register_writes_for_hart_pc(0, 0x0000_0000_8000_0000)
            .expect("register write should be present");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].register, "x1");
        assert_eq!(writes[0].value, 2);
    }

    #[test]
    fn parse_rvfi_keeps_latest_memory_write_for_same_hart_pc_addr() {
        let rvfi = parse_rvfi(
            Cursor::new(
                "\
3 0x0000000080000000 (0x00000023) mem 0x0000000000001000 0x00000000000000aa 0x01 [0]0xaa@0x0000000000001000 clk_start=10 clk_end=10 clk_span=1\n\
3 0x0000000080000000 (0x00000023) mem 0x0000000000001000 0x00000000000000bb 0x01 [0]0xbb@0x0000000000001000 clk_start=20 clk_end=20 clk_span=1\n",
            ),
            0,
        )
        .expect("repeated static pc memory write should keep the latest value");

        let writes = rvfi
            .memory_writes_for_hart_pc(0, 0x0000_0000_8000_0000)
            .expect("memory write should be present");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].address, 0x1000);
        assert_eq!(writes[0].value, 0xbb);
        assert_eq!(writes[0].byte_writes.len(), 1);
        assert_eq!(writes[0].byte_writes[0].value, 0xbb);
        assert_eq!(writes[0].timing.clk_start, Some(20));
        assert_eq!(writes[0].timing.clk_end, Some(20));
    }

    #[test]
    fn parse_rvfi_rejects_ambiguous_conflicting_memory_write_for_same_hart_pc_addr() {
        let err = parse_rvfi(
            Cursor::new(
                "\
3 0x0000000080000000 (0x00000023) mem 0x0000000000001000 0x00000000000000aa 0x01 [0]0xaa@0x0000000000001000 clk_start=10 clk_end=10 clk_span=1\n\
3 0x0000000080000000 (0x00000023) mem 0x0000000000001000 0x00000000000000bb 0x01 [0]0xbb@0x0000000000001000 clk_start=10 clk_end=10 clk_span=1\n",
            ),
            0,
        )
        .expect_err("same-cycle conflicting memory write should fail");

        assert!(format!("{err}").contains("conflicting ARA memory write"));
    }

    #[test]
    fn parse_rvfi_rejects_second_exception_for_same_hart_pc() {
        let err = parse_rvfi(
            Cursor::new(
                "\
illegal exception @ 0x0000000080000000 (0x00000073)\n\
breakpoint exception @ 0x0000000080000000 (0x00000073)\n",
            ),
            0,
        )
        .expect_err("duplicate exception should fail");

        assert!(format!("{err}").contains("conflicting ARA exception"));
    }

    #[test]
    fn parse_rvfi_merges_conflicting_timing_records_for_same_hart_pc() {
        let rvfi = parse_rvfi(
            Cursor::new(
                "\
3 0x0000000080000000 (0x00000013) clk_start=1 clk_end=2 clk_span=2\n\
3 0x0000000080000000 (0x00000013) clk_start=3 clk_end=4 clk_span=2\n",
            ),
            0,
        )
        .expect("conflicting timing records should merge deterministically");

        let timing = rvfi
            .instruction_timing_for_hart_pc(0, 0x0000_0000_8000_0000)
            .expect("merged timing should be recorded");
        assert_eq!(timing.timing.clk_start, Some(1));
        assert_eq!(timing.timing.clk_end, Some(4));
        assert_eq!(timing.timing.clk_span, Some(4));
    }

    #[test]
    fn merge_rvfi_traces_dedups_duplicate_timing_for_same_hart_pc() {
        let first = parse_rvfi(
            Cursor::new("3 0x0000000080000000 (0x00000013) clk_start=8 clk_end=8 clk_span=1\n"),
            0,
        )
        .expect("parse first rvfi");
        let second = parse_rvfi(
            Cursor::new("3 0x0000000080000000 (0x00000013) clk_start=8 clk_end=8 clk_span=1\n"),
            0,
        )
        .expect("parse second rvfi");

        let merged = merge_rvfi_traces(vec![first, second]).expect("merge duplicate timings");
        let timing = merged
            .instruction_timing_for_hart_pc(0, 0x0000_0000_8000_0000)
            .expect("merged timing should be present");
        assert_eq!(timing.timing.clk_start, Some(8));
        assert_eq!(timing.timing.clk_end, Some(8));
        assert_eq!(timing.timing.clk_span, Some(1));
    }

    #[test]
    fn merge_rvfi_traces_keeps_latest_register_write_for_same_hart_pc_reg() {
        let first = parse_rvfi(
            Cursor::new(
                "3 0x00000000800014cc (0x00000013) x10 0x0000000000000001 clk_start=10 clk_end=11 clk_span=2\n",
            ),
            1,
        )
        .expect("parse first rvfi");
        let second = parse_rvfi(
            Cursor::new(
                "3 0x00000000800014cc (0x00000013) x10 0x0000000000000002 clk_start=20 clk_end=21 clk_span=2\n",
            ),
            1,
        )
        .expect("parse second rvfi");

        let merged =
            merge_rvfi_traces(vec![first, second]).expect("merge repeated register writes");
        let writes = merged
            .register_writes_for_hart_pc(1, 0x0000_0000_8000_14cc)
            .expect("merged register write should be present");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].register, "x10");
        assert_eq!(writes[0].value, 2);
        assert_eq!(writes[0].timing.clk_start, Some(20));
        assert_eq!(writes[0].timing.clk_end, Some(21));
    }

    #[test]
    fn merge_rvfi_traces_keeps_latest_memory_write_for_same_hart_pc_addr() {
        let first = parse_rvfi(
            Cursor::new(
                "3 0x0000000080001496 (0x00000023) mem 0x0000000080004000 0x00000000000000aa 0x01 [0]0xaa@0x0000000080004000 clk_start=10 clk_end=11 clk_span=2\n",
            ),
            1,
        )
        .expect("parse first rvfi");
        let second = parse_rvfi(
            Cursor::new(
                "3 0x0000000080001496 (0x00000023) mem 0x0000000080004000 0x00000000000000bb 0x01 [0]0xbb@0x0000000080004000 clk_start=20 clk_end=21 clk_span=2\n",
            ),
            1,
        )
        .expect("parse second rvfi");

        let merged =
            merge_rvfi_traces(vec![first, second]).expect("merge repeated static pc memory writes");
        let writes = merged
            .memory_writes_for_hart_pc(1, 0x0000_0000_8000_1496)
            .expect("merged memory write should be present");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].address, 0x0000_0000_8000_4000);
        assert_eq!(writes[0].value, 0xbb);
        assert_eq!(writes[0].timing.clk_start, Some(20));
        assert_eq!(writes[0].timing.clk_end, Some(21));
    }

    #[test]
    fn parse_rvfi_keeps_lr_memory_side_effect_when_present() {
        let rvfi = parse_rvfi(Cursor::new(
            "3 0x0000000080000150 (0x1404292f) x18 0xffffffff98d3f2f7 mem 0x000000008fffffe0 0x0000000000000000 0x0f [0]0x00@0x000000008fffffe0 [1]0x00@0x000000008fffffe1 [2]0x00@0x000000008fffffe2 [3]0x00@0x000000008fffffe3\n",
        ), 0)
        .expect("parse rvfi");

        let writes = rvfi
            .memory_writes_for_hart_pc(0, 0x0000_0000_8000_0150)
            .expect("lr memory side-effect should be captured");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].address, 0x0000_0000_8fff_ffe0);
        assert_eq!(writes[0].value, 0);
        assert_eq!(writes[0].mask, 0x0f);
        assert_eq!(writes[0].byte_writes.len(), 4);
        let writes = rvfi
            .register_writes_for_hart_pc(0, 0x0000_0000_8000_0150)
            .expect("lr register write should be captured");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].register, "x18");
    }

    #[test]
    fn parse_rvfi_keeps_failed_store_conditional_memory_side_effect_when_present() {
        let rvfi = parse_rvfi(Cursor::new(
            "3 0x000000008000018c (0x1af4b8af) x17 0x0000000000000001 mem 0x000000008fffffd8 0x366b5cc38f13f52f 0xff [0]0x2f@0x000000008fffffd8 [1]0xf5@0x000000008fffffd9 [2]0x13@0x000000008fffffda [3]0x8f@0x000000008fffffdb [4]0xc3@0x000000008fffffdc [5]0x5c@0x000000008fffffdd [6]0x6b@0x000000008fffffde [7]0x36@0x000000008fffffdf\n",
        ), 0)
        .expect("parse rvfi");

        let writes = rvfi
            .memory_writes_for_hart_pc(0, 0x0000_0000_8000_018c)
            .expect("failed sc memory side-effect should be captured");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].address, 0x0000_0000_8fff_ffd8);
        assert_eq!(writes[0].value, 0x366b_5cc3_8f13_f52f);
        assert_eq!(writes[0].mask, 0xff);
        assert_eq!(writes[0].byte_writes.len(), 8);
        let writes = rvfi
            .register_writes_for_hart_pc(0, 0x0000_0000_8000_018c)
            .expect("sc register write should be captured");
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].register, "x17");
        assert_eq!(writes[0].value, 1);
    }

    #[test]
    fn parse_rvfi_keeps_store_with_trailing_clock_fields() {
        let rvfi = parse_rvfi(Cursor::new(
            "3 0x0000000080000102 (0xe088) mem 0x000000008ffffff0 0x5e5c7962b2bc7d96 0xff [0]0x96@0x000000008ffffff0 [1]0x7d@0x000000008ffffff1 [2]0xbc@0x000000008ffffff2 [3]0xb2@0x000000008ffffff3 [4]0x62@0x000000008ffffff4 [5]0x79@0x000000008ffffff5 [6]0x5c@0x000000008ffffff6 [7]0x5e@0x000000008ffffff7 clk_start=543 clk_end=549 clk_span=7\n",
        ), 0)
        .expect("parse rvfi");

        let writes = rvfi
            .memory_writes_for_hart_pc(0, 0x0000_0000_8000_0102)
            .expect("store pc should be captured");
        assert_eq!(writes.len(), 1);
        let write = &writes[0];
        assert_eq!(write.address, 0x0000_0000_8fff_fff0);
        assert_eq!(write.mask, 0xff);
        assert_eq!(write.width, 8);
        assert_eq!(write.byte_writes.len(), 8);
        assert_eq!(write.byte_writes[0].address, 0x0000_0000_8fff_fff0);
        assert_eq!(write.byte_writes[0].value, 0x96);
        assert_eq!(write.byte_writes[7].address, 0x0000_0000_8fff_fff7);
        assert_eq!(write.byte_writes[7].value, 0x5e);
        assert_eq!(write.timing.clk_start, Some(543));
        assert_eq!(write.timing.clk_end, Some(549));
        assert_eq!(write.timing.clk_span, Some(7));
        assert!(write.timing.is_consistent());
    }

    #[test]
    fn streaming_parse_existing_ara_traces_ignores_incomplete_rvfi_line() {
        let tmp = tempdir().expect("create tempdir");
        let trace_path = tmp.path().join("trace_rvfi_hart_00000000.dasm");
        fs::write(
            &trace_path,
            concat!(
                "3 0x0000000080000102 (0xe088) mem 0x000000008ffffff0 0x5e5c7962b2bc7d96 0xff ",
                "[0]0x96@0x000000008ffffff0 [1]0x7d@0x000000008ffffff1 ",
                "[2]0xbc@0x000000008ffffff2 [3]0xb2@0x000000008ffffff3 ",
                "[4]0x62@0x000000008ffffff4 [5]0x79@0x000000008ffffff5 ",
                "[6]0x5c@0x000000008ffffff6 [7]0x5e@0x000000008ffffff7 ",
                "clk_start=543 clk_end=549 clk_span=7\n",
                "3 0x0000000080000104 (0xe894) mem 0x000000008ffffff8 0x0000000063ff8000 0xff ",
                "[0]0x00@0x000000008ffffff8 [1]0x80@0x000000008ffffff9 ",
                "[2]0xff@0x000000008ffffffa",
            ),
        )
        .expect("write trace");

        let rvfi = parse_existing_ara_traces(tmp.path(), 1)
            .expect("live parse should ignore trailing partial line")
            .expect("trace should parse");

        assert!(
            rvfi.memory_writes
                .contains_key(&TraceKey::new(0, 0x0000_0000_8000_0102))
        );
        assert!(
            !rvfi
                .memory_writes
                .contains_key(&TraceKey::new(0, 0x0000_0000_8000_0104))
        );
    }

    #[test]
    fn parse_rvfi_keeps_timing_only_instruction_lines() {
        let rvfi = parse_rvfi(
            Cursor::new("3 0x0000000080000106 (0x0001) clk_start=551 clk_end=553 clk_span=3\n"),
            0,
        )
        .expect("parse rvfi");

        let timing = rvfi
            .instruction_timing_for_hart_pc(0, 0x0000_0000_8000_0106)
            .expect("timing-only pc should be captured");
        assert_eq!(timing.timing.clk_start, Some(551));
        assert_eq!(timing.timing.clk_end, Some(553));
        assert_eq!(timing.timing.clk_span, Some(3));
        assert!(timing.timing.is_consistent());
    }

    #[test]
    fn parse_rvfi_timing_only_line_does_not_synthesize_exception() {
        let rvfi = parse_rvfi(
            Cursor::new("3 0x0000000080000590 (0x00000073) clk_start=100 clk_end=104 clk_span=5\n"),
            0,
        )
        .expect("parse rvfi");

        assert!(
            rvfi.exception_for_hart_pc(0, 0x0000_0000_8000_0590)
                .is_none(),
            "timing-only lines must not synthesize exceptions"
        );
    }

    #[test]
    fn parse_rvfi_keeps_explicit_exception_even_with_timing_only_line() {
        let rvfi = parse_rvfi(
            Cursor::new(
                "\
3 0x0000000080000590 (0x00000073) clk_start=100 clk_end=104 clk_span=5\n\
illegal exception @ 0x0000000080000590 (0x00000073)\n",
            ),
            0,
        )
        .expect("parse rvfi");

        let event = rvfi
            .exception_for_hart_pc(0, 0x0000_0000_8000_0590)
            .expect("exception should be recorded");
        assert_eq!(event.kind, "illegal");
        assert_eq!(event.code, None);
    }
}
