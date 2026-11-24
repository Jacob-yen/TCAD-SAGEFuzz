use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};
use once_cell::sync::Lazy;
use regex::Regex;
use riscv_instruction::RiscvInstruction;
use serde::Serialize;
use thiserror::Error;

use super::march;
use super::{
    asm::{build_asm_with_vector_state, linker_script},
    metadata::isa_profiles,
};
use riscv_fuzz_plugin_api::doctor::parse_global_cycles_from_text;
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
    tracer::{Tracer, build_raw_hart_execution_output},
};

/// Convenience alias: explicit per-hart asm lines.
pub(crate) type PerHartAsm<'a> = &'a [(u32, &'a [String])];

pub(crate) const SPIKE_BINARY_ENV: &str = "CX_RISCV_CORES_SPIKE";
pub(crate) const SPIKE_EXTRA_ARGS_ENV: &str = "RISCV_FUZZ_SPIKE_EXTRA_ARGS";
pub(crate) const SPIKE_ISA_SUFFIX_ENV: &str = "RISCV_FUZZ_SPIKE_ISA_SUFFIX";
pub(crate) const SPIKE_ENABLE_VECTOR_STATE_ENV: &str = "RISCV_FUZZ_SPIKE_ENABLE_VECTOR_STATE";
const DEFAULT_SPIKE_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Error)]
pub enum SpikeError {
    #[error("environment variable {var} is not set")]
    EnvVarNotSet { var: String },

    #[error("Spike binary not found: {path}")]
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

impl From<SpikeError> for ImplExecutionError {
    fn from(value: SpikeError) -> Self {
        match value {
            SpikeError::ProcessError(err) => err.into(),
            SpikeError::LogParseError(err) => err.into(),
            SpikeError::ParseError(err) => err.into(),
            SpikeError::ElfError(err) => err.into(),
            SpikeError::IoError(err) => err.into(),
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

fn resolve_wrapper_elf_path(elf_path: &Path, run_dir: &Path) -> Result<PathBuf, SpikeError> {
    if elf_path.is_absolute() {
        if elf_path.is_file() {
            return Ok(elf_path.to_path_buf());
        }
        return Err(SpikeError::BuildElfError(format!(
            "built ELF path does not exist: {}",
            elf_path.display()
        )));
    }

    if elf_path.exists() {
        return fs::canonicalize(elf_path).map_err(|err| {
            SpikeError::BuildElfError(format!(
                "failed to canonicalize built ELF {}: {err}",
                elf_path.display()
            ))
        });
    }

    let Some(file_name) = elf_path.file_name() else {
        return Err(SpikeError::BuildElfError(format!(
            "built ELF path has no file name: {}",
            elf_path.display()
        )));
    };

    let candidate = run_dir.join(file_name);
    if !candidate.is_file() {
        return Err(SpikeError::BuildElfError(format!(
            "built ELF path does not exist: {}; run-dir candidate also missing: {}",
            elf_path.display(),
            candidate.display()
        )));
    }

    fs::canonicalize(&candidate).map_err(|err| {
        SpikeError::BuildElfError(format!(
            "failed to canonicalize run-dir ELF candidate {}: {err}",
            candidate.display()
        ))
    })
}

/// Spike 日志中的寄存器变化记录
#[derive(Debug, Clone, Serialize)]
pub struct RegisterWrite {
    pub hart: u32,
    pub index: u8,
    pub pc: u64,
    pub register: String,
    pub value: u64,
}

/// Spike 日志中的内存变化记录
#[derive(Debug, Clone, Serialize)]
pub struct MemoryWrite {
    pub hart: u32,
    pub pc: u64,
    pub address: u64,
    pub value: u64,
    pub width: usize,
}

/// Spike 日志中的异常记录
#[derive(Debug, Clone, Serialize)]
pub struct ExceptionEvent {
    pub hart: u32,
    pub cause: String,
    pub pc: u64,
    pub mcause: Option<u64>,
}

/// Spike 日志解析结果
#[derive(Debug, Clone, Serialize, Default)]
pub struct SpikeTrace {
    pub register_writes: HashMap<u64, Vec<RegisterWrite>>,
    pub memory_writes: HashMap<u64, Vec<MemoryWrite>>,
    pub exceptions: HashMap<u32, HashMap<u64, ExceptionEvent>>,
}

#[derive(Debug, Clone)]
struct StreamSourceIdentity {
    trace_inst_id: riscv_fuzz_plugin_api::TraceInstId,
    instruction: RiscvInstruction,
}

fn resolve_stream_trace_inst_identity(
    tracers: &HashMap<u32, Tracer>,
    hart_id: u32,
    pc: u64,
    strict_typed_mapping: bool,
) -> Result<Option<StreamSourceIdentity>, String> {
    let Some(tracer) = tracers.get(&hart_id) else {
        return if strict_typed_mapping {
            Err(format!(
                "Spike typed trace mapping missing tracer for hart={hart_id} pc=0x{pc:x}"
            ))
        } else {
            Ok(None)
        };
    };

    let Some(traced) = tracer.instructions().iter().find(|ti| ti.pc == pc) else {
        return Ok(None);
    };

    if strict_typed_mapping {
        let Some(local_index) = traced.trace_inst_local_index else {
            return if traced.source_instruction.is_none() {
                Ok(None)
            } else {
                Err(format!(
                    "Spike typed trace mapping missing trace instruction local index for hart={hart_id} pc=0x{pc:x}"
                ))
            };
        };
        let instruction = traced
            .parsed_disassembly
            .clone()
            .or_else(|| traced.source_instruction.clone())
            .ok_or_else(|| {
                format!(
                    "Spike typed trace mapping missing decoded instruction for hart={hart_id} pc=0x{pc:x}"
                )
            })?;

        return Ok(Some(StreamSourceIdentity {
            trace_inst_id: riscv_fuzz_plugin_api::TraceInstId::new(hart_id, local_index),
            instruction,
        }));
    }

    let Some(local_index) = traced.trace_inst_local_index else {
        return Ok(None);
    };
    let Some(instruction) = traced
        .parsed_disassembly
        .clone()
        .or_else(|| traced.source_instruction.clone())
    else {
        return Ok(None);
    };
    Ok(Some(StreamSourceIdentity {
        trace_inst_id: riscv_fuzz_plugin_api::TraceInstId::new(hart_id, local_index),
        instruction,
    }))
}

struct StreamedPcRow<'a> {
    pc: u64,
    hart_id: u32,
    reg_writes: &'a [TraitRegisterValue],
    mem_writes: &'a [(u64, MemoryWrite)],
    exceptions: &'a [ExceptionEvent],
    tracers: &'a HashMap<u32, Tracer>,
    strict_typed_mapping: bool,
}

fn emit_streamed_pc_row(
    row: StreamedPcRow<'_>,
    emit_fn: &mut dyn FnMut(riscv_fuzz_plugin_api::streaming::InstructionEvent) -> bool,
) -> Result<bool, String> {
    let Some(identity) = resolve_stream_trace_inst_identity(
        row.tracers,
        row.hart_id,
        row.pc,
        row.strict_typed_mapping,
    )?
    else {
        return Ok(true);
    };

    let mut mem_vals: Vec<riscv_fuzz_plugin_api::MemValue> = Vec::new();
    for (_pc, mw) in row
        .mem_writes
        .iter()
        .filter(|(_, mw)| mw.hart == row.hart_id)
    {
        for (addr, byte) in store_bytes_from_write(mw) {
            mem_vals.push(riscv_fuzz_plugin_api::MemValue { addr, value: byte });
        }
    }
    mem_vals.sort_by_key(|m| m.addr);

    let exception_causes: Vec<riscv_fuzz_plugin_api::ExceptionCause> = row
        .exceptions
        .iter()
        .filter(|e| e.hart == row.hart_id && !e.cause.contains("interrupt"))
        .map(|e| {
            if let Some(code) = e.mcause {
                riscv_fuzz_plugin_api::ExceptionCause::from_code(code, e.cause.clone())
            } else {
                riscv_fuzz_plugin_api::ExceptionCause::unknown(e.cause.clone())
            }
        })
        .collect();

    let mut sorted_regs = row.reg_writes.to_vec();
    sorted_regs.sort_by(|a, b| compare_register_names(&a.name, &b.name));

    let asm_inst = riscv_fuzz_plugin_api::AsmInstructionExecution {
        pc: row.pc,
        instruction: identity.instruction,
        source_annotation: None,
        trace_inst_id: Some(identity.trace_inst_id),
        exceptions: exception_causes,
        register_writes: sorted_regs,
        memory_writes: mem_vals,
        timing: riscv_fuzz_plugin_api::InstructionTiming::default(),
    };

    Ok(emit_fn(
        riscv_fuzz_plugin_api::streaming::InstructionEvent::Instruction {
            hart_id: row.hart_id,
            instruction: asm_inst,
        },
    ))
}

/// 基本配置，描述 Spike 运行所需的环境。
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub work_dir: PathBuf,
    pub isa_base: ISABase,
    pub core_id: String,
    pub spike_args: Vec<String>,
    pub timeout: Option<Duration>,
    pub isa_suffix: Option<String>,
    pub enable_vector_state: bool,
    extension_override: Option<IsaExtensions>,
}

impl ExecutorConfig {
    pub fn new(work_dir: PathBuf, isa_base: ISABase, core_id: String) -> Self {
        Self {
            work_dir,
            isa_base,
            core_id,
            spike_args: Vec::new(),
            timeout: Some(DEFAULT_SPIKE_TIMEOUT),
            isa_suffix: None,
            enable_vector_state: false,
            extension_override: None,
        }
    }

    pub fn set_extension_override(&mut self, extensions: IsaExtensions) {
        self.extension_override = Some(extensions);
    }

    fn resolved_extensions(&self) -> IsaExtensions {
        if let Some(ref override_map) = self.extension_override {
            override_map.clone()
        } else {
            isa_profiles().for_isa(self.isa_base).extensions.clone()
        }
    }

    fn effective_isa_string(&self, isa_string: &str) -> String {
        let mut effective = isa_string.to_string();
        if let Some(suffix) = &self.isa_suffix {
            let suffix = suffix.trim();
            if !suffix.is_empty() {
                if suffix.starts_with('_') {
                    effective.push_str(suffix);
                } else {
                    effective.push('_');
                    effective.push_str(suffix);
                }
            }
        }
        effective
    }

    fn spike_binary(&self) -> Result<PathBuf, SpikeError> {
        let spike_binary = env::var(SPIKE_BINARY_ENV).map_err(|_| SpikeError::EnvVarNotSet {
            var: SPIKE_BINARY_ENV.to_string(),
        })?;
        Ok(PathBuf::from(spike_binary))
    }
}

/// Spike 执行器，只负责编译和执行，不封装输出解析。
#[derive(Debug)]
pub struct SpikeExecutor {
    config: ExecutorConfig,
}

impl SpikeExecutor {
    pub fn new(config: ExecutorConfig) -> Self {
        Self { config }
    }

    /// 编译汇编程序并运行 Spike，返回原始输出及构建产物。
    pub fn compile_and_run_with_dir(
        &self,
        run_dir: &Path,
        per_hart: PerHartAsm<'_>,
    ) -> Result<SpikeRunResult, SpikeError> {
        let total_lines: usize = per_hart.iter().map(|(_, lines)| lines.len()).sum();
        info!(
            "Starting Spike compile-and-run with {} testcase instructions across {} hart(s) in {}",
            total_lines,
            per_hart.len(),
            run_dir.display()
        );
        self.ensure_work_dir(run_dir)?;

        let extensions = self.config.resolved_extensions();
        let hart_count = per_hart.len().max(1) as u32;
        let program = self.build_program(per_hart)?;
        let asm_path = self.write_program(run_dir, &program)?;
        let build =
            build_elf_with_extensions(&asm_path, &extensions, linker_script()).map_err(|err| {
                error!(
                    "Failed to build ELF for Spike execution (asm={}, isa={:?}, impl={:?}): {:#}",
                    asm_path.display(),
                    self.config.isa_base,
                    self.config.core_id.clone(),
                    err
                );
                err
            })?;
        info!("ELF build completed at {}", build.executable_file.display());

        let isa_string = match &extensions {
            IsaExtensions::Rv32(extensions) => march::isa_from_rv32_extensions(extensions),
            IsaExtensions::Rv64(extensions) => march::isa_from_rv64_extensions(extensions),
        }
        .map_err(|err| {
            error!(
                "Failed to derive ISA string for Spike (isa={:?}): {:#}",
                self.config.isa_base, err
            );
            SpikeError::BuildElfError(format!("Failed to derive ISA string for Spike: {}", err))
        })?;

        let output = self.run_spike(run_dir, &build, &isa_string, hart_count)?;
        info!("Spike finished with status {:?}", output.status.code());

        Ok(output)
    }

    /// 编译汇编程序并运行 Spike，返回原始输出及构建产物。
    pub fn compile_and_run(&self, per_hart: PerHartAsm<'_>) -> Result<SpikeRunResult, SpikeError> {
        self.compile_and_run_with_dir(&self.config.work_dir, per_hart)
    }

    /// 只执行编译步骤，返回构建后的 ELF 结果。
    pub fn compile(&self, per_hart: PerHartAsm<'_>) -> Result<ElfBuildResult, SpikeError> {
        let total_lines: usize = per_hart.iter().map(|(_, lines)| lines.len()).sum();
        info!(
            "Compiling Spike program with {} testcase instructions across {} hart(s)",
            total_lines,
            per_hart.len()
        );
        self.ensure_work_dir(&self.config.work_dir)?;
        let extensions = self.config.resolved_extensions();
        let program = self.build_program(per_hart)?;
        let asm_path = self.write_program(&self.config.work_dir, &program)?;
        let build =
            build_elf_with_extensions(&asm_path, &extensions, linker_script()).map_err(|err| {
                error!(
                    "Failed to build ELF for Spike (asm={}, isa={:?}, impl={:?}): {:#}",
                    asm_path.display(),
                    self.config.isa_base,
                    self.config.core_id.clone(),
                    err
                );
                err
            })?;
        info!(
            "Spike compilation finished: {}",
            build.executable_file.display()
        );
        Ok(build)
    }

    fn ensure_work_dir(&self, dir: &Path) -> Result<(), SpikeError> {
        if dir.exists() {
            return Ok(());
        }

        fs::create_dir_all(dir).map_err(|err| {
            error!("Failed to create work directory {}: {}", dir.display(), err);
            SpikeError::from(err)
        })
    }

    fn write_program(&self, dir: &Path, program: &str) -> Result<PathBuf, SpikeError> {
        let asm_path = dir.join("program.S");
        fs::write(&asm_path, program).map_err(|err| {
            error!(
                "Failed to write assembly file {}: {}",
                asm_path.display(),
                err
            );
            err
        })?;
        debug!("Assembly written to {}", asm_path.display());
        Ok(asm_path)
    }

    fn build_program(&self, per_hart: PerHartAsm<'_>) -> Result<String, SpikeError> {
        Ok(build_asm_with_vector_state(
            per_hart,
            self.config.isa_base,
            self.config.enable_vector_state,
        )?)
    }

    fn run_spike(
        &self,
        run_dir: &Path,
        build: &ElfBuildResult,
        isa_string: &str,
        hart_count: u32,
    ) -> Result<SpikeRunResult, SpikeError> {
        let spike_binary = self.config.spike_binary()?;

        // Check if executable is available (either as file path or in PATH)
        if !is_command_available(&spike_binary) {
            return Err(SpikeError::BinaryNotFound {
                path: spike_binary.display().to_string(),
            });
        }

        let effective_isa = self.config.effective_isa_string(isa_string);

        let mut cmd = Command::new(&spike_binary);
        cmd.arg(format!("-p{}", hart_count.max(1)));
        cmd.arg(format!("--isa={}", effective_isa.to_uppercase()));
        cmd.arg("--log-commits");
        cmd.arg("-l");
        cmd.args(&self.config.spike_args);

        // Spike is executed with `current_dir(run_dir)`. Resolve the built ELF explicitly so
        // missing or ambiguous paths fail immediately instead of silently degrading.
        let elf_arg = resolve_wrapper_elf_path(&build.executable_file, run_dir)?;

        cmd.arg(elf_arg);
        cmd.current_dir(run_dir);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        configure_command_process_group(&mut cmd);

        let command_display = format!("{:?}", cmd);
        info!("Launching Spike command: {}", command_display);

        let mut child = cmd.spawn().map_err(|err| {
            error!("Failed to execute spike command: {}", err);
            SpikeError::from(ProcessError::SpawnFailed {
                command: command_display.clone(),
                source: err,
            })
        })?;

        let stdout_reader = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("failed to capture Spike stdout"))?;
        let stderr_reader = child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("failed to capture Spike stderr"))?;

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
        let start = Instant::now();
        let mut timed_out = false;
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
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
            .map_err(|_| io::Error::other("stdout reader panicked"))??;
        let stderr_bytes = stderr_handle
            .join()
            .map_err(|_| io::Error::other("stderr reader panicked"))??;

        let stdout_path = run_dir.join("spike_stdout.log");
        let stderr_path = run_dir.join("spike_stderr.log");
        fs::write(&stdout_path, &stdout_bytes).map_err(|err| {
            error!(
                "Failed to persist Spike stdout to {}: {}",
                stdout_path.display(),
                err
            );
            SpikeError::from(ProcessError::FileWriteFailed {
                path: stdout_path.clone(),
                source: err,
            })
        })?;
        fs::write(&stderr_path, &stderr_bytes).map_err(|err| {
            error!(
                "Failed to persist Spike stderr to {}: {}",
                stderr_path.display(),
                err
            );
            SpikeError::from(ProcessError::FileWriteFailed {
                path: stderr_path.clone(),
                source: err,
            })
        })?;

        let stdout = String::from_utf8_lossy(&stdout_bytes).to_string();
        let stderr = String::from_utf8_lossy(&stderr_bytes).to_string();

        if timed_out {
            if let Some(limit) = timeout {
                error!("Spike execution timed out after {:?}", limit);
                return Err(SpikeError::from(ProcessError::TimedOut {
                    command: command_display,
                    timeout: limit,
                }));
            } else {
                unreachable!("timed_out implies timeout was set");
            }
        }

        if !status.success() {
            error!(
                "Spike process exited unsuccessfully with status {:?}; stderr persisted to {}",
                status.code(),
                stderr_path.display()
            );
            return Err(SpikeError::from(ProcessError::ProcessFailed {
                command: command_display,
                stderr,
            }));
        }

        Ok(SpikeRunResult {
            build: build.clone(),
            isa: isa_string.to_string(),
            status,
            stdout,
            stderr,
            stdout_path,
            stderr_path,
        })
    }
}

/// Spike 执行结果，便于后续查看 Spike 的原始输出。
#[derive(Debug)]
pub struct SpikeRunResult {
    pub build: ElfBuildResult,
    pub isa: String,
    pub status: ExitStatus,
    pub stdout: String,
    pub stderr: String,
    pub stdout_path: PathBuf,
    pub stderr_path: PathBuf,
}

/// Helper function to compile regex patterns with error handling
fn compile_regex(pattern: &str) -> Result<Regex, LogParseError> {
    Regex::new(pattern).map_err(|e| LogParseError::RegexCompilationFailed {
        pattern: pattern.to_string(),
        source: e,
    })
}

// Module-level regex statics shared by both parse_spike_log and execute_streaming.
// Spike log format:
//   core   0: 3 0x<pc> (0x<instr>) [reg/mem writes...]
//   core   0: exception <kind>, epc 0x<pc>
static REG_CHANGE_LINE_RE: Lazy<Result<Regex, LogParseError>> = Lazy::new(|| {
    compile_regex(
        r"^core\s+(?P<hart>\d+):\s+3\s+0x(?P<pc>[0-9a-fA-F]+)\s+\(0x(?P<instr>[0-9a-fA-F]+)\)",
    )
});

static REG_MATCH_RE: Lazy<Result<Regex, LogParseError>> =
    Lazy::new(|| compile_regex(r"\b(?P<prefix>[xf])(?P<reg>\d+)\s+0x(?P<val>[0-9a-fA-F]+)"));

static MEM_RE: Lazy<Result<Regex, LogParseError>> =
    Lazy::new(|| compile_regex(r"mem\s+0x(?P<addr>[0-9a-fA-F]+)(?:\s+0x(?P<data>[0-9a-fA-F]+))?"));

static EXCEPTION_RE: Lazy<Result<Regex, LogParseError>> = Lazy::new(|| {
    compile_regex(
        r"^core\s+(?P<hart>\d+):\s+exception\s+(?P<kind>\w+),\s+epc\s+0x(?P<pc>[0-9a-fA-F]+)(?:,\s+(?:mcause|cause)\s+0x(?P<mcause>[0-9a-fA-F]+))?",
    )
});

fn format_optional_exception_mcause(mcause: Option<u64>) -> String {
    mcause.map_or_else(|| "unknown".to_string(), |code| format!("0x{code:x}"))
}

fn insert_exception_event(
    exceptions: &mut HashMap<u32, HashMap<u64, ExceptionEvent>>,
    event: ExceptionEvent,
) -> Result<(), SpikeError> {
    let per_hart = exceptions.entry(event.hart).or_default();
    if let Some(previous) = per_hart.get(&event.pc) {
        let detail = if previous.cause == event.cause && previous.mcause == event.mcause {
            format!(
                "duplicate Spike exception for hart={} pc=0x{:x}: cause='{}' mcause={}",
                event.hart,
                event.pc,
                event.cause,
                format_optional_exception_mcause(event.mcause)
            )
        } else {
            format!(
                "conflicting Spike exception for hart={} pc=0x{:x}: prev(cause='{}', mcause={}) new(cause='{}', mcause={})",
                event.hart,
                event.pc,
                previous.cause,
                format_optional_exception_mcause(previous.mcause),
                event.cause,
                format_optional_exception_mcause(event.mcause)
            )
        };
        return Err(SpikeError::from(LogParseError::InvalidLineFormat {
            line: detail,
        }));
    }
    per_hart.insert(event.pc, event);
    Ok(())
}

/// 解析 Spike 日志
fn parse_spike_log<R: BufRead>(reader: R) -> Result<SpikeTrace, SpikeError> {
    // Spike 日志格式示例：
    // core   0: 0x0000000000001000 (0x00000297) auipc   t0, 0x0
    // core   0: 3 0x0000000000001000 (0x00000297) x5  0x0000000000001000
    // core   0: 3 0x000000000000100c (0x0182b283) x5  0x0000000080000000 mem 0x0000000000001018
    // core   0: exception trap_load_address_misaligned, epc 0x000000008000005c
    // core   0:           tval 0x0000000080000027

    let reg_write_line_re = compiled_regex(&REG_CHANGE_LINE_RE)?;
    let reg_match_re = compiled_regex(&REG_MATCH_RE)?;
    let mem_re = compiled_regex(&MEM_RE)?;
    let exception_re = compiled_regex(&EXCEPTION_RE)?;

    // Use intermediate HashMaps to reject duplicate trace records explicitly.
    let mut temp_reg_writes: HashMap<u64, HashMap<String, RegisterWrite>> = HashMap::new();
    let mut temp_mem_writes: HashMap<u64, HashMap<String, MemoryWrite>> = HashMap::new();
    let mut exceptions: HashMap<u32, HashMap<u64, ExceptionEvent>> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // 检查异常
        if let Some(cap) = exception_re.captures(trimmed) {
            let hart_str = cap
                .name("hart")
                .ok_or_else(|| {
                    SpikeError::from(LogParseError::MissingCaptureGroup {
                        group: "hart".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let hart = hart_str.parse::<u32>().map_err(|e| {
                SpikeError::from(ParseError::IntParseError {
                    value: hart_str.to_string(),
                    source: e,
                })
            })?;
            let kind = cap
                .name("kind")
                .ok_or_else(|| {
                    SpikeError::from(LogParseError::MissingCaptureGroup {
                        group: "kind".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str()
                .to_string();
            let pc_str = cap
                .name("pc")
                .ok_or_else(|| {
                    SpikeError::from(LogParseError::MissingCaptureGroup {
                        group: "pc".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let pc = u64::from_str_radix(pc_str, 16).map_err(|e| {
                SpikeError::from(ParseError::HexParseError {
                    value: pc_str.to_string(),
                    source: e,
                })
            })?;
            let mcause = if let Some(mcause_match) = cap.name("mcause") {
                let mcause_str = mcause_match.as_str();
                Some(u64::from_str_radix(mcause_str, 16).map_err(|e| {
                    SpikeError::from(ParseError::HexParseError {
                        value: mcause_str.to_string(),
                        source: e,
                    })
                })?)
            } else {
                None
            };
            insert_exception_event(
                &mut exceptions,
                ExceptionEvent {
                    hart,
                    cause: kind,
                    pc,
                    mcause,
                },
            )?;
            continue;
        }

        // 检查寄存器/内存变化行
        if let Some(line_cap) = reg_write_line_re.captures(trimmed) {
            let hart_str = line_cap
                .name("hart")
                .ok_or_else(|| {
                    SpikeError::from(LogParseError::MissingCaptureGroup {
                        group: "hart".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();
            let hart = hart_str.parse::<u32>().map_err(|e| {
                SpikeError::from(ParseError::IntParseError {
                    value: hart_str.to_string(),
                    source: e,
                })
            })?;
            let pc_str = line_cap
                .name("pc")
                .ok_or_else(|| {
                    SpikeError::from(LogParseError::MissingCaptureGroup {
                        group: "pc".to_string(),
                        line: trimmed.to_string(),
                    })
                })?
                .as_str();

            let pc = u64::from_str_radix(pc_str, 16).map_err(|e| {
                SpikeError::from(ParseError::HexParseError {
                    value: pc_str.to_string(),
                    source: e,
                })
            })?;

            // Extract all register writes from this line (there may be multiple)
            for reg_cap in reg_match_re.captures_iter(trimmed) {
                let prefix = reg_cap
                    .name("prefix")
                    .ok_or_else(|| {
                        SpikeError::from(LogParseError::MissingCaptureGroup {
                            group: "prefix".to_string(),
                            line: trimmed.to_string(),
                        })
                    })?
                    .as_str();
                let reg_str = reg_cap
                    .name("reg")
                    .ok_or_else(|| {
                        SpikeError::from(LogParseError::MissingCaptureGroup {
                            group: "reg".to_string(),
                            line: trimmed.to_string(),
                        })
                    })?
                    .as_str();
                let val_str = reg_cap
                    .name("val")
                    .ok_or_else(|| {
                        SpikeError::from(LogParseError::MissingCaptureGroup {
                            group: "val".to_string(),
                            line: trimmed.to_string(),
                        })
                    })?
                    .as_str();

                let reg_idx = reg_str.parse::<u8>().map_err(|e| {
                    SpikeError::from(ParseError::IntParseError {
                        value: format!("{}{}", prefix, reg_str),
                        source: e,
                    })
                })?;

                let value = u64::from_str_radix(val_str, 16).map_err(|e| {
                    SpikeError::from(ParseError::HexParseError {
                        value: val_str.to_string(),
                        source: e,
                    })
                })?;

                let register_name = format!("{}{}", prefix, reg_idx);
                let uniq_reg_name = format!("h{}:{}", hart, register_name);
                let writes = temp_reg_writes.entry(pc).or_default();
                if let Some(previous) = writes.get(&uniq_reg_name) {
                    let detail = if previous.value == value {
                        format!(
                            "duplicate Spike register write for hart={hart} pc=0x{pc:x} {register_name}"
                        )
                    } else {
                        format!(
                            "conflicting Spike register write for hart={hart} pc=0x{pc:x} {register_name}: 0x{:x} vs 0x{value:x}",
                            previous.value
                        )
                    };
                    return Err(SpikeError::from(LogParseError::InvalidLineFormat {
                        line: detail,
                    }));
                }
                writes.insert(
                    uniq_reg_name,
                    RegisterWrite {
                        hart,
                        index: reg_idx,
                        pc,
                        register: register_name,
                        value,
                    },
                );
            }

            // 检查内存访问 - 一行可能包含多个 mem 条目（如 cbo.zero 会产生 64 个单字节条目）
            for mem_cap in mem_re.captures_iter(trimmed) {
                let addr_str = mem_cap
                    .name("addr")
                    .ok_or_else(|| {
                        SpikeError::from(LogParseError::MissingCaptureGroup {
                            group: "addr".to_string(),
                            line: trimmed.to_string(),
                        })
                    })?
                    .as_str();
                let address = u64::from_str_radix(addr_str, 16).map_err(|e| {
                    SpikeError::from(ParseError::HexParseError {
                        value: addr_str.to_string(),
                        source: e,
                    })
                })?;

                if let Some(data_match) = mem_cap.name("data") {
                    // data_match 只包含十六进制数字，不包含 "0x" 前缀
                    let digits = data_match.as_str();

                    if digits.is_empty() {
                        return Err(SpikeError::LogParseError(
                            LogParseError::InvalidLineFormat {
                                line: format!(
                                    "Empty data for memory store at PC 0x{:016x}, address 0x{:016x}: {}",
                                    pc, address, trimmed
                                ),
                            },
                        ));
                    }

                    // 计算字节宽度：每2个十六进制字符 = 1个字节
                    let byte_width = digits.len().div_ceil(2);

                    if byte_width > 8 {
                        return Err(SpikeError::LogParseError(
                            LogParseError::InvalidLineFormat {
                                line: format!(
                                    "Store width {} exceeds 8 bytes at PC 0x{:016x}: {}",
                                    byte_width, pc, trimmed
                                ),
                            },
                        ));
                    }

                    let store_value = u64::from_str_radix(digits, 16).map_err(|e| {
                        SpikeError::from(ParseError::HexParseError {
                            value: digits.to_string(),
                            source: e,
                        })
                    })?;

                    let uniq_mem_key = format!("h{}:{:x}", hart, address);
                    let writes = temp_mem_writes.entry(pc).or_default();
                    if let Some(previous) = writes.get(&uniq_mem_key) {
                        let detail = if previous.value == store_value
                            && previous.width == byte_width
                        {
                            format!(
                                "duplicate Spike memory write for hart={hart} pc=0x{pc:x} addr=0x{address:x}"
                            )
                        } else {
                            format!(
                                "conflicting Spike memory write for hart={hart} pc=0x{pc:x} addr=0x{address:x}"
                            )
                        };
                        return Err(SpikeError::from(LogParseError::InvalidLineFormat {
                            line: detail,
                        }));
                    }
                    writes.insert(
                        uniq_mem_key,
                        MemoryWrite {
                            hart,
                            pc,
                            address,
                            value: store_value,
                            width: byte_width,
                        },
                    );
                }
            }

            continue;
        }
    }

    // Convert intermediate HashMaps to final Vec format
    let register_writes: HashMap<u64, Vec<RegisterWrite>> = temp_reg_writes
        .into_iter()
        .map(|(pc, reg_map)| (pc, reg_map.into_values().collect()))
        .collect();

    let memory_writes: HashMap<u64, Vec<MemoryWrite>> = temp_mem_writes
        .into_iter()
        .map(|(pc, mem_map)| (pc, mem_map.into_values().collect()))
        .collect();

    let total_reg_writes: usize = register_writes.values().map(|v| v.len()).sum();
    let total_mem_writes: usize = memory_writes.values().map(|v| v.len()).sum();
    let total_exceptions: usize = exceptions.values().map(|per_hart| per_hart.len()).sum();

    debug!(
        "Spike trace parsed: {} PCs with register writes (total {}), {} PCs with memory writes (total {}), {} exception events",
        register_writes.len(),
        total_reg_writes,
        memory_writes.len(),
        total_mem_writes,
        total_exceptions
    );

    Ok(SpikeTrace {
        register_writes,
        memory_writes,
        exceptions,
    })
}

impl SpikeExecutor {
    pub fn execute<T: AsRef<Path>>(
        &self,
        run_folder: T,
        per_hart: PerHartAsm<'_>,
    ) -> Result<ExecutionOutput, SpikeError> {
        let total_lines: usize = per_hart.iter().map(|(_, lines)| lines.len()).sum();
        info!(
            "Executing Spike run for {} testcase instructions across {} hart(s)",
            total_lines,
            per_hart.len()
        );

        let run_dir = run_folder.as_ref().to_owned();
        info!("Using Spike run directory {}", run_dir.display());

        let run_result = self.compile_and_run_with_dir(&run_dir, per_hart)?;

        // 解析 Spike 日志（spike 输出所有trace到stderr）
        let trace = {
            let file = std::fs::File::open(&run_result.stderr_path).map_err(|e| {
                SpikeError::from(ProcessError::LogFileOpenFailed {
                    path: run_result.stderr_path.clone(),
                    source: e,
                })
            })?;
            let reader = BufReader::new(file);
            parse_spike_log(reader)?
        };

        let hart_count = per_hart.len().max(1) as u32;

        let perf = extract_perf_metrics_from_spike_trace(
            self.config.isa_base,
            self.config.core_id.clone(),
            &run_result.build.disassembly_file,
            &trace,
            &run_result.stdout,
            &run_result.stderr,
        );
        let total_exceptions: usize = trace
            .exceptions
            .values()
            .map(|per_hart| per_hart.len())
            .sum();

        info!(
            "Spike run complete: {} register writes, {} memory writes, {} exceptions",
            trace.register_writes.len(),
            trace.memory_writes.len(),
            total_exceptions
        );

        let mut per_hart_out = BTreeMap::new();
        for (hart_id, hart_insts) in per_hart.iter().copied() {
            let tracer = Tracer::new_with_section_labels(
                hart_insts,
                &run_result.build.disassembly_file,
                spike_section_labels_for_hart(hart_count, hart_id),
                self.config.isa_base,
            )
            .map_err(|e| {
                SpikeError::from(ElfError::DumpLoadFailed {
                    path: run_result.build.disassembly_file.clone(),
                    source: e,
                })
            })?;
            let hart_output = build_execution_output(&trace, hart_id, &tracer)?;
            per_hart_out.insert(hart_id, hart_output);
        }

        let mut out = ExecutionOutput::empty(self.config.core_id.clone(), self.config.isa_base);
        out.per_hart = per_hart_out;
        out.perf = perf;
        Ok(out)
    }

    /// Execute with streaming: compile synchronously, then read Spike stderr
    /// line-by-line and emit per-instruction events as instructions complete.
    ///
    /// The `emit` callback receives `InstructionEvent`s.  Returning `false` from
    /// `emit` signals early termination — Spike is killed and the method returns
    /// immediately.
    pub fn execute_streaming<T: AsRef<Path>>(
        &self,
        run_folder: T,
        per_hart: PerHartAsm<'_>,
        typed_sequences: Option<PerHartTypedSequences<'_>>,
        emit: &mut dyn FnMut(riscv_fuzz_plugin_api::streaming::InstructionEvent) -> bool,
    ) {
        use riscv_fuzz_plugin_api::streaming::InstructionEvent;

        let run_dir = run_folder.as_ref().to_owned();

        // 1. Compile ELF and build tracers — synchronous, fast.
        let (run_result, tracers) =
            match self.compile_for_streaming(&run_dir, per_hart, typed_sequences) {
                Ok(v) => v,
                Err(e) => {
                    emit(InstructionEvent::Failed {
                        error: riscv_fuzz_plugin_api::ImplExecutionError::from(e),
                    });
                    return;
                }
            };

        // 2. Launch Spike with stderr piped.
        let hart_count = per_hart.len().max(1) as u32;
        let extensions = self.config.resolved_extensions();
        let isa_string = match &extensions {
            IsaExtensions::Rv32(ext) => super::march::isa_from_rv32_extensions(ext),
            IsaExtensions::Rv64(ext) => super::march::isa_from_rv64_extensions(ext),
        };
        let isa_string = match isa_string {
            Ok(s) => s,
            Err(e) => {
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::Generic(e.to_string()),
                });
                return;
            }
        };

        let spike_binary = match self.config.spike_binary() {
            Ok(b) => b,
            Err(e) => {
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::from(e),
                });
                return;
            }
        };

        if !is_command_available(&spike_binary) {
            emit(InstructionEvent::Failed {
                error: riscv_fuzz_plugin_api::ImplExecutionError::Other(Box::new(
                    SpikeError::BinaryNotFound {
                        path: spike_binary.display().to_string(),
                    },
                )),
            });
            return;
        }

        let effective_isa = self.config.effective_isa_string(&isa_string);
        let mut cmd = std::process::Command::new(&spike_binary);
        cmd.arg(format!("-p{}", hart_count.max(1)));
        cmd.arg(format!("--isa={}", effective_isa.to_uppercase()));
        cmd.arg("--log-commits");
        cmd.arg("-l");
        cmd.args(&self.config.spike_args);

        let elf_arg = match resolve_wrapper_elf_path(&run_result.build.executable_file, &run_dir) {
            Ok(p) => p,
            Err(e) => {
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::from(e),
                });
                return;
            }
        };
        cmd.arg(elf_arg);
        cmd.current_dir(&run_dir);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        configure_command_process_group(&mut cmd);

        let command_display = format!("{:?}", cmd);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::from(
                        ProcessError::SpawnFailed {
                            command: command_display,
                            source: e,
                        },
                    ),
                });
                return;
            }
        };

        let stdout_pipe = match child.stdout.take() {
            Some(p) => p,
            None => {
                let _ = kill_child_process_group(&mut child);
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::Generic(
                        "failed to capture Spike stdout".to_string(),
                    ),
                });
                return;
            }
        };
        let stderr_pipe = match child.stderr.take() {
            Some(p) => p,
            None => {
                let _ = kill_child_process_group(&mut child);
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::Generic(
                        "failed to capture Spike stderr".to_string(),
                    ),
                });
                return;
            }
        };

        // 3. Drain stdout in a background thread (we don't need it for streaming,
        //    but must read it to prevent the pipe buffer from filling up).
        let stdout_handle = thread::spawn(move || -> Vec<u8> {
            use std::io::Read;
            let mut buf = Vec::new();
            let mut r = BufReader::new(stdout_pipe);
            let _ = r.read_to_end(&mut buf);
            buf
        });

        // 4. Parse stderr line-by-line, emit instructions.
        let timeout = self.config.timeout;

        // Per-PC buffered state — Spike can emit multiple lines per instruction.
        // We flush when we detect a new PC (instruction boundary).
        let mut pending_pc: Option<u64> = None;
        let mut pending_hart: u32 = 0;
        let mut pending_reg_writes: Vec<TraitRegisterValue> = Vec::new();
        let mut pending_mem_writes: Vec<(u64, MemoryWrite)> = Vec::new();
        let mut pending_exceptions: Vec<ExceptionEvent> = Vec::new();
        let mut stderr_lines: Vec<String> = Vec::new();
        let mut early_terminate = false;
        let strict_typed_mapping = typed_sequences.is_some();
        let parse_failed = std::cell::Cell::new(false);
        let fail_stream =
            |message: String,
             child: &mut std::process::Child,
             emit: &mut dyn FnMut(InstructionEvent) -> bool| {
                parse_failed.set(true);
                let _ = kill_child_process_group(child);
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::from(
                        LogParseError::InvalidLineFormat { line: message },
                    ),
                });
            };
        // Build regex references once.
        let reg_change_re = match compiled_regex(&REG_CHANGE_LINE_RE) {
            Ok(r) => r,
            Err(e) => {
                let _ = kill_child_process_group(&mut child);
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::from(e),
                });
                return;
            }
        };
        let reg_match_re = match compiled_regex(&REG_MATCH_RE) {
            Ok(r) => r,
            Err(e) => {
                let _ = kill_child_process_group(&mut child);
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::from(e),
                });
                return;
            }
        };
        let mem_re = match compiled_regex(&MEM_RE) {
            Ok(r) => r,
            Err(e) => {
                let _ = kill_child_process_group(&mut child);
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::from(e),
                });
                return;
            }
        };
        let exception_re = match compiled_regex(&EXCEPTION_RE) {
            Ok(r) => r,
            Err(e) => {
                let _ = kill_child_process_group(&mut child);
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::from(e),
                });
                return;
            }
        };

        let (stderr_tx, stderr_rx) = std::sync::mpsc::channel::<String>();
        let stderr_handle = thread::spawn(move || -> Result<Vec<String>, io::Error> {
            let stderr_reader = BufReader::new(stderr_pipe);
            let mut lines = Vec::new();
            for line_result in stderr_reader.lines() {
                let line = line_result?;
                if stderr_tx.send(line.clone()).is_err() {
                    break;
                }
                lines.push(line);
            }
            Ok(lines)
        });

        let mut timed_out_limit = None;
        let mut last_stderr_activity_at = Instant::now();

        'line_loop: loop {
            let line = match stderr_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(line) => {
                    last_stderr_activity_at = Instant::now();
                    line
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if let Some(limit) = timeout
                        && last_stderr_activity_at.elapsed() >= limit
                    {
                        timed_out_limit = Some(limit);
                        let _ = kill_child_process_group(&mut child);
                        break 'line_loop;
                    }
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break 'line_loop,
            };
            let trimmed = line.trim();
            stderr_lines.push(line.clone());

            // Check for exception line.
            if let Some(cap) = exception_re.captures(trimmed) {
                let Some(hart_match) = cap.name("hart") else {
                    fail_stream(
                        format!("Spike exception line missing hart: {trimmed}"),
                        &mut child,
                        emit,
                    );
                    break 'line_loop;
                };
                let hart: u32 = match hart_match.as_str().parse() {
                    Ok(hart) => hart,
                    Err(error) => {
                        fail_stream(
                            format!(
                                "Spike exception line has invalid hart '{}': {error}: {trimmed}",
                                hart_match.as_str()
                            ),
                            &mut child,
                            emit,
                        );
                        break 'line_loop;
                    }
                };
                let Some(kind_match) = cap.name("kind") else {
                    fail_stream(
                        format!("Spike exception line missing kind: {trimmed}"),
                        &mut child,
                        emit,
                    );
                    break 'line_loop;
                };
                let kind = kind_match.as_str().to_string();
                let Some(pc_match) = cap.name("pc") else {
                    fail_stream(
                        format!("Spike exception line missing pc: {trimmed}"),
                        &mut child,
                        emit,
                    );
                    break 'line_loop;
                };
                let pc = match u64::from_str_radix(pc_match.as_str(), 16) {
                    Ok(pc) => pc,
                    Err(error) => {
                        fail_stream(
                            format!(
                                "Spike exception line has invalid pc '{}': {error}: {trimmed}",
                                pc_match.as_str()
                            ),
                            &mut child,
                            emit,
                        );
                        break 'line_loop;
                    }
                };
                let mcause = if let Some(mcause_match) = cap.name("mcause") {
                    match u64::from_str_radix(mcause_match.as_str(), 16) {
                        Ok(mcause) => Some(mcause),
                        Err(error) => {
                            fail_stream(
                                format!(
                                    "Spike exception line has invalid mcause '{}': {error}: {trimmed}",
                                    mcause_match.as_str()
                                ),
                                &mut child,
                                emit,
                            );
                            break 'line_loop;
                        }
                    }
                } else {
                    None
                };

                let is_new_instruction = pending_pc.is_none_or(|p| p != pc || pending_hart != hart);
                if is_new_instruction {
                    if let Some(prev_pc) = pending_pc {
                        let keep = match emit_streamed_pc_row(
                            StreamedPcRow {
                                pc: prev_pc,
                                hart_id: pending_hart,
                                reg_writes: &pending_reg_writes,
                                mem_writes: &pending_mem_writes,
                                exceptions: &pending_exceptions,
                                tracers: &tracers,
                                strict_typed_mapping,
                            },
                            emit,
                        ) {
                            Ok(keep) => keep,
                            Err(message) => {
                                fail_stream(message, &mut child, emit);
                                break 'line_loop;
                            }
                        };
                        if !keep {
                            early_terminate = true;
                            break 'line_loop;
                        }
                    }
                    pending_pc = Some(pc);
                    pending_hart = hart;
                    pending_reg_writes.clear();
                    pending_mem_writes.clear();
                    pending_exceptions.clear();
                }

                if let Some(existing) = pending_exceptions.first() {
                    let detail = if existing.cause == kind && existing.mcause == mcause {
                        format!(
                            "duplicate Spike exception for hart={hart} pc=0x{pc:x}: cause='{kind}' mcause={}",
                            format_optional_exception_mcause(mcause)
                        )
                    } else {
                        format!(
                            "conflicting Spike exception for hart={hart} pc=0x{pc:x}: prev(cause='{}', mcause={}) new(cause='{}', mcause={})",
                            existing.cause,
                            format_optional_exception_mcause(existing.mcause),
                            kind,
                            format_optional_exception_mcause(mcause)
                        )
                    };
                    fail_stream(detail, &mut child, emit);
                    break 'line_loop;
                }

                pending_exceptions.push(ExceptionEvent {
                    hart,
                    cause: kind,
                    pc,
                    mcause,
                });
                continue;
            }

            // Check for commit line (register/memory writes).
            if let Some(line_cap) = reg_change_re.captures(trimmed) {
                let Some(hart_match) = line_cap.name("hart") else {
                    fail_stream(
                        format!("Spike commit line missing hart: {trimmed}"),
                        &mut child,
                        emit,
                    );
                    break 'line_loop;
                };
                let hart: u32 = match hart_match.as_str().parse() {
                    Ok(hart) => hart,
                    Err(error) => {
                        fail_stream(
                            format!(
                                "Spike commit line has invalid hart '{}': {error}: {trimmed}",
                                hart_match.as_str()
                            ),
                            &mut child,
                            emit,
                        );
                        break 'line_loop;
                    }
                };
                let Some(pc_match) = line_cap.name("pc") else {
                    fail_stream(
                        format!("Spike commit line missing pc: {trimmed}"),
                        &mut child,
                        emit,
                    );
                    break 'line_loop;
                };
                let pc: u64 = match u64::from_str_radix(pc_match.as_str(), 16) {
                    Ok(pc) => pc,
                    Err(error) => {
                        fail_stream(
                            format!(
                                "Spike commit line has invalid pc '{}': {error}: {trimmed}",
                                pc_match.as_str()
                            ),
                            &mut child,
                            emit,
                        );
                        break 'line_loop;
                    }
                };

                // Detect instruction boundary: new (hart, pc) pair.
                let is_new_instruction = pending_pc.is_none_or(|p| p != pc || pending_hart != hart);
                if is_new_instruction {
                    // Finalize the previous PC if any; the source run itself
                    // is emitted only after the log advances to a later source.
                    if let Some(prev_pc) = pending_pc {
                        let keep = match emit_streamed_pc_row(
                            StreamedPcRow {
                                pc: prev_pc,
                                hart_id: pending_hart,
                                reg_writes: &pending_reg_writes,
                                mem_writes: &pending_mem_writes,
                                exceptions: &pending_exceptions,
                                tracers: &tracers,
                                strict_typed_mapping,
                            },
                            emit,
                        ) {
                            Ok(keep) => keep,
                            Err(message) => {
                                fail_stream(message, &mut child, emit);
                                break 'line_loop;
                            }
                        };
                        if !keep {
                            early_terminate = true;
                            break 'line_loop;
                        }
                    }
                    pending_pc = Some(pc);
                    pending_hart = hart;
                    pending_reg_writes.clear();
                    pending_mem_writes.clear();
                    pending_exceptions.clear();
                }

                // Extract register writes from this commit line.
                for reg_cap in reg_match_re.captures_iter(trimmed) {
                    let Some(prefix_match) = reg_cap.name("prefix") else {
                        fail_stream(
                            format!("Spike register write missing prefix: {trimmed}"),
                            &mut child,
                            emit,
                        );
                        break 'line_loop;
                    };
                    let prefix = prefix_match.as_str();
                    let Some(reg_match) = reg_cap.name("reg") else {
                        fail_stream(
                            format!("Spike register write missing register index: {trimmed}"),
                            &mut child,
                            emit,
                        );
                        break 'line_loop;
                    };
                    let reg_idx: u8 = match reg_match.as_str().parse() {
                        Ok(reg_idx) => reg_idx,
                        Err(error) => {
                            fail_stream(
                                format!(
                                    "Spike register write has invalid register index '{}': {error}: {trimmed}",
                                    reg_match.as_str()
                                ),
                                &mut child,
                                emit,
                            );
                            break 'line_loop;
                        }
                    };
                    let Some(value_match) = reg_cap.name("val") else {
                        fail_stream(
                            format!("Spike register write missing value: {trimmed}"),
                            &mut child,
                            emit,
                        );
                        break 'line_loop;
                    };
                    let value: u64 = match u64::from_str_radix(value_match.as_str(), 16) {
                        Ok(value) => value,
                        Err(error) => {
                            fail_stream(
                                format!(
                                    "Spike register write has invalid value '{}': {error}: {trimmed}",
                                    value_match.as_str()
                                ),
                                &mut child,
                                emit,
                            );
                            break 'line_loop;
                        }
                    };
                    let reg_name = format!("{}{}", prefix, reg_idx);
                    if let Some(existing) = pending_reg_writes.iter().find(|r| r.name == reg_name) {
                        fail_stream(
                            format!(
                                "Spike duplicate register write for hart={hart} pc=0x{pc:x} {reg_name}: 0x{:x} and 0x{value:x}",
                                existing.value
                            ),
                            &mut child,
                            emit,
                        );
                        break 'line_loop;
                    }
                    pending_reg_writes.push(TraitRegisterValue {
                        name: reg_name,
                        value,
                    });
                }

                // Extract memory writes from this commit line.
                for mem_cap in mem_re.captures_iter(trimmed) {
                    let Some(address_match) = mem_cap.name("addr") else {
                        fail_stream(
                            format!("Spike memory write missing address: {trimmed}"),
                            &mut child,
                            emit,
                        );
                        break 'line_loop;
                    };
                    let address: u64 = match u64::from_str_radix(address_match.as_str(), 16) {
                        Ok(address) => address,
                        Err(error) => {
                            fail_stream(
                                format!(
                                    "Spike memory write has invalid address '{}': {error}: {trimmed}",
                                    address_match.as_str()
                                ),
                                &mut child,
                                emit,
                            );
                            break 'line_loop;
                        }
                    };
                    if let Some(data_match) = mem_cap.name("data") {
                        let digits = data_match.as_str();
                        if digits.is_empty() {
                            fail_stream(
                                format!(
                                    "Spike memory write has empty data for hart={hart} pc=0x{pc:x} addr=0x{address:x}: {trimmed}"
                                ),
                                &mut child,
                                emit,
                            );
                            break 'line_loop;
                        }
                        let byte_width = digits.len().div_ceil(2);
                        if byte_width > 8 {
                            fail_stream(
                                format!(
                                    "Spike memory write width {byte_width} exceeds 8 bytes for hart={hart} pc=0x{pc:x}: {trimmed}"
                                ),
                                &mut child,
                                emit,
                            );
                            break 'line_loop;
                        }
                        let store_value = match u64::from_str_radix(digits, 16) {
                            Ok(store_value) => store_value,
                            Err(error) => {
                                fail_stream(
                                    format!(
                                        "Spike memory write has invalid data '{digits}': {error}: {trimmed}"
                                    ),
                                    &mut child,
                                    emit,
                                );
                                break 'line_loop;
                            }
                        };
                        if let Some((_, existing)) = pending_mem_writes
                            .iter()
                            .find(|(_, m)| m.address == address)
                        {
                            fail_stream(
                                format!(
                                    "Spike duplicate memory write for hart={hart} pc=0x{pc:x} addr=0x{address:x}: 0x{:x} and 0x{store_value:x}",
                                    existing.value
                                ),
                                &mut child,
                                emit,
                            );
                            break 'line_loop;
                        }
                        pending_mem_writes.push((
                            pc,
                            MemoryWrite {
                                hart,
                                pc,
                                address,
                                value: store_value,
                                width: byte_width,
                            },
                        ));
                    }
                }
            }
        }

        if parse_failed.get() {
            let _ = child.wait();
            let _ = stdout_handle.join();
            let _ = stderr_handle.join();
            return;
        }

        if early_terminate {
            let _ = kill_child_process_group(&mut child);
            let _ = child.wait();
            let _ = stdout_handle.join();
            let _ = stderr_handle.join();
            return;
        }

        if let Some(limit) = timed_out_limit {
            let _ = child.wait();
            let stdout_bytes = stdout_handle.join().unwrap_or_default();
            if let Ok(Ok(joined_stderr_lines)) = stderr_handle.join()
                && joined_stderr_lines.len() > stderr_lines.len()
            {
                stderr_lines = joined_stderr_lines;
            }
            let stdout_path = run_dir.join("spike_stdout.log");
            let stderr_path = run_dir.join("spike_stderr.log");
            let _ = fs::write(&stdout_path, &stdout_bytes);
            let _ = fs::write(&stderr_path, stderr_lines.join("\n"));
            emit(InstructionEvent::Failed {
                error: riscv_fuzz_plugin_api::ImplExecutionError::from(ProcessError::TimedOut {
                    command: command_display.clone(),
                    timeout: limit,
                }),
            });
            return;
        }

        // Flush final pending instruction.
        if let Some(pc) = pending_pc {
            let keep = match emit_streamed_pc_row(
                StreamedPcRow {
                    pc,
                    hart_id: pending_hart,
                    reg_writes: &pending_reg_writes,
                    mem_writes: &pending_mem_writes,
                    exceptions: &pending_exceptions,
                    tracers: &tracers,
                    strict_typed_mapping,
                },
                emit,
            ) {
                Ok(keep) => keep,
                Err(message) => {
                    fail_stream(message, &mut child, emit);
                    return;
                }
            };
            if !keep {
                let _ = kill_child_process_group(&mut child);
                drop(stdout_handle);
                return;
            }
        }
        // Wait for process to exit.
        let status = match child.wait() {
            Ok(s) => s,
            Err(e) => {
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::Generic(e.to_string()),
                });
                return;
            }
        };

        let stdout_bytes = stdout_handle.join().unwrap_or_default();
        match stderr_handle.join() {
            Ok(Ok(joined_stderr_lines)) => {
                if joined_stderr_lines.len() > stderr_lines.len() {
                    stderr_lines = joined_stderr_lines;
                }
            }
            Ok(Err(e)) => {
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::Generic(e.to_string()),
                });
                return;
            }
            Err(_) => {
                emit(InstructionEvent::Failed {
                    error: riscv_fuzz_plugin_api::ImplExecutionError::Generic(
                        "Spike stderr reader panicked".to_string(),
                    ),
                });
                return;
            }
        }
        let stdout = String::from_utf8_lossy(&stdout_bytes).to_string();

        // Persist logs.
        let stdout_path = run_dir.join("spike_stdout.log");
        let stderr_path = run_dir.join("spike_stderr.log");
        let _ = fs::write(&stdout_path, &stdout_bytes);
        let _ = fs::write(&stderr_path, stderr_lines.join("\n"));

        if !status.success() {
            emit(InstructionEvent::Failed {
                error: riscv_fuzz_plugin_api::ImplExecutionError::from(
                    ProcessError::ProcessFailed {
                        command: command_display,
                        stderr: stderr_lines.join("\n"),
                    },
                ),
            });
            return;
        }

        let perf = PerfMetrics {
            global_cycles: parse_global_cycles_from_text(&stdout),
            ..Default::default()
        };

        emit(InstructionEvent::Completed { perf });
    }

    /// Compile ELF and build per-hart `Tracer` objects without running Spike.
    fn compile_for_streaming(
        &self,
        run_dir: &Path,
        per_hart: PerHartAsm<'_>,
        typed_sequences: Option<PerHartTypedSequences<'_>>,
    ) -> Result<
        (
            SpikeRunResultCompileOnly,
            std::collections::HashMap<u32, Tracer>,
        ),
        SpikeError,
    > {
        self.ensure_work_dir(run_dir)?;
        let extensions = self.config.resolved_extensions();
        let program = self.build_program(per_hart)?;
        let asm_path = self.write_program(run_dir, &program)?;
        let build =
            build_elf_with_extensions(&asm_path, &extensions, linker_script()).map_err(|err| {
                error!(
                    "Failed to build ELF for Spike streaming (asm={}, isa={:?}): {:#}",
                    asm_path.display(),
                    self.config.isa_base,
                    err
                );
                err
            })?;

        let hart_count = per_hart.len().max(1) as u32;
        let mut tracers = std::collections::HashMap::new();
        for (hart_id, hart_insts) in per_hart.iter().copied() {
            let labels = spike_section_labels_for_hart(hart_count, hart_id);
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
                .map_err(|e| {
                    SpikeError::from(ElfError::DumpLoadFailed {
                        path: build.disassembly_file.clone(),
                        source: e,
                    })
                })?;
            tracers.insert(hart_id, tracer);
        }

        Ok((SpikeRunResultCompileOnly { build }, tracers))
    }
}

/// Minimal compile result used by streaming (no process output yet).
struct SpikeRunResultCompileOnly {
    build: riscv_fuzz_toolchain::build_elf::ElfBuildResult,
}

fn extract_perf_metrics_from_spike_trace(
    _isa_base: ISABase,
    _riscv_impl: String,
    _dump_path: &Path,
    _trace: &SpikeTrace,
    stdout: &str,
    _stderr: &str,
) -> PerfMetrics {
    PerfMetrics {
        global_cycles: parse_global_cycles_from_text(stdout),
        ..Default::default()
    }
}

/// Build ExecutionOutput from Spike trace
fn build_execution_output(
    trace: &SpikeTrace,
    target_hart_id: u32,
    tracer: &Tracer,
) -> Result<HartExecutionOutput, SpikeError> {
    let observed_pcs = trace
        .register_writes
        .iter()
        .filter(|(_, writes)| writes.iter().any(|write| write.hart == target_hart_id))
        .map(|(pc, _)| *pc)
        .chain(
            trace
                .memory_writes
                .iter()
                .filter(|(_, writes)| writes.iter().any(|write| write.hart == target_hart_id))
                .map(|(pc, _)| *pc),
        )
        .chain(
            trace
                .exceptions
                .get(&target_hart_id)
                .into_iter()
                .flat_map(|per_hart| per_hart.keys().copied()),
        )
        .collect::<Vec<_>>();

    Ok(build_raw_hart_execution_output(
        tracer,
        target_hart_id,
        observed_pcs,
        |pc| {
            let mut regs = trace
                .register_writes
                .get(&pc)
                .into_iter()
                .flat_map(|writes| writes.iter())
                .filter(|write| write.hart == target_hart_id)
                .map(|write| TraitRegisterValue {
                    name: write.register.clone(),
                    value: write.value,
                })
                .collect::<Vec<_>>();
            regs.sort_by(|a, b| compare_register_names(&a.name, &b.name));
            regs
        },
        |pc| {
            let mut mems = trace
                .memory_writes
                .get(&pc)
                .into_iter()
                .flat_map(|writes| writes.iter())
                .filter(|write| write.hart == target_hart_id)
                .flat_map(store_bytes_from_write)
                .map(|(addr, value)| TraitMemValue { addr, value })
                .collect::<Vec<_>>();
            mems.sort_by_key(|mem| mem.addr);
            mems
        },
        |pc| {
            trace
                .exceptions
                .get(&target_hart_id)
                .and_then(|per_hart| per_hart.get(&pc))
                .filter(|event| !event.cause.contains("interrupt"))
                .map(|event| {
                    vec![if let Some(code) = event.mcause {
                        ExceptionCause::from_code(code, event.cause.clone())
                    } else {
                        warn!(
                            "Spike: unknown exception cause '{}' at PC 0x{:x}",
                            event.cause, event.pc
                        );
                        ExceptionCause::unknown(event.cause.clone())
                    }]
                })
                .unwrap_or_default()
        },
        |_pc| InstructionTiming::default(),
    ))
}

fn store_bytes_from_write(write: &MemoryWrite) -> Vec<(u64, u8)> {
    if write.width == 0 {
        return Vec::new();
    }

    let width = write.width.min(8);
    let data = write.value.to_le_bytes();
    (0..width)
        .map(|offset| (write.address + offset as u64, data[offset]))
        .collect()
}

fn spike_section_labels_for_hart(
    harts: u32,
    hart_id: u32,
) -> &'static [(&'static str, &'static str)] {
    match (harts > 1, hart_id) {
        (true, 0) => &[("hart0_testcase_code:", "hart0_post_wait:")],
        (true, 1) => &[("hart1_testcase_code:", "hart1_post_wait:")],
        _ => &[("testcase_code:", "exit:")],
    }
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

fn compiled_regex(lazy: &Lazy<Result<Regex, LogParseError>>) -> Result<&Regex, SpikeError> {
    match lazy.as_ref() {
        Ok(regex) => Ok(regex),
        Err(LogParseError::RegexCompilationFailed { pattern, source }) => {
            Err(SpikeError::from(LogParseError::RegexCompilationFailed {
                pattern: pattern.clone(),
                source: source.clone(),
            }))
        }
        Err(other) => Err(SpikeError::from(LogParseError::PatternMatchFailed {
            line: other.to_string(),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riscv_fuzz_plugin_api::{IsaExtensions, UNKNOWN_EXCEPTION_CODE};
    use riscv_fuzz_toolchain::trace_instruction::isa_base_to_decode_ctx;
    use riscv_fuzz_toolchain::tracer::{TracedInstruction, Tracer};
    use riscv_instruction::{AsmDecodeWithContext, RV32Extensions, RiscvInstruction};

    fn source_instruction_tracer(instructions: &[(u64, &str)]) -> Tracer {
        let ctx = isa_base_to_decode_ctx(ISABase::Rv64);
        Tracer::from_traced_instructions(
            instructions
                .iter()
                .enumerate()
                .map(|(idx, (pc, instruction))| TracedInstruction {
                    pc: *pc,
                    disassembly: instruction.to_string(),
                    parsed_disassembly: RiscvInstruction::decode_asm_with_context(instruction, ctx)
                        .ok(),
                    source_line: Some(instruction.to_string()),
                    trace_inst_local_index: Some(idx),
                    source_instruction: Some(
                        RiscvInstruction::decode_asm_with_context(instruction, ctx).unwrap(),
                    ),
                })
                .collect(),
        )
    }

    fn typed_sequence_tracer(entries: &[(u64, &str, usize)]) -> Tracer {
        let ctx = isa_base_to_decode_ctx(ISABase::Rv64);
        Tracer::from_traced_instructions(
            entries
                .iter()
                .map(|(pc, instruction, source_index)| {
                    let parsed =
                        RiscvInstruction::decode_asm_with_context(instruction, ctx).unwrap();
                    TracedInstruction {
                        pc: *pc,
                        disassembly: instruction.to_string(),
                        parsed_disassembly: Some(parsed.clone()),
                        source_line: Some(instruction.to_string()),
                        trace_inst_local_index: Some(*source_index),
                        source_instruction: Some(parsed),
                    }
                })
                .collect(),
        )
    }

    #[test]
    fn strict_typed_stream_trace_mapping_requires_trace_inst_local_index() {
        let ctx = isa_base_to_decode_ctx(ISABase::Rv64);
        let parsed = RiscvInstruction::decode_asm_with_context("addi x0, x0, 0", ctx).unwrap();
        let tracer = Tracer::from_traced_instructions(vec![TracedInstruction {
            pc: 0x1000,
            disassembly: "addi x0, x0, 0".to_string(),
            parsed_disassembly: Some(parsed.clone()),
            source_line: Some("addi x0, x0, 0".to_string()),
            trace_inst_local_index: None,
            source_instruction: Some(parsed),
        }]);
        let tracers = HashMap::from([(0, tracer)]);

        let err = resolve_stream_trace_inst_identity(&tracers, 0, 0x1000, true)
            .expect_err("typed trace mapping must reject missing trace instruction local index");
        assert!(err.contains("missing trace instruction local index"));
    }

    #[test]
    fn strict_typed_stream_trace_mapping_skips_framework_rows_without_trace_inst_id() {
        let ctx = isa_base_to_decode_ctx(ISABase::Rv64);
        let parsed = RiscvInstruction::decode_asm_with_context("addi x0, x0, 0", ctx).unwrap();
        let tracer = Tracer::from_traced_instructions(vec![TracedInstruction {
            pc: 0x1000,
            disassembly: "addi x0, x0, 0".to_string(),
            parsed_disassembly: Some(parsed),
            source_line: None,
            trace_inst_local_index: None,
            source_instruction: None,
        }]);
        let tracers = HashMap::from([(0, tracer)]);

        let identity = resolve_stream_trace_inst_identity(&tracers, 0, 0x1000, true)
            .expect("framework row without trace id should be skipped");
        assert!(identity.is_none());
    }

    #[test]
    fn strict_typed_stream_trace_mapping_requires_decoded_instruction() {
        let tracer = Tracer::from_traced_instructions(vec![TracedInstruction {
            pc: 0x1000,
            disassembly: "addi x0, x0, 0".to_string(),
            parsed_disassembly: None,
            source_line: Some("addi x0, x0, 0".to_string()),
            trace_inst_local_index: Some(0),
            source_instruction: None,
        }]);
        let tracers = HashMap::from([(0, tracer)]);

        let err = resolve_stream_trace_inst_identity(&tracers, 0, 0x1000, true)
            .expect_err("typed trace mapping must reject missing decoded instruction");
        assert!(err.contains("missing decoded instruction"));
    }

    #[test]
    fn executor_config_uses_extension_override() {
        let mut config =
            ExecutorConfig::new(PathBuf::from("/tmp"), ISABase::Rv32, "spike".to_string());
        let override_map = IsaExtensions::rv32(vec![RV32Extensions::I, RV32Extensions::F]);
        config.set_extension_override(override_map.clone());
        let resolved = config.resolved_extensions();

        assert_eq!(resolved, override_map);
    }

    #[test]
    fn executor_config_appends_isa_suffix_with_or_without_separator() {
        let mut config =
            ExecutorConfig::new(PathBuf::from("/tmp"), ISABase::Rv64, "spike".to_string());
        config.isa_suffix = Some("_zvl4096b".to_string());
        assert_eq!(config.effective_isa_string("rv64gcv"), "rv64gcv_zvl4096b");

        config.isa_suffix = Some("zvl4096b".to_string());
        assert_eq!(config.effective_isa_string("rv64gcv"), "rv64gcv_zvl4096b");
    }

    #[test]
    fn perf_cycles_only_use_stdout() {
        let perf = extract_perf_metrics_from_spike_trace(
            ISABase::Rv32,
            "spike".to_string(),
            Path::new("/tmp"),
            &SpikeTrace::default(),
            "",
            "Cycles: 99",
        );

        assert_eq!(perf.global_cycles, None);

        let perf = extract_perf_metrics_from_spike_trace(
            ISABase::Rv32,
            "spike".to_string(),
            Path::new("/tmp"),
            &SpikeTrace::default(),
            "Cycles: 77",
            "",
        );

        assert_eq!(perf.global_cycles, Some(77));
    }

    #[test]
    fn build_execution_output_filters_trace_by_target_hart() {
        let mut trace = SpikeTrace::default();
        trace.register_writes.insert(
            0x1000,
            vec![
                RegisterWrite {
                    hart: 0,
                    index: 5,
                    pc: 0x1000,
                    register: "x5".to_string(),
                    value: 5,
                },
                RegisterWrite {
                    hart: 1,
                    index: 6,
                    pc: 0x1000,
                    register: "x6".to_string(),
                    value: 6,
                },
            ],
        );

        let tracer = source_instruction_tracer(&[(0x1000, "addi x0, x0, 0")]);

        let hart0 = build_execution_output(&trace, 0, &tracer).expect("hart0 output");
        let hart1 = build_execution_output(&trace, 1, &tracer).expect("hart1 output");

        assert_eq!(hart0.instructions.len(), 1);
        assert_eq!(hart1.instructions.len(), 1);
        let hart0_main = &hart0.instructions[0];
        let hart1_main = &hart1.instructions[0];
        assert_eq!(hart0_main.register_writes.len(), 1);
        assert_eq!(hart1_main.register_writes.len(), 1);
        assert_eq!(hart0_main.register_writes[0].name, "x5");
        assert_eq!(hart1_main.register_writes[0].name, "x6");
    }

    #[test]
    fn parse_spike_log_rejects_duplicate_exception_for_same_hart_and_pc() {
        let log = "\
core   0: exception trap_illegal_instruction, epc 0x0000000080000022
core   0: exception trap_illegal_instruction, epc 0x0000000080000022
";

        let err = parse_spike_log(BufReader::new(log.as_bytes()))
            .expect_err("duplicate exception should fail");
        let message = err.to_string();
        assert!(message.contains("duplicate Spike exception for hart=0 pc=0x80000022"));
    }

    #[test]
    fn parse_spike_log_keeps_same_pc_exceptions_separate_per_hart() {
        let log = "\
core   0: exception trap_illegal_instruction, epc 0x0000000080000022
core   1: exception trap_store_access_fault, epc 0x0000000080000022
";

        let trace = parse_spike_log(BufReader::new(log.as_bytes())).expect("trace should parse");

        let hart0 = trace
            .exceptions
            .get(&0)
            .and_then(|events| events.get(&0x8000_0022))
            .expect("hart0 exception");
        let hart1 = trace
            .exceptions
            .get(&1)
            .and_then(|events| events.get(&0x8000_0022))
            .expect("hart1 exception");
        assert_eq!(hart0.cause, "trap_illegal_instruction");
        assert_eq!(hart1.cause, "trap_store_access_fault");
    }

    #[test]
    fn build_execution_output_leaves_unknown_exception_without_inferred_code() {
        let mut trace = SpikeTrace::default();
        trace.exceptions.insert(
            0,
            HashMap::from([(
                0x1000,
                ExceptionEvent {
                    hart: 0,
                    cause: "trap_illegal_instruction".to_string(),
                    pc: 0x1000,
                    mcause: None,
                },
            )]),
        );

        let tracer = source_instruction_tracer(&[(0x1000, "addi x0, x0, 0")]);
        let output = build_execution_output(&trace, 0, &tracer).expect("output");
        let main = &output.instructions[0];

        assert_eq!(main.exceptions.len(), 1);
        assert_eq!(main.exceptions[0].code, UNKNOWN_EXCEPTION_CODE);
        assert_eq!(main.exceptions[0].raw, "trap_illegal_instruction");
    }

    #[test]
    fn emit_streamed_pc_row_keeps_interleaved_dual_hart_trace_ids() {
        let tracers = HashMap::from([
            (
                0,
                typed_sequence_tracer(&[
                    (0x1000, "addi x8, x0, 0", 0),
                    (0x1004, "add x16, x19, x8", 1),
                ]),
            ),
            (1, typed_sequence_tracer(&[(0x2000, "addi x5, x5, 1", 0)])),
        ]);

        let mut events = Vec::new();

        assert!(
            emit_streamed_pc_row(
                StreamedPcRow {
                    pc: 0x1000,
                    hart_id: 0,
                    reg_writes: &[],
                    mem_writes: &[],
                    exceptions: &[],
                    tracers: &tracers,
                    strict_typed_mapping: true,
                },
                &mut |event| {
                    events.push(event);
                    true
                },
            )
            .expect("append hart0 pre")
        );
        assert_eq!(events.len(), 1);

        assert!(
            emit_streamed_pc_row(
                StreamedPcRow {
                    pc: 0x2000,
                    hart_id: 1,
                    reg_writes: &[],
                    mem_writes: &[],
                    exceptions: &[],
                    tracers: &tracers,
                    strict_typed_mapping: true,
                },
                &mut |event| {
                    events.push(event);
                    true
                },
            )
            .expect("append hart1 main")
        );
        assert_eq!(events.len(), 2);

        assert!(
            emit_streamed_pc_row(
                StreamedPcRow {
                    pc: 0x1004,
                    hart_id: 0,
                    reg_writes: &[],
                    mem_writes: &[],
                    exceptions: &[],
                    tracers: &tracers,
                    strict_typed_mapping: true,
                },
                &mut |event| {
                    events.push(event);
                    true
                },
            )
            .expect("append hart0 main")
        );
        assert_eq!(events.len(), 3);

        match &events[0] {
            riscv_fuzz_plugin_api::streaming::InstructionEvent::Instruction {
                hart_id,
                instruction,
            } => {
                assert_eq!(*hart_id, 0);
                assert_eq!(
                    instruction.trace_inst_id,
                    Some(riscv_fuzz_plugin_api::TraceInstId::new(0, 0))
                );
            }
            other => panic!("unexpected first event: {other:?}"),
        }

        match &events[1] {
            riscv_fuzz_plugin_api::streaming::InstructionEvent::Instruction {
                hart_id,
                instruction,
            } => {
                assert_eq!(*hart_id, 1);
                assert_eq!(
                    instruction.trace_inst_id,
                    Some(riscv_fuzz_plugin_api::TraceInstId::new(1, 0))
                );
            }
            other => panic!("unexpected second event: {other:?}"),
        }

        match &events[2] {
            riscv_fuzz_plugin_api::streaming::InstructionEvent::Instruction {
                hart_id,
                instruction,
            } => {
                assert_eq!(*hart_id, 0);
                assert_eq!(
                    instruction.trace_inst_id,
                    Some(riscv_fuzz_plugin_api::TraceInstId::new(0, 1))
                );
            }
            other => panic!("unexpected third event: {other:?}"),
        }
    }
}
