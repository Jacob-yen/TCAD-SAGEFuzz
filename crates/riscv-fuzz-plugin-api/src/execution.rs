use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt;

use riscv_instruction::{RegisterOperandConfig, RiscvInstruction};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ISABase;
use crate::register_name::{canonicalize_register_name, parse_register_name};

#[derive(Debug, Error)]
pub enum NormalizeError {
    #[error("memory address 0x{addr:x} is below expected start 0x{start:x}")]
    MemoryBelowStart { addr: u64, start: u64 },

    #[error("invalid register name in execution output: {name}")]
    InvalidRegisterName { name: String },

    #[error("unsupported register name in execution output: {name}")]
    UnsupportedRegisterName { name: String },

    #[error(
        "RV32 integer register write for {name} has non-32-bit value 0x{value:016x}; refusing to truncate log data"
    )]
    Rv32IntegerValueOutOfRange { name: String, value: u64 },

    #[error("duplicate register write for {name} at pc 0x{pc:x}")]
    DuplicateRegisterWrite { pc: u64, name: String },

    #[error("duplicate memory byte write at pc 0x{pc:x} addr 0x{addr:x}")]
    DuplicateMemoryWrite { pc: u64, addr: u64 },
}

pub const UNKNOWN_EXCEPTION_CODE: u64 = u64::MAX;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExceptionCause {
    pub code: u64,
    pub raw: String,
}

impl ExceptionCause {
    pub fn from_code(code: u64, raw: impl Into<String>) -> Self {
        Self {
            code,
            raw: raw.into(),
        }
    }

    pub fn unknown(raw: impl Into<String>) -> Self {
        Self {
            code: UNKNOWN_EXCEPTION_CODE,
            raw: raw.into(),
        }
    }
}

impl fmt::Display for ExceptionCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.raw.trim().is_empty() {
            write!(f, "{:#x}", self.code)
        } else {
            write!(f, "{:#x} (raw: {})", self.code, self.raw)
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize)]
pub struct RegisterValue {
    pub name: String,
    pub value: u64,
}

#[derive(PartialEq, Eq, Debug, Clone, Serialize, Deserialize)]
pub struct MemValue {
    pub addr: u64,
    pub value: u8,
}

#[derive(PartialEq, Eq, Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstructionTiming {
    #[serde(default)]
    pub clk_start: Option<u64>,
    #[serde(default)]
    pub clk_end: Option<u64>,
    #[serde(default)]
    pub clk_span: Option<u64>,
}

impl InstructionTiming {
    pub fn new(clk_start: Option<u64>, clk_end: Option<u64>, clk_span: Option<u64>) -> Self {
        Self {
            clk_start,
            clk_end,
            clk_span,
        }
        .normalized()
    }

    pub fn normalized(mut self) -> Self {
        if self.clk_span.is_none()
            && let (Some(start), Some(end)) = (self.clk_start, self.clk_end)
            && end >= start
        {
            self.clk_span = Some(end - start + 1);
        }
        self
    }

    pub fn is_recorded(&self) -> bool {
        self.clk_start.is_some() || self.clk_end.is_some() || self.clk_span.is_some()
    }

    pub fn is_consistent(&self) -> bool {
        if let (Some(start), Some(end)) = (self.clk_start, self.clk_end)
            && end < start
        {
            return false;
        }
        if let (Some(start), Some(end), Some(span)) = (self.clk_start, self.clk_end, self.clk_span)
        {
            return end >= start && span == end - start + 1;
        }
        true
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Default, Serialize, Deserialize)]
pub struct HartPerfMetrics {
    #[serde(default)]
    pub mcycle_start: Option<u64>,
    #[serde(default)]
    pub mcycle_end: Option<u64>,
    #[serde(default)]
    pub minstret_start: Option<u64>,
    #[serde(default)]
    pub minstret_end: Option<u64>,
}

impl HartPerfMetrics {
    pub fn mcycle_delta(&self) -> Option<u64> {
        match (self.mcycle_start, self.mcycle_end) {
            (Some(start), Some(end)) => Some(end.wrapping_sub(start)),
            _ => None,
        }
    }

    pub fn minstret_delta(&self) -> Option<u64> {
        match (self.minstret_start, self.minstret_end) {
            (Some(start), Some(end)) => Some(end.wrapping_sub(start)),
            _ => None,
        }
    }

    pub fn cpi(&self) -> Option<f64> {
        let cycles = self.mcycle_delta()? as f64;
        let insts = self.minstret_delta()? as f64;
        if insts == 0.0 {
            None
        } else {
            Some(cycles / insts)
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerfMetrics {
    #[serde(default)]
    pub global_cycles: Option<u64>,
    #[serde(default)]
    pub per_hart: BTreeMap<u32, HartPerfMetrics>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Stable identity for one caller-supplied traceable instruction after the
/// input program has been flattened per hart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TraceInstId {
    pub hart_id: u32,
    pub local_index: usize,
}

impl TraceInstId {
    pub fn new(hart_id: u32, local_index: usize) -> Self {
        Self {
            hart_id,
            local_index,
        }
    }
}

/// One actually-executed asm instruction along with its observable side
/// effects on the implementation. `trace_inst_id` points directly back to the
/// caller-supplied traceable instruction that produced this row. Framework
/// rows such as prologue/trap-handler/coverage code use `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsmInstructionExecution {
    pub pc: u64,
    pub instruction: RiscvInstruction,
    #[serde(default)]
    pub source_annotation: Option<String>,
    #[serde(default)]
    pub trace_inst_id: Option<TraceInstId>,
    #[serde(default)]
    pub exceptions: Vec<ExceptionCause>,
    #[serde(default)]
    pub register_writes: Vec<RegisterValue>,
    #[serde(default)]
    pub memory_writes: Vec<MemValue>,
    #[serde(default)]
    pub timing: InstructionTiming,
}

/// Output for a single hart. The row order follows the implementation trace.
/// Testcase rows carry `trace_inst_id`; framework rows carry `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HartExecutionOutput {
    pub hart_id: u32,
    #[serde(default)]
    pub instructions: Vec<AsmInstructionExecution>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionOutput {
    #[serde(default)]
    pub per_hart: BTreeMap<u32, HartExecutionOutput>,
    pub core_id: String,
    pub isa_base: ISABase,
    #[serde(default)]
    pub perf: PerfMetrics,
}

impl ExecutionOutput {
    pub fn empty(core_id: impl Into<String>, isa_base: ISABase) -> Self {
        Self {
            per_hart: BTreeMap::new(),
            core_id: core_id.into(),
            isa_base,
            perf: PerfMetrics::default(),
        }
    }

    pub fn empty_for_harts(
        core_id: impl Into<String>,
        isa_base: ISABase,
        hart_ids: impl IntoIterator<Item = u32>,
    ) -> Self {
        let mut output = Self::empty(core_id, isa_base);
        output.per_hart = hart_ids
            .into_iter()
            .map(|hart_id| (hart_id, HartExecutionOutput::empty(hart_id)))
            .collect();
        output
    }

    pub fn instruction_count(&self) -> usize {
        self.per_hart
            .values()
            .map(HartExecutionOutput::instruction_count)
            .sum::<usize>()
    }

    pub fn normalize(
        &mut self,
        mem_range: (u64, u64),
        test_register_config: &RegisterOperandConfig,
        temp_register_num_range: (u8, u8),
    ) -> Result<(), NormalizeError> {
        for hart in self.per_hart.values_mut() {
            hart.normalize(
                self.isa_base,
                mem_range,
                test_register_config,
                temp_register_num_range,
            )?;
        }

        Ok(())
    }
}

fn normalize_register_writes(
    regs: &mut Vec<RegisterValue>,
    isa_base: ISABase,
    _test_register_config: &RegisterOperandConfig,
    _temp_min: u8,
    _temp_max: u8,
) -> Result<(), NormalizeError> {
    let mut retained = Vec::with_capacity(regs.len());
    for reg_val in std::mem::take(regs) {
        let parsed = parse_register_name(&reg_val.name).ok_or_else(|| {
            NormalizeError::InvalidRegisterName {
                name: reg_val.name.clone(),
            }
        })?;

        match parsed.bank {
            riscv_instruction::RegisterBank::Integer => {
                if isa_base == ISABase::Rv32 && reg_val.value > u32::MAX as u64 {
                    return Err(NormalizeError::Rv32IntegerValueOutOfRange {
                        name: reg_val.name,
                        value: reg_val.value,
                    });
                }
                retained.push(reg_val);
            }
            riscv_instruction::RegisterBank::FloatingPoint
            | riscv_instruction::RegisterBank::Vector => retained.push(reg_val),
        }
    }
    *regs = retained;
    Ok(())
}

fn normalize_memory_writes_with_policy(
    mems: &mut Vec<MemValue>,
    mem_start: u64,
    mem_end: u64,
    _reject_below_start: bool,
) -> Result<(), NormalizeError> {
    mems.retain_mut(|mem_val| {
        if (mem_val.addr >= mem_start) && (mem_val.addr <= mem_end) {
            mem_val.addr -= mem_start;
            true
        } else {
            false
        }
    });
    Ok(())
}

fn validate_no_duplicate_instruction_effects(
    asm: &AsmInstructionExecution,
) -> Result<(), NormalizeError> {
    let mut registers = BTreeSet::new();
    for write in &asm.register_writes {
        let key = canonicalize_register_name(&write.name).unwrap_or_else(|| write.name.clone());
        if !registers.insert(key.clone()) {
            return Err(NormalizeError::DuplicateRegisterWrite {
                pc: asm.pc,
                name: key,
            });
        }
    }

    let mut memory = BTreeSet::new();
    for write in &asm.memory_writes {
        if !memory.insert(write.addr) {
            return Err(NormalizeError::DuplicateMemoryWrite {
                pc: asm.pc,
                addr: write.addr,
            });
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn normalize_asm_instruction(
    asm: &mut AsmInstructionExecution,
    isa_base: ISABase,
    test_register_config: &RegisterOperandConfig,
    temp_min: u8,
    temp_max: u8,
    mem_start: u64,
    mem_end: u64,
    reject_memory_below_start: bool,
) -> Result<(), NormalizeError> {
    validate_no_duplicate_instruction_effects(asm)?;
    normalize_register_writes(
        &mut asm.register_writes,
        isa_base,
        test_register_config,
        temp_min,
        temp_max,
    )?;
    normalize_memory_writes_with_policy(
        &mut asm.memory_writes,
        mem_start,
        mem_end,
        reject_memory_below_start,
    )?;
    Ok(())
}

impl HartExecutionOutput {
    pub fn empty(hart_id: u32) -> Self {
        Self {
            hart_id,
            instructions: Vec::new(),
        }
    }

    /// Total number of retired asm rows on this hart.
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    pub fn trace_inst_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|instruction| instruction.trace_inst_id.is_some())
            .count()
    }

    pub fn iter_source_asm(&self) -> impl Iterator<Item = &AsmInstructionExecution> + '_ {
        self.instructions
            .iter()
            .filter(|instruction| instruction.trace_inst_id.is_some())
    }

    pub fn iter_asm(&self) -> impl Iterator<Item = &AsmInstructionExecution> + '_ {
        self.instructions.iter()
    }

    pub fn normalize(
        &mut self,
        isa_base: ISABase,
        mem_range: (u64, u64),
        test_register_config: &RegisterOperandConfig,
        temp_register_num_range: (u8, u8),
    ) -> Result<(), NormalizeError> {
        let mem_start = mem_range.0;
        let mem_end = mem_range.1;
        let (temp_min, temp_max) = temp_register_num_range;

        for asm in &mut self.instructions {
            normalize_asm_instruction(
                asm,
                isa_base,
                test_register_config,
                temp_min,
                temp_max,
                mem_start,
                mem_end,
                asm.trace_inst_id.is_some(),
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AsmInstructionExecution, ExecutionOutput, HartExecutionOutput, MemValue, RegisterValue,
        TraceInstId,
    };
    use crate::ISABase;
    use riscv_instruction::{
        AsmDecodeWithContext, IsaDecodeContext, RegisterOperandConfig, RiscvInstruction,
    };

    fn rv64(asm: &str) -> RiscvInstruction {
        RiscvInstruction::decode_asm_with_context(asm, IsaDecodeContext::RV64).unwrap()
    }

    #[test]
    fn normalize_rejects_rv32_integer_register_values_that_need_truncation() {
        let mut output = ExecutionOutput::empty("test", ISABase::Rv32);
        let main_inst = AsmInstructionExecution {
            pc: 0x1000,
            instruction: rv64("addi x6, x0, 1"),
            source_annotation: None,
            trace_inst_id: Some(TraceInstId::new(0, 0)),
            exceptions: Vec::new(),
            register_writes: vec![
                RegisterValue {
                    name: "x6".to_string(),
                    value: 0xabcd_ef01_8765_4321,
                },
                RegisterValue {
                    name: "f4".to_string(),
                    value: 0xabcd_ef01_8765_4321,
                },
            ],
            memory_writes: vec![MemValue {
                addr: 0x1000,
                value: 0xaa,
            }],
            timing: Default::default(),
        };
        let framework_inst = AsmInstructionExecution {
            pc: 0x1004,
            instruction: rv64("ebreak"),
            source_annotation: None,
            trace_inst_id: None,
            exceptions: Vec::new(),
            register_writes: vec![RegisterValue {
                name: "x31".to_string(),
                value: 0x0123_4567_89ab_cdef,
            }],
            memory_writes: vec![MemValue {
                addr: 0x1008,
                value: 0xbb,
            }],
            timing: Default::default(),
        };

        output.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![main_inst, framework_inst],
            },
        );

        let err = output
            .normalize(
                (0x1000, 0x1fff),
                &RegisterOperandConfig::default(),
                (28, 31),
            )
            .expect_err("normalize should reject hidden RV32 truncation");
        assert!(matches!(
            err,
            super::NormalizeError::Rv32IntegerValueOutOfRange { .. }
        ));
    }

    #[test]
    fn normalize_keeps_rv64_integer_registers_unmodified() {
        let mut output = ExecutionOutput::empty("test", ISABase::Rv64);
        let main_inst = AsmInstructionExecution {
            pc: 0x1000,
            instruction: rv64("addi x5, x0, 1"),
            source_annotation: None,
            trace_inst_id: Some(TraceInstId::new(0, 0)),
            exceptions: Vec::new(),
            register_writes: vec![RegisterValue {
                name: "x5".to_string(),
                value: 0xffff_ffff_1234_5678,
            }],
            memory_writes: Vec::new(),
            timing: Default::default(),
        };
        let framework_inst = AsmInstructionExecution {
            pc: 0x1004,
            instruction: rv64("ebreak"),
            source_annotation: None,
            trace_inst_id: None,
            exceptions: Vec::new(),
            register_writes: vec![RegisterValue {
                name: "x6".to_string(),
                value: 0xffff_ffff_8765_4321,
            }],
            memory_writes: Vec::new(),
            timing: Default::default(),
        };

        output.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![main_inst, framework_inst],
            },
        );

        output
            .normalize((0, u64::MAX), &RegisterOperandConfig::default(), (28, 31))
            .expect("normalize");

        let main = &output.per_hart[&0].instructions[0];
        assert_eq!(main.register_writes[0].value, 0xffff_ffff_1234_5678);
        let framework = &output.per_hart[&0].instructions[1];
        assert_eq!(framework.register_writes[0].value, 0xffff_ffff_8765_4321);
    }

    #[test]
    fn normalize_accepts_abi_register_alias_names() {
        let mut output = ExecutionOutput::empty("test", ISABase::Rv64);
        let main_inst = AsmInstructionExecution {
            pc: 0x1000,
            instruction: rv64("addi x15, x0, 1"),
            source_annotation: None,
            trace_inst_id: Some(TraceInstId::new(0, 0)),
            exceptions: Vec::new(),
            register_writes: vec![
                RegisterValue {
                    name: "a5".to_string(),
                    value: 0x1234,
                },
                RegisterValue {
                    name: "fa6".to_string(),
                    value: 0x5678,
                },
            ],
            memory_writes: Vec::new(),
            timing: Default::default(),
        };

        output.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![main_inst],
            },
        );

        output
            .normalize((0, u64::MAX), &RegisterOperandConfig::default(), (28, 31))
            .expect("normalize should accept ABI register aliases");

        let writes = &output.per_hart[&0].instructions[0].register_writes;
        assert_eq!(writes[0].name, "a5");
        assert_eq!(writes[1].name, "fa6");
    }
}
