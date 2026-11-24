use std::collections::{BTreeMap, HashMap};

use riscv_instruction::RiscvInstruction;
use serde::{Deserialize, Serialize};

pub fn format_register_context(context: &RegisterContext) -> String {
    if context.entries.is_empty() {
        "-".to_string()
    } else {
        context
            .entries
            .iter()
            .map(|entry| format!("{}=0x{:x}", entry.name, entry.value))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

pub fn format_memory_context(context: &MemoryContext) -> String {
    if context.base_register.is_empty() || context.entries.is_empty() {
        "-".to_string()
    } else {
        let mut parts = Vec::new();
        parts.push(format!(
            "{}=0x{:x}",
            context.base_register, context.base_offset
        ));

        for entry in &context.entries {
            let label = match entry.width {
                MemoryValueWidth::Byte => "byte",
                MemoryValueWidth::Half => "half",
                MemoryValueWidth::Word => "word",
                MemoryValueWidth::Dword => "dword",
            };
            let addresses = entry
                .addresses
                .iter()
                .map(|addr| format!("0x{:x}", addr))
                .collect::<Vec<_>>()
                .join(",");
            parts.push(format!("{label} [{addresses}]=0x{:x}", entry.value));
        }

        parts.join("; ")
    }
}

use crate::execution::{
    ExceptionCause, InstructionTiming, MemValue, PerfMetrics, RegisterValue, TraceInstId,
};
use crate::isa::ISABase;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterContextEntry {
    pub name: String,
    pub value: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterContext {
    pub entries: Vec<RegisterContextEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryValueWidth {
    Byte,
    Half,
    Word,
    Dword,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryContextEntry {
    pub width: MemoryValueWidth,
    pub addresses: Vec<u64>,
    pub value: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryContext {
    pub base_register: String,
    pub base_offset: u64,
    pub entries: Vec<MemoryContextEntry>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionContext {
    pub registers_before: HashMap<String, u64>,
    pub memory_before: BTreeMap<u64, u8>,
    pub register_context: RegisterContext,
    pub memory_context: MemoryContext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsmInstructionContextOutput {
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
    pub context: InstructionContext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HartExecutionContextOutput {
    pub hart_id: u32,
    #[serde(default)]
    pub instructions: Vec<AsmInstructionContextOutput>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogicalInstructionContextOutput {
    pub hart_id: u32,
    pub instruction_index: usize,
    pub local_instruction_index: usize,
    pub trace_inst_id: TraceInstId,
    pub instruction: RiscvInstruction,
    #[serde(default)]
    pub asm_instruction_indices: Vec<usize>,
    #[serde(default)]
    pub exceptions: Vec<ExceptionCause>,
    #[serde(default)]
    pub register_writes: Vec<RegisterValue>,
    #[serde(default)]
    pub memory_writes: Vec<MemValue>,
    #[serde(default)]
    pub timing: InstructionTiming,
    pub context: InstructionContext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionContextOutput {
    #[serde(default)]
    pub per_hart: BTreeMap<u32, HartExecutionContextOutput>,
    #[serde(default)]
    pub framework_per_hart: BTreeMap<u32, HartExecutionContextOutput>,
    #[serde(default)]
    pub logical_view: Vec<LogicalInstructionContextOutput>,
    pub mem_range: (u64, u64),
    pub core_id: String,
    pub isa_base: ISABase,
    #[serde(default)]
    pub perf: PerfMetrics,
}

impl HartExecutionContextOutput {
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }
}

impl ExecutionContextOutput {
    pub fn raw_instruction_count(&self) -> usize {
        self.per_hart
            .values()
            .map(HartExecutionContextOutput::instruction_count)
            .sum::<usize>()
            + self
                .framework_per_hart
                .values()
                .map(HartExecutionContextOutput::instruction_count)
                .sum::<usize>()
    }

    pub fn raw_exception_count(&self) -> usize {
        self.per_hart
            .values()
            .flat_map(|hart| hart.instructions.iter())
            .chain(
                self.framework_per_hart
                    .values()
                    .flat_map(|hart| hart.instructions.iter()),
            )
            .map(|instruction| instruction.exceptions.len())
            .sum()
    }

    pub fn logical_instruction_count(&self) -> usize {
        self.logical_view.len()
    }

    pub fn logical_instruction(&self, index: usize) -> Option<&LogicalInstructionContextOutput> {
        self.logical_view.get(index)
    }

    pub fn logical_instruction_by_hart_local(
        &self,
        hart_id: u32,
        local_instruction_index: usize,
    ) -> Option<&LogicalInstructionContextOutput> {
        self.logical_view.iter().find(|instruction| {
            instruction.hart_id == hart_id
                && instruction.local_instruction_index == local_instruction_index
        })
    }
}
