use riscv_fuzz_plugin_api::{
    ExecutionOutput, ISABase, PerfMetrics,
    context::{ExecutionContextOutput, LogicalInstructionContextOutput},
};
use riscv_fuzz_toolchain::{
    ContextBuildError, build_execution_context_output, build_partial_execution_context_output,
};
use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::InstructionSequence;

#[derive(Debug, Clone, PartialEq)]
pub struct DiffExecutionView {
    pub logical_view: Vec<LogicalInstructionContextOutput>,
    pub mem_range: (u64, u64),
    pub core_id: String,
    pub isa_base: ISABase,
    pub perf: PerfMetrics,
}

impl DiffExecutionView {
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

pub(crate) fn diff_execution_view_from_context(value: ExecutionContextOutput) -> DiffExecutionView {
    DiffExecutionView {
        logical_view: value.logical_view,
        mem_range: value.mem_range,
        core_id: value.core_id,
        isa_base: value.isa_base,
        perf: value.perf,
    }
}

pub(crate) fn build_diff_execution_view(
    output: ExecutionOutput,
    mem_range: (u64, u64),
    per_hart_input_instructions: &[(u32, Vec<InstructionSequence<RiscvInstruction>>)],
) -> Result<DiffExecutionView, ContextBuildError> {
    build_execution_context_output(output, mem_range, per_hart_input_instructions)
        .map(diff_execution_view_from_context)
}

pub(crate) fn build_partial_diff_execution_view(
    output: ExecutionOutput,
    mem_range: (u64, u64),
    per_hart_input_instructions: &[(u32, Vec<InstructionSequence<RiscvInstruction>>)],
) -> Result<DiffExecutionView, ContextBuildError> {
    build_partial_execution_context_output(output, mem_range, per_hart_input_instructions)
        .map(diff_execution_view_from_context)
}

pub(crate) fn build_diff_execution_view_with_weak_fallback(
    output: ExecutionOutput,
    mem_range: (u64, u64),
    per_hart_input_instructions: &[(u32, Vec<InstructionSequence<RiscvInstruction>>)],
) -> Result<DiffExecutionView, ContextBuildError> {
    match build_diff_execution_view(output.clone(), mem_range, per_hart_input_instructions) {
        Ok(view) => Ok(view),
        Err(ContextBuildError::MissingLogicalInstructionExecution { .. }) => {
            build_partial_diff_execution_view(output, mem_range, per_hart_input_instructions)
        }
        Err(error) => Err(error),
    }
}
