use std::collections::BTreeMap;

use crate::{
    AsmInstructionExecution, ExceptionCause, ExecutionOutput, HartExecutionOutput, HartPerfMetrics,
    ISABase, InstructionTiming, MemValue, PerfMetrics, RegisterValue, TraceInstId,
};
use riscv_instruction::{AsmDecodeWithContext, IsaDecodeContext, RiscvInstruction};

use super::common::{
    decode_message, encode_message, isa_base_from_proto, isa_base_to_proto, pb, required,
};

pub(crate) fn encode_execution_output(output: &ExecutionOutput) -> Result<Vec<u8>, String> {
    let message = execution_output_to_proto(output)?;
    Ok(encode_message(&message))
}

pub(crate) fn decode_execution_output(bytes: &[u8]) -> Result<ExecutionOutput, String> {
    let message: pb::ExecutionOutput = decode_message(bytes, "execution output")?;
    execution_output_from_proto(message)
}

fn execution_output_to_proto(output: &ExecutionOutput) -> Result<pb::ExecutionOutput, String> {
    Ok(pb::ExecutionOutput {
        per_hart: output
            .per_hart
            .values()
            .map(hart_execution_output_to_proto)
            .collect::<Result<Vec<_>, _>>()?,
        core_id: output.core_id.clone(),
        isa_base: isa_base_to_proto(output.isa_base),
        perf: Some(perf_metrics_to_proto(&output.perf)?),
    })
}

pub(crate) fn isa_base_to_ctx(isa_base: ISABase) -> IsaDecodeContext {
    match isa_base {
        ISABase::Rv32 => IsaDecodeContext::RV32,
        ISABase::Rv64 => IsaDecodeContext::RV64,
    }
}

fn execution_output_from_proto(message: pb::ExecutionOutput) -> Result<ExecutionOutput, String> {
    let isa_base = isa_base_from_proto(message.isa_base, "execution output.isa_base")?;
    let ctx = isa_base_to_ctx(isa_base);
    Ok(ExecutionOutput {
        per_hart: per_hart_execution_output_from_proto(message.per_hart, ctx)?,
        core_id: message.core_id,
        isa_base,
        perf: perf_metrics_from_proto(required(message.perf, "execution output.perf")?)?,
    })
}

fn hart_execution_output_to_proto(
    output: &HartExecutionOutput,
) -> Result<pb::HartExecutionOutput, String> {
    Ok(pb::HartExecutionOutput {
        hart_id: output.hart_id,
        instructions: output
            .instructions
            .iter()
            .map(asm_instruction_execution_to_proto)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn per_hart_execution_output_from_proto(
    entries: Vec<pb::HartExecutionOutput>,
    ctx: IsaDecodeContext,
) -> Result<BTreeMap<u32, HartExecutionOutput>, String> {
    let mut per_hart = BTreeMap::new();
    for entry in entries {
        let hart_id = entry.hart_id;
        let decoded = HartExecutionOutput {
            hart_id,
            instructions: entry
                .instructions
                .into_iter()
                .map(|i| asm_instruction_execution_from_proto(i, ctx))
                .collect::<Result<Vec<_>, _>>()?,
        };

        if per_hart.insert(hart_id, decoded).is_some() {
            return Err(format!(
                "duplicate execution output entry for hart {}",
                hart_id
            ));
        }
    }
    Ok(per_hart)
}

fn exception_cause_from_proto(message: pb::ExceptionCause) -> ExceptionCause {
    ExceptionCause {
        code: message.code,
        raw: message.raw,
    }
}

fn exception_cause_to_proto(cause: &ExceptionCause) -> pb::ExceptionCause {
    pb::ExceptionCause {
        code: cause.code,
        raw: cause.raw.clone(),
    }
}

fn trace_inst_id_to_proto(id: TraceInstId) -> Result<pb::TraceInstId, String> {
    Ok(pb::TraceInstId {
        hart_id: id.hart_id,
        local_index: u64::try_from(id.local_index).map_err(|_| {
            format!(
                "execution output trace source local_index {} does not fit into u64",
                id.local_index
            )
        })?,
    })
}

fn trace_inst_id_from_proto(message: pb::TraceInstId) -> Result<TraceInstId, String> {
    Ok(TraceInstId {
        hart_id: message.hart_id,
        local_index: usize::try_from(message.local_index).map_err(|_| {
            format!(
                "execution output trace source local_index {} does not fit into usize",
                message.local_index
            )
        })?,
    })
}

pub(crate) fn asm_instruction_execution_to_proto(
    instruction: &AsmInstructionExecution,
) -> Result<pb::AsmInstructionExecution, String> {
    Ok(pb::AsmInstructionExecution {
        pc: instruction.pc,
        instruction: instruction.instruction.to_string(),
        source_annotation: instruction.source_annotation.clone(),
        trace_inst_id: instruction
            .trace_inst_id
            .map(trace_inst_id_to_proto)
            .transpose()?,
        exceptions: instruction
            .exceptions
            .iter()
            .map(exception_cause_to_proto)
            .collect(),
        register_writes: instruction
            .register_writes
            .iter()
            .map(|register| pb::RegisterValue {
                name: register.name.clone(),
                value: register.value,
            })
            .collect(),
        memory_writes: instruction
            .memory_writes
            .iter()
            .map(mem_value_to_proto)
            .collect::<Result<Vec<_>, _>>()?,
        timing: Some(instruction_timing_to_proto(&instruction.timing)),
    })
}

pub(crate) fn asm_instruction_execution_from_proto(
    instruction: pb::AsmInstructionExecution,
    ctx: IsaDecodeContext,
) -> Result<AsmInstructionExecution, String> {
    Ok(AsmInstructionExecution {
        pc: instruction.pc,
        instruction: RiscvInstruction::decode_asm_with_context(&instruction.instruction, ctx)
            .map_err(|e| format!("execution output.instruction parse error: {e}"))?,
        source_annotation: instruction.source_annotation,
        trace_inst_id: instruction
            .trace_inst_id
            .map(trace_inst_id_from_proto)
            .transpose()?,
        exceptions: instruction
            .exceptions
            .into_iter()
            .map(exception_cause_from_proto)
            .collect(),
        register_writes: instruction
            .register_writes
            .into_iter()
            .map(|register| RegisterValue {
                name: register.name,
                value: register.value,
            })
            .collect(),
        memory_writes: instruction
            .memory_writes
            .into_iter()
            .map(mem_value_from_proto)
            .collect::<Result<Vec<_>, _>>()?,
        timing: instruction_timing_from_proto(required(
            instruction.timing,
            "execution output.hart.instructions.timing",
        )?),
    })
}

fn mem_value_to_proto(value: &MemValue) -> Result<pb::MemValue, String> {
    Ok(pb::MemValue {
        addr: value.addr,
        value: u32::from(value.value),
    })
}

fn mem_value_from_proto(message: pb::MemValue) -> Result<MemValue, String> {
    Ok(MemValue {
        addr: message.addr,
        value: u8::try_from(message.value).map_err(|_| {
            format!(
                "execution output.mem_value {} does not fit into u8",
                message.value
            )
        })?,
    })
}

fn instruction_timing_to_proto(timing: &InstructionTiming) -> pb::InstructionTiming {
    pb::InstructionTiming {
        clk_start: timing.clk_start,
        clk_end: timing.clk_end,
        clk_span: timing.clk_span,
    }
}

fn instruction_timing_from_proto(message: pb::InstructionTiming) -> InstructionTiming {
    InstructionTiming::new(message.clk_start, message.clk_end, message.clk_span)
}

pub(crate) fn perf_metrics_to_proto(perf: &PerfMetrics) -> Result<pb::PerfMetrics, String> {
    Ok(pb::PerfMetrics {
        global_cycles: perf.global_cycles,
        per_hart: perf
            .per_hart
            .iter()
            .map(|(hart_id, metrics)| pb::PerHartPerfMetrics {
                hart_id: *hart_id,
                metrics: Some(hart_perf_metrics_to_proto(metrics)),
            })
            .collect(),
        notes: perf.notes.clone(),
    })
}

pub(crate) fn perf_metrics_from_proto(message: pb::PerfMetrics) -> Result<PerfMetrics, String> {
    let mut per_hart = BTreeMap::new();
    for entry in message.per_hart {
        let metrics = hart_perf_metrics_from_proto(required(
            entry.metrics,
            "execution output.perf.per_hart.metrics",
        )?);
        if per_hart.insert(entry.hart_id, metrics).is_some() {
            return Err(format!(
                "duplicate perf metrics entry for hart {}",
                entry.hart_id
            ));
        }
    }

    Ok(PerfMetrics {
        global_cycles: message.global_cycles,
        per_hart,
        notes: message.notes,
    })
}

fn hart_perf_metrics_to_proto(metrics: &HartPerfMetrics) -> pb::HartPerfMetrics {
    pb::HartPerfMetrics {
        mcycle_start: metrics.mcycle_start,
        mcycle_end: metrics.mcycle_end,
        minstret_start: metrics.minstret_start,
        minstret_end: metrics.minstret_end,
    }
}

fn hart_perf_metrics_from_proto(message: pb::HartPerfMetrics) -> HartPerfMetrics {
    HartPerfMetrics {
        mcycle_start: message.mcycle_start,
        mcycle_end: message.mcycle_end,
        minstret_start: message.minstret_start,
        minstret_end: message.minstret_end,
    }
}
