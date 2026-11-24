use crate::ISABase;
use crate::error::ImplExecutionError;
use crate::streaming::InstructionEvent;

use super::common::{decode_message, encode_message, pb, required};
use super::execution_output::{
    asm_instruction_execution_from_proto, asm_instruction_execution_to_proto, isa_base_to_ctx,
    perf_metrics_from_proto, perf_metrics_to_proto,
};

pub(crate) fn encode_instruction_event(event: &InstructionEvent) -> Result<Vec<u8>, String> {
    let kind = match event {
        InstructionEvent::Instruction {
            hart_id,
            instruction,
        } => pb::streaming_instruction_event::Kind::Instruction(pb::StreamingInstructionRowEvent {
            hart_id: *hart_id,
            instruction: Some(asm_instruction_execution_to_proto(instruction)?),
        }),
        InstructionEvent::Completed { perf } => {
            pb::streaming_instruction_event::Kind::Completed(perf_metrics_to_proto(perf)?)
        }
        InstructionEvent::Failed { error } => {
            pb::streaming_instruction_event::Kind::Failed(error.to_string())
        }
    };

    Ok(encode_message(&pb::StreamingInstructionEvent {
        kind: Some(kind),
    }))
}

pub(crate) fn decode_instruction_event(
    bytes: &[u8],
    isa_base: ISABase,
) -> Result<InstructionEvent, String> {
    let message: pb::StreamingInstructionEvent =
        decode_message(bytes, "streaming instruction event")?;
    let ctx = isa_base_to_ctx(isa_base);
    let kind = required(message.kind, "streaming instruction event.kind")?;
    match kind {
        pb::streaming_instruction_event::Kind::Instruction(event) => {
            Ok(InstructionEvent::Instruction {
                hart_id: event.hart_id,
                instruction: asm_instruction_execution_from_proto(
                    required(event.instruction, "streaming instruction.instruction")?,
                    ctx,
                )?,
            })
        }
        pb::streaming_instruction_event::Kind::Completed(perf) => Ok(InstructionEvent::Completed {
            perf: perf_metrics_from_proto(perf)?,
        }),
        pb::streaming_instruction_event::Kind::Failed(message) => Ok(InstructionEvent::Failed {
            error: ImplExecutionError::Generic(message),
        }),
    }
}
