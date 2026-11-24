use std::path::PathBuf;

use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::InstructionSequence;

use crate::{AlignmentMode, CoverageMode, ExecuteRequest, FloatExtension, ISABase, IsaExtensions};

use super::common::{
    coverage_mode_from_proto, coverage_mode_to_proto, decode_message, decode_optional_extensions,
    encode_message, encode_optional_extensions, float_extension_from_proto,
    float_extension_to_proto, isa_base_from_proto, isa_base_to_proto, path_to_proto_string, pb,
};

/// Schema version for the bincode-encoded `Vec<InstructionSequence<RiscvInstruction>>`
/// payload carried over the wire.
pub const TYPED_PAYLOAD_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BuildAsmRequest {
    pub hart_inputs: Vec<(u32, Vec<InstructionSequence<RiscvInstruction>>)>,
    pub isa_base: ISABase,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExecuteRequestPayload {
    pub run_root: PathBuf,
    pub isa_base: ISABase,
    pub hart_inputs: Vec<(u32, Vec<InstructionSequence<RiscvInstruction>>)>,
    pub extension_override: Option<IsaExtensions>,
    pub float_mode_override: Option<FloatExtension>,
    pub alignment_mode: AlignmentMode,
    pub coverage_mode: CoverageMode,
}

fn bincode_encode_typed(seqs: &[InstructionSequence<RiscvInstruction>]) -> Result<Vec<u8>, String> {
    bincode::serde::encode_to_vec(seqs, bincode::config::standard())
        .map_err(|err| format!("failed to bincode-encode typed testcase instructions: {err}"))
}

fn bincode_decode_typed(
    bytes: &[u8],
) -> Result<Vec<InstructionSequence<RiscvInstruction>>, String> {
    let (value, _read): (Vec<InstructionSequence<RiscvInstruction>>, usize) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard()).map_err(|err| {
            format!("failed to bincode-decode typed testcase instructions: {err}")
        })?;
    Ok(value)
}

fn hart_input_to_proto(
    hart_id: u32,
    sequences: &[InstructionSequence<RiscvInstruction>],
) -> Result<pb::HartExecutionInput, String> {
    Ok(pb::HartExecutionInput {
        hart_id,
        typed_payload: bincode_encode_typed(sequences)?,
    })
}

fn hart_input_from_proto(
    message: pb::HartExecutionInput,
    schema_version: u32,
    field: &str,
) -> Result<(u32, Vec<InstructionSequence<RiscvInstruction>>), String> {
    if schema_version != TYPED_PAYLOAD_SCHEMA_VERSION {
        return Err(format!(
            "{field}: typed payload schema version mismatch (got {schema_version}, expected {})",
            TYPED_PAYLOAD_SCHEMA_VERSION
        ));
    }
    Ok((
        message.hart_id,
        bincode_decode_typed(&message.typed_payload)?,
    ))
}

pub(crate) fn encode_build_asm_request(
    hart_inputs: crate::PerHartTypedSequences<'_>,
    isa_base: ISABase,
) -> Result<Vec<u8>, String> {
    let message = pb::BuildAsmRequest {
        isa_base: isa_base_to_proto(isa_base),
        hart_inputs: hart_inputs
            .iter()
            .map(|(hart_id, sequences)| hart_input_to_proto(*hart_id, sequences))
            .collect::<Result<Vec<_>, _>>()?,
        typed_payload_schema_version: TYPED_PAYLOAD_SCHEMA_VERSION,
    };
    Ok(encode_message(&message))
}

pub(crate) fn decode_build_asm_request(bytes: &[u8]) -> Result<BuildAsmRequest, String> {
    let message: pb::BuildAsmRequest = decode_message(bytes, "build asm request")?;
    let schema_version = message.typed_payload_schema_version;
    let hart_inputs = message
        .hart_inputs
        .into_iter()
        .map(|hart_input| {
            hart_input_from_proto(
                hart_input,
                schema_version,
                "build asm request.hart_inputs.typed_payload",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BuildAsmRequest {
        hart_inputs,
        isa_base: isa_base_from_proto(message.isa_base, "build asm request.isa_base")?,
    })
}

pub(crate) fn encode_build_asm_response(asm: &str) -> Vec<u8> {
    encode_message(&pb::BuildAsmResponse {
        asm: asm.to_string(),
    })
}

pub(crate) fn decode_build_asm_response(bytes: &[u8]) -> Result<String, String> {
    let message: pb::BuildAsmResponse = decode_message(bytes, "build asm response")?;
    Ok(message.asm)
}

pub(crate) fn encode_execute_request(request: ExecuteRequest<'_>) -> Result<Vec<u8>, String> {
    let message = execute_request_to_proto(request)?;
    Ok(encode_message(&message))
}

pub(crate) fn decode_execute_request(bytes: &[u8]) -> Result<ExecuteRequestPayload, String> {
    let message: pb::ExecuteRequest = decode_message(bytes, "execute request")?;
    execute_request_from_proto(message)
}

fn execute_request_to_proto(request: ExecuteRequest<'_>) -> Result<pb::ExecuteRequest, String> {
    Ok(pb::ExecuteRequest {
        run_root: path_to_proto_string(request.run_root, "execute request.run_root")?,
        isa_base: isa_base_to_proto(request.isa_base),
        hart_inputs: request
            .hart_inputs
            .iter()
            .map(|(hart_id, sequences)| hart_input_to_proto(*hart_id, sequences))
            .collect::<Result<Vec<_>, _>>()?,
        extension_override: encode_optional_extensions(request.extension_override)?,
        allow_unaligned: request.alignment_mode.allows_unaligned(),
        coverage_mode: coverage_mode_to_proto(request.coverage_mode),
        float_mode_override: request.float_mode_override.map(float_extension_to_proto),
        typed_payload_schema_version: TYPED_PAYLOAD_SCHEMA_VERSION,
    })
}

fn execute_request_from_proto(
    message: pb::ExecuteRequest,
) -> Result<ExecuteRequestPayload, String> {
    let schema_version = message.typed_payload_schema_version;
    let hart_inputs = message
        .hart_inputs
        .into_iter()
        .map(|hart_input| {
            hart_input_from_proto(
                hart_input,
                schema_version,
                "execute request.hart_inputs.typed_payload",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ExecuteRequestPayload {
        run_root: PathBuf::from(message.run_root),
        isa_base: isa_base_from_proto(message.isa_base, "execute request.isa_base")?,
        hart_inputs,
        extension_override: decode_optional_extensions(
            message.extension_override,
            "execute request.extension_override",
        )?,
        float_mode_override: message
            .float_mode_override
            .map(|value| float_extension_from_proto(value, "execute request.float_mode_override"))
            .transpose()?,
        alignment_mode: if message.allow_unaligned {
            AlignmentMode::AllowUnaligned
        } else {
            AlignmentMode::RequireAligned
        },
        coverage_mode: coverage_mode_from_proto(
            message.coverage_mode,
            "execute request.coverage_mode",
        )?,
    })
}
