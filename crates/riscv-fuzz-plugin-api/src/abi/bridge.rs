use super::memory::plugin_slice_as_bytes;
use super::{PluginCallResult, PluginSlice, PluginStreamingEmit, copy_and_free_plugin_bytes};
use crate::{
    ConfigError, ExecuteRequest, ExecutionOutput, ISABase, ImplExecutionError, InstructionEvent,
    PerHartTypedSequences, PluginMetadataRecord, RiscvCorePlugin, protocol,
};

pub fn metadata_payload_call(plugin: &dyn RiscvCorePlugin) -> PluginCallResult {
    match protocol::encode_plugin_metadata_record(&PluginMetadataRecord {
        core_metadata: plugin.metadata().clone(),
        linker_script: plugin.linker_script().to_string(),
        doctor_report: plugin.doctor(),
    }) {
        Ok(bytes) => PluginCallResult::ok(bytes),
        Err(error) => PluginCallResult::err(error),
    }
}

fn with_decoded_execute_request<F>(input: PluginSlice, f: F) -> PluginCallResult
where
    F: for<'a> FnOnce(ExecuteRequest<'a>) -> PluginCallResult,
{
    let request = match protocol::decode_execute_request(unsafe { plugin_slice_as_bytes(input) }) {
        Ok(request) => request,
        Err(error) => return PluginCallResult::err(error),
    };
    let hart_inputs = request
        .hart_inputs
        .iter()
        .map(|(hart_id, sequences)| (*hart_id, sequences.as_slice()))
        .collect::<Vec<_>>();

    f(ExecuteRequest {
        run_root: &request.run_root,
        isa_base: request.isa_base,
        hart_inputs: &hart_inputs,
        extension_override: request.extension_override.as_ref(),
        float_mode_override: request.float_mode_override,
        alignment_mode: request.alignment_mode,
        coverage_mode: request.coverage_mode,
    })
}

pub fn build_asm_payload_call(
    plugin: &dyn RiscvCorePlugin,
    input: PluginSlice,
) -> PluginCallResult {
    let request = match protocol::decode_build_asm_request(unsafe { plugin_slice_as_bytes(input) })
    {
        Ok(request) => request,
        Err(error) => return PluginCallResult::err(error),
    };
    let hart_inputs = request
        .hart_inputs
        .iter()
        .map(|(hart_id, sequences)| (*hart_id, sequences.as_slice()))
        .collect::<Vec<_>>();

    match plugin.build_asm(hart_inputs.as_slice(), request.isa_base) {
        Ok(asm) => PluginCallResult::ok(protocol::encode_build_asm_response(&asm)),
        Err(error) => PluginCallResult::err(error.to_string()),
    }
}

pub fn execute_payload_call(plugin: &dyn RiscvCorePlugin, input: PluginSlice) -> PluginCallResult {
    with_decoded_execute_request(input, |request| match plugin.execute(request) {
        Ok(output) => match protocol::encode_execution_output(&output) {
            Ok(bytes) => PluginCallResult::ok(bytes),
            Err(error) => PluginCallResult::err(error),
        },
        Err(error) => PluginCallResult::err(error.to_string()),
    })
}

/// # Safety
///
/// `callback_ctx` must be valid for the host-provided `emit` callback, and
/// `emit` must accept `PluginSlice` values only for the duration of each call.
pub unsafe fn execute_streaming_payload_call(
    plugin: &dyn RiscvCorePlugin,
    input: PluginSlice,
    callback_ctx: *mut std::ffi::c_void,
    emit: PluginStreamingEmit,
) -> PluginCallResult {
    let mut encode_error: Option<String> = None;
    with_decoded_execute_request(input, |request| {
        plugin.execute_streaming(request, &mut |event| {
            let bytes = match protocol::encode_instruction_event(&event) {
                Ok(bytes) => bytes,
                Err(error) => {
                    encode_error = Some(error);
                    return false;
                }
            };
            unsafe { emit(callback_ctx, bytes.as_slice().into()) }
        });

        if let Some(error) = encode_error {
            PluginCallResult::err(error)
        } else {
            PluginCallResult::ok(Vec::new())
        }
    })
}

pub fn validate_execute_request_payload_call(
    plugin: &dyn RiscvCorePlugin,
    input: PluginSlice,
) -> PluginCallResult {
    with_decoded_execute_request(input, |request| {
        match plugin.validate_execute_request(&request) {
            Ok(()) => PluginCallResult::ok(Vec::new()),
            Err(error) => PluginCallResult::err(error.to_string()),
        }
    })
}

pub fn encode_build_asm_request(
    hart_inputs: PerHartTypedSequences<'_>,
    isa_base: ISABase,
) -> Result<Vec<u8>, String> {
    protocol::encode_build_asm_request(hart_inputs, isa_base)
}

pub fn decode_build_asm_response(bytes: &[u8]) -> Result<String, String> {
    protocol::decode_build_asm_response(bytes)
}

pub fn encode_execute_request(request: ExecuteRequest<'_>) -> Result<Vec<u8>, String> {
    protocol::encode_execute_request(request)
}

pub fn decode_execution_output(bytes: &[u8]) -> Result<ExecutionOutput, String> {
    protocol::decode_execution_output(bytes)
}

pub fn read_streaming_instruction_event(
    bytes: &[u8],
    isa_base: ISABase,
) -> Result<InstructionEvent, String> {
    protocol::decode_instruction_event(bytes, isa_base)
}

pub fn decode_plugin_metadata_record(bytes: &[u8]) -> Result<PluginMetadataRecord, String> {
    protocol::decode_plugin_metadata_record(bytes)
}

pub fn decode_plugin_doctor_report(bytes: &[u8]) -> Result<crate::PluginDoctorReport, String> {
    protocol::decode_plugin_doctor_report(bytes)
}

pub fn read_plugin_call_payload<T>(
    bytes: &[u8],
    decode: impl FnOnce(&[u8]) -> Result<T, String>,
) -> Result<T, String> {
    decode(bytes)
}

pub fn read_build_asm_response(bytes: &[u8]) -> Result<String, ConfigError> {
    decode_build_asm_response(bytes)
        .map_err(|message| ConfigError::InvalidConfiguration { message })
}

pub fn read_execution_output(bytes: &[u8]) -> Result<ExecutionOutput, ImplExecutionError> {
    decode_execution_output(bytes).map_err(ImplExecutionError::Generic)
}

pub fn take_plugin_call_bytes(
    free_bytes: unsafe extern "C" fn(super::PluginBytes),
    bytes: super::PluginBytes,
) -> Vec<u8> {
    copy_and_free_plugin_bytes(free_bytes, bytes)
}
