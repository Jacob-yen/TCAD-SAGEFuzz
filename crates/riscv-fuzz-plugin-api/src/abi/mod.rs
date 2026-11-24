mod bridge;
mod descriptor;
mod memory;
mod types;

pub use bridge::{
    build_asm_payload_call, decode_build_asm_response, decode_execution_output,
    decode_plugin_doctor_report, decode_plugin_metadata_record, encode_build_asm_request,
    encode_execute_request, execute_payload_call, execute_streaming_payload_call,
    metadata_payload_call, read_build_asm_response, read_execution_output,
    read_plugin_call_payload, read_streaming_instruction_event, take_plugin_call_bytes,
    validate_execute_request_payload_call,
};
pub use descriptor::plugin_descriptor_symbol_name_for_stem;
pub use memory::{copy_and_free_plugin_bytes, free_plugin_bytes};
pub use types::{
    PLUGIN_CALL_STATUS_ERR, PLUGIN_CALL_STATUS_OK, PluginBytes, PluginCallResult,
    PluginDescriptorV3, PluginSlice, PluginStreamingEmit, RISCV_FUZZ_PLUGIN_ABI_VERSION,
    RISCV_FUZZ_PLUGIN_DESCRIPTOR_SYMBOL_PREFIX,
};
