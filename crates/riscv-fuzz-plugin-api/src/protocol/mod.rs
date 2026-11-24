mod common;
mod doctor;
mod execute;
mod execution_output;
mod generated;
mod metadata;
mod streaming;

#[cfg(test)]
mod tests;

pub use metadata::PluginMetadataRecord;

pub(crate) use doctor::{
    decode_plugin_doctor_report, plugin_doctor_report_from_proto, plugin_doctor_report_to_proto,
};
pub(crate) use execute::{
    decode_build_asm_request, decode_build_asm_response, decode_execute_request,
    encode_build_asm_request, encode_build_asm_response, encode_execute_request,
};
pub(crate) use execution_output::{decode_execution_output, encode_execution_output};
pub(crate) use metadata::{decode_plugin_metadata_record, encode_plugin_metadata_record};
pub(crate) use streaming::{decode_instruction_event, encode_instruction_event};
