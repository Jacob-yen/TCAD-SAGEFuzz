use serde::{Deserialize, Serialize};

use crate::{
    CorePluginMetadata, IsaExecutionProfile, IsaExecutionProfileMap, PluginDoctorReport,
    UnalignedAccessSupport, metadata::CoverageModeSupport,
};

use super::common::{
    decode_extensions, decode_message, encode_extensions, encode_message, pb, required,
};
use super::{plugin_doctor_report_from_proto, plugin_doctor_report_to_proto};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginMetadataRecord {
    pub core_metadata: CorePluginMetadata,
    pub linker_script: String,
    pub doctor_report: PluginDoctorReport,
}

pub(crate) fn encode_plugin_metadata_record(
    record: &PluginMetadataRecord,
) -> Result<Vec<u8>, String> {
    let message = plugin_metadata_to_proto(record)?;
    Ok(encode_message(&message))
}

pub(crate) fn decode_plugin_metadata_record(bytes: &[u8]) -> Result<PluginMetadataRecord, String> {
    let message: pb::PluginMetadataRecord = decode_message(bytes, "plugin metadata")?;
    plugin_metadata_from_proto(message)
}

fn plugin_metadata_to_proto(
    record: &PluginMetadataRecord,
) -> Result<pb::PluginMetadataRecord, String> {
    Ok(pb::PluginMetadataRecord {
        core_metadata: Some(core_metadata_to_proto(&record.core_metadata)?),
        linker_script: record.linker_script.clone(),
        doctor_report: Some(plugin_doctor_report_to_proto(&record.doctor_report)),
    })
}

fn plugin_metadata_from_proto(
    message: pb::PluginMetadataRecord,
) -> Result<PluginMetadataRecord, String> {
    let core_metadata = core_metadata_from_proto(required(
        message.core_metadata,
        "plugin metadata.core_metadata",
    )?)?;
    let doctor_report = message
        .doctor_report
        .map(plugin_doctor_report_from_proto)
        .transpose()?
        .unwrap_or_else(|| PluginDoctorReport::empty(core_metadata.core_id.clone()));

    Ok(PluginMetadataRecord {
        core_metadata,
        linker_script: message.linker_script,
        doctor_report,
    })
}

fn core_metadata_to_proto(metadata: &CorePluginMetadata) -> Result<pb::CorePluginMetadata, String> {
    Ok(pb::CorePluginMetadata {
        core_id: metadata.core_id.clone(),
        isa_profiles: Some(isa_profile_map_to_proto(&metadata.isa_profiles)?),
        program_mem_range: Some(pb::ProgramMemoryRange {
            start: metadata.program_mem_range.0,
            end: metadata.program_mem_range.1,
        }),
    })
}

fn core_metadata_from_proto(message: pb::CorePluginMetadata) -> Result<CorePluginMetadata, String> {
    let program_mem_range = required(message.program_mem_range, "core metadata.program_mem_range")?;
    Ok(CorePluginMetadata {
        core_id: message.core_id,
        isa_profiles: isa_profile_map_from_proto(required(
            message.isa_profiles,
            "core metadata.isa_profiles",
        )?)?,
        program_mem_range: (program_mem_range.start, program_mem_range.end),
    })
}

fn isa_profile_map_to_proto(
    profiles: &IsaExecutionProfileMap,
) -> Result<pb::IsaExecutionProfileMap, String> {
    Ok(pb::IsaExecutionProfileMap {
        rv32: Some(isa_profile_to_proto(&profiles.rv32)?),
        rv64: Some(isa_profile_to_proto(&profiles.rv64)?),
    })
}

fn isa_profile_map_from_proto(
    message: pb::IsaExecutionProfileMap,
) -> Result<IsaExecutionProfileMap, String> {
    Ok(IsaExecutionProfileMap {
        rv32: isa_profile_from_proto(required(message.rv32, "isa profile map.rv32")?)?,
        rv64: isa_profile_from_proto(required(message.rv64, "isa profile map.rv64")?)?,
    })
}

fn isa_profile_to_proto(profile: &IsaExecutionProfile) -> Result<pb::IsaExecutionProfile, String> {
    Ok(pb::IsaExecutionProfile {
        supported: profile.supported,
        supports_dual_hart: profile.supports_dual_hart,
        coverage_mode_support: Some(pb::CoverageModeSupport {
            off: profile.coverage_mode_support.off,
            light: profile.coverage_mode_support.light,
            full: profile.coverage_mode_support.full,
        }),
        unaligned_access_support: Some(pb::UnalignedAccessSupport {
            aligned: profile.unaligned_access_support.aligned,
            unaligned: profile.unaligned_access_support.unaligned,
        }),
        extensions: Some(encode_extensions(&profile.extensions)?),
    })
}

fn isa_profile_from_proto(message: pb::IsaExecutionProfile) -> Result<IsaExecutionProfile, String> {
    Ok(IsaExecutionProfile {
        supported: message.supported,
        supports_dual_hart: message.supports_dual_hart,
        coverage_mode_support: coverage_mode_support_from_proto(required(
            message.coverage_mode_support,
            "isa profile.coverage_mode_support",
        )?),
        unaligned_access_support: unaligned_access_support_from_proto(required(
            message.unaligned_access_support,
            "isa profile.unaligned_access_support",
        )?),
        extensions: decode_extensions(message.extensions, "isa profile.extensions")?,
    })
}

fn coverage_mode_support_from_proto(message: pb::CoverageModeSupport) -> CoverageModeSupport {
    CoverageModeSupport {
        off: message.off,
        light: message.light,
        full: message.full,
    }
}

fn unaligned_access_support_from_proto(
    message: pb::UnalignedAccessSupport,
) -> UnalignedAccessSupport {
    UnalignedAccessSupport {
        aligned: message.aligned,
        unaligned: message.unaligned,
    }
}
