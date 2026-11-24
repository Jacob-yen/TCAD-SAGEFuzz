use std::path::Path;

use prost::Message;
use riscv_instruction::{RV32Extensions, RV64Extensions};

use crate::{CoverageMode, FloatExtension, ISABase, IsaExtensions};

pub(crate) use super::generated::riscv_fuzz::plugin::v1 as pb;

pub(crate) fn encode_message<M>(message: &M) -> Vec<u8>
where
    M: Message,
{
    let mut bytes = Vec::with_capacity(message.encoded_len());
    message
        .encode(&mut bytes)
        .expect("encoding protobuf message into Vec<u8> should never fail");
    bytes
}

pub(crate) fn decode_message<M>(bytes: &[u8], label: &str) -> Result<M, String>
where
    M: Message + Default,
{
    M::decode(bytes).map_err(|error| format!("failed to decode {label} protobuf: {error}"))
}

pub(crate) fn encode_optional_extensions(
    extensions: Option<&IsaExtensions>,
) -> Result<Option<pb::ExtensionSet>, String> {
    extensions.map(extensions_to_proto).transpose()
}

pub(crate) fn decode_optional_extensions(
    extensions: Option<pb::ExtensionSet>,
    field: &str,
) -> Result<Option<IsaExtensions>, String> {
    extensions
        .map(|extensions| extensions_from_proto(extensions, field))
        .transpose()
}

pub(crate) fn encode_extensions(extensions: &IsaExtensions) -> Result<pb::ExtensionSet, String> {
    extensions_to_proto(extensions)
}

pub(crate) fn decode_extensions(
    extensions: Option<pb::ExtensionSet>,
    field: &str,
) -> Result<IsaExtensions, String> {
    required(extensions, field).and_then(|extensions| extensions_from_proto(extensions, field))
}

pub(crate) fn isa_base_to_proto(isa_base: ISABase) -> i32 {
    match isa_base {
        ISABase::Rv32 => pb::IsaBase::Rv32 as i32,
        ISABase::Rv64 => pb::IsaBase::Rv64 as i32,
    }
}

pub(crate) fn isa_base_from_proto(value: i32, field: &str) -> Result<ISABase, String> {
    match pb::IsaBase::try_from(value).map_err(|_| format!("invalid {field} value: {value}"))? {
        pb::IsaBase::Rv32 => Ok(ISABase::Rv32),
        pb::IsaBase::Rv64 => Ok(ISABase::Rv64),
        pb::IsaBase::Unspecified => Err(format!("{field} must not be unspecified")),
    }
}

pub(crate) fn coverage_mode_to_proto(mode: CoverageMode) -> i32 {
    match mode {
        CoverageMode::Off => pb::CoverageMode::Off as i32,
        CoverageMode::Full => pb::CoverageMode::Full as i32,
        CoverageMode::Light => pb::CoverageMode::Light as i32,
    }
}

pub(crate) fn coverage_mode_from_proto(value: i32, field: &str) -> Result<CoverageMode, String> {
    match pb::CoverageMode::try_from(value)
        .map_err(|_| format!("invalid {field} value: {value}"))?
    {
        pb::CoverageMode::Off => Ok(CoverageMode::Off),
        pb::CoverageMode::Full => Ok(CoverageMode::Full),
        pb::CoverageMode::Light => Ok(CoverageMode::Light),
        pb::CoverageMode::Unspecified => Err(format!("{field} must not be unspecified")),
    }
}

pub(crate) fn float_extension_to_proto(choice: FloatExtension) -> i32 {
    match choice {
        FloatExtension::None => pb::FloatExtension::None as i32,
        FloatExtension::F => pb::FloatExtension::F as i32,
        FloatExtension::FD => pb::FloatExtension::Fd as i32,
    }
}

pub(crate) fn float_extension_from_proto(
    value: i32,
    field: &str,
) -> Result<FloatExtension, String> {
    match pb::FloatExtension::try_from(value)
        .map_err(|_| format!("invalid {field} value: {value}"))?
    {
        pb::FloatExtension::None => Ok(FloatExtension::None),
        pb::FloatExtension::F => Ok(FloatExtension::F),
        pb::FloatExtension::Fd => Ok(FloatExtension::FD),
        pb::FloatExtension::Unspecified => Err(format!("{field} must not be unspecified")),
    }
}

pub(crate) fn path_to_proto_string(path: &Path, field: &str) -> Result<String, String> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{field} contains non-UTF-8 data: {}", path.display()))
}

pub(crate) fn required<T>(value: Option<T>, field: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("missing required protobuf field `{field}`"))
}

fn extensions_to_proto(extensions: &IsaExtensions) -> Result<pb::ExtensionSet, String> {
    match extensions {
        IsaExtensions::Rv32(values) => Ok(pb::ExtensionSet {
            isa_base: isa_base_to_proto(ISABase::Rv32),
            names: values
                .iter()
                .map(|extension| {
                    serde_plain::to_string(extension).map_err(|error| {
                        format!(
                            "failed to serialize RV32 extension {:?}: {error}",
                            extension
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        }),
        IsaExtensions::Rv64(values) => Ok(pb::ExtensionSet {
            isa_base: isa_base_to_proto(ISABase::Rv64),
            names: values
                .iter()
                .map(|extension| {
                    serde_plain::to_string(extension).map_err(|error| {
                        format!(
                            "failed to serialize RV64 extension {:?}: {error}",
                            extension
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        }),
    }
}

fn extensions_from_proto(message: pb::ExtensionSet, field: &str) -> Result<IsaExtensions, String> {
    match isa_base_from_proto(message.isa_base, field)? {
        ISABase::Rv32 => message
            .names
            .into_iter()
            .map(|name| {
                serde_plain::from_str::<RV32Extensions>(&name).map_err(|error| {
                    format!("failed to parse RV32 extension `{name}` in {field}: {error}")
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(IsaExtensions::rv32),
        ISABase::Rv64 => message
            .names
            .into_iter()
            .map(|name| {
                serde_plain::from_str::<RV64Extensions>(&name).map_err(|error| {
                    format!("failed to parse RV64 extension `{name}` in {field}: {error}")
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(IsaExtensions::rv64),
    }
}
