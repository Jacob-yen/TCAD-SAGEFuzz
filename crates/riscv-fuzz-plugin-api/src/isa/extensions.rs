use riscv_instruction::{RV32Extensions, RV64Extensions};
use serde::{Deserialize, Serialize};

use super::ISABase;

fn is_vector_extension_name(name: &str) -> bool {
    name == "v" || name.starts_with("zv") || name.starts_with("zve")
}

fn rv32_extension_name(extension: &RV32Extensions) -> String {
    format!("{extension:?}").to_ascii_lowercase()
}

fn rv64_extension_name(extension: &RV64Extensions) -> String {
    format!("{extension:?}").to_ascii_lowercase()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "isa_base", content = "extensions", rename_all = "lowercase")]
pub enum IsaExtensions {
    Rv32(Vec<RV32Extensions>),
    Rv64(Vec<RV64Extensions>),
}

impl IsaExtensions {
    pub fn rv32(extensions: Vec<RV32Extensions>) -> Self {
        Self::Rv32(extensions)
    }

    pub fn rv64(extensions: Vec<RV64Extensions>) -> Self {
        Self::Rv64(extensions)
    }

    pub fn isa_base(&self) -> ISABase {
        match self {
            Self::Rv32(_) => ISABase::Rv32,
            Self::Rv64(_) => ISABase::Rv64,
        }
    }

    pub fn as_rv32(&self) -> Option<&[RV32Extensions]> {
        match self {
            Self::Rv32(extensions) => Some(extensions),
            Self::Rv64(_) => None,
        }
    }

    pub fn as_rv32_mut(&mut self) -> Option<&mut Vec<RV32Extensions>> {
        match self {
            Self::Rv32(extensions) => Some(extensions),
            Self::Rv64(_) => None,
        }
    }

    pub fn as_rv64(&self) -> Option<&[RV64Extensions]> {
        match self {
            Self::Rv32(_) => None,
            Self::Rv64(extensions) => Some(extensions),
        }
    }

    pub fn as_rv64_mut(&mut self) -> Option<&mut Vec<RV64Extensions>> {
        match self {
            Self::Rv32(_) => None,
            Self::Rv64(extensions) => Some(extensions),
        }
    }

    pub fn contains_rv32(&self, extension: RV32Extensions) -> bool {
        matches!(self, Self::Rv32(extensions) if extensions.contains(&extension))
    }

    pub fn contains_rv64(&self, extension: RV64Extensions) -> bool {
        matches!(self, Self::Rv64(extensions) if extensions.contains(&extension))
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Rv32(extensions) => extensions.is_empty(),
            Self::Rv64(extensions) => extensions.is_empty(),
        }
    }

    pub fn has_enabled_float(&self) -> bool {
        match self {
            Self::Rv32(extensions) => {
                extensions.contains(&RV32Extensions::F)
                    || extensions.contains(&RV32Extensions::D)
                    || extensions.contains(&RV32Extensions::Q)
            }
            Self::Rv64(extensions) => {
                extensions.contains(&RV64Extensions::F)
                    || extensions.contains(&RV64Extensions::D)
                    || extensions.contains(&RV64Extensions::Q)
            }
        }
    }

    pub fn has_enabled_vector(&self) -> bool {
        match self {
            Self::Rv32(extensions) => extensions
                .iter()
                .map(rv32_extension_name)
                .any(|name| is_vector_extension_name(&name)),
            Self::Rv64(extensions) => extensions
                .iter()
                .map(rv64_extension_name)
                .any(|name| is_vector_extension_name(&name)),
        }
    }
}
