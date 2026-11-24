use std::{fmt::Display, str::FromStr};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize, Hash, ValueEnum)]
pub enum ISABase {
    Rv32,
    Rv64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseISABaseError;

impl std::fmt::Display for ParseISABaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expected `rv32` or `rv64`")
    }
}

impl std::error::Error for ParseISABaseError {}

impl Display for ISABase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Rv32 => "RV32",
            Self::Rv64 => "RV64",
        };
        write!(f, "{name}")
    }
}

impl ISABase {
    pub fn word_size(self) -> usize {
        match self {
            Self::Rv32 => 4,
            Self::Rv64 => 8,
        }
    }

    pub fn instruction_max_access_width(self) -> usize {
        match self {
            Self::Rv32 => 4,
            Self::Rv64 => 8,
        }
    }
}

impl AsRef<str> for ISABase {
    fn as_ref(&self) -> &str {
        match self {
            Self::Rv32 => "rv32",
            Self::Rv64 => "rv64",
        }
    }
}

impl FromStr for ISABase {
    type Err = ParseISABaseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rv32" => Ok(Self::Rv32),
            "rv64" => Ok(Self::Rv64),
            _ => Err(ParseISABaseError),
        }
    }
}
