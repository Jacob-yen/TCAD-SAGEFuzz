use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum CoverageMode {
    #[default]
    Off = 0,
    Full = 1,
    Light = 2,
}

impl CoverageMode {
    pub fn is_enabled(self) -> bool {
        self != Self::Off
    }
}
