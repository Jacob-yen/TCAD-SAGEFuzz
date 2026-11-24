use std::{
    fs, io,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use thiserror::Error;
use toml::Value;

use crate::testcase::TestCaseConfig;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("failed to parse TOML {path}: {source}")]
    ParseToml {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("testcase config {path} must contain a [testcase_config] table")]
    MissingTestCaseTable { path: PathBuf },

    #[error(
        "testcase config {path} must explicitly set [testcase_config].unaligned_access_required to true or false"
    )]
    MissingUnalignedAccessRequired { path: PathBuf },

    #[error("testcase config {path} must contain [testcase_config].hart_config")]
    MissingHartConfig { path: PathBuf },
}

#[derive(Debug, Clone, Deserialize)]
struct TestCaseConfigFile {
    testcase_config: TestCaseConfig,
}

pub fn load_testcase_config(path: &Path) -> Result<TestCaseConfig, ConfigError> {
    let content = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    parse_testcase_config_toml(&content, path)
}

pub fn parse_testcase_config_toml(
    content: &str,
    source_path: impl AsRef<Path>,
) -> Result<TestCaseConfig, ConfigError> {
    let source_path = source_path.as_ref().to_path_buf();
    let value: Value = toml::from_str(content).map_err(|source| ConfigError::ParseToml {
        path: source_path.clone(),
        source,
    })?;
    let testcase_table = value
        .get("testcase_config")
        .and_then(Value::as_table)
        .ok_or_else(|| ConfigError::MissingTestCaseTable {
            path: source_path.clone(),
        })?;

    if !testcase_table.contains_key("unaligned_access_required") {
        return Err(ConfigError::MissingUnalignedAccessRequired { path: source_path });
    }

    if !testcase_table.contains_key("hart_config") {
        return Err(ConfigError::MissingHartConfig {
            path: source_path.clone(),
        });
    }

    let parsed: TestCaseConfigFile =
        value
            .try_into()
            .map_err(|source: toml::de::Error| ConfigError::ParseToml {
                path: source_path.clone(),
                source,
            })?;

    Ok(parsed.testcase_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn base_testcase_config() -> &'static str {
        r#"
[testcase_config]
isa_base = "Rv32"
extension_insts_count = 16
mem_size = 32
float_extension = "none"
unaligned_access_required = false
hart_config = { kind = "single" }

[testcase_config.random_config.memory_access_config]
memory_base_address_range = [0, 0]
memory_offset_immediate_range = [0, 16]
memory_base_register_number_range = [8, 12]

[testcase_config.random_config.register_operand_config]
integer_register_number_range = [13, 20]
floating_point_register_number_range = [13, 20]
vector_register_number_range = [13, 20]
"#
    }

    #[test]
    fn testcase_loader_requires_explicit_hart_config() {
        let dir = tempdir().expect("tempdir should be created");
        let path = dir.path().join("missing_hart_config.toml");
        fs::write(
            &path,
            base_testcase_config().replace("hart_config = { kind = \"single\" }\n", ""),
        )
        .expect("testcase config should be written");
        let error = load_testcase_config(&path).expect_err("hart_config must be explicit");

        assert!(error.to_string().contains("[testcase_config].hart_config"));
    }

    #[test]
    fn testcase_loader_accepts_dual_hart_config() {
        let dir = tempdir().expect("tempdir should be created");
        let path = dir.path().join("dual_hart.toml");
        fs::write(
            &path,
            base_testcase_config().replace(
                "hart_config = { kind = \"single\" }",
                "hart_config = { kind = \"dual\", sync = { shared_address_ratio = 0.25, interaction_slot_ratio = 0.5, independent_block_len_range = [4, 8], interaction_block_len_range = [2, 4], prefer_focused_sequences = false } }",
            ),
        )
        .expect("testcase config should be written");
        let config = load_testcase_config(&path).expect("dual-hart config should load");

        assert_eq!(config.hart_count(), 2);
    }
}
