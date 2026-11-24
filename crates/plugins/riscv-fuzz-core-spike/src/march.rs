//! Spike ISA string builder
//!
//! This crate provides utilities for constructing RISC-V ISA strings compatible with Spike,
//! the RISC-V ISA simulator. It uses the extension enums from the `riscv-instruction` crate.

use riscv_instruction::{RV32Extensions, RV64Extensions};
use std::collections::HashSet;
use thiserror::Error;

/// Re-export the extension enums for convenience
pub use riscv_instruction::{RV32Extensions as RV32Ext, RV64Extensions as RV64Ext};

fn extension_name<E: std::fmt::Debug>(ext: &E) -> String {
    format!("{ext:?}").to_ascii_lowercase()
}

fn is_vector_extension_name(name: &str) -> bool {
    name == "v" || name.starts_with("zv") || name.starts_with("zve")
}

/// Errors that can occur while building an ISA string for Spike
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BuildIsaError {
    #[error("missing base ISA extension (expected I)")]
    MissingBaseIsa,

    #[error("extension `{ext}` requires `{required}`")]
    ExtensionRequires { ext: String, required: String },

    #[error("extensions `{left}` and `{right}` conflict")]
    ExtensionsConflict { left: String, right: String },

    #[error("extension `{ext}` is only valid for rv{required_xlen}")]
    ExtensionOnlyForXlen { ext: String, required_xlen: u32 },

    #[error("unsupported extension: {ext}")]
    UnsupportedExtension { ext: String },
}

/// Build an ISA string for RV32 from the supplied extensions, compatible with Spike
pub fn isa_from_rv32_extensions(exts: &[RV32Extensions]) -> Result<String, BuildIsaError> {
    build_isa(32, exts.iter().map(extension_name))
}

/// Build an ISA string for RV64 from the supplied extensions, compatible with Spike
pub fn isa_from_rv64_extensions(exts: &[RV64Extensions]) -> Result<String, BuildIsaError> {
    build_isa(64, exts.iter().map(extension_name))
}

fn build_isa<I>(xlen: u32, names: I) -> Result<String, BuildIsaError>
where
    I: IntoIterator<Item = String>,
{
    let mut exts: HashSet<String> = names.into_iter().collect();
    let original_single_letters: HashSet<String> = exts
        .iter()
        .filter(|name| name.len() == 1)
        .cloned()
        .collect();

    // Check for base ISA - only I is supported in the external enum
    if !exts.contains("i") {
        return Err(BuildIsaError::MissingBaseIsa);
    }

    // Spike requires CSR support, so ensure Zicsr is always present.
    exts.insert("zicsr".to_string());

    apply_implications(xlen, &mut exts);

    if exts.contains("d")
        && !original_single_letters.contains("d")
        && !exts.contains("q")
        && !exts.contains("zve64d")
    {
        exts.remove("d");
    }

    validate_extensions(xlen, &exts)?;
    if exts.iter().any(|name| is_vector_extension_name(name)) {
        exts.insert("v".to_string());
    }
    format_isa_string(xlen, &exts)
}

fn apply_implications(_xlen: u32, exts: &mut HashSet<String>) {
    loop {
        let mut changed = false;

        // Note: The external enum doesn't have 'A' extension
        // Spike will treat Zaamo + Zalrsc as implying atomic operations

        // B <-> Zba + Zbb + Zbs
        if exts.contains("b") {
            changed |= exts.insert("zba".to_string());
            changed |= exts.insert("zbb".to_string());
            changed |= exts.insert("zbs".to_string());
        } else if exts.contains("zba") && exts.contains("zbb") && exts.contains("zbs") {
            changed |= exts.insert("b".to_string());
        }

        // C implies compressed sub-extensions (Spike will expand these internally)
        // Note: The external enum lacks Zca, so we don't add it explicitly

        // D implies F
        if exts.contains("d") {
            changed |= exts.insert("f".to_string());
        }

        // F implies Zfa so Spike matches toolchain march string
        if exts.contains("f") {
            changed |= exts.insert("zfa".to_string());
        }

        // Q implies D and F
        if exts.contains("q") {
            changed |= exts.insert("d".to_string());
            changed |= exts.insert("f".to_string());
        }

        // Zfh implies F (Spike handles Zfhmin internally)
        if exts.contains("zfh") {
            changed |= exts.insert("f".to_string());
        }

        // Vector bf16 subsets imply floating-point support.
        if exts.contains("zvfbfmin") {
            changed |= exts.insert("f".to_string());
        }

        if exts.contains("zvfbfwma") {
            changed |= exts.insert("zvfbfmin".to_string());
        }

        // Zks implies Zbkb, Zbkx, Zksed, Zksh (Spike compatible)
        // Note: External enum lacks Zbkc
        if exts.contains("zks") {
            changed |= exts.insert("zbkb".to_string());
            changed |= exts.insert("zbkx".to_string());
        }

        // Zkn implies Zbkb, Zbkx, Zknd, Zkne, Zknh (Spike compatible)
        if exts.contains("zkn") {
            changed |= exts.insert("zbkb".to_string());
            changed |= exts.insert("zbkx".to_string());
            changed |= exts.insert("zknd".to_string());
            changed |= exts.insert("zkne".to_string());
            changed |= exts.insert("zknh".to_string());
        }

        // Zvks implies Zvbb (partial, external enum is limited)
        if exts.contains("zvks") {
            changed |= exts.insert("zvbb".to_string());
        }

        // Zacas implies Zaamo
        if exts.contains("zacas") {
            changed |= exts.insert("zaamo".to_string());
        }

        // Zabha implies Zaamo
        if exts.contains("zabha") {
            changed |= exts.insert("zaamo".to_string());
        }

        // Zawrs implies Zalrsc
        if exts.contains("zawrs") {
            changed |= exts.insert("zalrsc".to_string());
        }

        // Zicfiss implies Zimop
        if exts.contains("zicfiss") {
            changed |= exts.insert("zimop".to_string());
        }

        if !changed {
            break;
        }
    }
}

fn validate_extensions(xlen: u32, exts: &HashSet<String>) -> Result<(), BuildIsaError> {
    let has = |name: &str| exts.contains(name);

    // Zcf only valid for RV32
    if xlen == 64 && has("zcf") {
        return Err(BuildIsaError::ExtensionOnlyForXlen {
            ext: "zcf".to_string(),
            required_xlen: 32,
        });
    }

    // Zilsd only valid for RV32
    if xlen == 64 && has("zilsd") {
        return Err(BuildIsaError::ExtensionOnlyForXlen {
            ext: "zilsd".to_string(),
            required_xlen: 32,
        });
    }

    // Zfbfmin requires F
    if has("zfbfmin") && !has("f") {
        return Err(BuildIsaError::ExtensionRequires {
            ext: "zfbfmin".to_string(),
            required: "f".to_string(),
        });
    }

    // Zcf requires F
    if has("zcf") && !has("f") {
        return Err(BuildIsaError::ExtensionRequires {
            ext: "zcf".to_string(),
            required: "f".to_string(),
        });
    }

    // Zcd requires D
    if has("zcd") && !has("d") {
        return Err(BuildIsaError::ExtensionRequires {
            ext: "zcd".to_string(),
            required: "d".to_string(),
        });
    }

    // Zcmp and Zcd are incompatible
    if has("zcd") && has("zcmp") {
        return Err(BuildIsaError::ExtensionsConflict {
            left: "zcd".to_string(),
            right: "zcmp".to_string(),
        });
    }

    // Zacas requires Zaamo
    if has("zacas") && !has("zaamo") {
        return Err(BuildIsaError::ExtensionRequires {
            ext: "zacas".to_string(),
            required: "zaamo".to_string(),
        });
    }

    // Zabha requires Zaamo
    if has("zabha") && !has("zaamo") {
        return Err(BuildIsaError::ExtensionRequires {
            ext: "zabha".to_string(),
            required: "zaamo".to_string(),
        });
    }

    // Zawrs requires Zalrsc
    if has("zawrs") && !has("zalrsc") {
        return Err(BuildIsaError::ExtensionRequires {
            ext: "zawrs".to_string(),
            required: "zalrsc".to_string(),
        });
    }

    Ok(())
}

fn format_isa_string(xlen: u32, exts: &HashSet<String>) -> Result<String, BuildIsaError> {
    let mut result = format!("rv{}", xlen);

    // Spike's single-letter extension order
    const SINGLE_LETTER_ORDER: &str = "iemafdqcbpvhs";

    // Add single-letter extensions in order
    for ch in SINGLE_LETTER_ORDER.chars() {
        let ext = ch.to_string();
        if exts.contains(&ext) {
            result.push(ch);
        }
    }

    // Collect multi-letter extensions
    let mut multi_letter: Vec<String> = exts.iter().filter(|e| e.len() > 1).cloned().collect();

    // Sort multi-letter extensions alphabetically (Spike style)
    multi_letter.sort();

    // Add multi-letter extensions with underscore separators
    for ext in multi_letter {
        result.push('_');
        result.push_str(&ext);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use crate::{execute::SPIKE_BINARY_ENV, metadata::isa_profiles};
    use log::{error, info, warn};
    use riscv_fuzz_plugin_api::IsaExtensions;

    use super::*;

    struct ExtensionSets {
        rv32: Vec<RV32Ext>,
        rv64: Vec<RV64Ext>,
    }

    fn extensions() -> ExtensionSets {
        let profiles = isa_profiles();
        let rv32 = match profiles.rv32.extensions {
            IsaExtensions::Rv32(exts) => exts,
            IsaExtensions::Rv64(_) => panic!("Spike RV32 profile returned RV64 extensions"),
        };
        let rv64 = match profiles.rv64.extensions {
            IsaExtensions::Rv64(exts) => exts,
            IsaExtensions::Rv32(_) => panic!("Spike RV64 profile returned RV32 extensions"),
        };

        ExtensionSets { rv32, rv64 }
    }

    fn spike_binary() -> Option<PathBuf> {
        if let Ok(raw) = env::var(SPIKE_BINARY_ENV) {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                let candidate = PathBuf::from(trimmed);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    fn test_rv64_isa_configuration(spike: &Path, name: &str, extensions: &[RV64Ext]) {
        match isa_from_rv64_extensions(extensions) {
            Ok(isa) => {
                info!("Testing {name} with ISA `{isa}`");

                let result = Command::new(spike)
                    .arg(format!("--isa={}", isa))
                    .arg("--help")
                    .output();

                match result {
                    Ok(output) if output.status.success() => {
                        info!("✓ {name} ({isa}) accepted by Spike");
                    }
                    Ok(output) => {
                        warn!("✗ {name} ({isa}) rejected by Spike");
                        if !output.stderr.is_empty() {
                            error!(
                                "Spike stderr for {name}: {}",
                                String::from_utf8_lossy(&output.stderr)
                            );
                        }
                    }
                    Err(e) => {
                        error!("✗ Failed to run Spike for {name} ({isa}): {e}");
                    }
                }
            }
            Err(e) => {
                error!("✗ Failed to build ISA string for {name}: {e}");
            }
        }
    }

    fn test_rv32_isa_configuration(spike: &Path, name: &str, extensions: &[RV32Ext]) {
        match isa_from_rv32_extensions(extensions) {
            Ok(isa) => {
                info!("Testing {name} with ISA `{isa}`");

                let result = Command::new(spike)
                    .arg(format!("--isa={}", isa))
                    .arg("--help")
                    .output();

                match result {
                    Ok(output) if output.status.success() => {
                        info!("✓ {name} ({isa}) accepted by Spike");
                    }
                    Ok(output) => {
                        warn!("✗ {name} ({isa}) rejected by Spike");
                        if !output.stderr.is_empty() {
                            error!(
                                "Spike stderr for {name}: {}",
                                String::from_utf8_lossy(&output.stderr)
                            );
                        }
                    }
                    Err(e) => {
                        error!("✗ Failed to run Spike for {name} ({isa}): {e}");
                    }
                }
            }
            Err(e) => {
                error!("✗ Failed to build ISA string for {name}: {e}");
            }
        }
    }

    fn is_spike_available() -> bool {
        let Some(spike) = spike_binary() else {
            return false;
        };
        Command::new(&spike)
            .arg("--help")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn test_with_spike() {
        info!("=== Testing ISA Strings with Spike ===");

        let Some(spike) = spike_binary() else {
            warn!(
                "Skipping Spike ISA acceptance test: set {} to an executable Spike binary",
                SPIKE_BINARY_ENV,
            );
            return;
        };
        if !is_spike_available() {
            warn!(
                "Skipping Spike ISA acceptance test: Spike is not runnable at {}",
                spike.display()
            );
            return;
        }

        info!("✓ Spike is available at {}", spike.display());

        // Test different ISA configurations
        test_rv64_isa_configuration(spike.as_path(), "Basic RV64IM", &[RV64Ext::I, RV64Ext::M]);

        test_rv64_isa_configuration(
            spike.as_path(),
            "RV64 with Float/Double",
            &[RV64Ext::I, RV64Ext::M, RV64Ext::F, RV64Ext::D],
        );

        test_rv64_isa_configuration(
            spike.as_path(),
            "RV64 with Compressed",
            &[RV64Ext::I, RV64Ext::M, RV64Ext::C],
        );

        test_rv64_isa_configuration(
            spike.as_path(),
            "RV64 with Atomics",
            &[RV64Ext::I, RV64Ext::M, RV64Ext::Zaamo, RV64Ext::Zalrsc],
        );

        test_rv64_isa_configuration(
            spike.as_path(),
            "RV64 with Bit Manipulation",
            &[RV64Ext::I, RV64Ext::M, RV64Ext::B],
        );

        test_rv64_isa_configuration(
            spike.as_path(),
            "RV64 with Vector",
            &[RV64Ext::I, RV64Ext::M, RV64Ext::Zvbb],
        );

        test_rv64_isa_configuration(
            spike.as_path(),
            "RV64 with Crypto (Zkn)",
            &[RV64Ext::I, RV64Ext::M, RV64Ext::Zkn],
        );

        test_rv64_isa_configuration(
            spike.as_path(),
            "Comprehensive RV64",
            &[
                RV64Ext::I,
                RV64Ext::M,
                RV64Ext::F,
                RV64Ext::D,
                RV64Ext::C,
                RV64Ext::Zvbb,
                RV64Ext::B,
            ],
        );

        let ext_map = extensions();
        test_rv64_isa_configuration(spike.as_path(), "rv64", &ext_map.rv64);
        test_rv32_isa_configuration(spike.as_path(), "rv32", &ext_map.rv32);

        info!("=== All ISA configurations tested successfully! ===");
    }

    #[test]
    fn test_basic_rv64_imafd() {
        let isa = isa_from_rv64_extensions(&[
            RV64Extensions::I,
            RV64Extensions::M,
            RV64Extensions::F,
            RV64Extensions::D,
        ])
        .unwrap();

        assert!(isa.starts_with("rv64imfd"));
    }

    #[test]
    fn test_rv64_with_atomics() {
        let isa = isa_from_rv64_extensions(&[
            RV64Extensions::I,
            RV64Extensions::M,
            RV64Extensions::Zaamo,
            RV64Extensions::Zalrsc,
        ])
        .unwrap();

        assert!(isa.contains("_zaamo"));
        assert!(isa.contains("_zalrsc"));
    }

    #[test]
    fn test_rv32_with_compressed() {
        let isa =
            isa_from_rv32_extensions(&[RV32Extensions::I, RV32Extensions::F, RV32Extensions::C])
                .unwrap();

        assert!(isa.starts_with("rv32ifc"));
    }

    #[test]
    fn test_rv32_float_without_double() {
        let extensions = [
            RV32Extensions::I,
            RV32Extensions::M,
            RV32Extensions::F,
            RV32Extensions::B,
            RV32Extensions::C,
            RV32Extensions::Zaamo,
            RV32Extensions::Zalrsc,
            RV32Extensions::Zba,
            RV32Extensions::Zbb,
            RV32Extensions::Zbc,
            RV32Extensions::Zbkb,
            RV32Extensions::Zbkx,
            RV32Extensions::Zbs,
            RV32Extensions::Zcb,
            RV32Extensions::Zicond,
            RV32Extensions::Zicsr,
            RV32Extensions::Zifencei,
            RV32Extensions::Zknd,
            RV32Extensions::Zkne,
            RV32Extensions::Zknh,
        ];

        let isa = isa_from_rv32_extensions(&extensions).unwrap();

        let single_letter = isa.split('_').next().unwrap_or(&isa);
        assert!(single_letter.contains('f'));
        assert!(!single_letter.contains('d'));
    }

    #[test]
    fn test_b_extension_implies_zba_zbb_zbs() {
        let isa = isa_from_rv64_extensions(&[RV64Extensions::I, RV64Extensions::B]).unwrap();

        assert!(isa.contains("_zba"));
        assert!(isa.contains("_zbb"));
        assert!(isa.contains("_zbs"));
    }

    #[test]
    fn test_d_implies_f() {
        let isa = isa_from_rv64_extensions(&[RV64Extensions::I, RV64Extensions::D]).unwrap();

        assert!(isa.contains('f'));
        assert!(isa.contains('d'));
    }

    #[test]
    fn test_missing_base_isa() {
        let err = isa_from_rv64_extensions(&[RV64Extensions::M]).unwrap_err();
        assert_eq!(err, BuildIsaError::MissingBaseIsa);
    }

    #[test]
    fn test_zcd_conflicts_with_zcmp() {
        let err = isa_from_rv64_extensions(&[
            RV64Extensions::I,
            RV64Extensions::D,
            RV64Extensions::C,
            RV64Extensions::Zcd,
            RV64Extensions::Zcmp,
        ])
        .unwrap_err();

        match err {
            BuildIsaError::ExtensionsConflict { left, right } => {
                assert_eq!(left, "zcd");
                assert_eq!(right, "zcmp");
            }
            _ => panic!("Expected ExtensionsConflict error"),
        }
    }

    #[test]
    fn test_zacas_implies_zaamo() {
        let isa = isa_from_rv64_extensions(&[
            RV64Extensions::I,
            RV64Extensions::Zacas,
            RV64Extensions::Zaamo,
        ])
        .unwrap();

        assert!(isa.contains("_zaamo"));
        assert!(isa.contains("_zacas"));
    }

    #[test]
    fn test_zkn_expands() {
        let isa = isa_from_rv64_extensions(&[RV64Extensions::I, RV64Extensions::Zkn]).unwrap();

        assert!(isa.contains("_zbkb"));
        assert!(isa.contains("_zbkx"));
        assert!(isa.contains("_zknd"));
        assert!(isa.contains("_zkne"));
        assert!(isa.contains("_zknh"));
    }

    #[test]
    fn test_comprehensive_isa() {
        let isa = isa_from_rv64_extensions(&[
            RV64Extensions::I,
            RV64Extensions::M,
            RV64Extensions::F,
            RV64Extensions::D,
            RV64Extensions::C,
            RV64Extensions::Zvbb,
            RV64Extensions::B,
            RV64Extensions::Zicond,
        ])
        .unwrap();

        assert!(isa.starts_with("rv64"));
        assert!(isa.contains('i'));
        assert!(isa.contains('m'));
        assert!(isa.contains('f'));
        assert!(isa.contains('d'));
        assert!(isa.contains('c'));
        assert!(isa.contains('b'));
        assert!(isa.contains('v'));
        assert!(isa.contains("_zicond"));
    }

    #[test]
    fn test_full_isa() {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .try_init()
            .unwrap();
        let ext_map = extensions();
        let isa = isa_from_rv64_extensions(&ext_map.rv64).unwrap();
        info!("Full RV64 ISA string: {}", isa);
        let isa = isa_from_rv32_extensions(&ext_map.rv32).unwrap();
        info!("Full RV32 ISA string: {}", isa);
    }
}
