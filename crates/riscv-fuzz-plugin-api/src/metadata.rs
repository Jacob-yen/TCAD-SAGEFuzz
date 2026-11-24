use serde::{Deserialize, Serialize};

use crate::{CoverageMode, ISABase, IsaExtensions};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentMode {
    RequireAligned,
    AllowUnaligned,
}

impl AlignmentMode {
    pub fn allows_unaligned(self) -> bool {
        matches!(self, AlignmentMode::AllowUnaligned)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnalignedAccessSupport {
    pub aligned: bool,
    pub unaligned: bool,
}

impl UnalignedAccessSupport {
    pub const NONE: Self = Self {
        aligned: false,
        unaligned: false,
    };

    pub fn supports(self, mode: AlignmentMode) -> bool {
        match mode {
            AlignmentMode::AllowUnaligned => self.unaligned,
            AlignmentMode::RequireAligned => self.aligned,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FloatExtension {
    /// 不使用浮点（默认）
    #[default]
    None,
    /// 仅使用 F 扩展
    F,
    /// 同时使用 F 与 D
    FD,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageModeSupport {
    pub off: bool,
    pub light: bool,
    pub full: bool,
}

impl CoverageModeSupport {
    pub const NONE: Self = Self {
        off: false,
        light: false,
        full: false,
    };

    pub const ALL: Self = Self {
        off: true,
        light: true,
        full: true,
    };

    pub fn supports(self, mode: CoverageMode) -> bool {
        match mode {
            CoverageMode::Off => self.off,
            CoverageMode::Light => self.light,
            CoverageMode::Full => self.full,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsaExecutionProfile {
    pub supported: bool,
    #[serde(default)]
    pub supports_dual_hart: bool,
    #[serde(default)]
    pub coverage_mode_support: CoverageModeSupport,
    pub unaligned_access_support: UnalignedAccessSupport,
    pub extensions: IsaExtensions,
}

impl IsaExecutionProfile {
    pub fn unsupported(isa_base: ISABase) -> Self {
        Self {
            supported: false,
            supports_dual_hart: false,
            coverage_mode_support: CoverageModeSupport::NONE,
            unaligned_access_support: UnalignedAccessSupport::NONE,
            extensions: empty_extensions(isa_base),
        }
    }

    pub fn supports_float_choice(&self, choice: FloatExtension) -> bool {
        if !self.supported {
            return false;
        }

        matches!(
            (self.max_float_extension(), choice),
            (_, FloatExtension::None)
                | (FloatExtension::FD, FloatExtension::F | FloatExtension::FD)
                | (FloatExtension::F, FloatExtension::F)
        )
    }

    pub fn supports_alignment_mode(&self, mode: AlignmentMode) -> bool {
        self.supported && self.unaligned_access_support.supports(mode)
    }

    pub fn supports_coverage_mode(&self, mode: CoverageMode) -> bool {
        self.supported && self.coverage_mode_support.supports(mode)
    }

    pub fn active_float_extension(&self, extensions: &IsaExtensions) -> FloatExtension {
        if !self.supported {
            return FloatExtension::None;
        }

        match extensions {
            IsaExtensions::Rv32(extensions) => {
                let has_d = extensions.contains(&riscv_instruction::RV32Extensions::D);
                let has_f = extensions.contains(&riscv_instruction::RV32Extensions::F);
                if has_d {
                    FloatExtension::FD
                } else if has_f {
                    FloatExtension::F
                } else {
                    FloatExtension::None
                }
            }
            IsaExtensions::Rv64(extensions) => {
                let has_d = extensions.contains(&riscv_instruction::RV64Extensions::D);
                let has_f = extensions.contains(&riscv_instruction::RV64Extensions::F);
                if has_d {
                    FloatExtension::FD
                } else if has_f {
                    FloatExtension::F
                } else {
                    FloatExtension::None
                }
            }
        }
    }

    pub fn max_float_extension(&self) -> FloatExtension {
        self.active_float_extension(&self.extensions)
    }

    pub fn supported_extensions(&self) -> &IsaExtensions {
        &self.extensions
    }

    pub fn validate_float_choice(&self, choice: FloatExtension) -> Result<(), String> {
        if !self.supported {
            return Err("ISA profile is not supported".to_string());
        }

        if !self.supports_float_choice(choice) {
            return Err(format!(
                "requested float mode `{choice:?}` is not supported by this plugin"
            ));
        }

        Ok(())
    }

    pub fn extensions_for_float_choice(
        &self,
        choice: FloatExtension,
    ) -> Result<IsaExtensions, String> {
        self.validate_float_choice(choice)?;
        Ok(self.supported_extensions().clone())
    }

    pub fn resolve_extensions(
        &self,
        extension_override: Option<&IsaExtensions>,
        float_mode_override: Option<FloatExtension>,
    ) -> Result<IsaExtensions, String> {
        if extension_override.is_some() && float_mode_override.is_some() {
            return Err(
                "execute request cannot set both `extension_override` and `float_mode_override`"
                    .to_string(),
            );
        }

        if let Some(extensions) = extension_override {
            self.validate_extension_override(extensions)?;
            return Ok(extensions.clone());
        }

        if let Some(choice) = float_mode_override {
            return self.extensions_for_float_choice(choice);
        }

        Ok(self.supported_extensions().clone())
    }

    pub fn validate_extension_override(&self, extensions: &IsaExtensions) -> Result<(), String> {
        if !self.supported {
            return Err("ISA profile is not supported".to_string());
        }

        match (&self.extensions, extensions) {
            (IsaExtensions::Rv32(supported), IsaExtensions::Rv32(requested)) => {
                for extension in requested {
                    if !supported.contains(extension) {
                        return Err(format!(
                            "requested RV32 extension `{extension:?}` is not supported by this plugin"
                        ));
                    }
                }
            }
            (IsaExtensions::Rv64(supported), IsaExtensions::Rv64(requested)) => {
                for extension in requested {
                    if !supported.contains(extension) {
                        return Err(format!(
                            "requested RV64 extension `{extension:?}` is not supported by this plugin"
                        ));
                    }
                }
            }
            (IsaExtensions::Rv32(_), IsaExtensions::Rv64(_))
            | (IsaExtensions::Rv64(_), IsaExtensions::Rv32(_)) => {
                return Err(
                    "requested extension override ISA does not match the selected ISA base"
                        .to_string(),
                );
            }
        }

        let requested_float = self.active_float_extension(extensions);
        self.validate_float_choice(requested_float)?;

        Ok(())
    }
}

fn empty_extensions(isa_base: ISABase) -> IsaExtensions {
    match isa_base {
        ISABase::Rv32 => IsaExtensions::rv32(Vec::new()),
        ISABase::Rv64 => IsaExtensions::rv64(Vec::new()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsaExecutionProfileMap {
    pub rv32: IsaExecutionProfile,
    pub rv64: IsaExecutionProfile,
}

impl IsaExecutionProfileMap {
    pub fn for_isa(&self, isa_base: ISABase) -> &IsaExecutionProfile {
        match isa_base {
            ISABase::Rv32 => &self.rv32,
            ISABase::Rv64 => &self.rv64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorePluginMetadata {
    pub core_id: String,
    pub isa_profiles: IsaExecutionProfileMap,
    pub program_mem_range: (u64, u64),
}

#[cfg(test)]
mod tests {
    use super::*;
    use riscv_instruction::{RV32Extensions, RV64Extensions};

    fn toggled_profile_rv32() -> IsaExecutionProfile {
        IsaExecutionProfile {
            supported: true,
            supports_dual_hart: false,
            coverage_mode_support: CoverageModeSupport::ALL,
            unaligned_access_support: UnalignedAccessSupport {
                aligned: true,
                unaligned: false,
            },
            extensions: IsaExtensions::rv32(vec![
                RV32Extensions::I,
                RV32Extensions::F,
                RV32Extensions::D,
                RV32Extensions::C,
                RV32Extensions::Zca,
                RV32Extensions::Zcf,
                RV32Extensions::Zcd,
                RV32Extensions::Zicsr,
            ]),
        }
    }

    fn toggled_profile_rv64() -> IsaExecutionProfile {
        IsaExecutionProfile {
            supported: true,
            supports_dual_hart: false,
            coverage_mode_support: CoverageModeSupport::ALL,
            unaligned_access_support: UnalignedAccessSupport {
                aligned: true,
                unaligned: false,
            },
            extensions: IsaExtensions::rv64(vec![
                RV64Extensions::I,
                RV64Extensions::F,
                RV64Extensions::D,
                RV64Extensions::C,
                RV64Extensions::Zca,
                RV64Extensions::Zcd,
                RV64Extensions::Zicsr,
            ]),
        }
    }

    #[test]
    fn float_choice_returns_full_supported_rv32_extension_set() {
        let extensions = toggled_profile_rv32()
            .extensions_for_float_choice(FloatExtension::F)
            .expect("resolve rv32 F-capable extensions");

        let IsaExtensions::Rv32(values) = extensions else {
            panic!("expected rv32 extensions");
        };

        assert!(values.contains(&RV32Extensions::F));
        assert!(values.contains(&RV32Extensions::D));
        assert!(values.contains(&RV32Extensions::Zcf));
        assert!(values.contains(&RV32Extensions::Zcd));
    }

    #[test]
    fn float_choice_none_keeps_full_supported_rv32_extension_set() {
        let extensions = toggled_profile_rv32()
            .extensions_for_float_choice(FloatExtension::None)
            .expect("resolve rv32 integer-only request");

        let IsaExtensions::Rv32(values) = extensions else {
            panic!("expected rv32 extensions");
        };

        assert!(values.contains(&RV32Extensions::F));
        assert!(values.contains(&RV32Extensions::D));
        assert!(values.contains(&RV32Extensions::Zcf));
        assert!(values.contains(&RV32Extensions::Zcd));
    }

    #[test]
    fn float_choice_returns_full_supported_rv64_extension_set() {
        let extensions = toggled_profile_rv64()
            .extensions_for_float_choice(FloatExtension::F)
            .expect("resolve rv64 F-capable extensions");

        let IsaExtensions::Rv64(values) = extensions else {
            panic!("expected rv64 extensions");
        };

        assert!(values.contains(&RV64Extensions::F));
        assert!(values.contains(&RV64Extensions::D));
        assert!(values.contains(&RV64Extensions::Zcd));
    }
}
