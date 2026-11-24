use riscv_fuzz_plugin_api::{ExecuteRequest, RiscvCorePlugin};

use crate::error::CapabilityError;

pub(crate) fn validate_execute_request_metadata(
    plugin: &dyn RiscvCorePlugin,
    core_id: &str,
    request: &ExecuteRequest<'_>,
) -> Result<(), CapabilityError> {
    let profile = plugin.metadata().isa_profiles.for_isa(request.isa_base);
    if !profile.supported {
        return Err(CapabilityError::UnsupportedIsaBase {
            core_id: core_id.to_owned(),
            isa_base: request.isa_base,
        });
    }
    let hart_count = request.effective_hart_count();
    if hart_count > 1 && !profile.supports_dual_hart {
        return Err(CapabilityError::UnsupportedHartCount {
            core_id: core_id.to_owned(),
            isa_base: request.isa_base,
            hart_count,
        });
    }
    if !profile.supports_alignment_mode(request.alignment_mode) {
        return Err(CapabilityError::UnsupportedAlignmentMode {
            core_id: core_id.to_owned(),
            isa_base: request.isa_base,
            alignment_mode: request.alignment_mode,
        });
    }
    if !profile.supports_coverage_mode(request.coverage_mode) {
        return Err(CapabilityError::UnsupportedCoverageMode {
            core_id: core_id.to_owned(),
            isa_base: request.isa_base,
            coverage_mode: request.coverage_mode,
        });
    }
    profile
        .resolve_extensions(request.extension_override, request.float_mode_override)
        .map_err(|message| CapabilityError::InvalidRequest {
            core_id: core_id.to_owned(),
            message,
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use riscv_fuzz_plugin_api::{
        AlignmentMode, ConfigError, CorePluginMetadata, CoverageMode, CoverageModeSupport,
        ExecutionOutput, ISABase, IsaExecutionProfile, IsaExecutionProfileMap,
        PerHartTypedSequences, UnalignedAccessSupport, metadata::FloatExtension,
    };
    use riscv_instruction::{AsmDecodeWithContext, IsaDecodeContext, RiscvInstruction};
    use riscv_instruction_types::InstructionSequence;
    use std::path::Path;

    struct TestPlugin {
        metadata: CorePluginMetadata,
    }

    impl RiscvCorePlugin for TestPlugin {
        fn metadata(&self) -> &CorePluginMetadata {
            &self.metadata
        }

        fn linker_script(&self) -> &str {
            ""
        }

        fn build_asm(
            &self,
            _hart_inputs: PerHartTypedSequences<'_>,
            _isa_base: ISABase,
        ) -> Result<String, ConfigError> {
            Ok(String::new())
        }

        fn execute(
            &self,
            _request: ExecuteRequest<'_>,
        ) -> Result<ExecutionOutput, riscv_fuzz_plugin_api::ImplExecutionError> {
            Ok(ExecutionOutput::empty("test", ISABase::Rv32))
        }

        fn validate_execute_request(
            &self,
            _request: &ExecuteRequest<'_>,
        ) -> Result<(), riscv_fuzz_plugin_api::ImplExecutionError> {
            Ok(())
        }
    }

    fn request<'a>(hart_inputs: PerHartTypedSequences<'a>) -> ExecuteRequest<'a> {
        ExecuteRequest {
            run_root: Path::new("/tmp"),
            isa_base: ISABase::Rv32,
            hart_inputs,
            extension_override: None,
            float_mode_override: None,
            alignment_mode: AlignmentMode::RequireAligned,
            coverage_mode: CoverageMode::Off,
        }
    }

    #[test]
    fn metadata_validate_rejects_unsupported_hart_count() {
        let plugin = TestPlugin {
            metadata: CorePluginMetadata {
                core_id: "test".to_string(),
                isa_profiles: IsaExecutionProfileMap {
                    rv32: IsaExecutionProfile {
                        supported: true,
                        supports_dual_hart: false,
                        coverage_mode_support: CoverageModeSupport::ALL,
                        unaligned_access_support: UnalignedAccessSupport {
                            aligned: true,
                            unaligned: false,
                        },
                        extensions: riscv_fuzz_plugin_api::IsaExtensions::rv32(Vec::new()),
                    },
                    rv64: IsaExecutionProfile::unsupported(ISABase::Rv64),
                },
                program_mem_range: (0, 0),
            },
        };
        let insts = vec![InstructionSequence::new(
            RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
                .unwrap(),
        )];
        let hart_inputs = [(0, insts.as_slice()), (1, insts.as_slice())];

        let error =
            validate_execute_request_metadata(&plugin, "test", &request(&hart_inputs)).unwrap_err();
        assert!(matches!(
            error,
            CapabilityError::UnsupportedHartCount { hart_count: 2, .. }
        ));
    }

    #[test]
    fn metadata_validate_rejects_unsupported_float_mode_request() {
        let plugin = TestPlugin {
            metadata: CorePluginMetadata {
                core_id: "test".to_string(),
                isa_profiles: IsaExecutionProfileMap {
                    rv32: IsaExecutionProfile {
                        supported: true,
                        supports_dual_hart: true,
                        coverage_mode_support: CoverageModeSupport::ALL,
                        unaligned_access_support: UnalignedAccessSupport {
                            aligned: true,
                            unaligned: false,
                        },
                        extensions: riscv_fuzz_plugin_api::IsaExtensions::rv32(Vec::new()),
                    },
                    rv64: IsaExecutionProfile::unsupported(ISABase::Rv64),
                },
                program_mem_range: (0, 0),
            },
        };
        let insts = vec![InstructionSequence::new(
            RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
                .unwrap(),
        )];
        let hart_inputs = [(0, insts.as_slice())];
        let error = validate_execute_request_metadata(
            &plugin,
            "test",
            &ExecuteRequest {
                run_root: Path::new("/tmp"),
                isa_base: ISABase::Rv32,
                hart_inputs: &hart_inputs,
                extension_override: None,
                float_mode_override: Some(FloatExtension::F),
                alignment_mode: AlignmentMode::RequireAligned,
                coverage_mode: CoverageMode::Off,
            },
        )
        .unwrap_err();

        assert!(matches!(error, CapabilityError::InvalidRequest { .. }));
    }
}
