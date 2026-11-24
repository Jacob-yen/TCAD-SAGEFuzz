use crate::doctor::PluginDoctorReport;
use crate::error::{ConfigError, ImplExecutionError};
use crate::execution::ExecutionOutput;
use crate::isa::ISABase;
use crate::metadata::CorePluginMetadata;
use crate::streaming::InstructionEvent;

use super::{ExecuteRequest, PerHartTypedSequences};

pub trait RiscvCorePlugin: Sync + Send {
    fn metadata(&self) -> &CorePluginMetadata;

    fn doctor(&self) -> PluginDoctorReport {
        PluginDoctorReport::empty(self.metadata().core_id.clone())
    }

    fn linker_script(&self) -> &str;

    fn build_asm(
        &self,
        hart_inputs: PerHartTypedSequences<'_>,
        isa_base: ISABase,
    ) -> Result<String, ConfigError>;

    fn validate_execute_request(
        &self,
        request: &ExecuteRequest<'_>,
    ) -> Result<(), ImplExecutionError>;

    fn execute(&self, request: ExecuteRequest<'_>) -> Result<ExecutionOutput, ImplExecutionError>;

    /// Execute the request and emit per-instruction events via `emit` as execution proceeds.
    ///
    /// The default implementation runs `execute()` to completion and then emits
    /// all instructions followed by a `Completed` event.  Plugins that support
    /// real-time streaming (e.g. Spike reading stderr line-by-line) override this
    /// to emit events earlier, enabling callers to detect the first diff and cancel
    /// remaining runs without waiting for the full ELF to finish.
    ///
    /// The callback receives an `InstructionEvent`.  Each streamed
    /// `InstructionEvent::Instruction` must already be the final result for one
    /// `(hart_id, pc)` asm row; plugins must not later resend that same row with
    /// extra writes/timing/exceptions filled in. Returning `false` from the
    /// callback signals that the caller wants early termination; the plugin should
    /// stop execution as soon as it can (best-effort: not all plugins can honour
    /// this).  Returning `true` continues normally.
    ///
    /// # Current usage status (audit, 2026-05)
    ///
    /// - Every built-in core plugin overrides this method and emits true
    ///   streaming snapshots from the simulator's live stdout/stderr or growing
    ///   trace files. The buffered default remains only for third-party plugins
    ///   that have not implemented real-time streaming yet.
    /// - The testcase engine routes normal execution through this method and
    ///   collects events back into `ExecutionOutput`. The diff runner consumes
    ///   the stream directly so it can compare ready trace instruction rows
    ///   and cancel remaining executions after the first localized divergence.
    fn execute_streaming(
        &self,
        request: ExecuteRequest<'_>,
        emit: &mut dyn FnMut(InstructionEvent) -> bool,
    ) {
        match self.execute(request) {
            Ok(output) => {
                let mut keep_going = true;
                'outer: for hart in output.per_hart.values() {
                    for asm in &hart.instructions {
                        keep_going = emit(InstructionEvent::Instruction {
                            hart_id: hart.hart_id,
                            instruction: asm.clone(),
                        });
                        if !keep_going {
                            break 'outer;
                        }
                    }
                }
                if keep_going {
                    emit(InstructionEvent::Completed { perf: output.perf });
                }
            }
            Err(error) => {
                emit(InstructionEvent::Failed { error });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AlignmentMode, AsmInstructionExecution, CorePluginMetadata, CoverageModeSupport,
        HartExecutionOutput, ISABase, InstructionTiming, IsaExecutionProfile,
        IsaExecutionProfileMap, IsaExtensions, UnalignedAccessSupport,
    };
    use riscv_instruction::{
        AsmDecodeWithContext, IsaDecodeContext, RV32Extensions, RV64Extensions, RiscvInstruction,
    };
    use riscv_instruction_types::InstructionSequence;
    use std::collections::BTreeMap;
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

        fn validate_execute_request(
            &self,
            request: &ExecuteRequest<'_>,
        ) -> Result<(), ImplExecutionError> {
            request.validate_basic_plugin_request(self.metadata())?;
            Ok(())
        }

        fn execute(
            &self,
            request: ExecuteRequest<'_>,
        ) -> Result<ExecutionOutput, ImplExecutionError> {
            let instruction =
                RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
                    .expect("test instruction should parse");
            let asm = AsmInstructionExecution {
                pc: 0x8000_0000,
                instruction: instruction.clone(),
                source_annotation: None,
                trace_inst_id: None,
                exceptions: Vec::new(),
                register_writes: Vec::new(),
                memory_writes: Vec::new(),
                timing: InstructionTiming::default(),
            };
            let mut per_hart = BTreeMap::new();
            for hart_input in request.hart_inputs {
                let &(hart_id, _) = hart_input;
                per_hart.insert(
                    hart_id,
                    HartExecutionOutput {
                        hart_id,
                        instructions: vec![asm.clone()],
                    },
                );
            }
            Ok(ExecutionOutput {
                per_hart,
                core_id: "test".to_string(),
                isa_base: request.isa_base,
                perf: Default::default(),
            })
        }
    }

    fn test_plugin() -> TestPlugin {
        TestPlugin {
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
                        extensions: IsaExtensions::rv32(vec![
                            RV32Extensions::I,
                            RV32Extensions::M,
                            RV32Extensions::F,
                            RV32Extensions::Zicsr,
                        ]),
                    },
                    rv64: IsaExecutionProfile {
                        supported: true,
                        supports_dual_hart: false,
                        coverage_mode_support: CoverageModeSupport::ALL,
                        unaligned_access_support: UnalignedAccessSupport {
                            aligned: true,
                            unaligned: false,
                        },
                        extensions: IsaExtensions::rv64(vec![
                            RV64Extensions::I,
                            RV64Extensions::M,
                            RV64Extensions::F,
                            RV64Extensions::D,
                            RV64Extensions::Zicsr,
                        ]),
                    },
                },
                program_mem_range: (0x8000_0000, 0x8000_0fff),
            },
        }
    }

    #[test]
    fn plugin_validate_rejects_duplicate_harts() {
        let insts = vec!["addi x1, x0, 1".to_string()];
        let seqs = insts
            .iter()
            .map(|inst| {
                InstructionSequence::new(
                    RiscvInstruction::decode_asm_with_context(inst, IsaDecodeContext::RV32)
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let hart_inputs = [(0, seqs.as_slice()), (0, seqs.as_slice())];
        let request = ExecuteRequest {
            run_root: Path::new("/tmp"),
            isa_base: ISABase::Rv32,
            hart_inputs: &hart_inputs,
            extension_override: None,
            float_mode_override: None,
            alignment_mode: AlignmentMode::RequireAligned,
            coverage_mode: crate::CoverageMode::Off,
        };

        let error = test_plugin()
            .validate_execute_request(&request)
            .unwrap_err();
        assert!(error.to_string().contains("duplicate hart input"));
    }

    #[test]
    fn plugin_validate_rejects_unsupported_extension_override() {
        let insts = vec!["addi x1, x0, 1".to_string()];
        let seqs = insts
            .iter()
            .map(|inst| {
                InstructionSequence::new(
                    RiscvInstruction::decode_asm_with_context(inst, IsaDecodeContext::RV32)
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let hart_inputs = [(0, seqs.as_slice())];
        let override_ext = IsaExtensions::rv32(vec![
            RV32Extensions::I,
            RV32Extensions::M,
            RV32Extensions::D,
        ]);
        let request = ExecuteRequest {
            run_root: Path::new("/tmp"),
            isa_base: ISABase::Rv32,
            hart_inputs: &hart_inputs,
            extension_override: Some(&override_ext),
            float_mode_override: None,
            alignment_mode: AlignmentMode::RequireAligned,
            coverage_mode: crate::CoverageMode::Off,
        };

        let error = test_plugin()
            .validate_execute_request(&request)
            .unwrap_err();
        assert!(error.to_string().contains("requested RV32 extension"));
    }

    #[test]
    fn plugin_validate_accepts_float_subset_extension_override() {
        let insts = vec!["addi x1, x0, 1".to_string()];
        let seqs = insts
            .iter()
            .map(|inst| {
                InstructionSequence::new(
                    RiscvInstruction::decode_asm_with_context(inst, IsaDecodeContext::RV32)
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let hart_inputs = [(0, seqs.as_slice())];
        let override_ext = IsaExtensions::rv32(vec![
            RV32Extensions::I,
            RV32Extensions::M,
            RV32Extensions::Zicsr,
        ]);
        let request = ExecuteRequest {
            run_root: Path::new("/tmp"),
            isa_base: ISABase::Rv32,
            hart_inputs: &hart_inputs,
            extension_override: Some(&override_ext),
            float_mode_override: None,
            alignment_mode: AlignmentMode::RequireAligned,
            coverage_mode: crate::CoverageMode::Off,
        };

        test_plugin().validate_execute_request(&request).unwrap();
    }

    #[test]
    fn default_streaming_does_not_emit_completed_after_cancel() {
        let insts = vec!["addi x1, x0, 1".to_string()];
        let seqs = insts
            .iter()
            .map(|inst| {
                InstructionSequence::new(
                    RiscvInstruction::decode_asm_with_context(inst, IsaDecodeContext::RV32)
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let hart_inputs = [(0, seqs.as_slice())];
        let request = ExecuteRequest {
            run_root: Path::new("/tmp"),
            isa_base: ISABase::Rv32,
            hart_inputs: &hart_inputs,
            extension_override: None,
            float_mode_override: None,
            alignment_mode: AlignmentMode::RequireAligned,
            coverage_mode: crate::CoverageMode::Off,
        };
        let mut instruction_events = 0usize;
        let mut completed_events = 0usize;

        test_plugin().execute_streaming(request, &mut |event| match event {
            InstructionEvent::Instruction { .. } => {
                instruction_events += 1;
                false
            }
            InstructionEvent::Completed { .. } => {
                completed_events += 1;
                false
            }
            _ => true,
        });

        assert_eq!(instruction_events, 1);
        assert_eq!(completed_events, 0);
    }

    #[test]
    fn plugin_validate_rejects_conflicting_extension_and_float_overrides() {
        let insts = vec!["addi x1, x0, 1".to_string()];
        let seqs = insts
            .iter()
            .map(|inst| {
                InstructionSequence::new(
                    RiscvInstruction::decode_asm_with_context(inst, IsaDecodeContext::RV32)
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let hart_inputs = [(0, seqs.as_slice())];
        let override_ext = IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::M,
            RV64Extensions::F,
            RV64Extensions::D,
            RV64Extensions::Zicsr,
        ]);
        let request = ExecuteRequest {
            run_root: Path::new("/tmp"),
            isa_base: ISABase::Rv64,
            hart_inputs: &hart_inputs,
            extension_override: Some(&override_ext),
            float_mode_override: Some(crate::FloatExtension::F),
            alignment_mode: AlignmentMode::RequireAligned,
            coverage_mode: crate::CoverageMode::Off,
        };

        let error = test_plugin()
            .validate_execute_request(&request)
            .unwrap_err();
        assert!(error.to_string().contains("cannot set both"));
    }

    #[test]
    fn plugin_validate_accepts_supported_float_mode_override() {
        let seqs = vec![InstructionSequence::new(
            RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
                .unwrap(),
        )];
        let hart_inputs = [(0, seqs.as_slice())];
        let request = ExecuteRequest {
            run_root: Path::new("/tmp"),
            isa_base: ISABase::Rv64,
            hart_inputs: &hart_inputs,
            extension_override: None,
            float_mode_override: Some(crate::FloatExtension::F),
            alignment_mode: AlignmentMode::RequireAligned,
            coverage_mode: crate::CoverageMode::Off,
        };

        test_plugin().validate_execute_request(&request).unwrap();
    }
}
