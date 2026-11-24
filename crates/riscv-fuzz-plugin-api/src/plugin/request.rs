use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::InstructionSequence;

use crate::coverage::CoverageMode;
use crate::error::ImplExecutionError;
use crate::execution::ExecutionOutput;
use crate::isa::{ISABase, IsaExtensions};
use crate::metadata::CorePluginMetadata;
use crate::metadata::{AlignmentMode, FloatExtension};

pub type PerHartTypedSequences<'a> = &'a [(u32, &'a [InstructionSequence<RiscvInstruction>])];

pub fn typed_sequences_for_hart<'a>(
    typed_sequences: Option<PerHartTypedSequences<'a>>,
    hart_id: u32,
) -> Option<&'a [InstructionSequence<RiscvInstruction>]> {
    typed_sequences?
        .iter()
        .find_map(|(candidate_hart, sequences)| (*candidate_hart == hart_id).then_some(*sequences))
        .filter(|sequences| !sequences.is_empty())
}

pub fn validate_typed_execution_output(
    output: &mut ExecutionOutput,
    typed_sequences: PerHartTypedSequences<'_>,
) -> Result<(), ImplExecutionError> {
    for &(hart_id, expected_sequences) in typed_sequences {
        let hart = output.per_hart.get(&hart_id).ok_or_else(|| {
            ImplExecutionError::Generic(format!(
                "plugin output is missing hart {hart_id} for typed execution"
            ))
        })?;
        let expected_flat_count = flattened_sequence_instruction_count(expected_sequences);
        for instruction in &hart.instructions {
            let Some(trace_inst_id) = instruction.trace_inst_id else {
                continue;
            };
            if trace_inst_id.hart_id != hart_id {
                return Err(ImplExecutionError::Generic(format!(
                    "plugin output row at pc 0x{:x} for hart {hart_id} points at hart {}",
                    instruction.pc, trace_inst_id.hart_id
                )));
            }
            if trace_inst_id.local_index >= expected_flat_count {
                return Err(ImplExecutionError::Generic(format!(
                    "plugin output row at pc 0x{:x} for hart {hart_id} points at missing trace instruction {} (available={expected_flat_count})",
                    instruction.pc, trace_inst_id.local_index
                )));
            }
        }
    }
    Ok(())
}

fn validate_unique_pcs_per_hart(output: &ExecutionOutput) -> Result<(), ImplExecutionError> {
    for (&hart_id, hart) in &output.per_hart {
        let mut seen_pcs = BTreeSet::new();
        for instruction in &hart.instructions {
            if !seen_pcs.insert(instruction.pc) {
                return Err(ImplExecutionError::Generic(format!(
                    "plugin output for hart {hart_id} contains duplicate asm row pc 0x{:x}",
                    instruction.pc
                )));
            }
        }
    }
    Ok(())
}

pub fn flattened_sequence_instruction_count(
    sequences: &[InstructionSequence<RiscvInstruction>],
) -> usize {
    sequences
        .iter()
        .map(|sequence| sequence.pre_instructions.len() + 1 + sequence.post_instructions.len())
        .sum()
}

fn render_sequences_as_asm_lines(
    sequences: &[InstructionSequence<RiscvInstruction>],
) -> Vec<String> {
    let mut lines = Vec::new();
    for seq in sequences {
        for pseudo in &seq.pre_instructions {
            lines.push(pseudo.to_string());
        }
        lines.push(seq.instruction.to_string());
        for pseudo in &seq.post_instructions {
            lines.push(pseudo.to_string());
        }
    }
    lines
}

#[derive(Debug, Clone, Copy)]
pub struct ExecuteRequest<'a> {
    pub run_root: &'a Path,
    pub isa_base: ISABase,
    pub hart_inputs: PerHartTypedSequences<'a>,
    pub extension_override: Option<&'a IsaExtensions>,
    pub float_mode_override: Option<FloatExtension>,
    pub alignment_mode: AlignmentMode,
    pub coverage_mode: CoverageMode,
}

pub fn build_per_hart_asm(
    hart_inputs: PerHartTypedSequences<'_>,
    isa_base: ISABase,
    build: impl FnOnce(&[(u32, &[String])], ISABase) -> Result<String, crate::error::ConfigError>,
) -> Result<String, crate::error::ConfigError> {
    let per_hart_owned: Vec<(u32, Vec<String>)> = hart_inputs
        .iter()
        .map(|(hart_id, sequences)| (*hart_id, render_sequences_as_asm_lines(sequences)))
        .collect();
    let per_hart_borrowed: Vec<(u32, &[String])> = per_hart_owned
        .iter()
        .map(|(hart_id, lines)| (*hart_id, lines.as_slice()))
        .collect();
    build(per_hart_borrowed.as_slice(), isa_base)
}

pub fn validate_execute_request_with_config<C>(
    request: &ExecuteRequest<'_>,
    metadata: &CorePluginMetadata,
    build_config: impl FnOnce(&ExecuteRequest<'_>, &Path, ISABase) -> Result<C, ImplExecutionError>,
) -> Result<(), ImplExecutionError> {
    request.validate_basic_plugin_request(metadata)?;
    let _ = build_config(request, request.run_root, request.isa_base)?;
    Ok(())
}

pub fn execute_with_config<C>(
    request: ExecuteRequest<'_>,
    build_config: impl FnOnce(&ExecuteRequest<'_>, &Path, ISABase) -> Result<C, ImplExecutionError>,
    execute: impl FnOnce(C, &Path, &[(u32, &[String])]) -> Result<ExecutionOutput, ImplExecutionError>,
) -> Result<ExecutionOutput, ImplExecutionError> {
    let run_root = request.run_root;
    let isa_base = request.isa_base;
    let config = build_config(&request, run_root, isa_base)?;
    let output = request.with_per_hart_asm_lines(|per_hart| execute(config, run_root, per_hart))?;
    request.finalize_execution_output(output)
}

impl<'a> ExecuteRequest<'a> {
    /// Per-hart asm-rendered input instructions, derived directly from
    /// `hart_inputs`. No marker parsing.
    pub fn per_hart_asm_lines(&self) -> Vec<(u32, Vec<String>)> {
        self.hart_inputs
            .iter()
            .map(|(hart_id, sequences)| (*hart_id, render_sequences_as_asm_lines(sequences)))
            .collect()
    }

    pub fn with_per_hart_asm_lines<R>(&self, f: impl FnOnce(&[(u32, &[String])]) -> R) -> R {
        let per_hart_owned = self.per_hart_asm_lines();
        let per_hart_borrowed: Vec<(u32, &[String])> = per_hart_owned
            .iter()
            .map(|(hart_id, lines)| (*hart_id, lines.as_slice()))
            .collect();
        f(per_hart_borrowed.as_slice())
    }

    pub fn with_per_hart_asm_and_typed_sequences<R>(
        &self,
        f: impl FnOnce(&[(u32, &[String])], Option<PerHartTypedSequences<'_>>) -> R,
    ) -> R {
        let per_hart_owned = self.per_hart_asm_lines();
        let per_hart_borrowed: Vec<(u32, &[String])> = per_hart_owned
            .iter()
            .map(|(hart_id, lines)| (*hart_id, lines.as_slice()))
            .collect();
        f(per_hart_borrowed.as_slice(), Some(self.hart_inputs))
    }

    pub fn effective_hart_ids(&self) -> Vec<u32> {
        self.hart_inputs
            .iter()
            .map(|(hart_id, _)| *hart_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn effective_hart_count(&self) -> usize {
        self.effective_hart_ids().len()
    }

    pub fn hart_instruction_counts(&self) -> Vec<(u32, usize)> {
        let mut per_hart: BTreeMap<u32, usize> = BTreeMap::new();
        for &(hart_id, sequences) in self.hart_inputs {
            *per_hart.entry(hart_id).or_default() +=
                flattened_sequence_instruction_count(sequences);
        }
        per_hart.into_iter().collect()
    }

    pub fn validate_basic_plugin_request(
        &self,
        metadata: &CorePluginMetadata,
    ) -> Result<Vec<u32>, ImplExecutionError> {
        if self.hart_inputs.is_empty() {
            return Err(ImplExecutionError::Generic(
                "execute request must contain at least one hart input".to_string(),
            ));
        }

        let mut seen_harts = BTreeSet::new();
        for &(hart_id, _) in self.hart_inputs {
            if !seen_harts.insert(hart_id) {
                return Err(ImplExecutionError::Generic(format!(
                    "duplicate hart input for hart {}",
                    hart_id
                )));
            }
        }

        let effective_hart_ids = self.effective_hart_ids();
        if effective_hart_ids.is_empty() {
            return Err(ImplExecutionError::Generic(
                "execute request must resolve to at least one hart".to_string(),
            ));
        }
        if effective_hart_ids.len() > 2 {
            return Err(ImplExecutionError::Generic(format!(
                "execute request resolves to unsupported hart count {}",
                effective_hart_ids.len()
            )));
        }

        let profile = metadata.isa_profiles.for_isa(self.isa_base);
        if effective_hart_ids.len() > 1 && !profile.supports_dual_hart {
            return Err(ImplExecutionError::Generic(format!(
                "implementation {} does not support hart count {}",
                metadata.core_id,
                effective_hart_ids.len()
            )));
        }

        profile
            .resolve_extensions(self.extension_override, self.float_mode_override)
            .map_err(ImplExecutionError::Generic)?;

        Ok(effective_hart_ids)
    }

    pub fn finalize_execution_output(
        &self,
        mut output: ExecutionOutput,
    ) -> Result<ExecutionOutput, ImplExecutionError> {
        let expected_hart_ids = self
            .hart_instruction_counts()
            .iter()
            .map(|(hart_id, _)| *hart_id)
            .collect::<BTreeSet<_>>();
        let actual_hart_ids = output.per_hart.keys().copied().collect::<BTreeSet<_>>();

        if actual_hart_ids != expected_hart_ids {
            return Err(ImplExecutionError::Generic(format!(
                "plugin returned mismatched per_hart outputs: expected {:?}, got {:?}",
                expected_hart_ids, actual_hart_ids
            )));
        }

        for hart_id in expected_hart_ids {
            output.perf.per_hart.entry(hart_id).or_default();
        }

        validate_unique_pcs_per_hart(&output)?;
        validate_typed_execution_output(&mut output, self.hart_inputs)?;

        Ok(output)
    }

    pub fn resolved_extensions(
        &self,
        metadata: &CorePluginMetadata,
    ) -> Result<IsaExtensions, ImplExecutionError> {
        metadata
            .isa_profiles
            .for_isa(self.isa_base)
            .resolve_extensions(self.extension_override, self.float_mode_override)
            .map_err(ImplExecutionError::Generic)
    }

    pub fn apply_config_common_fields<C>(
        &self,
        metadata: &CorePluginMetadata,
        config: &mut C,
        mut set_extensions: impl FnMut(&mut C, IsaExtensions),
        mut set_coverage_mode: impl FnMut(&mut C, CoverageMode),
    ) -> Result<(), ImplExecutionError> {
        let extensions = self.resolved_extensions(metadata)?;
        set_extensions(config, extensions);
        set_coverage_mode(config, self.coverage_mode);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AsmInstructionExecution, ExecutionOutput, HartExecutionOutput, InstructionTiming,
        TraceInstId,
    };
    use crate::{CoverageMode, ISABase};
    use riscv_instruction::{AsmDecodeWithContext, IsaDecodeContext};

    fn seq(asm: &str) -> InstructionSequence<RiscvInstruction> {
        InstructionSequence::decode_asm_with_context(asm, IsaDecodeContext::RV32).unwrap()
    }

    fn asm(pc: u64) -> AsmInstructionExecution {
        AsmInstructionExecution {
            pc,
            instruction: RiscvInstruction::decode_asm_with_context(
                "addi x1, x0, 1",
                IsaDecodeContext::RV32,
            )
            .unwrap(),
            source_annotation: None,
            trace_inst_id: None,
            exceptions: Vec::new(),
            register_writes: Vec::new(),
            memory_writes: Vec::new(),
            timing: InstructionTiming::default(),
        }
    }

    fn asm_for_source(pc: u64, local_index: usize) -> AsmInstructionExecution {
        let mut instruction = asm(pc);
        instruction.trace_inst_id = Some(TraceInstId::new(0, local_index));
        instruction
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
    fn per_hart_asm_lines_preserves_explicit_hart_inputs() {
        let hart0 = vec![seq("addi x1, x0, 1")];
        let hart1 = vec![seq("addi x2, x0, 2")];
        let hart_inputs = [(0, hart0.as_slice()), (1, hart1.as_slice())];

        assert_eq!(
            request(&hart_inputs).per_hart_asm_lines(),
            vec![
                (0, vec!["addi x1, x0, 1".to_string()]),
                (1, vec!["addi x2, x0, 2".to_string()]),
            ]
        );
    }

    #[test]
    fn effective_hart_count_matches_hart_inputs() {
        let hart0 = vec![seq("addi x1, x0, 1")];
        let hart1 = vec![seq("addi x2, x0, 2")];
        let hart_inputs = [(0, hart0.as_slice()), (1, hart1.as_slice())];

        let request = request(&hart_inputs);
        assert_eq!(request.effective_hart_ids(), vec![0, 1]);
        assert_eq!(request.effective_hart_count(), 2);
        assert_eq!(request.hart_instruction_counts(), vec![(0, 1), (1, 1)]);
    }

    #[test]
    fn hart_instruction_counts_use_typed_sequence_count() {
        let typed = vec![seq("li x5, 42\naddi x6, x5, 1"), seq("addi x7, x0, 2")];
        let hart_inputs = [(0, typed.as_slice())];

        assert_eq!(
            request(&hart_inputs).hart_instruction_counts(),
            vec![(0, 3)]
        );
    }

    #[test]
    fn per_hart_asm_and_typed_sequences_preserves_typed_inputs() {
        let hart0 = vec![seq("addi x1, x0, 1")];
        let hart1 = vec![seq("addi x2, x0, 2")];
        let hart_inputs = [(0, hart0.as_slice()), (1, hart1.as_slice())];

        request(&hart_inputs).with_per_hart_asm_and_typed_sequences(|per_hart, typed| {
            assert_eq!(
                per_hart,
                &[
                    (0, &["addi x1, x0, 1".to_string()][..]),
                    (1, &["addi x2, x0, 2".to_string()][..]),
                ]
            );
            assert_eq!(typed.expect("typed sequences"), &hart_inputs);
        });
    }

    #[test]
    fn basic_plugin_request_validation_rejects_duplicate_harts() {
        let insts = vec![seq("addi x1, x0, 1")];
        let hart_inputs = [(0, insts.as_slice()), (0, insts.as_slice())];
        let metadata = CorePluginMetadata {
            core_id: "test".to_string(),
            isa_profiles: crate::IsaExecutionProfileMap {
                rv32: crate::IsaExecutionProfile {
                    supported: true,
                    supports_dual_hart: true,
                    coverage_mode_support: crate::CoverageModeSupport::ALL,
                    unaligned_access_support: crate::UnalignedAccessSupport {
                        aligned: true,
                        unaligned: false,
                    },
                    extensions: crate::IsaExtensions::rv32(vec![]),
                },
                rv64: crate::IsaExecutionProfile::unsupported(ISABase::Rv64),
            },
            program_mem_range: (0, 0),
        };

        let error = request(&hart_inputs)
            .validate_basic_plugin_request(&metadata)
            .unwrap_err();
        assert!(error.to_string().contains("duplicate hart input"));
    }

    #[test]
    fn finalize_execution_output_rejects_mismatched_harts() {
        let hart0 = vec![seq("addi x1, x0, 1")];
        let hart1 = vec![seq("addi x2, x0, 2")];
        let hart_inputs = [(0, hart0.as_slice()), (1, hart1.as_slice())];
        let request = request(&hart_inputs);
        let output = ExecutionOutput::empty_for_harts("test", ISABase::Rv32, [0u32]);

        let error = request.finalize_execution_output(output).unwrap_err();
        assert!(error.to_string().contains("mismatched per_hart outputs"));
    }

    #[test]
    fn finalize_execution_output_rejects_out_of_range_trace_inst_id_for_typed_queue() {
        let sequence = InstructionSequence {
            pre_instructions: vec![
                RiscvInstruction::decode_asm_with_context("addi x5, x0, 5", IsaDecodeContext::RV32)
                    .unwrap(),
            ],
            instruction: RiscvInstruction::decode_asm_with_context(
                "addi x6, x5, 1",
                IsaDecodeContext::RV32,
            )
            .unwrap(),
            post_instructions: vec![
                RiscvInstruction::decode_asm_with_context("addi x7, x6, 2", IsaDecodeContext::RV32)
                    .unwrap(),
            ],
        };
        let hart_inputs = [(0, std::slice::from_ref(&sequence))];
        let request = request(&hart_inputs);

        let mut output = ExecutionOutput::empty_for_harts("test", ISABase::Rv32, [0u32]);
        output.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![asm_for_source(0x100c, 3)],
            },
        );

        let err = request
            .finalize_execution_output(output)
            .expect_err("out-of-range trace instruction id must be rejected");
        assert!(
            err.to_string()
                .contains("points at missing trace instruction 3")
        );
    }

    #[test]
    fn finalize_execution_output_accepts_multiple_pcs_for_one_trace_inst_id() {
        let sequence = InstructionSequence {
            pre_instructions: vec![
                RiscvInstruction::decode_asm_with_context("addi x5, x0, 5", IsaDecodeContext::RV32)
                    .unwrap(),
            ],
            instruction: RiscvInstruction::decode_asm_with_context(
                "addi x6, x5, 1",
                IsaDecodeContext::RV32,
            )
            .unwrap(),
            post_instructions: Vec::new(),
        };
        let hart_inputs = [(0, std::slice::from_ref(&sequence))];
        let request = request(&hart_inputs);

        let mut output = ExecutionOutput::empty_for_harts("test", ISABase::Rv32, [0u32]);
        output.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![
                    asm_for_source(0x1000, 0),
                    asm_for_source(0x1004, 0),
                    asm_for_source(0x1008, 1),
                ],
            },
        );

        request
            .finalize_execution_output(output)
            .expect("multiple PCs on one trace instruction id should be accepted");
    }

    #[test]
    fn finalize_execution_output_rejects_trace_inst_id_for_wrong_hart() {
        let expected = InstructionSequence::new(
            RiscvInstruction::decode_asm_with_context("addi x5, x0, 5", IsaDecodeContext::RV32)
                .unwrap(),
        );
        let hart_inputs = [(0, std::slice::from_ref(&expected))];
        let request = request(&hart_inputs);

        let mut output = ExecutionOutput::empty_for_harts("test", ISABase::Rv32, [0u32]);
        output.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: {
                    let mut row = asm(0x1000);
                    row.trace_inst_id = Some(TraceInstId::new(1, 0));
                    vec![row]
                },
            },
        );

        let err = request
            .finalize_execution_output(output)
            .expect_err("cross-hart trace instruction id must be rejected");
        assert!(err.to_string().contains("points at hart 1"));
    }

    #[test]
    fn finalize_execution_output_rejects_duplicate_pcs_on_one_hart() {
        let expected = InstructionSequence::new(
            RiscvInstruction::decode_asm_with_context("addi x5, x0, 5", IsaDecodeContext::RV32)
                .unwrap(),
        );
        let hart_inputs = [(0, std::slice::from_ref(&expected))];
        let request = request(&hart_inputs);

        let mut output = ExecutionOutput::empty_for_harts("test", ISABase::Rv32, [0u32]);
        output.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![asm_for_source(0x1000, 0), asm_for_source(0x1000, 0)],
            },
        );

        let err = request
            .finalize_execution_output(output)
            .expect_err("duplicate asm-row pcs must be rejected");
        assert!(err.to_string().contains("duplicate asm row pc 0x1000"));
    }
}
