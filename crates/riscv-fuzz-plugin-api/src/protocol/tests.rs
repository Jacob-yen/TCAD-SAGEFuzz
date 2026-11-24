use std::{collections::BTreeMap, path::PathBuf};

use riscv_instruction::{
    AsmDecodeWithContext, IsaDecodeContext, RV32Extensions, RV64Extensions, RiscvInstruction,
};
use riscv_instruction_types::InstructionSequence;

use super::common::{encode_message, pb};
use super::*;
use crate::{
    AlignmentMode, AsmInstructionExecution, CorePluginMetadata, CoverageMode, ExceptionCause,
    ExecuteRequest, ExecutionOutput, HartExecutionOutput, HartPerfMetrics, ISABase,
    InstructionEvent, InstructionTiming, IsaExecutionProfile, IsaExecutionProfileMap,
    IsaExtensions, MemValue, PerfMetrics, RegisterValue, TraceInstId, UnalignedAccessSupport,
    metadata::CoverageModeSupport,
};

#[test]
fn metadata_roundtrip_preserves_profiles_and_extensions() {
    let record = PluginMetadataRecord {
        core_metadata: CorePluginMetadata {
            core_id: "spike".to_string(),
            isa_profiles: IsaExecutionProfileMap {
                rv32: IsaExecutionProfile {
                    supported: true,
                    supports_dual_hart: false,
                    coverage_mode_support: CoverageModeSupport {
                        off: true,
                        light: true,
                        full: false,
                    },
                    unaligned_access_support: UnalignedAccessSupport {
                        aligned: true,
                        unaligned: false,
                    },
                    extensions: IsaExtensions::rv32(vec![
                        RV32Extensions::I,
                        RV32Extensions::M,
                        RV32Extensions::Zicsr,
                    ]),
                },
                rv64: IsaExecutionProfile {
                    supported: true,
                    supports_dual_hart: true,
                    coverage_mode_support: CoverageModeSupport::ALL,
                    unaligned_access_support: UnalignedAccessSupport {
                        aligned: true,
                        unaligned: true,
                    },
                    extensions: IsaExtensions::rv64(vec![
                        RV64Extensions::I,
                        RV64Extensions::F,
                        RV64Extensions::D,
                    ]),
                },
            },
            program_mem_range: (0x8000_0000, 0x8000_0fff),
        },
        linker_script: "SECTIONS { .text : { *(.text*) } }".to_string(),
        doctor_report: crate::PluginDoctorReport::empty("spike"),
    };

    let bytes = encode_plugin_metadata_record(&record).unwrap();
    let decoded = decode_plugin_metadata_record(&bytes).unwrap();

    assert_eq!(decoded, record);
}

#[test]
fn metadata_roundtrip_preserves_unsupported_isa_profiles() {
    let record = PluginMetadataRecord {
        core_metadata: CorePluginMetadata {
            core_id: "kronos".to_string(),
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
                        RV32Extensions::Zicsr,
                        RV32Extensions::Zifencei,
                    ]),
                },
                rv64: IsaExecutionProfile::unsupported(ISABase::Rv64),
            },
            program_mem_range: (0x800, 0x1fff),
        },
        linker_script: "SECTIONS { .text : { *(.text*) } }".to_string(),
        doctor_report: crate::PluginDoctorReport::empty("kronos"),
    };

    let bytes = encode_plugin_metadata_record(&record).unwrap();
    let decoded = decode_plugin_metadata_record(&bytes).unwrap();

    assert_eq!(decoded, record);
}

#[test]
fn execute_request_roundtrip_preserves_optional_fields() {
    let payload = execute::ExecuteRequestPayload {
        run_root: PathBuf::from("/tmp/run-root"),
        isa_base: ISABase::Rv64,
        hart_inputs: vec![
            (
                0,
                vec![InstructionSequence::new(
                    RiscvInstruction::decode_asm_with_context(
                        "addi x1, x0, 1",
                        IsaDecodeContext::RV64,
                    )
                    .unwrap(),
                )],
            ),
            (
                1,
                vec![InstructionSequence::new(
                    RiscvInstruction::decode_asm_with_context("ecall", IsaDecodeContext::RV64)
                        .unwrap(),
                )],
            ),
        ],
        extension_override: Some(IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::M,
            RV64Extensions::Zicsr,
        ])),
        float_mode_override: Some(crate::FloatExtension::FD),
        alignment_mode: AlignmentMode::AllowUnaligned,
        coverage_mode: CoverageMode::Light,
    };
    let hart_inputs = payload
        .hart_inputs
        .iter()
        .map(|(hart_id, sequences)| (*hart_id, sequences.as_slice()))
        .collect::<Vec<_>>();

    let bytes = encode_execute_request(ExecuteRequest {
        run_root: &payload.run_root,
        isa_base: payload.isa_base,
        hart_inputs: &hart_inputs,
        extension_override: payload.extension_override.as_ref(),
        float_mode_override: payload.float_mode_override,
        alignment_mode: payload.alignment_mode,
        coverage_mode: payload.coverage_mode,
    })
    .unwrap();

    let decoded = decode_execute_request(&bytes).unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn execute_request_roundtrip_preserves_typed_input_instructions() {
    let inst = RiscvInstruction::decode_asm_with_context("addi x1, x0, 7", IsaDecodeContext::RV64)
        .unwrap();
    let seqs: Vec<InstructionSequence<RiscvInstruction>> =
        vec![InstructionSequence::new(inst.clone())];

    let hart_inputs = [(0, seqs.as_slice())];

    let bytes = encode_execute_request(ExecuteRequest {
        run_root: PathBuf::from("/tmp/run-root").as_path(),
        isa_base: ISABase::Rv64,
        hart_inputs: &hart_inputs,
        extension_override: None,
        float_mode_override: None,
        alignment_mode: AlignmentMode::RequireAligned,
        coverage_mode: CoverageMode::Off,
    })
    .unwrap();

    let decoded = decode_execute_request(&bytes).unwrap();
    assert_eq!(decoded.hart_inputs.len(), 1);
    let (hart_id, decoded_seqs) = &decoded.hart_inputs[0];
    assert_eq!(*hart_id, 0);
    assert_eq!(decoded_seqs, &seqs);
}

#[test]
fn execution_output_roundtrip_preserves_perf_and_memory() {
    let mut per_hart = BTreeMap::new();
    per_hart.insert(
        0,
        HartPerfMetrics {
            mcycle_start: Some(10),
            mcycle_end: Some(20),
            minstret_start: Some(2),
            minstret_end: Some(6),
        },
    );
    let source_inst =
        RiscvInstruction::decode_asm_with_context("addi x1, x0, 42", IsaDecodeContext::RV64)
            .unwrap();
    let output = ExecutionOutput {
        per_hart: BTreeMap::from([(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![
                    AsmInstructionExecution {
                        pc: 0x8000_0000,
                        instruction: source_inst.clone(),
                        source_annotation: Some("addi x1, x0, 42".to_string()),
                        trace_inst_id: Some(TraceInstId::new(0, 3)),
                        exceptions: vec![ExceptionCause::from_code(11, "ecall from M-mode")],
                        register_writes: vec![RegisterValue {
                            name: "x1".to_string(),
                            value: 42,
                        }],
                        memory_writes: vec![MemValue {
                            addr: 0x1000,
                            value: 0xaa,
                        }],
                        timing: InstructionTiming::new(Some(11), Some(17), Some(7)),
                    },
                    AsmInstructionExecution {
                        pc: 0x8000_0004,
                        instruction: RiscvInstruction::decode_asm_with_context(
                            "ebreak",
                            IsaDecodeContext::RV64,
                        )
                        .unwrap(),
                        source_annotation: Some("ebreak".to_string()),
                        trace_inst_id: None,
                        exceptions: Vec::new(),
                        register_writes: Vec::new(),
                        memory_writes: Vec::new(),
                        timing: InstructionTiming::new(Some(18), Some(18), Some(1)),
                    },
                ],
            },
        )]),
        core_id: "spike".to_string(),
        isa_base: ISABase::Rv64,
        perf: PerfMetrics {
            global_cycles: Some(100),
            per_hart,
            notes: vec!["stable".to_string()],
        },
    };

    let bytes = encode_execution_output(&output).unwrap();
    let decoded = decode_execution_output(&bytes).unwrap();

    assert_eq!(decoded, output);
}

#[test]
fn execution_output_decode_rejects_missing_perf() {
    let bytes = encode_message(&pb::ExecutionOutput {
        per_hart: Vec::new(),
        core_id: "spike".to_string(),
        isa_base: pb::IsaBase::Rv64 as i32,
        perf: None,
    });

    let error = decode_execution_output(&bytes).unwrap_err();
    assert!(error.contains("missing required protobuf field `execution output.perf`"));
}

#[test]
fn streaming_instruction_event_roundtrip_preserves_trace_inst_id() {
    let instruction =
        RiscvInstruction::decode_asm_with_context("addi x1, x0, 42", IsaDecodeContext::RV64)
            .unwrap();
    let asm = AsmInstructionExecution {
        pc: 0x8000_0000,
        instruction: instruction.clone(),
        source_annotation: Some("addi x1, x0, 42".to_string()),
        trace_inst_id: Some(TraceInstId::new(1, 7)),
        exceptions: vec![ExceptionCause::from_code(11, "ecall from M-mode")],
        register_writes: vec![RegisterValue {
            name: "x1".to_string(),
            value: 42,
        }],
        memory_writes: vec![MemValue {
            addr: 0x1000,
            value: 0xaa,
        }],
        timing: InstructionTiming::new(Some(11), Some(17), Some(7)),
    };

    let bytes = encode_instruction_event(&InstructionEvent::Instruction {
        hart_id: 1,
        instruction: asm.clone(),
    })
    .unwrap();

    match decode_instruction_event(&bytes, ISABase::Rv64).unwrap() {
        InstructionEvent::Instruction {
            hart_id,
            instruction: decoded_asm,
        } => {
            assert_eq!(hart_id, 1);
            assert_eq!(decoded_asm, asm);
        }
        other => panic!("expected instruction event, got {other:?}"),
    }
}
