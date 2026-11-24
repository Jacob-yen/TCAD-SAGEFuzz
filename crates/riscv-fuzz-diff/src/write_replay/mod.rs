//! 写差异 replay 与历史最小化逻辑。
//! Write-difference replay and history minimization.
//!
//! 当 runner 观察到写差异时，这个模块会围绕目标指令重建最小 testcase，
//! When the runner sees a write difference, this module rebuilds a minimal testcase around that
//! instruction
//! 以回答两个问题：
//! to answer two questions:
//! - 单独执行目标指令时，差异是否仍然存在？
//!   Does the target instruction alone still diverge?
//! - 如果不存在，复现该差异所需的最短历史前缀是什么？
//!   If not, what shortest surviving history prefix is required to reproduce the divergence?

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use riscv_fuzz_plugin_api::{CoverageMode, ISABase, parse_register_name};
use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;
use riscv_instruction::RegisterBank;
use riscv_instruction::{
    AsmDecodeWithContext, IsaDecodeContext, RV32_I_LBU, RV32IInstructions, RV32Instruction,
    RV64_I_LBU, RV64IInstructions, RV64Instruction, RiscvInstruction,
};
use riscv_instruction_types::{
    BaseAddressRegister, Immediate, IntegerRegister, IsaProfile, ValidatedValue,
};

use super::{
    detection::{WriteDifference, detect_unique_exceptions, detect_unique_writes},
    iteration::{
        iteration_summary_path, persist_iteration_impl_artifacts, persist_iteration_testcase,
        run_iteration, run_iteration_with_per_impl_testcases,
    },
    localization::record_first_write_localization,
    normalize::{
        determine_test_length, logical_instruction_at_location,
        logical_instruction_effects_unfiltered, test_instruction_location,
    },
    reporting::{
        write_execution_state_report, write_history_candidate_report,
        write_history_minimization_error_report, write_history_minimization_report,
        write_replay_write_failure_report, write_replay_write_report,
    },
    types::DiffAnalysisError,
    view::DiffExecutionView,
};

mod history;
mod probe;
mod replay;

use history::{HistoryMinimizationConfig, minimize_history_write_difference};
use probe::run_write_probe_verification;
use replay::build_replay_testcases;

const HISTORY_MIN_START_CANDIDATE_LOSS_REASON: &str = "历史最小化起始候选未保留目标指令写入差异";

fn x_reg(reg: u8) -> IntegerRegister {
    IntegerRegister::new(reg).expect("valid integer register")
}

fn base_reg(reg: u8) -> BaseAddressRegister {
    BaseAddressRegister::new(reg).expect("valid base register")
}

fn isa_profile(isa_base: ISABase) -> IsaProfile {
    IsaProfile::new(
        match isa_base {
            ISABase::Rv32 => riscv_instruction_types::XLen::X32,
            ISABase::Rv64 => riscv_instruction_types::XLen::X64,
        },
        None,
    )
}

fn reg_name_to_x_reg(name: &str) -> Result<IntegerRegister, DiffAnalysisError> {
    let Some(parsed) = parse_register_name(name) else {
        return Err(DiffAnalysisError::TestInstructionCountMismatch {
            details: format!("unsupported probe register name '{name}'"),
        });
    };
    if parsed.bank != RegisterBank::Integer {
        return Err(DiffAnalysisError::TestInstructionCountMismatch {
            details: format!("unsupported non-integer probe register name '{name}'"),
        });
    }
    IntegerRegister::new(parsed.index).map_err(|err| {
        DiffAnalysisError::TestInstructionCountMismatch {
            details: format!("invalid probe register '{name}': {err}"),
        }
    })
}

fn copy_reg_instruction(
    isa_base: ISABase,
    rd: u8,
    rs1: &str,
) -> Result<RiscvInstruction, DiffAnalysisError> {
    let _ = reg_name_to_x_reg(rs1)?;
    let asm = format!("xori x{rd}, {rs1}, 0");
    let ctx = match isa_base {
        ISABase::Rv32 => IsaDecodeContext::RV32,
        ISABase::Rv64 => IsaDecodeContext::RV64,
    };
    RiscvInstruction::decode_asm_with_context(&asm, ctx).map_err(|err| {
        DiffAnalysisError::TestInstructionCountMismatch {
            details: format!("failed to build register probe `{asm}`: {err}"),
        }
    })
}

fn load_byte_instruction(isa_base: ISABase, rd: u8, rs1: u8) -> RiscvInstruction {
    let imm = Immediate::<12, true>::new(0).expect("valid zero offset");
    match isa_base {
        ISABase::Rv32 => {
            RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::LBU(RV32_I_LBU {
                xd: x_reg(rd),
                xs1: base_reg(rs1),
                imm,
            })))
        }
        ISABase::Rv64 => {
            RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::LBU(RV64_I_LBU {
                xd: x_reg(rd),
                xs1: base_reg(rs1),
                imm,
            })))
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum WriteDifferenceCause {
    SingleInstruction,
    DependsOnHistory,
    LikelyFalsePositive,
}

impl std::fmt::Display for WriteDifferenceCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            WriteDifferenceCause::SingleInstruction => "单条测试指令引发差异",
            WriteDifferenceCause::DependsOnHistory => "历史测试指令与当前指令共同引发差异",
            WriteDifferenceCause::LikelyFalsePositive => {
                "疑似误报（replay/history-min 未能保留差异）"
            }
        })
    }
}

#[derive(Debug, Clone)]
pub(super) enum WriteProbeOutcome {
    Confirmed,
    Refuted,
    MismatchDisappeared,
    Inconclusive { reason: String },
}

impl WriteProbeOutcome {
    pub(super) fn summary(&self) -> String {
        match self {
            WriteProbeOutcome::Confirmed => "确认（探针观测到差异）".to_string(),
            WriteProbeOutcome::Refuted => "排除（探针观测一致，疑似误报/仅写事件差异）".to_string(),
            WriteProbeOutcome::MismatchDisappeared => {
                "差异消失（插入探针后不再复现，疑似日志对齐/解析问题）".to_string()
            }
            WriteProbeOutcome::Inconclusive { reason } => format!("不确定（{reason}）"),
        }
    }
}

fn effective_write_difference_cause(
    cause: WriteDifferenceCause,
    probe_outcome: Option<&WriteProbeOutcome>,
) -> WriteDifferenceCause {
    if matches!(cause, WriteDifferenceCause::SingleInstruction)
        && matches!(
            probe_outcome,
            Some(WriteProbeOutcome::Refuted | WriteProbeOutcome::MismatchDisappeared)
        )
    {
        WriteDifferenceCause::LikelyFalsePositive
    } else {
        cause
    }
}

fn is_history_start_candidate_failure(err: &DiffAnalysisError) -> bool {
    matches!(
        err,
        DiffAnalysisError::HistoryMinimizationFailure { reason }
            if reason == HISTORY_MIN_START_CANDIDATE_LOSS_REASON
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn classify_write_difference(
    diff: &WriteDifference,
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
    impl_order: &[String],
    iteration_dir: &Path,
    verification_round: usize,
    history_min_threshold: Option<usize>,
    coverage_mode: CoverageMode,
    emit_execution_output_json: bool,
    emit_execution_report_md: bool,
    cleanup_history_candidates: bool,
    verify_write_with_probes: bool,
    write_probe_max_registers: usize,
    write_probe_max_memory_bytes: usize,
) -> Result<(WriteDifferenceCause, Option<WriteProbeOutcome>), DiffAnalysisError> {
    let replay_dir = iteration_dir.join(format!("write_replay_{:03}", verification_round));
    fs::create_dir_all(&replay_dir)?;
    let replay_report_path = replay_dir.join("report_01_write_replay.md");

    let replay = match build_replay_testcases(diff, testcase, outputs, impl_order, &replay_dir) {
        Ok(value) => value,
        Err(err) => {
            write_replay_write_failure_report(
                &replay_report_path,
                testcase,
                diff,
                impl_order,
                diff.test_index,
                &format!("写入重放构造失败：{err}"),
            )?;
            return Ok((WriteDifferenceCause::SingleInstruction, None));
        }
    };
    let replay_testcase = replay.canonical_testcase;
    let target_index = replay.target_index;

    // replay 会重建目标指令执行前一刻的体系结构状态，
    // Replay reconstructs the architectural pre-state immediately before the mismatching
    // instruction,
    // 然后只执行该指令本身；如果同样的写差异仍然存在，说明问题局部属于该指令。
    // then executes only that instruction; if the same write difference survives, the bug is
    // local to that instruction.
    // 否则，它就依赖于更早的历史。
    // Otherwise, it depends on prior history.
    let replay_execution = match run_iteration_with_per_impl_testcases(
        &replay_testcase,
        &replay.per_impl_testcases,
        impl_order,
        &replay_dir,
        coverage_mode,
        emit_execution_output_json,
        emit_execution_report_md,
    ) {
        Ok(execution) => execution,
        Err(err) => {
            write_replay_write_failure_report(
                &replay_report_path,
                testcase,
                diff,
                impl_order,
                target_index,
                &format!("写入重放执行失败：{err}"),
            )?;
            return Ok((WriteDifferenceCause::SingleInstruction, None));
        }
    };
    let replay_outputs = replay_execution.outputs;

    let summary_path = iteration_summary_path(&replay_dir);
    if let Err(err) = write_execution_state_report(
        &summary_path,
        &replay_testcase,
        &replay_outputs,
        impl_order,
        Some(&replay_execution.timings),
        None,
    ) {
        write_replay_write_failure_report(
            &replay_report_path,
            testcase,
            diff,
            impl_order,
            target_index,
            &format!("写入重放汇总报告生成失败：{err}"),
        )?;
        return Ok((WriteDifferenceCause::SingleInstruction, None));
    }

    let replay_differences =
        match detect_unique_writes(impl_order, &replay_testcase, &replay_outputs) {
            Ok(differences) => differences,
            Err(err) => {
                write_replay_write_failure_report(
                    &replay_report_path,
                    testcase,
                    diff,
                    impl_order,
                    target_index,
                    &format!("写入重放差异检测失败：{err}"),
                )?;
                return Ok((WriteDifferenceCause::SingleInstruction, None));
            }
        };
    let mut first_difference: Option<WriteDifference> = None;
    let mut target_difference: Option<WriteDifference> = None;
    for diff_entry in replay_differences {
        if first_difference.is_none() {
            first_difference = Some(diff_entry.clone());
        }
        if diff_entry.test_index == target_index && target_difference.is_none() {
            target_difference = Some(diff_entry);
        }
    }

    if let Some(first) = &first_difference {
        // 只有当 replay 里第一个复现出来的差异仍然是目标差异时，结果才有效。
        // Replay is only valid if the first reproduced difference is still the target one.
        // 更早的位置出现差异，说明重建后的 testcase 已经偏离原本要追踪的信号。
        // An earlier mismatch means the reconstructed testcase drifted from the intended signal.
        if first.test_index != target_index {
            let failure_message = format!(
                "写入重放失败：首个差异出现在测试指令 #{}, 预期为 #{}。",
                first.test_index, target_index
            );
            write_replay_write_failure_report(
                &replay_report_path,
                &replay_testcase,
                first,
                impl_order,
                target_index,
                &failure_message,
            )?;
            return Ok((WriteDifferenceCause::SingleInstruction, None));
        }
    }

    let difference_persists = target_difference.is_some();
    let mut cause = if difference_persists {
        WriteDifferenceCause::SingleInstruction
    } else {
        WriteDifferenceCause::DependsOnHistory
    };

    let report_difference = if let Some(diff_entry) = target_difference {
        diff_entry
    } else {
        build_replay_difference_snapshot(
            &replay_testcase,
            &replay_outputs,
            impl_order,
            target_index,
        )?
    };

    if matches!(cause, WriteDifferenceCause::DependsOnHistory)
        && let Err(err) = minimize_history_write_difference(
            diff,
            testcase,
            impl_order,
            iteration_dir,
            HistoryMinimizationConfig {
                verification_round,
                history_min_threshold,
                coverage_mode,
                emit_execution_output_json,
                emit_execution_report_md,
                cleanup_history_candidates,
            },
        )
    {
        let minimize_root =
            iteration_dir.join(format!("write_history_min_{:03}", verification_round));
        fs::create_dir_all(&minimize_root)?;
        let error_report = minimize_root.join("report_00_history_min_error.md");
        write_history_minimization_error_report(&error_report, diff, &err.to_string())?;
        if is_history_start_candidate_failure(&err) {
            cause = WriteDifferenceCause::LikelyFalsePositive;
        }
    }

    let mut probe_outcome: Option<WriteProbeOutcome> = None;
    if verify_write_with_probes && difference_persists {
        match run_write_probe_verification(
            &report_difference,
            &replay_testcase,
            &replay.per_impl_testcases,
            impl_order,
            &replay_dir,
            target_index,
            coverage_mode,
            emit_execution_output_json,
            emit_execution_report_md,
            write_probe_max_registers,
            write_probe_max_memory_bytes,
        ) {
            Ok(outcome) => probe_outcome = Some(outcome),
            Err(err) => {
                let note_path = replay_dir.join("probe_verification_error.txt");
                let _ = fs::write(&note_path, format!("probe verification failed: {err}\n"));
                probe_outcome = Some(WriteProbeOutcome::Inconclusive {
                    reason: format!("probe verification failed: {err}"),
                });
            }
        }
    }

    cause = effective_write_difference_cause(cause, probe_outcome.as_ref());

    write_replay_write_report(
        &replay_report_path,
        &replay_testcase,
        &report_difference,
        impl_order,
        &cause.to_string(),
    )?;

    record_first_write_localization();

    Ok((cause, probe_outcome))
}

fn build_replay_difference_snapshot(
    replay_testcase: &GeneratedTestCase,
    replay_outputs: &HashMap<String, DiffExecutionView>,
    impl_order: &[String],
    target_index: usize,
) -> Result<WriteDifference, DiffAnalysisError> {
    let location = test_instruction_location(impl_order, replay_testcase, target_index)?;
    let mut global_index = Some(location.global_index);

    let mut register_map = BTreeMap::new();
    let mut memory_map = BTreeMap::new();
    let mut register_contexts = BTreeMap::new();
    let mut memory_contexts = BTreeMap::new();

    for impl_ref in impl_order {
        let output =
            replay_outputs
                .get(impl_ref)
                .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                    core_id: impl_ref.to_string(),
                })?;

        let instruction = logical_instruction_at_location(output, location).ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "write replay missing execution context at test index {} (hart {} local {}) for {}",
                    target_index,
                    location.hart_id,
                    location.hart_local_instruction_index,
                    impl_ref
                ),
            }
        })?;
        match global_index {
            Some(existing) if existing != instruction.instruction_index => {
                return Err(DiffAnalysisError::TestInstructionCountMismatch {
                    details: format!(
                        "write replay test index {} maps to different global indices: {} vs {} in {}",
                        target_index, existing, instruction.instruction_index, impl_ref
                    ),
                });
            }
            Some(_) => {}
            None => global_index = Some(instruction.instruction_index),
        }
        let (registers, memory) = logical_instruction_effects_unfiltered(instruction);

        register_map.insert(impl_ref.clone(), registers);
        memory_map.insert(impl_ref.clone(), memory);
        register_contexts.insert(
            impl_ref.clone(),
            instruction.context.register_context.clone(),
        );
        memory_contexts.insert(impl_ref.clone(), instruction.context.memory_context.clone());
    }

    Ok(WriteDifference {
        test_index: target_index,
        global_index: global_index.expect("non-empty implementation order produced rows"),
        register_changes: register_map,
        memory_changes: memory_map,
        register_contexts,
        memory_contexts,
    })
}

#[cfg(all(test, feature = "internal-tests"))]
mod tests {
    use super::*;
    use crate::write_replay::probe::{
        collect_mismatching_integer_registers, collect_mismatching_memory_offsets,
    };
    use crate::write_replay::replay::infer_target_hart_for_test_index;
    use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
    use riscv_fuzz_testcase_engine::testcase::{
        GeneratedTestCase, HartConfig, MulticoreSyncConfig, StructuredTestInstructions,
        TestCaseConfig,
    };
    use riscv_instruction::{RV32_I_ADDI, RV32Extensions, RV32IInstructions, RV32Instruction};
    use riscv_instruction_types::{
        Immediate, InstructionSequence, IntegerRegister, MemoryAccessConfig, PseudoInstruction,
        RandomConfig, RegisterOperandConfig, ValidatedValue,
    };

    fn addi(rd: u8, rs1: u8, imm: i16) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(RV32_I_ADDI {
            xd: IntegerRegister::new(rd).expect("valid rd"),
            xs1: IntegerRegister::new(rs1).expect("valid rs1"),
            imm: Immediate::<12>::new(imm.into()).expect("valid addi immediate"),
        })))
    }

    fn minimal_multicore_testcase(
        impl_order: &[String],
        per_impl_test_harts: &HashMap<String, Vec<Vec<RiscvInstruction>>>,
    ) -> GeneratedTestCase {
        if impl_order.is_empty() {
            return GeneratedTestCase {
                config: TestCaseConfig {
                    isa_base: ISABase::Rv32,
                    extension_insts_count: 0,
                    extension_inst_scaling: None,
                    mem_size: 64,
                    float_extension: Default::default(),
                    unaligned_access_required: Some(false),
                    random_config: RandomConfig {
                        memory_access_config: MemoryAccessConfig {
                            memory_base_address_range: MemoryAccessConfig::default()
                                .memory_base_address_range,
                            memory_offset_immediate_range: (0, 0),
                            memory_base_register_number_range: (10, 15),
                        },
                        register_operand_config: RegisterOperandConfig {
                            integer_register_number_range: (16, 25),
                            floating_point_register_number_range: (16, 25),
                            vector_register_number_range: (16, 25),
                        },
                        ..Default::default()
                    },
                    seed: None,
                    hart_config: HartConfig::Dual {
                        sync: MulticoreSyncConfig::default(),
                    },
                },
                harts: vec![
                    riscv_fuzz_testcase_engine::testcase::HartProgram {
                        hart_id: 0,
                        init: Vec::new(),
                        test: Vec::new(),
                        shared_init_sequence_indices: Default::default(),
                        shared_test_sequence_indices: Default::default(),
                    },
                    riscv_fuzz_testcase_engine::testcase::HartProgram {
                        hart_id: 1,
                        init: Vec::new(),
                        test: Vec::new(),
                        shared_init_sequence_indices: Default::default(),
                        shared_test_sequence_indices: Default::default(),
                    },
                ],
                mem_range: HashMap::new(),
                extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
                multicore_sync_metrics: None,
            };
        }

        let config = TestCaseConfig {
            isa_base: ISABase::Rv32,
            extension_insts_count: 0,
            extension_inst_scaling: None,
            mem_size: 64,
            float_extension: Default::default(),
            unaligned_access_required: Some(false),
            random_config: RandomConfig {
                memory_access_config: MemoryAccessConfig {
                    memory_base_address_range: MemoryAccessConfig::default()
                        .memory_base_address_range,
                    memory_offset_immediate_range: (0, 0),
                    memory_base_register_number_range: (10, 15),
                },
                register_operand_config: RegisterOperandConfig {
                    integer_register_number_range: (16, 25),
                    floating_point_register_number_range: (16, 25),
                    vector_register_number_range: (16, 25),
                },
                ..Default::default()
            },
            seed: None,
            hart_config: HartConfig::Dual {
                sync: MulticoreSyncConfig::default(),
            },
        };

        let mut mem_range = HashMap::new();
        let canonical_test_program: Vec<Vec<InstructionSequence<RiscvInstruction>>> =
            per_impl_test_harts
                .get(
                    impl_order
                        .first()
                        .expect("impl_order must be non-empty for minimal_multicore_testcase"),
                )
                .expect("per_impl_test_harts missing first implementation")
                .iter()
                .map(|hart| hart.iter().cloned().map(InstructionSequence::new).collect())
                .collect();
        for impl_ref in impl_order {
            mem_range.insert(impl_ref.clone(), (0x8000_0000u64, 0x8000_003fu64));
        }

        let hart_count = canonical_test_program.len().max(match config.hart_count() {
            0 => 0,
            n => n as usize,
        });
        let harts = (0..hart_count)
            .map(
                |hart_id| riscv_fuzz_testcase_engine::testcase::HartProgram {
                    hart_id: hart_id as u32,
                    init: Vec::new(),
                    test: canonical_test_program
                        .get(hart_id)
                        .cloned()
                        .unwrap_or_default(),
                    shared_init_sequence_indices: Default::default(),
                    shared_test_sequence_indices: Default::default(),
                },
            )
            .collect();

        GeneratedTestCase {
            config,
            harts,
            mem_range,
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        }
    }

    #[test]
    fn infer_target_hart_for_test_index_maps_traceable_indices() {
        let impl_order = vec!["spike".to_string(), "rocket".to_string()];
        let mut programs: HashMap<String, Vec<Vec<RiscvInstruction>>> = HashMap::new();
        for impl_ref in &impl_order {
            programs.insert(
                impl_ref.clone(),
                vec![vec![addi(1, 1, 1)], vec![addi(2, 2, 1), addi(3, 3, 1)]],
            );
        }
        let testcase = minimal_multicore_testcase(&impl_order, &programs);

        assert_eq!(infer_target_hart_for_test_index(&testcase, 0).unwrap(), 0);
        assert_eq!(infer_target_hart_for_test_index(&testcase, 1).unwrap(), 1);
        assert_eq!(infer_target_hart_for_test_index(&testcase, 2).unwrap(), 1);
    }

    #[test]
    fn infer_target_hart_for_test_index_errors_on_inconsistent_mapping() {
        let testcase = GeneratedTestCase {
            config: TestCaseConfig {
                isa_base: ISABase::Rv32,
                extension_insts_count: 0,
                extension_inst_scaling: None,
                mem_size: 64,
                float_extension: Default::default(),
                unaligned_access_required: Some(false),
                random_config: RandomConfig {
                    memory_access_config: MemoryAccessConfig {
                        memory_base_address_range: MemoryAccessConfig::default()
                            .memory_base_address_range,
                        memory_offset_immediate_range: (0, 0),
                        memory_base_register_number_range: (10, 15),
                    },
                    register_operand_config: RegisterOperandConfig {
                        integer_register_number_range: (16, 25),
                        floating_point_register_number_range: (16, 25),
                        vector_register_number_range: (16, 25),
                    },
                    ..Default::default()
                },
                seed: None,
                hart_config: HartConfig::Dual {
                    sync: MulticoreSyncConfig::default(),
                },
            },
            harts: vec![riscv_fuzz_testcase_engine::testcase::HartProgram {
                hart_id: 0,
                init: Vec::new(),
                test: vec![InstructionSequence::new(addi(1, 1, 1))],
                shared_init_sequence_indices: Default::default(),
                shared_test_sequence_indices: Default::default(),
            }],
            mem_range: HashMap::new(),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        };

        let err = infer_target_hart_for_test_index(&testcase, 1)
            .expect_err("should fail for missing test index");
        assert!(matches!(
            err,
            DiffAnalysisError::TestInstructionCountMismatch { .. }
        ));
    }

    #[test]
    fn infer_target_hart_for_test_index_rejects_empty_impl_order() {
        let testcase = minimal_multicore_testcase(&[], &HashMap::new());

        let err = infer_target_hart_for_test_index(&testcase, 0)
            .expect_err("empty testcase should not resolve hart mapping");
        assert!(matches!(
            err,
            DiffAnalysisError::TestInstructionCountMismatch { .. }
        ));
    }

    #[test]
    fn select_traceable_instruction_pair_skips_non_traceable_lines() {
        let mut block = StructuredTestInstructions::default();
        block.push_pseudo_to_hart(
            0,
            PseudoInstruction::Label {
                name: "foo".to_string(),
            },
        );
        block.push_pseudo_to_hart(
            0,
            PseudoInstruction::Label {
                name: "bar".to_string(),
            },
        );
        block.push_instruction_to_hart(0, addi(1, 0, 1));
        block.push_instruction_to_hart(1, addi(2, 0, 2));

        assert_eq!(block.traceable_instruction_at(0).as_deref(), Some("foo:"));
        assert_eq!(block.traceable_instruction_at(1).as_deref(), Some("bar:"));
        assert_eq!(
            block.traceable_instruction_at(2).as_deref(),
            Some("addi x1, x0, 1")
        );
        assert_eq!(
            block.traceable_instruction_at(3).as_deref(),
            Some("addi x2, x0, 2")
        );
        assert!(block.traceable_instruction_at(4).is_none());
    }

    #[test]
    fn probe_helpers_collect_mismatching_registers_and_memory() {
        let impl_order = vec!["spike".to_string(), "rocket".to_string()];

        let mut register_changes = BTreeMap::new();
        register_changes.insert(
            "spike".to_string(),
            vec![
                ("x0".to_string(), 0),
                ("x2".to_string(), 0x2),
                ("x5".to_string(), 0x5),
                ("f1".to_string(), 0xdeadbeef),
            ],
        );
        register_changes.insert("rocket".to_string(), vec![("x2".to_string(), 0x3)]);

        let mut memory_changes = BTreeMap::new();
        memory_changes.insert("spike".to_string(), vec![(0x10, 0xaa), (0x11, 0xbb)]);
        memory_changes.insert("rocket".to_string(), vec![(0x10, 0xaa), (0x11, 0xbc)]);

        let diff = WriteDifference {
            test_index: 0,
            global_index: 0,
            register_changes,
            memory_changes,
            register_contexts: BTreeMap::new(),
            memory_contexts: BTreeMap::new(),
        };

        let regs = collect_mismatching_integer_registers(&diff, &impl_order);
        assert_eq!(regs, vec!["x2".to_string(), "x5".to_string()]);

        let mems = collect_mismatching_memory_offsets(&diff, &impl_order);
        assert_eq!(mems, vec![0x11]);
    }

    #[test]
    fn copy_reg_instruction_uses_non_move_probe_form() {
        let rv32 = copy_reg_instruction(ISABase::Rv32, 8, "x14").expect("rv32 probe copy");
        assert_eq!(rv32.to_string(), "xori x8, x14, 0");

        let rv64 = copy_reg_instruction(ISABase::Rv64, 9, "x13").expect("rv64 probe copy");
        assert_eq!(rv64.to_string(), "xori x9, x13, 0");
    }

    #[test]
    fn likely_false_positive_cause_has_stable_label() {
        assert_eq!(
            WriteDifferenceCause::LikelyFalsePositive.to_string(),
            "疑似误报（replay/history-min 未能保留差异）"
        );
    }

    #[test]
    fn probe_refuted_single_instruction_becomes_likely_false_positive() {
        assert!(matches!(
            effective_write_difference_cause(
                WriteDifferenceCause::SingleInstruction,
                Some(&WriteProbeOutcome::Refuted),
            ),
            WriteDifferenceCause::LikelyFalsePositive
        ));
        assert!(matches!(
            effective_write_difference_cause(
                WriteDifferenceCause::SingleInstruction,
                Some(&WriteProbeOutcome::MismatchDisappeared),
            ),
            WriteDifferenceCause::LikelyFalsePositive
        ));
    }

    #[test]
    fn confirmed_probe_keeps_single_instruction_cause() {
        assert!(matches!(
            effective_write_difference_cause(
                WriteDifferenceCause::SingleInstruction,
                Some(&WriteProbeOutcome::Confirmed),
            ),
            WriteDifferenceCause::SingleInstruction
        ));
        assert!(matches!(
            effective_write_difference_cause(
                WriteDifferenceCause::DependsOnHistory,
                Some(&WriteProbeOutcome::Refuted),
            ),
            WriteDifferenceCause::DependsOnHistory
        ));
    }
}
