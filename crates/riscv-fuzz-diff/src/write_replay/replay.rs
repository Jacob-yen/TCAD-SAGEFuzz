use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

use riscv_fuzz_plugin_api::context::{MemoryContext, MemoryValueWidth, RegisterContext};
use riscv_fuzz_testcase_engine::instruction::{
    generate_memory_context_restore_instructions, generate_register_context_restore_instructions,
};
use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;
use riscv_instruction::RiscvInstruction;

use super::{DiffAnalysisError, WriteDifference};
use crate::normalize::{logical_instruction_at_location, test_instruction_location};
use crate::{iteration::persist_iteration_testcase_at, view::DiffExecutionView};

pub(super) struct ReplayTestcases {
    pub(super) canonical_testcase: GeneratedTestCase,
    pub(super) per_impl_testcases: HashMap<String, GeneratedTestCase>,
    pub(super) target_index: usize,
}

pub(super) fn build_replay_testcases(
    diff: &WriteDifference,
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
    impl_order: &[String],
    replay_dir: &Path,
) -> Result<ReplayTestcases, DiffAnalysisError> {
    if impl_order.is_empty() {
        return Err(DiffAnalysisError::TestInstructionCountMismatch {
            details: "no implementations available for write difference classification".to_string(),
        });
    }

    let mut canonical_testcase = None;
    let mut per_impl_testcases = HashMap::new();
    let mut expected_len = None;
    let temp_range = testcase
        .config
        .random_config
        .memory_access_config
        .memory_base_register_number_range;
    let target_hart = infer_target_hart_for_test_index(testcase, diff.test_index)?;
    let original_entry = testcase
        .traceable_test_entry_at(diff.test_index)
        .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
            details: format!(
                "missing traceable test instruction {} for replay construction",
                diff.test_index
            ),
        })?;

    for impl_ref in impl_order {
        let output =
            outputs
                .get(impl_ref)
                .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                    core_id: impl_ref.to_string(),
                })?;

        let location = test_instruction_location(impl_order, testcase, diff.test_index)?;
        let instruction = logical_instruction_at_location(output, location).ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing context for test instruction {} (hart {} local {}) in {}",
                    diff.test_index,
                    location.hart_id,
                    location.hart_local_instruction_index,
                    impl_ref
                ),
            }
        })?;
        let context = &instruction.context;

        let mut program = testcase
            .harts
            .iter()
            .map(|hart| (hart.hart_id, Vec::new()))
            .collect::<BTreeMap<_, Vec<_>>>();

        // 为每个实现恢复目标指令执行前一刻观测到的体系结构状态，
        // Restore architectural state as observed right before the target instruction for each
        // implementation,
        // 再把原始目标指令追加为最后一个 traceable 步骤。
        // then append the original target instruction as the final traceable step.
        let memory_before = replay_memory_before(&context.memory_context)?;
        let memory_restore = generate_memory_context_restore_instructions(
            &memory_before,
            output.mem_range.0,
            output.isa_base,
            temp_range,
        )?;
        for entry in memory_restore {
            push_entry_to_hart(&mut program, target_hart, entry);
        }

        let registers_before = replay_registers_before(&context.register_context)?;
        let register_restore = generate_register_context_restore_instructions(
            &registers_before,
            output.isa_base,
            temp_range,
        )?;
        for entry in register_restore {
            push_entry_to_hart(&mut program, target_hart, entry);
        }

        push_entry_to_hart(&mut program, target_hart, original_entry.clone());

        let len: usize = program.values().map(Vec::len).sum();
        let replay_testcase = testcase.with_test_sequences_by_hart(program.into_iter().collect());
        if let Some(expected) = expected_len {
            if len != expected {
                let partial_path = replay_dir.join(format!("testcase_{impl_ref}.json"));
                persist_iteration_testcase_at(&replay_testcase, &partial_path)?;

                return Err(DiffAnalysisError::TestInstructionCountMismatch {
                    details: format!(
                        "write replay produced {} traceable test instructions for {}, expected {}; partial replay testcase saved to {}",
                        len,
                        impl_ref,
                        expected,
                        partial_path.display()
                    ),
                });
            }
        } else {
            expected_len = Some(len);
            canonical_testcase = Some(replay_testcase.clone());
        }

        per_impl_testcases.insert(
            impl_ref.clone(),
            testcase_for_single_impl(&replay_testcase, impl_ref)?,
        );
    }

    let total_len =
        expected_len.ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
            details: "write replay produced no traceable test instructions".to_string(),
        })?;

    let target_index = total_len.checked_sub(1).ok_or_else(|| {
        DiffAnalysisError::TestInstructionCountMismatch {
            details: "write replay produced empty instruction list".to_string(),
        }
    })?;

    Ok(ReplayTestcases {
        canonical_testcase: canonical_testcase.ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: "write replay produced no canonical testcase".to_string(),
            }
        })?,
        per_impl_testcases,
        target_index,
    })
}

pub(super) fn infer_target_hart_for_test_index(
    testcase: &GeneratedTestCase,
    target_test_index: usize,
) -> Result<u32, DiffAnalysisError> {
    if !testcase.config.is_dual_hart() {
        return Ok(0);
    }

    if let Some(hart_id) = testcase.hart_for_test_index(target_test_index) {
        return Ok(hart_id);
    }

    Err(DiffAnalysisError::TestInstructionCountMismatch {
        details: format!(
            "missing traceable test instruction {} while inferring target hart",
            target_test_index
        ),
    })
}

fn push_entry_to_hart(
    program: &mut BTreeMap<
        u32,
        Vec<riscv_instruction_types::InstructionSequence<RiscvInstruction>>,
    >,
    hart_id: u32,
    entry: RiscvInstruction,
) {
    program
        .entry(hart_id)
        .or_default()
        .push(riscv_instruction_types::InstructionSequence::new(entry));
}

fn testcase_for_single_impl(
    testcase: &GeneratedTestCase,
    impl_ref: &str,
) -> Result<GeneratedTestCase, DiffAnalysisError> {
    let mut single_impl = testcase.clone();
    single_impl
        .mem_range
        .retain(|core_id, _| core_id == impl_ref);
    if single_impl.mem_range.is_empty() {
        return Err(DiffAnalysisError::MissingInstructions {
            core_id: impl_ref.to_string(),
        });
    }
    Ok(single_impl)
}

fn replay_registers_before(
    context: &RegisterContext,
) -> Result<HashMap<String, u64>, DiffAnalysisError> {
    let mut registers_before = HashMap::new();
    for entry in &context.entries {
        if registers_before
            .insert(entry.name.clone(), entry.value)
            .is_some()
        {
            return Err(DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "replay register context contains duplicate entry for {}",
                    entry.name
                ),
            });
        }
    }
    Ok(registers_before)
}

fn replay_memory_before(context: &MemoryContext) -> Result<BTreeMap<u64, u8>, DiffAnalysisError> {
    let mut memory_before = BTreeMap::new();
    for entry in &context.entries {
        let width = match entry.width {
            MemoryValueWidth::Byte => 1,
            MemoryValueWidth::Half => 2,
            MemoryValueWidth::Word => 4,
            MemoryValueWidth::Dword => 8,
        };
        if entry.addresses.len() != width {
            return Err(DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "replay memory context entry width mismatch: expected {width} byte(s), got {}",
                    entry.addresses.len()
                ),
            });
        }
        for (byte_index, addr) in entry.addresses.iter().enumerate() {
            let byte = ((entry.value >> (byte_index * 8)) & 0xff) as u8;
            if memory_before.insert(*addr, byte).is_some() {
                return Err(DiffAnalysisError::TestInstructionCountMismatch {
                    details: format!(
                        "replay memory context contains duplicate byte entry at addr 0x{addr:x}"
                    ),
                });
            }
        }
    }
    Ok(memory_before)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::path::Path;

    use riscv_fuzz_plugin_api::context::{
        ExecutionContextOutput, HartExecutionContextOutput, InstructionContext,
        LogicalInstructionContextOutput, MemoryContext, MemoryContextEntry, MemoryValueWidth,
        RegisterContext, RegisterContextEntry,
    };
    use riscv_fuzz_plugin_api::{ISABase, IsaExtensions, PerfMetrics};
    use riscv_fuzz_testcase_engine::testcase::{
        GeneratedTestCase, HartConfig, HartProgram, TestCaseConfig,
    };
    use riscv_instruction::{RV32Extensions, RV32IInstructions, RV32Instruction, RiscvInstruction};
    use riscv_instruction_types::{
        Immediate, InstructionSequence, IntegerRegister, MemoryAccessConfig, RandomConfig,
        RegisterOperandConfig, ValidatedValue,
    };

    use super::{ReplayTestcases, build_replay_testcases};
    use crate::view::DiffExecutionView;
    use crate::write_replay::WriteDifference;

    fn x_reg(reg: u8) -> IntegerRegister {
        IntegerRegister::new(reg).expect("valid integer register")
    }

    fn addi(rd: u8, rs1: u8, imm: i16) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(
            riscv_instruction::RV32_I_ADDI {
                xd: x_reg(rd),
                xs1: x_reg(rs1),
                imm: Immediate::<12>::new(imm.into()).expect("valid addi immediate"),
            },
        )))
    }

    fn sample_testcase() -> GeneratedTestCase {
        GeneratedTestCase {
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
                        memory_base_register_number_range: (8, 12),
                    },
                    register_operand_config: RegisterOperandConfig {
                        integer_register_number_range: (13, 20),
                        floating_point_register_number_range: (13, 20),
                        vector_register_number_range: (13, 20),
                    },
                    ..Default::default()
                },
                seed: None,
                hart_config: HartConfig::Single,
            },
            harts: vec![HartProgram {
                hart_id: 0,
                init: Vec::new(),
                test: vec![InstructionSequence::new(addi(13, 0, 1))],
                shared_init_sequence_indices: Default::default(),
                shared_test_sequence_indices: Default::default(),
            }],
            mem_range: HashMap::from([
                ("spike".to_string(), (0x1_0000, 0x1_003f)),
                ("picorv32".to_string(), (0x2_0000, 0x2_003f)),
            ]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        }
    }

    fn logical_instruction_with_sp(value: u64) -> LogicalInstructionContextOutput {
        LogicalInstructionContextOutput {
            hart_id: 0,
            instruction_index: 0,
            local_instruction_index: 0,
            instruction: addi(13, 0, 1),
            trace_inst_id: riscv_fuzz_plugin_api::TraceInstId::new(0, 0),
            asm_instruction_indices: vec![0],
            exceptions: Vec::new(),
            register_writes: Vec::new(),
            memory_writes: Vec::new(),
            timing: Default::default(),
            context: InstructionContext {
                registers_before: HashMap::from([("x2".to_string(), value)]),
                register_context: RegisterContext {
                    entries: vec![RegisterContextEntry {
                        name: "x2".to_string(),
                        value,
                    }],
                },
                memory_context: MemoryContext {
                    base_register: "x2".to_string(),
                    base_offset: 0x10,
                    entries: vec![MemoryContextEntry {
                        width: MemoryValueWidth::Word,
                        addresses: vec![0x10, 0x11, 0x12, 0x13],
                        value: 0x4433_2211,
                    }],
                },
                ..Default::default()
            },
        }
    }

    fn execution_output(core_id: &str, mem_range: (u64, u64), sp: u64) -> DiffExecutionView {
        crate::view::diff_execution_view_from_context(ExecutionContextOutput {
            per_hart: BTreeMap::from([(
                0,
                HartExecutionContextOutput {
                    hart_id: 0,
                    instructions: Vec::new(),
                },
            )]),
            framework_per_hart: BTreeMap::new(),
            logical_view: vec![logical_instruction_with_sp(sp)],
            mem_range,
            core_id: core_id.to_string(),
            isa_base: ISABase::Rv32,
            perf: PerfMetrics::default(),
        })
    }

    fn sample_outputs(testcase: &GeneratedTestCase) -> HashMap<String, DiffExecutionView> {
        HashMap::from([
            (
                "spike".to_string(),
                execution_output("spike", testcase.mem_range["spike"], 0x1_0010),
            ),
            (
                "picorv32".to_string(),
                execution_output("picorv32", testcase.mem_range["picorv32"], 0x2_0010),
            ),
        ])
    }

    fn sample_difference() -> WriteDifference {
        WriteDifference {
            test_index: 0,
            global_index: 0,
            register_changes: BTreeMap::new(),
            memory_changes: BTreeMap::new(),
            register_contexts: BTreeMap::new(),
            memory_contexts: BTreeMap::new(),
        }
    }

    fn build_sample_replay_testcases() -> ReplayTestcases {
        let testcase = sample_testcase();
        let outputs = sample_outputs(&testcase);
        build_replay_testcases(
            &sample_difference(),
            &testcase,
            &outputs,
            &["spike".to_string(), "picorv32".to_string()],
            Path::new("."),
        )
        .expect("replay testcase construction should succeed")
    }

    #[test]
    fn build_replay_testcases_preserves_impl_specific_restore_state() {
        let replay = build_sample_replay_testcases();

        assert_eq!(
            replay
                .canonical_testcase
                .traceable_test_instruction_at(replay.target_index)
                .as_deref(),
            Some("addi x13, x0, 1")
        );
        assert_eq!(
            replay.per_impl_testcases["spike"]
                .traceable_test_instruction_at(replay.target_index)
                .as_deref(),
            Some("addi x13, x0, 1")
        );
        assert_eq!(
            replay.per_impl_testcases["picorv32"]
                .traceable_test_instruction_at(replay.target_index)
                .as_deref(),
            Some("addi x13, x0, 1")
        );
        assert_eq!(
            replay.per_impl_testcases["spike"]
                .traceable_test_instruction_at(replay.target_index - 1)
                .as_deref(),
            Some("li x2, 65552")
        );
        assert_eq!(
            replay.per_impl_testcases["picorv32"]
                .traceable_test_instruction_at(replay.target_index - 1)
                .as_deref(),
            Some("li x2, 131088")
        );
    }

    #[test]
    fn build_replay_testcases_trim_impl_specific_mem_ranges() {
        let replay = build_sample_replay_testcases();

        assert_eq!(
            replay.per_impl_testcases["spike"]
                .mem_range
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["spike".to_string()]
        );
        assert_eq!(
            replay.per_impl_testcases["picorv32"]
                .mem_range
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["picorv32".to_string()]
        );
    }

    #[test]
    fn build_replay_testcases_restore_only_instruction_relevant_context() {
        let replay = build_sample_replay_testcases();
        let spike_test = &replay.per_impl_testcases["spike"];
        let rendered = spike_test.harts[0]
            .test
            .iter()
            .map(|seq| seq.instruction.to_string())
            .collect::<Vec<_>>();

        assert_eq!(rendered[0], "li x8, 65552");
        assert_eq!(rendered[1], "li x9, 17");
        assert_eq!(rendered[2], "sb x9, 0(x8)");
        assert_eq!(rendered[3], "li x8, 65553");
        assert_eq!(rendered[4], "li x9, 34");
        assert_eq!(rendered[5], "sb x9, 0(x8)");
        assert_eq!(rendered[6], "li x8, 65554");
        assert_eq!(rendered[7], "li x9, 51");
        assert_eq!(rendered[8], "sb x9, 0(x8)");
        assert_eq!(rendered[9], "li x8, 65555");
        assert_eq!(rendered[10], "li x9, 68");
        assert_eq!(rendered[11], "sb x9, 0(x8)");
        assert_eq!(rendered[12], "li x2, 65552");
        assert_eq!(rendered[13], "addi x13, x0, 1");
    }
}
