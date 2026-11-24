//! 从单次执行快照中检测逐指令的异常差异和写差异。
//! Detect per-instruction exception and write differences from one execution snapshot.
//!
//! 检测阶段以逻辑测试指令为粒度工作，
//! Detection works at logical test-instruction granularity,
//! 并保留每个实现对应指令执行前的寄存器/内存上下文，
//! and it also preserves the pre-instruction register/memory context for each implementation,
//! 这样后续报告和 replay 就能解释差异而不必重新运行 testcase。
//! so later reporting and replay can explain the mismatch without rerunning the testcase.

use std::collections::{BTreeMap, HashMap};

use riscv_fuzz_plugin_api::{
    ExceptionCause,
    context::{MemoryContext, RegisterContext},
};
use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;

use super::{
    normalize::{
        PhaseInstructionLocation, WriteEffect, determine_test_length,
        logical_instruction_at_location, logical_instruction_effects_for_diff,
        test_instruction_locations,
    },
    types::DiffAnalysisError,
    view::DiffExecutionView,
};

#[derive(Clone)]
pub(crate) struct ExceptionDifference {
    pub(crate) test_index: usize,
    pub(crate) global_index: usize,
    pub(crate) per_impl: BTreeMap<String, Vec<ExceptionCause>>,
    pub(crate) register_contexts: BTreeMap<String, RegisterContext>,
    pub(crate) memory_contexts: BTreeMap<String, MemoryContext>,
}

#[derive(Clone, Default)]
pub(crate) struct ExceptionDetectionResult {
    pub(crate) trigger_mismatches: Vec<ExceptionDifference>,
    pub(crate) cause_mismatches: Vec<ExceptionDifference>,
}

#[derive(Clone)]
pub(crate) struct WriteDifference {
    pub(crate) test_index: usize,
    pub(crate) global_index: usize,
    pub(crate) register_changes: BTreeMap<String, Vec<(String, u64)>>,
    pub(crate) memory_changes: BTreeMap<String, Vec<(u64, u8)>>,
    pub(crate) register_contexts: BTreeMap<String, RegisterContext>,
    pub(crate) memory_contexts: BTreeMap<String, MemoryContext>,
}

fn record_global_index(
    slot: &mut Option<usize>,
    observed: usize,
    test_idx: usize,
    impl_ref: &str,
) -> Result<(), DiffAnalysisError> {
    match slot {
        Some(existing) if *existing != observed => {
            Err(DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "test instruction {test_idx} maps to different global indices: {existing} vs {observed} in {impl_ref}"
                ),
            })
        }
        Some(_) => Ok(()),
        None => {
            *slot = Some(observed);
            Ok(())
        }
    }
}

pub(crate) fn detect_unique_exceptions(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
) -> Result<ExceptionDetectionResult, DiffAnalysisError> {
    let test_len = determine_test_length(impl_order, testcase)?;
    if test_len == 0 {
        return Ok(ExceptionDetectionResult::default());
    }
    let test_locations = test_instruction_locations(impl_order, testcase)?;

    let mut result = ExceptionDetectionResult::default();

    for location in test_locations {
        let test_idx = location.phase_index;
        let mut global_idx = None;
        let mut rows = Vec::with_capacity(impl_order.len());
        let mut saw_empty = false;
        let mut saw_non_empty = false;
        let mut mismatch_seen = false;
        let mut baseline_codes: Option<Vec<u64>> = None;

        for impl_ref in impl_order {
            let output =
                outputs
                    .get(impl_ref)
                    .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                        core_id: impl_ref.to_string(),
                    })?;
            let instruction_ctx = logical_instruction_at_location(output, location).ok_or_else(|| {
                DiffAnalysisError::TestInstructionCountMismatch {
                    details: format!(
                        "missing execution context for test instruction {} (hart {} local {}) in {}",
                        test_idx,
                        location.hart_id,
                        location.hart_local_instruction_index,
                        impl_ref
                    ),
                }
            })?;
            record_global_index(
                &mut global_idx,
                instruction_ctx.instruction_index,
                test_idx,
                impl_ref,
            )?;
            record_global_index(&mut global_idx, location.global_index, test_idx, "testcase")?;
            let mut causes = instruction_ctx.exceptions.clone();
            causes.sort();
            let cause_codes = causes.iter().map(|cause| cause.code).collect::<Vec<_>>();
            if causes.is_empty() {
                saw_empty = true;
            } else {
                saw_non_empty = true;
            }
            match &baseline_codes {
                Some(existing) if existing != &cause_codes => mismatch_seen = true,
                None => baseline_codes = Some(cause_codes),
                _ => {}
            }
            rows.push((
                impl_ref.clone(),
                causes,
                &instruction_ctx.context.register_context,
                &instruction_ctx.context.memory_context,
            ));
        }

        if saw_non_empty && mismatch_seen {
            let global_idx = global_idx.expect("non-empty implementation order produced rows");
            // 这里要区分“有的实现触发异常、有的没有”与“所有实现都触发异常，但异常原因不同”。
            // Distinguish "some impls trapped, some did not" from "all trapped but disagreed
            // on cause".
            // runner 只会移除前者；后者只出报告，不改变 testcase 结构。
            // The runner only removes the former; the latter is report-only.
            let mut per_impl = BTreeMap::new();
            let mut register_contexts = BTreeMap::new();
            let mut memory_contexts = BTreeMap::new();
            for (impl_ref, causes, register_context, memory_context) in rows {
                per_impl.insert(impl_ref.clone(), causes);
                register_contexts.insert(impl_ref.clone(), register_context.clone());
                memory_contexts.insert(impl_ref, memory_context.clone());
            }
            let diff = ExceptionDifference {
                test_index: test_idx,
                global_index: global_idx,
                per_impl,
                register_contexts,
                memory_contexts,
            };
            if saw_empty {
                result.trigger_mismatches.push(diff);
            } else {
                result.cause_mismatches.push(diff);
            }
        }
    }

    Ok(result)
}

pub(crate) fn detect_unique_exceptions_at_location(
    impl_order: &[String],
    _testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
    location: PhaseInstructionLocation,
) -> Result<ExceptionDetectionResult, DiffAnalysisError> {
    let mut result = ExceptionDetectionResult::default();
    let test_idx = location.phase_index;
    let mut global_idx = None;
    let mut rows = Vec::with_capacity(impl_order.len());
    let mut saw_empty = false;
    let mut saw_non_empty = false;
    let mut mismatch_seen = false;
    let mut baseline_codes: Option<Vec<u64>> = None;

    for impl_ref in impl_order {
        let output =
            outputs
                .get(impl_ref)
                .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                    core_id: impl_ref.to_string(),
                })?;
        let instruction_ctx = logical_instruction_at_location(output, location).ok_or_else(
            || DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing execution context for test instruction {} (hart {} local {}) in {}",
                    test_idx, location.hart_id, location.hart_local_instruction_index, impl_ref
                ),
            },
        )?;
        record_global_index(
            &mut global_idx,
            instruction_ctx.instruction_index,
            test_idx,
            impl_ref,
        )?;
        record_global_index(&mut global_idx, location.global_index, test_idx, "testcase")?;
        let mut causes = instruction_ctx.exceptions.clone();
        causes.sort();
        let cause_codes = causes.iter().map(|cause| cause.code).collect::<Vec<_>>();
        if causes.is_empty() {
            saw_empty = true;
        } else {
            saw_non_empty = true;
        }
        match &baseline_codes {
            Some(existing) if existing != &cause_codes => mismatch_seen = true,
            None => baseline_codes = Some(cause_codes),
            _ => {}
        }
        rows.push((
            impl_ref.clone(),
            causes,
            &instruction_ctx.context.register_context,
            &instruction_ctx.context.memory_context,
        ));
    }

    if saw_non_empty && mismatch_seen {
        let global_idx = global_idx.expect("non-empty implementation order produced rows");
        let mut per_impl = BTreeMap::new();
        let mut register_contexts = BTreeMap::new();
        let mut memory_contexts = BTreeMap::new();
        for (impl_ref, causes, register_context, memory_context) in rows {
            per_impl.insert(impl_ref.clone(), causes);
            register_contexts.insert(impl_ref.clone(), register_context.clone());
            memory_contexts.insert(impl_ref, memory_context.clone());
        }
        let diff = ExceptionDifference {
            test_index: test_idx,
            global_index: global_idx,
            per_impl,
            register_contexts,
            memory_contexts,
        };
        if saw_empty {
            result.trigger_mismatches.push(diff);
        } else {
            result.cause_mismatches.push(diff);
        }
    }
    Ok(result)
}

pub(crate) fn detect_unique_writes(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
) -> Result<Vec<WriteDifference>, DiffAnalysisError> {
    let test_len = determine_test_length(impl_order, testcase)?;
    if test_len == 0 {
        return Ok(Vec::new());
    }
    let test_locations = test_instruction_locations(impl_order, testcase)?;

    let mut result = Vec::new();
    let register_scope = &testcase.config;

    for location in test_locations {
        let test_idx = location.phase_index;
        let mut global_idx = None;
        let mut rows = Vec::with_capacity(impl_order.len());
        let mut mismatch_seen = false;
        let mut baseline: Option<WriteEffect> = None;

        for impl_ref in impl_order {
            let ctx =
                outputs
                    .get(impl_ref)
                    .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                        core_id: impl_ref.to_string(),
                    })?;
            let instruction = logical_instruction_at_location(ctx, location).ok_or_else(|| {
                DiffAnalysisError::TestInstructionCountMismatch {
                    details: format!(
                        "missing execution context for test instruction {} (hart {} local {}) in {}",
                        test_idx,
                        location.hart_id,
                        location.hart_local_instruction_index,
                        impl_ref
                    ),
                }
            })?;
            record_global_index(
                &mut global_idx,
                instruction.instruction_index,
                test_idx,
                impl_ref,
            )?;
            record_global_index(&mut global_idx, location.global_index, test_idx, "testcase")?;
            let (regs, mems) = logical_instruction_effects_for_diff(instruction, register_scope);
            match &baseline {
                Some((baseline_regs, baseline_mems))
                    if baseline_regs != &regs || baseline_mems != &mems =>
                {
                    mismatch_seen = true;
                }
                None => baseline = Some((regs.clone(), mems.clone())),
                _ => {}
            }
            rows.push((
                impl_ref.clone(),
                regs,
                mems,
                &instruction.context.register_context,
                &instruction.context.memory_context,
            ));
        }

        if mismatch_seen {
            let global_idx = global_idx.expect("non-empty implementation order produced rows");
            // 写入在比较前会先做规范化，
            // Writes are compared after canonicalization,
            // 因此单纯的日志顺序差异不会制造假阳性的写差异。
            // so logging order alone does not create a false-positive write difference.
            let mut register_map = BTreeMap::new();
            let mut memory_map = BTreeMap::new();
            let mut register_contexts = BTreeMap::new();
            let mut memory_contexts = BTreeMap::new();
            for (impl_ref, regs, mems, register_context, memory_context) in rows {
                register_map.insert(impl_ref.clone(), regs);
                memory_map.insert(impl_ref.clone(), mems);
                register_contexts.insert(impl_ref.clone(), register_context.clone());
                memory_contexts.insert(impl_ref, memory_context.clone());
            }
            result.push(WriteDifference {
                test_index: test_idx,
                global_index: global_idx,
                register_changes: register_map,
                memory_changes: memory_map,
                register_contexts,
                memory_contexts,
            });
        }
    }

    Ok(result)
}

pub(crate) fn detect_unique_writes_at_location(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
    location: PhaseInstructionLocation,
) -> Result<Vec<WriteDifference>, DiffAnalysisError> {
    let test_idx = location.phase_index;
    let mut global_idx = None;
    let mut rows = Vec::with_capacity(impl_order.len());
    let mut mismatch_seen = false;
    let mut baseline: Option<WriteEffect> = None;

    for impl_ref in impl_order {
        let ctx = outputs
            .get(impl_ref)
            .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.to_string(),
            })?;
        let instruction = logical_instruction_at_location(ctx, location).ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing execution context for test instruction {} (hart {} local {}) in {}",
                    test_idx, location.hart_id, location.hart_local_instruction_index, impl_ref
                ),
            }
        })?;
        record_global_index(
            &mut global_idx,
            instruction.instruction_index,
            test_idx,
            impl_ref,
        )?;
        record_global_index(&mut global_idx, location.global_index, test_idx, "testcase")?;
        let (regs, mems) = logical_instruction_effects_for_diff(instruction, &testcase.config);
        match &baseline {
            Some((baseline_regs, baseline_mems))
                if baseline_regs != &regs || baseline_mems != &mems =>
            {
                mismatch_seen = true;
            }
            None => baseline = Some((regs.clone(), mems.clone())),
            _ => {}
        }
        rows.push((
            impl_ref.clone(),
            regs,
            mems,
            &instruction.context.register_context,
            &instruction.context.memory_context,
        ));
    }

    if !mismatch_seen {
        return Ok(Vec::new());
    }

    let global_idx = global_idx.expect("non-empty implementation order produced rows");
    let mut register_map = BTreeMap::new();
    let mut memory_map = BTreeMap::new();
    let mut register_contexts = BTreeMap::new();
    let mut memory_contexts = BTreeMap::new();
    for (impl_ref, regs, mems, register_context, memory_context) in rows {
        register_map.insert(impl_ref.clone(), regs);
        memory_map.insert(impl_ref.clone(), mems);
        register_contexts.insert(impl_ref.clone(), register_context.clone());
        memory_contexts.insert(impl_ref, memory_context.clone());
    }

    Ok(vec![WriteDifference {
        test_index: test_idx,
        global_index: global_idx,
        register_changes: register_map,
        memory_changes: memory_map,
        register_contexts,
        memory_contexts,
    }])
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    use riscv_fuzz_plugin_api::{
        ISABase, IsaExtensions, PerfMetrics, RegisterValue,
        context::{
            ExecutionContextOutput, HartExecutionContextOutput, InstructionContext,
            LogicalInstructionContextOutput,
        },
    };
    use riscv_fuzz_testcase_engine::testcase::{
        GeneratedTestCase, HartConfig, HartProgram, TestCaseConfig,
    };
    use riscv_instruction::{
        RV32_I_ADDI, RV32Extensions, RV32IInstructions, RV32Instruction, RiscvInstruction,
    };
    use riscv_instruction_types::{
        Immediate, InstructionSequence, IntegerRegister, MemoryAccessConfig, RandomConfig,
        RegisterOperandConfig, ValidatedValue,
    };

    use super::*;
    use crate::view::DiffExecutionView;

    fn x_reg(reg: u8) -> IntegerRegister {
        IntegerRegister::new(reg).expect("valid integer register")
    }

    fn addi(rd: u8, rs1: u8, imm: i16) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(RV32_I_ADDI {
            xd: x_reg(rd),
            xs1: x_reg(rs1),
            imm: Immediate::<12>::new(imm.into()).expect("valid addi immediate"),
        })))
    }

    fn testcase_with_two_harts_init_and_test() -> GeneratedTestCase {
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
                    sync: Default::default(),
                },
            },
            harts: vec![
                HartProgram {
                    hart_id: 0,
                    init: vec![InstructionSequence::new(addi(16, 0, 1))],
                    test: vec![InstructionSequence::new(addi(17, 0, 2))],
                    shared_init_sequence_indices: BTreeSet::new(),
                    shared_test_sequence_indices: BTreeSet::new(),
                },
                HartProgram {
                    hart_id: 1,
                    init: vec![InstructionSequence::new(addi(18, 0, 3))],
                    test: vec![InstructionSequence::new(addi(19, 0, 4))],
                    shared_init_sequence_indices: BTreeSet::new(),
                    shared_test_sequence_indices: BTreeSet::new(),
                },
            ],
            mem_range: HashMap::from([
                ("spike".to_string(), (0x1_0000, 0x1_003f)),
                ("xiangshan".to_string(), (0x2_0000, 0x2_003f)),
            ]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        }
    }

    fn reg_write(name: &str, value: u64) -> RegisterValue {
        RegisterValue {
            name: name.to_string(),
            value,
        }
    }

    fn logical_instruction(
        hart_id: u32,
        instruction_index: usize,
        local_instruction_index: usize,
        instruction: RiscvInstruction,
        register_writes: Vec<RegisterValue>,
    ) -> LogicalInstructionContextOutput {
        LogicalInstructionContextOutput {
            hart_id,
            instruction_index,
            local_instruction_index,
            instruction,
            trace_inst_id: riscv_fuzz_plugin_api::TraceInstId::new(
                hart_id,
                local_instruction_index,
            ),
            asm_instruction_indices: vec![local_instruction_index],
            exceptions: Vec::new(),
            register_writes,
            memory_writes: Vec::new(),
            timing: Default::default(),
            context: InstructionContext::default(),
        }
    }

    fn output(core_id: &str, mem_range: (u64, u64), values: [u64; 4]) -> DiffExecutionView {
        crate::view::diff_execution_view_from_context(ExecutionContextOutput {
            per_hart: BTreeMap::from([
                (
                    0,
                    HartExecutionContextOutput {
                        hart_id: 0,
                        instructions: Vec::new(),
                    },
                ),
                (
                    1,
                    HartExecutionContextOutput {
                        hart_id: 1,
                        instructions: Vec::new(),
                    },
                ),
            ]),
            framework_per_hart: BTreeMap::new(),
            logical_view: vec![
                logical_instruction(0, 0, 0, addi(16, 0, 1), vec![reg_write("x16", values[0])]),
                logical_instruction(0, 1, 1, addi(17, 0, 2), vec![reg_write("x17", values[1])]),
                logical_instruction(1, 2, 0, addi(18, 0, 3), vec![reg_write("x18", values[2])]),
                logical_instruction(1, 3, 1, addi(19, 0, 4), vec![reg_write("x19", values[3])]),
            ],
            mem_range,
            core_id: core_id.to_string(),
            isa_base: ISABase::Rv32,
            perf: PerfMetrics::default(),
        })
    }

    #[test]
    fn detect_unique_writes_uses_hart_local_test_layout() {
        let testcase = testcase_with_two_harts_init_and_test();
        let impl_order = vec!["spike".to_string(), "xiangshan".to_string()];
        let outputs = HashMap::from([
            (
                "spike".to_string(),
                output("spike", testcase.mem_range["spike"], [1, 2, 3, 4]),
            ),
            (
                "xiangshan".to_string(),
                output("xiangshan", testcase.mem_range["xiangshan"], [1, 99, 3, 4]),
            ),
        ]);

        let diffs = detect_unique_writes(&impl_order, &testcase, &outputs)
            .expect("write detection should succeed");

        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].test_index, 0);
        assert_eq!(diffs[0].global_index, 1);
        assert_eq!(
            diffs[0].register_changes.get("xiangshan"),
            Some(&vec![("x17".to_string(), 99)])
        );
    }

    #[test]
    fn detect_unique_writes_does_not_read_hart1_init_as_hart0_test() {
        let testcase = testcase_with_two_harts_init_and_test();
        let impl_order = vec!["spike".to_string(), "xiangshan".to_string()];
        let outputs = HashMap::from([
            (
                "spike".to_string(),
                output("spike", testcase.mem_range["spike"], [1, 2, 3, 4]),
            ),
            (
                "xiangshan".to_string(),
                output("xiangshan", testcase.mem_range["xiangshan"], [1, 2, 99, 4]),
            ),
        ]);

        let diffs = detect_unique_writes(&impl_order, &testcase, &outputs)
            .expect("write detection should succeed");

        assert!(diffs.is_empty());
    }
}
