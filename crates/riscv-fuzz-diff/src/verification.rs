//! 初始化阶段的校验逻辑。
//! Initialization-stage verification.
//!
//! 在分析 testcase 指令之前，归约器要求所有实现先对初始化前缀的效果达成一致。
//! Before reasoning about testcase instructions, the reducer requires every implementation to
//! agree on the effects of the initialization prefix.
//! 如果初始化阶段已经分叉，后续写差异/异常差异分析就不再可信，因此必须立即失败。
//! If init diverges, later write/exception analysis is not trustworthy and the run fails
//! immediately.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;

use super::normalize::{
    MemoryWriteSet, PhaseInstructionLocation, RegisterWriteSet, init_instruction_locations,
    logical_instruction_at_location, logical_instruction_effects_for_diff,
};
use super::view::DiffExecutionView;

#[derive(Debug, Clone)]
pub(crate) struct InitializationSummary {
    pub(crate) init_instruction_count: usize,
}

#[derive(Debug)]
pub(crate) struct InitializationMismatchData {
    pub(crate) instruction_index: usize,
    pub(crate) reason: String,
    pub(crate) per_impl_registers: BTreeMap<String, RegisterWriteSet>,
    pub(crate) per_impl_memory: BTreeMap<String, MemoryWriteSet>,
}

#[derive(Clone)]
struct InitEffect {
    registers: RegisterWriteSet,
    memory: MemoryWriteSet,
}

type InitEffectMaps = (
    BTreeMap<String, RegisterWriteSet>,
    BTreeMap<String, MemoryWriteSet>,
);

fn empty_init_effect() -> InitEffect {
    InitEffect {
        registers: Vec::new(),
        memory: Vec::new(),
    }
}

fn present_init_effects_are_empty(
    per_impl_registers: &BTreeMap<String, RegisterWriteSet>,
    per_impl_memory: &BTreeMap<String, MemoryWriteSet>,
) -> bool {
    !per_impl_registers.is_empty()
        && per_impl_registers.values().all(Vec::is_empty)
        && per_impl_memory.values().all(Vec::is_empty)
}

fn collect_init_effect_maps_at(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
    location: PhaseInstructionLocation,
) -> InitEffectMaps {
    let mut per_impl_registers = BTreeMap::new();
    let mut per_impl_memory = BTreeMap::new();

    for impl_ref in impl_order {
        if let Some(instruction) = outputs
            .get(impl_ref)
            .and_then(|ctx| logical_instruction_at_location(ctx, location))
        {
            let (registers, memory) =
                logical_instruction_effects_for_diff(instruction, &testcase.config);
            per_impl_registers.insert(impl_ref.clone(), registers);
            per_impl_memory.insert(impl_ref.clone(), memory);
        }
    }

    (per_impl_registers, per_impl_memory)
}

#[derive(Clone)]
pub(crate) struct InitializationBugMarker {
    pub(crate) tag: &'static str,
    pub(crate) fingerprint: String,
}

pub(crate) fn classify_initialization_mismatch(
    mismatch: &InitializationMismatchData,
) -> InitializationBugMarker {
    // 指纹需要跨运行保持稳定，即便消息里带有具体寄存器编号之类的数值噪声。
    // Fingerprints should stay stable across run-to-run numeric noise such as concrete register
    // indices in messages.
    // 同时它又要能够区分实质上不同的 mismatch 形态。
    // At the same time, they still need to distinguish materially different mismatch shapes.
    let tag = classify_initialization_reason_tag(&mismatch.reason);
    let mut signature = String::new();
    signature.push_str("tag=");
    signature.push_str(tag);
    signature.push('|');
    signature.push_str("reason=");
    signature.push_str(&normalize_text_for_fingerprint(&mismatch.reason));
    signature.push('|');

    let mut impls = BTreeSet::new();
    impls.extend(mismatch.per_impl_registers.keys().cloned());
    impls.extend(mismatch.per_impl_memory.keys().cloned());

    for impl_ref in impls {
        signature.push_str("impl=");
        signature.push_str(&impl_ref.to_string());
        signature.push(';');

        let regs = mismatch.per_impl_registers.get(&impl_ref).expect(
            "initialization mismatch fingerprint missing register entry for implementation",
        );
        let mut reg_names = regs
            .iter()
            .map(|(name, _)| normalize_text_for_fingerprint(name))
            .collect::<Vec<_>>();
        reg_names.sort();
        reg_names.dedup();
        signature.push_str("reg_names=");
        signature.push_str(&reg_names.join(","));
        signature.push(';');
        signature.push_str("reg_count=");
        signature.push_str(&regs.len().to_string());
        signature.push(';');

        let mem_count = mismatch
            .per_impl_memory
            .get(&impl_ref)
            .expect("initialization mismatch fingerprint missing memory entry for implementation")
            .len();
        signature.push_str("mem_count=");
        signature.push_str(&mem_count.to_string());
        signature.push('|');
    }

    let hash = stable_fnv1a_64(&signature);
    InitializationBugMarker {
        tag,
        fingerprint: format!("{tag}-{hash:016x}"),
    }
}

fn classify_initialization_reason_tag(reason: &str) -> &'static str {
    if reason.contains("寄存器与内存写入不一致") {
        "INIT_REG_MEM_OVERLAP_MISMATCH"
    } else if reason.contains("寄存器写入不一致") {
        "INIT_REG_OVERLAP_MISMATCH"
    } else if reason.contains("内存写入不一致") {
        "INIT_MEM_OVERLAP_MISMATCH"
    } else if reason.contains("初始化执行数据长度不足") {
        "INIT_TRACE_LENGTH_SHORT"
    } else if reason.contains("缺少") && reason.contains("执行输出") {
        "INIT_MISSING_EXECUTION_OUTPUT"
    } else if reason.contains("缺少") && reason.contains("初始化指令") {
        "INIT_MISSING_INIT_INSTRUCTIONS"
    } else {
        "INIT_OTHER_MISMATCH"
    }
}

fn normalize_text_for_fingerprint(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_digit = false;
    for ch in input.chars() {
        if ch.is_ascii_digit() {
            if !in_digit {
                out.push('#');
                in_digit = true;
            }
        } else {
            in_digit = false;
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

fn stable_fnv1a_64(input: &str) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET_BASIS;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

pub(crate) fn verify_initialization(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
) -> Result<InitializationSummary, InitializationMismatchData> {
    if impl_order.is_empty() {
        return Err(InitializationMismatchData {
            instruction_index: 0,
            reason: "没有可用于初始化验证的 RISC-V 实现".to_string(),
            per_impl_registers: BTreeMap::new(),
            per_impl_memory: BTreeMap::new(),
        });
    }

    let init_locations = init_instruction_locations(impl_order, testcase).map_err(|err| {
        InitializationMismatchData {
            instruction_index: 0,
            reason: err.to_string(),
            per_impl_registers: BTreeMap::new(),
            per_impl_memory: BTreeMap::new(),
        }
    })?;
    let init_len = init_locations.len();
    let mut effects_per_impl: Vec<(String, Vec<InitEffect>)> = Vec::new();

    for impl_ref in impl_order {
        let rendered_len = testcase
            .init_rendered_line_count_of(impl_ref)
            .ok_or_else(|| InitializationMismatchData {
                instruction_index: 0,
                reason: format!("缺少 {} 的初始化指令", impl_ref),
                per_impl_registers: BTreeMap::new(),
                per_impl_memory: BTreeMap::new(),
            })?;
        if rendered_len != init_len {
            return Err(InitializationMismatchData {
                instruction_index: 0,
                reason: format!(
                    "{} 的初始化指令数量为 {}，但期望 {}",
                    impl_ref, rendered_len, init_len
                ),
                per_impl_registers: BTreeMap::new(),
                per_impl_memory: BTreeMap::new(),
            });
        }

        let ctx = outputs
            .get(impl_ref)
            .ok_or_else(|| InitializationMismatchData {
                instruction_index: 0,
                reason: format!("缺少 {} 的执行输出", impl_ref),
                per_impl_registers: BTreeMap::new(),
                per_impl_memory: BTreeMap::new(),
            })?;

        let mut effects = Vec::with_capacity(init_len);
        for location in &init_locations {
            let Some(instruction) = logical_instruction_at_location(ctx, *location) else {
                let (per_impl_registers, per_impl_memory) =
                    collect_init_effect_maps_at(impl_order, testcase, outputs, *location);
                if present_init_effects_are_empty(&per_impl_registers, &per_impl_memory) {
                    effects.push(empty_init_effect());
                    continue;
                }
                return Err(InitializationMismatchData {
                    instruction_index: location.phase_index,
                    reason: format!(
                        "实现 {} 缺少初始化指令 {} 的执行数据：hart {} local {}",
                        impl_ref,
                        location.phase_index,
                        location.hart_id,
                        location.hart_local_instruction_index
                    ),
                    per_impl_registers,
                    per_impl_memory,
                });
            };
            if instruction.asm_instruction_indices.is_empty() {
                let (per_impl_registers, per_impl_memory) =
                    collect_init_effect_maps_at(impl_order, testcase, outputs, *location);
                if present_init_effects_are_empty(&per_impl_registers, &per_impl_memory) {
                    effects.push(empty_init_effect());
                    continue;
                }
                return Err(InitializationMismatchData {
                    instruction_index: location.phase_index,
                    reason: format!(
                        "实现 {} 缺少初始化指令 {} 的执行数据：hart {} local {}",
                        impl_ref,
                        location.phase_index,
                        location.hart_id,
                        location.hart_local_instruction_index
                    ),
                    per_impl_registers,
                    per_impl_memory,
                });
            }
            let (registers, memory) =
                logical_instruction_effects_for_diff(instruction, &testcase.config);
            effects.push(InitEffect { registers, memory });
        }

        effects_per_impl.push((impl_ref.clone(), effects));
    }

    for idx in 0..init_len {
        let (reference_impl, reference_effects) = effects_per_impl
            .first()
            .expect("effects_per_impl is non-empty; qed");
        let reference = &reference_effects[idx];

        // 这里用第一个实现作为稳定参考进行比较，
        // Compare against the first implementation as a stable reference,
        // 但错误里仍然携带所有实现的写入映射，以保证报告是对称且可审计的。
        // but include the full per-implementation write maps in the error so reports remain
        // symmetric and auditable.
        for (impl_item, item_effects) in effects_per_impl.iter().skip(1) {
            let effect = &item_effects[idx];
            let registers_diff = effect.registers != reference.registers;
            let memory_diff = effect.memory != reference.memory;
            if !registers_diff && !memory_diff {
                continue;
            }

            let reason = match (registers_diff, memory_diff) {
                (true, true) => format!(
                    "实现 {} 的寄存器与内存写入不一致（参考实现 {}）",
                    impl_item, reference_impl
                ),
                (true, false) => {
                    format!(
                        "实现 {} 的寄存器写入不一致（参考实现 {}）",
                        impl_item, reference_impl
                    )
                }
                (false, true) => {
                    format!(
                        "实现 {} 的内存写入不一致（参考实现 {}）",
                        impl_item, reference_impl
                    )
                }
                (false, false) => unreachable!(),
            };

            let mut per_impl_registers = BTreeMap::new();
            let mut per_impl_memory = BTreeMap::new();
            for (core_id, impl_effects) in &effects_per_impl {
                let impl_effect = &impl_effects[idx];
                per_impl_registers.insert(core_id.clone(), impl_effect.registers.clone());
                per_impl_memory.insert(core_id.clone(), impl_effect.memory.clone());
            }

            return Err(InitializationMismatchData {
                instruction_index: idx,
                reason,
                per_impl_registers,
                per_impl_memory,
            });
        }
    }

    Ok(InitializationSummary {
        init_instruction_count: init_len,
    })
}

pub(crate) fn verify_initialization_at_location(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, DiffExecutionView>,
    location: PhaseInstructionLocation,
) -> Result<(), InitializationMismatchData> {
    if impl_order.is_empty() {
        return Err(InitializationMismatchData {
            instruction_index: location.phase_index,
            reason: "没有可用于初始化验证的 RISC-V 实现".to_string(),
            per_impl_registers: BTreeMap::new(),
            per_impl_memory: BTreeMap::new(),
        });
    }

    let mut effects_per_impl = Vec::with_capacity(impl_order.len());
    for impl_ref in impl_order {
        let ctx = outputs
            .get(impl_ref)
            .ok_or_else(|| InitializationMismatchData {
                instruction_index: location.phase_index,
                reason: format!("缺少 {} 的执行输出", impl_ref),
                per_impl_registers: BTreeMap::new(),
                per_impl_memory: BTreeMap::new(),
            })?;
        let Some(instruction) = logical_instruction_at_location(ctx, location) else {
            let (per_impl_registers, per_impl_memory) =
                collect_init_effect_maps_at(impl_order, testcase, outputs, location);
            if present_init_effects_are_empty(&per_impl_registers, &per_impl_memory) {
                effects_per_impl.push((impl_ref.clone(), empty_init_effect()));
                continue;
            }
            return Err(InitializationMismatchData {
                instruction_index: location.phase_index,
                reason: format!(
                    "实现 {} 缺少初始化指令 {} 的执行数据：hart {} local {}",
                    impl_ref,
                    location.phase_index,
                    location.hart_id,
                    location.hart_local_instruction_index
                ),
                per_impl_registers,
                per_impl_memory,
            });
        };
        if instruction.asm_instruction_indices.is_empty() {
            let (per_impl_registers, per_impl_memory) =
                collect_init_effect_maps_at(impl_order, testcase, outputs, location);
            if present_init_effects_are_empty(&per_impl_registers, &per_impl_memory) {
                effects_per_impl.push((impl_ref.clone(), empty_init_effect()));
                continue;
            }
            return Err(InitializationMismatchData {
                instruction_index: location.phase_index,
                reason: format!(
                    "实现 {} 缺少初始化指令 {} 的执行数据：hart {} local {}",
                    impl_ref,
                    location.phase_index,
                    location.hart_id,
                    location.hart_local_instruction_index
                ),
                per_impl_registers,
                per_impl_memory,
            });
        }
        let (registers, memory) =
            logical_instruction_effects_for_diff(instruction, &testcase.config);
        effects_per_impl.push((impl_ref.clone(), InitEffect { registers, memory }));
    }

    let (reference_impl, reference_effect) = effects_per_impl
        .first()
        .expect("effects_per_impl is non-empty; qed");
    for (impl_ref, effect) in effects_per_impl.iter().skip(1) {
        let registers_diff = effect.registers != reference_effect.registers;
        let memory_diff = effect.memory != reference_effect.memory;
        if !registers_diff && !memory_diff {
            continue;
        }

        let reason = match (registers_diff, memory_diff) {
            (true, true) => format!(
                "实现 {} 的寄存器与内存写入不一致（参考实现 {}）",
                impl_ref, reference_impl
            ),
            (true, false) => format!(
                "实现 {} 的寄存器写入不一致（参考实现 {}）",
                impl_ref, reference_impl
            ),
            (false, true) => format!(
                "实现 {} 的内存写入不一致（参考实现 {}）",
                impl_ref, reference_impl
            ),
            (false, false) => unreachable!(),
        };

        let mut per_impl_registers = BTreeMap::new();
        let mut per_impl_memory = BTreeMap::new();
        for (core_id, impl_effect) in &effects_per_impl {
            per_impl_registers.insert(core_id.clone(), impl_effect.registers.clone());
            per_impl_memory.insert(core_id.clone(), impl_effect.memory.clone());
        }

        return Err(InitializationMismatchData {
            instruction_index: location.phase_index,
            reason,
            per_impl_registers,
            per_impl_memory,
        });
    }

    Ok(())
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

    fn minimal_testcase() -> GeneratedTestCase {
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
                hart_config: HartConfig::Single,
            },
            harts: Vec::new(),
            mem_range: HashMap::new(),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        }
    }

    fn testcase_with_one_init_instruction() -> GeneratedTestCase {
        let mut testcase = minimal_testcase();
        testcase.harts = vec![HartProgram {
            hart_id: 0,
            init: vec![InstructionSequence::new(addi(16, 0, 1))],
            test: Vec::new(),
            shared_init_sequence_indices: BTreeSet::new(),
            shared_test_sequence_indices: BTreeSet::new(),
        }];
        testcase.mem_range = HashMap::from([
            ("spike".to_string(), (0x1_0000, 0x1_003f)),
            ("xiangshan".to_string(), (0x2_0000, 0x2_003f)),
        ]);
        testcase
    }

    fn testcase_with_one_temp_init_instruction() -> GeneratedTestCase {
        let mut testcase = minimal_testcase();
        testcase.harts = vec![HartProgram {
            hart_id: 0,
            init: vec![InstructionSequence::new(addi(8, 0, 1))],
            test: Vec::new(),
            shared_init_sequence_indices: BTreeSet::new(),
            shared_test_sequence_indices: BTreeSet::new(),
        }];
        testcase.mem_range = HashMap::from([
            ("spike".to_string(), (0x1_0000, 0x1_003f)),
            ("xiangshan".to_string(), (0x2_0000, 0x2_003f)),
        ]);
        testcase
    }

    fn testcase_with_two_harts_and_hart0_test_between_init_ranges() -> GeneratedTestCase {
        let mut testcase = minimal_testcase();
        testcase.config.hart_config = HartConfig::Dual {
            sync: Default::default(),
        };
        testcase.harts = vec![
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
                test: Vec::new(),
                shared_init_sequence_indices: BTreeSet::new(),
                shared_test_sequence_indices: BTreeSet::new(),
            },
        ];
        testcase.mem_range = HashMap::from([
            ("spike".to_string(), (0x1_0000, 0x1_003f)),
            ("xiangshan".to_string(), (0x2_0000, 0x2_003f)),
        ]);
        testcase
    }

    fn logical_instruction(
        hart_id: u32,
        instruction_index: usize,
        local_instruction_index: usize,
        instruction: RiscvInstruction,
        asm_instruction_indices: Vec<usize>,
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
            asm_instruction_indices,
            exceptions: Vec::new(),
            register_writes,
            memory_writes: Vec::new(),
            timing: Default::default(),
            context: InstructionContext::default(),
        }
    }

    fn logical_init_instruction(
        asm_instruction_indices: Vec<usize>,
        register_writes: Vec<RegisterValue>,
    ) -> LogicalInstructionContextOutput {
        logical_instruction(
            0,
            0,
            0,
            addi(16, 0, 1),
            asm_instruction_indices,
            register_writes,
        )
    }

    fn reg_write(name: &str, value: u64) -> RegisterValue {
        RegisterValue {
            name: name.to_string(),
            value,
        }
    }

    fn execution_output(
        core_id: &str,
        mem_range: (u64, u64),
        logical_view: Vec<LogicalInstructionContextOutput>,
    ) -> DiffExecutionView {
        let mut per_hart = BTreeMap::new();
        for instruction in &logical_view {
            per_hart
                .entry(instruction.hart_id)
                .or_insert_with(|| HartExecutionContextOutput {
                    hart_id: instruction.hart_id,
                    instructions: Vec::new(),
                });
        }

        crate::view::diff_execution_view_from_context(ExecutionContextOutput {
            per_hart,
            framework_per_hart: BTreeMap::new(),
            logical_view,
            mem_range,
            core_id: core_id.to_string(),
            isa_base: ISABase::Rv32,
            perf: PerfMetrics::default(),
        })
    }

    #[test]
    fn verify_initialization_rejects_empty_impl_order() {
        let err = verify_initialization(&[], &minimal_testcase(), &HashMap::new())
            .expect_err("empty implementation order should fail");
        assert!(err.reason.contains("没有可用于初始化验证的 RISC-V 实现"));
    }

    #[test]
    fn verify_initialization_reports_missing_streamed_instruction_not_empty_write() {
        let testcase = testcase_with_one_init_instruction();
        let impl_order = vec!["spike".to_string(), "xiangshan".to_string()];
        let outputs = HashMap::from([
            (
                "spike".to_string(),
                execution_output(
                    "spike",
                    testcase.mem_range["spike"],
                    vec![logical_init_instruction(
                        vec![0],
                        vec![RegisterValue {
                            name: "x16".to_string(),
                            value: 1,
                        }],
                    )],
                ),
            ),
            (
                "xiangshan".to_string(),
                execution_output(
                    "xiangshan",
                    testcase.mem_range["xiangshan"],
                    vec![logical_init_instruction(Vec::new(), Vec::new())],
                ),
            ),
        ]);

        let err = verify_initialization(&impl_order, &testcase, &outputs)
            .expect_err("missing streamed instruction should fail as missing execution data");

        assert_eq!(err.instruction_index, 0);
        assert!(err.reason.contains("缺少初始化指令"));
        assert!(err.reason.contains("执行数据"));
        assert_eq!(
            classify_initialization_mismatch(&err).tag,
            "INIT_MISSING_INIT_INSTRUCTIONS"
        );
        assert_eq!(
            err.per_impl_registers.get("spike"),
            Some(&vec![("x16".to_string(), 1)])
        );
        assert_eq!(err.per_impl_registers.get("xiangshan"), Some(&Vec::new()));
    }

    #[test]
    fn verify_initialization_allows_missing_trace_for_empty_visible_temp_setup() {
        let testcase = testcase_with_one_temp_init_instruction();
        let impl_order = vec!["spike".to_string(), "xiangshan".to_string()];
        let outputs = HashMap::from([
            (
                "spike".to_string(),
                execution_output("spike", testcase.mem_range["spike"], Vec::new()),
            ),
            (
                "xiangshan".to_string(),
                execution_output(
                    "xiangshan",
                    testcase.mem_range["xiangshan"],
                    vec![logical_instruction(
                        0,
                        0,
                        0,
                        addi(8, 0, 1),
                        vec![0],
                        vec![reg_write("x8", 1)],
                    )],
                ),
            ),
        ]);

        let summary = verify_initialization(&impl_order, &testcase, &outputs)
            .expect("missing trace for invisible temp setup should not block diffing");
        assert_eq!(summary.init_instruction_count, 1);

        verify_initialization_at_location(
            &impl_order,
            &testcase,
            &outputs,
            PhaseInstructionLocation {
                phase_index: 0,
                global_index: 0,
                hart_id: 0,
                hart_local_instruction_index: 0,
            },
        )
        .expect("streaming init check should also ignore invisible temp setup");
    }

    #[test]
    fn verify_initialization_uses_hart_local_init_locations_not_flat_combined_prefix() {
        let testcase = testcase_with_two_harts_and_hart0_test_between_init_ranges();
        let impl_order = vec!["spike".to_string(), "xiangshan".to_string()];

        let spike_output = execution_output(
            "spike",
            testcase.mem_range["spike"],
            vec![
                logical_instruction(0, 0, 0, addi(16, 0, 1), vec![0], vec![reg_write("x16", 1)]),
                // This is hart0 test. It sits at combined index 1, exactly where the old
                // init-prefix verifier mistakenly looked for hart1 init.
                logical_instruction(0, 1, 1, addi(17, 0, 2), vec![1], vec![reg_write("x17", 2)]),
                logical_instruction(1, 2, 0, addi(18, 0, 3), vec![0], vec![reg_write("x18", 3)]),
            ],
        );
        let xiangshan_output = execution_output(
            "xiangshan",
            testcase.mem_range["xiangshan"],
            vec![
                logical_instruction(0, 0, 0, addi(16, 0, 1), vec![0], vec![reg_write("x16", 1)]),
                logical_instruction(0, 1, 1, addi(17, 0, 2), vec![1], vec![reg_write("x17", 99)]),
                logical_instruction(1, 2, 0, addi(18, 0, 3), vec![0], vec![reg_write("x18", 3)]),
            ],
        );
        let outputs = HashMap::from([
            ("spike".to_string(), spike_output),
            ("xiangshan".to_string(), xiangshan_output),
        ]);

        let summary = verify_initialization(&impl_order, &testcase, &outputs)
            .expect("hart0 test mismatch must not be reported as an init mismatch");

        assert_eq!(summary.init_instruction_count, 2);
    }

    #[test]
    fn verify_initialization_reports_real_hart1_init_mismatch_by_phase_index() {
        let testcase = testcase_with_two_harts_and_hart0_test_between_init_ranges();
        let impl_order = vec!["spike".to_string(), "xiangshan".to_string()];

        let spike_output = execution_output(
            "spike",
            testcase.mem_range["spike"],
            vec![
                logical_instruction(0, 0, 0, addi(16, 0, 1), vec![0], vec![reg_write("x16", 1)]),
                logical_instruction(0, 1, 1, addi(17, 0, 2), vec![1], vec![reg_write("x17", 2)]),
                logical_instruction(1, 2, 0, addi(18, 0, 3), vec![0], vec![reg_write("x18", 3)]),
            ],
        );
        let xiangshan_output = execution_output(
            "xiangshan",
            testcase.mem_range["xiangshan"],
            vec![
                logical_instruction(0, 0, 0, addi(16, 0, 1), vec![0], vec![reg_write("x16", 1)]),
                logical_instruction(0, 1, 1, addi(17, 0, 2), vec![1], vec![reg_write("x17", 2)]),
                logical_instruction(1, 2, 0, addi(18, 0, 3), vec![0], vec![reg_write("x18", 4)]),
            ],
        );
        let outputs = HashMap::from([
            ("spike".to_string(), spike_output),
            ("xiangshan".to_string(), xiangshan_output),
        ]);

        let err = verify_initialization(&impl_order, &testcase, &outputs)
            .expect_err("real hart1 init mismatch should still fail initialization");

        assert_eq!(err.instruction_index, 1);
        assert_eq!(
            err.per_impl_registers.get("spike"),
            Some(&vec![("x18".to_string(), 3)])
        );
        assert_eq!(
            err.per_impl_registers.get("xiangshan"),
            Some(&vec![("x18".to_string(), 4)])
        );
    }
}
