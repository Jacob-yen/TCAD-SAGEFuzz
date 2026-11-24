//! diff-analysis 共享的规范化辅助函数。
//! Shared normalization helpers for diff-analysis.
//!
//! crate 其余部分只比较 traceable 的测试指令，以及 testcase 可见范围内的写入。
//! The rest of the crate compares only traceable testcase instructions and only writes inside the
//! testcase-visible scope.
//! 把这套策略收敛在这里，可以避免每个阶段各自发明一套“同一条指令”或“同一组写入”的定义。
//! Keeping that policy here avoids each stage inventing its own notion of "same instruction" or
//! "same write set".

use riscv_fuzz_plugin_api::{
    MemValue, RegisterValue, canonicalize_register_name, context::LogicalInstructionContextOutput,
    parse_register_name,
};
use riscv_fuzz_testcase_engine::testcase::{GeneratedTestCase, HartPhase, TestCaseConfig};
use riscv_instruction::RegisterBank;

use super::{types::DiffAnalysisError, view::DiffExecutionView};

pub(crate) type RegisterWriteSet = Vec<(String, u64)>;
pub(crate) type MemoryWriteSet = Vec<(u64, u8)>;
pub(crate) type WriteEffect = (RegisterWriteSet, MemoryWriteSet);

fn ensure_impl_order_non_empty(impl_order: &[String]) -> Result<(), DiffAnalysisError> {
    if impl_order.is_empty() {
        return Err(DiffAnalysisError::TestInstructionCountMismatch {
            details: "no RISC-V implementations available".to_string(),
        });
    }
    Ok(())
}

fn determine_phase_length(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    phase: HartPhase,
) -> Result<usize, DiffAnalysisError> {
    ensure_impl_order_non_empty(impl_order)?;
    Ok(match phase {
        HartPhase::Init => testcase.init_traceable_len(),
        HartPhase::Test => testcase.test_traceable_len(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PhaseInstructionLocation {
    pub(crate) phase_index: usize,
    pub(crate) global_index: usize,
    pub(crate) hart_id: u32,
    pub(crate) hart_local_instruction_index: usize,
}

pub(crate) fn phase_instruction_locations(
    testcase: &GeneratedTestCase,
    phase: HartPhase,
) -> Vec<PhaseInstructionLocation> {
    let mut phase_index = 0usize;
    let mut hart_base_index = 0usize;
    let mut locations = Vec::new();

    for hart in &testcase.harts {
        let init_len: usize = hart
            .init
            .iter()
            .map(|sequence| sequence.pre_instructions.len() + 1 + sequence.post_instructions.len())
            .sum();
        let test_len: usize = hart
            .test
            .iter()
            .map(|sequence| sequence.pre_instructions.len() + 1 + sequence.post_instructions.len())
            .sum();
        let hart_local_base = match phase {
            HartPhase::Init => 0,
            HartPhase::Test => init_len,
        };
        let mut hart_local_instruction_index = hart_local_base;
        for sequence in hart.sequences(phase) {
            let traceable_len =
                sequence.pre_instructions.len() + 1 + sequence.post_instructions.len();
            for _ in 0..traceable_len {
                locations.push(PhaseInstructionLocation {
                    phase_index,
                    global_index: hart_base_index + hart_local_instruction_index,
                    hart_id: hart.hart_id,
                    hart_local_instruction_index,
                });
                phase_index += 1;
                hart_local_instruction_index += 1;
            }
        }
        hart_base_index += init_len + test_len;
    }

    locations
}

pub(crate) fn phase_instruction_location_by_hart_local(
    testcase: &GeneratedTestCase,
    phase: HartPhase,
    hart_id: u32,
    hart_local_instruction_index: usize,
) -> Option<PhaseInstructionLocation> {
    phase_instruction_locations(testcase, phase)
        .into_iter()
        .find(|location| {
            location.hart_id == hart_id
                && location.hart_local_instruction_index == hart_local_instruction_index
        })
}

pub(crate) fn test_instruction_locations(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
) -> Result<Vec<PhaseInstructionLocation>, DiffAnalysisError> {
    ensure_impl_order_non_empty(impl_order)?;
    Ok(phase_instruction_locations(testcase, HartPhase::Test))
}

pub(crate) fn test_instruction_location(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    test_index: usize,
) -> Result<PhaseInstructionLocation, DiffAnalysisError> {
    test_instruction_locations(impl_order, testcase)?
        .get(test_index)
        .copied()
        .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
            details: format!("missing traceable test instruction {test_index}"),
        })
}

pub(crate) fn init_instruction_locations(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
) -> Result<Vec<PhaseInstructionLocation>, DiffAnalysisError> {
    ensure_impl_order_non_empty(impl_order)?;
    Ok(phase_instruction_locations(testcase, HartPhase::Init))
}

pub(crate) fn logical_instruction_at_location(
    output: &DiffExecutionView,
    location: PhaseInstructionLocation,
) -> Option<&LogicalInstructionContextOutput> {
    output
        .logical_instruction_by_hart_local(location.hart_id, location.hart_local_instruction_index)
}

pub(crate) fn determine_test_length(
    impl_order: &[String],
    testcase: &GeneratedTestCase,
) -> Result<usize, DiffAnalysisError> {
    // traceable 指令是 testcase 生成、执行报告和后续定位之间共享的索引契约。
    // Traceable instructions are the shared indexing contract between testcase generation,
    // execution reports, and later localization.
    // 指令块里的指令标记行和注释行不算作可执行测试步骤。
    // Directive/comment lines inside the block do not count as executable test steps.
    determine_phase_length(impl_order, testcase, HartPhase::Test)
}

fn canonicalize_register_changes(
    changes: &[RegisterValue],
    mut include: impl FnMut(&RegisterValue) -> bool,
) -> Vec<(String, u64)> {
    let mut entries = changes
        .iter()
        .filter(|entry| include(entry))
        .filter_map(|entry| canonicalize_register_name(&entry.name).map(|name| (name, entry.value)))
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

pub(crate) fn canonicalize_register_changes_for_diff(
    changes: &[RegisterValue],
    config: &TestCaseConfig,
) -> Vec<(String, u64)> {
    // 忽略 fuzz 寄存器范围之外的写入；这些写入可能来自 wrapper 或初始化样板代码。
    // Ignore writes outside the fuzzed register scope; those writes may come from wrapper/setup
    // code.
    // 它们不应该主导 testcase 可见的差分行为。
    // They should not dominate testcase-visible differential behavior.
    canonicalize_register_changes(changes, |entry| register_in_test_scope(&entry.name, config))
}

pub(crate) fn canonicalize_register_changes_unfiltered(
    changes: &[RegisterValue],
) -> Vec<(String, u64)> {
    canonicalize_register_changes(changes, |_| true)
}

fn register_in_test_scope(name: &str, config: &TestCaseConfig) -> bool {
    let Some(parsed) = parse_register_name(name) else {
        return false;
    };

    let register_config = &config.random_config.register_operand_config;
    match parsed.bank {
        RegisterBank::Integer => {
            let (min, max) = register_config.integer_register_number_range;
            parsed.index >= min && parsed.index <= max
        }
        RegisterBank::FloatingPoint => {
            let (min, max) = register_config.floating_point_register_number_range;
            parsed.index >= min && parsed.index <= max
        }
        RegisterBank::Vector => {
            let (min, max) = register_config.vector_register_number_range;
            parsed.index >= min && parsed.index <= max
        }
    }
}

pub(crate) fn canonicalize_memory_changes(changes: &[MemValue]) -> Vec<(u64, u8)> {
    // 对 diff-analysis 而言，日志记录顺序不具有语义意义，因此按地址排序做规范化。
    // Logging order is not semantically relevant for diff-analysis, so normalize by address.
    let mut entries = changes
        .iter()
        .map(|entry| (entry.addr, entry.value))
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

pub(crate) fn logical_instruction_effects_for_diff(
    instruction: &LogicalInstructionContextOutput,
    config: &TestCaseConfig,
) -> WriteEffect {
    (
        canonicalize_register_changes_for_diff(&instruction.register_writes, config),
        canonicalize_memory_changes(&instruction.memory_writes),
    )
}

pub(crate) fn logical_instruction_effects_unfiltered(
    instruction: &LogicalInstructionContextOutput,
) -> WriteEffect {
    (
        canonicalize_register_changes_unfiltered(&instruction.register_writes),
        canonicalize_memory_changes(&instruction.memory_writes),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
    use riscv_fuzz_testcase_engine::testcase::{GeneratedTestCase, HartConfig, TestCaseConfig};
    use riscv_instruction::RV32Extensions;
    use riscv_instruction_types::{MemoryAccessConfig, RandomConfig, RegisterOperandConfig};

    use super::*;

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

    #[test]
    fn determine_test_length_rejects_empty_impl_order() {
        let err = determine_test_length(&[], &minimal_testcase())
            .expect_err("empty implementation order should fail");
        assert!(format!("{err}").contains("no RISC-V implementations available"));
    }

    #[test]
    fn register_scope_accepts_abi_alias_names() {
        let testcase = minimal_testcase();
        assert!(register_in_test_scope("a7", &testcase.config));
        assert!(register_in_test_scope("fa7", &testcase.config));
        assert!(!register_in_test_scope("a0", &testcase.config));
    }

    #[test]
    fn canonicalize_register_changes_maps_aliases_to_physical_names() {
        let testcase = minimal_testcase();
        let changes = vec![
            RegisterValue {
                name: "a7".to_string(),
                value: 1,
            },
            RegisterValue {
                name: "fa7".to_string(),
                value: 2,
            },
        ];

        let canonical = canonicalize_register_changes_for_diff(&changes, &testcase.config);
        assert_eq!(
            canonical,
            vec![("f17".to_string(), 2), ("x17".to_string(), 1)]
        );
    }
}
