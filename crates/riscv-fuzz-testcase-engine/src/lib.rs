use riscv_fuzz_plugin_api::ISABase;
use riscv_fuzz_plugin_api::metadata::FloatExtension;
use riscv_instruction_types::RandomGenerationError;
use thiserror::Error;

pub mod config;
pub mod cov_guided;
pub mod execution_artifacts;
pub mod impls;
pub mod instruction;
mod instruction_layout;
pub mod rvv_artifact_settings;
pub mod rvv_catalog_legal;
pub mod rvv_compositional_flow;
pub mod rvv_constraints;
pub(crate) mod rvv_legal;
pub mod rvv_model;
pub mod rvv_model_program;
pub(crate) mod rvv_negative;
pub mod rvv_paper_graph;
pub mod rvv_program_legal;
pub mod rvv_semantic_flow;
mod rvv_spike_guard_model;
pub mod rvv_stateflow;
pub mod sequence_filter;
pub mod sequence_plan;
mod smart_spice;
pub mod sync_report;
pub mod testcase;

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("need at least {required} temporary registers, but only {available} available")]
    InsufficientTempRegisters { required: usize, available: usize },

    #[error("memory range not found for RISC-V implementation: {core_id}")]
    MemRangeNotFound { core_id: String },

    #[error("memory range configuration failed: {source}")]
    MemRange {
        #[from]
        #[source]
        source: MemRangeError,
    },

    #[error("failed to generate random instruction sequences: {source}")]
    RandomSequence {
        #[from]
        #[source]
        source: RandomGenerationError,
    },

    #[error(
        "memory access offset range [{min}, {max}] is incompatible with mem_size {mem_size} and max access width {width}"
    )]
    InvalidMemAccessOffset {
        min: i64,
        max: i64,
        mem_size: u64,
        width: u64,
    },

    #[error("memory start address {addr:#x} exceeds supported range")]
    MemoryAddressOutOfRange { addr: u64 },

    #[error("memory range map is empty")]
    EmptyMemRange,

    #[error(
        "multicore memory window size overflow for mem_size {mem_size} and hart_count {hart_count}"
    )]
    MulticoreMemoryWindowOverflow { mem_size: u64, hart_count: u32 },

    #[error("extension instruction scaling min {min} exceeds max {max}")]
    InvalidExtensionScaling { min: usize, max: usize },

    #[error("multicore sync config field `{field}` must be in [0, 1], got {value}")]
    InvalidMulticoreSyncRatio { field: &'static str, value: f64 },

    #[error(
        "multicore sync config range `{field}` must satisfy 1 <= min <= max, got [{min}, {max}]"
    )]
    InvalidMulticoreSyncBlockRange {
        field: &'static str,
        min: usize,
        max: usize,
    },

    #[error(
        "requested double-precision extension for {isa_base:?} but intersection does not include D"
    )]
    RequestedDoubleUnsupported { isa_base: ISABase },

    #[error(
        "requested float extension {choice:?} for {isa_base:?} but not supported by selected implementations"
    )]
    RequestedFloatUnsupported {
        isa_base: ISABase,
        choice: FloatExtension,
    },

    #[error(
        "requested F-only for {isa_base:?} but no F-only binary available for at least one implementation"
    )]
    RequestedFloatToggleUnsupported { isa_base: ISABase },

    #[error(
        "multicore instruction generation requested but {core_id} does not support multicore templates"
    )]
    UnsupportedMulticore { core_id: String },

    #[error("failed to build canonical structured testcase: {message}")]
    StructuredInstructionBuildFailed { message: String },
}

#[derive(Debug, Error)]
pub enum MemRangeError {
    #[error("no RISC-V implementations available to compute memory range")]
    NoImplementations,

    #[error("requested memory size {requested} exceeds minimum available memory size {available}")]
    MemorySizeExceedsAvailable { requested: u64, available: u64 },

    #[error("requested memory size {size} is not aligned to word size {word_size}")]
    MemoryNotWordAligned { size: u64, word_size: u64 },

    #[error(
        "requested memory size {size} is less than the maximum instruction access width {width}"
    )]
    MemorySizeTooSmall { size: u64, width: u64 },

    #[error("memory range size {actual} should equal mem_size {expected}")]
    MemoryRangeSizeMismatch { actual: u64, expected: u64 },
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::instruction::{
        generate_initialization_instructions_for_harts,
        generate_initialization_instructions_for_impls,
        generate_memory_context_restore_instructions,
    };
    use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
    use riscv_instruction::{RV64Extensions, RV64IInstructions, RV64Instruction, RiscvInstruction};
    use riscv_instruction_types::{MemoryAccessConfig, RandomConfig, RegisterOperandConfig};

    fn sample_register_config() -> RegisterOperandConfig {
        RegisterOperandConfig {
            integer_register_number_range: (13, 20),
            floating_point_register_number_range: (13, 20),
            vector_register_number_range: (16, 15),
        }
    }

    #[test]
    fn rv64_f_only_initialization_uses_single_precision_moves() {
        let mut mem_range = HashMap::new();
        mem_range.insert("spike".to_string(), (0x8000_2000, 0x8000_203f));

        let blocks = generate_initialization_instructions_for_impls(
            &["spike".to_string()],
            &IsaExtensions::rv64(vec![RV64Extensions::I, RV64Extensions::F]),
            ISABase::Rv64,
            &mem_range,
            &RandomConfig {
                register_operand_config: sample_register_config(),
                memory_access_config: MemoryAccessConfig {
                    memory_base_address_range: MemoryAccessConfig::default()
                        .memory_base_address_range,
                    memory_offset_immediate_range: MemoryAccessConfig::default()
                        .memory_offset_immediate_range,
                    memory_base_register_number_range: (8, 10),
                },
                ..RandomConfig::default()
            },
        )
        .expect("RV64 F-only init generation should succeed");

        let block = blocks.get("spike").expect("spike block present");
        let lines = block.rendered_lines();
        assert!(lines.iter().any(|line| line.contains("fmv.w.x")));
        assert!(!lines.iter().any(|line| line.contains("fmv.d.x")));
    }

    #[test]
    fn rv64_fd_initialization_keeps_double_precision_moves() {
        let mut mem_range = HashMap::new();
        mem_range.insert("spike".to_string(), (0x8000_2000, 0x8000_203f));

        let blocks = generate_initialization_instructions_for_impls(
            &["spike".to_string()],
            &IsaExtensions::rv64(vec![
                RV64Extensions::I,
                RV64Extensions::F,
                RV64Extensions::D,
            ]),
            ISABase::Rv64,
            &mem_range,
            &RandomConfig {
                register_operand_config: sample_register_config(),
                memory_access_config: MemoryAccessConfig {
                    memory_base_address_range: MemoryAccessConfig::default()
                        .memory_base_address_range,
                    memory_offset_immediate_range: MemoryAccessConfig::default()
                        .memory_offset_immediate_range,
                    memory_base_register_number_range: (8, 10),
                },
                ..RandomConfig::default()
            },
        )
        .expect("RV64 FD init generation should succeed");

        let block = blocks.get("spike").expect("spike block present");
        assert!(
            block
                .rendered_lines()
                .iter()
                .any(|line| line.contains("fmv.d.x"))
        );
    }

    #[test]
    fn dual_hart_initialization_randomizes_registers_on_both_harts() {
        let block = generate_initialization_instructions_for_harts(
            &IsaExtensions::rv64(vec![RV64Extensions::I]),
            ISABase::Rv64,
            16,
            &RandomConfig {
                register_operand_config: RegisterOperandConfig {
                    integer_register_number_range: (13, 14),
                    floating_point_register_number_range: (13, 12),
                    vector_register_number_range: (16, 15),
                },
                memory_access_config: MemoryAccessConfig {
                    memory_base_address_range: MemoryAccessConfig::default()
                        .memory_base_address_range,
                    memory_offset_immediate_range: MemoryAccessConfig::default()
                        .memory_offset_immediate_range,
                    memory_base_register_number_range: (8, 10),
                },
                ..RandomConfig::default()
            },
            2,
        )
        .expect("dual-hart init generation should succeed");

        let per_hart = block.rendered_lines_per_hart();
        let hart0 = per_hart
            .iter()
            .find(|(hart_id, _)| *hart_id == 0)
            .map(|(_, lines)| lines)
            .expect("hart0 init should exist");
        let hart1 = per_hart
            .iter()
            .find(|(hart_id, _)| *hart_id == 1)
            .map(|(_, lines)| lines)
            .expect("hart1 init should exist");

        for lines in [hart0, hart1] {
            assert!(lines.iter().any(|line| line.starts_with("li x13,")));
            assert!(lines.iter().any(|line| line.starts_with("li x14,")));
            assert!(lines.iter().any(|line| line == "li x8, 0"));
            assert!(lines.iter().any(|line| line == "li x9, 0"));
            assert!(lines.iter().any(|line| line == "li x10, 0"));
        }

        assert!(
            hart0.iter().any(|line| line.starts_with("sd x10,")),
            "shared memory init should be emitted on hart0"
        );
        assert!(
            !hart1.iter().any(|line| line.starts_with("sd x10,")),
            "shared memory init should not be duplicated on hart1"
        );
    }

    #[test]
    fn rv64_memory_context_restore_uses_rv64_typed_sb() {
        let memory_context = BTreeMap::from([(0_u64, 0x12_u8), (4_u64, 0x34_u8)]);

        let instructions = generate_memory_context_restore_instructions(
            &memory_context,
            0x8000_0000,
            ISABase::Rv64,
            (8, 9),
        )
        .expect("RV64 memory restore should succeed");

        assert_eq!(instructions.len(), 6);
        assert!(matches!(
            instructions[2],
            RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::SB(_)))
        ));
        assert!(matches!(
            instructions[5],
            RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::SB(_)))
        ));
    }
}
