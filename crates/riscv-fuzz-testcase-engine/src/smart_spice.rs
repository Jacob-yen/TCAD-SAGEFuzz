use rand::Rng;
use rand::prelude::IndexedRandom;
use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
use riscv_instruction::{
    RV32_I_FENCE, RV32_I_LW, RV32_I_SW, RV32_M_DIV, RV32_M_MULH, RV32_M_MULHSU, RV32_M_MULHU,
    RV32_M_REM, RV32_Zca_C_ADDI, RV32_Zca_C_LI, RV32_Zca_C_MV, RV32_Zca_C_NOP, RV32_Zicsr_CSRRS,
    RV32Extensions, RV32IInstructions, RV32Instruction, RV32MInstructions, RV32ZcaInstructions,
    RV32ZicsrInstructions, RV64_I_FENCE, RV64_I_LW, RV64_I_SW, RV64_M_DIV, RV64_M_MULH,
    RV64_M_MULHSU, RV64_M_MULHU, RV64_M_REM, RV64_Zca_C_ADDI, RV64_Zca_C_LI, RV64_Zca_C_MV,
    RV64_Zca_C_NOP, RV64_Zicsr_CSRRS, RV64Extensions, RV64IInstructions, RV64Instruction,
    RV64MInstructions, RV64ZcaInstructions, RV64ZicsrInstructions,
    RestrictedImmediate6ForbiddenRange0To0, RestrictedIntegerRegisterXdForbiddenRange0To0,
    RestrictedIntegerRegisterXs2ForbiddenRange0To0, RiscvInstruction,
};
use riscv_instruction_types::{
    BaseAddressRegister, FenceMode, Immediate, InstructionSequence, IntegerRegister, IsaProfile,
    LoadImmediatePurpose, PseudoInstruction, RandomConfig, UImmediate, ValidatedValue, WritableCsr,
    XLenAddress, XLenImmediate,
};

/// A chunk of generated instructions for extension-aware rare patterns.
#[derive(Debug, Clone)]
pub(crate) enum SpiceChunk {
    /// Instruction sequence generated from riscv-instruction-crates.
    Seq(InstructionSequence<RiscvInstruction>),
    /// Flattened standard instructions built from riscv-instruction-crates types.
    Structured(Vec<RiscvInstruction>),
}

fn x_reg(reg: u8) -> IntegerRegister {
    IntegerRegister::new(reg).expect("valid integer register")
}

fn base_reg(reg: u8) -> BaseAddressRegister {
    BaseAddressRegister::new(reg).expect("valid base register")
}

fn restricted_rd(reg: u8) -> RestrictedIntegerRegisterXdForbiddenRange0To0 {
    RestrictedIntegerRegisterXdForbiddenRange0To0::new(reg).expect("valid non-zero rd")
}

fn restricted_rs2(reg: u8) -> RestrictedIntegerRegisterXs2ForbiddenRange0To0 {
    RestrictedIntegerRegisterXs2ForbiddenRange0To0::new(reg).expect("valid non-zero rs2")
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

fn li_value_pseudo(
    rd: u8,
    immediate: i64,
    purpose: LoadImmediatePurpose,
    isa_base: ISABase,
) -> RiscvInstruction {
    RiscvInstruction::from(PseudoInstruction::LoadImmediate {
        rd: x_reg(rd),
        immediate: XLenImmediate::from_i64_for(isa_profile(isa_base), immediate)
            .expect("load-immediate value must fit active xlen"),
        purpose,
    })
}

fn li_bits_pseudo(
    rd: u8,
    immediate: u64,
    purpose: LoadImmediatePurpose,
    isa_base: ISABase,
) -> RiscvInstruction {
    RiscvInstruction::from(PseudoInstruction::LoadImmediate {
        rd: x_reg(rd),
        immediate: XLenImmediate::from_unsigned_bits_for(isa_profile(isa_base), immediate),
        purpose,
    })
}

fn li_address_pseudo(
    rd: u8,
    address: u64,
    purpose: LoadImmediatePurpose,
    isa_base: ISABase,
) -> RiscvInstruction {
    let address = XLenAddress::from_u64_for(isa_profile(isa_base), address)
        .expect("load-immediate address must fit active xlen");
    RiscvInstruction::from(PseudoInstruction::LoadImmediate {
        rd: x_reg(rd),
        immediate: XLenImmediate::from_address(address),
        purpose,
    })
}

fn instruction_entry(instruction: RiscvInstruction) -> RiscvInstruction {
    instruction
}

fn rv32_lw(rd: u8, offset: i32, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::LW(RV32_I_LW {
        imm: Immediate::<12, true>::new(offset).expect("valid LW offset"),
        xs1: base_reg(rs1),
        xd: x_reg(rd),
    })))
}

fn rv64_lw(rd: u8, offset: i32, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::LW(RV64_I_LW {
        imm: Immediate::<12, true>::new(offset).expect("valid LW offset"),
        xs1: base_reg(rs1),
        xd: x_reg(rd),
    })))
}

fn rv32_sw(rs2: u8, offset: i32, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::SW(RV32_I_SW {
        imm: Immediate::<12, true>::new(offset).expect("valid SW offset"),
        xs2: x_reg(rs2),
        xs1: base_reg(rs1),
    })))
}

fn rv64_sw(rs2: u8, offset: i32, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::SW(RV64_I_SW {
        imm: Immediate::<12, true>::new(offset).expect("valid SW offset"),
        xs2: x_reg(rs2),
        xs1: base_reg(rs1),
    })))
}

fn rv32_fence() -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::FENCE(RV32_I_FENCE {
        fm: UImmediate::<4, false>::new(0b1000).expect("valid fence fm"),
        pred: FenceMode::new(0b1111).expect("valid fence pred"),
        succ: FenceMode::new(0b1111).expect("valid fence succ"),
        xs1: x_reg(0),
        xd: x_reg(0),
    })))
}

fn rv64_fence() -> RiscvInstruction {
    RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::FENCE(RV64_I_FENCE {
        fm: UImmediate::<4, false>::new(0b1000).expect("valid fence fm"),
        pred: FenceMode::new(0b1111).expect("valid fence pred"),
        succ: FenceMode::new(0b1111).expect("valid fence succ"),
        xs1: x_reg(0),
        xd: x_reg(0),
    })))
}

/// Generate a spice chunk that adapts to the implementation's supported extensions.
pub(crate) fn generate_smart_spice<R: Rng>(
    rng: &mut R,
    extensions: &IsaExtensions,
    isa_base: ISABase,
    random_config: &RandomConfig,
    mem_start: u64,
    mem_end: u64,
) -> Option<SpiceChunk> {
    if rng.random_bool(0.7) {
        generate_extension_based_spice(rng, extensions, isa_base, random_config)
    } else {
        Some(generate_universal_spice(rng, isa_base, mem_start, mem_end))
    }
}

fn generate_extension_based_spice<R: Rng>(
    rng: &mut R,
    extensions: &IsaExtensions,
    isa_base: ISABase,
    random_config: &RandomConfig,
) -> Option<SpiceChunk> {
    match isa_base {
        ISABase::Rv32 => {
            let rv32 = extensions.as_rv32()?;
            if rv32.is_empty() {
                return None;
            }

            if rng.random_bool(0.5)
                && let Some(chunk) = generate_targeted_rv32_spice(rng, rv32, random_config)
            {
                return Some(chunk);
            }

            let ext = rv32.choose(rng)?;
            let seq = ext.random_sequence_with_rng(rng, random_config).ok()?;
            Some(SpiceChunk::Seq(seq))
        }
        ISABase::Rv64 => {
            let rv64 = extensions.as_rv64()?;
            if rv64.is_empty() {
                return None;
            }

            if rng.random_bool(0.5)
                && let Some(chunk) = generate_targeted_rv64_spice(rng, rv64, random_config)
            {
                return Some(chunk);
            }

            let ext = rv64.choose(rng)?;
            let seq = ext.random_sequence_with_rng(rng, random_config).ok()?;
            Some(SpiceChunk::Seq(seq))
        }
    }
}

fn generate_targeted_rv32_spice<R: Rng>(
    rng: &mut R,
    extensions: &[RV32Extensions],
    _random_config: &RandomConfig,
) -> Option<SpiceChunk> {
    let mut pattern_choices = Vec::new();

    if extensions.contains(&RV32Extensions::M) {
        pattern_choices.push(0);
    }

    if extensions.contains(&RV32Extensions::C) {
        pattern_choices.push(1);
    }

    if extensions.contains(&RV32Extensions::Zicsr) {
        pattern_choices.push(2);
    }

    if pattern_choices.is_empty() {
        return None;
    }

    let choice = pattern_choices.choose(rng)?;
    Some(match choice {
        0 => generate_m_extension_corner_cases(),
        1 => generate_c_extension_patterns_rv32(),
        _ => generate_zicsr_patterns_rv32(),
    })
}

fn generate_targeted_rv64_spice<R: Rng>(
    rng: &mut R,
    extensions: &[RV64Extensions],
    _random_config: &RandomConfig,
) -> Option<SpiceChunk> {
    let mut pattern_choices = Vec::new();

    if extensions.contains(&RV64Extensions::M) {
        pattern_choices.push(0);
    }

    if extensions.contains(&RV64Extensions::C) {
        pattern_choices.push(1);
    }

    if extensions.contains(&RV64Extensions::Zicsr) {
        pattern_choices.push(2);
    }

    if pattern_choices.is_empty() {
        return None;
    }

    let choice = pattern_choices.choose(rng)?;
    Some(match choice {
        0 => generate_m_extension_corner_cases_rv64(),
        1 => generate_c_extension_patterns_rv64(),
        _ => generate_zicsr_patterns_rv64(),
    })
}

fn generate_universal_spice<R: Rng>(
    rng: &mut R,
    isa_base: ISABase,
    mem_start: u64,
    mem_end: u64,
) -> SpiceChunk {
    let choice = rng.random_range(0..5);

    let instructions = match choice {
        0 => {
            let misaligned_addr = mem_start.saturating_add(rng.random_range(1u64..=3u64));
            vec![
                li_address_pseudo(
                    8,
                    misaligned_addr,
                    LoadImmediatePurpose::BaseAddress,
                    isa_base,
                ),
                instruction_entry(match isa_base {
                    ISABase::Rv32 => rv32_lw(13, 0, 8),
                    ISABase::Rv64 => rv64_lw(13, 0, 8),
                }),
            ]
        }
        1 => {
            let misaligned_addr = mem_start.saturating_add(rng.random_range(1u64..=3u64));
            vec![
                li_address_pseudo(
                    8,
                    misaligned_addr,
                    LoadImmediatePurpose::BaseAddress,
                    isa_base,
                ),
                instruction_entry(match isa_base {
                    ISABase::Rv32 => rv32_sw(0, 0, 8),
                    ISABase::Rv64 => rv64_sw(0, 0, 8),
                }),
            ]
        }
        2 => vec![
            li_address_pseudo(8, mem_start, LoadImmediatePurpose::BaseAddress, isa_base),
            instruction_entry(match isa_base {
                ISABase::Rv32 => rv32_lw(13, 0, 8),
                ISABase::Rv64 => rv64_lw(13, 0, 8),
            }),
            instruction_entry(match isa_base {
                ISABase::Rv32 => rv32_sw(13, 4, 8),
                ISABase::Rv64 => rv64_sw(13, 4, 8),
            }),
        ],
        3 => {
            let aligned_end = mem_end.saturating_sub(3) & !3u64;
            vec![
                li_address_pseudo(8, aligned_end, LoadImmediatePurpose::BaseAddress, isa_base),
                instruction_entry(match isa_base {
                    ISABase::Rv32 => rv32_lw(13, 0, 8),
                    ISABase::Rv64 => rv64_lw(13, 0, 8),
                }),
            ]
        }
        _ => vec![instruction_entry(match isa_base {
            ISABase::Rv32 => rv32_fence(),
            ISABase::Rv64 => rv64_fence(),
        })],
    };

    SpiceChunk::Structured(instructions)
}

fn generate_m_extension_corner_cases() -> SpiceChunk {
    SpiceChunk::Structured(vec![
        li_bits_pseudo(
            13,
            0x8000_0000,
            LoadImmediatePurpose::Generic,
            ISABase::Rv32,
        ),
        li_value_pseudo(14, -1, LoadImmediatePurpose::Generic, ISABase::Rv32),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::M(
            RV32MInstructions::DIV(RV32_M_DIV {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(15),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::M(
            RV32MInstructions::DIV(RV32_M_DIV {
                xs2: x_reg(0),
                xs1: x_reg(13),
                xd: x_reg(9),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::M(
            RV32MInstructions::REM(RV32_M_REM {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(15),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::M(
            RV32MInstructions::MULH(RV32_M_MULH {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(15),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::M(
            RV32MInstructions::MULHU(RV32_M_MULHU {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(9),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::M(
            RV32MInstructions::MULHSU(RV32_M_MULHSU {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(8),
            }),
        ))),
    ])
}

fn generate_m_extension_corner_cases_rv64() -> SpiceChunk {
    SpiceChunk::Structured(vec![
        li_value_pseudo(13, i64::MIN, LoadImmediatePurpose::Generic, ISABase::Rv64),
        li_value_pseudo(14, -1, LoadImmediatePurpose::Generic, ISABase::Rv64),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::M(
            RV64MInstructions::DIV(RV64_M_DIV {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(15),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::M(
            RV64MInstructions::DIV(RV64_M_DIV {
                xs2: x_reg(0),
                xs1: x_reg(13),
                xd: x_reg(9),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::M(
            RV64MInstructions::REM(RV64_M_REM {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(15),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::M(
            RV64MInstructions::MULH(RV64_M_MULH {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(15),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::M(
            RV64MInstructions::MULHU(RV64_M_MULHU {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(9),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::M(
            RV64MInstructions::MULHSU(RV64_M_MULHSU {
                xs2: x_reg(14),
                xs1: x_reg(13),
                xd: x_reg(8),
            }),
        ))),
    ])
}

fn generate_c_extension_patterns_rv32() -> SpiceChunk {
    SpiceChunk::Structured(vec![
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::Zca(
            RV32ZcaInstructions::C_NOP(RV32_Zca_C_NOP),
        ))),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::Zca(
            RV32ZcaInstructions::C_LI(RV32_Zca_C_LI {
                imm: Immediate::<6, false>::new(1).expect("valid c.li immediate"),
                xd: restricted_rd(8),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::Zca(
            RV32ZcaInstructions::C_ADDI(RV32_Zca_C_ADDI {
                imm: RestrictedImmediate6ForbiddenRange0To0::<false>::new(2)
                    .expect("valid c.addi immediate"),
                xd: restricted_rd(8),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::Zca(
            RV32ZcaInstructions::C_MV(RV32_Zca_C_MV {
                xd: restricted_rd(9),
                xs2: restricted_rs2(8),
            }),
        ))),
    ])
}

fn generate_c_extension_patterns_rv64() -> SpiceChunk {
    SpiceChunk::Structured(vec![
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::Zca(
            RV64ZcaInstructions::C_NOP(RV64_Zca_C_NOP),
        ))),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::Zca(
            RV64ZcaInstructions::C_LI(RV64_Zca_C_LI {
                imm: Immediate::<6, false>::new(1).expect("valid c.li immediate"),
                xd: restricted_rd(8),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::Zca(
            RV64ZcaInstructions::C_ADDI(RV64_Zca_C_ADDI {
                imm: RestrictedImmediate6ForbiddenRange0To0::<false>::new(2)
                    .expect("valid c.addi immediate"),
                xd: restricted_rd(8),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::Zca(
            RV64ZcaInstructions::C_MV(RV64_Zca_C_MV {
                xd: restricted_rd(9),
                xs2: restricted_rs2(8),
            }),
        ))),
    ])
}

fn generate_zicsr_patterns_rv32() -> SpiceChunk {
    SpiceChunk::Structured(vec![
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::Zicsr(
            RV32ZicsrInstructions::CSRRS(RV32_Zicsr_CSRRS {
                csr: WritableCsr::Cycle,
                xs1: x_reg(0),
                xd: x_reg(13),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV32(RV32Instruction::Zicsr(
            RV32ZicsrInstructions::CSRRS(RV32_Zicsr_CSRRS {
                csr: WritableCsr::Instret,
                xs1: x_reg(0),
                xd: x_reg(14),
            }),
        ))),
    ])
}

fn generate_zicsr_patterns_rv64() -> SpiceChunk {
    SpiceChunk::Structured(vec![
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::Zicsr(
            RV64ZicsrInstructions::CSRRS(RV64_Zicsr_CSRRS {
                csr: WritableCsr::Cycle,
                xs1: x_reg(0),
                xd: x_reg(13),
            }),
        ))),
        instruction_entry(RiscvInstruction::RV64(RV64Instruction::Zicsr(
            RV64ZicsrInstructions::CSRRS(RV64_Zicsr_CSRRS {
                csr: WritableCsr::Instret,
                xs1: x_reg(0),
                xd: x_reg(14),
            }),
        ))),
    ])
}
