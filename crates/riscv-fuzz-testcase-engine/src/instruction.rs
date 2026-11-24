use std::{
    collections::{BTreeMap, HashMap},
    convert::TryFrom,
};

use crate::{GenerationError, testcase::StructuredTestInstructions};
use rand::{SeedableRng as _, rngs::StdRng};
use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
use riscv_instruction::{
    RV32_F_FMV_W_X, RV32_I_SB, RV32_I_SLL, RV32_I_SRL, RV32_I_SW, RV32_Zvl32b_VMV_V_X,
    RV32_Zvl32b_VSETVLI, RV32FInstructions, RV32IInstructions, RV32Instruction,
    RV32Zvl32bInstructions, RV64_D_FMV_D_X, RV64_I_SB, RV64_I_SD, RV64_Zvl32b_VMV_V_X,
    RV64_Zvl32b_VSETVLI, RV64DInstructions, RV64Extensions, RV64IInstructions, RV64Instruction,
    RV64Zvl32bInstructions, RiscvInstruction,
};
use riscv_instruction_types::{
    BaseAddressRegister, FloatingPointRegister, Immediate, IntegerRegister, IsaProfile,
    LoadImmediatePurpose, PseudoInstruction, UImmediate, ValidatedValue, VectorRegister,
    XLenAddress, XLenImmediate,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloatInitTransfer {
    Single,
    Double,
}

fn float_init_transfer(extensions: &IsaExtensions, isa_base: ISABase) -> FloatInitTransfer {
    match isa_base {
        ISABase::Rv32 => FloatInitTransfer::Single,
        ISABase::Rv64 => {
            if extensions.contains_rv64(RV64Extensions::D) {
                FloatInitTransfer::Double
            } else {
                FloatInitTransfer::Single
            }
        }
    }
}

fn x_reg(reg: u8) -> IntegerRegister {
    IntegerRegister::new(reg).expect("valid integer register")
}

fn base_reg(reg: u8) -> BaseAddressRegister {
    BaseAddressRegister::new(reg).expect("valid base address register")
}

fn f_reg(reg: u8) -> FloatingPointRegister {
    FloatingPointRegister::new(reg).expect("valid floating-point register")
}

fn v_reg(reg: u8) -> VectorRegister {
    VectorRegister::new(reg).expect("valid vector register")
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
) -> PseudoInstruction {
    PseudoInstruction::LoadImmediate {
        rd: x_reg(rd),
        immediate: XLenImmediate::from_i64_for(isa_profile(isa_base), immediate)
            .expect("load-immediate value must fit active xlen"),
        purpose,
    }
}

fn li_bits_pseudo(
    rd: u8,
    immediate: u64,
    purpose: LoadImmediatePurpose,
    isa_base: ISABase,
) -> PseudoInstruction {
    PseudoInstruction::LoadImmediate {
        rd: x_reg(rd),
        immediate: XLenImmediate::from_unsigned_bits_for(isa_profile(isa_base), immediate),
        purpose,
    }
}

fn li_word_pseudo(
    rd: u8,
    value: u64,
    purpose: LoadImmediatePurpose,
    isa_base: ISABase,
) -> PseudoInstruction {
    match isa_base {
        ISABase::Rv32 => li_bits_pseudo(rd, value & u32::MAX as u64, purpose, isa_base),
        ISABase::Rv64 => li_bits_pseudo(rd, value, purpose, isa_base),
    }
}

fn li_address_pseudo(
    rd: u8,
    address: u64,
    purpose: LoadImmediatePurpose,
    isa_base: ISABase,
) -> PseudoInstruction {
    let address = XLenAddress::from_u64_for(isa_profile(isa_base), address)
        .expect("load-immediate address must fit active xlen");
    PseudoInstruction::LoadImmediate {
        rd: x_reg(rd),
        immediate: XLenImmediate::from_address(address),
        purpose,
    }
}

fn rv32_sll(rd: u8, rs1: u8, rs2: u8) -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::SLL(RV32_I_SLL {
        xd: x_reg(rd),
        xs1: x_reg(rs1),
        xs2: x_reg(rs2),
    })))
}

fn rv32_srl(rd: u8, rs1: u8, rs2: u8) -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::SRL(RV32_I_SRL {
        xd: x_reg(rd),
        xs1: x_reg(rs1),
        xs2: x_reg(rs2),
    })))
}

fn rv32_sw(rs2: u8, offset: i64, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::SW(RV32_I_SW {
        xs2: x_reg(rs2),
        xs1: base_reg(rs1),
        imm: Immediate::<12, true>::new(offset as i32).expect("valid SW offset"),
    })))
}

fn rv64_sd(rs2: u8, offset: i64, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::SD(RV64_I_SD {
        xs2: IntegerRegister::new(rs2).expect("valid SD source register"),
        xs1: base_reg(rs1),
        imm: Immediate::<12, true>::new(offset as i32).expect("valid SD offset"),
    })))
}

fn rv32_fmv_w_x(fd: u8, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::F(RV32FInstructions::FMV_W_X(
        RV32_F_FMV_W_X {
            xs1: x_reg(rs1),
            fd: f_reg(fd),
        },
    )))
}

fn rv64_fmv_d_x(fd: u8, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV64(RV64Instruction::D(RV64DInstructions::FMV_D_X(
        RV64_D_FMV_D_X {
            xs1: x_reg(rs1),
            fd: f_reg(fd),
        },
    )))
}

fn rv32_sb(rs2: u8, offset: i64, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::SB(RV32_I_SB {
        xs2: x_reg(rs2),
        xs1: base_reg(rs1),
        imm: Immediate::<12, true>::new(offset as i32).expect("valid SB offset"),
    })))
}

fn rv64_sb(rs2: u8, offset: i64, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::SB(RV64_I_SB {
        xs2: x_reg(rs2),
        xs1: base_reg(rs1),
        imm: Immediate::<12, true>::new(offset as i32).expect("valid SB offset"),
    })))
}

fn vector_vtype_immediate(isa_base: ISABase) -> UImmediate<11, false> {
    // RVV vtype immediate: bits [2:0] encode LMUL, bits [5:3] encode SEW.
    // Use m1 with XLEN-sized elements for deterministic vector initialization.
    let vtypei = match isa_base {
        ISABase::Rv32 => 0b010 << 3, // e32, m1
        ISABase::Rv64 => 0b011 << 3, // e64, m1
    };
    UImmediate::<11, false>::new(vtypei).expect("valid vsetvli vtype immediate")
}

fn rv32_vsetvli(rd: u8, rs1: u8, isa_base: ISABase) -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::Zvl32b(RV32Zvl32bInstructions::VSETVLI(
        RV32_Zvl32b_VSETVLI {
            xd: x_reg(rd),
            xs1: x_reg(rs1),
            vtypei: vector_vtype_immediate(isa_base),
        },
    )))
}

fn rv64_vsetvli(rd: u8, rs1: u8, isa_base: ISABase) -> RiscvInstruction {
    RiscvInstruction::RV64(RV64Instruction::Zvl32b(RV64Zvl32bInstructions::VSETVLI(
        RV64_Zvl32b_VSETVLI {
            xd: x_reg(rd),
            xs1: x_reg(rs1),
            vtypei: vector_vtype_immediate(isa_base),
        },
    )))
}

fn rv32_vmv_v_x(vd: u8, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::Zvl32b(RV32Zvl32bInstructions::VMV_V_X(
        RV32_Zvl32b_VMV_V_X {
            xs1: x_reg(rs1),
            vd: v_reg(vd),
        },
    )))
}

fn rv64_vmv_v_x(vd: u8, rs1: u8) -> RiscvInstruction {
    RiscvInstruction::RV64(RV64Instruction::Zvl32b(RV64Zvl32bInstructions::VMV_V_X(
        RV64_Zvl32b_VMV_V_X {
            xs1: x_reg(rs1),
            vd: v_reg(vd),
        },
    )))
}

fn push_pseudo_to_hart(
    program: &mut StructuredTestInstructions,
    hart_idx: usize,
    pseudo: PseudoInstruction,
) {
    program.push_pseudo_to_hart(hart_idx, pseudo);
}

fn push_instruction_to_hart(
    program: &mut StructuredTestInstructions,
    hart_idx: usize,
    instruction: RiscvInstruction,
) {
    program.push_instruction_to_hart(hart_idx, instruction);
}

fn offset_init_program_base_addresses(program: &mut StructuredTestInstructions, offset: i64) {
    if offset == 0 {
        return;
    }

    for hart in &mut program.harts {
        for instruction in hart {
            if let RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
                immediate,
                purpose,
                ..
            }) = instruction
                && *purpose == LoadImmediatePurpose::BaseAddress
            {
                let next = immediate.as_i64().saturating_add(offset);
                *immediate = match immediate {
                    XLenImmediate::X32(_) => XLenImmediate::X32(next as i32),
                    XLenImmediate::X64(_) => XLenImmediate::X64(next),
                };
            }
        }
    }
}

pub fn generate_initialization_instructions(
    extensions: &IsaExtensions,
    isa_base: ISABase,
    mem_size: u64,
    random_config: &riscv_instruction_types::RandomConfig,
) -> Result<StructuredTestInstructions, GenerationError> {
    generate_initialization_instructions_for_harts(extensions, isa_base, mem_size, random_config, 1)
}

fn temp_registers(
    random_config: &riscv_instruction_types::RandomConfig,
) -> Result<Vec<u8>, GenerationError> {
    let (temp_reg_start, temp_reg_end) = random_config
        .memory_access_config
        .memory_base_register_number_range;
    let temp_regs: Vec<u8> = (temp_reg_start..=temp_reg_end).collect();

    if temp_regs.len() < 3 {
        return Err(GenerationError::InsufficientTempRegisters {
            required: 3,
            available: temp_regs.len(),
        });
    }

    Ok(temp_regs)
}

fn append_operand_register_initialization<R: rand::Rng>(
    program: &mut StructuredTestInstructions,
    hart_idx: usize,
    extensions: &IsaExtensions,
    isa_base: ISABase,
    random_config: &riscv_instruction_types::RandomConfig,
    temp_regs: &[u8],
    include_vector: bool,
    rng: &mut R,
) {
    let register_config = &random_config.register_operand_config;
    let support_vector = include_vector && extensions.has_enabled_vector();
    let float_transfer = float_init_transfer(extensions, isa_base);

    let int_values: Vec<u64> = {
        let (int_start, int_end) = register_config.integer_register_number_range;
        (int_start..=int_end)
            .map(|_| match isa_base {
                ISABase::Rv32 => {
                    let value = riscv_instruction_types::sample_signed_value(
                        rng,
                        i32::MIN as i64,
                        i32::MAX as i64,
                        32,
                    );
                    value as u64
                }
                ISABase::Rv64 => {
                    let value =
                        riscv_instruction_types::sample_signed_value(rng, i64::MIN, i64::MAX, 64);
                    value as u64
                }
            })
            .collect()
    };

    let fp_values: Vec<u64> = if extensions.has_enabled_float() {
        let (fp_start, fp_end) = register_config.floating_point_register_number_range;
        (fp_start..=fp_end)
            // Float init uses raw randomized bit patterns so init-time divergences remain
            // visible instead of being masked by a curated stable subset.
            .map(|_| match float_transfer {
                FloatInitTransfer::Single => {
                    riscv_instruction_types::sample_unsigned_value(rng, 0, u32::MAX as u64, 32)
                }
                FloatInitTransfer::Double => {
                    riscv_instruction_types::sample_unsigned_value(rng, 0, u64::MAX, 64)
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    let vector_values: Vec<u64> = if support_vector {
        let (vec_start, vec_end) = register_config.vector_register_number_range;
        if vec_start > vec_end {
            Vec::new()
        } else {
            (vec_start..=vec_end)
                .map(|_| match isa_base {
                    ISABase::Rv32 => riscv_instruction_types::sample_vector_pattern(rng, 32),
                    ISABase::Rv64 => riscv_instruction_types::sample_vector_pattern(rng, 64),
                })
                .collect()
        }
    } else {
        Vec::new()
    };

    let temp_reg0 = temp_regs[0];
    let temp_reg1 = temp_regs[1];

    let (int_start, _) = register_config.integer_register_number_range;
    for (i, &value) in int_values.iter().enumerate() {
        let reg_num = int_start + i as u8;
        push_pseudo_to_hart(
            program,
            hart_idx,
            li_bits_pseudo(reg_num, value, LoadImmediatePurpose::Generic, isa_base),
        );
    }

    let (fp_start, _) = register_config.floating_point_register_number_range;
    for (i, &value) in fp_values.iter().enumerate() {
        let reg_num = fp_start + i as u8;
        push_pseudo_to_hart(
            program,
            hart_idx,
            li_bits_pseudo(temp_reg0, value, LoadImmediatePurpose::Generic, isa_base),
        );

        match float_transfer {
            FloatInitTransfer::Single => {
                push_instruction_to_hart(program, hart_idx, rv32_fmv_w_x(reg_num, temp_reg0));
            }
            FloatInitTransfer::Double => {
                push_instruction_to_hart(program, hart_idx, rv64_fmv_d_x(reg_num, temp_reg0));
            }
        }
    }

    if support_vector && !vector_values.is_empty() {
        match isa_base {
            ISABase::Rv32 => {
                push_instruction_to_hart(program, hart_idx, rv32_vsetvli(temp_reg1, 0, isa_base))
            }
            ISABase::Rv64 => {
                push_instruction_to_hart(program, hart_idx, rv64_vsetvli(temp_reg1, 0, isa_base))
            }
        }

        let (vec_start, _) = register_config.vector_register_number_range;
        for (i, &value) in vector_values.iter().enumerate() {
            let vreg_num = vec_start + i as u8;
            push_pseudo_to_hart(
                program,
                hart_idx,
                li_word_pseudo(temp_reg0, value, LoadImmediatePurpose::Generic, isa_base),
            );
            match isa_base {
                ISABase::Rv32 => {
                    push_instruction_to_hart(program, hart_idx, rv32_vmv_v_x(vreg_num, temp_reg0))
                }
                ISABase::Rv64 => {
                    push_instruction_to_hart(program, hart_idx, rv64_vmv_v_x(vreg_num, temp_reg0))
                }
            }
        }
    }
}

fn append_memory_initialization<R: rand::Rng>(
    program: &mut StructuredTestInstructions,
    hart_idx: usize,
    isa_base: ISABase,
    mem_size: u64,
    temp_regs: &[u8],
    rng: &mut R,
) {
    let word_size = isa_base.word_size() as u64;
    let word_count = mem_size / word_size;

    let mem_data: Vec<u64> = (0..word_count)
        .map(|_| match isa_base {
            ISABase::Rv32 => {
                riscv_instruction_types::sample_unsigned_value(rng, 0, u32::MAX as u64, 32)
            }
            ISABase::Rv64 => riscv_instruction_types::sample_unsigned_value(rng, 0, u64::MAX, 64),
        })
        .collect();

    let temp_reg0 = temp_regs[0];
    let temp_reg1 = temp_regs[1];
    let temp_reg2 = temp_regs[2];

    if !mem_data.is_empty() {
        push_pseudo_to_hart(
            program,
            hart_idx,
            li_address_pseudo(temp_reg1, 0, LoadImmediatePurpose::BaseAddress, isa_base),
        );
        if matches!(isa_base, ISABase::Rv32) {
            push_pseudo_to_hart(
                program,
                hart_idx,
                li_value_pseudo(temp_reg0, 32, LoadImmediatePurpose::Generic, isa_base),
            );
            push_instruction_to_hart(program, hart_idx, rv32_sll(temp_reg1, temp_reg1, temp_reg0));
            push_instruction_to_hart(program, hart_idx, rv32_srl(temp_reg1, temp_reg1, temp_reg0));
        }

        for (i, &data) in mem_data.iter().enumerate() {
            let offset = (i as u64) * word_size;
            push_pseudo_to_hart(
                program,
                hart_idx,
                li_bits_pseudo(temp_reg2, data, LoadImmediatePurpose::Generic, isa_base),
            );

            match isa_base {
                ISABase::Rv32 => {
                    let offset_i64 = i64::try_from(offset).ok();
                    push_instruction_to_hart(
                        program,
                        hart_idx,
                        rv32_sw(
                            temp_reg2,
                            offset_i64.expect("offset fits in i64"),
                            temp_reg1,
                        ),
                    );
                }
                ISABase::Rv64 => {
                    let offset_i64 = i64::try_from(offset).ok();
                    push_instruction_to_hart(
                        program,
                        hart_idx,
                        rv64_sd(
                            temp_reg2,
                            offset_i64.expect("offset fits in i64"),
                            temp_reg1,
                        ),
                    );
                }
            }
        }
    }
}

fn append_temp_register_clear(
    program: &mut StructuredTestInstructions,
    hart_idx: usize,
    isa_base: ISABase,
    temp_regs: &[u8],
) {
    for &temp_reg in temp_regs {
        push_pseudo_to_hart(
            program,
            hart_idx,
            li_value_pseudo(temp_reg, 0, LoadImmediatePurpose::Generic, isa_base),
        );
    }
}

pub fn generate_initialization_instructions_for_harts(
    extensions: &IsaExtensions,
    isa_base: ISABase,
    mem_size: u64,
    random_config: &riscv_instruction_types::RandomConfig,
    hart_count: u32,
) -> Result<StructuredTestInstructions, GenerationError> {
    generate_initialization_instructions_for_harts_with_rng(
        extensions,
        isa_base,
        mem_size,
        random_config,
        hart_count,
        &mut rand::rng(),
    )
}

/// Keep model-only generated programs reproducible without changing the
/// historical initialization behavior of other testcase generators.
pub fn generate_initialization_instructions_for_harts_seeded(
    extensions: &IsaExtensions,
    isa_base: ISABase,
    mem_size: u64,
    random_config: &riscv_instruction_types::RandomConfig,
    hart_count: u32,
    seed: u64,
) -> Result<StructuredTestInstructions, GenerationError> {
    generate_initialization_instructions_for_harts_with_rng(
        extensions,
        isa_base,
        mem_size,
        random_config,
        hart_count,
        &mut StdRng::seed_from_u64(seed),
    )
}

pub(crate) fn generate_compact_initialization_instructions_seeded(
    extensions: &IsaExtensions,
    isa_base: ISABase,
    random_config: &riscv_instruction_types::RandomConfig,
    hart_count: u32,
    seed: u64,
) -> Result<StructuredTestInstructions, GenerationError> {
    let temp_regs = temp_registers(random_config)?;
    let mut program = StructuredTestInstructions::default();
    let mut rng = StdRng::seed_from_u64(seed);
    for hart_idx in 0..hart_count.max(1) as usize {
        append_operand_register_initialization(
            &mut program,
            hart_idx,
            extensions,
            isa_base,
            random_config,
            &temp_regs,
            false,
            &mut rng,
        );
        append_temp_register_clear(&mut program, hart_idx, isa_base, &temp_regs);
    }
    Ok(program)
}

fn generate_initialization_instructions_for_harts_with_rng<R: rand::Rng>(
    extensions: &IsaExtensions,
    isa_base: ISABase,
    mem_size: u64,
    random_config: &riscv_instruction_types::RandomConfig,
    hart_count: u32,
    rng: &mut R,
) -> Result<StructuredTestInstructions, GenerationError> {
    let temp_regs = temp_registers(random_config)?;
    let mut program = StructuredTestInstructions::default();
    let hart_count = hart_count.max(1);

    for hart_idx in 0..hart_count as usize {
        append_operand_register_initialization(
            &mut program,
            hart_idx,
            extensions,
            isa_base,
            random_config,
            &temp_regs,
            true,
            rng,
        );
        if hart_idx == 0 {
            append_memory_initialization(
                &mut program,
                hart_idx,
                isa_base,
                mem_size,
                &temp_regs,
                rng,
            );
        }
        append_temp_register_clear(&mut program, hart_idx, isa_base, &temp_regs);
    }

    Ok(program)
}

#[cfg(test)]
mod compact_init_tests {
    use super::*;
    use riscv_instruction::RV64Extensions;

    #[test]
    fn compact_init_keeps_scalar_sources_without_bulk_memory_or_vector_setup() {
        let extensions = IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::F,
            RV64Extensions::D,
            RV64Extensions::Zvl32b,
        ]);
        let init = generate_compact_initialization_instructions_seeded(
            &extensions,
            ISABase::Rv64,
            &riscv_instruction_types::RandomConfig::default(),
            1,
            19,
        )
        .expect("compact register initialization");
        let lines = init.rendered_lines();
        assert!(lines.iter().any(|line| line.starts_with("fmv.d.x")));
        assert!(!lines.iter().any(|line| line.starts_with('v')));
        assert!(!lines.iter().any(|line| line.starts_with("sd ")));
    }
}

pub fn generate_initialization_instructions_for_impls(
    impls: &[String],
    extensions: &IsaExtensions,
    isa_base: ISABase,
    mem_range: &HashMap<String, (u64, u64)>,
    random_config: &riscv_instruction_types::RandomConfig,
) -> Result<HashMap<String, StructuredTestInstructions>, GenerationError> {
    let canonical = generate_initialization_instructions(
        extensions,
        isa_base,
        mem_range
            .values()
            .next()
            .map(|(start, end)| end.saturating_sub(*start).saturating_add(1))
            .ok_or(GenerationError::EmptyMemRange)?,
        random_config,
    )?;

    let mut result = HashMap::new();
    for impl_ref in impls {
        let (start, _end) =
            mem_range
                .get(impl_ref)
                .ok_or_else(|| GenerationError::MemRangeNotFound {
                    core_id: format!("{:?}", impl_ref),
                })?;
        let offset = i64::try_from(*start)
            .map_err(|_| GenerationError::MemoryAddressOutOfRange { addr: *start })?;
        let mut program = canonical.clone();
        offset_init_program_base_addresses(&mut program, offset);
        result.insert(impl_ref.clone(), program);
    }

    Ok(result)
}

pub fn generate_register_context_restore_instructions(
    register_context: &HashMap<String, u64>,
    isa_base: ISABase,
    temp_register_num_range: (u8, u8),
) -> Result<Vec<RiscvInstruction>, GenerationError> {
    if register_context.is_empty() {
        return Ok(Vec::new());
    }

    let (temp_start, temp_end) = temp_register_num_range;
    let temp_registers: Vec<u8> = (temp_start..=temp_end).collect();

    let needs_float_helper = register_context.keys().any(|name| name.starts_with('f'));
    if needs_float_helper && temp_registers.is_empty() {
        return Err(GenerationError::InsufficientTempRegisters {
            required: 1,
            available: 0,
        });
    }

    let float_helper = temp_registers.first().copied();

    let mut registers: Vec<(&String, &u64)> = register_context.iter().collect();
    registers.sort_by(|(left, _), (right, _)| {
        fn reg_key(name: &str) -> (char, u8) {
            let mut chars = name.chars();
            if let Some(prefix) = chars.next() {
                if let Ok(index) = chars.as_str().parse::<u8>() {
                    return (prefix, index);
                }
                return (prefix, 0);
            }
            ('~', 0)
        }

        reg_key(left).cmp(&reg_key(right))
    });

    let mut instructions = Vec::new();

    for (name, &value) in registers {
        let mut chars = name.chars();
        let Some(prefix) = chars.next() else {
            continue;
        };
        let digits = chars.as_str();
        let parsed_index = digits.parse::<u8>();

        match prefix {
            'x' => {
                let reg_index = match parsed_index {
                    Ok(idx) => idx,
                    Err(err) => {
                        log::warn!(
                            "skip register context '{}' due to parse error: {}",
                            name,
                            err
                        );
                        continue;
                    }
                };

                if reg_index == 0 {
                    continue;
                }
                instructions.push(
                    li_word_pseudo(reg_index, value, LoadImmediatePurpose::Generic, isa_base)
                        .into(),
                );
            }
            'f' => {
                let Some(helper) = float_helper else {
                    return Err(GenerationError::InsufficientTempRegisters {
                        required: 1,
                        available: 0,
                    });
                };

                let reg_index = match parsed_index {
                    Ok(idx) => idx,
                    Err(err) => {
                        log::warn!("skip float context '{}' due to parse error: {}", name, err);
                        continue;
                    }
                };
                instructions.push(
                    li_word_pseudo(helper, value, LoadImmediatePurpose::Generic, isa_base).into(),
                );
                instructions.push(match isa_base {
                    ISABase::Rv32 => rv32_fmv_w_x(reg_index, helper),
                    ISABase::Rv64 => rv64_fmv_d_x(reg_index, helper),
                });
            }
            'v' => {
                log::warn!(
                    "skip vector register '{}' while generating context recovery instructions",
                    name
                );
            }
            _ => {
                log::warn!(
                    "skip unsupported register '{}' while generating context recovery instructions",
                    name
                );
            }
        }
    }

    Ok(instructions)
}

pub fn generate_memory_context_restore_instructions(
    memory_context: &BTreeMap<u64, u8>,
    mem_start: u64,
    isa_base: ISABase,
    temp_register_num_range: (u8, u8),
) -> Result<Vec<RiscvInstruction>, GenerationError> {
    if memory_context.is_empty() {
        return Ok(Vec::new());
    }

    let (temp_start, temp_end) = temp_register_num_range;
    let temp_registers: Vec<u8> = (temp_start..=temp_end).collect();

    if temp_registers.len() < 2 {
        return Err(GenerationError::InsufficientTempRegisters {
            required: 2,
            available: temp_registers.len(),
        });
    }

    let addr_reg = temp_registers[0];
    let value_reg = temp_registers[1];
    let mut instructions = Vec::new();

    for (&offset, &byte_value) in memory_context {
        let absolute_addr =
            mem_start
                .checked_add(offset)
                .ok_or(GenerationError::MemoryAddressOutOfRange {
                    addr: mem_start.saturating_add(offset),
                })?;
        let absolute_addr =
            i64::try_from(absolute_addr).map_err(|_| GenerationError::MemoryAddressOutOfRange {
                addr: absolute_addr,
            })?;

        instructions.push(
            li_address_pseudo(
                addr_reg,
                absolute_addr as u64,
                LoadImmediatePurpose::Generic,
                isa_base,
            )
            .into(),
        );
        instructions.push(
            li_value_pseudo(
                value_reg,
                i64::from(byte_value as i8),
                LoadImmediatePurpose::Generic,
                isa_base,
            )
            .into(),
        );
        instructions.push(match isa_base {
            ISABase::Rv32 => rv32_sb(value_reg, 0, addr_reg),
            ISABase::Rv64 => rv64_sb(value_reg, 0, addr_reg),
        });
    }

    Ok(instructions)
}
