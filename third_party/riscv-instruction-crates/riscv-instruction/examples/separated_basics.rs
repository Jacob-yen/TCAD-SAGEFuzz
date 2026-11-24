use riscv_instruction::*;
use riscv_instruction_types::{
    AsmEncode, InstructionSequence, MemoryAccessConfig, MemoryAccessInstruction, RandomConfig,
    RandomInstructionSequence, RegisterOperandConfig,
};
use serde_json::{from_str as json_from_str, to_string_pretty as json_to_string_pretty};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::rng();
    let config = RandomConfig {
        memory_access_config: MemoryAccessConfig {
            memory_base_register_number_range: (8, 12),
            memory_offset_immediate_range: (0, 16),
            ..MemoryAccessConfig::default()
        },
        register_operand_config: RegisterOperandConfig {
            integer_register_number_range: (13, 20),
            ..RegisterOperandConfig::default()
        },
        ..RandomConfig::default()
    };

    let random_sequence = RiscvInstruction::random_sequence_with_rng(&mut rng, &config)?;
    println!(
        "Random separated instruction sequence:\n{}",
        random_sequence.encode_asm()
    );

    let serialized = json_to_string_pretty(&random_sequence)?;
    println!("Serialized Sequence JSON:\n{}", serialized);
    let restored: InstructionSequence<RiscvInstruction> = json_from_str(&serialized)?;
    assert_eq!(restored, random_sequence);

    println!(
        "RV32I exposes {} instructions; RV64I exposes {} instructions",
        RV32Extensions::I.instruction_count(),
        RV64Extensions::I.instruction_count()
    );

    let xd = IntegerRegister::new(5)?;
    let xs1 = IntegerRegister::new(6)?;
    let xs2 = IntegerRegister::new(7)?;
    let imm12 = riscv_instruction_types::Immediate::<12>::new(200)?;

    let addi = RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(RV32_I_ADDI {
        imm: imm12,
        xs1,
        xd,
    })));
    println!("Separated RV32 ADDI: {}", addi.encode_asm());

    let mulw = RiscvInstruction::RV64(RV64Instruction::M(RV64MInstructions::MULW(RV64_M_MULW {
        xs2,
        xs1,
        xd,
    })));
    println!("Separated RV64 MULW: {}", mulw.encode_asm());

    let base = riscv_instruction_types::BaseAddressRegister::new(8)?;
    let lw_offset = riscv_instruction_types::Immediate::<12, true>::new(16)?;
    let lw = RV32_I_LW {
        imm: lw_offset,
        xs1: base,
        xd,
    };
    if let Some(offset) = lw.offset_operand_value() {
        println!("RV32 LW struct offset: {}", offset);
    }

    let lw_instruction =
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::LW(lw.clone())));
    if let Some(offset) = lw_instruction.offset_operand_value() {
        println!("RV32 LW enum offset: {}", offset);
    }

    Ok(())
}
