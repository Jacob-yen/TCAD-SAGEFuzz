use riscv_instruction::{GenerationOrder, GenerationPlan, RV64Extensions, generate_instructions, generate_sequences};
use riscv_instruction_types::{AsmEncode, MemoryAccessConfig, RandomConfig, RegisterOperandConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let random_config = RandomConfig {
        register_operand_config: RegisterOperandConfig {
            integer_register_number_range: (8, 20),
            ..RegisterOperandConfig::default()
        },
        memory_access_config: MemoryAccessConfig {
            memory_base_address_range: (0x8000_0000, 0x8000_0FFF),
            memory_offset_immediate_range: (-64, 64),
            memory_base_register_number_range: (8, 12),
        },
        edge_case_probability: 0.35,
        ..RandomConfig::default()
    };

    let plan = GenerationPlan::new([(RV64Extensions::I, 6), (RV64Extensions::M, 4)])
        .with_order(GenerationOrder::Sequential);

    let sequences = generate_sequences(&plan, &random_config)?;
    let instructions = generate_instructions(&plan, &random_config)?;

    println!("Generated {} separated sequences", sequences.len());
    println!(
        "Collected {} separated main instructions",
        instructions.len()
    );

    for (idx, seq) in sequences.iter().take(8).enumerate() {
        println!("[{}] {}", idx, seq.encode_asm());
    }

    Ok(())
}
