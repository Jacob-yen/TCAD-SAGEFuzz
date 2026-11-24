//! Export the generator's actual per-opcode parameter domains for an RVV DUT.
use std::{env, fs};

use rand::{SeedableRng as _, rngs::StdRng};
use riscv_fuzz_testcase_engine::rvv_catalog_legal::{RvvSamplingProfile, domain_for_asm};
use riscv_instruction::{RV64Instruction, RV64Zvl32bInstructions, RiscvInstruction};
use riscv_instruction_types::{RandomConfig, build_instruction_sequence};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 || !matches!(args[1].as_str(), "ara" | "xiangshan") {
        return Err("usage: cargo run -p riscv-fuzz-testcase-engine --example export_rvv_domains -- <ara|xiangshan> <output.json>".into());
    }
    let subject = &args[1];
    let profile = RvvSamplingProfile::from_subject_and_env(subject)?;
    let random_config = RandomConfig::default();
    let mut domains = Vec::new();
    for index in 0..RV64Zvl32bInstructions::VARIANT_COUNT {
        let mut rng = StdRng::seed_from_u64(index as u64 ^ 0x5643_5346_4341_5441);
        let variant = RV64Zvl32bInstructions::random_at_index(index, &mut rng, &random_config)?;
        let sequence = build_instruction_sequence(
            RiscvInstruction::RV64(RV64Instruction::Zvl32b(variant)),
            &mut rng,
            &random_config,
        )?;
        let asm = sequence.instruction.to_string();
        let domain = domain_for_asm(&asm, profile, (13, 31))
            .ok_or_else(|| format!("cannot describe catalog index {index}: {asm}"))?;
        if domain.configurations.is_empty() {
            return Err(format!("empty domain for catalog index {index}: {asm}").into());
        }
        domains.push(domain);
    }
    let report = serde_json::json!({
        "subject": subject,
        "profile": profile,
        "vector_register_policy_range": [13, 31],
        "domain_semantics": "configurations are joint-feasible; operand register lists are projections and must obey pairwise relation rules",
        "opcodes": domains,
    });
    fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
