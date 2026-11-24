use rand::{Rng, SeedableRng, rngs::StdRng};
use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::{AsmDecodeWithContext, InstructionSequence, IsaDecodeContext};

use crate::{
    GenerationError, rvv_artifact_settings,
    testcase::{GeneratedTestCase, HartPhase},
};

const BEGIN: &str = "addi x0, x0, 1441";
const END: &str = "addi x0, x0, 1442";

fn sampled(seed: u64, percent: u8) -> bool {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x4e45_4741_5449_5645);
    rng.random_range(0..100u8) < percent
}

fn defect_first_enabled() -> bool {
    std::env::var("RISCV_FUZZ_RVV_DEFECT_FIRST").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn decode(line: &str) -> Result<RiscvInstruction, GenerationError> {
    RiscvInstruction::decode_asm_with_context(line, IsaDecodeContext::RV64).map_err(|error| {
        GenerationError::StructuredInstructionBuildFailed {
            message: format!("cannot build expected-Illegal probe `{line}`: {error}"),
        }
    })
}

fn mask_overlap_probe(seed: u64) -> Result<InstructionSequence<RiscvInstruction>, GenerationError> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5643_4e45_4741_5449);
    let defect_first = defect_first_enabled();
    let sew = [8, 16, 32, 64][rng.random_range(0..4)];
    let (lmul, width, encoding) =
        [("m1", 1, 0), ("m2", 2, 1), ("m4", 4, 2), ("m8", 8, 3)][rng.random_range(0..4)];
    debug_assert!(crate::rvv_legal::vtype_supported(
        sew,
        lmul,
        rvv_artifact_settings::ELEN_BITS
    ));
    let groups = crate::rvv_legal::legal_group_bases((1, 31), width);
    let left = groups[rng.random_range(0..groups.len())];
    let right = groups[rng.random_range(0..groups.len())];
    let control_dest = groups[rng.random_range(0..groups.len())];
    let avl = rng.random_range(1..=16);
    let sampled_policy = rng.random_range(0..2) << 6 | rng.random_range(0..2) << 7;
    let policy = if defect_first { 0 } else { sampled_policy };
    let vtype = encoding | (sew.trailing_zeros() - 3) << 3 | policy;
    let form = rng.random_range(0..3);
    let immediate = rng.random_range(-16..=15);
    let operation = |dest| match form {
        0 => format!("vmerge.vvm v{dest}, v{left}, v{right}, v0"),
        1 => format!("vmerge.vxm v{dest}, v{left}, x15, v0"),
        _ => format!("vmerge.vim v{dest}, v{left}, {immediate}, v0"),
    };
    let mut probe = InstructionSequence::new(decode(&operation(0))?);
    let mut setup = vec![BEGIN.to_string(), "csrrw x0, vstart, x0".to_string()];
    setup.push(if defect_first {
        format!("vsetvli x8, x0, {vtype}")
    } else {
        format!("vsetivli x8, {avl}, {vtype}")
    });
    setup.extend([
        format!("li x14, {}", rng.random::<i64>()),
        "vmv.v.x v0, x14".to_string(),
        format!("li x14, {}", rng.random::<i64>()),
        format!("vmv.v.x v{left}, x14"),
        format!("li x15, {}", rng.random::<i64>()),
        format!("vmv.v.x v{right}, x15"),
    ]);
    if defect_first {
        setup.push(format!("vmv.v.x v{control_dest}, x14"));
        setup.push(format!("vsetivli x8, {avl}, {vtype}"));
    }
    setup.push(operation(control_dest));
    probe.pre_instructions = setup
        .iter()
        .map(|line| decode(line))
        .collect::<Result<_, _>>()?;
    probe.post_instructions = ["csrrw x0, vstart, x0", END]
        .iter()
        .map(|line| decode(line))
        .collect::<Result<_, _>>()?;
    Ok(probe)
}

pub(crate) fn append_expected_illegal_probe(
    testcase: &mut GeneratedTestCase,
) -> Result<(), GenerationError> {
    let percent = rvv_artifact_settings::NEGATIVE_PROGRAM_PERCENT;
    if percent == 0 {
        return Ok(());
    }
    let seed =
        testcase
            .config
            .seed
            .ok_or_else(|| GenerationError::StructuredInstructionBuildFailed {
                message: "negative sampling requires an explicit seed".to_string(),
            })?;
    if sampled(seed, percent) {
        testcase.harts[0]
            .sequences_mut(HartPhase::Test)
            .push(mask_overlap_probe(seed)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_is_per_program_and_reproducible() {
        assert!(rvv_artifact_settings::NEGATIVE_PROGRAM_PERCENT <= 100);
        for seed in 0..1000 {
            assert!(!sampled(seed, 0));
            assert!(sampled(seed, 100));
            assert_eq!(sampled(seed, 5), sampled(seed, 5));
            assert!(!sampled(seed, 5) || sampled(seed, 10));
        }
        let bucket_counts = (0..10)
            .map(|bucket| {
                (bucket * 100..(bucket + 1) * 100)
                    .filter(|seed| sampled(*seed, 5))
                    .count()
            })
            .collect::<Vec<_>>();
        assert!(bucket_counts.iter().any(|count| *count != 5));
    }

    #[test]
    fn negative_probe_preserves_alignment_and_randomness() {
        let mut operations = std::collections::BTreeSet::new();
        let mut forms = std::collections::BTreeSet::new();
        for seed in 0..300 {
            let probe = mask_overlap_probe(seed).unwrap();
            let line = probe.instruction.to_string();
            assert!(line.split_once(' ').unwrap().1.starts_with("v0,"));
            assert!(line.ends_with("v0"));
            let control = probe.pre_instructions.last().unwrap().to_string();
            assert!(!control.split_once(' ').unwrap().1.starts_with("v0,"));
            forms.insert(line.split_whitespace().next().unwrap().to_string());
            operations.insert(line);
            let setup = probe.pre_instructions[2].to_string();
            let imm: u32 = setup.rsplit(',').next().unwrap().trim().parse().unwrap();
            let width = 1u8 << (imm & 7);
            for token in probe
                .instruction
                .to_string()
                .split([',', ' '])
                .filter(|x| x.starts_with('v') && !x.contains('.'))
            {
                let base = token[1..].parse::<u8>().unwrap();
                assert!(crate::rvv_legal::valid_group(base, width));
            }
        }
        assert_eq!(forms.len(), 3);
        assert!(operations.len() > 100);
    }
}
