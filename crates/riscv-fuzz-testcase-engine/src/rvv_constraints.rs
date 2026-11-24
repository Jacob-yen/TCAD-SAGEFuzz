use rand::{SeedableRng as _, rngs::StdRng};
use riscv_fuzz_plugin_api::ISABase;
use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::{AsmDecodeWithContext, InstructionSequence, IsaDecodeContext};

use crate::{
    GenerationError, rvv_legal,
    testcase::{GeneratedTestCase, HartPhase},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RvvConfigConstraint {
    pub isa_base: ISABase,
    pub sew: u16,
    pub vl: u8,
    pub lmul: &'static str,
    pub tail_policy: &'static str,
    pub mask_policy: &'static str,
}

impl RvvConfigConstraint {
    pub fn new(
        isa_base: ISABase,
        sew: u16,
        vl: u8,
        lmul: &'static str,
        tail_policy: &'static str,
        mask_policy: &'static str,
    ) -> Self {
        Self {
            isa_base,
            sew,
            vl,
            lmul,
            tail_policy,
            mask_policy,
        }
    }

    pub fn vsetivli_vtype_immediate(self) -> u16 {
        // RVV vtype immediate: vlmul in bits [2:0], vsew in bits [5:3],
        // vta in bit 6, vma in bit 7.  e32/m1/tu/mu is 0b010 << 3.
        let vsew_bits = match self.sew {
            8 => 0,
            16 => 1,
            32 => 2,
            64 => 3,
            _ => 2,
        };
        let vlmul_bits = match self.lmul {
            "m1" => 0,
            "m2" => 1,
            "m4" => 2,
            "m8" => 3,
            "mf2" => 0b111,
            "mf4" => 0b110,
            "mf8" => 0b101,
            _ => 0,
        };
        let vta_bit = u16::from(self.tail_policy == "ta") << 6;
        let vma_bit = u16::from(self.mask_policy == "ma") << 7;
        (vsew_bits << 3) | vlmul_bits | vta_bit | vma_bit
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RvvConstraintInsertionReport {
    pub constrained_instructions: usize,
    pub inserted_instructions: usize,
    pub operand_repaired_instructions: usize,
}

fn isa_decode_context(isa_base: ISABase) -> IsaDecodeContext {
    match isa_base {
        ISABase::Rv32 => IsaDecodeContext::RV32,
        ISABase::Rv64 => IsaDecodeContext::RV64,
    }
}

fn decode_constraint_instruction(
    asm: &str,
    isa_base: ISABase,
) -> Result<RiscvInstruction, GenerationError> {
    RiscvInstruction::decode_asm_with_context(asm, isa_decode_context(isa_base)).map_err(|source| {
        GenerationError::StructuredInstructionBuildFailed {
            message: format!("failed to decode RVV constraint instruction `{asm}`: {source}"),
        }
    })
}

fn instruction_mnemonic(instruction: &RiscvInstruction) -> String {
    instruction
        .to_string()
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn is_random_rvv_instruction_mnemonic(mnemonic: &str) -> bool {
    mnemonic.starts_with('v') && !matches!(mnemonic, "vsetvli" | "vsetivli" | "vsetvl")
}

fn instruction_asm_parts(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim();
    let split = trimmed.find(char::is_whitespace)?;
    let mnemonic = &trimmed[..split];
    let operands = trimmed[split..].trim();
    Some((mnemonic, operands))
}

fn split_operands(operands: &str) -> Vec<String> {
    operands
        .split(',')
        .map(|operand| operand.trim().to_string())
        .collect()
}

fn join_instruction(mnemonic: &str, operands: &[String]) -> String {
    if operands.is_empty() {
        mnemonic.to_string()
    } else {
        format!("{} {}", mnemonic, operands.join(", "))
    }
}

fn parse_vector_operand(token: &str) -> Option<u8> {
    let token = token.trim();
    let token = token.strip_suffix(".t").unwrap_or(token);
    let number = token.strip_prefix('v')?.parse::<u8>().ok()?;
    (number < 32).then_some(number)
}

fn replace_vector_operand(token: &str, reg: u8) -> String {
    if token.trim().ends_with(".t") {
        format!("v{reg}.t")
    } else {
        format!("v{reg}")
    }
}

fn legal_group_candidates(range: (u8, u8), width: u8, align_power_of_two: bool) -> Vec<u8> {
    let (min, max) = range;
    (min..=max.min(31))
        .filter(|base| base.saturating_add(width.saturating_sub(1)) <= max.min(31))
        .filter(|base| !align_power_of_two || !matches!(width, 2 | 4 | 8) || base % width == 0)
        .collect()
}

fn choose_legal_group_base<R: rand::Rng>(
    rng: &mut R,
    range: (u8, u8),
    width: u8,
    align_power_of_two: bool,
) -> u8 {
    let candidates = legal_group_candidates(range, width, align_power_of_two);
    let candidates = if candidates.is_empty() {
        legal_group_candidates((1, 31), width, align_power_of_two)
    } else {
        candidates
    };
    candidates[rng.random_range(0..candidates.len())]
}

fn is_legal_group_base(base: u8, range: (u8, u8), width: u8, align_power_of_two: bool) -> bool {
    legal_group_candidates(range, width, align_power_of_two).contains(&base)
}

fn choose_distinct_vector_reg<R: rand::Rng>(rng: &mut R, range: (u8, u8), forbidden: &[u8]) -> u8 {
    let candidates = (range.0..=range.1.min(31))
        .filter(|reg| !forbidden.contains(reg))
        .collect::<Vec<_>>();
    let candidates = if candidates.is_empty() {
        (1u8..32)
            .filter(|reg| !forbidden.contains(reg))
            .collect::<Vec<_>>()
    } else {
        candidates
    };
    candidates[rng.random_range(0..candidates.len())]
}

fn parse_digits_after<'a>(value: &'a str, prefix: &str) -> Option<(u8, &'a str)> {
    let rest = value.strip_prefix(prefix)?;
    let digit_len = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .map(char::len_utf8)
        .sum::<usize>();
    if digit_len == 0 {
        return None;
    }
    let number = rest[..digit_len].parse::<u8>().ok()?;
    Some((number, &rest[digit_len..]))
}

fn whole_register_group_width(mnemonic: &str) -> Option<u8> {
    if let Some((nf, rest)) = parse_digits_after(mnemonic, "vl") {
        if rest.starts_with("re") {
            return Some(nf);
        }
    }
    if let Some((nf, rest)) = parse_digits_after(mnemonic, "vs") {
        if rest.starts_with("r.v") {
            return Some(nf);
        }
    }
    None
}

fn segment_field_count(mnemonic: &str) -> Option<u8> {
    let idx = mnemonic.find("seg")? + "seg".len();
    let rest = &mnemonic[idx..];
    let digit_len = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .map(char::len_utf8)
        .sum::<usize>();
    if digit_len == 0 {
        return None;
    }
    rest[..digit_len].parse::<u8>().ok()
}

fn is_indexed_segment_mnemonic(mnemonic: &str) -> bool {
    mnemonic.contains("xseg")
}

fn is_widening_mnemonic(mnemonic: &str) -> bool {
    (mnemonic.starts_with("vw") || mnemonic.starts_with("vfw")) && !mnemonic.contains("red")
}

fn is_narrowing_mnemonic(mnemonic: &str) -> bool {
    mnemonic.starts_with("vn") || mnemonic.starts_with("vfncvt")
}

fn is_extend_mnemonic(mnemonic: &str) -> bool {
    mnemonic.starts_with("vsext") || mnemonic.starts_with("vzext")
}

fn vector_operand_indices(operands: &[String]) -> Vec<usize> {
    operands
        .iter()
        .enumerate()
        .filter_map(|(idx, operand)| parse_vector_operand(operand).map(|_| idx))
        .collect()
}

fn repair_widening_or_narrowing_operands<R: rand::Rng>(
    mnemonic: &str,
    operands: &mut [String],
    vector_range: (u8, u8),
    rng: &mut R,
) -> bool {
    if !(is_widening_mnemonic(mnemonic)
        || is_narrowing_mnemonic(mnemonic)
        || is_extend_mnemonic(mnemonic))
    {
        return false;
    }

    let vector_indices = vector_operand_indices(operands);
    if vector_indices.len() < 2 {
        return false;
    }

    let dest_idx = vector_indices[0];
    let src_idx = vector_indices[1];
    let Some(original_dest) = parse_vector_operand(&operands[dest_idx]) else {
        return false;
    };
    let Some(original_src) = parse_vector_operand(&operands[src_idx]) else {
        return false;
    };

    let dest_width = if is_widening_mnemonic(mnemonic) { 2 } else { 1 };
    let src_width = if is_narrowing_mnemonic(mnemonic) {
        2
    } else {
        1
    };
    let dest_legal = is_legal_group_base(original_dest, vector_range, dest_width, true);
    let src_legal = is_legal_group_base(original_src, vector_range, src_width, true);
    let dest_group = (original_dest..original_dest.saturating_add(dest_width)).collect::<Vec<_>>();
    let src_group = (original_src..original_src.saturating_add(src_width)).collect::<Vec<_>>();
    let overlap = dest_group.iter().any(|reg| src_group.contains(reg));

    if dest_legal && src_legal && !overlap {
        return false;
    }

    let dest = if dest_legal {
        original_dest
    } else {
        choose_legal_group_base(rng, vector_range, dest_width, true)
    };
    let dest_group = (dest..dest.saturating_add(dest_width)).collect::<Vec<_>>();
    let src = if src_legal && !dest_group.contains(&original_src) {
        original_src
    } else if src_width > 1 {
        let mut candidates = legal_group_candidates(vector_range, src_width, true)
            .into_iter()
            .filter(|base| {
                let group = (*base..base.saturating_add(src_width)).collect::<Vec<_>>();
                group.iter().all(|reg| !dest_group.contains(reg))
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = legal_group_candidates((1, 31), src_width, true)
                .into_iter()
                .filter(|base| {
                    let group = (*base..base.saturating_add(src_width)).collect::<Vec<_>>();
                    group.iter().all(|reg| !dest_group.contains(reg))
                })
                .collect();
        }
        candidates[rng.random_range(0..candidates.len())]
    } else {
        choose_distinct_vector_reg(rng, vector_range, &dest_group)
    };

    let mut changed = false;
    if original_dest != dest {
        operands[dest_idx] = replace_vector_operand(&operands[dest_idx], dest);
        changed = true;
    }
    if original_src != src {
        operands[src_idx] = replace_vector_operand(&operands[src_idx], src);
        changed = true;
    }
    changed
}

fn repair_rvv_operand_line<R: rand::Rng>(
    line: &str,
    vector_range: (u8, u8),
    rng: &mut R,
) -> Option<String> {
    let (mnemonic, operand_text) = instruction_asm_parts(line)?;
    let mnemonic_lower = mnemonic.to_ascii_lowercase();
    if !is_random_rvv_instruction_mnemonic(&mnemonic_lower) {
        return None;
    }
    let mut operands = split_operands(operand_text);
    let mut changed = false;

    if let Some(width) = whole_register_group_width(&mnemonic_lower) {
        if !operands.is_empty() {
            if let Some(original_base) = parse_vector_operand(&operands[0]) {
                if !is_legal_group_base(original_base, vector_range, width, true) {
                    let base = choose_legal_group_base(rng, vector_range, width, true);
                    operands[0] = replace_vector_operand(&operands[0], base);
                    changed = true;
                }
            }
        }
    } else if let Some(nf) = segment_field_count(&mnemonic_lower) {
        if !operands.is_empty() {
            if let Some(original_base) = parse_vector_operand(&operands[0]) {
                let base = if is_legal_group_base(original_base, vector_range, nf, false) {
                    original_base
                } else {
                    let repaired = choose_legal_group_base(rng, vector_range, nf, false);
                    operands[0] = replace_vector_operand(&operands[0], repaired);
                    changed = true;
                    repaired
                };
                let field_group = (base..base.saturating_add(nf)).collect::<Vec<_>>();
                if is_indexed_segment_mnemonic(&mnemonic_lower) && operands.len() > 2 {
                    if let Some(original_index) = parse_vector_operand(&operands[2]) {
                        if field_group.contains(&original_index) {
                            let index = choose_distinct_vector_reg(rng, vector_range, &field_group);
                            operands[2] = replace_vector_operand(&operands[2], index);
                            changed = true;
                        }
                    }
                }
            }
        }
    } else if repair_widening_or_narrowing_operands(
        &mnemonic_lower,
        &mut operands,
        vector_range,
        rng,
    ) {
        changed = true;
    }

    changed.then(|| join_instruction(mnemonic, &operands))
}

fn legal_constraint_config_for_mnemonic<R: rand::Rng>(
    isa_base: ISABase,
    mnemonic: &str,
    rng: &mut R,
) -> RvvConfigConstraint {
    let max_sew = match isa_base {
        ISABase::Rv32 => 32,
        ISABase::Rv64 => 64,
    };
    // The legacy ambient path still uses m1 operand repair. Within that
    // supported layout, sample the selected opcode's legal SEW set instead
    // of pinning entire families (including vf2/vf4) to their maximum SEW.
    let eligible = [8u16, 16, 32, 64]
        .into_iter()
        .filter(|sew| {
            *sew <= max_sew && rvv_legal::opcode_config_eligible(mnemonic, *sew, "m1", 64)
        })
        .collect::<Vec<_>>();
    let sew = if mnemonic == "vfmv.f.s" {
        32 // Preserve the existing NaN-boxing probe's selected width.
    } else {
        eligible[rng.random_range(0..eligible.len())]
    };
    let vl_choices: &[u8] = if mnemonic == "vfmv.f.s" {
        &[1]
    } else if mnemonic.contains("seg") || mnemonic.contains("ff") {
        &[1, 2]
    } else {
        &[1, 2, 3, 4]
    };
    let tail_policy = if rng.random::<bool>() { "ta" } else { "tu" };
    let mask_policy = if rng.random::<bool>() { "ma" } else { "mu" };
    RvvConfigConstraint::new(
        isa_base,
        sew,
        vl_choices[rng.random_range(0..vl_choices.len())],
        "m1",
        tail_policy,
        mask_policy,
    )
}

fn rvv_constraint_prefix_lines<R: rand::Rng>(
    isa_base: ISABase,
    mnemonic: &str,
    rng: &mut R,
) -> Vec<String> {
    let config = legal_constraint_config_for_mnemonic(isa_base, mnemonic, rng);
    vec![
        format!(
            "vsetivli x0, {}, {}",
            config.vl,
            config.vsetivli_vtype_immediate()
        ),
        "csrrw x0, vstart, x0".to_string(),
    ]
}

fn global_rvv_constraint_prefix_lines(isa_base: ISABase) -> Vec<String> {
    let config = RvvConfigConstraint::new(isa_base, 32, 4, "m1", "tu", "mu");
    vec![
        format!(
            "vsetivli x0, {}, {}",
            config.vl,
            config.vsetivli_vtype_immediate()
        ),
        "csrrw x0, vstart, x0".to_string(),
    ]
}

fn repair_sequence_operands<R: rand::Rng>(
    sequence: &mut InstructionSequence<RiscvInstruction>,
    isa_base: ISABase,
    vector_range: (u8, u8),
    rng: &mut R,
) -> Result<Option<bool>, GenerationError> {
    let mnemonic = instruction_mnemonic(&sequence.instruction);
    if !is_random_rvv_instruction_mnemonic(&mnemonic) {
        return Ok(None);
    }

    let original_line = sequence.instruction.to_string();
    let mut operand_repaired = false;
    if let Some(repaired_line) = repair_rvv_operand_line(&original_line, vector_range, rng) {
        let repaired_instruction = decode_constraint_instruction(&repaired_line, isa_base)?;
        sequence.instruction = repaired_instruction;
        operand_repaired = true;
    }

    Ok(Some(operand_repaired))
}

fn constrain_sequence<R: rand::Rng>(
    sequence: &mut InstructionSequence<RiscvInstruction>,
    isa_base: ISABase,
    vector_range: (u8, u8),
    rng: &mut R,
) -> Result<Option<(usize, bool)>, GenerationError> {
    let Some(operand_repaired) = repair_sequence_operands(sequence, isa_base, vector_range, rng)?
    else {
        return Ok(None);
    };
    let mnemonic = instruction_mnemonic(&sequence.instruction);
    let mut prefix = rvv_constraint_prefix_lines(isa_base, &mnemonic, rng)
        .iter()
        .map(|line| decode_constraint_instruction(line, isa_base))
        .collect::<Result<Vec<_>, _>>()?;
    let inserted = prefix.len();
    prefix.append(&mut sequence.pre_instructions);
    sequence.pre_instructions = prefix;
    Ok(Some((inserted, operand_repaired)))
}

fn constrain_random_rvv_testcase_with_first_hart_prefix_skip(
    testcase: &mut GeneratedTestCase,
    first_hart_test_prefix_skip: usize,
) -> Result<RvvConstraintInsertionReport, GenerationError> {
    let seed = testcase.config.seed.unwrap_or(0) ^ 0x4353_5452_5643_0001;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut report = RvvConstraintInsertionReport::default();

    let vector_range = testcase
        .config
        .random_config
        .register_operand_config
        .vector_register_number_range;

    for (hart_idx, hart) in testcase.harts.iter_mut().enumerate() {
        let skip = if hart_idx == 0 {
            first_hart_test_prefix_skip
        } else {
            0
        };
        for sequence in hart.sequences_mut(HartPhase::Test).iter_mut().skip(skip) {
            if let Some((inserted, operand_repaired)) =
                constrain_sequence(sequence, testcase.config.isa_base, vector_range, &mut rng)?
            {
                report.constrained_instructions += 1;
                report.inserted_instructions += inserted;
                if operand_repaired {
                    report.operand_repaired_instructions += 1;
                }
            }
        }
    }

    Ok(report)
}

fn constrain_random_rvv_testcase_with_global_prefix(
    testcase: &mut GeneratedTestCase,
) -> Result<RvvConstraintInsertionReport, GenerationError> {
    let seed = testcase.config.seed.unwrap_or(0) ^ 0x4353_5452_5643_0001;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut report = RvvConstraintInsertionReport::default();

    let isa_base = testcase.config.isa_base;
    let vector_range = testcase
        .config
        .random_config
        .register_operand_config
        .vector_register_number_range;

    for hart in testcase.harts.iter_mut() {
        let sequences = hart.sequences_mut(HartPhase::Test);
        let mut first_rvv_idx = None;
        for (idx, sequence) in sequences.iter_mut().enumerate() {
            if let Some(operand_repaired) =
                repair_sequence_operands(sequence, isa_base, vector_range, &mut rng)?
            {
                first_rvv_idx.get_or_insert(idx);
                report.constrained_instructions += 1;
                if operand_repaired {
                    report.operand_repaired_instructions += 1;
                }
            }
        }

        if let Some(idx) = first_rvv_idx {
            let mut prefix = global_rvv_constraint_prefix_lines(isa_base)
                .iter()
                .map(|line| decode_constraint_instruction(line, isa_base))
                .collect::<Result<Vec<_>, _>>()?;
            let inserted = prefix.len();
            prefix.append(&mut sequences[idx].pre_instructions);
            sequences[idx].pre_instructions = prefix;
            report.inserted_instructions += inserted;
        }
    }

    Ok(report)
}

pub fn constrain_random_rvv_testcase(
    testcase: &mut GeneratedTestCase,
) -> Result<RvvConstraintInsertionReport, GenerationError> {
    constrain_random_rvv_testcase_with_global_prefix(testcase)
}

pub fn constrain_random_rvv_testcase_after_first_hart_prefix(
    testcase: &mut GeneratedTestCase,
    first_hart_test_prefix_len: usize,
) -> Result<RvvConstraintInsertionReport, GenerationError> {
    constrain_random_rvv_testcase_with_first_hart_prefix_skip(testcase, first_hart_test_prefix_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repaired_line(asm: &str, seed: u64) -> String {
        let mut rng = StdRng::seed_from_u64(seed);
        repair_rvv_operand_line(asm, (13, 31), &mut rng).expect("line should be repaired")
    }

    fn first_vector_reg(asm: &str) -> u8 {
        let (_, operands) = instruction_asm_parts(asm).expect("asm parts");
        parse_vector_operand(&split_operands(operands)[0]).expect("first vector operand")
    }

    #[test]
    fn operand_repair_aligns_whole_register_groups() {
        let repaired = repaired_line("vs4r.v v19, 0(x12)", 1);
        decode_constraint_instruction(&repaired, ISABase::Rv64).expect("repaired line decodes");
        let base = first_vector_reg(&repaired);
        assert_eq!(base % 4, 0, "{repaired}");
        assert!(base + 3 <= 31, "{repaired}");

        let repaired = repaired_line("vl8re64.v v15, (x10)", 2);
        decode_constraint_instruction(&repaired, ISABase::Rv64).expect("repaired line decodes");
        let base = first_vector_reg(&repaired);
        assert_eq!(base % 8, 0, "{repaired}");
        assert!(base + 7 <= 31, "{repaired}");
    }

    #[test]
    fn operand_repair_bounds_segment_register_groups() {
        let repaired = repaired_line("vlseg7e32.v v29, (x10)", 3);
        decode_constraint_instruction(&repaired, ISABase::Rv64).expect("repaired line decodes");
        let base = first_vector_reg(&repaired);
        assert!(base + 6 <= 31, "{repaired}");

        let repaired = repaired_line("vsoxseg6ei16.v v28, (x8), v29, v0.t", 4);
        decode_constraint_instruction(&repaired, ISABase::Rv64).expect("repaired line decodes");
        let (_, operands) = instruction_asm_parts(&repaired).expect("asm parts");
        let operands = split_operands(operands);
        let base = parse_vector_operand(&operands[0]).expect("segment base");
        let index = parse_vector_operand(&operands[2]).expect("index register");
        assert!(base + 5 <= 31, "{repaired}");
        assert!(!(base..base + 6).contains(&index), "{repaired}");
    }

    #[test]
    fn operand_repair_separates_widening_operands() {
        let repaired = repaired_line("vwaddu.vx v23, v17, x13, v0.t", 5);
        decode_constraint_instruction(&repaired, ISABase::Rv64).expect("repaired line decodes");
        let (_, operands) = instruction_asm_parts(&repaired).expect("asm parts");
        let operands = split_operands(operands);
        let dest = parse_vector_operand(&operands[0]).expect("dest");
        let src = parse_vector_operand(&operands[1]).expect("src");
        assert_eq!(dest % 2, 0, "{repaired}");
        assert_ne!(dest, src, "{repaired}");
    }

    #[test]
    fn vfmv_f_s_constraint_preserves_nan_box_source_width() {
        let mut rng = StdRng::seed_from_u64(2);
        let config = legal_constraint_config_for_mnemonic(ISABase::Rv64, "vfmv.f.s", &mut rng);
        assert_eq!(config.sew, 32);
        assert_eq!(config.vl, 1);
    }

    #[test]
    fn extension_config_sampling_reaches_each_supported_sew() {
        let mut rng = StdRng::seed_from_u64(0x5643_4d4f_4445_4c);
        let mut vf2 = std::collections::BTreeSet::new();
        let mut vf4 = std::collections::BTreeSet::new();
        for _ in 0..64 {
            vf2.insert(
                legal_constraint_config_for_mnemonic(ISABase::Rv64, "vzext.vf2", &mut rng).sew,
            );
            vf4.insert(
                legal_constraint_config_for_mnemonic(ISABase::Rv64, "vzext.vf4", &mut rng).sew,
            );
        }
        assert_eq!(vf2, [16, 32, 64].into_iter().collect());
        assert_eq!(vf4, [32, 64].into_iter().collect());
    }

    #[test]
    fn rvv_constraint_prefixes_decode() {
        let mut rng = StdRng::seed_from_u64(1);
        for mnemonic in ["vadd.vv", "vfadd.vv", "vzext.vf8", "vlseg2e8ff.v"] {
            for line in rvv_constraint_prefix_lines(ISABase::Rv64, mnemonic, &mut rng) {
                decode_constraint_instruction(&line, ISABase::Rv64)
                    .unwrap_or_else(|err| panic!("{line} should decode: {err:?}"));
            }
        }
    }
}
