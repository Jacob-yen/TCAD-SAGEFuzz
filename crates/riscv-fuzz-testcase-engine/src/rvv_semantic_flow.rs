//! Optional VC-backed SF demo. An SF goal fixes the dataflow roles; VC's
//! opcode domain supplies the execution configuration and validates each
//! constructed instruction before it enters the program.

use std::env;

use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use riscv_fuzz_plugin_api::ISABase;
use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::{AsmDecodeWithContext, InstructionSequence, IsaDecodeContext};

use crate::{
    GenerationError,
    rvv_catalog_legal::{RvvSamplingProfile, check_asm_under_config, domain_for_asm},
    rvv_constraints::RvvConfigConstraint,
    rvv_legal,
    testcase::{GeneratedTestCase, HartPhase},
};

pub const TARGET_ENV: &str = "RISCV_FUZZ_RVV_SEMANTIC_TARGET";
pub const BEGIN_MARKER: &str = "addi x0, x0, 1451";
pub const END_MARKER: &str = "addi x0, x0, 1452";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlowGoal {
    Data,
    Mask,
    Config,
}

impl FlowGoal {
    fn name(self) -> &'static str {
        match self {
            Self::Data => "data",
            Self::Mask => "mask",
            Self::Config => "config",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "data" => Some(Self::Data),
            "mask" => Some(Self::Mask),
            "config" => Some(Self::Config),
            _ => None,
        }
    }
}

fn error(message: impl Into<String>) -> GenerationError {
    GenerationError::StructuredInstructionBuildFailed {
        message: message.into(),
    }
}

fn decode(line: &str) -> Result<RiscvInstruction, GenerationError> {
    RiscvInstruction::decode_asm_with_context(line, IsaDecodeContext::RV64)
        .map_err(|source| error(format!("semantic flow cannot decode `{line}`: {source}")))
}

fn config_choice(
    raw: &str,
    domain: &crate::rvv_catalog_legal::RvvOpcodeDomain,
    rng: &mut StdRng,
) -> Result<(FlowGoal, u16, &'static str), GenerationError> {
    if raw == "auto" {
        let goal = [FlowGoal::Data, FlowGoal::Mask, FlowGoal::Config][rng.random_range(0..3)];
        let selected = &domain.configurations[rng.random_range(0..domain.configurations.len())];
        return Ok((goal, selected.sew, selected.lmul));
    }
    let tokens = raw.split(':').collect::<Vec<_>>();
    if tokens.len() != 3 {
        return Err(error(format!("{TARGET_ENV} must be auto or goal:SEW:LMUL")));
    }
    let goal = FlowGoal::parse(tokens[0])
        .ok_or_else(|| error(format!("unknown semantic flow goal: {}", tokens[0])))?;
    let sew = tokens[1]
        .parse::<u16>()
        .map_err(|_| error(format!("invalid semantic flow SEW: {}", tokens[1])))?;
    let selected = domain
        .configurations
        .iter()
        .find(|cfg| cfg.sew == sew && cfg.lmul == tokens[2])
        .ok_or_else(|| {
            error(format!(
                "semantic flow configuration e{sew},{} has no legal domain",
                tokens[2]
            ))
        })?;
    Ok((goal, selected.sew, selected.lmul))
}

fn choose_groups(
    cfg: RvvConfigConstraint,
    range: (u8, u8),
    rng: &mut StdRng,
) -> Result<(u8, u8), GenerationError> {
    let width = rvv_legal::register_width(cfg.sew, cfg.lmul, cfg.sew)
        .ok_or_else(|| error("semantic flow has invalid register group width"))?;
    let mut bases = rvv_legal::legal_group_bases(range, width);
    if bases.len() < 2 {
        bases = rvv_legal::legal_group_bases((1, 31), width);
    }
    let pairs = bases
        .iter()
        .copied()
        .flat_map(|src| {
            bases
                .iter()
                .copied()
                .filter(move |dst| !rvv_legal::groups_overlap(src, width, *dst, width))
                .map(move |dst| (src, dst))
        })
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return Err(error(
            "semantic flow cannot allocate two disjoint legal vector groups",
        ));
    }
    Ok(pairs[rng.random_range(0..pairs.len())])
}

/// Insert one explicit producer-consumer-observer relation before the older
/// SF/VC/catalog blocks. No candidate or opcode is discarded after Spike.
pub fn maybe_inject(
    testcase: &mut GeneratedTestCase,
    subject: &str,
    insertion: usize,
) -> Result<usize, GenerationError> {
    let Some(raw) = env::var(TARGET_ENV).ok() else {
        return Ok(0);
    };
    let profile = RvvSamplingProfile::from_subject_and_env(subject).map_err(error)?;
    let seed = testcase
        .config
        .seed
        .ok_or_else(|| error("semantic flow requires a seed"))?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5346_5643_444f_4d41);
    let range = testcase
        .config
        .random_config
        .register_operand_config
        .vector_register_number_range;
    let domain = domain_for_asm("vadd.vx v16, v8, x13", profile, range)
        .ok_or_else(|| error("cannot derive semantic flow opcode domain"))?;
    if domain.configurations.is_empty() {
        return Err(error("semantic flow has an empty VC opcode domain"));
    }
    let (goal, sew, lmul) = config_choice(raw.trim(), &domain, &mut rng)?;
    let choice = domain
        .configurations
        .iter()
        .find(|cfg| cfg.sew == sew && cfg.lmul == lmul)
        .expect("selected configuration came from the domain");
    let min_vl = 2u8;
    if choice.sampled_avl_max < min_vl {
        return Err(error(format!(
            "semantic flow needs at least two active elements at e{sew},{lmul}"
        )));
    }
    let vl = rng.random_range(min_vl..=choice.sampled_avl_max.min(4));
    let cfg = RvvConfigConstraint::new(
        ISABase::Rv64,
        sew,
        vl,
        lmul,
        if rng.random_bool(0.5) { "ta" } else { "tu" },
        if rng.random_bool(0.5) { "ma" } else { "mu" },
    );
    let (src, dst) = choose_groups(cfg, range, &mut rng)?;
    let alu = ["vadd.vx", "vxor.vx", "vsub.vx", "vmaxu.vx"][rng.random_range(0..4)];
    let (start, end) = testcase.mem_range.get(subject).copied().ok_or_else(|| {
        GenerationError::MemRangeNotFound {
            core_id: subject.to_string(),
        }
    })?;
    let source_addr = if end.saturating_sub(start) >= 0x400 {
        start + 0x200
    } else {
        start
    };
    let source_addr = source_addr.div_ceil(8) * 8;
    let output_addr = source_addr + 0x40;
    if output_addr + 0x68 >= end {
        return Err(error(
            "semantic flow needs a writable 0xa8-byte memory window",
        ));
    }
    let lane_pattern = if sew == 64 { 0 } else { 1u64 << sew };
    let mut lines = vec![
        BEGIN_MARKER.to_string(),
        format!("li x11, {source_addr}"),
        format!("li x12, {output_addr}"),
        format!("li x13, {}", rng.random_range(1..=15)),
        format!("li x15, {lane_pattern}"),
        "sd x15, 0(x11)".to_string(),
        "li x15, 1".to_string(),
        "sd x15, 8(x11)".to_string(),
        "sd x0, 0(x12)".to_string(),
        "sd x0, 8(x12)".to_string(),
        format!("vsetivli x6, {vl}, {}", cfg.vsetivli_vtype_immediate()),
        "csrrw x0, vstart, x0".to_string(),
    ];
    let mut vector_lines = vec![format!("vle{sew}.v v{src}, (x11)")];
    if goal == FlowGoal::Config {
        lines.push("csrr x7, vl".to_string());
        lines.push("sd x7, 64(x12)".to_string());
        lines.push("ld x16, 64(x12)".to_string());
    }
    if goal == FlowGoal::Mask {
        vector_lines.push(format!("vmseq.vi v0, v{src}, 0"));
    }
    vector_lines.push(format!(
        "{alu} v{dst}, v{src}, x13{}",
        if goal == FlowGoal::Mask { ", v0.t" } else { "" }
    ));
    vector_lines.push(format!("vse{sew}.v v{dst}, (x12)"));
    for line in &vector_lines {
        let issues = check_asm_under_config(line, cfg, profile.elen_bits);
        if !issues.is_empty() {
            return Err(error(format!(
                "VC invariant failed for semantic flow `{line}`: {issues:?}"
            )));
        }
    }
    lines.extend(vector_lines);
    lines.push("ld x14, 0(x12)".to_string());
    lines.push(END_MARKER.to_string());
    let instructions = lines
        .iter()
        .map(|line| decode(line).map(InstructionSequence::new))
        .collect::<Result<Vec<_>, _>>()?;
    let len = instructions.len();
    let hart = &mut testcase.harts[0];
    hart.sequences_mut(HartPhase::Test)
        .splice(insertion..insertion, instructions);
    let shifted = hart
        .shared_sequence_indices(HartPhase::Test)
        .iter()
        .map(|index| {
            if *index >= insertion {
                *index + len
            } else {
                *index
            }
        })
        .collect();
    *hart.shared_sequence_indices_mut(HartPhase::Test) = shifted;
    eprintln!(
        "RVV semantic-flow goal={} sew={} lmul={} vl={} alu={} src=v{} dst=v{} seed={}",
        goal.name(),
        sew,
        lmul,
        vl,
        alu,
        src,
        dst,
        seed,
    );
    Ok(len)
}
