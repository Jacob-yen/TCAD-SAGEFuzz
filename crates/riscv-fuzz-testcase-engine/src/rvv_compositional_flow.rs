//! Compositional RVV state-flow generation with VC-checked instructions.

use crate::{
    GenerationError,
    rvv_catalog_legal::{RvvSamplingProfile, check_asm_under_config, domain_for_asm},
    rvv_constraints::RvvConfigConstraint,
    rvv_legal,
    testcase::{GeneratedTestCase, HartPhase},
};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use riscv_fuzz_plugin_api::ISABase;
use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::{AsmDecodeWithContext, InstructionSequence, IsaDecodeContext};
use serde::{Deserialize, Serialize};
use std::env;

pub const TARGET_ENV: &str = "RISCV_FUZZ_RVV_COMPOSITION";
pub const BEGIN_MARKER: &str = "addi x0, x0, 1461";
pub const END_MARKER: &str = "addi x0, x0, 1462";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Producer {
    Unit,
    Stride,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Predicate {
    None,
    Eq,
    Ne,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MaskStep {
    None,
    First,
    Second,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Operator {
    Add,
    Xor,
    Sub,
    Maxu,
}
impl Operator {
    fn mnemonic(self) -> &'static str {
        match self {
            Self::Add => "vadd.vx",
            Self::Xor => "vxor.vx",
            Self::Sub => "vsub.vx",
            Self::Maxu => "vmaxu.vx",
        }
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Alias {
    Inplace,
    Disjoint,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Observer {
    Memory,
    Scalar,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompositionSpec {
    pub producer: Producer,
    pub predicate: Predicate,
    pub mask_step: MaskStep,
    pub op1: Operator,
    pub op2: Option<Operator>,
    pub alias1: Alias,
    pub alias2: Option<Alias>,
    pub observer: Observer,
    pub control_readback: bool,
    pub sew: u16,
    pub lmul: String,
    pub vl: u8,
}

fn error(message: impl Into<String>) -> GenerationError {
    GenerationError::StructuredInstructionBuildFailed {
        message: message.into(),
    }
}
fn decode(line: &str) -> Result<RiscvInstruction, GenerationError> {
    RiscvInstruction::decode_asm_with_context(line, IsaDecodeContext::RV64).map_err(|source| {
        error(format!(
            "compositional flow cannot decode `{line}`: {source}"
        ))
    })
}
fn disjoint_base(
    previous: u8,
    width: u8,
    bases: &[u8],
    rng: &mut StdRng,
) -> Result<u8, GenerationError> {
    let choices = bases
        .iter()
        .copied()
        .filter(|base| !rvv_legal::groups_overlap(*base, width, previous, width))
        .collect::<Vec<_>>();
    if choices.is_empty() {
        return Err(error("no disjoint vector group for SF edge"));
    }
    Ok(choices[rng.random_range(0..choices.len())])
}
fn validate_shape(spec: &CompositionSpec) -> Result<(), GenerationError> {
    if spec.op2.is_some() != spec.alias2.is_some() {
        return Err(error("op2 and alias2 must both exist or both be absent"));
    }
    match (spec.predicate, spec.mask_step, spec.op2.is_some()) {
        (Predicate::None, MaskStep::None, _) => Ok(()),
        (Predicate::Eq | Predicate::Ne, MaskStep::First, _) => Ok(()),
        (Predicate::Eq | Predicate::Ne, MaskStep::Second, true) => Ok(()),
        _ => Err(error("mask producer and consumer step are inconsistent")),
    }
}

/// The SF graph fixes producer/consumer/observer edges. VC supplies the
/// configuration and checks each concrete vector instruction before insertion.
pub fn maybe_inject(
    testcase: &mut GeneratedTestCase,
    subject: &str,
    insertion: usize,
) -> Result<usize, GenerationError> {
    let Some(raw) = env::var(TARGET_ENV).ok() else {
        return Ok(0);
    };
    let spec: CompositionSpec = serde_json::from_str(&raw)
        .map_err(|source| error(format!("invalid {TARGET_ENV} JSON: {source}")))?;
    validate_shape(&spec)?;
    let profile = RvvSamplingProfile::from_subject_and_env(subject).map_err(error)?;
    let seed = testcase
        .config
        .seed
        .ok_or_else(|| error("composition requires a seed"))?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x434f_4d50_4f53_4544);
    let vector_range = testcase
        .config
        .random_config
        .register_operand_config
        .vector_register_number_range;
    let domain = domain_for_asm(
        &format!("{} v16, v8, x13", spec.op1.mnemonic()),
        profile,
        vector_range,
    )
    .ok_or_else(|| error("cannot derive VC domain for composition"))?;
    let choice = domain
        .configurations
        .iter()
        .find(|item| item.sew == spec.sew && item.lmul == spec.lmul)
        .ok_or_else(|| error(format!("no VC domain for e{},{}", spec.sew, spec.lmul)))?;
    if spec.vl < 2 || spec.vl > choice.sampled_avl_max.min(4) {
        return Err(error("composition VL must be in 2..=min(VLMAX,4)"));
    }
    let cfg = RvvConfigConstraint::new(ISABase::Rv64, spec.sew, spec.vl, choice.lmul, "tu", "mu");
    let width = rvv_legal::register_width(cfg.sew, cfg.lmul, cfg.sew)
        .ok_or_else(|| error("invalid composition register width"))?;
    let mut bases = rvv_legal::legal_group_bases(vector_range, width);
    if bases.len() < 2 {
        bases = rvv_legal::legal_group_bases((1, 31), width);
    }
    if bases.len() < 2 {
        return Err(error("composition needs two vector groups"));
    }
    let source = bases[rng.random_range(0..bases.len())];
    let first = match spec.alias1 {
        Alias::Inplace => source,
        Alias::Disjoint => disjoint_base(source, width, &bases, &mut rng)?,
    };
    let second = match spec.alias2 {
        None => None,
        Some(Alias::Inplace) => Some(first),
        Some(Alias::Disjoint) => Some(disjoint_base(first, width, &bases, &mut rng)?),
    };
    let final_base = second.unwrap_or(first);
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
    let output_addr = source_addr + 0x80;
    if output_addr + 0x68 >= end {
        return Err(error("composition needs a writable memory window"));
    }
    let elem_bytes = cfg.sew / 8;
    let stride = elem_bytes
        * if spec.producer == Producer::Stride {
            2
        } else {
            1
        };
    let scalar_store = match cfg.sew {
        8 => "sb",
        16 => "sh",
        32 => "sw",
        64 => "sd",
        _ => return Err(error("unsupported SEW")),
    };
    let scalar_load = match cfg.sew {
        8 => "lbu",
        16 => "lhu",
        32 => "lwu",
        64 => "ld",
        _ => unreachable!(),
    };
    let k1: u8 = rng.random_range(1..=15);
    let k2: u8 = rng.random_range(1..=15);
    let mut lines = vec![
        BEGIN_MARKER.to_string(),
        format!("li x11, {source_addr}"),
        format!("li x12, {output_addr}"),
        format!("li x9, {stride}"),
    ];
    for element in 0..4 {
        lines.push(format!("li x15, {element}"));
        lines.push(format!("{scalar_store} x15, {}(x11)", element * stride));
    }
    for offset in [0, 8, 16, 24] {
        lines.push(format!("sd x0, {offset}(x12)"));
    }
    lines.extend([
        format!(
            "vsetivli x6, {}, {}",
            cfg.vl,
            cfg.vsetivli_vtype_immediate()
        ),
        "csrrw x0, vstart, x0".to_string(),
    ]);
    let mut vector_lines = vec![match spec.producer {
        Producer::Unit => format!("vle{}.v v{source}, (x11)", cfg.sew),
        Producer::Stride => format!("vlse{}.v v{source}, (x11), x9", cfg.sew),
    }];
    match spec.predicate {
        Predicate::None => {}
        Predicate::Eq => vector_lines.push(format!("vmseq.vi v0, v{source}, 0")),
        Predicate::Ne => vector_lines.push(format!("vmsne.vi v0, v{source}, 0")),
    }
    if spec.alias1 == Alias::Disjoint && spec.mask_step == MaskStep::First {
        vector_lines.push(format!("vmv.v.i v{first}, 5"));
    }
    vector_lines.push(format!("li x13, {k1}"));
    vector_lines.push(format!(
        "{} v{first}, v{source}, x13{}",
        spec.op1.mnemonic(),
        if spec.mask_step == MaskStep::First {
            ", v0.t"
        } else {
            ""
        }
    ));
    if let (Some(op2), Some(second)) = (spec.op2, second) {
        if spec.alias2 == Some(Alias::Disjoint) && spec.mask_step == MaskStep::Second {
            vector_lines.push(format!("vmv.v.i v{second}, 7"));
        }
        vector_lines.push(format!("li x13, {k2}"));
        vector_lines.push(format!(
            "{} v{second}, v{first}, x13{}",
            op2.mnemonic(),
            if spec.mask_step == MaskStep::Second {
                ", v0.t"
            } else {
                ""
            }
        ));
    }
    vector_lines.push(format!("vse{}.v v{final_base}, (x12)", cfg.sew));
    for line in &vector_lines {
        if line.starts_with('v') {
            let issues = check_asm_under_config(line, cfg, profile.elen_bits);
            if !issues.is_empty() {
                return Err(error(format!(
                    "VC precondition failed for `{line}`: {issues:?}"
                )));
            }
        }
    }
    lines.extend(vector_lines);
    lines.push(format!("{scalar_load} x14, 0(x12)"));
    lines.push(format!("{scalar_load} x15, {elem_bytes}(x12)"));
    if spec.observer == Observer::Scalar {
        let extraction = format!("vmv.x.s x17, v{final_base}");
        let issues = check_asm_under_config(&extraction, cfg, profile.elen_bits);
        if !issues.is_empty() {
            return Err(error(format!(
                "VC precondition failed for `{extraction}`: {issues:?}"
            )));
        }
        lines.push(extraction);
        lines.push("sd x17, 64(x12)".to_string());
    }
    if spec.control_readback {
        lines.push("csrr x18, vl".to_string());
        lines.push("sd x18, 80(x12)".to_string());
        lines.push("ld x19, 80(x12)".to_string());
    }
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
    let selection = serde_json::json!({"spec": spec, "src": source, "first": first, "second": second, "scalar1": k1, "scalar2": k2, "marker_begin": 1461, "marker_end": 1462});
    eprintln!("RVV composition {selection}");
    Ok(len)
}
