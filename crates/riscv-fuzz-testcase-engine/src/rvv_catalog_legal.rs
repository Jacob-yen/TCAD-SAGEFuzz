//! Context-aware instantiation of a *selected* base-V grammar opcode.
//! Failed operand/configuration draws never substitute a different opcode.

use std::collections::BTreeSet;

use rand::{Rng, SeedableRng as _, rngs::StdRng};
use riscv_fuzz_plugin_api::ISABase;
use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::{AsmDecodeWithContext, InstructionSequence, IsaDecodeContext};
use serde::Serialize;

use crate::{
    GenerationError, rvv_artifact_settings, rvv_constraints::RvvConfigConstraint, rvv_legal,
    rvv_spike_guard_model,
};

#[derive(Clone, Copy)]
struct Footprint {
    operand: usize,
    align: u8,
    span: u8,
    eew: u16,
    fixed_mask: bool,
    allow_zero: bool,
}

/// A concrete execution profile and a *sampling* budget. The architectural
/// VL range is reported separately from the bounded AVL range we exercise.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct RvvSamplingProfile {
    pub elen_bits: u16,
    pub vlen_bits: u32,
    pub sampled_avl_cap: u8,
    pub memory_avl_cap: u8,
    pub alias_percent: u8,
    pub original_operand_percent: u8,
    /// Cycle through every eligible SEW/LMUL before repeating; each cycle's
    /// order is seed-permuted. The alternative is independent random draws.
    pub config_sampling: &'static str,
}

impl RvvSamplingProfile {
    pub fn for_subject(subject: &str) -> Self {
        Self {
            elen_bits: rvv_artifact_settings::ELEN_BITS,
            vlen_bits: if subject.eq_ignore_ascii_case("ara") {
                rvv_artifact_settings::ARA_VLEN_BITS
            } else {
                rvv_artifact_settings::XIANGSHAN_VLEN_BITS
            },
            sampled_avl_cap: 31,
            memory_avl_cap: 4,
            alias_percent: 25,
            original_operand_percent: 50,
            config_sampling: "cycle",
        }
    }

    pub fn from_subject_and_env(subject: &str) -> Result<Self, String> {
        if !matches!(subject.to_ascii_lowercase().as_str(), "ara" | "xiangshan") {
            return Err(format!(
                "no RVV VLEN profile is recorded for subject `{subject}`"
            ));
        }
        let mut profile = Self::for_subject(subject);
        fn override_value<T: std::str::FromStr>(name: &str, current: T) -> Result<T, String> {
            match std::env::var(name) {
                Ok(raw) => raw.parse().map_err(|_| format!("invalid {name}={raw}")),
                Err(std::env::VarError::NotPresent) => Ok(current),
                Err(error) => Err(format!("cannot read {name}: {error}")),
            }
        }
        profile.sampled_avl_cap =
            override_value("RISCV_FUZZ_RVV_AVL_CAP", profile.sampled_avl_cap)?;
        profile.memory_avl_cap =
            override_value("RISCV_FUZZ_RVV_MEMORY_AVL_CAP", profile.memory_avl_cap)?;
        profile.alias_percent =
            override_value("RISCV_FUZZ_RVV_ALIAS_PERCENT", profile.alias_percent)?;
        profile.original_operand_percent = override_value(
            "RISCV_FUZZ_RVV_ORIGINAL_OPERAND_PERCENT",
            profile.original_operand_percent,
        )?;
        profile.config_sampling = match std::env::var("RISCV_FUZZ_RVV_CONFIG_SAMPLING") {
            Ok(value) if value == "random" => "random",
            Ok(value) if value == "cycle" => "cycle",
            Ok(value) => return Err(format!("invalid RISCV_FUZZ_RVV_CONFIG_SAMPLING={value}")),
            Err(std::env::VarError::NotPresent) => profile.config_sampling,
            Err(error) => {
                return Err(format!(
                    "cannot read RISCV_FUZZ_RVV_CONFIG_SAMPLING: {error}"
                ));
            }
        };
        if profile.elen_bits < 8
            || !profile.elen_bits.is_power_of_two()
            || profile.vlen_bits < u32::from(profile.elen_bits)
            || profile.vlen_bits > 65_536
            || !profile.vlen_bits.is_power_of_two()
            || !(1..=31).contains(&profile.sampled_avl_cap)
            || !(1..=31).contains(&profile.memory_avl_cap)
            || profile.alias_percent > 100
            || profile.original_operand_percent > 100
        {
            return Err("invalid RVV profile: require power-of-two VLEN>=ELEN>=8, AVL caps 1..31, and probabilities 0..100".into());
        }
        if profile.elen_bits != 64 {
            return Err("this base-V sampler currently models ELEN=64 only".into());
        }
        Ok(profile)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RvvOperandDomain {
    pub operand_index: usize,
    pub eew: u16,
    pub register_span: u8,
    pub register_alignment: u8,
    pub legal_base_registers: Vec<u8>,
    pub fixed_mask: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RvvConfigDomain {
    pub sew: u16,
    pub lmul: &'static str,
    pub architectural_vl_max: u32,
    pub sampled_avl_min: u8,
    pub sampled_avl_max: u8,
    pub operands: Vec<RvvOperandDomain>,
    /// Joint-feasible classes for the first two non-fixed vector operands.
    pub first_pair_alias_classes: Vec<String>,
}

/// The supported domain of one already-selected grammar instruction. Operand
/// base lists are projections; register-pair overlap rules are applied when
/// choosing a joint assignment, so their Cartesian product is not implied.
#[derive(Debug, Clone, Serialize)]
pub struct RvvOpcodeDomain {
    pub mnemonic: String,
    pub profile: RvvSamplingProfile,
    pub configurations: Vec<RvvConfigDomain>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RvvPreconditionIssue {
    pub code: &'static str,
    pub detail: String,
}

/// Check an existing instruction against the same operand footprint and
/// register relation rules used when constructing a catalog target. This is
/// diagnostic: callers must retain the original program and its outcome.
pub fn check_asm_under_config(
    asm: &str,
    cfg: RvvConfigConstraint,
    elen_bits: u16,
) -> Vec<RvvPreconditionIssue> {
    let Some((mnemonic, args)) = asm.trim().split_once(char::is_whitespace) else {
        return vec![RvvPreconditionIssue {
            code: "instruction_shape",
            detail: format!("instruction has no operands: {asm}"),
        }];
    };
    if mnemonic.starts_with("vset") {
        return Vec::new();
    }
    let mut issues = Vec::new();
    if !rvv_legal::opcode_config_eligible(mnemonic, cfg.sew, cfg.lmul, elen_bits) {
        issues.push(RvvPreconditionIssue {
            code: "opcode_vtype",
            detail: format!(
                "{mnemonic} is ineligible at e{},{}/ELEN={elen_bits}",
                cfg.sew, cfg.lmul
            ),
        });
        return issues;
    }
    let operands = args
        .split(',')
        .map(|value| value.trim().to_string())
        .collect::<Vec<_>>();
    let vector_bases = operands
        .iter()
        .filter_map(|operand| vector_number(operand))
        .collect::<Vec<_>>();
    let source_base_conflict = rvv_spike_guard_model::distinct_base_ranks(mnemonic)
        .and_then(|(left, right)| vector_bases.get(left).zip(vector_bases.get(right)))
        .is_some_and(|(left, right)| left == right);
    let Some(shapes) = footprints(mnemonic, &operands, cfg) else {
        issues.push(RvvPreconditionIssue {
            code: "operand_footprint",
            detail: format!("cannot derive a legal operand footprint for {asm}"),
        });
        return issues;
    };
    let mut occupied = Vec::new();
    for shape in shapes {
        let token = &operands[shape.operand];
        let Some(base) = vector_number(token) else {
            issues.push(RvvPreconditionIssue {
                code: "vector_register",
                detail: format!("invalid vector operand {token} in {asm}"),
            });
            continue;
        };
        if shape.fixed_mask {
            if base != 0 {
                issues.push(RvvPreconditionIssue {
                    code: "fixed_mask",
                    detail: format!("{token} must be v0 in {asm}"),
                });
            }
            continue;
        }
        if !allowed_bases(mnemonic, cfg, shape, (0, 31), &occupied).contains(&base) {
            issues.push(RvvPreconditionIssue {
                code: "register_relation",
                detail: format!("{token} violates group alignment/range/overlap in {asm}"),
            });
        }
        occupied.push((base, shape));
    }
    if source_base_conflict && !issues.iter().any(|issue| issue.code == "register_relation") {
        issues.push(RvvPreconditionIssue {
            code: "spike_register_base_inequality",
            detail: format!("Spike requires distinct destination and source bases in {asm}"),
        });
    }
    issues
}

fn digits_after(value: &str, marker: &str) -> Option<u16> {
    let suffix = value.split_once(marker)?.1;
    let digits = suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn segment_fields(mnemonic: &str) -> Option<u8> {
    digits_after(mnemonic, "seg").and_then(|n| u8::try_from(n).ok())
}

fn whole_width(mnemonic: &str) -> Option<u8> {
    if let Some(width) = rvv_spike_guard_model::whole_register_alignment(mnemonic) {
        return Some(width);
    }
    if mnemonic.starts_with("vmv") && mnemonic.contains("r.v") {
        return digits_after(mnemonic, "vmv").and_then(|n| u8::try_from(n).ok());
    }
    if mnemonic.starts_with("vl") && mnemonic.contains("re") {
        return digits_after(mnemonic, "vl").and_then(|n| u8::try_from(n).ok());
    }
    if mnemonic.starts_with("vs") && mnemonic.contains("r.v") {
        return digits_after(mnemonic, "vs").and_then(|n| u8::try_from(n).ok());
    }
    None
}

/// Whole-register memory transfers do not read `vtype`, including when
/// `vill` is set. Whole-register moves are deliberately excluded here.
pub(crate) fn is_whole_register_load_store(mnemonic: &str) -> bool {
    whole_width(mnemonic).is_some() && (mnemonic.starts_with("vl") || mnemonic.starts_with("vs"))
}

fn memory_eew(mnemonic: &str) -> Option<u16> {
    if matches!(mnemonic, "vlm.v" | "vsm.v") {
        return Some(8);
    }
    let memory_family = mnemonic.starts_with("vl")
        || [
            "vse", "vsse", "vsux", "vsox", "vsseg", "vssseg", "vs1r", "vs2r", "vs4r", "vs8r",
        ]
        .iter()
        .any(|prefix| mnemonic.starts_with(prefix));
    if !memory_family {
        return None;
    }
    if mnemonic.contains("ei") {
        return digits_after(mnemonic, "ei");
    }
    let mut found = None;
    for (index, ch) in mnemonic.char_indices() {
        if ch == 'e' {
            let digits = mnemonic[index + 1..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>();
            if let Ok(value) = digits.parse() {
                found = Some(value);
            }
        }
    }
    found
}

fn is_fp_double(mnemonic: &str) -> bool {
    mnemonic.starts_with("vfw") || mnemonic.starts_with("vfncvt")
}

fn is_integer_double(mnemonic: &str) -> bool {
    mnemonic.starts_with("vw")
        || mnemonic.starts_with("vnclip")
        || mnemonic.starts_with("vnsra")
        || mnemonic.starts_with("vnsrl")
}

fn is_narrow(mnemonic: &str) -> bool {
    mnemonic.starts_with("vfncvt")
        || mnemonic.starts_with("vnclip")
        || mnemonic.starts_with("vnsra")
        || mnemonic.starts_with("vnsrl")
}

fn mask_result(mnemonic: &str) -> bool {
    mnemonic.ends_with(".mm")
        || [
            "vmseq", "vmsne", "vmslt", "vmsle", "vmsgt", "vmsge", "vmfeq", "vmfne", "vmflt",
            "vmfle", "vmfgt", "vmfge", "vmadc", "vmsbc", "vmsbf", "vmsif", "vmsof",
        ]
        .iter()
        .any(|prefix| mnemonic.starts_with(prefix))
}

fn literal_v0_selector(mnemonic: &str) -> bool {
    mnemonic.starts_with("vmerge")
        || mnemonic.starts_with("vfmerge")
        || mnemonic.starts_with("vadc")
        || mnemonic.starts_with("vsbc")
        || mnemonic.starts_with("vmadc")
        || mnemonic.starts_with("vmsbc")
}

fn requires_disjoint(mnemonic: &str) -> bool {
    matches!(mnemonic, "vmsbf.m" | "vmsif.m" | "vmsof.m")
        || mnemonic.starts_with("viota")
        || is_fp_double(mnemonic)
        || is_integer_double(mnemonic)
        || mnemonic.starts_with("vzext")
        || mnemonic.starts_with("vsext")
        || mnemonic.starts_with("vrgather")
        || mnemonic.starts_with("vslideup")
        || mnemonic.starts_with("vslide1up")
        || mnemonic.starts_with("vfslide1up")
        || mnemonic == "vcompress.vm"
        || mnemonic.contains("xseg")
        || mnemonic.starts_with("vmv") && mnemonic.contains("r.v")
}

fn decode(line: &str, isa_base: ISABase) -> Result<RiscvInstruction, GenerationError> {
    let context = match isa_base {
        ISABase::Rv32 => IsaDecodeContext::RV32,
        ISABase::Rv64 => IsaDecodeContext::RV64,
    };
    RiscvInstruction::decode_asm_with_context(line, context).map_err(|source| {
        GenerationError::StructuredInstructionBuildFailed {
            message: format!("cannot instantiate selected RVV opcode `{line}`: {source}"),
        }
    })
}

fn vector_number(token: &str) -> Option<u8> {
    let value = token.trim().strip_suffix(".t").unwrap_or(token.trim());
    value
        .strip_prefix('v')?
        .parse::<u8>()
        .ok()
        .filter(|n| *n < 32)
}

fn footprints(
    mnemonic: &str,
    operands: &[String],
    cfg: RvvConfigConstraint,
) -> Option<Vec<Footprint>> {
    let vector = operands
        .iter()
        .enumerate()
        .filter_map(|(index, token)| vector_number(token).map(|_| index))
        .collect::<Vec<_>>();
    if vector.is_empty() {
        return Some(Vec::new());
    }
    let group = rvv_legal::register_width(cfg.sew, cfg.lmul, cfg.sew)?;
    let whole = whole_width(mnemonic);
    let memory = memory_eew(mnemonic);
    let indexed = memory.is_some() && mnemonic.contains("ei");
    let segment = segment_fields(mnemonic).unwrap_or(1);
    let data_emul = if whole.is_some() || mnemonic == "vlm.v" || mnemonic == "vsm.v" {
        1
    } else if indexed {
        group
    } else if let Some(eew) = memory {
        rvv_legal::register_width(cfg.sew, cfg.lmul, eew)?
    } else {
        group
    };
    let data_total = u16::from(data_emul) * u16::from(segment);
    if data_total > 8 {
        return None;
    }
    let index_emul = if indexed {
        rvv_legal::register_width(cfg.sew, cfg.lmul, memory?)?
    } else {
        1
    };
    let wide = if is_fp_double(mnemonic) || is_integer_double(mnemonic) {
        Some(rvv_legal::register_width(cfg.sew, cfg.lmul, cfg.sew * 2)?)
    } else {
        None
    };
    let indexed_operand = indexed.then(|| {
        *vector
            .iter()
            .rev()
            .find(|&&index| !operands[index].ends_with(".t"))
            .expect("indexed memory has an index register")
    });
    let masked = operands.iter().any(|x| x.ends_with(".t")) || literal_v0_selector(mnemonic);
    let result_is_mask = mask_result(mnemonic);
    let mask_logical = mnemonic.ends_with(".mm");
    let extension_factor = rvv_legal::extension_factor(mnemonic);
    let mut result = Vec::with_capacity(vector.len());
    for (rank, &operand) in vector.iter().enumerate() {
        let token = &operands[operand];
        let mask_source = match mnemonic {
            "vcpop.m" | "vfirst.m" => rank == 0,
            "vmsbf.m" | "vmsif.m" | "vmsof.m" | "viota.m" => rank == 1,
            "vcompress.vm" => rank + 1 == vector.len(),
            _ => false,
        };
        let fixed_mask = token.ends_with(".t")
            || (rank + 1 == vector.len() && literal_v0_selector(mnemonic) && token == "v0");
        let mut align = if let Some(n) = whole { n } else { group };
        let mut span = align;
        let mut eew = cfg.sew;
        if memory.is_some() && whole.is_none() && rank == 0 {
            align = data_emul;
            span = u8::try_from(data_total).ok()?;
            eew = if indexed { cfg.sew } else { memory? };
        }
        if Some(operand) == indexed_operand && rank != 0 {
            align = index_emul;
            span = index_emul;
            eew = memory?;
        }
        if mnemonic == "vrgatherei16.vv" && rank + 1 == vector.len() {
            align = rvv_legal::register_width(cfg.sew, cfg.lmul, 16)?;
            span = align;
            eew = 16;
        }
        if let Some(wide_width) = wide {
            if (is_narrow(mnemonic) && rank == 1)
                || (!is_narrow(mnemonic) && rank == 0)
                || (!is_narrow(mnemonic) && mnemonic.contains(".w") && rank == 1)
            {
                align = wide_width;
                span = wide_width;
                eew = cfg.sew * 2;
            }
        }
        if rank == 1
            && let Some(factor) = extension_factor
        {
            eew = cfg.sew / factor;
            align = rvv_legal::register_width(cfg.sew, cfg.lmul, eew)?;
            span = align;
        }
        if result_is_mask && rank == 0
            || fixed_mask
            || mask_source
            || mnemonic == "vlm.v"
            || mnemonic == "vsm.v"
            || mask_logical
        {
            align = 1;
            span = 1;
            eew = 1;
        }
        if let Some(source_align) =
            rvv_spike_guard_model::required_lmul_alignment(mnemonic, rank, cfg.lmul)
        {
            align = align.max(source_align);
        }
        let allow_zero =
            !masked || (result_is_mask && rank == 0) || mnemonic == "vlm.v" || mnemonic == "vsm.v";
        result.push(Footprint {
            operand,
            align,
            span,
            eew,
            fixed_mask,
            allow_zero: allow_zero && !(masked && !result_is_mask && rank == 0),
        });
    }
    Some(result)
}

fn allowed_bases(
    mnemonic: &str,
    cfg: RvvConfigConstraint,
    footprint: Footprint,
    range: (u8, u8),
    occupied: &[(u8, Footprint)],
) -> Vec<u8> {
    let disjoint = requires_disjoint(mnemonic);
    let conditional_overlap = rvv_legal::extension_factor(mnemonic).is_some()
        || ((is_fp_double(mnemonic) || is_integer_double(mnemonic)) && !mnemonic.contains("red"));
    let indexed_store = mnemonic.starts_with("vsuxei") || mnemonic.starts_with("vsoxei");
    (range.0..=range.1.min(31))
        .filter(|base| *base > 0 || footprint.allow_zero)
        .filter(|base| *base % footprint.align == 0)
        .filter(|base| u16::from(*base) + u16::from(footprint.span) <= 32)
        .filter(|base| u16::from(*base) + u16::from(footprint.span) - 1 <= u16::from(range.1))
        .filter(|base| {
            occupied
                .iter()
                .enumerate()
                .all(|(prior_rank, (taken, prior))| {
                    if !rvv_legal::groups_overlap(*base, footprint.span, *taken, prior.span) {
                        return true;
                    }
                    if disjoint && !conditional_overlap {
                        return false;
                    }
                    if prior_rank == 0 {
                        if indexed_store {
                            // Both operands are sources; a register cannot be read at two EEWs.
                            return prior.eew == footprint.eew;
                        }
                        return rvv_legal::destination_source_overlap_allowed(
                            *taken,
                            prior.span,
                            prior.eew,
                            *base,
                            footprint.span,
                            footprint.eew,
                            rvv_legal::emul_at_least_one(cfg.sew, cfg.lmul, footprint.eew),
                        );
                    }
                    !disjoint && prior.eew == footprint.eew
                })
        })
        .collect()
}

fn bases_in_policy_range(
    mnemonic: &str,
    cfg: RvvConfigConstraint,
    footprint: Footprint,
    vector_range: (u8, u8),
    occupied: &[(u8, Footprint)],
) -> Vec<u8> {
    let preferred = allowed_bases(mnemonic, cfg, footprint, vector_range, occupied);
    if preferred.is_empty() {
        allowed_bases(mnemonic, cfg, footprint, (0, 31), occupied)
    } else {
        preferred
    }
}

fn joint_assignment_exists(
    mnemonic: &str,
    cfg: RvvConfigConstraint,
    vector_range: (u8, u8),
    shapes: &[Footprint],
    rank: usize,
    occupied: &mut Vec<(u8, Footprint)>,
) -> bool {
    if rank == shapes.len() {
        return true;
    }
    let shape = shapes[rank];
    if shape.fixed_mask {
        return joint_assignment_exists(mnemonic, cfg, vector_range, shapes, rank + 1, occupied);
    }
    for base in bases_in_policy_range(mnemonic, cfg, shape, vector_range, occupied) {
        occupied.push((base, shape));
        let found =
            joint_assignment_exists(mnemonic, cfg, vector_range, shapes, rank + 1, occupied);
        occupied.pop();
        if found {
            return true;
        }
    }
    false
}

fn alias_class(a: (u8, Footprint), b: (u8, Footprint)) -> &'static str {
    if !rvv_legal::groups_overlap(a.0, a.1.span, b.0, b.1.span) {
        "disjoint"
    } else if a.1.eew == b.1.eew {
        "same_eew_overlap"
    } else if a.1.eew < b.1.eew {
        "low_end_overlap"
    } else {
        "high_end_overlap"
    }
}

fn collect_alias_classes(
    mnemonic: &str,
    cfg: RvvConfigConstraint,
    vector_range: (u8, u8),
    shapes: &[Footprint],
    rank: usize,
    occupied: &mut Vec<(u8, Footprint)>,
    classes: &mut BTreeSet<String>,
) {
    // With fixed operand EEWs there are at most two classes: disjoint and
    // either equal-EEW overlap or the one permitted end-overlap direction.
    if classes.len() == 2 {
        return;
    }
    if occupied.len() == 2 {
        if joint_assignment_exists(mnemonic, cfg, vector_range, shapes, rank, occupied) {
            classes.insert(alias_class(occupied[0], occupied[1]).to_string());
        }
        return;
    }
    if rank == shapes.len() {
        return;
    }
    let shape = shapes[rank];
    if shape.fixed_mask {
        collect_alias_classes(
            mnemonic,
            cfg,
            vector_range,
            shapes,
            rank + 1,
            occupied,
            classes,
        );
        return;
    }
    for base in bases_in_policy_range(mnemonic, cfg, shape, vector_range, occupied) {
        occupied.push((base, shape));
        collect_alias_classes(
            mnemonic,
            cfg,
            vector_range,
            shapes,
            rank + 1,
            occupied,
            classes,
        );
        occupied.pop();
    }
}

fn domain_for_parts(
    mnemonic: &str,
    operands: &[String],
    profile: RvvSamplingProfile,
    vector_range: (u8, u8),
) -> RvvOpcodeDomain {
    let mut configurations = Vec::new();
    for sew in [8u16, 16, 32, 64] {
        for lmul in ["mf8", "mf4", "mf2", "m1", "m2", "m4", "m8"] {
            if !rvv_legal::opcode_config_eligible(mnemonic, sew, lmul, profile.elen_bits) {
                continue;
            }
            let cfg = RvvConfigConstraint::new(ISABase::Rv64, sew, 1, lmul, "ta", "ma");
            let Some(shapes) = footprints(mnemonic, operands, cfg) else {
                continue;
            };
            if !joint_assignment_exists(mnemonic, cfg, vector_range, &shapes, 0, &mut Vec::new()) {
                continue;
            }
            let (num, den) = rvv_legal::lmul_ratio(lmul).expect("enumerated LMUL");
            let architectural_vl_max =
                profile.vlen_bits * u32::from(num) / (u32::from(sew) * u32::from(den));
            let cap = if memory_eew(mnemonic).is_some() {
                profile.memory_avl_cap.min(profile.sampled_avl_cap)
            } else {
                profile.sampled_avl_cap
            };
            let sampled_avl_max = architectural_vl_max.min(u32::from(cap)) as u8;
            if sampled_avl_max == 0 {
                continue;
            }
            let operands = shapes
                .iter()
                .map(|shape| RvvOperandDomain {
                    operand_index: shape.operand,
                    eew: shape.eew,
                    register_span: shape.span,
                    register_alignment: shape.align,
                    legal_base_registers: if shape.fixed_mask {
                        vec![0]
                    } else {
                        bases_in_policy_range(mnemonic, cfg, *shape, vector_range, &[])
                    },
                    fixed_mask: shape.fixed_mask,
                })
                .collect();
            let mut alias_classes = BTreeSet::new();
            collect_alias_classes(
                mnemonic,
                cfg,
                vector_range,
                &shapes,
                0,
                &mut Vec::new(),
                &mut alias_classes,
            );
            configurations.push(RvvConfigDomain {
                sew,
                lmul,
                architectural_vl_max,
                sampled_avl_min: 1,
                sampled_avl_max,
                operands,
                first_pair_alias_classes: alias_classes.into_iter().collect(),
            });
        }
    }
    RvvOpcodeDomain {
        mnemonic: mnemonic.to_string(),
        profile,
        configurations,
    }
}

pub fn domain_for_asm(
    asm: &str,
    profile: RvvSamplingProfile,
    vector_range: (u8, u8),
) -> Option<RvvOpcodeDomain> {
    let (mnemonic, args) = asm.split_once(char::is_whitespace)?;
    let operands = args
        .split(',')
        .map(|x| x.trim().to_string())
        .collect::<Vec<_>>();
    Some(domain_for_parts(mnemonic, &operands, profile, vector_range))
}

fn instantiate_with_profile<R: Rng + ?Sized>(
    mnemonic: &str,
    original: &[String],
    cfg: RvvConfigConstraint,
    vector_range: (u8, u8),
    profile: RvvSamplingProfile,
    rng: &mut R,
) -> Option<Vec<String>> {
    let shapes = footprints(mnemonic, original, cfg)?;
    let mut operands = original.to_vec();
    let mut occupied = Vec::new();
    sample_joint_assignment(
        mnemonic,
        cfg,
        vector_range,
        profile,
        &shapes,
        0,
        &mut occupied,
        &mut operands,
        rng,
    )
    .then_some(operands)
}

fn sample_joint_assignment<R: Rng + ?Sized>(
    mnemonic: &str,
    cfg: RvvConfigConstraint,
    vector_range: (u8, u8),
    profile: RvvSamplingProfile,
    shapes: &[Footprint],
    rank: usize,
    occupied: &mut Vec<(u8, Footprint)>,
    operands: &mut [String],
    rng: &mut R,
) -> bool {
    if rank == shapes.len() {
        return true;
    }
    let footprint = shapes[rank];
    if footprint.fixed_mask {
        operands[footprint.operand] = if operands[footprint.operand].ends_with(".t") {
            "v0.t".to_string()
        } else {
            "v0".to_string()
        };
        return sample_joint_assignment(
            mnemonic,
            cfg,
            vector_range,
            profile,
            shapes,
            rank + 1,
            occupied,
            operands,
            rng,
        );
    }
    let mut candidates = bases_in_policy_range(mnemonic, cfg, footprint, vector_range, occupied);
    for i in (1..candidates.len()).rev() {
        let j = rng.random_range(0..=i);
        candidates.swap(i, j);
    }
    if occupied.len() == 1 && rng.random_range(0..100) < profile.alias_percent {
        candidates.sort_by_key(|base| {
            !rvv_legal::groups_overlap(*base, footprint.span, occupied[0].0, occupied[0].1.span)
        });
    } else if let Some(original_base) = vector_number(&operands[footprint.operand])
        && rng.random_range(0..100) < profile.original_operand_percent
        && let Some(position) = candidates.iter().position(|base| *base == original_base)
    {
        candidates.swap(0, position);
    }
    let previous = operands[footprint.operand].clone();
    let suffix = if previous.ends_with(".t") { ".t" } else { "" };
    for base in candidates {
        operands[footprint.operand] = format!("v{base}{suffix}");
        occupied.push((base, footprint));
        let found = sample_joint_assignment(
            mnemonic,
            cfg,
            vector_range,
            profile,
            shapes,
            rank + 1,
            occupied,
            operands,
            rng,
        );
        occupied.pop();
        if found {
            return true;
        }
    }
    operands[footprint.operand] = previous;
    false
}

#[cfg(test)]
fn instantiate<R: Rng + ?Sized>(
    mnemonic: &str,
    original: &[String],
    cfg: RvvConfigConstraint,
    vector_range: (u8, u8),
    rng: &mut R,
) -> Option<Vec<String>> {
    instantiate_with_profile(
        mnemonic,
        original,
        cfg,
        vector_range,
        RvvSamplingProfile::for_subject("xiangshan"),
        rng,
    )
}

fn safe_memory_operands<R: Rng + ?Sized>(
    mnemonic: &str,
    operands: &mut [String],
    cfg: RvvConfigConstraint,
    safe_base: u64,
    safe_memory_bytes: u64,
    profile: RvvSamplingProfile,
    rng: &mut R,
) -> Result<Option<Vec<String>>, GenerationError> {
    let Some(eew) = memory_eew(mnemonic) else {
        return Ok(None);
    };
    let base_position = operands
        .iter()
        .position(|token| token.contains("(x"))
        .ok_or_else(|| GenerationError::StructuredInstructionBuildFailed {
            message: format!("missing base operand for {mnemonic}"),
        })?;
    let base_reg = if rng.random_range(0..2) == 0 {
        "x11"
    } else {
        "x12"
    };
    operands[base_position] = format!("({base_reg})");
    if let Some(width) = whole_width(mnemonic) {
        let bytes = u64::from(width) * u64::from(profile.vlen_bits / 8);
        if bytes > safe_memory_bytes {
            return Err(GenerationError::StructuredInstructionBuildFailed {
                message: format!("{mnemonic} needs {bytes} scratch bytes, has {safe_memory_bytes}"),
            });
        }
        let lmul = match width {
            1 => "m1",
            2 => "m2",
            4 => "m4",
            8 => "m8",
            _ => {
                return Err(GenerationError::StructuredInstructionBuildFailed {
                    message: format!("invalid whole-register width for {mnemonic}"),
                });
            }
        };
        let setup_cfg = RvvConfigConstraint::new(cfg.isa_base, 8, 1, lmul, "tu", "mu");
        return Ok(Some(vec![
            format!("li {base_reg}, {safe_base}"),
            format!("li x5, {bytes}"),
            format!("vsetvli x0, x5, {}", setup_cfg.vsetivli_vtype_immediate()),
            "vmv.v.i v0, 0".to_string(),
            format!("vs{width}r.v v0, 0({base_reg})"),
        ]));
    }
    if safe_memory_bytes < 0x200 {
        return Err(GenerationError::StructuredInstructionBuildFailed {
            message: format!("{mnemonic} needs 512 scratch bytes, has {safe_memory_bytes}"),
        });
    }
    let offset = [0u64, 8, 16, 24][rng.random_range(0..4)];
    let mut setup = vec![format!("li {base_reg}, {safe_base}")];
    setup.push(format!("li x15, {}", rng.random::<i64>()));
    for byte_offset in (0..0x200).step_by(8) {
        setup.push(format!("sd x15, {byte_offset}({base_reg})"));
    }
    setup.push(format!("li {base_reg}, {}", safe_base + offset));

    let indexed = mnemonic.contains("ei");
    let masked = operands.iter().any(|token| token == "v0.t");
    if indexed || masked {
        let setup_sew = if indexed { eew } else { 8 };
        let setup_cfg = RvvConfigConstraint::new(cfg.isa_base, setup_sew, 1, "m1", "ta", "ma");
        setup.push("li x5, 4096".to_string());
        setup.push(format!(
            "vsetvli x0, x5, {}",
            setup_cfg.vsetivli_vtype_immediate()
        ));
        if indexed {
            let index_position = operands
                .iter()
                .enumerate()
                .rev()
                .find(|(position, token)| {
                    *position > base_position
                        && !token.ends_with(".t")
                        && vector_number(token).is_some()
                })
                .ok_or_else(|| GenerationError::StructuredInstructionBuildFailed {
                    message: format!("missing index operand for {mnemonic}"),
                })?
                .0;
            let index_base = vector_number(&operands[index_position]).ok_or_else(|| {
                GenerationError::StructuredInstructionBuildFailed {
                    message: format!("invalid index operand for {mnemonic}"),
                }
            })?;
            let width = rvv_legal::register_width(cfg.sew, cfg.lmul, eew).ok_or_else(|| {
                GenerationError::StructuredInstructionBuildFailed {
                    message: format!("invalid index EMUL for {mnemonic}"),
                }
            })?;
            let index_offset = [0u8, 8][rng.random_range(0..2)];
            for reg in index_base..index_base + width {
                setup.push(format!("vmv.v.i v{reg}, {index_offset}"));
            }
        }
        if masked {
            setup.push("vmv.v.i v0, -1".to_string());
        }
    }

    if let Some(stride_position) = operands
        .iter()
        .enumerate()
        .find(|(position, token)| {
            *position > base_position && token.starts_with('x') && token[1..].parse::<u8>().is_ok()
        })
        .map(|(position, _)| position)
    {
        let stride_reg = if rng.random_range(0..2) == 0 {
            "x13"
        } else {
            "x14"
        };
        operands[stride_position] = stride_reg.to_string();
        let width = u32::from(eew / 8).max(1) * u32::from(segment_fields(mnemonic).unwrap_or(1));
        let stride = if rng.random_range(0..2) == 0 {
            0
        } else {
            width
        };
        setup.push(format!("li {stride_reg}, {stride}"));
    }
    Ok(Some(setup))
}

fn initialize_vector_operands<R: Rng + ?Sized>(
    mnemonic: &str,
    operands: &[String],
    cfg: RvvConfigConstraint,
    profile: RvvSamplingProfile,
    rng: &mut R,
) -> Option<Vec<String>> {
    let shapes = footprints(mnemonic, operands, cfg)?;
    let mut registers = BTreeSet::new();
    for shape in shapes {
        let base = vector_number(&operands[shape.operand])?;
        for register in base..base.checked_add(shape.span)? {
            registers.insert(register);
        }
    }
    if registers.is_empty() {
        return Some(Vec::new());
    }
    let setup_cfg = RvvConfigConstraint::new(cfg.isa_base, 8, 1, "m1", "tu", "mu");
    let mut setup = vec![
        "csrrw x0, vstart, x0".to_string(),
        format!("li x5, {}", profile.vlen_bits / 8),
        format!("vsetvli x0, x5, {}", setup_cfg.vsetivli_vtype_immediate()),
    ];
    let mask_zero =
        operands.iter().any(|operand| operand == "v0.t") || literal_v0_selector(mnemonic);
    for register in registers {
        let immediate = if register == 0 && mask_zero {
            -1
        } else {
            rng.random_range(-16..=15)
        };
        setup.push(format!("vmv.v.i v{register}, {immediate}"));
    }
    Some(setup)
}

fn bridge_catalog_source(
    mnemonic: &str,
    operands: &[String],
    cfg: RvvConfigConstraint,
    profile: RvvSamplingProfile,
    bridge_addr: Option<u64>,
) -> Option<Vec<String>> {
    let address = bridge_addr?;
    if memory_eew(mnemonic).is_some()
        || operands
            .iter()
            .any(|operand| operand.contains("x23") || operand.contains("x24"))
    {
        return None;
    }
    let source = *footprints(mnemonic, operands, cfg)?.get(1)?;
    if source.fixed_mask {
        return None;
    }
    let base = vector_number(&operands[source.operand])?;
    let setup_cfg = RvvConfigConstraint::new(cfg.isa_base, 8, 1, "m1", "tu", "mu");
    Some(vec![
        format!("li x24, {address}"),
        "ld x23, 0(x24)".to_string(),
        format!("li x5, {}", profile.vlen_bits / 8),
        format!("vsetvli x0, x5, {}", setup_cfg.vsetivli_vtype_immediate()),
        format!("vmv.v.x v{base}, x23"),
    ])
}

#[cfg(test)]
pub(crate) fn constrain_selected_opcode<R: Rng + ?Sized>(
    sequence: &mut InstructionSequence<RiscvInstruction>,
    isa_base: ISABase,
    vector_range: (u8, u8),
    safe_memory_base: u64,
    rng: &mut R,
) -> Result<(), GenerationError> {
    constrain_selected_opcode_with_profile(
        sequence,
        isa_base,
        vector_range,
        safe_memory_base,
        u64::MAX,
        RvvSamplingProfile::for_subject("xiangshan"),
        None,
        None,
        rng,
    )
}

fn order_configurations<R: Rng + ?Sized>(
    domain: &mut RvvOpcodeDomain,
    schedule: Option<(u64, u8)>,
    rng: &mut R,
) {
    let count = domain.configurations.len();
    if count == 0 {
        return;
    }
    if domain.profile.config_sampling == "cycle"
        && let Some((round, placement_stream)) = schedule
    {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in domain.mnemonic.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let cycle = round / count as u64;
        let mut order_rng = StdRng::seed_from_u64(
            hash ^ cycle.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ u64::from(placement_stream).wrapping_mul(0xd1b5_4a32_d192_ed03),
        );
        for i in (1..count).rev() {
            let j = order_rng.random_range(0..=i);
            domain.configurations.swap(i, j);
        }
        domain
            .configurations
            .rotate_left((round % count as u64) as usize);
    } else {
        for i in (1..count).rev() {
            let j = rng.random_range(0..=i);
            domain.configurations.swap(i, j);
        }
    }
}

pub(crate) fn constrain_selected_opcode_with_profile<R: Rng + ?Sized>(
    sequence: &mut InstructionSequence<RiscvInstruction>,
    isa_base: ISABase,
    vector_range: (u8, u8),
    safe_memory_base: u64,
    safe_memory_bytes: u64,
    profile: RvvSamplingProfile,
    schedule: Option<(u64, u8)>,
    bridge_addr: Option<u64>,
    rng: &mut R,
) -> Result<(), GenerationError> {
    let line = sequence.instruction.to_string();
    let (mnemonic, args) = line.split_once(char::is_whitespace).ok_or_else(|| {
        GenerationError::StructuredInstructionBuildFailed {
            message: format!("selected RVV instruction lacks operands: {line}"),
        }
    })?;
    let operands = args
        .split(',')
        .map(|x| x.trim().to_string())
        .collect::<Vec<_>>();
    let mut domain = domain_for_parts(mnemonic, &operands, profile, vector_range);
    if domain.configurations.is_empty() {
        return Err(GenerationError::StructuredInstructionBuildFailed {
            message: format!("selected RVV opcode `{mnemonic}` has an empty legal domain"),
        });
    }
    order_configurations(&mut domain, schedule, rng);
    let mut last_error = None;
    for choice in domain.configurations {
        let cfg = RvvConfigConstraint::new(
            isa_base,
            choice.sew,
            rng.random_range(choice.sampled_avl_min..=choice.sampled_avl_max),
            choice.lmul,
            if rng.random_bool(0.5) { "ta" } else { "tu" },
            if rng.random_bool(0.5) { "ma" } else { "mu" },
        );
        let (candidate, mut setup) = if mnemonic.starts_with("vset") {
            let avl = cfg.vl;
            let rd = [5u8, 7, 9, 11][rng.random_range(0..4)];
            let imm = cfg.vsetivli_vtype_immediate();
            match mnemonic {
                "vsetivli" => (format!("vsetivli x{rd}, {avl}, {imm}"), Vec::new()),
                "vsetvli" => (
                    format!("vsetvli x{rd}, x6, {imm}"),
                    vec!["li x6, ".to_string() + &avl.to_string()],
                ),
                "vsetvl" => (
                    format!("vsetvl x{rd}, x6, x5"),
                    vec![format!("li x6, {avl}"), format!("li x5, {imm}")],
                ),
                _ => continue,
            }
        } else {
            let Some(mut operands) =
                instantiate_with_profile(mnemonic, &operands, cfg, vector_range, profile, rng)
            else {
                continue;
            };
            let Some(vector_setup) =
                initialize_vector_operands(mnemonic, &operands, cfg, profile, rng)
            else {
                continue;
            };
            let memory_setup = safe_memory_operands(
                mnemonic,
                &mut operands,
                cfg,
                safe_memory_base,
                safe_memory_bytes,
                profile,
                rng,
            )?
            .unwrap_or_default();
            let bridge_setup =
                bridge_catalog_source(mnemonic, &operands, cfg, profile, bridge_addr)
                    .unwrap_or_default();
            let mut setup = if is_whole_register_load_store(mnemonic) {
                memory_setup
                    .into_iter()
                    .chain(vector_setup)
                    .collect::<Vec<_>>()
            } else {
                vector_setup
                    .into_iter()
                    .chain(memory_setup)
                    .collect::<Vec<_>>()
            };
            setup.extend(bridge_setup);
            setup.extend([
                format!(
                    "vsetivli x0, {}, {}",
                    cfg.vl,
                    cfg.vsetivli_vtype_immediate()
                ),
                "csrrw x0, vstart, x0".to_string(),
            ]);
            (format!("{mnemonic} {}", operands.join(", ")), setup)
        };
        let issues = check_asm_under_config(&candidate, cfg, profile.elen_bits);
        if !issues.is_empty() {
            return Err(GenerationError::StructuredInstructionBuildFailed {
                message: format!("VC domain emitted {candidate} with unmet rules: {issues:?}"),
            });
        }
        match decode(&candidate, isa_base) {
            Ok(instruction) => {
                let mut decoded = Vec::new();
                for line in setup.drain(..) {
                    decoded.push(decode(&line, isa_base)?);
                }
                sequence.instruction = instruction;
                sequence.pre_instructions.extend(decoded);
                return Ok(());
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(
        last_error.unwrap_or_else(|| GenerationError::StructuredInstructionBuildFailed {
            message: format!("no legal state/operands found for selected opcode `{mnemonic}`"),
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;

    #[test]
    fn whole_register_memory_uses_a_bounded_initialized_scratch_region() {
        let profile = RvvSamplingProfile::for_subject("ara");
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 64, 2, "m1", "tu", "mu");
        let mut operands = vec!["v16".to_string(), "(x8)".to_string()];
        let mut rng = StdRng::seed_from_u64(19);
        let setup = safe_memory_operands(
            "vl8re64.v",
            &mut operands,
            cfg,
            0x8001_0200,
            4096,
            profile,
            &mut rng,
        )
        .unwrap()
        .unwrap();
        assert!(setup.iter().any(|line| line.starts_with("vs8r.v v0")));
        let narrow = safe_memory_operands(
            "vl1re64.v",
            &mut operands,
            cfg,
            0x8001_0200,
            512,
            profile,
            &mut rng,
        )
        .unwrap()
        .unwrap();
        assert!(narrow.iter().any(|line| line.starts_with("vs1r.v v0")));
        assert!(
            safe_memory_operands(
                "vl8re64.v",
                &mut operands,
                cfg,
                0x8001_0200,
                4095,
                profile,
                &mut rng,
            )
            .is_err()
        );
    }

    #[test]
    fn catalog_vector_operands_are_initialized_before_use() {
        let profile = RvvSamplingProfile::for_subject("ara");
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 32, 4, "m1", "tu", "mu");
        let operands = vec!["v16".into(), "v20".into(), "v24".into()];
        let setup = initialize_vector_operands(
            "vadd.vv",
            &operands,
            cfg,
            profile,
            &mut StdRng::seed_from_u64(19),
        )
        .unwrap();
        for base in [16, 20, 24] {
            assert!(
                setup
                    .iter()
                    .any(|line| line.starts_with(&format!("vmv.v.i v{base}, ")))
            );
        }
    }

    #[test]
    fn selected_compute_can_consume_the_stateflow_observation() {
        let profile = RvvSamplingProfile::for_subject("xiangshan");
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 32, 4, "m1", "tu", "mu");
        let operands = vec!["v16".into(), "v20".into(), "v24".into()];
        let setup = bridge_catalog_source("vadd.vv", &operands, cfg, profile, Some(0x8001_0080))
            .expect("source vector can consume the graph observation");
        assert!(setup.iter().any(|line| line == "ld x23, 0(x24)"));
        assert!(setup.iter().any(|line| line == "vmv.v.x v20, x23"));
        assert!(
            bridge_catalog_source("vle32.v", &operands, cfg, profile, Some(0x8001_0080)).is_none()
        );
    }

    #[test]
    fn catalog_legality_keeps_opcode_while_varying_configuration() {
        let mut rng = StdRng::seed_from_u64(123);
        let mut names = std::collections::BTreeSet::new();
        for _ in 0..32 {
            let mut sequence =
                InstructionSequence::new(decode("vfncvt.f.f.w v13, v18", ISABase::Rv64).unwrap());
            constrain_selected_opcode(
                &mut sequence,
                ISABase::Rv64,
                (13, 31),
                0x8001_0200,
                &mut rng,
            )
            .unwrap();
            assert!(sequence.instruction.to_string().starts_with("vfncvt.f.f.w"));
            names.insert(sequence.instruction.to_string());
        }
        assert!(names.len() > 1);
    }

    #[test]
    fn domain_tracks_dut_vlmax_separately_from_sampling_cap() {
        let operands = ["v8".into(), "v12".into(), "v16".into()];
        let xiangshan = domain_for_parts(
            "vadd.vv",
            &operands,
            RvvSamplingProfile::for_subject("xiangshan"),
            (13, 31),
        );
        let ara = domain_for_parts(
            "vadd.vv",
            &operands,
            RvvSamplingProfile::for_subject("ara"),
            (13, 31),
        );
        assert_eq!(xiangshan.configurations.len(), 22);
        assert_eq!(ara.configurations.len(), 22);
        let xs = xiangshan
            .configurations
            .iter()
            .find(|cfg| cfg.sew == 8 && cfg.lmul == "m8")
            .unwrap();
        let ar = ara
            .configurations
            .iter()
            .find(|cfg| cfg.sew == 8 && cfg.lmul == "m8")
            .unwrap();
        assert_eq!(xs.architectural_vl_max, 128);
        assert_eq!(ar.architectural_vl_max, 4096);
        assert_eq!(xs.sampled_avl_max, 31);
        assert_eq!(ar.sampled_avl_max, 31);
        assert!(
            xs.first_pair_alias_classes
                .contains(&"same_eew_overlap".to_string())
        );
        assert!(
            xs.first_pair_alias_classes
                .contains(&"disjoint".to_string())
        );
    }

    #[test]
    fn coverage_cycle_visits_every_config_before_repeating() {
        let operands = ["v8".into(), "v12".into(), "v16".into()];
        let original = domain_for_parts(
            "vadd.vv",
            &operands,
            RvvSamplingProfile::for_subject("xiangshan"),
            (13, 31),
        );
        let mut rng = StdRng::seed_from_u64(0x5eed);
        for stream in [0, 1] {
            let mut observed = BTreeSet::new();
            for round in 0..original.configurations.len() {
                let mut domain = original.clone();
                order_configurations(&mut domain, Some((round as u64, stream)), &mut rng);
                let first = &domain.configurations[0];
                observed.insert((first.sew, first.lmul));
            }
            assert_eq!(observed.len(), original.configurations.len());
        }
    }

    #[test]
    fn alias_probability_prioritizes_same_eew_overlap_too() {
        let mut profile = RvvSamplingProfile::for_subject("xiangshan");
        profile.alias_percent = 100;
        profile.original_operand_percent = 0;
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 8, 4, "m1", "ta", "ma");
        let original = ["v8".into(), "v12".into(), "v16".into()];
        let mut rng = StdRng::seed_from_u64(0xa11a5);
        for _ in 0..32 {
            let operands =
                instantiate_with_profile("vadd.vv", &original, cfg, (13, 31), profile, &mut rng)
                    .unwrap();
            assert_eq!(vector_number(&operands[0]), vector_number(&operands[1]));
        }
    }

    #[test]
    fn domain_exposes_legal_high_end_alias_without_low_end_alias() {
        let domain = domain_for_parts(
            "vzext.vf4",
            &["v8".into(), "v12".into()],
            RvvSamplingProfile::for_subject("xiangshan"),
            (0, 31),
        );
        let cfg = domain
            .configurations
            .iter()
            .find(|cfg| cfg.sew == 32 && cfg.lmul == "m4")
            .unwrap();
        assert!(
            cfg.first_pair_alias_classes
                .contains(&"high_end_overlap".to_string())
        );
        assert!(
            !cfg.first_pair_alias_classes
                .contains(&"low_end_overlap".to_string())
        );
    }

    #[test]
    fn slide_and_whole_register_shapes_are_not_confused_with_element_memory() {
        assert_eq!(memory_eew("vslide1down.vx"), None);
        assert_eq!(memory_eew("vslide1up.vx"), None);
        assert_eq!(memory_eew("vl4re8.v"), Some(8));
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 16, 2, "m4", "ta", "ma");
        let slide = footprints(
            "vslide1down.vx",
            &["v21".into(), "v20".into(), "x20".into()],
            cfg,
        )
        .unwrap();
        assert_eq!((slide[0].align, slide[0].span), (4, 4));
        let whole = footprints("vl4re8.v", &["v25".into(), "(x9)".into()], cfg).unwrap();
        assert_eq!((whole[0].align, whole[0].span), (4, 4));
    }

    #[test]
    fn spec_rules_expand_legal_configuration_and_mask_register_space() {
        let mut rng = StdRng::seed_from_u64(74);
        let ext = domain_for_parts(
            "vzext.vf4",
            &["v8".into(), "v12".into()],
            RvvSamplingProfile::for_subject("xiangshan"),
            (0, 31),
        )
        .configurations;
        assert!(ext.iter().any(|cfg| cfg.sew == 32 && cfg.lmul == "m1"));
        assert!(ext.iter().any(|cfg| cfg.sew == 64 && cfg.lmul == "m8"));
        assert!(!ext.iter().any(|cfg| cfg.sew == 16));
        let add = domain_for_parts(
            "vadd.vv",
            &["v8".into(), "v12".into(), "v16".into()],
            RvvSamplingProfile::for_subject("xiangshan"),
            (0, 31),
        )
        .configurations;
        assert!(add.iter().any(|cfg| cfg.sew == 8 && cfg.lmul == "mf8"));
        assert!(add.iter().any(|cfg| cfg.sew == 64 && cfg.lmul == "m8"));

        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 8, 4, "m4", "ta", "ma");
        let operands = vec!["v9".into(), "v19".into(), "v29".into()];
        let shape = footprints("vmor.mm", &operands, cfg).unwrap();
        assert!(
            shape
                .iter()
                .all(|operand| { operand.align == 1 && operand.span == 1 && operand.eew == 1 })
        );
        let result = instantiate("vmor.mm", &operands, cfg, (0, 31), &mut rng).unwrap();
        assert!(
            result
                .iter()
                .all(|operand| vector_number(operand).is_some())
        );
        assert!(
            result
                .iter()
                .any(|operand| vector_number(operand).unwrap() % 4 != 0)
        );

        let scalar_mask = footprints("vcpop.m", &["x5".into(), "v19".into()], cfg).unwrap();
        assert_eq!(
            (
                scalar_mask[0].align,
                scalar_mask[0].span,
                scalar_mask[0].eew
            ),
            (1, 1, 1)
        );
        let compress = footprints(
            "vcompress.vm",
            &["v8".into(), "v12".into(), "v19".into()],
            cfg,
        )
        .unwrap();
        assert_eq!(
            (compress[2].align, compress[2].span, compress[2].eew),
            (1, 1, 1)
        );
    }

    #[test]
    fn extension_source_uses_fractional_emul_and_samples_legal_overlap() {
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 32, 4, "m8", "tu", "mu");
        let original = vec!["v0".into(), "v6".into()];
        let shape = footprints("vzext.vf4", &original, cfg).unwrap();
        assert_eq!((shape[0].span, shape[0].eew), (8, 32));
        assert_eq!((shape[1].span, shape[1].eew), (2, 8));

        let mut rng = StdRng::seed_from_u64(812);
        let mut saw_overlap = false;
        let mut saw_disjoint = false;
        for _ in 0..100 {
            let generated = instantiate("vzext.vf4", &original, cfg, (0, 31), &mut rng).unwrap();
            let dest = vector_number(&generated[0]).unwrap();
            let source = vector_number(&generated[1]).unwrap();
            let overlap = rvv_legal::groups_overlap(dest, 8, source, 2);
            assert!(rvv_legal::destination_source_overlap_allowed(
                dest, 8, 32, source, 2, 8, true,
            ));
            saw_overlap |= overlap;
            saw_disjoint |= !overlap;
        }
        assert!(saw_overlap && saw_disjoint);
    }

    #[test]
    fn indexed_load_rejects_low_end_overlap_but_keeps_legal_high_end() {
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 16, 4, "m8", "tu", "ma");
        let original = vec!["v16".into(), "(x11)".into(), "v16".into(), "v0.t".into()];
        let shape = footprints("vluxei8.v", &original, cfg).unwrap();
        assert_eq!((shape[0].span, shape[0].eew), (8, 16));
        assert_eq!((shape[1].span, shape[1].eew), (4, 8));
        let mut rng = StdRng::seed_from_u64(0x1dee_08);
        let mut saw_high_end = false;
        let mut saw_disjoint = false;
        for _ in 0..128 {
            let generated = instantiate("vluxei8.v", &original, cfg, (16, 31), &mut rng).unwrap();
            let dest = vector_number(&generated[0]).unwrap();
            let index = vector_number(&generated[2]).unwrap();
            assert!(rvv_legal::destination_source_overlap_allowed(
                dest, 8, 16, index, 4, 8, true,
            ));
            saw_high_end |= index == dest + 4;
            saw_disjoint |= !rvv_legal::groups_overlap(dest, 8, index, 4);
        }
        assert!(saw_high_end && saw_disjoint);
    }

    #[test]
    fn indexed_store_sources_with_different_eew_never_overlap() {
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 16, 4, "m8", "tu", "ma");
        let original = vec!["v16".into(), "(x11)".into(), "v20".into(), "v0.t".into()];
        let mut rng = StdRng::seed_from_u64(0x5702_e8);
        for _ in 0..128 {
            let generated = instantiate("vsuxei8.v", &original, cfg, (16, 31), &mut rng).unwrap();
            let data = vector_number(&generated[0]).unwrap();
            let index = vector_number(&generated[2]).unwrap();
            assert!(!rvv_legal::groups_overlap(data, 8, index, 4));
        }
    }

    #[test]
    fn slide_one_up_never_aliases_its_source() {
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 32, 3, "m1", "ta", "mu");
        let mut profile = RvvSamplingProfile::for_subject("xiangshan");
        profile.alias_percent = 100;
        let mut rng = StdRng::seed_from_u64(423);
        for mnemonic in ["vslide1up.vx", "vfslide1up.vf"] {
            let scalar = if mnemonic.starts_with("vf") {
                "f13"
            } else {
                "x13"
            };
            let original = ["v18".into(), "v18".into(), scalar.into()];
            let domain = domain_for_parts(mnemonic, &original, profile, (13, 31));
            let shapes = domain
                .configurations
                .iter()
                .find(|item| item.sew == 32 && item.lmul == "m1")
                .unwrap();
            assert_eq!(
                shapes.first_pair_alias_classes,
                vec!["disjoint".to_string()]
            );
            for _ in 0..32 {
                let generated =
                    instantiate_with_profile(mnemonic, &original, cfg, (13, 31), profile, &mut rng)
                        .unwrap();
                assert_ne!(vector_number(&generated[0]), vector_number(&generated[1]));
            }
        }
    }

    #[test]
    fn mask_carry_result_can_use_legal_low_end_alias() {
        let cfg = RvvConfigConstraint::new(ISABase::Rv64, 8, 4, "m1", "tu", "mu");
        let legal = check_asm_under_config("vmadc.vxm v28, v28, x13, v0", cfg, 64);
        assert!(legal.is_empty(), "{legal:?}");
        let compare = check_asm_under_config("vmseq.vv v28, v28, v17", cfg, 64);
        assert!(compare.is_empty(), "{compare:?}");
        let disjoint = check_asm_under_config("vmsif.m v28, v28", cfg, 64);
        assert!(
            disjoint
                .iter()
                .any(|issue| issue.code == "register_relation")
        );
    }
}
