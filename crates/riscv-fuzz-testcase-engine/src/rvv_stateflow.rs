use std::env;

use rand::{SeedableRng as _, rngs::StdRng};
use riscv_fuzz_plugin_api::ISABase;
use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::{AsmDecodeWithContext, InstructionSequence, IsaDecodeContext};

use crate::{
    GenerationError,
    testcase::{GeneratedTestCase, HartPhase},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvvStateFlowFlavor {
    /// SSM-only ablation: preserve producer/consumer binding, but insert no
    /// RVV config/vstart setup and leave operand shapes unrepaired.
    StateFlowOnly,
    /// Preserve producer/consumer binding while the instruction instantiator
    /// obeys RVV config and operand-shape constraints.
    StateFlowConstrained,
    /// Same access families and random sequencing with safe memory operands.
    PositiveConstrained,
    /// ARA direct-execution profile: use the same constrained SSM+RCM
    /// construction, but broaden the generic RVV access families so historical
    /// ARA bugs can be reached without replaying fixed bug templates.
    AraGeneric,
    /// AraGeneric access families with legal/safe operand choices where possible.
    PositiveAraGeneric,
    /// ARA ablation: keep state-flow sequencing and generic risk families, but
    /// disable semantic RCM repairs such as global config/vstart setup.
    AraNoConstraints,
    /// ARA ablation: keep constrained generic RVV construction, but generate
    /// independent instructions instead of a producer/consumer sequence.
    AraNoSequence,
    /// ARA ablation placeholder for the pure random-RVV baseline.
    AraNoConstraintsNoSequence,
    /// Conservative ARA profile: keep constrained state-flow coverage, but
    /// avoid complex RVV memory forms that can stall ARA direct execution.
    AraSafe,
    /// Reuse the conservative constrained state-flow profile for model-only
    /// coverage experiments on either Ara or Xiangshan.
    CoverageConservative,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RvvStateFlowInjectionReport {
    pub inserted_blocks: usize,
    pub inserted_instructions: usize,
    pub state_flow_edges: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexedAlignmentScenario {
    Aligned,
    MaskedOffMisaligned,
    ActiveMisaligned,
}

impl IndexedAlignmentScenario {
    // Stable 70/20/10 buckets. The selected lane and offset remain random.
    fn for_seed(seed: u64) -> Self {
        match seed % 10 {
            7 | 8 => Self::MaskedOffMisaligned,
            9 => Self::ActiveMisaligned,
            _ => Self::Aligned,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TemplateRvvConfig {
    sew: u16,
    vl: u8,
    lmul: &'static str,
    tail_policy: &'static str,
    mask_policy: &'static str,
}

impl TemplateRvvConfig {
    fn vtype_immediate(self) -> u16 {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RvvLiveConfig {
    cfg: TemplateRvvConfig,
}

impl RvvLiveConfig {
    fn constrained_e8_m1() -> Self {
        Self {
            cfg: TemplateRvvConfig {
                sew: 8,
                vl: 8,
                lmul: "m1",
                tail_policy: "tu",
                mask_policy: "mu",
            },
        }
    }

    fn constrained_e16_m1() -> Self {
        Self {
            cfg: TemplateRvvConfig {
                sew: 16,
                vl: 8,
                lmul: "m1",
                tail_policy: "tu",
                mask_policy: "mu",
            },
        }
    }

    fn constrained_e32_m1() -> Self {
        Self {
            cfg: TemplateRvvConfig {
                sew: 32,
                vl: 8,
                lmul: "m1",
                tail_policy: "tu",
                mask_policy: "mu",
            },
        }
    }

    fn constrained_e32_m4() -> Self {
        Self {
            cfg: TemplateRvvConfig {
                sew: 32,
                vl: 8,
                lmul: "m4",
                tail_policy: "tu",
                mask_policy: "mu",
            },
        }
    }

    fn constrained_e64_m1() -> Self {
        Self {
            cfg: TemplateRvvConfig {
                sew: 64,
                vl: 8,
                lmul: "m1",
                tail_policy: "tu",
                mask_policy: "mu",
            },
        }
    }

    fn constrained_m1_for_sew(sew: u16) -> Self {
        match sew {
            8 => Self::constrained_e8_m1(),
            16 => Self::constrained_e16_m1(),
            32 => Self::constrained_e32_m1(),
            64 => Self::constrained_e64_m1(),
            _ => Self::constrained_e32_m1(),
        }
    }

    fn group_width(self) -> u8 {
        match self.cfg.lmul {
            "m2" => 2,
            "m4" => 4,
            "m8" => 8,
            _ => 1,
        }
    }

    fn vsetivli_line(self) -> String {
        format!(
            "vsetivli x0, {}, {}",
            self.cfg.vl,
            self.cfg.vtype_immediate()
        )
    }

    fn legal_group_bases(self, vector_range: (u8, u8)) -> Vec<u8> {
        let width = self.group_width();
        let min = vector_range.0.max(1);
        let max = vector_range.1.min(31);
        if min > max {
            return Vec::new();
        }
        (min..=max)
            .filter(|base| base.saturating_add(width.saturating_sub(1)) <= max)
            .filter(|base| !matches!(width, 2 | 4 | 8) || base % width == 0)
            .collect()
    }

    fn legal_segment_bases(self, vector_range: (u8, u8), field_count: u8) -> Vec<u8> {
        let width = self.group_width();
        let segment_width = width.saturating_mul(field_count);
        let min = vector_range.0.max(1);
        let max = vector_range.1.min(31);
        if min > max || segment_width == 0 {
            return Vec::new();
        }
        (min..=max)
            .filter(|base| base.saturating_add(segment_width.saturating_sub(1)) <= max)
            .filter(|base| !matches!(width, 2 | 4 | 8) || base % width == 0)
            .collect()
    }

    fn legal_whole_register_bases(vector_range: (u8, u8), width: u8) -> Vec<u8> {
        let min = vector_range.0.max(1);
        let max = vector_range.1.min(31);
        if min > max || width == 0 {
            return Vec::new();
        }
        (min..=max)
            .filter(|base| base.saturating_add(width.saturating_sub(1)) <= max)
            .filter(|base| *base % width == 0)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LiveVecState {
    id: u32,
    base: u8,
    width: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AraRiskFocus {
    #[default]
    Auto,
    LargeMaskedSlide,
    SegmentLaterField,
    SparseMaskedSegment,
    IndexedMisalignment,
    MaskedIndexedMisalignment,
    UnmaskedIndexedMisalignment,
    WidenZeroExtend,
    MaskMergeV0Overlap,
    VstartMaskedStore,
}

impl AraRiskFocus {
    const ENV_VAR: &'static str = "RISCV_FUZZ_ARA_RISK_FOCUS";

    fn from_env() -> Self {
        let Ok(raw) = env::var(Self::ENV_VAR) else {
            return Self::Auto;
        };
        let normalized = raw.trim().to_ascii_lowercase().replace(['_', ' '], "-");
        match normalized.as_str() {
            "slide" | "large-slide" | "large-masked-slide" | "ara-459" => Self::LargeMaskedSlide,
            "segment" | "segment-later" | "later-segment" | "ara-461" => Self::SegmentLaterField,
            "sparse-segment" | "masked-segment" | "segment-lane" | "ara-462" => {
                Self::SparseMaskedSegment
            }
            "indexed" | "indexed-misalignment" => Self::IndexedMisalignment,
            "masked-indexed" | "masked-indexed-misalignment" | "ara-456" => {
                Self::MaskedIndexedMisalignment
            }
            "unmasked-indexed" | "unmasked-indexed-misalignment" | "ara-457" => {
                Self::UnmaskedIndexedMisalignment
            }
            "widen" | "widen-zero-extend" | "ara-452" => Self::WidenZeroExtend,
            "mask-merge" | "v0-overlap" | "ara-460" => Self::MaskMergeV0Overlap,
            "vstart-store" | "masked-vstart-store" | "nonzero-vstart-store" | "ara-458" => {
                Self::VstartMaskedStore
            }
            _ => Self::Auto,
        }
    }

    fn forced_access(
        self,
        pool: &RvvStatePool,
        episode: usize,
        step: usize,
    ) -> Option<AccessModel> {
        if episode != 0 {
            return None;
        }
        match self {
            Self::Auto => None,
            Self::LargeMaskedSlide if step == 0 => Some(AccessModel::LargeMaskedSlideWindowObserve),
            Self::SegmentLaterField if step == 0 => {
                Some(AccessModel::SegmentLaterFieldStrictObserve)
            }
            Self::SparseMaskedSegment if step == 0 => {
                Some(AccessModel::SparseMaskedSegmentLaneObserve)
            }
            Self::IndexedMisalignment if step == 0 => {
                Some(AccessModel::IndexedMisalignmentSensitivity)
            }
            Self::MaskedIndexedMisalignment if step == 0 => {
                Some(AccessModel::MaskedIndexedMisalignmentProbe)
            }
            Self::UnmaskedIndexedMisalignment if step == 0 => {
                Some(AccessModel::UnmaskedIndexedMisalignmentProbe)
            }
            Self::WidenZeroExtend if step == 0 => Some(AccessModel::WidenZeroExtendSlideObserve),
            Self::VstartMaskedStore if step == 0 => Some(AccessModel::VstartMaskedStoreProbe),
            Self::MaskMergeV0Overlap if step <= 3 => {
                if !pool.mask_live {
                    Some(AccessModel::ProduceMask)
                } else if pool.live_vecs.len() < 2 {
                    Some(AccessModel::LoadVec)
                } else {
                    Some(AccessModel::MaskMergeState)
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StateFlowOptions {
    constraints_enabled: bool,
    sequence_enabled: bool,
    scalar_interleave_enabled: bool,
    rvv_setup_enabled: bool,
    ara_generic_enabled: bool,
    ara_risk_focus: AraRiskFocus,
    ara_safe_enabled: bool,
    coverage_focused: bool,
    positive_memory: bool,
    indexed_alignment: IndexedAlignmentScenario,
    episode_count: usize,
    steps_per_episode: usize,
}

impl StateFlowOptions {
    fn for_flavor(flavor: RvvStateFlowFlavor) -> Self {
        let mut options = match flavor {
            RvvStateFlowFlavor::StateFlowOnly => Self {
                constraints_enabled: false,
                sequence_enabled: true,
                scalar_interleave_enabled: true,
                rvv_setup_enabled: false,
                ara_generic_enabled: false,
                ara_risk_focus: AraRiskFocus::Auto,
                ara_safe_enabled: false,
                coverage_focused: false,
                positive_memory: false,
                indexed_alignment: IndexedAlignmentScenario::Aligned,
                episode_count: 2,
                steps_per_episode: 8,
            },
            RvvStateFlowFlavor::StateFlowConstrained | RvvStateFlowFlavor::PositiveConstrained => {
                Self {
                    constraints_enabled: true,
                    sequence_enabled: true,
                    scalar_interleave_enabled: true,
                    rvv_setup_enabled: false,
                    ara_generic_enabled: false,
                    ara_risk_focus: AraRiskFocus::Auto,
                    ara_safe_enabled: false,
                    coverage_focused: false,
                    positive_memory: false,
                    indexed_alignment: IndexedAlignmentScenario::Aligned,
                    episode_count: 2,
                    steps_per_episode: 8,
                }
            }
            RvvStateFlowFlavor::AraGeneric | RvvStateFlowFlavor::PositiveAraGeneric => Self {
                constraints_enabled: true,
                sequence_enabled: true,
                scalar_interleave_enabled: true,
                rvv_setup_enabled: false,
                ara_generic_enabled: true,
                ara_risk_focus: AraRiskFocus::from_env(),
                ara_safe_enabled: false,
                coverage_focused: false,
                positive_memory: false,
                indexed_alignment: IndexedAlignmentScenario::Aligned,
                episode_count: 2,
                steps_per_episode: 8,
            },
            RvvStateFlowFlavor::AraNoConstraints => Self {
                constraints_enabled: false,
                sequence_enabled: true,
                scalar_interleave_enabled: true,
                rvv_setup_enabled: false,
                ara_generic_enabled: true,
                ara_risk_focus: AraRiskFocus::from_env(),
                ara_safe_enabled: false,
                coverage_focused: false,
                positive_memory: false,
                indexed_alignment: IndexedAlignmentScenario::Aligned,
                episode_count: 2,
                steps_per_episode: 8,
            },
            RvvStateFlowFlavor::AraNoSequence => Self {
                constraints_enabled: true,
                sequence_enabled: false,
                scalar_interleave_enabled: true,
                rvv_setup_enabled: false,
                ara_generic_enabled: false,
                ara_risk_focus: AraRiskFocus::Auto,
                ara_safe_enabled: false,
                coverage_focused: false,
                positive_memory: false,
                indexed_alignment: IndexedAlignmentScenario::Aligned,
                episode_count: 1,
                steps_per_episode: 8,
            },
            RvvStateFlowFlavor::AraNoConstraintsNoSequence => Self {
                constraints_enabled: false,
                sequence_enabled: false,
                scalar_interleave_enabled: true,
                rvv_setup_enabled: true,
                ara_generic_enabled: false,
                ara_risk_focus: AraRiskFocus::Auto,
                ara_safe_enabled: false,
                coverage_focused: false,
                positive_memory: false,
                indexed_alignment: IndexedAlignmentScenario::Aligned,
                episode_count: 1,
                steps_per_episode: 8,
            },
            RvvStateFlowFlavor::AraSafe | RvvStateFlowFlavor::CoverageConservative => Self {
                constraints_enabled: true,
                sequence_enabled: true,
                scalar_interleave_enabled: true,
                rvv_setup_enabled: false,
                ara_generic_enabled: false,
                ara_risk_focus: AraRiskFocus::Auto,
                ara_safe_enabled: true,
                coverage_focused: false,
                positive_memory: false,
                indexed_alignment: IndexedAlignmentScenario::Aligned,
                episode_count: 2,
                steps_per_episode: 6,
            },
        };
        options.positive_memory = matches!(
            flavor,
            RvvStateFlowFlavor::PositiveConstrained | RvvStateFlowFlavor::PositiveAraGeneric
        );
        // The random-program scenario schedule is seed based. Keep an
        // unrelated process-level risk focus from replacing its probe.
        if options.positive_memory {
            options.ara_risk_focus = AraRiskFocus::Auto;
        }
        if matches!(
            flavor,
            RvvStateFlowFlavor::AraGeneric | RvvStateFlowFlavor::StateFlowConstrained
        ) && env::var("RISCV_FUZZ_RVV_COVERAGE_PROFILE").is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        }) {
            options.coverage_focused = true;
            // Preserve two producer/consumer episodes while letting more cases
            // finish within the fixed RTL timeout.
            options.steps_per_episode = 6;
        }
        options
    }

    fn register_constraints_enabled(self) -> bool {
        self.constraints_enabled || self.ara_safe_enabled
    }

    fn vector_setup_enabled(self) -> bool {
        self.constraints_enabled || self.rvv_setup_enabled || self.ara_safe_enabled
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RvvStatePool {
    config: RvvLiveConfig,
    vector_range: (u8, u8),
    live_vecs: Vec<LiveVecState>,
    used_groups: Vec<(u8, u8)>,
    mask_live: bool,
    next_state_id: u32,
}

impl RvvStatePool {
    fn new(config: RvvLiveConfig, vector_range: (u8, u8)) -> Self {
        Self {
            config,
            vector_range,
            live_vecs: Vec::new(),
            used_groups: Vec::new(),
            mask_live: false,
            next_state_id: 0,
        }
    }

    fn record_group(&mut self, base: u8, width: u8) {
        self.live_vecs.retain(|value| {
            !crate::rvv_legal::groups_overlap(base, width, value.base, value.width)
        });
        self.used_groups.retain(|(start, end)| {
            !crate::rvv_legal::groups_overlap(base, width, *start, end - start + 1)
        });
        self.used_groups
            .push((base, base.saturating_add(width.saturating_sub(1))));
    }

    fn set_config(&mut self, config: RvvLiveConfig) {
        self.config = config;
    }

    fn allocate_constrained_group<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> u8 {
        self.allocate_constrained_group_with_width(rng, self.config.group_width())
    }

    fn allocate_constrained_group_with_width<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        width: u8,
    ) -> u8 {
        let candidates = crate::rvv_legal::legal_group_bases(self.vector_range, width);
        let candidates = if candidates.is_empty() {
            crate::rvv_legal::legal_group_bases((1, 31), width)
        } else {
            candidates
        };
        let non_overlapping = candidates
            .iter()
            .copied()
            .filter(|base| {
                let group = (*base, base.saturating_add(width.saturating_sub(1)));
                self.used_groups
                    .iter()
                    .all(|used| !group_overlaps(group, *used))
            })
            .collect::<Vec<_>>();
        let pool = if non_overlapping.is_empty() {
            candidates
        } else {
            non_overlapping
        };
        let base = pool[rng.random_range(0..pool.len())];
        self.record_group(base, width);
        base
    }

    fn allocate_ambient_group<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> u8 {
        let base = rng.random_range(1..=31);
        self.record_group(base, 1);
        base
    }

    fn allocate_group<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        constraints_enabled: bool,
    ) -> u8 {
        if constraints_enabled {
            self.allocate_constrained_group(rng)
        } else {
            self.allocate_ambient_group(rng)
        }
    }

    fn produce_vec<R: rand::Rng + ?Sized>(&mut self, rng: &mut R, constraints_enabled: bool) -> u8 {
        let base = self.allocate_group(rng, constraints_enabled);
        self.push_vec(base, self.config.group_width());
        base
    }

    fn produce_vec_disjoint_from<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        source_base: u8,
    ) -> u8 {
        // vslideup has an opcode-specific non-overlap rule, independent of
        // ordinary same-EEW overlap. Do not fall back to the source group
        // when the general live-group pool is exhausted.
        let width = self.config.group_width();
        let safe = |range| {
            crate::rvv_legal::legal_group_bases(range, width)
                .into_iter()
                .filter(|base| !crate::rvv_legal::groups_overlap(*base, width, source_base, width))
                .collect::<Vec<_>>()
        };
        let candidates = safe(self.vector_range);
        let candidates = if candidates.is_empty() {
            safe((1, 31))
        } else {
            candidates
        };
        let unused = candidates
            .iter()
            .copied()
            .filter(|base| {
                self.used_groups.iter().all(|(start, end)| {
                    !crate::rvv_legal::groups_overlap(*base, width, *start, end - start + 1)
                })
            })
            .collect::<Vec<_>>();
        let pool = if unused.is_empty() {
            &candidates
        } else {
            &unused
        };
        let base = pool[rng.random_range(0..pool.len())];
        self.record_group(base, width);
        self.push_vec(base, width);
        base
    }

    fn produce_vec_for_width<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        width: u8,
        constraints_enabled: bool,
    ) -> u8 {
        let base = if constraints_enabled {
            self.allocate_constrained_group_with_width(rng, width)
        } else {
            self.allocate_ambient_group(rng)
        };
        self.push_vec(base, width);
        base
    }

    fn push_vec(&mut self, base: u8, width: u8) {
        let id = self.next_state_id;
        self.next_state_id = self.next_state_id.saturating_add(1);
        self.live_vecs.push(LiveVecState { id, base, width });
    }

    fn consume_vec<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> u8 {
        self.consume_vec_for_width(rng, self.config.group_width())
    }

    fn consume_vec_for_width<R: rand::Rng + ?Sized>(&self, rng: &mut R, width: u8) -> u8 {
        let compatible = self
            .live_vecs
            .iter()
            .filter(|state| crate::rvv_legal::valid_group(state.base, width))
            .collect::<Vec<_>>();
        let choices = if compatible.is_empty() {
            self.live_vecs.iter().collect::<Vec<_>>()
        } else {
            compatible
        };
        choices[rng.random_range(0..choices.len())].base
    }

    fn consume_latest_vec(&self) -> u8 {
        self.live_vecs
            .iter()
            .rev()
            .find(|state| crate::rvv_legal::valid_group(state.base, self.config.group_width()))
            .or_else(|| self.live_vecs.last())
            .map(|state| state.base)
            .expect("state-flow generation must create a producer before a consumer")
    }

    fn consume_two_vecs<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> (u8, u8) {
        let compatible = self
            .live_vecs
            .iter()
            .filter(|state| crate::rvv_legal::valid_group(state.base, self.config.group_width()))
            .collect::<Vec<_>>();
        let compatible = if compatible.is_empty() {
            self.live_vecs.iter().collect::<Vec<_>>()
        } else {
            compatible
        };
        if compatible.len() == 1 {
            let base = compatible[0].base;
            return (base, base);
        }
        let left_idx = rng.random_range(0..compatible.len());
        let mut right_idx = rng.random_range(0..compatible.len());
        if right_idx == left_idx {
            right_idx = (right_idx + 1) % compatible.len();
        }
        (compatible[left_idx].base, compatible[right_idx].base)
    }

    fn ensure_compatible_live<R: rand::Rng + ?Sized>(
        &mut self,
        lines: &mut Vec<String>,
        rng: &mut R,
    ) {
        let width = self.config.group_width();
        if self
            .live_vecs
            .iter()
            .any(|state| crate::rvv_legal::valid_group(state.base, width))
        {
            return;
        }
        let base = self.produce_vec(rng, true);
        lines.push(format!("vmv.v.i v{base}, 0"));
    }

    fn allocate_segment_base<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        field_count: u8,
        constraints_enabled: bool,
    ) -> u8 {
        if !constraints_enabled {
            let base = rng.random_range(1..=31);
            self.record_group(base, field_count);
            return base;
        }
        let candidates = self
            .config
            .legal_segment_bases(self.vector_range, field_count);
        let candidates = if candidates.is_empty() {
            RvvLiveConfig {
                cfg: TemplateRvvConfig {
                    sew: 32,
                    vl: 4,
                    lmul: "m1",
                    tail_policy: "tu",
                    mask_policy: "mu",
                },
            }
            .legal_segment_bases((1, 31), field_count)
        } else {
            candidates
        };
        let width = self.config.group_width().saturating_mul(field_count);
        let non_overlapping = candidates
            .iter()
            .copied()
            .filter(|base| {
                let group = (*base, base.saturating_add(width.saturating_sub(1)));
                self.used_groups
                    .iter()
                    .all(|used| !group_overlaps(group, *used))
            })
            .collect::<Vec<_>>();
        let pool = if non_overlapping.is_empty() {
            candidates
        } else {
            non_overlapping
        };
        let base = pool[rng.random_range(0..pool.len())];
        self.record_group(base, width);
        base
    }

    fn allocate_indexed_segment_pair<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        field_count: u8,
        index_eew: u16,
        preferred_base: Option<u8>,
    ) -> (u8, u8) {
        let data_width = self.config.group_width();
        let total_width = data_width.saturating_mul(field_count);
        let index_width =
            crate::rvv_legal::register_width(self.config.cfg.sew, self.config.cfg.lmul, index_eew)
                .expect("indexed-segment configuration must be legal");
        let pairs_for = |range| {
            self.config
                .legal_segment_bases(range, field_count)
                .into_iter()
                .flat_map(|base| {
                    crate::rvv_legal::legal_group_bases(range, index_width)
                        .into_iter()
                        .filter(move |index| {
                            !crate::rvv_legal::groups_overlap(
                                base,
                                total_width,
                                *index,
                                index_width,
                            )
                        })
                        .map(move |index| (base, index))
                })
                .collect::<Vec<_>>()
        };
        let mut pairs = pairs_for(self.vector_range);
        if pairs.is_empty() {
            pairs = pairs_for((1, 31));
        }
        if let Some(preferred) = preferred_base {
            let preferred_pairs = pairs
                .iter()
                .copied()
                .filter(|(base, _)| *base == preferred)
                .collect::<Vec<_>>();
            if !preferred_pairs.is_empty() {
                pairs = preferred_pairs;
            }
        }
        let free = pairs
            .iter()
            .copied()
            .filter(|(base, index)| {
                self.used_groups.iter().all(|(start, end)| {
                    let width = end - start + 1;
                    !crate::rvv_legal::groups_overlap(*base, total_width, *start, width)
                        && !crate::rvv_legal::groups_overlap(*index, index_width, *start, width)
                })
            })
            .collect::<Vec<_>>();
        let pool = if free.is_empty() { &pairs } else { &free };
        let (base, index) = pool[rng.random_range(0..pool.len())];
        self.record_group(index, index_width);
        self.record_group(base, total_width);
        (base, index)
    }

    fn allocate_preferred_segment_base<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        field_count: u8,
        constraints_enabled: bool,
        preferred: u8,
    ) -> u8 {
        if constraints_enabled {
            let width = self.config.group_width().saturating_mul(field_count);
            let group = (preferred, preferred.saturating_add(width.saturating_sub(1)));
            let legal = self
                .config
                .legal_segment_bases(self.vector_range, field_count)
                .contains(&preferred);
            let free = self
                .used_groups
                .iter()
                .all(|used| !group_overlaps(group, *used));
            if legal && free {
                self.record_group(preferred, width);
                return preferred;
            }
        }
        self.allocate_segment_base(rng, field_count, constraints_enabled)
    }

    fn allocate_whole_register_base<R: rand::Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        width: u8,
        constraints_enabled: bool,
    ) -> u8 {
        if !constraints_enabled {
            let base = rng.random_range(1..=31);
            self.record_group(base, width);
            return base;
        }
        let candidates = RvvLiveConfig::legal_whole_register_bases(self.vector_range, width);
        let candidates = if candidates.is_empty() {
            RvvLiveConfig::legal_whole_register_bases((1, 31), width)
        } else {
            candidates
        };
        let non_overlapping = candidates
            .iter()
            .copied()
            .filter(|base| {
                let group = (*base, base.saturating_add(width.saturating_sub(1)));
                self.used_groups
                    .iter()
                    .all(|used| !group_overlaps(group, *used))
            })
            .collect::<Vec<_>>();
        let pool = if non_overlapping.is_empty() {
            candidates
        } else {
            non_overlapping
        };
        let base = pool[rng.random_range(0..pool.len())];
        self.record_group(base, width);
        base
    }

    fn push_vec_group(&mut self, base: u8, width: u8) {
        let field_width = self.config.group_width();
        for field in 0..width {
            let reg = base.saturating_add(field.saturating_mul(field_width));
            if reg < 32 {
                self.push_vec(reg, field_width);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum AccessModel {
    ProduceMask,
    LoadVec,
    AluVx,
    CompareToMask,
    MergeWithMask,
    MaskedMemoryRoundTrip,
    MaskMemoryRoundTrip,
    MaskMergeState,
    UnitFofLoad,
    SegmentFofLoad,
    IndexedLoad,
    IndexedSegmentLoad,
    StridedSegmentLoad,
    CarryWithMask,
    MaskLogical,
    SegmentLoad,
    TailHighTransition,
    VstartMaskedStore,
    VstartMaskedStoreProbe,
    SlideWithMask,
    LargeMaskedSlideWindowObserve,
    WidenZeroExtendSlideObserve,
    IndexedMisalignmentSensitivity,
    MaskedIndexedMisalignmentProbe,
    UnmaskedIndexedMisalignmentProbe,
    SegmentLaterFieldObserve,
    SegmentLaterFieldStrictObserve,
    SparseMaskedSegmentLaneObserve,
    MaskMergeV0OverlapProbe,
    ConfigReadback,
    FractionalGather,
    FpScalarObserve,
    WidenZeroExtend,
    WholeRegister4RoundTrip,
    WholeRegister8RoundTrip,
    WholeRegister8Load,
    StridedSegmentE64,
    FaultingStridedStore,
    ScalarObserve,
}

// A pure VC/SF run may ask for a semantic relation, not a fixed testcase.
// The selected access still samples its configuration, registers, and data.
fn focused_access_from_env() -> Option<AccessModel> {
    let raw = std::env::var("RISCV_FUZZ_RVV_SF_FOCUS").ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "config-vl" => Some(AccessModel::ConfigReadback),
        "segment-fields" => Some(AccessModel::SegmentLaterFieldStrictObserve),
        "masked-indexed" => Some(AccessModel::MaskedIndexedMisalignmentProbe),
        "unmasked-indexed" => Some(AccessModel::UnmaskedIndexedMisalignmentProbe),
        "vstart-store" => Some(AccessModel::VstartMaskedStoreProbe),
        "masked-slide" => Some(AccessModel::LargeMaskedSlideWindowObserve),
        "mask-segment" => Some(AccessModel::SparseMaskedSegmentLaneObserve),
        "unit-fof" => Some(AccessModel::UnitFofLoad),
        "segment-fof" => Some(AccessModel::SegmentFofLoad),
        "fault-store" => Some(AccessModel::FaultingStridedStore),
        _ => None,
    }
}

fn indexed_alignment_for_access(
    requested: IndexedAlignmentScenario,
    access: AccessModel,
    focused_access: Option<AccessModel>,
    scheduled_indexed: bool,
) -> IndexedAlignmentScenario {
    if focused_access == Some(access) {
        match access {
            AccessModel::MaskedIndexedMisalignmentProbe => {
                return IndexedAlignmentScenario::MaskedOffMisaligned;
            }
            AccessModel::UnmaskedIndexedMisalignmentProbe => {
                return IndexedAlignmentScenario::ActiveMisaligned;
            }
            _ => {}
        }
    }
    if scheduled_indexed {
        requested
    } else {
        IndexedAlignmentScenario::Aligned
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct AraRiskCost {
    heavy_memory: bool,
    segment: bool,
    fof: bool,
    indexed: bool,
    whole_register: bool,
    vstart_store: bool,
    slide: bool,
    faulting_strided_store: bool,
}

impl AraRiskCost {
    fn for_access(access: AccessModel) -> Self {
        match access {
            AccessModel::UnitFofLoad => Self {
                heavy_memory: true,
                fof: true,
                ..Self::default()
            },
            AccessModel::SegmentFofLoad => Self {
                heavy_memory: true,
                segment: true,
                fof: true,
                ..Self::default()
            },
            AccessModel::IndexedLoad
            | AccessModel::IndexedMisalignmentSensitivity
            | AccessModel::MaskedIndexedMisalignmentProbe
            | AccessModel::UnmaskedIndexedMisalignmentProbe => Self {
                heavy_memory: true,
                indexed: true,
                ..Self::default()
            },
            AccessModel::IndexedSegmentLoad => Self {
                heavy_memory: true,
                segment: true,
                indexed: true,
                ..Self::default()
            },
            AccessModel::SegmentLaterFieldObserve | AccessModel::SegmentLaterFieldStrictObserve => {
                Self {
                    heavy_memory: true,
                    segment: true,
                    indexed: true,
                    ..Self::default()
                }
            }
            AccessModel::SparseMaskedSegmentLaneObserve => Self {
                heavy_memory: true,
                segment: true,
                ..Self::default()
            },
            AccessModel::StridedSegmentLoad
            | AccessModel::StridedSegmentE64
            | AccessModel::SegmentLoad => Self {
                heavy_memory: true,
                segment: true,
                ..Self::default()
            },
            AccessModel::WholeRegister4RoundTrip
            | AccessModel::WholeRegister8RoundTrip
            | AccessModel::WholeRegister8Load => Self {
                heavy_memory: true,
                whole_register: true,
                ..Self::default()
            },
            AccessModel::FaultingStridedStore => Self {
                heavy_memory: true,
                faulting_strided_store: true,
                ..Self::default()
            },
            AccessModel::VstartMaskedStore | AccessModel::VstartMaskedStoreProbe => Self {
                vstart_store: true,
                ..Self::default()
            },
            AccessModel::SlideWithMask
            | AccessModel::LargeMaskedSlideWindowObserve
            | AccessModel::WidenZeroExtendSlideObserve => Self {
                slide: true,
                ..Self::default()
            },
            _ => Self::default(),
        }
    }

    fn is_specialized(self) -> bool {
        self.heavy_memory || self.vstart_store || self.slide
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct AraRiskBudget {
    heavy_memory_ops: usize,
    segment_ops: usize,
    fof_ops: usize,
    indexed_ops: usize,
    whole_register_ops: usize,
    vstart_store_ops: usize,
    slide_ops: usize,
    faulting_strided_store_ops: usize,
    last_was_heavy_memory: bool,
}

impl AraRiskBudget {
    const MAX_HEAVY_MEMORY_OPS: usize = 4;
    const MAX_SEGMENT_OPS: usize = 2;
    const MAX_FOF_OPS: usize = 1;
    const MAX_INDEXED_OPS: usize = 2;
    const MAX_WHOLE_REGISTER_OPS: usize = 1;
    const MAX_VSTART_STORE_OPS: usize = 1;
    const MAX_SLIDE_OPS: usize = 1;
    const MAX_FAULTING_STRIDED_STORE_OPS: usize = 1;

    fn allows(&self, access: AccessModel) -> bool {
        let cost = AraRiskCost::for_access(access);
        if !cost.is_specialized() {
            return true;
        }
        if cost.heavy_memory {
            if self.heavy_memory_ops >= Self::MAX_HEAVY_MEMORY_OPS {
                return false;
            }
            if self.last_was_heavy_memory {
                return false;
            }
        }
        if cost.segment && self.segment_ops >= Self::MAX_SEGMENT_OPS {
            return false;
        }
        if cost.fof && self.fof_ops >= Self::MAX_FOF_OPS {
            return false;
        }
        if cost.indexed && self.indexed_ops >= Self::MAX_INDEXED_OPS {
            return false;
        }
        if cost.whole_register && self.whole_register_ops >= Self::MAX_WHOLE_REGISTER_OPS {
            return false;
        }
        if cost.vstart_store && self.vstart_store_ops >= Self::MAX_VSTART_STORE_OPS {
            return false;
        }
        if cost.slide && self.slide_ops >= Self::MAX_SLIDE_OPS {
            return false;
        }
        if cost.faulting_strided_store
            && self.faulting_strided_store_ops >= Self::MAX_FAULTING_STRIDED_STORE_OPS
        {
            return false;
        }
        true
    }

    fn record(&mut self, access: AccessModel) {
        let cost = AraRiskCost::for_access(access);
        if cost.heavy_memory {
            self.heavy_memory_ops = self.heavy_memory_ops.saturating_add(1);
            self.last_was_heavy_memory = true;
        } else {
            self.last_was_heavy_memory = false;
        }
        if cost.segment {
            self.segment_ops = self.segment_ops.saturating_add(1);
        }
        if cost.fof {
            self.fof_ops = self.fof_ops.saturating_add(1);
        }
        if cost.indexed {
            self.indexed_ops = self.indexed_ops.saturating_add(1);
        }
        if cost.whole_register {
            self.whole_register_ops = self.whole_register_ops.saturating_add(1);
        }
        if cost.vstart_store {
            self.vstart_store_ops = self.vstart_store_ops.saturating_add(1);
        }
        if cost.slide {
            self.slide_ops = self.slide_ops.saturating_add(1);
        }
        if cost.faulting_strided_store {
            self.faulting_strided_store_ops = self.faulting_strided_store_ops.saturating_add(1);
        }
    }
}

fn isa_decode_context(isa_base: ISABase) -> IsaDecodeContext {
    match isa_base {
        ISABase::Rv32 => IsaDecodeContext::RV32,
        ISABase::Rv64 => IsaDecodeContext::RV64,
    }
}

fn decode_instruction(
    asm: &str,
    isa_base: ISABase,
) -> Result<InstructionSequence<RiscvInstruction>, GenerationError> {
    let instruction = RiscvInstruction::decode_asm_with_context(asm, isa_decode_context(isa_base))
        .map_err(|source| GenerationError::StructuredInstructionBuildFailed {
            message: format!("failed to decode RVV state-flow instruction `{asm}`: {source}"),
        })?;
    Ok(InstructionSequence::new(instruction))
}

fn decode_block(
    lines: &[String],
    isa_base: ISABase,
) -> Result<Vec<InstructionSequence<RiscvInstruction>>, GenerationError> {
    lines
        .iter()
        .map(|line| decode_instruction(line, isa_base))
        .collect()
}

fn instruction_mnemonic(instruction: &RiscvInstruction) -> String {
    instruction
        .to_string()
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn is_rvv_mnemonic(mnemonic: &str) -> bool {
    mnemonic.starts_with('v')
}

fn remove_ambient_rvv_instructions_for_phase(
    testcase: &mut GeneratedTestCase,
    phase: HartPhase,
) -> usize {
    let mut removed = 0usize;
    for hart in testcase.harts.iter_mut() {
        let mapping = {
            let sequences = hart.sequences_mut(phase);
            let old = std::mem::take(sequences);
            let mut mapping = vec![None; old.len()];
            let mut kept = Vec::with_capacity(old.len());
            for (old_idx, sequence) in old.into_iter().enumerate() {
                let mnemonic = instruction_mnemonic(&sequence.instruction);
                if is_rvv_mnemonic(&mnemonic) {
                    removed = removed.saturating_add(1);
                } else {
                    mapping[old_idx] = Some(kept.len());
                    kept.push(sequence);
                }
            }
            *sequences = kept;
            mapping
        };

        let shared = hart.shared_sequence_indices_mut(phase);
        *shared = shared
            .iter()
            .filter_map(|idx| mapping.get(*idx).copied().flatten())
            .collect();
    }
    removed
}

pub fn remove_ambient_rvv_test_instructions(testcase: &mut GeneratedTestCase) -> usize {
    remove_ambient_rvv_instructions_for_phase(testcase, HartPhase::Test)
}

/// Keep the random RVV component broad while leaving complex memory forms to
/// the state-flow models that can enforce NF, EEW, EMUL and alignment together.
pub fn remove_ambient_complex_memory_test_instructions(testcase: &mut GeneratedTestCase) -> usize {
    const SEGMENT_PREFIXES: &[&str] = &[
        "vlseg", "vsseg", "vlsseg", "vssseg", "vluxseg", "vloxseg", "vsuxseg", "vsoxseg",
    ];
    let mut removed = 0usize;
    for hart in testcase.harts.iter_mut() {
        let mapping = {
            let sequences = hart.sequences_mut(HartPhase::Test);
            let old = std::mem::take(sequences);
            let mut mapping = vec![None; old.len()];
            let mut kept = Vec::with_capacity(old.len());
            for (old_idx, sequence) in old.into_iter().enumerate() {
                let mnemonic = instruction_mnemonic(&sequence.instruction);
                if SEGMENT_PREFIXES
                    .iter()
                    .any(|prefix| mnemonic.starts_with(prefix))
                    || mnemonic.contains("ff.v")
                {
                    removed = removed.saturating_add(1);
                } else {
                    mapping[old_idx] = Some(kept.len());
                    kept.push(sequence);
                }
            }
            *sequences = kept;
            mapping
        };
        let shared = hart.shared_sequence_indices_mut(HartPhase::Test);
        *shared = shared
            .iter()
            .filter_map(|idx| mapping.get(*idx).copied().flatten())
            .collect();
    }
    removed
}

pub fn remove_ambient_rvv_init_instructions(testcase: &mut GeneratedTestCase) -> usize {
    remove_ambient_rvv_instructions_for_phase(testcase, HartPhase::Init)
}

pub fn clear_ambient_test_instructions(testcase: &mut GeneratedTestCase) -> usize {
    let mut removed = 0usize;
    for hart in testcase.harts.iter_mut() {
        let test = hart.sequences_mut(HartPhase::Test);
        removed = removed.saturating_add(test.len());
        test.clear();
        hart.shared_sequence_indices_mut(HartPhase::Test).clear();
    }
    removed
}

/// Keep only randomly generated RVV test instructions in the test phase.
///
/// This is used by the ARA no-constraints/no-sequence ablation as a pure
/// random-RVV baseline: no state-flow block, no risk-family probe, and no RVV
/// operand/config repair pass.
pub fn retain_only_ambient_rvv_test_instructions(testcase: &mut GeneratedTestCase) -> usize {
    let mut removed = 0usize;
    for hart in testcase.harts.iter_mut() {
        let mapping = {
            let sequences = hart.sequences_mut(HartPhase::Test);
            let old = std::mem::take(sequences);
            let mut mapping = vec![None; old.len()];
            let mut kept = Vec::with_capacity(old.len());
            for (old_idx, sequence) in old.into_iter().enumerate() {
                let mnemonic = instruction_mnemonic(&sequence.instruction);
                if is_rvv_mnemonic(&mnemonic) {
                    mapping[old_idx] = Some(kept.len());
                    kept.push(sequence);
                } else {
                    removed = removed.saturating_add(1);
                }
            }
            *sequences = kept;
            mapping
        };

        let shared = hart.shared_sequence_indices_mut(HartPhase::Test);
        *shared = shared
            .iter()
            .filter_map(|idx| mapping.get(*idx).copied().flatten())
            .collect();
    }
    removed
}

fn aligned_stateflow_address(
    start: u64,
    end: u64,
    model_bytes: u64,
) -> Result<u64, GenerationError> {
    let aligned_start = start.div_ceil(8).saturating_mul(8);
    let preferred = aligned_start.saturating_add(0x40);
    if preferred.saturating_add(model_bytes) <= end {
        Ok(preferred)
    } else if aligned_start.saturating_add(model_bytes) <= end {
        Ok(aligned_start)
    } else {
        Err(GenerationError::StructuredInstructionBuildFailed {
            message: format!(
                "RVV state-flow needs at least {model_bytes} writable bytes in mem range {start:#x}..{end:#x}"
            ),
        })
    }
}

fn signed_literal(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

fn push_data_words(lines: &mut Vec<String>, base_reg: &str, start_offset: i32, words: &[u64]) {
    for (idx, value) in words.iter().copied().enumerate() {
        lines.push(format!("li x14, {}", signed_literal(value)));
        lines.push(format!(
            "sd x14, {}({base_reg})",
            start_offset + (idx as i32 * 8)
        ));
    }
}

fn biased_word32<R: rand::Rng + ?Sized>(rng: &mut R, lane: u8, salt: u16) -> u64 {
    let hi = rng.random_range(0x10u64..0xf0);
    let mid = u64::from(salt & 0x00ff);
    let lo = u64::from(lane).saturating_mul(0x11) ^ rng.random_range(0u64..0x100);
    ((hi << 24) | (mid << 16) | (u64::from(lane) << 8) | lo) & 0xffff_ffff
}

fn push_data_words32(lines: &mut Vec<String>, base_reg: &str, words: &[u64]) {
    for (idx, value) in words.iter().copied().enumerate() {
        lines.push(format!("li x14, {}", value & 0xffff_ffff));
        lines.push(format!("sw x14, {}({base_reg})", idx as i32 * 4));
    }
}

fn group_overlaps(left: (u8, u8), right: (u8, u8)) -> bool {
    left.0 <= right.1 && right.0 <= left.1
}

fn choose_constrained_config<R: rand::Rng + ?Sized>(
    rng: &mut R,
    vector_range: (u8, u8),
) -> RvvLiveConfig {
    let choices = [
        TemplateRvvConfig {
            sew: 32,
            vl: 8,
            lmul: "m1",
            tail_policy: "tu",
            mask_policy: "mu",
        },
        TemplateRvvConfig {
            sew: 32,
            vl: 8,
            lmul: "m2",
            tail_policy: "tu",
            mask_policy: "mu",
        },
        TemplateRvvConfig {
            sew: 32,
            vl: 8,
            lmul: "m4",
            tail_policy: "tu",
            mask_policy: "mu",
        },
        TemplateRvvConfig {
            sew: 32,
            vl: 4,
            lmul: "m1",
            tail_policy: "ta",
            mask_policy: "mu",
        },
        TemplateRvvConfig {
            sew: 32,
            vl: 6,
            lmul: "m1",
            tail_policy: "tu",
            mask_policy: "ma",
        },
        TemplateRvvConfig {
            sew: 64,
            vl: 8,
            lmul: "m1",
            tail_policy: "tu",
            mask_policy: "mu",
        },
        TemplateRvvConfig {
            sew: 64,
            vl: 4,
            lmul: "m1",
            tail_policy: "ta",
            mask_policy: "mu",
        },
    ];
    let legal = choices
        .iter()
        .copied()
        .map(|cfg| RvvLiveConfig { cfg })
        .filter(|cfg| cfg.legal_group_bases(vector_range).len() >= 4)
        .collect::<Vec<_>>();
    let pool = if legal.is_empty() {
        vec![RvvLiveConfig { cfg: choices[0] }]
    } else {
        legal
    };
    pool[rng.random_range(0..pool.len())]
}

fn choose_ara_generic_config<R: rand::Rng + ?Sized>(
    rng: &mut R,
    vector_range: (u8, u8),
) -> RvvLiveConfig {
    let choices = [
        TemplateRvvConfig {
            sew: 32,
            vl: 4,
            lmul: "m1",
            tail_policy: "tu",
            mask_policy: "mu",
        },
        TemplateRvvConfig {
            sew: 32,
            vl: 4,
            lmul: "m2",
            tail_policy: "tu",
            mask_policy: "mu",
        },
        TemplateRvvConfig {
            sew: 32,
            vl: 4,
            lmul: "m4",
            tail_policy: "tu",
            mask_policy: "mu",
        },
        TemplateRvvConfig {
            sew: 64,
            vl: 4,
            lmul: "m1",
            tail_policy: "tu",
            mask_policy: "mu",
        },
    ];
    let legal = choices
        .iter()
        .copied()
        .map(|cfg| RvvLiveConfig { cfg })
        .filter(|cfg| cfg.legal_group_bases(vector_range).len() >= 4)
        .collect::<Vec<_>>();
    let pool = if legal.is_empty() {
        vec![RvvLiveConfig { cfg: choices[0] }]
    } else {
        legal
    };
    pool[rng.random_range(0..pool.len())]
}

fn choose_ambient_config<R: rand::Rng + ?Sized>(rng: &mut R) -> RvvLiveConfig {
    const SEW_CHOICES: &[u16] = &[8, 16, 32, 64];
    const LMUL_CHOICES: &[&str] = &["m1", "m2", "m4", "m8", "mf2", "mf4", "mf8"];
    RvvLiveConfig {
        cfg: TemplateRvvConfig {
            sew: choose_from_slice(rng, SEW_CHOICES),
            vl: rng.random_range(0..=31),
            lmul: choose_from_slice(rng, LMUL_CHOICES),
            tail_policy: if rng.random_range(0..2) == 0 {
                "tu"
            } else {
                "ta"
            },
            mask_policy: if rng.random_range(0..2) == 0 {
                "mu"
            } else {
                "ma"
            },
        },
    }
}

fn choose_stateflow_config<R: rand::Rng + ?Sized>(
    rng: &mut R,
    options: StateFlowOptions,
    vector_range: (u8, u8),
) -> RvvLiveConfig {
    if options.ara_generic_enabled && options.constraints_enabled {
        choose_ara_generic_config(rng, vector_range)
    } else if options.vector_setup_enabled() {
        choose_constrained_config(rng, vector_range)
    } else {
        choose_ambient_config(rng)
    }
}

fn scalar_noise_line<R: rand::Rng + ?Sized>(rng: &mut R) -> &'static str {
    const NOISE: &[&str] = &[
        "addi x5, x5, 1",
        "xor x6, x6, x5",
        "add x7, x7, x6",
        "andi x8, x8, 255",
        "slli x9, x9, 1",
        "srli x9, x9, 1",
        "or x6, x6, x7",
    ];
    NOISE[rng.random_range(0..NOISE.len())]
}

fn maybe_push_scalar_noise<R: rand::Rng + ?Sized>(
    lines: &mut Vec<String>,
    rng: &mut R,
    options: StateFlowOptions,
) {
    if options.scalar_interleave_enabled && rng.random_range(0..3) == 0 {
        lines.push(scalar_noise_line(rng).to_string());
    }
}

fn push_rvv_line<R: rand::Rng + ?Sized>(
    lines: &mut Vec<String>,
    rng: &mut R,
    options: StateFlowOptions,
    line: String,
) {
    lines.push(line);
    maybe_push_scalar_noise(lines, rng, options);
}

fn push_attribution_fence(lines: &mut Vec<String>) {
    lines.push("addi x6, x6, 0".to_string());
}

fn choose_vector_data_base<R: rand::Rng + ?Sized>(rng: &mut R) -> &'static str {
    if rng.random_range(0..2) == 0 {
        "x11"
    } else {
        "x12"
    }
}

fn choose_vector_stride<R: rand::Rng + ?Sized>(rng: &mut R) -> &'static str {
    if rng.random_range(0..3) == 0 {
        "x14"
    } else {
        "x13"
    }
}

fn choose_from_slice<R: rand::Rng + ?Sized, T: Copy>(rng: &mut R, values: &[T]) -> T {
    values[rng.random_range(0..values.len())]
}

fn ensure_m1_sew<R: rand::Rng + ?Sized>(
    lines: &mut Vec<String>,
    pool: &mut RvvStatePool,
    _rng: &mut R,
    options: StateFlowOptions,
    sew: u16,
) {
    if options.constraints_enabled {
        if pool.config.cfg.sew != sew || pool.config.group_width() != 1 {
            let config = RvvLiveConfig::constrained_m1_for_sew(sew);
            push_local_config_setup(lines, pool, options, config);
        }
    }
}

fn ensure_indexed_load_config<R: rand::Rng + ?Sized>(
    lines: &mut Vec<String>,
    pool: &mut RvvStatePool,
    rng: &mut R,
    options: StateFlowOptions,
) {
    if !options.positive_memory {
        ensure_m1_sew(lines, pool, rng, options, 32);
        return;
    }
    // The index preload initializes eight 32-bit offsets. A preceding dynamic
    // vsetvli can leave VL larger than that, so bound VL before both loads.
    let previous = pool.config.cfg;
    push_local_config_setup(
        lines,
        pool,
        options,
        RvvLiveConfig {
            cfg: TemplateRvvConfig {
                sew: 32,
                vl: choose_from_slice(rng, &[2, 4, 8]),
                lmul: "m1",
                tail_policy: previous.tail_policy,
                mask_policy: previous.mask_policy,
            },
        },
    );
}

fn ensure_memory_eew<R: rand::Rng + ?Sized>(
    lines: &mut Vec<String>,
    pool: &mut RvvStatePool,
    rng: &mut R,
    options: StateFlowOptions,
    eew: u16,
) -> u8 {
    let cfg = pool.config.cfg;
    if let Some(width) = crate::rvv_legal::register_width(cfg.sew, cfg.lmul, eew) {
        return width;
    }
    if !options.constraints_enabled {
        return 1;
    }
    // Resample a valid configuration for this same memory opcode.  Keep more
    // than one LMUL choice so legality does not collapse to a fixed e32/m1.
    let lmul = choose_from_slice(rng, &["m1", "m2", "m4"]);
    let config = RvvLiveConfig {
        cfg: TemplateRvvConfig {
            sew: eew,
            vl: choose_from_slice(rng, &[2, 4, 8]),
            lmul,
            tail_policy: cfg.tail_policy,
            mask_policy: cfg.mask_policy,
        },
    };
    push_local_config_setup(lines, pool, options, config);
    crate::rvv_legal::register_width(eew, lmul, eew).expect("chosen memory config is legal")
}

fn push_local_config_setup(
    lines: &mut Vec<String>,
    pool: &mut RvvStatePool,
    options: StateFlowOptions,
    config: RvvLiveConfig,
) {
    pool.set_config(config);
    if options.vector_setup_enabled() {
        lines.push(config.vsetivli_line());
        lines.push("csrrw x0, vstart, x0".to_string());
    }
}

fn push_dynamic_config_setup(
    lines: &mut Vec<String>,
    pool: &mut RvvStatePool,
    options: StateFlowOptions,
    config: RvvLiveConfig,
    rd: &str,
    rs1: &str,
) {
    pool.set_config(config);
    if options.vector_setup_enabled() {
        lines.push(format!(
            "vsetvli {rd}, {rs1}, {}",
            config.cfg.vtype_immediate()
        ));
        lines.push("csrrw x0, vstart, x0".to_string());
    }
}

fn flavor_weight(options: StateFlowOptions, constrained: u32, stateflow_only: u32) -> u32 {
    if options.constraints_enabled {
        constrained
    } else {
        stateflow_only
    }
}

fn apply_ara_risk_budget(
    options: StateFlowOptions,
    candidates: Vec<(AccessModel, u32)>,
    ara_budget: Option<&AraRiskBudget>,
) -> Vec<(AccessModel, u32)> {
    let Some(budget) = ara_budget.filter(|_| options.ara_generic_enabled) else {
        return candidates;
    };
    let filtered = candidates
        .iter()
        .copied()
        .filter(|(access, _)| budget.allows(*access))
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        candidates
            .into_iter()
            .filter(|(access, _)| !AraRiskCost::for_access(*access).heavy_memory)
            .collect()
    } else {
        filtered
    }
}

fn weighted_access_candidates(
    pool: &RvvStatePool,
    options: StateFlowOptions,
    ara_budget: Option<&AraRiskBudget>,
) -> Vec<(AccessModel, u32)> {
    let mut candidates = Vec::new();
    if !pool.mask_live {
        candidates.push((AccessModel::ProduceMask, 100));
        return apply_ara_risk_budget(options, candidates, ara_budget);
    }
    if pool.live_vecs.is_empty() {
        if options.ara_safe_enabled {
            candidates.extend([
                (AccessModel::LoadVec, 30),
                (AccessModel::MaskMemoryRoundTrip, 12),
            ]);
            return apply_ara_risk_budget(options, candidates, ara_budget);
        }
        if options.ara_generic_enabled {
            candidates.extend([
                (AccessModel::LoadVec, 18),
                (AccessModel::MaskMemoryRoundTrip, 8),
                (AccessModel::IndexedLoad, 6),
                (AccessModel::LargeMaskedSlideWindowObserve, 24),
                (AccessModel::SegmentLaterFieldStrictObserve, 24),
                (AccessModel::WidenZeroExtendSlideObserve, 14),
                (AccessModel::MaskedIndexedMisalignmentProbe, 10),
                (AccessModel::UnmaskedIndexedMisalignmentProbe, 10),
                (AccessModel::SparseMaskedSegmentLaneObserve, 8),
                (AccessModel::VstartMaskedStoreProbe, 10),
            ]);
            return apply_ara_risk_budget(options, candidates, ara_budget);
        }
        candidates.extend([
            (AccessModel::LoadVec, 26),
            (
                AccessModel::MaskMemoryRoundTrip,
                flavor_weight(options, 18, 10),
            ),
            (AccessModel::SegmentLoad, flavor_weight(options, 8, 6)),
            (
                AccessModel::StridedSegmentLoad,
                flavor_weight(options, 8, 6),
            ),
            (
                AccessModel::StridedSegmentE64,
                flavor_weight(options, 12, 9),
            ),
            (
                AccessModel::WholeRegister4RoundTrip,
                flavor_weight(options, 12, 9),
            ),
            (AccessModel::UnitFofLoad, flavor_weight(options, 8, 6)),
            (AccessModel::SegmentFofLoad, flavor_weight(options, 8, 6)),
        ]);
        return apply_ara_risk_budget(options, candidates, ara_budget);
    }

    if options.ara_safe_enabled {
        candidates.extend([
            (AccessModel::LoadVec, 12),
            (AccessModel::AluVx, 16),
            (AccessModel::CompareToMask, 12),
            (AccessModel::MaskedMemoryRoundTrip, 10),
            (AccessModel::MaskMemoryRoundTrip, 8),
            (AccessModel::MaskMergeState, 6),
            (AccessModel::CarryWithMask, 5),
            (AccessModel::MaskLogical, 7),
            (AccessModel::TailHighTransition, 5),
            (AccessModel::ScalarObserve, 8),
        ]);
        if options.constraints_enabled {
            candidates.push((AccessModel::ConfigReadback, 6));
        }
        if pool.live_vecs.len() >= 2 {
            candidates.push((AccessModel::MergeWithMask, 12));
        }
        return apply_ara_risk_budget(options, candidates, ara_budget);
    }

    candidates.extend([
        (AccessModel::LoadVec, 8),
        (AccessModel::AluVx, 12),
        (AccessModel::CompareToMask, 10),
        (
            AccessModel::MaskedMemoryRoundTrip,
            flavor_weight(options, 10, 7),
        ),
        (
            AccessModel::MaskMemoryRoundTrip,
            flavor_weight(options, 10, 6),
        ),
        (
            AccessModel::MaskMergeState,
            if options.ara_generic_enabled {
                2
            } else {
                flavor_weight(options, 6, 5)
            },
        ),
        (AccessModel::CarryWithMask, 5),
        (AccessModel::MaskLogical, 7),
        (AccessModel::TailHighTransition, 5),
        (AccessModel::FpScalarObserve, flavor_weight(options, 12, 9)),
        (AccessModel::WidenZeroExtend, flavor_weight(options, 10, 8)),
        (
            AccessModel::WholeRegister4RoundTrip,
            flavor_weight(options, 4, 3),
        ),
        (AccessModel::ScalarObserve, 6),
    ]);
    if options.constraints_enabled {
        candidates.push((AccessModel::ConfigReadback, 8));
    }
    if pool.live_vecs.len() >= 2 {
        candidates.push((AccessModel::MergeWithMask, flavor_weight(options, 12, 10)));
    }
    if !options.constraints_enabled || pool.config.group_width() == 1 {
        candidates.push((AccessModel::SegmentLoad, flavor_weight(options, 10, 7)));
        candidates.push((
            AccessModel::StridedSegmentLoad,
            flavor_weight(options, 10, 7),
        ));
        candidates.push((AccessModel::StridedSegmentE64, flavor_weight(options, 7, 5)));
        candidates.push((AccessModel::WholeRegister8RoundTrip, 1));
        candidates.push((AccessModel::UnitFofLoad, flavor_weight(options, 8, 6)));
        candidates.push((AccessModel::SegmentFofLoad, flavor_weight(options, 8, 6)));
        if options.ara_generic_enabled {
            candidates.push((AccessModel::IndexedLoad, 12));
            candidates.push((AccessModel::IndexedSegmentLoad, 12));
        }
    }
    if options.ara_generic_enabled {
        candidates.extend([
            (AccessModel::VstartMaskedStore, 8),
            (AccessModel::SlideWithMask, 5),
            (AccessModel::LargeMaskedSlideWindowObserve, 18),
            (AccessModel::WidenZeroExtendSlideObserve, 14),
            (AccessModel::IndexedMisalignmentSensitivity, 7),
            (AccessModel::MaskedIndexedMisalignmentProbe, 10),
            (AccessModel::UnmaskedIndexedMisalignmentProbe, 10),
            (AccessModel::SegmentLaterFieldObserve, 6),
            (AccessModel::SegmentLaterFieldStrictObserve, 13),
            (AccessModel::SparseMaskedSegmentLaneObserve, 7),
            (AccessModel::VstartMaskedStoreProbe, 10),
        ]);
    }
    candidates.push((
        AccessModel::FaultingStridedStore,
        flavor_weight(options, 3, 2),
    ));
    apply_ara_risk_budget(options, candidates, ara_budget)
}

fn choose_weighted_access<R: rand::Rng + ?Sized>(
    rng: &mut R,
    candidates: &[(AccessModel, u32)],
) -> AccessModel {
    let total = candidates.iter().map(|(_, weight)| *weight).sum::<u32>();
    let mut draw = rng.random_range(0..total);
    for (model, weight) in candidates {
        if draw < *weight {
            return *model;
        }
        draw -= *weight;
    }
    candidates
        .last()
        .map(|(model, _)| *model)
        .unwrap_or(AccessModel::ProduceMask)
}

fn ara_post_focus_access(pool: &RvvStatePool) -> AccessModel {
    if !pool.mask_live {
        AccessModel::ProduceMask
    } else if pool.live_vecs.is_empty() {
        AccessModel::LoadVec
    } else {
        AccessModel::ScalarObserve
    }
}

fn ara_initial_focus_access<R: rand::Rng + ?Sized>(rng: &mut R) -> AccessModel {
    let candidates = [
        (AccessModel::LargeMaskedSlideWindowObserve, 18),
        (AccessModel::SegmentLaterFieldStrictObserve, 18),
        (AccessModel::SparseMaskedSegmentLaneObserve, 10),
        (AccessModel::WidenZeroExtendSlideObserve, 18),
        (AccessModel::MaskedIndexedMisalignmentProbe, 14),
        (AccessModel::UnmaskedIndexedMisalignmentProbe, 14),
        (AccessModel::IndexedMisalignmentSensitivity, 8),
        (AccessModel::VstartMaskedStoreProbe, 10),
    ];
    choose_weighted_access(rng, &candidates)
}

fn ara_singleton_focus_access(focus: AraRiskFocus) -> Option<AccessModel> {
    match focus {
        AraRiskFocus::Auto => None,
        AraRiskFocus::LargeMaskedSlide => Some(AccessModel::LargeMaskedSlideWindowObserve),
        AraRiskFocus::SegmentLaterField => Some(AccessModel::SegmentLaterFieldStrictObserve),
        AraRiskFocus::SparseMaskedSegment => Some(AccessModel::SparseMaskedSegmentLaneObserve),
        AraRiskFocus::IndexedMisalignment => Some(AccessModel::IndexedMisalignmentSensitivity),
        AraRiskFocus::MaskedIndexedMisalignment => {
            Some(AccessModel::MaskedIndexedMisalignmentProbe)
        }
        AraRiskFocus::UnmaskedIndexedMisalignment => {
            Some(AccessModel::UnmaskedIndexedMisalignmentProbe)
        }
        AraRiskFocus::WidenZeroExtend => Some(AccessModel::WidenZeroExtendSlideObserve),
        AraRiskFocus::MaskMergeV0Overlap => Some(AccessModel::MaskMergeV0OverlapProbe),
        AraRiskFocus::VstartMaskedStore => Some(AccessModel::VstartMaskedStoreProbe),
    }
}

fn ara_singleton_access_candidates(
    options: StateFlowOptions,
    ara_budget: Option<&AraRiskBudget>,
) -> Vec<(AccessModel, u32)> {
    apply_ara_risk_budget(
        options,
        vec![
            (AccessModel::LargeMaskedSlideWindowObserve, 12),
            (AccessModel::SegmentLaterFieldStrictObserve, 12),
            (AccessModel::SparseMaskedSegmentLaneObserve, 10),
            (AccessModel::WidenZeroExtendSlideObserve, 14),
            (AccessModel::MaskedIndexedMisalignmentProbe, 14),
            (AccessModel::UnmaskedIndexedMisalignmentProbe, 14),
            (AccessModel::IndexedMisalignmentSensitivity, 6),
            (AccessModel::VstartMaskedStoreProbe, 8),
            (AccessModel::MaskMergeV0OverlapProbe, 8),
        ],
        ara_budget,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexedMisalignmentMode {
    Mixed,
    MaskedOff,
    UnmaskedTrap,
}

fn emit_indexed_misalignment_probe<R: rand::Rng + ?Sized>(
    lines: &mut Vec<String>,
    pool: &mut RvvStatePool,
    rng: &mut R,
    options: StateFlowOptions,
    mode: IndexedMisalignmentMode,
) -> usize {
    // Only four index values are initialized. Keep lanes 4..7 from reading
    // stale vector contents before the intended alignment probe.
    let mut config = RvvLiveConfig::constrained_e32_m1();
    config.cfg.vl = 4;
    push_local_config_setup(lines, pool, options, config);

    let fault_lane = if rng.random_range(0..4) == 0 {
        choose_from_slice(rng, &[1u8, 2, 3])
    } else {
        0
    };
    let misaligned = if options.positive_memory
        && options.indexed_alignment == IndexedAlignmentScenario::Aligned
    {
        choose_from_slice(rng, &[4u32, 8, 12, 16])
    } else {
        choose_from_slice(rng, &[1u32, 2, 3, 5])
    };
    for lane in 0..4u8 {
        let value = if lane == fault_lane {
            misaligned
        } else {
            u32::from(lane + 1) * 4
        };
        lines.push(format!("li x14, {value}"));
        lines.push(format!("sw x14, {}(x11)", u32::from(lane) * 4));
    }
    for (offset, value) in [
        (0, 0x1111_1111u64),
        (4, 0x2222_2222),
        (8, 0x3333_3333),
        (12, 0x4444_4444),
        (16, 0x5555_5555),
    ] {
        lines.push(format!("li x14, {value}"));
        lines.push(format!("sw x14, {offset}(x12)"));
    }

    let index = pool.produce_vec(rng, options.register_constraints_enabled());
    lines.push(format!("vle32.v v{index}, (x11)"));
    push_attribution_fence(lines);

    let vd = pool.produce_vec(rng, options.register_constraints_enabled());
    let poison = choose_from_slice(rng, &[0xdead_beefu64, 0xcafe_babe, 0xaaaa_aaaau64]);
    lines.push(format!("li x14, {poison}"));
    lines.push(format!("vmv.v.x v{vd}, x14"));

    let masked = match mode {
        IndexedMisalignmentMode::Mixed => rng.random_range(0..2) == 0,
        IndexedMisalignmentMode::MaskedOff => true,
        IndexedMisalignmentMode::UnmaskedTrap => false,
    };
    if masked {
        let mask_src = pool.produce_vec(rng, options.register_constraints_enabled());
        lines.push(format!("vid.v v{mask_src}"));
        lines.push(format!("vmsne.vi v0, v{mask_src}, {fault_lane}"));
        push_attribution_fence(lines);
        pool.mask_live = true;
        lines.push(format!("vluxei32.v v{vd}, (x12), v{index}, v0.t"));
    } else {
        lines.push(format!("vluxei32.v v{vd}, (x12), v{index}"));
    }
    push_attribution_fence(lines);
    lines.push("csrrw x15, vstart, x0".to_string());
    lines.push("sd x15, 160(x12)".to_string());
    lines.push(format!("vse32.v v{vd}, (x12)"));
    4
}

fn emit_access<R: rand::Rng + ?Sized>(
    lines: &mut Vec<String>,
    pool: &mut RvvStatePool,
    rng: &mut R,
    options: StateFlowOptions,
    access: AccessModel,
) -> usize {
    if options.register_constraints_enabled() && !pool.live_vecs.is_empty() {
        pool.ensure_compatible_live(lines, rng);
    }
    match access {
        AccessModel::ProduceMask => {
            let line = if options.ara_safe_enabled || rng.random_range(0..3) == 0 {
                "vlm.v v0, (x10)"
            } else {
                "vl1re64.v v0, (x10)"
            };
            lines.push(line.to_string());
            push_attribution_fence(lines);
            pool.mask_live = true;
            1
        }
        AccessModel::LoadVec => {
            let width = ensure_memory_eew(lines, pool, rng, options, 32);
            let vd = pool.produce_vec_for_width(rng, width, options.register_constraints_enabled());
            let masked = pool.mask_live && rng.random_range(0..2) == 0;
            let line = if masked {
                format!("vle32.v v{vd}, (x11), v0.t")
            } else {
                format!("vle32.v v{vd}, (x11)")
            };
            push_rvv_line(lines, rng, options, line);
            1
        }
        AccessModel::AluVx => {
            let vs = pool.consume_vec(rng);
            let vd = pool.produce_vec(rng, options.register_constraints_enabled());
            let mnemonic = ["vadd.vx", "vsub.vx", "vxor.vx"][rng.random_range(0..3)];
            push_rvv_line(lines, rng, options, format!("{mnemonic} v{vd}, v{vs}, x13"));
            1
        }
        AccessModel::CompareToMask => {
            let vs = pool.consume_vec(rng);
            let mnemonic = ["vmseq.vi", "vmsne.vi", "vmsle.vi"][rng.random_range(0..3)];
            push_rvv_line(lines, rng, options, format!("{mnemonic} v0, v{vs}, 0"));
            pool.mask_live = true;
            1
        }
        AccessModel::MergeWithMask => {
            let (left, right) = pool.consume_two_vecs(rng);
            let vd = pool.produce_vec(rng, options.register_constraints_enabled());
            push_rvv_line(
                lines,
                rng,
                options,
                format!("vmerge.vvm v{vd}, v{left}, v{right}, v0"),
            );
            1
        }
        AccessModel::MaskedMemoryRoundTrip => {
            let width = ensure_memory_eew(lines, pool, rng, options, 32);
            let compatible = pool
                .live_vecs
                .iter()
                .any(|state| crate::rvv_legal::valid_group(state.base, width));
            let vs = if compatible {
                pool.consume_vec_for_width(rng, width)
            } else {
                let base =
                    pool.produce_vec_for_width(rng, width, options.register_constraints_enabled());
                lines.push(format!("vle32.v v{base}, (x11)"));
                base
            };
            push_rvv_line(lines, rng, options, format!("vse32.v v{vs}, (x12), v0.t"));
            let vd = pool.produce_vec_for_width(rng, width, options.register_constraints_enabled());
            push_rvv_line(lines, rng, options, format!("vle32.v v{vd}, (x12), v0.t"));
            2
        }
        AccessModel::MaskMemoryRoundTrip => {
            let mut edges = 2;
            if rng.random_range(0..2) == 0 {
                lines.push("vlm.v v0, (x10)".to_string());
                push_attribution_fence(lines);
                edges += 1;
            }
            lines.push("vsm.v v0, (x10)".to_string());
            push_attribution_fence(lines);
            lines.push("vlm.v v0, (x10)".to_string());
            push_attribution_fence(lines);
            pool.mask_live = true;
            edges
        }
        AccessModel::MaskMergeState => {
            let (left, right) = pool.consume_two_vecs(rng);
            let merged = pool.produce_vec(rng, options.register_constraints_enabled());
            lines.push(format!("vmerge.vvm v{merged}, v{left}, v{right}, v0"));
            push_attribution_fence(lines);
            // The merge selector is v0, so its data result cannot also be v0.
            // Compare the legal data result back into v0 before the mask store.
            lines.push(format!("vmsne.vi v0, v{merged}, 0"));
            push_attribution_fence(lines);
            lines.push("vsm.v v0, (x10)".to_string());
            push_attribution_fence(lines);
            pool.mask_live = true;
            3
        }
        AccessModel::MaskMergeV0OverlapProbe => {
            let config = RvvLiveConfig {
                cfg: TemplateRvvConfig {
                    sew: 8,
                    vl: 10,
                    lmul: "m4",
                    tail_policy: "tu",
                    mask_policy: "mu",
                },
            };
            push_local_config_setup(lines, pool, options, config);
            lines.push("li x14, -1".to_string());
            lines.push("vmv.v.x v0, x14".to_string());
            let left = pool.produce_vec(rng, options.register_constraints_enabled());
            let right = pool.produce_vec(rng, options.register_constraints_enabled());
            lines.push("li x14, 51".to_string());
            lines.push(format!("vmv.v.x v{left}, x14"));
            lines.push("li x14, 68".to_string());
            lines.push(format!("vmv.v.x v{right}, x14"));
            let destination = if options.positive_memory {
                pool.produce_vec(rng, options.register_constraints_enabled())
            } else {
                0
            };
            lines.push(format!("vmerge.vvm v{destination}, v{left}, v{right}, v0"));
            push_attribution_fence(lines);
            if options.positive_memory {
                lines.push(format!("vmsne.vi v0, v{destination}, 0"));
                push_attribution_fence(lines);
            }
            lines.push("vsm.v v0, (x10)".to_string());
            push_attribution_fence(lines);
            pool.mask_live = true;
            3 + usize::from(options.positive_memory)
        }
        AccessModel::UnitFofLoad => {
            let sew = choose_from_slice(rng, &[8u16, 16, 32, 64]);
            ensure_m1_sew(lines, pool, rng, options, sew);
            let vd = pool.produce_vec(rng, options.register_constraints_enabled());
            let base_reg = choose_vector_data_base(rng);
            let masked = pool.mask_live && rng.random_range(0..2) == 0;
            let line = if masked {
                format!("vle{sew}ff.v v{vd}, ({base_reg}), v0.t")
            } else {
                format!("vle{sew}ff.v v{vd}, ({base_reg})")
            };
            lines.push(line);
            push_attribution_fence(lines);
            if !masked && focused_access_from_env() == Some(AccessModel::UnitFofLoad) {
                // The first unmasked active element is observable on Ara as a
                // scalar commit, and on XiangShan as both vector and GPR state.
                lines.push("csrr x17, vl".to_string());
                lines.push("sd x17, 208(x12)".to_string());
                lines.push(format!("vmv.x.s x18, v{vd}"));
                lines.push("sd x18, 216(x12)".to_string());
                3
            } else {
                1
            }
        }
        AccessModel::SegmentFofLoad => {
            let sew = choose_from_slice(rng, &[8u16, 16, 32]);
            ensure_m1_sew(lines, pool, rng, options, sew);
            let field_count = choose_from_slice(rng, &[2u8, 3, 4, 8]);
            if options.positive_memory {
                let mut config = pool.config;
                config.cfg.vl = choose_from_slice(rng, &[1, 2, 4]);
                push_local_config_setup(lines, pool, options, config);
            }
            let base = pool.allocate_segment_base(
                rng,
                field_count,
                options.register_constraints_enabled(),
            );
            let base_reg = choose_vector_data_base(rng);
            let masked = pool.mask_live && rng.random_range(0..2) == 0;
            let line = if masked {
                format!("vlseg{field_count}e{sew}ff.v v{base}, ({base_reg}), v0.t")
            } else {
                format!("vlseg{field_count}e{sew}ff.v v{base}, ({base_reg})")
            };
            lines.push(line);
            push_attribution_fence(lines);
            pool.push_vec_group(base, field_count);
            if !masked && focused_access_from_env() == Some(AccessModel::SegmentFofLoad) {
                lines.push("csrr x17, vl".to_string());
                lines.push("sd x17, 152(x12)".to_string());
                for field in 0..field_count {
                    lines.push(format!("vmv.x.s x18, v{}", base + field));
                    lines.push(format!("sd x18, {}(x12)", 160 + i32::from(field) * 8));
                }
                usize::from(field_count) * 2 + 1
            } else {
                usize::from(field_count)
            }
        }
        AccessModel::IndexedLoad => {
            ensure_indexed_load_config(lines, pool, rng, options);
            if options.positive_memory {
                for lane in 0..8u8 {
                    let offset = choose_from_slice(rng, &[0u8, 4, 8, 12, 16, 20, 24, 28]);
                    lines.push(format!("li x14, {offset}"));
                    lines.push(format!("sw x14, {}(x11)", u32::from(lane) * 4));
                }
            }
            let index = pool.allocate_group(rng, options.register_constraints_enabled());
            lines.push(format!("vle32.v v{index}, (x11)"));
            push_attribution_fence(lines);
            let vd = pool.produce_vec(rng, options.register_constraints_enabled());
            let masked = pool.mask_live && rng.random_range(0..2) == 0;
            let line = if masked {
                format!("vluxei32.v v{vd}, (x11), v{index}, v0.t")
            } else {
                format!("vluxei32.v v{vd}, (x11), v{index}")
            };
            lines.push(line);
            push_attribution_fence(lines);
            2
        }
        AccessModel::IndexedSegmentLoad => {
            ensure_indexed_load_config(lines, pool, rng, options);
            if options.positive_memory {
                for lane in 0..8u8 {
                    let offset = choose_from_slice(rng, &[0u8, 4, 8, 12, 16, 20, 24, 28]);
                    lines.push(format!("li x14, {offset}"));
                    lines.push(format!("sw x14, {}(x11)", u32::from(lane) * 4));
                }
            }
            let field_count = choose_from_slice(rng, &[2u8, 3, 5]);
            let (base, index) = if options.register_constraints_enabled() {
                pool.allocate_indexed_segment_pair(rng, field_count, 32, None)
            } else {
                let index = pool.allocate_group(rng, false);
                let base = pool.allocate_segment_base(rng, field_count, false);
                (base, index)
            };
            lines.push(format!("vle32.v v{index}, (x11)"));
            push_attribution_fence(lines);
            let masked = pool.mask_live && rng.random_range(0..2) == 0;
            let line = if masked {
                format!("vluxseg{field_count}ei32.v v{base}, (x11), v{index}, v0.t")
            } else {
                format!("vluxseg{field_count}ei32.v v{base}, (x11), v{index}")
            };
            lines.push(line);
            push_attribution_fence(lines);
            pool.push_vec_group(base, field_count);
            usize::from(field_count) + 1
        }
        AccessModel::StridedSegmentLoad => {
            let sew = choose_from_slice(rng, &[8u16, 16, 32, 64]);
            ensure_m1_sew(lines, pool, rng, options, sew);
            let field_count = choose_from_slice(rng, &[2u8, 3, 4, 5]);
            let base = pool.allocate_segment_base(
                rng,
                field_count,
                options.register_constraints_enabled(),
            );
            let base_reg = choose_vector_data_base(rng);
            let stride_reg = choose_vector_stride(rng);
            if options.positive_memory {
                let bytes = sew / 8;
                let stride = choose_from_slice(rng, &[0, bytes, 2 * bytes]);
                lines.push(format!("li {stride_reg}, {stride}"));
            }
            let masked = pool.mask_live && rng.random_range(0..2) == 0;
            let line = if masked {
                format!("vlsseg{field_count}e{sew}.v v{base}, ({base_reg}), {stride_reg}, v0.t")
            } else {
                format!("vlsseg{field_count}e{sew}.v v{base}, ({base_reg}), {stride_reg}")
            };
            lines.push(line);
            push_attribution_fence(lines);
            pool.push_vec_group(base, field_count);
            usize::from(field_count)
        }
        AccessModel::CarryWithMask => {
            let vs = pool.consume_vec(rng);
            let vd = pool.produce_vec(rng, options.register_constraints_enabled());
            let mnemonic = ["vmadc.vxm", "vmsbc.vxm", "vadc.vxm"][rng.random_range(0..3)];
            push_rvv_line(
                lines,
                rng,
                options,
                format!("{mnemonic} v{vd}, v{vs}, x13, v0"),
            );
            1
        }
        AccessModel::MaskLogical => {
            let mnemonic = ["vmor.mm", "vmxor.mm", "vmorn.mm", "vmnand.mm"][rng.random_range(0..4)];
            push_rvv_line(lines, rng, options, format!("{mnemonic} v0, v0, v0"));
            pool.mask_live = true;
            1
        }
        AccessModel::SegmentLoad => {
            let sew = choose_from_slice(rng, &[8u16, 16, 32, 64]);
            ensure_m1_sew(lines, pool, rng, options, sew);
            let field_count = choose_from_slice(rng, &[2u8, 3, 4]);
            if options.positive_memory {
                let mut config = pool.config;
                config.cfg.vl = choose_from_slice(rng, &[1, 2, 4]);
                push_local_config_setup(lines, pool, options, config);
            }
            let base = pool.allocate_segment_base(
                rng,
                field_count,
                options.register_constraints_enabled(),
            );
            let base_reg = choose_vector_data_base(rng);
            lines.push(format!(
                "vlseg{field_count}e{sew}.v v{base}, ({base_reg}), v0.t"
            ));
            push_attribution_fence(lines);
            pool.push_vec_group(base, field_count);
            usize::from(field_count)
        }
        AccessModel::TailHighTransition => {
            if options.constraints_enabled {
                let config = RvvLiveConfig {
                    cfg: TemplateRvvConfig {
                        sew: 64,
                        vl: 4,
                        lmul: "m1",
                        tail_policy: if options.ara_generic_enabled {
                            "tu"
                        } else {
                            "ta"
                        },
                        mask_policy: "mu",
                    },
                };
                push_local_config_setup(lines, pool, options, config);
            }
            let vd = pool.produce_vec(rng, options.register_constraints_enabled());
            push_rvv_line(lines, rng, options, format!("vmv.v.x v{vd}, x13"));
            if options.constraints_enabled {
                let config = RvvLiveConfig {
                    cfg: TemplateRvvConfig {
                        sew: 32,
                        vl: 8,
                        lmul: "m1",
                        tail_policy: "tu",
                        mask_policy: "mu",
                    },
                };
                push_local_config_setup(lines, pool, options, config);
            }
            push_rvv_line(lines, rng, options, format!("vadd.vi v{vd}, v{vd}, 1"));
            push_rvv_line(lines, rng, options, format!("vsrl.vi v{vd}, v{vd}, 1"));
            3
        }
        AccessModel::VstartMaskedStore => {
            ensure_m1_sew(lines, pool, rng, options, 32);
            let vs = pool.consume_latest_vec();
            let start = choose_from_slice(rng, &[1u8, 2, 3]);
            lines.push(format!("li x5, {start}"));
            lines.push("csrrw x0, vstart, x5".to_string());
            lines.push(format!("vse32.v v{vs}, (x12), v0.t"));
            push_attribution_fence(lines);
            lines.push("csrrw x6, vstart, x0".to_string());
            for (idx, offset) in [0, 4, 8, 12].into_iter().enumerate() {
                lines.push(format!("lw x15, {offset}(x12)"));
                lines.push(format!("sd x15, {}(x12)", 96 + idx as i32 * 8));
            }
            lines.push("sd x6, 128(x12)".to_string());
            4
        }
        AccessModel::VstartMaskedStoreProbe => {
            let config = RvvLiveConfig {
                cfg: TemplateRvvConfig {
                    sew: 32,
                    vl: choose_from_slice(rng, &[4u8, 4, 5, 8]),
                    lmul: "m1",
                    tail_policy: "tu",
                    mask_policy: "mu",
                },
            };
            push_local_config_setup(lines, pool, options, config);

            let src_salt = rng.random_range(0x10u16..0xf0);
            let dst_salt = src_salt ^ rng.random_range(1u16..0xff);
            let mut src_words = [0u64; 4];
            let mut dst_words = [0u64; 4];
            for lane in 0..4u8 {
                let idx = lane as usize;
                src_words[idx] = biased_word32(rng, lane, src_salt);
                dst_words[idx] = biased_word32(rng, lane, dst_salt);
                if dst_words[idx] == src_words[idx] {
                    dst_words[idx] ^= 0x00ff_0000;
                }
            }
            push_data_words32(lines, "x11", &src_words);
            push_data_words32(lines, "x12", &dst_words);

            let src = pool.produce_vec(rng, options.register_constraints_enabled());
            lines.push(format!("vle32.v v{src}, (x11)"));
            let (start, use_vector_mask, mask_word) = choose_from_slice(
                rng,
                &[
                    (1u8, true, 0u64),
                    (1, true, 0),
                    (1, false, 0xau64),
                    (2, false, 0xc),
                    (1, false, 0xe),
                    (3, false, 0x8),
                ],
            );
            if use_vector_mask {
                let mask_idx = pool.produce_vec(rng, options.register_constraints_enabled());
                let mask_tmp = pool.produce_vec(rng, options.register_constraints_enabled());
                lines.push(format!("vid.v v{mask_idx}"));
                lines.push(format!("vand.vi v{mask_tmp}, v{mask_idx}, 1"));
                lines.push(format!("vmsne.vi v0, v{mask_tmp}, 0"));
            } else {
                lines.push(format!("li x14, {mask_word}"));
                lines.push("sd x14, 0(x10)".to_string());
                lines.push("vlm.v v0, (x10)".to_string());
            }
            push_attribution_fence(lines);
            pool.mask_live = true;
            lines.push(format!("li x5, {start}"));
            lines.push("csrrw x0, vstart, x5".to_string());
            lines.push(format!("vse32.v v{src}, (x12), v0.t"));
            push_attribution_fence(lines);
            lines.push("csrrw x6, vstart, x0".to_string());
            for (idx, offset) in [0, 4, 8, 12].into_iter().enumerate() {
                lines.push(format!("lw x15, {offset}(x12)"));
                lines.push(format!("sd x15, {}(x12)", 96 + idx as i32 * 8));
            }
            lines.push("sd x6, 128(x12)".to_string());
            6
        }
        AccessModel::SlideWithMask => {
            let vs = pool.consume_latest_vec();
            let vd = pool.produce_vec_disjoint_from(rng, vs);
            let vl = pool.config.cfg.vl.max(1);
            let offsets = [
                0u16,
                1,
                u16::from(vl.saturating_sub(1)),
                u16::from(vl),
                u16::from(vl) + 1,
                31,
                64,
            ];
            let offset = choose_from_slice(rng, &offsets);
            lines.push(format!("li x6, {offset}"));
            let line = if pool.mask_live && rng.random_range(0..2) == 0 {
                format!("vslideup.vx v{vd}, v{vs}, x6, v0.t")
            } else {
                format!("vslideup.vx v{vd}, v{vs}, x6")
            };
            push_rvv_line(lines, rng, options, line);
            lines.push(format!("vmv.x.s x15, v{vd}"));
            lines.push("sd x15, 136(x12)".to_string());
            2
        }
        AccessModel::LargeMaskedSlideWindowObserve => {
            let config = RvvLiveConfig {
                cfg: TemplateRvvConfig {
                    sew: 8,
                    vl: 0,
                    lmul: "m1",
                    tail_policy: "tu",
                    mask_policy: "mu",
                },
            };
            pool.set_config(config);
            let slide_offset = choose_from_slice(rng, &[128u16, 160, 192, 224, 256, 256, 256]);
            let active_span = if slide_offset >= 256 {
                choose_from_slice(rng, &[4u16, 8, 16])
            } else {
                choose_from_slice(rng, &[4u16, 8, 16, 24, 32])
            };
            let vl = slide_offset.saturating_add(active_span);
            let active_byte = i32::from(slide_offset / 8);
            for offset in [0, 8, 16, 24, 32, 40, 48, 56] {
                lines.push("li x14, 0".to_string());
                lines.push(format!("sd x14, {offset}(x10)"));
            }
            let active_mask = choose_from_slice(rng, &[-1i64, -1, -1, 0x0f, 0xf0, 0x55]);
            lines.push(format!("li x14, {active_mask}"));
            // This is one mask byte; an eight-byte store at an arbitrary byte
            // index can fault before the vector slide under test is reached.
            lines.push(format!("sb x14, {active_byte}(x10)"));
            lines.push(format!("li x5, {vl}"));
            push_dynamic_config_setup(lines, pool, options, config, "x0", "x5");

            let dst = pool.produce_vec(rng, options.register_constraints_enabled());
            let src = pool.produce_vec_disjoint_from(rng, dst);
            let obs = pool.produce_vec(rng, options.register_constraints_enabled());
            let fill = choose_from_slice(rng, &[0xaau16, 0x55, 0x7f, 0xcc]);
            lines.push(format!("li x6, {fill}"));
            lines.push(format!("vmv.v.x v{dst}, x6"));
            lines.push(format!("vid.v v{src}"));
            lines.push("vlm.v v0, (x10)".to_string());
            push_attribution_fence(lines);
            pool.mask_live = true;
            lines.push(format!("li x6, {slide_offset}"));
            lines.push(format!("vslideup.vx v{dst}, v{src}, x6, v0.t"));
            push_attribution_fence(lines);
            let high_lane = vl.saturating_sub(1).max(slide_offset);
            for (lane, offset) in [
                (slide_offset.saturating_sub(1), 0),
                (slide_offset, 8),
                (slide_offset.saturating_add(1).min(high_lane), 16),
                (high_lane, 24),
            ] {
                lines.push(format!("li x6, {lane}"));
                lines.push(format!("vslidedown.vx v{obs}, v{dst}, x6"));
                lines.push(format!("vmv.x.s x15, v{obs}"));
                lines.push("andi x15, x15, 255".to_string());
                lines.push(format!("sd x15, {offset}(x12)"));
            }
            6
        }
        AccessModel::WidenZeroExtendSlideObserve => {
            let narrow_config = RvvLiveConfig {
                cfg: TemplateRvvConfig {
                    sew: 16,
                    vl: choose_from_slice(rng, &[1u8, 2, 4]),
                    lmul: "mf4",
                    tail_policy: "tu",
                    mask_policy: "mu",
                },
            };
            push_local_config_setup(lines, pool, options, narrow_config);
            let src = pool.produce_vec(rng, options.register_constraints_enabled());
            let high = choose_from_slice(rng, &[1u64, 2, 0x12, 0x7f, 0x80, 0xa0, 0xff]);
            let low = choose_from_slice(rng, &[0u64, 1, 0x4f, 0x7f, 0x80, 0xff]);
            let scalar = (high << 8) | low;
            lines.push(format!("li x9, {scalar}"));
            lines.push(format!("vmv.s.x v{src}, x9"));

            let wide_config = RvvLiveConfig {
                cfg: TemplateRvvConfig {
                    sew: 64,
                    vl: choose_from_slice(rng, &[2u8, 3, 8, 16, 27, 31]),
                    lmul: "m1",
                    tail_policy: "tu",
                    mask_policy: "mu",
                },
            };
            push_local_config_setup(lines, pool, options, wide_config);
            let ext = pool.produce_vec(rng, options.register_constraints_enabled());
            lines.push(format!("vzext.vf8 v{ext}, v{src}"));
            lines.push(format!("vmv.x.s x15, v{ext}"));
            lines.push("sd x15, 144(x12)".to_string());
            let shifted = pool.produce_vec(rng, options.register_constraints_enabled());
            let amount = choose_from_slice(rng, &[1u8, 2, 5]);
            lines.push(format!("vslidedown.vi v{shifted}, v{ext}, {amount}"));
            lines.push(format!("vmv.x.s x15, v{shifted}"));
            lines.push("sd x15, 152(x12)".to_string());
            4
        }
        AccessModel::IndexedMisalignmentSensitivity => emit_indexed_misalignment_probe(
            lines,
            pool,
            rng,
            options,
            IndexedMisalignmentMode::Mixed,
        ),
        AccessModel::MaskedIndexedMisalignmentProbe => emit_indexed_misalignment_probe(
            lines,
            pool,
            rng,
            options,
            IndexedMisalignmentMode::MaskedOff,
        ),
        AccessModel::UnmaskedIndexedMisalignmentProbe => emit_indexed_misalignment_probe(
            lines,
            pool,
            rng,
            options,
            IndexedMisalignmentMode::UnmaskedTrap,
        ),
        AccessModel::SegmentLaterFieldObserve => {
            if rng.random_range(0..2) == 0 {
                let config = RvvLiveConfig {
                    cfg: TemplateRvvConfig {
                        sew: 32,
                        vl: 1,
                        lmul: "m1",
                        tail_policy: "tu",
                        mask_policy: "mu",
                    },
                };
                push_local_config_setup(lines, pool, options, config);
                let index_offset = choose_from_slice(rng, &[0u32, 0, 0, 4, 8]);
                lines.push(format!("li x14, {index_offset}"));
                lines.push("sw x14, 0(x11)".to_string());
                let field_base = rng.random_range(0x1000u64..0x7000) & !0xff;
                let field_step = choose_from_slice(rng, &[0x80u64, 0x100, 0x180, 0x200]);
                for field in 0..5u8 {
                    let offset = index_offset as i32 + i32::from(field) * 4;
                    let value = field_base + field_step * u64::from(field);
                    lines.push(format!("li x14, {value}"));
                    lines.push(format!("sw x14, {offset}(x12)"));
                }
                let preferred_base = choose_from_slice(rng, &[18u8, 18, 16, 20, 24]);
                let (base, index) = if options.register_constraints_enabled() {
                    pool.allocate_indexed_segment_pair(rng, 5, 32, Some(preferred_base))
                } else {
                    let index = pool.produce_vec(rng, false);
                    let base = pool.allocate_preferred_segment_base(rng, 5, false, preferred_base);
                    (base, index)
                };
                lines.push(format!("vle32.v v{index}, (x11)"));
                push_attribution_fence(lines);
                lines.push(format!("vluxseg5ei32.v v{base}, (x12), v{index}"));
                push_attribution_fence(lines);
                pool.push_vec_group(base, 5);
                for field in 0..5u8 {
                    let reg = base + field;
                    lines.push(format!("vmv.x.s x15, v{reg}"));
                    lines.push(format!("sd x15, {}(x12)", 96 + i32::from(field) * 8));
                }
                6
            } else {
                let config = RvvLiveConfig {
                    cfg: TemplateRvvConfig {
                        sew: 64,
                        vl: 1,
                        lmul: "m1",
                        tail_policy: "tu",
                        mask_policy: "mu",
                    },
                };
                push_local_config_setup(lines, pool, options, config);
                let e64_tag = rng.random_range(0x10u64..0x70);
                for field in 0..3u8 {
                    let value =
                        ((e64_tag + u64::from(field)) << 56) | rng.random_range(0x1000u64..0x8000);
                    lines.push(format!("li x14, {}", signed_literal(value)));
                    lines.push(format!("sd x14, {}(x12)", i32::from(field) * 8));
                }
                let stride = choose_from_slice(rng, &[16u8, 24, 32, 32, 40, 64]);
                lines.push(format!("li x6, {stride}"));
                let preferred_base = choose_from_slice(rng, &[18u8, 18, 13, 20, 24]);
                let base = pool.allocate_preferred_segment_base(
                    rng,
                    3,
                    options.register_constraints_enabled(),
                    preferred_base,
                );
                lines.push(format!("vlsseg3e64.v v{base}, (x12), x6"));
                push_attribution_fence(lines);
                pool.push_vec_group(base, 3);
                for field in 0..3u8 {
                    let reg = base + field;
                    lines.push(format!("vmv.x.s x15, v{reg}"));
                    lines.push(format!("sd x15, {}(x12)", 96 + i32::from(field) * 8));
                }
                4
            }
        }
        AccessModel::SegmentLaterFieldStrictObserve => {
            if rng.random_range(0..4) != 0 {
                let config = RvvLiveConfig {
                    cfg: TemplateRvvConfig {
                        sew: 32,
                        vl: 1,
                        lmul: "m1",
                        tail_policy: "tu",
                        mask_policy: "mu",
                    },
                };
                push_local_config_setup(lines, pool, options, config);
                let index_offset = choose_from_slice(rng, &[0u32, 0, 0, 4, 8]);
                lines.push(format!("li x14, {index_offset}"));
                lines.push("sw x14, 0(x11)".to_string());
                let field_base = rng.random_range(0x1000u64..0x7000) & !0xff;
                let field_step = choose_from_slice(rng, &[0x80u64, 0x100, 0x180, 0x200]);
                for field in 0..5u8 {
                    let offset = index_offset as i32 + i32::from(field) * 4;
                    let value = field_base + field_step * u64::from(field);
                    lines.push(format!("li x14, {value}"));
                    lines.push(format!("sw x14, {offset}(x12)"));
                }
                let preferred_base = choose_from_slice(rng, &[18u8, 18, 16, 20, 24]);
                let (base, index) = if options.register_constraints_enabled() {
                    pool.allocate_indexed_segment_pair(rng, 5, 32, Some(preferred_base))
                } else {
                    let index = pool.produce_vec(rng, false);
                    let base = pool.allocate_preferred_segment_base(rng, 5, false, preferred_base);
                    (base, index)
                };
                lines.push(format!("vle32.v v{index}, (x11)"));
                push_attribution_fence(lines);
                lines.push(format!("vluxseg5ei32.v v{base}, (x12), v{index}"));
                push_attribution_fence(lines);
                pool.push_vec_group(base, 5);
                for field in [1u8, 2, 3, 4, 0] {
                    let reg = base + field;
                    lines.push(format!("vmv.x.s x15, v{reg}"));
                    lines.push(format!("sd x15, {}(x12)", 96 + i32::from(field) * 8));
                }
                6
            } else {
                let config = RvvLiveConfig {
                    cfg: TemplateRvvConfig {
                        sew: 64,
                        vl: 1,
                        lmul: "m1",
                        tail_policy: "tu",
                        mask_policy: "mu",
                    },
                };
                push_local_config_setup(lines, pool, options, config);
                let e64_tag = rng.random_range(0x10u64..0x70);
                for field in 0..3u8 {
                    let value =
                        ((e64_tag + u64::from(field)) << 56) | rng.random_range(0x1000u64..0x8000);
                    lines.push(format!("li x14, {}", signed_literal(value)));
                    lines.push(format!("sd x14, {}(x12)", i32::from(field) * 8));
                }
                let stride = choose_from_slice(rng, &[16u8, 24, 32, 32, 40, 64]);
                lines.push(format!("li x6, {stride}"));
                let preferred_base = choose_from_slice(rng, &[18u8, 18, 13, 20, 24]);
                let base = pool.allocate_preferred_segment_base(
                    rng,
                    3,
                    options.register_constraints_enabled(),
                    preferred_base,
                );
                lines.push(format!("vlsseg3e64.v v{base}, (x12), x6"));
                push_attribution_fence(lines);
                pool.push_vec_group(base, 3);
                for field in [1u8, 2, 0] {
                    let reg = base + field;
                    lines.push(format!("vmv.x.s x15, v{reg}"));
                    lines.push(format!("sd x15, {}(x12)", 96 + i32::from(field) * 8));
                }
                4
            }
        }
        AccessModel::SparseMaskedSegmentLaneObserve => {
            let config = RvvLiveConfig {
                cfg: TemplateRvvConfig {
                    sew: 16,
                    vl: 6,
                    lmul: "m1",
                    tail_policy: "tu",
                    mask_policy: "mu",
                },
            };
            push_local_config_setup(lines, pool, options, config);
            let active_lane = choose_from_slice(rng, &[5u8, 5, 5, 1, 2, 3]);
            for lane in 0..6u8 {
                let first = 4096 + u64::from(lane);
                let second = 8192 + u64::from(lane);
                let base_offset = i32::from(lane) * 4;
                lines.push(format!("li x14, {first}"));
                lines.push(format!("sh x14, {base_offset}(x11)"));
                lines.push(format!("li x14, {second}"));
                lines.push(format!("sh x14, {}(x11)", base_offset + 2));
            }
            let base = pool.allocate_segment_base(rng, 2, options.register_constraints_enabled());
            lines.push("li x14, 4369".to_string());
            lines.push(format!("vmv.v.x v{base}, x14"));
            lines.push("li x14, 8738".to_string());
            lines.push(format!("vmv.v.x v{}, x14", base + 1));
            let mask = 1u64 << active_lane;
            lines.push(format!("li x14, {mask}"));
            lines.push("sd x14, 0(x10)".to_string());
            lines.push("vlm.v v0, (x10)".to_string());
            push_attribution_fence(lines);
            pool.mask_live = true;
            lines.push(format!("vlseg2e16.v v{base}, (x11), v0.t"));
            push_attribution_fence(lines);
            pool.push_vec_group(base, 2);
            let obs0 = pool.produce_vec(rng, options.register_constraints_enabled());
            lines.push(format!("vslidedown.vi v{obs0}, v{base}, {active_lane}"));
            lines.push(format!("vmv.x.s x15, v{obs0}"));
            lines.push("sd x15, 176(x12)".to_string());
            let obs1 = pool.produce_vec(rng, options.register_constraints_enabled());
            lines.push(format!(
                "vslidedown.vi v{obs1}, v{}, {active_lane}",
                base + 1
            ));
            lines.push(format!("vmv.x.s x15, v{obs1}"));
            lines.push("sd x15, 184(x12)".to_string());
            4
        }
        AccessModel::ConfigReadback => {
            if !options.constraints_enabled {
                return 0;
            }
            let config = pool.config;
            pool.set_config(config);
            let vtype = config.cfg.vtype_immediate();
            if options.coverage_focused {
                // Explicit control-state producer/consumer: vsetvl writes vl
                // and vtype, then a vector write and scalar extraction consume
                // the new configuration before it is restored. The rd result
                // observes vl without relying on read-only CSR pseudo parsing.
                let destination = pool.produce_vec(rng, options.register_constraints_enabled());
                lines.push("li x13, 7".to_string());
                lines.push(format!("li x14, {vtype}"));
                lines.push("vsetvl x15, x13, x14".to_string());
                lines.push("sd x15, 200(x12)".to_string());
                lines.push(format!("vmv.v.i v{destination}, 3"));
                push_attribution_fence(lines);
                lines.push(format!("vmv.x.s x16, v{destination}"));
                lines.push("sd x16, 208(x12)".to_string());
                lines.push("li x14, 8".to_string());
                lines.push(config.vsetivli_line());
            } else {
                lines.push(format!("vsetvli x15, x13, {vtype}"));
                lines.push("csrr x17, vl".to_string());
                lines.push("sd x17, 200(x12)".to_string());
                lines.push(format!("vsetvli x16, x14, {vtype}"));
                lines.push("csrr x17, vl".to_string());
                lines.push("sd x17, 208(x12)".to_string());
                lines.push("xor x18, x15, x16".to_string());
            }
            2
        }
        AccessModel::FractionalGather => {
            let previous = pool.config;
            let fractional = RvvLiveConfig {
                cfg: TemplateRvvConfig {
                    sew: 8,
                    vl: 24,
                    lmul: "mf2",
                    tail_policy: "tu",
                    mask_policy: "mu",
                },
            };
            push_local_config_setup(lines, pool, options, fractional);
            // Even-numbered operands keep register selection simple while
            // the memory store observes the gather result without relying on
            // the scalar extraction that stalled in the bounded Ara probe.
            let mut even_bases = fractional
                .legal_group_bases(pool.vector_range)
                .into_iter()
                .filter(|base| base % 2 == 0)
                .collect::<Vec<_>>();
            let (source, destination) = if even_bases.len() >= 2 {
                let source = even_bases.remove(rng.random_range(0..even_bases.len()));
                let destination = even_bases.remove(rng.random_range(0..even_bases.len()));
                for base in [source, destination] {
                    pool.record_group(base, 1);
                    pool.push_vec(base, 1);
                }
                (source, destination)
            } else {
                (
                    pool.produce_vec(rng, options.register_constraints_enabled()),
                    pool.produce_vec(rng, options.register_constraints_enabled()),
                )
            };
            lines.push(format!("vmv.v.i v{source}, 3"));
            push_attribution_fence(lines);
            lines.push(format!("vrgather.vi v{destination}, v{source}, 3"));
            push_attribution_fence(lines);
            lines.push(format!("vse8.v v{destination}, (x12)"));
            push_attribution_fence(lines);
            push_local_config_setup(lines, pool, options, previous);
            2
        }
        AccessModel::FpScalarObserve => {
            if options.constraints_enabled && !matches!(pool.config.cfg.sew, 32 | 64) {
                let previous = pool.config.cfg;
                let config = RvvLiveConfig {
                    cfg: TemplateRvvConfig {
                        sew: choose_from_slice(rng, &[32, 64]),
                        vl: previous.vl.max(1),
                        lmul: choose_from_slice(rng, &["m1", "m2"]),
                        tail_policy: previous.tail_policy,
                        mask_policy: previous.mask_policy,
                    },
                };
                push_local_config_setup(lines, pool, options, config);
                pool.ensure_compatible_live(lines, rng);
            }
            let vs = pool.consume_latest_vec();
            let fd = choose_from_slice(rng, &["f0", "f16", "f23"]);
            push_rvv_line(lines, rng, options, format!("vfmv.f.s {fd}, v{vs}"));
            lines.push(format!("fmv.x.d x15, {fd}"));
            lines.push("sd x15, 72(x11)".to_string());
            1
        }
        AccessModel::WidenZeroExtend => {
            if options.constraints_enabled && pool.config.cfg.sew != 64 {
                let config = RvvLiveConfig::constrained_e64_m1();
                push_local_config_setup(lines, pool, options, config);
            }
            let vs = pool.produce_vec(rng, options.register_constraints_enabled());
            let load = if pool.mask_live {
                format!("vle8.v v{vs}, (x11), v0.t")
            } else {
                format!("vle8.v v{vs}, (x11)")
            };
            push_rvv_line(lines, rng, options, load);
            let vd = pool.produce_vec(rng, options.register_constraints_enabled());
            push_rvv_line(lines, rng, options, format!("vzext.vf8 v{vd}, v{vs}"));
            2
        }
        AccessModel::WholeRegister4RoundTrip => {
            if options.constraints_enabled && pool.config.group_width() != 4 {
                let config = RvvLiveConfig::constrained_e32_m4();
                push_local_config_setup(lines, pool, options, config);
            }
            let base =
                pool.allocate_whole_register_base(rng, 4, options.register_constraints_enabled());
            let base_reg = choose_vector_data_base(rng);
            let stride_reg = choose_vector_stride(rng);
            if options.positive_memory {
                let stride = choose_from_slice(rng, &[0u8, 4, 8, 12]);
                lines.push(format!("li {stride_reg}, {stride}"));
            }
            lines.push(format!("vlse32.v v{base}, ({base_reg}), {stride_reg}"));
            push_attribution_fence(lines);
            lines.push(format!("vs4r.v v{base}, 0(x12)"));
            push_attribution_fence(lines);
            pool.push_vec_group(base, 4);
            4
        }
        AccessModel::WholeRegister8RoundTrip => {
            let base =
                pool.allocate_whole_register_base(rng, 8, options.register_constraints_enabled());
            let base_reg = choose_vector_data_base(rng);
            lines.push(format!("vl8re64.v v{base}, ({base_reg})"));
            push_attribution_fence(lines);
            lines.push(format!("vs8r.v v{base}, 0(x12)"));
            push_attribution_fence(lines);
            pool.push_vec_group(base, 8);
            8
        }
        AccessModel::WholeRegister8Load => {
            let base =
                pool.allocate_whole_register_base(rng, 8, options.register_constraints_enabled());
            let base_reg = choose_vector_data_base(rng);
            lines.push(format!("vl8re64.v v{base}, ({base_reg})"));
            push_attribution_fence(lines);
            pool.push_vec_group(base, 8);
            8
        }
        AccessModel::StridedSegmentE64 => {
            ensure_m1_sew(lines, pool, rng, options, 64);
            let field_count = 3u8;
            let base = pool.allocate_segment_base(
                rng,
                field_count,
                options.register_constraints_enabled(),
            );
            let base_reg = choose_vector_data_base(rng);
            let stride_reg = choose_vector_stride(rng);
            if options.positive_memory {
                let stride = choose_from_slice(rng, &[0u8, 8, 16]);
                lines.push(format!("li {stride_reg}, {stride}"));
            }
            let masked = pool.mask_live && rng.random_range(0..2) == 0;
            let line = if masked {
                format!("vlsseg3e64.v v{base}, ({base_reg}), {stride_reg}, v0.t")
            } else {
                format!("vlsseg3e64.v v{base}, ({base_reg}), {stride_reg}")
            };
            lines.push(line);
            push_attribution_fence(lines);
            pool.push_vec_group(base, field_count);
            usize::from(field_count)
        }
        AccessModel::FaultingStridedStore => {
            let targeted_fault = focused_access_from_env()
                == Some(AccessModel::FaultingStridedStore)
                && !lines.iter().any(|line| line.starts_with("vsse16.v "));
            if targeted_fault {
                // Keep the fault-progress relation reachable even in the
                // positive-memory profile used by the model-only generator.
                lines.push("li x20, 0".to_string());
            }
            if options.constraints_enabled {
                if pool.config.cfg.sew != 16 {
                    let config = RvvLiveConfig::constrained_e16_m1();
                    pool.set_config(config);
                    if options.constraints_enabled {
                        lines.push(config.vsetivli_line());
                    }
                }
                lines.push("csrrw x0, vstart, x0".to_string());
                pool.ensure_compatible_live(lines, rng);
            }
            let vs = pool.consume_latest_vec();
            let masked = pool.mask_live && rng.random_range(0..2) == 0;
            let line = if masked {
                format!("vsse16.v v{vs}, (x20), x21, v0.t")
            } else {
                format!("vsse16.v v{vs}, (x20), x21")
            };
            lines.push(line);
            push_attribution_fence(lines);
            if targeted_fault {
                lines.push("mv x20, x12".to_string());
            }
            1
        }
        AccessModel::ScalarObserve => {
            let vs = pool.consume_latest_vec();
            push_rvv_line(lines, rng, options, format!("vmv.x.s x15, v{vs}"));
            lines.push("sd x15, 64(x11)".to_string());
            1
        }
    }
}

fn build_stateflow_lines_and_edges(
    base_addr: u64,
    seed: u64,
    vector_range: (u8, u8),
    flavor: RvvStateFlowFlavor,
) -> (Vec<String>, usize) {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5353_4d52_434d_0001);
    let mut options = StateFlowOptions::for_flavor(flavor);
    let focused_access = focused_access_from_env();
    if focused_access.is_some() {
        // A focused access emits its complete producer/configuration/consumer
        // relation. Keep unrelated SF episodes from obscuring an early target
        // difference or consuming the DUT's cycle budget before readback.
        options.episode_count = 1;
        options.steps_per_episode = 1;
    }
    if options.positive_memory {
        options.indexed_alignment = IndexedAlignmentScenario::for_seed(seed);
    }
    let out_addr = base_addr + 0x80;
    let mask_addr = base_addr + 0x100;
    let mut lines = vec![
        format!("li x11, {base_addr}"),
        format!("li x12, {out_addr}"),
        format!("li x10, {mask_addr}"),
        "li x13, 7".to_string(),
    ];
    let data_words = if options.ara_generic_enabled || options.ara_safe_enabled {
        [4, 8, 12, 16, 20, 24, 28, 32, 0, 36, 40, 44, 48, 52, 56, 60]
    } else {
        [
            0x0000_0001_0000_0002,
            0x0000_0003_0000_0004,
            0xffff_ffff_0000_0005,
            0x8000_0000_7fff_ffff,
            0x0f0f_0f0f_0f0f_0f0f,
            0x5555_5555_5555_5555,
            0xaaaa_aaaa_aaaa_aaaa,
            0x7fff_ffff_8000_0000,
            0x0102_0304_0506_0708,
            0xfefd_fcfb_faf9_f8f7,
            0x1122_3344_5566_7788,
            0x8877_6655_4433_2211,
            0x0000_00ff_0000_0080,
            0xffff_ff80_0000_007f,
            0x3f80_0000_7fc0_0001,
            0x7f80_0001_ff80_0001,
        ]
    };
    let out_words = if options.ara_generic_enabled || options.ara_safe_enabled {
        [255, 128, 127, 64, 63, 32, 31, 16, 15, 8, 7, 5, 4, 3, 2, 1]
    } else {
        [
            0xffff_ffff_ffff_ffff,
            0xaaaa_aaaa_aaaa_aaaa,
            0x5555_5555_5555_5555,
            0x0f0f_0f0f_0f0f_0f0f,
            0x0102_0304_0506_0708,
            0xfefd_fcfb_faf9_f8f7,
            0x1122_3344_5566_7788,
            0x8877_6655_4433_2211,
            0x0000_00ff_0000_0080,
            0xffff_ff80_0000_007f,
            0x3f80_0000_7fc0_0001,
            0x7f80_0001_ff80_0001,
            0x0000_0001_0000_0002,
            0x0000_0003_0000_0004,
            0xffff_ffff_0000_0005,
            0x8000_0000_7fff_ffff,
        ]
    };
    let mask_words = if options.ara_generic_enabled || options.ara_safe_enabled {
        [0xaa, 0x55, 0xf0, 0x0f, 0xff, 0, 0x33, 0xcc]
    } else {
        [
            0x0000_0000_0000_00aa,
            0x0000_0000_0000_0055,
            0x0000_0000_0000_00f0,
            0x0000_0000_0000_000f,
            0xffff_ffff_ffff_ffff,
            0x0000_0000_0000_0000,
            0x3333_3333_3333_3333,
            0xcccc_cccc_cccc_cccc,
        ]
    };
    let preload_words = if focused_access.is_some() { 8 } else { 16 };
    push_data_words(&mut lines, "x11", 0, &data_words[..preload_words]);
    push_data_words(&mut lines, "x12", 0, &out_words[..preload_words]);
    push_data_words(&mut lines, "x10", 0, &mask_words);
    lines.push("li x14, 8".to_string());
    lines.push(if options.positive_memory {
        format!("li x20, {out_addr}")
    } else {
        "li x20, 0".to_string()
    });
    lines.push("li x21, 16".to_string());

    let mut state_flow_edges = 0usize;
    let mut ara_budget = if options.ara_generic_enabled {
        Some(AraRiskBudget::default())
    } else {
        None
    };
    for episode in 0..options.episode_count {
        if !options.sequence_enabled {
            if episode > 0 {
                maybe_push_scalar_noise(&mut lines, &mut rng, options);
            }
            for _step in 0..options.steps_per_episode {
                let config = choose_stateflow_config(&mut rng, options, vector_range);
                let mut pool = RvvStatePool::new(config, vector_range);
                if options.vector_setup_enabled() {
                    lines.push(config.vsetivli_line());
                    lines.push("csrrw x0, vstart, x0".to_string());
                }
                let access = if options.ara_generic_enabled {
                    ara_singleton_focus_access(options.ara_risk_focus).unwrap_or_else(|| {
                        let candidates =
                            ara_singleton_access_candidates(options, ara_budget.as_ref());
                        choose_weighted_access(&mut rng, &candidates)
                    })
                } else {
                    let candidates =
                        weighted_access_candidates(&pool, options, ara_budget.as_ref());
                    choose_weighted_access(&mut rng, &candidates)
                };
                state_flow_edges += emit_access(&mut lines, &mut pool, &mut rng, options, access);
                if let Some(budget) = ara_budget.as_mut() {
                    budget.record(access);
                }
            }
            continue;
        }

        let config = choose_stateflow_config(&mut rng, options, vector_range);
        let mut pool = RvvStatePool::new(config, vector_range);
        if options.vector_setup_enabled() {
            lines.push(config.vsetivli_line());
            lines.push("csrrw x0, vstart, x0".to_string());
        }
        if episode > 0 {
            maybe_push_scalar_noise(&mut lines, &mut rng, options);
        }

        // Defer the specialized Ara risk probe in coverage mode so a fault
        // cannot hide the ordinary producer/consumer paths in either episode.
        let risk_position = if options.coverage_focused {
            (options.episode_count - 1, options.steps_per_episode - 1)
        } else {
            (0, 1)
        };
        for step in 0..options.steps_per_episode {
            let scheduled_indexed = options.positive_memory
                && options.ara_risk_focus == AraRiskFocus::Auto
                && episode == 0
                && step == 1;
            let access = if episode == 0 && step == 0 && focused_access.is_some() {
                focused_access.expect("focus was checked")
            } else if scheduled_indexed
                && options.indexed_alignment == IndexedAlignmentScenario::MaskedOffMisaligned
            {
                AccessModel::MaskedIndexedMisalignmentProbe
            } else if scheduled_indexed
                && options.indexed_alignment == IndexedAlignmentScenario::ActiveMisaligned
            {
                AccessModel::UnmaskedIndexedMisalignmentProbe
            } else if options.coverage_focused && episode == 0 && step == 0 && seed % 4 == 0 {
                AccessModel::ConfigReadback
            } else if options.coverage_focused && episode == 0 && step == 0 && seed % 4 == 1 {
                AccessModel::FractionalGather
            } else if options.ara_generic_enabled {
                if let Some(forced) = options.ara_risk_focus.forced_access(&pool, episode, step) {
                    forced
                } else if options.ara_risk_focus != AraRiskFocus::Auto {
                    ara_post_focus_access(&pool)
                } else if (episode, step) == risk_position {
                    ara_initial_focus_access(&mut rng)
                } else {
                    let candidates =
                        weighted_access_candidates(&pool, options, ara_budget.as_ref());
                    choose_weighted_access(&mut rng, &candidates)
                }
            } else {
                let candidates = weighted_access_candidates(&pool, options, ara_budget.as_ref());
                choose_weighted_access(&mut rng, &candidates)
            };
            // Preserve the named masked-off/active-fault focus; only unrelated
            // random indexed accesses are forced back to aligned addresses.
            let mut access_options = options;
            access_options.indexed_alignment = indexed_alignment_for_access(
                options.indexed_alignment,
                access,
                focused_access,
                scheduled_indexed,
            );
            state_flow_edges +=
                emit_access(&mut lines, &mut pool, &mut rng, access_options, access);
            if let Some(budget) = ara_budget.as_mut() {
                budget.record(access);
            }
        }
        if !pool.live_vecs.is_empty() {
            state_flow_edges += emit_access(
                &mut lines,
                &mut pool,
                &mut rng,
                options,
                AccessModel::ScalarObserve,
            );
        }
    }
    (lines, state_flow_edges)
}

#[cfg(test)]
fn build_stateflow_lines(
    base_addr: u64,
    seed: u64,
    vector_range: (u8, u8),
    flavor: RvvStateFlowFlavor,
) -> Vec<String> {
    build_stateflow_lines_and_edges(base_addr, seed, vector_range, flavor).0
}

fn shift_shared_indices(
    indices: &mut std::collections::BTreeSet<usize>,
    insertion: usize,
    inserted_len: usize,
) {
    if inserted_len == 0 || indices.is_empty() {
        return;
    }
    *indices = indices
        .iter()
        .map(|idx| {
            if *idx >= insertion {
                idx + inserted_len
            } else {
                *idx
            }
        })
        .collect();
}

pub fn inject_vector_state_enable(
    testcase: &mut GeneratedTestCase,
) -> Result<usize, GenerationError> {
    if testcase.config.isa_base != ISABase::Rv64 || testcase.harts.is_empty() {
        return Ok(0);
    }

    let lines = [
        "li x5, 13824".to_string(),
        "csrrs x0, mstatus, x5".to_string(),
    ];
    let instructions = decode_block(&lines, testcase.config.isa_base)?;
    let inserted_len = instructions.len();

    for hart in testcase.harts.iter_mut() {
        let test = hart.sequences_mut(HartPhase::Test);
        test.splice(0..0, instructions.clone());
        shift_shared_indices(
            hart.shared_sequence_indices_mut(HartPhase::Test),
            0,
            inserted_len,
        );
    }

    Ok(inserted_len)
}

pub fn inject_ambient_rvv_config_setup(
    testcase: &mut GeneratedTestCase,
) -> Result<usize, GenerationError> {
    if testcase.config.isa_base != ISABase::Rv64 || testcase.harts.is_empty() {
        return Ok(0);
    }

    let config = RvvLiveConfig::constrained_e32_m1();
    let lines = [config.vsetivli_line(), "csrrw x0, vstart, x0".to_string()];
    let instructions = decode_block(&lines, testcase.config.isa_base)?;
    let inserted_len = instructions.len();

    for hart in testcase.harts.iter_mut() {
        let insertion = hart.sequences(HartPhase::Test).len().min(2);
        let test = hart.sequences_mut(HartPhase::Test);
        test.splice(insertion..insertion, instructions.clone());
        shift_shared_indices(
            hart.shared_sequence_indices_mut(HartPhase::Test),
            insertion,
            inserted_len,
        );
    }

    Ok(inserted_len)
}

pub fn inject_hardware_rvv_stateflow_with_flavor(
    testcase: &mut GeneratedTestCase,
    core_ids: &[String],
    flavor: RvvStateFlowFlavor,
) -> Result<RvvStateFlowInjectionReport, GenerationError> {
    if testcase.config.isa_base != ISABase::Rv64 {
        return Ok(RvvStateFlowInjectionReport::default());
    }

    let Some(core_id) = core_ids.first() else {
        return Ok(RvvStateFlowInjectionReport::default());
    };
    let Some((start, end)) = testcase.mem_range.get(core_id).copied() else {
        return Err(GenerationError::MemRangeNotFound {
            core_id: core_id.clone(),
        });
    };
    if testcase.harts.is_empty() {
        return Ok(RvvStateFlowInjectionReport::default());
    }

    let model_bytes = 0x140;
    let base_addr = aligned_stateflow_address(start, end, model_bytes)?;
    let seed = testcase.config.seed.unwrap_or(0);
    let vector_range = testcase
        .config
        .random_config
        .register_operand_config
        .vector_register_number_range;
    let (lines, state_flow_edges) =
        build_stateflow_lines_and_edges(base_addr, seed, vector_range, flavor);
    let instructions = decode_block(&lines, testcase.config.isa_base)?;
    let inserted_len = instructions.len();

    let hart = &mut testcase.harts[0];
    let test = hart.sequences_mut(HartPhase::Test);
    let insertion = 0;
    test.splice(insertion..insertion, instructions);
    shift_shared_indices(
        hart.shared_sequence_indices_mut(HartPhase::Test),
        insertion,
        inserted_len,
    );

    Ok(RvvStateFlowInjectionReport {
        inserted_blocks: 1,
        inserted_instructions: inserted_len,
        state_flow_edges,
    })
}

pub fn append_hardware_rvv_stateflow_with_flavor(
    testcase: &mut GeneratedTestCase,
    core_ids: &[String],
    flavor: RvvStateFlowFlavor,
) -> Result<RvvStateFlowInjectionReport, GenerationError> {
    if testcase.config.isa_base != ISABase::Rv64 {
        return Ok(RvvStateFlowInjectionReport::default());
    }

    let Some(core_id) = core_ids.first() else {
        return Ok(RvvStateFlowInjectionReport::default());
    };
    let Some((start, end)) = testcase.mem_range.get(core_id).copied() else {
        return Err(GenerationError::MemRangeNotFound {
            core_id: core_id.clone(),
        });
    };
    if testcase.harts.is_empty() {
        return Ok(RvvStateFlowInjectionReport::default());
    }

    let model_bytes = 0x140;
    let base_addr = aligned_stateflow_address(start, end, model_bytes)?;
    let seed = testcase.config.seed.unwrap_or(0) ^ 0x4152_415f_4150_5045;
    let vector_range = testcase
        .config
        .random_config
        .register_operand_config
        .vector_register_number_range;
    let (lines, state_flow_edges) =
        build_stateflow_lines_and_edges(base_addr, seed, vector_range, flavor);
    let instructions = decode_block(&lines, testcase.config.isa_base)?;
    let inserted_len = instructions.len();

    let hart = &mut testcase.harts[0];
    let insertion = hart.sequences(HartPhase::Test).len();
    let test = hart.sequences_mut(HartPhase::Test);
    test.splice(insertion..insertion, instructions);
    shift_shared_indices(
        hart.shared_sequence_indices_mut(HartPhase::Test),
        insertion,
        inserted_len,
    );

    Ok(RvvStateFlowInjectionReport {
        inserted_blocks: 1,
        inserted_instructions: inserted_len,
        state_flow_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rendered(flavor: RvvStateFlowFlavor, seed: u64) -> String {
        build_stateflow_lines(0x8001_0000, seed, (13, 31), flavor).join("\n")
    }

    #[test]
    fn slideup_avoids_source_even_when_all_live_groups_were_used() {
        let config = RvvLiveConfig {
            cfg: TemplateRvvConfig {
                sew: 32,
                vl: 8,
                lmul: "m4",
                tail_policy: "tu",
                mask_policy: "mu",
            },
        };
        let mut destinations = std::collections::BTreeSet::new();
        for seed in 0..64 {
            let mut pool = RvvStatePool::new(config, (13, 31));
            for base in [16, 20, 24, 28] {
                pool.record_group(base, 4);
            }
            pool.push_vec(28, 4);
            pool.mask_live = true;
            let mut rng = StdRng::seed_from_u64(2044300315 + seed);
            let mut lines = Vec::new();
            emit_access(
                &mut lines,
                &mut pool,
                &mut rng,
                StateFlowOptions::for_flavor(RvvStateFlowFlavor::PositiveAraGeneric),
                AccessModel::SlideWithMask,
            );
            let slide = lines
                .iter()
                .find(|line| line.starts_with("vslideup.vx "))
                .unwrap();
            let register = slide.split_whitespace().nth(1).unwrap();
            let vd: u8 = register
                .trim_end_matches(',')
                .trim_start_matches('v')
                .parse()
                .unwrap();
            assert!(crate::rvv_legal::valid_group(vd, 4));
            assert!(!crate::rvv_legal::groups_overlap(vd, 4, 28, 4), "{slide}");
            destinations.insert(vd);
        }
        assert!(
            destinations.len() >= 2,
            "slide destinations should remain randomized"
        );
    }

    #[test]
    fn focused_indexed_access_keeps_its_intentional_alignment_state() {
        assert_eq!(
            indexed_alignment_for_access(
                IndexedAlignmentScenario::Aligned,
                AccessModel::MaskedIndexedMisalignmentProbe,
                Some(AccessModel::MaskedIndexedMisalignmentProbe),
                false,
            ),
            IndexedAlignmentScenario::MaskedOffMisaligned,
        );
        assert_eq!(
            indexed_alignment_for_access(
                IndexedAlignmentScenario::Aligned,
                AccessModel::UnmaskedIndexedMisalignmentProbe,
                Some(AccessModel::UnmaskedIndexedMisalignmentProbe),
                false,
            ),
            IndexedAlignmentScenario::ActiveMisaligned,
        );
        assert_eq!(
            indexed_alignment_for_access(
                IndexedAlignmentScenario::ActiveMisaligned,
                AccessModel::IndexedLoad,
                None,
                false,
            ),
            IndexedAlignmentScenario::Aligned,
        );
    }

    #[test]
    fn focused_control_and_fractional_gather_decode() {
        let mut options = StateFlowOptions::for_flavor(RvvStateFlowFlavor::AraGeneric);
        options.coverage_focused = true;
        let config = RvvLiveConfig::constrained_e32_m1();
        let mut pool = RvvStatePool::new(config, (13, 31));
        let mut rng = StdRng::seed_from_u64(2026101601);
        let mut lines = Vec::new();
        emit_access(
            &mut lines,
            &mut pool,
            &mut rng,
            options,
            AccessModel::ConfigReadback,
        );
        emit_access(
            &mut lines,
            &mut pool,
            &mut rng,
            options,
            AccessModel::FractionalGather,
        );
        decode_block(&lines, ISABase::Rv64).expect("focused state-flow block decodes");
        let text = lines.join("\n");
        assert!(text.contains("vsetvl x15, x13, x14"));
        assert!(text.contains("vsetivli x0, 24, 7"));
        assert!(text.contains("vrgather.vi"));
        assert!(text.contains("vse8.v"));
        assert!(text.contains("vmv.x.s"));
        assert_eq!(pool.config, config);
    }

    #[test]
    fn ara_safe_stateflow_decodes_and_avoids_risky_memory_forms() {
        for seed in 2026052600..2026052664 {
            let lines =
                build_stateflow_lines(0x8001_0000, seed, (13, 31), RvvStateFlowFlavor::AraSafe);
            decode_block(&lines, ISABase::Rv64).expect("ARA-safe stateflow block decodes");
            let text = lines.join("\n");
            assert!(text.contains("vsetivli"));
            assert!(text.contains("csrrw x0, vstart, x0"));
            assert!(text.contains("vle32.v") || text.contains("vse32.v"));
            for risky in [
                "vlseg", "vlsseg", "vle8ff", "vle16ff", "vle32ff", "vle64ff", "vl1re", "vl4re",
                "vl8re", "vs4r", "vs8r", "vlse", "vsse",
            ] {
                assert!(
                    !text.contains(risky),
                    "ARA-safe stateflow should not contain {risky}:\n{text}"
                );
            }
        }
    }

    #[test]
    fn ara_generic_stateflow_decodes_and_covers_generic_risk_families() {
        let mut saw_indexed = false;
        let mut saw_segment = false;
        let mut saw_fof = false;
        let mut saw_whole_register = false;
        let mut saw_vstart_store = false;
        let mut saw_slide = false;
        let mut saw_large_slide_window = false;
        let mut saw_widen_shift_observe = false;
        let mut saw_misaligned_index_probe = false;
        let mut saw_segment_later_field_probe = false;
        let mut saw_segment_later_field_strict_probe = false;
        let mut saw_sparse_mask_segment_probe = false;

        for seed in 2026052600..2026053600 {
            let lines =
                build_stateflow_lines(0x8001_0000, seed, (13, 31), RvvStateFlowFlavor::AraGeneric);
            decode_block(&lines, ISABase::Rv64).expect("ARA generic stateflow block decodes");
            let text = lines.join("\n");
            assert!(!text.contains("x22"));
            assert!(text.contains("vsetivli"));
            assert!(text.contains("csrrw x0, vstart, x0"));

            saw_indexed |= text.contains("vluxei32.v") || text.contains("vluxseg");
            saw_segment |= text.contains("vlseg") || text.contains("vlsseg");
            saw_fof |= text.contains("ff.v");
            saw_whole_register |= text.contains("vl8re64.v") || text.contains("vs8r.v");
            saw_vstart_store |= text.contains("csrrw x0, vstart, x5") && text.contains("vse32.v");
            saw_slide |= text.contains("vslideup.vx");
            saw_large_slide_window |= (text.contains("li x5, 260")
                || text.contains("li x5, 264")
                || text.contains("li x5, 272"))
                && text.contains("li x6, 256")
                && text.contains("vslideup.vx")
                && text.contains("andi x15, x15, 255");
            saw_widen_shift_observe |= text.contains("vzext.vf8") && text.contains("vslidedown.vi");
            saw_misaligned_index_probe |= text.contains("li x14, 1")
                && text.contains("sw x14, 0(x11)")
                && text.contains("vluxei32.v")
                && text.contains("(x12)");
            saw_segment_later_field_probe |=
                text.contains("vluxseg5ei32.v") || text.contains("vlsseg3e64.v");
            saw_segment_later_field_strict_probe |= text.contains("vluxseg5ei32.v")
                && text.contains("vmv.x.s x15, v")
                && text.contains("sd x15, 104(x12)");
            saw_sparse_mask_segment_probe |=
                text.contains("vlseg2e16.v") && text.contains("vslidedown.vi");
        }

        assert!(
            saw_indexed,
            "ARA generic profile should cover indexed vector memory"
        );
        assert!(
            saw_segment,
            "ARA generic profile should cover segment memory"
        );
        assert!(
            saw_fof,
            "ARA generic profile should cover fault-only-first loads"
        );
        assert!(
            saw_whole_register,
            "ARA generic profile should cover whole-register transfers"
        );
        assert!(
            saw_vstart_store,
            "ARA generic profile should cover vstart stores"
        );
        assert!(
            saw_slide,
            "ARA generic profile should cover masked slide state"
        );
        assert!(
            saw_large_slide_window,
            "ARA generic profile should cover large masked slide windows"
        );
        assert!(
            saw_widen_shift_observe,
            "ARA generic profile should cover zero-extend plus lane-shift observation"
        );
        assert!(
            saw_misaligned_index_probe,
            "ARA generic profile should cover indexed misalignment sensitivity"
        );
        assert!(
            saw_segment_later_field_probe,
            "ARA generic profile should cover segment later-field observation"
        );
        assert!(
            saw_segment_later_field_strict_probe,
            "ARA generic profile should cover strict segment later-field-first observation"
        );
        assert!(
            saw_sparse_mask_segment_probe,
            "ARA generic profile should cover sparse-mask segment lane observation"
        );
    }

    #[test]
    fn ara_ablation_flavors_have_expected_knobs() {
        let full = StateFlowOptions::for_flavor(RvvStateFlowFlavor::AraGeneric);
        assert!(full.constraints_enabled);
        assert!(full.sequence_enabled);

        let no_constraints = StateFlowOptions::for_flavor(RvvStateFlowFlavor::AraNoConstraints);
        assert!(!no_constraints.constraints_enabled);
        assert!(no_constraints.sequence_enabled);
        assert!(no_constraints.ara_generic_enabled);
        assert!(!no_constraints.register_constraints_enabled());
        assert!(!no_constraints.vector_setup_enabled());

        let no_sequence = StateFlowOptions::for_flavor(RvvStateFlowFlavor::AraNoSequence);
        assert!(no_sequence.constraints_enabled);
        assert!(!no_sequence.sequence_enabled);
        assert!(!no_sequence.ara_generic_enabled);
        assert!(no_sequence.vector_setup_enabled());

        let neither = StateFlowOptions::for_flavor(RvvStateFlowFlavor::AraNoConstraintsNoSequence);
        assert!(!neither.constraints_enabled);
        assert!(!neither.sequence_enabled);
        assert!(!neither.ara_generic_enabled);
        assert!(!neither.register_constraints_enabled());
        assert!(neither.vector_setup_enabled());
    }

    #[test]
    fn positive_flavors_keep_mask_merge_opcode_without_v0_destination() {
        for flavor in [
            RvvStateFlowFlavor::PositiveAraGeneric,
            RvvStateFlowFlavor::PositiveConstrained,
        ] {
            let options = StateFlowOptions::for_flavor(flavor);
            assert!(options.positive_memory);
            for seed in 2026101600..2026101664 {
                let lines = build_stateflow_lines(0x8001_0000, seed, (13, 31), flavor);
                let rendered = lines.join("\n");
                assert!(!rendered.contains("vmerge.vvm v0,"), "{seed}: {rendered}");
                assert!(
                    rendered.contains("li x20, 2147549312"),
                    "{seed}: {rendered}"
                );
                decode_block(&lines, ISABase::Rv64).expect("positive SF block decodes");
            }
        }
    }

    #[test]
    fn indexed_alignment_schedule_has_bounded_reproducible_bug_probes() {
        let scenarios = (0..100)
            .map(IndexedAlignmentScenario::for_seed)
            .collect::<Vec<_>>();
        assert_eq!(
            scenarios
                .iter()
                .filter(|s| **s == IndexedAlignmentScenario::Aligned)
                .count(),
            70
        );
        assert_eq!(
            scenarios
                .iter()
                .filter(|s| **s == IndexedAlignmentScenario::MaskedOffMisaligned)
                .count(),
            20
        );
        assert_eq!(
            scenarios
                .iter()
                .filter(|s| **s == IndexedAlignmentScenario::ActiveMisaligned)
                .count(),
            10
        );

        for (seed, masked) in [(7, true), (8, true), (9, false)] {
            let lines = build_stateflow_lines(
                0x8001_0000,
                seed,
                (13, 31),
                RvvStateFlowFlavor::PositiveAraGeneric,
            );
            let probe = lines
                .iter()
                .find(|line| line.starts_with("vluxei32.v "))
                .expect("scheduled indexed-load probe");
            assert_eq!(probe.ends_with("v0.t"), masked, "{seed}: {probe}");
            assert!(
                lines
                    .iter()
                    .any(|line| line.starts_with("vsetivli x0, 4, 16"))
            );
        }
    }

    #[test]
    fn positive_indexed_preload_bounds_vl_after_dynamic_config() {
        let mut rng = StdRng::seed_from_u64(20261016);
        let mut pool = RvvStatePool::new(RvvLiveConfig::constrained_e32_m1(), (13, 31));
        pool.config.cfg.vl = 16;
        let mut lines = Vec::new();
        let options = StateFlowOptions::for_flavor(RvvStateFlowFlavor::PositiveAraGeneric);
        ensure_indexed_load_config(&mut lines, &mut pool, &mut rng, options);
        assert!(pool.config.cfg.vl <= 8);
        assert!(
            lines
                .first()
                .is_some_and(|line| line.starts_with("vsetivli x0,"))
        );
        decode_block(&lines, ISABase::Rv64).expect("bounded indexed setup decodes");
    }

    #[test]
    fn indexed_segment_pairs_never_overlap() {
        let mut rng = StdRng::seed_from_u64(20261016);
        let mut pool = RvvStatePool::new(RvvLiveConfig::constrained_e32_m1(), (13, 31));
        for field_count in [2, 3, 5].into_iter().cycle().take(128) {
            let (base, index) = pool.allocate_indexed_segment_pair(&mut rng, field_count, 32, None);
            assert!(!crate::rvv_legal::groups_overlap(
                base,
                field_count,
                index,
                1
            ));
        }
    }

    #[test]
    fn ara_constrained_no_sequence_stateflow_blocks_decode() {
        for seed in 2026052600..2026052664 {
            let lines = build_stateflow_lines(
                0x8001_0000,
                seed,
                (13, 31),
                RvvStateFlowFlavor::AraNoSequence,
            );
            decode_block(&lines, ISABase::Rv64)
                .expect("ARA constrained no-sequence stateflow block decodes");
        }
    }

    #[test]
    fn ara_no_constraints_no_sequence_emits_rvv_setup_without_register_constraints() {
        let options = StateFlowOptions::for_flavor(RvvStateFlowFlavor::AraNoConstraintsNoSequence);
        assert!(!options.register_constraints_enabled());
        for seed in 2026052600..2026052616 {
            let lines = build_stateflow_lines(
                0x8001_0000,
                seed,
                (13, 31),
                RvvStateFlowFlavor::AraNoConstraintsNoSequence,
            );
            decode_block(&lines, ISABase::Rv64)
                .expect("ARA no-constraints/no-sequence stateflow block decodes");
            let text = lines.join("\n");
            assert!(
                text.contains("vsetivli") && text.contains("csrrw x0, vstart, x0"),
                "RISCSmith ARA baseline should set RVV state before random RVV ops:\n{text}"
            );
        }
    }

    #[test]
    fn ara_no_sequence_stateflow_drops_specialized_risk_families() {
        for seed in 2026052600..2026052800 {
            let text = rendered(RvvStateFlowFlavor::AraNoSequence, seed);
            assert!(!text.contains("csrrw x0, vstart, x5"));
            assert!(!text.contains("vmerge.vvm v0,"));
            assert!(!text.contains("vluxseg5ei32.v"));
            assert!(!text.contains("vlsseg3e64.v"));
            assert!(!text.contains("andi x15, x15, 255"));
        }
    }

    #[test]
    fn ara_generic_stateflow_keeps_heavy_memory_budget_bounded() {
        for seed in 2026052600..2026052860 {
            let lines =
                build_stateflow_lines(0x8001_0000, seed, (13, 31), RvvStateFlowFlavor::AraGeneric);
            let heavy_memory_ops = lines
                .iter()
                .filter(|line| {
                    line.contains("ff.v")
                        || line.contains("vluxei")
                        || line.contains("vloxei")
                        || line.contains("vluxseg")
                        || line.contains("vloxseg")
                        || line.contains("vlsseg")
                        || line.contains("vlseg")
                        || line.contains("vl8re64.v")
                        || line.contains("vs4r.v")
                        || line.contains("vsse16.v")
                })
                .count();
            let text = lines.join("\n");
            assert!(
                heavy_memory_ops <= AraRiskBudget::MAX_HEAVY_MEMORY_OPS,
                "ARA generic should bound heavy memory ops, got {heavy_memory_ops}:\n{text}"
            );
            assert!(
                text.matches("csrrw x0, vstart, x5").count() <= AraRiskBudget::MAX_VSTART_STORE_OPS,
                "ARA generic should bound nonzero-vstart store probes:\n{text}"
            );
            assert!(
                text.matches("vslideup.vx").count() <= AraRiskBudget::MAX_SLIDE_OPS,
                "ARA generic should bound slide probes:\n{text}"
            );
        }
    }

    #[test]
    fn ara_risk_budget_does_not_apply_to_xiangshan_stateflow_flavors() {
        let constrained = StateFlowOptions::for_flavor(RvvStateFlowFlavor::StateFlowConstrained);
        let mut budget = AraRiskBudget::default();
        for _ in 0..AraRiskBudget::MAX_HEAVY_MEMORY_OPS {
            budget.record(AccessModel::SegmentLoad);
            budget.record(AccessModel::LoadVec);
        }
        let candidates = apply_ara_risk_budget(
            constrained,
            vec![(AccessModel::SegmentLoad, 1), (AccessModel::LoadVec, 1)],
            Some(&budget),
        );
        assert!(
            candidates
                .iter()
                .any(|(access, _)| *access == AccessModel::SegmentLoad),
            "non-ARA stateflow flavors must not be filtered by ARA risk budget"
        );
    }

    #[test]
    fn ara_risk_focus_maps_remaining_bugs_to_specific_families() {
        let config = RvvLiveConfig::constrained_e32_m1();
        let pool = RvvStatePool::new(config, (13, 31));
        assert_eq!(
            AraRiskFocus::WidenZeroExtend.forced_access(&pool, 0, 0),
            Some(AccessModel::WidenZeroExtendSlideObserve)
        );
        assert_eq!(
            AraRiskFocus::LargeMaskedSlide.forced_access(&pool, 0, 0),
            Some(AccessModel::LargeMaskedSlideWindowObserve)
        );
        assert_eq!(
            AraRiskFocus::VstartMaskedStore.forced_access(&pool, 0, 0),
            Some(AccessModel::VstartMaskedStoreProbe)
        );
        assert_eq!(
            AraRiskFocus::MaskedIndexedMisalignment.forced_access(&pool, 0, 0),
            Some(AccessModel::MaskedIndexedMisalignmentProbe)
        );
        assert_eq!(
            AraRiskFocus::UnmaskedIndexedMisalignment.forced_access(&pool, 0, 0),
            Some(AccessModel::UnmaskedIndexedMisalignmentProbe)
        );
        assert_eq!(
            AraRiskFocus::WidenZeroExtend.forced_access(&pool, 0, 1),
            None,
            "single-family focus should not keep forcing risky probes after the first step"
        );
    }

    #[test]
    fn constrained_stateflow_does_not_use_ara_generic_scratch_or_indexed_ops() {
        for seed in 2026052600..2026052664 {
            let lines = build_stateflow_lines(
                0x8001_0000,
                seed,
                (13, 31),
                RvvStateFlowFlavor::StateFlowConstrained,
            );
            let text = lines.join("\n");
            assert!(!text.contains("x22"));
            assert!(!text.contains("vlux"));
            assert!(!text.contains("vslidedown.vi"));
            assert!(!text.contains("sw x14, 0(x11)"));
        }
    }

    #[test]
    fn constrained_stateflow_block_decodes() {
        let lines = build_stateflow_lines(
            0x8001_0000,
            2026052600,
            (13, 31),
            RvvStateFlowFlavor::StateFlowConstrained,
        );
        let decoded = decode_block(&lines, ISABase::Rv64).expect("stateflow block decodes");
        assert_eq!(decoded.len(), lines.len());
        let text = lines.join("\n");
        assert!(text.contains("vl1re64.v v0") || text.contains("vlm.v v0"));
        assert!(text.contains("vle32.v"));
        assert!(text.contains("vmv.x.s"));
        assert!(text.contains("csrrw x0, vstart, x0"));
    }

    #[test]
    fn constrained_stateflow_many_seeds_decode() {
        for seed in 2026052600..2026052664 {
            let lines = build_stateflow_lines(
                0x8001_0000,
                seed,
                (13, 31),
                RvvStateFlowFlavor::StateFlowConstrained,
            );
            decode_block(&lines, ISABase::Rv64).expect("stateflow block decodes");
        }
    }

    #[test]
    fn stateflow_only_keeps_state_model_without_config_or_vstart() {
        let mut saw_rcm_heavy_state_model = false;
        for seed in 2026052600..2026052664 {
            let lines = build_stateflow_lines(
                0x8001_0000,
                seed,
                (13, 31),
                RvvStateFlowFlavor::StateFlowOnly,
            );
            decode_block(&lines, ISABase::Rv64).expect("stateflow-only block decodes");
            let text = lines.join("\n");
            for line in &lines {
                let mnemonic = line.split_whitespace().next().unwrap_or_default();
                assert!(
                    !matches!(mnemonic, "vsetivli" | "vsetvli" | "vsetvl"),
                    "SSM-only must not insert vector config: {line}"
                );
                assert!(
                    !line.contains("vstart"),
                    "SSM-only must not insert vstart setup: {line}"
                );
            }
            saw_rcm_heavy_state_model |= [
                "vlseg",
                "vlsseg",
                "vle8ff",
                "vle16ff",
                "vle32ff",
                "vle64ff",
                "vsse",
                "vl8re",
                "vs8r",
                "vs4r",
                "vfmv.f.s",
                "vzext.vf8",
                "vlm.v",
                "vsm.v",
            ]
            .iter()
            .any(|model| text.contains(model));
        }
        assert!(
            saw_rcm_heavy_state_model,
            "SSM-only should keep the same state-model access families as SSM+RCM"
        );
    }

    #[test]
    fn ambient_stateflow_keeps_edges_but_varies_config() {
        let left = rendered(RvvStateFlowFlavor::StateFlowOnly, 2026052601);
        let right = rendered(RvvStateFlowFlavor::StateFlowOnly, 2026052602);
        assert_ne!(left, right);
        assert!(left.contains("vle32.v"));
        assert!(left.contains("vmv.x.s"));
    }
}
