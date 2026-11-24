use std::collections::BTreeSet;

use rand::{Rng as _, SeedableRng as _, rngs::StdRng, seq::SliceRandom as _};
use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
use riscv_instruction::{RV64Extensions, RiscvInstruction};
use riscv_instruction_types::{AsmDecodeWithContext, InstructionSequence, IsaDecodeContext};

use crate::{
    GenerationError,
    testcase::{GeneratedTestCase, HartPhase},
};

/// Minimal RVV semantic contract extracted from the ISA-facing sources we trust:
/// static instruction shape from riscv-unified-db and dynamic requirements from
/// Spike's RVV instruction bodies/macros.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvvObligationKind {
    /// `vset{i}vli`: `rd` and the architectural `vl` CSR must agree.
    VlVtypeConfig,
    /// `vmv.x.s`: scalar integer destination must receive `sext_xlen(vs2[0])`.
    ScalarExtractSignExt,
    /// `vfmv.f.s` observed through `fmv.x.d`: single-precision value must be NaN-boxed.
    FloatExtractNanBox,
    /// Mask values are packed in memory and must round-trip through `vlm.v`/`vsm.v`.
    MaskMemoryRoundTrip,
    /// Mask-producing operations must preserve mask carrier semantics even with `v0` aliasing.
    MaskMergeCarry,
    /// Tail/high vector-register state must survive legal vtype/vl transitions.
    TailHighTransition,
    /// Faulting vector memory operations must expose architectural progress in `vstart`.
    VstartFaultProgress,
    /// Whole-register load/store operations must preserve register-layout bytes.
    WholeRegisterRoundTrip,
    /// Strided element producers must feed whole-register stores with the same register layout.
    StridedWholeRegisterStore,
    /// Widening integer operations must zero-extend from the source element width.
    WideningZeroExtend,
    /// Segment memory operations must map fields to vector-register groups correctly.
    SegmentMemoryLayout,
    /// Fault-only-first segment loads must update data and `vl` consistently.
    FaultOnlyFirstSegment,
    /// Indexed segment stores followed by FOF loads must preserve index/data lane semantics.
    IndexedSegmentFof,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvvCarrierKind {
    VectorRegister,
    IntegerRegister,
    FloatingRegister,
    Memory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RvvRegionContract {
    pub carrier: RvvCarrierKind,
    pub operand: &'static str,
    pub region: &'static str,
    pub ty: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RvvInstructionContract {
    pub name: &'static str,
    pub obligation: RvvObligationKind,
    pub reads: &'static [RvvRegionContract],
    pub writes: &'static [RvvRegionContract],
    pub postcondition: &'static str,
}

const VMV_X_S_READS: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::VectorRegister,
    operand: "vs2",
    region: "element[0]",
    ty: "intSEW",
}];
const VMV_X_S_WRITES: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::IntegerRegister,
    operand: "rd",
    region: "xlen",
    ty: "intXLEN",
}];

const VFMV_F_S_READS: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::VectorRegister,
    operand: "vs2",
    region: "element[0]",
    ty: "bitsSEW",
}];
const VFMV_F_S_WRITES: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::FloatingRegister,
    operand: "fd",
    region: "flen",
    ty: "floatFLEN",
}];

const CONFIG_WRITES: &[RvvRegionContract] = &[
    RvvRegionContract {
        carrier: RvvCarrierKind::IntegerRegister,
        operand: "rd",
        region: "xlen",
        ty: "vl",
    },
    RvvRegionContract {
        carrier: RvvCarrierKind::VectorRegister,
        operand: "csr",
        region: "vl/vtype",
        ty: "config",
    },
];
const MASK_MEMORY_READS: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::Memory,
    operand: "rs1",
    region: "packed_mask[0..ceil(vl/8))",
    ty: "mask",
}];
const MASK_MEMORY_WRITES: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::Memory,
    operand: "rs1",
    region: "packed_mask[0..ceil(vl/8))",
    ty: "mask",
}];
const MASK_MERGE_READS: &[RvvRegionContract] = &[
    RvvRegionContract {
        carrier: RvvCarrierKind::VectorRegister,
        operand: "vs1",
        region: "active_elements",
        ty: "intSEW",
    },
    RvvRegionContract {
        carrier: RvvCarrierKind::VectorRegister,
        operand: "vs2",
        region: "active_elements",
        ty: "intSEW",
    },
    RvvRegionContract {
        carrier: RvvCarrierKind::VectorRegister,
        operand: "v0",
        region: "mask_bits",
        ty: "mask",
    },
];
const MASK_MERGE_WRITES: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::VectorRegister,
    operand: "vd",
    region: "mask/vector_bits",
    ty: "intSEW",
}];
const VECTOR_STATE_READS: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::VectorRegister,
    operand: "vs",
    region: "active_tail_and_high_bits",
    ty: "bits",
}];
const VECTOR_STATE_WRITES: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::VectorRegister,
    operand: "vd",
    region: "active_tail_and_high_bits",
    ty: "bits",
}];
const MEMORY_VECTOR_READS: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::Memory,
    operand: "rs1",
    region: "addressed_bytes",
    ty: "bytes",
}];
const MEMORY_VECTOR_WRITES: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::Memory,
    operand: "rs1",
    region: "addressed_bytes",
    ty: "bytes",
}];
const VSTART_WRITES: &[RvvRegionContract] = &[RvvRegionContract {
    carrier: RvvCarrierKind::VectorRegister,
    operand: "csr",
    region: "vstart",
    ty: "element_index",
}];

pub fn scalar_extract_sign_ext_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vmv.x.s",
        obligation: RvvObligationKind::ScalarExtractSignExt,
        reads: VMV_X_S_READS,
        writes: VMV_X_S_WRITES,
        postcondition: "rd = sext_xlen(vs2.element0_as_intSEW)",
    }
}

pub fn float_extract_nan_box_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vfmv.f.s",
        obligation: RvvObligationKind::FloatExtractNanBox,
        reads: VFMV_F_S_READS,
        writes: VFMV_F_S_WRITES,
        postcondition: "fd = nan_box_if_sew_less_than_flen(vs2.element0_bits)",
    }
}

pub fn vl_vtype_config_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vsetivli",
        obligation: RvvObligationKind::VlVtypeConfig,
        reads: &[],
        writes: CONFIG_WRITES,
        postcondition: "rd = vl = legal_vl(avl, vtype)",
    }
}

pub fn mask_memory_round_trip_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vlm.v/vsm.v",
        obligation: RvvObligationKind::MaskMemoryRoundTrip,
        reads: MASK_MEMORY_READS,
        writes: MASK_MEMORY_WRITES,
        postcondition: "packed mask bits loaded into v0 are stored back byte-equivalently",
    }
}

pub fn mask_merge_carry_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vmerge.vvm",
        obligation: RvvObligationKind::MaskMergeCarry,
        reads: MASK_MERGE_READS,
        writes: MASK_MERGE_WRITES,
        postcondition: "vd[i] = v0.mask[i] ? vs1[i] : vs2[i], including legal carrier aliasing",
    }
}

pub fn tail_high_transition_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vtype-transition",
        obligation: RvvObligationKind::TailHighTransition,
        reads: VECTOR_STATE_READS,
        writes: VECTOR_STATE_WRITES,
        postcondition: "active elements follow the new vtype while preserved/agnostic regions obey policy",
    }
}

pub fn vstart_fault_progress_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "faulting-vector-memory",
        obligation: RvvObligationKind::VstartFaultProgress,
        reads: MEMORY_VECTOR_READS,
        writes: VSTART_WRITES,
        postcondition: "on a precise vector memory fault, vstart names the first uncompleted element",
    }
}

pub fn whole_register_round_trip_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vl<nf>re<eew>.v/vs<nf>r.v",
        obligation: RvvObligationKind::WholeRegisterRoundTrip,
        reads: MEMORY_VECTOR_READS,
        writes: MEMORY_VECTOR_WRITES,
        postcondition: "whole-register memory transfer preserves contiguous register-group layout",
    }
}

pub fn strided_whole_register_store_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vlse<eew>.v/vs<nf>r.v",
        obligation: RvvObligationKind::StridedWholeRegisterStore,
        reads: MEMORY_VECTOR_READS,
        writes: MEMORY_VECTOR_WRITES,
        postcondition: "strided element loads populate a register group that whole-register stores preserve",
    }
}

pub fn widening_zero_extend_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vzext.vf*",
        obligation: RvvObligationKind::WideningZeroExtend,
        reads: VECTOR_STATE_READS,
        writes: VECTOR_STATE_WRITES,
        postcondition: "each destination element is zero_extend(source_element)",
    }
}

pub fn segment_memory_layout_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vlseg/vlsseg/vsoxseg",
        obligation: RvvObligationKind::SegmentMemoryLayout,
        reads: MEMORY_VECTOR_READS,
        writes: VECTOR_STATE_WRITES,
        postcondition: "segment field i maps to vector register vd+i with the specified EEW/stride/index",
    }
}

pub fn fault_only_first_segment_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vlseg<nf>e<eew>ff.v",
        obligation: RvvObligationKind::FaultOnlyFirstSegment,
        reads: MEMORY_VECTOR_READS,
        writes: VECTOR_STATE_WRITES,
        postcondition: "FOF segment load writes completed fields and sets vl to the completed element count",
    }
}

pub fn indexed_segment_fof_contract() -> RvvInstructionContract {
    RvvInstructionContract {
        name: "vsoxseg<nf>ei<eew>.v/vle<eew>ff.v",
        obligation: RvvObligationKind::IndexedSegmentFof,
        reads: MEMORY_VECTOR_READS,
        writes: MEMORY_VECTOR_WRITES,
        postcondition: "indexed segment stores and FOF reloads observe the same lane address mapping",
    }
}

fn obligation_contract(kind: RvvObligationKind) -> RvvInstructionContract {
    match kind {
        RvvObligationKind::VlVtypeConfig => vl_vtype_config_contract(),
        RvvObligationKind::ScalarExtractSignExt => scalar_extract_sign_ext_contract(),
        RvvObligationKind::FloatExtractNanBox => float_extract_nan_box_contract(),
        RvvObligationKind::MaskMemoryRoundTrip => mask_memory_round_trip_contract(),
        RvvObligationKind::MaskMergeCarry => mask_merge_carry_contract(),
        RvvObligationKind::TailHighTransition => tail_high_transition_contract(),
        RvvObligationKind::VstartFaultProgress => vstart_fault_progress_contract(),
        RvvObligationKind::WholeRegisterRoundTrip => whole_register_round_trip_contract(),
        RvvObligationKind::StridedWholeRegisterStore => strided_whole_register_store_contract(),
        RvvObligationKind::WideningZeroExtend => widening_zero_extend_contract(),
        RvvObligationKind::SegmentMemoryLayout => segment_memory_layout_contract(),
        RvvObligationKind::FaultOnlyFirstSegment => fault_only_first_segment_contract(),
        RvvObligationKind::IndexedSegmentFof => indexed_segment_fof_contract(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RvvOperandAssignment {
    mask: u8,
    scalar_extract_vs: u8,
    float_extract_vs: u8,
    mask_merge_vs1: u8,
    mask_merge_vs2: u8,
    mask_merge_vd: u8,
    tail_state_vd: u8,
    fault_progress_vd: u8,
    whole_register_base: u8,
    strided_whole_base: u8,
    widening_src: u8,
    widening_dst: u8,
    segment_masked_base: u8,
    segment_strided_base: u8,
    unit_fof_vd: u8,
    segment_fof_e8_base: u8,
    segment_fof_e32_base: u8,
    indexed_segment_base: u8,
    indexed_segment_index: u8,
    indexed_unit_fof_vd: u8,
}

fn ambient_group_base<R: rand::Rng + ?Sized>(rng: &mut R, width: u8) -> u8 {
    let max_base = 32u8.saturating_sub(width).max(1);
    let candidates = (1u8..=max_base).collect::<Vec<_>>();
    candidates[rng.random_range(0..candidates.len())]
}

fn contract_group_base<R: rand::Rng + ?Sized>(rng: &mut R, width: u8) -> u8 {
    let max_base = 32u8.saturating_sub(width).max(1);
    let candidates = (1u8..=max_base)
        .filter(|base| !matches!(width, 2 | 4 | 8) || base % width == 0)
        .collect::<Vec<_>>();
    candidates[rng.random_range(0..candidates.len())]
}

fn contract_distinct_reg<R: rand::Rng + ?Sized>(rng: &mut R, forbidden: &[u8]) -> u8 {
    let candidates = (1u8..32)
        .filter(|reg| !forbidden.contains(reg))
        .collect::<Vec<_>>();
    candidates[rng.random_range(0..candidates.len())]
}

fn ambient_rvv_obligation_operand_assignment(
    seed: u64,
    obligation: RvvObligationKind,
) -> RvvOperandAssignment {
    let mut rng = StdRng::seed_from_u64(seed ^ ((obligation as u64) << 33) ^ 0x4f50_4552_5656_0001);
    RvvOperandAssignment {
        mask: 0,
        scalar_extract_vs: ambient_group_base(&mut rng, 1),
        float_extract_vs: ambient_group_base(&mut rng, 1),
        mask_merge_vs1: ambient_group_base(&mut rng, 1),
        mask_merge_vs2: ambient_group_base(&mut rng, 1),
        mask_merge_vd: ambient_group_base(&mut rng, 1),
        tail_state_vd: ambient_group_base(&mut rng, 1),
        fault_progress_vd: ambient_group_base(&mut rng, 1),
        whole_register_base: ambient_group_base(&mut rng, 8),
        strided_whole_base: ambient_group_base(&mut rng, 8),
        widening_src: ambient_group_base(&mut rng, 1),
        widening_dst: ambient_group_base(&mut rng, 1),
        segment_masked_base: ambient_group_base(&mut rng, 2),
        segment_strided_base: ambient_group_base(&mut rng, 3),
        unit_fof_vd: ambient_group_base(&mut rng, 1),
        segment_fof_e8_base: ambient_group_base(&mut rng, 2),
        segment_fof_e32_base: ambient_group_base(&mut rng, 2),
        indexed_segment_base: ambient_group_base(&mut rng, 6),
        indexed_segment_index: ambient_group_base(&mut rng, 1),
        indexed_unit_fof_vd: ambient_group_base(&mut rng, 1),
    }
}

fn contract_rvv_obligation_operand_assignment(seed: u64) -> RvvOperandAssignment {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x434f_4e54_5256_5601);
    // The mask-merge obligation executes with LMUL=4.  Draw three distinct
    // four-register groups instead of independently drawing single registers;
    // v0..v3 remain available for the mask carrier.
    let mut merge_groups = crate::rvv_legal::legal_group_bases((1, 31), 4);
    let take_merge_group = |groups: &mut Vec<u8>, rng: &mut StdRng| {
        let index = rng.random_range(0..groups.len());
        groups.swap_remove(index)
    };
    let mask_merge_vs1 = take_merge_group(&mut merge_groups, &mut rng);
    let mask_merge_vs2 = take_merge_group(&mut merge_groups, &mut rng);
    let mask_merge_vd = take_merge_group(&mut merge_groups, &mut rng);
    let widening_src = contract_group_base(&mut rng, 1);
    let widening_dst = contract_distinct_reg(&mut rng, &[widening_src]);
    let indexed_segment_base = contract_group_base(&mut rng, 6);
    let indexed_segment_group =
        (indexed_segment_base..indexed_segment_base + 6).collect::<Vec<_>>();
    let indexed_segment_index = contract_distinct_reg(&mut rng, &indexed_segment_group);

    RvvOperandAssignment {
        mask: 0,
        scalar_extract_vs: contract_group_base(&mut rng, 1),
        float_extract_vs: contract_group_base(&mut rng, 1),
        mask_merge_vs1,
        mask_merge_vs2,
        mask_merge_vd,
        tail_state_vd: contract_group_base(&mut rng, 1),
        fault_progress_vd: contract_group_base(&mut rng, 1),
        whole_register_base: contract_group_base(&mut rng, 8),
        // The strided/whole-register probe initializes eight consecutive
        // vector registers before its vs4r observation. Reserve all eight.
        strided_whole_base: contract_group_base(&mut rng, 8),
        widening_src,
        widening_dst,
        segment_masked_base: contract_group_base(&mut rng, 2),
        segment_strided_base: contract_group_base(&mut rng, 3),
        unit_fof_vd: contract_group_base(&mut rng, 1),
        segment_fof_e8_base: contract_group_base(&mut rng, 2),
        segment_fof_e32_base: contract_group_base(&mut rng, 2),
        indexed_segment_base,
        indexed_segment_index,
        indexed_unit_fof_vd: contract_group_base(&mut rng, 1),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HardwareRvvModelBlockKind {
    VlVtypeConfig,
    ScalarExtractSignExt,
    FloatExtractNanBox,
    MaskMemoryRoundTrip,
    MaskMergeCarry,
    TailHighTransition,
    VstartFaultProgress,
    WholeRegisterRoundTrip,
    StridedWholeRegisterStore,
    WideningZeroExtend,
    AraWideningZeroExtendProbe,
    SegmentMaskedLoadLayout,
    SegmentStridedLoadLayout,
    FaultOnlyFirstLoad,
    FaultOnlyFirstSegmentE8,
    FaultOnlyFirstSegmentE32,
    PmpSegmentFof,
    MemoryEdgeUnitFof,
    MemoryEdgeSegmentFof,
    IndexedSegmentFof,
    AraMaskedIndexedMisalignedNoTrap,
    AraIndexedMisalignedTrapCause,
    AraMaskedStoreVstart,
    AraMaskedVslideupLargeOffset,
    AraMaskedDestV0Overlap,
    AraIndexedSegmentLaterField,
    AraStridedSegmentLaterField,
    AraMaskedSegmentLane5,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HardwareRvvModelBlock {
    pub kind: HardwareRvvModelBlockKind,
    pub contract: RvvInstructionContract,
    pub instructions: Vec<InstructionSequence<RiscvInstruction>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HardwareRvvInjectionReport {
    pub inserted_blocks: usize,
    pub inserted_instructions: usize,
    pub block_kinds: Vec<HardwareRvvModelBlockKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareRvvInjectionFlavor {
    /// Keep only the obligation instruction skeleton. RVV configuration and
    /// vector operands are randomized without enforcing the semantic constraints
    /// that make the contract reliably executable.
    AmbientRandom,
    /// Preserve each obligation block contract-specific RVV configuration and
    /// legal operand shape. This is the constraints+contract flavor.
    ContractConstrained,
    /// Preserve the same RVV obligation opcodes, but give ordinary layout
    /// probes aligned in-bounds operands. Fault-oriented blocks stay explicit.
    ContractPositive,
}

fn obligation_kind_for_block(kind: HardwareRvvModelBlockKind) -> RvvObligationKind {
    match kind {
        HardwareRvvModelBlockKind::VlVtypeConfig => RvvObligationKind::VlVtypeConfig,
        HardwareRvvModelBlockKind::ScalarExtractSignExt => RvvObligationKind::ScalarExtractSignExt,
        HardwareRvvModelBlockKind::FloatExtractNanBox => RvvObligationKind::FloatExtractNanBox,
        HardwareRvvModelBlockKind::MaskMemoryRoundTrip => RvvObligationKind::MaskMemoryRoundTrip,
        HardwareRvvModelBlockKind::MaskMergeCarry => RvvObligationKind::MaskMergeCarry,
        HardwareRvvModelBlockKind::TailHighTransition => RvvObligationKind::TailHighTransition,
        HardwareRvvModelBlockKind::VstartFaultProgress => RvvObligationKind::VstartFaultProgress,
        HardwareRvvModelBlockKind::WholeRegisterRoundTrip => {
            RvvObligationKind::WholeRegisterRoundTrip
        }
        HardwareRvvModelBlockKind::StridedWholeRegisterStore => {
            RvvObligationKind::StridedWholeRegisterStore
        }
        HardwareRvvModelBlockKind::WideningZeroExtend
        | HardwareRvvModelBlockKind::AraWideningZeroExtendProbe => {
            RvvObligationKind::WideningZeroExtend
        }
        HardwareRvvModelBlockKind::SegmentMaskedLoadLayout
        | HardwareRvvModelBlockKind::SegmentStridedLoadLayout => {
            RvvObligationKind::SegmentMemoryLayout
        }
        HardwareRvvModelBlockKind::FaultOnlyFirstLoad
        | HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE8
        | HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE32
        | HardwareRvvModelBlockKind::PmpSegmentFof
        | HardwareRvvModelBlockKind::MemoryEdgeUnitFof
        | HardwareRvvModelBlockKind::MemoryEdgeSegmentFof => {
            RvvObligationKind::FaultOnlyFirstSegment
        }
        HardwareRvvModelBlockKind::IndexedSegmentFof => RvvObligationKind::IndexedSegmentFof,
        HardwareRvvModelBlockKind::AraMaskedIndexedMisalignedNoTrap
        | HardwareRvvModelBlockKind::AraIndexedMisalignedTrapCause
        | HardwareRvvModelBlockKind::AraIndexedSegmentLaterField
        | HardwareRvvModelBlockKind::AraStridedSegmentLaterField
        | HardwareRvvModelBlockKind::AraMaskedSegmentLane5 => {
            RvvObligationKind::SegmentMemoryLayout
        }
        HardwareRvvModelBlockKind::AraMaskedStoreVstart => RvvObligationKind::VstartFaultProgress,
        HardwareRvvModelBlockKind::AraMaskedVslideupLargeOffset
        | HardwareRvvModelBlockKind::AraMaskedDestV0Overlap => RvvObligationKind::MaskMergeCarry,
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
            message: format!("failed to decode hardware-rvv instruction `{asm}`: {source}"),
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

fn aligned_model_address(start: u64, end: u64) -> Result<u64, GenerationError> {
    const MODEL_BYTES: u64 = 0x1c0;
    let aligned_start = start.div_ceil(8).saturating_mul(8);
    let preferred = aligned_start.saturating_add(0x20);
    let addr = if preferred.saturating_add(MODEL_BYTES) <= end {
        preferred
    } else if aligned_start.saturating_add(MODEL_BYTES) <= end {
        aligned_start
    } else {
        return Err(GenerationError::StructuredInstructionBuildFailed {
            message: format!(
                "hardware-rvv needs at least {MODEL_BYTES} writable bytes in mem range {start:#x}..{end:#x}"
            ),
        });
    };
    Ok(addr)
}

fn has_rv64_extension(extensions: &IsaExtensions, extension: RV64Extensions) -> bool {
    extensions
        .as_rv64()
        .map(|values| values.contains(&extension))
        .unwrap_or(false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HardwareRvvAddressMode {
    /// Preserve the historical direct-runner probe address used by the generic
    /// XiangShan-oriented RVV obligation generators.
    LegacyHighMemoryProbe,
    /// Keep all model scratch/probe addresses inside the generated testcase
    /// memory range, except explicitly sampled XiangShan high-DRAM windows.
    MemRangeLocal,
    /// Ara has a different accessible memory map. Keep every sampled probe
    /// inside the testcase-owned range rather than using XiangShan addresses.
    AraLocal,
}

const DIRECT_RUNNER_DRAM_END: u64 = 0x9000_0000;
const HIGH_MEMORY_PROBE_OFFSET: u64 = 0x160;

fn high_memory_probe_base(base_addr: u64, address_mode: HardwareRvvAddressMode) -> u64 {
    match address_mode {
        HardwareRvvAddressMode::LegacyHighMemoryProbe => DIRECT_RUNNER_DRAM_END - 0x48,
        HardwareRvvAddressMode::MemRangeLocal | HardwareRvvAddressMode::AraLocal => {
            base_addr + HIGH_MEMORY_PROBE_OFFSET
        }
    }
}

fn signed_literal(value: u64) -> i64 {
    i64::from_ne_bytes(value.to_ne_bytes())
}

// Keep the target's old destination state varied without reviving the 19-vreg
// ambient initialization.  This only changes memory-load destinations, which
// cannot be an index or arithmetic source for the target instruction.
fn sparse_load_context_enabled() -> bool {
    std::env::var("RISCV_FUZZ_RVV_SPARSE_CONTEXT")
        .is_ok_and(|value| matches!(value.trim(), "1" | "true" | "yes"))
}

fn push_sparse_load_destination_context(
    lines: &mut Vec<String>,
    vd: u8,
    seed: u64,
    variant: u64,
    domain: u64,
    enabled: bool,
) {
    if enabled && variant % 2 == 1 {
        let mut rng = StdRng::seed_from_u64(seed ^ variant.rotate_left(17) ^ domain);
        // A signed 12-bit value keeps `li` to one machine instruction while
        // still varying every byte pattern relevant to e8 load destinations.
        let value: i16 = rng.random_range(-2048..=2047);
        lines.push(format!("li x31, {value}"));
        lines.push(format!("vmv.v.x v{vd}, x31"));
    } else {
        lines.push(format!("vmv.v.i v{vd}, 0"));
    }
}

fn push_data_words(lines: &mut Vec<String>, base_reg: &str, start_offset: i32, words: &[u64]) {
    for (idx, value) in words.iter().copied().enumerate() {
        lines.push(format!("li x12, {}", signed_literal(value)));
        lines.push(format!(
            "sd x12, {}({base_reg})",
            start_offset + (idx as i32 * 8)
        ));
    }
}

fn push_vector_splat(lines: &mut Vec<String>, vector_reg: u8, value: u64) {
    lines.push(format!("li x8, {}", signed_literal(value)));
    lines.push(format!("vmv.v.x v{vector_reg}, x8"));
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

fn template_cfg(
    _isa_base: ISABase,
    sew: u16,
    vl: u8,
    lmul: &'static str,
    tail_policy: &'static str,
    mask_policy: &'static str,
) -> TemplateRvvConfig {
    TemplateRvvConfig {
        sew,
        vl,
        lmul,
        tail_policy,
        mask_policy,
    }
}

fn vsetivli_line(rd: &str, cfg: TemplateRvvConfig) -> String {
    format!("vsetivli {rd}, {}, {}", cfg.vl, cfg.vtype_immediate())
}

fn ambient_random_vtype_immediate<R: rand::Rng>(rng: &mut R) -> u16 {
    // Random vtype bits intentionally do not restrict vsew/lmul/policy to a
    // legal contract. They only stay inside the immediate field accepted by the
    // assembler/decoder.
    rng.random_range(0..=0xff)
}

fn ambient_random_avl<R: rand::Rng>(rng: &mut R) -> u8 {
    rng.random_range(0..=31)
}

fn vset_destination(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    let opcode_end = trimmed.find(char::is_whitespace)?;
    let opcode = &trimmed[..opcode_end].to_ascii_lowercase();
    if !matches!(opcode.as_str(), "vsetivli" | "vsetvli" | "vsetvl") {
        return None;
    }
    let rd = trimmed[opcode_end..].trim().split(',').next()?.trim();
    (!rd.is_empty()).then_some(rd)
}

fn ambient_randomize_vset_line<R: rand::Rng>(line: &str, rng: &mut R) -> Option<String> {
    let rd = vset_destination(line)?;
    Some(format!(
        "vsetivli {rd}, {}, {}",
        ambient_random_avl(rng),
        ambient_random_vtype_immediate(rng)
    ))
}

fn asm_parts(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim();
    let opcode_end = trimmed.find(char::is_whitespace)?;
    Some((&trimmed[..opcode_end], trimmed[opcode_end..].trim()))
}

fn is_rvv_operand_instruction(mnemonic: &str) -> bool {
    let mnemonic = mnemonic.to_ascii_lowercase();
    mnemonic.starts_with('v') && !matches!(mnemonic.as_str(), "vsetivli" | "vsetvli" | "vsetvl")
}

fn vector_operand_kind(token: &str) -> Option<bool> {
    let token = token.trim();
    let is_mask = token.ends_with(".t");
    let vector = token.strip_suffix(".t").unwrap_or(token);
    let number = vector.strip_prefix('v')?.parse::<u8>().ok()?;
    (number < 32).then_some(is_mask)
}

fn ambient_randomize_rvv_operands_line<R: rand::Rng>(line: &str, rng: &mut R) -> Option<String> {
    let (mnemonic, operands) = asm_parts(line)?;
    if !is_rvv_operand_instruction(mnemonic) {
        return None;
    }

    let mnemonic_lower = mnemonic.to_ascii_lowercase();
    let operand_tokens = operands.split(',').collect::<Vec<_>>();
    let operand_count = operand_tokens.len();
    let mut rewritten = false;
    let operands = operand_tokens
        .iter()
        .enumerate()
        .map(|(idx, operand)| {
            let operand = operand.trim();
            let is_mask_memory_carrier =
                matches!(mnemonic_lower.as_str(), "vlm.v" | "vsm.v") && idx == 0;
            if is_mask_memory_carrier {
                rewritten |= operand != "v0";
                return "v0".to_string();
            }
            let is_fixed_mask =
                operand == "v0" && idx + 1 == operand_count && mnemonic_lower.ends_with(".vvm");
            if is_fixed_mask {
                rewritten = true;
                return "v0".to_string();
            }
            match vector_operand_kind(operand) {
                Some(true) => {
                    rewritten = true;
                    "v0.t".to_string()
                }
                Some(false) => {
                    rewritten = true;
                    format!("v{}", rng.random_range(0..32))
                }
                None => operand.to_string(),
            }
        })
        .collect::<Vec<_>>();

    rewritten.then(|| format!("{mnemonic} {}", operands.join(", ")))
}

fn ambient_randomize_block(
    block: HardwareRvvModelBlock,
    isa_base: ISABase,
    seed: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let mut rng = StdRng::seed_from_u64(seed ^ ((block.kind as u64) << 32) ^ 0x414d_4249_5247_5656);
    let mut rewritten = 0usize;
    let lines = block
        .instructions
        .iter()
        .map(|sequence| {
            let line = sequence.to_string();
            if let Some(randomized) = ambient_randomize_vset_line(&line, &mut rng) {
                rewritten += 1;
                randomized
            } else if let Some(randomized) = ambient_randomize_rvv_operands_line(&line, &mut rng) {
                rewritten += 1;
                randomized
            } else {
                line
            }
        })
        .collect::<Vec<_>>();
    if rewritten == 0 {
        return Ok(block);
    }
    Ok(HardwareRvvModelBlock {
        instructions: decode_block(&lines, isa_base)?,
        ..block
    })
}

fn vl_vtype_config_model_block(
    isa_base: ISABase,
    base_addr: u64,
    variant_index: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let variant = variant_index % 8;
    let (sew, avl, lmul, tail, mask) = match variant {
        0 => (8, 16, "m1", "ta", "ma"),
        1 => (8, 16, "m1", "tu", "mu"),
        2 => (8, 31, "m2", "ta", "ma"),
        3 => (16, 16, "m1", "ta", "mu"),
        4 => (32, 7, "m1", "tu", "ma"),
        5 => (64, 8, "m2", "ta", "ma"),
        6 => (8, 23, "m4", "tu", "mu"),
        _ => (8, 1, "mf2", "ta", "mu"),
    };
    let config = template_cfg(isa_base, sew, avl, lmul, tail, mask);
    let lines = vec![
        format!("li x11, {base_addr}"),
        format!("li x10, {avl}"),
        format!("vsetvli x6, x10, {}", config.vtype_immediate()),
        "csrr x8, vl".to_string(),
        format!("vsetvli x7, x10, {}", config.vtype_immediate()),
        "csrr x9, vl".to_string(),
        "sd x6, 0(x11)".to_string(),
        "sd x7, 8(x11)".to_string(),
        "sd x8, 16(x11)".to_string(),
        "sd x9, 24(x11)".to_string(),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::VlVtypeConfig,
        contract: obligation_contract(RvvObligationKind::VlVtypeConfig),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn sign_ext_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let cfg = template_cfg(isa_base, 32, 1, "m1", "tu", "mu");
    let vtype = cfg.vtype_immediate();
    let poison_cfg = template_cfg(isa_base, 64, 1, "m1", "tu", "mu");
    let poison_vtype = poison_cfg.vtype_immediate();
    let value = i32::from_ne_bytes(0xe01d7b92u32.to_ne_bytes()) as i64;
    let poison = i64::from_ne_bytes(0xaaaa_aaaa_aaaa_aaaau64.to_ne_bytes());
    let vs = operands.scalar_extract_vs;
    let lines = vec![
        format!("li x10, {base_addr}"),
        format!("li x13, {poison}"),
        format!("vsetivli x0, {}, {poison_vtype}", poison_cfg.vl),
        format!("vmv.v.x v{vs}, x13"),
        format!("li x12, {value}"),
        "sw x12, 0(x10)".to_string(),
        format!("vsetivli x0, {}, {vtype}", cfg.vl),
        format!("vle32.v v{vs}, (x10)"),
        format!("vmv.x.s x15, v{vs}"),
        "sd x15, 8(x10)".to_string(),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::ScalarExtractSignExt,
        contract: obligation_contract(RvvObligationKind::ScalarExtractSignExt),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn float_nan_box_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let cfg = template_cfg(isa_base, 32, 1, "m1", "tu", "mu");
    let vtype = cfg.vtype_immediate();
    let vs = operands.float_extract_vs;
    let lines = vec![
        format!("li x10, {base_addr}"),
        "li x12, 1431655765".to_string(), // 0x55555555
        "sw x12, 0(x10)".to_string(),
        format!("vsetivli x0, {}, {vtype}", cfg.vl),
        format!("vle32.v v{vs}, (x10)"),
        format!("vfmv.f.s f23, v{vs}"),
        "fmv.x.d x15, f23".to_string(),
        "sd x15, 16(x10)".to_string(),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::FloatExtractNanBox,
        contract: obligation_contract(RvvObligationKind::FloatExtractNanBox),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn mask_memory_round_trip_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
    variant_index: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let variant = (variant_index % 4) as usize;
    let clear_vl = [16, 24, 31, 16][variant];
    let load_vl = [10, 7, 15, 23][variant];
    let mask_bits = [0x380u64, 0x45, 0x4215, 0x421081][variant];
    let live_lane = [7, 6, 14, 22][variant];
    // On a VLEN=128 subject, e8/m1 caps VL at 16. Use m2 for longer variants
    // so the chosen observed lane is genuinely active rather than past vl.
    let clear_lmul = if clear_vl > 16 { "m2" } else { "m1" };
    let load_lmul = if load_vl > 16 { "m2" } else { "m1" };
    let clear_config = template_cfg(isa_base, 8, clear_vl, clear_lmul, "tu", "mu");
    let load_config = template_cfg(isa_base, 8, load_vl, load_lmul, "tu", "mu");
    let mask = operands.mask;
    let data = if load_lmul == "m2" {
        (operands.scalar_extract_vs & !1).max(2)
    } else {
        operands.scalar_extract_vs
    };
    let lines = vec![
        format!("li x11, {base_addr}"),
        format!("li x12, {mask_bits}"),
        "sd x12, 0(x11)".to_string(),
        vsetivli_line("x0", clear_config),
        format!("vmv.v.i v{mask}, 0"),
        vsetivli_line("x0", load_config),
        format!("vlm.v v{mask}, (x11)"),
        "addi x13, x11, 16".to_string(),
        format!("vsm.v v{mask}, (x13)"),
        "ld x15, 16(x11)".to_string(),
        format!("vmv.v.i v{data}, 0"),
        format!("vadd.vi v{data}, v{data}, 1, v{mask}.t"),
        "addi x14, x11, 32".to_string(),
        format!("vse8.v v{data}, (x14)"),
        format!("lbu x16, {live_lane}(x14)"),
        "lbu x17, 1(x14)".to_string(),
        "sd x16, 64(x11)".to_string(),
        "sd x17, 72(x11)".to_string(),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::MaskMemoryRoundTrip,
        contract: obligation_contract(RvvObligationKind::MaskMemoryRoundTrip),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn mask_merge_carry_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let config = template_cfg(isa_base, 16, 8, "m4", "tu", "mu");
    let mask = operands.mask;
    let vs1 = operands.mask_merge_vs1;
    let vs2 = operands.mask_merge_vs2;
    let vd = operands.mask_merge_vd;
    let lines = vec![
        format!("li x11, {base_addr}"),
        vsetivli_line("x0", config),
        format!("vmv.v.i v{vs1}, 1"),
        format!("vmv.v.i v{vs2}, 0"),
        format!("vmv.v.i v{mask}, -1"),
        format!("vmerge.vvm v{vd}, v{vs2}, v{vs1}, v{mask}"),
        format!("vsm.v v{vd}, (x11)"),
        "ld x15, 0(x11)".to_string(),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::MaskMergeCarry,
        contract: obligation_contract(RvvObligationKind::MaskMergeCarry),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn tail_high_transition_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let e16_tama = template_cfg(isa_base, 16, 4, "m1", "ta", "ma");
    let e8_tama = template_cfg(isa_base, 8, 4, "m1", "ta", "ma");
    let vd = operands.tail_state_vd;
    let lines = vec![
        format!("li x11, {base_addr}"),
        vsetivli_line("x0", e16_tama),
        format!("vmv.v.i v{vd}, -1"),
        format!("vsrl.vi v{vd}, v{vd}, 8"),
        vsetivli_line("x0", e8_tama),
        format!("li x12, {}", e16_tama.vtype_immediate()),
        // rd=rs1=x0 is reserved when the SEW/LMUL transition changes VLMAX.
        // An explicit AVL retains the intended four active elements legally.
        "li x10, 4".to_string(),
        "vsetvl x0, x10, x12".to_string(),
        format!("vadd.vi v{vd}, v{vd}, 1"),
        format!("vs1r.v v{vd}, 0(x11)"),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::TailHighTransition,
        contract: obligation_contract(RvvObligationKind::TailHighTransition),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn vstart_fault_progress_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let config = template_cfg(isa_base, 64, 4, "m1", "tu", "mu");
    let vd = operands.fault_progress_vd;
    let lines = vec![
        format!("li x11, {base_addr}"),
        "li x12, -1048576".to_string(),
        vsetivli_line("x0", config),
        format!("vmv.v.i v{vd}, 0"),
        format!("vsse16.v v{vd}, (x11), x12"),
        "addi x15, x0, 0".to_string(),
        "sd x15, 40(x11)".to_string(),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::VstartFaultProgress,
        contract: obligation_contract(RvvObligationKind::VstartFaultProgress),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn whole_register_round_trip_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
    address_mode: HardwareRvvAddressMode,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let probe_addr = high_memory_probe_base(base_addr, address_mode);
    let base = operands.whole_register_base;
    let mut lines = vec![format!("li x11, {probe_addr}")];
    let words = [
        0x0000_0000_0001_0000,
        0x741d_51bb_851a_03db,
        0x0000_0000_0000_0000,
        0xfcfe_328c_5459_0753,
        0x0000_0000_0000_0002,
        0x5653_98a2_8c45_70dc,
        0x0000_0000_0000_00ff,
        0x638a_4d4a_34d0_3fc9,
    ];
    push_data_words(&mut lines, "x11", 0, &words);
    let e64_m1 = template_cfg(isa_base, 64, 1, "m1", "tu", "mu");
    let e8_m8 = template_cfg(isa_base, 8, 10, "m8", "tu", "mu");
    lines.push(format!("vsetvli x9, x0, {}", e64_m1.vtype_immediate()));
    for (offset, value) in [
        0x5555_5555_5555_5555,
        0x21d4_76b6_1b5a_c963,
        0x0f0f_0f0f_0f0f_0f0f,
        0xffff_ffff_ffff_ffff,
        0xffff_ffff_ffff_ffff,
        0x0f0f_0f0f_0f0f_0f0f,
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0000,
    ]
    .into_iter()
    .enumerate()
    {
        push_vector_splat(&mut lines, base + offset as u8, value);
    }
    lines.extend([
        "li x8, 0".to_string(),
        "li x9, 0".to_string(),
        "li x10, 0".to_string(),
        "li x11, 0".to_string(),
        "li x12, 0".to_string(),
        format!("li x2, {}", probe_addr + 64),
        "c.sdsp x11, 16(sp)".to_string(),
        vsetivli_line("x8", e8_m8),
        format!("li x10, {probe_addr}"),
        format!("vl8re64.v v{base}, (x10)"),
        format!("li x11, {}", base_addr + 160),
        format!("vs8r.v v{base}, 0(x11)"),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::WholeRegisterRoundTrip,
        contract: obligation_contract(RvvObligationKind::WholeRegisterRoundTrip),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn strided_whole_register_store_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
    address_mode: HardwareRvvAddressMode,
    positive: bool,
    seed: u64,
    variant_index: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let variant = variant_index;
    let diversified = positive && address_mode != HardwareRvvAddressMode::LegacyHighMemoryProbe;
    let probe_addr = if diversified {
        if address_mode == HardwareRvvAddressMode::AraLocal {
            [
                base_addr + 0x160,
                base_addr + 0x100,
                base_addr + 0x140,
                base_addr + 0x120,
            ][((variant / 16) % 4) as usize]
        } else {
            match (variant / 16) % 4 {
                0 => 0x8fff_ffb8,
                1 => high_memory_probe_base(base_addr, address_mode),
                2 => 0x8fff_ffa8,
                _ => 0x8fff_ffb0,
            }
        }
    } else {
        high_memory_probe_base(base_addr, address_mode)
    };
    // Keep the whole-register observer aligned for every sampled target
    // vtype. The first high window includes v24 and the byte-lane boundary;
    // later windows retain randomized data and operands.
    let base = if diversified {
        [24u8, 8, 16, 0][(variant % 4) as usize]
    } else {
        operands.strided_whole_base
    };
    let mut lines = vec![format!("li x9, {probe_addr}")];
    let mut words = [
        0x0000_0000_0000_1000,
        0x0000_0000_0000_007f,
        0x4792_b1db_7b17_d676,
        0x0000_0000_0000_0000,
        0xed8a_db74_c7d1_dae3,
        0xb6c1_377e_e61d_f71f,
        0x988a_9d98_c177_c9df,
        0x0000_0000_0000_00ff,
    ];
    if diversified {
        let mut rng = StdRng::seed_from_u64(seed ^ 0x5354_5249_4445_4444);
        for word in &mut words {
            *word = rng.random();
        }
        match (variant / 64) % 4 {
            0 => {
                words[0] = 0x1000;
                words[1] = 0x7f;
                words[2] = 0x4792_b1db_7b17_d676;
            }
            1 => {
                let low = rng.random::<u32>();
                let high = 0xff00_0000 | (rng.random::<u32>() & 0x00ff_ffff);
                words[2] = (u64::from(high) << 32) | u64::from(low);
            }
            2 => {
                let low = rng.random::<u32>();
                let high = 0x8000_0000 | (rng.random::<u32>() & 0x00ff_ffff);
                words[2] = (u64::from(high) << 32) | u64::from(low);
            }
            _ => {}
        }
    }
    push_data_words(&mut lines, "x9", 0, &words);
    let e64_m1 = template_cfg(isa_base, 64, 1, "m1", "tu", "mu");
    // For EEW=32, each pair has EMUL=(32/SEW)*LMUL=4. Rotate this
    // equivalent legal domain independently of the register group, alias
    // write and high-address class instead of fixing the probe at e64,m8.
    let target_sew = if diversified {
        [64u16, 32, 16, 8][((variant % 4 + variant / 4 + variant / 16) % 4) as usize]
    } else {
        64
    };
    let target_lmul = match target_sew / 8 {
        1 => "m1",
        2 => "m2",
        4 => "m4",
        8 => "m8",
        _ => unreachable!("sampled strided config must have EMUL=4"),
    };
    let target_cfg = template_cfg(isa_base, target_sew, 16, target_lmul, "tu", "mu");
    lines.push(format!("vsetvli x9, x0, {}", e64_m1.vtype_immediate()));
    for (offset, value) in [
        0x5555_5555_5555_5555,
        0x88c9_9f10_7473_cef1,
        0x7d52_ea60_babd_5063,
        0xf8fe_ed0c_9722_535a,
        0x5555_5555_5555_5555,
        0xffff_ffff_ffff_ffff,
        0xaaaa_aaaa_aaaa_aaaa,
        0x45fd_5d50_3ef5_4b77,
    ]
    .into_iter()
    .enumerate()
    {
        push_vector_splat(&mut lines, base + offset as u8, value);
    }
    // Cycle the preceding scalar store independently of the vector group and
    // address.  In particular, v24 + the high-memory byte window can now be
    // paired with both aligned and misaligned same-line stores.
    let alias_offset = if diversified {
        [64u64, 8, 7, 16][((variant / 4) % 4) as usize]
    } else {
        64
    };
    lines.extend([
        "li x8, 0".to_string(),
        "li x9, 0".to_string(),
        "li x10, 0".to_string(),
        "li x11, 0".to_string(),
        "li x12, 0".to_string(),
        "li x13, 32767".to_string(),
        format!("li x2, {}", probe_addr + alias_offset),
        "c.sdsp x13, 0(sp)".to_string(),
    ]);
    // A bounded predecessor consumes the same initialized memory and writes
    // the destination group before the mixed-EEW strided load. This exposes
    // unit, segment and strided-segment producer paths without relying on a
    // long random prefix or changing the target's legal EMUL=4 domain.
    if diversified {
        let producer = ((variant / 4 + variant / 16) % 4) as usize;
        if producer != 0 {
            let e64_m1_two = template_cfg(isa_base, 64, 2, "m1", "tu", "mu");
            lines.push(vsetivli_line("x0", e64_m1_two));
            lines.push(format!("li x11, {probe_addr}"));
            match producer {
                1 => {
                    lines.push("li x13, 24".to_string());
                    lines.push(format!("vlsseg3e64.v v{base}, (x11), x13"));
                }
                2 => lines.push(format!("vle64.v v{base}, (x11)")),
                3 => lines.push(format!("vlseg3e64.v v{base}, (x11)")),
                _ => unreachable!(),
            }
        }
        // Make a real memory producer for the strided load, in addition to
        // register-group predecessors. A coprime period keeps this edge from
        // being locked to one SEW/LMUL, address window, or alias class.
        if variant % 5 == 4 {
            let lane = [5u64, 0, 7, 15][((variant / 5) % 4) as usize];
            let mut rng =
                StdRng::seed_from_u64(seed ^ variant.rotate_left(11) ^ 0x5354_4f52_455f_3332);
            let high_byte = match (variant / 5) % 4 {
                0 => rng.random_range(1u32..=0x7e),
                1 => 0x7f,
                2 => 0x80,
                _ => 0xff,
            };
            let value = (high_byte << 24) | (rng.random::<u32>() & 0x00ff_ffff);
            // v4 is outside every sampled four-register target group. Fill
            // its full register first so the one-lane store has defined tail.
            lines.extend([
                "li x7, -1".to_string(),
                format!(
                    "vsetvli x0, x7, {}",
                    template_cfg(isa_base, 32, 1, "m1", "tu", "mu").vtype_immediate()
                ),
                "vmv.v.i v4, 0".to_string(),
                vsetivli_line("x0", template_cfg(isa_base, 32, 1, "m1", "tu", "mu")),
                format!("li x13, {value}"),
                "vmv.v.x v4, x13".to_string(),
                format!("li x11, {}", probe_addr + 4 * lane),
                "vse32.v v4, (x11)".to_string(),
            ]);
        }
    }
    lines.extend([
        // The same EEW/EMUL choice can have a much larger VLMAX on Ara than
        // on XiangShan. Bound the actual strided access to 16 elements.
        "li x7, 16".to_string(),
        format!("vsetvli x8, x7, {}", target_cfg.vtype_immediate()),
        format!("li x9, {probe_addr}"),
        "li x13, 4".to_string(),
        format!("vlse32.v v{base}, (x9), x13"),
        format!("li x12, {}", base_addr + 192),
        format!("vs4r.v v{base}, 0(x12)"),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::StridedWholeRegisterStore,
        contract: obligation_contract(RvvObligationKind::StridedWholeRegisterStore),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn widening_zero_extend_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let narrow = template_cfg(isa_base, 16, 1, "mf4", "tu", "mu");
    let wide = template_cfg(isa_base, 64, 27, "m1", "tu", "mu");
    let src = operands.widening_src;
    let dst = operands.widening_dst;
    let lines = vec![
        format!("li x11, {base_addr}"),
        "li x12, 41039".to_string(), // 0xa04f
        vsetivli_line("x0", narrow),
        format!("vmv.s.x v{src}, x12"),
        vsetivli_line("x0", wide),
        format!("vzext.vf8 v{dst}, v{src}"),
        format!("vse64.v v{dst}, (x11)"),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::WideningZeroExtend,
        contract: obligation_contract(RvvObligationKind::WideningZeroExtend),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn ara_widening_zero_extend_probe_model_block(
    isa_base: ISABase,
    base_addr: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let out_addr = base_addr + 0x20;
    let narrow = template_cfg(isa_base, 16, 1, "mf4", "tu", "mu");
    let wide = template_cfg(isa_base, 64, 27, "m1", "tu", "mu");
    let lines = vec![
        format!("li x11, {out_addr}"),
        "li x12, 41039".to_string(),
        vsetivli_line("x0", narrow),
        "vmv.s.x v20, x12".to_string(),
        vsetivli_line("x0", wide),
        "vzext.vf8 v12, v20".to_string(),
        "vmv.x.s x20, v12".to_string(),
        "vslidedown.vi v15, v12, 1".to_string(),
        "vmv.x.s x21, v15".to_string(),
        "sd x20, 0(x11)".to_string(),
        "sd x21, 8(x11)".to_string(),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::AraWideningZeroExtendProbe,
        contract: obligation_contract(RvvObligationKind::WideningZeroExtend),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn segment_masked_load_layout_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
    address_mode: HardwareRvvAddressMode,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let probe_addr = high_memory_probe_base(base_addr, address_mode);
    let e64_m1 = template_cfg(isa_base, 64, 1, "m1", "tu", "mu");
    let base = operands.segment_masked_base;
    let mask = operands.mask;
    let mut lines = vec![
        "li x13, 0".to_string(),
        format!("li x16, {}", signed_literal(0x35b8_c53e_3930_a213)),
        format!("vsetvli x9, x0, {}", e64_m1.vtype_immediate()),
    ];
    push_vector_splat(&mut lines, mask, 0xffff_ffff_ffff_ffff);
    push_vector_splat(&mut lines, base + 1, 0xffff_ffff_ffff_ffff);
    lines.extend([
        format!("li x9, {probe_addr}"),
        format!("li x12, {}", signed_literal(0x9ff5_97b5_4a2e_1009)),
        "sd x12, 16(x9)".to_string(),
        "li x14, 0".to_string(),
        format!("li x21, {}", probe_addr + 8),
        "sd x14, 16(x21)".to_string(),
        "sub x20, x13, x16".to_string(),
        format!("li x21, {}", probe_addr + 40),
        "sd x20, 0(x21)".to_string(),
        format!("li x9, {}", probe_addr + 18),
        format!("vlseg2e16.v v{base}, (x9), v{mask}.t"),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::SegmentMaskedLoadLayout,
        contract: obligation_contract(RvvObligationKind::SegmentMemoryLayout),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn segment_strided_load_layout_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
    address_mode: HardwareRvvAddressMode,
    positive: bool,
    seed: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let probe_addr = high_memory_probe_base(base_addr, address_mode);
    let e64_m1 = template_cfg(isa_base, 64, 1, "m1", "tu", "mu");
    let e32_m1 = template_cfg(isa_base, 32, 1, "m1", "tu", "mu");
    let base = operands.segment_strided_base;
    let mut lines = vec![
        format!("li x13, {}", signed_literal(0xfe7b_d404_6a92_3944)),
        "li x18, 0".to_string(),
        format!("vsetvli x9, x0, {}", e64_m1.vtype_immediate()),
    ];
    push_vector_splat(&mut lines, 17, 0x75b0_34a7_97af_f2dc);
    push_vector_splat(&mut lines, 18, 0xffff_ffff_ffff_ffff);
    lines.extend([
        format!("li x9, {probe_addr}"),
        format!("li x12, {}", signed_literal(0x3803_444b_2551_a2ae)),
        "sd x12, 16(x9)".to_string(),
        "li x12, 4096".to_string(),
        "sd x12, 24(x9)".to_string(),
        // Keep the FP narrowing opcode, but use FP64 -> FP32 rather than an
        // unsupported FP128 -> FP64 conversion.  Restore the memory probe's
        // original e64/m1 configuration before its segment load.
        vsetivli_line("x0", e32_m1),
        "vfncvt.f.f.w v13, v18".to_string(),
        format!("vsetvli x9, x0, {}", e64_m1.vtype_immediate()),
        "remuw x19, x18, x13".to_string(),
        "vmv.x.s x15, v17".to_string(),
        format!("li x12, {}", probe_addr + 8),
        "sw x13, 0(x12)".to_string(),
        "fence.i".to_string(),
        "sltu x19, x15, x19".to_string(),
    ]);
    if positive {
        let vl = [1u8, 2, 3][(seed as usize) % 3];
        lines.push(format!("li x8, {}", probe_addr + 24));
        lines.push("li x19, 24".to_string());
        lines.push(vsetivli_line(
            "x0",
            template_cfg(isa_base, 64, vl, "m1", "tu", "mu"),
        ));
    } else {
        lines.push(format!("li x8, {}", probe_addr + 21));
    }
    lines.push(format!("vlsseg3e64.v v{base}, (x8), x19"));
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::SegmentStridedLoadLayout,
        contract: obligation_contract(RvvObligationKind::SegmentMemoryLayout),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn fault_only_first_load_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
    seed: u64,
    variant_index: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let deny_start = base_addr + 0x80 + (seed % 3) * 16;
    let deny_end = deny_start + 8;
    let (accessible_elements, scheduled_vd) = pmp_unit_fof_scenario(variant_index);
    let cfg = template_cfg(isa_base, 8, 8, "m1", "tu", "mu");
    let vd = if variant_index < 124 {
        scheduled_vd
    } else {
        operands.unit_fof_vd
    };
    let mut lines = vec![
        format!("li x5, {}", deny_start >> 2),
        "csrrw x0, pmpaddr0, x5".to_string(),
        format!("li x6, {}", deny_end >> 2),
        "csrrw x0, pmpaddr1, x6".to_string(),
        "li x7, 34816".to_string(),
        "csrrw x0, pmpcfg0, x7".to_string(),
        "li x10, 8".to_string(),
        format!("vsetvli x28, x10, {}", cfg.vtype_immediate()),
    ];
    push_sparse_load_destination_context(
        &mut lines,
        vd,
        seed,
        variant_index,
        0x504d_5055_4e49_5431,
        sparse_load_context_enabled(),
    );
    lines.extend([
        format!("li x11, {}", deny_start - accessible_elements),
        format!("vle8ff.v v{vd}, (x11)"),
        "csrr x27, vl".to_string(),
        "csrrw x29, vstart, x0".to_string(),
        format!("li x5, {base_addr}"),
        "sw x29, 0(x5)".to_string(),
        "sw x27, 8(x5)".to_string(),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::FaultOnlyFirstLoad,
        contract: obligation_contract(RvvObligationKind::FaultOnlyFirstSegment),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn pmp_unit_fof_scenario(variant: u64) -> (u64, u8) {
    // Four boundary positions per destination, including a full eight-element
    // read immediately before the denied region. The first cycle covers the
    // original v8 shape, and later cycles eventually visit every non-mask vd.
    let accessible = [8u64, 7, 5, 3][(variant % 4) as usize];
    let vd = 1 + (((variant / 4).wrapping_mul(5).wrapping_add(7) % 31) as u8);
    (accessible, vd)
}

fn fault_only_first_segment_e8_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let cfg = template_cfg(isa_base, 8, 8, "m1", "tu", "mu");
    let base = operands.segment_fof_e8_base;
    let mut lines = vec![format!("li x11, {base_addr}")];
    push_data_words(
        &mut lines,
        "x11",
        0,
        &[
            0x0102_0304_0506_0708,
            0x1112_1314_1516_1718,
            0x2122_2324_2526_2728,
            0x3132_3334_3536_3738,
        ],
    );
    lines.extend([
        vsetivli_line("x0", cfg),
        format!("vmv.v.i v{base}, 0"),
        format!("vmv.v.i v{}, 0", base + 1),
        format!("vlseg2e8ff.v v{base}, (x11)"),
        "addi x13, x11, 112".to_string(),
        format!("vs2r.v v{base}, 0(x13)"),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE8,
        contract: obligation_contract(RvvObligationKind::FaultOnlyFirstSegment),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn fault_only_first_segment_e32_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
    address_mode: HardwareRvvAddressMode,
    positive: bool,
    seed: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let probe_addr = high_memory_probe_base(base_addr, address_mode);
    let e64_m1 = template_cfg(isa_base, 64, 1, "m1", "tu", "mu");
    let base = operands.segment_fof_e32_base;
    let mut lines = vec![
        format!("li x13, {}", signed_literal(0x60dd_92a3_ecf5_bcd7)),
        format!("li x14, {}", signed_literal(0x6a81_406f_bfd2_6f3e)),
        format!("li x15, {}", signed_literal(0x5d4c_83e7_a011_bab6)),
        "li x16, -2048".to_string(),
        format!("li x17, {}", signed_literal(0x05d5_12ae_a884_1af0)),
        format!("li x18, {}", signed_literal(0xfbe0_dbb0_53f8_1b37)),
        "li x19, -129".to_string(),
        "li x20, -128".to_string(),
        format!("vsetvli x9, x0, {}", e64_m1.vtype_immediate()),
    ];
    for (reg, value) in [
        (13, 0xaaaa_aaaa_aaaa_aaaa),
        (14, 0xe8f8_75b2_60d6_20f1),
        (15, 0xaaaa_aaaa_aaaa_aaaa),
        (16, 0x0f0f_0f0f_0f0f_0f0f),
        (17, 0x43a7_95d7_e6d6_f1bf),
        (18, 0x41ff_7079_b17b_6979),
        (19, 0x38a9_82ab_6018_bf71),
        (20, 0x0f0f_0f0f_0f0f_0f0f),
    ] {
        push_vector_splat(&mut lines, reg, value);
    }
    lines.push(format!("li x9, {probe_addr}"));
    let words = [
        0xdea0_cf61_5541_b173,
        0xb89d_35a5_ad6c_0955,
        0xc35e_e50d_e2d6_6c47,
        0xc0c7_072d_21a3_ba47,
        0x279b_ee01_1041_d6e8,
        0x5a72_2c65_a16d_457c,
        0x0000_0000_0001_0000,
        0x0000_0000_0000_0001,
    ];
    push_data_words(&mut lines, "x9", 0, &words);
    lines.extend([
        format!("li x8, {}", probe_addr + 24),
        "sd x13, 16(x8)".to_string(),
        "mulw x15, x15, x15".to_string(),
        "divu x18, x19, x13".to_string(),
        "mulhu x14, x20, x17".to_string(),
    ]);
    if positive {
        let vl = [1u8, 2, 4][(seed as usize) % 3];
        lines.push(format!("li x12, {}", probe_addr + 24));
        lines.push(vsetivli_line(
            "x0",
            template_cfg(isa_base, 32, vl, "m1", "tu", "mu"),
        ));
    } else {
        lines.push(format!("li x12, {}", probe_addr + 26));
    }
    lines.push(format!("vlseg2e32ff.v v{base}, (x12)"));
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE32,
        contract: obligation_contract(RvvObligationKind::FaultOnlyFirstSegment),
        instructions: decode_block(&lines, isa_base)?,
    })
}

// Exercise a later-element segment FOF fault across a locked PMP boundary.
// The boundary and data vary with the seed; the legal field/count/base
// combinations rotate without relying on a lucky seed collision.
fn pmp_segment_fof_model_block(
    isa_base: ISABase,
    base_addr: u64,
    seed: u64,
    variant_index: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x504d_5053_4547_4646);
    let (fields, scheduled_base, accessible_elements) = pmp_segment_fof_scenario(variant_index);
    let base = if variant_index < 45 {
        scheduled_base
    } else {
        rng.random_range(1..=(32u8 - fields))
    };
    let deny_start = (base_addr + 0x80 + (seed % 3) * 16 + 7) & !7;
    let deny_end = deny_start + 8;
    let load_addr = deny_start - u64::from(fields) * accessible_elements;
    let mut lines = vec![format!("li x11, {}", deny_start - 64)];
    let words = (0..8).map(|_| rng.random::<u64>()).collect::<Vec<_>>();
    push_data_words(&mut lines, "x11", 0, &words);
    lines.extend([
        format!("li x5, {}", deny_start >> 2),
        "csrrw x0, pmpaddr0, x5".to_string(),
        format!("li x6, {}", deny_end >> 2),
        "csrrw x0, pmpaddr1, x6".to_string(),
        "li x7, 34816".to_string(),
        "csrrw x0, pmpcfg0, x7".to_string(),
        vsetivli_line("x0", template_cfg(isa_base, 8, 8, "m1", "tu", "mu")),
    ]);
    for field in 0..fields {
        lines.push(format!("vmv.v.i v{}, 0", base + field));
    }
    lines.extend([
        format!("li x12, {load_addr}"),
        format!("vlseg{fields}e8ff.v v{base}, (x12)"),
        "csrr x17, vl".to_string(),
        format!("li x10, {base_addr}"),
        "sd x17, 0(x10)".to_string(),
    ]);
    // The FOF load and VL readback are the architectural observation points.
    // One active-data readback is sufficient; extracting every segment field
    // after a fault can stall the DUT and hide the original FOF behavior.
    lines.push(format!("vmv.x.s x18, v{base}"));
    lines.push("sd x18, 8(x10)".to_string());
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::PmpSegmentFof,
        contract: obligation_contract(RvvObligationKind::FaultOnlyFirstSegment),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn pmp_segment_fof_scenario(variant: u64) -> (u8, u8, u64) {
    const SHAPES: [(u8, u64); 9] = [
        (2, 6),
        (2, 5),
        (2, 3),
        (3, 6),
        (3, 5),
        (3, 3),
        (4, 6),
        (4, 5),
        (4, 3),
    ];
    const BASES: [u8; 5] = [16, 8, 12, 20, 24];
    let (fields, accessible) = SHAPES[(variant % 9) as usize];
    // Coprime cycle lengths visit each shape/base pair once in 45 attempts.
    let base = BASES[(variant % 5) as usize];
    (fields, base, accessible)
}

fn scenario_variant_index(seed: u64) -> u64 {
    std::env::var("RISCV_FUZZ_RVV_VARIANT_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(seed)
}

fn memory_edge_probe_base(
    base_addr: u64,
    variant: u64,
    address_mode: HardwareRvvAddressMode,
) -> u64 {
    let high = [0x8fff_ffb8, 0x8fff_ffa8, 0x8fff_ffb0];
    // Pair the original high-DRAM byte window with both misaligned and
    // aligned FOF loads. The aligned path is a Positive reference case even
    // on Spike builds that trap the original misaligned e32 access.
    if address_mode != HardwareRvvAddressMode::AraLocal && matches!(variant % 8, 0 | 1 | 4 | 6) {
        high[((variant / 2) as usize) % high.len()]
    } else if address_mode == HardwareRvvAddressMode::AraLocal {
        base_addr + [0x160, 0x140, 0x120, 0x100][((variant / 2) % 4) as usize]
    } else {
        base_addr + 0x160 + ((variant / 2) & 1) * 8
    }
}

fn memory_edge_unit_destination(scenario: u64) -> u8 {
    1 + ((scenario.wrapping_mul(7).wrapping_add(13) % 31) as u8)
}

// Make the indexed store an actual producer for the subsequent unit FOF load.
// With VL=1 only element zero writes, so unordered indexed stores cannot race
// with one another.  The index group v16..v17 is separate from v8..v13.
fn push_memory_edge_indexed_producer(
    lines: &mut Vec<String>,
    isa_base: ISABase,
    load_addr: u64,
    index_offset: u64,
    fields: u8,
    first_byte: u8,
    rng: &mut StdRng,
) {
    lines.push(vsetivli_line(
        "x0",
        template_cfg(isa_base, 16, 1, "m2", "tu", "mu"),
    ));
    lines.push(format!("li x6, {index_offset}"));
    lines.push("vmv.v.x v16, x6".to_string());
    lines.push(vsetivli_line(
        "x0",
        template_cfg(isa_base, 8, 1, "m1", "tu", "mu"),
    ));
    for field in 0..fields {
        let value = if field == 0 { first_byte } else { rng.random() };
        lines.push(format!("li x6, {value}"));
        lines.push(format!("vmv.v.x v{}, x6", 8 + field));
    }
    lines.push("vmv.v.i v0, 1".to_string());
    lines.push(format!("li x9, {}", load_addr - index_offset));
    lines.push(format!("vsoxseg{fields}ei16.v v8, (x9), v16, v0.t"));
}

// Byte-oriented load after overlapping stores, with an active first element.
fn memory_edge_unit_fof_model_block(
    isa_base: ISABase,
    base_addr: u64,
    address_mode: HardwareRvvAddressMode,
    seed: u64,
    variant_index: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x4d45_4d55_464f_4631);
    let scenario = variant_index / 4;
    let relation = variant_index % 4;
    let probe = memory_edge_probe_base(base_addr, scenario, address_mode);
    let offset = [26u64, 24, 27, 30][(scenario % 4) as usize];
    // First expose all four relations at the original v14/high-address
    // shape. Then use coprime 31-register and 4-relation cycles: every
    // non-mask destination appears quickly, and every pair appears by 128.
    let vd = if variant_index < 4 {
        14
    } else {
        memory_edge_unit_destination(variant_index - 3)
    };
    let config_sew = [64u16, 8, 16, 32][(scenario % 4) as usize];
    let mask_pattern = [0xffu8, 0x55, 0x33, 0x01][((variant_index / 4) % 4) as usize];
    let vl = match config_sew {
        64 => 2,
        32 => [2u8, 4][rng.random_range(0..2)],
        _ => [2u8, 4, 6, 8][rng.random_range(0..4)],
    };
    let mut lines = vec![format!("li x9, {probe}")];
    let mut words = (0..8).map(|_| rng.random::<u64>()).collect::<Vec<_>>();
    if scenario % 16 == 0 {
        words[3] = 0x2c92_056b_ca47_0fac;
    }
    push_data_words(&mut lines, "x9", 0, &words);
    if relation >= 2 {
        let overlay = rng.random::<u32>();
        lines.push(format!("li x14, {overlay}"));
        lines.push("sw x14, 28(x9)".to_string());
        lines.push(format!("li x14, {}", overlay >> 16));
        lines.push("sh x14, 26(x9)".to_string());
    }
    if relation == 1 || relation == 3 {
        let index_offset = [19u64, 0, 4, 2][(scenario % 4) as usize];
        let fields = [6u8, 2, 3, 4][(scenario % 4) as usize];
        let first_byte = ((words[(offset / 8) as usize] >> ((offset % 8) * 8)) & 0xff) as u8;
        push_memory_edge_indexed_producer(
            &mut lines,
            isa_base,
            probe + offset,
            index_offset,
            fields,
            first_byte,
            &mut rng,
        );
    }
    if mask_pattern == 0xff {
        lines.extend([
            vsetivli_line(
                "x0",
                template_cfg(isa_base, config_sew, vl, "m1", "tu", "mu"),
            ),
            "li x14, -1".to_string(),
            "vmv.v.x v0, x14".to_string(),
        ]);
    } else {
        // Define every bit of v0 at e8/m1 VLMAX before narrowing VL. The
        // first mask bit remains active; later active/masked-off lanes vary.
        lines.extend([
            "li x7, -1".to_string(),
            format!(
                "vsetvli x0, x7, {}",
                template_cfg(isa_base, 8, 1, "m1", "tu", "mu").vtype_immediate()
            ),
            format!("li x14, {mask_pattern}"),
            "vmv.v.x v0, x14".to_string(),
            vsetivli_line(
                "x0",
                template_cfg(isa_base, config_sew, vl, "m1", "tu", "mu"),
            ),
        ]);
    }
    push_sparse_load_destination_context(
        &mut lines,
        vd,
        seed,
        variant_index,
        0x4d45_4d55_4e49_5431,
        sparse_load_context_enabled(),
    );
    lines.extend([
        format!("li x8, {}", probe + offset),
        format!("vle8ff.v v{vd}, (x8), v0.t"),
        "csrr x17, vl".to_string(),
        format!("li x10, {base_addr}"),
        "sd x17, 0(x10)".to_string(),
        "addi x15, x10, 16".to_string(),
        format!("vse8.v v{vd}, (x15)"),
        "lbu x18, 0(x15)".to_string(),
        "sd x18, 32(x10)".to_string(),
        "lbu x19, 1(x15)".to_string(),
        "sd x19, 40(x10)".to_string(),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::MemoryEdgeUnitFof,
        contract: obligation_contract(RvvObligationKind::FaultOnlyFirstSegment),
        instructions: decode_block(&lines, isa_base)?,
    })
}

// Element-oriented segment FOF load.  Cross each memory/configuration shape
// with no alias, a nearby misaligned store, an aligned store, and a partial
// overwrite.  The high-address historical shape and preceding scalar store
// can therefore occur in the same program instead of disjoint variants.
fn memory_edge_segment_fof_model_block(
    isa_base: ISABase,
    base_addr: u64,
    address_mode: HardwareRvvAddressMode,
    seed: u64,
    variant_index: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x4d45_4d53_464f_4632);
    let scenario = variant_index / 4;
    let relation = variant_index % 4;
    let probe = memory_edge_probe_base(base_addr, scenario, address_mode);
    let (sew, fields, mut offset, config_sew) = [
        (32u16, 2u8, 26u64, 64u16),
        (32, 2, 24, 64),
        (32, 3, 24, 32),
        (32, 4, 28, 32),
        (32, 2, 24, 32),
        (16, 2, 26, 16),
        (16, 3, 24, 16),
        (8, 4, 26, 8),
    ][(scenario % 8) as usize];
    if scenario % 8 == 2 && (scenario / 8) % 2 == 1 {
        offset = 26; // retain a less frequent NF3 misalignment probe
    }
    let base = if matches!(scenario % 8, 0 | 1) {
        20
    } else if scenario % 16 == 15 {
        0
    } else {
        rng.random_range(1..=(32u8 - fields))
    };
    let sampled_vl = if config_sew == 64 {
        2
    } else {
        [1u8, 2, 4][rng.random_range(0..3)]
    };
    // Every active field must come from the initialized 64-byte source
    // window. This also prevents a Positive FOF case from silently shrinking
    // VL only because the sampled group ran past its prepared data.
    let bytes_per_segment = u64::from(fields) * u64::from(sew / 8);
    let initialized_vl = ((64 - offset) / bytes_per_segment).max(1) as u8;
    let vl = sampled_vl.min(initialized_vl);
    let mut lines = vec![format!("li x9, {probe}")];
    let mut words = (0..8).map(|_| rng.random::<u64>()).collect::<Vec<_>>();
    if scenario % 16 == 0 {
        words[3] = 0xc0c7_072d_21a3_ba47;
    }
    push_data_words(&mut lines, "x9", 0, &words);
    if relation == 3 {
        let overlay = rng.random::<u32>();
        lines.push(format!("li x14, {overlay}"));
        lines.push("sw x14, 28(x9)".to_string());
        lines.push(format!("li x14, {}", overlay >> 16));
        lines.push("sh x14, 26(x9)".to_string());
    }
    if relation == 1 {
        // The original NF2/high-address relation uses an unaligned nearby
        // store. Other shapes keep an aligned same-line alias for Positive
        // comparison instead of paying a scalar trap on every variant.
        let historical_alias = scenario % 8 == 0;
        lines.push(format!(
            "li x15, {}",
            probe + if historical_alias { 27 } else { 24 }
        ));
        let value = if scenario % 16 == 0 {
            0x60dd_92a3_ecf5_bcd7
        } else {
            rng.random::<u64>()
        };
        lines.push(format!("li x13, {}", signed_literal(value)));
        lines.push(format!(
            "sd x13, {}(x15)",
            if historical_alias { 11 } else { 0 }
        ));
    } else if relation == 2 {
        lines.push(format!("li x15, {}", probe + 40));
        lines.push(format!("li x13, {}", signed_literal(rng.random::<u64>())));
        lines.push("sd x13, 0(x15)".to_string());
    }
    lines.push(vsetivli_line(
        "x0",
        template_cfg(isa_base, 64, 2, "m1", "tu", "mu"),
    ));
    for field in 0..fields {
        lines.push(format!("li x6, {}", signed_literal(rng.random::<u64>())));
        lines.push(format!("vmv.v.x v{}, x6", base + field));
    }
    lines.extend([
        vsetivli_line(
            "x0",
            template_cfg(isa_base, config_sew, vl, "m1", "tu", "mu"),
        ),
        format!("li x12, {}", probe + offset),
        format!("vlseg{fields}e{sew}ff.v v{base}, (x12)"),
        "csrr x17, vl".to_string(),
        format!("li x10, {base_addr}"),
        "sd x17, 0(x10)".to_string(),
    ]);
    // The FOF writes every destination field, which DiffTest can compare.
    // An immediate second-field consumer stalls this pinned DUT on the
    // aligned e32 variant; retain one active first-field scalar observation.
    // XLEN-extension-only differences belong to the scalar-extract family.
    lines.push(format!("vmv.x.s x18, v{base}"));
    lines.push("sd x18, 8(x10)".to_string());
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::MemoryEdgeSegmentFof,
        contract: obligation_contract(RvvObligationKind::FaultOnlyFirstSegment),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn indexed_segment_fof_model_block(
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
    address_mode: HardwareRvvAddressMode,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let probe_addr = high_memory_probe_base(base_addr, address_mode);
    let e64_m1 = template_cfg(isa_base, 64, 1, "m1", "tu", "mu");
    let base = operands.indexed_segment_base;
    let index = operands.indexed_segment_index;
    let mask = operands.mask;
    let fof_vd = operands.indexed_unit_fof_vd;
    let mut lines = vec![
        "li x13, 128".to_string(),
        format!("li x14, {}", signed_literal(0x1490_534e_200a_47ba)),
        format!("li x15, {}", signed_literal(0xfdf7_618d_2496_ba28)),
        format!("li x16, {}", signed_literal(0xc7be_21a9_94d6_aa8f)),
        format!("li x17, {}", signed_literal(0xcffd_b586_2808_9f8a)),
        "li x18, -2".to_string(),
        format!("li x19, {}", signed_literal(0x2210_dc3c_ed26_82e5)),
        format!("li x20, {}", signed_literal(0xd928_8d53_1a6e_8a27)),
        format!("vsetvli x9, x0, {}", e64_m1.vtype_immediate()),
    ];
    for (reg, value) in [
        (0, 0x0000_0000_0000_0000),
        (13, 0x0000_0000_0000_0000),
        (14, 0xaaaa_aaaa_aaaa_aaaa),
        (15, 0x0f0f_0f0f_0f0f_0f0f),
        (16, 0x0000_0000_0000_0000),
        (17, 0x0f0f_0f0f_0f0f_0f0f),
        (18, 0x5555_5555_5555_5555),
        (19, 0x0f0f_0f0f_0f0f_0f0f),
        (20, 0xffff_ffff_ffff_ffff),
    ] {
        push_vector_splat(&mut lines, reg, value);
    }
    lines.push(format!("li x9, {probe_addr}"));
    let words = [
        0x0000_0000_0000_7fff,
        0x0000_0000_0000_0100,
        0x0000_0000_0000_1000,
        0x2c92_056b_ca47_0fac,
        0x666a_dfe2_9621_898f,
        0x2b07_c73b_2b2e_eb90,
        0xbe60_ebff_7a3a_f395,
        0x9d41_418a_adb3_5b20,
    ];
    push_data_words(&mut lines, "x9", 0, &words);
    lines.extend([
        "srai x14, x19, 10".to_string(),
        "add x20, x20, x19".to_string(),
        "subw x15, x13, x15".to_string(),
        format!("li x21, {}", probe_addr + 24),
        "sd x16, 0(x21)".to_string(),
        "addw x17, x17, x16".to_string(),
        format!("li x9, {}", probe_addr + 7),
        format!("vsoxseg6ei16.v v{base}, (x9), v{index}, v{mask}.t"),
        "fence.i".to_string(),
        "mulh x14, x13, x17".to_string(),
        "ori x18, x16, -1816".to_string(),
        "fence.i".to_string(),
        "li x8, -1".to_string(),
        format!("vmv.v.x v{mask}, x8"),
        format!("li x8, {}", probe_addr + 26),
        format!("vle8ff.v v{fof_vd}, (x8), v{mask}.t"),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::IndexedSegmentFof,
        contract: obligation_contract(RvvObligationKind::IndexedSegmentFof),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn ara_masked_indexed_misaligned_no_trap_model_block(
    isa_base: ISABase,
    base_addr: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let idx_addr = base_addr;
    let mem_addr = base_addr + 0x40;
    let out_addr = base_addr + 0x80;
    let config = template_cfg(isa_base, 32, 4, "m1", "tu", "mu");
    let mut lines = vec![
        format!("li x10, {idx_addr}"),
        format!("li x11, {mem_addr}"),
        format!("li x12, {out_addr}"),
    ];
    push_data_words(
        &mut lines,
        "x10",
        0,
        &[0x0000_0004_0000_0001, 0x0000_000c_0000_0008],
    );
    push_data_words(
        &mut lines,
        "x11",
        0,
        &[0x2222_2222_1111_1111, 0x4444_4444_3333_3333],
    );
    lines.push(format!("li x12, {out_addr}"));
    lines.extend([
        vsetivli_line("x0", config),
        "li x13, -559038737".to_string(),
        "vmv.v.x v8, x13".to_string(),
        "vle32.v v4, (x10)".to_string(),
        "vid.v v1".to_string(),
        "vmsne.vi v0, v1, 0".to_string(),
        "vluxei32.v v8, (x11), v4, v0.t".to_string(),
        "vse32.v v8, (x12)".to_string(),
        "csrrw x14, vstart, x0".to_string(),
        "sd x14, 32(x12)".to_string(),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::AraMaskedIndexedMisalignedNoTrap,
        contract: obligation_contract(RvvObligationKind::SegmentMemoryLayout),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn ara_indexed_misaligned_trap_cause_model_block(
    isa_base: ISABase,
    base_addr: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let idx_addr = base_addr;
    let mem_addr = base_addr + 0x40;
    let out_addr = base_addr + 0x80;
    let config = template_cfg(isa_base, 32, 4, "m1", "tu", "mu");
    let mut lines = vec![
        format!("li x10, {idx_addr}"),
        format!("li x11, {mem_addr}"),
        format!("li x12, {out_addr}"),
    ];
    push_data_words(
        &mut lines,
        "x10",
        0,
        &[0x0000_0004_0000_0001, 0x0000_000c_0000_0008],
    );
    push_data_words(
        &mut lines,
        "x11",
        0,
        &[0x2222_2222_1111_1111, 0x4444_4444_3333_3333],
    );
    lines.push(format!("li x12, {out_addr}"));
    lines.extend([
        vsetivli_line("x0", config),
        "vle32.v v4, (x10)".to_string(),
        "vluxei32.v v8, (x11), v4".to_string(),
        "vse32.v v8, (x12)".to_string(),
        "csrrw x14, vstart, x0".to_string(),
        "sd x14, 32(x12)".to_string(),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::AraIndexedMisalignedTrapCause,
        contract: obligation_contract(RvvObligationKind::SegmentMemoryLayout),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn ara_masked_store_vstart_model_block(
    isa_base: ISABase,
    base_addr: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let src_addr = base_addr;
    let dst_addr = base_addr + 0x40;
    let config = template_cfg(isa_base, 32, 4, "m1", "tu", "mu");
    let mut lines = vec![format!("li x10, {src_addr}"), format!("li x11, {dst_addr}")];
    push_data_words(
        &mut lines,
        "x10",
        0,
        &[0x2222_2222_1111_1111, 0x4444_4444_3333_3333],
    );
    push_data_words(
        &mut lines,
        "x11",
        0,
        &[0xbbbb_bbbb_aaaa_aaaa, 0xdddd_dddd_cccc_cccc],
    );
    lines.extend([
        vsetivli_line("x0", config),
        "vle32.v v8, (x10)".to_string(),
        "vid.v v1".to_string(),
        "vand.vi v2, v1, 1".to_string(),
        "vmsne.vi v0, v2, 0".to_string(),
        "li x5, 1".to_string(),
        "csrrw x0, vstart, x5".to_string(),
        "vse32.v v8, (x11), v0.t".to_string(),
        "lw x12, 0(x11)".to_string(),
        "sd x12, 40(x11)".to_string(),
        "lw x12, 4(x11)".to_string(),
        "sd x12, 48(x11)".to_string(),
        "lw x12, 8(x11)".to_string(),
        "sd x12, 56(x11)".to_string(),
        "lw x12, 12(x11)".to_string(),
        "sd x12, 64(x11)".to_string(),
        "csrrw x6, vstart, x0".to_string(),
        "sd x6, 72(x11)".to_string(),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::AraMaskedStoreVstart,
        contract: obligation_contract(RvvObligationKind::VstartFaultProgress),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn ara_masked_vslideup_large_offset_model_block(
    isa_base: ISABase,
    base_addr: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let mask_addr = base_addr;
    let out_addr = base_addr + 0x80;
    let config = template_cfg(isa_base, 8, 0, "m1", "tu", "mu");
    let mut lines = vec![
        format!("li x11, {mask_addr}"),
        format!("li x12, {out_addr}"),
    ];
    push_data_words(&mut lines, "x11", 0, &[0, 0, 0, 0, 0xffff_ffff_ffff_ffff]);
    lines.push(format!("li x12, {out_addr}"));
    lines.extend([
        "li x5, 300".to_string(),
        format!("vsetvli x0, x5, {}", config.vtype_immediate()),
        "li x6, 170".to_string(),
        "vmv.v.x v1, x6".to_string(),
        "vid.v v2".to_string(),
        "vlm.v v0, (x11)".to_string(),
        "li x6, 256".to_string(),
        "vslideup.vx v1, v2, x6, v0.t".to_string(),
        "li x6, 255".to_string(),
        "vslidedown.vx v3, v1, x6".to_string(),
        "vmv.x.s x15, v3".to_string(),
        "sd x15, 0(x12)".to_string(),
        "li x6, 256".to_string(),
        "vslidedown.vx v3, v1, x6".to_string(),
        "vmv.x.s x15, v3".to_string(),
        "sd x15, 8(x12)".to_string(),
        "li x6, 257".to_string(),
        "vslidedown.vx v3, v1, x6".to_string(),
        "vmv.x.s x15, v3".to_string(),
        "sd x15, 16(x12)".to_string(),
        "li x6, 299".to_string(),
        "vslidedown.vx v3, v1, x6".to_string(),
        "vmv.x.s x15, v3".to_string(),
        "sd x15, 24(x12)".to_string(),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::AraMaskedVslideupLargeOffset,
        contract: obligation_contract(RvvObligationKind::MaskMergeCarry),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn ara_masked_dest_v0_overlap_model_block(
    isa_base: ISABase,
    base_addr: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let out_addr = base_addr;
    let config = template_cfg(isa_base, 8, 10, "m4", "tu", "mu");
    let lines = vec![
        format!("li x11, {out_addr}"),
        vsetivli_line("x8", config),
        "li x5, -1".to_string(),
        "vmv.v.x v0, x5".to_string(),
        "li x5, 51".to_string(),
        "vmv.v.x v20, x5".to_string(),
        "vmerge.vim v0, v20, -2, v0".to_string(),
        "vsm.v v0, (x11)".to_string(),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::AraMaskedDestV0Overlap,
        contract: obligation_contract(RvvObligationKind::MaskMergeCarry),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn ara_indexed_segment_later_field_model_block(
    isa_base: ISABase,
    base_addr: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let idx_addr = base_addr;
    let mem_addr = base_addr + 0x40;
    let out_addr = base_addr + 0x80;
    let config = template_cfg(isa_base, 32, 1, "m1", "tu", "mu");
    let mut lines = vec![
        format!("li x10, {idx_addr}"),
        format!("li x11, {mem_addr}"),
        format!("li x12, {out_addr}"),
    ];
    push_data_words(&mut lines, "x10", 0, &[0]);
    push_data_words(
        &mut lines,
        "x11",
        0,
        &[
            0x0000_1900_0000_1800,
            0x0000_2100_0000_2000,
            0x0000_0000_0000_2200,
        ],
    );
    lines.push(format!("li x12, {out_addr}"));
    lines.extend([
        vsetivli_line("x0", config),
        "vle32.v v14, (x10)".to_string(),
        "vluxseg5ei32.v v18, (x11), v14".to_string(),
        "vse32.v v18, (x12)".to_string(),
        "addi x13, x12, 8".to_string(),
        "vse32.v v19, (x13)".to_string(),
        "addi x13, x12, 16".to_string(),
        "vse32.v v20, (x13)".to_string(),
        "addi x13, x12, 24".to_string(),
        "vse32.v v21, (x13)".to_string(),
        "addi x13, x12, 32".to_string(),
        "vse32.v v22, (x13)".to_string(),
        "lwu x15, 0(x12)".to_string(),
        "sd x15, 80(x12)".to_string(),
        "lwu x15, 8(x12)".to_string(),
        "sd x15, 88(x12)".to_string(),
        "lwu x15, 16(x12)".to_string(),
        "sd x15, 96(x12)".to_string(),
        "lwu x15, 24(x12)".to_string(),
        "sd x15, 104(x12)".to_string(),
        "lwu x15, 32(x12)".to_string(),
        "sd x15, 112(x12)".to_string(),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::AraIndexedSegmentLaterField,
        contract: obligation_contract(RvvObligationKind::SegmentMemoryLayout),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn ara_strided_segment_later_field_model_block(
    isa_base: ISABase,
    base_addr: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let mem_addr = base_addr;
    let out_addr = base_addr + 0x80;
    let config = template_cfg(isa_base, 64, 1, "m1", "tu", "mu");
    let mut lines = vec![format!("li x11, {mem_addr}"), format!("li x12, {out_addr}")];
    push_data_words(
        &mut lines,
        "x11",
        0,
        &[
            0x1800_0000_0000_0000,
            0x1900_0000_0000_0000,
            0x2000_0000_0000_0000,
        ],
    );
    lines.push(format!("li x12, {out_addr}"));
    lines.extend([
        vsetivli_line("x0", config),
        "li x6, 32".to_string(),
        "vlsseg3e64.v v18, (x11), x6".to_string(),
        "vse64.v v18, (x12)".to_string(),
        "addi x13, x12, 8".to_string(),
        "vse64.v v19, (x13)".to_string(),
        "addi x13, x12, 16".to_string(),
        "vse64.v v20, (x13)".to_string(),
        "ld x15, 0(x12)".to_string(),
        "sd x15, 64(x12)".to_string(),
        "ld x15, 8(x12)".to_string(),
        "sd x15, 72(x12)".to_string(),
        "ld x15, 16(x12)".to_string(),
        "sd x15, 80(x12)".to_string(),
    ]);
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::AraStridedSegmentLaterField,
        contract: obligation_contract(RvvObligationKind::SegmentMemoryLayout),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn ara_masked_segment_lane5_model_block(
    isa_base: ISABase,
    base_addr: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let mask_addr = base_addr;
    let seg_addr = base_addr + 0x40;
    let out_addr = base_addr + 0x80;
    let config = template_cfg(isa_base, 16, 6, "m1", "tu", "mu");
    let lines = vec![
        format!("li x10, {mask_addr}"),
        format!("li x11, {seg_addr}"),
        format!("li x12, {out_addr}"),
        "li x13, 32".to_string(),
        "sd x13, 0(x10)".to_string(),
        "li x13, 4101".to_string(),
        "sh x13, 20(x11)".to_string(),
        "li x13, 8197".to_string(),
        "sh x13, 22(x11)".to_string(),
        vsetivli_line("x0", config),
        "li x13, 4369".to_string(),
        "vmv.v.x v13, x13".to_string(),
        "li x13, 8738".to_string(),
        "vmv.v.x v14, x13".to_string(),
        "vlm.v v0, (x10)".to_string(),
        "vlseg2e16.v v13, (x11), v0.t".to_string(),
        "vslidedown.vi v15, v13, 5".to_string(),
        "vmv.x.s x15, v15".to_string(),
        "sd x15, 0(x12)".to_string(),
        "vslidedown.vi v15, v14, 5".to_string(),
        "vmv.x.s x15, v15".to_string(),
        "sd x15, 8(x12)".to_string(),
    ];
    Ok(HardwareRvvModelBlock {
        kind: HardwareRvvModelBlockKind::AraMaskedSegmentLane5,
        contract: obligation_contract(RvvObligationKind::SegmentMemoryLayout),
        instructions: decode_block(&lines, isa_base)?,
    })
}

fn available_block_kinds(extensions: &IsaExtensions) -> Vec<HardwareRvvModelBlockKind> {
    let mut kinds = vec![
        HardwareRvvModelBlockKind::VlVtypeConfig,
        HardwareRvvModelBlockKind::ScalarExtractSignExt,
        HardwareRvvModelBlockKind::MaskMemoryRoundTrip,
        HardwareRvvModelBlockKind::MaskMergeCarry,
        HardwareRvvModelBlockKind::TailHighTransition,
        HardwareRvvModelBlockKind::VstartFaultProgress,
        HardwareRvvModelBlockKind::WholeRegisterRoundTrip,
        HardwareRvvModelBlockKind::StridedWholeRegisterStore,
        HardwareRvvModelBlockKind::WideningZeroExtend,
        HardwareRvvModelBlockKind::SegmentMaskedLoadLayout,
        HardwareRvvModelBlockKind::SegmentStridedLoadLayout,
        HardwareRvvModelBlockKind::FaultOnlyFirstLoad,
        HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE8,
        HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE32,
        HardwareRvvModelBlockKind::IndexedSegmentFof,
    ];
    if has_rv64_extension(extensions, RV64Extensions::F)
        && has_rv64_extension(extensions, RV64Extensions::D)
    {
        kinds.insert(2, HardwareRvvModelBlockKind::FloatExtractNanBox);
    }
    kinds
}

fn shuffled_model_index(attempt: u64, schedule_seed: u64, count: usize) -> usize {
    let epoch = attempt / count as u64;
    let position = attempt as usize % count;
    let mut order = (0..count).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(
        schedule_seed ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x5643_5f4b_494e_4453,
    );
    order.shuffle(&mut rng);
    order[position]
}

fn sampled_model_index(seed: u64, count: usize) -> usize {
    let attempt = std::env::var("RISCV_FUZZ_RVV_VC_KIND_INDEX")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    if let Some(attempt) = attempt {
        let schedule_seed = std::env::var("RISCV_FUZZ_RVV_VC_SCHEDULE_SEED")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(seed);
        return shuffled_model_index(attempt, schedule_seed, count);
    }
    // Mix the low seed bits before reducing to a kind index. A fixed 1-in-4
    // broad schedule must not permanently exclude 12 of 16 VC kinds.
    let mut value = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((value ^ (value >> 31)) % count as u64) as usize
}

#[allow(dead_code)]
fn legacy_ara_regression_block_kinds(
    _extensions: &IsaExtensions,
) -> Vec<HardwareRvvModelBlockKind> {
    vec![
        HardwareRvvModelBlockKind::AraWideningZeroExtendProbe,
        HardwareRvvModelBlockKind::AraMaskedIndexedMisalignedNoTrap,
        HardwareRvvModelBlockKind::AraIndexedMisalignedTrapCause,
        HardwareRvvModelBlockKind::AraMaskedStoreVstart,
        HardwareRvvModelBlockKind::AraMaskedVslideupLargeOffset,
        HardwareRvvModelBlockKind::AraMaskedDestV0Overlap,
        HardwareRvvModelBlockKind::AraIndexedSegmentLaterField,
        HardwareRvvModelBlockKind::AraStridedSegmentLaterField,
        HardwareRvvModelBlockKind::AraMaskedSegmentLane5,
        HardwareRvvModelBlockKind::SegmentMaskedLoadLayout,
        HardwareRvvModelBlockKind::SegmentStridedLoadLayout,
        HardwareRvvModelBlockKind::WholeRegisterRoundTrip,
        HardwareRvvModelBlockKind::StridedWholeRegisterStore,
        HardwareRvvModelBlockKind::FaultOnlyFirstLoad,
        HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE8,
        HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE32,
        HardwareRvvModelBlockKind::IndexedSegmentFof,
    ]
}

fn make_block(
    kind: HardwareRvvModelBlockKind,
    isa_base: ISABase,
    base_addr: u64,
    operands: RvvOperandAssignment,
    address_mode: HardwareRvvAddressMode,
    positive: bool,
    seed: u64,
    variant_index: u64,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    match kind {
        HardwareRvvModelBlockKind::VlVtypeConfig => {
            vl_vtype_config_model_block(isa_base, base_addr, variant_index)
        }
        HardwareRvvModelBlockKind::ScalarExtractSignExt => {
            sign_ext_model_block(isa_base, base_addr, operands)
        }
        HardwareRvvModelBlockKind::FloatExtractNanBox => {
            float_nan_box_model_block(isa_base, base_addr, operands)
        }
        HardwareRvvModelBlockKind::MaskMemoryRoundTrip => {
            mask_memory_round_trip_model_block(isa_base, base_addr, operands, variant_index)
        }
        HardwareRvvModelBlockKind::MaskMergeCarry => {
            mask_merge_carry_model_block(isa_base, base_addr, operands)
        }
        HardwareRvvModelBlockKind::TailHighTransition => {
            tail_high_transition_model_block(isa_base, base_addr, operands)
        }
        HardwareRvvModelBlockKind::VstartFaultProgress => {
            vstart_fault_progress_model_block(isa_base, base_addr, operands)
        }
        HardwareRvvModelBlockKind::WholeRegisterRoundTrip => {
            whole_register_round_trip_model_block(isa_base, base_addr, operands, address_mode)
        }
        HardwareRvvModelBlockKind::StridedWholeRegisterStore => {
            strided_whole_register_store_model_block(
                isa_base,
                base_addr,
                operands,
                address_mode,
                positive,
                seed,
                variant_index,
            )
        }
        HardwareRvvModelBlockKind::WideningZeroExtend => {
            widening_zero_extend_model_block(isa_base, base_addr, operands)
        }
        HardwareRvvModelBlockKind::AraWideningZeroExtendProbe => {
            ara_widening_zero_extend_probe_model_block(isa_base, base_addr)
        }
        HardwareRvvModelBlockKind::SegmentMaskedLoadLayout => {
            segment_masked_load_layout_model_block(isa_base, base_addr, operands, address_mode)
        }
        HardwareRvvModelBlockKind::SegmentStridedLoadLayout => {
            segment_strided_load_layout_model_block(
                isa_base,
                base_addr,
                operands,
                address_mode,
                positive,
                seed,
            )
        }
        HardwareRvvModelBlockKind::FaultOnlyFirstLoad => {
            fault_only_first_load_model_block(isa_base, base_addr, operands, seed, variant_index)
        }
        HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE8 => {
            fault_only_first_segment_e8_model_block(isa_base, base_addr, operands)
        }
        HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE32 => {
            fault_only_first_segment_e32_model_block(
                isa_base,
                base_addr,
                operands,
                address_mode,
                positive,
                seed,
            )
        }
        HardwareRvvModelBlockKind::PmpSegmentFof => {
            pmp_segment_fof_model_block(isa_base, base_addr, seed, variant_index)
        }
        HardwareRvvModelBlockKind::MemoryEdgeUnitFof => {
            memory_edge_unit_fof_model_block(isa_base, base_addr, address_mode, seed, variant_index)
        }
        HardwareRvvModelBlockKind::MemoryEdgeSegmentFof => memory_edge_segment_fof_model_block(
            isa_base,
            base_addr,
            address_mode,
            seed,
            variant_index,
        ),
        HardwareRvvModelBlockKind::IndexedSegmentFof => {
            indexed_segment_fof_model_block(isa_base, base_addr, operands, address_mode)
        }
        HardwareRvvModelBlockKind::AraMaskedIndexedMisalignedNoTrap => {
            ara_masked_indexed_misaligned_no_trap_model_block(isa_base, base_addr)
        }
        HardwareRvvModelBlockKind::AraIndexedMisalignedTrapCause => {
            ara_indexed_misaligned_trap_cause_model_block(isa_base, base_addr)
        }
        HardwareRvvModelBlockKind::AraMaskedStoreVstart => {
            ara_masked_store_vstart_model_block(isa_base, base_addr)
        }
        HardwareRvvModelBlockKind::AraMaskedVslideupLargeOffset => {
            ara_masked_vslideup_large_offset_model_block(isa_base, base_addr)
        }
        HardwareRvvModelBlockKind::AraMaskedDestV0Overlap => {
            ara_masked_dest_v0_overlap_model_block(isa_base, base_addr)
        }
        HardwareRvvModelBlockKind::AraIndexedSegmentLaterField => {
            ara_indexed_segment_later_field_model_block(isa_base, base_addr)
        }
        HardwareRvvModelBlockKind::AraStridedSegmentLaterField => {
            ara_strided_segment_later_field_model_block(isa_base, base_addr)
        }
        HardwareRvvModelBlockKind::AraMaskedSegmentLane5 => {
            ara_masked_segment_lane5_model_block(isa_base, base_addr)
        }
    }
}

#[cfg(test)]
fn make_block_with_flavor(
    kind: HardwareRvvModelBlockKind,
    isa_base: ISABase,
    base_addr: u64,
    seed: u64,
    flavor: HardwareRvvInjectionFlavor,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    make_block_with_flavor_and_address(
        kind,
        isa_base,
        base_addr,
        seed,
        flavor,
        HardwareRvvAddressMode::LegacyHighMemoryProbe,
    )
}

fn make_block_with_flavor_and_address(
    kind: HardwareRvvModelBlockKind,
    isa_base: ISABase,
    base_addr: u64,
    seed: u64,
    flavor: HardwareRvvInjectionFlavor,
    address_mode: HardwareRvvAddressMode,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    make_block_with_flavor_and_address_and_variant(
        kind,
        isa_base,
        base_addr,
        seed,
        flavor,
        address_mode,
        None,
    )
}

fn make_block_with_flavor_and_address_and_variant(
    kind: HardwareRvvModelBlockKind,
    isa_base: ISABase,
    base_addr: u64,
    seed: u64,
    flavor: HardwareRvvInjectionFlavor,
    address_mode: HardwareRvvAddressMode,
    variant_override: Option<u64>,
) -> Result<HardwareRvvModelBlock, GenerationError> {
    let obligation = obligation_kind_for_block(kind);
    let operands = match flavor {
        HardwareRvvInjectionFlavor::AmbientRandom => {
            ambient_rvv_obligation_operand_assignment(seed, obligation)
        }
        HardwareRvvInjectionFlavor::ContractConstrained
        | HardwareRvvInjectionFlavor::ContractPositive => {
            contract_rvv_obligation_operand_assignment(seed ^ ((obligation as u64) << 33))
        }
    };
    let block = make_block(
        kind,
        isa_base,
        base_addr,
        operands,
        address_mode,
        flavor == HardwareRvvInjectionFlavor::ContractPositive,
        seed,
        variant_override.unwrap_or_else(|| scenario_variant_index(seed)),
    )?;
    match flavor {
        HardwareRvvInjectionFlavor::AmbientRandom => ambient_randomize_block(block, isa_base, seed),
        HardwareRvvInjectionFlavor::ContractConstrained
        | HardwareRvvInjectionFlavor::ContractPositive => Ok(block),
    }
}

fn shift_shared_indices(indices: &mut BTreeSet<usize>, insertion: usize, inserted_len: usize) {
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

fn inject_hardware_rvv_blocks_with_flavor_and_address(
    testcase: &mut GeneratedTestCase,
    core_ids: &[String],
    flavor: HardwareRvvInjectionFlavor,
    address_mode: HardwareRvvAddressMode,
) -> Result<HardwareRvvInjectionReport, GenerationError> {
    if testcase.config.isa_base != ISABase::Rv64 {
        return Ok(HardwareRvvInjectionReport::default());
    }

    let Some(core_id) = core_ids.first() else {
        return Ok(HardwareRvvInjectionReport::default());
    };
    let Some((start, end)) = testcase.mem_range.get(core_id).copied() else {
        return Err(GenerationError::MemRangeNotFound {
            core_id: core_id.clone(),
        });
    };
    let base_addr = aligned_model_address(start, end)?;
    let address_mode = if address_mode == HardwareRvvAddressMode::MemRangeLocal
        && core_id.eq_ignore_ascii_case("ara")
    {
        HardwareRvvAddressMode::AraLocal
    } else {
        address_mode
    };
    let kinds = available_block_kinds(&testcase.extensions);
    if kinds.is_empty() || testcase.harts.is_empty() {
        return Ok(HardwareRvvInjectionReport::default());
    }

    let case_seed = testcase.config.seed.unwrap_or(0);
    let hart = &mut testcase.harts[0];
    let test = hart.sequences_mut(HartPhase::Test);
    let insertion = 0;
    let focus = std::env::var("RISCV_FUZZ_RVV_VC_FOCUS")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let broad_variant = if focus.is_empty() {
        std::env::var("RISCV_FUZZ_RVV_VC_KIND_INDEX")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(|attempt| attempt / kinds.len() as u64)
    } else {
        None
    };
    let kind = if flavor == HardwareRvvInjectionFlavor::ContractPositive {
        match focus.as_str() {
            "strided-whole" => HardwareRvvModelBlockKind::StridedWholeRegisterStore,
            "scalar-extract" => HardwareRvvModelBlockKind::ScalarExtractSignExt,
            "fp-extract" => HardwareRvvModelBlockKind::FloatExtractNanBox,
            "widen-zero" => HardwareRvvModelBlockKind::WideningZeroExtend,
            "mask-roundtrip" => HardwareRvvModelBlockKind::MaskMemoryRoundTrip,
            "fault-progress" => HardwareRvvModelBlockKind::VstartFaultProgress,
            "pmp-unit-fof" => HardwareRvvModelBlockKind::FaultOnlyFirstLoad,
            "pmp-segment-fof" => HardwareRvvModelBlockKind::PmpSegmentFof,
            "memory-edge-unit-fof" => HardwareRvvModelBlockKind::MemoryEdgeUnitFof,
            "memory-edge-segment-fof" => HardwareRvvModelBlockKind::MemoryEdgeSegmentFof,
            "config-vl" => HardwareRvvModelBlockKind::VlVtypeConfig,
            _ => kinds[sampled_model_index(case_seed, kinds.len())],
        }
    } else {
        kinds[sampled_model_index(case_seed, kinds.len())]
    };
    let block = make_block_with_flavor_and_address_and_variant(
        kind,
        testcase.config.isa_base,
        base_addr,
        case_seed,
        flavor,
        address_mode,
        broad_variant,
    )?;
    let inserted_len = block.instructions.len();

    test.splice(insertion..insertion, block.instructions);
    shift_shared_indices(
        hart.shared_sequence_indices_mut(HartPhase::Test),
        insertion,
        inserted_len,
    );

    Ok(HardwareRvvInjectionReport {
        inserted_blocks: 1,
        inserted_instructions: inserted_len,
        block_kinds: vec![kind],
    })
}

pub fn inject_hardware_rvv_blocks_with_flavor(
    testcase: &mut GeneratedTestCase,
    core_ids: &[String],
    flavor: HardwareRvvInjectionFlavor,
) -> Result<HardwareRvvInjectionReport, GenerationError> {
    inject_hardware_rvv_blocks_with_flavor_and_address(
        testcase,
        core_ids,
        flavor,
        HardwareRvvAddressMode::LegacyHighMemoryProbe,
    )
}

/// Use only the testcase's writable memory window for model probes. This
/// avoids relying on a subject-specific high DRAM address in model-only runs.
pub fn inject_hardware_rvv_blocks_local_with_flavor(
    testcase: &mut GeneratedTestCase,
    core_ids: &[String],
    flavor: HardwareRvvInjectionFlavor,
) -> Result<HardwareRvvInjectionReport, GenerationError> {
    inject_hardware_rvv_blocks_with_flavor_and_address(
        testcase,
        core_ids,
        flavor,
        HardwareRvvAddressMode::MemRangeLocal,
    )
}

pub fn inject_hardware_rvv_blocks(
    testcase: &mut GeneratedTestCase,
    core_ids: &[String],
) -> Result<HardwareRvvInjectionReport, GenerationError> {
    inject_hardware_rvv_blocks_with_flavor(
        testcase,
        core_ids,
        HardwareRvvInjectionFlavor::AmbientRandom,
    )
}

#[allow(dead_code)]
fn inject_legacy_ara_regression_rvv_block(
    testcase: &mut GeneratedTestCase,
    core_ids: &[String],
) -> Result<HardwareRvvInjectionReport, GenerationError> {
    if testcase.config.isa_base != ISABase::Rv64 {
        return Ok(HardwareRvvInjectionReport::default());
    }

    let Some(core_id) = core_ids.first() else {
        return Ok(HardwareRvvInjectionReport::default());
    };
    let Some((start, end)) = testcase.mem_range.get(core_id).copied() else {
        return Err(GenerationError::MemRangeNotFound {
            core_id: core_id.clone(),
        });
    };
    let base_addr = aligned_model_address(start, end)?;
    let kinds = legacy_ara_regression_block_kinds(&testcase.extensions);
    if kinds.is_empty() || testcase.harts.is_empty() {
        return Ok(HardwareRvvInjectionReport::default());
    }

    let case_seed = testcase.config.seed.unwrap_or(0);
    let hart = &mut testcase.harts[0];
    let test = hart.sequences_mut(HartPhase::Test);
    let insertion = 0;
    let kind = kinds[(case_seed as usize) % kinds.len()];
    let block = make_block_with_flavor_and_address(
        kind,
        testcase.config.isa_base,
        base_addr,
        case_seed,
        HardwareRvvInjectionFlavor::ContractConstrained,
        HardwareRvvAddressMode::MemRangeLocal,
    )?;
    let inserted_len = block.instructions.len();

    test.splice(insertion..insertion, block.instructions);
    shift_shared_indices(
        hart.shared_sequence_indices_mut(HartPhase::Test),
        insertion,
        inserted_len,
    );

    Ok(HardwareRvvInjectionReport {
        inserted_blocks: 1,
        inserted_instructions: inserted_len,
        block_kinds: vec![kind],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_context_prefills_one_group_without_long_initialization() {
        let mut baseline = Vec::new();
        push_sparse_load_destination_context(&mut baseline, 14, 31, 1, 7, false);
        assert_eq!(baseline, ["vmv.v.i v14, 0"]);

        let mut sparse = Vec::new();
        push_sparse_load_destination_context(&mut sparse, 14, 31, 1, 7, true);
        assert_eq!(sparse.len(), 2);
        assert!(sparse[0].starts_with("li x31, "));
        assert_eq!(sparse[1], "vmv.v.x v14, x31");

        let mut same = Vec::new();
        push_sparse_load_destination_context(&mut same, 14, 31, 1, 7, true);
        assert_eq!(sparse, same);
        let mut even_variant = Vec::new();
        push_sparse_load_destination_context(&mut even_variant, 14, 31, 0, 7, true);
        assert_eq!(even_variant, baseline);
    }

    #[test]
    fn broad_vc_schedule_covers_every_kind_despite_four_seed_stride() {
        for schedule_seed in [2041300000, 2042300000] {
            let selected = (0..16)
                .map(|attempt| shuffled_model_index(attempt, schedule_seed, 16))
                .collect::<BTreeSet<_>>();
            assert_eq!(selected.len(), 16);
            assert!(
                selected.contains(&0),
                "VL/Vtype obligation must be reachable"
            );
        }
    }

    #[test]
    fn memory_edge_unit_sweeps_all_legal_nonmask_destinations() {
        let destination = |variant: u64| {
            if variant < 4 {
                14
            } else {
                memory_edge_unit_destination(variant - 3)
            }
        };
        let registers = (0..35).map(destination).collect::<BTreeSet<_>>();
        assert_eq!(registers.len(), 31);
        assert_eq!(memory_edge_unit_destination(0), 14);
        assert!(!registers.contains(&0));
        let pairs = (4..128)
            .map(|variant| (destination(variant), variant % 4))
            .collect::<BTreeSet<_>>();
        assert_eq!(pairs.len(), 31 * 4);
    }

    #[test]
    fn mask_roundtrip_observes_active_predicate_effect_for_each_vl() {
        for variant in 0..4 {
            let block = make_block_with_flavor_and_address_and_variant(
                HardwareRvvModelBlockKind::MaskMemoryRoundTrip,
                ISABase::Rv64,
                0x8001_0000,
                2026102100 + variant,
                HardwareRvvInjectionFlavor::ContractPositive,
                HardwareRvvAddressMode::MemRangeLocal,
                Some(variant),
            )
            .expect("mask load and active masked consumer decode");
            let lines = block
                .instructions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let mask_load = lines
                .iter()
                .position(|line| line.starts_with("vlm.v "))
                .unwrap();
            let consumer = lines
                .iter()
                .position(|line| line.starts_with("vadd.vi "))
                .unwrap();
            let observer = lines
                .iter()
                .position(|line| line.starts_with("lbu x16, "))
                .unwrap();
            assert!(mask_load < consumer && consumer < observer);
        }
    }

    #[test]
    fn memory_edge_unit_can_pair_high_v14_load_with_indexed_producer() {
        let block = make_block_with_flavor_and_address_and_variant(
            HardwareRvvModelBlockKind::MemoryEdgeUnitFof,
            ISABase::Rv64,
            0x8001_0000,
            2026102101,
            HardwareRvvInjectionFlavor::ContractPositive,
            HardwareRvvAddressMode::MemRangeLocal,
            Some(1),
        )
        .expect("indexed producer and unit FOF decode together");
        let lines = block
            .instructions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let joined = lines.join("\n");
        assert!(joined.contains("li x9, 2415919039"), "{joined}");
        assert!(
            joined.contains("vsoxseg6ei16.v v8, (x9), v16, v0.t"),
            "{joined}"
        );
        assert!(joined.contains("li x8, 2415919058"), "{joined}");
        assert!(joined.contains("vle8ff.v v14, (x8), v0.t"), "{joined}");
        assert!(joined.contains("li x6, 71"), "{joined}"); // original active byte 0x47
        let producer = lines
            .iter()
            .position(|line| line.starts_with("vsoxseg6ei16.v"))
            .unwrap();
        let consumer = lines
            .iter()
            .position(|line| line.starts_with("vle8ff.v"))
            .unwrap();
        assert!(producer < consumer);
    }

    #[test]
    fn memory_edge_unit_sparse_mask_has_defined_bits_and_two_observed_lanes() {
        let block = make_block_with_flavor_and_address_and_variant(
            HardwareRvvModelBlockKind::MemoryEdgeUnitFof,
            ISABase::Rv64,
            0x8001_0000,
            2026102304,
            HardwareRvvInjectionFlavor::ContractPositive,
            HardwareRvvAddressMode::MemRangeLocal,
            Some(4),
        )
        .expect("sparse mask and unit FOF decode together");
        let lines = block
            .instructions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let mask_fill = lines
            .iter()
            .position(|line| line == "vmv.v.x v0, x14")
            .unwrap();
        let target = lines
            .iter()
            .position(|line| line.starts_with("vle8ff.v"))
            .unwrap();
        let second_lane = lines
            .iter()
            .position(|line| line == "lbu x19, 1(x15)")
            .unwrap();
        assert!(lines.contains(&"li x14, 85".to_string()));
        assert!(lines.contains(&"li x7, -1".to_string()));
        assert!(mask_fill < target && target < second_lane);
    }

    #[test]
    fn memory_edge_segment_can_combine_high_e64_v20_and_nearby_store() {
        let block = make_block_with_flavor_and_address_and_variant(
            HardwareRvvModelBlockKind::MemoryEdgeSegmentFof,
            ISABase::Rv64,
            0x8001_0000,
            2026102101,
            HardwareRvvInjectionFlavor::ContractPositive,
            HardwareRvvAddressMode::MemRangeLocal,
            Some(1),
        )
        .expect("high-address segment FOF and preceding scalar store decode together");
        let lines = block
            .instructions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let joined = lines.join("\n");
        assert!(joined.contains("li x15, 2415919059"), "{joined}");
        assert!(joined.contains("sd x13, 11(x15)"), "{joined}");
        assert!(joined.contains("li x12, 2415919058"), "{joined}");
        assert!(joined.contains("vlseg2e32ff.v v20, (x12)"), "{joined}");
        assert!(joined.contains("vsetivli x0, 2, 24"), "{joined}");
        let producer = lines
            .iter()
            .position(|line| line == "sd x13, 11(x15)")
            .unwrap();
        let consumer = lines
            .iter()
            .position(|line| line.starts_with("vlseg2e32ff.v"))
            .unwrap();
        assert!(producer < consumer);
    }

    #[test]
    fn memory_edge_segment_retains_aligned_e64_positive_counterpart() {
        let block = make_block_with_flavor_and_address_and_variant(
            HardwareRvvModelBlockKind::MemoryEdgeSegmentFof,
            ISABase::Rv64,
            0x8001_0000,
            2026102106,
            HardwareRvvInjectionFlavor::ContractPositive,
            HardwareRvvAddressMode::MemRangeLocal,
            Some(6),
        )
        .expect("aligned e64/m1 segment FOF decodes");
        let lines = block
            .instructions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(lines.contains("li x12, 2415919056"), "{lines}");
        assert!(lines.contains("vsetivli x0, 2, 24"), "{lines}");
        assert!(lines.contains("vlseg2e32ff.v v20, (x12)"), "{lines}");
        assert!(lines.contains("sd x13, 0(x15)"), "{lines}");
    }

    #[test]
    fn pmp_unit_fof_rotates_boundary_and_destination_without_omitting_full_vl() {
        assert_eq!(pmp_unit_fof_scenario(0), (8, 8));
        let pairs = (0..124).map(pmp_unit_fof_scenario).collect::<BTreeSet<_>>();
        assert_eq!(pairs.len(), 124);
        assert_eq!(
            pairs
                .iter()
                .map(|(_, vd)| vd)
                .collect::<BTreeSet<_>>()
                .len(),
            31
        );
    }

    #[test]
    fn pmp_segment_fof_rotates_all_field_boundary_base_pairs() {
        assert_eq!(pmp_segment_fof_scenario(0), (2, 16, 6));
        let scenarios = (0..45)
            .map(pmp_segment_fof_scenario)
            .collect::<BTreeSet<_>>();
        assert_eq!(scenarios.len(), 45);
        assert!(scenarios.contains(&(2, 16, 6)));
    }

    #[test]
    fn positive_strided_whole_reaches_high_active_data_window() {
        let block = make_block_with_flavor_and_address(
            HardwareRvvModelBlockKind::StridedWholeRegisterStore,
            ISABase::Rv64,
            0x8001_0000,
            0,
            HardwareRvvInjectionFlavor::ContractPositive,
            HardwareRvvAddressMode::MemRangeLocal,
        )
        .expect("positive high-memory strided whole block decodes");
        let lines = block
            .instructions
            .iter()
            .map(|instruction| instruction.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(lines.contains("li x9, 2415919032"), "{lines}");
        assert!(lines.contains("vlse32.v v24"), "{lines}");
        assert!(lines.contains("vs4r.v v24"), "{lines}");
        assert!(lines.contains(&format!(
            "li x12, {}",
            signed_literal(0x4792_b1db_7b17_d676)
        )));
    }

    #[test]
    fn strided_whole_high_v24_cycles_preceding_alias_store() {
        for (variant, offset) in [(4, 8u64), (8, 7)] {
            let block = make_block_with_flavor_and_address_and_variant(
                HardwareRvvModelBlockKind::StridedWholeRegisterStore,
                ISABase::Rv64,
                0x8001_0000,
                2026102101,
                HardwareRvvInjectionFlavor::ContractPositive,
                HardwareRvvAddressMode::MemRangeLocal,
                Some(variant),
            )
            .expect("strided load with preceding alias store decodes");
            let lines = block
                .instructions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let joined = lines.join("\n");
            assert!(
                joined.contains(&format!("li x2, {}", 0x8fff_ffb8 + offset)),
                "{joined}"
            );
            assert!(joined.contains("vlse32.v v24, (x9), x13"), "{joined}");
            assert!(
                joined.contains(&format!(
                    "li x12, {}",
                    signed_literal(0x4792_b1db_7b17_d676)
                )),
                "{joined}"
            );
            let producer = lines
                .iter()
                .position(|line| line == "c.sdsp x13, 0(sp)")
                .unwrap();
            let consumer = lines
                .iter()
                .position(|line| line.starts_with("vlse32.v"))
                .unwrap();
            assert!(producer < consumer);
        }
    }

    #[test]
    fn strided_whole_vector_store_feeds_target_memory_across_vtypes() {
        let mut target_vtypes = BTreeSet::new();
        for variant in (4..80).filter(|variant| variant % 5 == 4) {
            let block = make_block_with_flavor_and_address_and_variant(
                HardwareRvvModelBlockKind::StridedWholeRegisterStore,
                ISABase::Rv64,
                0x8001_0000,
                2026102304,
                HardwareRvvInjectionFlavor::ContractPositive,
                HardwareRvvAddressMode::MemRangeLocal,
                Some(variant),
            )
            .expect("vector memory producer and strided target decode together");
            let lines = block
                .instructions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let producer = lines
                .iter()
                .position(|line| line == "vse32.v v4, (x11)")
                .unwrap();
            let target = lines
                .iter()
                .position(|line| line.starts_with("vlse32.v"))
                .unwrap();
            assert!(producer < target);
            target_vtypes.extend(
                lines
                    .iter()
                    .filter(|line| line.starts_with("vsetvli x8, x7, "))
                    .cloned(),
            );
            if variant == 4 {
                assert!(lines.contains(&"li x11, 2415919052".to_string()));
                assert!(lines.contains(&"vlse32.v v24, (x9), x13".to_string()));
            }
        }
        assert_eq!(target_vtypes.len(), 4);
    }

    #[test]
    fn ara_memory_relations_use_local_window_and_bounded_vl() {
        for kind in [
            HardwareRvvModelBlockKind::StridedWholeRegisterStore,
            HardwareRvvModelBlockKind::MemoryEdgeUnitFof,
            HardwareRvvModelBlockKind::MemoryEdgeSegmentFof,
        ] {
            let block = make_block_with_flavor_and_address_and_variant(
                kind,
                ISABase::Rv64,
                0x8001_0000,
                2026102304,
                HardwareRvvInjectionFlavor::ContractPositive,
                HardwareRvvAddressMode::AraLocal,
                Some(4),
            )
            .expect("Ara-local memory dependency decodes");
            let lines = block
                .instructions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            assert!(lines.iter().all(|line| !line.contains("2415919")));
            if kind == HardwareRvvModelBlockKind::StridedWholeRegisterStore {
                assert!(lines.contains(&"li x7, 16".to_string()));
                assert!(
                    lines
                        .iter()
                        .any(|line| line.starts_with("vsetvli x8, x7, "))
                );
            }
        }
    }

    #[test]
    fn strided_whole_probe_reserves_all_eight_initialized_registers() {
        for seed in 0..1024 {
            let base = contract_rvv_obligation_operand_assignment(seed).strided_whole_base;
            assert_eq!(base % 8, 0);
            assert!(base + 7 <= 31, "seed {seed} chose v{base}..v{}", base + 7);
        }
    }

    const ALL_KINDS: &[HardwareRvvModelBlockKind] = &[
        HardwareRvvModelBlockKind::VlVtypeConfig,
        HardwareRvvModelBlockKind::ScalarExtractSignExt,
        HardwareRvvModelBlockKind::FloatExtractNanBox,
        HardwareRvvModelBlockKind::MaskMemoryRoundTrip,
        HardwareRvvModelBlockKind::MaskMergeCarry,
        HardwareRvvModelBlockKind::TailHighTransition,
        HardwareRvvModelBlockKind::VstartFaultProgress,
        HardwareRvvModelBlockKind::WholeRegisterRoundTrip,
        HardwareRvvModelBlockKind::StridedWholeRegisterStore,
        HardwareRvvModelBlockKind::WideningZeroExtend,
        HardwareRvvModelBlockKind::SegmentMaskedLoadLayout,
        HardwareRvvModelBlockKind::SegmentStridedLoadLayout,
        HardwareRvvModelBlockKind::FaultOnlyFirstLoad,
        HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE8,
        HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE32,
        HardwareRvvModelBlockKind::IndexedSegmentFof,
    ];

    #[test]
    fn positive_contract_keeps_every_vc_vector_opcode() {
        for kind in ALL_KINDS {
            let render = |flavor| {
                make_block_with_flavor_and_address(
                    *kind,
                    ISABase::Rv64,
                    0x8001_0000,
                    2026101600,
                    flavor,
                    HardwareRvvAddressMode::MemRangeLocal,
                )
                .unwrap()
                .instructions
                .iter()
                .filter_map(|seq| {
                    let line = seq.to_string();
                    let name = line.split_whitespace().next()?.to_string();
                    (name.starts_with('v') && !name.starts_with("vset")).then_some(name)
                })
                .collect::<Vec<_>>()
            };
            let constrained = render(HardwareRvvInjectionFlavor::ContractConstrained);
            let positive = render(HardwareRvvInjectionFlavor::ContractPositive);
            if *kind == HardwareRvvModelBlockKind::StridedWholeRegisterStore {
                // Positive strided tests may add one legal memory producer,
                // but must retain the original mixed-EEW load/store target.
                let original = positive
                    .into_iter()
                    .filter(|name| {
                        !matches!(name.as_str(), "vle64.v" | "vlseg3e64.v" | "vlsseg3e64.v")
                    })
                    .collect::<Vec<_>>();
                assert_eq!(constrained, original, "{kind:?}");
            } else {
                assert_eq!(constrained, positive, "{kind:?}");
            }
        }
    }

    #[test]
    fn generic_model_blocks_keep_legacy_probe_address() {
        let base_addr = 0x8001_0000;
        let legacy_probe = (DIRECT_RUNNER_DRAM_END - 0x48).to_string();
        let local_probe = (base_addr + HIGH_MEMORY_PROBE_OFFSET).to_string();

        let legacy = make_block_with_flavor(
            HardwareRvvModelBlockKind::WholeRegisterRoundTrip,
            ISABase::Rv64,
            base_addr,
            2026052600,
            HardwareRvvInjectionFlavor::ContractConstrained,
        )
        .expect("legacy-address block decodes");
        let local = make_block_with_flavor_and_address(
            HardwareRvvModelBlockKind::WholeRegisterRoundTrip,
            ISABase::Rv64,
            base_addr,
            2026052600,
            HardwareRvvInjectionFlavor::ContractConstrained,
            HardwareRvvAddressMode::MemRangeLocal,
        )
        .expect("mem-range-local block decodes");

        let legacy_rendered = legacy
            .instructions
            .iter()
            .map(|seq| seq.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let local_rendered = local
            .instructions
            .iter()
            .map(|seq| seq.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(legacy_rendered.contains(&format!("li x11, {legacy_probe}")));
        assert!(!legacy_rendered.contains(&format!("li x11, {local_probe}")));
        assert!(local_rendered.contains(&format!("li x11, {local_probe}")));
    }

    #[test]
    fn all_model_blocks_decode() {
        for kind in ALL_KINDS {
            let block = make_block_with_flavor(
                *kind,
                ISABase::Rv64,
                0x8001_0000,
                2026052600 + *kind as u64,
                HardwareRvvInjectionFlavor::AmbientRandom,
            )
            .unwrap_or_else(|err| panic!("{kind:?} should decode: {err:?}"));
            assert_eq!(block.kind, *kind);
            assert!(!block.instructions.is_empty());
        }
    }

    #[test]
    fn all_ambient_model_blocks_decode_with_random_operands() {
        for kind in ALL_KINDS {
            let block = make_block_with_flavor(
                *kind,
                ISABase::Rv64,
                0x8001_0000,
                2026052700 + *kind as u64,
                HardwareRvvInjectionFlavor::AmbientRandom,
            )
            .unwrap_or_else(|err| panic!("ambient {kind:?} should decode: {err:?}"));
            assert_eq!(block.kind, *kind);
            assert!(!block.instructions.is_empty());
        }
    }

    #[test]
    fn contract_constrained_float_obligation_preserves_nan_box_shape() {
        let block = make_block_with_flavor(
            HardwareRvvModelBlockKind::FloatExtractNanBox,
            ISABase::Rv64,
            0x8001_0000,
            2026052602,
            HardwareRvvInjectionFlavor::ContractConstrained,
        )
        .expect("contract-constrained block decodes");
        let rendered = block
            .instructions
            .iter()
            .map(|seq| seq.to_string())
            .collect::<Vec<_>>()
            .join("\\n");
        assert!(rendered.contains("vle32.v v"), "{rendered}");
        assert!(rendered.contains("vfmv.f.s f23, v"), "{rendered}");
        assert!(rendered.contains("vsetivli x0, 1, 16"), "{rendered}");
    }

    #[test]
    fn contract_constrained_mask_merge_preserves_contract_config() {
        let block = make_block_with_flavor(
            HardwareRvvModelBlockKind::MaskMergeCarry,
            ISABase::Rv64,
            0x8001_0000,
            2026052604,
            HardwareRvvInjectionFlavor::ContractConstrained,
        )
        .expect("contract-constrained block decodes");
        let rendered = block
            .instructions
            .iter()
            .map(|seq| seq.to_string())
            .collect::<Vec<_>>()
            .join("\\n");
        assert!(rendered.contains("vsetivli x0, 8, 10"), "{rendered}");
        assert!(rendered.contains("vmerge.vvm"), "{rendered}");
    }

    #[test]
    fn mask_memory_obligation_keeps_v0_carrier_with_ambient_randomization() {
        let block = make_block_with_flavor(
            HardwareRvvModelBlockKind::MaskMemoryRoundTrip,
            ISABase::Rv64,
            0x8001_0000,
            2026052603,
            HardwareRvvInjectionFlavor::AmbientRandom,
        )
        .expect("ambient block decodes");
        let rendered = block
            .instructions
            .iter()
            .map(|seq| seq.to_string())
            .collect::<Vec<_>>()
            .join("\\n");
        assert!(rendered.contains("vlm.v v0"), "{rendered}");
        assert!(rendered.contains("vsm.v v0"), "{rendered}");
    }

    #[test]
    fn sign_ext_obligation_keeps_observation_chain_without_constraint_mode() {
        let block = make_block_with_flavor(
            HardwareRvvModelBlockKind::ScalarExtractSignExt,
            ISABase::Rv64,
            0x8001_0000,
            2026052601,
            HardwareRvvInjectionFlavor::AmbientRandom,
        )
        .expect("ambient block decodes");
        let rendered = block
            .instructions
            .iter()
            .map(|seq| seq.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("vmv.v.x v"));
        assert!(rendered.contains("vle32.v v"));
        assert!(rendered.contains("vmv.x.s x15, v"));
    }

    #[test]
    fn ambient_random_flavor_varies_configs_and_vector_operands_by_seed() {
        let left = make_block_with_flavor(
            HardwareRvvModelBlockKind::ScalarExtractSignExt,
            ISABase::Rv64,
            0x8001_0000,
            2026052600,
            HardwareRvvInjectionFlavor::AmbientRandom,
        )
        .expect("left block decodes");
        let right = make_block_with_flavor(
            HardwareRvvModelBlockKind::ScalarExtractSignExt,
            ISABase::Rv64,
            0x8001_0000,
            2026052601,
            HardwareRvvInjectionFlavor::AmbientRandom,
        )
        .expect("right block decodes");

        let left_rendered = left
            .instructions
            .iter()
            .map(|seq| seq.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let right_rendered = right
            .instructions
            .iter()
            .map(|seq| seq.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(left.kind, right.kind);
        assert_eq!(left.instructions.len(), right.instructions.len());
        assert_ne!(left_rendered, right_rendered);
        assert!(left_rendered.contains("vmv.x.s x15, v"));
        assert!(left_rendered.contains("vle32.v v"));
        assert!(left_rendered.contains("vsetivli"));
    }

    #[test]
    fn rotation_covers_all_available_kinds_for_contiguous_seeds() {
        let mut seen = BTreeSet::new();
        let kinds = [
            HardwareRvvModelBlockKind::VlVtypeConfig,
            HardwareRvvModelBlockKind::ScalarExtractSignExt,
            HardwareRvvModelBlockKind::FloatExtractNanBox,
            HardwareRvvModelBlockKind::MaskMemoryRoundTrip,
            HardwareRvvModelBlockKind::MaskMergeCarry,
            HardwareRvvModelBlockKind::TailHighTransition,
            HardwareRvvModelBlockKind::VstartFaultProgress,
            HardwareRvvModelBlockKind::WholeRegisterRoundTrip,
            HardwareRvvModelBlockKind::StridedWholeRegisterStore,
            HardwareRvvModelBlockKind::WideningZeroExtend,
            HardwareRvvModelBlockKind::SegmentMaskedLoadLayout,
            HardwareRvvModelBlockKind::SegmentStridedLoadLayout,
            HardwareRvvModelBlockKind::FaultOnlyFirstLoad,
            HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE8,
            HardwareRvvModelBlockKind::FaultOnlyFirstSegmentE32,
            HardwareRvvModelBlockKind::IndexedSegmentFof,
        ];
        for seed in 2026052200u64..2026052300 {
            seen.insert(kinds[(seed as usize) % kinds.len()]);
        }
        assert_eq!(seen.len(), kinds.len());
    }
}
