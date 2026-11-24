use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::Write as _,
    fs,
    path::Path,
};

use riscv_fuzz_plugin_api::FloatExtension;
use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
use riscv_instruction::{
    MemoryAccessInstruction, RV32_I_ADDI, RV32IInstructions, RV32Instruction, RiscvInstruction,
};
use riscv_instruction_types::{
    CsrSelectionConfig, Immediate, InstructionSequence, IntegerRegister, MemoryAccessConfig,
    PseudoInstruction, RandomConfig, RegisterOperandConfig, ValidatedValue,
};
use serde::{
    Deserialize, Serialize,
    de::{self, Deserializer},
};

use crate::sync_report::{fmt_percent, write_metric_definitions_section};
use riscv_fuzz_toolchain::{ArtifactIoError, write_json_pretty, write_string_file};

pub type RenderedInstructionPair = (String, Option<i64>);
pub type RenderedPairsPerHart = Vec<(u32, Vec<RenderedInstructionPair>)>;
pub type TypedSequencesPerHart = Vec<(u32, Vec<InstructionSequence<RiscvInstruction>>)>;

fn rendered_pair(instruction: &RiscvInstruction) -> RenderedInstructionPair {
    (instruction.to_string(), instruction.offset_operand_value())
}

fn flatten_sequence(sequence: &InstructionSequence<RiscvInstruction>) -> Vec<RiscvInstruction> {
    let mut instructions =
        Vec::with_capacity(sequence.pre_instructions.len() + 1 + sequence.post_instructions.len());
    instructions.extend(sequence.pre_instructions.iter().cloned());
    instructions.push(sequence.instruction.clone());
    instructions.extend(sequence.post_instructions.iter().cloned());
    instructions
}

fn sequence_traceable_len(sequence: &InstructionSequence<RiscvInstruction>) -> usize {
    sequence.pre_instructions.len() + 1 + sequence.post_instructions.len()
}

fn sequence_traceable_entry_at(
    sequence: &InstructionSequence<RiscvInstruction>,
    traceable_index: usize,
    current: &mut usize,
) -> Option<RiscvInstruction> {
    for instruction in &sequence.pre_instructions {
        if *current == traceable_index {
            return Some(instruction.clone());
        }
        *current += 1;
    }
    if *current == traceable_index {
        return Some(sequence.instruction.clone());
    }
    *current += 1;
    for instruction in &sequence.post_instructions {
        if *current == traceable_index {
            return Some(instruction.clone());
        }
        *current += 1;
    }
    None
}

fn sequence_traceable_entries(
    sequence: &InstructionSequence<RiscvInstruction>,
) -> impl Iterator<Item = &RiscvInstruction> {
    sequence
        .pre_instructions
        .iter()
        .chain(std::iter::once(&sequence.instruction))
        .chain(sequence.post_instructions.iter())
}

fn rendered_pairs_for_sequence(
    sequence: &InstructionSequence<RiscvInstruction>,
) -> Vec<RenderedInstructionPair> {
    sequence_traceable_entries(sequence)
        .map(rendered_pair)
        .collect()
}

fn sequence_without_removed_test_indices(
    sequence: &InstructionSequence<RiscvInstruction>,
    traceable_start: usize,
    remove: &BTreeSet<usize>,
) -> Option<InstructionSequence<RiscvInstruction>> {
    let retained = sequence_traceable_entries(sequence)
        .enumerate()
        .filter_map(|(local_idx, instruction)| {
            let global_idx = traceable_start + local_idx;
            (!remove.contains(&global_idx)).then(|| instruction.clone())
        })
        .collect::<Vec<_>>();

    if retained.len() == sequence_traceable_len(sequence) {
        return Some(sequence.clone());
    }
    if retained.is_empty() {
        return None;
    }

    let mut retained = retained.into_iter();
    let instruction = retained
        .next()
        .expect("non-empty retained instructions must contain a head entry");
    Some(InstructionSequence {
        pre_instructions: Vec::new(),
        instruction,
        post_instructions: retained.collect(),
    })
}

fn instruction_is_shared_base_address_marker(instruction: &RiscvInstruction) -> bool {
    matches!(
        instruction,
        RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
            purpose: riscv_instruction_types::LoadImmediatePurpose::BaseAddress,
            ..
        })
    )
}

fn hart_from_sequences(
    sequences: Vec<InstructionSequence<RiscvInstruction>>,
) -> Vec<RiscvInstruction> {
    let mut instructions = Vec::new();
    for sequence in &sequences {
        instructions.extend(flatten_sequence(sequence));
    }
    instructions
}

fn hart_rendered_pairs(hart: &[RiscvInstruction]) -> Vec<RenderedInstructionPair> {
    hart.iter().map(rendered_pair).collect()
}

fn hart_traceable_entry_at(
    hart: &[RiscvInstruction],
    traceable_index: usize,
    current: &mut usize,
) -> Option<RiscvInstruction> {
    for instruction in hart {
        if *current == traceable_index {
            return Some(instruction.clone());
        }
        *current += 1;
    }
    None
}

fn typed_nop_instruction() -> RiscvInstruction {
    RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(RV32_I_ADDI {
        xd: IntegerRegister::new(0).expect("x0 is valid"),
        xs1: IntegerRegister::new(0).expect("x0 is valid"),
        imm: Immediate::<12>::new(0).expect("zero immediate is valid"),
    })))
}

fn relocation_sets_empty(sets: &[BTreeSet<usize>]) -> bool {
    sets.iter().all(BTreeSet::is_empty)
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StructuredTestInstructions {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub harts: Vec<Vec<RiscvInstruction>>,
    #[serde(default, skip_serializing_if = "relocation_sets_empty")]
    pub shared_base_address_indices: Vec<BTreeSet<usize>>,
}

impl StructuredTestInstructions {
    pub fn from_harts(harts: Vec<Vec<InstructionSequence<RiscvInstruction>>>) -> Self {
        let harts = harts
            .into_iter()
            .map(hart_from_sequences)
            .collect::<Vec<_>>();
        let shared_base_address_indices = vec![BTreeSet::new(); harts.len()];
        Self {
            harts,
            shared_base_address_indices,
        }
    }

    pub fn from_hart_programs(harts: &[HartProgram], phase: HartPhase) -> Self {
        let hart_count = harts
            .iter()
            .map(|hart| hart.hart_id as usize + 1)
            .max()
            .unwrap_or(0);
        let mut program = Self {
            harts: vec![Vec::new(); hart_count],
            shared_base_address_indices: vec![BTreeSet::new(); hart_count],
        };

        for hart in harts {
            let hart_idx = hart.hart_id as usize;
            let sequences = hart.sequences(phase);
            let shared_sequences = hart.shared_sequence_indices(phase);
            let mut flat_idx = 0usize;
            for (seq_idx, sequence) in sequences.iter().enumerate() {
                for instruction in flatten_sequence(sequence) {
                    if shared_sequences.contains(&seq_idx)
                        && instruction_is_shared_base_address_marker(&instruction)
                    {
                        program.mark_shared_base_address_index(hart_idx, flat_idx);
                    }
                    program.push_instruction_to_hart(hart_idx, instruction);
                    flat_idx += 1;
                }
            }
        }

        program.normalize();
        program
    }

    fn ensure_relocation_shape(&mut self) {
        while self.shared_base_address_indices.len() < self.harts.len() {
            self.shared_base_address_indices.push(BTreeSet::new());
        }
    }

    pub fn normalize(&mut self) {
        self.ensure_relocation_shape();
    }

    pub fn ensure_non_empty_harts(&mut self) {
        self.ensure_relocation_shape();
        for hart in &mut self.harts {
            if hart.is_empty() {
                hart.push(typed_nop_instruction());
            }
        }
    }

    pub fn push_to_hart<T>(&mut self, hart_idx: usize, instruction: T)
    where
        T: Into<RiscvInstruction>,
    {
        while self.harts.len() <= hart_idx {
            self.harts.push(Vec::new());
        }
        self.ensure_relocation_shape();
        self.harts[hart_idx].push(instruction.into());
    }

    pub fn mark_shared_base_address_index(&mut self, hart_idx: usize, instruction_idx: usize) {
        self.ensure_relocation_shape();
        while self.shared_base_address_indices.len() <= hart_idx {
            self.shared_base_address_indices.push(BTreeSet::new());
        }
        self.shared_base_address_indices[hart_idx].insert(instruction_idx);
    }

    pub fn push_pseudo_to_hart(&mut self, hart_idx: usize, pseudo: PseudoInstruction) {
        self.push_to_hart(hart_idx, pseudo);
    }

    pub fn push_instruction_to_hart(&mut self, hart_idx: usize, instruction: RiscvInstruction) {
        self.push_to_hart(hart_idx, instruction);
    }

    pub fn rendered_pairs_for_hart(&self, hart_idx: usize) -> Option<Vec<RenderedInstructionPair>> {
        self.harts
            .get(hart_idx)
            .map(|hart| hart_rendered_pairs(hart))
    }

    pub fn rendered_pairs(&self) -> Vec<RenderedInstructionPair> {
        let mut pairs = Vec::new();
        for hart in self.harts.iter() {
            pairs.extend(hart_rendered_pairs(hart));
        }
        pairs
    }

    pub fn rendered_pairs_per_hart(&self) -> RenderedPairsPerHart {
        self.harts
            .iter()
            .enumerate()
            .map(|(idx, hart)| (idx as u32, hart_rendered_pairs(hart)))
            .collect()
    }

    pub fn rendered_lines_per_hart(&self) -> Vec<(u32, Vec<String>)> {
        self.harts
            .iter()
            .enumerate()
            .map(|(idx, hart)| {
                (
                    idx as u32,
                    hart_rendered_pairs(hart)
                        .into_iter()
                        .map(|(line, _)| line)
                        .collect(),
                )
            })
            .collect()
    }

    pub fn rendered_lines(&self) -> Vec<String> {
        self.rendered_pairs()
            .into_iter()
            .map(|(line, _)| line)
            .collect()
    }

    pub fn rendered_line_count(&self) -> usize {
        self.harts
            .iter()
            .map(|hart| hart_rendered_pairs(hart).len())
            .sum()
    }

    pub fn traceable_instruction_count(&self) -> usize {
        self.harts.iter().map(Vec::len).sum()
    }

    pub fn traceable_instruction_at(&self, traceable_index: usize) -> Option<String> {
        self.traceable_entry_at(traceable_index)
            .map(|instruction| instruction.to_string())
    }

    pub fn traceable_entry_at(&self, traceable_index: usize) -> Option<RiscvInstruction> {
        let mut current = 0usize;
        for hart in &self.harts {
            if let Some(entry) = hart_traceable_entry_at(hart, traceable_index, &mut current) {
                return Some(entry);
            }
        }
        None
    }

    pub fn remove_matching_instructions<F>(&mut self, mut should_remove: F)
    where
        F: FnMut(&RiscvInstruction) -> bool,
    {
        self.ensure_relocation_shape();
        for (hart_idx, hart) in self.harts.iter_mut().enumerate() {
            let shared = self
                .shared_base_address_indices
                .get(hart_idx)
                .cloned()
                .unwrap_or_default();
            let mut new_shared = BTreeSet::new();
            let mut new_hart = Vec::with_capacity(hart.len());
            for (idx, instruction) in hart.iter().cloned().enumerate() {
                if should_remove(&instruction) {
                    continue;
                }
                let new_idx = new_hart.len();
                if shared.contains(&idx) {
                    new_shared.insert(new_idx);
                }
                new_hart.push(instruction);
            }
            *hart = new_hart;
            if let Some(slot) = self.shared_base_address_indices.get_mut(hart_idx) {
                *slot = new_shared;
            }
        }
    }

    pub fn without_test_indices(&self, remove: &BTreeSet<usize>) -> Self {
        if remove.is_empty() {
            return self.clone();
        }

        let mut traceable_idx = 0usize;
        let mut harts = Vec::with_capacity(self.harts.len());
        let mut shared_base_address_indices = Vec::with_capacity(self.harts.len());
        for (hart_idx, hart) in self.harts.iter().enumerate() {
            let shared = self
                .shared_base_address_indices
                .get(hart_idx)
                .cloned()
                .unwrap_or_default();
            let mut new_hart = Vec::with_capacity(hart.len());
            let mut new_shared = BTreeSet::new();
            for (idx, instruction) in hart.iter().cloned().enumerate() {
                if remove.contains(&traceable_idx) {
                    traceable_idx += 1;
                    continue;
                }
                let new_idx = new_hart.len();
                if shared.contains(&idx) {
                    new_shared.insert(new_idx);
                }
                new_hart.push(instruction);
                traceable_idx += 1;
            }
            harts.push(new_hart);
            shared_base_address_indices.push(new_shared);
        }
        Self {
            harts,
            shared_base_address_indices,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HartConfig {
    #[default]
    Single,
    Dual {
        sync: MulticoreSyncConfig,
    },
}

impl HartConfig {
    pub fn hart_count(&self) -> u32 {
        match self {
            Self::Single => 1,
            Self::Dual { .. } => 2,
        }
    }

    pub fn is_dual(&self) -> bool {
        matches!(self, Self::Dual { .. })
    }

    pub fn sync_config(&self) -> Option<&MulticoreSyncConfig> {
        match self {
            Self::Single => None,
            Self::Dual { sync } => Some(sync),
        }
    }
}

fn default_testcase_isa_base() -> ISABase {
    ISABase::Rv64
}

fn default_testcase_extension_insts_count() -> usize {
    200
}

fn default_extension_inst_scaling() -> Option<ExtensionInstScaling> {
    Some(ExtensionInstScaling::default())
}

fn default_testcase_mem_size() -> u64 {
    64
}

fn default_unaligned_access_required() -> Option<bool> {
    Some(false)
}

fn default_testcase_random_config() -> RandomConfig {
    RandomConfig {
        isa_profile: riscv_instruction_types::IsaProfile::rv64(),
        memory_access_config: MemoryAccessConfig {
            memory_base_address_range: MemoryAccessConfig::default().memory_base_address_range,
            memory_offset_immediate_range: (0, 16),
            memory_base_register_number_range: (8, 12),
        },
        register_operand_config: RegisterOperandConfig {
            integer_register_number_range: (13, 20),
            floating_point_register_number_range: (13, 20),
            vector_register_number_range: (13, 20),
        },
        append_fflags_readback: false,
        csr_selection_config: CsrSelectionConfig {
            machine_csr_operands_enabled: true,
            ..CsrSelectionConfig::default()
        },
        edge_case_probability: 0.2,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestCaseConfig {
    #[serde(default = "default_testcase_isa_base")]
    pub isa_base: ISABase,
    #[serde(default = "default_testcase_extension_insts_count")]
    pub extension_insts_count: usize,
    #[serde(default = "default_extension_inst_scaling")]
    pub extension_inst_scaling: Option<ExtensionInstScaling>,
    #[serde(default = "default_testcase_mem_size")]
    pub mem_size: u64,
    #[serde(default)]
    pub float_extension: FloatExtension,
    #[serde(default = "default_unaligned_access_required")]
    pub unaligned_access_required: Option<bool>,
    #[serde(default = "default_testcase_random_config")]
    pub random_config: RandomConfig,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub hart_config: HartConfig,
}

impl Default for TestCaseConfig {
    fn default() -> Self {
        Self {
            isa_base: default_testcase_isa_base(),
            extension_insts_count: default_testcase_extension_insts_count(),
            extension_inst_scaling: default_extension_inst_scaling(),
            mem_size: default_testcase_mem_size(),
            float_extension: FloatExtension::default(),
            unaligned_access_required: default_unaligned_access_required(),
            random_config: default_testcase_random_config(),
            seed: None,
            hart_config: HartConfig::default(),
        }
    }
}

impl TestCaseConfig {
    pub fn hart_count(&self) -> u32 {
        self.hart_config.hart_count()
    }

    pub fn is_dual_hart(&self) -> bool {
        self.hart_config.is_dual()
    }

    pub fn multicore_sync_config(&self) -> Option<&MulticoreSyncConfig> {
        self.hart_config.sync_config()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MulticoreSyncConfig {
    #[serde(default = "default_interaction_slot_ratio")]
    pub interaction_slot_ratio: f64,
    #[serde(default = "default_shared_address_ratio")]
    pub shared_address_ratio: f64,
    #[serde(
        default = "default_independent_block_len_range",
        deserialize_with = "deserialize_usize_range"
    )]
    pub independent_block_len_range: (usize, usize),
    #[serde(
        default = "default_interaction_block_len_range",
        deserialize_with = "deserialize_usize_range"
    )]
    pub interaction_block_len_range: (usize, usize),
    #[serde(default = "default_prefer_focused_sequences")]
    pub prefer_focused_sequences: bool,
}

impl Default for MulticoreSyncConfig {
    fn default() -> Self {
        Self {
            interaction_slot_ratio: default_interaction_slot_ratio(),
            shared_address_ratio: default_shared_address_ratio(),
            independent_block_len_range: default_independent_block_len_range(),
            interaction_block_len_range: default_interaction_block_len_range(),
            prefer_focused_sequences: default_prefer_focused_sequences(),
        }
    }
}

const fn default_interaction_slot_ratio() -> f64 {
    0.35
}

const fn default_shared_address_ratio() -> f64 {
    0.70
}

const fn default_independent_block_len_range() -> (usize, usize) {
    (4, 16)
}

const fn default_interaction_block_len_range() -> (usize, usize) {
    (2, 8)
}

const fn default_prefer_focused_sequences() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MulticoreSyncMetrics {
    pub hart0_instruction_count: usize,
    pub hart1_instruction_count: usize,
    pub hart0_timed_instruction_count: usize,
    pub hart1_timed_instruction_count: usize,
    pub timing_coverage_rate: f64,
    pub cycle_window_start: u64,
    pub cycle_window_end: u64,
    pub covered_cycle_count: usize,
    pub hart0_active_cycle_count: usize,
    pub hart1_active_cycle_count: usize,
    pub simultaneous_cycle_count: usize,
    pub simultaneous_cycle_rate: f64,
    pub register_overlap_cycle_count: usize,
    pub both_memory_cycle_count: usize,
    pub same_address_cycle_count: usize,
    pub joint_overlap_cycle_count: usize,
    pub register_overlap_cycle_rate: f64,
    pub register_overlap_cycle_jaccard: f64,
    pub same_address_cycle_rate: f64,
    pub same_address_given_both_memory_cycle_rate: f64,
    pub memory_overlap_cycle_jaccard: f64,
    pub joint_overlap_cycle_rate: f64,
    pub cycle_contention_score: f64,
    pub configured_interaction_slot_ratio: f64,
    pub configured_shared_address_ratio: f64,
    pub configured_independent_block_len_range: (usize, usize),
    pub configured_interaction_block_len_range: (usize, usize),
    pub prefer_focused_sequences: bool,
}

pub fn render_multicore_sync_metrics_markdown(metrics: &MulticoreSyncMetrics) -> String {
    let mut out = String::new();
    writeln!(&mut out, "# Multicore Sync Metrics").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "These metrics are computed from real per-instruction cycle intervals (`clk_start..=clk_end`)."
    )
    .unwrap();
    writeln!(
        &mut out,
        "Resource overlap is counted only on cycles where both harts are active at the same time."
    )
    .unwrap();
    writeln!(
        &mut out,
        "Idle gaps inside [`cycle_window_start`, `cycle_window_end`] where neither hart is active are excluded from `covered_cycle_count` and from the `Simul` denominator."
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    write_metric_definitions_section(&mut out);
    writeln!(&mut out, "## Core Metrics").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "| Metric | Value |").unwrap();
    writeln!(&mut out, "| --- | ---: |").unwrap();
    writeln!(
        &mut out,
        "| Simul | {} |",
        fmt_percent(metrics.simultaneous_cycle_rate)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| Reg | {} |",
        fmt_percent(metrics.register_overlap_cycle_rate)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| Mem | {} |",
        fmt_percent(metrics.same_address_given_both_memory_cycle_rate)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| Score | {} |",
        fmt_percent(metrics.cycle_contention_score)
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Coverage And Window").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "| Field | Value |").unwrap();
    writeln!(&mut out, "| --- | ---: |").unwrap();
    writeln!(
        &mut out,
        "| hart0_instruction_count | {} |",
        metrics.hart0_instruction_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| hart1_instruction_count | {} |",
        metrics.hart1_instruction_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| hart0_timed_instruction_count | {} |",
        metrics.hart0_timed_instruction_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| hart1_timed_instruction_count | {} |",
        metrics.hart1_timed_instruction_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| timing_coverage_rate | {} |",
        fmt_percent(metrics.timing_coverage_rate)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| cycle_window_start | {} |",
        metrics.cycle_window_start
    )
    .unwrap();
    writeln!(
        &mut out,
        "| cycle_window_end | {} |",
        metrics.cycle_window_end
    )
    .unwrap();
    writeln!(
        &mut out,
        "| covered_cycle_count | {} |",
        metrics.covered_cycle_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| hart0_active_cycle_count | {} |",
        metrics.hart0_active_cycle_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| hart1_active_cycle_count | {} |",
        metrics.hart1_active_cycle_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| simultaneous_cycle_count | {} |",
        metrics.simultaneous_cycle_count
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Diagnostics").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "| Field | Value |").unwrap();
    writeln!(&mut out, "| --- | ---: |").unwrap();
    writeln!(
        &mut out,
        "| register_overlap_cycle_count | {} |",
        metrics.register_overlap_cycle_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| both_memory_cycle_count | {} |",
        metrics.both_memory_cycle_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| same_address_cycle_count | {} |",
        metrics.same_address_cycle_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| joint_overlap_cycle_count | {} |",
        metrics.joint_overlap_cycle_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| register_overlap_cycle_jaccard | {} |",
        fmt_percent(metrics.register_overlap_cycle_jaccard)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| same_address_cycle_rate | {} |",
        fmt_percent(metrics.same_address_cycle_rate)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| same_address_given_both_memory_cycle_rate | {} |",
        fmt_percent(metrics.same_address_given_both_memory_cycle_rate)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| memory_overlap_cycle_jaccard | {} |",
        fmt_percent(metrics.memory_overlap_cycle_jaccard)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| joint_overlap_cycle_rate | {} |",
        fmt_percent(metrics.joint_overlap_cycle_rate)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| configured_interaction_slot_ratio | {} |",
        fmt_percent(metrics.configured_interaction_slot_ratio)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| configured_shared_address_ratio | {} |",
        fmt_percent(metrics.configured_shared_address_ratio)
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Strategy").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "- configured_interaction_slot_ratio = {}",
        fmt_percent(metrics.configured_interaction_slot_ratio)
    )
    .unwrap();
    writeln!(
        &mut out,
        "- configured_shared_address_ratio = {}",
        fmt_percent(metrics.configured_shared_address_ratio)
    )
    .unwrap();
    writeln!(
        &mut out,
        "- configured_independent_block_len_range = [{}, {}]",
        metrics.configured_independent_block_len_range.0,
        metrics.configured_independent_block_len_range.1
    )
    .unwrap();
    writeln!(
        &mut out,
        "- configured_interaction_block_len_range = [{}, {}]",
        metrics.configured_interaction_block_len_range.0,
        metrics.configured_interaction_block_len_range.1
    )
    .unwrap();
    writeln!(
        &mut out,
        "- prefer_focused_sequences = {}",
        metrics.prefer_focused_sequences
    )
    .unwrap();
    out
}

pub fn write_multicore_sync_metric_artifacts(
    output_dir: &Path,
    metrics: Option<&MulticoreSyncMetrics>,
) -> std::io::Result<()> {
    let Some(metrics) = metrics else {
        return Ok(());
    };

    fs::create_dir_all(output_dir)?;

    let json_path = output_dir.join("multicore_sync_metrics.json");
    let md_path = output_dir.join("multicore_sync_metrics.md");
    write_json_pretty(&json_path, metrics).map_err(|err| match err {
        ArtifactIoError::Io(source) => source,
        ArtifactIoError::Serialization(source) => std::io::Error::other(source),
    })?;
    write_string_file(&md_path, render_multicore_sync_metrics_markdown(metrics))?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtensionInstScaling {
    pub multiplier: f64,
    pub min_insts: usize,
    pub max_insts: usize,
}

impl Default for ExtensionInstScaling {
    fn default() -> Self {
        Self {
            multiplier: 1.0,
            min_insts: 20,
            max_insts: 1000,
        }
    }
}

fn deserialize_usize_range<'de, D>(deserializer: D) -> Result<(usize, usize), D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RangeRepr {
        Single(usize),
        Pair([usize; 2]),
    }

    let repr = RangeRepr::deserialize(deserializer)?;
    let (min, max) = match repr {
        RangeRepr::Single(value) => (value, value),
        RangeRepr::Pair([start, end]) => (start, end),
    };

    if min == 0 || min > max {
        return Err(de::Error::custom(format!(
            "usize range must satisfy 1 <= min <= max, got [{min}, {max}]"
        )));
    }

    Ok((min, max))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HartPhase {
    Init,
    Test,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct HartProgram {
    pub hart_id: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub init: Vec<InstructionSequence<RiscvInstruction>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub test: Vec<InstructionSequence<RiscvInstruction>>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub shared_init_sequence_indices: BTreeSet<usize>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub shared_test_sequence_indices: BTreeSet<usize>,
}

impl HartProgram {
    pub fn sequences(&self, phase: HartPhase) -> &[InstructionSequence<RiscvInstruction>] {
        match phase {
            HartPhase::Init => &self.init,
            HartPhase::Test => &self.test,
        }
    }

    pub fn sequences_mut(
        &mut self,
        phase: HartPhase,
    ) -> &mut Vec<InstructionSequence<RiscvInstruction>> {
        match phase {
            HartPhase::Init => &mut self.init,
            HartPhase::Test => &mut self.test,
        }
    }

    pub fn shared_sequence_indices(&self, phase: HartPhase) -> &BTreeSet<usize> {
        match phase {
            HartPhase::Init => &self.shared_init_sequence_indices,
            HartPhase::Test => &self.shared_test_sequence_indices,
        }
    }

    pub fn shared_sequence_indices_mut(&mut self, phase: HartPhase) -> &mut BTreeSet<usize> {
        match phase {
            HartPhase::Init => &mut self.shared_init_sequence_indices,
            HartPhase::Test => &mut self.shared_test_sequence_indices,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeneratedTestCase {
    pub config: TestCaseConfig,
    pub harts: Vec<HartProgram>,
    pub mem_range: HashMap<String, (u64, u64)>,
    pub extensions: IsaExtensions,
    pub multicore_sync_metrics: Option<MulticoreSyncMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedTestCase {
    pub config: TestCaseConfig,
    pub harts: Vec<HartProgram>,
    pub mem_range: HashMap<String, (u64, u64)>,
    pub extensions: IsaExtensions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multicore_sync_metrics: Option<MulticoreSyncMetrics>,
}

impl Serialize for GeneratedTestCase {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        PersistedTestCase {
            config: self.config.clone(),
            harts: self.harts.clone(),
            mem_range: self.mem_range.clone(),
            extensions: self.extensions.clone(),
            multicore_sync_metrics: self.multicore_sync_metrics.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GeneratedTestCase {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let persisted = PersistedTestCase::deserialize(deserializer)?;
        Ok(Self {
            config: persisted.config,
            harts: persisted.harts,
            mem_range: persisted.mem_range,
            extensions: persisted.extensions,
            multicore_sync_metrics: persisted.multicore_sync_metrics,
        })
    }
}

fn structured_to_phase_sequences(
    program: &StructuredTestInstructions,
) -> Vec<(Vec<InstructionSequence<RiscvInstruction>>, BTreeSet<usize>)> {
    program
        .harts
        .iter()
        .enumerate()
        .map(|(hart_idx, hart)| {
            let shared_flat = program
                .shared_base_address_indices
                .get(hart_idx)
                .cloned()
                .unwrap_or_default();
            let mut sequences = Vec::with_capacity(hart.len());
            let mut shared_sequences = BTreeSet::new();
            for (inst_idx, instruction) in hart.iter().cloned().enumerate() {
                sequences.push(InstructionSequence::new(instruction));
                if shared_flat.contains(&inst_idx) {
                    shared_sequences.insert(inst_idx);
                }
            }
            (sequences, shared_sequences)
        })
        .collect()
}

fn empty_hart_program(hart_id: u32) -> HartProgram {
    HartProgram {
        hart_id,
        init: Vec::new(),
        test: Vec::new(),
        shared_init_sequence_indices: BTreeSet::new(),
        shared_test_sequence_indices: BTreeSet::new(),
    }
}

fn ensure_hart_program_count(harts: &mut Vec<HartProgram>, hart_count: usize) {
    while harts.len() < hart_count {
        harts.push(empty_hart_program(harts.len() as u32));
    }
}

fn apply_phase_program(
    harts: &mut Vec<HartProgram>,
    phase: HartPhase,
    program: &StructuredTestInstructions,
) {
    let phase_harts = structured_to_phase_sequences(program);
    let hart_count = harts.len().max(phase_harts.len());
    ensure_hart_program_count(harts, hart_count);
    for (hart_idx, hart) in harts.iter_mut().enumerate().take(hart_count) {
        let (sequences, shared_sequences) = phase_harts
            .get(hart_idx)
            .cloned()
            .unwrap_or_else(|| (Vec::new(), BTreeSet::new()));
        *hart.sequences_mut(phase) = sequences;
        *hart.shared_sequence_indices_mut(phase) = shared_sequences;
    }
}

fn merge_hart_payloads<T>(
    init: Vec<(u32, Vec<T>)>,
    test: Vec<(u32, Vec<T>)>,
) -> Vec<(u32, Vec<T>)> {
    init.into_iter()
        .zip(test)
        .map(
            |((hart_id, mut init_values), (test_hart_id, test_values))| {
                debug_assert_eq!(hart_id, test_hart_id);
                init_values.extend(test_values);
                (hart_id, init_values)
            },
        )
        .collect()
}

fn phase_traceable_len(harts: &[HartProgram], phase: HartPhase) -> usize {
    harts
        .iter()
        .flat_map(|hart| hart.sequences(phase))
        .map(sequence_traceable_len)
        .sum()
}

fn phase_traceable_entry_at(
    harts: &[HartProgram],
    phase: HartPhase,
    traceable_index: usize,
) -> Option<RiscvInstruction> {
    let mut current = 0usize;
    for hart in harts {
        for sequence in hart.sequences(phase) {
            if let Some(entry) =
                sequence_traceable_entry_at(sequence, traceable_index, &mut current)
            {
                return Some(entry);
            }
        }
    }
    None
}

fn phase_rendered_pairs_per_hart(
    harts: &[HartProgram],
    phase: HartPhase,
    impl_ref: &str,
    mem_range: &HashMap<String, (u64, u64)>,
) -> Result<RenderedPairsPerHart, String> {
    Ok(
        materialized_phase_sequences_per_hart_for_impl(harts, phase, impl_ref, mem_range)?
            .into_iter()
            .map(|(hart_id, sequences)| {
                let rendered = sequences
                    .iter()
                    .flat_map(rendered_pairs_for_sequence)
                    .collect::<Vec<_>>();
                (hart_id, rendered)
            })
            .collect(),
    )
}

fn materialized_phase_sequences_per_hart_for_impl(
    harts: &[HartProgram],
    phase: HartPhase,
    impl_ref: &str,
    mem_range: &HashMap<String, (u64, u64)>,
) -> Result<TypedSequencesPerHart, String> {
    let (start, _end) = mem_range
        .get(impl_ref)
        .copied()
        .ok_or_else(|| format!("mem_range missing implementation {impl_ref}"))?;

    Ok(harts
        .iter()
        .map(|hart| {
            (
                hart.hart_id,
                materialize_hart_sequences_for_runtime(hart, phase, start as i64),
            )
        })
        .collect())
}

fn materialize_hart_sequences_for_runtime(
    hart: &HartProgram,
    phase: HartPhase,
    shared_offset: i64,
) -> Vec<InstructionSequence<RiscvInstruction>> {
    hart.sequences(phase)
        .iter()
        .map(|sequence| {
            let mut sequence = sequence.clone();
            for instruction in sequence
                .pre_instructions
                .iter_mut()
                .chain(std::iter::once(&mut sequence.instruction))
                .chain(sequence.post_instructions.iter_mut())
            {
                reoffset_instruction_for_runtime(instruction, shared_offset);
            }
            sequence
        })
        .collect()
}

fn reoffset_instruction_for_runtime(instruction: &mut RiscvInstruction, shared_offset: i64) {
    if let RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
        immediate, purpose, ..
    }) = instruction
        && *purpose == riscv_instruction_types::LoadImmediatePurpose::BaseAddress
    {
        let next = immediate.as_i64().saturating_add(shared_offset);
        *immediate = match immediate {
            riscv_instruction_types::XLenImmediate::X32(_) => {
                riscv_instruction_types::XLenImmediate::X32(next as i32)
            }
            riscv_instruction_types::XLenImmediate::X64(_) => {
                riscv_instruction_types::XLenImmediate::X64(next)
            }
        };
    }
}

impl GeneratedTestCase {
    pub fn impl_refs(&self) -> Vec<String> {
        self.mem_range
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn init_program(&self) -> StructuredTestInstructions {
        StructuredTestInstructions::from_hart_programs(&self.harts, HartPhase::Init)
    }

    pub fn test_program(&self) -> StructuredTestInstructions {
        StructuredTestInstructions::from_hart_programs(&self.harts, HartPhase::Test)
    }

    pub fn set_init_program(&mut self, program: StructuredTestInstructions) {
        self.replace_phase_program(HartPhase::Init, program);
    }

    pub fn set_test_program(&mut self, program: StructuredTestInstructions) {
        self.replace_phase_program(HartPhase::Test, program);
    }

    pub fn push_test_instruction_to_hart(
        &mut self,
        hart_idx: usize,
        instruction: RiscvInstruction,
    ) {
        self.ensure_hart(hart_idx);
        self.harts[hart_idx]
            .test
            .push(InstructionSequence::new(instruction));
    }

    pub fn push_test_pseudo_to_hart(&mut self, hart_idx: usize, pseudo: PseudoInstruction) {
        self.push_test_instruction_to_hart(hart_idx, pseudo.into());
    }

    pub fn hart_for_test_index(&self, target_test_index: usize) -> Option<u32> {
        let mut current = 0usize;
        for hart in &self.harts {
            for sequence in &hart.test {
                let seq_len = sequence_traceable_len(sequence);
                if target_test_index < current + seq_len {
                    return Some(hart.hart_id);
                }
                current += seq_len;
            }
        }
        None
    }

    fn ensure_hart(&mut self, hart_idx: usize) {
        ensure_hart_program_count(&mut self.harts, hart_idx + 1);
    }

    fn replace_phase_program(&mut self, phase: HartPhase, program: StructuredTestInstructions) {
        apply_phase_program(&mut self.harts, phase, &program);
    }

    pub fn with_test_sequences_by_hart(
        &self,
        mut sequences_by_hart: BTreeMap<u32, Vec<InstructionSequence<RiscvInstruction>>>,
    ) -> Self {
        let mut updated = self.clone();
        if let Some(max_hart_id) = sequences_by_hart.keys().copied().max() {
            updated.ensure_hart(max_hart_id as usize);
        }
        for hart in &mut updated.harts {
            hart.test = sequences_by_hart.remove(&hart.hart_id).unwrap_or_default();
            hart.shared_test_sequence_indices.clear();
        }
        updated
    }

    pub fn with_test_sequences_per_hart(&self, sequences_per_hart: TypedSequencesPerHart) -> Self {
        self.with_test_sequences_by_hart(sequences_per_hart.into_iter().collect())
    }

    fn materialized_phase_sequences_per_hart(
        &self,
        phase: HartPhase,
        impl_ref: &str,
    ) -> Result<TypedSequencesPerHart, String> {
        materialized_phase_sequences_per_hart_for_impl(
            &self.harts,
            phase,
            impl_ref,
            &self.mem_range,
        )
    }

    pub fn combined_rendered_lines_of(&self, impl_ref: &str) -> Option<Vec<String>> {
        Some(
            self.combined_rendered_pairs_per_hart_of(impl_ref)?
                .into_iter()
                .flat_map(|(_, pairs)| pairs.into_iter().map(|(line, _)| line))
                .collect(),
        )
    }

    pub fn combined_rendered_pairs_per_hart_of(
        &self,
        impl_ref: &str,
    ) -> Option<RenderedPairsPerHart> {
        let init =
            phase_rendered_pairs_per_hart(&self.harts, HartPhase::Init, impl_ref, &self.mem_range)
                .ok()?;
        let test =
            phase_rendered_pairs_per_hart(&self.harts, HartPhase::Test, impl_ref, &self.mem_range)
                .ok()?;
        Some(merge_hart_payloads(init, test))
    }

    pub fn combined_typed_sequences_per_hart_of(
        &self,
        impl_ref: &str,
    ) -> Option<TypedSequencesPerHart> {
        let init = self
            .materialized_phase_sequences_per_hart(HartPhase::Init, impl_ref)
            .ok()?;
        let test = self
            .materialized_phase_sequences_per_hart(HartPhase::Test, impl_ref)
            .ok()?;
        Some(merge_hart_payloads(init, test))
    }

    pub fn combined_mem_offsets_of(&self, impl_ref: &str) -> Option<Vec<Option<i64>>> {
        Some(
            self.combined_rendered_pairs_per_hart_of(impl_ref)?
                .into_iter()
                .flat_map(|(_, pairs)| pairs.into_iter().map(|(_, offset)| offset))
                .collect(),
        )
    }

    pub fn init_traceable_len(&self) -> usize {
        phase_traceable_len(&self.harts, HartPhase::Init)
    }

    pub fn test_traceable_len(&self) -> usize {
        phase_traceable_len(&self.harts, HartPhase::Test)
    }

    pub fn traceable_test_entry_at(&self, test_index: usize) -> Option<RiscvInstruction> {
        phase_traceable_entry_at(&self.harts, HartPhase::Test, test_index)
    }

    pub fn traceable_test_instruction_at(&self, test_index: usize) -> Option<String> {
        self.traceable_test_entry_at(test_index)
            .map(|instruction| instruction.to_string())
    }

    pub fn traceable_test_pair_at(&self, test_index: usize) -> Option<(String, Option<i64>)> {
        self.traceable_test_entry_at(test_index).map(|instruction| {
            let offset = instruction.offset_operand_value();
            (instruction.to_string(), offset)
        })
    }

    pub fn init_rendered_line_count_of(&self, impl_ref: &str) -> Option<usize> {
        Some(
            phase_rendered_pairs_per_hart(&self.harts, HartPhase::Init, impl_ref, &self.mem_range)
                .ok()?
                .into_iter()
                .map(|(_, pairs)| pairs.len())
                .sum(),
        )
    }

    pub fn init_rendered_line_at_of(&self, impl_ref: &str, index: usize) -> Option<String> {
        let mut current = 0usize;
        for (_hart_id, pairs) in
            phase_rendered_pairs_per_hart(&self.harts, HartPhase::Init, impl_ref, &self.mem_range)
                .ok()?
        {
            if index < current + pairs.len() {
                return pairs.into_iter().nth(index - current).map(|(line, _)| line);
            }
            current += pairs.len();
        }
        None
    }

    pub fn without_test_indices(&self, remove: &BTreeSet<usize>) -> Self {
        if remove.is_empty() {
            return self.clone();
        }

        let mut new_case = self.clone();
        let mut traceable_idx = 0usize;
        for hart in &mut new_case.harts {
            let original_shared = hart.shared_test_sequence_indices.clone();
            let mut new_test = Vec::new();
            let mut new_shared = BTreeSet::new();
            for (seq_idx, sequence) in hart.test.iter().enumerate() {
                let seq_start = traceable_idx;
                traceable_idx += sequence_traceable_len(sequence);
                let Some(reduced_sequence) =
                    sequence_without_removed_test_indices(sequence, seq_start, remove)
                else {
                    continue;
                };
                let new_seq_idx = new_test.len();
                new_test.push(reduced_sequence);
                if original_shared.contains(&seq_idx) {
                    new_shared.insert(new_seq_idx);
                }
            }
            hart.test = new_test;
            hart.shared_test_sequence_indices = new_shared;
        }
        new_case
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impls::compute_runtime_multicore_sync_metrics_from_context;
    use riscv_fuzz_plugin_api::FloatExtension;
    use riscv_fuzz_plugin_api::context::{
        ExecutionContextOutput, HartExecutionContextOutput, InstructionContext,
        LogicalInstructionContextOutput,
    };
    use riscv_fuzz_plugin_api::{InstructionTiming, PerfMetrics};
    use riscv_instruction::{RV32Extensions, RV32IInstructions, RV32Instruction};
    use riscv_instruction_types::{
        Immediate, InstructionSequence, IntegerRegister, PseudoInstruction, ValidatedValue,
        XLenImmediate,
    };

    /// Build a `Vec<HartProgram>` from a separate init/test pair.
    /// Used only by tests now that the runtime always carries `harts`
    /// directly; production code never round-trips through phase pairs.
    fn hart_programs_from_structured(
        init_program: &StructuredTestInstructions,
        test_program: &StructuredTestInstructions,
    ) -> Vec<HartProgram> {
        let init = structured_to_phase_sequences(init_program);
        let test = structured_to_phase_sequences(test_program);
        let hart_count = init.len().max(test.len());
        let mut harts = Vec::with_capacity(hart_count);
        for hart_idx in 0..hart_count {
            let (init_sequences, init_shared) = init
                .get(hart_idx)
                .cloned()
                .unwrap_or_else(|| (Vec::new(), BTreeSet::new()));
            let (test_sequences, test_shared) = test
                .get(hart_idx)
                .cloned()
                .unwrap_or_else(|| (Vec::new(), BTreeSet::new()));
            harts.push(HartProgram {
                hart_id: hart_idx as u32,
                init: init_sequences,
                test: test_sequences,
                shared_init_sequence_indices: init_shared,
                shared_test_sequence_indices: test_shared,
            });
        }
        harts
    }

    fn sample_config(hart_count: u32) -> TestCaseConfig {
        TestCaseConfig {
            isa_base: ISABase::Rv32,
            extension_insts_count: 1,
            extension_inst_scaling: None,
            mem_size: 64,
            float_extension: FloatExtension::None,
            unaligned_access_required: Some(false),
            random_config: RandomConfig {
                memory_access_config: MemoryAccessConfig {
                    memory_base_address_range: MemoryAccessConfig::default()
                        .memory_base_address_range,
                    memory_offset_immediate_range: (0, 0),
                    memory_base_register_number_range: (10, 15),
                },
                register_operand_config: RegisterOperandConfig {
                    integer_register_number_range: (16, 25),
                    floating_point_register_number_range: (16, 25),
                    vector_register_number_range: (16, 25),
                },
                ..RandomConfig::default()
            },
            seed: Some(7),
            hart_config: match hart_count {
                1 => HartConfig::Single,
                2 => HartConfig::Dual {
                    sync: MulticoreSyncConfig::default(),
                },
                _ => panic!("sample_config only supports 1 or 2 harts, got {hart_count}"),
            },
        }
    }

    fn addi(rd: u8, rs1: u8, imm: i16) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(
            riscv_instruction::RV32_I_ADDI {
                xd: IntegerRegister::new(rd).expect("valid rd"),
                xs1: IntegerRegister::new(rs1).expect("valid rs1"),
                imm: Immediate::<12>::new(imm.into()).expect("valid immediate"),
            },
        )))
    }

    fn li(
        rd: u8,
        immediate: i64,
        purpose: riscv_instruction_types::LoadImmediatePurpose,
    ) -> RiscvInstruction {
        RiscvInstruction::from(PseudoInstruction::LoadImmediate {
            rd: IntegerRegister::new(rd).expect("valid register"),
            immediate: XLenImmediate::X32(
                i32::try_from(immediate).expect("test immediate must fit rv32 xlen"),
            ),
            purpose,
        })
    }

    fn seq_with_pre(
        pre_instructions: Vec<RiscvInstruction>,
        instruction: RiscvInstruction,
    ) -> InstructionSequence<RiscvInstruction> {
        InstructionSequence {
            pre_instructions,
            instruction,
            post_instructions: Vec::new(),
        }
    }

    fn structured_program() -> StructuredTestInstructions {
        StructuredTestInstructions::from_harts(vec![
            vec![seq_with_pre(
                vec![li(
                    8,
                    0x20,
                    riscv_instruction_types::LoadImmediatePurpose::BaseAddress,
                )],
                addi(16, 0, 1),
            )],
            vec![InstructionSequence::new(addi(17, 16, 2))],
        ])
    }

    fn two_hart_program(
        hart0: Vec<RiscvInstruction>,
        hart1: Vec<RiscvInstruction>,
    ) -> StructuredTestInstructions {
        StructuredTestInstructions::from_harts(vec![
            hart0.into_iter().map(InstructionSequence::new).collect(),
            hart1.into_iter().map(InstructionSequence::new).collect(),
        ])
    }

    fn materialize_phase_for_test(
        testcase: &GeneratedTestCase,
        phase: HartPhase,
        impl_ref: &str,
    ) -> StructuredTestInstructions {
        let mut program = StructuredTestInstructions {
            harts: Vec::new(),
            shared_base_address_indices: Vec::new(),
        };
        let materialized = testcase
            .materialized_phase_sequences_per_hart(phase, impl_ref)
            .expect("test materialization should succeed");
        for (hart, (_hart_id, sequences)) in testcase.harts.iter().zip(materialized.into_iter()) {
            let mut hart_instructions = Vec::new();
            let mut shared = BTreeSet::new();
            let mut flat_idx = 0usize;
            let shared_sequences = hart.shared_sequence_indices(phase);
            for (seq_idx, sequence) in sequences.iter().enumerate() {
                for instruction in flatten_sequence(sequence) {
                    if shared_sequences.contains(&seq_idx)
                        && instruction_is_shared_base_address_marker(&instruction)
                    {
                        shared.insert(flat_idx);
                    }
                    hart_instructions.push(instruction);
                    flat_idx += 1;
                }
            }
            program.harts.push(hart_instructions);
            program.shared_base_address_indices.push(shared);
        }
        program.normalize();
        program
    }

    fn empty_context_hart_output(hart_id: u32) -> HartExecutionContextOutput {
        HartExecutionContextOutput {
            hart_id,
            instructions: Vec::new(),
        }
    }

    fn timed_logical_instruction(
        hart_id: u32,
        instruction_index: usize,
        local_instruction_index: usize,
        instruction: RiscvInstruction,
        cycle: u64,
    ) -> LogicalInstructionContextOutput {
        LogicalInstructionContextOutput {
            hart_id,
            instruction_index,
            local_instruction_index,
            instruction,
            trace_inst_id: riscv_fuzz_plugin_api::TraceInstId::new(
                hart_id,
                local_instruction_index,
            ),
            asm_instruction_indices: vec![local_instruction_index],
            exceptions: Vec::new(),
            register_writes: Vec::new(),
            memory_writes: Vec::new(),
            timing: InstructionTiming::new(Some(cycle), Some(cycle), Some(1)),
            context: InstructionContext::default(),
        }
    }

    #[test]
    fn structured_program_renders_to_expected_lines() {
        let program = structured_program();
        assert_eq!(
            program.rendered_lines(),
            vec![
                "li x8, 32".to_string(),
                "addi x16, x0, 1".to_string(),
                "addi x17, x16, 2".to_string(),
            ]
        );
        assert_eq!(
            program
                .rendered_pairs()
                .into_iter()
                .map(|(_, offset)| offset)
                .collect::<Vec<_>>(),
            vec![None, None, None]
        );
        assert_eq!(
            program.rendered_lines_per_hart(),
            vec![
                (
                    0u32,
                    vec!["li x8, 32".to_string(), "addi x16, x0, 1".to_string()],
                ),
                (1u32, vec!["addi x17, x16, 2".to_string()]),
            ]
        );
    }

    #[test]
    fn generated_testcase_serializes_structured_programs() {
        let init_program = StructuredTestInstructions::from_harts(vec![vec![seq_with_pre(
            vec![li(
                10,
                0,
                riscv_instruction_types::LoadImmediatePurpose::Generic,
            )],
            typed_nop_instruction(),
        )]]);

        let program = structured_program();

        let testcase = GeneratedTestCase {
            config: sample_config(2),
            harts: hart_programs_from_structured(&init_program, &program),
            mem_range: HashMap::from([("spike".to_string(), (0x1000, 0x103f))]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        };

        let json = serde_json::to_string_pretty(&testcase).expect("serialize testcase");
        assert!(json.contains("\"harts\""));
        assert!(json.contains("\"init\""));
        assert!(json.contains("\"test\""));
        assert!(!json.contains("\"test_insts\""));
        assert!(!json.contains("\"shared\""));

        let restored: GeneratedTestCase =
            serde_json::from_str(&json).expect("deserialize compact testcase");
        assert_eq!(restored.test_program(), testcase.test_program());
    }

    #[test]
    fn compact_persistence_normalizes_line_based_testcases() {
        let testcase = GeneratedTestCase {
            config: sample_config(1),
            harts: hart_programs_from_structured(
                &StructuredTestInstructions::default(),
                &two_hart_program(vec![addi(1, 0, 1)], vec![addi(2, 0, 2)]),
            ),
            mem_range: HashMap::from([("spike".to_string(), (0x1000, 0x103f))]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        };

        let compact = &testcase;
        assert_eq!(
            compact.test_program().rendered_lines(),
            testcase.test_program().rendered_lines()
        );
    }

    #[test]
    fn persistence_canonicalizes_shared_unoffset_programs() {
        let testcase = GeneratedTestCase {
            config: sample_config(2),
            harts: hart_programs_from_structured(
                &StructuredTestInstructions::default(),
                &two_hart_program(
                    vec![li(
                        5,
                        64,
                        riscv_instruction_types::LoadImmediatePurpose::BaseAddress,
                    )],
                    Vec::new(),
                ),
            ),
            mem_range: HashMap::from([
                ("spike".to_string(), (0x1000, 0x103f)),
                ("ovpsim".to_string(), (0x2000, 0x203f)),
            ]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        };

        let json = serde_json::to_string_pretty(&testcase).expect("serialize testcase");
        assert!(json.contains("\"harts\""));
        assert!(json.contains("\"hart_id\""));
        assert!(json.contains("\"immediate\""));
        assert!(json.contains("\"X32\": 64"));

        let restored: GeneratedTestCase =
            serde_json::from_str(&json).expect("deserialize testcase");
        assert_eq!(
            materialize_phase_for_test(&restored, HartPhase::Test, "spike"),
            two_hart_program(
                vec![li(
                    5,
                    0x1000 + 64,
                    riscv_instruction_types::LoadImmediatePurpose::BaseAddress,
                )],
                Vec::new(),
            )
        );
        assert_eq!(
            materialize_phase_for_test(&restored, HartPhase::Test, "ovpsim"),
            two_hart_program(
                vec![li(
                    5,
                    0x2000 + 64,
                    riscv_instruction_types::LoadImmediatePurpose::BaseAddress,
                )],
                Vec::new(),
            )
        );
    }

    #[test]
    fn dual_harts_share_the_same_materialized_base_region() {
        let mut test_program = two_hart_program(
            vec![li(
                5,
                8,
                riscv_instruction_types::LoadImmediatePurpose::BaseAddress,
            )],
            vec![li(
                6,
                16,
                riscv_instruction_types::LoadImmediatePurpose::BaseAddress,
            )],
        );
        test_program.mark_shared_base_address_index(1, 0);

        let testcase = GeneratedTestCase {
            config: sample_config(2),
            harts: hart_programs_from_structured(
                &StructuredTestInstructions::default(),
                &test_program,
            ),
            mem_range: HashMap::from([("spike".to_string(), (0x1000, 0x103f))]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        };

        let materialized = materialize_phase_for_test(&testcase, HartPhase::Test, "spike");
        let mut expected = two_hart_program(
            vec![li(
                5,
                0x1000 + 8,
                riscv_instruction_types::LoadImmediatePurpose::BaseAddress,
            )],
            vec![li(
                6,
                0x1000 + 16,
                riscv_instruction_types::LoadImmediatePurpose::BaseAddress,
            )],
        );
        expected.mark_shared_base_address_index(1, 0);
        assert_eq!(materialized, expected);
    }

    #[test]
    fn structured_test_instructions_rejects_legacy_hart_wrapper() {
        let json = r#"{
            "harts": [
                {
                    "instructions": []
                }
            ]
        }"#;

        let err = serde_json::from_str::<StructuredTestInstructions>(json)
            .expect_err("legacy hart wrapper should be rejected");

        assert!(
            err.to_string().contains("invalid type: map")
                || err.to_string().contains("expected a sequence")
        );
    }

    #[test]
    fn structured_test_instructions_reject_legacy_line_blocks() {
        let json = r#"{
            "lines": ["addi x1, x0, 1"],
            "mem_offsets": [null]
        }"#;

        let err = serde_json::from_str::<StructuredTestInstructions>(json)
            .expect_err("legacy line-based block should be rejected");

        assert!(err.to_string().contains("unknown field `lines`"));
    }

    #[test]
    fn removing_test_indices_preserves_structured_programs() {
        let program = structured_program();
        let testcase = GeneratedTestCase {
            config: sample_config(2),
            harts: hart_programs_from_structured(&StructuredTestInstructions::default(), &program),
            mem_range: HashMap::from([("spike".to_string(), (0x1000, 0x103f))]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        };

        let reduced = testcase.without_test_indices(&BTreeSet::from([0usize]));
        assert_eq!(
            reduced.test_program().rendered_lines(),
            vec![
                "addi x16, x0, 1".to_string(),
                "addi x17, x16, 2".to_string(),
            ]
        );
    }

    #[test]
    fn combined_rendered_lines_use_hart_major_order() {
        let init_program = StructuredTestInstructions::from_harts(vec![
            vec![InstructionSequence::new(addi(1, 0, 10))],
            vec![InstructionSequence::new(addi(2, 0, 20))],
        ]);
        let test_program = StructuredTestInstructions::from_harts(vec![
            vec![InstructionSequence::new(addi(3, 0, 30))],
            vec![InstructionSequence::new(addi(4, 0, 40))],
        ]);
        let testcase = GeneratedTestCase {
            config: sample_config(2),
            harts: hart_programs_from_structured(&init_program, &test_program),
            mem_range: HashMap::from([("spike".to_string(), (0x1000, 0x103f))]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        };

        assert_eq!(
            testcase.combined_rendered_lines_of("spike").unwrap(),
            vec![
                "addi x1, x0, 10".to_string(),
                "addi x3, x0, 30".to_string(),
                "addi x2, x0, 20".to_string(),
                "addi x4, x0, 40".to_string(),
            ]
        );
    }

    #[test]
    fn removing_test_indices_preserves_sequence_shape_when_partially_trimmed() {
        let testcase = GeneratedTestCase {
            config: sample_config(1),
            harts: vec![HartProgram {
                hart_id: 0,
                init: Vec::new(),
                test: vec![InstructionSequence {
                    pre_instructions: vec![addi(1, 0, 1), addi(2, 0, 2)],
                    instruction: addi(3, 0, 3),
                    post_instructions: vec![addi(4, 0, 4)],
                }],
                shared_init_sequence_indices: BTreeSet::new(),
                shared_test_sequence_indices: BTreeSet::from([0usize]),
            }],
            mem_range: HashMap::from([("spike".to_string(), (0x1000, 0x103f))]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        };

        let reduced = testcase.without_test_indices(&BTreeSet::from([1usize, 2usize]));
        assert_eq!(reduced.harts[0].test.len(), 1);
        assert_eq!(
            reduced.harts[0].shared_test_sequence_indices,
            BTreeSet::from([0usize])
        );
        assert_eq!(
            reduced.harts[0].test[0],
            InstructionSequence {
                pre_instructions: Vec::new(),
                instruction: addi(1, 0, 1),
                post_instructions: vec![addi(4, 0, 4)],
            }
        );
    }

    #[test]
    fn multicore_sync_metrics_exclude_fully_idle_gaps_from_denominator() {
        let testcase = GeneratedTestCase {
            config: sample_config(2),
            harts: Vec::new(),
            mem_range: HashMap::new(),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        };

        let output = ExecutionContextOutput {
            per_hart: BTreeMap::from([
                (0, empty_context_hart_output(0)),
                (1, empty_context_hart_output(1)),
            ]),
            framework_per_hart: BTreeMap::new(),
            logical_view: vec![
                timed_logical_instruction(0, 0, 0, addi(16, 0, 1), 10),
                timed_logical_instruction(1, 1, 0, addi(18, 0, 1), 12),
                timed_logical_instruction(1, 2, 1, addi(19, 0, 1), 18),
                timed_logical_instruction(0, 3, 1, addi(17, 0, 1), 20),
            ],
            mem_range: (0, 63),
            core_id: "ibex".to_string(),
            isa_base: ISABase::Rv32,
            perf: PerfMetrics::default(),
        };

        let metrics =
            compute_runtime_multicore_sync_metrics_from_context(&testcase, "ibex", &output)
                .expect("metrics");

        assert_eq!(metrics.cycle_window_start, 10);
        assert_eq!(metrics.cycle_window_end, 20);
        assert_eq!(
            metrics.cycle_window_end - metrics.cycle_window_start + 1,
            11
        );
        assert_eq!(metrics.hart0_active_cycle_count, 2);
        assert_eq!(metrics.hart1_active_cycle_count, 2);
        assert_eq!(metrics.covered_cycle_count, 4);
        assert_eq!(metrics.simultaneous_cycle_count, 0);
        assert_eq!(metrics.simultaneous_cycle_rate, 0.0);

        let markdown = render_multicore_sync_metrics_markdown(&metrics);
        assert!(markdown.contains(
            "Idle gaps inside [`cycle_window_start`, `cycle_window_end`] where neither hart is active are excluded"
        ));
    }

    #[test]
    fn random_config_default_matches_latest_upstream_defaults() {
        let random_config = RandomConfig::default();

        assert!(!random_config.append_fflags_readback);
        assert!(
            !random_config
                .csr_selection_config
                .machine_csr_operands_enabled
        );
        assert!(
            !random_config
                .csr_selection_config
                .supervisor_csr_operands_enabled
        );
        assert!(
            !random_config
                .csr_selection_config
                .hypervisor_csr_operands_enabled
        );
        assert!(
            !random_config
                .csr_selection_config
                .virtual_supervisor_csr_operands_enabled
        );
        assert!(!random_config.csr_selection_config.rnmi_csr_operands_enabled);
        assert!(
            !random_config
                .csr_selection_config
                .floating_csr_operands_enabled
        );
        assert!(
            !random_config
                .csr_selection_config
                .vector_csr_operands_enabled
        );
        assert!(
            !random_config
                .csr_selection_config
                .debug_csr_operands_enabled
        );
    }

    #[test]
    fn testcase_config_rejects_legacy_random_and_register_field_names() {
        let json = r#"{
            "isa_base": "Rv32",
            "extension_insts_count": 1,
            "mem_size": 64,
            "hart_config": { "kind": "single" },
            "mem_access_offset": [0, 0],
            "test_register_config": {
                "integer_register_range": [13, 20],
                "floating_point_register_range": [14, 21],
                "vector_register_range": [15, 22]
            },
            "temp_register_range": [10, 15]
        }"#;

        let err =
            serde_json::from_str::<TestCaseConfig>(json).expect_err("legacy fields should fail");

        let msg = err.to_string();
        assert!(
            msg.contains("mem_access_offset")
                || msg.contains("test_register_config")
                || msg.contains("temp_register_range")
        );
    }

    #[test]
    fn testcase_config_missing_random_config_uses_public_schema_defaults() {
        let json = r#"{
            "isa_base": "Rv32",
            "extension_insts_count": 1,
            "mem_size": 64,
            "hart_config": { "kind": "single" }
        }"#;

        let config: TestCaseConfig =
            serde_json::from_str(json).expect("missing random_config should deserialize");

        assert!(!config.random_config.append_fflags_readback);
        assert!(
            config
                .random_config
                .csr_selection_config
                .machine_csr_operands_enabled
        );
        assert!(
            !config
                .random_config
                .csr_selection_config
                .supervisor_csr_operands_enabled
        );
        assert!(
            !config
                .random_config
                .csr_selection_config
                .hypervisor_csr_operands_enabled
        );
        assert_eq!(
            config
                .random_config
                .memory_access_config
                .memory_offset_immediate_range,
            (0, 16)
        );
        assert_eq!(
            config
                .random_config
                .register_operand_config
                .integer_register_number_range,
            (13, 20)
        );
    }
}
