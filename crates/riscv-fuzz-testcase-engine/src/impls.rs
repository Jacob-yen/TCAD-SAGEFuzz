use std::{
    collections::{BTreeSet, HashMap, HashSet},
    convert::TryFrom,
    fs,
    path::Path,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use log::info;
use rand::{SeedableRng as _, rngs::StdRng, seq::SliceRandom};
use riscv_fuzz_plugin_api::context::{
    ExecutionContextOutput, InstructionContext, LogicalInstructionContextOutput, MemoryValueWidth,
};
use riscv_fuzz_plugin_api::metadata::FloatExtension;
use riscv_fuzz_plugin_api::{
    AlignmentMode, CoverageMode, ExecuteRequest, ExecutionOutput, ISABase, InstructionTiming,
    IsaExtensions, StreamingCollector,
};
use riscv_fuzz_plugin_manager::PluginManager;
use riscv_fuzz_toolchain::context_extract::ContextExtractionError;
use riscv_fuzz_toolchain::execution_context::{
    ContextBuildError, build_execution_context_output, build_partial_execution_context_output,
};
use riscv_instruction::{
    RV32Extensions, RV64Extensions, RiscvInstruction, generate_sequences_with_rng,
};
use riscv_instruction_types::{
    CsrSelectionConfig, InstructionSequence, LoadImmediatePurpose, PseudoInstruction, RandomConfig,
    RegisterBank, RegisterNumbering, RegisterOperandConfig, RuntimeMemoryBase,
    RuntimeMemoryOperand, RuntimeOperandAccess, RuntimeOperandUsage, RuntimeRegister,
    RuntimeRegisterAssignment, RuntimeRegisterOperand,
};
use thiserror::Error;

use crate::{
    GenerationError, MemRangeError,
    instruction::{
        generate_compact_initialization_instructions_seeded,
        generate_initialization_instructions_for_harts,
        generate_initialization_instructions_for_harts_seeded,
    },
    sequence_filter::{
        remove_instructions_requiring_unsupported_extensions, remove_special_instructions,
    },
    sequence_plan::{
        build_rv32_generation_counts, build_rv64_generation_counts, ordered_generation_plan,
    },
    testcase::{
        GeneratedTestCase, HartProgram, MulticoreSyncConfig, MulticoreSyncMetrics,
        StructuredTestInstructions, TestCaseConfig,
    },
};

const MULTICORE_AMO_MUL: usize = 8;
const MULTICORE_AMO_MIN: usize = 32;
const MULTICORE_LRSC_MUL: usize = 4;
const MULTICORE_LRSC_MIN: usize = 16;
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryAccessWindow {
    pub register_min: u64,
    pub register_max: u64,
    pub offset_min: i64,
    pub offset_max: i64,
    pub max_access_width: u64,
}

fn has_enabled_csr_pool(csr_selection_config: &CsrSelectionConfig) -> bool {
    csr_selection_config.machine_csr_operands_enabled
        || csr_selection_config.supervisor_csr_operands_enabled
        || csr_selection_config.hypervisor_csr_operands_enabled
        || csr_selection_config.virtual_supervisor_csr_operands_enabled
        || csr_selection_config.rnmi_csr_operands_enabled
        || csr_selection_config.floating_csr_operands_enabled
        || csr_selection_config.vector_csr_operands_enabled
        || csr_selection_config.debug_csr_operands_enabled
}

fn has_compressed_register_overlap(register_config: &RegisterOperandConfig) -> bool {
    let (min, max) = register_config.integer_register_number_range;
    let c_min = 8u8;
    let c_max = 15u8;
    min <= c_max && max >= c_min
}

fn normalize_sequence_alignment(
    mut sequences: Vec<InstructionSequence<RiscvInstruction>>,
    strict_base_alignment: Option<u64>,
) -> Vec<InstructionSequence<RiscvInstruction>> {
    let Some(align) = strict_base_alignment.filter(|align| *align > 1) else {
        return sequences;
    };

    for sequence in &mut sequences {
        crate::instruction_layout::align_mem_base_immediates(sequence, align);
    }

    sequences
}

fn testcase_rng(seed: Option<u64>) -> StdRng {
    if let Some(seed) = seed {
        info!("testcase generation: using explicit seed={seed}");
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_os_rng()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PhysicalRegisterKey {
    bank: RegisterBank,
    physical_number: u8,
}

impl PhysicalRegisterKey {
    fn assignment(self) -> RuntimeRegisterAssignment {
        RuntimeRegisterAssignment {
            bank: self.bank,
            number: self.physical_number,
            numbering: RegisterNumbering::Physical,
        }
    }
}

impl From<RuntimeRegister> for PhysicalRegisterKey {
    fn from(register: RuntimeRegister) -> Self {
        Self {
            bank: register.bank,
            physical_number: register.physical_number,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MulticoreCoupledSlot {
    hart0_slot: usize,
    hart1_slot: usize,
    register_match: bool,
    memory_match: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
struct MulticoreCouplingPlan {
    hart0_slot_count: usize,
    hart1_slot_count: usize,
    target_match_count: usize,
    target_shared_address_count: usize,
    matched_slots: Vec<MulticoreCoupledSlot>,
    coupled_slots: HashSet<usize>,
    shared_address_slots: HashSet<usize>,
}

#[derive(Debug, Default)]
struct CycleResourceUsage {
    registers: HashSet<String>,
    memory_addresses: HashSet<u64>,
}

#[derive(Debug, Error)]
pub enum MulticoreSyncMetricError {
    #[error("cycle-based multicore sync metrics require hart_count = 2, found {hart_count}")]
    UnsupportedHartCount { hart_count: u32 },

    #[error("missing typed testcase instructions or memory range for implementation {core_id}")]
    MissingInstructions { core_id: String },

    #[error("missing execution output for hart {hart_id}")]
    MissingHartOutput { hart_id: u32 },

    #[error("hart {hart_id} has no usable recorded instruction timings")]
    NoRecordedTiming { hart_id: u32 },

    #[error(
        "hart {hart_id} instruction {instruction_index} has no usable cycle interval: start={clk_start:?}, end={clk_end:?}, span={clk_span:?}"
    )]
    MissingTimingInterval {
        hart_id: u32,
        instruction_index: usize,
        clk_start: Option<u64>,
        clk_end: Option<u64>,
        clk_span: Option<u64>,
    },

    #[error(
        "failed to extract registers for hart {hart_id} instruction {instruction_index} `{instruction}`"
    )]
    RegisterExtraction {
        hart_id: u32,
        instruction_index: usize,
        instruction: String,
        #[source]
        source: ContextExtractionError,
    },

    #[error("failed to build sync-metric execution view for {core_id}: {source}")]
    ContextBuild {
        core_id: String,
        #[source]
        source: ContextBuildError,
    },
}

fn validate_multicore_sync_config(config: &MulticoreSyncConfig) -> Result<(), GenerationError> {
    for (field, value) in [
        ("interaction_slot_ratio", config.interaction_slot_ratio),
        ("shared_address_ratio", config.shared_address_ratio),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(GenerationError::InvalidMulticoreSyncRatio { field, value });
        }
    }

    Ok(())
}

fn target_slot_count(aligned_slot_count: usize, ratio: f64) -> usize {
    ((aligned_slot_count as f64) * ratio)
        .round()
        .clamp(0.0, aligned_slot_count as f64) as usize
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn register_overlap_score(register_rate: f64, joint_rate: f64) -> f64 {
    (register_rate * 0.65 + joint_rate * 0.35).clamp(0.0, 1.0)
}

fn memory_contention_score(
    same_address_rate: f64,
    same_address_given_both_memory_rate: f64,
    memory_jaccard: f64,
) -> f64 {
    (same_address_rate * 0.45 + same_address_given_both_memory_rate * 0.35 + memory_jaccard * 0.20)
        .clamp(0.0, 1.0)
}

fn timing_cycle_interval(timing: &InstructionTiming) -> Option<(u64, u64)> {
    match (timing.clk_start, timing.clk_end, timing.clk_span) {
        (Some(start), Some(end), _) if end >= start => Some((start, end)),
        (Some(start), None, Some(span)) if span > 0 => start
            .checked_add(span.saturating_sub(1))
            .map(|end| (start, end)),
        (None, Some(end), Some(span)) if span > 0 => end
            .checked_add(1)
            .and_then(|exclusive_end| exclusive_end.checked_sub(span))
            .map(|start| (start, end)),
        _ => None,
    }
}

fn effective_metric_cycle_interval(timing: &InstructionTiming) -> Option<(u64, u64)> {
    // Sync metrics must reflect the recorded trace window directly for every implementation.
    timing_cycle_interval(timing)
}

fn memory_value_width_bytes(width: &MemoryValueWidth) -> usize {
    match width {
        MemoryValueWidth::Byte => 1,
        MemoryValueWidth::Half => 2,
        MemoryValueWidth::Word => 4,
        MemoryValueWidth::Dword => 8,
    }
}

fn inferred_memory_access_width_bytes(mnemonic: &str, isa_base: ISABase) -> Option<usize> {
    if mnemonic.starts_with("amo") || mnemonic.starts_with("lr") || mnemonic.starts_with("sc") {
        return Some(if isa_base == ISABase::Rv64 && mnemonic.ends_with('d') {
            8
        } else {
            4
        });
    }

    let is_one_of = |candidates: &[&str]| candidates.contains(&mnemonic);

    if is_one_of(&["lb", "lbu", "sb", "flb", "fsb", "clb", "clbu", "csb"]) {
        return Some(1);
    }
    if is_one_of(&["lh", "lhu", "sh", "flh", "fsh", "clh", "clhu", "csh"]) {
        return Some(2);
    }
    if is_one_of(&[
        "lw", "lwu", "sw", "flw", "fsw", "clw", "csw", "clwsp", "cswsp",
    ]) {
        return Some(4);
    }
    if is_one_of(&["ld", "sd", "fld", "fsd", "cld", "csd", "cldsp", "csdsp"]) {
        return Some(8);
    }

    None
}

fn instruction_register_usage(
    instruction: &RiscvInstruction,
) -> Result<HashSet<String>, ContextExtractionError> {
    Ok(instruction
        .register_operands()
        .into_iter()
        .map(|operand| match operand.register.bank {
            RegisterBank::Integer => format!("x{}", operand.register.physical_number),
            RegisterBank::FloatingPoint => format!("f{}", operand.register.physical_number),
            RegisterBank::Vector => format!("v{}", operand.register.physical_number),
        })
        .collect())
}

fn instruction_memory_addresses_from_context(
    instruction: &RiscvInstruction,
    context: &InstructionContext,
    memory_changes: &[riscv_fuzz_plugin_api::MemValue],
    isa_base: ISABase,
) -> HashSet<u64> {
    let entries = &context.memory_context.entries;
    if entries.is_empty() {
        return memory_changes.iter().map(|entry| entry.addr).collect();
    }

    if let Some(width) = inferred_instruction_memory_access_width_bytes(instruction, isa_base)
        && let Some(entry) = entries
            .iter()
            .find(|entry| memory_value_width_bytes(&entry.width) == width)
    {
        return entry.addresses.iter().copied().collect();
    }

    entries
        .iter()
        .filter_map(|entry| entry.addresses.first().copied())
        .min()
        .into_iter()
        .collect()
}

fn build_cycle_usage_map(
    hart_id: u32,
    output: &[&LogicalInstructionContextOutput],
    isa_base: ISABase,
) -> Result<HashMap<u64, CycleResourceUsage>, MulticoreSyncMetricError> {
    let mut usage = HashMap::<u64, CycleResourceUsage>::new();
    let mut recorded_timing_count = 0usize;

    for (instruction_index, output_entry) in output.iter().enumerate() {
        let instruction = &output_entry.instruction;
        let context = &output_entry.context;
        let memory_changes = &output_entry.memory_writes;
        let timing = &output_entry.timing;
        let Some((clk_start, clk_end)) = effective_metric_cycle_interval(timing) else {
            return Err(MulticoreSyncMetricError::MissingTimingInterval {
                hart_id,
                instruction_index,
                clk_start: timing.clk_start,
                clk_end: timing.clk_end,
                clk_span: timing.clk_span,
            });
        };
        recorded_timing_count += 1;

        let registers = instruction_register_usage(instruction).map_err(|source| {
            MulticoreSyncMetricError::RegisterExtraction {
                hart_id,
                instruction_index,
                instruction: instruction.to_string(),
                source,
            }
        })?;
        let memory_addresses = instruction_memory_addresses_from_context(
            instruction,
            context,
            memory_changes,
            isa_base,
        );

        for cycle in clk_start..=clk_end {
            let cycle_usage = usage.entry(cycle).or_default();
            cycle_usage.registers.extend(registers.iter().cloned());
            cycle_usage
                .memory_addresses
                .extend(memory_addresses.iter().copied());
        }
    }

    if recorded_timing_count == 0 {
        return Err(MulticoreSyncMetricError::NoRecordedTiming { hart_id });
    }

    Ok(usage)
}

fn collect_hart_logical_view(
    output: &ExecutionContextOutput,
    hart_id: u32,
) -> Vec<&LogicalInstructionContextOutput> {
    let mut entries = output
        .logical_view
        .iter()
        .filter(|entry| entry.hart_id == hart_id)
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.local_instruction_index);
    entries
}

pub(crate) fn compute_runtime_multicore_sync_metrics_from_context(
    testcase: &GeneratedTestCase,
    impl_ref: &str,
    output: &ExecutionContextOutput,
) -> Result<MulticoreSyncMetrics, MulticoreSyncMetricError> {
    let hart_count = testcase.config.hart_count();
    if hart_count != 2 {
        return Err(MulticoreSyncMetricError::UnsupportedHartCount { hart_count });
    }

    let sync = testcase
        .config
        .multicore_sync_config()
        .expect("dual-hart testcase must carry multicore sync config");

    let _ = impl_ref;
    let hart0_output = output
        .per_hart
        .get(&0)
        .ok_or(MulticoreSyncMetricError::MissingHartOutput { hart_id: 0 })?;
    let hart1_output = output
        .per_hart
        .get(&1)
        .ok_or(MulticoreSyncMetricError::MissingHartOutput { hart_id: 1 })?;
    let hart0_logical_view = collect_hart_logical_view(output, 0);
    let hart1_logical_view = collect_hart_logical_view(output, 1);

    if hart0_output.hart_id != 0 || hart1_output.hart_id != 1 {
        return Err(MulticoreSyncMetricError::MissingHartOutput { hart_id: 0 });
    }

    let hart0_cycles = build_cycle_usage_map(0, &hart0_logical_view, output.isa_base)?;
    let hart1_cycles = build_cycle_usage_map(1, &hart1_logical_view, output.isa_base)?;

    let mut covered_cycles = HashSet::with_capacity(hart0_cycles.len() + hart1_cycles.len());
    covered_cycles.extend(hart0_cycles.keys().copied());
    covered_cycles.extend(hart1_cycles.keys().copied());

    let cycle_window_start = covered_cycles.iter().copied().min().unwrap_or(0);
    let cycle_window_end = covered_cycles.iter().copied().max().unwrap_or(0);

    let simultaneous_cycles: Vec<u64> = hart0_cycles
        .keys()
        .filter(|cycle| hart1_cycles.contains_key(cycle))
        .copied()
        .collect();

    let mut register_overlap_cycle_count = 0usize;
    let mut both_memory_cycle_count = 0usize;
    let mut same_address_cycle_count = 0usize;
    let mut joint_overlap_cycle_count = 0usize;
    let mut register_pair_intersection = 0usize;
    let mut register_pair_union = 0usize;
    let mut memory_pair_intersection = 0usize;
    let mut memory_pair_union = 0usize;

    for cycle in &simultaneous_cycles {
        let hart0_usage = hart0_cycles
            .get(cycle)
            .expect("cycle existence already checked for hart0");
        let hart1_usage = hart1_cycles
            .get(cycle)
            .expect("cycle existence already checked for hart1");

        let register_overlap_count = hart0_usage
            .registers
            .intersection(&hart1_usage.registers)
            .count();
        let register_union_count = hart0_usage.registers.union(&hart1_usage.registers).count();
        let same_address_count = hart0_usage
            .memory_addresses
            .intersection(&hart1_usage.memory_addresses)
            .count();
        let memory_union_count = hart0_usage
            .memory_addresses
            .union(&hart1_usage.memory_addresses)
            .count();

        if register_overlap_count > 0 {
            register_overlap_cycle_count += 1;
        }

        let both_memory =
            !hart0_usage.memory_addresses.is_empty() && !hart1_usage.memory_addresses.is_empty();
        if both_memory {
            both_memory_cycle_count += 1;
        }
        if same_address_count > 0 {
            same_address_cycle_count += 1;
        }
        if register_overlap_count > 0 && same_address_count > 0 {
            joint_overlap_cycle_count += 1;
        }

        register_pair_intersection += register_overlap_count;
        register_pair_union += register_union_count;
        memory_pair_intersection += same_address_count;
        memory_pair_union += memory_union_count;
    }

    let covered_cycle_count = covered_cycles.len();
    let simultaneous_cycle_count = simultaneous_cycles.len();
    let simultaneous_cycle_rate = rate(simultaneous_cycle_count, covered_cycle_count);
    let register_overlap_cycle_rate = rate(register_overlap_cycle_count, simultaneous_cycle_count);
    let same_address_cycle_rate = rate(same_address_cycle_count, simultaneous_cycle_count);
    let same_address_given_both_memory_cycle_rate =
        rate(same_address_cycle_count, both_memory_cycle_count);
    let joint_overlap_cycle_rate = rate(joint_overlap_cycle_count, simultaneous_cycle_count);
    let register_overlap_cycle_jaccard = rate(register_pair_intersection, register_pair_union);
    let memory_overlap_cycle_jaccard = rate(memory_pair_intersection, memory_pair_union);
    let cycle_contention_score = (simultaneous_cycle_rate * 0.20
        + memory_contention_score(
            same_address_cycle_rate,
            same_address_given_both_memory_cycle_rate,
            memory_overlap_cycle_jaccard,
        ) * 0.55
        + register_overlap_score(register_overlap_cycle_rate, joint_overlap_cycle_rate) * 0.25)
        .clamp(0.0, 1.0);

    Ok(MulticoreSyncMetrics {
        hart0_instruction_count: hart0_logical_view.len(),
        hart1_instruction_count: hart1_logical_view.len(),
        hart0_timed_instruction_count: hart0_logical_view.len(),
        hart1_timed_instruction_count: hart1_logical_view.len(),
        timing_coverage_rate: 1.0,
        cycle_window_start,
        cycle_window_end,
        covered_cycle_count,
        hart0_active_cycle_count: hart0_cycles.len(),
        hart1_active_cycle_count: hart1_cycles.len(),
        simultaneous_cycle_count,
        simultaneous_cycle_rate,
        register_overlap_cycle_count,
        both_memory_cycle_count,
        same_address_cycle_count,
        joint_overlap_cycle_count,
        register_overlap_cycle_rate,
        register_overlap_cycle_jaccard,
        same_address_cycle_rate,
        same_address_given_both_memory_cycle_rate,
        memory_overlap_cycle_jaccard,
        joint_overlap_cycle_rate,
        cycle_contention_score,
        configured_interaction_slot_ratio: sync.interaction_slot_ratio,
        configured_shared_address_ratio: sync.shared_address_ratio,
        configured_independent_block_len_range: sync.independent_block_len_range,
        configured_interaction_block_len_range: sync.interaction_block_len_range,
        prefer_focused_sequences: sync.prefer_focused_sequences,
    })
}

pub fn compute_runtime_multicore_sync_metrics(
    testcase: &GeneratedTestCase,
    impl_ref: &str,
    output: &ExecutionOutput,
) -> Result<MulticoreSyncMetrics, MulticoreSyncMetricError> {
    let per_hart_typed = testcase
        .combined_typed_sequences_per_hart_of(impl_ref)
        .ok_or_else(|| MulticoreSyncMetricError::MissingInstructions {
            core_id: impl_ref.to_string(),
        })?;
    let mem_range = testcase.mem_range.get(impl_ref).copied().ok_or_else(|| {
        MulticoreSyncMetricError::MissingInstructions {
            core_id: impl_ref.to_string(),
        }
    })?;
    let context_output = build_execution_context_output_with_weak_fallback(
        output.clone(),
        mem_range,
        &per_hart_typed,
    )
    .map_err(|source| MulticoreSyncMetricError::ContextBuild {
        core_id: impl_ref.to_string(),
        source,
    })?;
    compute_runtime_multicore_sync_metrics_from_context(testcase, impl_ref, &context_output)
}

fn filter_extensions_for_float_mode(
    mut extensions: IsaExtensions,
    float_extension: FloatExtension,
) -> IsaExtensions {
    fn extension_name<E: std::fmt::Debug>(extension: &E) -> String {
        format!("{extension:?}").to_ascii_lowercase()
    }

    fn is_float_related_name(name: &str) -> bool {
        matches!(name, "f" | "d" | "q" | "zfa" | "zcf" | "zcd")
            || name.starts_with("zfh")
            || name.ends_with("inx")
    }

    fn keep_by_name(name: &str, float_extension: FloatExtension) -> bool {
        match float_extension {
            FloatExtension::FD => true,
            FloatExtension::F => {
                !matches!(name, "d" | "q" | "zcd")
                    && !name.ends_with("dinx")
                    && !name.starts_with("zfh")
            }
            FloatExtension::None => !is_float_related_name(name),
        }
    }

    match &mut extensions {
        IsaExtensions::Rv32(values) => {
            values.retain(|ext| keep_by_name(&extension_name(ext), float_extension))
        }
        IsaExtensions::Rv64(values) => {
            values.retain(|ext| keep_by_name(&extension_name(ext), float_extension))
        }
    }

    extensions
}

pub fn extensions_for_config(
    impls: &[String],
    testcase_config: &TestCaseConfig,
) -> Result<IsaExtensions, GenerationError> {
    let plugin_manager = PluginManager::global();
    let isa_profile = plugin_manager.isa_profile(&impls[0], testcase_config.isa_base);

    match testcase_config.float_extension {
        FloatExtension::None => {}
        FloatExtension::F => {
            if !isa_profile.supports_float_choice(FloatExtension::F) {
                return Err(GenerationError::RequestedFloatUnsupported {
                    isa_base: testcase_config.isa_base,
                    choice: FloatExtension::F,
                });
            }
        }
        FloatExtension::FD => {
            if !isa_profile.supports_float_choice(FloatExtension::FD) {
                return Err(GenerationError::RequestedDoubleUnsupported {
                    isa_base: testcase_config.isa_base,
                });
            }
        }
    }

    Ok(filter_extensions_for_float_mode(
        plugin_manager.extensions_for_impls(impls, testcase_config.isa_base),
        testcase_config.float_extension,
    ))
}

/// 为给定的 isa_base 和请求的内存大小，计算每个实现的合理的内存范围
///
/// # 参数要求
/// 1. 请求的内存大小不超过所有实现中最小的可用内存大小
/// 2. 请求的内存大小是字对齐的(是word_size的整数倍)
/// 3. 请求的内存大小大于等于该isa_base最大指令访问宽度
///
/// # 保证
/// 1. 所有实现都有相同的内存大小
/// 2. 内存范围对齐且有效
/// 3. 内存范围足够大，可以执行任何指令
///
/// # 注意
/// 使用者需要确保指令访问的地址在范围内。例如，如果范围是闭区间 [0x80000000, 0x8000001F]（大小32字节），
/// 指令 `LW x1, 0(x2)` 中 x2 的值应该在 [0x80000000, 0x8000001C] 范围内，这样访问地址
/// 0x8000001C + 4 = 0x80000020 才不会越界。
/// 这个限制通过 `MemoryAccessConfig::memory_base_address_range` 来保证。
///
/// # 返回值
/// HashMap 的值是闭区间 [start, end]，即 start 和 end 都是合法的可访问地址
pub fn mem_range(
    impls: &[String],
    isa_base: ISABase,
    mem_size: u64,
) -> Result<HashMap<String, (u64, u64)>, MemRangeError> {
    let plugin_manager = PluginManager::global();
    let min_mem_size = plugin_manager
        .min_program_mem_size(impls)
        .ok_or(MemRangeError::NoImplementations)?;
    if mem_size > min_mem_size {
        return Err(MemRangeError::MemorySizeExceedsAvailable {
            requested: mem_size,
            available: min_mem_size,
        });
    }

    let word_size = isa_base.word_size() as u64;
    if !mem_size.is_multiple_of(word_size) {
        return Err(MemRangeError::MemoryNotWordAligned {
            size: mem_size,
            word_size,
        });
    }

    let max_access_width = isa_base.instruction_max_access_width() as u64;
    if mem_size < max_access_width {
        return Err(MemRangeError::MemorySizeTooSmall {
            size: mem_size,
            width: max_access_width,
        });
    }

    let mut result = HashMap::new();
    for impl_ref in impls {
        let (_start, end) = plugin_manager.program_mem_range(impl_ref);
        let unaligned_offset = end - mem_size;
        let aligned_offset = (unaligned_offset / word_size) * word_size;
        let mem_end = aligned_offset + mem_size - 1;
        result.insert(impl_ref.clone(), (aligned_offset, mem_end));
    }

    for (start, end) in result.values() {
        let actual_size = end - start + 1;
        if actual_size != mem_size {
            return Err(MemRangeError::MemoryRangeSizeMismatch {
                actual: actual_size,
                expected: mem_size,
            });
        }
    }

    Ok(result)
}

pub(crate) fn base_memory_access_window(
    testcase_config: &TestCaseConfig,
) -> Result<MemoryAccessWindow, GenerationError> {
    let max_access_width = testcase_config.isa_base.instruction_max_access_width() as u64;
    let (offset_min, offset_max) = testcase_config
        .random_config
        .memory_access_config
        .memory_offset_immediate_range;
    let register_min = if offset_min < 0 {
        let neg = offset_min
            .checked_neg()
            .ok_or(GenerationError::InvalidMemAccessOffset {
                min: offset_min,
                max: offset_max,
                mem_size: testcase_config.mem_size,
                width: max_access_width,
            })?;
        u64::try_from(neg).map_err(|_| GenerationError::InvalidMemAccessOffset {
            min: offset_min,
            max: offset_max,
            mem_size: testcase_config.mem_size,
            width: max_access_width,
        })?
    } else {
        0
    };
    let mem_upper_limit =
        testcase_config
            .mem_size
            .checked_sub(1)
            .ok_or(GenerationError::InvalidMemAccessOffset {
                min: offset_min,
                max: offset_max,
                mem_size: testcase_config.mem_size,
                width: max_access_width,
            })?;
    let access_limit_i128 = i128::from(testcase_config.mem_size)
        - i128::from(max_access_width)
        - i128::from(offset_max);
    if access_limit_i128 < 0 {
        return Err(GenerationError::InvalidMemAccessOffset {
            min: offset_min,
            max: offset_max,
            mem_size: testcase_config.mem_size,
            width: max_access_width,
        });
    }
    let register_max_i128 = access_limit_i128.min(i128::from(mem_upper_limit));
    if register_max_i128 < i128::from(register_min) {
        return Err(GenerationError::InvalidMemAccessOffset {
            min: offset_min,
            max: offset_max,
            mem_size: testcase_config.mem_size,
            width: max_access_width,
        });
    }

    Ok(MemoryAccessWindow {
        register_min,
        register_max: register_max_i128 as u64,
        offset_min,
        offset_max,
        max_access_width,
    })
}

pub(crate) fn random_config_for_window(
    testcase_config: &TestCaseConfig,
    register_min: u64,
    register_max: u64,
) -> RandomConfig {
    let mut config = testcase_config.random_config.clone();
    config.isa_profile = match testcase_config.isa_base {
        ISABase::Rv32 => riscv_instruction_types::IsaProfile::rv32(),
        ISABase::Rv64 => riscv_instruction_types::IsaProfile::rv64(),
    };
    config.memory_access_config.memory_base_address_range = (register_min, register_max);
    config
}

pub fn generate_random_testcase(
    impls: &[String],
    testcase_config: TestCaseConfig,
) -> Result<GeneratedTestCase, GenerationError> {
    let extensions = extensions_for_config(impls, &testcase_config)?;
    generate_random_testcase_with_extensions(impls, testcase_config, extensions)
}

pub fn generate_hardware_constraints_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let mut testcase =
        generate_random_testcase_with_extensions(impls, testcase_config, extensions)?;
    crate::rvv_constraints::constrain_random_rvv_testcase(&mut testcase)?;
    Ok(testcase)
}

pub fn generate_hardware_obligation_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let mut testcase =
        generate_random_testcase_with_extensions(impls, testcase_config, extensions)?;
    crate::rvv_model::inject_hardware_rvv_blocks_with_flavor(
        &mut testcase,
        impls,
        crate::rvv_model::HardwareRvvInjectionFlavor::AmbientRandom,
    )?;
    Ok(testcase)
}

pub fn generate_hardware_constraints_obligation_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let mut testcase =
        generate_random_testcase_with_extensions(impls, testcase_config, extensions)?;
    let injection = crate::rvv_model::inject_hardware_rvv_blocks_with_flavor(
        &mut testcase,
        impls,
        crate::rvv_model::HardwareRvvInjectionFlavor::ContractConstrained,
    )?;
    crate::rvv_constraints::constrain_random_rvv_testcase_after_first_hart_prefix(
        &mut testcase,
        injection.inserted_instructions,
    )?;
    Ok(testcase)
}

pub fn generate_hardware_stateflow_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let mut testcase =
        generate_random_testcase_with_extensions(impls, testcase_config, extensions)?;
    crate::rvv_stateflow::remove_ambient_rvv_test_instructions(&mut testcase);
    crate::rvv_stateflow::inject_hardware_rvv_stateflow_with_flavor(
        &mut testcase,
        impls,
        crate::rvv_stateflow::RvvStateFlowFlavor::StateFlowOnly,
    )?;
    Ok(testcase)
}

pub fn generate_hardware_stateflow_constraints_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let mut testcase =
        generate_random_testcase_with_extensions(impls, testcase_config, extensions)?;
    crate::rvv_stateflow::remove_ambient_rvv_test_instructions(&mut testcase);
    crate::rvv_stateflow::inject_hardware_rvv_stateflow_with_flavor(
        &mut testcase,
        impls,
        crate::rvv_stateflow::RvvStateFlowFlavor::StateFlowConstrained,
    )?;
    Ok(testcase)
}

const ARA_DIRECT_MIN_MEM_SIZE: u64 = 2048;

fn ara_direct_testcase_config(mut testcase_config: TestCaseConfig) -> TestCaseConfig {
    testcase_config.mem_size = testcase_config.mem_size.max(ARA_DIRECT_MIN_MEM_SIZE);
    testcase_config
}

pub fn generate_hardware_ara_safe_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let mut testcase = generate_random_testcase_with_extensions(
        impls,
        ara_direct_testcase_config(testcase_config),
        extensions,
    )?;
    crate::rvv_stateflow::remove_ambient_rvv_init_instructions(&mut testcase);
    crate::rvv_stateflow::clear_ambient_test_instructions(&mut testcase);
    crate::rvv_stateflow::inject_hardware_rvv_stateflow_with_flavor(
        &mut testcase,
        impls,
        crate::rvv_stateflow::RvvStateFlowFlavor::AraSafe,
    )?;
    crate::rvv_stateflow::inject_vector_state_enable(&mut testcase)?;
    Ok(testcase)
}

fn generate_hardware_ara_risk_flavor_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
    flavor: crate::rvv_stateflow::RvvStateFlowFlavor,
) -> Result<GeneratedTestCase, GenerationError> {
    let mut testcase = generate_random_testcase_with_extensions(
        impls,
        ara_direct_testcase_config(testcase_config),
        extensions,
    )?;
    crate::rvv_stateflow::remove_ambient_rvv_init_instructions(&mut testcase);
    crate::rvv_stateflow::remove_ambient_rvv_test_instructions(&mut testcase);
    crate::rvv_stateflow::inject_hardware_rvv_stateflow_with_flavor(&mut testcase, impls, flavor)?;
    crate::rvv_stateflow::inject_vector_state_enable(&mut testcase)?;
    Ok(testcase)
}

fn env_flag_enabled(name: &str) -> bool {
    let Ok(raw) = std::env::var(name) else {
        return false;
    };
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn ara_append_sage_to_riscsmith_enabled() -> bool {
    env_flag_enabled("RISCV_FUZZ_ARA_APPEND_SAGE_TO_RISCSMITH")
}

pub fn generate_hardware_ara_risk_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    if ara_append_sage_to_riscsmith_enabled() {
        let coverage_focused = env_flag_enabled("RISCV_FUZZ_RVV_COVERAGE_PROFILE");
        let mut testcase = if coverage_focused {
            let mut testcase = generate_random_testcase_with_extensions(
                impls,
                ara_direct_testcase_config(testcase_config),
                extensions,
            )?;
            crate::rvv_stateflow::remove_ambient_rvv_init_instructions(&mut testcase);
            crate::rvv_stateflow::remove_ambient_complex_memory_test_instructions(&mut testcase);
            crate::rvv_constraints::constrain_random_rvv_testcase(&mut testcase)?;
            crate::rvv_stateflow::inject_vector_state_enable(&mut testcase)?;
            testcase
        } else {
            generate_hardware_ara_risk_no_constraints_no_sequence_testcase_with_extensions(
                impls,
                testcase_config,
                extensions,
            )?
        };
        if coverage_focused {
            crate::rvv_stateflow::inject_hardware_rvv_stateflow_with_flavor(
                &mut testcase,
                impls,
                crate::rvv_stateflow::RvvStateFlowFlavor::AraGeneric,
            )?;
        } else {
            crate::rvv_stateflow::append_hardware_rvv_stateflow_with_flavor(
                &mut testcase,
                impls,
                crate::rvv_stateflow::RvvStateFlowFlavor::AraGeneric,
            )?;
        }
        return Ok(testcase);
    }

    generate_hardware_ara_risk_flavor_testcase_with_extensions(
        impls,
        testcase_config,
        extensions,
        crate::rvv_stateflow::RvvStateFlowFlavor::AraGeneric,
    )
}

pub fn generate_hardware_ara_risk_no_constraints_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    generate_hardware_ara_risk_flavor_testcase_with_extensions(
        impls,
        testcase_config,
        extensions,
        crate::rvv_stateflow::RvvStateFlowFlavor::AraNoConstraints,
    )
}

pub fn generate_hardware_ara_risk_no_sequence_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let mut testcase = generate_random_testcase_with_extensions(
        impls,
        ara_direct_testcase_config(testcase_config),
        extensions,
    )?;
    crate::rvv_stateflow::remove_ambient_rvv_init_instructions(&mut testcase);
    crate::rvv_constraints::constrain_random_rvv_testcase(&mut testcase)?;
    crate::rvv_stateflow::inject_vector_state_enable(&mut testcase)?;
    Ok(testcase)
}

pub fn generate_hardware_ara_risk_no_constraints_no_sequence_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let mut testcase = generate_random_testcase_with_extensions(
        impls,
        ara_direct_testcase_config(testcase_config),
        extensions,
    )?;
    crate::rvv_stateflow::remove_ambient_rvv_init_instructions(&mut testcase);
    crate::rvv_stateflow::retain_only_ambient_rvv_test_instructions(&mut testcase);
    crate::rvv_stateflow::inject_vector_state_enable(&mut testcase)?;
    crate::rvv_stateflow::inject_ambient_rvv_config_setup(&mut testcase)?;
    Ok(testcase)
}

pub fn generate_hardware_rvv_random_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    generate_random_testcase_with_extensions(impls, testcase_config, extensions)
}

pub fn generate_hardware_rvv_constraints_only_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    generate_hardware_constraints_testcase_with_extensions(impls, testcase_config, extensions)
}

pub fn generate_hardware_rvv_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    generate_hardware_constraints_obligation_testcase_with_extensions(
        impls,
        testcase_config,
        extensions,
    )
}

pub fn generate_random_testcase_with_extensions(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    generate_random_testcase_with_extensions_internal(
        impls,
        testcase_config,
        extensions,
        None,
        false,
    )
}

/// Build the usual executable harness and seeded initial state without
/// generating an ambient RV64 test-instruction stream.
pub(crate) fn generate_model_skeleton_with_seeded_initialization(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let seed =
        testcase_config
            .seed
            .ok_or_else(|| GenerationError::StructuredInstructionBuildFailed {
                message: "model-only generation requires an explicit seed".to_string(),
            })?;
    generate_random_testcase_with_extensions_internal(
        impls,
        testcase_config,
        extensions,
        Some(seed ^ 0x5643_5346_494e_4954),
        false,
    )
}

pub(crate) fn generate_model_skeleton_with_compact_initialization(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
) -> Result<GeneratedTestCase, GenerationError> {
    let seed =
        testcase_config
            .seed
            .ok_or_else(|| GenerationError::StructuredInstructionBuildFailed {
                message: "model-only generation requires an explicit seed".to_string(),
            })?;
    generate_random_testcase_with_extensions_internal(
        impls,
        testcase_config,
        extensions,
        Some(seed ^ 0x5643_5346_494e_4954),
        true,
    )
}

fn generate_random_testcase_with_extensions_internal(
    impls: &[String],
    testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
    initialization_seed: Option<u64>,
    compact_initialization: bool,
) -> Result<GeneratedTestCase, GenerationError> {
    if let Some(sync) = testcase_config.multicore_sync_config() {
        validate_multicore_sync_config(sync)?;
    }

    let mut rng = testcase_rng(testcase_config.seed);
    let hart_count = testcase_config.hart_count();
    let is_dual_hart = testcase_config.is_dual_hart();

    let mem_range = mem_range(impls, testcase_config.isa_base, testcase_config.mem_size)?;

    let mem_window = base_memory_access_window(&testcase_config)?;
    let (mut register_min, mut register_max) = constrain_multicore_register_window(
        &mut rng,
        is_dual_hart,
        mem_window.register_min,
        mem_window.register_max,
        testcase_config.isa_base.word_size() as u64,
    );

    // If unaligned accesses are disallowed, force base registers onto a conservative
    // alignment boundary so loads/stores/AMOs don't spuriously trap.
    if testcase_config.unaligned_access_required == Some(false) {
        let align = mem_window.max_access_width.max(1);
        register_min = register_min.div_ceil(align).saturating_mul(align);
        register_max = (register_max / align).saturating_mul(align);
        if register_max < register_min {
            return Err(GenerationError::InvalidMemAccessOffset {
                min: mem_window.offset_min,
                max: mem_window.offset_max,
                mem_size: testcase_config.mem_size,
                width: mem_window.max_access_width,
            });
        }
    }

    let strict_base_alignment = (testcase_config.unaligned_access_required == Some(false))
        .then_some(testcase_config.isa_base.instruction_max_access_width() as u64);

    let config = random_config_for_window(&testcase_config, register_min, register_max);
    let csr_selection_config = config.csr_selection_config;

    match testcase_config.isa_base {
        ISABase::Rv32 => {
            info!(
                "Extensions for RV32: {:?}",
                extensions
                    .as_rv32()
                    .expect("RV32 extensions expected for RV32 testcase")
            );
            let mut counts = build_rv32_generation_counts(
                extensions
                    .as_rv32()
                    .expect("RV32 extensions expected for RV32 testcase"),
                &testcase_config,
            )?;
            apply_multicore_atomic_bias_rv32(is_dual_hart, &mut counts);

            if !has_enabled_csr_pool(&csr_selection_config)
                && counts.remove(&RV32Extensions::Zicsr).is_some()
            {
                info!(
                    "Skipping RV32 Zicsr generation because csr_selection_config enables no writable CSR groups"
                );
            }

            if !has_compressed_register_overlap(&config.register_operand_config)
                && counts.remove(&RV32Extensions::C).is_some()
            {
                info!(
                    "Skipping RV32 C generation because integer register range {:?} has no overlap with x8..x15",
                    config.register_operand_config.integer_register_number_range
                );
            }
            let plan = ordered_generation_plan(&counts);

            let mut hart0_insts = generate_sequences_with_rng(&mut rng, &plan, &config)?;

            hart0_insts = remove_special_instructions(hart0_insts);
            hart0_insts = remove_instructions_requiring_unsupported_extensions(
                hart0_insts,
                &extensions,
                testcase_config.isa_base,
            );
            hart0_insts = remove_rv32_toolchain_unsupported_sequences(hart0_insts);
            if uses_cva6(impls) {
                hart0_insts = remove_cva6_unsupported_sequences(hart0_insts);
            }
            hart0_insts = normalize_sequence_alignment(hart0_insts, strict_base_alignment);

            let mut hart1_insts = Vec::new();
            if is_dual_hart {
                let mut generated = generate_sequences_with_rng(&mut rng, &plan, &config)?;
                generated = remove_special_instructions(generated);
                generated = remove_instructions_requiring_unsupported_extensions(
                    generated,
                    &extensions,
                    testcase_config.isa_base,
                );
                generated = remove_rv32_toolchain_unsupported_sequences(generated);
                if uses_cva6(impls) {
                    generated = remove_cva6_unsupported_sequences(generated);
                }
                hart1_insts = normalize_sequence_alignment(generated, strict_base_alignment);
            }

            let (hart1_insts, multicore_plan) = match testcase_config.multicore_sync_config() {
                Some(sync) => {
                    couple_multicore_hart_streams(&hart0_insts, &hart1_insts, sync, &mut rng)
                }
                None => (hart1_insts, MulticoreCouplingPlan::default()),
            };
            let mut init_program = match initialization_seed {
                Some(seed) if compact_initialization => {
                    generate_compact_initialization_instructions_seeded(
                        &extensions,
                        testcase_config.isa_base,
                        &config,
                        hart_count,
                        seed,
                    )?
                }
                Some(seed) => generate_initialization_instructions_for_harts_seeded(
                    &extensions,
                    testcase_config.isa_base,
                    testcase_config.mem_size,
                    &config,
                    hart_count,
                    seed,
                )?,
                None => generate_initialization_instructions_for_harts(
                    &extensions,
                    testcase_config.isa_base,
                    testcase_config.mem_size,
                    &config,
                    hart_count,
                )?,
            };
            remove_special_asm_lines(&mut init_program);

            let harts = canonical_hart_programs_from_sequences(
                &init_program,
                &hart0_insts,
                &hart1_insts,
                hart_count,
                &multicore_plan.shared_address_slots,
            );

            Ok(GeneratedTestCase {
                config: testcase_config,
                harts,
                mem_range,
                extensions: extensions.clone(),
                multicore_sync_metrics: None,
            })
        }
        ISABase::Rv64 => {
            info!(
                "Extensions for RV64: {:?}",
                extensions
                    .as_rv64()
                    .expect("RV64 extensions expected for RV64 testcase")
            );
            let mut counts = build_rv64_generation_counts(
                extensions
                    .as_rv64()
                    .expect("RV64 extensions expected for RV64 testcase"),
                &testcase_config,
            )?;
            apply_multicore_atomic_bias_rv64(is_dual_hart, &mut counts);

            if !has_enabled_csr_pool(&csr_selection_config)
                && counts.remove(&RV64Extensions::Zicsr).is_some()
            {
                info!(
                    "Skipping RV64 Zicsr generation because csr_selection_config enables no writable CSR groups"
                );
            }

            if !has_compressed_register_overlap(&config.register_operand_config)
                && counts.remove(&RV64Extensions::C).is_some()
            {
                info!(
                    "Skipping RV64 C generation because integer register range {:?} has no overlap with x8..x15",
                    config.register_operand_config.integer_register_number_range
                );
            }
            let plan = ordered_generation_plan(&counts);

            let mut hart0_insts = if initialization_seed.is_some() {
                Vec::new()
            } else {
                generate_sequences_with_rng(&mut rng, &plan, &config)?
            };

            hart0_insts = remove_special_instructions(hart0_insts);
            hart0_insts = remove_instructions_requiring_unsupported_extensions(
                hart0_insts,
                &extensions,
                testcase_config.isa_base,
            );
            if uses_cva6(impls) {
                hart0_insts = remove_cva6_unsupported_sequences(hart0_insts);
            }
            hart0_insts = normalize_sequence_alignment(hart0_insts, strict_base_alignment);

            let mut hart1_insts = Vec::new();
            if is_dual_hart && initialization_seed.is_none() {
                let mut generated = generate_sequences_with_rng(&mut rng, &plan, &config)?;
                generated = remove_special_instructions(generated);
                generated = remove_instructions_requiring_unsupported_extensions(
                    generated,
                    &extensions,
                    testcase_config.isa_base,
                );
                if uses_cva6(impls) {
                    generated = remove_cva6_unsupported_sequences(generated);
                }
                hart1_insts = normalize_sequence_alignment(generated, strict_base_alignment);
            }

            let (hart1_insts, multicore_plan) = match testcase_config.multicore_sync_config() {
                Some(sync) => {
                    couple_multicore_hart_streams(&hart0_insts, &hart1_insts, sync, &mut rng)
                }
                None => (hart1_insts, MulticoreCouplingPlan::default()),
            };
            let mut init_program = match initialization_seed {
                Some(seed) if compact_initialization => {
                    generate_compact_initialization_instructions_seeded(
                        &extensions,
                        testcase_config.isa_base,
                        &config,
                        hart_count,
                        seed,
                    )?
                }
                Some(seed) => generate_initialization_instructions_for_harts_seeded(
                    &extensions,
                    testcase_config.isa_base,
                    testcase_config.mem_size,
                    &config,
                    hart_count,
                    seed,
                )?,
                None => generate_initialization_instructions_for_harts(
                    &extensions,
                    testcase_config.isa_base,
                    testcase_config.mem_size,
                    &config,
                    hart_count,
                )?,
            };
            remove_special_asm_lines(&mut init_program);

            let harts = canonical_hart_programs_from_sequences(
                &init_program,
                &hart0_insts,
                &hart1_insts,
                hart_count,
                &multicore_plan.shared_address_slots,
            );

            Ok(GeneratedTestCase {
                config: testcase_config,
                harts,
                mem_range,
                extensions,
                multicore_sync_metrics: None,
            })
        }
    }
}

pub fn execute<P: AsRef<Path>>(
    testcase: &GeneratedTestCase,
    run_dir: P,
    coverage_mode: CoverageMode,
) -> Result<HashMap<String, ExecutionOutput>, Box<dyn std::error::Error>> {
    let plugin_manager = PluginManager::global();
    let run_root = run_dir.as_ref();
    let allow_unaligned = testcase.config.unaligned_access_required.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "unaligned_access_required must be explicitly set for testcase execution",
        )
    })?;
    let alignment_mode = if allow_unaligned {
        AlignmentMode::AllowUnaligned
    } else {
        AlignmentMode::RequireAligned
    };

    let impl_refs: Vec<String> = testcase.impl_refs().to_vec();
    let mut typed_inputs = HashMap::new();
    for impl_ref in &impl_refs {
        let per_hart_typed = testcase
            .combined_typed_sequences_per_hart_of(impl_ref)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "Combined instruction block missing for implementation: {:?}",
                        impl_ref
                    ),
                )
            })?;
        typed_inputs.insert(impl_ref.clone(), per_hart_typed);
    }

    // Pre-flight validation (sequential — cheap).
    for impl_ref in &impl_refs {
        let per_hart_typed = typed_inputs.get(impl_ref).expect(
            "typed input map must contain every implementation collected during testcase materialization",
        );
        let dir = run_root.join(impl_ref);
        let hart_inputs = per_hart_typed
            .iter()
            .map(|(hart_id, seqs)| (*hart_id, seqs.as_slice()))
            .collect::<Vec<_>>();
        plugin_manager.validate_with_request(
            impl_ref,
            ExecuteRequest {
                run_root: &dir,
                isa_base: testcase.config.isa_base,
                hart_inputs: &hart_inputs,
                extension_override: Some(&testcase.extensions),
                float_mode_override: None,
                alignment_mode,
                coverage_mode,
            },
        )?;
    }

    // Execute all implementations in parallel, one thread per implementation.
    let testcase_arc = Arc::new(testcase.clone());
    let typed_inputs_arc = Arc::new(typed_inputs);
    let run_root_buf = run_root.to_path_buf();

    type WorkerResult = (
        String,
        Duration,
        Result<ExecutionOutput, Box<dyn std::error::Error + Send + Sync>>,
    );

    let handles: Vec<thread::JoinHandle<WorkerResult>> = impl_refs
        .into_iter()
        .map(|impl_ref| {
            let tc = Arc::clone(&testcase_arc);
            let typed_inputs = Arc::clone(&typed_inputs_arc);
            let root = run_root_buf.clone();
            thread::spawn(move || {
                let plugin_manager = PluginManager::global();
                let per_hart_typed = match typed_inputs.get(&impl_ref) {
                    Some(v) => v,
                    None => {
                        return (
                            impl_ref.clone(),
                            Duration::ZERO,
                            Err(Box::new(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                format!(
                                    "Combined typed sequences missing for implementation: {:?}",
                                    impl_ref
                                ),
                            ))
                                as Box<dyn std::error::Error + Send + Sync>),
                        );
                    }
                };
                let dir = root.join(&impl_ref);
                let hart_input_refs = per_hart_typed
                    .iter()
                    .map(|(hart_id, seqs)| (*hart_id, seqs.as_slice()))
                    .collect::<Vec<_>>();
                let start = Instant::now();
                let request = ExecuteRequest {
                    run_root: &dir,
                    isa_base: tc.config.isa_base,
                    hart_inputs: &hart_input_refs,
                    extension_override: Some(&tc.extensions),
                    float_mode_override: None,
                    alignment_mode,
                    coverage_mode,
                };
                let mut collector = StreamingCollector::new();
                let streaming_result = plugin_manager.execute_streaming_with_request(
                    &impl_ref,
                    request,
                    &mut |event| collector.push(event),
                );
                let (per_hart, perf) = match streaming_result {
                    Ok(()) => match collector.finish() {
                        Ok(v) => v,
                        Err(e) => {
                            return (
                                impl_ref.clone(),
                                start.elapsed(),
                                Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                            );
                        }
                    },
                    Err(e) => {
                        return (
                            impl_ref.clone(),
                            start.elapsed(),
                            Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                        );
                    }
                };
                let mut output = ExecutionOutput {
                    per_hart,
                    core_id: impl_ref.clone(),
                    isa_base: tc.config.isa_base,
                    perf,
                };
                if output.per_hart.is_empty() {
                    output.per_hart = hart_input_refs
                        .iter()
                        .map(|input| {
                            (
                                input.0,
                                riscv_fuzz_plugin_api::HartExecutionOutput::empty(input.0),
                            )
                        })
                        .collect();
                }
                let output = match request.finalize_execution_output(output) {
                    Ok(v) => v,
                    Err(e) => {
                        return (
                            impl_ref.clone(),
                            start.elapsed(),
                            Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                        );
                    }
                };
                let mut output = output;
                let elapsed = start.elapsed();
                let range = match tc.mem_range.get(&impl_ref).copied() {
                    Some(r) => r,
                    None => {
                        return (
                            impl_ref.clone(),
                            elapsed,
                            Err(Box::new(std::io::Error::new(
                                std::io::ErrorKind::NotFound,
                                format!(
                                    "Memory range not found for implementation: {:?}",
                                    impl_ref
                                ),
                            ))
                                as Box<dyn std::error::Error + Send + Sync>),
                        );
                    }
                };
                if let Err(e) = output.normalize(
                    range,
                    &tc.config.random_config.register_operand_config,
                    tc.config
                        .random_config
                        .memory_access_config
                        .memory_base_register_number_range,
                ) {
                    return (
                        impl_ref.clone(),
                        elapsed,
                        Err(Box::new(e) as Box<dyn std::error::Error + Send + Sync>),
                    );
                }
                (impl_ref, elapsed, Ok(output))
            })
        })
        .collect();

    let mut result = HashMap::new();
    let mut durations: Vec<(String, Duration)> = Vec::new();
    let mut first_err: Option<Box<dyn std::error::Error>> = None;

    for handle in handles {
        match handle.join() {
            Ok((impl_ref, elapsed, Ok(output))) => {
                result.insert(impl_ref.clone(), output);
                durations.push((impl_ref, elapsed));
            }
            Ok((impl_ref, _, Err(e))) => {
                if first_err.is_none() {
                    first_err = Some(Box::new(std::io::Error::other(format!("{impl_ref}: {e}"))));
                }
            }
            Err(panic) => {
                if first_err.is_none() {
                    first_err = Some(Box::new(std::io::Error::other(format!(
                        "worker thread panicked: {panic:?}"
                    ))));
                }
            }
        }
    }

    if let Some(err) = first_err {
        return Err(err);
    }

    if !durations.is_empty() {
        durations.sort_by_key(|(impl_ref, _)| impl_ref.clone());
        let mut report = String::from(
            "# Execution Timings\n\n| Implementation | Duration (s) |\n| --- | --- |\n",
        );
        for (impl_ref, duration) in durations {
            report.push_str(&format!(
                "| {} | {:.3} |\n",
                impl_ref,
                duration.as_secs_f64()
            ));
        }
        let timing_path = run_root.join("execution_timings.md");
        fs::write(timing_path, report)?;
    }

    Ok(result)
}

pub(crate) fn build_execution_context_output_with_weak_fallback(
    output: ExecutionOutput,
    mem_range: (u64, u64),
    per_hart_input_instructions: &[(u32, Vec<InstructionSequence<RiscvInstruction>>)],
) -> Result<ExecutionContextOutput, ContextBuildError> {
    match build_execution_context_output(output.clone(), mem_range, per_hart_input_instructions) {
        Ok(context) => Ok(context),
        Err(ContextBuildError::MissingLogicalInstructionExecution { .. }) => {
            build_partial_execution_context_output(output, mem_range, per_hart_input_instructions)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[allow(dead_code)]
fn offset_mem_base_immediates(sequence: &mut InstructionSequence<RiscvInstruction>, offset: i64) {
    crate::instruction_layout::offset_mem_base_immediates(sequence, offset)
}

#[cfg(test)]
#[allow(dead_code)]
fn align_mem_base_immediates(sequence: &mut InstructionSequence<RiscvInstruction>, align: u64) {
    crate::instruction_layout::align_mem_base_immediates(sequence, align)
}

#[cfg(test)]
#[allow(dead_code)]
fn effective_mem_offset(sequence: &InstructionSequence<RiscvInstruction>) -> Option<i64> {
    crate::instruction_layout::effective_mem_offset(sequence)
}

fn constrain_multicore_register_window<R: rand::Rng>(
    rng: &mut R,
    multicore: bool,
    register_min: u64,
    register_max: u64,
    word_size: u64,
) -> (u64, u64) {
    if !multicore || register_min >= register_max {
        return (register_min, register_max);
    }

    let span = register_max.saturating_sub(register_min).saturating_add(1);
    if span <= 2 {
        return (register_min, register_max);
    }

    let focus_words = 16;
    let focus_span = word_size
        .saturating_mul(focus_words)
        .max(word_size.saturating_mul(2));
    let window = span.min(focus_span);

    if window >= span {
        return (register_min, register_max);
    }

    let max_start = register_max.saturating_sub(window.saturating_sub(1));
    let start = if max_start > register_min {
        rng.random_range(register_min..=max_start)
    } else {
        register_min
    };
    let end = start
        .saturating_add(window.saturating_sub(1))
        .min(register_max);

    (start, end)
}

#[cfg(test)]
#[allow(dead_code)]
fn adjust_base_address_immediates(instructions: &mut [RiscvInstruction], offset: i64) {
    crate::instruction_layout::adjust_base_address_immediates(instructions, offset)
}

#[cfg(test)]
#[allow(dead_code)]
fn align_base_address_immediates(instructions: &mut [RiscvInstruction], offset: i64, align: u64) {
    crate::instruction_layout::align_base_address_immediates(instructions, offset, align)
}

fn remove_special_asm_lines(program: &mut StructuredTestInstructions) {
    fn is_forbidden(line: &str) -> bool {
        let mnemonic = line.split_whitespace().next().unwrap_or_default();
        matches!(
            mnemonic,
            "ebreak" | "c.ebreak" | "ecall" | "mret" | "sret" | "dret" | "mnret" | "wfi"
        )
    }

    program.remove_matching_instructions(|instruction| is_forbidden(&instruction.to_string()));
}

fn normalized_text_mnemonic(text: &str) -> String {
    text.split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
        .replace('.', "")
}

fn normalized_instruction_mnemonic(instruction: &RiscvInstruction) -> String {
    normalized_text_mnemonic(&instruction.to_string())
}

fn inferred_instruction_memory_access_width_bytes(
    instruction: &RiscvInstruction,
    isa_base: ISABase,
) -> Option<usize> {
    inferred_memory_access_width_bytes(&normalized_instruction_mnemonic(instruction), isa_base)
}

fn runtime_register_operand_priority(operand: &RuntimeRegisterOperand) -> (u8, u8) {
    let role_priority = match operand.usage {
        RuntimeOperandUsage::MemoryBase => 0,
        RuntimeOperandUsage::Source | RuntimeOperandUsage::Destination => 1,
        RuntimeOperandUsage::Aggregate => 2,
        RuntimeOperandUsage::SyntaxOnly => 3,
        RuntimeOperandUsage::MemoryOffset | RuntimeOperandUsage::CsrAddress => 4,
    };
    let usage_priority = match operand.usage {
        RuntimeOperandUsage::MemoryBase => 0,
        RuntimeOperandUsage::Source => 1,
        RuntimeOperandUsage::Destination => 2,
        _ => 3,
    };
    (role_priority, usage_priority)
}

fn sorted_component_register_operands<T: RuntimeOperandAccess>(
    component: &T,
) -> Vec<RuntimeRegisterOperand> {
    let mut operands = component.register_operands();
    operands.sort_by_key(runtime_register_operand_priority);
    operands
}

fn push_unique_register_key(
    ordered: &mut Vec<PhysicalRegisterKey>,
    seen: &mut HashSet<PhysicalRegisterKey>,
    register: RuntimeRegister,
) {
    let key = PhysicalRegisterKey::from(register);
    if seen.insert(key) {
        ordered.push(key);
    }
}

fn sequence_runtime_register_keys_in_priority_order(
    sequence: &InstructionSequence<RiscvInstruction>,
) -> Vec<PhysicalRegisterKey> {
    let mut ordered = Vec::new();
    let mut seen = HashSet::new();

    for operand in sorted_component_register_operands(&sequence.instruction) {
        push_unique_register_key(&mut ordered, &mut seen, operand.register);
    }
    for pseudo in &sequence.pre_instructions {
        for operand in sorted_component_register_operands(pseudo) {
            push_unique_register_key(&mut ordered, &mut seen, operand.register);
        }
    }
    for pseudo in &sequence.post_instructions {
        for operand in sorted_component_register_operands(pseudo) {
            push_unique_register_key(&mut ordered, &mut seen, operand.register);
        }
    }

    ordered
}

fn sequence_runtime_register_set(
    sequence: &InstructionSequence<RiscvInstruction>,
) -> HashSet<PhysicalRegisterKey> {
    sequence_runtime_register_keys_in_priority_order(sequence)
        .into_iter()
        .collect()
}

fn sequence_register_overlap_count(
    hart0: &InstructionSequence<RiscvInstruction>,
    hart1: &InstructionSequence<RiscvInstruction>,
) -> usize {
    let hart0_registers = sequence_runtime_register_set(hart0);
    let hart1_registers = sequence_runtime_register_set(hart1);
    hart0_registers.intersection(&hart1_registers).count()
}

fn sequence_has_runtime_registers(sequence: &InstructionSequence<RiscvInstruction>) -> bool {
    !sequence_runtime_register_keys_in_priority_order(sequence).is_empty()
}

fn sequence_has_runtime_memory_operands(sequence: &InstructionSequence<RiscvInstruction>) -> bool {
    !sequence.instruction.memory_operands().is_empty()
}

fn sequence_looks_like_zero_operand_sync(sequence: &InstructionSequence<RiscvInstruction>) -> bool {
    matches!(
        normalized_instruction_mnemonic(&sequence.instruction).as_str(),
        "wrssto" | "wrsnto"
    )
}

fn sequence_is_multicore_focus(sequence: &InstructionSequence<RiscvInstruction>) -> bool {
    if sequence_has_runtime_memory_operands(sequence)
        || sequence_looks_like_zero_operand_sync(sequence)
    {
        return true;
    }

    sequence_runtime_register_keys_in_priority_order(sequence).len() >= 3
}

fn sequence_base_address_immediate(
    sequence: &InstructionSequence<RiscvInstruction>,
    register: PhysicalRegisterKey,
) -> Option<i64> {
    if register.bank != RegisterBank::Integer {
        return None;
    }

    sequence
        .pre_instructions
        .iter()
        .chain(sequence.post_instructions.iter())
        .find_map(|instruction| match instruction {
            RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
                immediate,
                purpose,
                ..
            }) if *purpose == LoadImmediatePurpose::BaseAddress => {
                let matches_register = instruction
                    .register_operands()
                    .first()
                    .map(|operand| PhysicalRegisterKey::from(operand.register))
                    == Some(register);
                matches_register.then_some(immediate.as_i64())
            }
            _ => None,
        })
}

fn set_sequence_base_address_immediate(
    sequence: &mut InstructionSequence<RiscvInstruction>,
    register: PhysicalRegisterKey,
    value: i64,
) -> bool {
    if register.bank != RegisterBank::Integer {
        return false;
    }

    let mut changed = false;
    for instruction in sequence
        .pre_instructions
        .iter_mut()
        .chain(sequence.post_instructions.iter_mut())
    {
        let matches_register = match instruction {
            RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate { purpose, .. })
                if *purpose == LoadImmediatePurpose::BaseAddress =>
            {
                instruction
                    .register_operands()
                    .first()
                    .map(|operand| PhysicalRegisterKey::from(operand.register))
                    == Some(register)
            }
            _ => false,
        };

        if !matches_register {
            continue;
        }

        if let RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate { immediate, .. }) =
            instruction
            && immediate.as_i64() != value
        {
            *immediate = match immediate {
                riscv_instruction_types::XLenImmediate::X32(_) => {
                    riscv_instruction_types::XLenImmediate::X32(value as i32)
                }
                riscv_instruction_types::XLenImmediate::X64(_) => {
                    riscv_instruction_types::XLenImmediate::X64(value)
                }
            };
            changed = true;
        }
    }

    changed
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedMemoryBase {
    Fixed(RuntimeMemoryBase),
    Register {
        register: PhysicalRegisterKey,
        base_value: i64,
    },
}

fn resolve_sequence_memory_base(
    sequence: &InstructionSequence<RiscvInstruction>,
    operand: &RuntimeMemoryOperand,
) -> Option<ResolvedMemoryBase> {
    if let Some(fixed_base) = operand.fixed_base {
        return Some(ResolvedMemoryBase::Fixed(fixed_base));
    }

    let base_register = operand.base_register.as_ref()?;
    let register = PhysicalRegisterKey::from(base_register.register);
    let base_value = sequence_base_address_immediate(sequence, register)?;
    Some(ResolvedMemoryBase::Register {
        register,
        base_value,
    })
}

fn sequence_true_memory_match_count(
    hart0: &InstructionSequence<RiscvInstruction>,
    hart1: &InstructionSequence<RiscvInstruction>,
) -> usize {
    let hart0_memory = hart0.instruction.memory_operands();
    let hart1_memory = hart1.instruction.memory_operands();

    hart0_memory
        .iter()
        .zip(hart1_memory.iter())
        .filter(|(hart0_operand, hart1_operand)| {
            if hart0_operand.offset_value != hart1_operand.offset_value {
                return false;
            }

            match (
                resolve_sequence_memory_base(hart0, hart0_operand),
                resolve_sequence_memory_base(hart1, hart1_operand),
            ) {
                (
                    Some(ResolvedMemoryBase::Fixed(hart0_base)),
                    Some(ResolvedMemoryBase::Fixed(hart1_base)),
                ) => hart0_base == hart1_base,
                (
                    Some(ResolvedMemoryBase::Register {
                        register: hart0_register,
                        base_value: hart0_base_value,
                    }),
                    Some(ResolvedMemoryBase::Register {
                        register: hart1_register,
                        base_value: hart1_base_value,
                    }),
                ) => hart0_register == hart1_register && hart0_base_value == hart1_base_value,
                _ => false,
            }
        })
        .count()
}

fn build_sequence_register_mapping(
    hart0: &InstructionSequence<RiscvInstruction>,
    hart1: &InstructionSequence<RiscvInstruction>,
    required_pairs: &[(PhysicalRegisterKey, PhysicalRegisterKey)],
) -> HashMap<PhysicalRegisterKey, PhysicalRegisterKey> {
    let mut mapping = HashMap::new();
    let mut used_targets = HashSet::new();

    for &(source, target) in required_pairs {
        if source.bank != target.bank || source == target {
            continue;
        }
        mapping.insert(source, target);
        used_targets.insert(target);
    }

    let hart0_registers = sequence_runtime_register_keys_in_priority_order(hart0);
    for source in sequence_runtime_register_keys_in_priority_order(hart1) {
        if mapping.contains_key(&source) {
            continue;
        }

        let Some(target) = hart0_registers.iter().copied().find(|target| {
            target.bank == source.bank && !used_targets.contains(target) && *target != source
        }) else {
            continue;
        };

        mapping.insert(source, target);
        used_targets.insert(target);
    }

    mapping
}

fn apply_component_register_mapping<T: RuntimeOperandAccess>(
    component: &mut T,
    mapping: &HashMap<PhysicalRegisterKey, PhysicalRegisterKey>,
) -> bool {
    let mut changed = false;
    for operand in component.register_operands() {
        let source = PhysicalRegisterKey::from(operand.register);
        let Some(target) = mapping.get(&source).copied() else {
            continue;
        };
        if source == target {
            continue;
        }

        if let Ok(updated) = component.set_register_operand(operand.name, target.assignment()) {
            changed |= updated;
        }
    }
    changed
}

fn apply_sequence_register_mapping(
    sequence: &mut InstructionSequence<RiscvInstruction>,
    mapping: &HashMap<PhysicalRegisterKey, PhysicalRegisterKey>,
) -> bool {
    let mut changed = false;
    for pseudo in &mut sequence.pre_instructions {
        changed |= apply_component_register_mapping(pseudo, mapping);
    }
    changed |= apply_component_register_mapping(&mut sequence.instruction, mapping);
    for pseudo in &mut sequence.post_instructions {
        changed |= apply_component_register_mapping(pseudo, mapping);
    }
    changed
}

fn required_memory_base_matches(
    hart0: &InstructionSequence<RiscvInstruction>,
    hart1: &InstructionSequence<RiscvInstruction>,
) -> Vec<(PhysicalRegisterKey, PhysicalRegisterKey)> {
    let hart0_memory = hart0.instruction.memory_operands();
    let hart1_memory = hart1.instruction.memory_operands();
    let mut required = Vec::new();

    for (hart0_operand, hart1_operand) in hart0_memory.iter().zip(hart1_memory.iter()) {
        let (Some(hart0_base), Some(hart1_base)) = (
            hart0_operand.base_register.as_ref(),
            hart1_operand.base_register.as_ref(),
        ) else {
            continue;
        };

        required.push((
            PhysicalRegisterKey::from(hart1_base.register),
            PhysicalRegisterKey::from(hart0_base.register),
        ));
    }

    required
}

fn align_sequence_memory_offsets_to_target(
    hart0: &InstructionSequence<RiscvInstruction>,
    hart1: &mut InstructionSequence<RiscvInstruction>,
) -> bool {
    let hart0_memory = hart0.instruction.memory_operands();
    let hart1_memory_len = hart1.instruction.memory_operands().len();
    let pair_count = hart0_memory.len().min(hart1_memory_len);
    let mut changed = false;

    for (index, target_operand) in hart0_memory.iter().take(pair_count).enumerate() {
        let Some(offset) = target_operand.offset_value else {
            continue;
        };

        if let Ok(updated) = hart1.instruction.set_memory_offset_value(index, offset) {
            changed |= updated;
        }
    }

    changed
}

fn align_sequence_base_address_setup_to_target(
    hart0: &InstructionSequence<RiscvInstruction>,
    hart1: &mut InstructionSequence<RiscvInstruction>,
    required_pairs: &[(PhysicalRegisterKey, PhysicalRegisterKey)],
) -> bool {
    let mut changed = false;
    for &(_, target_register) in required_pairs {
        let Some(target_base_value) = sequence_base_address_immediate(hart0, target_register)
        else {
            continue;
        };
        changed |= set_sequence_base_address_immediate(hart1, target_register, target_base_value);
    }
    changed
}

fn attempt_register_coupling(
    hart0: &InstructionSequence<RiscvInstruction>,
    hart1: &InstructionSequence<RiscvInstruction>,
) -> Option<(InstructionSequence<RiscvInstruction>, bool, bool)> {
    let before_register_overlap = sequence_register_overlap_count(hart0, hart1);
    let before_memory_match = sequence_true_memory_match_count(hart0, hart1);
    let mapping = build_sequence_register_mapping(hart0, hart1, &[]);
    if mapping.is_empty() {
        return None;
    }

    let mut candidate = hart1.clone();
    if !apply_sequence_register_mapping(&mut candidate, &mapping) {
        return None;
    }

    let after_register_overlap = sequence_register_overlap_count(hart0, &candidate);
    let after_memory_match = sequence_true_memory_match_count(hart0, &candidate);
    let register_match = after_register_overlap > before_register_overlap;
    let memory_match = after_memory_match > before_memory_match;

    register_match.then_some((candidate, register_match, memory_match))
}

fn attempt_memory_coupling(
    hart0: &InstructionSequence<RiscvInstruction>,
    hart1: &InstructionSequence<RiscvInstruction>,
) -> Option<(InstructionSequence<RiscvInstruction>, bool, bool)> {
    let hart0_memory = hart0.instruction.memory_operands();
    let hart1_memory = hart1.instruction.memory_operands();
    if hart0_memory.is_empty() || hart1_memory.is_empty() {
        return None;
    }

    let before_register_overlap = sequence_register_overlap_count(hart0, hart1);
    let before_memory_match = sequence_true_memory_match_count(hart0, hart1);
    let required_pairs = required_memory_base_matches(hart0, hart1);
    let mapping = build_sequence_register_mapping(hart0, hart1, &required_pairs);
    let mut candidate = hart1.clone();
    let mut changed = apply_sequence_register_mapping(&mut candidate, &mapping);
    changed |= align_sequence_memory_offsets_to_target(hart0, &mut candidate);
    changed |= align_sequence_base_address_setup_to_target(hart0, &mut candidate, &required_pairs);
    if !changed {
        return None;
    }

    let after_register_overlap = sequence_register_overlap_count(hart0, &candidate);
    let after_memory_match = sequence_true_memory_match_count(hart0, &candidate);
    let register_match = after_register_overlap > before_register_overlap;
    let memory_match = after_memory_match > before_memory_match;

    memory_match.then_some((candidate, register_match, memory_match))
}

fn shuffled_candidate_slots<R, F>(
    sequences: &[InstructionSequence<RiscvInstruction>],
    predicate: F,
    prefer_focused_sequences: bool,
    rng: &mut R,
) -> Vec<usize>
where
    R: rand::Rng,
    F: Fn(&InstructionSequence<RiscvInstruction>) -> bool,
{
    let mut focused = Vec::new();
    let mut other = Vec::new();

    for (idx, sequence) in sequences.iter().enumerate() {
        if !predicate(sequence) {
            continue;
        }

        if prefer_focused_sequences && sequence_is_multicore_focus(sequence) {
            focused.push(idx);
        } else {
            other.push(idx);
        }
    }

    focused.shuffle(rng);
    other.shuffle(rng);
    if prefer_focused_sequences {
        focused.extend(other);
        focused
    } else {
        let mut all = focused;
        all.extend(other);
        all.shuffle(rng);
        all
    }
}

fn couple_multicore_hart_streams<R: rand::Rng>(
    hart0_sequences: &[InstructionSequence<RiscvInstruction>],
    hart1_sequences: &[InstructionSequence<RiscvInstruction>],
    config: &MulticoreSyncConfig,
    rng: &mut R,
) -> (
    Vec<InstructionSequence<RiscvInstruction>>,
    MulticoreCouplingPlan,
) {
    let mut coupled = hart1_sequences.to_vec();
    if hart0_sequences.is_empty() || hart1_sequences.is_empty() {
        return (coupled, MulticoreCouplingPlan::default());
    }

    let target_match_count =
        target_slot_count(hart1_sequences.len(), config.interaction_slot_ratio);
    let target_shared_address_count =
        target_slot_count(target_match_count, config.shared_address_ratio);
    let mut plan = MulticoreCouplingPlan {
        hart0_slot_count: hart0_sequences.len(),
        hart1_slot_count: hart1_sequences.len(),
        target_match_count,
        target_shared_address_count,
        matched_slots: Vec::new(),
        coupled_slots: HashSet::new(),
        shared_address_slots: HashSet::new(),
    };

    if target_match_count == 0 {
        return (coupled, plan);
    }

    let memory_hart0_slots = shuffled_candidate_slots(
        hart0_sequences,
        sequence_has_runtime_memory_operands,
        config.prefer_focused_sequences,
        rng,
    );
    let register_hart0_slots = shuffled_candidate_slots(
        hart0_sequences,
        sequence_has_runtime_registers,
        config.prefer_focused_sequences,
        rng,
    );
    let memory_hart1_slots = shuffled_candidate_slots(
        hart1_sequences,
        sequence_has_runtime_memory_operands,
        config.prefer_focused_sequences,
        rng,
    );
    let register_hart1_slots = shuffled_candidate_slots(
        hart1_sequences,
        sequence_has_runtime_registers,
        config.prefer_focused_sequences,
        rng,
    );

    let mut used_hart1_slots = HashSet::new();

    for hart1_slot in memory_hart1_slots {
        if plan.matched_slots.len() >= plan.target_match_count
            || plan.shared_address_slots.len() >= plan.target_shared_address_count
        {
            break;
        }
        if used_hart1_slots.contains(&hart1_slot) {
            continue;
        }

        for &hart0_slot in &memory_hart0_slots {
            let Some((rewritten, register_match, memory_match)) =
                attempt_memory_coupling(&hart0_sequences[hart0_slot], &coupled[hart1_slot])
            else {
                continue;
            };

            coupled[hart1_slot] = rewritten;
            used_hart1_slots.insert(hart1_slot);
            plan.coupled_slots.insert(hart1_slot);
            if memory_match {
                plan.shared_address_slots.insert(hart1_slot);
            }
            plan.matched_slots.push(MulticoreCoupledSlot {
                hart0_slot,
                hart1_slot,
                register_match,
                memory_match,
            });
            break;
        }
    }

    for hart1_slot in register_hart1_slots {
        if plan.matched_slots.len() >= plan.target_match_count {
            break;
        }
        if used_hart1_slots.contains(&hart1_slot) {
            continue;
        }

        for &hart0_slot in &register_hart0_slots {
            let Some((rewritten, register_match, memory_match)) =
                attempt_register_coupling(&hart0_sequences[hart0_slot], &coupled[hart1_slot])
            else {
                continue;
            };

            coupled[hart1_slot] = rewritten;
            used_hart1_slots.insert(hart1_slot);
            plan.coupled_slots.insert(hart1_slot);
            if memory_match {
                plan.shared_address_slots.insert(hart1_slot);
            }
            plan.matched_slots.push(MulticoreCoupledSlot {
                hart0_slot,
                hart1_slot,
                register_match,
                memory_match,
            });
            break;
        }
    }

    (coupled, plan)
}

fn is_rv32_toolchain_unsupported_mnemonic(mnemonic: &str) -> bool {
    matches!(mnemonic, "ld" | "sd" | "cld" | "csd" | "cldsp" | "csdsp")
}

fn remove_rv32_toolchain_unsupported_sequences(
    sequences: Vec<InstructionSequence<RiscvInstruction>>,
) -> Vec<InstructionSequence<RiscvInstruction>> {
    sequences
        .into_iter()
        .filter(|sequence| {
            let mnemonic = normalized_instruction_mnemonic(&sequence.instruction);
            !is_rv32_toolchain_unsupported_mnemonic(&mnemonic)
        })
        .collect()
}

fn uses_cva6(impls: &[String]) -> bool {
    impls.iter().any(|impl_ref| impl_ref == "cva6")
}

fn is_cva6_unsupported_narrow_mem_mnemonic(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "lb" | "lbu" | "lh" | "lhu" | "sb" | "sh" | "clbu" | "clh" | "clhu" | "csb" | "csh"
    )
}

fn remove_cva6_unsupported_sequences(
    sequences: Vec<InstructionSequence<RiscvInstruction>>,
) -> Vec<InstructionSequence<RiscvInstruction>> {
    sequences
        .into_iter()
        .filter(|sequence| {
            let mnemonic = normalized_instruction_mnemonic(&sequence.instruction);
            !is_cva6_unsupported_narrow_mem_mnemonic(&mnemonic)
        })
        .collect()
}

#[cfg(test)]
#[allow(dead_code)]
fn adjusted_hart_instruction_sequences(
    sequences: &[InstructionSequence<RiscvInstruction>],
    mem_offset: i64,
    strict_base_alignment: Option<u64>,
) -> Vec<InstructionSequence<RiscvInstruction>> {
    let mut adjusted = Vec::with_capacity(sequences.len());
    for seq in sequences {
        let mut inst = seq.clone();
        offset_mem_base_immediates(&mut inst, mem_offset);
        if let Some(align) = strict_base_alignment {
            align_mem_base_immediates(&mut inst, align);
        }
        adjusted.push(inst);
    }
    adjusted
}

fn canonical_hart_programs_from_sequences(
    init_program: &StructuredTestInstructions,
    hart0_sequences: &[InstructionSequence<RiscvInstruction>],
    hart1_sequences: &[InstructionSequence<RiscvInstruction>],
    hart_count: u32,
    shared_slots: &HashSet<usize>,
) -> Vec<HartProgram> {
    let init_sequences = init_program
        .harts
        .iter()
        .map(|hart| {
            hart.iter()
                .cloned()
                .map(InstructionSequence::new)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut harts = Vec::with_capacity(hart_count.max(init_sequences.len() as u32) as usize);
    for hart_id in 0..hart_count.max(init_sequences.len() as u32) {
        let init = init_sequences
            .get(hart_id as usize)
            .cloned()
            .unwrap_or_default();
        let (test, shared_test_sequence_indices) = match hart_id {
            0 => (hart0_sequences.to_vec(), HashSet::new()),
            1 if hart_count == 2 => (hart1_sequences.to_vec(), shared_slots.clone()),
            _ => (Vec::new(), HashSet::new()),
        };
        harts.push(HartProgram {
            hart_id,
            init,
            test,
            shared_init_sequence_indices: BTreeSet::new(),
            shared_test_sequence_indices: shared_test_sequence_indices.into_iter().collect(),
        });
    }
    harts
}

fn apply_multicore_atomic_bias_rv32(multicore: bool, counts: &mut HashMap<RV32Extensions, usize>) {
    if !multicore {
        return;
    }

    fn bump(
        counts: &mut HashMap<RV32Extensions, usize>,
        ext: RV32Extensions,
        min: usize,
        mul: usize,
    ) {
        if let Some(entry) = counts.get_mut(&ext) {
            let next = (*entry).saturating_mul(mul).max(min);
            *entry = next;
        }
    }

    let atomic_extra_min = (MULTICORE_AMO_MIN / 2).max(8);
    let reservation_wait_min = (MULTICORE_LRSC_MIN / 2).max(4);

    // In multicore mode, prioritize atomic families only.
    bump(
        counts,
        RV32Extensions::Zaamo,
        MULTICORE_AMO_MIN,
        MULTICORE_AMO_MUL,
    );
    bump(
        counts,
        RV32Extensions::Zalrsc,
        MULTICORE_LRSC_MIN,
        MULTICORE_LRSC_MUL,
    );
    // Zawrs is reservation-related, but should stay below actual AMO/LR/SC traffic.
    bump(counts, RV32Extensions::Zawrs, reservation_wait_min, 2);
    bump(counts, RV32Extensions::Zalasr, atomic_extra_min, 3);
    bump(counts, RV32Extensions::Zabha, atomic_extra_min, 3);
    bump(counts, RV32Extensions::Zacas, atomic_extra_min, 3);
}

fn apply_multicore_atomic_bias_rv64(multicore: bool, counts: &mut HashMap<RV64Extensions, usize>) {
    if !multicore {
        return;
    }

    fn bump(
        counts: &mut HashMap<RV64Extensions, usize>,
        ext: RV64Extensions,
        min: usize,
        mul: usize,
    ) {
        if let Some(entry) = counts.get_mut(&ext) {
            let next = (*entry).saturating_mul(mul).max(min);
            *entry = next;
        }
    }

    let atomic_extra_min = (MULTICORE_AMO_MIN / 2).max(8);
    let reservation_wait_min = (MULTICORE_LRSC_MIN / 2).max(4);

    // In multicore mode, prioritize atomic families only.
    bump(
        counts,
        RV64Extensions::Zaamo,
        MULTICORE_AMO_MIN,
        MULTICORE_AMO_MUL,
    );
    bump(
        counts,
        RV64Extensions::Zalrsc,
        MULTICORE_LRSC_MIN,
        MULTICORE_LRSC_MUL,
    );
    // Zawrs is reservation-related, but should stay below actual AMO/LR/SC traffic.
    bump(counts, RV64Extensions::Zawrs, reservation_wait_min, 2);
    bump(counts, RV64Extensions::Zalasr, atomic_extra_min, 3);
    bump(counts, RV64Extensions::Zabha, atomic_extra_min, 3);
    bump(counts, RV64Extensions::Zacas, atomic_extra_min, 3);
}

#[cfg(all(test, feature = "internal-tests"))]
mod helper_tests {
    use super::*;
    use crate::testcase::StructuredTestInstructions;
    use rand::RngCore;
    use riscv_fuzz_plugin_api::context::{
        ExecutionContextOutput, HartExecutionContextOutput, InstructionContext,
        LogicalInstructionContextOutput, MemoryContext, MemoryContextEntry, RegisterContext,
    };
    use riscv_fuzz_plugin_api::{InstructionTiming, MemValue, PerfMetrics};
    use riscv_instruction::MemoryAccessInstruction;
    use riscv_instruction::{
        RV32_I_ADD, RV32_I_ADDI, RV32_I_LB, RV32_I_LBU, RV32_I_LHU, RV32_I_LW, RV32_I_OR,
        RV32_I_SH, RV32_I_SUB, RV32_I_SW, RV32_I_XOR, RV32IInstructions, RV32Instruction,
        RV64_I_LD, RV64_I_SD, RV64IInstructions, RV64Instruction,
    };
    use riscv_instruction_types::{
        AsmDecodeWithContext, BaseAddressRegister, Immediate, IntegerRegister, IsaDecodeContext,
        MemoryAccessConfig, RandomConfig, ValidatedValue, XLenImmediate,
    };
    use std::collections::BTreeMap;

    fn li(rd: u8, immediate: i64, purpose: LoadImmediatePurpose) -> RiscvInstruction {
        RiscvInstruction::from(PseudoInstruction::LoadImmediate {
            rd: IntegerRegister::new(rd).expect("valid register"),
            immediate: XLenImmediate::X64(immediate),
            purpose,
        })
    }

    fn x_reg(reg: u8) -> IntegerRegister {
        IntegerRegister::new(reg).expect("valid register")
    }

    fn base_reg(reg: u8) -> BaseAddressRegister {
        BaseAddressRegister::new(reg).expect("valid base register")
    }

    fn decode_instruction(asm: &str, ctx: IsaDecodeContext) -> RiscvInstruction {
        RiscvInstruction::decode_asm_with_context(asm, ctx).expect("parse instruction")
    }

    fn decode_sequence(asm: &str, ctx: IsaDecodeContext) -> InstructionSequence<RiscvInstruction> {
        InstructionSequence::<RiscvInstruction>::decode_asm_with_context(asm, ctx)
            .expect("parse instruction sequence")
    }

    fn rv32_lw(rd: u8, offset: i32, rs1: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::LW(RV32_I_LW {
            xd: x_reg(rd),
            xs1: base_reg(rs1),
            imm: Immediate::<12, true>::new(offset).expect("valid lw offset"),
        })))
    }

    fn rv32_sw(rs2: u8, offset: i32, rs1: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::SW(RV32_I_SW {
            xs2: x_reg(rs2),
            xs1: base_reg(rs1),
            imm: Immediate::<12, true>::new(offset).expect("valid sw offset"),
        })))
    }

    fn rv32_lb(rd: u8, offset: i32, rs1: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::LB(RV32_I_LB {
            xd: x_reg(rd),
            xs1: base_reg(rs1),
            imm: Immediate::<12, true>::new(offset).expect("valid lb offset"),
        })))
    }

    fn rv32_lbu(rd: u8, offset: i32, rs1: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::LBU(RV32_I_LBU {
            xd: x_reg(rd),
            xs1: base_reg(rs1),
            imm: Immediate::<12, true>::new(offset).expect("valid lbu offset"),
        })))
    }

    fn rv32_lhu(rd: u8, offset: i32, rs1: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::LHU(RV32_I_LHU {
            xd: x_reg(rd),
            xs1: base_reg(rs1),
            imm: Immediate::<12, true>::new(offset).expect("valid lhu offset"),
        })))
    }

    fn rv32_sh(rs2: u8, offset: i32, rs1: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::SH(RV32_I_SH {
            xs2: x_reg(rs2),
            xs1: base_reg(rs1),
            imm: Immediate::<12, true>::new(offset).expect("valid sh offset"),
        })))
    }

    fn rv32_addi(rd: u8, rs1: u8, imm: i32) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(RV32_I_ADDI {
            xd: x_reg(rd),
            xs1: x_reg(rs1),
            imm: Immediate::<12>::new(imm).expect("valid addi immediate"),
        })))
    }

    fn rv32_add(rd: u8, rs1: u8, rs2: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADD(RV32_I_ADD {
            xd: x_reg(rd),
            xs1: x_reg(rs1),
            xs2: x_reg(rs2),
        })))
    }

    fn rv32_xor(rd: u8, rs1: u8, rs2: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::XOR(RV32_I_XOR {
            xd: x_reg(rd),
            xs1: x_reg(rs1),
            xs2: x_reg(rs2),
        })))
    }

    fn rv32_sub(rd: u8, rs1: u8, rs2: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::SUB(RV32_I_SUB {
            xd: x_reg(rd),
            xs1: x_reg(rs1),
            xs2: x_reg(rs2),
        })))
    }

    fn rv32_or(rd: u8, rs1: u8, rs2: u8) -> RiscvInstruction {
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::OR(RV32_I_OR {
            xd: x_reg(rd),
            xs1: x_reg(rs1),
            xs2: x_reg(rs2),
        })))
    }

    fn rv64_ld(rd: u8, offset: i32, rs1: u8) -> RiscvInstruction {
        RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::LD(RV64_I_LD {
            xd: x_reg(rd),
            xs1: base_reg(rs1),
            imm: Immediate::<12, true>::new(offset).expect("valid ld offset"),
        })))
    }

    fn rv64_sd(rs2: u8, offset: i32, rs1: u8) -> RiscvInstruction {
        RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::SD(RV64_I_SD {
            xs2: IntegerRegister::new(rs2).expect("valid sd rs2"),
            xs1: base_reg(rs1),
            imm: Immediate::<12, true>::new(offset).expect("valid sd offset"),
        })))
    }

    fn instruction_context_with_addresses(addresses: &[u64]) -> InstructionContext {
        let mut memory_before = BTreeMap::new();
        for &addr in addresses {
            memory_before.insert(addr, 0);
        }

        InstructionContext {
            registers_before: HashMap::new(),
            memory_before,
            register_context: RegisterContext {
                entries: Vec::new(),
            },
            memory_context: if addresses.is_empty() {
                MemoryContext::default()
            } else {
                MemoryContext {
                    base_register: "x5".to_string(),
                    base_offset: addresses[0],
                    entries: vec![MemoryContextEntry {
                        width: MemoryValueWidth::Word,
                        addresses: addresses.to_vec(),
                        value: 0,
                    }],
                }
            },
        }
    }

    fn logical_instruction_context_output(
        hart_id: u32,
        instruction_index: usize,
        local_instruction_index: usize,
        instruction: &str,
        context: InstructionContext,
        memory_writes: Vec<MemValue>,
        timing: InstructionTiming,
    ) -> LogicalInstructionContextOutput {
        LogicalInstructionContextOutput {
            hart_id,
            instruction_index,
            local_instruction_index,
            instruction: decode_instruction(instruction, IsaDecodeContext::RV32),
            trace_inst_id: riscv_fuzz_plugin_api::TraceInstId::new(
                hart_id,
                local_instruction_index,
            ),
            asm_instruction_indices: vec![local_instruction_index],
            exceptions: Vec::new(),
            register_writes: Vec::new(),
            memory_writes,
            timing,
            context,
        }
    }

    fn generated_testcase(
        impl_ref: &str,
        config: TestCaseConfig,
        test_program: StructuredTestInstructions,
        extensions: IsaExtensions,
    ) -> GeneratedTestCase {
        let mut testcase = GeneratedTestCase {
            config,
            harts: Vec::new(),
            mem_range: HashMap::from([(impl_ref.to_string(), (0, 63))]),
            extensions,
            multicore_sync_metrics: None,
        };
        testcase.set_test_program(test_program);
        testcase
    }

    #[cfg(feature = "plugin-tests")]
    fn configure_plugin_dirs() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .expect("workspace root")
            .to_path_buf();
        let joined = std::env::join_paths([
            root.join("target/debug"),
            root.join("target/release"),
            root.join("riscv_fuzz_plugins"),
        ])
        .expect("join plugin dirs");
        unsafe { std::env::set_var("RISCV_FUZZ_PLUGIN_DIRS", joined) };
    }

    fn flat_structured_program(
        hart0_entries: Vec<RiscvInstruction>,
        hart1_entries: Vec<RiscvInstruction>,
    ) -> StructuredTestInstructions {
        let mut program = StructuredTestInstructions::default();
        for instruction in hart0_entries {
            program.push_instruction_to_hart(0, instruction);
        }
        for instruction in hart1_entries {
            program.push_instruction_to_hart(1, instruction);
        }
        program
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

    fn seq_full(
        pre_instructions: Vec<RiscvInstruction>,
        instruction: RiscvInstruction,
        post_instructions: Vec<RiscvInstruction>,
    ) -> InstructionSequence<RiscvInstruction> {
        InstructionSequence {
            pre_instructions,
            instruction,
            post_instructions,
        }
    }

    #[test]
    fn offsets_only_base_address_li() {
        let mut seq = seq_full(
            vec![
                li(5, 10, LoadImmediatePurpose::BaseAddress),
                li(6, 20, LoadImmediatePurpose::Generic),
            ],
            rv32_sw(1, 0, 5),
            vec![li(7, 30, LoadImmediatePurpose::BaseAddress)],
        );

        offset_mem_base_immediates(&mut seq, 0x100);

        assert_eq!(
            seq.pre_instructions[0],
            li(5, 0x100 + 10, LoadImmediatePurpose::BaseAddress)
        );
        assert_eq!(
            seq.pre_instructions[1],
            li(6, 20, LoadImmediatePurpose::Generic)
        );
        assert_eq!(
            seq.post_instructions[0],
            li(7, 30, LoadImmediatePurpose::BaseAddress)
        );
    }

    #[test]
    fn rv32_toolchain_filter_removes_doubleword_load_store_mnemonics() {
        let sequences = vec![
            InstructionSequence::new(rv64_ld(16, 0, 13)),
            InstructionSequence::new(rv64_sd(20, 0, 15)),
            InstructionSequence::new(rv32_lw(16, 0, 13)),
        ];

        let filtered = remove_rv32_toolchain_unsupported_sequences(sequences);
        let rendered: Vec<String> = filtered.iter().map(ToString::to_string).collect();

        assert_eq!(rendered, vec![String::from("lw x16, 0(x13)")]);
    }

    #[test]
    fn cva6_filter_removes_narrow_memory_mnemonics() {
        let sequences = vec![
            InstructionSequence::new(rv32_lb(16, 0, 13)),
            InstructionSequence::new(rv32_lbu(16, 0, 13)),
            InstructionSequence::new(rv32_lhu(16, 0, 13)),
            InstructionSequence::new(rv32_sh(20, 0, 15)),
            InstructionSequence::new(rv32_lw(16, 0, 13)),
            decode_sequence("lr.w x16, (x13)", IsaDecodeContext::RV32),
        ];

        let filtered = remove_cva6_unsupported_sequences(sequences);
        let rendered: Vec<String> = filtered.iter().map(ToString::to_string).collect();

        assert_eq!(
            rendered,
            vec![
                String::from("lw x16, 0(x13)"),
                String::from("lr.w x16, (x13)"),
            ]
        );
    }

    #[test]
    fn handles_zero_offset_without_change() {
        let mut seq = seq_with_pre(
            vec![li(2, 0x200, LoadImmediatePurpose::BaseAddress)],
            decode_instruction("sd x4, 0(x2)", IsaDecodeContext::RV64),
        );

        offset_mem_base_immediates(&mut seq, 0);

        assert_eq!(
            seq.pre_instructions[0],
            li(2, 0x200, LoadImmediatePurpose::BaseAddress)
        );
    }

    #[test]
    fn ignores_non_baseaddress_purpose() {
        let mut seq = seq_with_pre(
            vec![li(8, 123, LoadImmediatePurpose::CsrValue)],
            decode_instruction("csrrw x1, mstatus, x8", IsaDecodeContext::RV64),
        );

        offset_mem_base_immediates(&mut seq, 0x10);

        assert_eq!(
            seq.pre_instructions[0],
            li(8, 123, LoadImmediatePurpose::CsrValue)
        );
    }

    #[test]
    fn aligns_base_address_immediate_down_for_strict_mode() {
        let mut seq = seq_with_pre(
            vec![li(5, 0x23, LoadImmediatePurpose::BaseAddress)],
            rv32_sw(1, 0, 5),
        );

        align_mem_base_immediates(&mut seq, 8);

        assert_eq!(
            seq.pre_instructions[0],
            li(5, 0x20, LoadImmediatePurpose::BaseAddress)
        );
        assert_eq!(seq.instruction.offset_operand_value(), Some(0));
    }

    #[test]
    fn aligns_base_address_immediate_using_instruction_offset() {
        let mut seq = seq_with_pre(
            vec![li(5, 0x23, LoadImmediatePurpose::BaseAddress)],
            rv32_sw(1, 3, 5),
        );

        align_mem_base_immediates(&mut seq, 8);

        assert_eq!(
            seq.pre_instructions[0],
            li(5, 0x1d, LoadImmediatePurpose::BaseAddress)
        );
        let effective_address =
            if let RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
                immediate, ..
            }) = &seq.pre_instructions[0]
            {
                i128::from(immediate.as_i64())
                    + i128::from(seq.instruction.offset_operand_value().unwrap())
            } else {
                unreachable!("expected load immediate");
            };
        assert_eq!(effective_address, 0x20);
    }

    #[test]
    fn aligns_base_address_immediate_for_zero_offset_parenthesized_memory_ops() {
        let mut seq = seq_with_pre(
            vec![li(9, 0x23, LoadImmediatePurpose::BaseAddress)],
            decode_instruction("amomaxu.w.rl x20, x16, (x9)", IsaDecodeContext::RV32),
        );

        align_mem_base_immediates(&mut seq, 4);

        assert_eq!(
            seq.pre_instructions[0],
            li(9, 0x20, LoadImmediatePurpose::BaseAddress)
        );
        assert_eq!(effective_mem_offset(&seq), Some(0));
    }

    #[test]
    fn alignment_helper_ignores_non_memory_instruction_offsets() {
        let mut seq = seq_with_pre(
            vec![li(5, 0x23, LoadImmediatePurpose::BaseAddress)],
            rv32_addi(1, 2, 7),
        );

        align_mem_base_immediates(&mut seq, 8);

        assert_eq!(
            seq.pre_instructions[0],
            li(5, 0x23, LoadImmediatePurpose::BaseAddress)
        );
    }

    #[test]
    fn random_config_keeps_provided_csr_selection_config() {
        let random_config = RandomConfig {
            append_fflags_readback: false,
            csr_selection_config: CsrSelectionConfig {
                machine_csr_operands_enabled: true,
                supervisor_csr_operands_enabled: false,
                hypervisor_csr_operands_enabled: false,
                virtual_supervisor_csr_operands_enabled: false,
                rnmi_csr_operands_enabled: false,
                floating_csr_operands_enabled: true,
                vector_csr_operands_enabled: false,
                debug_csr_operands_enabled: true,
            },
            ..RandomConfig::default()
        };

        assert!(!random_config.append_fflags_readback);
        let csr_selection_config = random_config.csr_selection_config;
        assert!(csr_selection_config.machine_csr_operands_enabled);
        assert!(csr_selection_config.floating_csr_operands_enabled);
        assert!(csr_selection_config.debug_csr_operands_enabled);
        assert!(!csr_selection_config.supervisor_csr_operands_enabled);
    }

    #[test]
    fn multicore_atomic_bias_only_bumps_atomic_extensions_rv32() {
        let mut counts = HashMap::from([
            (RV32Extensions::I, 5usize),
            (RV32Extensions::Zaamo, 5usize),
            (RV32Extensions::Zalrsc, 5usize),
            (RV32Extensions::Zawrs, 5usize),
            (RV32Extensions::Zifencei, 5usize),
            (RV32Extensions::Zicbom, 5usize),
            (RV32Extensions::Zicboz, 5usize),
            (RV32Extensions::Zalasr, 5usize),
            (RV32Extensions::Zabha, 5usize),
            (RV32Extensions::Zacas, 5usize),
        ]);

        apply_multicore_atomic_bias_rv32(true, &mut counts);

        assert_eq!(counts[&RV32Extensions::I], 5);
        assert_eq!(counts[&RV32Extensions::Zifencei], 5);
        assert_eq!(counts[&RV32Extensions::Zicbom], 5);
        assert_eq!(counts[&RV32Extensions::Zicboz], 5);
        assert_eq!(
            counts[&RV32Extensions::Zaamo],
            5usize
                .saturating_mul(MULTICORE_AMO_MUL)
                .max(MULTICORE_AMO_MIN)
        );
        assert_eq!(
            counts[&RV32Extensions::Zalrsc],
            5usize
                .saturating_mul(MULTICORE_LRSC_MUL)
                .max(MULTICORE_LRSC_MIN)
        );
        assert_eq!(
            counts[&RV32Extensions::Zawrs],
            5usize
                .saturating_mul(2)
                .max((MULTICORE_LRSC_MIN / 2).max(4))
        );
        assert_eq!(
            counts[&RV32Extensions::Zalasr],
            (MULTICORE_AMO_MIN / 2).max(8)
        );
        assert_eq!(
            counts[&RV32Extensions::Zabha],
            (MULTICORE_AMO_MIN / 2).max(8)
        );
        assert_eq!(
            counts[&RV32Extensions::Zacas],
            (MULTICORE_AMO_MIN / 2).max(8)
        );
    }

    #[test]
    fn multicore_atomic_bias_only_bumps_atomic_extensions_rv64() {
        let mut counts = HashMap::from([
            (RV64Extensions::I, 5usize),
            (RV64Extensions::Zaamo, 5usize),
            (RV64Extensions::Zalrsc, 5usize),
            (RV64Extensions::Zawrs, 5usize),
            (RV64Extensions::Zifencei, 5usize),
            (RV64Extensions::Zicbom, 5usize),
            (RV64Extensions::Zicboz, 5usize),
            (RV64Extensions::Zalasr, 5usize),
            (RV64Extensions::Zabha, 5usize),
            (RV64Extensions::Zacas, 5usize),
        ]);

        apply_multicore_atomic_bias_rv64(true, &mut counts);

        assert_eq!(counts[&RV64Extensions::I], 5);
        assert_eq!(counts[&RV64Extensions::Zifencei], 5);
        assert_eq!(counts[&RV64Extensions::Zicbom], 5);
        assert_eq!(counts[&RV64Extensions::Zicboz], 5);
        assert_eq!(
            counts[&RV64Extensions::Zaamo],
            5usize
                .saturating_mul(MULTICORE_AMO_MUL)
                .max(MULTICORE_AMO_MIN)
        );
        assert_eq!(
            counts[&RV64Extensions::Zalrsc],
            5usize
                .saturating_mul(MULTICORE_LRSC_MUL)
                .max(MULTICORE_LRSC_MIN)
        );
        assert_eq!(
            counts[&RV64Extensions::Zawrs],
            5usize
                .saturating_mul(2)
                .max((MULTICORE_LRSC_MIN / 2).max(4))
        );
        assert_eq!(
            counts[&RV64Extensions::Zalasr],
            (MULTICORE_AMO_MIN / 2).max(8)
        );
        assert_eq!(
            counts[&RV64Extensions::Zabha],
            (MULTICORE_AMO_MIN / 2).max(8)
        );
        assert_eq!(
            counts[&RV64Extensions::Zacas],
            (MULTICORE_AMO_MIN / 2).max(8)
        );
    }

    #[test]
    fn rejects_invalid_multicore_sync_ratios() {
        let err = validate_multicore_sync_config(&crate::testcase::MulticoreSyncConfig {
            interaction_slot_ratio: 1.2,
            shared_address_ratio: 0.5,
            prefer_focused_sequences: true,
            ..Default::default()
        })
        .expect_err("ratio above 1 should be rejected");

        assert!(matches!(
            err,
            GenerationError::InvalidMulticoreSyncRatio {
                field: "interaction_slot_ratio",
                value
            } if value == 1.2
        ));
    }

    #[test]
    fn multicore_sync_metrics_capture_shared_address_overlap_by_cycle() {
        let testcase = generated_testcase(
            "ibex",
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
                        memory_base_register_number_range: (8, 12),
                    },
                    register_operand_config: RegisterOperandConfig {
                        integer_register_number_range: (0, 31),
                        floating_point_register_number_range: (0, 31),
                        vector_register_number_range: (0, 31),
                    },
                    ..RandomConfig::default()
                },
                seed: Some(7),
                hart_config: crate::testcase::HartConfig::Dual {
                    sync: crate::testcase::MulticoreSyncConfig::default(),
                },
            },
            flat_structured_program(
                vec![li(5, 0, LoadImmediatePurpose::Generic), rv32_sw(1, 0, 5)],
                vec![li(5, 0, LoadImmediatePurpose::Generic), rv32_lw(2, 0, 5)],
            ),
            IsaExtensions::rv32(vec![RV32Extensions::I]),
        );

        let output = ExecutionContextOutput {
            per_hart: BTreeMap::from([
                (
                    0,
                    HartExecutionContextOutput {
                        hart_id: 0,
                        instructions: Vec::new(),
                    },
                ),
                (
                    1,
                    HartExecutionContextOutput {
                        hart_id: 1,
                        instructions: Vec::new(),
                    },
                ),
            ]),
            framework_per_hart: BTreeMap::new(),
            logical_view: vec![
                logical_instruction_context_output(
                    0,
                    0,
                    0,
                    "li x5, 0",
                    instruction_context_with_addresses(&[]),
                    Vec::new(),
                    InstructionTiming::new(Some(10), Some(10), Some(1)),
                ),
                logical_instruction_context_output(
                    0,
                    1,
                    1,
                    "sw x1, 0(x5)",
                    instruction_context_with_addresses(&[0, 1, 2, 3]),
                    Vec::new(),
                    InstructionTiming::new(Some(11), Some(13), Some(3)),
                ),
                logical_instruction_context_output(
                    1,
                    2,
                    0,
                    "li x5, 0",
                    instruction_context_with_addresses(&[]),
                    Vec::new(),
                    InstructionTiming::new(Some(10), Some(10), Some(1)),
                ),
                logical_instruction_context_output(
                    1,
                    3,
                    1,
                    "lw x2, 0(x5)",
                    instruction_context_with_addresses(&[0, 1, 2, 3]),
                    Vec::new(),
                    InstructionTiming::new(Some(12), Some(14), Some(3)),
                ),
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
        assert_eq!(metrics.cycle_window_end, 14);
        assert_eq!(metrics.covered_cycle_count, 5);
        assert_eq!(metrics.simultaneous_cycle_count, 3);
        assert_eq!(metrics.same_address_cycle_count, 2);
        assert_eq!(metrics.both_memory_cycle_count, 2);
        assert_eq!(metrics.register_overlap_cycle_count, 3);
        assert_eq!(metrics.same_address_cycle_rate, 2.0 / 3.0);
        assert_eq!(metrics.same_address_given_both_memory_cycle_rate, 1.0);
        assert!(metrics.cycle_contention_score > 0.0);
    }

    #[test]
    fn memory_coupling_rewrites_true_address_match_without_cloning_opcode() {
        let hart0 = seq_with_pre(
            vec![li(5, 64, LoadImmediatePurpose::BaseAddress)],
            rv32_sw(18, 4, 5),
        );
        let hart1 = seq_with_pre(
            vec![li(7, 128, LoadImmediatePurpose::BaseAddress)],
            rv32_lw(21, 12, 7),
        );

        let (rewritten, register_match, memory_match) =
            attempt_memory_coupling(&hart0, &hart1).expect("memory coupling should succeed");

        assert!(register_match);
        assert!(memory_match);
        assert_eq!(sequence_true_memory_match_count(&hart0, &rewritten), 1);
        assert_eq!(rewritten.pre_instructions[0].to_string(), "li x5, 64");
        assert_eq!(rewritten.instruction.to_string(), "lw x18, 4(x5)");
        assert_eq!(
            normalized_instruction_mnemonic(&rewritten.instruction),
            "lw"
        );
    }

    #[test]
    fn random_pair_coupling_counts_only_real_successes() {
        let hart0 = vec![
            seq_with_pre(
                vec![li(5, 64, LoadImmediatePurpose::BaseAddress)],
                rv32_sw(18, 4, 5),
            ),
            InstructionSequence::new(rv32_add(11, 12, 13)),
            seq_with_pre(
                vec![li(6, 96, LoadImmediatePurpose::BaseAddress)],
                rv32_sw(19, 8, 6),
            ),
            InstructionSequence::new(rv32_xor(9, 10, 11)),
        ];
        let hart1 = vec![
            seq_with_pre(
                vec![li(7, 128, LoadImmediatePurpose::BaseAddress)],
                rv32_lw(21, 12, 7),
            ),
            InstructionSequence::new(rv32_sub(20, 19, 18)),
            seq_with_pre(
                vec![li(8, 160, LoadImmediatePurpose::BaseAddress)],
                rv32_lw(22, 0, 8),
            ),
            InstructionSequence::new(rv32_or(24, 25, 26)),
        ];

        let mut rng = StdRng::seed_from_u64(7);
        let (rewritten_hart1, plan) = couple_multicore_hart_streams(
            &hart0,
            &hart1,
            &crate::testcase::MulticoreSyncConfig {
                interaction_slot_ratio: 0.75,
                shared_address_ratio: 2.0 / 3.0,
                prefer_focused_sequences: true,
                ..Default::default()
            },
            &mut rng,
        );

        assert_eq!(plan.target_match_count, 3);
        assert_eq!(plan.matched_slots.len(), plan.target_match_count);
        assert_eq!(plan.target_shared_address_count, 2);
        assert_eq!(
            plan.shared_address_slots.len(),
            plan.target_shared_address_count
        );

        for matched in &plan.matched_slots {
            let before = &hart1[matched.hart1_slot];
            let after = &rewritten_hart1[matched.hart1_slot];
            let target = &hart0[matched.hart0_slot];

            assert_ne!(before.to_string(), after.to_string());

            if matched.register_match {
                assert!(
                    sequence_register_overlap_count(target, after)
                        > sequence_register_overlap_count(target, before)
                );
            }

            if matched.memory_match {
                assert!(
                    sequence_true_memory_match_count(target, after)
                        > sequence_true_memory_match_count(target, before)
                );
                assert!(plan.shared_address_slots.contains(&matched.hart1_slot));
                assert_eq!(normalized_instruction_mnemonic(&after.instruction), "lw");
            }
        }
    }

    #[test]
    fn multicore_sync_metrics_fallback_to_memory_changes_when_context_is_empty() {
        let testcase = generated_testcase(
            "ibex",
            TestCaseConfig {
                isa_base: ISABase::Rv32,
                extension_insts_count: 0,
                extension_inst_scaling: None,
                mem_size: 1,
                float_extension: FloatExtension::None,
                unaligned_access_required: Some(false),
                random_config: RandomConfig {
                    memory_access_config: MemoryAccessConfig {
                        memory_base_address_range: MemoryAccessConfig::default()
                            .memory_base_address_range,
                        memory_offset_immediate_range: (0, 0),
                        memory_base_register_number_range: (5, 5),
                    },
                    register_operand_config: RegisterOperandConfig {
                        integer_register_number_range: (5, 5),
                        floating_point_register_number_range: (0, 0),
                        vector_register_number_range: (0, 0),
                    },
                    ..RandomConfig::default()
                },
                seed: Some(7),
                hart_config: crate::testcase::HartConfig::Dual {
                    sync: MulticoreSyncConfig {
                        interaction_slot_ratio: 1.0,
                        shared_address_ratio: 1.0,
                        prefer_focused_sequences: true,
                        ..Default::default()
                    },
                },
            },
            flat_structured_program(vec![rv32_sw(5, 0, 5)], vec![rv32_sw(5, 0, 5)]),
            IsaExtensions::rv32(vec![RV32Extensions::I]),
        );

        let output = ExecutionContextOutput {
            per_hart: BTreeMap::from([
                (
                    0,
                    HartExecutionContextOutput {
                        hart_id: 0,
                        instructions: Vec::new(),
                    },
                ),
                (
                    1,
                    HartExecutionContextOutput {
                        hart_id: 1,
                        instructions: Vec::new(),
                    },
                ),
            ]),
            framework_per_hart: BTreeMap::new(),
            logical_view: vec![
                logical_instruction_context_output(
                    0,
                    0,
                    0,
                    "sw x5, 0(x5)",
                    instruction_context_with_addresses(&[]),
                    vec![
                        MemValue {
                            addr: 0x10,
                            value: 0xaa,
                        },
                        MemValue {
                            addr: 0x11,
                            value: 0xbb,
                        },
                    ],
                    InstructionTiming::new(Some(20), Some(22), Some(3)),
                ),
                logical_instruction_context_output(
                    1,
                    1,
                    0,
                    "sw x5, 0(x5)",
                    instruction_context_with_addresses(&[]),
                    vec![MemValue {
                        addr: 0x10,
                        value: 0xcc,
                    }],
                    InstructionTiming::new(Some(21), Some(23), Some(3)),
                ),
            ],
            mem_range: (0, 63),
            core_id: "ibex".to_string(),
            isa_base: ISABase::Rv32,
            perf: PerfMetrics::default(),
        };

        let metrics =
            compute_runtime_multicore_sync_metrics_from_context(&testcase, "ibex", &output)
                .expect("metrics");

        assert_eq!(metrics.both_memory_cycle_count, 2);
        assert_eq!(metrics.same_address_cycle_count, 2);
    }

    #[test]
    fn boom_multicore_sync_metrics_keep_raw_single_cycle_memory_ops() {
        let testcase = generated_testcase(
            "boom",
            TestCaseConfig {
                isa_base: ISABase::Rv32,
                extension_insts_count: 0,
                extension_inst_scaling: None,
                mem_size: 1,
                float_extension: FloatExtension::None,
                unaligned_access_required: Some(false),
                random_config: RandomConfig {
                    memory_access_config: MemoryAccessConfig {
                        memory_base_address_range: MemoryAccessConfig::default()
                            .memory_base_address_range,
                        memory_offset_immediate_range: (0, 0),
                        memory_base_register_number_range: (5, 7),
                    },
                    register_operand_config: RegisterOperandConfig {
                        integer_register_number_range: (5, 7),
                        floating_point_register_number_range: (0, 0),
                        vector_register_number_range: (0, 0),
                    },
                    ..RandomConfig::default()
                },
                seed: Some(11),
                hart_config: crate::testcase::HartConfig::Dual {
                    sync: MulticoreSyncConfig {
                        interaction_slot_ratio: 1.0,
                        shared_address_ratio: 1.0,
                        prefer_focused_sequences: true,
                        ..Default::default()
                    },
                },
            },
            flat_structured_program(
                vec![rv32_addi(5, 0, 0), rv32_sw(6, 0, 5)],
                vec![rv32_addi(5, 0, 0), rv32_sw(7, 0, 5)],
            ),
            IsaExtensions::rv32(vec![RV32Extensions::I]),
        );

        let output = ExecutionContextOutput {
            per_hart: BTreeMap::from([
                (
                    0,
                    HartExecutionContextOutput {
                        hart_id: 0,
                        instructions: Vec::new(),
                    },
                ),
                (
                    1,
                    HartExecutionContextOutput {
                        hart_id: 1,
                        instructions: Vec::new(),
                    },
                ),
            ]),
            framework_per_hart: BTreeMap::new(),
            logical_view: vec![
                logical_instruction_context_output(
                    0,
                    0,
                    0,
                    "addi x5, x0, 0",
                    instruction_context_with_addresses(&[]),
                    Vec::new(),
                    InstructionTiming::new(Some(10), Some(10), Some(1)),
                ),
                logical_instruction_context_output(
                    0,
                    1,
                    1,
                    "sw x6, 0(x5)",
                    instruction_context_with_addresses(&[0x10]),
                    vec![MemValue {
                        addr: 0x10,
                        value: 0xaa,
                    }],
                    InstructionTiming::new(Some(20), Some(20), Some(1)),
                ),
                logical_instruction_context_output(
                    1,
                    2,
                    0,
                    "addi x5, x0, 0",
                    instruction_context_with_addresses(&[]),
                    Vec::new(),
                    InstructionTiming::new(Some(12), Some(12), Some(1)),
                ),
                logical_instruction_context_output(
                    1,
                    3,
                    1,
                    "sw x7, 0(x5)",
                    instruction_context_with_addresses(&[0x10]),
                    vec![MemValue {
                        addr: 0x10,
                        value: 0xbb,
                    }],
                    InstructionTiming::new(Some(18), Some(18), Some(1)),
                ),
            ],
            mem_range: (0, 63),
            core_id: "boom".to_string(),
            isa_base: ISABase::Rv32,
            perf: PerfMetrics::default(),
        };

        let metrics =
            compute_runtime_multicore_sync_metrics_from_context(&testcase, "boom", &output)
                .expect("metrics");

        assert_eq!(metrics.covered_cycle_count, 4);
        assert_eq!(metrics.simultaneous_cycle_count, 0);
        assert_eq!(metrics.register_overlap_cycle_count, 0);
        assert_eq!(metrics.both_memory_cycle_count, 0);
        assert_eq!(metrics.same_address_cycle_count, 0);
        assert_eq!(metrics.cycle_contention_score, 0.0);
    }

    #[test]
    fn filter_extensions_for_float_mode_matches_requested_active_float_set() {
        let rv64 = IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::F,
            RV64Extensions::D,
            RV64Extensions::Zfa,
            RV64Extensions::Zfh,
            RV64Extensions::Zcd,
        ]);

        let none = filter_extensions_for_float_mode(rv64.clone(), FloatExtension::None);
        assert!(matches!(&none, IsaExtensions::Rv64(values) if values == &vec![RV64Extensions::I]));

        let single = filter_extensions_for_float_mode(rv64.clone(), FloatExtension::F);
        assert!(matches!(&single, IsaExtensions::Rv64(values)
            if values.contains(&RV64Extensions::I)
                && values.contains(&RV64Extensions::F)
                && values.contains(&RV64Extensions::Zfa)
                && !values.contains(&RV64Extensions::Zfh)
                && !values.contains(&RV64Extensions::D)
                && !values.contains(&RV64Extensions::Zcd)));

        let double = filter_extensions_for_float_mode(rv64, FloatExtension::FD);
        assert!(matches!(&double, IsaExtensions::Rv64(values)
            if values.contains(&RV64Extensions::I)
                && values.contains(&RV64Extensions::F)
                && values.contains(&RV64Extensions::D)
                && values.contains(&RV64Extensions::Zfa)
                && values.contains(&RV64Extensions::Zfh)
                && values.contains(&RV64Extensions::Zcd)));
    }

    #[test]
    fn rocket_metric_interval_uses_raw_timing_window() {
        let timing = InstructionTiming::new(Some(20), Some(22), Some(3));

        assert_eq!(effective_metric_cycle_interval(&timing), Some((20, 22)));
    }

    #[test]
    fn vex_metric_interval_uses_raw_timing_window() {
        let timing = InstructionTiming::new(Some(40), Some(52), Some(13));

        assert_eq!(effective_metric_cycle_interval(&timing), Some((40, 52)));
    }

    #[test]
    fn explicit_seed_makes_rng_deterministic() {
        let mut first = testcase_rng(Some(7));
        let mut second = testcase_rng(Some(7));

        assert_eq!(first.next_u64(), second.next_u64());
        assert_eq!(first.next_u64(), second.next_u64());
        assert_eq!(first.next_u64(), second.next_u64());
    }

    #[test]
    #[cfg(feature = "plugin-tests")]
    fn multicore_generation_builds_atomic_focused_hart_streams() {
        configure_plugin_dirs();
        let impls = PluginManager::normalize_impls(vec!["cva6".to_string(), "vex".to_string()]);
        let testcase = generate_random_testcase(
            &impls,
            TestCaseConfig {
                isa_base: ISABase::Rv32,
                extension_insts_count: 16,
                extension_inst_scaling: None,
                mem_size: 64,
                float_extension: FloatExtension::None,
                unaligned_access_required: Some(false),
                random_config: RandomConfig {
                    memory_access_config: MemoryAccessConfig {
                        memory_base_address_range: MemoryAccessConfig::default()
                            .memory_base_address_range,
                        memory_offset_immediate_range: (0, 16),
                        memory_base_register_number_range: (8, 12),
                    },
                    register_operand_config: RegisterOperandConfig {
                        integer_register_number_range: (13, 20),
                        floating_point_register_number_range: (13, 20),
                        vector_register_number_range: (13, 20),
                    },
                    ..RandomConfig::default()
                },
                seed: None,
                hart_config: crate::testcase::HartConfig::Dual {
                    sync: crate::testcase::MulticoreSyncConfig::default(),
                },
            },
        )
        .expect("multicore testcase generation should succeed");

        assert!(
            testcase.multicore_sync_metrics.is_none(),
            "generation should not synthesize slot-based sync metrics anymore"
        );

        fn looks_like_atomic(line: &str) -> bool {
            let mnemonic = line
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .trim_start_matches('.');

            mnemonic.starts_with("amo") || mnemonic.starts_with("lr") || mnemonic.starts_with("sc")
        }

        for impl_ref in &impls {
            let per_hart_lines = testcase
                .combined_rendered_pairs_per_hart_of(impl_ref)
                .expect("each implementation should have generated instructions")
                .into_iter()
                .map(|(hart_id, pairs)| {
                    (
                        hart_id,
                        pairs.into_iter().map(|(line, _)| line).collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            let hart0: Vec<String> = per_hart_lines
                .iter()
                .find(|(hart_id, _)| *hart_id == 0)
                .map(|(_, lines)| lines.clone())
                .unwrap_or_default();
            let hart1: Vec<String> = per_hart_lines
                .iter()
                .find(|(hart_id, _)| *hart_id == 1)
                .map(|(_, lines)| lines.clone())
                .unwrap_or_default();

            assert!(!hart0.is_empty(), "hart0 stream should not be empty");
            assert!(!hart1.is_empty(), "hart1 stream should not be empty");
            assert!(
                hart0.iter().any(|line| looks_like_atomic(line)),
                "hart0 should contain atomic-focused instructions"
            );
            assert!(
                hart1.iter().any(|line| looks_like_atomic(line)),
                "hart1 should contain atomic-focused instructions"
            );
        }
    }
}

#[cfg(all(test, feature = "plugin-tests"))]
mod integration_tests {
    use super::*;
    use std::path::PathBuf;

    const PLUGIN_DIRS_ENV: &str = "RISCV_FUZZ_PLUGIN_DIRS";

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .expect("workspace root")
            .to_path_buf()
    }

    fn configure_plugin_dirs() {
        let root = workspace_root();
        let joined = std::env::join_paths([
            root.join("target/debug"),
            root.join("target/release"),
            root.join("riscv_fuzz_plugins"),
        ])
        .expect("join plugin dirs");
        unsafe { std::env::set_var(PLUGIN_DIRS_ENV, joined) };
    }

    #[test]
    fn print_pairwise_extension_supersets() {
        use std::collections::HashSet;
        configure_plugin_dirs();
        let plugin_manager = PluginManager::global();
        let all_impls = plugin_manager.registered_impls();

        for isa in [ISABase::Rv32, ISABase::Rv64] {
            let supporting: Vec<String> = all_impls
                .iter()
                .filter(|impl_ref| plugin_manager.supports_isa_base(impl_ref, isa))
                .cloned()
                .collect();

            if supporting.len() < 2 {
                continue;
            }

            match isa {
                ISABase::Rv32 => {
                    let baseline_map = plugin_manager.extensions_for_impls(
                        &PluginManager::normalize_impls(supporting.clone()),
                        isa,
                    );
                    let baseline_vec = baseline_map
                        .as_rv32()
                        .expect("RV32 extensions expected for RV32 testcase");
                    let baseline_len = baseline_vec.len();
                    let baseline_set: HashSet<_> = baseline_vec.iter().copied().collect();

                    for (idx, left) in supporting.iter().enumerate() {
                        for right in supporting.iter().skip(idx + 1) {
                            let pair_map = plugin_manager.extensions_for_impls(
                                &PluginManager::normalize_impls(vec![
                                    (*left).to_string(),
                                    (*right).to_string(),
                                ]),
                                isa,
                            );
                            let pair_vec = pair_map
                                .as_rv32()
                                .expect("RV32 extensions expected for RV32 testcase");

                            if pair_vec.len() <= baseline_len {
                                continue;
                            }

                            let pair_set: HashSet<_> = pair_vec.iter().copied().collect();
                            let extras: Vec<_> =
                                pair_set.difference(&baseline_set).copied().collect();

                            if extras.is_empty() {
                                continue;
                            }

                            let shared = {
                                let mut shared: Vec<_> =
                                    pair_set.iter().map(|ext| format!("{:?}", ext)).collect();
                                shared.sort();
                                shared.join(", ")
                            };
                            let extras = {
                                let mut extras: Vec<_> =
                                    extras.into_iter().map(|ext| format!("{:?}", ext)).collect();
                                extras.sort();
                                extras.join(", ")
                            };

                            info!(
                                "[rv32] {} + {} -> {} shared extensions: [{}]; extra vs baseline: [{}]",
                                left,
                                right,
                                pair_set.len(),
                                shared,
                                extras
                            );
                        }
                    }
                }
                ISABase::Rv64 => {
                    let baseline_map = plugin_manager.extensions_for_impls(
                        &PluginManager::normalize_impls(supporting.clone()),
                        isa,
                    );
                    let baseline_vec = baseline_map
                        .as_rv64()
                        .expect("RV64 extensions expected for RV64 testcase");
                    let baseline_len = baseline_vec.len();
                    let baseline_set: HashSet<_> = baseline_vec.iter().copied().collect();

                    for (idx, left) in supporting.iter().enumerate() {
                        for right in supporting.iter().skip(idx + 1) {
                            let pair_map = plugin_manager.extensions_for_impls(
                                &PluginManager::normalize_impls(vec![
                                    (*left).to_string(),
                                    (*right).to_string(),
                                ]),
                                isa,
                            );
                            let pair_vec = pair_map
                                .as_rv64()
                                .expect("RV64 extensions expected for RV64 testcase");

                            if pair_vec.len() <= baseline_len {
                                continue;
                            }

                            let pair_set: HashSet<_> = pair_vec.iter().copied().collect();
                            let extras: Vec<_> =
                                pair_set.difference(&baseline_set).copied().collect();

                            if extras.is_empty() {
                                continue;
                            }

                            let shared = {
                                let mut shared: Vec<_> =
                                    pair_set.iter().map(|ext| format!("{:?}", ext)).collect();
                                shared.sort();
                                shared.join(", ")
                            };
                            let extras = {
                                let mut extras: Vec<_> =
                                    extras.into_iter().map(|ext| format!("{:?}", ext)).collect();
                                extras.sort();
                                extras.join(", ")
                            };

                            info!(
                                "[rv64] {} + {} -> {} shared extensions: [{}]; extra vs baseline: [{}]",
                                left,
                                right,
                                pair_set.len(),
                                shared,
                                extras
                            );
                        }
                    }
                }
            }
        }
    }
}
