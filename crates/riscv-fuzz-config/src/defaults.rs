//! Compile-time defaults for every configuration field.
//!
//! Every default value used by `schema.rs` lives here so users can find one
//! authoritative source for "what happens if I leave this field out". Helpers
//! are exposed as `pub(crate)` const fns and consumed via `#[serde(default = ...)]`.

#![allow(dead_code)]

use riscv_fuzz_plugin_api::{ISABase, coverage::CoverageMode, metadata::FloatExtension};

use crate::schema::{I64Range, RunnerMode, U8Range, UsizeRange};

// ----------- workload -----------
pub(crate) const fn workload_isa_base() -> ISABase {
    ISABase::Rv64
}
pub(crate) const fn workload_hart_count() -> u32 {
    1
}
pub(crate) const fn workload_require_unaligned_memory_access() -> bool {
    false
}
pub(crate) const fn workload_float_extension() -> FloatExtension {
    FloatExtension::None
}
pub(crate) const fn workload_target_extension_instruction_count() -> usize {
    200
}

// memory
pub(crate) const fn memory_region_size_kib() -> u64 {
    64
}
pub(crate) const fn memory_access_offset_range_bytes() -> I64Range {
    I64Range { min: 0, max: 16 }
}

// registers
pub(crate) const fn registers_scratch() -> U8Range {
    U8Range { min: 8, max: 12 }
}
pub(crate) const fn registers_integer() -> U8Range {
    U8Range { min: 13, max: 20 }
}
pub(crate) const fn registers_floating_point() -> U8Range {
    U8Range { min: 13, max: 20 }
}
pub(crate) const fn registers_vector() -> U8Range {
    U8Range { min: 13, max: 20 }
}

// csr operands
pub(crate) const fn csr_machine() -> bool {
    true
}
pub(crate) const fn csr_supervisor() -> bool {
    false
}
pub(crate) const fn csr_hypervisor() -> bool {
    false
}
pub(crate) const fn csr_virtual_supervisor() -> bool {
    false
}
pub(crate) const fn csr_rnmi() -> bool {
    false
}
pub(crate) const fn csr_floating() -> bool {
    false
}
pub(crate) const fn csr_vector() -> bool {
    false
}
pub(crate) const fn csr_debug() -> bool {
    false
}

// random
pub(crate) const fn random_append_fflags_readback() -> bool {
    false
}
pub(crate) const fn random_edge_case_probability() -> f64 {
    0.2
}

// scaling
pub(crate) const fn scaling_multiplier() -> f64 {
    1.0
}
pub(crate) const fn scaling_min_count() -> usize {
    20
}
pub(crate) const fn scaling_max_count() -> usize {
    1000
}

// multicore
pub(crate) const fn multicore_shared_address_ratio() -> f64 {
    0.70
}
pub(crate) const fn multicore_interaction_slot_ratio() -> f64 {
    0.35
}
pub(crate) const fn multicore_independent_block_length_range() -> UsizeRange {
    UsizeRange { min: 4, max: 16 }
}
pub(crate) const fn multicore_interaction_block_length_range() -> UsizeRange {
    UsizeRange { min: 2, max: 8 }
}
pub(crate) const fn multicore_prefer_focused_sequences() -> bool {
    true
}

// runner
pub(crate) const fn runner_mode() -> RunnerMode {
    RunnerMode::RandomDiff
}

// random_diff & diff_against_spike
pub(crate) const fn diff_max_iterations() -> usize {
    200
}
pub(crate) const fn diff_minimization_min_history_iterations() -> usize {
    50
}
pub(crate) const fn diff_enable_reduction() -> bool {
    true
}

// coverage_guided
pub(crate) const fn cov_max_iterations() -> usize {
    0 // 0 = unbounded
}
pub(crate) const fn cov_mutations_per_candidate() -> u32 {
    5
}
pub(crate) const fn cov_fresh_program_probability() -> f64 {
    0.10
}
pub(crate) const fn cov_chaos_block_injection_probability() -> f64 {
    0.20
}
pub(crate) const fn cov_instruction_block_count_range() -> UsizeRange {
    UsizeRange {
        min: 512,
        max: 2048,
    }
}
pub(crate) const fn cov_corpus_max_entries() -> usize {
    1000
}
pub(crate) const fn cov_stats_print_interval() -> u64 {
    100
}

// reporting
pub(crate) const fn reporting_coverage_collection() -> CoverageMode {
    CoverageMode::Off
}
pub(crate) const fn reporting_emit_execution_json() -> bool {
    false
}
pub(crate) const fn reporting_emit_execution_markdown() -> bool {
    false
}
pub(crate) const fn reporting_delete_successful_iteration_artifacts() -> bool {
    false
}
pub(crate) const fn reporting_delete_successful_run_artifacts() -> bool {
    false
}
pub(crate) const fn reporting_keep_uninteresting_workdirs() -> bool {
    false
}

pub(crate) const fn write_probe_enabled() -> bool {
    true
}
pub(crate) const fn write_probe_max_registers() -> usize {
    8
}
pub(crate) const fn write_probe_max_memory_bytes() -> usize {
    16
}
pub(crate) const fn diff_dedup_enabled() -> bool {
    true
}

// cores
pub(crate) const fn cores_concurrency() -> usize {
    1
}
pub(crate) const fn cores_execution_timeout_seconds() -> u64 {
    30
}

pub(crate) fn cores_default_search_directories() -> Vec<std::path::PathBuf> {
    riscv_fuzz_plugin_api::default_plugin_search_directories()
}
