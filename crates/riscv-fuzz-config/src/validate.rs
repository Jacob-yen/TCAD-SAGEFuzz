//! Cross-field validation rules for [`ExperimentConfig`].
//!
//! Validation is split in two phases so the I/O-free portion can run before we
//! touch the filesystem or spawn plugin discovery:
//!
//! - [`validate_structural`]: pure, runs without any I/O. Catches structural
//!   errors (R1–R10) before we attempt to discover plugins.
//! - [`validate_against_runtime_metadata`]: requires a snapshot of discovered
//!   core metadata. Run once cores have been discovered to enforce R11
//!   (workload ↔ core capability compatibility).
//!
//! Each [`ValidationError`] variant maps to exactly one rule from the
//! refactor plan. Errors carry enough context for end users to fix their
//! configuration without reading source.

use std::collections::HashMap;

use riscv_fuzz_plugin_api::{
    ISABase,
    coverage::CoverageMode,
    metadata::{CorePluginMetadata, FloatExtension, IsaExecutionProfile},
};
use riscv_fuzz_testcase_engine::testcase::HartConfig;
use thiserror::Error;

use crate::schema::{ExperimentConfig, RunnerMode, UsizeRange};

#[derive(Debug, Error)]
pub enum ValidationError {
    // --- R1 ---
    #[error(
        "runner.mode = coverage_guided requires reporting.coverage_collection != off; got `off`"
    )]
    CoverageGuidedRequiresCoverage,

    // --- R2 / R3 ---
    #[error("testcase_config.hart_config.kind = single must not declare dual-hart sync state")]
    MulticoreSyncOnSingleHart,
    #[error("testcase_config.hart_config.kind = dual requires sync settings")]
    MulticoreSyncMissingForDualHart,

    // --- R4 ---
    #[error(
        "testcase_config.float_extension = none conflicts with testcase_config.random_config.csr_selection_config.floating_csr_operands_enabled = true"
    )]
    FloatNoneVsCsrFloating,
    #[error(
        "testcase_config.float_extension = none conflicts with testcase_config.random_config.append_fflags_readback = true"
    )]
    FloatNoneVsFflagsReadback,

    // --- R5 ---
    #[error(
        "testcase_config.random_config.csr_selection_config.floating_csr_operands_enabled = true requires testcase_config.float_extension != none"
    )]
    CsrFloatingRequiresFloatExtension,

    // --- R6 ---
    #[error(
        "testcase_config.random_config.csr_selection_config.vector_csr_operands_enabled = true is not supported in this build (V codegen unimplemented)"
    )]
    VectorCsrNotSupported,

    // --- R7 ---
    #[error(
        "testcase_config.random_config.memory_access_config.memory_offset_immediate_range upper bound {upper} exceeds mem_size * 1024 = {region_bytes}"
    )]
    MemoryAccessOffsetExceedsRegion { upper: i64, region_bytes: i64 },

    // --- R8 ---
    #[error(
        "testcase_config.random_config.memory_access_config.memory_base_register_number_range [{s_min}..={s_max}] overlaps register_operand_config.integer_register_number_range [{i_min}..={i_max}]"
    )]
    ScratchOverlapsIntegerRegisters {
        s_min: u8,
        s_max: u8,
        i_min: u8,
        i_max: u8,
    },

    // --- R9 ---
    #[error("runner.mode = {mode} requires section [runner.{section}] to be present")]
    RunnerSubtableMissing {
        mode: &'static str,
        section: &'static str,
    },
    #[error(
        "runner.mode = {mode} but unrelated section [runner.{section}] is also present; only one runner subtable is allowed at a time"
    )]
    RunnerSubtableExtraneous {
        mode: &'static str,
        section: &'static str,
    },

    // --- R10 ---
    #[error("{field} range is invalid: lower bound {min} > upper bound {max}")]
    RangeMinExceedsMax {
        field: &'static str,
        min: u64,
        max: u64,
    },

    // --- R11 ---
    #[error("core `{core}` is not compatible with workload: {reason}")]
    CoreIncompatibleWithWorkload { core: String, reason: String },
    #[error("core `{core}` was selected but no plugin with that core_id was discovered")]
    CoreNotDiscovered { core: String },
}

/// Run all I/O-free validation rules against `cfg`.
///
/// This collects the *first* failure for any given rule but does not
/// short-circuit between rules — callers see one error at a time, walking
/// rules R1 → R11 in order so reports stay deterministic.
pub fn validate_structural(cfg: &ExperimentConfig) -> Result<(), ValidationError> {
    let testcase_config = &cfg.testcase_config;
    let runner = &cfg.runner;
    let reporting = &cfg.reporting;
    let random_config = &testcase_config.random_config;
    let memory_access = &random_config.memory_access_config;
    let register_operands = &random_config.register_operand_config;
    let csr_selection = &random_config.csr_selection_config;

    // R1
    if matches!(runner.mode, RunnerMode::CoverageGuided)
        && matches!(reporting.coverage_collection, CoverageMode::Off)
    {
        return Err(ValidationError::CoverageGuidedRequiresCoverage);
    }

    // R2 / R3
    match &testcase_config.hart_config {
        HartConfig::Single => {}
        HartConfig::Dual { sync } => {
            check_usize_range(
                "testcase_config.hart_config.sync.independent_block_len_range",
                tuple_to_usize_range(sync.independent_block_len_range),
            )?;
            check_usize_range(
                "testcase_config.hart_config.sync.interaction_block_len_range",
                tuple_to_usize_range(sync.interaction_block_len_range),
            )?;
        }
    }
    match testcase_config.hart_count() {
        1 | 2 => {}
        count => {
            return Err(ValidationError::RangeMinExceedsMax {
                field: "testcase_config.hart_config",
                min: count as u64,
                max: 2,
            });
        }
    }

    // R4
    if matches!(testcase_config.float_extension, FloatExtension::None) {
        if csr_selection.floating_csr_operands_enabled {
            return Err(ValidationError::FloatNoneVsCsrFloating);
        }
        if random_config.append_fflags_readback {
            return Err(ValidationError::FloatNoneVsFflagsReadback);
        }
    }

    // R5
    if csr_selection.floating_csr_operands_enabled
        && matches!(testcase_config.float_extension, FloatExtension::None)
    {
        return Err(ValidationError::CsrFloatingRequiresFloatExtension);
    }

    // R6
    if csr_selection.vector_csr_operands_enabled {
        return Err(ValidationError::VectorCsrNotSupported);
    }

    // R7
    let region_bytes = (testcase_config.mem_size as i64).saturating_mul(1024);
    let upper = memory_access.memory_offset_immediate_range.1;
    if upper > region_bytes {
        return Err(ValidationError::MemoryAccessOffsetExceedsRegion {
            upper,
            region_bytes,
        });
    }

    // R8 — disjoint scratch / integer ranges.
    let s = memory_access.memory_base_register_number_range;
    let i = register_operands.integer_register_number_range;
    let overlaps = !(s.1 < i.0 || i.1 < s.0);
    if overlaps {
        return Err(ValidationError::ScratchOverlapsIntegerRegisters {
            s_min: s.0,
            s_max: s.1,
            i_min: i.0,
            i_max: i.1,
        });
    }

    // R10 — range sanity.
    check_u8_tuple(
        "testcase_config.random_config.memory_access_config.memory_base_register_number_range",
        memory_access.memory_base_register_number_range,
    )?;
    check_u8_tuple(
        "testcase_config.random_config.register_operand_config.integer_register_number_range",
        register_operands.integer_register_number_range,
    )?;
    check_u8_tuple(
        "testcase_config.random_config.register_operand_config.floating_point_register_number_range",
        register_operands.floating_point_register_number_range,
    )?;
    check_u8_tuple(
        "testcase_config.random_config.register_operand_config.vector_register_number_range",
        register_operands.vector_register_number_range,
    )?;
    check_i64_tuple(
        "testcase_config.random_config.memory_access_config.memory_offset_immediate_range",
        memory_access.memory_offset_immediate_range,
    )?;
    if let Some(scaling) = testcase_config.extension_inst_scaling.as_ref()
        && scaling.min_insts > scaling.max_insts
    {
        return Err(ValidationError::RangeMinExceedsMax {
            field: "testcase_config.extension_inst_scaling.{min_insts,max_insts}",
            min: scaling.min_insts as u64,
            max: scaling.max_insts as u64,
        });
    }
    if let Some(cov) = runner.coverage_guided.as_ref() {
        check_usize_range(
            "runner.coverage_guided.instruction_block_count_range",
            cov.instruction_block_count_range,
        )?;
    }

    // R9 — exactly one runner subtable, matching runner.mode.
    enforce_runner_subtables(runner)?;

    Ok(())
}

fn check_u8_tuple(name: &'static str, r: (u8, u8)) -> Result<(), ValidationError> {
    if r.0 > r.1 {
        Err(ValidationError::RangeMinExceedsMax {
            field: name,
            min: r.0 as u64,
            max: r.1 as u64,
        })
    } else {
        Ok(())
    }
}

fn check_i64_tuple(name: &'static str, r: (i64, i64)) -> Result<(), ValidationError> {
    if r.0 > r.1 {
        Err(ValidationError::RangeMinExceedsMax {
            field: name,
            min: r.0 as u64,
            max: r.1 as u64,
        })
    } else {
        Ok(())
    }
}

fn check_usize_range(name: &'static str, r: UsizeRange) -> Result<(), ValidationError> {
    if r.min > r.max {
        Err(ValidationError::RangeMinExceedsMax {
            field: name,
            min: r.min as u64,
            max: r.max as u64,
        })
    } else {
        Ok(())
    }
}

fn tuple_to_usize_range(range: (usize, usize)) -> UsizeRange {
    UsizeRange {
        min: range.0,
        max: range.1,
    }
}

type RunnerSubtableCheck = (&'static str, bool, fn(RunnerMode) -> bool);

fn enforce_runner_subtables(runner: &crate::schema::RunnerConfig) -> Result<(), ValidationError> {
    let pairs: [RunnerSubtableCheck; 4] = [
        ("replay", runner.replay.is_some(), |m| {
            matches!(m, RunnerMode::Replay)
        }),
        ("random_diff", runner.random_diff.is_some(), |m| {
            matches!(m, RunnerMode::RandomDiff)
        }),
        (
            "diff_against_spike",
            runner.diff_against_spike.is_some(),
            |m| matches!(m, RunnerMode::DiffAgainstSpike),
        ),
        ("coverage_guided", runner.coverage_guided.is_some(), |m| {
            matches!(m, RunnerMode::CoverageGuided)
        }),
    ];

    let mode_str = mode_str(&runner.mode);

    for (name, present, matches_mode) in pairs {
        let expected = matches_mode(runner.mode);
        if expected && !present {
            return Err(ValidationError::RunnerSubtableMissing {
                mode: mode_str,
                section: name,
            });
        }
        if !expected && present {
            return Err(ValidationError::RunnerSubtableExtraneous {
                mode: mode_str,
                section: name,
            });
        }
    }

    Ok(())
}

fn mode_str(m: &RunnerMode) -> &'static str {
    match m {
        RunnerMode::Replay => "replay",
        RunnerMode::RandomDiff => "random_diff",
        RunnerMode::DiffAgainstSpike => "diff_against_spike",
        RunnerMode::CoverageGuided => "coverage_guided",
    }
}

/// Validate that every selected core in `cfg.cores.selected` has a discovered
/// plugin whose metadata satisfies the workload (R11). On mismatch, the
/// returned error names the failing capability.
///
/// The `discovered` map is keyed by `core_id` (the new field name added in
/// step 6 of the refactor; until then plugin code still calls it `core_id`).
pub fn validate_against_runtime_metadata(
    cfg: &ExperimentConfig,
    discovered: &HashMap<String, CorePluginMetadata>,
) -> Result<(), ValidationError> {
    for core_id in &cfg.cores.selected {
        let Some(meta) = discovered.get(core_id) else {
            return Err(ValidationError::CoreNotDiscovered {
                core: core_id.clone(),
            });
        };
        validate_core_runtime_requirements(
            core_id,
            meta,
            cfg.testcase_config.isa_base,
            cfg.testcase_config.hart_count(),
            cfg.testcase_config.unaligned_access_required,
            cfg.testcase_config.float_extension,
            cfg.reporting.coverage_collection,
        )?;
    }
    Ok(())
}

pub fn validate_core_runtime_requirements(
    core: &str,
    metadata: &CorePluginMetadata,
    isa_base: ISABase,
    hart_count: u32,
    require_unaligned_memory_access: Option<bool>,
    float_extension: FloatExtension,
    coverage_mode: CoverageMode,
) -> Result<(), ValidationError> {
    let profile = profile_for(&metadata.isa_profiles, isa_base);
    check_core(
        core,
        isa_base,
        profile,
        hart_count,
        require_unaligned_memory_access,
        float_extension,
        coverage_mode,
    )
}

fn profile_for(
    profiles: &riscv_fuzz_plugin_api::metadata::IsaExecutionProfileMap,
    base: ISABase,
) -> &IsaExecutionProfile {
    match base {
        ISABase::Rv32 => &profiles.rv32,
        ISABase::Rv64 => &profiles.rv64,
    }
}

fn check_core(
    core: &str,
    isa_base: ISABase,
    profile: &IsaExecutionProfile,
    hart_count: u32,
    require_unaligned_memory_access: Option<bool>,
    float_extension: FloatExtension,
    coverage_mode: CoverageMode,
) -> Result<(), ValidationError> {
    if !profile.supported {
        return Err(ValidationError::CoreIncompatibleWithWorkload {
            core: core.into(),
            reason: format!("does not support {isa_base:?}"),
        });
    }
    if hart_count == 2 && !profile.supports_dual_hart {
        return Err(ValidationError::CoreIncompatibleWithWorkload {
            core: core.into(),
            reason: "workload requires dual-hart but core profile.supports_dual_hart = false"
                .into(),
        });
    }
    if !profile.coverage_mode_support.supports(coverage_mode) {
        return Err(ValidationError::CoreIncompatibleWithWorkload {
            core: core.into(),
            reason: format!("does not support reporting.coverage_collection = {coverage_mode:?}"),
        });
    }
    if let Some(require_unaligned_memory_access) = require_unaligned_memory_access {
        let alignment_mode = if require_unaligned_memory_access {
            riscv_fuzz_plugin_api::metadata::AlignmentMode::AllowUnaligned
        } else {
            riscv_fuzz_plugin_api::metadata::AlignmentMode::RequireAligned
        };
        if !profile.unaligned_access_support.supports(alignment_mode) {
            return Err(ValidationError::CoreIncompatibleWithWorkload {
                core: core.into(),
                reason: format!(
                    "does not support testcase_config.unaligned_access_required = {}",
                    require_unaligned_memory_access
                ),
            });
        }
    }
    let needs_float = !matches!(float_extension, FloatExtension::None);
    if needs_float && !profile.extensions.has_enabled_float() {
        return Err(ValidationError::CoreIncompatibleWithWorkload {
            core: core.into(),
            reason: format!(
                "testcase_config.float_extension = {:?} but core advertises no F/D/Q extension",
                float_extension
            ),
        });
    }
    if matches!(float_extension, FloatExtension::FD) {
        let has_d = match &profile.extensions {
            riscv_fuzz_plugin_api::IsaExtensions::Rv32(exts) => {
                exts.contains(&riscv_instruction::RV32Extensions::D)
            }
            riscv_fuzz_plugin_api::IsaExtensions::Rv64(exts) => {
                exts.contains(&riscv_instruction::RV64Extensions::D)
            }
        };
        if !has_d {
            return Err(ValidationError::CoreIncompatibleWithWorkload {
                core: core.into(),
                reason: "testcase_config.float_extension = fd requires D extension".into(),
            });
        }
    }
    Ok(())
}
