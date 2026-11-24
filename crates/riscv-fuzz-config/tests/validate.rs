//! Validator negative tests for testcase_config-based schema.

use riscv_fuzz_config::{
    CoverageMode, ExperimentConfig, FloatExtension, HartConfig, MulticoreSyncConfig,
    ValidationError,
    schema::{CoverageGuidedConfig, ReportingConfig, RunnerConfig, RunnerMode, UsizeRange},
    validate_core_runtime_requirements, validate_structural,
};
use riscv_fuzz_plugin_api::{
    CorePluginMetadata, CoverageModeSupport, ISABase, IsaExecutionProfile, IsaExecutionProfileMap,
    IsaExtensions, metadata::UnalignedAccessSupport,
};
use riscv_instruction::RV64Extensions;

fn ok_random_diff() -> ExperimentConfig {
    ExperimentConfig {
        runner: RunnerConfig {
            mode: RunnerMode::RandomDiff,
            random_diff: Some(riscv_fuzz_config::RandomDiffConfig::default()),
            ..RunnerConfig::default()
        },
        ..ExperimentConfig::default()
    }
}

fn rv64_metadata(profile: IsaExecutionProfile) -> CorePluginMetadata {
    CorePluginMetadata {
        core_id: "test-core".to_string(),
        isa_profiles: IsaExecutionProfileMap {
            rv32: IsaExecutionProfile::unsupported(ISABase::Rv32),
            rv64: profile,
        },
        program_mem_range: (0, 4096),
    }
}

fn compatible_rv64_profile() -> IsaExecutionProfile {
    IsaExecutionProfile {
        supported: true,
        supports_dual_hart: true,
        coverage_mode_support: CoverageModeSupport::ALL,
        unaligned_access_support: UnalignedAccessSupport {
            aligned: true,
            unaligned: true,
        },
        extensions: IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::F,
            RV64Extensions::D,
        ]),
    }
}

#[test]
fn r1_coverage_guided_requires_coverage() {
    let cfg = ExperimentConfig {
        runner: RunnerConfig {
            mode: RunnerMode::CoverageGuided,
            coverage_guided: Some(CoverageGuidedConfig::default()),
            ..RunnerConfig::default()
        },
        reporting: ReportingConfig {
            coverage_collection: CoverageMode::Off,
            ..ReportingConfig::default()
        },
        ..ExperimentConfig::default()
    };
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::CoverageGuidedRequiresCoverage)
    ));
}

#[test]
fn r3_float_none_vs_csr_floating() {
    let mut cfg = ok_random_diff();
    cfg.testcase_config.float_extension = FloatExtension::None;
    cfg.testcase_config
        .random_config
        .csr_selection_config
        .floating_csr_operands_enabled = true;
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::FloatNoneVsCsrFloating)
    ));
}

#[test]
fn r4_float_none_vs_fflags_readback() {
    let mut cfg = ok_random_diff();
    cfg.testcase_config.float_extension = FloatExtension::None;
    cfg.testcase_config.random_config.append_fflags_readback = true;
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::FloatNoneVsFflagsReadback)
    ));
}

#[test]
fn r6_vector_csr_unsupported() {
    let mut cfg = ok_random_diff();
    cfg.testcase_config
        .random_config
        .csr_selection_config
        .vector_csr_operands_enabled = true;
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::VectorCsrNotSupported)
    ));
}

#[test]
fn r7_memory_offset_exceeds_region() {
    let mut cfg = ok_random_diff();
    cfg.testcase_config.mem_size = 1;
    cfg.testcase_config
        .random_config
        .memory_access_config
        .memory_offset_immediate_range = (0, 4096);
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::MemoryAccessOffsetExceedsRegion { .. })
    ));
}

#[test]
fn r8_scratch_overlaps_integer_registers() {
    let mut cfg = ok_random_diff();
    cfg.testcase_config
        .random_config
        .memory_access_config
        .memory_base_register_number_range = (5, 12);
    cfg.testcase_config
        .random_config
        .register_operand_config
        .integer_register_number_range = (10, 20);
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::ScratchOverlapsIntegerRegisters { .. })
    ));
}

#[test]
fn r9_runner_subtable_missing() {
    let cfg = ExperimentConfig {
        runner: RunnerConfig {
            mode: RunnerMode::CoverageGuided,
            coverage_guided: None,
            ..RunnerConfig::default()
        },
        reporting: ReportingConfig {
            coverage_collection: CoverageMode::Light,
            ..ReportingConfig::default()
        },
        ..ExperimentConfig::default()
    };
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::RunnerSubtableMissing { .. })
    ));
}

#[test]
fn r9_runner_subtable_extraneous() {
    let cfg = ExperimentConfig {
        runner: RunnerConfig {
            mode: RunnerMode::RandomDiff,
            random_diff: Some(riscv_fuzz_config::RandomDiffConfig::default()),
            coverage_guided: Some(CoverageGuidedConfig::default()),
            ..RunnerConfig::default()
        },
        ..ExperimentConfig::default()
    };
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::RunnerSubtableExtraneous { .. })
    ));
}

#[test]
fn r10_scaling_min_exceeds_max() {
    let mut cfg = ok_random_diff();
    cfg.testcase_config
        .extension_inst_scaling
        .as_mut()
        .expect("default scaling")
        .min_insts = 100;
    cfg.testcase_config
        .extension_inst_scaling
        .as_mut()
        .expect("default scaling")
        .max_insts = 10;
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::RangeMinExceedsMax { .. })
    ));
}

#[test]
fn r10_block_count_range_min_exceeds_max() {
    let cfg = ExperimentConfig {
        runner: RunnerConfig {
            mode: RunnerMode::CoverageGuided,
            coverage_guided: Some(CoverageGuidedConfig {
                instruction_block_count_range: UsizeRange { min: 100, max: 10 },
                ..CoverageGuidedConfig::default()
            }),
            ..RunnerConfig::default()
        },
        reporting: ReportingConfig {
            coverage_collection: CoverageMode::Light,
            ..ReportingConfig::default()
        },
        ..ExperimentConfig::default()
    };
    assert!(matches!(
        validate_structural(&cfg),
        Err(ValidationError::RangeMinExceedsMax { .. })
    ));
}

#[test]
fn random_diff_default_validates_clean() {
    let cfg = ok_random_diff();
    assert!(validate_structural(&cfg).is_ok());
}

#[test]
fn coverage_guided_default_validates_clean() {
    let cfg = ExperimentConfig {
        runner: RunnerConfig {
            mode: RunnerMode::CoverageGuided,
            coverage_guided: Some(CoverageGuidedConfig::default()),
            ..RunnerConfig::default()
        },
        reporting: ReportingConfig {
            coverage_collection: CoverageMode::Light,
            ..ReportingConfig::default()
        },
        ..ExperimentConfig::default()
    };
    assert!(validate_structural(&cfg).is_ok());
}

#[test]
fn dual_hart_random_diff_validates_clean() {
    let mut cfg = ok_random_diff();
    cfg.testcase_config.hart_config = HartConfig::Dual {
        sync: MulticoreSyncConfig::default(),
    };
    assert!(validate_structural(&cfg).is_ok());
}

#[test]
fn runtime_core_validation_accepts_compatible_metadata() {
    let metadata = rv64_metadata(compatible_rv64_profile());
    validate_core_runtime_requirements(
        "test-core",
        &metadata,
        ISABase::Rv64,
        2,
        Some(true),
        FloatExtension::FD,
        CoverageMode::Light,
    )
    .expect("compatible metadata should pass runtime validation");
}

#[test]
fn runtime_core_validation_rejects_missing_dual_hart_support() {
    let mut profile = compatible_rv64_profile();
    profile.supports_dual_hart = false;
    let metadata = rv64_metadata(profile);
    let err = validate_core_runtime_requirements(
        "test-core",
        &metadata,
        ISABase::Rv64,
        2,
        Some(true),
        FloatExtension::FD,
        CoverageMode::Light,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ValidationError::CoreIncompatibleWithWorkload { .. }
    ));
}

#[test]
fn runtime_core_validation_skips_alignment_check_when_unspecified() {
    let mut profile = compatible_rv64_profile();
    profile.unaligned_access_support = UnalignedAccessSupport {
        aligned: true,
        unaligned: false,
    };
    let metadata = rv64_metadata(profile);
    validate_core_runtime_requirements(
        "test-core",
        &metadata,
        ISABase::Rv64,
        1,
        None,
        FloatExtension::None,
        CoverageMode::Off,
    )
    .expect("alignment should be skipped when unspecified");
}
