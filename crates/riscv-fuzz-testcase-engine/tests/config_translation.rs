use std::path::PathBuf;

use riscv_fuzz_plugin_api::{ISABase, default_plugin_search_directories};
use riscv_fuzz_testcase_engine::cov_guided::CovGuidedSettings;
use riscv_fuzz_testcase_engine::testcase::{
    ExtensionInstScaling, HartConfig, MulticoreSyncConfig, TestCaseConfig,
};

#[test]
fn testcase_default_matches_public_schema_defaults() {
    let tc = TestCaseConfig::default();
    assert!(matches!(tc.isa_base, ISABase::Rv64));
    assert_eq!(tc.extension_insts_count, 200);
    assert_eq!(
        tc.extension_inst_scaling.as_ref(),
        Some(&ExtensionInstScaling {
            multiplier: 1.0,
            min_insts: 20,
            max_insts: 1000,
        })
    );
    assert!(matches!(tc.hart_config, HartConfig::Single));
    assert_eq!(
        tc.random_config
            .register_operand_config
            .integer_register_number_range,
        (13, 20)
    );
    assert_eq!(
        tc.random_config
            .register_operand_config
            .floating_point_register_number_range,
        (13, 20)
    );
    assert_eq!(
        tc.random_config
            .register_operand_config
            .vector_register_number_range,
        (13, 20)
    );
    assert_eq!(
        tc.random_config
            .memory_access_config
            .memory_offset_immediate_range,
        (0, 16)
    );
    assert_eq!(
        tc.random_config
            .memory_access_config
            .memory_base_register_number_range,
        (8, 12)
    );
    assert!(
        tc.random_config
            .csr_selection_config
            .machine_csr_operands_enabled
    );
    assert_eq!(tc.unaligned_access_required, Some(false));
    assert!(tc.multicore_sync_config().is_none());
}

#[test]
fn dual_hart_config_exposes_sync_fields_directly() {
    let tc = TestCaseConfig {
        hart_config: HartConfig::Dual {
            sync: MulticoreSyncConfig {
                shared_address_ratio: 0.25,
                interaction_slot_ratio: 0.5,
                independent_block_len_range: (4, 8),
                interaction_block_len_range: (2, 4),
                prefer_focused_sequences: false,
            },
        },
        ..TestCaseConfig::default()
    };
    let sync = tc
        .multicore_sync_config()
        .expect("dual-hart config must carry sync config");
    assert!(matches!(tc.hart_config, HartConfig::Dual { .. }));
    assert_eq!(sync.shared_address_ratio, 0.25);
    assert_eq!(sync.interaction_slot_ratio, 0.5);
    assert_eq!(sync.independent_block_len_range, (4, 8));
    assert_eq!(sync.interaction_block_len_range, (2, 4));
    assert!(!sync.prefer_focused_sequences);
}

#[test]
fn coverage_guided_settings_accept_runtime_testcase_config_shape() {
    let settings = CovGuidedSettings {
        testcase_config: TestCaseConfig::default(),
        core_id: "spike".to_string(),
        run_dir: PathBuf::from("artifacts"),
        max_runs: Some(100),
        coverage_mode: riscv_fuzz_plugin_api::coverage::CoverageMode::Off,
        coverage_profile: Some("line-branch".to_string()),
        coverage_types: Some(vec!["line".to_string(), "branch".to_string()]),
        seed: None,
        mutation_steps: 7,
        fresh_probability: 0.1,
        spice_probability: 0.3,
        min_sequences: 16,
        max_sequences: 64,
        max_corpus: 500,
        stats_every: 25,
        keep_noninteresting: false,
    };

    assert_eq!(settings.min_sequences, 16);
    assert_eq!(settings.max_sequences, 64);
    assert_eq!(settings.mutation_steps, 7);
    assert_eq!(settings.fresh_probability, 0.1);
    assert_eq!(settings.spice_probability, 0.3);
    assert_eq!(settings.stats_every, 25);
    assert_eq!(settings.coverage_profile.as_deref(), Some("line-branch"));
    assert_eq!(
        settings.coverage_types.as_ref().expect("coverage types"),
        &vec!["line".to_string(), "branch".to_string()]
    );
}

#[test]
fn plugin_default_search_directories_stay_single_sourced() {
    assert_eq!(
        default_plugin_search_directories(),
        vec![PathBuf::from("target/release")]
    );
}
