//! Round-trip and default-value tests for the schema.

use riscv_fuzz_config::{
    CoverageMode, ExperimentConfig, FloatExtension, HartConfig, ISABase, RunnerMode,
};

#[test]
fn defaults_yield_an_rv64_random_diff_testcase_config() {
    let cfg = ExperimentConfig::default();
    assert!(matches!(cfg.testcase_config.isa_base, ISABase::Rv64));
    assert_eq!(cfg.testcase_config.hart_count(), 1);
    assert!(matches!(
        cfg.testcase_config.float_extension,
        FloatExtension::None
    ));
    assert!(matches!(
        cfg.testcase_config.hart_config,
        HartConfig::Single
    ));
    assert!(matches!(cfg.runner.mode, RunnerMode::RandomDiff));
    assert!(matches!(
        cfg.reporting.coverage_collection,
        CoverageMode::Off
    ));
    assert!(cfg.reporting.dedup.enabled);
    assert!(cfg.reporting.dedup.database_path.is_none());
    assert!(cfg.cores.selected.is_empty());
    assert!(riscv_fuzz_config::RandomDiffConfig::default().enable_reduction);
    assert_eq!(cfg.cores.concurrency, 1);
}

#[test]
fn empty_toml_round_trips_to_defaults() {
    let cfg: ExperimentConfig = toml::from_str("").expect("empty TOML must parse via defaults");
    let default = ExperimentConfig::default();
    let lhs = toml::to_string(&cfg).unwrap();
    let rhs = toml::to_string(&default).unwrap();
    assert_eq!(lhs, rhs);
}

#[test]
fn minimal_random_diff_testcase_config_parses() {
    let toml_text = r#"
[testcase_config]
isa_base = "Rv64"
extension_insts_count = 200
float_extension = "fd"
unaligned_access_required = false
hart_config = { kind = "single" }

[testcase_config.random_config.memory_access_config]
memory_base_address_range = [0, 0]
memory_offset_immediate_range = [0, 16]
memory_base_register_number_range = [8, 12]

[testcase_config.random_config.register_operand_config]
integer_register_number_range = [13, 20]
floating_point_register_number_range = [13, 20]
vector_register_number_range = [13, 20]

[runner]
mode = "random_diff"

[runner.random_diff]
max_iterations = 10
enable_reduction = false
minimization_min_history_iterations = 3

[reporting]
coverage_collection = "off"

[reporting.dedup]
enabled = true
database_path = "artifacts/local_diff_dedup.jsonl"

[cores]
selected = ["spike", "rocket"]
concurrency = 2
"#;
    let cfg: ExperimentConfig = toml::from_str(toml_text).expect("config must parse");
    assert_eq!(cfg.testcase_config.extension_insts_count, 200);
    assert!(matches!(
        cfg.testcase_config.float_extension,
        FloatExtension::FD
    ));
    assert!(matches!(cfg.runner.mode, RunnerMode::RandomDiff));
    let rd = cfg
        .runner
        .random_diff
        .as_ref()
        .expect("random_diff present");
    assert_eq!(rd.max_iterations, 10);
    assert_eq!(rd.minimization_min_history_iterations, 3);
    assert!(!rd.enable_reduction);
    assert!(cfg.reporting.dedup.enabled);
    assert_eq!(
        cfg.reporting.dedup.database_path.as_deref(),
        Some(std::path::Path::new("artifacts/local_diff_dedup.jsonl"))
    );
    assert_eq!(cfg.cores.selected, vec!["spike", "rocket"]);
    assert_eq!(cfg.cores.concurrency, 2);
}

#[test]
fn coverage_guided_section_parses() {
    let toml_text = r#"
[runner]
mode = "coverage_guided"

[runner.coverage_guided]
max_iterations = 100
mutations_per_candidate = 7
fresh_program_probability = 0.15
chaos_block_injection_probability = 0.25
instruction_block_count_range = [16, 64]
corpus_max_entries = 200
stats_print_interval = 25

[reporting]
coverage_collection = "light"
"#;
    let cfg: ExperimentConfig = toml::from_str(toml_text).expect("config must parse");
    let cov = cfg.runner.coverage_guided.as_ref().expect("cov present");
    assert_eq!(cov.max_iterations, 100);
    assert_eq!(cov.mutations_per_candidate, 7);
    assert_eq!(cov.instruction_block_count_range.min, 16);
    assert_eq!(cov.instruction_block_count_range.max, 64);
    assert_eq!(cov.corpus_max_entries, 200);
    assert!(matches!(
        cfg.reporting.coverage_collection,
        CoverageMode::Light
    ));
}

#[test]
fn dual_hart_with_sync_parses() {
    let toml_text = r#"
[testcase_config]
isa_base = "Rv64"
extension_insts_count = 200
float_extension = "fd"
unaligned_access_required = false
hart_config = { kind = "dual", sync = { shared_address_ratio = 0.0, interaction_slot_ratio = 0.0, independent_block_len_range = [4, 16], interaction_block_len_range = [2, 8], prefer_focused_sequences = true } }

[runner]
mode = "random_diff"

[runner.random_diff]
max_iterations = 5
minimization_min_history_iterations = 1
"#;
    let cfg: ExperimentConfig = toml::from_str(toml_text).expect("dual-hart parses");
    assert_eq!(cfg.testcase_config.hart_count(), 2);
    let sync = cfg
        .testcase_config
        .multicore_sync_config()
        .expect("multicore present");
    assert_eq!(sync.shared_address_ratio, 0.0);
    assert_eq!(sync.independent_block_len_range, (4, 16));
}

#[test]
fn rejects_unknown_top_level_section() {
    let toml_text = r#"
[bogus]
foo = 1
"#;
    let result: Result<ExperimentConfig, _> = toml::from_str(toml_text);
    assert!(result.is_err(), "unknown sections must be rejected");
}

#[test]
fn rejects_unknown_testcase_config_field() {
    let toml_text = r#"
[testcase_config]
isa_base = "Rv64"
extension_insts_count = 100
unaligned_access_required = false
hart_config = { kind = "single" }
weird_extra_field = 7
"#;
    let result: Result<ExperimentConfig, _> = toml::from_str(toml_text);
    assert!(result.is_err(), "unknown fields must be rejected");
}

#[test]
fn tuple_ranges_require_explicit_pairs() {
    let toml_text = r#"
[testcase_config]
isa_base = "Rv64"
extension_insts_count = 100
unaligned_access_required = false
hart_config = { kind = "single" }

[testcase_config.random_config.memory_access_config]
memory_base_address_range = [0, 0]
memory_offset_immediate_range = 16
memory_base_register_number_range = [8, 12]
"#;
    let result: Result<ExperimentConfig, _> = toml::from_str(toml_text);
    assert!(
        result.is_err(),
        "engine-native tuple ranges should reject scalar shorthand"
    );
}
