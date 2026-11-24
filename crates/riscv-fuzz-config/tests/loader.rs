//! Loader integration tests: overlays + CLI overrides.

use std::path::PathBuf;

use riscv_fuzz_config::{
    CliOverrides, RunnerMode, load_experiment_config, parse_experiment_config,
};
use tempfile::tempdir;

fn write(path: &std::path::Path, body: &str) {
    std::fs::write(path, body).unwrap();
}

fn base_random_diff() -> &'static str {
    r#"
[testcase_config]
isa_base = "Rv64"
extension_insts_count = 100
float_extension = "fd"
unaligned_access_required = false
hart_config = { kind = "single" }

[runner]
mode = "random_diff"

[runner.random_diff]
max_iterations = 10
minimization_min_history_iterations = 3

[reporting]
coverage_collection = "off"

[cores]
selected = ["spike"]
concurrency = 1
"#
}

#[test]
fn parses_inline_random_diff() {
    let cfg = parse_experiment_config(base_random_diff()).expect("parses");
    assert!(matches!(cfg.runner.mode, RunnerMode::RandomDiff));
    let rd = cfg.runner.random_diff.as_ref().unwrap();
    assert_eq!(rd.max_iterations, 10);
}

#[test]
fn overlay_replaces_section_wholesale() {
    let dir = tempdir().unwrap();
    let base = dir.path().join("base.toml");
    let overlay = dir.path().join("over.toml");
    write(&base, base_random_diff());
    write(
        &overlay,
        r#"
[runner]
mode = "coverage_guided"

[runner.coverage_guided]
max_iterations = 50
mutations_per_candidate = 4
fresh_program_probability = 0.1
chaos_block_injection_probability = 0.2
instruction_block_count_range = [16, 32]
corpus_max_entries = 100
stats_print_interval = 10

[reporting]
coverage_collection = "light"
"#,
    );

    let cfg = load_experiment_config(&base, &[overlay], &CliOverrides::default())
        .expect("overlay merge succeeds");
    assert!(matches!(cfg.runner.mode, RunnerMode::CoverageGuided));
    let cov = cfg.runner.coverage_guided.as_ref().unwrap();
    assert_eq!(cov.max_iterations, 50);
    assert_eq!(cfg.cores.selected, vec!["spike".to_string()]);
}

#[test]
fn cli_overrides_apply_last() {
    let dir = tempdir().unwrap();
    let base = dir.path().join("base.toml");
    write(&base, base_random_diff());

    let overrides = CliOverrides {
        cores: Some(vec!["rocket".to_string(), "spike".to_string()]),
        concurrency: Some(8),
        max_iterations: Some(999),
        run_dir: Some(PathBuf::from("/tmp/run")),
        seed: Some(42),
    };
    let cfg = load_experiment_config(&base, &[], &overrides).expect("loads");
    assert_eq!(cfg.cores.selected, vec!["rocket", "spike"]);
    assert_eq!(cfg.cores.concurrency, 8);
    assert_eq!(cfg.runner.seed, Some(42));
    assert_eq!(cfg.runner.random_diff.as_ref().unwrap().max_iterations, 999);
}

#[test]
fn rejects_unknown_top_level_section_in_overlay() {
    let dir = tempdir().unwrap();
    let base = dir.path().join("base.toml");
    let overlay = dir.path().join("over.toml");
    write(&base, base_random_diff());
    write(&overlay, "[bogus]\nfoo = 1\n");
    let err = load_experiment_config(&base, &[overlay], &CliOverrides::default())
        .expect_err("unknown section must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("bogus"), "msg = {msg}");
}

#[test]
fn validation_runs_after_merge() {
    let dir = tempdir().unwrap();
    let base = dir.path().join("base.toml");
    write(&base, base_random_diff());
    let overlay = dir.path().join("over.toml");
    write(
        &overlay,
        r#"
[testcase_config]
isa_base = "Rv64"
extension_insts_count = 100
float_extension = "fd"
unaligned_access_required = false
hart_config = { kind = "dual", sync = { independent_block_len_range = [8, 4] } }
"#,
    );
    let err = load_experiment_config(&base, &[overlay], &CliOverrides::default())
        .expect_err("invalid dual-hart sync range should trip");
    let msg = err.to_string();
    assert!(
        msg.contains("independent_block_len_range") || msg.contains("usize range"),
        "got: {err}"
    );
}
