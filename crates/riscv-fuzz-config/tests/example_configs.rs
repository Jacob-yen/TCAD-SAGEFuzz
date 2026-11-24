//! Smoke-test that every shipped configs/experiments/*.toml parses and validates.

use std::path::PathBuf;

use riscv_fuzz_config::{CliOverrides, load_experiment_config};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn every_shipped_experiment_loads_and_validates() {
    let dir = workspace_root().join("configs").join("experiments");
    assert!(dir.is_dir(), "configs/experiments missing at {dir:?}");

    let mut count = 0usize;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        let result = load_experiment_config(&path, &[], &CliOverrides::default());
        if let Err(err) = &result {
            panic!("failed to load {}: {err}", path.display());
        }
        count += 1;
    }
    assert!(
        count >= 5,
        "expected at least 5 example experiments, got {count}"
    );
}
