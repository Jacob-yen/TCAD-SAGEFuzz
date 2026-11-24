//! Audit existing generated assembly without rewriting or discarding programs.
use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
};

use riscv_fuzz_testcase_engine::{
    rvv_catalog_legal::RvvSamplingProfile, rvv_program_legal::audit_program_asm,
};

fn programs_under(root: &Path, found: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if root.is_file() {
        if root.file_name().is_some_and(|name| name == "program.S") {
            found.push(root.to_path_buf());
        }
        return Ok(());
    }
    for item in fs::read_dir(root)? {
        let path = item?.path();
        if path.is_dir() {
            programs_under(&path, found)?;
        } else if path.file_name().is_some_and(|name| name == "program.S") {
            found.push(path);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 4 || !matches!(args[1].as_str(), "ara" | "xiangshan") {
        return Err(
            "usage: audit_rvv_programs <ara|xiangshan> <program.S-or-generated-root> <output.json>"
                .into(),
        );
    }
    let profile = RvvSamplingProfile::from_subject_and_env(&args[1])?;
    let mut paths = Vec::new();
    programs_under(Path::new(&args[2]), &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err("no program.S found".into());
    }
    let mut rows = Vec::with_capacity(paths.len());
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut checked = 0usize;
    let mut programs_with_issues = 0usize;
    for path in paths {
        let source = fs::read_to_string(&path)?;
        let audit = audit_program_asm(&source, profile);
        checked += audit.checked_vector_instructions;
        programs_with_issues += usize::from(!audit.issues.is_empty());
        for issue in &audit.issues {
            *counts.entry(issue.code).or_default() += 1;
        }
        rows.push(serde_json::json!({"program": path, "audit": audit}));
    }
    let result = serde_json::json!({
        "subject": args[1], "profile": profile,
        "programs": rows.len(), "checked_vector_instructions": checked,
        "programs_with_issues": programs_with_issues,
        "issue_counts": counts,
        "scope": "static test-body path; traps and mstatus.VS/FS are not modeled",
        "cases": rows,
    });
    fs::write(&args[3], serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
