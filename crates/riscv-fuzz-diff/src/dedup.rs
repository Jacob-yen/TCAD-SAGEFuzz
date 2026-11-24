//! Automatic persistent de-duplication for diff findings.
//!
//! The store intentionally separates a human-facing family tag from the strict
//! duplicate key. The family tag helps browsing related reports, while the
//! strict fingerprint is the only key used to call a finding a duplicate.

use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;

use super::{
    detection::{ExceptionDifference, WriteDifference},
    types::{DiffAnalysisError, DiffDedupSettings},
    verification::InitializationMismatchData,
};

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_STORE_FILE: &str = ".riscv_fuzz_diff_dedup.jsonl";

#[derive(Debug, Clone)]
pub(crate) struct DiffDedupEvent {
    pub(crate) kind: &'static str,
    pub(crate) auto_tag: String,
    pub(crate) family_fingerprint: String,
    pub(crate) strict_fingerprint: String,
    pub(crate) summary: String,
    pub(crate) collision_guard: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DiffDedupStatus {
    FirstSeen,
    Duplicate {
        previous_count: usize,
        first_seen_run_root: Option<String>,
        first_seen_report_path: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct DiffDedupOutcome {
    pub(crate) event: DiffDedupEvent,
    pub(crate) status: DiffDedupStatus,
    pub(crate) database_path: PathBuf,
    pub(crate) malformed_records_ignored: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExceptionDedupKind {
    Trigger,
    Cause,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredDedupRecord {
    schema_version: u32,
    event_kind: String,
    auto_tag: String,
    family_fingerprint: String,
    strict_fingerprint: String,
    created_unix_seconds: u64,
    run_root: String,
    report_path: String,
    test_index: usize,
    global_index: usize,
    impl_order: Vec<String>,
    summary: String,
    collision_guard: String,
}

struct StoreLock {
    path: PathBuf,
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub(crate) fn record_write_difference(
    settings: &DiffDedupSettings,
    run_root: &Path,
    report_path: &Path,
    diff: &WriteDifference,
    impl_order: &[String],
    testcase: &GeneratedTestCase,
) -> Result<Option<DiffDedupOutcome>, DiffAnalysisError> {
    if !settings.enabled {
        return Ok(None);
    }

    let event = build_write_dedup_event(diff, impl_order, testcase)?;
    record_event(
        settings,
        run_root,
        report_path,
        event,
        diff.test_index,
        diff.global_index,
        impl_order,
    )
}

pub(crate) fn record_exception_difference(
    settings: &DiffDedupSettings,
    run_root: &Path,
    report_path: &Path,
    diff: &ExceptionDifference,
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    kind: ExceptionDedupKind,
) -> Result<Option<DiffDedupOutcome>, DiffAnalysisError> {
    if !settings.enabled {
        return Ok(None);
    }

    let event = build_exception_dedup_event(diff, impl_order, testcase, kind)?;
    record_event(
        settings,
        run_root,
        report_path,
        event,
        diff.test_index,
        diff.global_index,
        impl_order,
    )
}

pub(crate) fn record_initialization_mismatch(
    settings: &DiffDedupSettings,
    run_root: &Path,
    report_path: &Path,
    mismatch: &InitializationMismatchData,
    impl_order: &[String],
    testcase: &GeneratedTestCase,
) -> Result<Option<DiffDedupOutcome>, DiffAnalysisError> {
    if !settings.enabled {
        return Ok(None);
    }

    let event = build_initialization_dedup_event(mismatch, impl_order, testcase);
    record_event(
        settings,
        run_root,
        report_path,
        event,
        mismatch.instruction_index,
        mismatch.instruction_index,
        impl_order,
    )
}

fn record_event(
    settings: &DiffDedupSettings,
    run_root: &Path,
    report_path: &Path,
    event: DiffDedupEvent,
    test_index: usize,
    global_index: usize,
    impl_order: &[String],
) -> Result<Option<DiffDedupOutcome>, DiffAnalysisError> {
    let database_path = resolve_database_path(settings, run_root);
    let _lock = acquire_store_lock(&database_path)?;

    let mut previous_count = 0usize;
    let mut first_seen_run_root = None;
    let mut first_seen_report_path = None;
    let mut malformed_records_ignored = 0usize;

    if database_path.is_file() {
        let file = File::open(&database_path)?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record = match serde_json::from_str::<StoredDedupRecord>(&line) {
                Ok(record) => record,
                Err(_) => {
                    malformed_records_ignored += 1;
                    continue;
                }
            };
            if record.event_kind == event.kind
                && record.strict_fingerprint == event.strict_fingerprint
            {
                previous_count += 1;
                if first_seen_run_root.is_none() {
                    first_seen_run_root = Some(record.run_root);
                    first_seen_report_path = Some(record.report_path);
                }
            }
        }
    }

    let record = StoredDedupRecord {
        schema_version: SCHEMA_VERSION,
        event_kind: event.kind.to_string(),
        auto_tag: event.auto_tag.clone(),
        family_fingerprint: event.family_fingerprint.clone(),
        strict_fingerprint: event.strict_fingerprint.clone(),
        created_unix_seconds: unix_seconds_now(),
        run_root: run_root.display().to_string(),
        report_path: report_path.display().to_string(),
        test_index,
        global_index,
        impl_order: impl_order.to_vec(),
        summary: event.summary.clone(),
        collision_guard: event.collision_guard.to_string(),
    };

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&database_path)?;
    serde_json::to_writer(&mut file, &record)?;
    file.write_all(b"\n")?;

    let status = if previous_count == 0 {
        DiffDedupStatus::FirstSeen
    } else {
        DiffDedupStatus::Duplicate {
            previous_count,
            first_seen_run_root,
            first_seen_report_path,
        }
    };

    Ok(Some(DiffDedupOutcome {
        event,
        status,
        database_path,
        malformed_records_ignored,
    }))
}

fn resolve_database_path(settings: &DiffDedupSettings, run_root: &Path) -> PathBuf {
    let path = if let Some(path) = &settings.database_path {
        path.clone()
    } else {
        default_database_path(run_root)
    };
    absolutize_existing_or_new_path(&path)
}

fn absolutize_existing_or_new_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .unwrap_or_else(|_| path.to_path_buf())
}

fn default_database_path(run_root: &Path) -> PathBuf {
    run_root
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(run_root)
        .join(DEFAULT_STORE_FILE)
}

fn acquire_store_lock(database_path: &Path) -> io::Result<StoreLock> {
    if let Some(parent) = database_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let lock_path = lock_path_for(database_path);
    for _ in 0..250 {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock_path)
        {
            Ok(_) => return Ok(StoreLock { path: lock_path }),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {
                thread::sleep(Duration::from_millis(20));
            }
            Err(err) => return Err(err),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        format!(
            "timed out acquiring diff dedup store lock {}",
            lock_path.display()
        ),
    ))
}

fn lock_path_for(path: &Path) -> PathBuf {
    let mut raw = path.as_os_str().to_os_string();
    raw.push(".lock");
    PathBuf::from(raw)
}

fn build_write_dedup_event(
    diff: &WriteDifference,
    impl_order: &[String],
    testcase: &GeneratedTestCase,
) -> Result<DiffDedupEvent, DiffAnalysisError> {
    let instruction = testcase
        .traceable_test_instruction_at(diff.test_index)
        .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
            details: format!("missing traceable test instruction {}", diff.test_index),
        })?;
    build_write_dedup_event_for_instruction(diff, impl_order, &instruction)
}

fn build_write_dedup_event_for_instruction(
    diff: &WriteDifference,
    impl_order: &[String],
    instruction: &str,
) -> Result<DiffDedupEvent, DiffAnalysisError> {
    let mnemonic = instruction
        .split_whitespace()
        .next()
        .unwrap_or("<unknown>")
        .to_ascii_lowercase();
    let instruction_profile = normalize_instruction_profile(instruction);
    let mode = write_effect_mode(diff, impl_order)?;

    let mut family_by_impl = BTreeMap::new();
    let mut strict_by_impl = BTreeMap::new();
    for impl_ref in impl_order {
        let regs = diff.register_changes.get(impl_ref).ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing register write entry for instruction {} in {}",
                    diff.global_index, impl_ref
                ),
            }
        })?;
        let mem = diff.memory_changes.get(impl_ref).ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing memory write entry for instruction {} in {}",
                    diff.global_index, impl_ref
                ),
            }
        })?;
        let (family, strict) = effect_signatures(regs, mem);
        family_by_impl.insert(impl_ref.clone(), family);
        strict_by_impl.insert(impl_ref.clone(), strict);
    }

    let family_groups = grouped_signature(&family_by_impl);
    let strict_groups = grouped_signature(&strict_by_impl);
    let partition = partition_label(&strict_by_impl);
    let auto_tag = format!("WRITE/{mnemonic}/{mode}/{partition}");

    let family_basis = format!(
        "schema={SCHEMA_VERSION}|kind=write|mnemonic={mnemonic}|instruction={instruction_profile}|mode={mode}|groups={family_groups}"
    );
    let strict_basis = format!(
        "schema={SCHEMA_VERSION}|kind=write|mnemonic={mnemonic}|instruction={instruction_profile}|mode={mode}|groups={strict_groups}"
    );

    let family_fingerprint = format!("WRFAM-{:016x}", stable_fnv1a_64(&family_basis));
    let strict_fingerprint = format!("WRSTRICT-{:016x}", stable_fnv1a_64(&strict_basis));
    let summary = format!(
        "write diff at test #{} global #{}: {} ({mode}, {partition})",
        diff.test_index, diff.global_index, mnemonic
    );

    Ok(DiffDedupEvent {
        kind: "write",
        auto_tag,
        family_fingerprint,
        strict_fingerprint,
        summary,
        collision_guard: "duplicate matching uses the strict fingerprint: normalized instruction, core outcome partition, register targets, relative memory byte layout, and exact write values must all match",
    })
}

fn build_exception_dedup_event(
    diff: &ExceptionDifference,
    impl_order: &[String],
    testcase: &GeneratedTestCase,
    kind: ExceptionDedupKind,
) -> Result<DiffDedupEvent, DiffAnalysisError> {
    let instruction = testcase
        .traceable_test_instruction_at(diff.test_index)
        .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
            details: format!("missing traceable test instruction {}", diff.test_index),
        })?;
    let mnemonic = instruction
        .split_whitespace()
        .next()
        .unwrap_or("<unknown>")
        .to_ascii_lowercase();
    let instruction_profile = normalize_instruction_profile(&instruction);
    let kind_label = match kind {
        ExceptionDedupKind::Trigger => "trigger",
        ExceptionDedupKind::Cause => "cause",
    };

    let mut family_by_impl = BTreeMap::new();
    let mut strict_by_impl = BTreeMap::new();
    for impl_ref in impl_order {
        let causes = diff.per_impl.get(impl_ref).ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing exception entry for instruction {} in {}",
                    diff.global_index, impl_ref
                ),
            }
        })?;
        let mut codes = causes.iter().map(|cause| cause.code).collect::<Vec<_>>();
        codes.sort_unstable();
        let family = if codes.is_empty() {
            "trap=none".to_string()
        } else {
            format!("trap=count={};codes={}", codes.len(), code_classes(&codes))
        };
        let strict = if codes.is_empty() {
            "trap=none".to_string()
        } else {
            format!(
                "trap=count={};codes={}",
                codes.len(),
                codes
                    .iter()
                    .map(|code| format!("0x{code:x}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        family_by_impl.insert(impl_ref.clone(), family);
        strict_by_impl.insert(impl_ref.clone(), strict);
    }

    let family_groups = grouped_signature(&family_by_impl);
    let strict_groups = grouped_signature(&strict_by_impl);
    let partition = partition_label(&strict_by_impl);
    let auto_tag = format!("EXCEPTION/{kind_label}/{mnemonic}/{partition}");
    let family_basis = format!(
        "schema={SCHEMA_VERSION}|kind=exception-{kind_label}|mnemonic={mnemonic}|instruction={instruction_profile}|groups={family_groups}"
    );
    let strict_basis = format!(
        "schema={SCHEMA_VERSION}|kind=exception-{kind_label}|mnemonic={mnemonic}|instruction={instruction_profile}|groups={strict_groups}"
    );

    Ok(DiffDedupEvent {
        kind: match kind {
            ExceptionDedupKind::Trigger => "exception-trigger",
            ExceptionDedupKind::Cause => "exception-cause",
        },
        auto_tag,
        family_fingerprint: format!("EXFAM-{:016x}", stable_fnv1a_64(&family_basis)),
        strict_fingerprint: format!("EXSTRICT-{:016x}", stable_fnv1a_64(&strict_basis)),
        summary: format!(
            "exception {kind_label} diff at test #{} global #{}: {} ({partition})",
            diff.test_index, diff.global_index, mnemonic
        ),
        collision_guard: "duplicate matching uses the strict fingerprint: normalized instruction, core exception partition, and exact exception cause codes must all match",
    })
}

fn build_initialization_dedup_event(
    mismatch: &InitializationMismatchData,
    impl_order: &[String],
    testcase: &GeneratedTestCase,
) -> DiffDedupEvent {
    let instruction = impl_order
        .iter()
        .find_map(|impl_ref| {
            testcase.init_rendered_line_at_of(impl_ref, mismatch.instruction_index)
        })
        .unwrap_or_else(|| "<missing-init-instruction>".to_string());
    let mnemonic = instruction
        .split_whitespace()
        .next()
        .unwrap_or("<unknown>")
        .to_ascii_lowercase();
    let instruction_profile = normalize_instruction_profile(&instruction);

    let mut family_by_impl = BTreeMap::new();
    let mut strict_by_impl = BTreeMap::new();
    for impl_ref in impl_order {
        let regs = mismatch
            .per_impl_registers
            .get(impl_ref)
            .cloned()
            .unwrap_or_default();
        let mem = mismatch
            .per_impl_memory
            .get(impl_ref)
            .cloned()
            .unwrap_or_default();
        let (family, strict) = effect_signatures(&regs, &mem);
        family_by_impl.insert(impl_ref.clone(), family);
        strict_by_impl.insert(impl_ref.clone(), strict);
    }

    let family_groups = grouped_signature(&family_by_impl);
    let strict_groups = grouped_signature(&strict_by_impl);
    let partition = partition_label(&strict_by_impl);
    let normalized_reason = normalize_instruction_profile(&mismatch.reason);
    let auto_tag = format!("INIT/{mnemonic}/{partition}");
    let family_basis = format!(
        "schema={SCHEMA_VERSION}|kind=init|mnemonic={mnemonic}|instruction={instruction_profile}|reason={normalized_reason}|groups={family_groups}"
    );
    let strict_basis = format!(
        "schema={SCHEMA_VERSION}|kind=init|mnemonic={mnemonic}|instruction={instruction_profile}|reason={normalized_reason}|groups={strict_groups}"
    );

    DiffDedupEvent {
        kind: "initialization",
        auto_tag,
        family_fingerprint: format!("INFAM-{:016x}", stable_fnv1a_64(&family_basis)),
        strict_fingerprint: format!("INSTRICT-{:016x}", stable_fnv1a_64(&strict_basis)),
        summary: format!(
            "initialization diff at init #{}: {} ({partition})",
            mismatch.instruction_index, mnemonic
        ),
        collision_guard: "duplicate matching uses the strict fingerprint: normalized init instruction, normalized mismatch reason, core outcome partition, register targets, relative memory byte layout, and exact write values must all match",
    }
}

fn effect_signatures(regs: &[(String, u64)], mem: &[(u64, u8)]) -> (String, String) {
    let family_regs = regs
        .iter()
        .map(|(name, value)| {
            format!(
                "{}:{}",
                normalize_register_name(name),
                value_class_u64(*value)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let strict_regs = regs
        .iter()
        .map(|(name, value)| format!("{}=0x{value:016x}", normalize_register_name(name)))
        .collect::<Vec<_>>()
        .join(",");

    let (family_mem, strict_mem) = memory_signatures(mem);
    (
        format!("regs=[{family_regs}];mem=[{family_mem}]"),
        format!("regs=[{strict_regs}];mem=[{strict_mem}]"),
    )
}

fn memory_signatures(mem: &[(u64, u8)]) -> (String, String) {
    if mem.is_empty() {
        return ("-".to_string(), "-".to_string());
    }

    let min_addr = mem.iter().map(|(addr, _)| *addr).min().unwrap_or(0);
    let max_addr = mem.iter().map(|(addr, _)| *addr).max().unwrap_or(min_addr);
    let mut family = Vec::with_capacity(mem.len());
    let mut strict = Vec::with_capacity(mem.len());
    for (addr, value) in mem {
        let offset = addr.saturating_sub(min_addr);
        family.push(format!("{offset}:{}", value_class_u8(*value)));
        strict.push(format!("{offset}=0x{value:02x}"));
    }
    (
        format!(
            "count={};span={};bytes={}",
            mem.len(),
            max_addr.saturating_sub(min_addr) + 1,
            family.join(",")
        ),
        format!(
            "count={};span={};bytes={}",
            mem.len(),
            max_addr.saturating_sub(min_addr) + 1,
            strict.join(",")
        ),
    )
}

fn write_effect_mode(
    diff: &WriteDifference,
    impl_order: &[String],
) -> Result<&'static str, DiffAnalysisError> {
    let mut has_regs = false;
    let mut has_mem = false;
    for impl_ref in impl_order {
        let regs = diff.register_changes.get(impl_ref).ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing register write entry for instruction {} in {}",
                    diff.global_index, impl_ref
                ),
            }
        })?;
        let mem = diff.memory_changes.get(impl_ref).ok_or_else(|| {
            DiffAnalysisError::TestInstructionCountMismatch {
                details: format!(
                    "missing memory write entry for instruction {} in {}",
                    diff.global_index, impl_ref
                ),
            }
        })?;
        has_regs |= !regs.is_empty();
        has_mem |= !mem.is_empty();
    }
    Ok(match (has_regs, has_mem) {
        (true, true) => "reg+mem",
        (true, false) => "reg",
        (false, true) => "mem",
        (false, false) => "empty",
    })
}

fn grouped_signature(by_impl: &BTreeMap<String, String>) -> String {
    let mut by_signature: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (impl_ref, signature) in by_impl {
        by_signature
            .entry(signature.as_str())
            .or_default()
            .push(impl_ref.as_str());
    }

    by_signature
        .into_iter()
        .map(|(signature, mut impls)| {
            impls.sort_unstable();
            format!("{}=>{}", impls.join("+"), signature)
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn partition_label(by_impl: &BTreeMap<String, String>) -> String {
    let mut groups: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (impl_ref, signature) in by_impl {
        groups
            .entry(signature.as_str())
            .or_default()
            .push(impl_ref.as_str());
    }
    if groups.len() == by_impl.len() {
        let impls = by_impl.keys().map(String::as_str).collect::<Vec<_>>();
        if impls.len() == 2 {
            return format!("{}-vs-{}", impls[0], impls[1]);
        }
        return format!("all-distinct-{}", impls.join("+"));
    }
    let largest = groups.values().map(Vec::len).max().unwrap_or(0);
    let outliers = groups
        .values()
        .filter(|impls| impls.len() < largest)
        .flat_map(|impls| impls.iter().copied())
        .collect::<Vec<_>>();
    if largest > 1 && !outliers.is_empty() {
        format!("majority-vs-{}", outliers.join("+"))
    } else {
        format!("{}-groups", groups.len())
    }
}

fn normalize_instruction_profile(instruction: &str) -> String {
    normalize_text_digits(instruction.trim())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

fn normalize_register_name(name: &str) -> String {
    normalize_text_digits(&name.to_ascii_lowercase())
}

fn normalize_text_digits(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_digit = false;
    for ch in input.chars() {
        if ch.is_ascii_digit() {
            if !in_digit {
                out.push('#');
                in_digit = true;
            }
        } else {
            in_digit = false;
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

fn code_classes(codes: &[u64]) -> String {
    codes
        .iter()
        .map(|code| {
            if *code == u64::MAX {
                "unknown".to_string()
            } else if *code < 16 {
                "standard".to_string()
            } else {
                "custom".to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn value_class_u64(value: u64) -> &'static str {
    match value {
        0 => "zero",
        u64::MAX => "all-ones",
        _ if value & 0xffff_ffff_0000_0000 != 0 => "wide",
        _ if value & 0xffff_0000 != 0 => "u32",
        _ if value & 0xff00 != 0 => "u16",
        _ => "u8",
    }
}

fn value_class_u8(value: u8) -> &'static str {
    match value {
        0 => "zero",
        0xff => "all-ones",
        _ if value & 0xf0 != 0 => "hi-nibble",
        _ => "lo-nibble",
    }
}

fn stable_fnv1a_64(input: &str) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET_BASIS;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, fs};

    use super::*;

    fn diff_with_values(spike_value: u64, dut_value: u64) -> WriteDifference {
        WriteDifference {
            test_index: 0,
            global_index: 4,
            register_changes: BTreeMap::from([
                ("spike".to_string(), vec![("x10".to_string(), spike_value)]),
                ("dut".to_string(), vec![("x10".to_string(), dut_value)]),
            ]),
            memory_changes: BTreeMap::from([
                ("spike".to_string(), Vec::new()),
                ("dut".to_string(), Vec::new()),
            ]),
            register_contexts: BTreeMap::new(),
            memory_contexts: BTreeMap::new(),
        }
    }

    #[test]
    fn strict_fingerprint_separates_exact_write_values() {
        let impls = vec!["spike".to_string(), "dut".to_string()];
        let first = build_write_dedup_event_for_instruction(
            &diff_with_values(1, 2),
            &impls,
            "addi x10, x0, 1",
        )
        .expect("first signature");
        let second = build_write_dedup_event_for_instruction(
            &diff_with_values(1, 3),
            &impls,
            "addi x10, x0, 1",
        )
        .expect("second signature");

        assert_eq!(first.auto_tag, second.auto_tag);
        assert!(first.auto_tag.ends_with("/dut-vs-spike"));
        assert_eq!(first.family_fingerprint, second.family_fingerprint);
        assert_ne!(first.strict_fingerprint, second.strict_fingerprint);
    }

    #[test]
    fn store_reports_second_strict_match_as_duplicate() {
        let tmp = std::env::temp_dir().join(format!(
            "riscv_fuzz_dedup_test_{}_{}",
            std::process::id(),
            unix_seconds_now()
        ));
        fs::create_dir_all(&tmp).expect("create temp dir");
        let db_path = tmp.join("dedup.jsonl");
        let settings = DiffDedupSettings {
            enabled: true,
            database_path: Some(db_path),
        };
        let impls = vec!["spike".to_string(), "dut".to_string()];
        let diff = diff_with_values(1, 2);

        let first = record_write_difference_for_instruction_for_test(
            &settings,
            &tmp.join("run_1"),
            &tmp.join("run_1/report.md"),
            &diff,
            &impls,
            "addi x10, x0, 1",
        )
        .expect("first record")
        .expect("enabled");
        let second = record_write_difference_for_instruction_for_test(
            &settings,
            &tmp.join("run_2"),
            &tmp.join("run_2/report.md"),
            &diff,
            &impls,
            "addi x10, x0, 1",
        )
        .expect("second record")
        .expect("enabled");

        assert_eq!(first.status, DiffDedupStatus::FirstSeen);
        assert!(first.database_path.is_absolute());
        assert!(matches!(
            second.status,
            DiffDedupStatus::Duplicate {
                previous_count: 1,
                ..
            }
        ));
        let _ = fs::remove_dir_all(tmp);
    }

    fn record_write_difference_for_instruction_for_test(
        settings: &DiffDedupSettings,
        run_root: &Path,
        report_path: &Path,
        diff: &WriteDifference,
        impl_order: &[String],
        instruction: &str,
    ) -> Result<Option<DiffDedupOutcome>, DiffAnalysisError> {
        if !settings.enabled {
            return Ok(None);
        }
        let event = build_write_dedup_event_for_instruction(diff, impl_order, instruction)?;
        let database_path = resolve_database_path(settings, run_root);
        let _lock = acquire_store_lock(&database_path)?;
        let mut previous_count = 0usize;
        if database_path.is_file() {
            let file = File::open(&database_path)?;
            for line in BufReader::new(file).lines() {
                let record: StoredDedupRecord = serde_json::from_str(&line?)?;
                if record.event_kind == event.kind
                    && record.strict_fingerprint == event.strict_fingerprint
                {
                    previous_count += 1;
                }
            }
        }
        let record = StoredDedupRecord {
            schema_version: SCHEMA_VERSION,
            event_kind: event.kind.to_string(),
            auto_tag: event.auto_tag.clone(),
            family_fingerprint: event.family_fingerprint.clone(),
            strict_fingerprint: event.strict_fingerprint.clone(),
            created_unix_seconds: unix_seconds_now(),
            run_root: run_root.display().to_string(),
            report_path: report_path.display().to_string(),
            test_index: diff.test_index,
            global_index: diff.global_index,
            impl_order: impl_order.to_vec(),
            summary: event.summary.clone(),
            collision_guard: event.collision_guard.to_string(),
        };
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&database_path)?;
        serde_json::to_writer(&mut file, &record)?;
        file.write_all(b"\n")?;
        let status = if previous_count == 0 {
            DiffDedupStatus::FirstSeen
        } else {
            DiffDedupStatus::Duplicate {
                previous_count,
                first_seen_run_root: None,
                first_seen_report_path: None,
            }
        };
        Ok(Some(DiffDedupOutcome {
            event,
            status,
            database_path,
            malformed_records_ignored: 0,
        }))
    }
}
