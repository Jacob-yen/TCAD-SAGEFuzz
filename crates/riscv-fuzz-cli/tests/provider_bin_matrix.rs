use riscv_fuzz_diff::{
    DiffAnalysisError, DiffAnalysisSettings, DiffDedupSettings, run_diff_analysis_with_testcase,
};
use riscv_fuzz_plugin_api::context::ExecutionContextOutput;
use riscv_fuzz_plugin_api::metadata::FloatExtension;
use riscv_fuzz_plugin_api::streaming::{InstructionEvent, StreamingCollector};
use riscv_fuzz_plugin_api::{
    AlignmentMode, CoverageMode, ExecuteRequest, ExecutionOutput, HartExecutionOutput, ISABase,
    IsaExtensions,
};
use riscv_fuzz_plugin_manager::PluginManager;
use riscv_fuzz_testcase_engine::{
    execution_artifacts::{
        ExecutionArtifactOptions, MulticoreSyncMetricFailureMode, execute_testcase_with_artifacts,
    },
    impls::{execute, extensions_for_config, generate_random_testcase_with_extensions},
    testcase::{
        GeneratedTestCase, HartConfig, MulticoreSyncConfig, MulticoreSyncMetrics,
        StructuredTestInstructions, TestCaseConfig,
    },
};
use riscv_fuzz_toolchain::execution_context::{
    ContextBuildError, build_execution_context_output, build_partial_execution_context_output,
    generate_execution_report, generate_execution_timeline_svg_from_output,
};
use riscv_instruction::RiscvInstruction;
use riscv_instruction_types::{
    AsmDecodeWithContext, CsrSelectionConfig, IntegerRegister, IsaDecodeContext, IsaProfile,
    LoadImmediatePurpose, MemoryAccessConfig, PseudoInstruction, RandomConfig,
    RegisterOperandConfig, ValidatedValue, XLen, XLenImmediate,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    env,
    fs::{self, File},
    hash::{Hash, Hasher},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::Instant,
};

const PLUGIN_DIRS_ENV: &str = "RISCV_FUZZ_PLUGIN_DIRS";
const EXEC_TIMEOUT_ENV: &str = "RISCV_FUZZ_EXEC_TIMEOUT_SECS";
const PROVIDER_ROOT_ENV: &str = "CX_RISCV_CORES_ROOT";
const PROVIDER_ARTIFACT_DIR_ENV: &str = "CX_RISCV_CORES_ARTIFACT_DIR";
const PROVIDER_FILTER_ENV: &str = "RISCV_PROVIDER_BIN_FILTER";
const DIFF_HEALTH_RUNS_ENV: &str = "RISCV_PROVIDER_DIFF_HEALTH_RUNS";
const DIFF_HEALTH_CASE_CONCURRENCY_ENV: &str = "RISCV_PROVIDER_DIFF_HEALTH_CASE_CONCURRENCY";
const DIFF_HEALTH_MAX_PARALLEL_ENV: &str = "RISCV_PROVIDER_DIFF_HEALTH_MAX_PARALLEL";
const RUN_ROOT_ENV: &str = "RISCV_PROVIDER_BIN_ARTIFACT_DIR";
const XIANGSHAN_DIFF_1C_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_DIFFTEST_RV64_1C_SO";
const XIANGSHAN_DIFF_2C_ENV: &str = "CX_RISCV_CORES_XIANGSHAN_DIFFTEST_RV64_2C_SO";
const SPIKE_ENV: &str = "CX_RISCV_CORES_SPIKE";

static TEST_LOCK: Mutex<()> = Mutex::new(());
static TEST_PLUGIN_BUNDLE_BUILD: OnceLock<Result<(), String>> = OnceLock::new();

const MULTICORE_SYNC_HEALTH_SAMPLE_COUNT: usize = 2;
const MULTICORE_SYNC_HEALTH_IMPLS: &[&str] = &[
    "boom",
    "cva6",
    "ibex",
    "kronos",
    "picorv32",
    "rocket",
    "vex",
    "xiangshan",
];
const SINGLE_HART_DIFF_HEALTH_IMPLS: &[&str] = &[
    "boom",
    "cva6",
    "ibex",
    "kronos",
    "openc906",
    "openc910",
    "picorv32",
    "rocket",
    "vex",
    "xiangshan",
];
const DUAL_HART_DIFF_HEALTH_IMPLS: &[&str] = &[
    "boom",
    "cva6",
    "ibex",
    "kronos",
    "picorv32",
    "rocket",
    "vex",
    "xiangshan",
];
const DIFF_HEALTH_MAX_REDUCTION_ITERATIONS: usize = 5;

fn isa_profile(isa_base: ISABase) -> IsaProfile {
    IsaProfile::new(
        match isa_base {
            ISABase::Rv32 => XLen::X32,
            ISABase::Rv64 => XLen::X64,
        },
        None,
    )
}

fn isa_profile_for_float(isa_base: ISABase, float_extension: FloatExtension) -> IsaProfile {
    let flen = match float_extension {
        FloatExtension::None => None,
        FloatExtension::F => Some(riscv_instruction_types::FLen::F32),
        FloatExtension::FD => Some(riscv_instruction_types::FLen::F64),
    };
    IsaProfile::new(
        match isa_base {
            ISABase::Rv32 => XLen::X32,
            ISABase::Rv64 => XLen::X64,
        },
        flen,
    )
}

#[derive(Clone, Debug, Serialize)]
struct ProviderBinCase {
    id: String,
    impl_ref: String,
    source_label: String,
    binary_path: PathBuf,
    binary_name: String,
    isa_base: ISABase,
    requested_float_extension: FloatExtension,
    coverage_mode: CoverageMode,
    allow_unaligned: bool,
    hart_count: u32,
}

#[derive(Debug, Clone)]
struct HartLogExceptionSummary {
    instruction_index: usize,
}

#[derive(Debug, Clone)]
struct HartLogView {
    instruction_count: usize,
    observed_slots: Vec<bool>,
    register_changes: Vec<Vec<riscv_fuzz_plugin_api::RegisterValue>>,
    memory_changes: Vec<Vec<riscv_fuzz_plugin_api::MemValue>>,
    exceptions: Vec<HartLogExceptionSummary>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct MinMeanMaxSummary {
    min: f64,
    mean: f64,
    max: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct MulticoreSyncHealthThresholds {
    min_simultaneous_cycle_rate_mean: f64,
    min_register_overlap_cycle_rate_mean: f64,
    min_memory_contention_rate_mean: f64,
    min_cycle_contention_score_mean: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ProviderMulticoreSyncHealthSampleSummary {
    sample_index: usize,
    seed: u64,
    metrics: MulticoreSyncMetrics,
}

#[derive(Debug, Serialize)]
struct ProviderMulticoreSyncHealthCaseSummary {
    status: String,
    id: String,
    impl_ref: String,
    source_label: String,
    binary_path: String,
    binary_name: String,
    isa_base: String,
    requested_float_extension: String,
    effective_float_extension: String,
    coverage_mode: String,
    allow_unaligned: bool,
    hart_count: u32,
    sample_count: usize,
    seed_start: u64,
    seed_end: u64,
    timeout_secs: u64,
    multicore_sync_config: MulticoreSyncConfig,
    thresholds: MulticoreSyncHealthThresholds,
    simultaneous_cycle_rate: MinMeanMaxSummary,
    register_overlap_cycle_rate: MinMeanMaxSummary,
    memory_contention_rate: MinMeanMaxSummary,
    cycle_contention_score: MinMeanMaxSummary,
    same_address_overlap_observed: bool,
    samples: Vec<ProviderMulticoreSyncHealthSampleSummary>,
    failed_checks: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProviderMulticoreSyncHealthMatrixSummary {
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    average_simultaneous_cycle_rate: f64,
    average_memory_contention_rate: f64,
    average_cycle_contention_score: f64,
    average_register_overlap_cycle_rate: f64,
    cases_with_same_address_overlap: usize,
    case_ids: Vec<String>,
    failures: Vec<String>,
}

struct EnvVarGuard {
    key: String,
    prev: Option<String>,
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => unsafe { env::set_var(&self.key, v) },
            None => unsafe { env::remove_var(&self.key) },
        }
    }
}

fn set_env_var(key: impl Into<String>, value: impl AsRef<str>) -> EnvVarGuard {
    let key = key.into();
    let prev = env::var(&key).ok();
    unsafe { env::set_var(&key, value.as_ref()) };
    EnvVarGuard { key, prev }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn ensure_test_plugin_bundle() {
    TEST_PLUGIN_BUNDLE_BUILD
        .get_or_init(|| {
            let root = workspace_root();
            let script = root.join("scripts").join("build_plugin_bundle.sh");
            if !script.is_file() {
                return Err(format!(
                    "plugin bundle build script does not exist: {}",
                    script.display()
                ));
            }

            let output = Command::new("bash")
                .arg(&script)
                .current_dir(&root)
                .output()
                .map_err(|err| {
                    format!(
                        "failed to launch plugin bundle build script {}: {err}",
                        script.display()
                    )
                })?;

            if output.status.success() {
                return Ok(());
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!(
                "plugin bundle build script {} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
                script.display(),
                output.status.code(),
                stdout.trim(),
                stderr.trim(),
            ))
        })
        .as_ref()
        .unwrap_or_else(|err| panic!("failed to refresh dynamic plugin bundle: {err}"));
}

fn configure_plugin_dirs() -> EnvVarGuard {
    ensure_test_plugin_bundle();
    let root = workspace_root();
    let joined = std::env::join_paths([
        root.join("target/debug"),
        root.join("target/release"),
        root.join("riscv_fuzz_plugins"),
    ])
    .expect("join plugin dirs");
    set_env_var(PLUGIN_DIRS_ENV, joined.to_string_lossy())
}

fn provider_root() -> PathBuf {
    if let Ok(root) = env::var(PROVIDER_ROOT_ENV) {
        let path = PathBuf::from(root.trim());
        if path.is_dir() {
            return path;
        }
    }

    let fallback = workspace_root()
        .parent()
        .expect("workspace parent for provider root")
        .join("cx-riscv-cores");
    assert!(
        fallback.is_dir(),
        "provider root does not exist: {}",
        fallback.display()
    );
    fallback
}

fn local_source_provider_root() -> Option<PathBuf> {
    let candidate = workspace_root()
        .parent()
        .expect("workspace parent for local provider root")
        .join("cx-riscv-cores");
    candidate.is_dir().then_some(candidate)
}

fn provider_artifact_dir(provider_root: &Path) -> PathBuf {
    if let Ok(dir) = env::var(PROVIDER_ARTIFACT_DIR_ENV) {
        let path = PathBuf::from(dir.trim());
        if path.is_dir() {
            return path;
        }
    }

    let fallback = provider_root.join("artifacts");
    assert!(
        fallback.is_dir(),
        "provider artifact dir does not exist: {}",
        fallback.display()
    );
    fallback
}

fn run_root_for() -> PathBuf {
    if let Ok(root) = env::var(RUN_ROOT_ENV) {
        let trimmed = root.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed);
            fs::create_dir_all(&path)
                .unwrap_or_else(|e| panic!("failed to create run root {}: {e}", path.display()));
            return path;
        }
    }

    let path = env::current_dir()
        .expect("current dir")
        .join("temp")
        .join("provider_bin_matrix");
    fs::create_dir_all(&path)
        .unwrap_or_else(|e| panic!("failed to create run root {}: {e}", path.display()));
    path
}

fn normalize_case_filter() -> Option<String> {
    let raw = env::var(PROVIDER_FILTER_ENV).ok()?;
    let trimmed = raw.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn resolve_spike_path() -> PathBuf {
    if let Ok(path) = env::var(SPIKE_ENV) {
        let candidate = PathBuf::from(path.trim());
        if candidate.is_file() {
            return candidate;
        }
    }

    let fallback = PathBuf::from("/opt/riscv/bin/spike");
    assert!(
        fallback.is_file(),
        "Spike binary not found; set {SPIKE_ENV} or install /opt/riscv/bin/spike"
    );
    fallback
}

fn ensure_aux_provider_envs(
    provider_root: &Path,
    artifact_dir: &Path,
    spike_path: &Path,
) -> Vec<EnvVarGuard> {
    let mut guards = Vec::new();
    guards.push(set_env_var(
        PROVIDER_ROOT_ENV,
        provider_root.to_string_lossy(),
    ));
    guards.push(set_env_var(
        PROVIDER_ARTIFACT_DIR_ENV,
        artifact_dir.to_string_lossy(),
    ));
    guards.push(set_env_var(SPIKE_ENV, spike_path.to_string_lossy()));

    let xiangshan_diff_1c = artifact_dir.join("xiangshan_difftest_rv64_1c_so");
    let xiangshan_diff_2c = artifact_dir.join("xiangshan_difftest_rv64_2c_so");
    assert!(
        xiangshan_diff_1c.is_file(),
        "missing XiangShan DiffTest library: {}",
        xiangshan_diff_1c.display()
    );
    assert!(
        xiangshan_diff_2c.is_file(),
        "missing XiangShan DiffTest library: {}",
        xiangshan_diff_2c.display()
    );
    guards.push(set_env_var(
        XIANGSHAN_DIFF_1C_ENV,
        xiangshan_diff_1c.to_string_lossy(),
    ));
    guards.push(set_env_var(
        XIANGSHAN_DIFF_2C_ENV,
        xiangshan_diff_2c.to_string_lossy(),
    ));

    guards
}

fn cx_artifact_env_name(name: &str) -> String {
    format!(
        "CX_RISCV_CORES_{}",
        name.replace(['-', '.'], "_").to_ascii_uppercase()
    )
}

fn configure_provider_binary_envs(provider_root: &Path, artifact_dir: &Path) -> Vec<EnvVarGuard> {
    let mut env_paths: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut ingest_dir = |dir: &Path| {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.ends_with(".json") {
                continue;
            }
            env_paths.entry(cx_artifact_env_name(name)).or_insert(path);
        }
    };

    ingest_dir(artifact_dir);
    if let Some(local_root) = local_source_provider_root()
        && local_root != provider_root
    {
        ingest_dir(&local_root.join("artifacts"));
    }

    env_paths
        .into_iter()
        .map(|(key, path)| set_env_var(key, path.to_string_lossy()))
        .collect()
}

fn collect_provider_bins(provider_root: &Path, artifact_dir: &Path) -> Vec<ProviderBinCase> {
    let mut chosen: BTreeMap<String, (String, PathBuf)> = BTreeMap::new();

    let mut ingest_dir = |dir: &Path, source_label: &str, prefer: bool| {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name == "spike" || name.ends_with("_so") {
                continue;
            }
            if prefer || !chosen.contains_key(name) {
                chosen.insert(name.to_string(), (source_label.to_string(), path));
            }
        }
    };

    ingest_dir(artifact_dir, "artifacts", false);

    let mut build_roots = vec![provider_root.to_path_buf()];
    if let Some(local_root) = local_source_provider_root()
        && local_root != provider_root
    {
        build_roots.push(local_root);
    }

    for build_root in build_roots {
        let cores_dir = build_root.join("cores");
        let Ok(entries) = fs::read_dir(&cores_dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let build_result = entry.path().join("build_result");
            let Some(core_name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
                continue;
            };
            if build_result.is_dir() {
                ingest_dir(&build_result, &format!("build_result:{core_name}"), true);
            }
        }
    }

    let filter = normalize_case_filter();
    let mut out = Vec::new();
    for (name, (source_label, path)) in chosen {
        let Some(case) = parse_provider_bin_case(&name, path, source_label) else {
            continue;
        };
        if let Some(filter) = &filter
            && !case.id.to_ascii_lowercase().contains(filter)
        {
            continue;
        }
        out.push(case);
    }

    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

fn parse_provider_bin_case(
    binary_name: &str,
    binary_path: PathBuf,
    source_label: String,
) -> Option<ProviderBinCase> {
    let impl_ref = if binary_name.starts_with("cva6_") {
        "cva6"
    } else if binary_name.starts_with("boom_") {
        "boom"
    } else if binary_name.starts_with("ibex_") {
        "ibex"
    } else if binary_name.starts_with("kronos_") {
        "kronos"
    } else if binary_name.starts_with("openc906_") {
        "openc906"
    } else if binary_name.starts_with("openc910_") {
        "openc910"
    } else if binary_name.starts_with("picorv32_") {
        "picorv32"
    } else if binary_name.starts_with("rocket-chip_") {
        "rocket"
    } else if binary_name.starts_with("vexriscv_") {
        "vex"
    } else if binary_name.starts_with("xiangshan_") {
        "xiangshan"
    } else {
        return None;
    };

    let isa_base = if binary_name.contains("_rv32") {
        ISABase::Rv32
    } else if binary_name.contains("_rv64") {
        ISABase::Rv64
    } else {
        return None;
    };

    let requested_float_extension =
        if binary_name.contains("rv32fd") || binary_name.contains("rv64fd") {
            FloatExtension::FD
        } else if binary_name.contains("rv32f") || binary_name.contains("rv64f") {
            FloatExtension::F
        } else {
            FloatExtension::None
        };

    let coverage_mode = if binary_name.contains("_cov_light") {
        CoverageMode::Light
    } else if binary_name.contains("_cov") {
        CoverageMode::Full
    } else {
        CoverageMode::Off
    };

    let hart_count = if binary_name.contains("_2c") { 2 } else { 1 };
    let allow_unaligned = binary_name.contains("_unaligned_");

    Some(ProviderBinCase {
        id: binary_name.to_string(),
        impl_ref: impl_ref.to_string(),
        source_label,
        binary_path,
        binary_name: binary_name.to_string(),
        isa_base,
        requested_float_extension,
        coverage_mode,
        allow_unaligned,
        hart_count,
    })
}

fn effective_float_extension(case: &ProviderBinCase) -> FloatExtension {
    let impls = metadata_impls(case);
    let plugin_manager = PluginManager::global();
    clamp_float_extension(
        case.requested_float_extension,
        max_float_extension_for_extensions(
            &plugin_manager.extensions_for_impls(&impls, case.isa_base),
        ),
    )
}

fn max_float_extension_for_extensions(extensions: &IsaExtensions) -> FloatExtension {
    match extensions {
        IsaExtensions::Rv32(values) => {
            let has_d = values.contains(&riscv_instruction::RV32Extensions::D);
            let has_f = values.contains(&riscv_instruction::RV32Extensions::F);
            if has_d {
                FloatExtension::FD
            } else if has_f {
                FloatExtension::F
            } else {
                FloatExtension::None
            }
        }
        IsaExtensions::Rv64(values) => {
            let has_d = values.contains(&riscv_instruction::RV64Extensions::D);
            let has_f = values.contains(&riscv_instruction::RV64Extensions::F);
            if has_d {
                FloatExtension::FD
            } else if has_f {
                FloatExtension::F
            } else {
                FloatExtension::None
            }
        }
    }
}

fn clamp_float_extension(
    requested: FloatExtension,
    supported_max: FloatExtension,
) -> FloatExtension {
    match (requested, supported_max) {
        (_, FloatExtension::None) => FloatExtension::None,
        (FloatExtension::FD, FloatExtension::F) => FloatExtension::F,
        (requested, _) => requested,
    }
}

fn metadata_impls(case: &ProviderBinCase) -> Vec<String> {
    vec![case.impl_ref.clone()]
}

fn target_binary_env_var(case: &ProviderBinCase) -> &'static str {
    match case.impl_ref.as_str() {
        "cva6" => match (
            case.isa_base,
            case.hart_count,
            case.coverage_mode,
            effective_float_extension(case),
        ) {
            (ISABase::Rv32, 1, CoverageMode::Off, FloatExtension::None) => {
                "CX_RISCV_CORES_CVA6_RV32_1C"
            }
            (ISABase::Rv32, 1, CoverageMode::Full, FloatExtension::None) => {
                "CX_RISCV_CORES_CVA6_RV32_1C_COV"
            }
            (ISABase::Rv32, 1, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_CVA6_RV32_1C_COV_LIGHT"
            }
            (ISABase::Rv32, 1, CoverageMode::Off, FloatExtension::F) => {
                "CX_RISCV_CORES_CVA6_RV32F_1C"
            }
            (ISABase::Rv32, 1, CoverageMode::Full, FloatExtension::F) => {
                "CX_RISCV_CORES_CVA6_RV32F_1C_COV"
            }
            (ISABase::Rv32, 1, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_CVA6_RV32F_1C_COV_LIGHT"
            }
            (ISABase::Rv64, 1, CoverageMode::Off, _) => "CX_RISCV_CORES_CVA6_RV64_1C",
            (ISABase::Rv64, 1, CoverageMode::Full, _) => "CX_RISCV_CORES_CVA6_RV64_1C_COV",
            (ISABase::Rv64, 1, CoverageMode::Light, _) => "CX_RISCV_CORES_CVA6_RV64_1C_COV_LIGHT",
            (ISABase::Rv32, 2, CoverageMode::Off, FloatExtension::None) => {
                "CX_RISCV_CORES_CVA6_RV32_2C"
            }
            (ISABase::Rv32, 2, CoverageMode::Full, FloatExtension::None) => {
                "CX_RISCV_CORES_CVA6_RV32_2C_COV"
            }
            (ISABase::Rv32, 2, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_CVA6_RV32_2C_COV_LIGHT"
            }
            (ISABase::Rv32, 2, CoverageMode::Off, FloatExtension::F) => {
                "CX_RISCV_CORES_CVA6_RV32F_2C"
            }
            (ISABase::Rv32, 2, CoverageMode::Full, FloatExtension::F) => {
                "CX_RISCV_CORES_CVA6_RV32F_2C_COV"
            }
            (ISABase::Rv32, 2, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_CVA6_RV32F_2C_COV_LIGHT"
            }
            (ISABase::Rv64, 2, CoverageMode::Off, _) => "CX_RISCV_CORES_CVA6_RV64_2C",
            (ISABase::Rv64, 2, CoverageMode::Full, _) => "CX_RISCV_CORES_CVA6_RV64_2C_COV",
            (ISABase::Rv64, 2, CoverageMode::Light, _) => "CX_RISCV_CORES_CVA6_RV64_2C_COV_LIGHT",
            _ => panic!("unsupported CVA6 provider case: {}", case.id),
        },
        "boom" => match (case.isa_base, case.hart_count, case.coverage_mode) {
            (ISABase::Rv64, 1, CoverageMode::Off) => "CX_RISCV_CORES_BOOM_RV64FD_1C",
            (ISABase::Rv64, 1, CoverageMode::Full) => "CX_RISCV_CORES_BOOM_RV64FD_1C_COV",
            (ISABase::Rv64, 1, CoverageMode::Light) => "CX_RISCV_CORES_BOOM_RV64FD_1C_COV_LIGHT",
            (ISABase::Rv64, 2, CoverageMode::Off) => "CX_RISCV_CORES_BOOM_RV64FD_2C",
            (ISABase::Rv64, 2, CoverageMode::Full) => "CX_RISCV_CORES_BOOM_RV64FD_2C_COV",
            (ISABase::Rv64, 2, CoverageMode::Light) => "CX_RISCV_CORES_BOOM_RV64FD_2C_COV_LIGHT",
            _ => panic!("unsupported BOOM provider case: {}", case.id),
        },
        "ibex" => match (case.hart_count, case.coverage_mode) {
            (1, CoverageMode::Off) => "CX_RISCV_CORES_IBEX_RV32IMC_1C",
            (1, CoverageMode::Full) => "CX_RISCV_CORES_IBEX_RV32IMC_1C_COV",
            (1, CoverageMode::Light) => "CX_RISCV_CORES_IBEX_RV32IMC_1C_COV_LIGHT",
            (2, CoverageMode::Off) => "CX_RISCV_CORES_IBEX_RV32IMC_2C",
            (2, CoverageMode::Full) => "CX_RISCV_CORES_IBEX_RV32IMC_2C_COV",
            (2, CoverageMode::Light) => "CX_RISCV_CORES_IBEX_RV32IMC_2C_COV_LIGHT",
            _ => panic!("unsupported Ibex provider case: {}", case.id),
        },
        "kronos" => match (case.hart_count, case.coverage_mode) {
            (1, CoverageMode::Off) => "CX_RISCV_CORES_KRONOS_RV32_1C",
            (1, CoverageMode::Full) => "CX_RISCV_CORES_KRONOS_RV32_1C_COV",
            (1, CoverageMode::Light) => "CX_RISCV_CORES_KRONOS_RV32_1C_COV_LIGHT",
            (2, CoverageMode::Off) => "CX_RISCV_CORES_KRONOS_RV32_2C",
            (2, CoverageMode::Full) => "CX_RISCV_CORES_KRONOS_RV32_2C_COV",
            (2, CoverageMode::Light) => "CX_RISCV_CORES_KRONOS_RV32_2C_COV_LIGHT",
            _ => panic!("unsupported Kronos provider case: {}", case.id),
        },
        "openc906" => match (
            case.hart_count,
            case.coverage_mode,
            effective_float_extension(case),
        ) {
            (1, CoverageMode::Off, FloatExtension::None) => "CX_RISCV_CORES_OPENC906_RV64_1C",
            (1, CoverageMode::Full, FloatExtension::None) => "CX_RISCV_CORES_OPENC906_RV64_1C_COV",
            (1, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_OPENC906_RV64_1C_COV_LIGHT"
            }
            (1, CoverageMode::Off, FloatExtension::F) => "CX_RISCV_CORES_OPENC906_RV64F_1C",
            (1, CoverageMode::Full, FloatExtension::F) => "CX_RISCV_CORES_OPENC906_RV64F_1C_COV",
            (1, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_OPENC906_RV64F_1C_COV_LIGHT"
            }
            (1, CoverageMode::Off, FloatExtension::FD) => "CX_RISCV_CORES_OPENC906_RV64FD_1C",
            (1, CoverageMode::Full, FloatExtension::FD) => "CX_RISCV_CORES_OPENC906_RV64FD_1C_COV",
            (1, CoverageMode::Light, FloatExtension::FD) => {
                "CX_RISCV_CORES_OPENC906_RV64FD_1C_COV_LIGHT"
            }
            _ => panic!("unsupported OpenC906 provider case: {}", case.id),
        },
        "openc910" => match (
            case.hart_count,
            case.coverage_mode,
            effective_float_extension(case),
        ) {
            (1, CoverageMode::Off, FloatExtension::None) => "CX_RISCV_CORES_OPENC910_RV64_1C",
            (1, CoverageMode::Full, FloatExtension::None) => "CX_RISCV_CORES_OPENC910_RV64_1C_COV",
            (1, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_OPENC910_RV64_1C_COV_LIGHT"
            }
            (1, CoverageMode::Off, FloatExtension::F) => "CX_RISCV_CORES_OPENC910_RV64F_1C",
            (1, CoverageMode::Full, FloatExtension::F) => "CX_RISCV_CORES_OPENC910_RV64F_1C_COV",
            (1, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_OPENC910_RV64F_1C_COV_LIGHT"
            }
            (1, CoverageMode::Off, FloatExtension::FD) => "CX_RISCV_CORES_OPENC910_RV64FD_1C",
            (1, CoverageMode::Full, FloatExtension::FD) => "CX_RISCV_CORES_OPENC910_RV64FD_1C_COV",
            (1, CoverageMode::Light, FloatExtension::FD) => {
                "CX_RISCV_CORES_OPENC910_RV64FD_1C_COV_LIGHT"
            }
            (2, CoverageMode::Off, FloatExtension::None) => "CX_RISCV_CORES_OPENC910_RV64_2C",
            (2, CoverageMode::Full, FloatExtension::None) => "CX_RISCV_CORES_OPENC910_RV64_2C_COV",
            (2, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_OPENC910_RV64_2C_COV_LIGHT"
            }
            (2, CoverageMode::Off, FloatExtension::F) => "CX_RISCV_CORES_OPENC910_RV64F_2C",
            (2, CoverageMode::Full, FloatExtension::F) => "CX_RISCV_CORES_OPENC910_RV64F_2C_COV",
            (2, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_OPENC910_RV64F_2C_COV_LIGHT"
            }
            (2, CoverageMode::Off, FloatExtension::FD) => "CX_RISCV_CORES_OPENC910_RV64FD_2C",
            (2, CoverageMode::Full, FloatExtension::FD) => "CX_RISCV_CORES_OPENC910_RV64FD_2C_COV",
            (2, CoverageMode::Light, FloatExtension::FD) => {
                "CX_RISCV_CORES_OPENC910_RV64FD_2C_COV_LIGHT"
            }
            _ => panic!("unsupported OpenC910 provider case: {}", case.id),
        },
        "picorv32" => match (case.hart_count, case.coverage_mode) {
            (1, CoverageMode::Off) => "CX_RISCV_CORES_PICORV32_RV32_1C",
            (1, CoverageMode::Full) => "CX_RISCV_CORES_PICORV32_RV32_1C_COV",
            (1, CoverageMode::Light) => "CX_RISCV_CORES_PICORV32_RV32_1C_COV_LIGHT",
            (2, CoverageMode::Off) => "CX_RISCV_CORES_PICORV32_RV32_2C",
            (2, CoverageMode::Full) => "CX_RISCV_CORES_PICORV32_RV32_2C_COV",
            (2, CoverageMode::Light) => "CX_RISCV_CORES_PICORV32_RV32_2C_COV_LIGHT",
            _ => panic!("unsupported PicoRV32 provider case: {}", case.id),
        },
        "rocket" => match (
            case.isa_base,
            case.hart_count,
            case.coverage_mode,
            effective_float_extension(case),
        ) {
            (ISABase::Rv32, 1, CoverageMode::Off, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32_1C"
            }
            (ISABase::Rv32, 1, CoverageMode::Full, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32_1C_COV"
            }
            (ISABase::Rv32, 1, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32_1C_COV_LIGHT"
            }
            (ISABase::Rv32, 2, CoverageMode::Off, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32_2C"
            }
            (ISABase::Rv32, 2, CoverageMode::Full, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32_2C_COV"
            }
            (ISABase::Rv32, 2, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32_2C_COV_LIGHT"
            }
            (ISABase::Rv32, 1, CoverageMode::Off, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32F_1C"
            }
            (ISABase::Rv32, 1, CoverageMode::Full, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32F_1C_COV"
            }
            (ISABase::Rv32, 1, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32F_1C_COV_LIGHT"
            }
            (ISABase::Rv32, 2, CoverageMode::Off, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32F_2C"
            }
            (ISABase::Rv32, 2, CoverageMode::Full, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32F_2C_COV"
            }
            (ISABase::Rv32, 2, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32F_2C_COV_LIGHT"
            }
            (ISABase::Rv32, 1, CoverageMode::Off, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_1C"
            }
            (ISABase::Rv32, 1, CoverageMode::Full, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_1C_COV"
            }
            (ISABase::Rv32, 1, CoverageMode::Light, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_1C_COV_LIGHT"
            }
            (ISABase::Rv32, 2, CoverageMode::Off, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_2C"
            }
            (ISABase::Rv32, 2, CoverageMode::Full, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_2C_COV"
            }
            (ISABase::Rv32, 2, CoverageMode::Light, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_2C_COV_LIGHT"
            }
            (ISABase::Rv64, 1, CoverageMode::Off, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64_1C"
            }
            (ISABase::Rv64, 1, CoverageMode::Full, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64_1C_COV"
            }
            (ISABase::Rv64, 1, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64_1C_COV_LIGHT"
            }
            (ISABase::Rv64, 2, CoverageMode::Off, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64_2C"
            }
            (ISABase::Rv64, 2, CoverageMode::Full, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64_2C_COV"
            }
            (ISABase::Rv64, 2, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64_2C_COV_LIGHT"
            }
            (ISABase::Rv64, 1, CoverageMode::Off, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64F_1C"
            }
            (ISABase::Rv64, 1, CoverageMode::Full, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64F_1C_COV"
            }
            (ISABase::Rv64, 1, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64F_1C_COV_LIGHT"
            }
            (ISABase::Rv64, 2, CoverageMode::Off, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64F_2C"
            }
            (ISABase::Rv64, 2, CoverageMode::Full, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64F_2C_COV"
            }
            (ISABase::Rv64, 2, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64F_2C_COV_LIGHT"
            }
            (ISABase::Rv64, 1, CoverageMode::Off, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_1C"
            }
            (ISABase::Rv64, 1, CoverageMode::Full, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_1C_COV"
            }
            (ISABase::Rv64, 1, CoverageMode::Light, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_1C_COV_LIGHT"
            }
            (ISABase::Rv64, 2, CoverageMode::Off, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_2C"
            }
            (ISABase::Rv64, 2, CoverageMode::Full, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_2C_COV"
            }
            (ISABase::Rv64, 2, CoverageMode::Light, FloatExtension::FD) => {
                "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_2C_COV_LIGHT"
            }
            _ => panic!("unsupported Rocket provider case: {}", case.id),
        },
        "vex" => match (
            case.hart_count,
            case.coverage_mode,
            effective_float_extension(case),
        ) {
            (1, CoverageMode::Off, FloatExtension::None) => "CX_RISCV_CORES_VEXRISCV_RV32_1C",
            (1, CoverageMode::Full, FloatExtension::None) => "CX_RISCV_CORES_VEXRISCV_RV32_1C_COV",
            (1, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_VEXRISCV_RV32_1C_COV_LIGHT"
            }
            (2, CoverageMode::Off, FloatExtension::None) => "CX_RISCV_CORES_VEXRISCV_RV32_2C",
            (2, CoverageMode::Full, FloatExtension::None) => "CX_RISCV_CORES_VEXRISCV_RV32_2C_COV",
            (2, CoverageMode::Light, FloatExtension::None) => {
                "CX_RISCV_CORES_VEXRISCV_RV32_2C_COV_LIGHT"
            }
            (1, CoverageMode::Off, FloatExtension::F) => "CX_RISCV_CORES_VEXRISCV_RV32F_1C",
            (1, CoverageMode::Full, FloatExtension::F) => "CX_RISCV_CORES_VEXRISCV_RV32F_1C_COV",
            (1, CoverageMode::Light, FloatExtension::F) => {
                "CX_RISCV_CORES_VEXRISCV_RV32F_1C_COV_LIGHT"
            }
            (1, CoverageMode::Off, FloatExtension::FD) => "CX_RISCV_CORES_VEXRISCV_RV32FD_1C",
            (1, CoverageMode::Full, FloatExtension::FD) => "CX_RISCV_CORES_VEXRISCV_RV32FD_1C_COV",
            (1, CoverageMode::Light, FloatExtension::FD) => {
                "CX_RISCV_CORES_VEXRISCV_RV32FD_1C_COV_LIGHT"
            }
            (2, CoverageMode::Off, FloatExtension::FD) => "CX_RISCV_CORES_VEXRISCV_RV32FD_2C",
            (2, CoverageMode::Full, FloatExtension::FD) => "CX_RISCV_CORES_VEXRISCV_RV32FD_2C_COV",
            (2, CoverageMode::Light, FloatExtension::FD) => {
                "CX_RISCV_CORES_VEXRISCV_RV32FD_2C_COV_LIGHT"
            }
            _ => panic!("unsupported Vex provider case: {}", case.id),
        },
        "xiangshan" => match (case.hart_count, case.coverage_mode, case.allow_unaligned) {
            (1, CoverageMode::Off, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C",
            (1, CoverageMode::Full, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C_COV",
            (1, CoverageMode::Light, true) => {
                "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C_COV_LIGHT"
            }
            (1, CoverageMode::Off, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C",
            (1, CoverageMode::Full, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C_COV",
            (1, CoverageMode::Light, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C_COV_LIGHT",
            (2, CoverageMode::Off, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C",
            (2, CoverageMode::Full, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C_COV",
            (2, CoverageMode::Light, true) => {
                "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C_COV_LIGHT"
            }
            (2, CoverageMode::Off, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C",
            (2, CoverageMode::Full, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C_COV",
            (2, CoverageMode::Light, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C_COV_LIGHT",
            _ => panic!("unsupported XiangShan provider case: {}", case.id),
        },
        other => panic!("unsupported provider implementation: {other}"),
    }
}

fn default_timeout_secs(case: &ProviderBinCase) -> u64 {
    match (case.impl_ref.as_str(), case.hart_count, case.isa_base) {
        // XiangShan dual-hart providers complete reliably on the full-size sync-health
        // testcase, but their wall-clock runtime is substantially longer than the rest of
        // the provider matrix. Keep the testcase size intact and relax the watchdog instead.
        ("xiangshan", 2, _) => 1800,
        ("xiangshan", 1, _) => 180,
        (_, 2, ISABase::Rv64) => 180,
        (_, 2, ISABase::Rv32) => 120,
        ("rocket", 1, _) | ("cva6", 1, _) => 120,
        _ => 90,
    }
}

fn case_seed(case: &ProviderBinCase, attempt: u64) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    case.id.hash(&mut hasher);
    attempt.hash(&mut hasher);
    hasher.finish()
}

fn build_block_from_rendered_parts(
    lines: Vec<String>,
    _mem_offsets: Vec<Option<i64>>,
    isa_base: ISABase,
) -> StructuredTestInstructions {
    build_block_from_per_hart_lines(vec![(0, lines)], _mem_offsets, isa_base)
}

fn build_block_from_per_hart_lines(
    per_hart: Vec<(usize, Vec<String>)>,
    _mem_offsets: Vec<Option<i64>>,
    isa_base: ISABase,
) -> StructuredTestInstructions {
    let ctx = match isa_base {
        ISABase::Rv32 => IsaDecodeContext::RV32,
        ISABase::Rv64 => IsaDecodeContext::RV64,
    };
    let mut block = StructuredTestInstructions::default();
    for (hart_idx, lines) in per_hart {
        for line in lines {
            let is_shared_base_address = line.trim_start().starts_with("li x9,");
            let instruction = if is_shared_base_address {
                RiscvInstruction::from(PseudoInstruction::LoadImmediate {
                    rd: IntegerRegister::new(9).expect("valid register"),
                    immediate: XLenImmediate::from_i64_for(isa_profile(isa_base), 0)
                        .expect("base-address immediate must fit active xlen"),
                    purpose: LoadImmediatePurpose::BaseAddress,
                })
            } else {
                RiscvInstruction::decode_asm_with_context(&line, ctx).unwrap_or_else(|err| {
                    panic!(
                        "failed to parse rendered instruction `{line}` into RiscvInstruction: {err}"
                    )
                })
            };
            let instruction_idx = block.harts.get(hart_idx).map_or(0, Vec::len);
            block.push_instruction_to_hart(hart_idx, instruction);
            if is_shared_base_address {
                block.mark_shared_base_address_index(hart_idx, instruction_idx);
            }
        }
    }
    block
}

fn write_json_pretty(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let file =
        File::create(path).map_err(|e| format!("failed to create {}: {e}", path.display()))?;
    serde_json::to_writer_pretty(file, value)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))
}

fn write_text(path: &Path, text: &str) -> Result<(), String> {
    let mut file =
        File::create(path).map_err(|e| format!("failed to create {}: {e}", path.display()))?;
    file.write_all(text.as_bytes())
        .map_err(|e| format!("failed to write {}: {e}", path.display()))
}

fn write_generated_testcase(path: &Path, testcase: &GeneratedTestCase) -> Result<(), String> {
    write_json_pretty(path, testcase)
}

fn ratio_or_one(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        1.0
    } else {
        values.iter().sum::<f64>() / (values.len() as f64)
    }
}

fn min_mean_max_summary(values: &[f64]) -> MinMeanMaxSummary {
    if values.is_empty() {
        return MinMeanMaxSummary {
            min: 0.0,
            mean: 0.0,
            max: 0.0,
        };
    }

    let mut min = values[0];
    let mut max = values[0];
    let mut sum = 0.0;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += *value;
    }

    MinMeanMaxSummary {
        min,
        mean: sum / (values.len() as f64),
        max,
    }
}

fn fmt_percent(value: f64) -> String {
    format!("{:.2}%", value * 100.0)
}

fn aggressive_multicore_sync_config() -> MulticoreSyncConfig {
    MulticoreSyncConfig {
        interaction_slot_ratio: 0.85,
        shared_address_ratio: 1.0,
        prefer_focused_sequences: true,
        ..MulticoreSyncConfig::default()
    }
}

fn multicore_sync_health_thresholds(case: &ProviderBinCase) -> MulticoreSyncHealthThresholds {
    match case.impl_ref.as_str() {
        "cva6" => MulticoreSyncHealthThresholds {
            min_simultaneous_cycle_rate_mean: 0.10,
            min_register_overlap_cycle_rate_mean: 0.90,
            min_memory_contention_rate_mean: 0.95,
            min_cycle_contention_score_mean: 0.75,
        },
        "ibex" => MulticoreSyncHealthThresholds {
            min_simultaneous_cycle_rate_mean: 0.50,
            min_register_overlap_cycle_rate_mean: 0.80,
            min_memory_contention_rate_mean: 0.95,
            min_cycle_contention_score_mean: 0.70,
        },
        "kronos" => MulticoreSyncHealthThresholds {
            min_simultaneous_cycle_rate_mean: 0.80,
            min_register_overlap_cycle_rate_mean: 0.75,
            min_memory_contention_rate_mean: 0.95,
            min_cycle_contention_score_mean: 0.75,
        },
        "rocket" => MulticoreSyncHealthThresholds {
            min_simultaneous_cycle_rate_mean: 0.02,
            min_register_overlap_cycle_rate_mean: 0.05,
            min_memory_contention_rate_mean: 0.0,
            min_cycle_contention_score_mean: 0.02,
        },
        "vex" => MulticoreSyncHealthThresholds {
            min_simultaneous_cycle_rate_mean: 0.08,
            min_register_overlap_cycle_rate_mean: 0.05,
            min_memory_contention_rate_mean: 0.0,
            min_cycle_contention_score_mean: 0.02,
        },
        "picorv32" => MulticoreSyncHealthThresholds {
            min_simultaneous_cycle_rate_mean: 0.005,
            min_register_overlap_cycle_rate_mean: 0.05,
            min_memory_contention_rate_mean: 0.20,
            min_cycle_contention_score_mean: 0.02,
        },
        "xiangshan" => MulticoreSyncHealthThresholds {
            min_simultaneous_cycle_rate_mean: 0.005,
            min_register_overlap_cycle_rate_mean: 0.05,
            min_memory_contention_rate_mean: 0.75,
            min_cycle_contention_score_mean: 0.02,
        },
        _ => MulticoreSyncHealthThresholds {
            min_simultaneous_cycle_rate_mean: 0.005,
            min_register_overlap_cycle_rate_mean: 0.05,
            min_memory_contention_rate_mean: 0.0,
            min_cycle_contention_score_mean: 0.02,
        },
    }
}

fn multicore_sync_health_extension_inst_count(case: &ProviderBinCase) -> usize {
    match case.impl_ref.as_str() {
        "ibex" => 96,
        "kronos" => 96,
        "picorv32" => 48,
        "rocket" => 32,
        "vex" => 48,
        _ => 80,
    }
}

fn generate_target_constrained_testcase(
    case: &ProviderBinCase,
    config: TestCaseConfig,
) -> Result<GeneratedTestCase, String> {
    let target_impls = metadata_impls(case);
    let target_extensions = extensions_for_config(&target_impls, &config)
        .map_err(|err| format!("extension resolution failed for {}: {err}", case.id))?;
    generate_random_testcase_with_extensions(&target_impls, config, target_extensions)
        .map_err(|err| format!("testcase generation failed for {}: {err}", case.id))
}

fn mnemonic_of(line: &str) -> &str {
    line.split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_start_matches('.')
}

fn line_has_memory_candidate(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.starts_with('#') || trimmed.ends_with(':') {
        return false;
    }

    let mnemonic = mnemonic_of(trimmed);
    matches!(
        mnemonic,
        "lb" | "lbu"
            | "lh"
            | "lhu"
            | "lw"
            | "lwu"
            | "ld"
            | "flw"
            | "fld"
            | "sb"
            | "sh"
            | "sw"
            | "sd"
            | "fsw"
            | "fsd"
            | "c.lw"
            | "c.ld"
            | "c.flw"
            | "c.fld"
            | "c.sw"
            | "c.sd"
            | "c.fsw"
            | "c.fsd"
            | "cbo.zero"
    ) || mnemonic.starts_with("amo")
        || mnemonic.starts_with("lr.")
        || mnemonic.starts_with("sc.")
}

fn block_has_per_hart_memory_candidates(block: &StructuredTestInstructions) -> bool {
    let mut hart_candidates = BTreeMap::new();
    for (hart_id, insts) in block.rendered_lines_per_hart() {
        let has_memory = insts.iter().any(|line| line_has_memory_candidate(line));
        hart_candidates.insert(hart_id, has_memory);
    }

    hart_candidates.get(&0).copied().unwrap_or(false)
        && hart_candidates.get(&1).copied().unwrap_or(false)
}

fn multicore_sync_health_seed(case: &ProviderBinCase, sample_index: usize, attempt: u64) -> u64 {
    case_seed(case, 10_000 + (sample_index as u64 * 32) + attempt)
}

fn build_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    sample_index: usize,
    multicore_sync_config: &MulticoreSyncConfig,
) -> Result<(u64, GeneratedTestCase), String> {
    assert_eq!(
        case.hart_count, 2,
        "multicore sync health only supports dual-hart"
    );

    for attempt in 0..12u64 {
        let seed = multicore_sync_health_seed(case, sample_index, attempt);
        let mut testcase = generate_target_constrained_testcase(
            case,
            TestCaseConfig {
                isa_base: case.isa_base,
                extension_insts_count: multicore_sync_health_extension_inst_count(case),
                extension_inst_scaling: None,
                mem_size: match case.isa_base {
                    ISABase::Rv32 => 32,
                    ISABase::Rv64 => 16,
                },
                float_extension: effective_float_extension(case),
                unaligned_access_required: Some(case.allow_unaligned),
                random_config: RandomConfig {
                    memory_access_config: MemoryAccessConfig {
                        memory_base_address_range: MemoryAccessConfig::default()
                            .memory_base_address_range,
                        memory_offset_immediate_range: (0, 0),
                        memory_base_register_number_range: (8, 12),
                    },
                    register_operand_config: RegisterOperandConfig {
                        integer_register_number_range: (13, 20),
                        floating_point_register_number_range: (13, 20),
                        vector_register_number_range: (13, 20),
                    },
                    csr_selection_config: CsrSelectionConfig {
                        machine_csr_operands_enabled: false,
                        supervisor_csr_operands_enabled: false,
                        hypervisor_csr_operands_enabled: false,
                        virtual_supervisor_csr_operands_enabled: false,
                        rnmi_csr_operands_enabled: false,
                        floating_csr_operands_enabled: false,
                        vector_csr_operands_enabled: false,
                        debug_csr_operands_enabled: false,
                    },
                    isa_profile: isa_profile_for_float(
                        case.isa_base,
                        effective_float_extension(case),
                    ),
                    ..RandomConfig::default()
                },
                seed: Some(seed),
                hart_config: HartConfig::Dual {
                    sync: multicore_sync_config.clone(),
                },
            },
        )?;
        stabilize_multicore_sync_health_testcase(case, &mut testcase);

        if block_has_per_hart_memory_candidates(&testcase.test_program()) {
            return Ok((seed, testcase));
        }
    }

    Err(format!(
        "{} could not produce a dual-hart testcase with memory candidates on both harts after 12 attempts",
        case.id
    ))
}

fn stabilize_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    stabilize_boom_multicore_sync_health_testcase(case, testcase);
    stabilize_cva6_multicore_sync_health_testcase(case, testcase);
    stabilize_ibex_multicore_sync_health_testcase(case, testcase);
    stabilize_kronos_multicore_sync_health_testcase(case, testcase);
    stabilize_picorv32_multicore_sync_health_testcase(case, testcase);
    stabilize_rocket_multicore_sync_health_testcase(case, testcase);
    stabilize_vex_multicore_sync_health_testcase(case, testcase);
    stabilize_xiangshan_multicore_sync_health_testcase(case, testcase);
}

fn stabilize_boom_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    stabilize_named_rv64_hotspot_multicore_sync_health_testcase("boom", 16, case, testcase);
}

fn stabilize_picorv32_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    stabilize_named_rv32_hotspot_multicore_sync_health_testcase(
        "picorv32", 0, 0, 24, case, testcase,
    );
}

fn build_rv32_hotspot_multicore_sync_hart_chunk(
    mem_start: u64,
    first_value: u32,
    stagger_memory_pairs: bool,
) -> Vec<String> {
    let mut lines = vec![
        format!("li x9, 0x{mem_start:x}"),
        format!("li x13, {first_value}"),
        format!("li x14, {}", first_value + 1),
    ];

    if stagger_memory_pairs {
        lines.extend([
            "sw x13, 0(x9)".to_string(),
            "lw x15, 0(x9)".to_string(),
            "sw x14, 0(x9)".to_string(),
            "lw x16, 0(x9)".to_string(),
            "addi x13, x13, 1".to_string(),
            "addi x14, x14, 1".to_string(),
            "sw x13, 0(x9)".to_string(),
            "lw x17, 0(x9)".to_string(),
            "sw x14, 0(x9)".to_string(),
            "lw x18, 0(x9)".to_string(),
            "addi x13, x13, 1".to_string(),
            "addi x14, x14, 1".to_string(),
            "sw x13, 0(x9)".to_string(),
            "lw x19, 0(x9)".to_string(),
            "sw x14, 0(x9)".to_string(),
            "lw x20, 0(x9)".to_string(),
        ]);
    } else {
        lines.extend([
            "sw x13, 0(x9)".to_string(),
            "sw x14, 0(x9)".to_string(),
            "lw x15, 0(x9)".to_string(),
            "lw x16, 0(x9)".to_string(),
            "addi x13, x13, 1".to_string(),
            "addi x14, x14, 1".to_string(),
            "sw x13, 0(x9)".to_string(),
            "sw x14, 0(x9)".to_string(),
            "lw x17, 0(x9)".to_string(),
            "lw x18, 0(x9)".to_string(),
            "addi x13, x13, 1".to_string(),
            "addi x14, x14, 1".to_string(),
            "sw x13, 0(x9)".to_string(),
            "sw x14, 0(x9)".to_string(),
            "lw x19, 0(x9)".to_string(),
            "lw x20, 0(x9)".to_string(),
        ]);
    }

    lines
}

fn extend_rv32_hotspot_multicore_sync_hart(
    lines: &mut Vec<String>,
    mem_offsets: &mut Vec<Option<i64>>,
    mem_start: u64,
    first_value: u32,
    stagger_memory_pairs: bool,
    repeat_count: usize,
) {
    for _ in 0..repeat_count {
        for (idx, line) in build_rv32_hotspot_multicore_sync_hart_chunk(
            mem_start,
            first_value,
            stagger_memory_pairs,
        )
        .into_iter()
        .enumerate()
        {
            lines.push(line);
            mem_offsets.push((idx >= 3).then_some(0));
        }
    }
}

fn build_rv32_reg_overlap_multicore_sync_hart_chunk(seed: u32) -> Vec<String> {
    vec![
        format!("li x21, {seed}"),
        format!("li x22, {}", seed + 3),
        "add x23, x21, x22".to_string(),
        "xor x24, x23, x22".to_string(),
        "sub x25, x24, x21".to_string(),
        "or x26, x25, x23".to_string(),
        "and x21, x26, x24".to_string(),
        "addi x22, x21, 5".to_string(),
        "slli x23, x22, 1".to_string(),
        "srli x24, x23, 1".to_string(),
        "xori x25, x24, 7".to_string(),
        "add x26, x25, x22".to_string(),
    ]
}

fn extend_rv32_reg_overlap_multicore_sync_hart(
    lines: &mut Vec<String>,
    mem_offsets: &mut Vec<Option<i64>>,
    seed: u32,
    repeat_count: usize,
) {
    for _ in 0..repeat_count {
        for line in build_rv32_reg_overlap_multicore_sync_hart_chunk(seed) {
            lines.push(line);
            mem_offsets.push(None);
        }
    }
}

fn build_vex_rv32_hotspot_multicore_sync_hart_chunk(
    mem_start: u64,
    first_value: u32,
) -> Vec<String> {
    vec![
        format!("li x9, 0x{mem_start:x}"),
        format!("li x13, {first_value}"),
        format!("li x14, {}", first_value + 1),
        "sw x13, 0(x9)".to_string(),
        "lw x15, 0(x9)".to_string(),
        "sw x14, 0(x9)".to_string(),
        "lw x16, 0(x9)".to_string(),
        "addi x13, x13, 1".to_string(),
        "addi x14, x14, 1".to_string(),
        "sw x13, 0(x9)".to_string(),
        "lw x17, 0(x9)".to_string(),
        "sw x14, 0(x9)".to_string(),
        "lw x18, 0(x9)".to_string(),
        "addi x13, x13, 1".to_string(),
        "addi x14, x14, 1".to_string(),
        "sw x13, 0(x9)".to_string(),
        "lw x19, 0(x9)".to_string(),
        "sw x14, 0(x9)".to_string(),
        "lw x20, 0(x9)".to_string(),
    ]
}

fn extend_vex_rv32_hotspot_multicore_sync_hart(
    lines: &mut Vec<String>,
    mem_offsets: &mut Vec<Option<i64>>,
    mem_start: u64,
    first_value: u32,
    repeat_count: usize,
) {
    for _ in 0..repeat_count {
        for (idx, line) in build_vex_rv32_hotspot_multicore_sync_hart_chunk(mem_start, first_value)
            .into_iter()
            .enumerate()
        {
            lines.push(line);
            let is_mem = matches!(idx, 3 | 4 | 5 | 6 | 9 | 10 | 11 | 12 | 15 | 16 | 17 | 18);
            mem_offsets.push(is_mem.then_some(0));
        }
    }
}

fn build_vex_rv32_hotspot_multicore_sync_block(
    mem_start: u64,
    repeat_count: usize,
) -> StructuredTestInstructions {
    let mut hart0_lines = Vec::new();
    let mut hart0_offsets = Vec::new();
    let mut hart1_lines = Vec::new();
    let mut hart1_offsets = Vec::new();

    extend_vex_rv32_hotspot_multicore_sync_hart(
        &mut hart0_lines,
        &mut hart0_offsets,
        mem_start,
        1,
        repeat_count,
    );
    extend_vex_rv32_hotspot_multicore_sync_hart(
        &mut hart1_lines,
        &mut hart1_offsets,
        mem_start,
        9,
        repeat_count,
    );

    let mut all_offsets = hart0_offsets;
    all_offsets.extend(hart1_offsets);
    build_block_from_per_hart_lines(
        vec![(0, hart0_lines), (1, hart1_lines)],
        all_offsets,
        ISABase::Rv32,
    )
}

fn build_rv64_hotspot_multicore_sync_hart_chunk(
    mem_start: u64,
    first_value: u64,
    stagger_memory_pairs: bool,
) -> Vec<String> {
    let mut lines = vec![
        format!("li x9, 0x{mem_start:x}"),
        format!("li x13, {first_value}"),
        format!("li x14, {}", first_value + 1),
    ];

    if stagger_memory_pairs {
        lines.extend([
            "sd x13, 0(x9)".to_string(),
            "ld x15, 0(x9)".to_string(),
            "sd x14, 0(x9)".to_string(),
            "ld x16, 0(x9)".to_string(),
            "addi x13, x13, 1".to_string(),
            "addi x14, x14, 1".to_string(),
            "sd x13, 0(x9)".to_string(),
            "ld x17, 0(x9)".to_string(),
            "sd x14, 0(x9)".to_string(),
            "ld x18, 0(x9)".to_string(),
            "addi x13, x13, 1".to_string(),
            "addi x14, x14, 1".to_string(),
            "sd x13, 0(x9)".to_string(),
            "ld x19, 0(x9)".to_string(),
            "sd x14, 0(x9)".to_string(),
            "ld x20, 0(x9)".to_string(),
        ]);
    } else {
        lines.extend([
            "sd x13, 0(x9)".to_string(),
            "sd x14, 0(x9)".to_string(),
            "ld x15, 0(x9)".to_string(),
            "ld x16, 0(x9)".to_string(),
            "addi x13, x13, 1".to_string(),
            "addi x14, x14, 1".to_string(),
            "sd x13, 0(x9)".to_string(),
            "sd x14, 0(x9)".to_string(),
            "ld x17, 0(x9)".to_string(),
            "ld x18, 0(x9)".to_string(),
            "addi x13, x13, 1".to_string(),
            "addi x14, x14, 1".to_string(),
            "sd x13, 0(x9)".to_string(),
            "sd x14, 0(x9)".to_string(),
            "ld x19, 0(x9)".to_string(),
            "ld x20, 0(x9)".to_string(),
        ]);
    }

    lines
}

fn extend_rv64_hotspot_multicore_sync_hart(
    lines: &mut Vec<String>,
    mem_offsets: &mut Vec<Option<i64>>,
    mem_start: u64,
    first_value: u64,
    stagger_memory_pairs: bool,
    repeat_count: usize,
) {
    for _ in 0..repeat_count {
        for (idx, line) in build_rv64_hotspot_multicore_sync_hart_chunk(
            mem_start,
            first_value,
            stagger_memory_pairs,
        )
        .into_iter()
        .enumerate()
        {
            lines.push(line);
            mem_offsets.push((idx >= 3).then_some(0));
        }
    }
}

fn build_rv32_hotspot_multicore_sync_block(
    mem_start: u64,
    prelude_repeat_count: usize,
    suffix_repeat_count: usize,
    repeat_count: usize,
) -> StructuredTestInstructions {
    let mut hart0_lines = Vec::new();
    let mut hart0_offsets = Vec::new();
    let mut hart1_lines = Vec::new();
    let mut hart1_offsets = Vec::new();

    extend_rv32_reg_overlap_multicore_sync_hart(
        &mut hart0_lines,
        &mut hart0_offsets,
        1,
        prelude_repeat_count,
    );
    extend_rv32_hotspot_multicore_sync_hart(
        &mut hart0_lines,
        &mut hart0_offsets,
        mem_start,
        1,
        false,
        repeat_count,
    );
    extend_rv32_reg_overlap_multicore_sync_hart(
        &mut hart0_lines,
        &mut hart0_offsets,
        1,
        suffix_repeat_count,
    );
    extend_rv32_reg_overlap_multicore_sync_hart(
        &mut hart1_lines,
        &mut hart1_offsets,
        1,
        prelude_repeat_count,
    );
    extend_rv32_hotspot_multicore_sync_hart(
        &mut hart1_lines,
        &mut hart1_offsets,
        mem_start,
        9,
        true,
        repeat_count,
    );
    extend_rv32_reg_overlap_multicore_sync_hart(
        &mut hart1_lines,
        &mut hart1_offsets,
        1,
        suffix_repeat_count,
    );

    let mut all_offsets = hart0_offsets;
    all_offsets.extend(hart1_offsets);
    build_block_from_per_hart_lines(
        vec![(0, hart0_lines), (1, hart1_lines)],
        all_offsets,
        ISABase::Rv32,
    )
}

fn build_rv64_hotspot_multicore_sync_block(
    mem_start: u64,
    repeat_count: usize,
) -> StructuredTestInstructions {
    let mut hart0_lines = Vec::new();
    let mut hart0_offsets = Vec::new();
    let mut hart1_lines = Vec::new();
    let mut hart1_offsets = Vec::new();

    extend_rv64_hotspot_multicore_sync_hart(
        &mut hart0_lines,
        &mut hart0_offsets,
        mem_start,
        1,
        false,
        repeat_count,
    );
    extend_rv64_hotspot_multicore_sync_hart(
        &mut hart1_lines,
        &mut hart1_offsets,
        mem_start,
        9,
        true,
        repeat_count,
    );

    let mut all_offsets = hart0_offsets;
    all_offsets.extend(hart1_offsets);
    build_block_from_per_hart_lines(
        vec![(0, hart0_lines), (1, hart1_lines)],
        all_offsets,
        ISABase::Rv64,
    )
}

fn stabilize_named_rv32_hotspot_multicore_sync_health_testcase(
    expected_impl: &str,
    prelude_repeat_count: usize,
    suffix_repeat_count: usize,
    repeat_count: usize,
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    if case.impl_ref != expected_impl || case.hart_count != 2 || case.isa_base != ISABase::Rv32 {
        return;
    }

    let Some((mem_start, _)) = testcase.mem_range.get(&case.impl_ref).copied() else {
        return;
    };

    let init_block = build_block_from_rendered_parts(
        vec![
            format!("li x9, 0x{mem_start:x}"),
            "li x10, 0".to_string(),
            "sw x10, 0(x9)".to_string(),
        ],
        vec![None, None, Some(0)],
        case.isa_base,
    );
    testcase.set_init_program(init_block);
    testcase.set_test_program(build_rv32_hotspot_multicore_sync_block(
        mem_start,
        prelude_repeat_count,
        suffix_repeat_count,
        repeat_count,
    ));
}

fn stabilize_named_rv64_hotspot_multicore_sync_health_testcase(
    expected_impl: &str,
    repeat_count: usize,
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    if case.impl_ref != expected_impl || case.hart_count != 2 || case.isa_base != ISABase::Rv64 {
        return;
    }

    let Some((mem_start, _)) = testcase.mem_range.get(&case.impl_ref).copied() else {
        return;
    };

    let init_block = build_block_from_rendered_parts(
        vec![
            format!("li x9, 0x{mem_start:x}"),
            "li x10, 0".to_string(),
            "sd x10, 0(x9)".to_string(),
        ],
        vec![None, None, Some(0)],
        case.isa_base,
    );
    testcase.set_init_program(init_block);
    testcase.set_test_program(build_rv64_hotspot_multicore_sync_block(
        mem_start,
        repeat_count,
    ));
}

fn stabilize_cva6_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    stabilize_named_rv32_hotspot_multicore_sync_health_testcase("cva6", 0, 0, 96, case, testcase);
}

fn stabilize_ibex_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    stabilize_named_rv32_hotspot_multicore_sync_health_testcase("ibex", 0, 0, 16, case, testcase);
}

fn stabilize_kronos_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    stabilize_named_rv32_hotspot_multicore_sync_health_testcase("kronos", 0, 0, 24, case, testcase);
}

fn stabilize_rocket_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    stabilize_named_rv32_hotspot_multicore_sync_health_testcase("rocket", 0, 0, 32, case, testcase);
}

fn stabilize_vex_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    if case.impl_ref != "vex" || case.hart_count != 2 || case.isa_base != ISABase::Rv32 {
        return;
    }

    let Some((mem_start, _)) = testcase.mem_range.get(&case.impl_ref).copied() else {
        return;
    };

    let init_block = build_block_from_rendered_parts(
        vec![
            format!("li x9, 0x{mem_start:x}"),
            "li x10, 0".to_string(),
            "sw x10, 0(x9)".to_string(),
        ],
        vec![None, None, Some(0)],
        case.isa_base,
    );
    testcase.set_init_program(init_block);
    testcase.set_test_program(build_vex_rv32_hotspot_multicore_sync_block(mem_start, 32));
}

fn stabilize_xiangshan_multicore_sync_health_testcase(
    case: &ProviderBinCase,
    testcase: &mut GeneratedTestCase,
) {
    stabilize_named_rv64_hotspot_multicore_sync_health_testcase("xiangshan", 32, case, testcase);
}

fn multicore_sync_health_case_score(case: &ProviderBinCase) -> Option<i32> {
    if case.hart_count != 2 || case.coverage_mode != CoverageMode::Off {
        return None;
    }
    if !MULTICORE_SYNC_HEALTH_IMPLS.contains(&case.impl_ref.as_str()) {
        return None;
    }

    let mut score = 0i32;
    score += match case.requested_float_extension {
        FloatExtension::None => 40,
        FloatExtension::F => 20,
        FloatExtension::FD => 10,
    };
    if !case.allow_unaligned {
        score += 20;
    }
    if case.isa_base == ISABase::Rv32 {
        score += 10;
    }
    Some(score)
}

fn collect_multicore_sync_health_cases(cases: &[ProviderBinCase]) -> Vec<ProviderBinCase> {
    let mut selected = Vec::new();

    for impl_ref in MULTICORE_SYNC_HEALTH_IMPLS {
        let mut best_case: Option<&ProviderBinCase> = None;
        let mut best_score = i32::MIN;

        for case in cases.iter().filter(|case| case.impl_ref == *impl_ref) {
            let Some(score) = multicore_sync_health_case_score(case) else {
                continue;
            };

            let should_replace = match best_case {
                None => true,
                Some(current) => {
                    score > best_score || (score == best_score && case.id < current.id)
                }
            };
            if should_replace {
                best_case = Some(case);
                best_score = score;
            }
        }

        if let Some(case) = best_case {
            selected.push(case.clone());
        }
    }

    selected
}

fn render_provider_multicore_sync_health_report(
    path: &Path,
    summary: &ProviderMulticoreSyncHealthCaseSummary,
) -> Result<(), String> {
    let mut markdown = String::new();
    markdown.push_str("# Provider Multicore Sync Health Report\n\n");
    markdown.push_str("| Field | Value |\n|---|---|\n");
    markdown.push_str(&format!("| Status | {} |\n", summary.status));
    markdown.push_str(&format!("| Case | {} |\n", summary.id));
    markdown.push_str(&format!("| Impl | {} |\n", summary.impl_ref));
    markdown.push_str(&format!("| Source | {} |\n", summary.source_label));
    markdown.push_str(&format!("| Binary | `{}` |\n", summary.binary_path));
    markdown.push_str(&format!("| ISA | {} |\n", summary.isa_base));
    markdown.push_str(&format!("| Hart Count | {} |\n", summary.hart_count));
    markdown.push_str(&format!(
        "| Float | requested={} effective={} |\n",
        summary.requested_float_extension, summary.effective_float_extension
    ));
    markdown.push_str(&format!("| Coverage | {} |\n", summary.coverage_mode));
    markdown.push_str(&format!(
        "| Allow Unaligned | {} |\n",
        summary.allow_unaligned
    ));
    markdown.push_str(&format!("| Samples | {} |\n", summary.sample_count));
    markdown.push_str(&format!(
        "| Seed Range | {}..={} |\n",
        summary.seed_start, summary.seed_end
    ));
    markdown.push_str(&format!("| Timeout (s) | {} |\n", summary.timeout_secs));
    markdown.push_str(&format!(
        "| Same-Address Overlap Observed | {} |\n\n",
        summary.same_address_overlap_observed
    ));

    markdown.push_str("## Metric Definitions\n\n");
    markdown.push_str("All ratio-style sync metrics below are reported as percentages.\n\n");
    markdown.push_str(
        "For `Simul`, the denominator is the union of cycles where at least one hart is active; idle gaps where neither hart is active are excluded.\n\n",
    );
    markdown
        .push_str("- `Simul` = simultaneous active-cycle ratio within that active-cycle union\n");
    markdown.push_str("- `Reg` = ratio of simultaneous cycles where the two harts touch overlapping register sets\n");
    markdown.push_str("- `Mem` = ratio of both-memory simultaneous cycles where the two harts hit the same address\n");
    markdown.push_str(
        "- `Score` = weighted combined ratio across the synchronization patterns above\n\n",
    );

    markdown.push_str("## Strategy\n\n");
    markdown.push_str(&format!(
        "- interaction_slot_ratio = {:.2}%\n",
        summary.multicore_sync_config.interaction_slot_ratio * 100.0
    ));
    markdown.push_str(&format!(
        "- shared_address_ratio = {:.2}%\n",
        summary.multicore_sync_config.shared_address_ratio * 100.0
    ));
    markdown.push_str(&format!(
        "- prefer_focused_sequences = {}\n\n",
        summary.multicore_sync_config.prefer_focused_sequences
    ));

    markdown.push_str("## Thresholds\n\n");
    markdown.push_str(&format!(
        "- min_simul_mean = {}\n",
        fmt_percent(summary.thresholds.min_simultaneous_cycle_rate_mean)
    ));
    markdown.push_str(&format!(
        "- min_reg_mean = {}\n",
        fmt_percent(summary.thresholds.min_register_overlap_cycle_rate_mean)
    ));
    markdown.push_str(&format!(
        "- min_mem_mean = {}\n",
        fmt_percent(summary.thresholds.min_memory_contention_rate_mean)
    ));
    markdown.push_str(&format!(
        "- min_score_mean = {}\n\n",
        fmt_percent(summary.thresholds.min_cycle_contention_score_mean)
    ));

    markdown.push_str("## Aggregate Metrics\n\n");
    markdown.push_str("| Metric | Min | Mean | Max |\n");
    markdown.push_str("|---|---:|---:|---:|\n");
    for (label, stats) in [
        ("Simul", summary.simultaneous_cycle_rate),
        ("Reg", summary.register_overlap_cycle_rate),
        ("Mem", summary.memory_contention_rate),
        ("Score", summary.cycle_contention_score),
    ] {
        markdown.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            label,
            fmt_percent(stats.min),
            fmt_percent(stats.mean),
            fmt_percent(stats.max)
        ));
    }
    markdown.push('\n');

    markdown.push_str("## Samples\n\n");
    markdown.push_str("| Sample | Seed | Covered Cycles | Simul | Reg | Mem | Score |\n");
    markdown.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for sample in &summary.samples {
        markdown.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            sample.sample_index,
            sample.seed,
            sample.metrics.covered_cycle_count,
            fmt_percent(sample.metrics.simultaneous_cycle_rate),
            fmt_percent(sample.metrics.register_overlap_cycle_rate),
            fmt_percent(sample.metrics.same_address_given_both_memory_cycle_rate),
            fmt_percent(sample.metrics.cycle_contention_score),
        ));
    }
    markdown.push('\n');

    markdown.push_str("## Failed Checks\n\n");
    if summary.failed_checks.is_empty() {
        markdown.push_str("- none\n");
    } else {
        for failed_check in &summary.failed_checks {
            markdown.push_str(&format!("- {}\n", failed_check));
        }
    }

    write_text(path, &markdown)
}

fn render_provider_multicore_sync_health_matrix_report(
    path: &Path,
    summaries: &[ProviderMulticoreSyncHealthCaseSummary],
    matrix_summary: &ProviderMulticoreSyncHealthMatrixSummary,
) -> Result<(), String> {
    let mut markdown = String::new();
    markdown.push_str("# Provider Multicore Sync Health Matrix Summary\n\n");
    markdown.push_str("| Field | Value |\n|---|---|\n");
    markdown.push_str(&format!(
        "| Total Cases | {} |\n",
        matrix_summary.total_cases
    ));
    markdown.push_str(&format!(
        "| Passed Cases | {} |\n",
        matrix_summary.passed_cases
    ));
    markdown.push_str(&format!(
        "| Failed Cases | {} |\n",
        matrix_summary.failed_cases
    ));
    markdown.push_str(&format!(
        "| Average Simul | {} |\n",
        fmt_percent(matrix_summary.average_simultaneous_cycle_rate)
    ));
    markdown.push_str(&format!(
        "| Average Reg | {} |\n",
        fmt_percent(matrix_summary.average_register_overlap_cycle_rate)
    ));
    markdown.push_str(&format!(
        "| Average Mem | {} |\n",
        fmt_percent(matrix_summary.average_memory_contention_rate)
    ));
    markdown.push_str(&format!(
        "| Average Score | {} |\n",
        fmt_percent(matrix_summary.average_cycle_contention_score)
    ));
    markdown.push_str(&format!(
        "| Cases With Same-Address Overlap | {} |\n\n",
        matrix_summary.cases_with_same_address_overlap
    ));

    markdown.push_str("| Case | Impl | Status | Samples | Simul | Reg | Mem | Score | Error |\n");
    markdown.push_str("|---|---|---|---:|---:|---:|---:|---:|---|\n");
    for summary in summaries {
        markdown.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            summary.id,
            summary.impl_ref,
            summary.status,
            summary.sample_count,
            fmt_percent(summary.simultaneous_cycle_rate.mean),
            fmt_percent(summary.register_overlap_cycle_rate.mean),
            fmt_percent(summary.memory_contention_rate.mean),
            fmt_percent(summary.cycle_contention_score.mean),
            summary.error.clone().unwrap_or_default()
        ));
    }
    markdown.push('\n');

    if !matrix_summary.failures.is_empty() {
        markdown.push_str("## Failures\n\n");
        for failure in &matrix_summary.failures {
            markdown.push_str(&format!("- {}\n", failure));
        }
    }

    write_text(path, &markdown)
}

fn run_provider_multicore_sync_health_case(
    case: &ProviderBinCase,
    run_root: &Path,
) -> Result<ProviderMulticoreSyncHealthCaseSummary, String> {
    let case_dir = run_root.join(&case.id);
    fs::create_dir_all(&case_dir)
        .map_err(|e| format!("failed to create case dir {}: {e}", case_dir.display()))?;

    let timeout_secs = default_timeout_secs(case);
    let _timeout_guard = set_env_var(EXEC_TIMEOUT_ENV, timeout_secs.to_string());
    let _target_binary_guard = set_env_var(
        target_binary_env_var(case),
        case.binary_path.to_string_lossy(),
    );

    let multicore_sync_config = aggressive_multicore_sync_config();
    let thresholds = multicore_sync_health_thresholds(case);
    let mut samples = Vec::with_capacity(MULTICORE_SYNC_HEALTH_SAMPLE_COUNT);

    for sample_index in 0..MULTICORE_SYNC_HEALTH_SAMPLE_COUNT {
        let sample_dir = case_dir.join(format!("sample_{sample_index:04}"));
        fs::create_dir_all(&sample_dir)
            .map_err(|e| format!("failed to create {}: {e}", sample_dir.display()))?;

        let (seed, testcase) =
            build_multicore_sync_health_testcase(case, sample_index, &multicore_sync_config)?;

        let single_impl = std::slice::from_ref(&case.impl_ref);
        let result = execute_testcase_with_artifacts(
            &testcase,
            single_impl,
            &sample_dir,
            CoverageMode::Off,
            ExecutionArtifactOptions {
                testcase_filename: "generated_testcase.json",
                emit_execution_output_json: true,
                emit_execution_report_md: true,
                emit_execution_timeline_svg: true,
                multicore_sync_metric_failure_mode: MulticoreSyncMetricFailureMode::Strict,
            },
        )
        .map_err(|e| {
            format!(
                "multicore-sync-health execution failed for {} sample {}: {e}",
                case.id, sample_index
            )
        })?;
        let metrics = result
            .multicore_sync_metrics
            .get(&case.impl_ref)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "multicore-sync-health execution did not produce multicore sync metrics for {} sample {}",
                    case.id, sample_index
                )
            })?;

        samples.push(ProviderMulticoreSyncHealthSampleSummary {
            sample_index,
            seed,
            metrics,
        });
    }

    let simultaneous_cycle_rate = min_mean_max_summary(
        &samples
            .iter()
            .map(|sample| sample.metrics.simultaneous_cycle_rate)
            .collect::<Vec<_>>(),
    );
    let register_overlap_cycle_rate = min_mean_max_summary(
        &samples
            .iter()
            .map(|sample| sample.metrics.register_overlap_cycle_rate)
            .collect::<Vec<_>>(),
    );
    let memory_contention_rate = min_mean_max_summary(
        &samples
            .iter()
            .map(|sample| sample.metrics.same_address_given_both_memory_cycle_rate)
            .collect::<Vec<_>>(),
    );
    let cycle_contention_score = min_mean_max_summary(
        &samples
            .iter()
            .map(|sample| sample.metrics.cycle_contention_score)
            .collect::<Vec<_>>(),
    );

    let mut failed_checks = Vec::new();
    if samples
        .iter()
        .any(|sample| sample.metrics.covered_cycle_count == 0)
    {
        failed_checks.push("at least one sample produced zero covered cycles".to_string());
    }
    if samples.iter().any(|sample| {
        sample.metrics.hart0_instruction_count == 0 || sample.metrics.hart1_instruction_count == 0
    }) {
        failed_checks
            .push("at least one sample executed zero instructions on one hart".to_string());
    }
    if samples
        .iter()
        .any(|sample| sample.metrics.timing_coverage_rate < 0.999)
    {
        failed_checks
            .push("timing coverage dropped below 0.999 for at least one sample".to_string());
    }
    if samples
        .iter()
        .all(|sample| sample.metrics.both_memory_cycle_count == 0)
    {
        failed_checks.push(
            "all samples produced zero simultaneous dual-memory cycles in the overlap window"
                .to_string(),
        );
    }
    if simultaneous_cycle_rate.mean < thresholds.min_simultaneous_cycle_rate_mean {
        failed_checks.push(format!(
            "simul.mean below threshold: {} < {}",
            fmt_percent(simultaneous_cycle_rate.mean),
            fmt_percent(thresholds.min_simultaneous_cycle_rate_mean)
        ));
    }
    if register_overlap_cycle_rate.mean < thresholds.min_register_overlap_cycle_rate_mean {
        failed_checks.push(format!(
            "reg.mean below threshold: {} < {}",
            fmt_percent(register_overlap_cycle_rate.mean),
            fmt_percent(thresholds.min_register_overlap_cycle_rate_mean)
        ));
    }
    if memory_contention_rate.mean < thresholds.min_memory_contention_rate_mean {
        failed_checks.push(format!(
            "mem.mean below threshold: {} < {}",
            fmt_percent(memory_contention_rate.mean),
            fmt_percent(thresholds.min_memory_contention_rate_mean)
        ));
    }
    if cycle_contention_score.mean < thresholds.min_cycle_contention_score_mean {
        failed_checks.push(format!(
            "score.mean below threshold: {} < {}",
            fmt_percent(cycle_contention_score.mean),
            fmt_percent(thresholds.min_cycle_contention_score_mean)
        ));
    }

    let config_mismatch = samples.iter().any(|sample| {
        let metrics = &sample.metrics;
        (metrics.configured_interaction_slot_ratio - multicore_sync_config.interaction_slot_ratio)
            .abs()
            > 1e-9
            || (metrics.configured_shared_address_ratio
                - multicore_sync_config.shared_address_ratio)
                .abs()
                > 1e-9
            || metrics.prefer_focused_sequences != multicore_sync_config.prefer_focused_sequences
    });
    if config_mismatch {
        failed_checks
            .push("reported metrics do not echo the requested multicore sync config".to_string());
    }

    let same_address_overlap_observed = samples
        .iter()
        .any(|sample| sample.metrics.same_address_cycle_count > 0);
    let seed_start = samples.first().map(|sample| sample.seed).unwrap_or(0);
    let seed_end = samples.last().map(|sample| sample.seed).unwrap_or(0);
    let summary = ProviderMulticoreSyncHealthCaseSummary {
        status: if failed_checks.is_empty() {
            "pass".to_string()
        } else {
            "fail".to_string()
        },
        id: case.id.clone(),
        impl_ref: case.impl_ref.clone(),
        source_label: case.source_label.clone(),
        binary_path: case.binary_path.display().to_string(),
        binary_name: case.binary_name.clone(),
        isa_base: format!("{:?}", case.isa_base),
        requested_float_extension: format!("{:?}", case.requested_float_extension),
        effective_float_extension: format!("{:?}", effective_float_extension(case)),
        coverage_mode: format!("{:?}", case.coverage_mode),
        allow_unaligned: case.allow_unaligned,
        hart_count: case.hart_count,
        sample_count: samples.len(),
        seed_start,
        seed_end,
        timeout_secs,
        multicore_sync_config,
        thresholds,
        simultaneous_cycle_rate,
        register_overlap_cycle_rate,
        memory_contention_rate,
        cycle_contention_score,
        same_address_overlap_observed,
        samples,
        failed_checks,
        error: None,
    };
    render_provider_multicore_sync_health_report(
        &case_dir.join("multicore_sync_health_report.md"),
        &summary,
    )?;
    Ok(summary)
}

fn render_execution_reports(
    run_dir: &Path,
    outputs: &HashMap<String, ExecutionOutput>,
    impls: &[String],
) -> Result<(), String> {
    for impl_ref in impls {
        let output = outputs
            .get(impl_ref)
            .ok_or_else(|| format!("missing execution output for {impl_ref}"))?;
        let impl_dir = run_dir.join(impl_ref);
        fs::create_dir_all(&impl_dir)
            .map_err(|e| format!("failed to create {}: {e}", impl_dir.display()))?;
        write_json_pretty(&impl_dir.join("execution_output.json"), output)?;
        generate_execution_timeline_svg_from_output(
            output,
            impl_dir.join("execution_timeline.svg"),
        )
        .map_err(|e| format!("failed to generate execution timeline for {impl_ref}: {e}"))?;
        generate_execution_report(output, impl_dir.join("execution_report.md"))
            .map_err(|e| format!("failed to generate execution report for {impl_ref}: {e}"))?;
    }
    Ok(())
}

fn aggregate_as_hart_context_output(output: &ExecutionContextOutput, hart_id: u32) -> HartLogView {
    let mut instructions = output
        .logical_view
        .iter()
        .filter(|instruction| instruction.hart_id == hart_id)
        .collect::<Vec<_>>();
    instructions.sort_by_key(|instruction| instruction.local_instruction_index);

    HartLogView {
        instruction_count: instructions.len(),
        observed_slots: instructions
            .iter()
            .map(|instruction| {
                !instruction.asm_instruction_indices.is_empty()
                    || instruction.timing.is_recorded()
                    || !instruction.register_writes.is_empty()
                    || !instruction.memory_writes.is_empty()
                    || !instruction.exceptions.is_empty()
            })
            .collect(),
        register_changes: instructions
            .iter()
            .map(|instruction| instruction.register_writes.clone())
            .collect(),
        memory_changes: instructions
            .iter()
            .map(|instruction| instruction.memory_writes.clone())
            .collect(),
        exceptions: instructions
            .iter()
            .flat_map(|instruction| {
                instruction
                    .exceptions
                    .iter()
                    .map(|_| HartLogExceptionSummary {
                        instruction_index: instruction.local_instruction_index,
                    })
            })
            .collect(),
    }
}

fn build_execution_context_output_with_weak_fallback(
    output: ExecutionOutput,
    mem_range: (u64, u64),
    per_hart_input_instructions: &[(
        u32,
        Vec<riscv_instruction::InstructionSequence<riscv_instruction::RiscvInstruction>>,
    )],
) -> Result<ExecutionContextOutput, ContextBuildError> {
    match build_execution_context_output(output.clone(), mem_range, per_hart_input_instructions) {
        Ok(context) => Ok(context),
        Err(ContextBuildError::MissingLogicalInstructionExecution { .. }) => {
            build_partial_execution_context_output(output, mem_range, per_hart_input_instructions)
        }
        Err(error) => Err(error),
    }
}

fn build_context_outputs_for_testcase(
    testcase: &GeneratedTestCase,
    outputs: &HashMap<String, ExecutionOutput>,
) -> Result<HashMap<String, ExecutionContextOutput>, String> {
    let mut contexts = HashMap::new();
    for (impl_ref, output) in outputs {
        let per_hart_typed = testcase
            .combined_typed_sequences_per_hart_of(impl_ref)
            .ok_or_else(|| format!("missing combined typed instruction block for {impl_ref}"))?;
        let mem_range = testcase
            .mem_range
            .get(impl_ref)
            .copied()
            .ok_or_else(|| format!("missing memory range for {impl_ref}"))?;
        let context = build_execution_context_output_with_weak_fallback(
            output.clone(),
            mem_range,
            &per_hart_typed,
        )
        .map_err(|e| format!("failed to build execution context output for {impl_ref}: {e}"))?;
        contexts.insert(impl_ref.clone(), context);
    }
    Ok(contexts)
}

// This health sweep exercises the dual-hart overlap strategy on real provider binaries and
// verifies that the reported metrics are derived from real cycle windows with non-trivial
// simultaneous execution, register overlap, and memory overlap activity.
#[test]
fn provider_bins_have_multicore_sync_metric_health() {
    let _guard = TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _plugin_dirs_guard = configure_plugin_dirs();

    let provider_root = provider_root();
    let artifact_dir = provider_artifact_dir(&provider_root);
    let spike_path = resolve_spike_path();
    let _provider_env_guards = ensure_aux_provider_envs(&provider_root, &artifact_dir, &spike_path);
    let _provider_binary_env_guards = configure_provider_binary_envs(&provider_root, &artifact_dir);

    let all_cases = collect_provider_bins(&provider_root, &artifact_dir);
    let cases = collect_multicore_sync_health_cases(&all_cases);
    assert!(
        !cases.is_empty(),
        "no multicore sync health provider bins found under {}",
        artifact_dir.display()
    );

    let run_root = run_root_for().join("provider_multicore_sync_health");
    fs::create_dir_all(&run_root)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", run_root.display()));

    let mut failures = Vec::new();
    let mut summaries = Vec::new();

    for case in cases {
        let case_dir = run_root.join(&case.id);
        let summary = match run_provider_multicore_sync_health_case(&case, &run_root) {
            Ok(summary) => summary,
            Err(error) => ProviderMulticoreSyncHealthCaseSummary {
                status: "fail".to_string(),
                id: case.id.clone(),
                impl_ref: case.impl_ref.clone(),
                source_label: case.source_label.clone(),
                binary_path: case.binary_path.display().to_string(),
                binary_name: case.binary_name.clone(),
                isa_base: format!("{:?}", case.isa_base),
                requested_float_extension: format!("{:?}", case.requested_float_extension),
                effective_float_extension: format!("{:?}", effective_float_extension(&case)),
                coverage_mode: format!("{:?}", case.coverage_mode),
                allow_unaligned: case.allow_unaligned,
                hart_count: case.hart_count,
                sample_count: 0,
                seed_start: 0,
                seed_end: 0,
                timeout_secs: default_timeout_secs(&case),
                multicore_sync_config: aggressive_multicore_sync_config(),
                thresholds: multicore_sync_health_thresholds(&case),
                simultaneous_cycle_rate: min_mean_max_summary(&[]),
                register_overlap_cycle_rate: min_mean_max_summary(&[]),
                memory_contention_rate: min_mean_max_summary(&[]),
                cycle_contention_score: min_mean_max_summary(&[]),
                same_address_overlap_observed: false,
                samples: Vec::new(),
                failed_checks: Vec::new(),
                error: Some(error.clone()),
            },
        };

        fs::create_dir_all(&case_dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", case_dir.display()));
        write_json_pretty(
            &case_dir.join("multicore_sync_health_summary.json"),
            &summary,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        write_text(
            &case_dir.join("multicore_sync_health_summary.txt"),
            &format!(
                "status={}\nid={}\nimpl={}\nsource={}\nbinary={}\nisa={}\nfloat_request={}\nfloat_effective={}\ncoverage={}\nallow_unaligned={}\nhart_count={}\nsample_count={}\nseed_start={}\nseed_end={}\ntimeout_secs={}\nsimul_mean_pct={:.2}\nreg_mean_pct={:.2}\nmem_mean_pct={:.2}\nscore_mean_pct={:.2}\nsame_address_overlap_observed={}\nerror={:?}\n",
                summary.status,
                summary.id,
                summary.impl_ref,
                summary.source_label,
                summary.binary_path,
                summary.isa_base,
                summary.requested_float_extension,
                summary.effective_float_extension,
                summary.coverage_mode,
                summary.allow_unaligned,
                summary.hart_count,
                summary.sample_count,
                summary.seed_start,
                summary.seed_end,
                summary.timeout_secs,
                summary.simultaneous_cycle_rate.mean * 100.0,
                summary.register_overlap_cycle_rate.mean * 100.0,
                summary.memory_contention_rate.mean * 100.0,
                summary.cycle_contention_score.mean * 100.0,
                summary.same_address_overlap_observed,
                summary.error,
            ),
        )
        .unwrap_or_else(|e| panic!("{e}"));

        if summary.error.is_none() {
            render_provider_multicore_sync_health_report(
                &case_dir.join("multicore_sync_health_report.md"),
                &summary,
            )
            .unwrap_or_else(|e| panic!("{e}"));
        }

        if let Some(error) = &summary.error {
            failures.push(format!("{}: {}", summary.id, error));
        } else if !summary.failed_checks.is_empty() {
            failures.push(format!(
                "{}: {}",
                summary.id,
                summary.failed_checks.join("; ")
            ));
        }

        summaries.push(summary);
    }

    let cases_with_same_address_overlap = summaries
        .iter()
        .filter(|summary| summary.same_address_overlap_observed)
        .count();
    if summaries.len() > 1 && cases_with_same_address_overlap == 0 {
        failures.push(
            "no selected provider observed any same-address overlap on real overlapping cycles"
                .to_string(),
        );
    }

    let matrix_summary = ProviderMulticoreSyncHealthMatrixSummary {
        total_cases: summaries.len(),
        passed_cases: summaries
            .iter()
            .filter(|summary| summary.status == "pass")
            .count(),
        failed_cases: summaries
            .iter()
            .filter(|summary| summary.status != "pass")
            .count(),
        average_simultaneous_cycle_rate: average(
            &summaries
                .iter()
                .map(|summary| summary.simultaneous_cycle_rate.mean)
                .collect::<Vec<_>>(),
        ),
        average_memory_contention_rate: average(
            &summaries
                .iter()
                .map(|summary| summary.memory_contention_rate.mean)
                .collect::<Vec<_>>(),
        ),
        average_cycle_contention_score: average(
            &summaries
                .iter()
                .map(|summary| summary.cycle_contention_score.mean)
                .collect::<Vec<_>>(),
        ),
        average_register_overlap_cycle_rate: average(
            &summaries
                .iter()
                .map(|summary| summary.register_overlap_cycle_rate.mean)
                .collect::<Vec<_>>(),
        ),
        cases_with_same_address_overlap,
        case_ids: summaries.iter().map(|summary| summary.id.clone()).collect(),
        failures: failures.clone(),
    };

    write_json_pretty(&run_root.join("matrix_summary.json"), &matrix_summary)
        .unwrap_or_else(|e| panic!("{e}"));
    write_json_pretty(&run_root.join("case_summaries.json"), &summaries)
        .unwrap_or_else(|e| panic!("{e}"));
    render_provider_multicore_sync_health_matrix_report(
        &run_root.join("matrix_summary.md"),
        &summaries,
        &matrix_summary,
    )
    .unwrap_or_else(|e| panic!("{e}"));

    if !failures.is_empty() {
        panic!(
            "provider bins found {} failing multicore-sync-health case(s) under {}:\n{}",
            failures.len(),
            run_root.display(),
            failures.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// Compact log/streaming health helpers
// ---------------------------------------------------------------------------
//
// Spike diff health 会复用这一套轻量 testcase 和 shape 检查：
// - 对同一个小 testcase 比对 target vs Spike 的寄存器/内存/异常形状
// - 再跑一次 streaming cancel/full probe，确认事件流行为稳定
// 这些 helper 本身不再作为独立测试入口存在。
const LITE_LOG_HEALTH_SHAPE_MATCH_THRESHOLD: f64 = 0.90;
const LITE_LOG_HEALTH_TAIL_PROBE_SLOT_COUNT: usize = 5;
const LITE_LOG_HEALTH_CANCEL_PROBE_FRAMEWORK_PREFIX_LIMIT: usize = 64;

#[derive(Clone, Debug, Serialize)]
struct LiteLogHealthCase {
    id: String,
    impl_ref: String,
    plugin_source_path: String,
    isa_base: ISABase,
    requested_float_extension: FloatExtension,
    coverage_mode: CoverageMode,
    allow_unaligned: bool,
    hart_count: u32,
}

#[derive(Debug, Clone, Serialize)]
struct LiteSlotMismatch {
    instruction_index: usize,
    kind: String,
    target: Vec<String>,
    spike: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct LiteHartLogHealthSummary {
    hart_id: u32,
    target_instruction_count: usize,
    spike_instruction_count: usize,
    compared_slot_count: usize,
    shape_matched_slot_count: usize,
    failed_slot_count: usize,
    shape_match_ratio: f64,
    shape_match_threshold: f64,
    register_slot_match_ratio: f64,
    memory_slot_match_ratio: f64,
    exception_slot_match_ratio: f64,
    target_active_slot_count: usize,
    spike_active_slot_count: usize,
    register_mismatch_count: usize,
    memory_mismatch_count: usize,
    exception_mismatch_count: usize,
    instruction_presence_mismatch_count: usize,
    tail_probe_slot_count: usize,
    tail_probe_shape_matched_slot_count: usize,
    tail_probe_failed_slot_count: usize,
    tail_probe_inactive_slot_count: usize,
    sample_mismatches: Vec<LiteSlotMismatch>,
    failed_checks: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LiteStreamingCancelProbeSummary {
    impl_ref: String,
    event_count: usize,
    instruction_event_count: usize,
    source_mapped_instruction_event_count: usize,
    framework_instruction_event_count: usize,
    completed_event_count: usize,
    failed_event_count: usize,
    cancelled_after_first_instruction: bool,
    elapsed_ms: u128,
    failed_checks: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct LiteStreamingFullProbeSummary {
    impl_ref: String,
    event_count: usize,
    instruction_event_count: usize,
    source_mapped_instruction_event_count: usize,
    framework_instruction_event_count: usize,
    completed_event_count: usize,
    failed_event_count: usize,
    elapsed_ms: u128,
    harts: Vec<LiteHartLogHealthSummary>,
    failed_checks: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct LiteLogHealthCaseSummary {
    status: String,
    id: String,
    impl_ref: String,
    plugin_source_path: String,
    isa_base: String,
    requested_float_extension: String,
    effective_float_extension: String,
    coverage_mode: String,
    allow_unaligned: bool,
    hart_count: u32,
    testcase_seed: u64,
    timeout_secs: u64,
    harts: Vec<LiteHartLogHealthSummary>,
    streaming_cancel_probes: Vec<LiteStreamingCancelProbeSummary>,
    streaming_full_probes: Vec<LiteStreamingFullProbeSummary>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProviderDiffHealthRunSummary {
    sample_index: usize,
    seed: u64,
    status: String,
    elapsed_ms: u128,
    exception_removal_rounds: usize,
    write_removal_rounds: usize,
    output_root: String,
    bug_category: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProviderDiffHealthCaseSummary {
    status: String,
    suite: String,
    id: String,
    impl_ref: String,
    source_label: String,
    binary_path: String,
    binary_name: String,
    isa_base: String,
    requested_float_extension: String,
    effective_float_extension: String,
    coverage_mode: String,
    allow_unaligned: bool,
    hart_count: u32,
    timeout_secs: u64,
    requested_runs: usize,
    case_concurrency: usize,
    diff_run_passes: usize,
    diff_run_iteration_limits: usize,
    diff_run_failures: usize,
    diff_runs: Vec<ProviderDiffHealthRunSummary>,
    log_health: Option<LiteLogHealthCaseSummary>,
    failed_checks: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProviderDiffHealthMatrixSummary {
    suite: String,
    total_cases: usize,
    passed_cases: usize,
    failed_cases: usize,
    requested_runs: usize,
    case_concurrency: usize,
    max_parallel_budget: usize,
    case_parallelism: usize,
    diff_run_passes: usize,
    diff_run_iteration_limits: usize,
    diff_run_failures: usize,
    case_ids: Vec<String>,
    failures: Vec<String>,
}

fn lite_register_name_set(entries: &[riscv_fuzz_plugin_api::RegisterValue]) -> BTreeSet<String> {
    entries.iter().map(|entry| entry.name.clone()).collect()
}

fn lite_memory_address_set(entries: &[riscv_fuzz_plugin_api::MemValue]) -> BTreeSet<u64> {
    entries.iter().map(|entry| entry.addr).collect()
}

fn lite_exception_present(
    exceptions: &[HartLogExceptionSummary],
    instruction_index: usize,
) -> bool {
    exceptions
        .iter()
        .any(|exc| exc.instruction_index == instruction_index)
}

fn lite_register_name_set_at(view: &HartLogView, instruction_index: usize) -> BTreeSet<String> {
    view.register_changes
        .get(instruction_index)
        .map(|entries| lite_register_name_set(entries))
        .unwrap_or_default()
}

fn lite_memory_address_set_at(view: &HartLogView, instruction_index: usize) -> BTreeSet<u64> {
    view.memory_changes
        .get(instruction_index)
        .map(|entries| lite_memory_address_set(entries))
        .unwrap_or_default()
}

fn lite_slot_active(view: &HartLogView, instruction_index: usize) -> bool {
    view.register_changes
        .get(instruction_index)
        .is_some_and(|entries| !entries.is_empty())
        || view
            .memory_changes
            .get(instruction_index)
            .is_some_and(|entries| !entries.is_empty())
        || lite_exception_present(&view.exceptions, instruction_index)
}

fn lite_slot_observed(view: &HartLogView, instruction_index: usize) -> bool {
    view.observed_slots
        .get(instruction_index)
        .copied()
        .unwrap_or(false)
}

fn lite_exception_shape_label(has_exception: bool) -> Vec<String> {
    if has_exception {
        vec!["exception".to_string()]
    } else {
        vec!["no_exception".to_string()]
    }
}

fn lite_presence_shape_label(present: bool) -> Vec<String> {
    if present {
        vec!["present".to_string()]
    } else {
        vec!["missing".to_string()]
    }
}

fn lite_metadata_impls(case: &LiteLogHealthCase) -> Vec<String> {
    vec!["spike".to_string(), case.impl_ref.clone()]
}

fn lite_testcase_impls(case: &LiteLogHealthCase) -> Vec<String> {
    vec!["spike".to_string(), case.impl_ref.clone()]
}

fn lite_case_seed(case: &LiteLogHealthCase, attempt: u64) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    case.id.hash(&mut hasher);
    attempt.hash(&mut hasher);
    hasher.finish()
}

fn lite_effective_float_extension(case: &LiteLogHealthCase) -> FloatExtension {
    let impls = lite_metadata_impls(case);
    let plugin_manager = PluginManager::global();
    clamp_float_extension(
        case.requested_float_extension,
        max_float_extension_for_extensions(
            &plugin_manager.extensions_for_impls(&impls, case.isa_base),
        ),
    )
}

fn lite_exec_timeout_secs() -> u64 {
    env::var(EXEC_TIMEOUT_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(180)
}

fn generate_lite_metadata_constrained_testcase(
    case: &LiteLogHealthCase,
    config: TestCaseConfig,
) -> Result<GeneratedTestCase, String> {
    let target_extensions = extensions_for_config(&lite_metadata_impls(case), &config)
        .map_err(|err| format!("extension resolution failed for {}: {err}", case.id))?;
    generate_random_testcase_with_extensions(&lite_testcase_impls(case), config, target_extensions)
        .map_err(|err| format!("testcase generation failed for {}: {err}", case.id))
}

fn append_lite_single_hart_signal_probe(
    block: &mut StructuredTestInstructions,
    isa_base: ISABase,
) -> Result<(), String> {
    let ctx = match isa_base {
        ISABase::Rv32 => IsaDecodeContext::RV32,
        ISABase::Rv64 => IsaDecodeContext::RV64,
    };

    // Probe a load/store at the start of the shared testcase memory region.
    // We construct a typed `LoadImmediate { purpose: BaseAddress, immediate: 0 }`:
    // the engine's runtime relocation pass adds the implementation's shared
    // `mem_start` to every BaseAddress-tagged LI before rendering, matching the
    // pipeline used by `random_diff` / `replay`.
    let rendered = "li x13, 19".to_string();
    block.push_instruction_to_hart(
        0,
        RiscvInstruction::decode_asm_with_context(&rendered, ctx)
            .map_err(|err| format!("failed to parse lite signal probe `{rendered}`: {err}"))?,
    );

    block.push_instruction_to_hart(
        0,
        RiscvInstruction::from(PseudoInstruction::LoadImmediate {
            rd: IntegerRegister::new(9).expect("x9 is a valid integer register"),
            immediate: XLenImmediate::from_i64_for(isa_profile(isa_base), 0)
                .expect("base-address immediate must fit active xlen"),
            purpose: LoadImmediatePurpose::BaseAddress,
        }),
    );

    let memory_ops = match isa_base {
        ISABase::Rv32 => vec![
            "sw x13, 0(x9)".to_string(),
            "lw x14, 0(x9)".to_string(),
            "ecall".to_string(),
            "addi x7, x7, 1".to_string(),
        ],
        ISABase::Rv64 => vec![
            "sd x13, 0(x9)".to_string(),
            "ld x14, 0(x9)".to_string(),
            "ecall".to_string(),
            "addi x7, x7, 1".to_string(),
        ],
    };

    for rendered in memory_ops {
        block.push_instruction_to_hart(
            0,
            RiscvInstruction::decode_asm_with_context(&rendered, ctx)
                .map_err(|err| format!("failed to parse lite signal probe `{rendered}`: {err}"))?,
        );
    }

    Ok(())
}

fn append_lite_dual_hart_signal_probe(
    block: &mut StructuredTestInstructions,
    isa_base: ISABase,
) -> Result<(), String> {
    let ctx = match isa_base {
        ISABase::Rv32 => IsaDecodeContext::RV32,
        ISABase::Rv64 => IsaDecodeContext::RV64,
    };
    let second_hart_offset = match isa_base {
        ISABase::Rv32 => 8,
        ISABase::Rv64 => 16,
    };

    for (hart_idx, value, offset) in [(0usize, 19, 0u32), (1usize, 37, second_hart_offset)] {
        let init = format!("li x13, {value}");
        block.push_instruction_to_hart(
            hart_idx,
            RiscvInstruction::decode_asm_with_context(&init, ctx)
                .map_err(|err| format!("failed to parse lite dual-hart probe `{init}`: {err}"))?,
        );
        block.push_instruction_to_hart(
            hart_idx,
            RiscvInstruction::from(PseudoInstruction::LoadImmediate {
                rd: IntegerRegister::new(9).expect("x9 is a valid integer register"),
                immediate: XLenImmediate::from_i64_for(isa_profile(isa_base), 0)
                    .expect("base-address immediate must fit active xlen"),
                purpose: LoadImmediatePurpose::BaseAddress,
            }),
        );

        let ops = match isa_base {
            ISABase::Rv32 => vec![
                format!("sw x13, {offset}(x9)"),
                format!("lw x14, {offset}(x9)"),
                "xor x15, x14, x13".to_string(),
                "addi x16, x15, 1".to_string(),
            ],
            ISABase::Rv64 => vec![
                format!("sd x13, {offset}(x9)"),
                format!("ld x14, {offset}(x9)"),
                "xor x15, x14, x13".to_string(),
                "addi x16, x15, 1".to_string(),
            ],
        };

        for rendered in ops {
            block.push_instruction_to_hart(
                hart_idx,
                RiscvInstruction::decode_asm_with_context(&rendered, ctx).map_err(|err| {
                    format!("failed to parse lite dual-hart probe `{rendered}`: {err}")
                })?,
            );
        }
    }

    Ok(())
}

fn lite_diff_health_sync_config() -> MulticoreSyncConfig {
    MulticoreSyncConfig {
        interaction_slot_ratio: 0.0,
        shared_address_ratio: 0.0,
        prefer_focused_sequences: false,
        ..MulticoreSyncConfig::default()
    }
}

fn build_lite_log_health_testcase_with_sample(
    case: &LiteLogHealthCase,
    sample_index: usize,
) -> Result<(u64, GeneratedTestCase), String> {
    let mem_size = match case.isa_base {
        ISABase::Rv64 => 32,
        ISABase::Rv32 => 64,
    };

    let seed = lite_case_seed(case, sample_index as u64 * 16);
    let mut testcase = generate_lite_metadata_constrained_testcase(
        case,
        TestCaseConfig {
            isa_base: case.isa_base,
            extension_insts_count: 0,
            extension_inst_scaling: None,
            mem_size,
            float_extension: lite_effective_float_extension(case),
            unaligned_access_required: Some(case.allow_unaligned),
            random_config: RandomConfig {
                memory_access_config: MemoryAccessConfig {
                    memory_base_address_range: MemoryAccessConfig::default()
                        .memory_base_address_range,
                    memory_offset_immediate_range: (0, 0),
                    memory_base_register_number_range: (8, 12),
                },
                register_operand_config: RegisterOperandConfig {
                    integer_register_number_range: (13, 20),
                    floating_point_register_number_range: (13, 20),
                    vector_register_number_range: (13, 20),
                },
                csr_selection_config: CsrSelectionConfig {
                    machine_csr_operands_enabled: false,
                    supervisor_csr_operands_enabled: false,
                    hypervisor_csr_operands_enabled: false,
                    virtual_supervisor_csr_operands_enabled: false,
                    rnmi_csr_operands_enabled: false,
                    floating_csr_operands_enabled: false,
                    vector_csr_operands_enabled: false,
                    debug_csr_operands_enabled: false,
                },
                isa_profile: isa_profile_for_float(
                    case.isa_base,
                    lite_effective_float_extension(case),
                ),
                ..RandomConfig::default()
            },
            seed: Some(seed),
            hart_config: match case.hart_count {
                1 => HartConfig::Single,
                2 => HartConfig::Dual {
                    sync: lite_diff_health_sync_config(),
                },
                other => panic!("unsupported test hart count: {other}"),
            },
        },
    )?;

    let mut test_program = StructuredTestInstructions::default();
    match case.hart_count {
        1 => append_lite_single_hart_signal_probe(&mut test_program, case.isa_base)?,
        2 => append_lite_dual_hart_signal_probe(&mut test_program, case.isa_base)?,
        other => panic!("unsupported test hart count: {other}"),
    }
    testcase.set_test_program(test_program);

    Ok((seed, testcase))
}

fn build_lite_log_health_testcase(
    case: &LiteLogHealthCase,
) -> Result<(u64, GeneratedTestCase), String> {
    build_lite_log_health_testcase_with_sample(case, 0)
}

fn lite_alignment_mode(testcase: &GeneratedTestCase) -> Result<AlignmentMode, String> {
    match testcase.config.unaligned_access_required {
        Some(true) => Ok(AlignmentMode::AllowUnaligned),
        Some(false) => Ok(AlignmentMode::RequireAligned),
        None => Err("unaligned_access_required is not set".to_string()),
    }
}

fn run_lite_streaming_cancel_probe(
    case: &LiteLogHealthCase,
    testcase: &GeneratedTestCase,
    case_dir: &Path,
) -> Vec<LiteStreamingCancelProbeSummary> {
    lite_testcase_impls(case)
        .into_iter()
        .map(|impl_ref| run_lite_streaming_cancel_probe_impl(&impl_ref, testcase, case, case_dir))
        .collect()
}

fn run_lite_streaming_cancel_probe_impl(
    impl_ref: &str,
    testcase: &GeneratedTestCase,
    case: &LiteLogHealthCase,
    case_dir: &Path,
) -> LiteStreamingCancelProbeSummary {
    let started = Instant::now();
    let mut summary = LiteStreamingCancelProbeSummary {
        impl_ref: impl_ref.to_string(),
        event_count: 0,
        instruction_event_count: 0,
        source_mapped_instruction_event_count: 0,
        framework_instruction_event_count: 0,
        completed_event_count: 0,
        failed_event_count: 0,
        cancelled_after_first_instruction: false,
        elapsed_ms: 0,
        failed_checks: Vec::new(),
        error: None,
    };

    let alignment_mode = match lite_alignment_mode(testcase) {
        Ok(mode) => mode,
        Err(error) => {
            summary.error = Some(error);
            summary
                .failed_checks
                .push("streaming probe setup failed".to_string());
            return summary;
        }
    };
    let Some(per_hart_typed) = testcase.combined_typed_sequences_per_hart_of(impl_ref) else {
        summary.error = Some(format!("missing typed sequences for {impl_ref}"));
        summary
            .failed_checks
            .push("streaming probe setup failed".to_string());
        return summary;
    };

    let probe_dir = case_dir.join("streaming_cancel_probe").join(impl_ref);
    if let Err(error) = fs::create_dir_all(&probe_dir) {
        summary.error = Some(format!(
            "failed to create streaming probe dir {}: {error}",
            probe_dir.display()
        ));
        summary
            .failed_checks
            .push("streaming probe setup failed".to_string());
        return summary;
    }

    let hart_inputs = per_hart_typed
        .iter()
        .map(|(hart_id, seqs)| (*hart_id, seqs.as_slice()))
        .collect::<Vec<_>>();
    let request = ExecuteRequest {
        run_root: &probe_dir,
        isa_base: testcase.config.isa_base,
        hart_inputs: &hart_inputs,
        extension_override: Some(&testcase.extensions),
        float_mode_override: None,
        alignment_mode,
        coverage_mode: case.coverage_mode,
    };

    let plugin_manager = PluginManager::global();
    let result = plugin_manager.execute_streaming_with_request(impl_ref, request, &mut |event| {
        summary.event_count += 1;
        match event {
            InstructionEvent::Instruction { instruction, .. } => {
                summary.instruction_event_count += 1;
                if instruction.trace_inst_id.is_some() {
                    summary.source_mapped_instruction_event_count += 1;
                    summary.cancelled_after_first_instruction = true;
                    false
                } else {
                    summary.framework_instruction_event_count += 1;
                    if summary.framework_instruction_event_count
                        >= LITE_LOG_HEALTH_CANCEL_PROBE_FRAMEWORK_PREFIX_LIMIT
                    {
                        summary.error = Some(format!(
                            "streaming cancel probe saw {} framework-only Instruction event(s) without reaching a source-mapped instruction",
                            summary.framework_instruction_event_count
                        ));
                        false
                    } else {
                        true
                    }
                }
            }
            InstructionEvent::Completed { .. } => {
                summary.completed_event_count += 1;
                false
            }
            InstructionEvent::Failed { error } => {
                summary.failed_event_count += 1;
                summary.error = Some(error.to_string());
                false
            }
        }
    });
    summary.elapsed_ms = started.elapsed().as_millis();

    if let Err(error) = result {
        summary.error = Some(error.to_string());
    }
    if summary.error.is_some() {
        summary
            .failed_checks
            .push("streaming execution returned an error".to_string());
    }
    if summary.failed_event_count > 0 {
        summary
            .failed_checks
            .push("streaming emitted a Failed event".to_string());
    }
    if summary.instruction_event_count == 0 {
        summary
            .failed_checks
            .push("streaming emitted no Instruction before completion/cancel".to_string());
    }
    if summary.source_mapped_instruction_event_count == 0 {
        summary.failed_checks.push(
            "streaming emitted no source-mapped Instruction before completion/cancel".to_string(),
        );
    }
    if !summary.cancelled_after_first_instruction {
        summary
            .failed_checks
            .push("streaming cancel probe never requested cancellation".to_string());
    }
    if summary.completed_event_count > 0 {
        summary.failed_checks.push(
            "streaming emitted Completed after the callback requested cancellation".to_string(),
        );
    }
    if summary.elapsed_ms >= u128::from(lite_exec_timeout_secs()) * 1000 {
        summary.failed_checks.push(format!(
            "streaming cancel probe elapsed {}ms, at or above timeout",
            summary.elapsed_ms
        ));
    }

    summary
}

fn lite_add_full_streaming_event_checks(summary: &mut LiteStreamingFullProbeSummary) {
    if summary.error.is_some() {
        summary
            .failed_checks
            .push("full streaming execution returned an error".to_string());
    }
    if summary.failed_event_count > 0 {
        summary
            .failed_checks
            .push("full streaming emitted a Failed event".to_string());
    }
    if summary.event_count == 0 {
        summary
            .failed_checks
            .push("full streaming emitted no events".to_string());
    }
    if summary.instruction_event_count == 0 {
        summary
            .failed_checks
            .push("full streaming emitted no Instruction events".to_string());
    }
    if summary.source_mapped_instruction_event_count == 0 {
        summary
            .failed_checks
            .push("full streaming emitted no source-mapped Instruction events".to_string());
    }
    if summary.completed_event_count == 0 {
        summary
            .failed_checks
            .push("full streaming did not emit Completed".to_string());
    }
    if summary.completed_event_count > 1 {
        summary.failed_checks.push(format!(
            "full streaming emitted Completed {} times",
            summary.completed_event_count
        ));
    }
    if summary.elapsed_ms >= u128::from(lite_exec_timeout_secs()) * 1000 {
        summary.failed_checks.push(format!(
            "full streaming elapsed {}ms, at or above timeout",
            summary.elapsed_ms
        ));
    }
}

fn lite_streaming_probe_hart_ids(
    case: &LiteLogHealthCase,
    streaming_output: &ExecutionContextOutput,
    reference_output: &ExecutionContextOutput,
) -> Vec<u32> {
    let mut hart_ids = BTreeSet::new();
    hart_ids.extend(
        streaming_output
            .logical_view
            .iter()
            .map(|instruction| instruction.hart_id),
    );
    hart_ids.extend(
        reference_output
            .logical_view
            .iter()
            .map(|instruction| instruction.hart_id),
    );
    if hart_ids.is_empty() {
        hart_ids.extend(0..case.hart_count);
    }
    hart_ids.into_iter().collect()
}

fn run_lite_streaming_full_probe(
    case: &LiteLogHealthCase,
    testcase: &GeneratedTestCase,
    case_dir: &Path,
    reference_outputs: &HashMap<String, ExecutionContextOutput>,
) -> Vec<LiteStreamingFullProbeSummary> {
    lite_testcase_impls(case)
        .into_iter()
        .map(|impl_ref| {
            run_lite_streaming_full_probe_impl(
                &impl_ref,
                testcase,
                case,
                case_dir,
                reference_outputs.get(&impl_ref),
            )
        })
        .collect()
}

fn run_lite_streaming_full_probe_impl(
    impl_ref: &str,
    testcase: &GeneratedTestCase,
    case: &LiteLogHealthCase,
    case_dir: &Path,
    reference_output: Option<&ExecutionContextOutput>,
) -> LiteStreamingFullProbeSummary {
    let started = Instant::now();
    let mut summary = LiteStreamingFullProbeSummary {
        impl_ref: impl_ref.to_string(),
        event_count: 0,
        instruction_event_count: 0,
        source_mapped_instruction_event_count: 0,
        framework_instruction_event_count: 0,
        completed_event_count: 0,
        failed_event_count: 0,
        elapsed_ms: 0,
        harts: Vec::new(),
        failed_checks: Vec::new(),
        error: None,
    };

    let Some(reference_output) = reference_output else {
        summary.error = Some(format!("missing reference execution output for {impl_ref}"));
        summary
            .failed_checks
            .push("full streaming probe setup failed".to_string());
        return summary;
    };
    let alignment_mode = match lite_alignment_mode(testcase) {
        Ok(mode) => mode,
        Err(error) => {
            summary.error = Some(error);
            summary
                .failed_checks
                .push("full streaming probe setup failed".to_string());
            return summary;
        }
    };
    let Some(per_hart_typed) = testcase.combined_typed_sequences_per_hart_of(impl_ref) else {
        summary.error = Some(format!("missing typed sequences for {impl_ref}"));
        summary
            .failed_checks
            .push("full streaming probe setup failed".to_string());
        return summary;
    };

    let probe_parent = case_dir.join("streaming_full_probe");
    let probe_dir = probe_parent.join(impl_ref);
    if let Err(error) = fs::create_dir_all(&probe_dir) {
        summary.error = Some(format!(
            "failed to create full streaming probe dir {}: {error}",
            probe_dir.display()
        ));
        summary
            .failed_checks
            .push("full streaming probe setup failed".to_string());
        return summary;
    }

    let hart_inputs = per_hart_typed
        .iter()
        .map(|(hart_id, seqs)| (*hart_id, seqs.as_slice()))
        .collect::<Vec<_>>();
    let request = ExecuteRequest {
        run_root: &probe_dir,
        isa_base: testcase.config.isa_base,
        hart_inputs: &hart_inputs,
        extension_override: Some(&testcase.extensions),
        float_mode_override: None,
        alignment_mode,
        coverage_mode: case.coverage_mode,
    };

    let plugin_manager = PluginManager::global();
    let mut collector = StreamingCollector::new();
    let streaming_result =
        plugin_manager.execute_streaming_with_request(impl_ref, request, &mut |event| {
            summary.event_count += 1;
            match &event {
                InstructionEvent::Instruction { instruction, .. } => {
                    summary.instruction_event_count += 1;
                    if instruction.trace_inst_id.is_some() {
                        summary.source_mapped_instruction_event_count += 1;
                    } else {
                        summary.framework_instruction_event_count += 1;
                    }
                }
                InstructionEvent::Completed { .. } => {
                    summary.completed_event_count += 1;
                }
                InstructionEvent::Failed { error } => {
                    summary.failed_event_count += 1;
                    summary.error = Some(error.to_string());
                }
            }
            collector.push(event)
        });
    summary.elapsed_ms = started.elapsed().as_millis();

    let mut streaming_output = None;
    let mut streaming_context = None;
    match streaming_result {
        Err(error) => {
            summary.error = Some(error.to_string());
        }
        Ok(()) => match collector.finish() {
            Err(error) => {
                summary.error = Some(error.to_string());
            }
            Ok((per_hart, perf)) => {
                let mut output = ExecutionOutput {
                    per_hart,
                    core_id: impl_ref.to_string(),
                    isa_base: testcase.config.isa_base,
                    perf,
                };
                if output.per_hart.is_empty() {
                    output.per_hart = hart_inputs
                        .iter()
                        .map(|input| (input.0, HartExecutionOutput::empty(input.0)))
                        .collect();
                }

                match request.finalize_execution_output(output) {
                    Err(error) => {
                        summary.error = Some(error.to_string());
                    }
                    Ok(mut output) => match testcase.mem_range.get(impl_ref).copied() {
                        None => {
                            summary.error = Some(format!(
                                "missing memory range for implementation {impl_ref}"
                            ));
                        }
                        Some(range) => {
                            if let Err(error) = output.normalize(
                                range,
                                &testcase.config.random_config.register_operand_config,
                                testcase
                                    .config
                                    .random_config
                                    .memory_access_config
                                    .memory_base_register_number_range,
                            ) {
                                summary.error = Some(error.to_string());
                            } else {
                                streaming_output = Some(output.clone());
                                match build_execution_context_output_with_weak_fallback(
                                    output,
                                    range,
                                    &per_hart_typed,
                                ) {
                                    Ok(context) => {
                                        streaming_context = Some(context);
                                    }
                                    Err(error) => {
                                        summary.error = Some(error.to_string());
                                    }
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    if let Some(streaming_output) = streaming_output {
        let impl_ref_string = impl_ref.to_string();
        let mut outputs = HashMap::new();
        outputs.insert(impl_ref_string.clone(), streaming_output);
        if let Err(error) = render_execution_reports(&probe_parent, &outputs, &[impl_ref_string]) {
            summary.error = Some(error);
        }
    }

    if let Some(streaming_context) = streaming_context {
        for hart_id in lite_streaming_probe_hart_ids(case, &streaming_context, reference_output) {
            let streaming_view = aggregate_as_hart_context_output(&streaming_context, hart_id);
            let reference_view = aggregate_as_hart_context_output(reference_output, hart_id);
            summary.harts.push(lite_summarize_hart(
                hart_id,
                &streaming_view,
                &reference_view,
            ));
        }
    }

    lite_add_full_streaming_event_checks(&mut summary);
    summary
}

fn lite_summarize_hart(
    hart_id: u32,
    target: &HartLogView,
    spike: &HartLogView,
) -> LiteHartLogHealthSummary {
    let total_slot_count = target.instruction_count.max(spike.instruction_count);
    let target_active_slot_count = (0..target.instruction_count)
        .filter(|idx| lite_slot_active(target, *idx))
        .count();
    let spike_active_slot_count = (0..spike.instruction_count)
        .filter(|idx| lite_slot_active(spike, *idx))
        .count();
    let mut compared_slot_count = 0usize;
    let mut shape_matched_slot_count = 0usize;
    let mut failed_slot_count = 0usize;
    let mut register_matches = 0usize;
    let mut memory_matches = 0usize;
    let mut exception_matches = 0usize;
    let mut sample_mismatches: Vec<LiteSlotMismatch> = Vec::new();
    let mut register_mismatch_count = 0usize;
    let mut memory_mismatch_count = 0usize;
    let mut exception_mismatch_count = 0usize;
    let mut instruction_presence_mismatch_count = 0usize;
    let tail_probe_start = total_slot_count.saturating_sub(LITE_LOG_HEALTH_TAIL_PROBE_SLOT_COUNT);
    let mut tail_probe_slot_count = 0usize;
    let mut tail_probe_shape_matched_slot_count = 0usize;
    let mut tail_probe_failed_slot_count = 0usize;
    let mut tail_probe_inactive_slot_count = 0usize;
    const SAMPLE_LIMIT: usize = 24;

    for idx in 0..total_slot_count {
        compared_slot_count += 1;
        let target_present = idx < target.instruction_count;
        let spike_present = idx < spike.instruction_count;
        let target_regs = lite_register_name_set_at(target, idx);
        let spike_regs = lite_register_name_set_at(spike, idx);
        let target_addrs = lite_memory_address_set_at(target, idx);
        let spike_addrs = lite_memory_address_set_at(spike, idx);
        let target_exception = lite_exception_present(&target.exceptions, idx);
        let spike_exception = lite_exception_present(&spike.exceptions, idx);
        let presence_match = target_present == spike_present;
        let regs_match = target_regs == spike_regs;
        let mem_match = target_addrs == spike_addrs;
        let exception_match = target_exception == spike_exception;

        if presence_match {
            // Presence is folded into the per-slot shape match below, but not
            // reported as a ratio because equal logical lengths are expected.
        } else {
            instruction_presence_mismatch_count += 1;
            if sample_mismatches.len() < SAMPLE_LIMIT {
                sample_mismatches.push(LiteSlotMismatch {
                    instruction_index: idx,
                    kind: "instruction_presence".to_string(),
                    target: lite_presence_shape_label(target_present),
                    spike: lite_presence_shape_label(spike_present),
                });
            }
        }

        if regs_match {
            register_matches += 1;
        } else {
            register_mismatch_count += 1;
            if sample_mismatches.len() < SAMPLE_LIMIT {
                sample_mismatches.push(LiteSlotMismatch {
                    instruction_index: idx,
                    kind: "register_names".to_string(),
                    target: target_regs.into_iter().collect(),
                    spike: spike_regs.into_iter().collect(),
                });
            }
        }

        if mem_match {
            memory_matches += 1;
        } else {
            memory_mismatch_count += 1;
            if sample_mismatches.len() < SAMPLE_LIMIT {
                sample_mismatches.push(LiteSlotMismatch {
                    instruction_index: idx,
                    kind: "memory_addresses".to_string(),
                    target: target_addrs
                        .iter()
                        .map(|addr| format!("0x{addr:x}"))
                        .collect(),
                    spike: spike_addrs
                        .iter()
                        .map(|addr| format!("0x{addr:x}"))
                        .collect(),
                });
            }
        }

        if exception_match {
            exception_matches += 1;
        } else {
            exception_mismatch_count += 1;
            if sample_mismatches.len() < SAMPLE_LIMIT {
                sample_mismatches.push(LiteSlotMismatch {
                    instruction_index: idx,
                    kind: "exception_presence".to_string(),
                    target: lite_exception_shape_label(target_exception),
                    spike: lite_exception_shape_label(spike_exception),
                });
            }
        }

        if presence_match && regs_match && mem_match && exception_match {
            shape_matched_slot_count += 1;
        } else {
            failed_slot_count += 1;
        }

        if idx >= tail_probe_start {
            tail_probe_slot_count += 1;
            let target_observed = lite_slot_observed(target, idx);
            let spike_observed = lite_slot_observed(spike, idx);
            if target_observed == spike_observed {
                tail_probe_shape_matched_slot_count += 1;
            } else {
                tail_probe_failed_slot_count += 1;
                if !target_observed || !spike_observed {
                    tail_probe_inactive_slot_count += 1;
                }
                if sample_mismatches.len() < SAMPLE_LIMIT {
                    sample_mismatches.push(LiteSlotMismatch {
                        instruction_index: idx,
                        kind: "tail_probe_unobserved".to_string(),
                        target: vec![if target_observed {
                            "observed".to_string()
                        } else {
                            "unobserved".to_string()
                        }],
                        spike: vec![if spike_observed {
                            "observed".to_string()
                        } else {
                            "unobserved".to_string()
                        }],
                    });
                }
            }
        }
    }

    let shape_match_ratio = ratio_or_one(shape_matched_slot_count, compared_slot_count);
    let register_ratio = ratio_or_one(register_matches, compared_slot_count);
    let memory_ratio = ratio_or_one(memory_matches, compared_slot_count);
    let exception_ratio = ratio_or_one(exception_matches, compared_slot_count);
    let mut failed_checks = Vec::new();
    if target_active_slot_count == 0 {
        failed_checks.push("target hart log has zero active instruction slots".to_string());
    }
    if compared_slot_count == 0 {
        failed_checks.push("no instruction slots were compared".to_string());
    }
    if shape_match_ratio < LITE_LOG_HEALTH_SHAPE_MATCH_THRESHOLD {
        failed_checks.push(format!(
            "log shape match ratio {:.2}% is below {:.2}% ({} matched, {} failed, {} compared)",
            shape_match_ratio * 100.0,
            LITE_LOG_HEALTH_SHAPE_MATCH_THRESHOLD * 100.0,
            shape_matched_slot_count,
            failed_slot_count,
            compared_slot_count
        ));
    }
    if tail_probe_slot_count < LITE_LOG_HEALTH_TAIL_PROBE_SLOT_COUNT {
        failed_checks.push(format!(
            "tail log probe expected {} slots but only {} were present",
            LITE_LOG_HEALTH_TAIL_PROBE_SLOT_COUNT, tail_probe_slot_count
        ));
    }
    if tail_probe_failed_slot_count > 0 {
        failed_checks.push(format!(
            "tail log probe had {} unobserved slot(s), indicating possible streaming truncation or tail parse loss",
            tail_probe_failed_slot_count
        ));
    }
    if tail_probe_inactive_slot_count > 0 {
        failed_checks.push(format!(
            "tail log probe had {} inactive slot(s), indicating possible truncated or missing tail events",
            tail_probe_inactive_slot_count
        ));
    }

    LiteHartLogHealthSummary {
        hart_id,
        target_instruction_count: target.instruction_count,
        spike_instruction_count: spike.instruction_count,
        compared_slot_count,
        shape_matched_slot_count,
        failed_slot_count,
        shape_match_ratio,
        shape_match_threshold: LITE_LOG_HEALTH_SHAPE_MATCH_THRESHOLD,
        register_slot_match_ratio: register_ratio,
        memory_slot_match_ratio: memory_ratio,
        exception_slot_match_ratio: exception_ratio,
        target_active_slot_count,
        spike_active_slot_count,
        register_mismatch_count,
        memory_mismatch_count,
        exception_mismatch_count,
        instruction_presence_mismatch_count,
        tail_probe_slot_count,
        tail_probe_shape_matched_slot_count,
        tail_probe_failed_slot_count,
        tail_probe_inactive_slot_count,
        sample_mismatches,
        failed_checks,
    }
}

fn run_lite_log_health_case(
    case: &LiteLogHealthCase,
    run_root: &Path,
) -> Result<LiteLogHealthCaseSummary, String> {
    let case_dir = run_root.join(&case.id);
    fs::create_dir_all(&case_dir)
        .map_err(|e| format!("failed to create case dir {}: {e}", case_dir.display()))?;

    let timeout_secs = lite_exec_timeout_secs();
    let (seed, testcase) = build_lite_log_health_testcase(case)?;
    write_generated_testcase(&case_dir.join("generated_testcase.json"), &testcase)?;

    let outputs = execute(&testcase, &case_dir, case.coverage_mode).map_err(|e| {
        format!(
            "single-hart log correctness execution failed for {}: {e}",
            case.id
        )
    })?;
    let context_outputs = build_context_outputs_for_testcase(&testcase, &outputs)?;
    render_execution_reports(&case_dir, &outputs, &lite_testcase_impls(case))?;
    let streaming_cancel_probes = run_lite_streaming_cancel_probe(case, &testcase, &case_dir);
    let streaming_full_probes =
        run_lite_streaming_full_probe(case, &testcase, &case_dir, &context_outputs);

    let target = context_outputs
        .get(&case.impl_ref)
        .ok_or_else(|| format!("missing target output for {}", case.impl_ref))?;
    let spike = context_outputs
        .get("spike")
        .ok_or_else(|| "missing spike output".to_string())?;

    let hart_ids: Vec<u32> = if case.hart_count == 2 {
        vec![0, 1]
    } else {
        vec![0]
    };

    let mut harts = Vec::new();
    for hart_id in hart_ids {
        let target_view = aggregate_as_hart_context_output(target, hart_id);
        let spike_view = aggregate_as_hart_context_output(spike, hart_id);
        harts.push(lite_summarize_hart(hart_id, &target_view, &spike_view));
    }

    if let Some(target_streaming_probe) = streaming_full_probes
        .iter()
        .find(|probe| probe.impl_ref == case.impl_ref && probe.error.is_none())
    {
        harts = target_streaming_probe.harts.clone();
    }

    let status = if harts.iter().all(|h| h.failed_checks.is_empty())
        && streaming_cancel_probes
            .iter()
            .all(|probe| probe.failed_checks.is_empty())
        && streaming_full_probes.iter().all(|probe| {
            probe.failed_checks.is_empty()
                && probe.harts.iter().all(|hart| hart.failed_checks.is_empty())
        }) {
        "pass".to_string()
    } else {
        "fail".to_string()
    };

    Ok(LiteLogHealthCaseSummary {
        status,
        id: case.id.clone(),
        impl_ref: case.impl_ref.clone(),
        plugin_source_path: case.plugin_source_path.clone(),
        isa_base: format!("{:?}", case.isa_base),
        requested_float_extension: format!("{:?}", case.requested_float_extension),
        effective_float_extension: format!("{:?}", lite_effective_float_extension(case)),
        coverage_mode: format!("{:?}", case.coverage_mode),
        allow_unaligned: case.allow_unaligned,
        hart_count: case.hart_count,
        testcase_seed: seed,
        timeout_secs,
        harts,
        streaming_cancel_probes,
        streaming_full_probes,
        error: None,
    })
}

// 这些 case 是 cx-riscv-cores 仓库本身的构建/二进制限制，不是 plugin 日志层面的问题；
// 测试不应据此判定 fail。每条都附了原因。
//
// 加新条目时务必在条目里说明 *为什么* 这是基础设施问题而不是真 bug，
// 不然以后没人敢删。
//
// xiangshan 双核之前因为 NEMU difftest 模型在共享内存原子操作（amocas.d 等）上跟
// XiangShan RTL 不一致而 abort；通过让 lite 测试用 0/0 的 MulticoreSyncConfig
// （两个 hart 完全独立，不共享地址）已经规避掉了，所以不再需要 skip。
fn lite_log_health_error_summary(
    case: &LiteLogHealthCase,
    error: String,
) -> LiteLogHealthCaseSummary {
    LiteLogHealthCaseSummary {
        status: "fail".to_string(),
        id: case.id.clone(),
        impl_ref: case.impl_ref.clone(),
        plugin_source_path: case.plugin_source_path.clone(),
        isa_base: format!("{:?}", case.isa_base),
        requested_float_extension: format!("{:?}", case.requested_float_extension),
        effective_float_extension: format!("{:?}", lite_effective_float_extension(case)),
        coverage_mode: format!("{:?}", case.coverage_mode),
        allow_unaligned: case.allow_unaligned,
        hart_count: case.hart_count,
        testcase_seed: 0,
        timeout_secs: lite_exec_timeout_secs(),
        harts: Vec::new(),
        streaming_cancel_probes: Vec::new(),
        streaming_full_probes: Vec::new(),
        error: Some(error),
    }
}

fn provider_pair_effective_float_extension(case: &ProviderBinCase) -> FloatExtension {
    let impls = vec!["spike".to_string(), case.impl_ref.clone()];
    let plugin_manager = PluginManager::global();
    clamp_float_extension(
        case.requested_float_extension,
        max_float_extension_for_extensions(
            &plugin_manager.extensions_for_impls(&impls, case.isa_base),
        ),
    )
}

fn preferred_diff_health_isa_score(case: &ProviderBinCase) -> i32 {
    match case.impl_ref.as_str() {
        "boom" | "cva6" | "openc906" | "openc910" | "rocket" | "xiangshan" => {
            if case.isa_base == ISABase::Rv64 {
                40
            } else {
                0
            }
        }
        _ => {
            if case.isa_base == ISABase::Rv32 {
                40
            } else {
                0
            }
        }
    }
}

fn provider_diff_health_case_score(
    case: &ProviderBinCase,
    hart_count: u32,
    allowed_impls: &[&str],
) -> Option<i32> {
    if case.hart_count != hart_count || case.coverage_mode != CoverageMode::Off {
        return None;
    }
    if !allowed_impls.contains(&case.impl_ref.as_str()) {
        return None;
    }

    let mut score = preferred_diff_health_isa_score(case);
    score += match case.requested_float_extension {
        FloatExtension::None => 30,
        FloatExtension::F => 20,
        FloatExtension::FD => 10,
    };
    if !case.allow_unaligned {
        score += 20;
    }
    Some(score)
}

fn collect_provider_diff_health_cases(
    cases: &[ProviderBinCase],
    hart_count: u32,
    allowed_impls: &[&str],
) -> Vec<ProviderBinCase> {
    let mut selected = Vec::new();

    for impl_ref in allowed_impls {
        let mut best_case: Option<&ProviderBinCase> = None;
        let mut best_score = i32::MIN;

        for case in cases.iter().filter(|case| case.impl_ref == *impl_ref) {
            let Some(score) = provider_diff_health_case_score(case, hart_count, allowed_impls)
            else {
                continue;
            };

            let should_replace = match best_case {
                None => true,
                Some(current) => {
                    score > best_score || (score == best_score && case.id < current.id)
                }
            };
            if should_replace {
                best_case = Some(case);
                best_score = score;
            }
        }

        if let Some(case) = best_case {
            selected.push(case.clone());
        }
    }

    selected
}

fn diff_health_runs() -> usize {
    env::var(DIFF_HEALTH_RUNS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

fn diff_health_case_concurrency() -> usize {
    env::var(DIFF_HEALTH_CASE_CONCURRENCY_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

fn diff_health_max_parallel_budget() -> usize {
    env::var(DIFF_HEALTH_MAX_PARALLEL_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

fn diff_health_effective_case_concurrency() -> usize {
    diff_health_case_concurrency().min(diff_health_max_parallel_budget())
}

fn diff_health_case_parallelism(case_count: usize) -> usize {
    let case_concurrency = diff_health_effective_case_concurrency();
    let budget = diff_health_max_parallel_budget();
    (budget / case_concurrency).max(1).min(case_count.max(1))
}

fn configure_provider_diff_health_timeout(cases: &[ProviderBinCase]) -> Option<EnvVarGuard> {
    if env::var(EXEC_TIMEOUT_ENV).is_ok() {
        return None;
    }

    let timeout_secs = cases
        .iter()
        .map(default_timeout_secs)
        .max()
        .unwrap_or_else(lite_exec_timeout_secs);
    Some(set_env_var(EXEC_TIMEOUT_ENV, timeout_secs.to_string()))
}

fn configure_selected_provider_case_binaries(cases: &[ProviderBinCase]) -> Vec<EnvVarGuard> {
    let mut bindings: BTreeMap<String, PathBuf> = BTreeMap::new();
    for case in cases {
        let env_var = target_binary_env_var(case).to_string();
        match bindings.get(&env_var) {
            Some(existing) if *existing != case.binary_path => {
                panic!(
                    "provider diff health selected conflicting binaries for {}: {} vs {}",
                    env_var,
                    existing.display(),
                    case.binary_path.display()
                );
            }
            Some(_) => {}
            None => {
                bindings.insert(env_var, case.binary_path.clone());
            }
        }
    }

    bindings
        .into_iter()
        .map(|(key, path)| set_env_var(key, path.to_string_lossy()))
        .collect()
}

fn missing_expected_provider_impls(
    cases: &[ProviderBinCase],
    expected_impls: &[&str],
) -> Vec<String> {
    let present: BTreeSet<&str> = cases.iter().map(|case| case.impl_ref.as_str()).collect();
    expected_impls
        .iter()
        .copied()
        .filter(|impl_ref| !present.contains(impl_ref))
        .map(ToOwned::to_owned)
        .collect()
}

fn lite_case_from_provider_case(case: &ProviderBinCase) -> LiteLogHealthCase {
    LiteLogHealthCase {
        id: case.id.clone(),
        impl_ref: case.impl_ref.clone(),
        plugin_source_path: case.binary_path.display().to_string(),
        isa_base: case.isa_base,
        requested_float_extension: case.requested_float_extension,
        coverage_mode: case.coverage_mode,
        allow_unaligned: case.allow_unaligned,
        hart_count: case.hart_count,
    }
}

fn lite_log_health_failure_messages(summary: &LiteLogHealthCaseSummary) -> Vec<String> {
    let mut failures = Vec::new();
    if let Some(error) = &summary.error {
        failures.push(error.clone());
        return failures;
    }

    for hart in &summary.harts {
        if !hart.failed_checks.is_empty() {
            failures.push(format!(
                "hart{}: {}",
                hart.hart_id,
                hart.failed_checks.join("; ")
            ));
        }
    }
    for probe in &summary.streaming_cancel_probes {
        if !probe.failed_checks.is_empty() {
            failures.push(format!(
                "{} streaming cancel probe: {}",
                probe.impl_ref,
                probe.failed_checks.join("; ")
            ));
        }
    }
    for probe in &summary.streaming_full_probes {
        if !probe.failed_checks.is_empty() {
            failures.push(format!(
                "{} full streaming probe: {}",
                probe.impl_ref,
                probe.failed_checks.join("; ")
            ));
        }
        for hart in &probe.harts {
            if !hart.failed_checks.is_empty() {
                failures.push(format!(
                    "{} full streaming probe hart{}: {}",
                    probe.impl_ref,
                    hart.hart_id,
                    hart.failed_checks.join("; ")
                ));
            }
        }
    }

    failures
}

fn run_provider_diff_health_sample(
    case: &ProviderBinCase,
    case_dir: &Path,
    sample_index: usize,
    suite_timeout_secs: u64,
) -> ProviderDiffHealthRunSummary {
    let run_dir = case_dir.join(format!("run_{sample_index:04}"));
    let started = Instant::now();
    let mut summary = ProviderDiffHealthRunSummary {
        sample_index,
        seed: 0,
        status: "fail".to_string(),
        elapsed_ms: 0,
        exception_removal_rounds: 0,
        write_removal_rounds: 0,
        output_root: run_dir.display().to_string(),
        bug_category: None,
        error: None,
    };

    if let Err(error) = fs::create_dir_all(&run_dir) {
        summary.elapsed_ms = started.elapsed().as_millis();
        summary.error = Some(format!(
            "failed to create diff health run dir {}: {error}",
            run_dir.display()
        ));
        return summary;
    }

    let lite_case = lite_case_from_provider_case(case);
    let (seed, testcase) =
        match build_lite_log_health_testcase_with_sample(&lite_case, sample_index) {
            Ok(value) => value,
            Err(error) => {
                summary.elapsed_ms = started.elapsed().as_millis();
                summary.error = Some(error);
                return summary;
            }
        };
    summary.seed = seed;
    if let Err(error) =
        write_generated_testcase(&run_dir.join("generated_testcase.json"), &testcase)
    {
        summary.elapsed_ms = started.elapsed().as_millis();
        summary.error = Some(error);
        return summary;
    }

    let settings = DiffAnalysisSettings {
        testcase_config: testcase.config.clone(),
        coverage_mode: case.coverage_mode,
        execution_timeout_seconds: suite_timeout_secs,
        max_iterations: Some(DIFF_HEALTH_MAX_REDUCTION_ITERATIONS),
        enable_reduction: true,
        history_min_test_threshold: None,
        emit_execution_output_json: false,
        emit_execution_report_md: false,
        cleanup_successful_iteration_artifacts: false,
        cleanup_successful_diff_run: false,
        verify_write_with_probes: true,
        write_probe_max_registers: 8,
        write_probe_max_memory_bytes: 16,
        dedup: DiffDedupSettings {
            enabled: false,
            database_path: None,
        },
        generation_extensions_override: None,
        execution_extensions_override: None,
    };
    let core_ids = vec!["spike".to_string(), case.impl_ref.clone()];
    let diff_config = settings.with_run_root(run_dir.clone());

    match run_diff_analysis_with_testcase(&core_ids, diff_config, testcase) {
        Ok(result) => {
            summary.status = "pass".to_string();
            summary.exception_removal_rounds = result.exception_removal_rounds;
            summary.write_removal_rounds = result.write_removal_rounds;
            summary.output_root = result.output_root.display().to_string();
        }
        Err(error @ DiffAnalysisError::IterationLimitReached(_)) => {
            summary.status = "iteration_limit".to_string();
            summary.bug_category = Some(error.bug_category().to_string());
            summary.error = Some(error.to_string());
        }
        Err(error) => {
            summary.bug_category = Some(error.bug_category().to_string());
            summary.error = Some(error.to_string());
        }
    }

    summary.elapsed_ms = started.elapsed().as_millis();
    summary
}

fn run_provider_diff_health_samples_parallel(
    case: &ProviderBinCase,
    case_dir: &Path,
    requested_runs: usize,
    case_concurrency: usize,
    suite_timeout_secs: u64,
) -> Vec<ProviderDiffHealthRunSummary> {
    let jobs = case_concurrency.min(requested_runs.max(1));
    let queue = Arc::new(Mutex::new(VecDeque::from(
        (0..requested_runs).collect::<Vec<_>>(),
    )));
    let case = Arc::new(case.clone());
    let case_dir = Arc::new(case_dir.to_path_buf());
    let mut handles = Vec::new();

    for _ in 0..jobs {
        let queue = Arc::clone(&queue);
        let case = Arc::clone(&case);
        let case_dir = Arc::clone(&case_dir);
        handles.push(thread::spawn(move || {
            let mut summaries = Vec::new();
            loop {
                let sample_index = {
                    let mut queue = queue
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    queue.pop_front()
                };
                let Some(sample_index) = sample_index else {
                    break;
                };
                summaries.push(run_provider_diff_health_sample(
                    &case,
                    &case_dir,
                    sample_index,
                    suite_timeout_secs,
                ));
            }
            summaries
        }));
    }

    let mut summaries = Vec::new();
    for handle in handles {
        summaries.extend(
            handle
                .join()
                .expect("provider diff health sample worker panicked"),
        );
    }
    summaries.sort_by_key(|summary| summary.sample_index);
    summaries
}

fn provider_diff_health_error_summary(
    case: &ProviderBinCase,
    suite: &str,
    requested_runs: usize,
    case_concurrency: usize,
    timeout_secs: u64,
    log_health: Option<LiteLogHealthCaseSummary>,
    error: String,
) -> ProviderDiffHealthCaseSummary {
    let mut failed_checks = Vec::new();
    if let Some(log_health) = &log_health {
        failed_checks.extend(lite_log_health_failure_messages(log_health));
    }

    ProviderDiffHealthCaseSummary {
        status: "fail".to_string(),
        suite: suite.to_string(),
        id: case.id.clone(),
        impl_ref: case.impl_ref.clone(),
        source_label: case.source_label.clone(),
        binary_path: case.binary_path.display().to_string(),
        binary_name: case.binary_name.clone(),
        isa_base: format!("{:?}", case.isa_base),
        requested_float_extension: format!("{:?}", case.requested_float_extension),
        effective_float_extension: format!("{:?}", provider_pair_effective_float_extension(case)),
        coverage_mode: format!("{:?}", case.coverage_mode),
        allow_unaligned: case.allow_unaligned,
        hart_count: case.hart_count,
        timeout_secs,
        requested_runs,
        case_concurrency,
        diff_run_passes: 0,
        diff_run_iteration_limits: 0,
        diff_run_failures: 0,
        diff_runs: Vec::new(),
        log_health,
        failed_checks,
        error: Some(error),
    }
}

fn run_provider_diff_health_case(
    case: &ProviderBinCase,
    run_root: &Path,
    suite: &str,
    include_log_health: bool,
    requested_runs: usize,
    case_concurrency: usize,
    suite_timeout_secs: u64,
) -> ProviderDiffHealthCaseSummary {
    let case_dir = run_root.join(&case.id);
    if let Err(error) = fs::create_dir_all(&case_dir) {
        return provider_diff_health_error_summary(
            case,
            suite,
            requested_runs,
            case_concurrency,
            suite_timeout_secs,
            None,
            format!("failed to create case dir {}: {error}", case_dir.display()),
        );
    }

    let log_health = if include_log_health {
        let lite_case = lite_case_from_provider_case(case);
        let log_root = case_dir.join("lite_log_health");
        Some(
            run_lite_log_health_case(&lite_case, &log_root)
                .unwrap_or_else(|error| lite_log_health_error_summary(&lite_case, error)),
        )
    } else {
        None
    };

    let diff_runs = run_provider_diff_health_samples_parallel(
        case,
        &case_dir,
        requested_runs,
        case_concurrency,
        suite_timeout_secs,
    );
    let diff_run_passes = diff_runs
        .iter()
        .filter(|summary| summary.status == "pass")
        .count();
    let diff_run_iteration_limits = diff_runs
        .iter()
        .filter(|summary| summary.status == "iteration_limit")
        .count();
    let diff_run_failures = diff_runs
        .iter()
        .filter(|summary| summary.status == "fail")
        .count();

    let mut failed_checks = Vec::new();
    if let Some(log_health) = &log_health {
        failed_checks.extend(lite_log_health_failure_messages(log_health));
    }
    for run in &diff_runs {
        if run.status == "fail" {
            let category = run
                .bug_category
                .as_deref()
                .unwrap_or("UNKNOWN_DIFF_HEALTH_FAILURE");
            let error = run.error.as_deref().unwrap_or("unknown error");
            failed_checks.push(format!(
                "diff run {} {}: {}",
                run.sample_index, category, error
            ));
        }
    }

    let status = if failed_checks.is_empty() {
        "pass".to_string()
    } else {
        "fail".to_string()
    };

    ProviderDiffHealthCaseSummary {
        status,
        suite: suite.to_string(),
        id: case.id.clone(),
        impl_ref: case.impl_ref.clone(),
        source_label: case.source_label.clone(),
        binary_path: case.binary_path.display().to_string(),
        binary_name: case.binary_name.clone(),
        isa_base: format!("{:?}", case.isa_base),
        requested_float_extension: format!("{:?}", case.requested_float_extension),
        effective_float_extension: format!("{:?}", provider_pair_effective_float_extension(case)),
        coverage_mode: format!("{:?}", case.coverage_mode),
        allow_unaligned: case.allow_unaligned,
        hart_count: case.hart_count,
        timeout_secs: suite_timeout_secs,
        requested_runs,
        case_concurrency,
        diff_run_passes,
        diff_run_iteration_limits,
        diff_run_failures,
        diff_runs,
        log_health,
        failed_checks,
        error: None,
    }
}

fn run_provider_diff_health_cases_parallel(
    cases: Vec<ProviderBinCase>,
    run_root: &Path,
    suite: &'static str,
    include_log_health: bool,
) -> Vec<ProviderDiffHealthCaseSummary> {
    let requested_runs = diff_health_runs();
    let case_concurrency = diff_health_effective_case_concurrency();
    let case_parallelism = diff_health_case_parallelism(cases.len());
    let suite_timeout_secs = lite_exec_timeout_secs();
    let queue = Arc::new(Mutex::new(VecDeque::from(cases)));
    let run_root = Arc::new(run_root.to_path_buf());
    let mut handles = Vec::new();

    for _ in 0..case_parallelism {
        let queue = Arc::clone(&queue);
        let run_root = Arc::clone(&run_root);
        handles.push(thread::spawn(move || {
            let mut summaries = Vec::new();
            loop {
                let case = {
                    let mut queue = queue
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    queue.pop_front()
                };
                let Some(case) = case else {
                    break;
                };
                summaries.push(run_provider_diff_health_case(
                    &case,
                    &run_root,
                    suite,
                    include_log_health,
                    requested_runs,
                    case_concurrency,
                    suite_timeout_secs,
                ));
            }
            summaries
        }));
    }

    let mut summaries = Vec::new();
    for handle in handles {
        summaries.extend(
            handle
                .join()
                .expect("provider diff health case worker panicked"),
        );
    }
    summaries.sort_by(|left, right| left.id.cmp(&right.id));
    summaries
}

fn provider_diff_health_failures(summaries: &[ProviderDiffHealthCaseSummary]) -> Vec<String> {
    let mut failures = Vec::new();
    for summary in summaries {
        if let Some(error) = &summary.error {
            failures.push(format!("{}: {}", summary.id, error));
        }
        for check in &summary.failed_checks {
            failures.push(format!("{}: {}", summary.id, check));
        }
    }
    failures
}

fn provider_diff_health_matrix_summary(
    suite: &str,
    summaries: &[ProviderDiffHealthCaseSummary],
    failures: Vec<String>,
) -> ProviderDiffHealthMatrixSummary {
    ProviderDiffHealthMatrixSummary {
        suite: suite.to_string(),
        total_cases: summaries.len(),
        passed_cases: summaries
            .iter()
            .filter(|summary| summary.status == "pass")
            .count(),
        failed_cases: summaries
            .iter()
            .filter(|summary| summary.status != "pass")
            .count(),
        requested_runs: diff_health_runs(),
        case_concurrency: diff_health_effective_case_concurrency(),
        max_parallel_budget: diff_health_max_parallel_budget(),
        case_parallelism: diff_health_case_parallelism(summaries.len()),
        diff_run_passes: summaries
            .iter()
            .map(|summary| summary.diff_run_passes)
            .sum(),
        diff_run_iteration_limits: summaries
            .iter()
            .map(|summary| summary.diff_run_iteration_limits)
            .sum(),
        diff_run_failures: summaries
            .iter()
            .map(|summary| summary.diff_run_failures)
            .sum(),
        case_ids: summaries.iter().map(|summary| summary.id.clone()).collect(),
        failures,
    }
}

#[test]
fn provider_bins_have_single_hart_spike_diff_health() {
    let _guard = TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _plugin_dirs_guard = configure_plugin_dirs();

    let provider_root = provider_root();
    let artifact_dir = provider_artifact_dir(&provider_root);
    let spike_path = resolve_spike_path();
    let _provider_env_guards = ensure_aux_provider_envs(&provider_root, &artifact_dir, &spike_path);
    let _provider_binary_env_guards = configure_provider_binary_envs(&provider_root, &artifact_dir);

    let all_cases = collect_provider_bins(&provider_root, &artifact_dir);
    let cases = collect_provider_diff_health_cases(&all_cases, 1, SINGLE_HART_DIFF_HEALTH_IMPLS);
    assert!(
        !cases.is_empty(),
        "no canonical single-hart provider bins found for spike diff health"
    );

    let _suite_timeout_guard = configure_provider_diff_health_timeout(&cases);
    let _selected_binary_guards = configure_selected_provider_case_binaries(&cases);

    let run_root = run_root_for().join("provider_single_hart_spike_diff_health");
    fs::create_dir_all(&run_root)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", run_root.display()));

    let summaries = run_provider_diff_health_cases_parallel(
        cases.clone(),
        &run_root,
        "single_hart_spike_diff_health",
        true,
    );
    let mut failures = provider_diff_health_failures(&summaries);
    if normalize_case_filter().is_none() {
        for missing in missing_expected_provider_impls(&cases, SINGLE_HART_DIFF_HEALTH_IMPLS) {
            failures.push(format!(
                "missing canonical single-hart provider case for {missing}"
            ));
        }
    }

    let matrix_summary = provider_diff_health_matrix_summary(
        "single_hart_spike_diff_health",
        &summaries,
        failures.clone(),
    );
    write_json_pretty(&run_root.join("matrix_summary.json"), &matrix_summary)
        .unwrap_or_else(|e| panic!("{e}"));
    write_json_pretty(&run_root.join("case_summaries.json"), &summaries)
        .unwrap_or_else(|e| panic!("{e}"));

    if !failures.is_empty() {
        panic!(
            "single-hart spike diff health found {} failing provider case(s) under {}:\n{}",
            failures.len(),
            run_root.display(),
            failures.join("\n")
        );
    }
}

#[test]
fn provider_bins_have_dual_hart_spike_diff_health() {
    let _guard = TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _plugin_dirs_guard = configure_plugin_dirs();

    let provider_root = provider_root();
    let artifact_dir = provider_artifact_dir(&provider_root);
    let spike_path = resolve_spike_path();
    let _provider_env_guards = ensure_aux_provider_envs(&provider_root, &artifact_dir, &spike_path);
    let _provider_binary_env_guards = configure_provider_binary_envs(&provider_root, &artifact_dir);

    let all_cases = collect_provider_bins(&provider_root, &artifact_dir);
    let cases = collect_provider_diff_health_cases(&all_cases, 2, DUAL_HART_DIFF_HEALTH_IMPLS);
    assert!(
        !cases.is_empty(),
        "no canonical dual-hart provider bins found for spike diff health"
    );

    let _suite_timeout_guard = configure_provider_diff_health_timeout(&cases);
    let _selected_binary_guards = configure_selected_provider_case_binaries(&cases);

    let run_root = run_root_for().join("provider_dual_hart_spike_diff_health");
    fs::create_dir_all(&run_root)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", run_root.display()));

    let summaries = run_provider_diff_health_cases_parallel(
        cases.clone(),
        &run_root,
        "dual_hart_spike_diff_health",
        true,
    );
    let mut failures = provider_diff_health_failures(&summaries);
    if normalize_case_filter().is_none() {
        for missing in missing_expected_provider_impls(&cases, DUAL_HART_DIFF_HEALTH_IMPLS) {
            failures.push(format!(
                "missing canonical dual-hart provider case for {missing}"
            ));
        }
    }

    let matrix_summary = provider_diff_health_matrix_summary(
        "dual_hart_spike_diff_health",
        &summaries,
        failures.clone(),
    );
    write_json_pretty(&run_root.join("matrix_summary.json"), &matrix_summary)
        .unwrap_or_else(|e| panic!("{e}"));
    write_json_pretty(&run_root.join("case_summaries.json"), &summaries)
        .unwrap_or_else(|e| panic!("{e}"));

    if !failures.is_empty() {
        panic!(
            "dual-hart spike diff health found {} failing provider case(s) under {}:\n{}",
            failures.len(),
            run_root.display(),
            failures.join("\n")
        );
    }
}

#[test]
fn weak_fallback_uses_partial_context_for_missing_user_instruction_execution() {
    let decode =
        |asm: &str| RiscvInstruction::decode_asm_with_context(asm, IsaDecodeContext::RV64).unwrap();
    let source = riscv_instruction::InstructionSequence::new(decode("addi x1, x0, 1"));
    let output = ExecutionOutput {
        per_hart: BTreeMap::from([(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: Vec::new(),
            },
        )]),
        core_id: "test".to_string(),
        isa_base: ISABase::Rv64,
        perf: Default::default(),
    };

    let full_error =
        build_execution_context_output(output.clone(), (0, 0xff), &[(0, vec![source.clone()])])
            .expect_err("full builder should reject missing logical execution");
    assert!(matches!(
        full_error,
        ContextBuildError::MissingLogicalInstructionExecution {
            hart_id: 0,
            instruction_index: 0,
        }
    ));

    let context =
        build_execution_context_output_with_weak_fallback(output, (0, 0xff), &[(0, vec![source])])
            .expect("weak fallback should build partial context");
    assert_eq!(context.logical_view.len(), 1);
    let logical = &context.logical_view[0];
    assert_eq!(logical.hart_id, 0);
    assert_eq!(logical.instruction_index, 0);
    assert!(logical.asm_instruction_indices.is_empty());
    assert!(logical.register_writes.is_empty());
    assert!(logical.memory_writes.is_empty());
    assert!(logical.exceptions.is_empty());
    assert!(!logical.timing.is_recorded());
}
