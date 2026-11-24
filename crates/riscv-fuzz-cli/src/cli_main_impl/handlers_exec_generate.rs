use std::{
    collections::VecDeque,
    env, fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
};

use log::info;
use riscv_fuzz_config::ExperimentConfig;
use riscv_fuzz_diff::{DiffExecutionView, compare_execution_views};
use riscv_fuzz_plugin_api::{CoverageMode, ISABase, IsaExtensions};
use riscv_fuzz_plugin_manager::PluginManager;
use riscv_fuzz_testcase_engine::execution_artifacts::{
    ExecutionArtifactOptions, MulticoreSyncMetricFailureMode, execute_testcase_with_artifacts,
};
use riscv_fuzz_toolchain::build_elf::build_elf_with_extensions;
use riscv_fuzz_toolchain::{
    build_execution_context_output, build_partial_execution_context_output, write_json_pretty,
    write_string_file,
};
use riscv_instruction::{RV32Extensions, RV64Extensions, RiscvInstruction};
use serde::{Deserialize, Serialize};

use super::{CliError, Result, ResultExt, set_env_var};

macro_rules! bail {
    ($($arg:tt)*) => {
        return Err(CliError::new(format!($($arg)*)));
    };
}

const SPIKE_REFERENCE_CORE_ID: &str = "spike";
const SPIKE_ISA_SUFFIX_ENV: &str = "RISCV_FUZZ_SPIKE_ISA_SUFFIX";
const SPIKE_ENABLE_VECTOR_STATE_ENV: &str = "RISCV_FUZZ_SPIKE_ENABLE_VECTOR_STATE";

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(super) enum ExecGeneratedGenerator {
    /// Existing extension-aware random hardware generator.
    Hardware,
    /// Original hardware generator plus RVV instruction constraints, without obligation blocks.
    HardwareConstraints,
    /// Original hardware generator plus RVV obligation-model blocks, without RVV constraints.
    HardwareObligation,
    /// Original hardware generator plus RVV constraints and obligation-model blocks.
    HardwareConstraintsObligation,
    /// Original hardware generator plus an SSM state-flow RVV block, with constraints disabled.
    #[value(
        name = "hardware-stateflow",
        alias = "hardware-ssm",
        alias = "hardware-state-flow"
    )]
    HardwareStateFlow,
    /// Original hardware generator plus an SSM state-flow RVV block with fused RCM instantiation.
    #[value(
        name = "hardware-stateflow-constraints",
        alias = "hardware-ssm-rcm",
        alias = "hardware-state-flow-constraints"
    )]
    HardwareStateFlowConstraints,
    /// Constraint-aware bidirectional RVV state-flow generator.
    #[value(name = "hardware-tcad")]
    HardwareTcad,
    /// VC obligation block plus constrained SF block; no ambient test instructions.
    #[value(name = "hardware-vc-sf-only")]
    HardwareVcSfOnly,
    /// VC+SF only, positive operands, and probability-sampled expected Illegal.
    #[value(name = "hardware-vc-sf-only-negative")]
    HardwareVcSfOnlyNegative,
    /// VC+SF blocks and one explicit base-V grammar opcode per seed.
    #[value(name = "hardware-vc-sf-catalog")]
    HardwareVcSfCatalog,
    /// VC+SF catalog sweep with the conservative state-flow template.
    #[value(name = "hardware-vc-sf-catalog-safe")]
    HardwareVcSfCatalogSafe,
    /// VC+SF with seeded random opcode order covering the full base-V catalog
    /// in every 627-case epoch, and random operands for the selected opcode.
    #[value(name = "hardware-vc-sf-random")]
    HardwareVcSfRandom,
    /// ARA-specific generic SSM+RCM profile for RVV risk exploration.
    #[value(name = "hardware-ara-risk", alias = "ara-risk")]
    HardwareAraRisk,
    /// ARA ablation: remove all RVV legality/operand constraints but keep sequencing.
    #[value(
        name = "hardware-ara-risk-no-constraints",
        alias = "ara-risk-no-constraints"
    )]
    HardwareAraRiskNoConstraints,
    /// ARA ablation: remove SSM sequencing and ARA risk families, keeping only RVV constraints.
    #[value(name = "hardware-ara-risk-no-sequence", alias = "ara-risk-no-sequence")]
    HardwareAraRiskNoSequence,
    /// ARA ablation: pure random RVV instructions, without constraints or ARA risk families.
    #[value(
        name = "hardware-ara-risk-no-constraints-no-sequence",
        alias = "ara-risk-no-constraints-no-sequence"
    )]
    HardwareAraRiskNoConstraintsNoSequence,
    /// Conservative ARA state-flow profile for direct execution smoke tests.
    #[value(name = "hardware-ara-safe", alias = "ara-safe")]
    HardwareAraSafe,
    /// Legacy alias for `hardware`; keeps full hardware ISA extensions.
    HardwareRvvRandom,
    /// Legacy alias for `hardware-constraints`; keeps full hardware ISA extensions.
    HardwareRvvConstraintsOnly,
    /// Legacy alias for `hardware-constraints-obligation`; keeps full hardware ISA extensions.
    HardwareRvv,
}

fn generate_testcase_for_mode(
    core_ids: &[String],
    testcase_config: riscv_fuzz_testcase_engine::testcase::TestCaseConfig,
    extensions: IsaExtensions,
    generator: ExecGeneratedGenerator,
) -> std::result::Result<
    riscv_fuzz_testcase_engine::testcase::GeneratedTestCase,
    riscv_fuzz_testcase_engine::GenerationError,
> {
    match generator {
        ExecGeneratedGenerator::Hardware | ExecGeneratedGenerator::HardwareRvvRandom => {
            riscv_fuzz_testcase_engine::impls::generate_random_testcase_with_extensions(
                core_ids,
                testcase_config,
                extensions,
            )
        }
        ExecGeneratedGenerator::HardwareConstraints
        | ExecGeneratedGenerator::HardwareRvvConstraintsOnly => riscv_fuzz_testcase_engine::impls::generate_hardware_constraints_testcase_with_extensions(
            core_ids,
            testcase_config,
            extensions,
        ),
        ExecGeneratedGenerator::HardwareObligation => riscv_fuzz_testcase_engine::impls::generate_hardware_obligation_testcase_with_extensions(
            core_ids,
            testcase_config,
            extensions,
        ),
        ExecGeneratedGenerator::HardwareConstraintsObligation | ExecGeneratedGenerator::HardwareRvv => {
            riscv_fuzz_testcase_engine::impls::generate_hardware_constraints_obligation_testcase_with_extensions(
                core_ids,
                testcase_config,
                extensions,
            )
        }
        ExecGeneratedGenerator::HardwareStateFlow => {
            riscv_fuzz_testcase_engine::impls::generate_hardware_stateflow_testcase_with_extensions(
                core_ids,
                testcase_config,
                extensions,
            )
        }
        ExecGeneratedGenerator::HardwareStateFlowConstraints => {
            riscv_fuzz_testcase_engine::impls::generate_hardware_stateflow_constraints_testcase_with_extensions(
                core_ids,
                testcase_config,
                extensions,
            )
        }
        ExecGeneratedGenerator::HardwareTcad => {
            riscv_fuzz_testcase_engine::rvv_paper_graph::generate(
                core_ids,
                testcase_config,
                extensions,
            )
        }
        ExecGeneratedGenerator::HardwareVcSfOnly
        | ExecGeneratedGenerator::HardwareVcSfOnlyNegative
        | ExecGeneratedGenerator::HardwareVcSfCatalog
        | ExecGeneratedGenerator::HardwareVcSfCatalogSafe
        | ExecGeneratedGenerator::HardwareVcSfRandom => {
            let mode = match generator {
                ExecGeneratedGenerator::HardwareVcSfCatalog =>
                    riscv_fuzz_testcase_engine::rvv_model_program::RvvModelProgramMode::CatalogSweep,
                ExecGeneratedGenerator::HardwareVcSfCatalogSafe =>
                    riscv_fuzz_testcase_engine::rvv_model_program::RvvModelProgramMode::CatalogSweepSafe,
                ExecGeneratedGenerator::HardwareVcSfRandom =>
                    riscv_fuzz_testcase_engine::rvv_model_program::RvvModelProgramMode::RandomCatalog,
                ExecGeneratedGenerator::HardwareVcSfOnlyNegative =>
                    riscv_fuzz_testcase_engine::rvv_model_program::RvvModelProgramMode::PureNegative,
                _ => riscv_fuzz_testcase_engine::rvv_model_program::RvvModelProgramMode::Pure,
            };
            riscv_fuzz_testcase_engine::rvv_model_program::generate_rvv_model_program_with_extensions(
                core_ids,
                testcase_config,
                extensions,
                mode,
            )
        }
        ExecGeneratedGenerator::HardwareAraRisk => {
            riscv_fuzz_testcase_engine::impls::generate_hardware_ara_risk_testcase_with_extensions(
                core_ids,
                testcase_config,
                extensions,
            )
        }
        ExecGeneratedGenerator::HardwareAraRiskNoConstraints => {
            riscv_fuzz_testcase_engine::impls::generate_hardware_ara_risk_no_constraints_testcase_with_extensions(
                core_ids,
                testcase_config,
                extensions,
            )
        }
        ExecGeneratedGenerator::HardwareAraRiskNoSequence => {
            riscv_fuzz_testcase_engine::impls::generate_hardware_ara_risk_no_sequence_testcase_with_extensions(
                core_ids,
                testcase_config,
                extensions,
            )
        }
        ExecGeneratedGenerator::HardwareAraRiskNoConstraintsNoSequence => {
            riscv_fuzz_testcase_engine::impls::generate_hardware_ara_risk_no_constraints_no_sequence_testcase_with_extensions(
                core_ids,
                testcase_config,
                extensions,
            )
        }
        ExecGeneratedGenerator::HardwareAraSafe => {
            riscv_fuzz_testcase_engine::impls::generate_hardware_ara_safe_testcase_with_extensions(
                core_ids,
                testcase_config,
                extensions,
            )
        }
    }
}

impl ExecGeneratedGenerator {
    fn is_ara_only(self) -> bool {
        matches!(
            self,
            ExecGeneratedGenerator::HardwareAraRisk
                | ExecGeneratedGenerator::HardwareAraRiskNoConstraints
                | ExecGeneratedGenerator::HardwareAraRiskNoSequence
                | ExecGeneratedGenerator::HardwareAraRiskNoConstraintsNoSequence
                | ExecGeneratedGenerator::HardwareAraSafe
        )
    }
}

fn effective_generator_for_core(
    core_id: &str,
    generator: ExecGeneratedGenerator,
) -> Result<ExecGeneratedGenerator> {
    let core = core_id.to_ascii_lowercase();
    match (core.as_str(), generator) {
        ("ara", ExecGeneratedGenerator::Hardware) => Ok(ExecGeneratedGenerator::HardwareAraRisk),
        ("ara", _) => Ok(generator),
        (_, generator) if generator.is_ara_only() => Err(CliError::new(format!(
            "generator {generator:?} is ARA-only and cannot be used for core {core_id}"
        ))),
        _ => Ok(generator),
    }
}

struct ExecGeneratedExtensionSets {
    generation: IsaExtensions,
    artifacts: IsaExtensions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectExecutionJudgement {
    XiangShanDifftest,
    AraExecution,
    GenericExecution,
}

impl DirectExecutionJudgement {
    fn for_core(core_id: &str) -> Self {
        match core_id.to_ascii_lowercase().as_str() {
            "xiangshan" => Self::XiangShanDifftest,
            "ara" => Self::AraExecution,
            _ => Self::GenericExecution,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::XiangShanDifftest => "xiangshan_difftest_commit_log",
            Self::AraExecution => "ara_vs_spike_execution_trace",
            Self::GenericExecution => "generic_execution_trace",
        }
    }

    fn strict_mismatch_capable(self) -> bool {
        matches!(self, Self::XiangShanDifftest | Self::AraExecution)
    }

    fn uses_xiangshan_commit_log(self) -> bool {
        matches!(self, Self::XiangShanDifftest)
    }

    fn uses_spike_reference(self) -> bool {
        matches!(self, Self::AraExecution)
    }

    fn is_pass_result(self, result: &str) -> bool {
        match self {
            Self::XiangShanDifftest => result == "pass",
            Self::AraExecution | Self::GenericExecution => result == "completed",
        }
    }

    fn case_is_pass(self, case: &DirectDifftestBatchCase) -> bool {
        case.status == "completed"
            && case
                .result
                .as_deref()
                .map(|result| self.is_pass_result(result))
                .unwrap_or(false)
    }

    fn classify_error_result(self, error: &str) -> &'static str {
        match self {
            Self::XiangShanDifftest => "execution_error",
            Self::AraExecution => classify_ara_error_result(error),
            Self::GenericExecution => classify_generic_error_result(error),
        }
    }
}

fn direct_execution_core_ids(core_id: &str, judgement: DirectExecutionJudgement) -> Vec<String> {
    if judgement.uses_spike_reference() {
        PluginManager::normalize_impls(vec![
            core_id.to_string(),
            SPIKE_REFERENCE_CORE_ID.to_string(),
        ])
    } else {
        PluginManager::normalize_impls(vec![core_id.to_string()])
    }
}

fn align_spike_reference_mem_range(
    testcase: &mut riscv_fuzz_testcase_engine::testcase::GeneratedTestCase,
    target_core_id: &str,
) -> Result<()> {
    let target_range = testcase
        .mem_range
        .get(target_core_id)
        .copied()
        .ok_or_else(|| CliError::new(format!("missing mem_range for {target_core_id}")))?;
    if !testcase.mem_range.contains_key(SPIKE_REFERENCE_CORE_ID) {
        return Err(CliError::new(format!(
            "missing mem_range for {} reference",
            SPIKE_REFERENCE_CORE_ID
        )));
    }
    testcase
        .mem_range
        .insert(SPIKE_REFERENCE_CORE_ID.to_string(), target_range);
    Ok(())
}

fn classify_ara_error_result(error: &str) -> &'static str {
    let lower = error.to_ascii_lowercase();
    if lower.contains("timed out after") || lower.contains("timeout") {
        "timeout"
    } else if lower.contains("invalid line format")
        || lower.contains("trace") && lower.contains("parse")
        || lower.contains("memory mask") && lower.contains("8-byte data field")
    {
        "trace_parse_error"
    } else if lower.contains("mismatch") {
        "mismatch_or_abort"
    } else {
        "execution_error"
    }
}

fn classify_generic_error_result(error: &str) -> &'static str {
    let lower = error.to_ascii_lowercase();
    if lower.contains("timed out after") || lower.contains("timeout") {
        "timeout"
    } else if lower.contains("mismatch") || lower.contains("abort") {
        "mismatch_or_abort"
    } else {
        "execution_error"
    }
}

fn ara_spike_reference_isa_suffix_env_value() -> String {
    let suffix = format!(
        "_zvl{}b",
        riscv_fuzz_testcase_engine::rvv_artifact_settings::ARA_VLEN_BITS
    );
    let Ok(existing) = env::var(SPIKE_ISA_SUFFIX_ENV) else {
        return suffix;
    };
    let remaining = existing
        .trim()
        .split('_')
        .filter(|part| {
            !part
                .to_ascii_lowercase()
                .strip_prefix("zvl")
                .and_then(|value| value.strip_suffix('b'))
                .is_some_and(|digits| {
                    !digits.is_empty() && digits.bytes().all(|c| c.is_ascii_digit())
                })
        })
        .collect::<Vec<_>>()
        .join("_");
    if remaining.is_empty() {
        suffix
    } else {
        format!("{remaining}{suffix}")
    }
}

fn set_ara_spike_reference_env(judgement: DirectExecutionJudgement) -> Vec<super::EnvVarGuard> {
    let mut guards = Vec::new();
    if judgement.uses_spike_reference() {
        let suffix = ara_spike_reference_isa_suffix_env_value();
        guards.push(set_env_var(SPIKE_ISA_SUFFIX_ENV, &suffix));
        guards.push(set_env_var(SPIKE_ENABLE_VECTOR_STATE_ENV, "1"));
    }
    guards
}

fn exec_generated_extension_sets(
    core_ids: &[String],
    testcase_config: &riscv_fuzz_testcase_engine::testcase::TestCaseConfig,
    disable_zacas: bool,
    _generator: ExecGeneratedGenerator,
) -> Result<ExecGeneratedExtensionSets> {
    let mut artifacts =
        riscv_fuzz_testcase_engine::impls::extensions_for_config(core_ids, testcase_config)
            .with_context(|| "failed to resolve ISA extensions from workload config")?;
    remove_gnu_as_unsupported_extensions(&mut artifacts);
    if disable_zacas {
        remove_zacas(&mut artifacts);
    }
    let generation = artifacts.clone();
    Ok(ExecGeneratedExtensionSets {
        generation,
        artifacts,
    })
}

fn remove_zacas(extensions: &mut IsaExtensions) {
    match extensions {
        IsaExtensions::Rv32(values) => values.retain(|ext| *ext != RV32Extensions::Zacas),
        IsaExtensions::Rv64(values) => values.retain(|ext| *ext != RV64Extensions::Zacas),
    }
}

fn remove_gnu_as_unsupported_extensions(extensions: &mut IsaExtensions) {
    match extensions {
        IsaExtensions::Rv32(values) => values.retain(|ext| {
            *ext != RV32Extensions::B
                && *ext != RV32Extensions::Zaamo
                && *ext != RV32Extensions::Zabha
                && *ext != RV32Extensions::Zacas
                && *ext != RV32Extensions::Zalasr
                && *ext != RV32Extensions::Zalrsc
                && *ext != RV32Extensions::Zawrs
                && *ext != RV32Extensions::Zcmop
                && *ext != RV32Extensions::Zimop
        }),
        IsaExtensions::Rv64(values) => values.retain(|ext| {
            *ext != RV64Extensions::B
                && *ext != RV64Extensions::Zaamo
                && *ext != RV64Extensions::Zabha
                && *ext != RV64Extensions::Zacas
                && *ext != RV64Extensions::Zalasr
                && *ext != RV64Extensions::Zalrsc
                && *ext != RV64Extensions::Zawrs
                && *ext != RV64Extensions::Zcmop
                && *ext != RV64Extensions::Zimop
        }),
    }
}

/// `generate` subcommand: produce program.S/.elf for a single core without
/// executing. The caller names the target core via `--core` because the
/// experiment config doesn't fix one for generation.
pub(super) fn run_generate_unified(
    cfg: &ExperimentConfig,
    run_dir: PathBuf,
    core_id: String,
    isa_override: Option<ISABase>,
    disable_zacas: bool,
    minimal_artifacts: bool,
    generator: ExecGeneratedGenerator,
) -> Result<()> {
    let mut testcase_config = cfg.testcase_config.clone();
    if let Some(isa) = isa_override {
        testcase_config.isa_base = isa;
    }
    let isa_base = testcase_config.isa_base;
    let plugin_manager = PluginManager::global();
    if !plugin_manager.supports_isa_base(&core_id, isa_base) {
        bail!("core {core_id} does not support ISA base {isa_base}");
    }
    let plugin = plugin_manager
        .plugin(&core_id)
        .map_err(|error| CliError::new(error.to_string()))?;

    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let core_ids = PluginManager::normalize_impls(vec![core_id.clone()]);
    let generator = effective_generator_for_core(&core_id, generator)?;
    let extension_sets =
        exec_generated_extension_sets(&core_ids, &testcase_config, disable_zacas, generator)?;
    let mut testcase = generate_testcase_for_mode(
        &core_ids,
        testcase_config.clone(),
        extension_sets.generation,
        generator,
    )
    .with_context(|| "failed to generate testcase from workload config")?;
    testcase.extensions = extension_sets.artifacts;

    let instructions = testcase
        .combined_rendered_lines_of(&core_id)
        .ok_or_else(|| CliError::new(format!("missing instructions for {core_id}")))?;
    let typed_inputs = testcase
        .combined_typed_sequences_per_hart_of(&core_id)
        .ok_or_else(|| {
            CliError::new(format!("missing typed instruction sequences for {core_id}"))
        })?;
    let typed_input_refs = typed_inputs
        .iter()
        .map(|(hart_id, seqs)| (*hart_id, seqs.as_slice()))
        .collect::<Vec<_>>();

    if !minimal_artifacts {
        let testcase_path = run_dir.join("testcase.json");
        write_json_pretty(&testcase_path, &testcase)
            .with_context(|| format!("failed to write {}", testcase_path.display()))?;

        let mut snapshot_content = instructions.join("\n");
        snapshot_content.push('\n');
        let snapshot_path = run_dir.join("input_instructions.asm");
        write_string_file(&snapshot_path, snapshot_content)
            .with_context(|| format!("failed to write {}", snapshot_path.display()))?;
    }

    let asm_program = plugin
        .build_asm(typed_input_refs.as_slice(), isa_base)
        .with_context(|| format!("failed to build assembly for {core_id} ({isa_base})"))?;
    let asm_path = run_dir.join("program.S");
    write_string_file(&asm_path, asm_program)
        .with_context(|| format!("failed to write {}", asm_path.display()))?;

    let build_result =
        build_elf_with_extensions(&asm_path, &testcase.extensions, plugin.linker_script())
            .with_context(|| format!("failed to build ELF artifacts for {core_id} ({isa_base})"))?;

    let test_len = testcase.test_traceable_len();
    if test_len == 0 {
        return Err(CliError::new(format!(
            "generated testcase is missing test instructions for core {core_id}"
        )));
    }
    let artifact_note = if minimal_artifacts {
        "minimal artifact mode (only .S/.s/.elf emitted)"
    } else {
        "full artifact set emitted"
    };
    info!(
        "Generated {test_len} test instructions for {core_id} ({isa_base}); ELF at {}; {artifact_note}",
        build_result.executable_file.display(),
    );

    Ok(())
}

pub(super) fn run_exec_generated_unified(
    cfg: &ExperimentConfig,
    run_dir: PathBuf,
    core_id: String,
    isa_override: Option<ISABase>,
    disable_zacas: bool,
    generator: ExecGeneratedGenerator,
) -> Result<()> {
    let mut testcase_config = cfg.testcase_config.clone();
    if let Some(isa) = isa_override {
        testcase_config.isa_base = isa;
    }
    let isa_base = testcase_config.isa_base;
    let plugin_manager = PluginManager::global();
    let requested_core_ids = PluginManager::normalize_impls(vec![core_id.clone()]);
    if requested_core_ids.len() != 1 {
        bail!("exec-generated expects exactly one core, got {requested_core_ids:?}");
    }
    let core_id = requested_core_ids[0].clone();
    let generator = effective_generator_for_core(&core_id, generator)?;
    if !plugin_manager.supports_isa_base(&core_id, isa_base) {
        bail!("core {core_id} does not support ISA base {isa_base}");
    }
    let judgement = DirectExecutionJudgement::for_core(&core_id);
    let core_ids = direct_execution_core_ids(&core_id, judgement);
    for exec_core_id in &core_ids {
        if !plugin_manager.supports_isa_base(exec_core_id, isa_base) {
            bail!("core {exec_core_id} does not support ISA base {isa_base}");
        }
    }

    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let extension_sets =
        exec_generated_extension_sets(&core_ids, &testcase_config, disable_zacas, generator)?;
    let mut testcase = generate_testcase_for_mode(
        &core_ids,
        testcase_config,
        extension_sets.generation,
        generator,
    )
    .with_context(|| "failed to generate testcase from workload config")?;
    testcase.extensions = extension_sets.artifacts;
    if judgement.uses_spike_reference() {
        align_spike_reference_mem_range(&mut testcase, &core_id)?;
    }

    let instructions = testcase
        .combined_rendered_lines_of(&core_id)
        .ok_or_else(|| CliError::new(format!("missing instructions for {core_id}")))?;
    let mut snapshot_content = instructions.join("\n");
    snapshot_content.push('\n');
    let snapshot_path = run_dir.join("user_instructions.asm");
    write_string_file(&snapshot_path, snapshot_content)
        .with_context(|| format!("failed to write {}", snapshot_path.display()))?;

    let vector_instruction_count = instructions
        .iter()
        .filter(|line| line.trim_start().starts_with('v'))
        .count();
    let contains_fof = instructions.iter().any(|line| is_fof_instruction(line));
    let memory_instruction_count = memory_looking_instruction_count(&instructions);
    let memory_range = testcase
        .mem_range
        .get(&core_id)
        .map(|(start, end)| format!("0x{start:016x}..0x{end:016x}"))
        .unwrap_or_else(|| "unknown".to_string());
    let mem_cfg = &testcase.config.random_config.memory_access_config;
    let reg_cfg = &testcase.config.random_config.register_operand_config;
    let summary_path = run_dir.join("user_instruction_summary.md");
    write_string_file(
        &summary_path,
        format!(
            "# Generated Execution Summary\n\n- Core: {core_id}\n- ISA: {isa_base}\n- Generator: `{generator:?}`\n- Init traceable instructions: {}\n- Test traceable instructions: {}\n- Combined rendered user instructions: {}\n- Vector-looking instructions: {vector_instruction_count}\n- Memory-looking instructions: {memory_instruction_count}\n- Contains fault-only-first load: {contains_fof}\n- Memory size bytes: {}\n- Effective memory range: `{}`\n- Memory offset immediate range: `{}..{}`\n- Memory base/scratch registers: `x{}..x{}`\n- Integer operand registers: `x{}..x{}`\n- Floating operand registers: `f{}..f{}`\n- Vector operand registers: `v{}..{}`\n",
            testcase.init_traceable_len(),
            testcase.test_traceable_len(),
            instructions.len(),
            testcase.config.mem_size,
            memory_range,
            mem_cfg.memory_offset_immediate_range.0,
            mem_cfg.memory_offset_immediate_range.1,
            mem_cfg.memory_base_register_number_range.0,
            mem_cfg.memory_base_register_number_range.1,
            reg_cfg.integer_register_number_range.0,
            reg_cfg.integer_register_number_range.1,
            reg_cfg.floating_point_register_number_range.0,
            reg_cfg.floating_point_register_number_range.1,
            reg_cfg.vector_register_number_range.0,
            reg_cfg.vector_register_number_range.1,
        ),
    )
    .with_context(|| format!("failed to write {}", summary_path.display()))?;

    let _ara_spike_reference_env_guards = set_ara_spike_reference_env(judgement);
    let execution_result = match execute_testcase_with_artifacts(
        &testcase,
        &core_ids,
        &run_dir,
        CoverageMode::Off,
        ExecutionArtifactOptions {
            testcase_filename: "testcase.json",
            emit_execution_output_json: true,
            emit_execution_report_md: true,
            emit_execution_timeline_svg: false,
            multicore_sync_metric_failure_mode: MulticoreSyncMetricFailureMode::BestEffort,
        },
    ) {
        Ok(result) => result,
        Err(error) => {
            let error_text = error.to_string();
            write_difftest_error_summary(&run_dir, &core_id, isa_base, judgement, &error_text)?;
            if judgement.uses_xiangshan_commit_log()
                && commit_log_indicates_difftest_pass(&run_dir, &core_id)
            {
                info!(
                    "Generated testcase for {core_id} reached XiangShan GOODTRAP with no DiffTest mismatch markers, but artifact parsing failed: {error_text}"
                );
                return Ok(());
            }
            return Err(CliError::new(format!(
                "failed to directly execute generated testcase for {core_id} in {}: {error}",
                run_dir.display()
            )));
        }
    };

    write_difftest_success_summary(
        &run_dir,
        &core_id,
        isa_base,
        judgement,
        &testcase,
        &execution_result,
    )?;

    info!(
        "Generated and directly executed {} user instructions ({} test, {} vector-looking) for {core_id} ({isa_base}); artifacts under {}; DiffTest result at {}",
        instructions.len(),
        testcase.test_traceable_len(),
        vector_instruction_count,
        run_dir.display(),
        run_dir.join("difftest_result.md").display(),
    );

    Ok(())
}

pub(super) fn run_exec_generated_batch_unified(
    cfg: &ExperimentConfig,
    run_dir: PathBuf,
    core_id: String,
    isa_override: Option<ISABase>,
    cases: usize,
    workers: usize,
    seed_start: Option<u64>,
    disable_zacas: bool,
    generator: ExecGeneratedGenerator,
) -> Result<()> {
    if cases == 0 {
        bail!("exec-generated-batch requires --cases > 0");
    }
    if workers == 0 {
        bail!("exec-generated-batch requires --workers > 0");
    }

    let normalized_core_ids = PluginManager::normalize_impls(vec![core_id]);
    if normalized_core_ids.len() != 1 {
        bail!("exec-generated-batch expects exactly one core, got {normalized_core_ids:?}");
    }
    let core_id = normalized_core_ids[0].clone();
    let generator = effective_generator_for_core(&core_id, generator)?;
    let judgement = DirectExecutionJudgement::for_core(&core_id);
    let _ara_spike_reference_env_guards = set_ara_spike_reference_env(judgement);

    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let base_seed = seed_start.or_else(|| cfg.effective_seed()).unwrap_or(0);
    let worker_count = workers.min(cases);
    let pending = Arc::new(Mutex::new((0..cases).collect::<VecDeque<_>>()));
    let results = Arc::new(Mutex::new(Vec::with_capacity(cases)));
    let cfg = Arc::new(cfg.clone());
    let run_dir = Arc::new(run_dir);
    let core_id = Arc::new(core_id);

    let mut handles = Vec::with_capacity(worker_count);
    for worker in 0..worker_count {
        let pending = Arc::clone(&pending);
        let results = Arc::clone(&results);
        let cfg = Arc::clone(&cfg);
        let run_dir = Arc::clone(&run_dir);
        let core_id = Arc::clone(&core_id);
        handles.push(thread::spawn(move || {
            loop {
                let index = {
                    let mut pending = pending.lock().expect("batch work queue mutex poisoned");
                    pending.pop_front()
                };
                let Some(index) = index else {
                    break;
                };

                let seed = base_seed.wrapping_add(index as u64);
                let case_dir = run_dir.join(format!("case_{index:03}"));
                let mut case_cfg = (*cfg).clone();
                case_cfg.testcase_config.seed = Some(seed);
                let execution_error = run_exec_generated_unified(
                    &case_cfg,
                    case_dir.clone(),
                    (*core_id).clone(),
                    isa_override,
                    disable_zacas,
                    generator,
                )
                .err()
                .map(|error| error.to_string());
                let summary = read_difftest_summary(&case_dir);
                let has_fof = fs::read_to_string(case_dir.join("user_instructions.asm"))
                    .ok()
                    .is_some_and(|source| contains_fof_before_negative(&source));
                let fof_requires_spec_review = has_fof
                    && (summary
                        .as_ref()
                        .is_some_and(|result| result.result == "mismatch_or_abort")
                        || execution_error
                            .as_ref()
                            .is_some_and(|error| error.to_ascii_lowercase().contains("mismatch")));
                let case_result = DirectDifftestBatchCase {
                    index,
                    seed,
                    worker,
                    run_dir: case_dir.display().to_string(),
                    status: if execution_error.is_some() {
                        "failed".to_string()
                    } else {
                        "completed".to_string()
                    },
                    result: summary.as_ref().map(|summary| summary.result.clone()),
                    goodtrap: summary.as_ref().map(|summary| summary.goodtrap),
                    difftest_abort: summary.as_ref().map(|summary| summary.difftest_abort),
                    reference_model: summary.and_then(|summary| summary.reference_model),
                    error: execution_error,
                    fof_requires_spec_review,
                };
                results
                    .lock()
                    .expect("batch result mutex poisoned")
                    .push(case_result);
            }
        }));
    }

    let mut panicked_workers = 0usize;
    for handle in handles {
        if handle.join().is_err() {
            panicked_workers += 1;
        }
    }

    let mut case_results = results
        .lock()
        .map_err(|_| CliError::new("batch result mutex poisoned"))?
        .clone();
    case_results.sort_by_key(|case| case.index);

    let passed = case_results
        .iter()
        .filter(|case| judgement.case_is_pass(case))
        .count();
    let mismatch_or_abort = case_results
        .iter()
        .filter(|case| case.result.as_deref() == Some("mismatch_or_abort"))
        .count();
    let failed = case_results
        .iter()
        .filter(|case| case.status == "failed")
        .count()
        + panicked_workers;
    let summary = DirectDifftestBatchSummary {
        judgement: judgement.label().to_string(),
        strict_mismatch_capable: judgement.strict_mismatch_capable(),
        total_cases: cases,
        workers: worker_count,
        seed_start: base_seed,
        completed_cases: case_results.len(),
        passed,
        non_pass: cases.saturating_sub(passed),
        mismatch_or_abort,
        fof_requires_spec_review: case_results
            .iter()
            .filter(|case| case.fof_requires_spec_review)
            .count(),
        failed,
        panicked_workers,
        cases: case_results,
    };
    write_batch_summary_files(&run_dir, &summary, judgement)?;

    if summary.non_pass > 0 || summary.panicked_workers > 0 {
        return Err(CliError::new(format!(
            "exec-generated-batch completed with {} successful / {} total under {} judgement; see {}",
            summary.passed,
            summary.total_cases,
            summary.judgement,
            run_dir.join("batch_summary.md").display()
        )));
    }

    info!(
        "Generated and directly executed {} testcases for {} with {} workers; all passed; summary at {}",
        summary.total_cases,
        core_id,
        summary.workers,
        run_dir.join("batch_summary.md").display(),
    );

    Ok(())
}

fn memory_looking_instruction_count(instructions: &[String]) -> usize {
    instructions
        .iter()
        .filter(|line| {
            let trimmed = line.trim_start();
            trimmed.contains('(') && trimmed.contains(')')
        })
        .count()
}

fn is_fof_instruction(line: &str) -> bool {
    line.trim_start()
        .split_whitespace()
        .next()
        .is_some_and(|mnemonic| mnemonic.starts_with("vl") && mnemonic.contains("ff.v"))
}

fn contains_fof_before_negative(source: &str) -> bool {
    source
        .lines()
        .take_while(|line| !line.contains("addi x0, x0, 1441"))
        .any(is_fof_instruction)
}

#[derive(Debug, Clone, Serialize)]
struct DirectDifftestBatchCase {
    index: usize,
    seed: u64,
    worker: usize,
    run_dir: String,
    status: String,
    result: Option<String>,
    goodtrap: Option<bool>,
    difftest_abort: Option<bool>,
    reference_model: Option<String>,
    error: Option<String>,
    fof_requires_spec_review: bool,
}

#[derive(Debug, Serialize)]
struct DirectDifftestBatchSummary {
    judgement: String,
    strict_mismatch_capable: bool,
    total_cases: usize,
    workers: usize,
    seed_start: u64,
    completed_cases: usize,
    passed: usize,
    non_pass: usize,
    mismatch_or_abort: usize,
    fof_requires_spec_review: usize,
    failed: usize,
    panicked_workers: usize,
    cases: Vec<DirectDifftestBatchCase>,
}

fn read_difftest_summary(run_dir: &Path) -> Option<DirectDifftestSummary> {
    let path = run_dir.join("difftest_result.json");
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_batch_summary_files(
    run_dir: &Path,
    summary: &DirectDifftestBatchSummary,
    judgement: DirectExecutionJudgement,
) -> Result<()> {
    let json_path = run_dir.join("batch_summary.json");
    write_json_pretty(&json_path, summary)
        .with_context(|| format!("failed to write {}", json_path.display()))?;

    let mut rows = String::new();
    for case in summary
        .cases
        .iter()
        .filter(|case| !judgement.case_is_pass(case))
    {
        rows.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            case.index,
            case.seed,
            case.worker,
            case.status,
            case.result.as_deref().unwrap_or("missing"),
            case.run_dir,
            case.fof_requires_spec_review,
            case.error.as_deref().unwrap_or(""),
        ));
    }
    if rows.is_empty() {
        rows.push_str("| - | - | - | - | - | - | - | - |\n");
    }

    let md = format!(
        "# Direct Execution Batch Summary\n\n- Judgement: `{}`\n- Strict mismatch capable: `{}`\n- Cases: `{}`\n- Completed cases: `{}`\n- Workers: `{}`\n- Seed start: `{}`\n- Passed: `{}`\n- Non-pass: `{}`\n- Mismatch/abort: `{}`\n- FOF cases requiring specification review: `{}`\n- Failed executions: `{}`\n- Panicked workers: `{}`\n\n## Non-pass Cases\n\n| Case | Seed | Worker | Status | Result | Run dir | FOF review | Error |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n{}",
        summary.judgement,
        summary.strict_mismatch_capable,
        summary.total_cases,
        summary.completed_cases,
        summary.workers,
        summary.seed_start,
        summary.passed,
        summary.non_pass,
        summary.mismatch_or_abort,
        summary.fof_requires_spec_review,
        summary.failed,
        summary.panicked_workers,
        rows,
    );
    let md_path = run_dir.join("batch_summary.md");
    write_string_file(&md_path, md)
        .with_context(|| format!("failed to write {}", md_path.display()))
}

#[derive(Debug, Serialize, Deserialize)]
struct DirectDifftestSummary {
    core: String,
    isa: String,
    #[serde(default)]
    judgement: String,
    #[serde(default)]
    strict_mismatch_capable: bool,
    result: String,
    #[serde(default)]
    fof_requires_spec_review: bool,
    goodtrap: bool,
    difftest_abort: bool,
    reference_model: Option<String>,
    commit_log: String,
    rows: usize,
    traceable_rows: usize,
    exception_rows: usize,
    global_cycles: Option<u64>,
    markers: Vec<String>,
    error: Option<String>,
}

fn write_difftest_success_summary(
    run_dir: &Path,
    core_id: &str,
    isa_base: ISABase,
    judgement: DirectExecutionJudgement,
    testcase: &riscv_fuzz_testcase_engine::testcase::GeneratedTestCase,
    execution_result: &riscv_fuzz_testcase_engine::execution_artifacts::ExecutionArtifactsRunResult,
) -> Result<()> {
    let output = execution_result
        .outputs
        .get(core_id)
        .ok_or_else(|| CliError::new(format!("missing execution output for {core_id}")))?;
    let rows = output
        .per_hart
        .values()
        .map(|hart| hart.instructions.len())
        .sum();
    let traceable_rows = output
        .per_hart
        .values()
        .flat_map(|hart| hart.instructions.iter())
        .filter(|inst| inst.trace_inst_id.is_some())
        .count();
    let exception_rows = output
        .per_hart
        .values()
        .flat_map(|hart| hart.instructions.iter())
        .filter(|inst| !inst.exceptions.is_empty())
        .count();

    let (result, goodtrap, difftest_abort, reference_model, commit_log, markers) =
        if judgement.uses_xiangshan_commit_log() {
            let commit_log_path = run_dir.join(core_id).join("skiptrap.commit.log");
            let (log_text, commit_log) = match fs::read_to_string(&commit_log_path) {
                Ok(log_text) => (log_text, commit_log_path.display().to_string()),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    (String::new(), String::new())
                }
                Err(err) => {
                    return Err(CliError::new(format!(
                        "failed to read {}: {err}",
                        commit_log_path.display()
                    )));
                }
            };
            let markers = difftest_abort_markers(&log_text);
            let goodtrap = log_text.contains("HIT GOOD TRAP");
            let difftest_abort = !markers.is_empty();
            let result = if difftest_abort {
                "mismatch_or_abort"
            } else if goodtrap {
                "pass"
            } else if log_text.is_empty() {
                "completed"
            } else {
                "completed_without_goodtrap_marker"
            };
            (
                result.to_string(),
                goodtrap,
                difftest_abort,
                reference_model_line(&log_text),
                commit_log,
                markers,
            )
        } else if judgement.uses_spike_reference() {
            match compare_with_spike_reference(testcase, execution_result, core_id) {
                Ok(()) => (
                    "completed".to_string(),
                    false,
                    false,
                    Some(SPIKE_REFERENCE_CORE_ID.to_string()),
                    String::new(),
                    Vec::new(),
                ),
                Err(marker) => (
                    "mismatch_or_abort".to_string(),
                    false,
                    true,
                    Some(SPIKE_REFERENCE_CORE_ID.to_string()),
                    String::new(),
                    vec![marker],
                ),
            }
        } else {
            (
                "completed".to_string(),
                false,
                false,
                None,
                String::new(),
                Vec::new(),
            )
        };

    let fof_requires_spec_review = result == "mismatch_or_abort"
        && fs::read_to_string(run_dir.join("user_instructions.asm"))
            .ok()
            .is_some_and(|source| contains_fof_before_negative(&source));
    let summary = DirectDifftestSummary {
        core: core_id.to_string(),
        isa: isa_base.to_string(),
        judgement: judgement.label().to_string(),
        strict_mismatch_capable: judgement.strict_mismatch_capable(),
        result,
        fof_requires_spec_review,
        goodtrap,
        difftest_abort,
        reference_model,
        commit_log,
        rows,
        traceable_rows,
        exception_rows,
        global_cycles: output.perf.global_cycles,
        markers,
        error: None,
    };
    write_difftest_summary_files(run_dir, &summary)
}

fn build_direct_diff_view(
    testcase: &riscv_fuzz_testcase_engine::testcase::GeneratedTestCase,
    execution_result: &riscv_fuzz_testcase_engine::execution_artifacts::ExecutionArtifactsRunResult,
    core_id: &str,
) -> std::result::Result<DiffExecutionView, String> {
    let output = execution_result
        .outputs
        .get(core_id)
        .ok_or_else(|| format!("missing execution output for {core_id}"))?
        .clone();
    let mem_range = testcase
        .mem_range
        .get(core_id)
        .copied()
        .ok_or_else(|| format!("missing mem_range for {core_id}"))?;
    let inputs = testcase
        .combined_typed_sequences_per_hart_of(core_id)
        .ok_or_else(|| format!("missing typed instruction sequences for {core_id}"))?;
    let context = match build_execution_context_output(output.clone(), mem_range, &inputs) {
        Ok(context) => context,
        Err(strict_error) => build_partial_execution_context_output(output, mem_range, &inputs)
            .map_err(|partial_error| {
                format!(
                    "failed to build logical execution view for {core_id}: strict={strict_error}; partial={partial_error}"
                )
            })?,
    };

    Ok(DiffExecutionView {
        logical_view: context.logical_view,
        mem_range: context.mem_range,
        core_id: context.core_id,
        isa_base: context.isa_base,
        perf: context.perf,
    })
}

fn instruction_mnemonic(instruction: &RiscvInstruction) -> String {
    instruction
        .to_string()
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn strip_vector_memory_writes_for_ara_reference(view: &mut DiffExecutionView) {
    for instruction in &mut view.logical_view {
        if instruction_mnemonic(&instruction.instruction).starts_with('v') {
            instruction.memory_writes.clear();
        }
    }
}

fn compare_with_spike_reference(
    testcase: &riscv_fuzz_testcase_engine::testcase::GeneratedTestCase,
    execution_result: &riscv_fuzz_testcase_engine::execution_artifacts::ExecutionArtifactsRunResult,
    core_id: &str,
) -> std::result::Result<(), String> {
    let mut reference =
        build_direct_diff_view(testcase, execution_result, SPIKE_REFERENCE_CORE_ID)?;
    let mut target = build_direct_diff_view(testcase, execution_result, core_id)?;
    strip_vector_memory_writes_for_ara_reference(&mut reference);
    strip_vector_memory_writes_for_ara_reference(&mut target);
    compare_execution_views(&reference, &target).map_err(|mismatch| {
        let location = mismatch
            .instruction_index
            .map(|index| format!(" at logical instruction #{index}"))
            .unwrap_or_default();
        format!(
            "ARA/Spike reference mismatch{location}: {}",
            mismatch.message
        )
    })
}

fn write_difftest_error_summary(
    run_dir: &Path,
    core_id: &str,
    isa_base: ISABase,
    judgement: DirectExecutionJudgement,
    error: &str,
) -> Result<()> {
    let (result, goodtrap, difftest_abort, reference_model, commit_log, markers) =
        if judgement.uses_xiangshan_commit_log() {
            let commit_log_path = run_dir.join(core_id).join("skiptrap.commit.log");
            let log_text = fs::read_to_string(&commit_log_path).unwrap_or_default();
            let markers = difftest_abort_markers(&log_text);
            let goodtrap = log_text.contains("HIT GOOD TRAP");
            let result = if !markers.is_empty() {
                "mismatch_or_abort"
            } else if goodtrap {
                "pass"
            } else {
                "execution_error"
            };
            (
                result.to_string(),
                goodtrap,
                !markers.is_empty(),
                reference_model_line(&log_text),
                commit_log_path.display().to_string(),
                markers,
            )
        } else {
            (
                judgement.classify_error_result(error).to_string(),
                false,
                false,
                None,
                String::new(),
                direct_execution_error_markers(error),
            )
        };

    let fof_requires_spec_review = result == "mismatch_or_abort"
        && fs::read_to_string(run_dir.join("user_instructions.asm"))
            .ok()
            .is_some_and(|source| contains_fof_before_negative(&source));
    let summary = DirectDifftestSummary {
        core: core_id.to_string(),
        isa: isa_base.to_string(),
        judgement: judgement.label().to_string(),
        strict_mismatch_capable: judgement.strict_mismatch_capable(),
        result,
        fof_requires_spec_review,
        goodtrap,
        difftest_abort,
        reference_model,
        commit_log,
        rows: 0,
        traceable_rows: 0,
        exception_rows: 0,
        global_cycles: None,
        markers,
        error: Some(error.to_string()),
    };
    write_difftest_summary_files(run_dir, &summary)
}

fn direct_execution_error_markers(error: &str) -> Vec<String> {
    error
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(4)
        .map(ToString::to_string)
        .collect()
}

fn commit_log_indicates_difftest_pass(run_dir: &Path, core_id: &str) -> bool {
    let commit_log_path = run_dir.join(core_id).join("skiptrap.commit.log");
    let Ok(log_text) = fs::read_to_string(commit_log_path) else {
        return false;
    };
    log_text.contains("HIT GOOD TRAP") && difftest_abort_markers(&log_text).is_empty()
}

fn difftest_abort_markers(log_text: &str) -> Vec<String> {
    log_text
        .lines()
        .filter(|line| {
            line.contains(" different at pc = ")
                || line.contains("ABORT at pc =")
                || line.contains("HIT BAD TRAP")
        })
        .take(16)
        .map(|line| line.trim().to_string())
        .collect()
}

fn reference_model_line(log_text: &str) -> Option<String> {
    log_text
        .lines()
        .find(|line| line.contains("The reference model is"))
        .map(|line| line.trim().to_string())
}

fn write_difftest_summary_files(run_dir: &Path, summary: &DirectDifftestSummary) -> Result<()> {
    let json_path = run_dir.join("difftest_result.json");
    write_json_pretty(&json_path, summary)
        .with_context(|| format!("failed to write {}", json_path.display()))?;

    let markers = if summary.markers.is_empty() {
        "- none\n".to_string()
    } else {
        summary
            .markers
            .iter()
            .map(|marker| format!("- `{marker}`\n"))
            .collect()
    };
    let reference_model = summary.reference_model.as_deref().unwrap_or("not reported");
    let error = summary
        .error
        .as_ref()
        .map(|error| format!("\n- Error: `{error}`"))
        .unwrap_or_default();
    let md = format!(
        "# Direct Execution Result\n\n- Result: `{}`\n- FOF specification review required: `{}`\n- Core: `{}`\n- ISA: `{}`\n- Judgement: `{}`\n- Strict mismatch capable: `{}`\n- Good trap: `{}`\n- DiffTest abort/mismatch marker: `{}`\n- Reference model: `{}`\n- Commit log: `{}`\n- Rows: `{}`\n- Traceable rows: `{}`\n- Exception rows: `{}`\n- Global cycles: `{}`{}\n\n## Markers\n{}",
        summary.result,
        summary.fof_requires_spec_review,
        summary.core,
        summary.isa,
        summary.judgement,
        summary.strict_mismatch_capable,
        summary.goodtrap,
        summary.difftest_abort,
        reference_model,
        summary.commit_log,
        summary.rows,
        summary.traceable_rows,
        summary.exception_rows,
        summary
            .global_cycles
            .map(|cycles| cycles.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        error,
        markers,
    );
    let md_path = run_dir.join("difftest_result.md");
    write_string_file(&md_path, md)
        .with_context(|| format!("failed to write {}", md_path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fault_only_first_cases_are_identified_for_review() {
        assert!(is_fof_instruction("vle32ff.v v8, (x11)"));
        assert!(is_fof_instruction("vlseg2e8ff.v v8, (x11)"));
        assert!(!is_fof_instruction("vle32.v v8, (x11)"));
        assert!(contains_fof_before_negative(
            "vle32ff.v v8, (x11)\naddi x0, x0, 1441"
        ));
        assert!(!contains_fof_before_negative(
            "addi x0, x0, 1441\nvle32ff.v v8, (x11)"
        ));
    }

    fn batch_case(status: &str, result: Option<&str>) -> DirectDifftestBatchCase {
        DirectDifftestBatchCase {
            index: 0,
            seed: 0,
            worker: 0,
            run_dir: String::new(),
            status: status.to_string(),
            result: result.map(ToString::to_string),
            goodtrap: None,
            difftest_abort: None,
            reference_model: None,
            error: None,
            fof_requires_spec_review: false,
        }
    }

    #[test]
    fn ara_default_hardware_uses_risk_generator() {
        assert_eq!(
            effective_generator_for_core("ara", ExecGeneratedGenerator::Hardware).unwrap(),
            ExecGeneratedGenerator::HardwareAraRisk
        );
        assert_eq!(
            effective_generator_for_core("xiangshan", ExecGeneratedGenerator::Hardware).unwrap(),
            ExecGeneratedGenerator::Hardware
        );
        assert_eq!(
            effective_generator_for_core("ara", ExecGeneratedGenerator::HardwareRvvRandom).unwrap(),
            ExecGeneratedGenerator::HardwareRvvRandom
        );
        for ara_only in [
            ExecGeneratedGenerator::HardwareAraRisk,
            ExecGeneratedGenerator::HardwareAraRiskNoConstraints,
            ExecGeneratedGenerator::HardwareAraRiskNoSequence,
            ExecGeneratedGenerator::HardwareAraRiskNoConstraintsNoSequence,
            ExecGeneratedGenerator::HardwareAraSafe,
        ] {
            assert!(effective_generator_for_core("xiangshan", ara_only).is_err());
        }
    }

    #[test]
    fn ara_completed_result_is_successful_spike_referenced_execution() {
        let judgement = DirectExecutionJudgement::for_core("ara");
        assert_eq!(judgement.label(), "ara_vs_spike_execution_trace");
        assert!(judgement.strict_mismatch_capable());
        assert_eq!(
            direct_execution_core_ids("ara", judgement),
            vec!["ara".to_string(), "spike".to_string()]
        );
        assert!(judgement.case_is_pass(&batch_case("completed", Some("completed"))));
        assert!(!judgement.case_is_pass(&batch_case("failed", Some("completed"))));
        assert!(!judgement.case_is_pass(&batch_case("completed", Some("timeout"))));
    }

    #[test]
    fn xiangshan_keeps_difftest_pass_semantics() {
        let judgement = DirectExecutionJudgement::for_core("xiangshan");
        assert_eq!(judgement.label(), "xiangshan_difftest_commit_log");
        assert!(judgement.strict_mismatch_capable());
        assert!(judgement.case_is_pass(&batch_case("completed", Some("pass"))));
        assert!(!judgement.case_is_pass(&batch_case("completed", Some("completed"))));
    }

    #[test]
    fn ara_execution_errors_are_classified_without_xiangshan_markers() {
        let judgement = DirectExecutionJudgement::for_core("ara");
        assert_eq!(
            judgement.classify_error_result("process timed out after 90s"),
            "timeout"
        );
        assert_eq!(
            judgement.classify_error_result("invalid line format: ARA RVFI memory mask"),
            "trace_parse_error"
        );
        assert_eq!(
            judgement.classify_error_result("detected mismatch in trace"),
            "mismatch_or_abort"
        );
    }
}
