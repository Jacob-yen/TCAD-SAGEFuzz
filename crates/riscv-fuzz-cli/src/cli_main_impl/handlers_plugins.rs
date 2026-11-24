use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use riscv_fuzz_plugin_api::{
    CorePluginMetadata, CoverageModeSupport, FloatExtension, ISABase, IsaExecutionProfile,
    IsaExtensions, PluginDoctorCheck, PluginDoctorStatus, UnalignedAccessSupport, coverage,
    doctor as plugin_doctor,
};
use riscv_fuzz_plugin_manager::{
    PluginDiscoveryCandidate, PluginDiscoveryReport, PluginDiscoverySource, PluginManager,
    RegisteredPlugin,
};
use riscv_fuzz_toolchain::write_string_file;

use crate::cli_main_impl::{CliError, Result, ResultExt};

pub(super) fn run_plugin_catalog_mode(output: Option<PathBuf>) -> Result<()> {
    let plugins = PluginManager::load()
        .map_err(CliError::new)?
        .registered_plugins();
    let markdown = render_plugin_catalog_markdown(&plugins);
    write_markdown_report(
        output,
        &markdown,
        "plugin catalog",
        Some(format!(
            "Wrote plugin catalog for {} plugins to {{path}}",
            plugins.len()
        )),
    )
}

pub(super) fn run_plugin_doctor_mode(output: Option<PathBuf>) -> Result<()> {
    let discovery = PluginManager::discover();
    let load_state = match PluginManager::from_discovery(discovery.clone()) {
        Ok(manager) => PluginRegistryLoadState::Loaded(manager.registered_plugins()),
        Err(error) => PluginRegistryLoadState::Failed(error),
    };
    let markdown = render_plugin_doctor_markdown(&discovery, &load_state);
    write_markdown_report(
        output,
        &markdown,
        "plugin doctor report",
        Some("Wrote plugin doctor report to {path}".to_string()),
    )
}

fn write_markdown_report(
    output: Option<PathBuf>,
    markdown: &str,
    report_name: &str,
    success_message: Option<String>,
) -> Result<()> {
    if let Some(output) = output {
        write_string_file(&output, markdown)
            .with_context(|| format!("failed to write {} to {}", report_name, output.display()))?;

        if let Some(message) = success_message {
            println!(
                "{}",
                message.replace("{path}", &output.display().to_string())
            );
        }
    } else {
        print!("{markdown}");
    }

    Ok(())
}

fn render_plugin_catalog_markdown(plugins: &[RegisteredPlugin]) -> String {
    let mut out = String::new();
    writeln!(&mut out, "# Loaded RISC-V Plugin Catalog").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "Found **{}** dynamically loaded plugin{}.",
        plugins.len(),
        if plugins.len() == 1 { "" } else { "s" }
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "> This catalog reflects the metadata returned by the **actually loaded shared libraries**, not unpublished source changes. Check each plugin's `Source Path` to see which artifact won discovery."
    )
    .unwrap();
    writeln!(&mut out).unwrap();

    if plugins.is_empty() {
        writeln!(&mut out, "No plugins were discovered.").unwrap();
        return out;
    }

    writeln!(&mut out, "## Summary").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "| Plugin | Library | Program Memory | RV32 | RV64 |"
    )
    .unwrap();
    writeln!(&mut out, "| --- | --- | --- | --- | --- |").unwrap();
    for plugin in plugins {
        writeln!(
            &mut out,
            "| `{}` | `{}` | {} | {} | {} |",
            escape_table_cell(&plugin.core_id),
            escape_table_cell(&display_file_name(&plugin.source_path)),
            escape_table_cell(&format_mem_range(plugin.metadata.program_mem_range)),
            escape_table_cell(&profile_summary(
                plugin.metadata.isa_profiles.for_isa(ISABase::Rv32)
            )),
            escape_table_cell(&profile_summary(
                plugin.metadata.isa_profiles.for_isa(ISABase::Rv64)
            )),
        )
        .unwrap();
    }
    writeln!(&mut out).unwrap();

    writeln!(&mut out, "## ISA Capability Matrix").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "| Plugin | ISA | Supported | Dual Hart | Unaligned | Float | Float Select | Coverage | Extensions | Vector |"
    )
    .unwrap();
    writeln!(
        &mut out,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    .unwrap();
    for plugin in plugins {
        for isa in [ISABase::Rv32, ISABase::Rv64] {
            let profile = plugin.metadata.isa_profiles.for_isa(isa);
            writeln!(
                &mut out,
                "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} |",
                escape_table_cell(&plugin.core_id),
                escape_table_cell(isa_label(isa)),
                yes_no(profile.supported),
                yes_no(profile.supported && profile.supports_dual_hart),
                escape_table_cell(&format_unaligned_support(profile.unaligned_access_support)),
                escape_table_cell(&format_float_support(profile.max_float_extension())),
                yes_no(profile.supported && profile_supports_float_selection(profile)),
                escape_table_cell(&format_coverage_modes(profile.coverage_mode_support)),
                escape_table_cell(&format_extension_count(&profile.extensions)),
                yes_no(profile.supported && profile.extensions.has_enabled_vector()),
            )
            .unwrap();
        }
    }
    writeln!(&mut out).unwrap();

    for plugin in plugins {
        render_plugin_section(&mut out, plugin);
    }

    out
}

#[derive(Debug, Clone)]
enum PluginRegistryLoadState {
    Loaded(Vec<RegisteredPlugin>),
    Failed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DoctorSeverity {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone)]
struct DoctorFinding {
    severity: DoctorSeverity,
    message: String,
}

#[derive(Debug, Clone)]
struct PluginRuntimeDoctorReport {
    plugin_id: String,
    overall_status: DoctorSeverity,
    checks: Vec<PluginRuntimeDoctorCheck>,
}

#[derive(Debug, Clone)]
struct PluginRuntimeDoctorCheck {
    label: String,
    env_var: String,
    status: DoctorSeverity,
    value: String,
    note: String,
}

fn render_plugin_doctor_markdown(
    discovery: &PluginDiscoveryReport,
    load_state: &PluginRegistryLoadState,
) -> String {
    let mut out = String::new();
    let runtime_reports = build_plugin_runtime_reports(load_state);
    let findings = collect_doctor_findings(discovery, load_state, &runtime_reports);
    let overall_status = findings
        .iter()
        .map(|finding| finding.severity)
        .max()
        .unwrap_or(DoctorSeverity::Pass);
    let selected_candidate_count = discovery.candidates.iter().filter(|c| c.selected).count();
    let loaded_plugin_count = match load_state {
        PluginRegistryLoadState::Loaded(plugins) => Some(plugins.len()),
        PluginRegistryLoadState::Failed(_) => None,
    };

    writeln!(&mut out, "# RISC-V Plugin Doctor").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "Inspects the **same discovery rules** used by `riscv_fuzz_test` before the plugin registry is built."
    )
    .unwrap();
    writeln!(&mut out).unwrap();

    writeln!(&mut out, "## Summary").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "| Check | Value |").unwrap();
    writeln!(&mut out, "| --- | --- |").unwrap();
    writeln!(
        &mut out,
        "| Overall Status | `{}` |",
        doctor_severity_label(overall_status)
    )
    .unwrap();
    writeln!(
        &mut out,
        "| Effective Search Directories | {} |",
        discovery.configured_dirs.len()
    )
    .unwrap();
    writeln!(
        &mut out,
        "| Candidate Entries | {} |",
        discovery.candidates.len()
    )
    .unwrap();
    writeln!(
        &mut out,
        "| Selected Candidates | {} |",
        selected_candidate_count
    )
    .unwrap();
    writeln!(
        &mut out,
        "| Loaded Plugins | {} |",
        loaded_plugin_count
            .map(|count| count.to_string())
            .unwrap_or_else(|| "registry load failed".to_string())
    )
    .unwrap();
    writeln!(&mut out).unwrap();

    writeln!(&mut out, "## Environment").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "| Variable | Value |").unwrap();
    writeln!(&mut out, "| --- | --- |").unwrap();
    writeln!(
        &mut out,
        "| `RISCV_FUZZ_PLUGIN_DIRS` | {} |",
        markdown_table_value(discovery.plugin_dirs_env.as_deref())
    )
    .unwrap();
    writeln!(
        &mut out,
        "| `RISCV_FUZZ_PLUGIN_FILES` | {} |",
        markdown_table_value(discovery.plugin_files_env.as_deref())
    )
    .unwrap();
    writeln!(&mut out).unwrap();

    if discovery.plugin_dirs_env.is_none() {
        writeln!(
            &mut out,
            "> `RISCV_FUZZ_PLUGIN_DIRS` is unset, so discovery falls back to the built-in directories: {}.",
            discovery
                .configured_dirs
                .iter()
                .map(|scan| markdown_code(&scan.path.display().to_string()))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(&mut out).unwrap();
    }

    writeln!(&mut out, "## Effective Search Directories").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "| Directory | Exists | Candidate Entries | Notes |"
    )
    .unwrap();
    writeln!(&mut out, "| --- | --- | --- | --- |").unwrap();
    for scan in &discovery.configured_dirs {
        writeln!(
            &mut out,
            "| {} | {} | {} | {} |",
            markdown_code(&scan.path.display().to_string()),
            yes_no(scan.exists),
            scan.candidate_entries,
            escape_table_cell(scan.read_error.as_deref().unwrap_or("-")),
        )
        .unwrap();
    }
    writeln!(&mut out).unwrap();

    writeln!(&mut out, "## Candidate Libraries").unwrap();
    writeln!(&mut out).unwrap();
    if discovery.candidates.is_empty() {
        writeln!(
            &mut out,
            "No explicit file entries or directory candidates were discovered."
        )
        .unwrap();
        writeln!(&mut out).unwrap();
    } else {
        writeln!(
            &mut out,
            "| Source | Path | Stem | Selected | Matches Dir Heuristic | Notes |"
        )
        .unwrap();
        writeln!(&mut out, "| --- | --- | --- | --- | --- | --- |").unwrap();
        for candidate in &discovery.candidates {
            writeln!(
                &mut out,
                "| `{}` | {} | {} | {} | {} | {} |",
                discovery_source_label(candidate.source),
                markdown_code(&candidate.path.display().to_string()),
                candidate
                    .stem
                    .as_deref()
                    .map(markdown_code)
                    .unwrap_or_else(|| "-".to_string()),
                yes_no(candidate.selected),
                yes_no(candidate.matches_directory_naming_heuristic),
                escape_table_cell(&candidate_notes(candidate)),
            )
            .unwrap();
        }
        writeln!(&mut out).unwrap();
    }

    writeln!(&mut out, "## Registry Load").unwrap();
    writeln!(&mut out).unwrap();
    match load_state {
        PluginRegistryLoadState::Loaded(plugins) => {
            writeln!(
                &mut out,
                "Successfully loaded **{}** plugin{}.",
                plugins.len(),
                if plugins.len() == 1 { "" } else { "s" }
            )
            .unwrap();
            writeln!(&mut out).unwrap();
            if plugins.is_empty() {
                writeln!(&mut out, "No plugins were registered.").unwrap();
                writeln!(&mut out).unwrap();
            } else {
                writeln!(&mut out, "| Impl ID | Source Path | RV32 | RV64 |").unwrap();
                writeln!(&mut out, "| --- | --- | --- | --- |").unwrap();
                for plugin in plugins {
                    writeln!(
                        &mut out,
                        "| `{}` | {} | {} | {} |",
                        escape_table_cell(&plugin.core_id),
                        markdown_code(&plugin.source_path.display().to_string()),
                        escape_table_cell(&profile_summary(
                            plugin.metadata.isa_profiles.for_isa(ISABase::Rv32)
                        )),
                        escape_table_cell(&profile_summary(
                            plugin.metadata.isa_profiles.for_isa(ISABase::Rv64)
                        )),
                    )
                    .unwrap();
                }
                writeln!(&mut out).unwrap();
            }
        }
        PluginRegistryLoadState::Failed(error) => {
            writeln!(&mut out, "Registry load failed.").unwrap();
            writeln!(&mut out).unwrap();
            writeln!(&mut out, "```text").unwrap();
            write!(&mut out, "{}", error.trim_end()).unwrap();
            writeln!(&mut out).unwrap();
            writeln!(&mut out, "```").unwrap();
            writeln!(&mut out).unwrap();
        }
    }

    writeln!(&mut out, "## Plugin Runtime Environment").unwrap();
    writeln!(&mut out).unwrap();
    if runtime_reports.is_empty() {
        writeln!(
            &mut out,
            "Skipped plugin runtime environment checks because the plugin registry did not load successfully."
        )
        .unwrap();
        writeln!(&mut out).unwrap();
    } else {
        writeln!(
            &mut out,
            "| Plugin | Status | Checked Env Vars | Failing Checks |"
        )
        .unwrap();
        writeln!(&mut out, "| --- | --- | --- | --- |").unwrap();
        for report in &runtime_reports {
            let failing_checks = report
                .checks
                .iter()
                .filter(|check| check.status == DoctorSeverity::Fail)
                .count();
            writeln!(
                &mut out,
                "| `{}` | `{}` | {} | {} |",
                escape_table_cell(&report.plugin_id),
                doctor_severity_label(report.overall_status),
                report.checks.len(),
                failing_checks,
            )
            .unwrap();
        }
        writeln!(&mut out).unwrap();

        for report in &runtime_reports {
            writeln!(&mut out, "### `{}`", report.plugin_id).unwrap();
            writeln!(&mut out).unwrap();
            writeln!(
                &mut out,
                "| Requirement | Env Var | Status | Value | Notes |"
            )
            .unwrap();
            writeln!(&mut out, "| --- | --- | --- | --- | --- |").unwrap();
            for check in &report.checks {
                writeln!(
                    &mut out,
                    "| {} | `{}` | `{}` | {} | {} |",
                    escape_table_cell(&check.label),
                    escape_table_cell(&check.env_var),
                    doctor_severity_label(check.status),
                    markdown_code(&check.value),
                    escape_table_cell(&check.note),
                )
                .unwrap();
            }
            writeln!(&mut out).unwrap();
        }
    }

    writeln!(&mut out, "## Findings").unwrap();
    writeln!(&mut out).unwrap();
    if findings.is_empty() {
        writeln!(
            &mut out,
            "- PASS: Plugin discovery and registry load look healthy."
        )
        .unwrap();
    } else {
        for finding in findings {
            writeln!(
                &mut out,
                "- {}: {}",
                doctor_severity_label(finding.severity),
                finding.message
            )
            .unwrap();
        }
    }
    writeln!(&mut out).unwrap();

    out
}

fn collect_doctor_findings(
    discovery: &PluginDiscoveryReport,
    load_state: &PluginRegistryLoadState,
    runtime_reports: &[PluginRuntimeDoctorReport],
) -> Vec<DoctorFinding> {
    let mut findings = Vec::new();
    let selected_candidate_count = discovery.candidates.iter().filter(|c| c.selected).count();

    if discovery.plugin_dirs_env.is_some() {
        for scan in &discovery.configured_dirs {
            if !scan.exists {
                findings.push(DoctorFinding {
                    severity: DoctorSeverity::Warn,
                    message: format!(
                        "configured search directory {} does not exist",
                        scan.path.display()
                    ),
                });
            }
        }
    }

    for scan in &discovery.configured_dirs {
        if let Some(error) = &scan.read_error {
            findings.push(DoctorFinding {
                severity: DoctorSeverity::Warn,
                message: format!(
                    "failed to scan search directory {}: {}",
                    scan.path.display(),
                    error
                ),
            });
        }
    }

    for candidate in &discovery.candidates {
        if !candidate.selected
            && let Some(reason) = &candidate.skip_reason
        {
            findings.push(DoctorFinding {
                severity: DoctorSeverity::Warn,
                message: format!(
                    "candidate {} was skipped: {}",
                    candidate.path.display(),
                    reason
                ),
            });
        }
    }

    if selected_candidate_count == 0 {
        findings.push(DoctorFinding {
            severity: DoctorSeverity::Warn,
            message: "no plugin candidates survived discovery; set RISCV_FUZZ_PLUGIN_DIRS / RISCV_FUZZ_PLUGIN_FILES or build plugin shared libraries first".to_string(),
        });
    }

    match load_state {
        PluginRegistryLoadState::Loaded(plugins) => {
            if plugins.is_empty() {
                findings.push(DoctorFinding {
                    severity: DoctorSeverity::Warn,
                    message: "plugin registry loaded successfully but no plugins were registered"
                        .to_string(),
                });
            }
        }
        PluginRegistryLoadState::Failed(error) => {
            findings.push(DoctorFinding {
                severity: DoctorSeverity::Fail,
                message: format!("plugin registry load failed: {}", first_line(error)),
            });
        }
    }

    for report in runtime_reports {
        for check in &report.checks {
            if check.status == DoctorSeverity::Pass {
                continue;
            }
            findings.push(DoctorFinding {
                severity: check.status,
                message: format!(
                    "plugin {} requirement `{}` via {}: {}",
                    report.plugin_id, check.label, check.env_var, check.note
                ),
            });
        }
    }

    findings
}

fn doctor_severity_label(severity: DoctorSeverity) -> &'static str {
    match severity {
        DoctorSeverity::Pass => "PASS",
        DoctorSeverity::Warn => "WARN",
        DoctorSeverity::Fail => "FAIL",
    }
}

fn discovery_source_label(source: PluginDiscoverySource) -> &'static str {
    match source {
        PluginDiscoverySource::ExplicitFileEnv => "explicit-file-env",
        PluginDiscoverySource::SearchDirectory => "search-directory",
    }
}

fn candidate_notes(candidate: &PluginDiscoveryCandidate) -> String {
    if let Some(reason) = &candidate.skip_reason {
        return reason.clone();
    }

    if candidate.source == PluginDiscoverySource::ExplicitFileEnv
        && !candidate.matches_directory_naming_heuristic
    {
        return "selected from RISCV_FUZZ_PLUGIN_FILES; explicit file paths bypass directory filename filtering".to_string();
    }

    "-".to_string()
}

fn markdown_table_value(value: Option<&str>) -> String {
    value
        .map(markdown_code)
        .unwrap_or_else(|| "_unset_".to_string())
}

fn markdown_code(value: &str) -> String {
    format!(
        "`{}`",
        escape_table_cell(&value.replace('`', "\\`")).replace('\r', "")
    )
}

fn first_line(value: &str) -> &str {
    value.lines().next().unwrap_or(value)
}

#[derive(Debug, Clone, Copy)]
enum ExecutableValidation {
    None,
    VexMulticoreTraceMarkers { require_freg_trace: bool },
}

fn build_plugin_runtime_reports(
    load_state: &PluginRegistryLoadState,
) -> Vec<PluginRuntimeDoctorReport> {
    match load_state {
        PluginRegistryLoadState::Loaded(plugins) => {
            plugins.iter().map(build_plugin_runtime_report).collect()
        }
        PluginRegistryLoadState::Failed(_) => Vec::new(),
    }
}

fn build_plugin_runtime_report(plugin: &RegisteredPlugin) -> PluginRuntimeDoctorReport {
    let checks = match plugin.core_id.as_str() {
        "spike" => spike_runtime_checks(plugin),
        "picorv32" => picorv32_runtime_checks(plugin),
        "ibex" => ibex_runtime_checks(plugin),
        "kronos" => kronos_runtime_checks(plugin),
        "vex" => vex_runtime_checks(plugin),
        "rocket" => rocket_runtime_checks(plugin),
        "cva6" => cva6_runtime_checks(plugin),
        "xiangshan" => xiangshan_runtime_checks(plugin),
        _ => plugin
            .doctor_report
            .checks
            .iter()
            .map(convert_plugin_doctor_check)
            .collect(),
    };
    PluginRuntimeDoctorReport {
        plugin_id: plugin.core_id.clone(),
        overall_status: checks
            .iter()
            .map(|check| check.status)
            .max()
            .unwrap_or(DoctorSeverity::Pass),
        checks,
    }
}

fn convert_plugin_doctor_check(check: &PluginDoctorCheck) -> PluginRuntimeDoctorCheck {
    PluginRuntimeDoctorCheck {
        label: check.label.clone(),
        env_var: check.env_var.clone().unwrap_or_else(|| "-".to_string()),
        status: convert_plugin_doctor_status(check.status),
        value: check.value.clone(),
        note: check.note.clone(),
    }
}

fn convert_plugin_doctor_status(status: PluginDoctorStatus) -> DoctorSeverity {
    match status {
        PluginDoctorStatus::Pass => DoctorSeverity::Pass,
        PluginDoctorStatus::Warn => DoctorSeverity::Warn,
        PluginDoctorStatus::Fail => DoctorSeverity::Fail,
    }
}

fn spike_runtime_checks(_plugin: &RegisteredPlugin) -> Vec<PluginRuntimeDoctorCheck> {
    vec![required_executable_path_or_command_check(
        "shared Spike executable for all supported ISA/hart combinations",
        "CX_RISCV_CORES_SPIKE",
        "Spike accepts either an absolute/relative executable path or a command name resolvable from PATH",
    )]
}

fn picorv32_runtime_checks(plugin: &RegisteredPlugin) -> Vec<PluginRuntimeDoctorCheck> {
    let mut checks = Vec::new();
    let rv32 = plugin.metadata.isa_profiles.for_isa(ISABase::Rv32);
    for mode in supported_coverage_modes(rv32) {
        checks.push(required_executable_path_check(
            format!("rv32 1-hart {}", coverage_requirement_label(mode)),
            picorv32_binary_env_var(false, mode),
            "PicoRV32 source expects a concrete binary path in the environment",
            ExecutableValidation::None,
        ));
        if rv32.supports_dual_hart {
            checks.push(required_executable_path_check(
                format!("rv32 2-hart {}", coverage_requirement_label(mode)),
                picorv32_binary_env_var(true, mode),
                "PicoRV32 dual-hart execution uses a dedicated 2-core wrapper binary",
                ExecutableValidation::None,
            ));
        }
    }
    checks
}

fn ibex_runtime_checks(plugin: &RegisteredPlugin) -> Vec<PluginRuntimeDoctorCheck> {
    let mut checks = Vec::new();
    let rv32 = plugin.metadata.isa_profiles.for_isa(ISABase::Rv32);
    for mode in supported_coverage_modes(rv32) {
        checks.push(required_executable_path_check(
            format!("rv32 1-hart {}", coverage_requirement_label(mode)),
            ibex_binary_env_var(false, mode),
            "Ibex source expects a concrete binary path in the environment",
            ExecutableValidation::None,
        ));
        if rv32.supports_dual_hart {
            checks.push(required_executable_path_check(
                format!("rv32 2-hart {}", coverage_requirement_label(mode)),
                ibex_binary_env_var(true, mode),
                "Ibex dual-hart execution uses a dedicated 2-core wrapper binary",
                ExecutableValidation::None,
            ));
        }
    }
    checks
}

fn kronos_runtime_checks(plugin: &RegisteredPlugin) -> Vec<PluginRuntimeDoctorCheck> {
    let mut checks = Vec::new();
    let rv32 = plugin.metadata.isa_profiles.for_isa(ISABase::Rv32);
    for mode in supported_coverage_modes(rv32) {
        checks.push(required_executable_path_check(
            format!("rv32 1-hart {}", coverage_requirement_label(mode)),
            kronos_binary_env_var(false, mode),
            "Kronos source expects a concrete binary path in the environment",
            ExecutableValidation::None,
        ));
        if rv32.supports_dual_hart {
            checks.push(required_executable_path_check(
                format!("rv32 2-hart {}", coverage_requirement_label(mode)),
                kronos_binary_env_var(true, mode),
                "Kronos dual-hart execution uses a dedicated 2-core wrapper binary",
                ExecutableValidation::None,
            ));
        }
    }
    checks
}

fn vex_runtime_checks(plugin: &RegisteredPlugin) -> Vec<PluginRuntimeDoctorCheck> {
    let mut checks = Vec::new();
    let rv32 = plugin.metadata.isa_profiles.for_isa(ISABase::Rv32);
    let need_no_float_family = rv32.supports_float_choice(FloatExtension::None);
    let need_f_family = rv32.supports_float_choice(FloatExtension::F);
    let need_fd_family = rv32.supports_float_choice(FloatExtension::FD);

    for mode in supported_coverage_modes(rv32) {
        if need_no_float_family {
            checks.push(required_executable_path_check(
                format!(
                    "rv32 1-hart {} no-float profile",
                    coverage_requirement_label(mode)
                ),
                vex_binary_env_var(false, mode, VexBinaryFamily::NoFloat),
                "Vex uses a dedicated single-hart RV32 wrapper when the resolved extensions disable F and D",
                ExecutableValidation::None,
            ));
        }
        if need_f_family {
            checks.push(required_executable_path_check(
                format!(
                    "rv32 1-hart {} f-only float profile",
                    coverage_requirement_label(mode)
                ),
                vex_binary_env_var(false, mode, VexBinaryFamily::FOnly),
                "Vex uses a dedicated single-hart F-only wrapper when the resolved extensions enable F but not D",
                ExecutableValidation::None,
            ));
        }
        if need_fd_family {
            checks.push(required_executable_path_check(
                format!(
                    "rv32 1-hart {} fd float profile",
                    coverage_requirement_label(mode)
                ),
                vex_binary_env_var(false, mode, VexBinaryFamily::FD),
                "Vex uses a dedicated single-hart RV32FD wrapper when the resolved extensions include D",
                ExecutableValidation::None,
            ));
        }

        if rv32.supports_dual_hart {
            if need_no_float_family {
                checks.push(required_executable_path_check(
                    format!(
                        "rv32 2-hart {} no-float profile",
                        coverage_requirement_label(mode)
                    ),
                    vex_binary_env_var(true, mode, VexBinaryFamily::NoFloat),
                    "Vex multicore no-float wrapper must embed run.regTrace/run.memTrace/run.logTrace markers",
                    ExecutableValidation::VexMulticoreTraceMarkers {
                        require_freg_trace: false,
                    },
                ));
            }
            if need_fd_family {
                checks.push(required_executable_path_check(
                    format!(
                        "rv32 2-hart {} fd float profile",
                        coverage_requirement_label(mode)
                    ),
                    vex_binary_env_var(true, mode, VexBinaryFamily::FD),
                    "Vex multicore FD wrapper must embed run.regTrace/run.fregTrace/run.memTrace/run.logTrace markers",
                    ExecutableValidation::VexMulticoreTraceMarkers {
                        require_freg_trace: true,
                    },
                ));
            }
            if need_f_family {
                checks.push(warn_runtime_check(
                    format!(
                        "rv32 2-hart {} f-only float profile",
                        coverage_requirement_label(mode)
                    ),
                    "-",
                    "unsupported",
                    "VexRiscv SMP provider only ships rv32 and rv32fd wrappers; the plugin returns an explicit error for 2-hart F-only requests instead of falling back",
                ));
            }
        }
    }

    append_timeout_override_checks(&mut checks);
    checks
}

fn rocket_runtime_checks(plugin: &RegisteredPlugin) -> Vec<PluginRuntimeDoctorCheck> {
    let mut checks = Vec::new();

    for isa in [ISABase::Rv32, ISABase::Rv64] {
        let profile = plugin.metadata.isa_profiles.for_isa(isa);
        if !profile.supported {
            continue;
        }

        let need_int_family = profile.supports_float_choice(FloatExtension::None);
        let need_f_family = profile.supports_float_choice(FloatExtension::F);
        let need_fd_family = profile.supports_float_choice(FloatExtension::FD);

        for mode in supported_coverage_modes(profile) {
            if need_int_family {
                checks.push(required_executable_path_check(
                    format!(
                        "{} 1-hart {} no-float profile",
                        isa_label(isa),
                        coverage_requirement_label(mode)
                    ),
                    rocket_binary_env_var(isa, false, mode, RocketBinaryFamily::Integer),
                    "Rocket uses a dedicated single-hart integer/no-float wrapper when the resolved extensions disable F and D",
                    ExecutableValidation::None,
                ));
            }
            if need_f_family {
                checks.push(required_executable_path_check(
                    format!(
                        "{} 1-hart {} f-only float profile",
                        isa_label(isa),
                        coverage_requirement_label(mode)
                    ),
                    rocket_binary_env_var(isa, false, mode, RocketBinaryFamily::SingleFloat),
                    "Rocket uses a dedicated single-hart F-only wrapper when the resolved extensions enable F but not D",
                    ExecutableValidation::None,
                ));
            }

            if need_fd_family {
                checks.push(required_executable_path_check(
                    format!(
                        "{} 1-hart {} fd float profile",
                        isa_label(isa),
                        coverage_requirement_label(mode)
                    ),
                    rocket_binary_env_var(isa, false, mode, RocketBinaryFamily::DoubleFloat),
                    "Rocket uses a dedicated single-hart FD wrapper when the resolved extensions include D",
                    ExecutableValidation::None,
                ));
            }

            if profile.supports_dual_hart {
                if need_int_family {
                    checks.push(required_executable_path_check(
                        format!(
                            "{} 2-hart {} no-float profile",
                            isa_label(isa),
                            coverage_requirement_label(mode)
                        ),
                        rocket_binary_env_var(isa, true, mode, RocketBinaryFamily::Integer),
                        "Rocket uses a dedicated dual-hart integer/no-float wrapper when the resolved extensions disable F and D",
                        ExecutableValidation::None,
                    ));
                }
                if need_f_family {
                    checks.push(required_executable_path_check(
                        format!(
                            "{} 2-hart {} f-only float profile",
                            isa_label(isa),
                            coverage_requirement_label(mode)
                        ),
                        rocket_binary_env_var(isa, true, mode, RocketBinaryFamily::SingleFloat),
                        "Rocket uses a dedicated dual-hart F-only wrapper when the resolved extensions enable F but not D",
                        ExecutableValidation::None,
                    ));
                }

                if need_fd_family {
                    checks.push(required_executable_path_check(
                        format!(
                            "{} 2-hart {} fd float profile",
                            isa_label(isa),
                            coverage_requirement_label(mode)
                        ),
                        rocket_binary_env_var(isa, true, mode, RocketBinaryFamily::DoubleFloat),
                        "Rocket uses a dedicated dual-hart FD wrapper when the resolved extensions include D",
                        ExecutableValidation::None,
                    ));
                }
            }
        }
    }

    append_timeout_override_checks(&mut checks);
    checks
}

fn cva6_runtime_checks(plugin: &RegisteredPlugin) -> Vec<PluginRuntimeDoctorCheck> {
    let mut checks = Vec::new();

    for isa in [ISABase::Rv32, ISABase::Rv64] {
        let profile = plugin.metadata.isa_profiles.for_isa(isa);
        if !profile.supported {
            continue;
        }
        let need_int_family = profile.supports_float_choice(FloatExtension::None);
        let need_f_family = profile.supports_float_choice(FloatExtension::F);
        let need_fd_family = profile.supports_float_choice(FloatExtension::FD);

        for mode in supported_coverage_modes(profile) {
            if need_int_family {
                checks.push(required_executable_path_check(
                    format!(
                        "{} 1-hart {} no-float profile",
                        isa_label(isa),
                        coverage_requirement_label(mode)
                    ),
                    cva6_binary_env_var(isa, false, mode, Cva6BinaryFamily::Integer),
                    "CVA6 uses a dedicated single-hart integer/no-float wrapper when the resolved extensions disable F and D",
                    ExecutableValidation::None,
                ));
            }
            if need_f_family {
                checks.push(required_executable_path_check(
                    format!(
                        "{} 1-hart {} f-only float profile",
                        isa_label(isa),
                        coverage_requirement_label(mode)
                    ),
                    cva6_binary_env_var(isa, false, mode, Cva6BinaryFamily::SingleFloat),
                    "CVA6 uses a dedicated single-hart F-only wrapper when the resolved extensions enable F but not D",
                    ExecutableValidation::None,
                ));
            }
            if need_fd_family {
                checks.push(required_executable_path_check(
                    format!(
                        "{} 1-hart {} fd float profile",
                        isa_label(isa),
                        coverage_requirement_label(mode)
                    ),
                    cva6_binary_env_var(isa, false, mode, Cva6BinaryFamily::DoubleFloat),
                    "CVA6 uses a dedicated single-hart FD wrapper when the resolved extensions include D",
                    ExecutableValidation::None,
                ));
            }
            if profile.supports_dual_hart {
                if need_int_family {
                    checks.push(required_executable_path_check(
                        format!(
                            "{} 2-hart {} no-float profile",
                            isa_label(isa),
                            coverage_requirement_label(mode)
                        ),
                        cva6_binary_env_var(isa, true, mode, Cva6BinaryFamily::Integer),
                        "CVA6 uses a dedicated dual-hart integer/no-float wrapper when the resolved extensions disable F and D",
                        ExecutableValidation::None,
                    ));
                }
                if need_f_family {
                    checks.push(required_executable_path_check(
                        format!(
                            "{} 2-hart {} f-only float profile",
                            isa_label(isa),
                            coverage_requirement_label(mode)
                        ),
                        cva6_binary_env_var(isa, true, mode, Cva6BinaryFamily::SingleFloat),
                        "CVA6 uses a dedicated dual-hart F-only wrapper when the resolved extensions enable F but not D",
                        ExecutableValidation::None,
                    ));
                }
                if need_fd_family {
                    checks.push(required_executable_path_check(
                        format!(
                            "{} 2-hart {} fd float profile",
                            isa_label(isa),
                            coverage_requirement_label(mode)
                        ),
                        cva6_binary_env_var(isa, true, mode, Cva6BinaryFamily::DoubleFloat),
                        "CVA6 uses a dedicated dual-hart FD wrapper when the resolved extensions include D",
                        ExecutableValidation::None,
                    ));
                }
            }
        }
    }

    checks.push(required_directory_with_executable_check(
        "RISCV toolchain root for `riscv64-unknown-elf-nm`",
        "RISCV",
        "bin/riscv64-unknown-elf-nm",
        "CVA6 resolves `tohost` by invoking `$RISCV/bin/riscv64-unknown-elf-nm`",
    ));
    append_timeout_override_checks(&mut checks);
    checks
}

fn xiangshan_runtime_checks(plugin: &RegisteredPlugin) -> Vec<PluginRuntimeDoctorCheck> {
    let mut checks = Vec::new();
    let rv64 = plugin.metadata.isa_profiles.for_isa(ISABase::Rv64);
    if rv64.supported {
        for mode in supported_coverage_modes(rv64) {
            if rv64.unaligned_access_support.unaligned {
                checks.push(required_executable_path_check(
                    format!(
                        "rv64 1-hart {} with allow_unaligned=true",
                        coverage_requirement_label(mode)
                    ),
                    xiangshan_emu_env_var(false, mode, true),
                    "XiangShan uses an unaligned-capable single-hart emulator when allow_unaligned=true",
                    ExecutableValidation::None,
                ));
            }
            if rv64.unaligned_access_support.aligned {
                checks.push(required_executable_path_check(
                    format!(
                        "rv64 1-hart {} with allow_unaligned=false",
                        coverage_requirement_label(mode)
                    ),
                    xiangshan_emu_env_var(false, mode, false),
                    "XiangShan uses an aligned-only single-hart emulator when allow_unaligned=false",
                    ExecutableValidation::None,
                ));
            }

            if rv64.supports_dual_hart {
                if rv64.unaligned_access_support.unaligned {
                    checks.push(required_executable_path_check(
                        format!(
                            "rv64 2-hart {} with allow_unaligned=true",
                            coverage_requirement_label(mode)
                        ),
                        xiangshan_emu_env_var(true, mode, true),
                        "XiangShan uses an unaligned-capable dual-hart emulator when allow_unaligned=true",
                        ExecutableValidation::None,
                    ));
                }
                if rv64.unaligned_access_support.aligned {
                    checks.push(required_executable_path_check(
                        format!(
                            "rv64 2-hart {} with allow_unaligned=false",
                            coverage_requirement_label(mode)
                        ),
                        xiangshan_emu_env_var(true, mode, false),
                        "XiangShan uses an aligned-only dual-hart emulator when allow_unaligned=false",
                        ExecutableValidation::None,
                    ));
                }
            }
        }

        checks.push(required_file_check(
            "rv64 1-hart DiffTest reference library",
            xiangshan_diff_env_var(false),
            "XiangShan execution always injects an external DiffTest reference `.so` for single-hart runs",
        ));
        if rv64.supports_dual_hart {
            checks.push(required_file_check(
                "rv64 2-hart DiffTest reference library",
                xiangshan_diff_env_var(true),
                "XiangShan execution always injects an external DiffTest reference `.so` for dual-hart runs",
            ));
        }
    }

    append_timeout_override_checks(&mut checks);
    checks
}

fn append_timeout_override_checks(checks: &mut Vec<PluginRuntimeDoctorCheck>) {
    let primary_var = "RISCV_FUZZ_EXEC_TIMEOUT_SECS";
    checks.push(convert_plugin_doctor_check(
        &plugin_doctor::execution_timeout_override_check(primary_var),
    ));
}

fn supported_coverage_modes(profile: &IsaExecutionProfile) -> Vec<coverage::CoverageMode> {
    plugin_doctor::supported_coverage_modes(profile)
}

fn coverage_requirement_label(mode: coverage::CoverageMode) -> &'static str {
    plugin_doctor::coverage_requirement_label(mode)
}

fn picorv32_binary_env_var(multicore: bool, mode: coverage::CoverageMode) -> &'static str {
    match (multicore, mode) {
        (false, coverage::CoverageMode::Off) => "CX_RISCV_CORES_PICORV32_RV32_1C",
        (false, coverage::CoverageMode::Full) => "CX_RISCV_CORES_PICORV32_RV32_1C_COV",
        (false, coverage::CoverageMode::Light) => "CX_RISCV_CORES_PICORV32_RV32_1C_COV_LIGHT",
        (true, coverage::CoverageMode::Off) => "CX_RISCV_CORES_PICORV32_RV32_2C",
        (true, coverage::CoverageMode::Full) => "CX_RISCV_CORES_PICORV32_RV32_2C_COV",
        (true, coverage::CoverageMode::Light) => "CX_RISCV_CORES_PICORV32_RV32_2C_COV_LIGHT",
    }
}

fn ibex_binary_env_var(multicore: bool, mode: coverage::CoverageMode) -> &'static str {
    match (multicore, mode) {
        (false, coverage::CoverageMode::Off) => "CX_RISCV_CORES_IBEX_RV32IMC_1C",
        (false, coverage::CoverageMode::Full) => "CX_RISCV_CORES_IBEX_RV32IMC_1C_COV",
        (false, coverage::CoverageMode::Light) => "CX_RISCV_CORES_IBEX_RV32IMC_1C_COV_LIGHT",
        (true, coverage::CoverageMode::Off) => "CX_RISCV_CORES_IBEX_RV32IMC_2C",
        (true, coverage::CoverageMode::Full) => "CX_RISCV_CORES_IBEX_RV32IMC_2C_COV",
        (true, coverage::CoverageMode::Light) => "CX_RISCV_CORES_IBEX_RV32IMC_2C_COV_LIGHT",
    }
}

fn kronos_binary_env_var(multicore: bool, mode: coverage::CoverageMode) -> &'static str {
    match (multicore, mode) {
        (false, coverage::CoverageMode::Off) => "CX_RISCV_CORES_KRONOS_RV32_1C",
        (false, coverage::CoverageMode::Full) => "CX_RISCV_CORES_KRONOS_RV32_1C_COV",
        (false, coverage::CoverageMode::Light) => "CX_RISCV_CORES_KRONOS_RV32_1C_COV_LIGHT",
        (true, coverage::CoverageMode::Off) => "CX_RISCV_CORES_KRONOS_RV32_2C",
        (true, coverage::CoverageMode::Full) => "CX_RISCV_CORES_KRONOS_RV32_2C_COV",
        (true, coverage::CoverageMode::Light) => "CX_RISCV_CORES_KRONOS_RV32_2C_COV_LIGHT",
    }
}

#[derive(Debug, Clone, Copy)]
enum Cva6BinaryFamily {
    Integer,
    SingleFloat,
    DoubleFloat,
}

fn cva6_binary_env_var(
    isa: ISABase,
    multicore: bool,
    mode: coverage::CoverageMode,
    family: Cva6BinaryFamily,
) -> &'static str {
    match (isa, multicore, mode, family) {
        (ISABase::Rv32, false, coverage::CoverageMode::Off, Cva6BinaryFamily::Integer) => {
            "CX_RISCV_CORES_CVA6_RV32_1C"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Full, Cva6BinaryFamily::Integer) => {
            "CX_RISCV_CORES_CVA6_RV32_1C_COV"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Light, Cva6BinaryFamily::Integer) => {
            "CX_RISCV_CORES_CVA6_RV32_1C_COV_LIGHT"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Off, Cva6BinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_CVA6_RV32F_1C"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Full, Cva6BinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_CVA6_RV32F_1C_COV"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Light, Cva6BinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_CVA6_RV32F_1C_COV_LIGHT"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Off, _) => "CX_RISCV_CORES_CVA6_RV64_1C",
        (ISABase::Rv64, false, coverage::CoverageMode::Full, _) => {
            "CX_RISCV_CORES_CVA6_RV64_1C_COV"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Light, _) => {
            "CX_RISCV_CORES_CVA6_RV64_1C_COV_LIGHT"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Off, Cva6BinaryFamily::Integer) => {
            "CX_RISCV_CORES_CVA6_RV32_2C"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Full, Cva6BinaryFamily::Integer) => {
            "CX_RISCV_CORES_CVA6_RV32_2C_COV"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Light, Cva6BinaryFamily::Integer) => {
            "CX_RISCV_CORES_CVA6_RV32_2C_COV_LIGHT"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Off, Cva6BinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_CVA6_RV32F_2C"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Full, Cva6BinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_CVA6_RV32F_2C_COV"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Light, Cva6BinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_CVA6_RV32F_2C_COV_LIGHT"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Off, _) => "CX_RISCV_CORES_CVA6_RV64_2C",
        (ISABase::Rv64, true, coverage::CoverageMode::Full, _) => "CX_RISCV_CORES_CVA6_RV64_2C_COV",
        (ISABase::Rv64, true, coverage::CoverageMode::Light, _) => {
            "CX_RISCV_CORES_CVA6_RV64_2C_COV_LIGHT"
        }
        (ISABase::Rv32, _, _, Cva6BinaryFamily::DoubleFloat) => {
            panic!("CVA6 does not provide an RV32FD wrapper")
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RocketBinaryFamily {
    Integer,
    SingleFloat,
    DoubleFloat,
}

fn rocket_binary_env_var(
    isa: ISABase,
    multicore: bool,
    mode: coverage::CoverageMode,
    family: RocketBinaryFamily,
) -> &'static str {
    match (isa, multicore, mode, family) {
        (ISABase::Rv32, false, coverage::CoverageMode::Off, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32_1C"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Full, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32_1C_COV"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Light, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32_1C_COV_LIGHT"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Off, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32_2C"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Full, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32_2C_COV"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Light, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32_2C_COV_LIGHT"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Off, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32F_1C"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Full, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32F_1C_COV"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Light, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32F_1C_COV_LIGHT"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Off, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32F_2C"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Full, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32F_2C_COV"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Light, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32F_2C_COV_LIGHT"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Off, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_1C"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Full, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_1C_COV"
        }
        (ISABase::Rv32, false, coverage::CoverageMode::Light, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_1C_COV_LIGHT"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Off, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_2C"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Full, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_2C_COV"
        }
        (ISABase::Rv32, true, coverage::CoverageMode::Light, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV32FD_2C_COV_LIGHT"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Off, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64_1C"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Full, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64_1C_COV"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Light, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64_1C_COV_LIGHT"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Off, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64_2C"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Full, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64_2C_COV"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Light, RocketBinaryFamily::Integer) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64_2C_COV_LIGHT"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Off, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64F_1C"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Full, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64F_1C_COV"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Light, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64F_1C_COV_LIGHT"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Off, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64F_2C"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Full, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64F_2C_COV"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Light, RocketBinaryFamily::SingleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64F_2C_COV_LIGHT"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Off, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_1C"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Full, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_1C_COV"
        }
        (ISABase::Rv64, false, coverage::CoverageMode::Light, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_1C_COV_LIGHT"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Off, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_2C"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Full, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_2C_COV"
        }
        (ISABase::Rv64, true, coverage::CoverageMode::Light, RocketBinaryFamily::DoubleFloat) => {
            "CX_RISCV_CORES_ROCKET_CHIP_RV64FD_2C_COV_LIGHT"
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum VexBinaryFamily {
    NoFloat,
    FOnly,
    FD,
}

fn vex_binary_env_var(
    multicore: bool,
    mode: coverage::CoverageMode,
    family: VexBinaryFamily,
) -> &'static str {
    match (multicore, mode, family) {
        (false, coverage::CoverageMode::Off, VexBinaryFamily::NoFloat) => {
            "CX_RISCV_CORES_VEXRISCV_RV32_1C"
        }
        (false, coverage::CoverageMode::Full, VexBinaryFamily::NoFloat) => {
            "CX_RISCV_CORES_VEXRISCV_RV32_1C_COV"
        }
        (false, coverage::CoverageMode::Light, VexBinaryFamily::NoFloat) => {
            "CX_RISCV_CORES_VEXRISCV_RV32_1C_COV_LIGHT"
        }
        (false, coverage::CoverageMode::Off, VexBinaryFamily::FOnly) => {
            "CX_RISCV_CORES_VEXRISCV_RV32F_1C"
        }
        (false, coverage::CoverageMode::Full, VexBinaryFamily::FOnly) => {
            "CX_RISCV_CORES_VEXRISCV_RV32F_1C_COV"
        }
        (false, coverage::CoverageMode::Light, VexBinaryFamily::FOnly) => {
            "CX_RISCV_CORES_VEXRISCV_RV32F_1C_COV_LIGHT"
        }
        (false, coverage::CoverageMode::Off, VexBinaryFamily::FD) => {
            "CX_RISCV_CORES_VEXRISCV_RV32FD_1C"
        }
        (false, coverage::CoverageMode::Full, VexBinaryFamily::FD) => {
            "CX_RISCV_CORES_VEXRISCV_RV32FD_1C_COV"
        }
        (false, coverage::CoverageMode::Light, VexBinaryFamily::FD) => {
            "CX_RISCV_CORES_VEXRISCV_RV32FD_1C_COV_LIGHT"
        }
        (true, coverage::CoverageMode::Off, VexBinaryFamily::NoFloat) => {
            "CX_RISCV_CORES_VEXRISCV_RV32_2C"
        }
        (true, coverage::CoverageMode::Full, VexBinaryFamily::NoFloat) => {
            "CX_RISCV_CORES_VEXRISCV_RV32_2C_COV"
        }
        (true, coverage::CoverageMode::Light, VexBinaryFamily::NoFloat) => {
            "CX_RISCV_CORES_VEXRISCV_RV32_2C_COV_LIGHT"
        }
        (true, coverage::CoverageMode::Off, VexBinaryFamily::FD) => {
            "CX_RISCV_CORES_VEXRISCV_RV32FD_2C"
        }
        (true, coverage::CoverageMode::Full, VexBinaryFamily::FD) => {
            "CX_RISCV_CORES_VEXRISCV_RV32FD_2C_COV"
        }
        (true, coverage::CoverageMode::Light, VexBinaryFamily::FD) => {
            "CX_RISCV_CORES_VEXRISCV_RV32FD_2C_COV_LIGHT"
        }
        (true, _, VexBinaryFamily::FOnly) => {
            panic!("Vex multicore builds do not provide an F-only wrapper")
        }
    }
}

fn xiangshan_emu_env_var(
    multicore: bool,
    mode: coverage::CoverageMode,
    allow_unaligned: bool,
) -> &'static str {
    match (multicore, mode, allow_unaligned) {
        (false, coverage::CoverageMode::Off, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C",
        (false, coverage::CoverageMode::Full, true) => {
            "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C_COV"
        }
        (false, coverage::CoverageMode::Light, true) => {
            "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C_COV_LIGHT"
        }
        (false, coverage::CoverageMode::Off, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C",
        (false, coverage::CoverageMode::Full, false) => {
            "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C_COV"
        }
        (false, coverage::CoverageMode::Light, false) => {
            "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C_COV_LIGHT"
        }
        (true, coverage::CoverageMode::Off, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C",
        (true, coverage::CoverageMode::Full, true) => {
            "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C_COV"
        }
        (true, coverage::CoverageMode::Light, true) => {
            "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C_COV_LIGHT"
        }
        (true, coverage::CoverageMode::Off, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C",
        (true, coverage::CoverageMode::Full, false) => {
            "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C_COV"
        }
        (true, coverage::CoverageMode::Light, false) => {
            "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C_COV_LIGHT"
        }
    }
}

fn xiangshan_diff_env_var(multicore: bool) -> &'static str {
    if multicore {
        "CX_RISCV_CORES_XIANGSHAN_DIFFTEST_RV64_2C_SO"
    } else {
        "CX_RISCV_CORES_XIANGSHAN_DIFFTEST_RV64_1C_SO"
    }
}

fn required_executable_path_check(
    label: impl Into<String>,
    env_var: &'static str,
    note: impl Into<String>,
    validation: ExecutableValidation,
) -> PluginRuntimeDoctorCheck {
    convert_plugin_doctor_check(
        &plugin_doctor::required_executable_path_check_with_validation(
            label,
            env_var,
            note,
            |path| match validate_executable_path(path, validation) {
                Some(error) => Err(error),
                None => Ok(()),
            },
        ),
    )
}

fn convert_runtime_check(check: plugin_doctor::PluginDoctorCheck) -> PluginRuntimeDoctorCheck {
    convert_plugin_doctor_check(&check)
}

fn required_executable_path_or_command_check(
    label: impl Into<String>,
    env_var: &'static str,
    note: impl Into<String>,
) -> PluginRuntimeDoctorCheck {
    convert_runtime_check(plugin_doctor::required_executable_path_or_command_check(
        label, env_var, note,
    ))
}

fn required_file_check(
    label: impl Into<String>,
    env_var: &'static str,
    note: impl Into<String>,
) -> PluginRuntimeDoctorCheck {
    convert_runtime_check(plugin_doctor::required_file_check(label, env_var, note))
}

fn required_directory_with_executable_check(
    label: impl Into<String>,
    env_var: &'static str,
    relative_executable: &'static str,
    note: impl Into<String>,
) -> PluginRuntimeDoctorCheck {
    convert_runtime_check(plugin_doctor::required_directory_with_executable_check(
        label,
        env_var,
        relative_executable,
        note,
    ))
}

fn validate_executable_path(path: &Path, validation: ExecutableValidation) -> Option<String> {
    match validation {
        ExecutableValidation::None => None,
        ExecutableValidation::VexMulticoreTraceMarkers { require_freg_trace } => {
            if vex_binary_contains_trace_markers(path, require_freg_trace) {
                None
            } else {
                let markers = if require_freg_trace {
                    "run.regTrace/run.fregTrace/run.memTrace/run.logTrace"
                } else {
                    "run.regTrace/run.memTrace/run.logTrace"
                };
                Some(format!(
                    "binary is missing required multicore trace markers ({markers})"
                ))
            }
        }
    }
}

fn vex_binary_contains_trace_markers(path: &Path, require_freg_trace: bool) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    let has_reg = bytes
        .windows(b"run.regTrace".len())
        .any(|w| w == b"run.regTrace");
    let has_mem = bytes
        .windows(b"run.memTrace".len())
        .any(|w| w == b"run.memTrace");
    let has_log = bytes
        .windows(b"run.logTrace".len())
        .any(|w| w == b"run.logTrace");
    let has_freg = bytes
        .windows(b"run.fregTrace".len())
        .any(|w| w == b"run.fregTrace");
    has_reg && has_mem && has_log && (!require_freg_trace || has_freg)
}

fn warn_runtime_check(
    label: impl Into<String>,
    env_var: impl Into<String>,
    value: impl Into<String>,
    note: impl Into<String>,
) -> PluginRuntimeDoctorCheck {
    PluginRuntimeDoctorCheck {
        label: label.into(),
        env_var: env_var.into(),
        status: DoctorSeverity::Warn,
        value: value.into(),
        note: note.into(),
    }
}

fn render_plugin_section(out: &mut String, plugin: &RegisteredPlugin) {
    writeln!(out, "## `{}`", plugin.core_id).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Field | Value |").unwrap();
    writeln!(out, "| --- | --- |").unwrap();
    writeln!(
        out,
        "| Impl ID | `{}` |",
        escape_table_cell(&plugin.core_id)
    )
    .unwrap();
    writeln!(
        out,
        "| Library | `{}` |",
        escape_table_cell(&display_file_name(&plugin.source_path))
    )
    .unwrap();
    writeln!(
        out,
        "| Source Path | `{}` |",
        escape_table_cell(&plugin.source_path.display().to_string())
    )
    .unwrap();
    writeln!(
        out,
        "| Program Memory | {} |",
        escape_table_cell(&format_mem_range(plugin.metadata.program_mem_range))
    )
    .unwrap();
    writeln!(
        out,
        "| Linker Script | {} lines |",
        plugin.linker_script.lines().count()
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "### ISA Profiles").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| ISA | Supported | Dual Hart | Unaligned | Float | Float Select | Coverage | Extensions |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- | --- | --- | --- |").unwrap();
    for isa in [ISABase::Rv32, ISABase::Rv64] {
        let profile = plugin.metadata.isa_profiles.for_isa(isa);
        writeln!(
            out,
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |",
            escape_table_cell(isa_label(isa)),
            yes_no(profile.supported),
            yes_no(profile.supported && profile.supports_dual_hart),
            escape_table_cell(&format_unaligned_support(profile.unaligned_access_support)),
            escape_table_cell(&format_float_support(profile.max_float_extension())),
            yes_no(profile.supported && profile_supports_float_selection(profile)),
            escape_table_cell(&format_coverage_modes(profile.coverage_mode_support)),
            escape_table_cell(&format_extension_count(&profile.extensions)),
        )
        .unwrap();
    }
    writeln!(out).unwrap();

    render_extensions_block(out, &plugin.metadata, ISABase::Rv32);
    render_extensions_block(out, &plugin.metadata, ISABase::Rv64);

    writeln!(out, "<details>").unwrap();
    writeln!(out, "<summary>Linker script</summary>").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "```ld").unwrap();
    write!(out, "{}", plugin.linker_script.trim_end()).unwrap();
    writeln!(out).unwrap();
    writeln!(out, "```").unwrap();
    writeln!(out, "</details>").unwrap();
    writeln!(out).unwrap();
}

fn render_extensions_block(out: &mut String, metadata: &CorePluginMetadata, isa: ISABase) {
    let profile = metadata.isa_profiles.for_isa(isa);
    if !profile.supported {
        return;
    }

    let extensions = &profile.extensions;

    writeln!(out, "### {} Extensions", isa_label(isa)).unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "{}",
        render_extension_badges(extensions)
            .unwrap_or_else(|| "_No extensions defined._".to_string())
    )
    .unwrap();
    writeln!(out).unwrap();
}

fn render_extension_badges(extensions: &IsaExtensions) -> Option<String> {
    let names: Vec<String> = match extensions {
        IsaExtensions::Rv32(list) => list.iter().map(|ext| format!("{ext:?}")).collect(),
        IsaExtensions::Rv64(list) => list.iter().map(|ext| format!("{ext:?}")).collect(),
    };

    if names.is_empty() {
        return None;
    }

    Some(
        names
            .into_iter()
            .map(|name| format!("`{name}`"))
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn display_file_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn format_mem_range(range: (u64, u64)) -> String {
    let (start, end) = range;
    let size = end.saturating_sub(start).saturating_add(1);
    format!("`0x{start:016x}` - `0x{end:016x}` ({})", format_bytes(size))
}

fn format_bytes(size: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;

    if size >= GIB {
        format!("{:.2} GiB", size as f64 / GIB as f64)
    } else if size >= MIB {
        format!("{:.2} MiB", size as f64 / MIB as f64)
    } else if size >= KIB {
        format!("{:.2} KiB", size as f64 / KIB as f64)
    } else {
        format!("{size} B")
    }
}

fn profile_summary(profile: &IsaExecutionProfile) -> String {
    if !profile.supported {
        return "unsupported".to_string();
    }

    let ext_count = match &profile.extensions {
        IsaExtensions::Rv32(list) => list.len(),
        IsaExtensions::Rv64(list) => list.len(),
    };

    format!(
        "{} ext, {}, {}",
        ext_count,
        format_float_support(profile.max_float_extension()),
        format_unaligned_support(profile.unaligned_access_support)
    )
}

fn format_extension_count(extensions: &IsaExtensions) -> String {
    match extensions {
        IsaExtensions::Rv32(list) => format!("{}", list.len()),
        IsaExtensions::Rv64(list) => format!("{}", list.len()),
    }
}

fn format_coverage_modes(support: CoverageModeSupport) -> String {
    let mut modes = Vec::new();
    if support.off {
        modes.push("off");
    }
    if support.light {
        modes.push("light");
    }
    if support.full {
        modes.push("full");
    }
    if modes.is_empty() {
        "none".to_string()
    } else {
        modes.join(", ")
    }
}

fn format_unaligned_support(support: UnalignedAccessSupport) -> String {
    match (support.aligned, support.unaligned) {
        (false, false) => "none".to_string(),
        (true, false) => "aligned".to_string(),
        (false, true) => "unaligned-only".to_string(),
        (true, true) => "aligned + unaligned".to_string(),
    }
}

fn format_float_support(support: FloatExtension) -> String {
    match support {
        FloatExtension::None => "none".to_string(),
        FloatExtension::F => "f".to_string(),
        FloatExtension::FD => "f+d".to_string(),
    }
}

fn profile_supports_float_selection(profile: &IsaExecutionProfile) -> bool {
    profile.supported && profile.max_float_extension() != FloatExtension::None
}

fn escape_table_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', "<br>")
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn isa_label(isa: ISABase) -> &'static str {
    plugin_doctor::isa_label(isa)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, sync::Mutex};

    use riscv_fuzz_plugin_api::{
        CorePluginMetadata, CoverageModeSupport, IsaExecutionProfile, IsaExecutionProfileMap,
        PluginDoctorReport, UnalignedAccessSupport,
    };
    use riscv_instruction::{RV32Extensions, RV64Extensions};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => unsafe { env::set_var(self.key, value) },
                None => unsafe { env::remove_var(self.key) },
            }
        }
    }

    fn set_env_var(key: &'static str, value: Option<&str>) -> EnvVarGuard {
        let previous = env::var_os(key);
        match value {
            Some(value) => unsafe { env::set_var(key, value) },
            None => unsafe { env::remove_var(key) },
        }
        EnvVarGuard { key, previous }
    }

    fn sample_plugin() -> RegisteredPlugin {
        RegisteredPlugin {
            core_id: "sample".to_string(),
            source_path: PathBuf::from("/tmp/libriscv_fuzz_core_sample.so"),
            metadata: CorePluginMetadata {
                core_id: "sample".to_string(),
                isa_profiles: IsaExecutionProfileMap {
                    rv32: IsaExecutionProfile {
                        supported: true,
                        supports_dual_hart: false,
                        coverage_mode_support: CoverageModeSupport::ALL,
                        unaligned_access_support: UnalignedAccessSupport {
                            aligned: true,
                            unaligned: false,
                        },
                        extensions: IsaExtensions::rv32(vec![
                            RV32Extensions::I,
                            RV32Extensions::M,
                            RV32Extensions::F,
                            RV32Extensions::Zicsr,
                        ]),
                    },
                    rv64: IsaExecutionProfile {
                        supported: true,
                        supports_dual_hart: true,
                        coverage_mode_support: CoverageModeSupport {
                            off: true,
                            light: true,
                            full: false,
                        },
                        unaligned_access_support: UnalignedAccessSupport {
                            aligned: true,
                            unaligned: true,
                        },
                        extensions: IsaExtensions::rv64(vec![
                            RV64Extensions::I,
                            RV64Extensions::M,
                            RV64Extensions::F,
                            RV64Extensions::D,
                        ]),
                    },
                },
                program_mem_range: (0x1000, 0x1fff),
            },
            doctor_report: PluginDoctorReport::empty("sample"),
            linker_script: "SECTIONS { .text : { *(.text*) } }".to_string(),
        }
    }

    fn sample_plugin_with_impl(core_id: &str) -> RegisteredPlugin {
        let mut plugin = sample_plugin();
        plugin.core_id = core_id.to_string();
        plugin.metadata.core_id = core_id.to_string();
        plugin.doctor_report.plugin_id = core_id.to_string();
        plugin
    }

    fn sample_discovery_report() -> PluginDiscoveryReport {
        PluginDiscoveryReport {
            plugin_files_env: Some(
                "/tmp/custom_plugin.so:/tmp/libriscv_fuzz_core_shadow.so".to_string(),
            ),
            plugin_dirs_env: Some("/tmp/plugins:/tmp/missing".to_string()),
            configured_dirs: vec![
                riscv_fuzz_plugin_manager::PluginDirectoryScan {
                    path: PathBuf::from("/tmp/plugins"),
                    exists: true,
                    read_error: None,
                    candidate_entries: 2,
                },
                riscv_fuzz_plugin_manager::PluginDirectoryScan {
                    path: PathBuf::from("/tmp/missing"),
                    exists: false,
                    read_error: None,
                    candidate_entries: 0,
                },
            ],
            candidates: vec![
                riscv_fuzz_plugin_manager::PluginDiscoveryCandidate {
                    path: PathBuf::from("/tmp/custom_plugin.so"),
                    source: PluginDiscoverySource::ExplicitFileEnv,
                    exists: true,
                    is_file: true,
                    matches_directory_naming_heuristic: false,
                    stem: Some("custom_plugin".to_string()),
                    selected: true,
                    skip_reason: None,
                },
                riscv_fuzz_plugin_manager::PluginDiscoveryCandidate {
                    path: PathBuf::from("/tmp/plugins/libriscv_fuzz_core_sample.so"),
                    source: PluginDiscoverySource::SearchDirectory,
                    exists: true,
                    is_file: true,
                    matches_directory_naming_heuristic: true,
                    stem: Some("riscv_fuzz_core_sample".to_string()),
                    selected: true,
                    skip_reason: None,
                },
                riscv_fuzz_plugin_manager::PluginDiscoveryCandidate {
                    path: PathBuf::from("/tmp/plugins/libriscv_fuzz_core_sample_copy.so"),
                    source: PluginDiscoverySource::SearchDirectory,
                    exists: true,
                    is_file: true,
                    matches_directory_naming_heuristic: true,
                    stem: Some("riscv_fuzz_core_sample".to_string()),
                    selected: true,
                    skip_reason: Some(
                        "duplicate plugin stem `riscv_fuzz_core_sample` retained as a fallback candidate after an earlier match"
                            .to_string(),
                    ),
                },
            ],
        }
    }

    #[test]
    fn markdown_report_contains_key_sections() {
        let markdown = render_plugin_catalog_markdown(&[sample_plugin()]);
        assert!(markdown.contains("# Loaded RISC-V Plugin Catalog"));
        assert!(markdown.contains("## Summary"));
        assert!(markdown.contains("## ISA Capability Matrix"));
        assert!(markdown.contains("## `sample`"));
        assert!(markdown.contains("### rv32 Extensions"));
        assert!(markdown.contains("### rv64 Extensions"));
        assert!(markdown.contains("<details>"));
    }

    #[test]
    fn table_cells_escape_pipes() {
        assert_eq!(escape_table_cell("a|b"), "a\\|b");
    }

    #[test]
    fn doctor_report_contains_discovery_and_failure_sections() {
        let markdown = render_plugin_doctor_markdown(
            &sample_discovery_report(),
            &PluginRegistryLoadState::Failed(
                "failed to load RISC-V plugin library /tmp/custom_plugin.so via riscv_fuzz_plugin_descriptor_v3__custom_plugin: boom"
                    .to_string(),
            ),
        );

        assert!(markdown.contains("# RISC-V Plugin Doctor"));
        assert!(markdown.contains("## Effective Search Directories"));
        assert!(markdown.contains("## Candidate Libraries"));
        assert!(markdown.contains("explicit file paths bypass directory filename filtering"));
        assert!(markdown.contains(
            "duplicate plugin stem `riscv_fuzz_core_sample` retained as a fallback candidate"
        ));
        assert!(markdown.contains("Registry load failed."));
        assert!(markdown.contains("- FAIL: plugin registry load failed: failed to load RISC-V plugin library /tmp/custom_plugin.so"));
    }

    #[test]
    fn doctor_report_emits_pass_when_registry_load_is_clean() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let current_exe = env::current_exe().expect("current test binary path");
        let current_exe = current_exe.to_string_lossy().into_owned();
        let _spike_env = set_env_var("CX_RISCV_CORES_SPIKE", Some(&current_exe));

        let mut discovery = sample_discovery_report();
        discovery.plugin_dirs_env = None;
        discovery.configured_dirs = vec![riscv_fuzz_plugin_manager::PluginDirectoryScan {
            path: PathBuf::from("riscv_fuzz_plugins"),
            exists: true,
            read_error: None,
            candidate_entries: 1,
        }];
        discovery.candidates = vec![riscv_fuzz_plugin_manager::PluginDiscoveryCandidate {
            path: PathBuf::from("riscv_fuzz_plugins/libriscv_fuzz_core_sample.so"),
            source: PluginDiscoverySource::SearchDirectory,
            exists: true,
            is_file: true,
            matches_directory_naming_heuristic: true,
            stem: Some("riscv_fuzz_core_sample".to_string()),
            selected: true,
            skip_reason: None,
        }];

        let markdown = render_plugin_doctor_markdown(
            &discovery,
            &PluginRegistryLoadState::Loaded(vec![sample_plugin_with_impl("spike")]),
        );

        assert!(markdown.contains("| Overall Status | `PASS` |"));
        assert!(markdown.contains("Successfully loaded **1** plugin."));
        assert!(markdown.contains("## Plugin Runtime Environment"));
        assert!(markdown.contains("- PASS: Plugin discovery and registry load look healthy."));
    }

    #[test]
    fn spike_runtime_report_fails_when_binary_env_is_missing() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _spike_env = set_env_var("CX_RISCV_CORES_SPIKE", None);

        let report = build_plugin_runtime_report(&sample_plugin_with_impl("spike"));

        assert_eq!(report.plugin_id, "spike");
        assert_eq!(report.overall_status, DoctorSeverity::Fail);
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.env_var == "CX_RISCV_CORES_SPIKE"
                    && check.status == DoctorSeverity::Fail)
        );
    }
}
