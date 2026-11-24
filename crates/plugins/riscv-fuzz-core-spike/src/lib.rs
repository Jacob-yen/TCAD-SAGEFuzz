mod asm;
mod execute;
pub mod march;
mod metadata;

use asm::{build_asm, linker_script};
pub use execute::{
    ExceptionEvent, ExecutorConfig, MemoryWrite, RegisterWrite, SpikeExecutor, SpikeRunResult,
    SpikeTrace,
};
use metadata::{isa_profiles, program_mem_range};
use riscv_fuzz_plugin_api::{
    ExecuteRequest, ExecutionOutput, ISABase, ImplExecutionError, PluginDoctorReport,
    doctor::{
        parse_optional_nonzero_timeout_secs_env, push_execution_timeout_override_check,
        required_executable_path_or_command_check,
    },
    streaming::InstructionEvent,
};

const EXEC_TIMEOUT_ENV: &str = "RISCV_FUZZ_EXEC_TIMEOUT_SECS";

fn doctor_report() -> PluginDoctorReport {
    let mut checks = vec![required_executable_path_or_command_check(
        "shared Spike executable for all supported ISA/hart combinations",
        execute::SPIKE_BINARY_ENV,
        "Spike accepts either an absolute/relative executable path or a command name resolvable from PATH",
    )];
    push_execution_timeout_override_check(&mut checks, EXEC_TIMEOUT_ENV);
    checks.push(riscv_fuzz_plugin_api::PluginDoctorCheck::pass_env(
        "extra Spike args",
        execute::SPIKE_EXTRA_ARGS_ENV,
        std::env::var(execute::SPIKE_EXTRA_ARGS_ENV).unwrap_or_default(),
        "Optional whitespace-separated arguments appended to Spike command lines",
    ));
    checks.push(riscv_fuzz_plugin_api::PluginDoctorCheck::pass_env(
        "ISA suffix override",
        execute::SPIKE_ISA_SUFFIX_ENV,
        std::env::var(execute::SPIKE_ISA_SUFFIX_ENV).unwrap_or_default(),
        "Optional suffix appended to the derived Spike --isa string, for example _zvl4096b",
    ));
    checks.push(riscv_fuzz_plugin_api::PluginDoctorCheck::pass_env(
        "Spike vector-state wrapper enable",
        execute::SPIKE_ENABLE_VECTOR_STATE_ENV,
        std::env::var(execute::SPIKE_ENABLE_VECTOR_STATE_ENV).unwrap_or_default(),
        "Optional boolean; when true, the Spike wrapper sets mstatus.VS dirty for vector reference execution",
    ));
    PluginDoctorReport::new("spike", checks)
}

fn apply_timeout_override(config: &mut ExecutorConfig) -> Result<(), ImplExecutionError> {
    if let Some(timeout) = parse_optional_nonzero_timeout_secs_env(EXEC_TIMEOUT_ENV)
        .map_err(ImplExecutionError::Generic)?
    {
        config.timeout = Some(timeout);
    }
    Ok(())
}

fn apply_extra_args_override(config: &mut ExecutorConfig) {
    if let Ok(raw) = std::env::var(execute::SPIKE_EXTRA_ARGS_ENV) {
        config
            .spike_args
            .extend(raw.split_whitespace().map(ToString::to_string));
    }
}

fn apply_isa_suffix_override(config: &mut ExecutorConfig) {
    if let Ok(raw) = std::env::var(execute::SPIKE_ISA_SUFFIX_ENV) {
        let suffix = raw.trim();
        if !suffix.is_empty() {
            config.isa_suffix = Some(suffix.to_string());
        }
    }
}

fn env_bool_enabled(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn apply_vector_state_override(config: &mut ExecutorConfig) {
    if let Ok(raw) = std::env::var(execute::SPIKE_ENABLE_VECTOR_STATE_ENV) {
        config.enable_vector_state = env_bool_enabled(&raw);
    }
}

fn validate_execute_request(
    request: &ExecuteRequest<'_>,
    metadata: &riscv_fuzz_plugin_api::CorePluginMetadata,
) -> Result<(), ImplExecutionError> {
    riscv_fuzz_plugin_api::validate_execute_request_with_config(
        request,
        metadata,
        build_spike_config,
    )
}

fn build_spike_config(
    request: &ExecuteRequest<'_>,
    run_root: &std::path::Path,
    isa_base: ISABase,
) -> Result<ExecutorConfig, ImplExecutionError> {
    let mut config = ExecutorConfig::new(run_root.to_path_buf(), isa_base, "spike".to_string());
    request.apply_config_common_fields(
        __plugin_metadata(),
        &mut config,
        |config, extensions| config.set_extension_override(extensions),
        |_, _| {},
    )?;
    apply_timeout_override(&mut config)?;
    apply_extra_args_override(&mut config);
    apply_isa_suffix_override(&mut config);
    apply_vector_state_override(&mut config);
    if request.alignment_mode.allows_unaligned() {
        config.spike_args.push("--misaligned".to_string());
    }
    Ok(config)
}

fn execute_request(request: ExecuteRequest<'_>) -> Result<ExecutionOutput, ImplExecutionError> {
    riscv_fuzz_plugin_api::execute_with_config(
        request,
        build_spike_config,
        |config, run_root, per_hart| {
            SpikeExecutor::new(config)
                .execute(run_root, per_hart)
                .map_err(ImplExecutionError::from)
        },
    )
}

fn execute_streaming_request(
    request: ExecuteRequest<'_>,
    emit: &mut dyn FnMut(InstructionEvent) -> bool,
) {
    let run_root = request.run_root;
    let isa_base = request.isa_base;

    let config = match build_spike_config(&request, run_root, isa_base) {
        Ok(c) => c,
        Err(e) => {
            emit(InstructionEvent::Failed { error: e });
            return;
        }
    };
    let executor = SpikeExecutor::new(config);
    request.with_per_hart_asm_and_typed_sequences(|per_hart, typed_sequences| {
        executor.execute_streaming(run_root, per_hart, typed_sequences, emit);
    });
}

riscv_fuzz_plugin_api::declare_riscv_core_plugin!(
    plugin_type: SpikePlugin,
    static: SPIKE_PLUGIN,
    getter: spike_plugin,
    core_id: "spike",
    doctor: doctor_report,
    linker_script: linker_script,
    build_asm: build_asm,
    validate: validate_execute_request,
    execute: execute_request,
    execute_streaming: execute_streaming_request,
    export_symbol: "riscv_fuzz_plugin_descriptor_v6__riscv_fuzz_core_spike",
);

#[cfg(test)]
mod tests {
    use super::*;

    fn doctor_env_vars(report: &PluginDoctorReport) -> Vec<&str> {
        report
            .checks
            .iter()
            .filter_map(|check| check.env_var.as_deref())
            .collect()
    }

    #[test]
    fn doctor_reports_timeout_override_contract() {
        let report = doctor_report();
        let env_vars = doctor_env_vars(&report);

        assert!(env_vars.contains(&execute::SPIKE_BINARY_ENV));
        assert!(env_vars.contains(&EXEC_TIMEOUT_ENV));
        assert!(env_vars.contains(&execute::SPIKE_EXTRA_ARGS_ENV));
        assert!(env_vars.contains(&execute::SPIKE_ISA_SUFFIX_ENV));
    }
}
