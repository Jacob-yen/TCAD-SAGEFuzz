mod asm;
pub mod execute;
mod metadata;

use asm::{build_asm, linker_script};
pub use execute::{
    AraExecutor, ExceptionEvent, ExceptionSource, ExecutorConfig, MemoryWrite, RegisterWrite,
    RvfiTrace, TraceConfig,
};
use metadata::{isa_profiles, program_mem_range};
use riscv_fuzz_plugin_api::{
    ExecuteRequest, ExecutionOutput, ISABase, ImplExecutionError, PluginDoctorCheck,
    PluginDoctorReport,
    doctor::{
        parse_optional_nonzero_timeout_secs_env, push_execution_timeout_override_check,
        required_executable_path_check,
    },
    streaming::InstructionEvent,
};

const EXEC_TIMEOUT_ENV: &str = "RISCV_FUZZ_EXEC_TIMEOUT_SECS";
const ARA_TERM_AFTER_CYCLES_ENV: &str = "RISCV_FUZZ_ARA_TERM_AFTER_CYCLES";
const ARA_RV64_BINARY_ENV: &str = "CX_RISCV_CORES_ARA_RV64_1C";

fn apply_timeout_override(config: &mut ExecutorConfig) -> Result<(), ImplExecutionError> {
    if let Some(timeout) = parse_optional_nonzero_timeout_secs_env(EXEC_TIMEOUT_ENV)
        .map_err(ImplExecutionError::Generic)?
    {
        config.timeout = Some(timeout);
    }
    Ok(())
}

fn apply_term_after_cycles_override(config: &mut ExecutorConfig) -> Result<(), ImplExecutionError> {
    let Ok(raw) = std::env::var(ARA_TERM_AFTER_CYCLES_ENV) else {
        return Ok(());
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(());
    }
    let cycles = trimmed.parse::<u64>().map_err(|err| {
        ImplExecutionError::Generic(format!(
            "invalid {ARA_TERM_AFTER_CYCLES_ENV} value `{trimmed}`: {err}"
        ))
    })?;
    if cycles == 0 {
        return Err(ImplExecutionError::Generic(format!(
            "invalid {ARA_TERM_AFTER_CYCLES_ENV} value `{trimmed}`: must be non-zero"
        )));
    }
    config.term_after_cycles = cycles;
    Ok(())
}

fn append_timeout_override_checks(checks: &mut Vec<PluginDoctorCheck>) {
    push_execution_timeout_override_check(checks, EXEC_TIMEOUT_ENV);
}

fn doctor_report() -> PluginDoctorReport {
    let mut checks = Vec::new();
    checks.push(required_executable_path_check(
        "ARA RV64 single-hart Verilator binary",
        ARA_RV64_BINARY_ENV,
        "ARA expects the built `Vara_tb_verilator` path in the environment",
    ));
    append_timeout_override_checks(&mut checks);

    PluginDoctorReport::new("ara", checks)
}

fn build_ara_config(
    request: &ExecuteRequest<'_>,
    run_root: &std::path::Path,
    isa_base: ISABase,
) -> Result<ExecutorConfig, ImplExecutionError> {
    let mut config = ExecutorConfig::new(run_root.to_path_buf(), isa_base, "ara".to_string())
        .map_err(ImplExecutionError::from)?;
    request.apply_config_common_fields(
        __plugin_metadata(),
        &mut config,
        |config, extensions| config.set_extension_override(extensions),
        |config, coverage_mode| config.coverage_mode = coverage_mode,
    )?;
    apply_timeout_override(&mut config)?;
    apply_term_after_cycles_override(&mut config)?;
    Ok(config)
}

fn validate_execute_request(
    request: &ExecuteRequest<'_>,
    metadata: &riscv_fuzz_plugin_api::CorePluginMetadata,
) -> Result<(), ImplExecutionError> {
    riscv_fuzz_plugin_api::validate_execute_request_with_config(request, metadata, build_ara_config)
}

fn execute_request(request: ExecuteRequest<'_>) -> Result<ExecutionOutput, ImplExecutionError> {
    riscv_fuzz_plugin_api::execute_with_config(
        request,
        build_ara_config,
        |config, run_root, per_hart| {
            AraExecutor::new(config)
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
    let config = match build_ara_config(&request, run_root, isa_base) {
        Ok(config) => config,
        Err(error) => {
            emit(InstructionEvent::Failed { error });
            return;
        }
    };
    let executor = AraExecutor::new(config);
    request.with_per_hart_asm_and_typed_sequences(|per_hart, typed_sequences| {
        executor.execute_streaming(run_root, per_hart, typed_sequences, emit);
    });
}

riscv_fuzz_plugin_api::declare_riscv_core_plugin!(
    plugin_type: AraPlugin,
    static: ARA_PLUGIN,
    getter: ara_plugin,
    core_id: "ara",
    doctor: doctor_report,
    linker_script: linker_script,
    build_asm: build_asm,
    validate: validate_execute_request,
    execute: execute_request,
    execute_streaming: execute_streaming_request,
    export_symbol: "riscv_fuzz_plugin_descriptor_v6__riscv_fuzz_core_ara",
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
    fn doctor_requires_rv64_single_hart_binary() {
        let report = doctor_report();
        let env_vars = doctor_env_vars(&report);

        assert!(env_vars.contains(&ARA_RV64_BINARY_ENV));
    }
}
