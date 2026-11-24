mod asm;
pub mod execute;
mod metadata;

use asm::{build_asm, linker_script};
pub use execute::{ExecutorConfig, XiangShanExecutor};
use metadata::{isa_profiles, program_mem_range};
use riscv_fuzz_plugin_api::{
    CoverageMode, ExecuteRequest, ExecutionOutput, ISABase, ImplExecutionError, PluginDoctorCheck,
    PluginDoctorReport,
    doctor::{
        coverage_requirement_label, parse_optional_nonzero_timeout_secs_env,
        push_execution_timeout_override_check, required_executable_path_check, required_file_check,
        supported_coverage_modes,
    },
    streaming::InstructionEvent,
};

const EXEC_TIMEOUT_ENV: &str = "RISCV_FUZZ_EXEC_TIMEOUT_SECS";

fn apply_timeout_override(
    config: &mut ExecutorConfig,
) -> Result<(), riscv_fuzz_plugin_api::ImplExecutionError> {
    if let Some(timeout) = parse_optional_nonzero_timeout_secs_env(EXEC_TIMEOUT_ENV)
        .map_err(riscv_fuzz_plugin_api::ImplExecutionError::Generic)?
    {
        config.timeout = Some(timeout);
        config.timeout_overridden = true;
    }
    Ok(())
}

fn xiangshan_emu_env_var(
    multicore: bool,
    mode: CoverageMode,
    allow_unaligned: bool,
) -> &'static str {
    match (multicore, mode, allow_unaligned) {
        (false, CoverageMode::Off, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C",
        (false, CoverageMode::Full, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C_COV",
        (false, CoverageMode::Light, true) => {
            "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_1C_COV_LIGHT"
        }
        (false, CoverageMode::Off, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C",
        (false, CoverageMode::Full, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C_COV",
        (false, CoverageMode::Light, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_1C_COV_LIGHT",
        (true, CoverageMode::Off, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C",
        (true, CoverageMode::Full, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C_COV",
        (true, CoverageMode::Light, true) => "CX_RISCV_CORES_XIANGSHAN_RV64_UNALIGNED_2C_COV_LIGHT",
        (true, CoverageMode::Off, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C",
        (true, CoverageMode::Full, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C_COV",
        (true, CoverageMode::Light, false) => "CX_RISCV_CORES_XIANGSHAN_RV64_ALIGNED_2C_COV_LIGHT",
    }
}

fn xiangshan_diff_env_var(multicore: bool) -> &'static str {
    if multicore {
        "CX_RISCV_CORES_XIANGSHAN_DIFFTEST_RV64_2C_SO"
    } else {
        "CX_RISCV_CORES_XIANGSHAN_DIFFTEST_RV64_1C_SO"
    }
}

fn append_timeout_override_checks(checks: &mut Vec<PluginDoctorCheck>) {
    push_execution_timeout_override_check(checks, EXEC_TIMEOUT_ENV);
}

fn doctor_report() -> PluginDoctorReport {
    let mut checks = Vec::new();
    let profiles = isa_profiles();
    let rv64 = profiles.for_isa(ISABase::Rv64);

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
    PluginDoctorReport::new("xiangshan", checks)
}

fn build_xiangshan_config(
    request: &ExecuteRequest<'_>,
    run_root: &std::path::Path,
    isa_base: ISABase,
) -> Result<ExecutorConfig, ImplExecutionError> {
    let mut config = ExecutorConfig::new(run_root.to_path_buf(), isa_base, "xiangshan".to_string());
    if let Some(map) = request.extension_override {
        config.set_extension_override(map.clone());
    }
    config.allow_unaligned = request.alignment_mode.allows_unaligned();
    config.coverage_mode = request.coverage_mode;
    apply_timeout_override(&mut config)?;
    Ok(config)
}

fn validate_execute_request(
    request: &ExecuteRequest<'_>,
    metadata: &riscv_fuzz_plugin_api::CorePluginMetadata,
) -> Result<(), ImplExecutionError> {
    riscv_fuzz_plugin_api::validate_execute_request_with_config(
        request,
        metadata,
        build_xiangshan_config,
    )
}

fn execute_request(request: ExecuteRequest<'_>) -> Result<ExecutionOutput, ImplExecutionError> {
    riscv_fuzz_plugin_api::execute_with_config(
        request,
        build_xiangshan_config,
        |config, run_root, per_hart| {
            XiangShanExecutor::new(config)
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
    let config = match build_xiangshan_config(&request, run_root, isa_base) {
        Ok(config) => config,
        Err(error) => {
            emit(InstructionEvent::Failed { error });
            return;
        }
    };
    let executor = XiangShanExecutor::new(config);
    request.with_per_hart_asm_and_typed_sequences(|per_hart, typed_sequences| {
        executor.execute_streaming(run_root, per_hart, typed_sequences, emit);
    });
}

riscv_fuzz_plugin_api::declare_riscv_core_plugin!(
    plugin_type: XiangShanPlugin,
    static: XIANGSHAN_PLUGIN,
    getter: xiangshan_plugin,
    core_id: "xiangshan",
    doctor: doctor_report,
    linker_script: linker_script,
    build_asm: build_asm,
    validate: validate_execute_request,
    execute: execute_request,
    execute_streaming: execute_streaming_request,
    export_symbol: "riscv_fuzz_plugin_descriptor_v6__riscv_fuzz_core_xiangshan",
);
