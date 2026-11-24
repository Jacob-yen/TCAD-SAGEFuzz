#[macro_export]
macro_rules! export_c_plugin_descriptor {
    ($symbol_name:literal, $plugin_getter:path) => {
        unsafe extern "C" fn __riscv_fuzz_plugin_metadata_payload() -> $crate::abi::PluginCallResult
        {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                $crate::abi::metadata_payload_call($plugin_getter())
            }))
            .unwrap_or_else(|_| {
                $crate::abi::PluginCallResult::err(
                    "plugin panicked while building metadata payload",
                )
            })
        }

        unsafe extern "C" fn __riscv_fuzz_plugin_build_asm_payload(
            input: $crate::abi::PluginSlice,
        ) -> $crate::abi::PluginCallResult {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                $crate::abi::build_asm_payload_call($plugin_getter(), input)
            }))
            .unwrap_or_else(|_| {
                $crate::abi::PluginCallResult::err("plugin panicked while building asm payload")
            })
        }

        unsafe extern "C" fn __riscv_fuzz_plugin_execute_payload(
            input: $crate::abi::PluginSlice,
        ) -> $crate::abi::PluginCallResult {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                $crate::abi::execute_payload_call($plugin_getter(), input)
            }))
            .unwrap_or_else(|_| {
                $crate::abi::PluginCallResult::err("plugin panicked while executing payload")
            })
        }

        unsafe extern "C" fn __riscv_fuzz_plugin_execute_streaming_payload(
            input: $crate::abi::PluginSlice,
            callback_ctx: *mut std::ffi::c_void,
            emit: $crate::abi::PluginStreamingEmit,
        ) -> $crate::abi::PluginCallResult {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // SAFETY: this exported ABI entry point forwards the host-owned
                // callback context and callback function without retaining them.
                unsafe {
                    $crate::abi::execute_streaming_payload_call(
                        $plugin_getter(),
                        input,
                        callback_ctx,
                        emit,
                    )
                }
            }))
            .unwrap_or_else(|_| {
                $crate::abi::PluginCallResult::err(
                    "plugin panicked while executing streaming payload",
                )
            })
        }

        unsafe extern "C" fn __riscv_fuzz_plugin_validate_execute_request_payload(
            input: $crate::abi::PluginSlice,
        ) -> $crate::abi::PluginCallResult {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                $crate::abi::validate_execute_request_payload_call($plugin_getter(), input)
            }))
            .unwrap_or_else(|_| {
                $crate::abi::PluginCallResult::err(
                    "plugin panicked while validating execute request payload",
                )
            })
        }

        static RISCV_FUZZ_PLUGIN_DESCRIPTOR_V3: $crate::abi::PluginDescriptorV3 =
            $crate::abi::PluginDescriptorV3 {
                abi_version: $crate::abi::RISCV_FUZZ_PLUGIN_ABI_VERSION,
                metadata_payload: __riscv_fuzz_plugin_metadata_payload,
                build_asm_payload: __riscv_fuzz_plugin_build_asm_payload,
                validate_execute_request_payload:
                    __riscv_fuzz_plugin_validate_execute_request_payload,
                execute_payload: __riscv_fuzz_plugin_execute_payload,
                execute_streaming_payload: __riscv_fuzz_plugin_execute_streaming_payload,
                free_bytes: $crate::abi::free_plugin_bytes,
            };

        #[unsafe(export_name = $symbol_name)]
        pub extern "C" fn __riscv_fuzz_plugin_descriptor_export()
        -> *const $crate::abi::PluginDescriptorV3 {
            &RISCV_FUZZ_PLUGIN_DESCRIPTOR_V3
        }
    };
}

#[macro_export]
macro_rules! declare_riscv_core_plugin {
    (
        plugin_type: $plugin_type:ident,
        static: $static_name:ident,
        getter: $getter:ident,
        core_id: $core_id:literal,
        doctor: $doctor:path,
        linker_script: $linker_script:path,
        build_asm: $build_asm:path,
        execute: $execute:path,
        export_symbol: $symbol_name:literal
        $(,)?
    ) => {
        struct $plugin_type;

        static $static_name: $plugin_type = $plugin_type;

        fn $getter() -> &'static dyn $crate::plugin::RiscvCorePlugin {
            &$static_name
        }

        fn __plugin_metadata() -> &'static $crate::metadata::CorePluginMetadata {
            static METADATA: std::sync::OnceLock<$crate::metadata::CorePluginMetadata> =
                std::sync::OnceLock::new();
            METADATA.get_or_init(|| $crate::metadata::CorePluginMetadata {
                core_id: $core_id.to_string(),
                isa_profiles: isa_profiles(),
                program_mem_range: program_mem_range(),
            })
        }

        impl $crate::plugin::RiscvCorePlugin for $plugin_type {
            fn metadata(&self) -> &$crate::metadata::CorePluginMetadata {
                __plugin_metadata()
            }

            fn doctor(&self) -> $crate::doctor::PluginDoctorReport {
                $doctor()
            }

            fn linker_script(&self) -> &str {
                $linker_script()
            }

            fn build_asm(
                &self,
                hart_inputs: $crate::plugin::PerHartTypedSequences<'_>,
                isa_base: $crate::isa::ISABase,
            ) -> Result<String, $crate::error::ConfigError> {
                $crate::plugin::build_per_hart_asm(hart_inputs, isa_base, $build_asm)
            }

            fn validate_execute_request(
                &self,
                _request: &$crate::plugin::ExecuteRequest<'_>,
            ) -> Result<(), $crate::error::ImplExecutionError> {
                Ok(())
            }

            fn execute(
                &self,
                request: $crate::plugin::ExecuteRequest<'_>,
            ) -> Result<$crate::execution::ExecutionOutput, $crate::error::ImplExecutionError> {
                $execute(request)
            }
        }

        $crate::export_c_plugin_descriptor!($symbol_name, $getter);
    };
    (
        plugin_type: $plugin_type:ident,
        static: $static_name:ident,
        getter: $getter:ident,
        core_id: $core_id:literal,
        doctor: $doctor:path,
        linker_script: $linker_script:path,
        build_asm: $build_asm:path,
        validate: $validate:path,
        execute: $execute:path,
        execute_streaming: $execute_streaming:path,
        export_symbol: $symbol_name:literal
        $(,)?
    ) => {
        struct $plugin_type;

        static $static_name: $plugin_type = $plugin_type;

        fn $getter() -> &'static dyn $crate::plugin::RiscvCorePlugin {
            &$static_name
        }

        fn __plugin_metadata() -> &'static $crate::metadata::CorePluginMetadata {
            static METADATA: std::sync::OnceLock<$crate::metadata::CorePluginMetadata> =
                std::sync::OnceLock::new();
            METADATA.get_or_init(|| $crate::metadata::CorePluginMetadata {
                core_id: $core_id.to_string(),
                isa_profiles: isa_profiles(),
                program_mem_range: program_mem_range(),
            })
        }

        impl $crate::plugin::RiscvCorePlugin for $plugin_type {
            fn metadata(&self) -> &$crate::metadata::CorePluginMetadata {
                __plugin_metadata()
            }

            fn doctor(&self) -> $crate::doctor::PluginDoctorReport {
                $doctor()
            }

            fn linker_script(&self) -> &str {
                $linker_script()
            }

            fn build_asm(
                &self,
                hart_inputs: $crate::plugin::PerHartTypedSequences<'_>,
                isa_base: $crate::isa::ISABase,
            ) -> Result<String, $crate::error::ConfigError> {
                $crate::plugin::build_per_hart_asm(hart_inputs, isa_base, $build_asm)
            }

            fn validate_execute_request(
                &self,
                request: &$crate::plugin::ExecuteRequest<'_>,
            ) -> Result<(), $crate::error::ImplExecutionError> {
                $validate(request, __plugin_metadata())
            }

            fn execute(
                &self,
                request: $crate::plugin::ExecuteRequest<'_>,
            ) -> Result<$crate::execution::ExecutionOutput, $crate::error::ImplExecutionError> {
                $execute(request)
            }

            fn execute_streaming(
                &self,
                request: $crate::plugin::ExecuteRequest<'_>,
                emit: &mut dyn FnMut($crate::streaming::InstructionEvent) -> bool,
            ) {
                $execute_streaming(request, emit)
            }
        }

        $crate::export_c_plugin_descriptor!($symbol_name, $getter);
    };
    (
        plugin_type: $plugin_type:ident,
        static: $static_name:ident,
        getter: $getter:ident,
        core_id: $core_id:literal,
        doctor: $doctor:path,
        linker_script: $linker_script:path,
        build_asm: $build_asm:path,
        validate: $validate:path,
        execute: $execute:path,
        export_symbol: $symbol_name:literal
        $(,)?
    ) => {
        struct $plugin_type;

        static $static_name: $plugin_type = $plugin_type;

        fn $getter() -> &'static dyn $crate::plugin::RiscvCorePlugin {
            &$static_name
        }

        fn __plugin_metadata() -> &'static $crate::metadata::CorePluginMetadata {
            static METADATA: std::sync::OnceLock<$crate::metadata::CorePluginMetadata> =
                std::sync::OnceLock::new();
            METADATA.get_or_init(|| $crate::metadata::CorePluginMetadata {
                core_id: $core_id.to_string(),
                isa_profiles: isa_profiles(),
                program_mem_range: program_mem_range(),
            })
        }

        impl $crate::plugin::RiscvCorePlugin for $plugin_type {
            fn metadata(&self) -> &$crate::metadata::CorePluginMetadata {
                __plugin_metadata()
            }

            fn doctor(&self) -> $crate::doctor::PluginDoctorReport {
                $doctor()
            }

            fn linker_script(&self) -> &str {
                $linker_script()
            }

            fn build_asm(
                &self,
                hart_inputs: $crate::plugin::PerHartTypedSequences<'_>,
                isa_base: $crate::isa::ISABase,
            ) -> Result<String, $crate::error::ConfigError> {
                $crate::plugin::build_per_hart_asm(hart_inputs, isa_base, $build_asm)
            }

            fn validate_execute_request(
                &self,
                request: &$crate::plugin::ExecuteRequest<'_>,
            ) -> Result<(), $crate::error::ImplExecutionError> {
                $validate(request, __plugin_metadata())
            }

            fn execute(
                &self,
                request: $crate::plugin::ExecuteRequest<'_>,
            ) -> Result<$crate::execution::ExecutionOutput, $crate::error::ImplExecutionError> {
                $execute(request)
            }
        }

        $crate::export_c_plugin_descriptor!($symbol_name, $getter);
    };
}
