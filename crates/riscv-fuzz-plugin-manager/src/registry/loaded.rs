use std::path::{Path, PathBuf};

use libloading::Library;
use riscv_fuzz_plugin_api::abi::{
    PLUGIN_CALL_STATUS_OK, PluginCallResult, PluginDescriptorV3, decode_build_asm_response,
    decode_plugin_metadata_record, encode_build_asm_request, encode_execute_request,
    take_plugin_call_bytes,
};
use riscv_fuzz_plugin_api::{
    ConfigError, CorePluginMetadata, ExecuteRequest, ExecutionOutput, ISABase, ImplExecutionError,
    InstructionEvent, PluginDoctorReport, PluginMetadataRecord, RiscvCorePlugin,
};

use super::discovery::normalized_plugin_stem;

type PluginDescriptorFactory = unsafe extern "C" fn() -> *const PluginDescriptorV3;

pub(super) struct DynamicPlugin {
    _library: Library,
    descriptor: *const PluginDescriptorV3,
    metadata: PluginMetadataRecord,
    source_path: PathBuf,
}

unsafe impl Send for DynamicPlugin {}
unsafe impl Sync for DynamicPlugin {}

impl DynamicPlugin {
    pub(super) fn load(path: &Path) -> Result<Self, String> {
        let library = unsafe {
            Library::new(path).map_err(|error| {
                format!("failed to open shared library {}: {error}", path.display())
            })?
        };

        let descriptor_symbol = descriptor_symbol_name_for_path(path)?;
        let descriptor = unsafe {
            let factory: libloading::Symbol<'_, PluginDescriptorFactory> =
                library.get(&descriptor_symbol).map_err(|error| {
                    format!(
                        "failed to resolve symbol `{}` from {}: {error}",
                        String::from_utf8_lossy(&descriptor_symbol[..descriptor_symbol.len() - 1]),
                        path.display()
                    )
                })?;

            let descriptor = factory();
            if descriptor.is_null() {
                return Err(format!(
                    "plugin descriptor symbol `{}` returned null in {}",
                    String::from_utf8_lossy(&descriptor_symbol[..descriptor_symbol.len() - 1]),
                    path.display()
                ));
            }
            descriptor
        };

        let descriptor_ref = unsafe { &*descriptor };
        if descriptor_ref.abi_version != riscv_fuzz_plugin_api::abi::RISCV_FUZZ_PLUGIN_ABI_VERSION {
            return Err(format!(
                "plugin ABI version mismatch for {}: expected {}, got {}",
                path.display(),
                riscv_fuzz_plugin_api::abi::RISCV_FUZZ_PLUGIN_ABI_VERSION,
                descriptor_ref.abi_version
            ));
        }

        let metadata = read_call_result(
            path,
            descriptor_ref,
            unsafe { (descriptor_ref.metadata_payload)() },
            decode_plugin_metadata_record,
        )?;

        Ok(Self {
            _library: library,
            descriptor,
            metadata,
            source_path: std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()),
        })
    }

    fn descriptor(&self) -> &PluginDescriptorV3 {
        unsafe { &*self.descriptor }
    }

    pub(super) fn source_path(&self) -> &Path {
        &self.source_path
    }

    fn call_with_payload<R>(
        &self,
        payload: &[u8],
        func: unsafe extern "C" fn(riscv_fuzz_plugin_api::abi::PluginSlice) -> PluginCallResult,
        decode: impl FnOnce(&[u8]) -> Result<R, String>,
    ) -> Result<R, String> {
        let result = unsafe { func(payload.into()) };
        read_call_result(&self.source_path, self.descriptor(), result, decode)
    }
}

impl RiscvCorePlugin for DynamicPlugin {
    fn metadata(&self) -> &CorePluginMetadata {
        &self.metadata.core_metadata
    }

    fn doctor(&self) -> PluginDoctorReport {
        self.metadata.doctor_report.clone()
    }

    fn linker_script(&self) -> &str {
        &self.metadata.linker_script
    }

    fn build_asm(
        &self,
        hart_inputs: riscv_fuzz_plugin_api::PerHartTypedSequences<'_>,
        isa_base: ISABase,
    ) -> Result<String, ConfigError> {
        let payload = encode_build_asm_request(hart_inputs, isa_base)
            .map_err(|message| ConfigError::InvalidConfiguration { message })?;
        self.call_with_payload(
            &payload,
            self.descriptor().build_asm_payload,
            decode_build_asm_response,
        )
        .map_err(|error| ConfigError::InvalidConfiguration { message: error })
    }

    fn execute(&self, request: ExecuteRequest<'_>) -> Result<ExecutionOutput, ImplExecutionError> {
        let payload = encode_execute_request(request).map_err(ImplExecutionError::Generic)?;
        self.call_with_payload(
            &payload,
            self.descriptor().execute_payload,
            riscv_fuzz_plugin_api::abi::decode_execution_output,
        )
        .map_err(ImplExecutionError::Generic)
    }

    fn execute_streaming(
        &self,
        request: ExecuteRequest<'_>,
        emit: &mut dyn FnMut(InstructionEvent) -> bool,
    ) {
        let isa_base = request.isa_base;
        let payload = match encode_execute_request(request) {
            Ok(payload) => payload,
            Err(error) => {
                emit(InstructionEvent::Failed {
                    error: ImplExecutionError::Generic(error),
                });
                return;
            }
        };

        struct CallbackState<'a> {
            isa_base: ISABase,
            emit: &'a mut dyn FnMut(InstructionEvent) -> bool,
            decode_error: Option<String>,
        }

        unsafe extern "C" fn emit_trampoline(
            ctx: *mut std::ffi::c_void,
            event: riscv_fuzz_plugin_api::abi::PluginSlice,
        ) -> bool {
            let state = unsafe { &mut *(ctx as *mut CallbackState<'_>) };
            let bytes = unsafe { std::slice::from_raw_parts(event.ptr, event.len) };
            match riscv_fuzz_plugin_api::abi::read_streaming_instruction_event(
                bytes,
                state.isa_base,
            ) {
                Ok(event) => (state.emit)(event),
                Err(error) => {
                    state.decode_error = Some(error);
                    false
                }
            }
        }

        let mut state = CallbackState {
            isa_base,
            emit,
            decode_error: None,
        };
        let result = unsafe {
            (self.descriptor().execute_streaming_payload)(
                payload.as_slice().into(),
                (&mut state as *mut CallbackState<'_>).cast(),
                emit_trampoline,
            )
        };

        let bytes = take_plugin_call_bytes(self.descriptor().free_bytes, result.bytes);
        if result.status != PLUGIN_CALL_STATUS_OK {
            let message = format!(
                "plugin streaming call failed in {}: {}",
                self.source_path.display(),
                String::from_utf8_lossy(&bytes)
            );
            (state.emit)(InstructionEvent::Failed {
                error: ImplExecutionError::Generic(message),
            });
            return;
        }

        if let Some(error) = state.decode_error {
            (state.emit)(InstructionEvent::Failed {
                error: ImplExecutionError::Generic(format!(
                    "failed to decode plugin streaming event from {}: {error}",
                    self.source_path.display()
                )),
            });
        }
    }

    fn validate_execute_request(
        &self,
        request: &ExecuteRequest<'_>,
    ) -> Result<(), ImplExecutionError> {
        let payload = encode_execute_request(*request)
            .map_err(riscv_fuzz_plugin_api::ImplExecutionError::Generic)?;
        self.call_with_payload(
            &payload,
            self.descriptor().validate_execute_request_payload,
            |_bytes| Ok(()),
        )
        .map_err(ImplExecutionError::Generic)
    }
}

fn descriptor_symbol_name_for_path(path: &Path) -> Result<Vec<u8>, String> {
    let stem = normalized_plugin_stem(path)?;
    Ok(format!(
        "{}\0",
        riscv_fuzz_plugin_api::abi::plugin_descriptor_symbol_name_for_stem(&stem)
    )
    .into_bytes())
}

fn read_call_result<T>(
    source_path: &Path,
    descriptor: &PluginDescriptorV3,
    result: PluginCallResult,
    decode: impl FnOnce(&[u8]) -> Result<T, String>,
) -> Result<T, String> {
    let bytes = take_plugin_call_bytes(descriptor.free_bytes, result.bytes);
    if result.status != PLUGIN_CALL_STATUS_OK {
        return Err(format!(
            "plugin call failed in {}: {}",
            source_path.display(),
            String::from_utf8_lossy(&bytes)
        ));
    }

    decode(&bytes).map_err(|error| {
        format!(
            "failed to decode plugin payload from {}: {error}",
            source_path.display()
        )
    })
}
