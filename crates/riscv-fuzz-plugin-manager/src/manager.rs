use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use riscv_fuzz_plugin_api::{
    ConfigError, CorePluginMetadata, ExecuteRequest, ExecutionOutput, ISABase, InstructionEvent,
    IsaExecutionProfile, IsaExtensions, PerHartTypedSequences, PluginDoctorReport, RiscvCorePlugin,
    StreamingCollector,
};

use crate::{
    error::{CapabilityError, ExecutionError, LookupError},
    registry::{PluginDiscoveryReport, PluginRegistry, RegisteredPlugin, plugin_discovery_report},
    validate::validate_execute_request_metadata,
};

static GLOBAL_PLUGIN_MANAGER: OnceLock<Result<PluginManager, String>> = OnceLock::new();

#[derive(Clone)]
pub struct PluginManager {
    registry: Arc<PluginRegistry>,
}

#[derive(Clone, Copy)]
pub struct LoadedPlugin<'a> {
    core_id: &'a str,
    plugin: &'a dyn RiscvCorePlugin,
}

impl PluginManager {
    pub fn discover() -> PluginDiscoveryReport {
        plugin_discovery_report()
    }

    pub fn load() -> Result<Self, String> {
        Self::from_discovery(Self::discover())
    }

    pub fn from_discovery(discovery: PluginDiscoveryReport) -> Result<Self, String> {
        let registry = PluginRegistry::load(discovery)?;
        Ok(Self {
            registry: Arc::new(registry),
        })
    }

    pub fn try_global() -> Result<&'static Self, String> {
        GLOBAL_PLUGIN_MANAGER
            .get_or_init(Self::load)
            .as_ref()
            .map_err(Clone::clone)
    }

    pub fn global() -> &'static Self {
        Self::try_global().unwrap_or_else(|error| {
            panic!("failed to initialize RISC-V plugin manager: {error}");
        })
    }

    pub fn discovery_report(&self) -> &PluginDiscoveryReport {
        self.registry.discovery_report()
    }

    pub fn registered_impls(&self) -> Vec<String> {
        self.registry.registered_impls()
    }

    pub fn registered_plugins(&self) -> Vec<RegisteredPlugin> {
        self.registry.registered_plugins()
    }

    pub fn try_plugin(&self, core_id: impl AsRef<str>) -> Option<LoadedPlugin<'_>> {
        self.registry
            .try_plugin(core_id.as_ref())
            .map(LoadedPlugin::new)
    }

    pub fn plugin(&self, core_id: impl AsRef<str>) -> Result<LoadedPlugin<'_>, LookupError> {
        let plugin = self.registry.plugin(core_id.as_ref())?;
        Ok(LoadedPlugin::new(plugin))
    }

    pub fn normalize_impls<I>(impls: I) -> Vec<String>
    where
        I: IntoIterator<Item = String>,
    {
        normalize_impls_inner(impls)
    }

    pub fn supported_impls_for_isa(&self, isa_base: ISABase) -> Vec<String> {
        self.registered_impls()
            .into_iter()
            .filter(|impl_ref| self.supports_isa_base(impl_ref, isa_base))
            .collect()
    }

    pub fn filter_impls_by_unaligned_access_requirement(
        &self,
        impls: &[String],
        requirement: Option<bool>,
        overrides: &HashMap<String, bool>,
    ) -> Vec<String> {
        requirement.map_or_else(
            || impls.to_vec(),
            |required| {
                impls
                    .iter()
                    .filter(|impl_ref| {
                        self.supported_unaligned_access_modes_with_overrides(impl_ref, overrides)
                            .contains(&required)
                    })
                    .cloned()
                    .collect()
            },
        )
    }

    pub fn extensions_for_impls(&self, impls: &[String], isa_base: ISABase) -> IsaExtensions {
        if impls.is_empty() {
            return empty_extensions(isa_base);
        }

        let mut intersection = self
            .loaded_plugin_or_panic(&impls[0])
            .metadata()
            .isa_profiles
            .for_isa(isa_base)
            .extensions
            .clone();

        for impl_ref in &impls[1..] {
            let ext = self
                .loaded_plugin_or_panic(impl_ref)
                .metadata()
                .isa_profiles
                .for_isa(isa_base)
                .extensions
                .clone();
            match (&mut intersection, ext) {
                (IsaExtensions::Rv32(current), IsaExtensions::Rv32(other)) => {
                    current.retain(|extension| other.contains(extension));
                }
                (IsaExtensions::Rv64(current), IsaExtensions::Rv64(other)) => {
                    current.retain(|extension| other.contains(extension));
                }
                _ => unreachable!("extension ISA mismatch for supported implementation"),
            }
        }

        intersection
    }

    pub fn min_program_mem_size(&self, impls: &[String]) -> Option<u64> {
        impls
            .iter()
            .map(|impl_ref| {
                let (start, end) = self.program_mem_range(impl_ref);
                end - start
            })
            .min()
    }

    pub fn isa_profile(&self, core_id: impl AsRef<str>, isa_base: ISABase) -> IsaExecutionProfile {
        self.loaded_plugin_or_panic(core_id).isa_profile(isa_base)
    }

    pub fn supports_isa_base(&self, core_id: impl AsRef<str>, isa_base: ISABase) -> bool {
        self.isa_profile(core_id, isa_base).supported
    }

    pub fn supports_dual_hart(&self, core_id: impl AsRef<str>, isa_base: ISABase) -> bool {
        self.loaded_plugin_or_panic(core_id)
            .supports_dual_hart(isa_base)
    }

    pub fn program_mem_range(&self, core_id: impl AsRef<str>) -> (u64, u64) {
        self.loaded_plugin_or_panic(core_id).program_mem_range()
    }

    pub fn supported_unaligned_access_modes_with_overrides(
        &self,
        core_id: impl AsRef<str>,
        overrides: &HashMap<String, bool>,
    ) -> Vec<bool> {
        self.loaded_plugin_or_panic(core_id)
            .supported_unaligned_access_modes_with_overrides(overrides)
    }

    pub fn validate_with_request(
        &self,
        core_id: impl AsRef<str>,
        request: ExecuteRequest<'_>,
    ) -> Result<(), ExecutionError> {
        self.plugin(core_id.as_ref())?
            .validate_with_request(request)
    }

    pub fn execute_with_request(
        &self,
        core_id: impl AsRef<str>,
        request: ExecuteRequest<'_>,
    ) -> Result<ExecutionOutput, ExecutionError> {
        self.plugin(core_id.as_ref())?.execute_with_request(request)
    }

    pub fn execute_streaming_with_request(
        &self,
        core_id: impl AsRef<str>,
        request: ExecuteRequest<'_>,
        emit: &mut dyn FnMut(InstructionEvent) -> bool,
    ) -> Result<(), ExecutionError> {
        self.plugin(core_id.as_ref())?
            .execute_streaming_with_request(request, emit)
    }

    fn loaded_plugin_or_panic(&self, core_id: impl AsRef<str>) -> LoadedPlugin<'_> {
        let core_id = core_id.as_ref();
        self.plugin(core_id)
            .unwrap_or_else(|_| panic!("{}", missing_plugin_message(core_id)))
    }
}

impl<'a> LoadedPlugin<'a> {
    fn new(plugin: &'a dyn RiscvCorePlugin) -> Self {
        Self {
            core_id: plugin.metadata().core_id.as_str(),
            plugin,
        }
    }

    pub fn core_id(&self) -> &str {
        self.core_id
    }

    pub fn metadata(&self) -> &'a CorePluginMetadata {
        self.plugin.metadata()
    }

    pub fn doctor_report(&self) -> PluginDoctorReport {
        self.plugin.doctor()
    }

    pub fn linker_script(&self) -> &'a str {
        self.plugin.linker_script()
    }

    pub fn build_asm(
        &self,
        hart_inputs: PerHartTypedSequences<'_>,
        isa_base: ISABase,
    ) -> Result<String, ConfigError> {
        self.plugin.build_asm(hart_inputs, isa_base)
    }

    pub fn isa_profile(&self, isa_base: ISABase) -> IsaExecutionProfile {
        self.metadata().isa_profiles.for_isa(isa_base).clone()
    }

    pub fn supports_isa_base(&self, isa_base: ISABase) -> bool {
        self.isa_profile(isa_base).supported
    }

    pub fn supports_dual_hart(&self, isa_base: ISABase) -> bool {
        let profile = self.isa_profile(isa_base);
        profile.supported && profile.supports_dual_hart
    }

    pub fn program_mem_range(&self) -> (u64, u64) {
        self.metadata().program_mem_range
    }

    pub fn supported_unaligned_access_modes_with_overrides(
        &self,
        overrides: &HashMap<String, bool>,
    ) -> Vec<bool> {
        let profiles = self.metadata().isa_profiles.clone();
        let mut base_modes = Vec::new();
        let mut aligned = false;
        let mut unaligned = false;
        for isa_base in [ISABase::Rv32, ISABase::Rv64] {
            let profile = profiles.for_isa(isa_base);
            if !profile.supported {
                continue;
            }
            aligned |= profile.unaligned_access_support.aligned;
            unaligned |= profile.unaligned_access_support.unaligned;
        }
        if aligned {
            base_modes.push(false);
        }
        if unaligned {
            base_modes.push(true);
        }

        if let Some(&forced) = overrides.get(self.core_id)
            && base_modes.contains(&forced)
        {
            return vec![forced];
        }
        base_modes
    }

    pub fn validate_execute_request_metadata(
        &self,
        request: &ExecuteRequest<'_>,
    ) -> Result<(), CapabilityError> {
        validate_execute_request_metadata(self.plugin, self.core_id, request)
    }

    pub fn validate_with_request(&self, request: ExecuteRequest<'_>) -> Result<(), ExecutionError> {
        let isa_base = request.isa_base;
        self.validate_execute_request_metadata(&request)?;
        self.plugin
            .validate_execute_request(&request)
            .map_err(|source| ExecutionError::ImplementationFailed {
                core_id: self.core_id.to_owned(),
                isa_base,
                source,
            })
    }

    pub fn execute_with_request(
        &self,
        request: ExecuteRequest<'_>,
    ) -> Result<ExecutionOutput, ExecutionError> {
        let core_id = self.core_id.to_owned();
        let isa_base = request.isa_base;
        self.validate_execute_request_metadata(&request)?;
        self.plugin
            .validate_execute_request(&request)
            .map_err(|source| ExecutionError::ImplementationFailed {
                core_id: core_id.clone(),
                isa_base,
                source,
            })?;

        let mut collector = StreamingCollector::new();
        self.plugin
            .execute_streaming(request, &mut |event| collector.push(event));
        let (per_hart, perf) =
            collector
                .finish()
                .map_err(|source| ExecutionError::ImplementationFailed {
                    core_id: core_id.clone(),
                    isa_base,
                    source,
                })?;
        request
            .finalize_execution_output(ExecutionOutput {
                per_hart,
                core_id,
                isa_base,
                perf,
            })
            .map_err(|source| ExecutionError::ImplementationFailed {
                core_id: self.core_id.to_owned(),
                isa_base,
                source,
            })
    }

    pub fn execute_streaming_with_request(
        &self,
        request: ExecuteRequest<'_>,
        emit: &mut dyn FnMut(InstructionEvent) -> bool,
    ) -> Result<(), ExecutionError> {
        let core_id = self.core_id.to_owned();
        let isa_base = request.isa_base;
        self.validate_execute_request_metadata(&request)?;
        self.plugin
            .validate_execute_request(&request)
            .map_err(|source| ExecutionError::ImplementationFailed {
                core_id: core_id.clone(),
                isa_base,
                source,
            })?;

        self.plugin.execute_streaming(request, emit);
        Ok(())
    }
}

fn normalize_impls_inner<I>(impls: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut unique: Vec<String> = impls.into_iter().collect();
    unique.sort_unstable();
    unique.dedup();
    unique
}

fn empty_extensions(isa_base: ISABase) -> IsaExtensions {
    match isa_base {
        ISABase::Rv32 => IsaExtensions::rv32(Vec::new()),
        ISABase::Rv64 => IsaExtensions::rv64(Vec::new()),
    }
}

fn missing_plugin_message(core_id: &str) -> String {
    format!(
        "RISC-V implementation plugin `{core_id}` is not available; build the plugin bundle into `target/release/`, or configure `RISCV_FUZZ_PLUGIN_DIRS` / `RISCV_FUZZ_PLUGIN_FILES`"
    )
}
