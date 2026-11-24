mod discovery;
mod loaded;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use riscv_fuzz_plugin_api::{CorePluginMetadata, PluginDoctorReport, RiscvCorePlugin};

use crate::error::LookupError;

#[derive(Debug, Clone)]
pub struct RegisteredPlugin {
    pub core_id: String,
    pub source_path: PathBuf,
    pub metadata: CorePluginMetadata,
    pub doctor_report: PluginDoctorReport,
    pub linker_script: String,
}

pub use discovery::{
    PluginDirectoryScan, PluginDiscoveryCandidate, PluginDiscoveryReport, PluginDiscoverySource,
    plugin_discovery_report,
};

pub(crate) struct PluginRegistry {
    discovery: PluginDiscoveryReport,
    plugins: HashMap<String, loaded::DynamicPlugin>,
}

impl PluginRegistry {
    pub(crate) fn load(discovery: PluginDiscoveryReport) -> Result<Self, String> {
        let mut plugins = HashMap::new();
        let mut loaded_candidate_stems = HashSet::new();

        for candidate in discovery
            .candidates
            .iter()
            .filter(|candidate| candidate.selected)
        {
            let stem = candidate.stem.as_deref().ok_or_else(|| {
                format!(
                    "selected plugin candidate {} is missing a normalized stem",
                    candidate.path.display()
                )
            })?;

            if loaded_candidate_stems.contains(stem) {
                continue;
            }

            let loaded = load_candidate_with_fallback(&discovery, stem)?;
            loaded_candidate_stems.insert(stem.to_string());

            let core_id = loaded.metadata().core_id.clone();
            if core_id.is_empty() {
                return Err(format!(
                    "plugin {} returned an empty implementation id",
                    loaded.source_path().display()
                ));
            }

            let previous = plugins.insert(core_id.clone(), loaded);
            if previous.is_some() {
                return Err(format!(
                    "duplicate dynamically loaded RISC-V plugin for {core_id} while scanning {}",
                    candidate.path.display()
                ));
            }
        }

        Ok(Self { discovery, plugins })
    }

    pub(crate) fn discovery_report(&self) -> &PluginDiscoveryReport {
        &self.discovery
    }

    pub(crate) fn try_plugin(&self, core_id: &str) -> Option<&dyn RiscvCorePlugin> {
        self.plugins
            .get(core_id)
            .map(|plugin| plugin as &dyn RiscvCorePlugin)
    }

    pub(crate) fn plugin(&self, core_id: &str) -> Result<&dyn RiscvCorePlugin, LookupError> {
        self.try_plugin(core_id)
            .ok_or_else(|| LookupError::PluginNotFound {
                core_id: core_id.to_owned(),
            })
    }

    pub(crate) fn registered_impls(&self) -> Vec<String> {
        let mut ids: Vec<_> = self.plugins.keys().cloned().collect();
        ids.sort_unstable();
        ids
    }

    pub(crate) fn registered_plugins(&self) -> Vec<RegisteredPlugin> {
        let mut plugins: Vec<_> = self
            .plugins
            .iter()
            .map(|(core_id, plugin)| registered_plugin(core_id, plugin))
            .collect();
        plugins.sort_by(|left, right| left.core_id.cmp(&right.core_id));
        plugins
    }
}

fn registered_plugin(core_id: &str, plugin: &loaded::DynamicPlugin) -> RegisteredPlugin {
    RegisteredPlugin {
        core_id: core_id.to_owned(),
        source_path: plugin.source_path().to_path_buf(),
        metadata: plugin.metadata().clone(),
        doctor_report: plugin.doctor(),
        linker_script: plugin.linker_script().to_string(),
    }
}

fn load_candidate_with_fallback(
    discovery: &PluginDiscoveryReport,
    stem: &str,
) -> Result<loaded::DynamicPlugin, String> {
    let mut load_errors = Vec::new();

    for candidate in discovery
        .candidates
        .iter()
        .filter(|candidate| candidate.selected && candidate.stem.as_deref() == Some(stem))
    {
        match loaded::DynamicPlugin::load(&candidate.path) {
            Ok(plugin) => return Ok(plugin),
            Err(error) => load_errors.push(format_load_error(&candidate.path, stem, error)),
        }
    }

    Err(load_errors.join("\n"))
}

fn format_load_error(path: &std::path::Path, stem: &str, error: String) -> String {
    let symbol_name = riscv_fuzz_plugin_api::abi::plugin_descriptor_symbol_name_for_stem(stem);
    format!(
        "failed to load RISC-V plugin library {} via {}: {error}",
        path.display(),
        symbol_name,
    )
}
