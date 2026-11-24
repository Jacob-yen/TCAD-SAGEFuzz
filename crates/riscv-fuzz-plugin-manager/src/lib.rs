pub mod error;
mod manager;
mod registry;
mod validate;

pub use manager::{LoadedPlugin, PluginManager};
pub use registry::{
    PluginDirectoryScan, PluginDiscoveryCandidate, PluginDiscoveryReport, PluginDiscoverySource,
    RegisteredPlugin,
};
