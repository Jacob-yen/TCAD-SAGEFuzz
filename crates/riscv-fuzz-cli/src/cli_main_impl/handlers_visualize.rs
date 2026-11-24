use std::{fs, path::PathBuf};

use riscv_fuzz_plugin_api::ExecutionOutput;
use riscv_fuzz_toolchain::generate_execution_timeline_svg_from_output;
use serde_json::from_reader;

use super::{CliError, Result, ResultExt};

fn load_execution_output(path: &PathBuf) -> Result<ExecutionOutput> {
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open execution output {}", path.display()))?;
    from_reader(file)
        .with_context(|| format!("failed to parse execution output {}", path.display()))
}

pub(super) fn run_render_timeline_mode(execution_output: PathBuf, output: PathBuf) -> Result<()> {
    let execution_output_path = execution_output;
    let execution_output = load_execution_output(&execution_output_path)?;

    generate_execution_timeline_svg_from_output(&execution_output, &output).map_err(|err| {
        CliError::new(format!(
            "failed to render execution timeline {}: {err}",
            output.display()
        ))
    })?;

    log::info!(
        "Rendered execution timeline from {} to {}",
        execution_output_path.display(),
        output.display()
    );
    Ok(())
}
