use std::path::Path;

use riscv_fuzz_toolchain::write_string_file;

use crate::types::DiffAnalysisError;

pub(crate) fn write_string_report(path: &Path, content: String) -> Result<(), DiffAnalysisError> {
    write_string_file(path, content)?;
    Ok(())
}
