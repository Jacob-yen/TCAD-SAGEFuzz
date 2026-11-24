use std::{fs, path::PathBuf, process::Command};

use riscv_fuzz_plugin_api::IsaExtensions;

use crate::error::BuildElfError;
use crate::{mabi::mabi_from_extensions, march::march_from_extensions};

#[derive(Debug, Clone)]
pub struct ElfBuildResult {
    pub assembly_file: PathBuf,
    pub object_file: PathBuf,
    pub executable_file: PathBuf,
    pub disassembly_file: PathBuf,
}

pub fn build_elf_with_extensions<P: AsRef<std::path::Path>>(
    assembly_file: P,
    extensions: &IsaExtensions,
    linker_script_content: &str,
) -> Result<ElfBuildResult, BuildElfError> {
    let march =
        march_from_extensions(extensions).map_err(|source| BuildElfError::MarchBuild { source })?;
    let mabi =
        mabi_from_extensions(extensions).map_err(|source| BuildElfError::MabiBuild { source })?;

    let tool_prefix = "riscv64-unknown-elf";

    let assembly_file_path = assembly_file.as_ref().to_path_buf();
    let linker_script_path = assembly_file_path.with_extension("ld");
    fs::write(&linker_script_path, linker_script_content).map_err(|source| {
        BuildElfError::LinkerScriptWrite {
            path: linker_script_path.clone(),
            source,
        }
    })?;

    let preprocessed_file: Result<PathBuf, BuildElfError> =
        if assembly_file_path.extension().is_some_and(|e| e == "S") {
            let preprocessed_path = assembly_file_path.with_extension("s");

            let gcc_cmd = format!("{}-gcc", tool_prefix);
            let mut cmd = Command::new(&gcc_cmd);
            cmd.arg(format!("-march={}", march));
            cmd.arg(format!("-mabi={}", mabi));
            cmd.arg("-E");
            cmd.arg(&assembly_file_path);
            cmd.arg("-o");
            cmd.arg(&preprocessed_path);

            execute_command(&mut cmd, &gcc_cmd, "preprocessing")?;
            Ok(preprocessed_path)
        } else {
            Ok(assembly_file_path.clone())
        };

    let input_for_assembler = preprocessed_file?;

    let object_file = assembly_file_path.with_extension("o");
    let as_cmd = format!("{}-as", tool_prefix);
    let mut cmd = Command::new(&as_cmd);
    cmd.arg(format!("-mabi={}", mabi));
    cmd.arg("-g")
        .arg("-o")
        .arg(&object_file)
        .arg(format!("-march={}", march))
        .arg(&input_for_assembler);

    execute_command(&mut cmd, &as_cmd, "assembling")?;

    let executable_file = assembly_file_path.with_extension("elf");
    let ld_cmd = format!("{}-ld", tool_prefix);
    let mut cmd = Command::new(&ld_cmd);
    cmd.arg("-o")
        .arg(&executable_file)
        .arg(&object_file)
        .arg("-T")
        .arg(linker_script_path);

    match extensions.isa_base() {
        riscv_fuzz_plugin_api::ISABase::Rv32 => cmd.arg("-m").arg("elf32lriscv"),
        riscv_fuzz_plugin_api::ISABase::Rv64 => cmd.arg("-m").arg("elf64lriscv"),
    };

    execute_command(&mut cmd, &ld_cmd, "linking")?;

    let disassembly_file = assembly_file_path.with_extension("dump");
    let objdump_cmd = format!("{}-objdump", tool_prefix);
    let mut cmd = Command::new(&objdump_cmd);
    cmd.arg("-d")
        .arg("-S")
        .arg(&executable_file)
        .arg("--disassembler-options=no-aliases,numeric");

    let output = execute_command(&mut cmd, &objdump_cmd, "disassembling")?;

    fs::write(&disassembly_file, output.stdout).map_err(|source| {
        BuildElfError::DisassemblyWrite {
            path: disassembly_file.clone(),
            source,
        }
    })?;

    Ok(ElfBuildResult {
        assembly_file: input_for_assembler,
        object_file,
        executable_file,
        disassembly_file,
    })
}

fn execute_command(
    cmd: &mut Command,
    command: &str,
    stage: &'static str,
) -> Result<std::process::Output, BuildElfError> {
    log::info!("Executing {} command: {:?}", stage, cmd);
    let output = cmd.output().map_err(|source| BuildElfError::CommandSpawn {
        stage,
        command: command.to_string(),
        source,
    })?;

    if output.status.success() {
        Ok(output)
    } else {
        let stderr = {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stderr_trimmed = stderr.trim();
            if stderr_trimmed.is_empty() {
                String::from_utf8_lossy(&output.stdout).trim().to_owned()
            } else {
                stderr_trimmed.to_owned()
            }
        };

        Err(BuildElfError::CommandFailure {
            stage,
            command: command.to_string(),
            stderr,
        })
    }
}

#[cfg(test)]
mod test {
    use riscv_fuzz_plugin_api::IsaExtensions;
    use riscv_instruction::RV64Extensions;

    #[test]
    fn test_build_elf() {
        let simple_asm = r#"    .section .text
    .globl _start
_start:
    li a0, 42
    li a7, 93
"#;
        let asm_path = std::env::temp_dir().join("test_simple.S");
        std::fs::write(&asm_path, simple_asm).unwrap();
        let extensions = IsaExtensions::rv64(vec![RV64Extensions::I]);
        let linker_script = r#"OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
  . = 0x80000000;
  .text : { *(.text*) }
  .data : { *(.data*) }
  .bss  : { *(.bss*) *(COMMON) }
}"#;

        let result = super::build_elf_with_extensions(&asm_path, &extensions, linker_script);
        assert!(result.is_ok());
        let build_result = result.unwrap();

        let _ = std::fs::remove_file(&asm_path);
        let _ = std::fs::remove_file(asm_path.with_extension("ld"));
        let _ = std::fs::remove_file(build_result.assembly_file);
        let _ = std::fs::remove_file(build_result.object_file);
        let _ = std::fs::remove_file(build_result.executable_file);
        let _ = std::fs::remove_file(build_result.disassembly_file);
    }
}
