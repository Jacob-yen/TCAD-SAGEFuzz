mod extension_requirements;
pub mod instruction_fixer;
mod instruction_semantics;
pub mod memory_access;
pub mod operand_extractor;
pub(crate) mod operand_inference;
pub mod parser;
pub mod report_generator;

#[cfg(test)]
mod tests {
    use super::memory_access::with_memory_access_info;
    use super::parser::parse_insts_from_riscv_unified_db;
    use riscv_instruction_schema::Instruction;
    use std::{
        path::Path,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn memory_access_annotation_covers_expected_instructions()
    -> Result<(), Box<dyn std::error::Error>> {
        // --- File-loading section, same as before ---
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let candidates = [
            "assets/riscv_instructions.json",
            "../assets/riscv_instructions.json",
            "../../assets/riscv_instructions.json",
        ];
        let mut found_path = None;
        for rel in candidates.iter() {
            let p = Path::new(manifest_dir).join(rel);
            if p.exists() {
                found_path = Some(p);
                break;
            }
        }
        let path = found_path.ok_or_else(|| {
            let tried = candidates
                .iter()
                .map(|r| format!("{}", Path::new(manifest_dir).join(r).display()))
                .collect::<Vec<_>>()
                .join(" | ");
            format!("未找到 riscv_instructions.json，尝试的路径: {}", tried)
        })?;
        let data = std::fs::read_to_string(&path)?;
        let instructions: Vec<Instruction> = serde_json::from_str(&data)?;
        let total_instructions = instructions.len();

        println!("\n内存访问指令及其地址构成 (新逻辑):");
        println!(
            "{:<22} | {:<15} | {:<15}",
            "指令名称", "基址寄存器", "偏移量操作数"
        );
        println!("{:-<22}-+-{:-<15}-+-{:-<15}", "", "", "");
        let (new_instructions, mem_instruction_count) = with_memory_access_info(instructions);

        for inst in new_instructions
            .iter()
            .filter(|inst| inst.memory_access.is_some())
        {
            let info = inst.memory_access.as_ref().unwrap();
            let base_disp = info
                .base_register_operand
                .as_deref()
                .or(info.base_register_fixed.as_deref())
                .unwrap_or("?");
            let off_disp = info.offset_operand.as_deref().unwrap_or("N/A (zero)");
            println!("{:<22} | {:<15} | {:<15}", inst.name, base_disp, off_disp);
        }

        println!("\n--- 总结 ---");
        println!(
            "从 {} 读取到 {} 条指令，共找到 {} 条内存访问指令。",
            path.display(),
            total_instructions,
            mem_instruction_count
        );

        // Assertion checks.
        assert!(
            mem_instruction_count >= 428,
            "过滤出的内存访问指令数量过少: {}",
            mem_instruction_count
        );
        assert!(
            mem_instruction_count <= total_instructions,
            "内存访问指令数量异常，超过了总指令数量"
        );

        let lw_inst = new_instructions.iter().find(|i| i.name == "lw").unwrap();
        assert_eq!(
            lw_inst
                .memory_access
                .as_ref()
                .unwrap()
                .base_register_operand,
            Some("xs1".to_string())
        );
        assert_eq!(
            lw_inst.memory_access.as_ref().unwrap().offset_operand,
            Some("imm".to_string())
        );

        let c_lwsp_inst = new_instructions
            .iter()
            .find(|i| i.name == "c.lwsp")
            .unwrap();
        assert_eq!(
            c_lwsp_inst
                .memory_access
                .as_ref()
                .unwrap()
                .base_register_fixed,
            Some("sp".to_string())
        );
        assert_eq!(
            c_lwsp_inst.memory_access.as_ref().unwrap().offset_operand,
            Some("uimm".to_string())
        );

        let amoswap_w_inst = new_instructions
            .iter()
            .find(|i| i.name == "amoswap.w")
            .unwrap();
        assert_eq!(
            amoswap_w_inst
                .memory_access
                .as_ref()
                .unwrap()
                .base_register_operand,
            Some("xs1".to_string())
        );
        assert_eq!(
            amoswap_w_inst
                .memory_access
                .as_ref()
                .unwrap()
                .offset_operand,
            None
        );

        if let Some(prefetch_i_inst) = new_instructions.iter().find(|i| i.name == "prefetch.i") {
            assert_eq!(
                prefetch_i_inst
                    .memory_access
                    .as_ref()
                    .unwrap()
                    .base_register_operand,
                Some("xs1".to_string())
            );
            assert_eq!(
                prefetch_i_inst
                    .memory_access
                    .as_ref()
                    .unwrap()
                    .offset_operand,
                Some("imm".to_string())
            );
        }

        Ok(())
    }

    #[test]
    fn parser_accepts_dynamic_extension_directories() -> Result<(), Box<dyn std::error::Error>> {
        let unique_suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "riscv-parser-dynamic-extension-{}-{}",
            std::process::id(),
            unique_suffix
        ));
        let extension_dir = root.join("arch/inst/Znewext");
        std::fs::create_dir_all(&extension_dir)?;

        let yaml = r#"
$schema: https://example.com/schema
kind: instruction
name: dyn.add
long_name: Dynamic Add
description: dynamic extension smoke test
definedBy:
  xlen: 32
base: 32
assembly: dyn.add rd, rs1
encoding:
  match: "0"
  variables:
    - name: rd
      location: "11-7"
    - name: rs1
      location: "19-15"
"#;

        let instruction_path = extension_dir.join("dyn.add.yaml");
        std::fs::write(&instruction_path, yaml)?;

        let instructions = parse_insts_from_riscv_unified_db(&root)?;

        std::fs::remove_dir_all(&root)?;

        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0].extension.to_string(), "Znewext");
        assert_eq!(instructions[0].name, "dyn.add");

        Ok(())
    }

    #[test]
    fn parser_prefers_defined_by_extension_name_over_directory_name()
    -> Result<(), Box<dyn std::error::Error>> {
        let unique_suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "riscv-parser-defined-by-extension-{}-{}",
            std::process::id(),
            unique_suffix
        ));
        let extension_dir = root.join("arch/inst/C");
        std::fs::create_dir_all(&extension_dir)?;

        let yaml = r#"
$schema: https://example.com/schema
kind: instruction
name: dyn.c
long_name: Dynamic Compressed
description: dynamic extension smoke test
definedBy:
  extension:
    name: Zca
assembly: dyn.c xs1
encoding:
  match: "0"
  variables:
    - name: xs1
      location: "11-7"
"#;

        let instruction_path = extension_dir.join("dyn.c.yaml");
        std::fs::write(&instruction_path, yaml)?;

        let instructions = parse_insts_from_riscv_unified_db(&root)?;

        std::fs::remove_dir_all(&root)?;

        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0].extension.to_string(), "Zca");
        assert_eq!(instructions[0].name, "dyn.c");

        Ok(())
    }

    #[test]
    fn parser_uses_extension_metadata_to_constrain_supported_isa_bases()
    -> Result<(), Box<dyn std::error::Error>> {
        let unique_suffix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "riscv-parser-extension-base-support-{}-{}",
            std::process::id(),
            unique_suffix
        ));
        let extension_dir = root.join("arch/inst/Zcf");
        let ext_metadata_dir = root.join("arch/ext");
        std::fs::create_dir_all(&extension_dir)?;
        std::fs::create_dir_all(&ext_metadata_dir)?;

        let extension_yaml = r#"
name: Zcf
requirements:
  allOf:
    - extension:
        name: F
    - xlen: 32
"#;
        std::fs::write(ext_metadata_dir.join("Zcf.yaml"), extension_yaml)?;

        let instruction_yaml = r#"
$schema: https://example.com/schema
kind: instruction
name: dyn.zcf
long_name: Dynamic Zcf
description: extension metadata xlen smoke test
definedBy:
  extension:
    name: Zcf
assembly: dyn.zcf xs1
encoding:
  match: "0"
  variables:
    - name: xs1
      location: "11-7"
"#;

        let instruction_path = extension_dir.join("dyn.zcf.yaml");
        std::fs::write(&instruction_path, instruction_yaml)?;

        let instructions = parse_insts_from_riscv_unified_db(&root)?;

        std::fs::remove_dir_all(&root)?;

        assert_eq!(instructions.len(), 1);
        assert_eq!(instructions[0].extension.to_string(), "Zcf");
        assert_eq!(
            instructions[0].isa_bases,
            vec![riscv_instruction_schema::ISABase::RV32]
        );

        Ok(())
    }
}
