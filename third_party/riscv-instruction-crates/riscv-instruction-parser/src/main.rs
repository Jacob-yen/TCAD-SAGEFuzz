use riscv_instruction_parser::{
    instruction_fixer::fix_instructions_and_write_audit_reports,
    memory_access::with_memory_access_info, parser::parse_insts_from_riscv_unified_db,
    report_generator::generate_detailed_extension_report,
};
use riscv_instruction_schema::save_instructions_to_json;
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    let input_path = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/riscv-unified-db"));
    let output_dir = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets"));

    fs::create_dir_all(&output_dir)?;

    let mut instructions = parse_insts_from_riscv_unified_db(&input_path)?;
    println!("总指令数量: {}", instructions.len());

    let (fixer_report, fixer_report_paths) =
        fix_instructions_and_write_audit_reports(&mut instructions, &input_path, &output_dir)?;
    println!("修复完成！");
    println!("发生变化的指令数量: {}", fixer_report.changed_instructions);
    println!("修复审计目录: {}", fixer_report_paths.output_dir.display());
    println!("修复审计 JSON: {}", fixer_report_paths.json_path.display());
    println!(
        "修复审计 Markdown: {}",
        fixer_report_paths.repair_markdown_path.display()
    );

    let (instructions, mem_instruction_count) = with_memory_access_info(instructions);

    let json_path = output_dir.join("riscv_instructions.json");
    save_instructions_to_json(&instructions, &json_path)?;
    println!(
        "已保存为 {} (含内存访问信息，共 {} 条内存相关指令)",
        json_path.display(),
        mem_instruction_count
    );

    let report_path = output_dir.join("riscv_detailed_extension_report.md");
    generate_detailed_extension_report(&instructions, &report_path.to_string_lossy())?;
    println!("已生成详细扩展报告 {}", report_path.display());

    Ok(())
}
