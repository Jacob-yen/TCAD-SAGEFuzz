use riscv_instruction_schema::{
    AssemblySyntax, ISABase, ISAExtension, Instruction, Operand, OperandKind, OperandRestriction,
    RegisterRole,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;

pub fn generate_detailed_extension_report(
    instructions: &[Instruction],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# RISC-V 指令按扩展分类详细报告\n\n");

    // Capture the current time in a simplified form.
    report.push_str(&format!(
        "**报告生成时间**: {}\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));

    // Collect all instructions and group them by extension.
    let mut extension_groups: BTreeMap<ISAExtension, Vec<&Instruction>> = BTreeMap::new();

    for inst in instructions.iter() {
        extension_groups
            .entry(inst.extension.clone())
            .or_insert_with(Vec::new)
            .push(inst);
    }

    // Generate the overview table.
    report.push_str("## 📊 概览统计\n\n");
    report.push_str("| 扩展 | 标准指令数量 | 压缩指令数量 | 总计 | 描述 |\n");
    report.push_str("|------|-------------|-------------|------|------|\n");

    for (extension, instructions_in_ext) in &extension_groups {
        let standard_count = instructions_in_ext
            .iter()
            .filter(|inst| !inst.name.starts_with("c."))
            .count();
        let compressed_count = instructions_in_ext
            .iter()
            .filter(|inst| inst.name.starts_with("c."))
            .count();
        let total_count = instructions_in_ext.len();
        let description = get_extension_description(extension);

        report.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            extension, standard_count, compressed_count, total_count, description
        ));
    }

    report.push_str("\n");

    // Generate detailed tables for each extension.
    for (extension, instructions_in_ext) in &extension_groups {
        report.push_str(&format!("## 🔧 {} 扩展指令\n\n", extension));
        report.push_str(&format!(
            "**扩展描述**: {}\n\n",
            get_extension_description(extension)
        ));
        report.push_str(&format!(
            "**指令总数**: {} 条\n\n",
            instructions_in_ext.len()
        ));

        // Group by instruction kind.
        let mut standard_instructions: Vec<&Instruction> = Vec::new();
        let mut compressed_instructions: Vec<&Instruction> = Vec::new();

        for inst in instructions_in_ext {
            if inst.name.starts_with("c.") {
                compressed_instructions.push(inst);
            } else {
                standard_instructions.push(inst);
            }
        }

        // Standard-instruction table.
        if !standard_instructions.is_empty() {
            report.push_str("### 📝 标准指令\n\n");
            generate_instruction_table(&mut report, &standard_instructions);
            report.push_str("\n");
        }

        // Compressed-instruction table.
        if !compressed_instructions.is_empty() {
            report.push_str("### 📦 压缩指令\n\n");
            generate_instruction_table(&mut report, &compressed_instructions);
            report.push_str("\n");
        }

        report.push_str("---\n\n");
    }

    // Generate the ISA-base compatibility report.
    report.push_str("## 🏗️ ISA基础架构兼容性\n\n");
    generate_isa_compatibility_report(&mut report, instructions);

    // Generate the operand-usage report.
    report.push_str("## 📋 操作数使用统计\n\n");
    generate_operand_usage_report(&mut report, instructions);

    // Generate statistics grouped by operand count.
    report.push_str("## 📏 按操作数数量分组统计\n\n");
    generate_operand_count_statistics(&mut report, instructions);

    fs::write(output_path, report)?;

    Ok(())
}

fn generate_instruction_table(report: &mut String, instructions: &[&Instruction]) {
    report.push_str("| 指令名称 | ISA支持 | 操作数数量 | 汇编语法 | 操作数名称 | 操作数长度(RV32/RV64) | 操作数限制 |\n");
    report.push_str("|----------|---------|-----------|----------|------------|-------------------|----------|\n");

    let mut sorted_instructions = instructions.to_vec();
    sorted_instructions.sort_by(|a, b| a.name.cmp(&b.name));

    for inst in sorted_instructions {
        let isa_support = format_isa_bases(&inst.isa_bases);
        let operand_count = inst.operands.len();
        let assembly_syntax = format_assembly_syntax(&inst.assembly_syntax);
        let operand_names = format_operand_names(&inst.operands);
        let operand_lengths = format_operand_lengths(&inst.operands);
        let operand_restrictions = format_operand_restrictions(&inst.operands);

        report.push_str(&format!(
            "| `{}` | {} | {} | `{}` | {} | {} | {} |\n",
            inst.name,
            isa_support,
            operand_count,
            escape_markdown(&assembly_syntax),
            operand_names,
            operand_lengths,
            operand_restrictions
        ));
    }
}

fn generate_isa_compatibility_report(report: &mut String, instructions: &[Instruction]) {
    let mut rv32_only = 0;
    let mut rv64_only = 0;
    let mut both = 0;
    let mut rv32_extensions: BTreeSet<ISAExtension> = BTreeSet::new();
    let mut rv64_extensions: BTreeSet<ISAExtension> = BTreeSet::new();
    let mut both_extensions: BTreeSet<ISAExtension> = BTreeSet::new();

    for inst in instructions.iter() {
        let has_rv32 = inst.isa_bases.contains(&ISABase::RV32);
        let has_rv64 = inst.isa_bases.contains(&ISABase::RV64);

        match (has_rv32, has_rv64) {
            (true, true) => {
                both += 1;
                both_extensions.insert(inst.extension.clone());
            }
            (true, false) => {
                rv32_only += 1;
                rv32_extensions.insert(inst.extension.clone());
            }
            (false, true) => {
                rv64_only += 1;
                rv64_extensions.insert(inst.extension.clone());
            }
            (false, false) => {} // 不应该发生
        }
    }

    report.push_str("| ISA基础 | 指令数量 | 相关扩展 |\n");
    report.push_str("|---------|----------|----------|\n");
    report.push_str(&format!(
        "| 仅RV32 | {} | {} |\n",
        rv32_only,
        format_extension_list(&rv32_extensions)
    ));
    report.push_str(&format!(
        "| 仅RV64 | {} | {} |\n",
        rv64_only,
        format_extension_list(&rv64_extensions)
    ));
    report.push_str(&format!(
        "| RV32和RV64 | {} | {} |\n",
        both,
        format_extension_list(&both_extensions)
    ));

    report.push_str("\n");
}

fn generate_operand_usage_report(report: &mut String, instructions: &[Instruction]) {
    let mut operand_counts: HashMap<String, usize> = HashMap::new();
    let mut operand_in_extensions: HashMap<String, BTreeSet<ISAExtension>> = HashMap::new();
    let mut operand_lengths: HashMap<String, HashMap<ISABase, BTreeSet<u8>>> = HashMap::new();
    let mut operand_restrictions_map: HashMap<String, Vec<String>> = HashMap::new();
    let mut operand_semantics: HashMap<String, BTreeSet<String>> = HashMap::new();

    for inst in instructions.iter() {
        for operand in &inst.operands {
            *operand_counts.entry(operand.name.clone()).or_insert(0) += 1;
            operand_in_extensions
                .entry(operand.name.clone())
                .or_insert_with(BTreeSet::new)
                .insert(inst.extension.clone());

            // Collect operand semantic information.
            operand_semantics
                .entry(operand.name.clone())
                .or_insert_with(BTreeSet::new)
                .insert(format_operand_semantics(operand, inst));

            // Collect operand bit-length information.
            let lengths = operand_lengths
                .entry(operand.name.clone())
                .or_insert_with(HashMap::new);
            for (base, length) in &operand.bit_lengths {
                lengths
                    .entry(*base)
                    .or_insert_with(BTreeSet::new)
                    .insert(*length);
            }

            // Collect operand restriction information.
            if let Some(restrictions) = &operand.restrictions {
                let restriction_desc = format_single_operand_restrictions(restrictions);
                if !restriction_desc.is_empty() {
                    operand_restrictions_map
                        .entry(operand.name.clone())
                        .or_insert_with(Vec::new)
                        .push(restriction_desc);
                }
            }
        }
    }

    let mut operand_vec: Vec<_> = operand_counts.iter().collect();
    operand_vec.sort_by_key(|(_, count)| std::cmp::Reverse(**count));

    report.push_str("### 🏷️ 操作数详细统计\n\n");
    report.push_str(
        "| 操作数名称 | 使用次数 | 操作数语义 | 出现在扩展 | 长度分布(RV32/RV64) | 限制条件 |\n",
    );
    report.push_str(
        "|------------|----------|------------|------------|-------------------|----------|\n",
    );

    for (operand_name, count) in operand_vec.iter() {
        let operand_semantics_str = operand_semantics
            .get(*operand_name)
            .map(|kinds| {
                if kinds.len() == 1 {
                    kinds.iter().next().cloned().unwrap()
                } else {
                    kinds.iter().cloned().collect::<Vec<_>>().join(", ")
                }
            })
            .unwrap_or_else(|| "未知".to_string());

        let extensions = operand_in_extensions
            .get(*operand_name)
            .map(|exts| format_extension_list(exts))
            .unwrap_or_default();

        let lengths = operand_lengths
            .get(*operand_name)
            .map(|length_map| format_operand_length_distribution(length_map))
            .unwrap_or_else(|| "未知".to_string());

        let restrictions = operand_restrictions_map
            .get(*operand_name)
            .map(|restr_vec| {
                let unique_restrictions: BTreeSet<_> = restr_vec.iter().collect();
                if unique_restrictions.is_empty() {
                    "无".to_string()
                } else {
                    unique_restrictions
                        .into_iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("; ")
                }
            })
            .unwrap_or_else(|| "无".to_string());

        report.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} |\n",
            operand_name, count, operand_semantics_str, extensions, lengths, restrictions
        ));
    }

    report.push_str("\n");

    // Append operand-length statistics.
    generate_operand_length_statistics(report, instructions);

    // Append operand-restriction statistics.
    generate_operand_restriction_statistics(report, instructions);
}

fn generate_operand_count_statistics(report: &mut String, instructions: &[Instruction]) {
    let mut operand_count_stats: HashMap<usize, Vec<String>> = HashMap::new();

    for inst in instructions.iter() {
        let count = inst.operands.len();
        operand_count_stats
            .entry(count)
            .or_insert_with(Vec::new)
            .push(inst.name.clone());
    }

    let mut count_vec: Vec<_> = operand_count_stats.iter().collect();
    count_vec.sort_by_key(|(count, _)| **count);

    report.push_str("| 操作数数量 | 指令数量 | 示例指令 |\n");
    report.push_str("|-----------|----------|----------|\n");

    for (count, inst_names) in count_vec {
        let examples = if inst_names.len() <= 5 {
            inst_names
                .iter()
                .map(|s| format!("`{}`", s))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            let first_five: Vec<String> = inst_names
                .iter()
                .take(5)
                .map(|s| format!("`{}`", s))
                .collect();
            format!("{}, ... (共{}条)", first_five.join(", "), inst_names.len())
        };

        report.push_str(&format!(
            "| {} | {} | {} |\n",
            count,
            inst_names.len(),
            examples
        ));
    }

    report.push_str("\n");
}

fn format_operand_semantics(operand: &Operand, instruction: &Instruction) -> String {
    if let Some(immediate_kind) = operand.runtime_immediate_kind(instruction) {
        return format!("{} Immediate", immediate_kind);
    }

    let isa_base = instruction
        .isa_bases
        .first()
        .copied()
        .or_else(|| operand.bit_lengths.keys().next().copied());
    if let Some(isa_base) = isa_base {
        if let Some(register_semantics) = operand.register_semantics(isa_base, instruction) {
            let mut label = register_semantics.runtime_type_label().to_string();
            match register_semantics.role {
                RegisterRole::Plain => {}
                RegisterRole::MemoryBase => label.push_str(" (memory base)"),
                RegisterRole::SavedAlias => label.push_str(" (ABI alias preferred)"),
            }
            return label;
        }
    }

    match &operand.kind {
        OperandKind::Unknown => "Unknown".to_string(),
        OperandKind::Register(_) | OperandKind::Immediate(_) => operand.kind.to_string(),
        OperandKind::Csr(_) => "CSR".to_string(),
        OperandKind::RoundMode => "Round Mode".to_string(),
        OperandKind::FenceMode => "Fence Mode".to_string(),
        OperandKind::FliConstant => "FLI Constant".to_string(),
        OperandKind::SavedRegListWithStackAdj { xlen_dependent } => {
            if *xlen_dependent {
                "Saved Register List With Stack Adjustment (xlen-dependent)".to_string()
            } else {
                "Saved Register List With Stack Adjustment".to_string()
            }
        }
        OperandKind::RegisterPair(spec) => {
            let mut label = format!("{} Register Pair", spec.bank);
            if spec.distinct {
                label.push_str(" (distinct)");
            }
            label
        }
    }
}

fn get_extension_description(extension: &ISAExtension) -> String {
    extension.to_string()
}

fn format_isa_bases(bases: &[ISABase]) -> String {
    if bases.len() == 2 {
        "RV32/64".to_string()
    } else if bases.contains(&ISABase::RV32) {
        "RV32".to_string()
    } else if bases.contains(&ISABase::RV64) {
        "RV64".to_string()
    } else {
        "未知".to_string()
    }
}

fn format_assembly_syntax(syntax: &AssemblySyntax) -> String {
    match syntax {
        AssemblySyntax::Format(format) => format.clone(),
        AssemblySyntax::RustCode(_) => {
            format!("Rust代码: 略")
        }
    }
}

fn format_extension_list(extensions: &BTreeSet<ISAExtension>) -> String {
    if extensions.len() <= 3 {
        extensions
            .iter()
            .map(|e| format!("{}", e))
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        let first_three: Vec<String> = extensions
            .iter()
            .take(3)
            .map(|e| format!("{}", e))
            .collect();
        format!("{}, ... (共{}个)", first_three.join(", "), extensions.len())
    }
}

fn escape_markdown(text: &str) -> String {
    text.replace("|", "|").replace("{", "{").replace("}", "}")
}

fn format_operand_names(operands: &[Operand]) -> String {
    if operands.is_empty() {
        "无".to_string()
    } else {
        operands
            .iter()
            .map(|op| format!("`{}`", op.name))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn format_operand_lengths(operands: &[Operand]) -> String {
    // Print per-instruction bit lengths for RV32 and RV64.
    if operands.is_empty() {
        "无".to_string()
    } else {
        let mut lengths = Vec::new();
        for op in operands {
            let mut length_str = String::new();
            for (base, length) in &op.bit_lengths {
                if !length_str.is_empty() {
                    length_str.push_str(", ");
                }
                length_str.push_str(&format!("{}:{}", base, length));
            }
            lengths.push(format!("`{}`: {}", op.name, length_str));
        }
        lengths.join("; ")
    }
}

fn format_operand_restrictions(operands: &[Operand]) -> String {
    if operands.is_empty() {
        "无".to_string()
    } else {
        let restrictions: Vec<String> = operands
            .iter()
            .filter_map(|op| {
                op.restrictions
                    .as_ref()
                    .map(|r| {
                        let desc = format_single_operand_restrictions(r);
                        if desc.is_empty() {
                            None
                        } else {
                            Some(format!("`{}`: {}", op.name, desc))
                        }
                    })
                    .flatten()
            })
            .collect();

        if restrictions.is_empty() {
            "无".to_string()
        } else {
            restrictions.join("; ")
        }
    }
}

fn format_single_operand_restrictions(restrictions: &OperandRestriction) -> String {
    let mut parts = Vec::new();

    if let Some(multiple) = restrictions.multiple_of {
        parts.push(format!("{}的倍数", multiple));
    }

    if let Some((min, max)) = restrictions.min_max {
        parts.push(format!("范围[{},{}]", min, max));
    }

    if !restrictions.forbidden_values.is_empty() {
        let forbidden_str = restrictions
            .forbidden_values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        parts.push(format!("禁止:{}", forbidden_str));
    }

    if restrictions.odd_only {
        parts.push("奇数".to_string());
    }

    parts.join(", ")
}
fn format_operand_length_distribution(length_map: &HashMap<ISABase, BTreeSet<u8>>) -> String {
    let format_lengths = |base: &ISABase| {
        length_map.get(base).map(|set| {
            set.iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
    };

    let rv32_lengths = format_lengths(&ISABase::RV32);
    let rv64_lengths = format_lengths(&ISABase::RV64);

    match (rv32_lengths, rv64_lengths) {
        (Some(rv32), Some(rv64)) => format!("RV32:{}, RV64:{}", rv32, rv64),
        (Some(rv32), None) => format!("RV32:{}, RV64:无", rv32),
        (None, Some(rv64)) => format!("RV32:无, RV64:{}", rv64),
        (None, None) => "RV32:无, RV64:无".to_string(),
    }
}

fn generate_operand_length_statistics(report: &mut String, instructions: &[Instruction]) {
    let mut length_stats: HashMap<(ISABase, u8), usize> = HashMap::new();

    for inst in instructions.iter() {
        for operand in &inst.operands {
            for (base, length) in &operand.bit_lengths {
                *length_stats.entry((*base, *length)).or_insert(0) += 1;
            }
        }
    }

    report.push_str("### 📐 操作数长度分布统计\n\n");
    report.push_str("| ISA基础 | 位长度 | 使用次数 | 占比 |\n");
    report.push_str("|---------|--------|----------|------|\n");

    let total_operands: usize = length_stats.values().sum();
    let mut stats_vec: Vec<_> = length_stats.iter().collect();
    stats_vec.sort_by_key(|((base, length), _)| (*base, *length));

    for ((base, length), count) in stats_vec {
        let percentage = if total_operands > 0 {
            *count as f64 / total_operands as f64 * 100.0
        } else {
            0.0
        };

        report.push_str(&format!(
            "| {} | {} | {} | {:.1}% |\n",
            base, length, count, percentage
        ));
    }

    report.push_str("\n");
}

fn generate_operand_restriction_statistics(report: &mut String, instructions: &[Instruction]) {
    let mut restriction_stats: HashMap<String, usize> = HashMap::new();
    let mut total_operands = 0;
    let mut restricted_operands = 0;

    for inst in instructions.iter() {
        for operand in &inst.operands {
            total_operands += 1;

            if let Some(restrictions) = &operand.restrictions {
                restricted_operands += 1;

                if restrictions.multiple_of.is_some() {
                    *restriction_stats.entry("倍数限制".to_string()).or_insert(0) += 1;
                }

                if restrictions.min_max.is_some() {
                    *restriction_stats.entry("范围限制".to_string()).or_insert(0) += 1;
                }

                if !restrictions.forbidden_values.is_empty() {
                    *restriction_stats
                        .entry("禁止值限制".to_string())
                        .or_insert(0) += 1;
                }

                if restrictions.odd_only {
                    *restriction_stats.entry("奇数限制".to_string()).or_insert(0) += 1;
                }
            }
        }
    }

    report.push_str("### 🚫 操作数限制条件统计\n\n");
    report.push_str("| 限制类型 | 使用次数 | 占受限操作数比例 |\n");
    report.push_str("|----------|----------|------------------|\n");

    for (restriction_type, count) in restriction_stats.iter() {
        let percentage = if restricted_operands > 0 {
            *count as f64 / restricted_operands as f64 * 100.0
        } else {
            0.0
        };

        report.push_str(&format!(
            "| {} | {} | {:.1}% |\n",
            restriction_type, count, percentage
        ));
    }

    report.push_str(&format!(
        "\n**总操作数**: {} 个，**受限操作数**: {} 个 ({:.1}%)\n\n",
        total_operands,
        restricted_operands,
        if total_operands > 0 {
            restricted_operands as f64 / total_operands as f64 * 100.0
        } else {
            0.0
        }
    ));
}
