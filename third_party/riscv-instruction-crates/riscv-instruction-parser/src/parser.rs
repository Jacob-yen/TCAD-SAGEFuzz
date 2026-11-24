use crate::{
    extension_requirements::{
        ExtensionRequirementClosure, ExtensionSupportedBases, load_extension_metadata_index,
        resolve_instruction_required_extensions,
    },
    instruction_semantics::infer_instruction_semantics,
    operand_extractor::extract_operands_from_asm_without_name,
    operand_inference::{infer_operand_bit_length, is_synthesizable_operand_name},
};
use riscv_instruction_schema::*;
use std::{
    collections::{HashMap, HashSet}, // Keep `HashSet` imported here.
    fs,
    path::{Path, PathBuf},
};

pub fn parse_insts_from_riscv_unified_db<P: AsRef<Path>>(
    riscv_unified_db_repo_path: P,
) -> Result<Vec<Instruction>, Box<dyn std::error::Error>> {
    let inst_path = resolve_inst_path(riscv_unified_db_repo_path.as_ref())?;
    let extension_metadata_index =
        load_extension_metadata_index(riscv_unified_db_repo_path.as_ref())?;

    let mut all_instructions = Vec::new();
    let mut extension_dirs = fs::read_dir(&inst_path)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    extension_dirs.sort();

    // Read all extension directories.
    for path in extension_dirs {
        let extension_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        let fallback_extension = ISAExtension::new(extension_name);
        let mut instruction_files = fs::read_dir(&path)?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("yaml"))
            .collect::<Vec<_>>();
        instruction_files.sort();

        // Read all YAML files in each extension directory.
        for inst_path in instruction_files {
            let instruction = parse_instruction_file(
                &inst_path,
                &fallback_extension,
                &extension_metadata_index.requirement_closure,
                &extension_metadata_index.supported_bases,
            )?;
            all_instructions.push(instruction);
        }
    }

    Ok(all_instructions)
}

fn resolve_inst_path(root: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let candidates = [
        root.join("arch/inst"),
        root.join("spec/std/isa/inst"),
        root.to_path_buf(),
    ];

    for candidate in candidates {
        if looks_like_inst_root(&candidate)? {
            return Ok(candidate);
        }
    }

    Err(format!(
        "未找到指令定义目录，尝试过: {}",
        [
            root.join("arch/inst"),
            root.join("spec/std/isa/inst"),
            root.to_path_buf(),
        ]
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(" | ")
    )
    .into())
}

fn looks_like_inst_root(path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    if !path.is_dir() {
        return Ok(false);
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if !child.is_dir() {
            continue;
        }

        let has_yaml = fs::read_dir(&child)?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .any(|file| file.path().extension().and_then(|s| s.to_str()) == Some("yaml"));

        if has_yaml {
            return Ok(true);
        }
    }

    Ok(false)
}

fn parse_instruction_file(
    file_path: &Path,
    fallback_extension: &ISAExtension,
    extension_requirement_closure: &ExtensionRequirementClosure,
    extension_supported_bases: &ExtensionSupportedBases,
) -> Result<Instruction, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;
    let yaml_inst: YamlInstruction = serde_yaml::from_str(&content)?;
    let extension = resolve_instruction_extension(&yaml_inst, fallback_extension);

    // Determine the supported ISA bases.
    let isa_bases = determine_isa_bases(&yaml_inst, &extension, extension_supported_bases);
    let required_extensions = resolve_instruction_required_extensions(
        &yaml_inst,
        &extension,
        &isa_bases,
        extension_requirement_closure,
    );

    // Parse operands.
    let operands = parse_operands(&yaml_inst, &isa_bases, file_path)?;
    let semantics = infer_instruction_semantics(&yaml_inst, &operands);

    // Clear the assembly syntax when it is identical to the instruction name,
    // or to the instruction name with dots replaced by underscores.
    let name_with_underscores = yaml_inst.name.replace('.', "_");
    let assembly_syntax =
        if yaml_inst.assembly == yaml_inst.name || yaml_inst.assembly == name_with_underscores {
            AssemblySyntax::Format(String::new())
        } else {
            AssemblySyntax::Format(yaml_inst.assembly.clone())
        };

    let instruction = Instruction {
        name: yaml_inst.name,
        extension,
        isa_bases,
        required_extensions,
        operands,
        assembly_syntax,
        memory_access: None,
        semantics,
    };

    Ok(instruction)
}

fn resolve_instruction_extension(
    yaml_inst: &YamlInstruction,
    fallback_extension: &ISAExtension,
) -> ISAExtension {
    yaml_inst
        .defined_by
        .as_mapping()
        .and_then(|mapping| mapping.get(serde_yaml::Value::String("extension".to_string())))
        .and_then(|value| value.as_mapping())
        .and_then(|mapping| mapping.get(serde_yaml::Value::String("name".to_string())))
        .and_then(|value| value.as_str())
        .map(ISAExtension::new)
        .unwrap_or_else(|| fallback_extension.clone())
}

fn parse_operands(
    yaml_inst: &YamlInstruction,
    isa_bases: &[ISABase], // New parameter.
    file_path: &Path,
) -> Result<Vec<Operand>, Box<dyn std::error::Error>> {
    let mut operands = Vec::new();

    if let Some(encoding) = &yaml_inst.encoding {
        match encoding {
            Encoding::Simple { variables, .. } => {
                if let Some(variables) = variables {
                    for var in variables {
                        // Simple encodings use the same bit width for every ISA base,
                        // but we still populate only the bases the instruction actually supports.
                        let bit_lengths_map = parse_bit_range_simple(&var.location, isa_bases)?;
                        let operand_bit_length =
                            bit_lengths_map.values().next().copied().unwrap_or(0);
                        let max_value_for_operand =
                            if operand_bit_length > 0 && operand_bit_length < 8 {
                                ((1u16 << operand_bit_length) - 1) as u64
                            } else if operand_bit_length == 8 {
                                u64::MAX
                            } else if operand_bit_length > 8 {
                                u64::MAX
                            } else {
                                0
                            };

                        let restrictions = parse_restrictions(var, max_value_for_operand);

                        let operand = Operand {
                            name: var.name.clone(),
                            kind: OperandKind::Unknown,
                            bit_lengths: bit_lengths_map,
                            restrictions,
                            per_isa_restrictions: HashMap::new(),
                        };

                        operands.push(operand);
                    }
                }
            }
            Encoding::PerISA { rv32, rv64 } => {
                // Collect all variable names.
                let mut all_variables: HashMap<String, HashMap<ISABase, String>> = HashMap::new();

                if let Some(rv32_enc) = rv32 {
                    if let Some(vars) = &rv32_enc.variables {
                        for var in vars {
                            all_variables
                                .entry(var.name.clone())
                                .or_insert_with(HashMap::new)
                                .insert(ISABase::RV32, var.location.clone());
                        }
                    }
                }

                if let Some(rv64_enc) = rv64 {
                    if let Some(vars) = &rv64_enc.variables {
                        for var in vars {
                            all_variables
                                .entry(var.name.clone())
                                .or_insert_with(HashMap::new)
                                .insert(ISABase::RV64, var.location.clone());
                        }
                    }
                }

                // Create an operand for each variable.
                for (var_name, locations) in all_variables {
                    let mut bit_lengths_map = HashMap::new();
                    let mut max_bit_length = 0u8;

                    // Compute bit lengths for each ISA base.
                    for (isa_base, location) in locations {
                        let bit_length = calculate_bit_length(&location)?;
                        bit_lengths_map.insert(isa_base, bit_length);
                        max_bit_length = max_bit_length.max(bit_length);
                    }

                    let max_value_for_operand = if max_bit_length > 0 && max_bit_length < 8 {
                        ((1u16 << max_bit_length) - 1) as u64
                    } else if max_bit_length == 8 {
                        u64::MAX
                    } else if max_bit_length > 8 {
                        u64::MAX
                    } else {
                        0
                    };

                    // Compute per-ISA restrictions from the ISA-specific variable definitions.
                    //
                    // For RV64-only instructions, use the RV64 encoding's constraints.
                    // For shared RV32+RV64 types where restrictions differ, store them in
                    // per_isa_restrictions so the code-gen can generate correctly-typed fields
                    // per ISA. When restrictions are identical, keep the shared restrictions field.
                    let rv64_only = isa_bases == [ISABase::RV64];

                    let get_restriction = |enc: Option<&SimpleEncoding>| -> Option<OperandRestriction> {
                        enc.and_then(|e| e.variables.as_ref())
                            .and_then(|vars| vars.iter().find(|v| v.name == var_name))
                            .and_then(|var| parse_restrictions(var, max_value_for_operand))
                    };
                    let var_in_enc = |enc: Option<&SimpleEncoding>| -> bool {
                        enc.and_then(|e| e.variables.as_ref())
                            .map(|vars| vars.iter().any(|v| v.name == var_name))
                            .unwrap_or(false)
                    };

                    let (restrictions, per_isa_restrictions) = if rv64_only {
                        let r = if var_in_enc(rv64.as_ref()) {
                            get_restriction(rv64.as_ref())
                        } else {
                            get_restriction(rv32.as_ref())
                        };
                        (r, HashMap::new())
                    } else {
                        let rv32_restriction = get_restriction(rv32.as_ref());
                        let rv64_restriction = get_restriction(rv64.as_ref());
                        if rv32_restriction == rv64_restriction {
                            // Same (or both None): store as shared restriction
                            (rv32_restriction, HashMap::new())
                        } else {
                            // Differ: store each ISA's restriction separately; common fallback = None
                            let mut per_isa = HashMap::new();
                            per_isa.insert(ISABase::RV32, rv32_restriction);
                            per_isa.insert(ISABase::RV64, rv64_restriction);
                            (None, per_isa)
                        }
                    };

                    let operand = Operand {
                        name: var_name,
                        kind: OperandKind::Unknown,
                        bit_lengths: bit_lengths_map,
                        restrictions,
                        per_isa_restrictions,
                    };

                    operands.push(operand);
                }
            }
        }
    } else if let Some(format_value) = &yaml_inst.format {
        operands = parse_operands_from_format(format_value, isa_bases, file_path)?;
    }

    if operands.is_empty() {
        operands = infer_operands_from_assembly(&yaml_inst.assembly, isa_bases);
    }

    Ok(operands)
}

fn infer_operands_from_assembly(assembly: &str, isa_bases: &[ISABase]) -> Vec<Operand> {
    let mut inferred: Vec<String> = extract_operands_from_asm_without_name(assembly)
        .into_iter()
        .filter(|name| is_synthesizable_operand_name(name))
        .collect();
    inferred.sort();

    inferred
        .into_iter()
        .map(|name| Operand {
            bit_lengths: isa_bases
                .iter()
                .copied()
                .map(|base| (base, infer_operand_bit_length(&name)))
                .collect(),
            name,
            kind: OperandKind::Unknown,
            restrictions: None,
            per_isa_restrictions: HashMap::new(),
        })
        .collect()
}

fn parse_operands_from_format(
    format_value: &serde_yaml::Value,
    isa_bases: &[ISABase],
    file_path: &Path,
) -> Result<Vec<Operand>, Box<dyn std::error::Error>> {
    let isa_root = resolve_isa_definition_root(file_path)?;
    let Some(inherits_value) = format_value
        .as_mapping()
        .and_then(|mapping| mapping.get(serde_yaml::Value::String("$inherits".to_string())))
    else {
        return Ok(Vec::new());
    };

    let inherit_refs = match inherits_value {
        serde_yaml::Value::String(value) => vec![value.clone()],
        serde_yaml::Value::Sequence(values) => values
            .iter()
            .filter_map(|value| value.as_str().map(str::to_owned))
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };

    let Some(subtype_ref) = inherit_refs
        .iter()
        .find(|inherit_ref| inherit_ref.starts_with("inst_subtype/"))
    else {
        return Ok(Vec::new());
    };

    let subtype_data = load_yaml_ref_value(&isa_root, subtype_ref)?;
    let Some(variables_mapping) = subtype_data
        .as_mapping()
        .and_then(|mapping| mapping.get(serde_yaml::Value::String("variables".to_string())))
        .and_then(|value| value.as_mapping())
    else {
        return Ok(Vec::new());
    };

    let mut operands = Vec::new();

    for (operand_name_value, variable_ref_value) in variables_mapping {
        let Some(operand_name) = operand_name_value.as_str() else {
            continue;
        };

        let Some(variable_ref) = variable_ref_value
            .as_mapping()
            .and_then(|mapping| mapping.get(serde_yaml::Value::String("$inherits".to_string())))
            .and_then(|value| value.as_str())
        else {
            continue;
        };

        let variable_data = load_yaml_ref_value(&isa_root, variable_ref)?;
        let inherited_variable_ref = variable_data
            .as_mapping()
            .and_then(|mapping| mapping.get(serde_yaml::Value::String("$inherits".to_string())))
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                format!(
                    "格式继承变量 {} 缺少 $inherits: {}",
                    operand_name, variable_ref
                )
            })?;

        let inst_type_variable_data = load_yaml_ref_value(&isa_root, inherited_variable_ref)?;
        let location = inst_type_variable_data
            .as_mapping()
            .and_then(|mapping| mapping.get(serde_yaml::Value::String("location".to_string())))
            .and_then(|value| match value {
                serde_yaml::Value::String(s) => Some(s.clone()),
                serde_yaml::Value::Number(n) => Some(n.to_string()),
                _ => None,
            })
            .ok_or_else(|| {
                format!(
                    "无法从格式继承解析 location: {} -> {}",
                    variable_ref, inherited_variable_ref
                )
            })?;

        let bit_lengths_map = parse_bit_range_simple(&location, isa_bases)?;
        let operand_bit_length = bit_lengths_map.values().next().copied().unwrap_or(0);
        let max_value_for_operand = if operand_bit_length > 0 && operand_bit_length < 8 {
            ((1u16 << operand_bit_length) - 1) as u64
        } else if operand_bit_length >= 8 {
            u64::MAX
        } else {
            0
        };

        let restrictions = parse_restrictions_from_mapping(&variable_data, max_value_for_operand);

        operands.push(Operand {
            name: operand_name.to_string(),
            kind: OperandKind::Unknown,
            bit_lengths: bit_lengths_map,
            restrictions,
            per_isa_restrictions: HashMap::new(),
        });
    }

    Ok(operands)
}

fn resolve_isa_definition_root(file_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    for ancestor in file_path.ancestors() {
        if ancestor.join("inst_subtype").is_dir() && ancestor.join("inst_type").is_dir() {
            return Ok(ancestor.to_path_buf());
        }
    }

    Err(format!("无法从路径推导 ISA 定义根目录: {}", file_path.display()).into())
}

fn load_yaml_ref_value(
    isa_root: &Path,
    reference: &str,
) -> Result<serde_yaml::Value, Box<dyn std::error::Error>> {
    let (relative_path, fragment) = reference
        .split_once('#')
        .map(|(path, fragment)| (path, Some(fragment)))
        .unwrap_or((reference, None));
    let full_path = isa_root.join(relative_path);
    let yaml_value: serde_yaml::Value = serde_yaml::from_str(&fs::read_to_string(&full_path)?)?;

    if let Some(fragment) = fragment {
        resolve_yaml_fragment(&yaml_value, fragment)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "无法解析 YAML 引用片段 {}#{}",
                    full_path.display(),
                    fragment
                )
            })
            .map_err(Into::into)
    } else {
        Ok(yaml_value)
    }
}

fn resolve_yaml_fragment<'a>(
    value: &'a serde_yaml::Value,
    fragment: &str,
) -> Option<&'a serde_yaml::Value> {
    let mut current = value;
    for segment in fragment.trim_start_matches('/').split('/') {
        if segment.is_empty() {
            continue;
        }
        current = current
            .as_mapping()?
            .get(serde_yaml::Value::String(segment.to_string()))?;
    }
    Some(current)
}

fn parse_restrictions_from_mapping(
    mapping_value: &serde_yaml::Value,
    max_value_for_operand: u64,
) -> Option<OperandRestriction> {
    let Some(mapping) = mapping_value.as_mapping() else {
        return None;
    };

    let variable = Variable {
        name: String::new(),
        location: String::new(),
        sign_extend: None,
        left_shift: mapping
            .get(serde_yaml::Value::String("left_shift".to_string()))
            .and_then(|value| value.as_u64())
            .and_then(|value| u8::try_from(value).ok()),
        alias: mapping
            .get(serde_yaml::Value::String("alias".to_string()))
            .and_then(|value| value.as_str())
            .map(str::to_owned),
        not_values: mapping
            .get(serde_yaml::Value::String("not".to_string()))
            .cloned(),
    };

    parse_restrictions(&variable, max_value_for_operand)
}
fn parse_bit_range_simple(
    location: &str,
    isa_bases: &[ISABase], // New parameter.
) -> Result<HashMap<ISABase, u8>, Box<dyn std::error::Error>> {
    let bit_length = calculate_bit_length(location)?;

    let mut bit_lengths = HashMap::new();
    for base in isa_bases {
        bit_lengths.insert(*base, bit_length);
    }

    Ok(bit_lengths)
}

fn calculate_bit_length(location: &str) -> Result<u8, Box<dyn std::error::Error>> {
    let mut total_length = 0u8;

    // Handle multiple bit ranges separated by `|`, for example `31-25|11-7` or `12|6-2`.
    for part in location.split('|') {
        let part = part.trim();

        if let Some((high_str, low_str)) = part.split_once('-') {
            // Bit-range format: high-low.
            let high: u8 = high_str
                .trim()
                .parse()
                .map_err(|e| format!("无法解析高位 '{}': {}", high_str.trim(), e))?;
            let low: u8 = low_str
                .trim()
                .parse()
                .map_err(|e| format!("无法解析低位 '{}': {}", low_str.trim(), e))?;

            if high < low {
                return Err(format!("位范围无效: {} < {}", high, low).into());
            }

            let range_length = high - low + 1;
            total_length += range_length;
        } else {
            total_length += 1;
        }
    }

    Ok(total_length)
}

fn parse_restrictions(var: &Variable, max_value_for_operand: u64) -> Option<OperandRestriction> {
    let mut restriction = OperandRestriction::default();
    let mut has_any_restriction_field_set = false;

    // Parse the `left_shift` restriction.
    if let Some(left_shift_yaml) = &var.left_shift {
        has_any_restriction_field_set = true;
        let multiple_of_value = 1u16 << left_shift_yaml; // 2^shift_amount
        if multiple_of_value <= u16::MAX {
            restriction.multiple_of = Some(multiple_of_value.try_into().unwrap());
        }
    }

    // Parse the `not` restriction.
    if let Some(not_values_yaml) = &var.not_values {
        has_any_restriction_field_set = true;
        let mut temp_forbidden_values = HashSet::new();

        match not_values_yaml {
            serde_yaml::Value::Number(n) => {
                if let Some(val_u64) = n.as_u64() {
                    if val_u64 <= i64::MAX as u64 {
                        let val_i64 = val_u64 as i64;
                        temp_forbidden_values.insert(val_i64);
                    }
                }
            }
            serde_yaml::Value::Sequence(seq) => {
                for item in seq {
                    if let Some(n_u64) = item.as_u64() {
                        if n_u64 <= i64::MAX as u64 {
                            let val_i64 = n_u64 as i64;
                            temp_forbidden_values.insert(val_i64);
                        }
                    }
                }
            }
            _ => {} // Ignore other value kinds.
        }

        // If `multiple_of` is already present, filter `forbidden_values` against it.
        if let Some(multiple_of) = restriction.multiple_of {
            // Remove forbidden values that already violate `multiple_of`; they are invalid anyway.
            temp_forbidden_values.retain(|&val| val % (multiple_of as i64) == 0);

            // Store the filtered forbidden values directly.
            if !temp_forbidden_values.is_empty() {
                let mut forbidden_values: Vec<i64> = temp_forbidden_values.into_iter().collect();
                forbidden_values.sort_unstable();
                restriction.forbidden_values = forbidden_values;
            }
        } else {
            // Check whether all odd values are forbidden, which implies `multiple_of: 2`.
            let all_odd_numbers_in_range: HashSet<i64> =
                (1..=max_value_for_operand as i64).step_by(2).collect();

            if !all_odd_numbers_in_range.is_empty()
                && all_odd_numbers_in_range.is_subset(&temp_forbidden_values)
            {
                restriction.multiple_of = Some(2);

                // Compute the values explained by that pattern.
                let mut explained_by_pattern = all_odd_numbers_in_range;
                // Include 0 if it is also forbidden.
                if temp_forbidden_values.contains(&0) {
                    explained_by_pattern.insert(0);
                }

                // Remaining forbidden values not covered by `multiple_of: 2` and possibly forbidden 0.
                let remaining_forbidden: Vec<i64> = temp_forbidden_values
                    .difference(&explained_by_pattern)
                    .cloned()
                    .collect();

                let mut sorted_remaining: Vec<i64> = remaining_forbidden;
                sorted_remaining.sort_unstable(); // Keep the output stable.
                restriction.forbidden_values = sorted_remaining;
            } else {
                // If the `multiple_of: 2` pattern does not apply, keep the collected forbidden values.
                let mut all_forbidden_from_yaml: Vec<_> =
                    temp_forbidden_values.into_iter().collect();
                all_forbidden_from_yaml.sort_unstable(); // Keep the output stable.
                restriction.forbidden_values = all_forbidden_from_yaml;
            }
        }
    }

    // Return `Some` whenever any restriction field was explicitly set.
    if restriction != OperandRestriction::default() || has_any_restriction_field_set {
        if restriction != OperandRestriction::default() {
            Some(restriction)
        } else if has_any_restriction_field_set {
            // For cases such as `not: []`.
            Some(OperandRestriction::default()) // Return an explicit but empty restriction.
        } else {
            None
        }
    } else {
        None
    }
}

fn determine_isa_bases(
    yaml_inst: &YamlInstruction,
    extension: &ISAExtension,
    extension_supported_bases: &ExtensionSupportedBases,
) -> Vec<ISABase> {
    // First check whether the instruction declares an explicit `base` field.
    if let Some(base_value) = &yaml_inst.base {
        match base_value {
            32 => return vec![ISABase::RV32],
            64 => return vec![ISABase::RV64],
            _ => {}
        }
    }

    let mut xlen_constraints = HashSet::new();
    collect_defined_by_xlens(&yaml_inst.defined_by, &mut xlen_constraints);

    if xlen_constraints.len() == 1 {
        if xlen_constraints.contains(&32) {
            return vec![ISABase::RV32];
        }
        if xlen_constraints.contains(&64) {
            return vec![ISABase::RV64];
        }
    }

    let mut isa_bases = if !xlen_constraints.is_empty() {
        let mut isa_bases = Vec::new();
        if xlen_constraints.contains(&32) {
            isa_bases.push(ISABase::RV32);
        }
        if xlen_constraints.contains(&64) {
            isa_bases.push(ISABase::RV64);
        }
        if isa_bases.is_empty() {
            vec![ISABase::RV32, ISABase::RV64]
        } else {
            isa_bases
        }
    } else {
        vec![ISABase::RV32, ISABase::RV64]
    };

    if let Some(supported_bases) = extension_supported_bases.get(extension.as_str()) {
        let filtered = isa_bases
            .iter()
            .copied()
            .filter(|isa_base| supported_bases.contains(isa_base))
            .collect::<Vec<_>>();
        if !filtered.is_empty() {
            isa_bases = filtered;
        }
    }

    isa_bases
}

fn collect_defined_by_xlens(value: &serde_yaml::Value, xlens: &mut HashSet<u8>) {
    match value {
        serde_yaml::Value::Mapping(mapping) => {
            if let Some(xlen_value) = mapping.get(serde_yaml::Value::String("xlen".to_string())) {
                let parsed_xlen = xlen_value
                    .as_u64()
                    .and_then(|value| u8::try_from(value).ok())
                    .or_else(|| {
                        xlen_value
                            .as_i64()
                            .and_then(|value| u8::try_from(value).ok())
                    });

                if let Some(xlen) = parsed_xlen {
                    if matches!(xlen, 32 | 64) {
                        xlens.insert(xlen);
                    }
                }
            }

            for child in mapping.values() {
                collect_defined_by_xlens(child, xlens);
            }
        }
        serde_yaml::Value::Sequence(sequence) => {
            for child in sequence {
                collect_defined_by_xlens(child, xlens);
            }
        }
        _ => {}
    }
}
