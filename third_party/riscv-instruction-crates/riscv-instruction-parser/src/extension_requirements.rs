use riscv_instruction_schema::{ISABase, ISAExtension, YamlInstruction};
use serde::Deserialize;
use serde_yaml::{Mapping, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

/// Transitive extension dependency closure keyed by extension name and ISA base.
pub(crate) type ExtensionRequirementClosure = HashMap<String, BTreeMap<ISABase, BTreeSet<String>>>;
/// ISA bases on which an extension itself can exist, inferred from extension metadata.
pub(crate) type ExtensionSupportedBases = HashMap<String, BTreeSet<ISABase>>;

pub(crate) struct ExtensionMetadataIndex {
    pub requirement_closure: ExtensionRequirementClosure,
    pub supported_bases: ExtensionSupportedBases,
}

#[derive(Debug, Deserialize)]
struct YamlExtension {
    pub name: String,
    #[serde(default)]
    pub requirements: Option<Value>,
    #[serde(default)]
    pub versions: Vec<YamlExtensionVersion>,
}

#[derive(Debug, Default, Deserialize)]
struct YamlExtensionVersion {
    #[serde(default)]
    pub requirements: Option<Value>,
}

pub(crate) fn load_extension_metadata_index(
    unified_db_root: &Path,
) -> Result<ExtensionMetadataIndex, Box<dyn std::error::Error>> {
    let Some(ext_path) = resolve_ext_path(unified_db_root)? else {
        return Ok(ExtensionMetadataIndex {
            requirement_closure: HashMap::new(),
            supported_bases: HashMap::new(),
        });
    };

    let mut direct_requirements: HashMap<String, BTreeMap<ISABase, BTreeSet<String>>> =
        HashMap::new();
    let mut supported_bases = HashMap::new();
    let mut extension_files = fs::read_dir(&ext_path)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("yaml"))
        .collect::<Vec<_>>();
    extension_files.sort();

    for file_path in extension_files {
        let content = fs::read_to_string(&file_path)?;
        let yaml_ext: YamlExtension = serde_yaml::from_str(&content)?;
        let extension_name = yaml_ext.name.clone();
        let mut per_base = BTreeMap::new();

        for isa_base in [ISABase::RV32, ISABase::RV64] {
            let mut deps = BTreeSet::new();

            if let Some(requirements) = yaml_ext.requirements.as_ref() {
                deps.extend(mandatory_extensions_for_base(requirements, isa_base));
            }

            for version in &yaml_ext.versions {
                if let Some(requirements) = version.requirements.as_ref() {
                    deps.extend(mandatory_extensions_for_base(requirements, isa_base));
                }
            }

            deps.remove(&extension_name);
            if !deps.is_empty() {
                per_base.insert(isa_base, deps);
            }
        }

        direct_requirements.insert(extension_name.clone(), per_base);
        supported_bases.insert(extension_name, determine_supported_bases(&yaml_ext));
    }

    Ok(ExtensionMetadataIndex {
        requirement_closure: compute_requirement_closure(&direct_requirements),
        supported_bases,
    })
}

fn determine_supported_bases(yaml_ext: &YamlExtension) -> BTreeSet<ISABase> {
    [ISABase::RV32, ISABase::RV64]
        .into_iter()
        .filter(|&isa_base| extension_supports_base(yaml_ext, isa_base))
        .collect()
}

fn extension_supports_base(yaml_ext: &YamlExtension, isa_base: ISABase) -> bool {
    let top_level_allows = yaml_ext
        .requirements
        .as_ref()
        .is_none_or(|requirements| base_condition_truth(requirements, isa_base) != Some(false));

    if !top_level_allows {
        return false;
    }

    if yaml_ext.versions.is_empty() {
        return true;
    }

    yaml_ext.versions.iter().any(|version| {
        version
            .requirements
            .as_ref()
            .is_none_or(|requirements| base_condition_truth(requirements, isa_base) != Some(false))
    })
}

pub(crate) fn resolve_instruction_required_extensions(
    yaml_inst: &YamlInstruction,
    instruction_extension: &ISAExtension,
    isa_bases: &[ISABase],
    extension_requirement_closure: &ExtensionRequirementClosure,
) -> BTreeMap<ISABase, Vec<ISAExtension>> {
    let mut required_extensions = BTreeMap::new();

    for &isa_base in isa_bases {
        let mut dependencies = extension_requirement_closure
            .get(instruction_extension.as_str())
            .and_then(|per_base| per_base.get(&isa_base))
            .cloned()
            .unwrap_or_default();

        let coverage = extension_coverage_for_base(
            instruction_extension,
            isa_base,
            extension_requirement_closure,
        );
        let instruction_companions = select_instruction_companion_extensions(
            positive_extension_branches_for_base(&yaml_inst.defined_by, isa_base),
            &coverage,
        );
        for companion in instruction_companions {
            dependencies.insert(companion.clone());
            if let Some(companion_deps) = extension_requirement_closure
                .get(&companion)
                .and_then(|per_base| per_base.get(&isa_base))
            {
                dependencies.extend(companion_deps.iter().cloned());
            }
        }
        dependencies.remove(instruction_extension.as_str());

        if !dependencies.is_empty() {
            required_extensions.insert(
                isa_base,
                dependencies.into_iter().map(ISAExtension::new).collect(),
            );
        }
    }

    required_extensions
}

fn resolve_ext_path(root: &Path) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let candidates = [root.join("arch/ext"), root.join("spec/std/isa/ext")];

    for candidate in candidates {
        if looks_like_ext_root(&candidate)? {
            return Ok(Some(candidate));
        }
    }

    Ok(None)
}

fn looks_like_ext_root(path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    if !path.is_dir() {
        return Ok(false);
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if child.extension().and_then(|value| value.to_str()) == Some("yaml") {
            return Ok(true);
        }
    }

    Ok(false)
}

fn compute_requirement_closure(
    direct_requirements: &HashMap<String, BTreeMap<ISABase, BTreeSet<String>>>,
) -> ExtensionRequirementClosure {
    let mut memo = HashMap::new();
    let mut closure = HashMap::new();

    for extension in direct_requirements.keys() {
        let mut per_base = BTreeMap::new();

        for isa_base in [ISABase::RV32, ISABase::RV64] {
            let deps = collect_transitive_dependencies(
                extension,
                isa_base,
                direct_requirements,
                &mut memo,
                &mut HashSet::new(),
            );

            if !deps.is_empty() {
                per_base.insert(isa_base, deps);
            }
        }

        closure.insert(extension.clone(), per_base);
    }

    closure
}

fn collect_transitive_dependencies(
    extension: &str,
    isa_base: ISABase,
    direct_requirements: &HashMap<String, BTreeMap<ISABase, BTreeSet<String>>>,
    memo: &mut HashMap<(String, ISABase), BTreeSet<String>>,
    visiting: &mut HashSet<(String, ISABase)>,
) -> BTreeSet<String> {
    let key = (extension.to_string(), isa_base);
    if let Some(cached) = memo.get(&key) {
        return cached.clone();
    }
    if !visiting.insert(key.clone()) {
        return BTreeSet::new();
    }

    let direct = direct_requirements
        .get(extension)
        .and_then(|per_base| per_base.get(&isa_base))
        .cloned()
        .unwrap_or_default();

    let mut all = direct.clone();
    for dep in direct {
        all.extend(collect_transitive_dependencies(
            &dep,
            isa_base,
            direct_requirements,
            memo,
            visiting,
        ));
    }

    visiting.remove(&key);
    memo.insert(key, all.clone());
    all
}

fn extension_coverage_for_base(
    instruction_extension: &ISAExtension,
    isa_base: ISABase,
    extension_requirement_closure: &ExtensionRequirementClosure,
) -> BTreeSet<String> {
    let mut coverage = BTreeSet::from([instruction_extension.to_string()]);
    if let Some(deps) = extension_requirement_closure
        .get(instruction_extension.as_str())
        .and_then(|per_base| per_base.get(&isa_base))
    {
        coverage.extend(deps.iter().cloned());
    }
    coverage
}

fn select_instruction_companion_extensions(
    branches: Vec<BTreeSet<String>>,
    coverage: &BTreeSet<String>,
) -> BTreeSet<String> {
    if branches.is_empty() {
        return BTreeSet::new();
    }

    let relevant = branches
        .iter()
        .filter(|branch| !branch.is_disjoint(coverage))
        .cloned()
        .collect::<Vec<_>>();

    let selected = if !relevant.is_empty() {
        relevant
    } else if branches.len() == 1 {
        branches
    } else {
        Vec::new()
    };

    intersect_branch_extensions(selected)
        .difference(coverage)
        .cloned()
        .collect()
}

fn mandatory_extensions_for_base(value: &Value, isa_base: ISABase) -> BTreeSet<String> {
    intersect_branch_extensions(positive_extension_branches_for_base(value, isa_base))
}

fn positive_extension_branches_for_base(value: &Value, isa_base: ISABase) -> Vec<BTreeSet<String>> {
    match value {
        Value::Mapping(mapping) => positive_extension_branches_from_mapping(mapping, isa_base),
        Value::Sequence(sequence) => combine_all_branches(
            sequence
                .iter()
                .map(|child| positive_extension_branches_for_base(child, isa_base)),
        ),
        _ => vec![BTreeSet::new()],
    }
}

fn positive_extension_branches_from_mapping(
    mapping: &Mapping,
    isa_base: ISABase,
) -> Vec<BTreeSet<String>> {
    if let Some(children) = mapping_sequence(mapping, "allOf") {
        return combine_all_branches(
            children
                .iter()
                .map(|child| positive_extension_branches_for_base(child, isa_base)),
        );
    }

    if let Some(children) = mapping_sequence(mapping, "anyOf") {
        return combine_any_branches(
            children
                .iter()
                .map(|child| positive_extension_branches_for_base(child, isa_base)),
        );
    }

    if let Some(children) = mapping_sequence(mapping, "oneOf") {
        return combine_any_branches(
            children
                .iter()
                .map(|child| positive_extension_branches_for_base(child, isa_base)),
        );
    }

    if let Some(not_value) = mapping_value(mapping, "not") {
        return negative_condition_branches(not_value, isa_base);
    }

    if let Some(extension_value) = mapping_value(mapping, "extension") {
        return positive_extension_branches_for_base(extension_value, isa_base);
    }

    if let Some(name) = extension_name_from_mapping(mapping) {
        return vec![BTreeSet::from([name])];
    }

    if let Some(xlen_value) = mapping_value(mapping, "xlen") {
        return if xlen_matches_base(xlen_value, isa_base) {
            vec![BTreeSet::new()]
        } else {
            Vec::new()
        };
    }

    if matches!(
        base_condition_truth(&Value::Mapping(mapping.clone()), isa_base),
        Some(false)
    ) {
        return Vec::new();
    }

    combine_all_branches(
        mapping
            .values()
            .map(|child| positive_extension_branches_for_base(child, isa_base)),
    )
}

fn negative_condition_branches(value: &Value, isa_base: ISABase) -> Vec<BTreeSet<String>> {
    match base_condition_truth(value, isa_base) {
        Some(true) => Vec::new(),
        _ => vec![BTreeSet::new()],
    }
}

fn combine_all_branches<I>(children: I) -> Vec<BTreeSet<String>>
where
    I: IntoIterator<Item = Vec<BTreeSet<String>>>,
{
    let mut combined = vec![BTreeSet::new()];

    for child in children {
        if child.is_empty() {
            return Vec::new();
        }

        let mut next = Vec::new();
        for prefix in &combined {
            for suffix in &child {
                let mut merged = prefix.clone();
                merged.extend(suffix.iter().cloned());
                next.push(merged);
            }
        }
        combined = next;
    }

    combined
}

fn combine_any_branches<I>(children: I) -> Vec<BTreeSet<String>>
where
    I: IntoIterator<Item = Vec<BTreeSet<String>>>,
{
    children
        .into_iter()
        .flat_map(|branches| branches.into_iter())
        .collect()
}

fn intersect_branch_extensions(branches: Vec<BTreeSet<String>>) -> BTreeSet<String> {
    let mut iter = branches.into_iter();
    let Some(first) = iter.next() else {
        return BTreeSet::new();
    };

    iter.fold(first, |current, branch| {
        current.intersection(&branch).cloned().collect()
    })
}

fn base_condition_truth(value: &Value, isa_base: ISABase) -> Option<bool> {
    match value {
        Value::Mapping(mapping) => {
            if let Some(children) = mapping_sequence(mapping, "allOf") {
                return truth_all_of(children, isa_base);
            }

            if let Some(children) = mapping_sequence(mapping, "anyOf") {
                return truth_any_of(children, isa_base);
            }

            if let Some(children) = mapping_sequence(mapping, "oneOf") {
                return truth_one_of(children, isa_base);
            }

            if let Some(not_value) = mapping_value(mapping, "not") {
                return base_condition_truth(not_value, isa_base).map(|truth| !truth);
            }

            if let Some(xlen_value) = mapping_value(mapping, "xlen") {
                return Some(xlen_matches_base(xlen_value, isa_base));
            }

            if let Some(extension_value) = mapping_value(mapping, "extension") {
                return base_condition_truth(extension_value, isa_base);
            }

            if extension_name_from_mapping(mapping).is_some() {
                return None;
            }

            truth_all_of(mapping.values(), isa_base)
        }
        Value::Sequence(sequence) => truth_all_of(sequence.iter(), isa_base),
        _ => None,
    }
}

fn truth_all_of<'a, I>(values: I, isa_base: ISABase) -> Option<bool>
where
    I: IntoIterator<Item = &'a Value>,
{
    let mut saw_unknown = false;

    for value in values {
        match base_condition_truth(value, isa_base) {
            Some(true) => {}
            Some(false) => return Some(false),
            None => saw_unknown = true,
        }
    }

    if saw_unknown { None } else { Some(true) }
}

fn truth_any_of<'a, I>(values: I, isa_base: ISABase) -> Option<bool>
where
    I: IntoIterator<Item = &'a Value>,
{
    let mut saw_unknown = false;

    for value in values {
        match base_condition_truth(value, isa_base) {
            Some(true) => return Some(true),
            Some(false) => {}
            None => saw_unknown = true,
        }
    }

    if saw_unknown { None } else { Some(false) }
}

fn truth_one_of<'a, I>(values: I, isa_base: ISABase) -> Option<bool>
where
    I: IntoIterator<Item = &'a Value>,
{
    let mut truths = 0usize;
    let mut saw_unknown = false;

    for value in values {
        match base_condition_truth(value, isa_base) {
            Some(true) => truths += 1,
            Some(false) => {}
            None => saw_unknown = true,
        }
    }

    if saw_unknown {
        if truths > 1 { Some(false) } else { None }
    } else {
        Some(truths == 1)
    }
}

fn xlen_matches_base(value: &Value, isa_base: ISABase) -> bool {
    let expected = match isa_base {
        ISABase::RV32 => 32,
        ISABase::RV64 => 64,
    };

    value
        .as_u64()
        .and_then(|number| u8::try_from(number).ok())
        .or_else(|| value.as_i64().and_then(|number| u8::try_from(number).ok()))
        == Some(expected)
}

fn mapping_value<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(&Value::String(key.to_string()))
}

fn mapping_sequence<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Vec<Value>> {
    mapping_value(mapping, key).and_then(Value::as_sequence)
}

fn extension_name_from_mapping(mapping: &Mapping) -> Option<String> {
    let mut name = None;

    for key in mapping.keys() {
        let key = key.as_str()?;
        match key {
            "name" => {
                name = mapping_value(mapping, "name")
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
            "version" => {}
            _ => return None,
        }
    }

    name
}

#[cfg(test)]
mod tests {
    use super::{
        ExtensionRequirementClosure, determine_supported_bases, mandatory_extensions_for_base,
        resolve_instruction_required_extensions,
    };
    use riscv_instruction_schema::{ISABase, ISAExtension, YamlInstruction};
    use serde_yaml::Value;
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    fn yaml_instruction(defined_by: &str) -> YamlInstruction {
        YamlInstruction {
            schema: "inst_schema.json#".to_string(),
            kind: "instruction".to_string(),
            name: "demo".to_string(),
            long_name: "Demo".to_string(),
            description: riscv_instruction_schema::Description::String("demo".to_string()),
            defined_by: serde_yaml::from_str(defined_by).expect("valid yaml"),
            base: None,
            assembly: "demo".to_string(),
            format: None,
            encoding: None,
            access: None,
            data_independent_timing: None,
            operation: None,
            sail: None,
        }
    }

    fn yaml_extension(yaml: &str) -> super::YamlExtension {
        serde_yaml::from_str(yaml).expect("valid yaml")
    }

    #[test]
    fn supported_bases_respect_extension_level_xlen_constraints() {
        let yaml_ext = yaml_extension(
            r#"
name: Zcf
requirements:
  allOf:
    - extension:
        name: F
    - xlen: 32
"#,
        );

        assert_eq!(
            determine_supported_bases(&yaml_ext),
            BTreeSet::from([ISABase::RV32])
        );
    }

    #[test]
    fn mandatory_extension_extraction_intersects_alternatives() {
        let requirements: Value = serde_yaml::from_str(
            r#"
allOf:
  - extension:
      name: Zca
  - anyOf:
      - not:
          extension:
            name: F
      - xlen: 64
      - extension:
          name: Zcf
  - oneOf:
      - not:
          extension:
            name: D
      - extension:
          name: Zcd
"#,
        )
        .expect("valid yaml");

        assert_eq!(
            mandatory_extensions_for_base(&requirements, ISABase::RV32),
            BTreeSet::from(["Zca".to_string()])
        );
        assert_eq!(
            mandatory_extensions_for_base(&requirements, ISABase::RV64),
            BTreeSet::from(["Zca".to_string()])
        );
    }

    #[test]
    fn instruction_companions_respect_primary_coverage_and_xlen_split() {
        let yaml_inst = yaml_instruction(
            r#"
anyOf:
  - allOf:
      - xlen: 64
      - extension:
          name: Zca
  - extension:
      name: Zclsd
"#,
        );

        let mut closure: ExtensionRequirementClosure = HashMap::new();
        closure.insert(
            "C".to_string(),
            BTreeMap::from([
                (ISABase::RV32, BTreeSet::from(["Zca".to_string()])),
                (ISABase::RV64, BTreeSet::from(["Zca".to_string()])),
            ]),
        );

        let required = resolve_instruction_required_extensions(
            &yaml_inst,
            &ISAExtension::new("C"),
            &[ISABase::RV32, ISABase::RV64],
            &closure,
        );

        assert_eq!(
            required.get(&ISABase::RV32).cloned(),
            Some(vec![ISAExtension::new("Zca"), ISAExtension::new("Zclsd")])
        );
        assert_eq!(
            required.get(&ISABase::RV64).cloned(),
            Some(vec![ISAExtension::new("Zca")])
        );
    }

    #[test]
    fn umbrella_extensions_prefer_branches_they_already_cover() {
        let yaml_inst = yaml_instruction(
            r#"
extension:
  anyOf:
    - name: Zbb
    - name: Zbkb
"#,
        );

        let mut closure: ExtensionRequirementClosure = HashMap::new();
        closure.insert(
            "B".to_string(),
            BTreeMap::from([(
                ISABase::RV64,
                BTreeSet::from(["Zba".to_string(), "Zbb".to_string(), "Zbs".to_string()]),
            )]),
        );

        let required = resolve_instruction_required_extensions(
            &yaml_inst,
            &ISAExtension::new("B"),
            &[ISABase::RV64],
            &closure,
        );

        assert_eq!(
            required.get(&ISABase::RV64).cloned(),
            Some(vec![
                ISAExtension::new("Zba"),
                ISAExtension::new("Zbb"),
                ISAExtension::new("Zbs"),
            ])
        );
    }
}
