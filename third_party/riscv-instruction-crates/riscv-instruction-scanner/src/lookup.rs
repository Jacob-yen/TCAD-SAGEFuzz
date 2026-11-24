use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;

use riscv_instruction_schema::Instruction;

use crate::data::load_instructions;
use crate::scan::{build_mnemonic_index, lookup_with_fallback};

pub struct LookupReport {
    pub queried: usize,
    pub matched: usize,
    pub written_to: Option<std::path::PathBuf>,
}

pub fn run(
    names: Vec<String>,
    out_path: Option<&Path>,
) -> Result<LookupReport, Box<dyn std::error::Error>> {
    let catalog = load_instructions()?;
    let index = build_mnemonic_index(&catalog);
    let by_name: HashMap<&str, &Instruction> =
        catalog.iter().map(|i| (i.name.as_str(), i)).collect();

    let mut matched = 0usize;

    let mut writer: Box<dyn Write> = match out_path {
        Some(p) => Box::new(std::fs::File::create(p)?),
        None => Box::new(std::io::stdout().lock()),
    };
    let mut csv_w = csv::Writer::from_writer(&mut writer);
    csv_w.write_record([
        "input",
        "normalized",
        "found",
        "extension",
        "isa_bases",
        "operands",
        "assembly_syntax",
    ])?;

    for raw in &names {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (canonical, found) = match lookup_with_fallback(&index, trimmed) {
            Some(c) => (c, true),
            None => (trimmed.to_ascii_lowercase().replace('_', "."), false),
        };
        let inst = if found { by_name.get(canonical.as_str()).copied() } else { None };
        if found {
            matched += 1;
        }
        let ext = inst.map(|i| i.extension.to_string()).unwrap_or_default();
        let bases = inst
            .map(|i| {
                i.isa_bases
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(";")
            })
            .unwrap_or_default();
        let operands = inst
            .map(|i| {
                i.operands
                    .iter()
                    .map(|o| o.name.as_str())
                    .collect::<Vec<_>>()
                    .join(";")
            })
            .unwrap_or_default();
        let syntax = inst
            .map(|i| match &i.assembly_syntax {
                riscv_instruction_schema::AssemblySyntax::Format(s) => s.clone(),
                riscv_instruction_schema::AssemblySyntax::RustCode(_) => "<rust>".into(),
            })
            .unwrap_or_default();
        csv_w.write_record([
            trimmed,
            &canonical,
            &found.to_string(),
            &ext,
            &bases,
            &operands,
            &syntax,
        ])?;
    }
    csv_w.flush()?;
    drop(csv_w);

    Ok(LookupReport {
        queried: names.len(),
        matched,
        written_to: out_path.map(|p| p.to_path_buf()),
    })
}

pub fn read_names_from_stdin() -> std::io::Result<Vec<String>> {
    let stdin = std::io::stdin();
    let mut out = Vec::new();
    for line in stdin.lock().lines() {
        let line = line?;
        for tok in line.split(|c: char| c == ',' || c.is_whitespace()) {
            if !tok.is_empty() {
                out.push(tok.to_string());
            }
        }
    }
    Ok(out)
}

pub fn read_names_from_file(path: &Path) -> std::io::Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in content.lines() {
        for tok in line.split(|c: char| c == ',' || c.is_whitespace()) {
            if !tok.is_empty() {
                out.push(tok.to_string());
            }
        }
    }
    Ok(out)
}
