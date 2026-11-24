use std::collections::BTreeSet;
use std::path::Path;

use riscv_instruction_schema::{AssemblySyntax, ISABase, Instruction, OperandKind};

use crate::data::load_instructions;

/// Emit `extensions.csv` and `instructions.csv` into `out_dir` from the
/// embedded RISC-V instruction catalog. Returns the two paths written.
pub fn run(out_dir: &Path) -> Result<(std::path::PathBuf, std::path::PathBuf), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(out_dir)?;
    let instructions = load_instructions()?;

    let ext_path = out_dir.join("extensions.csv");
    write_extensions_csv(&instructions, &ext_path)?;

    let inst_path = out_dir.join("instructions.csv");
    write_instructions_csv(&instructions, &inst_path)?;

    Ok((ext_path, inst_path))
}

fn write_extensions_csv(instructions: &[Instruction], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Agg {
        count: usize,
        rv32: bool,
        rv64: bool,
        required: BTreeSet<String>,
    }

    let mut by_ext: BTreeMap<String, Agg> = BTreeMap::new();
    for inst in instructions {
        let entry = by_ext.entry(inst.extension.to_string()).or_default();
        entry.count += 1;
        for base in &inst.isa_bases {
            match base {
                ISABase::RV32 => entry.rv32 = true,
                ISABase::RV64 => entry.rv64 = true,
            }
        }
        for reqs in inst.required_extensions.values() {
            for r in reqs {
                entry.required.insert(r.to_string());
            }
        }
    }

    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "extension",
        "instruction_count",
        "supports_rv32",
        "supports_rv64",
        "required_extensions",
    ])?;
    for (name, agg) in by_ext {
        w.write_record([
            name,
            agg.count.to_string(),
            agg.rv32.to_string(),
            agg.rv64.to_string(),
            agg.required.into_iter().collect::<Vec<_>>().join(";"),
        ])?;
    }
    w.flush()?;
    Ok(())
}

fn write_instructions_csv(instructions: &[Instruction], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut sorted: Vec<&Instruction> = instructions.iter().collect();
    sorted.sort_by(|a, b| {
        a.extension
            .to_string()
            .cmp(&b.extension.to_string())
            .then_with(|| a.name.cmp(&b.name))
    });

    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "name",
        "extension",
        "isa_bases",
        "required_extensions_rv32",
        "required_extensions_rv64",
        "operand_count",
        "operands",
        "assembly_syntax",
        "has_memory_access",
        "touches_fp",
        "touches_vector",
        "branches",
        "jumps",
        "csr_access",
    ])?;

    for inst in sorted {
        let bases = inst
            .isa_bases
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(";");
        let req = |b: ISABase| {
            inst.required_extensions
                .get(&b)
                .map(|v| v.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(";"))
                .unwrap_or_default()
        };
        let operands = inst
            .operands
            .iter()
            .map(|o| format!("{}:{}", o.name, operand_kind_short(&o.kind)))
            .collect::<Vec<_>>()
            .join(";");
        let syntax = match &inst.assembly_syntax {
            AssemblySyntax::Format(s) => s.clone(),
            AssemblySyntax::RustCode(_) => String::from("<rust>"),
        };
        let csr_access = match &inst.semantics.csr {
            Some(c) => match (c.reads, c.writes) {
                (true, true) => "rw",
                (true, false) => "r",
                (false, true) => "w",
                _ => "",
            },
            None => "",
        }
        .to_string();
        let (branches, jumps) = inst
            .semantics
            .control_flow
            .as_ref()
            .map(|cf| (cf.branches, cf.jumps))
            .unwrap_or((false, false));

        w.write_record([
            inst.name.clone(),
            inst.extension.to_string(),
            bases,
            req(ISABase::RV32),
            req(ISABase::RV64),
            inst.operands.len().to_string(),
            operands,
            syntax,
            inst.memory_access.is_some().to_string(),
            inst.uses_floating_point_registers().to_string(),
            inst.uses_vector_registers().to_string(),
            branches.to_string(),
            jumps.to_string(),
            csr_access,
        ])?;
    }
    w.flush()?;
    Ok(())
}

fn operand_kind_short(kind: &OperandKind) -> &'static str {
    match kind {
        OperandKind::Unknown => "unknown",
        OperandKind::Register(_) => "reg",
        OperandKind::Immediate(_) => "imm",
        OperandKind::Csr(_) => "csr",
        OperandKind::RoundMode => "rm",
        OperandKind::FenceMode => "fence",
        OperandKind::FliConstant => "fli",
        OperandKind::SavedRegListWithStackAdj { .. } => "reglist",
        OperandKind::RegisterPair(_) => "regpair",
    }
}
