use std::path::PathBuf;
use std::process::ExitCode;

use riscv_instruction_scanner::{catalog, lookup, scan, verdict};

const USAGE: &str = "\
riscv-scan — RISC-V extension/instruction CSV exporter, project scanner, and mnemonic lookup

USAGE:
    riscv-scan catalog --out <DIR>
        Export the full RISC-V instruction catalog (extensions.csv,
        instructions.csv) embedded in this binary.

    riscv-scan scan --target <DIR> --out <DIR> [--include-hdl]
                    [--exclude <GLOB> ...]
        Walk a project directory and emit candidate CSVs:
            instructions_found.csv    one row per matched instruction (asm + hdl
                                      occurrence counts, per-class columns)
            extensions_found.csv      one row per extension with hits
            march_flags.csv           every -march=... string seen (g expanded)
            path_class_summary.csv    file / hit counts by path class
                                      (Core/Tests/Vendor/Examples/Docs/Software/Other)

        Default sources: .s/.S/.asm files, Rust asm!/global_asm!/naked_asm!
        macros, C/C++ __asm__ blocks. With --include-hdl also walks
        .scala/.v/.sv/.svh/.vhd/.vhdl/.chisel files looking for:
          1. `MNEMONIC ->`, `MNEMONIC =>` (rocket-chip / BOOM Chisel decode tables)
          2. \\`define INST_MNEMONIC` and `localparam MNEMONIC ...` (Verilog/SV)
          3. `pkg::OPNAME` and `PREFIX_OPNAME` (CVA6 ariane_pkg::ADD / ibex ALU_CLZ)
        `_` in the matched name is auto-converted to `.` (so `AMOMAXU_W` matches
        `amomaxu.w`). The third pattern covers cores that use an internal op-enum
        rather than raw mnemonic string literals in decode tables.

        Path classification: each file is labelled Core / Tests / Vendor /
        Examples / Docs / Software / Other based on directory name heuristics.
        A WARNING is printed when Tests+Vendor hits outnumber Core hits —
        use --exclude to filter noise before interpreting results.

        --exclude <GLOB>  Skip files/dirs matching this glob pattern (relative
                          to target root). May be repeated. Examples:
                            --exclude 'vendor/**'
                            --exclude 'riscv-tests/**'
                            --exclude 'cv32e40x/**'

    riscv-scan lookup --names <a,b,c> [--out <FILE>]
    riscv-scan lookup --file <FILE> [--out <FILE>]
    riscv-scan lookup --stdin [--out <FILE>]
        Resolve instruction mnemonics against the catalog. Accepts both the
        canonical dotted form (`fcvt.lu.s`) and underscore form (`FCVT_LU_S`).
        Emits CSV: input,normalized,found,extension,isa_bases,operands,
        assembly_syntax. Writes to stdout if --out is omitted.

    riscv-scan verdict --scan <DIR> [--out <FILE>] [--set <SPEC> ...]
        Synthesize a supported.csv verdict from scan output.

        --scan <DIR>   Directory written by 'riscv-scan scan --out' (must contain
                       extensions_found.csv and march_flags.csv).
        --out <FILE>   Output path. Default: <scan-dir>/supported.csv.
        --set <SPEC>   Override or add a row. Format: EXTENSION=STATUS or
                       EXTENSION=STATUS:NOTES. STATUS in: used, declared,
                       intrinsics-only, unused. May be repeated.
                       Examples:
                         --set 'B=used:parameter-gated via RV32B'
                         --set 'F=declared:only in test binary, not core RTL'

        If supported.csv already exists, scan-derived fields (instructions,
        files, evidence_kind) are refreshed, but manually set status and notes
        are preserved.

    riscv-scan --help
";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let cmd = match args.next() {
        Some(c) => c,
        None => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let rest: Vec<String> = args.collect();

    match cmd.as_str() {
        "--help" | "-h" | "help" => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        "catalog" => match parse_args(&rest, &["--out"], &[], &[]) {
            Ok((map, _, _)) => {
                let out = PathBuf::from(map.get("--out").unwrap());
                match catalog::run(&out) {
                    Ok((ext, inst)) => {
                        println!("wrote {}", ext.display());
                        println!("wrote {}", inst.display());
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("catalog failed: {e}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(e) => {
                eprintln!("{e}\n{USAGE}");
                ExitCode::from(2)
            }
        },
        "scan" => match parse_args(&rest, &["--target", "--out"], &["--include-hdl"], &["--exclude"]) {
            Ok((map, flags, multi)) => {
                let target = PathBuf::from(map.get("--target").unwrap());
                let out = PathBuf::from(map.get("--out").unwrap());
                let exclude = multi.get("--exclude").cloned().unwrap_or_default();
                let opts = scan::ScanOptions {
                    include_hdl: flags.contains(&"--include-hdl".to_string()),
                    exclude,
                };
                match scan::run(&target, &out, opts) {
                    Ok(report) => {
                        println!(
                            "scanned {} files ({} HDL), {} instruction occurrences",
                            report.files_scanned,
                            report.hdl_files_scanned,
                            report.instruction_hits
                        );
                        println!("wrote {}", report.instructions_csv.display());
                        println!("wrote {}", report.extensions_csv.display());
                        println!("wrote {}", report.march_csv.display());
                        println!("wrote {}", report.path_class_summary_csv.display());
                        println!("wrote {}", report.evidence_jsonl.display());
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("scan failed: {e}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(e) => {
                eprintln!("{e}\n{USAGE}");
                ExitCode::from(2)
            }
        },
        "lookup" => match parse_args(&rest, &[], &["--stdin"], &[]) {
            Ok((map, flags, _)) => {
                let names_res: Result<Vec<String>, Box<dyn std::error::Error>> = if let Some(list) =
                    map.get("--names")
                {
                    Ok(list
                        .split(|c: char| c == ',' || c.is_whitespace())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .collect())
                } else if let Some(path) = map.get("--file") {
                    lookup::read_names_from_file(std::path::Path::new(path))
                        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })
                } else if flags.contains(&"--stdin".to_string()) {
                    lookup::read_names_from_stdin()
                        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })
                } else {
                    Err("lookup needs one of --names, --file, --stdin".into())
                };
                let names = match names_res {
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("{e}\n{USAGE}");
                        return ExitCode::from(2);
                    }
                };
                let out_path = map.get("--out").map(PathBuf::from);
                match lookup::run(names, out_path.as_deref()) {
                    Ok(rep) => {
                        eprintln!("looked up {} mnemonics, {} matched", rep.queried, rep.matched);
                        if let Some(p) = rep.written_to {
                            eprintln!("wrote {}", p.display());
                        }
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("lookup failed: {e}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(e) => {
                eprintln!("{e}\n{USAGE}");
                ExitCode::from(2)
            }
        },
        "verdict" => match parse_args(&rest, &["--scan"], &[], &["--set"]) {
            Ok((map, _, multi)) => {
                let scan_dir = PathBuf::from(map.get("--scan").unwrap());
                let out_file = map.get("--out").map(PathBuf::from);
                let set_specs = multi.get("--set").cloned().unwrap_or_default();
                let opts = verdict::VerdictOptions { scan_dir, out_file, set_specs };
                match verdict::run(opts) {
                    Ok(rep) => {
                        println!(
                            "wrote {} ({} extensions: {} used, {} declared, {} other)",
                            rep.written_to.display(),
                            rep.total,
                            rep.used,
                            rep.declared,
                            rep.other,
                        );
                        ExitCode::SUCCESS
                    }
                    Err(e) => {
                        eprintln!("verdict failed: {e}");
                        ExitCode::FAILURE
                    }
                }
            }
            Err(e) => {
                eprintln!("{e}\n{USAGE}");
                ExitCode::from(2)
            }
        },
        other => {
            eprintln!("unknown subcommand: {other}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Parse CLI args supporting:
///   `--key value`    for keys in `value_keys` (last value wins)
///   `--flag`         for bare boolean switches in `bool_flags`
///   `--key value`    for keys in `multi_keys`  (all values collected)
///
/// Returns `(single_values, set_flags, multi_values)`.
/// Errors if any key from `required` (subset of value_keys) is missing.
fn parse_args(
    args: &[String],
    required: &[&str],
    bool_flags: &[&str],
    multi_keys: &[&str],
) -> Result<
    (
        std::collections::HashMap<String, String>,
        Vec<String>,
        std::collections::HashMap<String, Vec<String>>,
    ),
    String,
> {
    let mut values: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut flags: Vec<String> = Vec::new();
    let mut multi: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        if !key.starts_with("--") {
            return Err(format!("expected --flag, got `{key}`"));
        }
        if bool_flags.contains(&key.as_str()) {
            flags.push(key.clone());
            i += 1;
            continue;
        }
        if multi_keys.contains(&key.as_str()) {
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("flag `{key}` is missing a value"))?;
            multi.entry(key.clone()).or_default().push(value.clone());
            i += 2;
            continue;
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("flag `{key}` is missing a value"))?;
        values.insert(key.clone(), value.clone());
        i += 2;
    }
    for r in required {
        if !values.contains_key(*r) {
            return Err(format!("missing required flag {r}"));
        }
    }
    Ok((values, flags, multi))
}
