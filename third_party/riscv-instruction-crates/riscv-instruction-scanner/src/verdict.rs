use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

use crate::data::load_instructions;
use crate::scan::build_ext_index;

pub struct VerdictOptions {
    pub scan_dir: PathBuf,
    pub out_file: Option<PathBuf>,
    pub set_specs: Vec<String>,
}

pub struct VerdictReport {
    pub written_to: PathBuf,
    pub total: usize,
    pub used: usize,
    pub declared: usize,
    pub other: usize,
}

#[derive(Debug, Clone)]
struct Row {
    extension: String,
    support_status: String,
    isa_bases: String,
    evidence_kind: String,
    distinct_instructions: String,
    sample_files: String,
    notes: String,
    #[allow(dead_code)]
    is_manual: bool,
}

#[derive(Debug, Clone)]
struct ScanRow {
    extension: String,
    isa_bases: String,
    evidence_kind: String,
    distinct_instructions: String,
    sample_files: String,
}

fn parse_set_spec(s: &str) -> Result<(String, String, String), String> {
    let (ext, rest) = s
        .split_once('=')
        .ok_or_else(|| format!("invalid --set spec `{s}`: expected EXTENSION=STATUS or EXTENSION=STATUS:NOTES"))?;
    let ext = ext.trim().to_string();
    if ext.is_empty() {
        return Err(format!("invalid --set spec `{s}`: extension name is empty"));
    }
    let (status, notes) = match rest.split_once(':') {
        Some((st, n)) => (st.trim().to_string(), n.to_string()),
        None => (rest.trim().to_string(), String::new()),
    };
    const VALID: &[&str] = &["used", "declared", "intrinsics-only", "unused"];
    if !VALID.contains(&status.as_str()) {
        return Err(format!(
            "invalid status `{status}` in --set spec `{s}`: must be one of {}",
            VALID.join(", ")
        ));
    }
    Ok((ext, status, notes))
}

/// Strip surrounding double-quotes and unescape `""` → `"` (minimal CSV unquoting).
fn unquote_field(s: &str) -> String {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s[1..s.len() - 1].replace("\"\"", "\"")
    } else {
        s.to_string()
    }
}

fn read_csv_rows(path: &Path) -> io::Result<Vec<HashMap<String, String>>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();
    let header_line = match lines.next() {
        Some(Ok(h)) => h,
        _ => return Ok(Vec::new()),
    };
    let headers: Vec<String> = header_line.split(',').map(|s| s.trim().to_string()).collect();
    let mut rows = Vec::new();
    for line in lines.flatten() {
        if line.trim().is_empty() {
            continue;
        }
        // Simple CSV split — fields containing commas are not quoted in our format.
        // notes field may contain commas; we split on first N-1 commas only.
        let parts = split_csv_row(&line, headers.len());
        let mut map = HashMap::new();
        for (i, h) in headers.iter().enumerate() {
            let raw = parts.get(i).cloned().unwrap_or_default();
            map.insert(h.clone(), unquote_field(&raw));
        }
        rows.push(map);
    }
    Ok(rows)
}

/// Split a CSV row into exactly `n_cols` fields. The last field captures everything remaining
/// (handles commas in `notes`).
fn split_csv_row(line: &str, n_cols: usize) -> Vec<String> {
    if n_cols == 0 {
        return Vec::new();
    }
    let mut parts: Vec<String> = Vec::with_capacity(n_cols);
    let mut remaining = line;
    for i in 0..n_cols {
        if i == n_cols - 1 {
            parts.push(remaining.to_string());
        } else if let Some(pos) = remaining.find(',') {
            parts.push(remaining[..pos].to_string());
            remaining = &remaining[pos + 1..];
        } else {
            parts.push(remaining.to_string());
            remaining = "";
        }
    }
    parts
}

fn read_extensions_found(scan_dir: &Path) -> Result<Vec<ScanRow>, String> {
    let path = scan_dir.join("extensions_found.csv");
    if !path.exists() {
        return Err(format!(
            "scan dir does not contain extensions_found.csv; run 'riscv-scan scan' first (looked in {})",
            scan_dir.display()
        ));
    }
    let rows = read_csv_rows(&path).map_err(|e| format!("failed to read extensions_found.csv: {e}"))?;
    let mut result = Vec::new();
    for row in rows {
        let ext = row.get("extension").cloned().unwrap_or_default();
        if ext.is_empty() {
            continue;
        }
        let evidence = row.get("evidence_kind").cloned().unwrap_or_default();
        let instrs = row.get("instructions").cloned().unwrap_or_default();
        let bases = row.get("isa_bases").cloned().unwrap_or_default();
        let files = row.get("sample_files").cloned().unwrap_or_default();
        // sample_files: take at most 5 (semicolon-separated)
        let files_limited = files
            .split(';')
            .filter(|s| !s.is_empty())
            .take(5)
            .collect::<Vec<_>>()
            .join(";");
        result.push(ScanRow {
            extension: ext,
            isa_bases: bases,
            evidence_kind: evidence,
            distinct_instructions: instrs,
            sample_files: files_limited,
        });
    }
    Ok(result)
}

fn read_march_declared(
    scan_dir: &Path,
    used_set: &HashSet<String>,
    ext_index: &HashMap<String, String>,
) -> Result<Vec<String>, String> {
    let path = scan_dir.join("march_flags.csv");
    if !path.exists() {
        // march_flags.csv is optional
        return Ok(Vec::new());
    }
    let rows = read_csv_rows(&path).map_err(|e| format!("failed to read march_flags.csv: {e}"))?;
    let mut declared = Vec::new();
    for row in rows {
        // march_flags.csv columns: march_string, base, declared_extensions, sample_files
        let expanded = row.get("declared_extensions").cloned().unwrap_or_default();
        for raw_ext in expanded.split(';') {
            let raw = raw_ext.trim();
            if raw.is_empty() {
                continue;
            }
            // Canonicalize: "c" → "C", "zba" → "Zba", already-canonical "Zba" → "Zba"
            let ext = ext_index
                .get(&raw.to_ascii_lowercase())
                .cloned()
                .unwrap_or_else(|| raw.to_string());
            if used_set.contains(&ext) {
                continue;
            }
            if !declared.contains(&ext) {
                declared.push(ext);
            }
        }
    }
    Ok(declared)
}

fn read_existing_supported(
    path: &Path,
    ext_index: &HashMap<String, String>,
) -> Result<HashMap<String, Row>, String> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let rows =
        read_csv_rows(path).map_err(|e| format!("failed to read existing supported.csv: {e}"))?;
    let mut map = HashMap::new();
    for row in rows {
        let raw_ext = row.get("extension").cloned().unwrap_or_default();
        if raw_ext.is_empty() {
            continue;
        }
        // Canonicalize extension name (handles stale lowercase entries like "c" → "C")
        let ext = ext_index
            .get(&raw_ext.to_ascii_lowercase())
            .cloned()
            .unwrap_or_else(|| raw_ext.clone());
        let r = Row {
            extension: ext.clone(),
            support_status: row.get("support_status").cloned().unwrap_or_default(),
            isa_bases: row.get("isa_bases").cloned().unwrap_or_default(),
            evidence_kind: row.get("evidence_kind").cloned().unwrap_or_default(),
            distinct_instructions: row.get("distinct_instructions").cloned().unwrap_or_default(),
            sample_files: row.get("sample_files").cloned().unwrap_or_default(),
            notes: row.get("notes").cloned().unwrap_or_default(),
            is_manual: false,
        };
        map.insert(ext, r);
    }
    Ok(map)
}

fn write_supported(path: &Path, rows: &[Row]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create output directory: {e}"))?;
        }
    }
    let mut out = String::new();
    out.push_str("extension,support_status,isa_bases,evidence_kind,distinct_instructions,sample_files,notes\n");
    for row in rows {
        // Escape quotes in notes (simple: replace " with "")
        let notes_escaped = row.notes.replace('"', "\"\"");
        // Only quote notes if it contains comma or quote
        let notes_field = if notes_escaped.contains(',') || notes_escaped.contains('"') {
            format!("\"{}\"", notes_escaped)
        } else {
            notes_escaped.clone()
        };
        out.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            row.extension,
            row.support_status,
            row.isa_bases,
            row.evidence_kind,
            row.distinct_instructions,
            row.sample_files,
            notes_field,
        ));
    }
    fs::write(path, out).map_err(|e| format!("failed to write {}: {e}", path.display()))
}

pub fn run(opts: VerdictOptions) -> Result<VerdictReport, String> {
    // Build extension canonicalization index from catalog
    let catalog =
        load_instructions().map_err(|e| format!("failed to load instruction catalog: {e}"))?;
    let ext_index = build_ext_index(&catalog);

    // Step 1: read extensions_found.csv
    let scan_rows = read_extensions_found(&opts.scan_dir)?;
    let mut used_map: HashMap<String, ScanRow> = HashMap::new();
    let used_set: HashSet<String> = scan_rows.iter().map(|r| r.extension.clone()).collect();
    for r in scan_rows {
        used_map.insert(r.extension.clone(), r);
    }

    // Step 2: read march_flags.csv for declared-only extensions
    let declared_exts = read_march_declared(&opts.scan_dir, &used_set, &ext_index)?;

    // Determine output path
    let out_path = opts
        .out_file
        .clone()
        .unwrap_or_else(|| opts.scan_dir.join("supported.csv"));

    // Step 3: load existing supported.csv
    let existing = read_existing_supported(&out_path, &ext_index)?;

    // Step 4: parse --set overrides
    let mut overrides: HashMap<String, (String, String)> = HashMap::new(); // ext -> (status, notes)
    for spec in &opts.set_specs {
        let (ext, status, notes) = parse_set_spec(spec)?;
        overrides.insert(ext, (status, notes));
    }

    // Step 5: build union of all extension names
    let mut all_exts: Vec<String> = Vec::new();
    for ext in used_map.keys() {
        if !all_exts.contains(ext) {
            all_exts.push(ext.clone());
        }
    }
    for ext in &declared_exts {
        if !all_exts.contains(ext) {
            all_exts.push(ext.clone());
        }
    }
    for ext in existing.keys() {
        if !all_exts.contains(ext) {
            all_exts.push(ext.clone());
        }
    }
    for ext in overrides.keys() {
        if !all_exts.contains(ext) {
            all_exts.push(ext.clone());
        }
    }
    all_exts.sort();

    // Step 6: merge
    let mut output_rows: Vec<Row> = Vec::new();
    for ext in &all_exts {
        // Determine scan-derived status and data
        let (scan_status, scan_bases, scan_evidence, scan_instrs, scan_files) =
            if let Some(sr) = used_map.get(ext) {
                ("used", sr.isa_bases.clone(), sr.evidence_kind.clone(), sr.distinct_instructions.clone(), sr.sample_files.clone())
            } else if declared_exts.contains(ext) {
                ("declared", String::new(), "march".to_string(), String::new(), String::new())
            } else {
                ("", String::new(), String::new(), String::new(), String::new())
            };

        let existing_row = existing.get(ext);

        // Determine support_status:
        // priority: override > existing-manual > scan-derived
        let (support_status, is_manual) = if let Some((ov_status, _)) = overrides.get(ext) {
            (ov_status.clone(), true)
        } else if let Some(er) = existing_row {
            if !scan_status.is_empty() && er.support_status != scan_status {
                // existing differs from scan → treat as manually set
                (er.support_status.clone(), true)
            } else {
                // same as scan or no scan data
                let st = if scan_status.is_empty() {
                    er.support_status.clone()
                } else {
                    scan_status.to_string()
                };
                (st, false)
            }
        } else {
            (scan_status.to_string(), false)
        };

        if support_status.is_empty() {
            // No data source at all — skip (shouldn't happen given all_exts construction)
            continue;
        }

        // evidence_kind: always use latest scan data; else fall back to existing
        let evidence_kind = if !scan_evidence.is_empty() {
            scan_evidence
        } else if let Some(er) = existing_row {
            er.evidence_kind.clone()
        } else {
            String::new()
        };

        // isa_bases: always use latest scan; fall back to existing
        let isa_bases = if !scan_bases.is_empty() {
            scan_bases
        } else if let Some(er) = existing_row {
            er.isa_bases.clone()
        } else {
            String::new()
        };

        // distinct_instructions and sample_files: always latest scan
        let distinct_instructions = if !scan_instrs.is_empty() {
            scan_instrs
        } else if let Some(er) = existing_row {
            er.distinct_instructions.clone()
        } else {
            String::new()
        };

        let sample_files = if !scan_files.is_empty() {
            scan_files
        } else if let Some(er) = existing_row {
            er.sample_files.clone()
        } else {
            String::new()
        };

        // notes: merge override notes + existing notes (no duplicates, newline-separated)
        let notes = {
            let mut parts: Vec<String> = Vec::new();
            if let Some((_, ov_notes)) = overrides.get(ext) {
                if !ov_notes.is_empty() {
                    parts.push(ov_notes.clone());
                }
            }
            if let Some(er) = existing_row {
                if !er.notes.is_empty() {
                    // Only add existing notes if not already contained in override notes
                    let already = parts.iter().any(|p| p.contains(&er.notes) || er.notes.contains(p.as_str()));
                    if !already {
                        parts.push(er.notes.clone());
                    }
                }
            }
            parts.join(" | ")
        };

        output_rows.push(Row {
            extension: ext.clone(),
            support_status,
            isa_bases,
            evidence_kind,
            distinct_instructions,
            sample_files,
            notes,
            is_manual,
        });
    }

    // Write output
    write_supported(&out_path, &output_rows)?;

    let used_count = output_rows.iter().filter(|r| r.support_status == "used").count();
    let declared_count = output_rows.iter().filter(|r| r.support_status == "declared").count();
    let other_count = output_rows.len() - used_count - declared_count;

    Ok(VerdictReport {
        written_to: out_path,
        total: output_rows.len(),
        used: used_count,
        declared: declared_count,
        other: other_count,
    })
}
