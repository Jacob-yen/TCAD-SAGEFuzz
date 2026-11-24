use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};

use globset::{Glob, GlobSet, GlobSetBuilder};
use regex::Regex;
use riscv_instruction_schema::Instruction;
use walkdir::WalkDir;

use crate::data::load_instructions;

const MAX_SAMPLE_PATHS: usize = 5;
const MAX_EVIDENCE_PER_INST: usize = 8;
const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_SNIPPET_LEN: usize = 240;

// ---------------------------------------------------------------------------
// PathClass
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum PathClass {
    Core,
    Tests,
    Vendor,
    Examples,
    Docs,
    Software,
    Other,
}

impl fmt::Display for PathClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathClass::Core => write!(f, "Core"),
            PathClass::Tests => write!(f, "Tests"),
            PathClass::Vendor => write!(f, "Vendor"),
            PathClass::Examples => write!(f, "Examples"),
            PathClass::Docs => write!(f, "Docs"),
            PathClass::Software => write!(f, "Software"),
            PathClass::Other => write!(f, "Other"),
        }
    }
}

/// Classify a relative path by checking each path component against known
/// category keywords. Earlier (higher-priority) categories win.
pub fn classify_path(rel: &Path) -> PathClass {
    for component in rel.components() {
        let seg = component.as_os_str().to_string_lossy().to_ascii_lowercase();
        // Vendor / external sources
        if matches!(
            seg.as_str(),
            "vendor"
                | "vendored"
                | "external"
                | "extern"
                | "third_party"
                | "thirdparty"
                | "3rdparty"
                | "submodules"
                | "deps"
        ) {
            return PathClass::Vendor;
        }
        // Well-known vendored test suites by name
        if seg.contains("riscv-tests")
            || seg.contains("riscv_tests")
            || seg.contains("riscv-arch-test")
            || seg.contains("riscv_arch_test")
            || seg.contains("riscv-compliance")
            || seg.contains("core-v-verif")
            || seg.contains("coremark")
            || seg.contains("dhrystone")
        {
            return PathClass::Vendor;
        }
        // Test suites
        if matches!(
            seg.as_str(),
            "tests"
                | "test"
                | "verif"
                | "verification"
                | "tb"
                | "testbench"
                | "testbenches"
                | "sim"
                | "simulation"
                | "cocotb"
                | "uvm"
                | "dv"
        ) {
            return PathClass::Tests;
        }
        // Examples / demos
        if matches!(seg.as_str(), "examples" | "example" | "demo" | "demos" | "sample" | "samples") {
            return PathClass::Examples;
        }
        // Documentation
        if matches!(seg.as_str(), "docs" | "doc" | "documentation" | "manual" | "manuals") {
            return PathClass::Docs;
        }
        // Software / firmware (bare-metal apps that are not the core itself)
        if matches!(
            seg.as_str(),
            "software" | "firmware" | "sw" | "apps" | "app" | "bootloader" | "bsp" | "hal"
        ) {
            return PathClass::Software;
        }
        // Core RTL / source
        if matches!(seg.as_str(), "src" | "rtl" | "hdl" | "generators" | "core" | "ip") {
            return PathClass::Core;
        }
    }
    PathClass::Other
}

// ---------------------------------------------------------------------------
// Internal data structures
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Evidence {
    pub file: String,
    pub line: usize,
    pub kind: String, // "asm", "sv-enum", "scala-opcode", "python-key", ...
    pub snippet: String,
}

#[derive(Default, Debug)]
struct InstHit {
    asm_count: usize,
    hdl_count: usize,
    files: BTreeSet<String>,
    per_class: BTreeMap<PathClass, usize>,
    evidence: Vec<Evidence>,
}

impl InstHit {
    fn total(&self) -> usize {
        self.asm_count + self.hdl_count
    }
    fn evidence_kind(&self) -> String {
        match (self.asm_count > 0, self.hdl_count > 0) {
            (true, true) => "asm;hdl".into(),
            (true, false) => "asm".into(),
            (false, true) => "hdl".into(),
            (false, false) => String::new(),
        }
    }
    fn push_evidence(&mut self, ev: Evidence) {
        if self.evidence.len() < MAX_EVIDENCE_PER_INST {
            self.evidence.push(ev);
        }
    }
}

fn snippet_at(body: &str, byte_offset: usize) -> (usize, String) {
    let mut line_no = 1usize;
    let mut line_start = 0usize;
    for (i, ch) in body.char_indices() {
        if i >= byte_offset { break; }
        if ch == '\n' {
            line_no += 1;
            line_start = i + 1;
        }
    }
    let line_end = body[line_start..].find('\n').map(|p| line_start + p).unwrap_or(body.len());
    let raw = &body[line_start..line_end];
    let trimmed = raw.trim();
    let snip = if trimmed.len() > MAX_SNIPPET_LEN {
        format!("{}…", &trimmed[..MAX_SNIPPET_LEN])
    } else {
        trimmed.to_string()
    };
    (line_no, snip)
}

fn record_hit(
    hits: &mut HashMap<String, InstHit>,
    canonical: String,
    rel: &str,
    body: &str,
    byte_offset: usize,
    kind: &str,
    is_asm: bool,
    path_class: PathClass,
    class_asm: &mut BTreeMap<PathClass, usize>,
    class_hdl: &mut BTreeMap<PathClass, usize>,
    total_hits: &mut usize,
) {
    let h = hits.entry(canonical).or_default();
    if is_asm { h.asm_count += 1; } else { h.hdl_count += 1; }
    *h.per_class.entry(path_class).or_default() += 1;
    if h.files.len() < MAX_SAMPLE_PATHS {
        h.files.insert(rel.to_string());
    }
    let (line, snip) = snippet_at(body, byte_offset);
    h.push_evidence(Evidence {
        file: rel.to_string(),
        line,
        kind: kind.to_string(),
        snippet: snip,
    });
    *total_hits += 1;
    if is_asm {
        *class_asm.entry(path_class).or_default() += 1;
    } else {
        *class_hdl.entry(path_class).or_default() += 1;
    }
}

#[derive(Default)]
struct MarchHit {
    raw: BTreeSet<String>,
    files: BTreeSet<String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct ScanOptions {
    pub include_hdl: bool,
    /// Glob patterns (relative to target root) to exclude.
    pub exclude: Vec<String>,
}

pub struct ScanReport {
    pub instructions_csv: PathBuf,
    pub extensions_csv: PathBuf,
    pub march_csv: PathBuf,
    pub path_class_summary_csv: PathBuf,
    pub evidence_jsonl: PathBuf,
    pub files_scanned: usize,
    pub instruction_hits: usize,
    pub hdl_files_scanned: usize,
}

pub fn run(
    target: &Path,
    out_dir: &Path,
    opts: ScanOptions,
) -> Result<ScanReport, Box<dyn std::error::Error>> {
    if !target.is_dir() {
        return Err(format!("target is not a directory: {}", target.display()).into());
    }
    std::fs::create_dir_all(out_dir)?;

    let catalog = load_instructions()?;
    let mnemonic_index = build_mnemonic_index(&catalog);
    let ext_index = build_ext_index(&catalog);

    // Build exclude GlobSet
    let exclude_set = build_exclude_set(&opts.exclude)?;

    let asm_macro_re = Regex::new(r"\b(?:global_asm|asm|naked_asm)\s*!\s*\(").unwrap();
    let c_asm_open_re = Regex::new(r"__asm__?\s*(?:__volatile__\s*)?\(").unwrap();
    let string_lit_re = Regex::new(r#""((?:\\.|[^"\\])*)""#).unwrap();
    let march_re = Regex::new(r"-march=([A-Za-z0-9_]+)").unwrap();
    // rocket-chip / BOOM Chisel decode tables: `MNEMONIC -> List(...)` or `MNEMONIC =>`
    let hdl_decode_re = Regex::new(r"\b([A-Z][A-Z0-9_]{1,})\s*(?:->|=>)").unwrap();
    // Verilog/SV: `` `define INST_MNEMONIC `` or `localparam MNEMONIC`
    let hdl_define_re =
        Regex::new(r"(?:`define|localparam|parameter)\s+(?:[A-Z_]*INST_)?([A-Z][A-Z0-9_]{1,})\b")
            .unwrap();
    // pkg::OPNAME style (CVA6 ariane_pkg::ADD, ariane_pkg::SH1ADD) and
    // PREFIX_OPNAME style (ibex ALU_CLZ, ALU_BEXT, ALU_CLMUL).
    // The trailing `\b` avoids matching in mid-identifier positions.
    let hdl_pkg_enum_re = Regex::new(
        r"(?:[a-z][a-z0-9_]*::)?([A-Z]{2,}[A-Z0-9_]*)\b",
    )
    .unwrap();
    // Mnemonic string literals in HDL tracer/logging code (ibex_tracer.sv style):
    // decode_r_insn("bclr"), decode_cr_insn("c.add"), etc.
    // Only matches lowercase strings that could be instruction names.
    let hdl_str_re = Regex::new(r#""([a-z][a-z0-9.]*[a-z0-9])""#).unwrap();
    // T-Head / similar RTLs put mnemonics in line-end comments:
    //   32'b00000???...:  //amoadd .w   or   //c.fld   or   //fence.i
    // The space inside "amoadd .w" is T-Head's style — collapse it before lookup.
    let hdl_comment_re = Regex::new(r"//\s*([a-z][a-z0-9]*(?:\s*\.[a-z][a-z0-9]*)*)").unwrap();
    // Scala: `def C_ADD = BitPat(...)` — rocket-chip / boom Instructions.scala style.
    // Captures the constant name between `def ` and ` =`.
    let scala_def_re = Regex::new(r"\bdef\s+([A-Z][A-Z0-9_]{1,})\s*=").unwrap();
    // Scala: `Opcode("name")` — riscv-torture Inst.scala style.
    // Captures lowercase mnemonic strings inside the Opcode constructor.
    let scala_opcode_re = Regex::new(r#"\bOpcode\s*\(\s*"([a-z][a-z0-9.]*)"\s*\)"#).unwrap();
    // SystemVerilog typedef enum members: bare uppercase identifiers followed by `,` or `}`.
    // Used to capture members of `typedef enum { ADD, ADDI, ... } riscv_instr_name_t;`
    // (riscv-dv's primary instruction enumeration). Block boundaries are computed by
    // find_sv_enum_blocks() so we don't match arbitrary uppercase identifiers in the file.
    let sv_enum_member_re = Regex::new(r"(?m)^\s*([A-Z][A-Z0-9_]{1,})\s*[,}]").unwrap();
    // Python (Cascade, DIFUZZ-RTL): mnemonic string literals (single or double quoted).
    // Match lowercase tokens that look like RISC-V mnemonics: start with letter,
    // continue with [a-z0-9.] only.
    let py_str_re = Regex::new(r#"['"]([a-z][a-z0-9.]*)['"]"#).unwrap();
    // Scala ISA strings: "rv32gc", "rv64imafdcvh", "rv64imafdch_zba_zbb" etc.
    // Must start with rv32/rv64 followed by letters and underscored extensions.
    let scala_isa_re = Regex::new(r#""(rv(?:32|64)[a-z0-9_]+)""#).unwrap();
    // Scala extension name string literals in ISAExtensions / supportedExtensions lists:
    // "zca", "zcb", "zimop", "zihintntl" — lowercase only, 2-32 chars, no spaces.
    // Guarded by neighbour comma/bracket/paren to reduce false positives.
    let scala_ext_str_re = Regex::new(r#"(?:,|\[|\(|=)\s*"([a-z][a-z0-9]{1,31})""#).unwrap();
    // HDL/Makefile parameters that imply ISA extension support:
    // COMPRESSED_ISA=1, COMPRESSED=yes, WITH_COMPRESSED=1, RVC=1, ENABLE_COMPRESSED=yes →  C / Zca / Zcd / Zcf
    // AMO_SUPPORT=1, AMO_ENABLED=yes, WITH_AMO=1, ENABLE_AMO=yes              → Zaamo / Zalrsc
    // ENABLE_COUNTERS=1                                                        → Zicsr (rdcycle/rdtime)
    let hdl_param_re = Regex::new(
        r"(?:parameter\s+|^|\s)(COMPRESSED_ISA|COMPRESSED|WITH_COMPRESSED|RVC|ENABLE_COMPRESSED|AMO_SUPPORT|AMO_ENABLED|WITH_AMO|ENABLE_AMO|ENABLE_COUNTERS)\b[^;#\n]*?(?:1|yes|YES|true|TRUE)\b",
    ).unwrap();

    let mut hits: HashMap<String, InstHit> = HashMap::new();
    let mut march = MarchHit::default();
    // Extensions declared via Scala ISA string or extension-list literals (evidence_kind = "scala-isa")
    let mut scala_ext_declared: BTreeSet<String> = BTreeSet::new();
    let mut files_scanned = 0usize;
    let mut hdl_files_scanned = 0usize;
    let mut total_hits = 0usize;

    // per-class file counters for the summary CSV
    let mut class_files: BTreeMap<PathClass, usize> = BTreeMap::new();
    let mut class_asm: BTreeMap<PathClass, usize> = BTreeMap::new();
    let mut class_hdl: BTreeMap<PathClass, usize> = BTreeMap::new();

    for entry in WalkDir::new(target).follow_links(false).into_iter().filter_entry(|e| {
        let n = e.file_name().to_string_lossy();
        !(e.depth() > 0
            && (n == ".git"
                || n == "target"
                || n == "node_modules"
                || n == ".direnv"
                || n == ".bloop"
                || n == ".metals"
                || n == "build"
                || n.starts_with(".idea")))
    }) {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let rel_path = path.strip_prefix(target).unwrap_or(path);
        let rel = rel_path.to_string_lossy().to_string();

        // Apply user-supplied exclude globs
        if !exclude_set.is_empty() && exclude_set.is_match(rel_path) {
            continue;
        }

        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.len() > MAX_FILE_BYTES {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        let body = match std::fs::read_to_string(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        files_scanned += 1;

        let path_class = classify_path(rel_path);
        *class_files.entry(path_class).or_default() += 1;

        let mut asm_regions_with_offsets: Vec<(String, usize)> = Vec::new();
        let is_hdl = matches!(
            ext.as_deref(),
            Some("scala") | Some("v") | Some("sv") | Some("svh") | Some("vh")
                | Some("vhd") | Some("vhdl") | Some("chisel")
        );
        let is_scala = ext.as_deref() == Some("scala");
        // Makefiles and shell scripts that pass build parameters (COMPRESSED=yes, AMO=yes)
        let is_build_script = matches!(
            ext.as_deref(),
            Some("mk") | Some("sh") | Some("bash") | Some("cmake") | Some("tcl")
        ) || matches!(
            path.file_name().and_then(|s| s.to_str()),
            Some("Makefile") | Some("makefile") | Some("GNUmakefile") | Some("build.sh")
                | Some("CMakeLists.txt") | Some("Dockerfile")
        );
        match ext.as_deref() {
            Some("s") | Some("asm") => asm_regions_with_offsets.push((body.clone(), 0)),
            Some("rs") => collect_macro_strings_with_offsets(&body, &asm_macro_re, &string_lit_re, &mut asm_regions_with_offsets),
            Some("c") | Some("h") | Some("cc") | Some("cpp") | Some("cxx") | Some("hpp")
            | Some("hh") | Some("hxx") => {
                collect_macro_strings_with_offsets(&body, &c_asm_open_re, &string_lit_re, &mut asm_regions_with_offsets)
            }
            _ => {}
        }

        for region_info in &asm_regions_with_offsets {
            let (region, base_offset) = region_info;
            for tok in tokenize_asm_with_offsets(region) {
                if let Some(canonical) = mnemonic_index.get(&tok.text.to_ascii_lowercase()) {
                    record_hit(
                        &mut hits,
                        canonical.clone(),
                        &rel,
                        &body,
                        base_offset + tok.offset,
                        "asm",
                        true,
                        path_class,
                        &mut class_asm,
                        &mut class_hdl,
                        &mut total_hits,
                    );
                }
            }
        }

        if opts.include_hdl && is_hdl {
            hdl_files_scanned += 1;
            for cap in hdl_decode_re.captures_iter(&body) {
                let m = cap.get(1).unwrap();
                if let Some(canonical) = lookup_with_fallback(&mnemonic_index, m.as_str()) {
                    record_hit(&mut hits, canonical, &rel, &body, m.start(), "hdl-decode",
                        false, path_class, &mut class_asm, &mut class_hdl, &mut total_hits);
                }
            }
            for cap in hdl_define_re.captures_iter(&body) {
                let m = cap.get(1).unwrap();
                if let Some(canonical) = lookup_with_fallback(&mnemonic_index, m.as_str()) {
                    record_hit(&mut hits, canonical, &rel, &body, m.start(), "hdl-define",
                        false, path_class, &mut class_asm, &mut class_hdl, &mut total_hits);
                }
            }
            for cap in hdl_pkg_enum_re.captures_iter(&body) {
                let m = cap.get(1).unwrap();
                let raw = m.as_str();
                if raw.len() < 2 { continue; }
                if let Some(canonical) = lookup_with_fallback(&mnemonic_index, raw) {
                    record_hit(&mut hits, canonical, &rel, &body, m.start(), "hdl-pkg-enum",
                        false, path_class, &mut class_asm, &mut class_hdl, &mut total_hits);
                }
            }
            for cap in hdl_str_re.captures_iter(&body) {
                let m = cap.get(1).unwrap();
                if let Some(canonical) = lookup_with_fallback(&mnemonic_index, m.as_str()) {
                    record_hit(&mut hits, canonical, &rel, &body, m.start(), "hdl-str",
                        false, path_class, &mut class_asm, &mut class_hdl, &mut total_hits);
                }
            }
            for cap in hdl_comment_re.captures_iter(&body) {
                let m = cap.get(1).unwrap();
                let normalized = m.as_str().replace(' ', "");
                if let Some(canonical) = lookup_with_fallback(&mnemonic_index, &normalized) {
                    record_hit(&mut hits, canonical, &rel, &body, m.start(), "hdl-comment",
                        false, path_class, &mut class_asm, &mut class_hdl, &mut total_hits);
                }
            }
            for cap in scala_def_re.captures_iter(&body) {
                let m = cap.get(1).unwrap();
                if let Some(canonical) = lookup_with_fallback(&mnemonic_index, m.as_str()) {
                    record_hit(&mut hits, canonical, &rel, &body, m.start(), "scala-def",
                        false, path_class, &mut class_asm, &mut class_hdl, &mut total_hits);
                }
            }
            // SystemVerilog typedef enum members: bare uppercase identifiers followed by `,` or `}`
            // inside a `typedef enum { ... }` block (riscv-dv riscv_instr_name_t style).
            for (start, end) in find_sv_enum_blocks(&body) {
                let block = &body[start..end];
                for cap in sv_enum_member_re.captures_iter(block) {
                    let m = cap.get(1).unwrap();
                    let raw = m.as_str();
                    if SV_ENUM_BLACKLIST.contains(&raw) { continue; }
                    if let Some(canonical) = lookup_sv_enum(&mnemonic_index, raw) {
                        record_hit(&mut hits, canonical, &rel, &body, start + m.start(), "sv-enum",
                            false, path_class, &mut class_asm, &mut class_hdl, &mut total_hits);
                    }
                }
            }
        }

        // Scala source code (riscv-torture Inst.scala): `Opcode("name")` literals
        if is_scala {
            for cap in scala_opcode_re.captures_iter(&body) {
                let m = cap.get(1).unwrap();
                if let Some(canonical) = mnemonic_index.get(&m.as_str().to_ascii_lowercase()) {
                    record_hit(&mut hits, canonical.clone(), &rel, &body, m.start(), "scala-opcode",
                        false, path_class, &mut class_asm, &mut class_hdl, &mut total_hits);
                }
            }
        }

        // Python source code (Cascade, DIFUZZ-RTL): mnemonic string literals.
        // Only consider lowercase strings that look like RISC-V mnemonics (start with
        // letter, contain only [a-z0-9.]) to avoid collecting unrelated strings.
        if ext.as_deref() == Some("py") {
            for cap in py_str_re.captures_iter(&body) {
                let m = cap.get(1).unwrap();
                let raw = m.as_str();
                if raw.len() < 2 { continue; }
                if let Some(canonical) = mnemonic_index.get(raw) {
                    record_hit(&mut hits, canonical.clone(), &rel, &body, m.start(), "python-str",
                        false, path_class, &mut class_asm, &mut class_hdl, &mut total_hits);
                }
            }
        }

        // Scala ISA strings and extension-list literals (both HDL and non-HDL Scala files).
        // These don't contribute to instruction hit counts but feed the scala-isa declared set.
        if is_scala {
            // "rv64imafdcvh", "rv64gc_zba_zbb" etc. — parsed like -march= strings
            for cap in scala_isa_re.captures_iter(&body) {
                let raw = cap.get(1).unwrap().as_str();
                let (_, exts) = parse_march(raw);
                for ext_name in exts {
                    if let Some(canonical) = ext_index.get(&ext_name) {
                        scala_ext_declared.insert(canonical.clone());
                    }
                }
                march.raw.insert(raw.to_string());
                if march.files.len() < MAX_SAMPLE_PATHS {
                    march.files.insert(rel.clone());
                }
            }
            // Extension name string literals: "zca", "zcb", "zimop", etc.
            for cap in scala_ext_str_re.captures_iter(&body) {
                let raw = cap.get(1).unwrap().as_str();
                if let Some(canonical) = ext_index.get(raw) {
                    scala_ext_declared.insert(canonical.clone());
                }
            }
        }

        // HDL parameters and build scripts that imply extension support.
        // Only run when --include-hdl is set (HDL files) or always for build scripts.
        if (opts.include_hdl && is_hdl) || is_build_script {
            for cap in hdl_param_re.captures_iter(&body) {
                let param = cap.get(1).unwrap().as_str();
                let implied: &[&str] = match param {
                    "COMPRESSED_ISA" | "COMPRESSED" | "WITH_COMPRESSED"
                    | "RVC" | "ENABLE_COMPRESSED" => &["C", "Zca", "Zcd", "Zcf"],
                    "AMO_SUPPORT" | "AMO_ENABLED" | "WITH_AMO" | "ENABLE_AMO" => {
                        &["Zaamo", "Zalrsc"]
                    }
                    "ENABLE_COUNTERS" => &["Zicsr"],
                    _ => &[],
                };
                for ext_name in implied {
                    scala_ext_declared.insert(ext_name.to_string());
                }
                if !implied.is_empty() {
                    if march.files.len() < MAX_SAMPLE_PATHS {
                        march.files.insert(rel.clone());
                    }
                }
            }
        }

        for cap in march_re.captures_iter(&body) {
            let raw = cap.get(1).unwrap().as_str().to_ascii_lowercase();
            march.raw.insert(raw);
            if march.files.len() < MAX_SAMPLE_PATHS {
                march.files.insert(rel.clone());
            }
        }
    }

    let by_name: HashMap<&str, &Instruction> =
        catalog.iter().map(|i| (i.name.as_str(), i)).collect();

    // Fold scala-declared extensions into the march raw set so they appear in
    // march_flags.csv and get picked up by verdict as declared evidence.
    if !scala_ext_declared.is_empty() {
        let exts_str = scala_ext_declared.iter().cloned().collect::<Vec<_>>().join("_");
        march.raw.insert(format!("scala-isa:{}", exts_str));
    }

    let instructions_csv = out_dir.join("instructions_found.csv");
    write_instructions_found(&hits, &by_name, &instructions_csv)?;

    let extensions_csv = out_dir.join("extensions_found.csv");
    write_extensions_found(&hits, &by_name, &extensions_csv)?;

    let march_csv = out_dir.join("march_flags.csv");
    write_march(&march, &march_csv)?;

    let path_class_summary_csv = out_dir.join("path_class_summary.csv");
    write_path_class_summary(
        &class_files,
        &class_asm,
        &class_hdl,
        &hits,
        &by_name,
        &path_class_summary_csv,
    )?;

    // Warn when non-core paths dominate the hit count
    let core_hits = class_asm.get(&PathClass::Core).copied().unwrap_or(0)
        + class_hdl.get(&PathClass::Core).copied().unwrap_or(0);
    let noise_hits = [PathClass::Tests, PathClass::Vendor]
        .iter()
        .map(|c| {
            class_asm.get(c).copied().unwrap_or(0) + class_hdl.get(c).copied().unwrap_or(0)
        })
        .sum::<usize>();
    if total_hits > 0 && noise_hits > core_hits {
        eprintln!(
            "WARNING: Non-core paths (Tests+Vendor) account for {} hits vs {} Core hits. \
             Use --exclude to filter vendored / test directories and re-scan for accurate results.",
            noise_hits, core_hits
        );
    }

    let evidence_jsonl = out_dir.join("instructions_evidence.jsonl");
    write_evidence_jsonl(&hits, &evidence_jsonl)?;

    Ok(ScanReport {
        instructions_csv,
        extensions_csv,
        march_csv,
        path_class_summary_csv,
        evidence_jsonl,
        files_scanned,
        instruction_hits: total_hits,
        hdl_files_scanned,
    })
}

// ---------------------------------------------------------------------------
// Helper: build GlobSet from exclude patterns
// ---------------------------------------------------------------------------

fn build_exclude_set(patterns: &[String]) -> Result<GlobSet, Box<dyn std::error::Error>> {
    let mut builder = GlobSetBuilder::new();
    for pat in patterns {
        builder.add(Glob::new(pat)?);
    }
    Ok(builder.build()?)
}

// ---------------------------------------------------------------------------
// CSV writers
// ---------------------------------------------------------------------------

fn write_instructions_found(
    hits: &HashMap<String, InstHit>,
    by_name: &HashMap<&str, &Instruction>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rows: Vec<_> = hits.iter().collect();
    rows.sort_by(|a, b| b.1.total().cmp(&a.1.total()).then_with(|| a.0.cmp(b.0)));

    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "instruction",
        "extension",
        "isa_bases",
        "evidence_kind",
        "asm_occurrences",
        "hdl_occurrences",
        "core_hits",
        "tests_hits",
        "vendor_hits",
        "sample_files",
        "evidence_locations",
    ])?;
    for (name, hit) in rows {
        let inst = by_name.get(name.as_str());
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
        let samples = hit.files.iter().cloned().collect::<Vec<_>>().join(";");
        let core_h = hit.per_class.get(&PathClass::Core).copied().unwrap_or(0);
        let tests_h = hit.per_class.get(&PathClass::Tests).copied().unwrap_or(0);
        let vendor_h = hit.per_class.get(&PathClass::Vendor).copied().unwrap_or(0);
        let evidence_locs = hit
            .evidence
            .iter()
            .map(|e| format!("{}:{}({})", e.file, e.line, e.kind))
            .collect::<Vec<_>>()
            .join(";");
        w.write_record([
            name,
            &ext,
            &bases,
            &hit.evidence_kind(),
            &hit.asm_count.to_string(),
            &hit.hdl_count.to_string(),
            &core_h.to_string(),
            &tests_h.to_string(),
            &vendor_h.to_string(),
            &samples,
            &evidence_locs,
        ])?;
    }
    w.flush()?;
    Ok(())
}

/// Write per-instruction evidence as JSONL (one JSON object per line) so the
/// AI / human reviewer can verify each match against the actual source file
/// and line. Each record includes the instruction name, the file, the line
/// number, the kind of pattern that matched, and a trimmed snippet of the line.
fn write_evidence_jsonl(
    hits: &HashMap<String, InstHit>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut names: Vec<&String> = hits.keys().collect();
    names.sort();
    for name in names {
        let hit = &hits[name];
        for ev in &hit.evidence {
            let obj = serde_json::json!({
                "instruction": name,
                "file": ev.file,
                "line": ev.line,
                "kind": ev.kind,
                "snippet": ev.snippet,
            });
            writeln!(w, "{}", obj)?;
        }
    }
    w.flush()?;
    Ok(())
}

fn write_extensions_found(
    hits: &HashMap<String, InstHit>,
    by_name: &HashMap<&str, &Instruction>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    #[derive(Default)]
    struct Agg {
        asm_total: usize,
        hdl_total: usize,
        core_total: usize,
        tests_total: usize,
        vendor_total: usize,
        distinct: BTreeSet<String>,
        files: BTreeSet<String>,
        bases: BTreeSet<String>,
    }
    let mut by_ext: BTreeMap<String, Agg> = BTreeMap::new();
    for (name, hit) in hits {
        let Some(inst) = by_name.get(name.as_str()) else { continue };
        let ext = inst.extension.to_string();
        let agg = by_ext.entry(ext).or_default();
        agg.asm_total += hit.asm_count;
        agg.hdl_total += hit.hdl_count;
        agg.core_total += hit.per_class.get(&PathClass::Core).copied().unwrap_or(0);
        agg.tests_total += hit.per_class.get(&PathClass::Tests).copied().unwrap_or(0);
        agg.vendor_total += hit.per_class.get(&PathClass::Vendor).copied().unwrap_or(0);
        agg.distinct.insert(name.clone());
        for b in &inst.isa_bases {
            agg.bases.insert(b.to_string());
        }
        for f in &hit.files {
            if agg.files.len() < MAX_SAMPLE_PATHS {
                agg.files.insert(f.clone());
            }
        }
    }

    let mut rows: Vec<_> = by_ext.into_iter().collect();
    rows.sort_by(|a, b| {
        (b.1.asm_total + b.1.hdl_total)
            .cmp(&(a.1.asm_total + a.1.hdl_total))
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "extension",
        "isa_bases",
        "distinct_instructions",
        "asm_occurrences",
        "hdl_occurrences",
        "core_hits",
        "tests_hits",
        "vendor_hits",
        "evidence_kind",
        "instructions",
        "sample_files",
    ])?;
    for (ext, agg) in rows {
        let kind = match (agg.asm_total > 0, agg.hdl_total > 0) {
            (true, true) => "asm;hdl",
            (true, false) => "asm",
            (false, true) => "hdl",
            _ => "",
        };
        let bases = agg.bases.iter().cloned().collect::<Vec<_>>().join(";");
        let insts = agg.distinct.iter().cloned().collect::<Vec<_>>().join(";");
        let files = agg.files.iter().cloned().collect::<Vec<_>>().join(";");
        w.write_record([
            ext,
            bases,
            agg.distinct.len().to_string(),
            agg.asm_total.to_string(),
            agg.hdl_total.to_string(),
            agg.core_total.to_string(),
            agg.tests_total.to_string(),
            agg.vendor_total.to_string(),
            kind.to_string(),
            insts,
            files,
        ])?;
    }
    w.flush()?;
    Ok(())
}

fn write_march(march: &MarchHit, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut w = csv::Writer::from_path(path)?;
    w.write_record(["march_string", "base", "declared_extensions", "sample_files"])?;
    let files = march.files.iter().cloned().collect::<Vec<_>>().join(";");
    for raw in &march.raw {
        let (base, exts) = parse_march(raw);
        // For the scala-isa sentinel the exts are already canonical; for real march strings
        // they are lowercase and need to stay as-is (verdict.rs will canonicalize them).
        w.write_record([raw.clone(), base, exts.join(";"), files.clone()])?;
    }
    w.flush()?;
    Ok(())
}

fn write_path_class_summary(
    class_files: &BTreeMap<PathClass, usize>,
    class_asm: &BTreeMap<PathClass, usize>,
    class_hdl: &BTreeMap<PathClass, usize>,
    hits: &HashMap<String, InstHit>,
    by_name: &HashMap<&str, &Instruction>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build top extensions per class
    let mut class_ext_hits: BTreeMap<PathClass, BTreeMap<String, usize>> = BTreeMap::new();
    for (name, hit) in hits {
        let Some(inst) = by_name.get(name.as_str()) else { continue };
        let ext_name = inst.extension.to_string();
        for (cls, cnt) in &hit.per_class {
            *class_ext_hits.entry(*cls).or_default().entry(ext_name.clone()).or_default() += cnt;
        }
    }

    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "path_class",
        "files_scanned",
        "asm_occurrences",
        "hdl_occurrences",
        "top_extensions",
    ])?;

    let all_classes = [
        PathClass::Core,
        PathClass::Tests,
        PathClass::Vendor,
        PathClass::Examples,
        PathClass::Docs,
        PathClass::Software,
        PathClass::Other,
    ];

    for cls in all_classes {
        let files = class_files.get(&cls).copied().unwrap_or(0);
        let asm = class_asm.get(&cls).copied().unwrap_or(0);
        let hdl = class_hdl.get(&cls).copied().unwrap_or(0);
        let top_exts = if let Some(ext_map) = class_ext_hits.get(&cls) {
            let mut ev: Vec<_> = ext_map.iter().collect();
            ev.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            ev.iter()
                .take(5)
                .map(|(e, c)| format!("{}({})", e, c))
                .collect::<Vec<_>>()
                .join(";")
        } else {
            String::new()
        };
        w.write_record([
            cls.to_string(),
            files.to_string(),
            asm.to_string(),
            hdl.to_string(),
            top_exts,
        ])?;
    }
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_macro_strings(
    source: &str,
    open_re: &Regex,
    string_re: &Regex,
    out: &mut Vec<String>,
) {
    for m in open_re.find_iter(source) {
        let after = &source[m.end()..];
        let close = match find_balanced_close(after) {
            Some(idx) => idx,
            None => continue,
        };
        let inside = &after[..close];
        let mut joined = String::new();
        for cap in string_re.captures_iter(inside) {
            let raw = cap.get(1).unwrap().as_str();
            joined.push_str(&unescape(raw));
            joined.push('\n');
        }
        if !joined.is_empty() {
            out.push(joined);
        }
    }
}

/// Variant of collect_macro_strings that records, for each emitted region, the
/// byte offset in the ORIGINAL source where the first contributing string
/// literal began. We use this offset to attribute asm-region hits back to the
/// correct line in the original file (snippet_at).
fn collect_macro_strings_with_offsets(
    source: &str,
    open_re: &Regex,
    string_re: &Regex,
    out: &mut Vec<(String, usize)>,
) {
    for m in open_re.find_iter(source) {
        let after = &source[m.end()..];
        let close = match find_balanced_close(after) {
            Some(idx) => idx,
            None => continue,
        };
        let inside = &after[..close];
        let macro_call_start = m.start();
        let mut joined = String::new();
        for cap in string_re.captures_iter(inside) {
            let raw = cap.get(1).unwrap().as_str();
            joined.push_str(&unescape(raw));
            joined.push('\n');
        }
        if !joined.is_empty() {
            out.push((joined, macro_call_start));
        }
    }
}

/// Find every `typedef enum { ... }` block in a SystemVerilog/Verilog source,
/// returning (start_byte_of_open_brace+1, end_byte_of_close_brace_exclusive).
/// Used to scope SV-enum member matching so we don't pick up unrelated
/// uppercase identifiers elsewhere in the file.
pub fn find_sv_enum_blocks(body: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let re = Regex::new(r"\btypedef\s+enum\b[^{]*\{").unwrap();
    for m in re.find_iter(body) {
        let open = m.end() - 1; // position of '{'
        // Find matching '}' with simple brace counting (SV enums don't nest braces).
        let bytes = body.as_bytes();
        let mut depth: i32 = 0;
        let mut i = open;
        while i < bytes.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        out.push((open + 1, i));
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }
    out
}

/// Names that appear inside SystemVerilog typedef enum blocks but are NOT
/// instruction mnemonics. Filter them out before mnemonic lookup.
pub const SV_ENUM_BLACKLIST: &[&str] = &[
    "INVALID_INSTR", "NOP", "ILLEGAL", "RESERVED", "DEFAULT", "NONE", "UNKNOWN",
];

/// Lookup an SV enum member token. Direct match first, then `_`→`.` (BOOM-style
/// AMOMAXU_W → amomaxu.w). Unlike lookup_with_fallback, do NOT strip a leading
/// PREFIX_ — riscv-dv enum members are bare mnemonics (C_FLD is c.fld, not fld).
pub fn lookup_sv_enum(index: &HashMap<String, String>, raw: &str) -> Option<String> {
    let lower = raw.to_ascii_lowercase();
    if let Some(c) = index.get(&lower) { return Some(c.clone()); }
    if lower.contains('_') {
        let dotted = lower.replace('_', ".");
        if let Some(c) = index.get(&dotted) { return Some(c.clone()); }
    }
    None
}

/// Token produced by tokenize_asm_with_offsets — the mnemonic plus its byte
/// offset within the input region (so callers can locate it back in the file).
pub struct AsmTok<'a> {
    pub text: &'a str,
    pub offset: usize,
}

/// Like tokenize_asm but also records the byte offset of each mnemonic within
/// the input string. Used to attach line numbers to asm-region hits.
fn tokenize_asm_with_offsets(text: &str) -> Vec<AsmTok<'_>> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut line_start = 0usize;
    while line_start <= bytes.len() {
        let line_end = bytes[line_start..]
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| line_start + p)
            .unwrap_or(bytes.len());
        let raw_line = &text[line_start..line_end];
        // strip `#` and `//` comments
        let cut = raw_line.find('#').unwrap_or(raw_line.len());
        let mut effective = &raw_line[..cut];
        if let Some(p) = effective.find("//") { effective = &effective[..p]; }
        let trimmed = effective.trim_start();
        let lead = effective.len() - trimmed.len();
        if !trimmed.is_empty() && !trimmed.starts_with('.') {
            // skip optional `label:`
            let (after_label, label_skip) = match trimmed.find(':') {
                Some(p) => (trimmed[p + 1..].trim_start(), p + 1 + (trimmed[p + 1..].len() - trimmed[p + 1..].trim_start().len())),
                None => (trimmed, 0),
            };
            if !after_label.is_empty() {
                let first_end = after_label
                    .find(|c: char| c.is_whitespace() || c == ',')
                    .unwrap_or(after_label.len());
                let first = &after_label[..first_end];
                if !first.is_empty()
                    && first.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_')
                    && first.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
                {
                    let off = line_start + lead + label_skip;
                    out.push(AsmTok { text: first, offset: off });
                }
            }
        }
        if line_end >= bytes.len() { break; }
        line_start = line_end + 1;
    }
    out
}

fn find_balanced_close(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth: i32 = 1;
    let mut i = 0;
    let mut in_string = false;
    let mut in_char = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    while i < bytes.len() {
        let c = bytes[i];
        if in_line_comment {
            if c == b'\n' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }
        if in_block_comment {
            if c == b'*' && bytes.get(i + 1) == Some(&b'/') {
                in_block_comment = false;
                i += 2;
                continue;
            }
            i += 1;
            continue;
        }
        if in_string {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if in_char {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'\'' {
                in_char = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'\'' => in_char = true,
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                in_line_comment = true;
                i += 2;
                continue;
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                in_block_comment = true;
                i += 2;
                continue;
            }
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn unescape(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('\'') => out.push('\''),
            Some('0') => out.push('\0'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

pub fn build_mnemonic_index(catalog: &[Instruction]) -> HashMap<String, String> {
    let mut idx = HashMap::new();
    for inst in catalog {
        let lower = inst.name.to_ascii_lowercase();
        idx.insert(lower.clone(), inst.name.clone());
        // Also index without dots so that patterns like `fencei` or `fence_i` resolve.
        if lower.contains('.') {
            let nodot = lower.replace('.', "");
            idx.entry(nodot.clone()).or_insert_with(|| inst.name.clone());
            let underscore = lower.replace('.', "_");
            idx.entry(underscore).or_insert_with(|| inst.name.clone());
        }
    }
    idx
}

/// Build a lowercase-name → canonical-ISAExtension-string index for the catalog.
/// Used to canonicalize extension names found in Scala strings and march flags.
pub fn build_ext_index(catalog: &[Instruction]) -> HashMap<String, String> {
    let mut idx: HashMap<String, String> = HashMap::new();
    for inst in catalog {
        let ext = inst.extension.to_string();
        idx.entry(ext.to_ascii_lowercase()).or_insert(ext);
    }
    idx
}

/// Try `raw` lower-cased directly; if it misses, try replacing `_` with `.`
/// (BOOM/rocket-chip: `AMOMAXU_W` → `amomaxu.w`); and if the token looks like
/// `PREFIX_OPNAME`, also try just the suffix after the first underscore
/// (ibex: `ALU_BCLR` → `bclr`, `ALU_SH1ADD` → `sh1add`).
pub fn lookup_with_fallback(index: &HashMap<String, String>, raw: &str) -> Option<String> {
    let lower = raw.to_ascii_lowercase();
    if let Some(c) = index.get(&lower) {
        return Some(c.clone());
    }
    if lower.contains('_') {
        let dotted = lower.replace('_', ".");
        if let Some(c) = index.get(&dotted) {
            return Some(c.clone());
        }
        // Strip leading PREFIX_ (e.g. ALU_BCLR → bclr, ALU_SH1ADD → sh1add).
        if let Some((_prefix, suffix)) = lower.split_once('_') {
            if !suffix.is_empty() {
                if let Some(c) = index.get(suffix) {
                    return Some(c.clone());
                }
                let suffix_dotted = suffix.replace('_', ".");
                if let Some(c) = index.get(&suffix_dotted) {
                    return Some(c.clone());
                }
            }
        }
    }
    None
}

fn tokenize_asm(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw_line in text.lines() {
        let line = match raw_line.split_once('#') {
            Some((before, _)) => before,
            None => raw_line,
        };
        let line = match line.split_once("//") {
            Some((before, _)) => before,
            None => line,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('.') {
            continue;
        }
        let after_label = match trimmed.split_once(':') {
            Some((_, rest)) => rest.trim(),
            None => trimmed,
        };
        if after_label.is_empty() {
            continue;
        }
        let first = after_label
            .split(|c: char| c.is_whitespace() || c == ',')
            .next()
            .unwrap_or("");
        if first.is_empty() {
            continue;
        }
        if first
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_')
            && first.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
        {
            out.push(first.to_string());
        }
    }
    out
}

fn parse_march(raw: &str) -> (String, Vec<String>) {
    // Sentinel inserted for scala-isa extension declarations; extensions already canonical.
    if let Some(rest) = raw.strip_prefix("scala-isa:") {
        let exts = rest.split('_').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
        return ("scala".to_string(), exts);
    }

    let lower = raw.to_ascii_lowercase();
    let (base, rest) = if let Some(r) = lower.strip_prefix("rv64") {
        ("rv64".to_string(), r)
    } else if let Some(r) = lower.strip_prefix("rv32") {
        ("rv32".to_string(), r)
    } else {
        return (String::new(), Vec::new());
    };

    let mut parts = rest.split('_');
    let head = parts.next().unwrap_or("");
    let mut exts: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let push = |exts: &mut Vec<String>, seen: &mut BTreeSet<String>, name: &str| {
        if name.is_empty() {
            return;
        }
        if seen.insert(name.to_string()) {
            exts.push(name.to_string());
        }
    };
    for ch in head.chars() {
        match ch {
            'g' => {
                for name in ["i", "m", "a", "f", "d", "zicsr", "zifencei"] {
                    push(&mut exts, &mut seen, name);
                }
            }
            other => push(&mut exts, &mut seen, &other.to_string()),
        }
    }
    for tail in parts {
        push(&mut exts, &mut seen, tail);
    }
    (base, exts)
}
