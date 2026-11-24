use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::OpenOptions;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use log::{info, warn};
use rand::Rng as _;
use rand::SeedableRng as _;
use rand::prelude::IndexedRandom as _;
use rand::rngs::StdRng;
use rand::seq::SliceRandom as _;
use riscv_fuzz_plugin_api::coverage;
use riscv_fuzz_plugin_api::{
    AlignmentMode, ExecuteRequest, ISABase, IsaExtensions, StreamingCollector,
    flattened_sequence_instruction_count,
};
use riscv_fuzz_plugin_manager::PluginManager;
use riscv_fuzz_toolchain::{ArtifactIoError, write_json_pretty};
use riscv_instruction::{RiscvInstruction, generate_sequences_with_rng};
use riscv_instruction_types::{
    InstructionSequence, LoadImmediatePurpose, NumericOperandMutation, PseudoInstruction,
    RandomConfig, XLenImmediate,
};
use serde::{Deserialize, Serialize};

use crate::{
    GenerationError,
    impls::{
        base_memory_access_window, extensions_for_config, mem_range, random_config_for_window,
    },
    instruction::generate_initialization_instructions_for_impls,
    instruction_layout::{
        offset_mem_base_immediates, offset_structured_base_immediates, sequence_to_lines,
    },
    sequence_filter::remove_special_instructions,
    sequence_plan::{
        build_rv32_generation_counts, build_rv64_generation_counts, ordered_generation_plan,
    },
    smart_spice::{SpiceChunk, generate_smart_spice},
    testcase::TestCaseConfig,
};

/// Interesting values for mutation: boundary values and special cases that often
/// trigger edge conditions in CPU implementations.
const INTERESTING_I64: &[i64] = &[
    // Zero and small values
    0,
    1,
    -1,
    2,
    -2,
    // 8-bit boundaries
    0x7F,
    0x80,
    -0x80,
    0xFF,
    -0xFF,
    // 16-bit boundaries
    0x7FFF,
    0x8000,
    -0x8000,
    0xFFFF,
    -0xFFFF,
    // 32-bit boundaries
    0x7FFFFFFF,
    0x80000000_u32 as i64,
    -0x80000000,
    0xFFFFFFFF_u32 as i64,
    // 64-bit boundaries (for RV64)
    0x7FFFFFFFFFFFFFFF,
    -0x8000000000000000_i64,
    // Page boundaries (4KB)
    0x1000,
    -0x1000,
    0xFFF,
    -0xFFF,
    // Cache line boundaries (64B)
    0x40,
    -0x40,
    0x3F,
    -0x3F,
];

#[derive(Debug, Clone)]
pub struct CovGuidedSettings {
    pub testcase_config: TestCaseConfig,
    pub core_id: String,
    pub run_dir: PathBuf,
    pub max_runs: Option<u64>,
    pub coverage_mode: coverage::CoverageMode,
    /// Optional built-in coverage profile id (e.g. `line-branch`, `all`).
    ///
    /// Ignored when `coverage_types` is provided.
    pub coverage_profile: Option<String>,
    /// Optional explicit override for coverage types to guide on.
    /// If present, this takes precedence over `coverage_profile`.
    pub coverage_types: Option<Vec<String>>,
    pub seed: Option<u64>,
    pub mutation_steps: u32,
    pub fresh_probability: f64,
    pub spice_probability: f64,
    pub min_sequences: usize,
    pub max_sequences: usize,
    pub max_corpus: usize,
    pub stats_every: u64,
    pub keep_noninteresting: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
enum ProgramChunk {
    Seq(InstructionSequence<RiscvInstruction>),
    Structured(Vec<RiscvInstruction>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GuidedProgram {
    chunks: Vec<ProgramChunk>,
}

#[derive(Debug, Clone)]
struct CorpusEntry {
    id: u64,
    program: GuidedProgram,
    hit_hashes: HashSet<u64>,
}

#[derive(Debug, Serialize)]
struct RunMeta {
    run_id: u64,
    corpus_id: Option<u64>,
    parent_id: Option<u64>,
    new_hits: usize,
    total_hits: usize,
    elapsed_ms: u128,
    seed: u64,
    coverage_mode: coverage::CoverageMode,
    coverage_profile: Option<String>,
    coverage_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoverageHistoryEntry {
    run_id: u64,
    corpus_id: Option<u64>,
    parent_id: Option<u64>,
    new_hits: usize,
    total_hits: usize,
    seen_total: usize,
    corpus_size: usize,
    elapsed_ms: u128,
    timestamp_ms: u128,
    status: String,
    seed: u64,
    logical_instruction_count: usize,
    sequence_count: usize,
    coverage_mode: coverage::CoverageMode,
    coverage_profile: Option<String>,
    coverage_types: Vec<String>,
    interesting: bool,
}

const RANDOM_SANITIZED_SEQUENCE_MAX_ATTEMPTS: u32 = 50;

pub fn run_cov_guided(settings: CovGuidedSettings) -> Result<(), String> {
    if !settings.coverage_mode.is_enabled() {
        return Err(format!(
            "cov-guided requires --coverage-mode full|light (got {:?})",
            settings.coverage_mode
        ));
    }
    if !(0.0..=1.0).contains(&settings.fresh_probability) {
        return Err(format!(
            "fresh_probability must be in [0, 1], got {}",
            settings.fresh_probability
        ));
    }
    if !(0.0..=1.0).contains(&settings.spice_probability) {
        return Err(format!(
            "spice_probability must be in [0, 1], got {}",
            settings.spice_probability
        ));
    }
    if settings.mutation_steps == 0 {
        return Err("mutation_steps must be >= 1".to_string());
    }
    if settings.stats_every == 0 {
        return Err("stats_every must be >= 1".to_string());
    }
    if settings.min_sequences == 0 {
        return Err("min_sequences must be >= 1".to_string());
    }
    if settings.max_sequences == 0 {
        return Err("max_sequences must be >= 1".to_string());
    }
    if settings.min_sequences > settings.max_sequences {
        return Err(format!(
            "min_sequences ({}) must be <= max_sequences ({})",
            settings.min_sequences, settings.max_sequences
        ));
    }
    if settings.max_corpus == 0 {
        return Err("max_corpus must be >= 1".to_string());
    }

    let core_id = settings.core_id.clone();
    let plugin_manager = PluginManager::global();
    let isa_profile = plugin_manager.isa_profile(&core_id, settings.testcase_config.isa_base);

    if !isa_profile.supported {
        return Err(format!(
            "cov-guided impl {} does not support isa_base={}",
            core_id, settings.testcase_config.isa_base
        ));
    }
    if settings.testcase_config.is_dual_hart() {
        return Err("cov-guided currently supports only single-hart testcase_config".to_string());
    }

    let allow_unaligned = settings
        .testcase_config
        .unaligned_access_required
        .ok_or_else(|| {
            "testcase_config.unaligned_access_required must be explicitly set".to_string()
        })?;
    let alignment_mode = if allow_unaligned {
        AlignmentMode::AllowUnaligned
    } else {
        AlignmentMode::RequireAligned
    };
    if !isa_profile
        .unaligned_access_support
        .supports(alignment_mode)
    {
        return Err(format!(
            "cov-guided impl {} does not support allow_unaligned={}",
            core_id, allow_unaligned
        ));
    }

    ensure_cov_guided_mode_supported(
        core_id.clone(),
        settings.testcase_config.isa_base,
        settings.coverage_mode,
    )?;
    fs::create_dir_all(&settings.run_dir).map_err(|err| {
        format!(
            "failed to create run_dir {}: {err}",
            settings.run_dir.display()
        )
    })?;
    let queue_dir = settings.run_dir.join("queue");
    let crashes_dir = settings.run_dir.join("crashes");
    let work_dir = settings.run_dir.join("work");
    fs::create_dir_all(&queue_dir)
        .map_err(|err| format!("failed to create {}: {err}", queue_dir.display()))?;
    fs::create_dir_all(&crashes_dir)
        .map_err(|err| format!("failed to create {}: {err}", crashes_dir.display()))?;
    fs::create_dir_all(&work_dir)
        .map_err(|err| format!("failed to create {}: {err}", work_dir.display()))?;

    let seed = settings.seed.unwrap_or_else(now_seed);
    info!("cov-guided: seed={seed}");
    let mut rng = StdRng::seed_from_u64(seed);

    let impls = PluginManager::normalize_impls([core_id.clone()]);
    let extensions = extensions_for_config(&impls, &settings.testcase_config)
        .map_err(|err| format!("failed to compute extension map: {err}"))?;
    let mem_range_map = mem_range(
        &impls,
        settings.testcase_config.isa_base,
        settings.testcase_config.mem_size,
    )
    .map_err(|err| format!("failed to compute memory range: {err}"))?;
    let (mem_start, mem_end) = mem_range_map
        .get(&core_id)
        .copied()
        .ok_or_else(|| "memory range missing for implementation".to_string())?;
    let mem_offset = i64::try_from(mem_start)
        .map_err(|_| format!("memory start address out of range: {mem_start:#x}"))?;

    let mem_window = base_memory_access_window(&settings.testcase_config)
        .map_err(|err| format!("failed to compute memory access window: {err}"))?;
    let random_config = random_config_for_window(
        &settings.testcase_config,
        mem_window.register_min,
        mem_window.register_max,
    );

    // Fixed init block for this run (kept constant so that coverage deltas reflect mutations).
    let init_map = generate_initialization_instructions_for_impls(
        &impls,
        &extensions,
        settings.testcase_config.isa_base,
        &mem_range_map,
        &random_config,
    )
    .map_err(|err| format!("failed to generate init instructions: {err}"))?;
    let init_block = init_map
        .get(&core_id)
        .ok_or_else(|| "init instruction block missing for implementation".to_string())?;
    let init_sequences = init_block
        .harts
        .first()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(InstructionSequence::new)
        .collect::<Vec<_>>();

    let (coverage_types, resolved_coverage_profile) = resolve_cov_guided_coverage_types(
        settings.coverage_profile.as_deref(),
        settings.coverage_types.as_deref(),
    )?;
    let type_filter: Vec<&str> = normalize_type_filter(&coverage_types);

    let mut seen = load_seen_hashes(&settings.run_dir).unwrap_or_else(|err| {
        warn!("failed to load seen hashes; starting fresh: {err}");
        HashSet::new()
    });

    let impl_dir_name = core_id.to_string();
    let (mut corpus, mut next_corpus_id) =
        load_existing_corpus(&queue_dir, &impl_dir_name, &type_filter)
            .map_err(|err| format!("failed to load existing corpus: {err}"))?;

    let mut hash_freq: HashMap<u64, u32> = HashMap::new();
    for entry in &corpus {
        for &h in &entry.hit_hashes {
            *hash_freq.entry(h).or_insert(0) += 1;
        }
    }

    info!(
        "cov-guided: starting with corpus_size={} seen_hashes={}",
        corpus.len(),
        seen.len()
    );

    // Main loop. Resume numbering from the existing history file so repeated
    // invocations append a single monotonic timeline instead of restarting from zero.
    let mut run_id: u64 = load_next_history_run_id(&settings.run_dir).unwrap_or_else(|err| {
        warn!("failed to load next run id; restarting from zero: {err}");
        0
    });
    let mut last_stats = Instant::now();
    let mut since_last_new: u64 = 0;
    loop {
        if let Some(limit) = settings.max_runs
            && run_id >= limit
        {
            break;
        }

        if corpus.len() >= settings.max_corpus {
            // Keep running, but only attempt to find additional coverage (no corpus growth).
            // This avoids unbounded disk growth during long sessions.
            if run_id.is_multiple_of(settings.stats_every) {
                info!(
                    "cov-guided: corpus maxed ({}), continuing without adding new entries",
                    settings.max_corpus
                );
            }
        }

        // Simple stagnation handling: when we haven't found anything new for a while,
        // bias toward fresh generation and more "spice" to shake the state-space.
        let mut effective_fresh = settings.fresh_probability;
        let mut effective_spice = settings.spice_probability;
        let mut effective_mutation_steps = settings.mutation_steps;
        if since_last_new >= 200 {
            effective_fresh = effective_fresh.max(0.20);
            effective_spice = effective_spice.max(0.25);
            effective_mutation_steps = effective_mutation_steps.max(6);
        }
        let generation_params = GuidedGenerationParams {
            testcase_config: &settings.testcase_config,
            extensions: &extensions,
            random_config: &random_config,
            mem_start,
            mem_end,
            min_sequences: settings.min_sequences,
            max_sequences: settings.max_sequences,
            spice_probability: effective_spice,
        };

        let pick_fresh = corpus.is_empty() || rng.random::<f64>() < effective_fresh;
        let (parent_id, program_result) = if pick_fresh {
            (
                None,
                generate_fresh_program(&mut rng, &generation_params)
                    .map_err(|err| format!("failed to generate random program: {err}")),
            )
        } else {
            let parent = choose_parent_by_rarity(&mut rng, &corpus, &hash_freq);
            (
                Some(parent.id),
                mutate_program(
                    &mut rng,
                    &parent.program,
                    &corpus,
                    &generation_params,
                    effective_mutation_steps,
                )
                .map_err(|err| format!("failed to mutate corpus program id_{}: {err}", parent.id)),
            )
        };
        let program = match program_result {
            Ok(program) => program,
            Err(err) => {
                warn!("run {run_id} failed before execution: {err}");
                since_last_new = since_last_new.saturating_add(1);

                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_else(|_| Duration::from_secs(0))
                    .as_millis();
                let history_entry = CoverageHistoryEntry {
                    run_id,
                    corpus_id: None,
                    parent_id,
                    new_hits: 0,
                    total_hits: 0,
                    seen_total: seen.len(),
                    corpus_size: corpus.len(),
                    elapsed_ms: 0,
                    timestamp_ms,
                    status: "generation_fail".to_string(),
                    seed,
                    logical_instruction_count: 0,
                    sequence_count: 0,
                    coverage_mode: settings.coverage_mode,
                    coverage_profile: resolved_coverage_profile.clone(),
                    coverage_types: coverage_types.clone(),
                    interesting: false,
                };
                if let Err(append_err) = append_coverage_history(&settings.run_dir, &history_entry)
                {
                    warn!("failed to append coverage history: {append_err}");
                }
                run_id = run_id.saturating_add(1);
                continue;
            }
        };

        let work_run_dir = work_dir.join(format!("run_{run_id:08}"));
        let impl_dir = work_run_dir.join(&impl_dir_name);
        if let Err(err) = fs::create_dir_all(&impl_dir) {
            return Err(format!(
                "failed to create work run dir {}: {err}",
                impl_dir.display()
            ));
        }

        let typed_program = materialize_program_typed(&init_sequences, &program, mem_offset);
        let instructions = render_program_lines(&typed_program);
        let logical_instruction_count = flattened_sequence_instruction_count(&typed_program);
        let sequence_count = program.chunks.len();
        if let Err(err) = write_json_pretty(work_run_dir.join("program.json"), &program) {
            warn!("failed to write program.json: {err}");
        }
        if let Err(err) = fs::write(
            work_run_dir.join("input_instructions.asm"),
            instructions.join("\n") + "\n",
        ) {
            warn!("failed to write input_instructions.asm: {err}");
        }

        let start = Instant::now();
        // cov_guided programs are single-hart (hart 0) — materialize_program produces a flat list.
        let hart_inputs = vec![(0u32, typed_program.as_slice())];
        let mut collector = StreamingCollector::new();
        let run_result: Result<(), String> = plugin_manager
            .execute_streaming_with_request(
                &core_id,
                ExecuteRequest {
                    run_root: &impl_dir,
                    isa_base: settings.testcase_config.isa_base,
                    hart_inputs: &hart_inputs,
                    extension_override: Some(&extensions),
                    float_mode_override: None,
                    alignment_mode,
                    coverage_mode: settings.coverage_mode,
                },
                &mut |event| collector.push(event),
            )
            .map_err(|err| err.to_string())
            .and_then(|()| {
                collector
                    .finish()
                    .map(|_| ())
                    .map_err(|err| err.to_string())
            });
        let elapsed_ms = start.elapsed().as_millis();

        match run_result {
            Ok(()) => {
                let hits = if settings.coverage_mode.is_enabled() {
                    match load_run_hit_hashes(&impl_dir, &type_filter) {
                        Ok(hits) => hits,
                        Err(err) => {
                            warn!("run {run_id} failed to load coverage: {err}");
                            let dst = crashes_dir.join(format!("run_{run_id:08}"));
                            if let Err(rename_err) = fs::rename(&work_run_dir, &dst) {
                                warn!(
                                    "failed to move crash dir {} -> {}: {rename_err}",
                                    work_run_dir.display(),
                                    dst.display()
                                );
                            }
                            return Err(format!("run {run_id} failed to load coverage: {err}"));
                        }
                    }
                } else {
                    HashSet::new()
                };

                let new_hits = hits.difference(&seen).count();
                let total_hits = hits.len();

                let mut meta = RunMeta {
                    run_id,
                    corpus_id: None,
                    parent_id,
                    new_hits,
                    total_hits,
                    elapsed_ms,
                    seed,
                    coverage_mode: settings.coverage_mode,
                    coverage_profile: resolved_coverage_profile.clone(),
                    coverage_types: coverage_types.clone(),
                };

                let keep_dir = new_hits > 0 && corpus.len() < settings.max_corpus;
                if keep_dir {
                    let corpus_id = next_corpus_id;
                    next_corpus_id += 1;
                    meta.corpus_id = Some(corpus_id);

                    seen.extend(hits.iter().copied());
                    persist_seen_hashes(&settings.run_dir, &seen).unwrap_or_else(|err| {
                        warn!("failed to persist seen hashes: {err}");
                    });

                    let dst = queue_dir.join(format!(
                        "id_{corpus_id:08}_run_{run_id:08}_new_{new_hits:04}"
                    ));
                    if let Err(err) = fs::rename(&work_run_dir, &dst) {
                        warn!(
                            "failed to move interesting run dir {} -> {}: {err}",
                            work_run_dir.display(),
                            dst.display()
                        );
                    } else if let Err(err) = write_json_pretty(dst.join("meta.json"), &meta) {
                        warn!("failed to write meta.json: {err}");
                    }

                    for &h in &hits {
                        *hash_freq.entry(h).or_insert(0) += 1;
                    }
                    corpus.push(CorpusEntry {
                        id: corpus_id,
                        program,
                        hit_hashes: hits,
                    });

                    info!(
                        "cov-guided: +new_coverage run={} corpus_id={} new_hits={} total_hits={} corpus={} seen={}",
                        run_id,
                        corpus_id,
                        new_hits,
                        total_hits,
                        corpus.len(),
                        seen.len()
                    );
                    since_last_new = 0;
                } else if settings.keep_noninteresting {
                    let dst = work_run_dir.join("meta.json");
                    if let Err(err) = write_json_pretty(dst, &meta) {
                        warn!("failed to write meta.json: {err}");
                    }
                    since_last_new = since_last_new.saturating_add(1);
                } else if let Err(err) = fs::remove_dir_all(&work_run_dir) {
                    warn!(
                        "failed to delete non-interesting run dir {}: {err}",
                        work_run_dir.display()
                    );
                    since_last_new = since_last_new.saturating_add(1);
                } else {
                    since_last_new = since_last_new.saturating_add(1);
                }

                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_else(|_| Duration::from_secs(0))
                    .as_millis();
                let history_entry = CoverageHistoryEntry {
                    run_id,
                    corpus_id: meta.corpus_id,
                    parent_id,
                    new_hits,
                    total_hits,
                    seen_total: seen.len(),
                    corpus_size: corpus.len(),
                    elapsed_ms,
                    timestamp_ms,
                    status: "ok".to_string(),
                    seed,
                    logical_instruction_count,
                    sequence_count,
                    coverage_mode: settings.coverage_mode,
                    coverage_profile: resolved_coverage_profile.clone(),
                    coverage_types: coverage_types.clone(),
                    interesting: meta.corpus_id.is_some(),
                };
                if let Err(err) = append_coverage_history(&settings.run_dir, &history_entry) {
                    warn!("failed to append coverage history: {err}");
                }
            }
            Err(err) => {
                warn!("run {run_id} failed: {err}");
                // Keep failing programs for inspection.
                let dst = crashes_dir.join(format!("run_{run_id:08}"));
                if let Err(rename_err) = fs::rename(&work_run_dir, &dst) {
                    warn!(
                        "failed to move crash dir {} -> {}: {rename_err}",
                        work_run_dir.display(),
                        dst.display()
                    );
                }
                since_last_new = since_last_new.saturating_add(1);

                let timestamp_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_else(|_| Duration::from_secs(0))
                    .as_millis();
                let history_entry = CoverageHistoryEntry {
                    run_id,
                    corpus_id: None,
                    parent_id,
                    new_hits: 0,
                    total_hits: 0,
                    seen_total: seen.len(),
                    corpus_size: corpus.len(),
                    elapsed_ms: start.elapsed().as_millis(),
                    timestamp_ms,
                    status: "fail".to_string(),
                    seed,
                    logical_instruction_count,
                    sequence_count,
                    coverage_mode: settings.coverage_mode,
                    coverage_profile: resolved_coverage_profile.clone(),
                    coverage_types: coverage_types.clone(),
                    interesting: false,
                };
                if let Err(append_err) = append_coverage_history(&settings.run_dir, &history_entry)
                {
                    warn!("failed to append coverage history: {append_err}");
                }
            }
        }

        if run_id.is_multiple_of(settings.stats_every) {
            let dt = last_stats.elapsed();
            last_stats = Instant::now();
            info!(
                "cov-guided: runs={} corpus={} seen={} dt={:.3}s",
                run_id,
                corpus.len(),
                seen.len(),
                dt.as_secs_f64()
            );

            if run_id.is_multiple_of(settings.stats_every * 10)
                && let Err(err) = plot_coverage_growth(&settings.run_dir)
            {
                warn!("failed to plot coverage growth: {err}");
            }
        }

        run_id = run_id.saturating_add(1);
    }

    persist_seen_hashes(&settings.run_dir, &seen)
        .map_err(|err| format!("failed to persist seen hashes: {err}"))?;

    if let Err(err) = plot_coverage_growth(&settings.run_dir) {
        warn!("failed to plot final coverage growth: {err}");
    }

    info!(
        "cov-guided finished: runs={} corpus={} seen={}",
        run_id,
        corpus.len(),
        seen.len()
    );
    Ok(())
}

fn ensure_cov_guided_mode_supported(
    core_id: String,
    isa_base: ISABase,
    coverage_mode: coverage::CoverageMode,
) -> Result<(), String> {
    let support = PluginManager::global()
        .isa_profile(&core_id, isa_base)
        .coverage_mode_support;
    if !support.supports(coverage_mode) {
        return Err(format!(
            "cov-guided coverage mode {:?} is not supported by implementation {} for ISA {} (supported: off={}, light={}, full={})",
            coverage_mode, core_id, isa_base, support.off, support.light, support.full
        ));
    }

    Ok(())
}

fn now_seed() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    now.as_nanos() as u64
}

fn is_all_coverage_types(types: &[String]) -> bool {
    types.len() == 1
        && (types[0].eq_ignore_ascii_case("all") || types[0].trim().eq_ignore_ascii_case("*"))
}

fn normalize_coverage_types(types: &[String]) -> Vec<String> {
    types
        .iter()
        .map(|raw| raw.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn coverage_types_for_profile(profile: &str) -> Option<Vec<String>> {
    match profile {
        "line" => Some(vec!["line".to_string()]),
        "branch" => Some(vec!["branch".to_string()]),
        "toggle" => Some(vec!["toggle".to_string()]),
        "line-branch" | "branch-line" => Some(vec!["line".to_string(), "branch".to_string()]),
        "all" | "*" => Some(vec!["all".to_string()]),
        _ => None,
    }
}

fn resolve_cov_guided_coverage_types(
    coverage_profile: Option<&str>,
    explicit_coverage_types: Option<&[String]>,
) -> Result<(Vec<String>, Option<String>), String> {
    if let Some(explicit) = explicit_coverage_types {
        let mut normalized = normalize_coverage_types(explicit);
        if normalized.is_empty() {
            return Err("cov-guided coverage types cannot be empty".to_string());
        }

        if is_all_coverage_types(&normalized) {
            return Ok((vec!["all".to_string()], None));
        }

        normalized.sort();
        normalized.dedup();
        return Ok((normalized, None));
    }

    let requested_profile_id = if let Some(profile) = coverage_profile {
        let normalized = profile.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Err("coverage profile id cannot be empty".to_string());
        }
        normalized
    } else {
        "line-branch".to_string()
    };

    let mut normalized = coverage_types_for_profile(&requested_profile_id).ok_or_else(|| {
        format!(
            "unknown cov-guided coverage profile `{}` (available: [line, branch, toggle, line-branch, all])",
            requested_profile_id
        )
    })?;
    if normalized.is_empty() {
        return Err(format!(
            "invalid cov-guided coverage profile `{}`: coverage_types cannot be empty",
            requested_profile_id
        ));
    }

    if is_all_coverage_types(&normalized) {
        return Ok((vec!["all".to_string()], Some(requested_profile_id)));
    }

    normalized.sort();
    normalized.dedup();
    Ok((normalized, Some(requested_profile_id)))
}

fn normalize_type_filter(types: &[String]) -> Vec<&str> {
    let mut filter: Vec<&str> = types.iter().map(|s| s.as_str()).collect();
    filter.retain(|value| !value.trim().is_empty());
    if filter.len() == 1 && (filter[0].eq_ignore_ascii_case("all") || filter[0] == "*") {
        filter.clear();
    }
    filter
}

fn generate_random_sequences<R: rand::Rng>(
    rng: &mut R,
    testcase_config: &TestCaseConfig,
    extensions: &IsaExtensions,
    random_config: &RandomConfig,
) -> Result<Vec<InstructionSequence<RiscvInstruction>>, GenerationError> {
    match testcase_config.isa_base {
        ISABase::Rv32 => {
            let counts = build_rv32_generation_counts(
                extensions
                    .as_rv32()
                    .expect("RV32 extensions expected for RV32 testcase"),
                testcase_config,
            )?;
            let insts =
                generate_sequences_with_rng(rng, &ordered_generation_plan(&counts), random_config)?;
            Ok(remove_special_instructions(insts))
        }
        ISABase::Rv64 => {
            let counts = build_rv64_generation_counts(
                extensions
                    .as_rv64()
                    .expect("RV64 extensions expected for RV64 testcase"),
                testcase_config,
            )?;
            let insts =
                generate_sequences_with_rng(rng, &ordered_generation_plan(&counts), random_config)?;
            Ok(remove_special_instructions(insts))
        }
    }
}

/// Generate a smart spice chunk using the new extension-aware system
fn generate_spice<R: rand::Rng>(
    rng: &mut R,
    extensions: &IsaExtensions,
    isa_base: ISABase,
    random_config: &RandomConfig,
    mem_start: u64,
    mem_end: u64,
) -> Result<ProgramChunk, String> {
    let chunk = generate_smart_spice(rng, extensions, isa_base, random_config, mem_start, mem_end)
        .ok_or_else(|| {
            format!("smart spice generation returned no chunk for isa_base={isa_base:?}")
        })?;

    match chunk {
        SpiceChunk::Seq(seq) => {
            let filtered = remove_special_instructions(vec![seq]);
            if let Some(seq) = filtered.into_iter().next() {
                Ok(ProgramChunk::Seq(seq))
            } else {
                Err(
                    "smart spice sequence was fully removed by special-instruction filtering"
                        .to_string(),
                )
            }
        }
        SpiceChunk::Structured(entries) => Ok(ProgramChunk::Structured(entries)),
    }
}

fn random_chunk<R: rand::Rng>(
    rng: &mut R,
    testcase_config: &TestCaseConfig,
    extensions: &IsaExtensions,
    random_config: &RandomConfig,
    mem_start: u64,
    mem_end: u64,
    spice_probability: f64,
) -> Result<ProgramChunk, String> {
    if rng.random::<f64>() < spice_probability {
        return generate_spice(
            rng,
            extensions,
            testcase_config.isa_base,
            random_config,
            mem_start,
            mem_end,
        );
    }

    random_sanitized_sequence(rng, testcase_config, extensions, random_config)
        .map(ProgramChunk::Seq)
}

struct GuidedGenerationParams<'a> {
    testcase_config: &'a TestCaseConfig,
    extensions: &'a IsaExtensions,
    random_config: &'a RandomConfig,
    mem_start: u64,
    mem_end: u64,
    min_sequences: usize,
    max_sequences: usize,
    spice_probability: f64,
}

fn random_chunk_from_params<R: rand::Rng>(
    rng: &mut R,
    params: &GuidedGenerationParams<'_>,
) -> Result<ProgramChunk, String> {
    random_chunk(
        rng,
        params.testcase_config,
        params.extensions,
        params.random_config,
        params.mem_start,
        params.mem_end,
        params.spice_probability,
    )
}

fn generate_fresh_program<R: rand::Rng>(
    rng: &mut R,
    params: &GuidedGenerationParams<'_>,
) -> Result<GuidedProgram, String> {
    let target_len = if params.min_sequences == params.max_sequences {
        params.min_sequences
    } else {
        rng.random_range(params.min_sequences..=params.max_sequences)
    };

    // Start with a batch generated via the existing extension-aware generator, then top up with
    // per-sequence generation as needed. This keeps instruction diversity but lets us control
    // final program size.
    let mut reservoir = generate_random_sequences(
        rng,
        params.testcase_config,
        params.extensions,
        params.random_config,
    )
    .map_err(|err| format!("failed to generate sequence reservoir: {err}"))?;
    reservoir.shuffle(rng);
    let mut reservoir_iter = reservoir.into_iter();

    let mut chunks = Vec::with_capacity(target_len);
    while chunks.len() < target_len {
        if rng.random::<f64>() < params.spice_probability {
            chunks.push(generate_spice(
                rng,
                params.extensions,
                params.testcase_config.isa_base,
                params.random_config,
                params.mem_start,
                params.mem_end,
            )?);
            continue;
        }

        if let Some(seq) = reservoir_iter.next() {
            chunks.push(ProgramChunk::Seq(seq));
            continue;
        }

        let chunk = random_chunk_from_params(rng, params)?;
        chunks.push(chunk);
    }

    if chunks.is_empty() {
        return Err("fresh program generation produced zero chunks".to_string());
    }

    Ok(GuidedProgram { chunks })
}

fn random_sanitized_sequence<R: rand::Rng>(
    rng: &mut R,
    testcase_config: &TestCaseConfig,
    extensions: &IsaExtensions,
    random_config: &RandomConfig,
) -> Result<InstructionSequence<RiscvInstruction>, String> {
    let mut attempts = 0u32;
    let mut last_error = "no generation attempts performed".to_string();
    while attempts < RANDOM_SANITIZED_SEQUENCE_MAX_ATTEMPTS {
        attempts += 1;

        let seq = match testcase_config.isa_base {
            ISABase::Rv32 => {
                let ext = extensions
                    .as_rv32()
                    .expect("RV32 extensions expected for RV32 testcase")
                    .choose(rng)
                    .ok_or_else(|| {
                        "no RV32 extensions available for sanitized sequence generation".to_string()
                    })?;
                match ext.random_sequence_with_rng(rng, random_config) {
                    Ok(seq) => seq,
                    Err(err) => {
                        last_error = format!(
                            "attempt {attempts}: failed to generate RV32 sequence for {:?}: {err}",
                            ext
                        );
                        continue;
                    }
                }
            }
            ISABase::Rv64 => {
                let ext = extensions
                    .as_rv64()
                    .expect("RV64 extensions expected for RV64 testcase")
                    .choose(rng)
                    .ok_or_else(|| {
                        "no RV64 extensions available for sanitized sequence generation".to_string()
                    })?;
                match ext.random_sequence_with_rng(rng, random_config) {
                    Ok(seq) => seq,
                    Err(err) => {
                        last_error = format!(
                            "attempt {attempts}: failed to generate RV64 sequence for {:?}: {err}",
                            ext
                        );
                        continue;
                    }
                }
            }
        };

        let filtered = remove_special_instructions(vec![seq]);
        if let Some(seq) = filtered.into_iter().next() {
            return Ok(seq);
        }
        last_error = format!(
            "attempt {attempts}: generated sequence was removed by special-instruction filtering"
        );
    }
    Err(format!(
        "failed to generate a sanitized sequence after {} attempts: {}",
        RANDOM_SANITIZED_SEQUENCE_MAX_ATTEMPTS, last_error
    ))
}

/// Mutate the immediate values in a sequence's pre_instructions.
/// This allows fine-grained exploration of value space without changing instruction types.
fn rebuild_xlen_immediate_like(original: XLenImmediate, value: i64) -> Option<XLenImmediate> {
    match original {
        XLenImmediate::X32(_) => i32::try_from(value).ok().map(XLenImmediate::X32),
        XLenImmediate::X64(_) => Some(XLenImmediate::X64(value)),
    }
}

fn mutate_immediates<R: rand::Rng>(rng: &mut R, seq: &mut InstructionSequence<RiscvInstruction>) {
    if seq.pre_instructions.is_empty() {
        return;
    }

    let idx = rng.random_range(0..seq.pre_instructions.len());
    if let RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
        immediate, purpose, ..
    }) = &mut seq.pre_instructions[idx]
    {
        if *purpose == LoadImmediatePurpose::BaseAddress {
            return;
        }

        let current = immediate.as_i64();
        let strategy = rng.random_range(0..5);
        let mutated = match strategy {
            0 => current.saturating_add(1),
            1 => current.saturating_sub(1),
            2 => current.wrapping_neg(),
            3 => {
                let bit = rng.random_range(0..64);
                current ^ (1i64 << bit)
            }
            _ => {
                let delta = rng.random_range(-16..=16);
                current.saturating_add(delta)
            }
        };
        if let Some(next) = rebuild_xlen_immediate_like(*immediate, mutated) {
            *immediate = next;
        }
    }
}

/// Apply an interesting boundary value to a random immediate in the sequence.
/// Interesting values are known to trigger edge cases in CPU implementations.
fn apply_interesting_value<R: rand::Rng>(
    rng: &mut R,
    seq: &mut InstructionSequence<RiscvInstruction>,
) {
    if seq.pre_instructions.is_empty() || INTERESTING_I64.is_empty() {
        return;
    }

    let idx = rng.random_range(0..seq.pre_instructions.len());
    if let RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
        immediate, purpose, ..
    }) = &mut seq.pre_instructions[idx]
    {
        if *purpose == LoadImmediatePurpose::BaseAddress {
            return;
        }

        let interesting_val = INTERESTING_I64[rng.random_range(0..INTERESTING_I64.len())];
        if let Some(next) = rebuild_xlen_immediate_like(*immediate, interesting_val) {
            *immediate = next;
        }
    }
}

fn mutate_instruction_operands<R: rand::Rng>(
    rng: &mut R,
    seq: &mut InstructionSequence<RiscvInstruction>,
    config: &RandomConfig,
) {
    let _ = seq.instruction.mutate_numeric_operands(rng, config);
}

/// Regenerate an instruction from the same extension family but with potentially
/// different operands. This keeps the instruction class but explores the parameter space.
fn regenerate_from_same_extension<R: rand::Rng>(
    rng: &mut R,
    chunk: &ProgramChunk,
    testcase_config: &TestCaseConfig,
    extensions: &IsaExtensions,
    random_config: &RandomConfig,
) -> Option<InstructionSequence<RiscvInstruction>> {
    // Only works on instruction-sequence chunks.
    if !matches!(chunk, ProgramChunk::Seq(_)) {
        return None;
    }

    // Generate a new random instruction sequence
    // Note: We can't easily determine which specific extension the original instruction
    // came from without complex reverse mapping, so we just generate a new random one
    // from any extension. This still provides diversity within instruction types.
    match testcase_config.isa_base {
        ISABase::Rv32 => {
            let rv32_extensions = extensions
                .as_rv32()
                .expect("RV32 extensions expected for RV32 testcase");
            if rv32_extensions.is_empty() {
                return None;
            }
            let ext = rv32_extensions.choose(rng)?;
            ext.random_sequence_with_rng(rng, random_config).ok()
        }
        ISABase::Rv64 => {
            let rv64_extensions = extensions
                .as_rv64()
                .expect("RV64 extensions expected for RV64 testcase");
            if rv64_extensions.is_empty() {
                return None;
            }
            let ext = rv64_extensions.choose(rng)?;
            ext.random_sequence_with_rng(rng, random_config).ok()
        }
    }
    .and_then(|seq| {
        // Filter to ensure we don't generate problematic instructions
        let filtered = remove_special_instructions(vec![seq]);
        filtered.into_iter().next()
    })
}

fn mutate_program<R: rand::Rng>(
    rng: &mut R,
    parent: &GuidedProgram,
    corpus: &[CorpusEntry],
    params: &GuidedGenerationParams<'_>,
    steps: u32,
) -> Result<GuidedProgram, String> {
    let mut child = parent.clone();
    if child.chunks.is_empty() {
        let chunk = random_chunk_from_params(rng, params)?;
        child.chunks.push(chunk);
        return Ok(child);
    }

    for _ in 0..steps {
        let choice = rng.random_range(0u32..100);
        match choice {
            // Replace (reduced from 45% to 25%)
            0..=24 => {
                let idx = rng.random_range(0..child.chunks.len());
                let chunk = random_chunk_from_params(rng, params)?;
                child.chunks[idx] = chunk;
            }
            // Insert (reduced from 30% to 20%)
            25..=44 => {
                if child.chunks.len() >= params.max_sequences {
                    continue;
                }
                let idx = rng.random_range(0..=child.chunks.len());
                let chunk = random_chunk_from_params(rng, params)?;
                child.chunks.insert(idx, chunk);
            }
            // Delete (reduced from 15% to 10%)
            45..=54 => {
                if child.chunks.len() <= 1 {
                    continue;
                }
                let idx = rng.random_range(0..child.chunks.len());
                child.chunks.remove(idx);
            }
            // Swap (same 5%)
            55..=59 => {
                if child.chunks.len() <= 1 {
                    continue;
                }
                let a = rng.random_range(0..child.chunks.len());
                let mut b = rng.random_range(0..child.chunks.len());
                if a == b {
                    b = (b + 1) % child.chunks.len();
                }
                child.chunks.swap(a, b);
            }
            // Mutate immediates: modify LoadImmediate values in pre_instructions (10%)
            60..=69 => {
                let idx = rng.random_range(0..child.chunks.len());
                if let ProgramChunk::Seq(seq) = &mut child.chunks[idx] {
                    mutate_immediates(rng, seq);
                }
            }
            // Mutate instruction operands: change numeric operands in the main instruction (10%)
            70..=79 => {
                let idx = rng.random_range(0..child.chunks.len());
                if let ProgramChunk::Seq(seq) = &mut child.chunks[idx] {
                    mutate_instruction_operands(rng, seq, params.random_config);
                }
            }
            // Apply interesting value: use boundary/special values (10%)
            80..=89 => {
                let idx = rng.random_range(0..child.chunks.len());
                if let ProgramChunk::Seq(seq) = &mut child.chunks[idx] {
                    apply_interesting_value(rng, seq);
                }
            }
            // Regenerate from same extension: keep extension but new instruction (5%)
            90..=94 => {
                let idx = rng.random_range(0..child.chunks.len());
                if let Some(new_seq) = regenerate_from_same_extension(
                    rng,
                    &child.chunks[idx],
                    params.testcase_config,
                    params.extensions,
                    params.random_config,
                ) {
                    child.chunks[idx] = ProgramChunk::Seq(new_seq);
                }
            }
            // Splice from another corpus entry (5%)
            _ => {
                if corpus.len() <= 1 {
                    continue;
                }
                let Some(donor) = corpus.choose(rng).map(|e| &e.program) else {
                    continue;
                };
                if donor.chunks.is_empty() {
                    continue;
                }
                let start = rng.random_range(0..donor.chunks.len());
                let end = rng.random_range(start + 1..=donor.chunks.len());
                let slice = &donor.chunks[start..end];
                if slice.is_empty() {
                    continue;
                }
                let mut insert_at = rng.random_range(0..=child.chunks.len());
                for chunk in slice.iter().cloned() {
                    if child.chunks.len() >= params.max_sequences {
                        break;
                    }
                    child.chunks.insert(insert_at, chunk);
                    insert_at = insert_at.saturating_add(1);
                }
            }
        }
    }

    if child.chunks.len() > params.max_sequences {
        child.chunks.truncate(params.max_sequences);
    }

    // Ensure the program does not shrink too much; longer tests help exercise multi-cycle and
    // slow-path logic in the core, which tends to unlock additional Verilator line/branch points.
    while child.chunks.len() < params.min_sequences {
        if child.chunks.len() >= params.max_sequences {
            break;
        }
        let chunk = random_chunk_from_params(rng, params)?;
        child.chunks.push(chunk);
    }
    Ok(child)
}

fn materialize_program_typed(
    init_sequences: &[InstructionSequence<RiscvInstruction>],
    program: &GuidedProgram,
    mem_offset: i64,
) -> Vec<InstructionSequence<RiscvInstruction>> {
    let mut sequences = init_sequences.to_vec();

    for chunk in &program.chunks {
        match chunk {
            ProgramChunk::Seq(seq) => {
                let mut seq = seq.clone();
                offset_mem_base_immediates(&mut seq, mem_offset);
                sequences.push(seq);
            }
            ProgramChunk::Structured(entries) => {
                let mut entries = entries.clone();
                offset_structured_base_immediates(&mut entries, mem_offset);
                sequences.extend(entries.into_iter().map(InstructionSequence::new));
            }
        }
    }
    sequences
}

fn render_program_lines(program: &[InstructionSequence<RiscvInstruction>]) -> Vec<String> {
    program.iter().flat_map(sequence_to_lines).collect()
}

fn load_seen_hashes(run_dir: &Path) -> io::Result<HashSet<u64>> {
    let path = run_dir.join("seen_hashes.json");
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let file = fs::File::open(path)?;
    let values: Vec<u64> = serde_json::from_reader(file)?;
    Ok(values.into_iter().collect())
}

fn persist_seen_hashes(run_dir: &Path, seen: &HashSet<u64>) -> io::Result<()> {
    let path = run_dir.join("seen_hashes.json");
    let mut values: Vec<u64> = seen.iter().copied().collect();
    values.sort_unstable();
    write_json_pretty(path, &values).map_err(|err| match err {
        ArtifactIoError::Io(source) => source,
        ArtifactIoError::Serialization(source) => io::Error::other(source),
    })
}

fn load_existing_corpus(
    queue_dir: &Path,
    impl_dir_name: &str,
    type_filter: &[&str],
) -> io::Result<(Vec<CorpusEntry>, u64)> {
    if !queue_dir.exists() {
        return Ok((Vec::new(), 0));
    }

    let mut entries: Vec<(u64, PathBuf)> = Vec::new();
    for item in fs::read_dir(queue_dir)? {
        let item = item?;
        let path = item.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(id) = parse_queue_id(name) else {
            continue;
        };
        entries.push((id, path));
    }
    entries.sort_by_key(|(id, _)| *id);

    let mut corpus = Vec::new();
    let mut max_id: Option<u64> = None;
    for (id, dir) in entries {
        max_id = Some(max_id.map_or(id, |cur| cur.max(id)));
        let program_path = dir.join("program.json");
        if !program_path.exists() {
            continue;
        }
        let file = fs::File::open(program_path)?;
        let program: GuidedProgram = serde_json::from_reader(file)?;

        let hit_hashes =
            load_run_hit_hashes(&dir.join(impl_dir_name), type_filter).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "failed to load coverage for existing corpus entry id_{id:08} ({}): {err}",
                        dir.display()
                    ),
                )
            })?;

        corpus.push(CorpusEntry {
            id,
            program,
            hit_hashes,
        });
    }

    let next_id = max_id.map_or(0, |id| id.saturating_add(1));
    Ok((corpus, next_id))
}

fn load_run_hit_hashes(impl_run_dir: &Path, type_filter: &[&str]) -> Result<HashSet<u64>, String> {
    let verilator_dat_path = impl_run_dir.join("coverage.dat");
    if verilator_dat_path.is_file() {
        return riscv_fuzz_toolchain::coverage::parse_verilator_hit_hashes(
            &verilator_dat_path,
            type_filter,
        )
        .map_err(|err| {
            format!(
                "failed to parse Verilator coverage from {}: {err}",
                verilator_dat_path.display()
            )
        });
    }

    let spike_log_path = impl_run_dir.join("spike_stderr.log");
    if spike_log_path.is_file() {
        let hits = riscv_fuzz_toolchain::coverage::parse_spike_hit_hashes(&spike_log_path)
            .map_err(|err| {
                format!(
                    "failed to parse Spike commit-log coverage from {}: {err}",
                    spike_log_path.display()
                )
            })?;
        if hits.is_empty() {
            return Err(format!(
                "failed to parse Spike commit-log coverage: no hits found in {}",
                spike_log_path.display()
            ));
        }
        return Ok(hits);
    }

    let rocket_log_path = impl_run_dir.join("rocket_stderr.log");
    if rocket_log_path.is_file() {
        let hits = riscv_fuzz_toolchain::coverage::parse_rocket_hit_hashes(&rocket_log_path)
            .map_err(|err| {
                format!(
                    "failed to parse Rocket commit-log coverage from {}: {err}",
                    rocket_log_path.display()
                )
            })?;
        if hits.is_empty() {
            return Err(format!(
                "failed to parse Rocket commit-log coverage: no hits found in {}",
                rocket_log_path.display()
            ));
        }
        return Ok(hits);
    }

    Err(format!(
        "failed to parse coverage in {}: no supported artifact found (expected coverage.dat, spike_stderr.log, or rocket_stderr.log)",
        impl_run_dir.display()
    ))
}

fn parse_queue_id(name: &str) -> Option<u64> {
    // Expected format: id_00000042_run_... (or similar)
    let rest = name.strip_prefix("id_")?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

fn rarity_score(entry: &CorpusEntry, hash_freq: &HashMap<u64, u32>) -> f64 {
    if entry.hit_hashes.is_empty() {
        return 0.0;
    }

    entry
        .hit_hashes
        .iter()
        .map(|h| {
            let freq = hash_freq.get(h).copied().unwrap_or(1).max(1);
            1.0 / freq as f64
        })
        .sum()
}

fn choose_parent_by_rarity<'a, R: rand::Rng>(
    rng: &mut R,
    corpus: &'a [CorpusEntry],
    hash_freq: &HashMap<u64, u32>,
) -> &'a CorpusEntry {
    debug_assert!(!corpus.is_empty(), "caller should handle empty corpus");

    // Tournament selection over a small random subset to bias toward rare coverage without
    // having to recompute weights for the whole corpus on every iteration.
    let tournament = 8usize.min(corpus.len());

    let mut best = corpus.choose(rng).expect("corpus non-empty");
    let mut best_score = rarity_score(best, hash_freq);

    for _ in 1..tournament {
        let cand = corpus.choose(rng).expect("corpus non-empty");
        let cand_score = rarity_score(cand, hash_freq);
        if cand_score > best_score {
            best = cand;
            best_score = cand_score;
        }
    }

    best
}

fn append_coverage_history(run_dir: &Path, entry: &CoverageHistoryEntry) -> io::Result<()> {
    let path = run_dir.join("coverage_history.jsonl");
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, entry)?;
    writeln!(file)?;
    Ok(())
}

fn load_next_history_run_id(run_dir: &Path) -> io::Result<u64> {
    let path = run_dir.join("coverage_history.jsonl");
    if !path.exists() {
        return Ok(0);
    }

    let mut next_run_id = 0u64;
    let content = fs::read_to_string(path)?;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        let Some(run_id) = value.get("run_id").and_then(|value| value.as_u64()) else {
            continue;
        };
        next_run_id = next_run_id.max(run_id.saturating_add(1));
    }

    Ok(next_run_id)
}

fn plot_coverage_growth(run_dir: &Path) -> Result<(), String> {
    use std::process::Command;

    let history_path = run_dir.join("coverage_history.jsonl");
    if !history_path.exists() {
        return Ok(());
    }

    // Call Python matplotlib script for high-quality plotting.
    // During crate extraction, this implementation may be compiled from a sub-crate;
    // keep compatibility with both root and sub-crate manifest layouts.
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script_path = {
        let local = manifest_dir.join("scripts").join("plot_coverage.py");
        if local.is_file() {
            local
        } else {
            manifest_dir
                .join("..")
                .join("..")
                .join("..")
                .join("scripts")
                .join("plot_coverage.py")
        }
    };

    if !script_path.is_file() {
        return Err(format!(
            "plotting script not found: {}",
            script_path.display()
        ));
    }

    let output = Command::new("python3")
        .arg(&script_path)
        .arg(run_dir)
        .output()
        .map_err(|err| format!("failed to execute plotting script: {err}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("plotting script failed: {stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.is_empty() {
        info!("{}", stdout.trim());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guided_program_rejects_legacy_raw_chunks() {
        let json = r#"{
            "chunks": [
                {
                    "kind": "raw",
                    "data": ["addi x1, x0, 1"]
                }
            ]
        }"#;

        let err =
            serde_json::from_str::<GuidedProgram>(json).expect_err("legacy raw chunk should fail");

        assert!(err.to_string().contains("unknown variant `raw`"));
    }

    #[test]
    fn guided_program_rejects_legacy_sequence_top_level() {
        let json = r#"{
            "sequences": []
        }"#;

        let err = serde_json::from_str::<GuidedProgram>(json)
            .expect_err("legacy top-level sequences field should fail");

        assert!(
            err.to_string().contains("unknown field `sequences`")
                || err.to_string().contains("missing field `chunks`")
        );
    }
}
