use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use serde::Serialize;

use riscv_fuzz_plugin_api::{CoverageMode, ISABase};
use riscv_fuzz_plugin_manager::PluginManager;
use riscv_fuzz_testcase_engine::execution_artifacts::{
    ExecutionArtifactOptions, MulticoreSyncMetricFailureMode, execute_testcase_with_artifacts,
};
use riscv_fuzz_testcase_engine::sync_report::{fmt_percent, write_metric_definitions_section};
use riscv_fuzz_testcase_engine::testcase::{MulticoreSyncConfig, MulticoreSyncMetrics};
use riscv_fuzz_toolchain::{write_json_pretty, write_string_file};

use super::{CliError, Result, ResultExt};

#[derive(Debug, Clone, Serialize)]
struct MulticoreSyncSampleRecord {
    sample_index: usize,
    seed: u64,
    metrics: MulticoreSyncMetrics,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct NumericSummary {
    min: f64,
    mean: f64,
    stddev: f64,
    p10: f64,
    p50: f64,
    p90: f64,
    max: f64,
}

#[derive(Debug, Clone, Serialize)]
struct SampleHighlight {
    sample_index: usize,
    seed: u64,
    value: f64,
}

#[derive(Debug, Serialize)]
struct MulticoreSyncExperimentSummary {
    core_id: String,
    isa_base: String,
    sample_count: usize,
    seed_start: u64,
    seed_end: u64,
    multicore_sync_config: MulticoreSyncConfig,
    metrics: BTreeMap<String, NumericSummary>,
    highlights: BTreeMap<String, SampleHighlight>,
}

#[derive(Clone, Copy)]
struct MetricSpec {
    key: &'static str,
    label: &'static str,
    extractor: fn(&MulticoreSyncMetrics) -> f64,
}

const METRIC_SPECS: &[MetricSpec] = &[
    MetricSpec {
        key: "simul",
        label: "Simul",
        extractor: |metrics| metrics.simultaneous_cycle_rate,
    },
    MetricSpec {
        key: "reg",
        label: "Reg",
        extractor: |metrics| metrics.register_overlap_cycle_rate,
    },
    MetricSpec {
        key: "mem",
        label: "Mem",
        extractor: |metrics| metrics.same_address_given_both_memory_cycle_rate,
    },
    MetricSpec {
        key: "score",
        label: "Score",
        extractor: |metrics| metrics.cycle_contention_score,
    },
];

fn percentile(sorted_values: &[f64], q: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }

    let idx = ((sorted_values.len().saturating_sub(1)) as f64 * q)
        .round()
        .clamp(0.0, sorted_values.len().saturating_sub(1) as f64) as usize;
    sorted_values[idx]
}

fn summarize_values(values: &[f64]) -> NumericSummary {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let count = sorted.len().max(1) as f64;
    let mean = sorted.iter().sum::<f64>() / count;
    let variance = sorted
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / count;

    NumericSummary {
        min: *sorted.first().unwrap_or(&0.0),
        mean,
        stddev: variance.sqrt(),
        p10: percentile(&sorted, 0.10),
        p50: percentile(&sorted, 0.50),
        p90: percentile(&sorted, 0.90),
        max: *sorted.last().unwrap_or(&0.0),
    }
}

fn best_sample_by(
    samples: &[MulticoreSyncSampleRecord],
    extractor: fn(&MulticoreSyncMetrics) -> f64,
) -> SampleHighlight {
    let best = samples
        .iter()
        .max_by(|left, right| extractor(&left.metrics).total_cmp(&extractor(&right.metrics)))
        .expect("sample list should be non-empty");
    SampleHighlight {
        sample_index: best.sample_index,
        seed: best.seed,
        value: extractor(&best.metrics),
    }
}

fn render_summary_markdown(
    summary: &MulticoreSyncExperimentSummary,
    samples: &[MulticoreSyncSampleRecord],
) -> String {
    let mut out = String::new();
    writeln!(&mut out, "# Multicore Sync Experiment").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "All overlap metrics below are computed from real instruction cycle windows (`clk_start..=clk_end`).").unwrap();
    writeln!(
        &mut out,
        "Reported covered cycles are the union of cycles where either hart is active, so gaps where both harts are idle are excluded."
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "| Field | Value |").unwrap();
    writeln!(&mut out, "| --- | --- |").unwrap();
    writeln!(&mut out, "| Implementation | `{}` |", summary.core_id).unwrap();
    writeln!(&mut out, "| ISA Base | `{}` |", summary.isa_base).unwrap();
    writeln!(&mut out, "| Samples | {} |", summary.sample_count).unwrap();
    writeln!(
        &mut out,
        "| Seed Range | {}..={} |",
        summary.seed_start, summary.seed_end
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Strategy").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "- interaction_slot_ratio = {:.2}%",
        summary.multicore_sync_config.interaction_slot_ratio * 100.0
    )
    .unwrap();
    writeln!(
        &mut out,
        "- shared_address_ratio = {:.2}%",
        summary.multicore_sync_config.shared_address_ratio * 100.0
    )
    .unwrap();
    writeln!(
        &mut out,
        "- prefer_focused_sequences = {}",
        summary.multicore_sync_config.prefer_focused_sequences
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    write_metric_definitions_section(&mut out);
    writeln!(&mut out, "## Aggregate Metrics").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "| Metric | Mean | Stddev | P10 | P50 | P90 | Min | Max |"
    )
    .unwrap();
    writeln!(
        &mut out,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();
    for spec in METRIC_SPECS {
        let stats = summary
            .metrics
            .get(spec.key)
            .expect("every configured metric should be summarized");
        writeln!(
            &mut out,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            spec.label,
            fmt_percent(stats.mean),
            fmt_percent(stats.stddev),
            fmt_percent(stats.p10),
            fmt_percent(stats.p50),
            fmt_percent(stats.p90),
            fmt_percent(stats.min),
            fmt_percent(stats.max)
        )
        .unwrap();
    }
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Best Samples").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "| Metric | Sample Index | Seed | Value |").unwrap();
    writeln!(&mut out, "| --- | ---: | ---: | ---: |").unwrap();
    for spec in METRIC_SPECS {
        let highlight = summary
            .highlights
            .get(spec.key)
            .expect("every configured metric should have a highlight");
        writeln!(
            &mut out,
            "| {} | {} | {} | {} |",
            spec.label,
            highlight.sample_index,
            highlight.seed,
            fmt_percent(highlight.value)
        )
        .unwrap();
    }
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## First Samples").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "| Sample | Seed | Covered Cycles | Simul | Reg | Mem | Score |"
    )
    .unwrap();
    writeln!(
        &mut out,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();
    for sample in samples.iter().take(10) {
        writeln!(
            &mut out,
            "| {} | {} | {} | {} | {} | {} | {} |",
            sample.sample_index,
            sample.seed,
            sample.metrics.covered_cycle_count,
            fmt_percent(sample.metrics.simultaneous_cycle_rate),
            fmt_percent(sample.metrics.register_overlap_cycle_rate),
            fmt_percent(sample.metrics.same_address_given_both_memory_cycle_rate),
            fmt_percent(sample.metrics.cycle_contention_score),
        )
        .unwrap();
    }
    out
}

fn next_seed(base_seed: u64, sample_index: usize) -> Result<u64> {
    base_seed
        .checked_add(sample_index as u64)
        .ok_or_else(|| CliError::new(format!("seed overflow at sample index {sample_index}")))
}

fn write_summary_artifacts(
    run_dir: &Path,
    summary: &MulticoreSyncExperimentSummary,
    samples: &[MulticoreSyncSampleRecord],
) -> Result<()> {
    let summary_json_path = run_dir.join("multicore_sync_experiment_summary.json");
    let summary_md_path = run_dir.join("multicore_sync_experiment_summary.md");
    let samples_json_path = run_dir.join("multicore_sync_experiment_samples.json");

    write_json_pretty(&summary_json_path, summary)
        .with_context(|| format!("failed to write {}", summary_json_path.display()))?;

    write_string_file(&summary_md_path, render_summary_markdown(summary, samples))
        .with_context(|| format!("failed to write {}", summary_md_path.display()))?;

    write_json_pretty(&samples_json_path, samples)
        .with_context(|| format!("failed to write {}", samples_json_path.display()))?;

    Ok(())
}

fn sample_directory(run_dir: &Path, sample_index: usize) -> PathBuf {
    run_dir.join(format!("sample_{sample_index:04}"))
}

pub(super) fn run_measure_multicore_sync_mode(
    cfg: &riscv_fuzz_config::ExperimentConfig,
    run_dir: PathBuf,
    core_id: String,
    isa_override: Option<ISABase>,
    samples: usize,
    seed_start: Option<u64>,
) -> Result<()> {
    if samples == 0 {
        return Err(CliError::new("--samples must be >= 1"));
    }

    let mut testcase_config = cfg.testcase_config.clone();
    if let Some(isa) = isa_override {
        testcase_config.isa_base = isa;
    }
    let testcase_hart_count = testcase_config.hart_count();
    if !testcase_config.is_dual_hart() {
        return Err(CliError::new(format!(
            "multicore sync measurement requires workload.hart_count = 2, found {}",
            testcase_hart_count
        )));
    }

    let plugin_manager = PluginManager::global();
    if !plugin_manager.supports_isa_base(&core_id, testcase_config.isa_base) {
        return Err(CliError::new(format!(
            "core {core_id} does not support ISA base {}",
            testcase_config.isa_base
        )));
    }

    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let core_ids = PluginManager::normalize_impls(vec![core_id.clone()]);
    let base_seed = seed_start.or(testcase_config.seed).unwrap_or(0);
    let mut sample_records = Vec::with_capacity(samples);

    for sample_index in 0..samples {
        let seed = next_seed(base_seed, sample_index)?;
        let mut sample_config = testcase_config.clone();
        sample_config.seed = Some(seed);
        let testcase =
            riscv_fuzz_testcase_engine::impls::generate_random_testcase(&core_ids, sample_config)
                .with_context(|| {
                format!(
                    "failed to generate sample {} for core {}",
                    sample_index, core_id,
                )
            })?;

        let sample_dir = sample_directory(&run_dir, sample_index);
        fs::create_dir_all(&sample_dir)
            .with_context(|| format!("failed to create {}", sample_dir.display()))?;

        let result = execute_testcase_with_artifacts(
            &testcase,
            &core_ids,
            &sample_dir,
            CoverageMode::Off,
            ExecutionArtifactOptions {
                testcase_filename: "testcase.json",
                emit_execution_output_json: true,
                emit_execution_report_md: false,
                emit_execution_timeline_svg: true,
                multicore_sync_metric_failure_mode: MulticoreSyncMetricFailureMode::Strict,
            },
        )
        .map_err(|err| {
            CliError::new(format!(
                "failed to execute sample {} for implementation {}: {}",
                sample_index, core_id, err
            ))
        })?;
        let metrics = result
            .multicore_sync_metrics
            .get(&core_id)
            .cloned()
            .ok_or_else(|| {
                CliError::new(format!(
                    "sample {} execution did not produce multicore sync metrics for implementation {}",
                    sample_index, core_id
                ))
            })?;
        sample_records.push(MulticoreSyncSampleRecord {
            sample_index,
            seed,
            metrics,
        });
    }

    let mut metrics = BTreeMap::new();
    for spec in METRIC_SPECS {
        let values: Vec<f64> = sample_records
            .iter()
            .map(|sample| (spec.extractor)(&sample.metrics))
            .collect();
        metrics.insert(spec.key.to_string(), summarize_values(&values));
    }

    let mut highlights = BTreeMap::new();
    for spec in METRIC_SPECS {
        highlights.insert(
            spec.key.to_string(),
            best_sample_by(&sample_records, spec.extractor),
        );
    }

    let summary = MulticoreSyncExperimentSummary {
        core_id: core_id.clone(),
        isa_base: testcase_config.isa_base.to_string(),
        sample_count: sample_records.len(),
        seed_start: base_seed,
        seed_end: next_seed(base_seed, sample_records.len().saturating_sub(1))?,
        multicore_sync_config: testcase_config
            .multicore_sync_config()
            .expect("dual-hart measure-sync config must carry multicore sync config")
            .clone(),
        metrics,
        highlights,
    };

    write_summary_artifacts(&run_dir, &summary, &sample_records)?;

    log::info!(
        "Measured cycle-based multicore sync for {} ({}): {} samples written to {}",
        core_id,
        testcase_config.isa_base,
        sample_records.len(),
        run_dir.display()
    );

    Ok(())
}
