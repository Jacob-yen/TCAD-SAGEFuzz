use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    env,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use riscv_fuzz_plugin_api::{
    AlignmentMode, CoverageMode, ExecuteRequest, ExecutionOutput, HartExecutionOutput,
    InstructionEvent, StreamingCollector, validate_typed_execution_output,
};
use riscv_fuzz_plugin_manager::PluginManager;
use riscv_fuzz_testcase_engine::testcase::GeneratedTestCase;

use super::{
    normalize::{WriteEffect, logical_instruction_effects_for_diff},
    types::DiffAnalysisError,
    view::{DiffExecutionView, build_partial_diff_execution_view},
};

enum WorkerMessage {
    Event {
        impl_ref: String,
        event: InstructionEvent,
    },
    Finished {
        impl_ref: String,
        elapsed: Duration,
        result: Result<(), String>,
    },
}

#[derive(Default)]
struct ImplStreamState {
    collector: StreamingCollector,
    progress_by_hart: BTreeMap<u32, HartStreamingProgress>,
    raw_output: Option<ExecutionOutput>,
    context: Option<DiffExecutionView>,
    context_dirty: bool,
    finished: bool,
    elapsed: Option<Duration>,
}

#[derive(Debug, Default, Clone)]
struct HartStreamingProgress {
    seen_trace_local_indices: BTreeSet<usize>,
    last_observed_local_index: Option<usize>,
    sealed_local_prefix_len: usize,
}

#[derive(Default, Clone, PartialEq, Eq)]
struct HartIndexLayout {
    hart_instruction_counts: BTreeMap<u32, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StreamDivergencePoint {
    pub(super) hart_id: u32,
    pub(super) local_index: usize,
}

pub(super) struct StreamingDiffExecutionResult {
    pub(super) outputs: HashMap<String, DiffExecutionView>,
    pub(super) raw_outputs: HashMap<String, ExecutionOutput>,
    pub(super) first_divergence: Option<StreamDivergencePoint>,
    pub(super) timings: HashMap<String, Duration>,
}

type TypedSequencesPerHart = Vec<(
    u32,
    Vec<riscv_instruction_types::InstructionSequence<riscv_instruction::RiscvInstruction>>,
)>;

#[derive(Clone)]
struct StreamingExecutionCase {
    testcase: GeneratedTestCase,
    per_hart_typed: TypedSequencesPerHart,
}

pub(super) fn execute_testcase_streaming_diff_detailed(
    testcase: &GeneratedTestCase,
    impl_order: &[String],
    run_dir: &Path,
    coverage_mode: CoverageMode,
) -> Result<StreamingDiffExecutionResult, DiffAnalysisError> {
    let mut cases = HashMap::new();
    for impl_ref in impl_order {
        let per_hart_typed = testcase
            .combined_typed_sequences_per_hart_of(impl_ref)
            .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            })?;
        cases.insert(
            impl_ref.clone(),
            StreamingExecutionCase {
                testcase: testcase.clone(),
                per_hart_typed,
            },
        );
    }
    execute_streaming_diff_cases(testcase, impl_order, run_dir, coverage_mode, cases)
}

pub(super) fn execute_per_impl_testcases_streaming_diff(
    canonical_testcase: &GeneratedTestCase,
    per_impl_testcases: &HashMap<String, GeneratedTestCase>,
    impl_order: &[String],
    run_dir: &Path,
    coverage_mode: CoverageMode,
) -> Result<StreamingDiffExecutionResult, DiffAnalysisError> {
    let mut cases = HashMap::new();
    for impl_ref in impl_order {
        let testcase = per_impl_testcases.get(impl_ref).ok_or_else(|| {
            DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            }
        })?;
        let per_hart_typed = testcase
            .combined_typed_sequences_per_hart_of(impl_ref)
            .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            })?;
        cases.insert(
            impl_ref.clone(),
            StreamingExecutionCase {
                testcase: testcase.clone(),
                per_hart_typed,
            },
        );
    }
    execute_streaming_diff_cases(
        canonical_testcase,
        impl_order,
        run_dir,
        coverage_mode,
        cases,
    )
}

fn execute_streaming_diff_cases(
    compare_testcase: &GeneratedTestCase,
    impl_order: &[String],
    run_dir: &Path,
    coverage_mode: CoverageMode,
    cases: HashMap<String, StreamingExecutionCase>,
) -> Result<StreamingDiffExecutionResult, DiffAnalysisError> {
    let plugin_manager = PluginManager::global();
    let mut hart_layouts = HashMap::new();
    for impl_ref in impl_order {
        let case = cases
            .get(impl_ref)
            .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            })?;
        let layout = hart_index_layout(&case.per_hart_typed);
        hart_layouts.insert(impl_ref.clone(), layout);
    }
    let early_compare_enabled = hart_layouts
        .values()
        .next()
        .is_some_and(|first| hart_layouts.values().all(|layout| layout == first));
    let compare_hart_instruction_counts = hart_layouts
        .values()
        .next()
        .map(|layout| layout.hart_instruction_counts.clone())
        .unwrap_or_default();
    let seal_lookahead = streaming_logical_completion_lookahead();

    // Pre-flight validation before starting worker threads, so capability errors
    // fail deterministically and do not race with partially emitted traces.
    for impl_ref in impl_order {
        let dir = run_dir.join(impl_ref);
        let case = cases
            .get(impl_ref)
            .expect("cases was populated for every impl");
        let alignment_mode = alignment_mode_for(&case.testcase)?;
        let hart_inputs = hart_inputs_for(&case.per_hart_typed);
        plugin_manager
            .validate_with_request(
                impl_ref,
                ExecuteRequest {
                    run_root: &dir,
                    isa_base: case.testcase.config.isa_base,
                    hart_inputs: &hart_inputs,
                    extension_override: Some(&case.testcase.extensions),
                    float_mode_override: None,
                    alignment_mode,
                    coverage_mode,
                },
            )
            .map_err(|err| DiffAnalysisError::Execution(Box::new(err)))?;
    }

    let (tx, rx) = mpsc::channel::<WorkerMessage>();
    let cancel = Arc::new(AtomicBool::new(false));
    let cases_arc = Arc::new(cases);
    let run_dir = run_dir.to_path_buf();
    let mut handles = Vec::with_capacity(impl_order.len());

    for impl_ref in impl_order.iter().cloned() {
        let tx = tx.clone();
        let cancel = Arc::clone(&cancel);
        let cases = Arc::clone(&cases_arc);
        let root = run_dir.clone();
        handles.push(thread::spawn(move || {
            let started = Instant::now();
            let plugin_manager = PluginManager::global();
            let case = match cases.get(&impl_ref) {
                Some(v) => v,
                None => {
                    let _ = tx.send(WorkerMessage::Finished {
                        impl_ref,
                        elapsed: started.elapsed(),
                        result: Err("missing typed inputs for streaming worker".to_string()),
                    });
                    return;
                }
            };
            let alignment_mode = match alignment_mode_for(&case.testcase) {
                Ok(mode) => mode,
                Err(err) => {
                    let _ = tx.send(WorkerMessage::Finished {
                        impl_ref,
                        elapsed: started.elapsed(),
                        result: Err(err.to_string()),
                    });
                    return;
                }
            };
            let hart_inputs = hart_inputs_for(&case.per_hart_typed);
            let dir = root.join(&impl_ref);
            let request = ExecuteRequest {
                run_root: &dir,
                isa_base: case.testcase.config.isa_base,
                hart_inputs: &hart_inputs,
                extension_override: Some(&case.testcase.extensions),
                float_mode_override: None,
                alignment_mode,
                coverage_mode,
            };

            let result = plugin_manager
                .execute_streaming_with_request(&impl_ref, request, &mut |event| {
                    if cancel.load(Ordering::SeqCst) {
                        return false;
                    }
                    tx.send(WorkerMessage::Event {
                        impl_ref: impl_ref.clone(),
                        event,
                    })
                    .is_ok()
                        && !cancel.load(Ordering::SeqCst)
                })
                .map_err(|err| err.to_string());

            let _ = tx.send(WorkerMessage::Finished {
                impl_ref,
                elapsed: started.elapsed(),
                result,
            });
        }));
    }
    drop(tx);

    let mut states = impl_order
        .iter()
        .cloned()
        .map(|impl_ref| (impl_ref, ImplStreamState::default()))
        .collect::<HashMap<_, _>>();
    let mut first_error: Option<String> = None;
    let mut first_divergence: Option<StreamDivergencePoint> = None;
    let mut next_compare_by_hart = compare_hart_instruction_counts
        .keys()
        .copied()
        .map(|hart_id| (hart_id, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut finished_count = 0usize;
    let event_gap_timeout = streaming_event_gap_timeout();
    let mut last_event_at = HashMap::<String, Instant>::new();

    while finished_count < impl_order.len() {
        let message = match event_gap_timeout {
            Some(_) => match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(message) => message,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if first_divergence.is_none()
                        && let Some((impl_ref, gap, limit)) =
                            expired_streaming_gap(&states, &last_event_at, event_gap_timeout)
                    {
                        cancel.store(true, Ordering::SeqCst);
                        if first_error.is_none() {
                            first_error = Some(format!(
                                "{impl_ref}: streaming event gap timeout after {:.3}s (limit {:.3}s)",
                                gap.as_secs_f64(),
                                limit.as_secs_f64()
                            ));
                        }
                    }
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            },
            None => match rx.recv() {
                Ok(message) => message,
                Err(_) => break,
            },
        };

        match message {
            WorkerMessage::Event { impl_ref, event } => {
                last_event_at.insert(impl_ref.clone(), Instant::now());
                let layout = hart_layouts
                    .get(&impl_ref)
                    .expect("hart_layouts was populated for every impl");
                let validated = {
                    let state = states
                        .get(&impl_ref)
                        .expect("state was initialized for every impl");
                    validate_streaming_instruction_event(&event, layout, state)
                };
                let validated = match validated {
                    Ok(validated) => validated,
                    Err(message) => {
                        cancel.store(true, Ordering::SeqCst);
                        if first_error.is_none() {
                            first_error = Some(format!("{impl_ref}: {message}"));
                        }
                        continue;
                    }
                };
                let state = states
                    .get_mut(&impl_ref)
                    .expect("state was initialized for every impl");
                let keep_going = state.collector.push(event);
                if let Some(validated) = validated {
                    let _ = update_progress_for_instruction(
                        state,
                        validated.hart_id,
                        validated.local_index,
                        layout,
                        seal_lookahead,
                    );
                    state.context_dirty = true;
                }
                if let Some(error) = state.collector.error.as_ref() {
                    cancel.store(true, Ordering::SeqCst);
                    if first_error.is_none() {
                        first_error = Some(format!("{impl_ref}: {error}"));
                    }
                    continue;
                }
                if !keep_going {
                    state.finished = true;
                }
                if first_error.is_none()
                    && first_divergence.is_none()
                    && early_compare_enabled
                    && let Some(diverged_at) = advance_streaming_compare(
                        compare_testcase,
                        impl_order,
                        cases_arc.as_ref(),
                        &mut states,
                        &mut next_compare_by_hart,
                        &compare_hart_instruction_counts,
                    )?
                {
                    first_divergence = Some(diverged_at);
                    cancel.store(true, Ordering::SeqCst);
                }
            }
            WorkerMessage::Finished {
                impl_ref,
                elapsed,
                result,
            } => {
                finished_count += 1;
                let state = states
                    .get_mut(&impl_ref)
                    .expect("state was initialized for every impl");
                state.finished = true;
                state.elapsed = Some(elapsed);
                let layout = hart_layouts
                    .get(&impl_ref)
                    .expect("hart_layouts was populated for every impl");
                let flushed_indices = flush_finished_impl_progress(state, layout);
                if !flushed_indices.is_empty() {
                    state.context_dirty = true;
                }
                if let Err(err) = result
                    && first_divergence.is_none()
                    && first_error.is_none()
                {
                    cancel.store(true, Ordering::SeqCst);
                    first_error = Some(format!("{impl_ref}: {err}"));
                }
                if first_error.is_none()
                    && first_divergence.is_none()
                    && early_compare_enabled
                    && let Some(diverged_at) = advance_streaming_compare(
                        compare_testcase,
                        impl_order,
                        cases_arc.as_ref(),
                        &mut states,
                        &mut next_compare_by_hart,
                        &compare_hart_instruction_counts,
                    )?
                {
                    first_divergence = Some(diverged_at);
                    cancel.store(true, Ordering::SeqCst);
                }
            }
        }
    }

    for handle in handles {
        if let Err(panic) = handle.join()
            && first_error.is_none()
        {
            first_error = Some(format!("streaming worker panicked: {panic:?}"));
        }
    }

    if let Some(error) = first_error {
        return Err(DiffAnalysisError::Execution(Box::new(
            std::io::Error::other(error),
        )));
    }

    if let Some(diverged_at) = first_divergence {
        let keep_by_hart = compared_prefix_keep_limits(&next_compare_by_hart, diverged_at);
        for impl_ref in impl_order {
            let state = states
                .get_mut(impl_ref)
                .expect("state was initialized for every impl");
            trim_stream_state_after_divergence(state, &keep_by_hart);
        }
    }

    for impl_ref in impl_order {
        let state = states
            .get_mut(impl_ref)
            .expect("state was initialized for every impl");
        let case = cases_arc
            .get(impl_ref)
            .expect("cases was populated for every impl");
        refresh_context(&case.testcase, impl_ref, &case.per_hart_typed, state)?;
    }

    let mut timings = HashMap::new();
    for impl_ref in impl_order {
        let elapsed = states
            .get(impl_ref)
            .and_then(|state| state.elapsed)
            .unwrap_or_default();
        timings.insert(impl_ref.clone(), elapsed);
    }
    let mut outputs = HashMap::new();
    let mut raw_outputs = HashMap::new();
    for impl_ref in impl_order {
        let state = states
            .remove(impl_ref)
            .expect("state was initialized for every impl");
        let raw_output =
            state
                .raw_output
                .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                    core_id: impl_ref.clone(),
                })?;
        let context = state
            .context
            .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            })?;
        raw_outputs.insert(impl_ref.clone(), raw_output);
        outputs.insert(impl_ref.clone(), context);
    }
    Ok(StreamingDiffExecutionResult {
        outputs,
        raw_outputs,
        first_divergence,
        timings,
    })
}

fn streaming_event_gap_timeout() -> Option<Duration> {
    env::var("RISCV_FUZZ_STREAM_EVENT_TIMEOUT_SECS")
        .ok()
        .or_else(|| env::var("RISCV_FUZZ_EXEC_TIMEOUT_SECS").ok())
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
}

fn alignment_mode_for(testcase: &GeneratedTestCase) -> Result<AlignmentMode, DiffAnalysisError> {
    let allow_unaligned = testcase.config.unaligned_access_required.ok_or_else(|| {
        DiffAnalysisError::Execution(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "unaligned_access_required must be explicitly set for streaming testcase execution",
        )))
    })?;
    Ok(if allow_unaligned {
        AlignmentMode::AllowUnaligned
    } else {
        AlignmentMode::RequireAligned
    })
}

fn expired_streaming_gap(
    states: &HashMap<String, ImplStreamState>,
    last_event_at: &HashMap<String, Instant>,
    timeout: Option<Duration>,
) -> Option<(String, Duration, Duration)> {
    let timeout = timeout?;
    let now = Instant::now();
    states.iter().find_map(|(impl_ref, state)| {
        if state.finished {
            return None;
        }
        let last_event = last_event_at.get(impl_ref)?;
        let gap = now.saturating_duration_since(*last_event);
        (gap >= timeout).then(|| (impl_ref.clone(), gap, timeout))
    })
}

fn hart_inputs_for<'a>(
    per_hart_typed: &'a TypedSequencesPerHart,
) -> Vec<(
    u32,
    &'a [riscv_instruction_types::InstructionSequence<riscv_instruction::RiscvInstruction>],
)> {
    per_hart_typed
        .iter()
        .map(|(hart_id, seqs)| (*hart_id, seqs.as_slice()))
        .collect()
}

fn sequence_traceable_len(
    sequence: &riscv_instruction_types::InstructionSequence<riscv_instruction::RiscvInstruction>,
) -> usize {
    sequence.pre_instructions.len() + 1 + sequence.post_instructions.len()
}

fn hart_index_layout(per_hart_typed: &TypedSequencesPerHart) -> HartIndexLayout {
    let mut hart_instruction_counts = BTreeMap::new();
    for (hart_id, seqs) in per_hart_typed {
        let count = seqs.iter().map(sequence_traceable_len).sum::<usize>();
        hart_instruction_counts.insert(*hart_id, count);
    }
    HartIndexLayout {
        hart_instruction_counts,
    }
}

fn streaming_logical_completion_lookahead() -> usize {
    env::var("RISCV_FUZZ_STREAM_TRACE_LOOKAHEAD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ValidatedStreamInstruction {
    hart_id: u32,
    local_index: usize,
}

fn validate_streaming_instruction_event(
    event: &InstructionEvent,
    hart_layout: &HartIndexLayout,
    state: &ImplStreamState,
) -> Result<Option<ValidatedStreamInstruction>, String> {
    let InstructionEvent::Instruction {
        hart_id,
        instruction,
    } = event
    else {
        return Ok(None);
    };
    let Some(trace_inst_id) = instruction.trace_inst_id else {
        return Ok(None);
    };
    if trace_inst_id.hart_id != *hart_id {
        return Err(format!(
            "streaming row at pc 0x{:x} for hart {} points at hart {}",
            instruction.pc, hart_id, trace_inst_id.hart_id
        ));
    }
    let Some(instruction_count) = hart_layout.hart_instruction_counts.get(hart_id).copied() else {
        return Err(format!(
            "streaming row at pc 0x{:x} references missing hart {}",
            instruction.pc, hart_id
        ));
    };
    if trace_inst_id.local_index >= instruction_count {
        return Err(format!(
            "streaming row at pc 0x{:x} for hart {} points at missing trace instruction {} (available={})",
            instruction.pc, hart_id, trace_inst_id.local_index, instruction_count
        ));
    }
    if let Some(last) = state
        .progress_by_hart
        .get(hart_id)
        .and_then(|progress| progress.last_observed_local_index)
        && trace_inst_id.local_index < last
    {
        return Err(format!(
            "streaming rows for hart {} regressed from trace instruction {} back to {} at pc 0x{:x}",
            hart_id, last, trace_inst_id.local_index, instruction.pc
        ));
    }
    Ok(Some(ValidatedStreamInstruction {
        hart_id: *hart_id,
        local_index: trace_inst_id.local_index,
    }))
}

fn update_progress_for_instruction(
    state: &mut ImplStreamState,
    hart_id: u32,
    local_index: usize,
    hart_layout: &HartIndexLayout,
    lookahead: usize,
) -> Vec<usize> {
    let progress = state.progress_by_hart.entry(hart_id).or_default();
    progress.last_observed_local_index = Some(
        progress
            .last_observed_local_index
            .map_or(local_index, |last| last.max(local_index)),
    );
    progress.seen_trace_local_indices.insert(local_index);
    seal_progress_indices(progress, hart_id, hart_layout, lookahead, false)
}

fn flush_finished_impl_progress(
    state: &mut ImplStreamState,
    hart_layout: &HartIndexLayout,
) -> Vec<usize> {
    let mut flushed = Vec::new();
    for hart_id in hart_layout
        .hart_instruction_counts
        .keys()
        .copied()
        .collect::<Vec<_>>()
    {
        let Some(progress) = state.progress_by_hart.get_mut(&hart_id) else {
            continue;
        };
        flushed.extend(seal_progress_indices(
            progress,
            hart_id,
            hart_layout,
            0,
            true,
        ));
    }
    flushed
}

fn seal_progress_indices(
    progress: &mut HartStreamingProgress,
    hart_id: u32,
    hart_layout: &HartIndexLayout,
    lookahead: usize,
    flush_tail: bool,
) -> Vec<usize> {
    let Some(hart_instruction_count) = hart_layout.hart_instruction_counts.get(&hart_id).copied()
    else {
        return Vec::new();
    };
    let mut contiguous_seen = progress.sealed_local_prefix_len;
    while progress.seen_trace_local_indices.contains(&contiguous_seen) {
        contiguous_seen += 1;
    }
    let target = if flush_tail {
        contiguous_seen
    } else {
        contiguous_seen.saturating_sub(lookahead)
    }
    .min(hart_instruction_count);
    let newly_sealed = (progress.sealed_local_prefix_len..target).collect::<Vec<_>>();
    progress.sealed_local_prefix_len = target;
    newly_sealed
}

fn all_impls_ready_at_hart_local(
    states: &HashMap<String, ImplStreamState>,
    impl_order: &[String],
    hart_id: u32,
    local_index: usize,
) -> bool {
    impl_order.iter().all(|impl_ref| {
        states
            .get(impl_ref)
            .and_then(|state| state.progress_by_hart.get(&hart_id))
            .is_some_and(|progress| progress.sealed_local_prefix_len > local_index)
    })
}

fn advance_streaming_compare(
    compare_testcase: &GeneratedTestCase,
    impl_order: &[String],
    cases: &HashMap<String, StreamingExecutionCase>,
    states: &mut HashMap<String, ImplStreamState>,
    next_compare_by_hart: &mut BTreeMap<u32, usize>,
    compare_hart_instruction_counts: &BTreeMap<u32, usize>,
) -> Result<Option<StreamDivergencePoint>, DiffAnalysisError> {
    for (&hart_id, &instruction_count) in compare_hart_instruction_counts {
        let Some(next_local_index) = next_compare_by_hart.get_mut(&hart_id) else {
            continue;
        };
        while *next_local_index < instruction_count
            && all_impls_ready_at_hart_local(states, impl_order, hart_id, *next_local_index)
        {
            refresh_ready_contexts(cases, states, impl_order)?;
            if contexts_diverge_at(
                compare_testcase,
                impl_order,
                states,
                hart_id,
                *next_local_index,
            )? {
                return Ok(Some(StreamDivergencePoint {
                    hart_id,
                    local_index: *next_local_index,
                }));
            }
            *next_local_index += 1;
        }
    }
    Ok(None)
}

fn refresh_ready_contexts(
    cases: &HashMap<String, StreamingExecutionCase>,
    states: &mut HashMap<String, ImplStreamState>,
    impl_order: &[String],
) -> Result<(), DiffAnalysisError> {
    for impl_ref in impl_order {
        let state = states
            .get_mut(impl_ref)
            .expect("state was initialized for every impl");
        if state.context.is_some() && !state.context_dirty {
            continue;
        }
        let case = cases
            .get(impl_ref)
            .expect("cases was populated for every impl");
        refresh_context(&case.testcase, impl_ref, &case.per_hart_typed, state)?;
        state.context_dirty = false;
    }
    Ok(())
}

fn compared_prefix_keep_limits(
    next_compare_by_hart: &BTreeMap<u32, usize>,
    divergence: StreamDivergencePoint,
) -> BTreeMap<u32, Option<usize>> {
    next_compare_by_hart
        .iter()
        .map(|(&hart_id, &next_local_index)| {
            let keep_through = if hart_id == divergence.hart_id {
                Some(divergence.local_index)
            } else {
                next_local_index.checked_sub(1)
            };
            (hart_id, keep_through)
        })
        .collect()
}

fn trim_stream_state_after_divergence(
    state: &mut ImplStreamState,
    keep_through_by_hart: &BTreeMap<u32, Option<usize>>,
) {
    for (&hart_id, hart_output) in &mut state.collector.per_hart {
        hart_output.instructions.retain(|instruction| {
            let Some(trace_inst_id) = instruction.trace_inst_id else {
                return true;
            };
            if trace_inst_id.hart_id != hart_id {
                return false;
            }
            keep_through_by_hart
                .get(&hart_id)
                .and_then(|limit| *limit)
                .is_some_and(|limit| trace_inst_id.local_index <= limit)
        });
    }
    state.raw_output = None;
    state.context = None;
    state.context_dirty = true;
}

fn build_validated_execution_output(
    testcase: &GeneratedTestCase,
    impl_ref: &str,
    per_hart_typed: &TypedSequencesPerHart,
    state: &ImplStreamState,
) -> Result<ExecutionOutput, DiffAnalysisError> {
    let mut per_hart = state.collector.per_hart.clone();
    for (hart_id, _) in per_hart_typed {
        per_hart
            .entry(*hart_id)
            .or_insert_with(|| HartExecutionOutput::empty(*hart_id));
    }
    let mut perf = state.collector.perf.clone().unwrap_or_default();
    for (hart_id, _) in per_hart_typed {
        perf.per_hart.entry(*hart_id).or_default();
    }
    let mut output = ExecutionOutput {
        per_hart,
        core_id: impl_ref.to_string(),
        isa_base: testcase.config.isa_base,
        perf,
    };
    let range = testcase.mem_range.get(impl_ref).copied().ok_or_else(|| {
        DiffAnalysisError::MissingInstructions {
            core_id: impl_ref.to_string(),
        }
    })?;
    output
        .normalize(
            range,
            &testcase.config.random_config.register_operand_config,
            testcase
                .config
                .random_config
                .memory_access_config
                .memory_base_register_number_range,
        )
        .map_err(|err| DiffAnalysisError::Execution(Box::new(err)))?;
    let typed_by_hart = per_hart_typed
        .iter()
        .map(|(hart_id, sequences)| (*hart_id, sequences.as_slice()))
        .collect::<Vec<_>>();
    validate_typed_execution_output(&mut output, typed_by_hart.as_slice())
        .map_err(|err| DiffAnalysisError::Execution(Box::new(err)))?;
    Ok(output)
}

fn refresh_context(
    testcase: &GeneratedTestCase,
    impl_ref: &str,
    per_hart_typed: &TypedSequencesPerHart,
    state: &mut ImplStreamState,
) -> Result<(), DiffAnalysisError> {
    let output = build_validated_execution_output(testcase, impl_ref, per_hart_typed, state)?;
    let range = testcase.mem_range.get(impl_ref).copied().ok_or_else(|| {
        DiffAnalysisError::MissingInstructions {
            core_id: impl_ref.to_string(),
        }
    })?;
    state.raw_output = Some(output.clone());
    state.context = Some(
        build_partial_diff_execution_view(output, range, per_hart_typed)
            .map_err(|err| DiffAnalysisError::Execution(Box::new(err)))?,
    );
    state.context_dirty = false;
    Ok(())
}

fn contexts_diverge_at(
    testcase: &GeneratedTestCase,
    impl_order: &[String],
    states: &HashMap<String, ImplStreamState>,
    hart_id: u32,
    local_index: usize,
) -> Result<bool, DiffAnalysisError> {
    let mut baseline_exceptions: Option<Vec<u64>> = None;
    let mut baseline_writes: Option<WriteEffect> = None;

    for impl_ref in impl_order {
        let state = states
            .get(impl_ref)
            .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                core_id: impl_ref.clone(),
            })?;
        let context =
            state
                .context
                .as_ref()
                .ok_or_else(|| DiffAnalysisError::MissingInstructions {
                    core_id: impl_ref.clone(),
                })?;
        let instruction =
            context
                .logical_instruction_by_hart_local(hart_id, local_index)
                .ok_or_else(|| DiffAnalysisError::TestInstructionCountMismatch {
                    details: format!(
                        "missing streaming execution context for hart {hart_id} instruction {local_index} in {impl_ref}"
                    ),
                })?;

        let mut exceptions = instruction
            .exceptions
            .iter()
            .map(|cause| cause.code)
            .collect::<Vec<_>>();
        exceptions.sort_unstable();
        if let Some(baseline) = &baseline_exceptions {
            if baseline != &exceptions {
                return Ok(true);
            }
        } else {
            baseline_exceptions = Some(exceptions);
        }

        let writes = logical_instruction_effects_for_diff(instruction, &testcase.config);
        if let Some((baseline_regs, baseline_mems)) = &baseline_writes {
            if baseline_regs != &writes.0 || baseline_mems != &writes.1 {
                return Ok(true);
            }
        } else {
            baseline_writes = Some(writes);
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use riscv_fuzz_plugin_api::{
        AsmInstructionExecution, HartExecutionOutput, ISABase, InstructionTiming, IsaExtensions,
        TraceInstId,
    };
    use riscv_fuzz_testcase_engine::testcase::{
        GeneratedTestCase, HartConfig, HartProgram, TestCaseConfig,
    };
    use riscv_instruction::RV32Extensions;
    use riscv_instruction::{AsmDecodeWithContext, IsaDecodeContext, RiscvInstruction};
    use riscv_instruction_types::{
        InstructionSequence, MemoryAccessConfig, RandomConfig, RegisterOperandConfig,
    };

    fn decoded_addi() -> RiscvInstruction {
        RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
            .expect("test instruction should decode")
    }

    fn decoded_addi_x2() -> RiscvInstruction {
        RiscvInstruction::decode_asm_with_context("addi x2, x0, 2", IsaDecodeContext::RV32)
            .expect("test instruction should decode")
    }

    fn instruction_event(
        hart_id: u32,
        local_index: usize,
        instruction: RiscvInstruction,
    ) -> InstructionEvent {
        InstructionEvent::Instruction {
            hart_id,
            instruction: asm_for_source(hart_id, local_index, local_index as u64, instruction),
        }
    }

    fn asm_for(pc: u64, instruction: RiscvInstruction) -> AsmInstructionExecution {
        AsmInstructionExecution {
            pc,
            instruction,
            source_annotation: None,
            trace_inst_id: None,
            exceptions: Vec::new(),
            register_writes: Vec::new(),
            memory_writes: Vec::new(),
            timing: InstructionTiming::default(),
        }
    }

    fn asm_for_source(
        hart_id: u32,
        local_index: usize,
        pc: u64,
        instruction: RiscvInstruction,
    ) -> AsmInstructionExecution {
        let mut asm = asm_for(pc, instruction);
        asm.trace_inst_id = Some(TraceInstId::new(hart_id, local_index));
        asm
    }

    #[test]
    fn sealing_waits_for_future_trace_instructions() {
        let addi = decoded_addi();
        let seqs = vec![
            InstructionSequence::new(addi.clone()),
            InstructionSequence::new(addi.clone()),
            InstructionSequence::new(addi),
        ];
        let layout = hart_index_layout(&vec![(0, seqs)]);
        let mut state = ImplStreamState::default();

        let first = validate_streaming_instruction_event(
            &instruction_event(0, 0, decoded_addi()),
            &layout,
            &state,
        )
        .expect("first event should validate")
        .expect("first event should map to a trace instruction");
        assert!(
            update_progress_for_instruction(
                &mut state,
                first.hart_id,
                first.local_index,
                &layout,
                2
            )
            .is_empty()
        );

        let second = validate_streaming_instruction_event(
            &instruction_event(0, 1, decoded_addi()),
            &layout,
            &state,
        )
        .expect("second event should validate")
        .expect("second event should map to a trace instruction");
        assert!(
            update_progress_for_instruction(
                &mut state,
                second.hart_id,
                second.local_index,
                &layout,
                2
            )
            .is_empty()
        );

        let third = validate_streaming_instruction_event(
            &instruction_event(0, 2, decoded_addi()),
            &layout,
            &state,
        )
        .expect("third event should validate")
        .expect("third event should map to a trace instruction");
        assert_eq!(
            update_progress_for_instruction(
                &mut state,
                third.hart_id,
                third.local_index,
                &layout,
                2
            ),
            vec![0]
        );
    }

    #[test]
    fn finishing_flushes_the_remaining_trace_prefix() {
        let seqs = vec![
            InstructionSequence::new(decoded_addi()),
            InstructionSequence::new(decoded_addi()),
        ];
        let layout = hart_index_layout(&vec![(0, seqs)]);
        let mut state = ImplStreamState::default();

        for local_index in 0..2 {
            let validated = validate_streaming_instruction_event(
                &instruction_event(0, local_index, decoded_addi()),
                &layout,
                &state,
            )
            .expect("stream event should validate")
            .expect("stream event should map to a trace instruction");
            let _ = update_progress_for_instruction(
                &mut state,
                validated.hart_id,
                validated.local_index,
                &layout,
                2,
            );
        }

        assert_eq!(
            flush_finished_impl_progress(&mut state, &layout),
            vec![0, 1]
        );
    }

    #[test]
    fn validation_rejects_out_of_range_trace_inst_id() {
        let seqs = vec![InstructionSequence::new(decoded_addi())];
        let layout = hart_index_layout(&vec![(0, seqs)]);
        let state = ImplStreamState::default();

        let error = validate_streaming_instruction_event(
            &instruction_event(0, 1, decoded_addi()),
            &layout,
            &state,
        )
        .expect_err("out-of-range local index must be rejected");
        assert!(error.contains("missing trace instruction 1"));
    }

    #[test]
    fn validation_rejects_decreasing_trace_indices() {
        let seqs = vec![
            InstructionSequence::new(decoded_addi()),
            InstructionSequence::new(decoded_addi()),
        ];
        let layout = hart_index_layout(&vec![(0, seqs)]);
        let mut state = ImplStreamState::default();
        let validated = validate_streaming_instruction_event(
            &instruction_event(0, 1, decoded_addi()),
            &layout,
            &state,
        )
        .expect("first event should validate")
        .expect("first event should map to a trace instruction");
        let _ = update_progress_for_instruction(
            &mut state,
            validated.hart_id,
            validated.local_index,
            &layout,
            1,
        );

        let error = validate_streaming_instruction_event(
            &instruction_event(0, 0, decoded_addi()),
            &layout,
            &state,
        )
        .expect_err("trace-index regressions must be rejected");
        assert!(error.contains("regressed from trace instruction 1 back to 0"));
    }

    #[test]
    fn trim_stream_state_discards_later_trace_rows() {
        let seqs = vec![
            InstructionSequence::new(decoded_addi()),
            InstructionSequence::new(decoded_addi()),
            InstructionSequence::new(decoded_addi()),
        ];
        let _layout = hart_index_layout(&vec![(0, seqs)]);
        let mut state = ImplStreamState::default();
        state.collector.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![
                    asm_for_source(0, 0, 0x1000, decoded_addi()),
                    asm_for_source(0, 1, 0x1004, decoded_addi()),
                    asm_for_source(0, 2, 0x1008, decoded_addi()),
                ],
            },
        );

        let keep = compared_prefix_keep_limits(
            &BTreeMap::from([(0, 1usize)]),
            StreamDivergencePoint {
                hart_id: 0,
                local_index: 1,
            },
        );
        trim_stream_state_after_divergence(&mut state, &keep);
        assert_eq!(state.collector.per_hart[&0].instructions.len(), 2);
        assert_eq!(
            state.collector.per_hart[&0].instructions[1]
                .trace_inst_id
                .expect("second row should still map")
                .local_index,
            1
        );
    }

    fn minimal_testcase() -> GeneratedTestCase {
        GeneratedTestCase {
            config: TestCaseConfig {
                isa_base: ISABase::Rv32,
                extension_insts_count: 0,
                extension_inst_scaling: None,
                mem_size: 64,
                float_extension: Default::default(),
                unaligned_access_required: Some(false),
                random_config: RandomConfig {
                    memory_access_config: MemoryAccessConfig {
                        memory_base_address_range: MemoryAccessConfig::default()
                            .memory_base_address_range,
                        memory_offset_immediate_range: (0, 0),
                        memory_base_register_number_range: (10, 15),
                    },
                    register_operand_config: RegisterOperandConfig {
                        integer_register_number_range: (16, 25),
                        floating_point_register_number_range: (16, 25),
                        vector_register_number_range: (16, 25),
                    },
                    ..Default::default()
                },
                seed: None,
                hart_config: HartConfig::Single,
            },
            harts: vec![HartProgram {
                hart_id: 0,
                init: Vec::new(),
                test: vec![InstructionSequence::new(decoded_addi())],
                shared_init_sequence_indices: BTreeSet::new(),
                shared_test_sequence_indices: BTreeSet::new(),
            }],
            mem_range: HashMap::from([("spike".to_string(), (0x1_0000, 0x1_003f))]),
            extensions: IsaExtensions::rv32(vec![RV32Extensions::I]),
            multicore_sync_metrics: None,
        }
    }

    fn two_hart_testcase() -> GeneratedTestCase {
        GeneratedTestCase {
            config: TestCaseConfig {
                hart_config: HartConfig::Dual {
                    sync: Default::default(),
                },
                ..minimal_testcase().config
            },
            harts: vec![
                HartProgram {
                    hart_id: 0,
                    init: Vec::new(),
                    test: vec![InstructionSequence::new(decoded_addi())],
                    shared_init_sequence_indices: BTreeSet::new(),
                    shared_test_sequence_indices: BTreeSet::new(),
                },
                HartProgram {
                    hart_id: 1,
                    init: Vec::new(),
                    test: vec![InstructionSequence::new(decoded_addi())],
                    shared_init_sequence_indices: BTreeSet::new(),
                    shared_test_sequence_indices: BTreeSet::new(),
                },
            ],
            mem_range: HashMap::from([
                ("spike".to_string(), (0x1_0000, 0x1_003f)),
                ("rocket".to_string(), (0x1_0000, 0x1_003f)),
            ]),
            ..minimal_testcase()
        }
    }

    #[test]
    fn refresh_context_normalizes_same_shape_streamed_source_to_testcase() {
        let testcase = minimal_testcase();
        let expected = InstructionSequence::new(decoded_addi());
        let conflicting = InstructionSequence::new(decoded_addi_x2());
        let per_hart_typed = vec![(0, vec![expected.clone()])];
        let mut state = ImplStreamState::default();
        state.collector.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![asm_for_source(
                    0,
                    0,
                    0x1000,
                    conflicting.instruction.clone(),
                )],
            },
        );
        state.collector.perf = Some(Default::default());

        refresh_context(&testcase, "spike", &per_hart_typed, &mut state)
            .expect("mismatched streamed source should be normalized to typed input");
        assert!(state.context.is_some());
    }

    #[test]
    fn advance_streaming_compare_detects_first_divergence_once_all_impls_are_sealed() {
        let mut testcase = minimal_testcase();
        testcase
            .mem_range
            .insert("rocket".to_string(), (0x1_0000, 0x1_003f));
        testcase
            .config
            .random_config
            .register_operand_config
            .integer_register_number_range = (1, 25);
        let per_hart_typed = vec![(0, vec![InstructionSequence::new(decoded_addi())])];
        let cases = HashMap::from([
            (
                "spike".to_string(),
                StreamingExecutionCase {
                    testcase: testcase.clone(),
                    per_hart_typed: per_hart_typed.clone(),
                },
            ),
            (
                "rocket".to_string(),
                StreamingExecutionCase {
                    testcase: testcase.clone(),
                    per_hart_typed: per_hart_typed.clone(),
                },
            ),
        ]);

        let mut spike_state = ImplStreamState::default();
        let mut spike_row = asm_for_source(0, 0, 0x1000, decoded_addi());
        spike_row
            .register_writes
            .push(riscv_fuzz_plugin_api::RegisterValue {
                name: "x1".to_string(),
                value: 1,
            });
        spike_state.collector.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![spike_row],
            },
        );
        spike_state.collector.perf = Some(Default::default());
        spike_state.progress_by_hart.insert(
            0,
            HartStreamingProgress {
                seen_trace_local_indices: BTreeSet::from([0]),
                last_observed_local_index: Some(0),
                sealed_local_prefix_len: 1,
            },
        );
        refresh_context(&testcase, "spike", &per_hart_typed, &mut spike_state)
            .expect("spike context should refresh");

        let mut rocket_state = ImplStreamState::default();
        let mut rocket_row = asm_for_source(0, 0, 0x1000, decoded_addi());
        rocket_row
            .register_writes
            .push(riscv_fuzz_plugin_api::RegisterValue {
                name: "x1".to_string(),
                value: 2,
            });
        rocket_state.collector.per_hart.insert(
            0,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![rocket_row],
            },
        );
        rocket_state.collector.perf = Some(Default::default());
        rocket_state.progress_by_hart.insert(
            0,
            HartStreamingProgress {
                seen_trace_local_indices: BTreeSet::from([0]),
                last_observed_local_index: Some(0),
                sealed_local_prefix_len: 1,
            },
        );
        refresh_context(&testcase, "rocket", &per_hart_typed, &mut rocket_state)
            .expect("rocket context should refresh");

        let mut states = HashMap::from([
            ("spike".to_string(), spike_state),
            ("rocket".to_string(), rocket_state),
        ]);
        let mut next_compare_by_hart = BTreeMap::from([(0, 0usize)]);
        let impl_order = vec!["spike".to_string(), "rocket".to_string()];

        let divergence = advance_streaming_compare(
            &testcase,
            &impl_order,
            &cases,
            &mut states,
            &mut next_compare_by_hart,
            &BTreeMap::from([(0, 1usize)]),
        )
        .expect("streaming compare should run");
        assert_eq!(
            divergence,
            Some(StreamDivergencePoint {
                hart_id: 0,
                local_index: 0,
            })
        );
    }

    #[test]
    fn advance_streaming_compare_allows_hart1_to_diverge_before_hart0_is_ready() {
        let mut testcase = two_hart_testcase();
        testcase
            .config
            .random_config
            .register_operand_config
            .integer_register_number_range = (1, 25);
        let per_hart_typed = vec![
            (0, vec![InstructionSequence::new(decoded_addi())]),
            (1, vec![InstructionSequence::new(decoded_addi())]),
        ];
        let cases = HashMap::from([
            (
                "spike".to_string(),
                StreamingExecutionCase {
                    testcase: testcase.clone(),
                    per_hart_typed: per_hart_typed.clone(),
                },
            ),
            (
                "rocket".to_string(),
                StreamingExecutionCase {
                    testcase: testcase.clone(),
                    per_hart_typed: per_hart_typed.clone(),
                },
            ),
        ]);

        let mut spike_state = ImplStreamState::default();
        let mut spike_row = asm_for_source(1, 0, 0x2000, decoded_addi());
        spike_row
            .register_writes
            .push(riscv_fuzz_plugin_api::RegisterValue {
                name: "x1".to_string(),
                value: 1,
            });
        spike_state.collector.per_hart.insert(
            1,
            HartExecutionOutput {
                hart_id: 1,
                instructions: vec![spike_row],
            },
        );
        spike_state.collector.perf = Some(Default::default());
        spike_state.progress_by_hart.insert(
            1,
            HartStreamingProgress {
                seen_trace_local_indices: BTreeSet::from([0]),
                last_observed_local_index: Some(0),
                sealed_local_prefix_len: 1,
            },
        );
        refresh_context(&testcase, "spike", &per_hart_typed, &mut spike_state)
            .expect("spike context should refresh");

        let mut rocket_state = ImplStreamState::default();
        let mut rocket_row = asm_for_source(1, 0, 0x2000, decoded_addi());
        rocket_row
            .register_writes
            .push(riscv_fuzz_plugin_api::RegisterValue {
                name: "x1".to_string(),
                value: 2,
            });
        rocket_state.collector.per_hart.insert(
            1,
            HartExecutionOutput {
                hart_id: 1,
                instructions: vec![rocket_row],
            },
        );
        rocket_state.collector.perf = Some(Default::default());
        rocket_state.progress_by_hart.insert(
            1,
            HartStreamingProgress {
                seen_trace_local_indices: BTreeSet::from([0]),
                last_observed_local_index: Some(0),
                sealed_local_prefix_len: 1,
            },
        );
        refresh_context(&testcase, "rocket", &per_hart_typed, &mut rocket_state)
            .expect("rocket context should refresh");

        let mut states = HashMap::from([
            ("spike".to_string(), spike_state),
            ("rocket".to_string(), rocket_state),
        ]);
        let mut next_compare_by_hart = BTreeMap::from([(0, 0usize), (1, 0usize)]);
        let impl_order = vec!["spike".to_string(), "rocket".to_string()];

        let divergence = advance_streaming_compare(
            &testcase,
            &impl_order,
            &cases,
            &mut states,
            &mut next_compare_by_hart,
            &BTreeMap::from([(0, 1usize), (1, 1usize)]),
        )
        .expect("streaming compare should run");
        assert_eq!(
            divergence,
            Some(StreamDivergencePoint {
                hart_id: 1,
                local_index: 0,
            })
        );
    }
}
