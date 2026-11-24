use crate::artifact_io::{
    ArtifactIoError, ensure_parent_dir, write_json_pretty, write_string_file,
};
use crate::context_extract::{
    ContextExtractionError, extract_memory_context, extract_register_context,
};
use riscv_fuzz_plugin_api::context::{
    AsmInstructionContextOutput, ExecutionContextOutput, HartExecutionContextOutput,
    InstructionContext, LogicalInstructionContextOutput,
};
use riscv_fuzz_plugin_api::{
    AsmInstructionExecution, ExceptionCause, ExecutionOutput, HartExecutionOutput, ISABase,
    InstructionTiming, MemValue, RegisterValue, TraceInstId, canonicalize_register_name,
};
use riscv_instruction::MemoryAccessInstruction;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::io::Write;
use std::path::Path;
use thiserror::Error;

/// Errors while building rich execution contexts from raw asm traces.
#[derive(Debug, Error)]
pub enum ContextBuildError {
    #[error(
        "instruction metadata length mismatch: instructions={instructions}, metadata={metadata}"
    )]
    InstructionMetadataLengthMismatch {
        instructions: usize,
        metadata: usize,
    },

    #[error(
        "hart {hart_id} raw instruction maps to missing logical instruction index {instruction_index} (available={available})"
    )]
    InvalidLogicalInstructionMapping {
        hart_id: u32,
        instruction_index: usize,
        available: usize,
    },

    #[error(
        "hart {hart_id} missing raw asm instructions for logical instruction index {instruction_index}"
    )]
    MissingLogicalInstructionExecution {
        hart_id: u32,
        instruction_index: usize,
    },

    #[error(
        "hart {hart_id} logical instruction index {instruction_index} contains duplicate register write for {name}"
    )]
    DuplicateLogicalRegisterWrite {
        hart_id: u32,
        instruction_index: usize,
        name: String,
    },

    #[error(
        "hart {hart_id} logical instruction index {instruction_index} contains duplicate memory byte write at addr 0x{addr:x}"
    )]
    DuplicateLogicalMemoryWrite {
        hart_id: u32,
        instruction_index: usize,
        addr: u64,
    },

    #[error(transparent)]
    ContextExtraction(#[from] ContextExtractionError),
}

#[derive(Debug, Error)]
pub enum TimelineRenderError {
    #[error("failed to render execution timeline: {message}")]
    Custom { message: String },
}

#[derive(Debug, Error)]
pub enum ExecutionArtifactError {
    #[error(transparent)]
    Io(#[from] ArtifactIoError),
    #[error("report generation failed: {0}")]
    Report(#[source] Box<dyn std::error::Error>),
}

#[derive(Debug, Clone)]
struct HartTraceableInput {
    base_index: usize,
    instructions: Vec<riscv_instruction::RiscvInstruction>,
    offsets: Vec<Option<i64>>,
}

#[derive(Debug, Clone)]
struct LogicalInstructionAccumulator {
    local_instruction_index: usize,
    instruction: riscv_instruction::RiscvInstruction,
    trace_inst_id: TraceInstId,
    offset: Option<i64>,
    asm_instruction_indices: Vec<usize>,
    first_context: Option<InstructionContext>,
    exceptions: Vec<ExceptionCause>,
    register_writes: HashMap<String, u64>,
    memory_writes: BTreeMap<u64, u8>,
    timings: Vec<InstructionTiming>,
}

fn format_instruction_timing(timing: Option<&InstructionTiming>) -> String {
    let Some(timing) = timing else {
        return "-".to_string();
    };
    if !timing.is_recorded() {
        return "-".to_string();
    }

    let start = timing
        .clk_start
        .map(|v| v.to_string())
        .unwrap_or_else(|| "?".to_string());
    let end = timing
        .clk_end
        .map(|v| v.to_string())
        .unwrap_or_else(|| "?".to_string());
    let span = timing
        .clk_span
        .map(|v| v.to_string())
        .unwrap_or_else(|| "?".to_string());
    format!("{start}/{end}/{span}")
}

fn escape_markdown_cell(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

fn format_trace_slot(trace_inst_id: Option<TraceInstId>) -> String {
    match trace_inst_id {
        Some(trace) => format!("hart{}#{}", trace.hart_id, trace.local_index),
        None => "-".to_string(),
    }
}

fn format_register_writes(register_writes: &[RegisterValue]) -> String {
    if register_writes.is_empty() {
        return "-".to_string();
    }

    let mut writes = register_writes.to_vec();
    writes.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.value.cmp(&b.value)));
    writes
        .into_iter()
        .map(|write| format!("{}=0x{:x}", write.name, write.value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_memory_writes(memory_writes: &[MemValue]) -> String {
    if memory_writes.is_empty() {
        return "-".to_string();
    }

    let mut writes = memory_writes.to_vec();
    writes.sort_by_key(|write| write.addr);
    writes
        .into_iter()
        .map(|write| format!("0x{:x}=0x{:02x}", write.addr, write.value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_exception_causes(exceptions: &[ExceptionCause]) -> String {
    if exceptions.is_empty() {
        "-".to_string()
    } else {
        exceptions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Context calculator that accumulates register and memory changes
/// to compute the execution context at any point in time.
#[derive(Debug, Clone, Default)]
struct ContextCalculator {
    registers: HashMap<String, u64>,
    memory: BTreeMap<u64, u8>,
}

impl ContextCalculator {
    fn apply_register_changes(&mut self, changes: &[RegisterValue]) {
        for change in changes {
            if let Some(canonical) = canonicalize_register_name(&change.name) {
                self.registers.insert(canonical, change.value);
            }
        }
    }

    fn apply_memory_changes(&mut self, changes: &[MemValue]) {
        for change in changes {
            self.memory.insert(change.addr, change.value);
        }
    }
}

fn build_instruction_context(
    instruction: &riscv_instruction::RiscvInstruction,
    offset_hint: Option<i64>,
    calculator: &ContextCalculator,
    isa_base: ISABase,
    mem_range: (u64, u64),
) -> Result<InstructionContext, ContextBuildError> {
    let registers_before = calculator.registers.clone();
    let memory_before = calculator.memory.clone();

    let register_context =
        extract_register_context(instruction, &registers_before)?.unwrap_or_default();
    let memory_context = extract_memory_context(
        instruction,
        offset_hint,
        &registers_before,
        &memory_before,
        isa_base,
        mem_range,
    )?
    .unwrap_or_default();

    Ok(InstructionContext {
        registers_before,
        memory_before,
        register_context,
        memory_context,
    })
}

fn build_hart_traceable_inputs(
    per_hart_input_instructions: &[(
        u32,
        Vec<riscv_instruction::InstructionSequence<riscv_instruction::RiscvInstruction>>,
    )],
) -> Result<BTreeMap<u32, HartTraceableInput>, ContextBuildError> {
    let mut base_index = 0usize;
    let mut out = BTreeMap::new();
    for (hart_id, input_sequences) in per_hart_input_instructions.iter() {
        let mut instructions = Vec::new();
        let mut offsets = Vec::new();

        for sequence in input_sequences {
            for instruction in sequence
                .pre_instructions
                .iter()
                .chain(std::iter::once(&sequence.instruction))
                .chain(sequence.post_instructions.iter())
            {
                offsets.push(instruction.offset_operand_value());
                instructions.push(instruction.clone());
            }
        }

        let count = instructions.len();
        out.insert(
            *hart_id,
            HartTraceableInput {
                base_index,
                instructions,
                offsets,
            },
        );
        base_index += count;
    }

    Ok(out)
}

fn build_context_output_for_rows<'a>(
    hart_id: u32,
    rows: impl IntoIterator<Item = &'a AsmInstructionExecution>,
    isa_base: ISABase,
    mem_range: (u64, u64),
) -> Result<HartExecutionContextOutput, ContextBuildError> {
    let mut calculator = ContextCalculator::default();
    let mut instructions = Vec::new();

    for raw in rows {
        let context =
            build_instruction_context(&raw.instruction, None, &calculator, isa_base, mem_range)?;
        instructions.push(AsmInstructionContextOutput {
            pc: raw.pc,
            instruction: raw.instruction.clone(),
            source_annotation: raw.source_annotation.clone(),
            trace_inst_id: raw.trace_inst_id,
            exceptions: raw.exceptions.clone(),
            register_writes: raw.register_writes.clone(),
            memory_writes: raw.memory_writes.clone(),
            timing: raw.timing.clone(),
            context,
        });
        calculator.apply_register_changes(&raw.register_writes);
        calculator.apply_memory_changes(&raw.memory_writes);
    }

    Ok(HartExecutionContextOutput {
        hart_id,
        instructions,
    })
}

fn build_source_mapped_hart_context_output(
    hart_output: &HartExecutionOutput,
    isa_base: ISABase,
    mem_range: (u64, u64),
) -> Result<HartExecutionContextOutput, ContextBuildError> {
    let rows = hart_output
        .instructions
        .iter()
        .filter(|asm| asm.trace_inst_id.is_some());

    build_context_output_for_rows(hart_output.hart_id, rows, isa_base, mem_range)
}

fn build_framework_hart_context_output(
    hart_output: &HartExecutionOutput,
    isa_base: ISABase,
    mem_range: (u64, u64),
) -> Result<HartExecutionContextOutput, ContextBuildError> {
    let rows = hart_output
        .instructions
        .iter()
        .filter(|asm| asm.trace_inst_id.is_none());
    build_context_output_for_rows(hart_output.hart_id, rows, isa_base, mem_range)
}

fn framework_end_context(hart_output: &HartExecutionContextOutput) -> ContextCalculator {
    let mut calculator = ContextCalculator::default();
    for instruction in &hart_output.instructions {
        calculator.apply_register_changes(&instruction.register_writes);
        calculator.apply_memory_changes(&instruction.memory_writes);
    }
    calculator
}

fn merge_framework_state_into_context(
    context: &mut InstructionContext,
    framework_context: &ContextCalculator,
    mem_range: (u64, u64),
) {
    for (name, value) in &framework_context.registers {
        context
            .registers_before
            .entry(name.clone())
            .or_insert(*value);
    }

    let normalized_mem_max = mem_range.1.checked_sub(mem_range.0);
    for (addr, value) in &framework_context.memory {
        let is_normalized_program_addr = normalized_mem_max.is_some_and(|max| *addr <= max);
        if is_normalized_program_addr {
            context.memory_before.entry(*addr).or_insert(*value);
        }
    }
}

fn recorded_timing_span(timing: &InstructionTiming) -> Option<u64> {
    timing
        .clk_span
        .or_else(|| match (timing.clk_start, timing.clk_end) {
            (Some(start), Some(end)) if end >= start => Some(end - start + 1),
            _ => None,
        })
}

fn aggregate_instruction_timings(timings: &[InstructionTiming]) -> InstructionTiming {
    let recorded: Vec<&InstructionTiming> = timings
        .iter()
        .filter(|timing| timing.is_recorded())
        .collect();
    if recorded.is_empty() {
        return InstructionTiming::default();
    }

    let clk_start = recorded.iter().filter_map(|timing| timing.clk_start).min();
    let clk_end = recorded.iter().filter_map(|timing| timing.clk_end).max();
    let summed_span: u64 = recorded
        .iter()
        .filter_map(|timing| recorded_timing_span(timing))
        .sum();

    let clk_span = match (clk_start, clk_end) {
        (Some(start), Some(end)) if end >= start => Some(end - start + 1),
        _ if summed_span > 0 => Some(summed_span),
        _ => None,
    };

    InstructionTiming::new(clk_start, clk_end, clk_span)
}

fn build_logical_instruction_context(
    instruction: &riscv_instruction::RiscvInstruction,
    offset_hint: Option<i64>,
    base_context: &InstructionContext,
    isa_base: ISABase,
    mem_range: (u64, u64),
) -> Result<InstructionContext, ContextBuildError> {
    let registers_before = base_context.registers_before.clone();
    let memory_before = base_context.memory_before.clone();
    let register_context =
        extract_register_context(instruction, &registers_before)?.unwrap_or_default();
    let memory_context = extract_memory_context(
        instruction,
        offset_hint,
        &registers_before,
        &memory_before,
        isa_base,
        mem_range,
    )?
    .unwrap_or_default();

    Ok(InstructionContext {
        registers_before,
        memory_before,
        register_context,
        memory_context,
    })
}

fn derive_logical_view_for_hart(
    hart_output: &HartExecutionContextOutput,
    framework_hart_output: &HartExecutionContextOutput,
    hart_input: &HartTraceableInput,
    isa_base: ISABase,
    mem_range: (u64, u64),
    allow_missing_logical_instructions: bool,
) -> Result<Vec<LogicalInstructionContextOutput>, ContextBuildError> {
    let mut accumulators = hart_input
        .instructions
        .iter()
        .enumerate()
        .map(|(local_index, instruction)| LogicalInstructionAccumulator {
            local_instruction_index: local_index,
            instruction: instruction.clone(),
            trace_inst_id: TraceInstId::new(hart_output.hart_id, local_index),
            offset: hart_input.offsets.get(local_index).copied().flatten(),
            asm_instruction_indices: Vec::new(),
            first_context: None,
            exceptions: Vec::new(),
            register_writes: HashMap::new(),
            memory_writes: BTreeMap::new(),
            timings: Vec::new(),
        })
        .collect::<Vec<_>>();

    let framework_context = framework_end_context(framework_hart_output);

    for (raw_index, instruction) in hart_output.instructions.iter().enumerate() {
        let Some(trace_inst_id) = instruction.trace_inst_id else {
            continue;
        };
        if trace_inst_id.hart_id != hart_output.hart_id {
            return Err(ContextBuildError::InvalidLogicalInstructionMapping {
                hart_id: hart_output.hart_id,
                instruction_index: trace_inst_id.local_index,
                available: accumulators.len(),
            });
        }
        let logical_index = trace_inst_id.local_index;

        let Some(acc) = accumulators.get_mut(logical_index) else {
            return Err(ContextBuildError::InvalidLogicalInstructionMapping {
                hart_id: hart_output.hart_id,
                instruction_index: logical_index,
                available: accumulators.len(),
            });
        };

        if acc.first_context.is_none() {
            let mut merged_context = instruction.context.clone();
            merge_framework_state_into_context(&mut merged_context, &framework_context, mem_range);
            acc.first_context = Some(merged_context);
        }
        acc.asm_instruction_indices.push(raw_index);
        acc.exceptions.extend(instruction.exceptions.clone());
        for reg in &instruction.register_writes {
            // Multiple raw PCs may belong to the same logical source instruction; keep
            // the last architectural value observed within that instruction.
            acc.register_writes.insert(reg.name.clone(), reg.value);
        }
        for mem in &instruction.memory_writes {
            // Likewise for memory bytes: the logical instruction should expose the
            // final byte value after all constituent raw PCs retire.
            acc.memory_writes.insert(mem.addr, mem.value);
        }
        if instruction.timing.is_recorded() {
            acc.timings.push(instruction.timing.clone());
        }
    }

    let mut logical_view = Vec::with_capacity(accumulators.len());
    for acc in accumulators {
        let Some(first_context) = acc.first_context.as_ref() else {
            if allow_missing_logical_instructions {
                logical_view.push(LogicalInstructionContextOutput {
                    hart_id: hart_output.hart_id,
                    instruction_index: hart_input.base_index + acc.local_instruction_index,
                    local_instruction_index: acc.local_instruction_index,
                    instruction: acc.instruction,
                    trace_inst_id: acc.trace_inst_id,
                    asm_instruction_indices: Vec::new(),
                    exceptions: Vec::new(),
                    register_writes: Vec::new(),
                    memory_writes: Vec::new(),
                    timing: InstructionTiming::default(),
                    context: InstructionContext::default(),
                });
                continue;
            }
            return Err(ContextBuildError::MissingLogicalInstructionExecution {
                hart_id: hart_output.hart_id,
                instruction_index: hart_input.base_index + acc.local_instruction_index,
            });
        };

        let context = build_logical_instruction_context(
            &acc.instruction,
            acc.offset,
            first_context,
            isa_base,
            mem_range,
        )?;

        let mut register_writes = acc
            .register_writes
            .into_iter()
            .map(|(name, value)| RegisterValue { name, value })
            .collect::<Vec<_>>();
        register_writes.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.value.cmp(&b.value)));

        let memory_writes = acc
            .memory_writes
            .into_iter()
            .map(|(addr, value)| MemValue { addr, value })
            .collect::<Vec<_>>();

        logical_view.push(LogicalInstructionContextOutput {
            hart_id: hart_output.hart_id,
            instruction_index: hart_input.base_index + acc.local_instruction_index,
            local_instruction_index: acc.local_instruction_index,
            trace_inst_id: acc.trace_inst_id,
            instruction: acc.instruction,
            asm_instruction_indices: acc.asm_instruction_indices,
            exceptions: acc.exceptions,
            register_writes,
            memory_writes,
            timing: aggregate_instruction_timings(&acc.timings),
            context,
        });
    }

    Ok(logical_view)
}

fn build_execution_context_output_with_missing_policy(
    output: ExecutionOutput,
    mem_range: (u64, u64),
    per_hart_input_instructions: &[(
        u32,
        Vec<riscv_instruction::InstructionSequence<riscv_instruction::RiscvInstruction>>,
    )],
    allow_missing_logical_instructions: bool,
) -> Result<ExecutionContextOutput, ContextBuildError> {
    let ExecutionOutput {
        per_hart,
        core_id,
        isa_base,
        perf,
    } = output;
    let hart_inputs = build_hart_traceable_inputs(per_hart_input_instructions)?;

    let mut hart_ids: BTreeSet<u32> = BTreeSet::new();
    hart_ids.extend(hart_inputs.keys().copied());
    hart_ids.extend(per_hart.keys().copied());

    let mut per_hart_context = BTreeMap::new();
    let mut framework_per_hart_context = BTreeMap::new();
    let mut logical_view = Vec::new();

    for hart_id in hart_ids {
        let hart_output = per_hart
            .get(&hart_id)
            .cloned()
            .unwrap_or_else(|| HartExecutionOutput::empty(hart_id));
        let hart_context =
            build_source_mapped_hart_context_output(&hart_output, isa_base, mem_range)?;
        let template_raw_hart_context =
            build_framework_hart_context_output(&hart_output, isa_base, mem_range)?;
        if let Some(hart_input) = hart_inputs.get(&hart_id) {
            logical_view.extend(derive_logical_view_for_hart(
                &hart_context,
                &template_raw_hart_context,
                hart_input,
                isa_base,
                mem_range,
                allow_missing_logical_instructions,
            )?);
        }
        per_hart_context.insert(hart_id, hart_context);
        framework_per_hart_context.insert(hart_id, template_raw_hart_context);
    }

    logical_view.sort_by_key(|entry| entry.instruction_index);

    Ok(ExecutionContextOutput {
        per_hart: per_hart_context,
        framework_per_hart: framework_per_hart_context,
        logical_view,
        mem_range,
        core_id,
        isa_base,
        perf,
    })
}

pub fn build_execution_context_output(
    output: ExecutionOutput,
    mem_range: (u64, u64),
    per_hart_input_instructions: &[(
        u32,
        Vec<riscv_instruction::InstructionSequence<riscv_instruction::RiscvInstruction>>,
    )],
) -> Result<ExecutionContextOutput, ContextBuildError> {
    build_execution_context_output_with_missing_policy(
        output,
        mem_range,
        per_hart_input_instructions,
        false,
    )
}

pub fn build_partial_execution_context_output(
    output: ExecutionOutput,
    mem_range: (u64, u64),
    per_hart_input_instructions: &[(
        u32,
        Vec<riscv_instruction::InstructionSequence<riscv_instruction::RiscvInstruction>>,
    )],
) -> Result<ExecutionContextOutput, ContextBuildError> {
    build_execution_context_output_with_missing_policy(
        output,
        mem_range,
        per_hart_input_instructions,
        true,
    )
}

fn resolve_timing_bounds(timing: &InstructionTiming) -> Option<(u64, u64, u64)> {
    if !timing.is_recorded() {
        return None;
    }

    if let (Some(start), Some(end)) = (timing.clk_start, timing.clk_end) {
        if end < start {
            return None;
        }
        return Some((start, end, end - start + 1));
    }

    if let (Some(start), Some(span)) = (timing.clk_start, timing.clk_span) {
        if span == 0 {
            return None;
        }
        let end = start.checked_add(span - 1)?;
        return Some((start, end, span));
    }

    if let (Some(end), Some(span)) = (timing.clk_end, timing.clk_span) {
        if span == 0 {
            return None;
        }
        let start = end.checked_sub(span - 1)?;
        return Some((start, end, span));
    }

    None
}

fn escape_svg_text(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn nice_tick_step(span: u64, target_tick_count: u64) -> u64 {
    let target_tick_count = target_tick_count.max(1);
    let rough = ((span.max(1) as f64) / target_tick_count as f64).max(1.0);
    let exponent = rough.log10().floor();
    let base = 10f64.powf(exponent);

    for multiplier in [1.0, 2.0, 5.0, 10.0] {
        let candidate = (base * multiplier).round() as u64;
        if candidate as f64 >= rough {
            return candidate.max(1);
        }
    }

    1
}

fn timeline_pixels_per_cycle(cycle_span: u64) -> f64 {
    let cycle_span = cycle_span.max(1) as f64;
    (4200.0 / cycle_span).clamp(0.5, 10.0)
}

fn render_raw_execution_timeline_svg(
    output: &ExecutionOutput,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut lanes = Vec::new();
    let mut global_start = None::<u64>;
    let mut global_end = None::<u64>;

    for (&hart_id, hart_output) in &output.per_hart {
        let entries = hart_output
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(raw_index, instruction)| {
                let (clk_start, clk_end, clk_span) = resolve_timing_bounds(&instruction.timing)?;
                global_start = Some(
                    global_start
                        .map(|value| value.min(clk_start))
                        .unwrap_or(clk_start),
                );
                global_end = Some(
                    global_end
                        .map(|value| value.max(clk_end))
                        .unwrap_or(clk_end),
                );
                Some((raw_index, instruction, clk_start, clk_end, clk_span))
            })
            .collect::<Vec<_>>();
        lanes.push((hart_id, entries));
    }

    let Some(global_start) = global_start else {
        return Err(Box::new(TimelineRenderError::Custom {
            message: "no recorded timing windows in execution output".to_string(),
        }));
    };
    let global_end = global_end.expect("global_start exists when global_end is collected");
    let cycle_span = global_end.saturating_sub(global_start).saturating_add(1);
    let pixels_per_cycle = timeline_pixels_per_cycle(cycle_span);
    let left_margin = 140.0;
    let right_margin = 36.0;
    let top_margin = 48.0;
    let lane_height = 72.0;
    let lane_gap = 18.0;
    let block_height = 34.0;
    let axis_height = 24.0;
    let width = left_margin + right_margin + (cycle_span as f64 * pixels_per_cycle);
    let height = top_margin
        + axis_height
        + lanes.len() as f64 * lane_height
        + lanes.len().saturating_sub(1) as f64 * lane_gap
        + 28.0;

    let mut svg = String::new();
    writeln!(
        &mut svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.1}\" height=\"{height:.1}\" viewBox=\"0 0 {width:.1} {height:.1}\">"
    )?;
    writeln!(
        &mut svg,
        "<style>.title{{font:600 16px sans-serif;fill:#16202a}} .axis{{font:11px monospace;fill:#4b5563}} .label{{font:12px monospace;fill:#16202a}} .inst{{font:10px monospace;fill:#16202a}} .empty{{font:12px sans-serif;fill:#6b7280}}</style>"
    )?;
    writeln!(
        &mut svg,
        "<text class=\"title\" x=\"24\" y=\"28\">{} Raw Execution Timeline</text>",
        escape_svg_text(&output.core_id)
    )?;

    let axis_y = top_margin;
    writeln!(
        &mut svg,
        "<line x1=\"{left_margin:.1}\" y1=\"{axis_y:.1}\" x2=\"{:.1}\" y2=\"{axis_y:.1}\" stroke=\"#cbd5e1\" stroke-width=\"1\"/>",
        width - right_margin
    )?;
    let tick_step = nice_tick_step(cycle_span, 10);
    let mut tick = global_start;
    while tick <= global_end {
        let x = left_margin + (tick.saturating_sub(global_start) as f64 * pixels_per_cycle);
        writeln!(
            &mut svg,
            "<line x1=\"{x:.1}\" y1=\"{axis_y:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\" stroke=\"#cbd5e1\" stroke-width=\"1\"/>",
            axis_y + 8.0
        )?;
        writeln!(
            &mut svg,
            "<text class=\"axis\" x=\"{x:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{}</text>",
            axis_y - 8.0,
            tick
        )?;
        if global_end - tick < tick_step {
            break;
        }
        tick += tick_step;
    }

    for (lane_idx, (hart_id, entries)) in lanes.iter().enumerate() {
        let lane_top = top_margin + axis_height + lane_idx as f64 * (lane_height + lane_gap);
        let lane_center_y = lane_top + lane_height / 2.0;
        writeln!(
            &mut svg,
            "<text class=\"label\" x=\"24\" y=\"{:.1}\">hart {hart_id}</text>",
            lane_center_y + 4.0
        )?;
        writeln!(
            &mut svg,
            "<rect x=\"{left_margin:.1}\" y=\"{lane_top:.1}\" width=\"{:.1}\" height=\"{lane_height:.1}\" rx=\"10\" fill=\"#f8fafc\" stroke=\"#e2e8f0\"/>",
            width - left_margin - right_margin
        )?;

        if entries.is_empty() {
            writeln!(
                &mut svg,
                "<text class=\"empty\" x=\"{:.1}\" y=\"{:.1}\">No recorded timing windows</text>",
                left_margin + 16.0,
                lane_center_y + 4.0
            )?;
            continue;
        }

        for (raw_index, instruction, clk_start, clk_end, clk_span) in entries {
            let start_x =
                left_margin + (clk_start.saturating_sub(global_start) as f64 * pixels_per_cycle);
            let end_x =
                left_margin + (clk_end.saturating_sub(global_start) as f64 * pixels_per_cycle);
            let rect_x = start_x;
            let rect_width = ((end_x - start_x) + pixels_per_cycle).max(10.0);
            let rect_y = lane_top + (lane_height - block_height) / 2.0;
            let color = if instruction.trace_inst_id.is_some() {
                "#dbeafe"
            } else {
                "#fef3c7"
            };
            let stroke = if instruction.trace_inst_id.is_some() {
                "#2563eb"
            } else {
                "#d97706"
            };
            let label = format!(
                "#{raw_index} {} [{}]",
                instruction.instruction,
                format_trace_slot(instruction.trace_inst_id)
            );
            writeln!(
                &mut svg,
                "<rect x=\"{rect_x:.1}\" y=\"{rect_y:.1}\" width=\"{rect_width:.1}\" height=\"{block_height:.1}\" rx=\"8\" fill=\"{color}\" stroke=\"{stroke}\" stroke-width=\"1.2\"/>"
            )?;
            writeln!(
                &mut svg,
                "<text class=\"inst\" x=\"{:.1}\" y=\"{:.1}\">{}</text>",
                rect_x + 8.0,
                rect_y + 20.0,
                escape_svg_text(&label)
            )?;
            writeln!(
                &mut svg,
                "<text class=\"inst\" x=\"{:.1}\" y=\"{:.1}\">{}-{} ({})</text>",
                rect_x + 8.0,
                rect_y + 31.0,
                clk_start,
                clk_end,
                clk_span
            )?;
        }
    }

    writeln!(&mut svg, "</svg>")?;
    Ok(svg)
}

pub fn generate_execution_timeline_svg_from_output<P: AsRef<Path>>(
    output: &ExecutionOutput,
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    let svg = render_raw_execution_timeline_svg(output)?;
    write_string_file(path.as_ref(), svg)?;
    Ok(())
}

pub fn generate_execution_report<P: AsRef<Path>>(
    output: &ExecutionOutput,
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = path.as_ref();
    let mut report = fs::File::create(path)?;

    let total_rows = output.instruction_count();
    let exception_rows = output
        .per_hart
        .values()
        .flat_map(|hart| hart.instructions.iter())
        .filter(|instruction| !instruction.exceptions.is_empty())
        .count();
    let write_rows = output
        .per_hart
        .values()
        .flat_map(|hart| hart.instructions.iter())
        .filter(|instruction| {
            !instruction.register_writes.is_empty() || !instruction.memory_writes.is_empty()
        })
        .count();
    let timed_rows = output
        .per_hart
        .values()
        .flat_map(|hart| hart.instructions.iter())
        .filter(|instruction| instruction.timing.is_recorded())
        .count();
    let traceable_rows = output
        .per_hart
        .values()
        .flat_map(|hart| hart.instructions.iter())
        .filter(|instruction| instruction.trace_inst_id.is_some())
        .count();

    writeln!(report, "# {} Execution Output\n", output.core_id)?;
    writeln!(
        report,
        "ISA: `{:?}`. Rows: {}. Traceable rows: {}. Rows with writes: {}. Rows with exceptions: {}. Rows with timing: {}.",
        output.isa_base, total_rows, traceable_rows, write_rows, exception_rows, timed_rows
    )?;
    if let Some(global_cycles) = output.perf.global_cycles {
        writeln!(report, "Global cycles: `{}`.", global_cycles)?;
    }
    if !output.perf.notes.is_empty() {
        writeln!(report, "Perf notes: {}.", output.perf.notes.join(" | "))?;
    }
    writeln!(report)?;
    writeln!(
        report,
        "| Hart | Raw | Kind | Trace Slot | PC | ASM | Source | Register Writes | Memory Writes | Exceptions | Timing |"
    )?;
    writeln!(
        report,
        "|------|-----:|------|------------|----|-----|--------|-----------------|---------------|------------|--------|"
    )?;

    for (hart_id, hart_output) in &output.per_hart {
        for (raw_index, instruction) in hart_output.instructions.iter().enumerate() {
            let kind = if instruction.trace_inst_id.is_some() {
                "traceable"
            } else {
                "framework"
            };
            writeln!(
                report,
                "| {} | {} | {} | {} | `0x{:x}` | `{}` | `{}` | {} | {} | {} | {} |",
                hart_id,
                raw_index,
                kind,
                escape_markdown_cell(&format_trace_slot(instruction.trace_inst_id)),
                instruction.pc,
                escape_markdown_cell(&instruction.instruction.to_string()),
                escape_markdown_cell(instruction.source_annotation.as_deref().unwrap_or("-")),
                escape_markdown_cell(&format_register_writes(&instruction.register_writes)),
                escape_markdown_cell(&format_memory_writes(&instruction.memory_writes)),
                escape_markdown_cell(&format_exception_causes(&instruction.exceptions)),
                format_instruction_timing(Some(&instruction.timing)),
            )?;
        }
    }

    Ok(())
}

pub fn write_raw_execution_output_json<P: AsRef<Path>>(
    output: &ExecutionOutput,
    path: P,
) -> Result<(), ExecutionArtifactError> {
    write_json_pretty(path.as_ref(), output)?;
    Ok(())
}

pub fn emit_raw_execution_artifacts<P: AsRef<Path>>(
    output: &ExecutionOutput,
    output_dir: P,
    emit_execution_output_json: bool,
    emit_execution_report_md: bool,
    emit_execution_timeline_svg: bool,
) -> Result<(), ExecutionArtifactError> {
    if !emit_execution_output_json && !emit_execution_report_md && !emit_execution_timeline_svg {
        return Ok(());
    }

    let output_dir = output_dir.as_ref();
    ensure_parent_dir(output_dir.join(".artifact_io_anchor")).map_err(ArtifactIoError::from)?;
    fs::create_dir_all(output_dir).map_err(ArtifactIoError::from)?;

    if emit_execution_output_json {
        write_raw_execution_output_json(output, output_dir.join("execution_output.json"))?;
    }

    if emit_execution_timeline_svg {
        generate_execution_timeline_svg_from_output(
            output,
            output_dir.join("execution_timeline.svg"),
        )
        .map_err(ExecutionArtifactError::Report)?;
    }

    if emit_execution_report_md {
        generate_execution_report(output, output_dir.join("execution_report.md"))
            .map_err(ExecutionArtifactError::Report)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use riscv_fuzz_plugin_api::{AsmInstructionExecution, ISABase, PerfMetrics, TraceInstId};
    use riscv_instruction::{
        AsmDecodeWithContext, InstructionSequence, IsaDecodeContext, RiscvInstruction,
    };

    fn raw_instruction(
        pc: u64,
        instruction: &str,
        source_annotation: Option<&str>,
        trace_inst_id: Option<TraceInstId>,
        register_writes: Vec<RegisterValue>,
        memory_writes: Vec<MemValue>,
        timing: InstructionTiming,
    ) -> AsmInstructionExecution {
        AsmInstructionExecution {
            pc,
            instruction: RiscvInstruction::decode_asm_with_context(
                instruction,
                IsaDecodeContext::RV64,
            )
            .unwrap(),
            source_annotation: source_annotation.map(str::to_string),
            trace_inst_id,
            exceptions: Vec::new(),
            register_writes,
            memory_writes,
            timing,
        }
    }

    #[test]
    fn generate_execution_report_emits_single_raw_table() {
        let output = ExecutionOutput {
            per_hart: BTreeMap::from([(
                0,
                HartExecutionOutput {
                    hart_id: 0,
                    instructions: vec![
                        raw_instruction(
                            0x100,
                            "addi x1, x0, 1",
                            Some("addi x1, x0, 1"),
                            Some(TraceInstId::new(0, 0)),
                            vec![RegisterValue {
                                name: "x1".to_string(),
                                value: 1,
                            }],
                            Vec::new(),
                            InstructionTiming::new(Some(10), Some(10), None),
                        ),
                        raw_instruction(
                            0x104,
                            "ebreak",
                            Some("trap shim"),
                            None,
                            Vec::new(),
                            Vec::new(),
                            InstructionTiming::default(),
                        ),
                    ],
                },
            )]),
            core_id: "spike".to_string(),
            isa_base: ISABase::Rv64,
            perf: PerfMetrics::default(),
        };

        let report_path = std::env::temp_dir().join(format!(
            "riscv_fuzz_execution_report_{}_{}.md",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        generate_execution_report(&output, &report_path).expect("report");
        let markdown = std::fs::read_to_string(&report_path).expect("report contents");
        let _ = std::fs::remove_file(&report_path);

        assert!(markdown.contains("# spike Execution Output"));
        assert!(markdown.contains("| Hart | Raw | Kind | Trace Slot | PC | ASM | Source | Register Writes | Memory Writes | Exceptions | Timing |"));
        assert!(markdown.contains("| 0 | 0 | traceable | hart0#0 | `0x100` | `addi x1, x0, 1` | `addi x1, x0, 1` | x1=0x1 | - | - | 10/10/1 |"));
        assert!(markdown.contains(
            "| 0 | 1 | framework | - | `0x104` | `ebreak` | `trap shim` | - | - | - | - |"
        ));
        assert_eq!(markdown.matches("\n| ").count(), 3);
        assert!(!markdown.contains("## "));
    }

    #[test]
    fn build_execution_context_output_preserves_raw_instructions_and_derives_logical_view() {
        let expanded_li = InstructionSequence {
            pre_instructions: vec![
                RiscvInstruction::decode_asm_with_context("addi x5, x0, 1", IsaDecodeContext::RV64)
                    .unwrap(),
            ],
            instruction: RiscvInstruction::decode_asm_with_context(
                "addi x5, x5, 4",
                IsaDecodeContext::RV64,
            )
            .unwrap(),
            post_instructions: Vec::new(),
        };
        let store = InstructionSequence::new(
            RiscvInstruction::decode_asm_with_context("sb x5, 0(x6)", IsaDecodeContext::RV64)
                .unwrap(),
        );
        let output = ExecutionOutput {
            per_hart: BTreeMap::from([(
                0,
                HartExecutionOutput {
                    hart_id: 0,
                    instructions: vec![
                        raw_instruction(
                            0x100,
                            "addi x5, x0, 1",
                            Some("li x5, 0x1004"),
                            Some(TraceInstId::new(0, 0)),
                            vec![RegisterValue {
                                name: "x5".to_string(),
                                value: 0x1000,
                            }],
                            Vec::new(),
                            InstructionTiming::new(Some(10), Some(10), None),
                        ),
                        raw_instruction(
                            0x104,
                            "addi x5, x5, 4",
                            Some("li x5, 0x1004"),
                            Some(TraceInstId::new(0, 1)),
                            vec![RegisterValue {
                                name: "x5".to_string(),
                                value: 0x1004,
                            }],
                            Vec::new(),
                            InstructionTiming::new(Some(11), Some(11), None),
                        ),
                        raw_instruction(
                            0x108,
                            "sb x5, 0(x6)",
                            Some("sb x5, 0(x6)"),
                            Some(TraceInstId::new(0, 2)),
                            Vec::new(),
                            vec![MemValue {
                                addr: 0x20,
                                value: 0xaa,
                            }],
                            InstructionTiming::new(Some(13), Some(13), None),
                        ),
                        raw_instruction(
                            0x10c,
                            "ebreak",
                            Some("ebreak"),
                            None,
                            Vec::new(),
                            Vec::new(),
                            InstructionTiming::new(Some(12), Some(12), None),
                        ),
                    ],
                },
            )]),
            core_id: "spike".to_string(),
            isa_base: ISABase::Rv64,
            perf: PerfMetrics::default(),
        };

        let per_hart = vec![(0u32, vec![expanded_li, store])];
        let context_output =
            build_execution_context_output(output, (0, 0xff), &per_hart).expect("context output");

        assert_eq!(context_output.per_hart[&0].instructions.len(), 3);
        assert_eq!(context_output.framework_per_hart[&0].instructions.len(), 1);
        assert!(
            context_output.framework_per_hart[&0].instructions[0]
                .trace_inst_id
                .is_none()
        );
        assert_eq!(context_output.logical_view.len(), 3);
        assert_eq!(
            context_output.logical_view[0].asm_instruction_indices,
            vec![0]
        );
        assert_eq!(
            context_output.logical_view[0].register_writes,
            vec![RegisterValue {
                name: "x5".to_string(),
                value: 0x1000,
            }]
        );
        assert_eq!(context_output.logical_view[0].timing.clk_start, Some(10));
        assert_eq!(context_output.logical_view[0].timing.clk_end, Some(10));
        assert_eq!(
            context_output.logical_view[1].asm_instruction_indices,
            vec![1]
        );
        assert_eq!(context_output.logical_view[1].instruction_index, 1);
        assert_eq!(
            context_output.logical_view[1].register_writes,
            vec![RegisterValue {
                name: "x5".to_string(),
                value: 0x1004,
            }]
        );
        assert_eq!(context_output.logical_view[1].timing.clk_start, Some(11));
        assert_eq!(context_output.logical_view[1].timing.clk_end, Some(11));
        assert_eq!(
            context_output.logical_view[2].asm_instruction_indices,
            vec![2]
        );
        assert_eq!(context_output.logical_view[2].instruction_index, 2);
    }

    #[test]
    fn logical_view_maps_flattened_pre_main_and_post_rows_by_trace_inst_id() {
        let decode = |asm: &str| {
            RiscvInstruction::decode_asm_with_context(asm, IsaDecodeContext::RV64).unwrap()
        };
        let source = InstructionSequence {
            pre_instructions: vec![decode("addi x1, x0, 1"), decode("addi x2, x0, 2")],
            instruction: decode("addi x3, x0, 3"),
            post_instructions: vec![decode("addi x4, x0, 4"), decode("addi x5, x0, 5")],
        };
        let output = ExecutionOutput {
            per_hart: BTreeMap::from([(
                0,
                HartExecutionOutput {
                    hart_id: 0,
                    instructions: vec![
                        raw_instruction(
                            0x100,
                            "addi x1, x0, 1",
                            Some("addi x1, x0, 1"),
                            Some(TraceInstId::new(0, 0)),
                            Vec::new(),
                            Vec::new(),
                            InstructionTiming::default(),
                        ),
                        raw_instruction(
                            0x104,
                            "addi x2, x0, 2",
                            Some("addi x2, x0, 2"),
                            Some(TraceInstId::new(0, 1)),
                            Vec::new(),
                            Vec::new(),
                            InstructionTiming::default(),
                        ),
                        raw_instruction(
                            0x108,
                            "addi x3, x0, 3",
                            Some("addi x3, x0, 3"),
                            Some(TraceInstId::new(0, 2)),
                            Vec::new(),
                            Vec::new(),
                            InstructionTiming::default(),
                        ),
                        raw_instruction(
                            0x10c,
                            "addi x4, x0, 4",
                            Some("addi x4, x0, 4"),
                            Some(TraceInstId::new(0, 3)),
                            Vec::new(),
                            Vec::new(),
                            InstructionTiming::default(),
                        ),
                        raw_instruction(
                            0x110,
                            "addi x5, x0, 5",
                            Some("addi x5, x0, 5"),
                            Some(TraceInstId::new(0, 4)),
                            Vec::new(),
                            Vec::new(),
                            InstructionTiming::default(),
                        ),
                    ],
                },
            )]),
            core_id: "spike".to_string(),
            isa_base: ISABase::Rv64,
            perf: PerfMetrics::default(),
        };
        let context_output =
            build_execution_context_output(output, (0, 0xff), &[(0u32, vec![source])])
                .expect("context output");

        assert_eq!(context_output.logical_view.len(), 5);
        assert_eq!(
            context_output
                .logical_view
                .iter()
                .map(|logical| logical.asm_instruction_indices.clone())
                .collect::<Vec<_>>(),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );
    }

    #[test]
    fn logical_view_keeps_multiple_raw_pcs_on_the_same_trace_instruction() {
        let decode = |asm: &str| {
            RiscvInstruction::decode_asm_with_context(asm, IsaDecodeContext::RV64).unwrap()
        };
        let source = InstructionSequence {
            pre_instructions: vec![decode("addi x1, x0, 1")],
            instruction: decode("addi x2, x0, 2"),
            post_instructions: Vec::new(),
        };
        let output = ExecutionOutput {
            per_hart: BTreeMap::from([(
                0,
                HartExecutionOutput {
                    hart_id: 0,
                    instructions: vec![
                        AsmInstructionExecution {
                            pc: 0x100,
                            instruction: decode("addi x1, x0, 1"),
                            source_annotation: Some("addi x1, x0, 1".to_string()),
                            trace_inst_id: Some(TraceInstId::new(0, 0)),
                            exceptions: Vec::new(),
                            register_writes: vec![RegisterValue {
                                name: "x1".to_string(),
                                value: 1,
                            }],
                            memory_writes: vec![MemValue {
                                addr: 0x20,
                                value: 0x11,
                            }],
                            timing: InstructionTiming::default(),
                        },
                        AsmInstructionExecution {
                            pc: 0x104,
                            instruction: decode("addi x1, x0, 1"),
                            source_annotation: Some("addi x1, x0, 1".to_string()),
                            trace_inst_id: Some(TraceInstId::new(0, 0)),
                            exceptions: Vec::new(),
                            register_writes: vec![RegisterValue {
                                name: "x1".to_string(),
                                value: 2,
                            }],
                            memory_writes: vec![MemValue {
                                addr: 0x20,
                                value: 0x22,
                            }],
                            timing: InstructionTiming::default(),
                        },
                        AsmInstructionExecution {
                            pc: 0x108,
                            instruction: decode("addi x2, x0, 2"),
                            source_annotation: Some("addi x2, x0, 2".to_string()),
                            trace_inst_id: Some(TraceInstId::new(0, 1)),
                            exceptions: Vec::new(),
                            register_writes: Vec::new(),
                            memory_writes: Vec::new(),
                            timing: InstructionTiming::default(),
                        },
                    ],
                },
            )]),
            core_id: "spike".to_string(),
            isa_base: ISABase::Rv64,
            perf: PerfMetrics::default(),
        };
        let context_output =
            build_execution_context_output(output, (0, 0xff), &[(0u32, vec![source])])
                .expect("context output");

        assert_eq!(context_output.logical_view.len(), 2);
        assert_eq!(
            context_output.logical_view[0].asm_instruction_indices,
            vec![0, 1]
        );
        assert_eq!(
            context_output.logical_view[0].register_writes,
            vec![RegisterValue {
                name: "x1".to_string(),
                value: 2,
            }]
        );
        assert_eq!(
            context_output.logical_view[0].memory_writes,
            vec![MemValue {
                addr: 0x20,
                value: 0x22,
            }]
        );
        assert_eq!(
            context_output.logical_view[1].asm_instruction_indices,
            vec![2]
        );
    }

    #[test]
    fn logical_view_inherits_framework_state_for_fixed_stack_base_instructions() {
        let source_inst =
            RiscvInstruction::decode_asm_with_context("c.lwsp x13, 12(sp)", IsaDecodeContext::RV32)
                .unwrap();
        let output = ExecutionOutput {
            per_hart: BTreeMap::from([(
                0,
                HartExecutionOutput {
                    hart_id: 0,
                    instructions: vec![
                        AsmInstructionExecution {
                            pc: 0x100,
                            instruction: source_inst.clone(),
                            source_annotation: Some("c.lwsp x13, 12(sp)".to_string()),
                            trace_inst_id: Some(TraceInstId::new(0, 0)),
                            exceptions: Vec::new(),
                            register_writes: vec![RegisterValue {
                                name: "x13".to_string(),
                                value: 0x13,
                            }],
                            memory_writes: Vec::new(),
                            timing: InstructionTiming::new(Some(2), Some(2), None),
                        },
                        AsmInstructionExecution {
                            pc: 0x80,
                            instruction: RiscvInstruction::decode_asm_with_context(
                                "addi x2, x0, 32",
                                IsaDecodeContext::RV32,
                            )
                            .unwrap(),
                            source_annotation: Some("addi x2, x0, 32".to_string()),
                            trace_inst_id: None,
                            exceptions: Vec::new(),
                            register_writes: vec![RegisterValue {
                                name: "x2".to_string(),
                                value: 32,
                            }],
                            memory_writes: vec![MemValue {
                                addr: 0x8000_1000,
                                value: 0xaa,
                            }],
                            timing: InstructionTiming::new(Some(1), Some(1), None),
                        },
                    ],
                },
            )]),
            core_id: "spike".to_string(),
            isa_base: ISABase::Rv32,
            perf: PerfMetrics::default(),
        };

        let per_hart = vec![(0u32, vec![InstructionSequence::new(source_inst)])];
        let context_output = build_execution_context_output(output, (0x20, 0x5f), &per_hart)
            .expect("context output");

        assert_eq!(
            context_output.logical_view[0]
                .context
                .registers_before
                .get("x2"),
            Some(&32)
        );
        assert!(
            context_output.logical_view[0]
                .context
                .memory_before
                .is_empty()
        );
        assert_eq!(
            context_output.logical_view[0]
                .context
                .register_context
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            vec!["x2"]
        );
    }

    #[test]
    fn timeline_pixels_per_cycle_clamps_large_timelines_down() {
        let pixels_per_cycle = timeline_pixels_per_cycle(10_000);
        assert!(pixels_per_cycle <= 1.0);
        assert!(pixels_per_cycle >= 0.35);
    }
}
