use log::{debug, error};
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use riscv_fuzz_plugin_api::{
    AsmInstructionExecution, ExceptionCause, HartExecutionOutput, ISABase, InstructionTiming,
    MemValue, RegisterValue, TraceInstId,
};
use riscv_instruction::{AsmDecodeWithContext, RiscvInstruction};
use riscv_instruction_types::InstructionSequence;

use crate::trace_instruction::{TraceInstructionInfo, isa_base_to_decode_ctx};

#[derive(Clone)]
struct FlatSequenceEntry {
    instruction: RiscvInstruction,
}

fn flatten_sequences_to_instructions(
    sequences: &[InstructionSequence<RiscvInstruction>],
) -> Vec<FlatSequenceEntry> {
    let mut flat = Vec::new();
    for sequence in sequences {
        for instruction in sequence
            .pre_instructions
            .iter()
            .chain(std::iter::once(&sequence.instruction))
            .chain(sequence.post_instructions.iter())
        {
            flat.push(FlatSequenceEntry {
                instruction: instruction.clone(),
            });
        }
    }
    flat
}

/// 一条被追踪的指令，包含 PC、汇编文本、以及（可选）解析为类型化 IR 的两路结果。
/// A single traced instruction with PC, raw disassembly, and the two
/// optionally-parsed typed-IR views.
///
/// `parsed_disassembly` 与 `source_instruction` 是有意保留的双字段：
/// `parsed_disassembly` and `source_instruction` are intentionally kept separate:
/// - `source_instruction` 来自生成器，是该 PC 处指令的"权威"类型化形式；
///   if any, it is the canonical typed form produced by the testcase generator.
/// - `parsed_disassembly` 由 dump 反汇编文本现场重新解析得到；某些指令
///   （fence/lui/large-imm 等）受当前解析器的能力限制，可能与 source 不完全一致，
///   甚至无法解析（`None`）。
///   It is re-parsed from the disassembly text. Limitations in the current
///   parser (fence/lui/large-imm cases) mean it may differ from the source
///   form, or be `None` outright.
///
/// 在 trace 比对/统计阶段两者各有用：source 表达"应当执行什么"，
/// In trace comparison and analytics, the two play distinct roles: source says
/// parsed 表达"反汇编日志里看到了什么"。合并它们会丢掉 ground-truth 与
/// what should have executed; parsed says what the disassembly observed.
/// observed 之间的差异信号，因此即使大多数情况下两者相等，也不要折叠成一个字段。
/// Collapsing them would erase the ground-truth-vs-observed signal even when
/// they happen to agree, so they stay split.
#[derive(Debug, Clone, PartialEq)]
pub struct TracedInstruction {
    pub pc: u64,
    pub disassembly: String,
    /// 反汇编文本即时解析得到的类型化指令；可能因解析器限制而为 `None`。
    /// Typed instruction re-parsed from the disassembly text. May be `None`
    /// when the parser cannot represent the form (e.g. fence/lui/large-imm).
    pub parsed_disassembly: Option<RiscvInstruction>,
    pub source_line: Option<String>,
    pub trace_inst_local_index: Option<usize>,
    /// 来自生成器的权威类型化指令（ground truth），与 `parsed_disassembly` 在
    /// The canonical typed instruction supplied by the generator (ground truth);
    /// 解析器有限制时可能不一致——这种不一致本身就是 trace 比对要看的信号。
    /// when the disassembly parser falls short, divergence from
    /// `parsed_disassembly` is itself a signal trace comparison wants to see.
    pub source_instruction: Option<RiscvInstruction>,
}

#[derive(Debug, Clone)]
struct DumpBlock {
    source_line: Option<String>,
    instructions: Vec<(u64, String)>,
}

pub struct Tracer {
    instructions: Vec<TracedInstruction>,
    pc_to_instruction_index: HashMap<u64, usize>,
    pc_to_trace_inst_idx: HashMap<u64, usize>,
    trace_inst_idx_to_pcs: HashMap<usize, Vec<u64>>,
}

impl Tracer {
    pub fn new<P: AsRef<Path>>(
        instructions: &[String],
        dump_file_path: P,
        isa_base: ISABase,
    ) -> std::io::Result<Self> {
        let path = dump_file_path.as_ref();
        debug!("Loading dump file: {}", path.display());
        let dump_content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to read dump file {}: {}", path.display(), e);
                return Err(e);
            }
        };

        Self::from_dump_str(&dump_content, instructions, isa_base)
    }

    pub fn new_with_section_labels<P: AsRef<Path>>(
        instructions: &[String],
        dump_file_path: P,
        sections: &[(&str, &str)],
        isa_base: ISABase,
    ) -> std::io::Result<Self> {
        let path = dump_file_path.as_ref();
        debug!(
            "Loading dump file with labeled sections: {}",
            path.display()
        );
        let dump_content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to read dump file {}: {}", path.display(), e);
                return Err(e);
            }
        };

        Self::from_dump_str_in_sections(&dump_content, instructions, sections, isa_base)
    }

    pub fn new_for_sequences<P: AsRef<Path>>(
        sequences: &[InstructionSequence<RiscvInstruction>],
        dump_file_path: P,
        isa_base: ISABase,
    ) -> std::io::Result<Self> {
        let path = dump_file_path.as_ref();
        debug!("Loading dump file for typed sequences: {}", path.display());
        let dump_content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to read dump file {}: {}", path.display(), e);
                return Err(e);
            }
        };

        Self::from_sequence_dump_str(&dump_content, &dump_content, sequences, isa_base)
    }

    pub fn new_for_sequences_with_section_labels<P: AsRef<Path>>(
        sequences: &[InstructionSequence<RiscvInstruction>],
        dump_file_path: P,
        sections: &[(&str, &str)],
        isa_base: ISABase,
    ) -> std::io::Result<Self> {
        let path = dump_file_path.as_ref();
        debug!(
            "Loading dump file with labeled sections for typed sequences: {}",
            path.display()
        );
        let dump_content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                error!("Failed to read dump file {}: {}", path.display(), e);
                return Err(e);
            }
        };
        let filtered = filter_dump_to_labeled_sections(&dump_content, sections)?;

        Self::from_sequence_dump_str(&dump_content, &filtered, sequences, isa_base)
    }

    fn from_dump_str(
        dump: &str,
        instructions: &[String],
        isa_base: ISABase,
    ) -> std::io::Result<Self> {
        Self::from_full_and_mapping_dump(dump, dump, instructions, isa_base)
    }

    fn from_dump_str_in_sections(
        dump: &str,
        instructions: &[String],
        sections: &[(&str, &str)],
        isa_base: ISABase,
    ) -> std::io::Result<Self> {
        let filtered = filter_dump_to_labeled_sections(dump, sections)?;
        Self::from_full_and_mapping_dump(dump, &filtered, instructions, isa_base)
    }

    pub fn get_trace_inst_local_index(&self, pc: &u64) -> Option<usize> {
        self.pc_to_trace_inst_idx.get(pc).copied()
    }

    pub fn get_pcs_for_trace_inst(&self, idx: usize) -> Option<&Vec<u64>> {
        self.trace_inst_idx_to_pcs.get(&idx)
    }

    pub fn trace_inst_count(&self) -> usize {
        self.trace_inst_idx_to_pcs.len()
    }

    pub fn instructions(&self) -> &[TracedInstruction] {
        &self.instructions
    }

    pub fn instruction_for_pc(&self, pc: &u64) -> Option<&TracedInstruction> {
        self.pc_to_instruction_index
            .get(pc)
            .and_then(|index| self.instructions.get(*index))
    }

    pub fn from_traced_instructions(instructions: Vec<TracedInstruction>) -> Self {
        let mut pc_to_instruction_index = HashMap::new();
        let mut pc_to_trace_inst_idx = HashMap::new();
        let mut trace_inst_idx_to_pcs: HashMap<usize, Vec<u64>> = HashMap::new();

        for (index, instruction) in instructions.iter().enumerate() {
            assert!(
                pc_to_instruction_index
                    .insert(instruction.pc, index)
                    .is_none(),
                "duplicate PC 0x{:x} while building traced instruction index",
                instruction.pc
            );
            if let Some(local_index) = instruction.trace_inst_local_index {
                assert!(
                    pc_to_trace_inst_idx
                        .insert(instruction.pc, local_index)
                        .is_none(),
                    "duplicate PC 0x{:x} while building trace instruction local index",
                    instruction.pc
                );
                trace_inst_idx_to_pcs
                    .entry(local_index)
                    .or_default()
                    .push(instruction.pc);
            }
        }

        Self {
            instructions,
            pc_to_instruction_index,
            pc_to_trace_inst_idx,
            trace_inst_idx_to_pcs,
        }
    }

    fn from_full_and_mapping_dump(
        full_dump: &str,
        mapping_dump: &str,
        instructions: &[String],
        isa_base: ISABase,
    ) -> std::io::Result<Self> {
        let ctx = isa_base_to_decode_ctx(isa_base);
        let full_blocks = parse_dump_blocks(full_dump);
        let trace_instruction_info = TraceInstructionInfo::build(instructions, isa_base);
        let (pc_to_trace_inst_idx, trace_inst_idx_to_pcs, trace_mapping_by_pc) =
            match_trace_instruction_mapping(mapping_dump, &trace_instruction_info)?;

        let mut instructions = Vec::new();
        let mut pc_to_instruction_index = HashMap::new();
        for block in full_blocks {
            for (pc, disasm_str) in block.instructions {
                let normalized = normalize_for_parse(&disasm_str);
                let parsed_disassembly =
                    RiscvInstruction::decode_asm_with_context(&normalized, ctx).ok();
                let mapping = trace_mapping_by_pc.get(&pc);
                let traced = TracedInstruction {
                    pc,
                    disassembly: disasm_str,
                    parsed_disassembly,
                    source_line: block.source_line.clone(),
                    trace_inst_local_index: mapping.map(|(index, _)| *index),
                    source_instruction: mapping.and_then(|(_, info)| info.instruction.clone()),
                };
                insert_unique_pc_value(
                    &mut pc_to_instruction_index,
                    pc,
                    instructions.len(),
                    "traced instruction index",
                )?;
                instructions.push(traced);
            }
        }

        Ok(Self {
            instructions,
            pc_to_instruction_index,
            pc_to_trace_inst_idx,
            trace_inst_idx_to_pcs,
        })
    }

    fn from_sequence_dump_str(
        full_dump: &str,
        mapping_dump: &str,
        sequences: &[InstructionSequence<RiscvInstruction>],
        isa_base: ISABase,
    ) -> std::io::Result<Self> {
        let ctx = isa_base_to_decode_ctx(isa_base);
        let flat = flatten_sequences_to_instructions(sequences);
        let flat_lines = flat
            .iter()
            .map(|entry| entry.instruction.to_string())
            .collect::<Vec<_>>();
        let trace_instruction_info = TraceInstructionInfo::build(&flat_lines, isa_base);
        let (pc_to_trace_inst_idx, trace_inst_idx_to_pcs, trace_mapping_by_pc) =
            match_trace_instruction_mapping_contiguous(mapping_dump, &trace_instruction_info)?;

        let full_blocks = parse_dump_blocks(full_dump);
        let mut instructions = Vec::new();
        let mut pc_to_instruction_index = HashMap::new();
        for block in full_blocks {
            for (pc, disasm_str) in block.instructions {
                let normalized = normalize_for_parse(&disasm_str);
                let parsed_disassembly =
                    RiscvInstruction::decode_asm_with_context(&normalized, ctx).ok();
                let mapping = trace_mapping_by_pc.get(&pc);
                let flat_index = mapping.map(|(index, _)| *index);
                let flat_entry = flat_index.and_then(|index| flat.get(index));
                let traced = TracedInstruction {
                    pc,
                    disassembly: disasm_str,
                    parsed_disassembly,
                    source_line: block.source_line.clone(),
                    trace_inst_local_index: flat_index,
                    source_instruction: flat_entry
                        .map(|entry| entry.instruction.clone())
                        .or_else(|| mapping.and_then(|(_, info)| info.instruction.clone())),
                };
                insert_unique_pc_value(
                    &mut pc_to_instruction_index,
                    pc,
                    instructions.len(),
                    "traced instruction index",
                )?;
                instructions.push(traced);
            }
        }

        Ok(Self {
            instructions,
            pc_to_instruction_index,
            pc_to_trace_inst_idx,
            trace_inst_idx_to_pcs,
        })
    }
}

fn build_hart_execution_output_filtered<FReg, FMem, FExc, FTiming, FInclude>(
    tracer: &Tracer,
    hart_id: u32,
    mut register_writes_for_pc: FReg,
    mut memory_writes_for_pc: FMem,
    mut exceptions_for_pc: FExc,
    mut timing_for_pc: FTiming,
    mut include_pc: FInclude,
) -> HartExecutionOutput
where
    FReg: FnMut(u64) -> Vec<RegisterValue>,
    FMem: FnMut(u64) -> Vec<MemValue>,
    FExc: FnMut(u64) -> Vec<ExceptionCause>,
    FTiming: FnMut(u64) -> InstructionTiming,
    FInclude: FnMut(u64) -> bool,
{
    let mut instructions = Vec::new();

    for traced in tracer.instructions() {
        if !include_pc(traced.pc) {
            continue;
        }

        let is_user = traced.trace_inst_local_index.is_some();

        let disassembly = traced.parsed_disassembly.clone().or_else(|| {
            if is_user {
                traced.source_instruction.clone()
            } else {
                None
            }
        });
        let Some(disassembly) = disassembly else {
            continue;
        };

        instructions.push(AsmInstructionExecution {
            pc: traced.pc,
            instruction: disassembly,
            source_annotation: traced.source_line.clone(),
            trace_inst_id: traced
                .trace_inst_local_index
                .map(|local_index| TraceInstId::new(hart_id, local_index)),
            exceptions: exceptions_for_pc(traced.pc),
            register_writes: register_writes_for_pc(traced.pc),
            memory_writes: memory_writes_for_pc(traced.pc),
            timing: timing_for_pc(traced.pc).normalized(),
        });
    }

    HartExecutionOutput {
        hart_id,
        instructions,
    }
}

pub fn build_hart_execution_output<FReg, FMem, FExc, FTiming>(
    tracer: &Tracer,
    hart_id: u32,
    register_writes_for_pc: FReg,
    memory_writes_for_pc: FMem,
    exceptions_for_pc: FExc,
    timing_for_pc: FTiming,
) -> HartExecutionOutput
where
    FReg: FnMut(u64) -> Vec<RegisterValue>,
    FMem: FnMut(u64) -> Vec<MemValue>,
    FExc: FnMut(u64) -> Vec<ExceptionCause>,
    FTiming: FnMut(u64) -> InstructionTiming,
{
    build_hart_execution_output_filtered(
        tracer,
        hart_id,
        register_writes_for_pc,
        memory_writes_for_pc,
        exceptions_for_pc,
        timing_for_pc,
        |_| true,
    )
}

pub fn build_raw_hart_execution_output<FReg, FMem, FExc, FTiming>(
    tracer: &Tracer,
    hart_id: u32,
    observed_pcs: impl IntoIterator<Item = u64>,
    register_writes_for_pc: FReg,
    memory_writes_for_pc: FMem,
    exceptions_for_pc: FExc,
    timing_for_pc: FTiming,
) -> HartExecutionOutput
where
    FReg: FnMut(u64) -> Vec<RegisterValue>,
    FMem: FnMut(u64) -> Vec<MemValue>,
    FExc: FnMut(u64) -> Vec<ExceptionCause>,
    FTiming: FnMut(u64) -> InstructionTiming,
{
    let observed_pcs = observed_pcs.into_iter().collect::<HashSet<_>>();
    build_hart_execution_output_filtered(
        tracer,
        hart_id,
        register_writes_for_pc,
        memory_writes_for_pc,
        exceptions_for_pc,
        timing_for_pc,
        |pc| observed_pcs.contains(&pc),
    )
}

pub fn build_template_raw_hart_execution_output<FReg, FMem, FExc, FTiming>(
    tracer: &Tracer,
    hart_id: u32,
    observed_pcs: impl IntoIterator<Item = u64>,
    register_writes_for_pc: FReg,
    memory_writes_for_pc: FMem,
    exceptions_for_pc: FExc,
    timing_for_pc: FTiming,
) -> HartExecutionOutput
where
    FReg: FnMut(u64) -> Vec<RegisterValue>,
    FMem: FnMut(u64) -> Vec<MemValue>,
    FExc: FnMut(u64) -> Vec<ExceptionCause>,
    FTiming: FnMut(u64) -> InstructionTiming,
{
    let mut output = build_raw_hart_execution_output(
        tracer,
        hart_id,
        observed_pcs,
        register_writes_for_pc,
        memory_writes_for_pc,
        exceptions_for_pc,
        timing_for_pc,
    );
    output
        .instructions
        .retain(|instruction| instruction.trace_inst_id.is_none());
    output
}

fn instruction_ready_before_cycle(
    instruction: &AsmInstructionExecution,
    stable_before_cycle: Option<u64>,
    include_untimed: bool,
) -> bool {
    let Some(limit) = stable_before_cycle else {
        return include_untimed;
    };
    // Streaming snapshots should only expose rows whose retirement window is
    // complete. Using `clk_start` first can publish a row that still gains
    // writes/timing later as more trace data arrives.
    if let Some(cycle) = instruction.timing.clk_end {
        cycle < limit
    } else if instruction.timing.clk_start.is_some() {
        false
    } else {
        include_untimed
    }
}

pub fn filter_hart_output_for_streaming_by_tracer(
    hart: &HartExecutionOutput,
    _tracer: &Tracer,
    stable_before_cycle: Option<u64>,
    include_untimed: bool,
) -> HartExecutionOutput {
    let instructions = hart
        .instructions
        .iter()
        .take_while(|instruction| {
            instruction_ready_before_cycle(instruction, stable_before_cycle, include_untimed)
        })
        .cloned()
        .collect();

    HartExecutionOutput {
        hart_id: hart.hart_id,
        instructions,
    }
}

pub fn filter_per_hart_output_for_streaming_by_tracer(
    per_hart: &BTreeMap<u32, HartExecutionOutput>,
    tracers: &BTreeMap<u32, Tracer>,
    stable_before_cycle_by_hart: &BTreeMap<u32, Option<u64>>,
    include_untimed: bool,
) -> BTreeMap<u32, HartExecutionOutput> {
    per_hart
        .iter()
        .map(|(hart_id, hart)| {
            let filtered = tracers
                .get(hart_id)
                .map(|tracer| {
                    filter_hart_output_for_streaming_by_tracer(
                        hart,
                        tracer,
                        stable_before_cycle_by_hart.get(hart_id).copied().flatten(),
                        include_untimed,
                    )
                })
                .unwrap_or_else(|| HartExecutionOutput {
                    hart_id: hart.hart_id,
                    instructions: Vec::new(),
                });
            (*hart_id, filtered)
        })
        .collect()
}

fn parse_dump_blocks(dump: &str) -> Vec<DumpBlock> {
    let mut blocks: Vec<DumpBlock> = Vec::new();
    let mut active: Option<usize> = None;

    for line in dump.lines() {
        if let Some((pc, disasm_text)) = parse_pc_line(line) {
            let disassembly = disasm_text.unwrap_or_else(|| "<unknown>".to_string());
            if let Some(idx) = active {
                blocks[idx].instructions.push((pc, disassembly));
            } else {
                blocks.push(DumpBlock {
                    source_line: None,
                    instructions: vec![(pc, disassembly)],
                });
                active = Some(blocks.len() - 1);
            }
            continue;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed.starts_with('.') || trimmed.ends_with(':') {
            active = None;
            continue;
        }

        let normalized = normalize_whitespace(trimmed);
        if normalized.is_empty() {
            active = None;
            continue;
        }

        blocks.push(DumpBlock {
            source_line: Some(normalized),
            instructions: Vec::new(),
        });
        active = Some(blocks.len() - 1);
    }

    blocks
}

type PcToTraceInstIdx = HashMap<u64, usize>;
type TraceInstIdxToPcs = HashMap<usize, Vec<u64>>;
type TraceMappingByPc = HashMap<u64, (usize, TraceInstructionInfo)>;
type TraceMappingResult = (PcToTraceInstIdx, TraceInstIdxToPcs, TraceMappingByPc);

fn insert_unique_pc_value<T>(
    map: &mut HashMap<u64, T>,
    pc: u64,
    value: T,
    what: &str,
) -> std::io::Result<()> {
    if map.insert(pc, value).is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("duplicate PC 0x{pc:x} while building {what}"),
        ));
    }
    Ok(())
}

fn build_trace_mapping_from_window(
    matchable_blocks: &[(String, Vec<u64>)],
    trace_instruction_info: &[TraceInstructionInfo],
    win_start: usize,
) -> std::io::Result<TraceMappingResult> {
    let mut pc_to_trace_inst_idx: PcToTraceInstIdx = HashMap::new();
    let mut trace_inst_idx_to_pcs: TraceInstIdxToPcs = HashMap::new();
    let mut trace_mapping_by_pc: TraceMappingByPc = HashMap::new();

    for (local_index, info) in trace_instruction_info.iter().enumerate() {
        let block_index = win_start + local_index;
        let pcs = &matchable_blocks[block_index].1;
        if pcs.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Matched instruction '{}' at dump index {} has no PC entries",
                    info.instruction_text, block_index
                ),
            ));
        }

        for &pc in pcs {
            insert_unique_pc_value(
                &mut pc_to_trace_inst_idx,
                pc,
                local_index,
                "trace instruction local index",
            )?;
            insert_unique_pc_value(
                &mut trace_mapping_by_pc,
                pc,
                (local_index, info.clone()),
                "trace mapping by pc",
            )?;
        }
        trace_inst_idx_to_pcs.insert(local_index, pcs.clone());
    }

    Ok((
        pc_to_trace_inst_idx,
        trace_inst_idx_to_pcs,
        trace_mapping_by_pc,
    ))
}

fn match_trace_instruction_mapping_contiguous(
    dump: &str,
    trace_instruction_info: &[TraceInstructionInfo],
) -> std::io::Result<TraceMappingResult> {
    let dump_blocks = parse_dump_blocks(dump);
    let matchable_blocks: Vec<_> = dump_blocks
        .iter()
        .filter_map(|block| {
            block.source_line.as_ref().map(|source| {
                (
                    normalize_for_source_mapping(source),
                    block
                        .instructions
                        .iter()
                        .map(|(pc, _)| *pc)
                        .collect::<Vec<_>>(),
                )
            })
        })
        .collect();
    let input_seq = trace_instruction_info
        .iter()
        .map(|info| normalize_for_source_mapping(&info.instruction_text))
        .collect::<Vec<_>>();
    if input_seq.is_empty() {
        return Ok((HashMap::new(), HashMap::new(), HashMap::new()));
    }

    if matchable_blocks.len() < input_seq.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Dump contains only {} traceable lines but need {} traceable input instructions",
                matchable_blocks.len(),
                input_seq.len()
            ),
        ));
    }

    let matching_window_starts = matchable_blocks
        .windows(input_seq.len())
        .enumerate()
        .filter_map(|(start, window)| {
            window
                .iter()
                .zip(input_seq.iter())
                .all(|((dump_line, _), source_line)| dump_line == source_line)
                .then_some(start)
        })
        .collect::<Vec<_>>();

    match matching_window_starts.as_slice() {
        [] => {
            let first_mismatch = matchable_blocks
                .iter()
                .zip(input_seq.iter())
                .enumerate()
                .find(|(_, ((dump_line, _), source_line))| dump_line != *source_line)
                .map(|(idx, ((dump_line, _), source_line))| {
                    format!(" first mismatch at {idx}: dump='{dump_line}' source='{source_line}'")
                })
                .unwrap_or_else(|| {
                    format!(
                        " no exact contiguous window among {} traceable dump lines",
                        matchable_blocks.len()
                    )
                });
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Could not match typed testcase instruction sequence exactly (len={}).{first_mismatch}",
                    input_seq.len()
                ),
            ))
        }
        [window_start] => build_trace_mapping_from_window(
            &matchable_blocks,
            trace_instruction_info,
            *window_start,
        ),
        starts => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Typed testcase instruction sequence is ambiguous in dump: matched {} exact contiguous windows at starts {:?}",
                starts.len(),
                starts
            ),
        )),
    }
}

fn match_trace_instruction_mapping(
    dump: &str,
    trace_instruction_info: &[TraceInstructionInfo],
) -> std::io::Result<TraceMappingResult> {
    let dump_blocks = parse_dump_blocks(dump);
    let matchable_blocks: Vec<_> = dump_blocks
        .iter()
        .filter_map(|block| {
            block.source_line.as_ref().map(|source| {
                (
                    source.clone(),
                    block
                        .instructions
                        .iter()
                        .map(|(pc, _)| *pc)
                        .collect::<Vec<_>>(),
                )
            })
        })
        .collect();

    let input_seq: Vec<String> = trace_instruction_info
        .iter()
        .map(|info| normalize_for_source_mapping(&info.instruction_text))
        .collect();
    if input_seq.is_empty() {
        return Ok((HashMap::new(), HashMap::new(), HashMap::new()));
    }

    if matchable_blocks.len() < input_seq.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Dump contains only {} traceable lines but need {} traceable input instructions",
                matchable_blocks.len(),
                input_seq.len()
            ),
        ));
    }

    let best_window = {
        let n = matchable_blocks.len();
        let m = input_seq.len();
        let mut best: Option<(usize, usize)> = None;

        let mut i = 0;
        while i < n {
            let mut j = 0;
            let mut k = i;
            while k < n && j < m {
                if normalize_for_source_mapping(&matchable_blocks[k].0) == input_seq[j] {
                    j += 1;
                }
                k += 1;
            }

            if j == m {
                let start = i;
                let end_exclusive = k;
                let span = end_exclusive - start;
                match best {
                    None => best = Some((start, end_exclusive)),
                    Some((bs, be)) => {
                        let bspan = be - bs;
                        if span < bspan {
                            best = Some((start, end_exclusive));
                        }
                    }
                }
            }
            i += 1;
        }

        best
    };

    let (win_start, win_end) = best_window.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Could not match testcase instruction sequence (len={}) in dump; wrappers may have reordered code unexpectedly",
                input_seq.len()
            ),
        )
    })?;

    let mut pc_to_trace_inst_idx: PcToTraceInstIdx = HashMap::new();
    let mut trace_inst_idx_to_pcs: TraceInstIdxToPcs = HashMap::new();
    let mut trace_mapping_by_pc: TraceMappingByPc = HashMap::new();
    let mut cursor = win_start;
    for (local_index, target) in input_seq.iter().enumerate() {
        while cursor < win_end
            && normalize_for_source_mapping(&matchable_blocks[cursor].0) != *target
        {
            cursor += 1;
        }
        if cursor >= win_end {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Internal mapping failure: matched window does not contain instruction {} ({})",
                    local_index, target
                ),
            ));
        }

        let pcs = &matchable_blocks[cursor].1;
        if pcs.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Matched instruction '{}' at dump index {} has no PC entries",
                    target, cursor
                ),
            ));
        }

        for &pc in pcs {
            insert_unique_pc_value(
                &mut pc_to_trace_inst_idx,
                pc,
                local_index,
                "trace instruction local index",
            )?;
            insert_unique_pc_value(
                &mut trace_mapping_by_pc,
                pc,
                (local_index, trace_instruction_info[local_index].clone()),
                "trace mapping by pc",
            )?;
        }
        trace_inst_idx_to_pcs.insert(local_index, pcs.clone());
        cursor += 1;
    }

    Ok((
        pc_to_trace_inst_idx,
        trace_inst_idx_to_pcs,
        trace_mapping_by_pc,
    ))
}

fn parse_pc_line(line: &str) -> Option<(u64, Option<String>)> {
    let trimmed = line.trim_start();
    let (pc_text, rest) = trimmed.split_once(':')?;
    if pc_text.is_empty() || !pc_text.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return None;
    }
    if rest.trim().is_empty() {
        return None;
    }
    let pc = u64::from_str_radix(pc_text, 16).ok()?;

    let fields: Vec<&str> = rest.split_whitespace().collect();
    let has_encoded_disassembly = matches!(
        fields.as_slice(),
        [encoding, decoded @ ..]
            if !decoded.is_empty() && encoding.chars().all(|ch| ch.is_ascii_hexdigit())
    );
    if trimmed.len() == line.len() && !has_encoded_disassembly {
        return None;
    }

    let disasm_text = match fields.as_slice() {
        [encoding, decoded @ ..]
            if !decoded.is_empty() && encoding.chars().all(|ch| ch.is_ascii_hexdigit()) =>
        {
            Some(normalize_whitespace(&decoded.join(" ")))
        }
        [] => None,
        _ => Some(normalize_whitespace(rest)),
    };

    Some((pc, disasm_text.filter(|text| !text.is_empty())))
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_for_parse(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 8);
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        result.push(c);
        if c == ',' && chars.peek() != Some(&' ') {
            result.push(' ');
        }
    }
    result
}

fn normalize_for_source_mapping(s: &str) -> String {
    normalize_whitespace(&normalize_for_parse(s))
}

fn line_matches_label(line: &str, label: &str) -> bool {
    let trimmed = line.trim();
    if trimmed == label {
        return true;
    }

    if let Some(name) = label.strip_suffix(':') {
        trimmed.ends_with(&format!("<{name}>:"))
    } else {
        false
    }
}

fn filter_dump_to_labeled_sections(
    dump: &str,
    sections: &[(&str, &str)],
) -> std::io::Result<String> {
    let lines: Vec<&str> = dump.lines().collect();
    let mut cursor = 0usize;
    let mut out = String::new();

    for (start_label, end_label) in sections {
        let start_idx = (cursor..lines.len())
            .find(|&idx| line_matches_label(lines[idx], start_label))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("dump is missing start label '{start_label}'"),
                )
            })?;
        let end_idx = ((start_idx + 1)..lines.len())
            .find(|&idx| line_matches_label(lines[idx], end_label))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("dump is missing end label '{end_label}'"),
                )
            })?;

        for line in &lines[start_idx..end_idx] {
            out.push_str(line);
            out.push('\n');
        }
        cursor = end_idx;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        TracedInstruction, Tracer, build_raw_hart_execution_output,
        filter_dump_to_labeled_sections, parse_pc_line,
    };
    use riscv_fuzz_plugin_api::{
        AsmInstructionExecution, ISABase, InstructionTiming, RegisterValue, TraceInstId,
    };
    use riscv_instruction::{AsmDecodeWithContext, IsaDecodeContext, RiscvInstruction};
    use riscv_instruction_types::InstructionSequence;

    #[test]
    fn objdump_source_lines_collect_pc_spans_for_pseudo_instructions() {
        let dump = r#"
0000000080000000 <_start>:
 testcase_code:
    li x16, 0xd6b071f042b88cd2
    80000020:	fd6b0837          	lui	x16,0xfd6b0
    80000024:	71f8081b          	addiw	x16,x16,1823
    80000028:	0832                	c.slli	x16,0xc
    8000002a:	04380813          	addi	x16,x16,67
    addi x17, x16, 1
    8000002e:	00180893          	addi	x17,x16,1
"#;
        let instructions = vec![
            String::from("li x16, 0xd6b071f042b88cd2"),
            String::from("addi x17, x16, 1"),
        ];

        let tracer = Tracer::from_dump_str(dump, &instructions, ISABase::Rv64).expect("parse dump");

        assert_eq!(
            tracer.get_pcs_for_trace_inst(0).cloned(),
            Some(vec![0x8000_0020, 0x8000_0024, 0x8000_0028, 0x8000_002a])
        );
        assert_eq!(
            tracer.get_pcs_for_trace_inst(1).cloned(),
            Some(vec![0x8000_002e])
        );
        assert_eq!(
            tracer
                .instruction_for_pc(&0x8000_0024)
                .and_then(|inst| inst.trace_inst_local_index),
            Some(0)
        );
        assert_eq!(
            tracer
                .instruction_for_pc(&0x8000_0024)
                .and_then(|inst| inst.trace_inst_local_index),
            Some(0)
        );
        assert_eq!(tracer.trace_inst_count(), 2);
    }

    #[test]
    fn source_mapped_raw_output_falls_back_to_source_instruction_when_raw_parse_missing() {
        let source_instruction =
            RiscvInstruction::decode_asm_with_context("li x9, 1", IsaDecodeContext::RV32)
                .expect("decode source instruction");
        let tracer = Tracer::from_traced_instructions(vec![TracedInstruction {
            pc: 0x8000_0050,
            disassembly: "lui x9,0x80020".to_string(),
            parsed_disassembly: None,
            source_line: Some("li x9, 1".to_string()),
            trace_inst_local_index: Some(0),
            source_instruction: Some(source_instruction),
        }]);

        let output = build_raw_hart_execution_output(
            &tracer,
            0,
            [0x8000_0050],
            |pc| {
                (pc == 0x8000_0050)
                    .then(|| RegisterValue {
                        name: "x9".to_string(),
                        value: 0x8002_0000,
                    })
                    .into_iter()
                    .collect()
            },
            |_| Vec::new(),
            |_| Vec::new(),
            |_| InstructionTiming::default(),
        );

        assert_eq!(output.instructions.len(), 1);
        assert_eq!(
            output.instructions[0].trace_inst_id,
            Some(TraceInstId::new(0, 0))
        );
        assert_eq!(output.instructions[0].register_writes[0].name, "x9");
    }

    #[test]
    fn streaming_tracer_waits_for_instruction_end_cycle() {
        let instruction = AsmInstructionExecution {
            pc: 0x1000,
            instruction: RiscvInstruction::decode_asm_with_context(
                "addi x1, x0, 1",
                IsaDecodeContext::RV32,
            )
            .expect("decode test instruction"),
            source_annotation: None,
            trace_inst_id: Some(TraceInstId::new(0, 0)),
            exceptions: Vec::new(),
            register_writes: Vec::new(),
            memory_writes: Vec::new(),
            timing: InstructionTiming::new(Some(10), None, None),
        };
        assert!(!super::instruction_ready_before_cycle(
            &instruction,
            Some(20),
            false,
        ));

        let finished = AsmInstructionExecution {
            timing: InstructionTiming::new(Some(10), Some(12), None),
            ..instruction
        };
        assert!(super::instruction_ready_before_cycle(
            &finished,
            Some(20),
            false,
        ));
    }

    #[test]
    fn sequence_tracer_maps_each_pc_to_flat_trace_instruction_id() {
        let addi_x1 =
            RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
                .unwrap();
        let addi_x2 =
            RiscvInstruction::decode_asm_with_context("addi x2, x0, 2", IsaDecodeContext::RV32)
                .unwrap();
        let addi_x3 =
            RiscvInstruction::decode_asm_with_context("addi x3, x0, 3", IsaDecodeContext::RV32)
                .unwrap();
        let sequence = InstructionSequence {
            pre_instructions: vec![addi_x1.clone(), addi_x2.clone()],
            instruction: addi_x3.clone(),
            post_instructions: vec![addi_x1],
        };
        let dump = r#"
0000000080000000 <_start>:
 testcase_code:
    addi x1, x0, 1
    80000020:	00100093          	addi	x1,x0,1
    addi x2, x0, 2
    80000024:	00200113          	addi	x2,x0,2
    addi x3, x0, 3
    80000028:	00300193          	addi	x3,x0,3
    addi x1, x0, 1
    8000002c:	00100093          	addi	x1,x0,1
"#;

        let tracer = Tracer::from_sequence_dump_str(
            dump,
            dump,
            std::slice::from_ref(&sequence),
            ISABase::Rv32,
        )
        .expect("parse sequence dump");

        for (local_index, expected_pcs) in [
            (0, vec![0x8000_0020]),
            (1, vec![0x8000_0024]),
            (2, vec![0x8000_0028]),
            (3, vec![0x8000_002c]),
        ] {
            assert_eq!(
                tracer.get_pcs_for_trace_inst(local_index).cloned(),
                Some(expected_pcs)
            );
        }
        for (pc, local_index) in [
            (0x8000_0020, 0),
            (0x8000_0024, 1),
            (0x8000_0028, 2),
            (0x8000_002c, 3),
        ] {
            let traced = tracer.instruction_for_pc(&pc).expect("traced pc");
            assert_eq!(traced.trace_inst_local_index, Some(local_index));
        }
    }

    #[test]
    fn sequence_tracer_assigns_flat_trace_instruction_indices() {
        let addi =
            RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
                .unwrap();
        let sequence = InstructionSequence {
            pre_instructions: vec![addi.clone()],
            instruction: addi.clone(),
            post_instructions: vec![addi],
        };
        let dump = r#"
0000000080000000 <_start>:
 testcase_code:
    addi x1, x0, 1
    80000020:	00100093          	addi	x1,x0,1
    addi x1, x0, 1
    80000024:	00100093          	addi	x1,x0,1
    addi x1, x0, 1
    80000028:	00100093          	addi	x1,x0,1
"#;

        let tracer = Tracer::from_sequence_dump_str(dump, dump, &[sequence], ISABase::Rv32)
            .expect("parse sequence dump");

        let source_indices = [0x8000_0020, 0x8000_0024, 0x8000_0028]
            .into_iter()
            .map(|pc| {
                tracer
                    .instruction_for_pc(&pc)
                    .and_then(|traced| traced.trace_inst_local_index)
            })
            .collect::<Vec<_>>();
        assert_eq!(source_indices, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn sequence_tracer_maps_compressed_load_after_multi_pc_pre() {
        let pre =
            RiscvInstruction::decode_asm_with_context("li x12, 917456", IsaDecodeContext::RV64)
                .unwrap();
        let main =
            RiscvInstruction::decode_asm_with_context("c.lw x15, 8(x12)", IsaDecodeContext::RV64)
                .unwrap();
        let sequence = InstructionSequence {
            pre_instructions: vec![pre],
            instruction: main,
            post_instructions: Vec::new(),
        };
        let dump = r#"
0000000000000000 <_start>:
 testcase_code:
    li x12, 917456
 69e:	000e0637          	lui	x12,0xe0
 6a2:	fd06061b          	addiw	x12,x12,-48 # dffd0 <fromhost+0x9ffc8>
    c.lw x15, 8(x12)
 6a6:	461c                	c.lw	x15,8(x12)
    fcvt.lu.s x14, f15, dyn
 6a8:	c037f753          	fcvt.lu.s	x14,f15
"#;

        let tracer = Tracer::from_sequence_dump_str(
            dump,
            dump,
            std::slice::from_ref(&sequence),
            ISABase::Rv64,
        )
        .expect("parse sequence dump");

        assert_eq!(
            tracer.get_pcs_for_trace_inst(0).cloned(),
            Some(vec![0x69e, 0x6a2])
        );
        assert_eq!(tracer.get_pcs_for_trace_inst(1).cloned(), Some(vec![0x6a6]));
        for (pc, local_index) in [(0x69e, 0), (0x6a2, 0), (0x6a6, 1)] {
            let traced = tracer.instruction_for_pc(&pc).expect("traced pc");
            assert_eq!(traced.trace_inst_local_index, Some(local_index));
        }
    }

    #[test]
    fn sequence_tracer_maps_lhu_between_multi_pc_pre_and_following_float() {
        let load_sequence = InstructionSequence {
            pre_instructions: vec![
                RiscvInstruction::decode_asm_with_context("li x12, 917434", IsaDecodeContext::RV64)
                    .unwrap(),
            ],
            instruction: RiscvInstruction::decode_asm_with_context(
                "lhu x13, 14(x12)",
                IsaDecodeContext::RV64,
            )
            .unwrap(),
            post_instructions: Vec::new(),
        };
        let fmin_sequence = InstructionSequence::new(
            RiscvInstruction::decode_asm_with_context(
                "fmin.s f18, f16, f14",
                IsaDecodeContext::RV64,
            )
            .unwrap(),
        );
        let dump = r#"
0000000000000000 <_start>:
 testcase_code:
    li x12, 917434
 270:	000e0637          	lui	x12,0xe0
 274:	fba6061b          	addiw	x12,x12,-70 # dffba <fromhost+0x9ffb2>
    lhu x13, 14(x12)
 278:	00e65683          	lhu	x13,14(x12)
    fmin.s f18, f16, f14
 27c:	28e80953          	fmin.s	f18,f16,f14
 exit:
"#;

        let tracer = Tracer::from_sequence_dump_str(
            dump,
            dump,
            &[load_sequence.clone(), fmin_sequence.clone()],
            ISABase::Rv64,
        )
        .expect("parse sequence dump");

        assert_eq!(
            tracer.get_pcs_for_trace_inst(0).cloned(),
            Some(vec![0x270, 0x274])
        );
        assert_eq!(tracer.get_pcs_for_trace_inst(1).cloned(), Some(vec![0x278]));
        assert_eq!(tracer.get_pcs_for_trace_inst(2).cloned(), Some(vec![0x27c]));
        let traced_load = tracer.instruction_for_pc(&0x278).expect("traced lhu pc");
        assert_eq!(traced_load.trace_inst_local_index, Some(1));

        let traced_fmin = tracer.instruction_for_pc(&0x27c).expect("traced fmin pc");
        assert_eq!(traced_fmin.trace_inst_local_index, Some(2));
    }

    #[test]
    fn typed_sequence_tracer_rejects_ambiguous_exact_windows() {
        let sequence = InstructionSequence::new(
            RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
                .unwrap(),
        );
        let dump = r#"
0000000080000000 <_start>:
 testcase_code:
    addi x1, x0, 1
    80000020:	00100093          	addi	x1,x0,1
    addi x1, x0, 1
    80000024:	00100093          	addi	x1,x0,1
"#;

        let err = match Tracer::from_sequence_dump_str(dump, dump, &[sequence], ISABase::Rv32) {
            Ok(_) => panic!("ambiguous typed windows must fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("ambiguous"));
    }

    #[test]
    fn typed_sequence_tracer_rejects_non_contiguous_subsequence_match() {
        let first =
            RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
                .unwrap();
        let second =
            RiscvInstruction::decode_asm_with_context("addi x2, x0, 2", IsaDecodeContext::RV32)
                .unwrap();
        let injected =
            RiscvInstruction::decode_asm_with_context("addi x3, x0, 3", IsaDecodeContext::RV32)
                .unwrap();
        let sequences = vec![
            InstructionSequence::new(first),
            InstructionSequence::new(second),
        ];
        let dump = format!(
            r#"
0000000080000000 <_start>:
 testcase_code:
    addi x1, x0, 1
    80000020:	00100093          	addi	x1,x0,1
    {}
    80000024:	00300193          	{}
    addi x2, x0, 2
    80000028:	00200113          	addi	x2,x0,2
"#,
            injected, injected
        );

        let err = match Tracer::from_sequence_dump_str(&dump, &dump, &sequences, ISABase::Rv32) {
            Ok(_) => panic!("typed subsequence fallback must not be allowed"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("Could not match typed testcase instruction sequence exactly")
        );
    }

    #[test]
    fn typed_raw_output_uses_flat_trace_inst_id_from_partial_metadata() {
        let instruction =
            RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
                .unwrap();
        let tracer = Tracer::from_traced_instructions(vec![TracedInstruction {
            pc: 0x8000_0020,
            disassembly: "addi x1, x0, 1".to_string(),
            parsed_disassembly: Some(instruction.clone()),
            source_line: Some("addi x1, x0, 1".to_string()),
            trace_inst_local_index: Some(0),
            source_instruction: Some(instruction.clone()),
        }]);

        let output = build_raw_hart_execution_output(
            &tracer,
            0,
            [0x8000_0020],
            |_| Vec::new(),
            |_| Vec::new(),
            |_| Vec::new(),
            |_| InstructionTiming::default(),
        );

        assert_eq!(output.instructions.len(), 1);
        assert_eq!(output.instructions[0].pc, 0x8000_0020);
        assert_eq!(
            output.instructions[0].trace_inst_id,
            Some(TraceInstId::new(0, 0))
        );
    }

    #[test]
    fn objdump_parser_ignores_label_only_numeric_lines() {
        assert_eq!(parse_pc_line("2:"), None);
        assert_eq!(
            parse_pc_line(" 2: 0001                 c.nop"),
            Some((0x2, Some("c.nop".to_string())))
        );
    }

    #[test]
    fn objdump_parser_ignores_unindented_numeric_local_labels_in_picorv32_style_source() {
        let dump = r#"
00000000 <_start>:
testcase_code_h0:
    c.nop
  2:	0001                	c.nop
    j 2f
  4:	a011                	c.j	8 <testcase_code_h0+0x8>
1:  addi x5, x5, -3
  6:	12f5                	c.addi	x5,-3
2:
    addi x1, x0, 1
  8:	00100093          	addi	x1,x0,1

0000000c <exit_h0>:
exit_h0:
"#;

        let tracer = Tracer::from_dump_str_in_sections(
            dump,
            &[],
            &[("testcase_code_h0:", "exit_h0:")],
            ISABase::Rv32,
        )
        .expect("parse dump");

        let pcs: Vec<u64> = tracer.instructions().iter().map(|inst| inst.pc).collect();
        assert_eq!(pcs, vec![0x2, 0x4, 0x6, 0x8]);
        assert_eq!(
            tracer
                .instruction_for_pc(&0x6)
                .and_then(|inst| inst.source_line.as_deref()),
            Some("1: addi x5, x5, -3")
        );
    }

    #[test]
    fn filter_dump_to_labeled_sections_keeps_only_requested_ranges() {
        let dump = r#"
0000000080000000 <_start>:
testcase_code:
    addi x1, x1, 1
    80000000: 00108093           addi x1,x1,1
testcase_code_end:
exit:
    addi x2, x2, 1
    80000004: 00110113           addi x2,x2,1
hart1_testcase_code:
    addi x3, x3, 1
    80000008: 00118193           addi x3,x3,1
hart1_testcase_code_end:
"#;

        let filtered = filter_dump_to_labeled_sections(
            dump,
            &[
                ("testcase_code:", "testcase_code_end:"),
                ("hart1_testcase_code:", "hart1_testcase_code_end:"),
            ],
        )
        .expect("filter dump");

        assert!(filtered.contains("testcase_code:"));
        assert!(filtered.contains("hart1_testcase_code:"));
        assert!(!filtered.contains("exit:"));
        assert!(!filtered.contains("addi x2, x2, 1"));
    }

    #[test]
    fn tracer_keeps_unmapped_template_instructions_alongside_source_mapping() {
        let dump = r#"
0000000080000000 <_start>:
testcase_code:
    addi x1, x0, 1
    80000000: 00100093           addi x1,x0,1
    ebreak
    80000004: 00100073           ebreak
    addi x2, x0, 2
    80000008: 00200113           addi x2,x0,2
"#;
        let instructions = vec!["addi x1, x0, 1".to_string(), "addi x2, x0, 2".to_string()];

        let tracer = Tracer::from_dump_str(dump, &instructions, ISABase::Rv64).expect("parse dump");
        let pcs: Vec<u64> = tracer.instructions().iter().map(|inst| inst.pc).collect();
        assert_eq!(pcs, vec![0x8000_0000, 0x8000_0004, 0x8000_0008]);
        assert_eq!(
            tracer
                .instruction_for_pc(&0x8000_0004)
                .and_then(|inst| inst.trace_inst_local_index),
            None
        );
        assert_eq!(
            tracer
                .instruction_for_pc(&0x8000_0004)
                .map(|inst| inst.disassembly.to_string())
                .as_deref(),
            Some("ebreak")
        );
    }
}
