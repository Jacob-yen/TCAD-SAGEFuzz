//! Streaming execution API for real-time per-instruction result delivery.
//!
//! Each streamed row is the same flat asm execution record used in
//! `ExecutionOutput`. Testcase rows carry `AsmInstructionExecution::trace_inst_id`;
//! framework rows carry `None`.

use std::collections::BTreeMap;
use std::{fs, io, path::Path};

use crate::error::ImplExecutionError;
use crate::execution::{AsmInstructionExecution, HartExecutionOutput, PerfMetrics};

/// A single event emitted by a streaming plugin execution.
#[derive(Debug)]
pub enum InstructionEvent {
    /// A retired asm row on one hart. `instruction.trace_inst_id` determines
    /// whether it maps to a testcase input instruction or framework code.
    Instruction {
        hart_id: u32,
        instruction: AsmInstructionExecution,
    },
    /// Execution finished. The caller must not expect further instruction rows
    /// for this implementation after receiving this variant.
    Completed { perf: PerfMetrics },
    /// Execution failed with an error.
    Failed { error: ImplExecutionError },
}

/// Collects streaming events into a full `HartExecutionOutput` per hart.
pub struct StreamingCollector {
    pub per_hart: BTreeMap<u32, HartExecutionOutput>,
    pub perf: Option<PerfMetrics>,
    pub error: Option<ImplExecutionError>,
}

impl StreamingCollector {
    pub fn new() -> Self {
        Self {
            per_hart: BTreeMap::new(),
            perf: None,
            error: None,
        }
    }

    fn ensure_hart(&mut self, hart_id: u32) -> &mut HartExecutionOutput {
        self.per_hart
            .entry(hart_id)
            .or_insert_with(|| HartExecutionOutput::empty(hart_id))
    }

    fn merge_timing(
        existing: &mut crate::execution::InstructionTiming,
        incoming: &crate::execution::InstructionTiming,
    ) -> bool {
        if !incoming.is_recorded() {
            return false;
        }
        if !existing.is_recorded() {
            *existing = incoming.clone();
            return true;
        }

        let start = [existing.clk_start, incoming.clk_start]
            .into_iter()
            .flatten()
            .min();
        let end = [existing.clk_end, incoming.clk_end]
            .into_iter()
            .flatten()
            .max();
        let merged = if start.is_some() || end.is_some() {
            crate::execution::InstructionTiming::new(start.or(end), end.or(start), None)
        } else {
            let span = [existing.clk_span, incoming.clk_span]
                .into_iter()
                .flatten()
                .max();
            crate::execution::InstructionTiming::new(None, None, span)
        };
        let changed = *existing != merged;
        *existing = merged;
        changed
    }

    fn merge_asm_row(
        existing: &mut AsmInstructionExecution,
        incoming: &AsmInstructionExecution,
    ) -> Result<bool, String> {
        if existing.pc != incoming.pc {
            return Err(format!(
                "streaming plugin attempted to merge different pcs 0x{:x} and 0x{:x}",
                existing.pc, incoming.pc
            ));
        }
        if existing.instruction != incoming.instruction {
            return Err(format!(
                "streaming plugin changed an already collected row at pc 0x{:x}",
                incoming.pc
            ));
        }
        if existing.trace_inst_id != incoming.trace_inst_id {
            return Err(format!(
                "streaming plugin changed trace instruction id at pc 0x{:x} from {:?} to {:?}",
                incoming.pc, existing.trace_inst_id, incoming.trace_inst_id
            ));
        }

        let mut changed = false;
        match (&existing.source_annotation, &incoming.source_annotation) {
            (None, Some(annotation)) => {
                existing.source_annotation = Some(annotation.clone());
                changed = true;
            }
            (Some(left), Some(right)) if left != right => {
                return Err(format!(
                    "streaming plugin changed source annotation at pc 0x{:x}",
                    incoming.pc
                ));
            }
            _ => {}
        }

        for exception in &incoming.exceptions {
            if !existing.exceptions.contains(exception) {
                existing.exceptions.push(exception.clone());
                changed = true;
            }
        }

        for write in &incoming.register_writes {
            if let Some(current) = existing
                .register_writes
                .iter_mut()
                .find(|current| current.name == write.name)
            {
                if current != write {
                    *current = write.clone();
                    changed = true;
                }
            } else {
                existing.register_writes.push(write.clone());
                changed = true;
            }
        }
        existing
            .register_writes
            .sort_by(|left, right| left.name.cmp(&right.name));

        for write in &incoming.memory_writes {
            if let Some(current) = existing
                .memory_writes
                .iter_mut()
                .find(|current| current.addr == write.addr)
            {
                if current != write {
                    *current = write.clone();
                    changed = true;
                }
            } else {
                existing.memory_writes.push(write.clone());
                changed = true;
            }
        }
        existing.memory_writes.sort_by_key(|write| write.addr);

        changed |= Self::merge_timing(&mut existing.timing, &incoming.timing);
        Ok(changed)
    }

    fn merge_instruction(
        hart_output: &mut HartExecutionOutput,
        instruction: AsmInstructionExecution,
    ) -> Result<bool, String> {
        if let Some(existing) = hart_output
            .instructions
            .iter_mut()
            .find(|row| row.pc == instruction.pc)
        {
            if existing.trace_inst_id.is_none() && instruction.trace_inst_id.is_none() {
                return Self::merge_asm_row(existing, &instruction);
            }
            return Err(format!(
                "streaming plugin emitted multiple results for the same asm row at pc 0x{:x}",
                instruction.pc
            ));
        }

        hart_output.instructions.push(instruction);
        Ok(true)
    }

    /// Feed a single event into the collector. Returns false if execution ended
    /// (either `Completed` or `Failed`), true if more events are expected.
    pub fn push(&mut self, event: InstructionEvent) -> bool {
        match event {
            InstructionEvent::Instruction {
                hart_id,
                instruction,
            } => {
                if let Err(message) =
                    Self::merge_instruction(self.ensure_hart(hart_id), instruction)
                {
                    self.error = Some(ImplExecutionError::Generic(message));
                    return false;
                }
                true
            }
            InstructionEvent::Completed { perf } => {
                self.perf = Some(perf);
                false
            }
            InstructionEvent::Failed { error } => {
                self.error = Some(error);
                false
            }
        }
    }

    /// Consume the collector, returning a final per-hart map if no error
    /// occurred.
    pub fn finish(
        self,
    ) -> Result<(BTreeMap<u32, HartExecutionOutput>, PerfMetrics), ImplExecutionError> {
        if let Some(err) = self.error {
            return Err(err);
        }
        Ok((self.per_hart, self.perf.unwrap_or_default()))
    }
}

impl Default for StreamingCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EmissionKey {
    hart_id: u32,
    pc: u64,
}

/// Emits only the newly stable rows from repeated `HartExecutionOutput`
/// snapshots.
pub struct StreamingOutputDelta {
    emitted: BTreeMap<EmissionKey, AsmInstructionExecution>,
}

impl StreamingOutputDelta {
    pub fn new() -> Self {
        Self {
            emitted: BTreeMap::new(),
        }
    }

    pub fn emit_per_hart_delta(
        &mut self,
        per_hart: &BTreeMap<u32, HartExecutionOutput>,
        _final_snapshot: bool,
        emit: &mut dyn FnMut(InstructionEvent) -> bool,
    ) -> bool {
        for (hart_id, hart_output) in per_hart {
            if !self.emit_hart_delta(*hart_id, hart_output, emit) {
                return false;
            }
        }
        true
    }

    fn emit_hart_delta(
        &mut self,
        hart_id: u32,
        hart_output: &HartExecutionOutput,
        emit: &mut dyn FnMut(InstructionEvent) -> bool,
    ) -> bool {
        for instruction in &hart_output.instructions {
            let key = EmissionKey {
                hart_id,
                pc: instruction.pc,
            };
            if let Some(existing) = self.emitted.get_mut(&key) {
                let mut merged = existing.clone();
                let changed = match StreamingCollector::merge_asm_row(&mut merged, instruction) {
                    Ok(changed) => changed,
                    Err(message) => {
                        return emit(InstructionEvent::Failed {
                            error: ImplExecutionError::Generic(message),
                        });
                    }
                };
                if !changed {
                    continue;
                }
                if instruction.trace_inst_id.is_none() {
                    *existing = merged.clone();
                    if !emit(InstructionEvent::Instruction {
                        hart_id,
                        instruction: merged,
                    }) {
                        return false;
                    }
                    continue;
                }
                return emit(InstructionEvent::Failed {
                    error: ImplExecutionError::Generic(format!(
                        "streaming plugin attempted to update asm row pc 0x{:x} after it was already emitted; streamed asm rows must be final",
                        instruction.pc
                    )),
                });
            }
            if !emit(InstructionEvent::Instruction {
                hart_id,
                instruction: instruction.clone(),
            }) {
                return false;
            }
            self.emitted.insert(key, instruction.clone());
        }

        true
    }
}

impl Default for StreamingOutputDelta {
    fn default() -> Self {
        Self::new()
    }
}

fn instruction_ready_by_timing(
    instruction: &AsmInstructionExecution,
    stable_before_cycle: Option<u64>,
    include_untimed: bool,
) -> bool {
    let Some(limit) = stable_before_cycle else {
        return include_untimed;
    };
    // A streamed row is only stable once its retirement window has closed.
    // Using `clk_start` first can emit a row before later trace parsing fills in
    // its final writes or end cycle, which then looks like a forbidden update.
    if let Some(cycle) = instruction.timing.clk_end {
        cycle < limit
    } else if instruction.timing.clk_start.is_some() {
        false
    } else {
        include_untimed
    }
}

/// Return a snapshot prefix whose rows are ready to emit according to timing.
pub fn filter_hart_output_by_timing(
    hart: &HartExecutionOutput,
    stable_before_cycle: Option<u64>,
    include_untimed: bool,
) -> HartExecutionOutput {
    HartExecutionOutput {
        hart_id: hart.hart_id,
        instructions: hart
            .instructions
            .iter()
            .filter(|instruction| {
                instruction_ready_by_timing(instruction, stable_before_cycle, include_untimed)
            })
            .cloned()
            .collect(),
    }
}

pub fn filter_per_hart_output_by_timing(
    per_hart: &BTreeMap<u32, HartExecutionOutput>,
    stable_before_cycle: Option<u64>,
    include_untimed: bool,
) -> BTreeMap<u32, HartExecutionOutput> {
    per_hart
        .iter()
        .map(|(hart_id, hart)| {
            (
                *hart_id,
                filter_hart_output_by_timing(hart, stable_before_cycle, include_untimed),
            )
        })
        .collect()
}

pub fn complete_line_prefix_len(bytes: &[u8], include_trailing_partial: bool) -> Option<usize> {
    if bytes.is_empty() {
        return None;
    }
    let complete_len = if include_trailing_partial {
        bytes.len()
    } else {
        bytes
            .iter()
            .rposition(|byte| *byte == b'\n')
            .map(|index| index + 1)
            .unwrap_or(0)
    };
    (complete_len > 0).then_some(complete_len)
}

pub fn read_complete_utf8_prefix(
    path: &Path,
    include_trailing_partial: bool,
) -> io::Result<Option<(String, usize)>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let Some(complete_len) = complete_line_prefix_len(&bytes, include_trailing_partial) else {
        return Ok(None);
    };
    Ok(Some((
        String::from_utf8_lossy(&bytes[..complete_len]).into_owned(),
        complete_len,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InstructionTiming, TraceInstId};
    use riscv_instruction::{AsmDecodeWithContext, IsaDecodeContext, RiscvInstruction};

    fn addi() -> RiscvInstruction {
        RiscvInstruction::decode_asm_with_context("addi x1, x0, 1", IsaDecodeContext::RV32)
            .expect("test instruction should decode")
    }

    fn asm(pc: u64) -> AsmInstructionExecution {
        AsmInstructionExecution {
            pc,
            instruction: addi(),
            source_annotation: None,
            trace_inst_id: None,
            exceptions: Vec::new(),
            register_writes: Vec::new(),
            memory_writes: Vec::new(),
            timing: InstructionTiming::default(),
        }
    }

    #[test]
    fn collector_rejects_duplicate_rows_by_pc() {
        let mut first = asm(0x1000);
        first.trace_inst_id = Some(TraceInstId::new(0, 0));
        let mut second = first.clone();
        second.register_writes.push(crate::RegisterValue {
            name: "x1".to_string(),
            value: 1,
        });

        let mut collector = StreamingCollector::new();
        assert!(collector.push(InstructionEvent::Instruction {
            hart_id: 0,
            instruction: first,
        }));
        assert!(!collector.push(InstructionEvent::Instruction {
            hart_id: 0,
            instruction: second,
        }));
        let error = collector
            .finish()
            .expect_err("duplicate streamed asm row must be rejected");
        assert!(
            error
                .to_string()
                .contains("multiple results for the same asm row")
        );
    }

    #[test]
    fn collector_merges_duplicate_framework_rows_by_pc() {
        let first = asm(0x1000);
        let mut second = first.clone();
        second.register_writes.push(crate::RegisterValue {
            name: "x1".to_string(),
            value: 2,
        });

        let mut collector = StreamingCollector::new();
        assert!(collector.push(InstructionEvent::Instruction {
            hart_id: 0,
            instruction: first,
        }));
        assert!(collector.push(InstructionEvent::Instruction {
            hart_id: 0,
            instruction: second,
        }));

        let (per_hart, _) = collector.finish().expect("framework rows should merge");
        let hart0 = per_hart.get(&0).expect("hart 0 output");
        assert_eq!(hart0.instructions.len(), 1);
        assert_eq!(hart0.instructions[0].register_writes[0].value, 2);
    }

    #[test]
    fn delta_rejects_late_updates_to_emitted_rows() {
        let mut delta = StreamingOutputDelta::new();
        let mut first = asm(0x1000);
        first.trace_inst_id = Some(TraceInstId::new(0, 0));
        let mut updated = first.clone();
        updated.register_writes.push(crate::RegisterValue {
            name: "x1".to_string(),
            value: 1,
        });

        let initial = BTreeMap::from([(
            0u32,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![first],
            },
        )]);
        let changed = BTreeMap::from([(
            0u32,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![updated],
            },
        )]);

        let mut events = Vec::new();
        assert!(delta.emit_per_hart_delta(&initial, false, &mut |event| {
            events.push(event);
            true
        }));
        assert!(delta.emit_per_hart_delta(&changed, false, &mut |event| {
            events.push(event);
            true
        }));
        assert!(matches!(
            events.last(),
            Some(InstructionEvent::Failed { error })
                if error
                    .to_string()
                    .contains("attempted to update asm row pc 0x1000")
        ));
    }

    #[test]
    fn delta_allows_late_updates_to_framework_rows() {
        let mut delta = StreamingOutputDelta::new();
        let first = asm(0x1000);
        let mut updated = first.clone();
        updated.register_writes.push(crate::RegisterValue {
            name: "x1".to_string(),
            value: 2,
        });

        let initial = BTreeMap::from([(
            0u32,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![first],
            },
        )]);
        let changed = BTreeMap::from([(
            0u32,
            HartExecutionOutput {
                hart_id: 0,
                instructions: vec![updated],
            },
        )]);

        let mut collector = StreamingCollector::new();
        assert!(delta.emit_per_hart_delta(&initial, false, &mut |event| collector.push(event)));
        assert!(delta.emit_per_hart_delta(&changed, false, &mut |event| collector.push(event)));

        let (per_hart, _) = collector.finish().expect("framework delta should merge");
        let hart0 = per_hart.get(&0).expect("hart 0 output");
        assert_eq!(hart0.instructions.len(), 1);
        assert_eq!(hart0.instructions[0].register_writes[0].value, 2);
    }

    #[test]
    fn timing_filter_waits_for_instruction_end_cycle() {
        let mut instruction = asm(0x1000);
        instruction.timing = InstructionTiming::new(Some(10), None, None);
        let hart = HartExecutionOutput {
            hart_id: 0,
            instructions: vec![instruction.clone()],
        };
        assert!(
            filter_hart_output_by_timing(&hart, Some(20), false)
                .instructions
                .is_empty()
        );

        instruction.timing = InstructionTiming::new(Some(10), Some(12), None);
        let hart = HartExecutionOutput {
            hart_id: 0,
            instructions: vec![instruction],
        };
        assert_eq!(
            filter_hart_output_by_timing(&hart, Some(20), false)
                .instructions
                .len(),
            1
        );
    }

    #[test]
    fn complete_line_prefix_handles_partial_tail() {
        assert_eq!(complete_line_prefix_len(b"a\nb", false), Some(2));
        assert_eq!(complete_line_prefix_len(b"a\nb", true), Some(3));
        assert_eq!(complete_line_prefix_len(b"", true), None);
    }
}
