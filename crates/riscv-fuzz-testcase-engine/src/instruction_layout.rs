//! Instruction layout / base-address rewrite helpers shared by generation
//! and coverage-guided mutation paths.

#![allow(dead_code)]

use riscv_instruction::{MemoryAccessInstruction, RiscvInstruction, RuntimeOperandAccess};
use riscv_instruction_types::{
    InstructionSequence, LoadImmediatePurpose, PseudoInstruction, XLenImmediate,
};

pub(crate) fn sequence_to_lines(sequence: &InstructionSequence<RiscvInstruction>) -> Vec<String> {
    let mut lines =
        Vec::with_capacity(sequence.pre_instructions.len() + 1 + sequence.post_instructions.len());
    for pseudo in &sequence.pre_instructions {
        lines.push(pseudo.to_string());
    }
    lines.push(sequence.instruction.to_string());
    for pseudo in &sequence.post_instructions {
        lines.push(pseudo.to_string());
    }
    lines
}

pub(crate) fn structured_entries_to_lines(entries: &[RiscvInstruction]) -> Vec<String> {
    entries.iter().map(ToString::to_string).collect()
}

pub(crate) fn offset_mem_base_immediates(
    sequence: &mut InstructionSequence<RiscvInstruction>,
    offset: i64,
) {
    if offset == 0 {
        return;
    }

    adjust_base_address_immediates(&mut sequence.pre_instructions, offset);
}

pub(crate) fn offset_structured_base_immediates(entries: &mut [RiscvInstruction], offset: i64) {
    if offset == 0 {
        return;
    }

    adjust_base_address_immediates(entries, offset);
}

pub(crate) fn align_mem_base_immediates(
    sequence: &mut InstructionSequence<RiscvInstruction>,
    align: u64,
) {
    if align <= 1 {
        return;
    }

    let Some(offset) = effective_mem_offset(sequence) else {
        return;
    };

    align_base_address_immediates(&mut sequence.pre_instructions, offset, align);
}

pub(crate) fn effective_mem_offset(
    sequence: &InstructionSequence<RiscvInstruction>,
) -> Option<i64> {
    if let Some(offset) = sequence.instruction.offset_operand_value() {
        return Some(offset);
    }

    let has_base_address = sequence.pre_instructions.iter().any(|instruction| {
        matches!(
            instruction,
            RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
                purpose: LoadImmediatePurpose::BaseAddress,
                ..
            })
        )
    });
    if !has_base_address || sequence.instruction.memory_operands().is_empty() {
        return None;
    }

    Some(0)
}

pub(crate) fn adjust_base_address_immediates(instructions: &mut [RiscvInstruction], offset: i64) {
    for instruction in instructions {
        if let RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
            immediate, purpose, ..
        }) = instruction
            && *purpose == LoadImmediatePurpose::BaseAddress
        {
            let next = immediate.as_i64().saturating_add(offset);
            *immediate = match immediate {
                XLenImmediate::X32(_) => XLenImmediate::X32(next as i32),
                XLenImmediate::X64(_) => XLenImmediate::X64(next),
            };
        }
    }
}

pub(crate) fn align_base_address_immediates(
    instructions: &mut [RiscvInstruction],
    offset: i64,
    align: u64,
) {
    let align = i128::from(align.max(1));
    for instruction in instructions {
        if let RiscvInstruction::Pseudo(PseudoInstruction::LoadImmediate {
            immediate,
            purpose,
            ..
        }) = instruction
        {
            if *purpose != LoadImmediatePurpose::BaseAddress {
                continue;
            }

            let effective_address = i128::from(immediate.as_i64()) + i128::from(offset);
            let aligned_effective_address = effective_address.div_euclid(align) * align;
            let aligned_base = aligned_effective_address - i128::from(offset);
            let aligned_base =
                aligned_base.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
            *immediate = match immediate {
                XLenImmediate::X32(_) => XLenImmediate::X32(aligned_base as i32),
                XLenImmediate::X64(_) => XLenImmediate::X64(aligned_base),
            };
        }
    }
}
