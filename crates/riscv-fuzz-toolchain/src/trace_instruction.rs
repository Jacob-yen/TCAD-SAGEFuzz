use riscv_fuzz_plugin_api::ISABase;
use riscv_instruction::{AsmDecodeWithContext, IsaDecodeContext, RiscvInstruction};
use serde::{Deserialize, Serialize};

pub fn isa_base_to_decode_ctx(isa_base: ISABase) -> IsaDecodeContext {
    match isa_base {
        ISABase::Rv32 => IsaDecodeContext::RV32,
        ISABase::Rv64 => IsaDecodeContext::RV64,
    }
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

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct TraceInstructionInfo {
    pub instruction: Option<RiscvInstruction>,
    pub instruction_text: String,
}

impl TraceInstructionInfo {
    pub fn build(instructions: &[String], isa_base: ISABase) -> Vec<Self> {
        let ctx = isa_base_to_decode_ctx(isa_base);
        let traceable = filter_traceable_input_instructions(instructions);
        let mut result = Vec::with_capacity(traceable.len());
        for inst_str in traceable {
            let instruction =
                RiscvInstruction::decode_asm_with_context(&normalize_for_parse(&inst_str), ctx)
                    .ok();
            let info = TraceInstructionInfo {
                instruction,
                instruction_text: inst_str.clone(),
            };
            result.push(info);
        }
        result
    }
}

pub fn is_traceable_input_instruction(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty()
        && !trimmed.starts_with('#')
        && !trimmed.starts_with("//")
        && !trimmed.starts_with(';')
        && !trimmed.starts_with('.')
        && !trimmed.ends_with(':')
}

pub fn filter_traceable_input_instructions(instructions: &[String]) -> Vec<String> {
    instructions
        .iter()
        .filter(|line| is_traceable_input_instruction(line))
        .cloned()
        .collect()
}

pub fn filter_traceable_instruction_pairs(
    instructions: &[String],
    offsets: &[Option<i64>],
) -> (Vec<String>, Vec<Option<i64>>) {
    let mut filtered_insts = Vec::new();
    let mut filtered_offsets = Vec::new();

    for (idx, line) in instructions.iter().enumerate() {
        if is_traceable_input_instruction(line) {
            filtered_insts.push(line.clone());
            filtered_offsets.push(offsets.get(idx).copied().flatten());
        }
    }

    (filtered_insts, filtered_offsets)
}
