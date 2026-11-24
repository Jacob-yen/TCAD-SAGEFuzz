use std::collections::{BTreeMap, HashMap};

use regex::Regex;
use riscv_fuzz_plugin_api::context::{
    MemoryContext, MemoryContextEntry, MemoryValueWidth, RegisterContext, RegisterContextEntry,
};
use riscv_fuzz_plugin_api::{ISABase, canonicalize_register_name};
use riscv_instruction::{RegisterBank, RiscvInstruction, RuntimeMemoryBase, RuntimeOperandAccess};
use thiserror::Error;

/// Regex construction failures while extracting instruction contexts.
#[derive(Debug, Error)]
pub enum RegexError {
    #[error("failed to compile regex pattern '{pattern}': {source}")]
    CompilationFailed {
        pattern: String,
        #[source]
        source: regex::Error,
    },
}

/// Failures while deriving register/memory context from an instruction.
#[derive(Debug, Error)]
pub enum ContextExtractionError {
    #[error(transparent)]
    Regex(#[from] RegexError),

    #[error("failed to parse memory offset '{text}': {source}")]
    OffsetParseError {
        text: String,
        #[source]
        source: std::num::ParseIntError,
    },
}

pub fn extract_register_context(
    instruction: &RiscvInstruction,
    registers: &HashMap<String, u64>,
) -> Result<Option<RegisterContext>, ContextExtractionError> {
    let mut reg_names = instruction
        .register_operands()
        .into_iter()
        .map(|operand| {
            runtime_register_name(operand.register.bank, operand.register.physical_number)
        })
        .collect::<Vec<_>>();
    for memory_operand in instruction.memory_operands() {
        if let Some(RuntimeMemoryBase::Register(register)) = memory_operand.fixed_base {
            reg_names.push(runtime_register_name(
                register.bank,
                register.physical_number,
            ));
        }
    }
    reg_names.sort();
    reg_names.dedup();

    if reg_names.is_empty() {
        return Ok(None);
    }

    let mut entries = Vec::new();
    for reg in reg_names {
        if let Some(&value) = registers.get(&reg) {
            entries.push(RegisterContextEntry { name: reg, value });
        }
    }

    if entries.is_empty() {
        Ok(None)
    } else {
        Ok(Some(RegisterContext { entries }))
    }
}

pub fn extract_memory_context(
    instruction: &RiscvInstruction,
    offset_hint: Option<i64>,
    registers_before: &HashMap<String, u64>,
    memory_before: &BTreeMap<u64, u8>,
    isa_base: ISABase,
    program_mem_range: (u64, u64),
) -> Result<Option<MemoryContext>, ContextExtractionError> {
    let Some(memory_operand) = instruction.memory_operands().into_iter().next() else {
        return Ok(None);
    };

    let base_name = if let Some(base_register) = memory_operand.base_register.as_ref() {
        runtime_register_name(
            base_register.register.bank,
            base_register.register.physical_number,
        )
    } else if let Some(RuntimeMemoryBase::Register(register)) = memory_operand.fixed_base {
        runtime_register_name(register.bank, register.physical_number)
    } else {
        return Ok(None);
    };

    let base_val = if let Some(val) = registers_before.get(&base_name) {
        *val
    } else {
        return Ok(None);
    };

    let offset = offset_hint.or(memory_operand.offset_value).unwrap_or(0);

    let effective_addr = if offset >= 0 {
        base_val.wrapping_add(offset as u64)
    } else {
        base_val.wrapping_sub((-offset) as u64)
    };

    let (mem_start, mem_end) = program_mem_range;
    if mem_end < mem_start {
        return Ok(None);
    }

    if effective_addr < mem_start || effective_addr > mem_end {
        return Ok(None);
    }

    if base_val < mem_start || base_val > mem_end {
        return Ok(None);
    }

    let normalized_base = effective_addr - mem_start;
    let base_offset = base_val - mem_start;
    let max_offset = mem_end - mem_start;

    let mut entries = Vec::new();

    fn gather_width(
        width: usize,
        normalized_base: u64,
        max_offset: u64,
        memory_before: &BTreeMap<u64, u8>,
        kind: MemoryValueWidth,
    ) -> Option<MemoryContextEntry> {
        if width == 0 {
            return None;
        }

        let mut collected = Vec::with_capacity(width);
        for i in 0..width {
            let addr = normalized_base + i as u64;
            if addr > max_offset {
                return None;
            }
            let byte = *memory_before.get(&addr)?;
            collected.push((byte, addr));
        }

        let mut value = 0u64;
        for (idx, (byte, _)) in collected.iter().enumerate() {
            value |= (*byte as u64) << (idx * 8);
        }

        let addresses = collected.into_iter().map(|(_, addr)| addr).collect();

        Some(MemoryContextEntry {
            width: kind,
            addresses,
            value,
        })
    }

    if let Some(entry) = gather_width(
        1,
        normalized_base,
        max_offset,
        memory_before,
        MemoryValueWidth::Byte,
    ) {
        entries.push(entry);
    }

    if let Some(entry) = gather_width(
        2,
        normalized_base,
        max_offset,
        memory_before,
        MemoryValueWidth::Half,
    ) {
        entries.push(entry);
    }

    if let Some(entry) = gather_width(
        4,
        normalized_base,
        max_offset,
        memory_before,
        MemoryValueWidth::Word,
    ) {
        entries.push(entry);
    }

    if isa_base == ISABase::Rv64
        && let Some(entry) = gather_width(
            8,
            normalized_base,
            max_offset,
            memory_before,
            MemoryValueWidth::Dword,
        )
    {
        entries.push(entry);
    }

    if entries.is_empty() {
        Ok(None)
    } else {
        Ok(Some(MemoryContext {
            base_register: base_name,
            base_offset,
            entries,
        }))
    }
}

fn get_reg_pattern() -> Result<Regex, ContextExtractionError> {
    Regex::new(r"\b([xf]\d+|zero|ra|sp|gp|tp|t[0-6]|s(?:[0-9]|1[01])|a[0-7]|fp)\b")
        .map_err(|source| RegexError::CompilationFailed {
            pattern: r"\b([xf]\d+|zero|ra|sp|gp|tp|t[0-6]|s(?:[0-9]|1[01])|a[0-7]|fp)\b"
                .to_string(),
            source,
        })
        .map_err(ContextExtractionError::from)
}

fn normalize_register_name(name: &str) -> String {
    canonicalize_register_name(name).unwrap_or_else(|| name.to_string())
}

fn runtime_register_name(bank: RegisterBank, physical_number: u8) -> String {
    match bank {
        RegisterBank::Integer => format!("x{}", physical_number),
        RegisterBank::FloatingPoint => format!("f{}", physical_number),
        RegisterBank::Vector => format!("v{}", physical_number),
    }
}

pub fn extract_registers_from_instruction(
    instruction: &str,
) -> Result<Vec<String>, ContextExtractionError> {
    let reg_pattern = get_reg_pattern()?;

    let mut registers = Vec::new();
    for cap in reg_pattern.captures_iter(instruction) {
        if let Some(reg) = cap.get(1) {
            let reg_name = normalize_register_name(reg.as_str());
            if !registers.contains(&reg_name) {
                registers.push(reg_name);
            }
        }
    }

    Ok(registers)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use riscv_fuzz_plugin_api::ISABase;
    use riscv_instruction::{AsmDecodeWithContext, IsaDecodeContext, RiscvInstruction};

    use super::{extract_memory_context, extract_register_context};

    fn rv32(asm: &str) -> RiscvInstruction {
        RiscvInstruction::decode_asm_with_context(asm, IsaDecodeContext::RV32)
            .unwrap_or_else(|err| panic!("failed to parse `{asm}`: {err}"))
    }

    #[test]
    fn extract_register_context_includes_fixed_stack_base_register() {
        let instruction = rv32("c.lwsp x13, 12(sp)");
        let registers =
            HashMap::from([("x2".to_string(), 0x1000), ("x13".to_string(), 0xffff_fff3)]);

        let context = extract_register_context(&instruction, &registers)
            .expect("register context extraction should succeed")
            .expect("register context should exist");

        assert_eq!(
            context
                .entries
                .iter()
                .map(|entry| (entry.name.as_str(), entry.value))
                .collect::<Vec<_>>(),
            vec![("x13", 0xffff_fff3), ("x2", 0x1000)]
        );
    }

    #[test]
    fn extract_memory_context_uses_fixed_stack_base_register() {
        let instruction = rv32("c.lwsp x13, 12(sp)");
        let registers = HashMap::from([("x2".to_string(), 0x1000)]);
        let memory = BTreeMap::from([(0x0c, 0x78), (0x0d, 0x56), (0x0e, 0x34), (0x0f, 0x12)]);

        let context = extract_memory_context(
            &instruction,
            None,
            &registers,
            &memory,
            ISABase::Rv32,
            (0x1000, 0x103f),
        )
        .expect("memory context extraction should succeed")
        .expect("memory context should exist");

        assert_eq!(context.base_register, "x2");
        assert_eq!(context.base_offset, 0);
        assert_eq!(context.entries.len(), 3);
        assert_eq!(context.entries[0].addresses, vec![0x0c]);
        assert_eq!(context.entries[1].addresses, vec![0x0c, 0x0d]);
        assert_eq!(context.entries[2].addresses, vec![0x0c, 0x0d, 0x0e, 0x0f]);
        assert_eq!(context.entries[2].value, 0x1234_5678);
    }
}
