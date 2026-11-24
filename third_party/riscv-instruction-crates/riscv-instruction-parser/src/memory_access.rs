use regex::Regex;
use riscv_instruction_schema::{AssemblySyntax, Instruction, MemoryAddressInfo};

pub fn with_memory_access_info(mut instructions: Vec<Instruction>) -> (Vec<Instruction>, usize) {
    let count = annotate_memory_access_info(&mut instructions);
    (instructions, count)
}

pub fn annotate_memory_access_info(instructions: &mut [Instruction]) -> usize {
    let addr_re = Regex::new(
        r"(?x)
        (?P<offset>\{[^}]+\})?
        \s*
        \(
        \s*
        (?P<base>sp|\{[^}]+\})
        \s*
        \)
        ",
    )
    .expect("memory access regex should compile");

    let mut mem_instruction_count = 0;

    for inst in instructions {
        let syntax_str = match &inst.assembly_syntax {
            AssemblySyntax::Format(s) => s.as_str(),
            AssemblySyntax::RustCode(s) => s.as_str(),
        };

        let mut mem_info: Option<MemoryAddressInfo> = None;

        if inst.name == "lpad" {
            mem_info = Some(MemoryAddressInfo {
                base_register_fixed: Some("PC".to_string()),
                base_register_operand: None,
                offset_operand: Some("uimm".to_string()),
            });
        } else if inst.name == "c.addi4spn" || inst.name == "c.addi16sp" {
            mem_info = Some(MemoryAddressInfo {
                base_register_fixed: Some("sp".to_string()),
                base_register_operand: None,
                offset_operand: inst
                    .operands
                    .iter()
                    .find(|operand| operand.name == "uimm" || operand.name == "imm")
                    .map(|operand| operand.name.clone()),
            });
        } else if let Some(caps) = addr_re.captures(syntax_str) {
            let mut base_register_operand = None;
            let mut base_register_fixed = None;

            if let Some(base_match) = caps.name("base") {
                let base_str = base_match.as_str();
                if base_str == "sp" {
                    base_register_fixed = Some("sp".to_string());
                } else if base_str.starts_with('{') && base_str.ends_with('}') {
                    let mut name = base_str[1..base_str.len() - 1].to_string();
                    if name.ends_with("_enc") {
                        name.truncate(name.len() - 4);
                    }
                    base_register_operand = Some(name);
                }
            }

            if base_register_operand.is_some() || base_register_fixed.is_some() {
                let offset_operand = caps
                    .name("offset")
                    .map(|m| m.as_str())
                    .map(|s| {
                        let mut name = s[1..s.len() - 1].to_string();
                        if name.ends_with("_enc") {
                            name.truncate(name.len() - 4);
                        }
                        name
                    });

                mem_info = Some(MemoryAddressInfo {
                    base_register_operand,
                    base_register_fixed,
                    offset_operand,
                });
            }
        }

        if mem_info.is_some() {
            mem_instruction_count += 1;
        }
        inst.memory_access = mem_info;
    }

    mem_instruction_count
}
