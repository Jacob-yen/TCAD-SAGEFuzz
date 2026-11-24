//! Instruction semantic inference from unified-db fields.
//!
//! The parser keeps raw YAML conversion close to upstream data, but a few
//! higher-level behaviors are still worth extracting here so later stages do
//! not fall back to instruction-name matching.

use crate::operand_inference::infer_operand_kind;
use regex::Regex;
use riscv_instruction_schema::{
    InstructionControlFlowSemantics, InstructionCsrSemantics, InstructionSemantics, Operand,
    OperandKind, RegisterBank, RegisterSpec, YamlInstruction,
};
use std::sync::OnceLock;

pub(crate) fn infer_instruction_semantics(
    yaml_inst: &YamlInstruction,
    operands: &[Operand],
) -> InstructionSemantics {
    InstructionSemantics {
        csr: infer_csr_semantics(yaml_inst, operands),
        control_flow: infer_control_flow_semantics(yaml_inst, operands),
        touches_floating_point: operands.iter().any(is_floating_point_register_operand),
        touches_vector_registers: operands.iter().any(is_vector_register_operand),
    }
}

fn infer_csr_semantics(
    yaml_inst: &YamlInstruction,
    operands: &[Operand],
) -> Option<InstructionCsrSemantics> {
    let address_operand = operands
        .iter()
        .find(|operand| is_csr_operand(operand))
        .map(|operand| operand.name.clone())?;

    let operation = yaml_inst.operation.as_deref().unwrap_or_default();
    let reads = csr_read_regex().is_match(operation);
    let writes = csr_write_regex().is_match(operation);
    let source_register_operand =
        writes.then(|| first_source_integer_register_operand_name(operands));

    Some(InstructionCsrSemantics {
        address_operand,
        reads,
        writes,
        source_register_operand: source_register_operand.flatten(),
    })
}

fn infer_control_flow_semantics(
    yaml_inst: &YamlInstruction,
    operands: &[Operand],
) -> Option<InstructionControlFlowSemantics> {
    let operation = yaml_inst.operation.as_deref()?;
    let jump_expr = capture_jump_expression(operation)?;
    let (target_register_operand, target_fixed_register) =
        resolve_jump_target(operation, &jump_expr, operands);

    Some(InstructionControlFlowSemantics {
        jumps: true,
        branches: jump_expr.contains("$pc") && contains_if_guard(operation),
        target_register_operand,
        target_fixed_register,
    })
}

fn is_csr_operand(operand: &Operand) -> bool {
    operand.name.eq_ignore_ascii_case("csr")
        || matches!(resolved_operand_kind(operand), OperandKind::Csr(_))
}

fn is_floating_point_register_operand(operand: &Operand) -> bool {
    matches!(
        resolved_operand_kind(operand),
        OperandKind::Register(RegisterSpec {
            bank: RegisterBank::FloatingPoint
        })
    )
}

fn is_vector_register_operand(operand: &Operand) -> bool {
    matches!(
        resolved_operand_kind(operand),
        OperandKind::Register(RegisterSpec {
            bank: RegisterBank::Vector
        })
    )
}

fn first_source_integer_register_operand_name(operands: &[Operand]) -> Option<String> {
    operands
        .iter()
        .find(|operand| {
            operand.name != "csr"
                && matches!(
                    resolved_operand_kind(operand),
                    OperandKind::Register(RegisterSpec {
                        bank: RegisterBank::Integer
                    })
                )
                && looks_like_source_operand_name(&operand.name)
        })
        .map(|operand| operand.name.clone())
}

fn looks_like_source_operand_name(name: &str) -> bool {
    name.starts_with("rs")
        || name.starts_with("xs")
        || name.starts_with("r1")
        || name.starts_with("r2")
}

fn resolved_operand_kind(operand: &Operand) -> OperandKind {
    match operand.kind {
        OperandKind::Unknown => infer_operand_kind(&operand.name),
        _ => operand.kind.clone(),
    }
}

fn capture_jump_expression(operation: &str) -> Option<String> {
    jump_regex()
        .captures(operation)
        .and_then(|captures| captures.name("expr"))
        .map(|expr| expr.as_str().trim().to_string())
}

fn resolve_jump_target(
    operation: &str,
    jump_expr: &str,
    operands: &[Operand],
) -> (Option<String>, Option<String>) {
    if jump_expr.contains("$pc") {
        return (None, None);
    }

    if let Some(target) = resolve_register_reference(jump_expr, operands) {
        return target;
    }

    let identifier = jump_expr.trim();
    if !identifier_regex().is_match(identifier) {
        return (None, None);
    }

    let assignment_regex = Regex::new(&format!(
        r"(?m)^\s*(?:[A-Za-z0-9_<>'\[\]]+\s+)?{}\s*=\s*(?P<expr>[^;]+);",
        regex::escape(identifier)
    ))
    .expect("assignment lookup regex should compile");

    assignment_regex
        .captures(operation)
        .and_then(|captures| captures.name("expr"))
        .map(|expr| resolve_jump_target(operation, expr.as_str().trim(), operands))
        .unwrap_or((None, None))
}

fn resolve_register_reference(
    expression: &str,
    operands: &[Operand],
) -> Option<(Option<String>, Option<String>)> {
    let captures = x_register_reference_regex().captures(expression)?;
    let register_ref = captures.name("reg")?.as_str();

    if operands.iter().any(|operand| operand.name == register_ref) {
        return Some((Some(register_ref.to_string()), None));
    }

    if register_ref.chars().all(|ch| ch.is_ascii_digit()) {
        return Some((None, Some(format!("x{}", register_ref))));
    }

    register_ref
        .strip_prefix('x')
        .filter(|suffix| suffix.chars().all(|ch| ch.is_ascii_digit()))
        .map(|suffix| (None, Some(format!("x{}", suffix))))
}

fn csr_read_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\bcsr_[A-Za-z0-9_]*read\s*\(").unwrap())
}

fn csr_write_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\bcsr_[A-Za-z0-9_]*write\s*\(").unwrap())
}

fn jump_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?s)\bjump\s*\((?P<expr>[^;]*?)\);").unwrap())
}

fn x_register_reference_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"X\[(?P<reg>[A-Za-z0-9_]+)\]").unwrap())
}

fn identifier_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").unwrap())
}

fn contains_if_guard(operation: &str) -> bool {
    operation.contains("if (") || operation.contains("if(")
}

#[cfg(test)]
mod tests {
    use super::infer_instruction_semantics;
    use riscv_instruction_schema::{
        InstructionControlFlowSemantics, InstructionCsrSemantics, Operand,
    };
    use std::collections::HashMap;

    fn operand(name: &str) -> Operand {
        Operand {
            name: name.to_string(),
            kind: Default::default(),
            bit_lengths: HashMap::new(),
            restrictions: None,
            per_isa_restrictions: HashMap::new(),
        }
    }

    fn yaml(name: &str, operation: &str) -> riscv_instruction_schema::YamlInstruction {
        riscv_instruction_schema::YamlInstruction {
            schema: "inst_schema.json#".to_string(),
            kind: "instruction".to_string(),
            name: name.to_string(),
            long_name: name.to_string(),
            description: riscv_instruction_schema::Description::String(name.to_string()),
            defined_by: serde_yaml::Value::Null,
            base: None,
            assembly: name.to_string(),
            format: None,
            encoding: None,
            access: None,
            data_independent_timing: None,
            operation: Some(operation.to_string()),
            sail: None,
        }
    }

    #[test]
    fn csr_semantics_are_extracted_from_operation_text() {
        let semantics = infer_instruction_semantics(
            &yaml(
                "csrrw",
                "X[xd] = csr_sw_read(csr_handle);\ncsr_sw_write(csr_handle, X[xs1]);",
            ),
            &[operand("xd"), operand("csr"), operand("xs1")],
        );

        assert_eq!(
            semantics.csr,
            Some(InstructionCsrSemantics {
                address_operand: "csr".to_string(),
                reads: true,
                writes: true,
                source_register_operand: Some("xs1".to_string()),
            })
        );
    }

    #[test]
    fn jump_target_registers_are_resolved_through_temporaries() {
        let semantics = infer_instruction_semantics(
            &yaml(
                "jalr",
                "XReg addr = (X[xs1] + $signed(imm)) & ~MXLEN'1;\njump(addr);",
            ),
            &[operand("xd"), operand("imm"), operand("xs1")],
        );

        assert_eq!(
            semantics.control_flow,
            Some(InstructionControlFlowSemantics {
                jumps: true,
                branches: false,
                target_register_operand: Some("xs1".to_string()),
                target_fixed_register: None,
            })
        );
    }

    #[test]
    fn fixed_jump_registers_are_preserved() {
        let semantics =
            infer_instruction_semantics(&yaml("cm.popret", "jump(X[1]);"), &[operand("reg_list")]);

        assert_eq!(
            semantics.control_flow,
            Some(InstructionControlFlowSemantics {
                jumps: true,
                branches: false,
                target_register_operand: None,
                target_fixed_register: Some("x1".to_string()),
            })
        );
    }
}
