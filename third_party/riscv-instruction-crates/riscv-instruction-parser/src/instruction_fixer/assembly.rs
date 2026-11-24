//! Final assembly syntax reconstruction.
//!
//! All fixer rules mutate an intermediate "operands part + placeholder set"
//! representation first. Only after that state is stable do we rebuild the
//! exported `AssemblySyntax`.

use riscv_instruction_schema::*;
use std::collections::HashSet;

/// Rebuild the final exported `AssemblySyntax` after all repair rules have
/// normalized operands, placeholders, and special-case instruction behavior.
///
/// Most instructions can go back to a plain `Format` string. A smaller set
/// needs `RustCode` because the rendered mnemonic or operands depend on runtime
/// operand values, such as aq/rl suffixes, vector mask suppression, or merged
/// compressed operands.
pub(crate) fn generate_assembly_syntax(
    name: &str,
    assembly_syntax: &str,
    syntax_operands: &HashSet<String>,
    instruction: &Instruction,
) -> AssemblySyntax {
    let operands_segment = wrap_operands_with_braces(assembly_syntax, syntax_operands);

    // `c.lui` needs an extra immediate transformation during rendering, so a
    // static format string is not expressive enough.
    if name == "c.lui" {
        AssemblySyntax::RustCode(
            "format!(\"c.lui {}, 0x{:x}\", xd.encode_asm(), imm.get() as u32 & 0xfffff)"
                .to_string(),
        )
    } else if name == "fcvtmod.w.d" {
        AssemblySyntax::Format(format!("fcvtmod.w.d {}, rtz", operands_segment))
    // The `cm.*` pop/push family uses a merged synthetic operand and must
    // reconstruct the printable register list and stack adjustment at runtime.
    } else if matches!(name, "cm.pop" | "cm.push" | "cm.popretz" | "cm.popret") {
        AssemblySyntax::RustCode(generate_cm_instruction_assembly_code(name))
    // Some compressed load/store instructions model the offset like a boolean
    // choice in IR, so the real displacement text must be reconstructed.
    } else if matches!(name, "c.sh" | "c.lh" | "c.lhu") {
        AssemblySyntax::RustCode(generate_compressed_offset_assembly_code(
            name,
            &operands_segment,
        ))
    } else if matches!(name, "vs1r.v" | "vs2r.v" | "vs4r.v" | "vs8r.v") {
        AssemblySyntax::Format(format!("{} {{vs3}}, 0({{xs1}})", name))
    } else if is_aqrl_instruction(instruction) {
        AssemblySyntax::RustCode(generate_aqrl_assembly_code(
            name,
            &operands_segment,
            instruction,
        ))
    } else if is_mop_instruction(instruction) {
        AssemblySyntax::RustCode(generate_mop_assembly_code(name, &operands_segment))
    } else if is_vector_instruction_with_vm(instruction) {
        AssemblySyntax::RustCode(generate_vector_vm_assembly_code(name, &operands_segment))
    } else if matches!(
        name,
        "sspush.x1" | "sspush.x5" | "sspopchk.x1" | "sspopchk.x5"
    ) {
        AssemblySyntax::Format(name.replace('.', " "))
    } else {
        let final_assembly = if operands_segment.trim().is_empty() {
            name.to_string()
        } else {
            format!("{} {}", name, operands_segment)
        };
        AssemblySyntax::Format(final_assembly)
    }
}

/// Wrap placeholder names with braces so they can be reused as format slots.
fn wrap_operands_with_braces(asm_without_name: &str, syntax_operands: &HashSet<String>) -> String {
    if asm_without_name.trim().is_empty() || syntax_operands.is_empty() {
        return asm_without_name.to_string();
    }

    let mut result = asm_without_name.to_string();

    for operand_name in syntax_operands {
        result = result.replace(operand_name, &format!("{{{}}}", operand_name));
    }

    result
}

fn is_aqrl_instruction(instruction: &Instruction) -> bool {
    instruction
        .operands
        .iter()
        .any(|operand| operand.name == "aq" || operand.name == "rl")
}

fn needs_address_wrapping(name: &str) -> bool {
    matches!(
        name,
        "ssamoswap.d" | "ssamoswap.w" | "sc.d" | "sc.w" | "lr.d" | "lr.w"
    )
}

/// 1-bit operands that are used as boolean conditions in `RustCode` blocks,
/// not as assembly-formatted strings. Also includes locally-computed string
/// variables used in format captures.
const BOOL_LIKE_OPERANDS: &[&str] = &["aq", "rl", "vm", "n", "vm_suffix", "suffix"];

fn is_bool_like(field: &str) -> bool {
    BOOL_LIKE_OPERANDS.contains(&field)
}

/// Given a format string containing `{field}` placeholders, collect field names
/// that are NOT bool-like (i.e. those that need `encode_asm()`).
fn asm_encode_fields_from_format(fmt: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut rest = fmt;
    while let Some(open) = rest.find('{') {
        rest = &rest[open + 1..];
        if let Some(close) = rest.find('}') {
            let name = &rest[..close];
            rest = &rest[close + 1..];
            if !name.is_empty() && !name.contains('{') && !is_bool_like(name) {
                if !fields.contains(&name.to_string()) {
                    fields.push(name.to_string());
                }
            }
        }
    }
    fields
}

/// Replace `{field}` with `{field_enc}` in a format string for fields that
/// need `encode_asm()`.
fn replace_with_enc_suffix(fmt: &str, enc_fields: &[String]) -> String {
    let mut result = fmt.to_string();
    for f in enc_fields {
        result = result.replace(&format!("{{{}}}", f), &format!("{{{}_enc}}", f));
    }
    result
}

/// Generate `let field_enc = field.encode_asm();` declarations for enc_fields.
fn enc_let_bindings(enc_fields: &[String]) -> String {
    enc_fields
        .iter()
        .map(|f| format!("    let {f}_enc = {f}.encode_asm();\n"))
        .collect()
}

/// Build runtime formatting code for instructions whose aq/rl suffix depends on
/// boolean operands rather than a fixed mnemonic spelling.
fn generate_aqrl_assembly_code(
    name: &str,
    operands_segment: &str,
    instruction: &Instruction,
) -> String {
    let has_aq = instruction
        .operands
        .iter()
        .any(|operand| operand.name == "aq");
    let has_rl = instruction
        .operands
        .iter()
        .any(|operand| operand.name == "rl");

    if needs_address_wrapping(name) {
        let operands_without_braces = operands_segment.replace('{', "").replace('}', "");

        let operands: Vec<&str> = operands_without_braces
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let (prefix_operands, last_operand) = if !operands.is_empty() {
            let last_idx = operands.len() - 1;
            let prefix = if last_idx > 0 {
                let prefix_with_braces: Vec<String> = operands[..last_idx]
                    .iter()
                    .map(|op| format!("{{{}}}", op))
                    .collect();
                format!(" {}, ", prefix_with_braces.join(", "))
            } else {
                " ".to_string()
            };
            (prefix, operands[last_idx])
        } else {
            (" ".to_string(), "")
        };

        let last_operand = last_operand.replace('(', "").replace(')', "");

        // Collect all non-bool fields that appear in the format string.
        let format_template = format!("{name}{{suffix}}{prefix_operands}0({{{last_operand}}})");
        let enc_fields = asm_encode_fields_from_format(&format_template);
        let bindings = enc_let_bindings(&enc_fields);
        let escaped_prefix = replace_with_enc_suffix(&prefix_operands.replace('\\', "\\\\").replace('"', "\\\""), &enc_fields);
        let enc_last = if enc_fields.contains(&last_operand.to_string()) {
            format!("{last_operand}_enc")
        } else {
            last_operand.clone()
        };

        match (has_aq, has_rl) {
            (true, true) => format!(
                r#"{{
{bindings}    let suffix = if *aq && *rl {{
        ".aqrl"
    }} else if *aq {{
        ".aq"
    }} else if *rl {{
        ".rl"
    }} else {{
        ""
    }};
    format!("{name}{{suffix}}{escaped_prefix}0({{{enc_last}}})")
}}"#
            ),
            (true, false) => format!(
                r#"{{
{bindings}    let suffix = if *aq {{
        ".aq"
    }} else {{
        ""
    }};
    format!("{name}{{suffix}}{escaped_prefix}0({{{enc_last}}})")
}}"#
            ),
            (false, true) => format!(
                r#"{{
{bindings}    let suffix = if *rl {{
        ".rl"
    }} else {{
        ""
    }};
    format!("{name}{{suffix}}{escaped_prefix}0({{{enc_last}}})")
}}"#
            ),
            (false, false) => {
                format!(r#"{{{bindings}    format!("{name}{escaped_prefix}0({{{enc_last}}})")}}"#)
            }
        }
    } else {
        let operands_part = if operands_segment.trim().is_empty() {
            String::new()
        } else {
            format!(" {}", operands_segment)
        };

        let format_template = format!("{name}{{suffix}}{operands_part}");
        let enc_fields = asm_encode_fields_from_format(&format_template);
        let bindings = enc_let_bindings(&enc_fields);
        let escaped_operands = replace_with_enc_suffix(
            &operands_part.replace('\\', "\\\\").replace('"', "\\\""),
            &enc_fields,
        );

        match (has_aq, has_rl) {
            (true, true) => format!(
                r#"{{
{bindings}    let suffix = if *aq && *rl {{
        ".aqrl"
    }} else if *aq {{
        ".aq"
    }} else if *rl {{
        ".rl"
    }} else {{
        ""
    }};
    format!("{name}{{suffix}}{escaped_operands}")
}}"#
            ),
            (true, false) => format!(
                r#"{{
{bindings}    let suffix = if *aq {{
        ".aq"
    }} else {{
        ""
    }};
    format!("{name}{{suffix}}{escaped_operands}")
}}"#
            ),
            (false, true) => format!(
                r#"{{
{bindings}    let suffix = if *rl {{
        ".rl"
    }} else {{
        ""
    }};
    format!("{name}{{suffix}}{escaped_operands}")
}}"#
            ),
            (false, false) => {
                format!(r#"{{{bindings}    format!("{name}{escaped_operands}")}}"#)
            }
        }
    }
}

fn is_mop_instruction(instruction: &Instruction) -> bool {
    (instruction.name.starts_with("mop.") || instruction.name.starts_with("c.mop"))
        && instruction
            .operands
            .iter()
            .any(|operand| operand.name == "n")
}

/// `mop.*` and `c.mop.*` encode the printed mnemonic suffix from operand `n`.
fn generate_mop_assembly_code(name: &str, operands_segment: &str) -> String {
    let operands_part = if operands_segment.trim().is_empty() {
        String::new()
    } else {
        format!(" {}", operands_segment)
    };

    // `n` is a 1-bit bool-like field used numerically; other fields need encode_asm.
    let format_template = format!("{name}.{{}}{operands_part}");
    let enc_fields = asm_encode_fields_from_format(&format_template);
    let bindings = enc_let_bindings(&enc_fields);
    let escaped_operands = replace_with_enc_suffix(
        &operands_part.replace('\\', "\\\\").replace('"', "\\\""),
        &enc_fields,
    );

    let name_prefix = if name.ends_with(".n") {
        &name[..name.len() - 2]
    } else {
        name
    };

    format!(r#"{{{bindings}    format!("{name_prefix}.{{}}{escaped_operands}", n)}}"#)
}

fn is_vector_instruction_with_vm(instruction: &Instruction) -> bool {
    instruction.name.starts_with('v')
        && instruction
            .operands
            .iter()
            .any(|operand| operand.name == "vm")
}

/// Vector instructions with `vm` print `, v0.t` only when masking is enabled.
fn generate_vector_vm_assembly_code(name: &str, operands_segment: &str) -> String {
    let operands_without_vm = operands_segment
        .replace(", {vm}", "")
        .replace("{vm}, ", "")
        .replace("{vm}", "");

    let operands_part = if operands_without_vm.trim().is_empty() {
        String::new()
    } else {
        format!(" {}", operands_without_vm)
    };

    let format_template = format!("{name}{operands_part}{{vm_suffix}}");
    let enc_fields = asm_encode_fields_from_format(&format_template);
    let bindings = enc_let_bindings(&enc_fields);
    let escaped_operands = replace_with_enc_suffix(
        &operands_part.replace('\\', "\\\\").replace('"', "\\\""),
        &enc_fields,
    );

    format!(
        r#"{{
{bindings}    let vm_suffix = if *vm {{
        ""
    }} else {{
        ", v0.t"
    }};
    format!("{name}{escaped_operands}{{vm_suffix}}")
}}"#
    )
}

/// Reconstruct compressed offsets whose textual form is not directly stored in
/// the repaired syntax placeholder string.
fn generate_compressed_offset_assembly_code(name: &str, operands_segment: &str) -> String {
    let operands_without_braces = operands_segment.replace('{', "").replace('}', "");

    let operands: Vec<&str> = operands_without_braces
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    match name {
        "c.sh" => {
            if operands.len() >= 3 {
                let xs2 = operands[0];
                let xs1 = operands[2];
                let xs2_enc = format!("{xs2}_enc");
                let xs1_enc = format!("{xs1}_enc");
                format!(
                    r#"{{
    let {xs2_enc} = {xs2}.encode_asm();
    let {xs1_enc} = {xs1}.encode_asm();
    let offset = if *uimm {{
        "2"
    }} else {{
        "0"
    }};
    format!("{name} {{{xs2_enc}}}, {{offset}}({{{xs1_enc}}})")
}}"#
                )
            } else {
                r#"{
    let xs2_enc = xs2.encode_asm();
    let xs1_enc = xs1.encode_asm();
    format!("c.sh {xs2_enc}, 0({xs1_enc})")
}"#
                .to_string()
            }
        }
        "c.lh" | "c.lhu" => {
            if operands.len() >= 3 {
                let xd = operands[0];
                let xs1 = operands[2];
                let xd_enc = format!("{xd}_enc");
                let xs1_enc = format!("{xs1}_enc");
                format!(
                    r#"{{
    let {xd_enc} = {xd}.encode_asm();
    let {xs1_enc} = {xs1}.encode_asm();
    let offset = if *uimm {{
        "2"
    }} else {{
        "0"
    }};
    format!("{name} {{{xd_enc}}}, {{offset}}({{{xs1_enc}}})")
}}"#
                )
            } else {
                let suffix = &name[2..]; // "lh" or "lhu"
                format!(
                    r#"{{
    let xd_enc = xd.encode_asm();
    let xs1_enc = xs1.encode_asm();
    format!("c.{suffix} {{xd_enc}}, 0({{xs1_enc}})")
}}"#
                )
            }
        }
        _ => unreachable!(),
    }
}

/// Materialize the printable `cm.*` syntax from the merged synthetic operand.
fn generate_cm_instruction_assembly_code(name: &str) -> String {
    match name {
        "cm.pop" => r#"{
    let reg_list_str = saved_reg_list_with_stack_adj.get_saved_reg_list_string();
    let stack_adj = saved_reg_list_with_stack_adj.get_stack_adjustment();
    format!("cm.pop {}, {}", reg_list_str, stack_adj)
}"#
        .to_string(),
        "cm.push" => r#"{
    let reg_list_str = saved_reg_list_with_stack_adj.get_saved_reg_list_string();
    let stack_adj = saved_reg_list_with_stack_adj.get_stack_adjustment();
    format!("cm.push {}, -{}", reg_list_str, stack_adj)
}"#
        .to_string(),
        "cm.popretz" => r#"{
    let reg_list_str = saved_reg_list_with_stack_adj.get_saved_reg_list_string();
    let stack_adj = saved_reg_list_with_stack_adj.get_stack_adjustment();
    format!("cm.popretz {}, {}", reg_list_str, stack_adj)
}"#
        .to_string(),
        "cm.popret" => r#"{
    let reg_list_str = saved_reg_list_with_stack_adj.get_saved_reg_list_string();
    let stack_adj = saved_reg_list_with_stack_adj.get_stack_adjustment();
    format!("cm.popret {}, {}", reg_list_str, stack_adj)
}"#
        .to_string(),
        _ => unreachable!(),
    }
}
