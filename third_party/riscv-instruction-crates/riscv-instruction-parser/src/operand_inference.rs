//! Shared fallback inference for operand metadata.
//!
//! Parser-side assembly inference and fixer-side `Unknown -> kind` repair used
//! to maintain two mostly-overlapping name heuristics. Keep the fallback logic
//! in one place so later refactors only need to change one module.

use riscv_instruction_schema::{
    CsrSpec, ImmediateKind, ImmediateSpec, OperandKind, RegisterBank, RegisterPairSpec,
    RegisterSpec,
};

const REGISTER_PREFIXES: &[&str] = &[
    "rd", "rs", "rt", "fd", "fs", "ft", "vd", "vs", "vt", "xd", "xs", "hd", "hs", "qd", "qs",
];

const SYNTHESIZABLE_OPERANDS: &[&str] = &[
    "aq",
    "rl",
    "rm",
    "vm",
    "pred",
    "succ",
    "nf",
    "csr",
    "reg_list",
    "stack_adj",
    "rlist",
    "spimm",
    "ordering",
];

pub(crate) fn is_synthesizable_operand_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    if is_concrete_register_literal(name) {
        return false;
    }

    name.contains("imm")
        || SYNTHESIZABLE_OPERANDS.contains(&name)
        || REGISTER_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

pub(crate) fn infer_operand_bit_length(name: &str) -> u8 {
    if REGISTER_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(prefix))
    {
        return 5;
    }

    match name {
        "aq" | "rl" | "vm" => 1,
        "rm" => 3,
        _ if name.contains("imm") => 12,
        _ => 0,
    }
}

pub(crate) fn infer_operand_kind(name: &str) -> OperandKind {
    match name {
        "xd" | "xs1" | "xs2" | "rd" | "rs1" | "rs2" | "rs3" => {
            OperandKind::Register(RegisterSpec {
                bank: RegisterBank::Integer,
            })
        }
        "fd" | "fs1" | "fs2" | "fs3" | "qd" | "qs1" | "qs2" | "qs3" | "hd" | "dd" => {
            OperandKind::Register(RegisterSpec {
                bank: RegisterBank::FloatingPoint,
            })
        }
        "vd" | "vs1" | "vs2" | "vs3" => OperandKind::Register(RegisterSpec {
            bank: RegisterBank::Vector,
        }),
        "csr" => OperandKind::Csr(CsrSpec),
        "rm" | "frm" => OperandKind::RoundMode,
        "shamt" => OperandKind::Immediate(ImmediateSpec {
            kind: ImmediateKind::ShiftAmount,
        }),
        "uimm" | "n" | "rnum" | "fm" | "vtypei" | "vm" | "aq" | "rl" | "bs" | "nf" => {
            OperandKind::Immediate(ImmediateSpec {
                kind: ImmediateKind::Unsigned,
            })
        }
        "imm" | "imm12" | "offset" | "spimm" => OperandKind::Immediate(ImmediateSpec {
            kind: ImmediateKind::Signed,
        }),
        "pred" | "succ" | "ordering" => OperandKind::FenceMode,
        "r1s" | "r2s" => OperandKind::Register(RegisterSpec {
            bank: RegisterBank::Integer,
        }),
        "ne_r1s_r2s" => OperandKind::RegisterPair(RegisterPairSpec {
            bank: RegisterBank::Integer,
            distinct: true,
        }),
        _ => {
            if name.starts_with('x') {
                OperandKind::Register(RegisterSpec {
                    bank: RegisterBank::Integer,
                })
            } else if name.starts_with('f') || name.starts_with('q') || name.starts_with('h') {
                OperandKind::Register(RegisterSpec {
                    bank: RegisterBank::FloatingPoint,
                })
            } else if name.starts_with('v') {
                OperandKind::Register(RegisterSpec {
                    bank: RegisterBank::Vector,
                })
            } else if name.contains("imm") || name.contains("offset") {
                OperandKind::Immediate(ImmediateSpec {
                    kind: ImmediateKind::Signed,
                })
            } else {
                OperandKind::Immediate(ImmediateSpec {
                    kind: ImmediateKind::Unsigned,
                })
            }
        }
    }
}

fn is_concrete_register_literal(name: &str) -> bool {
    name.len() > 1
        && name[1..].chars().all(|c| c.is_ascii_digit())
        && matches!(name.chars().next(), Some('x' | 'f' | 'v'))
}

#[cfg(test)]
mod tests {
    use super::{infer_operand_bit_length, infer_operand_kind, is_synthesizable_operand_name};
    use riscv_instruction_schema::{ImmediateKind, OperandKind, RegisterBank, RegisterSpec};

    #[test]
    fn synthesizable_names_skip_concrete_register_literals() {
        assert!(!is_synthesizable_operand_name("x1"));
        assert!(!is_synthesizable_operand_name("f12"));
        assert!(!is_synthesizable_operand_name("v3"));
        assert!(is_synthesizable_operand_name("rs1"));
        assert!(is_synthesizable_operand_name("uimm"));
    }

    #[test]
    fn bit_length_inference_matches_existing_fallbacks() {
        assert_eq!(infer_operand_bit_length("rs1"), 5);
        assert_eq!(infer_operand_bit_length("rm"), 3);
        assert_eq!(infer_operand_bit_length("uimm"), 12);
        assert_eq!(infer_operand_bit_length("aq"), 1);
    }

    #[test]
    fn kind_inference_covers_register_and_immediate_families() {
        assert!(matches!(
            infer_operand_kind("rs1"),
            OperandKind::Register(RegisterSpec {
                bank: RegisterBank::Integer
            })
        ));
        assert!(matches!(
            infer_operand_kind("fs2"),
            OperandKind::Register(RegisterSpec {
                bank: RegisterBank::FloatingPoint
            })
        ));
        assert!(matches!(
            infer_operand_kind("uimm"),
            OperandKind::Immediate(spec) if spec.kind == ImmediateKind::Unsigned
        ));
        assert!(matches!(
            infer_operand_kind("offset"),
            OperandKind::Immediate(spec) if spec.kind == ImmediateKind::Signed
        ));
    }
}
