use riscv_instruction::RegisterBank;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParsedRegisterName {
    pub bank: RegisterBank,
    pub index: u8,
}

impl ParsedRegisterName {
    pub fn canonical_name(self) -> String {
        match self.bank {
            RegisterBank::Integer => format!("x{}", self.index),
            RegisterBank::FloatingPoint => format!("f{}", self.index),
            RegisterBank::Vector => format!("v{}", self.index),
        }
    }
}

pub fn parse_register_name(name: &str) -> Option<ParsedRegisterName> {
    let trimmed = name.trim().trim_end_matches(',').trim();
    if trimmed.is_empty() {
        return None;
    }
    let lowered = trimmed.to_ascii_lowercase();

    parse_integer_register_name(&lowered)
        .map(|index| ParsedRegisterName {
            bank: RegisterBank::Integer,
            index,
        })
        .or_else(|| {
            parse_float_register_name(&lowered).map(|index| ParsedRegisterName {
                bank: RegisterBank::FloatingPoint,
                index,
            })
        })
        .or_else(|| {
            parse_vector_register_name(&lowered).map(|index| ParsedRegisterName {
                bank: RegisterBank::Vector,
                index,
            })
        })
}

pub fn canonicalize_register_name(name: &str) -> Option<String> {
    parse_register_name(name).map(ParsedRegisterName::canonical_name)
}

fn parse_integer_register_name(name: &str) -> Option<u8> {
    if let Some(rest) = name.strip_prefix('x') {
        return rest.parse::<u8>().ok().filter(|idx| *idx <= 31);
    }

    match name {
        "zero" => Some(0),
        "ra" => Some(1),
        "sp" => Some(2),
        "gp" => Some(3),
        "tp" => Some(4),
        "t0" => Some(5),
        "t1" => Some(6),
        "t2" => Some(7),
        "s0" | "fp" => Some(8),
        "s1" => Some(9),
        "a0" => Some(10),
        "a1" => Some(11),
        "a2" => Some(12),
        "a3" => Some(13),
        "a4" => Some(14),
        "a5" => Some(15),
        "a6" => Some(16),
        "a7" => Some(17),
        "s2" => Some(18),
        "s3" => Some(19),
        "s4" => Some(20),
        "s5" => Some(21),
        "s6" => Some(22),
        "s7" => Some(23),
        "s8" => Some(24),
        "s9" => Some(25),
        "s10" => Some(26),
        "s11" => Some(27),
        "t3" => Some(28),
        "t4" => Some(29),
        "t5" => Some(30),
        "t6" => Some(31),
        _ => None,
    }
}

fn parse_float_register_name(name: &str) -> Option<u8> {
    if let Some(rest) = name.strip_prefix('f')
        && let Ok(index) = rest.parse::<u8>()
        && index <= 31
    {
        return Some(index);
    }

    match name {
        "ft0" => Some(0),
        "ft1" => Some(1),
        "ft2" => Some(2),
        "ft3" => Some(3),
        "ft4" => Some(4),
        "ft5" => Some(5),
        "ft6" => Some(6),
        "ft7" => Some(7),
        "fs0" => Some(8),
        "fs1" => Some(9),
        "fa0" => Some(10),
        "fa1" => Some(11),
        "fa2" => Some(12),
        "fa3" => Some(13),
        "fa4" => Some(14),
        "fa5" => Some(15),
        "fa6" => Some(16),
        "fa7" => Some(17),
        "fs2" => Some(18),
        "fs3" => Some(19),
        "fs4" => Some(20),
        "fs5" => Some(21),
        "fs6" => Some(22),
        "fs7" => Some(23),
        "fs8" => Some(24),
        "fs9" => Some(25),
        "fs10" => Some(26),
        "fs11" => Some(27),
        "ft8" => Some(28),
        "ft9" => Some(29),
        "ft10" => Some(30),
        "ft11" => Some(31),
        _ => None,
    }
}

fn parse_vector_register_name(name: &str) -> Option<u8> {
    let base = name.split('.').next().unwrap_or(name);
    if let Some(rest) = base.strip_prefix('v') {
        return rest.parse::<u8>().ok().filter(|idx| *idx <= 31);
    }
    None
}

#[cfg(test)]
mod tests {
    use riscv_instruction::RegisterBank;

    use super::{ParsedRegisterName, canonicalize_register_name, parse_register_name};

    #[test]
    fn parse_integer_alias_names() {
        assert_eq!(
            parse_register_name("a5"),
            Some(ParsedRegisterName {
                bank: RegisterBank::Integer,
                index: 15,
            })
        );
        assert_eq!(canonicalize_register_name("fp"), Some("x8".to_string()));
    }

    #[test]
    fn parse_float_alias_names() {
        assert_eq!(
            parse_register_name("fa6"),
            Some(ParsedRegisterName {
                bank: RegisterBank::FloatingPoint,
                index: 16,
            })
        );
        assert_eq!(canonicalize_register_name("fs2"), Some("f18".to_string()));
    }

    #[test]
    fn parse_vector_names() {
        assert_eq!(
            parse_register_name("v3.t"),
            Some(ParsedRegisterName {
                bank: RegisterBank::Vector,
                index: 3,
            })
        );
        assert_eq!(canonicalize_register_name("v31"), Some("v31".to_string()));
    }
}
