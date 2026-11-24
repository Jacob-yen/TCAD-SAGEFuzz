use crate::AsmParseError;
use crate::validated::sign_extend;

/// Encode `self` to canonical RISC-V assembly text.
///
/// This is the primary assembly-layer serialization trait. Every operand type
/// and every instruction type implements this. `impl Display` delegates here so
/// `.to_string()` continues to work on all types.
pub trait AsmEncode {
    fn encode_asm(&self) -> String;
}

/// Decode a RISC-V assembly text fragment back into a typed value.
///
/// This is the primary assembly-layer deserialization trait. It replaces the
/// former `ParseAsmOperand` trait. `impl FromStr` delegates here so `.parse()`
/// continues to work on all types.
pub trait AsmDecode: Sized {
    fn decode_asm(input: &str) -> Result<Self, AsmParseError>;
}

/// RISC-V ISA base context for assembly decoding.
///
/// Top-level instruction enums are ISA-base-aware and use this context to
/// disambiguate between RV32 and RV64 variants when decoding assembly text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaDecodeContext {
    /// 32-bit base ISA
    RV32,
    /// 64-bit base ISA
    RV64,
}

/// Decode a RISC-V assembly text fragment with an ISA context.
///
/// Used by top-level instruction enums (`RiscvInstruction`) that contain both
/// RV32 and RV64 variants and need runtime ISA context to disambiguate.
/// Operand types and single-ISA instruction enums use [`AsmDecode`] instead.
pub trait AsmDecodeWithContext: Sized {
    fn decode_asm_with_context(
        input: &str,
        ctx: IsaDecodeContext,
    ) -> Result<Self, AsmParseError>;
}

/// Low-level parsing of a bare numeric literal (decimal, hex `0x…`, binary
/// `0b…`, octal `0o…`) with an optional leading sign.
///
/// Used internally by macro-generated `AsmDecode` impls for `ValidatedValue`
/// newtypes. Not part of the public API surface.
#[doc(hidden)]
pub trait AsmScalar: Sized {
    fn parse_asm_scalar(input: &str) -> Result<Self, AsmParseError>;
}

// ── internal numeric helpers ──────────────────────────────────────────────

fn parse_prefixed_u64(input: &str) -> Result<(bool, u64), String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err("literal is empty".to_string());
    }

    let (negative, rest) = if let Some(rest) = trimmed.strip_prefix('-') {
        (true, rest)
    } else if let Some(rest) = trimmed.strip_prefix('+') {
        (false, rest)
    } else {
        (false, trimmed)
    };

    if rest.is_empty() {
        return Err("literal is missing digits".to_string());
    }

    let (radix, digits) =
        if let Some(hex) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
            (16, hex)
        } else if let Some(bin) = rest.strip_prefix("0b").or_else(|| rest.strip_prefix("0B")) {
            (2, bin)
        } else if let Some(oct) = rest.strip_prefix("0o").or_else(|| rest.strip_prefix("0O")) {
            (8, oct)
        } else {
            (10, rest)
        };

    if digits.is_empty() {
        return Err("literal is missing digits after radix prefix".to_string());
    }

    let value = u64::from_str_radix(digits, radix)
        .map_err(|err| format!("invalid numeric literal `{trimmed}`: {err}"))?;
    Ok((negative, value))
}

fn parse_signed_i64_literal(input: &str) -> Result<i64, String> {
    let trimmed = input.trim();
    let (negative, magnitude) = parse_prefixed_u64(trimmed)?;
    if negative {
        if magnitude == (i64::MAX as u64) + 1 {
            return Ok(i64::MIN);
        }
        let signed = i64::try_from(magnitude)
            .map_err(|_| format!("signed literal `{trimmed}` is out of range for i64"))?;
        Ok(-signed)
    } else {
        i64::try_from(magnitude)
            .map_err(|_| format!("signed literal `{trimmed}` is out of range for i64"))
    }
}

fn parse_unsigned_u64_literal(input: &str) -> Result<u64, String> {
    let trimmed = input.trim();
    let (negative, magnitude) = parse_prefixed_u64(trimmed)?;
    if negative {
        return Err(format!("unsigned literal `{trimmed}` must not be negative"));
    }
    Ok(magnitude)
}

// These are kept pub so macro-generated code in consumer crates can call them.
// They are not part of the documented public API.
#[doc(hidden)]
pub fn parse_asm_i64_literal(input: &str) -> Result<i64, AsmParseError> {
    parse_signed_i64_literal(input).map_err(|err| AsmParseError::invalid_operand("i64", err))
}

#[doc(hidden)]
pub fn parse_asm_u64_literal(input: &str) -> Result<u64, AsmParseError> {
    parse_unsigned_u64_literal(input).map_err(|err| AsmParseError::invalid_operand("u64", err))
}

#[doc(hidden)]
pub fn parse_asm_sign_extended_literal(
    input: &str,
    rendered_bits: u8,
) -> Result<i64, AsmParseError> {
    let raw = parse_asm_u64_literal(input)?;
    Ok(sign_extend(raw, rendered_bits))
}

#[doc(hidden)]
pub fn asm_first_token(input: &str) -> Option<&str> {
    input.split_ascii_whitespace().next()
}

/// Match `input` against a template pattern containing `{placeholder}` slots
/// and return the captured text for each slot.
///
/// Used by macro-generated `AsmDecode` impls for instruction structs.
#[doc(hidden)]
pub fn parse_asm_pattern(
    ty: &'static str,
    pattern: &str,
    input: &str,
) -> Result<std::collections::BTreeMap<String, String>, AsmParseError> {
    let pattern = pattern.trim();
    let input = input.trim();

    if pattern.is_empty() {
        if input.is_empty() {
            return Ok(std::collections::BTreeMap::new());
        }
        return Err(AsmParseError::syntax_mismatch(
            ty,
            format!("expected empty assembly, got `{input}`"),
        ));
    }

    if input.is_empty() {
        return Err(AsmParseError::empty_input(ty));
    }

    let mut captures = std::collections::BTreeMap::new();
    let mut pattern_pos = 0usize;
    let mut input_pos = 0usize;

    while let Some(open_offset) = pattern[pattern_pos..].find('{') {
        let open = pattern_pos + open_offset;
        let close = pattern[open..].find('}').ok_or_else(|| {
            AsmParseError::syntax_mismatch(ty, format!("malformed asm pattern `{pattern}`"))
        })? + open;

        let literal = &pattern[pattern_pos..open];
        if !input[input_pos..].starts_with(literal) {
            return Err(AsmParseError::syntax_mismatch(
                ty,
                format!("expected literal segment `{literal}` while matching `{input}`"),
            ));
        }
        input_pos += literal.len();

        let placeholder = &pattern[open + 1..close];
        pattern_pos = close + 1;

        let next_open = pattern[pattern_pos..]
            .find('{')
            .map(|offset| pattern_pos + offset)
            .unwrap_or(pattern.len());
        let next_literal = &pattern[pattern_pos..next_open];

        let capture_end = if next_literal.is_empty() {
            input.len()
        } else {
            input[input_pos..]
                .find(next_literal)
                .map(|offset| input_pos + offset)
                .ok_or_else(|| {
                    AsmParseError::syntax_mismatch(
                        ty,
                        format!(
                            "expected trailing literal `{next_literal}` while matching `{input}`"
                        ),
                    )
                })?
        };

        let captured = input[input_pos..capture_end].trim();
        if captured.is_empty() {
            return Err(AsmParseError::syntax_mismatch(
                ty,
                format!("missing value for operand `{placeholder}` in `{input}`"),
            ));
        }

        captures.insert(placeholder.to_string(), captured.to_string());
        input_pos = capture_end;
    }

    let trailing = &pattern[pattern_pos..];
    if !input[input_pos..].starts_with(trailing) {
        return Err(AsmParseError::syntax_mismatch(
            ty,
            format!("expected trailing literal `{trailing}` while matching `{input}`"),
        ));
    }
    input_pos += trailing.len();

    if input[input_pos..].trim().is_empty() {
        Ok(captures)
    } else {
        Err(AsmParseError::syntax_mismatch(
            ty,
            format!("unexpected trailing text `{}`", input[input_pos..].trim()),
        ))
    }
}

// ── AsmScalar impls for primitive numerics ────────────────────────────────

macro_rules! impl_signed_asm_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl AsmScalar for $ty {
                fn parse_asm_scalar(input: &str) -> Result<Self, AsmParseError> {
                    let value = parse_signed_i64_literal(input)
                        .map_err(|err| AsmParseError::invalid_operand(stringify!($ty), err))?;
                    <$ty>::try_from(value).map_err(|_| {
                        AsmParseError::invalid_operand(
                            stringify!($ty),
                            format!("literal `{}` is out of range for {}", input.trim(), stringify!($ty)),
                        )
                    })
                }
            }
        )*
    };
}

macro_rules! impl_unsigned_asm_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl AsmScalar for $ty {
                fn parse_asm_scalar(input: &str) -> Result<Self, AsmParseError> {
                    let value = parse_unsigned_u64_literal(input)
                        .map_err(|err| AsmParseError::invalid_operand(stringify!($ty), err))?;
                    <$ty>::try_from(value).map_err(|_| {
                        AsmParseError::invalid_operand(
                            stringify!($ty),
                            format!("literal `{}` is out of range for {}", input.trim(), stringify!($ty)),
                        )
                    })
                }
            }
        )*
    };
}

impl_signed_asm_scalar!(i8, i16, i32, i64);
impl_unsigned_asm_scalar!(u8, u16, u32, u64);

// ── AsmDecode for primitives ──────────────────────────────────────────────

impl AsmDecode for bool {
    fn decode_asm(input: &str) -> Result<Self, AsmParseError> {
        match input.trim() {
            "0" | "false" => Ok(false),
            "1" | "true" => Ok(true),
            other => Err(AsmParseError::invalid_operand(
                "bool",
                format!("expected one of `0`, `1`, `false`, `true`, got `{other}`"),
            )),
        }
    }
}

impl AsmDecode for String {
    fn decode_asm(input: &str) -> Result<Self, AsmParseError> {
        Ok(input.trim().to_string())
    }
}
