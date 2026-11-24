use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum XLenImmediate {
    X32(i32),
    X64(i64),
}

impl XLenImmediate {
    pub const fn xlen(self) -> XLen {
        match self {
            Self::X32(_) => XLen::X32,
            Self::X64(_) => XLen::X64,
        }
    }

    pub const fn as_i64(self) -> i64 {
        match self {
            Self::X32(value) => value as i64,
            Self::X64(value) => value,
        }
    }

    pub fn from_i64_for(profile: IsaProfile, value: i64) -> Result<Self, RandomGenerationError> {
        match profile.xlen {
            XLen::X32 => i32::try_from(value)
                .map(Self::X32)
                .map_err(|_| RandomGenerationError::new("XLenImmediate", format!("value {value} exceeds RV32 signed range"))),
            XLen::X64 => Ok(Self::X64(value)),
        }
    }

    pub fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        config: &RandomConfig,
    ) -> Result<Self, RandomGenerationError> {
        match config.isa_profile.xlen {
            XLen::X32 => Ok(Self::X32(rng.random::<i32>())),
            XLen::X64 => Ok(Self::X64(rng.random::<i64>())),
        }
    }

    pub const fn from_unsigned_bits_for(profile: IsaProfile, value: u64) -> Self {
        match profile.xlen {
            XLen::X32 => Self::X32(value as u32 as i32),
            XLen::X64 => Self::X64(value as i64),
        }
    }

    pub const fn from_address(address: XLenAddress) -> Self {
        match address {
            XLenAddress::X32(value) => Self::X32(value as i32),
            XLenAddress::X64(value) => Self::X64(value as i64),
        }
    }

    pub fn decode_with_context(input: &str, ctx: IsaDecodeContext) -> Result<Self, AsmParseError> {
        let value = parse_asm_i64_literal(input)?;
        match ctx {
            IsaDecodeContext::RV32 => i32::try_from(value)
                .map(Self::X32)
                .map_err(|_| AsmParseError::invalid_operand("XLenImmediate", format!("literal `{input}` exceeds RV32 signed range"))),
            IsaDecodeContext::RV64 => Ok(Self::X64(value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum XLenAddress {
    X32(u32),
    X64(u64),
}

impl XLenAddress {
    pub const fn xlen(self) -> XLen {
        match self {
            Self::X32(_) => XLen::X32,
            Self::X64(_) => XLen::X64,
        }
    }

    pub const fn as_u64(self) -> u64 {
        match self {
            Self::X32(value) => value as u64,
            Self::X64(value) => value,
        }
    }

    pub fn from_u64_for(profile: IsaProfile, value: u64) -> Result<Self, RandomGenerationError> {
        match profile.xlen {
            XLen::X32 => u32::try_from(value)
                .map(Self::X32)
                .map_err(|_| RandomGenerationError::new("XLenAddress", format!("address {value:#x} exceeds RV32 address range"))),
            XLen::X64 => Ok(Self::X64(value)),
        }
    }

    pub fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        config: &RandomConfig,
    ) -> Result<Self, RandomGenerationError> {
        let (min, max) = config.memory_access_config.memory_base_address_range;
        if min > max {
            return Err(RandomGenerationError::new(
                "XLenAddress",
                format!("invalid configured address range {min:#x}..={max:#x}"),
            ));
        }
        let value = if min == max { min } else { rng.random_range(min..=max) };
        Self::from_u64_for(config.isa_profile, value)
    }

    pub fn decode_with_context(input: &str, ctx: IsaDecodeContext) -> Result<Self, AsmParseError> {
        let value = parse_asm_u64_literal(input)?;
        match ctx {
            IsaDecodeContext::RV32 => u32::try_from(value)
                .map(Self::X32)
                .map_err(|_| AsmParseError::invalid_operand("XLenAddress", format!("literal `{input}` exceeds RV32 address range"))),
            IsaDecodeContext::RV64 => Ok(Self::X64(value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PseudoRequirements {
    pub min_flen: Option<FLen>,
}

/// Purpose metadata for `li` pseudo instructions injected ahead of real instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadImmediatePurpose {
    /// Prepare a base address register used by a subsequent memory access.
    BaseAddress,
    /// Seed the source register value consumed by a CSR write instruction.
    CsrValue,
    /// Generic/unspecified `li` usage.
    Generic,
}

/// RISC-V pseudo instructions for common operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PseudoInstruction {
    /// Label definition - label:
    Label { name: String },
    /// Load address - la rd, symbol (expands to auipc + addi)
    LoadAddress {
        rd: IntegerRegister,
        address: XLenAddress,
    },
    /// Load immediate - li rd, imm (expands to lui + addi or just addi)
    LoadImmediate {
        rd: IntegerRegister,
        immediate: XLenImmediate,
        purpose: LoadImmediatePurpose,
    },
    /// Load address of label - lla rd, label (expands to auipc + addi)
    LoadLabelAddress { rd: IntegerRegister, label: String },
    /// Move register - mv rd, rs (expands to addi rd, rs, 0)
    Move {
        rd: IntegerRegister,
        rs: IntegerRegister,
    },
    /// No-op - nop (expands to addi x0, x0, 0)
    Nop,
    /// Jump - j label (expands to jal x0, label)
    Jump { label: String },
    /// Jump register - jr rs (expands to jalr x0, 0(rs))
    JumpRegister { rs: IntegerRegister },
    /// Return - ret (expands to jalr x0, 0(ra))
    Return,
    /// Call - call symbol
    Call { symbol: String },
    /// Tail call - tail symbol
    Tail { symbol: String },
    /// Read fflags CSR into a general-purpose register
    ReadFflags { rd: IntegerRegister },
    /// Read the read-only RVV vl CSR into a general-purpose register.
    ReadVl { rd: IntegerRegister },
}

impl Random for PseudoInstruction {
    type Output = Self;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        let variant = rng.random_range(0..6);
        match variant {
            0 => Ok(PseudoInstruction::LoadAddress {
                rd: IntegerRegister::random_with_rng(rng, config)?,
                address: XLenAddress::random_with_rng(rng, config)?,
            }),
            1 => Ok(PseudoInstruction::LoadImmediate {
                rd: IntegerRegister::random_with_rng(rng, config)?,
                immediate: XLenImmediate::random_with_rng(rng, config)?,
                purpose: LoadImmediatePurpose::Generic,
            }),
            2 => Ok(PseudoInstruction::LoadLabelAddress {
                rd: IntegerRegister::random_with_rng(rng, config)?,
                label: String::random_with_rng(rng, config)?,
            }),
            3 => Ok(PseudoInstruction::Move {
                rd: IntegerRegister::random_with_rng(rng, config)?,
                rs: IntegerRegister::random_with_rng(rng, config)?,
            }),
            4 => Ok(PseudoInstruction::Jump {
                label: String::random_with_rng(rng, config)?,
            }),
            _ => Ok(PseudoInstruction::Call {
                symbol: String::random_with_rng(rng, config)?,
            }),
        }
    }
}

impl AsmEncode for PseudoInstruction {
    fn encode_asm(&self) -> String {
        match self {
            PseudoInstruction::Label { name } => format!("{}:", name),
            PseudoInstruction::LoadAddress { rd, address } => {
                format!("la {}, 0x{:x}", rd.encode_asm(), address.as_u64())
            }
            PseudoInstruction::LoadImmediate { rd, immediate, .. } => {
                format!("li {}, {}", rd.encode_asm(), immediate.as_i64())
            }
            PseudoInstruction::LoadLabelAddress { rd, label } => {
                format!("lla {}, {}", rd.encode_asm(), label)
            }
            PseudoInstruction::Move { rd, rs } => {
                format!("mv {}, {}", rd.encode_asm(), rs.encode_asm())
            }
            PseudoInstruction::Nop => "nop".to_string(),
            PseudoInstruction::Jump { label } => format!("j {}", label),
            PseudoInstruction::JumpRegister { rs } => format!("jr {}", rs.encode_asm()),
            PseudoInstruction::Return => "ret".to_string(),
            PseudoInstruction::Call { symbol } => format!("call {}", symbol),
            PseudoInstruction::Tail { symbol } => format!("tail {}", symbol),
            PseudoInstruction::ReadFflags { rd } => format!("frflags {}", rd.encode_asm()),
            PseudoInstruction::ReadVl { rd } => format!("csrr {}, vl", rd.encode_asm()),
        }
    }
}

impl RuntimeOperandAccess for PseudoInstruction {
    fn register_operands(&self) -> Vec<RuntimeRegisterOperand> {
        match self {
            Self::LoadAddress { rd, .. }
            | Self::LoadImmediate { rd, .. }
            | Self::LoadLabelAddress { rd, .. }
            | Self::ReadFflags { rd }
            | Self::ReadVl { rd } => vec![RuntimeRegisterOperand {
                name: "rd",
                usage: RuntimeOperandUsage::Destination,
                role: RuntimeRegisterRole::Plain,
                register: runtime_register(*rd),
            }],
            Self::Move { rd, rs } => vec![
                RuntimeRegisterOperand {
                    name: "rd",
                    usage: RuntimeOperandUsage::Destination,
                    role: RuntimeRegisterRole::Plain,
                    register: runtime_register(*rd),
                },
                RuntimeRegisterOperand {
                    name: "rs",
                    usage: RuntimeOperandUsage::Source,
                    role: RuntimeRegisterRole::Plain,
                    register: runtime_register(*rs),
                },
            ],
            Self::JumpRegister { rs } => vec![RuntimeRegisterOperand {
                name: "rs",
                usage: RuntimeOperandUsage::Source,
                role: RuntimeRegisterRole::Plain,
                register: runtime_register(*rs),
            }],
            Self::Label { .. }
            | Self::Nop
            | Self::Jump { .. }
            | Self::Return
            | Self::Call { .. }
            | Self::Tail { .. } => Vec::new(),
        }
    }

    fn set_register_operand(
        &mut self,
        operand_name: &str,
        register: RuntimeRegisterAssignment,
    ) -> Result<bool, OperandAccessError> {
        match (self, operand_name) {
            (Self::LoadAddress { rd, .. }, "rd")
            | (Self::LoadImmediate { rd, .. }, "rd")
            | (Self::LoadLabelAddress { rd, .. }, "rd")
            | (Self::ReadFflags { rd }, "rd")
            | (Self::ReadVl { rd }, "rd") => {
                set_register_from_runtime_assignment(rd, operand_name, register)
            }
            (Self::Move { rd, .. }, "rd") => {
                set_register_from_runtime_assignment(rd, operand_name, register)
            }
            (Self::Move { rs, .. }, "rs") | (Self::JumpRegister { rs }, "rs") => {
                set_register_from_runtime_assignment(rs, operand_name, register)
            }
            _ => Err(OperandAccessError::missing_register_operand(operand_name)),
        }
    }
}

impl MemoryAccessInstruction for PseudoInstruction {
    fn offset_operand_value(&self) -> Option<i64> {
        None
    }
}

impl NumericOperandMutation for PseudoInstruction {
    fn mutate_numeric_operands<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        _config: &RandomConfig,
    ) -> bool {
        match self {
            Self::LoadAddress { address, .. } => {
                if let Ok(next) = XLenAddress::random_with_rng(rng, _config) {
                    *address = next;
                    true
                } else {
                    false
                }
            }
            Self::LoadImmediate { immediate, .. } => {
                if let Ok(next) = XLenImmediate::random_with_rng(rng, _config) {
                    *immediate = next;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

impl InstructionSequenceHints for PseudoInstruction {}

impl InstructionSequenceItem for PseudoInstruction {
    fn as_pseudo_instruction(&self) -> Option<&PseudoInstruction> {
        Some(self)
    }

    fn as_pseudo_instruction_mut(&mut self) -> Option<&mut PseudoInstruction> {
        Some(self)
    }
}

impl PseudoInstruction {
    pub fn requirements(&self) -> PseudoRequirements {
        match self {
            Self::ReadFflags { .. } => PseudoRequirements {
                min_flen: Some(FLen::F32),
            },
            _ => PseudoRequirements::default(),
        }
    }

    pub fn validate_for_profile(&self, profile: IsaProfile) -> Result<(), RandomGenerationError> {
        match self {
            Self::LoadAddress { address, .. } => {
                if address.xlen() != profile.xlen {
                    return Err(RandomGenerationError::new(
                        "PseudoInstruction",
                        format!("address payload requires {:?} but profile is {:?}", address.xlen(), profile.xlen),
                    ));
                }
            }
            Self::LoadImmediate { immediate, .. } => {
                if immediate.xlen() != profile.xlen {
                    return Err(RandomGenerationError::new(
                        "PseudoInstruction",
                        format!("immediate payload requires {:?} but profile is {:?}", immediate.xlen(), profile.xlen),
                    ));
                }
            }
            _ => {}
        }

        if let Some(min_flen) = self.requirements().min_flen {
            if !profile.supports_flen(min_flen) {
                return Err(RandomGenerationError::new(
                    "PseudoInstruction",
                    format!("pseudo instruction requires FLEN >= {}", min_flen.bits()),
                ));
            }
        }

        Ok(())
    }
}

fn parse_known_pseudo_instruction(
    trimmed: &str,
    ctx: IsaDecodeContext,
) -> Result<Option<PseudoInstruction>, AsmParseError> {
    let is_label = trimmed.ends_with(':')
        && !trimmed[..trimmed.len() - 1]
            .chars()
            .any(|ch| ch.is_ascii_whitespace());
    if is_label {
        return Ok(Some(PseudoInstruction::Label {
            name: trimmed[..trimmed.len() - 1].to_string(),
        }));
    }

    if let Ok(mut captures) = parse_asm_pattern("PseudoInstruction", "la {rd}, {address}", trimmed)
    {
        let rd =
            IntegerRegister::decode_asm(&captures.remove("rd").expect("capture exists"))?;
        let address = XLenAddress::decode_with_context(
            &captures.remove("address").expect("capture exists"),
            ctx,
        )?;
        return Ok(Some(PseudoInstruction::LoadAddress { rd, address }));
    }

    if let Ok(mut captures) =
        parse_asm_pattern("PseudoInstruction", "li {rd}, {immediate}", trimmed)
    {
        let rd =
            IntegerRegister::decode_asm(&captures.remove("rd").expect("capture exists"))?;
        let immediate = XLenImmediate::decode_with_context(
            &captures.remove("immediate").expect("capture exists"),
            ctx,
        )?;
        return Ok(Some(PseudoInstruction::LoadImmediate {
            rd,
            immediate,
            purpose: LoadImmediatePurpose::Generic,
        }));
    }

    if let Ok(mut captures) = parse_asm_pattern("PseudoInstruction", "lla {rd}, {label}", trimmed) {
        let rd =
            IntegerRegister::decode_asm(&captures.remove("rd").expect("capture exists"))?;
        let label = captures.remove("label").expect("capture exists");
        return Ok(Some(PseudoInstruction::LoadLabelAddress { rd, label }));
    }

    if let Ok(mut captures) = parse_asm_pattern("PseudoInstruction", "mv {rd}, {rs}", trimmed) {
        let rd =
            IntegerRegister::decode_asm(&captures.remove("rd").expect("capture exists"))?;
        let rs =
            IntegerRegister::decode_asm(&captures.remove("rs").expect("capture exists"))?;
        return Ok(Some(PseudoInstruction::Move { rd, rs }));
    }

    if trimmed == "nop" {
        return Ok(Some(PseudoInstruction::Nop));
    }

    if let Ok(mut captures) = parse_asm_pattern("PseudoInstruction", "j {label}", trimmed) {
        let label = captures.remove("label").expect("capture exists");
        return Ok(Some(PseudoInstruction::Jump { label }));
    }

    if let Ok(mut captures) = parse_asm_pattern("PseudoInstruction", "jr {rs}", trimmed) {
        let rs =
            IntegerRegister::decode_asm(&captures.remove("rs").expect("capture exists"))?;
        return Ok(Some(PseudoInstruction::JumpRegister { rs }));
    }

    if trimmed == "ret" {
        return Ok(Some(PseudoInstruction::Return));
    }

    if let Ok(mut captures) = parse_asm_pattern("PseudoInstruction", "call {symbol}", trimmed) {
        let symbol = captures.remove("symbol").expect("capture exists");
        return Ok(Some(PseudoInstruction::Call { symbol }));
    }

    if let Ok(mut captures) = parse_asm_pattern("PseudoInstruction", "tail {symbol}", trimmed) {
        let symbol = captures.remove("symbol").expect("capture exists");
        return Ok(Some(PseudoInstruction::Tail { symbol }));
    }

    if let Ok(mut captures) = parse_asm_pattern("PseudoInstruction", "frflags {rd}", trimmed) {
        let rd =
            IntegerRegister::decode_asm(&captures.remove("rd").expect("capture exists"))?;
        return Ok(Some(PseudoInstruction::ReadFflags { rd }));
    }

    if let Ok(mut captures) =
        parse_asm_pattern("PseudoInstruction", "csrrs {rd}, fflags, x0", trimmed)
    {
        let rd =
            IntegerRegister::decode_asm(&captures.remove("rd").expect("capture exists"))?;
        return Ok(Some(PseudoInstruction::ReadFflags { rd }));
    }

    for pattern in ["csrr {rd}, vl", "csrrs {rd}, vl, x0"] {
        if let Ok(mut captures) = parse_asm_pattern("PseudoInstruction", pattern, trimmed) {
            let rd = IntegerRegister::decode_asm(&captures.remove("rd").expect("capture exists"))?;
            return Ok(Some(PseudoInstruction::ReadVl { rd }));
        }
    }

    Ok(None)
}

impl AsmDecode for PseudoInstruction {
    fn decode_asm(_input: &str) -> Result<Self, AsmParseError> {
        Err(AsmParseError::invalid_instruction(
            "PseudoInstruction",
            "XLEN-sensitive pseudo instructions require decode_asm_with_context(...) instead of decode_asm()",
        ))
    }
}

impl AsmDecodeWithContext for PseudoInstruction {
    fn decode_asm_with_context(input: &str, ctx: IsaDecodeContext) -> Result<Self, AsmParseError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(AsmParseError::empty_input("PseudoInstruction"));
        }

        if let Some(pseudo) = parse_known_pseudo_instruction(trimmed, ctx)? {
            return Ok(pseudo);
        }

        Err(AsmParseError::invalid_instruction(
            "PseudoInstruction",
            format!("unsupported pseudo instruction `{trimmed}`"),
        ))
    }
}

impl PseudoInstruction {
    pub fn validate_asm_decode_context(&self, ctx: IsaDecodeContext) -> Result<(), RandomGenerationError> {
        let expected_xlen = match ctx {
            IsaDecodeContext::RV32 => XLen::X32,
            IsaDecodeContext::RV64 => XLen::X64,
        };

        match self {
            Self::LoadAddress { address, .. } if address.xlen() != expected_xlen => Err(
                RandomGenerationError::new(
                    "PseudoInstruction",
                    format!("address payload requires {:?} but decode context is {:?}", address.xlen(), expected_xlen),
                ),
            ),
            Self::LoadImmediate { immediate, .. } if immediate.xlen() != expected_xlen => Err(
                RandomGenerationError::new(
                    "PseudoInstruction",
                    format!("immediate payload requires {:?} but decode context is {:?}", immediate.xlen(), expected_xlen),
                ),
            ),
            _ => Ok(()),
        }
    }
}

impl std::fmt::Display for PseudoInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.encode_asm())
    }
}

impl std::str::FromStr for PseudoInstruction {
    type Err = AsmParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::decode_asm(s)
    }
}
