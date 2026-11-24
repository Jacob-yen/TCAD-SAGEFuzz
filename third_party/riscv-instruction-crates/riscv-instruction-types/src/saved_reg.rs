use std::fmt;

use rand::seq::IndexedRandom as _;
use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SavedRegListError {
    InvalidRegisterRange,
    InvalidStackAdjustment,
    MismatchedStackSize,
}

impl fmt::Display for SavedRegListError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRegisterRange => write!(f, "Invalid register range, must start from s0"),
            Self::InvalidStackAdjustment => write!(f, "Invalid stack adjustment value"),
            Self::MismatchedStackSize => write!(f, "Stack adjustment doesn't match register count"),
        }
    }
}

impl std::error::Error for SavedRegListError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedRegListWithStackAdj<const XLEN: u8> {
    /// End of the saved integer-register list.
    /// `None` means only `ra`; `Some(n)` means `s0` through `s[n]`.
    pub saved_register_end: Option<SavedIntegerRegister>,
    /// Stack adjustment amount.
    /// This must be positive and 16-byte aligned.
    pub stack_adj: u32,
}

/// Saved-register list representation for RV32I.
pub type SavedRegListWithStackAdjRv32 = SavedRegListWithStackAdj<32>;
/// Saved-register list representation for RV64I.
pub type SavedRegListWithStackAdjRv64 = SavedRegListWithStackAdj<64>;

impl<const XLEN: u8> SavedRegListWithStackAdj<XLEN> {
    /// Create a new saved-register list.
    /// The list always includes `ra` and then `s0` through `s[saved_end]`.
    pub fn new(saved_end: u8, stack_adj: u32) -> Result<Self, SavedRegListError> {
        // `saved_register_end` may not be 10.
        if saved_end == 10 {
            return Err(SavedRegListError::InvalidRegisterRange);
        }

        let saved_register_end = Some(
            SavedIntegerRegister::new(saved_end)
                .map_err(|_| SavedRegListError::InvalidRegisterRange)?,
        );

        // Validate the stack adjustment.
        if stack_adj == 0 || stack_adj % 16 != 0 {
            return Err(SavedRegListError::InvalidStackAdjustment);
        }

        let instance = Self {
            saved_register_end,
            stack_adj,
        };

        // Validate that the stack size is legal for this register list.
        if !instance.is_valid_combination() {
            return Err(SavedRegListError::MismatchedStackSize);
        }

        Ok(instance)
    }

    /// Construct the `ra`-only form.
    pub fn ra_only(stack_adj: u32) -> Result<Self, SavedRegListError> {
        if stack_adj == 0 || stack_adj % 16 != 0 {
            return Err(SavedRegListError::InvalidStackAdjustment);
        }

        let instance = Self {
            saved_register_end: None,
            stack_adj,
        };

        // Validate that the stack size is legal for this register list.
        if !instance.is_valid_combination() {
            return Err(SavedRegListError::MismatchedStackSize);
        }

        Ok(instance)
    }

    /// Return the number of registers in the list, including `ra`.
    pub fn register_count(&self) -> u8 {
        match self.saved_register_end {
            None => 1,                        // Only `ra`
            Some(end) => 1 + (end.get() + 1), // `ra` plus `s0` through `s[end]`
        }
    }

    /// Return whether the list contains only `ra`.
    pub fn is_ra_only(&self) -> bool {
        self.saved_register_end.is_none()
    }

    /// Derive the `rlist` encoding from the register count.
    /// This is used to determine `stack_adj_base`.
    fn get_rlist_encoding(&self) -> u8 {
        match self.saved_register_end {
            None => 4,                  // {ra}
            Some(end) => 5 + end.get(), // 5 + the `s[end]` index
        }
    }

    /// Return the base stack size and the allowed `stack_adj` values.
    fn get_stack_adj_info(&self) -> (u32, Vec<u32>) {
        let rlist = self.get_rlist_encoding();

        match XLEN {
            32 => {
                // RV32I layout
                match rlist {
                    4..=7 => (16, vec![16, 32, 48, 64]),   // {ra} 至 {ra, s0-s2}
                    8..=11 => (32, vec![32, 48, 64, 80]),  // {ra, s0-s3} 至 {ra, s0-s6}
                    12..=15 => (48, vec![48, 64, 80, 96]), // {ra, s0-s7} 至 {ra, s0-s10}
                    16 => (64, vec![64, 80, 96, 112]),     // {ra, s0-s11}
                    _ => (16, vec![16, 32, 48, 64]),       // Should be unreachable for valid rlist
                }
            }
            64 => {
                // RV64I layout
                match rlist {
                    4..=5 => (16, vec![16, 32, 48, 64]),      // {ra} 至 {ra, s0}
                    6..=7 => (32, vec![32, 48, 64, 80]),      // {ra, s0-s1} 至 {ra, s0-s2}
                    8..=9 => (48, vec![48, 64, 80, 96]),      // {ra, s0-s3} 至 {ra, s0-s4}
                    10..=11 => (64, vec![64, 80, 96, 112]),   // {ra, s0-s5} 至 {ra, s0-s6}
                    12..=13 => (80, vec![80, 96, 112, 128]),  // {ra, s0-s7} 至 {ra, s0-s8}
                    14..=15 => (96, vec![96, 112, 128, 144]), // {ra, s0-s9} 至 {ra, s0-s10}
                    16 => (112, vec![112, 128, 144, 160]),    // {ra, s0-s11}
                    _ => (16, vec![16, 32, 48, 64]), // Should be unreachable for valid rlist
                }
            }
            _ => panic!("Unsupported XLEN: {}", XLEN),
        }
    }

    /// Check whether the register-count and stack-adjustment combination is valid.
    pub fn is_valid_combination(&self) -> bool {
        let (_, valid_adjustments) = self.get_stack_adj_info();
        valid_adjustments.contains(&self.stack_adj)
    }

    /// Return all valid stack-adjustment values.
    pub fn valid_stack_adjustments(&self) -> Vec<u32> {
        let (_, valid_adjustments) = self.get_stack_adj_info();
        valid_adjustments
    }

    pub fn get_saved_reg_list_string(&self) -> String {
        // Build the register-list string the same way `Display` does.
        let mut reg_list = String::new();
        reg_list.push_str("{ra");
        if let Some(end) = self.saved_register_end {
            let end_val = end.get();
            if end_val == 0 {
                reg_list.push_str(", s0");
            } else {
                reg_list.push_str(&format!(", s0-s{}", end_val));
            }
        }
        reg_list.push_str("}");
        reg_list
    }

    pub fn get_stack_adjustment(&self) -> u32 {
        self.stack_adj
    }

    fn parse_reg_list_text(reg_list_text: &str) -> Result<Option<u8>, AsmParseError> {
        let trimmed = reg_list_text.trim();
        if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
            return Err(AsmParseError::invalid_operand(
                "SavedRegListWithStackAdj",
                format!("expected register list like `{{ra}}`, got `{trimmed}`"),
            ));
        }

        let inner = trimmed[1..trimmed.len() - 1].trim();
        if inner == "ra" {
            return Ok(None);
        }

        let suffix = inner.strip_prefix("ra, ").ok_or_else(|| {
            AsmParseError::invalid_operand(
                "SavedRegListWithStackAdj",
                format!("expected register list to start with `ra`, got `{trimmed}`"),
            )
        })?;

        if suffix == "s0" {
            return Ok(Some(0));
        }

        let end_text = suffix.strip_prefix("s0-s").ok_or_else(|| {
            AsmParseError::invalid_operand(
                "SavedRegListWithStackAdj",
                format!("unsupported saved register list `{trimmed}`"),
            )
        })?;
        let end = <u8 as AsmScalar>::parse_asm_scalar(end_text)?;
        Ok(Some(end))
    }

    pub fn parse_display_parts(reg_list_text: &str, stack_adj: i64) -> Result<Self, AsmParseError> {
        if stack_adj <= 0 {
            return Err(AsmParseError::invalid_operand(
                "SavedRegListWithStackAdj",
                format!("stack adjustment must be positive, got `{stack_adj}`"),
            ));
        }

        let stack_adj = u32::try_from(stack_adj).map_err(|_| {
            AsmParseError::invalid_operand(
                "SavedRegListWithStackAdj",
                format!("stack adjustment `{stack_adj}` is out of range for u32"),
            )
        })?;

        match Self::parse_reg_list_text(reg_list_text)? {
            None => Self::ra_only(stack_adj).map_err(|err| {
                AsmParseError::invalid_operand("SavedRegListWithStackAdj", err.to_string())
            }),
            Some(end) => Self::new(end, stack_adj).map_err(|err| {
                AsmParseError::invalid_operand("SavedRegListWithStackAdj", err.to_string())
            }),
        }
    }
}

impl<const XLEN: u8> crate::AsmEncode for SavedRegListWithStackAdj<XLEN> {
    fn encode_asm(&self) -> String {
        let mut s = String::from("{ra");
        if let Some(end) = self.saved_register_end {
            let end_val = end.get();
            if end_val == 0 {
                s.push_str(", s0");
            } else {
                s.push_str(&format!(", s0-s{}", end_val));
            }
        }
        s.push_str(&format!("}}, {}", self.stack_adj));
        s
    }
}

impl<const XLEN: u8> crate::AsmDecode for SavedRegListWithStackAdj<XLEN> {
    fn decode_asm(input: &str) -> Result<Self, crate::AsmParseError> {
        let (reg_list_text, stack_adj_text) = input.rsplit_once(',').ok_or_else(|| {
            crate::AsmParseError::invalid_operand(
                "SavedRegListWithStackAdj",
                format!("expected `{{ra}}, <stack_adj>`, got `{}`", input.trim()),
            )
        })?;
        let stack_adj = crate::parse_asm_i64_literal(stack_adj_text)?;
        Self::parse_display_parts(reg_list_text, stack_adj)
    }
}

impl<const XLEN: u8> std::fmt::Display for SavedRegListWithStackAdj<XLEN> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&crate::AsmEncode::encode_asm(self))
    }
}

impl<const XLEN: u8> std::str::FromStr for SavedRegListWithStackAdj<XLEN> {
    type Err = crate::AsmParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        <Self as crate::AsmDecode>::decode_asm(s)
    }
}

impl<const XLEN: u8> Random for SavedRegListWithStackAdj<XLEN> {
    type Output = Self;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        const TYPE_NAME: &str = stringify!(SavedRegListWithStackAdj);

        // Randomly choose the register count.
        let end_reg_options: [Option<u8>; 12] = [
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(7),
            Some(8),
            Some(9),
            Some(11),
        ];

        let saved_end = *end_reg_options.choose(rng).ok_or_else(|| {
            RandomGenerationError::new(TYPE_NAME, "no register options available")
        })?;

        let saved_register_end = match saved_end {
            None => None,
            Some(end) => Some(SavedIntegerRegister::new(end).map_err(|err| {
                RandomGenerationError::new(stringify!(SavedIntegerRegister), err)
            })?),
        };

        // Build a temporary instance so the valid `stack_adj` values can be queried.
        let temp_instance = Self {
            saved_register_end,
            stack_adj: 16, // 临时值
        };

        let valid_adjustments = temp_instance.valid_stack_adjustments();
        let stack_adj = *valid_adjustments.choose(rng).ok_or_else(|| {
            RandomGenerationError::new(TYPE_NAME, "no valid stack adjustments available")
        })?;

        let result = match saved_end {
            None => Self::ra_only(stack_adj),
            Some(end) => Self::new(end, stack_adj),
        };

        result.map_err(|err| RandomGenerationError::new(TYPE_NAME, err.to_string()))
    }
}

impl<const XLEN: u8> NumericOperandMutation for SavedRegListWithStackAdj<XLEN> {
    fn mutate_numeric_operands<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        _config: &RandomConfig,
    ) -> bool {
        let original = self.clone();
        let valid = self.valid_stack_adjustments();
        if valid.is_empty() {
            return false;
        }
        if let Some(choice) = valid.choose(rng) {
            self.stack_adj = *choice;
        }
        *self != original
    }
}

impl<const XLEN: u8> MemoryAware for SavedRegListWithStackAdj<XLEN> {
    fn is_memory_aware() -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotEqualCompressedSavedIntegerRegisterPair {
    pub reg1: CompressedSavedIntegerRegister,
    pub reg2: CompressedSavedIntegerRegister,
}

impl Random for NotEqualCompressedSavedIntegerRegisterPair {
    type Output = Self;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        const TYPE_NAME: &str = stringify!(NotEqualCompressedSavedIntegerRegisterPair);
        const MAX_ATTEMPTS: usize = 256;

        let reg1 = CompressedSavedIntegerRegister::random_with_rng(rng, config)?;
        for attempt in 0..MAX_ATTEMPTS {
            let reg2 = CompressedSavedIntegerRegister::random_with_rng(rng, config)?;
            if reg2 != reg1 {
                return Ok(Self { reg1, reg2 });
            }

            if attempt + 1 == MAX_ATTEMPTS {
                return Err(RandomGenerationError::exhausted(
                    TYPE_NAME,
                    MAX_ATTEMPTS,
                    Some(format!("unable to find register distinct from {}", reg1.encode_asm())),
                ));
            }
        }

        Err(RandomGenerationError::new(
            TYPE_NAME,
            "unexpected failure while generating distinct registers",
        ))
    }
}

impl MemoryAware for NotEqualCompressedSavedIntegerRegisterPair {
    fn is_memory_aware() -> bool {
        false
    }
}

impl crate::AsmEncode for NotEqualCompressedSavedIntegerRegisterPair {
    fn encode_asm(&self) -> String {
        format!("{}, {}", self.reg1.encode_asm(), self.reg2.encode_asm())
    }
}

impl crate::AsmDecode for NotEqualCompressedSavedIntegerRegisterPair {
    fn decode_asm(input: &str) -> Result<Self, crate::AsmParseError> {
        let (reg1_text, reg2_text) = input.split_once(',').ok_or_else(|| {
            crate::AsmParseError::invalid_operand(
                "NotEqualCompressedSavedIntegerRegisterPair",
                format!("expected `<reg1>, <reg2>`, got `{}`", input.trim()),
            )
        })?;

        let reg1 = CompressedSavedIntegerRegister::decode_asm(reg1_text)?;
        let reg2 = CompressedSavedIntegerRegister::decode_asm(reg2_text)?;
        if reg1 == reg2 {
            return Err(crate::AsmParseError::invalid_operand(
                "NotEqualCompressedSavedIntegerRegisterPair",
                "compressed saved register pair must contain distinct registers",
            ));
        }

        Ok(Self { reg1, reg2 })
    }
}

impl std::fmt::Display for NotEqualCompressedSavedIntegerRegisterPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&crate::AsmEncode::encode_asm(self))
    }
}

impl std::str::FromStr for NotEqualCompressedSavedIntegerRegisterPair {
    type Err = crate::AsmParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        <Self as crate::AsmDecode>::decode_asm(s)
    }
}
