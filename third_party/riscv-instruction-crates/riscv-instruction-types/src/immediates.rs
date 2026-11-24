use rand::seq::IndexedRandom as _;
use riscv_instruction_macros::{DeriveRandom, DeriveValidatedValue};
use serde::{Deserialize, Serialize};

use crate::validated::{
    mutate_validated_fixed_i32, mutate_validated_signed_with_sampler, mutate_validated_u8,
    mutate_validated_u16, mutate_validated_unsigned_with_sampler, random_validated_fixed_i32,
    random_validated_signed_with_sampler, random_validated_unsigned_with_sampler,
};
use crate::*;

macro_rules! impl_mem_aware_immediate {
    ($type_name:ident) => {
        impl<const BITS: u8, const MEM_AWARE: bool> MemoryAware for $type_name<BITS, MEM_AWARE> {
            fn is_memory_aware() -> bool {
                MEM_AWARE
            }
        }
    };
}

macro_rules! impl_signed_sampled_immediate {
    ($type_name:ident, $display_name:expr, |$rng:ident, $min:ident, $max:ident, $config_ref:ident| $sample:expr) => {
        impl<const BITS: u8, const MEM_AWARE: bool> Random for $type_name<BITS, MEM_AWARE> {
            type Output = Self;

            fn random_with_rng<R: rand::Rng>(
                rng: &mut R,
                config: &RandomConfig,
            ) -> Result<Self::Output, RandomGenerationError> {
                random_validated_signed_with_sampler::<Self, _, _>(
                    rng,
                    config,
                    $display_name,
                    |$rng, $min, $max| {
                        let $config_ref = config;
                        $sample
                    },
                )
            }
        }

        impl<const BITS: u8, const MEM_AWARE: bool> NumericOperandMutation
            for $type_name<BITS, MEM_AWARE>
        {
            fn mutate_numeric_operands<R: rand::Rng>(
                &mut self,
                rng: &mut R,
                config: &RandomConfig,
            ) -> bool {
                mutate_validated_signed_with_sampler::<Self, _, _>(
                    self,
                    rng,
                    config,
                    |$rng, $min, $max| {
                        let $config_ref = config;
                        $sample
                    },
                )
            }
        }
    };
}

macro_rules! impl_unsigned_sampled_immediate {
    ($type_name:ident, $display_name:expr, |$rng:ident, $min:ident, $max:ident, $config_ref:ident| $sample:expr) => {
        impl<const BITS: u8, const MEM_AWARE: bool> Random for $type_name<BITS, MEM_AWARE> {
            type Output = Self;

            fn random_with_rng<R: rand::Rng>(
                rng: &mut R,
                config: &RandomConfig,
            ) -> Result<Self::Output, RandomGenerationError> {
                random_validated_unsigned_with_sampler::<Self, _, _>(
                    rng,
                    config,
                    $display_name,
                    |$rng, $min, $max| {
                        let $config_ref = config;
                        $sample
                    },
                )
            }
        }

        impl<const BITS: u8, const MEM_AWARE: bool> NumericOperandMutation
            for $type_name<BITS, MEM_AWARE>
        {
            fn mutate_numeric_operands<R: rand::Rng>(
                &mut self,
                rng: &mut R,
                config: &RandomConfig,
            ) -> bool {
                mutate_validated_unsigned_with_sampler::<Self, _, _>(
                    self,
                    rng,
                    config,
                    |$rng, $min, $max| {
                        let $config_ref = config;
                        $sample
                    },
                )
            }
        }
    };
}

macro_rules! impl_fixed_i32_immediate {
    ($type_name:ident, $display_name:expr, $fixed_value:expr) => {
        impl<const BITS: u8, const MEM_AWARE: bool> Random for $type_name<BITS, MEM_AWARE> {
            type Output = Self;

            fn random_with_rng<R: rand::Rng>(
                _rng: &mut R,
                _config: &RandomConfig,
            ) -> Result<Self::Output, RandomGenerationError> {
                random_validated_fixed_i32::<Self>($display_name, $fixed_value)
            }
        }

        impl<const BITS: u8, const MEM_AWARE: bool> NumericOperandMutation
            for $type_name<BITS, MEM_AWARE>
        {
            fn mutate_numeric_operands<R: rand::Rng>(
                &mut self,
                _rng: &mut R,
                _config: &RandomConfig,
            ) -> bool {
                mutate_validated_fixed_i32(self, $fixed_value)
            }
        }
    };
}

macro_rules! impl_fli_immediate_wrapper {
    ($type_name:ident, $required_width:expr) => {
        impl crate::AsmEncode for $type_name {
            fn encode_asm(&self) -> String {
                self.0.encode_asm()
            }
        }

        impl std::fmt::Display for $type_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(&crate::AsmEncode::encode_asm(self))
            }
        }

        impl crate::AsmDecode for $type_name {
            fn decode_asm(input: &str) -> Result<Self, crate::AsmParseError> {
                Ok(Self(FliConstant::decode_asm(input)?))
            }
        }

        impl std::str::FromStr for $type_name {
            type Err = crate::AsmParseError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                <Self as crate::AsmDecode>::decode_asm(s)
            }
        }

        impl Random for $type_name {
            type Output = Self;

            fn random_with_rng<R: rand::Rng>(
                rng: &mut R,
                config: &RandomConfig,
            ) -> Result<Self::Output, RandomGenerationError> {
                if !config.supports_float_width($required_width) {
                    return Err(RandomGenerationError::new(
                        stringify!($type_name),
                        format!("requires FLEN >= {}", $required_width),
                    ));
                }
                Ok(Self(random_fli_constant(rng)))
            }
        }

        impl MemoryAware for $type_name {
            fn is_memory_aware() -> bool {
                false
            }
        }

        impl NumericOperandMutation for $type_name {
            fn mutate_numeric_operands<R: rand::Rng>(
                &mut self,
                rng: &mut R,
                config: &RandomConfig,
            ) -> bool {
                if !config.supports_float_width($required_width) {
                    return false;
                }
                let original = *self;
                self.0 = random_fli_constant(rng);
                *self != original
            }
        }
    };
}

// Small validated scalar wrappers share the same non-memory-aware,
// MIN/MAX-based numeric-mutation behavior.
// Small validated scalar wrappers share the same non-memory-aware mutation glue.
macro_rules! impl_non_memory_aware_validated_mutation {
    ($type_name:ident, $mutate_fn:ident) => {
        impl MemoryAware for $type_name {
            fn is_memory_aware() -> bool {
                false
            }
        }

        impl NumericOperandMutation for $type_name {
            fn mutate_numeric_operands<R: rand::Rng>(
                &mut self,
                rng: &mut R,
                _config: &RandomConfig,
            ) -> bool {
                $mutate_fn(self, rng, Self::MIN, Self::MAX)
            }
        }
    };
}

/// Immediate value with bit length validation
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue, DeriveRandom,
)]
#[validated(
    min = "-(1i32 << (BITS - 1))",
    max = "(1i32 << (BITS - 1)) - 1",
    name = "Immediate"
)]
pub struct Immediate<const BITS: u8, const MEM_AWARE: bool = false>(i32);

impl_mem_aware_immediate!(Immediate);

impl<const BITS: u8, const MEM_AWARE: bool> NumericOperandMutation for Immediate<BITS, MEM_AWARE> {
    fn mutate_numeric_operands<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        config: &RandomConfig,
    ) -> bool {
        mutate_validated_signed_with_sampler::<Self, _, _>(self, rng, config, |rng, min, max| {
            sample_signed_value(rng, min, max, BITS)
        })
    }
}

/// Unsigned immediate value with bit length validation
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue, DeriveRandom,
)]
#[validated(
    min = "0",
    max = "((1u64 << BITS) - 1) as u32",
    name = "Unsigned immediate"
)]
pub struct UImmediate<const BITS: u8, const MEM_AWARE: bool = false>(u32);

impl_mem_aware_immediate!(UImmediate);

/// Shift amount immediate (shamt)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue)]
#[validated(min = "0", max = "((1u64 << BITS) - 1) as u32", name = "Shift amount")]
pub struct ShiftAmount<const BITS: u8, const MEM_AWARE: bool = false>(u32);

impl_mem_aware_immediate!(ShiftAmount);

impl_unsigned_sampled_immediate!(
    ShiftAmount,
    stringify!(ShiftAmount),
    |rng, min, max, _config| sample_shamt_value(rng, min, max, BITS)
);

/// Compressed signed immediate value (C/Zc* encodings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue)]
#[validated(
    min = "-(1i32 << (BITS - 1))",
    max = "(1i32 << (BITS - 1)) - 1",
    name = "Compressed immediate"
)]
pub struct CompressedImmediate<const BITS: u8, const MEM_AWARE: bool = false>(i32);

impl_mem_aware_immediate!(CompressedImmediate);

impl_signed_sampled_immediate!(
    CompressedImmediate,
    stringify!(CompressedImmediate),
    |rng, min, max, _config| sample_compressed_signed_immediate(rng, min, max, BITS)
);

/// Compressed unsigned immediate value (C/Zc* encodings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue)]
#[validated(
    min = "0",
    max = "((1u64 << BITS) - 1) as u32",
    name = "Compressed unsigned immediate"
)]
pub struct CompressedUImmediate<const BITS: u8, const MEM_AWARE: bool = false>(u32);

impl_mem_aware_immediate!(CompressedUImmediate);

impl_unsigned_sampled_immediate!(
    CompressedUImmediate,
    stringify!(CompressedUImmediate),
    |rng, min, max, _config| sample_compressed_unsigned_immediate(rng, min, max, BITS)
);

/// Fixed immediate for linearized branch/jump (PC-relative, 32-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue)]
#[validated(
    min = "-(1i32 << (BITS - 1))",
    max = "(1i32 << (BITS - 1)) - 1",
    name = "Linear branch immediate"
)]
pub struct LinearBranchImmediate<const BITS: u8, const MEM_AWARE: bool = false>(i32);

impl_mem_aware_immediate!(LinearBranchImmediate);

impl_fixed_i32_immediate!(LinearBranchImmediate, stringify!(LinearBranchImmediate), 4);

/// Fixed immediate for linearized compressed branch/jump (PC-relative, 16-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue)]
#[validated(
    min = "-(1i32 << (BITS - 1))",
    max = "(1i32 << (BITS - 1)) - 1",
    name = "Linear compressed branch immediate"
)]
pub struct LinearCompressedBranchImmediate<const BITS: u8, const MEM_AWARE: bool = false>(i32);

impl_mem_aware_immediate!(LinearCompressedBranchImmediate);

impl_fixed_i32_immediate!(
    LinearCompressedBranchImmediate,
    stringify!(LinearCompressedBranchImmediate),
    2
);

/// Fixed immediate for linearized JALR (rs1 + imm, imm = 0)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue)]
#[validated(
    min = "-(1i32 << (BITS - 1))",
    max = "(1i32 << (BITS - 1)) - 1",
    name = "Linear JALR immediate"
)]
pub struct LinearJalrImmediate<const BITS: u8, const MEM_AWARE: bool = false>(i32);

impl_mem_aware_immediate!(LinearJalrImmediate);

impl_fixed_i32_immediate!(LinearJalrImmediate, stringify!(LinearJalrImmediate), 0);

/// Vector signed immediate value (used by V-extension immediate forms).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue)]
#[validated(
    min = "-(1i32 << (BITS - 1))",
    max = "(1i32 << (BITS - 1)) - 1",
    name = "Vector immediate"
)]
pub struct VectorImmediate<const BITS: u8, const MEM_AWARE: bool = false>(i32);

impl_mem_aware_immediate!(VectorImmediate);

impl_signed_sampled_immediate!(
    VectorImmediate,
    stringify!(VectorImmediate),
    |rng, min, max, config| sample_vector_signed_immediate(rng, min, max, BITS, config)
);

/// Vector unsigned immediate value (used by V-extension immediate forms).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue)]
#[validated(
    min = "0",
    max = "((1u64 << BITS) - 1) as u32",
    name = "Vector unsigned immediate"
)]
pub struct VectorUImmediate<const BITS: u8, const MEM_AWARE: bool = false>(u32);

impl_mem_aware_immediate!(VectorUImmediate);

impl_unsigned_sampled_immediate!(
    VectorUImmediate,
    stringify!(VectorUImmediate),
    |rng, min, max, config| sample_vector_unsigned_immediate(rng, min, max, BITS, config)
);

impl<const BITS: u8, const MEM_AWARE: bool> NumericOperandMutation for UImmediate<BITS, MEM_AWARE> {
    fn mutate_numeric_operands<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        config: &RandomConfig,
    ) -> bool {
        mutate_validated_unsigned_with_sampler::<Self, _, _>(self, rng, config, |rng, min, max| {
            sample_unsigned_value(rng, min, max, BITS)
        })
    }
}

/// CSR address
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue, DeriveRandom,
)]
#[validated(min = "0", max = "0xFFF", name = "CSR address", display = "0x{:x}")]
pub struct CSRAddress(u16);

impl_non_memory_aware_validated_mutation!(CSRAddress, mutate_validated_u16);

/// Rounding mode
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue, DeriveRandom,
)]
#[validated(min = "0", max = "7", name = "Rounding mode", skip_display)]
pub struct RoundingMode(u8);

impl_non_memory_aware_validated_mutation!(RoundingMode, mutate_validated_u8);

impl crate::AsmEncode for RoundingMode {
    fn encode_asm(&self) -> String {
        let mnemonic = match self.0 {
            0 => "rne",
            1 => "rtz",
            2 => "rdn",
            3 => "rup",
            4 => "rmm",
            5 | 6 | 7 => "dyn",
            _ => unreachable!(),
        };
        mnemonic.to_string()
    }
}

impl crate::AsmDecode for RoundingMode {
    fn decode_asm(input: &str) -> Result<Self, crate::AsmParseError> {
        let value = match input.trim().to_ascii_lowercase().as_str() {
            "rne" => 0,
            "rtz" => 1,
            "rdn" => 2,
            "rup" => 3,
            "rmm" => 4,
            "dyn" => 7,
            other => {
                return Err(crate::AsmParseError::invalid_operand(
                    "RoundingMode",
                    format!("unknown rounding mode `{other}`"),
                ));
            }
        };
        Self::new(value).map_err(|err| crate::AsmParseError::invalid_operand("RoundingMode", err))
    }
}

impl std::fmt::Display for RoundingMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&crate::AsmEncode::encode_asm(self))
    }
}

impl std::str::FromStr for RoundingMode {
    type Err = crate::AsmParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        <Self as crate::AsmDecode>::decode_asm(s)
    }
}

/// Fence mode
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue, DeriveRandom,
)]
#[validated(
    min = "0",
    max = "15",
    name = "Fence mode",
    skip_display,
    forbidden = "0"
)]
pub struct FenceMode(u8);

impl_non_memory_aware_validated_mutation!(FenceMode, mutate_validated_u8);

impl crate::AsmEncode for FenceMode {
    fn encode_asm(&self) -> String {
        let value = self.0;
        let mut mnemonic = String::new();
        if (value & 8) != 0 { mnemonic.push('i'); }
        if (value & 4) != 0 { mnemonic.push('o'); }
        if (value & 2) != 0 { mnemonic.push('r'); }
        if (value & 1) != 0 { mnemonic.push('w'); }
        mnemonic
    }
}

impl crate::AsmDecode for FenceMode {
    fn decode_asm(input: &str) -> Result<Self, crate::AsmParseError> {
        let trimmed = input.trim().to_ascii_lowercase();
        if trimmed.is_empty() {
            return Err(crate::AsmParseError::empty_input("FenceMode"));
        }

        let mut value = 0u8;
        for ch in trimmed.chars() {
            match ch {
                'i' => value |= 8,
                'o' => value |= 4,
                'r' => value |= 2,
                'w' => value |= 1,
                other => {
                    return Err(crate::AsmParseError::invalid_operand(
                        "FenceMode",
                        format!("unknown fence mode flag `{other}`"),
                    ));
                }
            }
        }

        Self::new(value).map_err(|err| crate::AsmParseError::invalid_operand("FenceMode", err))
    }
}

impl std::fmt::Display for FenceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&crate::AsmEncode::encode_asm(self))
    }
}

impl std::str::FromStr for FenceMode {
    type Err = crate::AsmParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        <Self as crate::AsmDecode>::decode_asm(s)
    }
}

/// The 32 hardware constants that `fli` can load in the RISC-V `Zfa` extension.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FliConstant {
    /// Index 0: -1.0
    NegativeOne,
    /// Index 1: minimum normal
    MinNormal,
    /// Index 2: 1.0 * 2^-16
    PowerOfTwoN16,
    /// Index 3: 1.0 * 2^-15
    PowerOfTwoN15,
    /// Index 4: 1.0 * 2^-8
    PowerOfTwoN8,
    /// Index 5: 1.0 * 2^-7
    PowerOfTwoN7,
    /// Index 6: 1.0 * 2^-4
    PowerOfTwoN4,
    /// Index 7: 1.0 * 2^-3
    PowerOfTwoN3,
    /// Index 8: 0.25
    Point25,
    /// Index 9: 0.3125
    Point3125,
    /// Index 10: 0.375
    Point375,
    /// Index 11: 0.4375
    Point4375,
    /// Index 12: 0.5
    Point5,
    /// Index 13: 0.625
    Point625,
    /// Index 14: 0.75
    Point75,
    /// Index 15: 0.875
    Point875,
    /// Index 16: 1.0
    One,
    /// Index 17: 1.25
    OnePoint25,
    /// Index 18: 1.5
    OnePoint5,
    /// Index 19: 1.75
    OnePoint75,
    /// Index 20: 2.0
    Two,
    /// Index 21: 2.5
    TwoPoint5,
    /// Index 22: 3.0
    Three,
    /// Index 23: 4.0
    Four,
    /// Index 24: 8.0
    Eight,
    /// Index 25: 16.0
    Sixteen,
    /// Index 26: 2^7
    PowerOfTwo7,
    /// Index 27: 2^8
    PowerOfTwo8,
    /// Index 28: 2^15
    PowerOfTwo15,
    /// Index 29: 2^16
    PowerOfTwo16,
    /// Index 30: +inf
    PositiveInfinity,
    /// Index 31: NaN
    NaN,
}

const FLI_ALL: [FliConstant; 32] = [
    FliConstant::NegativeOne,
    FliConstant::MinNormal,
    FliConstant::PowerOfTwoN16,
    FliConstant::PowerOfTwoN15,
    FliConstant::PowerOfTwoN8,
    FliConstant::PowerOfTwoN7,
    FliConstant::PowerOfTwoN4,
    FliConstant::PowerOfTwoN3,
    FliConstant::Point25,
    FliConstant::Point3125,
    FliConstant::Point375,
    FliConstant::Point4375,
    FliConstant::Point5,
    FliConstant::Point625,
    FliConstant::Point75,
    FliConstant::Point875,
    FliConstant::One,
    FliConstant::OnePoint25,
    FliConstant::OnePoint5,
    FliConstant::OnePoint75,
    FliConstant::Two,
    FliConstant::TwoPoint5,
    FliConstant::Three,
    FliConstant::Four,
    FliConstant::Eight,
    FliConstant::Sixteen,
    FliConstant::PowerOfTwo7,
    FliConstant::PowerOfTwo8,
    FliConstant::PowerOfTwo15,
    FliConstant::PowerOfTwo16,
    FliConstant::PositiveInfinity,
    FliConstant::NaN,
];

const FLI_SPECIAL: [FliConstant; 8] = [
    FliConstant::NegativeOne,
    FliConstant::MinNormal,
    FliConstant::One,
    FliConstant::Two,
    FliConstant::PowerOfTwoN16,
    FliConstant::PowerOfTwo16,
    FliConstant::PositiveInfinity,
    FliConstant::NaN,
];

fn random_fli_constant<R: rand::Rng>(rng: &mut R) -> FliConstant {
    if rng.random_ratio(2, 3) {
        *FLI_SPECIAL.choose(rng).unwrap()
    } else {
        *FLI_ALL.choose(rng).unwrap()
    }
}

impl Random for FliConstant {
    type Output = Self;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(random_fli_constant(rng))
    }
}

// Updated `AsmEncode` implementation for FliConstant.
impl crate::AsmEncode for FliConstant {
    fn encode_asm(&self) -> String {
        let literal = match self {
            Self::NegativeOne => "-1.0",
            Self::MinNormal => "min",
            Self::PowerOfTwoN16 => "0x1.0p-16",
            Self::PowerOfTwoN15 => "0x1.0p-15",
            Self::PowerOfTwoN8 => "0x1.0p-8",
            Self::PowerOfTwoN7 => "0x1.0p-7",
            Self::PowerOfTwoN4 => "0x1.0p-4",
            Self::PowerOfTwoN3 => "0x1.0p-3",
            Self::Point25 => "0.25",
            Self::Point3125 => "0.3125",
            Self::Point375 => "0.375",
            Self::Point4375 => "0.4375",
            Self::Point5 => "0.5",
            Self::Point625 => "0.625",
            Self::Point75 => "0.75",
            Self::Point875 => "0.875",
            Self::One => "1.0",
            Self::OnePoint25 => "1.25",
            Self::OnePoint5 => "1.5",
            Self::OnePoint75 => "1.75",
            Self::Two => "2.0",
            Self::TwoPoint5 => "2.5",
            Self::Three => "3.0",
            Self::Four => "4.0",
            Self::Eight => "8.0",
            Self::Sixteen => "16.0",
            Self::PowerOfTwo7 => "128.0",
            Self::PowerOfTwo8 => "256.0",
            Self::PowerOfTwo15 => "32768.0",
            Self::PowerOfTwo16 => "65536.0",
            Self::PositiveInfinity => "inf",
            Self::NaN => "nan",
        };
        literal.to_string()
    }
}

impl crate::AsmDecode for FliConstant {
    fn decode_asm(input: &str) -> Result<Self, crate::AsmParseError> {
        match input.trim().to_ascii_lowercase().as_str() {
            "-1.0" => Ok(Self::NegativeOne),
            "min" => Ok(Self::MinNormal),
            "0x1.0p-16" => Ok(Self::PowerOfTwoN16),
            "0x1.0p-15" => Ok(Self::PowerOfTwoN15),
            "0x1.0p-8" => Ok(Self::PowerOfTwoN8),
            "0x1.0p-7" => Ok(Self::PowerOfTwoN7),
            "0x1.0p-4" => Ok(Self::PowerOfTwoN4),
            "0x1.0p-3" => Ok(Self::PowerOfTwoN3),
            "0.25" => Ok(Self::Point25),
            "0.3125" => Ok(Self::Point3125),
            "0.375" => Ok(Self::Point375),
            "0.4375" => Ok(Self::Point4375),
            "0.5" => Ok(Self::Point5),
            "0.625" => Ok(Self::Point625),
            "0.75" => Ok(Self::Point75),
            "0.875" => Ok(Self::Point875),
            "1.0" => Ok(Self::One),
            "1.25" => Ok(Self::OnePoint25),
            "1.5" => Ok(Self::OnePoint5),
            "1.75" => Ok(Self::OnePoint75),
            "2.0" => Ok(Self::Two),
            "2.5" => Ok(Self::TwoPoint5),
            "3.0" => Ok(Self::Three),
            "4.0" => Ok(Self::Four),
            "8.0" => Ok(Self::Eight),
            "16.0" => Ok(Self::Sixteen),
            "128.0" => Ok(Self::PowerOfTwo7),
            "256.0" => Ok(Self::PowerOfTwo8),
            "32768.0" => Ok(Self::PowerOfTwo15),
            "65536.0" => Ok(Self::PowerOfTwo16),
            "inf" => Ok(Self::PositiveInfinity),
            "nan" => Ok(Self::NaN),
            other => Err(crate::AsmParseError::invalid_operand(
                "FliConstant",
                format!("unknown fli literal `{other}`"),
            )),
        }
    }
}

impl std::fmt::Display for FliConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&crate::AsmEncode::encode_asm(self))
    }
}

impl std::str::FromStr for FliConstant {
    type Err = crate::AsmParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        <Self as crate::AsmDecode>::decode_asm(s)
    }
}

impl MemoryAware for FliConstant {
    fn is_memory_aware() -> bool {
        false
    }
}

impl NumericOperandMutation for FliConstant {
    fn mutate_numeric_operands<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        _config: &RandomConfig,
    ) -> bool {
        let original = *self;
        *self = random_fli_constant(rng);
        *self != original
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FloatImmediate16(pub FliConstant);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FloatImmediate32(pub FliConstant);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FloatImmediate64(pub FliConstant);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FloatImmediate128(pub FliConstant);

impl_fli_immediate_wrapper!(FloatImmediate16, 32);
impl_fli_immediate_wrapper!(FloatImmediate32, 32);
impl_fli_immediate_wrapper!(FloatImmediate64, 64);
impl_fli_immediate_wrapper!(FloatImmediate128, 128);
