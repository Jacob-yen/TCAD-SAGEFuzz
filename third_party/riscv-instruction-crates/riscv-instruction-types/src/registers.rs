use std::{
    fmt::{self, Display},
    marker::PhantomData,
};

use rand::seq::IndexedRandom as _;
use serde::{Deserialize, Serialize};


use crate::{
    AsmDecode, AsmEncode, AsmParseError, AsmScalar, MemoryAware, NumericOperandMutation,
    OperandAccessError, Random, RandomConfig, RandomGenerationError, RegisterNumbering,
    RuntimeRegister, RuntimeRegisterAssignment, ValidatedValue,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegisterBank {
    Integer,
    FloatingPoint,
    Vector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegisterRenderStyle {
    Integer,
    FloatingPoint,
    Vector,
    SavedInteger,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegisterRandomSource {
    IntegerRegisterRange,
    FloatingPointRegisterRange,
    VectorRegisterRange,
    MemoryRegisterNumberRange,
}

/// Shared register profile metadata.
///
/// This keeps storage layout, display style, physical-register mapping, and
/// random-source configuration in one place so public register types do not
/// have to duplicate the same implementation.
pub trait RegisterProfile: 'static {
    const TYPE_NAME: &'static str;
    const NAME: &'static str;
    const BANK: RegisterBank;
    const RENDER_STYLE: RegisterRenderStyle;
    const MIN: u8;
    const MAX: u8;
    const FORBIDDEN: &'static [u8] = &[];
    const MULTIPLE_OF: Option<u8> = None;
    const ODD_ONLY: bool = false;
    const RANDOM_SOURCE: Option<RegisterRandomSource>;

    fn physical_index(value: u8) -> u8 {
        value
    }
}

/// Architectural register value with pluggable profile metadata.
///
/// Public register types differ only by profile; the stored representation is
/// still a single `u8`.
pub struct Register<P: RegisterProfile>(u8, PhantomData<fn() -> P>);

impl<P: RegisterProfile> Clone for Register<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: RegisterProfile> Copy for Register<P> {}

impl<P: RegisterProfile> PartialEq for Register<P> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<P: RegisterProfile> Eq for Register<P> {}

impl<P: RegisterProfile> std::hash::Hash for Register<P> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<P: RegisterProfile> fmt::Debug for Register<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(P::TYPE_NAME).field(&self.0).finish()
    }
}

impl<P: RegisterProfile> Serialize for Register<P> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, P: RegisterProfile> Deserialize<'de> for Register<P> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

impl<P: RegisterProfile> Register<P> {
    pub const fn encoded_value(self) -> u8 {
        self.0
    }

    pub fn physical_index(self) -> u8 {
        P::physical_index(self.0)
    }

    fn available_values(config: &RandomConfig) -> Result<Vec<u8>, RandomGenerationError> {
        let mut values: Vec<u8> = (P::MIN..=P::MAX)
            .filter(|value| !P::FORBIDDEN.contains(value))
            .filter(|value| match P::MULTIPLE_OF {
                Some(multiple) => *value % multiple == 0,
                None => true,
            })
            .filter(|value| !P::ODD_ONLY || *value % 2 == 1)
            .collect();

        if let Some(source) = P::RANDOM_SOURCE {
            let (config_min, config_max, invalid_message) = match source {
                RegisterRandomSource::IntegerRegisterRange => {
                    let (min, max) = config.register_operand_config.integer_register_number_range;
                    (min, max, format!("invalid configured range {min}..={max}"))
                }
                RegisterRandomSource::FloatingPointRegisterRange => {
                    let (min, max) = config
                        .register_operand_config
                        .floating_point_register_number_range;
                    (min, max, format!("invalid configured range {min}..={max}"))
                }
                RegisterRandomSource::VectorRegisterRange => {
                    let (min, max) = config.register_operand_config.vector_register_number_range;
                    (min, max, format!("invalid configured range {min}..={max}"))
                }
                RegisterRandomSource::MemoryRegisterNumberRange => {
                    let (min, max) = config
                        .memory_access_config
                        .memory_base_register_number_range;
                    (min, max, format!("invalid configured range {min}..={max}"))
                }
            };

            if config_min > config_max {
                return Err(RandomGenerationError::new(P::TYPE_NAME, invalid_message));
            }

            values.retain(|value| {
                let physical_index = P::physical_index(*value);
                (config_min..=config_max).contains(&physical_index)
            });

            if values.is_empty() {
                let overlap_message = match source {
                    RegisterRandomSource::IntegerRegisterRange => format!(
                        "no {} values overlap with configured integer register range (x{}-x{})",
                        P::TYPE_NAME,
                        config_min,
                        config_max
                    ),
                    RegisterRandomSource::FloatingPointRegisterRange => format!(
                        "no {} values overlap with configured floating-point register range (f{}-f{})",
                        P::TYPE_NAME,
                        config_min,
                        config_max
                    ),
                    RegisterRandomSource::VectorRegisterRange => format!(
                        "no {} values overlap with configured vector register range (v{}-v{})",
                        P::TYPE_NAME,
                        config_min,
                        config_max
                    ),
                    RegisterRandomSource::MemoryRegisterNumberRange => {
                        if P::MIN == 8 && P::MAX == 15 {
                            format!(
                                "no intersection between compressed register range (x8-x15) and configured range (x{}-x{})",
                                config_min, config_max
                            )
                        } else {
                            format!(
                                "no {} values overlap with configured memory register range (x{}-x{})",
                                P::TYPE_NAME,
                                config_min,
                                config_max
                            )
                        }
                    }
                };

                return Err(RandomGenerationError::new(P::TYPE_NAME, overlap_message));
            }
        }

        Ok(values)
    }
}

impl<P: RegisterProfile> ValidatedValue<u8> for Register<P> {
    const MIN: u8 = P::MIN;
    const MAX: u8 = P::MAX;
    const MULTIPLE_OF: Option<u8> = P::MULTIPLE_OF;
    const FORBIDDEN: &'static [u8] = P::FORBIDDEN;
    const ODD_ONLY: bool = P::ODD_ONLY;
    type Error = String;

    fn new(value: u8) -> Result<Self, Self::Error> {
        if Self::FORBIDDEN.contains(&value) {
            return Err(format!(
                "{} value {} is forbidden for this register profile",
                P::NAME,
                value
            ));
        }

        if let Some(multiple) = Self::MULTIPLE_OF {
            if value % multiple != 0 {
                return Err(format!("{} must be a multiple of {}", P::NAME, multiple));
            }
        }

        if Self::ODD_ONLY && value % 2 == 0 {
            return Err(format!("{} must be odd", P::NAME));
        }

        if value < Self::MIN || value > Self::MAX {
            return Err(format!(
                "{} {} exceeds range ({} to {})",
                P::NAME,
                value,
                Self::MIN,
                Self::MAX
            ));
        }

        Ok(Self(value, PhantomData))
    }

    fn get(&self) -> u8 {
        self.0
    }

    fn set(&mut self, value: u8) -> Result<(), Self::Error> {
        *self = Self::new(value)?;
        Ok(())
    }
}

impl<P: RegisterProfile> AsmEncode for Register<P> {
    fn encode_asm(&self) -> String {
        match P::RENDER_STYLE {
            RegisterRenderStyle::Integer => format!("x{}", self.0),
            RegisterRenderStyle::FloatingPoint => format!("f{}", self.0),
            RegisterRenderStyle::Vector => format!("v{}", self.0),
            RegisterRenderStyle::SavedInteger => format!("s{}", self.0),
        }
    }
}

impl<P: RegisterProfile> AsmDecode for Register<P> {
    fn decode_asm(input: &str) -> Result<Self, AsmParseError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(AsmParseError::empty_input(P::TYPE_NAME));
        }

        let prefix = match P::RENDER_STYLE {
            RegisterRenderStyle::Integer => "x",
            RegisterRenderStyle::FloatingPoint => "f",
            RegisterRenderStyle::Vector => "v",
            RegisterRenderStyle::SavedInteger => "s",
        };

        let value_text = trimmed.strip_prefix(prefix).ok_or_else(|| {
            AsmParseError::invalid_operand(
                P::TYPE_NAME,
                format!("expected register with `{prefix}` prefix, got `{trimmed}`"),
            )
        })?;
        let value = <u8 as AsmScalar>::parse_asm_scalar(value_text)?;
        Self::new(value).map_err(|err| AsmParseError::invalid_operand(P::TYPE_NAME, err))
    }
}

impl<P: RegisterProfile> fmt::Display for Register<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.encode_asm())
    }
}

impl<P: RegisterProfile> std::str::FromStr for Register<P> {
    type Err = AsmParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::decode_asm(s)
    }
}

impl<P: RegisterProfile> Random for Register<P> {
    type Output = Self;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        let available = Self::available_values(config)?;

        let use_boundary_value = if config.edge_case_probability <= 0.0 {
            false
        } else if config.edge_case_probability >= 1.0 {
            true
        } else {
            rng.random_range(0.0..1.0) < config.edge_case_probability
        };

        let choice = if use_boundary_value {
            let mut boundary_candidates =
                vec![*available.first().unwrap(), *available.last().unwrap()];
            if let Some(mid) = available.get(available.len() / 2) {
                boundary_candidates.push(*mid);
            }
            boundary_candidates.sort_unstable();
            boundary_candidates.dedup();
            *boundary_candidates.choose(rng).unwrap()
        } else {
            *available.choose(rng).unwrap()
        };

        Self::new(choice).map_err(|err| RandomGenerationError::new(P::TYPE_NAME, err))
    }
}

impl<P: RegisterProfile> MemoryAware for Register<P> {
    fn is_memory_aware() -> bool {
        false
    }
}

impl<P: RegisterProfile> NumericOperandMutation for Register<P> {
    fn mutate_numeric_operands<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        config: &RandomConfig,
    ) -> bool {
        if let Ok(new_value) = Self::random_with_rng(rng, config) {
            let changed = new_value != *self;
            *self = new_value;
            return changed;
        }
        false
    }
}

/// Common register trait for future schema-driven codegen.
///
/// This leaves one uniform interface for future schema-driven code generation
/// so upper layers only need to care about the bank and physical register
/// number rather than a specific public alias type.
pub trait ArchitecturalRegister {
    const BANK: RegisterBank;

    fn encoded_value(self) -> u8;
    fn physical_index(self) -> u8;
}

impl<P: RegisterProfile> ArchitecturalRegister for Register<P> {
    const BANK: RegisterBank = P::BANK;

    fn encoded_value(self) -> u8 {
        self.encoded_value()
    }

    fn physical_index(self) -> u8 {
        self.physical_index()
    }
}

const SAVED_INTEGER_REGISTER_TO_X: [u8; 12] = [8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];
const COMPRESSED_SAVED_INTEGER_REGISTER_TO_X: [u8; 8] = [8, 9, 18, 19, 20, 21, 22, 23];

pub struct IntegerRegisterProfile;
impl RegisterProfile for IntegerRegisterProfile {
    const TYPE_NAME: &'static str = "IntegerRegister";
    const NAME: &'static str = "Integer register";
    const BANK: RegisterBank = RegisterBank::Integer;
    const RENDER_STYLE: RegisterRenderStyle = RegisterRenderStyle::Integer;
    const MIN: u8 = 0;
    const MAX: u8 = 31;
    const RANDOM_SOURCE: Option<RegisterRandomSource> =
        Some(RegisterRandomSource::IntegerRegisterRange);
}

pub struct BaseAddressRegisterProfile;
impl RegisterProfile for BaseAddressRegisterProfile {
    const TYPE_NAME: &'static str = "BaseAddressRegister";
    const NAME: &'static str = "Base address register";
    const BANK: RegisterBank = RegisterBank::Integer;
    const RENDER_STYLE: RegisterRenderStyle = RegisterRenderStyle::Integer;
    const MIN: u8 = 0;
    const MAX: u8 = 31;
    const RANDOM_SOURCE: Option<RegisterRandomSource> =
        Some(RegisterRandomSource::MemoryRegisterNumberRange);
}

pub struct CompressedBaseAddressRegisterProfile;
impl RegisterProfile for CompressedBaseAddressRegisterProfile {
    const TYPE_NAME: &'static str = "CompressedBaseAddressRegister";
    const NAME: &'static str = "Compressed base address register";
    const BANK: RegisterBank = RegisterBank::Integer;
    const RENDER_STYLE: RegisterRenderStyle = RegisterRenderStyle::Integer;
    const MIN: u8 = 8;
    const MAX: u8 = 15;
    const RANDOM_SOURCE: Option<RegisterRandomSource> =
        Some(RegisterRandomSource::MemoryRegisterNumberRange);
}

pub struct SavedIntegerRegisterProfile;
impl RegisterProfile for SavedIntegerRegisterProfile {
    const TYPE_NAME: &'static str = "SavedIntegerRegister";
    const NAME: &'static str = "Saved Integer register";
    const BANK: RegisterBank = RegisterBank::Integer;
    const RENDER_STYLE: RegisterRenderStyle = RegisterRenderStyle::SavedInteger;
    const MIN: u8 = 0;
    const MAX: u8 = 11;
    const RANDOM_SOURCE: Option<RegisterRandomSource> =
        Some(RegisterRandomSource::IntegerRegisterRange);

    fn physical_index(value: u8) -> u8 {
        SAVED_INTEGER_REGISTER_TO_X[value as usize]
    }
}

pub struct CompressedSavedIntegerRegisterProfile;
impl RegisterProfile for CompressedSavedIntegerRegisterProfile {
    const TYPE_NAME: &'static str = "CompressedSavedIntegerRegister";
    const NAME: &'static str = "Compressed Saved Integer register";
    const BANK: RegisterBank = RegisterBank::Integer;
    const RENDER_STYLE: RegisterRenderStyle = RegisterRenderStyle::SavedInteger;
    const MIN: u8 = 0;
    const MAX: u8 = 7;
    const RANDOM_SOURCE: Option<RegisterRandomSource> =
        Some(RegisterRandomSource::IntegerRegisterRange);

    fn physical_index(value: u8) -> u8 {
        COMPRESSED_SAVED_INTEGER_REGISTER_TO_X[value as usize]
    }
}

pub struct FloatingPointRegisterProfile;
impl RegisterProfile for FloatingPointRegisterProfile {
    const TYPE_NAME: &'static str = "FloatingPointRegister";
    const NAME: &'static str = "Floating-point register";
    const BANK: RegisterBank = RegisterBank::FloatingPoint;
    const RENDER_STYLE: RegisterRenderStyle = RegisterRenderStyle::FloatingPoint;
    const MIN: u8 = 0;
    const MAX: u8 = 31;
    const RANDOM_SOURCE: Option<RegisterRandomSource> =
        Some(RegisterRandomSource::FloatingPointRegisterRange);
}

pub struct VectorRegisterProfile;
impl RegisterProfile for VectorRegisterProfile {
    const TYPE_NAME: &'static str = "VectorRegister";
    const NAME: &'static str = "Vector register";
    const BANK: RegisterBank = RegisterBank::Vector;
    const RENDER_STYLE: RegisterRenderStyle = RegisterRenderStyle::Vector;
    const MIN: u8 = 0;
    const MAX: u8 = 31;
    const RANDOM_SOURCE: Option<RegisterRandomSource> =
        Some(RegisterRandomSource::VectorRegisterRange);
}

pub struct CompressedIntegerRegisterProfile;
impl RegisterProfile for CompressedIntegerRegisterProfile {
    const TYPE_NAME: &'static str = "CompressedIntegerRegister";
    const NAME: &'static str = "Compressed integer register";
    const BANK: RegisterBank = RegisterBank::Integer;
    const RENDER_STYLE: RegisterRenderStyle = RegisterRenderStyle::Integer;
    const MIN: u8 = 8;
    const MAX: u8 = 15;
    const RANDOM_SOURCE: Option<RegisterRandomSource> =
        Some(RegisterRandomSource::IntegerRegisterRange);
}

pub struct CompressedFloatingPointRegisterProfile;
impl RegisterProfile for CompressedFloatingPointRegisterProfile {
    const TYPE_NAME: &'static str = "CompressedFloatingPointRegister";
    const NAME: &'static str = "Compressed floating-point register";
    const BANK: RegisterBank = RegisterBank::FloatingPoint;
    const RENDER_STYLE: RegisterRenderStyle = RegisterRenderStyle::FloatingPoint;
    const MIN: u8 = 8;
    const MAX: u8 = 15;
    const RANDOM_SOURCE: Option<RegisterRandomSource> =
        Some(RegisterRandomSource::FloatingPointRegisterRange);
}

/// Integer register identifier (x0-x31)
pub type IntegerRegister = Register<IntegerRegisterProfile>;

/// Base address register for memory access instructions (x0-x31).
/// This alias keeps the old public API while reusing the unified register implementation.
pub type BaseAddressRegister = Register<BaseAddressRegisterProfile>;

/// Compressed base address register for memory access instructions (x8-x15).
pub type CompressedBaseAddressRegister = Register<CompressedBaseAddressRegisterProfile>;

/// Saved integer register identifier (s0-s11).
pub type SavedIntegerRegister = Register<SavedIntegerRegisterProfile>;

/// Compressed saved integer register identifier (s0-s7).
pub type CompressedSavedIntegerRegister = Register<CompressedSavedIntegerRegisterProfile>;

/// Floating-point register identifier (f0-f31)
pub type FloatingPointRegister = Register<FloatingPointRegisterProfile>;

/// Vector register identifier (v0-v31)
pub type VectorRegister = Register<VectorRegisterProfile>;

/// Compressed integer register identifier (x8-x15)
pub type CompressedIntegerRegister = Register<CompressedIntegerRegisterProfile>;

/// Compressed floating-point register identifier (f8-f15)
pub type CompressedFloatingPointRegister = Register<CompressedFloatingPointRegisterProfile>;

/// Projection used by random sequence setup to recover a physical integer register.
pub trait SequenceIntegerRegisterOperand {
    fn sequence_integer_register(self) -> Option<IntegerRegister>;
}

impl<P: RegisterProfile> SequenceIntegerRegisterOperand for Register<P> {
    fn sequence_integer_register(self) -> Option<IntegerRegister> {
        if P::BANK != RegisterBank::Integer {
            return None;
        }

        IntegerRegister::new(self.physical_index()).ok()
    }
}

pub trait RuntimeRegisterOperandValue: ValidatedValue<u8> + Copy {
    const TYPE_NAME: &'static str;
    const BANK: RegisterBank;
    const RENDER_STYLE: RegisterRenderStyle;

    fn physical_number(encoded_number: u8) -> u8 {
        encoded_number
    }
}

impl<P: RegisterProfile> RuntimeRegisterOperandValue for Register<P> {
    const TYPE_NAME: &'static str = P::TYPE_NAME;
    const BANK: RegisterBank = P::BANK;
    const RENDER_STYLE: RegisterRenderStyle = P::RENDER_STYLE;

    fn physical_number(encoded_number: u8) -> u8 {
        P::physical_index(encoded_number)
    }
}

pub fn runtime_register<V: RuntimeRegisterOperandValue>(value: V) -> RuntimeRegister {
    let encoded_number = value.get();
    RuntimeRegister {
        bank: V::BANK,
        render_style: V::RENDER_STYLE,
        encoded_number,
        physical_number: V::physical_number(encoded_number),
    }
}

pub fn runtime_fixed_integer_register(number: u8) -> RuntimeRegister {
    RuntimeRegister {
        bank: RegisterBank::Integer,
        render_style: RegisterRenderStyle::Integer,
        encoded_number: number,
        physical_number: number,
    }
}

fn register_from_physical_number<V: RuntimeRegisterOperandValue>(
    physical_number: u8,
) -> Result<V, OperandAccessError>
where
    V::Error: Display,
{
    for encoded in V::MIN..=V::MAX {
        if V::FORBIDDEN.contains(&encoded) {
            continue;
        }
        if let Some(multiple) = V::MULTIPLE_OF
            && encoded % multiple != 0
        {
            continue;
        }
        if V::ODD_ONLY && encoded % 2 == 0 {
            continue;
        }
        if V::physical_number(encoded) == physical_number {
            return V::new(encoded).map_err(|err| {
                OperandAccessError::new(
                    V::TYPE_NAME,
                    format!("failed to construct runtime register from physical number {physical_number}: {err}"),
                )
            });
        }
    }

    Err(OperandAccessError::new(
        V::TYPE_NAME,
        format!(
            "physical register {} is not valid for {}",
            physical_number,
            V::TYPE_NAME
        ),
    ))
}

pub fn set_register_from_runtime_assignment<V: RuntimeRegisterOperandValue>(
    target: &mut V,
    operand_name: &str,
    assignment: RuntimeRegisterAssignment,
) -> Result<bool, OperandAccessError>
where
    V::Error: Display,
{
    if assignment.bank != V::BANK {
        return Err(OperandAccessError::new(
            format!("register operand `{operand_name}`"),
            format!(
                "bank mismatch: operand expects {:?} register, got {:?}",
                V::BANK,
                assignment.bank
            ),
        ));
    }

    let new_value = match assignment.numbering {
        RegisterNumbering::Encoded => V::new(assignment.number).map_err(|err| {
            OperandAccessError::new(
                format!("register operand `{operand_name}`"),
                format!("invalid encoded register {}: {err}", assignment.number),
            )
        })?,
        RegisterNumbering::Physical => register_from_physical_number::<V>(assignment.number)
            .map_err(|err| {
                OperandAccessError::new(format!("register operand `{operand_name}`"), err.message)
            })?,
    };

    let changed = target.get() != new_value.get();
    *target = new_value;
    Ok(changed)
}

pub fn set_validated_i32_operand_from_i64<T>(
    target: &mut T,
    operand_name: &str,
    value: i64,
) -> Result<bool, OperandAccessError>
where
    T: ValidatedValue<i32>,
    T::Error: Display,
{
    let value = i32::try_from(value).map_err(|_| {
        OperandAccessError::new(
            format!("operand `{operand_name}`"),
            format!("value {value} does not fit in i32"),
        )
    })?;
    let changed = target.get() != value;
    target.set(value).map_err(|err| {
        OperandAccessError::new(
            format!("operand `{operand_name}`"),
            format!("failed to set signed immediate to {value}: {err}"),
        )
    })?;
    Ok(changed)
}

pub fn set_validated_u32_operand_from_i64<T>(
    target: &mut T,
    operand_name: &str,
    value: i64,
) -> Result<bool, OperandAccessError>
where
    T: ValidatedValue<u32>,
    T::Error: Display,
{
    let value = u32::try_from(value).map_err(|_| {
        OperandAccessError::new(
            format!("operand `{operand_name}`"),
            format!("value {value} does not fit in u32"),
        )
    })?;
    let changed = target.get() != value;
    target.set(value).map_err(|err| {
        OperandAccessError::new(
            format!("operand `{operand_name}`"),
            format!("failed to set unsigned immediate to {value}: {err}"),
        )
    })?;
    Ok(changed)
}

pub fn set_bool_operand_from_i64(
    target: &mut bool,
    operand_name: &str,
    value: i64,
) -> Result<bool, OperandAccessError> {
    let new_value = match value {
        0 => false,
        1 => true,
        other => {
            return Err(OperandAccessError::new(
                format!("operand `{operand_name}`"),
                format!("boolean operand only accepts 0 or 1, got {other}"),
            ));
        }
    };

    let changed = *target != new_value;
    *target = new_value;
    Ok(changed)
}
