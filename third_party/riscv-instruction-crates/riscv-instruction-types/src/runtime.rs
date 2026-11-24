use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};

use crate::{
    FLen, InstructionSequence, IntegerRegister, LoadImmediatePurpose, PseudoInstruction, Random,
    RandomConfig, RandomGenerationError, RegisterBank, RegisterRenderStyle, ValidatedValue,
    WritableCsr, XLenAddress, XLenImmediate,
};

pub trait RandomInstructionSequence {
    type Output;
    fn random_sequence_with_rng<R: rand::Rng>(
        rng: &mut R,
        config: &RandomConfig,
    ) -> Result<InstructionSequence<Self::Output>, RandomGenerationError>;
}

/// Runtime hints used when assembling a safe random instruction sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsrWriteSequenceSetup {
    pub csr: WritableCsr,
    pub rs1: IntegerRegister,
}

/// Structured sequence-generation metadata for generated instructions and wrappers.
pub trait InstructionSequenceHints {
    fn sequence_memory_base_registers(&self) -> Vec<IntegerRegister> {
        Vec::new()
    }

    fn sequence_csr_write_setup(&self) -> Option<CsrWriteSequenceSetup> {
        None
    }

    fn sequence_pc_relative_register(&self) -> Option<IntegerRegister> {
        None
    }

    fn sequence_touches_floating_point(&self) -> bool {
        false
    }
}

/// Access to pseudo-instruction state when a sequence is stored as a uniform instruction type.
pub trait InstructionSequenceItem {
    fn as_pseudo_instruction(&self) -> Option<&PseudoInstruction> {
        None
    }

    fn as_pseudo_instruction_mut(&mut self) -> Option<&mut PseudoInstruction> {
        None
    }

    fn is_pseudo_instruction(&self) -> bool {
        self.as_pseudo_instruction().is_some()
    }
}

/// Build a typed instruction sequence by injecting any required pseudo instructions around
/// the provided main instruction.
pub fn build_instruction_sequence<T, R>(
    main_instruction: T,
    rng: &mut R,
    config: &RandomConfig,
) -> Result<InstructionSequence<T>, RandomGenerationError>
where
    T: InstructionSequenceHints + From<PseudoInstruction> + InstructionSequenceItem,
    R: rand::Rng,
{
    let mut pre_instructions = Vec::new();

    for target_reg in main_instruction.sequence_memory_base_registers() {
        let address = XLenAddress::random_with_rng(rng, config)?;
        let immediate = XLenImmediate::from_address(address);

        if ValidatedValue::get(&target_reg) != 0 {
            pre_instructions.push(T::from(PseudoInstruction::LoadImmediate {
                rd: target_reg,
                immediate,
                purpose: LoadImmediatePurpose::BaseAddress,
            }));
        }
    }

    if let Some(csr_setup) = main_instruction.sequence_csr_write_setup() {
        if ValidatedValue::get(&csr_setup.rs1) != 0 {
            let immediate = XLenImmediate::from_unsigned_bits_for(
                config.isa_profile,
                csr_setup.csr.random_legal_value(rng),
            );
            pre_instructions.push(T::from(PseudoInstruction::LoadImmediate {
                rd: csr_setup.rs1,
                immediate,
                purpose: LoadImmediatePurpose::CsrValue,
            }));
        }
    }

    let mut pc_label: Option<String> = None;
    if let Some(target_reg) = main_instruction.sequence_pc_relative_register() {
        if ValidatedValue::get(&target_reg) != 0 {
            let label = format!(".Lriscv_jalr_{:016x}", rng.random::<u64>());
            pre_instructions.push(T::from(PseudoInstruction::LoadLabelAddress {
                rd: target_reg,
                label: label.clone(),
            }));
            pc_label = Some(label);
        }
    }

    if !pre_instructions.is_empty() {
        let mut seen = ::std::collections::HashSet::new();
        pre_instructions.retain(|instruction| match instruction.as_pseudo_instruction() {
            Some(PseudoInstruction::LoadImmediate { rd, .. })
            | Some(PseudoInstruction::LoadAddress { rd, .. })
            | Some(PseudoInstruction::LoadLabelAddress { rd, .. })
            | Some(PseudoInstruction::Move { rd, .. })
            | Some(PseudoInstruction::ReadFflags { rd })
            | Some(PseudoInstruction::ReadVl { rd }) => seen.insert(ValidatedValue::get(rd)),
            Some(
                PseudoInstruction::Label { .. }
                | PseudoInstruction::Nop
                | PseudoInstruction::Jump { .. }
                | PseudoInstruction::JumpRegister { .. }
                | PseudoInstruction::Return
                | PseudoInstruction::Call { .. }
                | PseudoInstruction::Tail { .. },
            ) => true,
            None => true,
        });
    }

    let mut post_instructions = Vec::new();

    if let Some(label) = pc_label {
        post_instructions.push(T::from(PseudoInstruction::Label { name: label }));
    }

    if config.append_fflags_readback
        && config.isa_profile.supports_flen(FLen::F32)
        && main_instruction.sequence_touches_floating_point()
    {
        let target_reg = <IntegerRegister as Random>::random_with_rng(rng, config)?;
        post_instructions.push(T::from(PseudoInstruction::ReadFflags { rd: target_reg }));
    }

    let sequence = InstructionSequence {
        pre_instructions,
        instruction: main_instruction,
        post_instructions,
    };
    sequence.validate_for_profile(config.isa_profile)?;
    Ok(sequence)
}

/// Trait for types that have a memory-aware const generic parameter
pub trait MemoryAware {
    /// Returns true if this type is memory-aware (needs immediate range constraints)
    fn is_memory_aware() -> bool;
}

/// Trait implemented by instruction structs and enums that describe memory accesses.
///
/// Implementors can return the runtime value of their offset operand (if any).
pub trait MemoryAccessInstruction {
    /// Returns the offset operand value (in signed 64-bit form) when present.
    fn offset_operand_value(&self) -> Option<i64>;
}

/// Numbering space used when setting a register operand at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisterNumbering {
    /// Use the register type's encoded/newtype-local number directly.
    Encoded,
    /// Use the architectural physical register number (for example `x18` / `f9` / `v3`).
    Physical,
}

/// Runtime role metadata for a register operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeRegisterRole {
    Plain,
    MemoryBase,
    SavedAlias,
}

/// Runtime usage metadata for an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeOperandUsage {
    Destination,
    Source,
    MemoryBase,
    MemoryOffset,
    CsrAddress,
    Aggregate,
    SyntaxOnly,
}

/// Runtime description of a concrete architectural register value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeRegister {
    pub bank: RegisterBank,
    pub render_style: RegisterRenderStyle,
    pub encoded_number: u8,
    pub physical_number: u8,
}

impl Display for RuntimeRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(
            &RegisterDisplayName {
                bank: self.bank,
                render_style: self.render_style,
                number: self.encoded_number,
            },
            f,
        )
    }
}

/// Small display wrapper used by runtime register metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterDisplayName {
    bank: RegisterBank,
    render_style: RegisterRenderStyle,
    number: u8,
}

impl Display for RegisterDisplayName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.render_style {
            RegisterRenderStyle::Integer => write!(f, "x{}", self.number),
            RegisterRenderStyle::FloatingPoint => write!(f, "f{}", self.number),
            RegisterRenderStyle::Vector => write!(f, "v{}", self.number),
            RegisterRenderStyle::SavedInteger => write!(f, "s{}", self.number),
        }
    }
}

/// Assignment request used when rewriting a register operand uniformly at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeRegisterAssignment {
    pub bank: RegisterBank,
    pub number: u8,
    pub numbering: RegisterNumbering,
}

/// Runtime description of a register operand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeRegisterOperand {
    pub name: &'static str,
    pub usage: RuntimeOperandUsage,
    pub role: RuntimeRegisterRole,
    pub register: RuntimeRegister,
}

/// Runtime description of a memory operand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMemoryOperand {
    pub index: usize,
    pub base_register_operand_name: Option<&'static str>,
    pub base_register: Option<RuntimeRegisterOperand>,
    pub fixed_base: Option<RuntimeMemoryBase>,
    pub offset_operand_name: Option<&'static str>,
    pub offset_value: Option<i64>,
}

impl RuntimeMemoryOperand {
    pub fn effective_address_from_base_value(&self, base_value: u64) -> Option<u64> {
        let offset = self.offset_value.unwrap_or(0);
        if offset >= 0 {
            base_value.checked_add(offset as u64)
        } else {
            base_value.checked_sub(offset.unsigned_abs())
        }
    }
}

/// Fixed memory base anchors that are encoded by the ISA form instead of an operand field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeMemoryBase {
    Register(RuntimeRegister),
    ProgramCounter,
}

/// Partial update for a memory operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RuntimeMemoryOperandPatch {
    pub base_register: Option<RuntimeRegisterAssignment>,
    pub offset_value: Option<i64>,
}

/// Errors produced while rewriting instruction operands through the uniform runtime API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperandAccessError {
    pub target: String,
    pub message: String,
}

impl OperandAccessError {
    pub fn new(target: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            message: message.into(),
        }
    }

    pub fn missing_register_operand(name: &str) -> Self {
        Self::new(
            format!("register operand `{name}`"),
            "instruction does not expose a register operand with this name",
        )
    }

    pub fn missing_memory_operand(index: usize) -> Self {
        Self::new(
            format!("memory operand #{index}"),
            "instruction does not expose a memory operand with this index",
        )
    }
}

impl Display for OperandAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.target, self.message)
    }
}

impl std::error::Error for OperandAccessError {}

/// Uniform runtime access to register and memory operands without hand-written enum matching.
pub trait RuntimeOperandAccess {
    fn register_operands(&self) -> Vec<RuntimeRegisterOperand> {
        Vec::new()
    }

    fn memory_operands(&self) -> Vec<RuntimeMemoryOperand> {
        Vec::new()
    }

    fn set_register_operand(
        &mut self,
        operand_name: &str,
        _register: RuntimeRegisterAssignment,
    ) -> Result<bool, OperandAccessError> {
        Err(OperandAccessError::missing_register_operand(operand_name))
    }

    fn set_memory_operand(
        &mut self,
        index: usize,
        _patch: RuntimeMemoryOperandPatch,
    ) -> Result<bool, OperandAccessError> {
        Err(OperandAccessError::missing_memory_operand(index))
    }

    fn set_memory_base_register(
        &mut self,
        index: usize,
        register: RuntimeRegisterAssignment,
    ) -> Result<bool, OperandAccessError> {
        self.set_memory_operand(
            index,
            RuntimeMemoryOperandPatch {
                base_register: Some(register),
                offset_value: None,
            },
        )
    }

    fn set_memory_offset_value(
        &mut self,
        index: usize,
        offset_value: i64,
    ) -> Result<bool, OperandAccessError> {
        self.set_memory_operand(
            index,
            RuntimeMemoryOperandPatch {
                base_register: None,
                offset_value: Some(offset_value),
            },
        )
    }
}

/// Trait for mutating numeric operands while keeping register numbers stable.
pub trait NumericOperandMutation {
    /// Attempt to mutate one or more numeric operands in-place.
    /// Returns true if any operand was changed.
    fn mutate_numeric_operands<R: rand::Rng>(&mut self, rng: &mut R, config: &RandomConfig)
    -> bool;
}
