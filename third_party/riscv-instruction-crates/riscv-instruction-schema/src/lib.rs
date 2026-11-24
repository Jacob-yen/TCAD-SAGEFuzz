use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Display,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ISAExtension(String);

impl ISAExtension {
    pub fn new<S: Into<String>>(value: S) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for ISAExtension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ISABase {
    RV32,
    RV64,
}

impl Display for ISABase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ISABase::RV32 => write!(f, "RV32"),
            ISABase::RV64 => write!(f, "RV64"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ISAModule {
    pub base: ISABase,
    pub extension: ISAExtension,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OperandKind {
    #[default]
    Unknown,
    Register(RegisterSpec),
    Immediate(ImmediateSpec),
    Csr(CsrSpec),
    RoundMode,
    FenceMode,
    FliConstant,
    SavedRegListWithStackAdj {
        xlen_dependent: bool,
    },
    RegisterPair(RegisterPairSpec),
}

fn bool_is_false(value: &bool) -> bool {
    !*value
}

impl Display for OperandKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperandKind::Unknown => write!(f, "Unknown"),
            OperandKind::Register(spec) => write!(f, "{} Register", spec.bank),
            OperandKind::Immediate(spec) => write!(f, "{} Immediate", spec.kind),
            OperandKind::Csr(_) => write!(f, "CSR"),
            OperandKind::RoundMode => write!(f, "Round Mode"),
            OperandKind::FenceMode => write!(f, "Fence Mode"),
            OperandKind::FliConstant => write!(f, "FLI Constant"),
            OperandKind::SavedRegListWithStackAdj { .. } => {
                write!(f, "Saved Register List With Stack Adjustment")
            }
            OperandKind::RegisterPair(spec) => write!(f, "{} Register Pair", spec.bank),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterSpec {
    pub bank: RegisterBank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisterBank {
    Integer,
    FloatingPoint,
    Vector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisterEncoding {
    Standard,
    Compressed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisterRole {
    Plain,
    MemoryBase,
    SavedAlias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterSemantics {
    pub bank: RegisterBank,
    pub encoding: RegisterEncoding,
    pub role: RegisterRole,
}

impl Display for RegisterBank {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterBank::Integer => write!(f, "Integer"),
            RegisterBank::FloatingPoint => write!(f, "Floating Point"),
            RegisterBank::Vector => write!(f, "Vector"),
        }
    }
}

impl RegisterEncoding {
    pub fn from_bit_length(bank: RegisterBank, bit_length: u8) -> Self {
        match bank {
            RegisterBank::Integer | RegisterBank::FloatingPoint => match bit_length {
                3 => RegisterEncoding::Compressed,
                5 => RegisterEncoding::Standard,
                _ => panic!(
                    "unsupported register bit length {} for {:?} register",
                    bit_length, bank
                ),
            },
            RegisterBank::Vector => RegisterEncoding::Standard,
        }
    }
}

impl RegisterBank {
    pub fn runtime_plain_type_name(self, encoding: RegisterEncoding) -> &'static str {
        match (self, encoding) {
            (RegisterBank::Integer, RegisterEncoding::Standard) => "IntegerRegister",
            (RegisterBank::Integer, RegisterEncoding::Compressed) => "CompressedIntegerRegister",
            (RegisterBank::FloatingPoint, RegisterEncoding::Standard) => "FloatingPointRegister",
            (RegisterBank::FloatingPoint, RegisterEncoding::Compressed) => {
                "CompressedFloatingPointRegister"
            }
            (RegisterBank::Vector, _) => "VectorRegister",
        }
    }

    pub fn runtime_display_format(self, _encoding: RegisterEncoding) -> &'static str {
        match self {
            RegisterBank::Integer => "x{}",
            RegisterBank::FloatingPoint => "f{}",
            RegisterBank::Vector => "v{}",
        }
    }

    pub fn runtime_type_label(self, encoding: RegisterEncoding) -> &'static str {
        match (self, encoding) {
            (RegisterBank::Integer, RegisterEncoding::Standard) => "Integer register",
            (RegisterBank::Integer, RegisterEncoding::Compressed) => "Compressed integer register",
            (RegisterBank::FloatingPoint, RegisterEncoding::Standard) => "Floating point register",
            (RegisterBank::FloatingPoint, RegisterEncoding::Compressed) => {
                "Compressed floating point register"
            }
            (RegisterBank::Vector, _) => "Vector register",
        }
    }

    pub fn restricted_random_domain(self, use_memory_register_range: bool) -> &'static str {
        if use_memory_register_range {
            return "memory_register";
        }

        match self {
            RegisterBank::Integer => "integer_register",
            RegisterBank::FloatingPoint => "floating_point_register",
            RegisterBank::Vector => "vector_register",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmediateSpec {
    pub kind: ImmediateKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImmediateKind {
    Signed,
    Unsigned,
    ShiftAmount,
    CompressedSigned,
    CompressedUnsigned,
    LinearBranch,
    LinearCompressedBranch,
    LinearJalr,
    VectorSigned,
    VectorUnsigned,
    Float16,
    Float32,
    Float64,
    Float128,
}

impl Display for ImmediateKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImmediateKind::Signed => write!(f, "Signed"),
            ImmediateKind::Unsigned => write!(f, "Unsigned"),
            ImmediateKind::ShiftAmount => write!(f, "Shift Amount"),
            ImmediateKind::CompressedSigned => write!(f, "Compressed Signed"),
            ImmediateKind::CompressedUnsigned => write!(f, "Compressed Unsigned"),
            ImmediateKind::LinearBranch => write!(f, "Linear Branch"),
            ImmediateKind::LinearCompressedBranch => write!(f, "Linear Compressed Branch"),
            ImmediateKind::LinearJalr => write!(f, "Linear JALR"),
            ImmediateKind::VectorSigned => write!(f, "Vector Signed"),
            ImmediateKind::VectorUnsigned => write!(f, "Vector Unsigned"),
            ImmediateKind::Float16 => write!(f, "Float16"),
            ImmediateKind::Float32 => write!(f, "Float32"),
            ImmediateKind::Float64 => write!(f, "Float64"),
            ImmediateKind::Float128 => write!(f, "Float128"),
        }
    }
}

impl ImmediateKind {
    pub fn is_restricted_runtime_family(self) -> bool {
        matches!(
            self,
            ImmediateKind::Signed
                | ImmediateKind::Unsigned
                | ImmediateKind::ShiftAmount
                | ImmediateKind::CompressedSigned
                | ImmediateKind::CompressedUnsigned
                | ImmediateKind::LinearBranch
                | ImmediateKind::LinearCompressedBranch
                | ImmediateKind::LinearJalr
                | ImmediateKind::VectorSigned
                | ImmediateKind::VectorUnsigned
        )
    }

    pub fn is_unsigned(self) -> bool {
        matches!(
            self,
            ImmediateKind::Unsigned
                | ImmediateKind::ShiftAmount
                | ImmediateKind::CompressedUnsigned
                | ImmediateKind::VectorUnsigned
        )
    }

    pub fn supports_inline_boolean_value(self) -> bool {
        matches!(
            self,
            ImmediateKind::Unsigned | ImmediateKind::VectorUnsigned
        )
    }

    pub fn runtime_base_type_name(self) -> &'static str {
        match self {
            ImmediateKind::Signed => "Immediate",
            ImmediateKind::Unsigned => "UImmediate",
            ImmediateKind::ShiftAmount => "ShiftAmount",
            ImmediateKind::CompressedSigned => "CompressedImmediate",
            ImmediateKind::CompressedUnsigned => "CompressedUImmediate",
            ImmediateKind::LinearBranch => "LinearBranchImmediate",
            ImmediateKind::LinearCompressedBranch => "LinearCompressedBranchImmediate",
            ImmediateKind::LinearJalr => "LinearJalrImmediate",
            ImmediateKind::VectorSigned => "VectorImmediate",
            ImmediateKind::VectorUnsigned => "VectorUImmediate",
            ImmediateKind::Float16 => "FloatImmediate16",
            ImmediateKind::Float32 => "FloatImmediate32",
            ImmediateKind::Float64 => "FloatImmediate64",
            ImmediateKind::Float128 => "FloatImmediate128",
        }
    }

    pub fn restricted_runtime_base_type_name(self, operand_name: &str) -> &'static str {
        match (self, operand_name) {
            (ImmediateKind::Signed, "nzimm") => "NZImmediate",
            _ => self.runtime_base_type_name(),
        }
    }

    pub fn sampler_function_name(self) -> Option<&'static str> {
        match self {
            ImmediateKind::Unsigned => Some("sample_unsigned_value"),
            ImmediateKind::ShiftAmount => Some("sample_shamt_value"),
            ImmediateKind::CompressedUnsigned => Some("sample_compressed_unsigned_immediate"),
            ImmediateKind::VectorUnsigned => Some("sample_vector_unsigned_immediate"),
            ImmediateKind::Signed => Some("sample_signed_value"),
            ImmediateKind::CompressedSigned => Some("sample_compressed_signed_immediate"),
            ImmediateKind::VectorSigned => Some("sample_vector_signed_immediate"),
            ImmediateKind::LinearBranch
            | ImmediateKind::LinearCompressedBranch
            | ImmediateKind::LinearJalr
            | ImmediateKind::Float16
            | ImmediateKind::Float32
            | ImmediateKind::Float64
            | ImmediateKind::Float128 => None,
        }
    }

    pub fn fixed_value(self) -> Option<i32> {
        match self {
            ImmediateKind::LinearBranch => Some(4),
            ImmediateKind::LinearCompressedBranch => Some(2),
            ImmediateKind::LinearJalr => Some(0),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CsrSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterPairSpec {
    pub bank: RegisterBank,
    pub distinct: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OperandRestriction {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiple_of: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_max: Option<(i64, i64)>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forbidden_values: Vec<i64>,
    #[serde(default)]
    pub odd_only: bool,
}

impl OperandRestriction {
    pub fn has_explicit_constraints(&self) -> bool {
        self.multiple_of.is_some()
            || self.min_max.is_some()
            || !self.forbidden_values.is_empty()
            || self.odd_only
    }
}

pub type OperandConstraints = OperandRestriction;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operand {
    pub name: String,
    #[serde(default)]
    pub kind: OperandKind,
    pub bit_lengths: HashMap<ISABase, u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restrictions: Option<OperandRestriction>,
    /// Per-ISA restriction overrides. When present for a given ISA base, takes precedence over
    /// the shared `restrictions` field. `None` values explicitly indicate "no restriction" for
    /// that ISA base (overriding any shared restriction).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub per_isa_restrictions: HashMap<ISABase, Option<OperandRestriction>>,
}

impl Operand {
    /// Return the restriction that applies for a specific ISA base.
    /// `per_isa_restrictions` overrides the shared `restrictions` when an entry is present.
    pub fn restrictions_for(&self, isa_base: ISABase) -> Option<&OperandRestriction> {
        if let Some(override_val) = self.per_isa_restrictions.get(&isa_base) {
            return override_val.as_ref();
        }
        self.restrictions.as_ref()
    }

    pub fn bit_length_for(&self, isa_base: ISABase) -> u8 {
        *self.bit_lengths.get(&isa_base).unwrap_or_else(|| {
            panic!(
                "operand '{}' does not define a bit length for ISA base '{}'",
                self.name, isa_base
            )
        })
    }

    pub fn register_semantics(
        &self,
        isa_base: ISABase,
        instruction: &Instruction,
    ) -> Option<RegisterSemantics> {
        let OperandKind::Register(spec) = self.kind else {
            return None;
        };

        let role = if instruction.is_memory_base_operand(&self.name) {
            RegisterRole::MemoryBase
        } else if spec.bank == RegisterBank::Integer && self.prefers_abi_aliases() {
            RegisterRole::SavedAlias
        } else {
            RegisterRole::Plain
        };

        Some(RegisterSemantics {
            bank: spec.bank,
            encoding: RegisterEncoding::from_bit_length(spec.bank, self.bit_length_for(isa_base)),
            role,
        })
    }

    pub fn runtime_immediate_kind(&self, instruction: &Instruction) -> Option<ImmediateKind> {
        let OperandKind::Immediate(spec) = self.kind else {
            return None;
        };

        Some(match spec.kind {
            ImmediateKind::Signed
                if instruction.uses_vector_registers()
                    && !instruction.is_memory_offset_operand(&self.name) =>
            {
                ImmediateKind::VectorSigned
            }
            ImmediateKind::Unsigned
                if instruction.uses_vector_registers()
                    && !instruction.is_memory_offset_operand(&self.name) =>
            {
                ImmediateKind::VectorUnsigned
            }
            other => other,
        })
    }

    pub fn usage(&self, instruction: &Instruction) -> OperandUsage {
        if instruction.is_memory_base_operand(&self.name) {
            return OperandUsage::MemoryBase;
        }
        if instruction.is_memory_offset_operand(&self.name) {
            return OperandUsage::MemoryOffset;
        }

        match self.kind {
            OperandKind::Csr(_) => OperandUsage::CsrAddress,
            OperandKind::SavedRegListWithStackAdj { .. } | OperandKind::RegisterPair(_) => {
                OperandUsage::Aggregate
            }
            _ if is_destination_operand_name(&self.name) => OperandUsage::Destination,
            _ if is_source_operand_name(&self.name) => OperandUsage::Source,
            _ => OperandUsage::SyntaxOnly,
        }
    }

    pub fn presentation(&self) -> OperandPresentation {
        if self.prefers_abi_aliases() {
            OperandPresentation::AbiAliasPreferred
        } else {
            OperandPresentation::PhysicalName
        }
    }

    /// Some 1-bit unsigned immediates are emitted as plain `bool` in the final model.
    pub fn uses_inline_boolean_value(&self, instruction: &Instruction) -> bool {
        self.runtime_immediate_kind(instruction)
            .is_some_and(ImmediateKind::supports_inline_boolean_value)
            && self.bit_lengths.values().all(|bit_length| *bit_length == 1)
    }

    pub fn supports_numeric_mutation(&self, instruction: &Instruction) -> bool {
        match self.kind {
            OperandKind::Unknown => panic!("operand '{}' still has unknown kind", self.name),
            OperandKind::Register(_) | OperandKind::RegisterPair(_) => false,
            OperandKind::Immediate(_) => self.runtime_immediate_kind(instruction).is_some(),
            OperandKind::Csr(_)
            | OperandKind::RoundMode
            | OperandKind::FenceMode
            | OperandKind::FliConstant
            | OperandKind::SavedRegListWithStackAdj { .. } => true,
        }
    }

    pub fn prefers_abi_aliases(&self) -> bool {
        matches!(
            self.kind,
            OperandKind::SavedRegListWithStackAdj { .. } | OperandKind::RegisterPair(_)
        ) || matches!(self.name.as_str(), "r1s" | "r2s")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperandUsage {
    Destination,
    Source,
    MemoryBase,
    MemoryOffset,
    CsrAddress,
    Aggregate,
    SyntaxOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperandPresentation {
    PhysicalName,
    AbiAliasPreferred,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instruction {
    pub name: String,
    pub extension: ISAExtension,
    pub isa_bases: Vec<ISABase>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub required_extensions: BTreeMap<ISABase, Vec<ISAExtension>>,
    pub operands: Vec<Operand>,
    pub assembly_syntax: AssemblySyntax,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_access: Option<MemoryAddressInfo>,
    #[serde(default, skip_serializing_if = "InstructionSemantics::is_empty")]
    pub semantics: InstructionSemantics,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InstructionSequenceSemantics {
    pub memory_base_operand: Option<String>,
    pub memory_base_fixed: Option<String>,
    pub writable_csr_operand: Option<String>,
    pub csr_source_register_operand: Option<String>,
    pub pc_relative_target_operand: Option<String>,
    pub pc_relative_target_fixed: Option<String>,
    pub touches_floating_point: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InstructionSemantics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub csr: Option<InstructionCsrSemantics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_flow: Option<InstructionControlFlowSemantics>,
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub touches_floating_point: bool,
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub touches_vector_registers: bool,
}

impl InstructionSemantics {
    pub fn is_empty(&self) -> bool {
        self.csr.is_none()
            && self.control_flow.is_none()
            && !self.touches_floating_point
            && !self.touches_vector_registers
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InstructionCsrSemantics {
    pub address_operand: String,
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub reads: bool,
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub writes: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_register_operand: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct InstructionControlFlowSemantics {
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub jumps: bool,
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub branches: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_register_operand: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_fixed_register: Option<String>,
}

impl RegisterSemantics {
    pub fn plain_runtime_type_name(self) -> &'static str {
        self.bank.runtime_plain_type_name(self.encoding)
    }

    pub fn runtime_display_format(self) -> &'static str {
        self.bank.runtime_display_format(self.encoding)
    }

    pub fn runtime_type_label(self) -> &'static str {
        self.bank.runtime_type_label(self.encoding)
    }

    pub fn preferred_alias_type_name(self, has_constraints: bool) -> Option<&'static str> {
        match (self.role, self.encoding, has_constraints) {
            (RegisterRole::MemoryBase, RegisterEncoding::Standard, false) => {
                Some("BaseAddressRegister")
            }
            (RegisterRole::MemoryBase, RegisterEncoding::Compressed, false) => {
                Some("CompressedBaseAddressRegister")
            }
            (RegisterRole::SavedAlias, RegisterEncoding::Standard, _) => {
                Some("SavedIntegerRegister")
            }
            (RegisterRole::SavedAlias, RegisterEncoding::Compressed, _) => {
                Some("CompressedSavedIntegerRegister")
            }
            _ => None,
        }
    }
}

impl Instruction {
    pub fn normalized_signature(&self) -> Self {
        let mut normalized = self.clone();
        normalized.isa_bases.clear();
        normalized
    }

    pub fn requires_xlen_specific_variant(&self) -> bool {
        let has_xlen_dependent_operand = self.operands.iter().any(|operand| {
            matches!(
                operand.kind,
                OperandKind::SavedRegListWithStackAdj {
                    xlen_dependent: true,
                }
            )
        });
        let has_isa_dependent_restrictions = self.operands.iter().any(|operand| {
            !operand.per_isa_restrictions.is_empty()
                && operand.restrictions_for(ISABase::RV32) != operand.restrictions_for(ISABase::RV64)
        });
        has_xlen_dependent_operand || has_isa_dependent_restrictions
    }

    pub fn required_extensions_for(&self, isa_base: ISABase) -> &[ISAExtension] {
        self.required_extensions
            .get(&isa_base)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn is_memory_base_operand(&self, operand_name: &str) -> bool {
        self.memory_access
            .as_ref()
            .and_then(|memory_access| memory_access.base_register_operand.as_deref())
            == Some(operand_name)
    }

    pub fn is_memory_offset_operand(&self, operand_name: &str) -> bool {
        self.memory_access
            .as_ref()
            .and_then(|memory_access| memory_access.offset_operand.as_deref())
            == Some(operand_name)
    }

    pub fn memory_base_operand_name(&self) -> Option<&str> {
        self.memory_access
            .as_ref()
            .and_then(|memory_access| memory_access.base_register_operand.as_deref())
    }

    pub fn memory_base_fixed_register(&self) -> Option<&str> {
        self.memory_access
            .as_ref()
            .and_then(|memory_access| memory_access.base_register_fixed.as_deref())
    }

    pub fn writable_csr_operand_name(&self) -> Option<&str> {
        self.semantics
            .csr
            .as_ref()
            .filter(|csr| csr.writes)
            .map(|csr| csr.address_operand.as_str())
    }

    pub fn csr_source_register_operand_name(&self) -> Option<&str> {
        self.semantics
            .csr
            .as_ref()
            .filter(|csr| csr.writes)
            .and_then(|csr| csr.source_register_operand.as_deref())
    }

    pub fn pc_relative_target_operand_name(&self) -> Option<&str> {
        self.semantics
            .control_flow
            .as_ref()
            .and_then(|control_flow| control_flow.target_register_operand.as_deref())
    }

    pub fn pc_relative_target_fixed_register(&self) -> Option<&str> {
        self.semantics
            .control_flow
            .as_ref()
            .and_then(|control_flow| control_flow.target_fixed_register.as_deref())
    }

    pub fn sequence_semantics(&self) -> InstructionSequenceSemantics {
        let mut semantics = InstructionSequenceSemantics {
            memory_base_operand: self.memory_base_operand_name().map(str::to_string),
            memory_base_fixed: self.memory_base_fixed_register().map(str::to_string),
            touches_floating_point: self.uses_floating_point_registers(),
            ..InstructionSequenceSemantics::default()
        };

        if let Some(csr_operand) = self.writable_csr_operand_name() {
            semantics.writable_csr_operand = Some(csr_operand.to_string());
            semantics.csr_source_register_operand =
                self.csr_source_register_operand_name().map(str::to_string);
        }

        if let Some(target_operand) = self.pc_relative_target_operand_name() {
            semantics.pc_relative_target_operand = Some(target_operand.to_string());
        }
        if let Some(target_fixed) = self.pc_relative_target_fixed_register() {
            semantics.pc_relative_target_fixed = Some(target_fixed.to_string());
        }

        semantics
    }

    pub fn uses_floating_point_registers(&self) -> bool {
        if self.semantics.touches_floating_point {
            return true;
        }

        self.operands.iter().any(|operand| {
            matches!(
                operand.kind,
                OperandKind::Register(RegisterSpec {
                    bank: RegisterBank::FloatingPoint
                })
            )
        })
    }

    pub fn uses_vector_registers(&self) -> bool {
        if self.semantics.touches_vector_registers {
            return true;
        }

        self.operands.iter().any(|operand| {
            matches!(
                operand.kind,
                OperandKind::Register(RegisterSpec {
                    bank: RegisterBank::Vector
                })
            )
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssemblySyntax {
    Format(String),
    RustCode(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct MemoryAddressInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_register_operand: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_register_fixed: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_operand: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Description {
    String(String),
    Structured(Vec<DescriptionItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptionItem {
    pub id: Option<String>,
    pub normative: Option<bool>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamlInstruction {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub kind: String,
    pub name: String,
    pub long_name: String,
    pub description: Description,
    #[serde(rename = "definedBy")]
    pub defined_by: serde_yaml::Value,
    pub base: Option<u8>,
    pub assembly: String,
    #[serde(default)]
    pub format: Option<serde_yaml::Value>,
    pub encoding: Option<Encoding>,
    pub access: Option<Access>,
    #[serde(rename = "data_independent_timing")]
    pub data_independent_timing: Option<bool>,
    #[serde(rename = "operation()")]
    pub operation: Option<String>,
    #[serde(rename = "sail()")]
    pub sail: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub enum Encoding {
    Simple {
        #[serde(rename = "match")]
        match_pattern: String,
        #[serde(default)]
        variables: Option<Vec<Variable>>,
    },
    PerISA {
        #[serde(rename = "RV32")]
        rv32: Option<SimpleEncoding>,
        #[serde(rename = "RV64")]
        rv64: Option<SimpleEncoding>,
    },
}

impl<'de> Deserialize<'de> for Encoding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        use serde_yaml::Value;

        let value = Value::deserialize(deserializer)?;

        if let Value::Mapping(ref map) = value {
            if map.contains_key(Value::String("match".to_string())) {
                let match_pattern = map
                    .get(Value::String("match".to_string()))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| D::Error::custom("Missing or invalid 'match' field"))?
                    .to_string();

                let variables = map
                    .get(Value::String("variables".to_string()))
                    .map(|v| serde_yaml::from_value(v.clone()))
                    .transpose()
                    .map_err(|e| D::Error::custom(format!("Invalid variables: {}", e)))?;

                return Ok(Encoding::Simple {
                    match_pattern,
                    variables,
                });
            }

            if map.contains_key(Value::String("RV32".to_string()))
                || map.contains_key(Value::String("RV64".to_string()))
            {
                let rv32 = map
                    .get(Value::String("RV32".to_string()))
                    .map(|v| serde_yaml::from_value(v.clone()))
                    .transpose()
                    .map_err(|e| D::Error::custom(format!("Invalid RV32: {}", e)))?;

                let rv64 = map
                    .get(Value::String("RV64".to_string()))
                    .map(|v| serde_yaml::from_value(v.clone()))
                    .transpose()
                    .map_err(|e| D::Error::custom(format!("Invalid RV64: {}", e)))?;

                return Ok(Encoding::PerISA { rv32, rv64 });
            }
        }

        Err(D::Error::custom("Unable to determine encoding format"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleEncoding {
    #[serde(rename = "match")]
    pub match_pattern: String,
    #[serde(default)]
    pub variables: Option<Vec<Variable>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    #[serde(deserialize_with = "deserialize_location")]
    pub location: String,
    pub sign_extend: Option<bool>,
    pub left_shift: Option<u8>,
    pub alias: Option<String>,
    #[serde(rename = "not")]
    pub not_values: Option<serde_yaml::Value>,
}

fn deserialize_location<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    let value: serde_yaml::Value = serde::Deserialize::deserialize(deserializer)?;
    match value {
        serde_yaml::Value::String(s) => Ok(s),
        serde_yaml::Value::Number(n) => Ok(n.to_string()),
        _ => Err(D::Error::custom("location must be a string or number")),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Access {
    pub s: Option<String>,
    pub u: Option<String>,
    pub vs: Option<String>,
    pub vu: Option<String>,
}

pub fn save_instructions_to_json<P: AsRef<std::path::Path>>(
    instructions: &[Instruction],
    output_path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(instructions)?;
    std::fs::write(output_path, json)?;
    Ok(())
}

pub fn save_instructions_to_yaml<P: AsRef<std::path::Path>>(
    instructions: &[Instruction],
    output_path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    let yaml = serde_yaml::to_string(instructions)?;
    std::fs::write(output_path, yaml)?;
    Ok(())
}

fn is_destination_operand_name(name: &str) -> bool {
    matches!(
        name,
        "rd" | "rsd" | "xd" | "fd" | "vd" | "vd_rd" | "vd_rs1" | "vd_vs3"
    ) || name.starts_with("xd")
        || name.starts_with("fd")
        || name.starts_with("vd")
}

fn is_source_operand_name(name: &str) -> bool {
    name == "csr"
        || name == "vm"
        || name.starts_with("rs")
        || name.starts_with("xs")
        || name.starts_with("fs")
        || name.starts_with("vs")
}

#[cfg(test)]
mod tests {
    use super::{
        AssemblySyntax, CsrSpec, ISABase, ISAExtension, ImmediateKind, ImmediateSpec, Instruction,
        InstructionCsrSemantics, InstructionSemantics, InstructionSequenceSemantics,
        MemoryAddressInfo, Operand, OperandKind, OperandUsage, RegisterBank, RegisterEncoding,
        RegisterRole, RegisterSpec,
    };
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn operand_kind_json_deserializes_directly() {
        let operand: Operand = serde_json::from_str(
            r#"{
                "name": "xd",
                "kind": { "Register": { "bank": "Integer" } },
                "bit_lengths": { "RV32": 5, "RV64": 5 }
            }"#,
        )
        .expect("operand json should deserialize");

        assert!(matches!(
            operand.kind,
            OperandKind::Register(RegisterSpec {
                bank: RegisterBank::Integer
            })
        ));
    }

    #[test]
    fn normalized_signature_ignores_isa_bases() {
        let base_instruction = Instruction {
            name: "foo".to_string(),
            extension: ISAExtension::new("Zfoo"),
            isa_bases: vec![ISABase::RV32],
            required_extensions: BTreeMap::new(),
            operands: vec![Operand {
                name: "xd".to_string(),
                kind: OperandKind::Register(RegisterSpec {
                    bank: RegisterBank::Integer,
                }),
                bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
                restrictions: None,
                per_isa_restrictions: HashMap::new(),
            }],
            assembly_syntax: AssemblySyntax::Format("foo {xd}".to_string()),
            memory_access: None,
            semantics: InstructionSemantics::default(),
        };

        let rv32 = base_instruction.clone();
        let rv64 = Instruction {
            isa_bases: vec![ISABase::RV64],
            ..base_instruction
        };

        assert_eq!(rv32.normalized_signature(), rv64.normalized_signature());
    }

    #[test]
    fn usage_and_runtime_immediate_kind_come_from_unified_instruction_model() {
        let instruction = Instruction {
            name: "lw".to_string(),
            extension: ISAExtension::new("I"),
            isa_bases: vec![ISABase::RV32, ISABase::RV64],
            required_extensions: BTreeMap::new(),
            operands: vec![
                Operand {
                    name: "xd".to_string(),
                    kind: OperandKind::Register(RegisterSpec {
                        bank: RegisterBank::Integer,
                    }),
                    bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
                    restrictions: None,
                per_isa_restrictions: HashMap::new(),
                },
                Operand {
                    name: "xs1".to_string(),
                    kind: OperandKind::Register(RegisterSpec {
                        bank: RegisterBank::Integer,
                    }),
                    bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
                    restrictions: None,
                per_isa_restrictions: HashMap::new(),
                },
                Operand {
                    name: "imm".to_string(),
                    kind: OperandKind::Immediate(ImmediateSpec {
                        kind: ImmediateKind::Signed,
                    }),
                    bit_lengths: HashMap::from([(ISABase::RV32, 12), (ISABase::RV64, 12)]),
                    restrictions: None,
                per_isa_restrictions: HashMap::new(),
                },
            ],
            assembly_syntax: AssemblySyntax::Format("lw {xd}, {imm}({xs1})".to_string()),
            memory_access: Some(MemoryAddressInfo {
                base_register_operand: Some("xs1".to_string()),
                base_register_fixed: None,
                offset_operand: Some("imm".to_string()),
            }),
            semantics: InstructionSemantics::default(),
        };

        assert_eq!(
            instruction.operands[0].usage(&instruction),
            OperandUsage::Destination
        );
        assert_eq!(
            instruction.operands[1].usage(&instruction),
            OperandUsage::MemoryBase
        );
        assert_eq!(
            instruction.operands[2].usage(&instruction),
            OperandUsage::MemoryOffset
        );
        assert_eq!(
            instruction.operands[2].runtime_immediate_kind(&instruction),
            Some(ImmediateKind::Signed)
        );
    }

    #[test]
    fn register_semantics_capture_encoding_and_role() {
        let instruction = Instruction {
            name: "c.sw".to_string(),
            extension: ISAExtension::new("C"),
            isa_bases: vec![ISABase::RV32, ISABase::RV64],
            required_extensions: BTreeMap::new(),
            operands: vec![
                Operand {
                    name: "xs1".to_string(),
                    kind: OperandKind::Register(RegisterSpec {
                        bank: RegisterBank::Integer,
                    }),
                    bit_lengths: HashMap::from([(ISABase::RV32, 3), (ISABase::RV64, 3)]),
                    restrictions: None,
                per_isa_restrictions: HashMap::new(),
                },
                Operand {
                    name: "r1s".to_string(),
                    kind: OperandKind::Register(RegisterSpec {
                        bank: RegisterBank::Integer,
                    }),
                    bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
                    restrictions: None,
                per_isa_restrictions: HashMap::new(),
                },
            ],
            assembly_syntax: AssemblySyntax::Format("c.sw {r1s}, 0({xs1})".to_string()),
            memory_access: Some(MemoryAddressInfo {
                base_register_operand: Some("xs1".to_string()),
                base_register_fixed: None,
                offset_operand: None,
            }),
            semantics: InstructionSemantics::default(),
        };

        let base_semantics = instruction.operands[0]
            .register_semantics(ISABase::RV32, &instruction)
            .expect("xs1 should be a register operand");
        assert_eq!(base_semantics.bank, RegisterBank::Integer);
        assert_eq!(base_semantics.encoding, RegisterEncoding::Compressed);
        assert_eq!(base_semantics.role, RegisterRole::MemoryBase);

        let saved_semantics = instruction.operands[1]
            .register_semantics(ISABase::RV32, &instruction)
            .expect("r1s should be a register operand");
        assert_eq!(saved_semantics.bank, RegisterBank::Integer);
        assert_eq!(saved_semantics.encoding, RegisterEncoding::Standard);
        assert_eq!(saved_semantics.role, RegisterRole::SavedAlias);
        assert_eq!(
            base_semantics.preferred_alias_type_name(false),
            Some("CompressedBaseAddressRegister")
        );
        assert_eq!(
            saved_semantics.preferred_alias_type_name(false),
            Some("SavedIntegerRegister")
        );
        assert_eq!(saved_semantics.plain_runtime_type_name(), "IntegerRegister");
    }

    #[test]
    fn xlen_dependent_aggregate_operands_force_split() {
        let instruction = Instruction {
            name: "cm.push".to_string(),
            extension: ISAExtension::new("Zcmp"),
            isa_bases: vec![ISABase::RV32, ISABase::RV64],
            required_extensions: BTreeMap::new(),
            operands: vec![Operand {
                name: "reg_list".to_string(),
                kind: OperandKind::SavedRegListWithStackAdj {
                    xlen_dependent: true,
                },
                bit_lengths: HashMap::from([(ISABase::RV32, 4), (ISABase::RV64, 4)]),
                restrictions: None,
                per_isa_restrictions: HashMap::new(),
            }],
            assembly_syntax: AssemblySyntax::Format("cm.push {reg_list}".to_string()),
            memory_access: None,
            semantics: InstructionSemantics::default(),
        };

        assert!(instruction.requires_xlen_specific_variant());
    }

    #[test]
    fn operand_helpers_capture_codegen_relevant_semantics() {
        let instruction = Instruction {
            name: "c.li".to_string(),
            extension: ISAExtension::new("C"),
            isa_bases: vec![ISABase::RV32, ISABase::RV64],
            required_extensions: BTreeMap::new(),
            operands: vec![Operand {
                name: "uimm".to_string(),
                kind: OperandKind::Immediate(ImmediateSpec {
                    kind: ImmediateKind::Unsigned,
                }),
                bit_lengths: HashMap::from([(ISABase::RV32, 1), (ISABase::RV64, 1)]),
                restrictions: None,
                per_isa_restrictions: HashMap::new(),
            }],
            assembly_syntax: AssemblySyntax::Format("c.li {uimm}".to_string()),
            memory_access: None,
            semantics: InstructionSemantics::default(),
        };

        let operand = &instruction.operands[0];
        assert!(operand.supports_numeric_mutation(&instruction));
        assert_eq!(
            operand.runtime_immediate_kind(&instruction),
            Some(ImmediateKind::Unsigned)
        );
        assert!(operand.uses_inline_boolean_value(&instruction));
        assert_eq!(
            ImmediateKind::Unsigned.runtime_base_type_name(),
            "UImmediate"
        );
        assert_eq!(ImmediateKind::LinearJalr.fixed_value(), Some(0));
        assert_eq!(
            ImmediateKind::VectorSigned.sampler_function_name(),
            Some("sample_vector_signed_immediate")
        );
    }

    #[test]
    fn vector_family_generic_immediates_resolve_to_vector_runtime_kind() {
        let instruction = Instruction {
            name: "vadd.vi".to_string(),
            extension: ISAExtension::new("V"),
            isa_bases: vec![ISABase::RV32, ISABase::RV64],
            required_extensions: BTreeMap::new(),
            operands: vec![Operand {
                name: "imm".to_string(),
                kind: OperandKind::Immediate(ImmediateSpec {
                    kind: ImmediateKind::Unsigned,
                }),
                bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
                restrictions: None,
                per_isa_restrictions: HashMap::new(),
            }],
            assembly_syntax: AssemblySyntax::Format("vadd.vi {imm}".to_string()),
            memory_access: None,
            semantics: InstructionSemantics {
                touches_vector_registers: true,
                ..InstructionSemantics::default()
            },
        };

        let operand = &instruction.operands[0];
        assert_eq!(
            operand.runtime_immediate_kind(&instruction),
            Some(ImmediateKind::VectorUnsigned)
        );
        assert!(operand.supports_numeric_mutation(&instruction));
    }

    #[test]
    fn sequence_semantics_capture_sequence_generation_hints() {
        let instruction = Instruction {
            name: "csrrw".to_string(),
            extension: ISAExtension::new("Zicsr"),
            isa_bases: vec![ISABase::RV32, ISABase::RV64],
            required_extensions: BTreeMap::new(),
            operands: vec![
                Operand {
                    name: "xd".to_string(),
                    kind: OperandKind::Register(RegisterSpec {
                        bank: RegisterBank::Integer,
                    }),
                    bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
                    restrictions: None,
                per_isa_restrictions: HashMap::new(),
                },
                Operand {
                    name: "csr".to_string(),
                    kind: OperandKind::Csr(CsrSpec),
                    bit_lengths: HashMap::from([(ISABase::RV32, 12), (ISABase::RV64, 12)]),
                    restrictions: None,
                per_isa_restrictions: HashMap::new(),
                },
                Operand {
                    name: "xs1".to_string(),
                    kind: OperandKind::Register(RegisterSpec {
                        bank: RegisterBank::Integer,
                    }),
                    bit_lengths: HashMap::from([(ISABase::RV32, 5), (ISABase::RV64, 5)]),
                    restrictions: None,
                per_isa_restrictions: HashMap::new(),
                },
            ],
            assembly_syntax: AssemblySyntax::Format("csrrw {xd}, {csr}, {xs1}".to_string()),
            memory_access: None,
            semantics: InstructionSemantics {
                csr: Some(InstructionCsrSemantics {
                    address_operand: "csr".to_string(),
                    reads: true,
                    writes: true,
                    source_register_operand: Some("xs1".to_string()),
                }),
                ..InstructionSemantics::default()
            },
        };

        assert_eq!(instruction.writable_csr_operand_name(), Some("csr"));
        assert_eq!(instruction.csr_source_register_operand_name(), Some("xs1"));
        assert_eq!(instruction.memory_base_operand_name(), None);
        assert_eq!(instruction.pc_relative_target_operand_name(), None);

        assert_eq!(
            instruction.sequence_semantics(),
            InstructionSequenceSemantics {
                writable_csr_operand: Some("csr".to_string()),
                csr_source_register_operand: Some("xs1".to_string()),
                ..InstructionSequenceSemantics::default()
            }
        );
    }
}
