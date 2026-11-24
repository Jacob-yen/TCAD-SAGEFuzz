use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum XLen {
    X32,
    X64,
}

impl XLen {
    pub const fn bits(self) -> u8 {
        match self {
            Self::X32 => 32,
            Self::X64 => 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FLen {
    F32,
    F64,
    F128,
}

impl FLen {
    pub const fn bits(self) -> u8 {
        match self {
            Self::F32 => 32,
            Self::F64 => 64,
            Self::F128 => 128,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IsaProfile {
    pub xlen: XLen,
    pub flen: Option<FLen>,
}

impl IsaProfile {
    pub const fn new(xlen: XLen, flen: Option<FLen>) -> Self {
        Self { xlen, flen }
    }

    pub const fn rv32() -> Self {
        Self::new(XLen::X32, None)
    }

    pub const fn rv64() -> Self {
        Self::new(XLen::X64, None)
    }

    pub const fn xlen_bits(self) -> u8 {
        self.xlen.bits()
    }

    pub const fn supports_flen(self, required: FLen) -> bool {
        match self.flen {
            Some(flen) => flen.bits() >= required.bits(),
            None => false,
        }
    }
}

/// Constrains which architectural register numbers may appear in operand
/// positions of the main instruction.
///
/// Read this in terms of a full instruction, not an abstract operand name.
/// For example:
/// - in `add x10, x11, x12`, these settings constrain `x10`, `x11`, and `x12`
/// - in `fadd.s f0, f1, f2`, they constrain `f0`, `f1`, and `f2`
/// - in `vadd.vv v8, v9, v10`, they constrain `v8`, `v9`, and `v10`
///
/// They also affect helper-selected integer registers that are chosen through
/// the normal `IntegerRegister` random path, for example the `rd` in an
/// auto-appended `csrrs x5, fflags, x0` post-instruction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterOperandConfig {
    /// Inclusive range of integer register numbers allowed in integer register
    /// operand slots.
    ///
    /// Example: in `add x10, x11, x12`, this range controls the numbers that
    /// may appear in place of `x10`, `x11`, and `x12`. In `lw x10, 16(x8)`, it
    /// controls the destination register `x10` but not the memory-base register
    /// `x8` when that base uses a memory-specific register wrapper.
    pub integer_register_number_range: (u8, u8),
    /// Inclusive range of floating-point register numbers allowed in
    /// floating-point register operand slots.
    ///
    /// Example: in `fadd.s f0, f1, f2`, this range controls the numbers that
    /// may appear in the `f0`, `f1`, and `f2` positions.
    pub floating_point_register_number_range: (u8, u8),
    /// Inclusive range of vector register numbers allowed in vector register
    /// operand slots.
    ///
    /// Example: in `vadd.vv v8, v9, v10`, this range controls the numbers that
    /// may appear in the `v8`, `v9`, and `v10` positions.
    pub vector_register_number_range: (u8, u8),
}

impl Default for RegisterOperandConfig {
    fn default() -> Self {
        Self {
            integer_register_number_range: (0, 31),
            floating_point_register_number_range: (0, 31),
            vector_register_number_range: (0, 31),
        }
    }
}

/// Controls the memory-specific parts of random generation.
///
/// Read this against a full memory instruction such as `lw x10, 16(x8)`:
/// - `x8` is the memory base register
/// - `16` is the memory offset immediate
/// - before the `lw`, sequence generation may inject `li x8, <addr>` to make
///   the base register usable
///
/// The three fields below map directly to those three concerns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccessConfig {
    /// Inclusive range of base address values that may be loaded into the
    /// memory base register during sequence setup.
    ///
    /// Example: if the main instruction is `lw x10, 16(x8)`, sequence
    /// generation may inject `li x8, 0x8000_0040` first. This field controls
    /// the allowed range for that injected address value.
    pub memory_base_address_range: (u64, u64),
    /// Inclusive range of memory offset immediates used by memory-aware
    /// immediate operand types.
    ///
    /// Example: in `lw x10, 16(x8)`, this field controls the `16`. In
    /// `sw x9, -32(x18)`, it controls the `-32`. It does not automatically
    /// constrain unrelated ALU immediates such as the `123` in
    /// `addi x10, x11, 123`.
    pub memory_offset_immediate_range: (i64, i64),
    /// Inclusive range of integer register numbers allowed in the memory base
    /// register position.
    ///
    /// Example: in `lw x10, 16(x8)`, this field controls which `xN` may appear
    /// in the `x8` position when the operand type is a memory-base register
    /// wrapper such as `BaseAddressRegister`.
    pub memory_base_register_number_range: (u8, u8),
}

impl Default for MemoryAccessConfig {
    fn default() -> Self {
        Self {
            memory_base_address_range: (0x10000000, 0x20000000), // 默认内存地址范围
            memory_offset_immediate_range: (-1000, 1000),        // 默认访存偏移范围
            memory_base_register_number_range: (5, 7), // 默认聚焦 t0-t2 作为访存基址寄存器
        }
    }
}

/// Filters which CSR operands may be chosen during random generation.
///
/// Read this against a full CSR instruction such as `csrrw x10, mstatus, x11`.
/// These toggles decide which CSR names may appear in the `mstatus` position.
/// They do not control the separate floating-point status readback path; that
/// path is handled by `RandomConfig::append_fflags_readback`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CsrSelectionConfig {
    /// Allow machine-level CSR operands in the CSR slot of instructions such
    /// as `csrrw x10, mstatus, x11`.
    ///
    /// When this is enabled, names such as `mstatus`, `misa`, `mtvec`,
    /// `mepc`, `mcause`, `cycle`, and `time` may be chosen in place of
    /// `mstatus`.
    #[serde(default = "CsrSelectionConfig::default_machine_csr_operands_enabled")]
    pub machine_csr_operands_enabled: bool,
    /// Allow supervisor-level CSR operands in the CSR slot of instructions
    /// such as `csrrw x10, satp, x11`.
    ///
    /// Typical examples include `satp`, `sstatus`, `sie`, `sip`, `stvec`, and
    /// `sepc`.
    #[serde(default = "CsrSelectionConfig::default_supervisor_csr_operands_enabled")]
    pub supervisor_csr_operands_enabled: bool,
    /// Allow hypervisor HS-level CSR operands in the CSR slot of instructions
    /// such as `csrrw x10, hstatus, x11`.
    ///
    /// Typical examples include `hstatus`, `hedeleg`, `hideleg`, `hgatp`, and
    /// `henvcfg`.
    #[serde(default = "CsrSelectionConfig::default_hypervisor_csr_operands_enabled")]
    pub hypervisor_csr_operands_enabled: bool,
    /// Allow virtual supervisor CSR operands in the CSR slot of instructions
    /// such as `csrrw x10, vsstatus, x11`.
    ///
    /// Typical examples include `vsstatus`, `vsie`, `vsepc`, and `vsatp`.
    #[serde(default = "CsrSelectionConfig::default_virtual_supervisor_csr_operands_enabled")]
    pub virtual_supervisor_csr_operands_enabled: bool,
    /// Allow RNMI / Smrnmi CSR operands in the CSR slot of instructions such
    /// as `csrrw x10, mnstatus, x11`.
    ///
    /// Typical examples include `mnstatus`, `mnscratch`, `mnepc`, and
    /// `mncause`.
    #[serde(default = "CsrSelectionConfig::default_rnmi_csr_operands_enabled")]
    pub rnmi_csr_operands_enabled: bool,
    /// Allow floating-point CSR operands in the CSR slot of instructions such
    /// as `csrrw x10, fflags, x11`.
    ///
    /// Typical examples include `fflags`, `frm`, and `fcsr`.
    #[serde(default = "CsrSelectionConfig::default_floating_csr_operands_enabled")]
    pub floating_csr_operands_enabled: bool,
    /// Allow vector CSR operands in the CSR slot of instructions such as
    /// `csrrw x10, vtype, x11`.
    ///
    /// Typical examples include `vtype`, `vl`, `vstart`, `vxrm`, `vxsat`, and
    /// `vlenb`.
    #[serde(default = "CsrSelectionConfig::default_vector_csr_operands_enabled")]
    pub vector_csr_operands_enabled: bool,
    /// Allow debug CSR operands in the CSR slot of instructions such as
    /// `csrrw x10, dcsr, x11`.
    ///
    /// Typical examples include `dcsr`, `dpc`, `dscratch0`, and `dscratch1`.
    #[serde(default = "CsrSelectionConfig::default_debug_csr_operands_enabled")]
    pub debug_csr_operands_enabled: bool,
}

impl CsrSelectionConfig {
    const fn default_machine_csr_operands_enabled() -> bool {
        false
    }

    const fn default_supervisor_csr_operands_enabled() -> bool {
        false
    }

    const fn default_hypervisor_csr_operands_enabled() -> bool {
        false
    }

    const fn default_virtual_supervisor_csr_operands_enabled() -> bool {
        false
    }

    const fn default_rnmi_csr_operands_enabled() -> bool {
        false
    }

    const fn default_floating_csr_operands_enabled() -> bool {
        false
    }

    const fn default_vector_csr_operands_enabled() -> bool {
        false
    }

    const fn default_debug_csr_operands_enabled() -> bool {
        false
    }
}

impl Default for CsrSelectionConfig {
    fn default() -> Self {
        Self {
            machine_csr_operands_enabled: Self::default_machine_csr_operands_enabled(),
            supervisor_csr_operands_enabled: Self::default_supervisor_csr_operands_enabled(),
            hypervisor_csr_operands_enabled: Self::default_hypervisor_csr_operands_enabled(),
            virtual_supervisor_csr_operands_enabled:
                Self::default_virtual_supervisor_csr_operands_enabled(),
            rnmi_csr_operands_enabled: Self::default_rnmi_csr_operands_enabled(),
            floating_csr_operands_enabled: Self::default_floating_csr_operands_enabled(),
            vector_csr_operands_enabled: Self::default_vector_csr_operands_enabled(),
            debug_csr_operands_enabled: Self::default_debug_csr_operands_enabled(),
        }
    }
}

/// Top-level configuration threaded through random operand generation and
/// random instruction-sequence generation.
///
/// Read this against a complete generated sequence. For example, if the final
/// sequence is:
/// - `li x8, 0x8000_0040`
/// - `lw x10, 16(x8)`
///
/// then different `RandomConfig` fields separately influence:
/// - which register number appears as `x10`
/// - which register number appears as `x8`
/// - which address is loaded by `li x8, ...`
/// - which offset appears as `16`
///
/// Likewise, if the main instruction is `fadd.s f0, f1, f2`, the config may
/// also control whether a post-instruction like `csrrs x5, fflags, x0` is
/// appended.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomConfig {
    /// Active ISA profile used when random generation must respect XLEN/FLEN.
    #[serde(default = "RandomConfig::default_isa_profile")]
    pub isa_profile: IsaProfile,
    /// Configuration for memory instructions and memory-aware operands.
    ///
    /// Example using `lw x10, 16(x8)`:
    /// - `memory_base_address_range` controls the `<addr>` in an injected
    ///   setup like `li x8, <addr>`
    /// - `memory_offset_immediate_range` controls the `16`
    /// - `memory_base_register_number_range` controls which `xN` may appear in
    ///   the base-register position instead of `x8`
    #[serde(default)]
    pub memory_access_config: MemoryAccessConfig,
    /// Configuration for register-number selection in regular operand slots of
    /// the main instruction.
    ///
    /// Example:
    /// - in `add x10, x11, x12`, this controls `x10/x11/x12`
    /// - in `fadd.s f0, f1, f2`, this controls `f0/f1/f2`
    /// - in `vadd.vv v8, v9, v10`, this controls `v8/v9/v10`
    #[serde(default)]
    pub register_operand_config: RegisterOperandConfig,
    /// Whether random sequence generation should append a post-instruction that
    /// reads back `fflags` after a floating-point instruction.
    ///
    /// Example: when enabled, a generated sequence containing
    /// `fadd.s f0, f1, f2` may become:
    /// - `fadd.s f0, f1, f2`
    /// - `csrrs x5, fflags, x0`
    #[serde(default = "RandomConfig::default_append_fflags_readback")]
    pub append_fflags_readback: bool,
    /// Configuration for which CSR names may appear in CSR operand positions.
    ///
    /// Example: in `csrrw x10, mstatus, x11`, this controls whether `mstatus`
    /// may be chosen in the CSR slot at all.
    #[serde(default)]
    pub csr_selection_config: CsrSelectionConfig,
    /// Probability that eligible operand samplers prefer edge-case values
    /// instead of ordinary in-range values.
    ///
    /// Example effects:
    /// - `lw x10, 0(x8)` is favored over a random large offset
    /// - `addi x10, x11, -2048` or `addi x10, x11, 2047` is favored over a
    ///   middle-of-range immediate
    /// - floating-point generators may prefer `+0`, `-0`, `inf`, or `nan`
    #[serde(default = "RandomConfig::default_edge_case_probability")]
    pub edge_case_probability: f64,
}

impl Default for RandomConfig {
    fn default() -> Self {
        Self {
            isa_profile: Self::default_isa_profile(),
            memory_access_config: MemoryAccessConfig::default(),
            register_operand_config: RegisterOperandConfig::default(),
            append_fflags_readback: Self::default_append_fflags_readback(),
            csr_selection_config: CsrSelectionConfig::default(),
            edge_case_probability: 0.2, // 默认20%概率偏向边界/特殊值
        }
    }
}

impl RandomConfig {
    pub const fn default_isa_profile() -> IsaProfile {
        IsaProfile::new(XLen::X64, Some(FLen::F64))
    }

    const fn default_edge_case_probability() -> f64 {
        0.2
    }

    const fn default_append_fflags_readback() -> bool {
        false
    }

    pub const fn xlen_bits(&self) -> u8 {
        self.isa_profile.xlen_bits()
    }

    pub const fn is_rv32(&self) -> bool {
        matches!(self.isa_profile.xlen, XLen::X32)
    }

    pub const fn is_rv64(&self) -> bool {
        matches!(self.isa_profile.xlen, XLen::X64)
    }

    pub const fn supports_float_width(&self, width: u8) -> bool {
        match width {
            16 | 32 => self.isa_profile.supports_flen(FLen::F32),
            64 => self.isa_profile.supports_flen(FLen::F64),
            128 => self.isa_profile.supports_flen(FLen::F128),
            _ => false,
        }
    }
}
