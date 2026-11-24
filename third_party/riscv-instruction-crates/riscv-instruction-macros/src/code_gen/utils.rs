use super::CodeGenerator;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use riscv_instruction_schema::{
    ISABase, ISAExtension, ImmediateKind, Instruction, Operand, OperandKind, OperandRestriction,
    RegisterBank, RegisterPairSpec, RegisterRole, RegisterSemantics,
};
use syn::Ident;

impl CodeGenerator {
    /// Generate the import block.
    pub fn generate_imports(&self) -> TokenStream {
        quote! {
            pub use riscv_instruction_types::*;
            use riscv_instruction_macros::{DeriveValidatedValue, DeriveAsmEncode, DeriveInstructionDisplay, DeriveRandom};
            use serde::{Deserialize, Serialize};
            use enum_iterator::Sequence;
        }
    }

    pub fn instruction_name_to_variant(&self, name: &str) -> String {
        name.replace('.', "_").replace('-', "_").to_uppercase()
    }

    fn sanitize_identifier_fragment(&self, value: &str) -> String {
        const RUST_KEYWORDS: &[&str] = &[
            "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn",
            "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
            "return", "self", "Self", "static", "struct", "super", "trait", "true", "type",
            "unsafe", "use", "where", "while", "async", "await", "dyn", "abstract", "become",
            "box", "do", "final", "macro", "override", "priv", "typeof", "unsized", "virtual",
            "yield", "try",
        ];

        let mut out = String::new();

        for (index, ch) in value.chars().enumerate() {
            let is_valid_start = ch == '_' || ch.is_ascii_alphabetic();
            let is_valid_continue = ch == '_' || ch.is_ascii_alphanumeric();

            if index == 0 {
                if is_valid_start {
                    out.push(ch);
                } else if ch.is_ascii_digit() {
                    out.push('_');
                    out.push(ch);
                } else {
                    out.push_str(&format!("_x{:X}_", ch as u32));
                }
            } else if is_valid_continue {
                out.push(ch);
            } else {
                out.push_str(&format!("_x{:X}_", ch as u32));
            }
        }

        if out.is_empty() {
            out.push('_');
        }

        if RUST_KEYWORDS.contains(&out.as_str()) {
            out.push('_');
        }

        out
    }

    pub fn extension_identifier_fragment(&self, extension: &ISAExtension) -> String {
        self.sanitize_identifier_fragment(extension.as_str())
    }

    pub fn extension_variant_ident(&self, extension: &ISAExtension) -> Ident {
        Ident::new(
            &self.extension_identifier_fragment(extension),
            Span::call_site(),
        )
    }

    fn operand_has_explicit_constraints(&self, restrictions: Option<&OperandRestriction>) -> bool {
        restrictions.is_some_and(OperandRestriction::has_explicit_constraints)
    }

    fn register_runtime_type_name(
        &self,
        operand: &Operand,
        semantics: RegisterSemantics,
        restriction: Option<&OperandRestriction>,
    ) -> String {
        let has_constraints = self.operand_has_explicit_constraints(restriction);

        if let Some(alias_type) = semantics.preferred_alias_type_name(has_constraints) {
            return alias_type.to_string();
        }

        let base_type = semantics.plain_runtime_type_name();
        if let Some(restriction) = restriction.filter(|r| r.has_explicit_constraints()) {
            let use_memory_register_range = semantics.role == RegisterRole::MemoryBase;
            return self.generate_restricted_register_type_name(
                base_type,
                &operand.name,
                restriction,
                use_memory_register_range,
            );
        }

        base_type.to_string()
    }

    fn restricted_immediate_type_name(
        &self,
        immediate_kind: ImmediateKind,
        operand_name: &str,
        bit_length: u8,
        restriction: Option<&OperandRestriction>,
        is_memory_aware: bool,
    ) -> Option<String> {
        let restriction = restriction.filter(|r| r.has_explicit_constraints())?;
        Some(format!(
            "{}< {} >",
            self.generate_restricted_immediate_type_name(
                immediate_kind,
                operand_name,
                bit_length,
                restriction,
            ),
            is_memory_aware
        ))
    }

    fn immediate_runtime_type_name(
        &self,
        immediate_kind: ImmediateKind,
        operand_name: &str,
        bit_length: u8,
        restriction: Option<&OperandRestriction>,
        is_memory_aware: bool,
    ) -> String {
        if !immediate_kind.is_restricted_runtime_family() {
            return immediate_kind.runtime_base_type_name().to_string();
        }

        if immediate_kind.supports_inline_boolean_value() && bit_length == 1 {
            return "bool".to_string();
        }

        if let Some(restricted) = self.restricted_immediate_type_name(
            immediate_kind,
            operand_name,
            bit_length,
            restriction,
            is_memory_aware,
        ) {
            return restricted;
        }

        format!(
            "{}<{}, {}>",
            immediate_kind.runtime_base_type_name(),
            bit_length,
            is_memory_aware
        )
    }

    pub fn operand_to_typed_struct(
        &self,
        operand: &Operand,
        isa_base: &ISABase,
        instruction: &riscv_instruction_schema::Instruction,
    ) -> String {
        let bit_length = operand.bit_length_for(*isa_base);
        let restrictions = operand.restrictions_for(*isa_base);
        let is_memory_aware = instruction.is_memory_offset_operand(&operand.name);
        if let Some(register_semantics) = operand.register_semantics(*isa_base, instruction) {
            return self.register_runtime_type_name(operand, register_semantics, restrictions);
        }

        if let Some(immediate_kind) = operand.runtime_immediate_kind(instruction) {
            return self.immediate_runtime_type_name(
                immediate_kind,
                &operand.name,
                bit_length,
                restrictions,
                is_memory_aware,
            );
        }

        match &operand.kind {
            OperandKind::Unknown => panic!("operand '{}' still has unknown kind", operand.name),
            OperandKind::Register(_) => {
                unreachable!("register operands should have been handled by register_semantics")
            }
            OperandKind::Immediate(_) => {
                unreachable!(
                    "immediate operands should have been handled by runtime_immediate_kind"
                )
            }
            OperandKind::Csr(_) => self.map_csr_runtime_type(instruction, operand),
            OperandKind::RoundMode => "RoundingMode".to_string(),
            OperandKind::FenceMode => "FenceMode".to_string(),
            OperandKind::FliConstant => "FliConstant".to_string(),
            OperandKind::SavedRegListWithStackAdj { .. } => match isa_base {
                ISABase::RV32 => "SavedRegListWithStackAdjRv32".to_string(),
                ISABase::RV64 => "SavedRegListWithStackAdjRv64".to_string(),
                // ISABase::RV128 => "SavedRegListWithStackAdjRv128".to_string(), // Placeholder for future RV128 support
            },
            OperandKind::RegisterPair(RegisterPairSpec {
                bank: RegisterBank::Integer,
                distinct: true,
            }) => "NotEqualCompressedSavedIntegerRegisterPair".to_string(),
            OperandKind::RegisterPair(_) => {
                unreachable!("unsupported register pair operand '{}'", operand.name)
            }
        }
    }

    fn map_csr_runtime_type(&self, instruction: &Instruction, operand: &Operand) -> String {
        if operand.name.to_ascii_lowercase() != "csr" {
            return "CSRAddress".to_string();
        }

        if instruction.writable_csr_operand_name() == Some(operand.name.as_str()) {
            "WritableCsr".to_string()
        } else {
            "ReadableCsr".to_string()
        }
    }

    pub fn generate_wrapper_enum_offset_impl(
        &self,
        enum_name: &Ident,
        variant_idents: &[Ident],
    ) -> TokenStream {
        if variant_idents.is_empty() {
            return quote! {
                impl MemoryAccessInstruction for #enum_name {
                    fn offset_operand_value(&self) -> Option<i64> {
                        None
                    }
                }
            };
        }

        let match_arms = variant_idents
            .iter()
            .map(|variant_ident| {
                quote! {
                    Self::#variant_ident(inner) => MemoryAccessInstruction::offset_operand_value(inner)
                }
            })
            .collect::<Vec<_>>();

        quote! {
            impl MemoryAccessInstruction for #enum_name {
                fn offset_operand_value(&self) -> Option<i64> {
                    match self {
                        #(#match_arms),*
                    }
                }
            }
        }
    }

    pub fn generate_wrapper_enum_numeric_mutation_impl(
        &self,
        enum_name: &Ident,
        variant_idents: &[Ident],
    ) -> TokenStream {
        if variant_idents.is_empty() {
            return quote! {
                impl NumericOperandMutation for #enum_name {
                    fn mutate_numeric_operands<R: rand::Rng>(
                        &mut self,
                        _rng: &mut R,
                        _config: &RandomConfig,
                    ) -> bool {
                        false
                    }
                }
            };
        }

        let match_arms = variant_idents
            .iter()
            .map(|variant_ident| {
                quote! {
                    Self::#variant_ident(inner) => {
                        NumericOperandMutation::mutate_numeric_operands(inner, rng, config)
                    }
                }
            })
            .collect::<Vec<_>>();

        quote! {
            impl NumericOperandMutation for #enum_name {
                fn mutate_numeric_operands<R: rand::Rng>(
                    &mut self,
                    rng: &mut R,
                    config: &RandomConfig,
                ) -> bool {
                    match self {
                        #(#match_arms),*
                    }
                }
            }
        }
    }

    pub fn generate_wrapper_enum_runtime_operand_access_impl(
        &self,
        enum_name: &Ident,
        variant_idents: &[Ident],
    ) -> TokenStream {
        if variant_idents.is_empty() {
            return quote! {
                impl RuntimeOperandAccess for #enum_name {
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
                }
            };
        }

        let register_arms = variant_idents
            .iter()
            .map(|variant_ident| {
                quote! {
                    Self::#variant_ident(inner) => RuntimeOperandAccess::register_operands(inner)
                }
            })
            .collect::<Vec<_>>();

        let memory_arms = variant_idents
            .iter()
            .map(|variant_ident| {
                quote! {
                    Self::#variant_ident(inner) => RuntimeOperandAccess::memory_operands(inner)
                }
            })
            .collect::<Vec<_>>();

        let set_register_arms = variant_idents
            .iter()
            .map(|variant_ident| {
                quote! {
                    Self::#variant_ident(inner) => {
                        RuntimeOperandAccess::set_register_operand(inner, operand_name, register)
                    }
                }
            })
            .collect::<Vec<_>>();

        let set_memory_arms = variant_idents
            .iter()
            .map(|variant_ident| {
                quote! {
                    Self::#variant_ident(inner) => {
                        RuntimeOperandAccess::set_memory_operand(inner, index, patch)
                    }
                }
            })
            .collect::<Vec<_>>();

        quote! {
            impl RuntimeOperandAccess for #enum_name {
                fn register_operands(&self) -> Vec<RuntimeRegisterOperand> {
                    match self {
                        #(#register_arms),*
                    }
                }

                fn memory_operands(&self) -> Vec<RuntimeMemoryOperand> {
                    match self {
                        #(#memory_arms),*
                    }
                }

                fn set_register_operand(
                    &mut self,
                    operand_name: &str,
                    register: RuntimeRegisterAssignment,
                ) -> Result<bool, OperandAccessError> {
                    match self {
                        #(#set_register_arms),*
                    }
                }

                fn set_memory_operand(
                    &mut self,
                    index: usize,
                    patch: RuntimeMemoryOperandPatch,
                ) -> Result<bool, OperandAccessError> {
                    match self {
                        #(#set_memory_arms),*
                    }
                }
            }
        }
    }

    pub fn generate_separated_instruction_enum_offset_impl(
        &self,
        enum_name: &Ident,
        instructions: &[&Instruction],
    ) -> TokenStream {
        if instructions.is_empty() {
            return quote! {
                impl MemoryAccessInstruction for #enum_name {
                    fn offset_operand_value(&self) -> Option<i64> {
                        None
                    }
                }
            };
        }

        let match_arms = instructions
            .iter()
            .map(|inst| {
                let variant_ident = Ident::new(
                    &self.instruction_name_to_variant(&inst.name),
                    Span::call_site(),
                );

                if inst.memory_access.is_some() {
                    quote! {
                        Self::#variant_ident(inner) => MemoryAccessInstruction::offset_operand_value(inner)
                    }
                } else {
                    quote! {
                        Self::#variant_ident(_) => None
                    }
                }
            })
            .collect::<Vec<_>>();

        quote! {
            impl MemoryAccessInstruction for #enum_name {
                fn offset_operand_value(&self) -> Option<i64> {
                    match self {
                        #(#match_arms),*
                    }
                }
            }
        }
    }

    pub fn generate_separated_instruction_enum_numeric_mutation_impl(
        &self,
        enum_name: &Ident,
        instructions: &[&Instruction],
    ) -> TokenStream {
        if instructions.is_empty() {
            return quote! {
                impl NumericOperandMutation for #enum_name {
                    fn mutate_numeric_operands<R: rand::Rng>(
                        &mut self,
                        _rng: &mut R,
                        _config: &RandomConfig,
                    ) -> bool {
                        false
                    }
                }
            };
        }

        let match_arms = instructions
            .iter()
            .map(|inst| {
                let variant_ident = Ident::new(
                    &self.instruction_name_to_variant(&inst.name),
                    Span::call_site(),
                );
                quote! {
                    Self::#variant_ident(inner) => {
                        NumericOperandMutation::mutate_numeric_operands(inner, rng, config)
                    }
                }
            })
            .collect::<Vec<_>>();

        quote! {
            impl NumericOperandMutation for #enum_name {
                fn mutate_numeric_operands<R: rand::Rng>(
                    &mut self,
                    rng: &mut R,
                    config: &RandomConfig,
                ) -> bool {
                    match self {
                        #(#match_arms),*
                    }
                }
            }
        }
    }

    pub fn generate_separated_instruction_enum_runtime_operand_access_impl(
        &self,
        enum_name: &Ident,
        instructions: &[&Instruction],
    ) -> TokenStream {
        if instructions.is_empty() {
            return quote! {
                impl RuntimeOperandAccess for #enum_name {
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
                }
            };
        }

        let register_arms = instructions
            .iter()
            .map(|inst| {
                let variant_ident = Ident::new(
                    &self.instruction_name_to_variant(&inst.name),
                    Span::call_site(),
                );
                quote! {
                    Self::#variant_ident(inner) => RuntimeOperandAccess::register_operands(inner)
                }
            })
            .collect::<Vec<_>>();

        let memory_arms = instructions
            .iter()
            .map(|inst| {
                let variant_ident = Ident::new(
                    &self.instruction_name_to_variant(&inst.name),
                    Span::call_site(),
                );
                quote! {
                    Self::#variant_ident(inner) => RuntimeOperandAccess::memory_operands(inner)
                }
            })
            .collect::<Vec<_>>();

        let set_register_arms = instructions
            .iter()
            .map(|inst| {
                let variant_ident = Ident::new(
                    &self.instruction_name_to_variant(&inst.name),
                    Span::call_site(),
                );
                quote! {
                    Self::#variant_ident(inner) => {
                        RuntimeOperandAccess::set_register_operand(inner, operand_name, register)
                    }
                }
            })
            .collect::<Vec<_>>();

        let set_memory_arms = instructions
            .iter()
            .map(|inst| {
                let variant_ident = Ident::new(
                    &self.instruction_name_to_variant(&inst.name),
                    Span::call_site(),
                );
                quote! {
                    Self::#variant_ident(inner) => {
                        RuntimeOperandAccess::set_memory_operand(inner, index, patch)
                    }
                }
            })
            .collect::<Vec<_>>();

        quote! {
            impl RuntimeOperandAccess for #enum_name {
                fn register_operands(&self) -> Vec<RuntimeRegisterOperand> {
                    match self {
                        #(#register_arms),*
                    }
                }

                fn memory_operands(&self) -> Vec<RuntimeMemoryOperand> {
                    match self {
                        #(#memory_arms),*
                    }
                }

                fn set_register_operand(
                    &mut self,
                    operand_name: &str,
                    register: RuntimeRegisterAssignment,
                ) -> Result<bool, OperandAccessError> {
                    match self {
                        #(#set_register_arms),*
                    }
                }

                fn set_memory_operand(
                    &mut self,
                    index: usize,
                    patch: RuntimeMemoryOperandPatch,
                ) -> Result<bool, OperandAccessError> {
                    match self {
                        #(#set_memory_arms),*
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CodeGenerator;
    use riscv_instruction_schema::{
        AssemblySyntax, ISABase, ISAExtension, ImmediateKind, ImmediateSpec, Instruction,
        InstructionSemantics, MemoryAddressInfo, Operand, OperandKind, RegisterBank, RegisterSpec,
    };
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn sanitize_identifier_fragment_handles_problematic_extensions() {
        let generator = CodeGenerator::new(vec![]);

        assert_eq!(
            generator.extension_identifier_fragment(&ISAExtension::new("32-bit-ext")),
            "_32_x2D_bit_x2D_ext"
        );
        assert_eq!(
            generator.extension_identifier_fragment(&ISAExtension::new("type")),
            "type_"
        );
        assert_eq!(
            generator.extension_identifier_fragment(&ISAExtension::new("v.ext+foo")),
            "v_x2E_ext_x2B_foo"
        );
    }

    #[test]
    fn separated_generation_sanitizes_identifiers_but_keeps_raw_extension_names() {
        let generator = CodeGenerator::new(vec![Instruction {
            name: "foo.bar".to_string(),
            extension: ISAExtension::new("32-bit-ext"),
            isa_bases: vec![ISABase::RV32],
            required_extensions: BTreeMap::new(),
            operands: vec![],
            assembly_syntax: AssemblySyntax::Format("foo.bar".to_string()),
            memory_access: None,
            semantics: InstructionSemantics::default(),
        }]);

        let generated = generator.generate_separated_instruction_enum().to_string();
        let normalized: String = generated.split_whitespace().collect();

        assert!(
            normalized.contains(
                "pubenumRV32Extensions{#[serde(rename=\"32-bit-ext\")]_32_x2D_bit_x2D_ext"
            )
        );
        assert!(normalized.contains("pubenumRV32_32_x2D_bit_x2D_extInstructions"));
        assert!(normalized.contains("pubstructRV32__32_x2D_bit_x2D_ext_FOO_BAR;"));
        assert!(normalized.contains("Self::_32_x2D_bit_x2D_ext=>\"32-bit-ext\""));
    }

    #[test]
    fn operand_to_typed_struct_prefers_register_semantics_for_memory_base_and_saved_alias() {
        let generator = CodeGenerator::new(vec![]);
        let instruction = Instruction {
            name: "c.sw".to_string(),
            extension: ISAExtension::new("C"),
            isa_bases: vec![ISABase::RV32],
            required_extensions: BTreeMap::new(),
            operands: vec![
                Operand {
                    name: "xs1".to_string(),
                    kind: OperandKind::Register(RegisterSpec {
                        bank: RegisterBank::Integer,
                    }),
                    bit_lengths: HashMap::from([(ISABase::RV32, 3)]),
                    restrictions: None,
                    per_isa_restrictions: HashMap::new(),
                },
                Operand {
                    name: "r1s".to_string(),
                    kind: OperandKind::Register(RegisterSpec {
                        bank: RegisterBank::Integer,
                    }),
                    bit_lengths: HashMap::from([(ISABase::RV32, 5)]),
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

        assert_eq!(
            generator.operand_to_typed_struct(
                &instruction.operands[0],
                &ISABase::RV32,
                &instruction
            ),
            "CompressedBaseAddressRegister"
        );
        assert_eq!(
            generator.operand_to_typed_struct(
                &instruction.operands[1],
                &ISABase::RV32,
                &instruction
            ),
            "SavedIntegerRegister"
        );
    }

    #[test]
    fn operand_to_typed_struct_uses_shared_immediate_helper() {
        let generator = CodeGenerator::new(vec![]);
        let instruction = Instruction {
            name: "lw".to_string(),
            extension: ISAExtension::new("I"),
            isa_bases: vec![ISABase::RV64],
            required_extensions: BTreeMap::new(),
            operands: vec![Operand {
                name: "imm".to_string(),
                kind: OperandKind::Immediate(ImmediateSpec {
                    kind: ImmediateKind::Unsigned,
                }),
                bit_lengths: HashMap::from([(ISABase::RV64, 1)]),
                restrictions: None,
                per_isa_restrictions: HashMap::new(),
            }],
            assembly_syntax: AssemblySyntax::Format("lw x1, {imm}(x2)".to_string()),
            memory_access: Some(MemoryAddressInfo {
                base_register_operand: Some("xs1".to_string()),
                base_register_fixed: None,
                offset_operand: Some("imm".to_string()),
            }),
            semantics: InstructionSemantics::default(),
        };

        assert_eq!(
            generator.operand_to_typed_struct(
                &instruction.operands[0],
                &ISABase::RV64,
                &instruction
            ),
            "bool"
        );
    }
}
