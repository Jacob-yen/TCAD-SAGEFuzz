use super::CodeGenerator;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use riscv_instruction_schema::{ISABase, ISAExtension, Instruction};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use syn::Ident;

impl CodeGenerator {
    fn extract_format_string_for_enum_parse(&self, code: &str) -> Option<String> {
        let marker = "format!(\"";
        let start = code.find(marker)? + marker.len();
        let rest = &code[start..];
        let mut escaped = false;

        for (idx, ch) in rest.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }

            match ch {
                '\\' => escaped = true,
                '"' => return Some(rest[..idx].to_string()),
                _ => {}
            }
        }

        None
    }

    fn instruction_display_mnemonic(&self, instruction: &Instruction) -> Option<String> {
        let token = match &instruction.assembly_syntax {
            riscv_instruction_schema::AssemblySyntax::Format(pattern) => {
                pattern.split_ascii_whitespace().next()?.to_string()
            }
            riscv_instruction_schema::AssemblySyntax::RustCode(code) => self
                .extract_format_string_for_enum_parse(code)?
                .split_ascii_whitespace()
                .next()?
                .to_string(),
        };

        Some(
            token
                .split('{')
                .next()
                .expect("split always returns at least one segment")
                .to_string(),
        )
    }

    fn instruction_parse_keys(&self, instruction: &Instruction) -> Vec<String> {
        if instruction.name.starts_with("ssamoswap.") {
            return vec![
                instruction.name.clone(),
                format!("{}.aq", instruction.name),
                format!("{}.rl", instruction.name),
                format!("{}.aqrl", instruction.name),
            ];
        }

        self.instruction_display_mnemonic(instruction)
            .map(|mnemonic| vec![mnemonic])
            .unwrap_or_else(|| vec![instruction.name.clone()])
    }


    fn struct_ident_for_instruction(
        &self,
        instruction: &Instruction,
        extension: &ISAExtension,
        isa_base: &ISABase,
    ) -> Ident {
        Ident::new(
            &format!(
                "{}_{}_{}",
                isa_base,
                self.extension_identifier_fragment(extension),
                self.instruction_name_to_variant(&instruction.name)
            ),
            Span::call_site(),
        )
    }

    fn generate_leaf_enum_from_str(
        &self,
        enum_name: &Ident,
        cases: Vec<(Vec<String>, Ident, Ident)>,
    ) -> TokenStream {
        let mut grouped: BTreeMap<String, Vec<(Ident, Ident)>> = BTreeMap::new();
        for (keys, variant_ident, inner_ident) in cases {
            for key in keys {
                grouped
                    .entry(key)
                    .or_default()
                    .push((variant_ident.clone(), inner_ident.clone()));
            }
        }

        let match_arms: Vec<_> = grouped
            .into_iter()
            .map(|(key, candidates)| {
                let attempts: Vec<_> = candidates
                    .into_iter()
                    .map(|(variant_ident, inner_ident)| {
                        quote! {
                            match <#inner_ident as AsmDecode>::decode_asm(trimmed) {
                                Ok(value) => return Ok(Self::#variant_ident(value)),
                                Err(err) => {
                                    if !err.is_mnemonic_mismatch() && first_error.is_none() {
                                        first_error = Some(err);
                                    }
                                }
                            };
                        }
                    })
                    .collect();

                quote! {
                    #key => {
                        let mut first_error = None;
                        #(#attempts)*
                        Err(first_error.unwrap_or_else(|| {
                            AsmParseError::invalid_instruction(
                                stringify!(#enum_name),
                                format!("unable to parse `{}`", trimmed),
                            )
                        }))
                    }
                }
            })
            .collect();

        quote! {
            impl AsmDecode for #enum_name {
                fn decode_asm(s: &str) -> Result<Self, AsmParseError> {
                    let trimmed = s.trim();
                    if trimmed.is_empty() {
                        return Err(AsmParseError::empty_input(stringify!(#enum_name)));
                    }

                    let mnemonic = asm_first_token(trimmed)
                        .ok_or_else(|| AsmParseError::empty_input(stringify!(#enum_name)))?;

                    match mnemonic {
                        #(#match_arms),*,
                        _ => Err(AsmParseError::new(
                            stringify!(#enum_name),
                            AsmParseErrorKind::MnemonicMismatch,
                            format!("unsupported mnemonic `{}`", mnemonic),
                        )),
                    }
                }
            }

            impl std::str::FromStr for #enum_name {
                type Err = AsmParseError;
                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    <Self as AsmDecode>::decode_asm(s)
                }
            }
        }
    }

    fn generate_wrapper_enum_from_str(
        &self,
        enum_name: &Ident,
        wrappers: &[(Ident, Ident)],
    ) -> TokenStream {
        let attempts: Vec<_> = wrappers
            .iter()
            .map(|(variant_ident, inner_ident)| {
                quote! {
                    match <#inner_ident as AsmDecode>::decode_asm(trimmed) {
                        Ok(value) => return Ok(Self::#variant_ident(value)),
                        Err(err) => {
                            if !err.is_mnemonic_mismatch() && first_error.is_none() {
                                first_error = Some(err);
                            }
                        }
                    };
                }
            })
            .collect();

        quote! {
            impl AsmDecode for #enum_name {
                fn decode_asm(s: &str) -> Result<Self, AsmParseError> {
                    let trimmed = s.trim();
                    if trimmed.is_empty() {
                        return Err(AsmParseError::empty_input(stringify!(#enum_name)));
                    }

                    let mut first_error = None;
                    #(#attempts)*

                    if let Some(err) = first_error {
                        Err(err)
                    } else {
                        let mnemonic = asm_first_token(trimmed).unwrap_or(trimmed);
                        Err(AsmParseError::new(
                            stringify!(#enum_name),
                            AsmParseErrorKind::MnemonicMismatch,
                            format!("unsupported mnemonic `{}`", mnemonic),
                        ))
                    }
                }
            }

            impl std::str::FromStr for #enum_name {
                type Err = AsmParseError;
                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    <Self as AsmDecode>::decode_asm(s)
                }
            }
        }
    }

    /// Generate `impl AsmDecodeWithContext` for the top-level separated
    /// `RiscvInstruction` which has `RV32(RV32Instruction)` / `RV64(RV64Instruction)`
    /// variants.  The context selects which ISA-specific inner enum to try first,
    /// then falls back to the other.
    fn generate_separated_top_level_with_context(
        &self,
        enum_name: &Ident,
        rv32_inner: Option<&Ident>,
        rv64_inner: Option<&Ident>,
        pseudo_inner: &Ident,
    ) -> TokenStream {
        let rv32_attempt = rv32_inner.map(|inner| quote! {
            match <#inner as AsmDecode>::decode_asm(trimmed) {
                Ok(value) => return Ok(Self::RV32(value)),
                Err(err) => {
                    if !err.is_mnemonic_mismatch() && first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            };
        });
        let rv64_attempt = rv64_inner.map(|inner| quote! {
            match <#inner as AsmDecode>::decode_asm(trimmed) {
                Ok(value) => return Ok(Self::RV64(value)),
                Err(err) => {
                    if !err.is_mnemonic_mismatch() && first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            };
        });

        let ctx_preferred = quote! {
            match ctx {
                IsaDecodeContext::RV32 => {
                    #rv32_attempt
                    #rv64_attempt
                }
                IsaDecodeContext::RV64 => {
                    #rv64_attempt
                    #rv32_attempt
                }
            }
        };

        quote! {
            impl AsmDecodeWithContext for #enum_name {
                fn decode_asm_with_context(
                    s: &str,
                    ctx: IsaDecodeContext,
                ) -> Result<Self, AsmParseError> {
                    let trimmed = s.trim();
                    if trimmed.is_empty() {
                        return Err(AsmParseError::empty_input(stringify!(#enum_name)));
                    }

                    let mut first_error = None;

                    #ctx_preferred

                    match <#pseudo_inner as AsmDecodeWithContext>::decode_asm_with_context(trimmed, ctx) {
                        Ok(value) => return Ok(Self::Pseudo(value)),
                        Err(err) => {
                            if !err.is_mnemonic_mismatch() && first_error.is_none() {
                                first_error = Some(err);
                            }
                        }
                    };

                    if let Some(err) = first_error {
                        Err(err)
                    } else {
                        let mnemonic = asm_first_token(trimmed).unwrap_or(trimmed);
                        Err(AsmParseError::new(
                            stringify!(#enum_name),
                            AsmParseErrorKind::MnemonicMismatch,
                            format!("unsupported mnemonic `{}`", mnemonic),
                        ))
                    }
                }
            }
        }
    }


    fn required_extensions_for_instruction_set(
        &self,
        instructions: &[&Instruction],
        isa_base: ISABase,
    ) -> Vec<String> {
        let mut required = BTreeSet::new();

        for instruction in instructions {
            required.extend(
                instruction
                    .required_extensions_for(isa_base)
                    .iter()
                    .map(ToString::to_string),
            );
        }

        required.into_iter().collect()
    }

    /// Generate fully separated instruction enums.
    /// Instructions remain split by extension and ISA base with no merging.
    pub fn generate_separated_enums(&self) -> TokenStream {
        let mut all_enums = TokenStream::new();
        let mut by_extension_and_isa: HashMap<(ISAExtension, ISABase), Vec<&Instruction>> =
            HashMap::new();

        // Group by extension and ISA base.
        for inst in &self.instructions {
            for &isa_base in &inst.isa_bases {
                by_extension_and_isa
                    .entry((inst.extension.clone(), isa_base))
                    .or_default()
                    .push(inst);
            }
        }

        // Generate an enum for each combination.
        for ((extension, isa_base), instructions) in by_extension_and_isa {
            let enum_name = Ident::new(
                &format!(
                    "{}{}Instructions",
                    isa_base,
                    self.extension_identifier_fragment(&extension)
                ),
                Span::call_site(),
            );
            let doc_comment = format!("{} {} instructions", isa_base, extension);

            let variants = instructions
                .iter()
                .map(|&inst| {
                    let variant_name = Ident::new(
                        &self.instruction_name_to_variant(&inst.name),
                        Span::call_site(),
                    );

                    let struct_name = format!(
                        "{}_{}_{}",
                        isa_base,
                        self.extension_identifier_fragment(&extension),
                        self.instruction_name_to_variant(&inst.name)
                    );
                    let struct_ident = Ident::new(&struct_name, Span::call_site());

                    quote! {
                        #variant_name(#struct_ident)
                    }
                })
                .collect::<Vec<_>>();

            all_enums.extend(quote! {
                #[doc = #doc_comment]
                #[allow(non_camel_case_types)]
                #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DeriveInstructionDisplay, DeriveRandom)]
                #[rustfmt::skip]
                pub enum #enum_name {
                    #(#variants),*
                }
            });

            let parse_cases: Vec<_> = instructions
                .iter()
                .map(|instruction| {
                    (
                        self.instruction_parse_keys(instruction),
                        Ident::new(
                            &self.instruction_name_to_variant(&instruction.name),
                            Span::call_site(),
                        ),
                        self.struct_ident_for_instruction(instruction, &extension, &isa_base),
                    )
                })
                .collect();
            all_enums.extend(self.generate_leaf_enum_from_str(&enum_name, parse_cases));

            // Generate VARIANT_COUNT + random_at_index for exhaustive per-instruction testing.
            let variant_count = instructions.len();
            let random_at_index_arms = instructions
                .iter()
                .enumerate()
                .map(|(idx, inst)| {
                    let variant_ident = Ident::new(
                        &self.instruction_name_to_variant(&inst.name),
                        Span::call_site(),
                    );
                    let struct_ident =
                        self.struct_ident_for_instruction(inst, &extension, &isa_base);
                    quote! {
                        #idx => Ok(Self::#variant_ident(
                            <#struct_ident as ::riscv_instruction_types::Random>::random_with_rng(rng, config)?
                        ))
                    }
                })
                .collect::<Vec<_>>();

            all_enums.extend(quote! {
                impl #enum_name {
                    pub const VARIANT_COUNT: usize = #variant_count;

                    pub fn random_at_index<R: rand::Rng>(
                        idx: usize,
                        rng: &mut R,
                        config: &::riscv_instruction_types::RandomConfig,
                    ) -> Result<Self, ::riscv_instruction_types::RandomGenerationError> {
                        match idx {
                            #(#random_at_index_arms,)*
                            _ => Err(::riscv_instruction_types::RandomGenerationError::new(
                                stringify!(#enum_name),
                                format!("variant index {} out of range (0..{})", idx, #variant_count),
                            )),
                        }
                    }
                }
            });

            let offset_impl =
                self.generate_separated_instruction_enum_offset_impl(&enum_name, &instructions);
            let numeric_impl = self.generate_separated_instruction_enum_numeric_mutation_impl(
                &enum_name,
                &instructions,
            );
            let runtime_impl = self
                .generate_separated_instruction_enum_runtime_operand_access_impl(
                    &enum_name,
                    &instructions,
                );
            all_enums.extend(offset_impl);
            all_enums.extend(numeric_impl);
            all_enums.extend(runtime_impl);
        }

        all_enums
    }

    /// Generate the main enum for separated mode.
    pub fn generate_separated_main_enum(&self) -> TokenStream {
        let mut by_extension_and_isa: HashMap<(ISAExtension, ISABase), Vec<&Instruction>> =
            HashMap::new();

        // Group by extension and ISA base.
        for inst in &self.instructions {
            for &isa_base in &inst.isa_bases {
                by_extension_and_isa
                    .entry((inst.extension.clone(), isa_base))
                    .or_default()
                    .push(inst);
            }
        }

        // Generate enums grouped by ISA base.
        let mut isa_enums = Vec::new();
        for isa_base in &[ISABase::RV32, ISABase::RV64] {
            let mut extensions_for_isa: Vec<_> = by_extension_and_isa
                .keys()
                .filter(|(_, base)| base == isa_base)
                .map(|(ext, _)| ext.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            extensions_for_isa.sort();

            if !extensions_for_isa.is_empty() {
                let isa_enum_name =
                    Ident::new(&format!("{}Instruction", isa_base), Span::call_site());
                let isa_doc = format!("{} instructions grouped by extension", isa_base);

                let mut extension_variants = Vec::new();
                let mut extension_variant_idents = Vec::new();
                let mut extension_inner_enums = Vec::new();
                for ext in &extensions_for_isa {
                    let ext_ident = self.extension_variant_ident(ext);
                    let enum_name = Ident::new(
                        &format!(
                            "{}{}Instructions",
                            isa_base,
                            self.extension_identifier_fragment(ext)
                        ),
                        Span::call_site(),
                    );
                    extension_variant_idents.push(ext_ident.clone());
                    extension_inner_enums.push(enum_name.clone());
                    extension_variants.push(quote! { #ext_ident(#enum_name) });
                }

                isa_enums.push(quote! {
                    #[doc = #isa_doc]
                    #[allow(non_camel_case_types)]
                    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DeriveInstructionDisplay, DeriveRandom)]
                    pub enum #isa_enum_name {
                        #(#extension_variants),*
                    }
                });

                let offset_impl = self
                    .generate_wrapper_enum_offset_impl(&isa_enum_name, &extension_variant_idents);
                let numeric_impl = self.generate_wrapper_enum_numeric_mutation_impl(
                    &isa_enum_name,
                    &extension_variant_idents,
                );
                let runtime_impl = self.generate_wrapper_enum_runtime_operand_access_impl(
                    &isa_enum_name,
                    &extension_variant_idents,
                );
                let parse_wrappers: Vec<_> = extension_variant_idents
                    .iter()
                    .cloned()
                    .zip(extension_inner_enums.iter().cloned())
                    .collect();
                isa_enums
                    .push(self.generate_wrapper_enum_from_str(&isa_enum_name, &parse_wrappers));
                isa_enums.push(offset_impl);
                isa_enums.push(numeric_impl);
                isa_enums.push(runtime_impl);
            }
        }

        // Generate the final top-level enum.
        let mut main_variants = Vec::new();
        let mut main_variant_idents = Vec::new();
        let mut main_inner_enums = Vec::new();
        for isa_base in &[ISABase::RV32, ISABase::RV64] {
            let has_instructions = by_extension_and_isa
                .keys()
                .any(|(_, base)| *base == *isa_base);
            if has_instructions {
                let isa_ident = Ident::new(&isa_base.to_string(), Span::call_site());
                let enum_name = Ident::new(&format!("{}Instruction", isa_base), Span::call_site());
                main_variant_idents.push(isa_ident.clone());
                main_inner_enums.push(enum_name.clone());
                main_variants.push(quote! { #isa_ident(#enum_name) });
            }
        }

        main_variant_idents.push(Ident::new("Pseudo", Span::call_site()));
        main_inner_enums.push(Ident::new("PseudoInstruction", Span::call_site()));
        main_variants.push(quote! { Pseudo(PseudoInstruction) });

        let mut result = TokenStream::new();
        result.extend(isa_enums.into_iter());

        let main_enum_name = Ident::new("RiscvInstruction", Span::call_site());
        if !main_variants.is_empty() {
            result.extend(quote! {
                /// Main RISC-V instruction enum, separated by ISA base.
                #[allow(non_camel_case_types)]
                #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DeriveInstructionDisplay, DeriveRandom)]
                pub enum #main_enum_name {
                    #(#main_variants),*
                }
            });
        } else {
            result.extend(quote! {
                /// Main RISC-V instruction enum.
                #[allow(non_camel_case_types)]
                #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, DeriveInstructionDisplay, DeriveRandom)]
                pub enum #main_enum_name {}
            });
        }

        let offset_impl =
            self.generate_wrapper_enum_offset_impl(&main_enum_name, &main_variant_idents);
        let numeric_impl =
            self.generate_wrapper_enum_numeric_mutation_impl(&main_enum_name, &main_variant_idents);
        let runtime_impl = self.generate_wrapper_enum_runtime_operand_access_impl(
            &main_enum_name,
            &main_variant_idents,
        );

        // `AsmDecodeWithContext` for separated RiscvInstruction: try context-preferred ISA first.
        let rv32_inner = main_variant_idents
            .iter()
            .zip(main_inner_enums.iter())
            .find(|(v, _)| v.to_string() == "RV32")
            .map(|(_, inner)| inner.clone());
        let rv64_inner = main_variant_idents
            .iter()
            .zip(main_inner_enums.iter())
            .find(|(v, _)| v.to_string() == "RV64")
            .map(|(_, inner)| inner.clone());
        let pseudo_inner = Ident::new("PseudoInstruction", Span::call_site());
        result.extend(self.generate_separated_top_level_with_context(
            &main_enum_name,
            rv32_inner.as_ref(),
            rv64_inner.as_ref(),
            &pseudo_inner,
        ));
        result.extend(quote! {
            impl From<PseudoInstruction> for #main_enum_name {
                fn from(value: PseudoInstruction) -> Self {
                    Self::Pseudo(value)
                }
            }

            impl InstructionSequenceItem for #main_enum_name {
                fn as_pseudo_instruction(&self) -> Option<&PseudoInstruction> {
                    match self {
                        Self::Pseudo(inner) => Some(inner),
                        _ => None,
                    }
                }

                fn as_pseudo_instruction_mut(&mut self) -> Option<&mut PseudoInstruction> {
                    match self {
                        Self::Pseudo(inner) => Some(inner),
                        _ => None,
                    }
                }
            }
        });
        result.extend(offset_impl);
        result.extend(numeric_impl);
        result.extend(runtime_impl);

        result
    }

    /// Generate separated-mode extension enums together with their random-generation methods.
    pub fn generate_separated_extension_enums_with_random(&self) -> TokenStream {
        let mut by_extension_and_isa: HashMap<(ISAExtension, ISABase), Vec<&Instruction>> =
            HashMap::new();

        // Group by extension and ISA base.
        for inst in &self.instructions {
            for &isa_base in &inst.isa_bases {
                by_extension_and_isa
                    .entry((inst.extension.clone(), isa_base))
                    .or_default()
                    .push(inst);
            }
        }

        let extension_enums = self.generate_separated_extension_enums(&by_extension_and_isa);
        let extension_impls = self.generate_separated_extension_random_impls(&by_extension_and_isa);

        quote! {
            #extension_enums
            #extension_impls
        }
    }

    /// Generate separated-mode extension enum definitions.
    fn generate_separated_extension_enums(
        &self,
        by_extension_and_isa: &HashMap<(ISAExtension, ISABase), Vec<&Instruction>>,
    ) -> TokenStream {
        let mut all_enums = TokenStream::new();

        // Collect extensions and group them by ISA base.
        let mut rv32_extensions = HashSet::new();
        let mut rv64_extensions = HashSet::new();

        for ((extension, isa_base), _) in by_extension_and_isa {
            match isa_base {
                ISABase::RV32 => {
                    rv32_extensions.insert(extension.clone());
                }
                ISABase::RV64 => {
                    rv64_extensions.insert(extension.clone());
                }
            }
        }

        // Generate the `RV32Extensions` enum.
        if !rv32_extensions.is_empty() {
            let mut rv32_variants = rv32_extensions.into_iter().collect::<Vec<_>>();
            rv32_variants.sort();

            let rv32_enum_variants = rv32_variants.iter().map(|ext| {
                let ident = self.extension_variant_ident(ext);
                let ext_name = ext.to_string();
                quote! {
                    #[serde(rename = #ext_name)]
                    #ident
                }
            });
            let rv32_display_arms = rv32_variants.iter().map(|ext| {
                let ident = self.extension_variant_ident(ext);
                let ext_name = ext.to_string();
                quote! { Self::#ident => #ext_name }
            });
            let rv32_required_arms = rv32_variants.iter().map(|ext| {
                let ident = self.extension_variant_ident(ext);
                let required = self.required_extensions_for_instruction_set(
                    by_extension_and_isa
                        .get(&(ext.clone(), ISABase::RV32))
                        .expect("extension instruction set exists"),
                    ISABase::RV32,
                );
                quote! { Self::#ident => &[#(#required),*] }
            });

            all_enums.extend(quote! {
                /// Available extensions for RV32 ISA base
                #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Sequence)]
                #[allow(non_camel_case_types)]
                pub enum RV32Extensions {
                    #(#rv32_enum_variants),*
                }

                impl std::fmt::Display for RV32Extensions {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let extension_name = match self {
                            #(#rv32_display_arms),*
                        };
                        write!(f, "{}", extension_name)
                    }
                }

                impl crate::march_util::MarchExtension for RV32Extensions {
                    fn required_extension_names(&self) -> &'static [&'static str] {
                        match self {
                            #(#rv32_required_arms),*
                        }
                    }
                }
            });
        }

        // Generate the `RV64Extensions` enum.
        if !rv64_extensions.is_empty() {
            let mut rv64_variants = rv64_extensions.into_iter().collect::<Vec<_>>();
            rv64_variants.sort();

            let rv64_enum_variants = rv64_variants.iter().map(|ext| {
                let ident = self.extension_variant_ident(ext);
                let ext_name = ext.to_string();
                quote! {
                    #[serde(rename = #ext_name)]
                    #ident
                }
            });
            let rv64_display_arms = rv64_variants.iter().map(|ext| {
                let ident = self.extension_variant_ident(ext);
                let ext_name = ext.to_string();
                quote! { Self::#ident => #ext_name }
            });
            let rv64_required_arms = rv64_variants.iter().map(|ext| {
                let ident = self.extension_variant_ident(ext);
                let required = self.required_extensions_for_instruction_set(
                    by_extension_and_isa
                        .get(&(ext.clone(), ISABase::RV64))
                        .expect("extension instruction set exists"),
                    ISABase::RV64,
                );
                quote! { Self::#ident => &[#(#required),*] }
            });

            all_enums.extend(quote! {
                /// Available extensions for RV64 ISA base
                #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Sequence)]
                #[allow(non_camel_case_types)]
                pub enum RV64Extensions {
                    #(#rv64_enum_variants),*
                }

                impl std::fmt::Display for RV64Extensions {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        let extension_name = match self {
                            #(#rv64_display_arms),*
                        };
                        write!(f, "{}", extension_name)
                    }
                }

                impl crate::march_util::MarchExtension for RV64Extensions {
                    fn required_extension_names(&self) -> &'static [&'static str] {
                        match self {
                            #(#rv64_required_arms),*
                        }
                    }
                }
            });
        }

        all_enums
    }

    /// Generate random-generation implementations for separated-mode extension enums.
    fn generate_separated_extension_random_impls(
        &self,
        by_extension_and_isa: &HashMap<(ISAExtension, ISABase), Vec<&Instruction>>,
    ) -> TokenStream {
        let mut all_impls = TokenStream::new();

        // Collect extensions and group them by ISA base.
        let mut rv32_extensions = HashSet::new();
        let mut rv64_extensions = HashSet::new();

        for ((extension, isa_base), _) in by_extension_and_isa {
            match isa_base {
                ISABase::RV32 => {
                    rv32_extensions.insert(extension.clone());
                }
                ISABase::RV64 => {
                    rv64_extensions.insert(extension.clone());
                }
            }
        }

        // Generate the implementation for `RV32Extensions`.
        if !rv32_extensions.is_empty() {
            let mut rv32_variants = rv32_extensions.into_iter().collect::<Vec<_>>();
            rv32_variants.sort();

            let rv32_match_arms = rv32_variants.iter().map(|ext| {
                let ext_ident = self.extension_variant_ident(ext);
                let enum_name = Ident::new(
                    &format!(
                        "RV32{}Instructions",
                        self.extension_identifier_fragment(ext)
                    ),
                    Span::call_site(),
                );

                quote! {
                    RV32Extensions::#ext_ident => {
                        Ok(RiscvInstruction::RV32(RV32Instruction::#ext_ident(
                            #enum_name::random_with_rng(rng, config)?
                        )))
                    }
                }
            });

            let rv32_sequence_match_arms = rv32_variants.iter().map(|ext| {
                let ext_ident = self.extension_variant_ident(ext);
                let enum_name = Ident::new(
                    &format!(
                        "RV32{}Instructions",
                        self.extension_identifier_fragment(ext)
                    ),
                    Span::call_site(),
                );

                quote! {
                    RV32Extensions::#ext_ident => {
                        let instruction = RiscvInstruction::RV32(RV32Instruction::#ext_ident(
                            #enum_name::random_with_rng(rng, config)?
                        ));
                        build_instruction_sequence(instruction, rng, config)
                    }
                }
            });

            let rv32_instruction_count_arms = rv32_variants.iter().map(|ext| {
                let ext_ident = self.extension_variant_ident(ext);
                let key = (ext.clone(), ISABase::RV32);
                let count = by_extension_and_isa
                    .get(&key)
                    .map(|instructions| instructions.len())
                    .unwrap_or(0);

                quote! {
                    RV32Extensions::#ext_ident => #count
                }
            });

            let rv32_all_sequences_arms = rv32_variants.iter().map(|ext| {
                let ext_ident = self.extension_variant_ident(ext);
                let enum_name = Ident::new(
                    &format!(
                        "RV32{}Instructions",
                        self.extension_identifier_fragment(ext)
                    ),
                    Span::call_site(),
                );
                quote! {
                    RV32Extensions::#ext_ident => {
                        let mut sequences = Vec::new();
                        for idx in 0..#enum_name::VARIANT_COUNT {
                            for _ in 0..samples_per_instruction {
                                match #enum_name::random_at_index(idx, rng, config) {
                                    Ok(inst) => {
                                        let instruction = RiscvInstruction::RV32(RV32Instruction::#ext_ident(inst));
                                        match build_instruction_sequence(instruction, rng, config) {
                                            Ok(seq) => sequences.push(Ok(seq)),
                                            Err(e) => sequences.push(Err(e)),
                                        }
                                    }
                                    Err(e) => sequences.push(Err(e)),
                                }
                            }
                        }
                        sequences
                    }
                }
            });

            all_impls.extend(quote! {
                impl RV32Extensions {
                    /// Generate a random instruction for this extension
                    pub fn random_instruction<R: rand::Rng>(
                        &self,
                        rng: &mut R,
                        config: &RandomConfig,
                    ) -> Result<RiscvInstruction, RandomGenerationError> {
                        match self {
                            #(#rv32_match_arms),*
                        }
                    }

                    /// Generate a random instruction sequence (including necessary pseudo instructions) for this extension
                    pub fn random_sequence_with_rng<R: rand::Rng>(
                        &self,
                        rng: &mut R,
                        config: &RandomConfig,
                    ) -> Result<InstructionSequence<RiscvInstruction>, RandomGenerationError> {
                        match self {
                            #(#rv32_sequence_match_arms),*
                        }
                    }

                    /// Return the total number of RV32 instructions exposed by this extension
                    pub const fn instruction_count(&self) -> usize {
                        match self {
                            #(#rv32_instruction_count_arms),*
                        }
                    }

                    /// Generate `samples_per_instruction` sequences for every instruction variant in this extension.
                    /// Used by assembler compatibility tests to guarantee full instruction coverage.
                    pub fn all_instruction_sequences_for_test<R: rand::Rng>(
                        &self,
                        rng: &mut R,
                        config: &RandomConfig,
                        samples_per_instruction: usize,
                    ) -> Vec<Result<InstructionSequence<RiscvInstruction>, RandomGenerationError>> {
                        match self {
                            #(#rv32_all_sequences_arms),*
                        }
                    }
                }
            });
        }

        // Generate the implementation for `RV64Extensions`.
        if !rv64_extensions.is_empty() {
            let mut rv64_variants = rv64_extensions.into_iter().collect::<Vec<_>>();
            rv64_variants.sort();

            let rv64_match_arms = rv64_variants.iter().map(|ext| {
                let ext_ident = self.extension_variant_ident(ext);
                let enum_name = Ident::new(
                    &format!(
                        "RV64{}Instructions",
                        self.extension_identifier_fragment(ext)
                    ),
                    Span::call_site(),
                );

                quote! {
                    RV64Extensions::#ext_ident => {
                        Ok(RiscvInstruction::RV64(RV64Instruction::#ext_ident(
                            #enum_name::random_with_rng(rng, config)?
                        )))
                    }
                }
            });

            let rv64_sequence_match_arms = rv64_variants.iter().map(|ext| {
                let ext_ident = self.extension_variant_ident(ext);
                let enum_name = Ident::new(
                    &format!(
                        "RV64{}Instructions",
                        self.extension_identifier_fragment(ext)
                    ),
                    Span::call_site(),
                );

                quote! {
                    RV64Extensions::#ext_ident => {
                        let instruction = RiscvInstruction::RV64(RV64Instruction::#ext_ident(
                            #enum_name::random_with_rng(rng, config)?
                        ));
                        build_instruction_sequence(instruction, rng, config)
                    }
                }
            });

            let rv64_instruction_count_arms = rv64_variants.iter().map(|ext| {
                let ext_ident = self.extension_variant_ident(ext);
                let key = (ext.clone(), ISABase::RV64);
                let count = by_extension_and_isa
                    .get(&key)
                    .map(|instructions| instructions.len())
                    .unwrap_or(0);

                quote! {
                    RV64Extensions::#ext_ident => #count
                }
            });

            let rv64_all_sequences_arms = rv64_variants.iter().map(|ext| {
                let ext_ident = self.extension_variant_ident(ext);
                let enum_name = Ident::new(
                    &format!(
                        "RV64{}Instructions",
                        self.extension_identifier_fragment(ext)
                    ),
                    Span::call_site(),
                );
                quote! {
                    RV64Extensions::#ext_ident => {
                        let mut sequences = Vec::new();
                        for idx in 0..#enum_name::VARIANT_COUNT {
                            for _ in 0..samples_per_instruction {
                                match #enum_name::random_at_index(idx, rng, config) {
                                    Ok(inst) => {
                                        let instruction = RiscvInstruction::RV64(RV64Instruction::#ext_ident(inst));
                                        match build_instruction_sequence(instruction, rng, config) {
                                            Ok(seq) => sequences.push(Ok(seq)),
                                            Err(e) => sequences.push(Err(e)),
                                        }
                                    }
                                    Err(e) => sequences.push(Err(e)),
                                }
                            }
                        }
                        sequences
                    }
                }
            });

            all_impls.extend(quote! {
                impl RV64Extensions {
                    /// Generate a random instruction for this extension
                    pub fn random_instruction<R: rand::Rng>(
                        &self,
                        rng: &mut R,
                        config: &RandomConfig,
                    ) -> Result<RiscvInstruction, RandomGenerationError> {
                        match self {
                            #(#rv64_match_arms),*
                        }
                    }

                    /// Generate a random instruction sequence (including necessary pseudo instructions) for this extension
                    pub fn random_sequence_with_rng<R: rand::Rng>(
                        &self,
                        rng: &mut R,
                        config: &RandomConfig,
                    ) -> Result<InstructionSequence<RiscvInstruction>, RandomGenerationError> {
                        match self {
                            #(#rv64_sequence_match_arms),*
                        }
                    }

                    /// Return the total number of RV64 instructions exposed by this extension
                    pub const fn instruction_count(&self) -> usize {
                        match self {
                            #(#rv64_instruction_count_arms),*
                        }
                    }

                    /// Generate `samples_per_instruction` sequences for every instruction variant in this extension.
                    /// Used by assembler compatibility tests to guarantee full instruction coverage.
                    pub fn all_instruction_sequences_for_test<R: rand::Rng>(
                        &self,
                        rng: &mut R,
                        config: &RandomConfig,
                        samples_per_instruction: usize,
                    ) -> Vec<Result<InstructionSequence<RiscvInstruction>, RandomGenerationError>> {
                        match self {
                            #(#rv64_all_sequences_arms),*
                        }
                    }
                }
            });
        }

        all_impls
    }
}
