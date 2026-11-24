use super::CodeGenerator;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use riscv_instruction_schema::{ImmediateKind, RegisterBank, RegisterEncoding, RegisterSemantics};
use std::hash::{Hash, Hasher};

impl CodeGenerator {
    fn format_identifier_number(value: i64) -> String {
        if value < 0 {
            format!("Neg{}", value.abs())
        } else {
            value.to_string()
        }
    }

    fn forbidden_identifier_fragment(&self, values: &[i64]) -> String {
        if values.is_empty() {
            return String::new();
        }

        if values.windows(2).all(|pair| pair[1] == pair[0] + 1) {
            return format!(
                "ForbiddenRange{}To{}",
                Self::format_identifier_number(values[0]),
                Self::format_identifier_number(*values.last().unwrap())
            );
        }

        if values.len() <= 8 {
            return format!(
                "Forbidden{}",
                values
                    .iter()
                    .map(|value| Self::format_identifier_number(*value))
                    .collect::<Vec<_>>()
                    .join("_")
            );
        }

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        values.hash(&mut hasher);
        format!("ForbiddenSet{:016x}", hasher.finish())
    }

    /// Generate a restricted register type name.
    pub fn generate_restricted_register_type_name(
        &self,
        base_type: &str,
        operand_name: &str,
        restriction: &riscv_instruction_schema::OperandRestriction,
        use_memory_register_range: bool,
    ) -> String {
        let mut name_parts = vec![format!("Restricted{}", base_type)];

        // Include the operand name to distinguish different register roles.
        let operand_name_capitalized = operand_name
            .chars()
            .next()
            .unwrap()
            .to_uppercase()
            .collect::<String>()
            + &operand_name[1..];
        name_parts.push(operand_name_capitalized);

        // Encode restriction information into the type name.
        if let Some(multiple) = restriction.multiple_of {
            name_parts.push(format!("Mul{}", multiple));
        }
        if let Some((min, max)) = restriction.min_max {
            // Rewrite negative values by replacing the sign with `Neg`.
            let min_str = if min < 0 {
                format!("Neg{}", min.abs())
            } else {
                min.to_string()
            };
            let max_str = if max < 0 {
                format!("Neg{}", max.abs())
            } else {
                max.to_string()
            };
            name_parts.push(format!("Range{}To{}", min_str, max_str));
        }
        if !restriction.forbidden_values.is_empty() {
            name_parts.push(self.forbidden_identifier_fragment(&restriction.forbidden_values));
        }
        if restriction.odd_only {
            name_parts.push("OddOnly".to_string());
        }

        if use_memory_register_range {
            name_parts.push("MemoryRange".to_string());
        }

        name_parts.join("")
    }

    fn immediate_inner_type(&self, kind: ImmediateKind) -> TokenStream {
        if kind.is_unsigned() {
            quote! { u32 }
        } else {
            quote! { i32 }
        }
    }

    fn immediate_default_attr_bounds(
        &self,
        kind: ImmediateKind,
        bit_length: u8,
    ) -> (String, String) {
        if kind.is_unsigned() {
            (
                "0".to_string(),
                format!("((1u64 << {}) - 1) as u32", bit_length),
            )
        } else {
            (
                format!("-(1i32 << ({} - 1))", bit_length),
                format!("(1i32 << ({} - 1)) - 1", bit_length),
            )
        }
    }

    fn immediate_base_bounds(
        &self,
        kind: ImmediateKind,
        bit_length: u8,
        restriction: &riscv_instruction_schema::OperandRestriction,
    ) -> (TokenStream, TokenStream) {
        if kind.is_unsigned() {
            if let Some((min, max)) = restriction.min_max {
                (quote! { #min as u64 }, quote! { #max as u64 })
            } else {
                (quote! { 0u64 }, quote! { ((1u64 << #bit_length) - 1) })
            }
        } else if let Some((min, max)) = restriction.min_max {
            (quote! { #min as i64 }, quote! { #max as i64 })
        } else {
            (
                quote! { (-(1i64 << (#bit_length - 1))) },
                quote! { ((1i64 << (#bit_length - 1)) - 1) },
            )
        }
    }

    fn generate_unsigned_immediate_mutation_impl(
        &self,
        type_name: &syn::Ident,
        sampler: &str,
        base_min: &TokenStream,
        base_max: &TokenStream,
        bit_length: u8,
    ) -> TokenStream {
        let sampler = format_ident!("{}", sampler);
        quote! {
            impl<const MEM_AWARE: bool> NumericOperandMutation for #type_name<MEM_AWARE> {
                fn mutate_numeric_operands<R: rand::Rng>(
                    &mut self,
                    rng: &mut R,
                    config: &RandomConfig,
                ) -> bool {
                    let (mut min_val, mut max_val) = (#base_min as i64, #base_max as i64);
                    if MEM_AWARE {
                        let (cfg_min, cfg_max) = config.memory_access_config.memory_offset_immediate_range;
                        if cfg_min > min_val {
                            min_val = cfg_min;
                        }
                        if cfg_max < max_val {
                            max_val = cfg_max;
                        }
                    }
                    if max_val < 0 {
                        return false;
                    }
                    let min_u = min_val.max(0) as u64;
                    let max_u = max_val as u64;
                    if min_u > max_u {
                        return false;
                    }
                    let candidate = #sampler(rng, min_u, max_u, #bit_length);
                    self.set(candidate as u32).is_ok()
                }
            }
        }
    }

    fn generate_signed_immediate_mutation_impl(
        &self,
        type_name: &syn::Ident,
        sampler: &str,
        base_min: &TokenStream,
        base_max: &TokenStream,
        bit_length: u8,
    ) -> TokenStream {
        let sampler = format_ident!("{}", sampler);
        quote! {
            impl<const MEM_AWARE: bool> NumericOperandMutation for #type_name<MEM_AWARE> {
                fn mutate_numeric_operands<R: rand::Rng>(
                    &mut self,
                    rng: &mut R,
                    config: &RandomConfig,
                ) -> bool {
                    let (mut min_val, mut max_val) = (#base_min, #base_max);
                    if MEM_AWARE {
                        let (cfg_min, cfg_max) = config.memory_access_config.memory_offset_immediate_range;
                        if cfg_min > min_val {
                            min_val = cfg_min;
                        }
                        if cfg_max < max_val {
                            max_val = cfg_max;
                        }
                    }
                    let candidate = #sampler(rng, min_val, max_val, #bit_length);
                    self.set(candidate as i32).is_ok()
                }
            }
        }
    }

    fn generate_fixed_signed_immediate_mutation_impl(
        &self,
        type_name: &syn::Ident,
        value: i32,
    ) -> TokenStream {
        quote! {
            impl<const MEM_AWARE: bool> NumericOperandMutation for #type_name<MEM_AWARE> {
                fn mutate_numeric_operands<R: rand::Rng>(
                    &mut self,
                    _rng: &mut R,
                    _config: &RandomConfig,
                ) -> bool {
                    self.set(#value).is_ok()
                }
            }
        }
    }

    fn generate_unsigned_immediate_random_impl(
        &self,
        type_name: &syn::Ident,
        sampler: &str,
        base_min: &TokenStream,
        base_max: &TokenStream,
        bit_length: u8,
    ) -> TokenStream {
        let sampler = format_ident!("{}", sampler);
        quote! {
            impl<const MEM_AWARE: bool> Random for #type_name<MEM_AWARE> {
                type Output = Self;

                fn random_with_rng<R: rand::Rng>(
                    rng: &mut R,
                    config: &RandomConfig,
                ) -> Result<Self::Output, RandomGenerationError> {
                    let (mut min_val, mut max_val) = (#base_min as i64, #base_max as i64);
                    if MEM_AWARE {
                        let (cfg_min, cfg_max) = config.memory_access_config.memory_offset_immediate_range;
                        if cfg_min > min_val {
                            min_val = cfg_min;
                        }
                        if cfg_max < max_val {
                            max_val = cfg_max;
                        }
                    }
                    if max_val < 0 {
                        return Err(RandomGenerationError::new(
                            stringify!(#type_name),
                            "configured max is negative",
                        ));
                    }
                    let min_u = min_val.max(0) as u64;
                    let max_u = max_val as u64;
                    if min_u > max_u {
                        return Err(RandomGenerationError::new(
                            stringify!(#type_name),
                            format!("invalid combined range [{}, {}]", min_u, max_u),
                        ));
                    }

                    const MAX_ATTEMPTS: usize = 10000;
                    let mut last_error: Option<String> = None;
                    for _ in 0..MAX_ATTEMPTS {
                        let candidate = #sampler(rng, min_u, max_u, #bit_length);
                        if Self::FORBIDDEN.contains(&(candidate as u32)) {
                            continue;
                        }
                        if Self::ODD_ONLY && candidate % 2 == 0 {
                            continue;
                        }
                        if let Some(multiple) = Self::MULTIPLE_OF {
                            if multiple != 0 && (candidate as u32) % multiple != 0 {
                                continue;
                            }
                        }
                        match Self::new(candidate as u32) {
                            Ok(value) => return Ok(value),
                            Err(err) => last_error = Some(err),
                        }
                    }

                    Err(RandomGenerationError::exhausted(
                        stringify!(#type_name),
                        MAX_ATTEMPTS,
                        last_error,
                    ))
                }
            }
        }
    }

    fn generate_signed_immediate_random_impl(
        &self,
        type_name: &syn::Ident,
        sampler: &str,
        base_min: &TokenStream,
        base_max: &TokenStream,
        bit_length: u8,
    ) -> TokenStream {
        let sampler = format_ident!("{}", sampler);
        quote! {
            impl<const MEM_AWARE: bool> Random for #type_name<MEM_AWARE> {
                type Output = Self;

                fn random_with_rng<R: rand::Rng>(
                    rng: &mut R,
                    config: &RandomConfig,
                ) -> Result<Self::Output, RandomGenerationError> {
                    let (mut min_val, mut max_val) = (#base_min, #base_max);
                    if MEM_AWARE {
                        let (cfg_min, cfg_max) = config.memory_access_config.memory_offset_immediate_range;
                        if cfg_min > min_val {
                            min_val = cfg_min;
                        }
                        if cfg_max < max_val {
                            max_val = cfg_max;
                        }
                    }
                    if min_val > max_val {
                        return Err(RandomGenerationError::new(
                            stringify!(#type_name),
                            format!("invalid combined range [{}, {}]", min_val, max_val),
                        ));
                    }

                    const MAX_ATTEMPTS: usize = 10000;
                    let mut last_error: Option<String> = None;
                    for _ in 0..MAX_ATTEMPTS {
                        let candidate = #sampler(rng, min_val, max_val, #bit_length);
                        if Self::FORBIDDEN.contains(&(candidate as i32)) {
                            continue;
                        }
                        if Self::ODD_ONLY && candidate % 2 == 0 {
                            continue;
                        }
                        if let Some(multiple) = Self::MULTIPLE_OF {
                            if multiple != 0 && (candidate as i32) % multiple != 0 {
                                continue;
                            }
                        }
                        match Self::new(candidate as i32) {
                            Ok(value) => return Ok(value),
                            Err(err) => last_error = Some(err),
                        }
                    }

                    Err(RandomGenerationError::exhausted(
                        stringify!(#type_name),
                        MAX_ATTEMPTS,
                        last_error,
                    ))
                }
            }
        }
    }

    fn generate_fixed_signed_immediate_random_impl(
        &self,
        type_name: &syn::Ident,
        value: i32,
    ) -> TokenStream {
        quote! {
            impl<const MEM_AWARE: bool> Random for #type_name<MEM_AWARE> {
                type Output = Self;

                fn random_with_rng<R: rand::Rng>(
                    _rng: &mut R,
                    _config: &RandomConfig,
                ) -> Result<Self::Output, RandomGenerationError> {
                    #type_name::new(#value)
                        .map_err(|err| RandomGenerationError::new(stringify!(#type_name), err))
                }
            }
        }
    }

    /// Generate a restricted immediate type name.
    pub fn generate_restricted_immediate_type_name(
        &self,
        immediate_kind: ImmediateKind,
        operand_name: &str,
        bit_length: u8,
        restriction: &riscv_instruction_schema::OperandRestriction,
    ) -> String {
        let mut name_parts = vec![format!("Restricted")];

        name_parts.push(
            immediate_kind
                .restricted_runtime_base_type_name(operand_name)
                .to_string(),
        );
        name_parts.push(format!("{}", bit_length));

        // Encode restriction information into the type name.
        if let Some(multiple) = restriction.multiple_of {
            name_parts.push(format!("Mul{}", multiple));
        }
        if let Some((min, max)) = restriction.min_max {
            // Rewrite negative values by replacing the sign with `Neg`.
            let min_str = if min < 0 {
                format!("Neg{}", min.abs())
            } else {
                min.to_string()
            };
            let max_str = if max < 0 {
                format!("Neg{}", max.abs())
            } else {
                max.to_string()
            };
            name_parts.push(format!("Range{}To{}", min_str, max_str));
        }
        if !restriction.forbidden_values.is_empty() {
            name_parts.push(self.forbidden_identifier_fragment(&restriction.forbidden_values));
        }
        if restriction.odd_only {
            name_parts.push("OddOnly".to_string());
        }

        // Do not add a `MemAware` suffix; distinguish it through the generic parameter instead.
        name_parts.join("")
    }

    /// Generate a restricted register type definition.
    pub fn generate_restricted_register_type_def(
        &self,
        semantics: RegisterSemantics,
        operand_name: &str,
        restriction: &riscv_instruction_schema::OperandRestriction,
        use_memory_register_range: bool,
    ) -> TokenStream {
        let base_type = semantics.plain_runtime_type_name();
        let type_name = syn::Ident::new(
            &self.generate_restricted_register_type_name(
                base_type,
                operand_name,
                restriction,
                use_memory_register_range,
            ),
            proc_macro2::Span::call_site(),
        );

        let display_format = semantics.runtime_display_format();
        let register_name = semantics.runtime_type_label();

        // Set default min/max bounds.
        let (min_str, max_str) = if let Some((min, max)) = restriction.min_max {
            (min.to_string(), max.to_string())
        } else {
            match (semantics.bank, semantics.encoding) {
                (RegisterBank::Integer, RegisterEncoding::Standard)
                | (RegisterBank::FloatingPoint, RegisterEncoding::Standard)
                | (RegisterBank::Vector, _) => ("0".to_string(), "31".to_string()),
                (RegisterBank::Integer, RegisterEncoding::Compressed)
                | (RegisterBank::FloatingPoint, RegisterEncoding::Compressed) => {
                    ("8".to_string(), "15".to_string())
                }
            }
        };

        let mut attrs = vec![
            quote! { min = #min_str },
            quote! { max = #max_str },
            quote! { name = #register_name },
            quote! { display = #display_format },
        ];

        let random_domain = semantics
            .bank
            .restricted_random_domain(use_memory_register_range);
        attrs.push(quote! { random_domain = #random_domain });

        if let Some(multiple) = restriction.multiple_of {
            let multiple_str = multiple.to_string();
            attrs.push(quote! { multiple_of = #multiple_str });
        }

        if !restriction.forbidden_values.is_empty() {
            let forbidden_str = restriction
                .forbidden_values
                .iter()
                .map(|v| format!("{}u8", v))
                .collect::<Vec<_>>()
                .join(",");
            attrs.push(quote! { forbidden = #forbidden_str });
        }

        if restriction.odd_only {
            attrs.push(quote! { odd_only = "true" });
        }

        let doc_comment = format!(
            "{} for {} with constraints: {:?}",
            register_name, operand_name, restriction
        );
        let bank_tokens = match semantics.bank {
            RegisterBank::Integer => quote!(RegisterBank::Integer),
            RegisterBank::FloatingPoint => quote!(RegisterBank::FloatingPoint),
            RegisterBank::Vector => quote!(RegisterBank::Vector),
        };
        let render_style_tokens = match (semantics.bank, semantics.role) {
            (RegisterBank::Integer, riscv_instruction_schema::RegisterRole::SavedAlias) => {
                quote!(RegisterRenderStyle::SavedInteger)
            }
            (RegisterBank::Integer, _) => quote!(RegisterRenderStyle::Integer),
            (RegisterBank::FloatingPoint, _) => quote!(RegisterRenderStyle::FloatingPoint),
            (RegisterBank::Vector, _) => quote!(RegisterRenderStyle::Vector),
        };

        quote! {
            #[doc = #doc_comment]
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue, DeriveRandom)]
            #[validated(#(#attrs),*)]
            pub struct #type_name(u8);

            impl MemoryAware for #type_name {
                fn is_memory_aware() -> bool { false }
            }

            impl RuntimeRegisterOperandValue for #type_name {
                const TYPE_NAME: &'static str = stringify!(#type_name);
                const BANK: RegisterBank = #bank_tokens;
                const RENDER_STYLE: RegisterRenderStyle = #render_style_tokens;
            }
        }
    }

    /// Generate a restricted immediate type definition.
    pub fn generate_restricted_immediate_type_def(
        &self,
        immediate_kind: ImmediateKind,
        operand_name: &str,
        bit_length: u8,
        restriction: &riscv_instruction_schema::OperandRestriction,
        is_memory_aware: bool,
    ) -> TokenStream {
        let type_name = syn::Ident::new(
            &self.generate_restricted_immediate_type_name(
                immediate_kind,
                operand_name,
                bit_length,
                restriction,
            ),
            proc_macro2::Span::call_site(),
        );

        let inner_type = self.immediate_inner_type(immediate_kind);

        let (min_str, max_str) = if let Some((min, max)) = restriction.min_max {
            (min.to_string(), max.to_string())
        } else {
            self.immediate_default_attr_bounds(immediate_kind, bit_length)
        };

        let mut attrs = vec![
            quote! { min = #min_str },
            quote! { max = #max_str },
            quote! { name = #operand_name },
        ];

        if let Some(multiple) = restriction.multiple_of {
            let multiple_str = multiple.to_string();
            attrs.push(quote! { multiple_of = #multiple_str });
        }

        if !restriction.forbidden_values.is_empty() {
            let forbidden_str = restriction
                .forbidden_values
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            attrs.push(quote! { forbidden = #forbidden_str });
        }

        if restriction.odd_only {
            attrs.push(quote! { odd_only = "true" });
        }

        let doc_comment = format!(
            "Restricted {} with bit length {}, constraints: {:?}, memory aware: {}",
            operand_name, bit_length, restriction, is_memory_aware
        );

        let (base_min, base_max) =
            self.immediate_base_bounds(immediate_kind, bit_length, restriction);

        let mutation_impl = if let Some(value) = immediate_kind.fixed_value() {
            self.generate_fixed_signed_immediate_mutation_impl(&type_name, value)
        } else if let Some(sampler) = immediate_kind.sampler_function_name() {
            if immediate_kind.is_unsigned() {
                self.generate_unsigned_immediate_mutation_impl(
                    &type_name, sampler, &base_min, &base_max, bit_length,
                )
            } else {
                self.generate_signed_immediate_mutation_impl(
                    &type_name, sampler, &base_min, &base_max, bit_length,
                )
            }
        } else {
            unreachable!("restricted immediate kinds must provide a sampler or fixed value")
        };

        let random_impl = if let Some(value) = immediate_kind.fixed_value() {
            self.generate_fixed_signed_immediate_random_impl(&type_name, value)
        } else if let Some(sampler) = immediate_kind.sampler_function_name() {
            if immediate_kind.is_unsigned() {
                self.generate_unsigned_immediate_random_impl(
                    &type_name, sampler, &base_min, &base_max, bit_length,
                )
            } else {
                self.generate_signed_immediate_random_impl(
                    &type_name, sampler, &base_min, &base_max, bit_length,
                )
            }
        } else {
            unreachable!("restricted immediate kinds must provide a sampler or fixed value")
        };

        // Add the generic parameter for restricted types and bind `MemoryAware` through an impl.
        quote! {
            #[doc = #doc_comment]
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DeriveValidatedValue)]
            #[validated(#(#attrs),*)]
            pub struct #type_name<const MEM_AWARE: bool = false>(#inner_type);

            impl<const MEM_AWARE: bool> MemoryAware for #type_name<MEM_AWARE> {
                fn is_memory_aware() -> bool { MEM_AWARE }
            }

            #random_impl
            #mutation_impl
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CodeGenerator;
    use riscv_instruction_schema::{
        ImmediateKind, OperandRestriction, RegisterBank, RegisterEncoding, RegisterSemantics,
    };

    #[test]
    fn restricted_memory_register_types_get_distinct_names_and_domains() {
        let generator = CodeGenerator::new(vec![]);
        let restriction = OperandRestriction {
            min_max: Some((8, 15)),
            ..OperandRestriction::default()
        };

        assert_eq!(
            generator.generate_restricted_register_type_name(
                "CompressedIntegerRegister",
                "xs1",
                &restriction,
                false,
            ),
            "RestrictedCompressedIntegerRegisterXs1Range8To15"
        );
        assert_eq!(
            generator.generate_restricted_register_type_name(
                "CompressedIntegerRegister",
                "xs1",
                &restriction,
                true,
            ),
            "RestrictedCompressedIntegerRegisterXs1Range8To15MemoryRange"
        );
        assert_eq!(
            RegisterBank::Integer.restricted_random_domain(true),
            "memory_register"
        );
    }

    #[test]
    fn restricted_immediate_defs_share_runtime_kinds() {
        let generator = CodeGenerator::new(vec![]);
        let restriction = OperandRestriction::default();

        let shamt = generator.generate_restricted_immediate_type_def(
            ImmediateKind::ShiftAmount,
            "shamt",
            5,
            &restriction,
            false,
        );
        let jalr = generator.generate_restricted_immediate_type_def(
            ImmediateKind::LinearJalr,
            "offset",
            12,
            &restriction,
            false,
        );

        let shamt_tokens = shamt.to_string();
        let jalr_tokens = jalr.to_string();

        assert!(shamt_tokens.contains("sample_shamt_value"));
        assert!(jalr_tokens.contains("self . set"));
        assert!(jalr_tokens.contains("map_err"));
        assert!(!jalr_tokens.contains("sample_"));
    }

    #[test]
    fn restricted_register_defs_use_schema_runtime_metadata() {
        let generator = CodeGenerator::new(vec![]);
        let restriction = OperandRestriction::default();
        let semantics = RegisterSemantics {
            bank: RegisterBank::FloatingPoint,
            encoding: RegisterEncoding::Compressed,
            role: riscv_instruction_schema::RegisterRole::Plain,
        };

        let tokens = generator
            .generate_restricted_register_type_def(semantics, "fs1", &restriction, false)
            .to_string();

        assert!(tokens.contains("Compressed floating point register"));
        assert!(tokens.contains("display = \"f{}\""));
        assert!(tokens.contains("random_domain = \"floating_point_register\""));
    }
}
