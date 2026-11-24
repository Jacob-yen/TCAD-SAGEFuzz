use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Data, DeriveInput, Field, Fields, FieldsNamed, Ident, Meta, Token, punctuated::Punctuated,
    spanned::Spanned as _,
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct SequenceArgs {
    memory_base: Option<String>,
    memory_base_fixed: Option<u8>,
    csr: Option<String>,
    csr_source: Option<String>,
    pc_relative: Option<String>,
    pc_relative_fixed: Option<u8>,
    touches_floating_point: bool,
}

fn parse_string_literal_value(value: &syn::Expr, key: &str) -> syn::Result<String> {
    let syn::Expr::Lit(expr_lit) = value else {
        return Err(syn::Error::new(
            value.span(),
            format!("`{key}` must be a string literal"),
        ));
    };
    let syn::Lit::Str(lit_str) = &expr_lit.lit else {
        return Err(syn::Error::new(
            expr_lit.lit.span(),
            format!("`{key}` must be a string literal"),
        ));
    };

    Ok(lit_str.value())
}

fn parse_bool_literal_value(value: &syn::Expr, key: &str) -> syn::Result<bool> {
    let syn::Expr::Lit(expr_lit) = value else {
        return Err(syn::Error::new(
            value.span(),
            format!("`{key}` must be a boolean literal"),
        ));
    };
    let syn::Lit::Bool(lit_bool) = &expr_lit.lit else {
        return Err(syn::Error::new(
            expr_lit.lit.span(),
            format!("`{key}` must be a boolean literal"),
        ));
    };

    Ok(lit_bool.value)
}

fn parse_fixed_register_number(value: &str, span: proc_macro2::Span) -> syn::Result<Option<u8>> {
    let normalized = value.trim().to_ascii_lowercase();
    let register_number = match normalized.as_str() {
        "pc" => return Ok(None),
        "zero" => 0,
        "ra" => 1,
        "sp" => 2,
        "gp" => 3,
        "tp" => 4,
        "t0" => 5,
        "t1" => 6,
        "t2" => 7,
        "s0" | "fp" => 8,
        "s1" => 9,
        "a0" => 10,
        "a1" => 11,
        "a2" => 12,
        "a3" => 13,
        "a4" => 14,
        "a5" => 15,
        "a6" => 16,
        "a7" => 17,
        "s2" => 18,
        "s3" => 19,
        "s4" => 20,
        "s5" => 21,
        "s6" => 22,
        "s7" => 23,
        "s8" => 24,
        "s9" => 25,
        "s10" => 26,
        "s11" => 27,
        "t3" => 28,
        "t4" => 29,
        "t5" => 30,
        "t6" => 31,
        _ => {
            let Some(number) = normalized.strip_prefix('x') else {
                return Err(syn::Error::new(
                    span,
                    format!(
                        "unsupported fixed register '{}', expected x0-x31 or an integer ABI alias",
                        value
                    ),
                ));
            };

            number.parse::<u8>().map_err(|_| {
                syn::Error::new(
                    span,
                    format!(
                        "unsupported fixed register '{}', expected x0-x31 or an integer ABI alias",
                        value
                    ),
                )
            })?
        }
    };

    if register_number > 31 {
        return Err(syn::Error::new(
            span,
            format!("fixed register '{}' is out of range", value),
        ));
    }

    Ok(Some(register_number))
}

fn parse_sequence_args(input_ast: &DeriveInput) -> syn::Result<SequenceArgs> {
    let Some(attr) = input_ast
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("sequence"))
    else {
        return Ok(SequenceArgs::default());
    };

    let metas = attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;
    let mut args = SequenceArgs::default();

    for meta in metas {
        match meta {
            Meta::Path(path) if path.is_ident("touches_floating_point") => {
                args.touches_floating_point = true;
            }
            Meta::NameValue(name_value) => {
                if name_value.path.is_ident("memory_base") {
                    args.memory_base = Some(parse_string_literal_value(
                        &name_value.value,
                        "memory_base",
                    )?);
                } else if name_value.path.is_ident("memory_base_fixed") {
                    let raw_value =
                        parse_string_literal_value(&name_value.value, "memory_base_fixed")?;
                    args.memory_base_fixed =
                        parse_fixed_register_number(&raw_value, name_value.value.span())?;
                } else if name_value.path.is_ident("csr") {
                    args.csr = Some(parse_string_literal_value(&name_value.value, "csr")?);
                } else if name_value.path.is_ident("csr_source") {
                    args.csr_source =
                        Some(parse_string_literal_value(&name_value.value, "csr_source")?);
                } else if name_value.path.is_ident("pc_relative") {
                    args.pc_relative = Some(parse_string_literal_value(
                        &name_value.value,
                        "pc_relative",
                    )?);
                } else if name_value.path.is_ident("pc_relative_fixed") {
                    let raw_value =
                        parse_string_literal_value(&name_value.value, "pc_relative_fixed")?;
                    args.pc_relative_fixed =
                        parse_fixed_register_number(&raw_value, name_value.value.span())?;
                } else if name_value.path.is_ident("touches_floating_point") {
                    args.touches_floating_point =
                        parse_bool_literal_value(&name_value.value, "touches_floating_point")?;
                }
            }
            _ => {}
        }
    }

    Ok(args)
}

fn find_named_field<'a>(fields_named: &'a FieldsNamed, field_name: &str) -> syn::Result<&'a Field> {
    fields_named
        .named
        .iter()
        .find(|field| {
            field
                .ident
                .as_ref()
                .is_some_and(|ident| ident == field_name)
        })
        .ok_or_else(|| {
            syn::Error::new(
                fields_named.span(),
                format!("unknown field `{field_name}` in #[sequence(...)]"),
            )
        })
}

/// Generate the implementation for the `Random` trait.
pub fn impl_random_derive(input_ast: &DeriveInput) -> TokenStream {
    let name: &Ident = &input_ast.ident;
    let generics = &input_ast.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let random_impl =
        generate_random_impl(input_ast, name, &impl_generics, &ty_generics, &where_clause);
    let sequence_hints_impl =
        generate_sequence_hints_impl(input_ast, name, &impl_generics, &ty_generics, &where_clause);
    let random_sequence_impl =
        generate_random_sequence_impl(name, &impl_generics, &ty_generics, &where_clause);

    quote! {
        #random_impl
        #sequence_hints_impl
        #random_sequence_impl
    }
}

/// Generate the `Random` trait implementation.
fn generate_random_impl(
    input_ast: &DeriveInput,
    name: &Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: &Option<&syn::WhereClause>,
) -> TokenStream {
    let random_trait_ident = quote!(::riscv_instruction_types::Random);
    let random_config_ident = quote!(::riscv_instruction_types::RandomConfig);
    let random_error_ident = quote!(::riscv_instruction_types::RandomGenerationError);
    let random_range_trait_ident = quote!(::riscv_instruction_types::ValidatedValueRandomRange);

    match &input_ast.data {
        Data::Enum(data_enum) => {
            let variants_data = &data_enum.variants;

            if variants_data.is_empty() {
                return syn::Error::new_spanned(
                    name,
                    "Cannot derive Random for an enum with no variants",
                )
                .to_compile_error();
            }

            let num_variants = variants_data.len();
            let mut match_arms = TokenStream::new();

            for (idx, variant) in variants_data.iter().enumerate() {
                let variant_ident: &Ident = &variant.ident;
                let mut field_initializers = TokenStream::new();

                match &variant.fields {
                    Fields::Named(fields_named) => {
                        for field in fields_named.named.iter() {
                            let field_name_ident = field
                                .ident
                                .as_ref()
                                .expect("Named field must have an identifier");
                            let field_ty = &field.ty;
                            field_initializers.extend(quote! {
                                #field_name_ident: <#field_ty as #random_trait_ident>::random_with_rng(rng, config)?,
                            });
                        }
                        match_arms.extend(quote! {
                            #idx => Ok(#name::#variant_ident { #field_initializers }),
                        });
                    }
                    Fields::Unnamed(fields_unnamed) => {
                        let mut tuple_fields = TokenStream::new();
                        for field in fields_unnamed.unnamed.iter() {
                            let field_ty = &field.ty;
                            tuple_fields.extend(quote! {
                                <#field_ty as #random_trait_ident>::random_with_rng(rng, config)?,
                            });
                        }
                        match_arms.extend(quote! {
                            #idx => Ok(#name::#variant_ident(#tuple_fields)),
                        });
                    }
                    Fields::Unit => {
                        match_arms.extend(quote! {
                            #idx => Ok(#name::#variant_ident),
                        });
                    }
                }
            }

            quote! {
                impl #impl_generics #random_trait_ident for #name #ty_generics #where_clause {
                    type Output = Self;

                    fn random_with_rng<R: rand::Rng>(
                        rng: &mut R,
                        config: &#random_config_ident,
                    ) -> Result<Self::Output, #random_error_ident> {
                        let variant_idx = rng.gen_range(0..#num_variants);
                        match variant_idx {
                            #match_arms
                            _=> Err(#random_error_ident::new(
                                stringify!(#name),
                                "variant index out of bounds, this should never happen if num_variants is correct",
                            )),
                        }
                    }
                }
            }
        }
        Data::Struct(data_struct) => {
            match &data_struct.fields {
                Fields::Named(fields_named) => {
                    // For structs with fields, generate a random value for each field.
                    let field_initializers = fields_named.named.iter().map(|field| {
                        let field_name = field.ident.as_ref().unwrap();
                        let field_ty = &field.ty;
                        quote! {
                            #field_name: <#field_ty as #random_trait_ident>::random_with_rng(rng, config)?
                        }
                    });

                    quote! {
                        impl #impl_generics #random_trait_ident for #name #ty_generics #where_clause {
                            type Output = Self;

                            fn random_with_rng<R: rand::Rng>(
                                rng: &mut R,
                                config: &#random_config_ident,
                            ) -> Result<Self::Output, #random_error_ident> {
                                Ok(Self {
                                    #(#field_initializers),*
                                })
                            }
                        }
                    }
                }
                Fields::Unnamed(fields_unnamed) => {
                    // For tuple structs, check whether this is a `ValidatedValue` newtype.
                    if fields_unnamed.unnamed.len() == 1 {
                        let inner_type = &fields_unnamed.unnamed.first().unwrap().ty;

                        // This may be a `ValidatedValue` newtype, so emit a specialized random implementation.
                        quote! {
                            impl #impl_generics #random_trait_ident for #name #ty_generics #where_clause {
                                type Output = Self;

                                fn random_with_rng<R: rand::Rng>(
                                    rng: &mut R,
                                    config: &#random_config_ident,
                                ) -> Result<Self::Output, #random_error_ident> {
                                    use rand::Rng;

                                    let (range_start, range_end) =
                                        <Self as #random_range_trait_ident<#inner_type>>::configured_random_range(config)?;

                                    if range_start > range_end {
                                        return Err(#random_error_ident::new(
                                            stringify!(#name),
                                            format!(
                                                "invalid combined range [{}, {}] computed from configuration",
                                                range_start, range_end
                                            ),
                                        ));
                                    }

                                    const MAX_ATTEMPTS: usize = 10000;
                                    let mut last_error: Option<String> = None;
                                    for _ in 0..MAX_ATTEMPTS {
                                        // Decide whether to generate an edge-case value.
                                        let use_boundary_value = if config.edge_case_probability <= 0.0 {
                                            false
                                        } else if config.edge_case_probability >= 1.0 {
                                            true
                                        } else {
                                            rng.gen_range(0.0..1.0) < config.edge_case_probability
                                        };

                                        let random_value = if use_boundary_value {
                                            // Build the edge-case candidate set.
                                            let mut boundary_candidates = Vec::new();

                                            // Basic edge-case values.
                                            boundary_candidates.push(Self::MIN);
                                            boundary_candidates.push(Self::MAX);

                                            // Add additional likely edge cases, using saturating ops to avoid overflow.
                                            if let Ok(zero) = Self::new(0) {
                                                boundary_candidates.push(0);
                                            }
                                            if let Ok(one) = Self::new(1) {
                                                boundary_candidates.push(1);
                                            }
                                            if let Ok(two) = Self::new(2) {
                                                boundary_candidates.push(2);
                                            }
                                            // Try adding `MAX - 1` and `MIN + 1`.
                                            if Self::MAX > Self::MIN {
                                                if let Ok(max_minus_one) = Self::new(Self::MAX.saturating_sub(1)) {
                                                    boundary_candidates.push(Self::MAX.saturating_sub(1));
                                                }
                                                if let Ok(min_plus_one) = Self::new(Self::MIN.saturating_add(1)) {
                                                    boundary_candidates.push(Self::MIN.saturating_add(1));
                                                }
                                            }

                                            // Range boundary values.
                                            boundary_candidates.push(range_start);
                                            boundary_candidates.push(range_end);

                                            // Range midpoint.
                                            if range_end > range_start {
                                                let mid = range_start + (range_end - range_start) / 2;
                                                boundary_candidates.push(mid);
                                            }

                                            // Filter candidates: deduplicate, exclude forbidden values, and enforce constraints.
                                            let mut valid_boundaries: Vec<_> = boundary_candidates
                                                .into_iter()
                                                .filter(|&v| v >= range_start && v <= range_end)
                                                .filter(|&v| !Self::FORBIDDEN.contains(&v))
                                                .collect();

                                            // Deduplicate.
                                            valid_boundaries.sort_unstable();
                                            valid_boundaries.dedup();

                                            // Randomly choose from the valid edge-case values.
                                            if !valid_boundaries.is_empty() {
                                                *valid_boundaries.get(rng.gen_range(0..valid_boundaries.len())).unwrap()
                                            } else {
                                                // Fall back to normal generation when no valid edge-case values remain.
                                                if range_start == range_end {
                                                    range_start
                                                } else {
                                                    rng.gen_range(range_start..=range_end)
                                                }
                                            }
                                        } else if let Some(multiple) = Self::MULTIPLE_OF {
                                            let min_multiple = if range_start >= 0 {
                                                (range_start + multiple - 1) / multiple
                                            } else {
                                                range_start / multiple
                                            };
                                            let max_multiple = range_end / multiple;

                                            if min_multiple <= max_multiple {
                                                let multiple_factor = if min_multiple == max_multiple {
                                                    min_multiple
                                                } else {
                                                    rng.gen_range(min_multiple..=max_multiple)
                                                };
                                                multiple_factor * multiple
                                            } else {
                                                if range_start == range_end {
                                                    range_start
                                                } else {
                                                    rng.gen_range(range_start..=range_end)
                                                }
                                            }
                                        } else if Self::ODD_ONLY {
                                            // Generate an odd value.
                                            // Ensure the starting value is odd.
                                            let odd_start = if range_start % 2 == 0 {
                                                range_start + 1
                                            } else {
                                                range_start
                                            };

                                            // Ensure the ending value is odd.
                                            let odd_end = if range_end % 2 == 0 {
                                                range_end - 1
                                            } else {
                                                range_end
                                            };

                                            if odd_start <= odd_end {
                                                // Count the number of odd values.
                                                let odd_count = (odd_end - odd_start) / 2 + 1;
                                                let random_index = rng.gen_range(0..odd_count);
                                                odd_start + random_index * 2
                                            } else {
                                                // Fall back to ranged generation when no valid odd values exist.
                                                if range_start == range_end {
                                                    range_start
                                                } else {
                                                    rng.gen_range(range_start..=range_end)
                                                }
                                            }
                                        } else {
                                            if range_start == range_end {
                                                range_start
                                            } else {
                                                rng.gen_range(range_start..=range_end)
                                            }
                                        };

                                        if Self::FORBIDDEN.contains(&random_value) {
                                            continue;
                                        }

                                        match Self::new(random_value) {
                                            Ok(instance) => return Ok(instance),
                                            Err(err) => {
                                                last_error = Some(err);
                                            }
                                        }
                                    }
                                    Err(#random_error_ident::exhausted(
                                        stringify!(#name),
                                        MAX_ATTEMPTS,
                                        last_error,
                                    ))
                                }
                            }
                        }
                    } else {
                        // For tuple structs with multiple fields, generate a random value for each field.
                        let field_initializers = fields_unnamed.unnamed.iter().map(|field| {
                            let field_ty = &field.ty;
                            quote! {
                                <#field_ty as #random_trait_ident>::random_with_rng(rng, config)?
                            }
                        });

                        quote! {
                            impl #impl_generics #random_trait_ident for #name #ty_generics #where_clause {
                                type Output = Self;

                                fn random_with_rng<R: rand::Rng>(
                                    rng: &mut R,
                                    config: &#random_config_ident,
                                ) -> Result<Self::Output, #random_error_ident> {
                                    Ok(Self(#(#field_initializers),*))
                                }
                            }
                        }
                    }
                }
                Fields::Unit => {
                    // For unit structs, return the struct directly.
                    quote! {
                        impl #impl_generics #random_trait_ident for #name #ty_generics #where_clause {
                            type Output = Self;

                            fn random_with_rng<R: rand::Rng>(
                                _rng: &mut R,
                                _config: &#random_config_ident,
                            ) -> Result<Self::Output, #random_error_ident> {
                                Ok(Self)
                            }
                        }
                    }
                }
            }
        }
        Data::Union(_) => {
            syn::Error::new_spanned(name, "Random derive macro cannot be used on unions.")
                .to_compile_error()
        }
    }
}

fn generate_sequence_hints_impl(
    input_ast: &DeriveInput,
    name: &Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: &Option<&syn::WhereClause>,
) -> TokenStream {
    let sequence_hints_trait_ident = quote!(::riscv_instruction_types::InstructionSequenceHints);
    let sequence_register_operand_ident =
        quote!(::riscv_instruction_types::SequenceIntegerRegisterOperand);
    let integer_register_ident = quote!(::riscv_instruction_types::IntegerRegister);
    let csr_setup_ident = quote!(::riscv_instruction_types::CsrWriteSequenceSetup);

    let sequence_args = match parse_sequence_args(input_ast) {
        Ok(args) => args,
        Err(err) => return err.to_compile_error(),
    };

    match &input_ast.data {
        Data::Struct(data_struct) => {
            let fields_named = match &data_struct.fields {
                Fields::Named(fields_named) => fields_named,
                Fields::Unnamed(_) | Fields::Unit => {
                    if sequence_args != SequenceArgs::default() {
                        return syn::Error::new_spanned(
                            name,
                            "`#[sequence(...)]` is only supported on structs with named fields",
                        )
                        .to_compile_error();
                    }

                    return quote! {
                        impl #impl_generics #sequence_hints_trait_ident for #name #ty_generics #where_clause {}
                    };
                }
            };

            let memory_base_field_ident = match &sequence_args.memory_base {
                Some(field_name) => match find_named_field(fields_named, field_name) {
                    Ok(field) => field.ident.clone(),
                    Err(err) => return err.to_compile_error(),
                },
                None => None,
            };
            let csr_field_ident = match &sequence_args.csr {
                Some(field_name) => match find_named_field(fields_named, field_name) {
                    Ok(field) => field.ident.clone(),
                    Err(err) => return err.to_compile_error(),
                },
                None => None,
            };
            let csr_source_field_ident = match &sequence_args.csr_source {
                Some(field_name) => match find_named_field(fields_named, field_name) {
                    Ok(field) => field.ident.clone(),
                    Err(err) => return err.to_compile_error(),
                },
                None => None,
            };
            let pc_relative_field_ident = match &sequence_args.pc_relative {
                Some(field_name) => match find_named_field(fields_named, field_name) {
                    Ok(field) => field.ident.clone(),
                    Err(err) => return err.to_compile_error(),
                },
                None => None,
            };

            let mut memory_register_pushes = Vec::new();
            if let Some(field_ident) = memory_base_field_ident {
                memory_register_pushes.push(quote! {
                    registers.push(
                        #sequence_register_operand_ident::sequence_integer_register(self.#field_ident)
                            .expect("sequence memory_base must map to an integer register"),
                    );
                });
            }
            if let Some(register_number) = sequence_args.memory_base_fixed {
                memory_register_pushes.push(quote! {
                    registers.push(
                        #integer_register_ident::new(#register_number)
                            .expect("sequence memory_base_fixed must be a valid integer register"),
                    );
                });
            }

            let csr_write_setup_body = match (csr_field_ident, csr_source_field_ident) {
                (Some(csr_field_ident), Some(csr_source_field_ident)) => quote! {
                    Some(#csr_setup_ident {
                        csr: self.#csr_field_ident,
                        rs1: #sequence_register_operand_ident::sequence_integer_register(self.#csr_source_field_ident)
                            .expect("sequence csr_source must map to an integer register"),
                    })
                },
                _ => quote! { None },
            };

            let pc_relative_body = match (pc_relative_field_ident, sequence_args.pc_relative_fixed)
            {
                (Some(field_ident), _) => quote! {
                    #sequence_register_operand_ident::sequence_integer_register(self.#field_ident)
                },
                (None, Some(register_number)) => quote! {
                    #integer_register_ident::new(#register_number).ok()
                },
                (None, None) => quote! { None },
            };

            let touches_floating_point = sequence_args.touches_floating_point;

            quote! {
                impl #impl_generics #sequence_hints_trait_ident for #name #ty_generics #where_clause {
                    fn sequence_memory_base_registers(&self) -> Vec<#integer_register_ident> {
                        let mut registers = Vec::new();
                        #(#memory_register_pushes)*
                        registers
                    }

                    fn sequence_csr_write_setup(&self) -> Option<#csr_setup_ident> {
                        #csr_write_setup_body
                    }

                    fn sequence_pc_relative_register(&self) -> Option<#integer_register_ident> {
                        #pc_relative_body
                    }

                    fn sequence_touches_floating_point(&self) -> bool {
                        #touches_floating_point
                    }
                }
            }
        }
        Data::Enum(data_enum) => {
            if sequence_args != SequenceArgs::default() {
                return syn::Error::new_spanned(
                    name,
                    "`#[sequence(...)]` is only supported on structs with named fields",
                )
                .to_compile_error();
            }

            let memory_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_ident = &variant.ident;
                match &variant.fields {
                    Fields::Unnamed(fields_unnamed) if fields_unnamed.unnamed.len() == 1 => {
                        quote! { Self::#variant_ident(inner) => #sequence_hints_trait_ident::sequence_memory_base_registers(inner) }
                    }
                    Fields::Named(_) => quote! { Self::#variant_ident { .. } => Vec::new() },
                    Fields::Unnamed(_) => quote! { Self::#variant_ident(..) => Vec::new() },
                    Fields::Unit => quote! { Self::#variant_ident => Vec::new() },
                }
            });
            let csr_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_ident = &variant.ident;
                match &variant.fields {
                    Fields::Unnamed(fields_unnamed) if fields_unnamed.unnamed.len() == 1 => {
                        quote! { Self::#variant_ident(inner) => #sequence_hints_trait_ident::sequence_csr_write_setup(inner) }
                    }
                    Fields::Named(_) => quote! { Self::#variant_ident { .. } => None },
                    Fields::Unnamed(_) => quote! { Self::#variant_ident(..) => None },
                    Fields::Unit => quote! { Self::#variant_ident => None },
                }
            });
            let pc_relative_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_ident = &variant.ident;
                match &variant.fields {
                    Fields::Unnamed(fields_unnamed) if fields_unnamed.unnamed.len() == 1 => {
                        quote! { Self::#variant_ident(inner) => #sequence_hints_trait_ident::sequence_pc_relative_register(inner) }
                    }
                    Fields::Named(_) => quote! { Self::#variant_ident { .. } => None },
                    Fields::Unnamed(_) => quote! { Self::#variant_ident(..) => None },
                    Fields::Unit => quote! { Self::#variant_ident => None },
                }
            });
            let floating_match_arms = data_enum.variants.iter().map(|variant| {
                let variant_ident = &variant.ident;
                match &variant.fields {
                    Fields::Unnamed(fields_unnamed) if fields_unnamed.unnamed.len() == 1 => {
                        quote! { Self::#variant_ident(inner) => #sequence_hints_trait_ident::sequence_touches_floating_point(inner) }
                    }
                    Fields::Named(_) => quote! { Self::#variant_ident { .. } => false },
                    Fields::Unnamed(_) => quote! { Self::#variant_ident(..) => false },
                    Fields::Unit => quote! { Self::#variant_ident => false },
                }
            });

            quote! {
                impl #impl_generics #sequence_hints_trait_ident for #name #ty_generics #where_clause {
                    fn sequence_memory_base_registers(&self) -> Vec<#integer_register_ident> {
                        match self {
                            #(#memory_match_arms),*
                        }
                    }

                    fn sequence_csr_write_setup(&self) -> Option<#csr_setup_ident> {
                        match self {
                            #(#csr_match_arms),*
                        }
                    }

                    fn sequence_pc_relative_register(&self) -> Option<#integer_register_ident> {
                        match self {
                            #(#pc_relative_match_arms),*
                        }
                    }

                    fn sequence_touches_floating_point(&self) -> bool {
                        match self {
                            #(#floating_match_arms),*
                        }
                    }
                }
            }
        }
        Data::Union(_) => {
            syn::Error::new_spanned(name, "Random derive macro cannot be used on unions.")
                .to_compile_error()
        }
    }
}

/// Generate the implementation for the `RandomInstructionSequence` trait.
fn generate_random_sequence_impl(
    name: &Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: &Option<&syn::WhereClause>,
) -> TokenStream {
    let random_sequence_trait_ident = quote!(::riscv_instruction_types::RandomInstructionSequence);
    let random_trait_ident = quote!(::riscv_instruction_types::Random);
    let random_config_ident = quote!(::riscv_instruction_types::RandomConfig);
    let random_error_ident = quote!(::riscv_instruction_types::RandomGenerationError);
    let instruction_sequence_ident = quote!(::riscv_instruction_types::InstructionSequence);

    quote! {
        impl #impl_generics #random_sequence_trait_ident for #name #ty_generics #where_clause {
            type Output = Self;

            fn random_sequence_with_rng<R: rand::Rng>(
                rng: &mut R,
                config: &#random_config_ident
            ) -> Result<#instruction_sequence_ident<Self::Output>, #random_error_ident> {
                let main_instruction = <Self as #random_trait_ident>::random_with_rng(rng, config)?;
                Ok(#instruction_sequence_ident::new(main_instruction))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SequenceArgs, parse_fixed_register_number, parse_sequence_args};
    use proc_macro2::Span;
    use syn::parse_quote;

    #[test]
    fn parse_sequence_args_extracts_structured_hints() {
        let input: syn::DeriveInput = parse_quote! {
            #[derive(DeriveRandom)]
            #[sequence(
                memory_base = "xs1",
                memory_base_fixed = "sp",
                csr = "csr",
                csr_source = "xs1",
                pc_relative = "xs1",
                touches_floating_point = true
            )]
            struct Example {
                xs1: IntegerRegister,
                csr: WritableCsr,
            }
        };

        assert_eq!(
            parse_sequence_args(&input).expect("sequence args should parse"),
            SequenceArgs {
                memory_base: Some("xs1".to_string()),
                memory_base_fixed: Some(2),
                csr: Some("csr".to_string()),
                csr_source: Some("xs1".to_string()),
                pc_relative: Some("xs1".to_string()),
                pc_relative_fixed: None,
                touches_floating_point: true,
            }
        );
    }

    #[test]
    fn fixed_register_parsing_supports_integer_aliases() {
        assert_eq!(
            parse_fixed_register_number("sp", Span::call_site()).expect("sp should parse"),
            Some(2)
        );
        assert_eq!(
            parse_fixed_register_number("a0", Span::call_site()).expect("a0 should parse"),
            Some(10)
        );
        assert_eq!(
            parse_fixed_register_number("PC", Span::call_site()).expect("pc should be ignored"),
            None
        );
    }
}
