use heck::ToSnakeCase;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Data, DeriveInput, LitBool, LitStr, Meta, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned as _,
};

#[derive(Default)]
pub struct ValidatedArgs {
    min: Option<LitStr>,
    max: Option<LitStr>,
    name: Option<LitStr>,
    display: Option<LitStr>,
    multiple_of: Option<LitStr>,
    random_domain: Option<LitStr>,
    skip_display: Option<LitBool>,
    forbidden: Option<LitStr>,
    odd_only: Option<LitBool>,
}

impl Parse for ValidatedArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = Self::default();
        let metas = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;

        for meta in metas {
            match meta {
                Meta::NameValue(nv) => {
                    if nv.path.is_ident("min") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                                args.min = Some(lit_str.clone());
                            }
                        }
                    } else if nv.path.is_ident("max") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                                args.max = Some(lit_str.clone());
                            }
                        }
                    } else if nv.path.is_ident("name") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                                args.name = Some(lit_str.clone());
                            }
                        }
                    } else if nv.path.is_ident("display") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                                args.display = Some(lit_str.clone());
                            }
                        }
                    } else if nv.path.is_ident("multiple_of") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                                args.multiple_of = Some(lit_str.clone());
                            }
                        }
                    } else if nv.path.is_ident("random_domain") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                                args.random_domain = Some(lit_str.clone());
                            }
                        }
                    } else if nv.path.is_ident("forbidden") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                                args.forbidden = Some(lit_str.clone());
                            }
                        }
                    } else if nv.path.is_ident("odd_only") {
                        if let syn::Expr::Lit(expr_lit) = &nv.value {
                            if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                                args.odd_only =
                                    Some(LitBool::new(lit_str.value() == "true", lit_str.span()));
                            }
                        }
                    }
                }
                Meta::Path(path) => {
                    if path.is_ident("skip_display") {
                        args.skip_display = Some(LitBool::new(true, path.span()));
                    } else if path.is_ident("odd_only") {
                        args.odd_only = Some(LitBool::new(true, path.span()));
                    }
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        meta,
                        "Unsupported attribute format",
                    ));
                }
            }
        }

        Ok(args)
    }
}

fn is_unsigned_inner_type(inner_type: &syn::Type) -> bool {
    match inner_type {
        syn::Type::Path(type_path) => type_path.path.segments.last().is_some_and(|segment| {
            matches!(
                segment.ident.to_string().as_str(),
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
            )
        }),
        _ => false,
    }
}

/// Generate the implementation for the `ValidatedValue` trait.
pub fn impl_validated_value(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let attr = ast
        .attrs
        .iter()
        .find(|a| a.path().is_ident("validated"))
        .expect("`validated` attribute is required for deriving `ValidatedValue`");
    let args: ValidatedArgs = attr
        .parse_args()
        .expect("Failed to parse `validated` attribute arguments");

    let inner_type = if let Data::Struct(s) = &ast.data {
        if let syn::Fields::Unnamed(f) = &s.fields {
            if f.unnamed.len() == 1 {
                f.unnamed.first().unwrap().ty.clone()
            } else {
                panic!("DeriveValidatedValue only supports newtype structs with one field");
            }
        } else {
            panic!("DeriveValidatedValue only supports newtype structs");
        }
    } else {
        panic!("DeriveValidatedValue can only be used on structs");
    };

    let min_expr: TokenStream = args
        .min
        .as_ref()
        .expect("`min` is required")
        .value()
        .parse()
        .unwrap();
    let max_expr: TokenStream = args
        .max
        .as_ref()
        .expect("`max` is required")
        .value()
        .parse()
        .unwrap();

    let name_in_msg = args
        .name
        .as_ref()
        .map(|n| n.value())
        .unwrap_or_else(|| name.to_string().to_snake_case().replace('_', " "));
    let error_msg = format!("{} {{}} exceeds range ({{}} to {{}})", name_in_msg);

    let multiple_check = if let Some(multiple_str) = &args.multiple_of {
        let multiple_expr: TokenStream = multiple_str.value().parse().unwrap();
        let err_msg = format!(
            "{} must be a multiple of {}",
            name_in_msg,
            multiple_str.value()
        );
        quote! {
            let multiple = #multiple_expr;
            if value % multiple != 0 {
                return Err(#err_msg.to_string());
            }
        }
    } else {
        quote! {}
    };

    let odd_only_check = if args.odd_only.as_ref().map_or(false, |b| b.value) {
        let err_msg = format!("{} must be odd", name_in_msg);
        quote! {
            if value % 2 == 0 {
                return Err(#err_msg.to_string());
            }
        }
    } else {
        quote! {}
    };

    let (forbidden_values, forbidden_check) = if let Some(forbidden_str) = &args.forbidden {
        let forbidden_str_value = forbidden_str.value();

        // Parse forbidden values according to the inner type.
        let forbidden_values: Vec<TokenStream> = if forbidden_str_value.is_empty() {
            Vec::new()
        } else {
            forbidden_str_value
                .split(',')
                .filter_map(|s| {
                    let trimmed = s.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        // Use the literal directly and let the compiler infer the type.
                        let value_token: TokenStream = trimmed.parse().unwrap();
                        Some(value_token)
                    }
                })
                .collect()
        };

        let err_msg = format!(
            "{} value {{}} is forbidden for this instruction",
            name_in_msg
        );
        let forbidden_check = if !forbidden_values.is_empty() {
            quote! {
                if Self::FORBIDDEN.contains(&value) {
                    return Err(format!(#err_msg, value));
                }
            }
        } else {
            quote! {}
        };

        (forbidden_values, forbidden_check)
    } else {
        (Vec::new(), quote! {})
    };

    let skip_display = args.skip_display.as_ref().map_or(false, |b| b.value);

    let display_impl = if !skip_display {
        let display_format = args
            .display
            .as_ref()
            .map(|d| d.value())
            .unwrap_or("{}".to_string());

        quote! {
            impl #impl_generics ::riscv_instruction_types::AsmEncode for #name #ty_generics #where_clause {
                fn encode_asm(&self) -> String {
                    format!(#display_format, self.0)
                }
            }

            impl #impl_generics ::std::fmt::Display for #name #ty_generics #where_clause {
                fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                    f.write_str(&::riscv_instruction_types::AsmEncode::encode_asm(self))
                }
            }
        }
    } else {
        quote! {}
    };

    let parse_operand_impl = if !skip_display {
        let display_format = args
            .display
            .as_ref()
            .map(|d| d.value())
            .unwrap_or("{}".to_string());

        let parse_value = if let Some((prefix, suffix)) = display_format.split_once("{}") {
            let expected_format = display_format.clone();
            quote! {
                let trimmed = input.trim();
                let value_text = trimmed
                    .strip_prefix(#prefix)
                    .and_then(|rest| rest.strip_suffix(#suffix))
                    .ok_or_else(|| ::riscv_instruction_types::AsmParseError::invalid_operand(
                        stringify!(#name),
                        format!(
                            "expected operand matching display format `{}`, got `{}`",
                            #expected_format,
                            trimmed,
                        ),
                    ))?;
                let value = <#inner_type as ::riscv_instruction_types::AsmScalar>::parse_asm_scalar(value_text)?;
            }
        } else {
            quote! {
                let value = <#inner_type as ::riscv_instruction_types::AsmScalar>::parse_asm_scalar(input)?;
            }
        };

        quote! {
            impl #impl_generics ::riscv_instruction_types::AsmDecode for #name #ty_generics #where_clause {
                fn decode_asm(
                    input: &str,
                ) -> Result<Self, ::riscv_instruction_types::AsmParseError> {
                    #parse_value
                    <Self as ::riscv_instruction_types::ValidatedValue<#inner_type>>::new(value)
                        .map_err(|err| ::riscv_instruction_types::AsmParseError::invalid_operand(
                            stringify!(#name),
                            err,
                        ))
                }
            }

            impl #impl_generics ::std::str::FromStr for #name #ty_generics #where_clause {
                type Err = ::riscv_instruction_types::AsmParseError;
                fn from_str(s: &str) -> Result<Self, Self::Err> {
                    <Self as ::riscv_instruction_types::AsmDecode>::decode_asm(s)
                }
            }
        }
    } else {
        quote! {}
    };

    let multiple_of_value = if let Some(multiple_str) = &args.multiple_of {
        let multiple_expr: TokenStream = multiple_str.value().parse().unwrap();
        quote! { Some(#multiple_expr) }
    } else {
        quote! { None }
    };

    let odd_only_value = if args.odd_only.as_ref().map_or(false, |b| b.value) {
        quote! { true }
    } else {
        quote! { false }
    };

    let validated_value_ident = quote!(::riscv_instruction_types::ValidatedValue<#inner_type>);
    let random_range_trait_ident =
        quote!(::riscv_instruction_types::ValidatedValueRandomRange<#inner_type>);
    let random_error_ident = quote!(::riscv_instruction_types::RandomGenerationError);
    let memory_aware_ident = quote!(::riscv_instruction_types::MemoryAware);
    let inner_is_unsigned = is_unsigned_inner_type(&inner_type);

    let configured_range_body = match args.random_domain.as_ref().map(|d| d.value()) {
        Some(domain) if domain == "integer_register" => quote! {
            let (config_min, config_max) = config.register_operand_config.integer_register_number_range;
            let actual_min = <Self as #validated_value_ident>::MIN.max(config_min as #inner_type);
            let actual_max = <Self as #validated_value_ident>::MAX.min(config_max as #inner_type);
            Ok((actual_min, actual_max))
        },
        Some(domain) if domain == "floating_point_register" => quote! {
            let (config_min, config_max) = config.register_operand_config.floating_point_register_number_range;
            let actual_min = <Self as #validated_value_ident>::MIN.max(config_min as #inner_type);
            let actual_max = <Self as #validated_value_ident>::MAX.min(config_max as #inner_type);
            Ok((actual_min, actual_max))
        },
        Some(domain) if domain == "vector_register" => quote! {
            let (config_min, config_max) = config.register_operand_config.vector_register_number_range;
            let actual_min = <Self as #validated_value_ident>::MIN.max(config_min as #inner_type);
            let actual_max = <Self as #validated_value_ident>::MAX.min(config_max as #inner_type);
            Ok((actual_min, actual_max))
        },
        Some(domain) if domain == "memory_register" => quote! {
            let (config_min, config_max) = config.memory_access_config.memory_base_register_number_range;
            let actual_min = <Self as #validated_value_ident>::MIN.max(config_min as #inner_type);
            let actual_max = <Self as #validated_value_ident>::MAX.min(config_max as #inner_type);
            Ok((actual_min, actual_max))
        },
        Some(domain) if domain == "auto" || domain.is_empty() => {
            if inner_is_unsigned {
                quote! {
                    if <Self as #memory_aware_ident>::is_memory_aware() {
                        let (min_imm, max_imm) = config.memory_access_config.memory_offset_immediate_range;
                        if max_imm < 0 {
                            return Err(#random_error_ident::new(
                                stringify!(#name),
                                "configured max is negative",
                            ));
                        }
                        let config_min = if min_imm < 0 { 0 } else { min_imm as #inner_type };
                        let actual_min = <Self as #validated_value_ident>::MIN.max(config_min);
                        let actual_max = <Self as #validated_value_ident>::MAX.min(max_imm as #inner_type);
                        Ok((actual_min, actual_max))
                    } else {
                        Ok((<Self as #validated_value_ident>::MIN, <Self as #validated_value_ident>::MAX))
                    }
                }
            } else {
                quote! {
                    if <Self as #memory_aware_ident>::is_memory_aware() {
                        let (min_imm, max_imm) = config.memory_access_config.memory_offset_immediate_range;
                        let actual_min = <Self as #validated_value_ident>::MIN.max(min_imm as #inner_type);
                        let actual_max = <Self as #validated_value_ident>::MAX.min(max_imm as #inner_type);
                        Ok((actual_min, actual_max))
                    } else {
                        Ok((<Self as #validated_value_ident>::MIN, <Self as #validated_value_ident>::MAX))
                    }
                }
            }
        }
        Some(other) => {
            return syn::Error::new(
                args.random_domain.as_ref().unwrap().span(),
                format!(
                    "unsupported random_domain '{}', expected one of: auto, integer_register, floating_point_register, vector_register, memory_register",
                    other
                ),
            )
            .to_compile_error();
        }
        None => {
            if inner_is_unsigned {
                quote! {
                    if <Self as #memory_aware_ident>::is_memory_aware() {
                        let (min_imm, max_imm) = config.memory_access_config.memory_offset_immediate_range;
                        if max_imm < 0 {
                            return Err(#random_error_ident::new(
                                stringify!(#name),
                                "configured max is negative",
                            ));
                        }
                        let config_min = if min_imm < 0 { 0 } else { min_imm as #inner_type };
                        let actual_min = <Self as #validated_value_ident>::MIN.max(config_min);
                        let actual_max = <Self as #validated_value_ident>::MAX.min(max_imm as #inner_type);
                        Ok((actual_min, actual_max))
                    } else {
                        Ok((<Self as #validated_value_ident>::MIN, <Self as #validated_value_ident>::MAX))
                    }
                }
            } else {
                quote! {
                    if <Self as #memory_aware_ident>::is_memory_aware() {
                        let (min_imm, max_imm) = config.memory_access_config.memory_offset_immediate_range;
                        let actual_min = <Self as #validated_value_ident>::MIN.max(min_imm as #inner_type);
                        let actual_max = <Self as #validated_value_ident>::MAX.min(max_imm as #inner_type);
                        Ok((actual_min, actual_max))
                    } else {
                        Ok((<Self as #validated_value_ident>::MIN, <Self as #validated_value_ident>::MAX))
                    }
                }
            }
        }
    };

    let random_range_impl = quote! {
        impl #impl_generics #random_range_trait_ident for #name #ty_generics #where_clause {
            fn configured_random_range(
                config: &::riscv_instruction_types::RandomConfig,
            ) -> Result<(#inner_type, #inner_type), #random_error_ident> {
                #configured_range_body
            }
        }
    };

    let sequence_integer_register_impl = match args.random_domain.as_ref().map(|d| d.value()) {
        Some(domain) if matches!(domain.as_str(), "integer_register" | "memory_register") => {
            quote! {
                impl #impl_generics ::riscv_instruction_types::SequenceIntegerRegisterOperand for #name #ty_generics #where_clause {
                    fn sequence_integer_register(
                        self,
                    ) -> Option<::riscv_instruction_types::IntegerRegister> {
                        ::riscv_instruction_types::IntegerRegister::new(
                            <Self as ::riscv_instruction_types::ValidatedValue<#inner_type>>::get(&self),
                        )
                        .ok()
                    }
                }
            }
        }
        _ => quote! {},
    };

    quote! {
        impl #impl_generics ValidatedValue<#inner_type> for #name #ty_generics #where_clause {
            const MIN: #inner_type = #min_expr;
            const MAX: #inner_type = #max_expr;
            const MULTIPLE_OF: Option<#inner_type> = #multiple_of_value;
            const FORBIDDEN: &'static [#inner_type] = &[#(#forbidden_values),*];
            const ODD_ONLY: bool = #odd_only_value;
            type Error = String;

            fn new(value: #inner_type) -> Result<Self, Self::Error> {
                #forbidden_check
                #multiple_check
                #odd_only_check
                if value >= Self::MIN && value <= Self::MAX {
                    Ok(Self(value))
                } else {
                    Err(format!(
                        #error_msg,
                        value,
                        Self::MIN,
                        Self::MAX
                    ))
                }
            }

            fn get(&self) -> #inner_type {
                self.0
            }

            fn set(&mut self, value: #inner_type) -> Result<(), Self::Error> {
                *self = Self::new(value)?;
                Ok(())
            }
        }

        #display_impl
        #parse_operand_impl
        #random_range_impl
        #sequence_integer_register_impl
    }
}
