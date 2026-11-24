use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields};

/// Derive `AsmEncode` for enums and structs.
/// Supports custom formatting through `#[asm("...")]` and `#[asm_code(...)]`.
pub fn impl_instruction_display(ast: &DeriveInput) -> TokenStream {
    let name = &ast.ident;

    match &ast.data {
        Data::Enum(data) => impl_enum_asm_encode(name, data),
        Data::Struct(data) => impl_struct_asm_encode(name, data, &ast.attrs),
        _ => {
            quote! { compile_error!("DeriveAsmEncode can only be derived for enums and structs"); }
        }
    }
}

/// Generate `AsmEncode` for an enum.
fn impl_enum_asm_encode(name: &syn::Ident, data: &syn::DataEnum) -> TokenStream {
    let variants = &data.variants;

    let match_arms = variants.iter().map(|variant| {
        let variant_ident = &variant.ident;
        let fields = &variant.fields;

        let field_names: Vec<_> = match fields {
            Fields::Named(fields) => fields
                .named
                .iter()
                .map(|f| f.ident.as_ref().unwrap().clone())
                .collect(),
            Fields::Unit => vec![],
            Fields::Unnamed(fields) => (0..fields.unnamed.len())
                .map(|i| syn::Ident::new(&format!("field_{}", i), proc_macro2::Span::call_site()))
                .collect(),
        };

        let match_pattern = match fields {
            Fields::Named(_) => quote! { Self::#variant_ident { #(#field_names),* } },
            Fields::Unit => quote! { Self::#variant_ident },
            Fields::Unnamed(_) => quote! { Self::#variant_ident(#(#field_names),*) },
        };

        // For single-field tuple variants, delegate to the inner type's AsmEncode.
        let arm_body = match fields {
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                let field_name = &field_names[0];
                quote! {
                    ::riscv_instruction_types::AsmEncode::encode_asm(#field_name)
                }
            }
            _ => {
                let default_str = variant.ident.to_string().to_lowercase().replace('_', ".");
                quote! {
                    #default_str.to_string()
                }
            }
        };

        quote! {
            #match_pattern => { #arm_body }
        }
    });

    quote! {
        impl ::riscv_instruction_types::AsmEncode for #name {
            fn encode_asm(&self) -> String {
                match self {
                    #(#match_arms),*
                }
            }
        }

        impl ::std::fmt::Display for #name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(&::riscv_instruction_types::AsmEncode::encode_asm(self))
            }
        }
    }
}

/// Generate `AsmEncode` for a struct.
fn impl_struct_asm_encode(
    name: &syn::Ident,
    data: &syn::DataStruct,
    attrs: &[syn::Attribute],
) -> TokenStream {
    let asm_attr = attrs.iter().find(|attr| attr.path().is_ident("asm"));
    let asm_code_attr = attrs.iter().find(|attr| attr.path().is_ident("asm_code"));

    let fields = &data.fields;

    let field_names: Vec<_> = match fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|f| f.ident.as_ref().unwrap().clone())
            .collect(),
        Fields::Unit => vec![],
        Fields::Unnamed(fields) => (0..fields.unnamed.len())
            .map(|i| syn::Ident::new(&format!("field_{}", i), proc_macro2::Span::call_site()))
            .collect(),
    };

    let encode_body = if let Some(attr) = asm_code_attr {
        match &attr.meta {
            syn::Meta::List(meta_list) => {
                let code_string = meta_list.tokens.to_string();
                let code_content = if code_string.starts_with('"') && code_string.ends_with('"') {
                    &code_string[1..code_string.len() - 1]
                } else {
                    &code_string
                };

                let processed_code = code_content
                    .replace("\\n", "\n")
                    .replace("\\{", "{")
                    .replace("\\}", "}")
                    .replace("\\\"", "\"");

                match processed_code.parse::<TokenStream>() {
                    Ok(code_tokens) => match fields {
                        Fields::Named(_) => {
                            quote! {
                                let Self { #(#field_names),* } = self;
                                format!("{}", #code_tokens)
                            }
                        }
                        Fields::Unnamed(_) => {
                            quote! {
                                let Self(#(#field_names),*) = self;
                                format!("{}", #code_tokens)
                            }
                        }
                        Fields::Unit => {
                            quote! {
                                format!("{}", #code_tokens)
                            }
                        }
                    },
                    Err(_) => {
                        quote! {
                            #processed_code.to_string()
                        }
                    }
                }
            }
            _ => {
                quote! {
                    compile_error!("malformed `asm_code` attribute. Expected `#[asm_code(...)]`").to_string()
                }
            }
        }
    } else if let Some(attr) = asm_attr {
        match attr.parse_args::<syn::LitStr>() {
            Ok(lit_str) => {
                let format_str = lit_str.value();

                if format_str.contains('{') {
                    match fields {
                        Fields::Named(_) => {
                            quote! {
                                let Self { #(#field_names),* } = self;
                                #(let #field_names = #field_names.encode_asm();)*
                                format!(#format_str)
                            }
                        }
                        Fields::Unnamed(_) => {
                            quote! {
                                let Self(#(#field_names),*) = self;
                                #(let #field_names = #field_names.encode_asm();)*
                                format!(#format_str)
                            }
                        }
                        Fields::Unit => {
                            quote! {
                                format!(#format_str)
                            }
                        }
                    }
                } else {
                    quote! {
                        #format_str.to_string()
                    }
                }
            }
            Err(_) => {
                quote! {
                    compile_error!("malformed `asm` attribute. Expected `#[asm(\"...\")]`").to_string()
                }
            }
        }
    } else {
        let default_str = name.to_string().to_lowercase().replace('_', ".");
        quote! {
            #default_str.to_string()
        }
    };

    quote! {
        impl ::riscv_instruction_types::AsmEncode for #name {
            fn encode_asm(&self) -> String {
                #encode_body
            }
        }

        impl ::std::fmt::Display for #name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(&::riscv_instruction_types::AsmEncode::encode_asm(self))
            }
        }
    }
}
