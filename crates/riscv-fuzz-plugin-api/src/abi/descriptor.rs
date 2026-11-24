pub fn plugin_descriptor_symbol_name_for_stem(stem: &str) -> String {
    format!(
        "{}{stem}",
        crate::abi::RISCV_FUZZ_PLUGIN_DESCRIPTOR_SYMBOL_PREFIX
    )
}
