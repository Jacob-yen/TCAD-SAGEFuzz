use super::CodeGenerator;
use proc_macro2::TokenStream;
use riscv_instruction_schema::{ISABase, Instruction, Operand, RegisterRole};
use std::collections::HashSet;

impl CodeGenerator {
    /// Generate all restricted type definitions for separated mode.
    pub fn generate_all_restricted_type_definitions_for_separated(
        &self,
        instructions: &[Instruction],
    ) -> (TokenStream, TokenStream) {
        let mut restricted_register_defs = TokenStream::new();
        let mut restricted_immediate_defs = TokenStream::new();
        let mut processed_register_combinations = HashSet::new();
        let mut processed_immediate_combinations = HashSet::new();

        for instruction in instructions {
            for operand in &instruction.operands {
                // In separated mode, every ISA base supported by the instruction must be considered.
                for isa_base in &instruction.isa_bases {
                    self.process_operand_for_restricted_type_defs(
                        operand,
                        isa_base,
                        instruction,
                        &mut processed_register_combinations,
                        &mut processed_immediate_combinations,
                        &mut restricted_register_defs,
                        &mut restricted_immediate_defs,
                    );
                }
            }
        }
        (restricted_register_defs, restricted_immediate_defs)
    }

    /// Process one operand and emit restricted type definitions when needed.
    pub fn process_operand_for_restricted_type_defs(
        &self,
        operand: &Operand,
        isa_base: &ISABase,
        instruction: &Instruction,
        processed_register_combinations: &mut HashSet<String>,
        processed_immediate_combinations: &mut HashSet<String>,
        restricted_register_defs: &mut TokenStream,
        restricted_immediate_defs: &mut TokenStream,
    ) {
        if let Some(restriction) = operand
            .restrictions_for(*isa_base)
            .filter(|restriction| restriction.has_explicit_constraints())
        {
            if let Some(register_semantics) = operand.register_semantics(*isa_base, instruction) {
                let base_type = register_semantics.plain_runtime_type_name();
                let use_memory_register_range = register_semantics.role == RegisterRole::MemoryBase;
                let type_name_key = self.generate_restricted_register_type_name(
                    base_type,
                    &operand.name,
                    restriction,
                    use_memory_register_range,
                );
                if processed_register_combinations.insert(type_name_key) {
                    let type_def = self.generate_restricted_register_type_def(
                        register_semantics,
                        &operand.name,
                        restriction,
                        use_memory_register_range,
                    );
                    restricted_register_defs.extend(type_def);
                }
                return;
            }

            if let Some(immediate_kind) = operand
                .runtime_immediate_kind(instruction)
                .filter(|kind| kind.is_restricted_runtime_family())
            {
                let bit_length = operand.bit_length_for(*isa_base);
                let is_memory_aware = instruction.is_memory_offset_operand(&operand.name);
                let type_name_key = self.generate_restricted_immediate_type_name(
                    immediate_kind,
                    &operand.name,
                    bit_length,
                    restriction,
                );

                if processed_immediate_combinations.insert(type_name_key) {
                    let type_def = self.generate_restricted_immediate_type_def(
                        immediate_kind,
                        &operand.name,
                        bit_length,
                        restriction,
                        is_memory_aware,
                    );
                    restricted_immediate_defs.extend(type_def);
                }
            }
        }
    }
}
