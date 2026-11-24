//! Ordered fixer rules.
//!
//! The sequence here matters. Earlier rules repair raw naming and syntax shape;
//! later rules depend on that normalized shape. The final two cleanup rules must
//! run last because they remove hard-coded placeholders from the syntax set.

mod constraints;
mod register_fixes;
mod syntax;

use riscv_instruction_schema::*;
use std::collections::HashSet;

use super::{audit::InstructionFixAudit, audited_step};

pub(crate) use constraints::assign_operand_kinds;

use constraints::{
    fix_add_uw_missing_operands, fix_aes64ks1i_rnum_range, fix_amocas_q_pair_register_constraints,
    fix_c_mop_n_instruction, fix_isabase_error_instructions, fix_prefetch_offset_alignment,
    fix_zclsd_register_restrictions,
};
use register_fixes::{
    fix_fixed_aqrl_operands, fix_floating_point_register_errors,
    fix_vector_register_errors_wrapper, fix_vsetvli_operand_names,
};
use syntax::{
    fix_clui_instruction, fix_cm_instructions_reg_list_stack_adj, fix_cm_mvsa01_instruction,
    fix_compressed_stack_adjustment_instructions, fix_compressed_stack_pointer_instructions,
    fix_fence_instruction, fix_fli_instruction_operands, fix_floating_point_load_store_addressing,
    fix_hypervisor_load_store_instructions, fix_missing_round_mode, fix_round_mode_removal,
    fix_sp_should_not_in_final_syntax_operands, fix_uimm_instead_of_imm,
    fix_vector_v0_hardcoded_operands,
};

pub(crate) fn apply_special_rules(
    operands_part: &mut String,
    instruction: &mut Instruction,
    final_syntax_operands: &mut HashSet<String>,
    audit: &mut Option<InstructionFixAudit>,
) {
    // Group 1: fix obvious naming and placeholder-shape issues first.
    // Group 1 handles obvious naming and placeholder-shape problems first.
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_vsetvli_operand_names",
        {
            fix_vsetvli_operand_names(instruction);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_fence_instruction",
        {
            fix_fence_instruction(operands_part, instruction, final_syntax_operands);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_clui_instruction",
        {
            fix_clui_instruction(operands_part, instruction, final_syntax_operands);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_round_mode_removal",
        {
            fix_round_mode_removal(operands_part, instruction, final_syntax_operands);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_fli_instruction_operands",
        {
            fix_fli_instruction_operands(operands_part, instruction, final_syntax_operands);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_uimm_instead_of_imm",
        {
            fix_uimm_instead_of_imm(operands_part, instruction, final_syntax_operands);
        }
    );
    // Group 2: once the basic syntax shape is stable, repair register-class
    // mismatches where unified-db used the wrong placeholder family.
    // Group 2 repairs register-class mistakes after the basic syntax shape is stable.
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_floating_point_load_store_addressing",
        {
            fix_floating_point_load_store_addressing(
                operands_part,
                instruction,
                final_syntax_operands,
            );
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_floating_point_register_errors",
        {
            fix_floating_point_register_errors(instruction, operands_part, final_syntax_operands);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_vector_register_errors_wrapper",
        {
            fix_vector_register_errors_wrapper(instruction, operands_part, final_syntax_operands);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_missing_round_mode",
        {
            fix_missing_round_mode(operands_part, instruction, final_syntax_operands);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_compressed_stack_pointer_instructions",
        {
            fix_compressed_stack_pointer_instructions(
                operands_part,
                instruction,
                final_syntax_operands,
            );
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_compressed_stack_adjustment_instructions",
        {
            fix_compressed_stack_adjustment_instructions(operands_part, instruction);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_fixed_aqrl_operands",
        {
            fix_fixed_aqrl_operands(instruction);
        }
    );
    // Group 3: repair semantic constraints and missing metadata that depend on
    // the already-correct operand names.
    // Group 3 repairs semantic constraints and missing metadata that depend on correct operand names.
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_add_uw_missing_operands",
        {
            fix_add_uw_missing_operands(instruction);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_hypervisor_load_store_instructions",
        {
            fix_hypervisor_load_store_instructions(operands_part, instruction);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_amocas_q_pair_register_constraints",
        {
            fix_amocas_q_pair_register_constraints(instruction);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_aes64ks1i_rnum_range",
        {
            fix_aes64ks1i_rnum_range(instruction);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_prefetch_offset_alignment",
        {
            fix_prefetch_offset_alignment(instruction);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_zclsd_register_restrictions",
        {
            fix_zclsd_register_restrictions(instruction);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_isabase_error_instructions",
        {
            fix_isabase_error_instructions(instruction);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_c_mop_n_instruction",
        {
            fix_c_mop_n_instruction(instruction);
        }
    );
    // Group 4: perform structural rewrites that merge multiple raw operands
    // into one higher-level synthetic operand for downstream codegen.
    // Group 4 performs structural rewrites that merge several raw operands into higher-level downstream operands.
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_cm_instructions_reg_list_stack_adj",
        {
            fix_cm_instructions_reg_list_stack_adj(
                instruction,
                operands_part,
                final_syntax_operands,
            );
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_cm_mvsa01_instruction",
        {
            fix_cm_mvsa01_instruction(instruction, operands_part, final_syntax_operands);
        }
    );

    // Group 5: cleanup must run last because these placeholders are intentionally
    // hard-coded in rendered syntax and should not survive as random operands.
    // Group 5 cleanup must run last because these placeholders are hard-coded
    // in the final assembly text and should not survive as random syntax operands.
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_vector_v0_hardcoded_operands",
        {
            fix_vector_v0_hardcoded_operands(final_syntax_operands);
        }
    );
    audited_step!(
        audit,
        instruction,
        *operands_part,
        *final_syntax_operands,
        "fix_sp_should_not_in_final_syntax_operands",
        {
            fix_sp_should_not_in_final_syntax_operands(final_syntax_operands);
        }
    );
}
