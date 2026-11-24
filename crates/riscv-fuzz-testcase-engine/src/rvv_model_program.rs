//! RVV-only test bodies composed from the obligation (VC) and constrained
//! state-flow (SF) models. Scalar and memory initialization is retained.

use rand::{SeedableRng as _, rngs::StdRng};
use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
use riscv_instruction::{
    RV64Extensions, RV64Instruction, RV64Zvl32bInstructions, RiscvInstruction,
};
use riscv_instruction_types::build_instruction_sequence;

use crate::{
    GenerationError,
    testcase::{GeneratedTestCase, HartPhase, TestCaseConfig},
};

/// Catalog modes assign one base-V grammar opcode to each seed. The random
/// mode permutes the catalog and samples legal state and operands for the
/// selected opcode without substituting another instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvvModelProgramMode {
    Pure,
    /// VC/SF blocks plus probability-sampled expected Illegal, without catalog.
    PureNegative,
    CatalogSweep,
    CatalogSweepSafe,
    RandomCatalog,
}

fn defect_first_enabled() -> bool {
    std::env::var("RISCV_FUZZ_RVV_DEFECT_FIRST").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub fn generate_rvv_model_program_with_extensions(
    core_ids: &[String],
    mut testcase_config: TestCaseConfig,
    extensions: IsaExtensions,
    mode: RvvModelProgramMode,
) -> Result<GeneratedTestCase, GenerationError> {
    if testcase_config.isa_base != ISABase::Rv64
        || !extensions.contains_rv64(RV64Extensions::Zvl32b)
    {
        return Err(GenerationError::StructuredInstructionBuildFailed {
            message: "the VC+SF RVV program requires RV64 with base V (Zvl32b)".to_string(),
        });
    }
    if testcase_config.hart_count() != 1 {
        return Err(GenerationError::StructuredInstructionBuildFailed {
            message: "the VC+SF RVV program currently supports one hart".to_string(),
        });
    }
    if core_ids
        .first()
        .is_some_and(|id| id.eq_ignore_ascii_case("ara"))
    {
        testcase_config.mem_size = testcase_config.mem_size.max(2048);
    }
    let mut testcase = crate::impls::generate_model_skeleton_with_seeded_initialization(
        core_ids,
        testcase_config,
        extensions,
    )?;
    crate::rvv_stateflow::remove_ambient_rvv_init_instructions(&mut testcase);
    crate::rvv_stateflow::clear_ambient_test_instructions(&mut testcase);

    // Each model block establishes its own configuration and live state.
    // A focused VC run places its target before SF; otherwise SF runs first.
    let positive = matches!(
        mode,
        RvvModelProgramMode::RandomCatalog | RvvModelProgramMode::PureNegative
    );
    let defect_first = mode == RvvModelProgramMode::RandomCatalog && defect_first_enabled();
    let vc_flavor = if positive {
        crate::rvv_model::HardwareRvvInjectionFlavor::ContractPositive
    } else {
        crate::rvv_model::HardwareRvvInjectionFlavor::ContractConstrained
    };
    let flavor = if mode == RvvModelProgramMode::CatalogSweepSafe {
        // This profile avoids the deliberately risky SF memory templates. It
        // still requires a reference run: the VC block and catalog target may
        // legitimately fault, and no static rule guarantees a positive case.
        crate::rvv_stateflow::RvvStateFlowFlavor::CoverageConservative
    } else if positive
        && core_ids
            .first()
            .is_some_and(|id| id.eq_ignore_ascii_case("ara"))
    {
        crate::rvv_stateflow::RvvStateFlowFlavor::PositiveAraGeneric
    } else if positive {
        crate::rvv_stateflow::RvvStateFlowFlavor::PositiveConstrained
    } else if core_ids
        .first()
        .is_some_and(|id| id.eq_ignore_ascii_case("ara"))
    {
        crate::rvv_stateflow::RvvStateFlowFlavor::AraGeneric
    } else {
        crate::rvv_stateflow::RvvStateFlowFlavor::StateFlowConstrained
    };
    let vc_focus = positive
        && std::env::var("RISCV_FUZZ_RVV_VC_FOCUS").is_ok_and(|value| !value.trim().is_empty());
    let sf_focus = positive
        && std::env::var("RISCV_FUZZ_RVV_SF_FOCUS").is_ok_and(|value| !value.trim().is_empty());
    if (mode == RvvModelProgramMode::PureNegative || defect_first) && (vc_focus || sf_focus) {
        // Focused VC/SF scenes initialize every operand they consume. The
        // ambient seeded register/memory preload otherwise costs hundreds of
        // scalar commits before the intended relation can execute on the DUT.
        for hart in testcase.harts.iter_mut() {
            hart.sequences_mut(HartPhase::Init).clear();
            hart.shared_sequence_indices_mut(HartPhase::Init).clear();
        }
    }
    if vc_focus {
        // One complete VC relation per program prevents an unrelated SF
        // difference from masking the target. Broad seeds still combine both.
        crate::rvv_model::inject_hardware_rvv_blocks_local_with_flavor(
            &mut testcase,
            core_ids,
            vc_flavor,
        )?;
    } else if sf_focus {
        crate::rvv_stateflow::inject_hardware_rvv_stateflow_with_flavor(
            &mut testcase,
            core_ids,
            flavor,
        )?;
    } else {
        crate::rvv_model::inject_hardware_rvv_blocks_local_with_flavor(
            &mut testcase,
            core_ids,
            vc_flavor,
        )?;
        crate::rvv_stateflow::inject_hardware_rvv_stateflow_with_flavor(
            &mut testcase,
            core_ids,
            flavor,
        )?;
    }
    let setup_prefix_len = if core_ids
        .first()
        .is_some_and(|id| id.eq_ignore_ascii_case("ara"))
    {
        crate::rvv_stateflow::inject_vector_state_enable(&mut testcase)?
    } else {
        0
    };

    // A complete model relation must reach its observer before an unrelated
    // composition or catalog instruction can trigger a first-difference abort.
    // Keep the coverage-oriented order as the default for existing runs.
    let extra_insertion = if defect_first {
        testcase.harts[0].sequences(HartPhase::Test).len()
    } else {
        setup_prefix_len
    };

    let semantic_prefix_len = if mode == RvvModelProgramMode::RandomCatalog {
        let legacy_len = crate::rvv_semantic_flow::maybe_inject(
            &mut testcase,
            core_ids.first().map(String::as_str).unwrap_or("unknown"),
            extra_insertion,
        )?;
        let composed_len = crate::rvv_compositional_flow::maybe_inject(
            &mut testcase,
            core_ids.first().map(String::as_str).unwrap_or("unknown"),
            extra_insertion + legacy_len,
        )?;
        legacy_len + composed_len
    } else {
        0
    };

    if !matches!(
        mode,
        RvvModelProgramMode::Pure | RvvModelProgramMode::PureNegative
    ) {
        append_catalog_coverage_target(
            &mut testcase,
            setup_prefix_len + semantic_prefix_len,
            core_ids.first().map(String::as_str).unwrap_or("unknown"),
            mode,
            true,
            None,
        )?;
    }
    if positive {
        let before = testcase.harts[0].sequences(HartPhase::Test).len();
        crate::rvv_negative::append_expected_illegal_probe(&mut testcase)?;
        if mode == RvvModelProgramMode::PureNegative || defect_first {
            let test = testcase.harts[0].sequences_mut(HartPhase::Test);
            if test.len() > before {
                let negative = test.pop().expect("sampled negative was appended");
                if defect_first {
                    // The sampled negative is this program's primary target.
                    // It initializes its own operands and remains explicitly
                    // labeled by the runner, so later positive work cannot
                    // hide an illegal-instruction discrepancy.
                    test.insert(setup_prefix_len, negative);
                } else if core_ids
                    .first()
                    .is_some_and(|id| id.eq_ignore_ascii_case("xiangshan"))
                {
                    // Preserve the previously confirmed XiangShan negative
                    // path until the standalone form has DUT evidence.
                    test.insert(setup_prefix_len, negative);
                } else {
                    // Ara's sampled negative is its own short program. A
                    // deliberate first difference must not mask a focused
                    // positive target in the same case.
                    test.truncate(setup_prefix_len);
                    test.push(negative);
                    testcase.harts[0]
                        .shared_sequence_indices_mut(HartPhase::Test)
                        .retain(|index| *index < setup_prefix_len);
                    // This probe initializes every input it consumes.
                    for hart in &mut testcase.harts {
                        hart.sequences_mut(HartPhase::Init).clear();
                        hart.shared_sequence_indices_mut(HartPhase::Init).clear();
                    }
                }
            }
        }
    }
    if std::env::var_os("RISCV_FUZZ_RVV_AUDIT").is_some() {
        let subject = core_ids.first().map(String::as_str).unwrap_or("unknown");
        match crate::rvv_catalog_legal::RvvSamplingProfile::from_subject_and_env(subject) {
            Ok(profile) => {
                let source = testcase.test_program().rendered_lines().join("\n");
                let report = crate::rvv_program_legal::audit_program_asm(&source, profile);
                eprintln!(
                    "RVV shared-precondition audit seed={:?} checked={} issues={} unknown_vtype={} unknown_vstart={} (diagnostic only)",
                    testcase.config.seed,
                    report.checked_vector_instructions,
                    report.issues.len(),
                    report.unknown_vtype_instructions,
                    report.unknown_vstart_preconditions,
                );
                for issue in report.issues {
                    eprintln!(
                        "RVV precondition line={} code={} instruction={} detail={}",
                        issue.line, issue.code, issue.instruction, issue.detail
                    );
                }
            }
            Err(error) => eprintln!("RVV shared-precondition audit unavailable: {error}"),
        }
    }
    Ok(testcase)
}

pub(crate) fn append_catalog_coverage_target(
    testcase: &mut GeneratedTestCase,
    setup_prefix_len: usize,
    subject: &str,
    mode: RvvModelProgramMode,
    allow_front: bool,
    bridge_addr: Option<u64>,
) -> Result<(), GenerationError> {
    let seed =
        testcase
            .config
            .seed
            .ok_or_else(|| GenerationError::StructuredInstructionBuildFailed {
                message:
                    "catalog sweep requires an explicit seed for reproducible opcode scheduling"
                        .to_string(),
            })?;
    let count = RV64Zvl32bInstructions::VARIANT_COUNT;
    let index = if mode == RvvModelProgramMode::RandomCatalog {
        shuffled_catalog_index(seed, count)
    } else {
        (seed % count as u64) as usize
    };
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5643_5346_4341_5441);
    let (memory_start, memory_end) = testcase.mem_range.get(subject).copied().ok_or_else(|| {
        GenerationError::StructuredInstructionBuildFailed {
            message: "catalog target needs a writable memory range".to_string(),
        }
    })?;
    let profile = crate::rvv_catalog_legal::RvvSamplingProfile::from_subject_and_env(subject)
        .map_err(|message| GenerationError::StructuredInstructionBuildFailed { message })?;
    let scratch_bytes = u64::from(profile.vlen_bits).max(0x200);
    let memory_offset =
        if memory_end.saturating_sub(memory_start).saturating_add(1) >= 0x200 + scratch_bytes {
            0x200
        } else {
            0
        };
    let safe_memory_base = (memory_start + memory_offset).div_ceil(8) * 8;
    let safe_memory_bytes = memory_end
        .saturating_sub(safe_memory_base)
        .saturating_add(1);
    let variant =
        RV64Zvl32bInstructions::random_at_index(index, &mut rng, &testcase.config.random_config)?;
    let instruction = RiscvInstruction::RV64(RV64Instruction::Zvl32b(variant));
    let sequence =
        build_instruction_sequence(instruction, &mut rng, &testcase.config.random_config)?;
    let prefix_len = testcase.harts[0].sequences(HartPhase::Test).len();
    testcase.harts[0]
        .sequences_mut(HartPhase::Test)
        .push(sequence);
    if mode == RvvModelProgramMode::RandomCatalog {
        let vector_range = testcase
            .config
            .random_config
            .register_operand_config
            .vector_register_number_range;
        let isa_base = testcase.config.isa_base;
        let target = &mut testcase.harts[0].sequences_mut(HartPhase::Test)[prefix_len];
        crate::rvv_catalog_legal::constrain_selected_opcode_with_profile(
            target,
            isa_base,
            vector_range,
            safe_memory_base,
            safe_memory_bytes,
            profile,
            Some(((seed / count as u64) / 2, ((seed / count as u64) % 2) as u8)),
            bridge_addr,
            &mut rng,
        )?;
    } else {
        crate::rvv_constraints::constrain_random_rvv_testcase_after_first_hart_prefix(
            testcase, prefix_len,
        )?;
    }

    // Alternate the location of the targeted opcode. Front placement lets a
    // complex memory opcode retire before a later SF stall; back placement
    // keeps the model dependency chain first in the other half of seeds.
    // In scheduled indexed fault probes, run the SF scenario before a random
    // catalog memory target can fault or alter its scratch registers.
    let scheduled_indexed =
        mode == RvvModelProgramMode::RandomCatalog && matches!(seed % 10, 7 | 8 | 9);
    let focused_model = (mode == RvvModelProgramMode::RandomCatalog && defect_first_enabled())
        || std::env::var_os("RISCV_FUZZ_RVV_VC_FOCUS").is_some()
        || std::env::var_os("RISCV_FUZZ_RVV_SF_FOCUS").is_some();
    if allow_front && seed.is_multiple_of(2) && !scheduled_indexed && !focused_model {
        let hart = &mut testcase.harts[0];
        let test = hart.sequences_mut(HartPhase::Test);
        let target = test.pop().expect("catalog target was just appended");
        test.insert(setup_prefix_len, target);
    }
    Ok(())
}

/// Randomize opcode order while still visiting every base-V opcode once in
/// each full catalog epoch.  The permutation is stable across epochs so
/// seed+count retries the *same* opcode with new random operands and state.
fn shuffled_catalog_index(seed: u64, count: usize) -> usize {
    let position = (seed % count as u64) as usize;
    let mut order = (0..count).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(0x5241_4e44_5643_5346);
    for i in (1..count).rev() {
        let other = rand::Rng::random_range(&mut rng, 0..=i);
        order.swap(i, other);
    }
    order[position]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_index_cycles_all_base_v_variants() {
        let count = RV64Zvl32bInstructions::VARIANT_COUNT;
        assert!(count >= 600);
        let scheduled = (0..count)
            .map(|seed| seed % count)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(scheduled.len(), count);
    }

    #[test]
    fn randomized_catalog_epoch_is_reproducible_and_complete() {
        let count = RV64Zvl32bInstructions::VARIANT_COUNT;
        for epoch in [0u64, 1, 3253910] {
            let first = (0..count)
                .map(|slot| shuffled_catalog_index(epoch * count as u64 + slot as u64, count))
                .collect::<Vec<_>>();
            let again = (0..count)
                .map(|slot| shuffled_catalog_index(epoch * count as u64 + slot as u64, count))
                .collect::<Vec<_>>();
            assert_eq!(first, again);
            assert_eq!(
                first
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                count
            );
        }
        assert_ne!(shuffled_catalog_index(0, count), 0);
        assert_eq!(
            shuffled_catalog_index(17, count),
            shuffled_catalog_index(17 + count as u64, count)
        );
    }

    #[test]
    fn randomized_catalog_keeps_each_selected_base_v_opcode() {
        let config = riscv_instruction_types::RandomConfig::default();
        let mut failures = Vec::new();
        for index in 0..RV64Zvl32bInstructions::VARIANT_COUNT {
            let mut rng = StdRng::seed_from_u64(index as u64 ^ 0x4256_4341_5441_4c4f);
            let result = RV64Zvl32bInstructions::random_at_index(index, &mut rng, &config)
                .and_then(|variant| {
                    build_instruction_sequence(
                        RiscvInstruction::RV64(RV64Instruction::Zvl32b(variant)),
                        &mut rng,
                        &config,
                    )
                });
            match result {
                Ok(mut sequence) => {
                    let selected = sequence
                        .instruction
                        .to_string()
                        .split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_string();
                    if let Err(error) = crate::rvv_catalog_legal::constrain_selected_opcode(
                        &mut sequence,
                        ISABase::Rv64,
                        (13, 31),
                        0x8001_0200,
                        &mut rng,
                    ) {
                        failures.push(format!("{index}: {selected}: {error}"));
                    } else if !sequence.instruction.to_string().starts_with(&selected) {
                        failures.push(format!("{index}: selected opcode changed: {selected}"));
                    }
                }
                Err(error) => failures.push(format!("{index}: grammar: {error}")),
            }
        }
        assert!(failures.is_empty(), "catalog legality gaps: {failures:?}");
    }

    #[test]
    fn base_v_catalog_variants_can_be_instantiated() {
        let config = riscv_instruction_types::RandomConfig::default();
        let mut failures = Vec::new();
        for index in 0..RV64Zvl32bInstructions::VARIANT_COUNT {
            let mut rng = StdRng::seed_from_u64(index as u64 ^ 0x5643_5346_4341_5441);
            let result = RV64Zvl32bInstructions::random_at_index(index, &mut rng, &config)
                .and_then(|variant| {
                    build_instruction_sequence(
                        RiscvInstruction::RV64(RV64Instruction::Zvl32b(variant)),
                        &mut rng,
                        &config,
                    )
                });
            if let Err(error) = result {
                failures.push(format!("{index}: {error}"));
            }
        }
        assert!(failures.is_empty(), "base-V catalog gaps: {failures:?}");
    }
}
