use std::{collections::HashMap, fmt::Debug, hash::Hash};

use riscv_instruction::{GenerationOrder, GenerationPlan, RV32Extensions, RV64Extensions};

use crate::{GenerationError, testcase::ExtensionInstScaling, testcase::TestCaseConfig};

fn validate_extension_inst_scaling(cfg: &ExtensionInstScaling) -> Result<(), GenerationError> {
    if cfg.min_insts > cfg.max_insts {
        return Err(GenerationError::InvalidExtensionScaling {
            min: cfg.min_insts,
            max: cfg.max_insts,
        });
    }

    Ok(())
}

fn apply_extension_inst_scaling(cfg: &ExtensionInstScaling, base: usize) -> usize {
    let scaled = (base as f64 * cfg.multiplier).round();
    let scaled = scaled.clamp(0.0, usize::MAX as f64) as usize;
    scaled.clamp(cfg.min_insts, cfg.max_insts)
}

pub fn ordered_generation_plan<E>(counts: &HashMap<E, usize>) -> GenerationPlan<E>
where
    E: Copy + Eq + Hash + Debug,
{
    let mut requests: Vec<(E, usize, String)> = counts
        .iter()
        .map(|(&ext, &count)| (ext, count, format!("{ext:?}")))
        .collect();
    requests.sort_by(|(_, _, left), (_, _, right)| left.cmp(right));
    GenerationPlan::new(requests.into_iter().map(|(ext, count, _)| (ext, count)))
        .with_order(GenerationOrder::RandomShuffle)
}

fn build_generation_counts<T, F>(
    extensions: &[T],
    testcase_config: &TestCaseConfig,
    instruction_count: F,
) -> Result<HashMap<T, usize>, GenerationError>
where
    T: Copy + Eq + Hash + Debug,
    F: Fn(&T) -> usize,
{
    let counts = if let Some(cfg) = testcase_config.extension_inst_scaling.as_ref() {
        validate_extension_inst_scaling(cfg)?;
        extensions
            .iter()
            .map(|ext| {
                (
                    *ext,
                    apply_extension_inst_scaling(cfg, instruction_count(ext)),
                )
            })
            .collect()
    } else {
        extensions
            .iter()
            .map(|ext| (*ext, testcase_config.extension_insts_count))
            .collect()
    };

    Ok(counts)
}

pub fn build_rv32_generation_counts(
    extensions: &[RV32Extensions],
    testcase_config: &TestCaseConfig,
) -> Result<HashMap<RV32Extensions, usize>, GenerationError> {
    build_generation_counts(
        extensions,
        testcase_config,
        RV32Extensions::instruction_count,
    )
}

pub fn build_rv64_generation_counts(
    extensions: &[RV64Extensions],
    testcase_config: &TestCaseConfig,
) -> Result<HashMap<RV64Extensions, usize>, GenerationError> {
    build_generation_counts(
        extensions,
        testcase_config,
        RV64Extensions::instruction_count,
    )
}
