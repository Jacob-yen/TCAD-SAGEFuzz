use rand::seq::SliceRandom as _;
use riscv_instruction_types::{InstructionSequence, RandomConfig, RandomGenerationError};

const DEFAULT_MAX_ATTEMPTS_PER_SEQUENCE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationOrder {
    Sequential,
    RandomShuffle,
}

pub trait ExtensionSequenceSource: Copy + std::fmt::Debug {
    type Instruction;

    fn random_sequence_with_rng<R: rand::Rng>(
        self,
        rng: &mut R,
        config: &RandomConfig,
    ) -> Result<InstructionSequence<Self::Instruction>, RandomGenerationError>;
}

#[derive(Debug, Clone)]
pub struct GenerationPlan<E> {
    requests: Vec<(E, usize)>,
    order: GenerationOrder,
}

impl<E> GenerationPlan<E> {
    pub fn new<I>(requests: I) -> Self
    where
        I: IntoIterator<Item = (E, usize)>,
    {
        Self {
            requests: requests.into_iter().collect(),
            order: GenerationOrder::RandomShuffle,
        }
    }

    pub fn uniform<I>(extensions: I, count_per_extension: usize) -> Self
    where
        I: IntoIterator<Item = E>,
    {
        Self::new(
            extensions
                .into_iter()
                .map(|extension| (extension, count_per_extension)),
        )
    }

    pub fn with_order(mut self, order: GenerationOrder) -> Self {
        self.order = order;
        self
    }

    pub fn requests(&self) -> &[(E, usize)] {
        &self.requests
    }

    pub fn order(&self) -> GenerationOrder {
        self.order
    }
}

pub fn generate_sequences<E>(
    plan: &GenerationPlan<E>,
    random_config: &RandomConfig,
) -> Result<Vec<InstructionSequence<E::Instruction>>, RandomGenerationError>
where
    E: ExtensionSequenceSource,
{
    let mut rng = rand::rng();
    generate_sequences_with_rng(&mut rng, plan, random_config)
}

pub fn generate_sequences_with_rng<E, R>(
    rng: &mut R,
    plan: &GenerationPlan<E>,
    random_config: &RandomConfig,
) -> Result<Vec<InstructionSequence<E::Instruction>>, RandomGenerationError>
where
    E: ExtensionSequenceSource,
    R: rand::Rng,
{
    let mut sequences = Vec::new();

    for &(extension, target_count) in plan.requests() {
        if target_count == 0 {
            continue;
        }

        let mut generated = 0usize;
        let mut last_error = None;
        let attempt_budget = target_count
            .saturating_mul(DEFAULT_MAX_ATTEMPTS_PER_SEQUENCE)
            .max(target_count);

        for _ in 0..attempt_budget {
            if generated >= target_count {
                break;
            }

            match extension.random_sequence_with_rng(rng, random_config) {
                Ok(sequence) => {
                    sequences.push(sequence);
                    generated += 1;
                }
                Err(err) => last_error = Some(err),
            }
        }

        if generated < target_count {
            let details = format!(
                "requested {target_count} sequences for extension {extension:?}, generated {generated}"
            );
            return Err(match last_error {
                Some(err) => err.with_context(details),
                None => RandomGenerationError::new("SequenceGenerator", details),
            });
        }
    }

    if plan.order() == GenerationOrder::RandomShuffle {
        sequences.shuffle(rng);
    }

    Ok(sequences)
}

pub fn generate_instructions<E>(
    plan: &GenerationPlan<E>,
    random_config: &RandomConfig,
) -> Result<Vec<E::Instruction>, RandomGenerationError>
where
    E: ExtensionSequenceSource,
{
    let mut rng = rand::rng();
    generate_instructions_with_rng(&mut rng, plan, random_config)
}

pub fn generate_instructions_with_rng<E, R>(
    rng: &mut R,
    plan: &GenerationPlan<E>,
    random_config: &RandomConfig,
) -> Result<Vec<E::Instruction>, RandomGenerationError>
where
    E: ExtensionSequenceSource,
    R: rand::Rng,
{
    let sequences = generate_sequences_with_rng(rng, plan, random_config)?;
    Ok(sequences
        .into_iter()
        .map(|sequence| sequence.instruction)
        .collect())
}

#[macro_export]
macro_rules! generate_sequence_generator {
    () => {
        impl $crate::sequence_generator::ExtensionSequenceSource for RV32Extensions {
            type Instruction = RiscvInstruction;

            fn random_sequence_with_rng<R: rand::Rng>(
                self,
                rng: &mut R,
                config: &riscv_instruction_types::RandomConfig,
            ) -> Result<
                riscv_instruction_types::InstructionSequence<Self::Instruction>,
                riscv_instruction_types::RandomGenerationError,
            > {
                RV32Extensions::random_sequence_with_rng(&self, rng, config)
            }
        }

        impl $crate::sequence_generator::ExtensionSequenceSource for RV64Extensions {
            type Instruction = RiscvInstruction;

            fn random_sequence_with_rng<R: rand::Rng>(
                self,
                rng: &mut R,
                config: &riscv_instruction_types::RandomConfig,
            ) -> Result<
                riscv_instruction_types::InstructionSequence<Self::Instruction>,
                riscv_instruction_types::RandomGenerationError,
            > {
                RV64Extensions::random_sequence_with_rng(&self, rng, config)
            }
        }

        pub use $crate::sequence_generator::{
            GenerationOrder, GenerationPlan, generate_instructions, generate_instructions_with_rng,
            generate_sequences, generate_sequences_with_rng,
        };
    };
}
