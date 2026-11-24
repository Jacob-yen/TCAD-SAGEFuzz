use crate::{RandomConfig, RandomGenerationError};

pub trait Random {
    type Output;
    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError>;
}

impl Random for bool {
    type Output = bool;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(rng.random())
    }
}

impl Random for u8 {
    type Output = u8;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(rng.random())
    }
}

impl Random for u16 {
    type Output = u16;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(rng.random())
    }
}

impl Random for u32 {
    type Output = u32;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(rng.random())
    }
}

impl Random for u64 {
    type Output = u64;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(rng.random())
    }
}

impl Random for i8 {
    type Output = i8;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(rng.random())
    }
}

impl Random for i16 {
    type Output = i16;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(rng.random())
    }
}

impl Random for i32 {
    type Output = i32;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(rng.random())
    }
}

impl Random for i64 {
    type Output = i64;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        Ok(rng.random())
    }
}

impl Random for String {
    type Output = String;

    fn random_with_rng<R: rand::Rng>(
        rng: &mut R,
        _config: &RandomConfig,
    ) -> Result<Self::Output, RandomGenerationError> {
        // Generate a random label name such as `label_123`.
        Ok(format!("label_{}", rng.random_range(0u32..1000u32)))
    }
}
