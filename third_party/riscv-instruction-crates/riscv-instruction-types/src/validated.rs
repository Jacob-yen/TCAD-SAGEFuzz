use rand::seq::IndexedRandom as _;

use crate::{NumericOperandMutation, RandomConfig, RandomGenerationError};

impl NumericOperandMutation for bool {
    fn mutate_numeric_operands<R: rand::Rng>(
        &mut self,
        rng: &mut R,
        _config: &RandomConfig,
    ) -> bool {
        let original = *self;
        *self = rng.random();
        *self != original
    }
}

pub fn sample_signed_value<R: rand::Rng>(rng: &mut R, min: i64, max: i64, bits: u8) -> i64 {
    let bounds = [
        0i64,
        1,
        -1,
        2,
        -2,
        0x7f,
        -0x80,
        0x80,
        -0x81,
        0x7ff,
        -0x800,
        0x800,
        -0x801,
        0x7fff,
        -0x8000,
        0x8000,
        -0x8001,
        0x7fffffff,
        -0x80000000,
        0x80000000,
    ];
    let mut candidates: Vec<i64> = bounds
        .iter()
        .copied()
        .filter(|v| *v >= min && *v <= max)
        .collect();

    if bits > 0 && bits <= 63 {
        let high = (1i64 << (bits - 1)) - 1;
        let low = -(1i64 << (bits - 1));
        for &v in &[low, low + 1, high - 1, high] {
            if v >= min && v <= max {
                candidates.push(v);
            }
        }
    }

    if !candidates.is_empty() && rng.random_ratio(2, 5) {
        if let Some(choice) = candidates.choose(rng) {
            return *choice;
        }
    }

    if min == max {
        return min;
    }
    rng.random_range(min..=max)
}

pub fn sample_unsigned_value<R: rand::Rng>(rng: &mut R, min: u64, max: u64, bits: u8) -> u64 {
    let bounds = [
        0u64, 1, 2, 3, 0x7f, 0x80, 0xff, 0x100, 0x7ff, 0x800, 0xfff, 0x1000, 0x7fff, 0x8000,
        0xffff, 0x1_0000, 0x7fffffff, 0x80000000, 0xffffffff,
    ];
    let mut candidates: Vec<u64> = bounds
        .iter()
        .copied()
        .filter(|v| *v >= min && *v <= max)
        .collect();

    if bits > 0 && bits <= 63 {
        let high = (1u64 << bits) - 1;
        for &v in &[high, high.saturating_sub(1)] {
            if v >= min && v <= max {
                candidates.push(v);
            }
        }
    }

    if !candidates.is_empty() && rng.random_ratio(2, 5) {
        if let Some(choice) = candidates.choose(rng) {
            return *choice;
        }
    }

    if min == max {
        return min;
    }
    rng.random_range(min..=max)
}

pub fn sample_shamt_value<R: rand::Rng>(rng: &mut R, min: u64, max: u64, bits: u8) -> u64 {
    let mut candidates = vec![0u64, 1, 2, 3, 4];

    if bits > 0 && bits < 64 {
        let max_bits = (1u64 << bits) - 1;
        let half = 1u64 << (bits.saturating_sub(1));
        candidates.extend_from_slice(&[
            max_bits,
            max_bits.saturating_sub(1),
            half,
            half.saturating_sub(1),
            max_bits / 2,
        ]);
    }

    candidates.retain(|v| *v >= min && *v <= max);
    candidates.sort_unstable();
    candidates.dedup();

    if !candidates.is_empty() && rng.random_ratio(3, 5) {
        return *candidates.choose(rng).unwrap();
    }

    if min == max {
        return min;
    }
    rng.random_range(min..=max)
}

pub fn sample_compressed_signed_immediate<R: rand::Rng>(
    rng: &mut R,
    min: i64,
    max: i64,
    bits: u8,
) -> i64 {
    let mut candidates = vec![0i64, 1, -1, 2, -2, 4, -4, 8, -8, 16, -16];

    if bits > 0 && bits <= 63 {
        let high = (1i64 << (bits - 1)) - 1;
        let low = -(1i64 << (bits - 1));
        candidates.extend_from_slice(&[low, low + 1, high - 1, high]);
    }

    for step in [2i64, 4, 8, 16] {
        for &mul in &[1i64, -1] {
            let v = step * mul;
            candidates.push(v);
        }
    }

    candidates.retain(|v| *v >= min && *v <= max);
    candidates.sort_unstable();
    candidates.dedup();

    if !candidates.is_empty() && rng.random_ratio(3, 5) {
        return *candidates.choose(rng).unwrap();
    }

    if min == max {
        return min;
    }
    rng.random_range(min..=max)
}

pub fn sample_compressed_unsigned_immediate<R: rand::Rng>(
    rng: &mut R,
    min: u64,
    max: u64,
    bits: u8,
) -> u64 {
    let mut candidates = vec![0u64, 1, 2, 3, 4, 8, 16, 32, 64];

    if bits > 0 && bits < 64 {
        let max_bits = (1u64 << bits) - 1;
        candidates.extend_from_slice(&[max_bits, max_bits.saturating_sub(1)]);
    }

    for step in [2u64, 4, 8, 16] {
        candidates.push(step);
        candidates.push(step.saturating_mul(2));
    }

    candidates.retain(|v| *v >= min && *v <= max);
    candidates.sort_unstable();
    candidates.dedup();

    if !candidates.is_empty() && rng.random_ratio(3, 5) {
        return *candidates.choose(rng).unwrap();
    }

    if min == max {
        return min;
    }
    rng.random_range(min..=max)
}

pub(crate) fn sign_extend(value: u64, bits: u8) -> i64 {
    if bits == 0 || bits >= 64 {
        return value as i64;
    }
    let sign_bit = 1u64 << (bits - 1);
    let mask = (1u64 << bits) - 1;
    let v = value & mask;
    if v & sign_bit != 0 {
        (v | (!mask)) as i64
    } else {
        v as i64
    }
}

pub fn sample_vector_signed_immediate<R: rand::Rng>(
    rng: &mut R,
    min: i64,
    max: i64,
    bits: u8,
    config: &RandomConfig,
) -> i64 {
    let mut candidates = vec![0i64, 1, -1, 2, -2];
    if bits > 0 && bits < 64 {
        let max_val = (1i64 << (bits - 1)) - 1;
        let min_val = -(1i64 << (bits - 1));
        for &v in &[min_val, min_val + 1, max_val - 1, max_val] {
            candidates.push(v);
        }
        let mask = (1u64 << bits) - 1;
        let patterns = [0x55u64, 0xAAu64, mask, mask.saturating_sub(1)];
        for &pat in &patterns {
            candidates.push(sign_extend(pat, bits));
        }
    }

    candidates.retain(|v| *v >= min && *v <= max);

    // Use the configured edge-case probability instead of a hard-coded `3/5`.
    let use_boundary = if config.edge_case_probability <= 0.0 {
        false
    } else if config.edge_case_probability >= 1.0 {
        !candidates.is_empty()
    } else {
        !candidates.is_empty() && rng.random_range(0.0..1.0) < config.edge_case_probability
    };

    if use_boundary {
        return *candidates.choose(rng).unwrap();
    }

    if min == max {
        return min;
    }
    rng.random_range(min..=max)
}

pub fn sample_vector_unsigned_immediate<R: rand::Rng>(
    rng: &mut R,
    min: u64,
    max: u64,
    bits: u8,
    config: &RandomConfig,
) -> u64 {
    let mut candidates = vec![0u64, 1, 2, 3];
    if bits > 0 && bits < 64 {
        let max_val = (1u64 << bits) - 1;
        candidates.push(max_val);
        candidates.push(max_val.saturating_sub(1));
        let mask = max_val;
        let patterns = [0x55u64, 0xAAu64, 0x0Fu64, 0xF0u64];
        for &pat in &patterns {
            candidates.push(pat & mask);
        }
    }

    candidates.retain(|v| *v >= min && *v <= max);

    // Use the configured edge-case probability instead of a hard-coded `3/5`.
    let use_boundary = if config.edge_case_probability <= 0.0 {
        false
    } else if config.edge_case_probability >= 1.0 {
        !candidates.is_empty()
    } else {
        !candidates.is_empty() && rng.random_range(0.0..1.0) < config.edge_case_probability
    };

    if use_boundary {
        return *candidates.choose(rng).unwrap();
    }

    if min == max {
        return min;
    }
    rng.random_range(min..=max)
}

fn format_nan_payload(mantissa: u64) -> u64 {
    let payload = mantissa & 0x000f_ffff_ffff_ffff;
    0x7ff0_0000_0000_0000u64 | payload | 0x0008_0000_0000_0000u64
}

pub fn sample_float_bits<R: rand::Rng>(rng: &mut R, width_bits: u8, config: &RandomConfig) -> u64 {
    if width_bits == 32 {
        let specials: [u32; 10] = [
            0x0000_0000, // +0.0
            0x8000_0000, // -0.0
            0x3f80_0000, // +1.0
            0xbf80_0000, // -1.0
            0x7f80_0000, // +Inf
            0xff80_0000, // -Inf
            0x7fc0_0000, // NaN
            0x0080_0000, // smallest normal
            0x007f_ffff, // largest subnormal
            0x0000_0001, // smallest subnormal
        ];

        // Use the configured edge-case probability.
        let use_boundary = if config.edge_case_probability <= 0.0 {
            false
        } else if config.edge_case_probability >= 1.0 {
            true
        } else {
            rng.random_range(0.0..1.0) < config.edge_case_probability
        };

        if use_boundary {
            return *specials.choose(rng).unwrap() as u64;
        }
        return rng.random::<u32>() as u64;
    }

    if width_bits == 64 {
        let specials: [u64; 12] = [
            0x0000_0000_0000_0000, // +0.0
            0x8000_0000_0000_0000, // -0.0
            0x3ff0_0000_0000_0000, // +1.0
            0xbff0_0000_0000_0000, // -1.0
            0x7ff0_0000_0000_0000, // +Inf
            0xfff0_0000_0000_0000, // -Inf
            0x7ff8_0000_0000_0000, // NaN
            0x0010_0000_0000_0000, // smallest normal
            0x000f_ffff_ffff_ffff, // largest subnormal
            0x0000_0000_0000_0001, // smallest subnormal
            0x7ff0_0000_0000_0001, // NaN with payload
            0x7fe0_0000_0000_0000, // near max normal
        ];

        // Use the configured edge-case probability.
        let use_boundary = if config.edge_case_probability <= 0.0 {
            false
        } else if config.edge_case_probability >= 1.0 {
            true
        } else {
            rng.random_range(0.0..1.0) < config.edge_case_probability
        };

        if use_boundary {
            // Generate a random NaN payload with 20% probability.
            if rng.random_ratio(1, 5) {
                let mantissa = rng.random::<u64>();
                return format_nan_payload(mantissa);
            }
            return *specials.choose(rng).unwrap();
        }
        return rng.random::<u64>();
    }

    rng.random::<u64>()
}

pub fn sample_vector_pattern<R: rand::Rng>(rng: &mut R, elem_width_bits: u8) -> u64 {
    let base = match rng.random_range(0u8..=5u8) {
        0 => 0u64,
        1 => u64::MAX,
        2 => 0x5555_5555_5555_5555,
        3 => 0xaaaa_aaaa_aaaa_aaaa,
        4 => 0x0f0f_0f0f_0f0f_0f0f,
        _ => rng.random::<u64>(),
    };

    if elem_width_bits >= 64 {
        return base;
    }
    let mask = if elem_width_bits == 0 {
        0u64
    } else {
        (1u64 << elem_width_bits) - 1
    };
    base & mask
}

fn mutate_validated_i32<V, R>(value: &mut V, rng: &mut R, range_start: i32, range_end: i32) -> bool
where
    V: ValidatedValue<i32>,
    R: rand::Rng,
{
    if range_start > range_end {
        return false;
    }

    let current = value.get();
    let strategy = rng.random_range(0u8..=4u8);
    let mut candidate = match strategy {
        0 => {
            let delta: i32 = rng.random_range(-16..=16);
            current.saturating_add(delta)
        }
        1 => current.saturating_add(1),
        2 => current.saturating_sub(1),
        3 => current.wrapping_neg(),
        _ => {
            let bit = rng.random_range(0u32..=15u32);
            current ^ (1i32 << bit)
        }
    };

    if candidate < range_start || candidate > range_end {
        candidate = if range_start == range_end {
            range_start
        } else {
            rng.random_range(range_start..=range_end)
        };
    }

    if candidate == current {
        let random_value = if range_start == range_end {
            range_start
        } else {
            rng.random_range(range_start..=range_end)
        };
        if random_value == current {
            return false;
        }
        candidate = random_value;
    }

    if V::FORBIDDEN.contains(&candidate) {
        return false;
    }
    if V::ODD_ONLY && candidate % 2 == 0 {
        return false;
    }
    if let Some(multiple) = V::MULTIPLE_OF {
        if multiple != 0 && candidate % multiple != 0 {
            return false;
        }
    }

    value.set(candidate).is_ok()
}

fn mutate_validated_u32<V, R>(value: &mut V, rng: &mut R, range_start: u32, range_end: u32) -> bool
where
    V: ValidatedValue<u32>,
    R: rand::Rng,
{
    if range_start > range_end {
        return false;
    }

    let current = value.get();
    let strategy = rng.random_range(0u8..=4u8);
    let mut candidate = match strategy {
        0 => {
            let delta: i32 = rng.random_range(-16..=16);
            if delta >= 0 {
                current.saturating_add(delta as u32)
            } else {
                current.saturating_sub((-delta) as u32)
            }
        }
        1 => current.saturating_add(1),
        2 => current.saturating_sub(1),
        3 => current,
        _ => {
            let bit = rng.random_range(0u32..=15u32);
            current ^ (1u32 << bit)
        }
    };

    if candidate < range_start || candidate > range_end {
        candidate = if range_start == range_end {
            range_start
        } else {
            rng.random_range(range_start..=range_end)
        };
    }

    if candidate == current {
        let random_value = if range_start == range_end {
            range_start
        } else {
            rng.random_range(range_start..=range_end)
        };
        if random_value == current {
            return false;
        }
        candidate = random_value;
    }

    if V::FORBIDDEN.contains(&candidate) {
        return false;
    }
    if V::ODD_ONLY && candidate % 2 == 0 {
        return false;
    }
    if let Some(multiple) = V::MULTIPLE_OF {
        if multiple != 0 && candidate % multiple != 0 {
            return false;
        }
    }

    value.set(candidate).is_ok()
}

pub(crate) fn mutate_validated_u16<V, R>(
    value: &mut V,
    rng: &mut R,
    range_start: u16,
    range_end: u16,
) -> bool
where
    V: ValidatedValue<u16>,
    R: rand::Rng,
{
    if range_start > range_end {
        return false;
    }

    let current = value.get();
    let strategy = rng.random_range(0u8..=4u8);
    let mut candidate = match strategy {
        0 => {
            let delta: i16 = rng.random_range(-16..=16);
            if delta >= 0 {
                current.saturating_add(delta as u16)
            } else {
                current.saturating_sub((-delta) as u16)
            }
        }
        1 => current.saturating_add(1),
        2 => current.saturating_sub(1),
        3 => current,
        _ => {
            let bit = rng.random_range(0u32..=15u32);
            current ^ (1u16 << bit)
        }
    };

    if candidate < range_start || candidate > range_end {
        candidate = if range_start == range_end {
            range_start
        } else {
            rng.random_range(range_start..=range_end)
        };
    }

    if candidate == current {
        let random_value = if range_start == range_end {
            range_start
        } else {
            rng.random_range(range_start..=range_end)
        };
        if random_value == current {
            return false;
        }
        candidate = random_value;
    }

    if V::FORBIDDEN.contains(&candidate) {
        return false;
    }
    if V::ODD_ONLY && candidate % 2 == 0 {
        return false;
    }
    if let Some(multiple) = V::MULTIPLE_OF {
        if multiple != 0 && candidate % multiple != 0 {
            return false;
        }
    }

    value.set(candidate).is_ok()
}

pub(crate) fn mutate_validated_u8<V, R>(
    value: &mut V,
    rng: &mut R,
    range_start: u8,
    range_end: u8,
) -> bool
where
    V: ValidatedValue<u8>,
    R: rand::Rng,
{
    if range_start > range_end {
        return false;
    }

    let current = value.get();
    let strategy = rng.random_range(0u8..=4u8);
    let mut candidate = match strategy {
        0 => {
            let delta: i8 = rng.random_range(-16..=16);
            if delta >= 0 {
                current.saturating_add(delta as u8)
            } else {
                current.saturating_sub((-delta) as u8)
            }
        }
        1 => current.saturating_add(1),
        2 => current.saturating_sub(1),
        3 => current,
        _ => {
            let bit = rng.random_range(0u32..=7u32);
            current ^ (1u8 << bit)
        }
    };

    if candidate < range_start || candidate > range_end {
        candidate = if range_start == range_end {
            range_start
        } else {
            rng.random_range(range_start..=range_end)
        };
    }

    if candidate == current {
        let random_value = if range_start == range_end {
            range_start
        } else {
            rng.random_range(range_start..=range_end)
        };
        if random_value == current {
            return false;
        }
        candidate = random_value;
    }

    if V::FORBIDDEN.contains(&candidate) {
        return false;
    }
    if V::ODD_ONLY && candidate % 2 == 0 {
        return false;
    }
    if let Some(multiple) = V::MULTIPLE_OF {
        if multiple != 0 && candidate % multiple != 0 {
            return false;
        }
    }

    value.set(candidate).is_ok()
}

pub fn mutate_validated_i32_with_config<V, R>(
    value: &mut V,
    rng: &mut R,
    config: &RandomConfig,
    is_memory_aware: bool,
) -> bool
where
    V: ValidatedValue<i32>,
    R: rand::Rng,
{
    let (mut min_val, mut max_val) = (V::MIN, V::MAX);
    if is_memory_aware {
        let (config_min, config_max) = config.memory_access_config.memory_offset_immediate_range;
        if config_min > min_val as i64 {
            min_val = config_min as i32;
        }
        if config_max < max_val as i64 {
            max_val = config_max as i32;
        }
    }

    mutate_validated_i32(value, rng, min_val, max_val)
}

pub fn mutate_validated_u32_with_config<V, R>(
    value: &mut V,
    rng: &mut R,
    config: &RandomConfig,
    is_memory_aware: bool,
) -> bool
where
    V: ValidatedValue<u32>,
    R: rand::Rng,
{
    let (mut min_val, mut max_val) = (V::MIN, V::MAX);
    if is_memory_aware {
        let (config_min, config_max) = config.memory_access_config.memory_offset_immediate_range;
        if config_min > min_val as i64 {
            min_val = config_min.max(0) as u32;
        }
        if config_max < max_val as i64 {
            if config_max < 0 {
                return false;
            }
            max_val = config_max as u32;
        }
    }

    mutate_validated_u32(value, rng, min_val, max_val)
}

fn configured_signed_random_bounds<V>(
    config: &RandomConfig,
) -> Result<(i64, i64), RandomGenerationError>
where
    V: ValidatedValueRandomRange<i32>,
{
    let (min_val, max_val) =
        <V as ValidatedValueRandomRange<i32>>::configured_random_range(config)?;
    Ok((min_val as i64, max_val as i64))
}

fn configured_unsigned_random_bounds<V>(
    config: &RandomConfig,
) -> Result<(u64, u64), RandomGenerationError>
where
    V: ValidatedValueRandomRange<u32>,
{
    let (min_val, max_val) =
        <V as ValidatedValueRandomRange<u32>>::configured_random_range(config)?;
    Ok((min_val as u64, max_val as u64))
}

pub(crate) fn random_validated_signed_with_sampler<V, R, S>(
    rng: &mut R,
    config: &RandomConfig,
    ty_name: &'static str,
    sampler: S,
) -> Result<V, RandomGenerationError>
where
    V: ValidatedValue<i32> + ValidatedValueRandomRange<i32>,
    V::Error: ToString,
    R: rand::Rng,
    S: Fn(&mut R, i64, i64) -> i64,
{
    let (min_val, max_val) = configured_signed_random_bounds::<V>(config)?;
    if min_val > max_val {
        return Err(RandomGenerationError::new(
            ty_name,
            format!("invalid combined range [{}, {}]", min_val, max_val),
        ));
    }

    const MAX_ATTEMPTS: usize = 10_000;
    let mut last_error: Option<String> = None;
    for _ in 0..MAX_ATTEMPTS {
        let candidate = sampler(rng, min_val, max_val) as i32;
        if V::FORBIDDEN.contains(&candidate) {
            continue;
        }
        if V::ODD_ONLY && candidate % 2 == 0 {
            continue;
        }
        if let Some(multiple) = V::MULTIPLE_OF {
            if multiple != 0 && candidate % multiple != 0 {
                continue;
            }
        }
        match V::new(candidate) {
            Ok(value) => return Ok(value),
            Err(err) => last_error = Some(err.to_string()),
        }
    }

    Err(RandomGenerationError::exhausted(
        ty_name,
        MAX_ATTEMPTS,
        last_error,
    ))
}

pub(crate) fn mutate_validated_signed_with_sampler<V, R, S>(
    value: &mut V,
    rng: &mut R,
    config: &RandomConfig,
    sampler: S,
) -> bool
where
    V: ValidatedValue<i32> + ValidatedValueRandomRange<i32>,
    R: rand::Rng,
    S: Fn(&mut R, i64, i64) -> i64,
{
    let (min_val, max_val) = match configured_signed_random_bounds::<V>(config) {
        Ok(range) => range,
        Err(_) => return false,
    };
    if min_val > max_val {
        return false;
    }

    const MAX_ATTEMPTS: usize = 256;
    for _ in 0..MAX_ATTEMPTS {
        let candidate = sampler(rng, min_val, max_val) as i32;
        if V::FORBIDDEN.contains(&candidate) {
            continue;
        }
        if V::ODD_ONLY && candidate % 2 == 0 {
            continue;
        }
        if let Some(multiple) = V::MULTIPLE_OF {
            if multiple != 0 && candidate % multiple != 0 {
                continue;
            }
        }
        if value.set(candidate).is_ok() {
            return true;
        }
    }

    false
}

pub(crate) fn random_validated_unsigned_with_sampler<V, R, S>(
    rng: &mut R,
    config: &RandomConfig,
    ty_name: &'static str,
    sampler: S,
) -> Result<V, RandomGenerationError>
where
    V: ValidatedValue<u32> + ValidatedValueRandomRange<u32>,
    V::Error: ToString,
    R: rand::Rng,
    S: Fn(&mut R, u64, u64) -> u64,
{
    let (min_u, max_u) = configured_unsigned_random_bounds::<V>(config)?;
    if min_u > max_u {
        return Err(RandomGenerationError::new(
            ty_name,
            format!("invalid combined range [{}, {}]", min_u, max_u),
        ));
    }

    const MAX_ATTEMPTS: usize = 10_000;
    let mut last_error: Option<String> = None;
    for _ in 0..MAX_ATTEMPTS {
        let candidate = sampler(rng, min_u, max_u) as u32;
        if V::FORBIDDEN.contains(&candidate) {
            continue;
        }
        if V::ODD_ONLY && candidate % 2 == 0 {
            continue;
        }
        if let Some(multiple) = V::MULTIPLE_OF {
            if multiple != 0 && candidate % multiple != 0 {
                continue;
            }
        }
        match V::new(candidate) {
            Ok(value) => return Ok(value),
            Err(err) => last_error = Some(err.to_string()),
        }
    }

    Err(RandomGenerationError::exhausted(
        ty_name,
        MAX_ATTEMPTS,
        last_error,
    ))
}

pub(crate) fn mutate_validated_unsigned_with_sampler<V, R, S>(
    value: &mut V,
    rng: &mut R,
    config: &RandomConfig,
    sampler: S,
) -> bool
where
    V: ValidatedValue<u32> + ValidatedValueRandomRange<u32>,
    R: rand::Rng,
    S: Fn(&mut R, u64, u64) -> u64,
{
    let (min_u, max_u) = match configured_unsigned_random_bounds::<V>(config) {
        Ok(range) => range,
        Err(_) => return false,
    };
    if min_u > max_u {
        return false;
    }

    const MAX_ATTEMPTS: usize = 256;
    for _ in 0..MAX_ATTEMPTS {
        let candidate = sampler(rng, min_u, max_u) as u32;
        if V::FORBIDDEN.contains(&candidate) {
            continue;
        }
        if V::ODD_ONLY && candidate % 2 == 0 {
            continue;
        }
        if let Some(multiple) = V::MULTIPLE_OF {
            if multiple != 0 && candidate % multiple != 0 {
                continue;
            }
        }
        if value.set(candidate).is_ok() {
            return true;
        }
    }

    false
}

pub(crate) fn random_validated_fixed_i32<V>(
    ty_name: &'static str,
    value: i32,
) -> Result<V, RandomGenerationError>
where
    V: ValidatedValue<i32>,
    V::Error: ToString,
{
    V::new(value).map_err(|err| RandomGenerationError::new(ty_name, err.to_string()))
}

pub(crate) fn mutate_validated_fixed_i32<V>(target: &mut V, value: i32) -> bool
where
    V: ValidatedValue<i32>,
{
    target.set(value).is_ok()
}

/// Common trait for validated value types
pub trait ValidatedValue<T: 'static> {
    const MIN: T;
    const MAX: T;
    const MULTIPLE_OF: Option<T> = None;
    const FORBIDDEN: &'static [T] = &[];
    const ODD_ONLY: bool = false;
    type Error;
    fn new(value: T) -> Result<Self, Self::Error>
    where
        Self: Sized;
    fn get(&self) -> T;
    fn set(&mut self, value: T) -> Result<(), Self::Error>;
}

/// Runtime hook describing how a validated newtype should derive its random range
/// from `RandomConfig`.
pub trait ValidatedValueRandomRange<T: 'static>: ValidatedValue<T> {
    fn configured_random_range(config: &RandomConfig) -> Result<(T, T), RandomGenerationError>;
}
