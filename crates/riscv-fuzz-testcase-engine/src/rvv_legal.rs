//! Shared register-footprint calculations for VC, SF, and grammar targets.
//! The opcode remains selected while legal configuration and operands vary.

pub(crate) fn lmul_ratio(lmul: &str) -> Option<(u16, u16)> {
    match lmul {
        "mf8" => Some((1, 8)),
        "mf4" => Some((1, 4)),
        "mf2" => Some((1, 2)),
        "m1" => Some((1, 1)),
        "m2" => Some((2, 1)),
        "m4" => Some((4, 1)),
        "m8" => Some((8, 1)),
        _ => None,
    }
}

/// RVV 1.0, integer extension: the source EEW/EMUL are the destination
/// SEW/LMUL divided by the encoded extension factor. This is an instruction
/// property, not a fixed SEW choice shared by an entire template.
pub(crate) fn extension_factor(mnemonic: &str) -> Option<u16> {
    let suffix = mnemonic
        .strip_prefix("vzext.vf")
        .or_else(|| mnemonic.strip_prefix("vsext.vf"))?;
    match suffix {
        "2" => Some(2),
        "4" => Some(4),
        "8" => Some(8),
        _ => None,
    }
}

/// RVV 1.0 §§3.4, 5, 7, 12–14: reusable vtype preconditions for the
/// base-V opcode families supported by the current ELEN=64 profile.
/// Instruction-specific operand/layout checks are applied separately.
pub(crate) fn opcode_config_eligible(mnemonic: &str, sew: u16, lmul: &str, elen: u16) -> bool {
    if !vtype_supported(sew, lmul, elen) {
        return false;
    }
    if !crate::rvv_spike_guard_model::config_allows(mnemonic, sew, lmul, elen) {
        return false;
    }
    // The current base-V profile does not request Zvfh. Keep e16 FP for a
    // separate, explicitly extension-enabled profile.
    let floating =
        (mnemonic.starts_with("vf") && mnemonic != "vfirst.m") || mnemonic.starts_with("vmf");
    if floating && !matches!(sew, 32 | 64) {
        return false;
    }
    let doubled = mnemonic.starts_with("vfw")
        || mnemonic.starts_with("vfncvt")
        || mnemonic.starts_with("vw")
        || mnemonic.starts_with("vnclip")
        || mnemonic.starts_with("vnsra")
        || mnemonic.starts_with("vnsrl");
    if doubled && sew.saturating_mul(2) > elen {
        return false;
    }
    if let Some(factor) = extension_factor(mnemonic) {
        if sew % factor != 0 || sew / factor < 8 {
            return false;
        }
        return register_width(sew, lmul, sew / factor).is_some();
    }
    true
}

/// RVV 1.0 §3.4.2: conditional destination/source group overlap. A source
/// with fractional EMUL cannot use the widening high-end exception.
pub(crate) fn destination_source_overlap_allowed(
    dest_base: u8,
    dest_width: u8,
    dest_eew: u16,
    source_base: u8,
    source_width: u8,
    source_eew: u16,
    source_emul_at_least_one: bool,
) -> bool {
    if !groups_overlap(dest_base, dest_width, source_base, source_width) {
        return true;
    }
    if dest_eew == source_eew {
        return true;
    }
    if dest_eew < source_eew {
        return dest_base == source_base;
    }
    source_emul_at_least_one
        && u16::from(dest_base) + u16::from(dest_width)
            == u16::from(source_base) + u16::from(source_width)
}

pub(crate) fn emul_at_least_one(sew: u16, lmul: &str, eew: u16) -> bool {
    lmul_ratio(lmul).is_some_and(|(num, den)| {
        u32::from(num) * u32::from(eew) >= u32::from(den) * u32::from(sew)
    })
}

/// The current base-V execution profile has ELEN=64. A supported vtype must
/// satisfy the architectural SEW/LMUL limit, including fractional LMUL.
pub(crate) fn vtype_supported(sew: u16, lmul: &str, elen: u16) -> bool {
    lmul_ratio(lmul).is_some_and(|(num, den)| {
        sew <= elen && u32::from(sew) * u32::from(den) <= u32::from(elen) * u32::from(num)
    })
}

/// Return the register count of an operand with EEW under the current vtype.
/// Fractional EMUL occupies one architectural register and is legal only in
/// the inclusive range 1/8..=8.
pub(crate) fn register_width(sew: u16, lmul: &str, eew: u16) -> Option<u8> {
    let (num, den) = lmul_ratio(lmul)?;
    if sew == 0 || eew == 0 {
        return None;
    }
    let numerator = u32::from(num) * u32::from(eew);
    let denominator = u32::from(den) * u32::from(sew);
    if numerator * 8 < denominator || numerator > denominator * 8 {
        return None;
    }
    if numerator <= denominator {
        return Some(1);
    }
    if numerator % denominator != 0 {
        return None;
    }
    let width = numerator / denominator;
    (width == 2 || width == 4 || width == 8).then_some(width as u8)
}

pub(crate) fn valid_group(base: u8, width: u8) -> bool {
    width > 0
        && base < 32
        && u16::from(base) + u16::from(width) <= 32
        && (width == 1 || base % width == 0)
}

pub(crate) fn groups_overlap(a: u8, a_width: u8, b: u8, b_width: u8) -> bool {
    u16::from(a) < u16::from(b) + u16::from(b_width)
        && u16::from(b) < u16::from(a) + u16::from(a_width)
}

pub(crate) fn legal_group_bases(range: (u8, u8), width: u8) -> Vec<u8> {
    (range.0.max(1)..=range.1.min(31))
        .filter(|base| valid_group(*base, width))
        .filter(|base| u16::from(*base) + u16::from(width) - 1 <= u16::from(range.1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_eew_changes_register_footprint() {
        assert!(!vtype_supported(64, "mf2", 64));
        assert!(vtype_supported(32, "mf2", 64));
        assert!(vtype_supported(16, "mf4", 64));
        assert_eq!(register_width(16, "m1", 32), Some(2));
        assert_eq!(register_width(8, "m1", 32), Some(4));
        assert_eq!(register_width(64, "m1", 32), Some(1));
        assert_eq!(register_width(8, "m8", 64), None);
        assert_eq!(register_width(64, "mf8", 8), None);
        assert!(valid_group(28, 4));
        assert!(!valid_group(29, 4));
    }

    #[test]
    fn group_candidates_preserve_alignment_and_range() {
        assert_eq!(legal_group_bases((13, 31), 4), vec![16, 20, 24, 28]);
        assert!(groups_overlap(16, 4, 19, 1));
        assert!(!groups_overlap(16, 4, 20, 4));
    }

    #[test]
    fn extension_configs_are_derived_per_opcode_not_fixed_to_e64() {
        assert!(opcode_config_eligible("vzext.vf2", 16, "m1", 64));
        assert!(opcode_config_eligible("vzext.vf4", 32, "m1", 64));
        assert!(opcode_config_eligible("vzext.vf8", 64, "m1", 64));
        assert!(!opcode_config_eligible("vzext.vf2", 8, "m1", 64));
        assert!(!opcode_config_eligible("vzext.vf8", 64, "mf8", 64));
        assert!(opcode_config_eligible("vadd.vv", 8, "mf8", 64));
        assert!(opcode_config_eligible("vadd.vv", 64, "m8", 64));
    }

    #[test]
    fn documented_overlap_exceptions_are_conditional() {
        assert!(destination_source_overlap_allowed(0, 8, 32, 6, 2, 8, true));
        assert!(!destination_source_overlap_allowed(0, 8, 32, 4, 2, 8, true));
        assert!(!destination_source_overlap_allowed(
            0, 1, 16, 0, 1, 8, false
        ));
        assert!(destination_source_overlap_allowed(0, 1, 16, 0, 2, 32, true));
        assert!(!destination_source_overlap_allowed(
            1, 1, 16, 0, 2, 32, true
        ));
    }
}
