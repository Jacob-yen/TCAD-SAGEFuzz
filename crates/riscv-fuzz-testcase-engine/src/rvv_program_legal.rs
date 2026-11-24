//! Read-only, state-aware audit of RVV instruction preconditions in a program.
//! This does not accept/reject generated candidates or model trap-handler state.

use riscv_fuzz_plugin_api::ISABase;
use serde::Serialize;

use crate::{
    rvv_catalog_legal::{RvvSamplingProfile, check_asm_under_config, is_whole_register_load_store},
    rvv_constraints::RvvConfigConstraint,
    rvv_legal,
};

#[derive(Debug, Clone, Serialize)]
pub struct RvvProgramIssue {
    pub line: usize,
    pub instruction: String,
    pub code: &'static str,
    pub detail: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RvvProgramAudit {
    pub checked_vector_instructions: usize,
    pub unknown_vtype_instructions: usize,
    pub unknown_vstart_preconditions: usize,
    pub configured_vtypes: usize,
    pub issues: Vec<RvvProgramIssue>,
    pub assumption: &'static str,
}

#[derive(Clone, Copy)]
enum VtypeState {
    Unknown,
    Vill,
    Legal { sew: u16, lmul: &'static str },
}

fn parse_number(value: &str) -> Option<i64> {
    let value = value.trim();
    if let Some(hex) = value.strip_prefix("0x") {
        i64::from_str_radix(hex, 16).ok()
    } else if let Some(hex) = value.strip_prefix("-0x") {
        i64::from_str_radix(hex, 16).ok().map(|n| -n)
    } else {
        value.parse().ok()
    }
}

fn xreg(value: &str) -> Option<usize> {
    value
        .trim()
        .strip_prefix('x')?
        .parse::<usize>()
        .ok()
        .filter(|n| *n < 32)
}

fn vtype_from_immediate(value: i64, profile: RvvSamplingProfile) -> VtypeState {
    if !(0..=255).contains(&value) {
        return VtypeState::Vill;
    }
    let imm = value as u16;
    let sew = 8u16.checked_shl(u32::from((imm >> 3) & 7)).unwrap_or(0);
    let lmul = match imm & 7 {
        0 => "m1",
        1 => "m2",
        2 => "m4",
        3 => "m8",
        5 => "mf8",
        6 => "mf4",
        7 => "mf2",
        _ => return VtypeState::Vill,
    };
    if rvv_legal::vtype_supported(sew, lmul, profile.elen_bits) {
        VtypeState::Legal { sew, lmul }
    } else {
        VtypeState::Vill
    }
}

fn needs_zero_vstart(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "vcompress.vm" | "vmsbf.m" | "vmsif.m" | "vmsof.m" | "viota.m" | "vcpop.m" | "vfirst.m"
    ) || ["vred", "vwred", "vfred", "vfwred"]
        .iter()
        .any(|prefix| mnemonic.starts_with(prefix))
}

fn scalar_instruction_writes_first_gpr(mnemonic: &str) -> bool {
    // Treat unmodeled scalar results as unknown. Stores and branches read
    // their first operand, so must not kill a known input used by a later
    // vsetvl or vstart write.
    !matches!(
        mnemonic,
        "sb" | "sh"
            | "sw"
            | "sd"
            | "beq"
            | "bne"
            | "blt"
            | "bge"
            | "bltu"
            | "bgeu"
            | "beqz"
            | "bnez"
            | "bgtz"
            | "bltz"
            | "blez"
            | "bgez"
            | "bgt"
            | "ble"
            | "bgtu"
            | "bleu"
            | "jr"
    )
}

/// Checks what is statically decidable if earlier instructions complete
/// normally. A memory trap can change `vstart`; the Spike trace must resolve
/// that path before the result is treated as a dynamic legality finding.
pub fn audit_program_asm(source: &str, profile: RvvSamplingProfile) -> RvvProgramAudit {
    let body = source
        .split_once("testcase_code:")
        .map(|(_, after)| after.split_once("exit:").map_or(after, |(body, _)| body))
        .unwrap_or(source);
    let mut result = RvvProgramAudit {
        assumption: "static path, assuming preceding instructions complete without a trap",
        ..RvvProgramAudit::default()
    };
    let mut vtype = VtypeState::Unknown;
    let mut vstart = None;
    let mut known_x = [None::<i64>; 32];
    known_x[0] = Some(0);
    for (index, raw) in body.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() || line.ends_with(':') || line.starts_with('.') {
            continue;
        }
        let Some((mnemonic, args)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let operands = args.split(',').map(str::trim).collect::<Vec<_>>();
        if mnemonic == "li" && operands.len() == 2 {
            if let Some(reg) = xreg(operands[0]) {
                if reg != 0 {
                    known_x[reg] = parse_number(operands[1]);
                }
            }
            continue;
        }
        if mnemonic == "addi" && operands.len() == 3 {
            if let Some(rd) = xreg(operands[0]) {
                if rd != 0 {
                    known_x[rd] = xreg(operands[1])
                        .and_then(|rs| known_x[rs])
                        .zip(parse_number(operands[2]))
                        .and_then(|(base, offset)| base.checked_add(offset));
                }
            }
            continue;
        }
        if mnemonic == "csrrw" && operands.len() == 3 && operands[1] == "vstart" {
            let next_vstart = xreg(operands[2]).and_then(|rs| known_x[rs]);
            if let Some(rd) = xreg(operands[0])
                && rd != 0
            {
                known_x[rd] = vstart;
            }
            vstart = next_vstart;
            continue;
        }
        if mnemonic == "csrw" && operands.len() == 2 && operands[0] == "vstart" {
            vstart = xreg(operands[1]).and_then(|rs| known_x[rs]);
            continue;
        }
        if mnemonic.starts_with("csr") && operands.iter().any(|operand| *operand == "vstart") {
            vstart = None;
        }
        if mnemonic.starts_with("vset") && operands.len() == 3 {
            let vtype_value = if mnemonic == "vsetvl" {
                xreg(operands[2]).and_then(|reg| known_x[reg])
            } else {
                parse_number(operands[2])
            };
            vtype = vtype_value.map_or(VtypeState::Unknown, |value| {
                vtype_from_immediate(value, profile)
            });
            vstart = Some(0);
            if let Some(rd) = xreg(operands[0])
                && rd != 0
            {
                known_x[rd] = None;
            }
            result.configured_vtypes += 1;
            continue;
        }
        if !mnemonic.starts_with('v') {
            if scalar_instruction_writes_first_gpr(mnemonic)
                && let Some(rd) = operands.first().and_then(|operand| xreg(operand))
                && rd != 0
            {
                known_x[rd] = None;
            }
            continue;
        }
        result.checked_vector_instructions += 1;
        let mut issue = |code, detail: String| {
            result.issues.push(RvvProgramIssue {
                line: index + 1,
                instruction: line.to_string(),
                code,
                detail,
            })
        };
        if needs_zero_vstart(mnemonic) {
            match vstart {
                Some(0) => {}
                Some(value) => issue(
                    "vstart",
                    format!("{mnemonic} requires vstart=0, got {value}"),
                ),
                None => result.unknown_vstart_preconditions += 1,
            }
        }
        match vtype {
            VtypeState::Legal { sew, lmul } => {
                let cfg = RvvConfigConstraint::new(ISABase::Rv64, sew, 1, lmul, "ta", "ma");
                for finding in check_asm_under_config(line, cfg, profile.elen_bits) {
                    issue(finding.code, finding.detail);
                }
            }
            VtypeState::Vill if is_whole_register_load_store(mnemonic) => {
                let cfg = RvvConfigConstraint::new(ISABase::Rv64, 8, 1, "m1", "ta", "ma");
                for finding in check_asm_under_config(line, cfg, profile.elen_bits) {
                    issue(finding.code, finding.detail);
                }
            }
            VtypeState::Vill => issue("vill", "current vtype is unsupported".to_string()),
            VtypeState::Unknown if is_whole_register_load_store(mnemonic) => {
                let cfg = RvvConfigConstraint::new(ISABase::Rv64, 8, 1, "m1", "ta", "ma");
                for finding in check_asm_under_config(line, cfg, profile.elen_bits) {
                    issue(finding.code, finding.detail);
                }
            }
            VtypeState::Unknown => result.unknown_vtype_instructions += 1,
        }
        if matches!(mnemonic, "vcpop.m" | "vfirst.m" | "vmv.x.s" | "vfmv.x.f")
            && let Some(rd) = operands.first().and_then(|operand| xreg(operand))
            && rd != 0
        {
            known_x[rd] = None;
        }
        // Normal completion resets vstart; a faulting dynamic path is not
        // represented in this static audit.
        vstart = Some(0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_checker_finds_invalid_alias_across_a_config_transition() {
        let profile = RvvSamplingProfile::for_subject("xiangshan");
        let body = "vsetivli x0, 3, 16\nvadd.vv v16, v17, v18\nvslide1up.vx v18, v18, x13\n";
        let report = audit_program_asm(body, profile);
        assert_eq!(report.checked_vector_instructions, 2);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].code, "register_relation");
    }

    #[test]
    fn vstart_requirement_is_checked_at_the_consuming_instruction() {
        let profile = RvvSamplingProfile::for_subject("xiangshan");
        let body =
            "vsetivli x0, 4, 16\nli x5, 2\ncsrrw x0, vstart, x5\nvcompress.vm v16, v17, v18\n";
        let report = audit_program_asm(body, profile);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].code, "vstart");
    }

    #[test]
    fn all_nonrestartable_mask_and_reduction_families_need_zero_vstart() {
        let profile = RvvSamplingProfile::for_subject("xiangshan");
        for instruction in [
            "vmsbf.m v16, v17",
            "vmsof.m v16, v17",
            "viota.m v16, v17",
            "vcpop.m x6, v17",
            "vfirst.m x6, v17",
            "vredsum.vs v16, v17, v18",
            "vwredsum.vs v16, v17, v18",
            "vfredusum.vs v16, v17, v18",
            "vfwredosum.vs v16, v17, v18",
        ] {
            let body =
                format!("vsetivli x0, 4, 16\nli x5, 2\ncsrrw x0, vstart, x5\n{instruction}\n");
            let report = audit_program_asm(&body, profile);
            assert!(
                report.issues.iter().any(|issue| issue.code == "vstart"),
                "missing vstart issue for {instruction}: {report:?}"
            );
        }
    }

    #[test]
    fn whole_register_load_does_not_depend_on_vtype() {
        let profile = RvvSamplingProfile::for_subject("xiangshan");
        let body = "vsetivli x0, 4, 256\nvl1re8.v v16, (x11)\nvadd.vv v16, v17, v18\n";
        let report = audit_program_asm(body, profile);
        assert_eq!(report.issues.len(), 1, "{report:?}");
        assert_eq!(report.issues[0].instruction, "vadd.vv v16, v17, v18");
        assert_eq!(report.issues[0].code, "vill");
    }

    #[test]
    fn unmodeled_scalar_writer_does_not_leave_stale_vtype_constant() {
        let profile = RvvSamplingProfile::for_subject("xiangshan");
        let body = "li x5, 16\nadd x5, x5, x6\nvsetvl x0, x7, x5\nvadd.vv v16, v17, v18\n";
        let report = audit_program_asm(body, profile);
        assert_eq!(report.unknown_vtype_instructions, 1);
        assert!(report.issues.is_empty());
    }
}
