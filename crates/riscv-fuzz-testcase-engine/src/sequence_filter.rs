use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

use riscv_fuzz_plugin_api::{ISABase, IsaExtensions};
use riscv_instruction::{
    RV32Instruction, RV32ZicsrInstructions, RV64Instruction, RV64ZicsrInstructions,
    RiscvInstruction,
};
use riscv_instruction_types::{InstructionSequence, ValidatedValue, WritableCsr};
use serde::Deserialize;

pub fn remove_special_instructions(
    instructions: Vec<InstructionSequence<RiscvInstruction>>,
) -> Vec<InstructionSequence<RiscvInstruction>> {
    instructions
        .into_iter()
        .filter_map(|instruction| {
            let keep = match &instruction.instruction {
                RiscvInstruction::RV32(RV32Instruction::Zicsr(instr)) => {
                    !rv32_zicsr_writes_forbidden_csr(instr)
                }
                RiscvInstruction::RV64(RV64Instruction::Zicsr(instr)) => {
                    !rv64_zicsr_writes_forbidden_csr(instr)
                }
                inst => !is_special_instruction_mnemonic(&normalized_instruction_mnemonic(inst)),
            };
            if keep { Some(instruction) } else { None }
        })
        .collect()
}

pub fn remove_instructions_requiring_unsupported_extensions(
    instructions: Vec<InstructionSequence<RiscvInstruction>>,
    extensions: &IsaExtensions,
    isa_base: ISABase,
) -> Vec<InstructionSequence<RiscvInstruction>> {
    let allowed = allowed_extension_names(extensions);
    let requirements = instruction_requirements();

    instructions
        .into_iter()
        .filter(|sequence| {
            let mnemonic = exact_instruction_mnemonic(&sequence.instruction);
            let Some(requirement) = requirements.get(&mnemonic) else {
                return true;
            };

            allowed.contains(&requirement.extension)
                && requirement
                    .required_extensions_for(isa_base)
                    .iter()
                    .all(|required| allowed.contains(required))
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct InstructionRequirementRecord {
    name: String,
    extension: String,
    #[serde(default)]
    required_extensions: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InstructionRequirement {
    extension: String,
    rv32_required: Vec<String>,
    rv64_required: Vec<String>,
}

impl InstructionRequirement {
    fn required_extensions_for(&self, isa_base: ISABase) -> &[String] {
        match isa_base {
            ISABase::Rv32 => &self.rv32_required,
            ISABase::Rv64 => &self.rv64_required,
        }
    }
}

fn instruction_requirements() -> &'static HashMap<String, InstructionRequirement> {
    static REQUIREMENTS: OnceLock<HashMap<String, InstructionRequirement>> = OnceLock::new();

    REQUIREMENTS.get_or_init(|| {
        serde_json::from_str::<Vec<InstructionRequirementRecord>>(include_str!(
            "../../../third_party/riscv-instruction-crates/assets/riscv_instructions.json"
        ))
        .expect("parse riscv instruction requirements")
        .into_iter()
        .map(|mut record| {
            (
                record.name.to_ascii_lowercase(),
                InstructionRequirement {
                    extension: record.extension,
                    rv32_required: record
                        .required_extensions
                        .remove("RV32")
                        .unwrap_or_default(),
                    rv64_required: record
                        .required_extensions
                        .remove("RV64")
                        .unwrap_or_default(),
                },
            )
        })
        .collect()
    })
}

fn allowed_extension_names(extensions: &IsaExtensions) -> HashSet<String> {
    let mut allowed = match extensions {
        IsaExtensions::Rv32(values) => values.iter().map(|ext| format!("{ext:?}")).collect(),
        IsaExtensions::Rv64(values) => values.iter().map(|ext| format!("{ext:?}")).collect(),
    };
    apply_known_extension_implications(&mut allowed);
    allowed
}

fn apply_known_extension_implications(allowed: &mut HashSet<String>) {
    loop {
        let mut changed = false;

        if allowed.contains("D") {
            changed |= allowed.insert("F".to_string());
        }
        if allowed.contains("Q") {
            changed |= allowed.insert("D".to_string());
            changed |= allowed.insert("F".to_string());
        }
        if allowed.contains("Zfh") {
            changed |= allowed.insert("Zfhmin".to_string());
            changed |= allowed.insert("F".to_string());
        }
        if allowed.contains("Zfhmin") {
            changed |= allowed.insert("F".to_string());
        }
        if allowed.contains("Zfa") {
            changed |= allowed.insert("F".to_string());
        }
        if allowed.contains("Zcd") {
            changed |= allowed.insert("D".to_string());
        }
        if allowed.contains("Zcf") {
            changed |= allowed.insert("F".to_string());
        }

        if !changed {
            break;
        }
    }
}

fn exact_instruction_mnemonic(instruction: &RiscvInstruction) -> String {
    instruction
        .to_string()
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn normalized_instruction_mnemonic(instruction: &RiscvInstruction) -> String {
    exact_instruction_mnemonic(instruction).replace('.', "")
}

fn is_special_instruction_mnemonic(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "ecall"
            | "ebreak"
            | "mret"
            | "wfi"
            | "auipc"
            | "jal"
            | "jalr"
            | "beq"
            | "bne"
            | "blt"
            | "bge"
            | "bltu"
            | "bgeu"
            | "caddi4spn"
            | "cebreak"
            | "cj"
            | "cjal"
            | "cjr"
            | "cjalr"
            | "cbeqz"
            | "cbnez"
            | "sret"
            | "sfencevma"
            | "dret"
            | "mnret"
            | "vsetvli"
            | "vsetvl"
            | "vsetivli"
            | "sfencewinval"
            | "hinvalvvma"
            | "sfenceinvalir"
            | "hinvalgvma"
            | "sinvalvma"
            | "hfencegvma"
            | "hfencevvma"
            | "wrssto"
            | "wrsnto"
    )
}

fn rv32_zicsr_writes_forbidden_csr(instr: &RV32ZicsrInstructions) -> bool {
    use RV32ZicsrInstructions::*;

    let (csr, reads_csr, writes_csr) = match instr {
        CSRRW(inner) => (inner.csr, inner.xd.get() != 0, true),
        CSRRWI(inner) => (inner.csr, inner.xd.get() != 0, true),
        CSRRS(inner) => {
            let writes = inner.xs1.get() != 0;
            (inner.csr, inner.xd.get() != 0, writes)
        }
        CSRRSI(inner) => {
            let writes = inner.uimm.get() != 0;
            (inner.csr, inner.xd.get() != 0, writes)
        }
        CSRRC(inner) => {
            let writes = inner.xs1.get() != 0;
            (inner.csr, inner.xd.get() != 0, writes)
        }
        CSRRCI(inner) => {
            let writes = inner.uimm.get() != 0;
            (inner.csr, inner.xd.get() != 0, writes)
        }
    };

    csr_access_is_forbidden(csr, reads_csr, writes_csr)
}

fn rv64_zicsr_writes_forbidden_csr(instr: &RV64ZicsrInstructions) -> bool {
    use RV64ZicsrInstructions::*;

    let (csr, reads_csr, writes_csr) = match instr {
        CSRRW(inner) => (inner.csr, inner.xd.get() != 0, true),
        CSRRWI(inner) => (inner.csr, inner.xd.get() != 0, true),
        CSRRS(inner) => {
            let writes = inner.xs1.get() != 0;
            (inner.csr, inner.xd.get() != 0, writes)
        }
        CSRRSI(inner) => {
            let writes = inner.uimm.get() != 0;
            (inner.csr, inner.xd.get() != 0, writes)
        }
        CSRRC(inner) => {
            let writes = inner.xs1.get() != 0;
            (inner.csr, inner.xd.get() != 0, writes)
        }
        CSRRCI(inner) => {
            let writes = inner.uimm.get() != 0;
            (inner.csr, inner.xd.get() != 0, writes)
        }
    };

    csr_access_is_forbidden(csr, reads_csr, writes_csr)
}

fn csr_access_is_forbidden(csr: WritableCsr, reads_csr: bool, _writes_csr: bool) -> bool {
    if reads_csr
        && matches!(
            csr,
            WritableCsr::Menvcfg | WritableCsr::Menvcfgh | WritableCsr::Mcounteren
        )
    {
        return true;
    }

    matches!(
        csr,
        WritableCsr::Mtvec
            | WritableCsr::Misa
            | WritableCsr::Mstatus
            | WritableCsr::Mie
            | WritableCsr::Mip
            | WritableCsr::Mscratch
            | WritableCsr::Mepc
            | WritableCsr::Mcause
            | WritableCsr::Mtval
            | WritableCsr::Mtval2
            | WritableCsr::Cycle
            | WritableCsr::Time
            | WritableCsr::Instret
            | WritableCsr::Sstatus
            | WritableCsr::Sie
            | WritableCsr::Sip
            | WritableCsr::Hie
            | WritableCsr::Hip
            | WritableCsr::Hvip
            | WritableCsr::Vsstatus
            | WritableCsr::Vsie
            | WritableCsr::Vsip
            | WritableCsr::Pmpcfg0
            | WritableCsr::Pmpcfg1
            | WritableCsr::Pmpcfg2
            | WritableCsr::Pmpcfg3
            | WritableCsr::Pmpaddr0
            | WritableCsr::Pmpaddr1
            | WritableCsr::Pmpaddr2
            | WritableCsr::Pmpaddr3
            | WritableCsr::Pmpaddr4
            | WritableCsr::Pmpaddr5
            | WritableCsr::Pmpaddr6
            | WritableCsr::Pmpaddr7
            | WritableCsr::Pmpaddr8
            | WritableCsr::Pmpaddr9
            | WritableCsr::Pmpaddr10
            | WritableCsr::Pmpaddr11
            | WritableCsr::Pmpaddr12
            | WritableCsr::Pmpaddr13
            | WritableCsr::Pmpaddr14
            | WritableCsr::Pmpaddr15
    )
}

#[cfg(test)]
mod tests {
    use riscv_instruction::{
        RV64_I_ADDI, RV64_Zicsr_CSRRW, RV64Extensions, RV64IInstructions, RV64Instruction,
        RV64ZicsrInstructions, RiscvInstruction,
    };
    use riscv_instruction_types::{
        Immediate, InstructionSequence, IntegerRegister, ValidatedValue, WritableCsr,
    };

    use super::*;

    fn rv64_fadd_d(rd: u8, rs1: u8, rs2: u8) -> RiscvInstruction {
        RiscvInstruction::RV64(riscv_instruction::RV64Instruction::D(
            riscv_instruction::RV64DInstructions::FADD_D(riscv_instruction::RV64_D_FADD_D {
                fd: riscv_instruction_types::FloatingPointRegister::new(rd).expect("valid fd"),
                fs1: riscv_instruction_types::FloatingPointRegister::new(rs1).expect("valid fs1"),
                fs2: riscv_instruction_types::FloatingPointRegister::new(rs2).expect("valid fs2"),
                rm: riscv_instruction_types::RoundingMode::new(0).expect("valid rne"),
            }),
        ))
    }

    fn rv64_fli_d(fd: u8) -> RiscvInstruction {
        RiscvInstruction::RV64(riscv_instruction::RV64Instruction::D(
            riscv_instruction::RV64DInstructions::FLI_D(riscv_instruction::RV64_D_FLI_D {
                fd: riscv_instruction_types::FloatingPointRegister::new(fd).expect("valid fd"),
                uimm: riscv_instruction_types::FliConstant::MinNormal,
            }),
        ))
    }

    fn rv64_fmin_h(fd: u8, fs1: u8, fs2: u8) -> RiscvInstruction {
        RiscvInstruction::RV64(riscv_instruction::RV64Instruction::Zfh(
            riscv_instruction::RV64ZfhInstructions::FMIN_H(riscv_instruction::RV64_Zfh_FMIN_H {
                fd: riscv_instruction_types::FloatingPointRegister::new(fd).expect("valid fd"),
                fs1: riscv_instruction_types::FloatingPointRegister::new(fs1).expect("valid fs1"),
                fs2: riscv_instruction_types::FloatingPointRegister::new(fs2).expect("valid fs2"),
            }),
        ))
    }

    fn seq(instruction: RiscvInstruction) -> InstructionSequence<RiscvInstruction> {
        InstructionSequence::new(instruction)
    }

    fn rv64_addi(rd: u8, rs1: u8, imm: i16) -> RiscvInstruction {
        RiscvInstruction::RV64(RV64Instruction::I(RV64IInstructions::ADDI(RV64_I_ADDI {
            xd: IntegerRegister::new(rd).expect("valid rd"),
            xs1: IntegerRegister::new(rs1).expect("valid rs1"),
            imm: Immediate::<12>::new(imm.into()).expect("valid addi immediate"),
        })))
    }

    fn rv64_csrrw(rd: u8, csr: WritableCsr, rs1: u8) -> RiscvInstruction {
        RiscvInstruction::RV64(RV64Instruction::Zicsr(RV64ZicsrInstructions::CSRRW(
            RV64_Zicsr_CSRRW {
                xd: IntegerRegister::new(rd).expect("valid rd"),
                xs1: IntegerRegister::new(rs1).expect("valid rs1"),
                csr,
            },
        )))
    }

    #[test]
    fn unsupported_extension_filter_drops_zfa_sequences_without_zfa() {
        let sequences = vec![seq(rv64_fadd_d(13, 14, 15)), seq(rv64_fli_d(16))];
        let allowed = IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::F,
            RV64Extensions::D,
            RV64Extensions::Zicsr,
        ]);

        let filtered = remove_instructions_requiring_unsupported_extensions(
            sequences,
            &allowed,
            ISABase::Rv64,
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(
            exact_instruction_mnemonic(&filtered[0].instruction),
            "fadd.d"
        );
    }

    #[test]
    fn unsupported_extension_filter_keeps_zfa_sequences_when_zfa_is_enabled() {
        let sequences = vec![seq(rv64_fli_d(16))];
        let allowed = IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::F,
            RV64Extensions::D,
            RV64Extensions::Zfa,
            RV64Extensions::Zicsr,
        ]);

        let filtered = remove_instructions_requiring_unsupported_extensions(
            sequences,
            &allowed,
            ISABase::Rv64,
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(
            exact_instruction_mnemonic(&filtered[0].instruction),
            "fli.d"
        );
    }

    #[test]
    fn unsupported_extension_filter_treats_zfh_as_satisfying_zfhmin_requirements() {
        let sequences = vec![seq(rv64_fmin_h(13, 14, 15))];
        let allowed = IsaExtensions::rv64(vec![
            RV64Extensions::I,
            RV64Extensions::F,
            RV64Extensions::Zfh,
            RV64Extensions::Zicsr,
        ]);

        let filtered = remove_instructions_requiring_unsupported_extensions(
            sequences,
            &allowed,
            ISABase::Rv64,
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(
            exact_instruction_mnemonic(&filtered[0].instruction),
            "fmin.h"
        );
    }

    #[test]
    fn remove_special_instructions_drops_mtval2_csr_accesses() {
        let sequences = vec![
            seq(rv64_csrrw(1, WritableCsr::Mtval2, 8)),
            seq(rv64_addi(1, 1, 1)),
        ];

        let filtered = remove_special_instructions(sequences);

        assert_eq!(filtered.len(), 1);
        assert_eq!(exact_instruction_mnemonic(&filtered[0].instruction), "addi");
    }

    #[test]
    fn remove_special_instructions_drops_unstable_machine_csrs() {
        let sequences = vec![
            seq(rv64_csrrw(1, WritableCsr::Cycle, 8)),
            seq(rv64_csrrw(1, WritableCsr::Time, 8)),
            seq(rv64_csrrw(1, WritableCsr::Instret, 8)),
            seq(rv64_csrrw(1, WritableCsr::Mcause, 8)),
            seq(rv64_csrrw(1, WritableCsr::Mscratch, 8)),
            seq(rv64_csrrw(1, WritableCsr::Mtval, 8)),
            seq(rv64_addi(1, 1, 1)),
        ];

        let filtered = remove_special_instructions(sequences);

        assert_eq!(filtered.len(), 1);
        assert_eq!(exact_instruction_mnemonic(&filtered[0].instruction), "addi");
    }
}
