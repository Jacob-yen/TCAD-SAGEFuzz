/// Demonstrates all serialization/deserialization layers:
///
///  1. Assembly layer   — AsmEncode::encode_asm / AsmDecodeWithContext::decode_asm_with_context
///                        (Display is a thin delegate to encode_asm)
///  2. Serde layer      — serde_json (structured JSON, round-trips identity)
///
/// Covers every hand-written impl:
///   Register<P>, Immediate, RoundingMode, FenceMode, FliConstant,
///   WritableCsr, ReadableCsr, SavedRegListWithStackAdj, PseudoInstruction,
///   InstructionSequence
/// Plus macro-generated impls (DeriveAsmEncode, DeriveValidatedValue,
/// code_gen structs and enums).
use riscv_instruction::*;
use riscv_instruction_types::{
    AsmDecode, AsmDecodeWithContext, AsmEncode, FloatingPointRegister, InstructionSequence,
    IntegerRegister, IsaDecodeContext, PseudoInstruction, RoundingMode,
    SavedRegListWithStackAdjRv32, WritableCsr,
};
use serde_json::{from_str as json_from_str, to_string_pretty as json_to_pretty};

fn section(title: &str) {
    println!("\n========== {} ==========", title);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ------------------------------------------------------------------ //
    // 1. Operand types — AsmEncode / AsmDecode
    // ------------------------------------------------------------------ //
    section("1. Operand types — AsmEncode / AsmDecode");

    // IntegerRegister
    let xd = IntegerRegister::new(5)?;
    let xs1 = IntegerRegister::new(6)?;
    println!("IntegerRegister encode_asm : {}", xd.encode_asm());
    let xd_rt = IntegerRegister::decode_asm("x5")?;
    assert_eq!(xd_rt, xd);
    println!("IntegerRegister decode_asm : {} -> ok", xd_rt.encode_asm());

    // FloatingPointRegister
    let fd = FloatingPointRegister::new(0)?;
    let fs1 = FloatingPointRegister::new(1)?;
    let fs2 = FloatingPointRegister::new(2)?;
    println!("FloatingPointRegister encode_asm : {}", fd.encode_asm());
    let fd_rt = FloatingPointRegister::decode_asm("f0")?;
    assert_eq!(fd_rt, fd);
    println!("FloatingPointRegister decode_asm : {} -> ok", fd_rt.encode_asm());

    // RoundingMode — from/to string
    let rm = RoundingMode::decode_asm("rne")?;
    println!("RoundingMode encode_asm : {}", rm.encode_asm());
    let rm_rt = RoundingMode::decode_asm(&rm.encode_asm())?;
    assert_eq!(rm_rt, rm);
    println!("RoundingMode decode_asm round-trip: ok");

    // WritableCsr
    let csr = WritableCsr::decode_asm("frm")?;
    println!("WritableCsr encode_asm : {}", csr.encode_asm());
    let csr_rt = WritableCsr::decode_asm(&csr.encode_asm())?;
    assert_eq!(csr_rt, csr);
    println!("WritableCsr decode_asm round-trip: ok");

    // SavedRegListWithStackAdj (Zcmp)
    let saved = SavedRegListWithStackAdjRv32::parse_display_parts("{ra, s0-s3}", 32)?;
    println!("SavedRegList encode_asm : {}", saved.encode_asm());
    let saved_rt = SavedRegListWithStackAdjRv32::decode_asm(&saved.encode_asm())?;
    assert_eq!(saved_rt, saved);
    println!("SavedRegList decode_asm round-trip: ok");

    // FliConstant
    use riscv_instruction_types::FliConstant;
    let fli = FliConstant::decode_asm("0x1.0p-16")?;
    println!("FliConstant encode_asm : {}", fli.encode_asm());
    let fli_rt = FliConstant::decode_asm(&fli.encode_asm())?;
    assert_eq!(fli_rt, fli);
    println!("FliConstant decode_asm round-trip: ok");

    // PseudoInstruction
    let pseudo = PseudoInstruction::decode_asm_with_context("li x5, 42", IsaDecodeContext::RV64)?;
    println!("PseudoInstruction encode_asm : {}", pseudo.encode_asm());
    let pseudo_rt = PseudoInstruction::decode_asm_with_context(&pseudo.encode_asm(), IsaDecodeContext::RV64)?;
    assert_eq!(pseudo_rt, pseudo);
    println!("PseudoInstruction decode_asm round-trip: ok");

    // ------------------------------------------------------------------ //
    // 2. Instruction structs (macro-generated) — AsmEncode / AsmDecode
    // ------------------------------------------------------------------ //
    section("2. Instruction structs — AsmEncode / AsmDecode");

    // ADDI (I extension)
    let addi = RV32_I_ADDI {
        imm: riscv_instruction_types::Immediate::<12>::new(100)?,
        xs1,
        xd,
    };
    println!("ADDI encode_asm : {}", addi.encode_asm());
    let addi_rt = RV32_I_ADDI::decode_asm(&addi.encode_asm())?;
    assert_eq!(addi_rt, addi);
    println!("ADDI decode_asm round-trip: ok");

    // FENCE (I extension, FenceMode operands)
    let fence = RV32_I_FENCE {
        fm: riscv_instruction_types::UImmediate::<4>::new(0)?,
        pred: riscv_instruction_types::FenceMode::decode_asm("iorw")?,
        succ: riscv_instruction_types::FenceMode::decode_asm("iorw")?,
        xs1: IntegerRegister::new(0)?,
        xd: IntegerRegister::new(0)?,
    };
    println!("FENCE encode_asm : {}", fence.encode_asm());
    let fence_rt = RV32_I_FENCE::decode_asm(&fence.encode_asm())?;
    assert_eq!(fence_rt, fence);
    println!("FENCE decode_asm round-trip: ok");

    // FADD.S (F extension, RoundingMode operand)
    let fadd_s = RV32_F_FADD_S {
        fd,
        fs1,
        fs2,
        rm: RoundingMode::decode_asm("rne")?,
    };
    println!("FADD.S encode_asm : {}", fadd_s.encode_asm());
    let fadd_s_rt = RV32_F_FADD_S::decode_asm(&fadd_s.encode_asm())?;
    assert_eq!(fadd_s_rt, fadd_s);
    println!("FADD.S decode_asm round-trip: ok");

    // CSRRW (Zicsr extension, WritableCsr operand)
    let csrrw = RV32_Zicsr_CSRRW {
        csr: WritableCsr::decode_asm("frm")?,
        xs1,
        xd,
    };
    println!("CSRRW encode_asm : {}", csrrw.encode_asm());
    let csrrw_rt = RV32_Zicsr_CSRRW::decode_asm(&csrrw.encode_asm())?;
    assert_eq!(csrrw_rt, csrrw);
    println!("CSRRW decode_asm round-trip: ok");

    // FLI.S (Zfa extension, FliConstant operand)
    let fli_s = RV32_Zfa_FLI_S {
        fd,
        uimm: FliConstant::decode_asm("0x1.0p-16")?,
    };
    println!("FLI.S encode_asm : {}", fli_s.encode_asm());
    let fli_s_rt = RV32_Zfa_FLI_S::decode_asm(&fli_s.encode_asm())?;
    assert_eq!(fli_s_rt, fli_s);
    println!("FLI.S decode_asm round-trip: ok");

    // ------------------------------------------------------------------ //
    // 3. Enum wrappers — AsmEncode / AsmDecode
    //    Display and FromStr are thin delegates to the same impls;
    //    encode_asm / decode_asm are the primary interface.
    // ------------------------------------------------------------------ //
    section("3. Enum wrappers — AsmEncode / AsmDecodeWithContext");

    let addi_instr =
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(addi.clone())));
    println!("RiscvInstruction encode_asm : {}", addi_instr.encode_asm());
    let addi_instr_rt = RiscvInstruction::decode_asm_with_context(&addi_instr.encode_asm(), IsaDecodeContext::RV32)?;
    assert_eq!(addi_instr_rt, addi_instr);
    println!("RiscvInstruction decode_asm round-trip: ok");

    let fadd_instr =
        RiscvInstruction::RV32(RV32Instruction::F(RV32FInstructions::FADD_S(fadd_s.clone())));
    println!("RiscvInstruction(FADD.S) encode_asm : {}", fadd_instr.encode_asm());
    let fadd_instr_rt = RiscvInstruction::decode_asm_with_context(&fadd_instr.encode_asm(), IsaDecodeContext::RV32)?;
    assert_eq!(fadd_instr_rt, fadd_instr);
    println!("RiscvInstruction(FADD.S) decode_asm round-trip: ok");

    // ------------------------------------------------------------------ //
    // 4. Serde JSON — structured round-trip
    // ------------------------------------------------------------------ //
    section("4. Serde JSON — structured round-trip");

    // Single instruction struct
    let addi_json = json_to_pretty(&addi)?;
    println!("ADDI JSON:\n{}", addi_json);
    let addi_from_json: RV32_I_ADDI = json_from_str(&addi_json)?;
    assert_eq!(addi_from_json, addi);
    println!("ADDI JSON round-trip: ok");

    // Top-level enum
    let fadd_json = json_to_pretty(&fadd_instr)?;
    println!("RiscvInstruction(FADD.S) JSON:\n{}", fadd_json);
    let fadd_from_json: RiscvInstruction = json_from_str(&fadd_json)?;
    assert_eq!(fadd_from_json, fadd_instr);
    println!("RiscvInstruction(FADD.S) JSON round-trip: ok");

    // ------------------------------------------------------------------ //
    // 5. InstructionSequence — AsmEncode / AsmDecode / Serde
    // ------------------------------------------------------------------ //
    section("5. InstructionSequence — AsmEncode / AsmDecode / Serde");

    // Sequence: pseudo pre-instruction + real instruction
    let la_pseudo = RiscvInstruction::Pseudo(PseudoInstruction::LoadAddress {
        rd: xd,
        address: XLenAddress::X32(0x1000),
    });
    let addi_main =
        RiscvInstruction::RV32(RV32Instruction::I(RV32IInstructions::ADDI(addi.clone())));

    let seq = InstructionSequence {
        pre_instructions: vec![la_pseudo],
        instruction: addi_main.clone(),
        post_instructions: vec![],
    };

    let seq_asm = seq.encode_asm();
    println!("Sequence encode_asm:\n{}", seq_asm);
    let seq_rt = InstructionSequence::<RiscvInstruction>::decode_asm_with_context(&seq_asm, IsaDecodeContext::RV32)?;
    assert_eq!(seq_rt, seq);
    println!("Sequence decode_asm round-trip: ok");

    // Single-instruction sequence
    let simple_seq = InstructionSequence::new(addi_main.clone());
    println!("Simple sequence encode_asm : {}", simple_seq.encode_asm());
    let simple_seq_rt =
        InstructionSequence::<RiscvInstruction>::decode_asm_with_context(&simple_seq.encode_asm(), IsaDecodeContext::RV32)?;
    assert_eq!(simple_seq_rt, simple_seq);
    println!("Simple sequence decode_asm round-trip: ok");

    // Serde JSON round-trip for InstructionSequence
    let seq_json = json_to_pretty(&seq)?;
    println!("Sequence JSON:\n{}", seq_json);
    let seq_from_json: InstructionSequence<RiscvInstruction> = json_from_str(&seq_json)?;
    assert_eq!(seq_from_json, seq);
    println!("Sequence JSON round-trip: ok");

    // ------------------------------------------------------------------ //
    // 6. Parse error — unknown mnemonic
    // ------------------------------------------------------------------ //
    section("6. Parse error — unknown mnemonic");

    let err = RiscvInstruction::decode_asm_with_context("nop_does_not_exist x1, x2", IsaDecodeContext::RV64);
    match err {
        Err(e) => println!("Expected error: {}", e),
        Ok(_) => panic!("should have failed"),
    }

    println!("\nAll assertions passed.");
    Ok(())
}
