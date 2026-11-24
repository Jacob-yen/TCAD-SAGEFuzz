use crate::*;

#[test]
fn integer_register_reports_configured_range_errors() {
    let mut rng = rand::rng();
    let config = RandomConfig {
        register_operand_config: RegisterOperandConfig {
            integer_register_number_range: (10, 5),
            ..RegisterOperandConfig::default()
        },
        ..RandomConfig::default()
    };

    let err = IntegerRegister::random_with_rng(&mut rng, &config)
        .expect_err("expected configuration error");
    assert_eq!(err.ty(), "IntegerRegister");
    assert!(err.to_string().contains("invalid configured range"));
}

#[test]
fn saved_integer_register_detects_disjoint_ranges() {
    let mut rng = rand::rng();
    let config = RandomConfig {
        register_operand_config: RegisterOperandConfig {
            integer_register_number_range: (0, 7),
            ..RegisterOperandConfig::default()
        },
        ..RandomConfig::default()
    };

    let err = SavedIntegerRegister::random_with_rng(&mut rng, &config)
        .expect_err("expected disjoint range error");
    assert_eq!(err.ty(), "SavedIntegerRegister");
    assert!(
        err.to_string()
            .contains("no SavedIntegerRegister values overlap")
    );
}

#[test]
fn compressed_saved_integer_register_detects_disjoint_ranges() {
    let mut rng = rand::rng();
    let config = RandomConfig {
        register_operand_config: RegisterOperandConfig {
            integer_register_number_range: (24, 31),
            ..RegisterOperandConfig::default()
        },
        ..RandomConfig::default()
    };

    let err = CompressedSavedIntegerRegister::random_with_rng(&mut rng, &config)
        .expect_err("expected disjoint range error");
    assert_eq!(err.ty(), "CompressedSavedIntegerRegister");
    assert!(
        err.to_string()
            .contains("no CompressedSavedIntegerRegister values overlap")
    );
}

#[test]
fn base_address_register_uses_mem_config_range() {
    let mut rng = rand::rng();
    let config = RandomConfig {
        register_operand_config: RegisterOperandConfig {
            integer_register_number_range: (20, 22),
            ..RegisterOperandConfig::default()
        },
        memory_access_config: MemoryAccessConfig {
            memory_base_register_number_range: (6, 6),
            ..MemoryAccessConfig::default()
        },
        ..RandomConfig::default()
    };

    let reg = BaseAddressRegister::random_with_rng(&mut rng, &config)
        .expect("expected valid base address register");
    assert_eq!(reg.get(), 6);
}

#[test]
fn validated_value_random_range_uses_memory_immediate_constraints() {
    let config = RandomConfig {
        memory_access_config: MemoryAccessConfig {
            memory_offset_immediate_range: (-5, 7),
            ..MemoryAccessConfig::default()
        },
        ..RandomConfig::default()
    };

    let (min, max) =
        <Immediate<12, true> as ValidatedValueRandomRange<i32>>::configured_random_range(&config)
            .expect("expected valid configured range");

    assert_eq!((min, max), (-5, 7));
}

#[test]
fn validated_value_random_range_clamps_unsigned_memory_minimum() {
    let config = RandomConfig {
        memory_access_config: MemoryAccessConfig {
            memory_offset_immediate_range: (-11, 5),
            ..MemoryAccessConfig::default()
        },
        ..RandomConfig::default()
    };

    let (min, max) =
        <UImmediate<5, true> as ValidatedValueRandomRange<u32>>::configured_random_range(&config)
            .expect("expected valid configured range");

    assert_eq!((min, max), (0, 5));
}

#[test]
fn validated_value_random_range_rejects_negative_unsigned_maximum() {
    let config = RandomConfig {
        memory_access_config: MemoryAccessConfig {
            memory_offset_immediate_range: (-11, -1),
            ..MemoryAccessConfig::default()
        },
        ..RandomConfig::default()
    };

    let err =
        <UImmediate<5, true> as ValidatedValueRandomRange<u32>>::configured_random_range(&config)
            .expect_err("expected negative unsigned maximum to be rejected");

    assert_eq!(err.ty(), "UImmediate");
    assert!(err.to_string().contains("configured max is negative"));
}

#[test]
fn validated_value_random_range_keeps_non_memory_values_unrestricted() {
    let config = RandomConfig {
        memory_access_config: MemoryAccessConfig {
            memory_offset_immediate_range: (100, 100),
            ..MemoryAccessConfig::default()
        },
        ..RandomConfig::default()
    };

    let (min, max) =
        <RoundingMode as ValidatedValueRandomRange<u8>>::configured_random_range(&config)
            .expect("expected valid configured range");

    assert_eq!((min, max), (RoundingMode::MIN, RoundingMode::MAX));
}

#[test]
fn pseudo_instruction_parses_canonical_asm() {
    let label = PseudoInstruction::decode_asm_with_context(".Lloop:", IsaDecodeContext::RV64)
        .expect("expected label to parse");
    assert_eq!(
        label,
        PseudoInstruction::Label {
            name: ".Lloop".to_string(),
        }
    );
    assert_eq!(label.encode_asm(), ".Lloop:");

    let load_address =
        PseudoInstruction::decode_asm_with_context("la x1, 0x10", IsaDecodeContext::RV64)
            .expect("expected `la` to parse");
    assert_eq!(
        load_address,
        PseudoInstruction::LoadAddress {
            rd: IntegerRegister::new(1).expect("valid register"),
            address: XLenAddress::X64(0x10),
        }
    );
    assert_eq!(load_address.encode_asm(), "la x1, 0x10");

    let load_immediate =
        PseudoInstruction::decode_asm_with_context("li x2, -4", IsaDecodeContext::RV64)
            .expect("expected `li` to parse");
    assert_eq!(
        load_immediate,
        PseudoInstruction::LoadImmediate {
            rd: IntegerRegister::new(2).expect("valid register"),
            immediate: XLenImmediate::X64(-4),
            purpose: LoadImmediatePurpose::Generic,
        }
    );
    assert_eq!(load_immediate.encode_asm(), "li x2, -4");

    let load_label =
        PseudoInstruction::decode_asm_with_context("lla x3, label_7", IsaDecodeContext::RV64)
            .expect("expected `lla` to parse");
    assert_eq!(
        load_label,
        PseudoInstruction::LoadLabelAddress {
            rd: IntegerRegister::new(3).expect("valid register"),
            label: "label_7".to_string(),
        }
    );
    assert_eq!(load_label.encode_asm(), "lla x3, label_7");

    let mv = PseudoInstruction::decode_asm_with_context("mv x5, x6", IsaDecodeContext::RV64)
        .expect("expected `mv` to parse");
    assert_eq!(
        mv,
        PseudoInstruction::Move {
            rd: IntegerRegister::new(5).expect("valid register"),
            rs: IntegerRegister::new(6).expect("valid register"),
        }
    );
    assert_eq!(mv.encode_asm(), "mv x5, x6");

    let nop = PseudoInstruction::decode_asm_with_context("nop", IsaDecodeContext::RV64)
        .expect("expected `nop` to parse");
    assert_eq!(nop, PseudoInstruction::Nop);
    assert_eq!(nop.encode_asm(), "nop");

    let jump = PseudoInstruction::decode_asm_with_context("j label_9", IsaDecodeContext::RV64)
        .expect("expected `j` to parse");
    assert_eq!(
        jump,
        PseudoInstruction::Jump {
            label: "label_9".to_string(),
        }
    );
    assert_eq!(jump.encode_asm(), "j label_9");

    let jump_register =
        PseudoInstruction::decode_asm_with_context("jr x7", IsaDecodeContext::RV64)
            .expect("expected `jr` to parse");
    assert_eq!(
        jump_register,
        PseudoInstruction::JumpRegister {
            rs: IntegerRegister::new(7).expect("valid register"),
        }
    );
    assert_eq!(jump_register.encode_asm(), "jr x7");

    let ret = PseudoInstruction::decode_asm_with_context("ret", IsaDecodeContext::RV64)
        .expect("expected `ret` to parse");
    assert_eq!(ret, PseudoInstruction::Return);
    assert_eq!(ret.encode_asm(), "ret");

    let call = PseudoInstruction::decode_asm_with_context("call symbol_1", IsaDecodeContext::RV64)
        .expect("expected `call` to parse");
    assert_eq!(
        call,
        PseudoInstruction::Call {
            symbol: "symbol_1".to_string(),
        }
    );
    assert_eq!(call.encode_asm(), "call symbol_1");

    let tail = PseudoInstruction::decode_asm_with_context("tail symbol_2", IsaDecodeContext::RV64)
        .expect("expected `tail` to parse");
    assert_eq!(
        tail,
        PseudoInstruction::Tail {
            symbol: "symbol_2".to_string(),
        }
    );
    assert_eq!(tail.encode_asm(), "tail symbol_2");

    let read_fflags =
        PseudoInstruction::decode_asm_with_context("frflags x4", IsaDecodeContext::RV64)
            .expect("expected pseudo `frflags` to parse");
    assert_eq!(
        read_fflags,
        PseudoInstruction::ReadFflags {
            rd: IntegerRegister::new(4).expect("valid register"),
        }
    );
    assert_eq!(read_fflags.encode_asm(), "frflags x4");

    let read_fflags_alias = PseudoInstruction::decode_asm_with_context(
        "csrrs x4, fflags, x0",
        IsaDecodeContext::RV64,
    )
    .expect("expected `csrrs ..., fflags, x0` alias to parse");
    assert_eq!(read_fflags_alias, read_fflags);

    let read_vl = PseudoInstruction::decode_asm_with_context("csrr x17, vl", IsaDecodeContext::RV64)
        .expect("read-only vl CSR must parse");
    assert_eq!(read_vl.encode_asm(), "csrr x17, vl");
    let read_vl_alias = PseudoInstruction::decode_asm_with_context(
        "csrrs x17, vl, x0", IsaDecodeContext::RV64,
    ).expect("read-only CSRRS alias must parse");
    assert_eq!(read_vl_alias, read_vl);
}

#[test]
fn pseudo_instruction_rejects_unknown_asm() {
    let err = PseudoInstruction::decode_asm_with_context("custom.op x1, x2", IsaDecodeContext::RV64)
        .expect_err("unknown pseudo assembly should be rejected");
    assert_eq!(err.kind(), AsmParseErrorKind::InvalidInstruction);
}

#[test]
fn pseudo_instruction_decode_uses_xlen_context() {
    let rv32 = PseudoInstruction::decode_asm_with_context("li x1, 7", IsaDecodeContext::RV32)
        .expect("expected rv32 pseudo parse");
    let rv64 = PseudoInstruction::decode_asm_with_context("li x1, 7", IsaDecodeContext::RV64)
        .expect("expected rv64 pseudo parse");

    assert_eq!(
        rv32,
        PseudoInstruction::LoadImmediate {
            rd: IntegerRegister::new(1).expect("valid register"),
            immediate: XLenImmediate::X32(7),
            purpose: LoadImmediatePurpose::Generic,
        }
    );
    assert_eq!(
        rv64,
        PseudoInstruction::LoadImmediate {
            rd: IntegerRegister::new(1).expect("valid register"),
            immediate: XLenImmediate::X64(7),
            purpose: LoadImmediatePurpose::Generic,
        }
    );
}

#[test]
fn pseudo_instruction_decode_without_context_is_rejected() {
    let err = PseudoInstruction::decode_asm("li x1, 7")
        .expect_err("pseudo decode without ISA context should be rejected");
    assert_eq!(err.kind(), AsmParseErrorKind::InvalidInstruction);
    assert!(err
        .to_string()
        .contains("decode_asm_with_context"));
}

#[test]
fn pseudo_instruction_decode_enforces_xlen_boundaries() {
    let rv32_imm_err = PseudoInstruction::decode_asm_with_context(
        "li x1, 2147483648",
        IsaDecodeContext::RV32,
    )
    .expect_err("rv32 should reject oversized li literal");
    assert_eq!(rv32_imm_err.kind(), AsmParseErrorKind::InvalidOperand);

    let rv32_addr_err = PseudoInstruction::decode_asm_with_context(
        "la x1, 0x100000000",
        IsaDecodeContext::RV32,
    )
    .expect_err("rv32 should reject oversized la literal");
    assert_eq!(rv32_addr_err.kind(), AsmParseErrorKind::InvalidOperand);

    let rv64_imm = PseudoInstruction::decode_asm_with_context(
        "li x1, 2147483648",
        IsaDecodeContext::RV64,
    )
    .expect("rv64 should accept larger li literal");
    assert_eq!(rv64_imm.encode_asm(), "li x1, 2147483648");

    let rv64_addr = PseudoInstruction::decode_asm_with_context(
        "la x1, 0x100000000",
        IsaDecodeContext::RV64,
    )
    .expect("rv64 should accept larger la literal");
    assert_eq!(rv64_addr.encode_asm(), "la x1, 0x100000000");
}

#[test]
fn pseudo_instruction_validate_for_profile_checks_xlen_and_flen() {
    let rv64_li = PseudoInstruction::LoadImmediate {
        rd: IntegerRegister::new(1).expect("valid register"),
        immediate: XLenImmediate::X64(7),
        purpose: LoadImmediatePurpose::Generic,
    };
    let rv64_la = PseudoInstruction::LoadAddress {
        rd: IntegerRegister::new(1).expect("valid register"),
        address: XLenAddress::X64(0x10),
    };
    let read_fflags = PseudoInstruction::ReadFflags {
        rd: IntegerRegister::new(4).expect("valid register"),
    };

    rv64_li
        .validate_for_profile(IsaProfile::new(XLen::X64, None))
        .expect("rv64 immediate should validate under rv64 profile");
    rv64_la
        .validate_for_profile(IsaProfile::new(XLen::X64, None))
        .expect("rv64 address should validate under rv64 profile");
    read_fflags
        .validate_for_profile(IsaProfile::new(XLen::X64, Some(FLen::F32)))
        .expect("frflags should validate when floating support exists");

    let li_err = rv64_li
        .validate_for_profile(IsaProfile::rv32())
        .expect_err("rv64 immediate should fail under rv32 profile");
    assert!(li_err.to_string().contains("immediate payload requires"));

    let la_err = rv64_la
        .validate_for_profile(IsaProfile::rv32())
        .expect_err("rv64 address should fail under rv32 profile");
    assert!(la_err.to_string().contains("address payload requires"));

    let fflags_err = read_fflags
        .validate_for_profile(IsaProfile::rv64())
        .expect_err("frflags should fail without floating support");
    assert!(fflags_err.to_string().contains("FLEN >= 32"));
}

#[test]
fn instruction_sequence_decode_without_context_is_rejected() {
    let err = InstructionSequence::<PseudoInstruction>::decode_asm("li x1, 7")
        .expect_err("sequence decode without ISA context should be rejected");
    assert_eq!(err.kind(), AsmParseErrorKind::InvalidInstruction);
    assert!(err
        .to_string()
        .contains("decode_asm_with_context"));
}

#[test]
fn instruction_sequence_validate_for_profile_checks_nested_pseudos() {
    let sequence = InstructionSequence {
        pre_instructions: vec![PseudoInstruction::LoadImmediate {
            rd: IntegerRegister::new(1).expect("valid register"),
            immediate: XLenImmediate::X64(7),
            purpose: LoadImmediatePurpose::Generic,
        }],
        instruction: PseudoInstruction::ReadFflags {
            rd: IntegerRegister::new(2).expect("valid register"),
        },
        post_instructions: vec![],
    };

    let xlen_err = sequence
        .validate_asm_decode_context(IsaDecodeContext::RV32)
        .expect_err("rv64 pseudo payload should fail under rv32 context");
    assert!(xlen_err.to_string().contains("immediate payload requires"));

    let flen_err = sequence
        .validate_for_profile(IsaProfile::rv64())
        .expect_err("frflags should fail without floating support");
    assert!(flen_err.to_string().contains("FLEN >= 32"));
}

#[test]
fn float_immediate_random_requires_matching_flen() {
    let mut rng = rand::rng();
    let f_only = RandomConfig {
        isa_profile: IsaProfile::new(XLen::X64, Some(FLen::F32)),
        ..RandomConfig::default()
    };
    let no_fp = RandomConfig {
        isa_profile: IsaProfile::new(XLen::X64, None),
        ..RandomConfig::default()
    };

    FloatImmediate32::random_with_rng(&mut rng, &f_only).expect("f32 should be supported");
    FloatImmediate64::random_with_rng(&mut rng, &f_only)
        .expect_err("f64 immediate should require D/FLEN64");
    FloatImmediate32::random_with_rng(&mut rng, &no_fp)
        .expect_err("float immediate should require floating profile");
}

#[test]
fn pseudo_xlen_payloads_serde_roundtrip() {
    let immediate = XLenImmediate::X64(42);
    let immediate_json = serde_json::to_string(&immediate).expect("serialize xlen immediate");
    let decoded_immediate: XLenImmediate =
        serde_json::from_str(&immediate_json).expect("deserialize xlen immediate");
    assert_eq!(decoded_immediate, immediate);

    let address = XLenAddress::X64(0x1000);
    let address_json = serde_json::to_string(&address).expect("serialize xlen address");
    let decoded_address: XLenAddress =
        serde_json::from_str(&address_json).expect("deserialize xlen address");
    assert_eq!(decoded_address, address);
}

#[test]
fn pseudo_instruction_serde_roundtrip_and_profile_validation() {
    let pseudo = PseudoInstruction::LoadImmediate {
        rd: IntegerRegister::new(5).expect("valid register"),
        immediate: XLenImmediate::X64(42),
        purpose: LoadImmediatePurpose::Generic,
    };
    let json = serde_json::to_string(&pseudo).expect("serialize pseudo instruction");
    let decoded: PseudoInstruction = serde_json::from_str(&json).expect("deserialize pseudo instruction");
    assert_eq!(decoded, pseudo);

    decoded
        .validate_for_profile(IsaProfile::new(XLen::X64, None))
        .expect("rv64 payload should validate under rv64 profile");
    decoded
        .validate_for_profile(IsaProfile::rv32())
        .expect_err("rv64 payload should fail under rv32 profile");
}

#[test]
fn parse_asm_i64_literal_accepts_i64_min() {
    let parsed = parse_asm_i64_literal("-9223372036854775808")
        .expect("expected i64::MIN literal to parse");
    assert_eq!(parsed, i64::MIN);

    let parsed_scalar = <i64 as AsmScalar>::parse_asm_scalar("-9223372036854775808")
        .expect("expected i64 scalar parser to accept min");
    assert_eq!(parsed_scalar, i64::MIN);
}

#[test]
fn asm_operand_parsers_roundtrip_special_operands() {
    let rounding_mode =
        RoundingMode::decode_asm("dyn").expect("expected canonical rounding mode to parse");
    assert_eq!(rounding_mode.encode_asm(), "dyn");

    let fence_mode =
        FenceMode::decode_asm("iorw").expect("expected canonical fence mode to parse");
    assert_eq!(fence_mode.encode_asm(), "iorw");

    let fli_constant =
        FliConstant::decode_asm("0x1.0p-16").expect("expected FLI constant to parse");
    assert_eq!(fli_constant, FliConstant::PowerOfTwoN16);
    assert_eq!(fli_constant.encode_asm(), "0x1.0p-16");

    let saved_reg_list = SavedRegListWithStackAdjRv32::parse_display_parts("{ra, s0-s1}", 32)
        .expect("expected saved register list to parse");
    assert_eq!(saved_reg_list.encode_asm(), "{ra, s0-s1}, 32");

    let readable_csr =
        ReadableCsr::decode_asm("fflags").expect("expected readable CSR to parse");
    assert_eq!(readable_csr.encode_asm(), "fflags");

    let writable_csr =
        WritableCsr::decode_asm("frm").expect("expected writable CSR to parse");
    assert_eq!(writable_csr.encode_asm(), "frm");
}

#[test]
fn parse_asm_sign_extended_literal_supports_c_lui_display_form() {
    let parsed = parse_asm_sign_extended_literal("0xfffff", 20)
        .expect("expected c.lui display literal to parse");
    assert_eq!(parsed, -1);
}
