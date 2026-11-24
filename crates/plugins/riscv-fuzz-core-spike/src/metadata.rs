use riscv_fuzz_plugin_api::IsaExtensions;
use riscv_fuzz_plugin_api::metadata::{
    CoverageModeSupport, IsaExecutionProfile, IsaExecutionProfileMap, UnalignedAccessSupport,
};
use riscv_instruction::{RV32Extensions, RV64Extensions};

fn rv32_extensions() -> Vec<RV32Extensions> {
    vec![
        RV32Extensions::I,
        RV32Extensions::M,
        RV32Extensions::F,
        RV32Extensions::D,
        RV32Extensions::C,
        RV32Extensions::Zca,
        RV32Extensions::Zcb,
        RV32Extensions::Zcf,
        RV32Extensions::Zcd,
        RV32Extensions::Zaamo,
        RV32Extensions::Zabha,
        RV32Extensions::Zacas,
        RV32Extensions::Zalrsc,
        RV32Extensions::Zawrs,
        RV32Extensions::B,
        RV32Extensions::Zba,
        RV32Extensions::Zbb,
        RV32Extensions::Zbc,
        RV32Extensions::Zbkb,
        RV32Extensions::Zbkx,
        RV32Extensions::Zbs,
        RV32Extensions::Zfh,
        RV32Extensions::Zicfiss,
        RV32Extensions::Zicond,
        RV32Extensions::Zicsr,
        RV32Extensions::Zifencei,
        RV32Extensions::Zihintntl,
        RV32Extensions::Zihintpause,
        RV32Extensions::Zimop,
        RV32Extensions::Zcmop,
        RV32Extensions::Zknd,
        RV32Extensions::Zkne,
        RV32Extensions::Zknh,
        RV32Extensions::Zksed,
        RV32Extensions::Zksh,
    ]
}

fn rv64_extensions() -> Vec<RV64Extensions> {
    vec![
        RV64Extensions::I,
        RV64Extensions::M,
        RV64Extensions::F,
        RV64Extensions::D,
        RV64Extensions::C,
        RV64Extensions::Zvl32b,
        RV64Extensions::Zfa,
        RV64Extensions::Zca,
        RV64Extensions::Zcb,
        RV64Extensions::Zcd,
        RV64Extensions::Zaamo,
        RV64Extensions::Zabha,
        RV64Extensions::Zacas,
        RV64Extensions::Zalrsc,
        RV64Extensions::Zawrs,
        RV64Extensions::B,
        RV64Extensions::Zba,
        RV64Extensions::Zbb,
        RV64Extensions::Zbc,
        RV64Extensions::Zbkb,
        RV64Extensions::Zbkx,
        RV64Extensions::Zbs,
        RV64Extensions::Zfh,
        RV64Extensions::Zicfiss,
        RV64Extensions::Zicond,
        RV64Extensions::Zicsr,
        RV64Extensions::Zifencei,
        RV64Extensions::Zihintntl,
        RV64Extensions::Zihintpause,
        RV64Extensions::Zimop,
        RV64Extensions::Zcmop,
        RV64Extensions::Zkn,
        RV64Extensions::Zknd,
        RV64Extensions::Zkne,
        RV64Extensions::Zknh,
        RV64Extensions::Zksed,
        RV64Extensions::Zksh,
    ]
}

pub fn isa_profiles() -> IsaExecutionProfileMap {
    let rv32 = IsaExecutionProfile {
        supported: true,
        supports_dual_hart: true,
        coverage_mode_support: CoverageModeSupport {
            off: true,
            light: false,
            full: false,
        },
        unaligned_access_support: UnalignedAccessSupport {
            aligned: true,
            unaligned: true,
        },
        extensions: IsaExtensions::rv32(rv32_extensions()),
    };
    let rv64 = IsaExecutionProfile {
        extensions: IsaExtensions::rv64(rv64_extensions()),
        ..rv32.clone()
    };

    IsaExecutionProfileMap { rv32, rv64 }
}

pub fn program_mem_range() -> (u64, u64) {
    // Start at 0x80002000 to avoid .text and .tohost sections
    // Use 256MB range (0x8000_0000 to 0x8FFF_FFFF) which is reasonable for both RV32 and RV64
    // This ensures program memory only contains data, not code or peripherals
    (0x80002000, 0x8FFFFFFF)
}
