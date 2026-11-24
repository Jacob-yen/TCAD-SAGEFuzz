use riscv_fuzz_plugin_api::IsaExtensions;
use riscv_fuzz_plugin_api::metadata::{
    CoverageModeSupport, IsaExecutionProfile, IsaExecutionProfileMap, UnalignedAccessSupport,
};
use riscv_instruction::RV64Extensions;

pub fn isa_profiles() -> IsaExecutionProfileMap {
    IsaExecutionProfileMap {
        rv32: IsaExecutionProfile::unsupported(riscv_fuzz_plugin_api::ISABase::Rv32),
        rv64: IsaExecutionProfile {
            supported: true,
            supports_dual_hart: true,
            coverage_mode_support: CoverageModeSupport::ALL,
            unaligned_access_support: UnalignedAccessSupport {
                aligned: true,
                unaligned: true,
            },
            extensions: IsaExtensions::rv64(vec![
                RV64Extensions::I,
                RV64Extensions::M,
                RV64Extensions::F,
                RV64Extensions::D,
                RV64Extensions::C,
                RV64Extensions::Zvl32b,
                RV64Extensions::Zca,
                RV64Extensions::Zcb,
                RV64Extensions::Zcd,
                RV64Extensions::Zcmop,
                RV64Extensions::Zaamo,
                RV64Extensions::Zalrsc,
                RV64Extensions::Zacas,
                RV64Extensions::Zawrs,
                RV64Extensions::B,
                RV64Extensions::Zba,
                RV64Extensions::Zbb,
                RV64Extensions::Zbc,
                RV64Extensions::Zbkb,
                RV64Extensions::Zbkx,
                RV64Extensions::Zbs,
                RV64Extensions::Zfa,
                RV64Extensions::Zfh,
                RV64Extensions::Zicond,
                RV64Extensions::Zicsr,
                RV64Extensions::Zifencei,
                RV64Extensions::Zihintntl,
                RV64Extensions::Zihintpause,
                RV64Extensions::Zimop,
                // Keep the default set scalar-only and march-valid for this plugin path.
                RV64Extensions::Zkn,
                RV64Extensions::Zknd,
                RV64Extensions::Zkne,
                RV64Extensions::Zknh,
                RV64Extensions::Zksed,
                RV64Extensions::Zksh,
            ]),
        },
    }
}

pub fn program_mem_range() -> (u64, u64) {
    // Keep fuzz memory inside the ELF-loaded .data segment so difftest sees the same
    // initialized RAM image, but place it far enough from .text that large dual-hart
    // programs cannot self-modify by accident.
    //
    // The linker script fixes .data at 0x8001_0000. Barrier flags consume the first
    // 0x40 bytes, and fuzz_mem_pool reserves the following 4096 bytes.
    (0x8001_0040, 0x8001_103F)
}
