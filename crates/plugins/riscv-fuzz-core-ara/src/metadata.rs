use riscv_fuzz_plugin_api::IsaExtensions;
use riscv_fuzz_plugin_api::metadata::{
    CoverageModeSupport, IsaExecutionProfile, IsaExecutionProfileMap, UnalignedAccessSupport,
};
use riscv_instruction::RV64Extensions;

pub const ARA_VERILATOR_RAM_BASE: u64 = 0x8000_0000;
pub const ARA_VERILATOR_RAM_SIZE: u64 = 0x0010_0000;
pub const ARA_VERILATOR_RAM_END: u64 = ARA_VERILATOR_RAM_BASE + ARA_VERILATOR_RAM_SIZE - 1;
pub const ARA_VECTOR_MEMORY_HEADROOM: u64 = 0x0001_0000;
pub const ARA_FUZZ_MEM_START: u64 = 0x8001_0040;
pub const ARA_FUZZ_MEM_END: u64 = ARA_VERILATOR_RAM_END - ARA_VECTOR_MEMORY_HEADROOM;

pub fn isa_profiles() -> IsaExecutionProfileMap {
    IsaExecutionProfileMap {
        rv32: IsaExecutionProfile::unsupported(riscv_fuzz_plugin_api::ISABase::Rv32),
        rv64: IsaExecutionProfile {
            supported: true,
            supports_dual_hart: false,
            coverage_mode_support: CoverageModeSupport {
                off: true,
                light: false,
                full: false,
            },
            unaligned_access_support: UnalignedAccessSupport {
                aligned: true,
                unaligned: false,
            },
            extensions: IsaExtensions::rv64(vec![
                RV64Extensions::I,
                RV64Extensions::M,
                RV64Extensions::F,
                RV64Extensions::D,
                RV64Extensions::C,
                RV64Extensions::Zvl32b,
                RV64Extensions::Zca,
                RV64Extensions::Zcd,
                RV64Extensions::Zaamo,
                RV64Extensions::Zalrsc,
                RV64Extensions::Zicsr,
                RV64Extensions::Zifencei,
            ]),
        },
    }
}

pub fn program_mem_range() -> (u64, u64) {
    // Current ara_tb.cpp registers only the first 1 MiB at 0x8000_0000 for
    // lowRISC memutil `-l ram,<elf>,elf`, even though the SV testbench models
    // a larger DRAM. Keep generated fuzz memory inside that registered window,
    // away from reset/.text/.tohost data near the bottom of RAM, and leave
    // tail headroom for RVV whole-register/segment memory operations whose
    // implicit access width can exceed the scalar mem_size window.
    (ARA_FUZZ_MEM_START, ARA_FUZZ_MEM_END)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzz_memory_stays_inside_verilator_ram_window() {
        let (start, end) = program_mem_range();
        assert!(start >= ARA_VERILATOR_RAM_BASE);
        assert!(end <= ARA_VERILATOR_RAM_END);
        assert_eq!(ARA_VERILATOR_RAM_SIZE, 1024 * 1024);
        assert!(ARA_VERILATOR_RAM_END - end >= ARA_VECTOR_MEMORY_HEADROOM);
    }
}
