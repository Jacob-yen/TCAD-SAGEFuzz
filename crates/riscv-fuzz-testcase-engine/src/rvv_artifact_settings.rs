pub const ELEN_BITS: u16 = 64;
pub const ARA_VLEN_BITS: u32 = 4096;
pub const XIANGSHAN_VLEN_BITS: u32 = 128;

pub const NEGATIVE_PROGRAM_PERCENT: u8 = 5;

const _: () = assert!(NEGATIVE_PROGRAM_PERCENT <= 100);
