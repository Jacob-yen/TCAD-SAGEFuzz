use std::ffi::c_void;

pub const RISCV_FUZZ_PLUGIN_ABI_VERSION: u32 = 6;
pub const RISCV_FUZZ_PLUGIN_DESCRIPTOR_SYMBOL_PREFIX: &str = "riscv_fuzz_plugin_descriptor_v6__";

pub const PLUGIN_CALL_STATUS_OK: u32 = 0;
pub const PLUGIN_CALL_STATUS_ERR: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginBytes {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
}

impl PluginBytes {
    pub const EMPTY: Self = Self {
        ptr: std::ptr::null_mut(),
        len: 0,
        cap: 0,
    };

    pub fn from_vec(bytes: Vec<u8>) -> Self {
        let mut bytes = std::mem::ManuallyDrop::new(bytes);
        Self {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
            cap: bytes.capacity(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginSlice {
    pub ptr: *const u8,
    pub len: usize,
}

impl From<&[u8]> for PluginSlice {
    fn from(value: &[u8]) -> Self {
        Self {
            ptr: value.as_ptr(),
            len: value.len(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginCallResult {
    pub status: u32,
    pub bytes: PluginBytes,
}

impl PluginCallResult {
    pub fn ok(bytes: Vec<u8>) -> Self {
        Self {
            status: PLUGIN_CALL_STATUS_OK,
            bytes: PluginBytes::from_vec(bytes),
        }
    }

    pub fn err(message: impl Into<String>) -> Self {
        Self {
            status: PLUGIN_CALL_STATUS_ERR,
            bytes: PluginBytes::from_vec(message.into().into_bytes()),
        }
    }
}

pub type PluginStreamingEmit = unsafe extern "C" fn(ctx: *mut c_void, event: PluginSlice) -> bool;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PluginDescriptorV3 {
    pub abi_version: u32,
    pub metadata_payload: unsafe extern "C" fn() -> PluginCallResult,
    pub build_asm_payload: unsafe extern "C" fn(PluginSlice) -> PluginCallResult,
    pub validate_execute_request_payload: unsafe extern "C" fn(PluginSlice) -> PluginCallResult,
    pub execute_payload: unsafe extern "C" fn(PluginSlice) -> PluginCallResult,
    pub execute_streaming_payload:
        unsafe extern "C" fn(PluginSlice, *mut c_void, PluginStreamingEmit) -> PluginCallResult,
    pub free_bytes: unsafe extern "C" fn(PluginBytes),
}
