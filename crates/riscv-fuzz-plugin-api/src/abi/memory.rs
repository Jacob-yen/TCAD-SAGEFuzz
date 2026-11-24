use std::slice;

use super::{PluginBytes, PluginSlice};

/// Reclaim a `PluginBytes` allocation that was originally produced by
/// `PluginBytes::from_vec` on the same side of the FFI boundary.
///
/// # Safety
///
/// `bytes` must be either the null/empty sentinel or have come from
/// [`PluginBytes::from_vec`]; mixing allocators or double-freeing is UB.
pub unsafe extern "C" fn free_plugin_bytes(bytes: PluginBytes) {
    if bytes.ptr.is_null() {
        return;
    }

    unsafe {
        let _ = Vec::from_raw_parts(bytes.ptr, bytes.len, bytes.cap);
    }
}

pub fn copy_and_free_plugin_bytes(
    free_bytes: unsafe extern "C" fn(PluginBytes),
    bytes: PluginBytes,
) -> Vec<u8> {
    if bytes.ptr.is_null() {
        return Vec::new();
    }

    let owned = unsafe { slice::from_raw_parts(bytes.ptr, bytes.len).to_vec() };
    unsafe { free_bytes(bytes) };
    owned
}

pub(crate) unsafe fn plugin_slice_as_bytes<'a>(slice: PluginSlice) -> &'a [u8] {
    if slice.ptr.is_null() {
        &[]
    } else {
        unsafe { slice::from_raw_parts(slice.ptr, slice.len) }
    }
}
