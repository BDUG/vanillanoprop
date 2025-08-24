//! Utilities for tracking process memory usage.

#[cfg(unix)]
use libc::{getrusage, rusage, RUSAGE_SELF};

/// Returns the peak resident set size (max memory usage) of the current process
/// in bytes.
#[cfg(target_os = "macos")]
pub fn peak_memory_bytes() -> u64 {
    unsafe {
        let mut usage: rusage = std::mem::zeroed();
        if getrusage(RUSAGE_SELF, &mut usage) == 0 {
            usage.ru_maxrss as u64
        } else {
            0
        }
    }
}

/// Returns the peak resident set size (max memory usage) of the current process
/// in bytes. On Linux, `ru_maxrss` is reported in kilobytes, so we convert to
/// bytes before returning.
#[cfg(all(unix, not(target_os = "macos")))]
pub fn peak_memory_bytes() -> u64 {
    unsafe {
        let mut usage: rusage = std::mem::zeroed();
        if getrusage(RUSAGE_SELF, &mut usage) == 0 {
            usage.ru_maxrss as u64 * 1024
        } else {
            0
        }
    }
}

/// Stub implementation for unsupported targets.
#[cfg(not(unix))]
pub fn peak_memory_bytes() -> u64 {
    0
}

