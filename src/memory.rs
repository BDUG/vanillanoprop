//! Utilities for tracking process memory usage.
//!
//! Supported on Linux, macOS and Windows. Returns `0` on other platforms.

#[cfg(unix)]
use libc::{getrusage, rusage, RUSAGE_SELF};

#[cfg(windows)]
use windows_sys::Win32::System::{
    ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
    Threading::GetCurrentProcess,
};

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

/// Returns the peak working set size of the current process in bytes on
/// Windows.
#[cfg(windows)]
pub fn peak_memory_bytes() -> u64 {
    unsafe {
        let handle = GetCurrentProcess();
        let mut counters: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        if GetProcessMemoryInfo(
            handle,
            &mut counters as *mut _,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ) != 0
        {
            counters.PeakWorkingSetSize as u64
        } else {
            0
        }
    }
}

/// Stub implementation for unsupported targets.
#[cfg(not(any(unix, windows)))]
pub fn peak_memory_bytes() -> u64 {
    0
}
