use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Logging levels for the simple logger.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error = 1,
    Warn = 2,
    Info = 3,
}

static LOG_LEVEL: AtomicUsize = AtomicUsize::new(LogLevel::Info as usize);

/// Set the global log level.
pub fn set_log_level(level: LogLevel) {
    LOG_LEVEL.store(level as usize, Ordering::Relaxed);
}

/// Check if a message at `level` should be logged.
pub fn enabled(level: LogLevel) -> bool {
    level as usize <= LOG_LEVEL.load(Ordering::Relaxed)
}

pub fn timestamp() -> String {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    format!("{}.{:03}", now.as_secs(), now.subsec_millis())
}

#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {{
        if $crate::util::simple_logger::enabled($crate::util::simple_logger::LogLevel::Info) {
            let ts = $crate::util::simple_logger::timestamp();
            println!("[INFO {ts}] {}", format!($($arg)*));
        }
    }};
}

#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {{
        if $crate::util::simple_logger::enabled($crate::util::simple_logger::LogLevel::Warn) {
            let ts = $crate::util::simple_logger::timestamp();
            eprintln!("[WARN {ts}] {}", format!($($arg)*));
        }
    }};
}

#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {{
        if $crate::util::simple_logger::enabled($crate::util::simple_logger::LogLevel::Error) {
            let ts = $crate::util::simple_logger::timestamp();
            eprintln!("[ERROR {ts}] {}", format!($($arg)*));
        }
    }};
}

