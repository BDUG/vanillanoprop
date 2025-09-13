use std::io::{self, Write};
use std::time::Instant;

/// A very small progress bar implementation.
pub struct ProgressBar {
    total: u64,
    current: u64,
    start: Instant,
    message: String,
}

impl ProgressBar {
    /// Create a new progress bar for `total` steps.
    pub fn new(total: u64) -> Self {
        Self {
            total,
            current: 0,
            start: Instant::now(),
            message: String::new(),
        }
    }

    /// Set the current position of the progress bar.
    pub fn set_position(&mut self, pos: u64) {
        self.current = pos;
        self.draw();
    }

    /// Increment the progress bar by `delta` steps.
    pub fn inc(&mut self, delta: u64) {
        self.current += delta;
        self.draw();
    }

    /// Update the message displayed next to the progress bar.
    pub fn set_message(&mut self, msg: String) {
        self.message = msg;
        self.draw();
    }

    /// Finish the progress bar with a final message.
    pub fn finish_with_message(&mut self, msg: &str) {
        self.message = msg.to_string();
        self.current = self.total;
        self.draw();
        println!();
    }

    fn draw(&self) {
        let elapsed = self.start.elapsed().as_secs();
        print!("\r[{}/{}] {} ({}s)", self.current, self.total, self.message, elapsed);
        let _ = io::stdout().flush();
    }
}

