use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use csv::Writer;
use serde::Serialize;

pub struct Logger {
    json: File,
    csv: Writer<File>,
}

#[derive(Serialize)]
pub struct MetricRecord {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
    pub f1: f32,
    pub lr: f32,
    pub kind: &'static str,
}

impl Logger {
    pub fn new(log_dir: Option<String>, experiment: Option<String>) -> std::io::Result<Self> {
        let base = log_dir.unwrap_or_else(|| "runs".to_string());
        let exp = experiment.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_else(|_| Duration::from_secs(0))
                .as_secs()
                .to_string()
        });
        let dir = PathBuf::from(base).join(exp);
        std::fs::create_dir_all(&dir)?;
        let json_path = dir.join("metrics.jsonl");
        let csv_path = dir.join("metrics.csv");
        let json = OpenOptions::new()
            .create(true)
            .append(true)
            .open(json_path)?;
        let csv_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(csv_path)?;
        let csv = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(csv_file);
        Ok(Logger { json, csv })
    }

    pub fn log<T: Serialize>(&mut self, metrics: &T) {
        if let Ok(line) = serde_json::to_string(metrics) {
            let _ = writeln!(self.json, "{}", line);
        }
        let _ = self.csv.serialize(metrics);
    }
}

/// Signals returned by callbacks to control training flow.
pub enum CallbackSignal {
    /// Continue training as normal.
    Continue,
    /// Stop training early.
    Stop,
}

/// Trait for hooking into various stages of the training loop.
pub trait Callback {
    /// Called once before training starts.
    fn on_train_begin(&mut self) {}

    /// Called at the beginning of each epoch.
    fn on_epoch_begin(&mut self, _epoch: usize) {}

    /// Called after each batch. Returning `Stop` will end training.
    fn on_batch_end(&mut self, _metrics: &MetricRecord) -> CallbackSignal {
        CallbackSignal::Continue
    }

    /// Called after each epoch. Returning `Stop` will end training.
    fn on_epoch_end(&mut self, _metrics: &MetricRecord) -> CallbackSignal {
        CallbackSignal::Continue
    }

    /// Called once after training ends.
    fn on_train_end(&mut self) {}
}

/// Stop training when a monitored metric fails to improve.
pub struct EarlyStopping {
    patience: usize,
    best: Option<f32>,
    wait: usize,
}

impl EarlyStopping {
    /// Create a new [`EarlyStopping`] callback monitoring F1 score.
    pub fn new(patience: usize) -> Self {
        Self {
            patience,
            best: None,
            wait: 0,
        }
    }
}

impl Callback for EarlyStopping {
    fn on_epoch_end(&mut self, metrics: &MetricRecord) -> CallbackSignal {
        let current = metrics.f1;
        if self.best.map_or(true, |b| current > b) {
            self.best = Some(current);
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                return CallbackSignal::Stop;
            }
        }
        CallbackSignal::Continue
    }
}

/// Save model snapshots whenever the monitored metric improves.
pub struct ModelSnapshot {
    save_fn: Box<dyn FnMut(&MetricRecord)>,
    best: Option<f32>,
}

impl ModelSnapshot {
    /// Create a new [`ModelSnapshot`] with a custom save function.
    pub fn new<F>(save: F) -> Self
    where
        F: FnMut(&MetricRecord) + 'static,
    {
        Self {
            save_fn: Box::new(save),
            best: None,
        }
    }
}

impl Callback for ModelSnapshot {
    fn on_epoch_end(&mut self, metrics: &MetricRecord) -> CallbackSignal {
        let current = metrics.f1;
        if self.best.map_or(true, |b| current > b) {
            self.best = Some(current);
            (self.save_fn)(metrics);
        }
        CallbackSignal::Continue
    }
}
