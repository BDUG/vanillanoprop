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
