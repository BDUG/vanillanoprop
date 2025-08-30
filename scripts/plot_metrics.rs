use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

use serde::Deserialize;

#[derive(Deserialize)]
struct MetricRecord {
    epoch: usize,
    step: usize,
    loss: f32,
    f1: f32,
    lr: f32,
    kind: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "runs/example/metrics.jsonl".to_string());
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut losses = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let rec: MetricRecord = serde_json::from_str(&line)?;
        if rec.kind == "batch" {
            losses.push(rec.loss);
        }
    }
    if losses.is_empty() {
        return Ok(());
    }
    let max_loss = losses.iter().fold(f32::MIN, |a, &b| a.max(b));
    for (i, loss) in losses.iter().enumerate() {
        let bar = if max_loss > 0.0 {
            ((loss / max_loss) * 50.0) as usize
        } else {
            0
        };
        println!("{:5} | {}", i, "*".repeat(bar));
    }
    Ok(())
}
