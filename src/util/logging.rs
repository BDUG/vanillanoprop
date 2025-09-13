use crate::info;

/// Format a message reporting the total number of matrix operations.
pub fn format_total_ops(count: usize) -> String {
    format!("Total matrix ops: {}", count)
}

/// Log the total number of matrix operations at info level.
pub fn log_total_ops(count: usize) {
    info!("{}", format_total_ops(count));
}

/// Format a checkpoint saved message.
pub fn format_checkpoint_saved(epoch: usize, f1: f32) -> String {
    format!(
        "Checkpoint saved at epoch {}: avg F1 improved to {:.4}",
        epoch, f1
    )
}

/// Log that a checkpoint was saved at info level.
pub fn log_checkpoint_saved(epoch: usize, f1: f32) {
    info!("{}", format_checkpoint_saved(epoch, f1));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_total_ops() {
        assert_eq!(format_total_ops(42), "Total matrix ops: 42");
    }

    #[test]
    fn test_format_checkpoint_saved() {
        assert_eq!(
            format_checkpoint_saved(3, 0.12345),
            "Checkpoint saved at epoch 3: avg F1 improved to 0.1235"
        );
    }
}
