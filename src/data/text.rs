use super::dataloader::Dataset;

/// Simple in-memory text dataset.
///
/// Each sample is represented as a tuple of the text string and an
/// associated label.  This is a minimal example primarily intended for
/// integration tests and examples.
pub struct TextDataset;

impl Dataset for TextDataset {
    type Item = (String, usize);

    fn load() -> Vec<Self::Item> {
        vec![
            ("hello world".to_string(), 0),
            ("foo bar".to_string(), 1),
        ]
    }
}
