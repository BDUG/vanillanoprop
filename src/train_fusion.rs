use crate::data::{AudioDataset, DataLoader, Mnist, TextDataset};
use crate::models::{Fusion, RNN, SimpleCNN, TransformerEncoder};
use crate::tensor::Tensor;
use crate::math::Matrix;

/// Example training loop demonstrating how to fuse outputs from multiple
/// encoders.  The function iterates over three data loaders in parallel and
/// combines the encoder representations using the [`Fusion`] layer.
pub fn run() {
    // Build data loaders for image, text and audio inputs.
    let image_loader = DataLoader::<Mnist>::new(1, true, None);
    let text_loader = DataLoader::<TextDataset>::new(1, true, None);
    let audio_loader = DataLoader::<AudioDataset>::new(1, true, None);

    // Instantiate simple encoders for each modality.
    let cnn = SimpleCNN::new(10);
    let mut rnn = RNN::new_lstm(1, 4, 2);
    let mut transformer = TransformerEncoder::new(1, 10, 8, 1, 16, 0.0);

    // Iterate over the loaders in lock-step.  Each iteration processes one
    // sample from each modality and fuses the resulting features.
    for (img_batch, (text_batch, audio_batch)) in image_loader.zip(text_loader.zip(audio_loader)) {
        let (img, _) = &img_batch[0];
        let (text, _) = &text_batch[0];
        let (_audio, _) = &audio_batch[0];

        let (feat, _) = cnn.forward(img);
        let cnn_t = Tensor::new(feat, vec![1, 28 * 28]);

        let text_vec: Vec<f32> = text.bytes().map(|b| b as f32).collect();
        let text_t = Tensor::new(text_vec.clone(), vec![text_vec.len(), 1]);
        let rnn_t = rnn.forward(&text_t);

        let audio_input = Matrix::from_vec(1, 10, vec![0.0; 10]);
        let transformer_t = transformer.forward(audio_input, None);

        let fused = Fusion::concat(&cnn_t, &rnn_t, &transformer_t);
        log::info!("fused representation has shape {:?}", fused.shape);
    }
}
