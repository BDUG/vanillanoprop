use super::{ResNet, TransformerEncoder};
use crate::layers::LinearT;
use crate::math::Matrix;
use crate::tensor::Tensor;

/// A minimal vision-language model that mirrors SmolVLM-Instruct.
/// It encodes an input image and a sequence of text tokens before
/// fusing the modalities with a linear projection.
pub struct SmolVLM {
    vision: ResNet,
    text: TransformerEncoder,
    fusion: LinearT,
    vocab_size: usize,
    text_dim: usize,
}

impl SmolVLM {
    /// Create a new [`SmolVLM`].
    ///
    /// `vocab_size` is the size of the text vocabulary, `vision_dim`
    /// controls the dimensionality of the image features and `text_dim`
    /// is the width of the text encoder.
    pub fn new(vocab_size: usize, vision_dim: usize, text_dim: usize) -> Self {
        let vision = ResNet::new(1, vision_dim, 2);
        let text = TransformerEncoder::new(2, vocab_size, text_dim, 2, text_dim * 4, 0.0);
        let fusion = LinearT::new(vision_dim + text_dim, text_dim);
        Self {
            vision,
            text,
            fusion,
            vocab_size,
            text_dim,
        }
    }

    fn to_one_hot(&self, tokens: &[usize]) -> Matrix {
        let mut m = Matrix::zeros(tokens.len(), self.vocab_size);
        for (t, &idx) in tokens.iter().enumerate() {
            if idx < self.vocab_size {
                m.set(t, idx, 1.0);
            }
        }
        m
    }

    /// Encode the image and text tokens and return a fused representation.
    pub fn forward(&mut self, image: &[u8], tokens: &[usize]) -> Tensor {
        let (vfeat, _) = self.vision.forward(image);
        let one_hot = self.to_one_hot(tokens);
        let text_t = self.text.forward(one_hot, None);
        let mut text_feat = vec![0.0f32; self.text_dim];
        text_feat.copy_from_slice(&text_t.data[0..self.text_dim]);
        let mut fused = Vec::with_capacity(vfeat.len() + text_feat.len());
        fused.extend_from_slice(&vfeat);
        fused.extend_from_slice(&text_feat);
        let fused_t = Tensor::new(fused, vec![1, vfeat.len() + text_feat.len()]);
        self.fusion.forward(&fused_t)
    }

    /// Mutable accessors used by weight loading utilities.
    pub fn vision_mut(&mut self) -> &mut ResNet {
        &mut self.vision
    }

    /// Mutable access to the text encoder.
    pub fn text_mut(&mut self) -> &mut TransformerEncoder {
        &mut self.text
    }

    /// Mutable access to the fusion layer.
    pub fn fusion_mut(&mut self) -> &mut LinearT {
        &mut self.fusion
    }

    /// Dimension of the text representation.
    pub fn text_dim(&self) -> usize {
        self.text_dim
    }

    /// Size of the vocabulary used by the text encoder.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
