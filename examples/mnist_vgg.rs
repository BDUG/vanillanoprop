use vanillanoprop::data::{DataLoader, Mnist};
use vanillanoprop::math::{self, Matrix};
use vanillanoprop::models::VGG;

fn main() {
    // Use a small VGG-style network with three blocks of [1,1,2] conv layers.
    let mut vgg = VGG::new(&[1, 1, 2], 10);
    let lr = 0.01f32;

    for (i, batch) in DataLoader::<Mnist>::new(32, true, None).take(5).enumerate() {
        let mut loss_sum = 0.0f32;
        for (img, label) in batch {
            let (feat, logits) = vgg.forward(img);
            let logits_m = Matrix::from_vec(1, logits.len(), logits);
            let (loss, grad, _) =
                math::softmax_cross_entropy(&logits_m, &[*label as usize], 0);
            loss_sum += loss;

            // SGD update for the classification head
            let grad_logits = grad.data;
            let (fc, bias) = vgg.parameters_mut();
            let rows = fc.rows;
            let cols = fc.cols;
            let mut grad_matrix = vec![0.0f32; rows * cols];
            for (c, &g) in grad_logits.iter().enumerate() {
                for (r, &f) in feat.iter().enumerate() {
                    grad_matrix[r * cols + c] = f * g;
                }
            }
            for (w, &g) in fc.data.iter_mut().zip(grad_matrix.iter()) {
                *w -= lr * g;
            }
            for (b, &g) in bias.iter_mut().zip(grad_logits.iter()) {
                *b -= lr * g;
            }
        }
        println!("batch {i} loss {}", loss_sum / batch.len() as f32);
    }
}
