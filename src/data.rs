use crate::math::Matrix;
use mnist::MnistBuilder;

/// Apply a 3x3 convolution kernel with zero padding to an image.
///
/// The image is provided as a flat slice in row-major order. The returned
/// vector has the same size as the input and contains the filtered pixels.
fn convolve3x3(img: &[u8], width: usize, height: usize, kernel: [[f32; 3]; 3]) -> Vec<u8> {
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for ky in 0..3 {
                for kx in 0..3 {
                    let ix = x as isize + kx as isize - 1;
                    let iy = y as isize + ky as isize - 1;
                    if ix >= 0 && ix < width as isize && iy >= 0 && iy < height as isize {
                        let idx = iy as usize * width + ix as usize;
                        acc += img[idx] as f32 * kernel[ky][kx];
                    }
                }
            }
            out[y * width + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

pub const START: &str = "<start>";
pub const END: &str = "<end>";

pub struct Vocab {
    pub stoi: std::collections::HashMap<String, usize>,
    pub itos: Vec<String>,
}

impl Vocab {
    /// Build a vocabulary for MNIST pixel values plus start/end tokens.
    pub fn build() -> Self {
        let mut itos: Vec<String> = (0..256).map(|i| i.to_string()).collect();
        itos.push(START.to_string());
        itos.push(END.to_string());
        let mut stoi = std::collections::HashMap::new();
        for (i, w) in itos.iter().enumerate() {
            stoi.insert(w.clone(), i);
        }
        Self { stoi, itos }
    }
}

/// Load a small portion of the MNIST dataset as (image, label) pairs.
pub fn load_pairs() -> Vec<(Vec<usize>, Vec<usize>)> {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(10)
        .finalize();
    let end_id = *Vocab::build().stoi.get(END).unwrap();
    mnist
        .trn_img
        .chunks(28 * 28)
        .zip(mnist.trn_lbl.iter())
        .map(|(img, &lbl)| {
            // Simple 3x3 mean blur to smooth the input image
            let kernel = [[1.0 / 9.0; 3]; 3];
            let processed = convolve3x3(img, 28, 28, kernel);
            let src: Vec<usize> = processed.iter().map(|&p| p as usize).collect();
            let tgt = vec![lbl as usize, end_id];
            (src, tgt)
        })
        .collect()
}

/// Download the MNIST dataset into the local `data/` directory.
///
/// This uses the `mnist` crate's built-in downloader which fetches the
/// required archive files and extracts them so that a file like
/// `data/train-images-idx3-ubyte` is available.
///
/// The function does not return the dataset; it simply ensures that the
/// files exist on disk for subsequent training runs.
pub fn download_mnist() {
    // `finalize` triggers the download when `download_and_extract` is
    // enabled. We drop the returned data since we only care about the
    // side effect of fetching the files.
    let _ = MnistBuilder::new().download_and_extract().finalize();
}

pub fn to_matrix(seq: &[usize], vocab_size: usize) -> Matrix {
    let mut m = Matrix::zeros(seq.len(), vocab_size);
    for (i, &tok) in seq.iter().enumerate() {
        m.set(i, tok, 1.0);
    }
    m
}
