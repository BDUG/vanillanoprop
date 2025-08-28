use cifar_10_loader::CifarDataset;
use mnist::MnistBuilder;

/// Trait for loading and batching datasets with optional preprocessing.
pub trait Dataset {
    /// Type representing a single sample from the dataset.
    type Item: Clone;

    /// Load the entire dataset into memory.
    fn load() -> Vec<Self::Item>;

    /// Optional preprocessing applied to each sample. Default is a no-op.
    fn preprocess(_sample: &mut Self::Item) {}

    /// Load the dataset and group samples into mini-batches of `batch_size`.
    fn batch(batch_size: usize) -> Vec<Vec<Self::Item>> {
        let mut data = Self::load();
        for sample in &mut data {
            Self::preprocess(sample);
        }
        data.chunks(batch_size).map(|c| c.to_vec()).collect()
    }
}

/// Loader for the full MNIST dataset.
pub struct Mnist;

impl Dataset for Mnist {
    type Item = (Vec<u8>, u8);

    fn load() -> Vec<Self::Item> {
        let mnist = MnistBuilder::new()
            .label_format_digit()
            .download_and_extract()
            .finalize();
        mnist
            .trn_img
            .chunks(28 * 28)
            .zip(mnist.trn_lbl.iter())
            .map(|(img, &lbl)| (img.to_vec(), lbl))
            .collect()
    }
}

/// Loader for the CIFAR-10 dataset.
pub struct Cifar10;

impl Dataset for Cifar10 {
    type Item = (Vec<u8>, u8);

    fn load() -> Vec<Self::Item> {
        let path = "data/cifar-10-batches-bin";
        let cifar = CifarDataset::new(path).expect("failed to load CIFAR-10 dataset");
        cifar
            .train_dataset
            .into_iter()
            .map(|img| (img.image.to_rgb().into_raw(), img.label))
            .collect()
    }
}

/// Apply a 3x3 convolution kernel with zero padding to an image.
///
/// The image is provided as a flat slice in row-major order. A caller supplied
/// buffer holds the 1-pixel padded copy of the image so the function can read
/// neighbouring pixels without bounds checks. The returned vector has the same
/// size as the input and contains the filtered pixels.
fn convolve3x3(
    img: &[u8],
    width: usize,
    height: usize,
    kernel: [[f32; 3]; 3],
    padded: &mut [u8],
) -> Vec<u8> {
    let padded_width = width + 2;

    // Fill buffer with zeros and copy image into the centred region
    padded.fill(0);
    for y in 0..height {
        let src = &img[y * width..(y + 1) * width];
        let dst_offset = (y + 1) * padded_width + 1;
        padded[dst_offset..dst_offset + width].copy_from_slice(src);
    }

    let mut out = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for ky in 0..3 {
                for kx in 0..3 {
                    let idx = (y + ky) * padded_width + (x + kx);
                    acc += padded[idx] as f32 * kernel[ky][kx];
                }
            }
            out[y * width + x] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Load a small portion of the MNIST dataset as `(pixels, label)` pairs.
pub fn load_pairs() -> Vec<(Vec<u8>, usize)> {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(10)
        .finalize();
    let kernel = [[1.0 / 9.0; 3]; 3];
    let mut padded = vec![0u8; (28 + 2) * (28 + 2)];
    mnist
        .trn_img
        .chunks(28 * 28)
        .zip(mnist.trn_lbl.iter())
        .map(|(img, &lbl)| {
            let pixels = convolve3x3(img, 28, 28, kernel, &mut padded);
            (pixels, lbl as usize)
        })
        .collect()
}

/// Return the dataset grouped into mini-batches of the given size.
///
/// Batching allows training code to accumulate gradients across several
/// samples before performing an optimisation step.  The final batch may be
/// smaller than `batch_size` if the total number of samples is not divisible
/// by it.
pub fn load_batches(batch_size: usize) -> Vec<Vec<(Vec<u8>, usize)>> {
    let pairs = load_pairs();
    pairs.chunks(batch_size).map(|c| c.to_vec()).collect()
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
