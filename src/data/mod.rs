pub mod dataloader;
pub mod text;
pub mod audio;
pub mod datasets;

pub use dataloader::{Cifar10, DataLoader, Dataset, Mnist};
pub use datasets::DatasetKind;
pub use text::TextDataset;
pub use audio::AudioDataset;

use mnist::MnistBuilder;

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
