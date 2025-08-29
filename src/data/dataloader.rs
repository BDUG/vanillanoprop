use cifar_10_loader::CifarDataset;
use mnist::MnistBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Trait describing a dataset that can be loaded entirely into memory.
pub trait Dataset {
    /// Type representing a single sample from the dataset.
    type Item: Clone;

    /// Load all samples for the dataset.
    fn load() -> Vec<Self::Item>;
}

/// Generic data loader supporting batching, optional shuffling and
/// preprocessing through user provided transforms.
pub struct DataLoader<D: Dataset> {
    data: Vec<D::Item>,
    batch_size: usize,
    index: usize,
}

impl<D: Dataset> DataLoader<D> {
    /// Create a new loader for `D`.
    ///
    /// `batch_size` controls how many samples are returned for each iteration.
    /// When `shuffle` is true the dataset is randomly permuted before
    /// iteration.  `transform` is an optional hook that can be used for
    /// preprocessing or data augmentation.
    pub fn new(
        batch_size: usize,
        shuffle: bool,
        mut transform: Option<Box<dyn FnMut(&mut D::Item)>>,
    ) -> Self {
        let mut data = D::load();
        if let Some(f) = transform.as_mut() {
            for sample in &mut data {
                f(sample);
            }
        }
        if shuffle {
            data.shuffle(&mut thread_rng());
        }
        Self {
            data,
            batch_size,
            index: 0,
        }
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = Vec<D::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }
        let end = (self.index + self.batch_size).min(self.data.len());
        let batch = self.data[self.index..end].to_vec();
        self.index = end;
        Some(batch)
    }
}

/// Loader for the full MNIST dataset.
pub struct Mnist;

impl Dataset for Mnist {
    type Item = (Vec<u8>, usize);

    fn load() -> Vec<Self::Item> {
        let mnist = MnistBuilder::new()
            .label_format_digit()
            .download_and_extract()
            .finalize();
        mnist
            .trn_img
            .chunks(28 * 28)
            .zip(mnist.trn_lbl.iter())
            .map(|(img, &lbl)| (img.to_vec(), lbl as usize))
            .collect()
    }
}

/// Loader for the CIFAR-10 dataset.
pub struct Cifar10;

impl Dataset for Cifar10 {
    type Item = (Vec<u8>, usize);

    fn load() -> Vec<Self::Item> {
        let path = "data/cifar-10-batches-bin";
        let cifar = CifarDataset::new(path).expect("failed to load CIFAR-10 dataset");
        cifar
            .train_dataset
            .into_iter()
            .map(|img| (img.image.to_rgb().into_raw(), img.label as usize))
            .collect()
    }
}
