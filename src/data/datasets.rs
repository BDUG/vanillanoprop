use std::str::FromStr;

/// Available datasets supported by the crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatasetKind {
    /// MNIST handwritten digits dataset.
    Mnist,
    /// CIFAR-10 image dataset.
    Cifar10,
}

impl DatasetKind {
    /// Parse a dataset name into a `DatasetKind`.
    pub fn from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "mnist" => Some(DatasetKind::Mnist),
            "cifar10" | "cifar-10" => Some(DatasetKind::Cifar10),
            _ => None,
        }
    }
}

impl FromStr for DatasetKind {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        DatasetKind::from_str(s).ok_or(())
    }
}
