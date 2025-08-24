use crate::math::Matrix;
use mnist::MnistBuilder;

static DATA: &[(&str, &str)] = &[
    ("hallo welt", "hello world"),
    ("ich liebe rust", "i love rust"),
    ("das ist gut", "that is good"),
    ("wie geht es dir", "how are you"),
];

pub const START: &str = "<start>";
pub const END: &str = "<end>";

pub struct Vocab {
    pub stoi: std::collections::HashMap<String, usize>,
    pub itos: Vec<String>,
}

impl Vocab {
    pub fn build() -> Self {
        let mut set = std::collections::HashSet::new();
        set.insert(START.to_string());
        set.insert(END.to_string());
        for (s, t) in DATA.iter() {
            for w in s.split_whitespace() {
                set.insert(w.to_string());
            }
            for w in t.split_whitespace() {
                set.insert(w.to_string());
            }
        }
        let mut itos: Vec<String> = set.into_iter().collect();
        itos.sort();
        let mut stoi = std::collections::HashMap::new();
        for (i, w) in itos.iter().enumerate() {
            stoi.insert(w.clone(), i);
        }
        Self { stoi, itos }
    }

    pub fn build_mnist() -> Self {
        let mut itos: Vec<String> = (0..256).map(|i| i.to_string()).collect();
        itos.push(START.to_string());
        itos.push(END.to_string());
        let mut stoi = std::collections::HashMap::new();
        for (i, w) in itos.iter().enumerate() {
            stoi.insert(w.clone(), i);
        }
        Self { stoi, itos }
    }

    pub fn encode(&self, s: &str) -> Vec<usize> {
        s.split_whitespace()
            .map(|w| *self.stoi.get(w).unwrap())
            .collect()
    }

    pub fn decode(&self, v: &[usize]) -> String {
        v.iter()
            .map(|i| self.itos[*i].clone())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub fn load_pairs() -> Vec<(Vec<usize>, Vec<usize>)> {
    let vocab = Vocab::build();
    DATA.iter()
        .map(|(a, b)| (vocab.encode(a), vocab.encode(b)))
        .collect()
}

pub fn load_mnist_pairs() -> Vec<(Vec<usize>, Vec<usize>)> {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(10)
        .finalize();
    let end_id = *Vocab::build_mnist().stoi.get(END).unwrap();
    mnist
        .trn_img
        .chunks(28 * 28)
        .zip(mnist.trn_lbl.iter())
        .map(|(img, &lbl)| {
            let src: Vec<usize> = img.iter().map(|&p| p as usize).collect();
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
    let _ = MnistBuilder::new()
        .download_and_extract()
        .finalize();
}

pub fn to_matrix(seq: &[usize], vocab_size: usize) -> Matrix {
    let mut m = Matrix::zeros(seq.len(), vocab_size);
    for (i, &tok) in seq.iter().enumerate() {
        m.set(i, tok, 1.0);
    }
    m
}
