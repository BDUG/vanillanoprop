use crate::math::Matrix;

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

pub fn to_matrix(seq: &[usize], vocab_size: usize) -> Matrix {
    let mut m = Matrix::zeros(seq.len(), vocab_size);
    for (i, &tok) in seq.iter().enumerate() {
        m.set(i, tok, 1.0);
    }
    m
}
