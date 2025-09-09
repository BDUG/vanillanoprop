use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::Path;

use serde::{de, Deserialize, Deserializer};
use serde_json::Value;

/// Simple tokenizer capable of loading a HuggingFace `tokenizer.json` file
/// and performing basic encode/decode operations for BPE and WordPiece
/// style models.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    model_type: ModelType,
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    unk_id: Option<u32>,
    /// Ranks of merge pairs used for BPE models.
    bpe_ranks: HashMap<(String, String), usize>,
    /// Whether to lowercase input during encoding.
    lowercase: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelType {
    BPE,
    WordPiece,
}

impl Tokenizer {
    /// Build a tokenizer from a HuggingFace `tokenizer.json` file.
    pub fn from_json(path: &Path) -> Result<Self, Box<dyn Error>> {
        let mut data = fs::read(path)?;
        if data.starts_with(&[0xEF, 0xBB, 0xBF]) {
            data.drain(..3);
        }
        let json: TokenizerConfig = serde_json::from_slice(&data)?;

        let mut vocab = json.model.vocab;
        if let Some(extra) = &json.added_tokens {
            for t in extra {
                vocab.insert(t.content.clone(), t.id);
            }
        }

        let mut id_to_token = vec![String::new(); vocab.len()];
        for (tok, id) in &vocab {
            let idx = *id as usize;
            if idx >= id_to_token.len() {
                id_to_token.resize(idx + 1, String::new());
            }
            id_to_token[idx] = tok.clone();
        }

        let unk_id = json
            .added_tokens
            .as_ref()
            .and_then(|toks| toks.iter().find(|t| t.special && t.content.contains("unk")))
            .map(|t| t.id)
            .or_else(|| vocab.get("[UNK]").copied());

        let mut bpe_ranks = HashMap::new();
        if let Some(merges) = json.model.merges {
            for (i, merge) in merges.iter().enumerate() {
                let mut parts = merge.split_whitespace();
                if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                    bpe_ranks.insert((a.to_string(), b.to_string()), i);
                }
            }
        }

        let model_type = match json.model.model_type.as_str() {
            "BPE" => ModelType::BPE,
            "WordPiece" => ModelType::WordPiece,
            other => return Err(format!("unsupported tokenizer model type {other}").into()),
        };

        let lowercase = json
            .normalizer
            .as_ref()
            .and_then(|n| n.lowercase)
            .unwrap_or(false);

        Ok(Tokenizer {
            model_type,
            vocab,
            id_to_token,
            unk_id,
            bpe_ranks,
            lowercase,
        })
    }

    /// Encode a piece of text into token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut text = text.to_string();
        if self.lowercase {
            text = text.to_lowercase();
        }
        match self.model_type {
            ModelType::WordPiece => self.encode_wordpiece(&text),
            ModelType::BPE => self.encode_bpe_text(&text),
        }
    }

    /// Decode token ids back into text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.id_to_token.get(id as usize).cloned())
            .collect();
        match self.model_type {
            ModelType::WordPiece => {
                let mut out = String::new();
                for tok in tokens {
                    if tok.starts_with("##") {
                        out.push_str(&tok[2..]);
                    } else {
                        if !out.is_empty() {
                            out.push(' ');
                        }
                        out.push_str(&tok);
                    }
                }
                out
            }
            ModelType::BPE => {
                let mut out = String::new();
                for tok in tokens {
                    out.push_str(&tok.replace("Ġ", " "));
                }
                out.trim().to_string()
            }
        }
    }

    fn encode_bpe_text(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for token in text.split_whitespace() {
            for piece in self.bpe(token) {
                if let Some(&id) = self.vocab.get(&piece) {
                    ids.push(id);
                } else if let Some(unk) = self.unk_id {
                    ids.push(unk);
                }
            }
        }
        ids
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        let mut pairs = get_pairs(&word);
        while !pairs.is_empty() {
            let mut min_pair: Option<(String, String)> = None;
            let mut min_rank = usize::MAX;
            for pair in &pairs {
                if let Some(&rank) = self.bpe_ranks.get(pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        min_pair = Some(pair.clone());
                    }
                }
            }
            let pair = match min_pair {
                Some(p) => p,
                None => break,
            };
            let mut new_word = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && (word[i].clone(), word[i + 1].clone()) == pair {
                    new_word.push(format!("{}{}", word[i], word[i + 1]));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;
            pairs = get_pairs(&word);
        }
        word
    }

    fn encode_wordpiece(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for token in text.split_whitespace() {
            let mut start = 0;
            let chars: Vec<char> = token.chars().collect();
            while start < chars.len() {
                let mut end = chars.len();
                let mut cur_id = None;
                while start < end {
                    let slice: String = chars[start..end].iter().collect();
                    let candidate = if start == 0 {
                        slice.clone()
                    } else {
                        format!("##{}", slice)
                    };
                    if let Some(&id) = self.vocab.get(&candidate) {
                        cur_id = Some(id);
                        break;
                    }
                    end -= 1;
                }
                if let Some(id) = cur_id {
                    ids.push(id);
                    start = end;
                } else {
                    if let Some(unk) = self.unk_id {
                        ids.push(unk);
                    }
                    break;
                }
            }
        }
        ids
    }
}

fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
    let mut pairs = HashSet::new();
    if word.len() < 2 {
        return pairs;
    }
    let mut prev = &word[0];
    for ch in &word[1..] {
        pairs.insert((prev.clone(), ch.clone()));
        prev = ch;
    }
    pairs
}

#[derive(Deserialize)]
struct TokenizerConfig {
    model: Model,
    #[serde(default)]
    added_tokens: Option<Vec<AddedToken>>,
    normalizer: Option<Normalizer>,
}

#[derive(Deserialize)]
struct Normalizer {
    #[serde(default)]
    lowercase: Option<bool>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
struct Model {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    #[serde(default, deserialize_with = "deserialize_merges")]
    merges: Option<Vec<String>>,
}

fn deserialize_merges<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt = Option::<Vec<Value>>::deserialize(deserializer)?;
    let Some(values) = opt else {
        return Ok(None);
    };

    let mut merges = Vec::with_capacity(values.len());
    for value in values {
        match value {
            Value::String(s) => merges.push(s),
            Value::Array(arr) => {
                if arr.len() != 2 {
                    return Err(de::Error::custom("merge arrays must have two elements"));
                }
                let a = arr[0]
                    .as_str()
                    .ok_or_else(|| de::Error::custom("merge elements must be strings"))?;
                let b = arr[1]
                    .as_str()
                    .ok_or_else(|| de::Error::custom("merge elements must be strings"))?;
                merges.push(format!("{a} {b}"));
            }
            _ => return Err(de::Error::custom("merge must be string or array")),
        }
    }

    Ok(Some(merges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn wordpiece_roundtrip() {
        let json = r###"{
            "model": {
                "type": "WordPiece",
                "vocab": {"[UNK]":0,"hello":1,"world":2,"##!":3}
            },
            "normalizer": {"lowercase": true}
        }"###;
        let mut path = std::env::temp_dir();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("tok_{unique}.json"));
        File::create(&path)
            .unwrap()
            .write_all(json.as_bytes())
            .unwrap();
        let tok = Tokenizer::from_json(&path).unwrap();
        let ids = tok.encode("Hello world!");
        assert_eq!(ids, vec![1, 2, 3]);
        let dec = tok.decode(&ids);
        assert_eq!(dec, "hello world!");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn loads_array_merges() {
        let json = r###"{
            "model": {
                "type": "BPE",
                "vocab": {"a":0,"b":1,"ab":2},
                "merges": [["a","b"]]
            }
        }"###;
        let mut path = std::env::temp_dir();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("tok_{unique}.json"));
        File::create(&path)
            .unwrap()
            .write_all(json.as_bytes())
            .unwrap();
        let tok = Tokenizer::from_json(&path).unwrap();
        assert_eq!(tok.encode("ab"), vec![2]);
        let _ = fs::remove_file(path);
    }
}
