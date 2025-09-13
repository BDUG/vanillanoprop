use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};

use native_tls::TlsConnector;

use crate::config::Config;

/// Local paths to important files fetched from the Hugging Face Hub.
pub struct HfFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub tokenizer_json: Option<PathBuf>,
    pub processor: Option<PathBuf>,
}

struct HttpResponse {
    status: u16,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

fn http_get(
    host: &str,
    path: &str,
    token: Option<&str>,
    redirects: u8,
) -> Result<HttpResponse, Box<dyn Error>> {
    if redirects == 0 {
        return Err("too many redirects".into());
    }

    let stream = TcpStream::connect((host, 443))?;
    let connector = TlsConnector::new()?;
    let mut stream = connector.connect(host, stream)?;

    let mut request = format!(
        "GET {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: vanillanoprop\r\nAccept: */*\r\nConnection: close\r\n",
        path, host
    );
    if let Some(t) = token {
        request.push_str(&format!("Authorization: Bearer {}\r\n", t));
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes())?;

    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("invalid HTTP response")?;
    let header_bytes = &buf[..split];
    let mut body = buf[split + 4..].to_vec();
    let header_str = String::from_utf8_lossy(header_bytes);
    let mut lines = header_str.lines();
    let status_line = lines.next().ok_or("missing status line")?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or("invalid status line")?
        .parse::<u16>()?;

    let mut headers = HashMap::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_string(), v.trim().to_string());
        }
    }

    if headers
        .get("Transfer-Encoding")
        .map(|v| v.eq_ignore_ascii_case("chunked"))
        == Some(true)
    {
        let mut decoded = Vec::new();
        let mut rest = body.as_slice();
        loop {
            let idx = rest
                .windows(2)
                .position(|w| w == b"\r\n")
                .ok_or("invalid chunk")?;
            let len_str = std::str::from_utf8(&rest[..idx])?;
            let len = usize::from_str_radix(len_str.trim(), 16)?;
            rest = &rest[idx + 2..];
            if len == 0 {
                break;
            }
            if rest.len() < len + 2 {
                return Err("truncated chunk".into());
            }
            decoded.extend_from_slice(&rest[..len]);
            rest = &rest[len + 2..];
        }
        body = decoded;
    }

    if matches!(status, 301 | 302 | 303 | 307 | 308) {
        if let Some(loc) = headers.get("Location") {
            let loc = loc.trim();
            if let Some(rest) = loc.strip_prefix("https://") {
                let pos = rest.find('/').unwrap_or(rest.len());
                let new_host = &rest[..pos];
                let new_path = &rest[pos..];
                return http_get(new_host, new_path, token, redirects - 1);
            }
        }
    }

    Ok(HttpResponse {
        status,
        headers,
        body,
    })
}

fn download(
    model_dir: &Path,
    model_id: &str,
    filename: &str,
    token: Option<&str>,
    optional: bool,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let path = model_dir.join(filename);
    if path.exists() {
        return Ok(Some(path));
    }
    let url_path = format!("/{}/resolve/main/{}", model_id, filename);
    let resp = http_get("huggingface.co", &url_path, token, 8)?;
    match resp.status {
        200 => {
            fs::create_dir_all(model_dir)?;
            fs::write(&path, resp.body)?;
            Ok(Some(path))
        }
        401 => Err("Invalid or expired Hugging Face token".into()),
        404 if optional => Ok(None),
        status => Err(format!("Failed to download {}: HTTP {}", filename, status).into()),
    }
}

/// Download `config.json` and weights for `model_id` from the Hugging Face Hub.
///
/// If `cache_dir` is provided, the files will be cached under this directory,
/// otherwise a temporary directory is used.
pub fn fetch_hf_files(
    model_id: &str,
    cache_dir: Option<&Path>,
    token: Option<&str>,
) -> Result<HfFiles, Box<dyn Error>> {
    let base = cache_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("hf-cache"));
    let model_dir = base.join(model_id);

    let config = download(&model_dir, model_id, "config.json", token, false)?
        .expect("config missing");
    let weights = match download(&model_dir, model_id, "model.safetensors", token, true)? {
        Some(p) => p,
        None => download(&model_dir, model_id, "pytorch_model.bin", token, false)?
            .expect("weights missing"),
    };
    let tokenizer = download(&model_dir, model_id, "tokenizer.model", token, true)?;
    let tokenizer_json = download(&model_dir, model_id, "tokenizer.json", token, true)?;
    let processor = match download(
        &model_dir,
        model_id,
        "preprocessor_config.json",
        token,
        true,
    )? {
        Some(p) => Some(p),
        None => download(&model_dir, model_id, "image_processor.json", token, true)?,
    };

    Ok(HfFiles {
        config,
        weights,
        tokenizer,
        tokenizer_json,
        processor,
    })
}

/// Convenience wrapper to fetch files using a [`Config`] for authentication.
///
/// Uses the `hf_token` field from the provided configuration if present.
pub fn fetch_hf_files_with_cfg(
    model_id: &str,
    cfg: &Config,
) -> Result<HfFiles, Box<dyn Error>> {
    fetch_hf_files(model_id, None, cfg.hf_token.as_deref())
}

