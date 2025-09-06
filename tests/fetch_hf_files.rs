use vanillanoprop::{config::Config, fetch_hf_files};

#[test]
fn fetches_all_optional_files() {
    let cfg = Config::from_path("backprop_config.toml").unwrap_or_default();
    let token = cfg
        .hf_token
        .filter(|t| !t.is_empty())
        .or_else(|| std::env::var("HF_TOKEN").ok());
    let token = match token {
        Some(t) => t,
        None => {
            eprintln!("Skipping test: no Hugging Face token provided");
            return;
        }
    };
    let files = fetch_hf_files("hf-internal-testing/tiny-random-clip", None, Some(&token))
        .expect("failed to fetch hf files");
    assert!(files.config.exists());
    assert!(files.weights.exists());
    assert!(files.tokenizer_json.is_some());
    assert!(files.processor.is_some());
}

#[test]
fn invalid_token_returns_error() {
    let res = fetch_hf_files(
        "hf-internal-testing/tiny-random-clip",
        None,
        Some("invalid_token"),
    );
    assert!(res.is_err());
    let err = match res {
        Ok(_) => panic!("expected error"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("Invalid or expired Hugging Face token")
            || msg.contains("Proxy failed to connect"),
        "unexpected error: {msg}"
    );
}
