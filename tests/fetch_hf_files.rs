use vanillanoprop::fetch_hf_files;

#[test]
fn fetches_all_optional_files() {
    let files = fetch_hf_files("hf-internal-testing/tiny-random-clip", None, None)
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
    assert!(
        err.to_string()
            .contains("Invalid or expired Hugging Face token"),
        "unexpected error: {err}"
    );
}
