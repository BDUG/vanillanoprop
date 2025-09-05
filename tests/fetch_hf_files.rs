use vanillanoprop::fetch_hf_files;

#[test]
fn fetches_all_optional_files() {
    let files = fetch_hf_files("hf-internal-testing/tiny-random-clip", None)
        .expect("failed to fetch hf files");
    assert!(files.config.exists());
    assert!(files.weights.exists());
    assert!(files.tokenizer_json.is_some());
    assert!(files.processor.is_some());
}
