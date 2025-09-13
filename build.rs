use std::env;
use std::fs;

fn main() {
    if env::var("CARGO_FEATURE_KAI").is_err() {
        return;
    }

    let mut build = cc::Build::new();
    for entry in fs::read_dir("c_src").expect("failed to read c_src directory") {
        let path = entry.expect("invalid path").path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("c") {
            build.file(path);
        }
    }

    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("armv7") {
        build.flag("-mcpu=cortex-a7");
    } else if target.contains("aarch64") || target.contains("arm64") {
        build.flag("-march=armv8-a");
    } else if target.starts_with("arm") {
        build.flag("-march=armv7-a");
    }

    build.compile("c_src");
}
