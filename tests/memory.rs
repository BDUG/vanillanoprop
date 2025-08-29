use vanillanoprop::memory;

#[cfg(any(unix, windows))]
#[test]
fn peak_memory_is_positive() {
    assert!(memory::peak_memory_bytes() > 0);
}
