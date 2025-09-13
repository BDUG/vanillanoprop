#[cfg(feature = "vlm")]
use image::{imageops::FilterType, DynamicImage, ImageBuffer, Luma};

#[cfg(feature = "vlm")]
#[test]
fn smolvlm_preprocess_pipeline() {
    let gradient: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_fn(32, 32, |x, _| Luma([x as u8 * 8]));
    let bytes = DynamicImage::ImageLuma8(gradient)
        .resize_exact(28, 28, FilterType::Triangle)
        .to_luma8()
        .into_raw();

    assert_eq!(bytes.len(), 28 * 28);
    assert!(bytes.iter().all(|&b| b <= u8::MAX));

    let mut histogram = [0u32; 256];
    for &b in &bytes {
        histogram[b as usize] += 1;
    }

    let indices = [
        1, 10, 19, 28, 37, 46, 55, 65, 74, 83, 92, 101, 110, 119, 129, 138, 147,
        156, 165, 174, 183, 193, 202, 211, 220, 229, 238, 247,
    ];
    let mut expected = [0u32; 256];
    for &i in &indices {
        expected[i] = 28;
    }

    assert_eq!(histogram, expected);
}
