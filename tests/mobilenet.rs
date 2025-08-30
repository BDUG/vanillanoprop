use vanillanoprop::models::MobileNet;

#[test]
fn mobilenet_forward_and_params() {
    let mut net = MobileNet::new(10);
    let img = vec![0u8; 28 * 28];
    let (feat, logits) = net.forward(&img);
    assert_eq!(feat.len(), 32);
    assert_eq!(logits.len(), 10);
    assert_eq!(net.parameter_count(), 1258);
}
