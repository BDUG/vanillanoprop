use vanillanoprop::fine_tune::{FineTune, FreezeSpec, LayerKind};
use vanillanoprop::layers::{Conv2d, LinearT};

#[test]
fn freeze_linear_layer() {
    let mut lin = LinearT::new(2, 2);
    let params = vec![(LayerKind::Linear, &mut lin)];
    let ft = FineTune::new(vec![FreezeSpec {
        kind: LayerKind::Linear,
        idx: 0,
    }]);
    let filtered = ft.filter(params);
    assert!(filtered.is_empty());
}

#[test]
fn freeze_conv_layer() {
    let mut conv = Conv2d::new(1, 1, 1, 1, 0);
    let params: Vec<(LayerKind, &mut LinearT)> = conv
        .parameters()
        .into_iter()
        .map(|p| (LayerKind::Conv, p))
        .collect();
    let ft = FineTune::new(vec![FreezeSpec {
        kind: LayerKind::Conv,
        idx: 0,
    }]);
    let filtered = ft.filter(params);
    assert!(filtered.is_empty());
}

#[test]
fn mixed_layers_freeze_conv() {
    let mut conv = Conv2d::new(1, 1, 1, 1, 0);
    let mut lin = LinearT::new(1, 1);
    let mut params: Vec<(LayerKind, &mut LinearT)> = conv
        .parameters()
        .into_iter()
        .map(|p| (LayerKind::Conv, p))
        .collect();
    params.push((LayerKind::Linear, &mut lin));
    let ft = FineTune::new(vec![FreezeSpec {
        kind: LayerKind::Conv,
        idx: 0,
    }]);
    let filtered = ft.filter(params);
    assert_eq!(filtered.len(), 1);
}
