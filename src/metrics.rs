pub fn f1_score(pred: &[usize], tgt: &[usize]) -> f32 {
    let len = pred.len().min(tgt.len());
    let mut tp = 0f32;
    for i in 0..len {
        if pred[i] == tgt[i] {
            tp += 1.0;
        }
    }
    let fp = (pred.len() as f32) - tp;
    let fn_ = (tgt.len() as f32) - tp;
    if tp == 0.0 {
        0.0
    } else {
        2.0 * tp / (2.0 * tp + fp + fn_)
    }
}
