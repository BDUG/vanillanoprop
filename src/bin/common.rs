use std::env;

/// Parses common CLI arguments across training binaries.
///
/// Returns a tuple `(model, optimizer, moe, num_experts, positional_args)`.
/// - `model` defaults to "transformer" if not specified.
/// - `optimizer` defaults to "sgd" if not specified.
/// - `moe` is true if `--moe` flag is present.
/// - `num_experts` reads the value after `--num-experts`, defaulting to 1.
/// - `positional_args` contains remaining positional arguments in order.
pub fn parse_cli<I>(mut args: I) -> (String, String, bool, usize, Vec<String>)
where
    I: Iterator<Item = String>,
{
    let mut model = "transformer".to_string();
    let mut opt = "sgd".to_string();
    let mut moe = false;
    let mut num_experts = 1usize;
    let mut positional = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--moe" => moe = true,
            "--num-experts" => {
                if let Some(n) = args.next() {
                    num_experts = n.parse().unwrap_or(1);
                }
            }
            _ => positional.push(arg),
        }
    }

    if let Some(m) = positional.get(0) {
        model = m.clone();
    }
    if let Some(o) = positional.get(1) {
        opt = o.clone();
    }

    (model, opt, moe, num_experts, positional)
}

/// Convenience wrapper that parses arguments from the current process
/// (skipping the binary name).
pub fn parse_env() -> (String, String, bool, usize, Vec<String>) {
    let args = env::args().skip(1);
    parse_cli(args)
}
