use std::env;
use vanillanoprop::optim::lr_scheduler::LrScheduleConfig;

/// Parses common CLI arguments across training binaries.
///
/// Returns a tuple `(model, optimizer, moe, num_experts, lr_schedule, resume, save_every, checkpoint_dir, positional_args)`.
/// - `model` defaults to "transformer" if not specified.
/// - `optimizer` defaults to "sgd" if not specified.
/// - `moe` is true if `--moe` flag is present.
/// - `num_experts` reads the value after `--num-experts`, defaulting to 1.
/// - `lr_schedule` selects the learning rate schedule: "step", "cosine" or
///   omitted for a constant rate. Additional hyperparameters can be provided via
///   `--lr-step-size`, `--lr-gamma` and `--lr-max-steps`.
/// - `resume` specifies a checkpoint file to load via `--resume <file>`.
/// - `save_every` saves checkpoints every N epochs with `--save-every N`.
/// - `checkpoint_dir` overrides the directory in which checkpoints are stored
///   using `--checkpoint-dir <dir>`.
/// - `positional_args` contains remaining positional arguments in order.
pub fn parse_cli<I>(
    mut args: I,
) -> (
    String,
    String,
    bool,
    usize,
    LrScheduleConfig,
    Option<String>,
    Option<usize>,
    Option<String>,
    Vec<String>,
)
where
    I: Iterator<Item = String>,
{
    let mut model = "transformer".to_string();
    let mut opt = "sgd".to_string();
    let mut moe = false;
    let mut num_experts = 1usize;
    let mut lr_sched = String::new();
    let mut step_size = 10usize;
    let mut gamma = 0.5f32;
    let mut max_steps = 1000usize;
    let mut resume = None;
    let mut save_every = None;
    let mut checkpoint_dir = None;
    let mut positional = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--moe" => moe = true,
            "--num-experts" => {
                if let Some(n) = args.next() {
                    num_experts = n.parse().unwrap_or(1);
                }
            }
            "--lr-schedule" => {
                if let Some(s) = args.next() {
                    lr_sched = s;
                }
            }
            "--lr-step-size" => {
                if let Some(v) = args.next() {
                    step_size = v.parse().unwrap_or(step_size);
                }
            }
            "--lr-gamma" => {
                if let Some(v) = args.next() {
                    gamma = v.parse().unwrap_or(gamma);
                }
            }
            "--lr-max-steps" => {
                if let Some(v) = args.next() {
                    max_steps = v.parse().unwrap_or(max_steps);
                }
            }
            "--resume" => {
                if let Some(p) = args.next() {
                    resume = Some(p);
                }
            }
            "--save-every" => {
                if let Some(v) = args.next() {
                    save_every = v.parse().ok();
                }
            }
            "--checkpoint-dir" => {
                if let Some(v) = args.next() {
                    checkpoint_dir = Some(v);
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

    let lr_schedule = match lr_sched.as_str() {
        "step" => LrScheduleConfig::Step { step_size, gamma },
        "cosine" => LrScheduleConfig::Cosine { max_steps },
        _ => LrScheduleConfig::Constant,
    };

    (
        model,
        opt,
        moe,
        num_experts,
        lr_schedule,
        resume,
        save_every,
        checkpoint_dir,
        positional,
    )
}

/// Convenience wrapper that parses arguments from the current process
/// (skipping the binary name).
pub fn parse_env(
) -> (
    String,
    String,
    bool,
    usize,
    LrScheduleConfig,
    Option<String>,
    Option<usize>,
    Option<String>,
    Vec<String>,
) {
    let args = env::args().skip(1);
    parse_cli(args)
}

fn main() {}
