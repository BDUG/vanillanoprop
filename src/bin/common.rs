use std::env;
use vanillanoprop::config::Config;
use vanillanoprop::fine_tune::FreezeSpec;
use vanillanoprop::optim::lr_scheduler::LrScheduleConfig;

/// Parses common CLI arguments across training binaries.
///
/// Returns a tuple `(model, optimizer, moe, num_experts, lr_schedule, resume, save_every, checkpoint_dir, log_dir, experiment_name, export_onnx, fine_tune, freeze_layers, auto_ml, config, positional_args)`.
/// - `model` defaults to "transformer" if not specified. Supported models include
///   "transformer", "cnn" and the new "lcm" large concept model.
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
/// - `log_dir` sets the base directory for metrics logs via `--log-dir <dir>`.
/// - `experiment_name` names the experiment for logging with `--experiment-name <name>`.
/// - `export_onnx` exports the model to ONNX via `--export-onnx <file>`.
/// - `fine_tune` specifies a Hugging Face model ID to initialise from using
///   `--fine-tune <id>`.
/// - `freeze_layers` is a comma-separated list of parameter indices to freeze via
///   `--freeze-layers 0,2,4`.
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
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Vec<FreezeSpec>,
    bool,
    Config,
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
    let mut lr_gamma = 0.5f32;
    let mut max_steps = 1000usize;
    let mut resume = None;
    let mut save_every = None;
    let mut checkpoint_dir = None;
    let mut log_dir = None;
    let mut experiment_name = None;
    let mut export_onnx = None;
    let mut fine_tune = None;
    let mut freeze_layers: Vec<FreezeSpec> = Vec::new();
    let mut auto_ml = false;
    let mut epochs = None;
    let mut batch_size = None;
    let mut gamma = None;
    let mut lam = None;
    let mut max_depth = None;
    let mut rollout_steps = None;
    let mut dataset = None;
    let mut config_path = None;
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
                    lr_gamma = v.parse().unwrap_or(lr_gamma);
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
            "--log-dir" => {
                if let Some(v) = args.next() {
                    log_dir = Some(v);
                }
            }
            "--experiment-name" => {
                if let Some(v) = args.next() {
                    experiment_name = Some(v);
                }
            }
            "--export-onnx" => {
                if let Some(v) = args.next() {
                    export_onnx = Some(v);
                }
            }
            "--fine-tune" => {
                if let Some(v) = args.next() {
                    fine_tune = Some(v);
                }
            }
            "--freeze-layers" => {
                if let Some(v) = args.next() {
                    freeze_layers = vanillanoprop::fine_tune::parse_freeze_list(&v);
                }
            }
            "--auto-ml" => {
                auto_ml = true;
            }
            "--epochs" => {
                if let Some(v) = args.next() {
                    epochs = v.parse().ok();
                }
            }
            "--batch-size" => {
                if let Some(v) = args.next() {
                    batch_size = v.parse().ok();
                }
            }
            "--gamma" => {
                if let Some(v) = args.next() {
                    gamma = v.parse().ok();
                }
            }
            "--lam" => {
                if let Some(v) = args.next() {
                    lam = v.parse().ok();
                }
            }
            "--max-depth" => {
                if let Some(v) = args.next() {
                    max_depth = v.parse().ok();
                }
            }
            "--rollout-steps" => {
                if let Some(v) = args.next() {
                    rollout_steps = v.parse().ok();
                }
            }
            "--dataset" => {
                if let Some(v) = args.next() {
                    dataset = Some(v);
                }
            }
            "--config" => {
                if let Some(v) = args.next() {
                    config_path = Some(v);
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
        "step" => LrScheduleConfig::Step {
            step_size,
            gamma: lr_gamma,
        },
        "cosine" => LrScheduleConfig::Cosine { max_steps },
        _ => LrScheduleConfig::Constant,
    };

    let mut config = config_path
        .as_deref()
        .and_then(Config::from_path)
        .unwrap_or_default();
    if let Some(e) = epochs {
        config.epochs = e;
    }
    if let Some(b) = batch_size {
        config.batch_size = b;
    }
    if let Some(g) = gamma {
        config.gamma = g;
    }
    if let Some(l) = lam {
        config.lam = l;
    }
    if let Some(d) = max_depth {
        config.max_depth = d;
    }
    if let Some(r) = rollout_steps {
        config.rollout_steps = r;
    }
    if let Some(d) = dataset {
        config.dataset = d;
    }

    (
        model,
        opt,
        moe,
        num_experts,
        lr_schedule,
        resume,
        save_every,
        checkpoint_dir,
        log_dir,
        experiment_name,
        export_onnx,
        fine_tune,
        freeze_layers,
        auto_ml,
        config,
        positional,
    )
}

/// Convenience wrapper that parses arguments from the current process
/// (skipping the binary name).
pub fn parse_env() -> (
    String,
    String,
    bool,
    usize,
    LrScheduleConfig,
    Option<String>,
    Option<usize>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Vec<FreezeSpec>,
    bool,
    Config,
    Vec<String>,
) {
    let args = env::args().skip(1);
    parse_cli(args)
}

fn main() {
    env_logger::init();
}
