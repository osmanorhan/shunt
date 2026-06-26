//! Model-capability benchmark runner.
//!
//!   cargo run -p shunt-bench --bin capability -- [catalog.toml] [runs] [--hard]
//!
//! --hard  : run only Medium + Hard tasks (skip trivial/easy)
//! runs    : number of runs per task (default 3)
//!
//! Reads a model catalog, runs every reachable model through the task suite,
//! prints a markdown scorecard + leaderboard, and writes capability-report.md.

use std::process::ExitCode;

use shunt_bench::capability::{Catalog, Difficulty, render_report, run_catalog_filtered};

/// Initialise file-based tracing so `post_chat` request bodies land in capability-debug.log.
/// Every LLM request/response body is logged at DEBUG level by shunt_infer.
fn init_debug_log() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("shunt_infer=debug"));
    let file = match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("capability-debug.log")
    {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(
            fmt::layer()
                .with_writer(std::sync::Mutex::new(file))
                .with_ansi(false),
        )
        .try_init();
}

fn main() -> ExitCode {
    init_debug_log();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let hard_only = args.iter().any(|a| a == "--hard");
    let task_filter: Option<&str> = args
        .iter()
        .find(|a| a.starts_with("--task="))
        .map(|a| a.trim_start_matches("--task="));
    let positional: Vec<&str> = args
        .iter()
        .filter(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .collect();

    let catalog_path = positional.first().copied().unwrap_or("models.toml");
    let runs: usize = positional.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);

    let text = match std::fs::read_to_string(catalog_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read catalog '{catalog_path}': {e}");
            eprintln!("usage: capability [catalog.toml] [runs] [--hard]");
            return ExitCode::FAILURE;
        }
    };
    let catalog = match Catalog::from_toml(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("invalid catalog '{catalog_path}': {e}");
            return ExitCode::FAILURE;
        }
    };
    if catalog.model.is_empty() {
        eprintln!("catalog '{catalog_path}' has no [[model]] entries");
        return ExitCode::FAILURE;
    }

    let min_difficulty = if hard_only {
        Some(Difficulty::Medium)
    } else {
        None
    };
    let label = if hard_only { " (medium+hard only)" } else { "" };
    let task_label = task_filter
        .map(|t| format!(" --task={t}"))
        .unwrap_or_default();
    eprintln!(
        "Running {} model(s) × suite{label}{task_label} × {runs} run(s)…\n",
        catalog.model.len()
    );

    let cards = run_catalog_filtered(&catalog, runs, min_difficulty, task_filter);
    let report = render_report(&cards);

    println!("\n{report}");
    if let Err(e) = std::fs::write("capability-report.md", &report) {
        eprintln!("(could not write capability-report.md: {e})");
    } else {
        eprintln!("\nwrote capability-report.md");
    }
    ExitCode::SUCCESS
}
