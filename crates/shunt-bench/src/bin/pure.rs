//! Pure agentic benchmark runner.
//!
//!   cargo run -p shunt-bench --bin pure -- [catalog.toml] [runs] [--hard] [--task=NAME]
//!
//! Reads a catalog of (model, engine, quant) entries, runs each through the
//! capability task suite using a thin, harness-free tool loop, and writes
//! pure-report.md + pure-logs/<entry>/<task>-run-N.md.

use std::process::ExitCode;

use shunt_bench::capability::{Catalog, Difficulty};
use shunt_bench::pure::{render_pure_report, run_catalog_filtered};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let hard_only  = args.iter().any(|a| a == "--hard");
    let bench_only = args.iter().any(|a| a == "--bench");
    let task_filter = args
        .iter()
        .find(|a| a.starts_with("--task="))
        .map(|a| a.trim_start_matches("--task=").to_string());
    let positional: Vec<&str> = args.iter().filter(|a| !a.starts_with("--")).map(String::as_str).collect();

    let catalog_path = positional.first().copied().unwrap_or("pure-models.toml");
    let runs: usize = positional.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);

    let text = match std::fs::read_to_string(catalog_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read catalog '{catalog_path}': {e}");
            eprintln!("usage: pure [catalog.toml] [runs] [--hard] [--task=NAME]");
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

    let min_difficulty = if hard_only { Some(Difficulty::Medium) } else { None };
    let label = if bench_only { " (bench)" } else if hard_only { " (medium+hard)" } else { "" };
    let task_label = task_filter.as_deref().map(|t| format!(" --task={t}")).unwrap_or_default();
    eprintln!(
        "pure-bench: {} model(s) × suite{label}{task_label} × {runs} run(s)\n",
        catalog.model.len()
    );

    let entries = run_catalog_filtered(&catalog, runs, min_difficulty, task_filter.as_deref(), bench_only);
    let report = render_pure_report(&entries);

    println!("\n{report}");
    match std::fs::write("pure-report.md", &report) {
        Ok(()) => eprintln!("\nwrote pure-report.md"),
        Err(e) => eprintln!("(could not write pure-report.md: {e})"),
    }
    ExitCode::SUCCESS
}
