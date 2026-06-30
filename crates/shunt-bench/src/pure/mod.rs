//! Pure agentic benchmark — measures model × engine × quant with a thin,
//! harness-free tool loop. See plan-bench.md for the design rationale.
//!
//! No imports from shunt-runtime, shunt-infer, shunt-localize, or shunt-edit.

pub mod agent_loop;
pub mod client;
pub mod metrics;
pub mod report;
pub mod tools;

use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::capability::{bench_suite, Catalog, Difficulty, ModelSpec, suite, CapabilityTask};

pub use report::{EntryScore, PureTaskScore, render_pure_report};

use agent_loop::{LoopConfig, run_loop};
use client::ChatClient;
use metrics::reduce;
use report::write_run_log;

/// Run every model in the catalog through the task suite × N runs each.
pub fn run_catalog(catalog: &Catalog, runs: usize) -> Vec<EntryScore> {
    run_catalog_filtered(catalog, runs, None, None, false)
}

pub fn run_catalog_filtered(
    catalog: &Catalog,
    runs: usize,
    min_difficulty: Option<Difficulty>,
    task_name: Option<&str>,
    bench_only: bool,
) -> Vec<EntryScore> {
    let all = if bench_only { bench_suite() } else { suite() };
    let tasks: Vec<CapabilityTask> = all
        .into_iter()
        .filter(|t| task_name.is_none_or(|n| t.name == n))
        .filter(|t| min_difficulty.is_none_or(|min| t.difficulty >= min))
        .collect();
    let tasks = std::sync::Arc::new(tasks);

    std::thread::scope(|scope| {
        let handles: Vec<_> = catalog
            .model
            .iter()
            .map(|spec| {
                let tasks = std::sync::Arc::clone(&tasks);
                scope.spawn(move || run_model(spec, &tasks, runs))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("pure bench thread panicked")).collect()
    })
}

fn run_model(spec: &ModelSpec, tasks: &[CapabilityTask], runs: usize) -> EntryScore {
    let entry_label = entry_label(spec);
    if !reachable(&spec.endpoint) {
        eprintln!("• {entry_label}: UNREACHABLE — skipped");
        return EntryScore {
            model: spec.name.clone(),
            engine: spec.engine.label().to_string(),
            quant: spec.quant.clone(),
            reachable: false,
            task_scores: Vec::new(),
        };
    }
    eprintln!("• {entry_label}");

    let client = match ChatClient::new(&spec.endpoint, spec.timeout_secs) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  client error: {e}");
            return EntryScore {
                model: spec.name.clone(),
                engine: spec.engine.label().to_string(),
                quant: spec.quant.clone(),
                reachable: false,
                task_scores: Vec::new(),
            };
        }
    };

    let loop_cfg = LoopConfig {
        model_id: spec.model_id.clone(),
        ..Default::default()
    };

    let entry = EntryScore {
        model: spec.name.clone(),
        engine: spec.engine.label().to_string(),
        quant: spec.quant.clone(),
        reachable: true,
        task_scores: Vec::new(),
    };

    let task_scores = tasks.iter().map(|task| {
        eprint!("    {:<22} {:<7} ", task.name, task.difficulty.label());
        let mut run_metrics = Vec::with_capacity(runs);

        for run_idx in 0..runs {
            let ws = task.workspace();
            let trace = run_loop(&client, &loop_cfg, &task.full_request(), ws.root());
            let passed = task.passed(&ws);

            let metrics = reduce(&trace, passed);
            eprint!("{}", metrics.outcome.glyph());

            let file_snapshots: Vec<(&str, String, bool)> = task
                .checks
                .iter()
                .map(|(rel, check)| {
                    let content = std::fs::read_to_string(ws.root().join(rel)).unwrap_or_default();
                    let ok = check(&content);
                    (*rel, content, ok)
                })
                .collect();

            write_run_log(&entry, task, run_idx + 1, &metrics, &trace, &file_snapshots);
            run_metrics.push(metrics);

            // Let the workspace drop before next run.
            drop(ws);
        }
        eprintln!();
        PureTaskScore {
            task: task.name.to_string(),
            difficulty: task.difficulty.label().to_string(),
            runs: run_metrics,
        }
    }).collect();

    EntryScore { task_scores, ..entry }
}

fn entry_label(spec: &ModelSpec) -> String {
    match &spec.quant {
        Some(q) => format!("{} [{}/{}]", spec.name, spec.engine.label(), q),
        None => format!("{} [{}]", spec.name, spec.engine.label()),
    }
}

fn reachable(endpoint: &str) -> bool {
    let hostport = endpoint
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap_or("");
    // Add default port if missing so ToSocketAddrs can resolve it.
    let hostport = if hostport.contains(':') { hostport.to_string() } else { format!("{hostport}:80") };
    match hostport.to_socket_addrs() {
        Ok(mut addrs) => addrs
            .next()
            .map(|addr| TcpStream::connect_timeout(&addr, Duration::from_secs(3)).is_ok())
            .unwrap_or(false),
        Err(_) => false,
    }
}
