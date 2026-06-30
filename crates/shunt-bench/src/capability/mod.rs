//! Model-capability benchmark.
//!
//! Runs every model in a catalog through a graded task suite, N times each, and
//! produces a comparable scorecard. The point is to **separate harness from model**:
//! the harness (drive_session → agent) is fixed; this measures how well each model
//! drives it. Expand by adding models (data, `model.rs`/`models.toml`) or tasks
//! (`task.rs::suite`). Adding an engine = one `Engine` arm + one match arm here.

mod model;
mod score;
mod task;

pub use model::{Catalog, Engine, ModelSpec};
pub use score::{ModelScorecard, Outcome, RunMetrics, TaskScore, render_report};
pub use task::{CapabilityTask, ContentCheck, Difficulty, bench_suite, suite};

use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::time::{Duration, Instant};

use shunt_core::machine::{StopReason, TaskState};
use shunt_infer::{OllamaProvider, OpenAiCompatProvider, ProviderCapabilities, ToolProvider};

use crate::harness::{self, ScenarioConfig, ScenarioResult};

/// Run the whole catalog and return one scorecard per model.
pub fn run_catalog(catalog: &Catalog, runs: usize) -> Vec<ModelScorecard> {
    run_catalog_filtered(catalog, runs, None, None)
}

/// Run the catalog, optionally filtering to tasks at or above `min_difficulty`
/// or to a single named task (for targeted debugging).
/// Models run concurrently; each model's task sequence is serial (single slot).
pub fn run_catalog_filtered(
    catalog: &Catalog,
    runs: usize,
    min_difficulty: Option<Difficulty>,
    task_name: Option<&str>,
) -> Vec<ModelScorecard> {
    let all = suite();
    let tasks: Vec<CapabilityTask> = all
        .into_iter()
        .filter(|t| {
            if let Some(name) = task_name {
                t.name == name
            } else {
                true
            }
        })
        .filter(|t| {
            if let Some(min) = min_difficulty {
                t.difficulty >= min
            } else {
                true
            }
        })
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
        handles
            .into_iter()
            .map(|h| h.join().expect("model benchmark thread panicked"))
            .collect()
    })
}

/// Run one model through the suite. Unreachable endpoints are skipped (not failed).
pub fn run_model(spec: &ModelSpec, tasks: &[CapabilityTask], runs: usize) -> ModelScorecard {
    let header = format!(
        "{} [{}] @ {}",
        spec.name,
        spec.engine.label(),
        spec.endpoint
    );
    if !reachable(&spec.endpoint) {
        eprintln!("• {header}: UNREACHABLE — skipped");
        return ModelScorecard {
            model: spec.name.clone(),
            engine: spec.engine.label().to_string(),
            reachable: false,
            tasks: Vec::new(),
        };
    }
    eprintln!("• {header}");

    // Engine dispatch: build the concrete provider, run the (generic) suite.
    let task_scores = if spec.engine.is_openai_compatible() {
        let caps = ProviderCapabilities::detect(&spec.model_id, &spec.endpoint);
        let provider = OpenAiCompatProvider::with_timeout(
            spec.endpoint.clone(),
            spec.model_id.clone(),
            Duration::from_secs(spec.timeout_secs),
        )
        .expect("valid provider config")
        .with_capabilities(caps);
        run_suite(&spec.name, provider, tasks, runs, spec.timeout_secs)
    } else {
        let provider = OllamaProvider::new(spec.endpoint.clone(), spec.model_id.clone());
        run_suite(&spec.name, provider, tasks, runs, spec.timeout_secs)
    };

    ModelScorecard {
        model: spec.name.clone(),
        engine: spec.engine.label().to_string(),
        reachable: true,
        tasks: task_scores,
    }
}

/// Data the run thread sends back before its workspace drops.
struct RunPayload {
    result: ScenarioResult,
    /// (rel_path, file_content) for every checked file.
    file_snapshots: Vec<(String, String)>,
    debug_log: Option<Vec<u8>>,
    elapsed: Duration,
}

/// The model-agnostic core: drive each task `runs` times and score it.
fn run_suite<P>(
    model_name: &str,
    provider: P,
    tasks: &[CapabilityTask],
    runs: usize,
    call_timeout_secs: u64,
) -> Vec<TaskScore>
where
    P: ToolProvider + Clone + Send + Sync + 'static,
{
    // Per-run wall-clock cap = enough for 3 retries + session overhead.
    // post_json waits call_timeout + 15s per attempt; 3 retries × that + 30s overhead.
    // Must exceed this so sessions always finish before we time them out — otherwise
    // the thread lingers, holds the server slot, and poisons subsequent tasks.
    let per_run_timeout = Duration::from_secs(call_timeout_secs * 3 + 45 + 30);
    // Drain timeout: after a run times out, wait up to this long for the thread to
    // finish its in-flight HTTP call and exit cleanly before starting the next task.
    let drain_timeout = Duration::from_secs(call_timeout_secs + 20);

    tasks
        .iter()
        .map(|task| {
            eprint!("    {:<18} {:<7} ", task.name, task.difficulty.label());
            let mut metrics = Vec::with_capacity(runs);
            for run_idx in 0..runs {
                let ws = task.workspace();
                let cfg = ScenarioConfig {
                    name: task.name,
                    request: task.full_request(),
                    prewarm: false,
                    ..Default::default()
                };
                let prov = provider.clone();
                let checks: Vec<ContentCheck> = task.checks.iter().copied().collect();
                let (tx, rx) = std::sync::mpsc::channel::<RunPayload>();
                let t0 = Instant::now();
                std::thread::spawn(move || {
                    let result = harness::run(&ws, cfg, prov);
                    let file_snapshots = checks
                        .iter()
                        .map(|(rel, _)| {
                            let content =
                                std::fs::read_to_string(ws.root().join(rel)).unwrap_or_default();
                            (rel.to_string(), content)
                        })
                        .collect();
                    let debug_log = std::fs::read(ws.root().join(".shunt/debug.log")).ok();
                    let elapsed = t0.elapsed();
                    let _ = tx.send(RunPayload {
                        result,
                        file_snapshots,
                        debug_log,
                        elapsed,
                    });
                    // ws drops here — TempDir cleaned up
                });

                let (outcome, payload) = match rx.recv_timeout(per_run_timeout) {
                    Ok(payload) => {
                        let passed = payload
                            .file_snapshots
                            .iter()
                            .zip(task.checks.iter())
                            .all(|((_, content), (_, check))| check(content));
                        (classify(&payload.result, passed), Some(payload))
                    }
                    Err(_) => {
                        eprint!("T"); // T = timed out
                        // Drain: wait for the thread to finish its in-flight HTTP call
                        // before starting the next task. Without this, the lingering
                        // thread holds the server slot and causes cascading failures.
                        let _ = rx.recv_timeout(drain_timeout);
                        let timeout_result = ScenarioResult {
                            name: task.name.to_string(),
                            final_state: TaskState::Stopped {
                                reason: StopReason::Failed {
                                    reason: format!(
                                        "benchmark per-run timeout ({}s)",
                                        per_run_timeout.as_secs()
                                    ),
                                },
                            },
                            notifications: Vec::new(),
                            total: per_run_timeout,
                            index_warm: None,
                        };
                        write_run_log(
                            model_name,
                            task,
                            run_idx + 1,
                            Outcome::NotCompleted,
                            &timeout_result,
                            &[],
                            None,
                        );
                        metrics.push(RunMetrics {
                            outcome: Outcome::NotCompleted,
                            tool_calls: 0,
                            elapsed: per_run_timeout,
                        });
                        continue;
                    }
                };

                eprint!("{}", outcome.glyph());
                if let Some(p) = payload {
                    write_run_log(
                        model_name,
                        task,
                        run_idx + 1,
                        outcome,
                        &p.result,
                        &p.file_snapshots,
                        p.debug_log.as_deref(),
                    );
                    metrics.push(RunMetrics {
                        outcome,
                        tool_calls: p.result.agent_tool_calls(),
                        elapsed: p.elapsed,
                    });
                }
                // Brief pause between runs: lets the previous harness tokio runtime fully
                // tear down before the next run creates a new one.  Without this, the first
                // inference call of the next run can hang (empty notification timeline).
                std::thread::sleep(Duration::from_secs(2));
            }
            eprintln!();
            TaskScore {
                task: task.name.to_string(),
                difficulty: task.difficulty.label().to_string(),
                runs: metrics,
            }
        })
        .collect()
}

fn classify(result: &ScenarioResult, correct: bool) -> Outcome {
    if !result.completed() {
        Outcome::NotCompleted
    } else if correct {
        Outcome::Success
    } else if result.change_proposed() {
        Outcome::WrongEdit
    } else {
        Outcome::NoEdit
    }
}

fn write_run_log(
    model_name: &str,
    task: &CapabilityTask,
    run_idx: usize,
    outcome: Outcome,
    result: &ScenarioResult,
    file_snapshots: &[(String, String)],
    debug_bytes: Option<&[u8]>,
) {
    let dir = Path::new("capability-logs").join(safe_name(model_name));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!(
            "(could not create capability log dir {}: {e})",
            dir.display()
        );
        return;
    }

    let stem = format!("{}-run-{run_idx}", safe_name(task.name));
    let md_path = dir.join(format!("{stem}.md"));
    let debug_path = dir.join(format!("{stem}.debug.log"));

    let mut log = String::new();
    log.push_str(&format!(
        "# {} / {} / run {}\n\n",
        model_name, task.name, run_idx
    ));
    log.push_str(&format!("- difficulty: {}\n", task.difficulty.label()));
    log.push_str(&format!("- outcome: {:?} `{}`\n", outcome, outcome.glyph()));
    log.push_str(&format!("- final_state: `{:?}`\n", result.final_state));
    log.push_str(&format!("- elapsed_ms: {}\n", result.total.as_millis()));
    log.push_str(&format!("- tool_calls: {}\n", result.agent_tool_calls()));
    log.push_str(&format!(
        "- change_proposed: {}\n",
        result.change_proposed()
    ));
    if let Some(reason) = result.stop_reason() {
        log.push_str(&format!("- stop_reason: `{reason}`\n"));
    }
    log.push_str("\n## Request\n\n");
    log.push_str(task.request);
    log.push_str("\n\n## Checked Files\n\n");
    for ((rel, content), (_, check)) in file_snapshots.iter().zip(task.checks.iter()) {
        log.push_str(&format!("### `{rel}`\n\n"));
        log.push_str(&format!("check_passed: `{}`\n\n", check(content)));
        log.push_str("```\n");
        log.push_str(content);
        if !content.ends_with('\n') {
            log.push('\n');
        }
        log.push_str("```\n\n");
    }
    log.push_str("## Notification Timeline\n\n");
    for (i, note) in result.notifications.iter().enumerate() {
        log.push_str(&format!("{}. `{:?}`\n", i + 1, note));
    }

    if let Err(e) = std::fs::write(&md_path, log) {
        eprintln!(
            "(could not write capability log {}: {e})",
            md_path.display()
        );
    }

    if let Some(bytes) = debug_bytes
        && let Err(e) = std::fs::write(&debug_path, bytes)
    {
        eprintln!("(could not write debug log {}: {e})", debug_path.display());
    }
}

fn safe_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Cheap TCP reachability check (no HTTP dep) so a down endpoint is skipped fast.
fn reachable(endpoint: &str) -> bool {
    let hostport = endpoint
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap_or("");
    match hostport.to_socket_addrs() {
        Ok(mut addrs) => addrs
            .next()
            .map(|addr| TcpStream::connect_timeout(&addr, Duration::from_secs(3)).is_ok())
            .unwrap_or(false),
        Err(_) => false,
    }
}
