//! Driver-based scenario harness.
//!
//! Drives the real session core (`drive_session`) exactly as `agent --once` and
//! the TUI do — there is no bench-only business path. Each scenario runs a task
//! to a terminal state against a fixture (or real) workspace, captures the
//! notification timeline + final state, and leaves the workspace mutated on disk
//! and a `.shunt/debug.log` to analyse.

use std::time::{Duration, Instant};

use shunt_core::machine::{AutonomyPolicy, Notification, TaskState};
use shunt_core::{ArtifactId, RecipeRef};
use shunt_infer::ToolProvider;
use shunt_runtime::driver::{AutoResponder, Responder, ScriptedResponder, drive_session};

use crate::fixtures::Workspace;

// ── Config ────────────────────────────────────────────────────────────────────

/// How the simulated developer responds to pauses.
pub enum DriverMode {
    /// Fully autonomous: approve plans, auto-answer agent questions. The
    /// "full agentic" channel.
    Auto,
    /// Script the developer's clarification answers in order; approve the plan.
    Scripted(Vec<String>),
}

pub struct ScenarioConfig {
    pub name: &'static str,
    pub request: String,
    /// Pre-warm the search index before timing the run.
    pub prewarm: bool,
    /// Autonomy policy: `headless()` (no pauses) or `agentic()` (approval pause).
    pub policy: AutonomyPolicy,
    pub mode: DriverMode,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            name: "unnamed",
            request: String::new(),
            prewarm: true,
            policy: AutonomyPolicy::headless(),
            mode: DriverMode::Auto,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

pub struct ScenarioResult {
    pub name: String,
    pub final_state: TaskState,
    pub notifications: Vec<Notification>,
    pub total: Duration,
    pub index_warm: Option<Duration>,
}

impl ScenarioResult {
    pub fn completed(&self) -> bool {
        matches!(self.final_state, TaskState::Completed)
    }

    pub fn failed(&self) -> bool {
        matches!(self.final_state, TaskState::Stopped { .. })
    }

    /// Reason string if the run stopped/failed.
    pub fn stop_reason(&self) -> Option<String> {
        match &self.final_state {
            TaskState::Stopped { reason } => Some(format!("{reason:?}")),
            _ => None,
        }
    }

    pub fn clarifications(&self) -> usize {
        self.notifications
            .iter()
            .filter(|n| matches!(n, Notification::ClarificationNeeded { .. }))
            .count()
    }

    /// Count of agent tool calls observed (a proxy for loop/thrash).
    pub fn agent_tool_calls(&self) -> usize {
        self.notifications
            .iter()
            .filter(|n| matches!(n, Notification::AgentToolCall { .. }))
            .count()
    }

    /// True if any change set was proposed.
    pub fn change_proposed(&self) -> bool {
        self.notifications
            .iter()
            .any(|n| matches!(n, Notification::ChangeProposed { .. }))
    }

    pub fn print(&self) {
        println!("=== {} ===", self.name);
        println!("  final   : {:?}", self.final_state);
        if let Some(warm) = self.index_warm {
            println!("  index warm  : {:>6} ms", warm.as_millis());
        }
        println!("  total   : {:>6} ms", self.total.as_millis());
        println!(
            "  signals : tool_calls={} clarifications={} change_proposed={}",
            self.agent_tool_calls(),
            self.clarifications(),
            self.change_proposed(),
        );
        if let Some(reason) = self.stop_reason() {
            println!("  stop    : {reason}");
        }
        println!();
    }
}

// ── Runner ────────────────────────────────────────────────────────────────────

/// Run one scenario against `workspace` using `provider`. Drives the unified
/// session core to a terminal state.
pub fn run<P>(workspace: &Workspace, config: ScenarioConfig, provider: P) -> ScenarioResult
where
    P: ToolProvider + Clone + Send + Sync + 'static,
{
    use shunt_localize::SemanticLocalizer;

    let index_warm = if config.prewarm {
        let localizer = SemanticLocalizer::default();
        let t = Instant::now();
        let _ = localizer.prewarm(workspace.root());
        Some(t.elapsed())
    } else {
        None
    };

    // File-based store inside the workspace — the session opens it per effect.
    let shunt_dir = workspace.root().join(".shunt");
    let _ = std::fs::create_dir_all(&shunt_dir);
    let now = time::OffsetDateTime::now_utc();
    let store_path = shunt_dir
        .join(format!("bench-{}.db", now.unix_timestamp_nanos()))
        .display()
        .to_string();
    let task_id = format!("bench-{}", now.unix_timestamp_nanos());
    let artifact_id = ArtifactId(format!("art-{}", now.unix_timestamp_nanos()));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime");

    let mut responder: Box<dyn Responder> = match config.mode {
        DriverMode::Auto => Box::new(AutoResponder::default()),
        DriverMode::Scripted(answers) => Box::new(ScriptedResponder::new(answers)),
    };

    let wall = Instant::now();
    let outcome = drive_session(
        task_id,
        artifact_id,
        workspace.root_str(),
        RecipeRef {
            id: "manual.inspect-propose".into(),
            version: "v1".into(),
        },
        store_path,
        provider,
        config.policy,
        rt.handle().clone(),
        config.request.clone(),
        responder.as_mut(),
    );
    let total = wall.elapsed();
    drop(rt); // wind down the session's effect pool

    ScenarioResult {
        name: config.name.to_string(),
        final_state: outcome.final_state,
        notifications: outcome.notifications,
        total,
        index_warm,
    }
}
