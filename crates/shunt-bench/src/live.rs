//! Live agentic feedback loop — drives the unified session core against a real
//! model server, exactly as `agent --once` and the TUI do.
//!
//! Gate: set FRAME_TEST_ENDPOINT=http://127.0.0.1:8080 to enable.
//!       Set FRAME_TEST_MODEL to override the model (default: gemma4-12b).
//!
//! Run:
//!   FRAME_TEST_ENDPOINT=http://127.0.0.1:8080 \
//!     cargo test -p shunt-bench live -- --nocapture --test-threads=1
//!
//! `--test-threads=1` matters: a single local llama.cpp slot serialises requests,
//! so parallel tests contend and time out.
//!
//! Each test drives `harness::run` (→ `drive_session`) to a terminal state, then
//! asserts on what actually happened: the workspace ON DISK (auto-applied under
//! the headless policy) and the captured notification timeline. A run that hits
//! the agent turn budget surfaces as `Stopped::Failed` and fails the test.
//!
//! Two tiers:
//!   * DEFAULT GATE — tasks the agent passes reliably today. Red = regression.
//!   * FRONTIER PROBE (#[ignore]) — tasks it does not pass reliably yet; run with
//!     `--ignored` to track capability. Each carries its dated baseline.
//!
//! Without the endpoint set, every test no-ops.

use std::time::Duration;

use shunt_infer::OpenAiCompatProvider;

const DEFAULT_MODEL: &str = "gemma4-12b";

/// Returns a live provider if `FRAME_TEST_ENDPOINT` is set, otherwise `None`.
pub fn live_provider() -> Option<OpenAiCompatProvider> {
    let endpoint = std::env::var("FRAME_TEST_ENDPOINT").ok()?;
    let model = std::env::var("FRAME_TEST_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    Some(
        OpenAiCompatProvider::with_timeout(endpoint, model, Duration::from_secs(120))
            .expect("valid live provider config"),
    )
}

/// Returns `true` if live integration tests are enabled.
pub fn is_live() -> bool {
    std::env::var("FRAME_TEST_ENDPOINT").is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{self, Workspace};
    use crate::harness::{self, DriverMode, ScenarioConfig, ScenarioResult};

    use shunt_core::machine::AutonomyPolicy;

    /// Read a workspace file after a run; empty string if missing.
    fn read_file(workspace: &Workspace, rel: &str) -> String {
        std::fs::read_to_string(workspace.root().join(rel)).unwrap_or_default()
    }

    /// Assert the agent actually completed (didn't thrash to the turn budget).
    fn assert_completed(result: &ScenarioResult) {
        assert!(
            result.completed(),
            "[{}] did not complete: {:?}",
            result.name,
            result.final_state
        );
    }

    // ── DEFAULT GATE ──────────────────────────────────────────────────────────

    /// EASY (regression floor): a single targeted constant edit in one TS file,
    /// fully autonomous (headless policy, no pauses).
    #[test]
    fn live_agent_change_timeout_ts() {
        let Some(provider) = live_provider() else {
            return;
        };
        let workspace = fixtures::ts_app();
        let config = ScenarioConfig {
            name: "agent_change_timeout",
            request: "in src/config.ts change the default timeout from 5000 to 30000".into(),
            ..Default::default()
        };
        let result = harness::run(&workspace, config, provider);
        result.print();
        assert_completed(&result);
        let cfg = read_file(&workspace, "src/config.ts");
        let landed = cfg.contains("30000") && !cfg.contains("5000");
        println!("  scorecard: timeout_changed={landed} (want true: 30000 present, 5000 gone)");
        assert!(landed, "timeout not updated in src/config.ts:\n{cfg}");
    }

    // ── FRONTIER PROBES (#[ignore]) ──────────────────────────────────────────
    //
    // Baseline (gemma4-12b, 2026-06-21): the agent intermittently emits
    // str_replace with an empty old_str (content-collection returns empty) and
    // loops to the turn budget. Promote to the gate once reliable.

    /// FRONTIER: add a self-contained function to one file.
    #[test]
    #[ignore = "frontier: agent str_replace/old_str flakiness on gemma4-12b (2026-06-21)"]
    fn live_agent_add_function_rust() {
        let Some(provider) = live_provider() else {
            return;
        };
        let workspace = fixtures::rust_cli();
        let config = ScenarioConfig {
            name: "agent_add_function",
            request: "add a public function named farewell to src/lib.rs that takes name: &str and returns a String saying goodbye to that name".into(),
            ..Default::default()
        };
        let result = harness::run(&workspace, config, provider);
        result.print();
        assert_completed(&result);
        let lib = read_file(&workspace, "src/lib.rs");
        let landed = lib.contains("fn farewell");
        println!("  scorecard: farewell_in_lib={landed} (want true)");
        assert!(landed, "farewell not found in src/lib.rs:\n{lib}");
    }

    /// FRONTIER: rename a symbol across two files (multi-file edit).
    #[test]
    #[ignore = "frontier: multi-file edit exceeds reliable agent turn budget (2026-06-21)"]
    fn live_agent_rename_across_files_rust() {
        let Some(provider) = live_provider() else {
            return;
        };
        let workspace = fixtures::rust_cli();
        let config = ScenarioConfig {
            name: "agent_rename_across_files",
            request: "rename the function greet to greet_user in src/lib.rs and update its caller in src/main.rs".into(),
            ..Default::default()
        };
        let result = harness::run(&workspace, config, provider);
        result.print();
        assert_completed(&result);
        let lib = read_file(&workspace, "src/lib.rs");
        let main = read_file(&workspace, "src/main.rs");
        println!(
            "  scorecard: def={} caller={}",
            lib.contains("fn greet_user"),
            main.contains("greet_user")
        );
        assert!(lib.contains("fn greet_user"), "lib.rs not renamed:\n{lib}");
        assert!(
            main.contains("greet_user"),
            "main.rs caller not updated:\n{main}"
        );
    }

    /// FRONTIER (interactive): drive the *approval gate* — `agentic()` policy makes
    /// the machine pause at `WaitingForUser::Approval`; the ScriptedResponder
    /// approves. Verifies the human-in-the-loop pause/resume path end to end.
    #[test]
    #[ignore = "frontier: exercises the interactive approval gate against a live model"]
    fn live_agent_interactive_approval_ts() {
        let Some(provider) = live_provider() else {
            return;
        };
        let workspace = fixtures::ts_app();
        let config = ScenarioConfig {
            name: "agent_interactive_approval",
            request: "in src/config.ts change the default timeout from 5000 to 30000".into(),
            policy: AutonomyPolicy::agentic(), // approval: Ask → pauses
            mode: DriverMode::Scripted(vec![]), // no clarifications scripted; approves plan
            ..Default::default()
        };
        let result = harness::run(&workspace, config, provider);
        result.print();
        assert_completed(&result);
        assert!(
            result.change_proposed(),
            "expected a ChangeProposed notification before approval"
        );
        let cfg = read_file(&workspace, "src/config.ts");
        assert!(
            cfg.contains("30000"),
            "timeout not updated after approval:\n{cfg}"
        );
    }
}
