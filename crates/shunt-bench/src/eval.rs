//! Probe-and-compose eval harness (M3).
//!
//! Runs scope resolution fixtures against the `ScopeOrchestrator` with a mock
//! provider and measures scope precision / recall against expected paths.
//!
//! Run with: `cargo test -p shunt-bench -- eval`

use std::collections::BTreeSet;
use std::fs;

use shunt_core::{ApprovalState, ArtifactId, TaskId, UnderstandingArtifact};
use shunt_runtime::probes::ScopeOrchestrator;
use tempfile::TempDir;
use time::macros::datetime;

use crate::mock::ScriptedProvider;

// ── EvalFixture ──────────────────────────────────────────────────────────────

/// A single eval scenario: workspace layout + request + expected scope.
pub struct EvalFixture {
    pub name: &'static str,
    /// Files to create: (relative path, contents).
    pub files: Vec<(&'static str, &'static str)>,
    pub request: &'static str,
    /// Paths that MUST appear in scope.
    pub must_contain: Vec<&'static str>,
    /// Maximum scope size (prevents hallucinated extra paths).
    pub max_scope: usize,
    /// Scripted provider responses for the NewFilePathProbe model call.
    /// Empty = probe will fail, which is fine if must_contain is met without it.
    pub provider_responses: Vec<serde_json::Value>,
}

/// Outcome of running one fixture.
#[derive(Debug)]
pub struct EvalResult {
    pub fixture: &'static str,
    pub scope: Vec<String>,
    pub hits: usize,
    pub precision: f32,
    pub recall: f32,
    pub passed: bool,
}

impl EvalResult {
    pub fn print(&self) {
        println!(
            "[{}] {} | scope={} hits={} precision={:.2} recall={:.2}",
            if self.passed { "PASS" } else { "FAIL" },
            self.fixture,
            self.scope.len(),
            self.hits,
            self.precision,
            self.recall,
        );
        for path in &self.scope {
            println!("    {path}");
        }
    }
}

// ── Runner ───────────────────────────────────────────────────────────────────

/// Run a fixture and return its `EvalResult`.
pub fn run_fixture(fixture: &EvalFixture) -> EvalResult {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path();

    // Build the workspace.
    for (path, contents) in &fixture.files {
        let abs = root.join(path);
        if let Some(parent) = abs.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&abs, contents).unwrap();
    }

    let artifact = make_artifact(fixture.request);
    let provider = ScriptedProvider::new(fixture.provider_responses.clone());
    let orchestrator = ScopeOrchestrator::default();

    let result = orchestrator.run(&root.to_string_lossy(), &artifact, &provider);
    let scope_set: BTreeSet<String> = result.target_scope.iter().cloned().collect();

    let must: BTreeSet<&str> = fixture.must_contain.iter().copied().collect();
    let hits = must.iter().filter(|p| scope_set.contains(**p)).count();

    let precision = if scope_set.is_empty() {
        0.0
    } else {
        hits as f32 / scope_set.len() as f32
    };
    let recall = if must.is_empty() {
        1.0
    } else {
        hits as f32 / must.len() as f32
    };

    let passed = recall >= 1.0 && scope_set.len() <= fixture.max_scope;

    EvalResult {
        fixture: fixture.name,
        scope: result.target_scope,
        hits,
        precision,
        recall,
        passed,
    }
}

fn make_artifact(request: &str) -> UnderstandingArtifact {
    UnderstandingArtifact {
        id: ArtifactId("eval-art".into()),
        task_id: TaskId("eval-task".into()),
        original_request: request.into(),
        interpreted_goal: request.into(),
        success_criteria: vec![],
        constraints: vec![],
        target_scope: vec![],
        work_contract: Default::default(),
        evidence: vec![],
        candidate_files: vec![],
        package_facts: vec![],
        manual_evidence: vec![],
        assumptions: vec![],
        ambiguities: vec![],
        selected_recipe: None,
        risks: vec![],
        confidence: 0.0,
        approval: ApprovalState::draft(),
        revision: 1,
        workspace_profile: shunt_core::WorkspaceProfile::default(),
        created_at: datetime!(2026-06-13 12:00 UTC),
        updated_at: datetime!(2026-06-13 12:00 UTC),
    }
}

// ── Built-in fixtures ─────────────────────────────────────────────────────────

/// "Add tokio dependency" — additive task with a Cargo.toml in the workspace.
/// Zero text matching: ManifestProbe finds Cargo.toml structurally.
pub fn add_dependency_rust() -> EvalFixture {
    EvalFixture {
        name: "add_dependency_rust",
        files: vec![
            (
                "Cargo.toml",
                "[package]\nname = \"my-app\"\nversion = \"0.1.0\"\n",
            ),
            ("src/main.rs", "fn main() {}"),
        ],
        request: "add tokio dependency to Cargo.toml",
        must_contain: vec!["Cargo.toml"],
        max_scope: 5,
        provider_responses: vec![],
    }
}

/// "Add express dependency" — same pattern but npm / TypeScript.
pub fn add_dependency_npm() -> EvalFixture {
    EvalFixture {
        name: "add_dependency_npm",
        files: vec![
            ("package.json", "{ \"name\": \"my-app\" }"),
            ("src/index.ts", "console.log('hi');"),
        ],
        request: "add express dependency to package.json",
        must_contain: vec!["package.json"],
        max_scope: 5,
        provider_responses: vec![],
    }
}

/// "Fix timeout" — modify task; ExistingFilesProbe finds the file via text match.
pub fn modify_existing_file() -> EvalFixture {
    EvalFixture {
        name: "modify_existing_file",
        files: vec![("src/client.rs", "pub fn call() { let timeout = 30; }")],
        request: "fix the timeout value in the client",
        must_contain: vec!["src/client.rs"],
        max_scope: 5,
        provider_responses: vec![],
    }
}

/// "Scaffold new file" — NewFilePathProbe invoked; provider returns new path.
pub fn scaffold_new_file() -> EvalFixture {
    EvalFixture {
        name: "scaffold_new_file",
        files: vec![("src/main.rs", "fn main() {}")],
        request: "create a new auth module",
        must_contain: vec!["src/auth.rs"],
        max_scope: 3,
        provider_responses: vec![serde_json::json!({ "path": "src/auth.rs" })],
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_probe_finds_cargo_toml_without_text_matching() {
        let result = run_fixture(&add_dependency_rust());
        result.print();
        assert!(
            result.passed,
            "fixture '{}' failed: scope={:?}",
            result.fixture, result.scope
        );
    }

    #[test]
    fn manifest_probe_finds_package_json_without_text_matching() {
        let result = run_fixture(&add_dependency_npm());
        result.print();
        assert!(
            result.passed,
            "fixture '{}' failed: scope={:?}",
            result.fixture, result.scope
        );
    }

    #[test]
    fn existing_files_probe_finds_modified_file() {
        let result = run_fixture(&modify_existing_file());
        result.print();
        assert!(
            result.passed,
            "fixture '{}' failed: scope={:?}",
            result.fixture, result.scope
        );
    }

    #[test]
    fn new_file_path_probe_resolves_scaffold_scope() {
        let result = run_fixture(&scaffold_new_file());
        result.print();
        assert!(
            result.passed,
            "fixture '{}' failed: scope={:?}",
            result.fixture, result.scope
        );
    }
}
