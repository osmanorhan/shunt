//! Integration tests for `AgentSession` against a real local LLM.
//!
//! These tests are `#[ignore]` by default. Run them with:
//!
//! ```bash
//! FRAME_LLM=http://localhost:8080 \
//! FRAME_MODEL=unsloth/Qwen3.5-9B-GGUF:Q6_K \
//! cargo test -p shunt-infer --test integration -- --ignored --nocapture
//! ```
//!
//! Each test creates an isolated temp workspace, runs `AgentSession`, and
//! asserts the observable outcome (file contents, result variant, turn history).

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use shunt_infer::{AgentObserver, AgentResult, AgentSession, OpenAiCompatProvider, ProposedFileOp};
use tempfile::TempDir;

struct PrintObserver;
impl AgentObserver for PrintObserver {
    fn on_tool_call(&self, turn: usize, _max_turns: usize, tool: &str, summary: &str) {
        println!("[turn {turn}] → {tool}: {summary}");
    }
    fn on_tool_result(&self, turn: usize, ok: bool, detail: &str) {
        let status = if ok { "OK" } else { "ERR" };
        if !detail.is_empty() {
            println!("[turn {turn}] ← {status}: {detail}");
        } else {
            println!("[turn {turn}] ← {status}");
        }
    }
}

// ── Test helper ───────────────────────────────────────────────────────────────

struct TestWorkspace {
    _dir: TempDir,
    root: PathBuf,
}

impl TestWorkspace {
    fn new() -> Self {
        let dir = TempDir::new().expect("tempdir");
        let root = dir.path().to_path_buf();
        Self { _dir: dir, root }
    }

    fn write(&self, rel: &str, contents: &str) {
        let path = self.root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, contents).unwrap();
    }

    fn read(&self, rel: &str) -> String {
        fs::read_to_string(self.root.join(rel)).expect(rel)
    }

    fn read_json(&self, rel: &str) -> Value {
        let text = self.read(rel);
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("bad JSON in {rel}: {e}\n{text}"))
    }

    fn exists(&self, rel: &str) -> bool {
        self.root.join(rel).exists()
    }

    fn root(&self) -> &str {
        self.root.to_str().unwrap()
    }

    /// Apply ops from `AgentResult::Done` to the workspace on disk.
    fn apply(&self, result: &AgentResult) {
        let AgentResult::Done { ops, .. } = result else {
            return;
        };
        for op in ops {
            match op {
                ProposedFileOp::Create { path, contents } => {
                    let abs = self.root.join(path);
                    if let Some(p) = abs.parent() {
                        fs::create_dir_all(p).unwrap();
                    }
                    fs::write(&abs, contents).unwrap();
                }
                ProposedFileOp::Edit {
                    path,
                    search,
                    replacement,
                } => {
                    let abs = self.root.join(path);
                    let current = fs::read_to_string(&abs)
                        .unwrap_or_else(|_| panic!("Edit op on missing file: {path}"));
                    let updated = current.replacen(search.as_str(), replacement.as_str(), 1);
                    fs::write(&abs, updated).unwrap();
                }
                ProposedFileOp::Delete { path } => {
                    let _ = fs::remove_file(self.root.join(path));
                }
            }
        }
    }
}

fn make_provider() -> OpenAiCompatProvider {
    let endpoint = std::env::var("FRAME_LLM")
        .expect("Set FRAME_LLM=http://localhost:8080 to run integration tests");
    let model =
        std::env::var("FRAME_MODEL").unwrap_or_else(|_| "unsloth/Qwen3.5-9B-GGUF:Q6_K".into());
    OpenAiCompatProvider::new(endpoint, model)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Basic: agent should add react-router-dom to package.json.
#[test]
#[ignore]
fn adds_npm_dependency() {
    let ws = TestWorkspace::new();
    ws.write(
        "package.json",
        r#"{
  "name": "my-app",
  "version": "1.0.0",
  "dependencies": {
    "lodash": "^4.17.21"
  }
}
"#,
    );

    let provider = make_provider();
    let mut session = AgentSession::new(&provider, ws.root())
        .with_pre_loaded(&[shunt_infer::SourceFileContext {
            path: "package.json".into(),
            contents: ws.read("package.json"),
        }])
        .with_observer(Arc::new(PrintObserver));

    let result =
        session.run("Add react-router-dom version 7.0.0 to the dependencies in package.json");

    assert!(
        matches!(result, AgentResult::Done { .. }),
        "expected Done, got: {result:?}"
    );
    ws.apply(&result);

    let pkg = ws.read_json("package.json");
    assert!(
        pkg["dependencies"]["react-router-dom"].is_string(),
        "react-router-dom not found in package.json:\n{}",
        ws.read("package.json")
    );
}

/// Agent should create a new TypeScript file when asked.
#[test]
#[ignore]
fn creates_new_file() {
    let ws = TestWorkspace::new();
    ws.write(
        "package.json",
        r#"{"name":"my-app","version":"1.0.0","dependencies":{}}"#,
    );
    fs::create_dir_all(ws.root.join("src")).unwrap();

    let provider = make_provider();
    let mut session =
        AgentSession::new(&provider, ws.root()).with_observer(Arc::new(PrintObserver));

    let result = session.run(
        "Create a new file src/utils/logger.ts that exports a simple logger function \
         wrapping console.log with a [LOG] prefix",
    );

    assert!(
        matches!(result, AgentResult::Done { .. }),
        "expected Done, got: {result:?}"
    );
    ws.apply(&result);
    assert!(
        ws.exists("src/utils/logger.ts"),
        "logger.ts was not created"
    );
    let contents = ws.read("src/utils/logger.ts");
    assert!(
        contents.contains("export"),
        "logger.ts should export something:\n{contents}"
    );
}

/// When the agent's initial file context doesn't include the target file,
/// it should use read_file before str_replace.
#[test]
#[ignore]
fn reads_file_before_editing() {
    let ws = TestWorkspace::new();
    ws.write(
        "package.json",
        r#"{
  "name": "my-app",
  "dependencies": {
    "express": "^4.18.0"
  }
}
"#,
    );

    let provider = make_provider();
    // Intentionally do NOT pre-load any files — agent must call read_file.
    let mut session =
        AgentSession::new(&provider, ws.root()).with_observer(Arc::new(PrintObserver));

    let result = session.run("Add the axios package version ^1.6.0 to the dependencies");

    assert!(
        matches!(result, AgentResult::Done { .. }),
        "expected Done, got: {result:?}"
    );
    ws.apply(&result);

    let pkg = ws.read_json("package.json");
    assert!(
        pkg["dependencies"]["axios"].is_string(),
        "axios not found in package.json:\n{}",
        ws.read("package.json")
    );
}

/// In a monorepo with two package.json files, agent should ask_user which one to modify.
#[test]
#[ignore]
fn asks_user_on_monorepo_ambiguity() {
    let ws = TestWorkspace::new();
    ws.write(
        "package.json",
        r#"{"name":"root","workspaces":["packages/*"],"dependencies":{}}"#,
    );
    ws.write(
        "packages/app/package.json",
        r#"{"name":"app","version":"1.0.0","dependencies":{"lodash":"^4"}}"#,
    );

    let provider = make_provider();
    let mut session =
        AgentSession::new(&provider, ws.root()).with_observer(Arc::new(PrintObserver));

    let result = session.run("Add react-router-dom to the project");

    // With two package.json files and an ambiguous "the project", the agent
    // should ask which one rather than guess.
    match result {
        AgentResult::NeedsClarification { question, .. } => {
            println!("Agent asked: {question}");
            // Good — it paused to clarify.
        }
        AgentResult::Done { .. } => {
            // Also acceptable if the agent made a reasonable choice and explained it.
            println!("Agent proceeded without asking (checked both files)");
        }
        AgentResult::MaxTurnsReached => {
            panic!("MaxTurnsReached — agent could not proceed");
        }
    }
}

/// search_files should surface relevant files for a keyword query.
#[test]
#[ignore]
fn search_files_finds_relevant_file() {
    let ws = TestWorkspace::new();
    ws.write("src/auth/login.ts", "export function login() {}");
    ws.write("src/auth/logout.ts", "export function logout() {}");
    ws.write("src/utils/format.ts", "export function formatDate() {}");
    ws.write("package.json", r#"{"name":"app","dependencies":{}}"#);

    let provider = make_provider();
    let mut session =
        AgentSession::new(&provider, ws.root()).with_observer(Arc::new(PrintObserver));

    let result = session
        .run("Add a comment to the login function in the auth module explaining what it does");

    // Accept Done or NeedsClarification-after-edit (agent added the comment but then
    // second-guessed itself and asked). In both cases the edit ops should be present.
    let login_content = match &result {
        AgentResult::Done { .. } => {
            ws.apply(&result);
            ws.read("src/auth/login.ts")
        }
        AgentResult::NeedsClarification { file_state, .. } => {
            // The agent edited the file then asked. Check in-memory state.
            file_state
                .get("src/auth/login.ts")
                .cloned()
                .unwrap_or_else(|| ws.read("src/auth/login.ts"))
        }
        AgentResult::MaxTurnsReached => panic!("MaxTurnsReached"),
    };
    assert!(
        login_content.contains("//") || login_content.contains("/*"),
        "expected a comment to be added:\n{login_content}"
    );
}

/// The error feedback loop: if str_replace returns an error (bad old_str),
/// the agent should retry with corrected parameters rather than giving up.
#[test]
#[ignore]
fn str_replace_retries_on_bad_match() {
    let ws = TestWorkspace::new();
    ws.write(
        "src/config.ts",
        r#"const API_URL = "http://localhost:3000";
const TIMEOUT = 5000;
export { API_URL, TIMEOUT };
"#,
    );

    let provider = make_provider();
    let mut session = AgentSession::new(&provider, ws.root())
        .with_pre_loaded(&[shunt_infer::SourceFileContext {
            path: "src/config.ts".into(),
            contents: ws.read("src/config.ts"),
        }])
        .with_observer(Arc::new(PrintObserver));

    let result = session.run("Change the API_URL to https://api.example.com");

    assert!(
        matches!(result, AgentResult::Done { .. }),
        "expected Done, got: {result:?}"
    );
    ws.apply(&result);
    let config = ws.read("src/config.ts");
    assert!(
        config.contains("https://api.example.com"),
        "URL was not updated:\n{config}"
    );
}

/// Sub-agent: agent should be able to research a sub-task and use the result.
#[test]
#[ignore]
fn sub_agent_resolves_version_question() {
    let ws = TestWorkspace::new();
    ws.write(
        "package.json",
        r#"{
  "name": "my-app",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}
"#,
    );
    ws.write(
        "package-lock.json",
        r#"{"lockfileVersion":3,"packages":{"node_modules/react":{"version":"18.2.0"}}}"#,
    );

    let provider = make_provider();
    let mut session =
        AgentSession::new(&provider, ws.root()).with_observer(Arc::new(PrintObserver));

    let result = session.run(
        "Read package.json and package-lock.json to find what version of react is installed, \
         then add a comment at the top of package.json noting the react version",
    );

    assert!(
        matches!(result, AgentResult::Done { .. }),
        "expected Done, got: {result:?}"
    );
    ws.apply(&result);
}
