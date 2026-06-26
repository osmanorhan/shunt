//! Exploratory harness — stretches the agent's capabilities to find the current frontier.
//!
//! NOT a regression suite. Tasks are tiered from trivial → hard → stretch.
//! Failures are expected at the frontier and guide what to improve next.
//!
//! ```bash
//! FRAME_LLM=http://localhost:8080 \
//! FRAME_MODEL=unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL \
//! cargo test -p shunt-infer --test harness -- --ignored --nocapture
//! ```
//!
//! To run a single tier:   ... -- --nocapture tier_1
//! To run one scenario:    ... -- --nocapture single -- <name>

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use shunt_infer::engine::{EngineKind, detect_engine};
use shunt_infer::{
    AgentObserver, AgentResult, AgentSession, InferResult, OllamaProvider, OpenAiCompatProvider,
    ProposedFileOp, ProviderCapabilities, SourceFileContext, ToolCall, ToolChoiceMode,
    ToolProvider, ToolSpec,
};
use serde_json::Value;
use tempfile::TempDir;

// ── Workspace ────────────────────────────────────────────────────────────────

struct Ws {
    _dir: TempDir,
    root: PathBuf,
}

impl Ws {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let root = dir.path().to_path_buf();
        Self { _dir: dir, root }
    }

    fn write(&self, rel: &str, contents: &str) -> &Self {
        let p = self.root.join(rel);
        if let Some(d) = p.parent() {
            fs::create_dir_all(d).unwrap();
        }
        fs::write(p, contents).unwrap();
        self
    }

    fn read(&self, rel: &str) -> String {
        fs::read_to_string(self.root.join(rel)).unwrap_or_else(|_| format!("<missing: {rel}>"))
    }

    fn read_json(&self, rel: &str) -> Value {
        serde_json::from_str(&self.read(rel)).unwrap_or(Value::Null)
    }

    fn exists(&self, rel: &str) -> bool {
        self.root.join(rel).exists()
    }

    fn root_str(&self) -> &str {
        self.root.to_str().unwrap()
    }

    fn apply(&self, result: &AgentResult) {
        let AgentResult::Done { ops, .. } = result else {
            return;
        };
        for op in ops {
            match op {
                ProposedFileOp::Create { path, contents } => {
                    let abs = self.root.join(path);
                    if let Some(d) = abs.parent() {
                        fs::create_dir_all(d).unwrap();
                    }
                    fs::write(&abs, contents).unwrap();
                }
                ProposedFileOp::Edit {
                    path,
                    search,
                    replacement,
                } => {
                    let abs = self.root.join(path);
                    if let Ok(cur) = fs::read_to_string(&abs) {
                        fs::write(&abs, cur.replacen(search.as_str(), replacement.as_str(), 1))
                            .unwrap();
                    }
                }
                ProposedFileOp::Delete { path } => {
                    let _ = fs::remove_file(self.root.join(path));
                }
            }
        }
    }
}

// ── Turn recorder ─────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Log(Arc<Mutex<Vec<Turn>>>);

#[derive(Clone)]
struct Turn {
    tool: String,
    ok: bool,
    detail: String,
}

impl Log {
    fn new() -> Self {
        Self(Arc::new(Mutex::new(vec![])))
    }
    fn turns(&self) -> Vec<Turn> {
        self.0.lock().unwrap().clone()
    }
    fn count(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

impl AgentObserver for Log {
    fn on_tool_call(&self, _t: usize, _max_turns: usize, tool: &str, summary: &str) {
        self.0.lock().unwrap().push(Turn {
            tool: tool.into(),
            ok: true,
            detail: summary.into(),
        });
    }
    fn on_tool_result(&self, _t: usize, ok: bool, detail: &str) {
        if let Some(last) = self.0.lock().unwrap().last_mut() {
            last.ok = ok;
            if !detail.is_empty() {
                last.detail = format!("{} → {}", last.detail, &detail[..detail.len().min(120)]);
            }
        }
    }
}

// ── AnyProvider: wraps OpenAiCompatProvider or OllamaProvider ────────────────

/// Routes to either backend so the harness can use ollama natively (for thinking
/// models that need `think: false`) or the OpenAI-compat shim for llama.cpp.
enum AnyProvider {
    Compat(OpenAiCompatProvider),
    #[allow(dead_code)]
    Ollama(OllamaProvider),
}

impl ToolProvider for AnyProvider {
    fn call_tool(&self, system: &str, user: &str, tool: &ToolSpec) -> InferResult<ToolCall> {
        match self {
            AnyProvider::Compat(p) => p.call_tool(system, user, tool),
            AnyProvider::Ollama(p) => p.call_tool(system, user, tool),
        }
    }

    fn generate_text(&self, system: &str, user: &str) -> InferResult<String> {
        match self {
            AnyProvider::Compat(p) => p.generate_text(system, user),
            AnyProvider::Ollama(p) => p.generate_text(system, user),
        }
    }
}

// ── Scenario ─────────────────────────────────────────────────────────────────

struct Scenario {
    name: &'static str,
    tier: u8,
    /// Brief note on what capability this is stretching.
    probes: &'static str,
    run: Box<dyn Fn(&AnyProvider) -> Run + Send>,
}

struct Run {
    outcome: Outcome,
    turns: usize,
    elapsed_s: u64,
    log: Vec<Turn>,
    note: Option<String>,
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Pass,
    Fail,
    #[allow(dead_code)]
    Partial(&'static str),
}

/// Build the provider, auto-selecting backend from the endpoint:
/// - Ollama (:11434) → `OllamaProvider` (native `/api/chat` with `think:false`)
/// - Everything else → `OpenAiCompatProvider` with auto-detected capabilities
fn make_provider() -> AnyProvider {
    let ep = std::env::var("FRAME_LLM").expect("Set FRAME_LLM");
    let model = std::env::var("FRAME_MODEL")
        .unwrap_or_else(|_| "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL".into());
    let mut caps = ProviderCapabilities::detect(&model, &ep);
    if detect_engine(&ep) == EngineKind::Ollama {
        // Ollama grammar-constrained decoding enforces JSON structure but not content —
        // models produce empty old_str/contents. Function calling via the /v1 shim
        // with NamedObject mode (tool_choice={type:function,function:{name:agent_action}})
        // routes arguments through tool_calls[].function.arguments, which is not affected
        // by the empty-content issue that occurs when thinking goes to reasoning_content.
        // qwen3-family small models (4B) can produce proper content via this path.
        caps.tool_choice_mode = ToolChoiceMode::NamedObject;
    }
    AnyProvider::Compat(OpenAiCompatProvider::new(&ep, &model).with_capabilities(caps))
}

/// Per-scenario turn cap. Override with HARNESS_MAX_TURNS=N.
/// Default 12 keeps each scenario ≤ ~90s for fast iteration.
/// Set 30 for thorough runs where you want to see if the model can power through.
fn harness_max_turns() -> usize {
    std::env::var("HARNESS_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12)
}

fn session<'a>(
    p: &'a AnyProvider,
    ws: &Ws,
    preload: &[(&str, &str)],
) -> (AgentSession<'a, AnyProvider>, Log) {
    let log = Log::new();
    let files: Vec<SourceFileContext> = preload
        .iter()
        .map(|(p, c)| SourceFileContext {
            path: p.to_string(),
            contents: c.to_string(),
        })
        .collect();
    let s = AgentSession::new(p, ws.root_str())
        .with_budget(shunt_infer::SessionBudget {
            max_turns: harness_max_turns(),
            stall_warn_at: 4,
            stall_abort_at: 7,
        })
        .with_pre_loaded(&files)
        .with_observer(Arc::new(log.clone()));
    (s, log)
}

fn run_scenario(s: &Scenario, p: &AnyProvider) -> Run {
    let t0 = Instant::now();
    let mut r = (s.run)(p);
    r.elapsed_s = t0.elapsed().as_secs();
    r
}

fn result_kind(r: &AgentResult) -> &'static str {
    match r {
        AgentResult::Done { .. } => "Done",
        AgentResult::NeedsClarification { .. } => "Clarify",
        AgentResult::MaxTurnsReached => "MaxTurns",
    }
}

// ── Scenario library ──────────────────────────────────────────────────────────

fn scenarios() -> Vec<Scenario> {
    vec![
        // ════════════════════════════════════════════════════════
        // TIER 1 — Fundamentals: single-file, obvious target
        // ════════════════════════════════════════════════════════
        Scenario {
            name: "t1_add_dep_preloaded",
            tier: 1,
            probes: "json str_replace with file in context",
            run: Box::new(|p| {
                let ws = Ws::new();
                let src = r#"{"name":"app","dependencies":{"lodash":"^4"}}"#;
                ws.write("package.json", src);
                let (mut s, log) = session(p, &ws, &[("package.json", src)]);
                let r = s.run("Add react@^18.0.0 to dependencies");
                ws.apply(&r);
                let turns = log.count();
                let pkg = ws.read_json("package.json");
                let pass = pkg["dependencies"]["react"].is_string();
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note: if !pass {
                        Some(ws.read("package.json"))
                    } else {
                        None
                    },
                }
            }),
        },
        Scenario {
            name: "t1_edit_constant",
            tier: 1,
            probes: "str_replace exact match",
            run: Box::new(|p| {
                let ws = Ws::new();
                let src = "export const BASE_URL = \"http://localhost:3000\";\n";
                ws.write("src/config.ts", src);
                let (mut s, log) = session(p, &ws, &[("src/config.ts", src)]);
                let r = s.run("Change BASE_URL to https://api.prod.example.com");
                ws.apply(&r);
                let turns = log.count();
                let content = ws.read("src/config.ts");
                let pass = content.contains("https://api.prod.example.com");
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note: if !pass { Some(content) } else { None },
                }
            }),
        },
        Scenario {
            name: "t1_create_file",
            tier: 1,
            probes: "write_file with parent dir creation",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Create src/utils/logger.ts exporting a log(msg: string) function that console.logs with a [LOG] prefix");
                ws.apply(&r);
                let turns = log.count();
                let exists = ws.exists("src/utils/logger.ts");
                let content = ws.read("src/utils/logger.ts");
                let pass = exists && content.contains("export");
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note: if !pass {
                        Some(format!("exists={exists}\n{content}"))
                    } else {
                        None
                    },
                }
            }),
        },
        Scenario {
            name: "t1_delete_file",
            tier: 1,
            probes: "delete_file + done immediately",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write("src/deprecated.ts", "// old code\nexport {};\n");
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let _result = s.run("Delete src/deprecated.ts");
                let turns = log.count();
                let pass = !ws.exists("src/deprecated.ts");
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note: if !pass {
                        Some("file still exists".into())
                    } else {
                        None
                    },
                }
            }),
        },
        Scenario {
            name: "t1_read_then_edit",
            tier: 1,
            probes: "read_file before edit (no preload)",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write(
                    "src/constants.ts",
                    "export const MAX_RETRIES = 3;\nexport const TIMEOUT = 5000;\n",
                );
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Set MAX_RETRIES to 10 in src/constants.ts");
                ws.apply(&r);
                let turns = log.count();
                let content = ws.read("src/constants.ts");
                let pass = content.contains("MAX_RETRIES = 10");
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note: if !pass { Some(content) } else { None },
                }
            }),
        },
        // ════════════════════════════════════════════════════════
        // TIER 2 — Context: agent must reason about structure
        // ════════════════════════════════════════════════════════
        Scenario {
            name: "t2_search_and_edit",
            tier: 2,
            probes: "search_files to locate target, then edit",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write(
                    "src/auth/login.ts",
                    "export function login(u: string, p: string) { return u === p; }\n",
                );
                ws.write(
                    "src/auth/logout.ts",
                    "export function logout() { return true; }\n",
                );
                ws.write(
                    "src/utils/format.ts",
                    "export function fmt(d: Date) { return d.toISOString(); }\n",
                );
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Add a // TODO: validate password strength comment above the login function body");
                ws.apply(&r);
                let turns = log.count();
                let content = ws.read("src/auth/login.ts");
                let pass = content.contains("TODO") || content.contains("//");
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note: if !pass { Some(content) } else { None },
                }
            }),
        },
        Scenario {
            name: "t2_ambiguous_target_asks",
            tier: 2,
            probes: "ask_user when target is genuinely ambiguous",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write(
                    "packages/web/package.json",
                    r#"{"name":"web","dependencies":{"react":"^18"}}"#,
                );
                ws.write(
                    "packages/api/package.json",
                    r#"{"name":"api","dependencies":{"express":"^4"}}"#,
                );
                ws.write(
                    "package.json",
                    r#"{"name":"root","workspaces":["packages/*"]}"#,
                );
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Add zod@^3 to the project");
                let turns = log.count();
                // Pass if it asked (conflict/ambiguity) or made a reasonable choice without looping
                let pass = !matches!(r, AgentResult::MaxTurnsReached);
                let note = match &r {
                    AgentResult::NeedsClarification { question, .. } => {
                        Some(format!("asked: {question}"))
                    }
                    AgentResult::Done { description, .. } => Some(format!("chose: {description}")),
                    AgentResult::MaxTurnsReached => Some("MaxTurnsReached".into()),
                };
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t2_conflict_detection",
            tier: 2,
            probes: "detect already-present dep, raise to user",
            run: Box::new(|p| {
                let ws = Ws::new();
                let src = r#"{"name":"app","dependencies":{"react":"^18","react-dom":"^18"}}"#;
                ws.write("package.json", src);
                let (mut s, log) = session(p, &ws, &[("package.json", src)]);
                let r = s.run("Add react to the dependencies");
                ws.apply(&r);
                let turns = log.count();
                let _pkg = ws.read_json("package.json");
                let react_count = ws.read("package.json").matches("\"react\"").count();
                // Pass: flagged OR left package intact (no duplication)
                let pass = matches!(r, AgentResult::NeedsClarification { .. }) || react_count <= 2;
                let note = Some(format!(
                    "{} (react mentions={})",
                    result_kind(&r),
                    react_count
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t2_multi_step_read_write",
            tier: 2,
            probes: "read two files, synthesize, write third",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write(
                    "package.json",
                    r#"{"name":"myapp","version":"2.3.1","dependencies":{"react":"^18.2.0"}}"#,
                );
                ws.write("src/index.ts", "// entry point\nexport {};\n");
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Read package.json to get the app name and version, then add a comment at the top of src/index.ts: // myapp v<version>");
                ws.apply(&r);
                let turns = log.count();
                let content = ws.read("src/index.ts");
                let pass = content.contains("myapp") && content.contains("2.3.1");
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note: if !pass { Some(content) } else { None },
                }
            }),
        },
        Scenario {
            name: "t2_ambiguous_str_replace",
            tier: 2,
            probes: "handle non-unique str_replace, retry with more context",
            run: Box::new(|p| {
                let ws = Ws::new();
                // Same pattern repeated — must include surrounding context to be unique
                let src = "const A = 1;\nconst B = 1;\nconst C = 1;\nexport { A, B, C };\n";
                ws.write("src/nums.ts", src);
                let (mut s, log) = session(p, &ws, &[("src/nums.ts", src)]);
                let r = s.run("Change only the value of B to 99");
                ws.apply(&r);
                let turns = log.count();
                let content = ws.read("src/nums.ts");
                let pass = content.contains("B = 99")
                    && content.contains("A = 1")
                    && content.contains("C = 1");
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note: if !pass { Some(content) } else { None },
                }
            }),
        },
        // ════════════════════════════════════════════════════════
        // TIER 3 — Reasoning: infer intent, multi-file, structure
        // ════════════════════════════════════════════════════════
        Scenario {
            name: "t3_rename_across_files",
            tier: 3,
            probes: "propagate rename through multiple files",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write(
                    "src/db/client.ts",
                    "export class DatabaseClient { connect() {} }\n",
                );
                ws.write("src/services/user.ts", "import { DatabaseClient } from '../db/client';\nexport class UserService { constructor(private db: DatabaseClient) {} }\n");
                ws.write("src/app.ts",          "import { DatabaseClient } from './db/client';\nconst db = new DatabaseClient();\n");
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Rename DatabaseClient to DbClient everywhere it's used");
                ws.apply(&r);
                let turns = log.count();
                let client = ws.read("src/db/client.ts");
                let service = ws.read("src/services/user.ts");
                let app = ws.read("src/app.ts");
                let pass = !client.contains("DatabaseClient")
                    && !service.contains("DatabaseClient")
                    && !app.contains("DatabaseClient")
                    && client.contains("DbClient");
                let note = Some(format!(
                    "client ok={}, service ok={}, app ok={}",
                    !client.contains("DatabaseClient"),
                    !service.contains("DatabaseClient"),
                    !app.contains("DatabaseClient"),
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t3_create_and_wire",
            tier: 3,
            probes: "create module + update barrel export + update consumer",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write(
                    "src/index.ts",
                    "export { UserService } from './services/user';\n",
                );
                ws.write("src/services/user.ts", "export class UserService {}\n");
                ws.write("src/app.ts", "import { UserService } from './index';\n");
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Add a new AuthService class in src/services/auth.ts, export it from src/index.ts, and import it in src/app.ts alongside UserService");
                ws.apply(&r);
                let turns = log.count();
                let auth_exists = ws.exists("src/services/auth.ts")
                    && ws.read("src/services/auth.ts").contains("AuthService");
                let index_exports = ws.read("src/index.ts").contains("AuthService");
                let app_imports = ws.read("src/app.ts").contains("AuthService");
                let pass = auth_exists && index_exports && app_imports;
                let note = Some(format!(
                    "auth={auth_exists}, index exports={index_exports}, app imports={app_imports}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t3_infer_pattern_and_extend",
            tier: 3,
            probes: "read existing pattern, add matching entry",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write("src/routes/index.ts",
            "import { Router } from 'express';\nimport userRouter from './user';\nimport authRouter from './auth';\nconst router = Router();\nrouter.use('/users', userRouter);\nrouter.use('/auth', authRouter);\nexport default router;\n");
                ws.write("src/routes/user.ts", "import { Router } from 'express';\nconst router = Router();\nrouter.get('/', (req, res) => res.json([]));\nexport default router;\n");
                ws.write("src/routes/auth.ts", "import { Router } from 'express';\nconst router = Router();\nrouter.post('/login', (req, res) => res.json({}));\nexport default router;\n");
                ws.write(
                    "package.json",
                    r#"{"name":"app","dependencies":{"express":"^4"}}"#,
                );
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Add a new products router following the same pattern as the existing routes, mount it at /products in the index");
                ws.apply(&r);
                let turns = log.count();
                let products_exists = ws.exists("src/routes/products.ts");
                let index = ws.read("src/routes/index.ts");
                let mounted = index.contains("products") && index.contains("/products");
                let pass = products_exists && mounted;
                let note = Some(format!(
                    "products.ts={products_exists}, mounted in index={mounted}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        // ════════════════════════════════════════════════════════
        // TIER 4 — Stretch: large context, deep reasoning, edge cases
        // ════════════════════════════════════════════════════════
        Scenario {
            name: "t4_large_file_targeted_edit",
            tier: 4,
            probes: "find right location in 150-line file without reading noise",
            run: Box::new(|p| {
                let ws = Ws::new();
                // 150-line file with a specific function buried in the middle
                let mut lines = Vec::new();
                lines.push("// Auto-generated config module".to_string());
                for i in 0..40 {
                    lines.push(format!("export const PARAM_{i} = {i};"))
                }
                lines.push("".into());
                lines.push("export function getConfig(env: string) {".into());
                lines.push("  if (env === 'prod') return { url: 'https://prod.example.com', timeout: 30000 };".into());
                lines.push("  if (env === 'staging') return { url: 'https://staging.example.com', timeout: 10000 };".into());
                lines.push("  return { url: 'http://localhost:3000', timeout: 5000 };".into());
                lines.push("}".into());
                lines.push("".into());
                for i in 40..100 {
                    lines.push(format!("export const OTHER_{i} = {i};"))
                }
                let src = lines.join("\n");
                ws.write("src/config.ts", &src);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("In src/config.ts, add a 'test' environment to getConfig that returns url: 'http://test.example.com' and timeout: 1000");
                ws.apply(&r);
                let turns = log.count();
                let content = ws.read("src/config.ts");
                let pass = content.contains("test")
                    && content.contains("test.example.com")
                    && content.contains("1000");
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note: if !pass {
                        Some(
                            content
                                .lines()
                                .filter(|l| l.contains("env") || l.contains("return"))
                                .collect::<Vec<_>>()
                                .join("\n"),
                        )
                    } else {
                        None
                    },
                }
            }),
        },
        Scenario {
            name: "t4_generate_test_file",
            tier: 4,
            probes: "read source, generate matching test file",
            run: Box::new(|p| {
                let ws = Ws::new();
                let src = "export function add(a: number, b: number): number { return a + b; }\nexport function multiply(a: number, b: number): number { return a * b; }\nexport function divide(a: number, b: number): number {\n  if (b === 0) throw new Error('division by zero');\n  return a / b;\n}\n";
                ws.write("src/math.ts", src);
                ws.write(
                    "package.json",
                    r#"{"name":"app","devDependencies":{"jest":"^29","ts-jest":"^29"}}"#,
                );
                let (mut s, log) = session(p, &ws, &[("src/math.ts", src)]);
                let r = s.run("Generate a Jest test file at src/math.test.ts covering add, multiply, and the division by zero case");
                ws.apply(&r);
                let turns = log.count();
                let content = ws.read("src/math.test.ts");
                let has_add = content.contains("add");
                let has_multiply = content.contains("multiply");
                let has_div_zero = content.contains("zero")
                    || content.contains("throw")
                    || content.contains("toThrow");
                let pass = ws.exists("src/math.test.ts") && has_add && has_multiply && has_div_zero;
                let note = Some(format!(
                    "add={has_add}, multiply={has_multiply}, div_zero={has_div_zero}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t4_refactor_class_to_functions",
            tier: 4,
            probes: "rewrite a class as named exports, preserve behavior",
            run: Box::new(|p| {
                let ws = Ws::new();
                let src = "export class MathUtils {\n  static add(a: number, b: number): number { return a + b; }\n  static subtract(a: number, b: number): number { return a - b; }\n  static multiply(a: number, b: number): number { return a * b; }\n}\n";
                ws.write("src/math.ts", src);
                let consumer =
                    "import { MathUtils } from './math';\nconsole.log(MathUtils.add(1, 2));\n";
                ws.write("src/main.ts", consumer);
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) =
                    session(p, &ws, &[("src/math.ts", src), ("src/main.ts", consumer)]);
                let r = s.run("Refactor src/math.ts to export add, subtract, multiply as standalone functions instead of a class, and update src/main.ts to use the new named imports");
                ws.apply(&r);
                let turns = log.count();
                let math = ws.read("src/math.ts");
                let main = ws.read("src/main.ts");
                let no_class = !math.contains("class MathUtils");
                let has_exports = math.contains("export function");
                let main_updated = !main.contains("MathUtils");
                let pass = no_class && has_exports && main_updated;
                let note = Some(format!(
                    "no_class={no_class}, has_exports={has_exports}, main_updated={main_updated}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t4_extract_shared_util",
            tier: 4,
            probes: "same helper duplicated in 3 files — extract to shared module, update all callers",
            run: Box::new(|p| {
                let ws = Ws::new();
                // Same formatDate helper copy-pasted in 3 files
                let helper = "function formatDate(d: Date): string { return d.toISOString().split('T')[0]; }";
                ws.write("src/users.ts",   &format!("{helper}\nexport function userLabel(name: string, created: Date) {{ return `${{name}} (${{formatDate(created)}})`; }}\n"));
                ws.write("src/posts.ts",   &format!("{helper}\nexport function postLabel(title: string, created: Date) {{ return `${{title}} - ${{formatDate(created)}}`; }}\n"));
                ws.write("src/reports.ts", &format!("{helper}\nexport function reportLabel(id: string, date: Date) {{ return `Report ${{id}} ${{formatDate(date)}}`; }}\n"));
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("The formatDate helper is duplicated in src/users.ts, src/posts.ts, and src/reports.ts. Extract it to src/utils/date.ts as an exported function, remove the local copies, and import it in all three files.");
                ws.apply(&r);
                let turns = log.count();
                let util_exists = ws.exists("src/utils/date.ts")
                    && ws.read("src/utils/date.ts").contains("formatDate");
                let users_clean = !ws.read("src/users.ts").contains("function formatDate")
                    && ws.read("src/users.ts").contains("formatDate");
                let posts_clean = !ws.read("src/posts.ts").contains("function formatDate")
                    && ws.read("src/posts.ts").contains("formatDate");
                let reports_clean = !ws.read("src/reports.ts").contains("function formatDate")
                    && ws.read("src/reports.ts").contains("formatDate");
                let pass = util_exists && users_clean && posts_clean && reports_clean;
                let note = Some(format!(
                    "util={util_exists}, users_clean={users_clean}, posts_clean={posts_clean}, reports_clean={reports_clean}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t4_convert_require_to_import",
            tier: 4,
            probes: "selectively convert require() to ES import — leave non-targeted requires alone",
            run: Box::new(|p| {
                let ws = Ws::new();
                let src = "\
const path = require('path');\n\
const fs = require('fs');\n\
const lodash = require('lodash');\n\
\n\
export function readFile(p: string) { return fs.readFileSync(path.join(p)); }\n\
export function chunk<T>(arr: T[], size: number) { return lodash.chunk(arr, size); }\n";
                ws.write("src/utils.ts", src);
                ws.write(
                    "package.json",
                    r#"{"name":"app","dependencies":{"lodash":"^4"}}"#,
                );
                let (mut s, log) = session(p, &ws, &[("src/utils.ts", src)]);
                let r = s.run("Convert the lodash require() to an ES import in src/utils.ts. Leave path and fs as require() since they are Node built-ins.");
                ws.apply(&r);
                let turns = log.count();
                let content = ws.read("src/utils.ts");
                let lodash_import = content.contains("import") && content.contains("lodash");
                let lodash_require_gone = !content.contains("require('lodash')");
                let path_kept = content.contains("require('path')");
                let fs_kept = content.contains("require('fs')");
                let pass = lodash_import && lodash_require_gone && path_kept && fs_kept;
                let note = Some(format!(
                    "lodash_import={lodash_import}, path_kept={path_kept}, fs_kept={fs_kept}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t4_dependency_upgrade_cascade",
            tier: 4,
            probes: "update dep version + fix breaking API usage in source",
            run: Box::new(|p| {
                let ws = Ws::new();
                // axios 0.x → 1.x: axios.get returns different type, error handling changed
                let pkg = r#"{"name":"app","dependencies":{"axios":"^0.27.0"}}"#;
                ws.write("package.json", pkg);
                let src = "import axios from 'axios';\nexport async function fetchUser(id: string) {\n  const response = await axios.get(`/users/${id}`);\n  return response.data;\n}\n";
                ws.write("src/api.ts", src);
                let (mut s, log) = session(p, &ws, &[("package.json", pkg), ("src/api.ts", src)]);
                let r = s.run("Upgrade axios to ^1.6.0 in package.json. The API is compatible but also add try/catch error handling to fetchUser that throws a new Error with the axios error message");
                ws.apply(&r);
                let turns = log.count();
                let new_pkg = ws.read_json("package.json");
                let version_ok = new_pkg["dependencies"]["axios"]
                    .as_str()
                    .is_some_and(|v| v.contains("1."));
                let api = ws.read("src/api.ts");
                let has_try_catch = api.contains("try") && api.contains("catch");
                let pass = version_ok && has_try_catch;
                let note = Some(format!(
                    "version_ok={version_ok}, try_catch={has_try_catch}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        // ════════════════════════════════════════════════════════
        // TIER 5 — Advanced: cascade changes, structural refactors
        // ════════════════════════════════════════════════════════
        Scenario {
            name: "t5_api_signature_cascade",
            tier: 5,
            probes: "add required param to a function, cascade update to 3 callers",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write("src/db.ts", "export function query(sql: string): Promise<any[]> {\n  return Promise.resolve([]);\n}\n");
                ws.write("src/users.ts",    "import { query } from './db';\nexport async function getUsers() { return query('SELECT * FROM users'); }\n");
                ws.write("src/posts.ts",    "import { query } from './db';\nexport async function getPosts() { return query('SELECT * FROM posts'); }\n");
                ws.write("src/comments.ts", "import { query } from './db';\nexport async function getComments() { return query('SELECT * FROM comments'); }\n");
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Add a required second parameter `tenantId: string` to the `query` function in src/db.ts. Update all three callers (users.ts, posts.ts, comments.ts) to pass 'default' as the tenantId.");
                ws.apply(&r);
                let turns = log.count();
                let db = ws.read("src/db.ts");
                let users = ws.read("src/users.ts");
                let posts = ws.read("src/posts.ts");
                let comments = ws.read("src/comments.ts");
                let db_sig_updated = db.contains("tenantId");
                let users_updated = users.contains("'default'") || users.contains("\"default\"");
                let posts_updated = posts.contains("'default'") || posts.contains("\"default\"");
                let comments_updated =
                    comments.contains("'default'") || comments.contains("\"default\"");
                let pass = db_sig_updated && users_updated && posts_updated && comments_updated;
                let note = Some(format!(
                    "db={db_sig_updated}, users={users_updated}, posts={posts_updated}, comments={comments_updated}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t5_break_circular_dep",
            tier: 5,
            probes: "detect circular import, extract shared types to new file, rewire both modules",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write(
                    "src/a.ts",
                    "\
import { BResult } from './b';\n\
export type AInput = { name: string };\n\
export function processA(input: AInput): string { return input.name.toUpperCase(); }\n\
export function combineAB(a: AInput, b: BResult): string { return `${a.name}-${b.value}`; }\n",
                );
                ws.write(
                    "src/b.ts",
                    "\
import { AInput } from './a';\n\
export type BResult = { value: number };\n\
export function processB(result: BResult): number { return result.value * 2; }\n\
export function buildB(a: AInput): BResult { return { value: a.name.length }; }\n",
                );
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("src/a.ts and src/b.ts have a circular dependency. Extract the shared types AInput and BResult into src/shared/types.ts, then update both files to import those types from '../shared/types' instead of from each other.");
                ws.apply(&r);
                let turns = log.count();
                let types_file = ws.exists("src/shared/types.ts");
                let types_content = ws.read("src/shared/types.ts");
                let types_has_ainput = types_content.contains("AInput");
                let types_has_bresult = types_content.contains("BResult");
                let a_content = ws.read("src/a.ts");
                let b_content = ws.read("src/b.ts");
                let a_no_circular =
                    !a_content.contains("from './b'") && !a_content.contains("from \"./b\"");
                let b_no_circular =
                    !b_content.contains("from './a'") && !b_content.contains("from \"./a\"");
                let pass = types_file
                    && types_has_ainput
                    && types_has_bresult
                    && a_no_circular
                    && b_no_circular;
                let note = Some(format!(
                    "types_file={types_file}, has_types={}, a_no_circular={a_no_circular}, b_no_circular={b_no_circular}",
                    types_has_ainput && types_has_bresult
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t5_migrate_express_to_fastify",
            tier: 5,
            probes: "rewrite Express route handlers to Fastify — different API shape, preserve logic",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write(
                    "src/routes/users.ts",
                    "\
import { Router, Request, Response } from 'express';\n\
const router = Router();\n\
router.get('/users', (req: Request, res: Response) => {\n\
  const limit = parseInt(req.query['limit'] as string) || 10;\n\
  res.status(200).json({ users: [], limit });\n\
});\n\
router.post('/users', (req: Request, res: Response) => {\n\
  const body = req.body as { name: string };\n\
  res.status(201).json({ id: 'new', name: body.name });\n\
});\n\
export default router;\n",
                );
                ws.write(
                    "src/routes/health.ts",
                    "\
import { Router, Request, Response } from 'express';\n\
const router = Router();\n\
router.get('/health', (_req: Request, res: Response) => {\n\
  res.status(200).json({ status: 'ok' });\n\
});\n\
export default router;\n",
                );
                ws.write("package.json", r#"{"name":"app","dependencies":{"express":"^4","fastify":"^4","@fastify/router":"^1"}}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Rewrite src/routes/users.ts and src/routes/health.ts from Express Router to Fastify route plugins. In Fastify: use `export default async function(fastify) { fastify.get('/path', async (request, reply) => { return reply.send({...}); }); }` pattern. Preserve the route paths and response shapes.");
                ws.apply(&r);
                let turns = log.count();
                let users = ws.read("src/routes/users.ts");
                let health = ws.read("src/routes/health.ts");
                let users_fastify = users.contains("fastify") && !users.contains("Router()");
                let health_fastify = health.contains("fastify") && !health.contains("Router()");
                let users_routes_ok = users.contains("/users");
                let health_route_ok = health.contains("/health");
                let pass = users_fastify && health_fastify && users_routes_ok && health_route_ok;
                let note = Some(format!(
                    "users_fastify={users_fastify}, health_fastify={health_fastify}, routes_ok={}",
                    users_routes_ok && health_route_ok
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t5_add_pagination",
            tier: 5,
            probes: "add pagination params to 3 route handlers + shared helper, maintaining consistency",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write("src/utils/pagination.ts", "// placeholder\nexport {};\n");
                ws.write("src/routes/users.ts",   "import { Router } from 'express';\nconst r = Router();\nr.get('/', (_req, res) => { res.json({ users: [] }); });\nexport default r;\n");
                ws.write("src/routes/posts.ts",   "import { Router } from 'express';\nconst r = Router();\nr.get('/', (_req, res) => { res.json({ posts: [] }); });\nexport default r;\n");
                ws.write("src/routes/comments.ts","import { Router } from 'express';\nconst r = Router();\nr.get('/', (_req, res) => { res.json({ comments: [] }); });\nexport default r;\n");
                ws.write(
                    "package.json",
                    r#"{"name":"app","dependencies":{"express":"^4"}}"#,
                );
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Add pagination to all three route handlers (users, posts, comments). Create a parsePagination(query) helper in src/utils/pagination.ts that extracts `page` (default 1) and `limit` (default 20, max 100) from query params. Import and use this helper in each route handler, returning page and limit in the response alongside the data.");
                ws.apply(&r);
                let turns = log.count();
                let pagination = ws.read("src/utils/pagination.ts");
                let users = ws.read("src/routes/users.ts");
                let posts = ws.read("src/routes/posts.ts");
                let comments = ws.read("src/routes/comments.ts");
                let helper_ok = pagination.contains("parsePagination")
                    && pagination.contains("page")
                    && pagination.contains("limit");
                let users_uses = users.contains("parsePagination")
                    || (users.contains("page") && users.contains("limit"));
                let posts_uses = posts.contains("parsePagination")
                    || (posts.contains("page") && posts.contains("limit"));
                let comments_uses = comments.contains("parsePagination")
                    || (comments.contains("page") && comments.contains("limit"));
                let pass = helper_ok && users_uses && posts_uses && comments_uses;
                let note = Some(format!(
                    "helper={helper_ok}, users={users_uses}, posts={posts_uses}, comments={comments_uses}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t5_monorepo_type_dedup",
            tier: 5,
            probes: "identical type defined in two packages — create shared package, update both consumers",
            run: Box::new(|p| {
                let ws = Ws::new();
                let user_type =
                    "export interface User { id: string; name: string; email: string; }\n";
                ws.write("packages/api/src/types.ts", user_type);
                ws.write("packages/web/src/types.ts", user_type);
                ws.write("packages/api/src/handlers.ts",
            "import { User } from './types';\nexport function getUser(id: string): User { return { id, name: 'Test', email: 'a@b.com' }; }\n");
                ws.write("packages/web/src/components.ts",
            "import { User } from './types';\nexport function userName(u: User): string { return u.name; }\n");
                ws.write(
                    "package.json",
                    r#"{"name":"root","workspaces":["packages/*"]}"#,
                );
                ws.write("packages/api/package.json", r#"{"name":"api"}"#);
                ws.write("packages/web/package.json", r#"{"name":"web"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("The User interface is duplicated in packages/api/src/types.ts and packages/web/src/types.ts. Create packages/shared/src/types.ts with the User interface, then update packages/api/src/handlers.ts and packages/web/src/components.ts to import User from '../../shared/src/types' instead of their local types files.");
                ws.apply(&r);
                let turns = log.count();
                let shared_exists = ws.exists("packages/shared/src/types.ts");
                let shared_has_user = ws.read("packages/shared/src/types.ts").contains("User");
                let handlers = ws.read("packages/api/src/handlers.ts");
                let components = ws.read("packages/web/src/components.ts");
                let handlers_updated = handlers.contains("shared");
                let components_updated = components.contains("shared");
                let pass =
                    shared_exists && shared_has_user && handlers_updated && components_updated;
                let note = Some(format!(
                    "shared={shared_exists}, handlers_updated={handlers_updated}, components_updated={components_updated}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        // ════════════════════════════════════════════════════════
        // TIER 6 — Extreme: context exhaustion, iterative feedback,
        //           deep reasoning chains, real-world complexity
        // ════════════════════════════════════════════════════════
        Scenario {
            name: "t6_context_exhaustion_refactor",
            tier: 6,
            probes: "8 files preloaded (~12KB context), cross-cutting interface change across all consumers",
            run: Box::new(|p| {
                let ws = Ws::new();
                // 8 substantial files — floods the context window
                let repo_svc = "export interface Repository<T> {\n  findById(id: string): Promise<T | null>;\n  findAll(opts?: { limit?: number; offset?: number }): Promise<T[]>;\n  save(entity: T): Promise<T>;\n  delete(id: string): Promise<void>;\n}\n\nexport abstract class BaseRepository<T> implements Repository<T> {\n  abstract findById(id: string): Promise<T | null>;\n  abstract findAll(opts?: { limit?: number; offset?: number }): Promise<T[]>;\n  abstract save(entity: T): Promise<T>;\n  abstract delete(id: string): Promise<void>;\n}\n";
                let user_repo = "import { BaseRepository } from './base';\nimport { User } from '../models/user';\n\nexport class UserRepository extends BaseRepository<User> {\n  async findById(id: string) { return null; }\n  async findAll(opts?: { limit?: number; offset?: number }) { return []; }\n  async save(user: User) { return user; }\n  async delete(id: string) { return; }\n}\n";
                let post_repo = "import { BaseRepository } from './base';\nimport { Post } from '../models/post';\n\nexport class PostRepository extends BaseRepository<Post> {\n  async findById(id: string) { return null; }\n  async findAll(opts?: { limit?: number; offset?: number }) { return []; }\n  async save(post: Post) { return post; }\n  async delete(id: string) { return; }\n}\n";
                let comment_repo = "import { BaseRepository } from './base';\nimport { Comment } from '../models/comment';\n\nexport class CommentRepository extends BaseRepository<Comment> {\n  async findById(id: string) { return null; }\n  async findAll(opts?: { limit?: number; offset?: number }) { return []; }\n  async save(comment: Comment) { return comment; }\n  async delete(id: string) { return; }\n}\n";
                let user_model = "export interface User { id: string; email: string; name: string; createdAt: Date; }\n";
                let post_model = "export interface Post { id: string; title: string; body: string; authorId: string; createdAt: Date; }\n";
                let comment_model = "export interface Comment { id: string; body: string; postId: string; authorId: string; createdAt: Date; }\n";
                let user_svc = "import { UserRepository } from '../repositories/user';\nimport { User } from '../models/user';\n\nexport class UserService {\n  constructor(private repo: UserRepository) {}\n  async getUser(id: string): Promise<User | null> { return this.repo.findById(id); }\n  async listUsers(limit = 20, offset = 0): Promise<User[]> { return this.repo.findAll({ limit, offset }); }\n  async createUser(data: Partial<User>): Promise<User> { return this.repo.save(data as User); }\n  async removeUser(id: string): Promise<void> { return this.repo.delete(id); }\n}\n";
                ws.write("src/repositories/base.ts", repo_svc);
                ws.write("src/repositories/user.ts", user_repo);
                ws.write("src/repositories/post.ts", post_repo);
                ws.write("src/repositories/comment.ts", comment_repo);
                ws.write("src/models/user.ts", user_model);
                ws.write("src/models/post.ts", post_model);
                ws.write("src/models/comment.ts", comment_model);
                ws.write("src/services/user.ts", user_svc);
                ws.write("package.json", r#"{"name":"app"}"#);
                let preload = [
                    ("src/repositories/base.ts", repo_svc),
                    ("src/repositories/user.ts", user_repo),
                    ("src/repositories/post.ts", post_repo),
                    ("src/repositories/comment.ts", comment_repo),
                    ("src/services/user.ts", user_svc),
                ];
                let (mut s, log) = session(p, &ws, &preload);
                let r = s.run("Add a `count(filter?: Partial<T>): Promise<number>` method to the Repository interface in src/repositories/base.ts AND implement it (returning 0) in all three concrete repositories (user.ts, post.ts, comment.ts). The BaseRepository abstract class must also declare it as abstract.");
                ws.apply(&r);
                let turns = log.count();
                let base = ws.read("src/repositories/base.ts");
                let user = ws.read("src/repositories/user.ts");
                let post = ws.read("src/repositories/post.ts");
                let comment = ws.read("src/repositories/comment.ts");
                let iface_ok = base.contains("count");
                let user_ok = user.contains("count");
                let post_ok = post.contains("count");
                let comment_ok = comment.contains("count");
                let pass = iface_ok && user_ok && post_ok && comment_ok;
                let note = Some(format!(
                    "interface={iface_ok}, user={user_ok}, post={post_ok}, comment={comment_ok}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t6_iterative_script_fix",
            tier: 6,
            probes: "write shell script → run it → read errors → fix → run again (real feedback loop)",
            run: Box::new(|p| {
                let ws = Ws::new();
                // A broken Node.js script — agent must run it, see the error, fix it, run again
                ws.write(
                    "src/seed.ts",
                    "\
import * as fs from 'fs';\n\
\n\
const users = [\n\
  { id: '1', name: 'Alice', email: 'alice@example.com' },\n\
  { id: '2', name: 'Bob',   email: 'bob@example.com' },\n\
];\n\
\n\
// BUG: writeFileSync arg should be string, not object\n\
fs.writeFileSync('users.json', users);\n\
console.log('Seeded', users.lenght, 'users');  // typo: lenght\n",
                );
                ws.write(
                    "package.json",
                    r#"{"name":"app","devDependencies":{"ts-node":"^10","typescript":"^5"}}"#,
                );
                ws.write("tsconfig.json", r#"{"compilerOptions":{"module":"commonjs","target":"es2020","esModuleInterop":true}}"#);
                let (mut s, log) = session(p, &ws, &[]);
                // deliberately give vague instruction — agent must discover and fix both bugs
                let r = s.run("Run src/seed.ts with ts-node and fix any errors you find until it runs cleanly. The script should write users.json correctly and print the right count.");
                ws.apply(&r);
                let turns = log.count();
                let script = ws.read("src/seed.ts");
                // Both bugs fixed: JSON.stringify present, typo fixed
                let json_fixed = script.contains("JSON.stringify");
                let typo_fixed = !script.contains("lenght") && script.contains("length");
                let pass = json_fixed && typo_fixed;
                let note = Some(format!("json_fixed={json_fixed}, typo_fixed={typo_fixed}"));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t6_full_feature_from_spec",
            tier: 6,
            probes: "implement a feature end-to-end from a spec comment: new model + repo + service + route + validation",
            run: Box::new(|p| {
                let ws = Ws::new();
                // Existing skeleton — agent must implement the full feature
                ws.write(
                    "src/models/user.ts",
                    "export interface User { id: string; email: string; name: string; }\n",
                );
                ws.write(
                    "src/routes/index.ts",
                    "export {}; // register routes here\n",
                );
                ws.write("src/db.ts", "export async function query(sql: string, params: unknown[] = []): Promise<unknown[]> { return []; }\n");
                ws.write(
                    "package.json",
                    r#"{"name":"app","dependencies":{"express":"^4","zod":"^3"}}"#,
                );
                ws.write("src/app.ts",
            "import express from 'express';\nconst app = express();\napp.use(express.json());\nexport default app;\n");
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run(concat!(
            "Implement a complete Tag feature:\n",
            "1. src/models/tag.ts — interface Tag { id: string; name: string; slug: string; }\n",
            "2. src/repositories/tag.ts — TagRepository with findAll(), findById(id), create(name) that generates slug from name (lowercase, spaces→dashes)\n",
            "3. src/routes/tags.ts — Express router GET /tags (list all) and POST /tags (create, validate name is non-empty string with zod)\n",
            "4. Register the tags router in src/routes/index.ts as: export { tagsRouter } from './tags'\n",
        ));
                ws.apply(&r);
                let turns = log.count();
                let model_ok =
                    ws.exists("src/models/tag.ts") && ws.read("src/models/tag.ts").contains("Tag");
                let repo_ok = ws.exists("src/repositories/tag.ts")
                    && ws.read("src/repositories/tag.ts").contains("create");
                let route_ok = ws.exists("src/routes/tags.ts")
                    && ws.read("src/routes/tags.ts").contains("/tags");
                let index_ok = ws.read("src/routes/index.ts").contains("tags");
                let pass = model_ok && repo_ok && route_ok && index_ok;
                let note = Some(format!(
                    "model={model_ok}, repo={repo_ok}, route={route_ok}, index_wired={index_ok}"
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
        Scenario {
            name: "t6_multi_file_rename_with_tests",
            tier: 6,
            probes: "rename + update tests + update imports across 6 files simultaneously",
            run: Box::new(|p| {
                let ws = Ws::new();
                ws.write("src/auth/TokenManager.ts",
            "export class TokenManager {\n  generate(userId: string): string { return `tok_${userId}`; }\n  verify(token: string): string | null { return token.startsWith('tok_') ? token.slice(4) : null; }\n}\n");
                ws.write("src/auth/AuthService.ts",
            "import { TokenManager } from './TokenManager';\nexport class AuthService {\n  private tm = new TokenManager();\n  login(id: string) { return this.tm.generate(id); }\n  verify(tok: string) { return this.tm.verify(tok); }\n}\n");
                ws.write("src/middleware/auth.ts",
            "import { TokenManager } from '../auth/TokenManager';\nconst tm = new TokenManager();\nexport function authMiddleware(token: string) { return tm.verify(token) !== null; }\n");
                ws.write("src/routes/users.ts",
            "import { AuthService } from '../auth/AuthService';\nconst auth = new AuthService();\nexport function getUser(token: string) { return auth.verify(token); }\n");
                ws.write("tests/TokenManager.test.ts",
            "import { TokenManager } from '../src/auth/TokenManager';\ntest('generates token', () => { const tm = new TokenManager(); expect(tm.generate('u1')).toMatch(/tok_/); });\n");
                ws.write("tests/AuthService.test.ts",
            "import { AuthService } from '../src/auth/AuthService';\ntest('login returns token', () => { const auth = new AuthService(); expect(auth.login('u1')).toBeDefined(); });\n");
                ws.write("package.json", r#"{"name":"app"}"#);
                let (mut s, log) = session(p, &ws, &[]);
                let r = s.run("Rename TokenManager to JwtManager everywhere: rename the class in src/auth/TokenManager.ts, rename the file to src/auth/JwtManager.ts, and update all imports and references in AuthService.ts, middleware/auth.ts, routes/users.ts, tests/TokenManager.test.ts (rename to tests/JwtManager.test.ts), and tests/AuthService.test.ts.");
                ws.apply(&r);
                let turns = log.count();
                let old_file_gone = !ws.exists("src/auth/TokenManager.ts");
                let new_file_exists = ws.exists("src/auth/JwtManager.ts");
                let new_class = ws.read("src/auth/JwtManager.ts").contains("JwtManager");
                let auth_svc_updated = !ws.read("src/auth/AuthService.ts").contains("TokenManager");
                let middleware_updated =
                    !ws.read("src/middleware/auth.ts").contains("TokenManager");
                let new_test_exists = ws.exists("tests/JwtManager.test.ts");
                let pass = old_file_gone
                    && new_file_exists
                    && new_class
                    && auth_svc_updated
                    && middleware_updated
                    && new_test_exists;
                let note = Some(format!(
                    "old_gone={old_file_gone}, new_file={new_file_exists}, class_ok={new_class}, auth_svc={auth_svc_updated}, middleware={middleware_updated}, test_renamed={new_test_exists}",
                ));
                Run {
                    outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                    turns,
                    elapsed_s: 0,
                    log: log.turns(),
                    note,
                }
            }),
        },
    ]
}

// ── Harness runner ────────────────────────────────────────────────────────────

fn flush() {
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

fn print_run(name: &str, tier: u8, r: &Run) {
    let symbol = match &r.outcome {
        Outcome::Pass => "✓",
        Outcome::Fail => "✗",
        Outcome::Partial(_) => "~",
    };
    let label = match &r.outcome {
        Outcome::Pass => "PASS",
        Outcome::Fail => "FAIL",
        Outcome::Partial(s) => s,
    };
    let tps = if r.turns > 0 && r.elapsed_s > 0 {
        format!("  {:.0}s/turn", r.elapsed_s as f64 / r.turns as f64)
    } else {
        String::new()
    };
    println!(
        "  T{tier} {symbol} {name:<40} {label:<8} turns={:2}  {}s{tps}",
        r.turns, r.elapsed_s
    );
    if let Some(note) = &r.note {
        for line in note.lines().take(3) {
            println!("      {line}");
        }
    }
    flush();
}

fn run_tier(tier: u8, p: &AnyProvider) -> (usize, usize) {
    let all = scenarios();
    let tier_scenarios: Vec<_> = all.iter().filter(|s| s.tier == tier).collect();
    let mut pass = 0;
    for s in &tier_scenarios {
        print!("  T{tier} … {:<40} ", s.name);
        flush();
        let r = run_scenario(s, p);
        if matches!(r.outcome, Outcome::Pass) {
            pass += 1;
        }
        print!("\r");
        print_run(s.name, tier, &r);
    }
    (pass, tier_scenarios.len())
}

#[test]
#[ignore]
fn run_harness() {
    let p = make_provider();
    let model = std::env::var("FRAME_MODEL").unwrap_or_else(|_| "default".into());

    println!("\n╔══ Exploratory Agent Harness ══════════════════════════════════╗");
    println!("║  {model}");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    flush();

    let mut total_pass = 0;
    let mut total = 0;
    let mut frontier_tier = 0u8;

    for tier in 1u8..=6 {
        let tier_name = match tier {
            1 => "TIER 1 — Fundamentals",
            2 => "TIER 2 — Context & Reasoning",
            3 => "TIER 3 — Multi-file & Structure",
            4 => "TIER 4 — Stretch",
            5 => "TIER 5 — Advanced Reasoning",
            6 => "TIER 6 — Extreme",
            _ => unreachable!(),
        };

        println!("── {tier_name} ──");
        flush();
        let (pass, n) = run_tier(tier, &p);
        println!("   {pass}/{n} passed\n");
        flush();
        total_pass += pass;
        total += n;

        // Track where the frontier is (last tier with any failures)
        if pass < n {
            frontier_tier = tier;
        }
    }

    println!("╔══ Result ═══════════════════════════════════════════════════════╗");
    println!("║  {total_pass}/{total} passed");
    if frontier_tier > 0 {
        let tier_label = match frontier_tier {
            1 => "Tier 1 — Fundamentals",
            2 => "Tier 2 — Context & Reasoning",
            3 => "Tier 3 — Multi-file & Structure",
            4 => "Tier 4 — Stretch",
            5 => "Tier 5 — Advanced Reasoning",
            6 => "Tier 6 — Extreme",
            _ => "Unknown",
        };
        println!("║  Frontier: {tier_label}");
    } else {
        println!("║  All tiers passed — time to add harder scenarios!");
    }
    println!("╚════════════════════════════════════════════════════════════════╝");
}

#[test]
#[ignore]
fn tier_1() {
    let p = make_provider();
    run_tier(1, &p);
}

#[test]
#[ignore]
fn tier_2() {
    let p = make_provider();
    run_tier(2, &p);
}

#[test]
#[ignore]
fn tier_3() {
    let p = make_provider();
    run_tier(3, &p);
}

#[test]
#[ignore]
fn tier_4() {
    let p = make_provider();
    run_tier(4, &p);
}

#[test]
#[ignore]
fn tier_5() {
    let p = make_provider();
    run_tier(5, &p);
}

#[test]
#[ignore]
fn tier_6() {
    let p = make_provider();
    run_tier(6, &p);
}

/// Run a comma-separated list of scenarios: FRAME_SCENARIOS=t3_rename_across_files,t2_multi_step_read_write cargo test ... run_named
#[test]
#[ignore]
fn run_named() {
    let names_raw = std::env::var("FRAME_SCENARIOS").expect("Set FRAME_SCENARIOS=name1,name2,...");
    let names: Vec<&str> = names_raw.split(',').map(str::trim).collect();
    let p = make_provider();
    let all = scenarios();
    let model = std::env::var("FRAME_MODEL").unwrap_or_else(|_| "default".into());
    println!("\n╔══ Targeted Harness ═══════════════════════════════════════════╗");
    println!("║  {model}");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    let mut pass = 0;
    for name in &names {
        let s = all.iter().find(|s| s.name == *name).unwrap_or_else(|| {
            eprintln!("Unknown scenario: {name}");
            std::process::exit(1);
        });
        print!("  T{} … {:<40} ", s.tier, s.name);
        flush();
        let r = run_scenario(s, &p);
        if matches!(r.outcome, Outcome::Pass) {
            pass += 1;
        }
        print!("\r");
        print_run(s.name, s.tier, &r);
        println!("  Tool trace:");
        for t in &r.log {
            let ok = if t.ok { "✓" } else { "✗" };
            println!(
                "    {ok} {}: {}",
                t.tool,
                &t.detail[..t.detail.len().min(300)]
            );
        }
    }
    println!("\n  {pass}/{} passed", names.len());
}

/// Run one scenario by name: cargo test ... single -- t3_rename_across_files
#[test]
#[ignore]
fn single() {
    let name = std::env::args()
        .skip_while(|a| a != "--")
        .nth(1)
        .expect("Usage: single -- <scenario_name>");
    let p = make_provider();
    let all = scenarios();
    let s = all.iter().find(|s| s.name == name).unwrap_or_else(|| {
        let names: Vec<_> = all.iter().map(|s| s.name).collect();
        eprintln!("Unknown: {name:?}\nAvailable: {names:#?}");
        std::process::exit(1);
    });

    println!(
        "Running: {} (tier {})\nProbes: {}\n",
        s.name, s.tier, s.probes
    );
    let r = run_scenario(s, &p);
    print_run(s.name, s.tier, &r);
    println!("\nTool trace:");
    for t in &r.log {
        let ok = if t.ok { "✓" } else { "✗" };
        println!(
            "  {ok} {}: {}",
            t.tool,
            &t.detail[..t.detail.len().min(200)]
        );
    }
}
