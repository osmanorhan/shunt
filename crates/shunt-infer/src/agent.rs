//! `AgentSession` — single tool-driven agent loop that replaces the staged
//! Observe → Clarify → Understand → Localize → Propose → Apply pipeline.
//!
//! The LLM is treated as a black box: it receives a workspace-aware system
//! prompt and a flat tool schema, then calls tools in whatever order makes
//! sense.  The session dispatches each tool call, feeds the result back via
//! the user prompt, and loops until `done` or `ask_user` is called.
//!
//! Key design decisions:
//! - Simulated multi-turn: the user prompt is rebuilt each turn with the full
//!   file state + action log.  This is reliable for local models that don't
//!   handle native multi-turn conversation well.
//! - `ask_user` pauses the loop and returns `NeedsClarification`.  The caller
//!   serialises the session state, surfaces the question to the user, and
//!   calls `AgentSession::resume` with the answer.
//! - `sub_agent` spawns a depth-1 session (no ask_user / sub_agent) for
//!   focused research or editing.  Its file edits and ops are merged back
//!   into the parent session after it completes.
//! - `search_files` uses the shunt-localize search index when available; falls back
//!   to a simple directory walk otherwise.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{ProposedCommand, ProposedFileOp, SourceFileContext, ToolProvider, ToolSpec};
use shunt_core::safety;

// ── System prompt ──────────────────────────────────────────────────────────────
// Migrated from: clarify.system.txt + understand.system.txt; proposal/editing
// now lives in this single agent prompt.

pub(crate) const AGENT_SYSTEM_PROMPT_BASE: &str = "\
You are a senior software engineer acting as a coding agent.
Each response is ONE JSON action — the most useful next step.

PRINCIPLES:
- Design before code. Use think to understand the problem and form a plan before any edit.
- Write as little code as possible. Every line is a liability — solve with the least that makes sense.
- Smallest change that solves the real problem, not just the surface request.
- Fail fast, never hide errors. Surface problems immediately; don't swallow or paper over them.
- Patterns and structure over conditionals. Reach for data structures and composition, not branching.
- Open for extension. Design so new behaviour can be added without modifying existing code.
- Code for others. Names state intent. Units are small and single-responsibility.
- Know the trade-off. When breaking a principle, be explicit in think about why.

WORKFLOW:
- Bias to action. The moment you have read the file containing the code to change, EDIT it with replace_lines (give the line numbers shown in the file). Do not keep searching or reading for related symbols, definitions, or callers you don't strictly need — the file in front of you is enough to make the change.
- A successful replace_lines HAS changed the file. The result IS the file. Call done immediately after — never re-read, re-search, or repeat the same replace_lines to 'verify'. There is nothing to verify; the file view in the response is authoritative.
- Spend think on HOW to make the edit, not on what else to look for. You have at most 1 think — don't burn turns exploring.
- Create new files BEFORE updating registrations or imports that reference them.
- After all edits look correct in the file view: call done immediately. Do NOT run commands just to confirm the edit landed — the file view IS the confirmation.
- Only run a build command if (a) you can see a Cargo.toml or package.json in the workspace AND (b) the task involves code that could have type or syntax errors not visible from reading the file.
  Node/TypeScript → command {\"command_action\":\"run\",\"command_line\":\"pnpm build\"}
  Rust            → command {\"command_action\":\"run\",\"command_line\":\"cargo check\"}
- Do NOT add new dependencies or edit manifests unless the task explicitly asks for a dependency/manifest change or the requested behavior cannot reasonably be implemented with existing code and standard libraries.
- After changing dependency manifests or lockfiles, call command {\"command_action\":\"install_dependencies\"} so the runtime can run the correct install step after apply.
- For long-running dev servers or daemons, use command actions start_service/status_service/stop_service. Do NOT wrap servers in timeout or background shell syntax.
- If a build command fails once for any reason (tool not found, missing config, no manifest), call done immediately — do NOT retry the same command.
- If build fails with a real compile error: think about root cause, replace_lines to fix, re-run build once.
- If you see a conflict or a simpler approach, surface it with ask_user before proceeding.
- write_file is for NEW files only — it errors if the file already exists.
- replace_lines is the single tool for ALL edits to existing files:
    • change lines: start_line and end_line within the file
    • append: set start_line to (last line + 1) — e.g. a 4-line file → start_line=5
    • delete lines: use replace_lines on the range; output nothing in the content step
- Preserve all export keywords when refactoring TypeScript.

Available tools:

• think — Record your reasoning before acting. No side effects.
  Max 1 think call per session. After that, act directly.
  Required: query (string — your plan, concern, or rationale)

• write_file — Create a brand-new file (content collected separately). Errors if the file already exists.
  Required: path (string)

• replace_lines — The ONLY tool for editing an existing file. Handles replace, append, and delete.
  Read the file first — loaded files show line numbers.
  Required: path, start_line, end_line (1-indexed, inclusive).
  You give ONLY the line range — replacement code is generated separately.
  • Replace: start_line/end_line within file. To change one line: start_line = end_line.
  • Append: set start_line = last_line + 1 (e.g. 4-line file → start_line=5, end_line=5).
  • Delete: use the line range; output nothing in the content step.

• delete_file — Remove a file.
  Required: path (string)

• read_file — Load a file into context.
  Required: path (string)

• search_files — Search file contents and paths by keyword or symbol. Empty query = list all.
  Required: query (string)

• command — Run commands, queue dependency installs, and manage services.
  Required: command_action = run, install_dependencies, start_service, status_service, or stop_service.
  run: command_line (string). Parsed into argv; not run through a shell.
  install_dependencies: optional path (omit it or use '.' / a root manifest path).
  start_service: service_name and command_line.
  status_service/stop_service: service_name.
  Examples: {\"tool\":\"command\",\"command_action\":\"run\",\"command_line\":\"cargo check\"}
            {\"tool\":\"command\",\"command_action\":\"install_dependencies\"}
            {\"tool\":\"command\",\"command_action\":\"start_service\",\"service_name\":\"api\",\"command_line\":\"python3 app.py\"}

• ask_user — Ask the user when genuinely blocked or when surfacing a better approach.
  Required: question (string), context (string)

• sub_agent — Spawn a focused sub-session for a bounded task.
  Required: task (string), context (string)

• done — Mark work complete.
  Required: description (string — what changed and what was verified)
  Optional: setup_commands (array of {program, args} objects)";

// ── Tool dispatch schema ───────────────────────────────────────────────────────

fn agent_schema(depth: u8, allow_think: bool, allow_search: bool) -> Value {
    let mut tools: Vec<&str> = vec![
        "write_file",
        "replace_lines",
        "delete_file",
        "read_file",
        "command",
        "done",
    ];
    if allow_think {
        tools.insert(0, "think");
    }
    if allow_search {
        tools.push("search_files");
    }
    if depth == 0 {
        tools.push("ask_user");
        tools.push("sub_agent");
    }
    // Content fields (contents, old_str, new_str, patch, new_content) are intentionally
    // excluded from the JSON schema. Local models refuse to generate large content in
    // grammar-constrained JSON. Content is collected via a separate plain-text call.
    // Note: maxLength / maxItems are omitted intentionally.
    // llama.cpp grammar FSM is O(n) in maxLength — adding them to 8+ fields creates
    // a grammar so large it takes minutes to compile before the first token is generated.
    let mut schema = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "tool":        { "type": "string", "enum": tools },
            "path":        { "type": "string" },
            // replace_lines positioning — integers only (content is collected
            // separately via generate_text; small models can't produce content
            // inside grammar-constrained JSON).
            "start_line":  { "type": "integer" },
            "end_line":    { "type": "integer" },
            "query":       { "type": "string" },
            "command_action": { "type": "string", "enum": ["run", "install_dependencies", "start_service", "status_service", "stop_service"] },
            "cmd":         { "type": "string" },
            "command_line": { "type": "string" },
            "service_name": { "type": "string" },
            "args":        { "type": "array", "items": { "type": "string" } },
            "question":    { "type": "string" },
            "context":     { "type": "string" },
            "task":        { "type": "string" },
            "description": { "type": "string" },
            "setup_commands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "program": { "type": "string" },
                        "args":    { "type": "array", "items": { "type": "string" } }
                    },
                    "required": ["program", "args"]
                }
            }
        },
        "required": ["tool"]
    });
    if !allow_think
        && !allow_search
        && let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut)
    {
        properties.remove("query");
    }
    schema
}

// Verifier is read-only: write/str_replace/delete are excluded from the grammar
// so the model physically cannot generate them even if it ignores the prompt.
fn verifier_schema(allow_think: bool) -> Value {
    let mut tools = vec!["read_file", "search_files", "command", "done"];
    if allow_think {
        tools.insert(0, "think");
    }
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "tool":        { "type": "string", "enum": tools },
            "path":        { "type": "string" },
            "query":       { "type": "string" },
            "command_action": { "type": "string", "enum": ["run", "start_service", "status_service", "stop_service"] },
            "cmd":         { "type": "string" },
            "command_line": { "type": "string" },
            "service_name": { "type": "string" },
            "args":        { "type": "array", "items": { "type": "string" } },
            "description": { "type": "string" }
        },
        "required": ["tool"]
    })
}

// ── Deserialized action ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct AgentAction {
    tool: String,
    path: Option<String>,
    start_line: Option<usize>,
    end_line: Option<usize>,
    patch: Option<String>,
    old_str: Option<String>,
    new_str: Option<String>,
    query: Option<String>,
    command_action: Option<String>,
    cmd: Option<String>,
    command_line: Option<String>,
    service_name: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    question: Option<String>,
    context: Option<String>,
    task: Option<String>,
    description: Option<String>,
    #[serde(default)]
    setup_commands: Vec<ProposedCommand>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AgentActionEnvelope {
    Direct(AgentAction),
    Action { action: AgentAction },
    Actions { actions: Vec<AgentAction> },
}

impl AgentActionEnvelope {
    fn into_action(self) -> AgentAction {
        match self {
            Self::Direct(action) | Self::Action { action } => action,
            Self::Actions { mut actions } => {
                actions.drain(..).next().unwrap_or_else(|| AgentAction {
                    tool: "think".into(),
                    path: None,
                    start_line: None,
                    end_line: None,
                    patch: None,
                    old_str: None,
                    new_str: None,
                    query: None,
                    command_action: None,
                    cmd: None,
                    command_line: None,
                    service_name: None,
                    args: vec![],
                    question: None,
                    context: Some(
                        "model returned an empty actions array; choose one action".into(),
                    ),
                    task: None,
                    description: None,
                    setup_commands: vec![],
                })
            }
        }
    }
}

// ── Public types ──────────────────────────────────────────────────────────────

/// One recorded agent turn: what tool was called and what the result was.
/// Serialisable so sessions can be saved and resumed after an `ask_user` pause.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentTurn {
    pub tool: String,
    pub result: String,
    pub ok: bool,
}

/// Content archived from hot context to cold storage when the token budget is
/// exceeded.  Compressed summary stays in hot context; full original is here
/// for potential recall.
#[derive(Debug, Clone)]
#[allow(dead_code)] // fields read by future recall mechanism
struct ColdEntry {
    turn_idx: usize,
    tool: String,
    original: String,
    compressed_to: String,
}

/// The outcome of running an agent session.
#[derive(Debug)]
pub enum AgentResult {
    /// All work is done.
    Done {
        ops: Vec<ProposedFileOp>,
        setup_commands: Vec<ProposedCommand>,
        description: String,
        /// Final file state — used to warm a follow-up fix session without re-reading from disk.
        file_state: HashMap<String, String>,
    },
    /// The agent needs the user to answer a question before it can continue.
    /// Save `turns`, `file_state`, and `partial_ops`, surface `question`, then call
    /// `AgentSession::resume` with the user's answer.
    NeedsClarification {
        question: String,
        context: String,
        turns: Vec<AgentTurn>,
        file_state: HashMap<String, String>,
        /// Ops already applied before the question — carry them so callers can
        /// surface partial work even if the session doesn't resume.
        partial_ops: Vec<ProposedFileOp>,
        /// Deferred setup commands queued before the question.
        queued_setup_commands: Vec<ProposedCommand>,
    },
    /// Hit the turn limit without calling `done`.
    MaxTurnsReached,
}

/// Notification callback for live progress display (e.g. TUI updates).
pub trait AgentObserver: Send + Sync {
    fn on_tool_call(&self, turn: usize, max_turns: usize, tool: &str, summary: &str);
    fn on_tool_result(&self, turn: usize, ok: bool, detail: &str);
    /// Emit a plain-text note (e.g. "Generating content for 'path'...").
    fn on_note(&self, _text: &str) {}
}

// ── Dispatch result (internal) ────────────────────────────────────────────────

enum Dispatch {
    Continue {
        result: String,
        ok: bool,
    },
    Done {
        description: String,
        setup_commands: Vec<ProposedCommand>,
    },
    NeedsClarification {
        question: String,
        context: String,
    },
}

// ── SessionBudget ───────────────────────────────────────────────────────────────

/// Turn budget and stall-detection thresholds for an agent session.
///
/// Derive from model capabilities with `SessionBudget::for_model(max_tokens)`, then
/// layer project overrides on top with `apply_override`.  All fields are public so
/// callers can construct custom budgets for testing or special sessions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionBudget {
    /// Hard turn ceiling — session aborts (returns MaxTurnsReached) after this many
    /// turns regardless of progress.
    pub max_turns: usize,
    /// Consecutive idle turns (think / read_file / search_files, no edits or commands)
    /// at which a STALL WARNING is injected into the next user prompt.
    pub stall_warn_at: usize,
    /// Consecutive idle turns at which the session is aborted.  Must be > stall_warn_at.
    pub stall_abort_at: usize,
}

impl Default for SessionBudget {
    fn default() -> Self {
        // Generous baseline for capable models (≥16K token budget).
        Self {
            max_turns: 20,
            stall_warn_at: 4,
            stall_abort_at: 7,
        }
    }
}

impl SessionBudget {
    /// Derive a reasonable budget from the model's token ceiling.
    ///
    /// Smaller models produce shorter reasoning chains and are more likely to drift
    /// into idle loops, so we give them tighter budgets and earlier warnings.
    pub fn for_model(max_tokens: u32) -> Self {
        if max_tokens <= 8192 {
            // Small / constrained local model
            Self {
                max_turns: 12,
                stall_warn_at: 3,
                stall_abort_at: 5,
            }
        } else if max_tokens <= 16384 {
            // Mid-range local model
            Self {
                max_turns: 16,
                stall_warn_at: 4,
                stall_abort_at: 6,
            }
        } else {
            Self::default()
        }
    }

    /// Convenience: budget for verifier sessions (read-only, shorter).
    pub fn for_verifier() -> Self {
        Self {
            max_turns: 12,
            stall_warn_at: 3,
            stall_abort_at: 5,
        }
    }

    /// Convenience: budget for sub-agent sessions (focused, no ask_user).
    pub fn for_sub_agent() -> Self {
        Self {
            max_turns: 10,
            stall_warn_at: 3,
            stall_abort_at: 5,
        }
    }

    /// Apply optional per-field project overrides.  `None` fields are left unchanged.
    pub fn apply_override(&mut self, o: &SessionBudgetOverride) {
        if let Some(v) = o.max_turns {
            self.max_turns = v;
        }
        if let Some(v) = o.stall_warn_at {
            self.stall_warn_at = v;
        }
        if let Some(v) = o.stall_abort_at {
            self.stall_abort_at = v;
        }
    }
}

/// Optional per-project overrides that layer on top of the model-derived budget.
/// Stored in `.shunt/config.toml` under `[agent]`.  Any `None` field keeps the
/// model-derived value.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionBudgetOverride {
    pub max_turns: Option<usize>,
    pub stall_warn_at: Option<usize>,
    pub stall_abort_at: Option<usize>,
}

impl SessionBudgetOverride {
    pub fn is_empty(&self) -> bool {
        self.max_turns.is_none() && self.stall_warn_at.is_none() && self.stall_abort_at.is_none()
    }
}

// ── AgentSession ──────────────────────────────────────────────────────────────

const MAX_CMD_OUTPUT: usize = 2048;
const REPAIR_TURN_EXTENSION: usize = 4;

struct BackgroundService {
    command_line: String,
    child: std::process::Child,
    stdout: Arc<Mutex<Vec<u8>>>,
    stderr: Arc<Mutex<Vec<u8>>>,
}

/// Gitignore-style patterns always blocked from `read_file` and file listing.
/// Users can extend (not replace) this list via `AgentSession::with_ignore_patterns`.
pub const DEFAULT_IGNORE_PATTERNS: &[&str] = &[
    "node_modules",
    ".git",
    "target",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".svelte-kit",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "*.lock",
    "*.lockb",
    "*.sum",
    "*.min.js",
    "*.min.css",
    "*.map",
    ".DS_Store",
];

// ── Workspace boundary guard ──────────────────────────────────────────────────

/// Resolve a model-supplied workspace path to an absolute canonical path,
/// rejecting anything that escapes the workspace root.
///
/// Handles:
/// - Absolute paths inside the workspace (accepted and normalized)
/// - `..` components (lexical normalization + canonical check)
/// - Symlinks pointing outside the workspace (canonicalize on existing paths)
/// - New files: canonicalize the deepest existing ancestor, rejoin the rest
///
/// Returns `Err(message)` when the path is disallowed.
fn resolve_in_workspace(workspace_root: &str, rel: &str) -> Result<PathBuf, String> {
    let rel = rel.trim();
    if rel.is_empty() {
        return Err("path is empty — provide the file path. Example: {\"tool\":\"read_file\",\"path\":\"src/main.rs\"}".into());
    }
    let root = Path::new(workspace_root)
        .canonicalize()
        .map_err(|e| format!("workspace root invalid: {e}"))?;

    let supplied = Path::new(rel);
    let raw = if supplied.is_absolute() {
        supplied.to_path_buf()
    } else {
        root.join(supplied)
    };

    // Fast path: the target already exists — canonicalize resolves symlinks too.
    if raw.exists() {
        let canonical = raw
            .canonicalize()
            .map_err(|e| format!("cannot resolve '{rel}': {e}"))?;
        if !canonical.starts_with(&root) {
            return Err(format!(
                "path '{rel}' escapes the workspace (symlink or ..)"
            ));
        }
        return Ok(canonical);
    }

    // Slow path: target doesn't exist yet (new file being created).
    // Walk up from raw until we find an existing ancestor, canonicalize
    // that, then re-append the non-existent tail.
    let mut existing = raw.clone();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        if existing.exists() {
            break;
        }
        match existing.file_name() {
            Some(name) => {
                tail.push(name.to_owned());
                existing.pop();
            }
            None => break, // hit filesystem root
        }
    }
    let canonical_base = existing.canonicalize().unwrap_or(existing);
    tail.reverse();
    let final_path = tail.iter().fold(canonical_base, |p, c| p.join(c));

    if !final_path.starts_with(&root) {
        return Err(format!("path '{rel}' escapes the workspace"));
    }
    Ok(final_path)
}

pub struct AgentSession<'a, P> {
    provider: &'a P,
    workspace_root: String,
    system_prompt: String,
    turns: Vec<AgentTurn>,
    /// Live file contents — updated as edits are applied.
    file_state: HashMap<String, String>,
    /// Accumulated file ops for the final `ChangeSet`.
    ops: Vec<ProposedFileOp>,
    depth: u8,
    /// When true, use verifier_schema() instead of agent_schema() — write/edit/delete tools
    /// are excluded from the grammar so the model physically cannot generate them.
    is_verifier: bool,
    budget: SessionBudget,
    wall_timeout: Option<Duration>,
    observer: Option<Arc<dyn AgentObserver>>,
    /// Original task — stored so dispatch can use it in two-phase generation.
    current_task: String,
    queued_setup_commands: Vec<ProposedCommand>,
    /// Paths successfully written in this session — guards against infinite write loops.
    written_paths: HashSet<String>,
    /// (path, start_line) of ranges already replaced this session — guards against
    /// re-editing the same spot in a loop (the post-edit repetition failure mode).
    /// (path, start_line, new_content) triples already applied — guards against
    /// re-applying the exact same replacement in a loop while allowing corrections.
    edited_lines: HashSet<(String, usize, String)>,
    /// Ranges that already generated no effective change. Repeating them usually
    /// means the model is stuck on the wrong file/range, so block before another
    /// replacement-generation call burns time.
    ineffective_edit_ranges: HashSet<(String, usize, usize)>,
    /// Successful command keys since the last file/setup change. Prevents burning
    /// turns on identical verification commands when there is nothing new to check.
    successful_commands_since_change: HashSet<String>,
    /// Extra gitignore-style patterns added on top of DEFAULT_IGNORE_PATTERNS.
    /// Use `with_ignore_patterns` to extend; these are never used to override defaults.
    extra_ignore_patterns: Vec<String>,
    /// Multi-turn conversation history: (role, content) pairs for past turns.
    /// Excludes the system prompt and task frame — those are the stable cached prefix.
    /// Each turn appends (assistant, action_json) + (user, result_text).
    conv_history: Vec<(String, String)>,
    /// Content evicted from `conv_history` when the token budget was exceeded.
    /// Compressed summaries stay in hot context; originals archived here for recall.
    cold_entries: Vec<ColdEntry>,
    /// Session-local long-running processes started via start_service.
    services: HashMap<String, BackgroundService>,
}

impl<'a, P: ToolProvider> AgentSession<'a, P> {
    /// Create a top-level session.  The system prompt is built from the workspace.
    pub fn new(provider: &'a P, workspace_root: &str) -> Self {
        let system_prompt = build_system_prompt(workspace_root);
        Self {
            provider,
            workspace_root: workspace_root.to_string(),
            system_prompt,
            turns: Vec::new(),
            file_state: HashMap::new(),
            ops: Vec::new(),
            depth: 0,
            is_verifier: false,
            budget: SessionBudget::default(),
            wall_timeout: None,
            observer: None,
            current_task: String::new(),
            queued_setup_commands: Vec::new(),
            written_paths: HashSet::new(),
            edited_lines: HashSet::new(),
            ineffective_edit_ranges: HashSet::new(),
            successful_commands_since_change: HashSet::new(),
            extra_ignore_patterns: Vec::new(),
            conv_history: Vec::new(),
            cold_entries: Vec::new(),
            services: HashMap::new(),
        }
    }

    /// Create a verifier session — QA mindset, read-only, reports PASS/FAIL.
    pub fn new_verifier(provider: &'a P, workspace_root: &str) -> Self {
        let system_prompt = build_verifier_prompt(workspace_root);
        Self {
            provider,
            workspace_root: workspace_root.to_string(),
            system_prompt,
            turns: Vec::new(),
            file_state: HashMap::new(),
            ops: Vec::new(),
            depth: 1,
            is_verifier: true,
            budget: SessionBudget::for_verifier(),
            wall_timeout: None,
            observer: None,
            current_task: String::new(),
            queued_setup_commands: Vec::new(),
            written_paths: HashSet::new(),
            edited_lines: HashSet::new(),
            ineffective_edit_ranges: HashSet::new(),
            successful_commands_since_change: HashSet::new(),
            extra_ignore_patterns: Vec::new(),
            conv_history: Vec::new(),
            cold_entries: Vec::new(),
            services: HashMap::new(),
        }
    }

    fn new_sub(provider: &'a P, workspace_root: &str) -> Self {
        let system_prompt = build_system_prompt(workspace_root);
        Self {
            provider,
            workspace_root: workspace_root.to_string(),
            system_prompt,
            turns: Vec::new(),
            file_state: HashMap::new(),
            ops: Vec::new(),
            depth: 1,
            is_verifier: false,
            budget: SessionBudget::for_sub_agent(),
            wall_timeout: None,
            observer: None,
            current_task: String::new(),
            queued_setup_commands: Vec::new(),
            written_paths: HashSet::new(),
            edited_lines: HashSet::new(),
            ineffective_edit_ranges: HashSet::new(),
            successful_commands_since_change: HashSet::new(),
            extra_ignore_patterns: Vec::new(),
            conv_history: Vec::new(),
            cold_entries: Vec::new(),
            services: HashMap::new(),
        }
    }

    /// Add extra gitignore-style ignore patterns on top of [`DEFAULT_IGNORE_PATTERNS`].
    /// Patterns follow gitignore syntax: `*.log`, `dist/`, `secret.txt`.
    /// Users can call this to extend the defaults; the defaults are always active.
    pub fn with_ignore_patterns(
        mut self,
        patterns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.extra_ignore_patterns
            .extend(patterns.into_iter().map(|p| p.into()));
        self
    }

    pub fn with_budget(mut self, budget: SessionBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn with_wall_timeout(mut self, timeout: Duration) -> Self {
        self.wall_timeout = Some(timeout);
        self
    }

    /// Seed the agent's file context with pre-localised files so it doesn't
    /// have to issue read_file calls for the most likely candidates.
    pub fn with_pre_loaded(mut self, files: &[SourceFileContext]) -> Self {
        for f in files {
            self.file_state.insert(f.path.clone(), f.contents.clone());
        }
        self
    }

    pub fn with_observer(mut self, obs: Arc<dyn AgentObserver>) -> Self {
        self.observer = Some(obs);
        self
    }

    /// Resume a session that was paused at `ask_user`.
    /// The caller provides the saved state and the user's answer.
    pub fn resume(
        provider: &'a P,
        workspace_root: &str,
        mut saved_turns: Vec<AgentTurn>,
        saved_file_state: HashMap<String, String>,
        saved_setup_commands: Vec<ProposedCommand>,
        user_answer: &str,
    ) -> AgentResult {
        // Patch the last turn (the ask_user that triggered the pause) with the answer.
        if let Some(last) = saved_turns.last_mut()
            && last.tool == "ask_user"
        {
            last.result = format!("User answered: {user_answer}");
        }
        let mut session = Self::new(provider, workspace_root);
        // Reconstruct multi-turn conversation history from saved turns.
        // Each turn becomes an (assistant, stub_json) + (user, result) pair.
        let conv_history: Vec<(String, String)> = saved_turns
            .iter()
            .flat_map(|t| {
                let assistant_stub = serde_json::json!({ "tool": t.tool }).to_string();
                [
                    ("assistant".to_string(), assistant_stub),
                    ("user".to_string(), t.result.clone()),
                ]
            })
            .collect();
        session.conv_history = conv_history;
        session.turns = saved_turns;
        session.file_state = saved_file_state;
        session.queued_setup_commands = saved_setup_commands;
        session.run_inner("(continuing after user answered)")
    }

    /// Start (or continue) the agent on `task`.
    pub fn run(&mut self, task: &str) -> AgentResult {
        self.run_inner(task)
    }

    fn run_inner(&mut self, task: &str) -> AgentResult {
        self.current_task = task.to_string();
        let started_at = Instant::now();
        // Stable prefix shared across all turns — forms the KV-cached root.
        let system_msg = crate::ChatMessage {
            role: "system".into(),
            content: self.system_prompt.clone(),
        };
        let task_frame = crate::ChatMessage {
            role: "user".into(),
            content: format!("TASK: {task}"),
        };

        let mut turn_idx = 0usize;
        while turn_idx < self.effective_max_turns() {
            if self.exceeded_wall_timeout(started_at) {
                if let Some(done) = self.try_autocomplete("wall-clock timeout") {
                    return done;
                }
                if let Some(o) = &self.observer {
                    o.on_note("ERROR: agent session wall-clock timeout exceeded");
                }
                self.stop_all_services();
                return AgentResult::MaxTurnsReached;
            }
            let schema = if self.is_verifier {
                verifier_schema(self.allow_think())
            } else {
                agent_schema(self.depth, self.allow_think(), self.allow_search())
            };
            let action_tool = ToolSpec {
                name: "agent_action".into(),
                description: "Generate the required structured output.".into(),
                parameters: schema,
            };
            // Evict old turn history if approaching the token budget.
            evict_history_if_needed(&mut self.conv_history, &mut self.cold_entries, &self.turns);

            // Build the ephemeral continuation message: current FILES state + nudge.
            // This is NOT stored in conv_history — it's rebuilt fresh every turn so
            // the model always sees current line numbers and a correct nudge.
            let continuation = build_continuation_msg(
                task,
                &self.file_state,
                &self.ops,
                &self.turns,
                &self.workspace_root,
                &self.budget,
            );

            // Inference payload = [system, task_frame] + conv_history + continuation.
            // The continuation is merged into the last message if it is already a user
            // message (avoids consecutive same-role messages that some chat templates
            // reject).  On the first turn conv_history is empty, so continuation folds
            // into the task frame.
            let mut inference_messages: Vec<crate::ChatMessage> =
                vec![system_msg.clone(), task_frame.clone()];
            for (role, content) in &self.conv_history {
                inference_messages.push(crate::ChatMessage {
                    role: role.clone(),
                    content: content.clone(),
                });
            }
            // Merge continuation into the last user message or append as new user msg.
            if let Some(last) = inference_messages.last_mut().filter(|m| m.role == "user") {
                last.content.push_str("\n\n");
                last.content.push_str(&continuation);
            } else {
                inference_messages.push(crate::ChatMessage {
                    role: "user".into(),
                    content: continuation,
                });
            }

            // Retry once on transient LLM errors. More retries can consume the entire
            // outer task budget when a local model server accepts a request but stalls.
            let action: AgentAction = {
                let mut last_err = None;
                let mut result = None;
                for attempt in 0..2u8 {
                    match self
                        .provider
                        .call_tool_from_messages(&inference_messages, &action_tool)
                    {
                        Ok(tc) => match serde_json::from_value::<AgentActionEnvelope>(tc.arguments)
                        {
                            Ok(a) => {
                                result = Some(a.into_action());
                                break;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "AgentSession turn {turn_idx} attempt {attempt}: action shape error: {e}"
                                );
                                last_err = Some(crate::InferError::InvalidOutput {
                                    retries: 1,
                                    reason: e.to_string(),
                                });
                                break;
                            }
                        },
                        Err(e) => {
                            tracing::warn!(
                                "AgentSession turn {turn_idx} attempt {attempt}: generate_from_messages failed: {e}"
                            );
                            last_err = Some(e);
                            if last_err.as_ref().is_some_and(is_invalid_action_error) {
                                break;
                            }
                            if self.exceeded_wall_timeout(started_at) {
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_secs(
                                2u64.pow(attempt as u32),
                            ));
                        }
                    }
                }
                match result {
                    Some(a) => a,
                    None => {
                        if let Some(ref err) = last_err
                            && is_invalid_action_error(&err)
                        {
                            let detail = format!(
                                "Error: previous model response was invalid or incomplete ({err}). Return exactly one valid JSON tool action that matches the schema."
                            );
                            tracing::warn!("AgentSession invalid action turn {turn_idx}: {err}");
                            if let Some(o) = &self.observer {
                                o.on_note(&detail);
                            }
                            self.conv_history.push((
                                "assistant".into(),
                                serde_json::json!({ "tool": "invalid_output" }).to_string(),
                            ));
                            self.conv_history.push(("user".into(), detail.clone()));
                            self.turns.push(AgentTurn {
                                tool: "invalid_output".into(),
                                result: detail,
                                ok: false,
                            });
                            turn_idx += 1;
                            continue;
                        }
                        let err_msg =
                            format!("all retries exhausted on turn {turn_idx}: {last_err:?}");
                        tracing::error!("AgentSession {err_msg}");
                        if let Some(o) = &self.observer {
                            o.on_note(&format!("ERROR: {err_msg}"));
                        }
                        self.stop_all_services();
                        return AgentResult::MaxTurnsReached;
                    }
                }
            };

            let tool = action.tool.clone();
            let summary = action_summary(&action);
            let effective_max_turns = self.effective_max_turns();

            // Capture compact JSON of the action BEFORE dispatch (borrows action).
            let action_json = action_to_compact_json(&action);

            let obs = self.observer.clone();
            if let Some(o) = &obs {
                o.on_tool_call(turn_idx, effective_max_turns, &tool, &summary);
            }

            match self.dispatch(action) {
                Dispatch::Done {
                    description,
                    setup_commands,
                } => {
                    if let Some(o) = &obs {
                        o.on_tool_result(turn_idx, true, &description);
                    }
                    let mut all_setup_commands = self.queued_setup_commands.clone();
                    for cmd in setup_commands {
                        if !all_setup_commands.iter().any(|existing| existing == &cmd) {
                            all_setup_commands.push(cmd);
                        }
                    }
                    // Record final turn in history.
                    self.conv_history.push(("assistant".into(), action_json));
                    self.conv_history.push(("user".into(), "done".into()));
                    self.turns.push(AgentTurn {
                        tool,
                        result: description.clone(),
                        ok: true,
                    });
                    self.stop_all_services();
                    return AgentResult::Done {
                        ops: std::mem::take(&mut self.ops),
                        setup_commands: all_setup_commands,
                        description,
                        file_state: self.file_state.clone(),
                    };
                }

                Dispatch::NeedsClarification { question, context } => {
                    if self.depth > 0 {
                        // Sub-agents cannot pause; treat as a note and continue.
                        let note = format!("(would ask: {question})");
                        self.conv_history.push(("assistant".into(), action_json));
                        self.conv_history.push(("user".into(), note.clone()));
                        self.turns.push(AgentTurn {
                            tool,
                            result: note,
                            ok: true,
                        });
                        turn_idx += 1;
                        continue;
                    }
                    if let Some(o) = &obs {
                        o.on_tool_result(turn_idx, true, "paused — waiting for user");
                    }
                    let waiting_msg = format!("Waiting for user: {question}");
                    self.conv_history.push(("assistant".into(), action_json));
                    self.conv_history.push(("user".into(), waiting_msg.clone()));
                    self.turns.push(AgentTurn {
                        tool: "ask_user".into(),
                        result: waiting_msg,
                        ok: true,
                    });
                    self.stop_all_services();
                    return AgentResult::NeedsClarification {
                        question,
                        context,
                        turns: self.turns.clone(),
                        file_state: self.file_state.clone(),
                        partial_ops: self.ops.clone(),
                        queued_setup_commands: self.queued_setup_commands.clone(),
                    };
                }

                Dispatch::Continue { result, ok } => {
                    if let Some(o) = &obs {
                        o.on_tool_result(turn_idx, ok, &result);
                    }
                    // Append turn to conversation history (compact: result only, no FILES).
                    self.conv_history.push(("assistant".into(), action_json));
                    self.conv_history.push(("user".into(), result.clone()));
                    self.turns.push(AgentTurn { tool, result, ok });

                    // Stall detection — idle streak (read/think/search without editing).
                    let idle_streak = self
                        .turns
                        .iter()
                        .rev()
                        .take_while(|t| {
                            matches!(t.tool.as_str(), "think" | "read_file" | "search_files")
                        })
                        .count();
                    if idle_streak > self.budget.stall_abort_at {
                        tracing::warn!(
                            "AgentSession stall: {idle_streak} consecutive idle turns — aborting"
                        );
                        self.stop_all_services();
                        return AgentResult::MaxTurnsReached;
                    }

                    // Failed-action loop detection: if the last 3 turns all failed
                    // on the same tool with ok:false the model is stuck in a loop
                    // (e.g. repeatedly emitting replace_lines with no path field).
                    // Abort immediately — retrying won't help.
                    let failed_streak: Vec<_> = self.turns.iter().rev().take(3).collect();
                    if failed_streak.len() == 3
                        && failed_streak.iter().all(|t| !t.ok)
                        && failed_streak.windows(2).all(|w| w[0].tool == w[1].tool)
                    {
                        let stuck_tool = &failed_streak[0].tool;
                        tracing::warn!(
                            "AgentSession: 3 consecutive failed '{stuck_tool}' calls — aborting error loop"
                        );
                        self.stop_all_services();
                        return AgentResult::MaxTurnsReached;
                    }
                }
            }
            turn_idx += 1;
        }

        if let Some(done) = self.try_autocomplete("turn limit") {
            return done;
        }

        // Turn limit reached without enough evidence to auto-complete.
        self.stop_all_services();
        AgentResult::MaxTurnsReached
    }

    fn try_autocomplete(&mut self, reason: &str) -> Option<AgentResult> {
        let last_turn = self.turns.last()?;
        if !(last_turn.ok
            && matches!(
                last_turn.tool.as_str(),
                "write_file" | "replace_lines" | "delete_file"
            ))
        {
            return None;
        }
        if self.ops.is_empty()
            || !missing_explicit_file_edits(&self.current_task, &self.file_state, &self.ops)
                .is_empty()
        {
            return None;
        }

        if let Some(o) = &self.observer {
            o.on_note(&format!(
                "Auto-completing after {reason}; runtime verification will validate the edits."
            ));
        }
        self.stop_all_services();
        Some(AgentResult::Done {
            ops: std::mem::take(&mut self.ops),
            setup_commands: std::mem::take(&mut self.queued_setup_commands),
            description: format!(
                "Applied edits before {reason}; runtime verification should validate the result."
            ),
            file_state: self.file_state.clone(),
        })
    }

    fn exceeded_wall_timeout(&self, started_at: Instant) -> bool {
        self.wall_timeout
            .map(|limit| started_at.elapsed() >= limit)
            .unwrap_or(false)
    }

    fn effective_max_turns(&self) -> usize {
        if self.is_repair_extension_eligible() {
            self.budget.max_turns + REPAIR_TURN_EXTENSION
        } else {
            self.budget.max_turns
        }
    }

    fn is_repair_extension_eligible(&self) -> bool {
        let wrote_any = self.turns.iter().any(|turn| {
            turn.ok
                && matches!(
                    turn.tool.as_str(),
                    "write_file" | "replace_lines" | "delete_file"
                )
        });
        let ran_any_command = self.turns.iter().any(|turn| {
            matches!(
                turn.tool.as_str(),
                "command" | "run_command" | "run_command_line" | "install_dependencies"
            )
        });
        wrote_any && ran_any_command
    }

    fn allow_think(&self) -> bool {
        !self
            .turns
            .iter()
            .any(|turn| turn.tool == "think" && turn.ok)
            && !self.turns.iter().any(|turn| {
                turn.ok
                    && matches!(
                        turn.tool.as_str(),
                        "write_file" | "replace_lines" | "delete_file"
                    )
            })
    }

    fn allow_search(&self) -> bool {
        !self.turns.iter().any(|turn| {
            turn.ok
                && matches!(
                    turn.tool.as_str(),
                    "write_file" | "replace_lines" | "delete_file"
                )
        })
    }

    fn stop_all_services(&mut self) {
        let names: Vec<String> = self.services.keys().cloned().collect();
        for name in names {
            let _ = self.stop_service(&name);
        }
    }

    fn start_service(&mut self, name: &str, command_line: &str) -> Dispatch {
        let name = name.trim();
        if name.is_empty() {
            return Dispatch::Continue {
                result: "Error: service_name is required. Example: {\"tool\":\"start_service\",\"service_name\":\"api\",\"command_line\":\"python3 app.py\"}".into(),
                ok: false,
            };
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        {
            return Dispatch::Continue {
                result: "Error: service_name may only contain letters, numbers, '-' and '_'."
                    .into(),
                ok: false,
            };
        }
        if self.services.contains_key(name) {
            return Dispatch::Continue {
                result: format!(
                    "Error: service '{name}' already exists. Use status_service or stop_service first."
                ),
                ok: false,
            };
        }

        let command_line = command_line.trim();
        if command_line.is_empty() {
            return Dispatch::Continue {
                result: "Error: command_line is required for start_service.".into(),
                ok: false,
            };
        }
        let argv = match split_command_line(command_line) {
            Ok(argv) => argv,
            Err(e) => {
                return Dispatch::Continue {
                    result: format!("Error parsing command_line: {e}"),
                    ok: false,
                };
            }
        };
        if argv.is_empty() {
            return Dispatch::Continue {
                result: "Error: command_line did not contain a program".into(),
                ok: false,
            };
        }
        if let Err(result) = check_command_safety(&argv[0], &argv[1..]) {
            return result;
        }

        let mut child = match std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .current_dir(&self.workspace_root)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                return Dispatch::Continue {
                    result: format!("Error spawning service '{name}': {e}"),
                    ok: false,
                };
            }
        };

        let stdout = Arc::new(Mutex::new(Vec::new()));
        let stderr = Arc::new(Mutex::new(Vec::new()));
        if let Some(pipe) = child.stdout.take() {
            spawn_log_reader(pipe, stdout.clone());
        }
        if let Some(pipe) = child.stderr.take() {
            spawn_log_reader(pipe, stderr.clone());
        }

        std::thread::sleep(std::time::Duration::from_millis(150));
        match child.try_wait() {
            Ok(Some(status)) => Dispatch::Continue {
                result: format!(
                    "Service '{name}' exited immediately with {status}.\n{}",
                    service_logs(&stdout, &stderr)
                )
                .trim()
                .to_string(),
                ok: false,
            },
            Ok(None) => {
                self.services.insert(
                    name.to_string(),
                    BackgroundService {
                        command_line: command_line.to_string(),
                        child,
                        stdout,
                        stderr,
                    },
                );
                Dispatch::Continue {
                    result: format!("Service '{name}' started: {command_line}"),
                    ok: true,
                }
            }
            Err(e) => Dispatch::Continue {
                result: format!("Error checking service '{name}': {e}"),
                ok: false,
            },
        }
    }

    fn status_service(&mut self, name: &str) -> Dispatch {
        let name = name.trim();
        let Some(service) = self.services.get_mut(name) else {
            return Dispatch::Continue {
                result: format!(
                    "Error: service '{name}' is not running. Active services: {}",
                    self.active_service_names()
                ),
                ok: false,
            };
        };
        let state = match service.child.try_wait() {
            Ok(Some(status)) => format!("exited with {status}"),
            Ok(None) => "running".into(),
            Err(e) => format!("status error: {e}"),
        };
        Dispatch::Continue {
            result: format!(
                "Service '{name}' ({}) is {state}.\n{}",
                service.command_line,
                service_logs(&service.stdout, &service.stderr)
            )
            .trim()
            .to_string(),
            ok: true,
        }
    }

    fn stop_service(&mut self, name: &str) -> Dispatch {
        let name = name.trim();
        let Some(mut service) = self.services.remove(name) else {
            return Dispatch::Continue {
                result: format!(
                    "Error: service '{name}' is not running. Active services: {}",
                    self.active_service_names()
                ),
                ok: false,
            };
        };
        let state = match service.child.try_wait() {
            Ok(Some(status)) => format!("already exited with {status}"),
            Ok(None) => {
                let _ = service.child.kill();
                match service.child.wait() {
                    Ok(status) => format!("stopped with {status}"),
                    Err(e) => format!("stop wait error: {e}"),
                }
            }
            Err(e) => format!("status error before stop: {e}"),
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        Dispatch::Continue {
            result: format!(
                "Service '{name}' ({}) {state}.\n{}",
                service.command_line,
                service_logs(&service.stdout, &service.stderr)
            )
            .trim()
            .to_string(),
            ok: true,
        }
    }

    fn active_service_names(&self) -> String {
        if self.services.is_empty() {
            "none".into()
        } else {
            let mut names: Vec<&str> = self.services.keys().map(String::as_str).collect();
            names.sort();
            names.join(", ")
        }
    }

    fn execute_deduped_command(&mut self, key: String, program: &str, args: &[String]) -> Dispatch {
        if self.successful_commands_since_change.contains(&key) {
            return Dispatch::Continue {
                result: format!(
                    "Error: command already succeeded with no intervening file or setup change: {}. \
                     Do not repeat successful verification commands. Call done if the task is satisfied, \
                     or edit/install something before running it again.",
                    render_command(program, args)
                ),
                ok: false,
            };
        }

        if let Err(e) = self.materialize_session_files() {
            return Dispatch::Continue {
                result: format!("Error preparing workspace for command: {e}"),
                ok: false,
            };
        }

        while let Some(command) = self.queued_setup_commands.first().cloned() {
            let setup_result =
                execute_command(&self.workspace_root, &command.program, &command.args);
            match setup_result {
                Dispatch::Continue { ok: true, .. } => {
                    self.queued_setup_commands.remove(0);
                }
                Dispatch::Continue { result, .. } => {
                    return Dispatch::Continue {
                        result: format!(
                            "Setup command failed before verification: {}\n{result}",
                            render_command(&command.program, &command.args)
                        ),
                        ok: false,
                    };
                }
                other => return other,
            }
        }

        let result = execute_command(&self.workspace_root, program, args);
        if matches!(result, Dispatch::Continue { ok: true, .. }) {
            self.successful_commands_since_change.insert(key);
        }
        result
    }

    fn materialize_session_files(&self) -> Result<(), String> {
        let mut changed = HashSet::new();
        for op in &self.ops {
            match op {
                ProposedFileOp::Create { path, .. } | ProposedFileOp::Edit { path, .. } => {
                    changed.insert(path.as_str());
                }
                ProposedFileOp::Delete { path } => {
                    let abs = resolve_in_workspace(&self.workspace_root, path)?;
                    match std::fs::remove_file(&abs) {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(format!("delete '{path}': {e}")),
                    }
                }
            }
        }

        for path in changed {
            let Some(contents) = self.file_state.get(path) else {
                continue;
            };
            let abs = resolve_in_workspace(&self.workspace_root, path)?;
            if let Some(parent) = abs.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("create parent for '{path}': {e}"))?;
            }
            std::fs::write(&abs, contents).map_err(|e| format!("write '{path}': {e}"))?;
        }

        Ok(())
    }

    fn queue_install_dependencies(&mut self, path: Option<&str>) -> Dispatch {
        let requested = path.unwrap_or(".").trim();
        let install_dir = match self.install_directory_for(requested) {
            Ok(dir) => dir,
            Err(e) => {
                return Dispatch::Continue {
                    result: format!("Error: {e}"),
                    ok: false,
                };
            }
        };
        let command = match self.install_command_for_dir(&install_dir) {
            Ok(cmd) => cmd,
            Err(e) => {
                return Dispatch::Continue {
                    result: format!("Error: {e}"),
                    ok: false,
                };
            }
        };
        if self
            .queued_setup_commands
            .iter()
            .any(|existing| existing == &command)
        {
            return Dispatch::Continue {
                result: format!(
                    "Install already queued for '{}': {} {}",
                    install_dir,
                    command.program,
                    command.args.join(" ")
                )
                .trim()
                .to_string(),
                ok: true,
            };
        }
        self.queued_setup_commands.push(command.clone());
        self.successful_commands_since_change.clear();
        Dispatch::Continue {
            result: format!(
                "Queued dependency install for '{}': {} {}. The runtime will run this after apply.",
                install_dir,
                command.program,
                command.args.join(" ")
            )
            .trim()
            .to_string(),
            ok: true,
        }
    }

    fn install_directory_for(&self, requested: &str) -> Result<String, String> {
        let trimmed = requested.trim();
        if trimmed.is_empty() {
            return Ok(".".into());
        }
        let manifest_names = [
            "package.json",
            "package-lock.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "bun.lock",
            "bun.lockb",
            "pyproject.toml",
            "requirements.txt",
            "Cargo.toml",
            "go.mod",
        ];
        let raw = Path::new(trimmed);
        let candidate = if raw
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| manifest_names.contains(&name))
            .unwrap_or(false)
        {
            match raw.parent() {
                Some(parent) if !parent.as_os_str().is_empty() => parent,
                _ => Path::new("."),
            }
        } else {
            raw
        };
        let canonical = resolve_in_workspace(&self.workspace_root, &candidate.to_string_lossy())?;
        if !canonical.is_dir() {
            return Err(format!(
                "'{}' is not a directory. Pass a workspace directory or manifest path.",
                trimmed
            ));
        }
        let root = Path::new(&self.workspace_root)
            .canonicalize()
            .map_err(|e| format!("workspace root invalid: {e}"))?;
        let rel = canonical.strip_prefix(&root).unwrap_or(&canonical);
        let rel_str = rel.to_string_lossy();
        let final_dir = if rel_str.is_empty() {
            ".".into()
        } else {
            rel_str.into_owned()
        };
        if final_dir != "." {
            return Err(format!(
                "install_dependencies currently supports only the workspace root; '{}' is a subdirectory.",
                final_dir
            ));
        }
        Ok(final_dir)
    }

    fn install_command_for_dir(&self, install_dir: &str) -> Result<ProposedCommand, String> {
        let dir = if install_dir == "." {
            PathBuf::new()
        } else {
            PathBuf::from(install_dir)
        };
        let has = |name: &str| self.session_path_exists(&dir.join(name));

        if has("package.json") {
            let command = if has("pnpm-lock.yaml") || has("pnpm-workspace.yaml") {
                ProposedCommand {
                    program: "pnpm".into(),
                    args: vec!["install".into()],
                }
            } else if has("yarn.lock") {
                ProposedCommand {
                    program: "yarn".into(),
                    args: vec!["install".into()],
                }
            } else if has("bun.lock") || has("bun.lockb") {
                ProposedCommand {
                    program: "bun".into(),
                    args: vec!["install".into()],
                }
            } else {
                ProposedCommand {
                    program: "npm".into(),
                    args: vec!["install".into()],
                }
            };
            return Ok(command);
        }

        if has("requirements.txt") {
            return Ok(ProposedCommand {
                program: "python3".into(),
                args: vec![
                    "-m".into(),
                    "pip".into(),
                    "install".into(),
                    "-r".into(),
                    "requirements.txt".into(),
                ],
            });
        }

        if has("pyproject.toml") {
            return Ok(ProposedCommand {
                program: "python3".into(),
                args: vec![
                    "-m".into(),
                    "pip".into(),
                    "install".into(),
                    "-e".into(),
                    ".".into(),
                ],
            });
        }

        if has("Cargo.toml") {
            return Ok(ProposedCommand {
                program: "cargo".into(),
                args: vec!["fetch".into()],
            });
        }

        if has("go.mod") {
            return Ok(ProposedCommand {
                program: "go".into(),
                args: vec!["mod".into(), "download".into()],
            });
        }

        Err(format!(
            "could not determine install command for '{}'. Expected package.json, requirements.txt, pyproject.toml, Cargo.toml, or go.mod.",
            install_dir
        ))
    }

    fn session_path_exists(&self, rel: &Path) -> bool {
        let rel_str = rel.to_string_lossy().into_owned();
        if self
            .ops
            .iter()
            .any(|op| matches!(op, ProposedFileOp::Delete { path } if path == &rel_str))
        {
            return false;
        }
        if self.file_state.contains_key(&rel_str)
            || self.ops.iter().any(|op| {
                matches!(op,
                    ProposedFileOp::Create { path, .. } | ProposedFileOp::Edit { path, .. }
                    if path == &rel_str)
            })
        {
            return true;
        }
        resolve_in_workspace(&self.workspace_root, &rel_str)
            .map(|abs| abs.exists())
            .unwrap_or(false)
    }

    fn dispatch_command(&mut self, action: &AgentAction) -> Dispatch {
        match action.command_action.as_deref().unwrap_or("") {
            "run" => self.dispatch_run_command(action),
            "install_dependencies" => self.queue_install_dependencies(action.path.as_deref()),
            "start_service" => {
                let name = action.service_name.as_deref().unwrap_or("");
                let line = action.command_line.as_deref().unwrap_or("");
                self.start_service(name, line)
            }
            "status_service" => {
                let name = action.service_name.as_deref().unwrap_or("");
                self.status_service(name)
            }
            "stop_service" => {
                let name = action.service_name.as_deref().unwrap_or("");
                self.stop_service(name)
            }
            "" => Dispatch::Continue {
                result: "Error: command_action is required. Use run, install_dependencies, start_service, status_service, or stop_service.".into(),
                ok: false,
            },
            other => Dispatch::Continue {
                result: format!(
                    "Error: unknown command_action '{other}'. Use run, install_dependencies, start_service, status_service, or stop_service."
                ),
                ok: false,
            },
        }
    }

    fn dispatch_run_command(&mut self, action: &AgentAction) -> Dispatch {
        if let Some(line) = action
            .command_line
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let argv = match split_command_line(line) {
                Ok(argv) => argv,
                Err(e) => {
                    return Dispatch::Continue {
                        result: format!("Error parsing command_line: {e}"),
                        ok: false,
                    };
                }
            };
            if argv.is_empty() {
                return Dispatch::Continue {
                    result: "Error: command_line did not contain a program".into(),
                    ok: false,
                };
            }
            return self.execute_deduped_command(format!("line:{line}"), &argv[0], &argv[1..]);
        }

        let (cmd, tail): (&str, &[String]) =
            if action.cmd.as_deref().unwrap_or("").is_empty() && !action.args.is_empty() {
                (action.args[0].as_str(), &action.args[1..])
            } else {
                (action.cmd.as_deref().unwrap_or(""), &action.args)
            };
        if cmd.is_empty() {
            return Dispatch::Continue {
                result: "Error: command_action=run requires command_line, or cmd plus args. Example: {\"tool\":\"command\",\"command_action\":\"run\",\"command_line\":\"cargo check\"}".into(),
                ok: false,
            };
        }
        self.execute_deduped_command(command_key(cmd, tail), cmd, tail)
    }

    fn dispatch(&mut self, action: AgentAction) -> Dispatch {
        match action.tool.as_str() {
            "think" => {
                let thought = action.query.as_deref().unwrap_or("").trim();
                if thought.is_empty() {
                    return Dispatch::Continue {
                        result: "Error: think requires a non-empty query with your reasoning."
                            .into(),
                        ok: false,
                    };
                }
                // Guard 1: cap total think calls — prevents think-loop burn-through.
                let think_count = self
                    .turns
                    .iter()
                    .filter(|t| t.tool == "think" && t.ok)
                    .count();
                let edited_before = self.turns.iter().any(|t| {
                    t.ok && matches!(
                        t.tool.as_str(),
                        "write_file" | "replace_lines" | "delete_file"
                    )
                });
                if edited_before {
                    return Dispatch::Continue {
                        result: "Error: files have already been edited. Do not think again. Act now: read/edit the next required file, use command if verification is necessary, or done."
                            .into(),
                        ok: false,
                    };
                }
                if think_count >= 3 {
                    return Dispatch::Continue {
                        result: "Error: you have already thought 3 times. \
                                 No more think calls allowed. Act now — \
                                 call read_file, write_file, replace_lines, command, or done."
                            .into(),
                        ok: false,
                    };
                }
                // Guard 2: reject consecutive think — one thought, then act.
                let prev_was_think = self
                    .turns
                    .last()
                    .map(|t| t.tool == "think" && t.ok)
                    .unwrap_or(false);
                if prev_was_think {
                    return Dispatch::Continue {
                        result: "Error: you just thought. Act now — call read_file, \
                                 write_file, replace_lines, command, or done. \
                                 You may think again after your next action."
                            .into(),
                        ok: false,
                    };
                }
                // Return the thought as the result so it appears verbatim in turn history.
                Dispatch::Continue {
                    result: thought.to_string(),
                    ok: true,
                }
            }

            // Surgical edit for small models: the model addresses the edit by
            // LINE NUMBER (from a line-numbered read — which it can do reliably),
            // the replacement CONTENT is collected via generate_text, and the file
            // is reassembled deterministically by shunt-edit. The model never has
            // to reproduce exact existing text (the broken str_replace approach).
            "replace_lines" => {
                let path = action.path.as_deref().unwrap_or("").to_string();
                let start = action.start_line.unwrap_or(0);
                let end = action.end_line.unwrap_or(start).max(start);
                if path.is_empty() {
                    return Dispatch::Continue {
                        result: "Error: path is required for replace_lines. \
                                 Example: {\"tool\":\"replace_lines\",\"path\":\"src/foo.rs\",\"start_line\":1,\"end_line\":1}".into(),
                        ok: false,
                    };
                }
                if start == 0 {
                    return Dispatch::Continue {
                        result: "Error: start_line is required (1-indexed). \
                                 Call read_file first to see line numbers, then use replace_lines with the correct start_line and end_line. \
                                 Example: {\"tool\":\"replace_lines\",\"path\":\"src/foo.rs\",\"start_line\":3,\"end_line\":3}".into(),
                        ok: false,
                    };
                }
                let current = match self.file_state.get(&path) {
                    Some(c) => c.clone(),
                    None => {
                        return Dispatch::Continue {
                            result: format!(
                                "Error: '{path}' is not loaded. Call read_file '{path}' first \
                                 so you can see its line numbers."
                            ),
                            ok: false,
                        };
                    }
                };

                let total_lines = current.lines().count();

                // Append case: start == total_lines + 1 means "insert after last line".
                // The model's natural instinct (start=N+1 on an N-line file) is correct —
                // route it to InsertAfter instead of erroring.
                let is_append = start == total_lines + 1;
                let (start, end) = if path.ends_with(".json") && !is_append {
                    (1, total_lines)
                } else {
                    (start, end)
                };

                // Auto-expand: if the target range ends mid-block (net open braces > 0),
                // extend end to include the matching closing brace. This handles the common
                // case where the model targets only the opening line of an if/for/while block
                // (end_line=0 normalized to start) instead of the full range.
                let end = if !is_append && end <= total_lines {
                    let file_lines: Vec<&str> = current.lines().collect();
                    let scan_end = end.min(file_lines.len());
                    let mut depth: i32 = 0;
                    for line in file_lines.iter().take(scan_end).skip(start - 1) {
                        for ch in line.chars() {
                            match ch {
                                '{' => depth += 1,
                                '}' => depth -= 1,
                                _ => {}
                            }
                        }
                    }
                    if depth > 0 {
                        let mut expanded = end;
                        for (i, line) in file_lines.iter().enumerate().skip(scan_end) {
                            for ch in line.chars() {
                                match ch {
                                    '{' => depth += 1,
                                    '}' => depth -= 1,
                                    _ => {}
                                }
                            }
                            if depth <= 0 {
                                expanded = i + 1; // 1-indexed
                                break;
                            }
                        }
                        expanded
                    } else {
                        end
                    }
                } else {
                    end
                };

                let range_key = (path.clone(), start, end);
                if self.ineffective_edit_ranges.contains(&range_key) {
                    let current_view =
                        shunt_edit::numbered_window(&current, 1, current.lines().count());
                    return Dispatch::Continue {
                        result: format!(
                            "Error: this replace_lines range already produced no file change: \
                             {path}:{start}-{end}. Current file:\n{current_view}\n\
                             Choose a different or broader range, or switch to another file that \
                             is required by the task. Do not repeat this range."
                        ),
                        ok: false,
                    };
                }

                // Show the model exactly the lines it is replacing (with numbers),
                // or for append, show the tail of the file as context.
                let target = if is_append {
                    shunt_edit::numbered_window(
                        &current,
                        total_lines.saturating_sub(2),
                        total_lines,
                    )
                } else {
                    shunt_edit::numbered_window(&current, start, end)
                };
                if !is_append && target.is_empty() {
                    return Dispatch::Continue {
                        result: format!(
                            "Error: lines {start}-{end} are out of range for '{path}' ({total_lines} lines). \
                             To append after the last line use start_line={}.",
                            total_lines + 1
                        ),
                        ok: false,
                    };
                }
                let task = &self.current_task;
                // Show context around target range for indentation/sibling awareness.
                let ctx_lo = start.saturating_sub(3).max(1);
                let ctx_hi = (end + 3).min(total_lines);
                let context_before =
                    shunt_edit::numbered_window(&current, ctx_lo, start.saturating_sub(1));
                let context_after = if is_append {
                    String::new()
                } else {
                    shunt_edit::numbered_window(&current, end + 1, ctx_hi)
                };
                let ctx_section = {
                    let mut s = String::new();
                    if !context_before.is_empty() || !context_after.is_empty() {
                        s.push_str(
                            "Surrounding context (do NOT reproduce these lines in your output):\n",
                        );
                        s.push_str(&context_before);
                        s.push_str(&context_after);
                        s.push('\n');
                    }
                    s
                };
                let (text_system, base_user) = if is_append {
                    let sys = "You output ONLY new source code to append to the file — \
                        no explanation, no markdown fences, no line numbers. \
                        Do NOT reproduce existing file content. Output only the new lines to add.";
                    let base = format!(
                        "Task: {task}\n\nAppend new content to '{path}'. End of file:\n{target}\n\
                         {ctx_section}\
                         Output ONLY the new lines to append (no line numbers, no fences):"
                    );
                    (sys, base)
                } else {
                    let sys = "You output ONLY the replacement source code for the given lines — \
                        no explanation, no markdown fences, no line numbers. \
                        If the task requires DELETING these lines entirely, send a COMPLETELY EMPTY response \
                        (zero characters — not the word 'None', not a space, not a comment, literally nothing). \
                        If the task requires MODIFYING these lines, output the corrected version. \
                        If the task requires ADDING to these lines, output the existing lines plus the addition. \
                        Match the surrounding indentation exactly.";
                    let base = format!(
                        "Task: {task}\n\nReplace lines {start}-{end} of '{path}':\n{target}\n\
                         {ctx_section}\
                         Output ONLY the replacement for lines {start}-{end} (no line numbers, no fences):"
                    );
                    (sys, base)
                };
                if let Some(o) = &self.observer {
                    if is_append {
                        o.on_note(&format!("Generating content to append to {path}…"));
                    } else {
                        o.on_note(&format!(
                            "Generating replacement for {path} lines {start}-{end}…"
                        ));
                    }
                }
                let new_content = match self.provider.generate_text(text_system, &base_user) {
                    Ok(g) => {
                        let s = strip_code_fences(g.trim());
                        // Empty string is valid: means "delete these lines".
                        // Also treat common deletion signals as truly empty output.
                        match s.to_lowercase().as_str() {
                            "none" | "null" | "empty" | "(empty)" | "<empty>" | "(none)" => {
                                "".to_owned()
                            }
                            _ => s,
                        }
                    }
                    Err(e) => {
                        return Dispatch::Continue {
                            result: format!(
                                "Error: replacement generation failed for '{path}': {e}"
                            ),
                            ok: false,
                        };
                    }
                };

                let applied_end = if is_append {
                    end
                } else {
                    expand_replace_end_for_overlap(&current, start, end, &new_content)
                };

                // Reassemble the file deterministically via shunt-edit.
                let edit = if is_append {
                    shunt_edit::Edit::InsertAfter {
                        after: total_lines,
                        new_text: new_content.clone(),
                    }
                } else {
                    shunt_edit::Edit::ReplaceLines {
                        start,
                        end: applied_end,
                        new_text: new_content.clone(),
                    }
                };
                match shunt_edit::apply(&current, &edit) {
                    Ok(updated) => {
                        if let Err(e) = validate_json_edit(&path, &updated) {
                            let current_view =
                                shunt_edit::numbered_window(&current, 1, current.lines().count());
                            return Dispatch::Continue {
                                result: format!(
                                    "Error: edit would make '{path}' invalid JSON: {e}. \
                                     Current file was left unchanged:\n{current_view}\n\
                                     Replace a broader range, usually the full JSON object or full file \
                                     lines 1-{}, with valid JSON.",
                                    current.lines().count()
                                ),
                                ok: false,
                            };
                        }

                        let edit_key = (path.clone(), start, new_content.clone());
                        let repeated = self.edited_lines.contains(&edit_key);
                        if repeated || updated == current {
                            self.ineffective_edit_ranges.insert(range_key);
                            let current_view =
                                shunt_edit::numbered_window(&current, 1, current.lines().count());
                            let reason = if repeated {
                                "you already tried this exact replacement"
                            } else {
                                "the replacement would leave the file unchanged"
                            };
                            return Dispatch::Continue {
                                result: format!(
                                    "Error: no effective edit — {reason}. Current file:\n\
                                     {current_view}\n\
                                     If the task is not satisfied, choose a different or broader \
                                     replace_lines range that covers the stale code. Do not repeat \
                                     the same edit."
                                ),
                                ok: false,
                            };
                        }

                        self.file_state.insert(path.clone(), updated.clone());
                        let updated_view =
                            shunt_edit::numbered_window(&updated, 1, updated.lines().count());
                        let warning = structural_warning_text(&updated, &path);
                        self.ineffective_edit_ranges.clear();
                        self.successful_commands_since_change.clear();
                        self.ops.push(ProposedFileOp::Create {
                            path: path.clone(),
                            contents: updated,
                        });
                        self.written_paths.insert(path.clone());
                        self.edited_lines.insert(edit_key);
                        let msg = if is_append {
                            format!(
                                "OK — appended to '{path}'. Current file:\n{updated_view}\n\
                                 {warning}\
                                 Edit applied. Call done now, or replace_lines on a \
                                 different file/range if more changes are needed."
                            )
                        } else {
                            format!(
                                "OK — replaced lines {start}-{applied_end} in '{path}'. Current file:\n\
                                 {updated_view}\n\
                                 {warning}\
                                 Edit applied. Call done now, or replace_lines on a \
                                 different file/range if more changes are needed."
                            )
                        };
                        Dispatch::Continue {
                            result: msg,
                            ok: true,
                        }
                    }
                    Err(e) => Dispatch::Continue {
                        result: format!("Error applying edit to '{path}': {e}"),
                        ok: false,
                    },
                }
            }

            "apply_patch" => {
                let path = action.path.as_deref().unwrap_or("");
                let patch = action.patch.as_deref().unwrap_or("");
                if patch.is_empty() {
                    return Dispatch::Continue {
                        result: "Error: patch is empty. Provide a unified diff with @@ hunks."
                            .into(),
                        ok: false,
                    };
                }
                let snapshot_before = self.file_state.get(path).cloned().unwrap_or_default();
                let result = apply_unified_patch(&mut self.file_state, path, patch);
                let ok = result.starts_with("OK");
                if ok {
                    self.ineffective_edit_ranges.clear();
                    self.successful_commands_since_change.clear();
                    let snapshot_after = self.file_state.get(path).cloned().unwrap_or_default();
                    self.ops.push(ProposedFileOp::Edit {
                        path: path.to_string(),
                        search: snapshot_before,
                        replacement: snapshot_after,
                    });
                }
                Dispatch::Continue { result, ok }
            }

            "str_replace" => {
                let path = action.path.as_deref().unwrap_or("");
                let old_str = action.old_str.as_deref().unwrap_or("");
                let new_str = action.new_str.as_deref().unwrap_or("");
                let result = apply_str_replace(&mut self.file_state, path, old_str, new_str);
                let ok = result == "OK";
                if ok {
                    self.ineffective_edit_ranges.clear();
                    self.successful_commands_since_change.clear();
                    self.ops.push(ProposedFileOp::Edit {
                        path: path.to_string(),
                        search: old_str.to_string(),
                        replacement: new_str.to_string(),
                    });
                }
                Dispatch::Continue { result, ok }
            }

            "write_file" => {
                let raw_path = action.path.as_deref().unwrap_or("").to_string();
                let abs_path = match resolve_in_workspace(&self.workspace_root, &raw_path) {
                    Ok(p) => p,
                    Err(e) => {
                        return Dispatch::Continue {
                            result: format!("Error: {e}"),
                            ok: false,
                        };
                    }
                };
                let path = abs_path
                    .strip_prefix(&self.workspace_root)
                    .unwrap_or(&abs_path)
                    .to_string_lossy()
                    .into_owned();
                // write_file is for NEW files only. Existing files must be edited
                // with replace_lines (which handles replace, append, and delete).
                if self.file_state.contains_key(&path) {
                    let n = self.file_state[&path].lines().count();
                    return Dispatch::Continue {
                        result: format!(
                            "Error: '{path}' already exists ({n} lines). \
                             Use replace_lines to edit it. \
                             To append, use start_line={} (one past the last line).",
                            n + 1
                        ),
                        ok: false,
                    };
                }
                // If the model already put content in the `query` field, use it directly —
                // BUT only for NEW files (not yet in file_state). For existing files, always
                // use the two-step so the "NEWLY CREATED files" context can be injected,
                // ensuring the model sees what new files need to be registered/imported.
                let file_already_exists = self.file_state.contains_key(&path);
                if let Some(inline) = action.query.as_deref().filter(|s| !s.trim().is_empty()) {
                    if file_already_exists {
                        // Fall through to two-step for existing files.
                    } else {
                        let contents = inline.to_string();
                        if let Some(parent) = abs_path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        if let Err(e) = std::fs::write(&abs_path, &contents) {
                            return Dispatch::Continue {
                                result: format!("Error writing '{path}': {e}"),
                                ok: false,
                            };
                        }
                        self.file_state.insert(path.clone(), contents.clone());
                        self.ineffective_edit_ranges.clear();
                        self.successful_commands_since_change.clear();
                        self.ops.push(ProposedFileOp::Create {
                            path: path.clone(),
                            contents,
                        });
                        self.written_paths.insert(path.clone());
                        return Dispatch::Continue {
                            result: format!("OK — '{path}' written to disk."),
                            ok: true,
                        };
                    } // close else (non-no-op inline content)
                }
                // No inline content (or inline was no-op) — fall through to two-step.
                let task = &self.current_task;
                let current = self.file_state.get(&path).cloned().unwrap_or_default();
                // Include other loaded files as context, capped to avoid bloating the prompt.
                // Each file is truncated to 120 lines — enough for structural understanding.
                const MAX_CONTEXT_FILES: usize = 6;
                const MAX_CONTEXT_LINES: usize = 120;
                let other_files: String = {
                    let mut entries: Vec<(&String, &String)> = self
                        .file_state
                        .iter()
                        .filter(|(p, _)| p.as_str() != path)
                        .collect();
                    // Prioritise smaller/recently-written files (manifest files first)
                    entries.sort_by_key(|(p, c)| (c.lines().count(), p.as_str().to_string()));
                    entries.truncate(MAX_CONTEXT_FILES);
                    entries
                        .iter()
                        .map(|(p, c)| {
                            let lines: Vec<&str> = c.lines().collect();
                            let preview = if lines.len() > MAX_CONTEXT_LINES {
                                format!(
                                    "{}\n... ({} more lines)",
                                    lines[..MAX_CONTEXT_LINES].join("\n"),
                                    lines.len() - MAX_CONTEXT_LINES
                                )
                            } else {
                                c.to_string()
                            };
                            format!("=== {p} ===\n{preview}\n")
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                // Split other files into "newly written this session" vs "pre-loaded context".
                // Newly-written files often need to be registered in the file being edited
                // (e.g. a new route component needs to appear in routes.ts).
                let newly_written: Vec<(&String, &String)> = self
                    .file_state
                    .iter()
                    .filter(|(p, _)| p.as_str() != path && self.written_paths.contains(*p))
                    .collect();
                let newly_written_block = if newly_written.is_empty() {
                    String::new()
                } else {
                    let list = newly_written
                        .iter()
                        .map(|(p, c)| format!("=== {p} (NEWLY CREATED) ===\n{c}\n"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!(
                        "Files NEWLY CREATED in this session (you must reference/register these in '{path}' if applicable):\n{list}\n"
                    )
                };
                let context_block = if other_files.is_empty() {
                    newly_written_block
                } else {
                    format!("{newly_written_block}Other files in workspace:\n{other_files}\n")
                };
                let text_system = "You are a code editor. \
                    Output ONLY the complete file content, starting from line 1. \
                    No code fences, no explanations, no commentary — raw file content only.";
                // Retry generate_text up to 3 times internally before surfacing an error.
                // Each retry escalates the prompt to be more explicit about required output.
                // This avoids burning agent turns on empty-code-fence retries.
                let contents = {
                    let base_user = if current.is_empty() {
                        format!(
                            "Task: {task}\n\n{context_block}Write the complete contents for '{path}':"
                        )
                    } else {
                        format!(
                            "Task: {task}\n\n{context_block}Current file '{path}':\n{current}\n\n\
                             Apply the task above. IMPORTANT: incorporate any NEWLY CREATED files listed above. \
                             Output ONLY the complete updated file, no explanation:"
                        )
                    };
                    let retry_user = format!(
                        "Task: {task}\n\n{context_block}Current file '{path}' ({} lines):\n{current}\n\n\
                         IMPORTANT: Output the COMPLETE updated file. Every single line. Start with line 1.\n\
                         Do not output code fences, explanations, or blank responses.",
                        current.lines().count()
                    );
                    if let Some(o) = &self.observer {
                        o.on_note(&format!("Generating content for '{path}'..."));
                    }
                    let mut result = None;
                    for attempt in 0u8..3 {
                        let prompt = if attempt == 0 {
                            &base_user
                        } else {
                            &retry_user
                        };
                        match self.provider.generate_text(text_system, prompt) {
                            Ok(generated) => {
                                let stripped = strip_code_fences(generated.trim());
                                if !stripped.trim().is_empty() {
                                    result = Some(stripped);
                                    break;
                                }
                                // Empty after stripping — retry with escalated prompt
                            }
                            Err(e) => {
                                return Dispatch::Continue {
                                    result: format!(
                                        "Error: text generation failed for '{path}': {e}"
                                    ),
                                    ok: false,
                                };
                            }
                        }
                    }
                    match result {
                        Some(c) => c,
                        None => {
                            let hint = if current.is_empty() {
                                format!("Write the complete new file content for '{path}'.")
                            } else {
                                format!(
                                    "Output the complete updated '{path}' ({} lines, starting with line 1).",
                                    current.lines().count()
                                )
                            };
                            return Dispatch::Continue {
                                result: format!(
                                    "Error: model produced empty output for '{path}' after 3 attempts. {hint}"
                                ),
                                ok: false,
                            };
                        }
                    }
                };
                if let Some(parent) = abs_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Err(e) = std::fs::write(&abs_path, &contents) {
                    return Dispatch::Continue {
                        result: format!("Error writing '{path}': {e}"),
                        ok: false,
                    };
                }
                self.file_state.insert(path.clone(), contents.clone());
                self.ineffective_edit_ranges.clear();
                self.successful_commands_since_change.clear();
                self.ops.push(ProposedFileOp::Create {
                    path: path.clone(),
                    contents,
                });
                self.written_paths.insert(path.clone());
                Dispatch::Continue {
                    result: format!("OK — '{path}' written to disk."),
                    ok: true,
                }
            }

            "delete_file" => {
                let raw_path = action.path.as_deref().unwrap_or("").to_string();
                let abs_path = match resolve_in_workspace(&self.workspace_root, &raw_path) {
                    Ok(p) => p,
                    Err(e) => {
                        return Dispatch::Continue {
                            result: format!("Error: {e}"),
                            ok: false,
                        };
                    }
                };
                let path = abs_path
                    .strip_prefix(&self.workspace_root)
                    .unwrap_or(&abs_path)
                    .to_string_lossy()
                    .into_owned();
                if let Err(e) = std::fs::remove_file(&abs_path) {
                    // Only fail if file actually exists; missing = already deleted.
                    if e.kind() != std::io::ErrorKind::NotFound {
                        return Dispatch::Continue {
                            result: format!("Error deleting '{path}': {e}"),
                            ok: false,
                        };
                    }
                }
                self.file_state.remove(&path);
                self.ineffective_edit_ranges.clear();
                self.successful_commands_since_change.clear();
                self.ops.push(ProposedFileOp::Delete { path });
                Dispatch::Continue {
                    result: "OK — file deleted from disk.".into(),
                    ok: true,
                }
            }

            "read_file" => {
                let raw_path = action.path.as_deref().unwrap_or("");
                let abs_path = match resolve_in_workspace(&self.workspace_root, raw_path) {
                    Ok(p) => p,
                    Err(e) => {
                        return Dispatch::Continue {
                            result: format!("Error: {e}"),
                            ok: false,
                        };
                    }
                };
                let path_owned = abs_path
                    .strip_prefix(&self.workspace_root)
                    .unwrap_or(&abs_path)
                    .to_string_lossy()
                    .into_owned();
                let path = path_owned.as_str();
                // Reject reads from ignored paths (node_modules, lock files, etc.)
                if is_path_ignored(path, &self.workspace_root, &self.extra_ignore_patterns) {
                    return Dispatch::Continue {
                        result: format!(
                            "Error: '{path}' is in an ignored directory (node_modules, build output, \
                             lock files, etc.) and cannot be read. \
                             Use search_files to find relevant source files instead."
                        ),
                        ok: false,
                    };
                }
                // Check file_state first — covers files written/edited in this session
                // but not yet on disk (write_file and str_replace update file_state).
                if let Some(contents) = self.file_state.get(path) {
                    let size = contents.len();
                    // Return an error so the model knows to stop re-reading and move on.
                    Dispatch::Continue {
                        result: format!(
                            "Error: '{path}' ({size} bytes) is already loaded — it is visible in the FILES section above. \
                             Do not read it again. Proceed to write_file or done."
                        ),
                        ok: false,
                    }
                } else {
                    match std::fs::read_to_string(&abs_path) {
                        Ok(contents) => {
                            let size = contents.len();
                            // Cap at 64KB to avoid flooding context with lock files / large assets.
                            const MAX_FILE_BYTES: usize = 64 * 1024;
                            if size > MAX_FILE_BYTES {
                                return Dispatch::Continue {
                                    result: format!(
                                        "Error: '{path}' is {size} bytes (>{MAX_FILE_BYTES} limit). \
                                         Read a smaller file or use search_files to find specific symbols."
                                    ),
                                    ok: false,
                                };
                            }
                            self.file_state.insert(path.to_string(), contents);
                            Dispatch::Continue {
                                result: format!("(loaded {size} bytes — see FILES section above)"),
                                ok: true,
                            }
                        }
                        Err(e) => Dispatch::Continue {
                            result: format!("Error reading '{path}': {e}"),
                            ok: false,
                        },
                    }
                }
            }

            "search_files" => {
                let query = action.query.as_deref().unwrap_or("");
                let result = search_files_in_workspace(
                    &self.workspace_root,
                    query,
                    &self.extra_ignore_patterns,
                );
                Dispatch::Continue { ok: true, result }
            }

            "command" => self.dispatch_command(&action),

            "run_command_line" => {
                let line = action.command_line.as_deref().unwrap_or("").trim();
                if line.is_empty() {
                    return Dispatch::Continue {
                        result: "Error: command_line is required. Example: {\"tool\":\"run_command_line\",\"command_line\":\"cargo check\"}".into(),
                        ok: false,
                    };
                }
                let argv = match split_command_line(line) {
                    Ok(argv) => argv,
                    Err(e) => {
                        return Dispatch::Continue {
                            result: format!("Error parsing command_line: {e}"),
                            ok: false,
                        };
                    }
                };
                if argv.is_empty() {
                    return Dispatch::Continue {
                        result: "Error: command_line did not contain a program".into(),
                        ok: false,
                    };
                }
                self.execute_deduped_command(format!("line:{line}"), &argv[0], &argv[1..])
            }

            "run_command" => {
                // Auto-fix: model sometimes puts the program name as args[0] instead of cmd.
                // e.g. {"args": ["ls", "-R"]} instead of {"cmd": "ls", "args": ["-R"]}
                let (cmd, tail): (&str, &[String]) =
                    if action.cmd.as_deref().unwrap_or("").is_empty() && !action.args.is_empty() {
                        (action.args[0].as_str(), &action.args[1..])
                    } else {
                        (action.cmd.as_deref().unwrap_or(""), &action.args)
                    };
                if cmd.is_empty() {
                    return Dispatch::Continue {
                        result: "Error: cmd is required. Example: {\"tool\":\"run_command\",\"cmd\":\"ls\",\"args\":[\"-la\"]}".into(),
                        ok: false,
                    };
                }
                self.execute_deduped_command(command_key(cmd, tail), cmd, tail)
            }

            "install_dependencies" => self.queue_install_dependencies(action.path.as_deref()),

            "start_service" => {
                let name = action.service_name.as_deref().unwrap_or("");
                let line = action.command_line.as_deref().unwrap_or("");
                self.start_service(name, line)
            }

            "status_service" => {
                let name = action.service_name.as_deref().unwrap_or("");
                self.status_service(name)
            }

            "stop_service" => {
                let name = action.service_name.as_deref().unwrap_or("");
                self.stop_service(name)
            }

            "ask_user" => Dispatch::NeedsClarification {
                question: action.question.unwrap_or_else(|| "?".into()),
                context: action.context.unwrap_or_default(),
            },

            "sub_agent" => {
                let task = action.task.as_deref().unwrap_or("").to_string();
                let context = action.context.as_deref().unwrap_or("");
                let full_task = if context.is_empty() {
                    task.clone()
                } else {
                    format!("{task}\nContext: {context}")
                };
                let mut sub = AgentSession::new_sub(self.provider, &self.workspace_root);
                sub.extra_ignore_patterns = self.extra_ignore_patterns.clone();
                // Warm sub with parent's current file_state so it doesn't re-read files.
                for (k, v) in &self.file_state {
                    sub.file_state.insert(k.clone(), v.clone());
                }
                let sub_result = sub.run(&full_task);
                // Merge sub's file edits back into parent state.
                for (path, content) in std::mem::take(&mut sub.file_state) {
                    self.file_state.insert(path, content);
                }
                self.ops.extend(std::mem::take(&mut sub.ops));
                let result = match sub_result {
                    AgentResult::Done {
                        description,
                        setup_commands,
                        ..
                    } => {
                        for cmd in setup_commands {
                            if !self
                                .queued_setup_commands
                                .iter()
                                .any(|existing| existing == &cmd)
                            {
                                self.queued_setup_commands.push(cmd);
                            }
                        }
                        description
                    }
                    AgentResult::MaxTurnsReached => {
                        "Sub-agent hit turn limit without a result.".into()
                    }
                    AgentResult::NeedsClarification { question, .. } => {
                        format!("Sub-agent was unsure: {question}")
                    }
                };
                Dispatch::Continue { result, ok: true }
            }

            "done" => Dispatch::Done {
                description: action
                    .description
                    .unwrap_or_else(|| "Applied changes".into()),
                setup_commands: action.setup_commands,
            },

            other => Dispatch::Continue {
                result: format!(
                    "Unknown tool '{other}'. Available tools: \
                      think, read_file, search_files, write_file, \
                     replace_lines, delete_file, command, ask_user, sub_agent, done. \
                      Correct the tool name and try again."
                ),
                ok: false,
            },
        }
    }
}

// ── apply_str_replace — 4-tier progressive matching ──────────────────────────

pub(crate) fn apply_str_replace(
    file_state: &mut HashMap<String, String>,
    path: &str,
    old_str: &str,
    new_str: &str,
) -> String {
    if old_str.is_empty() {
        return "Error: old_str is empty.".to_string();
    }
    if old_str == new_str {
        return "Error: old_str and new_str are identical — no change would be made.".to_string();
    }
    let current = match file_state.get(path) {
        Some(c) => c.clone(),
        None => {
            return format!(
                "Error: file '{path}' not found in context. Use read_file to load it first."
            );
        }
    };

    // Tier 1: exact match
    let count = current.matches(old_str).count();
    if count > 1 {
        return format!(
            "Error: old_str appears {count} times in '{path}'. \
             Include more surrounding lines to make it unique."
        );
    }
    if count == 1 {
        let updated = current.replacen(old_str, new_str, 1);
        file_state.insert(path.to_string(), updated);
        return "OK".to_string();
    }

    // Tier 2: right-strip trailing whitespace on each line
    if let Some(updated) = fuzzy_replace(&current, old_str, new_str, |l| l.trim_end().to_string()) {
        file_state.insert(path.to_string(), updated);
        return "OK".to_string();
    }

    // Tier 3: full trim (handles indent differences)
    if let Some(updated) = fuzzy_replace(&current, old_str, new_str, |l| l.trim().to_string()) {
        file_state.insert(path.to_string(), updated);
        return "OK".to_string();
    }

    // Tier 4: unicode normalization + right-strip (smart quotes, em-dash, etc.)
    if let Some(updated) = fuzzy_replace(&current, old_str, new_str, |l| {
        normalize_unicode(l.trim_end())
    }) {
        file_state.insert(path.to_string(), updated);
        return "OK".to_string();
    }

    format!(
        "Error: old_str not found in '{path}' after exact, whitespace-flexible, \
         and unicode-normalized matching. Verify the text appears in the FILES section."
    )
}

/// Line-by-line fuzzy replace: normalize each line, find a unique window match,
/// replace those original lines with new_str.
fn fuzzy_replace(
    content: &str,
    old_str: &str,
    new_str: &str,
    normalize: impl Fn(&str) -> String,
) -> Option<String> {
    let content_lines: Vec<&str> = content.lines().collect();
    let old_lines: Vec<&str> = old_str.lines().collect();
    if old_lines.is_empty() {
        return None;
    }
    let norm_content: Vec<String> = content_lines.iter().map(|l| normalize(l)).collect();
    let norm_old: Vec<String> = old_lines.iter().map(|l| normalize(l)).collect();
    let window = old_lines.len();

    let mut match_starts: Vec<usize> = Vec::new();
    'outer: for i in 0..=content_lines.len().saturating_sub(window) {
        for j in 0..window {
            if norm_content[i + j] != norm_old[j] {
                continue 'outer;
            }
        }
        match_starts.push(i);
    }

    if match_starts.len() != 1 {
        return None; // not found or ambiguous
    }

    let start = match_starts[0];
    let end = start + window;
    let new_lines: Vec<&str> = new_str.lines().collect();

    let mut result_lines: Vec<&str> = Vec::new();
    result_lines.extend_from_slice(&content_lines[..start]);
    result_lines.extend(new_lines.iter().copied());
    result_lines.extend_from_slice(&content_lines[end..]);

    let mut result = result_lines.join("\n");
    if content.ends_with('\n') && !result.ends_with('\n') {
        result.push('\n');
    }
    Some(result)
}

fn normalize_unicode(s: &str) -> String {
    s.replace(['\u{2018}', '\u{2019}'], "'")
        .replace(['\u{201C}', '\u{201D}'], "\"")
        .replace('\u{2013}', "-") // en dash
        .replace('\u{2014}', "--") // em dash
}

// ── apply_unified_patch ───────────────────────────────────────────────────────

/// Parse a unified diff and apply each hunk via apply_str_replace (with fuzzy matching).
fn apply_unified_patch(
    file_state: &mut HashMap<String, String>,
    path: &str,
    patch: &str,
) -> String {
    let mut hunks: Vec<(String, String)> = Vec::new();
    let mut in_hunk = false;
    let mut old_parts: Vec<String> = Vec::new();
    let mut new_parts: Vec<String> = Vec::new();

    let flush = |old_parts: &mut Vec<String>,
                 new_parts: &mut Vec<String>,
                 hunks: &mut Vec<(String, String)>| {
        let old_str = old_parts.join("\n");
        let new_str = new_parts.join("\n");
        if !old_str.is_empty() || !new_str.is_empty() {
            hunks.push((old_str, new_str));
        }
        old_parts.clear();
        new_parts.clear();
    };

    for line in patch.lines() {
        if line.starts_with("---") || line.starts_with("+++") {
            // File header lines — end current hunk if any
            if in_hunk {
                flush(&mut old_parts, &mut new_parts, &mut hunks);
                in_hunk = false;
            }
            continue;
        }
        if line.starts_with("@@") {
            if in_hunk {
                flush(&mut old_parts, &mut new_parts, &mut hunks);
            }
            in_hunk = true;
            continue;
        }
        if in_hunk {
            if let Some(rest) = line.strip_prefix('-') {
                old_parts.push(rest.to_string());
            } else if let Some(rest) = line.strip_prefix('+') {
                new_parts.push(rest.to_string());
            } else if let Some(rest) = line.strip_prefix(' ') {
                // Context line — present in both old and new
                old_parts.push(rest.to_string());
                new_parts.push(rest.to_string());
            }
            // Lines with no diff prefix (bare text) — treat as context
            else if !line.is_empty() {
                old_parts.push(line.to_string());
                new_parts.push(line.to_string());
            }
        }
    }
    if in_hunk {
        flush(&mut old_parts, &mut new_parts, &mut hunks);
    }

    if hunks.is_empty() {
        return "Error: no hunks found in patch. Use unified diff format with @@ markers and - / + prefixed lines.".to_string();
    }

    let mut applied = 0usize;
    let mut errors: Vec<String> = Vec::new();
    for (old_str, new_str) in &hunks {
        let result = apply_str_replace(file_state, path, old_str, new_str);
        if result == "OK" {
            applied += 1;
        } else {
            errors.push(result);
        }
    }

    if errors.is_empty() {
        format!("OK — {applied} hunk(s) applied")
    } else if applied > 0 {
        format!(
            "Partial: {applied}/{} hunk(s) applied. Errors: {}",
            hunks.len(),
            errors.join("; ")
        )
    } else {
        format!("Error: {}", errors.join("; "))
    }
}

// ── Ignore patterns ───────────────────────────────────────────────────────────

/// Build an `ignore::gitignore::Gitignore` from a combined set of patterns
/// (defaults + user extras).  Used to filter `read_file` and file listing.
fn build_ignore_matcher(workspace_root: &str, extra: &[String]) -> ignore::gitignore::Gitignore {
    let mut builder = ignore::gitignore::GitignoreBuilder::new(workspace_root);
    for pat in DEFAULT_IGNORE_PATTERNS {
        let _ = builder.add_line(None, pat);
    }
    for pat in extra {
        let _ = builder.add_line(None, pat);
    }
    builder
        .build()
        .unwrap_or_else(|_| ignore::gitignore::Gitignore::empty())
}

fn is_path_ignored(rel_path: &str, workspace_root: &str, extra: &[String]) -> bool {
    let gi = build_ignore_matcher(workspace_root, extra);
    let abs = Path::new(workspace_root).join(rel_path);
    // Check the full path and every component against ignore rules.
    // This catches "node_modules/foo/bar" even though the pattern is just "node_modules".
    gi.matched_path_or_any_parents(&abs, abs.is_dir())
        .is_ignore()
}

// ── search_files ──────────────────────────────────────────────────────────────

fn search_files_in_workspace(workspace_root: &str, query: &str, extra_ignore: &[String]) -> String {
    use shunt_localize::WorkspaceSearch;
    if query.trim().is_empty() {
        return list_workspace_files(workspace_root, extra_ignore);
    }
    let ws = WorkspaceSearch::new(workspace_root);
    let gi = build_ignore_matcher(workspace_root, extra_ignore);
    let root = Path::new(workspace_root);
    let hits: Vec<String> = ws
        .search_files(query)
        .into_iter()
        .filter(|p| {
            let abs = root.join(p);
            !gi.matched_path_or_any_parents(&abs, false).is_ignore()
        })
        .collect();
    if hits.is_empty() {
        format!(
            "No files found for '{query}'. Try a different keyword, or call search_files with empty query to list all files."
        )
    } else {
        hits.join("\n")
    }
}

fn list_workspace_files(workspace_root: &str, extra_ignore: &[String]) -> String {
    let root = Path::new(workspace_root);
    let gi = build_ignore_matcher(workspace_root, extra_ignore);
    // WalkBuilder already respects .gitignore; we add our defaults on top.
    let walker = ignore::WalkBuilder::new(root).hidden(true).build();
    let mut files: Vec<String> = walker
        .flatten()
        .filter(|e| e.path().is_file())
        .filter_map(|e| {
            let rel = e.path().strip_prefix(root).ok()?;
            let rel_str = rel.to_string_lossy().into_owned();
            if gi.matched_path_or_any_parents(e.path(), false).is_ignore() {
                return None;
            }
            Some(rel_str)
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return "No files in workspace.".into();
    }
    files.join("\n")
}

// ── System prompt builder ─────────────────────────────────────────────────────

fn build_system_prompt(workspace_root: &str) -> String {
    let root = Path::new(workspace_root);
    let gi = build_ignore_matcher(workspace_root, &[]);
    let tree = build_dir_tree(root, 0, 3, &mut 0, 60, &gi);
    let manifests = read_manifest_files(root);

    let mut prompt = AGENT_SYSTEM_PROMPT_BASE.to_string();
    prompt.push_str("\n\n---\n\n## Workspace\n\n");
    prompt.push_str(&format!("Root: {workspace_root}\n\n"));

    if !tree.is_empty() {
        prompt.push_str("### Directory Structure\n```\n");
        prompt.push_str(&tree);
        prompt.push_str("```\n\n");
    }

    if !manifests.is_empty() {
        prompt.push_str("### Key Files\n");
        for (name, contents) in &manifests {
            prompt.push_str(&format!("\n#### {name}\n```\n{contents}\n```\n"));
        }
    }

    prompt
}

const VERIFIER_SYSTEM_PROMPT_BASE: &str = "\
You are a QA engineer verifying that a coding agent's changes actually work.
Each response is ONE JSON action. You do NOT write or modify files.

APPROACH:
1. The changed files are already pre-loaded — read them to understand what was implemented.
2. Run the build first to confirm no compilation errors.
   Node/TypeScript → command {\"command_action\":\"run\",\"command_line\":\"pnpm build\"}
   Rust            → command {\"command_action\":\"run\",\"command_line\":\"cargo check\"}
3. Run surgical smoke tests matched to what changed:
   - New HTTP route → command start_service for the dev server, command run curl, command stop_service
   - New component   → inspect build output for warnings / missing exports
   - Modified logic  → run the existing test suite if one is present
4. Call done with EXACTLY this format in description:
   PASS: <what was tested and confirmed working>
   — or —
   FAIL: <specific what broke, exact error output>

CONSTRAINTS:
- Never call write_file, str_replace, or delete_file — read-only.
- For long-running dev servers or daemons, use command actions start_service/status_service/stop_service. Do NOT wrap servers in timeout or background shell syntax.
- Keep scope surgical — only test what the changes touch.
- If the build fails, stop there and report FAIL with the error.

Available tools: think, read_file, search_files, command, done";

fn build_verifier_prompt(workspace_root: &str) -> String {
    let root = Path::new(workspace_root);
    let gi = build_ignore_matcher(workspace_root, &[]);
    let manifests = read_manifest_files(root);

    let mut prompt = VERIFIER_SYSTEM_PROMPT_BASE.to_string();
    prompt.push_str("\n\n---\n\n## Workspace\n\n");
    prompt.push_str(&format!("Root: {workspace_root}\n\n"));

    if !manifests.is_empty() {
        prompt.push_str("### Key Files\n");
        for (name, contents) in &manifests {
            prompt.push_str(&format!("\n#### {name}\n```\n{contents}\n```\n"));
        }
    }

    let _ = gi;
    prompt
}

fn build_dir_tree(
    dir: &Path,
    depth: usize,
    max_depth: usize,
    line_count: &mut usize,
    max_lines: usize,
    gi: &ignore::gitignore::Gitignore,
) -> String {
    if depth > max_depth || *line_count >= max_lines {
        return String::new();
    }
    let mut out = String::new();
    let indent = "  ".repeat(depth);
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return out,
    };
    let mut names: Vec<_> = entries.flatten().collect();
    names.sort_by_key(|e| e.file_name());
    for entry in &names {
        if *line_count >= max_lines {
            break;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Skip hidden files/dirs
        if name_str.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let is_dir = path.is_dir();
        if gi.matched_path_or_any_parents(&path, is_dir).is_ignore() {
            continue;
        }
        if is_dir {
            out.push_str(&format!("{indent}{name_str}/\n"));
            *line_count += 1;
            out.push_str(&build_dir_tree(
                &path,
                depth + 1,
                max_depth,
                line_count,
                max_lines,
                gi,
            ));
        } else {
            out.push_str(&format!("{indent}{name_str}\n"));
            *line_count += 1;
        }
    }
    out
}

const MANIFEST_NAMES: &[&str] = &[
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pyproject.toml",
    "requirements.txt",
    "deno.json",
];
const README_NAMES: &[&str] = &["README.md", "README.txt", "README"];

fn read_manifest_files(root: &Path) -> Vec<(String, String)> {
    let mut result = Vec::new();
    for name in MANIFEST_NAMES {
        let path = root.join(name);
        if let Ok(contents) = std::fs::read_to_string(&path) {
            // Cap manifest contents at 200 lines.
            let capped: String = contents.lines().take(200).collect::<Vec<_>>().join("\n");
            result.push(((*name).to_string(), capped));
        }
    }
    // README — first 40 lines.
    for name in README_NAMES {
        let path = root.join(name);
        if let Ok(contents) = std::fs::read_to_string(&path) {
            let capped: String = contents.lines().take(40).collect::<Vec<_>>().join("\n");
            result.push(((*name).to_string(), capped));
            break;
        }
    }
    result
}

// ── Multi-turn context helpers ────────────────────────────────────────────────

/// Soft token ceiling for `conv_history` (excludes system + task frame + ephemeral).
/// ~4 chars per token; 2000 tokens ≈ 8KB. With a -c 8192 server this leaves
/// ~400 tokens for the system prompt, ~2000 for the ephemeral continuation
/// (FILES section), and ~3000 for model output.
const HISTORY_TOKEN_SOFT_LIMIT: usize = 2000;

/// Build the ephemeral continuation message (FILES + nudge) that is appended to
/// the last message in the inference payload every turn.  NOT stored in
/// `conv_history` — rebuilt fresh so line numbers and nudge text are always current.
fn build_continuation_msg(
    task: &str,
    file_state: &HashMap<String, String>,
    ops: &[ProposedFileOp],
    turns: &[AgentTurn],
    workspace_root: &str,
    budget: &SessionBudget,
) -> String {
    let kw_refs = task_keywords(task);
    let kw_refs: Vec<&str> = kw_refs.iter().map(String::as_str).collect();

    let mut msg = String::new();

    msg.push_str(&environment_profile(workspace_root));

    // Files section — always shows the CURRENT state with real line numbers.
    let mut paths: Vec<&String> = file_state.keys().collect();
    paths.sort();
    if !paths.is_empty() {
        msg.push_str("FILES (line numbers shown — use them with replace_lines):\n");
        for path in &paths {
            let contents = &file_state[*path];
            let numbered = render_file_numbered(contents, &kw_refs);
            msg.push_str(&format!("\n<file path=\"{path}\">\n{numbered}</file>\n"));
        }
    }

    // Hint at relevant unloaded files.
    let unloaded = find_likely_unloaded(workspace_root, task, file_state);
    if !unloaded.is_empty() {
        msg.push_str("\nOther files in workspace (not yet loaded): ");
        msg.push_str(&unloaded.join(", "));
        msg.push('\n');
    }

    // Contextual nudge based on turn history.
    msg.push_str(&build_nudge(task, file_state, ops, turns, budget));
    msg
}

fn environment_profile(workspace_root: &str) -> String {
    let mut commands = available_commands();
    commands.sort();
    commands.dedup();
    let listed = commands.iter().take(160).cloned().collect::<Vec<_>>();
    let suffix = if commands.len() > listed.len() {
        format!("; +{} more", commands.len() - listed.len())
    } else {
        String::new()
    };
    format!(
        "ENVIRONMENT:\nworkspace_root: {workspace_root}\navailable_commands: {}{}\n\n",
        listed.join(", "),
        suffix
    )
}

fn available_commands() -> Vec<String> {
    let Some(path) = std::env::var_os("PATH") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for dir in std::env::split_paths(&path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if meta.permissions().mode() & 0o111 == 0 {
                    continue;
                }
            }
            if let Some(name) = entry.file_name().to_str() {
                out.push(name.to_string());
            }
        }
    }
    out
}

fn split_command_line(input: &str) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut quote: Option<char> = None;
    while let Some(ch) = chars.next() {
        match (quote, ch) {
            (None, c) if c.is_whitespace() => {
                if !current.is_empty() {
                    args.push(std::mem::take(&mut current));
                }
            }
            (None, '\'' | '"') => quote = Some(ch),
            (Some(q), c) if c == q => quote = None,
            (_, '\\') => {
                if let Some(next) = chars.next() {
                    current.push(next);
                } else {
                    current.push('\\');
                }
            }
            (_, c) => current.push(c),
        }
    }
    if let Some(q) = quote {
        return Err(format!("unterminated {q} quote"));
    }
    if !current.is_empty() {
        args.push(current);
    }
    Ok(args)
}

fn check_command_safety(cmd: &str, args: &[String]) -> Result<(), Dispatch> {
    // Apply the shared safety classifier (shunt_core::safety). Command-line input
    // is parsed into argv first; nothing is executed through a shell here.
    let spec = shunt_core::CommandSpec::new(cmd, args.iter().map(String::as_str));
    match safety::classify(&spec) {
        safety::CommandSafety::Blocked { reason } => Err(Dispatch::Continue {
            result: format!("Error: command blocked — {reason}"),
            ok: false,
        }),
        safety::CommandSafety::Dangerous { reason } => Err(Dispatch::Continue {
            result: format!(
                "Error: command requires approval — {reason}. Use done.setup_commands for commands that need user review."
            ),
            ok: false,
        }),
        safety::CommandSafety::Safe => Ok(()),
    }
}

fn command_key(cmd: &str, args: &[String]) -> String {
    format!("argv:{}", render_command(cmd, args))
}

fn render_command(cmd: &str, args: &[String]) -> String {
    std::iter::once(cmd)
        .chain(args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

fn execute_command(workspace_root: &str, cmd: &str, args: &[String]) -> Dispatch {
    if let Err(result) = check_command_safety(cmd, args) {
        return result;
    }

    // Execvp-style: cmd is the program, args are its arguments.
    // Stdout/stderr drained on threads to prevent pipe-buffer deadlock.
    // Hard timeout kills the child so a hung build doesn't block forever.
    const AGENT_CMD_TIMEOUT_SECS: u64 = 60;
    let mut child = match std::process::Command::new(cmd)
        .args(args)
        .current_dir(workspace_root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return Dispatch::Continue {
                result: format!("Error spawning '{cmd}': {e}"),
                ok: false,
            };
        }
    };
    let mut stdout_pipe = child.stdout.take().expect("piped");
    let mut stderr_pipe = child.stderr.take().expect("piped");
    let (tx_out, rx_out) = std::sync::mpsc::channel::<Vec<u8>>();
    let (tx_err, rx_err) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        use std::io::Read;
        let _ = stdout_pipe.read_to_end(&mut buf);
        let _ = tx_out.send(buf);
    });
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        use std::io::Read;
        let _ = stderr_pipe.read_to_end(&mut buf);
        let _ = tx_err.send(buf);
    });
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(AGENT_CMD_TIMEOUT_SECS);
    let exit_status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Ok(s),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(format!(
                        "timed out after {AGENT_CMD_TIMEOUT_SECS}s — process killed"
                    ));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => break Err(format!("wait error: {e}")),
        }
    };
    let stdout_bytes = rx_out.recv().unwrap_or_default();
    let stderr_bytes = rx_err.recv().unwrap_or_default();
    match exit_status {
        Ok(status) => {
            let stdout = tail_utf8(&stdout_bytes, MAX_CMD_OUTPUT);
            let stderr = tail_utf8(&stderr_bytes, MAX_CMD_OUTPUT);
            let ok = status.success();
            let result = if stderr.is_empty() {
                stdout
            } else {
                format!("stdout:\n{stdout}\nstderr:\n{stderr}")
            };
            Dispatch::Continue {
                result: result.trim().to_string(),
                ok,
            }
        }
        Err(msg) => Dispatch::Continue {
            result: format!("Error: {msg}"),
            ok: false,
        },
    }
}

fn spawn_log_reader<R: Read + Send + 'static>(mut reader: R, sink: Arc<Mutex<Vec<u8>>>) {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    if let Ok(mut buf) = sink.lock() {
                        buf.extend_from_slice(&chunk[..n]);
                        if buf.len() > MAX_CMD_OUTPUT * 4 {
                            let keep_from = buf.len() - MAX_CMD_OUTPUT * 4;
                            buf.drain(..keep_from);
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });
}

fn service_logs(stdout: &Arc<Mutex<Vec<u8>>>, stderr: &Arc<Mutex<Vec<u8>>>) -> String {
    let stdout = stdout
        .lock()
        .map(|b| tail_utf8(&b, MAX_CMD_OUTPUT))
        .unwrap_or_default();
    let stderr = stderr
        .lock()
        .map(|b| tail_utf8(&b, MAX_CMD_OUTPUT))
        .unwrap_or_default();
    match (stdout.trim().is_empty(), stderr.trim().is_empty()) {
        (true, true) => "logs: <empty>".into(),
        (false, true) => format!("stdout:\n{}", stdout.trim()),
        (true, false) => format!("stderr:\n{}", stderr.trim()),
        (false, false) => format!("stdout:\n{}\nstderr:\n{}", stdout.trim(), stderr.trim()),
    }
}

fn expand_replace_end_for_overlap(
    current: &str,
    start: usize,
    end: usize,
    new_content: &str,
) -> usize {
    let current_lines: Vec<&str> = current.lines().collect();
    let new_lines: Vec<&str> = new_content.lines().collect();
    if end >= current_lines.len() {
        return end;
    }

    let tail = &current_lines[end..];
    let mut expanded = end + exact_line_overlap(new_lines.as_slice(), tail);
    expanded = expand_for_duplicated_tail_declarations(&current_lines, expanded, &new_lines);

    if new_lines.len() <= end.saturating_sub(start) + 1 {
        return expanded;
    }

    let new_keys = new_lines
        .iter()
        .filter_map(|line| statement_key(line))
        .collect::<HashSet<_>>();
    if !new_keys.is_empty() {
        while expanded < current_lines.len() {
            let Some(key) = statement_key(current_lines[expanded]) else {
                break;
            };
            if !new_keys.contains(&key) {
                break;
            }
            expanded += 1;
        }
    }
    expanded
}

fn expand_for_duplicated_tail_declarations(
    current_lines: &[&str],
    expanded: usize,
    new_lines: &[&str],
) -> usize {
    let new_declarations = new_lines
        .iter()
        .filter_map(|line| declaration_key(line))
        .collect::<HashSet<_>>();
    if new_declarations.is_empty() {
        return expanded;
    }

    let mut next_idx = expanded;
    let mut result = expanded;
    while next_idx < current_lines.len() {
        while next_idx < current_lines.len() && is_ignorable_tail_line(current_lines[next_idx]) {
            next_idx += 1;
        }
        if next_idx >= current_lines.len() {
            break;
        }
        let Some(key) = declaration_key(current_lines[next_idx]) else {
            break;
        };
        if !new_declarations.contains(&key) {
            break;
        }
        next_idx = declaration_block_end(current_lines, next_idx);
        result = next_idx;
    }
    result
}

fn declaration_block_end(lines: &[&str], start_idx: usize) -> usize {
    let mut depth = 0i32;
    let mut saw_brace = false;
    for (idx, line) in lines.iter().enumerate().skip(start_idx) {
        for ch in line.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    saw_brace = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if saw_brace && depth <= 0 {
            return idx + 1;
        }
        if !saw_brace && line.trim_end().ends_with(';') {
            return idx + 1;
        }
    }
    start_idx + 1
}

fn is_ignorable_tail_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#')
}

fn structural_warning_text(contents: &str, path: &str) -> String {
    if !is_structural_check_candidate(path) {
        return String::new();
    }
    let warnings = structural_warnings(contents);
    if warnings.is_empty() {
        return String::new();
    }

    format!(
        "Structural warning for '{path}': {}. If this is not intentional, replace the full affected block before calling done.\n",
        warnings.join("; ")
    )
}

fn is_structural_check_candidate(path: &str) -> bool {
    matches!(
        Path::new(path).extension().and_then(|ext| ext.to_str()),
        Some(
            "rs" | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "mjs"
                | "cjs"
                | "json"
                | "go"
                | "java"
                | "c"
                | "cc"
                | "cpp"
                | "h"
                | "hpp"
                | "cs"
        )
    )
}

fn structural_warnings(contents: &str) -> Vec<String> {
    let mut warnings = Vec::new();
    let mut brace_depth = 0i32;
    for ch in contents.chars() {
        match ch {
            '{' => brace_depth += 1,
            '}' => brace_depth -= 1,
            _ => {}
        }
    }
    if brace_depth > 0 {
        warnings.push(format!(
            "brace balance appears to have {brace_depth} unmatched '{{'"
        ));
    } else if brace_depth < 0 {
        warnings.push(format!(
            "brace balance appears to have {} unmatched '}}'",
            brace_depth.abs()
        ));
    }

    let mut seen: HashMap<String, usize> = HashMap::new();
    for (idx, line) in contents.lines().enumerate() {
        let Some(key) = declaration_key(line) else {
            continue;
        };
        if let Some(first_line) = seen.get(&key) {
            warnings.push(format!(
                "duplicate declaration '{key}' at lines {first_line} and {}",
                idx + 1
            ));
        } else {
            seen.insert(key, idx + 1);
        }
    }

    warnings
}

fn exact_line_overlap(new_lines: &[&str], tail: &[&str]) -> usize {
    let max_overlap = new_lines.len().min(tail.len());
    for overlap in (1..=max_overlap).rev() {
        if new_lines[new_lines.len() - overlap..]
            .iter()
            .zip(tail.iter().take(overlap))
            .all(|(a, b)| a.trim() == b.trim())
        {
            return overlap;
        }
    }
    0
}

fn statement_key(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let prefixes = [
        "let ", "const ", "var ", "pub ", "use ", "import ", "export ",
    ];
    if !prefixes.iter().any(|prefix| trimmed.starts_with(prefix)) {
        return None;
    }
    let head = trimmed
        .split(['=', '{', ';', '('])
        .next()
        .unwrap_or(trimmed)
        .trim();
    if head.is_empty() {
        None
    } else {
        Some(head.to_string())
    }
}

fn declaration_key(line: &str) -> Option<String> {
    let mut trimmed = line.trim();
    for prefix in ["export default ", "export ", "pub "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            trimmed = rest.trim_start();
        }
    }
    if let Some(rest) = trimmed.strip_prefix("async ") {
        trimmed = rest.trim_start();
    }

    for prefix in [
        "function ",
        "fn ",
        "class ",
        "interface ",
        "type ",
        "struct ",
        "enum ",
        "trait ",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let name = rest
                .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
                .next()
                .unwrap_or("")
                .trim();
            if !name.is_empty() {
                return Some(format!("{} {}", prefix.trim_end(), name));
            }
        }
    }

    None
}

fn is_invalid_action_error(err: &crate::InferError) -> bool {
    matches!(
        err,
        crate::InferError::InvalidOutput { .. }
            | crate::InferError::Json(_)
            | crate::InferError::EmptyResponse
            | crate::InferError::UnexpectedTool { .. }
    )
}

/// Build the contextual nudge text appended to the ephemeral continuation message.
fn build_nudge(
    task: &str,
    file_state: &HashMap<String, String>,
    ops: &[ProposedFileOp],
    turns: &[AgentTurn],
    budget: &SessionBudget,
) -> String {
    if turns.is_empty() {
        return "\nStart with think to reason about the approach, then explore what you need. \
                What is your first action?"
            .into();
    }
    let remaining = budget.max_turns.saturating_sub(turns.len());
    let idle_streak = turns
        .iter()
        .rev()
        .take_while(|t| matches!(t.tool.as_str(), "think" | "read_file" | "search_files"))
        .count();
    let wrote_any = turns.iter().any(|t| {
        matches!(
            t.tool.as_str(),
            "write_file" | "str_replace" | "delete_file" | "replace_lines"
        ) && t.ok
    });
    let last_verified = turns.iter().rev().find(|t| {
        matches!(
            t.tool.as_str(),
            "command" | "run_command" | "run_command_line"
        )
    });
    let last_verified_ok = last_verified
        .map(|t| {
            t.ok && !t.result.to_ascii_lowercase().contains("error")
                && !t.result.to_ascii_lowercase().contains("failed")
        })
        .unwrap_or(false);
    let last_action = turns.last().map(|t| t.tool.as_str()).unwrap_or("");
    let last_action_ok = turns.last().map(|t| t.ok).unwrap_or(false);
    let missing_explicit_paths = missing_explicit_file_edits(task, file_state, ops);

    if idle_streak >= budget.stall_warn_at {
        format!(
            "\nSTALL WARNING: {idle_streak} turns of reading/thinking without any edit or command. \
             You MUST call write_file, replace_lines, command, or done NOW. \
             No more reads or thinks. ({remaining} turns remaining)"
        )
    } else if wrote_any && !missing_explicit_paths.is_empty() {
        format!(
            "\nThe task explicitly names file(s) that are not edited yet: {}. \
             Load/edit those file(s) before running verification commands or calling done. \
             ({remaining} turns remaining)",
            missing_explicit_paths.join(", ")
        )
    } else if last_action == "done" {
        String::new()
    } else if wrote_any
        && matches!(last_action, "command" | "run_command" | "run_command_line")
        && last_verified_ok
    {
        format!(
            "\nBuild passed. Call done with a description of what changed and what was verified. \
             ({remaining} turns remaining)"
        )
    } else if wrote_any
        && matches!(last_action, "command" | "run_command" | "run_command_line")
        && !last_verified_ok
    {
        format!(
            "\nBuild failed. Use think to reason about the root cause, then use replace_lines \
             to fix the specific error. Re-run the build after fixing. \
             ({remaining} turns remaining)"
        )
    } else if wrote_any && !matches!(last_action, "command" | "run_command" | "run_command_line") {
        format!(
            "\nFiles written. Run the build now to verify your changes work before calling done. \
             ({remaining} turns remaining)"
        )
    } else if last_action == "think" && last_action_ok {
        format!(
            "\nThought recorded. Your NEXT action MUST NOT be think — \
             call read_file, write_file, replace_lines, command, or done now. \
             ({remaining} turns remaining)"
        )
    } else {
        format!("\nContinue. ({remaining} turns remaining)")
    }
}

fn missing_explicit_file_edits(
    task: &str,
    file_state: &HashMap<String, String>,
    ops: &[ProposedFileOp],
) -> Vec<String> {
    let changed: HashSet<&str> = ops
        .iter()
        .map(|op| match op {
            ProposedFileOp::Create { path, .. }
            | ProposedFileOp::Edit { path, .. }
            | ProposedFileOp::Delete { path } => path.as_str(),
        })
        .collect();

    explicit_file_paths(task)
        .into_iter()
        .filter(|path| file_state.contains_key(path) && !changed.contains(path.as_str()))
        .collect()
}

fn explicit_file_paths(task: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for raw in task.split_whitespace() {
        let token = raw.trim_matches(|c: char| {
            matches!(
                c,
                '`' | '\'' | '"' | ',' | '.' | ':' | ';' | '(' | ')' | '[' | ']'
            )
        });
        if token.contains('/') || looks_like_filename(token) {
            let path = token.trim_start_matches("./").to_string();
            if !path.is_empty() && !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    paths
}

fn looks_like_filename(token: &str) -> bool {
    let Some((stem, ext)) = token.rsplit_once('.') else {
        return false;
    };
    !stem.is_empty()
        && matches!(
            ext,
            "c" | "cc"
                | "cpp"
                | "cs"
                | "css"
                | "go"
                | "h"
                | "hpp"
                | "html"
                | "java"
                | "js"
                | "json"
                | "jsx"
                | "kt"
                | "lua"
                | "md"
                | "py"
                | "rb"
                | "rs"
                | "sh"
                | "sql"
                | "toml"
                | "ts"
                | "tsx"
                | "txt"
                | "yaml"
                | "yml"
        )
}

fn validate_json_edit(path: &str, contents: &str) -> Result<(), serde_json::Error> {
    if path.ends_with(".json") {
        serde_json::from_str::<serde_json::Value>(contents)?;
    }
    Ok(())
}

/// Extract task keywords used to highlight relevant lines in FILES rendering.
fn task_keywords(task: &str) -> Vec<String> {
    task.split_whitespace()
        .filter(|w| w.len() > 3)
        .map(|w| {
            w.to_ascii_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .take(6)
        .collect()
}

/// Serialize an `AgentAction` to compact JSON using only the non-None fields.
/// Used for the assistant message in the multi-turn conversation history.
fn action_to_compact_json(action: &AgentAction) -> String {
    let mut map = serde_json::Map::new();
    map.insert("tool".into(), serde_json::json!(action.tool));
    if let Some(p) = &action.path {
        map.insert("path".into(), serde_json::json!(p));
    }
    if let Some(sl) = action.start_line {
        map.insert("start_line".into(), serde_json::json!(sl));
    }
    if let Some(el) = action.end_line {
        map.insert("end_line".into(), serde_json::json!(el));
    }
    if let Some(q) = &action.query {
        map.insert("query".into(), serde_json::json!(q));
    }
    if let Some(command_action) = &action.command_action {
        map.insert("command_action".into(), serde_json::json!(command_action));
    }
    if let Some(cmd) = &action.cmd {
        map.insert("cmd".into(), serde_json::json!(cmd));
    }
    if let Some(command_line) = &action.command_line {
        map.insert("command_line".into(), serde_json::json!(command_line));
    }
    if let Some(service_name) = &action.service_name {
        map.insert("service_name".into(), serde_json::json!(service_name));
    }
    if !action.args.is_empty() {
        map.insert("args".into(), serde_json::json!(action.args));
    }
    if let Some(q) = &action.question {
        map.insert("question".into(), serde_json::json!(q));
    }
    if let Some(ctx) = &action.context {
        map.insert("context".into(), serde_json::json!(ctx));
    }
    if let Some(t) = &action.task {
        map.insert("task".into(), serde_json::json!(t));
    }
    if let Some(d) = &action.description {
        map.insert("description".into(), serde_json::json!(d));
    }
    serde_json::Value::Object(map).to_string()
}

/// Evict oldest messages from `conv_history` when the estimated token count
/// exceeds `HISTORY_TOKEN_SOFT_LIMIT`.
///
/// Eviction strategy (Working Set Model, oldest-first):
///   1. `think` user results  → compress to a short preview + char count
///   2. `search_files` results → keep first 5 lines, archive the rest
///   3. command output         → keep the most relevant tail
///   4. Other long results     → truncate to 300 chars
///
/// The last 3 turn pairs (6 messages) are never evicted.
fn evict_history_if_needed(
    history: &mut [(String, String)],
    cold: &mut Vec<ColdEntry>,
    turns: &[AgentTurn],
) {
    let estimated: usize = history.iter().map(|(_, c)| c.len() / 4).sum();
    if estimated <= HISTORY_TOKEN_SOFT_LIMIT {
        return;
    }
    // Protect last 3 turn pairs = last 6 messages from eviction.
    let protect_from = history.len().saturating_sub(6);

    for i in 0..protect_from {
        let re_check: usize = history.iter().map(|(_, c)| c.len() / 4).sum();
        if re_check <= HISTORY_TOKEN_SOFT_LIMIT {
            break;
        }
        let (role, content) = &history[i];
        if role != "user" || content.len() <= 120 {
            continue;
        }
        let turn_idx = i / 2;
        let tool = turns
            .get(turn_idx)
            .map(|t| t.tool.as_str())
            .unwrap_or("unknown");
        let compressed = compress_for_cold_storage(content, tool);
        if compressed.len() < content.len() {
            cold.push(ColdEntry {
                turn_idx,
                tool: tool.to_string(),
                original: content.clone(),
                compressed_to: compressed.clone(),
            });
            history[i].1 = compressed;
        }
    }
}

/// Produce a short summary of a tool result for cold-storage compression.
fn compress_for_cold_storage(content: &str, tool: &str) -> String {
    match tool {
        "think" => {
            let preview = content.chars().take(80).collect::<String>();
            format!(
                "[thought: {}… ({} chars — evicted)]",
                preview.trim(),
                content.len()
            )
        }
        "search_files" => {
            let lines: Vec<&str> = content.lines().collect();
            if lines.len() <= 5 {
                return content.to_string();
            }
            format!(
                "{}\n[{} more files — evicted to cold storage]",
                lines[..5].join("\n"),
                lines.len() - 5
            )
        }
        "command" | "run_command" | "run_command_line" | "start_service" | "status_service"
        | "stop_service" => {
            if content.len() <= 120 {
                return content.to_string();
            }
            // Keep last 120 chars (most relevant: errors and logs are at the end).
            let tail = &content[content.len() - 120..];
            format!("[process output truncated]\n…{tail}")
        }
        _ => {
            let char_count = content.chars().count();
            if char_count <= 300 {
                content.to_string()
            } else {
                let head: String = content.chars().take(300).collect();
                format!("{head}…[{} chars evicted]", char_count - 300)
            }
        }
    }
}

/// Return a short list of likely-relevant files on disk that aren't in file_state.
fn find_likely_unloaded(
    workspace_root: &str,
    task: &str,
    file_state: &HashMap<String, String>,
) -> Vec<String> {
    let task_lower = task.to_ascii_lowercase();
    let root = Path::new(workspace_root);
    let mut found = Vec::new();
    let walker = ignore::WalkBuilder::new(root)
        .max_depth(Some(4))
        .hidden(true)
        .build();
    for entry in walker.flatten().filter(|e| e.path().is_file()) {
        let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
        let rel_str = rel.to_string_lossy().to_string();
        if file_state.contains_key(&rel_str) {
            continue;
        }
        let name = rel
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_lowercase());
        let relevant = name
            .as_deref()
            .map(|n| {
                MANIFEST_NAMES.iter().any(|m| m.to_ascii_lowercase() == n)
                    || task_lower
                        .split_whitespace()
                        .any(|w| w.len() > 3 && n.contains(w))
            })
            .unwrap_or(false);
        if relevant {
            found.push(rel_str);
        }
        if found.len() >= 6 {
            break;
        }
    }
    found
}

// ── render_file_numbered ──────────────────────────────────────────────────────
//
// Render a loaded file with REAL line numbers (so `replace_lines` can address it).
// The numbers MUST be the file's true line numbers — small files render whole;
// large files show real-numbered windows (head + keyword regions + tail) with
// explicit "lines X-Y omitted" markers, never a renumbered reassembly.

/// Show files up to this many lines in full; window beyond it (head + keyword
/// regions + tail, with REAL line numbers). Generous — small models comprehend a
/// whole moderate file better than a stitched window, and a cold call on a few-K
/// prompt is only ~15s on a local GPU. `replace_lines` addresses by real number.
const FULL_RENDER_MAX_LINES: usize = 400;

fn render_file_numbered(contents: &str, keywords: &[&str]) -> String {
    let total = contents.lines().count();
    if total == 0 {
        return String::new();
    }
    if total <= FULL_RENDER_MAX_LINES {
        return shunt_edit::numbered(contents);
    }

    // Collect windows of interest, each as a (start, end) of REAL line numbers.
    let mut ranges: Vec<(usize, usize)> = vec![(1, 30.min(total))];
    for kw in keywords {
        let kwl = kw.to_ascii_lowercase();
        for (i, line) in contents.lines().enumerate() {
            if line.to_ascii_lowercase().contains(&kwl) {
                let ln = i + 1;
                ranges.push((ln.saturating_sub(4).max(1), (ln + 4).min(total)));
            }
        }
    }
    ranges.push((total.saturating_sub(9).max(1), total));
    ranges.sort_unstable();

    // Merge overlapping/adjacent windows.
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in ranges {
        match merged.last_mut() {
            Some(last) if s <= last.1 + 1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }

    let mut out = String::new();
    let mut prev_end = 0;
    for (s, e) in merged {
        if s > prev_end + 1 {
            out.push_str(&format!(
                "   … (lines {}-{} omitted) …\n",
                prev_end + 1,
                s - 1
            ));
        }
        out.push_str(&shunt_edit::numbered_window(contents, s, e));
        prev_end = e;
    }
    if prev_end < total {
        out.push_str(&format!(
            "   … (lines {}-{} omitted) …\n",
            prev_end + 1,
            total
        ));
    }
    out
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn action_summary(action: &AgentAction) -> String {
    match action.tool.as_str() {
        "replace_lines" => format!(
            "{}:{}-{}",
            action.path.as_deref().unwrap_or("?"),
            action.start_line.unwrap_or(0),
            action.end_line.unwrap_or(0),
        ),
        "apply_patch" => format!(
            "{} ({} lines)",
            action.path.as_deref().unwrap_or("?"),
            action.patch.as_deref().unwrap_or("").lines().count()
        ),
        "str_replace" => action.path.as_deref().unwrap_or("?").to_string(),
        "write_file" | "delete_file" | "read_file" => {
            action.path.as_deref().unwrap_or("?").to_string()
        }
        "think" => {
            let q = action.query.as_deref().unwrap_or("?");
            let truncated: String = q.chars().take(60).collect();
            if q.chars().count() > 60 {
                format!("{truncated}…")
            } else {
                truncated
            }
        }
        "search_files" => action.query.as_deref().unwrap_or("?").to_string(),
        "command" => command_action_summary(action),
        "run_command_line" => action.command_line.as_deref().unwrap_or("?").to_string(),
        "install_dependencies" => action.path.as_deref().unwrap_or(".").to_string(),
        "start_service" => format!(
            "{}: {}",
            action.service_name.as_deref().unwrap_or("?"),
            action.command_line.as_deref().unwrap_or("?")
        ),
        "status_service" | "stop_service" => {
            action.service_name.as_deref().unwrap_or("?").to_string()
        }
        "run_command" => {
            // Show "cmd args[0]" as the summary
            let cmd = action
                .cmd
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| action.args.first().map(|s| s.as_str()).unwrap_or("?"));
            let first_arg = action.args.first().map(|s| s.as_str()).unwrap_or("");
            if first_arg.is_empty() || action.cmd.as_deref().unwrap_or("").is_empty() {
                cmd.to_string()
            } else {
                format!("{cmd} {first_arg}")
            }
        }
        "ask_user" => action.question.as_deref().unwrap_or("?").to_string(),
        "sub_agent" => action.task.as_deref().unwrap_or("?").to_string(),
        "done" => action.description.as_deref().unwrap_or("done").to_string(),
        other => other.to_string(),
    }
}

fn command_action_summary(action: &AgentAction) -> String {
    match action.command_action.as_deref().unwrap_or("") {
        "run" => action
            .command_line
            .as_deref()
            .map(str::to_string)
            .unwrap_or_else(|| {
                let cmd = action.cmd.as_deref().unwrap_or("?");
                if action.args.is_empty() {
                    cmd.to_string()
                } else {
                    format!("{} {}", cmd, action.args.join(" "))
                }
            }),
        "install_dependencies" => format!("install {}", action.path.as_deref().unwrap_or(".")),
        "start_service" => format!(
            "start {}: {}",
            action.service_name.as_deref().unwrap_or("?"),
            action.command_line.as_deref().unwrap_or("?")
        ),
        "status_service" => format!("status {}", action.service_name.as_deref().unwrap_or("?")),
        "stop_service" => format!("stop {}", action.service_name.as_deref().unwrap_or("?")),
        other => other.to_string(),
    }
}

/// Strip leading/trailing markdown code fences that models sometimes add to plain-text output.
fn strip_code_fences(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.is_empty() {
        return s.to_string();
    }
    let start = if lines[0].starts_with("```") { 1 } else { 0 };
    let end = if lines.len() > 1 && lines[lines.len() - 1].starts_with("```") {
        lines.len() - 1
    } else {
        lines.len()
    };
    lines[start..end].join("\n")
}

fn tail_utf8(bytes: &[u8], max: usize) -> String {
    let s = String::from_utf8_lossy(bytes);
    if s.len() <= max {
        s.into_owned()
    } else {
        format!("…{}", &s[s.len() - max..])
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct DummyProvider;

    impl ToolProvider for DummyProvider {
        fn call_tool(
            &self,
            _system: &str,
            _user: &str,
            _tool: &crate::ToolSpec,
        ) -> crate::InferResult<crate::ToolCall> {
            panic!("DummyProvider should not be called by these tests")
        }
    }

    #[test]
    fn action_envelope_accepts_direct_action() {
        let action: AgentActionEnvelope = serde_json::from_value(serde_json::json!({
            "tool": "read_file",
            "path": "src/lib.rs"
        }))
        .unwrap();
        let action = action.into_action();
        assert_eq!(action.tool, "read_file");
        assert_eq!(action.path.as_deref(), Some("src/lib.rs"));
    }

    #[test]
    fn action_envelope_accepts_action_wrapper() {
        let action: AgentActionEnvelope = serde_json::from_value(serde_json::json!({
            "action": {"tool": "search_files", "query": "marker"}
        }))
        .unwrap();
        let action = action.into_action();
        assert_eq!(action.tool, "search_files");
        assert_eq!(action.query.as_deref(), Some("marker"));
    }

    #[test]
    fn action_envelope_accepts_actions_array() {
        let action: AgentActionEnvelope = serde_json::from_value(serde_json::json!({
            "actions": [
                {"tool": "think", "context": "inspect first"},
                {"tool": "read_file", "path": "src/lib.rs"}
            ]
        }))
        .unwrap();
        let action = action.into_action();
        assert_eq!(action.tool, "think");
        assert_eq!(action.context.as_deref(), Some("inspect first"));
    }

    #[test]
    fn action_envelope_accepts_command_line_action() {
        let action: AgentActionEnvelope = serde_json::from_value(serde_json::json!({
            "tool": "command",
            "command_action": "run",
            "command_line": "cargo check"
        }))
        .unwrap();
        let action = action.into_action();
        assert_eq!(action.tool, "command");
        assert_eq!(action.command_action.as_deref(), Some("run"));
        assert_eq!(action.command_line.as_deref(), Some("cargo check"));
    }

    #[test]
    fn action_envelope_accepts_service_action() {
        let action: AgentActionEnvelope = serde_json::from_value(serde_json::json!({
            "tool": "command",
            "command_action": "start_service",
            "service_name": "api",
            "command_line": "python3 app.py"
        }))
        .unwrap();
        let action = action.into_action();
        assert_eq!(action.tool, "command");
        assert_eq!(action.command_action.as_deref(), Some("start_service"));
        assert_eq!(action.service_name.as_deref(), Some("api"));
        assert_eq!(action.command_line.as_deref(), Some("python3 app.py"));
    }

    #[test]
    fn action_envelope_accepts_install_dependencies_action() {
        let action: AgentActionEnvelope = serde_json::from_value(serde_json::json!({
            "tool": "command",
            "command_action": "install_dependencies",
            "path": "."
        }))
        .unwrap();
        let action = action.into_action();
        assert_eq!(action.tool, "command");
        assert_eq!(
            action.command_action.as_deref(),
            Some("install_dependencies")
        );
        assert_eq!(action.path.as_deref(), Some("."));
    }

    #[test]
    fn split_command_line_preserves_quoted_arguments() {
        let argv = split_command_line("python3 -m pytest 'tests/a b.py'").unwrap();
        assert_eq!(argv, vec!["python3", "-m", "pytest", "tests/a b.py"]);
    }

    #[test]
    fn split_command_line_rejects_unterminated_quote() {
        let err = split_command_line("python3 -m pytest 'tests").unwrap_err();
        assert!(err.contains("unterminated"), "{err}");
    }

    #[test]
    fn run_command_line_still_uses_safety_policy() {
        let tmp = tempfile::tempdir().unwrap();
        let result = execute_command(
            tmp.path().to_str().unwrap(),
            "sh",
            &["-c".into(), "echo unsafe".into()],
        );
        match result {
            Dispatch::Continue { ok, result } => {
                assert!(!ok);
                assert!(result.contains("requires approval"), "{result}");
            }
            _ => panic!("expected command result"),
        }
    }

    #[test]
    fn command_materializes_session_file_state_before_running() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("value.txt"), "old").unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        session.file_state.insert("value.txt".into(), "new".into());
        session.ops.push(ProposedFileOp::Create {
            path: "value.txt".into(),
            contents: "new".into(),
        });

        let result = session.execute_deduped_command(
            "check:value".into(),
            "python3",
            &[
                "-c".into(),
                "from pathlib import Path; assert Path('value.txt').read_text() == 'new'".into(),
            ],
        );

        match result {
            Dispatch::Continue { ok, result } => assert!(ok, "{result}"),
            _ => panic!("expected command result"),
        }
    }

    #[test]
    fn command_runs_queued_setup_before_verification() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        session.queued_setup_commands.push(ProposedCommand {
            program: "python3".into(),
            args: vec![
                "-c".into(),
                "from pathlib import Path; Path('setup.txt').write_text('ready')".into(),
            ],
        });

        let result = session.execute_deduped_command(
            "check:setup".into(),
            "python3",
            &[
                "-c".into(),
                "from pathlib import Path; assert Path('setup.txt').read_text() == 'ready'".into(),
            ],
        );

        match result {
            Dispatch::Continue { ok, result } => assert!(ok, "{result}"),
            _ => panic!("expected command result"),
        }
        assert!(session.queued_setup_commands.is_empty());
    }

    #[test]
    fn json_edit_validation_rejects_invalid_json() {
        assert!(validate_json_edit("package.json", "{\"dependencies\":{}}").is_ok());
        assert!(validate_json_edit("package.json", "{\"dependencies\":{}").is_err());
        assert!(validate_json_edit("src/app.js", "not json").is_ok());
    }

    #[test]
    fn think_is_rejected_after_successful_edit() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        session.turns.push(AgentTurn {
            tool: "replace_lines".into(),
            result: "OK".into(),
            ok: true,
        });

        let result = session.dispatch(AgentAction {
            tool: "think".into(),
            path: None,
            start_line: None,
            end_line: None,
            patch: None,
            old_str: None,
            new_str: None,
            query: Some("what next".into()),
            command_action: None,
            cmd: None,
            command_line: None,
            service_name: None,
            args: vec![],
            question: None,
            context: None,
            task: None,
            description: None,
            setup_commands: vec![],
        });

        match result {
            Dispatch::Continue { ok, result } => {
                assert!(!ok);
                assert!(result.contains("Do not think again"), "{result}");
            }
            _ => panic!("expected rejected think result"),
        }
    }

    #[test]
    fn think_is_removed_from_schema_after_first_successful_think() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        assert!(session.allow_think());
        session.turns.push(AgentTurn {
            tool: "think".into(),
            result: "plan".into(),
            ok: true,
        });
        assert!(!session.allow_think());

        let schema = agent_schema(0, session.allow_think(), session.allow_search());
        let tools = schema["properties"]["tool"]["enum"].as_array().unwrap();
        assert!(!tools.iter().any(|tool| tool == "think"));
        assert!(tools.iter().any(|tool| tool == "command"));
    }

    #[test]
    fn search_and_query_are_removed_from_schema_after_edit() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        session.turns.push(AgentTurn {
            tool: "replace_lines".into(),
            result: "OK".into(),
            ok: true,
        });

        let schema = agent_schema(0, session.allow_think(), session.allow_search());
        let tools = schema["properties"]["tool"]["enum"].as_array().unwrap();
        assert!(!tools.iter().any(|tool| tool == "search_files"));
        assert!(schema["properties"].get("query").is_none());
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn autocomplete_returns_done_after_final_edit_when_named_files_are_satisfied() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        session.current_task = "Edit src/status.ts".into();
        session.file_state.insert(
            "src/status.ts".into(),
            "export const status = 'locked';\n".into(),
        );
        session.ops.push(ProposedFileOp::Create {
            path: "src/status.ts".into(),
            contents: "export const status = 'locked';\n".into(),
        });
        session.turns.push(AgentTurn {
            tool: "replace_lines".into(),
            result: "OK".into(),
            ok: true,
        });

        assert!(matches!(
            session.try_autocomplete("test"),
            Some(AgentResult::Done { .. })
        ));
    }

    #[test]
    fn nudge_blocks_verification_until_explicit_named_files_are_edited() {
        let mut file_state = HashMap::new();
        file_state.insert("src/main.rs".into(), "fn main() {}\n".into());
        file_state.insert("src/report.rs".into(), "pub struct ReportOptions;\n".into());
        file_state.insert("README.md".into(), "# report-cli\n".into());
        let ops = vec![
            ProposedFileOp::Edit {
                path: "src/main.rs".into(),
                search: "fn main() {}".into(),
                replacement: "fn main() { println!(\"json\"); }".into(),
            },
            ProposedFileOp::Edit {
                path: "src/report.rs".into(),
                search: "ReportOptions".into(),
                replacement: "ReportOptions { json: bool }".into(),
            },
        ];
        let turns = vec![
            AgentTurn {
                tool: "replace_lines".into(),
                result: "OK".into(),
                ok: true,
            },
            AgentTurn {
                tool: "run_command_line".into(),
                result: "cargo check passed".into(),
                ok: true,
            },
        ];

        let nudge = build_nudge(
            "Add a `--json` flag. Parse it in src/main.rs, thread it through ReportOptions in src/report.rs, and update README.md usage.",
            &file_state,
            &ops,
            &turns,
            &SessionBudget::for_model(8192),
        );

        assert!(nudge.contains("README.md"), "{nudge}");
        assert!(!nudge.contains("Build passed"), "{nudge}");
    }

    #[test]
    fn service_lifecycle_starts_reports_and_stops_process() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());

        match session.start_service("worker", "sleep 5") {
            Dispatch::Continue { ok, result } => {
                assert!(ok, "{result}");
                assert!(result.contains("Service 'worker' started"), "{result}");
            }
            _ => panic!("expected service start result"),
        }
        assert!(session.services.contains_key("worker"));

        match session.status_service("worker") {
            Dispatch::Continue { ok, result } => {
                assert!(ok, "{result}");
                assert!(result.contains("running"), "{result}");
            }
            _ => panic!("expected service status result"),
        }

        match session.stop_service("worker") {
            Dispatch::Continue { ok, result } => {
                assert!(ok, "{result}");
                assert!(result.contains("Service 'worker'"), "{result}");
            }
            _ => panic!("expected service stop result"),
        }
        assert!(!session.services.contains_key("worker"));
    }

    #[test]
    fn start_service_still_uses_safety_policy() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());

        match session.start_service("bad", "sh -c 'sleep 1'") {
            Dispatch::Continue { ok, result } => {
                assert!(!ok);
                assert!(result.contains("requires approval"), "{result}");
            }
            _ => panic!("expected service start result"),
        }
        assert!(session.services.is_empty());
    }

    #[test]
    fn install_dependencies_queues_pnpm_install_from_lockfile() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("package.json"), "{}\n").unwrap();
        std::fs::write(
            tmp.path().join("pnpm-lock.yaml"),
            "lockfileVersion: '9.0'\n",
        )
        .unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());

        match session.queue_install_dependencies(None) {
            Dispatch::Continue { ok, result } => {
                assert!(ok, "{result}");
                assert!(result.contains("pnpm install"), "{result}");
            }
            _ => panic!("expected install queue result"),
        }

        assert_eq!(
            session.queued_setup_commands,
            vec![ProposedCommand {
                program: "pnpm".into(),
                args: vec!["install".into()],
            }]
        );
    }

    #[test]
    fn install_dependencies_deduplicates_identical_command() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("requirements.txt"), "requests\n").unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());

        let _ = session.queue_install_dependencies(None);
        let second = session.queue_install_dependencies(Some("requirements.txt"));
        match second {
            Dispatch::Continue { ok, result } => {
                assert!(ok, "{result}");
                assert!(result.contains("already queued"), "{result}");
            }
            _ => panic!("expected install queue result"),
        }
        assert_eq!(session.queued_setup_commands.len(), 1);
    }

    #[test]
    fn install_dependencies_rejects_subdirectory_for_now() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("packages/web")).unwrap();
        std::fs::write(tmp.path().join("packages/web/package.json"), "{}\n").unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());

        match session.queue_install_dependencies(Some("packages/web")) {
            Dispatch::Continue { ok, result } => {
                assert!(!ok);
                assert!(result.contains("workspace root"), "{result}");
            }
            _ => panic!("expected install queue result"),
        }
        assert!(session.queued_setup_commands.is_empty());
    }

    #[test]
    fn continuation_includes_environment_profile() {
        let msg = build_continuation_msg(
            "inspect the project",
            &HashMap::new(),
            &[],
            &[],
            "/tmp/example-workspace",
            &SessionBudget::default(),
        );
        assert!(msg.contains("ENVIRONMENT:"));
        assert!(msg.contains("workspace_root: /tmp/example-workspace"));
    }

    #[test]
    fn wall_timeout_aborts_before_first_model_call() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap())
            .with_wall_timeout(Duration::from_millis(0));

        let result = session.run("inspect the project");
        assert!(matches!(result, AgentResult::MaxTurnsReached));
    }

    #[test]
    fn repair_turn_extension_activates_after_edit_and_command() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session =
            AgentSession::new(&provider, tmp.path().to_str().unwrap()).with_budget(SessionBudget {
                max_turns: 12,
                stall_warn_at: 3,
                stall_abort_at: 5,
            });

        assert_eq!(session.effective_max_turns(), 12);
        session.turns.push(AgentTurn {
            tool: "replace_lines".into(),
            result: "ok".into(),
            ok: true,
        });
        assert_eq!(session.effective_max_turns(), 12);
        session.turns.push(AgentTurn {
            tool: "run_command_line".into(),
            result: "Error: failing test".into(),
            ok: false,
        });
        assert_eq!(session.effective_max_turns(), 12 + REPAIR_TURN_EXTENSION);
    }

    #[test]
    fn overlap_expansion_consumes_duplicated_tail_statements() {
        let current = "fn main() {\n    let args = std::env::args();\n    let verbose = true;\n    let options = build(verbose);\n    println!(\"{}\", options);\n}\n";
        let new_content = "    let args = std::env::args();\n    let verbose = true;\n    let is_json = true;\n    let options = build(verbose, is_json);";

        let expanded = expand_replace_end_for_overlap(current, 2, 2, new_content);
        assert_eq!(expanded, 4);
    }

    #[test]
    fn overlap_expansion_consumes_duplicated_tail_declaration_block() {
        let current = "async function fetchJson(url: string): Promise<any | null> {\n  try {\n    const res = await fetch(url);\n    return await res.json();\n  } catch {\n    return null;\n  }\n}\n\nexport async function loadUser(id: string) {\n  const data = await fetchJson(`/users/${id}`);\n  return data;\n}\n";
        let new_content = "async function fetchJson(url: string): Promise<any> {\n  const res = await fetch(url);\n  return await res.json();\n}\n\nexport async function loadUser(id: string) {\n  try {\n    const data = await fetchJson(`/users/${id}`);";

        let expanded = expand_replace_end_for_overlap(current, 1, 8, new_content);
        assert_eq!(expanded, 13);
    }

    #[test]
    fn structural_warnings_detect_duplicate_declarations_and_unbalanced_braces() {
        let content =
            "export function loadUser() {\n  try {\n}\n\nexport function loadUser() {\n}\n";
        let warnings = structural_warnings(content);
        assert!(
            warnings.iter().any(|warning| warning.contains("unmatched")),
            "{warnings:?}"
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("duplicate declaration 'function loadUser'")),
            "{warnings:?}"
        );
    }

    #[test]
    fn exact_overlap_detects_reproduced_following_lines() {
        let overlap = exact_line_overlap(&["alpha", "beta", "gamma"], &["beta", "gamma", "delta"]);
        assert_eq!(overlap, 2);
    }

    #[test]
    fn apply_str_replace_ok() {
        let mut state: HashMap<String, String> = HashMap::new();
        state.insert("f.txt".into(), "hello world\n".into());
        let r = apply_str_replace(&mut state, "f.txt", "hello", "hi");
        assert_eq!(r, "OK");
        assert_eq!(state["f.txt"], "hi world\n");
    }

    #[test]
    fn apply_str_replace_not_found() {
        let mut state: HashMap<String, String> = HashMap::new();
        state.insert("f.txt".into(), "hello\n".into());
        let r = apply_str_replace(&mut state, "f.txt", "bye", "hi");
        assert!(r.starts_with("Error: old_str not found"), "{r}");
    }

    #[test]
    fn apply_str_replace_ambiguous() {
        let mut state: HashMap<String, String> = HashMap::new();
        state.insert("f.txt".into(), "x\nx\n".into());
        let r = apply_str_replace(&mut state, "f.txt", "x", "y");
        assert!(r.contains("2 times"), "{r}");
    }

    #[test]
    fn apply_str_replace_missing_file() {
        let mut state: HashMap<String, String> = HashMap::new();
        let r = apply_str_replace(&mut state, "missing.txt", "a", "b");
        assert!(r.starts_with("Error: file"), "{r}");
    }

    #[test]
    fn apply_str_replace_noop() {
        let mut state: HashMap<String, String> = HashMap::new();
        state.insert("f.txt".into(), "hello\n".into());
        let r = apply_str_replace(&mut state, "f.txt", "hello", "hello");
        assert!(r.contains("identical"), "{r}");
    }

    // ── resolve_in_workspace security tests ──────────────────────────────────

    #[test]
    fn rejects_absolute_path_outside_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_str().unwrap();
        let err = resolve_in_workspace(root, "/etc/passwd").unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }

    #[test]
    fn allows_absolute_path_inside_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let absolute = tmp.path().join("src/main.rs");
        let root = tmp.path().to_str().unwrap();
        let p = resolve_in_workspace(root, absolute.to_str().unwrap()).unwrap();
        assert!(p.starts_with(tmp.path()));
        assert!(p.ends_with("src/main.rs"));
    }

    #[test]
    fn rejects_dotdot_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_str().unwrap();
        let err = resolve_in_workspace(root, "../../etc/passwd").unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }

    #[test]
    fn allows_normal_relative_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_str().unwrap();
        let p = resolve_in_workspace(root, "src/main.rs").unwrap();
        assert!(p.starts_with(tmp.path()));
    }

    #[test]
    fn allows_existing_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("foo.txt"), "x").unwrap();
        let root = tmp.path().to_str().unwrap();
        let p = resolve_in_workspace(root, "foo.txt").unwrap();
        assert!(p.starts_with(tmp.path()));
        assert!(p.ends_with("foo.txt"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        // Create workspace/out -> /tmp/other (outside workspace)
        std::os::unix::fs::symlink(target.path(), tmp.path().join("out")).unwrap();
        let root = tmp.path().to_str().unwrap();
        let err = resolve_in_workspace(root, "out/secret.txt").unwrap_err();
        assert!(err.contains("escapes"), "{err}");
    }
}
