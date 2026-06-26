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
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::{ProposedCommand, ProposedFileOp, SourceFileContext, ToolProvider};
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
- Spend think on HOW to make the edit, not on what else to look for. You have at most 3 thinks — don't burn them exploring.
- Create new files BEFORE updating registrations or imports that reference them.
- After all edits look correct in the file view: call done immediately. Do NOT run commands just to confirm the edit landed — the file view IS the confirmation.
- Only run a build command if (a) you can see a Cargo.toml or package.json in the workspace AND (b) the task involves code that could have type or syntax errors not visible from reading the file.
  Node/TypeScript → run_command {\"cmd\":\"pnpm\",\"args\":[\"build\"]}
  Rust            → run_command {\"cmd\":\"cargo\",\"args\":[\"check\"]}
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
  Max 3 think calls per session. No two thinks in a row — act between them.
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

• run_command — Run a command and capture output. cmd is the program; args are its arguments.
  Required: cmd (string), args (array of strings)
  Do NOT use for installs — put those in done.setup_commands.

• ask_user — Ask the user when genuinely blocked or when surfacing a better approach.
  Required: question (string), context (string)

• sub_agent — Spawn a focused sub-session for a bounded task.
  Required: task (string), context (string)

• done — Mark work complete.
  Required: description (string — what changed and what was verified)
  Optional: setup_commands (array of {program, args} objects)";

// ── Tool dispatch schema ───────────────────────────────────────────────────────

fn agent_schema(depth: u8) -> Value {
    let mut tools: Vec<&str> = vec![
        "think",
        "write_file",
        "replace_lines",
        "delete_file",
        "read_file",
        "search_files",
        "run_command",
        "done",
    ];
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
    json!({
        "type": "object",
        "properties": {
            "tool":        { "type": "string", "enum": tools },
            "path":        { "type": "string" },
            // replace_lines positioning — integers only (content is collected
            // separately via generate_text; small models can't produce content
            // inside grammar-constrained JSON).
            "start_line":  { "type": "integer" },
            "end_line":    { "type": "integer" },
            "query":       { "type": "string" },
            "cmd":         { "type": "string" },
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
    })
}

// Verifier is read-only: write/str_replace/delete are excluded from the grammar
// so the model physically cannot generate them even if it ignores the prompt.
fn verifier_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "tool":        { "type": "string", "enum": ["think", "read_file", "search_files", "run_command", "done"] },
            "path":        { "type": "string" },
            "query":       { "type": "string" },
            "cmd":         { "type": "string" },
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
    cmd: Option<String>,
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
                    cmd: None,
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

/// Resolve a model-supplied relative path to an absolute canonical path,
/// rejecting anything that escapes the workspace root.
///
/// Handles:
/// - Absolute paths (rejected outright)
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
    if Path::new(rel).is_absolute() {
        return Err(format!("absolute paths are not allowed: '{rel}'"));
    }

    let root = Path::new(workspace_root)
        .canonicalize()
        .map_err(|e| format!("workspace root invalid: {e}"))?;

    let raw = root.join(rel);

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
    observer: Option<Arc<dyn AgentObserver>>,
    /// Original task — stored so dispatch can use it in two-phase generation.
    current_task: String,
    /// Paths successfully written in this session — guards against infinite write loops.
    written_paths: HashSet<String>,
    /// (path, start_line) of ranges already replaced this session — guards against
    /// re-editing the same spot in a loop (the post-edit repetition failure mode).
    /// (path, start_line, new_content) triples already applied — guards against
    /// re-applying the exact same replacement in a loop while allowing corrections.
    edited_lines: HashSet<(String, usize, String)>,
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
            observer: None,
            current_task: String::new(),
            written_paths: HashSet::new(),
            edited_lines: HashSet::new(),
            extra_ignore_patterns: Vec::new(),
            conv_history: Vec::new(),
            cold_entries: Vec::new(),
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
            observer: None,
            current_task: String::new(),
            written_paths: HashSet::new(),
            edited_lines: HashSet::new(),
            extra_ignore_patterns: Vec::new(),
            conv_history: Vec::new(),
            cold_entries: Vec::new(),
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
            observer: None,
            current_task: String::new(),
            written_paths: HashSet::new(),
            edited_lines: HashSet::new(),
            extra_ignore_patterns: Vec::new(),
            conv_history: Vec::new(),
            cold_entries: Vec::new(),
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
        session.run_inner("(continuing after user answered)")
    }

    /// Start (or continue) the agent on `task`.
    pub fn run(&mut self, task: &str) -> AgentResult {
        self.run_inner(task)
    }

    fn run_inner(&mut self, task: &str) -> AgentResult {
        self.current_task = task.to_string();
        let schema = if self.is_verifier {
            verifier_schema()
        } else {
            agent_schema(self.depth)
        };

        // Stable prefix shared across all turns — forms the KV-cached root.
        let system_msg = crate::ChatMessage {
            role: "system".into(),
            content: self.system_prompt.clone(),
        };
        let task_frame = crate::ChatMessage {
            role: "user".into(),
            content: format!("TASK: {task}"),
        };

        for turn_idx in 0..self.budget.max_turns {
            // Evict old turn history if approaching the token budget.
            evict_history_if_needed(&mut self.conv_history, &mut self.cold_entries, &self.turns);

            // Build the ephemeral continuation message: current FILES state + nudge.
            // This is NOT stored in conv_history — it's rebuilt fresh every turn so
            // the model always sees current line numbers and a correct nudge.
            let continuation = build_continuation_msg(
                task,
                &self.file_state,
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

            // Retry up to 3 times on transient LLM errors (network blips, rate limits).
            let action: AgentAction = {
                let mut last_err = None;
                let mut result = None;
                for attempt in 0..3u8 {
                    match self.provider.generate_from_messages::<AgentActionEnvelope>(
                        "agent_action",
                        &inference_messages,
                        &schema,
                    ) {
                        Ok(a) => {
                            result = Some(a.into_action());
                            break;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "AgentSession turn {turn_idx} attempt {attempt}: generate_from_messages failed: {e}"
                            );
                            last_err = Some(e);
                            std::thread::sleep(std::time::Duration::from_secs(
                                2u64.pow(attempt as u32),
                            ));
                        }
                    }
                }
                match result {
                    Some(a) => a,
                    None => {
                        let err_msg =
                            format!("all 3 retries exhausted on turn {turn_idx}: {last_err:?}");
                        tracing::error!("AgentSession {err_msg}");
                        if let Some(o) = &self.observer {
                            o.on_note(&format!("ERROR: {err_msg}"));
                        }
                        if !self.ops.is_empty() {
                            tracing::warn!(
                                "AgentSession: returning partial ops ({} ops) as implicit done",
                                self.ops.len()
                            );
                            return AgentResult::Done {
                                ops: std::mem::take(&mut self.ops),
                                setup_commands: vec![],
                                description:
                                    "Applied changes (agent hit retry limit after writing files)"
                                        .into(),
                                file_state: self.file_state.clone(),
                            };
                        }
                        return AgentResult::MaxTurnsReached;
                    }
                }
            };

            let tool = action.tool.clone();
            let summary = action_summary(&action);

            // Capture compact JSON of the action BEFORE dispatch (borrows action).
            let action_json = action_to_compact_json(&action);

            let obs = self.observer.clone();
            if let Some(o) = &obs {
                o.on_tool_call(turn_idx, self.budget.max_turns, &tool, &summary);
            }

            match self.dispatch(action) {
                Dispatch::Done {
                    description,
                    setup_commands,
                } => {
                    if let Some(o) = &obs {
                        o.on_tool_result(turn_idx, true, &description);
                    }
                    // Record final turn in history.
                    self.conv_history.push(("assistant".into(), action_json));
                    self.conv_history.push(("user".into(), "done".into()));
                    self.turns.push(AgentTurn {
                        tool,
                        result: description.clone(),
                        ok: true,
                    });
                    return AgentResult::Done {
                        ops: std::mem::take(&mut self.ops),
                        setup_commands,
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
                    return AgentResult::NeedsClarification {
                        question,
                        context,
                        turns: self.turns.clone(),
                        file_state: self.file_state.clone(),
                        partial_ops: self.ops.clone(),
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
                    if idle_streak >= self.budget.stall_abort_at {
                        tracing::warn!(
                            "AgentSession stall: {idle_streak} consecutive idle turns — aborting"
                        );
                        if !self.ops.is_empty() {
                            return AgentResult::Done {
                                ops: std::mem::take(&mut self.ops),
                                setup_commands: vec![],
                                description: "Applied changes (stall detected — too many reads without edits)".into(),
                                file_state: self.file_state.clone(),
                            };
                        }
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
                        return AgentResult::MaxTurnsReached;
                    }
                }
            }
        }

        // Turn limit reached. Commit any accumulated edits rather than discarding.
        if !self.ops.is_empty() {
            return AgentResult::Done {
                ops: std::mem::take(&mut self.ops),
                setup_commands: vec![],
                description: "Applied changes (reached the turn limit before calling done)".into(),
                file_state: self.file_state.clone(),
            };
        }
        AgentResult::MaxTurnsReached
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
                if think_count >= 3 {
                    return Dispatch::Continue {
                        result: "Error: you have already thought 3 times. \
                                 No more think calls allowed. Act now — \
                                 call read_file, write_file, replace_lines, run_command, or done."
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
                                 write_file, replace_lines, run_command, or done. \
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

                // Anti-thrash: reject the exact same (line, content) pair twice.
                // A different new_content on the same line is a legitimate correction.
                let edit_key = (path.clone(), start, new_content.clone());
                if self.edited_lines.contains(&edit_key) {
                    let current_view =
                        shunt_edit::numbered_window(&current, 1, current.lines().count());
                    return Dispatch::Continue {
                        result: format!(
                            "Error: duplicate edit — lines {start}-{end} of '{path}' \
                             already contain exactly what you wrote. The file is correct. \
                             Current file:\n{current_view}\n\
                             STOP repeating this edit. Your only valid next action is done."
                        ),
                        ok: false,
                    };
                }

                // Reassemble the file deterministically via shunt-edit.
                let edit = if is_append {
                    shunt_edit::Edit::InsertAfter {
                        after: total_lines,
                        new_text: new_content.clone(),
                    }
                } else {
                    shunt_edit::Edit::ReplaceLines {
                        start,
                        end,
                        new_text: new_content.clone(),
                    }
                };
                match shunt_edit::apply(&current, &edit) {
                    Ok(updated) => {
                        self.file_state.insert(path.clone(), updated.clone());
                        let updated_view =
                            shunt_edit::numbered_window(&updated, 1, updated.lines().count());
                        self.ops.push(ProposedFileOp::Create {
                            path: path.clone(),
                            contents: updated,
                        });
                        self.written_paths.insert(path.clone());
                        self.edited_lines.insert(edit_key);
                        let msg = if is_append {
                            format!(
                                "OK — appended to '{path}'. Current file:\n{updated_view}\n\
                                 Edit applied. Call done now, or replace_lines on a \
                                 different file/range if more changes are needed."
                            )
                        } else {
                            format!(
                                "OK — replaced lines {start}-{end} in '{path}'. Current file:\n\
                                 {updated_view}\n\
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
                // Apply the shared safety classifier (shunt_core::safety).
                // This covers blocked programs, dangerous operations, AND unwraps
                // sh -c wrappers so inner commands are also classified.
                let spec = shunt_core::CommandSpec::new(cmd, tail.iter().map(String::as_str));
                match safety::classify(&spec) {
                    safety::CommandSafety::Blocked { reason } => {
                        return Dispatch::Continue {
                            result: format!("Error: command blocked — {reason}"),
                            ok: false,
                        };
                    }
                    safety::CommandSafety::Dangerous { reason } => {
                        // In the agent context there is no user approval gate,
                        // so dangerous commands are also blocked.
                        return Dispatch::Continue {
                            result: format!(
                                "Error: command requires approval — {reason}. \
                                 Use done.setup_commands for commands that need user review."
                            ),
                            ok: false,
                        };
                    }
                    safety::CommandSafety::Safe => {}
                }
                // Execvp-style: cmd is the program, tail are its arguments.
                // Stdout/stderr drained on threads to prevent pipe-buffer deadlock.
                // Hard timeout kills the child so a hung build doesn't block forever.
                const AGENT_CMD_TIMEOUT_SECS: u64 = 60;
                let mut child = match std::process::Command::new(cmd)
                    .args(tail)
                    .current_dir(&self.workspace_root)
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
                let deadline = std::time::Instant::now()
                    + std::time::Duration::from_secs(AGENT_CMD_TIMEOUT_SECS);
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
                    AgentResult::Done { description, .. } => description,
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
                     replace_lines, delete_file, run_command, ask_user, sub_agent, done. \
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
   Node/TypeScript → run_command {\"cmd\":\"pnpm\",\"args\":[\"build\"]}
   Rust            → run_command {\"cmd\":\"cargo\",\"args\":[\"check\"]}
3. Run surgical smoke tests matched to what changed:
   - New HTTP route → start dev server, curl the endpoint, kill the server
   - New component   → inspect build output for warnings / missing exports
   - Modified logic  → run the existing test suite if one is present
4. Call done with EXACTLY this format in description:
   PASS: <what was tested and confirmed working>
   — or —
   FAIL: <specific what broke, exact error output>

CONSTRAINTS:
- Never call write_file, str_replace, or delete_file — read-only.
- To start a dev server temporarily: run_command {\"cmd\":\"timeout\",\"args\":[\"8\",\"pnpm\",\"dev\"]}
- Keep scope surgical — only test what the changes touch.
- If the build fails, stop there and report FAIL with the error.

Available tools: think, read_file, search_files, run_command, done";

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
    turns: &[AgentTurn],
    workspace_root: &str,
    budget: &SessionBudget,
) -> String {
    let kw_refs = task_keywords(task);
    let kw_refs: Vec<&str> = kw_refs.iter().map(String::as_str).collect();

    let mut msg = String::new();

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
    msg.push_str(&build_nudge(turns, budget));
    msg
}

/// Build the contextual nudge text appended to the ephemeral continuation message.
fn build_nudge(turns: &[AgentTurn], budget: &SessionBudget) -> String {
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
    let last_verified = turns.iter().rev().find(|t| t.tool == "run_command");
    let last_verified_ok = last_verified
        .map(|t| {
            t.ok && !t.result.to_ascii_lowercase().contains("error")
                && !t.result.to_ascii_lowercase().contains("failed")
        })
        .unwrap_or(false);
    let last_action = turns.last().map(|t| t.tool.as_str()).unwrap_or("");
    let last_action_ok = turns.last().map(|t| t.ok).unwrap_or(false);

    if idle_streak >= budget.stall_warn_at {
        format!(
            "\nSTALL WARNING: {idle_streak} turns of reading/thinking without any edit or command. \
             You MUST call write_file, replace_lines, run_command, or done NOW. \
             No more reads or thinks. ({remaining} turns remaining)"
        )
    } else if last_action == "done" {
        String::new()
    } else if wrote_any && last_action == "run_command" && last_verified_ok {
        format!(
            "\nBuild passed. Call done with a description of what changed and what was verified. \
             ({remaining} turns remaining)"
        )
    } else if wrote_any && last_action == "run_command" && !last_verified_ok {
        format!(
            "\nBuild failed. Use think to reason about the root cause, then use replace_lines \
             to fix the specific error. Re-run the build after fixing. \
             ({remaining} turns remaining)"
        )
    } else if wrote_any && last_action != "run_command" {
        format!(
            "\nFiles written. Run the build now to verify your changes work before calling done. \
             ({remaining} turns remaining)"
        )
    } else if last_action == "think" && last_action_ok {
        format!(
            "\nThought recorded. Your NEXT action MUST NOT be think — \
             call read_file, write_file, replace_lines, run_command, or done now. \
             ({remaining} turns remaining)"
        )
    } else {
        format!("\nContinue. ({remaining} turns remaining)")
    }
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
    if let Some(cmd) = &action.cmd {
        map.insert("cmd".into(), serde_json::json!(cmd));
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
///   3. `run_command` success  → compress to "[build: OK]"
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
        "run_command" => {
            if content.len() <= 120 {
                return content.to_string();
            }
            // Keep last 120 chars (most relevant: errors are at the end).
            let tail = &content[content.len() - 120..];
            format!("[command output truncated]\n…{tail}")
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
    fn rejects_absolute_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_str().unwrap();
        let err = resolve_in_workspace(root, "/etc/passwd").unwrap_err();
        assert!(err.contains("absolute"), "{err}");
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
