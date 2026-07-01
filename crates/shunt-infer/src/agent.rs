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
//! - `knowledge` queries an injected `KnowledgeLookup` backend for dependency/version evidence.
//! - `search_files` uses the shunt-localize search index when available; falls back
//!   to a simple directory walk otherwise.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    ProposedCommand, ProposedFileOp, SourceFileContext, ToolChoiceMode, ToolProvider, ToolSpec,
};
use shunt_core::safety;

// ── System prompt ──────────────────────────────────────────────────────────────
// Externalized to prompts/agent.system.txt so it can be A/B'd and regression-tested
// independently of the dispatch code (matching the other prompts/ files).

pub(crate) const AGENT_SYSTEM_PROMPT_BASE: &str = include_str!("../../../prompts/agent.system.txt");

// ── Tool registry ────────────────────────────────────────────────────────────
// Single source of truth for the action JSON-schema `tool` enum AND the system-prompt
// tool reference (the `{{TOOLS}}` placeholder). Deriving both from one list means the
// grammar the model is constrained to and the prose it is told about cannot drift.

struct ToolDef {
    name: &'static str,
    /// Offered in the agent (edit-capable) session.
    agent: bool,
    /// Offered in the read-only verifier session.
    verifier: bool,
    /// Only offered at depth 0 (top-level session) — e.g. ask_user.
    depth0_only: bool,
    /// Only offered when a knowledge-lookup backend is wired into the session.
    needs_knowledge: bool,
    /// System-prompt reference block (the `• name — …` text).
    doc: &'static str,
}

const TOOLS: &[ToolDef] = &[
    ToolDef {
        name: "read_file",
        agent: true,
        verifier: true,
        depth0_only: false,
        needs_knowledge: false,
        doc: "• read_file — Load a file into context (returned with line numbers).\n  Required: path (string)",
    },
    ToolDef {
        name: "search_files",
        agent: true,
        verifier: true,
        depth0_only: false,
        needs_knowledge: false,
        doc: "• search_files — Search file contents and paths by keyword or symbol. Empty query = list all.\n  Required: query (string)",
    },
    ToolDef {
        name: "knowledge",
        agent: true,
        verifier: false,
        depth0_only: false,
        needs_knowledge: true,
        doc: "• knowledge — Look up external facts on demand: current package versions, library APIs, and recommended practices from registries and docs. Use when the task needs up-to-date ecosystem knowledge you don't have locally.\n  Required: query (string)",
    },
    ToolDef {
        name: "edit",
        agent: true,
        verifier: false,
        depth0_only: false,
        needs_knowledge: false,
        doc: "• edit — Create a new file or modify an existing one. Required: path.\n  • New file: OMIT start_line; content is the whole file. Errors if the file already exists.\n  • Modify: read the file first, then give start_line/end_line (1-indexed, inclusive) of the range to replace. One line: start_line = end_line. Append: start_line = last_line + 1. Delete: give the range and omit content.",
    },
    ToolDef {
        name: "command",
        agent: true,
        verifier: true,
        depth0_only: false,
        needs_knowledge: false,
        doc: "• command — Run a command and see its output. Runs synchronously; use it for installs, scaffolding, builds, tests, and any other finite command.\n  Required: command_line (string): the full command exactly as you would type it in a terminal, e.g. \"npm run build\" or \"npm create vite@latest frontend -- --template react-ts\". Quote arguments containing spaces.\n  It is NOT run through a shell. Use finite commands only — not dev servers, watchers, or other long-running processes (there is no way to keep a process running past the command).\n  Commands run with stdin closed. Use standalone, non-interactive invocations and pass whichever flag the tool uses to suppress prompts (e.g. --yes, -y, --no-input, --force, --non-interactive).\n  Never use shell operators (&&, ||, ;, |, >) — one command_line per call. Never run a bare interactive program (node, python, bash, npm) with no arguments.\n  To run in a subdirectory use cwd (relative path within workspace), not \"cd X && Y\". Example: {\"command_line\":\"npm install\",\"cwd\":\"frontend\"}\n  Examples: {\"command_line\":\"node -v\"}\n            {\"command_line\":\"npm run build\",\"cwd\":\"frontend\"}\n            {\"command_line\":\"npm create vite@latest . -- --template react-ts\"}",
    },
    ToolDef {
        name: "ask_user",
        agent: true,
        verifier: false,
        depth0_only: true,
        needs_knowledge: false,
        doc: "• ask_user — Ask the user when genuinely blocked or when surfacing a better approach.\n  Required: question (string), context (string)",
    },
    ToolDef {
        name: "done",
        agent: true,
        verifier: true,
        depth0_only: false,
        needs_knowledge: false,
        doc: "• done — Mark work complete.\n  Required: description (string — what changed and what was verified)\n  Optional: setup_commands (array of {command_line} objects, each command_line as typed in a terminal)\n  Verification sessions: criteria is required — one entry per acceptance criterion ID given in the task, each {id, status: passed|failed|skipped, evidence}. Do not omit an ID; do not report passed without evidence you actually gathered.",
    },
];

/// Agent tool enum for the given depth. Search and re-reads stay available after edits
/// (recovery is bounded by the stall/loop detectors, not by removing tools). The
/// knowledge tool is only offered when a lookup backend is wired in.
fn agent_tool_names(depth: u8, has_knowledge: bool) -> Vec<&'static str> {
    TOOLS
        .iter()
        .filter(|t| t.agent)
        .filter(|t| !t.depth0_only || depth == 0)
        .filter(|t| !t.needs_knowledge || has_knowledge)
        .map(|t| t.name)
        .collect()
}

fn verifier_tool_names() -> Vec<&'static str> {
    TOOLS
        .iter()
        .filter(|t| t.verifier)
        .map(|t| t.name)
        .collect()
}

/// The `{{TOOLS}}` reference block injected into the system prompt.
fn tools_doc(verifier: bool, has_knowledge: bool) -> String {
    TOOLS
        .iter()
        .filter(|t| if verifier { t.verifier } else { t.agent })
        .filter(|t| !t.needs_knowledge || has_knowledge)
        .map(|t| t.doc)
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ── Edit content strategy ──────────────────────────────────────────────────────
// The edit tool is always single-shot: target selection and replacement content
// travel together in one structured model output. Transport differences stay in
// the provider layer, not in agent behavior.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditStrategy {
    SingleShot,
}

impl EditStrategy {
    fn for_capabilities(_tool_choice_mode: ToolChoiceMode) -> Self {
        Self::SingleShot
    }

    fn content_in_schema(self) -> bool {
        let _ = self;
        true
    }
}

// ── Tool dispatch schema ───────────────────────────────────────────────────────

fn agent_schema(depth: u8, single_shot: bool, has_knowledge: bool) -> Value {
    let tools = agent_tool_names(depth, has_knowledge);
    // Edit content is carried inline in `content`. maxLength / maxItems are omitted
    // intentionally — llama.cpp's grammar FSM is O(n) in maxLength and would stall
    // for minutes compiling the grammar before the first token.
    let mut schema = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "tool":        { "type": "string", "enum": tools },
            "path":        { "type": "string" },
            "start_line":  { "type": "integer" },
            "end_line":    { "type": "integer" },
            "query":       { "type": "string" },
            "command_line": { "type": "string" },
            "cwd":         { "type": "string" },
            "question":    { "type": "string" },
            "context":     { "type": "string" },
            "task":        { "type": "string" },
            "description": { "type": "string" },
            "setup_commands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "command_line": { "type": "string" },
                        "cwd":          { "type": "string" }
                    },
                    "required": ["command_line"]
                }
            }
        },
        "required": ["tool"]
    });
    let _ = single_shot;
    if let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        properties.insert("content".into(), json!({ "type": "string" }));
    }
    schema
}

// Verifier is read-only: write/edit/delete are excluded from the grammar so the model
// physically cannot generate them even if it ignores the prompt.
fn verifier_schema() -> Value {
    let tools = verifier_tool_names();
    json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "tool":        { "type": "string", "enum": tools },
            "path":        { "type": "string" },
            "query":       { "type": "string" },
            "command_line": { "type": "string" },
            "cwd":         { "type": "string" },
            "description": { "type": "string" },
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                        "id":       { "type": "string" },
                        "status":   { "type": "string", "enum": ["passed", "failed", "skipped"] },
                        "evidence": { "type": "string" }
                    },
                    "required": ["id", "status", "evidence"]
                }
            }
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
    /// Single-shot replacement/new-file content (native function-calling models).
    content: Option<String>,
    query: Option<String>,
    command_line: Option<String>,
    cwd: Option<String>,
    question: Option<String>,
    context: Option<String>,
    task: Option<String>,
    description: Option<String>,
    #[serde(default)]
    setup_commands: Vec<SetupCommandSpec>,
    #[serde(default)]
    criteria: Vec<RawCriterion>,
}

/// Wire shape for one `setup_commands` entry: a command line as typed in a
/// terminal, plus an optional working directory. Parsed into a `ProposedCommand`
/// via `split_command_line`.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct SetupCommandSpec {
    command_line: String,
    #[serde(default)]
    cwd: Option<String>,
}

/// Wire shape for one `criteria` entry on the `done` tool — status arrives as the
/// lowercase string the grammar constrains it to, not the `VerifierStatus` variant name.
#[derive(Debug, Deserialize)]
struct RawCriterion {
    id: String,
    status: String,
    evidence: String,
}

impl RawCriterion {
    /// Parse into the domain type. An unrecognised status fails closed — a verifier that
    /// emits garbage here should not be read as having passed.
    fn into_outcome(self) -> CriterionOutcome {
        let status = match self.status.as_str() {
            "passed" => shunt_core::VerifierStatus::Passed,
            "skipped" => shunt_core::VerifierStatus::Skipped,
            _ => shunt_core::VerifierStatus::Failed,
        };
        CriterionOutcome {
            id: self.id,
            status,
            evidence: self.evidence,
        }
    }
}

/// One criterion verdict reported by a verifier session's `done` call.
#[derive(Debug, Clone)]
pub struct CriterionOutcome {
    pub id: String,
    pub status: shunt_core::VerifierStatus,
    pub evidence: String,
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
                    // Empty actions array — emit a no-op read with no path so dispatch
                    // returns an informative "provide a path" error that nudges the model
                    // to choose a single concrete action next turn.
                    tool: "read_file".into(),
                    path: None,
                    start_line: None,
                    end_line: None,
                    content: None,
                    query: None,
                    command_line: None,
                    cwd: None,
                    question: None,
                    context: None,
                    task: None,
                    description: None,
                    setup_commands: vec![],
                    criteria: vec![],
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentPauseState {
    pub task: String,
    pub turns: Vec<AgentTurn>,
    pub file_state: HashMap<String, String>,
    pub partial_ops: Vec<ProposedFileOp>,
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

#[derive(Debug, Clone)]
struct CommandObservation {
    summary: String,
}

#[derive(Debug, Clone)]
struct CommandExpectationState {
    workspace_fingerprint: Option<u64>,
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
        /// Per-criterion verdicts reported via `done`. Empty for non-verifier sessions and for
        /// verifier sessions that never call `done` with `criteria` (treated as inconclusive by
        /// the caller, not as a pass).
        criteria: Vec<CriterionOutcome>,
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

/// On-demand external knowledge lookup (package versions, library APIs, best
/// practices). The agent loop calls this when the model uses the `knowledge` tool.
///
/// Defined here so `shunt-infer` stays independent of `shunt-knowledge`; the runtime
/// injects an implementation backed by `KnowledgeService` via
/// [`AgentSession::with_knowledge`].
pub trait KnowledgeLookup: Send + Sync {
    /// Resolve `query` to a human-readable evidence summary, or an error message.
    fn lookup(&self, query: &str) -> Result<String, String>;
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
        criteria: Vec<CriterionOutcome>,
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
    /// Maximum turns the session may spend without adding new evidence or changing
    /// workspace state. Progress extends the budget window.
    pub max_turns: usize,
    /// Consecutive idle turns (think / read_file / search_files, no edits or commands)
    /// at which a STALL WARNING is injected into the next user prompt.
    pub stall_warn_at: usize,
    /// Legacy idle-abort threshold retained for configuration compatibility. Idle turns
    /// no longer abort directly; the progress budget is the actual stop condition.
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
    observer: Option<Arc<dyn AgentObserver + Send + Sync>>,
    /// Original task — stored so dispatch can use it in two-phase generation.
    current_task: String,
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
    /// Latest command result rendered as a short observation so the next turn can treat
    /// it as fresh evidence instead of blindly repeating the command.
    latest_command: Option<CommandObservation>,
    /// Absolute completed-turn index of the latest turn that changed workspace state or
    /// added genuinely new evidence. The session budget is applied relative to this.
    last_progress_turn: usize,
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
    /// How edit content is produced — capability-gated (single-shot vs two-phase).
    edit_strategy: EditStrategy,
    /// On-demand external knowledge backend; when present the `knowledge` tool is offered.
    knowledge: Option<Arc<dyn KnowledgeLookup>>,
}

impl<'a, P: ToolProvider> AgentSession<'a, P> {
    /// Create a top-level session.  The system prompt is built from the workspace.
    pub fn new(provider: &'a P, workspace_root: &str) -> Self {
        let edit_strategy =
            EditStrategy::for_capabilities(provider.capabilities().tool_choice_mode);
        let system_prompt =
            build_system_prompt(workspace_root, edit_strategy.content_in_schema(), false);
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
            written_paths: HashSet::new(),
            edited_lines: HashSet::new(),
            ineffective_edit_ranges: HashSet::new(),
            latest_command: None,
            last_progress_turn: 0,
            extra_ignore_patterns: Vec::new(),
            conv_history: Vec::new(),
            cold_entries: Vec::new(),
            edit_strategy,
            knowledge: None,
        }
    }

    /// Create a verifier session — QA mindset, read-only, reports PASS/FAIL.
    /// `changed_paths` are the workspace-relative files the builder touched; used to
    /// locate the relevant manifest/project directory instead of assuming the root.
    pub fn new_verifier(provider: &'a P, workspace_root: &str, changed_paths: &[String]) -> Self {
        let system_prompt = build_verifier_prompt(workspace_root, changed_paths);
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
            written_paths: HashSet::new(),
            edited_lines: HashSet::new(),
            ineffective_edit_ranges: HashSet::new(),
            latest_command: None,
            last_progress_turn: 0,
            extra_ignore_patterns: Vec::new(),
            conv_history: Vec::new(),
            cold_entries: Vec::new(),
            // Verifier never edits — keep the same single-shot schema shape.
            edit_strategy: EditStrategy::SingleShot,
            knowledge: None,
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

    pub fn with_observer(mut self, obs: Arc<dyn AgentObserver + Send + Sync>) -> Self {
        self.observer = Some(obs);
        self
    }

    /// Wire in an external knowledge backend, enabling the `knowledge` tool.
    /// Rebuilds the system prompt so the tool reference matches the schema.
    pub fn with_knowledge(mut self, knowledge: Arc<dyn KnowledgeLookup>) -> Self {
        self.knowledge = Some(knowledge);
        if !self.is_verifier {
            self.system_prompt = build_system_prompt(
                &self.workspace_root,
                self.edit_strategy.content_in_schema(),
                self.knowledge.is_some(),
            );
        }
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
        session.last_progress_turn = session.turns.len();
        session.latest_command = latest_command_observation(&session.turns);
        session.run_inner("(continuing after user answered)")
    }

    pub fn resume_paused(
        provider: &'a P,
        workspace_root: &str,
        mut pause: AgentPauseState,
        user_answer: &str,
        budget: SessionBudget,
        observer: Option<Arc<dyn AgentObserver + Send + Sync>>,
        ignore_patterns: Vec<String>,
    ) -> AgentResult {
        if let Some(last) = pause.turns.last_mut()
            && last.tool == "ask_user"
        {
            last.result = format!("User answered: {user_answer}");
        }
        let conv_history: Vec<(String, String)> = pause
            .turns
            .iter()
            .flat_map(|t| {
                let assistant_stub = serde_json::json!({ "tool": t.tool }).to_string();
                [
                    ("assistant".to_string(), assistant_stub),
                    ("user".to_string(), t.result.clone()),
                ]
            })
            .collect();
        let mut session = Self::new(provider, workspace_root).with_budget(budget);
        if let Some(observer) = observer {
            session = session.with_observer(observer);
        }
        session.extra_ignore_patterns = ignore_patterns;
        session.conv_history = conv_history;
        session.turns = pause.turns;
        session.file_state = pause.file_state;
        session.ops = pause.partial_ops;
        session.last_progress_turn = session.turns.len();
        session.latest_command = latest_command_observation(&session.turns);
        session.run_inner(&pause.task)
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

        while self.turns.len() < self.effective_max_turns() {
            let turn_idx = self.turns.len();
            if self.exceeded_wall_timeout(started_at) {
                if let Some(done) = self.try_autocomplete("wall-clock timeout") {
                    return done;
                }
                if let Some(o) = &self.observer {
                    o.on_note("ERROR: agent session wall-clock timeout exceeded");
                }
                return AgentResult::MaxTurnsReached;
            }
            let schema = if self.is_verifier {
                verifier_schema()
            } else {
                agent_schema(
                    self.depth,
                    self.edit_strategy.content_in_schema(),
                    self.knowledge.is_some(),
                )
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
                self.latest_command.as_ref(),
                self.effective_max_turns().saturating_sub(self.turns.len()),
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
                            && is_invalid_action_error(err)
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
                            continue;
                        }
                        let err_msg =
                            format!("all retries exhausted on turn {turn_idx}: {last_err:?}");
                        tracing::error!("AgentSession {err_msg}");
                        if let Some(o) = &self.observer {
                            o.on_note(&format!("ERROR: {err_msg}"));
                        }
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
                    criteria,
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
                        criteria,
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
                    let made_progress = self.turn_made_progress(&tool, &result, ok);
                    self.turns.push(AgentTurn { tool, result, ok });
                    if made_progress {
                        self.last_progress_turn = self.turns.len();
                    }
                }
            }
        }

        if let Some(done) = self.try_autocomplete("turn limit") {
            return done;
        }

        // Turn limit reached without enough evidence to auto-complete.
        AgentResult::MaxTurnsReached
    }

    fn try_autocomplete(&mut self, reason: &str) -> Option<AgentResult> {
        let last_turn = self.turns.last()?;
        if !(last_turn.ok && matches!(last_turn.tool.as_str(), "edit" | "delete_file")) {
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
        Some(AgentResult::Done {
            ops: std::mem::take(&mut self.ops),
            setup_commands: Vec::new(),
            description: format!(
                "Applied edits before {reason}; runtime verification should validate the result."
            ),
            file_state: self.file_state.clone(),
            criteria: Vec::new(),
        })
    }

    fn exceeded_wall_timeout(&self, started_at: Instant) -> bool {
        self.wall_timeout
            .map(|limit| started_at.elapsed() >= limit)
            .unwrap_or(false)
    }

    fn effective_max_turns(&self) -> usize {
        let progress_budget = self.last_progress_turn + self.budget.max_turns;
        if self.is_repair_extension_eligible() {
            progress_budget + REPAIR_TURN_EXTENSION
        } else {
            progress_budget
        }
    }

    fn turn_made_progress(&self, tool: &str, result: &str, ok: bool) -> bool {
        if ok && matches!(tool, "edit" | "delete_file" | "ask_user") {
            return true;
        }
        self.result_adds_new_information(tool, result)
    }

    fn result_adds_new_information(&self, tool: &str, result: &str) -> bool {
        let evidence = turn_evidence_key(tool, result);
        !evidence.is_empty()
            && !self
                .turns
                .iter()
                .any(|turn| turn.tool == tool && turn_evidence_key(tool, &turn.result) == evidence)
    }

    fn is_repair_extension_eligible(&self) -> bool {
        let wrote_any = self
            .turns
            .iter()
            .any(|turn| turn.ok && matches!(turn.tool.as_str(), "edit" | "delete_file"));
        let ran_any_command = self
            .turns
            .iter()
            .any(|turn| matches!(turn.tool.as_str(), "command"));
        wrote_any && ran_any_command
    }

    fn execute_deduped_command(
        &mut self,
        key: String,
        program: &str,
        args: &[String],
        cwd: Option<String>,
    ) -> Dispatch {
        let command = ProposedCommand {
            program: program.to_string(),
            args: args.to_vec(),
            cwd,
            ..Default::default()
        };
        if let Err(e) = self.materialize_session_files() {
            return Dispatch::Continue {
                result: command_failure_result(
                    "workspace_materialization_error",
                    &render_command(program, args),
                    &format!("Error preparing workspace for command: {e}"),
                    Vec::new(),
                ),
                ok: false,
            };
        }

        let result = self.execute_command_with_expectations(&command);
        self.record_command_observation(key, &result);
        result
    }

    fn record_command_observation(&mut self, _key: String, result: &Dispatch) {
        if let Dispatch::Continue { result, .. } = result {
            self.latest_command = Some(CommandObservation {
                summary: summarize_command_result(result),
            });
        }
    }

    fn execute_command_with_expectations(&self, command: &ProposedCommand) -> Dispatch {
        let expectation_state = match command_expectation_state(
            &self.workspace_root,
            &self.extra_ignore_patterns,
            command,
        ) {
            Ok(state) => state,
            Err(message) => {
                return Dispatch::Continue {
                    result: command_failure_result(
                        "expectation_snapshot_error",
                        &render_command(&command.program, &command.args),
                        &message,
                        Vec::new(),
                    ),
                    ok: false,
                };
            }
        };
        let known_project_dirs: Vec<String> = discover_project_roots(
            Path::new(&self.workspace_root),
            &self.written_paths.iter().cloned().collect::<Vec<_>>(),
        )
        .into_iter()
        .filter_map(|(dir, _)| (!dir.is_empty()).then_some(dir))
        .collect();
        let result = execute_command(
            &self.workspace_root,
            command.cwd.as_deref(),
            &command.program,
            &command.args,
            &known_project_dirs,
        );
        verify_command_expectations(
            &self.workspace_root,
            &self.extra_ignore_patterns,
            command,
            expectation_state,
            result,
        )
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

    fn dispatch_run_command(&mut self, action: &AgentAction) -> Dispatch {
        let command = match command_spec(action) {
            Ok(command) => command,
            Err(result) => return result,
        };
        if let Err(result) = validate_command_spec(&command) {
            return result;
        }
        self.execute_deduped_command(
            command_key(&command.program, &command.args),
            &command.program,
            &command.args,
            command.cwd.clone(),
        )
    }

    /// Render a loaded file as a self-contained tool result: a `<file>` block with
    /// REAL line numbers. Stored once in `conv_history` (append-only) so the model
    /// always has current line numbers without a per-turn full re-dump.
    fn render_loaded_file(&self, path: &str, contents: &str) -> String {
        let kw_owned = task_keywords(&self.current_task);
        let kw_refs: Vec<&str> = kw_owned.iter().map(String::as_str).collect();
        let numbered = render_file_numbered(contents, &kw_refs);
        let total = contents.lines().count();
        format!(
            "Loaded '{path}' ({total} lines — line numbers shown, use them with edit):\n\
             <file path=\"{path}\">\n{numbered}</file>"
        )
    }

    /// Produce edit content from the inline tool payload. Empty content is a
    /// valid delete signal for range edits.
    fn resolve_edit_content(
        &self,
        inline: Option<&str>,
        is_append: bool,
    ) -> crate::InferResult<String> {
        let _ = self;
        match inline {
            Some(content) => Ok(content.to_string()),
            None if !is_append => Ok(String::new()),
            None => Err(crate::InferError::InvalidOutput {
                retries: 0,
                reason: "edit content missing from tool call".into(),
            }),
        }
    }

    fn dispatch(&mut self, action: AgentAction) -> Dispatch {
        match action.tool.as_str() {
            // `edit` with start_line: surgical line-range replacement/append/delete.
            // The model addresses by LINE NUMBER (from a line-numbered read), and the
            // file is reassembled deterministically by shunt-edit.
            "edit" if action.start_line.is_some() => {
                let path = action.path.as_deref().unwrap_or("").to_string();
                let start = action.start_line.unwrap_or(0);
                let end = action.end_line.unwrap_or(start).max(start);
                if path.is_empty() {
                    return Dispatch::Continue {
                        result: "Error: path is required for edit. \
                                 Example: {\"tool\":\"edit\",\"path\":\"src/foo.rs\",\"start_line\":1,\"end_line\":1}".into(),
                        ok: false,
                    };
                }
                if start == 0 {
                    return Dispatch::Continue {
                        result: "Error: start_line is required (1-indexed). \
                                 Call read_file first to see line numbers, then use edit with the correct start_line and end_line. \
                                 Example: {\"tool\":\"edit\",\"path\":\"src/foo.rs\",\"start_line\":3,\"end_line\":3}".into(),
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
                            "Error: this edit range already produced no file change: \
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
                let (_text_system, _base_user) = if is_append {
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
                let new_content = match self
                    .resolve_edit_content(action.content.as_deref(), is_append)
                {
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
                                "Error: replacement content missing or invalid for '{path}': {e}"
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
                                     edit range that covers the stale code. Do not repeat \
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
                                 Edit applied. Call done now, or edit on a \
                                 different file/range if more changes are needed."
                            )
                        } else {
                            format!(
                                "OK — replaced lines {start}-{applied_end} in '{path}'. Current file:\n\
                                 {updated_view}\n\
                                 {warning}\
                                 Edit applied. Call done now, or edit on a \
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

            // `edit` without start_line: create a new file (errors if it already exists).
            "edit" => {
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
                if self.file_state.contains_key(&path) {
                    let n = self.file_state[&path].lines().count();
                    return Dispatch::Continue {
                        result: format!(
                            "Error: '{path}' already exists ({n} lines). \
                             To modify it, use edit with start_line and end_line. \
                             To append, use start_line={} (one past the last line).",
                            n + 1
                        ),
                        ok: false,
                    };
                }
                // Single-shot: the model supplies the new file's content inline. (Existing
                // files are rejected above, so this always creates a brand-new file.)
                if let Some(inline) = action.content.as_deref().filter(|s| !s.trim().is_empty()) {
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
                    self.ops.push(ProposedFileOp::Create {
                        path: path.clone(),
                        contents,
                    });
                    self.written_paths.insert(path.clone());
                    return Dispatch::Continue {
                        result: format!("OK — '{path}' written to disk."),
                        ok: true,
                    };
                }
                // No inline content — fall through to two-phase generation.
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
                // but not yet on disk (edit updates file_state on each successful call).
                // History is append-only: re-reading returns the CURRENT numbered content
                // so the model always sees up-to-date line numbers without a per-turn re-dump.
                if let Some(contents) = self.file_state.get(path).cloned() {
                    Dispatch::Continue {
                        result: self.render_loaded_file(path, &contents),
                        ok: true,
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
                            self.file_state.insert(path.to_string(), contents.clone());
                            Dispatch::Continue {
                                result: self.render_loaded_file(path, &contents),
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

            "knowledge" => {
                let query = action.query.as_deref().unwrap_or("").trim();
                if query.is_empty() {
                    return Dispatch::Continue {
                        result: "Error: knowledge requires a non-empty query (e.g. a package name, API, or topic).".into(),
                        ok: false,
                    };
                }
                match &self.knowledge {
                    Some(backend) => match backend.lookup(query) {
                        Ok(evidence) => Dispatch::Continue {
                            result: evidence,
                            ok: true,
                        },
                        Err(e) => Dispatch::Continue {
                            result: format!("Knowledge lookup failed: {e}"),
                            ok: false,
                        },
                    },
                    None => Dispatch::Continue {
                        result: "Error: no knowledge backend is available in this session.".into(),
                        ok: false,
                    },
                }
            }

            "command" => self.dispatch_run_command(&action),

            "ask_user" => Dispatch::NeedsClarification {
                question: action.question.unwrap_or_else(|| "?".into()),
                context: action.context.unwrap_or_default(),
            },

            "done" => {
                let setup_commands = match action
                    .setup_commands
                    .iter()
                    .map(|spec| parse_command_line(&spec.command_line, spec.cwd.clone()))
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(commands) => commands,
                    Err(result) => return result,
                };
                Dispatch::Done {
                    description: action
                        .description
                        .unwrap_or_else(|| "Applied changes".into()),
                    setup_commands,
                    criteria: action
                        .criteria
                        .into_iter()
                        .map(RawCriterion::into_outcome)
                        .collect(),
                }
            }

            other => Dispatch::Continue {
                result: format!(
                    "Unknown tool '{other}'. Available tools: \
                     read_file, search_files, edit, command, ask_user, done. \
                     Correct the tool name and try again."
                ),
                ok: false,
            },
        }
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
            "No files found for '{query}'. Listing workspace files instead:\n{}",
            list_workspace_files(workspace_root, extra_ignore)
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

fn build_system_prompt(
    workspace_root: &str,
    single_shot_edits: bool,
    has_knowledge: bool,
) -> String {
    let root = Path::new(workspace_root);
    let gi = build_ignore_matcher(workspace_root, &[]);
    let tree = build_dir_tree(root, 0, 3, &mut 0, 60, &gi);
    let manifests = read_manifest_files(root);

    let os_name = match std::env::consts::OS {
        "macos" => "macOS",
        "windows" => "Windows",
        _ => "Linux",
    };
    let mut prompt = AGENT_SYSTEM_PROMPT_BASE
        .replace("{{TOOLS}}", &tools_doc(false, has_knowledge))
        .replace("{{OS}}", os_name);
    if single_shot_edits {
        prompt.push_str(
            "\n\nWhen you call edit, put the FULL code in the `content` field of the SAME action — \
             do not split it into a separate step. When modifying (start_line given), `content` \
             is the new text for lines start_line..end_line only (omit it to delete the range). \
             When creating (no start_line), `content` is the whole new file.",
        );
    }
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

const VERIFIER_SYSTEM_PROMPT_BASE: &str = include_str!("../../../prompts/verifier.system.txt");

fn build_verifier_prompt(workspace_root: &str, changed_paths: &[String]) -> String {
    let root = Path::new(workspace_root);
    let gi = build_ignore_matcher(workspace_root, &[]);
    let manifests = read_manifest_files(root);
    let project_roots = discover_project_roots(root, changed_paths);

    let mut prompt = VERIFIER_SYSTEM_PROMPT_BASE.replace("{{TOOLS}}", &tools_doc(true, false));
    prompt.push_str("\n\n---\n\n## Workspace\n\n");
    prompt.push_str(&format!("Root: {workspace_root}\n\n"));

    // Derived from the changed files, not a blind directory scan: tells the model
    // exactly which directory (if any) to pass as cwd for build/test commands,
    // instead of guessing or defaulting to the workspace root.
    let non_root: Vec<&(String, String)> = project_roots
        .iter()
        .filter(|(dir, _)| !dir.is_empty())
        .collect();
    if !non_root.is_empty() {
        prompt.push_str("### Project Directories (derived from changed files)\n");
        for (dir, manifest_name) in &non_root {
            prompt.push_str(&format!(
                "- `{dir}` (has {manifest_name}) — run build/test commands for this project with cwd=\"{dir}\"\n"
            ));
        }
        prompt.push('\n');
    }

    if !manifests.is_empty() {
        prompt.push_str("### Key Files\n");
        for (name, contents) in &manifests {
            prompt.push_str(&format!("\n#### {name}\n```\n{contents}\n```\n"));
        }
    }
    for (dir, manifest_name) in &non_root {
        let path = root.join(dir).join(manifest_name);
        if let Ok(contents) = std::fs::read_to_string(&path) {
            let capped: String = contents.lines().take(200).collect::<Vec<_>>().join("\n");
            prompt.push_str(&format!(
                "\n#### {dir}/{manifest_name}\n```\n{capped}\n```\n"
            ));
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

/// Discover manifest-bearing directories relevant to the given workspace-relative
/// changed paths: for each changed file, walk its directory chain up to the
/// workspace root checking for a known manifest (the same language-generic
/// `MANIFEST_NAMES` list used everywhere else). Driven entirely by files that
/// actually changed, not a blind recursive scan of the whole tree — bounded by
/// `changed_paths.len() * directory depth`.
///
/// An empty directory string means the workspace root itself. Language-agnostic
/// by construction: it works for any manifest already in `MANIFEST_NAMES`
/// (Cargo.toml, go.mod, pyproject.toml, requirements.txt, deno.json, package.json).
fn discover_project_roots(root: &Path, changed_paths: &[String]) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut seen_dirs = HashSet::new();

    let mut check_dir = |dir: &Path, found: &mut Vec<(String, String)>| {
        let rel = dir.to_string_lossy().to_string();
        let rel = if rel == "." { String::new() } else { rel };
        if !seen_dirs.insert(rel.clone()) {
            return;
        }
        for name in MANIFEST_NAMES {
            if root.join(dir).join(name).is_file() {
                found.push((rel.clone(), (*name).to_string()));
                break;
            }
        }
    };

    check_dir(Path::new(""), &mut found);
    for changed in changed_paths {
        let mut dir = Path::new(changed).parent();
        while let Some(d) = dir {
            check_dir(d, &mut found);
            if d.as_os_str().is_empty() {
                break;
            }
            dir = d.parent();
        }
    }
    found
}

// ── Multi-turn context helpers ────────────────────────────────────────────────

/// Soft token ceiling for `conv_history` (excludes system + task frame + ephemeral).
/// ~4 chars per token; 2000 tokens ≈ 8KB. File content now lives in append-only
/// history (the read/edit results), so this budget governs how much of that history
/// stays hot before eviction kicks in.
const HISTORY_TOKEN_SOFT_LIMIT: usize = 2000;

/// Build the ephemeral continuation message appended to the last message each turn.
/// NOT stored in `conv_history` — rebuilt fresh so the index and nudge are current.
///
/// File CONTENT is no longer dumped here: read/edit results carry their own numbered
/// content into the append-only history. This message only carries a compact
/// loaded-files index, an unloaded-files hint, and the contextual nudge, so the
/// merged-into-last-message tail stays small and the KV-cached prefix is preserved.
#[allow(clippy::too_many_arguments)]
fn build_continuation_msg(
    task: &str,
    file_state: &HashMap<String, String>,
    ops: &[ProposedFileOp],
    turns: &[AgentTurn],
    latest_command: Option<&CommandObservation>,
    remaining_without_progress: usize,
    workspace_root: &str,
    budget: &SessionBudget,
) -> String {
    let mut msg = String::new();

    // Environment profile (workspace root + available commands) is static for the
    // session — emit it once on the first turn, not every turn.
    if turns.is_empty() {
        msg.push_str(&environment_profile(workspace_root));
    }

    // Compact index of loaded files (names + line counts only — content is in the
    // read/edit results already in history).
    let mut paths: Vec<&String> = file_state.keys().collect();
    paths.sort();
    if !paths.is_empty() {
        msg.push_str("LOADED FILES (content shown in earlier read/edit results; re-read for current line numbers if unsure):\n");
        for path in &paths {
            let lines = file_state[*path].lines().count();
            msg.push_str(&format!("  {path} ({lines} lines)\n"));
        }
    }

    // Hint at relevant unloaded files.
    let unloaded = find_likely_unloaded(workspace_root, task, file_state);
    if !unloaded.is_empty() {
        msg.push_str("\nOther files in workspace (not yet loaded): ");
        msg.push_str(&unloaded.join(", "));
        msg.push('\n');
    }

    if let Some(command) = latest_command {
        msg.push_str("\nLATEST COMMAND RESULT:\n");
        msg.push_str(&command.summary);
        msg.push_str("\nTreat this as new evidence. Usually inspect, edit, queue setup, or choose a different command before retrying unchanged.\n");
    }

    // Contextual nudge based on turn history.
    msg.push_str(&build_nudge(
        task,
        file_state,
        ops,
        turns,
        latest_command,
        remaining_without_progress,
        budget,
    ));
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

/// Parse a terminal-style command line into a `ProposedCommand`. Expectation
/// fields are runtime bookkeeping, not model input — they default to empty here.
fn parse_command_line(line: &str, cwd: Option<String>) -> Result<ProposedCommand, Dispatch> {
    let argv = split_command_line(line.trim()).map_err(|e| Dispatch::Continue {
        result: format!("Error parsing command_line: {e}"),
        ok: false,
    })?;
    let Some((program, args)) = argv.split_first() else {
        return Err(Dispatch::Continue {
            result: "Error: command_line is empty. Provide the command as you would type it, e.g. {\"command_line\":\"npm run build\"}.".into(),
            ok: false,
        });
    };
    Ok(ProposedCommand {
        program: program.clone(),
        args: args.to_vec(),
        cwd,
        expect_workspace_change: false,
        expect_paths: Vec::new(),
    })
}

fn command_spec(action: &AgentAction) -> Result<ProposedCommand, Dispatch> {
    let line = action.command_line.as_deref().unwrap_or("").trim();
    if line.is_empty() {
        return Err(Dispatch::Continue {
            result:
                "Error: command requires command_line, e.g. {\"command_line\":\"npm run build\"}."
                    .into(),
            ok: false,
        });
    }
    parse_command_line(line, action.cwd.clone())
}

fn validate_command_spec(command: &ProposedCommand) -> Result<(), Dispatch> {
    if command.program.trim().is_empty() {
        return Err(Dispatch::Continue {
            result: command_failure_result(
                "missing_program",
                "",
                "Command is missing a program. Provide command_line as you would type it in a terminal.",
                Vec::new(),
            ),
            ok: false,
        });
    }
    validate_direct_argv(&command.program, &command.args)
        .map_err(|result| Dispatch::Continue { result, ok: false })
}

fn validate_direct_argv(cmd: &str, args: &[String]) -> Result<(), String> {
    let argv = std::iter::once(cmd.to_string())
        .chain(args.iter().cloned())
        .collect::<Vec<_>>();
    if let Some(op) = args
        .iter()
        .find(|arg| matches!(arg.as_str(), "&&" | "||" | ";" | "|" | ">" | ">>" | "<"))
    {
        return Err(command_failure_result(
            "unsupported_shell_operator",
            &render_command(cmd, args),
            &format!(
                "Shell operator '{op}' is unsupported because the command executes argv directly, not through a shell. Use one command_line per call."
            ),
            split_shell_operator_recovery(&argv),
        ));
    }
    if args.is_empty() && is_bare_interactive_command(cmd) {
        return Err(command_failure_result(
            "interactive_command_without_args",
            cmd,
            &format!(
                "'{cmd}' without args is interactive or non-actionable. Use explicit args like '{cmd} --version', or the full intended command line."
            ),
            vec![ProposedCommand {
                program: cmd.to_string(),
                args: vec!["--version".into()],
                cwd: None,
                expect_workspace_change: false,
                expect_paths: vec![],
            }],
        ));
    }
    Ok(())
}

fn split_shell_operator_recovery(argv: &[String]) -> Vec<ProposedCommand> {
    let mut commands = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut pending_cwd: Option<String> = None;
    for token in argv {
        if matches!(token.as_str(), "&&" | "||" | ";") {
            // `cd PATH` sets the working directory for the next command rather
            // than emitting a useless cd subprocess (cd is a shell builtin).
            if current.len() == 2 && current[0] == "cd" {
                pending_cwd = Some(current[1].clone());
                current.clear();
            } else {
                push_recovery_command(&mut commands, &mut current, pending_cwd.take());
            }
        } else if matches!(token.as_str(), "|" | ">" | ">>" | "<") {
            return Vec::new();
        } else {
            current.push(token.clone());
        }
    }
    // trailing `cd PATH` with no following command — nothing to emit
    if !(current.len() == 2 && current[0] == "cd") {
        push_recovery_command(&mut commands, &mut current, pending_cwd.take());
    }
    commands
}

fn push_recovery_command(
    commands: &mut Vec<ProposedCommand>,
    current: &mut Vec<String>,
    cwd: Option<String>,
) {
    if current.is_empty() {
        return;
    }
    commands.push(ProposedCommand {
        program: current[0].clone(),
        args: current[1..].to_vec(),
        cwd,
        expect_workspace_change: false,
        expect_paths: vec![],
    });
    current.clear();
}

fn is_bare_interactive_command(cmd: &str) -> bool {
    let Some(name) = std::path::Path::new(cmd)
        .file_name()
        .and_then(|n| n.to_str())
    else {
        return false;
    };
    matches!(
        name,
        "bash"
            | "fish"
            | "irb"
            | "node"
            | "npm"
            | "php"
            | "psql"
            | "python"
            | "python3"
            | "ruby"
            | "sh"
            | "sqlite3"
            | "zsh"
    )
}

fn check_command_safety(cmd: &str, args: &[String]) -> Result<(), Dispatch> {
    // Apply the shared safety classifier (shunt_core::safety). Command-line input
    // is parsed into argv first; nothing is executed through a shell here.
    let spec = shunt_core::CommandSpec::new(cmd, args.iter().map(String::as_str));
    match safety::classify(&spec) {
        safety::CommandSafety::Blocked { reason } => Err(Dispatch::Continue {
            result: command_failure_result(
                "safety_blocked",
                &spec.display(),
                &format!("Command blocked by safety policy: {reason}"),
                Vec::new(),
            ),
            ok: false,
        }),
        safety::CommandSafety::Dangerous { reason } => Err(Dispatch::Continue {
            result: command_failure_result(
                "requires_approval",
                &spec.display(),
                &format!(
                    "Command requires approval: {reason}. Queue it as setup if it is required for the final task, or ask the user."
                ),
                Vec::new(),
            ),
            ok: false,
        }),
        safety::CommandSafety::Safe => Ok(()),
    }
}

fn is_likely_long_running_command(cmd: &str, args: &[String]) -> bool {
    let Some(name) = std::path::Path::new(cmd)
        .file_name()
        .and_then(|n| n.to_str())
    else {
        return false;
    };

    let first = args.first().map(String::as_str);
    let second = args.get(1).map(String::as_str);

    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--watch" | "-w"))
    {
        return true;
    }

    if matches!(name, "vite" | "next" | "nuxt" | "astro")
        && matches!(first, Some("dev" | "start" | "preview"))
    {
        return true;
    }

    if matches!(name, "npm" | "pnpm" | "yarn" | "bun")
        && matches!(first, Some("run"))
        && matches!(
            second,
            Some("dev" | "start" | "serve" | "watch" | "preview")
        )
    {
        return true;
    }

    if matches!(name, "cargo") && matches!(first, Some("watch")) {
        return true;
    }

    false
}

fn command_key(cmd: &str, args: &[String]) -> String {
    format!("argv:{}", render_command(cmd, args))
}

fn command_expectation_state(
    workspace_root: &str,
    extra_ignore: &[String],
    command: &ProposedCommand,
) -> Result<CommandExpectationState, String> {
    let workspace_fingerprint = if command.expect_workspace_change {
        Some(workspace_fingerprint(workspace_root, extra_ignore)?)
    } else {
        None
    };
    Ok(CommandExpectationState {
        workspace_fingerprint,
    })
}

fn verify_command_expectations(
    workspace_root: &str,
    extra_ignore: &[String],
    command: &ProposedCommand,
    before: CommandExpectationState,
    result: Dispatch,
) -> Dispatch {
    let Dispatch::Continue {
        ok: true,
        result: detail,
    } = result
    else {
        return result;
    };
    let (exit_code, stdout, stderr) = command_outcome_streams(&detail);

    if !command.expect_paths.is_empty() {
        let mut missing = Vec::new();
        for path in &command.expect_paths {
            let abs = match resolve_in_workspace(workspace_root, path) {
                Ok(abs) => abs,
                Err(e) => {
                    return Dispatch::Continue {
                        result: command_failure_result(
                            "invalid_expect_path",
                            &render_command(&command.program, &command.args),
                            &format!("Invalid expect_paths entry '{path}': {e}"),
                            Vec::new(),
                        ),
                        ok: false,
                    };
                }
            };
            if !abs.exists() {
                missing.push(path.clone());
            }
        }
        if !missing.is_empty() {
            return Dispatch::Continue {
                result: command_failure_outcome_result(
                    "expected_paths_missing",
                    &render_command(&command.program, &command.args),
                    exit_code,
                    &stdout,
                    &stderr,
                    &format!(
                        "Command finished but did not produce the expected path(s): {}",
                        missing.join(", ")
                    ),
                ),
                ok: false,
            };
        }
    }

    if let Some(before_fingerprint) = before.workspace_fingerprint {
        match workspace_fingerprint(workspace_root, extra_ignore) {
            Ok(after_fingerprint) if after_fingerprint == before_fingerprint => {
                return Dispatch::Continue {
                    result: command_failure_outcome_result(
                        "expected_workspace_change_missing",
                        &render_command(&command.program, &command.args),
                        exit_code,
                        &stdout,
                        &stderr,
                        "Command finished but did not change the workspace.",
                    ),
                    ok: false,
                };
            }
            Err(message) => {
                return Dispatch::Continue {
                    result: command_failure_result(
                        "expectation_snapshot_error",
                        &render_command(&command.program, &command.args),
                        &message,
                        Vec::new(),
                    ),
                    ok: false,
                };
            }
            _ => {}
        }
    }

    Dispatch::Continue {
        result: detail,
        ok: true,
    }
}

fn command_outcome_streams(detail: &str) -> (i32, String, String) {
    let Ok(value) = serde_json::from_str::<Value>(detail) else {
        return (0, String::new(), String::new());
    };
    let Some(obj) = value.as_object() else {
        return (0, String::new(), String::new());
    };
    let exit_code = obj
        .get("exit_code")
        .and_then(Value::as_i64)
        .and_then(|n| i32::try_from(n).ok())
        .unwrap_or(0);
    let stdout = obj
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let stderr = obj
        .get("stderr")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    (exit_code, stdout, stderr)
}

fn workspace_fingerprint(workspace_root: &str, extra_ignore: &[String]) -> Result<u64, String> {
    use std::hash::{Hash, Hasher};

    let root = Path::new(workspace_root);
    let gi = build_ignore_matcher(workspace_root, extra_ignore);
    let walker = ignore::WalkBuilder::new(root).hidden(true).build();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for entry in walker.flatten() {
        let path = entry.path();
        if path == root
            || gi
                .matched_path_or_any_parents(path, path.is_dir())
                .is_ignore()
        {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        rel.hash(&mut hasher);
        if let Ok(meta) = entry.metadata() {
            meta.is_dir().hash(&mut hasher);
            meta.len().hash(&mut hasher);
            if let Ok(modified) = meta.modified()
                && let Ok(since_epoch) = modified.duration_since(std::time::UNIX_EPOCH)
            {
                since_epoch.as_secs().hash(&mut hasher);
                since_epoch.subsec_nanos().hash(&mut hasher);
            }
        }
    }
    Ok(hasher.finish())
}

fn render_command(cmd: &str, args: &[String]) -> String {
    std::iter::once(cmd)
        .chain(args.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

fn command_failure_result(
    failure_kind: &str,
    command: &str,
    message: &str,
    recovery: Vec<ProposedCommand>,
) -> String {
    let recovery = recovery
        .into_iter()
        .map(|command| {
            json!({
                "program": command.program,
                "args": command.args,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string_pretty(&json!({
        "status": "failed",
        "failure_kind": failure_kind,
        "command": command,
        "message": message,
        "recovery": recovery,
    }))
    .unwrap_or_else(|_| message.to_string())
}

fn summarize_command_result(result: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(result) else {
        return result.lines().next().unwrap_or(result).trim().to_string();
    };
    let Some(obj) = value.as_object() else {
        return result.lines().next().unwrap_or(result).trim().to_string();
    };
    let status = obj.get("status").and_then(Value::as_str).unwrap_or("");
    let command = obj
        .get("command")
        .and_then(Value::as_str)
        .unwrap_or("command");
    let failure_kind = obj
        .get("failure_kind")
        .and_then(Value::as_str)
        .unwrap_or("");
    let message = obj.get("message").and_then(Value::as_str).map(str::trim);
    let stderr = obj.get("stderr").and_then(Value::as_str).map(str::trim);
    let stdout = obj.get("stdout").and_then(Value::as_str).map(str::trim);
    match status {
        "success" => format!("{command} succeeded"),
        "failed" => {
            let detail = message
                .filter(|s| !s.is_empty())
                .or_else(|| stderr.filter(|s| !s.is_empty()))
                .or_else(|| stdout.filter(|s| !s.is_empty()))
                .unwrap_or("failed");
            if failure_kind.is_empty() {
                format!("{command} failed: {detail}")
            } else {
                format!("{command} failed [{failure_kind}]: {detail}")
            }
        }
        _ => result.lines().next().unwrap_or(result).trim().to_string(),
    }
}

fn turn_evidence_key(tool: &str, result: &str) -> String {
    match tool {
        "command" => summarize_command_result(result),
        _ => result.trim().to_string(),
    }
}

fn latest_command_observation(turns: &[AgentTurn]) -> Option<CommandObservation> {
    turns.iter().rev().find_map(|turn| {
        if matches!(turn.tool.as_str(), "command") {
            Some(CommandObservation {
                summary: summarize_command_result(&turn.result),
            })
        } else {
            None
        }
    })
}

fn command_success_result(command: &str, exit_code: i32, stdout: &str, stderr: &str) -> String {
    serde_json::to_string_pretty(&json!({
        "status": "success",
        "command": command,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    }))
    .unwrap_or_else(|_| format!("{command}: success"))
}

fn command_failure_outcome_result(
    failure_kind: &str,
    command: &str,
    exit_code: i32,
    stdout: &str,
    stderr: &str,
    message: &str,
) -> String {
    serde_json::to_string_pretty(&json!({
        "status": "failed",
        "failure_kind": failure_kind,
        "command": command,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "message": message,
        "recovery": [],
    }))
    .unwrap_or_else(|_| message.to_string())
}

/// Generic hint appended to execution failures when no `cwd` was set and other
/// changed files reveal a project directory elsewhere in the workspace. Not tied
/// to any language or package manager — it fires for any command (cargo, npm,
/// go, pip, ...) whenever `discover_project_roots` found a manifest outside the
/// workspace root, so the model doesn't have to be told about each tool by name.
fn cwd_hint(cwd: Option<&str>, known_project_dirs: &[String]) -> String {
    if cwd.is_some() || known_project_dirs.is_empty() {
        return String::new();
    }
    let plural = if known_project_dirs.len() == 1 {
        "y"
    } else {
        "ies"
    };
    format!(
        " Known project director{plural} in this workspace (based on files touched so far): {}. \
         If this command targets one of them, set cwd accordingly.",
        known_project_dirs.join(", ")
    )
}

fn execute_command(
    workspace_root: &str,
    cwd: Option<&str>,
    cmd: &str,
    args: &[String],
    known_project_dirs: &[String],
) -> Dispatch {
    if let Err(result) = validate_direct_argv(cmd, args) {
        return Dispatch::Continue { result, ok: false };
    }
    if let Err(result) = check_command_safety(cmd, args) {
        return result;
    }
    let rendered = render_command(cmd, args);
    if is_likely_long_running_command(cmd, args) {
        return Dispatch::Continue {
            result: command_failure_result(
                "long_running_command_not_supported",
                &rendered,
                "This command is likely to start a long-running dev server or watcher. Commands must be finite — finish the task without starting a persistent process.",
                Vec::new(),
            ),
            ok: false,
        };
    }

    let wd = match cwd {
        Some(rel) => std::path::Path::new(workspace_root).join(rel),
        None => std::path::PathBuf::from(workspace_root),
    };

    // Execvp-style: cmd is the program, args are its arguments.
    // Stdout/stderr drained on threads to prevent pipe-buffer deadlock.
    // Hard timeout kills the child so a hung build doesn't block forever.
    const AGENT_CMD_TIMEOUT_SECS: u64 = 120;
    let mut child = match std::process::Command::new(cmd)
        .args(args)
        .current_dir(&wd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return Dispatch::Continue {
                result: command_failure_outcome_result(
                    "spawn_error",
                    &rendered,
                    -1,
                    "",
                    "",
                    &format!(
                        "Error spawning '{cmd}': {e}.{}",
                        cwd_hint(cwd, known_project_dirs)
                    ),
                ),
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
                    break Err("timeout".to_string());
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
            let exit_code = status.code().unwrap_or(-1);
            let result = if ok {
                command_success_result(&rendered, exit_code, &stdout, &stderr)
            } else {
                command_failure_outcome_result(
                    "nonzero_exit",
                    &rendered,
                    exit_code,
                    &stdout,
                    &stderr,
                    &format!(
                        "Command exited with a nonzero status. Inspect stdout/stderr, fix the cause, then retry only after making a relevant change.{}",
                        cwd_hint(cwd, known_project_dirs)
                    ),
                )
            };
            Dispatch::Continue { result, ok }
        }
        Err(msg) if msg == "timeout" => Dispatch::Continue {
            result: command_failure_outcome_result(
                "timeout",
                &rendered,
                -1,
                &tail_utf8(&stdout_bytes, MAX_CMD_OUTPUT),
                &tail_utf8(&stderr_bytes, MAX_CMD_OUTPUT),
                &format!(
                    "Command timed out after {AGENT_CMD_TIMEOUT_SECS}s and was killed. Commands must be finite — do not run dev servers, watchers, or other long-running processes."
                ),
            ),
            ok: false,
        },
        Err(msg) => Dispatch::Continue {
            result: command_failure_outcome_result(
                "wait_error",
                &rendered,
                -1,
                &tail_utf8(&stdout_bytes, MAX_CMD_OUTPUT),
                &tail_utf8(&stderr_bytes, MAX_CMD_OUTPUT),
                &msg,
            ),
            ok: false,
        },
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
    latest_command: Option<&CommandObservation>,
    remaining_without_progress: usize,
    budget: &SessionBudget,
) -> String {
    if turns.is_empty() {
        return "\nReason about the approach, then explore what you need. \
                What is your first action?"
            .into();
    }
    let remaining = remaining_without_progress;
    let idle_streak = turns
        .iter()
        .rev()
        .take_while(|t| matches!(t.tool.as_str(), "read_file" | "search_files"))
        .count();
    let wrote_any = turns
        .iter()
        .any(|t| matches!(t.tool.as_str(), "edit" | "delete_file") && t.ok);
    let last_verified = turns
        .iter()
        .rev()
        .find(|t| matches!(t.tool.as_str(), "command"));
    let last_verified_ok = last_verified
        .map(|t| {
            t.ok && !t.result.to_ascii_lowercase().contains("error")
                && !t.result.to_ascii_lowercase().contains("failed")
        })
        .unwrap_or(false);
    let last_action = turns.last().map(|t| t.tool.as_str()).unwrap_or("");
    let missing_explicit_paths = missing_explicit_file_edits(task, file_state, ops);
    let latest_command_failed = latest_command
        .map(|command| command.summary.contains(" failed"))
        .unwrap_or(false);

    if idle_streak >= budget.stall_warn_at {
        format!(
            "\nSTALL WARNING: {idle_streak} turns of reading without any edit or command. \
             You MUST call edit, command, or done NOW. \
             No more reads. ({remaining} turns remaining)"
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
    } else if wrote_any && matches!(last_action, "command") && last_verified_ok {
        format!(
            "\nBuild passed. Call done with a description of what changed and what was verified. \
             ({remaining} turns remaining)"
        )
    } else if wrote_any && matches!(last_action, "command") && !last_verified_ok {
        format!(
            "\nBuild failed. Find the root cause, then use edit \
             to fix the specific error. Re-run the build after fixing. \
             ({remaining} turns remaining)"
        )
    } else if matches!(last_action, "command") && !last_verified_ok {
        format!(
            "\nThe last command failed. Use its result to choose a different command, queue setup, or edit before retrying. \
             Do not repeat the same command unchanged. ({remaining} turns remaining)"
        )
    } else if latest_command_failed {
        format!(
            "\nYou have fresh command failure evidence. Act on it: inspect, edit, queue setup, or choose a different command. \
             Do not ignore it and drift back to blind retries. ({remaining} turns remaining)"
        )
    } else if wrote_any && !matches!(last_action, "command") {
        format!(
            "\nFiles written. Run the build now to verify your changes work before calling done. \
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
    if let Some(command_line) = &action.command_line {
        map.insert("command_line".into(), serde_json::json!(command_line));
    }
    if let Some(cwd) = &action.cwd {
        map.insert("cwd".into(), serde_json::json!(cwd));
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
    if !action.setup_commands.is_empty() {
        map.insert(
            "setup_commands".into(),
            serde_json::json!(action.setup_commands),
        );
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
        "command" => {
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
// Render a loaded file with REAL line numbers (so `edit` can address them).
// The numbers MUST be the file's true line numbers — small files render whole;
// large files show real-numbered windows (head + keyword regions + tail) with
// explicit "lines X-Y omitted" markers, never a renumbered reassembly.

/// Show files up to this many lines in full; window beyond it (head + keyword
/// regions + tail, with REAL line numbers). Generous — small models comprehend a
/// whole moderate file better than a stitched window, and a cold call on a few-K
/// prompt is only ~15s on a local GPU. `edit` addresses by real line number.
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
        "edit" if action.start_line.is_some() => format!(
            "{}:{}-{}",
            action.path.as_deref().unwrap_or("?"),
            action.start_line.unwrap_or(0),
            action.end_line.unwrap_or(0),
        ),
        "edit" | "read_file" => action.path.as_deref().unwrap_or("?").to_string(),
        "search_files" => action.query.as_deref().unwrap_or("?").to_string(),
        "command" => action.command_line.as_deref().unwrap_or("?").to_string(),
        "ask_user" => action.question.as_deref().unwrap_or("?").to_string(),
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

    fn blank_action(tool: &str) -> AgentAction {
        AgentAction {
            tool: tool.into(),
            path: None,
            start_line: None,
            end_line: None,
            content: None,
            query: None,
            command_line: None,
            cwd: None,
            question: None,
            context: None,
            task: None,
            description: None,
            setup_commands: vec![],
            criteria: vec![],
        }
    }

    #[test]
    fn edit_strategy_selects_by_capability() {
        assert_eq!(
            EditStrategy::for_capabilities(ToolChoiceMode::RequiredString),
            EditStrategy::SingleShot
        );
        assert_eq!(
            EditStrategy::for_capabilities(ToolChoiceMode::NamedObject),
            EditStrategy::SingleShot
        );
        assert_eq!(
            EditStrategy::for_capabilities(ToolChoiceMode::JsonSchema),
            EditStrategy::SingleShot
        );
        assert!(EditStrategy::SingleShot.content_in_schema());
    }

    #[test]
    fn agent_schema_always_carries_content_field() {
        assert!(
            agent_schema(0, true, false)["properties"]
                .get("content")
                .is_some()
        );
        assert!(
            agent_schema(0, false, false)["properties"]
                .get("content")
                .is_some()
        );
    }

    #[test]
    fn single_shot_edit_applies_inline_content_without_a_text_call() {
        let tmp = tempfile::tempdir().unwrap();
        // DummyProvider defaults to RequiredString → SingleShot, and panics if
        // generate_text is ever called — proving the content came from the action.
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        assert_eq!(session.edit_strategy, EditStrategy::SingleShot);
        session.current_task = "edit".into();
        session.file_state.insert("f.rs".into(), "a\nb\nc\n".into());

        // `edit` with start_line routes to the line-range editor.
        let mut action = blank_action("edit");
        action.path = Some("f.rs".into());
        action.start_line = Some(2);
        action.end_line = Some(2);
        action.content = Some("B".into());
        match session.dispatch(action) {
            Dispatch::Continue { ok, result } => assert!(ok, "{result}"),
            _ => panic!("expected Continue"),
        }
        assert_eq!(session.file_state["f.rs"], "a\nB\nc\n");
    }

    #[test]
    fn edit_without_start_line_creates_a_new_file() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider; // SingleShot → uses inline content, no text call
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        session.current_task = "create".into();

        let mut action = blank_action("edit");
        action.path = Some("new.rs".into());
        action.content = Some("fn x() {}\n".into());
        match session.dispatch(action) {
            Dispatch::Continue { ok, result } => assert!(ok, "{result}"),
            _ => panic!("expected Continue"),
        }
        assert_eq!(session.file_state["new.rs"], "fn x() {}\n");
        assert!(tmp.path().join("new.rs").exists());
    }

    struct StubKnowledge;
    impl KnowledgeLookup for StubKnowledge {
        fn lookup(&self, query: &str) -> Result<String, String> {
            Ok(format!("evidence for {query}"))
        }
    }

    #[test]
    fn knowledge_tool_dispatches_to_backend() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap())
            .with_knowledge(Arc::new(StubKnowledge));
        assert!(session.knowledge.is_some());

        let mut action = blank_action("knowledge");
        action.query = Some("axum routing".into());
        match session.dispatch(action) {
            Dispatch::Continue { ok, result } => {
                assert!(ok, "{result}");
                assert_eq!(result, "evidence for axum routing");
            }
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn knowledge_tool_errors_without_backend() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        let mut action = blank_action("knowledge");
        action.query = Some("anything".into());
        match session.dispatch(action) {
            Dispatch::Continue { ok, result } => {
                assert!(!ok);
                assert!(result.contains("no knowledge backend"), "{result}");
            }
            _ => panic!("expected Continue"),
        }
    }

    #[test]
    fn edit_without_inline_content_deletes_range() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        assert_eq!(session.edit_strategy, EditStrategy::SingleShot);
        session.current_task = "edit".into();
        session.file_state.insert("f.rs".into(), "a\nb\nc\n".into());

        let mut action = blank_action("edit");
        action.path = Some("f.rs".into());
        action.start_line = Some(2);
        action.end_line = Some(2);
        match session.dispatch(action) {
            Dispatch::Continue { ok, result } => assert!(ok, "{result}"),
            _ => panic!("expected Continue"),
        }
        assert_eq!(session.file_state["f.rs"], "a\nc\n");
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
            "command_line": "program arg"
        }))
        .unwrap();
        let action = action.into_action();
        assert_eq!(action.tool, "command");
        assert_eq!(action.command_line.as_deref(), Some("program arg"));
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
    fn run_command_rejects_shell_operators() {
        let argv = split_command_line("first --version && second --version").unwrap();
        let err = validate_direct_argv(&argv[0], &argv[1..]).unwrap_err();
        assert!(err.contains("unsupported_shell_operator"), "{err}");
        assert!(err.contains("first"), "{err}");
        assert!(err.contains("second"), "{err}");
    }

    #[test]
    fn command_spec_parses_command_line_into_program_and_args() {
        let mut action = blank_action("command");
        action.command_line = Some("npm run build".into());
        let command = command_spec(&action).ok().expect("expected command");
        assert_eq!(command.program, "npm");
        assert_eq!(command.args, vec!["run", "build"]);
    }

    #[test]
    fn command_spec_requires_command_line() {
        let action = blank_action("command");
        match command_spec(&action) {
            Err(Dispatch::Continue { ok, result }) => {
                assert!(!ok);
                assert!(result.contains("command_line"), "{result}");
            }
            _ => panic!("expected Continue dispatch error"),
        }
    }

    #[test]
    fn shell_operator_recovery_maps_cd_to_cwd() {
        // `cd frontend && npm install` should recover as `npm install` with cwd="frontend",
        // not emit a useless `cd` subprocess followed by `npm install` at workspace root.
        let argv: Vec<String> = ["cd", "frontend", "&&", "npm", "install"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let recovered = split_shell_operator_recovery(&argv);
        assert_eq!(recovered.len(), 1, "{recovered:?}");
        assert_eq!(recovered[0].program, "npm");
        assert_eq!(recovered[0].args, vec!["install"]);
        assert_eq!(recovered[0].cwd.as_deref(), Some("frontend"));
    }

    #[test]
    fn run_command_rejects_bare_interactive_programs() {
        let err = validate_direct_argv("node", &[]).unwrap_err();
        assert!(err.contains("without args"), "{err}");
        assert!(validate_direct_argv("node", &["-v".into()]).is_ok());
        assert!(validate_direct_argv("npm", &["create".into(), "vite@latest".into()]).is_ok());
    }

    #[test]
    fn run_command_rejects_long_running_dev_server_commands() {
        let tmp = tempfile::tempdir().unwrap();
        let result = execute_command(
            tmp.path().to_str().unwrap(),
            None,
            "npm",
            &["run".into(), "dev".into(), "--prefix".into(), "web".into()],
            &[],
        );
        let Dispatch::Continue { ok, result } = result else {
            panic!("expected continue dispatch");
        };
        assert!(!ok);
        assert!(
            result.contains("long_running_command_not_supported"),
            "{result}"
        );
    }

    #[test]
    fn run_executes_command_line() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let mut session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        let action: AgentActionEnvelope = serde_json::from_value(serde_json::json!({
            "tool": "command",
            "command_line": "python3 -c \"print('one')\""
        }))
        .unwrap();

        match session.dispatch(action.into_action()) {
            Dispatch::Continue { ok, result } => {
                assert!(ok, "{result}");
                assert!(result.contains("one"), "{result}");
            }
            _ => panic!("expected command run result"),
        }
    }

    #[test]
    fn command_still_uses_safety_policy() {
        let tmp = tempfile::tempdir().unwrap();
        let result = execute_command(
            tmp.path().to_str().unwrap(),
            None,
            "sh",
            &["-c".into(), "echo unsafe".into()],
            &[],
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
            None,
        );

        match result {
            Dispatch::Continue { ok, result } => assert!(ok, "{result}"),
            _ => panic!("expected command result"),
        }
    }

    #[test]
    fn discover_project_roots_is_driven_by_changed_files_not_a_full_scan() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("frontend")).unwrap();
        std::fs::write(tmp.path().join("frontend/package.json"), "{}").unwrap();
        // A manifest that exists on disk but is unrelated to any changed file must
        // not surface — discovery follows the changed paths' directory chain, it
        // does not walk the whole tree.
        std::fs::create_dir_all(tmp.path().join("unrelated")).unwrap();
        std::fs::write(tmp.path().join("unrelated/Cargo.toml"), "").unwrap();

        let changed = vec!["frontend/src/App.tsx".to_string()];
        let found = discover_project_roots(tmp.path(), &changed);

        assert!(
            found
                .iter()
                .any(|(dir, name)| dir == "frontend" && name == "package.json"),
            "{found:?}"
        );
        assert!(
            !found.iter().any(|(dir, _)| dir == "unrelated"),
            "unrelated manifest should not be discovered: {found:?}"
        );
    }

    #[test]
    fn nonzero_exit_hints_known_project_dir_when_cwd_missing() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("frontend")).unwrap();
        std::fs::write(tmp.path().join("frontend/package.json"), "{}").unwrap();

        let result = execute_command(
            tmp.path().to_str().unwrap(),
            None,
            "false",
            &[],
            &["frontend".to_string()],
        );
        match result {
            Dispatch::Continue { ok, result } => {
                assert!(!ok);
                assert!(result.contains("frontend"), "{result}");
                assert!(result.contains("cwd"), "{result}");
            }
            _ => panic!("expected command result"),
        }
    }

    #[test]
    fn cwd_hint_is_silent_when_cwd_already_set_or_no_projects_known() {
        assert_eq!(cwd_hint(Some("frontend"), &["frontend".to_string()]), "");
        assert_eq!(cwd_hint(None, &[]), "");
    }

    #[test]
    fn command_expectation_requires_expected_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        let command = ProposedCommand {
            program: "python3".into(),
            args: vec![
                "-c".into(),
                "from pathlib import Path; Path('package.json').write_text('{}')".into(),
            ],
            cwd: None,
            expect_workspace_change: true,
            expect_paths: vec!["package.json".into()],
        };

        match session.execute_command_with_expectations(&command) {
            Dispatch::Continue { ok, result } => {
                assert!(ok, "{result}");
                assert!(tmp.path().join("package.json").exists());
            }
            _ => panic!("expected command result"),
        }
    }

    #[test]
    fn command_expectation_detects_missing_effect() {
        let tmp = tempfile::tempdir().unwrap();
        let provider = DummyProvider;
        let session = AgentSession::new(&provider, tmp.path().to_str().unwrap());
        let command = ProposedCommand {
            program: "python3".into(),
            args: vec!["-c".into(), "print('no change')".into()],
            cwd: None,
            expect_workspace_change: true,
            expect_paths: vec!["package.json".into()],
        };

        match session.execute_command_with_expectations(&command) {
            Dispatch::Continue { ok, result } => {
                assert!(!ok);
                assert!(result.contains("expected_paths_missing"), "{result}");
            }
            _ => panic!("expected command result"),
        }
    }

    #[test]
    fn json_edit_validation_rejects_invalid_json() {
        assert!(validate_json_edit("package.json", "{\"dependencies\":{}}").is_ok());
        assert!(validate_json_edit("package.json", "{\"dependencies\":{}").is_err());
        assert!(validate_json_edit("src/app.js", "not json").is_ok());
    }

    #[test]
    fn search_stays_available_after_edit() {
        // Recovery is no longer blocked: search/re-read remain after an edit lands.
        let schema = agent_schema(0, true, false);
        let tools = schema["properties"]["tool"]["enum"].as_array().unwrap();
        assert!(tools.iter().any(|tool| tool == "search_files"));
        assert!(schema["properties"].get("query").is_some());
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn knowledge_tool_only_present_when_backend_wired() {
        let without = agent_schema(0, true, false);
        let with = agent_schema(0, true, true);
        assert!(
            !without["properties"]["tool"]["enum"]
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t == "knowledge")
        );
        assert!(
            with["properties"]["tool"]["enum"]
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t == "knowledge")
        );
        assert!(tools_doc(false, true).contains("knowledge"));
        assert!(!tools_doc(false, false).contains("• knowledge"));
    }

    #[test]
    fn collapsed_tool_set_excludes_removed_tools() {
        let schema = agent_schema(0, true, false);
        let tools = schema["properties"]["tool"]["enum"].as_array().unwrap();
        for removed in ["think", "delete_file", "sub_agent"] {
            assert!(
                !tools.iter().any(|t| t == removed),
                "{removed} should be gone"
            );
        }
        for kept in ["read_file", "search_files", "edit", "command", "done"] {
            assert!(tools.iter().any(|t| t == kept), "{kept} should be present");
        }
        // write_file/replace_lines were replaced by the single `edit` tool.
        for folded in ["write_file", "replace_lines"] {
            assert!(
                !tools.iter().any(|t| t == folded),
                "{folded} should be folded into edit"
            );
        }
        // ask_user is depth-0 only.
        assert!(tools.iter().any(|t| t == "ask_user"));
        assert!(
            !agent_schema(1, true, false)["properties"]["tool"]["enum"]
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t == "ask_user")
        );
    }

    #[test]
    fn tools_doc_lists_each_enabled_tool() {
        let doc = tools_doc(false, false);
        for name in ["read_file", "search_files", "• edit", "command", "done"] {
            assert!(doc.contains(name), "doc should mention {name}");
        }
        assert!(!doc.contains("• think"));
        assert!(!doc.contains("sub_agent"));
        // Verifier doc excludes the edit tool.
        let vdoc = tools_doc(true, false);
        assert!(!vdoc.contains("• edit"));
        assert!(vdoc.contains("read_file"));
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
            tool: "edit".into(),
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
                tool: "edit".into(),
                result: "OK".into(),
                ok: true,
            },
            AgentTurn {
                tool: "command".into(),
                result: "program arg passed".into(),
                ok: true,
            },
        ];

        let nudge = build_nudge(
            "Add a `--json` flag. Parse it in src/main.rs, thread it through ReportOptions in src/report.rs, and update README.md usage.",
            &file_state,
            &ops,
            &turns,
            None,
            8,
            &SessionBudget::for_model(8192),
        );

        assert!(nudge.contains("README.md"), "{nudge}");
        assert!(!nudge.contains("Build passed"), "{nudge}");
    }

    #[test]
    fn continuation_includes_environment_profile() {
        let msg = build_continuation_msg(
            "inspect the project",
            &HashMap::new(),
            &[],
            &[],
            None,
            12,
            "/tmp/example-workspace",
            &SessionBudget::default(),
        );
        assert!(msg.contains("ENVIRONMENT:"));
        assert!(msg.contains("workspace_root: /tmp/example-workspace"));
    }

    #[test]
    fn continuation_includes_latest_command_result() {
        let msg = build_continuation_msg(
            "inspect the project",
            &HashMap::new(),
            &[],
            &[],
            Some(&CommandObservation {
                summary: "npm install failed [nonzero_exit]: package.json missing".into(),
            }),
            12,
            "/tmp/example-workspace",
            &SessionBudget::default(),
        );
        assert!(msg.contains("LATEST COMMAND RESULT:"), "{msg}");
        assert!(msg.contains("npm install failed"), "{msg}");
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
            tool: "edit".into(),
            result: "ok".into(),
            ok: true,
        });
        session.last_progress_turn = session.turns.len();
        assert_eq!(session.effective_max_turns(), 13);
        session.turns.push(AgentTurn {
            tool: "command".into(),
            result: "Error: failing test".into(),
            ok: false,
        });
        session.last_progress_turn = session.turns.len();
        assert_eq!(session.effective_max_turns(), 14 + REPAIR_TURN_EXTENSION);
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
    fn search_files_falls_back_to_listing_when_query_has_no_matches() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/main.rs"), "fn main() {}\n").unwrap();

        let result = search_files_in_workspace(
            tmp.path().to_str().unwrap(),
            "List all files in the workspace to understand the project structure.",
            &[],
        );

        assert!(
            result.contains("Listing workspace files instead:"),
            "{result}"
        );
        assert!(result.contains("src/main.rs"), "{result}");
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
