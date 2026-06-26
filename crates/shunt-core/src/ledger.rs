//! Work ledger — append-only log of every observation, model action, user
//! input, and authorization decision in a session.
//!
//! The ledger is the shared truth that context assembly reads on every turn.
//! Types here are pure data: no IO, no async, no state-machine logic.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::CommandSpec;

// ── ID types ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LedgerEntryId(pub String);

// ── Goal snapshot ─────────────────────────────────────────────────────────────

/// The agent's current understanding of the task goal.
/// Captured at session start and updated when the goal is revised.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoalSnapshot {
    pub original_request: String,
    pub interpreted_goal: String,
    pub success_criteria: Vec<String>,
    pub constraints: Vec<String>,
    pub confidence: f32,
}

// ── Workspace revision ────────────────────────────────────────────────────────

/// Monotonically incremented on every `ChangeApplied` observation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkspaceRevision {
    pub sequence: u64,
    pub content_hash: Option<String>,
}

// ── File snapshot ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSnapshot {
    pub path: String,
    pub content: String,
    pub content_hash: String,
    pub size_bytes: usize,
}

// ── Search hit ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub path: String,
    pub snippet: String,
    pub score: f32,
}

// ── Diff summary ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub files_changed: Vec<String>,
    pub lines_added: usize,
    pub lines_removed: usize,
    pub description: String,
}

// ── Action violation ──────────────────────────────────────────────────────────

/// Why a proposed action was rejected by the kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionViolation {
    pub code: String,
    pub message: String,
}

// ── Check outcome ─────────────────────────────────────────────────────────────

/// Outcome of a single verifier or acceptance check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckOutcome {
    pub name: String,
    pub passed: bool,
    pub summary: String,
    pub output: Option<String>,
}

// ── Observed fact ─────────────────────────────────────────────────────────────

/// A structured fact extracted from command output by a deterministic parser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedFact {
    pub key: String,
    pub value: String,
    pub source: String,
}

// ── Command status ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandStatus {
    Completed,
    Failed,
    Killed,
    Unavailable,
}

// ── Bounded output ────────────────────────────────────────────────────────────

/// Stdout or stderr capped at a configurable byte limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedOutput {
    pub content: String,
    pub truncated: bool,
    pub original_bytes: usize,
}

impl BoundedOutput {
    pub const DEFAULT_LIMIT: usize = 16_384;

    pub fn from_string(s: impl Into<String>, limit: usize) -> Self {
        let s = s.into();
        let original_bytes = s.len();
        if original_bytes <= limit {
            Self {
                content: s,
                truncated: false,
                original_bytes,
            }
        } else {
            let mut end = limit;
            while end > 0 && !s.is_char_boundary(end) {
                end -= 1;
            }
            Self {
                content: format!("{}…[{} bytes truncated]", &s[..end], original_bytes - end),
                truncated: true,
                original_bytes,
            }
        }
    }

    pub fn empty() -> Self {
        Self {
            content: String::new(),
            truncated: false,
            original_bytes: 0,
        }
    }
}

// ── Command observation ───────────────────────────────────────────────────────

/// Full observation from running a single shell command.
/// Carries everything the supervisor needs to reason about what happened:
/// exit code, output, parsed facts, workspace delta, and timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandObservation {
    pub command: CommandSpec,
    pub status: CommandStatus,
    pub exit_code: Option<i32>,
    pub stdout: BoundedOutput,
    pub stderr: BoundedOutput,
    pub parsed: Vec<ObservedFact>,
    pub workspace_delta: Option<DiffSummary>,
    pub elapsed_ms: u64,
}

// ── Tool observation ──────────────────────────────────────────────────────────

/// An observation produced by executing a deterministic tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolObservation {
    FilesRead {
        files: Vec<FileSnapshot>,
    },
    SearchResults {
        query: String,
        hits: Vec<SearchHit>,
    },
    ChangeApplied {
        revision: WorkspaceRevision,
        diff: DiffSummary,
    },
    ChangeRejected {
        violations: Vec<ActionViolation>,
    },
    CommandFinished(CommandObservation),
    ChecksFinished {
        outcomes: Vec<CheckOutcome>,
    },
    /// The same action was tried against an unchanged workspace and context —
    /// signals to the supervisor that the action made no progress.
    NoProgress {
        fingerprint: String,
        prior_entry: LedgerEntryId,
    },
}

// ── Action record ─────────────────────────────────────────────────────────────

/// A model action taken by the supervisor or a legacy node.
/// Populated by the runner when a model call completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecord {
    /// Matches the `call_id` from `ModelCallEvent` when available.
    pub call_id: Option<u64>,
    /// Activity label: "clarify", "understand", "propose", etc.
    pub phase: String,
    /// Tool/node name that was called.
    pub tool: String,
    pub elapsed_ms: u64,
    /// "valid", "invalid", or "error".
    pub outcome: String,
    /// Human-readable summary of what the model produced.
    pub summary: String,
}

// ── Worker result ─────────────────────────────────────────────────────────────

/// Result from a bounded worker assignment (placeholder for M5.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerResult {
    pub role: String,
    pub question: String,
    pub answer: serde_json::Value,
    pub confidence: f32,
}

// ── User observation ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UserObservationKind {
    AmbiguityAnswer { ambiguity_id: String },
    Approval,
    Rejection,
    PatchApproval,
    PatchRejection,
    DangerApproval,
    DangerRejection,
    Cancellation,
    Steer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserObservation {
    pub kind: UserObservationKind,
    /// The raw user content (answer text, steer message, etc.).
    pub content: String,
}

// ── Authorization observation ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationObservation {
    pub action_id: String,
    pub granted: bool,
    pub reason: Option<String>,
}

// ── Context summary ───────────────────────────────────────────────────────────

/// A compacted summary of older ledger entries.
/// Summaries never replace raw recent failures or user constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSummary {
    pub covering_entry_ids: Vec<LedgerEntryId>,
    pub summary: String,
    pub durable_facts: Vec<String>,
}

// ── Ledger entry ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LedgerEntry {
    ModelAction(ActionRecord),
    WorkerResult(WorkerResult),
    ToolObservation(ToolObservation),
    UserInput(UserObservation),
    Authorization(AuthorizationObservation),
    ContextSummary(ContextSummary),
}

impl LedgerEntry {
    /// One-line human-readable label for display and tracing.
    pub fn label(&self) -> String {
        match self {
            LedgerEntry::ModelAction(a) => format!("model:{} {}", a.phase, a.outcome),
            LedgerEntry::WorkerResult(w) => format!("worker:{}", w.role),
            LedgerEntry::ToolObservation(o) => match o {
                ToolObservation::FilesRead { files } => format!("read {} file(s)", files.len()),
                ToolObservation::SearchResults { hits, .. } => {
                    format!("search {} hit(s)", hits.len())
                }
                ToolObservation::ChangeApplied { diff, .. } => {
                    format!("change applied: {}", diff.description)
                }
                ToolObservation::ChangeRejected { violations } => {
                    format!("change rejected: {} violation(s)", violations.len())
                }
                ToolObservation::CommandFinished(c) => {
                    format!("cmd `{}` exit={:?}", c.command.display(), c.exit_code)
                }
                ToolObservation::ChecksFinished { outcomes } => {
                    let passed = outcomes.iter().filter(|o| o.passed).count();
                    format!("checks {}/{} passed", passed, outcomes.len())
                }
                ToolObservation::NoProgress { .. } => "no-progress".into(),
            },
            LedgerEntry::UserInput(u) => {
                let kind = match &u.kind {
                    UserObservationKind::AmbiguityAnswer { .. } => "answer",
                    UserObservationKind::Approval => "approve",
                    UserObservationKind::Rejection => "reject",
                    UserObservationKind::PatchApproval => "patch-approve",
                    UserObservationKind::PatchRejection => "patch-reject",
                    UserObservationKind::DangerApproval => "danger-approve",
                    UserObservationKind::DangerRejection => "danger-reject",
                    UserObservationKind::Cancellation => "cancel",
                    UserObservationKind::Steer => "steer",
                };
                format!("user:{kind}")
            }
            LedgerEntry::Authorization(a) => {
                format!(
                    "auth:{} {}",
                    a.action_id,
                    if a.granted { "granted" } else { "denied" }
                )
            }
            LedgerEntry::ContextSummary(_) => "context-summary".into(),
        }
    }
}

// ── Ledger entry record ───────────────────────────────────────────────────────

/// A `LedgerEntry` with its persistence metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntryRecord {
    pub id: LedgerEntryId,
    pub task_id: String,
    pub sequence: u64,
    pub entry: LedgerEntry,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

// ── Work ledger ───────────────────────────────────────────────────────────────

/// The append-only session ledger — the shared truth the supervisor reads on
/// every turn.  Loaded from the store at the start of each effect; new entries
/// are persisted as each effect completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkLedger {
    pub task_id: String,
    pub goal: GoalSnapshot,
    pub entries: Vec<LedgerEntryRecord>,
    pub current_revision: WorkspaceRevision,
}

impl WorkLedger {
    pub fn new(task_id: String, goal: GoalSnapshot) -> Self {
        Self {
            task_id,
            goal,
            entries: vec![],
            current_revision: WorkspaceRevision::default(),
        }
    }

    /// The N most recent entries (or all entries if fewer than N).
    pub fn recent(&self, n: usize) -> &[LedgerEntryRecord] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// All command observations in sequence order.
    pub fn command_observations(&self) -> Vec<&CommandObservation> {
        self.entries
            .iter()
            .filter_map(|r| {
                if let LedgerEntry::ToolObservation(ToolObservation::CommandFinished(obs)) =
                    &r.entry
                {
                    Some(obs)
                } else {
                    None
                }
            })
            .collect()
    }
}

// ── Agent budget ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBudget {
    pub max_turns: u32,
    pub turns_used: u32,
    pub max_tokens_per_turn: usize,
}

impl Default for AgentBudget {
    fn default() -> Self {
        Self {
            max_turns: 50,
            turns_used: 0,
            max_tokens_per_turn: 32_768,
        }
    }
}

// ── Agent frame ───────────────────────────────────────────────────────────────

/// The bounded context assembled from the ledger for each supervisor turn.
/// This is the input shape M5.4's supervisor model will receive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFrame {
    pub goal: GoalSnapshot,
    pub current_revision: WorkspaceRevision,
    pub recent_entries: Vec<LedgerEntryRecord>,
    pub remaining_budget: AgentBudget,
}

impl AgentFrame {
    /// Assemble a frame from a ledger, keeping the N most recent entries.
    pub fn from_ledger(ledger: &WorkLedger, recent_n: usize, budget: AgentBudget) -> Self {
        Self {
            goal: ledger.goal.clone(),
            current_revision: ledger.current_revision.clone(),
            recent_entries: ledger.recent(recent_n).to_vec(),
            remaining_budget: budget,
        }
    }

    /// Format recent entries as a compact context string for model prompts.
    /// Used until M5.4 introduces the full supervisor action protocol.
    pub fn format_context(&self) -> String {
        if self.recent_entries.is_empty() {
            return String::new();
        }
        let lines: Vec<String> = self
            .recent_entries
            .iter()
            .map(|r| format!("[{}] {}", r.sequence, r.entry.label()))
            .collect();
        format!(
            "Recent activity (last {} entries):\n{}",
            lines.len(),
            lines.join("\n")
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_output_truncates_at_limit() {
        let s = "a".repeat(100);
        let out = BoundedOutput::from_string(s, 10);
        assert!(out.truncated);
        assert_eq!(out.original_bytes, 100);
        assert!(out.content.starts_with("aaaaaaaaaa"));
    }

    #[test]
    fn bounded_output_no_truncate_under_limit() {
        let out = BoundedOutput::from_string("hello", 1024);
        assert!(!out.truncated);
        assert_eq!(out.content, "hello");
    }

    #[test]
    fn ledger_entry_label_model_action() {
        let entry = LedgerEntry::ModelAction(ActionRecord {
            call_id: None,
            phase: "clarify".into(),
            tool: "clarify_node".into(),
            elapsed_ms: 500,
            outcome: "valid".into(),
            summary: "clarified the request".into(),
        });
        assert_eq!(entry.label(), "model:clarify valid");
    }

    #[test]
    fn ledger_entry_label_command_finished() {
        let entry =
            LedgerEntry::ToolObservation(ToolObservation::CommandFinished(CommandObservation {
                command: CommandSpec::new("npm", ["test"]),
                status: CommandStatus::Completed,
                exit_code: Some(0),
                stdout: BoundedOutput::from_string("Tests passed", 1024),
                stderr: BoundedOutput::empty(),
                parsed: vec![],
                workspace_delta: None,
                elapsed_ms: 2_000,
            }));
        assert!(entry.label().contains("npm test"));
    }

    #[test]
    fn work_ledger_recent_returns_last_n() {
        let mut ledger = WorkLedger::new("task-1".into(), GoalSnapshot::default());
        for i in 0..5u64 {
            ledger.entries.push(LedgerEntryRecord {
                id: LedgerEntryId(format!("e{i}")),
                task_id: "task-1".into(),
                sequence: i,
                entry: LedgerEntry::ContextSummary(ContextSummary {
                    covering_entry_ids: vec![],
                    summary: format!("entry {i}"),
                    durable_facts: vec![],
                }),
                created_at: OffsetDateTime::UNIX_EPOCH,
            });
        }
        let recent = ledger.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].sequence, 2);
        assert_eq!(recent[2].sequence, 4);
    }

    #[test]
    fn agent_frame_format_context_empty() {
        let frame = AgentFrame::from_ledger(
            &WorkLedger::new("t".into(), GoalSnapshot::default()),
            10,
            AgentBudget::default(),
        );
        assert!(frame.format_context().is_empty());
    }
}
