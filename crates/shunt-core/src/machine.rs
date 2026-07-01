//! State machine contract for the Frame runtime.
//!
//! These types form the **pure** control layer:
//!
//!   TaskState   — where a task is right now (5 variants — M5.3)
//!   Command     — external input from clients (TUI, CLI, future editor)
//!   MachineEvent— results produced by effects, fed back into the machine
//!   Effect      — side effects the machine declares; executed by EffectRunner
//!   AutonomyPolicy — per-gate decide-or-ask policy
//!
//! None of these types perform IO.  The transition function
//! `fn transition(TaskState, MachineEvent, &AutonomyPolicy) -> (TaskState, Vec<Effect>)`
//! lives in shunt-runtime and is the only place control flow changes.

use serde::{Deserialize, Serialize};

use crate::{
    AmbiguityId, ArtifactId, CommandOutcome, CommandSpec, EvidenceRef, ExecutionStageKind,
    FrontierCaseId, FrontierReason, RecipeRef, StageStatus, TaskPhase, UncertaintyEvent,
};

// ── FileDiff ──────────────────────────────────────────────────────────────────

/// A line-level diff for a single file operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileDiff {
    pub path: String,
    pub lines: Vec<DiffLine>,
}

/// A single line in a unified diff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiffLine {
    Added(String),
    Removed(String),
    Context(String),
}

// ── TaskState ────────────────────────────────────────────────────────────────

/// All possible states a task can be in.  M5.3: collapsed to 4 variants.
///
/// Cognitive activity labels (clarifying, understanding, executing, …) are
/// `Note` notifications and ledger entries — not state variants.  Adding a new
/// capability does not require a new state arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskState {
    /// Agent is doing work autonomously (observe, infer, apply, setup, …).
    Running,

    /// Agent is paused waiting for a specific user response.
    WaitingForUser { request: UserRequest },

    /// Task completed successfully.
    Completed,

    /// Task stopped — cancelled, failed, or frontier raised.
    Stopped { reason: StopReason },
}

/// What kind of user input the agent needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UserRequest {
    /// One or more open clarification questions. `open[0]` is the current one.
    Clarification {
        open: Vec<PendingAmbiguity>,
        confidence: f32,
    },
    /// The coding agent paused via its `ask_user` tool.
    AgentQuestion {
        question_id: AmbiguityId,
        question: String,
        context: String,
    },
    /// User must approve or reject the proposed plan before execution.
    Approval {
        candidate_count: usize,
        snapshot: ArtifactSnapshot,
    },
    /// Dangerous commands (rm, sudo, …) detected; user must approve or skip.
    DangerousCommands {
        commands: Vec<CommandSpec>,
        reason: String,
    },
}

/// Why the task stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    Failed { reason: String },
    Cancelled,
    FrontierRaised { case: FrontierCaseId },
}

impl TaskState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Stopped { .. })
    }

    pub fn is_waiting(&self) -> bool {
        matches!(self, Self::WaitingForUser { .. })
    }

    /// Human-readable label for the TUI status strip.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::WaitingForUser { request } => match request {
                UserRequest::Clarification { .. } => "waiting · answer needed",
                UserRequest::AgentQuestion { .. } => "waiting · agent question",
                UserRequest::Approval { .. } => "waiting · approval needed",
                UserRequest::DangerousCommands { .. } => "waiting · dangerous commands",
            },
            Self::Completed => "completed",
            Self::Stopped { reason } => match reason {
                StopReason::Failed { .. } => "failed",
                StopReason::Cancelled => "cancelled",
                StopReason::FrontierRaised { .. } => "frontier raised",
            },
        }
    }

    /// Coarse `TaskPhase` for backward-compat with the store display field.
    pub fn as_phase(&self) -> Option<TaskPhase> {
        match self {
            Self::Running => None,
            Self::WaitingForUser { request } => match request {
                UserRequest::Clarification { .. } => Some(TaskPhase::Clarify),
                UserRequest::AgentQuestion { .. } => Some(TaskPhase::Execute),
                UserRequest::Approval { .. } => Some(TaskPhase::Agree),
                UserRequest::DangerousCommands { .. } => Some(TaskPhase::Execute),
            },
            Self::Completed | Self::Stopped { .. } => None,
        }
    }
}

// ── Command ──────────────────────────────────────────────────────────────────

/// External input from a client (TUI, CLI, future editor / ACP).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Submit a new task request.
    Submit { request: String },

    /// Answer an open clarifying ambiguity.
    Answer {
        ambiguity_id: AmbiguityId,
        answer: String,
    },

    /// Approve the plan (candidate files) and allow execution to begin.
    Approve,

    /// Reject the plan; task returns to Clarify.
    Reject,

    /// Revise the active goal mid-run and re-localize.
    Steer { message: String },

    /// Approve dangerous commands and allow them to run.
    ApproveDangerousCommands,

    /// Skip dangerous commands and complete without them.
    RejectDangerousCommands,

    /// Cancel the running task.
    Cancel,
}

// ── ArtifactPatch ────────────────────────────────────────────────────────────

/// A partial update to the understanding artifact.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactPatch {
    pub interpreted_goal: Option<String>,
    pub success_criteria: Option<Vec<String>>,
    pub constraints: Option<Vec<String>>,
    pub target_scope: Option<Vec<String>>,
    pub evidence: Option<Vec<EvidenceRef>>,
    pub assumptions: Option<Vec<crate::Assumption>>,
    pub ambiguities: Option<Vec<crate::Ambiguity>>,
    pub selected_recipe: Option<RecipeRef>,
    pub risks: Option<Vec<crate::Risk>>,
    pub confidence: Option<f32>,
    pub approval: Option<crate::ApprovalState>,
}

// ── MachineEvent ─────────────────────────────────────────────────────────────

/// Input events that drive the state machine.
#[derive(Debug, Clone)]
pub enum MachineEvent {
    ObserveCompleted,

    ClarifyCompleted {
        confidence: f32,
        open: Vec<PendingAmbiguity>,
        snapshot: ArtifactSnapshot,
    },

    UnderstandCompleted {
        confidence: f32,
        has_open_ambiguity: bool,
        snapshot: ArtifactSnapshot,
    },

    LocalizeCompleted {
        candidate_count: usize,
        confidence: f32,
        snapshot: ArtifactSnapshot,
    },

    /// The agent produced a change set (surfaced to the client). The machine
    /// decides — per `AutonomyPolicy` — whether to pause for approval or commit.
    ProposalReady {
        confidence: f32,
        op_count: usize,
        command_count: usize,
        snapshot: ArtifactSnapshot,
    },

    /// The agent paused to ask the developer a question (its `ask_user` tool).
    /// Drives the machine into `WaitingForUser::Clarification`.
    AgentAsked {
        ambiguity_id: String,
        question: String,
        context: String,
        options: Vec<String>,
    },

    /// A patch was applied to disk.  Carries setup commands the LLM proposed.
    PatchApplied {
        path: String,
        setup_commands: Vec<CommandSpec>,
    },

    /// All setup commands ran.
    SetupCompleted {
        outcomes: Vec<CommandOutcome>,
    },

    UncertaintyRaised(UncertaintyEvent),
    FrontierCreated {
        case_id: FrontierCaseId,
    },

    /// Dangerous commands were detected in the proposed change set.
    DangerCommandsDetected {
        commands: Vec<CommandSpec>,
        reason: String,
    },

    UserCommand(Command),

    EffectError {
        effect: String,
        reason: String,
    },
}

// ── Effect ───────────────────────────────────────────────────────────────────

/// Side effects declared by the transition function.
#[derive(Debug, Clone)]
pub enum Effect {
    Observe {
        workspace_root: String,
        artifact_id: ArtifactId,
        request: String,
    },
    RecordAnswer {
        artifact_id: ArtifactId,
        ambiguity_id: AmbiguityId,
        answer: String,
    },
    ResumeAgent {
        artifact_id: ArtifactId,
        question_id: AmbiguityId,
        answer: String,
    },
    ApplyArtifactPatch {
        artifact_id: ArtifactId,
        patch: Box<ArtifactPatch>,
    },
    RecordApproval {
        artifact_id: ArtifactId,
        approved: bool,
        note: Option<String>,
    },
    /// Run the agent to generate a change set and surface it (diffs, ops) — but
    /// do NOT write to disk. Produces `ProposalReady` (or `AgentAsked` if the
    /// agent needs input). The approval gate sits between this and `CommitChange`.
    ProposeChange {
        artifact_id: ArtifactId,
    },
    /// Apply the already-proposed change set to disk. Produces `PatchApplied`.
    CommitChange {
        artifact_id: ArtifactId,
    },
    RunSetup {
        artifact_id: ArtifactId,
    },
    Persist,
    Notify(Notification),
    RaiseFrontier {
        artifact_id: ArtifactId,
        reason: FrontierReason,
    },
}

// ── ArtifactSnapshot ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactSnapshot {
    pub interpreted_goal: String,
    pub confidence: f32,
    pub evidence_count: usize,
    pub candidate_paths: Vec<String>,
    pub open_ambiguity_count: usize,
    pub open_risks: Vec<String>,
}

// ── Notification ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Notification {
    TaskStarted,
    ObserveStarted,
    ObserveFinished {
        summary: String,
    },
    PhaseEntered {
        phase: TaskPhase,
        summary: String,
    },
    ModelCallStarted {
        phase: TaskPhase,
    },
    InferenceCallStarted {
        call_id: u64,
        tool: String,
        model: String,
        mode: String,
    },
    InferenceCallFinished {
        call_id: u64,
        tool: String,
        elapsed_ms: u64,
        outcome: String,
    },
    /// Streamed token chunk.  `is_thinking` = true for reasoning_content tokens.
    InferenceToken {
        call_id: u64,
        text: String,
        is_thinking: bool,
    },
    ModelCallFinished {
        phase: TaskPhase,
        summary: String,
        snapshot: Option<ArtifactSnapshot>,
    },
    LocalizeStarted,
    LocalizeFinished {
        summary: String,
        snapshot: ArtifactSnapshot,
    },
    ClarificationNeeded {
        ambiguity_id: String,
        question: String,
        options: Vec<String>,
        confidence: f32,
    },
    ApprovalNeeded {
        candidate_count: usize,
        snapshot: ArtifactSnapshot,
    },
    DangerousCommandsProposed {
        commands: Vec<CommandSpec>,
        reason: String,
    },
    ChangeProposed {
        description: String,
        ops: Vec<String>,
        commands: Vec<String>,
        diffs: Vec<FileDiff>,
    },
    SetupCommandStarted {
        display: String,
    },
    SetupStarted {
        count: usize,
    },
    SetupFinished {
        summary: String,
    },
    /// Agent chose a tool and is about to execute it.
    AgentToolCall {
        turn: usize,
        max_turns: usize,
        tool: String,
        summary: String,
    },
    /// Result of the agent's last tool call.
    AgentToolResult {
        turn: usize,
        ok: bool,
        detail: String,
    },
    UserInputNeeded {
        summary: String,
    },
    StageChanged {
        stage: ExecutionStageKind,
        status: StageStatus,
    },
    FrontierRaised {
        reason: FrontierReason,
        summary: String,
    },
    RunCompleted {
        summary: String,
    },
    RunFailed {
        summary: String,
    },
    /// A git stash was created before applying changes — undo is available.
    UndoAvailable,
    /// Plain informational text emitted by runners.
    Note {
        text: String,
    },
}

// ── AutonomyPolicy ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyPolicy {
    /// Gate before completing clarify when ambiguities exist.
    pub clarify: GateDecision,
    /// Gate before beginning execution (the Agree step).
    pub approval: GateDecision,
    /// Gate before running shell commands / tests.
    pub run_commands: GateDecision,
}

impl AutonomyPolicy {
    /// Single agentic mode: autonomous after the user approves the plan.
    pub fn agentic() -> Self {
        Self {
            clarify: GateDecision::AutoIfConfident { min: 0.60 },
            approval: GateDecision::Ask,
            run_commands: GateDecision::Auto,
        }
    }

    /// Fully autonomous headless mode: never pauses for a human. Used by
    /// `agent --once` and the in-process bench. The same core machine as the
    /// TUI — only the gate policy differs. Any `WaitingForUser` that still
    /// arises (e.g. an agent `ask_user`) is resolved by the driver's
    /// `Responder`, not by a human.
    pub fn headless() -> Self {
        Self {
            clarify: GateDecision::Auto,
            approval: GateDecision::Auto,
            run_commands: GateDecision::Auto,
        }
    }

    pub fn should_ask(&self, gate: GateKind, confidence: f32) -> bool {
        let decision = match gate {
            GateKind::Clarify => &self.clarify,
            GateKind::Approval => &self.approval,
            GateKind::RunCommands => &self.run_commands,
        };
        match decision {
            GateDecision::Ask => true,
            GateDecision::Auto => false,
            GateDecision::AutoIfConfident { min } => confidence < *min,
        }
    }
}

impl Default for AutonomyPolicy {
    fn default() -> Self {
        Self::agentic()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateDecision {
    Ask,
    Auto,
    AutoIfConfident { min: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateKind {
    Clarify,
    Approval,
    RunCommands,
}

// ── PendingAmbiguity ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingAmbiguity {
    pub id: AmbiguityId,
    pub question: String,
    pub options: Vec<String>,
}
