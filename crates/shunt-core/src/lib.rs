//! Core task objects for the Frame runtime.
//!
//! The runtime should revolve around a small set of stable objects:
//! understanding artifacts, recipe runs, uncertainty events, frontier cases,
//! correction packages, and adaptation packages.

pub mod ledger;
pub mod machine;
pub mod safety;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArtifactId(pub String);

/// Stable identifier for a single ambiguity within an understanding artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AmbiguityId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecipeRunId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FrontierCaseId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorrectionPackageId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdaptationPackageId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskPhase {
    Observe,
    Clarify,
    Understand,
    Localize,
    Agree,
    Execute,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeEventKind {
    TaskStarted,
    ObservationStarted,
    ObservationFinished,
    PhaseChanged,
    ModelCallStarted,
    ModelCallFinished,
    LocalizationStarted,
    SearchPlanned,
    SearchResultsUpdated,
    LocalizationFinished,
    ManualContextUpdated,
    UserInputRequested,
    ExecutionStageChanged,
    VerifierFinished,
    FrontierRecorded,
    RunCompleted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEvent {
    pub task_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe_run_id: Option<String>,
    pub kind: RuntimeEventKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<TaskPhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<ExecutionStageKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_status: Option<StageStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontier_reason: Option<FrontierReason>,
    pub summary: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalStatus {
    Draft,
    NeedsReview,
    Approved,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalState {
    pub status: ApprovalStatus,
    pub decided_by: Option<String>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub decided_at: Option<OffsetDateTime>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceKind {
    File,
    Symbol,
    UserInput,
    CommandOutput,
    Trace,
    Other,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub kind: EvidenceKind,
    pub locator: String,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackageVersionProvenance {
    ExactLock,
    ManifestRequirement,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackageFact {
    pub ecosystem: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requirement: Option<String>,
    pub version_provenance: PackageVersionProvenance,
    pub manifest_path: String,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManualVersionStatus {
    Exact,
    CompatibleRange,
    Unversioned,
    Mismatch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManualEvidence {
    pub ecosystem: String,
    pub package: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub version_status: ManualVersionStatus,
    pub source: String,
    pub locator: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub excerpt: String,
    pub relevance_reason: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManualQuery {
    pub original_request: String,
    pub interpreted_goal: String,
    pub located_paths: Vec<String>,
    pub requested_topics: Vec<String>,
    pub package_facts: Vec<PackageFact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssumptionStatus {
    Active,
    Confirmed,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Assumption {
    pub id: String,
    pub statement: String,
    pub evidence: Vec<EvidenceRef>,
    pub status: AssumptionStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmbiguityStatus {
    Open,
    Resolved,
}

/// Whether the ambiguity can be resolved by an automated lookup or requires a user decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AmbiguityKind {
    /// A factual question the agent can resolve by querying a package registry or docs
    /// (e.g. "what is the latest version of @remix-run/dev?").
    Lookup,
    /// A genuine choice only the user can make (architectural direction, scope, preference).
    #[default]
    UserDecision,
    /// A conflict or architectural fork in the road detected by static workspace analysis
    /// (e.g. "react-router is already bundled by Remix v2 — which approach should we use?").
    /// Always requires user input; the agent never auto-resolves these.
    ApproachChoice,
}

/// Structured snapshot of the workspace extracted deterministically before any LLM call.
/// Gives the clarify model concrete facts to reason about conflicts and topology.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceProfile {
    /// Detected runtimes: "node", "rust", "python", "go", …
    pub runtimes: Vec<String>,
    /// Framework markers with detected versions: "remix@2.5.0", "express@4", "vite@5"
    pub frameworks: Vec<String>,
    /// All dependency names (no versions) for quick conflict lookup
    pub dependencies: Vec<String>,
    /// Coarse project shape: "single-app" | "monorepo" | "library" | "service" | "unknown"
    pub topology: String,
    /// Statically detected conflicts: e.g. "react-router is a transitive dep of remix — adding it directly may cause version conflicts"
    pub conflicts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ambiguity {
    pub id: String,
    pub question: String,
    pub options: Vec<String>,
    /// Whether this can be auto-resolved by a registry/docs lookup or needs user input.
    #[serde(default)]
    pub kind: AmbiguityKind,
    pub status: AmbiguityStatus,
    pub resolution: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Risk {
    pub id: String,
    pub summary: String,
    pub severity: RiskSeverity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipeRef {
    pub id: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnderstandingArtifact {
    pub id: ArtifactId,
    pub task_id: TaskId,
    pub original_request: String,
    pub interpreted_goal: String,
    pub success_criteria: Vec<String>,
    pub constraints: Vec<String>,
    pub target_scope: Vec<String>,
    pub evidence: Vec<EvidenceRef>,
    #[serde(default)]
    pub candidate_files: Vec<CandidateFile>,
    #[serde(default)]
    pub package_facts: Vec<PackageFact>,
    #[serde(default)]
    pub manual_evidence: Vec<ManualEvidence>,
    pub assumptions: Vec<Assumption>,
    pub ambiguities: Vec<Ambiguity>,
    pub selected_recipe: Option<RecipeRef>,
    pub risks: Vec<Risk>,
    pub confidence: f32,
    pub approval: ApprovalState,
    pub revision: u32,
    /// Deterministic workspace snapshot populated during Observe before any LLM call.
    /// The clarify model uses this to detect conflicts and raise ApproachChoice ambiguities.
    #[serde(default)]
    pub workspace_profile: WorkspaceProfile,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateFile {
    pub path: String,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStageKind {
    Inspect,
    Propose,
    Verify,
    Apply,
    /// Runs after Apply: install dependencies, build steps, etc.
    /// Proposed by the LLM alongside the patch; gated by the run_commands policy.
    Setup,
    Validate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageStatus {
    Pending,
    Running,
    Passed,
    Failed,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifierStatus {
    Passed,
    Failed,
    Warning,
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifierOutcome {
    pub verifier: String,
    pub status: VerifierStatus,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageRecord {
    pub kind: ExecutionStageKind,
    pub status: StageStatus,
    pub summary: String,
    pub verifiers: Vec<VerifierOutcome>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub started_at: Option<OffsetDateTime>,
    #[serde(with = "time::serde::rfc3339::option")]
    pub finished_at: Option<OffsetDateTime>,
}

/// A single file operation within a `ChangeSet`.
/// Applied in order; the whole set rolls back atomically if any op fails.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum FileOp {
    /// Create a new file (or overwrite if it already exists).
    Create { path: String, contents: String },
    /// Apply a search-and-replace edit to an existing file.
    Edit {
        path: String,
        search: String,
        replacement: String,
    },
    /// Delete a file from the workspace.
    Delete { path: String },
}

impl FileOp {
    pub fn path(&self) -> &str {
        match self {
            FileOp::Create { path, .. } | FileOp::Edit { path, .. } | FileOp::Delete { path } => {
                path
            }
        }
    }
}

/// A multi-file atomic change plan: file operations + follow-up shell commands.
/// Applied all-or-nothing; partially applied ops are rolled back on failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChangeSet {
    pub ops: Vec<FileOp>,
    /// Shell commands to run after all ops succeed (install deps, build steps, etc.).
    #[serde(default)]
    pub commands: Vec<CommandSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposedChange {
    pub path: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contents: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecipeRun {
    pub id: RecipeRunId,
    pub task_id: TaskId,
    pub recipe: RecipeRef,
    pub current_stage: ExecutionStageKind,
    pub stages: Vec<StageRecord>,
    /// The atomic change set for this run (file ops + setup commands).
    #[serde(default)]
    pub change_set: Option<ChangeSet>,
    /// Outcomes of each executed setup command.
    #[serde(default)]
    pub setup_outcomes: Vec<CommandOutcome>,
    /// Legacy: kept for backward compat with stored records; not written by new code.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proposed_changes: Vec<ProposedChange>,
    /// Legacy: kept for backward compat with stored records; not written by new code.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub setup_commands: Vec<CommandSpec>,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

/// A structured shell command.  Programs and args are separate to avoid
/// shell injection — the executor uses `std::process::Command`, not a shell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSpec {
    pub program: String,
    pub args: Vec<String>,
}

impl CommandSpec {
    pub fn new(
        program: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            program: program.into(),
            args: args.into_iter().map(|a| a.into()).collect(),
        }
    }

    pub fn display(&self) -> String {
        if self.args.is_empty() {
            self.program.clone()
        } else {
            format!("{} {}", self.program, self.args.join(" "))
        }
    }
}

/// Result of executing a single `CommandSpec`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandOutcome {
    pub spec: CommandSpec,
    pub exit_code: i32,
    /// Last 2 KB of stdout.
    pub stdout: String,
    /// Last 2 KB of stderr.
    pub stderr: String,
    pub success: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UncertaintyKind {
    Ambiguity,
    MissingEvidence,
    LowConfidence,
    VerifierFailure,
    VerifierDisagreement,
    RetryExhausted,
    RecipeOscillation,
    ToolThrash,
    UserCorrection,
    PatchRejection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyEvent {
    pub task_id: TaskId,
    pub stage: Option<ExecutionStageKind>,
    pub kind: UncertaintyKind,
    pub summary: String,
    pub confidence: Option<f32>,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrontierReason {
    LowConfidence,
    VerifierFailure,
    RepeatedVerifierFailure,
    RecipeInstability,
    ToolChurn,
    MaterialUserCorrection,
    RepeatedPatchFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrontierStatus {
    Open,
    InReview,
    Corrected,
    Promoted,
    Closed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontierCase {
    pub id: FrontierCaseId,
    pub task_id: TaskId,
    pub artifact_id: ArtifactId,
    pub recipe_run_id: Option<RecipeRunId>,
    pub reason: FrontierReason,
    pub status: FrontierStatus,
    pub summary: String,
    pub uncertainty_events: Vec<UncertaintyEvent>,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionKind {
    ArtifactRevision,
    RecipeRevision,
    PatchRevision,
    NodeRevision,
    SupervisorTrace,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectionPackage {
    pub id: CorrectionPackageId,
    pub frontier_case_id: FrontierCaseId,
    pub kind: CorrectionKind,
    pub summary: String,
    pub validated: bool,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromotionTarget {
    Recipe,
    Node,
    Threshold,
    Dataset,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromotionStatus {
    Proposed,
    Validated,
    Promoted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptationPackage {
    pub id: AdaptationPackageId,
    pub source_corrections: Vec<CorrectionPackageId>,
    pub target: PromotionTarget,
    pub status: PromotionStatus,
    pub summary: String,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskRun {
    pub id: TaskId,
    pub workspace_root: String,
    pub phase: TaskPhase,
    pub current_artifact: ArtifactId,
    pub active_recipe_run: Option<RecipeRunId>,
    pub frontier_cases: Vec<FrontierCaseId>,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

impl ApprovalState {
    pub fn draft() -> Self {
        Self {
            status: ApprovalStatus::Draft,
            decided_by: None,
            decided_at: None,
            note: None,
        }
    }
}

impl ProposedChange {
    pub fn full_replace(
        path: impl Into<String>,
        description: impl Into<String>,
        contents: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            description: description.into(),
            search: None,
            replacement: None,
            contents: Some(contents.into()),
        }
    }

    pub fn search_replace(
        path: impl Into<String>,
        description: impl Into<String>,
        search: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            description: description.into(),
            search: Some(search.into()),
            replacement: Some(replacement.into()),
            contents: None,
        }
    }
}
