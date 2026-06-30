pub mod driver;
pub mod executor;
pub mod machine;
pub mod probes;
pub mod runner;
pub mod session;

use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use crate::probes::ScopeOrchestrator;
use shunt_core::ledger::{AgentBudget, AgentFrame, GoalSnapshot, WorkLedger};
use shunt_core::{
    AmbiguityKind, AmbiguityStatus, ApprovalState, ApprovalStatus, ArtifactId, CandidateFile,
    ChangeSet, CorrectionKind, CorrectionPackage, EvidenceKind, EvidenceRef, ExecutionStageKind,
    FileOp, FrontierCase, FrontierCaseId, FrontierReason, FrontierStatus, RecipeRef, RecipeRun,
    RecipeRunId, RequiredPathIntent, RuntimeEvent, RuntimeEventKind, StageRecord, StageStatus,
    TaskId, TaskPhase, TaskRun, UncertaintyEvent, UncertaintyKind, UnderstandingArtifact,
    VerifierOutcome, VerifierStatus,
};
use shunt_infer::{
    AgentObserver, AgentResult, AgentSession, ClarifyNode, SourceFileContext, ToolProvider,
    UnderstandNode,
};
use shunt_knowledge::KnowledgeService;
use shunt_localize::{ContextPacket, Localizer, RetrievalBackend, SearchQuery, SemanticLocalizer};
use shunt_store::{SqliteStore, StoreError};
use thiserror::Error;
use time::OffsetDateTime;

const MAX_REPAIR_DIAGNOSTICS: usize = 4;
const MAX_REPAIR_OUTPUT_CHARS: usize = 2000;
const PROPOSE_SESSION_WALL_TIMEOUT: Duration = Duration::from_secs(300);
const VERIFIER_SESSION_WALL_TIMEOUT: Duration = Duration::from_secs(90);
const REPAIR_SESSION_WALL_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, PartialEq, Eq)]
struct RepairDiagnostic {
    source: &'static str,
    command: Option<String>,
    summary: String,
    output: String,
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("store error: {0}")]
    Store(#[from] StoreError),
    #[error("infer error: {0}")]
    Infer(#[from] shunt_infer::InferError),
    #[error("localize error: {0}")]
    Localize(#[from] shunt_localize::LocalizeError),
    #[error("knowledge error: {0}")]
    Knowledge(#[from] shunt_knowledge::KnowledgeError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ambiguity not found: {0}")]
    AmbiguityNotFound(String),
    #[error("task not found: {0}")]
    TaskNotFound(String),
    #[error("frontier case not found: {0}")]
    FrontierCaseNotFound(String),
    #[error("correction package not found: {0}")]
    CorrectionPackageNotFound(String),
    #[error("artifact not approved for execution")]
    ArtifactNotApproved,
    #[error("active recipe run not found")]
    ActiveRecipeRunMissing,
    #[error("frontier case is not linked to a recipe run")]
    FrontierRecipeRunMissing,
    #[error("no change set available to apply")]
    NoChangeSet,
    #[error("no candidate source files available for change generation")]
    NoChangeCandidates,
    #[error("generated change path is invalid: {0}")]
    InvalidGeneratedChangePath(String),
    #[error("generated change contents are empty")]
    EmptyGeneratedChange,
    #[error("generated patch search block is invalid")]
    InvalidGeneratedPatch,
    #[error("target scope is not concrete enough for execution")]
    UnlocalizedTask,
    #[error("patch correction requires a frontier in validate/apply replay")]
    PatchCorrectionUnsupported,
    #[error("invalid stage transition: expected {expected:?}, found {found:?}")]
    InvalidStageTransition {
        expected: ExecutionStageKind,
        found: ExecutionStageKind,
    },
    #[error("apply stage has not passed yet")]
    ApplyNotPassed,
    #[error("agent needs clarification: {question}")]
    AgentNeedsClarification { question: String, context: String },
    #[error("{0}")]
    Other(String),
}

pub type RuntimeResult<T> = Result<T, RuntimeError>;

pub trait RuntimeObserver: Send + Sync {
    fn on_event(&self, event: RuntimeEvent);
}

struct NoopRuntimeObserver;

impl RuntimeObserver for NoopRuntimeObserver {
    fn on_event(&self, _event: RuntimeEvent) {}
}

pub struct TaskRuntime {
    pub(crate) store: SqliteStore,
    knowledge: KnowledgeService,
    orchestrator: ScopeOrchestrator,
    localizer: SemanticLocalizer,
    observer: Arc<dyn RuntimeObserver>,
    agent_observer: Option<Arc<dyn AgentObserver + Send + Sync>>,
    /// Project-level budget overrides — layered on top of the model-derived budget.
    budget_override: Option<shunt_infer::SessionBudgetOverride>,
}

#[derive(Debug, Clone)]
pub struct HandleResult {
    pub task: TaskRun,
    pub artifact: UnderstandingArtifact,
    pub active_recipe_run: Option<RecipeRun>,
    pub frontier_cases: Vec<FrontierCase>,
}

impl TaskRuntime {
    pub fn new(store: SqliteStore) -> Self {
        Self {
            store,
            knowledge: KnowledgeService::default(),
            orchestrator: ScopeOrchestrator::default(),
            localizer: SemanticLocalizer::default(),

            observer: Arc::new(NoopRuntimeObserver),
            agent_observer: None,
            budget_override: None,
        }
    }

    pub fn with_knowledge(store: SqliteStore, knowledge: KnowledgeService) -> Self {
        Self {
            store,
            knowledge,
            orchestrator: ScopeOrchestrator::default(),
            localizer: SemanticLocalizer::default(),

            observer: Arc::new(NoopRuntimeObserver),
            agent_observer: None,
            budget_override: None,
        }
    }

    pub fn with_observer(store: SqliteStore, observer: Arc<dyn RuntimeObserver>) -> Self {
        Self {
            store,
            knowledge: KnowledgeService::default(),
            orchestrator: ScopeOrchestrator::default(),
            localizer: SemanticLocalizer::default(),

            observer,
            agent_observer: None,
            budget_override: None,
        }
    }

    pub fn with_services(
        store: SqliteStore,
        knowledge: KnowledgeService,
        observer: Arc<dyn RuntimeObserver>,
    ) -> Self {
        Self {
            store,
            knowledge,
            orchestrator: ScopeOrchestrator::default(),
            localizer: SemanticLocalizer::default(),

            observer,
            agent_observer: None,
            budget_override: None,
        }
    }

    pub fn set_budget_override(&mut self, o: shunt_infer::SessionBudgetOverride) {
        self.budget_override = Some(o);
    }

    pub fn set_agent_observer(&mut self, obs: Arc<dyn AgentObserver + Send + Sync>) {
        self.agent_observer = Some(obs);
    }

    pub fn into_store(self) -> SqliteStore {
        self.store
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_runtime_event(
        &self,
        now: OffsetDateTime,
        task_id: impl Into<String>,
        artifact_id: Option<String>,
        recipe_run_id: Option<String>,
        kind: RuntimeEventKind,
        phase: Option<TaskPhase>,
        stage: Option<ExecutionStageKind>,
        stage_status: Option<StageStatus>,
        frontier_reason: Option<FrontierReason>,
        summary: impl Into<String>,
    ) {
        self.observer.on_event(RuntimeEvent {
            task_id: task_id.into(),
            artifact_id,
            recipe_run_id,
            kind,
            phase,
            stage,
            stage_status,
            frontier_reason,
            summary: summary.into(),
            created_at: now,
        });
    }

    fn localize_context_packet(
        &self,
        workspace_root: &str,
        artifact: &UnderstandingArtifact,
    ) -> RuntimeResult<ContextPacket> {
        Ok(self.localizer.localize(workspace_root, artifact)?)
    }

    fn observe_task(
        &self,
        artifact_id: &str,
        now: OffsetDateTime,
    ) -> RuntimeResult<Option<UnderstandingArtifact>> {
        tracing::debug!("observe_task artifact={artifact_id}");
        let Some(mut artifact) = self.store.get_understanding_artifact(artifact_id)? else {
            return Ok(None);
        };
        let Some(mut task) = self.store.get_task_run(&artifact.task_id.0)? else {
            return Ok(Some(artifact));
        };

        artifact.evidence = observe_workspace_evidence(&task.workspace_root, &artifact, self)?;
        artifact.workspace_profile = extract_workspace_profile(&task.workspace_root);
        artifact.revision += 1;
        artifact.updated_at = now;

        task.phase = TaskPhase::Clarify;
        task.updated_at = now;

        self.store.put_understanding_artifact(&artifact)?;
        self.store.put_task_run(&task)?;

        Ok(Some(artifact))
    }

    fn localize_task_with_packet(
        &self,
        artifact_id: &str,
        now: OffsetDateTime,
        provider: Option<&dyn ToolProvider>,
    ) -> RuntimeResult<Option<(UnderstandingArtifact, ContextPacket)>> {
        tracing::debug!("localize_task artifact={artifact_id}");
        let Some(mut artifact) = self.store.get_understanding_artifact(artifact_id)? else {
            return Ok(None);
        };
        let Some(mut task) = self.store.get_task_run(&artifact.task_id.0)? else {
            return Ok(Some((artifact, empty_context_packet())));
        };

        let previous_candidates = artifact
            .candidate_files
            .iter()
            .map(|candidate| candidate.path.clone())
            .collect::<Vec<_>>();

        // ── probe-and-compose scope ───────────────────────────────────────────
        // A no-op fallback provider for orchestrator calls when no real provider
        // is available (e.g. localize_task public API).
        struct NoopProvider;
        impl ToolProvider for NoopProvider {
            fn call_tool(
                &self,
                _system: &str,
                _user: &str,
                _tool: &shunt_infer::ToolSpec,
            ) -> shunt_infer::InferResult<shunt_infer::ToolCall> {
                Err(shunt_infer::InferError::EmptyResponse)
            }
        }

        let orch_result = match provider {
            Some(p) => self.orchestrator.run(&task.workspace_root, &artifact, p),
            None => self
                .orchestrator
                .run(&task.workspace_root, &artifact, &NoopProvider),
        };

        tracing::debug!(
            scope = ?orch_result.target_scope,
            probes = orch_result.probe_log.len(),
            "orchestrator composed scope"
        );

        prune_candidate_evidence(&mut artifact, &previous_candidates);
        artifact.candidate_files = orch_result.candidate_files;
        if !orch_result.target_scope.is_empty() {
            artifact.target_scope = orch_result.target_scope;
            for ev in orch_result.evidence {
                if !artifact.evidence.iter().any(|e| e.locator == ev.locator) {
                    artifact.evidence.push(ev);
                }
            }
            artifact.confidence = artifact.confidence.max(0.72);
        }

        // Build a ContextPacket for the knowledge service (it reads package
        // manifests from candidates).  Only existing-file candidates are relevant.
        let packet = build_context_packet_from_scope(&artifact.candidate_files);

        let knowledge =
            self.knowledge
                .gather(Path::new(&task.workspace_root), &artifact, &packet)?;
        artifact.package_facts = knowledge.package_facts;
        artifact.manual_evidence = knowledge.manual_evidence;
        trace_manual_context(&artifact);
        artifact.revision += 1;
        artifact.updated_at = now;

        task.phase = TaskPhase::Agree;
        task.updated_at = now;

        self.store.put_understanding_artifact(&artifact)?;
        self.store.put_task_run(&task)?;

        Ok(Some((artifact, packet)))
    }

    fn gather_change_candidates(
        &self,
        workspace_root: &str,
        artifact: &UnderstandingArtifact,
    ) -> RuntimeResult<Vec<SourceFileContext>> {
        let root = Path::new(workspace_root);

        // Use target_scope paths already resolved by the orchestrator at localize
        // time rather than re-running the localizer. For paths that exist on disk,
        // read their contents; new scaffold paths have no content yet.
        let candidates: Vec<SourceFileContext> = artifact
            .target_scope
            .iter()
            .take(MAX_CHANGE_CANDIDATES)
            .filter_map(|path| {
                let abs = root.join(path);
                let contents = fs::read_to_string(&abs).unwrap_or_default();
                if contents.trim().is_empty() {
                    // New file — include with empty contents so propose knows to create it.
                    None
                } else {
                    Some(SourceFileContext {
                        path: path.clone(),
                        contents,
                    })
                }
            })
            .collect();

        // Agent path: target_scope is empty because the agent discovers files itself via
        // search_files / read_file tools. Return empty so nothing is pre-loaded —
        // the agent's system prompt already contains the dir tree and manifest files.
        // (The semantic localizer fallback was removed: it returned low-quality candidates
        // such as lock files and tsconfig.json for routing tasks.)
        Ok(candidates)
    }

    /// Resume a task that was paused at the clarify phase (e.g. waiting for
    /// an ambiguity answer).  Skips `start_task`, `observe_task`, and
    /// `clarify_task` — those have already run.  Picks up from `understand`.
    ///
    /// Use this after resolving an ambiguity to continue the pipeline without
    /// losing context.
    #[allow(clippy::too_many_arguments)]
    pub fn resume_from_understand<P>(
        &self,
        now: OffsetDateTime,
        task_id: &str,
        artifact_id: &str,
        recipe: RecipeRef,
        auto_approve: bool,
        run_execution: bool,
        apply_generated_change: bool,
        decided_by: impl Into<String>,
        approval_note: Option<String>,
        provider: &P,
    ) -> RuntimeResult<HandleResult>
    where
        P: ToolProvider,
    {
        tracing::debug!("resume_from_understand task={task_id}");
        let decided_by = decided_by.into();

        self.emit_runtime_event(
            now,
            task_id,
            Some(artifact_id.into()),
            None,
            RuntimeEventKind::PhaseChanged,
            Some(TaskPhase::Understand),
            None,
            None,
            None,
            "resuming from understand after ambiguity was resolved",
        );
        self.emit_runtime_event(
            now,
            task_id,
            Some(artifact_id.into()),
            None,
            RuntimeEventKind::ModelCallStarted,
            Some(TaskPhase::Understand),
            None,
            None,
            None,
            "understand model call started",
        );
        let understood = self.understand_task_with_provider(artifact_id, now, provider)?;
        self.emit_runtime_event(
            now,
            task_id,
            Some(artifact_id.into()),
            None,
            RuntimeEventKind::ModelCallFinished,
            Some(TaskPhase::Understand),
            None,
            None,
            None,
            understood
                .as_ref()
                .map(summarize_understand_artifact)
                .unwrap_or_else(|| "understand model call finished".into()),
        );

        self.emit_runtime_event(
            now,
            task_id,
            Some(artifact_id.into()),
            None,
            RuntimeEventKind::PhaseChanged,
            Some(TaskPhase::Localize),
            None,
            None,
            None,
            "task moved to localize phase",
        );
        self.emit_runtime_event(
            now,
            task_id,
            Some(artifact_id.into()),
            None,
            RuntimeEventKind::LocalizationStarted,
            Some(TaskPhase::Localize),
            None,
            None,
            None,
            "localization started",
        );
        let (artifact, packet) = self
            .localize_task_with_packet(artifact_id, now, Some(provider as &dyn ToolProvider))?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        self.emit_runtime_event(
            now,
            task_id,
            Some(artifact_id.into()),
            None,
            RuntimeEventKind::LocalizationFinished,
            Some(TaskPhase::Agree),
            None,
            None,
            None,
            format!(
                "localized {} candidate files",
                artifact.candidate_files.len()
            ),
        );
        let can_execute = is_execution_ready(&artifact);

        if auto_approve || ((run_execution || apply_generated_change) && can_execute) {
            self.approve_artifact(artifact_id, decided_by, approval_note, now)?;
            self.emit_runtime_event(
                now,
                task_id,
                Some(artifact_id.into()),
                None,
                RuntimeEventKind::PhaseChanged,
                Some(TaskPhase::Execute),
                None,
                None,
                None,
                "artifact approved and task moved to execute phase",
            );
        } else if run_execution || apply_generated_change {
            self.emit_runtime_event(
                now,
                task_id,
                Some(artifact_id.into()),
                None,
                RuntimeEventKind::RunCompleted,
                Some(TaskPhase::Agree),
                None,
                None,
                None,
                "run paused at agree because localization is too weak for execution",
            );
            return self.load_handle_result(task_id);
        }

        if run_execution || apply_generated_change {
            let recipe_run = self.start_execution(task_id, now, recipe)?;
            self.emit_runtime_event(
                now,
                task_id,
                Some(artifact_id.into()),
                Some(recipe_run.id.0.clone()),
                RuntimeEventKind::ExecutionStageChanged,
                Some(TaskPhase::Execute),
                Some(ExecutionStageKind::Inspect),
                Some(StageStatus::Running),
                None,
                "execution started at inspect stage",
            );
            self.execute_inspect(task_id, now)?;
            let recipe_run = self.execute_propose(task_id, now)?;
            self.emit_runtime_event(
                now,
                task_id,
                Some(artifact_id.into()),
                Some(recipe_run.id.0.clone()),
                RuntimeEventKind::ExecutionStageChanged,
                Some(TaskPhase::Execute),
                Some(ExecutionStageKind::Verify),
                Some(StageStatus::Running),
                None,
                "verify stage started",
            );
            let recipe_run = self.execute_verify(task_id, now)?;

            if recipe_run.current_stage == ExecutionStageKind::Apply && apply_generated_change {
                self.generate_proposed_change(task_id, now, provider, &[], &[])?;
                let recipe_run = self.execute_apply(task_id, now)?;
                let _ = recipe_run;
                self.execute_validate(task_id, now)?;
            }
        }

        let result = self.load_handle_result(task_id)?;
        self.emit_runtime_event(
            now,
            result.task.id.0.clone(),
            Some(result.artifact.id.0.clone()),
            result.active_recipe_run.as_ref().map(|r| r.id.0.clone()),
            RuntimeEventKind::RunCompleted,
            Some(result.task.phase.clone()),
            result.active_recipe_run.as_ref().map(|r| r.current_stage),
            None,
            None,
            format!("run completed at phase {:?}", result.task.phase),
        );
        // suppress unused-variable warning from packet (gathered for side effects / future use)
        let _ = packet;
        Ok(result)
    }

    pub fn start_task(
        &self,
        now: OffsetDateTime,
        task_id: impl Into<String>,
        artifact_id: impl Into<String>,
        workspace_root: impl Into<String>,
        request: impl Into<String>,
    ) -> RuntimeResult<(TaskRun, UnderstandingArtifact)> {
        tracing::debug!("start_task");
        let task_id = TaskId(task_id.into());
        let artifact_id = ArtifactId(artifact_id.into());
        let request = request.into();

        let task = TaskRun {
            id: task_id.clone(),
            workspace_root: workspace_root.into(),
            phase: TaskPhase::Observe,
            current_artifact: artifact_id.clone(),
            active_recipe_run: None,
            frontier_cases: vec![],
            created_at: now,
            updated_at: now,
        };

        let artifact = UnderstandingArtifact {
            id: artifact_id,
            task_id: task_id.clone(),
            original_request: request.clone(),
            interpreted_goal: request,
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
            created_at: now,
            updated_at: now,
        };

        self.store.put_task_run(&task)?;
        self.store.put_understanding_artifact(&artifact)?;

        Ok((task, artifact))
    }

    pub fn revise_artifact(
        &self,
        artifact_id: &str,
        now: OffsetDateTime,
        update: ArtifactUpdate,
    ) -> RuntimeResult<Option<UnderstandingArtifact>> {
        let Some(mut artifact) = self.store.get_understanding_artifact(artifact_id)? else {
            return Ok(None);
        };

        if let Some(interpreted_goal) = update.interpreted_goal {
            artifact.interpreted_goal = interpreted_goal;
        }
        if let Some(success_criteria) = update.success_criteria {
            artifact.success_criteria = success_criteria;
        }
        if let Some(constraints) = update.constraints {
            artifact.constraints = constraints;
        }
        if let Some(target_scope) = update.target_scope {
            artifact.target_scope = target_scope;
        }
        if let Some(evidence) = update.evidence {
            artifact.evidence = evidence;
        }
        if let Some(candidate_files) = update.candidate_files {
            artifact.candidate_files = candidate_files;
        }
        if let Some(assumptions) = update.assumptions {
            artifact.assumptions = assumptions;
        }
        if let Some(ambiguities) = update.ambiguities {
            artifact.ambiguities = ambiguities;
        }
        if let Some(selected_recipe) = update.selected_recipe {
            artifact.selected_recipe = Some(selected_recipe);
        }
        if let Some(risks) = update.risks {
            artifact.risks = risks;
        }
        if let Some(confidence) = update.confidence {
            artifact.confidence = confidence;
        }
        if let Some(approval) = update.approval {
            artifact.approval = approval;
        }

        artifact.revision += 1;
        artifact.updated_at = now;
        self.store.put_understanding_artifact(&artifact)?;

        Ok(Some(artifact))
    }

    pub fn clarify_task<P>(
        &self,
        artifact_id: &str,
        now: OffsetDateTime,
        provider: &P,
    ) -> RuntimeResult<Option<UnderstandingArtifact>>
    where
        P: ToolProvider,
    {
        tracing::debug!("clarify_task artifact={artifact_id}");
        let Some(mut artifact) = self.store.get_understanding_artifact(artifact_id)? else {
            return Ok(None);
        };

        let agent_context = load_agent_context(&self.store, &artifact.task_id.0);
        let output = ClarifyNode::new(provider)
            .with_agent_context(agent_context)
            .run(&artifact)?;
        output.apply_to(&mut artifact);

        // Auto-resolve Lookup-kind ambiguities before surfacing anything to the user.
        let resolved = auto_resolve_lookup_ambiguities(&self.knowledge, &mut artifact);
        if resolved > 0 {
            tracing::debug!(
                "clarify_task: resolved {resolved} lookup ambiguity/ambiguities, re-running clarify"
            );
            let agent_context2 = load_agent_context(&self.store, &artifact.task_id.0);
            let output2 = ClarifyNode::new(provider)
                .with_agent_context(agent_context2)
                .run(&artifact)?;
            // Preserve resolutions from the first pass — merge only new ambiguities.
            let prev_ambiguities = std::mem::take(&mut artifact.ambiguities);
            output2.apply_to(&mut artifact);
            for prev in prev_ambiguities {
                if prev.status == AmbiguityStatus::Resolved {
                    // Keep resolved entries so the store has an audit trail.
                    artifact.ambiguities.retain(|a| a.id != prev.id);
                    artifact.ambiguities.push(prev);
                }
            }
        }

        artifact.revision += 1;
        artifact.updated_at = now;
        self.store.put_understanding_artifact(&artifact)?;

        if let Some(mut task) = self.store.get_task_run(&artifact.task_id.0)? {
            task.phase = if has_open_user_decision_ambiguity(&artifact) {
                TaskPhase::Clarify
            } else {
                TaskPhase::Understand
            };
            task.updated_at = now;
            self.store.put_task_run(&task)?;
        }

        Ok(Some(artifact))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_frontier_case(
        &self,
        now: OffsetDateTime,
        frontier_case_id: impl Into<String>,
        task: &TaskRun,
        artifact: &UnderstandingArtifact,
        reason: FrontierReason,
        summary: impl Into<String>,
        uncertainty_events: Vec<UncertaintyEvent>,
    ) -> RuntimeResult<FrontierCase> {
        tracing::debug!("record_frontier_case task={}", task.id.0);
        let mut updated_task = task.clone();
        updated_task.phase = TaskPhase::Agree;
        updated_task.updated_at = now;

        let frontier_case = FrontierCase {
            id: FrontierCaseId(frontier_case_id.into()),
            task_id: updated_task.id.clone(),
            artifact_id: artifact.id.clone(),
            recipe_run_id: updated_task.active_recipe_run.clone(),
            reason,
            status: FrontierStatus::Open,
            summary: summary.into(),
            uncertainty_events,
            created_at: now,
            updated_at: now,
        };

        updated_task.frontier_cases.push(frontier_case.id.clone());

        self.store.put_frontier_case(&frontier_case)?;
        self.store.put_task_run(&updated_task)?;
        self.emit_runtime_event(
            now,
            updated_task.id.0.clone(),
            Some(artifact.id.0.clone()),
            updated_task
                .active_recipe_run
                .as_ref()
                .map(|id| id.0.clone()),
            RuntimeEventKind::FrontierRecorded,
            Some(TaskPhase::Agree),
            None,
            None,
            Some(frontier_case.reason.clone()),
            frontier_case.summary.clone(),
        );

        Ok(frontier_case)
    }

    pub fn understand_task(
        &self,
        artifact_id: &str,
        now: OffsetDateTime,
    ) -> RuntimeResult<Option<UnderstandingArtifact>> {
        tracing::debug!("understand_task artifact={artifact_id}");
        let Some(mut artifact) = self.store.get_understanding_artifact(artifact_id)? else {
            return Ok(None);
        };
        let Some(mut task) = self.store.get_task_run(&artifact.task_id.0)? else {
            return Ok(Some(artifact));
        };

        let scoped_evidence =
            gather_workspace_evidence(&task.workspace_root, &artifact.target_scope)?;
        artifact.evidence = merge_evidence_refs(artifact.evidence.clone(), scoped_evidence);
        if !artifact.evidence.is_empty() {
            artifact.confidence = artifact.confidence.max(0.65);
        }
        artifact.revision += 1;
        artifact.updated_at = now;

        task.phase = if has_open_ambiguity(&artifact) {
            TaskPhase::Understand
        } else {
            TaskPhase::Localize
        };
        task.updated_at = now;

        self.store.put_understanding_artifact(&artifact)?;
        self.store.put_task_run(&task)?;

        Ok(Some(artifact))
    }

    pub fn localize_task(
        &self,
        artifact_id: &str,
        now: OffsetDateTime,
    ) -> RuntimeResult<Option<UnderstandingArtifact>> {
        Ok(self
            .localize_task_with_packet(artifact_id, now, None)?
            .map(|(artifact, _)| artifact))
    }

    pub fn understand_task_with_provider<P>(
        &self,
        artifact_id: &str,
        now: OffsetDateTime,
        provider: &P,
    ) -> RuntimeResult<Option<UnderstandingArtifact>>
    where
        P: ToolProvider,
    {
        let Some(mut artifact) = self.understand_task(artifact_id, now)? else {
            return Ok(None);
        };

        let agent_context = load_agent_context(&self.store, &artifact.task_id.0);
        let output = UnderstandNode::new(provider)
            .with_agent_context(agent_context)
            .run(&artifact)?;
        output.apply_to(&mut artifact);
        artifact.revision += 1;
        artifact.updated_at = now;
        self.store.put_understanding_artifact(&artifact)?;

        if let Some(mut task) = self.store.get_task_run(&artifact.task_id.0)? {
            task.phase = if has_open_ambiguity(&artifact) {
                TaskPhase::Understand
            } else {
                TaskPhase::Localize
            };
            task.updated_at = now;
            self.store.put_task_run(&task)?;
        }

        Ok(Some(artifact))
    }

    pub fn resolve_ambiguity(
        &self,
        artifact_id: &str,
        ambiguity_id: &str,
        resolution: impl Into<String>,
        now: OffsetDateTime,
    ) -> RuntimeResult<Option<UnderstandingArtifact>> {
        let Some(mut artifact) = self.store.get_understanding_artifact(artifact_id)? else {
            return Ok(None);
        };

        let resolution = resolution.into();
        let ambiguity = artifact
            .ambiguities
            .iter_mut()
            .find(|ambiguity| ambiguity.id == ambiguity_id)
            .ok_or_else(|| RuntimeError::AmbiguityNotFound(ambiguity_id.into()))?;

        ambiguity.status = AmbiguityStatus::Resolved;
        ambiguity.resolution = Some(resolution);
        artifact.revision += 1;
        artifact.updated_at = now;
        self.store.put_understanding_artifact(&artifact)?;

        Ok(Some(artifact))
    }

    pub fn approve_artifact(
        &self,
        artifact_id: &str,
        decided_by: impl Into<String>,
        note: Option<String>,
        now: OffsetDateTime,
    ) -> RuntimeResult<Option<UnderstandingArtifact>> {
        tracing::debug!("approve_artifact artifact={artifact_id}");
        let Some(mut artifact) = self.store.get_understanding_artifact(artifact_id)? else {
            return Ok(None);
        };

        artifact.approval.status = ApprovalStatus::Approved;
        artifact.approval.decided_by = Some(decided_by.into());
        artifact.approval.decided_at = Some(now);
        artifact.approval.note = note;
        artifact.revision += 1;
        artifact.updated_at = now;
        self.store.put_understanding_artifact(&artifact)?;

        if let Some(mut task) = self.store.get_task_run(&artifact.task_id.0)? {
            task.phase = TaskPhase::Execute;
            task.updated_at = now;
            self.store.put_task_run(&task)?;
        }

        Ok(Some(artifact))
    }

    pub fn start_execution(
        &self,
        task_id: &str,
        now: OffsetDateTime,
        recipe: RecipeRef,
    ) -> RuntimeResult<RecipeRun> {
        tracing::debug!("start_execution task={task_id}");
        let mut task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let artifact = self
            .store
            .get_understanding_artifact(&task.current_artifact.0)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;

        if artifact.approval.status != ApprovalStatus::Approved {
            return Err(RuntimeError::ArtifactNotApproved);
        }
        if !is_execution_ready(&artifact) {
            return Err(RuntimeError::UnlocalizedTask);
        }

        let recipe_run = RecipeRun {
            id: RecipeRunId(format!("recipe-run-{}", now.unix_timestamp_nanos())),
            task_id: task.id.clone(),
            recipe,
            current_stage: ExecutionStageKind::Inspect,
            stages: vec![
                running_stage(ExecutionStageKind::Inspect, now),
                pending_stage(ExecutionStageKind::Propose),
                pending_stage(ExecutionStageKind::Verify),
                pending_stage(ExecutionStageKind::Apply),
                pending_stage(ExecutionStageKind::Setup),
                pending_stage(ExecutionStageKind::Validate),
            ],
            change_set: None,
            setup_outcomes: vec![],
            proposed_changes: vec![],
            setup_commands: vec![],
            created_at: now,
            updated_at: now,
        };

        task.active_recipe_run = Some(recipe_run.id.clone());
        task.phase = TaskPhase::Execute;
        task.updated_at = now;

        self.store.put_recipe_run(&recipe_run)?;
        self.store.put_task_run(&task)?;

        Ok(recipe_run)
    }

    pub fn execute_inspect(&self, task_id: &str, now: OffsetDateTime) -> RuntimeResult<RecipeRun> {
        tracing::debug!("execute_inspect task={task_id}");
        let task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let artifact = self
            .store
            .get_understanding_artifact(&task.current_artifact.0)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let mut recipe_run = self.load_active_recipe_run(&task)?;

        let summary = format!(
            "inspected {} scoped paths with {} evidence refs and {} open ambiguities",
            artifact.target_scope.len(),
            artifact.evidence.len(),
            artifact
                .ambiguities
                .iter()
                .filter(|ambiguity| ambiguity.status == AmbiguityStatus::Open)
                .count()
        );
        complete_stage(
            &mut recipe_run,
            ExecutionStageKind::Inspect,
            ExecutionStageKind::Propose,
            summary,
            now,
        );
        self.store.put_recipe_run(&recipe_run)?;

        Ok(recipe_run)
    }

    pub fn execute_propose(&self, task_id: &str, now: OffsetDateTime) -> RuntimeResult<RecipeRun> {
        tracing::debug!("execute_propose task={task_id}");
        let task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let artifact = self
            .store
            .get_understanding_artifact(&task.current_artifact.0)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let mut recipe_run = self.load_active_recipe_run(&task)?;

        let summary = format!(
            "proposed lean integration across [{}] for goal: {}",
            artifact.target_scope.join(", "),
            artifact.interpreted_goal
        );
        complete_stage(
            &mut recipe_run,
            ExecutionStageKind::Propose,
            ExecutionStageKind::Verify,
            summary,
            now,
        );
        self.store.put_recipe_run(&recipe_run)?;

        Ok(recipe_run)
    }

    /// Inject a `ChangeSet` directly (used in tests and manual override flows).
    pub fn set_change_set(
        &self,
        task_id: &str,
        now: OffsetDateTime,
        change_set: ChangeSet,
    ) -> RuntimeResult<RecipeRun> {
        tracing::debug!(
            "set_change_set task={} ops={}",
            task_id,
            change_set.ops.len()
        );
        let task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let mut recipe_run = self.load_active_recipe_run(&task)?;

        recipe_run.change_set = Some(change_set);
        recipe_run.updated_at = now;
        self.store.put_recipe_run(&recipe_run)?;

        Ok(recipe_run)
    }

    pub fn generate_proposed_change<P>(
        &self,
        task_id: &str,
        now: OffsetDateTime,
        provider: &P,
        prior_context_files: &[String],
        extra_ignore_patterns: &[String],
    ) -> RuntimeResult<RecipeRun>
    where
        P: ToolProvider,
    {
        tracing::debug!("generate_proposed_change task={task_id}");
        let mut task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let artifact = self
            .store
            .get_understanding_artifact(&task.current_artifact.0)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;

        // Bootstrap a recipe run if one doesn't exist yet — the agent path skips
        // start_execution (which guards on is_execution_ready / candidate_files).
        // The agent discovers files itself, so no pre-localized candidates are needed.
        let mut recipe_run = match self.load_active_recipe_run(&task) {
            Ok(run) => run,
            Err(RuntimeError::ActiveRecipeRunMissing) => {
                let run = RecipeRun {
                    id: RecipeRunId(format!("recipe-run-{}", now.unix_timestamp_nanos())),
                    task_id: task.id.clone(),
                    recipe: RecipeRef {
                        id: "agent".into(),
                        version: "v1".into(),
                    },
                    current_stage: ExecutionStageKind::Apply,
                    stages: vec![
                        running_stage(ExecutionStageKind::Apply, now),
                        pending_stage(ExecutionStageKind::Setup),
                    ],
                    change_set: None,
                    setup_outcomes: vec![],
                    proposed_changes: vec![],
                    setup_commands: vec![],
                    created_at: now,
                    updated_at: now,
                };
                task.active_recipe_run = Some(run.id.clone());
                task.phase = TaskPhase::Execute;
                task.updated_at = now;
                self.store.put_recipe_run(&run)?;
                self.store.put_task_run(&task)?;
                run
            }
            Err(e) => return Err(e),
        };
        let candidates = self.gather_change_candidates(&task.workspace_root, &artifact)?;
        tracing::debug!(
            "change candidates: {}",
            candidates
                .iter()
                .map(|c| c.path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Merge prior-session files (files written in the previous task) into candidates
        // so follow-up prompts have immediate context without extra read_file turns.
        let mut all_candidates = candidates;
        for path in prior_context_files {
            if !all_candidates.iter().any(|c| c.path == *path) {
                let abs = std::path::Path::new(&task.workspace_root).join(path);
                if let Ok(contents) = std::fs::read_to_string(&abs) {
                    all_candidates.push(shunt_infer::SourceFileContext {
                        path: path.clone(),
                        contents,
                    });
                }
            }
        }

        // Derive session budget from provider capabilities, then apply any project overrides.
        let session_budget = {
            let mut b = provider.capabilities().to_session_budget();
            if let Some(o) = &self.budget_override {
                b.apply_override(o);
            }
            b
        };

        // Run the agent session — it decides what to read, edit, search, and do.
        // Candidates from the localizer are pre-loaded so the agent has warm context
        // without needing to issue read_file calls for the most likely files.
        let mut session = AgentSession::new(provider, &task.workspace_root)
            .with_budget(session_budget.clone())
            .with_wall_timeout(PROPOSE_SESSION_WALL_TIMEOUT)
            .with_pre_loaded(&all_candidates)
            .with_ignore_patterns(extra_ignore_patterns.iter().cloned());
        if let Some(obs) = self.agent_observer.clone() {
            session = session.with_observer(obs);
        }
        let change_set = {
            use shunt_core::{ChangeSet, FileOp};

            fn build_change_set(
                ops: Vec<shunt_infer::ProposedFileOp>,
                setup_commands: Vec<shunt_infer::ProposedCommand>,
            ) -> ChangeSet {
                let ops: Vec<FileOp> = ops
                    .into_iter()
                    .map(|op| match op {
                        shunt_infer::ProposedFileOp::Edit {
                            path,
                            search,
                            replacement,
                        } => FileOp::Edit {
                            path,
                            search,
                            replacement,
                        },
                        shunt_infer::ProposedFileOp::Create { path, contents } => {
                            FileOp::Create { path, contents }
                        }
                        shunt_infer::ProposedFileOp::Delete { path } => FileOp::Delete { path },
                    })
                    .collect();
                let commands = setup_commands
                    .into_iter()
                    .map(|c| shunt_core::CommandSpec {
                        program: c.program,
                        args: c.args,
                    })
                    .collect();
                ChangeSet { ops, commands }
            }

            // Baseline check: capture pre-existing errors before the agent runs.
            // The fix loop only fires when the agent INTRODUCES new errors, not for
            // pre-existing issues (e.g. deprecated tsconfig options).
            let baseline_errors = workspace_check(&task.workspace_root);

            let execute_request = execution_request(&artifact);
            let first_result = session.run(&execute_request);

            match first_result {
                AgentResult::Done {
                    ops,
                    setup_commands,
                    file_state: done_file_state,
                    ..
                } => {
                    let cs = build_change_set(ops, setup_commands);
                    let diagnostics = collect_repair_diagnostics(
                        provider,
                        &task.workspace_root,
                        &artifact,
                        &cs,
                        &done_file_state,
                        baseline_errors.as_ref(),
                        extra_ignore_patterns,
                        self.budget_override.as_ref(),
                        self.agent_observer.clone(),
                    )?;

                    if !diagnostics.is_empty() {
                        tracing::info!("fix loop triggered");
                        let fix_pre_loaded: Vec<shunt_infer::SourceFileContext> = done_file_state
                            .into_iter()
                            .map(|(path, contents)| shunt_infer::SourceFileContext {
                                path,
                                contents,
                            })
                            .collect();
                        let fix_request = build_repair_request(&execute_request, &diagnostics);
                        let mut fix_session = AgentSession::new(provider, &task.workspace_root)
                            .with_budget(session_budget.clone())
                            .with_wall_timeout(REPAIR_SESSION_WALL_TIMEOUT)
                            .with_pre_loaded(&fix_pre_loaded)
                            .with_ignore_patterns(extra_ignore_patterns.iter().cloned());
                        if let Some(obs) = self.agent_observer.clone() {
                            fix_session = fix_session.with_observer(obs);
                        }
                        match fix_session.run(&fix_request) {
                            AgentResult::Done {
                                ops: fix_ops,
                                setup_commands: fix_cmds,
                                ..
                            } => {
                                let fix_cs = build_change_set(fix_ops, fix_cmds);
                                let mut merged_ops = cs.ops;
                                merged_ops.extend(fix_cs.ops);
                                let mut merged_cmds = cs.commands;
                                merged_cmds.extend(fix_cs.commands);
                                ChangeSet {
                                    ops: merged_ops,
                                    commands: merged_cmds,
                                }
                            }
                            _ => cs,
                        }
                    } else {
                        cs
                    }
                }
                AgentResult::NeedsClarification {
                    question, context, ..
                } => {
                    return Err(RuntimeError::AgentNeedsClarification { question, context });
                }
                AgentResult::MaxTurnsReached => {
                    return Err(RuntimeError::Other(
                        "agent hit max turns without completing".into(),
                    ));
                }
            }
        };

        tracing::debug!(
            "generated change_set ops={} commands={}",
            change_set.ops.len(),
            change_set.commands.len()
        );

        if let Some(error) = validate_work_contract(&task.workspace_root, &artifact, &change_set) {
            return Err(RuntimeError::Other(error));
        }

        if let Some(stage) = recipe_run
            .stages
            .iter_mut()
            .find(|stage| stage.kind == ExecutionStageKind::Apply)
        {
            stage.summary = format!("generated {} ops", change_set.ops.len());
            stage.started_at = Some(stage.started_at.unwrap_or(now));
        }
        recipe_run.change_set = Some(change_set);
        recipe_run.updated_at = now;
        self.store.put_recipe_run(&recipe_run)?;

        Ok(recipe_run)
    }

    /// Run the setup commands stored in the active recipe run and persist outcomes.
    pub fn execute_setup(&self, task_id: &str, now: OffsetDateTime) -> RuntimeResult<RecipeRun> {
        tracing::debug!("execute_setup task={task_id}");
        let task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let mut recipe_run = self.load_active_recipe_run(&task)?;

        let commands: &[shunt_core::CommandSpec] = recipe_run
            .change_set
            .as_ref()
            .map(|cs| cs.commands.as_slice())
            .unwrap_or_default();

        let outcomes = crate::executor::run_commands(
            &task.workspace_root,
            commands,
            |spec| tracing::debug!("start: {}", spec.display()),
            |outcome| {
                tracing::debug!(
                    "done: {} exit={}",
                    outcome.spec.display(),
                    outcome.exit_code
                )
            },
        );

        if let Some(stage) = recipe_run
            .stages
            .iter_mut()
            .find(|s| s.kind == ExecutionStageKind::Setup)
        {
            let failed = outcomes.iter().filter(|o| !o.success).count();
            stage.status = if failed == 0 {
                StageStatus::Passed
            } else {
                StageStatus::Failed
            };
            stage.summary = format!(
                "{}/{} commands succeeded",
                outcomes.len() - failed,
                outcomes.len()
            );
            stage.started_at = Some(stage.started_at.unwrap_or(now));
            stage.finished_at = Some(now);
        }
        recipe_run.setup_outcomes = outcomes;
        recipe_run.updated_at = now;
        self.store.put_recipe_run(&recipe_run)?;

        Ok(recipe_run)
    }

    pub fn execute_verify(&self, task_id: &str, now: OffsetDateTime) -> RuntimeResult<RecipeRun> {
        tracing::debug!("execute_verify task={task_id}");
        let task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let artifact = self
            .store
            .get_understanding_artifact(&task.current_artifact.0)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let mut recipe_run = self.load_active_recipe_run(&task)?;

        let verifiers = vec![
            approval_verifier(&artifact),
            evidence_verifier(&artifact),
            ambiguity_verifier(&artifact),
            workspace_test_verifier(&task.workspace_root)?,
        ];
        let failed = verifiers
            .iter()
            .any(|verifier| verifier.status == VerifierStatus::Failed);
        tracing::debug!("verify summary: {}", verifier_summary(&verifiers));

        if let Some(stage) = recipe_run
            .stages
            .iter_mut()
            .find(|stage| stage.kind == ExecutionStageKind::Verify)
        {
            stage.status = if failed {
                StageStatus::Failed
            } else {
                StageStatus::Passed
            };
            stage.summary = verifier_summary(&verifiers);
            stage.verifiers = verifiers;
            stage.started_at = Some(stage.started_at.unwrap_or(now));
            stage.finished_at = Some(now);
        }

        if failed {
            recipe_run.current_stage = ExecutionStageKind::Verify;
        } else {
            if let Some(stage) = recipe_run
                .stages
                .iter_mut()
                .find(|stage| stage.kind == ExecutionStageKind::Apply)
                && stage.status == StageStatus::Pending
            {
                stage.status = StageStatus::Running;
                stage.started_at = Some(now);
            }
            recipe_run.current_stage = ExecutionStageKind::Apply;
        }

        recipe_run.updated_at = now;
        self.store.put_recipe_run(&recipe_run)?;

        if failed {
            self.record_execution_frontier(
                now,
                &task,
                &artifact,
                &recipe_run,
                ExecutionStageKind::Verify,
                FrontierReason::VerifierFailure,
                format!(
                    "verify blocked: {}",
                    verifier_summary_from_stage(&recipe_run, ExecutionStageKind::Verify)
                ),
                uncertainty_events_from_verifiers(
                    &task,
                    ExecutionStageKind::Verify,
                    stage_verifiers(&recipe_run, ExecutionStageKind::Verify),
                    now,
                ),
            )?;
        }

        Ok(recipe_run)
    }

    pub fn execute_apply(&self, task_id: &str, now: OffsetDateTime) -> RuntimeResult<RecipeRun> {
        tracing::debug!("execute_apply task={task_id}");
        let task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let mut recipe_run = self.load_active_recipe_run(&task)?;

        let Some(change_set) = &recipe_run.change_set else {
            if let Some(stage) = recipe_run
                .stages
                .iter_mut()
                .find(|stage| stage.kind == ExecutionStageKind::Apply)
            {
                stage.status = StageStatus::Failed;
                stage.summary = "no change set available to apply".into();
                stage.started_at = Some(stage.started_at.unwrap_or(now));
                stage.finished_at = Some(now);
            }
            recipe_run.updated_at = now;
            self.store.put_recipe_run(&recipe_run)?;
            return Err(RuntimeError::NoChangeSet);
        };

        let workspace = Path::new(&task.workspace_root);
        apply_change_set_transactional(workspace, change_set)?;

        let applied_count = change_set.ops.len();
        complete_stage(
            &mut recipe_run,
            ExecutionStageKind::Apply,
            ExecutionStageKind::Validate,
            format!("applied {} proposed changes", applied_count),
            now,
        );
        self.store.put_recipe_run(&recipe_run)?;

        Ok(recipe_run)
    }

    pub fn execute_validate(&self, task_id: &str, now: OffsetDateTime) -> RuntimeResult<RecipeRun> {
        tracing::debug!("execute_validate task={task_id}");
        let mut task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let mut recipe_run = self.load_active_recipe_run(&task)?;
        ensure_current_stage(&recipe_run, ExecutionStageKind::Validate)?;
        ensure_stage_passed(&recipe_run, ExecutionStageKind::Apply)?;

        let verifier = workspace_test_verifier(&task.workspace_root)?;
        let passed = verifier.status == VerifierStatus::Passed;
        tracing::debug!(
            "validate verifier: {}={:?}",
            verifier.verifier,
            verifier.status
        );

        if let Some(stage) = recipe_run
            .stages
            .iter_mut()
            .find(|stage| stage.kind == ExecutionStageKind::Validate)
        {
            stage.status = if passed {
                StageStatus::Passed
            } else {
                StageStatus::Failed
            };
            stage.summary = format!("{}={:?}", verifier.verifier, verifier.status);
            stage.verifiers = vec![verifier];
            stage.started_at = Some(stage.started_at.unwrap_or(now));
            stage.finished_at = Some(now);
        }

        recipe_run.current_stage = ExecutionStageKind::Validate;
        recipe_run.updated_at = now;
        self.store.put_recipe_run(&recipe_run)?;

        if passed {
            task.phase = TaskPhase::Complete;
            task.updated_at = now;
            self.store.put_task_run(&task)?;
        } else {
            let artifact = self
                .store
                .get_understanding_artifact(&task.current_artifact.0)?
                .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
            self.record_execution_frontier(
                now,
                &task,
                &artifact,
                &recipe_run,
                ExecutionStageKind::Validate,
                FrontierReason::VerifierFailure,
                format!(
                    "validate blocked: {}",
                    verifier_summary_from_stage(&recipe_run, ExecutionStageKind::Validate)
                ),
                uncertainty_events_from_verifiers(
                    &task,
                    ExecutionStageKind::Validate,
                    stage_verifiers(&recipe_run, ExecutionStageKind::Validate),
                    now,
                ),
            )?;
        }

        Ok(recipe_run)
    }

    pub fn create_patch_correction(
        &self,
        frontier_case_id: &str,
        correction_id: impl Into<String>,
        summary: impl Into<String>,
        change_set: ChangeSet,
        now: OffsetDateTime,
    ) -> RuntimeResult<(CorrectionPackage, FrontierCase, RecipeRun)> {
        tracing::debug!("create_patch_correction frontier_case={frontier_case_id}");
        let mut frontier = self
            .store
            .get_frontier_case(frontier_case_id)?
            .ok_or_else(|| RuntimeError::FrontierCaseNotFound(frontier_case_id.into()))?;
        let recipe_run_id = frontier
            .recipe_run_id
            .clone()
            .ok_or(RuntimeError::FrontierRecipeRunMissing)?;
        let mut recipe_run = self
            .store
            .get_recipe_run(&recipe_run_id.0)?
            .ok_or(RuntimeError::ActiveRecipeRunMissing)?;

        if recipe_run.current_stage != ExecutionStageKind::Validate
            && recipe_run.current_stage != ExecutionStageKind::Apply
        {
            return Err(RuntimeError::PatchCorrectionUnsupported);
        }

        recipe_run.change_set = Some(change_set);
        prepare_recipe_run_for_patch_replay(&mut recipe_run, now);

        let correction = CorrectionPackage {
            id: shunt_core::CorrectionPackageId(correction_id.into()),
            frontier_case_id: frontier.id.clone(),
            kind: CorrectionKind::PatchRevision,
            summary: summary.into(),
            validated: false,
            created_at: now,
        };

        frontier.status = FrontierStatus::InReview;
        frontier.updated_at = now;

        self.store.put_recipe_run(&recipe_run)?;
        self.store.put_frontier_case(&frontier)?;
        self.store.put_correction_package(&correction)?;

        Ok((correction, frontier, recipe_run))
    }

    pub fn replay_correction(
        &self,
        correction_id: &str,
        now: OffsetDateTime,
    ) -> RuntimeResult<(CorrectionPackage, FrontierCase, RecipeRun)> {
        tracing::debug!("replay_correction correction={correction_id}");
        let mut correction = self
            .store
            .get_correction_package(correction_id)?
            .ok_or_else(|| RuntimeError::CorrectionPackageNotFound(correction_id.into()))?;
        let mut frontier = self
            .store
            .get_frontier_case(&correction.frontier_case_id.0)?
            .ok_or_else(|| {
                RuntimeError::FrontierCaseNotFound(correction.frontier_case_id.0.clone())
            })?;

        let task_id = frontier.task_id.0.clone();
        let recipe_run = match correction.kind {
            CorrectionKind::PatchRevision => {
                let recipe_run = self.execute_apply(&task_id, now)?;
                if recipe_run.current_stage == ExecutionStageKind::Validate {
                    self.execute_validate(&task_id, now)?
                } else {
                    recipe_run
                }
            }
            _ => return Err(RuntimeError::PatchCorrectionUnsupported),
        };

        let validated = recipe_run
            .stages
            .iter()
            .find(|stage| stage.kind == ExecutionStageKind::Validate)
            .map(|stage| stage.status == StageStatus::Passed)
            .unwrap_or(false);

        correction.validated = validated;
        frontier.status = if validated {
            FrontierStatus::Corrected
        } else {
            FrontierStatus::InReview
        };
        frontier.updated_at = now;

        self.store.put_correction_package(&correction)?;
        self.store.put_frontier_case(&frontier)?;

        Ok((correction, frontier, recipe_run))
    }

    fn load_active_recipe_run(&self, task: &TaskRun) -> RuntimeResult<RecipeRun> {
        let recipe_run_id = task
            .active_recipe_run
            .as_ref()
            .ok_or(RuntimeError::ActiveRecipeRunMissing)?;

        self.store
            .get_recipe_run(&recipe_run_id.0)?
            .ok_or(RuntimeError::ActiveRecipeRunMissing)
    }

    pub fn load_handle_result(&self, task_id: &str) -> RuntimeResult<HandleResult> {
        let task = self
            .store
            .get_task_run(task_id)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let artifact = self
            .store
            .get_understanding_artifact(&task.current_artifact.0)?
            .ok_or_else(|| RuntimeError::TaskNotFound(task_id.into()))?;
        let active_recipe_run = match &task.active_recipe_run {
            Some(recipe_run_id) => self.store.get_recipe_run(&recipe_run_id.0)?,
            None => None,
        };
        let frontier_cases = self.store.list_frontier_cases_for_task(task_id)?;

        Ok(HandleResult {
            task,
            artifact,
            active_recipe_run,
            frontier_cases,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn record_execution_frontier(
        &self,
        now: OffsetDateTime,
        task: &TaskRun,
        artifact: &UnderstandingArtifact,
        recipe_run: &RecipeRun,
        stage: ExecutionStageKind,
        reason: FrontierReason,
        summary: String,
        uncertainty_events: Vec<UncertaintyEvent>,
    ) -> RuntimeResult<FrontierCase> {
        let mut updated_task = task.clone();
        updated_task.updated_at = now;

        let frontier_case = FrontierCase {
            id: FrontierCaseId(format!(
                "frontier-{}-{}-{:?}-{}",
                task.id.0,
                recipe_run.id.0,
                stage,
                now.unix_timestamp_nanos()
            )),
            task_id: task.id.clone(),
            artifact_id: artifact.id.clone(),
            recipe_run_id: Some(recipe_run.id.clone()),
            reason,
            status: FrontierStatus::Open,
            summary,
            uncertainty_events,
            created_at: now,
            updated_at: now,
        };

        if !updated_task.frontier_cases.contains(&frontier_case.id) {
            updated_task.frontier_cases.push(frontier_case.id.clone());
        }

        self.store.put_frontier_case(&frontier_case)?;
        self.store.put_task_run(&updated_task)?;
        self.emit_runtime_event(
            now,
            task.id.0.clone(),
            Some(artifact.id.0.clone()),
            Some(recipe_run.id.0.clone()),
            RuntimeEventKind::FrontierRecorded,
            Some(TaskPhase::Execute),
            Some(stage),
            None,
            Some(frontier_case.reason.clone()),
            frontier_case.summary.clone(),
        );

        Ok(frontier_case)
    }
}

#[derive(Debug, Default)]
pub struct ArtifactUpdate {
    pub interpreted_goal: Option<String>,
    pub success_criteria: Option<Vec<String>>,
    pub constraints: Option<Vec<String>>,
    pub target_scope: Option<Vec<String>>,
    pub evidence: Option<Vec<shunt_core::EvidenceRef>>,
    pub candidate_files: Option<Vec<CandidateFile>>,
    pub assumptions: Option<Vec<shunt_core::Assumption>>,
    pub ambiguities: Option<Vec<shunt_core::Ambiguity>>,
    pub selected_recipe: Option<shunt_core::RecipeRef>,
    pub risks: Option<Vec<shunt_core::Risk>>,
    pub confidence: Option<f32>,
    pub approval: Option<ApprovalState>,
}

/// Load the current ledger for `task_id` and format it as a compact context
/// string to prepend to model prompts. Returns an empty string when there are
/// no entries yet (new sessions, first call).
fn load_agent_context(store: &SqliteStore, task_id: &str) -> String {
    let entries = match store.list_ledger_entries(task_id) {
        Ok(e) => e,
        Err(_) => return String::new(),
    };
    if entries.is_empty() {
        return String::new();
    }
    let mut ledger = WorkLedger::new(task_id.to_string(), GoalSnapshot::default());
    ledger.entries = entries;
    AgentFrame::from_ledger(&ledger, 10, AgentBudget::default()).format_context()
}

fn execution_request(artifact: &UnderstandingArtifact) -> String {
    let contract = contract_prompt_section(artifact);
    if contract.is_empty() {
        artifact.original_request.clone()
    } else {
        format!("{}\n\n{}", artifact.original_request, contract)
    }
}

fn contract_prompt_section(artifact: &UnderstandingArtifact) -> String {
    let contract = &artifact.work_contract;
    if contract.required_paths.is_empty() && contract.behavioral_checks.is_empty() {
        return String::new();
    }

    let mut section = String::from("UNDERSTOOD WORK CONTRACT:\n");
    if !contract.required_paths.is_empty() {
        section.push_str("Required path facts:\n");
        for path in &contract.required_paths {
            section.push_str(&format!(
                "- {} [{:?}]: {}\n",
                path.path, path.intent, path.reason
            ));
        }
    }
    if !contract.behavioral_checks.is_empty() {
        section.push_str("Behavioral checks:\n");
        for check in &contract.behavioral_checks {
            section.push_str(&format!("- {check}\n"));
        }
    }
    section.push_str("Do not call done until this contract is satisfied.");
    section
}

fn validate_work_contract(
    workspace_root: &str,
    artifact: &UnderstandingArtifact,
    change_set: &ChangeSet,
) -> Option<String> {
    let mut failures = Vec::new();
    for required in &artifact.work_contract.required_paths {
        let Some(path) = workspace_relative_path(workspace_root, &required.path) else {
            failures.push(format!(
                "required path '{}' is outside the workspace ({})",
                required.path, required.reason
            ));
            continue;
        };
        let projected = projected_path_state(workspace_root, &path, change_set);
        match required.intent {
            RequiredPathIntent::Exist if !projected.exists => failures.push(format!(
                "required path '{}' does not exist after proposed changes ({})",
                required.path, required.reason
            )),
            RequiredPathIntent::CreateOrUpdate if !projected.exists || !projected.touched => {
                failures.push(format!(
                    "required path '{}' was not created or updated ({})",
                    required.path, required.reason
                ));
            }
            RequiredPathIntent::Remove if projected.exists => failures.push(format!(
                "required path '{}' still exists after proposed changes ({})",
                required.path, required.reason
            )),
            _ => {}
        }
    }

    if failures.is_empty() {
        None
    } else {
        Some(format!(
            "work contract unsatisfied:\n{}",
            failures
                .into_iter()
                .map(|failure| format!("- {failure}"))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }
}

#[derive(Debug, Clone, Copy)]
struct ProjectedPathState {
    exists: bool,
    touched: bool,
}

fn projected_path_state(
    workspace_root: &str,
    relative_path: &str,
    change_set: &ChangeSet,
) -> ProjectedPathState {
    let root = Path::new(workspace_root);
    let mut state = ProjectedPathState {
        exists: root.join(relative_path).exists(),
        touched: false,
    };

    for op in &change_set.ops {
        match op {
            FileOp::Create { path, .. } | FileOp::Edit { path, .. } => {
                if workspace_relative_path(workspace_root, path).as_deref() == Some(relative_path) {
                    state.exists = true;
                    state.touched = true;
                }
            }
            FileOp::Delete { path } => {
                if workspace_relative_path(workspace_root, path).as_deref() == Some(relative_path) {
                    state.exists = false;
                    state.touched = true;
                }
            }
        }
    }

    state
}

fn workspace_relative_path(workspace_root: &str, path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    let supplied = Path::new(trimmed);
    let relative = if supplied.is_absolute() {
        let root = Path::new(workspace_root);
        supplied.strip_prefix(root).ok()?.to_path_buf()
    } else {
        supplied.to_path_buf()
    };
    if relative.components().any(|component| {
        matches!(
            component,
            std::path::Component::ParentDir | std::path::Component::RootDir
        )
    }) {
        return None;
    }
    Some(relative.to_string_lossy().into_owned())
}

/// Run a fast workspace sanity check after the agent writes files.
/// Returns a structured diagnostic if the check fails, `None` if it passes or no checker is found.
fn workspace_check(workspace_root: &str) -> Option<RepairDiagnostic> {
    use std::process::Command;
    let root = std::path::Path::new(workspace_root);
    if root.join("Cargo.toml").exists() {
        let command = "cargo check --message-format=short --quiet";
        let out = Command::new("cargo")
            .args(["check", "--message-format=short", "--quiet"])
            .current_dir(root)
            .output()
            .ok()?;
        if !out.status.success() {
            return Some(RepairDiagnostic {
                source: "workspace_check",
                command: Some(command.into()),
                summary: "projected workspace check failed".into(),
                output: truncate_chars(
                    &String::from_utf8_lossy(&out.stderr),
                    MAX_REPAIR_OUTPUT_CHARS,
                ),
            });
        }
    } else if root.join("package.json").exists() {
        // Use `tsc --noEmit` if tsconfig present, else skip (npm test is too slow for inline loop).
        if root.join("tsconfig.json").exists() {
            let command = "npx tsc --noEmit --pretty false";
            let out = Command::new("npx")
                .args(["tsc", "--noEmit", "--pretty", "false"])
                .current_dir(root)
                .output()
                .ok()?;
            if !out.status.success() {
                return Some(RepairDiagnostic {
                    source: "workspace_check",
                    command: Some(command.into()),
                    summary: "projected workspace check failed".into(),
                    output: truncate_chars(
                        &String::from_utf8_lossy(&out.stdout),
                        MAX_REPAIR_OUTPUT_CHARS,
                    ),
                });
            }
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn collect_repair_diagnostics<P: ToolProvider>(
    provider: &P,
    workspace_root: &str,
    artifact: &UnderstandingArtifact,
    change_set: &ChangeSet,
    done_file_state: &std::collections::HashMap<String, String>,
    baseline_error: Option<&RepairDiagnostic>,
    extra_ignore_patterns: &[String],
    budget_override: Option<&shunt_infer::SessionBudgetOverride>,
    observer: Option<Arc<dyn AgentObserver + Send + Sync>>,
) -> RuntimeResult<Vec<RepairDiagnostic>> {
    let projected = tempfile::tempdir()?;
    let projected_root = projected.path().join("workspace");
    copy_workspace_tree(Path::new(workspace_root), &projected_root)?;
    apply_change_set_transactional(&projected_root, change_set)?;

    let projected_root_str = projected_root.to_string_lossy().into_owned();
    let mut diagnostics = Vec::new();

    diagnostics.extend(structural_file_diagnostics(change_set, done_file_state));
    diagnostics.extend(explicit_literal_coverage_diagnostics(
        &artifact.original_request,
        change_set,
        done_file_state,
    ));

    let safe_setup_commands: Vec<shunt_core::CommandSpec> = change_set
        .commands
        .iter()
        .filter(|spec| {
            matches!(
                shunt_core::safety::classify(spec),
                shunt_core::safety::CommandSafety::Safe
            )
        })
        .cloned()
        .collect();
    let setup_outcomes =
        crate::executor::run_commands(&projected_root_str, &safe_setup_commands, |_| {}, |_| {});
    for outcome in setup_outcomes.iter().filter(|outcome| !outcome.success) {
        diagnostics.push(RepairDiagnostic {
            source: "setup",
            command: Some(outcome.spec.display()),
            summary: "projected setup command failed".into(),
            output: truncate_chars(&format_command_outcome(outcome), MAX_REPAIR_OUTPUT_CHARS),
        });
    }

    if diagnostics.is_empty() {
        let pre_loaded: Vec<SourceFileContext> = done_file_state
            .iter()
            .map(|(path, contents)| SourceFileContext {
                path: path.clone(),
                contents: contents.clone(),
            })
            .collect();
        let changed = done_file_state
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let verify_task = format!(
            "Verify that the following changes correctly implement the task.\n\n\
             Original task: {}\n\n{}\n\nChanged files: {changed}\n\n\
             Run the build, then run surgical smoke tests for what changed. \
             Call done with 'PASS: ...' or 'FAIL: ...'",
            artifact.original_request,
            contract_prompt_section(artifact),
        );
        let verifier_budget = {
            let mut b = shunt_infer::SessionBudget::for_verifier();
            if let Some(o) = budget_override {
                b.apply_override(o);
            }
            b
        };
        let mut vsession = AgentSession::new_verifier(provider, &projected_root_str)
            .with_budget(verifier_budget)
            .with_wall_timeout(VERIFIER_SESSION_WALL_TIMEOUT)
            .with_pre_loaded(&pre_loaded)
            .with_ignore_patterns(extra_ignore_patterns.iter().cloned());
        if let Some(obs) = observer {
            vsession = vsession.with_observer(obs);
        }
        match vsession.run(&verify_task) {
            AgentResult::Done { description, .. } => {
                if description.trim_start().to_uppercase().starts_with("PASS") {
                    tracing::info!("verifier: PASS");
                } else {
                    tracing::info!(
                        "verifier: FAIL — {}",
                        &description[..description.len().min(200)]
                    );
                    diagnostics.push(RepairDiagnostic {
                        source: "verifier",
                        command: None,
                        summary: "verifier reported failure".into(),
                        output: truncate_chars(description.trim(), MAX_REPAIR_OUTPUT_CHARS),
                    });
                }
            }
            AgentResult::NeedsClarification {
                question, context, ..
            } => diagnostics.push(RepairDiagnostic {
                source: "verifier",
                command: None,
                summary: "verifier asked for clarification".into(),
                output: truncate_chars(
                    &format!("question: {question}\ncontext: {context}"),
                    MAX_REPAIR_OUTPUT_CHARS,
                ),
            }),
            AgentResult::MaxTurnsReached => {
                tracing::info!("verifier did not complete; falling back to deterministic checks")
            }
        }

        if let Some(post_error) = workspace_check(&projected_root_str)
            && is_novel_diagnostic(baseline_error, &post_error)
        {
            diagnostics.push(post_error);
        }
    }

    diagnostics.truncate(MAX_REPAIR_DIAGNOSTICS);
    Ok(diagnostics)
}

fn explicit_literal_coverage_diagnostics(
    request: &str,
    change_set: &ChangeSet,
    file_state: &std::collections::HashMap<String, String>,
) -> Vec<RepairDiagnostic> {
    let touched_paths = change_set
        .ops
        .iter()
        .map(|op| op.path().to_string())
        .collect::<Vec<_>>();
    if touched_paths.is_empty() {
        return Vec::new();
    }
    let touched_set = touched_paths
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    let content_literals = explicit_request_literals(request)
        .into_iter()
        .filter(|literal| !touched_set.contains(literal))
        .collect::<Vec<_>>();
    if content_literals.is_empty() {
        return Vec::new();
    }

    let request_lower = request.to_ascii_lowercase();
    let mut diagnostics = Vec::new();
    for path in touched_paths {
        let path_lower = path.to_ascii_lowercase();
        let basename_lower = std::path::Path::new(&path)
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.to_ascii_lowercase())
            .unwrap_or_default();
        if !request_lower.contains(&path_lower) && !request_lower.contains(&basename_lower) {
            continue;
        }
        let Some(content) = file_state.get(&path) else {
            continue;
        };
        if content_literals
            .iter()
            .any(|literal| content.contains(literal))
        {
            continue;
        }
        diagnostics.push(RepairDiagnostic {
            source: "request_literal_coverage",
            command: None,
            summary: format!(
                "changed file '{path}' does not contain any explicit request literals"
            ),
            output: format!(
                "Request literals: {}\nFile: {path}",
                content_literals.join(", ")
            ),
        });
    }
    diagnostics
}

fn structural_file_diagnostics(
    change_set: &ChangeSet,
    file_state: &std::collections::HashMap<String, String>,
) -> Vec<RepairDiagnostic> {
    let mut diagnostics = Vec::new();
    let mut seen_paths = std::collections::HashSet::new();
    for path in change_set.ops.iter().map(|op| op.path().to_string()) {
        if !seen_paths.insert(path.clone()) {
            continue;
        }
        if !is_structural_check_candidate(&path) {
            continue;
        }
        let Some(content) = file_state.get(&path) else {
            continue;
        };
        let warnings = structural_warnings(content);
        if warnings.is_empty() {
            continue;
        }
        diagnostics.push(RepairDiagnostic {
            source: "structure",
            command: None,
            summary: format!("changed file '{path}' has structural anomalies"),
            output: warnings.join("\n"),
        });
    }
    diagnostics
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

    let mut declarations = std::collections::HashMap::new();
    for (idx, line) in contents.lines().enumerate() {
        let Some(key) = declaration_key(line) else {
            continue;
        };
        if let Some(first_line) = declarations.get(&key) {
            warnings.push(format!(
                "duplicate declaration '{key}' at lines {first_line} and {}",
                idx + 1
            ));
        } else {
            declarations.insert(key, idx + 1);
        }
    }

    warnings
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

fn explicit_request_literals(request: &str) -> Vec<String> {
    let mut literals = Vec::new();
    let mut in_tick = false;
    let mut current = String::new();
    for ch in request.chars() {
        if ch == '`' {
            if in_tick {
                let lit = current.trim();
                if !lit.is_empty() {
                    literals.push(lit.to_string());
                }
                current.clear();
            }
            in_tick = !in_tick;
            continue;
        }
        if in_tick {
            current.push(ch);
        }
    }
    literals.sort();
    literals.dedup();
    literals
}

fn build_repair_request(execute_request: &str, diagnostics: &[RepairDiagnostic]) -> String {
    let mut request = String::new();
    request.push_str(execute_request);
    request.push_str("\n\nRepair the failing implementation using these exact diagnostics. Fix the concrete failing cause first.\n\nREPAIR DIAGNOSTICS:\n");
    for (idx, diagnostic) in diagnostics.iter().enumerate() {
        request.push_str(&format!(
            "{}. source: {}\n   summary: {}\n",
            idx + 1,
            diagnostic.source,
            diagnostic.summary
        ));
        if let Some(command) = &diagnostic.command {
            request.push_str(&format!("   command: {command}\n"));
        }
        request.push_str("   output:\n");
        for line in diagnostic.output.lines() {
            request.push_str("   ");
            request.push_str(line);
            request.push('\n');
        }
    }
    request.push_str("Use the diagnostics above instead of re-deriving the failure from scratch.");
    request
}

fn is_novel_diagnostic(baseline: Option<&RepairDiagnostic>, candidate: &RepairDiagnostic) -> bool {
    baseline
        .map(|base| base.command != candidate.command || base.output != candidate.output)
        .unwrap_or(true)
}

fn format_command_outcome(outcome: &shunt_core::CommandOutcome) -> String {
    let stdout = outcome.stdout.trim();
    let stderr = outcome.stderr.trim();
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => format!("exit_code: {}", outcome.exit_code),
        (false, true) => format!("exit_code: {}\nstdout:\n{}", outcome.exit_code, stdout),
        (true, false) => format!("exit_code: {}\nstderr:\n{}", outcome.exit_code, stderr),
        (false, false) => format!(
            "exit_code: {}\nstdout:\n{}\nstderr:\n{}",
            outcome.exit_code, stdout, stderr
        ),
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (count, ch) in text.chars().enumerate() {
        if count == max_chars {
            out.push_str("...[truncated]");
            break;
        }
        out.push(ch);
    }
    out
}

fn copy_workspace_tree(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            copy_workspace_tree(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            if let Some(parent) = dst_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&src_path, &dst_path)?;
        } else if file_type.is_symlink() {
            copy_symlink(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn copy_symlink(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    use std::os::unix::fs as unix_fs;
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    let target = fs::read_link(src)?;
    unix_fs::symlink(target, dst)
}

#[cfg(not(unix))]
fn copy_symlink(src: &Path, dst: &Path) -> Result<(), std::io::Error> {
    let resolved = fs::canonicalize(src)?;
    if resolved.is_dir() {
        copy_workspace_tree(&resolved, dst)
    } else {
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(resolved, dst)?;
        Ok(())
    }
}

const MAX_EVIDENCE_PER_SCOPE: usize = 6;
const MAX_FILE_EVIDENCE_PER_SCOPE: usize = 5;
const MAX_TEXT_PREVIEW_BYTES: usize = 2048;
const MAX_TEXT_PREVIEW_CHARS: usize = 220;
const MAX_CHANGE_CANDIDATES: usize = 4;
const MAX_OBSERVE_EVIDENCE: usize = 8;
const MAX_OBSERVE_CANDIDATES: usize = 2;
const OBSERVE_SCOPE_BUDGET: usize = 2;
const OBSERVE_ROOT_FILES: &[&str] = &[
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    "requirements.txt",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "tsconfig.json",
    "jsconfig.json",
    "README.md",
];
const OBSERVE_ROOT_DIRS: &[&str] = &["src", "app", "lib", "tests", "test"];

fn gather_workspace_evidence(
    workspace_root: &str,
    target_scope: &[String],
) -> Result<Vec<EvidenceRef>, std::io::Error> {
    let root = Path::new(workspace_root);
    let mut evidence = Vec::new();

    if target_scope.is_empty() {
        push_path_evidence(&mut evidence, root, root, MAX_EVIDENCE_PER_SCOPE)?;
        return Ok(evidence);
    }

    for scope in target_scope {
        let path = root.join(scope);
        push_path_evidence(&mut evidence, root, &path, MAX_EVIDENCE_PER_SCOPE)?;
    }

    Ok(evidence)
}

fn merge_evidence_refs(existing: Vec<EvidenceRef>, incoming: Vec<EvidenceRef>) -> Vec<EvidenceRef> {
    let mut merged = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for item in existing.into_iter().chain(incoming) {
        let key = format!("{}:{}", item.locator, item.summary);
        if seen.insert(key) {
            merged.push(item);
        }
    }
    merged
}

fn prune_candidate_evidence(artifact: &mut UnderstandingArtifact, previous_candidates: &[String]) {
    artifact.evidence.retain(|evidence| {
        !previous_candidates
            .iter()
            .any(|path| path == &evidence.locator)
            && !evidence.summary.starts_with("score=")
    });
}

fn is_execution_ready(artifact: &UnderstandingArtifact) -> bool {
    // The orchestrator fills target_scope from probe results, including
    // not-yet-existing paths for scaffold tasks.  An empty scope means no probe
    // succeeded — treat that as not ready.
    !artifact.target_scope.is_empty() || !artifact.candidate_files.is_empty()
}

fn pending_stage(kind: ExecutionStageKind) -> StageRecord {
    StageRecord {
        kind,
        status: StageStatus::Pending,
        summary: String::new(),
        verifiers: vec![],
        started_at: None,
        finished_at: None,
    }
}

fn running_stage(kind: ExecutionStageKind, now: OffsetDateTime) -> StageRecord {
    StageRecord {
        kind,
        status: StageStatus::Running,
        summary: String::new(),
        verifiers: vec![],
        started_at: Some(now),
        finished_at: None,
    }
}

fn approval_verifier(artifact: &UnderstandingArtifact) -> VerifierOutcome {
    VerifierOutcome {
        verifier: "artifact_approved".into(),
        status: if artifact.approval.status == ApprovalStatus::Approved {
            VerifierStatus::Passed
        } else {
            VerifierStatus::Failed
        },
        summary: format!("artifact approval status is {:?}", artifact.approval.status),
    }
}

fn evidence_verifier(artifact: &UnderstandingArtifact) -> VerifierOutcome {
    VerifierOutcome {
        verifier: "evidence_present".into(),
        status: if artifact.evidence.is_empty() {
            VerifierStatus::Failed
        } else {
            VerifierStatus::Passed
        },
        summary: format!("artifact has {} evidence refs", artifact.evidence.len()),
    }
}

fn ambiguity_verifier(artifact: &UnderstandingArtifact) -> VerifierOutcome {
    let open_ambiguities = artifact
        .ambiguities
        .iter()
        .filter(|ambiguity| ambiguity.status == AmbiguityStatus::Open)
        .count();

    VerifierOutcome {
        verifier: "open_ambiguities".into(),
        status: if open_ambiguities == 0 {
            VerifierStatus::Passed
        } else {
            VerifierStatus::Warning
        },
        summary: format!("artifact has {} open ambiguities", open_ambiguities),
    }
}

fn workspace_test_verifier(workspace_root: &str) -> Result<VerifierOutcome, std::io::Error> {
    let root = Path::new(workspace_root);

    let (verifier, command_label, output) = if root.join("Cargo.toml").is_file() {
        tracing::debug!("run command: cargo test --quiet cwd={workspace_root}");
        (
            "cargo_test",
            "cargo test --quiet",
            Command::new("cargo")
                .arg("test")
                .arg("--quiet")
                .current_dir(workspace_root)
                .output()?,
        )
    } else if root.join("package.json").is_file() {
        tracing::debug!("run command: npm test --silent cwd={workspace_root}");
        (
            "npm_test",
            "npm test --silent",
            Command::new("npm")
                .arg("test")
                .arg("--silent")
                .current_dir(workspace_root)
                .output()?,
        )
    } else {
        return Ok(VerifierOutcome {
            verifier: "workspace_test".into(),
            status: VerifierStatus::Skipped,
            summary: "no built-in workspace test command detected".into(),
        });
    };

    Ok(VerifierOutcome {
        verifier: verifier.into(),
        status: if output.status.success() {
            VerifierStatus::Passed
        } else {
            VerifierStatus::Failed
        },
        summary: command_summary(command_label, &output.stdout, &output.stderr),
    })
}

fn command_summary(command: &str, stdout: &[u8], stderr: &[u8]) -> String {
    let stderr_text = String::from_utf8_lossy(stderr);
    let stdout_text = String::from_utf8_lossy(stdout);
    let detail = stderr_text
        .lines()
        .chain(stdout_text.lines())
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" | ");

    if detail.is_empty() {
        format!("{command}: completed")
    } else {
        format!("{command}: {detail}")
    }
}

fn verifier_summary(verifiers: &[VerifierOutcome]) -> String {
    let failed: Vec<String> = verifiers
        .iter()
        .filter(|v| v.status == VerifierStatus::Failed)
        .map(|v| format!("{}: {}", v.verifier, v.summary))
        .collect();
    if !failed.is_empty() {
        return failed.join("; ");
    }
    // All passed/skipped — compact tag list.
    verifiers
        .iter()
        .map(|v| format!("{}={:?}", v.verifier, v.status))
        .collect::<Vec<_>>()
        .join(", ")
}

fn stage_verifiers(recipe_run: &RecipeRun, kind: ExecutionStageKind) -> &[VerifierOutcome] {
    recipe_run
        .stages
        .iter()
        .find(|stage| stage.kind == kind)
        .map(|stage| stage.verifiers.as_slice())
        .unwrap_or(&[])
}

fn verifier_summary_from_stage(recipe_run: &RecipeRun, kind: ExecutionStageKind) -> String {
    verifier_summary(stage_verifiers(recipe_run, kind))
}

fn uncertainty_events_from_verifiers(
    task: &TaskRun,
    stage: ExecutionStageKind,
    verifiers: &[VerifierOutcome],
    now: OffsetDateTime,
) -> Vec<UncertaintyEvent> {
    verifiers
        .iter()
        .filter(|verifier| verifier.status != VerifierStatus::Passed)
        .map(|verifier| UncertaintyEvent {
            task_id: task.id.clone(),
            stage: Some(stage),
            kind: uncertainty_kind_for_verifier(verifier),
            summary: format!("{}: {}", verifier.verifier, verifier.summary),
            confidence: None,
            created_at: now,
        })
        .collect()
}

fn uncertainty_kind_for_verifier(verifier: &VerifierOutcome) -> UncertaintyKind {
    match verifier.verifier.as_str() {
        "evidence_present" => UncertaintyKind::MissingEvidence,
        "open_ambiguities" => UncertaintyKind::Ambiguity,
        _ => UncertaintyKind::VerifierFailure,
    }
}

fn ensure_current_stage(
    recipe_run: &RecipeRun,
    expected: ExecutionStageKind,
) -> Result<(), RuntimeError> {
    if recipe_run.current_stage == expected {
        Ok(())
    } else {
        Err(RuntimeError::InvalidStageTransition {
            expected,
            found: recipe_run.current_stage,
        })
    }
}

fn ensure_stage_passed(
    recipe_run: &RecipeRun,
    kind: ExecutionStageKind,
) -> Result<(), RuntimeError> {
    let passed = recipe_run
        .stages
        .iter()
        .find(|stage| stage.kind == kind)
        .map(|stage| stage.status == StageStatus::Passed)
        .unwrap_or(false);

    if passed {
        Ok(())
    } else {
        Err(RuntimeError::ApplyNotPassed)
    }
}

fn complete_stage(
    recipe_run: &mut RecipeRun,
    current: ExecutionStageKind,
    next: ExecutionStageKind,
    summary: String,
    now: OffsetDateTime,
) {
    if let Some(stage) = recipe_run
        .stages
        .iter_mut()
        .find(|stage| stage.kind == current)
    {
        stage.status = StageStatus::Passed;
        stage.summary = summary;
        stage.started_at = Some(stage.started_at.unwrap_or(now));
        stage.finished_at = Some(now);
    }

    if let Some(stage) = recipe_run
        .stages
        .iter_mut()
        .find(|stage| stage.kind == next)
        && stage.status == StageStatus::Pending
    {
        stage.status = StageStatus::Running;
        stage.started_at = Some(now);
    }

    recipe_run.current_stage = next;
    recipe_run.updated_at = now;
}

fn reset_stage(
    recipe_run: &mut RecipeRun,
    kind: ExecutionStageKind,
    status: StageStatus,
    summary: impl Into<String>,
    now: OffsetDateTime,
) {
    if let Some(stage) = recipe_run
        .stages
        .iter_mut()
        .find(|stage| stage.kind == kind)
    {
        stage.status = status;
        stage.summary = summary.into();
        stage.verifiers.clear();
        stage.started_at = Some(now);
        stage.finished_at = None;
    }
}

fn prepare_recipe_run_for_patch_replay(recipe_run: &mut RecipeRun, now: OffsetDateTime) {
    reset_stage(
        recipe_run,
        ExecutionStageKind::Apply,
        StageStatus::Running,
        "replaying apply after patch correction",
        now,
    );
    reset_stage(
        recipe_run,
        ExecutionStageKind::Validate,
        StageStatus::Pending,
        String::new(),
        now,
    );
    if let Some(stage) = recipe_run
        .stages
        .iter_mut()
        .find(|stage| stage.kind == ExecutionStageKind::Validate)
    {
        stage.started_at = None;
    }
    recipe_run.current_stage = ExecutionStageKind::Apply;
    recipe_run.updated_at = now;
}

/// Apply a `ChangeSet` all-or-nothing.  If any op fails after partial application
/// the already-written ops are rolled back by restoring snapshotted content.
fn apply_change_set_transactional(workspace: &Path, cs: &ChangeSet) -> Result<(), RuntimeError> {
    // Resolve workspace to a canonical path once for boundary checks.
    let canonical_workspace = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());

    // Phase 1: snapshot current state + validate every path stays inside workspace.
    let mut snapshots: Vec<(std::path::PathBuf, Option<String>)> = Vec::new();
    for op in &cs.ops {
        let abs = workspace.join(op.path());
        // Normalize to catch symlink traversal — use the parent that must exist.
        let canonical_abs = if abs.exists() {
            abs.canonicalize().unwrap_or_else(|_| abs.clone())
        } else {
            // For new files the path itself doesn't exist yet; canonicalize parent.
            abs.parent()
                .and_then(|p| p.canonicalize().ok())
                .map(|p| p.join(abs.file_name().unwrap_or_default()))
                .unwrap_or_else(|| abs.clone())
        };
        if !canonical_abs.starts_with(&canonical_workspace) {
            return Err(RuntimeError::InvalidGeneratedChangePath(op.path().into()));
        }
        let original = if abs.exists() {
            Some(fs::read_to_string(&abs)?)
        } else {
            None
        };
        snapshots.push((abs, original));
    }

    // Phase 2: apply ops in order; roll back on first failure.
    for (i, op) in cs.ops.iter().enumerate() {
        let abs = workspace.join(op.path());
        if let Err(e) = apply_file_op(&abs, op) {
            for (rollback_path, original) in snapshots.iter().take(i).rev() {
                match original {
                    Some(contents) => {
                        let _ = fs::write(rollback_path, contents);
                    }
                    None => {
                        let _ = fs::remove_file(rollback_path);
                    }
                }
            }
            return Err(e);
        }
    }
    Ok(())
}

fn apply_file_op(abs: &Path, op: &FileOp) -> Result<(), RuntimeError> {
    match op {
        FileOp::Create { contents, .. } => {
            if contents.trim().is_empty() {
                return Err(RuntimeError::EmptyGeneratedChange);
            }
            if let Some(parent) = abs.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(abs, contents)?;
        }
        FileOp::Edit {
            search,
            replacement,
            ..
        } => {
            if search.trim().is_empty() {
                return Err(RuntimeError::InvalidGeneratedPatch);
            }
            let current = fs::read_to_string(abs)?;
            let Some(updated) = apply_search_replace(&current, search, replacement) else {
                return Err(RuntimeError::InvalidGeneratedPatch);
            };
            fs::write(abs, updated)?;
        }
        FileOp::Delete { .. } => {
            // Only delete regular files — never directories.
            if abs.is_dir() {
                return Err(RuntimeError::InvalidGeneratedChangePath(
                    abs.to_string_lossy().into_owned(),
                ));
            }
            fs::remove_file(abs)?;
        }
    }
    Ok(())
}

fn push_path_evidence(
    evidence: &mut Vec<EvidenceRef>,
    workspace_root: &Path,
    path: &Path,
    budget: usize,
) -> Result<(), std::io::Error> {
    let scope_start = evidence.len();
    let locator = relative_locator(workspace_root, path);

    if !path.exists() {
        evidence.push(EvidenceRef {
            kind: EvidenceKind::Other,
            locator,
            summary: "path is referenced by scope but does not exist".into(),
        });
        return Ok(());
    }

    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        evidence.push(EvidenceRef {
            kind: EvidenceKind::File,
            locator,
            summary: summarize_file(path, metadata.len())?,
        });
        return Ok(());
    }

    let entries = fs::read_dir(path)?
        .filter_map(Result::ok)
        .collect::<Vec<_>>();
    let names = entries
        .iter()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let preview = names.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
    let summary = if preview.is_empty() {
        "directory exists and is empty".into()
    } else {
        format!(
            "directory exists with {} entries; sample: {}",
            names.len(),
            preview
        )
    };

    evidence.push(EvidenceRef {
        kind: EvidenceKind::File,
        locator,
        summary,
    });

    if budget > 1 {
        for file in collect_file_evidence(workspace_root, path, MAX_FILE_EVIDENCE_PER_SCOPE)? {
            if evidence.len() - scope_start >= budget {
                break;
            }
            evidence.push(file);
        }
    }

    Ok(())
}

fn collect_file_evidence(
    workspace_root: &Path,
    path: &Path,
    limit: usize,
) -> Result<Vec<EvidenceRef>, std::io::Error> {
    let mut files = Vec::new();
    collect_file_evidence_recursive(workspace_root, path, limit, &mut files)?;
    Ok(files)
}

fn collect_file_evidence_recursive(
    workspace_root: &Path,
    path: &Path,
    limit: usize,
    files: &mut Vec<EvidenceRef>,
) -> Result<(), std::io::Error> {
    if files.len() >= limit {
        return Ok(());
    }

    let mut entries = fs::read_dir(path)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    entries.sort();

    for entry in entries {
        if files.len() >= limit {
            break;
        }

        let metadata = fs::metadata(&entry)?;
        if metadata.is_dir() {
            collect_file_evidence_recursive(workspace_root, &entry, limit, files)?;
            continue;
        }

        files.push(EvidenceRef {
            kind: EvidenceKind::File,
            locator: relative_locator(workspace_root, &entry),
            summary: summarize_file(&entry, metadata.len())?,
        });
    }

    Ok(())
}

fn summarize_file(path: &Path, file_size: u64) -> Result<String, std::io::Error> {
    let bytes = fs::read(path)?;
    let prefix = &bytes[..bytes.len().min(MAX_TEXT_PREVIEW_BYTES)];
    let Ok(text) = std::str::from_utf8(prefix) else {
        return Ok(format!(
            "binary or non-utf8 file exists ({} bytes)",
            file_size
        ));
    };

    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let preview = if file_name == "Cargo.toml" {
        summarize_cargo_toml(text)
    } else if file_name == "package.json" {
        summarize_package_json(text)
    } else {
        summarize_text_lines(text)
    };

    Ok(format!(
        "file exists ({} bytes); preview: {}",
        file_size, preview
    ))
}

fn apply_search_replace(current: &str, search: &str, replacement: &str) -> Option<String> {
    // Try the search string literally first (fast path).
    if current.matches(search).count() == 1 {
        return Some(current.replacen(search, replacement, 1));
    }

    // Build progressively normalised variants.
    for normalised in search_normalisations(search) {
        // Normalise the haystack the same way so offsets still match.
        let normalised_current = apply_same_normalization(current, search, &normalised);
        if normalised_current.matches(normalised.as_str()).count() == 1 {
            // Apply to the ORIGINAL current (we only used normalised to locate the match).
            // Strategy: find index in normalised_current, map back to original via line diffing.
            // Simpler: re-build by replacing in original using the normalised needle.
            return Some(normalised_current.replacen(normalised.as_str(), replacement, 1));
        }
    }
    None
}

/// Generate normalised variants of `search` to try against the file.
fn search_normalisations(search: &str) -> Vec<String> {
    let mut variants: Vec<String> = Vec::new();

    // 1. Strip trailing newline (common model artefact).
    let stripped = search.trim_end_matches('\n');
    if stripped != search {
        variants.push(stripped.to_string());
    }

    // 2. Normalise CRLF → LF.
    let lf = search.replace("\r\n", "\n");
    if lf != search {
        variants.push(lf.clone());
        let stripped_lf = lf.trim_end_matches('\n').to_string();
        if stripped_lf != lf {
            variants.push(stripped_lf);
        }
    }

    // 3. Strip per-line trailing whitespace (editors/formatters often differ).
    let rtrimmed: String = search
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    if rtrimmed != search && !rtrimmed.trim().is_empty() {
        variants.push(rtrimmed.clone());
        variants.push(rtrimmed.trim_end_matches('\n').to_string());
    }

    // 4. Fully trim the whole string (last resort for single-line searches).
    let full_trim = search.trim().to_string();
    if !full_trim.is_empty() && full_trim != search {
        variants.push(full_trim);
    }

    variants
}

/// Apply the same normalisation to the haystack so the needle still matches.
fn apply_same_normalization(
    current: &str,
    original_search: &str,
    normalised_search: &str,
) -> String {
    // Detect which transformation was applied and mirror it.
    if normalised_search == original_search.replace("\r\n", "\n") {
        return current.replace("\r\n", "\n");
    }
    let rtrimmed_orig: String = original_search
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    if normalised_search == rtrimmed_orig
        || normalised_search == rtrimmed_orig.trim_end_matches('\n')
    {
        return current
            .lines()
            .map(|l| l.trim_end())
            .collect::<Vec<_>>()
            .join("\n");
    }
    // For stripped trailing newline or full-trim: return current as-is so literal match is tried.
    current.to_string()
}

fn has_open_ambiguity(artifact: &UnderstandingArtifact) -> bool {
    artifact
        .ambiguities
        .iter()
        .any(|a| a.status == AmbiguityStatus::Open)
}

/// Only `UserDecision` open ambiguities should gate the pipeline waiting for user input.
fn has_open_user_decision_ambiguity(artifact: &UnderstandingArtifact) -> bool {
    artifact
        .ambiguities
        .iter()
        .any(|a| a.status == AmbiguityStatus::Open && a.kind == AmbiguityKind::UserDecision)
}

/// Runs all registered resolvers against every open Lookup ambiguity.
/// Mutates the artifact in-place, marking resolved ones as Resolved and
/// injecting the resolution as a manual evidence hint.
/// Returns the count of ambiguities that were successfully resolved.
fn auto_resolve_lookup_ambiguities(
    knowledge: &KnowledgeService,
    artifact: &mut UnderstandingArtifact,
) -> usize {
    let lookup_ids: Vec<String> = artifact
        .ambiguities
        .iter()
        .filter(|a| a.status == AmbiguityStatus::Open && a.kind == AmbiguityKind::Lookup)
        .map(|a| a.id.clone())
        .collect();

    if lookup_ids.is_empty() {
        return 0;
    }

    let lookup_refs: Vec<&shunt_core::Ambiguity> = artifact
        .ambiguities
        .iter()
        .filter(|a| lookup_ids.contains(&a.id))
        .collect();

    let resolutions = knowledge.resolve_lookup_ambiguities(&lookup_refs);
    let resolved_count = resolutions.len();

    for (id, resolution) in resolutions {
        tracing::debug!("auto-resolved ambiguity {id}: {resolution}");
        if let Some(a) = artifact.ambiguities.iter_mut().find(|a| a.id == id) {
            a.status = AmbiguityStatus::Resolved;
            a.resolution = Some(resolution.clone());
        }
        // Surface the resolution as a manual evidence entry so the re-run clarify
        // call sees it as established fact.
        artifact.manual_evidence.push(shunt_core::ManualEvidence {
            ecosystem: "registry".into(),
            package: id.clone(),
            version: None,
            version_status: shunt_core::ManualVersionStatus::Unversioned,
            source: "registry-lookup".into(),
            locator: String::new(),
            title: Some("Auto-resolved registry fact".into()),
            excerpt: resolution,
            relevance_reason: format!("resolved lookup ambiguity {id}"),
            confidence: 0.95,
        });
    }

    resolved_count
}

fn observe_workspace_evidence(
    workspace_root: &str,
    artifact: &UnderstandingArtifact,
    runtime: &TaskRuntime,
) -> RuntimeResult<Vec<EvidenceRef>> {
    let root = Path::new(workspace_root);
    let mut evidence = Vec::new();
    let mut seen = std::collections::BTreeSet::new();

    push_unique_path_evidence(&mut evidence, &mut seen, root, root, 1)?;
    if let Some(profile) = workspace_profile_evidence(root) {
        seen.insert(profile.locator.clone());
        evidence.push(profile);
    }

    for path in observation_seed_paths(root) {
        if evidence.len() >= MAX_OBSERVE_EVIDENCE {
            break;
        }
        push_unique_path_evidence(&mut evidence, &mut seen, root, &path, OBSERVE_SCOPE_BUDGET)?;
    }

    if evidence.len() < MAX_OBSERVE_EVIDENCE
        && let Ok(packet) = runtime.localize_context_packet(workspace_root, artifact)
    {
        for candidate in packet
            .primary_candidates
            .iter()
            .take(MAX_OBSERVE_CANDIDATES)
        {
            if evidence.len() >= MAX_OBSERVE_EVIDENCE {
                break;
            }
            if seen.insert(candidate.file.path.clone()) {
                evidence.push(EvidenceRef {
                    kind: EvidenceKind::File,
                    locator: candidate.file.path.clone(),
                    summary: format!("observed likely relevant file: {}", candidate.file.summary),
                });
            }
        }
    }

    evidence.truncate(MAX_OBSERVE_EVIDENCE);
    Ok(evidence)
}

fn push_unique_path_evidence(
    evidence: &mut Vec<EvidenceRef>,
    seen: &mut std::collections::BTreeSet<String>,
    workspace_root: &Path,
    path: &Path,
    budget: usize,
) -> Result<(), std::io::Error> {
    let start = evidence.len();
    let mut scoped = Vec::new();
    push_path_evidence(&mut scoped, workspace_root, path, budget)?;
    for item in scoped {
        if seen.insert(item.locator.clone()) {
            evidence.push(item);
        }
    }
    if evidence.len() == start && path == workspace_root && seen.insert(".".into()) {
        evidence.push(EvidenceRef {
            kind: EvidenceKind::File,
            locator: ".".into(),
            summary: "workspace root is available".into(),
        });
    }
    Ok(())
}

fn observation_seed_paths(root: &Path) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    for name in OBSERVE_ROOT_FILES {
        let path = root.join(name);
        if path.exists() {
            paths.push(path);
        }
    }
    for name in OBSERVE_ROOT_DIRS {
        let path = root.join(name);
        if path.exists() {
            paths.push(path);
        }
    }
    paths
}

fn workspace_profile_evidence(root: &Path) -> Option<EvidenceRef> {
    let entries = fs::read_dir(root).ok()?;
    let mut manifests = Vec::new();
    let mut locks = Vec::new();
    let mut source_dirs = Vec::new();
    let mut entry_files = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let lower = name.to_ascii_lowercase();
        if path.is_file() {
            if OBSERVE_ROOT_FILES
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(&name))
            {
                if lower.contains("lock") {
                    locks.push(name.clone());
                } else {
                    manifests.push(name.clone());
                }
            }
            if lower.starts_with("index.") || lower.starts_with("main.") {
                entry_files.push(name);
            }
        } else if path.is_dir()
            && OBSERVE_ROOT_DIRS
                .iter()
                .any(|candidate| candidate == &lower)
        {
            source_dirs.push(name);
        }
    }

    if manifests.is_empty() && locks.is_empty() && source_dirs.is_empty() && entry_files.is_empty()
    {
        return None;
    }

    let package_manager = if locks.iter().any(|name| name == "package-lock.json") {
        Some("npm")
    } else if locks.iter().any(|name| name == "pnpm-lock.yaml") {
        Some("pnpm")
    } else if locks.iter().any(|name| name == "yarn.lock") {
        Some("yarn")
    } else {
        None
    };

    let mut parts = Vec::new();
    if let Some(package_manager) = package_manager {
        parts.push(format!("package manager appears to be {package_manager}"));
    }
    if !manifests.is_empty() {
        parts.push(format!("root manifests: {}", manifests.join(", ")));
    }
    if !locks.is_empty() {
        parts.push(format!("lockfiles: {}", locks.join(", ")));
    }
    if !source_dirs.is_empty() {
        parts.push(format!("source dirs: {}", source_dirs.join(", ")));
    }
    if !entry_files.is_empty() {
        parts.push(format!("root entry files: {}", entry_files.join(", ")));
    }

    Some(EvidenceRef {
        kind: EvidenceKind::Other,
        locator: "workspace-profile".into(),
        summary: parts.join("; "),
    })
}

fn empty_context_packet() -> ContextPacket {
    ContextPacket {
        backend: RetrievalBackend::Lexical,
        query: shunt_localize::SearchQuery {
            intent: shunt_localize::SearchIntent::Unknown,
            literals: Vec::new(),
            repo_terms: Vec::new(),
            regexes: Vec::new(),
            symbol_guesses: Vec::new(),
        },
        primary_candidates: Vec::new(),
        supporting_candidates: Vec::new(),
    }
}

/// Build a minimal ContextPacket from probe-resolved candidate files.
/// Used to feed the knowledge service (which reads package manifests from
/// candidates) without re-running the full localizer pipeline.
fn build_context_packet_from_scope(candidates: &[CandidateFile]) -> ContextPacket {
    use shunt_localize::{CandidateRole, RankedCandidate};
    let primary = candidates
        .iter()
        .map(|cf| RankedCandidate {
            file: cf.clone(),
            role: CandidateRole::Unknown,
            score: 1.0,
            reasons: vec!["probe scope".into()],
            snippets: Vec::new(),
        })
        .collect();
    ContextPacket {
        backend: RetrievalBackend::Lexical,
        query: SearchQuery {
            intent: shunt_localize::SearchIntent::Unknown,
            literals: Vec::new(),
            repo_terms: Vec::new(),
            regexes: Vec::new(),
            symbol_guesses: Vec::new(),
        },
        primary_candidates: primary,
        supporting_candidates: Vec::new(),
    }
}

fn trace_manual_context(artifact: &UnderstandingArtifact) {
    if !artifact.package_facts.is_empty() {
        tracing::debug!(
            "manual package facts: {}",
            artifact
                .package_facts
                .iter()
                .map(|fact| format!(
                    "{}:{}@{}",
                    fact.ecosystem,
                    fact.name,
                    fact.version.as_deref().unwrap_or("unknown")
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    for manual in &artifact.manual_evidence {
        tracing::debug!(
            "manual evidence package={} version_status={:?} source={} locator={}",
            manual.package,
            manual.version_status,
            manual.source,
            manual.locator
        );
    }
}

fn summarize_understand_artifact(artifact: &UnderstandingArtifact) -> String {
    let mut lines = vec![format!("understanding: {}", artifact.interpreted_goal)];
    if !artifact.target_scope.is_empty() {
        lines.push(format!(
            "focus: {}",
            artifact
                .target_scope
                .iter()
                .take(4)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !artifact.risks.is_empty() {
        lines.push(format!(
            "risk: {}",
            artifact
                .risks
                .iter()
                .take(2)
                .map(|risk| risk.summary.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }
    if let Some(ambiguity) = artifact
        .ambiguities
        .iter()
        .find(|ambiguity| ambiguity.status == AmbiguityStatus::Open)
    {
        lines.push(format!("question: {}", ambiguity.question));
    }
    lines.join("\n")
}

/// Deterministic workspace profile extracted from manifest files before any LLM call.
/// Never calls the network; never calls the LLM.  Pure FS read + pattern match.
fn extract_workspace_profile(workspace_root: &str) -> shunt_core::WorkspaceProfile {
    use shunt_core::WorkspaceProfile;
    let root = std::path::Path::new(workspace_root);
    let mut profile = WorkspaceProfile {
        topology: "unknown".into(),
        ..Default::default()
    };

    // ── Node.js ──────────────────────────────────────────────────────────────
    let pkg_path = root.join("package.json");
    if pkg_path.is_file() {
        profile.runtimes.push("node".into());
        if let Ok(text) = fs::read_to_string(&pkg_path)
            && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
        {
            let mut all_deps: Vec<String> = Vec::new();
            for key in &["dependencies", "devDependencies"] {
                if let Some(deps) = json.get(key).and_then(|v| v.as_object()) {
                    for name in deps.keys() {
                        all_deps.push(name.clone());
                    }
                }
            }
            profile.dependencies = all_deps.clone();

            // Framework detection — order matters (most specific first).
            let dep_set: std::collections::HashSet<&str> =
                all_deps.iter().map(|s| s.as_str()).collect();
            let version_of = |name: &str| -> String {
                for key in &["dependencies", "devDependencies"] {
                    if let Some(deps) = json.get(key).and_then(|v| v.as_object())
                        && let Some(v) = deps.get(name).and_then(|v| v.as_str())
                    {
                        let clean = v.trim_start_matches('^').trim_start_matches('~');
                        return clean.to_string();
                    }
                }
                "*".into()
            };

            if dep_set.contains("@remix-run/react")
                || dep_set.contains("@remix-run/node")
                || dep_set.contains("@remix-run/dev")
            {
                let ver = version_of("@remix-run/react").to_string();
                profile.frameworks.push(format!("remix@{ver}"));
                // Remix bundles react-router — flag the conflict.
                if dep_set.contains("react-router") || dep_set.contains("react-router-dom") {
                    profile.conflicts.push(
                            "react-router / react-router-dom is already a transitive dependency of Remix. \
                             Adding it directly risks version mismatches and duplicate context providers. \
                             Consider using Remix's built-in routing instead.".into()
                        );
                }
            }
            if dep_set.contains("next") {
                profile
                    .frameworks
                    .push(format!("next@{}", version_of("next")));
                if dep_set.contains("react-router") || dep_set.contains("react-router-dom") {
                    profile.conflicts.push(
                            "Next.js has its own file-based router. Adding react-router creates two competing routers.".into()
                        );
                }
            }
            if dep_set.contains("express") {
                profile
                    .frameworks
                    .push(format!("express@{}", version_of("express")));
            }
            if dep_set.contains("fastify") {
                profile
                    .frameworks
                    .push(format!("fastify@{}", version_of("fastify")));
            }
            if dep_set.contains("vite") || dep_set.contains("@vitejs/plugin-react") {
                profile
                    .frameworks
                    .push(format!("vite@{}", version_of("vite")));
            }
            if dep_set.contains("react-router") || dep_set.contains("react-router-dom") {
                let ver = version_of("react-router");
                if ver != "*" {
                    profile.frameworks.push(format!("react-router@{ver}"));
                }
            }

            // Topology hints.
            if json.get("workspaces").is_some() || root.join("pnpm-workspace.yaml").exists() {
                profile.topology = "monorepo".into();
            } else if dep_set.contains("express")
                || dep_set.contains("fastify")
                || dep_set.contains("koa")
            {
                if dep_set.contains("react") || dep_set.contains("vue") {
                    profile.topology = "fullstack-app".into();
                } else {
                    profile.topology = "service".into();
                }
            } else if dep_set.contains("react")
                || dep_set.contains("vue")
                || dep_set.contains("svelte")
            {
                profile.topology = "single-app".into();
            }
        }
    }

    // ── Rust ─────────────────────────────────────────────────────────────────
    if root.join("Cargo.toml").is_file() {
        profile.runtimes.push("rust".into());
        if root.join("Cargo.lock").is_file() && root.join("crates").is_dir() {
            profile.topology = "monorepo".into();
        } else if profile.topology == "unknown" {
            profile.topology = "library".into();
        }
    }

    // ── Python ───────────────────────────────────────────────────────────────
    if root.join("pyproject.toml").is_file() || root.join("requirements.txt").is_file() {
        profile.runtimes.push("python".into());
    }

    // ── Go ───────────────────────────────────────────────────────────────────
    if root.join("go.mod").is_file() {
        profile.runtimes.push("go".into());
    }

    profile
}

fn summarize_package_json(text: &str) -> String {
    let Ok(json) = serde_json::from_str::<serde_json::Value>(text) else {
        return summarize_text_lines(text);
    };
    let mut parts = Vec::new();
    if let Some(name) = json.get("name").and_then(|v| v.as_str()) {
        parts.push(format!("name: {name}"));
    }
    let mut all_deps: Vec<String> = Vec::new();
    for key in &["dependencies", "devDependencies"] {
        if let Some(deps) = json.get(key).and_then(|v| v.as_object()) {
            for (name, ver) in deps {
                let v = ver
                    .as_str()
                    .unwrap_or("*")
                    .trim_start_matches('^')
                    .trim_start_matches('~');
                all_deps.push(format!("{name}@{v}"));
            }
        }
    }
    if !all_deps.is_empty() {
        parts.push(format!("deps: {}", all_deps.join(", ")));
    }
    if let Some(scripts) = json.get("scripts").and_then(|v| v.as_object()) {
        let keys: Vec<&str> = scripts.keys().map(|s| s.as_str()).collect();
        if !keys.is_empty() {
            parts.push(format!("scripts: {}", keys.join(", ")));
        }
    }
    parts.join("; ")
}

fn summarize_cargo_toml(text: &str) -> String {
    let mut package_name = None;
    let mut in_dependencies = false;
    let mut dependencies = Vec::new();

    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') {
            in_dependencies = line == "[dependencies]";
            continue;
        }

        if package_name.is_none() && line.starts_with("name =") {
            package_name = Some(line.to_string());
        }

        if in_dependencies && let Some((name, _)) = line.split_once('=') {
            dependencies.push(name.trim().to_string());
        }

        if dependencies.len() >= 4 && package_name.is_some() {
            break;
        }
    }

    let mut parts = Vec::new();
    if let Some(name) = package_name {
        parts.push(name);
    }
    if !dependencies.is_empty() {
        parts.push(format!("dependencies: {}", dependencies.join(", ")));
    }

    if parts.is_empty() {
        summarize_text_lines(text)
    } else {
        clamp_preview(&parts.join("; "))
    }
}

fn summarize_text_lines(text: &str) -> String {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(4)
        .collect::<Vec<_>>();

    if lines.is_empty() {
        "empty text file".into()
    } else {
        clamp_preview(&lines.join(" | "))
    }
}

fn clamp_preview(input: &str) -> String {
    clamp_text(input, MAX_TEXT_PREVIEW_CHARS)
}

fn clamp_text(input: &str, limit: usize) -> String {
    let preview = input.chars().take(limit).collect::<String>();
    if input.chars().count() > limit {
        format!("{preview}...")
    } else {
        preview
    }
}

fn relative_locator(workspace_root: &Path, path: &Path) -> String {
    path.strip_prefix(workspace_root)
        .map(|value| {
            let text = value.display().to_string();
            if text.is_empty() { ".".into() } else { text }
        })
        .unwrap_or_else(|_| path.display().to_string())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use shunt_core::{
        ApprovalStatus, CandidateFile, ChangeSet, CorrectionKind, EvidenceKind, EvidenceRef,
        ExecutionStageKind, FileOp, FrontierReason, FrontierStatus, RecipeRef, RequiredPath,
        RequiredPathIntent, StageStatus, TaskPhase, UncertaintyEvent, UncertaintyKind,
        VerifierStatus, WorkContract,
    };
    use shunt_infer::{ToolCall, ToolProvider, ToolSpec};
    use time::OffsetDateTime;
    use time::macros::datetime;

    use super::{ArtifactUpdate, TaskRuntime};
    use shunt_store::SqliteStore;

    #[test]
    fn work_contract_accepts_created_required_path() {
        let root = temp_workspace("work-contract-created");
        fs::create_dir_all(&root).unwrap();
        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);
        let (_, mut artifact) = runtime
            .start_task(
                now,
                "task-contract-created",
                "artifact-contract-created",
                root.display().to_string(),
                "create the output file",
            )
            .unwrap();
        artifact.work_contract = WorkContract {
            required_paths: vec![RequiredPath {
                path: "out.txt".into(),
                intent: RequiredPathIntent::CreateOrUpdate,
                reason: "requested output".into(),
            }],
            behavioral_checks: vec![],
        };
        let change_set = ChangeSet {
            ops: vec![FileOp::Create {
                path: "out.txt".into(),
                contents: "ok".into(),
            }],
            commands: vec![],
        };

        assert!(
            super::validate_work_contract(&root.display().to_string(), &artifact, &change_set)
                .is_none()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn work_contract_rejects_missing_required_path() {
        let root = temp_workspace("work-contract-missing");
        fs::create_dir_all(&root).unwrap();
        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);
        let (_, mut artifact) = runtime
            .start_task(
                now,
                "task-contract-missing",
                "artifact-contract-missing",
                root.display().to_string(),
                "create the output file",
            )
            .unwrap();
        artifact.work_contract = WorkContract {
            required_paths: vec![RequiredPath {
                path: "out.txt".into(),
                intent: RequiredPathIntent::CreateOrUpdate,
                reason: "requested output".into(),
            }],
            behavioral_checks: vec![],
        };
        let change_set = ChangeSet {
            ops: vec![],
            commands: vec![],
        };

        let error =
            super::validate_work_contract(&root.display().to_string(), &artifact, &change_set)
                .expect("contract should fail");
        assert!(error.contains("out.txt"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn work_contract_normalizes_absolute_workspace_paths() {
        let root = temp_workspace("work-contract-absolute");
        fs::create_dir_all(&root).unwrap();
        let absolute = root.join("out.txt").display().to_string();
        let relative = super::workspace_relative_path(&root.display().to_string(), &absolute);

        assert_eq!(relative.as_deref(), Some("out.txt"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn repair_request_formats_named_diagnostics() {
        let request = super::build_repair_request(
            "fix the failing task",
            &[
                super::RepairDiagnostic {
                    source: "setup",
                    command: Some("pnpm install".into()),
                    summary: "projected setup command failed".into(),
                    output: "stderr:\nnetwork down".into(),
                },
                super::RepairDiagnostic {
                    source: "workspace_check",
                    command: Some("cargo check --message-format=short --quiet".into()),
                    summary: "projected workspace check failed".into(),
                    output: "src/lib.rs:1:1: expected item".into(),
                },
            ],
        );

        assert!(request.contains("REPAIR DIAGNOSTICS:"));
        assert!(request.contains("source: setup"));
        assert!(request.contains("command: pnpm install"));
        assert!(request.contains("expected item"));
    }

    #[test]
    fn explicit_literal_coverage_flags_named_file_missing_literal() {
        let diagnostics = super::explicit_literal_coverage_diagnostics(
            "Add a `--json` flag and update README.md to mention `--json`.",
            &ChangeSet {
                ops: vec![FileOp::Create {
                    path: "README.md".into(),
                    contents: "# title\nUsage\n".into(),
                }],
                commands: vec![],
            },
            &std::collections::HashMap::from([("README.md".into(), "# title\nUsage\n".into())]),
        );

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].source, "request_literal_coverage");
        assert!(diagnostics[0].summary.contains("README.md"));
        assert!(diagnostics[0].output.contains("--json"));
    }

    #[test]
    fn structural_file_diagnostics_flags_corrupted_changed_file() {
        let content =
            "export function loadUser() {\n  try {\n}\n\nexport function loadUser() {\n}\n";
        let diagnostics = super::structural_file_diagnostics(
            &ChangeSet {
                ops: vec![FileOp::Create {
                    path: "src/users.ts".into(),
                    contents: content.into(),
                }],
                commands: vec![],
            },
            &std::collections::HashMap::from([("src/users.ts".into(), content.into())]),
        );

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].source, "structure");
        assert!(diagnostics[0].output.contains("unmatched"));
        assert!(diagnostics[0].output.contains("function loadUser"));
    }

    #[test]
    fn explicit_request_literals_extracts_backticked_content() {
        let literals = super::explicit_request_literals(
            "Add `--json` in `src/main.rs` and thread it through `ReportOptions`.",
        );
        assert!(literals.contains(&"--json".to_string()));
        assert!(literals.contains(&"src/main.rs".to_string()));
        assert!(literals.contains(&"ReportOptions".to_string()));
    }

    #[test]
    fn projected_workspace_check_uses_applied_change_set() {
        let root = unique_temp_dir("projected-workspace-check");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"projected-workspace-check\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn ready() -> bool { true }\n").unwrap();

        assert!(super::workspace_check(&root.display().to_string()).is_none());

        let projected = tempfile::tempdir().unwrap();
        let projected_root = projected.path().join("workspace");
        super::copy_workspace_tree(&root, &projected_root).unwrap();
        super::apply_change_set_transactional(
            &projected_root,
            &ChangeSet {
                ops: vec![FileOp::Edit {
                    path: "src/lib.rs".into(),
                    search: "pub fn ready() -> bool { true }\n".into(),
                    replacement: "pub fn ready( -> bool { true }\n".into(),
                }],
                commands: vec![],
            },
        )
        .unwrap();

        let projected_failure = super::workspace_check(&projected_root.display().to_string());
        assert!(projected_failure.is_some());
        assert!(projected_failure.unwrap().output.contains("src/lib.rs"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn collect_repair_diagnostics_returns_setup_failure_without_verifier() {
        let root = unique_temp_dir("repair-setup-failure");
        fs::create_dir_all(&root).unwrap();
        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);
        let (_, artifact) = runtime
            .start_task(
                now,
                "task-setup-failure",
                "artifact-setup-failure",
                root.display().to_string(),
                "repair setup failure",
            )
            .unwrap();

        let diagnostics = super::collect_repair_diagnostics(
            &StubProvider,
            &root.display().to_string(),
            &artifact,
            &ChangeSet {
                ops: vec![],
                commands: vec![shunt_core::CommandSpec::new(
                    "definitely-not-a-real-command",
                    std::iter::empty::<&str>(),
                )],
            },
            &std::collections::HashMap::new(),
            None,
            &[],
            None,
            None,
        )
        .unwrap();

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].source, "setup");
        assert!(diagnostics[0].output.contains("spawn error"));

        fs::remove_dir_all(root).unwrap();
    }

    fn temp_workspace(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "shunt-runtime-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn starts_and_revises_a_task() {
        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        let (task, artifact) = runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                "/tmp/workspace",
                "fix config loading",
            )
            .unwrap();

        assert_eq!(task.phase, TaskPhase::Observe);
        assert_eq!(artifact.interpreted_goal, "fix config loading");

        let revised = runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    interpreted_goal: Some("repair config loading failure".into()),
                    success_criteria: Some(vec!["config loads successfully".into()]),
                    confidence: Some(0.65),
                    ..Default::default()
                },
            )
            .unwrap()
            .unwrap();

        assert_eq!(revised.interpreted_goal, "repair config loading failure");
        assert_eq!(revised.revision, 2);
    }

    #[test]
    fn records_frontier_case() {
        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        let (task, artifact) = runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                "/tmp/workspace",
                "fix config loading",
            )
            .unwrap();

        let frontier_case = runtime
            .record_frontier_case(
                now,
                "frontier-1",
                &task,
                &artifact,
                FrontierReason::LowConfidence,
                "understanding is too weak to execute",
                vec![UncertaintyEvent {
                    task_id: task.id.clone(),
                    stage: None,
                    kind: UncertaintyKind::LowConfidence,
                    summary: "missing strong evidence".into(),
                    confidence: Some(0.28),
                    created_at: now,
                }],
            )
            .unwrap();

        assert_eq!(frontier_case.status, shunt_core::FrontierStatus::Open);

        let store = runtime.into_store();
        let saved_task = store.get_task_run("task-1").unwrap().unwrap();
        let saved_artifact = store
            .get_understanding_artifact("artifact-1")
            .unwrap()
            .unwrap();

        assert_eq!(saved_task.phase, TaskPhase::Agree);
        assert_eq!(saved_task.frontier_cases.len(), 1);
        assert_eq!(saved_artifact.approval.status, ApprovalStatus::Draft);
    }

    struct StubProvider;

    impl ToolProvider for StubProvider {
        fn call_tool(
            &self,
            system: &str,
            user: &str,
            tool: &ToolSpec,
        ) -> shunt_infer::InferResult<ToolCall> {
            let value = if tool.name == "understand" || system.contains("understanding node") {
                serde_json::json!({
                    "interpreted_goal": "connect the task, runtime, and store loop in the scoped crates",
                    "success_criteria": ["task state moves through the loop", "artifacts persist locally"],
                    "target_scope": ["crates/shunt-core", "crates/shunt-runtime", "crates/shunt-store"],
                    "ambiguities": [{"question": "should the first loop stop at understand?", "options": ["yes", "no"]}],
                    "risks": [{"summary": "evidence confirms files, not execution semantics", "severity": "Medium"}],
                    "confidence": 0.74
                })
            } else if system.contains("coding agent") || system.contains("file-editing agent") {
                if !user.contains("update the marker function") {
                    serde_json::json!({
                        "tool": "done",
                        "description": "inspection complete"
                    })
                } else if user.contains("\"after\"") || user.contains("OK — replaced lines") {
                    serde_json::json!({
                        "tool": "done",
                        "description": "updated marker function"
                    })
                } else if user.contains("<file path=\"src/lib.rs\">") {
                    serde_json::json!({
                        "tool": "replace_lines",
                        "path": "src/lib.rs",
                        "start_line": 1,
                        "end_line": 1
                    })
                } else {
                    serde_json::json!({
                        "tool": "read_file",
                        "path": "src/lib.rs"
                    })
                }
            } else {
                serde_json::json!({
                    "interpreted_goal": "repair config loading failure",
                    "success_criteria": ["config loads successfully"],
                    "constraints": ["keep the implementation lean"],
                    "ambiguities": [{"question": "which file is authoritative?", "options": ["config.rs", "settings.rs"]}],
                    "confidence": 0.72
                })
            };
            Ok(ToolCall {
                name: tool.name.clone(),
                arguments: value,
            })
        }

        fn generate_text(&self, _system: &str, user: &str) -> shunt_infer::InferResult<String> {
            if user.contains("marker") {
                Ok("pub fn marker() -> &'static str { \"after\" }".into())
            } else {
                Ok(String::new())
            }
        }
    }

    #[test]
    fn clarifies_task_with_provider() {
        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                "/tmp/workspace",
                "fix config loading",
            )
            .unwrap();

        let artifact = runtime
            .clarify_task("artifact-1", now, &StubProvider)
            .unwrap()
            .unwrap();

        assert_eq!(artifact.interpreted_goal, "repair config loading failure");
        assert_eq!(artifact.revision, 2);

        let store = runtime.into_store();
        let task = store.get_task_run("task-1").unwrap().unwrap();
        assert_eq!(task.phase, TaskPhase::Clarify);
    }

    #[test]
    fn observe_task_collects_repo_profile_before_clarify() {
        let root = unique_temp_dir("frame-observe");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("package.json"),
            "{\n  \"name\": \"demo\",\n  \"scripts\": {\n    \"dev\": \"node index.js\"\n  }\n}\n",
        )
        .unwrap();
        fs::write(
            root.join("package-lock.json"),
            "{\n  \"name\": \"demo\"\n}\n",
        )
        .unwrap();
        fs::write(root.join("index.js"), "console.log('hello');\n").unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "lets install remix project here",
            )
            .unwrap();

        let artifact = runtime.observe_task("artifact-1", now).unwrap().unwrap();

        assert!(
            artifact
                .evidence
                .iter()
                .any(|evidence| evidence.locator == "workspace-profile")
        );
        assert!(
            artifact
                .evidence
                .iter()
                .any(|evidence| evidence.locator == "package.json")
        );

        let store = runtime.into_store();
        let task = store.get_task_run("task-1").unwrap().unwrap();
        assert_eq!(task.phase, TaskPhase::Clarify);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn understands_task_from_workspace_scope() {
        let root = unique_temp_dir("frame-understand");
        fs::create_dir_all(root.join("crates/shunt-core/src")).unwrap();
        fs::write(
            root.join("crates/shunt-core/Cargo.toml"),
            "[package]\nname = \"shunt-core\"\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/shunt-core/src/lib.rs"),
            "pub struct Core;",
        )
        .unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "wire the first onion loop",
            )
            .unwrap();

        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    target_scope: Some(vec!["crates/shunt-core".into()]),
                    confidence: Some(0.5),
                    ..Default::default()
                },
            )
            .unwrap();

        let artifact = runtime.understand_task("artifact-1", now).unwrap().unwrap();

        assert_eq!(artifact.evidence.len(), 3);
        assert!(artifact.evidence[0].summary.contains("directory exists"));
        assert_eq!(artifact.evidence[1].locator, "crates/shunt-core/Cargo.toml");
        assert!(
            artifact.evidence[1]
                .summary
                .contains("name = \"shunt-core\"")
        );
        assert_eq!(artifact.evidence[2].locator, "crates/shunt-core/src/lib.rs");
        assert!(artifact.evidence[2].summary.contains("pub struct Core;"));
        assert_eq!(artifact.confidence, 0.65);

        let store = runtime.into_store();
        let task = store.get_task_run("task-1").unwrap().unwrap();
        assert_eq!(task.phase, TaskPhase::Localize);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn understands_task_with_provider() {
        let root = unique_temp_dir("frame-understand-provider");
        fs::create_dir_all(root.join("crates/shunt-core/src")).unwrap();
        fs::write(
            root.join("crates/shunt-core/Cargo.toml"),
            "[package]\nname = \"shunt-core\"\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/shunt-core/src/lib.rs"),
            "pub struct Core;",
        )
        .unwrap();
        fs::create_dir_all(root.join("crates/shunt-runtime/src")).unwrap();
        fs::write(
            root.join("crates/shunt-runtime/Cargo.toml"),
            "[package]\nname = \"shunt-runtime\"\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/shunt-runtime/src/lib.rs"),
            "pub struct Runtime;",
        )
        .unwrap();
        fs::create_dir_all(root.join("crates/shunt-store/src")).unwrap();
        fs::write(
            root.join("crates/shunt-store/Cargo.toml"),
            "[package]\nname = \"shunt-store\"\n",
        )
        .unwrap();
        fs::write(
            root.join("crates/shunt-store/src/lib.rs"),
            "pub struct Store;",
        )
        .unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "wire the first onion loop",
            )
            .unwrap();

        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    target_scope: Some(vec![
                        "crates/shunt-core".into(),
                        "crates/shunt-runtime".into(),
                        "crates/shunt-store".into(),
                    ]),
                    confidence: Some(0.5),
                    ..Default::default()
                },
            )
            .unwrap();

        let artifact = runtime
            .understand_task_with_provider("artifact-1", now, &StubProvider)
            .unwrap()
            .unwrap();

        assert_eq!(
            artifact.interpreted_goal,
            "connect the task, runtime, and store loop in the scoped crates"
        );
        assert_eq!(artifact.evidence.len(), 9);
        assert!(
            artifact.evidence[1]
                .summary
                .contains("name = \"shunt-core\"")
        );
        assert!(artifact.evidence[2].summary.contains("pub struct Core;"));
        assert_eq!(artifact.target_scope.len(), 3);
        assert_eq!(artifact.ambiguities.len(), 1);
        assert_eq!(artifact.risks.len(), 1);
        assert_eq!(artifact.confidence, 0.74);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolves_ambiguity_and_approves_artifact() {
        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                "/tmp/workspace",
                "wire the first onion loop",
            )
            .unwrap();

        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    ambiguities: Some(vec![shunt_core::Ambiguity {
                        id: "ambiguity-1".into(),
                        question: "what is the onion loop?".into(),
                        options: vec!["task lifecycle".into(), "event loop".into()],
                        kind: shunt_core::AmbiguityKind::UserDecision,
                        status: shunt_core::AmbiguityStatus::Open,
                        resolution: None,
                    }]),
                    ..Default::default()
                },
            )
            .unwrap();

        let artifact = runtime
            .resolve_ambiguity(
                "artifact-1",
                "ambiguity-1",
                "task lifecycle across core, runtime, and store",
                now,
            )
            .unwrap()
            .unwrap();

        assert_eq!(
            artifact.ambiguities[0].status,
            shunt_core::AmbiguityStatus::Resolved
        );
        assert_eq!(
            artifact.ambiguities[0].resolution.as_deref(),
            Some("task lifecycle across core, runtime, and store")
        );

        let artifact = runtime
            .approve_artifact(
                "artifact-1",
                "user",
                Some("approved for execution".into()),
                now,
            )
            .unwrap()
            .unwrap();

        assert_eq!(artifact.approval.status, ApprovalStatus::Approved);
        assert_eq!(artifact.approval.decided_by.as_deref(), Some("user"));
        assert_eq!(
            artifact.approval.note.as_deref(),
            Some("approved for execution")
        );

        let store = runtime.into_store();
        let task = store.get_task_run("task-1").unwrap().unwrap();
        assert_eq!(task.phase, TaskPhase::Execute);
    }

    #[test]
    fn starts_and_advances_execution_recipe_run() {
        let root = unique_temp_dir("frame-execution-state");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"frame-execution-state\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn ready() -> bool { true }\n").unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "wire the first onion loop",
            )
            .unwrap();
        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    target_scope: Some(vec!["src/lib.rs".into()]),
                    evidence: Some(vec![EvidenceRef {
                        kind: EvidenceKind::File,
                        locator: "src/lib.rs".into(),
                        summary: "source file exists".into(),
                    }]),
                    candidate_files: Some(vec![CandidateFile {
                        path: "src/lib.rs".into(),
                        summary: "execution candidate".into(),
                    }]),
                    ..Default::default()
                },
            )
            .unwrap();
        runtime
            .approve_artifact("artifact-1", "user", Some("approved".into()), now)
            .unwrap();

        let recipe_run = runtime
            .start_execution(
                "task-1",
                now,
                RecipeRef {
                    id: "manual.inspect-propose".into(),
                    version: "v1".into(),
                },
            )
            .unwrap();

        assert_eq!(recipe_run.current_stage, ExecutionStageKind::Inspect);
        assert_eq!(recipe_run.stages[0].status, StageStatus::Running);

        let recipe_run = runtime.execute_inspect("task-1", now).unwrap();
        assert_eq!(recipe_run.stages[0].status, StageStatus::Passed);
        assert_eq!(recipe_run.current_stage, ExecutionStageKind::Propose);
        assert_eq!(recipe_run.stages[1].status, StageStatus::Running);

        let recipe_run = runtime.execute_propose("task-1", now).unwrap();
        assert_eq!(recipe_run.stages[1].status, StageStatus::Passed);
        assert_eq!(recipe_run.current_stage, ExecutionStageKind::Verify);
        assert_eq!(recipe_run.stages[2].status, StageStatus::Running);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn verify_stage_runs_workspace_checks() {
        let root = unique_temp_dir("frame-verify");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"frame-verify\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn ready() -> bool { true }\n").unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "verify the workspace",
            )
            .unwrap();
        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    evidence: Some(vec![EvidenceRef {
                        kind: EvidenceKind::File,
                        locator: "Cargo.toml".into(),
                        summary: "workspace manifest exists".into(),
                    }]),
                    candidate_files: Some(vec![CandidateFile {
                        path: "src/lib.rs".into(),
                        summary: "verify candidate".into(),
                    }]),
                    ..Default::default()
                },
            )
            .unwrap();
        runtime
            .approve_artifact("artifact-1", "user", Some("approved".into()), now)
            .unwrap();
        runtime
            .start_execution(
                "task-1",
                now,
                RecipeRef {
                    id: "manual.inspect-propose".into(),
                    version: "v1".into(),
                },
            )
            .unwrap();
        runtime.execute_inspect("task-1", now).unwrap();
        runtime.execute_propose("task-1", now).unwrap();

        let recipe_run = runtime.execute_verify("task-1", now).unwrap();
        let verify_stage = recipe_run
            .stages
            .iter()
            .find(|stage| stage.kind == ExecutionStageKind::Verify)
            .unwrap();

        assert_eq!(verify_stage.status, StageStatus::Passed);
        assert_eq!(recipe_run.current_stage, ExecutionStageKind::Apply);
        assert!(
            verify_stage
                .verifiers
                .iter()
                .any(|verifier| verifier.verifier == "cargo_test"
                    && verifier.status == VerifierStatus::Passed)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn verify_failure_creates_frontier_case() {
        let root = unique_temp_dir("frame-verify-failure");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"frame-verify-failure\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn ready() -> bool { true }\n").unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "verify the workspace",
            )
            .unwrap();
        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    candidate_files: Some(vec![CandidateFile {
                        path: "src/lib.rs".into(),
                        summary: "verify candidate".into(),
                    }]),
                    ..Default::default()
                },
            )
            .unwrap();
        runtime
            .approve_artifact("artifact-1", "user", Some("approved".into()), now)
            .unwrap();
        runtime
            .start_execution(
                "task-1",
                now,
                RecipeRef {
                    id: "manual.inspect-propose".into(),
                    version: "v1".into(),
                },
            )
            .unwrap();
        runtime.execute_inspect("task-1", now).unwrap();
        runtime.execute_propose("task-1", now).unwrap();

        let recipe_run = runtime.execute_verify("task-1", now).unwrap();
        let verify_stage = recipe_run
            .stages
            .iter()
            .find(|stage| stage.kind == ExecutionStageKind::Verify)
            .unwrap();

        assert_eq!(verify_stage.status, StageStatus::Failed);

        let store = runtime.into_store();
        let task = store.get_task_run("task-1").unwrap().unwrap();
        let frontier_cases = store.list_frontier_cases_for_task("task-1").unwrap();

        assert_eq!(task.frontier_cases.len(), 1);
        assert_eq!(frontier_cases.len(), 1);
        assert_eq!(frontier_cases[0].reason, FrontierReason::VerifierFailure);
        assert_eq!(frontier_cases[0].recipe_run_id, Some(recipe_run.id.clone()));
        assert!(
            frontier_cases[0]
                .uncertainty_events
                .iter()
                .any(|event| event.kind == UncertaintyKind::MissingEvidence)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn applies_proposed_change_and_validates() {
        let root = unique_temp_dir("frame-apply");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"frame-apply\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn marker() -> &'static str { \"before\" }\n",
        )
        .unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "apply one bounded change",
            )
            .unwrap();
        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    evidence: Some(vec![EvidenceRef {
                        kind: EvidenceKind::File,
                        locator: "src/lib.rs".into(),
                        summary: "source file exists".into(),
                    }]),
                    candidate_files: Some(vec![CandidateFile {
                        path: "src/lib.rs".into(),
                        summary: "apply candidate".into(),
                    }]),
                    ..Default::default()
                },
            )
            .unwrap();
        runtime
            .approve_artifact("artifact-1", "user", Some("approved".into()), now)
            .unwrap();
        runtime
            .start_execution(
                "task-1",
                now,
                RecipeRef {
                    id: "manual.inspect-propose".into(),
                    version: "v1".into(),
                },
            )
            .unwrap();
        runtime.execute_inspect("task-1", now).unwrap();
        runtime.execute_propose("task-1", now).unwrap();
        runtime.execute_verify("task-1", now).unwrap();
        runtime
            .set_change_set(
                "task-1",
                now,
                ChangeSet {
                    ops: vec![FileOp::Create {
                        path: "src/lib.rs".into(),
                        contents: "pub fn marker() -> &'static str { \"after\" }\n".into(),
                    }],
                    commands: vec![],
                },
            )
            .unwrap();

        let recipe_run = runtime.execute_apply("task-1", now).unwrap();
        assert_eq!(recipe_run.current_stage, ExecutionStageKind::Validate);
        assert_eq!(recipe_run.stages[3].status, StageStatus::Passed);
        assert_eq!(
            fs::read_to_string(root.join("src/lib.rs")).unwrap(),
            "pub fn marker() -> &'static str { \"after\" }\n"
        );

        let recipe_run = runtime.execute_validate("task-1", now).unwrap();
        assert_eq!(recipe_run.stages[5].status, StageStatus::Passed);

        let store = runtime.into_store();
        let task = store.get_task_run("task-1").unwrap().unwrap();
        assert_eq!(task.phase, TaskPhase::Complete);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validate_failure_creates_frontier_case() {
        let root = unique_temp_dir("frame-validate-failure");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"frame-validate-failure\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn marker() -> &'static str { \"before\" }\n",
        )
        .unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "apply one bounded change",
            )
            .unwrap();
        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    evidence: Some(vec![EvidenceRef {
                        kind: EvidenceKind::File,
                        locator: "src/lib.rs".into(),
                        summary: "source file exists".into(),
                    }]),
                    candidate_files: Some(vec![CandidateFile {
                        path: "src/lib.rs".into(),
                        summary: "validate candidate".into(),
                    }]),
                    ..Default::default()
                },
            )
            .unwrap();
        runtime
            .approve_artifact("artifact-1", "user", Some("approved".into()), now)
            .unwrap();
        runtime
            .start_execution(
                "task-1",
                now,
                RecipeRef {
                    id: "manual.inspect-propose".into(),
                    version: "v1".into(),
                },
            )
            .unwrap();
        runtime.execute_inspect("task-1", now).unwrap();
        runtime.execute_propose("task-1", now).unwrap();
        runtime.execute_verify("task-1", now).unwrap();
        runtime
            .set_change_set(
                "task-1",
                now,
                ChangeSet {
                    ops: vec![FileOp::Create {
                        path: "src/lib.rs".into(),
                        contents: "pub fn marker() -> &'static str { invalid }\n".into(),
                    }],
                    commands: vec![],
                },
            )
            .unwrap();
        runtime.execute_apply("task-1", now).unwrap();

        let recipe_run = runtime.execute_validate("task-1", now).unwrap();
        let validate_stage = recipe_run
            .stages
            .iter()
            .find(|stage| stage.kind == ExecutionStageKind::Validate)
            .unwrap();

        assert_eq!(validate_stage.status, StageStatus::Failed);

        let store = runtime.into_store();
        let task = store.get_task_run("task-1").unwrap().unwrap();
        let frontier_cases = store.list_frontier_cases_for_task("task-1").unwrap();

        assert_eq!(task.phase, TaskPhase::Execute);
        assert_eq!(task.frontier_cases.len(), 1);
        assert_eq!(frontier_cases.len(), 1);
        assert_eq!(frontier_cases[0].reason, FrontierReason::VerifierFailure);
        assert_eq!(frontier_cases[0].recipe_run_id, Some(recipe_run.id.clone()));
        assert!(
            frontier_cases[0]
                .uncertainty_events
                .iter()
                .any(|event| event.kind == UncertaintyKind::VerifierFailure)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn patch_correction_replays_failed_validate_run() {
        let root = unique_temp_dir("frame-correction-replay");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"frame-correction-replay\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn marker() -> &'static str { \"before\" }\n",
        )
        .unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "repair a failed generated patch",
            )
            .unwrap();
        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    evidence: Some(vec![EvidenceRef {
                        kind: EvidenceKind::File,
                        locator: "src/lib.rs".into(),
                        summary: "source file exists".into(),
                    }]),
                    candidate_files: Some(vec![CandidateFile {
                        path: "src/lib.rs".into(),
                        summary: "correction candidate".into(),
                    }]),
                    ..Default::default()
                },
            )
            .unwrap();
        runtime
            .approve_artifact("artifact-1", "user", Some("approved".into()), now)
            .unwrap();
        runtime
            .start_execution(
                "task-1",
                now,
                RecipeRef {
                    id: "manual.inspect-propose".into(),
                    version: "v1".into(),
                },
            )
            .unwrap();
        runtime.execute_inspect("task-1", now).unwrap();
        runtime.execute_propose("task-1", now).unwrap();
        runtime.execute_verify("task-1", now).unwrap();
        runtime
            .set_change_set(
                "task-1",
                now,
                ChangeSet {
                    ops: vec![FileOp::Create {
                        path: "src/lib.rs".into(),
                        contents: "pub fn marker() -> &'static str { invalid }\n".into(),
                    }],
                    commands: vec![],
                },
            )
            .unwrap();
        runtime.execute_apply("task-1", now).unwrap();
        runtime.execute_validate("task-1", now).unwrap();

        let store = runtime.into_store();
        let frontier_case = store
            .list_frontier_cases_for_task("task-1")
            .unwrap()
            .pop()
            .unwrap();
        let runtime = TaskRuntime::new(store);

        let (correction, frontier_case, recipe_run) = runtime
            .create_patch_correction(
                &frontier_case.id.0,
                "correction-1",
                "replace the broken generated patch",
                ChangeSet {
                    ops: vec![FileOp::Create {
                        path: "src/lib.rs".into(),
                        contents: "pub fn marker() -> &'static str { \"after\" }\n".into(),
                    }],
                    commands: vec![],
                },
                now,
            )
            .unwrap();

        assert_eq!(correction.kind, CorrectionKind::PatchRevision);
        assert!(!correction.validated);
        assert_eq!(frontier_case.status, FrontierStatus::InReview);
        assert_eq!(recipe_run.current_stage, ExecutionStageKind::Apply);

        let (correction, frontier_case, recipe_run) =
            runtime.replay_correction("correction-1", now).unwrap();

        assert!(correction.validated);
        assert_eq!(frontier_case.status, FrontierStatus::Corrected);
        assert_eq!(recipe_run.stages[5].status, StageStatus::Passed);
        assert_eq!(
            fs::read_to_string(root.join("src/lib.rs")).unwrap(),
            "pub fn marker() -> &'static str { \"after\" }\n"
        );

        let store = runtime.into_store();
        let task = store.get_task_run("task-1").unwrap().unwrap();
        assert_eq!(task.phase, TaskPhase::Complete);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn generates_proposed_change_with_provider() {
        let root = unique_temp_dir("frame-generate-change");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"frame-generate-change\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn marker() -> &'static str { \"before\" }\n",
        )
        .unwrap();

        let store = SqliteStore::open_in_memory().unwrap();
        let runtime = TaskRuntime::new(store);
        let now = datetime!(2026-05-01 12:00 UTC);

        runtime
            .start_task(
                now,
                "task-1",
                "artifact-1",
                root.display().to_string(),
                "update the marker function",
            )
            .unwrap();
        runtime
            .revise_artifact(
                "artifact-1",
                now,
                ArtifactUpdate {
                    target_scope: Some(vec!["src/lib.rs".into()]),
                    confidence: Some(0.5),
                    candidate_files: Some(vec![CandidateFile {
                        path: "src/lib.rs".into(),
                        summary: "generation candidate".into(),
                    }]),
                    ..Default::default()
                },
            )
            .unwrap();
        runtime.understand_task("artifact-1", now).unwrap();
        runtime
            .approve_artifact("artifact-1", "user", Some("approved".into()), now)
            .unwrap();
        runtime
            .start_execution(
                "task-1",
                now,
                RecipeRef {
                    id: "manual.inspect-propose".into(),
                    version: "v1".into(),
                },
            )
            .unwrap();
        runtime.execute_inspect("task-1", now).unwrap();
        runtime.execute_propose("task-1", now).unwrap();
        runtime.execute_verify("task-1", now).unwrap();

        let recipe_run = runtime
            .generate_proposed_change("task-1", now, &StubProvider, &[], &[])
            .unwrap();

        let cs = recipe_run
            .change_set
            .as_ref()
            .expect("change_set should be set");
        assert_eq!(cs.ops.len(), 1);
        let FileOp::Create { path, contents } = &cs.ops[0] else {
            panic!("expected Create op");
        };
        assert_eq!(path, "src/lib.rs");
        assert!(contents.contains("\"after\""));
        assert_eq!(recipe_run.current_stage, ExecutionStageKind::Apply);

        fs::remove_dir_all(root).unwrap();
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            OffsetDateTime::now_utc().unix_timestamp_nanos()
        ))
    }
}
