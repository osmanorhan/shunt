//! Async `EffectRunner` — executes `Effect` variants declared by the state
//! machine and produces `MachineEvent`s to feed back into the next transition.
//!
//! The runner wraps the synchronous `TaskRuntime` / `ToolProvider` behind
//! `spawn_blocking` so the actor thread stays non-blocking.
//!
//! Design rule: the runner must NEVER mutate `TaskState`.  All it does is turn
//! effects into events.

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tracing::{debug, warn};

use shunt_core::ledger::{
    BoundedOutput, CommandObservation, CommandStatus, LedgerEntry, ToolObservation,
    UserObservation, UserObservationKind,
};
use shunt_core::machine::{Effect, MachineEvent, Notification};
use shunt_core::safety;
use shunt_core::{ArtifactId, FrontierCaseId};
use shunt_infer::{AgentObserver, ToolProvider};

use crate::{RecipeRef, RuntimeError, TaskRuntime};

// ── TxAgentObserver ───────────────────────────────────────────────────────────

/// Bridges `AgentObserver` events into the runner's `RunnerMessage` channel
/// so live agent turns appear as `Notification::AgentToolCall/Result`.
struct TxAgentObserver {
    tx: mpsc::Sender<RunnerMessage>,
}

impl AgentObserver for TxAgentObserver {
    fn on_tool_call(&self, turn: usize, max_turns: usize, tool: &str, summary: &str) {
        let _ = self
            .tx
            .blocking_send(RunnerMessage::Notification(Notification::AgentToolCall {
                turn,
                max_turns,
                tool: tool.into(),
                summary: summary.into(),
            }));
    }
    fn on_tool_result(&self, turn: usize, ok: bool, detail: &str) {
        let _ = self
            .tx
            .blocking_send(RunnerMessage::Notification(Notification::AgentToolResult {
                turn,
                ok,
                detail: detail.into(),
            }));
    }
    fn on_note(&self, text: &str) {
        let _ = self
            .tx
            .blocking_send(RunnerMessage::Notification(Notification::Note {
                text: text.into(),
            }));
    }
}

// ── RunnerSend / RunnerMessage ────────────────────────────────────────────────

/// Messages flowing from the async runner back into the session actor.
#[derive(Debug)]
pub enum RunnerMessage {
    /// The effect produced the given machine event; feed it into `transition`.
    Event(MachineEvent),
    /// A `Notify` effect was executed; the runner emits the resolved event here
    /// for broadcast to external observers (TUI, CLI, …).
    Notification(Notification),
}

// ── EffectRunner ─────────────────────────────────────────────────────────────

/// Runs `Effect` values produced by `TaskMachine::transition`.
///
/// `P` is the `ToolProvider` (LLM backend).  Because `TaskRuntime` holds a
/// `rusqlite::Connection` (not `Send`) it must be created inside the blocking
/// thread and kept there.  The runner constructs a new `TaskRuntime` per call
/// using the stored `store_path`.
pub struct EffectRunner<P: ToolProvider + Clone + Send + Sync + 'static> {
    store_path: String,
    provider: Arc<P>,
    /// Paths (relative to workspace_root) written by the previous task.
    /// Pre-loaded into the AgentSession so follow-up prompts have immediate context.
    pub prior_context_files: Vec<String>,
    /// Extra gitignore-style patterns forwarded to AgentSession on top of built-in defaults.
    pub ignore_patterns: Vec<String>,
}

impl<P: ToolProvider + Clone + Send + Sync + 'static> EffectRunner<P> {
    pub fn new(store_path: impl Into<String>, provider: P) -> Self {
        Self {
            store_path: store_path.into(),
            provider: Arc::new(provider),
            prior_context_files: Vec::new(),
            ignore_patterns: Vec::new(),
        }
    }

    pub fn with_prior_context(mut self, files: Vec<String>) -> Self {
        self.prior_context_files = files;
        self
    }

    pub fn with_ignore_patterns(mut self, patterns: Vec<String>) -> Self {
        self.ignore_patterns = patterns;
        self
    }

    /// Run a single effect, sending results back on `tx`.
    ///
    /// IO-heavy effects are dispatched to `spawn_blocking`; pure / cheap
    /// effects (Notify, Persist) are executed inline.
    ///
    /// Returns `true` if the caller should continue processing subsequent
    /// effects, or `false` if the batch should be aborted (e.g. a critical
    /// setup effect failed and remaining effects would be invalid).
    pub async fn run_effect(
        &self,
        effect: Effect,
        task_id: String,
        workspace_root: String,
        recipe: RecipeRef,
        tx: mpsc::Sender<RunnerMessage>,
    ) -> bool {
        match effect {
            // ── Observe ───────────────────────────────────────────────────
            Effect::Observe {
                artifact_id,
                request,
                ..
            } => {
                self.spawn_blocking_effect(
                    artifact_id.clone(),
                    "Observe",
                    {
                        let store_path = self.store_path.clone();
                        let artifact_id_inner = artifact_id.0.clone();
                        let task_id = task_id.clone();
                        let workspace_root = workspace_root.clone();
                        move || {
                            let rt = open_runtime(&store_path)?;
                            let now = time::OffsetDateTime::now_utc();
                            let _ = rt.start_task(
                                now,
                                task_id,
                                artifact_id_inner.clone(),
                                workspace_root,
                                request,
                            );
                            let _ = rt.observe_task(&artifact_id_inner, now);
                            Ok(MachineEvent::ObserveCompleted)
                        }
                    },
                    &tx,
                )
                .await;
            }

            // ── ProposeChange: run the agent, surface the plan, DON'T write ──
            Effect::ProposeChange { artifact_id } => {
                let provider = Arc::clone(&self.provider);
                let store_path = self.store_path.clone();
                let prior_context_files = self.prior_context_files.clone();
                let ignore_patterns = self.ignore_patterns.clone();
                let tx_clone = tx.clone();
                tokio::task::spawn_blocking(move || {
                    use shunt_core::FileOp;
                    use shunt_core::machine::{ArtifactSnapshot, FileDiff};
                    let mut rt = match open_runtime(&store_path) {
                        Ok(rt) => rt,
                        Err(err) => {
                            warn!(%err, "ProposeChange open_runtime");
                            runner_err(&tx_clone, "ProposeChange", err.to_string());
                            return;
                        }
                    };
                    rt.set_agent_observer(Arc::new(TxAgentObserver {
                        tx: tx_clone.clone(),
                    }));
                    let now = time::OffsetDateTime::now_utc();
                    if let Err(err) = rt.generate_proposed_change(
                        &task_id,
                        now,
                        provider.as_ref(),
                        &prior_context_files,
                        &ignore_patterns,
                    ) {
                        // Agent asked the developer a question → pause, don't fail.
                        if let RuntimeError::AgentNeedsClarification { question, context } = &err {
                            let _ = tx_clone.blocking_send(RunnerMessage::Event(
                                MachineEvent::AgentAsked {
                                    ambiguity_id: "agent-question".into(),
                                    question: question.clone(),
                                    context: context.clone(),
                                    options: vec![],
                                },
                            ));
                            return;
                        }
                        warn!(%err, "ProposeChange generate_proposed_change");
                        runner_err(&tx_clone, "ProposeChange", err.to_string());
                        return;
                    }

                    // Read the ChangeSet from the store and surface it — the
                    // client (TUI or headless driver) sees the plan before any write.
                    let (op_strs, op_paths, cmds_display, description, diffs) = {
                        let task = rt.store.get_task_run(&task_id).ok().flatten();
                        let change_set = task
                            .as_ref()
                            .and_then(|t| rt.load_active_recipe_run(t).ok())
                            .and_then(|r| r.change_set);
                        match change_set {
                            Some(cs) => {
                                let strs: Vec<String> = cs
                                    .ops
                                    .iter()
                                    .map(|op| match op {
                                        FileOp::Create { path, .. } => format!("create {path}"),
                                        FileOp::Edit { path, .. } => format!("edit {path}"),
                                        FileOp::Delete { path } => format!("delete {path}"),
                                    })
                                    .collect();
                                let paths: Vec<String> =
                                    cs.ops.iter().map(|op| op.path().to_string()).collect();
                                let cmds: Vec<String> =
                                    cs.commands.iter().map(|c| c.display()).collect();
                                let desc = format!("{} op(s)", strs.len());
                                let diffs: Vec<FileDiff> = cs
                                    .ops
                                    .iter()
                                    .map(|op| compute_file_diff(op, &workspace_root))
                                    .collect();
                                (strs, paths, cmds, desc, diffs)
                            }
                            None => (vec![], vec![], vec![], "no ops".into(), vec![]),
                        }
                    };
                    let command_count = cmds_display.len();
                    let _ = tx_clone.blocking_send(RunnerMessage::Notification(
                        Notification::ChangeProposed {
                            description,
                            ops: op_strs,
                            commands: cmds_display,
                            diffs,
                        },
                    ));

                    // Hand the decision (approve vs auto-commit) back to the machine.
                    let artifact = rt
                        .store
                        .get_understanding_artifact(&artifact_id.0)
                        .ok()
                        .flatten();
                    let confidence = artifact.as_ref().map(|a| a.confidence).unwrap_or(0.0);
                    let snapshot = ArtifactSnapshot {
                        interpreted_goal: artifact.map(|a| a.interpreted_goal).unwrap_or_default(),
                        confidence,
                        evidence_count: 0,
                        candidate_paths: op_paths.clone(),
                        open_ambiguity_count: 0,
                        open_risks: vec![],
                    };
                    let _ =
                        tx_clone.blocking_send(RunnerMessage::Event(MachineEvent::ProposalReady {
                            confidence,
                            op_count: op_paths.len(),
                            command_count,
                            snapshot,
                        }));
                });
            }

            Effect::ResumeAgent {
                artifact_id,
                question_id: _,
                answer,
            } => {
                let provider = Arc::clone(&self.provider);
                let store_path = self.store_path.clone();
                let ignore_patterns = self.ignore_patterns.clone();
                let workspace_root = workspace_root.clone();
                let tx_clone = tx.clone();
                tokio::task::spawn_blocking(move || {
                    use shunt_core::FileOp;
                    use shunt_core::machine::{ArtifactSnapshot, FileDiff};
                    let mut rt = match open_runtime(&store_path) {
                        Ok(rt) => rt,
                        Err(err) => {
                            warn!(%err, "ResumeAgent open_runtime");
                            runner_err(&tx_clone, "ResumeAgent", err.to_string());
                            return;
                        }
                    };
                    rt.set_agent_observer(Arc::new(TxAgentObserver {
                        tx: tx_clone.clone(),
                    }));
                    let now = time::OffsetDateTime::now_utc();
                    if let Err(err) =
                        rt.resume_agent(&task_id, now, provider.as_ref(), &answer, &ignore_patterns)
                    {
                        if let RuntimeError::AgentNeedsClarification { question, context } = &err {
                            let _ = tx_clone.blocking_send(RunnerMessage::Event(
                                MachineEvent::AgentAsked {
                                    ambiguity_id: "agent-question".into(),
                                    question: question.clone(),
                                    context: context.clone(),
                                    options: vec![],
                                },
                            ));
                            return;
                        }
                        warn!(%err, "ResumeAgent resume_agent");
                        runner_err(&tx_clone, "ResumeAgent", err.to_string());
                        return;
                    }

                    let (op_strs, op_paths, cmds_display, description, diffs) = {
                        let task = rt.store.get_task_run(&task_id).ok().flatten();
                        let change_set = task
                            .as_ref()
                            .and_then(|t| rt.load_active_recipe_run(t).ok())
                            .and_then(|r| r.change_set);
                        match change_set {
                            Some(cs) => {
                                let strs: Vec<String> = cs
                                    .ops
                                    .iter()
                                    .map(|op| match op {
                                        FileOp::Create { path, .. } => format!("create {path}"),
                                        FileOp::Edit { path, .. } => format!("edit {path}"),
                                        FileOp::Delete { path } => format!("delete {path}"),
                                    })
                                    .collect();
                                let paths: Vec<String> =
                                    cs.ops.iter().map(|op| op.path().to_string()).collect();
                                let cmds: Vec<String> =
                                    cs.commands.iter().map(|c| c.display()).collect();
                                let desc = format!("{} op(s)", strs.len());
                                let diffs: Vec<FileDiff> = cs
                                    .ops
                                    .iter()
                                    .map(|op| compute_file_diff(op, &workspace_root))
                                    .collect();
                                (strs, paths, cmds, desc, diffs)
                            }
                            None => (vec![], vec![], vec![], "no ops".into(), vec![]),
                        }
                    };
                    let command_count = cmds_display.len();
                    let _ = tx_clone.blocking_send(RunnerMessage::Notification(
                        Notification::ChangeProposed {
                            description,
                            ops: op_strs,
                            commands: cmds_display,
                            diffs,
                        },
                    ));

                    let artifact = rt
                        .store
                        .get_understanding_artifact(&artifact_id.0)
                        .ok()
                        .flatten();
                    let confidence = artifact.as_ref().map(|a| a.confidence).unwrap_or(0.0);
                    let snapshot = ArtifactSnapshot {
                        interpreted_goal: artifact.map(|a| a.interpreted_goal).unwrap_or_default(),
                        confidence,
                        evidence_count: 0,
                        candidate_paths: op_paths.clone(),
                        open_ambiguity_count: 0,
                        open_risks: vec![],
                    };
                    let _ =
                        tx_clone.blocking_send(RunnerMessage::Event(MachineEvent::ProposalReady {
                            confidence,
                            op_count: op_paths.len(),
                            command_count,
                            snapshot,
                        }));
                });
            }

            // ── CommitChange: apply the already-proposed change set to disk ──
            Effect::CommitChange { artifact_id: _ } => {
                let store_path = self.store_path.clone();
                let tx_clone = tx.clone();
                tokio::task::spawn_blocking(move || {
                    let rt = match open_runtime(&store_path) {
                        Ok(rt) => rt,
                        Err(err) => {
                            warn!(%err, "CommitChange open_runtime");
                            runner_err(&tx_clone, "CommitChange", err.to_string());
                            return;
                        }
                    };
                    let now = time::OffsetDateTime::now_utc();
                    // The change set was generated + stored by ProposeChange. Load
                    // its setup commands so we can safety-gate them before running.
                    let setup_commands = {
                        let task = rt.store.get_task_run(&task_id).ok().flatten();
                        task.as_ref()
                            .and_then(|t| rt.load_active_recipe_run(t).ok())
                            .and_then(|r| r.change_set)
                            .map(|cs| cs.commands)
                            .unwrap_or_default()
                    };

                    // Safety gate: classify proposed setup commands before running.
                    match safety::classify_all(&setup_commands) {
                        safety::CommandSafety::Blocked { reason } => {
                            runner_err(
                                &tx_clone,
                                "CommitChange",
                                format!("blocked command: {reason}"),
                            );
                            return;
                        }
                        safety::CommandSafety::Dangerous { reason } => {
                            let _ = tx_clone.blocking_send(RunnerMessage::Event(
                                MachineEvent::DangerCommandsDetected {
                                    commands: setup_commands,
                                    reason,
                                },
                            ));
                            return;
                        }
                        safety::CommandSafety::Safe => {}
                    }

                    // Stash current workspace state so the user can undo after apply.
                    let stashed = git_stash(&workspace_root);
                    let run = match rt.execute_apply(&task_id, now) {
                        Ok(r) => r,
                        Err(err) => {
                            warn!(%err, "CommitChange execute_apply");
                            runner_err(&tx_clone, "CommitChange", err.to_string());
                            return;
                        }
                    };

                    // Record the change in the ledger.
                    {
                        use shunt_core::ledger::{DiffSummary, WorkspaceRevision};
                        let files_changed: Vec<String> = run
                            .change_set
                            .as_ref()
                            .map(|cs| cs.ops.iter().map(|op| op.path().to_string()).collect())
                            .unwrap_or_default();
                        let op_count = files_changed.len();
                        append_ledger(
                            &store_path,
                            &task_id,
                            LedgerEntry::ToolObservation(ToolObservation::ChangeApplied {
                                revision: WorkspaceRevision {
                                    sequence: 0,
                                    content_hash: None,
                                },
                                diff: DiffSummary {
                                    files_changed,
                                    lines_added: 0,
                                    lines_removed: 0,
                                    description: format!("applied {op_count} file op(s)"),
                                },
                            }),
                        );
                    }

                    // Use change_set for path and setup_commands — legacy fields are always empty.
                    let path = run
                        .change_set
                        .as_ref()
                        .and_then(|cs| cs.ops.first())
                        .map(|op| op.path().to_string())
                        .unwrap_or_default();
                    if stashed {
                        let _ = tx_clone.blocking_send(RunnerMessage::Notification(
                            Notification::UndoAvailable,
                        ));
                    }
                    let _ =
                        tx_clone.blocking_send(RunnerMessage::Event(MachineEvent::PatchApplied {
                            path,
                            setup_commands,
                        }));
                });
            }

            // ── RunSetup ──────────────────────────────────────────────────
            Effect::RunSetup { artifact_id } => {
                let store_path = self.store_path.clone();
                let tx_clone = tx.clone();
                tokio::task::spawn_blocking(move || {
                    let rt = match open_runtime(&store_path) {
                        Ok(rt) => rt,
                        Err(err) => {
                            warn!(%err, "RunSetup open_runtime");
                            runner_err(&tx_clone, "RunSetup", err.to_string());
                            return;
                        }
                    };
                    let now = time::OffsetDateTime::now_utc();
                    let task = match rt.store.get_task_run(&task_id) {
                        Ok(Some(t)) => t,
                        Ok(None) => {
                            runner_err(&tx_clone, "RunSetup", format!("task not found: {task_id}"));
                            return;
                        }
                        Err(err) => {
                            runner_err(&tx_clone, "RunSetup", err.to_string());
                            return;
                        }
                    };
                    let run = match rt.load_active_recipe_run(&task) {
                        Ok(r) => r,
                        Err(err) => {
                            runner_err(&tx_clone, "RunSetup", err.to_string());
                            return;
                        }
                    };
                    // Emit per-command started notifications before running.
                    let commands = run
                        .change_set
                        .as_ref()
                        .map(|cs| cs.commands.clone())
                        .unwrap_or_default();
                    for cmd in &commands {
                        let _ = tx_clone.blocking_send(RunnerMessage::Notification(
                            Notification::SetupCommandStarted {
                                display: cmd.display(),
                            },
                        ));
                        debug!(display = %cmd.display(), "setup command starting");
                    }
                    let t0 = Instant::now();
                    let recipe_run = match rt.execute_setup(&task_id, now) {
                        Ok(r) => r,
                        Err(err) => {
                            warn!(%err, "RunSetup execute_setup");
                            runner_err(&tx_clone, "RunSetup", err.to_string());
                            return;
                        }
                    };
                    let elapsed_total = t0.elapsed().as_millis() as u64;
                    let outcomes = recipe_run.setup_outcomes.clone();
                    // Append one CommandFinished ledger entry per command.
                    let per_cmd_ms = if outcomes.is_empty() {
                        0
                    } else {
                        elapsed_total / outcomes.len() as u64
                    };
                    for outcome in &outcomes {
                        let status = if outcome.success {
                            CommandStatus::Completed
                        } else {
                            CommandStatus::Failed
                        };
                        append_ledger(
                            &store_path,
                            &task_id,
                            LedgerEntry::ToolObservation(ToolObservation::CommandFinished(
                                CommandObservation {
                                    command: outcome.spec.clone(),
                                    status,
                                    exit_code: Some(outcome.exit_code),
                                    stdout: BoundedOutput::from_string(
                                        &outcome.stdout,
                                        BoundedOutput::DEFAULT_LIMIT,
                                    ),
                                    stderr: BoundedOutput::from_string(
                                        &outcome.stderr,
                                        BoundedOutput::DEFAULT_LIMIT,
                                    ),
                                    parsed: vec![],
                                    workspace_delta: None,
                                    elapsed_ms: per_cmd_ms,
                                },
                            )),
                        );
                    }
                    let _ = tx_clone.blocking_send(RunnerMessage::Event(
                        MachineEvent::SetupCompleted { outcomes },
                    ));
                    let _ = artifact_id;
                });
            }

            // ── RecordAnswer ──────────────────────────────────────────────
            // Pure side-effect: write ambiguity answer to store.
            // The machine already transitioned before this effect fires,
            // so we must NOT send any MachineEvent on success.
            Effect::RecordAnswer {
                artifact_id,
                ambiguity_id,
                answer,
            } => {
                let store_path = self.store_path.clone();
                let aid = artifact_id.0.clone();
                let tx_err = tx.clone();
                tokio::task::spawn_blocking(move || {
                    let ambiguity_id_str = ambiguity_id.0.clone();
                    let answer_clone = answer.clone();
                    let rt = match open_runtime(&store_path) {
                        Ok(rt) => rt,
                        Err(err) => {
                            warn!(%err, "RecordAnswer open_runtime");
                            let _ = tx_err.blocking_send(RunnerMessage::Event(
                                MachineEvent::EffectError {
                                    effect: "RecordAnswer".into(),
                                    reason: err.to_string(),
                                },
                            ));
                            return;
                        }
                    };
                    let task_id_for_ledger = rt
                        .store
                        .get_understanding_artifact(&aid)
                        .ok()
                        .flatten()
                        .map(|a| a.task_id.0.clone())
                        .unwrap_or_default();
                    match rt.resolve_ambiguity(
                        &aid,
                        &ambiguity_id_str,
                        answer,
                        time::OffsetDateTime::now_utc(),
                    ) {
                        Ok(_) => {
                            if !task_id_for_ledger.is_empty() {
                                append_ledger(
                                    &store_path,
                                    &task_id_for_ledger,
                                    LedgerEntry::UserInput(UserObservation {
                                        kind: UserObservationKind::AmbiguityAnswer {
                                            ambiguity_id: ambiguity_id_str,
                                        },
                                        content: answer_clone,
                                    }),
                                );
                            }
                        }
                        Err(err) => {
                            warn!(%err, "RecordAnswer failed");
                            let _ = tx_err.blocking_send(RunnerMessage::Event(
                                MachineEvent::EffectError {
                                    effect: "RecordAnswer".into(),
                                    reason: err.to_string(),
                                },
                            ));
                        }
                    }
                });
            }

            // ── RecordApproval ────────────────────────────────────────────
            // Pure side-effect: persist approval and create recipe run.
            // Must NOT send a success event — machine already transitioned.
            Effect::RecordApproval {
                artifact_id,
                approved,
                note,
            } => {
                if approved {
                    let store_path = self.store_path.clone();
                    let aid = artifact_id.0.clone();
                    let recipe = recipe.clone();
                    match tokio::task::spawn_blocking(move || {
                        open_runtime(&store_path).and_then(|rt| {
                            let now = time::OffsetDateTime::now_utc();
                            let task_id_for_ledger = rt
                                .store
                                .get_understanding_artifact(&aid)
                                .ok()
                                .flatten()
                                .map(|a| a.task_id.0.clone())
                                .unwrap_or_default();
                            rt.approve_artifact(&aid, "session", note, now)?;
                            rt.start_execution(&task_id, now, recipe)?;
                            // Advance recipe run through the stub stages for bookkeeping.
                            let _ = rt.execute_inspect(&task_id, now);
                            let _ = rt.execute_propose(&task_id, now);
                            if !task_id_for_ledger.is_empty() {
                                append_ledger(
                                    &store_path,
                                    &task_id_for_ledger,
                                    LedgerEntry::UserInput(UserObservation {
                                        kind: UserObservationKind::Approval,
                                        content: "approved".into(),
                                    }),
                                );
                            }
                            Ok(())
                        })
                    })
                    .await
                    {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            warn!(%err, "RecordApproval failed");
                            let _ = tx
                                .send(RunnerMessage::Event(MachineEvent::EffectError {
                                    effect: "RecordApproval".into(),
                                    reason: err.to_string(),
                                }))
                                .await;
                            // Abort the effect batch — subsequent effects depend on the
                            // recipe run that was not created.
                            return false;
                        }
                        Err(join_err) => {
                            warn!(%join_err, "RecordApproval task panicked");
                            return false;
                        }
                    }
                }
            }

            // ── ApplyArtifactPatch ────────────────────────────────────────
            // Pure side-effect: update artifact fields.
            // Must NOT send a success event — machine already transitioned.
            Effect::ApplyArtifactPatch { artifact_id, patch } => {
                let store_path = self.store_path.clone();
                let aid = artifact_id.0.clone();
                let tx_err = tx.clone();
                tokio::task::spawn_blocking(move || {
                    let result = open_runtime(&store_path).and_then(|rt| {
                        let update = crate::ArtifactUpdate {
                            interpreted_goal: patch.interpreted_goal,
                            success_criteria: patch.success_criteria,
                            constraints: patch.constraints,
                            target_scope: patch.target_scope,
                            evidence: patch.evidence,
                            assumptions: patch.assumptions,
                            ambiguities: patch.ambiguities,
                            selected_recipe: patch.selected_recipe,
                            risks: patch.risks,
                            confidence: patch.confidence,
                            approval: patch.approval,
                            candidate_files: None,
                        };
                        rt.revise_artifact(&aid, time::OffsetDateTime::now_utc(), update)?;
                        Ok(())
                    });
                    if let Err(err) = result {
                        warn!(%err, "ApplyArtifactPatch failed");
                        let _ =
                            tx_err.blocking_send(RunnerMessage::Event(MachineEvent::EffectError {
                                effect: "ApplyArtifactPatch".into(),
                                reason: err.to_string(),
                            }));
                    }
                });
            }

            // ── RaiseFrontier ─────────────────────────────────────────────
            Effect::RaiseFrontier {
                artifact_id,
                reason,
            } => {
                self.spawn_blocking_effect(
                    artifact_id.clone(),
                    "RaiseFrontier",
                    {
                        let store_path = self.store_path.clone();
                        let aid = artifact_id.0.clone();
                        let task_id = task_id.clone();
                        move || {
                            let rt = open_runtime(&store_path)?;
                            let now = time::OffsetDateTime::now_utc();
                            let task = rt
                                .store
                                .get_task_run(&task_id)?
                                .ok_or_else(|| RuntimeError::TaskNotFound(task_id.clone()))?;
                            let artifact = rt
                                .store
                                .get_understanding_artifact(&aid)?
                                .ok_or_else(|| RuntimeError::TaskNotFound(aid.clone()))?;
                            let case = rt.record_frontier_case(
                                now,
                                format!("frontier-{}-{}", task_id, now.unix_timestamp_nanos()),
                                &task,
                                &artifact,
                                reason,
                                "frontier raised by state machine",
                                vec![],
                            )?;
                            Ok(MachineEvent::FrontierCreated {
                                case_id: FrontierCaseId(case.id.0),
                            })
                        }
                    },
                    &tx,
                )
                .await;
            }

            // ── Persist ───────────────────────────────────────────────────
            Effect::Persist => {
                // State persistence is done inline by each blocking effect.
                // This is a no-op at the runner level.
                debug!("Persist effect — no-op at runner level");
            }

            // ── Notify ────────────────────────────────────────────────────
            Effect::Notify(notification) => {
                debug!(?notification, "Notify effect");
                let _ = tx.send(RunnerMessage::Notification(notification)).await;
            }
        }
        true
    }

    async fn spawn_blocking_effect<F>(
        &self,
        _artifact_id: ArtifactId,
        effect_name: &'static str,
        f: F,
        tx: &mpsc::Sender<RunnerMessage>,
    ) where
        F: FnOnce() -> Result<MachineEvent, RuntimeError> + Send + 'static,
    {
        let tx = tx.clone();
        tokio::task::spawn_blocking(move || match f() {
            Ok(event) => {
                let _ = tx.blocking_send(RunnerMessage::Event(event));
            }
            Err(err) => {
                warn!(%err, effect_name, "effect runner error");
                let reason = err.to_string();
                let _ = tx.blocking_send(RunnerMessage::Notification(Notification::Note {
                    text: format!("✗ {effect_name} failed: {reason}"),
                }));
                let _ = tx.blocking_send(RunnerMessage::Event(MachineEvent::EffectError {
                    effect: effect_name.into(),
                    reason,
                }));
            }
        });
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn open_runtime(store_path: &str) -> Result<TaskRuntime, RuntimeError> {
    let store = shunt_store::SqliteStore::open(store_path).map_err(RuntimeError::Store)?;
    Ok(TaskRuntime::new(store))
}

/// Stash the workspace with git before applying changes (enables undo).
/// Returns true if a stash entry was created.
fn git_stash(workspace_root: &str) -> bool {
    // Only stash if the directory is a git repo.
    let is_git = std::process::Command::new("git")
        .args(["rev-parse", "--is-inside-work-tree"])
        .current_dir(workspace_root)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !is_git {
        return false;
    }

    // Check for any changes to stash (clean tree = nothing to stash).
    let has_changes = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(workspace_root)
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    if !has_changes {
        return false;
    }

    std::process::Command::new("git")
        .args([
            "stash",
            "push",
            "--include-untracked",
            "-m",
            "shunt: before apply",
        ])
        .current_dir(workspace_root)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Pop the most recent frame stash (undo last apply).
pub fn git_stash_pop(workspace_root: &str) -> Result<String, String> {
    let out = std::process::Command::new("git")
        .args(["stash", "pop"])
        .current_dir(workspace_root)
        .output()
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Send a plain-text progress note to the TUI activity log.
fn runner_note(tx: &mpsc::Sender<RunnerMessage>, text: impl Into<String>) {
    let _ = tx.blocking_send(RunnerMessage::Notification(Notification::Note {
        text: text.into(),
    }));
}

/// Emit a descriptive error note then signal EffectError to the state machine.
fn runner_err(tx: &mpsc::Sender<RunnerMessage>, effect: &str, reason: String) {
    runner_note(tx, format!("✗ {effect} failed: {reason}"));
    let _ = tx.blocking_send(RunnerMessage::Event(MachineEvent::EffectError {
        effect: effect.into(),
        reason,
    }));
}

/// Compute a `FileDiff` for one `FileOp`.
///
/// For `Edit`: diffs the search block (removed) against the replacement (added)
///   with a few lines of surrounding file context.
/// For `Create`: shows the new file contents as added lines (first 40 lines).
/// For `Delete`: one removed line for the path.
fn compute_file_diff(
    op: &shunt_core::FileOp,
    workspace_root: &str,
) -> shunt_core::machine::FileDiff {
    use shunt_core::{
        FileOp,
        machine::{DiffLine, FileDiff},
    };
    use std::path::Path;

    match op {
        FileOp::Edit {
            path,
            search,
            replacement,
        } => {
            let mut lines: Vec<DiffLine> = vec![];

            // Try to find surrounding context in the existing file.
            let abs = Path::new(workspace_root).join(path);
            let file_content = std::fs::read_to_string(&abs).unwrap_or_default();
            let ctx_before = context_lines_before(&file_content, search, 3);
            let ctx_after = context_lines_after(&file_content, search, 3);

            for l in &ctx_before {
                lines.push(DiffLine::Context(l.clone()));
            }
            for l in search.lines() {
                lines.push(DiffLine::Removed(l.to_string()));
            }
            for l in replacement.lines() {
                lines.push(DiffLine::Added(l.to_string()));
            }
            for l in &ctx_after {
                lines.push(DiffLine::Context(l.clone()));
            }

            FileDiff {
                path: path.clone(),
                lines,
            }
        }

        FileOp::Create { path, contents } => {
            // If file exists, show a full unified diff; otherwise, all lines as added.
            let abs = Path::new(workspace_root).join(path);
            let old = std::fs::read_to_string(&abs).unwrap_or_default();
            let lines = if old.is_empty() {
                contents
                    .lines()
                    .take(40)
                    .map(|l| DiffLine::Added(l.to_string()))
                    .collect()
            } else {
                unified_diff(&old, contents, 3)
            };
            FileDiff {
                path: path.clone(),
                lines,
            }
        }

        FileOp::Delete { path } => FileDiff {
            path: path.clone(),
            lines: vec![DiffLine::Removed(format!("(deleted) {path}"))],
        },
    }
}

/// Extract up to `n` lines immediately before the first occurrence of `search` in `text`.
fn context_lines_before(text: &str, search: &str, n: usize) -> Vec<String> {
    let pos = match text.find(search) {
        Some(p) => p,
        None => return vec![],
    };
    let before = &text[..pos];
    let lines: Vec<&str> = before.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].iter().map(|l| l.to_string()).collect()
}

/// Extract up to `n` lines immediately after the first occurrence of `search` in `text`.
fn context_lines_after(text: &str, search: &str, n: usize) -> Vec<String> {
    let pos = match text.find(search) {
        Some(p) => p,
        None => return vec![],
    };
    let after_start = pos + search.len();
    let after = &text[after_start..];
    after.lines().take(n).map(|l| l.to_string()).collect()
}

/// Produce a simple unified diff between `old` and `new` with `ctx` context lines.
fn unified_diff(old: &str, new: &str, ctx: usize) -> Vec<shunt_core::machine::DiffLine> {
    use shunt_core::machine::DiffLine;

    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();

    // Simple LCS-based diff using dynamic programming on line hashes.
    let m = old_lines.len();
    let n = new_lines.len();

    // Build LCS table.
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in (0..m).rev() {
        for j in (0..n).rev() {
            dp[i][j] = if old_lines[i] == new_lines[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }

    // Produce edit ops: `+` = added, `-` = removed, ` ` = common.
    #[derive(Debug, Clone)]
    enum EditOp {
        Keep(String),
        Remove(String),
        Add(String),
    }

    let mut ops: Vec<EditOp> = vec![];
    let (mut i, mut j) = (0, 0);
    while i < m || j < n {
        if i < m && j < n && old_lines[i] == new_lines[j] {
            ops.push(EditOp::Keep(old_lines[i].to_string()));
            i += 1;
            j += 1;
        } else if j < n && (i >= m || dp[i][j + 1] >= dp[i + 1][j]) {
            ops.push(EditOp::Add(new_lines[j].to_string()));
            j += 1;
        } else {
            ops.push(EditOp::Remove(old_lines[i].to_string()));
            i += 1;
        }
    }

    // Find ranges of changed ops and emit with context.
    let changed: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| !matches!(op, EditOp::Keep(_)))
        .map(|(i, _)| i)
        .collect();

    if changed.is_empty() {
        return vec![];
    }

    // Build included-op mask (changed + ctx lines around each change).
    let n_ops = ops.len();
    let mut include = vec![false; n_ops];
    for &ci in &changed {
        let lo = ci.saturating_sub(ctx);
        let hi = (ci + ctx + 1).min(n_ops);
        for item in include.iter_mut().take(hi).skip(lo) {
            *item = true;
        }
    }

    let mut result: Vec<DiffLine> = vec![];
    let mut last_included = None::<usize>;
    for (k, op) in ops.iter().enumerate() {
        if !include[k] {
            continue;
        }
        if let Some(prev) = last_included
            && k > prev + 1
        {
            result.push(DiffLine::Context("…".to_string()));
        }
        match op {
            EditOp::Keep(l) => result.push(DiffLine::Context(l.clone())),
            EditOp::Remove(l) => result.push(DiffLine::Removed(l.clone())),
            EditOp::Add(l) => result.push(DiffLine::Added(l.clone())),
        }
        last_included = Some(k);
    }
    result
}

/// Append one ledger entry for `task_id`.  Non-fatal: logs a warning on error.
fn append_ledger(store_path: &str, task_id: &str, entry: LedgerEntry) {
    match shunt_store::SqliteStore::open(store_path) {
        Ok(store) => {
            if let Err(e) = store.append_ledger_entry(task_id, entry) {
                warn!(task_id, %e, "append_ledger failed");
            }
        }
        Err(e) => warn!(task_id, %e, "append_ledger open failed"),
    }
}
