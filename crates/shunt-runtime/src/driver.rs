//! Programmatic session driver.
//!
//! `drive_session` runs a [`spawn_session`] task to a terminal state, using a
//! [`Responder`] to answer every `WaitingForUser` pause. It is the non-human
//! client of the exact same core the TUI drives: the TUI turns keystrokes into
//! `Command`s; a `Responder` turns pauses into `Command`s programmatically.
//!
//! Consumers:
//!   * `agent --once` — headless, [`AutoResponder`] + [`AutonomyPolicy::headless`].
//!   * the bench — in-process, [`ScriptedResponder`] scripting developer turns.

use std::collections::VecDeque;

use tokio::runtime::Handle;
use tokio::sync::broadcast::error::RecvError;

use shunt_core::machine::{AutonomyPolicy, Command, Notification, TaskState, UserRequest};
use shunt_core::{AmbiguityId, ArtifactId, RecipeRef};
use shunt_infer::ToolProvider;

use crate::session::spawn_session;

// ── Responder ───────────────────────────────────────────────────────────────

/// Decides how to answer a session pause. The TUI's human is the manual version;
/// these are the programmatic ones.
pub trait Responder: Send {
    fn respond(&mut self, request: &UserRequest) -> Command;
}

/// Fully autonomous: approve plans, answer agent questions with a fixed nudge,
/// reject dangerous commands. Bounded so a model that keeps asking can't loop
/// forever — after `max_answers` clarifications it cancels.
pub struct AutoResponder {
    answer: String,
    max_answers: usize,
    answers_given: usize,
    approve_dangerous: bool,
}

impl Default for AutoResponder {
    fn default() -> Self {
        Self {
            answer: "Proceed with your best judgment.".to_string(),
            max_answers: 3,
            answers_given: 0,
            approve_dangerous: false,
        }
    }
}

impl Responder for AutoResponder {
    fn respond(&mut self, request: &UserRequest) -> Command {
        match request {
            UserRequest::Clarification { open, .. } => {
                if self.answers_given >= self.max_answers {
                    return Command::Cancel;
                }
                self.answers_given += 1;
                let id = open
                    .first()
                    .map(|a| a.id.clone())
                    .unwrap_or_else(|| AmbiguityId("agent-question".into()));
                Command::Answer {
                    ambiguity_id: id,
                    answer: self.answer.clone(),
                }
            }
            UserRequest::Approval { .. } => Command::Approve,
            UserRequest::DangerousCommands { .. } => {
                if self.approve_dangerous {
                    Command::ApproveDangerousCommands
                } else {
                    Command::RejectDangerousCommands
                }
            }
        }
    }
}

/// Scripts the developer's side of a session: pops answers in order for
/// clarifications, and applies a fixed decision for the approval / dangerous
/// gates. For tests that exercise realistic multi-turn flows.
pub struct ScriptedResponder {
    answers: VecDeque<String>,
    approve_plan: bool,
    approve_dangerous: bool,
    /// Fallback used when the script runs dry but the agent keeps asking.
    fallback: String,
    exhausted_cancels: bool,
}

impl ScriptedResponder {
    /// Build from an ordered list of clarification answers. Approves plans,
    /// rejects dangerous commands by default.
    pub fn new(answers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            answers: answers.into_iter().map(Into::into).collect(),
            approve_plan: true,
            approve_dangerous: false,
            fallback: "Proceed with your best judgment.".to_string(),
            exhausted_cancels: true,
        }
    }

    pub fn reject_plan(mut self) -> Self {
        self.approve_plan = false;
        self
    }
}

impl Responder for ScriptedResponder {
    fn respond(&mut self, request: &UserRequest) -> Command {
        match request {
            UserRequest::Clarification { open, .. } => {
                let id = open
                    .first()
                    .map(|a| a.id.clone())
                    .unwrap_or_else(|| AmbiguityId("agent-question".into()));
                match self.answers.pop_front() {
                    Some(answer) => Command::Answer {
                        ambiguity_id: id,
                        answer,
                    },
                    None if self.exhausted_cancels => Command::Cancel,
                    None => Command::Answer {
                        ambiguity_id: id,
                        answer: self.fallback.clone(),
                    },
                }
            }
            UserRequest::Approval { .. } => {
                if self.approve_plan {
                    Command::Approve
                } else {
                    Command::Reject
                }
            }
            UserRequest::DangerousCommands { .. } => {
                if self.approve_dangerous {
                    Command::ApproveDangerousCommands
                } else {
                    Command::RejectDangerousCommands
                }
            }
        }
    }
}

// ── drive_session ─────────────────────────────────────────────────────────────

/// Outcome of driving a session to terminal: the final state plus the full,
/// ordered notification timeline (for logging / assertions).
pub struct DriveResult {
    pub final_state: TaskState,
    pub notifications: Vec<Notification>,
}

impl DriveResult {
    pub fn completed(&self) -> bool {
        matches!(self.final_state, TaskState::Completed)
    }

    /// Number of clarification pauses surfaced during the run.
    pub fn clarification_count(&self) -> usize {
        self.notifications
            .iter()
            .filter(|n| matches!(n, Notification::ClarificationNeeded { .. }))
            .count()
    }
}

/// Spawn a session and drive it to a terminal state, answering pauses via
/// `responder`. Blocks the calling thread (using `tokio_handle`) until the task
/// completes, stops, or all channels close.
#[allow(clippy::too_many_arguments)]
pub fn drive_session<P>(
    task_id: String,
    artifact_id: ArtifactId,
    workspace_root: String,
    recipe: RecipeRef,
    store_path: String,
    provider: P,
    policy: AutonomyPolicy,
    tokio_handle: Handle,
    request: String,
    responder: &mut dyn Responder,
) -> DriveResult
where
    P: ToolProvider + Clone + Send + Sync + 'static,
{
    let handle = spawn_session(
        task_id,
        artifact_id,
        workspace_root,
        recipe,
        store_path,
        provider,
        policy,
        tokio_handle.clone(),
        vec![],
        vec![],
    );
    let mut notif_rx = handle.subscribe();
    let mut state_rx = handle.state.clone();

    tokio_handle.block_on(async move {
        let mut notifications = Vec::new();
        let _ = handle.commands.send(Command::Submit { request }).await;

        let final_state = loop {
            tokio::select! {
                recv = notif_rx.recv() => match recv {
                    Ok(n) => notifications.push(n),
                    Err(RecvError::Lagged(_)) => continue,
                    Err(RecvError::Closed) => break state_rx.borrow().clone(),
                },
                changed = state_rx.changed() => {
                    if changed.is_err() {
                        break state_rx.borrow().clone();
                    }
                    let state = state_rx.borrow().clone();
                    match state {
                        TaskState::WaitingForUser { request } => {
                            let cmd = responder.respond(&request);
                            let _ = handle.commands.send(cmd).await;
                        }
                        s if s.is_terminal() => break s,
                        _ => {}
                    }
                }
            }
        };

        DriveResult {
            final_state,
            notifications,
        }
    })
}
