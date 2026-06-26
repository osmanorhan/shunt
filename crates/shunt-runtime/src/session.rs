//! Session actor: owns `TaskState` + drives `TaskMachine::transition` in a
//! dedicated `std::thread` (not a tokio task, because `rusqlite::Connection`
//! is not `Send`).
//!
//! External callers communicate through `SessionHandle`:
//!   - send `Command`s via an `mpsc::Sender`
//!   - receive `Notification`s via a `broadcast::Receiver`
//!   - observe the current `TaskState` via a `watch::Receiver`

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;

use tokio::runtime::Handle;
use tokio::sync::{broadcast, mpsc, watch};
use tracing::{debug, info};

use shunt_core::machine::{AutonomyPolicy, Command, Effect, MachineEvent, Notification, TaskState};
use shunt_core::{ArtifactId, RecipeRef};
use shunt_infer::{ModelCallEvent, ToolProvider};

use crate::{
    machine::{SessionContext, TaskMachine},
    runner::{EffectRunner, RunnerMessage},
};

// ── SessionHandle ─────────────────────────────────────────────────────────────

/// The public handle to a running session actor.
#[derive(Clone)]
pub struct SessionHandle {
    /// Send commands into the running session.
    pub commands: mpsc::Sender<Command>,
    /// Subscribe to notifications from the running session.
    pub notifications: broadcast::Sender<Notification>,
    /// Observe the current `TaskState` (non-blocking, always has the latest).
    pub state: watch::Receiver<TaskState>,
    /// The task ID for this session (for store lookups).
    pub task_id: String,
    /// The artifact ID for this session (for store lookups).
    pub artifact_id: ArtifactId,
}

impl SessionHandle {
    /// Convenience: send a command and forget.
    pub async fn send(&self, cmd: Command) {
        let _ = self.commands.send(cmd).await;
    }

    /// Subscribe to notifications.
    pub fn subscribe(&self) -> broadcast::Receiver<Notification> {
        self.notifications.subscribe()
    }

    /// Current state snapshot (non-blocking).
    pub fn current_state(&self) -> TaskState {
        self.state.borrow().clone()
    }
}

// ── spawn_session ─────────────────────────────────────────────────────────────

/// Spawn a session actor for a task.
///
/// The actor runs in a `std::thread`.  All IO effects are dispatched to a
/// Tokio `spawn_blocking` pool via the given `Handle`.
///
/// Returns a `SessionHandle` for the caller to interact with.
#[allow(clippy::too_many_arguments)]
pub fn spawn_session<P>(
    task_id: String,
    artifact_id: ArtifactId,
    workspace_root: String,
    recipe: RecipeRef,
    store_path: String,
    provider: P,
    policy: AutonomyPolicy,
    tokio_handle: Handle,
    prior_context_files: Vec<String>,
    ignore_patterns: Vec<String>,
) -> SessionHandle
where
    P: ToolProvider + Clone + Send + Sync + 'static,
{
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>(64);
    let (notif_tx, _) = broadcast::channel::<Notification>(128);
    let (state_tx, state_rx) = watch::channel(TaskState::Running);

    let ctx = SessionContext {
        task_id: task_id.clone(),
        artifact_id: artifact_id.clone(),
        workspace_root: workspace_root.clone(),
        recipe: recipe.clone(),
    };
    let notif_tx_clone = notif_tx.clone();
    let call_notifier = notif_tx.clone();
    let provider = provider.with_call_observer(Arc::new(move |event| {
        let notification = match event {
            ModelCallEvent::Started {
                call_id,
                tool,
                model,
                mode,
            } => Notification::InferenceCallStarted {
                call_id,
                tool,
                model,
                mode,
            },
            ModelCallEvent::Finished {
                call_id,
                tool,
                elapsed_ms,
                outcome,
            } => Notification::InferenceCallFinished {
                call_id,
                tool,
                elapsed_ms,
                outcome,
            },
            ModelCallEvent::TokenChunk {
                call_id,
                text,
                is_thinking,
            } => Notification::InferenceToken {
                call_id,
                text,
                is_thinking,
            },
        };
        let _ = call_notifier.send(notification);
    }));
    let runner = Arc::new(
        EffectRunner::new(store_path, provider)
            .with_prior_context(prior_context_files)
            .with_ignore_patterns(ignore_patterns),
    );

    thread::Builder::new()
        .name(format!("session-{task_id}"))
        .spawn(move || {
            actor_loop(
                ctx,
                policy,
                cmd_rx,
                notif_tx_clone,
                state_tx,
                runner,
                tokio_handle,
            );
        })
        .expect("failed to spawn session thread");

    SessionHandle {
        commands: cmd_tx,
        notifications: notif_tx,
        state: state_rx,
        task_id,
        artifact_id,
    }
}

// ── actor_loop ────────────────────────────────────────────────────────────────

fn actor_loop<P>(
    ctx: SessionContext,
    policy: AutonomyPolicy,
    mut cmd_rx: mpsc::Receiver<Command>,
    notif_tx: broadcast::Sender<Notification>,
    state_tx: watch::Sender<TaskState>,
    runner: Arc<EffectRunner<P>>,
    tokio_handle: Handle,
) where
    P: ToolProvider + Clone + Send + Sync + 'static,
{
    let (runner_tx, mut runner_rx) = mpsc::channel::<RunnerMessage>(128);

    let mut state = TaskState::Running;
    let mut pending_events: VecDeque<MachineEvent> = VecDeque::new();

    info!(task_id = %ctx.task_id, "session actor started");

    loop {
        // 1. Drain any internally queued events before blocking on channels.
        while let Some(event) = pending_events.pop_front() {
            debug!(?event, ?state, "processing queued event");
            let (next, effects) = TaskMachine::transition(state.clone(), event, &ctx, &policy);
            info!(
                task_id = %ctx.task_id,
                from = ?state,
                to   = ?next,
                effect_count = effects.len(),
                "state transition"
            );
            state = next.clone();
            let _ = state_tx.send(state.clone());

            if state.is_terminal() {
                info!(task_id = %ctx.task_id, "session reached terminal state — stopping");
                return;
            }

            dispatch_effects(
                effects,
                &ctx,
                Arc::clone(&runner),
                runner_tx.clone(),
                &tokio_handle,
            );
        }

        // 2. Wait for the next message from either the runner or a client command.
        let msg = tokio_handle.block_on(async {
            tokio::select! {
                Some(runner_msg) = runner_rx.recv() => ActorMsg::Runner(runner_msg),
                Some(cmd) = cmd_rx.recv() => ActorMsg::Command(cmd),
                else => ActorMsg::Closed,
            }
        });

        match msg {
            ActorMsg::Runner(RunnerMessage::Event(event)) => {
                debug!(?event, "runner event received");
                pending_events.push_back(event);
            }
            ActorMsg::Runner(RunnerMessage::Notification(notif)) => {
                debug!(?notif, "runner notification");
                let _ = notif_tx.send(notif);
            }
            ActorMsg::Command(cmd) => {
                debug!(?cmd, "command received");
                pending_events.push_back(MachineEvent::UserCommand(cmd));
            }
            ActorMsg::Closed => {
                info!(task_id = %ctx.task_id, "all senders dropped — session ending");
                return;
            }
        }
    }
}

// ── dispatch_effects ──────────────────────────────────────────────────────────

fn dispatch_effects<P>(
    effects: Vec<Effect>,
    ctx: &SessionContext,
    runner: Arc<EffectRunner<P>>,
    runner_tx: mpsc::Sender<RunnerMessage>,
    tokio_handle: &Handle,
) where
    P: ToolProvider + Clone + Send + Sync + 'static,
{
    let task_id = ctx.task_id.clone();
    let workspace_root = ctx.workspace_root.clone();
    let recipe = ctx.recipe.clone();
    tokio_handle.spawn(async move {
        for effect in effects {
            let cont = runner
                .run_effect(
                    effect,
                    task_id.clone(),
                    workspace_root.clone(),
                    recipe.clone(),
                    runner_tx.clone(),
                )
                .await;
            if !cont {
                break;
            }
        }
    });
}

// ── ActorMsg ──────────────────────────────────────────────────────────────────

enum ActorMsg {
    Runner(RunnerMessage),
    Command(Command),
    Closed,
}
