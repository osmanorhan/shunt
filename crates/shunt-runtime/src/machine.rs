//! Pure state machine for the Frame task lifecycle — M5.3.
//!
//! `TaskMachine::transition` is the only place control flow decisions are made.
//! It takes the current state + an incoming event + the autonomy policy and
//! returns the next state plus a list of effects to execute.  It is free of IO,
//! timestamps, and side effects — making it exhaustively unit-testable.
//!
//! States: Running | WaitingForUser | Completed | Stopped
//! All cognitive activity labels are Note notifications, not state variants.

use shunt_core::machine::{
    ArtifactPatch, AutonomyPolicy, Command, Effect, GateKind, MachineEvent, Notification,
    PendingAmbiguity, StopReason, TaskState, UserRequest,
};
use shunt_core::{ArtifactId, RecipeRef, TaskPhase};

// ── SessionContext ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SessionContext {
    pub task_id: String,
    pub artifact_id: ArtifactId,
    pub workspace_root: String,
    pub recipe: RecipeRef,
}

// ── TaskMachine ───────────────────────────────────────────────────────────────

pub struct TaskMachine;

impl TaskMachine {
    /// Pure state transition.  Returns `(next_state, effects)`.
    pub fn transition(
        state: TaskState,
        event: MachineEvent,
        ctx: &SessionContext,
        policy: &AutonomyPolicy,
    ) -> (TaskState, Vec<Effect>) {
        use MachineEvent::*;
        use TaskState::*;

        let aid = ctx.artifact_id.clone();

        match (state, event) {
            // ── Initial submit ────────────────────────────────────────────
            (Running, UserCommand(Command::Submit { request })) => (
                Running,
                vec![
                    Effect::Notify(Notification::TaskStarted),
                    Effect::Notify(Notification::ObserveStarted),
                    Effect::Observe {
                        workspace_root: ctx.workspace_root.clone(),
                        artifact_id: aid.clone(),
                        request,
                    },
                    Effect::Persist,
                ],
            ),

            // ── Observe → ProposeChange (agent plans, no write yet) ───────
            (Running, ObserveCompleted) => (
                Running,
                vec![
                    Effect::Notify(Notification::ObserveFinished {
                        summary: "workspace observed".into(),
                    }),
                    Effect::Notify(Notification::Note {
                        text: "→ agent: planning changes…".into(),
                    }),
                    Effect::ProposeChange {
                        artifact_id: aid.clone(),
                    },
                    Effect::Persist,
                ],
            ),

            // ── Proposal ready → approval gate (per policy) ───────────────
            (Running, ProposalReady { confidence, op_count, snapshot }) => {
                if op_count == 0 {
                    // Nothing to apply — the agent decided no change was needed.
                    (
                        Completed,
                        vec![
                            Effect::Notify(Notification::RunCompleted {
                                summary: "no changes were needed".into(),
                            }),
                            Effect::Persist,
                        ],
                    )
                } else if policy.should_ask(GateKind::Approval, confidence) {
                    (
                        WaitingForUser {
                            request: UserRequest::Approval {
                                candidate_count: op_count,
                                snapshot,
                            },
                        },
                        vec![
                            Effect::Notify(Notification::Note {
                                text: format!("awaiting approval for {op_count} change(s)"),
                            }),
                            Effect::Persist,
                        ],
                    )
                } else {
                    (
                        Running,
                        vec![
                            Effect::CommitChange {
                                artifact_id: aid.clone(),
                            },
                            Effect::Persist,
                        ],
                    )
                }
            }

            // ── Agent asked the developer a question → pause ──────────────
            (Running, AgentAsked { ambiguity_id, question, options }) => (
                WaitingForUser {
                    request: UserRequest::Clarification {
                        open: vec![PendingAmbiguity {
                            id: shunt_core::AmbiguityId(ambiguity_id.clone()),
                            question: question.clone(),
                            options: options.clone(),
                        }],
                        confidence: 0.0,
                    },
                },
                vec![
                    Effect::Notify(Notification::ClarificationNeeded {
                        ambiguity_id,
                        question,
                        options,
                        confidence: 0.0,
                    }),
                    Effect::Persist,
                ],
            ),

            // ── Answer a clarification question ───────────────────────────
            (
                WaitingForUser {
                    request: UserRequest::Clarification { open, .. },
                },
                UserCommand(Command::Answer {
                    ambiguity_id,
                    answer,
                }),
            ) => {
                let remaining: Vec<PendingAmbiguity> =
                    open.into_iter().filter(|a| a.id != ambiguity_id).collect();
                let mut effects = vec![Effect::RecordAnswer {
                    artifact_id: aid.clone(),
                    ambiguity_id,
                    answer,
                }];
                if remaining.is_empty() {
                    effects.extend([
                        Effect::Notify(Notification::Note {
                            text: "→ agent: planning changes…".into(),
                        }),
                        Effect::ProposeChange {
                            artifact_id: aid.clone(),
                        },
                        Effect::Persist,
                    ]);
                    (Running, effects)
                } else {
                    let next = &remaining[0];
                    effects.push(Effect::Notify(Notification::ClarificationNeeded {
                        ambiguity_id: next.id.0.clone(),
                        question: next.question.clone(),
                        options: next.options.clone(),
                        confidence: 0.0,
                    }));
                    effects.push(Effect::Persist);
                    (
                        WaitingForUser {
                            request: UserRequest::Clarification {
                                open: remaining,
                                confidence: 0.0,
                            },
                        },
                        effects,
                    )
                }
            }

            // ── User approves plan ────────────────────────────────────────
            (
                WaitingForUser {
                    request: UserRequest::Approval { .. },
                },
                UserCommand(Command::Approve),
            ) => (
                Running,
                vec![
                    Effect::RecordApproval {
                        artifact_id: aid.clone(),
                        approved: true,
                        note: None,
                    },
                    Effect::Notify(Notification::PhaseEntered {
                        phase: TaskPhase::Execute,
                        summary: "approved — applying".into(),
                    }),
                    Effect::CommitChange {
                        artifact_id: aid.clone(),
                    },
                    Effect::Persist,
                ],
            ),

            // ── User rejects plan → re-run agent ─────────────────────────
            (
                WaitingForUser {
                    request: UserRequest::Approval { .. },
                },
                UserCommand(Command::Reject),
            ) => (
                Running,
                vec![
                    Effect::Notify(Notification::Note {
                        text: "re-running agent…".into(),
                    }),
                    Effect::ProposeChange {
                        artifact_id: aid.clone(),
                    },
                    Effect::Persist,
                ],
            ),

            // ── Patch applied: no setup commands → done ───────────────────
            (Running, PatchApplied { setup_commands, .. }) if setup_commands.is_empty() => (
                Completed,
                vec![
                    Effect::Notify(Notification::RunCompleted {
                        summary: "task completed".into(),
                    }),
                    Effect::Persist,
                ],
            ),

            // ── Patch applied: setup commands → run them ──────────────────
            (Running, PatchApplied { setup_commands, .. }) => (
                Running,
                vec![
                    Effect::Notify(Notification::SetupStarted {
                        count: setup_commands.len(),
                    }),
                    Effect::RunSetup {
                        artifact_id: aid.clone(),
                    },
                    Effect::Persist,
                ],
            ),

            // ── Dangerous commands detected ───────────────────────────────
            (Running, DangerCommandsDetected { commands, reason }) => (
                WaitingForUser {
                    request: UserRequest::DangerousCommands {
                        commands: commands.clone(),
                        reason: reason.clone(),
                    },
                },
                vec![
                    Effect::Notify(Notification::DangerousCommandsProposed { commands, reason }),
                    Effect::Persist,
                ],
            ),

            // ── User approves dangerous commands ──────────────────────────
            (
                WaitingForUser {
                    request: UserRequest::DangerousCommands { commands, .. },
                },
                UserCommand(Command::ApproveDangerousCommands),
            ) => (
                Running,
                vec![
                    Effect::Notify(Notification::SetupStarted {
                        count: commands.len(),
                    }),
                    Effect::RunSetup {
                        artifact_id: aid.clone(),
                    },
                    Effect::Persist,
                ],
            ),

            // ── User rejects dangerous commands → complete without them ───
            (
                WaitingForUser {
                    request: UserRequest::DangerousCommands { .. },
                },
                UserCommand(Command::RejectDangerousCommands),
            ) => (
                Completed,
                vec![
                    Effect::Notify(Notification::RunCompleted {
                        summary: "task completed, setup commands skipped".into(),
                    }),
                    Effect::Persist,
                ],
            ),

            // ── Setup done → complete ─────────────────────────────────────
            (Running, SetupCompleted { outcomes }) => {
                let failed = outcomes.iter().filter(|o| !o.success).count();
                let summary = if failed == 0 {
                    format!("{} command(s) succeeded", outcomes.len())
                } else {
                    format!("{}/{} command(s) failed", failed, outcomes.len())
                };
                (
                    Completed,
                    vec![
                        Effect::Notify(Notification::SetupFinished { summary }),
                        Effect::Notify(Notification::RunCompleted {
                            summary: "task completed".into(),
                        }),
                        Effect::Persist,
                    ],
                )
            }

            // ── Steer: revise goal mid-run ────────────────────────────────
            (Running, UserCommand(Command::Steer { message })) => (
                Running,
                vec![
                    Effect::ApplyArtifactPatch {
                        artifact_id: aid.clone(),
                        patch: Box::new(ArtifactPatch {
                            interpreted_goal: Some(message),
                            ..Default::default()
                        }),
                    },
                    Effect::ProposeChange {
                        artifact_id: aid.clone(),
                    },
                    Effect::Persist,
                ],
            ),

            // ── Cancel from any non-terminal state ────────────────────────
            (state, UserCommand(Command::Cancel)) if !state.is_terminal() => (
                Stopped {
                    reason: StopReason::Cancelled,
                },
                vec![
                    Effect::Notify(Notification::RunFailed {
                        summary: "cancelled by user".into(),
                    }),
                    Effect::Persist,
                ],
            ),

            // ── Effect errors ─────────────────────────────────────────────
            (state, EffectError { effect, reason }) if !state.is_terminal() => {
                let msg = format!("{effect}: {reason}");
                (
                    Stopped {
                        reason: StopReason::Failed {
                            reason: msg.clone(),
                        },
                    },
                    vec![
                        Effect::Notify(Notification::RunFailed { summary: msg }),
                        Effect::Persist,
                    ],
                )
            }

            // ── Ignore anything else ──────────────────────────────────────
            (state, _) => (state, vec![]),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shunt_core::machine::{
        ArtifactSnapshot, AutonomyPolicy, GateDecision, MachineEvent, StopReason, TaskState,
        UserRequest,
    };
    use shunt_core::{AmbiguityId, ArtifactId, CommandOutcome, CommandSpec, RecipeRef};

    fn ctx() -> SessionContext {
        SessionContext {
            task_id: "task-1".into(),
            artifact_id: ArtifactId("art-1".into()),
            workspace_root: "/tmp/ws".into(),
            recipe: RecipeRef {
                id: "test".into(),
                version: "v1".into(),
            },
        }
    }

    fn ask_policy() -> AutonomyPolicy {
        AutonomyPolicy::agentic()
    }

    fn auto_policy() -> AutonomyPolicy {
        AutonomyPolicy {
            clarify: GateDecision::Auto,
            approval: GateDecision::Auto,
            run_commands: GateDecision::Auto,
        }
    }

    fn t(
        state: TaskState,
        event: MachineEvent,
        policy: &AutonomyPolicy,
    ) -> (TaskState, Vec<Effect>) {
        TaskMachine::transition(state, event, &ctx(), policy)
    }

    fn empty_snapshot() -> ArtifactSnapshot {
        ArtifactSnapshot {
            interpreted_goal: String::new(),
            confidence: 0.0,
            evidence_count: 0,
            candidate_paths: vec![],
            open_ambiguity_count: 0,
            open_risks: vec![],
        }
    }

    fn pending(id: &str, question: &str) -> PendingAmbiguity {
        PendingAmbiguity {
            id: AmbiguityId(id.into()),
            question: question.into(),
            options: vec![],
        }
    }

    // ── Submit ───────────────────────────────────────────────────────────────

    #[test]
    fn running_submit_starts_observe() {
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::UserCommand(Command::Submit {
                request: "fix it".into(),
            }),
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Running));
        assert!(effects.iter().any(|e| matches!(e, Effect::Observe { .. })));
    }

    // ── Observe → ProposeChange ──────────────────────────────────────────────

    #[test]
    fn observe_completed_triggers_propose() {
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::ObserveCompleted,
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Running));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::ProposeChange { .. }))
        );
    }

    // ── Proposal ready → approval gate ───────────────────────────────────────

    #[test]
    fn proposal_ready_auto_policy_commits() {
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::ProposalReady {
                confidence: 0.9,
                op_count: 2,
                snapshot: empty_snapshot(),
            },
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Running));
        assert!(effects.iter().any(|e| matches!(e, Effect::CommitChange { .. })));
    }

    #[test]
    fn proposal_ready_ask_policy_waits_for_approval() {
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::ProposalReady {
                confidence: 0.9,
                op_count: 2,
                snapshot: empty_snapshot(),
            },
            &ask_policy(),
        );
        assert!(matches!(
            next,
            TaskState::WaitingForUser {
                request: UserRequest::Approval { candidate_count: 2, .. }
            }
        ));
        assert!(!effects.iter().any(|e| matches!(e, Effect::CommitChange { .. })));
    }

    #[test]
    fn proposal_ready_zero_ops_completes() {
        let (next, _effects) = t(
            TaskState::Running,
            MachineEvent::ProposalReady {
                confidence: 0.9,
                op_count: 0,
                snapshot: empty_snapshot(),
            },
            &ask_policy(),
        );
        assert!(matches!(next, TaskState::Completed));
    }

    // ── Agent asked → clarification pause ────────────────────────────────────

    #[test]
    fn agent_asked_waits_for_clarification() {
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::AgentAsked {
                ambiguity_id: "agent-question".into(),
                question: "Which config file?".into(),
                options: vec![],
            },
            &auto_policy(),
        );
        assert!(matches!(
            next,
            TaskState::WaitingForUser {
                request: UserRequest::Clarification { .. }
            }
        ));
        assert!(effects.iter().any(|e| matches!(
            e,
            Effect::Notify(Notification::ClarificationNeeded { .. })
        )));
    }

    // ── Clarification answer ─────────────────────────────────────────────────

    #[test]
    fn answer_resumes_agent() {
        let (next, effects) = t(
            TaskState::WaitingForUser {
                request: UserRequest::Clarification {
                    open: vec![pending("q1", "Which auth flow?")],
                    confidence: 0.5,
                },
            },
            MachineEvent::UserCommand(Command::Answer {
                ambiguity_id: AmbiguityId("q1".into()),
                answer: "use OAuth".into(),
            }),
            &ask_policy(),
        );
        assert!(matches!(next, TaskState::Running));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::RecordAnswer { .. }))
        );
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::ProposeChange { .. }))
        );
    }

    #[test]
    fn answer_one_of_two_stays_waiting_emits_next() {
        let (next, effects) = t(
            TaskState::WaitingForUser {
                request: UserRequest::Clarification {
                    open: vec![pending("q1", "Auth?"), pending("q2", "Framework?")],
                    confidence: 0.5,
                },
            },
            MachineEvent::UserCommand(Command::Answer {
                ambiguity_id: AmbiguityId("q1".into()),
                answer: "OAuth".into(),
            }),
            &ask_policy(),
        );
        assert!(matches!(next, TaskState::WaitingForUser { .. }));
        assert!(effects.iter().any(|e| matches!(
            e,
            Effect::Notify(Notification::ClarificationNeeded { ambiguity_id, .. })
            if ambiguity_id == "q2"
        )));
    }

    // ── Approval gate ────────────────────────────────────────────────────────

    #[test]
    fn approve_triggers_commit() {
        let (next, effects) = t(
            TaskState::WaitingForUser {
                request: UserRequest::Approval {
                    candidate_count: 2,
                    snapshot: empty_snapshot(),
                },
            },
            MachineEvent::UserCommand(Command::Approve),
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Running));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::RecordApproval { approved: true, .. }))
        );
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::CommitChange { .. }))
        );
    }

    #[test]
    fn reject_reruns_agent() {
        let (next, effects) = t(
            TaskState::WaitingForUser {
                request: UserRequest::Approval {
                    candidate_count: 2,
                    snapshot: empty_snapshot(),
                },
            },
            MachineEvent::UserCommand(Command::Reject),
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Running));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::ProposeChange { .. }))
        );
    }

    // ── Patch applied ────────────────────────────────────────────────────────

    #[test]
    fn patch_applied_no_commands_completes() {
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::PatchApplied {
                path: "src/lib.rs".into(),
                setup_commands: vec![],
            },
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Completed));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::Notify(Notification::RunCompleted { .. })))
        );
    }

    #[test]
    fn patch_applied_with_commands_runs_setup() {
        let cmd = CommandSpec {
            program: "npm".into(),
            args: vec!["install".into()],
        };
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::PatchApplied {
                path: "package.json".into(),
                setup_commands: vec![cmd],
            },
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Running));
        assert!(effects.iter().any(|e| matches!(e, Effect::RunSetup { .. })));
    }

    // ── Setup ────────────────────────────────────────────────────────────────

    #[test]
    fn setup_completed_completes_task() {
        let outcome = CommandOutcome {
            spec: CommandSpec {
                program: "npm".into(),
                args: vec!["install".into()],
            },
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
            success: true,
        };
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::SetupCompleted {
                outcomes: vec![outcome],
            },
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Completed));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::Notify(Notification::RunCompleted { .. })))
        );
    }

    // ── Dangerous commands ───────────────────────────────────────────────────

    #[test]
    fn danger_detected_waits_for_user() {
        let cmd = CommandSpec {
            program: "rm".into(),
            args: vec!["-rf".into()],
        };
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::DangerCommandsDetected {
                commands: vec![cmd],
                reason: "rm -rf".into(),
            },
            &auto_policy(),
        );
        assert!(matches!(
            next,
            TaskState::WaitingForUser {
                request: UserRequest::DangerousCommands { .. }
            }
        ));
        assert!(effects.iter().any(|e| matches!(
            e,
            Effect::Notify(Notification::DangerousCommandsProposed { .. })
        )));
    }

    #[test]
    fn approve_dangerous_runs_setup() {
        let cmd = CommandSpec {
            program: "rm".into(),
            args: vec![],
        };
        let (next, effects) = t(
            TaskState::WaitingForUser {
                request: UserRequest::DangerousCommands {
                    commands: vec![cmd],
                    reason: "test".into(),
                },
            },
            MachineEvent::UserCommand(Command::ApproveDangerousCommands),
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Running));
        assert!(effects.iter().any(|e| matches!(e, Effect::RunSetup { .. })));
    }

    #[test]
    fn reject_dangerous_completes_without_setup() {
        let (next, effects) = t(
            TaskState::WaitingForUser {
                request: UserRequest::DangerousCommands {
                    commands: vec![],
                    reason: "test".into(),
                },
            },
            MachineEvent::UserCommand(Command::RejectDangerousCommands),
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Completed));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::Notify(Notification::RunCompleted { .. })))
        );
    }

    // ── Cancel ───────────────────────────────────────────────────────────────

    #[test]
    fn cancel_from_running_stops() {
        let (next, effects) = t(
            TaskState::Running,
            MachineEvent::UserCommand(Command::Cancel),
            &auto_policy(),
        );
        assert!(matches!(
            next,
            TaskState::Stopped {
                reason: StopReason::Cancelled
            }
        ));
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::Notify(Notification::RunFailed { .. })))
        );
    }

    #[test]
    fn cancel_from_waiting_stops() {
        let (next, _) = t(
            TaskState::WaitingForUser {
                request: UserRequest::Approval {
                    candidate_count: 1,
                    snapshot: empty_snapshot(),
                },
            },
            MachineEvent::UserCommand(Command::Cancel),
            &auto_policy(),
        );
        assert!(matches!(
            next,
            TaskState::Stopped {
                reason: StopReason::Cancelled
            }
        ));
    }

    #[test]
    fn cancel_in_terminal_state_is_ignored() {
        let (next, effects) = t(
            TaskState::Completed,
            MachineEvent::UserCommand(Command::Cancel),
            &auto_policy(),
        );
        assert!(matches!(next, TaskState::Completed));
        assert!(effects.is_empty());
    }

    // ── Effect error ──────────────────────────────────────────────────────────

    #[test]
    fn effect_error_stops_with_failure() {
        let (next, _) = t(
            TaskState::Running,
            MachineEvent::EffectError {
                effect: "ProposeChange".into(),
                reason: "timeout".into(),
            },
            &auto_policy(),
        );
        assert!(matches!(
            next,
            TaskState::Stopped {
                reason: StopReason::Failed { .. }
            }
        ));
    }

    // ── State helpers ─────────────────────────────────────────────────────────

    #[test]
    fn terminal_states_are_terminal() {
        assert!(TaskState::Completed.is_terminal());
        assert!(
            TaskState::Stopped {
                reason: StopReason::Cancelled
            }
            .is_terminal()
        );
        assert!(
            TaskState::Stopped {
                reason: StopReason::Failed { reason: "x".into() }
            }
            .is_terminal()
        );
        assert!(!TaskState::Running.is_terminal());
        assert!(
            !TaskState::WaitingForUser {
                request: UserRequest::Approval {
                    candidate_count: 0,
                    snapshot: empty_snapshot()
                }
            }
            .is_terminal()
        );
    }

    #[test]
    fn waiting_states_are_waiting() {
        assert!(
            TaskState::WaitingForUser {
                request: UserRequest::Approval {
                    candidate_count: 0,
                    snapshot: empty_snapshot()
                }
            }
            .is_waiting()
        );
        assert!(!TaskState::Running.is_waiting());
        assert!(!TaskState::Completed.is_waiting());
    }
}
