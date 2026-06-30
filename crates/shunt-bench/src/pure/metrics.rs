use super::agent_loop::{RunTrace, StopReason};
use crate::capability::Outcome;

#[derive(Debug, Clone)]
pub struct PureRunMetrics {
    pub outcome: Outcome,
    pub turns: usize,
    pub native_tool_pct: f32,
    pub valid_args_pct: f32,
    pub unknown_tool_count: usize,
    pub schema_mismatch_count: usize,
    pub thrash: usize,
    /// True if model read/searched before its first edit (or made no edits).
    pub read_before_edit: bool,
    pub total_tokens: u32,
    pub wall_secs: f32,
    pub stop_reason: String,
}

pub fn reduce(trace: &RunTrace, passed: bool) -> PureRunMetrics {
    let outcome = classify(trace, passed);

    let native_tool_pct = if trace.turns == 0 {
        0.0
    } else {
        trace.native_tool_turns as f32 / trace.turns as f32
    };

    let valid_args_pct = if trace.total_tool_calls == 0 {
        1.0
    } else {
        trace.valid_args_calls as f32 / trace.total_tool_calls as f32
    };

    PureRunMetrics {
        outcome,
        turns: trace.turns,
        native_tool_pct,
        valid_args_pct,
        unknown_tool_count: trace.unknown_tool_count,
        schema_mismatch_count: trace.schema_mismatch_count,
        thrash: trace.thrash_count,
        read_before_edit: trace.read_before_edit,
        total_tokens: trace.prompt_tokens + trace.completion_tokens,
        wall_secs: trace.wall_secs,
        stop_reason: trace.stop_reason.label().to_string(),
    }
}

fn classify(trace: &RunTrace, passed: bool) -> Outcome {
    // Check correctness first — a model that writes the right answer and burns
    // extra turns (or a verbose model that hits the cap) still solved the task.
    if passed {
        return Outcome::Success;
    }
    match &trace.stop_reason {
        StopReason::TurnCap | StopReason::HttpError(_) => Outcome::NotCompleted,
        StopReason::Finished | StopReason::NoToolCalls => {
            if trace.had_edit {
                Outcome::WrongEdit
            } else {
                Outcome::NoEdit
            }
        }
    }
}
