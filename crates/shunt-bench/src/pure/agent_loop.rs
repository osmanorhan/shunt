use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use serde_json::Value;

use super::client::{ChatClient, ChatRequest, Message};
use super::tools::{self, ParseError, ToolInvocation};

const SYSTEM_PROMPT: &str = "\
You are an agentic coding assistant working in a local file workspace. \
Use the provided tools to complete the user's task:\n\
1. Call list_files or search to understand the workspace.\n\
2. Call read_file before editing any file.\n\
3. Use edit_file to make changes (str_replace for targeted edits, write to create/overwrite).\n\
4. Call finish when all required changes are complete.\n\
Do not ask for clarification. Take action.";

pub struct LoopConfig {
    pub model_id: String,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: u32,
    pub turn_cap: usize,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            temperature: 0.2,
            top_p: 0.95,
            max_tokens: 4096,
            turn_cap: 20,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StopReason {
    Finished,
    NoToolCalls,
    TurnCap,
    HttpError(String),
}

impl StopReason {
    pub fn label(&self) -> &str {
        match self {
            StopReason::Finished => "finished",
            StopReason::NoToolCalls => "no_tool_calls",
            StopReason::TurnCap => "turn_cap",
            StopReason::HttpError(_) => "http_error",
        }
    }
}

/// Aggregated stats from one run.
#[derive(Debug)]
pub struct RunTrace {
    pub stop_reason: StopReason,
    pub turns: usize,
    /// Turns where model used real tool_calls (not content fallback).
    pub native_tool_turns: usize,
    /// Tool calls whose args were valid JSON + schema-valid.
    pub valid_args_calls: usize,
    pub total_tool_calls: usize,
    pub unknown_tool_count: usize,
    pub schema_mismatch_count: usize,
    /// Repeated identical (name, args) calls.
    pub thrash_count: usize,
    /// Model called read_file or search before its first edit_file.
    pub read_before_edit: bool,
    pub had_edit: bool,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub wall_secs: f32,
    /// Full message log for the run report.
    pub log: Vec<LogEntry>,
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub turn: usize,
    pub kind: LogKind,
}

#[derive(Debug, Clone)]
pub enum LogKind {
    ModelResponse {
        finish_reason: Option<String>,
        content_preview: Option<String>,
    },
    ToolCall {
        name: String,
        args_preview: String,
        result_preview: String,
        is_error: bool,
        valid: bool,
    },
    Stopped {
        reason: String,
    },
}

pub fn run_loop(
    client: &ChatClient,
    cfg: &LoopConfig,
    task_request: &str,
    workspace: &Path,
) -> RunTrace {
    let schemas = tools::schemas();
    let mut messages = vec![Message::system(SYSTEM_PROMPT), Message::user(task_request)];

    let mut turns = 0usize;
    let mut native_tool_turns = 0usize;
    let mut valid_args_calls = 0usize;
    let mut total_tool_calls = 0usize;
    let mut unknown_tool_count = 0usize;
    let mut schema_mismatch_count = 0usize;
    let mut thrash_count = 0usize;
    let mut had_read = false;
    let mut read_before_edit = true;
    let mut had_edit = false;
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;
    let mut seen_calls: HashSet<String> = HashSet::new();
    let mut log: Vec<LogEntry> = Vec::new();
    let t0 = Instant::now();

    let mut stop_reason = StopReason::TurnCap;

    'outer: for turn in 0..cfg.turn_cap {
        let req = ChatRequest {
            model: cfg.model_id.clone(),
            messages: messages.clone(),
            tools: schemas.clone(),
            tool_choice: "auto",
            temperature: cfg.temperature,
            top_p: cfg.top_p,
            max_tokens: cfg.max_tokens,
            stream: false,
        };

        let resp = match client.post(&req) {
            Ok(r) => r,
            Err(e) => {
                let err = e.clone();
                log.push(LogEntry {
                    turn,
                    kind: LogKind::Stopped {
                        reason: format!("http_error: {e}"),
                    },
                });
                stop_reason = StopReason::HttpError(err);
                break;
            }
        };

        if let Some(usage) = resp.usage {
            prompt_tokens = usage.prompt_tokens;
            completion_tokens += usage.completion_tokens;
        }

        let choice = match resp.choices.into_iter().next() {
            Some(c) => c,
            None => {
                log.push(LogEntry {
                    turn,
                    kind: LogKind::Stopped {
                        reason: "empty choices".into(),
                    },
                });
                stop_reason = StopReason::HttpError("empty choices".into());
                break;
            }
        };

        let has_tool_calls = choice
            .message
            .tool_calls
            .as_ref()
            .is_some_and(|t| !t.is_empty());

        log.push(LogEntry {
            turn,
            kind: LogKind::ModelResponse {
                finish_reason: choice.finish_reason.clone(),
                content_preview: choice.message.content.as_ref().map(|s| truncate(s, 120)),
            },
        });

        if has_tool_calls {
            native_tool_turns += 1;
        }
        turns += 1;

        if !has_tool_calls {
            stop_reason = StopReason::NoToolCalls;
            log.push(LogEntry {
                turn,
                kind: LogKind::Stopped {
                    reason: "no_tool_calls".into(),
                },
            });
            break;
        }

        let tool_calls = choice.message.tool_calls.unwrap_or_default();
        messages.push(Message::assistant_with_calls(tool_calls.clone()));

        for tc in &tool_calls {
            total_tool_calls += 1;

            let call_key = format!("{}:{}", tc.function.name, tc.function.arguments);
            if seen_calls.contains(&call_key) {
                thrash_count += 1;
            }
            seen_calls.insert(call_key);

            let args_result = serde_json::from_str::<Value>(&tc.function.arguments);
            let args = args_result.unwrap_or(Value::Null);
            let inv_result = tools::parse_invocation(&tc.function.name, &args);

            match &inv_result {
                Ok(_) => valid_args_calls += 1,
                Err(ParseError::UnknownTool(_)) => unknown_tool_count += 1,
                Err(ParseError::SchemaMismatch(_)) => schema_mismatch_count += 1,
            }

            let tool_result = if let Ok(inv) = &inv_result {
                track_rw(inv, &mut had_read, &mut had_edit, &mut read_before_edit);
                let r = tools::dispatch(inv, workspace);
                if matches!(inv, ToolInvocation::Finish { .. }) {
                    let res_preview = truncate(&r.content, 80);
                    log.push(LogEntry {
                        turn,
                        kind: LogKind::ToolCall {
                            name: tc.function.name.clone(),
                            args_preview: truncate(&tc.function.arguments, 80),
                            result_preview: res_preview,
                            is_error: r.is_error,
                            valid: true,
                        },
                    });
                    messages.push(Message::tool_result(
                        tc.id.clone(),
                        tc.function.name.clone(),
                        r.content,
                    ));
                    stop_reason = StopReason::Finished;
                    log.push(LogEntry {
                        turn,
                        kind: LogKind::Stopped {
                            reason: "finished".into(),
                        },
                    });
                    break 'outer;
                }
                r
            } else {
                let msg = match &inv_result {
                    Err(ParseError::UnknownTool(t)) => format!("unknown tool: {t}"),
                    Err(ParseError::SchemaMismatch(e)) => format!("schema error: {e}"),
                    Ok(_) => unreachable!(),
                };
                tools::ToolResult::err(msg)
            };

            log.push(LogEntry {
                turn,
                kind: LogKind::ToolCall {
                    name: tc.function.name.clone(),
                    args_preview: truncate(&tc.function.arguments, 80),
                    result_preview: truncate(&tool_result.content, 80),
                    is_error: tool_result.is_error,
                    valid: inv_result.is_ok(),
                },
            });

            messages.push(Message::tool_result(
                tc.id.clone(),
                tc.function.name.clone(),
                tool_result.content,
            ));
        }
    }

    if matches!(stop_reason, StopReason::TurnCap) {
        log.push(LogEntry {
            turn: cfg.turn_cap,
            kind: LogKind::Stopped {
                reason: "turn_cap".into(),
            },
        });
    }

    RunTrace {
        stop_reason,
        turns,
        native_tool_turns,
        valid_args_calls,
        total_tool_calls,
        unknown_tool_count,
        schema_mismatch_count,
        thrash_count,
        read_before_edit,
        had_edit,
        prompt_tokens,
        completion_tokens,
        wall_secs: t0.elapsed().as_secs_f32(),
        log,
    }
}

fn track_rw(
    inv: &ToolInvocation,
    had_read: &mut bool,
    had_edit: &mut bool,
    read_before_edit: &mut bool,
) {
    match inv {
        ToolInvocation::ReadFile { .. } | ToolInvocation::Search { .. } => {
            *had_read = true;
        }
        ToolInvocation::EditFile { .. } => {
            if !*had_read && !*had_edit {
                *read_before_edit = false;
            }
            *had_edit = true;
        }
        _ => {}
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max])
    }
}
