//! Queue-based scripted provider for offline testing.
//!
//! Each call to `call_tool` pops the next pre-built JSON value from the queue
//! and returns it as `ToolCall` arguments.  Callers must push responses in the
//! exact order the runtime will request them (Clarify → Understand → ProposeChange).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use shunt_infer::{InferError, InferResult, ToolCall, ToolProvider, ToolSpec};

#[derive(Debug, Clone)]
pub struct Call {
    pub system: String,
    pub user: String,
}

/// A scripted provider that returns pre-built JSON responses in order.
pub struct ScriptedProvider {
    responses: Arc<Mutex<VecDeque<Value>>>,
    calls: Arc<Mutex<Vec<Call>>>,
}

impl ScriptedProvider {
    pub fn new(responses: Vec<Value>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into())),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Number of scripted responses not yet consumed.
    pub fn remaining(&self) -> usize {
        self.responses.lock().unwrap().len()
    }

    /// Returns a snapshot of every (system, user) pair that was called.
    pub fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl ToolProvider for ScriptedProvider {
    fn call_tool(&self, system: &str, user: &str, tool: &ToolSpec) -> InferResult<ToolCall> {
        self.calls.lock().unwrap().push(Call {
            system: system.to_string(),
            user: user.chars().take(200).collect(),
        });

        let arguments = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or(InferError::EmptyResponse)?;

        Ok(ToolCall {
            name: tool.name.clone(),
            arguments,
        })
    }
}
