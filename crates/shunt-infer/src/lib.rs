pub mod agent;
pub mod engine;
pub mod registry;

pub use agent::{
    AgentObserver, AgentPauseState, AgentResult, AgentSession, AgentTurn, CriterionOutcome,
    DEFAULT_IGNORE_PATTERNS, KnowledgeLookup, SessionBudget, SessionBudgetOverride,
};

use reqwest::blocking::Client;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use shunt_core::{
    Ambiguity, AmbiguityKind, AmbiguityStatus, EvidenceRef, ManualEvidence, ManualVersionStatus,
    PackageFact, RequiredPath, RequiredPathIntent, Risk, RiskSeverity, UnderstandingArtifact,
    VerifierOutcome, VerifierStatus, WorkContract,
};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("http client error: {0}")]
    HttpClient(#[source] reqwest::Error),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("empty model response")]
    EmptyResponse,
    #[error("model call exceeded the {0}s per-call deadline")]
    Timeout(u64),
    #[error("model called unexpected tool: expected {expected}, got {actual}")]
    UnexpectedTool { expected: String, actual: String },
    #[error("model output failed validation after {retries} retries: {reason}")]
    InvalidOutput { retries: usize, reason: String },
    #[error("io error: {0}")]
    Io(String),
}

pub type InferResult<T> = Result<T, InferError>;

pub const MAX_RETRIES: usize = 3;

static NEXT_CALL_ID: AtomicU64 = AtomicU64::new(1);

/// Fallback reasoning-token cap for tool-decision calls on models that expose no
/// thinking budget of their own. Models that DO declare `thinking_budget_tokens`
/// (e.g. Gemma-4 at 2048) govern themselves — a smaller hardcoded cap would
/// guillotine the thought channel mid-stream, forcing the model to reopen a
/// `<|channel>thought` block *after* the tool call and re-emit, which puts the raw
/// generation out of gemma's grammar order and makes llama.cpp's peg parser 500.
const AGENT_REASONING_BUDGET: i32 = 384;

#[derive(Debug, Clone)]
pub enum ModelCallEvent {
    Started {
        call_id: u64,
        tool: String,
        model: String,
        mode: String,
    },
    Finished {
        call_id: u64,
        tool: String,
        elapsed_ms: u64,
        outcome: String,
    },
    /// Emitted for each streamed token/chunk during inference.
    TokenChunk {
        call_id: u64,
        /// Partial text — may be a JSON fragment when tool-calling.
        text: String,
        /// True when this token is from the model's internal reasoning chain
        /// (`reasoning_content`), false when it is actual output (`content` /
        /// tool-call arguments).  The TUI collapses thinking into a single
        /// indicator line and only shows real output in full.
        is_thinking: bool,
    },
}

pub type ModelCallObserver = Arc<dyn Fn(ModelCallEvent) + Send + Sync>;

// ── Tool-calling types ────────────────────────────────────────────────────────

/// Describes a single tool the model can call: its name, purpose, and the
/// JSON Schema of its parameters.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the expected arguments.
    pub parameters: serde_json::Value,
}

/// A single tool call returned by the model.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    /// Parsed JSON arguments matching the tool's parameter schema.
    pub arguments: serde_json::Value,
}

// ── ToolProvider trait ────────────────────────────────────────────────────────

/// The Layer-F provider contract.
///
/// `call_tool` is the primitive: one bounded model call, schema-declared.
/// `generate_structured` is a default-implemented helper that builds a ToolSpec
/// from the caller's schema, calls `call_tool`, and retries up to `MAX_RETRIES`
/// on malformed output — so the caller never panics on a bad response.
pub trait ToolProvider {
    fn call_tool(&self, system: &str, user: &str, tool: &ToolSpec) -> InferResult<ToolCall>;

    /// Return the capabilities of this provider's loaded model.
    /// Used by `AgentSession` to derive an appropriate `AgentBudget`.
    /// Default returns conservative capabilities; real providers override this.
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    fn with_call_observer(&self, _observer: ModelCallObserver) -> Self
    where
        Self: Sized + Clone,
    {
        self.clone()
    }

    /// Generate a structured `T` by passing `schema` to the backend for
    /// constrained decoding.  Retries up to `MAX_RETRIES` on JSON / shape errors;
    /// propagates HTTP errors immediately.
    fn generate_structured<T: DeserializeOwned>(
        &self,
        system: &str,
        user: &str,
        schema: &serde_json::Value,
    ) -> InferResult<T>
    where
        Self: Sized,
    {
        self.generate_structured_named("output", system, user, schema)
    }

    /// Generate a plain text response without tool calling.
    /// Used when the model needs to produce large content (file contents) that
    /// it won't reliably include in JSON tool call arguments.
    fn generate_text(&self, system: &str, user: &str) -> InferResult<String> {
        let _ = (system, user);
        Err(InferError::InvalidOutput {
            retries: 0,
            reason: "generate_text not implemented for this provider".into(),
        })
    }

    fn generate_structured_named<T: DeserializeOwned>(
        &self,
        name: &str,
        system: &str,
        user: &str,
        schema: &serde_json::Value,
    ) -> InferResult<T>
    where
        Self: Sized,
    {
        let tool = ToolSpec {
            name: name.into(),
            description: "Generate the required structured output.".into(),
            parameters: schema.clone(),
        };
        let mut last_err = String::new();
        for attempt in 0..MAX_RETRIES {
            match self.call_tool(system, user, &tool) {
                Ok(tc) => match serde_json::from_value::<T>(tc.arguments) {
                    Ok(v) => return Ok(v),
                    Err(e) => {
                        tracing::warn!(
                            "generate_structured attempt {}/{MAX_RETRIES} shape error: {e}",
                            attempt + 1
                        );
                        last_err = e.to_string();
                    }
                },
                Err(InferError::Json(e)) => {
                    tracing::warn!(
                        "generate_structured attempt {}/{MAX_RETRIES} json error: {e}",
                        attempt + 1
                    );
                    last_err = e.to_string();
                }
                Err(e) => return Err(e),
            }
        }
        Err(InferError::InvalidOutput {
            retries: MAX_RETRIES,
            reason: last_err,
        })
    }

    /// Multi-turn variant: call the model with a full conversation history instead
    /// of a flat (system, user) pair.  Default falls back to extracting the system
    /// message and the last user message, then calling `call_tool`.
    /// Override in providers that support KV-cached multi-turn for the cache benefit.
    fn call_tool_from_messages(
        &self,
        messages: &[ChatMessage],
        tool: &ToolSpec,
    ) -> InferResult<ToolCall> {
        let system = messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        let user = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        self.call_tool(system, user, tool)
    }

    /// Multi-turn variant of `generate_structured_named` — accepts the full
    /// conversation history.  Retries up to `MAX_RETRIES` on shape/JSON errors.
    fn generate_from_messages<T: DeserializeOwned>(
        &self,
        name: &str,
        messages: &[ChatMessage],
        schema: &serde_json::Value,
    ) -> InferResult<T>
    where
        Self: Sized,
    {
        let tool = ToolSpec {
            name: name.into(),
            description: "Generate the required structured output.".into(),
            parameters: schema.clone(),
        };
        let mut last_err = String::new();
        for attempt in 0..MAX_RETRIES {
            match self.call_tool_from_messages(messages, &tool) {
                Ok(tc) => match serde_json::from_value::<T>(tc.arguments) {
                    Ok(v) => return Ok(v),
                    Err(e) => {
                        tracing::warn!(
                            "generate_from_messages attempt {}/{MAX_RETRIES} shape error: {e}",
                            attempt + 1
                        );
                        last_err = e.to_string();
                    }
                },
                Err(InferError::Json(e)) => {
                    tracing::warn!(
                        "generate_from_messages attempt {}/{MAX_RETRIES} json error: {e}",
                        attempt + 1
                    );
                    last_err = e.to_string();
                }
                Err(e) => return Err(e),
            }
        }
        Err(InferError::InvalidOutput {
            retries: MAX_RETRIES,
            reason: last_err,
        })
    }
}

// ── OpenAI-compatible provider ────────────────────────────────────────────────

/// Provider for any OpenAI-compatible endpoint (Ollama /v1, llama.cpp, vLLM,
/// SGLang, OpenAI itself).  Uses the `tools` + `tool_choice` format so the
/// backend applies constrained decoding where supported; falls back to content
/// extraction when the model returns JSON in the message body instead.
#[derive(Clone)]
pub struct OpenAiCompatProvider {
    client: Client,
    endpoint: String,
    model: String,
    capabilities: ProviderCapabilities,
    observer: Option<ModelCallObserver>,
    call_timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenAiTransport {
    #[default]
    ChatCompletions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    NamedObject,
    #[default]
    RequiredString,
    AutoString,
    Omit,
    /// Use `response_format: json_schema` instead of tool calling.
    /// Triggers grammar-based constrained decoding on llama.cpp-compatible
    /// servers.  Best for thinking models (Gemma, Qwen-thinking) that ignore
    /// `tool_choice` and emit content JSON instead.
    JsonSchema,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProviderCapabilities {
    pub preferred_transport: OpenAiTransport,
    pub supports_chat_completions_tools: bool,
    pub supports_chat_completions_json_schema: bool,
    pub tool_choice_mode: ToolChoiceMode,
    /// Total token budget per call (thinking tokens + output tokens).
    pub max_tokens: u32,
    /// Prepend `/no_think` to every user message so the model skips its reasoning chain.
    pub disable_thinking: bool,
    /// llama.cpp `budget_tokens`: max reasoning tokens before forced output.
    pub thinking_budget_tokens: Option<u32>,
    /// Temperature for action-selection calls (grammar-constrained routing).
    pub temperature: f32,
    /// Temperature for content-generation calls. Falls back to `temperature` if None.
    pub content_temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    /// Send enable_thinking:false on content-generation calls.
    /// Required for Gemma-4 (empty-output bug); not needed for Qwen3.
    pub suppress_content_thinking: bool,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            preferred_transport: OpenAiTransport::ChatCompletions,
            supports_chat_completions_tools: true,
            supports_chat_completions_json_schema: true,
            tool_choice_mode: ToolChoiceMode::RequiredString,
            max_tokens: 32768,
            disable_thinking: false,
            thinking_budget_tokens: None,
            temperature: 0.7,
            content_temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            suppress_content_thinking: false,
        }
    }
}

impl ProviderCapabilities {
    /// Derive an appropriate `AgentBudget` for a session using this provider.
    /// Smaller token ceilings → tighter budgets and earlier stall warnings.
    pub fn to_session_budget(&self) -> agent::SessionBudget {
        agent::SessionBudget::for_model(self.max_tokens)
    }

    /// Probe a llama-server endpoint to detect if the loaded model generates thinking tokens.
    /// Returns true if the server's /props reports a non-"none" reasoning_format,
    /// or if the model registry identifies it as a thinking model.
    /// Use this at startup to warn users about potential slow inference.
    pub fn server_has_thinking(endpoint: &str) -> bool {
        let url = format!("{}/props", endpoint.trim_end_matches('/'));
        let Ok(resp) = reqwest::blocking::get(&url) else {
            return false;
        };
        let Ok(json) = resp.json::<serde_json::Value>() else {
            return false;
        };
        let format = json
            .pointer("/default_generation_settings/params/reasoning_format")
            .and_then(|v| v.as_str())
            .unwrap_or("none");
        format != "none"
    }

    /// Derive capabilities from the model ID and endpoint URL without user input.
    ///
    /// Resolution order:
    /// 1. Look up the model family in the built-in registry.
    /// 2. Use the single OpenAI-compatible path: chat-completions function tools.
    pub fn detect(model_id: &str, endpoint: &str) -> Self {
        use engine::{EngineKind, detect_engine};
        use registry::ModelRegistry;
        let engine = detect_engine(endpoint);
        let profile = ModelRegistry::with_defaults().resolve(model_id);
        let (supports_chat_completions_tools, supports_chat_completions_json_schema) = match engine
        {
            EngineKind::Ollama => (true, true),
            EngineKind::LlamaCpp => (true, true),
            EngineKind::Vllm => (true, true),
            EngineKind::Generic => (true, true),
        };
        let tool_choice_mode = ToolChoiceMode::RequiredString;
        let preferred_transport = OpenAiTransport::ChatCompletions;
        Self {
            preferred_transport,
            supports_chat_completions_tools,
            supports_chat_completions_json_schema,
            tool_choice_mode,
            max_tokens: profile.max_tokens,
            disable_thinking: false,
            thinking_budget_tokens: None,
            temperature: profile.temperature,
            content_temperature: profile.content_temperature,
            top_p: profile.top_p,
            top_k: profile.top_k,
            min_p: profile.min_p,
            presence_penalty: profile.presence_penalty,
            repetition_penalty: profile.repetition_penalty,
            suppress_content_thinking: false,
        }
    }
}

impl OpenAiCompatProvider {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_timeout(endpoint, model, Duration::from_secs(300))
            .expect("static reqwest client configuration should be valid")
    }

    pub fn with_timeout(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        timeout: Duration,
    ) -> InferResult<Self> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .build()
            .map_err(InferError::HttpClient)?;

        Ok(Self {
            client,
            endpoint: endpoint.into(),
            model: model.into(),
            capabilities: ProviderCapabilities::default(),
            observer: None,
            call_timeout: timeout,
        })
    }

    pub fn with_capabilities(mut self, capabilities: ProviderCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    fn request_for(&self, system: &str, user: &str, tool: &ToolSpec) -> ChatCompletionRequest {
        let user_content = if self.capabilities.disable_thinking {
            format!("/no_think\n{user}")
        } else {
            user.to_owned()
        };
        let tool_defs = openai_tool_definitions(tool);
        let tool_choice = Some(serde_json::json!("required"));

        ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                chat_message("system", system),
                chat_message("user", &user_content),
            ],
            temperature: Some(self.capabilities.temperature),
            max_tokens: self.capabilities.max_tokens,
            top_p: self.capabilities.top_p,
            top_k: self.capabilities.top_k,
            min_p: self.capabilities.min_p,
            presence_penalty: self.capabilities.presence_penalty,
            repetition_penalty: self.capabilities.repetition_penalty,
            tools: tool_defs,
            tool_choice,
            response_format: None,
            stream: None,
            budget_tokens: self.capabilities.thinking_budget_tokens,
            cache_prompt: Some(true),
            chat_template_kwargs: None,
            reasoning_budget: Some(
                self.capabilities
                    .thinking_budget_tokens
                    .map(|b| b as i32)
                    .unwrap_or(AGENT_REASONING_BUDGET),
            ),
        }
    }

    /// Build a `ChatCompletionRequest` from a full conversation history.
    /// Mirrors `request_for` but accepts an arbitrary `Vec<ChatMessage>` so
    /// prior turns participate in KV-cache prefix matching.
    fn request_from_messages(
        &self,
        messages: &[ChatMessage],
        tool: &ToolSpec,
    ) -> ChatCompletionRequest {
        // Apply /no_think prefix or chat_template_kwargs to the last user message.
        let processed: Vec<ChatMessage> = {
            let mut v = messages.to_vec();
            if self.capabilities.disable_thinking
                && let Some(last_user) = v.iter_mut().rev().find(|m| m.role == "user")
            {
                last_user.content = format!("/no_think\n{}", last_user.content);
            }
            v
        };

        let tool_defs = openai_tool_definitions(tool);
        let tool_choice = Some(serde_json::json!("required"));
        ChatCompletionRequest {
            model: self.model.clone(),
            messages: native_chat_messages(&processed, tool),
            temperature: Some(self.capabilities.temperature),
            max_tokens: self.capabilities.max_tokens,
            top_p: self.capabilities.top_p,
            top_k: self.capabilities.top_k,
            min_p: self.capabilities.min_p,
            presence_penalty: self.capabilities.presence_penalty,
            repetition_penalty: self.capabilities.repetition_penalty,
            tools: tool_defs,
            tool_choice,
            response_format: None,
            stream: None,
            budget_tokens: self.capabilities.thinking_budget_tokens,
            cache_prompt: Some(true),
            chat_template_kwargs: None,
            reasoning_budget: Some(
                self.capabilities
                    .thinking_budget_tokens
                    .map(|b| b as i32)
                    .unwrap_or(AGENT_REASONING_BUDGET),
            ),
        }
    }

    fn emit(&self, event: ModelCallEvent) {
        if let Some(observer) = &self.observer {
            observer(event);
        }
    }

    /// POST a chat-completion request with a HARD per-call deadline. The blocking
    /// HTTP call runs on a worker thread; if it doesn't return within the deadline
    /// we abandon it (the thread leaks, but the session never freezes) and return
    /// `Timeout`. Belt-and-braces over reqwest's own timeout, which has been seen
    /// to NOT fire on a server that accepts the connection but never responds.
    /// Also logs the full request body so config (e.g. enable_thinking) is visible.
    fn post_chat<T: DeserializeOwned + Send + 'static>(
        &self,
        req: &ChatCompletionRequest,
    ) -> InferResult<T> {
        self.post_json("/v1/chat/completions", req)
    }

    fn post_json<T: DeserializeOwned + Send + 'static, S: Serialize>(
        &self,
        path: &str,
        req: &S,
    ) -> InferResult<T> {
        let body = serde_json::to_vec(req).map_err(InferError::Json)?;
        tracing::debug!("── request body ──\n{}", String::from_utf8_lossy(&body));
        let client = self.client.clone();
        let deadline = self.call_timeout;
        let url = format!("{}{}", self.endpoint.trim_end_matches('/'), path);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Per-request timeout = the deadline, so a timed-out request is
            // CANCELLED (not left running). On a single-slot llama-server a leaked
            // request keeps hogging the slot and serialises every retry behind it —
            // that cascade, not the model, was the real "hang".
            let result = client
                .post(&url)
                .timeout(deadline)
                .header("content-type", "application/json")
                .body(body)
                .send()
                .and_then(reqwest::blocking::Response::error_for_status)
                .and_then(reqwest::blocking::Response::json::<T>);
            let _ = tx.send(result);
        });
        // Wait slightly longer than the request timeout so reqwest's cancellation
        // returns the real error before we give up (avoids a double abandonment).
        match rx.recv_timeout(deadline + Duration::from_secs(15)) {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(InferError::Http(e)),
            Err(_) => Err(InferError::Timeout(deadline.as_secs())),
        }
    }

    /// Plain text completion — no tool calling, just collect the content stream.
    pub fn call_text(&self, system: &str, user: &str) -> InferResult<String> {
        tracing::debug!(
            "\n═══ CALL_TEXT ═══  model={}  endpoint={}\n\
             ── system ({} chars) ──\n{}\n\
             ── user ({} chars) ──\n{}",
            self.model,
            self.endpoint,
            system.len(),
            system,
            user.len(),
            user
        );
        let content_temp = self
            .capabilities
            .content_temperature
            .unwrap_or(self.capabilities.temperature);
        // For models where suppress_content_thinking=true (Qwen3), also prepend /no_think
        // so the model skips reasoning even if the server's chat_template_kwargs isn't
        // sufficient on its own.
        let user_content =
            if self.capabilities.suppress_content_thinking && self.capabilities.disable_thinking {
                format!("/no_think\n{user}")
            } else {
                user.to_owned()
            };
        let req = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                chat_message("system", system),
                chat_message("user", &user_content),
            ],
            temperature: Some(content_temp),
            max_tokens: self.capabilities.max_tokens,
            top_p: self.capabilities.top_p,
            top_k: self.capabilities.top_k,
            min_p: self.capabilities.min_p,
            presence_penalty: self.capabilities.presence_penalty,
            repetition_penalty: self.capabilities.repetition_penalty,
            tools: vec![],
            tool_choice: None,
            response_format: None,
            stream: Some(false),
            budget_tokens: self.capabilities.thinking_budget_tokens,
            // Allow KV-cache reuse. The content generation prompt is a completely
            // different message (different system + user text) from any prior action
            // selection call, so the cache key won't collide. Forcing cache_prompt:false
            // caused KV-cache write/flush races after action selection and connection drops.
            cache_prompt: Some(true),
            // Only suppress thinking on content calls for models where it causes
            // empty output (Gemma-4: exhausts max_tokens in reasoning_content).
            // Qwen3: let the model reason about the code it's writing.
            chat_template_kwargs: if self.capabilities.suppress_content_thinking {
                Some(serde_json::json!({ "enable_thinking": false }))
            } else {
                None
            },
            reasoning_budget: None,
        };

        let resp: ChatCompletionResponse = self.post_chat(&req)?;

        let content = resp
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("")
            .to_string();
        tracing::debug!(
            "\n── CALL_TEXT response ({} chars) ──\n{}",
            content.len(),
            content
        );
        Ok(content)
    }

    fn call_tool_chat(&self, system: &str, user: &str, tool: &ToolSpec) -> InferResult<ToolCall> {
        tracing::debug!(
            "\n═══ CALL_TOOL ═══  model={}  endpoint={}\n\
             ── tool: {} ──\n\
             ── system ({} chars) ──\n{}\n\
             ── user ({} chars) ──\n{}",
            self.model,
            self.endpoint,
            tool.name,
            system.len(),
            system,
            user.len(),
            user
        );

        let call_id = NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        self.emit(ModelCallEvent::Started {
            call_id,
            tool: tool.name.clone(),
            model: self.model.clone(),
            mode: format!("{:?}", self.capabilities.tool_choice_mode),
        });

        let mut req = self.request_for(system, user, tool);
        req.stream = Some(false);

        let resp: ChatCompletionResponse = match self.post_chat(&req) {
            Ok(v) => v,
            Err(err) => {
                self.emit(ModelCallEvent::Finished {
                    call_id,
                    tool: tool.name.clone(),
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    outcome: format!("call_failed: {err}"),
                });
                return Err(err);
            }
        };
        self.finish_chat_tool_call(call_id, started, tool, resp, false)
    }

    fn call_tool_chat_from_messages(
        &self,
        messages: &[ChatMessage],
        tool: &ToolSpec,
    ) -> InferResult<ToolCall> {
        let call_id = NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        self.emit(ModelCallEvent::Started {
            call_id,
            tool: tool.name.clone(),
            model: self.model.clone(),
            mode: format!(
                "{:?}+history({})",
                self.capabilities.tool_choice_mode,
                messages.len()
            ),
        });

        let mut req = self.request_from_messages(messages, tool);
        req.stream = Some(false);

        let resp: ChatCompletionResponse = match self.post_chat(&req) {
            Ok(v) => v,
            Err(err) => {
                self.emit(ModelCallEvent::Finished {
                    call_id,
                    tool: tool.name.clone(),
                    elapsed_ms: started.elapsed().as_millis() as u64,
                    outcome: format!("call_failed: {err}"),
                });
                return Err(err);
            }
        };
        self.finish_chat_tool_call(call_id, started, tool, resp, true)
    }

    fn finish_chat_tool_call(
        &self,
        call_id: u64,
        started: Instant,
        tool: &ToolSpec,
        resp: ChatCompletionResponse,
        from_history: bool,
    ) -> InferResult<ToolCall> {
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let normalized = match normalize_chat_completion_message(
            resp.choices.into_iter().next().map(|c| c.message),
            &tool.name,
            self.capabilities.tool_choice_mode,
        ) {
            Ok(output) => output,
            Err(InferError::EmptyResponse) => {
                self.emit(ModelCallEvent::Finished {
                    call_id,
                    tool: tool.name.clone(),
                    elapsed_ms,
                    outcome: "empty_response".into(),
                });
                return Err(InferError::EmptyResponse);
            }
            Err(err) => {
                self.emit(ModelCallEvent::Finished {
                    call_id,
                    tool: tool.name.clone(),
                    elapsed_ms,
                    outcome: format!("parse_failed: {err}"),
                });
                return Err(err);
            }
        };
        self.finish_normalized_tool_call(call_id, elapsed_ms, tool, normalized, from_history)
    }

    fn finish_normalized_tool_call(
        &self,
        call_id: u64,
        elapsed_ms: u64,
        tool: &ToolSpec,
        normalized: NormalizedAssistantOutput,
        from_history: bool,
    ) -> InferResult<ToolCall> {
        let Some(tool_call) = normalized.tool_call else {
            self.emit(ModelCallEvent::Finished {
                call_id,
                tool: tool.name.clone(),
                elapsed_ms,
                outcome: "empty_response".into(),
            });
            return Err(InferError::EmptyResponse);
        };

        let actual_tool = tool_call.name;
        let args_str = tool_call.arguments_json;
        tracing::debug!("call_tool result for {actual_tool}: {args_str}");

        let (returned_tool_name, args_str) = if tool.name == "agent_action" {
            (
                tool.name.clone(),
                agent_action_arguments(&actual_tool, &args_str)?,
            )
        } else {
            (actual_tool.clone(), args_str)
        };

        if returned_tool_name != tool.name {
            self.emit(ModelCallEvent::Finished {
                call_id,
                tool: tool.name.clone(),
                elapsed_ms,
                outcome: format!("unexpected_tool: {actual_tool}"),
            });
            return Err(InferError::UnexpectedTool {
                expected: tool.name.clone(),
                actual: returned_tool_name,
            });
        }

        let arguments: serde_json::Value = match serde_json::from_str(&args_str) {
            Ok(v) => v,
            Err(err) => {
                self.emit(ModelCallEvent::Finished {
                    call_id,
                    tool: tool.name.clone(),
                    elapsed_ms,
                    outcome: format!("invalid_json: {err}"),
                });
                return Err(InferError::Json(err));
            }
        };

        self.emit(ModelCallEvent::Finished {
            call_id,
            tool: tool.name.clone(),
            elapsed_ms,
            outcome: if from_history {
                "tool_call_history".into()
            } else {
                "tool_call".into()
            },
        });
        Ok(ToolCall {
            name: returned_tool_name,
            arguments,
        })
    }
}

impl ToolProvider for OpenAiCompatProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities
    }

    fn generate_text(&self, system: &str, user: &str) -> InferResult<String> {
        self.call_text(system, user)
    }

    fn with_call_observer(&self, observer: ModelCallObserver) -> Self {
        let mut provider = self.clone();
        provider.observer = Some(observer);
        provider
    }

    fn call_tool(&self, system: &str, user: &str, tool: &ToolSpec) -> InferResult<ToolCall> {
        self.call_tool_chat(system, user, tool)
    }

    /// Multi-turn override: passes the full conversation history to the server.
    /// The stable prefix (system + prior turns) is KV-cached by llama.cpp so only
    /// the new last message needs fresh prefill, cutting per-turn cost significantly.
    fn call_tool_from_messages(
        &self,
        messages: &[ChatMessage],
        tool: &ToolSpec,
    ) -> InferResult<ToolCall> {
        self.call_tool_chat_from_messages(messages, tool)
    }
}

// ── Ollama provider ───────────────────────────────────────────────────────────

/// Provider for Ollama's native `/api/chat` endpoint.  Passes the tool's
/// parameter schema as the `format` field, enabling Ollama's grammar-based
/// constrained decoding (Ollama 0.5+).
#[derive(Clone)]
pub struct OllamaProvider {
    client: Client,
    endpoint: String,
    model: String,
    /// When false, skip the grammar `format` field so the model outputs free-form JSON.
    /// The response is parsed with `extract_json_object` (think-block stripping + JSON extraction).
    format_constrained: bool,
    observer: Option<ModelCallObserver>,
}

impl OllamaProvider {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_timeout(endpoint, model, Duration::from_secs(300))
            .expect("static reqwest client configuration should be valid")
    }

    pub fn with_timeout(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        timeout: Duration,
    ) -> InferResult<Self> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(timeout)
            .build()
            .map_err(InferError::HttpClient)?;
        Ok(Self {
            client,
            endpoint: endpoint.into(),
            model: model.into(),
            format_constrained: true,
            observer: None,
        })
    }

    /// Skip grammar-constrained decoding. The model outputs free-form JSON in the
    /// message content; think-blocks are stripped and the first JSON object extracted.
    pub fn unconstrained(self) -> Self {
        Self {
            format_constrained: false,
            ..self
        }
    }
}

impl ToolProvider for OllamaProvider {
    fn with_call_observer(&self, observer: ModelCallObserver) -> Self {
        let mut provider = self.clone();
        provider.observer = Some(observer);
        provider
    }

    fn call_tool(&self, system: &str, user: &str, tool: &ToolSpec) -> InferResult<ToolCall> {
        tracing::debug!(
            "ollama call start endpoint={} model={}",
            self.endpoint,
            self.model
        );
        tracing::debug!("tool: {} — {}", tool.name, tool.description);
        let call_id = NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        if let Some(observer) = &self.observer {
            observer(ModelCallEvent::Started {
                call_id,
                tool: tool.name.clone(),
                model: self.model.clone(),
                mode: "JsonSchema".into(),
            });
        }

        // Suppress chain-of-thought for thinking models: the /no_think instruction
        // works as a text hint without disabling the format constraint (unlike
        // the think:false API parameter which breaks grammar-constrained decoding).
        use registry::ModelRegistry;
        let profile = ModelRegistry::with_defaults().resolve(&self.model);
        let user_content = if profile.supports_thinking {
            format!("/no_think\n{user}")
        } else {
            user.to_owned()
        };

        let response = self
            .client
            .post(format!("{}/api/chat", self.endpoint.trim_end_matches('/')))
            .json(&OllamaChatRequest {
                model: self.model.clone(),
                messages: vec![
                    ChatMessage {
                        role: "system".into(),
                        content: system.into(),
                    },
                    ChatMessage {
                        role: "user".into(),
                        content: user_content,
                    },
                ],
                stream: false,
                format: if self.format_constrained {
                    Some(tool.parameters.clone())
                } else {
                    None
                },
                options: Some(OllamaOptions { temperature: 0.1 }),
                // think:false is safe when not using format (grammar constraint).
                // When format is set, think:false breaks constrained decoding.
                think: if self.format_constrained {
                    None
                } else {
                    Some(false)
                },
            })
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(reqwest::blocking::Response::json::<OllamaChatResponse>);

        let response = match response {
            Ok(response) => response,
            Err(err) => {
                if let Some(observer) = &self.observer {
                    observer(ModelCallEvent::Finished {
                        call_id,
                        tool: tool.name.clone(),
                        elapsed_ms: started.elapsed().as_millis() as u64,
                        outcome: format!("http_error: {err}"),
                    });
                }
                return Err(InferError::Http(err));
            }
        };

        let content = response.message.content;
        tracing::debug!("ollama raw response:\n{content}");

        let arguments = match serde_json::from_str(&extract_json_object(&content)) {
            Ok(arguments) => arguments,
            Err(err) => {
                if let Some(observer) = &self.observer {
                    observer(ModelCallEvent::Finished {
                        call_id,
                        tool: tool.name.clone(),
                        elapsed_ms: started.elapsed().as_millis() as u64,
                        outcome: format!("invalid_json: {err}"),
                    });
                }
                return Err(InferError::Json(err));
            }
        };
        if let Some(observer) = &self.observer {
            observer(ModelCallEvent::Finished {
                call_id,
                tool: tool.name.clone(),
                elapsed_ms: started.elapsed().as_millis() as u64,
                outcome: "json_schema".into(),
            });
        }
        Ok(ToolCall {
            name: tool.name.clone(),
            arguments,
        })
    }
}

// ── Nodes ─────────────────────────────────────────────────────────────────────

pub struct ClarifyNode<'a, P> {
    provider: &'a P,
    agent_context: Option<String>,
}

pub struct KnowledgePlanNode<'a, P> {
    provider: &'a P,
}

impl<'a, P> ClarifyNode<'a, P>
where
    P: ToolProvider,
{
    pub fn new(provider: &'a P) -> Self {
        Self {
            provider,
            agent_context: None,
        }
    }

    pub fn with_agent_context(mut self, ctx: String) -> Self {
        if !ctx.is_empty() {
            self.agent_context = Some(ctx);
        }
        self
    }

    pub fn run(&self, artifact: &UnderstandingArtifact) -> InferResult<ClarifyOutput> {
        let schema = serde_json::to_value(schemars::schema_for!(ClarifyOutput)).unwrap_or_default();
        self.provider.generate_structured_named(
            "clarify",
            clarify_system_prompt(),
            &clarify_user_prompt(artifact, self.agent_context.as_deref()),
            &schema,
        )
    }
}

impl<'a, P> KnowledgePlanNode<'a, P>
where
    P: ToolProvider,
{
    pub fn new(provider: &'a P) -> Self {
        Self { provider }
    }

    pub fn run(&self, artifact: &UnderstandingArtifact) -> InferResult<KnowledgePlanOutput> {
        let schema =
            serde_json::to_value(schemars::schema_for!(KnowledgePlanOutput)).unwrap_or_default();
        self.provider.generate_structured_named(
            "knowledge_plan",
            knowledge_plan_system_prompt(),
            &knowledge_plan_user_prompt(artifact),
            &schema,
        )
    }
}

pub struct UnderstandNode<'a, P> {
    provider: &'a P,
    agent_context: Option<String>,
}

impl<'a, P> UnderstandNode<'a, P>
where
    P: ToolProvider,
{
    pub fn new(provider: &'a P) -> Self {
        Self {
            provider,
            agent_context: None,
        }
    }

    pub fn with_agent_context(mut self, ctx: String) -> Self {
        if !ctx.is_empty() {
            self.agent_context = Some(ctx);
        }
        self
    }

    pub fn run(&self, artifact: &UnderstandingArtifact) -> InferResult<UnderstandOutput> {
        let schema =
            serde_json::to_value(schemars::schema_for!(UnderstandOutput)).unwrap_or_default();
        self.provider.generate_structured_named(
            "understand",
            understand_system_prompt(),
            &understand_user_prompt(artifact, self.agent_context.as_deref()),
            &schema,
        )
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ClarifyOutput {
    pub interpreted_goal: String,
    pub success_criteria: Vec<String>,
    pub constraints: Vec<String>,
    pub ambiguities: Vec<ClarifyAmbiguity>,
    pub confidence: f32,
}

/// Ambiguity kind as returned by the LLM.
/// `lookup` = the agent can resolve this by querying a registry or public docs.
/// `user_decision` = only the user can decide.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClarifyAmbiguityKind {
    Lookup,
    #[default]
    UserDecision,
    /// Architectural fork: multiple valid approaches exist or a conflict makes the naive approach dangerous.
    /// Always surfaces to the user; never auto-resolved.
    ApproachChoice,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ClarifyAmbiguity {
    /// One concise sentence the user sees. Max 300 characters.
    #[schemars(length(max = 300))]
    pub question: String,
    pub options: Vec<String>,
    /// Classification of this ambiguity. Defaults to `user_decision` if the LLM omits it.
    /// Excluded from the schema so the model never tries to generate an enum value;
    /// serde uses the Default impl (UserDecision) when the field is absent.
    #[serde(default)]
    #[schemars(skip)]
    pub kind: ClarifyAmbiguityKind,
}

impl ClarifyOutput {
    pub fn apply_to(self, artifact: &mut UnderstandingArtifact) {
        artifact.interpreted_goal = self.interpreted_goal;
        artifact.success_criteria = self.success_criteria;
        artifact.constraints = self.constraints;
        artifact.ambiguities = map_ambiguities(self.ambiguities);
        artifact.confidence = self.confidence.clamp(0.0, 1.0);
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct UnderstandOutput {
    pub interpreted_goal: String,
    pub success_criteria: Vec<String>,
    pub target_scope: Vec<String>,
    #[serde(default)]
    pub work_contract: UnderstandWorkContract,
    pub ambiguities: Vec<ClarifyAmbiguity>,
    pub risks: Vec<UnderstandRisk>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KnowledgePlanOutput {
    pub should_research: bool,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub requests: Vec<KnowledgePlanRequest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KnowledgePlanRequest {
    pub summary: String,
    #[serde(default)]
    pub package_hints: Vec<String>,
    #[serde(default)]
    pub ecosystem_hints: Vec<String>,
    #[serde(default)]
    pub search_queries: Vec<String>,
    #[serde(default)]
    pub source_hints: Vec<String>,
    #[serde(default)]
    pub freshness_required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, JsonSchema)]
pub struct UnderstandWorkContract {
    #[serde(default)]
    pub required_paths: Vec<UnderstandRequiredPath>,
    #[serde(default)]
    pub behavioral_checks: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct UnderstandRequiredPath {
    pub path: String,
    pub intent: UnderstandRequiredPathIntent,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UnderstandRequiredPathIntent {
    #[default]
    Exist,
    CreateOrUpdate,
    Remove,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct UnderstandRisk {
    pub summary: String,
    pub severity: UnderstandRiskSeverity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SourceFileContext {
    pub path: String,
    pub contents: String,
}

/// A setup command proposed by the LLM alongside a change set.
/// Kept separate from `shunt_core::CommandSpec` to keep infer's schema clean.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default, JsonSchema)]
pub struct ProposedCommand {
    pub program: String,
    pub args: Vec<String>,
    /// Working directory relative to workspace root. None means workspace root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(default)]
    pub expect_workspace_change: bool,
    #[serde(default)]
    pub expect_paths: Vec<String>,
}

/// A single file operation proposed by the LLM.
/// Uses an inline `op` discriminant so the LLM emits a self-describing object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ProposedFileOp {
    /// Create a new file (or overwrite an existing one entirely).
    Create { path: String, contents: String },
    /// Apply a targeted search-and-replace to an existing file.
    /// `search` must appear verbatim in the file; `replacement` replaces it.
    Edit {
        path: String,
        search: String,
        replacement: String,
    },
    /// Delete a file from the workspace.
    Delete { path: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum UnderstandRiskSeverity {
    Low,
    Medium,
    High,
}

impl UnderstandOutput {
    pub fn apply_to(self, artifact: &mut UnderstandingArtifact) {
        artifact.interpreted_goal = self.interpreted_goal;
        artifact.success_criteria = self.success_criteria;
        if !self.target_scope.is_empty() {
            artifact.target_scope = self.target_scope;
        }
        artifact.work_contract = WorkContract {
            required_paths: self
                .work_contract
                .required_paths
                .into_iter()
                .map(|path| RequiredPath {
                    path: path.path,
                    intent: match path.intent {
                        UnderstandRequiredPathIntent::Exist => RequiredPathIntent::Exist,
                        UnderstandRequiredPathIntent::CreateOrUpdate => {
                            RequiredPathIntent::CreateOrUpdate
                        }
                        UnderstandRequiredPathIntent::Remove => RequiredPathIntent::Remove,
                    },
                    reason: path.reason,
                })
                .collect(),
            behavioral_checks: self.work_contract.behavioral_checks,
        };
        artifact.ambiguities = map_ambiguities(self.ambiguities);
        artifact.risks = self
            .risks
            .into_iter()
            .enumerate()
            .map(|(index, risk)| Risk {
                id: format!("risk-{}", index + 1),
                summary: risk.summary,
                severity: match risk.severity {
                    UnderstandRiskSeverity::Low => RiskSeverity::Low,
                    UnderstandRiskSeverity::Medium => RiskSeverity::Medium,
                    UnderstandRiskSeverity::High => RiskSeverity::High,
                },
            })
            .collect();
        artifact.confidence = self.confidence.clamp(0.0, 1.0);
    }
}

// ── Probe — Layer X swarm primitive ──────────────────────────────────────────

/// Context handed to every probe at run time.
pub struct ProbeCtx {
    pub workspace_root: std::path::PathBuf,
    pub artifact: UnderstandingArtifact,
}

/// The narrow answer a probe produces.
pub struct ProbeResult {
    /// Structured answer from the probe (probe-specific shape).
    pub answer: serde_json::Value,
    /// Evidence refs that support the answer (file paths, package refs, …).
    pub evidence: Vec<EvidenceRef>,
    /// How confident this probe is in its answer (0..=1).
    pub confidence: f32,
}

/// A single, bounded, individually-verifiable question about the workspace.
///
/// Probes are the swarm primitives.  Each probe asks one narrow question and
/// exposes a cheap deterministic verifier.  The orchestrator runs them,
/// keeps only verified answers, and composes the results into scope.
pub trait Probe: Send + Sync {
    fn id(&self) -> &str;

    fn run(&self, ctx: &ProbeCtx, provider: &dyn ToolProvider) -> InferResult<ProbeResult>;

    /// Cheap deterministic check run immediately after `run`.
    /// Default: always pass — override to add structural verification.
    fn verify(&self, result: &ProbeResult, workspace: &Path) -> VerifierOutcome {
        let _ = (result, workspace);
        VerifierOutcome {
            verifier: self.id().to_string(),
            status: VerifierStatus::Passed,
            summary: format!("{}: accepted", self.id()),
        }
    }
}

// ── System / user prompts ─────────────────────────────────────────────────────

fn clarify_system_prompt() -> &'static str {
    include_str!("../../../prompts/clarify.system.txt")
}

fn understand_system_prompt() -> &'static str {
    include_str!("../../../prompts/understand.system.txt")
}

fn knowledge_plan_system_prompt() -> &'static str {
    include_str!("../../../prompts/knowledge-plan.system.txt")
}

fn clarify_user_prompt(artifact: &UnderstandingArtifact, agent_context: Option<&str>) -> String {
    let profile = &artifact.workspace_profile;
    let profile_str = format!(
        "runtimes: {runtimes}; frameworks: {frameworks}; topology: {topology}; conflicts: {conflicts}",
        runtimes = if profile.runtimes.is_empty() {
            "unknown".into()
        } else {
            profile.runtimes.join(", ")
        },
        frameworks = if profile.frameworks.is_empty() {
            "none detected".into()
        } else {
            profile.frameworks.join(", ")
        },
        topology = profile.topology,
        conflicts = if profile.conflicts.is_empty() {
            "none".into()
        } else {
            profile
                .conflicts
                .iter()
                .map(|c| format!("• {c}"))
                .collect::<Vec<_>>()
                .join(" ")
        },
    );
    let mut prompt = include_str!("../../../prompts/clarify.user.txt")
        .replace("{original_request}", &artifact.original_request)
        .replace("{workspace_profile}", &profile_str)
        .replace(
            "{observed_evidence}",
            &serde_json::to_string(&artifact.evidence).unwrap_or_else(|_| "[]".into()),
        )
        .replace("{draft_interpreted_goal}", &artifact.interpreted_goal)
        .replace(
            "{existing_constraints}",
            &serde_json::to_string(&artifact.constraints).unwrap_or_else(|_| "[]".into()),
        )
        .replace(
            "{existing_success_criteria}",
            &serde_json::to_string(&artifact.success_criteria).unwrap_or_else(|_| "[]".into()),
        );
    if let Some(ctx) = agent_context {
        prompt.push_str("\n\n");
        prompt.push_str(ctx);
    }
    prompt
}

fn understand_user_prompt(artifact: &UnderstandingArtifact, agent_context: Option<&str>) -> String {
    let package_facts = selected_package_facts_for_understand(artifact);
    let manual_evidence = selected_manual_evidence_for_understand(artifact);
    let mut prompt = include_str!("../../../prompts/understand.user.txt")
        .replace("{original_request}", &artifact.original_request)
        .replace("{draft_interpreted_goal}", &artifact.interpreted_goal)
        .replace(
            "{draft_success_criteria}",
            &serde_json::to_string(&artifact.success_criteria).unwrap_or_else(|_| "[]".into()),
        )
        .replace(
            "{constraints}",
            &serde_json::to_string(&artifact.constraints).unwrap_or_else(|_| "[]".into()),
        )
        .replace(
            "{target_scope}",
            &serde_json::to_string(&artifact.target_scope).unwrap_or_else(|_| "[]".into()),
        )
        .replace(
            "{evidence}",
            &serde_json::to_string(&artifact.evidence).unwrap_or_else(|_| "[]".into()),
        )
        .replace(
            "{package_facts}",
            &serde_json::to_string(&package_facts).unwrap_or_else(|_| "[]".into()),
        )
        .replace(
            "{manual_evidence}",
            &serde_json::to_string(&manual_evidence).unwrap_or_else(|_| "[]".into()),
        );
    if let Some(ctx) = agent_context {
        prompt.push_str("\n\n");
        prompt.push_str(ctx);
    }
    prompt
}

fn knowledge_plan_user_prompt(artifact: &UnderstandingArtifact) -> String {
    let profile = &artifact.workspace_profile;
    include_str!("../../../prompts/knowledge-plan.user.txt")
        .replace("{original_request}", &artifact.original_request)
        .replace("{draft_interpreted_goal}", &artifact.interpreted_goal)
        .replace(
            "{workspace_profile}",
            &format!(
                "runtimes: {}; frameworks: {}; topology: {}; conflicts: {}",
                if profile.runtimes.is_empty() {
                    "unknown".into()
                } else {
                    profile.runtimes.join(", ")
                },
                if profile.frameworks.is_empty() {
                    "none detected".into()
                } else {
                    profile.frameworks.join(", ")
                },
                profile.topology,
                if profile.conflicts.is_empty() {
                    "none".into()
                } else {
                    profile.conflicts.join(" | ")
                }
            ),
        )
        .replace(
            "{observed_evidence}",
            &serde_json::to_string(&artifact.evidence).unwrap_or_else(|_| "[]".into()),
        )
}

// ── Evidence selection helpers ────────────────────────────────────────────────

fn selected_package_facts_for_understand(artifact: &UnderstandingArtifact) -> Vec<PackageFact> {
    select_package_facts(artifact, 3, true)
}

fn select_package_facts(
    artifact: &UnderstandingArtifact,
    limit: usize,
    allow_unversioned: bool,
) -> Vec<PackageFact> {
    let selected_manuals = select_manual_evidence(artifact, limit, allow_unversioned);
    if selected_manuals.is_empty() {
        return Vec::new();
    }
    let selected_packages = selected_manuals
        .iter()
        .map(|manual| (manual.ecosystem.clone(), manual.package.clone()))
        .collect::<Vec<_>>();
    artifact
        .package_facts
        .iter()
        .filter(|fact| {
            selected_packages
                .iter()
                .any(|(ecosystem, package)| fact.ecosystem == *ecosystem && fact.name == *package)
        })
        .take(limit)
        .cloned()
        .collect()
}

fn selected_manual_evidence_for_understand(
    artifact: &UnderstandingArtifact,
) -> Vec<ManualEvidence> {
    select_manual_evidence(artifact, 2, true)
}

fn select_manual_evidence(
    artifact: &UnderstandingArtifact,
    limit: usize,
    allow_unversioned: bool,
) -> Vec<ManualEvidence> {
    let mut selected = artifact
        .manual_evidence
        .iter()
        .filter(|manual| {
            matches!(
                manual.version_status,
                ManualVersionStatus::Exact | ManualVersionStatus::CompatibleRange
            )
        })
        .cloned()
        .collect::<Vec<_>>();

    if selected.is_empty() && allow_unversioned {
        selected = artifact
            .manual_evidence
            .iter()
            .filter(|manual| {
                manual.version_status == ManualVersionStatus::Unversioned
                    && request_mentions_package(artifact, &manual.package)
            })
            .cloned()
            .collect::<Vec<_>>();
    }

    selected.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.package.cmp(&right.package))
    });
    selected.truncate(limit);
    selected
}

fn request_mentions_package(artifact: &UnderstandingArtifact, package: &str) -> bool {
    let request = format!(
        "{} {}",
        artifact.original_request, artifact.interpreted_goal
    )
    .to_ascii_lowercase();
    let package = package.to_ascii_lowercase();
    request.contains(&package)
        || package
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .filter(|part| !part.is_empty())
            .any(|part| request.contains(part))
}

fn map_ambiguities(ambiguities: Vec<ClarifyAmbiguity>) -> Vec<Ambiguity> {
    ambiguities
        .into_iter()
        .enumerate()
        .map(|(index, ambiguity)| {
            let kind = match ambiguity.kind {
                ClarifyAmbiguityKind::Lookup => AmbiguityKind::Lookup,
                ClarifyAmbiguityKind::UserDecision => AmbiguityKind::UserDecision,
                ClarifyAmbiguityKind::ApproachChoice => AmbiguityKind::ApproachChoice,
            };
            Ambiguity {
                id: format!("ambiguity-{}", index + 1),
                question: ambiguity.question,
                options: ambiguity.options,
                kind,
                status: AmbiguityStatus::Open,
                resolution: None,
            }
        })
        .collect()
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

fn extract_json_object(input: &str) -> String {
    let stripped = strip_think_blocks(input);
    // Strip markdown code fence (```json ... ``` or ``` ... ```)
    let inner = if let Some(rest) = stripped.strip_prefix("```json") {
        rest.trim_start_matches('\n')
            .rsplit_once("```")
            .map(|(body, _)| body)
            .unwrap_or(rest)
    } else if let Some(rest) = stripped.strip_prefix("```") {
        rest.trim_start_matches('\n')
            .rsplit_once("```")
            .map(|(body, _)| body)
            .unwrap_or(rest)
    } else {
        stripped
    };
    let start = inner.find('{').unwrap_or(0);
    let end = inner
        .rfind('}')
        .map(|index| index + 1)
        .unwrap_or(inner.len());
    inner[start..end].trim().to_string()
}

fn strip_think_blocks(input: &str) -> &str {
    let trimmed = input.trim();
    if let Some(rest) = trimmed.strip_prefix("<think>")
        && let Some(end) = rest.find("</think>")
    {
        return rest[end + "</think>".len()..].trim();
    }
    trimmed
}

fn strict_tool_schema(schema: &serde_json::Value) -> serde_json::Value {
    let definitions = schema
        .get("definitions")
        .or_else(|| schema.get("$defs"))
        .and_then(serde_json::Value::as_object);
    normalize_tool_schema(schema, definitions)
}

fn openai_tool_definitions(tool: &ToolSpec) -> Vec<OaiToolDefinition> {
    if tool.name == "agent_action" {
        return agent_action_tool_definitions(tool);
    }
    vec![openai_tool_definition(
        &tool.name,
        &tool.description,
        strict_tool_schema(&tool.parameters),
    )]
}

fn openai_tool_definition(
    name: &str,
    description: &str,
    parameters: serde_json::Value,
) -> OaiToolDefinition {
    OaiToolDefinition {
        type_: "function".into(),
        function: OaiToolFunction {
            name: name.into(),
            description: description.into(),
            strict: true,
            parameters,
        },
    }
}

fn agent_action_tool_definitions(tool: &ToolSpec) -> Vec<OaiToolDefinition> {
    let names = tool
        .parameters
        .pointer("/properties/tool/enum")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str);

    names
        .filter_map(|name| agent_action_schema_for(name).map(|schema| (name, schema)))
        .map(|(name, schema)| {
            openai_tool_definition(name, &format!("Agent action: {name}."), schema)
        })
        .collect()
}

fn object_schema(required: &[&str], properties: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "properties": properties,
        "required": required,
    })
}

fn agent_action_schema_for(name: &str) -> Option<serde_json::Value> {
    Some(match name {
        "read_file" => object_schema(
            &["path"],
            serde_json::json!({
                "path": {"type": "string"}
            }),
        ),
        "search_files" | "knowledge" => object_schema(
            &["query"],
            serde_json::json!({
                "query": {"type": "string"}
            }),
        ),
        "edit" => object_schema(
            &["path", "start_line", "end_line", "content"],
            serde_json::json!({
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "content": {"type": "string"}
            }),
        ),
        "command" => serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "command_line": {"type": "string"},
                "cwd": {"type": "string"}
            },
            "required": ["command_line"]
        }),
        "ask_user" => object_schema(
            &["question", "context"],
            serde_json::json!({
                "question": {"type": "string"},
                "context": {"type": "string"}
            }),
        ),
        "done" => object_schema(
            &["description"],
            serde_json::json!({
                "description": {"type": "string"}
            }),
        ),
        _ => return None,
    })
}

fn agent_action_arguments(tool_name: &str, arguments_json: &str) -> InferResult<String> {
    let mut arguments = match serde_json::from_str::<serde_json::Value>(arguments_json) {
        Ok(serde_json::Value::Object(arguments)) => arguments,
        Ok(_) => {
            return Err(InferError::InvalidOutput {
                retries: 0,
                reason: format!("tool '{tool_name}' arguments must be a JSON object"),
            });
        }
        Err(err) => return Err(InferError::Json(err)),
    };
    arguments.insert("tool".into(), serde_json::json!(tool_name));
    serde_json::to_string(&serde_json::Value::Object(arguments)).map_err(InferError::Json)
}

fn chat_message(role: &str, content: &str) -> serde_json::Value {
    serde_json::json!({"role": role, "content": content})
}

fn native_chat_messages(messages: &[ChatMessage], tool: &ToolSpec) -> Vec<serde_json::Value> {
    if tool.name != "agent_action" {
        return messages
            .iter()
            .map(|message| chat_message(&message.role, &message.content))
            .collect();
    }

    let mut out = Vec::new();
    let mut index = 0usize;
    while index < messages.len() {
        let message = &messages[index];
        if message.role == "assistant"
            && let Some((name, arguments)) = assistant_action_tool_call(&message.content)
        {
            let id = format!("call_history_{index}");
            out.push(serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments.to_string()
                    }
                }]
            }));
            if let Some(result) = messages.get(index + 1).filter(|next| next.role == "user") {
                out.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "name": name,
                    "content": result.content
                }));
                index += 2;
                continue;
            }
            index += 1;
            continue;
        }
        out.push(chat_message(&message.role, &message.content));
        index += 1;
    }
    out
}

fn assistant_action_tool_call(content: &str) -> Option<(String, serde_json::Value)> {
    let mut action =
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(content).ok()?;
    let name = action.remove("tool")?.as_str()?.to_string();
    agent_action_schema_for(&name)?;
    Some((name, serde_json::Value::Object(action)))
}

fn normalize_tool_schema(
    schema: &serde_json::Value,
    definitions: Option<&serde_json::Map<String, serde_json::Value>>,
) -> serde_json::Value {
    if let Some(reference) = schema.get("$ref").and_then(serde_json::Value::as_str)
        && let Some(name) = reference
            .strip_prefix("#/definitions/")
            .or_else(|| reference.strip_prefix("#/$defs/"))
        && let Some(target) = definitions.and_then(|defs| defs.get(name))
    {
        return normalize_tool_schema(target, definitions);
    }

    match schema {
        serde_json::Value::Object(object) => {
            let mut out = serde_json::Map::new();
            for (key, value) in object {
                if matches!(
                    key.as_str(),
                    "$schema" | "$defs" | "definitions" | "default" | "title" | "$ref"
                ) {
                    continue;
                }
                let normalized = match key.as_str() {
                    "properties" => serde_json::Value::Object(
                        value
                            .as_object()
                            .map(|properties| {
                                properties
                                    .iter()
                                    .map(|(name, property)| {
                                        (name.clone(), normalize_tool_schema(property, definitions))
                                    })
                                    .collect::<serde_json::Map<_, _>>()
                            })
                            .unwrap_or_default(),
                    ),
                    "items" => normalize_tool_schema(value, definitions),
                    "anyOf" | "oneOf" | "allOf" => serde_json::Value::Array(
                        value
                            .as_array()
                            .map(|items| {
                                items
                                    .iter()
                                    .map(|item| normalize_tool_schema(item, definitions))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    ),
                    _ => value.clone(),
                };
                out.insert(key.clone(), normalized);
            }

            if out.get("type").and_then(serde_json::Value::as_str) == Some("object") {
                out.entry("additionalProperties")
                    .or_insert_with(|| serde_json::json!(false));
                if let Some(properties) =
                    out.get("properties").and_then(serde_json::Value::as_object)
                {
                    let mut required = properties.keys().cloned().collect::<Vec<_>>();
                    required.sort();
                    out.insert("required".into(), serde_json::json!(required));
                }
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => serde_json::Value::Array(
            items
                .iter()
                .map(|item| normalize_tool_schema(item, definitions))
                .collect(),
        ),
        _ => schema.clone(),
    }
}

// ── Internal HTTP types: OpenAI-compat ───────────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    temperature: Option<f32>,
    /// Generous cap so thinking-model reasoning tokens don't exhaust the budget
    /// before the model can emit the actual structured output.
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OaiToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// llama.cpp extension: max reasoning/thinking tokens before forced output.
    /// Caps the thinking phase so `max_tokens - budget_tokens` remain for output.
    /// Servers that don't support this field silently ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u32>,
    /// llama.cpp's reasoning-token cap (the field `budget_tokens` was the wrong
    /// name and got ignored). For grammar TOOL-DECISION calls we keep thinking ON
    /// but capped: the model reasons in `reasoning_content` (which we discard) and
    /// then emits a SHORT, valid action — far better than thinking-off, which makes
    /// it dump rambling into the action's string fields and run to max_tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_budget: Option<i32>,
    /// llama.cpp extension: when false, the server skips KV-cache prefix matching
    /// and does not save this request's state for future reuse.
    /// Set to false on one-shot content calls to prevent stale KV state from a
    /// prior tool-call turn contaminating the generation (thinking-model issue).
    /// Servers that don't support this field silently ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_prompt: Option<bool>,
    /// llama.cpp / Jinja chat-template kwargs. We send `{"enable_thinking": false}`
    /// on pure content-generation calls (`call_text`): thinking models otherwise
    /// reason until they exhaust `max_tokens` and emit EMPTY content (the actual
    /// answer never leaves `reasoning_content`). Content generation needs no
    /// reasoning — the model's own thinking toggle is the right lever, not a code
    /// workaround. Chat templates without this key ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct OaiToolDefinition {
    #[serde(rename = "type")]
    type_: String,
    function: OaiToolFunction,
}

#[derive(Debug, Serialize)]
struct OaiToolFunction {
    name: String,
    description: String,
    strict: bool,
    parameters: serde_json::Value,
}

/// One message in an OpenAI-compatible chat conversation.
/// Pub so `AgentSession` can build multi-turn history and providers
/// can expose `call_tool_from_messages`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
struct AssistantMessage {
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    /// Present in tool-calling mode (non-grammar models). Grammar/JsonSchema
    /// models put the tool-call JSON in `content` instead.
    #[serde(default)]
    tool_calls: Option<Vec<RespToolCall>>,
}

#[derive(Debug, Deserialize)]
struct RespToolCall {
    function: RespFunction,
}

#[derive(Debug, Deserialize)]
struct RespFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, PartialEq, Eq)]
struct NormalizedAssistantOutput {
    text: Option<String>,
    thinking: Option<String>,
    tool_call: Option<NormalizedToolCall>,
}

#[derive(Debug, PartialEq, Eq)]
struct NormalizedToolCall {
    name: String,
    arguments_json: String,
}

fn normalize_chat_completion_message(
    message: Option<AssistantMessage>,
    tool_name: &str,
    tool_choice_mode: ToolChoiceMode,
) -> InferResult<NormalizedAssistantOutput> {
    let message = message.ok_or(InferError::EmptyResponse)?;
    let thinking = message
        .reasoning_content
        .map(|content| content.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(tc) = message.tool_calls.as_ref().and_then(|calls| calls.first()) {
        return Ok(NormalizedAssistantOutput {
            text: message
                .content
                .map(|content| content.trim().to_string())
                .filter(|s| !s.is_empty()),
            thinking,
            tool_call: Some(NormalizedToolCall {
                name: tc.function.name.clone(),
                arguments_json: tc.function.arguments.clone(),
            }),
        });
    }

    let _ = (tool_name, tool_choice_mode);
    let text = message
        .content
        .map(|content| content.trim().to_string())
        .filter(|s| !s.is_empty());

    Ok(NormalizedAssistantOutput {
        text,
        thinking,
        tool_call: None,
    })
}

// ── Internal HTTP types: Ollama ───────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaMessage,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    content: String,
}

// ── Server probing ─────────────────────────────────────────────────────────────

/// Returns true if the OpenAI-compatible server at `endpoint` accepts TCP connections.
/// Timeout is 400 ms — fast enough to probe several ports at startup.
pub fn probe_endpoint(endpoint: &str) -> bool {
    let host_port = endpoint
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_end_matches('/');
    let host_port = host_port.split('/').next().unwrap_or(host_port);
    let addr = if host_port.contains(':') {
        host_port.to_string()
    } else {
        format!("{host_port}:80")
    };
    let fallback = "127.0.0.1:8080".parse().expect("static socket address");
    std::net::TcpStream::connect_timeout(
        &addr.parse().unwrap_or(fallback),
        std::time::Duration::from_millis(400),
    )
    .is_ok()
}

/// Query `GET /v1/models` and return the list of model IDs, or None on failure.
pub fn list_models(endpoint: &str) -> Option<Vec<String>> {
    let url = format!("{}/v1/models", endpoint.trim_end_matches('/'));
    let resp = Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?
        .get(&url)
        .send()
        .ok()?;
    let json: serde_json::Value = resp.json().ok()?;
    let ids = json["data"]
        .as_array()?
        .iter()
        .filter_map(|model| model["id"].as_str().map(String::from))
        .collect::<Vec<_>>();
    (!ids.is_empty()).then_some(ids)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use shunt_core::{
        ApprovalState, ArtifactId, EvidenceKind, EvidenceRef, TaskId, UnderstandingArtifact,
    };
    use time::macros::datetime;

    use super::{
        AssistantMessage, ClarifyNode, OpenAiCompatProvider, OpenAiTransport, ProviderCapabilities,
        RespFunction, RespToolCall, ToolCall, ToolChoiceMode, ToolProvider, ToolSpec,
        UnderstandNode, extract_json_object, normalize_chat_completion_message,
    };

    struct StubProvider;

    impl ToolProvider for StubProvider {
        fn call_tool(
            &self,
            _system: &str,
            _user: &str,
            tool: &ToolSpec,
        ) -> super::InferResult<ToolCall> {
            let value = serde_json::json!({
                "interpreted_goal": "repair config loading failure",
                "success_criteria": ["config loads", "error path stays intact"],
                "constraints": ["keep patch minimal"],
                "ambiguities": [{"question": "which file owns loading?", "options": ["config.rs", "settings.rs"]}],
                "confidence": 0.68
            });
            Ok(ToolCall {
                name: tool.name.clone(),
                arguments: value,
            })
        }
    }

    #[test]
    fn clarify_node_applies_output() {
        let provider = StubProvider;
        let node = ClarifyNode::new(&provider);
        let now = datetime!(2026-05-01 12:00 UTC);
        let mut artifact = UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: TaskId("task-1".into()),
            original_request: "fix config loading".into(),
            interpreted_goal: "fix config loading".into(),
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

        let output = node.run(&artifact).unwrap();
        output.apply_to(&mut artifact);

        assert_eq!(artifact.interpreted_goal, "repair config loading failure");
        assert_eq!(artifact.ambiguities.len(), 1);
        assert_eq!(artifact.confidence, 0.68);
    }

    #[test]
    fn clarify_confidence_is_clamped() {
        let mut artifact = UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: TaskId("task-1".into()),
            original_request: "fix config loading".into(),
            interpreted_goal: "fix config loading".into(),
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
            created_at: datetime!(2026-05-01 12:00 UTC),
            updated_at: datetime!(2026-05-01 12:00 UTC),
        };

        super::ClarifyOutput {
            interpreted_goal: "fix config loading".into(),
            success_criteria: vec![],
            constraints: vec![],
            ambiguities: vec![],
            confidence: 1.8,
        }
        .apply_to(&mut artifact);

        assert_eq!(artifact.confidence, 1.0);
    }

    #[test]
    fn clarify_prompt_includes_observed_evidence() {
        let artifact = UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: TaskId("task-1".into()),
            original_request: "install remix here".into(),
            interpreted_goal: "install remix here".into(),
            success_criteria: vec![],
            constraints: vec![],
            target_scope: vec![],
            work_contract: Default::default(),
            evidence: vec![EvidenceRef {
                kind: EvidenceKind::Other,
                locator: "workspace-profile".into(),
                summary: "workspace profile evidence is available".into(),
            }],
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
            created_at: datetime!(2026-05-01 12:00 UTC),
            updated_at: datetime!(2026-05-01 12:00 UTC),
        };

        let prompt = super::clarify_user_prompt(&artifact, None);

        assert!(prompt.contains("observed_evidence"));
        assert!(prompt.contains("workspace-profile"));
        assert!(prompt.contains("workspace profile evidence is available"));
    }

    #[test]
    fn extracts_json_from_think_wrapped_response() {
        let input = "<think>\ninternal\n</think>\n\n{\"a\":1}";
        assert_eq!(extract_json_object(input), "{\"a\":1}");
    }

    #[test]
    fn understand_node_applies_grounded_output() {
        let provider = StubProvider;
        let node = UnderstandNode::new(&provider);
        let now = datetime!(2026-05-01 12:00 UTC);
        let mut artifact = UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: TaskId("task-1".into()),
            original_request: "wire the first onion loop".into(),
            interpreted_goal: "connect the onion loop".into(),
            success_criteria: vec![],
            constraints: vec!["keep the implementation lean".into()],
            target_scope: vec!["crates/shunt-core".into()],
            work_contract: Default::default(),
            evidence: vec![EvidenceRef {
                kind: EvidenceKind::File,
                locator: "crates/shunt-core/src/lib.rs".into(),
                summary: "file exists (100 bytes)".into(),
            }],
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

        let value = serde_json::json!({
            "interpreted_goal": "connect the task, runtime, and store loop using the scoped crates",
            "success_criteria": ["task state moves through the loop", "artifacts persist locally"],
            "target_scope": ["crates/shunt-core", "crates/shunt-runtime", "crates/shunt-store"],
            "ambiguities": [{"question": "should the loop stop at understand or reach execute?", "options": ["stop at understand", "reach execute"]}],
            "risks": [{"summary": "current evidence shows crate files, not execution semantics", "severity": "Medium"}],
            "confidence": 0.74
        });
        let output: super::UnderstandOutput = serde_json::from_value(value).unwrap();
        output.apply_to(&mut artifact);

        assert_eq!(
            artifact.interpreted_goal,
            "connect the task, runtime, and store loop using the scoped crates"
        );
        assert_eq!(artifact.target_scope.len(), 3);
        assert_eq!(artifact.ambiguities.len(), 1);
        assert_eq!(artifact.risks.len(), 1);
        assert_eq!(artifact.confidence, 0.74);
        let _ = node;
    }

    #[test]
    fn retry_on_invalid_output() {
        let count = 0usize;
        struct CountingProvider(std::cell::Cell<usize>);

        impl ToolProvider for CountingProvider {
            fn call_tool(
                &self,
                _system: &str,
                _user: &str,
                tool: &ToolSpec,
            ) -> super::InferResult<ToolCall> {
                let n = self.0.get();
                self.0.set(n + 1);
                // Return valid JSON but wrong shape for the first 2 calls, correct on 3rd.
                let value = if n < 2 {
                    serde_json::json!({"wrong": "shape"})
                } else {
                    serde_json::json!({
                        "interpreted_goal": "fixed on retry",
                        "success_criteria": [],
                        "constraints": [],
                        "ambiguities": [],
                        "confidence": 0.5
                    })
                };
                Ok(ToolCall {
                    name: tool.name.clone(),
                    arguments: value,
                })
            }
        }

        let provider = CountingProvider(std::cell::Cell::new(0));
        let schema =
            serde_json::to_value(schemars::schema_for!(super::ClarifyOutput)).unwrap_or_default();
        let result: super::InferResult<super::ClarifyOutput> =
            provider.generate_structured("sys", "usr", &schema);

        assert!(result.is_ok(), "should succeed on 3rd attempt");
        assert_eq!(result.unwrap().interpreted_goal, "fixed on retry");
        assert_eq!(provider.0.get(), 3);
        let _ = count;
    }

    #[test]
    fn exhausted_retries_returns_invalid_output_error() {
        struct AlwaysBadProvider;

        impl ToolProvider for AlwaysBadProvider {
            fn call_tool(
                &self,
                _system: &str,
                _user: &str,
                tool: &ToolSpec,
            ) -> super::InferResult<ToolCall> {
                Ok(ToolCall {
                    name: tool.name.clone(),
                    arguments: serde_json::json!({"wrong": "shape"}),
                })
            }
        }

        let schema =
            serde_json::to_value(schemars::schema_for!(super::ClarifyOutput)).unwrap_or_default();
        let result: super::InferResult<super::ClarifyOutput> =
            AlwaysBadProvider.generate_structured("sys", "usr", &schema);

        assert!(matches!(
            result,
            Err(super::InferError::InvalidOutput { retries: 3, .. })
        ));
    }

    fn request_json(mode: ToolChoiceMode) -> serde_json::Value {
        let provider = OpenAiCompatProvider::new("http://localhost:8080", "test-model")
            .with_capabilities(ProviderCapabilities {
                tool_choice_mode: mode,
                max_tokens: 32768,
                disable_thinking: false,
                thinking_budget_tokens: None,
                ..Default::default()
            });
        let request = provider.request_for(
            "system",
            "user",
            &ToolSpec {
                name: "output".into(),
                description: "structured output".into(),
                parameters: serde_json::json!({"type": "object"}),
            },
        );
        serde_json::to_value(request).unwrap()
    }

    fn request_agent_action_json() -> serde_json::Value {
        let provider = OpenAiCompatProvider::new("http://localhost:8080", "test-model")
            .with_capabilities(ProviderCapabilities {
                tool_choice_mode: ToolChoiceMode::RequiredString,
                max_tokens: 32768,
                ..Default::default()
            });
        let request = provider.request_for(
            "system",
            "user",
            &ToolSpec {
                name: "agent_action".into(),
                description: "structured output".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": ["read_file", "edit", "command", "done"]}
                    }
                }),
            },
        );
        serde_json::to_value(request).unwrap()
    }

    #[test]
    fn llama_cpp_tool_choice_is_a_required_string() {
        assert_eq!(
            request_json(ToolChoiceMode::RequiredString)["tool_choice"],
            serde_json::json!("required")
        );
    }

    #[test]
    fn named_object_tool_choice_targets_the_declared_tool() {
        assert_eq!(
            request_json(ToolChoiceMode::NamedObject)["tool_choice"],
            serde_json::json!("required")
        );
    }

    #[test]
    fn tool_choice_mode_does_not_change_single_request_path() {
        assert_eq!(
            request_json(ToolChoiceMode::Omit)["tool_choice"],
            serde_json::json!("required")
        );
        assert_eq!(
            request_json(ToolChoiceMode::JsonSchema)["tool_choice"],
            serde_json::json!("required")
        );
    }

    #[test]
    fn request_uses_strict_function_tools_not_response_format() {
        let req = request_json(ToolChoiceMode::JsonSchema);
        assert!(req.get("response_format").is_none());
        assert_eq!(req["tools"][0]["type"], "function");
        assert_eq!(req["tools"][0]["function"]["name"], "output");
        assert_eq!(req["tools"][0]["function"]["strict"], true);
        assert_eq!(
            req["tools"][0]["function"]["parameters"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn agent_action_expands_to_separate_function_tools() {
        let req = request_agent_action_json();
        let tools = req["tools"].as_array().unwrap();
        let names = tools
            .iter()
            .map(|tool| tool["function"]["name"].as_str().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["read_file", "edit", "command", "done"]);
        assert_eq!(
            tools[1]["function"]["parameters"]["required"],
            serde_json::json!(["path", "start_line", "end_line", "content"])
        );
        assert_eq!(
            tools[3]["function"]["parameters"]["required"],
            serde_json::json!(["description"])
        );
    }

    #[test]
    fn agent_action_arguments_wrap_selected_function_name() {
        let wrapped = super::agent_action_arguments("done", "{\"description\":\"ok\"}").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&wrapped).unwrap();

        assert_eq!(parsed["tool"], "done");
        assert_eq!(parsed["description"], "ok");
    }

    #[test]
    fn agent_action_history_uses_native_tool_messages() {
        let messages = vec![
            super::ChatMessage {
                role: "system".into(),
                content: "system".into(),
            },
            super::ChatMessage {
                role: "user".into(),
                content: "task".into(),
            },
            super::ChatMessage {
                role: "assistant".into(),
                content: r#"{"tool":"read_file","path":"src/lib.rs"}"#.into(),
            },
            super::ChatMessage {
                role: "user".into(),
                content: "loaded file".into(),
            },
        ];
        let tool = ToolSpec {
            name: "agent_action".into(),
            description: "structured output".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": ["read_file"]}
                }
            }),
        };

        let native = super::native_chat_messages(&messages, &tool);

        assert_eq!(native[2]["role"], "assistant");
        assert_eq!(native[2]["content"], serde_json::Value::Null);
        assert_eq!(native[2]["tool_calls"][0]["function"]["name"], "read_file");
        assert_eq!(native[3]["role"], "tool");
        assert_eq!(native[3]["name"], "read_file");
        assert_eq!(native[3]["tool_call_id"], native[2]["tool_calls"][0]["id"]);
    }

    #[test]
    fn json_schema_request_uses_provider_max_tokens() {
        let req = request_json(ToolChoiceMode::JsonSchema);
        assert_eq!(req["max_tokens"], 32768);
    }

    #[test]
    fn tool_schema_is_flat_strict_and_openai_compatible() {
        let schema = serde_json::json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "definitions": {
                "Request": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "freshness_required": {"type": "boolean", "default": false}
                    },
                    "required": ["summary"]
                }
            },
            "type": "object",
            "title": "KnowledgePlanOutput",
            "properties": {
                "should_research": {"type": "boolean"},
                "rationale": {"type": "string", "default": ""},
                "requests": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/Request"}
                }
            },
            "required": ["should_research"]
        });

        let normalized = super::strict_tool_schema(&schema);

        assert!(normalized.get("$schema").is_none());
        assert!(normalized.get("definitions").is_none());
        assert_eq!(normalized["additionalProperties"], false);
        assert_eq!(
            normalized["required"],
            serde_json::json!(["rationale", "requests", "should_research"])
        );
        assert_eq!(
            normalized["properties"]["requests"]["items"]["additionalProperties"],
            false
        );
        assert_eq!(
            normalized["properties"]["requests"]["items"]["required"],
            serde_json::json!(["freshness_required", "summary"])
        );
        assert!(
            normalized["properties"]["rationale"]
                .get("default")
                .is_none()
        );
    }

    #[test]
    fn extract_json_strips_markdown_fence() {
        let fenced = "```json\n{\"a\":1}\n```";
        assert_eq!(super::extract_json_object(fenced), "{\"a\":1}");
    }

    #[test]
    fn extract_json_strips_plain_fence() {
        let fenced = "```\n{\"a\":1}\n```";
        assert_eq!(super::extract_json_object(fenced), "{\"a\":1}");
    }

    #[test]
    fn normalize_chat_completion_prefers_tool_calls_when_present() {
        let normalized = normalize_chat_completion_message(
            Some(AssistantMessage {
                content: Some("ignored".into()),
                reasoning_content: Some("thoughts".into()),
                tool_calls: Some(vec![RespToolCall {
                    function: RespFunction {
                        name: "output".into(),
                        arguments: "{\"ok\":true}".into(),
                    },
                }]),
            }),
            "output",
            ToolChoiceMode::RequiredString,
        )
        .unwrap();

        assert_eq!(normalized.thinking.as_deref(), Some("thoughts"));
        assert_eq!(
            normalized.tool_call.unwrap().arguments_json,
            "{\"ok\":true}"
        );
    }

    #[test]
    fn normalize_chat_completion_requires_real_tool_call() {
        let normalized = normalize_chat_completion_message(
            Some(AssistantMessage {
                content: Some("<think>hidden</think>{\"ok\":true}".into()),
                reasoning_content: None,
                tool_calls: None,
            }),
            "output",
            ToolChoiceMode::JsonSchema,
        )
        .unwrap();

        assert!(normalized.tool_call.is_none());
    }

    #[test]
    fn ollama_detect_uses_chat_completion_tools() {
        let caps = super::ProviderCapabilities::detect("mistral", "http://127.0.0.1:11434/v1");
        assert_eq!(caps.preferred_transport, OpenAiTransport::ChatCompletions);
        assert_eq!(caps.tool_choice_mode, ToolChoiceMode::RequiredString);
    }

    #[test]
    fn thinking_model_detect_uses_tool_calling_when_supported() {
        let caps =
            super::ProviderCapabilities::detect("google/gemma3-9b-it", "http://127.0.0.1:8080/v1");
        assert_eq!(caps.tool_choice_mode, ToolChoiceMode::RequiredString);
        assert_eq!(caps.preferred_transport, OpenAiTransport::ChatCompletions);
    }
}
