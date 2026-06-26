//! Model catalog — **data**, not code. A model is one TOML entry; the benchmark
//! reads the catalog and runs every model through the suite. To add a model, add a
//! `[[model]]` row. To add a whole inference engine, add one `Engine` arm + one
//! match arm in `crate::capability::run_model` — nothing else changes.

use serde::Deserialize;

/// Inference engine = which provider speaks to the endpoint. llama.cpp and vLLM
/// are both OpenAI-compatible; Ollama uses its native `/api/chat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    #[serde(alias = "llama.cpp", alias = "llamacpp")]
    LlamaCpp,
    Vllm,
    Ollama,
}

impl Engine {
    pub fn label(self) -> &'static str {
        match self {
            Engine::LlamaCpp => "llama.cpp",
            Engine::Vllm => "vllm",
            Engine::Ollama => "ollama",
        }
    }
    /// OpenAI-compatible engines share one provider; only Ollama differs.
    pub fn is_openai_compatible(self) -> bool {
        matches!(self, Engine::LlamaCpp | Engine::Vllm)
    }
}

/// One catalogued model.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    /// Display name in the scorecard.
    pub name: String,
    pub engine: Engine,
    /// Base URL, e.g. `http://127.0.0.1:8080`.
    pub endpoint: String,
    /// Model id sent in the request (llama.cpp ignores it; vLLM/Ollama need it).
    pub model_id: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_timeout() -> u64 {
    300
}

/// The whole catalog file.
#[derive(Debug, Clone, Deserialize)]
pub struct Catalog {
    #[serde(default)]
    pub model: Vec<ModelSpec>,
}

impl Catalog {
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }
}
