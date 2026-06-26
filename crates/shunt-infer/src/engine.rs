/// The type of local inference server behind an endpoint URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// llama-server / llama.cpp (port 8080 default; /v1/chat/completions + grammar).
    LlamaCpp,
    /// Ollama native API (port 11434 default; /api/chat + /v1 shim).
    Ollama,
    /// vLLM (/v1/chat/completions; port 8000 default).
    Vllm,
    /// Any other OpenAI-compatible endpoint.
    Generic,
}

/// Detect the inference engine from an endpoint URL using heuristics.
///
/// Detection is purely lexical — no network probe is performed.
/// Extend by adding patterns in the appropriate branch; never modify caller code.
pub fn detect_engine(endpoint: &str) -> EngineKind {
    let e = endpoint.to_ascii_lowercase();
    if e.contains(":11434") || e.contains("ollama") {
        return EngineKind::Ollama;
    }
    if e.contains(":8000") || e.contains("vllm") {
        return EngineKind::Vllm;
    }
    if e.contains(":8080") || e.contains("llama") {
        return EngineKind::LlamaCpp;
    }
    EngineKind::Generic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_port_detected() {
        assert_eq!(detect_engine("http://127.0.0.1:11434"), EngineKind::Ollama);
    }

    #[test]
    fn ollama_name_in_url_detected() {
        assert_eq!(detect_engine("http://ollama.local/v1"), EngineKind::Ollama);
    }

    #[test]
    fn llama_cpp_port_detected() {
        assert_eq!(detect_engine("http://127.0.0.1:8080"), EngineKind::LlamaCpp);
    }

    #[test]
    fn vllm_port_detected() {
        assert_eq!(detect_engine("http://127.0.0.1:8000"), EngineKind::Vllm);
    }

    #[test]
    fn generic_fallback() {
        assert_eq!(detect_engine("http://127.0.0.1:9999"), EngineKind::Generic);
    }
}
