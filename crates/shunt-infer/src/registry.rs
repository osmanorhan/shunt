use crate::ToolChoiceMode;

/// Capability profile for a known local model family.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub tool_choice_mode: ToolChoiceMode,
    pub max_tokens: u32,
    pub supports_thinking: bool,
    /// When true, prepend `/no_think` to every user message so the model
    /// skips its reasoning chain and outputs directly.  Qwen3 respects this
    /// instruction; other models ignore the prefix harmlessly.
    pub disable_thinking: bool,
    /// llama.cpp `budget_tokens`: max thinking tokens before forced output.
    /// None = let the server decide (default: unrestricted within max_tokens).
    pub thinking_budget_tokens: Option<u32>,
    /// Temperature for action-selection calls (grammar-constrained routing).
    /// Qwen3 non-thinking/coding = 0.7, Gemma-4 = 1.0, unknown = 0.7.
    pub temperature: f32,
    /// Temperature for content-generation calls (call_text — writing actual code).
    /// Qwen3 thinking/precise-coding = 0.6. Falls back to `temperature` if None.
    pub content_temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    /// Presence penalty discourages repeating any previously-seen token.
    /// Recommended 1.5 for Qwen3 non-thinking mode to prevent action loops.
    pub presence_penalty: Option<f32>,
    /// Repetition penalty (multiplicative, llama.cpp extension).
    pub repetition_penalty: Option<f32>,
    /// When true, send enable_thinking:false on content-generation calls.
    /// Required for Gemma-4: without it the model exhausts max_tokens in
    /// reasoning_content and returns empty output (F7).
    /// Qwen3: false — let the model reason about the code it's about to write.
    pub suppress_content_thinking: bool,
}

impl Default for ModelProfile {
    fn default() -> Self {
        Self {
            tool_choice_mode: ToolChoiceMode::RequiredString,
            max_tokens: 32768,
            supports_thinking: false,
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

/// Extension point for adding support for a new model family.
/// Implement this trait and register via `ModelRegistry::register`.
pub trait ModelMatcher: Send + Sync {
    fn match_model(&self, model_id: &str) -> Option<ModelProfile>;
}

/// Registry of model-family matchers. First match wins.
/// Extend with `register` — never modify existing matchers.
pub struct ModelRegistry {
    matchers: Vec<Box<dyn ModelMatcher>>,
}

impl ModelRegistry {
    /// Create a registry populated with all built-in local-model matchers.
    pub fn with_defaults() -> Self {
        Self { matchers: vec![] }
            .register(Qwen3Matcher)
            .register(Qwen2Matcher)
            .register(Gemma4Matcher)
            .register(Gemma3Matcher)
            .register(GemmaMatcher)
            .register(MistralMatcher)
            .register(DeepSeekR1Matcher)
            .register(DeepSeekMatcher)
            .register(GlmMatcher)
    }

    pub fn register(mut self, m: impl ModelMatcher + 'static) -> Self {
        self.matchers.push(Box::new(m));
        self
    }

    /// Resolve a model ID to a profile. Falls back to `Default` if no matcher fires.
    pub fn resolve(&self, model_id: &str) -> ModelProfile {
        let id = model_id.to_ascii_lowercase();
        self.matchers
            .iter()
            .find_map(|m| m.match_model(&id))
            .unwrap_or_default()
    }
}

fn any(id: &str, pats: &[&str]) -> bool {
    pats.iter().any(|p| id.contains(p))
}

/// Qwen3 / QwQ — thinking models; use json_schema grammar decoding.
/// Thinking disabled via `/no_think`: responds in seconds instead of minutes.
/// Sampling: Unsloth non-thinking instruct mode (temp=0.7, top-p=0.8, top-k=20,
/// presence_penalty=1.5 to break action-repetition loops).
pub struct Qwen3Matcher;
impl ModelMatcher for Qwen3Matcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        any(id, &["qwen3", "qwq", "qwen-3"]).then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::JsonSchema,
            max_tokens: 32768,
            supports_thinking: true,
            disable_thinking: true, // /no_think on action-selection calls only
            thinking_budget_tokens: Some(512), // ~51s at 10 tok/s; 2048 exceeds 180s timeout
            temperature: 0.7,       // non-thinking instruct mode for routing
            content_temperature: Some(0.6), // thinking/precise-coding mode for code gen
            top_p: Some(0.8),
            top_k: Some(20),
            min_p: Some(0.0),
            presence_penalty: Some(1.5),
            repetition_penalty: None,
            suppress_content_thinking: true, // budget_tokens ignored; thinking takes 180s+ → timeout
        })
    }
}

/// Qwen2.x — non-thinking; use tool_choice required string.
pub struct Qwen2Matcher;
impl ModelMatcher for Qwen2Matcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        any(id, &["qwen2", "qwen1.5", "qwen-2", "qwen-1"]).then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::RequiredString,
            max_tokens: 8192,
            supports_thinking: false,
            disable_thinking: false,
            thinking_budget_tokens: None,
            ..Default::default()
        })
    }
}

/// Gemma 4 — thinking model; use json_schema grammar decoding.
/// Thinking suppressed via grammar (forced JSON from token 1); /no_think ignored.
/// Sampling: Google/Unsloth defaults (temp=1.0, top-p=0.95, top-k=64).
/// budget_tokens=2048 limits thinking to ~60s, leaving 6144 tokens for output.
pub struct Gemma4Matcher;
impl ModelMatcher for Gemma4Matcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        any(id, &["gemma-4", "gemma4", "gemma_4"]).then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::JsonSchema,
            max_tokens: 8192,
            supports_thinking: true,
            disable_thinking: false,
            thinking_budget_tokens: Some(2048),
            temperature: 1.0,
            content_temperature: None, // same 1.0 for content
            top_p: Some(0.95),
            top_k: Some(64),
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            suppress_content_thinking: true, // F7: empty output without this
        })
    }
}

/// Gemma 3 — thinking model; use json_schema grammar decoding.
/// Same sampling defaults as Gemma-4.
pub struct Gemma3Matcher;
impl ModelMatcher for Gemma3Matcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        any(id, &["gemma3", "gemma-3", "gemma_3"]).then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::JsonSchema,
            max_tokens: 8192,
            supports_thinking: true,
            disable_thinking: false,
            thinking_budget_tokens: Some(2048),
            temperature: 1.0,
            content_temperature: None,
            top_p: Some(0.95),
            top_k: Some(64),
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            suppress_content_thinking: true,
        })
    }
}

/// Gemma (other) — ignores tool_choice; use json_schema.
pub struct GemmaMatcher;
impl ModelMatcher for GemmaMatcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        id.contains("gemma").then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::JsonSchema,
            max_tokens: 8192,
            supports_thinking: false,
            disable_thinking: false,
            thinking_budget_tokens: None,
            temperature: 1.0,
            top_p: Some(0.95),
            top_k: Some(64),
            ..Default::default()
        })
    }
}

/// Mistral / Mixtral — use tool_choice required string.
pub struct MistralMatcher;
impl ModelMatcher for MistralMatcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        any(id, &["mistral", "mixtral"]).then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::RequiredString,
            max_tokens: 8192,
            ..Default::default()
        })
    }
}

/// DeepSeek-R1 / R1-Distill — thinking model; use json_schema.
pub struct DeepSeekR1Matcher;
impl ModelMatcher for DeepSeekR1Matcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        any(id, &["deepseek-r1", "r1-distill", "deepseek_r1"]).then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::JsonSchema,
            max_tokens: 8192,
            supports_thinking: true,
            thinking_budget_tokens: Some(2048),
            ..Default::default()
        })
    }
}

/// DeepSeek other (coder, v2, v3) — non-thinking; required string.
pub struct DeepSeekMatcher;
impl ModelMatcher for DeepSeekMatcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        id.contains("deepseek").then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::RequiredString,
            max_tokens: 8192,
            ..Default::default()
        })
    }
}

/// GLM-4 — non-thinking; required string.
pub struct GlmMatcher;
impl ModelMatcher for GlmMatcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        any(id, &["glm4", "glm-4", "glm_4"]).then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::RequiredString,
            max_tokens: 8192,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_resolves_to_json_schema() {
        let reg = ModelRegistry::with_defaults();
        let p = reg.resolve("unsloth/Qwen3.5-9B-GGUF:Q6_K");
        assert_eq!(p.tool_choice_mode, ToolChoiceMode::JsonSchema);
        assert!(p.supports_thinking);
        assert!(p.disable_thinking);
    }

    #[test]
    fn qwen2_resolves_to_required_string() {
        let reg = ModelRegistry::with_defaults();
        let p = reg.resolve("Qwen2.5-7B-Instruct");
        assert_eq!(p.tool_choice_mode, ToolChoiceMode::RequiredString);
        assert!(!p.supports_thinking);
    }

    #[test]
    fn deepseek_r1_resolves_to_json_schema() {
        let reg = ModelRegistry::with_defaults();
        let p = reg.resolve("deepseek-r1-distill-qwen-7b");
        assert_eq!(p.tool_choice_mode, ToolChoiceMode::JsonSchema);
    }

    #[test]
    fn gemma3_resolves_to_json_schema() {
        let reg = ModelRegistry::with_defaults();
        let p = reg.resolve("google/gemma3-9b-it");
        assert_eq!(p.tool_choice_mode, ToolChoiceMode::JsonSchema);
        assert!(p.supports_thinking);
    }

    #[test]
    fn gemma4_resolves_to_json_schema_with_thinking() {
        let reg = ModelRegistry::with_defaults();
        let p = reg.resolve("unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL");
        assert_eq!(p.tool_choice_mode, ToolChoiceMode::JsonSchema);
        assert!(p.supports_thinking);
        assert!(!p.disable_thinking); // /no_think doesn't work; grammar suppresses thinking
        assert_eq!(p.max_tokens, 8192);
    }

    #[test]
    fn unknown_model_falls_back_to_default() {
        let reg = ModelRegistry::with_defaults();
        let p = reg.resolve("some-unknown-model-xyz");
        assert_eq!(p.tool_choice_mode, ToolChoiceMode::RequiredString);
    }

    #[test]
    fn custom_matcher_can_be_registered() {
        struct MyMatcher;
        impl ModelMatcher for MyMatcher {
            fn match_model(&self, id: &str) -> Option<ModelProfile> {
                id.contains("custom").then_some(ModelProfile {
                    tool_choice_mode: ToolChoiceMode::JsonSchema,
                    max_tokens: 4096,
                    supports_thinking: false,
                    disable_thinking: false,
                    thinking_budget_tokens: None,
                    ..Default::default()
                })
            }
        }

        let reg = ModelRegistry::with_defaults().register(MyMatcher);
        let p = reg.resolve("my-custom-model-v1");
        assert_eq!(p.tool_choice_mode, ToolChoiceMode::JsonSchema);
        assert_eq!(p.max_tokens, 4096);
    }
}
