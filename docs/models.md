---
title: Models
layout: default
nav_order: 3
---

# Supported models
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

shunt auto-detects the loaded model from the server's `/v1/models` response and applies the appropriate decoding strategy, sampling parameters, and thinking configuration for that model family. No manual tuning is required.

---

## Gemma 4 (Google)

**Recommended default.** The 12B variant runs on ~8 GB VRAM.

| Model | Quant | VRAM |
|-------|-------|------|
| `gemma-4-12b-it` | UD-Q4_K_XL | ~8 GB |
| `gemma-4-26B-A4B-it` | UD-Q4_K_M | ~17 GB |

The 26B-A4B is a mixture-of-experts model — 26B total parameters, 4B active per forward pass.

### Server command

```sh
# 12B
llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL \
  --jinja -ngl 999 -fa on -c 8192

# 26B-A4B
llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M \
  --jinja -ngl 999 -fa on -c 8192
```

### shunt config

```toml
endpoint = "http://localhost:8080"
model    = "gemma-4-12b"
```

### Sampling profile

| Parameter | Value |
|-----------|-------|
| `max_tokens` | 8192 |
| `thinking_budget_tokens` | 2048 |
| `temperature` | 1.0 |
| `top_p` | 0.95 |
| `top_k` | 64 |

---

## Qwen (Alibaba)

Three practical options covering 6.5 GB to 22 GB VRAM. The 35B-A3B is a mixture-of-experts model with only 3B active parameters per forward pass.

| Model | Quant | VRAM |
|-------|-------|------|
| `Qwen3.5-9B` | UD-Q4_K_XL | ~6.5 GB |
| `Qwen3.6-27B` | Q4_K_S | ~16 GB |
| `Qwen3.6-35B-A3B` | UD-Q4_K_M | ~22 GB |

### Server command

```sh
# Qwen3.5-9B
llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL \
  --jinja -ngl 999 -fa on -c 8192

# Qwen3.6-27B
llama-server -hf unsloth/Qwen3.6-27B-GGUF:Q4_K_S \
  --jinja -ngl 999 -fa on -c 8192

# Qwen3.6-35B-A3B (MoE)
llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --jinja -ngl 999 -fa on -c 8192
```

### shunt config

```toml
endpoint = "http://localhost:8080"
model    = "qwen3.6-27b"   # or qwen3.5-9b
```

### Sampling profile

| Parameter | Value |
|-----------|-------|
| `max_tokens` | 32768 |
| `temperature` | 0.7 |
| `content_temperature` | 0.6 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `presence_penalty` | 1.5 |

---

## Key server flags

| Flag | Purpose |
|------|---------|
| `-hf <repo:quant>` | Download and run directly from Hugging Face |
| `--jinja` | Jinja2 chat template — required for tool-call formatting |
| `-ngl 999` | Offload all layers to GPU |
| `-fa` | Flash attention — significant speedup on CUDA |
| `-c 8192` | Context window size |
| `-sm layer` | Tensor parallel across multiple GPUs |

---

## Adding a custom model

Implement `ModelMatcher` in `crates/shunt-infer/src/registry.rs`:

```rust
pub struct MyModelMatcher;
impl ModelMatcher for MyModelMatcher {
    fn match_model(&self, id: &str) -> Option<ModelProfile> {
        id.contains("my-model").then_some(ModelProfile {
            tool_choice_mode: ToolChoiceMode::JsonSchema,
            max_tokens: 8192,
            temperature: 0.7,
            ..Default::default()
        })
    }
}
```

Register it in `ModelRegistry::with_defaults()`:

```rust
registry.register(MyModelMatcher)
```
