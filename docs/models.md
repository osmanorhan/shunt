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

**Recommended default.** The 12B QAT variant runs on ~10 GB VRAM.

| Model | VRAM |
|-------|------|
| `gemma-4-12b-it` | ~10 GB |
| `gemma-4-27b-it` | ~20 GB |

Download: [unsloth/gemma-4-12b-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-qat-GGUF)

### Server command

```sh
llama-server \
  --model gemma-4-12b-it-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 16384 \
  --flash-attn on \
  --jinja \
  --reasoning off
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

## Qwen 3.6 (Alibaba)

Stronger on multi-step reasoning tasks. The 27B model fits on two 16 GB cards or a single 24 GB card.

| Model | VRAM |
|-------|------|
| `Qwen3.6-7B` | ~6 GB |
| `Qwen3.6-27B` | ~18 GB |

Download: [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)

### Server command

```sh
# Single GPU
llama-server \
  --model Qwen3.6-27B-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 16384 \
  --flash-attn on \
  --jinja

# Two GPUs (tensor parallel)
llama-server \
  --model Qwen3.6-27B-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 32768 \
  --flash-attn on \
  --jinja \
  -sm layer
```

### shunt config

```toml
endpoint = "http://localhost:8080"
model    = "qwen3.6-27b"
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
| `--jinja` | Jinja2 chat template — required for correct tool-call formatting |
| `--reasoning off` | Disable server-side reasoning tokens (Gemma-4) |
| `--flash-attn on` | Flash attention — significant speedup on CUDA |
| `--ctx-size 16384` | Context window — 16K covers most tasks |
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
