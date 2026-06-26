---
title: Home
layout: home
nav_order: 1
---

# shunt

A coding agent built for local language models.
{: .fs-6 .fw-300 }

[Get started](#quick-start){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/osmanorhan/shunt){: .btn .fs-5 .mb-4 .mb-md-0 }

---

shunt runs a tool-use agent loop against a local LLM server. It indexes your workspace, finds relevant files, and edits them — pausing only when it needs input from you.

- **Grammar-constrained tool calls** — JSON schema grammar produces valid structured output on every turn
- **Model registry** — auto-detects Gemma 4, Qwen 3, and other supported local models and applies the right decoding strategy per family
- **Configurable thinking budget** — per-model-family reasoning token allocation
- **Workspace search** — hybrid lexical + semantic index over your codebase
- **Fully local** — no telemetry, no cloud calls, no API keys

---

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/osmanorhan/shunt/main/scripts/install.sh | sh
```

Installs to `~/.local/bin/shunt`.

| Platform | Architecture |
|----------|-------------|
| Linux | x86_64, aarch64 (statically linked) |
| macOS | Apple Silicon, Intel |

---

## Quick start

**1. Start a local model server**

```sh
# Gemma 4 12B (~8 GB VRAM)
llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL \
  --jinja -ngl 999 -fa on
```

**2. Configure shunt in your project**

```sh
cd your-project
shunt config init
```

Creates `.shunt/config.toml`:

```toml
endpoint = "http://localhost:8080"
model    = "gemma-4-12b"
```

**3. Run a task**

```sh
# Interactive TUI
shunt

# One-shot
shunt agent "add a rate-limit middleware to the Express router"
```

---

## Requirements

A local LLM server with an OpenAI-compatible API:

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** (`llama-server`) — recommended
- **[Ollama](https://ollama.com)**
- Any OpenAI-compatible endpoint

---

## Building from source

```sh
git clone https://github.com/osmanorhan/shunt
cd shunt
cargo build --release -p shunt-cli
./target/release/shunt --help
```

Requires Rust 1.85+.
