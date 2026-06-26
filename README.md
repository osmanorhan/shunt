# shunt

A coding agent for local language models.

```sh
shunt agent "add a rate-limit middleware to the Express router"
```

shunt indexes your workspace, reads the relevant files, and edits them through a tool-use loop — pausing when it needs input from you.

- **Grammar-constrained tool calls** — JSON schema grammar produces valid structured output on every turn
- **Model registry** — auto-detects Gemma 4, Qwen 3, DeepSeek R1, Mistral and applies the right decoding strategy per family
- **Configurable thinking budget** — per-model-family reasoning token allocation
- **Workspace search** — hybrid lexical + semantic index over your codebase
- **Fully local** — no telemetry, no cloud calls, no API keys

---

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/osmanorhan/shunt/main/scripts/install.sh | sh
```

Installs to `~/.local/bin/shunt`. Override with `SHUNT_INSTALL_DIR=/usr/local/bin`.

To install a specific version:
```sh
SHUNT_VERSION=v0.2.0 curl -fsSL .../install.sh | sh
```

| Platform | Architecture |
|----------|-------------|
| Linux | x86_64, aarch64 (statically linked) |
| macOS | Apple Silicon, Intel |

---

## Requirements

A local LLM server with an OpenAI-compatible API:

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** (`llama-server`) — recommended
- **[Ollama](https://ollama.com)**
- Any OpenAI-compatible endpoint

---

## Recommended models

### Gemma 4 (Google)

| Model | VRAM |
|-------|------|
| `gemma-4-12b-it` | ~10 GB |
| `gemma-4-27b-it` | ~20 GB |

Download: [unsloth/gemma-4-12b-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-qat-GGUF)

```sh
llama-server \
  --model gemma-4-12b-it-Q4_K_M.gguf \
  --port 8080 --ctx-size 16384 \
  --flash-attn on --jinja --reasoning off
```

```toml
endpoint = "http://localhost:8080"
model    = "gemma-4-12b"
```

### Qwen 3.6 (Alibaba)

| Model | VRAM |
|-------|------|
| `Qwen3.6-7B` | ~6 GB |
| `Qwen3.6-27B` | ~18 GB |

Download: [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)

```sh
llama-server \
  --model Qwen3.6-27B-Q4_K_M.gguf \
  --port 8080 --ctx-size 16384 \
  --flash-attn on --jinja \
  -sm layer   # spans two GPUs; omit for single card
```

```toml
endpoint = "http://localhost:8080"
model    = "qwen3.6-27b"
```

### Key server flags

| Flag | Purpose |
|------|---------|
| `--jinja` | Jinja2 chat template — required for tool-call formatting |
| `--reasoning off` | Disable server-side reasoning tokens (Gemma-4) |
| `--flash-attn on` | Flash attention — significant speedup on CUDA |
| `--ctx-size 16384` | Context window; 16K covers most tasks |
| `--spec-type draft-mtp --spec-draft-n-max 2` | Multi-token prediction: 1.4–2.2× speedup (Qwen3) |

---

## Setup

```sh
cd your-project
shunt config init
```

Creates `.shunt/config.toml`:

```toml
endpoint = "http://localhost:8080"
model    = "gemma-4-12b"
```

---

## Usage

```sh
# Interactive TUI
shunt

# One-shot
shunt agent "migrate the Prisma schema to add a nullable deletedAt column"
```

---

## How it works

The agent receives your workspace as context and calls tools to explore and edit it:

| Tool | What it does |
|------|-------------|
| `read_file` | Read a file |
| `search_files` | Search the workspace index |
| `replace_lines` | Edit a line range (replace, append, or delete) |
| `write_file` | Create a new file |
| `delete_file` | Remove a file |
| `run_command` | Run a shell command |
| `ask_user` | Ask a clarification question |
| `done` | Signal task complete |

The workspace index (`.shunt/index/`) is built on first run and updated incrementally.

---

## Configuration

`.shunt/config.toml`:

```toml
endpoint = "http://localhost:8080"
model    = "gemma4-12b"

[agent]
max_turns       = 24
thinking_budget = 8192
```

---

## Building from source

```sh
git clone https://github.com/osmanorhan/shunt
cd shunt
cargo build --release -p shunt-cli
./target/release/shunt --help
```

Requires Rust 1.85+.

---

## License

MIT
