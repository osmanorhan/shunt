# shunt

A coding agent built and tested against local language models.

Local models work differently from cloud models. Context windows are smaller, tool use is less reliable, and models built for reasoning can exhaust their token budget before writing a single line of output. Running an agent built for GPT-4 against a local 14B model will expose all of these gaps.

Local models have small context windows and source files are long. If the agent reads the wrong files, or reads full files when only a few lines matter, it burns through context and turns before it gets to the actual work. shunt builds a workspace index combining lexical search with tree-sitter code structure, detects what kind of change is being asked for, and gives the model the relevant snippets ranked by role rather than full file dumps. This keeps context spend low and leaves room for the actual edit.

Grammar-constrained decoding forces valid tool call output on every turn regardless of model size. The model registry detects which model is running and applies the right configuration for that family, including how to handle thinking budgets on reasoning models.

shunt is developed and benchmarked exclusively against local models. Every part of the agent loop is built around the failure points those models actually have.

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

| Model | Quant | VRAM |
|-------|-------|------|
| `gemma-4-12b-it` | UD-Q4_K_XL | ~8 GB |
| `gemma-4-26B-A4B-it` | UD-Q4_K_M | ~17 GB |

```sh
# 12B
llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL \
  --jinja -ngl 999 -fa -c 8192

# 26B-A4B (MoE — larger but only 4B active parameters)
llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M \
  --jinja -ngl 999 -fa -c 8192
```

```toml
endpoint = "http://localhost:8080"
model    = "gemma-4-12b"
```

### Qwen (Alibaba)

| Model | Quant | VRAM |
|-------|-------|------|
| `Qwen3.5-9B` | UD-Q4_K_XL | ~6.5 GB |
| `Qwen3.6-27B` | Q4_K_S | ~16 GB |
| `Qwen3.6-35B-A3B` | UD-Q4_K_M | ~22 GB |

The 35B-A3B is a mixture-of-experts model — 35B total parameters, 3B active per forward pass.

```sh
# Qwen3.5-9B (~6.5 GB)
llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL \
  --jinja -ngl 999 -fa -c 8192

# Qwen3.6-27B (~16 GB)
llama-server -hf unsloth/Qwen3.6-27B-GGUF:Q4_K_S \
  --jinja -ngl 999 -fa -c 8192

# Qwen3.6-35B-A3B MoE (~22 GB)
llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --jinja -ngl 999 -fa -c 8192
```

```toml
endpoint = "http://localhost:8080"
model    = "qwen3.6-27b"   # or qwen3.5-9b
```

### Key server flags

| Flag | Purpose |
|------|---------|
| `-hf <repo:quant>` | Download and run directly from Hugging Face |
| `--jinja` | Jinja2 chat template — required for correct tool-call formatting |
| `-ngl 999` | Offload all layers to GPU |
| `-fa` | Flash attention — significant speedup on CUDA |
| `-c 8192` | Context window size |
| `-sm layer` | Tensor parallel across multiple GPUs |

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
model    = "gemma-4-12b"

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
