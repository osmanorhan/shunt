# shunt

A coding agent built for local language models.

shunt is designed around the limits of local models instead of assuming cloud-model behavior.

* **Uses less context** — indexes your workspace and gives the model relevant snippets instead of dumping full files.
* **Finds better edit targets** — combines lexical search with tree-sitter code structure to rank files and symbols by relevance.
* **Handles smaller models reliably** — uses grammar-constrained decoding so tool calls stay valid even on local 9B–14B models.
* **Works with reasoning models** — manages thinking budgets so models do not burn all tokens before producing an answer or edit.
* **Adapts to model families** — detects the configured model and applies the right settings for Gemma, Qwen, and other supported local models.
* **Built for local-first workflows** — developed and benchmarked against local inference servers, not cloud APIs.

---

## Install

**Linux / macOS**

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/osmanorhan/shunt/releases/latest/download/shunt-cli-installer.sh | sh
```

**Windows** (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://github.com/osmanorhan/shunt/releases/latest/download/shunt-cli-installer.ps1 | iex"
```

| Platform | Architecture         |
| -------- | -------------------- |
| Linux    | x86_64, aarch64      |
| macOS    | Apple Silicon, Intel |
| Windows  | x86_64               |

---

## Requirements

A local LLM server with an OpenAI-compatible API:

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** (`llama-server`) — recommended
- **[Ollama](https://ollama.com)**
- Any OpenAI-compatible endpoint

---

## Recommended models

### Gemma 4

| Model                | Quant      | VRAM   |
| -------------------- | ---------- | ------ |
| `gemma-4-12b-it`     | UD-Q4_K_XL | ~8 GB  |
| `gemma-4-26B-A4B-it` | UD-Q4_K_M  | ~17 GB |

```sh
# 12B
llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL \
  --jinja -ngl 999 -fa on

# 26B-A4B
llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M \
  --jinja -ngl 999 -fa on
```

```toml
endpoint = "http://localhost:8080"
model    = "gemma-4-12b-it"
```

### Qwen

| Model             | Quant      | VRAM    |
| ----------------- | ---------- | ------- |
| `Qwen3.5-9B`      | UD-Q4_K_XL | ~6.5 GB |
| `Qwen3.6-27B`     | Q4_K_S     | ~16 GB  |
| `Qwen3.6-35B-A3B` | UD-Q4_K_M  | ~22 GB  |

The 35B-A3B model is a mixture-of-experts model: 35B total parameters, with 3B active per forward pass.


### Qwen

| Model | Quant | VRAM |
|-------|-------|------|
| `Qwen3.5-9B` | UD-Q4_K_XL | ~6.5 GB |
| `Qwen3.6-27B` | Q4_K_S | ~16 GB |
| `Qwen3.6-35B-A3B` | UD-Q4_K_M | ~22 GB |

The 35B-A3B is a mixture-of-experts model — 35B total parameters, 3B active per forward pass.

```sh
# Qwen3.5-9B (~6.5 GB)
llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL \
  --jinja -ngl 999 -fa on

# Qwen3.6-27B (~16 GB)
llama-server -hf unsloth/Qwen3.6-27B-GGUF:Q4_K_S \
  --jinja -ngl 999 -fa on

# Qwen3.6-35B-A3B MoE (~22 GB)
llama-server -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M \
  --jinja -ngl 999 -fa on
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
| `-fa on` | Flash attention — significant speedup on CUDA |
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
| `edit` | Create a new file (no `start_line`) or modify an existing one by line range |
| `command` | Run a shell command |
| `knowledge` | Query dependency/version evidence on demand |
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

Enable the repo's pre-commit checks with:

```sh
./scripts/install-git-hooks.sh
```

That hook runs `cargo fmt --all` and `cargo clippy --workspace --all-targets --locked -- -D warnings` before each commit.

---

## License

MIT
