---
title: Benchmarks
layout: default
nav_order: 2
---

# Benchmarks
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Capability suite

The capability suite is a task-level benchmark that runs shunt against a real workspace and verifies the edit it produces. Each task provides a concrete code-change request, a small TypeScript workspace, and a deterministic check on the resulting file.

Tasks are classified by difficulty:

| Difficulty | What it tests |
|-----------|--------------|
| **trivial** | Single constant or identifier change |
| **easy** | Add a short function in the right place |
| **medium** | Conditional branch insert, delete, or modification |
| **hard** | Multi-site rename, cross-file change, error handling refactor |

### Current status

10-task suite, both models at 100%:

| Model | Size | Overall | Avg time (hard tasks) |
|-------|------|---------|-----------------------|
| Qwen3.6-27B Q4_K_S | 27B | 10/10 | ~51 s |
| Gemma-4 12B QAT | 12B | 10/10 | ~87 s |

This suite is saturated at the current model sizes — see the pure-bench suite below for tasks that still discriminate.

---

## Methodology

### Task format

Each task provides:
- A `request` string — the task as a user would type it
- Source files written to a fresh temp workspace before the run
- A check closure run on the resulting file after the edit is committed

Checks verify semantic correctness: a "remove legacy branch" task checks that the word `legacy` is no longer present and that `modern` and `strict` still are, rather than asserting exact line content.

### Outcome glyphs

| Glyph | Meaning |
|-------|---------|
| ✓ | Check passed |
| ≈ | Edit made, check failed |
| ∅ | No edit made |
| ✗ | Session did not complete |

### Running the suite

```sh
cargo run -p shunt-bench --bin capability -- \
  crates/shunt-bench/gemma4-8080.toml 1

# Run a specific task N times
cargo run -p shunt-bench --bin capability -- \
  crates/shunt-bench/gemma4-8080.toml 3 --task=move_error_handling
```

Model config:

```toml
[[model]]
name         = "gemma-4-12b-qat"
engine       = "llamacpp"
endpoint     = "http://127.0.0.1:8080"
model_id     = "gemma4-12b"
timeout_secs = 120
```

---

## Pure-bench suite

The capability suite runs through shunt's own harness (localization, tree-sitter ranking,
grammar-constrained decoding). Pure-bench strips all of that away: a thin, harness-free tool
loop (`list_files` / `read_file` / `search` / `edit_file` / `finish`) talks to the model's
OpenAI-compatible endpoint directly. It measures the model itself — raw tool-use and reasoning
— decoupled from anything shunt's runtime does to help it.

```sh
cargo run -p shunt-bench --bin pure -- crates/shunt-bench/pure-models.toml 1
cargo run -p shunt-bench --bin pure -- crates/shunt-bench/pure-models.toml 3 --task=oncall_schedule_puzzle
```

### Current status

39-task suite, 1 run each:

| Model | Size | Overall |
|-------|------|---------|
| Gemma4-12B (llama.cpp, Q4_K_M) | 12B | 35/39 (90%) |
| Qwen3.5-4B (ollama) | 4B | 27/39 (69%) |

Fail points — Qwen3.5-4B (12): incomplete multi-file edits (`cra_to_vite`, `extract_auth_service`, `thread_correlation_id`), turn-cap timeouts on multi-step tasks (`node_health_route_smoke`), partial fixes left in place (`implement_optimistic_lock`, `prototype_pollution_fix`, `toctou_refresh_token`, `event_listener_leak`, `discriminated_union_types`, `idempotency_key`), a hallucinated fix on already-safe code (`no_op_sql_injection`), and misdiagnosing a vague bug report (`vague_stale_test`).

Fail points — Gemma4-12B (4): one no-edit (`implement_retry_backoff`), one turn-cap timeout (`toctou_refresh_token`), one partial fix (`idempotency_key`), one hallucinated fix on already-correct pooled-connection code (`no_op_already_pooled_connections`).

The no-op tasks (`no_op_sql_injection`, `no_op_cursor_pagination`, `no_op_already_pooled_connections`, `red_herring_*`) specifically catch models editing code that's already correct because a bug report sounds plausible — both models fail at least one.

Reasoning tasks (novel logic uncontaminated by common OSS patterns), 3 runs each:

| Task | Qwen3.5-4B | Gemma4-12B |
|------|-----------|-----------|
| `tiered_refund_grace_exception` | 3/3 | 3/3 |
| `hotel_stay_length_bug` | 3/3 | 3/3 |
| `back_to_back_meeting_overlap` | 3/3 | 3/3 |
| `oncall_schedule_puzzle` | 2/3 | 1/3 |

`oncall_schedule_puzzle` (4 engineers × 7 days, 6 interacting constraints) is the only discriminative reasoning task so far, and the only task in the suite where Gemma4-12B scores below Qwen3.5-4B — both models struggle with multi-variable constraint tracking specifically, not single-branch conditional logic. The other three reasoning tasks are solved reliably by both models and need more difficulty to be useful discriminators.

---

## Terminal-Bench 2.1

Terminal-Bench 2.1 should be run through Harbor, the official harness, so results stay comparable with the leaderboard. The repo provides a Harbor custom-agent import path that installs and runs `shunt` inside each task container while Harbor keeps the official task setup and grading path.

Prerequisites:

```sh
uv tool install harbor
harbor --help
docker info
```

Smoke-test Harbor with the official oracle:

```sh
harbor run -d terminal-bench/terminal-bench-2-1 -a oracle -l 5
```

Run `shunt` through the official Terminal-Bench 2.1 dataset:

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SHUNT_ENDPOINT="http://host.docker.internal:8080"
export SHUNT_MODEL="qwen3.6-27b"

harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  --agent-import-path "benchmarks.terminal_bench.shunt_agent:ShuntAgent" \
  -m "local/qwen3.6-27b" \
  -k 5
```

Useful overrides:

- `SHUNT_ENDPOINT` sets the OpenAI-compatible endpoint visible from the task container.
- `SHUNT_MODEL` sets the model id written to `.shunt/config.toml`.
- `SHUNT_INSTALL_COMMAND` replaces the default release installer, useful for testing a branch or private build.
- `SHUNT_TIMEOUT_SECS` sets the per-LLM-call timeout in `.shunt/config.toml`.
- `SHUNT_RUN_TIMEOUT_SECS` sets Harbor's command timeout for one task attempt.

For one task:

```sh
harbor run \
  -d terminal-bench/terminal-bench-2-1 \
  --agent-import-path "benchmarks.terminal_bench.shunt_agent:ShuntAgent" \
  -m "local/qwen3.6-27b" \
  --include-task-name "<task-name>"
```
