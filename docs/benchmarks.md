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

---

## Results — 2026-06-26

### Qwen3.6-27B Q4_K_S

| Task | Difficulty | Result | Avg time | Avg turns |
|------|-----------|--------|----------|-----------|
| change_constant | trivial | ✓ | 22 s | 7 |
| add_function | easy | ✓ | 30 s | 16 |
| fix_clamp | medium | ✓ | 49 s | 12 |
| rename_two_sites | hard | ✓ | 36 s | 12 |
| add_locked_branch | medium | ✓ | 39 s | 16 |
| remove_legacy_mode | medium | ✓ | 24 s | 12 |
| rename_export_import_call | hard | ✓ | 38 s | 12 |
| thread_config_field | hard | ✓ | 52 s | 13 |
| sync_pricing_test | hard | ✓ | 55 s | 20 |
| move_error_handling | hard | ✓ | 64 s | 13 |

**Overall: 10/10 (100%)**

---

### Gemma-4 12B QAT

| Task | Difficulty | Result | Avg time | Avg turns |
|------|-----------|--------|----------|-----------|
| change_constant | trivial | ✓ | 57 s | 8 |
| add_function | easy | ✓ | 13 s | 8 |
| fix_clamp | medium | ✓ | 72 s | 11 |
| rename_two_sites | hard | ✓ | 32 s | 12 |
| add_locked_branch | medium | ✓ | 24 s | 10 |
| remove_legacy_mode | medium | ✓ | 152 s | 11 |
| rename_export_import_call | hard | ✓ | 74 s | 11 |
| thread_config_field | hard | ✓ | 53 s | 12 |
| sync_pricing_test | hard | ✓ | 65 s | 12 |
| move_error_handling | hard | ✓ | 212 s | 13 |

**Overall: 10/10 (100%)**

---

### Summary

| Model | Size | Overall | Avg time (hard tasks) |
|-------|------|---------|-----------------------|
| Qwen3.6-27B Q4_K_S | 27B | 100% | ~51 s |
| Gemma-4 12B QAT | 12B | 100% | ~87 s |

Qwen3.6-27B is faster on hard tasks. Gemma-4 12B reaches the same success rate at a smaller model size.

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
