---
title: Architecture
layout: default
nav_order: 4
---

# Architecture
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Session flow

Tasks move through a state machine from submission to completion:

```
Submit → Observe → ProposeChange → ProposalReady
       → [approval policy]
              ├ pause: WaitingForUser::Approval → (Approve) → CommitChange
              └ auto:  CommitChange
       → CommitChange → Completed

Agent ask_user → AgentAsked → WaitingForUser::Clarification → (Answer) → ProposeChange
```

`ProposeChange` is the planning phase — the agent reads files, searches the workspace, and assembles a diff. `CommitChange` applies that diff to disk. The two phases are distinct states with distinct effect types.

---

## Crate layout

| Crate | Responsibility |
|-------|---------------|
| `shunt-core` | Domain types: `TaskState`, `Command`, `MachineEvent`, `Effect`, `Notification`, `AutonomyPolicy`. No IO. |
| `shunt-runtime` | `TaskMachine::transition` (pure), `spawn_session` actor, `EffectRunner`, programmatic client (`driver`). |
| `shunt-infer` | Inference providers, `AgentSession` edit loop, `ModelRegistry`. |
| `shunt-localize` | Workspace search: hybrid lexical (tantivy) + semantic index. |
| `shunt-knowledge` | Dependency and version evidence. |
| `shunt-store` | SQLite persistence — tasks, artifacts, ledger. |
| `shunt-cli` | `shunt` binary: ratatui TUI and `agent --once` headless mode. |
| `shunt-bench` | Capability benchmark: task fixtures, checks, run harness. |
| `shunt-edit` | Position-addressed file editing: `ReplaceLines`, append, delete. |

---

## Session actor

`spawn_session` owns the `TaskMachine` state in a Tokio actor. Clients interact through `SessionHandle`:

```
SessionHandle
  ├── send(Command)                    → drive transitions
  ├── subscribe() → Stream<Notification>
  └── watch()    → watch::Receiver<TaskState>
```

The actor passes `Command`s and `MachineEvent`s through `TaskMachine::transition`, which returns `Vec<Effect>`. The `EffectRunner` executes each effect and emits the resulting `MachineEvent`s back to the actor.

---

## Agent tool loop

During `ProposeChange`, `AgentSession` runs a tool-use loop:

1. Model receives workspace context — search results, file excerpts, task description
2. Model calls a tool: `read_file`, `search_files`, `edit`, `command`, `knowledge` (when a backend is wired), `ask_user`, or `done`
3. Tool result is returned to the model
4. Loop continues until `done` is called or the turn budget is reached

The tool set is defined once in a registry (`TOOLS` in `agent.rs`) that drives both the action schema and the system-prompt tool reference, so they cannot drift. `edit` creates a new file (no `start_line`) or modifies an existing one by line range (`start_line`/`end_line`); file deletion is done via `command` (`rm`). On OpenAI-compatible servers, `shunt-infer` sends strict native function tools and serializes prior turns as `assistant.tool_calls` plus `tool` results so the wire format stays aligned with the OpenAI tool-calling contract.

---

## Autonomy policy

Two built-in policies control approval behaviour:

| Policy | On proposal ready | Used by |
|--------|------------------|---------|
| `headless()` | Auto-approve | `agent --once`, benchmark |
| `agentic()` | Pause for human confirmation | TUI |

The TUI presents the proposed diff and waits for the user to approve or reject. `agent --once` applies changes automatically.

---

## Workspace search

The workspace index lives at `.shunt/index/` and combines two signals:

- **Lexical** (tantivy BM25): exact-match and token overlap
- **Semantic**: embedding-based similarity

`search_files` queries both and returns ranked file excerpts. The index is built on first run and updated incrementally.

---

## Benchmark

The capability suite in `shunt-bench` runs tasks end-to-end against a live model:

1. A task fixture writes source files to a temp workspace and submits a request
2. The full session stack processes the task
3. A deterministic check runs on the resulting file

Every run writes `.shunt/debug.log` — prompts, responses, tool calls, and state transitions — for inspection. See the [Benchmarks](benchmarks) page for current results.
