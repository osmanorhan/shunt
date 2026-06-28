# shunt

A coding agent built for local language models. shunt is designed around the limits of
local models (9B–14B, reasoning models) instead of assuming cloud-model behaviour: it
indexes the workspace and feeds relevant snippets, ranks edit targets with lexical +
tree-sitter structure, and uses grammar-constrained decoding so tool calls stay valid.

The runtime is a **pure state machine with effects pushed to the edges**. This shape is the
backbone of the codebase and the principles below exist to keep it that way.

## Commands

```sh
cargo build --release -p shunt-cli            # build the `shunt` binary
cargo test  --workspace --all-features        # tests
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all                               # format
./scripts/release-check.sh                    # full gate: fmt + clippy + test + doctest + release build
```

The capability benchmark needs a live model (`SHUNT_LLM` set to an OpenAI-compatible endpoint):

```sh
cargo run -p shunt-bench --bin capability -- ...   # see docs/benchmarks.md
```

Live integration tests are gated behind `SHUNT_RELEASE_LIVE=1` + `SHUNT_LLM` and are
`#[ignore]`d otherwise.

## Architecture

Tasks move through a state machine: `Submit → Observe → ProposeChange → ProposalReady →
[approval policy] → CommitChange → Completed`. `ProposeChange` plans and assembles a diff;
`CommitChange` applies it. `TaskMachine::transition` is **pure** — it takes a
`Command`/`MachineEvent` and returns `Vec<Effect>`. The `EffectRunner` performs IO and feeds
resulting events back. See `docs/architecture.md` for the full flow.

### Crate layering (dependencies point downward)

| Crate | Responsibility | May import |
|-------|----------------|-----------|
| `shunt-core` | **Domain** — `TaskState`, `Command`, `MachineEvent`, `Effect`, `Notification`, `AutonomyPolicy`. No IO. | serde, time only |
| `shunt-runtime` | `TaskMachine::transition` (pure), `spawn_session` actor, `EffectRunner`, `driver` client | core, infer, knowledge, localize, store |
| `shunt-infer` | Inference providers, `AgentSession` edit loop, `ModelRegistry` | core, edit, localize |
| `shunt-localize` | Workspace search: hybrid lexical (tantivy) + semantic index | core |
| `shunt-knowledge` | Dependency / version evidence | core, localize |
| `shunt-store` | SQLite persistence — tasks, artifacts, ledger | core |
| `shunt-edit` | Position-addressed editing: `ReplaceLines`, append, delete | (leaf) |
| `shunt-cli` | `shunt` binary: ratatui TUI + `agent --once` headless | core, infer, runtime, store |
| `shunt-bench` | Capability benchmark: fixtures, checks, harness | core, edit, infer, localize, runtime |

**`shunt-core` is the pure domain layer.** It depends only on `serde` and `time`. Importing
`tokio`, `tantivy`, `sqlx`/SQLite, `reqwest`, or any IO/framework crate into `shunt-core` is a
design failure — push that behaviour into a crate above it behind an `Effect` or interface.

---

# Engineering principles

These are binding. When a change conflicts with one, **stop and raise it** — don't quietly
trade it away. When a tradeoff is genuinely required, make it consciously and record why.

- **Design before code.** Validate the technical design is correct before writing it. Solve the
  problem behind the feature, not just the feature. Chop every problem into pieces each easy to
  understand and solve on their own.

- **Write as little code as possible.** Prefer fewer lines over large types/functions. Code is
  written to be changed and deleted — optimise for deletability. Every abstraction must earn its
  place; cut what doesn't.

- **Open for extension, closed for modification.** New behaviour arrives as a new
  implementation behind an existing seam (a new `Effect` variant + handler, a new provider in
  `ModelRegistry`, a new `AutonomyPolicy`, a new tool in the agent loop), not an edit to working
  code.

- **Patterns/structures over conditionals.** Replace `if`/`match`-on-type branching with
  polymorphism, enums dispatched through one site, strategy objects, lookups. Accumulating
  branching logic is a design smell to resolve, not grow. (The state machine already models flow
  as states + transitions — keep new flow there, not in scattered conditionals.)

- **Fail fast; never hide errors.** No workarounds, no silent fallbacks unless explicitly
  requested. If something is broken, break loudly — surface errors, never swallow them. Avoid
  `unwrap`/`expect` on fallible runtime paths; return `Result` and let it propagate to a seam
  that decides.

- **No monkeypatching, no ad-hoc string sniffing.** Don't detect structure by peeking at
  characters — checking for a leading `[` to guess an array, scanning for `{` to guess JSON,
  regex-matching a tool name out of raw text. Parse into a real type and dispatch on that type
  through one handler. Model output is structured at the source: rely on grammar-constrained
  decoding and JSON-schema tool calls so the model emits valid structure, then `serde`-deserialize
  it — never reconstruct meaning from string fragments downstream. If you find yourself
  string-matching to recover shape the model should have given you, fix it at the schema/grammar
  seam, not with a parser-by-hand.

- **Lean on model/LLM features over hand-rolled logic.** Prefer the platform's structured
  capabilities — grammar/JSON-schema constraints, tool/function calling, the chat template's own
  flags — to heavy `if`/`else` ladders and `match` cascades that re-derive what the model can be
  asked to produce directly. Branching that exists only to clean up or guess at unstructured model
  output is a smell; remove its cause.

- **Handle side effects deliberately.** Keep the core pure: `shunt-core` and
  `TaskMachine::transition` import no framework and do no IO. Push I/O, randomness, and
  persistence to the edges behind `Effect`/`EffectRunner` and trait interfaces. A side effect
  must be visible in the type/seam, never a surprise.

- **Simplicity over easiness.** The simple solution beats the convenient one. "It works" is not
  the bar — correct, clear, and changeable is the bar.

- **Code for others** (including future-you with a bad memory). Artifacts must be
  self-explanatory, debuggable, and deletable. Readable, well-shaped code is a deliverable, not a
  luxury. Leave the codebase healthier than you found it.

## House style

- **No comments explaining _what_** — structure and names carry it. Comments only for a
  non-obvious _why_ the code cannot express, and even then sparingly. (`//!` module docs stating
  intent, as in `shunt-core/src/lib.rs`, are welcome.)
- **Small, single-responsibility units.** If a function needs a comment to be readable, split it.
- **Names state intent.** Newtypes and value objects over primitives (`TaskId`, `ArtifactId`,
  `AmbiguityId` — not bare `String`).
- **The domain layer stays pure.** Importing IO/framework crates into `shunt-core` is forbidden.
  Treat it as a build failure even though it isn't yet enforced by an architecture test.
- **Tests assert invariants and exact sequences, never luck.** Stochastic / model-driven code is
  tested with seeded RNG, deterministic fixtures, and distribution tolerances. Benchmark checks
  are deterministic checks on resulting files.

## Project gotchas

These are hard-won lessons specific to driving small local models — violating them causes silent
stalls or empty output:

- **Never add `maxLength`/`maxItems` to agent JSON schemas.** llama.cpp's grammar FSM is O(n) and
  stalls for minutes. Express caps in prose, not the schema.
- **Suppress thinking for content/edit calls.** Use `chat_template_kwargs{enable_thinking:false}`
  (not `/no_think`) on `call_text`; for Qwen3 use `suppress_content_thinking:true`. Streaming on
  thinking models contaminates the KV cache → empty output; use non-streaming `call_text`.
- **Tool calls are non-streaming with `cache_prompt:false`** — streaming SSE hangs on real repos.
- Cap Gemma reasoning-model `max_tokens` (~8192); large budgets get exhausted before content
  appears.

