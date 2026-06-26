# shunt-localize

Workspace search and file localization for the shunt agent.

This crate is the retrieval layer inside [shunt](https://github.com/osmanorhan/shunt). Given a task description, it finds which files in a workspace are relevant and returns them as a ranked `ContextPacket` that the agent can load into its context window.

It contains a hybrid lexical + semantic search engine built on Tantivy and tree-sitter, and layers on top of it:

- **intent detection** — classifies a request as code / config / docs to pick the right retrieval path
- **query planning** — extracts literals, symbol guesses, and repo-specific terms from natural-language requests
- **candidate ranking** — merges lexical and semantic hits, assigns roles (Implementation, Config, Test, Docs), and scores them
- **context packing** — assembles the final `ContextPacket` with primary candidates, supporting candidates, and inline snippets
- **workspace index management** — builds and incrementally refreshes the on-disk index at `.shunt/index/`

---

## How it fits into shunt

```
user request
     │
     ▼
SemanticLocalizer.localize(workspace, artifact)
     │
     ├─ ArtifactQueryPlanner  →  SearchQuery (intent + literals + symbols)
     ├─ SemanticRetriever     →  ranked file hits  (HybridEngine)
     ├─ LexicalRetriever      →  lexical fallback  (Tantivy)
     ├─ SemanticCandidateRanker → RankedCandidate list
     └─ DefaultContextPacker  →  ContextPacket
                                      │
                               agent context window
```

---

## Basic usage

```rust
use shunt_localize::{Localizer, SemanticLocalizer};

let localizer = SemanticLocalizer::default();

// Warm the index before timing (optional but recommended for benchmarks)
localizer.prewarm("/path/to/workspace")?;

// Localize against an understanding artifact
let packet = localizer.localize("/path/to/workspace", &artifact)?;

for candidate in &packet.primary_candidates {
    println!("{} ({:?})", candidate.file.path, candidate.role);
}
```

The index is built on first call and cached under `.shunt/index/`:
- `.shunt/index/manifest.json` — file inventory and schema version
- `.shunt/index/engine.snapshot.json` — serialized engine state

Subsequent calls reopen from snapshot in milliseconds.

---

## Core types

### `SemanticLocalizer`

The default end-to-end localizer. Wraps `SemanticRetriever` (HybridEngine) with a lexical fallback via `DefaultLocalizer`. Implements the `Localizer` trait.

```rust
pub trait Localizer {
    fn localize(&self, workspace: &str, artifact: &UnderstandingArtifact)
        -> LocalizeResult<ContextPacket>;
    fn prewarm(&self, workspace: &str) -> LocalizeResult<()>;
}
```

### `ContextPacket`

The output of a localization pass.

```rust
pub struct ContextPacket {
    pub primary_candidates:    Vec<RankedCandidate>,   // highest confidence files
    pub supporting_candidates: Vec<RankedCandidate>,   // lower confidence, still relevant
    pub retrieval_backend:     RetrievalBackend,       // Lexical or Semantic
    pub raw_hits:              Vec<SearchHit>,         // matched lines with context
    pub retrieved_files:       Vec<RetrievedFile>,     // full file reads (when loaded)
    pub structured_files:      Vec<StructuredFile>,    // tree-sitter parsed structure
}
```

### `RankedCandidate`

```rust
pub struct RankedCandidate {
    pub file:     CandidateFile,      // path + summary
    pub role:     CandidateRole,      // Implementation | Config | Test | Docs | Callsite
    pub score:    f32,
    pub snippets: Vec<CandidateSnippet>,
    pub evidence: Vec<EvidenceRef>,
    pub reasons:  Vec<String>,
}
```

### `RetrievalBackend`

```rust
pub enum RetrievalBackend {
    Lexical,   // Tantivy BM25 — precise, works without a warmed index
    Semantic,  // HybridEngine — lexical + tree-sitter structure + fusion ranking
}
```

---

## Lower-level search engine

The search engine is part of this crate (`src/search/`). You can use it directly for lower-level queries:

```rust
use shunt_localize::{WorkspaceConfig, WorkspaceIndex, SearchOptions, ContextTarget};

let opened = WorkspaceIndex::open(std::path::Path::new("."), WorkspaceConfig::default())?;
let mut workspace = opened.workspace;
let engine = workspace.engine();

// File-level search
let files = engine.search_files("rate limit middleware", 8);

// Chunk-level search
let hits = engine.search_code("rate limit middleware", SearchOptions::agent());

// Context assembly
let context = engine.get_context(
    &[ContextTarget::File {
        path: files[0].file_path.clone(),
        anchor_chunks: files[0].anchor_chunks.clone(),
    }],
    &Default::default(),
)?;

// Exact follow-ups
let defs = engine.find_symbol("RateLimiter", 5);
let callers = engine.find_callers("apply_limit", 5);

// Graph expansion from exact hits
let graph = engine.graph_from_hits(&hits[..3], Default::default());

// Incremental refresh after file edits
let _report = workspace.refresh()?;
workspace.save()?;
```

### Supported languages

| Language | Chunking | Symbols | Calls | Imports |
|---|---|---|---|---|
| Rust | tree-sitter-rust | ✓ | ✓ | ✓ |
| TypeScript / JavaScript | tree-sitter-typescript / tree-sitter-javascript | ✓ | ✓ | ✓ |
| Python | tree-sitter-python | ✓ | ✓ | ✓ |

All other file types are indexed as plain-text chunks (lexical search only).

### Index location

The engine stores its index under `.shunt/index/` in the workspace root:

- `.shunt/index/manifest.json` — file inventory, schema version, ingest stats
- `.shunt/index/engine.snapshot.json` — serialized chunk store and graph; avoids full reparse on reopen

If either file is missing or corrupt, the engine rebuilds from source automatically.

---

## Pipeline customisation

The localization pipeline is fully composable. Each stage is a trait:

| Stage | Trait | Default impl |
|---|---|---|
| Query planning | `QueryPlanner` | `ArtifactQueryPlanner` |
| Retrieval | `Retriever` | `SemanticRetriever` / `LexicalRetriever` |
| Structure extraction | `StructureExtractor` | `TreeSitterStructureExtractor` |
| Candidate ranking | `CandidateRanker` | `SemanticCandidateRanker` / `TfIdfRanker` |
| Context packing | `ContextPacker` | `DefaultContextPacker` |

To swap a stage, construct a `PipelineLocalizer` directly:

```rust
use shunt_localize::{PipelineLocalizer, ArtifactQueryPlanner, LexicalRetriever,
                     TreeSitterStructureExtractor, TfIdfRanker, DefaultContextPacker};

let localizer = PipelineLocalizer::new(
    ArtifactQueryPlanner,
    LexicalRetriever,
    TreeSitterStructureExtractor,
    TfIdfRanker,
    DefaultContextPacker,
);
```

---

## Performance characteristics

Benchmarked on a synthetic 100k-file corpus on a Linux development machine:

| Operation | Time |
|---|---|
| Fresh index build | ~12.5s |
| Snapshot save | ~530ms |
| Snapshot reopen | ~7s |
| Incremental refresh (1 file changed) | ~10.6s |
| `search_files` avg / p95 latency | 230ms / 580ms |
| `search_code` avg / p95 latency | 235ms / 605ms |

RSS after reopen is approximately 1.3GB for a 100k-file corpus. For typical project sizes (thousands of files), index build takes under a second and latency is in the single-digit millisecond range.

---

## License

MIT — part of the [shunt](https://github.com/osmanorhan/shunt) workspace.
