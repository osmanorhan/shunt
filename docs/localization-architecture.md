---
nav_exclude: true
---

# Localization Architecture

Localization is a subsystem, not a prompt trick.

Its job is to turn a user request into a small, ranked, inspectable set of code locations before any edit proposal is generated.

## Pipeline

The pipeline should stay stable across languages:

1. `QueryPlanner`
2. `Retriever`
3. `StructureExtractor`
4. `CandidateRanker`
5. `ContextPacker`

`Localizer` is the composed interface the runtime calls.

## Roles

### `QueryPlanner`

Input:

- original request
- interpreted goal
- current artifact

Output:

- search intent
- literal terms
- regex terms
- symbol guesses

This stage should stay language-agnostic.
It should prefer authoritative user input over model-elaborated draft text.
In practice that means:

- use `original_request` first
- use concrete artifact scope hints when they are path-like
- do not let speculative interpreted text dominate retrieval

### `Retriever`

Input:

- workspace root
- planned query

Output:

- retrieved files
- lexical hits
- path hits

This stage is local-first, ignore-aware, and parallel.
It should gather lexical evidence without trying to guess symbols or language semantics.

### `StructureExtractor`

Input:

- query
- retrieved files

Output:

- focused snippets
- enclosing symbols when available
- extra structural hints

This is where per-language logic belongs.
The current fallback can be generic hit-centered snippets.
The interface stays stable when the implementation moves to `tree-sitter`.

### `CandidateRanker`

Input:

- query
- structured files

Output:

- ranked candidates with explicit reasons

Ranking should prefer edit-site likelihood, not raw token overlap alone.
The first general ranking baseline should be corpus-driven:

- query-term coverage
- lexical hit density
- path-term matches
- rarity weighting such as TF-IDF style scoring

Avoid repo-specific bonuses such as hardcoded file names or guessed symbol classes.

### `ContextPacker`

Input:

- query
- ranked candidates

Output:

- small context packet for the model

This stage controls context size and keeps traces inspectable.

## Language Strategy

The universal layer is:

- query planning
- lexical retrieval
- ranking contract
- context packing

The language-specific layer is:

- symbol extraction
- definition vs call-site classification
- enclosing scope extraction

That means the system generalizes by swapping or extending `StructureExtractor`, not by branching the whole runtime.

## Runtime Contract

Runtime flow:

1. `Clarify`
2. `Understand`
3. `Localize`
4. `Agree`
5. `Execute`

Execution should consume `ContextPacket`, not search the repo again ad hoc.

## Current State

Current `frame-localize` now has the pipeline interfaces and a lean default implementation:

- `ArtifactQueryPlanner`
- `LexicalRetriever`
- `SnippetStructureExtractor`
- `TfIdfRanker`
- `DefaultContextPacker`

This is still lexical-first, but it now avoids hardcoded stop-word tables, guessed symbols, and filename bonuses.
The architecture is ready for better structure extraction without another runtime rewrite.

## Next Steps

1. Add `tree-sitter`-based `StructureExtractor`
2. Distinguish likely definition files from caller/config/test files
3. Add candidate-role signals into ranking
4. Feed packed snippets, not broad file prefixes, into change generation