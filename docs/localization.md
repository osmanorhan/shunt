---
nav_exclude: true
---

# Localization

Localization is the code-context retrieval subsystem.

It is not optional glue around the agent.
It is the part that turns a vague user request into a small, ranked set of real code locations and snippets.

For small local models, this is one of the most important reliability levers.

## Purpose

Localization should answer:

- which files are likely relevant
- which snippets inside those files matter
- which symbol or block is the likely edit site
- what context should be shown to the model

The model should not search the whole repo itself.
The runtime should search first, then give the model a small, inspectable context packet.

## Design Goals

- local-first
- fast on medium and large repos
- ignore-aware
- parallel by default
- lexical retrieval first
- AST extraction only on top candidates
- tiny context packets
- deterministic and inspectable ranking

## Non-Goals

- no embeddings in v1
- no vector database in v1
- no full-repo AST parse before lexical narrowing
- no large persistent search index unless repeated runs prove it is needed

## Retrieval Stack

Recommended stack:

1. `ignore`
   - repo traversal
   - `.gitignore` and `.ignore` handling
   - file type filters
   - parallel walking

2. `aho-corasick`
   - fast multi-literal matching
   - cheap first-pass term scans

3. `grep` stack
   - `grep-regex`
   - `grep-searcher` or the `grep` facade
   - contextual line extraction
   - regex matches for identifiers and patterns

4. `tree-sitter`
   - parse only top K candidate files
   - extract enclosing function, impl, class, or block
   - improve snippet quality and ranking

## Pipeline

The pipeline should be:

1. Query planning
2. Parallel file walk
3. Lexical search
4. Candidate ranking
5. AST extraction on top K
6. Context packing

### 1. Query Planning

Input:

- user request
- interpreted goal
- current artifact

Output:

- literal terms
- regex terms
- symbol guesses
- intent

Intent examples:

- `code_fix`
- `config_fix`
- `test_fix`
- `docs`

Intent controls file filtering and ranking.

### 2. Parallel File Walk

Walk the repo with ignore rules respected.

Default behavior:

- skip hidden and ignored files
- skip binary files
- prefer code files
- include manifests only when intent suggests configuration or dependency work

### 3. Lexical Search

Use literals first, then regex when needed.

Search output should include:

- file path
- matched line
- line number
- local surrounding lines
- which term matched

### 4. Candidate Ranking

Rank files and snippets by explicit signals:

- path hits
- literal hit count
- co-occurrence of terms
- proximity of related hits
- extension/type priors
- symbol-like matches
- presence of likely networking, timeout, client, or error-handling constructs

This ranking must be inspectable in traces.

### 5. AST Extraction

Run only on top K files.

Extract:

- enclosing function or method
- enclosing impl or class
- nearby imports
- signature lines

This gives the model structurally meaningful snippets instead of raw file prefixes.

### 6. Context Packing

The model should receive:

- normalized task
- 3 to 5 candidate files max
- 1 to 3 snippets per file
- candidate score and reason
- enclosing symbol names if available

Do not send:

- whole files by default
- repo root summaries
- non-code files for code-fix tasks
- files with no direct lexical hit

## Data Model

The subsystem should produce:

- `SearchIntent`
- `SearchQuery`
- `SearchHit`
- `CandidateFile`
- `CandidateSnippet`
- `ContextPacket`

These should be runtime-visible and persistable when useful.

## Runtime Role

Localization belongs between `Understand` and `Agree`.

Flow:

1. `Clarify`
2. `Understand`
3. `Localize`
4. `Agree`
5. `Execute`

Execution should not begin unless localization has found concrete candidate files for code-edit tasks.

## Success Criteria

Localization is good enough for the first slice when:

- top 3 candidates usually contain the edit site
- snippets contain the actual editable block
- generated patches stop targeting nonexistent code
- traces clearly explain candidate ranking

## Implementation Order

1. `frame-localize` crate
2. query planner types
3. ignore-aware file walker
4. lexical search engine
5. ranking
6. context packet
7. tree-sitter extraction on top candidates
8. runtime integration