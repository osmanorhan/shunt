use super::chunk::{ChunkId, CodeChunk};
use super::tokenize::{split_identifier, unique_tokens};
use super::types::{HitSource, SearchHit};
use std::collections::HashMap;

/// Exact and expanded indexes for definitions, references, calls, and imports.
#[derive(Default)]
pub struct SymbolIndex {
    definitions: HashMap<String, Vec<ChunkId>>,
    definitions_exact: HashMap<String, Vec<ChunkId>>,
    references: HashMap<String, Vec<ChunkId>>,
    references_exact: HashMap<String, Vec<ChunkId>>,
    calls: HashMap<String, Vec<ChunkId>>,
    calls_exact: HashMap<String, Vec<ChunkId>>,
    imports: HashMap<String, Vec<ChunkId>>,
    imports_exact: HashMap<String, Vec<ChunkId>>,
}

impl SymbolIndex {
    /// Search symbol definitions first, then references, preserving exact index order.
    pub fn search(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        let mut hits = self.definitions_of(query, limit);
        if hits.len() < limit {
            hits.extend(self.references_to(query, limit - hits.len()));
        }
        dedupe_hits(hits, limit)
    }

    /// Build symbol indexes from active chunk metadata.
    pub fn build(chunks: &[CodeChunk]) -> Self {
        let mut definitions: HashMap<String, Vec<ChunkId>> = HashMap::new();
        let mut definitions_exact: HashMap<String, Vec<ChunkId>> = HashMap::new();
        let mut references: HashMap<String, Vec<ChunkId>> = HashMap::new();
        let mut references_exact: HashMap<String, Vec<ChunkId>> = HashMap::new();
        let mut calls: HashMap<String, Vec<ChunkId>> = HashMap::new();
        let mut calls_exact: HashMap<String, Vec<ChunkId>> = HashMap::new();
        let mut imports: HashMap<String, Vec<ChunkId>> = HashMap::new();
        let mut imports_exact: HashMap<String, Vec<ChunkId>> = HashMap::new();

        for chunk in chunks {
            if !chunk.active {
                continue;
            }
            for symbol in chunk_definitions(chunk) {
                definitions.entry(symbol).or_default().push(chunk.id);
            }
            for symbol in exact_chunk_definitions(chunk) {
                definitions_exact.entry(symbol).or_default().push(chunk.id);
            }
            for token in chunk_references(chunk) {
                references.entry(token).or_default().push(chunk.id);
            }
            for token in exact_chunk_references(chunk) {
                references_exact.entry(token).or_default().push(chunk.id);
            }
            for call in chunk_calls(chunk) {
                calls.entry(call).or_default().push(chunk.id);
            }
            for call in exact_symbols(&chunk.calls) {
                calls_exact.entry(call).or_default().push(chunk.id);
            }
            for import in chunk_imports(chunk) {
                imports.entry(import).or_default().push(chunk.id);
            }
            for import in exact_symbols(&chunk.imports) {
                imports_exact.entry(import).or_default().push(chunk.id);
            }
        }

        Self {
            definitions,
            definitions_exact,
            references,
            references_exact,
            calls,
            calls_exact,
            imports,
            imports_exact,
        }
    }

    pub fn definitions_of(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        lookup_expanded(
            &self.definitions,
            symbol,
            limit,
            HitSource::SymbolDefinition,
        )
    }

    pub fn definitions_of_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        lookup_exact(
            &self.definitions_exact,
            symbol,
            limit,
            HitSource::SymbolDefinition,
        )
    }

    pub fn references_to(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        lookup_expanded(&self.references, symbol, limit, HitSource::SymbolReference)
    }

    pub fn references_to_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        lookup_exact(
            &self.references_exact,
            symbol,
            limit,
            HitSource::SymbolReference,
        )
    }

    pub fn calls_to(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        lookup_expanded(&self.calls, symbol, limit, HitSource::Call)
    }

    pub fn calls_to_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        lookup_exact(&self.calls_exact, symbol, limit, HitSource::Call)
    }

    pub fn imports_of(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        lookup_expanded(&self.imports, symbol, limit, HitSource::Import)
    }

    pub fn imports_of_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        lookup_exact(&self.imports_exact, symbol, limit, HitSource::Import)
    }

    pub(crate) fn definition_chunk_ids_for_exact_symbol(&self, symbol: &str) -> Option<&[ChunkId]> {
        self.definitions_exact.get(symbol).map(Vec::as_slice)
    }

    pub(crate) fn reference_chunk_ids_for_exact_symbol(&self, symbol: &str) -> Option<&[ChunkId]> {
        self.references_exact.get(symbol).map(Vec::as_slice)
    }

    pub(crate) fn call_chunk_ids_for_exact_symbol(&self, symbol: &str) -> Option<&[ChunkId]> {
        self.calls_exact.get(symbol).map(Vec::as_slice)
    }

    pub(crate) fn import_chunk_ids_for_exact_symbol(&self, symbol: &str) -> Option<&[ChunkId]> {
        self.imports_exact.get(symbol).map(Vec::as_slice)
    }
}

fn lookup_exact(
    index: &HashMap<String, Vec<ChunkId>>,
    symbol: &str,
    limit: usize,
    source: HitSource,
) -> Vec<SearchHit> {
    let normalized = normalize_query_symbol(symbol);
    let Some(chunk_ids) = index.get(&normalized) else {
        return Vec::new();
    };

    let hits = chunk_ids
        .iter()
        .enumerate()
        .map(|(rank, chunk_id)| {
            SearchHit::new(
                *chunk_id,
                1.0 / (rank + 1) as f32,
                source,
                vec![normalized.clone()],
            )
        })
        .collect();

    dedupe_hits(hits, limit)
}

fn lookup_expanded(
    index: &HashMap<String, Vec<ChunkId>>,
    symbol: &str,
    limit: usize,
    source: HitSource,
) -> Vec<SearchHit> {
    let mut scores: HashMap<ChunkId, f32> = HashMap::new();
    let mut matches: HashMap<ChunkId, Vec<String>> = HashMap::new();

    for query_symbol in expand_symbol_query(symbol) {
        if let Some(chunk_ids) = index.get(&query_symbol) {
            for chunk_id in chunk_ids {
                *scores.entry(*chunk_id).or_insert(0.0) += 1.0;
                matches
                    .entry(*chunk_id)
                    .or_default()
                    .push(query_symbol.clone());
            }
        }
    }

    let hits = scores
        .into_iter()
        .map(|(chunk_id, score)| {
            SearchHit::new(
                chunk_id,
                score,
                source,
                matches.remove(&chunk_id).unwrap_or_default(),
            )
        })
        .collect();

    dedupe_hits(hits, limit)
}

fn chunk_definitions(chunk: &CodeChunk) -> Vec<String> {
    if chunk.definitions.is_empty() {
        return fallback_definitions(chunk);
    }

    expand_symbols(&chunk.definitions)
}

fn exact_chunk_definitions(chunk: &CodeChunk) -> Vec<String> {
    if chunk.definitions.is_empty() {
        return fallback_exact_definitions(chunk);
    }

    exact_symbols(&chunk.definitions)
}

fn chunk_references(chunk: &CodeChunk) -> Vec<String> {
    if chunk.references.is_empty() {
        return unique_tokens(&chunk.content);
    }

    expand_symbols(&chunk.references)
}

fn exact_chunk_references(chunk: &CodeChunk) -> Vec<String> {
    if chunk.references.is_empty() {
        return unique_tokens(&chunk.content);
    }

    exact_symbols(&chunk.references)
}

fn chunk_calls(chunk: &CodeChunk) -> Vec<String> {
    expand_symbols(&chunk.calls)
}

fn chunk_imports(chunk: &CodeChunk) -> Vec<String> {
    let mut imports = Vec::new();
    for import in &chunk.imports {
        let normalized = normalize_query_symbol(import);
        if !normalized.is_empty() {
            imports.push(normalized);
        }
        imports.extend(expand_symbol_query(import));
    }
    imports.sort();
    imports.dedup();
    imports
}

fn fallback_definitions(chunk: &CodeChunk) -> Vec<String> {
    let mut symbols = Vec::new();

    for line in chunk.content.lines() {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            continue;
        }

        let symbol = match tokens.as_slice() {
            ["pub", "fn", name, ..] | ["fn", name, ..] => clean_symbol(name),
            ["pub", "struct", name, ..] | ["struct", name, ..] => clean_symbol(name),
            ["pub", "enum", name, ..] | ["enum", name, ..] => clean_symbol(name),
            ["pub", "trait", name, ..] | ["trait", name, ..] => clean_symbol(name),
            ["impl", name, ..] => clean_symbol(name),
            _ => None,
        };

        if let Some(symbol) = symbol {
            symbols.extend(expand_symbol_query(&symbol));
        }
    }

    symbols
}

fn fallback_exact_definitions(chunk: &CodeChunk) -> Vec<String> {
    let mut symbols = Vec::new();

    for line in chunk.content.lines() {
        let tokens = line.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            continue;
        }

        let symbol = match tokens.as_slice() {
            ["pub", "fn", name, ..] | ["fn", name, ..] => clean_symbol(name),
            ["pub", "struct", name, ..] | ["struct", name, ..] => clean_symbol(name),
            ["pub", "enum", name, ..] | ["enum", name, ..] => clean_symbol(name),
            ["pub", "trait", name, ..] | ["trait", name, ..] => clean_symbol(name),
            ["impl", name, ..] => clean_symbol(name),
            _ => None,
        };

        if let Some(symbol) = symbol {
            symbols.push(symbol);
        }
    }

    symbols.sort();
    symbols.dedup();
    symbols
}

fn exact_symbols<'a>(symbols: impl IntoIterator<Item = &'a String>) -> Vec<String> {
    let mut exact = symbols
        .into_iter()
        .map(|symbol| normalize_query_symbol(symbol))
        .filter(|symbol| !symbol.is_empty())
        .collect::<Vec<_>>();
    exact.sort();
    exact.dedup();
    exact
}

fn expand_symbols<'a>(symbols: impl IntoIterator<Item = &'a String>) -> Vec<String> {
    let mut expanded = Vec::new();
    for symbol in symbols {
        expanded.extend(expand_symbol_query(symbol));
    }
    expanded.sort();
    expanded.dedup();
    expanded
}

fn clean_symbol(raw: &str) -> Option<String> {
    let cleaned = raw
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .trim_end_matches('{')
        .trim_end_matches('(')
        .to_ascii_lowercase();
    (!cleaned.is_empty()).then_some(cleaned)
}

fn expand_symbol_query(symbol: &str) -> Vec<String> {
    let lowered = normalize_query_symbol(symbol);
    let mut expanded = vec![lowered.clone()];
    for piece in split_identifier(&lowered) {
        if piece != lowered {
            expanded.push(piece);
        }
    }
    expanded.sort();
    expanded.dedup();
    expanded
}

pub(crate) fn normalize_exact_symbol(symbol: &str) -> String {
    symbol
        .trim()
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != ':')
        .to_ascii_lowercase()
}

fn normalize_query_symbol(symbol: &str) -> String {
    normalize_exact_symbol(symbol)
}

fn dedupe_hits(mut hits: Vec<SearchHit>, limit: usize) -> Vec<SearchHit> {
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.chunk_id.cmp(&right.chunk_id))
    });

    let mut deduped = Vec::new();
    let mut seen = HashMap::new();

    for hit in hits {
        if seen.insert(hit.chunk_id, ()).is_none() {
            deduped.push(hit);
        }
        if deduped.len() == limit {
            break;
        }
    }

    deduped
}

#[cfg(test)]
mod tests {
    use super::super::{ChunkKind, CodeChunk};
    use super::SymbolIndex;

    #[test]
    fn finds_definition_hits() {
        let chunks = vec![
            CodeChunk::new(
                1,
                "src/auth.rs",
                "rust",
                ChunkKind::Function,
                "pub fn refresh_access_token(user_id: UserId) {}",
            ),
            CodeChunk::new(
                2,
                "src/api.rs",
                "rust",
                ChunkKind::Function,
                "fn call_auth() { refresh_access_token(user_id); }",
            ),
        ];

        let index = SymbolIndex::build(&chunks);
        let hits = index.definitions_of("refresh_access_token", 5);
        assert_eq!(hits.first().map(|hit| hit.chunk_id), Some(1));
    }

    #[test]
    fn exact_lookup_does_not_match_split_tokens() {
        let mut chunk = CodeChunk::new(
            1,
            "src/engine.rs",
            "rust",
            ChunkKind::Struct,
            "pub struct HybridEngine;",
        );
        chunk.definitions = vec!["hybridengine".to_string()];

        let index = SymbolIndex::build(&[chunk]);
        assert_eq!(index.definitions_of_exact("HybridEngine", 5).len(), 1);
        assert_eq!(index.definitions_of_exact("engine", 5).len(), 0);
        assert_eq!(index.definitions_of("HybridEngine", 5).len(), 1);
    }

    #[test]
    fn finds_call_hits_from_structured_metadata() {
        let mut chunk = CodeChunk::new(
            1,
            "src/db.rs",
            "rust",
            ChunkKind::Function,
            "fn persist() { save_order(order); }",
        );
        chunk.calls = vec!["save_order".to_string()];

        let index = SymbolIndex::build(&[chunk]);
        let hits = index.calls_to("save_order", 5);
        assert_eq!(hits.first().map(|hit| hit.chunk_id), Some(1));
        assert_eq!(index.calls_to_exact("save_order", 5).len(), 1);
    }

    #[test]
    fn indexes_imports_separately() {
        let mut chunk = CodeChunk::new(
            1,
            "src/db.rs",
            "rust",
            ChunkKind::Module,
            "use super::db::Pool;",
        );
        chunk.imports = vec!["crate::db::pool".to_string()];

        let index = SymbolIndex::build(&[chunk]);
        assert_eq!(index.imports_of_exact("crate::db::pool", 5).len(), 1);
        assert_eq!(index.references_to_exact("crate::db::pool", 5).len(), 0);
    }
}
