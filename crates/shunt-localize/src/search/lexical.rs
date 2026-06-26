use super::chunk::{ChunkId, CodeChunk};
use super::tokenize::{split_identifier, tokenize, unique_tokens};
use super::types::{HitSource, SearchHit};
use std::collections::HashSet;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, STORED, STRING, Schema, TEXT, TantivyDocument, Value};
use tantivy::{Index, IndexReader, doc};

/// In-memory Tantivy index over code-aware chunk fields.
pub struct LexicalIndex {
    index: Index,
    reader: IndexReader,
    chunk_id: Field,
    path: Field,
    kind: Field,
    content: Field,
    comments: Field,
    identifier_parts: Field,
    exact_symbols: Field,
}

impl LexicalIndex {
    /// Build an index from active chunks.
    pub fn build(chunks: &[CodeChunk]) -> Self {
        let mut schema_builder = Schema::builder();
        let chunk_id = schema_builder.add_text_field("chunk_id", STORED);
        let path = schema_builder.add_text_field("path", TEXT | STORED);
        let kind = schema_builder.add_text_field("kind", TEXT | STORED);
        let content = schema_builder.add_text_field("content", TEXT | STORED);
        let comments = schema_builder.add_text_field("comments", TEXT | STORED);
        let identifier_parts = schema_builder.add_text_field("identifier_parts", TEXT | STORED);
        let exact_symbols = schema_builder.add_text_field("exact_symbols", STRING | STORED);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer = index
            .writer(50_000_000)
            .expect("in-memory tantivy writer can be created");

        for chunk in chunks {
            if !chunk.active {
                continue;
            }

            writer
                .add_document(doc!(
                    chunk_id => chunk.id.to_string(),
                    path => chunk.file_path.as_str(),
                    kind => chunk.chunk_type.as_tag(),
                    content => chunk.content.as_str(),
                    comments => comment_terms(chunk).join(" "),
                    identifier_parts => identifier_terms(chunk).join(" "),
                    exact_symbols => exact_symbol_terms(chunk).join(" "),
                ))
                .expect("tantivy document can be added");
        }

        writer.commit().expect("tantivy index can be committed");
        let reader = index.reader().expect("tantivy reader can be created");

        Self {
            index,
            reader,
            chunk_id,
            path,
            kind,
            content,
            comments,
            identifier_parts,
            exact_symbols,
        }
    }

    /// Search content, symbols, comments, paths, and chunk kinds.
    pub fn search(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        self.search_fields(query, limit, HitSource::Lexical, SearchMode::Broad)
    }

    /// Search path-oriented fields without content-heavy ranking.
    pub fn search_path(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        self.search_fields(query, limit, HitSource::Path, SearchMode::PathFocused)
    }

    fn search_fields(
        &self,
        query: &str,
        limit: usize,
        source: HitSource,
        mode: SearchMode,
    ) -> Vec<SearchHit> {
        if query.trim().is_empty() || limit == 0 {
            return Vec::new();
        }

        let query_terms = unique_tokens(query);
        let normalized_query = if query_terms.is_empty() {
            query.to_string()
        } else {
            query_terms.join(" ")
        };

        let mut query_parser = QueryParser::for_index(&self.index, mode.fields(self));
        for (field, boost) in mode.boosts(self) {
            query_parser.set_field_boost(field, boost);
        }

        let parsed_query = match query_parser.parse_query(&normalized_query) {
            Ok(query) => query,
            Err(_) => return Vec::new(),
        };

        let searcher = self.reader.searcher();
        let top_docs = searcher
            .search(&parsed_query, &TopDocs::with_limit(limit))
            .unwrap_or_default();
        let mut hits = Vec::with_capacity(top_docs.len());

        for (score, address) in top_docs {
            let Ok(document) = searcher.doc::<TantivyDocument>(address) else {
                continue;
            };
            let Some(chunk_id) = document
                .get_first(self.chunk_id)
                .and_then(|value| value.as_str())
                .and_then(|value| value.parse::<ChunkId>().ok())
            else {
                continue;
            };

            hits.push(SearchHit::new(
                chunk_id,
                score,
                source,
                matched_terms(
                    &document,
                    &[
                        self.path,
                        self.content,
                        self.comments,
                        self.identifier_parts,
                    ],
                    &query_terms,
                ),
            ));
        }

        hits
    }
}

#[derive(Clone, Copy)]
enum SearchMode {
    Broad,
    PathFocused,
}

impl SearchMode {
    fn fields(self, index: &LexicalIndex) -> Vec<Field> {
        match self {
            Self::Broad => vec![
                index.exact_symbols,
                index.identifier_parts,
                index.path,
                index.comments,
                index.content,
                index.kind,
            ],
            Self::PathFocused => vec![index.path, index.identifier_parts, index.kind],
        }
    }

    fn boosts(self, index: &LexicalIndex) -> Vec<(Field, f32)> {
        match self {
            Self::Broad => vec![
                (index.exact_symbols, 7.0),
                (index.identifier_parts, 4.0),
                (index.path, 3.0),
                (index.comments, 2.0),
                (index.kind, 1.5),
                (index.content, 1.0),
            ],
            Self::PathFocused => vec![
                (index.path, 5.0),
                (index.identifier_parts, 2.5),
                (index.kind, 1.0),
            ],
        }
    }
}

fn identifier_terms(chunk: &CodeChunk) -> Vec<String> {
    let mut terms = Vec::new();
    terms.extend(tokenize(&chunk.file_path));

    for symbol in chunk
        .definitions
        .iter()
        .chain(chunk.references.iter())
        .chain(chunk.calls.iter())
        .chain(chunk.imports.iter())
    {
        terms.extend(tokenize(symbol));
        terms.extend(split_identifier(symbol));
    }

    terms.sort();
    terms.dedup();
    terms
}

fn exact_symbol_terms(chunk: &CodeChunk) -> Vec<String> {
    let mut terms = Vec::new();

    for symbol in chunk
        .definitions
        .iter()
        .chain(chunk.references.iter())
        .chain(chunk.calls.iter())
        .chain(chunk.imports.iter())
    {
        let normalized = symbol
            .trim()
            .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != ':')
            .to_ascii_lowercase();
        if !normalized.is_empty() {
            terms.push(normalized);
        }
    }

    terms.sort();
    terms.dedup();
    terms
}

fn comment_terms(chunk: &CodeChunk) -> Vec<String> {
    let mut terms = Vec::new();

    for line in chunk.content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            terms.extend(tokenize(trimmed.trim_start_matches('/')));
        }
    }

    terms.sort();
    terms.dedup();
    terms
}

fn matched_terms(
    document: &TantivyDocument,
    fields: &[Field],
    query_terms: &[String],
) -> Vec<String> {
    let mut indexed_terms = HashSet::new();
    for field in fields {
        for value in document.get_all(*field) {
            if let Some(text) = value.as_str() {
                indexed_terms.extend(tokenize(text));
            }
        }
    }

    query_terms
        .iter()
        .filter(|term| indexed_terms.contains(term.as_str()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::{ChunkKind, CodeChunk};
    use super::LexicalIndex;

    #[test]
    fn ranks_exact_symbol_hits_first() {
        let mut target = CodeChunk::new(
            1,
            "src/auth.rs",
            "rust",
            ChunkKind::Function,
            "fn refresh_access_token(user_id: UserId) {}",
        );
        target.definitions = vec!["refresh_access_token".to_string()];
        let chunks = vec![
            target,
            CodeChunk::new(
                2,
                "src/db.rs",
                "rust",
                ChunkKind::Function,
                "fn store_session() {}",
            ),
        ];

        let index = LexicalIndex::build(&chunks);
        let hits = index.search("refresh_access_token", 5);
        assert_eq!(hits.first().map(|hit| hit.chunk_id), Some(1));
    }

    #[test]
    fn searches_path_terms() {
        let chunks = vec![
            CodeChunk::new(
                1,
                "src/auth/session.rs",
                "rust",
                ChunkKind::Function,
                "fn load() {}",
            ),
            CodeChunk::new(
                2,
                "src/db/pool.rs",
                "rust",
                ChunkKind::Function,
                "fn load() {}",
            ),
        ];

        let index = LexicalIndex::build(&chunks);
        let hits = index.search("auth session", 5);
        assert_eq!(hits.first().map(|hit| hit.chunk_id), Some(1));
    }

    #[test]
    fn finds_behavior_terms_from_comments() {
        let chunks = vec![
            CodeChunk::new(
                1,
                "src/retry.rs",
                "rust",
                ChunkKind::Function,
                "// Retry failed writes with backoff\nfn persist() {}",
            ),
            CodeChunk::new(
                2,
                "src/cache.rs",
                "rust",
                ChunkKind::Function,
                "fn cache_user() {}",
            ),
        ];

        let index = LexicalIndex::build(&chunks);
        let hits = index.search("retry failed writes", 5);
        assert_eq!(hits.first().map(|hit| hit.chunk_id), Some(1));
    }

    #[test]
    fn evidence_does_not_credit_substring_terms() {
        let chunks = vec![CodeChunk::new(
            1,
            "src/dispatch.rs",
            "rust",
            ChunkKind::Function,
            "fn dispatch_request() {}",
        )];

        let index = LexicalIndex::build(&chunks);
        let hits = index.search("it dispatch", 5);

        assert_eq!(hits[0].matched_terms, vec!["dispatch"]);
    }
}
