use super::{
    ChunkStore, CodeContext, ContextChunk, ContextOptions, ContextSource, ContextTarget, Evidence,
    HitSource, HybridEngine, RelationshipIndex, SearchError, SearchHit,
};
use std::collections::HashMap;

/// Policy-light context assembly over a chunk store and relationship index.
pub struct ContextAssembler<'a> {
    store: &'a ChunkStore,
    relationships: &'a RelationshipIndex,
}

impl<'a> ContextAssembler<'a> {
    pub fn new(store: &'a ChunkStore, relationships: &'a RelationshipIndex) -> Self {
        Self {
            store,
            relationships,
        }
    }

    pub fn assemble_targets(
        &self,
        targets: &[ContextTarget],
        options: &ContextOptions,
    ) -> Result<CodeContext, SearchError> {
        let mut target_chunks = Vec::new();

        for target in targets {
            match target {
                ContextTarget::Chunk(chunk_id) => {
                    if self.store.chunk(*chunk_id).is_none() {
                        return Err(SearchError::UnknownChunkId(*chunk_id));
                    }
                    target_chunks.push(vec![*chunk_id]);
                }
                ContextTarget::File {
                    path,
                    anchor_chunks,
                } => {
                    target_chunks.push(self.chunk_ids_for_context_file(
                        path,
                        anchor_chunks,
                        options,
                    ));
                }
            }
        }

        let seed_hits = distribute_seed_hits(&target_chunks, options.max_seed_hits);
        self.assemble_hits(&seed_hits, options)
            .map(|chunks| CodeContext { chunks })
    }

    pub fn assemble_hits(
        &self,
        seed_hits: &[SearchHit],
        options: &ContextOptions,
    ) -> Result<Vec<ContextChunk>, SearchError> {
        let mut context = Vec::new();
        if options.max_chunks == 0 {
            return Ok(context);
        }

        let seeds = seed_hits
            .iter()
            .take(options.max_seed_hits)
            .collect::<Vec<_>>();

        for seed in &seeds {
            push_context_chunk(
                &mut context,
                seed.chunk_id,
                ContextSource::Seed,
                seed.evidence.clone(),
                options.max_chunks,
            );
        }
        if context.len() >= options.max_chunks {
            return Ok(context);
        }

        let tokens = query_tokens(options.query.as_deref());
        let neighbors_by_seed = seeds
            .iter()
            .map(|seed| {
                self.ranked_context_neighbors(
                    seed.chunk_id,
                    options.max_neighbors_per_seed,
                    &options.allowed_neighbor_kinds,
                    &tokens,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        for neighbor_index in 0..options.max_neighbors_per_seed {
            for (seed, neighbors) in seeds.iter().zip(&neighbors_by_seed) {
                let Some(neighbor) = neighbors.get(neighbor_index) else {
                    continue;
                };
                push_context_chunk(
                    &mut context,
                    neighbor.chunk_id,
                    ContextSource::Neighbor {
                        seed_chunk_id: seed.chunk_id,
                        kind: neighbor.kind,
                        symbol: neighbor.symbol.clone(),
                    },
                    Vec::new(),
                    options.max_chunks,
                );
                if context.len() >= options.max_chunks {
                    return Ok(context);
                }
            }
        }

        Ok(context)
    }

    fn chunk_ids_for_context_file(
        &self,
        file_path: &str,
        anchor_chunks: &[super::ChunkId],
        options: &ContextOptions,
    ) -> Vec<super::ChunkId> {
        self.rank_chunks_in_file(
            file_path,
            anchor_chunks,
            options.max_seed_hits,
            &query_tokens(options.query.as_deref()),
        )
    }

    fn matching_file_chunks(&self, file_path: &str) -> Vec<&super::CodeChunk> {
        self.relationships
            .file_chunk_ids(file_path)
            .into_iter()
            .filter_map(|chunk_id| self.store.chunk(chunk_id))
            .collect()
    }

    fn rank_chunks_in_file(
        &self,
        file_path: &str,
        anchor_chunks: &[super::ChunkId],
        limit: usize,
        query_tokens: &[String],
    ) -> Vec<super::ChunkId> {
        let matches = self.matching_file_chunks(file_path);
        if matches.is_empty() || limit == 0 {
            return Vec::new();
        }

        let mut ranked_ids = Vec::new();
        let anchor_lines = anchor_chunks
            .iter()
            .filter_map(|chunk_id| self.store.chunk(*chunk_id))
            .filter(|chunk| chunk.file_path == file_path || chunk.file_path.ends_with(file_path))
            .map(|chunk| (chunk.id, chunk.start_line))
            .collect::<Vec<_>>();

        // Without a query, anchors are the only intent signal we have, so they
        // lead in their caller-provided order and structural priority fills the
        // rest. With a query, a chunk that strongly matches the query should be
        // able to outrank a weakly-related anchor, so anchors become a ranking
        // signal rather than an unconditional prefix.
        if query_tokens.is_empty() {
            for (id, _) in &anchor_lines {
                push_ranked_chunk_id(&mut ranked_ids, *id, limit);
            }
            if ranked_ids.len() >= limit {
                return ranked_ids;
            }
        }

        let mut ranked = matches
            .into_iter()
            .map(|chunk| {
                let is_anchor = anchor_lines.iter().any(|(id, _)| *id == chunk.id);
                let distance = anchor_lines
                    .iter()
                    .map(|(_, line)| chunk.start_line.abs_diff(*line))
                    .min()
                    .unwrap_or(u32::MAX);
                (
                    chunk.id,
                    chunk_query_match_score(chunk, query_tokens),
                    is_anchor,
                    distance,
                    structural_priority(chunk),
                    chunk.start_line,
                )
            })
            .collect::<Vec<_>>();

        ranked.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| right.2.cmp(&left.2))
                .then_with(|| left.3.cmp(&right.3))
                .then_with(|| right.4.cmp(&left.4))
                .then_with(|| left.5.cmp(&right.5))
                .then_with(|| left.0.cmp(&right.0))
        });

        for (chunk_id, ..) in ranked {
            push_ranked_chunk_id(&mut ranked_ids, chunk_id, limit);
            if ranked_ids.len() >= limit {
                break;
            }
        }

        ranked_ids
    }

    fn ranked_context_neighbors(
        &self,
        seed_chunk_id: super::ChunkId,
        limit: usize,
        allowed_kinds: &[super::NeighborKind],
        query_tokens: &[String],
    ) -> Result<Vec<super::ChunkNeighbor>, SearchError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let seed_chunk = self
            .store
            .chunk(seed_chunk_id)
            .ok_or(SearchError::UnknownChunkId(seed_chunk_id))?;
        let mut candidates = HashMap::<super::ChunkId, RankedNeighbor>::new();

        if allowed_kinds.contains(&super::NeighborKind::Parent)
            && let Some(parent_id) = self.relationships.parent_of(seed_chunk_id)
        {
            self.consider_context_candidate(
                &mut candidates,
                seed_chunk,
                super::ChunkNeighbor {
                    chunk_id: parent_id,
                    kind: super::NeighborKind::Parent,
                    symbol: None,
                },
                query_tokens,
            );
        }

        if allowed_kinds.contains(&super::NeighborKind::Child) {
            for child_id in self.relationships.children_of(seed_chunk_id) {
                self.consider_context_candidate(
                    &mut candidates,
                    seed_chunk,
                    super::ChunkNeighbor {
                        chunk_id: *child_id,
                        kind: super::NeighborKind::Child,
                        symbol: None,
                    },
                    query_tokens,
                );
            }
        }

        if allowed_kinds.contains(&super::NeighborKind::SameFile) {
            let nearby_limit = limit.saturating_mul(2).max(4);
            for same_file_id in self
                .relationships
                .nearby_file_chunk_ids(seed_chunk_id, nearby_limit)
            {
                self.consider_context_candidate(
                    &mut candidates,
                    seed_chunk,
                    super::ChunkNeighbor {
                        chunk_id: same_file_id,
                        kind: super::NeighborKind::SameFile,
                        symbol: None,
                    },
                    query_tokens,
                );
            }
        }

        for neighbor in self.relationships.related_neighbors(seed_chunk_id) {
            if !allowed_kinds.contains(&neighbor.kind) {
                continue;
            }
            self.consider_context_candidate(
                &mut candidates,
                seed_chunk,
                neighbor.clone(),
                query_tokens,
            );
        }

        let mut ranked = candidates.into_values().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.neighbor.chunk_id.cmp(&right.neighbor.chunk_id))
        });
        ranked.truncate(limit);
        Ok(ranked
            .into_iter()
            .map(|candidate| candidate.neighbor)
            .collect())
    }

    fn consider_context_candidate(
        &self,
        candidates: &mut HashMap<super::ChunkId, RankedNeighbor>,
        seed_chunk: &super::CodeChunk,
        neighbor: super::ChunkNeighbor,
        query_tokens: &[String],
    ) {
        let score = context_neighbor_rank(self, seed_chunk, &neighbor, query_tokens);
        let candidate = RankedNeighbor {
            neighbor: neighbor.clone(),
            score,
        };

        match candidates.get_mut(&neighbor.chunk_id) {
            Some(existing) if candidate.score > existing.score => *existing = candidate,
            None => {
                candidates.insert(neighbor.chunk_id, candidate);
            }
            _ => {}
        }
    }
}

impl HybridEngine {
    pub fn context_assembler(&self) -> ContextAssembler<'_> {
        ContextAssembler::new(self.chunk_store(), self.relationship_index())
    }

    pub fn get_context(
        &self,
        targets: &[ContextTarget],
        options: &ContextOptions,
    ) -> Result<CodeContext, SearchError> {
        self.context_assembler().assemble_targets(targets, options)
    }

    /// Assemble a bounded context window from caller-selected seed hits.
    pub fn context_from_hits(
        &self,
        seed_hits: &[SearchHit],
        options: &ContextOptions,
    ) -> Result<Vec<ContextChunk>, SearchError> {
        self.context_assembler().assemble_hits(seed_hits, options)
    }
}

fn push_seed_hit(seed_hits: &mut Vec<SearchHit>, chunk_id: super::ChunkId) {
    if !seed_hits.iter().any(|hit| hit.chunk_id == chunk_id) {
        seed_hits.push(SearchHit::new(
            chunk_id,
            1.0,
            HitSource::Similarity,
            Vec::new(),
        ));
    }
}

fn distribute_seed_hits(target_chunks: &[Vec<super::ChunkId>], limit: usize) -> Vec<SearchHit> {
    if limit == 0 {
        return Vec::new();
    }

    let mut seed_hits = Vec::new();
    let mut offset = 0usize;

    loop {
        let mut added = false;
        for chunk_ids in target_chunks {
            let Some(chunk_id) = chunk_ids.get(offset) else {
                continue;
            };
            push_seed_hit(&mut seed_hits, *chunk_id);
            added = true;
            if seed_hits.len() >= limit {
                return seed_hits;
            }
        }

        if !added {
            break;
        }
        offset += 1;
    }

    seed_hits
}

fn push_ranked_chunk_id(
    chunk_ids: &mut Vec<super::ChunkId>,
    chunk_id: super::ChunkId,
    limit: usize,
) {
    if chunk_ids.len() < limit && !chunk_ids.contains(&chunk_id) {
        chunk_ids.push(chunk_id);
    }
}

fn structural_priority(chunk: &super::CodeChunk) -> (u8, usize, usize, usize, bool) {
    (
        u8::from(chunk.parent_id.is_none()),
        chunk.definitions.len(),
        chunk.calls.len(),
        chunk.references.len() + chunk.imports.len(),
        !chunk.content.trim_start().starts_with("//"),
    )
}

type NeighborRank = (usize, u8, bool, bool, usize, std::cmp::Reverse<u32>, bool);

fn context_neighbor_rank(
    assembler: &ContextAssembler<'_>,
    seed_chunk: &super::CodeChunk,
    neighbor: &super::ChunkNeighbor,
    query_tokens: &[String],
) -> NeighborRank {
    let Some(neighbor_chunk) = assembler.store.chunk(neighbor.chunk_id) else {
        return (0, 0, false, false, 0, std::cmp::Reverse(u32::MAX), false);
    };

    let same_file = neighbor_chunk.file_path == seed_chunk.file_path;
    let distance = if same_file {
        neighbor_chunk.start_line.abs_diff(seed_chunk.start_line)
    } else {
        u32::MAX
    };
    let symbol_specificity = neighbor
        .symbol
        .as_deref()
        .map(context_symbol_specificity)
        .unwrap_or(0);
    let same_parent =
        seed_chunk.parent_id.is_some() && seed_chunk.parent_id == neighbor_chunk.parent_id;

    (
        chunk_query_match_score(neighbor_chunk, query_tokens),
        context_kind_priority(&neighbor.kind),
        same_file,
        same_parent,
        symbol_specificity,
        std::cmp::Reverse(distance),
        neighbor
            .symbol
            .as_deref()
            .is_some_and(is_context_specific_symbol),
    )
}

/// Tokens to drive query-aware re-ranking. Empty when the caller gave no query,
/// which makes every match count zero and preserves purely structural ordering.
fn query_tokens(query: Option<&str>) -> Vec<String> {
    query
        .map(super::tokenize::unique_tokens)
        .unwrap_or_default()
}

/// Score how well a chunk matches the query, for ranking only. Distinct-token
/// coverage dominates (how many of the query's terms appear at all), with a small
/// capped frequency bonus so a chunk that repeats the query terms wins ties. This
/// never admits a chunk that structural rules did not already surface.
fn chunk_query_match_score(chunk: &super::CodeChunk, tokens: &[String]) -> usize {
    if tokens.is_empty() {
        return 0;
    }

    let mut haystack = chunk.content.to_ascii_lowercase();
    for symbol in chunk
        .definitions
        .iter()
        .chain(chunk.references.iter())
        .chain(chunk.calls.iter())
        .chain(chunk.imports.iter())
    {
        haystack.push(' ');
        haystack.push_str(&symbol.to_ascii_lowercase());
    }

    let mut coverage = 0usize;
    let mut occurrences = 0usize;
    for token in tokens {
        let count = haystack.matches(token.as_str()).count();
        if count > 0 {
            coverage += 1;
            occurrences += count;
        }
    }

    coverage * 100 + occurrences.min(99)
}

fn context_kind_priority(kind: &super::NeighborKind) -> u8 {
    match kind {
        super::NeighborKind::Parent => 100,
        super::NeighborKind::Child => 95,
        super::NeighborKind::SameFile => 90,
        super::NeighborKind::Definition => 75,
        super::NeighborKind::Import => 60,
        super::NeighborKind::Call => 52,
        super::NeighborKind::Reference => 48,
    }
}

fn context_symbol_specificity(symbol: &str) -> usize {
    symbol
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
        .count()
}

fn is_context_specific_symbol(symbol: &str) -> bool {
    let symbol = symbol.trim().to_ascii_lowercase();
    symbol.len() > 4
        && !matches!(
            symbol.as_str(),
            "builder"
                | "bytes"
                | "config"
                | "context"
                | "error"
                | "future"
                | "header"
                | "headers"
                | "option"
                | "request"
                | "response"
                | "result"
                | "string"
                | "value"
        )
}

fn push_context_chunk(
    context: &mut Vec<ContextChunk>,
    chunk_id: u32,
    source: ContextSource,
    mut evidence: Vec<Evidence>,
    max_chunks: usize,
) {
    if context.len() < max_chunks && !context.iter().any(|chunk| chunk.chunk_id == chunk_id) {
        evidence.push(match &source {
            ContextSource::Seed => Evidence::ContextSeed,
            ContextSource::Neighbor {
                seed_chunk_id,
                kind,
                symbol,
            } => Evidence::ContextNeighbor {
                seed_chunk_id: *seed_chunk_id,
                kind: *kind,
                symbol: symbol.clone(),
            },
        });
        context.push(ContextChunk {
            chunk_id,
            source,
            evidence,
        });
    }
}

struct RankedNeighbor {
    neighbor: super::ChunkNeighbor,
    score: NeighborRank,
}

#[cfg(test)]
mod tests {
    use super::super::{ChunkKind, CodeChunk, EngineConfig, NeighborKind};
    use super::*;

    #[test]
    fn get_context_query_surfaces_matching_chunk_over_anchor_neighbors() {
        // A file whose query-relevant code lives far from the anchor. Without a
        // query, budget goes to the anchor and its nearest neighbor. With a
        // query, the matching chunk must outrank the unrelated near neighbor.
        let mut anchor = CodeChunk::new(
            1,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn alpha() {}",
        );
        anchor.start_line = 10;
        let mut near = CodeChunk::new(2, "src/lib.rs", "rust", ChunkKind::Function, "fn beta() {}");
        near.start_line = 20;
        let mut target = CodeChunk::new(
            3,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn render_widget() { widget_layout(); }",
        );
        target.start_line = 300;
        target.definitions.push("render_widget".to_string());

        let engine =
            HybridEngine::new(EngineConfig::default(), vec![anchor, near, target]).unwrap();
        let base = ContextOptions {
            max_chunks: 2,
            max_seed_hits: 2,
            max_neighbors_per_seed: 0,
            allowed_neighbor_kinds: Vec::new(),
            query: None,
        };
        let target_file = || {
            vec![ContextTarget::File {
                path: "src/lib.rs".to_string(),
                anchor_chunks: vec![1],
            }]
        };

        let without_query = engine.get_context(&target_file(), &base).unwrap();
        assert_eq!(
            without_query
                .chunks
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            vec![1, 2],
            "no query: anchor plus nearest neighbor"
        );

        let with_query = engine
            .get_context(&target_file(), &base.clone().with_query("widget"))
            .unwrap();
        assert!(
            with_query.chunks.iter().any(|chunk| chunk.chunk_id == 3),
            "query 'widget' should surface the matching chunk, got {:?}",
            with_query
                .chunks
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn builds_context_from_seed_hits_and_filtered_neighbors() {
        let mut parent = CodeChunk::new(1, "src/lib.rs", "rust", ChunkKind::Struct, "struct App;");
        let mut child = CodeChunk::new(
            2,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn run() { save_order(); }",
        );
        child.parent_id = Some(parent.id);
        child.calls.push("save_order".to_string());
        parent.definitions.push("App".to_string());
        let mut called = CodeChunk::new(
            3,
            "src/db.rs",
            "rust",
            ChunkKind::Function,
            "fn save_order() {}",
        );
        called.definitions.push("save_order".to_string());

        let engine = HybridEngine::new(EngineConfig::default(), vec![parent, child, called])
            .expect("engine builds");
        let hits = vec![SearchHit::new(
            2,
            1.0,
            super::HitSource::Lexical,
            Vec::new(),
        )];
        let options = ContextOptions {
            max_chunks: 4,
            max_seed_hits: 1,
            max_neighbors_per_seed: 8,
            allowed_neighbor_kinds: vec![NeighborKind::Parent, NeighborKind::Definition],
            query: None,
        };

        let context = engine
            .context_from_hits(&hits, &options)
            .expect("context builds");

        assert_eq!(
            context
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            vec![2, 1, 3]
        );
    }

    #[test]
    fn get_context_accepts_chunk_targets() {
        let chunk = CodeChunk::new(1, "src/lib.rs", "rust", ChunkKind::Function, "fn run() {}");
        let engine = HybridEngine::new(EngineConfig::default(), vec![chunk]).unwrap();

        let context = engine
            .get_context(&[ContextTarget::Chunk(1)], &ContextOptions::default())
            .unwrap();

        assert_eq!(
            context
                .chunks
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            vec![1]
        );
    }

    #[test]
    fn get_context_accepts_file_targets_with_budgeted_chunks() {
        let mut first = CodeChunk::new(
            1,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn first() {}",
        );
        first.start_line = 10;
        let mut second = CodeChunk::new(
            2,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn second() {}",
        );
        second.start_line = 20;
        let other = CodeChunk::new(
            3,
            "src/other.rs",
            "rust",
            ChunkKind::Function,
            "fn other() {}",
        );
        let engine = HybridEngine::new(EngineConfig::default(), vec![second, other, first])
            .expect("engine builds");
        let options = ContextOptions {
            max_chunks: 2,
            max_seed_hits: 2,
            max_neighbors_per_seed: 0,
            allowed_neighbor_kinds: Vec::new(),
            query: None,
        };

        let context = engine
            .get_context(
                &[ContextTarget::File {
                    path: "src/lib.rs".to_string(),
                    anchor_chunks: Vec::new(),
                }],
                &options,
            )
            .unwrap();

        assert_eq!(
            context
                .chunks
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn get_context_prefers_file_anchor_chunks_before_structural_fallback() {
        let mut first = CodeChunk::new(
            1,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn first() {}",
        );
        first.start_line = 10;
        let mut second = CodeChunk::new(
            2,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn second() {}",
        );
        second.start_line = 20;
        let mut third = CodeChunk::new(
            3,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn third() {}",
        );
        third.start_line = 30;
        let engine =
            HybridEngine::new(EngineConfig::default(), vec![first, second, third]).unwrap();
        let options = ContextOptions {
            max_chunks: 2,
            max_seed_hits: 2,
            max_neighbors_per_seed: 0,
            allowed_neighbor_kinds: Vec::new(),
            query: None,
        };

        let context = engine
            .get_context(
                &[ContextTarget::File {
                    path: "src/lib.rs".to_string(),
                    anchor_chunks: vec![3, 2],
                }],
                &options,
            )
            .unwrap();

        assert_eq!(
            context
                .chunks
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            vec![3, 2]
        );
    }

    #[test]
    fn get_context_distributes_file_target_seeds_round_robin() {
        let mut first_a = CodeChunk::new(1, "src/a.rs", "rust", ChunkKind::Function, "fn a1() {}");
        first_a.start_line = 10;
        let mut second_a = CodeChunk::new(2, "src/a.rs", "rust", ChunkKind::Function, "fn a2() {}");
        second_a.start_line = 20;
        let mut third_a = CodeChunk::new(3, "src/a.rs", "rust", ChunkKind::Function, "fn a3() {}");
        third_a.start_line = 30;
        let mut first_b = CodeChunk::new(4, "src/b.rs", "rust", ChunkKind::Function, "fn b1() {}");
        first_b.start_line = 10;
        let mut second_b = CodeChunk::new(5, "src/b.rs", "rust", ChunkKind::Function, "fn b2() {}");
        second_b.start_line = 20;

        let engine = HybridEngine::new(
            EngineConfig::default(),
            vec![first_a, second_a, third_a, first_b, second_b],
        )
        .unwrap();
        let options = ContextOptions {
            max_chunks: 4,
            max_seed_hits: 4,
            max_neighbors_per_seed: 0,
            allowed_neighbor_kinds: Vec::new(),
            query: None,
        };

        let context = engine
            .get_context(
                &[
                    ContextTarget::File {
                        path: "src/a.rs".to_string(),
                        anchor_chunks: vec![1, 2, 3],
                    },
                    ContextTarget::File {
                        path: "src/b.rs".to_string(),
                        anchor_chunks: vec![4, 5],
                    },
                ],
                &options,
            )
            .unwrap();

        assert_eq!(
            context
                .chunks
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            vec![1, 4, 2, 5]
        );
    }

    #[test]
    fn get_context_file_ranking_prefers_chunks_near_anchor_lines() {
        let mut first = CodeChunk::new(
            1,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn first() {}",
        );
        first.start_line = 10;
        let mut second = CodeChunk::new(
            2,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn second() {}",
        );
        second.start_line = 20;
        let mut third = CodeChunk::new(
            3,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn third() {}",
        );
        third.start_line = 30;
        let mut fourth = CodeChunk::new(
            4,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn fourth() {}",
        );
        fourth.start_line = 100;

        let engine = HybridEngine::new(
            EngineConfig::default(),
            vec![first, second, third.clone(), fourth],
        )
        .unwrap();
        let options = ContextOptions {
            max_chunks: 3,
            max_seed_hits: 3,
            max_neighbors_per_seed: 0,
            allowed_neighbor_kinds: Vec::new(),
            query: None,
        };

        let context = engine
            .get_context(
                &[ContextTarget::File {
                    path: "src/lib.rs".to_string(),
                    anchor_chunks: vec![third.id],
                }],
                &options,
            )
            .unwrap();

        assert_eq!(
            context
                .chunks
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
    }

    #[test]
    fn get_context_rejects_unknown_chunk_targets() {
        let engine = HybridEngine::new(EngineConfig::default(), Vec::new()).unwrap();

        assert!(matches!(
            engine.get_context(&[ContextTarget::Chunk(99)], &ContextOptions::default()),
            Err(SearchError::UnknownChunkId(99))
        ));
    }

    #[test]
    fn context_adds_all_seeds_before_any_neighbors() {
        let mut parent = CodeChunk::new(1, "src/lib.rs", "rust", ChunkKind::Struct, "struct App;");
        parent.definitions.push("app".to_string());
        let mut first = CodeChunk::new(
            2,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn first() { save_order(); }",
        );
        first.parent_id = Some(1);
        first.calls.push("save_order".to_string());
        let mut second = CodeChunk::new(
            3,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn second() { load_order(); }",
        );
        second.parent_id = Some(1);
        second.calls.push("load_order".to_string());
        let mut save = CodeChunk::new(
            4,
            "src/db.rs",
            "rust",
            ChunkKind::Function,
            "fn save_order() {}",
        );
        save.definitions.push("save_order".to_string());
        let mut load = CodeChunk::new(
            5,
            "src/db.rs",
            "rust",
            ChunkKind::Function,
            "fn load_order() {}",
        );
        load.definitions.push("load_order".to_string());

        let engine = HybridEngine::new(
            EngineConfig::default(),
            vec![parent, first, second, save, load],
        )
        .unwrap();
        let hits = vec![
            SearchHit::new(2, 1.0, super::HitSource::Lexical, Vec::new()),
            SearchHit::new(3, 0.9, super::HitSource::Lexical, Vec::new()),
        ];
        let options = ContextOptions {
            max_chunks: 2,
            max_seed_hits: 2,
            max_neighbors_per_seed: 2,
            allowed_neighbor_kinds: vec![NeighborKind::Parent, NeighborKind::Definition],
            query: None,
        };

        let context = engine.context_from_hits(&hits, &options).unwrap();

        assert_eq!(
            context
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn context_expands_neighbors_round_robin_across_seeds() {
        let parent = CodeChunk::new(1, "src/lib.rs", "rust", ChunkKind::Struct, "struct App;");
        let mut first = CodeChunk::new(
            2,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn first() { save_order(); }",
        );
        first.parent_id = Some(1);
        first.calls.push("save_order".to_string());
        let mut second = CodeChunk::new(
            3,
            "src/lib.rs",
            "rust",
            ChunkKind::Function,
            "fn second() { load_order(); }",
        );
        second.parent_id = Some(1);
        second.calls.push("load_order".to_string());
        let mut save = CodeChunk::new(
            4,
            "src/db.rs",
            "rust",
            ChunkKind::Function,
            "fn save_order() {}",
        );
        save.definitions.push("save_order".to_string());
        let mut load = CodeChunk::new(
            5,
            "src/db.rs",
            "rust",
            ChunkKind::Function,
            "fn load_order() {}",
        );
        load.definitions.push("load_order".to_string());

        let engine = HybridEngine::new(
            EngineConfig::default(),
            vec![parent, first, second, save, load],
        )
        .unwrap();
        let hits = vec![
            SearchHit::new(2, 1.0, super::HitSource::Lexical, Vec::new()),
            SearchHit::new(3, 0.9, super::HitSource::Lexical, Vec::new()),
        ];
        let options = ContextOptions {
            max_chunks: 5,
            max_seed_hits: 2,
            max_neighbors_per_seed: 2,
            allowed_neighbor_kinds: vec![NeighborKind::Parent, NeighborKind::Definition],
            query: None,
        };

        let context = engine.context_from_hits(&hits, &options).unwrap();

        assert_eq!(
            context
                .iter()
                .map(|chunk| chunk.chunk_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 1, 4, 5]
        );
    }
}
