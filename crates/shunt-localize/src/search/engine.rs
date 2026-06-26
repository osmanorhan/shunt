use super::chunk::{ChunkFileRecord, ChunkId, ChunkStore, CodeChunk};
use super::error::SearchError;
use super::graph::{RelationshipIndex, graph_hop_to_neighbor};
use super::lexical::LexicalIndex;
use super::symbol::SymbolIndex;
use super::types::{
    ChunkNeighbor, Evidence, FileHit, GraphDirection, GraphEdge, GraphOptions, GraphStepDirection,
    HitSource, NeighborKind, SearchHit, SearchOptions, SearchProfile,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct EngineConfig {
    _private: (),
}

pub trait SearchBackend {
    fn lexical(&self, query: &str, limit: usize) -> Vec<SearchHit>;
    fn path(&self, query: &str, limit: usize) -> Vec<SearchHit>;
    fn symbol(&self, query: &str, limit: usize) -> Vec<SearchHit>;
    fn references_to(&self, symbol: &str, limit: usize) -> Vec<SearchHit>;
    fn references_to_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit>;
    fn calls_to(&self, symbol: &str, limit: usize) -> Vec<SearchHit>;
    fn calls_to_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit>;
    fn imports_of(&self, symbol: &str, limit: usize) -> Vec<SearchHit>;
    fn imports_of_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit>;
    fn definitions_of(&self, symbol: &str, limit: usize) -> Vec<SearchHit>;
    fn definitions_of_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit>;
    fn children_of(&self, chunk_id: ChunkId) -> Result<Vec<ChunkId>, SearchError>;
    fn parent_of(&self, chunk_id: ChunkId) -> Result<Option<ChunkId>, SearchError>;
    fn neighbors_of(
        &self,
        chunk_id: ChunkId,
        limit: usize,
    ) -> Result<Vec<ChunkNeighbor>, SearchError>;
    fn traverse_graph(
        &self,
        seed_chunk_ids: &[ChunkId],
        options: &GraphOptions,
    ) -> Result<Vec<SearchHit>, SearchError>;
}

pub struct HybridEngine {
    store: ChunkStore,
    relationships: RelationshipIndex,
    lexical: LexicalIndex,
    symbols: SymbolIndex,
}

/// Deterministic reciprocal-rank fusion over primitive search results.
#[derive(Clone, Copy, Debug, Default)]
pub struct FusionRanker;

impl FusionRanker {
    pub fn fuse(&self, result_sets: &[Vec<SearchHit>], limit: usize) -> Vec<SearchHit> {
        fuse_hits_rrf(result_sets, limit)
    }
}

/// File-level aggregation over ranked chunk results and structural support.
pub struct FileRanker<'a> {
    store: &'a ChunkStore,
    relationships: &'a RelationshipIndex,
}

impl<'a> FileRanker<'a> {
    pub fn new(store: &'a ChunkStore, relationships: &'a RelationshipIndex) -> Self {
        Self {
            store,
            relationships,
        }
    }

    pub fn rank(&self, result_sets: &[Vec<SearchHit>], limit: usize) -> Vec<FileHit> {
        aggregate_file_hits(self.store, self.relationships, result_sets, limit)
    }
}

impl HybridEngine {
    pub fn new(_config: EngineConfig, chunks: Vec<CodeChunk>) -> Result<Self, SearchError> {
        Self::from_chunk_store(_config, ChunkStore::new(chunks)?)
    }

    pub fn from_chunk_store(_config: EngineConfig, store: ChunkStore) -> Result<Self, SearchError> {
        let lexical = LexicalIndex::build(store.chunks());
        let symbols = SymbolIndex::build(store.chunks());
        let relationships = RelationshipIndex::build(store.chunks(), &symbols);

        Ok(Self {
            store,
            relationships,
            lexical,
            symbols,
        })
    }

    pub fn chunks(&self) -> &[CodeChunk] {
        self.store.chunks()
    }

    pub fn active_chunks(&self) -> impl Iterator<Item = &CodeChunk> {
        self.store.active_chunks()
    }

    pub fn chunk(&self, chunk_id: ChunkId) -> Option<&CodeChunk> {
        self.store.chunk(chunk_id)
    }

    pub fn chunk_store(&self) -> &ChunkStore {
        &self.store
    }

    pub fn into_chunk_store(self) -> ChunkStore {
        self.store
    }

    /// Borrow the lexical primitive used by this engine.
    pub fn lexical_index(&self) -> &LexicalIndex {
        &self.lexical
    }

    /// Borrow the symbol primitive used by this engine.
    pub fn symbol_index(&self) -> &SymbolIndex {
        &self.symbols
    }

    /// Borrow the relationship primitive used by this engine.
    pub fn relationship_index(&self) -> &RelationshipIndex {
        &self.relationships
    }

    pub fn file_ranker(&self) -> FileRanker<'_> {
        FileRanker::new(&self.store, &self.relationships)
    }

    pub fn search(&self, query: &str, options: SearchOptions) -> Vec<SearchHit> {
        match options.profile {
            SearchProfile::Agent => self.search_agent(query, &options),
        }
    }

    pub fn search_files(&self, query: &str, limit: usize) -> Vec<FileHit> {
        if query.trim().is_empty() || limit == 0 {
            return Vec::new();
        }

        let search_limit = limit.saturating_mul(8).max(20);
        let result_sets = vec![
            self.path(query, search_limit),
            self.lexical(query, search_limit),
            self.symbol(query, search_limit),
        ];
        self.file_ranker().rank(&result_sets, limit)
    }

    pub fn search_code(&self, query: &str, options: SearchOptions) -> Vec<SearchHit> {
        self.search(query, options)
    }

    pub fn find_symbol(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.definitions_of_exact(symbol, limit)
    }

    pub fn find_callers(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.calls_to_exact(symbol, limit)
    }

    pub fn find_references(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.references_to_exact(symbol, limit)
    }

    pub fn graph_from_chunks(
        &self,
        seed_chunk_ids: &[ChunkId],
        options: &GraphOptions,
    ) -> Result<Vec<SearchHit>, SearchError> {
        self.traverse_graph(seed_chunk_ids, options)
    }

    pub fn graph_from_hits(
        &self,
        seed_hits: &[SearchHit],
        options: &GraphOptions,
    ) -> Result<Vec<SearchHit>, SearchError> {
        let mut seed_chunk_ids = Vec::with_capacity(seed_hits.len());
        for hit in seed_hits {
            if !seed_chunk_ids.contains(&hit.chunk_id) {
                seed_chunk_ids.push(hit.chunk_id);
            }
        }
        self.traverse_graph(&seed_chunk_ids, options)
    }

    pub fn save_snapshot(&self, path: &Path) -> Result<(), SearchError> {
        let snapshot = EngineSnapshot {
            version: 1,
            next_chunk_id: self.store.next_chunk_id(),
            chunks: self.store.chunks().to_vec(),
            file_records: self
                .store
                .file_records()
                .iter()
                .map(|(path, record)| SnapshotFileRecord {
                    path: path.to_string_lossy().into_owned(),
                    hash_hex: record.content_hash().to_hex().to_string(),
                    chunk_ids: record.chunk_ids().to_vec(),
                })
                .collect(),
        };

        let payload =
            serde_json::to_vec(&snapshot).map_err(|error| SearchError::SnapshotFormat {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        fs::write(path, payload).map_err(|error| SearchError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })
    }

    pub fn load_snapshot(path: &Path, config: EngineConfig) -> Result<Self, SearchError> {
        let payload = fs::read(path).map_err(|error| SearchError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        let snapshot: EngineSnapshot =
            serde_json::from_slice(&payload).map_err(|error| SearchError::SnapshotFormat {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;

        if snapshot.version != 1 {
            return Err(SearchError::SnapshotFormat {
                path: path.to_path_buf(),
                message: format!("unsupported snapshot version: {}", snapshot.version),
            });
        }

        let mut engine = Self::new(config, snapshot.chunks)?;
        engine.store.set_next_chunk_id(snapshot.next_chunk_id);
        engine.store.replace_file_records(
            snapshot
                .file_records
                .into_iter()
                .map(|record| {
                    let hash = blake3::Hash::from_hex(&record.hash_hex).map_err(|error| {
                        SearchError::SnapshotFormat {
                            path: path.to_path_buf(),
                            message: error.to_string(),
                        }
                    })?;
                    Ok((
                        std::path::PathBuf::from(record.path),
                        ChunkFileRecord::new(hash, record.chunk_ids),
                    ))
                })
                .collect::<Result<HashMap<_, _>, _>>()?,
        );
        Ok(engine)
    }

    fn search_agent(&self, query: &str, options: &SearchOptions) -> Vec<SearchHit> {
        let limit = options.limit.max(1);
        let search_limit = limit.saturating_mul(2).max(10);
        let result_sets = vec![
            self.path(query, search_limit),
            self.lexical(query, search_limit),
            self.symbol(query, search_limit),
        ];
        FusionRanker.fuse(&result_sets, limit)
    }

    pub(crate) fn chunk_store_mut(&mut self) -> &mut ChunkStore {
        &mut self.store
    }

    pub(crate) fn rebuild_indexes(&mut self) {
        self.lexical = LexicalIndex::build(self.store.chunks());
        self.symbols = SymbolIndex::build(self.store.chunks());
        self.relationships = RelationshipIndex::build(self.store.chunks(), &self.symbols);
    }
}

impl SearchBackend for HybridEngine {
    fn lexical(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        self.lexical.search(query, limit)
    }

    fn path(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        self.lexical.search_path(query, limit)
    }

    fn symbol(&self, query: &str, limit: usize) -> Vec<SearchHit> {
        self.symbols.search(query, limit)
    }

    fn references_to(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.symbols.references_to(symbol, limit)
    }

    fn references_to_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.symbols.references_to_exact(symbol, limit)
    }

    fn calls_to(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.symbols.calls_to(symbol, limit)
    }

    fn calls_to_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.symbols.calls_to_exact(symbol, limit)
    }

    fn imports_of(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.symbols.imports_of(symbol, limit)
    }

    fn imports_of_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.symbols.imports_of_exact(symbol, limit)
    }

    fn definitions_of(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.symbols.definitions_of(symbol, limit)
    }

    fn definitions_of_exact(&self, symbol: &str, limit: usize) -> Vec<SearchHit> {
        self.symbols.definitions_of_exact(symbol, limit)
    }

    fn children_of(&self, chunk_id: ChunkId) -> Result<Vec<ChunkId>, SearchError> {
        if self.store.chunk(chunk_id).is_none() {
            return Err(SearchError::UnknownChunkId(chunk_id));
        }

        Ok(self.relationships.children_of(chunk_id).to_vec())
    }

    fn parent_of(&self, chunk_id: ChunkId) -> Result<Option<ChunkId>, SearchError> {
        self.chunk(chunk_id)
            .ok_or(SearchError::UnknownChunkId(chunk_id))?;
        Ok(self.relationships.parent_of(chunk_id))
    }

    fn neighbors_of(
        &self,
        chunk_id: ChunkId,
        limit: usize,
    ) -> Result<Vec<ChunkNeighbor>, SearchError> {
        let chunk = self
            .chunk(chunk_id)
            .ok_or(SearchError::UnknownChunkId(chunk_id))?;
        let mut candidates = Vec::new();

        if let Some(parent_id) = self.relationships.parent_of(chunk_id) {
            push_neighbor_candidate(&mut candidates, parent_id, NeighborKind::Parent, None);
        }
        for child_id in self.relationships.children_of(chunk_id) {
            push_neighbor_candidate(&mut candidates, *child_id, NeighborKind::Child, None);
        }

        for sibling_id in self.relationships.file_chunk_ids(&chunk.file_path) {
            if let Some(sibling) = self.chunk(sibling_id) {
                if sibling.id == chunk.id {
                    continue;
                }
                let distance = sibling.start_line.abs_diff(chunk.start_line);
                push_neighbor_candidate_with_distance(
                    &mut candidates,
                    sibling_id,
                    NeighborKind::SameFile,
                    None,
                    distance,
                );
            }
        }

        for neighbor in self.relationships.related_neighbors(chunk_id) {
            push_neighbor_candidate(
                &mut candidates,
                neighbor.chunk_id,
                neighbor.kind,
                neighbor.symbol.clone(),
            );
        }

        candidates.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.distance.cmp(&right.distance))
                .then_with(|| left.neighbor.chunk_id.cmp(&right.neighbor.chunk_id))
        });
        let mut neighbors = Vec::new();
        for candidate in candidates {
            if neighbors.len() == limit {
                break;
            }
            if !neighbors.iter().any(|neighbor: &ChunkNeighbor| {
                neighbor.chunk_id == candidate.neighbor.chunk_id
                    && neighbor.kind == candidate.neighbor.kind
                    && neighbor.symbol == candidate.neighbor.symbol
            }) {
                neighbors.push(candidate.neighbor);
            }
        }

        Ok(neighbors)
    }

    fn traverse_graph(
        &self,
        seed_chunk_ids: &[ChunkId],
        options: &GraphOptions,
    ) -> Result<Vec<SearchHit>, SearchError> {
        for chunk_id in seed_chunk_ids {
            if self.chunk(*chunk_id).is_none() {
                return Err(SearchError::UnknownChunkId(*chunk_id));
            }
        }

        if seed_chunk_ids.is_empty() || options.max_results == 0 || options.max_visits == 0 {
            return Ok(Vec::new());
        }

        let allowed_kinds = options
            .allowed_kinds
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        if allowed_kinds.is_empty() {
            return Ok(Vec::new());
        }

        let mut seed_order = Vec::new();
        for chunk_id in seed_chunk_ids {
            if !seed_order.contains(chunk_id) {
                seed_order.push(*chunk_id);
            }
        }
        let seed_set = seed_order.iter().copied().collect::<HashSet<_>>();
        let mut frontier = VecDeque::new();
        let mut best_paths = HashMap::<ChunkId, GraphPathState>::new();
        let mut visit_count = 0usize;

        for seed_chunk_id in seed_order {
            frontier.push_back(GraphFrontierState {
                chunk_id: seed_chunk_id,
                depth: 0,
                priority: 0,
                path: Vec::new(),
            });
            if options.include_seeds {
                best_paths.insert(
                    seed_chunk_id,
                    GraphPathState {
                        depth: 0,
                        priority: 0,
                        path: Vec::new(),
                    },
                );
            }
        }

        while let Some(state) = frontier.pop_front() {
            if state.depth == options.max_hops || visit_count == options.max_visits {
                continue;
            }

            for candidate in graph_step_candidates(&self.relationships, state.chunk_id, options) {
                if !allowed_kinds.contains(&candidate.edge.kind) {
                    continue;
                }

                visit_count += 1;
                let next_depth = state.depth + 1;
                let next_priority = state.priority + graph_edge_priority(candidate.edge.kind);
                let next_chunk_id = candidate.neighbor.chunk_id;
                let mut next_path = state.path.clone();
                next_path.push(GraphPathStep {
                    edge: candidate.edge.clone(),
                    direction: candidate.direction,
                });

                if seed_set.contains(&next_chunk_id) && !options.include_seeds {
                    if next_depth < options.max_hops && visit_count < options.max_visits {
                        frontier.push_back(GraphFrontierState {
                            chunk_id: next_chunk_id,
                            depth: next_depth,
                            priority: next_priority,
                            path: next_path,
                        });
                    }
                    continue;
                }

                let should_replace = match best_paths.get(&next_chunk_id) {
                    Some(existing) => {
                        graph_path_better(next_depth, next_priority, &next_path, existing)
                    }
                    None => true,
                };
                if !should_replace {
                    continue;
                }

                best_paths.insert(
                    next_chunk_id,
                    GraphPathState {
                        depth: next_depth,
                        priority: next_priority,
                        path: next_path.clone(),
                    },
                );

                if next_depth < options.max_hops && visit_count < options.max_visits {
                    frontier.push_back(GraphFrontierState {
                        chunk_id: next_chunk_id,
                        depth: next_depth,
                        priority: next_priority,
                        path: next_path,
                    });
                }
            }
        }

        let mut hits = best_paths
            .into_iter()
            .map(|(chunk_id, path_state)| graph_hit_from_path(chunk_id, path_state))
            .collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.chunk_id.cmp(&right.chunk_id))
        });
        hits.truncate(options.max_results);
        Ok(hits)
    }
}

fn fuse_hits_rrf(result_sets: &[Vec<SearchHit>], limit: usize) -> Vec<SearchHit> {
    let mut scores: HashMap<ChunkId, f32> = HashMap::new();
    let mut matched_terms: HashMap<ChunkId, Vec<String>> = HashMap::new();
    let mut evidence: HashMap<ChunkId, Vec<Evidence>> = HashMap::new();
    const K: f32 = 60.0;

    for result_set in result_sets {
        for (rank, hit) in result_set.iter().enumerate() {
            let contribution = 1.0 / (K + rank as f32 + 1.0);
            *scores.entry(hit.chunk_id).or_insert(0.0) += contribution;
            let terms = matched_terms.entry(hit.chunk_id).or_default();
            for term in &hit.matched_terms {
                if !terms.contains(term) {
                    terms.push(term.clone());
                }
            }
            let hit_evidence = evidence.entry(hit.chunk_id).or_default();
            hit_evidence.extend(hit.evidence.clone());
            hit_evidence.push(Evidence::RankContribution {
                source: hit.source,
                rank,
                contribution,
            });
        }
    }

    let mut hits = scores
        .into_iter()
        .map(|(chunk_id, score)| SearchHit {
            chunk_id,
            score,
            source: HitSource::Similarity,
            matched_terms: matched_terms.remove(&chunk_id).unwrap_or_default(),
            evidence: evidence.remove(&chunk_id).unwrap_or_default(),
        })
        .collect::<Vec<_>>();

    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.chunk_id.cmp(&right.chunk_id))
    });
    hits.truncate(limit);
    hits
}

fn aggregate_file_hits(
    store: &ChunkStore,
    relationships: &RelationshipIndex,
    result_sets: &[Vec<SearchHit>],
    limit: usize,
) -> Vec<FileHit> {
    let mut records: HashMap<String, FileHitRecord> = HashMap::new();
    const K: f32 = 60.0;

    for result_set in result_sets {
        for (rank, hit) in result_set.iter().enumerate() {
            let Some(chunk) = store.chunk(hit.chunk_id) else {
                continue;
            };
            let direct_score = 1.0 / (K + rank as f32 + 1.0);
            let record = records.entry(chunk.file_path.clone()).or_default();
            let source_index = source_group_index(hit.source);
            record.best_direct_by_source[source_index] =
                record.best_direct_by_source[source_index].max(direct_score);
            if hit.source == HitSource::Path {
                record.path_hits += 1;
            }
            record.direct_source_mask |= source_mask(hit.source);
            *record.anchor_scores.entry(hit.chunk_id).or_insert(0.0) += direct_score;
            for term in &hit.matched_terms {
                push_unique_term(&mut record.matched_terms, term);
            }
            record.evidence.extend(hit.evidence.clone());
            record.evidence.push(Evidence::RankContribution {
                source: hit.source,
                rank,
                contribution: direct_score,
            });

            let mut best_support_by_file = HashMap::<String, StructuralSupport>::new();
            for neighbor in relationships.related_neighbors(hit.chunk_id) {
                let Some(target_chunk) = store.chunk(neighbor.chunk_id) else {
                    continue;
                };
                if target_chunk.file_path == chunk.file_path {
                    continue;
                }

                let support = best_support_by_file
                    .entry(target_chunk.file_path.clone())
                    .or_insert_with(|| StructuralSupport {
                        score: 0.0,
                        anchor_chunk: neighbor.chunk_id,
                        kind: neighbor.kind,
                        symbol: neighbor.symbol.clone(),
                    });
                let relation_score =
                    direct_score * structural_edge_weight(neighbor.kind, hit.source);
                if relation_score > support.score {
                    support.score = relation_score;
                    support.anchor_chunk = neighbor.chunk_id;
                    support.kind = neighbor.kind;
                    support.symbol = neighbor.symbol.clone();
                }
            }

            for (file_path, support) in best_support_by_file {
                let record = records.entry(file_path).or_default();
                record.best_structural_score = record.best_structural_score.max(support.score);
                *record
                    .anchor_scores
                    .entry(support.anchor_chunk)
                    .or_insert(0.0) += support.score;
                record.evidence.push(Evidence::Relationship {
                    from_chunk_id: hit.chunk_id,
                    kind: support.kind,
                    symbol: support.symbol,
                });
            }
        }
    }

    let mut hits = records
        .into_iter()
        .map(|(file_path, record)| {
            let direct_source_count = record.direct_source_mask.count_ones() as f32;
            let direct_quality = record.best_direct_by_source.iter().sum::<f32>();
            let structural_quality = if record.direct_source_mask != 0 {
                record.best_structural_score * 0.45
            } else {
                record.best_structural_score.min(0.010)
            };
            let stability_bonus =
                (record.path_hits.min(1) as f32 * 0.004) + (direct_source_count * 0.005);

            let mut anchor_chunks = record.anchor_scores.into_iter().collect::<Vec<_>>();
            anchor_chunks.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.0.cmp(&right.0))
            });

            FileHit {
                file_path,
                score: direct_quality + structural_quality + stability_bonus,
                matched_terms: record.matched_terms,
                anchor_chunks: anchor_chunks
                    .into_iter()
                    .map(|(chunk_id, _)| chunk_id)
                    .take(6)
                    .collect(),
                evidence: record.evidence,
            }
        })
        .collect::<Vec<_>>();

    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.file_path.cmp(&right.file_path))
    });
    hits.truncate(limit);
    hits
}

#[derive(Default)]
struct FileHitRecord {
    best_direct_by_source: [f32; 3],
    best_structural_score: f32,
    path_hits: usize,
    direct_source_mask: u8,
    matched_terms: Vec<String>,
    anchor_scores: HashMap<ChunkId, f32>,
    evidence: Vec<Evidence>,
}

struct StructuralSupport {
    score: f32,
    anchor_chunk: ChunkId,
    kind: NeighborKind,
    symbol: Option<String>,
}

fn structural_edge_weight(kind: NeighborKind, source: HitSource) -> f32 {
    let kind_weight = match kind {
        NeighborKind::Definition => 0.28,
        NeighborKind::Import => 0.20,
        NeighborKind::Reference => 0.16,
        NeighborKind::Call => 0.10,
        NeighborKind::Parent | NeighborKind::Child | NeighborKind::SameFile => 0.0,
    };
    let source_weight = match source {
        HitSource::Path => 0.20,
        HitSource::Lexical => 1.0,
        HitSource::SymbolDefinition
        | HitSource::SymbolReference
        | HitSource::Call
        | HitSource::Import
        | HitSource::GraphTraversal => 1.15,
        HitSource::Similarity => 0.8,
    };
    kind_weight * source_weight
}

fn push_unique_term(terms: &mut Vec<String>, term: &str) {
    if !terms.iter().any(|existing| existing == term) {
        terms.push(term.to_string());
    }
}

fn source_mask(source: HitSource) -> u8 {
    match source {
        HitSource::Path => 0b001,
        HitSource::Lexical => 0b010,
        HitSource::SymbolDefinition
        | HitSource::SymbolReference
        | HitSource::Call
        | HitSource::Import
        | HitSource::GraphTraversal
        | HitSource::Similarity => 0b100,
    }
}

fn source_group_index(source: HitSource) -> usize {
    match source {
        HitSource::Path => 0,
        HitSource::Lexical => 1,
        HitSource::SymbolDefinition
        | HitSource::SymbolReference
        | HitSource::Call
        | HitSource::Import
        | HitSource::GraphTraversal
        | HitSource::Similarity => 2,
    }
}

struct NeighborCandidate {
    neighbor: ChunkNeighbor,
    priority: u8,
    distance: u32,
}

#[derive(Clone)]
struct GraphPathStep {
    edge: GraphEdge,
    direction: GraphStepDirection,
}

struct GraphFrontierState {
    chunk_id: ChunkId,
    depth: usize,
    priority: u32,
    path: Vec<GraphPathStep>,
}

struct GraphPathState {
    depth: usize,
    priority: u32,
    path: Vec<GraphPathStep>,
}

struct GraphStepCandidate {
    edge: GraphEdge,
    direction: GraphStepDirection,
    neighbor: ChunkNeighbor,
}

#[derive(Deserialize, Serialize)]
struct EngineSnapshot {
    version: u32,
    next_chunk_id: ChunkId,
    chunks: Vec<CodeChunk>,
    file_records: Vec<SnapshotFileRecord>,
}

#[derive(Deserialize, Serialize)]
struct SnapshotFileRecord {
    path: String,
    hash_hex: String,
    chunk_ids: Vec<ChunkId>,
}

fn push_neighbor_candidate(
    candidates: &mut Vec<NeighborCandidate>,
    chunk_id: ChunkId,
    kind: NeighborKind,
    symbol: Option<String>,
) {
    push_neighbor_candidate_with_distance(candidates, chunk_id, kind, symbol, 0);
}

fn push_neighbor_candidate_with_distance(
    candidates: &mut Vec<NeighborCandidate>,
    chunk_id: ChunkId,
    kind: NeighborKind,
    symbol: Option<String>,
    distance: u32,
) {
    let priority = neighbor_priority(&kind);
    candidates.push(NeighborCandidate {
        neighbor: ChunkNeighbor {
            chunk_id,
            kind,
            symbol,
        },
        priority,
        distance,
    });
}

fn neighbor_priority(kind: &NeighborKind) -> u8 {
    match kind {
        NeighborKind::Parent => 100,
        NeighborKind::Child => 95,
        NeighborKind::Definition => 90,
        NeighborKind::SameFile => 80,
        NeighborKind::Call => 70,
        NeighborKind::Import => 60,
        NeighborKind::Reference => 50,
    }
}

fn graph_step_candidates(
    relationships: &RelationshipIndex,
    chunk_id: ChunkId,
    options: &GraphOptions,
) -> Vec<GraphStepCandidate> {
    let mut candidates = Vec::new();

    if matches!(
        options.direction,
        GraphDirection::Outgoing | GraphDirection::Both
    ) {
        for edge in relationships.outgoing_edges(chunk_id) {
            candidates.push(GraphStepCandidate {
                edge: edge.clone(),
                direction: GraphStepDirection::Outgoing,
                neighbor: graph_hop_to_neighbor(edge, GraphStepDirection::Outgoing),
            });
        }
    }

    if matches!(
        options.direction,
        GraphDirection::Incoming | GraphDirection::Both
    ) {
        for edge in relationships.incoming_edges(chunk_id) {
            candidates.push(GraphStepCandidate {
                edge: edge.clone(),
                direction: GraphStepDirection::Incoming,
                neighbor: graph_hop_to_neighbor(edge, GraphStepDirection::Incoming),
            });
        }
    }

    candidates.sort_by(|left, right| {
        graph_edge_priority(right.edge.kind)
            .cmp(&graph_edge_priority(left.edge.kind))
            .then_with(|| left.neighbor.chunk_id.cmp(&right.neighbor.chunk_id))
            .then_with(|| left.edge.symbol.cmp(&right.edge.symbol))
    });
    candidates.dedup_by(|left, right| {
        left.neighbor.chunk_id == right.neighbor.chunk_id
            && left.edge.kind == right.edge.kind
            && left.direction == right.direction
            && left.edge.symbol == right.edge.symbol
    });
    candidates
}

fn graph_edge_priority(kind: NeighborKind) -> u32 {
    match kind {
        NeighborKind::Definition => 70,
        NeighborKind::Reference => 60,
        NeighborKind::Call => 50,
        NeighborKind::Import => 40,
        NeighborKind::Parent => 30,
        NeighborKind::Child => 20,
        NeighborKind::SameFile => 10,
    }
}

fn graph_path_better(
    next_depth: usize,
    next_priority: u32,
    next_path: &[GraphPathStep],
    existing: &GraphPathState,
) -> bool {
    next_depth < existing.depth
        || (next_depth == existing.depth
            && (next_priority > existing.priority
                || (next_priority == existing.priority
                    && graph_path_signature(next_path) < graph_path_signature(&existing.path))))
}

fn graph_path_signature(
    path: &[GraphPathStep],
) -> Vec<(ChunkId, ChunkId, &'static str, &'static str, Option<&str>)> {
    path.iter()
        .map(|step| {
            (
                step.edge.from_chunk_id,
                step.edge.to_chunk_id,
                step.edge.kind.as_str(),
                step.direction.as_str(),
                step.edge.symbol.as_deref(),
            )
        })
        .collect()
}

fn graph_hit_from_path(chunk_id: ChunkId, path_state: GraphPathState) -> SearchHit {
    if path_state.path.is_empty() {
        return SearchHit {
            chunk_id,
            score: 1.0,
            source: HitSource::GraphTraversal,
            matched_terms: Vec::new(),
            evidence: vec![Evidence::GraphSeed {
                seed_chunk_id: chunk_id,
            }],
        };
    }

    let depth = path_state.depth.max(1);
    let score = 1.0 / depth as f32 + path_state.priority as f32 * 0.0001;
    let mut evidence = Vec::with_capacity(path_state.path.len());
    for (index, step) in path_state.path.into_iter().enumerate() {
        let (from_chunk_id, to_chunk_id) = match step.direction {
            GraphStepDirection::Outgoing => (step.edge.from_chunk_id, step.edge.to_chunk_id),
            GraphStepDirection::Incoming => (step.edge.to_chunk_id, step.edge.from_chunk_id),
        };
        evidence.push(Evidence::GraphHop {
            from_chunk_id,
            to_chunk_id,
            kind: step.edge.kind,
            direction: step.direction,
            symbol: step.edge.symbol,
            depth: index + 1,
        });
    }

    SearchHit {
        chunk_id,
        score,
        source: HitSource::GraphTraversal,
        matched_terms: Vec::new(),
        evidence,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        ChunkKind, CodeChunk, Evidence, GraphDirection, GraphOptions, NeighborKind, SearchHit,
        SearchOptions,
    };
    use super::{EngineConfig, HybridEngine, SearchBackend};
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn exposes_parent_child_relationships() {
        let mut parent = CodeChunk::new(
            1,
            "src/auth.rs",
            "rust",
            ChunkKind::Struct,
            "struct SessionManager;",
        );
        parent.start_line = 1;
        let mut child = CodeChunk::new(
            2,
            "src/auth.rs",
            "rust",
            ChunkKind::Function,
            "fn refresh_access_token() {}",
        );
        child.parent_id = Some(1);
        child.start_line = 3;

        let engine = HybridEngine::new(EngineConfig::default(), vec![parent, child]).unwrap();
        assert_eq!(engine.children_of(1).unwrap(), vec![2]);
        assert_eq!(engine.parent_of(2).unwrap(), Some(1));
    }

    #[test]
    fn search_code_wraps_agent_search() {
        let chunks = vec![CodeChunk::new(
            1,
            "src/auth.rs",
            "rust",
            ChunkKind::Function,
            "fn refresh_access_token() {}",
        )];
        let engine = HybridEngine::new(EngineConfig::default(), chunks).unwrap();

        assert_eq!(
            engine.search_code("refresh_access_token", SearchOptions::agent()),
            engine.search("refresh_access_token", SearchOptions::agent())
        );
    }

    #[test]
    fn search_files_aggregates_ranked_chunk_hits_by_file() {
        let chunks = vec![
            CodeChunk::new(
                1,
                "src/auth/session.rs",
                "rust",
                ChunkKind::Function,
                "fn refresh_access_token() {}",
            ),
            CodeChunk::new(
                2,
                "src/auth/session.rs",
                "rust",
                ChunkKind::Function,
                "fn validate_access_token() {}",
            ),
            CodeChunk::new(
                3,
                "src/db/pool.rs",
                "rust",
                ChunkKind::Function,
                "fn refresh_pool() {}",
            ),
        ];
        let engine = HybridEngine::new(EngineConfig::default(), chunks).unwrap();

        let hits = engine.search_files("auth access token", 5);

        assert_eq!(
            hits.first().map(|hit| hit.file_path.as_str()),
            Some("src/auth/session.rs")
        );
        assert_eq!(
            hits.iter()
                .filter(|hit| hit.file_path == "src/auth/session.rs")
                .count(),
            1
        );
    }

    #[test]
    fn search_files_uses_structural_support_for_related_files() {
        let mut caller = CodeChunk::new(
            1,
            "src/auth/login.rs",
            "rust",
            ChunkKind::Function,
            "fn login() { validate_password(); }",
        );
        caller.definitions = vec!["login".to_string()];
        caller.calls = vec!["validate_password".to_string()];

        let mut callee = CodeChunk::new(
            2,
            "src/auth/password.rs",
            "rust",
            ChunkKind::Function,
            "fn validate_password() {}",
        );
        callee.definitions = vec!["validate_password".to_string()];

        let distractor = CodeChunk::new(
            3,
            "src/ui/login_view.rs",
            "rust",
            ChunkKind::Function,
            "fn login_screen() {}",
        );

        let engine =
            HybridEngine::new(EngineConfig::default(), vec![caller, callee, distractor]).unwrap();

        let hits = engine.search_files("validate login", 3);

        let password_rank = hits
            .iter()
            .position(|hit| hit.file_path == "src/auth/password.rs")
            .expect("related file should appear");
        let distractor_rank = hits
            .iter()
            .position(|hit| hit.file_path == "src/ui/login_view.rs")
            .expect("lexical distractor should appear");

        assert!(hits[password_rank].anchor_chunks.contains(&2));
        assert!(password_rank < distractor_rank);
    }

    #[test]
    fn structural_symbols_are_evidence_not_query_matches() {
        let mut caller = CodeChunk::new(
            1,
            "src/service.rs",
            "rust",
            ChunkKind::Function,
            "fn refresh_workspace() { rebuild_index(); }",
        );
        caller.calls = vec!["rebuild_index".to_string()];
        let mut callee = CodeChunk::new(
            2,
            "src/index.rs",
            "rust",
            ChunkKind::Function,
            "fn rebuild_index() {}",
        );
        callee.definitions = vec!["rebuild_index".to_string()];
        let engine = HybridEngine::new(EngineConfig::default(), vec![caller, callee]).unwrap();

        let files = engine.file_ranker().rank(
            &[vec![SearchHit::new(
                1,
                1.0,
                super::HitSource::Lexical,
                vec!["refresh".to_string(), "workspace".to_string()],
            )]],
            5,
        );
        let related = files
            .iter()
            .find(|hit| hit.file_path == "src/index.rs")
            .expect("related definition should be present");

        assert!(related.matched_terms.is_empty());
        assert!(related.evidence.iter().any(|evidence| matches!(
            evidence,
            super::Evidence::Relationship {
                symbol: Some(symbol),
                ..
            } if symbol == "rebuild_index"
        )));
    }

    #[test]
    fn file_ranking_does_not_accumulate_repeated_weak_chunks() {
        let mut chunks = vec![CodeChunk::new(
            1,
            "src/target.rs",
            "rust",
            ChunkKind::Function,
            "fn target() {}",
        )];
        for id in 2..=12 {
            chunks.push(CodeChunk::new(
                id,
                "src/large.rs",
                "rust",
                ChunkKind::Function,
                format!("fn weak_{id}() {{}}"),
            ));
        }
        let engine = HybridEngine::new(EngineConfig::default(), chunks).unwrap();
        let mut result_set = vec![SearchHit::new(
            1,
            1.0,
            super::HitSource::Lexical,
            vec!["target".to_string()],
        )];
        result_set.extend((2..=12).map(|id| {
            SearchHit::new(id, 0.5, super::HitSource::Lexical, vec!["weak".to_string()])
        }));

        let files = engine.file_ranker().rank(&[result_set], 2);

        assert_eq!(files[0].file_path, "src/target.rs");
    }

    #[test]
    fn exact_agent_wrappers_use_exact_symbol_indexes() {
        let mut definition = CodeChunk::new(
            1,
            "src/auth.rs",
            "rust",
            ChunkKind::Function,
            "fn refresh_access_token() {}",
        );
        definition.definitions = vec!["refresh_access_token".to_string()];
        let mut caller = CodeChunk::new(
            2,
            "src/api.rs",
            "rust",
            ChunkKind::Function,
            "fn call_auth() { refresh_access_token(); }",
        );
        caller.calls = vec!["refresh_access_token".to_string()];
        caller.references = vec!["refresh_access_token".to_string()];
        let engine = HybridEngine::new(EngineConfig::default(), vec![definition, caller]).unwrap();

        assert_eq!(engine.find_symbol("refresh_access_token", 5)[0].chunk_id, 1);
        assert_eq!(
            engine.find_callers("refresh_access_token", 5)[0].chunk_id,
            2
        );
        assert!(
            engine
                .find_references("refresh_access_token", 5)
                .iter()
                .any(|hit| hit.chunk_id == 2)
        );
        assert!(engine.find_symbol("access", 5).is_empty());
    }

    #[test]
    fn exposes_neighbors_for_agent_expansion() {
        let mut parent =
            CodeChunk::new(1, "src/auth.rs", "rust", ChunkKind::Struct, "struct Auth;");
        let mut child = CodeChunk::new(
            2,
            "src/auth.rs",
            "rust",
            ChunkKind::Function,
            "fn login() { validate_password(); }",
        );
        child.parent_id = Some(1);
        child.definitions = vec!["login".to_string()];
        child.calls = vec!["validate_password".to_string()];
        let mut callee = CodeChunk::new(
            3,
            "src/password.rs",
            "rust",
            ChunkKind::Function,
            "fn validate_password() {}",
        );
        callee.definitions = vec!["validate_password".to_string()];
        parent.definitions = vec!["auth".to_string()];

        let engine =
            HybridEngine::new(EngineConfig::default(), vec![parent, child, callee]).unwrap();
        let neighbors = engine.neighbors_of(2, 10).unwrap();

        assert!(
            neighbors
                .iter()
                .any(|neighbor| neighbor.chunk_id == 1
                    && neighbor.kind == super::NeighborKind::Parent)
        );
        assert!(neighbors.iter().any(|neighbor| neighbor.chunk_id == 3
            && neighbor.kind == super::NeighborKind::Definition
            && neighbor.symbol.as_deref() == Some("validate_password")));
    }

    #[test]
    fn graph_traversal_expands_callers_of_callers_via_incoming_definition_edges() {
        let mut target = CodeChunk::new(
            1,
            "src/store.rs",
            "rust",
            ChunkKind::Function,
            "fn persist_order() {}",
        );
        target.definitions = vec!["persist_order".to_string()];

        let mut direct_caller = CodeChunk::new(
            2,
            "src/service.rs",
            "rust",
            ChunkKind::Function,
            "fn queue_order() { persist_order(); }",
        );
        direct_caller.definitions = vec!["queue_order".to_string()];
        direct_caller.calls = vec!["persist_order".to_string()];

        let mut upstream_caller = CodeChunk::new(
            3,
            "src/api.rs",
            "rust",
            ChunkKind::Function,
            "fn submit_order() { queue_order(); }",
        );
        upstream_caller.definitions = vec!["submit_order".to_string()];
        upstream_caller.calls = vec!["queue_order".to_string()];

        let engine = HybridEngine::new(
            EngineConfig::default(),
            vec![target, direct_caller, upstream_caller],
        )
        .unwrap();
        let direct_hits = engine.find_callers("persist_order", 5);

        let closure = engine
            .graph_from_hits(
                &direct_hits,
                &GraphOptions {
                    max_hops: 1,
                    max_results: 5,
                    max_visits: 20,
                    allowed_kinds: vec![NeighborKind::Definition],
                    direction: GraphDirection::Incoming,
                    include_seeds: false,
                },
            )
            .unwrap();

        assert_eq!(
            closure.iter().map(|hit| hit.chunk_id).collect::<Vec<_>>(),
            vec![3]
        );
        assert!(closure[0].evidence.iter().any(|evidence| matches!(
            evidence,
            Evidence::GraphHop {
                from_chunk_id: 2,
                to_chunk_id: 3,
                kind: NeighborKind::Definition,
                direction: super::GraphStepDirection::Incoming,
                symbol: Some(symbol),
                depth: 1,
            } if symbol == "queue_order"
        )));
    }

    #[test]
    fn graph_traversal_respects_hop_budgets_and_preserves_reference_then_caller_closure() {
        let mut definition = CodeChunk::new(
            1,
            "src/telemetry.rs",
            "rust",
            ChunkKind::Other("const".to_string()),
            "const DEFAULT_ENDPOINT: &str = \"https://ingest\";",
        );
        definition.definitions = vec!["DEFAULT_ENDPOINT".to_string()];

        let mut reference = CodeChunk::new(
            2,
            "src/telemetry.rs",
            "rust",
            ChunkKind::Function,
            "fn configure_default_client() -> &'static str { DEFAULT_ENDPOINT }",
        );
        reference.definitions = vec!["configure_default_client".to_string()];
        reference.references = vec!["DEFAULT_ENDPOINT".to_string()];

        let mut caller = CodeChunk::new(
            3,
            "src/app.rs",
            "rust",
            ChunkKind::Function,
            "fn boot_client() -> &'static str { configure_default_client() }",
        );
        caller.definitions = vec!["boot_client".to_string()];
        caller.calls = vec!["configure_default_client".to_string()];

        let engine =
            HybridEngine::new(EngineConfig::default(), vec![definition, reference, caller])
                .unwrap();

        let one_hop = engine
            .graph_from_chunks(
                &[1],
                &GraphOptions {
                    max_hops: 1,
                    max_results: 5,
                    max_visits: 20,
                    allowed_kinds: vec![NeighborKind::Reference],
                    direction: GraphDirection::Outgoing,
                    include_seeds: false,
                },
            )
            .unwrap();
        assert_eq!(
            one_hop.iter().map(|hit| hit.chunk_id).collect::<Vec<_>>(),
            vec![2]
        );
        assert!(one_hop[0].evidence.iter().any(|evidence| matches!(
            evidence,
            Evidence::GraphHop {
                from_chunk_id: 1,
                to_chunk_id: 2,
                kind: NeighborKind::Reference,
                direction: super::GraphStepDirection::Outgoing,
                symbol: Some(symbol),
                depth: 1,
            } if symbol == "default_endpoint"
        )));

        let two_hop = engine
            .graph_from_chunks(
                &[1],
                &GraphOptions {
                    max_hops: 2,
                    max_results: 1,
                    max_visits: 20,
                    allowed_kinds: vec![NeighborKind::Reference],
                    direction: GraphDirection::Outgoing,
                    include_seeds: false,
                },
            )
            .unwrap();

        assert_eq!(two_hop.len(), 1);
        assert_eq!(two_hop[0].chunk_id, 2);

        let expanded = engine
            .graph_from_hits(
                &one_hop,
                &GraphOptions {
                    max_hops: 1,
                    max_results: 5,
                    max_visits: 20,
                    allowed_kinds: vec![NeighborKind::Definition],
                    direction: GraphDirection::Incoming,
                    include_seeds: false,
                },
            )
            .unwrap();

        assert_eq!(
            expanded.iter().map(|hit| hit.chunk_id).collect::<Vec<_>>(),
            vec![3]
        );
        assert!(expanded[0].evidence.iter().any(|evidence| matches!(
            evidence,
            Evidence::GraphHop {
                from_chunk_id: 2,
                to_chunk_id: 3,
                kind: NeighborKind::Definition,
                direction: super::GraphStepDirection::Incoming,
                symbol: Some(symbol),
                depth: 1,
            } if symbol == "configure_default_client"
        )));
    }

    #[test]
    fn saves_and_loads_snapshot() {
        let mut definition = CodeChunk::new(
            1,
            "src/auth.rs",
            "rust",
            ChunkKind::Function,
            "fn refresh_access_token() {}",
        );
        definition.definitions = vec!["refresh_access_token".to_string()];
        let dir = tempdir().unwrap();
        let snapshot_path = dir.path().join("engine.json");

        let engine = HybridEngine::new(EngineConfig::default(), vec![definition]).unwrap();
        engine.save_snapshot(&snapshot_path).unwrap();

        let loaded = HybridEngine::load_snapshot(&snapshot_path, EngineConfig::default()).unwrap();
        assert_eq!(loaded.find_symbol("refresh_access_token", 5)[0].chunk_id, 1);
        assert!(
            loaded
                .chunk_store()
                .file_record(Path::new("src/auth.rs"))
                .is_some()
        );
    }
}
