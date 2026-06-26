use super::ChunkId;

/// Inspectable provenance explaining why a result was selected.
#[derive(Clone, Debug, PartialEq)]
pub enum Evidence {
    /// A primitive index matched terms in a chunk.
    Match {
        source: HitSource,
        matched_terms: Vec<String>,
    },
    /// Reciprocal-rank fusion contributed to the final score.
    RankContribution {
        source: HitSource,
        rank: usize,
        contribution: f32,
    },
    /// A graph edge supported selection of a related chunk or file.
    Relationship {
        from_chunk_id: ChunkId,
        kind: NeighborKind,
        symbol: Option<String>,
    },
    /// A chunk seeded graph traversal expansion.
    GraphSeed { seed_chunk_id: ChunkId },
    /// One hop taken during bounded graph traversal.
    GraphHop {
        from_chunk_id: ChunkId,
        to_chunk_id: ChunkId,
        kind: NeighborKind,
        direction: GraphStepDirection,
        symbol: Option<String>,
        depth: usize,
    },
    /// The caller selected this chunk as a context seed.
    ContextSeed,
    /// Structural expansion selected this context chunk.
    ContextNeighbor {
        seed_chunk_id: ChunkId,
        kind: NeighborKind,
        symbol: Option<String>,
    },
}

impl Evidence {
    /// Render stable human-readable provenance for traces and adapter reason fields.
    pub fn summary(&self) -> String {
        match self {
            Self::Match {
                source,
                matched_terms,
            } => format!(
                "{} match: {}",
                source.as_str(),
                if matched_terms.is_empty() {
                    "<no terms>".to_string()
                } else {
                    matched_terms.join(", ")
                }
            ),
            Self::RankContribution {
                source,
                rank,
                contribution,
            } => format!(
                "{} rank {} contribution {:.6}",
                source.as_str(),
                rank + 1,
                contribution
            ),
            Self::Relationship {
                from_chunk_id,
                kind,
                symbol,
            } => format!(
                "{} relationship from chunk {}{}",
                kind.as_str(),
                from_chunk_id,
                symbol
                    .as_deref()
                    .map(|symbol| format!(" via {symbol}"))
                    .unwrap_or_default()
            ),
            Self::GraphSeed { seed_chunk_id } => {
                format!("graph seed chunk {}", seed_chunk_id)
            }
            Self::GraphHop {
                from_chunk_id,
                to_chunk_id,
                kind,
                direction,
                symbol,
                depth,
            } => format!(
                "{} {} graph hop {}: {} -> {}{}",
                direction.as_str(),
                kind.as_str(),
                depth,
                from_chunk_id,
                to_chunk_id,
                symbol
                    .as_deref()
                    .map(|symbol| format!(" via {symbol}"))
                    .unwrap_or_default()
            ),
            Self::ContextSeed => "selected as context seed".to_string(),
            Self::ContextNeighbor {
                seed_chunk_id,
                kind,
                symbol,
            } => format!(
                "{} context neighbor of chunk {}{}",
                kind.as_str(),
                seed_chunk_id,
                symbol
                    .as_deref()
                    .map(|symbol| format!(" via {symbol}"))
                    .unwrap_or_default()
            ),
        }
    }
}

/// A file-level retrieval result with ranked anchors and structured provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct FileHit {
    pub file_path: String,
    pub score: f32,
    pub matched_terms: Vec<String>,
    pub anchor_chunks: Vec<ChunkId>,
    pub evidence: Vec<Evidence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HitSource {
    Lexical,
    Path,
    SymbolDefinition,
    SymbolReference,
    Call,
    Import,
    GraphTraversal,
    Similarity,
}

impl HitSource {
    /// Stable source name suitable for traces and serialized adapter output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Path => "path",
            Self::SymbolDefinition => "symbol-definition",
            Self::SymbolReference => "symbol-reference",
            Self::Call => "call",
            Self::Import => "import",
            Self::GraphTraversal => "graph",
            Self::Similarity => "fusion",
        }
    }
}

/// A chunk-level retrieval result with score and structured provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    pub chunk_id: ChunkId,
    pub score: f32,
    pub source: HitSource,
    pub matched_terms: Vec<String>,
    pub evidence: Vec<Evidence>,
}

impl SearchHit {
    /// Construct a primitive hit with evidence matching its legacy fields.
    pub fn new(
        chunk_id: ChunkId,
        score: f32,
        source: HitSource,
        matched_terms: Vec<String>,
    ) -> Self {
        let evidence = vec![Evidence::Match {
            source,
            matched_terms: matched_terms.clone(),
        }];
        Self {
            chunk_id,
            score,
            source,
            matched_terms,
            evidence,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchProfile {
    Agent,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchOptions {
    pub limit: usize,
    pub profile: SearchProfile,
}

impl SearchOptions {
    pub fn agent() -> Self {
        Self::default()
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            profile: SearchProfile::Agent,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NeighborKind {
    Parent,
    Child,
    SameFile,
    Definition,
    Reference,
    Call,
    Import,
}

impl NeighborKind {
    /// Stable relationship name suitable for traces and adapter output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Parent => "parent",
            Self::Child => "child",
            Self::SameFile => "same-file",
            Self::Definition => "definition",
            Self::Reference => "reference",
            Self::Call => "call",
            Self::Import => "import",
        }
    }
}

/// One typed relationship from a source chunk to a neighboring chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkNeighbor {
    pub chunk_id: ChunkId,
    pub kind: NeighborKind,
    pub symbol: Option<String>,
}

/// One exact or structural graph edge between code chunks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphEdge {
    pub from_chunk_id: ChunkId,
    pub to_chunk_id: ChunkId,
    pub kind: NeighborKind,
    pub symbol: Option<String>,
}

/// Direction used when traversing the relationship graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphDirection {
    Outgoing,
    Incoming,
    Both,
}

/// Per-hop direction recorded in graph traversal evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphStepDirection {
    Outgoing,
    Incoming,
}

impl GraphStepDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Outgoing => "outgoing",
            Self::Incoming => "incoming",
        }
    }
}

/// Options controlling bounded graph expansion from exact seed chunks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphOptions {
    /// Maximum number of hops from the provided seed chunks.
    pub max_hops: usize,
    /// Maximum number of results returned.
    pub max_results: usize,
    /// Maximum number of traversal states visited before stopping.
    pub max_visits: usize,
    /// Edge kinds allowed during graph traversal.
    pub allowed_kinds: Vec<NeighborKind>,
    /// Whether traversal follows outgoing, incoming, or both edge directions.
    pub direction: GraphDirection,
    /// Whether the provided seed chunks should be returned as graph hits.
    pub include_seeds: bool,
}

impl Default for GraphOptions {
    fn default() -> Self {
        Self {
            max_hops: 2,
            max_results: 20,
            max_visits: 200,
            allowed_kinds: vec![
                NeighborKind::Parent,
                NeighborKind::Child,
                NeighborKind::Definition,
                NeighborKind::Reference,
                NeighborKind::Call,
                NeighborKind::Import,
            ],
            direction: GraphDirection::Both,
            include_seeds: false,
        }
    }
}

/// Controls policy-light context assembly from caller-selected seed hits.
///
/// The library does not decide which search primitives to run. Callers pass
/// already-selected seed hits, and these options only bound structural
/// expansion around those hits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextOptions {
    /// Maximum number of chunks returned in the assembled context.
    pub max_chunks: usize,
    /// Maximum number of seed hits considered.
    pub max_seed_hits: usize,
    /// Maximum number of neighbors considered for each seed hit.
    pub max_neighbors_per_seed: usize,
    /// Neighbor kinds allowed during expansion.
    pub allowed_neighbor_kinds: Vec<NeighborKind>,
    /// Optional query intent used only to re-rank candidates that structural
    /// rules already admitted. This never triggers a new search; it only orders
    /// in-file chunks and neighbors so the window prefers ones that match the
    /// caller's query. `None` preserves purely structural ordering.
    pub query: Option<String>,
}

impl ContextOptions {
    /// Attach query intent for query-aware ranking, returning the updated options.
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        let query = query.into();
        self.query = if query.trim().is_empty() {
            None
        } else {
            Some(query)
        };
        self
    }
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            max_chunks: 10,
            max_seed_hits: 6,
            max_neighbors_per_seed: 3,
            allowed_neighbor_kinds: vec![
                NeighborKind::Parent,
                NeighborKind::Child,
                NeighborKind::SameFile,
                NeighborKind::Definition,
            ],
            query: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContextSource {
    /// The chunk came directly from the caller-provided seed hits.
    Seed,
    /// The chunk was added by structural expansion from a seed hit.
    Neighbor {
        seed_chunk_id: ChunkId,
        kind: NeighborKind,
        symbol: Option<String>,
    },
}

/// One chunk selected for an agent context window.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextChunk {
    pub chunk_id: ChunkId,
    pub source: ContextSource,
    pub evidence: Vec<Evidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContextTarget {
    File {
        path: String,
        anchor_chunks: Vec<ChunkId>,
    },
    Chunk(ChunkId),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CodeContext {
    pub chunks: Vec<ContextChunk>,
}
