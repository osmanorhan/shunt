mod chunk;
mod context;
mod crawler;
mod engine;
mod error;
mod graph;
mod ingest;
mod lexical;
mod lifecycle;
mod parser;
mod symbol;
mod tokenize;
mod types;

pub use chunk::{ChunkFileRecord, ChunkId, ChunkKind, ChunkStore, CodeChunk};
pub use context::ContextAssembler;
pub use crawler::CrawlConfig;
pub use engine::{EngineConfig, FileRanker, FusionRanker, HybridEngine, SearchBackend};
pub use error::SearchError;
pub use graph::RelationshipIndex;
pub use ingest::{IngestConfig, IngestStats, RefreshStats, WorkspaceIngestor};
pub use lexical::LexicalIndex;
pub use lifecycle::{
    LifecycleEvent, LifecycleEventKind, LifecycleObserver, NoopLifecycleObserver, WorkspaceConfig,
    WorkspaceFallbackReason, WorkspaceIndex, WorkspaceOpenMode, WorkspaceOpenReport,
    WorkspaceOpenResult, WorkspaceRefreshReport, WorkspaceSaveReport,
};
pub use symbol::SymbolIndex;
pub use types::{
    ChunkNeighbor, CodeContext, ContextChunk, ContextOptions, ContextSource, ContextTarget,
    Evidence, FileHit, GraphDirection, GraphEdge, GraphOptions, GraphStepDirection, HitSource,
    NeighborKind, SearchHit, SearchOptions, SearchProfile,
};
