use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::SearchError;

pub type ChunkId = u32;

/// Language-neutral structural category assigned by a parser.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ChunkKind {
    Function,
    Impl,
    Struct,
    Trait,
    Module,
    Comment,
    Other(String),
}

impl ChunkKind {
    pub fn as_tag(&self) -> &str {
        match self {
            Self::Function => "function",
            Self::Impl => "impl",
            Self::Struct => "struct",
            Self::Trait => "trait",
            Self::Module => "module",
            Self::Comment => "comment",
            Self::Other(tag) => tag.as_str(),
        }
    }
}

/// One parser-derived code region and its structured symbol metadata.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CodeChunk {
    pub id: ChunkId,
    pub file_path: String,
    pub language: String,
    pub chunk_type: ChunkKind,
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
    pub parent_id: Option<ChunkId>,
    pub definitions: Vec<String>,
    pub references: Vec<String>,
    pub calls: Vec<String>,
    pub imports: Vec<String>,
    pub active: bool,
}

impl CodeChunk {
    pub fn new(
        id: ChunkId,
        file_path: impl Into<String>,
        language: impl Into<String>,
        chunk_type: ChunkKind,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id,
            file_path: file_path.into(),
            language: language.into(),
            chunk_type,
            content: content.into(),
            start_line: 0,
            end_line: 0,
            parent_id: None,
            definitions: Vec::new(),
            references: Vec::new(),
            calls: Vec::new(),
            imports: Vec::new(),
            active: true,
        }
    }
}

/// Stored metadata for one indexed source file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkFileRecord {
    hash: blake3::Hash,
    chunk_ids: Vec<ChunkId>,
}

impl ChunkFileRecord {
    pub fn content_hash(&self) -> blake3::Hash {
        self.hash
    }

    pub fn chunk_ids(&self) -> &[ChunkId] {
        &self.chunk_ids
    }

    pub(crate) fn new(hash: blake3::Hash, chunk_ids: Vec<ChunkId>) -> Self {
        Self { hash, chunk_ids }
    }
}

/// Mutable source of truth for chunks, file membership, and chunk ID allocation.
pub struct ChunkStore {
    chunks: Vec<CodeChunk>,
    chunk_positions: HashMap<ChunkId, usize>,
    file_records: HashMap<PathBuf, ChunkFileRecord>,
    next_chunk_id: ChunkId,
}

impl ChunkStore {
    /// Build a store and validate that every chunk ID is unique.
    pub fn new(chunks: Vec<CodeChunk>) -> Result<Self, SearchError> {
        let mut chunk_positions = HashMap::with_capacity(chunks.len());
        let mut file_records: HashMap<PathBuf, ChunkFileRecord> = HashMap::new();
        let mut next_chunk_id = 1;

        for (index, chunk) in chunks.iter().enumerate() {
            if chunk_positions.insert(chunk.id, index).is_some() {
                return Err(SearchError::DuplicateChunkId(chunk.id));
            }
            next_chunk_id = next_chunk_id.max(chunk.id.saturating_add(1));
            file_records
                .entry(PathBuf::from(&chunk.file_path))
                .or_insert_with(|| ChunkFileRecord {
                    hash: blake3::hash(chunk.content.as_bytes()),
                    chunk_ids: Vec::new(),
                })
                .chunk_ids
                .push(chunk.id);
        }

        Ok(Self {
            chunks,
            chunk_positions,
            file_records,
            next_chunk_id,
        })
    }

    /// Return all chunks, including inactive historical chunks.
    pub fn chunks(&self) -> &[CodeChunk] {
        &self.chunks
    }

    /// Iterate over currently active chunks.
    pub fn active_chunks(&self) -> impl Iterator<Item = &CodeChunk> {
        self.chunks.iter().filter(|chunk| chunk.active)
    }

    /// Look up an active chunk by ID.
    pub fn chunk(&self, chunk_id: ChunkId) -> Option<&CodeChunk> {
        self.chunk_positions
            .get(&chunk_id)
            .and_then(|index| self.chunks.get(*index))
            .filter(|chunk| chunk.active)
    }

    /// Return indexed metadata for one source path.
    pub fn file_record(&self, path: &Path) -> Option<&ChunkFileRecord> {
        self.file_records.get(path)
    }

    /// Iterate over indexed source paths.
    pub fn indexed_paths(&self) -> impl Iterator<Item = &Path> {
        self.file_records.keys().map(PathBuf::as_path)
    }

    /// Return the next ID reserved for parser output.
    pub fn next_chunk_id(&self) -> ChunkId {
        self.next_chunk_id
    }

    pub(crate) fn next_chunk_id_mut(&mut self) -> &mut ChunkId {
        &mut self.next_chunk_id
    }

    pub(crate) fn set_next_chunk_id(&mut self, next_chunk_id: ChunkId) {
        self.next_chunk_id = self.next_chunk_id.max(next_chunk_id);
    }

    pub(crate) fn set_file_hash(&mut self, path: &Path, hash: blake3::Hash) {
        if let Some(record) = self.file_records.get_mut(path) {
            record.hash = hash;
        }
    }

    pub(crate) fn replace_file(
        &mut self,
        path: PathBuf,
        hash: blake3::Hash,
        new_chunks: Vec<CodeChunk>,
    ) -> usize {
        let chunk_ids_to_deactivate = self
            .file_records
            .get(&path)
            .map(|record| record.chunk_ids.clone())
            .unwrap_or_default();

        for chunk_id in &chunk_ids_to_deactivate {
            if let Some(index) = self.chunk_positions.get(chunk_id).copied()
                && let Some(chunk) = self.chunks.get_mut(index)
            {
                chunk.active = false;
            }
        }

        let mut new_chunk_ids = Vec::with_capacity(new_chunks.len());
        for chunk in new_chunks {
            let index = self.chunks.len();
            new_chunk_ids.push(chunk.id);
            self.chunk_positions.insert(chunk.id, index);
            self.chunks.push(chunk);
        }

        self.file_records.insert(
            path,
            ChunkFileRecord {
                hash,
                chunk_ids: new_chunk_ids,
            },
        );
        chunk_ids_to_deactivate.len()
    }

    pub(crate) fn remove_file(&mut self, path: &Path) -> usize {
        let Some(record) = self.file_records.remove(path) else {
            return 0;
        };

        let mut removed = 0;
        for chunk_id in record.chunk_ids {
            if let Some(index) = self.chunk_positions.get(&chunk_id).copied()
                && let Some(chunk) = self.chunks.get_mut(index)
                && chunk.active
            {
                chunk.active = false;
                removed += 1;
            }
        }
        removed
    }

    pub(crate) fn file_records(&self) -> &HashMap<PathBuf, ChunkFileRecord> {
        &self.file_records
    }

    pub(crate) fn replace_file_records(&mut self, file_records: HashMap<PathBuf, ChunkFileRecord>) {
        self.file_records = file_records;
    }
}
