use super::crawler::{CrawlConfig, crawl_files};
use super::parser::parse_file;
use super::{ChunkId, ChunkStore, EngineConfig, HybridEngine, SearchError};
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Compatibility configuration for constructing a complete engine from a workspace.
#[derive(Clone, Debug, Default)]
pub struct IngestConfig {
    pub engine: EngineConfig,
    pub crawl: CrawlConfig,
}

/// Counts produced by a full workspace ingest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IngestStats {
    pub files_seen: usize,
    pub files_parsed: usize,
    pub chunks_indexed: usize,
}

/// Counts produced by an incremental workspace refresh.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshStats {
    pub files_seen: usize,
    pub files_changed: usize,
    pub files_removed: usize,
    pub chunks_deactivated: usize,
    pub chunks_indexed: usize,
}

/// Deterministic filesystem ingestion and incremental refresh for a chunk store.
#[derive(Clone, Debug, Default)]
pub struct WorkspaceIngestor {
    crawl: CrawlConfig,
}

impl WorkspaceIngestor {
    /// Create an ingestor with explicit crawl policy.
    pub fn new(crawl: CrawlConfig) -> Self {
        Self { crawl }
    }

    /// Borrow the crawl policy used by this ingestor.
    pub fn crawl_config(&self) -> &CrawlConfig {
        &self.crawl
    }

    /// Parse a workspace into an independent chunk store.
    pub fn index_path(&self, root: &Path) -> Result<(ChunkStore, IngestStats), SearchError> {
        if !root.exists() || !root.is_dir() {
            return Err(SearchError::InvalidRoot(root.to_path_buf()));
        }

        let files = crawl_files(root, &self.crawl)?;
        let mut chunks = Vec::new();
        let mut next_id: ChunkId = 1;
        let mut files_parsed = 0;
        let mut file_hashes: HashMap<PathBuf, blake3::Hash> = HashMap::new();

        for path in &files {
            let Some(language) = language_for_path(path) else {
                continue;
            };
            let content = std::fs::read_to_string(path).map_err(|error| SearchError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
            file_hashes.insert(path.clone(), blake3::hash(content.as_bytes()));
            chunks.extend(parse_file(&mut next_id, path, language, &content)?);
            files_parsed += 1;
        }

        let stats = IngestStats {
            files_seen: files.len(),
            files_parsed,
            chunks_indexed: chunks.len(),
        };
        let mut store = ChunkStore::new(chunks)?;
        for (path, hash) in file_hashes {
            store.set_file_hash(&path, hash);
        }
        Ok((store, stats))
    }

    /// Apply changed and deleted files to an existing chunk store.
    pub fn refresh_path(
        &self,
        store: &mut ChunkStore,
        root: &Path,
    ) -> Result<RefreshStats, SearchError> {
        if !root.exists() || !root.is_dir() {
            return Err(SearchError::InvalidRoot(root.to_path_buf()));
        }

        let files = crawl_files(root, &self.crawl)?;
        let seen_paths = files.iter().cloned().collect::<HashSet<_>>();
        let mut files_changed = 0;
        let mut files_removed = 0;
        let mut chunks_deactivated = 0;
        let mut chunks_indexed = 0;

        for path in &files {
            let Some(language) = language_for_path(path) else {
                continue;
            };
            let content = std::fs::read_to_string(path).map_err(|error| SearchError::Io {
                path: path.clone(),
                message: error.to_string(),
            })?;
            let hash = blake3::hash(content.as_bytes());

            if store
                .file_record(path)
                .is_some_and(|record| record.content_hash() == hash)
            {
                continue;
            }

            let mut new_chunks = parse_file(store.next_chunk_id_mut(), path, language, &content)?;
            for chunk in &mut new_chunks {
                chunk.active = true;
            }

            files_changed += 1;
            chunks_indexed += new_chunks.len();
            chunks_deactivated += store.replace_file(path.clone(), hash, new_chunks);
        }

        let indexed_paths = store
            .indexed_paths()
            .map(Path::to_path_buf)
            .collect::<Vec<_>>();
        for indexed_path in indexed_paths {
            if !indexed_path.starts_with(root) || seen_paths.contains(&indexed_path) {
                continue;
            }
            let removed = store.remove_file(&indexed_path);
            if removed > 0 {
                files_removed += 1;
                chunks_deactivated += removed;
            }
        }

        Ok(RefreshStats {
            files_seen: files.len(),
            files_changed,
            files_removed,
            chunks_deactivated,
            chunks_indexed,
        })
    }
}

impl HybridEngine {
    pub fn index_path(
        root: &Path,
        config: IngestConfig,
    ) -> Result<(Self, IngestStats), SearchError> {
        let (store, stats) = WorkspaceIngestor::new(config.crawl).index_path(root)?;
        Ok((Self::from_chunk_store(config.engine, store)?, stats))
    }

    pub fn refresh_path(
        &mut self,
        root: &Path,
        crawl: &CrawlConfig,
    ) -> Result<RefreshStats, SearchError> {
        let stats =
            WorkspaceIngestor::new(crawl.clone()).refresh_path(self.chunk_store_mut(), root)?;
        if stats.files_changed > 0 || stats.files_removed > 0 {
            self.rebuild_indexes();
        }
        Ok(stats)
    }
}

fn language_for_path(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("rs") => Some("rust"),
        Some("js" | "mjs" | "cjs" | "jsx") => Some("javascript"),
        Some("ts") => Some("typescript"),
        Some("tsx") => Some("tsx"),
        Some("py") => Some("python"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{HybridEngine, SearchBackend};
    use super::IngestConfig;
    use std::fs;

    #[test]
    fn refresh_replaces_changed_file_chunks() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("lib.rs");
        fs::write(&source, "pub fn alphaunique() {}").unwrap();

        let (mut engine, stats) =
            HybridEngine::index_path(temp.path(), IngestConfig::default()).unwrap();
        assert_eq!(stats.files_parsed, 1);
        assert_eq!(engine.symbol("alphaunique", 5).len(), 1);

        fs::write(&source, "pub fn betaunique() {}").unwrap();
        let stats = engine
            .refresh_path(temp.path(), &IngestConfig::default().crawl)
            .unwrap();

        assert_eq!(stats.files_changed, 1);
        assert_eq!(stats.files_removed, 0);
        assert_eq!(engine.symbol("alphaunique", 5).len(), 0);
        assert_eq!(engine.symbol("betaunique", 5).len(), 1);
    }

    #[test]
    fn refresh_removes_deleted_file_chunks() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("lib.rs");
        fs::write(&source, "pub fn removable_symbol() {}").unwrap();

        let (mut engine, _) =
            HybridEngine::index_path(temp.path(), IngestConfig::default()).unwrap();
        assert_eq!(engine.symbol("removable_symbol", 5).len(), 1);

        fs::remove_file(&source).unwrap();
        let stats = engine
            .refresh_path(temp.path(), &IngestConfig::default().crawl)
            .unwrap();

        assert_eq!(stats.files_removed, 1);
        assert_eq!(engine.symbol("removable_symbol", 5).len(), 0);
    }

    #[test]
    fn refresh_batches_changed_and_deleted_files() {
        let temp = tempfile::tempdir().unwrap();
        let changed = temp.path().join("changed.rs");
        let removed = temp.path().join("removed.rs");
        fs::write(&changed, "pub fn firstunique() {}").unwrap();
        fs::write(&removed, "pub fn deleteunique() {}").unwrap();

        let (mut engine, _) =
            HybridEngine::index_path(temp.path(), IngestConfig::default()).unwrap();
        fs::write(&changed, "pub fn seconduq() {}").unwrap();
        fs::remove_file(&removed).unwrap();

        let stats = engine
            .refresh_path(temp.path(), &IngestConfig::default().crawl)
            .unwrap();

        assert_eq!(stats.files_changed, 1);
        assert_eq!(stats.files_removed, 1);
        assert_eq!(engine.symbol("firstunique", 5).len(), 0);
        assert_eq!(engine.symbol("deleteunique", 5).len(), 0);
        assert_eq!(engine.symbol("seconduq", 5).len(), 1);
    }

    #[test]
    fn refresh_updates_structured_symbol_indexes() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("lib.rs");
        fs::write(
            &source,
            r#"
            use super::db::Pool;

            pub fn caller(pool: Pool) {
                pool.execute();
            }
            "#,
        )
        .unwrap();

        let (mut engine, _) =
            HybridEngine::index_path(temp.path(), IngestConfig::default()).unwrap();
        assert_eq!(engine.definitions_of_exact("caller", 5).len(), 1);
        assert_eq!(engine.calls_to_exact("execute", 5).len(), 1);
        assert_eq!(engine.imports_of_exact("super::db::pool", 5).len(), 1);

        fs::write(
            &source,
            r#"
            use super::cache::Store;

            pub fn second(store: Store) {
                store.flush();
            }
            "#,
        )
        .unwrap();
        engine
            .refresh_path(temp.path(), &IngestConfig::default().crawl)
            .unwrap();

        assert_eq!(engine.definitions_of_exact("caller", 5).len(), 0);
        assert_eq!(engine.calls_to_exact("execute", 5).len(), 0);
        assert_eq!(engine.imports_of_exact("super::db::pool", 5).len(), 0);
        assert_eq!(engine.definitions_of_exact("second", 5).len(), 1);
        assert_eq!(engine.calls_to_exact("flush", 5).len(), 1);
        assert_eq!(engine.imports_of_exact("super::cache::store", 5).len(), 1);
    }

    #[test]
    fn indexes_typescript_and_javascript_files() {
        let temp = tempfile::tempdir().unwrap();
        let ts = temp.path().join("store.ts");
        let js = temp.path().join("server.js");
        fs::write(
            &ts,
            "export class SessionStore { loadUser() { return api.fetchUser(); } }",
        )
        .unwrap();
        fs::write(
            &js,
            "const renderPage = () => {}; const handle = () => renderPage();",
        )
        .unwrap();

        let (engine, stats) =
            HybridEngine::index_path(temp.path(), IngestConfig::default()).unwrap();

        assert_eq!(stats.files_parsed, 2);
        assert_eq!(engine.definitions_of_exact("sessionstore", 5).len(), 1);
        assert_eq!(engine.definitions_of_exact("loaduser", 5).len(), 1);
        assert_eq!(engine.definitions_of_exact("handle", 5).len(), 1);
        assert_eq!(engine.calls_to_exact("renderpage", 5).len(), 1);
    }

    #[test]
    fn indexes_python_files() {
        let temp = tempfile::tempdir().unwrap();
        let py = temp.path().join("service.py");
        fs::write(
            &py,
            "from client import ApiClient\n\nclass SessionStore:\n    def load_user(self, user_id):\n        return api.fetch_user(user_id)\n",
        )
        .unwrap();

        let (engine, stats) =
            HybridEngine::index_path(temp.path(), IngestConfig::default()).unwrap();

        assert_eq!(stats.files_parsed, 1);
        assert_eq!(engine.definitions_of_exact("sessionstore", 5).len(), 1);
        assert_eq!(engine.definitions_of_exact("load_user", 5).len(), 1);
        assert!(!engine.calls_to_exact("fetch_user", 5).is_empty());
    }
}
