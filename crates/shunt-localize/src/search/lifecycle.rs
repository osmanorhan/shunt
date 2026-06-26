use super::{HybridEngine, IngestConfig, RefreshStats, SearchError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const LAYOUT_VERSION: u32 = 1;
const SNAPSHOT_FILENAME: &str = "engine.snapshot.json";
const MANIFEST_FILENAME: &str = "manifest.json";

/// Configuration for the persisted workspace lifecycle.
#[derive(Clone, Debug)]
pub struct WorkspaceConfig {
    pub ingest: IngestConfig,
    pub storage_dir_name: String,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        Self {
            ingest: IngestConfig::default(),
            storage_dir_name: ".shunt/index".to_string(),
        }
    }
}

/// Persisted workspace handle owning one engine and its on-disk layout.
pub struct WorkspaceIndex {
    root: PathBuf,
    config: WorkspaceConfig,
    engine: HybridEngine,
}

impl WorkspaceIndex {
    /// Open a persisted workspace, falling back to a fresh rebuild on
    /// corrupt or version-mismatched persisted state.
    pub fn open(root: &Path, config: WorkspaceConfig) -> Result<WorkspaceOpenResult, SearchError> {
        Self::open_with_observer(root, config, &NoopLifecycleObserver)
    }

    /// Open a persisted workspace and emit lifecycle events to the observer.
    pub fn open_with_observer(
        root: &Path,
        config: WorkspaceConfig,
        observer: &dyn LifecycleObserver,
    ) -> Result<WorkspaceOpenResult, SearchError> {
        let started = Instant::now();
        let root = canonical_root(root)?;
        let layout = WorkspaceLayout::new(&root, &config.storage_dir_name);

        let (engine, mode, fallback_reason) =
            match load_engine_from_layout(&layout, &config.ingest)? {
                LoadWorkspaceState::Snapshot(engine) => {
                    (*engine, WorkspaceOpenMode::SnapshotLoad, None)
                }
                LoadWorkspaceState::Fresh => {
                    let (engine, stats) = HybridEngine::index_path(&root, config.ingest.clone())?;
                    persist_layout(&layout, &config, &engine)?;
                    let report = WorkspaceOpenReport {
                        mode: WorkspaceOpenMode::FreshIndex,
                        fallback_reason: None,
                        files_parsed: stats.files_parsed,
                        chunks_indexed: stats.chunks_indexed,
                        elapsed_ms: started.elapsed().as_millis(),
                    };
                    observer.record(&LifecycleEvent {
                        kind: LifecycleEventKind::Open,
                        root: root.clone(),
                        storage_dir: layout.storage_dir.clone(),
                        elapsed_ms: report.elapsed_ms,
                        open_mode: Some(report.mode),
                        fallback_reason: None,
                        files_parsed: report.files_parsed,
                        chunks_indexed: report.chunks_indexed,
                        files_changed: 0,
                        files_removed: 0,
                        snapshot_saved: true,
                    });
                    return Ok(WorkspaceOpenResult {
                        workspace: Self {
                            root,
                            config,
                            engine,
                        },
                        report,
                    });
                }
                LoadWorkspaceState::Fallback(reason) => {
                    let (engine, _stats) = HybridEngine::index_path(&root, config.ingest.clone())?;
                    persist_layout(&layout, &config, &engine)?;
                    (engine, WorkspaceOpenMode::FallbackRebuild, Some(reason))
                }
            };

        let (files_parsed, chunks_indexed) = active_counts(&engine);
        let report = WorkspaceOpenReport {
            mode,
            fallback_reason: fallback_reason.clone(),
            files_parsed,
            chunks_indexed,
            elapsed_ms: started.elapsed().as_millis(),
        };
        observer.record(&LifecycleEvent {
            kind: LifecycleEventKind::Open,
            root: root.clone(),
            storage_dir: layout.storage_dir.clone(),
            elapsed_ms: report.elapsed_ms,
            open_mode: Some(report.mode),
            fallback_reason,
            files_parsed: report.files_parsed,
            chunks_indexed: report.chunks_indexed,
            files_changed: 0,
            files_removed: 0,
            snapshot_saved: mode == WorkspaceOpenMode::FallbackRebuild,
        });
        Ok(WorkspaceOpenResult {
            workspace: Self {
                root,
                config,
                engine,
            },
            report,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn engine(&self) -> &HybridEngine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut HybridEngine {
        &mut self.engine
    }

    pub fn manifest_path(&self) -> PathBuf {
        WorkspaceLayout::new(&self.root, &self.config.storage_dir_name).manifest_path
    }

    pub fn snapshot_path(&self) -> PathBuf {
        WorkspaceLayout::new(&self.root, &self.config.storage_dir_name).snapshot_path
    }

    /// Persist the current in-memory engine and manifest layout.
    pub fn save(&self) -> Result<WorkspaceSaveReport, SearchError> {
        self.save_with_observer(&NoopLifecycleObserver)
    }

    /// Persist the current in-memory engine and emit a lifecycle event.
    pub fn save_with_observer(
        &self,
        observer: &dyn LifecycleObserver,
    ) -> Result<WorkspaceSaveReport, SearchError> {
        let started = Instant::now();
        let layout = WorkspaceLayout::new(&self.root, &self.config.storage_dir_name);
        persist_layout(&layout, &self.config, &self.engine)?;
        let (files_parsed, chunks_indexed) = active_counts(&self.engine);
        let report = WorkspaceSaveReport {
            manifest_path: layout.manifest_path.clone(),
            snapshot_path: layout.snapshot_path.clone(),
            files_parsed,
            chunks_indexed,
            elapsed_ms: started.elapsed().as_millis(),
        };
        observer.record(&LifecycleEvent {
            kind: LifecycleEventKind::Save,
            root: self.root.clone(),
            storage_dir: layout.storage_dir,
            elapsed_ms: report.elapsed_ms,
            open_mode: None,
            fallback_reason: None,
            files_parsed,
            chunks_indexed,
            files_changed: 0,
            files_removed: 0,
            snapshot_saved: true,
        });
        Ok(report)
    }

    /// Refresh the workspace from disk and persist the updated layout when it changes.
    pub fn refresh(&mut self) -> Result<WorkspaceRefreshReport, SearchError> {
        self.refresh_with_observer(&NoopLifecycleObserver)
    }

    /// Refresh the workspace from disk, persist on change, and emit a lifecycle event.
    pub fn refresh_with_observer(
        &mut self,
        observer: &dyn LifecycleObserver,
    ) -> Result<WorkspaceRefreshReport, SearchError> {
        let started = Instant::now();
        let stats = self
            .engine
            .refresh_path(&self.root, &self.config.ingest.crawl)?;
        let snapshot_saved = stats.files_changed > 0 || stats.files_removed > 0;
        if snapshot_saved {
            let layout = WorkspaceLayout::new(&self.root, &self.config.storage_dir_name);
            persist_layout(&layout, &self.config, &self.engine)?;
        }
        let (files_parsed, chunks_indexed) = active_counts(&self.engine);
        let report = WorkspaceRefreshReport {
            stats,
            snapshot_saved,
            files_parsed,
            chunks_indexed,
            elapsed_ms: started.elapsed().as_millis(),
        };
        observer.record(&LifecycleEvent {
            kind: LifecycleEventKind::Refresh,
            root: self.root.clone(),
            storage_dir: WorkspaceLayout::new(&self.root, &self.config.storage_dir_name)
                .storage_dir,
            elapsed_ms: report.elapsed_ms,
            open_mode: None,
            fallback_reason: None,
            files_parsed: report.files_parsed,
            chunks_indexed: report.chunks_indexed,
            files_changed: report.stats.files_changed,
            files_removed: report.stats.files_removed,
            snapshot_saved: report.snapshot_saved,
        });
        Ok(report)
    }

    pub fn into_engine(self) -> HybridEngine {
        self.engine
    }
}

/// Result returned by `WorkspaceIndex::open`.
pub struct WorkspaceOpenResult {
    pub workspace: WorkspaceIndex,
    pub report: WorkspaceOpenReport,
}

/// Startup summary for one workspace open.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceOpenReport {
    pub mode: WorkspaceOpenMode,
    pub fallback_reason: Option<WorkspaceFallbackReason>,
    pub files_parsed: usize,
    pub chunks_indexed: usize,
    pub elapsed_ms: u128,
}

/// Persistence summary for one explicit save.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceSaveReport {
    pub manifest_path: PathBuf,
    pub snapshot_path: PathBuf,
    pub files_parsed: usize,
    pub chunks_indexed: usize,
    pub elapsed_ms: u128,
}

/// Refresh summary including whether persisted state was updated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkspaceRefreshReport {
    pub stats: RefreshStats,
    pub snapshot_saved: bool,
    pub files_parsed: usize,
    pub chunks_indexed: usize,
    pub elapsed_ms: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkspaceOpenMode {
    FreshIndex,
    SnapshotLoad,
    FallbackRebuild,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkspaceFallbackReason {
    ManifestMissing,
    ManifestFormat(String),
    ManifestVersionMismatch {
        found: u32,
    },
    SnapshotMissing,
    RootMismatch {
        manifest_root: String,
        requested_root: String,
    },
    SnapshotLoad(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LifecycleEventKind {
    Open,
    Refresh,
    Save,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleEvent {
    pub kind: LifecycleEventKind,
    pub root: PathBuf,
    pub storage_dir: PathBuf,
    pub elapsed_ms: u128,
    pub open_mode: Option<WorkspaceOpenMode>,
    pub fallback_reason: Option<WorkspaceFallbackReason>,
    pub files_parsed: usize,
    pub chunks_indexed: usize,
    pub files_changed: usize,
    pub files_removed: usize,
    pub snapshot_saved: bool,
}

/// Observer hook for lifecycle timings and counts without adding a logging dependency.
pub trait LifecycleObserver {
    fn record(&self, event: &LifecycleEvent);
}

pub struct NoopLifecycleObserver;

impl LifecycleObserver for NoopLifecycleObserver {
    fn record(&self, _event: &LifecycleEvent) {}
}

#[derive(Deserialize, Serialize)]
struct WorkspaceManifest {
    layout_version: u32,
    root: String,
    snapshot_file: String,
    crawl: ManifestCrawlConfig,
}

#[derive(Deserialize, Serialize)]
struct ManifestCrawlConfig {
    extensions: Vec<String>,
    hidden: bool,
}

struct WorkspaceLayout {
    storage_dir: PathBuf,
    manifest_path: PathBuf,
    snapshot_path: PathBuf,
}

enum LoadWorkspaceState {
    Fresh,
    Snapshot(Box<HybridEngine>),
    Fallback(WorkspaceFallbackReason),
}

impl WorkspaceLayout {
    fn new(root: &Path, storage_dir_name: &str) -> Self {
        let storage_dir = root.join(storage_dir_name);
        Self {
            manifest_path: storage_dir.join(MANIFEST_FILENAME),
            snapshot_path: storage_dir.join(SNAPSHOT_FILENAME),
            storage_dir,
        }
    }
}

fn canonical_root(root: &Path) -> Result<PathBuf, SearchError> {
    if !root.exists() || !root.is_dir() {
        return Err(SearchError::InvalidRoot(root.to_path_buf()));
    }
    fs::canonicalize(root).map_err(|error| SearchError::Io {
        path: root.to_path_buf(),
        message: error.to_string(),
    })
}

fn load_engine_from_layout(
    layout: &WorkspaceLayout,
    ingest: &IngestConfig,
) -> Result<LoadWorkspaceState, SearchError> {
    let manifest_exists = layout.manifest_path.is_file();
    let snapshot_exists = layout.snapshot_path.is_file();

    if !manifest_exists && !snapshot_exists {
        return Ok(LoadWorkspaceState::Fresh);
    }
    if !manifest_exists {
        return Ok(LoadWorkspaceState::Fallback(
            WorkspaceFallbackReason::ManifestMissing,
        ));
    }
    if !snapshot_exists {
        return Ok(LoadWorkspaceState::Fallback(
            WorkspaceFallbackReason::SnapshotMissing,
        ));
    }

    let manifest = match read_manifest(&layout.manifest_path) {
        Ok(manifest) => manifest,
        Err(error) => {
            let message = match error {
                SearchError::ManifestFormat { message, .. } => message,
                other => other.to_string(),
            };
            return Ok(LoadWorkspaceState::Fallback(
                WorkspaceFallbackReason::ManifestFormat(message),
            ));
        }
    };

    if manifest.layout_version != LAYOUT_VERSION {
        return Ok(LoadWorkspaceState::Fallback(
            WorkspaceFallbackReason::ManifestVersionMismatch {
                found: manifest.layout_version,
            },
        ));
    }

    let requested_root = layout
        .storage_dir
        .parent()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    if manifest.root != requested_root {
        return Ok(LoadWorkspaceState::Fallback(
            WorkspaceFallbackReason::RootMismatch {
                manifest_root: manifest.root,
                requested_root,
            },
        ));
    }

    match HybridEngine::load_snapshot(&layout.snapshot_path, ingest.engine.clone()) {
        Ok(engine) => Ok(LoadWorkspaceState::Snapshot(Box::new(engine))),
        Err(error) => Ok(LoadWorkspaceState::Fallback(
            WorkspaceFallbackReason::SnapshotLoad(error.to_string()),
        )),
    }
}

fn persist_layout(
    layout: &WorkspaceLayout,
    config: &WorkspaceConfig,
    engine: &HybridEngine,
) -> Result<(), SearchError> {
    fs::create_dir_all(&layout.storage_dir).map_err(|error| SearchError::Io {
        path: layout.storage_dir.clone(),
        message: error.to_string(),
    })?;
    engine.save_snapshot(&layout.snapshot_path)?;
    write_manifest(&layout.manifest_path, layout, config)
}

fn read_manifest(path: &Path) -> Result<WorkspaceManifest, SearchError> {
    let payload = fs::read(path).map_err(|error| SearchError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    serde_json::from_slice(&payload).map_err(|error| SearchError::ManifestFormat {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

fn write_manifest(
    path: &Path,
    layout: &WorkspaceLayout,
    config: &WorkspaceConfig,
) -> Result<(), SearchError> {
    let mut extensions = config
        .ingest
        .crawl
        .extensions
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    extensions.sort();
    let manifest = WorkspaceManifest {
        layout_version: LAYOUT_VERSION,
        root: layout
            .storage_dir
            .parent()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        snapshot_file: SNAPSHOT_FILENAME.to_string(),
        crawl: ManifestCrawlConfig {
            extensions,
            hidden: config.ingest.crawl.hidden,
        },
    };
    let payload = serde_json::to_vec(&manifest).map_err(|error| SearchError::ManifestFormat {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    fs::write(path, payload).map_err(|error| SearchError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}

fn active_counts(engine: &HybridEngine) -> (usize, usize) {
    let files_parsed = engine
        .active_chunks()
        .map(|chunk| &chunk.file_path)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let chunks_indexed = engine.active_chunks().count();
    (files_parsed, chunks_indexed)
}

#[cfg(test)]
mod tests {
    use super::{
        LifecycleEvent, LifecycleEventKind, LifecycleObserver, WorkspaceConfig,
        WorkspaceFallbackReason, WorkspaceIndex, WorkspaceOpenMode,
    };
    use std::fs;
    use std::path::Path;
    use std::sync::Mutex;

    #[test]
    fn open_creates_workspace_layout_and_reuses_snapshot() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("lib.rs"), "pub fn first_symbol() {}").unwrap();

        let opened = WorkspaceIndex::open(temp.path(), WorkspaceConfig::default()).unwrap();
        assert_eq!(opened.report.mode, WorkspaceOpenMode::FreshIndex);
        assert!(opened.workspace.manifest_path().is_file());
        assert!(opened.workspace.snapshot_path().is_file());

        let reopened = WorkspaceIndex::open(temp.path(), WorkspaceConfig::default()).unwrap();
        assert_eq!(reopened.report.mode, WorkspaceOpenMode::SnapshotLoad);
        assert!(
            reopened
                .workspace
                .engine()
                .find_symbol("first_symbol", 5)
                .len()
                == 1
        );
    }

    #[test]
    fn open_falls_back_on_corrupt_manifest() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("lib.rs"), "pub fn first_symbol() {}").unwrap();

        let opened = WorkspaceIndex::open(temp.path(), WorkspaceConfig::default()).unwrap();
        fs::write(opened.workspace.manifest_path(), "{ not json").unwrap();

        let reopened = WorkspaceIndex::open(temp.path(), WorkspaceConfig::default()).unwrap();
        assert_eq!(reopened.report.mode, WorkspaceOpenMode::FallbackRebuild);
        assert!(matches!(
            reopened.report.fallback_reason,
            Some(WorkspaceFallbackReason::ManifestFormat(_))
        ));
    }

    #[test]
    fn open_falls_back_on_corrupt_snapshot() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("lib.rs"), "pub fn first_symbol() {}").unwrap();

        let opened = WorkspaceIndex::open(temp.path(), WorkspaceConfig::default()).unwrap();
        fs::write(opened.workspace.snapshot_path(), "{ not json").unwrap();

        let reopened = WorkspaceIndex::open(temp.path(), WorkspaceConfig::default()).unwrap();
        assert_eq!(reopened.report.mode, WorkspaceOpenMode::FallbackRebuild);
        assert!(matches!(
            reopened.report.fallback_reason,
            Some(WorkspaceFallbackReason::SnapshotLoad(_))
        ));
    }

    #[test]
    fn refresh_persists_changes_for_next_open() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("lib.rs");
        fs::write(&source, "pub fn alpha_symbol() {}").unwrap();

        let opened = WorkspaceIndex::open(temp.path(), WorkspaceConfig::default()).unwrap();
        let mut workspace = opened.workspace;
        fs::write(&source, "pub fn beta_symbol() {}").unwrap();

        let refresh = workspace.refresh().unwrap();
        assert_eq!(refresh.stats.files_changed, 1);
        assert!(refresh.snapshot_saved);

        let reopened = WorkspaceIndex::open(temp.path(), WorkspaceConfig::default()).unwrap();
        assert!(
            reopened
                .workspace
                .engine()
                .find_symbol("alpha_symbol", 5)
                .is_empty()
        );
        assert_eq!(
            reopened
                .workspace
                .engine()
                .find_symbol("beta_symbol", 5)
                .len(),
            1
        );
    }

    #[test]
    fn observer_receives_open_refresh_and_save_events() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("lib.rs");
        fs::write(&source, "pub fn first_symbol() {}").unwrap();
        let observer = RecordingObserver::default();

        let opened =
            WorkspaceIndex::open_with_observer(temp.path(), WorkspaceConfig::default(), &observer)
                .unwrap();
        let mut workspace = opened.workspace;
        let _ = workspace.save_with_observer(&observer).unwrap();
        fs::write(&source, "pub fn second_symbol() {}").unwrap();
        let _ = workspace.refresh_with_observer(&observer).unwrap();

        let events = observer.events.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|event| event.kind == LifecycleEventKind::Open)
        );
        assert!(
            events
                .iter()
                .any(|event| event.kind == LifecycleEventKind::Save)
        );
        assert!(
            events
                .iter()
                .any(|event| event.kind == LifecycleEventKind::Refresh)
        );
    }

    #[test]
    fn manifest_path_uses_frame_index_dir() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("lib.rs"), "pub fn first_symbol() {}").unwrap();
        let opened = WorkspaceIndex::open(temp.path(), WorkspaceConfig::default()).unwrap();
        assert!(
            opened
                .workspace
                .manifest_path()
                .ends_with(Path::new(".shunt/index/manifest.json"))
        );
        assert!(
            opened
                .workspace
                .snapshot_path()
                .ends_with(Path::new(".shunt/index/engine.snapshot.json"))
        );
    }

    #[derive(Default)]
    struct RecordingObserver {
        events: Mutex<Vec<LifecycleEvent>>,
    }

    impl LifecycleObserver for RecordingObserver {
        fn record(&self, event: &LifecycleEvent) {
            self.events.lock().unwrap().push(event.clone());
        }
    }
}
