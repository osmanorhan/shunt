use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use ignore::{WalkBuilder, WalkState};
use serde::{Deserialize, Serialize};
use shunt_core::{CandidateFile, UnderstandingArtifact};
mod search;
pub mod workspace_search;

// Re-export the embedded search engine types. Types that conflict with names
// already defined in this module (e.g. SearchHit, FileHit, Evidence) are
// given an "Engine" prefix so both can coexist.
pub use search::{
    ChunkFileRecord, ChunkId, ChunkKind, ChunkNeighbor, ChunkStore, CodeChunk, CodeContext,
    ContextAssembler, ContextChunk, ContextOptions, ContextSource, ContextTarget, CrawlConfig,
    EngineConfig, Evidence as EngineEvidence, FileHit, FileRanker, FusionRanker, GraphDirection,
    GraphEdge, GraphOptions, GraphStepDirection, HitSource, HybridEngine, IngestConfig,
    IngestStats, LexicalIndex, LifecycleEvent, LifecycleEventKind, LifecycleObserver, NeighborKind,
    NoopLifecycleObserver, RefreshStats, RelationshipIndex, SearchBackend, SearchError,
    SearchHit as EngineSearchHit, SearchOptions, SearchProfile, SymbolIndex, WorkspaceConfig,
    WorkspaceFallbackReason, WorkspaceIndex, WorkspaceIngestor, WorkspaceOpenMode,
    WorkspaceOpenReport, WorkspaceOpenResult, WorkspaceRefreshReport, WorkspaceSaveReport,
};
use thiserror::Error;
use tree_sitter::{Node, Parser, Tree};
pub use workspace_search::{TextHit, WorkspaceSearch};

const MAX_SNIPPETS_PER_FILE: usize = 3;
const MAX_CANDIDATES: usize = 5;
const HIT_CONTEXT_LINES: usize = 2;
const MERGE_GAP_LINES: usize = 3;
const MAX_PREVIEW_CHARS: usize = 220;
const DOC_INTENT_HINTS: &[&str] = &["docs", "documentation", "readme", "manual", "guide"];
const CONFIG_INTENT_HINTS: &[&str] = &[
    "install",
    "setup",
    "set up",
    "initialize",
    "init",
    "bootstrap",
    "create project",
    "scaffold",
    "configure",
    "config",
    "dependency",
    "framework",
];
const CODE_INTENT_HINTS: &[&str] = &["fix", "change", "update", "edit", "implement", "refactor"];
const ROOT_MANIFEST_FILES: &[&str] = &[
    "package.json",
    "cargo.toml",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pipfile",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
    "gemfile",
    "mix.exs",
    "deno.json",
    "deno.jsonc",
    "bunfig.toml",
];
const ROOT_LOCK_FILES: &[&str] = &[
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lockb",
    "cargo.lock",
    "poetry.lock",
    "pipfile.lock",
    "composer.lock",
    "gemfile.lock",
];
const SOURCE_ROOT_DIRS: &[&str] = &["src", "app", "lib", "cmd", "pkg", "internal"];
const TEST_ROOT_DIRS: &[&str] = &["test", "tests", "spec"];
const SETUP_ENTRYPOINT_NAMES: &[&str] = &["main", "index", "root", "app"];
const REPO_SETUP_TERMS: &[&str] = &[
    "config",
    "configuration",
    "dependency",
    "dependencies",
    "package",
    "packages",
    "requirement",
    "requirements",
    "script",
    "scripts",
    "workspace",
    "module",
    "build",
    "test",
    "env",
];

#[derive(Debug, Error)]
pub enum LocalizeError {
    #[error("workspace root is empty")]
    EmptyWorkspaceRoot,
    #[error("search query has no usable terms")]
    EmptyQuery,
    #[error("search backend error: {0}")]
    SearchBackend(String),
}

pub type LocalizeResult<T> = Result<T, LocalizeError>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchIntent {
    Code,
    Config,
    Docs,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RetrievalBackend {
    Lexical,
    Semantic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchQuery {
    pub intent: SearchIntent,
    pub literals: Vec<String>,
    #[serde(default)]
    pub repo_terms: Vec<String>,
    pub regexes: Vec<String>,
    pub symbol_guesses: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct WorkspaceProfile {
    root_manifests: Vec<String>,
    root_lockfiles: Vec<String>,
    root_configs: Vec<String>,
    source_roots: Vec<String>,
    test_roots: Vec<String>,
    searchable_terms: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchHit {
    pub path: String,
    pub line_number: usize,
    pub line: String,
    pub matched_term: String,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateSnippet {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub enclosing_symbol: Option<String>,
    pub reason: String,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateRole {
    Implementation,
    Callsite,
    Config,
    Test,
    Docs,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub file: CandidateFile,
    pub role: CandidateRole,
    pub score: f32,
    pub reasons: Vec<String>,
    pub snippets: Vec<CandidateSnippet>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextPacket {
    pub backend: RetrievalBackend,
    pub query: SearchQuery,
    pub primary_candidates: Vec<RankedCandidate>,
    pub supporting_candidates: Vec<RankedCandidate>,
}

impl ContextPacket {
    pub fn all_candidates(&self) -> impl Iterator<Item = &RankedCandidate> {
        self.primary_candidates
            .iter()
            .chain(self.supporting_candidates.iter())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RetrievedFile {
    pub path: String,
    pub contents: String,
    pub path_terms: Vec<String>,
    pub hits: Vec<SearchHit>,
    pub retrieval_reasons: Vec<String>,
    pub retrieval_score: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredFile {
    pub path: String,
    pub contents: String,
    pub path_terms: Vec<String>,
    pub hits: Vec<SearchHit>,
    pub snippets: Vec<CandidateSnippet>,
    pub identifier_hits: Vec<String>,
    pub role: CandidateRole,
    pub retrieval_reasons: Vec<String>,
    pub retrieval_score: Option<f32>,
}

pub trait QueryPlanner {
    fn build_query(
        &self,
        workspace_root: &Path,
        artifact: &UnderstandingArtifact,
    ) -> LocalizeResult<SearchQuery>;
}

pub trait Retriever {
    fn retrieve(
        &self,
        workspace_root: &Path,
        query: &SearchQuery,
    ) -> LocalizeResult<Vec<RetrievedFile>>;

    fn backend(&self) -> RetrievalBackend;
}

pub trait StructureExtractor {
    fn extract(
        &self,
        query: &SearchQuery,
        files: Vec<RetrievedFile>,
    ) -> LocalizeResult<Vec<StructuredFile>>;
}

pub trait CandidateRanker {
    fn rank(
        &self,
        query: &SearchQuery,
        files: Vec<StructuredFile>,
    ) -> LocalizeResult<Vec<RankedCandidate>>;
}

pub trait ContextPacker {
    fn pack(
        &self,
        query: SearchQuery,
        candidates: Vec<RankedCandidate>,
    ) -> LocalizeResult<ContextPacket>;
}

pub trait Localizer {
    fn localize(
        &self,
        workspace_root: &str,
        artifact: &UnderstandingArtifact,
    ) -> LocalizeResult<ContextPacket>;
}

pub struct ArtifactQueryPlanner;
pub struct LexicalRetriever;
#[derive(Clone, Default)]
pub struct SemanticRetriever {
    engines: Arc<Mutex<BTreeMap<PathBuf, HybridEngine>>>,
    warming: Arc<Mutex<BTreeSet<PathBuf>>>,
}
pub struct TreeSitterStructureExtractor;
pub struct TfIdfRanker;
pub struct SemanticCandidateRanker;
pub struct DefaultContextPacker;

pub struct PipelineLocalizer<P, R, S, K, C> {
    planner: P,
    retriever: R,
    extractor: S,
    ranker: K,
    packer: C,
}

pub type DefaultLocalizer = PipelineLocalizer<
    ArtifactQueryPlanner,
    LexicalRetriever,
    TreeSitterStructureExtractor,
    TfIdfRanker,
    DefaultContextPacker,
>;

type SemanticPipeline = PipelineLocalizer<
    ArtifactQueryPlanner,
    SemanticRetriever,
    TreeSitterStructureExtractor,
    SemanticCandidateRanker,
    DefaultContextPacker,
>;

pub struct SemanticLocalizer {
    index: SemanticPipeline,
    fallback: DefaultLocalizer,
}

impl Default for DefaultLocalizer {
    fn default() -> Self {
        Self {
            planner: ArtifactQueryPlanner,
            retriever: LexicalRetriever,
            extractor: TreeSitterStructureExtractor,
            ranker: TfIdfRanker,
            packer: DefaultContextPacker,
        }
    }
}

impl Default for SemanticLocalizer {
    fn default() -> Self {
        Self {
            index: PipelineLocalizer::new(
                ArtifactQueryPlanner,
                SemanticRetriever::default(),
                TreeSitterStructureExtractor,
                SemanticCandidateRanker,
                DefaultContextPacker,
            ),
            fallback: DefaultLocalizer::default(),
        }
    }
}

impl<P, R, S, K, C> PipelineLocalizer<P, R, S, K, C> {
    pub fn new(planner: P, retriever: R, extractor: S, ranker: K, packer: C) -> Self {
        Self {
            planner,
            retriever,
            extractor,
            ranker,
            packer,
        }
    }
}

impl QueryPlanner for ArtifactQueryPlanner {
    fn build_query(
        &self,
        workspace_root: &Path,
        artifact: &UnderstandingArtifact,
    ) -> LocalizeResult<SearchQuery> {
        let request = artifact.original_request.trim();
        let literals = tokenize_query(request);

        if literals.is_empty() {
            return Err(LocalizeError::EmptyQuery);
        }

        let intent = infer_search_intent(request);
        let profile = detect_workspace_profile(workspace_root);
        let repo_terms = repo_search_terms(&intent, &literals, &profile);
        let symbol_guesses = setup_symbol_guesses(&intent, &profile);

        if repo_terms.is_empty() && symbol_guesses.is_empty() {
            return Err(LocalizeError::EmptyQuery);
        }

        Ok(SearchQuery {
            intent,
            literals,
            repo_terms,
            regexes: vec![],
            symbol_guesses,
        })
    }
}

impl Retriever for LexicalRetriever {
    fn retrieve(
        &self,
        workspace_root: &Path,
        query: &SearchQuery,
    ) -> LocalizeResult<Vec<RetrievedFile>> {
        let matcher = build_matcher(query)?;
        let root = workspace_root
            .canonicalize()
            .unwrap_or_else(|_| workspace_root.to_path_buf());
        let files = Arc::new(Mutex::new(Vec::<RetrievedFile>::new()));
        let query = Arc::new(query.clone());
        let matcher = Arc::new(matcher);

        let mut builder = WalkBuilder::new(&root);
        builder.standard_filters(true);
        builder.git_ignore(true);
        builder.git_exclude(true);
        builder.hidden(false);
        builder.build_parallel().run(|| {
            let files = Arc::clone(&files);
            let query = Arc::clone(&query);
            let matcher = Arc::clone(&matcher);
            let root = root.clone();

            Box::new(move |entry| {
                let Ok(entry) = entry else {
                    return WalkState::Continue;
                };
                let path = entry.path();
                if !entry
                    .file_type()
                    .map(|file_type| file_type.is_file())
                    .unwrap_or(false)
                    || is_excluded_path(path)
                    || !path_allowed(path)
                {
                    return WalkState::Continue;
                }

                let Ok(contents) = fs::read_to_string(path) else {
                    return WalkState::Continue;
                };
                if contents.trim().is_empty() {
                    return WalkState::Continue;
                }

                let Some(file) =
                    retrieve_file(&root, path, &contents, &query, matcher.as_ref().as_ref())
                else {
                    return WalkState::Continue;
                };

                if let Ok(mut files) = files.lock() {
                    files.push(file);
                }
                WalkState::Continue
            })
        });

        Ok(files.lock().map(|guard| guard.clone()).unwrap_or_default())
    }

    fn backend(&self) -> RetrievalBackend {
        RetrievalBackend::Lexical
    }
}

impl Retriever for SemanticRetriever {
    fn retrieve(
        &self,
        workspace_root: &Path,
        query: &SearchQuery,
    ) -> LocalizeResult<Vec<RetrievedFile>> {
        let root = workspace_root
            .canonicalize()
            .unwrap_or_else(|_| workspace_root.to_path_buf());
        let engines = self
            .engines
            .lock()
            .map_err(|_| LocalizeError::SearchBackend("search index lock poisoned".into()))?;

        let Some(engine) = engines.get(&root) else {
            drop(engines);
            self.warm_path_async(root);
            return Ok(Vec::new());
        };
        let query_text = query
            .repo_terms
            .iter()
            .chain(query.symbol_guesses.iter())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        if query_text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let search_limit = MAX_CANDIDATES * 8;
        let result_sets = [
            engine
                .lexical_index()
                .search_path(&query_text, search_limit),
            engine.lexical_index().search(&query_text, search_limit),
            engine.symbol_index().search(&query_text, search_limit),
        ];
        let file_hits = engine.file_ranker().rank(&result_sets, MAX_CANDIDATES * 4);
        let mut files = Vec::new();

        for file_hit in file_hits {
            let absolute_path = PathBuf::from(&file_hit.file_path);
            if is_excluded_path(&absolute_path) || !path_allowed(&absolute_path) {
                continue;
            }
            let Ok(contents) = fs::read_to_string(&absolute_path) else {
                continue;
            };
            let path = relative_path(&root, &absolute_path);
            let lower_path = path.to_ascii_lowercase();
            let path_terms = query_path_terms(query)
                .iter()
                .filter(|term| lower_path.contains(term.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            let mut hits = file_hit
                .anchor_chunks
                .iter()
                .filter_map(|chunk_id| engine.chunk(*chunk_id))
                .map(|chunk| SearchHit {
                    path: path.clone(),
                    line_number: chunk.start_line.max(1) as usize,
                    line: chunk.content.lines().next().unwrap_or_default().to_string(),
                    matched_term: file_hit
                        .matched_terms
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "semantic".into()),
                    context_before: Vec::new(),
                    context_after: Vec::new(),
                })
                .collect::<Vec<_>>();
            if hits.is_empty() {
                hits.push(SearchHit {
                    path: path.clone(),
                    line_number: 1,
                    line: contents.lines().next().unwrap_or_default().to_string(),
                    matched_term: file_hit
                        .matched_terms
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "semantic".into()),
                    context_before: Vec::new(),
                    context_after: Vec::new(),
                });
            }

            let mut retrieval_reasons = file_hit
                .evidence
                .iter()
                .map(|evidence| evidence.summary())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .take(12)
                .collect::<Vec<_>>();
            retrieval_reasons.insert(0, format!("search score: {:.6}", file_hit.score));
            files.push(RetrievedFile {
                path,
                contents,
                path_terms,
                hits,
                retrieval_reasons,
                retrieval_score: Some(file_hit.score),
            });
        }

        Ok(files)
    }

    fn backend(&self) -> RetrievalBackend {
        RetrievalBackend::Semantic
    }
}

impl SemanticRetriever {
    fn warm_path_async(&self, workspace_root: PathBuf) {
        let mut warming = match self.warming.lock() {
            Ok(warming) => warming,
            Err(_) => return,
        };
        if warming.contains(&workspace_root) {
            return;
        }
        warming.insert(workspace_root.clone());
        drop(warming);

        let engines = Arc::clone(&self.engines);
        let warming = Arc::clone(&self.warming);
        thread::spawn(move || {
            if let Ok((engine, _)) =
                HybridEngine::index_path(&workspace_root, IngestConfig::default())
                && let Ok(mut engines) = engines.lock()
            {
                engines.insert(workspace_root.clone(), engine);
            }
            if let Ok(mut warming) = warming.lock() {
                warming.remove(&workspace_root);
            }
        });
    }

    pub(crate) fn warm_path_blocking(&self, workspace_root: &Path) -> LocalizeResult<()> {
        let root = workspace_root
            .canonicalize()
            .unwrap_or_else(|_| workspace_root.to_path_buf());
        let (engine, _) = HybridEngine::index_path(&root, IngestConfig::default())
            .map_err(|error| LocalizeError::SearchBackend(error.to_string()))?;
        let mut engines = self
            .engines
            .lock()
            .map_err(|_| LocalizeError::SearchBackend("search index lock poisoned".into()))?;
        engines.insert(root, engine);
        Ok(())
    }
}

impl StructureExtractor for TreeSitterStructureExtractor {
    fn extract(
        &self,
        query: &SearchQuery,
        files: Vec<RetrievedFile>,
    ) -> LocalizeResult<Vec<StructuredFile>> {
        let mut structured = Vec::new();

        for file in files {
            let (snippets, identifier_hits) = match detect_language(&file.path) {
                SourceLanguage::Rust => extract_rust_structure(&file, query)
                    .unwrap_or_else(|| (generic_snippets(&file), Vec::new())),
                SourceLanguage::Unknown => (generic_snippets(&file), Vec::new()),
            };
            let role = file_role(&file.path, &snippets, &file.hits);

            structured.push(StructuredFile {
                path: file.path,
                contents: file.contents,
                path_terms: file.path_terms,
                hits: file.hits,
                snippets,
                identifier_hits,
                role,
                retrieval_reasons: file.retrieval_reasons,
                retrieval_score: file.retrieval_score,
            });
        }

        Ok(structured)
    }
}

impl CandidateRanker for TfIdfRanker {
    fn rank(
        &self,
        query: &SearchQuery,
        files: Vec<StructuredFile>,
    ) -> LocalizeResult<Vec<RankedCandidate>> {
        let total_docs = files.len().max(1) as f32;
        let ranking_terms = query_ranking_terms(query);
        let document_frequency = document_frequency(&files, ranking_terms);
        let mut candidates = Vec::new();

        for file in files {
            let hit_counts = hit_counts(&file.hits);
            let path_terms = file.path_terms.iter().cloned().collect::<BTreeSet<_>>();
            let matched_terms = hit_counts
                .keys()
                .cloned()
                .chain(path_terms.iter().cloned())
                .collect::<BTreeSet<_>>();

            let mut score = 0.0;
            for term in &matched_terms {
                let df = *document_frequency.get(term).unwrap_or(&1) as f32;
                let idf = ((total_docs + 1.0) / (df + 1.0)).ln() + 1.0;
                let content_tf = *hit_counts.get(term).unwrap_or(&0) as f32;
                let path_tf = if path_terms.contains(term) { 1.0 } else { 0.0 };
                score += (content_tf * 2.0 + path_tf * 1.5) * idf;
            }
            score += snippet_term_coverage(&file.snippets, ranking_terms);
            score += identifier_hit_score(&file.identifier_hits, &document_frequency, total_docs);
            let cfg = RankingConfig::default();
            score += cfg.file_family_score(&file.path);
            score += cfg.intent_path_score(query, &file.path, file.role);

            let mut reasons = file.retrieval_reasons;
            if !matched_terms.is_empty() {
                reasons.push(format!(
                    "matched terms: {}",
                    matched_terms.iter().cloned().collect::<Vec<_>>().join(", ")
                ));
            }
            if !path_terms.is_empty() {
                reasons.push(format!(
                    "path hits: {}",
                    path_terms.into_iter().collect::<Vec<_>>().join(", ")
                ));
            }
            if !file.identifier_hits.is_empty() {
                reasons.push(format!(
                    "identifier hits: {}",
                    file.identifier_hits.join(", ")
                ));
            }
            reasons.push(format!("file family: {}", file_family(&file.path)));
            reasons.push(format!(
                "coverage: {}/{} query terms",
                matched_terms.len(),
                ranking_terms.len().max(1)
            ));

            let preview = file
                .snippets
                .first()
                .map(|snippet| clamp_preview(&snippet.text))
                .unwrap_or_else(|| clamp_preview(&file.contents));

            candidates.push(RankedCandidate {
                file: CandidateFile {
                    path: file.path,
                    summary: format!(
                        "score={score:.1}; reasons={}; snippet: {preview}",
                        reasons.join("; ")
                    ),
                },
                role: file.role,
                score,
                reasons,
                snippets: file.snippets,
            });
        }

        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.file.path.cmp(&right.file.path))
        });

        Ok(candidates)
    }
}

impl CandidateRanker for SemanticCandidateRanker {
    fn rank(
        &self,
        query: &SearchQuery,
        files: Vec<StructuredFile>,
    ) -> LocalizeResult<Vec<RankedCandidate>> {
        let mut candidates = files
            .into_iter()
            .map(|file| {
                let score = file.retrieval_score.unwrap_or_default()
                    + RankingConfig::default().intent_path_score(query, &file.path, file.role);
                let reasons = file.retrieval_reasons;
                let preview = file
                    .snippets
                    .first()
                    .map(|snippet| clamp_preview(&snippet.text))
                    .unwrap_or_else(|| clamp_preview(&file.contents));
                RankedCandidate {
                    file: CandidateFile {
                        path: file.path,
                        summary: format!(
                            "score={score:.6}; reasons={}; snippet: {preview}",
                            reasons.join("; ")
                        ),
                    },
                    role: file.role,
                    score,
                    reasons,
                    snippets: file.snippets,
                }
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.file.path.cmp(&right.file.path))
        });
        Ok(candidates)
    }
}

impl ContextPacker for DefaultContextPacker {
    fn pack(
        &self,
        query: SearchQuery,
        candidates: Vec<RankedCandidate>,
    ) -> LocalizeResult<ContextPacket> {
        let mut primary = Vec::new();
        let mut supporting = Vec::new();

        for candidate in candidates {
            if is_primary_role(candidate.role) {
                primary.push(candidate);
            } else {
                supporting.push(candidate);
            }
        }

        if primary.is_empty() && !supporting.is_empty() {
            primary = supporting
                .drain(..supporting.len().min(MAX_CANDIDATES))
                .collect();
        } else {
            primary.truncate(MAX_CANDIDATES);
            let remaining = MAX_CANDIDATES.saturating_sub(primary.len());
            supporting.truncate(remaining);
        }

        Ok(ContextPacket {
            backend: RetrievalBackend::Lexical,
            query,
            primary_candidates: primary,
            supporting_candidates: supporting,
        })
    }
}

impl<P, R, S, K, C> Localizer for PipelineLocalizer<P, R, S, K, C>
where
    P: QueryPlanner,
    R: Retriever,
    S: StructureExtractor,
    K: CandidateRanker,
    C: ContextPacker,
{
    fn localize(
        &self,
        workspace_root: &str,
        artifact: &UnderstandingArtifact,
    ) -> LocalizeResult<ContextPacket> {
        if workspace_root.trim().is_empty() {
            return Err(LocalizeError::EmptyWorkspaceRoot);
        }

        let root = PathBuf::from(workspace_root);
        let query = self.planner.build_query(&root, artifact)?;
        let retrieved = self.retriever.retrieve(&root, &query)?;
        let structured = self.extractor.extract(&query, retrieved)?;
        let ranked = self.ranker.rank(&query, structured)?;
        let mut packet = self.packer.pack(query, ranked)?;
        packet.backend = self.retriever.backend();
        Ok(packet)
    }
}

impl SemanticLocalizer {
    /// Pre-warm the search index for a workspace synchronously.
    /// Call this before the first localize to avoid cold-start empty results.
    /// In production the index warms asynchronously; this forces it to complete.
    pub fn prewarm(&self, workspace_root: &Path) -> LocalizeResult<()> {
        self.index.retriever.warm_path_blocking(workspace_root)
    }
}

impl Localizer for SemanticLocalizer {
    fn localize(
        &self,
        workspace_root: &str,
        artifact: &UnderstandingArtifact,
    ) -> LocalizeResult<ContextPacket> {
        let semantic_packet = self.index.localize(workspace_root, artifact).ok();
        let fallback_packet = self.fallback.localize(workspace_root, artifact).ok();

        match (semantic_packet, fallback_packet) {
            (Some(semantic), Some(fallback)) => {
                if should_prefer_fallback_packet(&semantic, &fallback) {
                    Ok(fallback)
                } else if semantic.all_candidates().next().is_some() {
                    Ok(semantic)
                } else {
                    Ok(fallback)
                }
            }
            (Some(packet), None) if packet.all_candidates().next().is_some() => Ok(packet),
            (None, Some(fallback)) => Ok(fallback),
            (Some(packet), None) => Ok(packet),
            (None, None) => self.fallback.localize(workspace_root, artifact),
        }
    }
}

fn should_prefer_fallback_packet(semantic: &ContextPacket, fallback: &ContextPacket) -> bool {
    if !matches!(semantic.query.intent, SearchIntent::Config) {
        return false;
    }

    let semantic_has_setup = semantic
        .all_candidates()
        .any(|candidate| is_setup_preferred_path(&candidate.file.path));
    let fallback_has_setup = fallback
        .all_candidates()
        .any(|candidate| is_setup_preferred_path(&candidate.file.path));

    fallback_has_setup && !semantic_has_setup
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceLanguage {
    Rust,
    Unknown,
}

fn detect_language(path: &str) -> SourceLanguage {
    match path.rsplit('.').next() {
        Some("rs") => SourceLanguage::Rust,
        _ => SourceLanguage::Unknown,
    }
}

fn is_primary_role(role: CandidateRole) -> bool {
    matches!(
        role,
        CandidateRole::Implementation | CandidateRole::Callsite | CandidateRole::Config
    )
}

fn build_matcher(query: &SearchQuery) -> LocalizeResult<Option<AhoCorasick>> {
    if query.repo_terms.is_empty() {
        return Ok(None);
    }

    AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .build(&query.repo_terms)
        .map(Some)
        .map_err(|_| LocalizeError::EmptyQuery)
}

fn retrieve_file(
    workspace_root: &Path,
    path: &Path,
    contents: &str,
    query: &SearchQuery,
    matcher: Option<&AhoCorasick>,
) -> Option<RetrievedFile> {
    let relative_path = relative_path(workspace_root, path);
    let lower_path = relative_path.to_ascii_lowercase();
    let path_terms = query_path_terms(query)
        .iter()
        .filter(|term| lower_path.contains(term.as_str()))
        .cloned()
        .collect::<Vec<_>>();

    let lines = contents.lines().collect::<Vec<_>>();
    let mut hits = Vec::new();
    if let Some(matcher) = matcher {
        for (index, line) in lines.iter().enumerate() {
            for found in matcher.find_iter(line) {
                let matched_term = query.repo_terms[found.pattern().as_usize()].clone();
                let start = index.saturating_sub(HIT_CONTEXT_LINES);
                let end = (index + HIT_CONTEXT_LINES + 1).min(lines.len());
                hits.push(SearchHit {
                    path: relative_path.clone(),
                    line_number: index + 1,
                    line: (*line).to_string(),
                    matched_term,
                    context_before: lines[start..index]
                        .iter()
                        .map(|line| (*line).to_string())
                        .collect(),
                    context_after: lines[index + 1..end]
                        .iter()
                        .map(|line| (*line).to_string())
                        .collect(),
                });
            }
        }
    }

    if hits.is_empty() && path_terms.is_empty() {
        return None;
    }

    Some(RetrievedFile {
        path: relative_path,
        contents: contents.to_string(),
        path_terms,
        hits,
        retrieval_reasons: Vec::new(),
        retrieval_score: None,
    })
}

fn generic_snippets(file: &RetrievedFile) -> Vec<CandidateSnippet> {
    let lines = file.contents.lines().collect::<Vec<_>>();
    let mut hit_lines = file
        .hits
        .iter()
        .map(|hit| hit.line_number.saturating_sub(1))
        .collect::<Vec<_>>();
    hit_lines.sort_unstable();
    hit_lines.dedup();

    cluster_hit_lines(&hit_lines)
        .into_iter()
        .take(MAX_SNIPPETS_PER_FILE)
        .map(|(start, end)| snippet_from_cluster(&file.path, &lines, start, end, &file.hits))
        .collect()
}

fn file_role(path: &str, snippets: &[CandidateSnippet], hits: &[SearchHit]) -> CandidateRole {
    match file_family(path) {
        "docs" => CandidateRole::Docs,
        "config" => CandidateRole::Config,
        _ if is_test_path(path) => CandidateRole::Test,
        _ => infer_code_role(snippets, hits),
    }
}

fn infer_code_role(snippets: &[CandidateSnippet], _hits: &[SearchHit]) -> CandidateRole {
    if snippets.iter().any(|snippet| {
        snippet
            .enclosing_symbol
            .as_deref()
            .map(|symbol| {
                symbol.starts_with("call_expression ")
                    || symbol.starts_with("method_call_expression ")
            })
            .unwrap_or(false)
    }) {
        CandidateRole::Callsite
    } else if snippets
        .iter()
        .any(|snippet| snippet.enclosing_symbol.is_some())
    {
        CandidateRole::Implementation
    } else {
        CandidateRole::Unknown
    }
}

fn extract_rust_structure(
    file: &RetrievedFile,
    query: &SearchQuery,
) -> Option<(Vec<CandidateSnippet>, Vec<String>)> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .ok()?;
    let tree = parser.parse(&file.contents, None)?;
    let line_starts = line_start_offsets(&file.contents);
    let lines = file.contents.lines().collect::<Vec<_>>();
    let mut hit_lines = file
        .hits
        .iter()
        .map(|hit| hit.line_number.saturating_sub(1))
        .collect::<Vec<_>>();
    hit_lines.sort_unstable();
    hit_lines.dedup();

    let snippets = cluster_hit_lines(&hit_lines)
        .into_iter()
        .take(MAX_SNIPPETS_PER_FILE)
        .map(|(start, end)| {
            rust_snippet_from_cluster(
                &file.path,
                &file.contents,
                &lines,
                &tree,
                &line_starts,
                start,
                end,
                &file.hits,
            )
        })
        .collect::<Vec<_>>();
    let identifier_hits = rust_identifier_hits(&tree, &file.contents, query);

    Some((snippets, identifier_hits))
}

fn cluster_hit_lines(hit_lines: &[usize]) -> Vec<(usize, usize)> {
    let Some(mut start) = hit_lines.first().copied() else {
        return Vec::new();
    };
    let mut end = start;
    let mut clusters = Vec::new();

    for line in hit_lines.iter().copied().skip(1) {
        if line <= end + MERGE_GAP_LINES {
            end = line;
        } else {
            clusters.push((start, end));
            start = line;
            end = line;
        }
    }
    clusters.push((start, end));
    clusters
}

#[allow(clippy::too_many_arguments)]
fn rust_snippet_from_cluster(
    path: &str,
    contents: &str,
    lines: &[&str],
    tree: &Tree,
    line_starts: &[usize],
    start: usize,
    end: usize,
    hits: &[SearchHit],
) -> CandidateSnippet {
    let start_byte = *line_starts.get(start).unwrap_or(&0);
    let end_line = end.min(line_starts.len().saturating_sub(1));
    let end_byte = if end_line + 1 < line_starts.len() {
        line_starts[end_line + 1]
    } else {
        contents.len()
    };

    let node = tree
        .root_node()
        .descendant_for_byte_range(start_byte, end_byte)
        .and_then(significant_rust_ancestor);

    if let Some(node) = node {
        return snippet_from_node(path, contents, lines, node, hits);
    }

    snippet_from_cluster(path, lines, start, end, hits)
}

fn snippet_from_cluster(
    path: &str,
    lines: &[&str],
    start: usize,
    end: usize,
    hits: &[SearchHit],
) -> CandidateSnippet {
    let expanded_start = start.saturating_sub(HIT_CONTEXT_LINES);
    let expanded_end = (end + HIT_CONTEXT_LINES + 1).min(lines.len());
    let matched_terms = hits
        .iter()
        .filter(|hit| {
            let line = hit.line_number.saturating_sub(1);
            line >= start && line <= end
        })
        .map(|hit| hit.matched_term.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    CandidateSnippet {
        path: path.into(),
        start_line: expanded_start + 1,
        end_line: expanded_end,
        enclosing_symbol: None,
        reason: format!(
            "clustered terms {} on lines {}-{}",
            matched_terms.join(", "),
            start + 1,
            end + 1
        ),
        text: lines[expanded_start..expanded_end].join("\n"),
    }
}

fn snippet_from_node(
    path: &str,
    contents: &str,
    lines: &[&str],
    node: Node<'_>,
    hits: &[SearchHit],
) -> CandidateSnippet {
    let start_line = node.start_position().row;
    let end_line = node.end_position().row;
    let expanded_start = start_line.saturating_sub(HIT_CONTEXT_LINES);
    let expanded_end = (end_line + HIT_CONTEXT_LINES + 1).min(lines.len());
    let matched_terms = hits
        .iter()
        .filter(|hit| {
            let line = hit.line_number.saturating_sub(1);
            line >= start_line && line <= end_line
        })
        .map(|hit| hit.matched_term.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    CandidateSnippet {
        path: path.into(),
        start_line: expanded_start + 1,
        end_line: expanded_end,
        enclosing_symbol: rust_node_label(node, contents),
        reason: format!(
            "structured terms {} in {}",
            matched_terms.join(", "),
            node.kind()
        ),
        text: lines[expanded_start..expanded_end].join("\n"),
    }
}

fn significant_rust_ancestor(mut node: Node<'_>) -> Option<Node<'_>> {
    loop {
        match node.kind() {
            "call_expression" | "method_call_expression" => return Some(node),
            "function_item" | "impl_item" | "struct_item" | "enum_item" | "trait_item"
            | "mod_item" | "const_item" | "static_item" => return Some(node),
            _ => {
                node = node.parent()?;
            }
        }
    }
}

fn rust_node_label(node: Node<'_>, contents: &str) -> Option<String> {
    let name = node
        .child_by_field_name("name")
        .and_then(|name| name.utf8_text(contents.as_bytes()).ok())
        .map(str::to_string);
    match name {
        Some(name) => Some(format!("{} {}", node.kind(), name)),
        None => Some(node.kind().to_string()),
    }
}

fn rust_identifier_hits(tree: &Tree, contents: &str, query: &SearchQuery) -> Vec<String> {
    let mut hits = BTreeSet::new();
    collect_rust_identifier_hits(tree.root_node(), contents, query, &mut hits);
    hits.into_iter().collect()
}

fn collect_rust_identifier_hits(
    node: Node<'_>,
    contents: &str,
    query: &SearchQuery,
    hits: &mut BTreeSet<String>,
) {
    match node.kind() {
        "identifier" | "type_identifier" | "field_identifier" => {
            if let Ok(text) = node.utf8_text(contents.as_bytes()) {
                let lower = text.to_ascii_lowercase();
                if query
                    .literals
                    .iter()
                    .any(|term| lower.contains(term.as_str()))
                {
                    hits.insert(text.to_string());
                }
            }
        }
        _ => {}
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_rust_identifier_hits(child, contents, query, hits);
    }
}

fn line_start_offsets(contents: &str) -> Vec<usize> {
    let mut offsets = vec![0];
    for (index, ch) in contents.char_indices() {
        if ch == '\n' {
            offsets.push(index + 1);
        }
    }
    offsets
}

fn document_frequency(files: &[StructuredFile], query_terms: &[String]) -> BTreeMap<String, usize> {
    let mut df = BTreeMap::new();

    for term in query_terms {
        let count = files
            .iter()
            .filter(|file| {
                file.path_terms.iter().any(|path_term| path_term == term)
                    || file.hits.iter().any(|hit| &hit.matched_term == term)
            })
            .count();
        df.insert(term.clone(), count.max(1));
    }

    df
}

fn hit_counts(hits: &[SearchHit]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for hit in hits {
        *counts.entry(hit.matched_term.clone()).or_insert(0) += 1;
    }
    counts
}

fn snippet_term_coverage(snippets: &[CandidateSnippet], query_terms: &[String]) -> f32 {
    let covered = snippets
        .iter()
        .flat_map(|snippet| {
            query_terms
                .iter()
                .filter(move |term| snippet.text.to_ascii_lowercase().contains(term.as_str()))
                .cloned()
        })
        .collect::<BTreeSet<_>>();
    covered.len() as f32
}

fn identifier_hit_score(
    identifier_hits: &[String],
    document_frequency: &BTreeMap<String, usize>,
    total_docs: f32,
) -> f32 {
    let mut score = 0.0;
    for identifier in identifier_hits {
        let lower = identifier.to_ascii_lowercase();
        for (term, df) in document_frequency {
            if lower.contains(term.as_str()) {
                let idf = ((total_docs + 1.0) / (*df as f32 + 1.0)).ln() + 1.0;
                score += idf * 1.5;
            }
        }
    }
    score
}

fn tokenize_query(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in input.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            push_query_token(&mut tokens, &current);
            current.clear();
        }
    }
    if !current.is_empty() {
        push_query_token(&mut tokens, &current);
    }

    tokens
}

fn infer_search_intent(request: &str) -> SearchIntent {
    let lower = request.to_ascii_lowercase();

    if DOC_INTENT_HINTS.iter().any(|term| lower.contains(term)) {
        SearchIntent::Docs
    } else if CONFIG_INTENT_HINTS.iter().any(|term| lower.contains(term)) {
        SearchIntent::Config
    } else if CODE_INTENT_HINTS.iter().any(|term| lower.contains(term)) {
        SearchIntent::Code
    } else {
        SearchIntent::Unknown
    }
}

fn detect_workspace_profile(workspace_root: &Path) -> WorkspaceProfile {
    let mut profile = WorkspaceProfile::default();
    let Ok(entries) = fs::read_dir(workspace_root) else {
        return profile;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if name.is_empty() {
            continue;
        }

        if path.is_dir() {
            if is_source_root_dir(&name) {
                push_symbol_guess(&mut profile.source_roots, &name);
            }
            if is_test_root_dir(&name) {
                push_symbol_guess(&mut profile.test_roots, &name);
            }
            collect_searchable_terms(&mut profile.searchable_terms, &name);
            continue;
        }

        if is_root_manifest_file(&name) {
            push_symbol_guess(&mut profile.root_manifests, &name);
        } else if is_root_lock_file(&name) {
            push_symbol_guess(&mut profile.root_lockfiles, &name);
        } else if is_root_config_file_name(&name) {
            push_symbol_guess(&mut profile.root_configs, &name);
        }

        collect_searchable_terms(&mut profile.searchable_terms, &name);
    }

    profile
}

fn repo_search_terms(
    intent: &SearchIntent,
    literals: &[String],
    profile: &WorkspaceProfile,
) -> Vec<String> {
    if !matches!(intent, SearchIntent::Config) {
        return literals.to_vec();
    }

    let mut terms = Vec::new();
    for literal in literals {
        if profile.searchable_terms.contains(literal)
            || REPO_SETUP_TERMS.iter().any(|term| term == literal)
        {
            push_symbol_guess(&mut terms, literal);
        }
    }
    terms
}

fn setup_symbol_guesses(intent: &SearchIntent, profile: &WorkspaceProfile) -> Vec<String> {
    let mut guesses = Vec::new();
    if matches!(intent, SearchIntent::Config) {
        for hint in &profile.root_manifests {
            push_symbol_guess(&mut guesses, hint);
        }
        for hint in &profile.root_lockfiles {
            push_symbol_guess(&mut guesses, hint);
        }
        for hint in &profile.root_configs {
            push_symbol_guess(&mut guesses, hint);
        }
        for hint in &profile.source_roots {
            push_symbol_guess(&mut guesses, hint);
        }
    }

    guesses
}

fn push_symbol_guess(guesses: &mut Vec<String>, value: &str) {
    if !guesses.iter().any(|existing| existing == value) {
        guesses.push(value.to_string());
    }
}

fn collect_searchable_terms(terms: &mut BTreeSet<String>, value: &str) {
    let mut current = String::new();
    for ch in value.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            if current.len() >= 3 {
                terms.insert(current.clone());
            }
            current.clear();
        }
    }
    if current.len() >= 3 {
        terms.insert(current);
    }
}

fn query_ranking_terms(query: &SearchQuery) -> &[String] {
    &query.repo_terms
}

fn query_path_terms(query: &SearchQuery) -> Vec<String> {
    let mut terms = query.repo_terms.clone();
    for guess in &query.symbol_guesses {
        if !terms.iter().any(|existing| existing == guess) {
            terms.push(guess.clone());
        }
    }
    terms
}

fn push_query_token(tokens: &mut Vec<String>, token: &str) {
    if token.chars().all(|ch| ch.is_ascii_digit()) {
        return;
    }
    if token.len() < 3 {
        return;
    }
    if !tokens.iter().any(|existing| existing == token) {
        tokens.push(token.to_string());
    }
}

fn path_allowed(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some(
            "rs" | "py"
                | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "go"
                | "java"
                | "kt"
                | "swift"
                | "c"
                | "h"
                | "cpp"
                | "hpp"
                | "toml"
                | "yaml"
                | "yml"
                | "json"
                | "ini"
                | "env"
                | "md"
                | "mdx"
                | "rst"
                | "txt"
        )
    )
}

fn file_family(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some(
            "rs" | "py" | "ts" | "tsx" | "js" | "jsx" | "go" | "java" | "kt" | "swift" | "c" | "h"
            | "cpp" | "hpp",
        ) => "code",
        Some("toml" | "yaml" | "yml" | "json" | "ini" | "env") => "config",
        Some("md" | "mdx" | "rst" | "txt") => "docs",
        _ => "unknown",
    }
}

/// Declarative scoring weights for candidate ranking.
///
/// All weights are additive scores applied on top of TF-IDF.  Positive values
/// boost a file; negative values penalise it.  The defaults encode the same
/// priorities the old hard-coded heuristics did, but are now explicit and
/// configurable.
#[derive(Debug, Clone)]
pub struct RankingConfig {
    /// Score added for code files (rs, py, ts, …).
    pub code_family: f32,
    /// Score added for config files (toml, yaml, json, …).
    pub config_family: f32,
    /// Score added for doc files (md, rst, txt).  Typically negative.
    pub docs_family: f32,
    /// Boost for root manifest files (Cargo.toml, package.json, …) on Config intent.
    pub root_manifest_boost: f32,
    /// Boost for root lock files (Cargo.lock, package-lock.json, …) on Config intent.
    pub root_lockfile_boost: f32,
    /// Boost for root config files (.env, tsconfig.json, …) on Config intent.
    pub root_config_boost: f32,
    /// Boost for source entrypoints (main.rs, index.ts, …) on Config intent.
    pub entrypoint_boost: f32,
    /// Extra score when the candidate role matches the search intent.
    pub role_match_bonus: f32,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            code_family: 3.0,
            config_family: 1.0,
            docs_family: -1.0,
            root_manifest_boost: 8.0,
            root_lockfile_boost: 7.0,
            root_config_boost: 5.0,
            entrypoint_boost: 3.0,
            role_match_bonus: 2.0,
        }
    }
}

impl RankingConfig {
    fn file_family_score(&self, path: &str) -> f32 {
        match file_family(path) {
            "code" => self.code_family,
            "config" => self.config_family,
            "docs" => self.docs_family,
            _ => 0.0,
        }
    }

    fn intent_path_score(&self, query: &SearchQuery, path: &str, role: CandidateRole) -> f32 {
        let lower = path.to_ascii_lowercase();
        match query.intent {
            SearchIntent::Config => {
                let mut score = 0.0;
                if is_root_manifest_path(&lower) {
                    score += self.root_manifest_boost;
                }
                if is_root_lockfile_path(&lower) {
                    score += self.root_lockfile_boost;
                }
                if is_root_config_path(&lower) {
                    score += self.root_config_boost;
                }
                if is_source_entrypoint_path(&lower) {
                    score += self.entrypoint_boost;
                }
                if matches!(role, CandidateRole::Config) {
                    score += self.role_match_bonus;
                }
                score
            }
            SearchIntent::Docs => {
                if matches!(role, CandidateRole::Docs) {
                    self.role_match_bonus + 1.0
                } else {
                    0.0
                }
            }
            SearchIntent::Code => {
                if matches!(
                    role,
                    CandidateRole::Implementation | CandidateRole::Callsite
                ) {
                    1.5
                } else {
                    0.0
                }
            }
            SearchIntent::Unknown => 0.0,
        }
    }
}

fn is_root_manifest_path(path: &str) -> bool {
    !path.contains('/') && is_root_manifest_file(path)
}

fn is_root_lockfile_path(path: &str) -> bool {
    !path.contains('/') && is_root_lock_file(path)
}

fn is_root_config_path(path: &str) -> bool {
    !path.contains('/') && is_root_config_file_name(path)
}

fn is_setup_preferred_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    is_root_manifest_path(&lower)
        || is_root_lockfile_path(&lower)
        || is_root_config_path(&lower)
        || is_source_entrypoint_path(&lower)
}

fn is_root_manifest_file(path: &str) -> bool {
    ROOT_MANIFEST_FILES
        .iter()
        .any(|candidate| candidate == &path)
}

fn is_root_lock_file(path: &str) -> bool {
    ROOT_LOCK_FILES.iter().any(|candidate| candidate == &path)
}

fn is_root_config_file_name(path: &str) -> bool {
    matches!(path, "tsconfig.json" | "jsconfig.json")
        || path.starts_with(".env")
        || path.contains(".config.")
        || path.ends_with("config.json")
        || path.ends_with("config.toml")
        || path.ends_with("config.yaml")
        || path.ends_with("config.yml")
}

fn is_source_root_dir(path: &str) -> bool {
    SOURCE_ROOT_DIRS.iter().any(|candidate| candidate == &path)
}

fn is_test_root_dir(path: &str) -> bool {
    TEST_ROOT_DIRS.iter().any(|candidate| candidate == &path)
}

fn is_source_entrypoint_path(path: &str) -> bool {
    let Some(file_name) = Path::new(path).file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let stem = file_name.split('.').next().unwrap_or(file_name);
    if !SETUP_ENTRYPOINT_NAMES
        .iter()
        .any(|candidate| candidate == &stem)
    {
        return false;
    }

    if !path.contains('/') {
        return true;
    }

    SOURCE_ROOT_DIRS
        .iter()
        .any(|root| path == format!("{root}/{file_name}") || path.starts_with(&format!("{root}/")))
}

fn is_test_path(path: &str) -> bool {
    path.split('/').any(|segment| segment == "tests")
        || path.ends_with("_test.rs")
        || path.ends_with("_spec.rs")
        || path.ends_with(".test.ts")
        || path.ends_with(".spec.ts")
        || path.ends_with(".test.js")
        || path.ends_with(".spec.js")
}

fn is_excluded_path(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some("target" | ".git" | ".frame" | "node_modules")
        )
    })
}

fn relative_path(workspace_root: &Path, path: &Path) -> String {
    path.strip_prefix(workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn clamp_preview(input: &str) -> String {
    let compact = input.replace('\n', " | ");
    if compact.chars().count() <= MAX_PREVIEW_CHARS {
        compact
    } else {
        let preview = compact.chars().take(MAX_PREVIEW_CHARS).collect::<String>();
        format!("{preview}...")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use shunt_core::{ApprovalState, ArtifactId, TaskId, UnderstandingArtifact};
    use time::macros::datetime;

    use super::{
        ArtifactQueryPlanner, DefaultLocalizer, Localizer, PipelineLocalizer, QueryPlanner,
        RetrievalBackend, SearchHit, SearchIntent, SearchQuery, SemanticLocalizer,
        StructureExtractor, TfIdfRanker, TreeSitterStructureExtractor,
    };

    fn test_artifact(request: &str) -> UnderstandingArtifact {
        UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: TaskId("task-1".into()),
            original_request: request.into(),
            interpreted_goal: request.into(),
            success_criteria: vec![],
            constraints: vec![],
            target_scope: vec![],
            evidence: vec![],
            candidate_files: vec![],
            package_facts: vec![],
            manual_evidence: vec![],
            assumptions: vec![],
            ambiguities: vec![],
            selected_recipe: None,
            risks: vec![],
            confidence: 0.0,
            approval: ApprovalState::draft(),
            revision: 1,
            workspace_profile: shunt_core::WorkspaceProfile::default(),
            created_at: datetime!(2026-05-05 12:00 UTC),
            updated_at: datetime!(2026-05-05 12:00 UTC),
        }
    }

    #[test]
    fn builds_query_from_original_request_not_draft_goal() {
        let root = std::env::temp_dir().join(format!(
            "shunt-localize-query-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn model_timeout() {}").unwrap();

        let artifact = UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: TaskId("task-1".into()),
            original_request: "fix model calling class timeout handling".into(),
            interpreted_goal: "likely in crates/shunt-cli or related crates".into(),
            success_criteria: vec![],
            constraints: vec![],
            target_scope: vec![],
            evidence: vec![],
            candidate_files: vec![],
            package_facts: vec![],
            manual_evidence: vec![],
            assumptions: vec![],
            ambiguities: vec![],
            selected_recipe: None,
            risks: vec![],
            confidence: 0.0,
            approval: ApprovalState::draft(),
            revision: 1,
            workspace_profile: shunt_core::WorkspaceProfile::default(),
            created_at: datetime!(2026-05-05 12:00 UTC),
            updated_at: datetime!(2026-05-05 12:00 UTC),
        };

        let query = ArtifactQueryPlanner.build_query(&root, &artifact).unwrap();

        assert_eq!(query.intent, SearchIntent::Code);
        assert!(query.literals.contains(&"model".to_string()));
        assert!(query.repo_terms.contains(&"model".to_string()));
        assert!(query.literals.contains(&"timeout".to_string()));
        assert!(!query.literals.contains(&"frame".to_string()));
        assert!(!query.literals.contains(&"related".to_string()));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn build_query_uses_workspace_structure_for_install_prompt() {
        let root = std::env::temp_dir().join(format!(
            "shunt-localize-install-query-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("package.json"), "{\n  \"name\": \"demo\"\n}\n").unwrap();

        let artifact = test_artifact("lets install remix project here");

        let query = ArtifactQueryPlanner.build_query(&root, &artifact).unwrap();

        assert_eq!(query.intent, SearchIntent::Config);
        assert!(query.literals.contains(&"install".to_string()));
        assert!(query.literals.contains(&"remix".to_string()));
        assert!(!query.repo_terms.contains(&"remix".to_string()));
        assert!(query.symbol_guesses.contains(&"package.json".to_string()));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn localizer_ranks_code_file_ahead_of_docs_for_edit_request() {
        let root = std::env::temp_dir().join(format!(
            "shunt-localize-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("README.md"), "timeout model calling notes\n").unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn model_timeout() -> bool {\n    true\n}\n",
        )
        .unwrap();

        let artifact = UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: TaskId("task-1".into()),
            original_request: "fix model timeout handling".into(),
            interpreted_goal: "fix timeout handling in model calls".into(),
            success_criteria: vec![],
            constraints: vec![],
            target_scope: vec![],
            evidence: vec![],
            candidate_files: vec![],
            package_facts: vec![],
            manual_evidence: vec![],
            assumptions: vec![],
            ambiguities: vec![],
            selected_recipe: None,
            risks: vec![],
            confidence: 0.0,
            approval: ApprovalState::draft(),
            revision: 1,
            workspace_profile: shunt_core::WorkspaceProfile::default(),
            created_at: datetime!(2026-05-05 12:00 UTC),
            updated_at: datetime!(2026-05-05 12:00 UTC),
        };

        let packet = DefaultLocalizer::default()
            .localize(&root.display().to_string(), &artifact)
            .unwrap();

        assert!(!packet.primary_candidates.is_empty());
        assert_eq!(packet.primary_candidates[0].file.path, "src/lib.rs");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn localizer_prefers_workspace_manifest_for_install_prompt() {
        let root = std::env::temp_dir().join(format!(
            "shunt-localize-setup-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("package.json"),
            "{\n  \"name\": \"demo\",\n  \"scripts\": {\n    \"dev\": \"node index.js\"\n  }\n}\n",
        )
        .unwrap();
        fs::write(root.join("src/index.js"), "console.log('hello');\n").unwrap();
        fs::write(root.join("README.md"), "project notes\n").unwrap();

        let artifact = test_artifact("lets install remix project here");

        let packet = DefaultLocalizer::default()
            .localize(&root.display().to_string(), &artifact)
            .unwrap();

        assert!(!packet.primary_candidates.is_empty());
        assert_eq!(packet.primary_candidates[0].file.path, "package.json");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn localizer_prefers_python_requirements_for_dependency_prompt() {
        let root = std::env::temp_dir().join(format!(
            "shunt-localize-python-setup-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("app")).unwrap();
        fs::write(root.join("requirements.txt"), "flask==3.0.0\n").unwrap();
        fs::write(root.join("app/main.py"), "print('hello')\n").unwrap();

        let artifact = test_artifact("install requests dependency here");

        let packet = DefaultLocalizer::default()
            .localize(&root.display().to_string(), &artifact)
            .unwrap();

        assert!(!packet.primary_candidates.is_empty());
        assert_eq!(packet.primary_candidates[0].file.path, "requirements.txt");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pipeline_constructor_is_usable() {
        let _pipeline: DefaultLocalizer = PipelineLocalizer::new(
            super::ArtifactQueryPlanner,
            super::LexicalRetriever,
            TreeSitterStructureExtractor,
            TfIdfRanker,
            super::DefaultContextPacker,
        );
    }

    #[test]
    fn snippet_clusters_merge_nearby_hits() {
        let lines = vec!["fn call() {", "  timeout();", "  model();", "}"];
        let hits = vec![
            SearchHit {
                path: "src/lib.rs".into(),
                line_number: 2,
                line: "  timeout();".into(),
                matched_term: "timeout".into(),
                context_before: vec![],
                context_after: vec![],
            },
            SearchHit {
                path: "src/lib.rs".into(),
                line_number: 3,
                line: "  model();".into(),
                matched_term: "model".into(),
                context_before: vec![],
                context_after: vec![],
            },
        ];
        let cluster = super::snippet_from_cluster("src/lib.rs", &lines, 1, 2, &hits);

        assert!(cluster.text.contains("timeout();"));
        assert!(cluster.text.contains("model();"));
        assert_eq!(cluster.enclosing_symbol, None);
    }

    #[test]
    fn search_query_shape_is_stable() {
        let query = SearchQuery {
            intent: SearchIntent::Unknown,
            literals: vec!["timeout".into()],
            repo_terms: vec!["timeout".into()],
            regexes: vec![],
            symbol_guesses: vec![],
        };
        assert_eq!(query.literals.len(), 1);
    }

    #[test]
    fn rust_structure_extractor_finds_enclosing_function() {
        let file = super::RetrievedFile {
            path: "src/lib.rs".into(),
            contents: "pub fn model_timeout() {\n    timeout();\n}\n".into(),
            path_terms: vec![],
            hits: vec![SearchHit {
                path: "src/lib.rs".into(),
                line_number: 2,
                line: "    timeout();".into(),
                matched_term: "timeout".into(),
                context_before: vec!["pub fn model_timeout() {".into()],
                context_after: vec!["}".into()],
            }],
            retrieval_reasons: Vec::new(),
            retrieval_score: None,
        };
        let structured = TreeSitterStructureExtractor
            .extract(
                &SearchQuery {
                    intent: SearchIntent::Unknown,
                    literals: vec!["timeout".into(), "model".into()],
                    repo_terms: vec!["timeout".into(), "model".into()],
                    regexes: vec![],
                    symbol_guesses: vec![],
                },
                vec![file],
            )
            .unwrap();

        assert_eq!(structured.len(), 1);
        assert_eq!(
            structured[0].snippets[0].enclosing_symbol.as_deref(),
            Some("function_item model_timeout")
        );
        assert!(
            structured[0]
                .identifier_hits
                .contains(&"model_timeout".to_string())
        );
    }

    #[test]
    fn semantic_localizer_maps_evidence_into_candidate_reasons() {
        let root = std::env::temp_dir().join(format!(
            "shunt-localize-semantic-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn refresh_workspace_index() {\n    rebuild_search_cache();\n}\n",
        )
        .unwrap();

        let mut artifact = test_artifact("refresh workspace search index");
        artifact.interpreted_goal = "refresh the local search index".into();
        let localizer = SemanticLocalizer::default();
        let first_packet = localizer
            .localize(&root.display().to_string(), &artifact)
            .unwrap();

        assert_eq!(first_packet.backend, RetrievalBackend::Lexical);
        assert_eq!(first_packet.primary_candidates[0].file.path, "src/lib.rs");

        localizer.index.retriever.warm_path_blocking(&root).unwrap();
        let packet = localizer
            .localize(&root.display().to_string(), &artifact)
            .unwrap();

        assert_eq!(packet.primary_candidates[0].file.path, "src/lib.rs");
        assert!(
            packet.primary_candidates[0]
                .reasons
                .iter()
                .any(|reason| reason.starts_with("search score:"))
        );
        fs::remove_dir_all(root).unwrap();
    }
}
