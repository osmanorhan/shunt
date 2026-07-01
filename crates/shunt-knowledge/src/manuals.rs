use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use reqwest::Url;
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use semver::{Version, VersionReq};
use serde::Deserialize;
use shunt_core::{ManualEvidence, ManualVersionStatus, PackageFact};
use time::OffsetDateTime;

use crate::config::{KnowledgeConfig, KnowledgeSourceKind};
use crate::{
    KnowledgeEvidence, KnowledgeFetchStatus, KnowledgeQuery, KnowledgeResearchRequest,
    KnowledgeResult, KnowledgeSourceRef,
};

const DEFAULT_CATALOG_PATH: &str = ".shunt/manuals/catalog.json";
const MAX_EXCERPT_CHARS: usize = 420;

pub(crate) trait KnowledgeSource: Send + Sync {
    fn kind(&self) -> KnowledgeSourceKind;

    fn search(
        &self,
        workspace_root: &Path,
        query: &KnowledgeQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<KnowledgeEvidence>>;
}

pub(crate) struct KnowledgeSourceRegistry {
    sources: Vec<Box<dyn KnowledgeSource>>,
}

struct FileCatalogSource {
    catalog_path: PathBuf,
}

struct RegistryMetadataSource {
    client: Box<dyn RegistryMetadataClient>,
}

struct DeepWikiSource {
    backend: Box<dyn SearchBackend>,
}

struct PublicSearchSource {
    backend: Box<dyn SearchBackend>,
}

struct DocsRsSource {
    client: Client,
}

struct RepositoryReadmeSource {
    metadata: Box<dyn RegistryMetadataClient>,
    client: Client,
}

trait RegistryMetadataClient: Send + Sync {
    fn npm_metadata(&self, package: &str) -> KnowledgeResult<Option<NpmPackageMetadata>>;
    fn crate_metadata(&self, package: &str) -> KnowledgeResult<Option<CratePackageMetadata>>;
}

struct HttpRegistryMetadataClient {
    client: Client,
}

trait SearchBackend: Send + Sync {
    fn search(&self, query: &str) -> KnowledgeResult<Vec<SearchResult>>;
}

struct DuckDuckGoSearchBackend {
    client: Client,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SearchResult {
    title: String,
    url: String,
    description: String,
    extra_snippets: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ManualCatalogEntry {
    ecosystem: String,
    package: String,
    #[serde(default)]
    version: Option<String>,
    source: String,
    locator: String,
    #[serde(default)]
    title: Option<String>,
    text: String,
    #[serde(default)]
    keywords: Vec<String>,
}

impl Default for KnowledgeSourceRegistry {
    fn default() -> Self {
        Self {
            sources: vec![
                Box::new(FileCatalogSource::default()),
                Box::new(RegistryMetadataSource::default()),
                Box::new(DeepWikiSource::default()),
                Box::new(DocsRsSource::default()),
                Box::new(RepositoryReadmeSource::default()),
                Box::new(PublicSearchSource::default()),
            ],
        }
    }
}

impl Default for FileCatalogSource {
    fn default() -> Self {
        Self::new(DEFAULT_CATALOG_PATH)
    }
}

impl Default for RegistryMetadataSource {
    fn default() -> Self {
        Self {
            client: Box::new(HttpRegistryMetadataClient::default()),
        }
    }
}

impl Default for DeepWikiSource {
    fn default() -> Self {
        Self {
            backend: Box::new(DuckDuckGoSearchBackend::default()),
        }
    }
}

impl Default for PublicSearchSource {
    fn default() -> Self {
        Self {
            backend: Box::new(DuckDuckGoSearchBackend::default()),
        }
    }
}

impl Default for DocsRsSource {
    fn default() -> Self {
        Self {
            client: http_client(),
        }
    }
}

impl Default for RepositoryReadmeSource {
    fn default() -> Self {
        Self {
            metadata: Box::new(HttpRegistryMetadataClient::default()),
            client: http_client(),
        }
    }
}

impl Default for HttpRegistryMetadataClient {
    fn default() -> Self {
        Self {
            client: http_client(),
        }
    }
}

impl Default for DuckDuckGoSearchBackend {
    fn default() -> Self {
        Self {
            client: http_client(),
        }
    }
}

fn http_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(5))
        .user_agent("shunt-agent/0.1")
        .build()
        .unwrap_or_default()
}

impl FileCatalogSource {
    fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            catalog_path: path.into(),
        }
    }

    fn load_catalog(&self, workspace_root: &Path) -> KnowledgeResult<Vec<ManualCatalogEntry>> {
        let path = if self.catalog_path.is_absolute() {
            self.catalog_path.clone()
        } else {
            workspace_root.join(&self.catalog_path)
        };
        if !path.exists() {
            return Ok(Vec::new());
        }
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }
}

impl KnowledgeSourceRegistry {
    pub(crate) fn search(
        &self,
        workspace_root: &Path,
        query: &KnowledgeQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<ManualEvidence>> {
        let config = KnowledgeConfig::load(workspace_root)?;
        let mut evidence = Vec::new();
        let mut seen = BTreeSet::new();

        for source in &self.sources {
            if !config.source_enabled(source.kind()) {
                continue;
            }
            let items = source.search(workspace_root, query, limit)?;
            if items.is_empty() && !query.research_requests.is_empty() {
                tracing::warn!(
                    source = ?source.kind(),
                    "knowledge source returned no evidence for planned research"
                );
            }
            for item in items {
                let item = item.into_manual_evidence();
                let key = format!(
                    "{}:{}:{}:{}",
                    item.ecosystem,
                    item.package,
                    item.version.as_deref().unwrap_or(""),
                    item.locator
                );
                if seen.insert(key) {
                    evidence.push(item);
                }
            }
        }

        evidence.sort_by(|left, right| {
            right
                .confidence
                .total_cmp(&left.confidence)
                .then_with(|| left.package.cmp(&right.package))
                .then_with(|| left.locator.cmp(&right.locator))
        });
        evidence.truncate(limit);
        Ok(evidence)
    }
}

impl KnowledgeSource for FileCatalogSource {
    fn kind(&self) -> KnowledgeSourceKind {
        KnowledgeSourceKind::Catalog
    }

    fn search(
        &self,
        workspace_root: &Path,
        query: &KnowledgeQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<KnowledgeEvidence>> {
        let catalog = self.load_catalog(workspace_root)?;
        if catalog.is_empty() {
            return Ok(Vec::new());
        }

        let mut matches = Vec::new();
        for fact in query_subjects(query) {
            let targeted = query_targets_package(query, &fact);
            for entry in &catalog {
                if entry.ecosystem != fact.ecosystem || entry.package != fact.name {
                    continue;
                }
                let version_status = version_status(&fact, entry);
                if version_status == ManualVersionStatus::Mismatch {
                    continue;
                }

                let overlap = token_overlap(query, entry);
                if overlap == 0 && !targeted {
                    continue;
                }

                let score =
                    version_status_score(version_status) + overlap as f32 + f32::from(targeted);
                matches.push((
                    score,
                    KnowledgeEvidence {
                        ecosystem: fact.ecosystem.clone(),
                        package: fact.name.clone(),
                        version: entry.version.clone(),
                        version_status,
                        source: KnowledgeSourceRef {
                            source: entry.source.clone(),
                            locator: entry.locator.clone(),
                            fetched_at: None,
                        },
                        title: entry.title.clone(),
                        excerpt: clamp_excerpt(&entry.text),
                        relevance_reason: relevance_reason(overlap, targeted, &fact.name),
                        confidence: score.min(10.0) / 10.0,
                        fetch_status: KnowledgeFetchStatus::Static,
                    },
                ));
            }
        }

        matches.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.package.cmp(&right.1.package))
                .then_with(|| left.1.source.locator.cmp(&right.1.source.locator))
        });

        let mut seen = BTreeSet::new();
        let mut evidence = Vec::new();
        for (_, item) in matches {
            let key = format!(
                "{}:{}:{}",
                item.package,
                item.version.as_deref().unwrap_or(""),
                item.source.locator
            );
            if seen.insert(key) {
                evidence.push(item);
            }
            if evidence.len() >= limit {
                break;
            }
        }

        Ok(evidence)
    }
}

impl KnowledgeSource for RegistryMetadataSource {
    fn kind(&self) -> KnowledgeSourceKind {
        KnowledgeSourceKind::RegistryMetadata
    }

    fn search(
        &self,
        _workspace_root: &Path,
        query: &KnowledgeQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<KnowledgeEvidence>> {
        let mut evidence = Vec::new();
        for fact in query_subjects(query) {
            let Some(item) = self.search_package(query, &fact)? else {
                continue;
            };
            evidence.push(item);
        }
        evidence.sort_by(|left, right| {
            right
                .confidence
                .total_cmp(&left.confidence)
                .then_with(|| left.package.cmp(&right.package))
                .then_with(|| left.source.locator.cmp(&right.source.locator))
        });
        evidence.truncate(limit);
        Ok(evidence)
    }
}

impl KnowledgeSource for DeepWikiSource {
    fn kind(&self) -> KnowledgeSourceKind {
        KnowledgeSourceKind::Deepwiki
    }

    fn search(
        &self,
        _workspace_root: &Path,
        query: &KnowledgeQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<KnowledgeEvidence>> {
        search_backed_results(query, limit, &*self.backend, SearchMode::DeepWiki)
    }
}

impl KnowledgeSource for PublicSearchSource {
    fn kind(&self) -> KnowledgeSourceKind {
        KnowledgeSourceKind::PublicSearch
    }

    fn search(
        &self,
        _workspace_root: &Path,
        query: &KnowledgeQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<KnowledgeEvidence>> {
        search_backed_results(query, limit, &*self.backend, SearchMode::PublicSearch)
    }
}

impl KnowledgeSource for DocsRsSource {
    fn kind(&self) -> KnowledgeSourceKind {
        KnowledgeSourceKind::DocsRs
    }

    fn search(
        &self,
        _workspace_root: &Path,
        query: &KnowledgeQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<KnowledgeEvidence>> {
        let mut evidence = Vec::new();
        for fact in query_subjects(query) {
            if fact.ecosystem != "cargo" || !query_targets_package(query, &fact) {
                continue;
            }
            let Some(item) = docs_rs_evidence(&self.client, query, &fact) else {
                continue;
            };
            evidence.push(item);
        }
        evidence.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
        evidence.truncate(limit);
        Ok(evidence)
    }
}

impl KnowledgeSource for RepositoryReadmeSource {
    fn kind(&self) -> KnowledgeSourceKind {
        KnowledgeSourceKind::RepositoryReadme
    }

    fn search(
        &self,
        _workspace_root: &Path,
        query: &KnowledgeQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<KnowledgeEvidence>> {
        let mut evidence = Vec::new();
        for fact in query_subjects(query) {
            if !query_targets_package(query, &fact) {
                continue;
            }
            let Some(item) =
                repository_readme_evidence(&*self.metadata, &self.client, query, &fact)
            else {
                continue;
            };
            evidence.push(item);
        }
        evidence.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
        evidence.truncate(limit);
        Ok(evidence)
    }
}

#[derive(Debug, Clone, Copy)]
enum SearchMode {
    DeepWiki,
    PublicSearch,
}

fn search_backed_results(
    query: &KnowledgeQuery,
    limit: usize,
    backend: &dyn SearchBackend,
    mode: SearchMode,
) -> KnowledgeResult<Vec<KnowledgeEvidence>> {
    let mut evidence = Vec::new();
    for fact in query_subjects(query) {
        if !query_targets_package(query, &fact) {
            continue;
        }
        let search_query = build_search_query(query, &fact, mode);
        let results = backend.search(&search_query)?;
        let Some(result) = select_search_result(results, &fact, mode) else {
            continue;
        };
        let excerpt = search_result_excerpt(&result);
        let overlap = token_overlap_text(query, &excerpt, &[]);
        if overlap == 0 {
            continue;
        }
        let (source, title_prefix) = match mode {
            SearchMode::DeepWiki => ("deepwiki", "DeepWiki"),
            SearchMode::PublicSearch => ("public-search", "Search"),
        };
        let score = version_status_score(ManualVersionStatus::Unversioned)
            + overlap as f32
            + search_result_quality_score(&result, &fact, mode);
        evidence.push(KnowledgeEvidence {
            ecosystem: fact.ecosystem.clone(),
            package: fact.name.clone(),
            version: None,
            version_status: ManualVersionStatus::Unversioned,
            source: KnowledgeSourceRef {
                source: source.into(),
                locator: result.url.clone(),
                fetched_at: Some(OffsetDateTime::now_utc()),
            },
            title: Some(format!("{title_prefix}: {}", result.title)),
            excerpt: clamp_excerpt(&excerpt),
            relevance_reason: relevance_reason(overlap, true, &fact.name),
            confidence: score.min(10.0) / 10.0,
            fetch_status: KnowledgeFetchStatus::Live,
        });
    }
    evidence.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.package.cmp(&right.package))
            .then_with(|| left.source.locator.cmp(&right.source.locator))
    });
    evidence.truncate(limit);
    Ok(evidence)
}

fn build_search_query(query: &KnowledgeQuery, fact: &PackageFact, mode: SearchMode) -> String {
    let mut parts = Vec::new();
    if matches!(mode, SearchMode::DeepWiki) {
        parts.push("site:deepwiki.com".to_string());
    }
    parts.push(fact.name.clone());
    if let Some(version) = &fact.version {
        parts.push(version.clone());
    }
    parts.extend(query.requested_topics.iter().take(4).cloned());
    parts.push(query.interpreted_goal.clone());
    parts.join(" ")
}

fn query_subjects(query: &KnowledgeQuery) -> Vec<PackageFact> {
    let mut subjects = query.package_facts.clone();
    let mut seen = subjects
        .iter()
        .map(|fact| format!("{}:{}", fact.ecosystem, normalized_search_key(&fact.name)))
        .collect::<BTreeSet<_>>();
    for fact in planned_package_facts(&query.research_requests) {
        let key = format!("{}:{}", fact.ecosystem, normalized_search_key(&fact.name));
        if seen.insert(key) {
            subjects.push(fact);
        }
    }
    subjects
}

fn planned_package_facts(requests: &[KnowledgeResearchRequest]) -> Vec<PackageFact> {
    let mut facts = Vec::new();
    for request in requests {
        for package in &request.package_hints {
            let ecosystem = request
                .ecosystem_hints
                .first()
                .cloned()
                .unwrap_or_else(|| "unknown".into());
            facts.push(PackageFact {
                ecosystem,
                name: package.to_ascii_lowercase(),
                version: None,
                requirement: None,
                version_provenance: shunt_core::PackageVersionProvenance::Unknown,
                manifest_path: "<research-plan>".into(),
                evidence: vec![],
                confidence: 0.45,
            });
        }
    }
    facts
}

fn search_result_excerpt(result: &SearchResult) -> String {
    let mut text = result.description.clone();
    if let Some(extra) = result.extra_snippets.first() {
        text.push(' ');
        text.push_str(extra);
    }
    text
}

fn select_search_result(
    results: Vec<SearchResult>,
    fact: &PackageFact,
    mode: SearchMode,
) -> Option<SearchResult> {
    results
        .into_iter()
        .filter(|result| search_result_quality_score(result, fact, mode) > 0.0)
        .max_by(|left, right| {
            search_result_quality_score(left, fact, mode)
                .total_cmp(&search_result_quality_score(right, fact, mode))
        })
}

fn search_result_quality_score(result: &SearchResult, fact: &PackageFact, mode: SearchMode) -> f32 {
    if matches!(mode, SearchMode::DeepWiki) {
        return 2.0;
    }
    curated_domain_score(&result.url, &fact.name).unwrap_or(0.0)
}

fn curated_domain_score(url: &str, package: &str) -> Option<f32> {
    let domain = Url::parse(url).ok()?.domain()?.to_ascii_lowercase();
    let package_key = normalized_search_key(package);
    let domain_key = normalized_search_key(&domain);
    if domain.ends_with("docs.rs")
        || domain.ends_with("github.com")
        || domain.ends_with("github.io")
        || domain.ends_with("readthedocs.io")
        || domain.ends_with("deepwiki.com")
        || domain.ends_with("npmjs.com")
        || domain.ends_with("crates.io")
    {
        return Some(2.0);
    }
    if [
        "developer.mozilla.org",
        "react.dev",
        "nextjs.org",
        "vite.dev",
        "tailwindcss.com",
        "typescriptlang.org",
        "rust-lang.org",
        "tokio.rs",
        "serde.rs",
        "svelte.dev",
        "angular.dev",
        "zod.dev",
        "tanstack.com",
        "freecodecamp.org",
        "logrocket.com",
    ]
    .iter()
    .any(|trusted| domain.ends_with(trusted))
    {
        return Some(1.5);
    }
    if domain_key.contains(&package_key) {
        return Some(1.0);
    }
    None
}

fn normalized_search_key(input: &str) -> String {
    input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn docs_rs_evidence(
    client: &Client,
    query: &KnowledgeQuery,
    fact: &PackageFact,
) -> Option<KnowledgeEvidence> {
    let version = fact.version.clone().unwrap_or_else(|| "latest".into());
    let url = format!("https://docs.rs/crate/{}/{version}", fact.name);
    let html = client
        .get(&url)
        .send()
        .ok()?
        .error_for_status()
        .ok()?
        .text()
        .ok()?;
    let excerpt = html_excerpt(&html, &["main", "body"])?;
    let overlap = token_overlap_text(query, &excerpt, &[]);
    if overlap == 0 && !query_targets_package(query, fact) {
        return None;
    }
    let score = version_status_score(version_status_for_version(fact, fact.version.as_deref()))
        + overlap as f32
        + 2.0;
    Some(KnowledgeEvidence {
        ecosystem: fact.ecosystem.clone(),
        package: fact.name.clone(),
        version: fact.version.clone(),
        version_status: version_status_for_version(fact, fact.version.as_deref()),
        source: KnowledgeSourceRef {
            source: "docs.rs".into(),
            locator: url,
            fetched_at: Some(OffsetDateTime::now_utc()),
        },
        title: Some(format!("docs.rs: {}", fact.name)),
        excerpt: clamp_excerpt(&excerpt),
        relevance_reason: relevance_reason(overlap, true, &fact.name),
        confidence: score.min(10.0) / 10.0,
        fetch_status: KnowledgeFetchStatus::Live,
    })
}

fn repository_readme_evidence(
    metadata: &dyn RegistryMetadataClient,
    client: &Client,
    query: &KnowledgeQuery,
    fact: &PackageFact,
) -> Option<KnowledgeEvidence> {
    let repo_url = match fact.ecosystem.as_str() {
        "cargo" => metadata
            .crate_metadata(&fact.name)
            .ok()??
            .repository_url()?,
        "npm" => metadata.npm_metadata(&fact.name).ok()??.repository_url()?,
        _ => return None,
    };
    let html = client
        .get(&repo_url)
        .send()
        .ok()?
        .error_for_status()
        .ok()?
        .text()
        .ok()?;
    let excerpt = html_excerpt(
        &html,
        &[
            "article.markdown-body",
            "div.markdown-body",
            "main",
            "article",
            "body",
        ],
    )?;
    let overlap = token_overlap_text(query, &excerpt, &[]);
    if overlap == 0 && !query_targets_package(query, fact) {
        return None;
    }
    let score = version_status_score(ManualVersionStatus::Unversioned) + overlap as f32 + 1.5;
    Some(KnowledgeEvidence {
        ecosystem: fact.ecosystem.clone(),
        package: fact.name.clone(),
        version: None,
        version_status: ManualVersionStatus::Unversioned,
        source: KnowledgeSourceRef {
            source: "repository-readme".into(),
            locator: repo_url,
            fetched_at: Some(OffsetDateTime::now_utc()),
        },
        title: Some(format!("Repository README: {}", fact.name)),
        excerpt: clamp_excerpt(&excerpt),
        relevance_reason: relevance_reason(overlap, true, &fact.name),
        confidence: score.min(10.0) / 10.0,
        fetch_status: KnowledgeFetchStatus::Live,
    })
}

fn html_excerpt(html: &str, selectors: &[&str]) -> Option<String> {
    let document = Html::parse_document(html);
    for pattern in selectors {
        let selector = Selector::parse(pattern).ok()?;
        let text = document
            .select(&selector)
            .flat_map(|node| node.text())
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

impl RegistryMetadataSource {
    #[cfg(test)]
    fn with_client(client: Box<dyn RegistryMetadataClient>) -> Self {
        Self { client }
    }

    fn search_package(
        &self,
        query: &KnowledgeQuery,
        fact: &PackageFact,
    ) -> KnowledgeResult<Option<KnowledgeEvidence>> {
        match fact.ecosystem.as_str() {
            "npm" => self.search_npm_package(query, fact),
            "cargo" => self.search_cargo_package(query, fact),
            _ => Ok(None),
        }
    }

    fn search_npm_package(
        &self,
        query: &KnowledgeQuery,
        fact: &PackageFact,
    ) -> KnowledgeResult<Option<KnowledgeEvidence>> {
        let Some(metadata) = self.client.npm_metadata(&fact.name)? else {
            return Ok(None);
        };
        let selected = metadata.select_for(fact);
        let Some(text) = selected.text() else {
            return Ok(None);
        };
        let overlap = token_overlap_text(query, &text, &selected.keywords);
        let targeted = query_targets_package(query, fact);
        if overlap == 0 && !targeted {
            return Ok(None);
        }

        let version_status = selected.version_status(fact);
        if version_status == ManualVersionStatus::Mismatch {
            return Ok(None);
        }

        let score = version_status_score(version_status) + overlap as f32 + f32::from(targeted);
        Ok(Some(KnowledgeEvidence {
            ecosystem: fact.ecosystem.clone(),
            package: fact.name.clone(),
            version: selected.version.clone(),
            version_status,
            source: KnowledgeSourceRef {
                source: "npm-registry".into(),
                locator: format!(
                    "https://registry.npmjs.org/{}",
                    fact.name.replace('/', "%2F")
                ),
                fetched_at: Some(OffsetDateTime::now_utc()),
            },
            title: Some(format!("{} package metadata", fact.name)),
            excerpt: clamp_excerpt(&text),
            relevance_reason: relevance_reason(overlap, targeted, &fact.name),
            confidence: score.min(10.0) / 10.0,
            fetch_status: KnowledgeFetchStatus::Live,
        }))
    }

    fn search_cargo_package(
        &self,
        query: &KnowledgeQuery,
        fact: &PackageFact,
    ) -> KnowledgeResult<Option<KnowledgeEvidence>> {
        let Some(metadata) = self.client.crate_metadata(&fact.name)? else {
            return Ok(None);
        };
        let Some(text) = metadata.text() else {
            return Ok(None);
        };
        let overlap = token_overlap_text(query, &text, &[]);
        let targeted = query_targets_package(query, fact);
        if overlap == 0 || !targeted {
            return Ok(None);
        }

        let score = version_status_score(ManualVersionStatus::Unversioned)
            + overlap as f32
            + f32::from(targeted);
        Ok(Some(KnowledgeEvidence {
            ecosystem: fact.ecosystem.clone(),
            package: fact.name.clone(),
            version: None,
            version_status: ManualVersionStatus::Unversioned,
            source: KnowledgeSourceRef {
                source: "crates-io".into(),
                locator: format!("https://crates.io/crates/{}", fact.name),
                fetched_at: Some(OffsetDateTime::now_utc()),
            },
            title: Some(format!("{} crate metadata", fact.name)),
            excerpt: clamp_excerpt(&text),
            relevance_reason: relevance_reason(overlap, targeted, &fact.name),
            confidence: score.min(10.0) / 10.0,
            fetch_status: KnowledgeFetchStatus::Live,
        }))
    }
}

impl RegistryMetadataClient for HttpRegistryMetadataClient {
    fn npm_metadata(&self, package: &str) -> KnowledgeResult<Option<NpmPackageMetadata>> {
        let encoded = package.replace('/', "%2F");
        let url = format!("https://registry.npmjs.org/{encoded}");
        Ok(self.client.get(url).send()?.error_for_status()?.json()?)
    }

    fn crate_metadata(&self, package: &str) -> KnowledgeResult<Option<CratePackageMetadata>> {
        let url = format!("https://crates.io/api/v1/crates/{package}");
        let response: CratesIoResponse = self.client.get(url).send()?.error_for_status()?.json()?;
        Ok(response.krate)
    }
}

impl SearchBackend for DuckDuckGoSearchBackend {
    fn search(&self, query: &str) -> KnowledgeResult<Vec<SearchResult>> {
        let html = self
            .client
            .get("https://html.duckduckgo.com/html/")
            .query(&[("q", query)])
            .send()?
            .error_for_status()?
            .text()?;
        Ok(parse_duckduckgo_results(&html))
    }
}

fn parse_duckduckgo_results(html: &str) -> Vec<SearchResult> {
    let document = Html::parse_document(html);
    let result_selector = match Selector::parse("div.result") {
        Ok(selector) => selector,
        Err(_) => return Vec::new(),
    };
    let title_selector = match Selector::parse("a.result__a") {
        Ok(selector) => selector,
        Err(_) => return Vec::new(),
    };
    let snippet_selector = match Selector::parse(".result__snippet") {
        Ok(selector) => selector,
        Err(_) => return Vec::new(),
    };

    document
        .select(&result_selector)
        .filter_map(|result| {
            let title_node = result.select(&title_selector).next()?;
            let title = title_node
                .text()
                .collect::<Vec<_>>()
                .join(" ")
                .trim()
                .to_string();
            let href = title_node.value().attr("href")?;
            let url = decode_duckduckgo_redirect(href);
            let description = result
                .select(&snippet_selector)
                .next()
                .map(|node| node.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .unwrap_or_default();
            if title.is_empty() || url.is_empty() {
                return None;
            }
            Some(SearchResult {
                title,
                url,
                description,
                extra_snippets: Vec::new(),
            })
        })
        .collect()
}

fn decode_duckduckgo_redirect(href: &str) -> String {
    let absolute = if href.starts_with("http://") || href.starts_with("https://") {
        href.to_string()
    } else {
        format!("https://html.duckduckgo.com{href}")
    };
    let Ok(url) = Url::parse(&absolute) else {
        return href.to_string();
    };
    url.query_pairs()
        .find_map(|(key, value)| (key == "uddg").then(|| value.into_owned()))
        .unwrap_or_else(|| href.to_string())
}

#[derive(Debug, Clone, Deserialize)]
struct NpmVersionMetadata {
    version: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    readme: Option<String>,
    #[serde(default)]
    keywords: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct NpmDistTags {
    latest: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct NpmPackageMetadata {
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    readme: Option<String>,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default)]
    homepage: Option<String>,
    #[serde(default)]
    repository: Option<NpmRepositoryField>,
    #[serde(rename = "dist-tags")]
    dist_tags: Option<NpmDistTags>,
    #[serde(default)]
    versions: std::collections::BTreeMap<String, NpmVersionMetadata>,
}

#[derive(Debug, Clone)]
struct SelectedNpmMetadata {
    version: Option<String>,
    description: Option<String>,
    readme: Option<String>,
    keywords: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum NpmRepositoryField {
    Url(String),
    Object { url: Option<String> },
}

impl NpmPackageMetadata {
    fn select_for(&self, fact: &PackageFact) -> SelectedNpmMetadata {
        let exact = fact
            .version
            .as_deref()
            .and_then(|version| self.versions.get(version))
            .cloned();
        let latest = self
            .dist_tags
            .as_ref()
            .and_then(|tags| tags.latest.as_deref())
            .and_then(|version| self.versions.get(version))
            .cloned();
        let selected = exact.or(latest);
        SelectedNpmMetadata {
            version: selected
                .as_ref()
                .and_then(|item| item.version.clone())
                .or_else(|| fact.version.clone()),
            description: selected
                .as_ref()
                .and_then(|item| item.description.clone())
                .or_else(|| self.description.clone()),
            readme: selected
                .as_ref()
                .and_then(|item| item.readme.clone())
                .or_else(|| self.readme.clone()),
            keywords: selected
                .as_ref()
                .map(|item| item.keywords.clone())
                .filter(|keywords| !keywords.is_empty())
                .unwrap_or_else(|| self.keywords.clone()),
        }
    }

    fn repository_url(&self) -> Option<String> {
        self.repository
            .as_ref()
            .and_then(NpmRepositoryField::url)
            .or_else(|| self.homepage.clone())
    }
}

impl NpmRepositoryField {
    fn url(&self) -> Option<String> {
        let raw = match self {
            Self::Url(url) => url.clone(),
            Self::Object { url } => url.clone()?,
        };
        normalize_repository_url(&raw)
    }
}

impl SelectedNpmMetadata {
    fn text(&self) -> Option<String> {
        self.readme
            .clone()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                self.description
                    .clone()
                    .filter(|value| !value.trim().is_empty())
            })
    }

    fn version_status(&self, fact: &PackageFact) -> ManualVersionStatus {
        match self.version.as_deref() {
            Some(version) => version_status_for_version(fact, Some(version)),
            None => ManualVersionStatus::Unversioned,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CratesIoResponse {
    #[serde(rename = "crate")]
    krate: Option<CratePackageMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
struct CratePackageMetadata {
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    documentation: Option<String>,
    #[serde(default)]
    homepage: Option<String>,
    #[serde(default)]
    repository: Option<String>,
}

impl CratePackageMetadata {
    fn text(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(description) = &self.description {
            parts.push(description.as_str());
        }
        if let Some(documentation) = &self.documentation {
            parts.push(documentation.as_str());
        }
        if let Some(homepage) = &self.homepage {
            parts.push(homepage.as_str());
        }
        if let Some(repository) = &self.repository {
            parts.push(repository.as_str());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    }

    fn repository_url(&self) -> Option<String> {
        self.repository
            .clone()
            .or_else(|| self.homepage.clone())
            .or_else(|| self.documentation.clone())
            .and_then(|url| normalize_repository_url(&url))
    }
}

fn normalize_repository_url(url: &str) -> Option<String> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return None;
    }
    let trimmed = trimmed
        .strip_prefix("git+")
        .unwrap_or(trimmed)
        .strip_suffix(".git")
        .unwrap_or(trimmed)
        .to_string();
    if let Some(rest) = trimmed.strip_prefix("git://") {
        return Some(format!("https://{rest}"));
    }
    if let Some(rest) = trimmed.strip_prefix("git@github.com:") {
        return Some(format!("https://github.com/{rest}"));
    }
    Some(trimmed)
}

fn version_status(fact: &PackageFact, entry: &ManualCatalogEntry) -> ManualVersionStatus {
    version_status_for_version(fact, entry.version.as_deref())
}

fn version_status_for_version(
    fact: &PackageFact,
    entry_version: Option<&str>,
) -> ManualVersionStatus {
    match (fact.version.as_deref(), entry_version) {
        (Some(fact_version), Some(entry_version)) if fact_version == entry_version => {
            ManualVersionStatus::Exact
        }
        (_, None) => ManualVersionStatus::Unversioned,
        (_, Some(entry_version))
            if matches_requirement(fact.requirement.as_deref(), entry_version) =>
        {
            ManualVersionStatus::CompatibleRange
        }
        (None, Some(_)) => ManualVersionStatus::Unversioned,
        _ => ManualVersionStatus::Mismatch,
    }
}

fn matches_requirement(requirement: Option<&str>, version: &str) -> bool {
    let Some(requirement) = requirement else {
        return false;
    };
    let Ok(requirement) = VersionReq::parse(requirement) else {
        return false;
    };
    let Ok(version) = Version::parse(version) else {
        return false;
    };
    requirement.matches(&version)
}

fn token_overlap(query: &KnowledgeQuery, entry: &ManualCatalogEntry) -> usize {
    let mut haystack = String::new();
    if let Some(title) = &entry.title {
        haystack.push_str(title);
        haystack.push(' ');
    }
    haystack.push_str(&entry.text);
    haystack.push(' ');
    haystack.push_str(&entry.keywords.join(" "));
    let haystack = haystack.to_ascii_lowercase();
    query
        .requested_topics
        .iter()
        .filter(|term| haystack.contains(term.as_str()))
        .count()
}

fn token_overlap_text(query: &KnowledgeQuery, text: &str, keywords: &[String]) -> usize {
    let mut haystack = text.to_ascii_lowercase();
    if !keywords.is_empty() {
        haystack.push(' ');
        haystack.push_str(&keywords.join(" ").to_ascii_lowercase());
    }
    query
        .requested_topics
        .iter()
        .filter(|term| haystack.contains(term.as_str()))
        .count()
}

fn query_targets_package(query: &KnowledgeQuery, fact: &PackageFact) -> bool {
    let request =
        format!("{} {}", query.original_request, query.interpreted_goal).to_ascii_lowercase();
    let package = fact.name.to_ascii_lowercase();
    request.contains(&package)
        || package
            .split(|ch: char| !ch.is_ascii_alphanumeric())
            .filter(|part| !part.is_empty())
            .any(|part| request.contains(part))
}

fn relevance_reason(overlap: usize, targeted: bool, package: &str) -> String {
    match (overlap, targeted) {
        (0, true) => format!("package {package} was named in the request"),
        (_, true) => format!("matched {overlap} request terms and package {package} was named"),
        _ => format!("matched {overlap} request terms for {package}"),
    }
}

fn version_status_score(status: ManualVersionStatus) -> f32 {
    match status {
        ManualVersionStatus::Exact => 6.0,
        ManualVersionStatus::CompatibleRange => 4.0,
        ManualVersionStatus::Unversioned => 2.0,
        ManualVersionStatus::Mismatch => 0.0,
    }
}

fn clamp_excerpt(input: &str) -> String {
    let compact = input.replace('\n', " ");
    if compact.chars().count() <= MAX_EXCERPT_CHARS {
        compact
    } else {
        let preview = compact.chars().take(MAX_EXCERPT_CHARS).collect::<String>();
        format!("{preview}...")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use shunt_core::{PackageFact, PackageVersionProvenance};

    use super::{
        DeepWikiSource, FileCatalogSource, KnowledgeSource, NpmDistTags, NpmPackageMetadata,
        NpmVersionMetadata, PublicSearchSource, RegistryMetadataClient, RegistryMetadataSource,
        SearchBackend, SearchResult,
    };
    use crate::{KnowledgeQuery, KnowledgeResult};
    use std::path::Path;

    #[test]
    fn file_catalog_provider_prefers_exact_versioned_manuals() {
        let root =
            std::env::temp_dir().join(format!("shunt-knowledge-catalog-{}", std::process::id()));
        let catalog_dir = root.join(".shunt/manuals");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&catalog_dir).unwrap();
        fs::write(
            catalog_dir.join("catalog.json"),
            serde_json::to_string(&vec![
                serde_json::json!({
                    "ecosystem": "cargo",
                    "package": "ratatui",
                    "version": "0.29.0",
                    "source": "deepwiki",
                    "locator": "ratatui/layout",
                    "title": "Layout",
                    "text": "Use the Layout API to split terminal areas cleanly.",
                    "keywords": ["layout", "split", "terminal"]
                }),
                serde_json::json!({
                    "ecosystem": "cargo",
                    "package": "ratatui",
                    "version": "0.28.0",
                    "source": "deepwiki",
                    "locator": "ratatui/old-layout",
                    "title": "Old Layout",
                    "text": "Legacy layout API.",
                    "keywords": ["layout"]
                }),
            ])
            .unwrap(),
        )
        .unwrap();

        let query = KnowledgeQuery {
            original_request: "fix ratatui layout rendering".into(),
            interpreted_goal: "repair ratatui layout rendering".into(),
            located_paths: vec!["src/main.rs".into()],
            requested_topics: vec!["layout".into()],
            package_facts: vec![PackageFact {
                ecosystem: "cargo".into(),
                name: "ratatui".into(),
                version: Some("0.29.0".into()),
                requirement: Some("0.29".into()),
                version_provenance: PackageVersionProvenance::ExactLock,
                manifest_path: "Cargo.toml".into(),
                evidence: vec![],
                confidence: 0.95,
            }],
            research_requests: vec![],
        };

        let results = FileCatalogSource::default()
            .search(&root, &query, 2)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source.locator, "ratatui/layout");
        assert_eq!(
            results[0].version_status,
            shunt_core::ManualVersionStatus::Exact
        );
    }

    #[test]
    fn registry_metadata_source_uses_exact_npm_version_when_available() {
        let source = RegistryMetadataSource::with_client(Box::new(FakeRegistryMetadataClient {
            npm: Some(NpmPackageMetadata {
                description: Some("schema validation".into()),
                readme: None,
                keywords: vec!["validation".into()],
                homepage: None,
                repository: None,
                dist_tags: Some(NpmDistTags {
                    latest: Some("4.0.0".into()),
                }),
                versions: BTreeMap::from([(
                    "3.22.4".into(),
                    NpmVersionMetadata {
                        version: Some("3.22.4".into()),
                        description: Some("exact zod docs".into()),
                        readme: Some("Use z.coerce.number().catch(3) for defaults.".into()),
                        keywords: vec!["coerce".into(), "number".into()],
                    },
                )]),
            }),
            krate: None,
        }));
        let query = KnowledgeQuery {
            original_request: "Use zod to parse RETRY_COUNT".into(),
            interpreted_goal: "Use zod to parse RETRY_COUNT".into(),
            located_paths: vec!["src/env.ts".into()],
            requested_topics: vec!["zod".into(), "coerce".into(), "number".into()],
            package_facts: vec![PackageFact {
                ecosystem: "npm".into(),
                name: "zod".into(),
                version: Some("3.22.4".into()),
                requirement: Some("^3.22.4".into()),
                version_provenance: PackageVersionProvenance::ExactLock,
                manifest_path: "package.json".into(),
                evidence: vec![],
                confidence: 0.95,
            }],
            research_requests: vec![],
        };

        let results = source.search(Path::new("."), &query, 2).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].version.as_deref(), Some("3.22.4"));
        assert_eq!(
            results[0].version_status,
            shunt_core::ManualVersionStatus::Exact
        );
        assert!(results[0].excerpt.contains("z.coerce.number().catch(3)"));
    }

    #[test]
    fn deepwiki_source_uses_search_result_snippets() {
        let source = DeepWikiSource {
            backend: Box::new(FakeSearchBackend {
                results: vec![SearchResult {
                    title: "Ratatui Layout".into(),
                    url: "https://deepwiki.com/ratatui/layout".into(),
                    description: "Use Layout::vertical for header/body splits.".into(),
                    extra_snippets: vec!["Constraint::Length(3) and Constraint::Min(0).".into()],
                }],
            }),
        };
        let query = KnowledgeQuery {
            original_request: "Fix ratatui layout rendering".into(),
            interpreted_goal: "Fix ratatui layout rendering".into(),
            located_paths: vec!["src/ui.rs".into()],
            requested_topics: vec!["layout".into(), "constraint".into()],
            package_facts: vec![PackageFact {
                ecosystem: "cargo".into(),
                name: "ratatui".into(),
                version: Some("0.29.0".into()),
                requirement: Some("0.29".into()),
                version_provenance: PackageVersionProvenance::ExactLock,
                manifest_path: "Cargo.toml".into(),
                evidence: vec![],
                confidence: 0.95,
            }],
            research_requests: vec![],
        };

        let results = source.search(Path::new("."), &query, 2).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source.source, "deepwiki");
        assert!(results[0].excerpt.contains("Layout::vertical"));
    }

    #[test]
    fn public_search_source_requires_explicit_package_target() {
        let source = PublicSearchSource {
            backend: Box::new(FakeSearchBackend {
                results: vec![SearchResult {
                    title: "Zod Guide".into(),
                    url: "https://example.com/zod".into(),
                    description: "Use z.coerce.number().int().min(1).catch(3).".into(),
                    extra_snippets: vec![],
                }],
            }),
        };
        let query = KnowledgeQuery {
            original_request: "Parse RETRY_COUNT in src/env.ts".into(),
            interpreted_goal: "Parse RETRY_COUNT in src/env.ts".into(),
            located_paths: vec!["src/env.ts".into()],
            requested_topics: vec!["coerce".into(), "number".into()],
            package_facts: vec![PackageFact {
                ecosystem: "npm".into(),
                name: "zod".into(),
                version: Some("3.22.4".into()),
                requirement: Some("^3.22.4".into()),
                version_provenance: PackageVersionProvenance::ExactLock,
                manifest_path: "package.json".into(),
                evidence: vec![],
                confidence: 0.95,
            }],
            research_requests: vec![],
        };

        let results = source.search(Path::new("."), &query, 2).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn file_catalog_source_keeps_explicit_package_mentions_without_topic_overlap() {
        let root = std::env::temp_dir().join(format!(
            "shunt-knowledge-explicit-package-{}",
            std::process::id()
        ));
        let catalog_dir = root.join(".shunt/manuals");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&catalog_dir).unwrap();
        fs::write(
            catalog_dir.join("catalog.json"),
            serde_json::to_string(&vec![serde_json::json!({
                "ecosystem": "npm",
                "package": "zod",
                "version": "3.22.4",
                "source": "manual-catalog",
                "locator": "zod/coerce",
                "title": "Coerce",
                "text": "Use z.coerce.number().int().min(1).catch(3).",
                "keywords": ["coerce", "number"]
            })])
            .unwrap(),
        )
        .unwrap();

        let query = KnowledgeQuery {
            original_request: "Use zod in src/env.ts".into(),
            interpreted_goal: "Use zod in src/env.ts".into(),
            located_paths: vec!["src/env.ts".into()],
            requested_topics: vec!["retry".into()],
            package_facts: vec![PackageFact {
                ecosystem: "npm".into(),
                name: "zod".into(),
                version: Some("3.22.4".into()),
                requirement: Some("^3.22.4".into()),
                version_provenance: PackageVersionProvenance::ExactLock,
                manifest_path: "package.json".into(),
                evidence: vec![],
                confidence: 0.95,
            }],
            research_requests: vec![],
        };

        let results = FileCatalogSource::default()
            .search(&root, &query, 2)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(
            results[0]
                .relevance_reason
                .contains("package zod was named")
        );
    }

    #[test]
    fn file_catalog_source_excludes_mismatched_versions() {
        let root = std::env::temp_dir().join(format!(
            "shunt-knowledge-version-mismatch-{}",
            std::process::id()
        ));
        let catalog_dir = root.join(".shunt/manuals");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&catalog_dir).unwrap();
        fs::write(
            catalog_dir.join("catalog.json"),
            serde_json::to_string(&vec![serde_json::json!({
                "ecosystem": "cargo",
                "package": "ratatui",
                "version": "0.28.0",
                "source": "manual-catalog",
                "locator": "ratatui/old-layout",
                "title": "Old Layout",
                "text": "Legacy layout API.",
                "keywords": ["layout"]
            })])
            .unwrap(),
        )
        .unwrap();

        let query = KnowledgeQuery {
            original_request: "Fix ratatui layout rendering".into(),
            interpreted_goal: "Fix ratatui layout rendering".into(),
            located_paths: vec!["src/ui.rs".into()],
            requested_topics: vec!["layout".into()],
            package_facts: vec![PackageFact {
                ecosystem: "cargo".into(),
                name: "ratatui".into(),
                version: Some("0.29.0".into()),
                requirement: Some("0.29".into()),
                version_provenance: PackageVersionProvenance::ExactLock,
                manifest_path: "Cargo.toml".into(),
                evidence: vec![],
                confidence: 0.95,
            }],
            research_requests: vec![],
        };

        let results = FileCatalogSource::default()
            .search(&root, &query, 2)
            .unwrap();

        assert!(results.is_empty());
    }

    struct FakeRegistryMetadataClient {
        npm: Option<NpmPackageMetadata>,
        krate: Option<super::CratePackageMetadata>,
    }

    struct FakeSearchBackend {
        results: Vec<SearchResult>,
    }

    impl RegistryMetadataClient for FakeRegistryMetadataClient {
        fn npm_metadata(&self, _package: &str) -> KnowledgeResult<Option<NpmPackageMetadata>> {
            Ok(self.npm.clone())
        }

        fn crate_metadata(
            &self,
            _package: &str,
        ) -> KnowledgeResult<Option<super::CratePackageMetadata>> {
            Ok(self.krate.clone())
        }
    }

    impl SearchBackend for FakeSearchBackend {
        fn search(&self, _query: &str) -> KnowledgeResult<Vec<SearchResult>> {
            Ok(self.results.clone())
        }
    }
}
