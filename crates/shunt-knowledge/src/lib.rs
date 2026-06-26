use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use shunt_core::{
    Ambiguity, EvidenceKind, EvidenceRef, ManualEvidence, ManualQuery, ManualVersionStatus,
    PackageFact, PackageVersionProvenance, UnderstandingArtifact,
};
use shunt_localize::ContextPacket;
use reqwest::blocking::Client;
use semver::{Version, VersionReq};
use serde::Deserialize;
use thiserror::Error;

const DEFAULT_CATALOG_PATH: &str = ".shunt/manuals/catalog.json";
const DEFAULT_MANUAL_LIMIT: usize = 3;
const MAX_EXCERPT_CHARS: usize = 420;

#[derive(Debug, Error)]
pub enum KnowledgeError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("toml error: {0}")]
    Toml(#[from] toml::de::Error),
}

pub type KnowledgeResult<T> = Result<T, KnowledgeError>;

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgePacket {
    pub package_facts: Vec<PackageFact>,
    pub manual_evidence: Vec<ManualEvidence>,
}

pub trait PackageResolver: Send + Sync {
    fn resolve(
        &self,
        workspace_root: &Path,
        artifact: &UnderstandingArtifact,
        packet: &ContextPacket,
    ) -> KnowledgeResult<Vec<PackageFact>>;
}

pub trait ManualProvider: Send + Sync {
    fn search(
        &self,
        workspace_root: &Path,
        query: &ManualQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<ManualEvidence>>;
}

pub struct ResolverRegistry {
    resolvers: Vec<Box<dyn PackageResolver>>,
}

pub struct ProviderRegistry {
    providers: Vec<Box<dyn ManualProvider>>,
}

pub struct KnowledgeService {
    resolvers: ResolverRegistry,
    providers: ProviderRegistry,
    manual_limit: usize,
}

pub struct CargoPackageResolver;
pub struct NpmPackageResolver;

pub struct FileCatalogProvider {
    catalog_path: PathBuf,
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

impl Default for ResolverRegistry {
    fn default() -> Self {
        Self {
            resolvers: vec![Box::new(CargoPackageResolver), Box::new(NpmPackageResolver)],
        }
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self {
            providers: vec![Box::new(FileCatalogProvider::default())],
        }
    }
}

impl Default for KnowledgeService {
    fn default() -> Self {
        Self {
            resolvers: ResolverRegistry::default(),
            providers: ProviderRegistry::default(),
            manual_limit: DEFAULT_MANUAL_LIMIT,
        }
    }
}

impl Default for FileCatalogProvider {
    fn default() -> Self {
        Self::new(DEFAULT_CATALOG_PATH)
    }
}

impl FileCatalogProvider {
    pub fn new(path: impl Into<PathBuf>) -> Self {
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

// ── AmbiguityResolver ────────────────────────────────────────────────────────

/// Tries to resolve a `Lookup`-kind ambiguity by querying external sources.
/// Returns `Some(resolution_text)` on success, `None` if it cannot help.
pub trait AmbiguityResolver: Send + Sync {
    fn resolve(&self, ambiguity: &Ambiguity) -> Option<String>;
}

/// Resolves package-version questions by querying the npm registry.
pub struct NpmRegistryResolver {
    client: Client,
}

/// Resolves crate-version questions by querying the crates.io API.
pub struct CratesIoResolver {
    client: Client,
}

impl Default for NpmRegistryResolver {
    fn default() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(5))
                .user_agent("shunt-agent/0.1")
                .build()
                .unwrap_or_default(),
        }
    }
}

impl Default for CratesIoResolver {
    fn default() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(5))
                .user_agent("shunt-agent/0.1")
                .build()
                .unwrap_or_default(),
        }
    }
}

impl AmbiguityResolver for NpmRegistryResolver {
    fn resolve(&self, ambiguity: &Ambiguity) -> Option<String> {
        let candidates = extract_npm_package_names(ambiguity);
        if candidates.is_empty() {
            return None;
        }
        let mut results: Vec<String> = Vec::new();
        for pkg in &candidates {
            if let Some(version) = npm_latest_version(&self.client, pkg) {
                results.push(format!("{pkg}@{version}"));
            }
        }
        if results.is_empty() {
            None
        } else {
            Some(format!("Latest npm versions: {}", results.join(", ")))
        }
    }
}

impl AmbiguityResolver for CratesIoResolver {
    fn resolve(&self, ambiguity: &Ambiguity) -> Option<String> {
        let candidates = extract_crate_names(ambiguity);
        if candidates.is_empty() {
            return None;
        }
        let mut results: Vec<String> = Vec::new();
        for krate in &candidates {
            if let Some(version) = crates_io_latest_version(&self.client, krate) {
                results.push(format!("{krate}@{version}"));
            }
        }
        if results.is_empty() {
            None
        } else {
            Some(format!("Latest crates.io versions: {}", results.join(", ")))
        }
    }
}

// ── Package name extraction ───────────────────────────────────────────────────

/// Extract npm package names from the ambiguity question and options.
/// Recognises `@scope/package` and bare `package-name` tokens.
fn extract_npm_package_names(ambiguity: &Ambiguity) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let text = format!("{} {}", ambiguity.question, ambiguity.options.join(" "));
    // Scoped packages: @scope/name
    let mut remaining = text.as_str();
    while let Some(at) = remaining.find('@') {
        let tail = &remaining[at + 1..];
        let end = tail
            .find(|c: char| c.is_whitespace() || c == ',' || c == ')' || c == '"' || c == '\'')
            .unwrap_or(tail.len());
        let token = &tail[..end];
        if token.contains('/') && !token.is_empty() {
            names.push(format!("@{token}"));
        }
        remaining = &remaining[at + 1..];
    }
    // Bare kebab-case package names from options (likely to be package names)
    for opt in &ambiguity.options {
        let opt = opt.trim();
        if !opt.starts_with('@')
            && opt.contains('-')
            && opt
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
            && opt.len() > 2
        {
            names.push(opt.to_string());
        }
    }
    names.sort();
    names.dedup();
    names
}

/// Extract crate names from the ambiguity text.
fn extract_crate_names(ambiguity: &Ambiguity) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    // Options that look like crate names: snake_case or kebab-case, no slashes
    for opt in &ambiguity.options {
        let opt = opt.trim();
        if !opt.contains('/')
            && (opt.contains('_') || opt.contains('-'))
            && opt
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            && opt.len() > 2
        {
            names.push(opt.to_string());
        }
    }
    names.sort();
    names.dedup();
    names
}

// ── Registry HTTP calls ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct NpmDistTags {
    latest: Option<String>,
}

#[derive(Deserialize)]
struct NpmRegistryResponse {
    #[serde(rename = "dist-tags")]
    dist_tags: Option<NpmDistTags>,
}

fn npm_latest_version(client: &Client, package: &str) -> Option<String> {
    let encoded = package.replace('/', "%2F");
    let url = format!("https://registry.npmjs.org/{encoded}");
    let resp: NpmRegistryResponse = client.get(&url).send().ok()?.json().ok()?;
    resp.dist_tags?.latest
}

#[derive(Deserialize)]
struct CratesIoResponse {
    #[serde(rename = "crate")]
    krate: Option<CratesIoKrate>,
}

#[derive(Deserialize)]
struct CratesIoKrate {
    newest_version: Option<String>,
}

fn crates_io_latest_version(client: &Client, krate: &str) -> Option<String> {
    let url = format!("https://crates.io/api/v1/crates/{krate}");
    let resp: CratesIoResponse = client.get(&url).send().ok()?.json().ok()?;
    resp.krate?.newest_version
}

// ── KnowledgeService ambiguity resolution ────────────────────────────────────

impl KnowledgeService {
    pub fn gather(
        &self,
        workspace_root: &Path,
        artifact: &UnderstandingArtifact,
        packet: &ContextPacket,
    ) -> KnowledgeResult<KnowledgePacket> {
        let package_facts = self.resolvers.resolve(workspace_root, artifact, packet)?;
        if package_facts.is_empty() {
            return Ok(KnowledgePacket {
                package_facts,
                manual_evidence: Vec::new(),
            });
        }

        if !should_fetch_manuals(artifact, packet, &package_facts) {
            return Ok(KnowledgePacket {
                package_facts,
                manual_evidence: Vec::new(),
            });
        }

        let query = ManualQuery {
            original_request: artifact.original_request.clone(),
            interpreted_goal: artifact.interpreted_goal.clone(),
            located_paths: packet
                .primary_candidates
                .iter()
                .map(|candidate| candidate.file.path.clone())
                .collect(),
            requested_topics: packet.query.literals.clone(),
            package_facts: package_facts.clone(),
        };
        let manual_evidence = self
            .providers
            .search(workspace_root, &query, self.manual_limit)?;

        Ok(KnowledgePacket {
            package_facts,
            manual_evidence,
        })
    }

    /// For each `Lookup`-kind open ambiguity, try every registered resolver.
    /// Returns a vec of `(ambiguity_id, resolution_text)` for those successfully resolved.
    pub fn resolve_lookup_ambiguities(
        &self,
        ambiguities: &[&Ambiguity],
        resolvers: &[Box<dyn AmbiguityResolver>],
    ) -> Vec<(String, String)> {
        let mut results = Vec::new();
        for ambiguity in ambiguities {
            for resolver in resolvers {
                if let Some(resolution) = resolver.resolve(ambiguity) {
                    results.push((ambiguity.id.clone(), resolution));
                    break; // first resolver that succeeds wins
                }
            }
        }
        results
    }
}

impl ProviderRegistry {
    pub fn search(
        &self,
        workspace_root: &Path,
        query: &ManualQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<ManualEvidence>> {
        let mut evidence = Vec::new();
        let mut seen = BTreeSet::new();

        for provider in &self.providers {
            for item in provider.search(workspace_root, query, limit)? {
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

impl ResolverRegistry {
    pub fn resolve(
        &self,
        workspace_root: &Path,
        artifact: &UnderstandingArtifact,
        packet: &ContextPacket,
    ) -> KnowledgeResult<Vec<PackageFact>> {
        let mut facts = Vec::new();
        let mut seen = BTreeSet::new();

        for resolver in &self.resolvers {
            for fact in resolver.resolve(workspace_root, artifact, packet)? {
                let key = format!(
                    "{}:{}:{}:{}",
                    fact.ecosystem,
                    fact.name,
                    fact.version.as_deref().unwrap_or(""),
                    fact.manifest_path
                );
                if seen.insert(key) {
                    facts.push(fact);
                }
            }
        }

        facts.sort_by(|left, right| {
            right
                .confidence
                .total_cmp(&left.confidence)
                .then_with(|| left.ecosystem.cmp(&right.ecosystem))
                .then_with(|| left.name.cmp(&right.name))
        });
        Ok(facts)
    }
}

impl PackageResolver for CargoPackageResolver {
    fn resolve(
        &self,
        workspace_root: &Path,
        _artifact: &UnderstandingArtifact,
        packet: &ContextPacket,
    ) -> KnowledgeResult<Vec<PackageFact>> {
        let relevant_manifests = relevant_manifest_files(workspace_root, packet, "Cargo.toml");
        if relevant_manifests.is_empty() {
            return Ok(Vec::new());
        }

        let import_hints = referenced_package_hints(packet);
        let query_terms = query_terms(packet);
        let locked_versions = cargo_lock_versions(workspace_root)?;
        let mut facts = Vec::new();
        for manifest in relevant_manifests {
            let text = fs::read_to_string(workspace_root.join(&manifest))?;
            let value: toml::Value = toml::from_str(&text)?;
            for dependency in cargo_dependencies(&value) {
                let dependency_name = dependency.name.clone();
                let relevance = dependency_relevance(&dependency.name, &import_hints, &query_terms);
                if relevance == 0.0 {
                    continue;
                }
                let locked = locked_versions
                    .get(&dependency.name)
                    .cloned()
                    .unwrap_or_default();
                let (version, version_provenance, confidence) = if locked.len() == 1 {
                    (
                        locked.into_iter().next(),
                        PackageVersionProvenance::ExactLock,
                        0.85 + relevance.min(0.14),
                    )
                } else if dependency.requirement.is_some() {
                    (
                        None,
                        PackageVersionProvenance::ManifestRequirement,
                        0.65 + relevance.min(0.14),
                    )
                } else {
                    (
                        None,
                        PackageVersionProvenance::Unknown,
                        0.45 + relevance.min(0.14),
                    )
                };
                facts.push(PackageFact {
                    ecosystem: "cargo".into(),
                    name: dependency_name.clone(),
                    version,
                    requirement: dependency.requirement,
                    version_provenance,
                    manifest_path: manifest.clone(),
                    evidence: vec![EvidenceRef {
                        kind: EvidenceKind::File,
                        locator: manifest.clone(),
                        summary: package_fact_summary("cargo", &dependency_name, relevance),
                    }],
                    confidence,
                });
            }
        }

        Ok(facts)
    }
}

impl PackageResolver for NpmPackageResolver {
    fn resolve(
        &self,
        workspace_root: &Path,
        _artifact: &UnderstandingArtifact,
        packet: &ContextPacket,
    ) -> KnowledgeResult<Vec<PackageFact>> {
        let manifests = relevant_manifest_files(workspace_root, packet, "package.json");
        if manifests.is_empty() {
            return Ok(Vec::new());
        }

        let import_hints = referenced_package_hints(packet);
        let query_terms = query_terms(packet);
        let mut facts = Vec::new();

        for manifest in manifests {
            let manifest_path = workspace_root.join(&manifest);
            let value: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&manifest_path)?)?;
            let lock_versions =
                npm_lock_versions(manifest_path.parent().unwrap_or(workspace_root))?;
            for dependency in npm_dependencies(&value) {
                let dependency_name = dependency.name.clone();
                let relevance = dependency_relevance(&dependency.name, &import_hints, &query_terms);
                if relevance == 0.0 {
                    continue;
                }

                let version = lock_versions.get(&dependency_name).cloned();
                let version_provenance = if version.is_some() {
                    PackageVersionProvenance::ExactLock
                } else if dependency.requirement.is_some() {
                    PackageVersionProvenance::ManifestRequirement
                } else {
                    PackageVersionProvenance::Unknown
                };
                let base_confidence = match version_provenance {
                    PackageVersionProvenance::ExactLock => 0.86,
                    PackageVersionProvenance::ManifestRequirement => 0.68,
                    PackageVersionProvenance::Unknown => 0.48,
                };

                facts.push(PackageFact {
                    ecosystem: "npm".into(),
                    name: dependency_name.clone(),
                    version,
                    requirement: dependency.requirement,
                    version_provenance,
                    manifest_path: manifest.clone(),
                    evidence: vec![EvidenceRef {
                        kind: EvidenceKind::File,
                        locator: manifest.clone(),
                        summary: package_fact_summary("npm", &dependency_name, relevance),
                    }],
                    confidence: base_confidence + relevance.min(0.12),
                });
            }
        }

        Ok(facts)
    }
}

impl ManualProvider for FileCatalogProvider {
    fn search(
        &self,
        workspace_root: &Path,
        query: &ManualQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<ManualEvidence>> {
        let catalog = self.load_catalog(workspace_root)?;
        if catalog.is_empty() {
            return Ok(Vec::new());
        }

        let mut matches = Vec::new();
        for fact in &query.package_facts {
            for entry in &catalog {
                if entry.ecosystem != fact.ecosystem || entry.package != fact.name {
                    continue;
                }
                let version_status = version_status(fact, entry);
                if version_status == ManualVersionStatus::Mismatch {
                    continue;
                }

                let overlap = token_overlap(query, entry);
                if overlap == 0 {
                    continue;
                }

                let score = version_status_score(version_status) + overlap as f32;
                matches.push((
                    score,
                    ManualEvidence {
                        ecosystem: fact.ecosystem.clone(),
                        package: fact.name.clone(),
                        version: entry.version.clone(),
                        version_status,
                        source: entry.source.clone(),
                        locator: entry.locator.clone(),
                        title: entry.title.clone(),
                        excerpt: clamp_excerpt(&entry.text),
                        relevance_reason: format!(
                            "matched {} request terms for {}",
                            overlap, fact.name
                        ),
                        confidence: score.min(10.0) / 10.0,
                    },
                ));
            }
        }

        matches.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.package.cmp(&right.1.package))
                .then_with(|| left.1.locator.cmp(&right.1.locator))
        });

        let mut seen = BTreeSet::new();
        let mut evidence = Vec::new();
        for (_, item) in matches {
            let key = format!(
                "{}:{}:{}",
                item.package,
                item.version.as_deref().unwrap_or(""),
                item.locator
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

#[derive(Debug)]
struct CargoDependency {
    name: String,
    requirement: Option<String>,
}

fn relevant_manifest_files(
    workspace_root: &Path,
    packet: &ContextPacket,
    manifest_name: &str,
) -> Vec<String> {
    let mut manifests = BTreeSet::new();
    for path in packet
        .primary_candidates
        .iter()
        .chain(packet.supporting_candidates.iter())
        .map(|candidate| candidate.file.path.as_str())
    {
        let path = workspace_root.join(path);
        let mut cursor = path.parent();
        while let Some(parent) = cursor {
            if !parent.starts_with(workspace_root) {
                break;
            }
            let manifest = parent.join(manifest_name);
            if manifest.is_file() {
                manifests.insert(relative_path(workspace_root, &manifest));
                break;
            }
            cursor = parent.parent();
        }
    }
    manifests.into_iter().collect()
}

fn cargo_lock_versions(
    workspace_root: &Path,
) -> KnowledgeResult<BTreeMap<String, BTreeSet<String>>> {
    let path = workspace_root.join("Cargo.lock");
    if !path.is_file() {
        return Ok(BTreeMap::new());
    }

    let value: toml::Value = toml::from_str(&fs::read_to_string(path)?)?;
    let mut versions = BTreeMap::new();
    if let Some(packages) = value.get("package").and_then(|value| value.as_array()) {
        for package in packages {
            let Some(name) = package.get("name").and_then(|value| value.as_str()) else {
                continue;
            };
            let Some(version) = package.get("version").and_then(|value| value.as_str()) else {
                continue;
            };
            versions
                .entry(name.to_string())
                .or_insert_with(BTreeSet::new)
                .insert(version.to_string());
        }
    }
    Ok(versions)
}

fn cargo_dependencies(value: &toml::Value) -> Vec<CargoDependency> {
    let mut dependencies = Vec::new();
    for section in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(table) = value.get(section).and_then(|value| value.as_table()) {
            collect_cargo_dependency_table(table, &mut dependencies);
        }
    }
    dependencies.sort_by(|left, right| left.name.cmp(&right.name));
    dependencies.dedup_by(|left, right| left.name == right.name);
    dependencies
}

fn collect_cargo_dependency_table(
    table: &toml::map::Map<String, toml::Value>,
    dependencies: &mut Vec<CargoDependency>,
) {
    for (name, value) in table {
        let requirement = if let Some(version) = value.as_str() {
            Some(version.to_string())
        } else {
            value
                .get("version")
                .and_then(|version| version.as_str())
                .map(str::to_string)
        };
        dependencies.push(CargoDependency {
            name: name.clone(),
            requirement,
        });
    }
}

fn npm_dependencies(value: &serde_json::Value) -> Vec<CargoDependency> {
    let mut dependencies = Vec::new();
    for key in ["dependencies", "devDependencies", "peerDependencies"] {
        let Some(table) = value.get(key).and_then(|value| value.as_object()) else {
            continue;
        };
        for (name, version) in table {
            dependencies.push(CargoDependency {
                name: name.clone(),
                requirement: version.as_str().map(str::to_string),
            });
        }
    }
    dependencies.sort_by(|left, right| left.name.cmp(&right.name));
    dependencies.dedup_by(|left, right| left.name == right.name);
    dependencies
}

fn npm_lock_versions(workspace_root: &Path) -> KnowledgeResult<BTreeMap<String, String>> {
    let path = workspace_root.join("package-lock.json");
    if !path.is_file() {
        return Ok(BTreeMap::new());
    }

    let value: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let mut versions = BTreeMap::new();

    if let Some(packages) = value.get("packages").and_then(|value| value.as_object()) {
        for (key, package) in packages {
            let Some(version) = package.get("version").and_then(|value| value.as_str()) else {
                continue;
            };
            let Some(name) = key.strip_prefix("node_modules/") else {
                continue;
            };
            versions.insert(name.to_string(), version.to_string());
        }
    }

    if versions.is_empty()
        && let Some(dependencies) = value
            .get("dependencies")
            .and_then(|value| value.as_object())
    {
        for (name, package) in dependencies {
            let Some(version) = package.get("version").and_then(|value| value.as_str()) else {
                continue;
            };
            versions.insert(name.clone(), version.to_string());
        }
    }

    Ok(versions)
}

fn should_fetch_manuals(
    artifact: &UnderstandingArtifact,
    packet: &ContextPacket,
    package_facts: &[PackageFact],
) -> bool {
    if package_facts.is_empty() {
        return false;
    }

    let import_hints = referenced_package_hints(packet);
    if !import_hints.is_empty() {
        return true;
    }

    let lower_request = format!(
        "{} {}",
        artifact.original_request, artifact.interpreted_goal
    )
    .to_ascii_lowercase();
    package_facts.iter().any(|fact| {
        lower_request.contains(&fact.name.to_ascii_lowercase())
            || packet
                .query
                .literals
                .iter()
                .any(|term| normalized_package_key(term) == normalized_package_key(&fact.name))
    })
}

fn query_terms(packet: &ContextPacket) -> BTreeSet<String> {
    packet
        .query
        .literals
        .iter()
        .map(|term| term.to_ascii_lowercase())
        .collect()
}

fn dependency_relevance(
    package_name: &str,
    import_hints: &BTreeSet<String>,
    query_terms: &BTreeSet<String>,
) -> f32 {
    let normalized = normalized_package_key(package_name);
    if import_hints
        .iter()
        .any(|hint| normalized_package_key(hint) == normalized)
    {
        return 0.14;
    }

    if query_terms.iter().any(|term| {
        normalized.contains(term) || term.contains(&normalized) || package_name.contains(term)
    }) {
        return 0.08;
    }

    0.0
}

fn package_fact_summary(ecosystem: &str, package_name: &str, relevance: f32) -> String {
    if relevance >= 0.14 {
        format!(
            "package fact derived from {ecosystem} dependency and localized import/reference for {package_name}"
        )
    } else if relevance > 0.0 {
        format!(
            "package fact derived from {ecosystem} dependency and request/query overlap for {package_name}"
        )
    } else {
        format!("package fact derived from {ecosystem} dependency manifest")
    }
}

fn referenced_package_hints(packet: &ContextPacket) -> BTreeSet<String> {
    let mut hints = BTreeSet::new();
    for candidate in packet
        .primary_candidates
        .iter()
        .chain(packet.supporting_candidates.iter())
    {
        for snippet in &candidate.snippets {
            collect_import_hints_from_text(&snippet.text, &mut hints);
        }
    }
    hints
}

fn collect_import_hints_from_text(text: &str, hints: &mut BTreeSet<String>) {
    for spec in extract_quoted_module_specs(text) {
        if let Some(package) = normalize_module_spec(&spec) {
            hints.insert(package);
        }
    }

    for line in text.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("use ")
            && let Some(package) = rest.split("::").next()
        {
            let package = package.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '_');
            if !package.is_empty() {
                hints.insert(package.to_string());
            }
        }
    }
}

fn extract_quoted_module_specs(text: &str) -> Vec<String> {
    let mut specs = Vec::new();
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let ch = bytes[index] as char;
        if ch == '\'' || ch == '"' {
            let quote = ch;
            let start = index + 1;
            let mut end = start;
            while end < bytes.len() {
                if bytes[end] as char == quote {
                    break;
                }
                end += 1;
            }
            if end < bytes.len() {
                let spec = &text[start..end];
                if looks_like_module_spec(spec) {
                    specs.push(spec.to_string());
                }
                index = end;
            }
        }
        index += 1;
    }
    specs
}

fn looks_like_module_spec(spec: &str) -> bool {
    spec.contains('/')
        || spec.contains('-')
        || spec.starts_with('@')
        || spec
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

fn normalize_module_spec(spec: &str) -> Option<String> {
    if spec.starts_with('.') || spec.starts_with('/') {
        return None;
    }
    if let Some(rest) = spec.strip_prefix('@') {
        let mut segments = rest.split('/');
        let scope = segments.next()?;
        let name = segments.next()?;
        return Some(format!("@{scope}/{name}"));
    }
    spec.split('/').next().map(str::to_string)
}

fn normalized_package_key(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn version_status(fact: &PackageFact, entry: &ManualCatalogEntry) -> ManualVersionStatus {
    match (fact.version.as_deref(), entry.version.as_deref()) {
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

fn token_overlap(query: &ManualQuery, entry: &ManualCatalogEntry) -> usize {
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

fn relative_path(workspace_root: &Path, path: &Path) -> String {
    path.strip_prefix(workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use shunt_core::{ApprovalState, ArtifactId, CandidateFile, TaskId, UnderstandingArtifact};
    use shunt_localize::{
        ContextPacket, RankedCandidate, RetrievalBackend, SearchIntent, SearchQuery,
    };
    use time::macros::datetime;

    use super::{
        CargoPackageResolver, FileCatalogProvider, ManualProvider, NpmPackageResolver,
        PackageResolver, ResolverRegistry,
    };

    fn artifact() -> UnderstandingArtifact {
        UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: TaskId("task-1".into()),
            original_request: "fix ratatui layout rendering".into(),
            interpreted_goal: "repair ratatui layout rendering".into(),
            success_criteria: vec![],
            constraints: vec![],
            target_scope: vec!["src/main.rs".into()],
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
            created_at: datetime!(2026-06-11 12:00 UTC),
            updated_at: datetime!(2026-06-11 12:00 UTC),
        }
    }

    fn packet(path: &str) -> ContextPacket {
        ContextPacket {
            backend: RetrievalBackend::Lexical,
            query: SearchQuery {
                intent: SearchIntent::Code,
                literals: vec!["ratatui".into(), "layout".into()],
                repo_terms: vec!["ratatui".into(), "layout".into()],
                regexes: vec![],
                symbol_guesses: vec![],
            },
            primary_candidates: vec![RankedCandidate {
                file: CandidateFile {
                    path: path.into(),
                    summary: "candidate".into(),
                },
                role: shunt_localize::CandidateRole::Implementation,
                score: 1.0,
                reasons: vec![],
                snippets: vec![],
            }],
            supporting_candidates: vec![],
        }
    }

    #[test]
    fn cargo_resolver_extracts_exact_dependency_versions() {
        let root =
            std::env::temp_dir().join(format!("shunt-knowledge-cargo-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\nratatui = \"0.29\"\ncrossterm = \"0.28\"\n",
        )
        .unwrap();
        fs::write(
            root.join("Cargo.lock"),
            "[[package]]\nname = \"ratatui\"\nversion = \"0.29.0\"\n\n[[package]]\nname = \"crossterm\"\nversion = \"0.28.1\"\n",
        )
        .unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();

        let facts = CargoPackageResolver
            .resolve(&root, &artifact(), &packet("src/main.rs"))
            .unwrap();

        assert_eq!(facts.len(), 1);
        assert!(
            facts.iter().any(|fact| {
                fact.name == "ratatui" && fact.version.as_deref() == Some("0.29.0")
            })
        );
    }

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

        let facts = ResolverRegistry::default()
            .resolve(&root, &artifact(), &packet("src/main.rs"))
            .unwrap_or_default();
        let query = shunt_core::ManualQuery {
            original_request: "fix ratatui layout rendering".into(),
            interpreted_goal: "repair ratatui layout rendering".into(),
            located_paths: vec!["src/main.rs".into()],
            requested_topics: vec!["layout".into()],
            package_facts: vec![shunt_core::PackageFact {
                ecosystem: "cargo".into(),
                name: "ratatui".into(),
                version: Some("0.29.0".into()),
                requirement: Some("0.29".into()),
                version_provenance: shunt_core::PackageVersionProvenance::ExactLock,
                manifest_path: "Cargo.toml".into(),
                evidence: vec![],
                confidence: 0.95,
            }],
        };

        let results = FileCatalogProvider::default()
            .search(&root, &query, 2)
            .unwrap();

        assert!(facts.is_empty());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].locator, "ratatui/layout");
        assert_eq!(
            results[0].version_status,
            shunt_core::ManualVersionStatus::Exact
        );
    }

    #[test]
    fn npm_resolver_extracts_only_imported_dependency_versions() {
        let root = std::env::temp_dir().join(format!("shunt-knowledge-npm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("package.json"),
            serde_json::json!({
                "name": "demo-node",
                "version": "1.0.0",
                "dependencies": {
                    "lodash": "^4.17.21",
                    "chalk": "^5.3.0"
                }
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            root.join("package-lock.json"),
            serde_json::json!({
                "name": "demo-node",
                "lockfileVersion": 3,
                "packages": {
                    "": {"name": "demo-node", "version": "1.0.0"},
                    "node_modules/lodash": {"version": "4.17.21"},
                    "node_modules/chalk": {"version": "5.3.0"}
                }
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            root.join("src/index.js"),
            "import chunk from 'lodash/chunk';\nexport const pick = (items) => chunk(items, 2);\n",
        )
        .unwrap();

        let packet = ContextPacket {
            backend: RetrievalBackend::Lexical,
            query: SearchQuery {
                intent: SearchIntent::Code,
                literals: vec!["lodash".into(), "chunk".into()],
                repo_terms: vec!["lodash".into(), "chunk".into()],
                regexes: vec![],
                symbol_guesses: vec![],
            },
            primary_candidates: vec![RankedCandidate {
                file: CandidateFile {
                    path: "src/index.js".into(),
                    summary: "candidate".into(),
                },
                role: shunt_localize::CandidateRole::Implementation,
                score: 1.0,
                reasons: vec![],
                snippets: vec![shunt_localize::CandidateSnippet {
                    path: "src/index.js".into(),
                    start_line: 1,
                    end_line: 2,
                    enclosing_symbol: None,
                    reason: "import hit".into(),
                    text: "import chunk from 'lodash/chunk';\nexport const pick = (items) => chunk(items, 2);".into(),
                }],
            }],
            supporting_candidates: vec![],
        };

        let facts = NpmPackageResolver
            .resolve(&root, &artifact(), &packet)
            .unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].name, "lodash");
        assert_eq!(facts[0].version.as_deref(), Some("4.17.21"));
        assert!(facts[0].evidence[0].summary.contains("import/reference"));
    }
}
