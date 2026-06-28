use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use shunt_core::{
    EvidenceKind, EvidenceRef, PackageFact, PackageVersionProvenance, UnderstandingArtifact,
};
use shunt_localize::ContextPacket;

use crate::KnowledgeResult;

pub(crate) trait PackageResolver: Send + Sync {
    fn resolve(
        &self,
        workspace_root: &Path,
        packet: &ContextPacket,
    ) -> KnowledgeResult<Vec<PackageFact>>;
}

pub(crate) struct PackageResolverRegistry {
    resolvers: Vec<Box<dyn PackageResolver>>,
}

struct CargoPackageResolver;
struct NpmPackageResolver;

#[derive(Debug)]
struct CargoDependency {
    name: String,
    requirement: Option<String>,
}

impl Default for PackageResolverRegistry {
    fn default() -> Self {
        Self {
            resolvers: vec![Box::new(CargoPackageResolver), Box::new(NpmPackageResolver)],
        }
    }
}

impl PackageResolverRegistry {
    pub(crate) fn resolve(
        &self,
        workspace_root: &Path,
        packet: &ContextPacket,
    ) -> KnowledgeResult<Vec<PackageFact>> {
        let mut facts = Vec::new();
        let mut seen = BTreeSet::new();

        for resolver in &self.resolvers {
            for fact in resolver.resolve(workspace_root, packet)? {
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

pub(crate) fn should_fetch_manuals(
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

fn relative_path(workspace_root: &Path, path: &Path) -> String {
    path.strip_prefix(workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use shunt_core::CandidateFile;
    use shunt_localize::{
        ContextPacket, RankedCandidate, RetrievalBackend, SearchIntent, SearchQuery,
    };

    use super::{
        CargoPackageResolver, NpmPackageResolver, PackageResolver, PackageResolverRegistry,
    };

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
            .resolve(&root, &packet("src/main.rs"))
            .unwrap();

        assert_eq!(facts.len(), 1);
        assert!(
            facts.iter().any(|fact| {
                fact.name == "ratatui" && fact.version.as_deref() == Some("0.29.0")
            })
        );
    }

    #[test]
    fn registry_returns_no_facts_without_relevant_package_overlap() {
        let root = std::env::temp_dir().join(format!(
            "shunt-knowledge-inventory-empty-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\ncrossterm = \"0.28\"\n",
        )
        .unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();

        let facts = PackageResolverRegistry::default()
            .resolve(&root, &packet("src/main.rs"))
            .unwrap_or_default();

        assert!(facts.is_empty());
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

        let facts = NpmPackageResolver.resolve(&root, &packet).unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].name, "lodash");
        assert_eq!(facts[0].version.as_deref(), Some("4.17.21"));
        assert!(facts[0].evidence[0].summary.contains("import/reference"));
    }
}
