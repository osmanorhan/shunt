use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use semver::{Version, VersionReq};
use serde::Deserialize;
use shunt_core::{ManualEvidence, ManualQuery, ManualVersionStatus, PackageFact};

use crate::KnowledgeResult;

const DEFAULT_CATALOG_PATH: &str = ".shunt/manuals/catalog.json";
const MAX_EXCERPT_CHARS: usize = 420;

pub(crate) trait ManualProvider: Send + Sync {
    fn search(
        &self,
        workspace_root: &Path,
        query: &ManualQuery,
        limit: usize,
    ) -> KnowledgeResult<Vec<ManualEvidence>>;
}

pub(crate) struct ManualProviderRegistry {
    providers: Vec<Box<dyn ManualProvider>>,
}

struct FileCatalogProvider {
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

impl Default for ManualProviderRegistry {
    fn default() -> Self {
        Self {
            providers: vec![Box::new(FileCatalogProvider::default())],
        }
    }
}

impl Default for FileCatalogProvider {
    fn default() -> Self {
        Self::new(DEFAULT_CATALOG_PATH)
    }
}

impl FileCatalogProvider {
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

impl ManualProviderRegistry {
    pub(crate) fn search(
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

#[cfg(test)]
mod tests {
    use std::fs;

    use shunt_core::{ManualQuery, PackageFact, PackageVersionProvenance};

    use super::{FileCatalogProvider, ManualProvider};

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

        let query = ManualQuery {
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
        };

        let results = FileCatalogProvider::default()
            .search(&root, &query, 2)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].locator, "ratatui/layout");
        assert_eq!(
            results[0].version_status,
            shunt_core::ManualVersionStatus::Exact
        );
    }
}
