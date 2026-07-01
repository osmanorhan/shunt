mod config;
mod context;
mod inventory;
mod lookup;
mod manuals;
mod types;

use std::path::Path;

use inventory::{PackageResolverRegistry, should_fetch_manuals};
use lookup::{LookupSource, default_lookup_sources};
use manuals::KnowledgeSourceRegistry;
use shunt_core::{Ambiguity, ManualEvidence, PackageFact, UnderstandingArtifact};
use shunt_localize::ContextPacket;
use thiserror::Error;

pub use context::{KnowledgeContext, KnowledgeManualContext, KnowledgePackageContext};
pub use types::{
    KnowledgeEvidence, KnowledgeFetchStatus, KnowledgeQuery, KnowledgeResearchRequest,
    KnowledgeSourceRef, LookupResolution,
};

const DEFAULT_MANUAL_LIMIT: usize = 3;

#[derive(Debug, Error)]
pub enum KnowledgeError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("toml error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("configuration error: {0}")]
    Configuration(String),
}

pub type KnowledgeResult<T> = Result<T, KnowledgeError>;

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgePacket {
    pub package_facts: Vec<PackageFact>,
    pub manual_evidence: Vec<ManualEvidence>,
}

pub struct KnowledgeService {
    inventory: PackageResolverRegistry,
    manuals: KnowledgeSourceRegistry,
    lookup_sources: Vec<Box<dyn LookupSource>>,
    manual_limit: usize,
}

impl Default for KnowledgeService {
    fn default() -> Self {
        Self {
            inventory: PackageResolverRegistry::default(),
            manuals: KnowledgeSourceRegistry::default(),
            lookup_sources: default_lookup_sources(),
            manual_limit: DEFAULT_MANUAL_LIMIT,
        }
    }
}

impl KnowledgeService {
    pub fn gather(
        &self,
        workspace_root: &Path,
        artifact: &UnderstandingArtifact,
        packet: &ContextPacket,
        research_requests: &[KnowledgeResearchRequest],
    ) -> KnowledgeResult<KnowledgePacket> {
        let package_facts = self.inventory.resolve(workspace_root, packet)?;
        if package_facts.is_empty() && research_requests.is_empty() {
            return Ok(KnowledgePacket {
                package_facts,
                manual_evidence: Vec::new(),
            });
        }

        if !should_fetch_manuals(artifact, packet, &package_facts) && research_requests.is_empty() {
            return Ok(KnowledgePacket {
                package_facts,
                manual_evidence: Vec::new(),
            });
        }

        let query = KnowledgeQuery {
            original_request: artifact.original_request.clone(),
            interpreted_goal: artifact.interpreted_goal.clone(),
            located_paths: packet
                .primary_candidates
                .iter()
                .map(|candidate| candidate.file.path.clone())
                .collect(),
            requested_topics: packet.query.literals.clone(),
            package_facts: package_facts.clone(),
            research_requests: research_requests.to_vec(),
        };
        let manual_evidence = self
            .manuals
            .search(workspace_root, &query, self.manual_limit)?;

        Ok(KnowledgePacket {
            package_facts,
            manual_evidence,
        })
    }

    /// Lightweight on-demand lookup for a free-text query (a package name, API, or
    /// topic). Reuses the manual sources without requiring a full
    /// `UnderstandingArtifact`/`ContextPacket`, so the agent loop can call it mid-task.
    /// Returns a human-readable evidence summary (empty string if nothing was found).
    pub fn query(&self, workspace_root: &Path, query: &str) -> KnowledgeResult<String> {
        let query = query.trim();
        let knowledge_query = KnowledgeQuery {
            original_request: query.to_string(),
            interpreted_goal: query.to_string(),
            located_paths: Vec::new(),
            requested_topics: vec![query.to_string()],
            package_facts: Vec::new(),
            research_requests: vec![KnowledgeResearchRequest {
                summary: query.to_string(),
                package_hints: Vec::new(),
                ecosystem_hints: Vec::new(),
                search_queries: vec![query.to_string()],
                source_hints: Vec::new(),
                freshness_required: true,
            }],
        };
        let evidence = self
            .manuals
            .search(workspace_root, &knowledge_query, self.manual_limit)?;
        Ok(render_manual_evidence(&evidence))
    }

    pub fn resolve_lookup_ambiguities(&self, ambiguities: &[&Ambiguity]) -> Vec<LookupResolution> {
        let mut results = Vec::new();
        for ambiguity in ambiguities {
            for source in &self.lookup_sources {
                if let Some(resolution) = source.resolve(ambiguity) {
                    results.push(resolution);
                    break;
                }
            }
        }
        results
    }

    pub fn agent_context(&self, artifact: &UnderstandingArtifact) -> String {
        KnowledgeContext::from_artifact(artifact).render_agent_context()
    }
}

/// Render manual evidence into a compact, model-readable summary.
fn render_manual_evidence(evidence: &[ManualEvidence]) -> String {
    if evidence.is_empty() {
        return "No external evidence found for that query.".to_string();
    }
    let mut out = String::from("Knowledge evidence:\n");
    for item in evidence {
        let version = item
            .version
            .as_deref()
            .map(|v| format!(" {v}"))
            .unwrap_or_default();
        let title = item.title.as_deref().unwrap_or(&item.package);
        out.push_str(&format!(
            "\n• {pkg}{version} ({ecosystem}) — {title}\n  {excerpt}\n  source: {locator}\n",
            pkg = item.package,
            ecosystem = item.ecosystem,
            excerpt = item.excerpt.trim(),
            locator = item.locator,
        ));
    }
    out
}
