mod context;
mod inventory;
mod lookup;
mod manuals;

use std::path::Path;

use inventory::{PackageResolverRegistry, should_fetch_manuals};
use lookup::{AmbiguityResolver, default_ambiguity_resolvers};
use manuals::ManualProviderRegistry;
use shunt_core::{Ambiguity, ManualEvidence, ManualQuery, PackageFact, UnderstandingArtifact};
use shunt_localize::ContextPacket;
use thiserror::Error;

pub use context::{KnowledgeContext, KnowledgeManualContext, KnowledgePackageContext};

const DEFAULT_MANUAL_LIMIT: usize = 3;

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

pub struct KnowledgeService {
    inventory: PackageResolverRegistry,
    manuals: ManualProviderRegistry,
    ambiguity_resolvers: Vec<Box<dyn AmbiguityResolver>>,
    manual_limit: usize,
}

impl Default for KnowledgeService {
    fn default() -> Self {
        Self {
            inventory: PackageResolverRegistry::default(),
            manuals: ManualProviderRegistry::default(),
            ambiguity_resolvers: default_ambiguity_resolvers(),
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
    ) -> KnowledgeResult<KnowledgePacket> {
        let package_facts = self.inventory.resolve(workspace_root, packet)?;
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
            .manuals
            .search(workspace_root, &query, self.manual_limit)?;

        Ok(KnowledgePacket {
            package_facts,
            manual_evidence,
        })
    }

    pub fn resolve_lookup_ambiguities(&self, ambiguities: &[&Ambiguity]) -> Vec<(String, String)> {
        let mut results = Vec::new();
        for ambiguity in ambiguities {
            for resolver in &self.ambiguity_resolvers {
                if let Some(resolution) = resolver.resolve(ambiguity) {
                    results.push((ambiguity.id.clone(), resolution));
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
