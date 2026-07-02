use shunt_core::{ManualEvidence, ManualVersionStatus, PackageFact};
use time::OffsetDateTime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnowledgeFetchStatus {
    Static,
    Live,
    Cached,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeSourceRef {
    pub source: String,
    pub locator: String,
    pub fetched_at: Option<OffsetDateTime>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeEvidence {
    pub ecosystem: String,
    pub package: String,
    pub version: Option<String>,
    pub version_status: ManualVersionStatus,
    pub source: KnowledgeSourceRef,
    pub title: Option<String>,
    pub excerpt: String,
    pub relevance_reason: String,
    pub confidence: f32,
    pub fetch_status: KnowledgeFetchStatus,
}

impl KnowledgeEvidence {
    pub(crate) fn into_manual_evidence(self) -> ManualEvidence {
        ManualEvidence {
            ecosystem: self.ecosystem,
            package: self.package,
            version: self.version,
            version_status: self.version_status,
            source: self.source.source,
            locator: self.source.locator,
            title: self.title,
            excerpt: self.excerpt,
            relevance_reason: self.relevance_reason,
            confidence: self.confidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeResearchRequest {
    pub summary: String,
    pub package_hints: Vec<String>,
    pub ecosystem_hints: Vec<String>,
    pub search_queries: Vec<String>,
    pub source_hints: Vec<String>,
    pub freshness_required: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeQuery {
    pub original_request: String,
    pub interpreted_goal: String,
    pub located_paths: Vec<String>,
    pub requested_topics: Vec<String>,
    pub package_facts: Vec<PackageFact>,
    pub research_requests: Vec<KnowledgeResearchRequest>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LookupResolution {
    pub ambiguity_id: String,
    pub resolution: String,
    pub evidence: ManualEvidence,
}
