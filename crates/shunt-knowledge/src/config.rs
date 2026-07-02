use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::KnowledgeResult;

const DEFAULT_CONFIG_PATH: &str = ".shunt/knowledge.toml";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum KnowledgeSourceKind {
    Catalog,
    RegistryMetadata,
    Deepwiki,
    DocsRs,
    RepositoryReadme,
    PublicSearch,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(crate) struct KnowledgeConfig {
    pub external_fetch: bool,
    pub sources: Vec<KnowledgeSourceKind>,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            external_fetch: true,
            sources: vec![
                KnowledgeSourceKind::Catalog,
                KnowledgeSourceKind::RegistryMetadata,
                KnowledgeSourceKind::Deepwiki,
                KnowledgeSourceKind::DocsRs,
                KnowledgeSourceKind::RepositoryReadme,
                KnowledgeSourceKind::PublicSearch,
            ],
        }
    }
}

impl KnowledgeConfig {
    pub(crate) fn load(workspace_root: &Path) -> KnowledgeResult<Self> {
        let path = workspace_root.join(DEFAULT_CONFIG_PATH);
        if !path.is_file() {
            return Ok(Self::default());
        }
        Ok(toml::from_str(&fs::read_to_string(path)?)?)
    }

    pub(crate) fn source_enabled(&self, kind: KnowledgeSourceKind) -> bool {
        self.sources.contains(&kind)
            && match kind {
                KnowledgeSourceKind::Catalog => true,
                KnowledgeSourceKind::RegistryMetadata
                | KnowledgeSourceKind::Deepwiki
                | KnowledgeSourceKind::DocsRs
                | KnowledgeSourceKind::RepositoryReadme
                | KnowledgeSourceKind::PublicSearch => self.external_fetch,
            }
    }
}
