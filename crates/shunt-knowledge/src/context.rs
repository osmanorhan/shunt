use std::collections::BTreeMap;

use shunt_core::{
    ManualEvidence, ManualVersionStatus, PackageFact, PackageVersionProvenance,
    UnderstandingArtifact,
};

const MAX_PACKAGES: usize = 3;
const MAX_MANUALS_PER_PACKAGE: usize = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeContext {
    pub packages: Vec<KnowledgePackageContext>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgePackageContext {
    pub ecosystem: String,
    pub name: String,
    pub version: Option<String>,
    pub requirement: Option<String>,
    pub version_provenance: PackageVersionProvenance,
    pub confidence: f32,
    pub manuals: Vec<KnowledgeManualContext>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeManualContext {
    pub version_status: ManualVersionStatus,
    pub source: String,
    pub locator: String,
    pub title: Option<String>,
    pub excerpt: String,
    pub confidence: f32,
}

impl KnowledgeContext {
    pub fn from_artifact(artifact: &UnderstandingArtifact) -> Self {
        Self::from_evidence(&artifact.package_facts, &artifact.manual_evidence)
    }

    pub fn from_evidence(
        package_facts: &[PackageFact],
        manual_evidence: &[ManualEvidence],
    ) -> Self {
        let manuals_by_package = group_manuals_by_package(manual_evidence);
        let mut packages = package_facts
            .iter()
            .map(|fact| KnowledgePackageContext {
                ecosystem: fact.ecosystem.clone(),
                name: fact.name.clone(),
                version: fact.version.clone(),
                requirement: fact.requirement.clone(),
                version_provenance: fact.version_provenance.clone(),
                confidence: fact.confidence,
                manuals: manuals_by_package
                    .get(&(fact.ecosystem.clone(), fact.name.clone()))
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect::<Vec<_>>();

        let existing = packages
            .iter()
            .map(|package| (package.ecosystem.clone(), package.name.clone()))
            .collect::<std::collections::BTreeSet<_>>();
        for ((ecosystem, name), manuals) in &manuals_by_package {
            if existing.contains(&(ecosystem.clone(), name.clone())) {
                continue;
            }
            packages.push(KnowledgePackageContext {
                ecosystem: ecosystem.clone(),
                name: name.clone(),
                version: manuals
                    .first()
                    .and_then(|manual| manual.locator.split('@').nth(1))
                    .map(str::to_string),
                requirement: None,
                version_provenance: PackageVersionProvenance::Unknown,
                confidence: manuals
                    .iter()
                    .map(|manual| manual.confidence)
                    .fold(0.35, f32::max),
                manuals: manuals.clone(),
            });
        }

        packages.sort_by(|left, right| {
            right
                .manuals
                .len()
                .cmp(&left.manuals.len())
                .then_with(|| right.confidence.total_cmp(&left.confidence))
                .then_with(|| left.ecosystem.cmp(&right.ecosystem))
                .then_with(|| left.name.cmp(&right.name))
        });
        packages.truncate(MAX_PACKAGES);
        Self { packages }
    }

    pub fn is_empty(&self) -> bool {
        self.packages.is_empty()
    }

    pub fn render_agent_context(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        let mut out = String::from("KNOWLEDGE CONTEXT:\n");
        out.push_str(
            "Follow these package-specific facts when editing code. Prefer the listed patterns over generic examples.\n",
        );
        for package in &self.packages {
            out.push_str("- ");
            out.push_str(&package.package_line());
            out.push('\n');
            for manual in &package.manuals {
                out.push_str("  - ");
                out.push_str(&manual.manual_line());
                out.push('\n');
            }
        }
        out.trim_end().to_string()
    }
}

impl KnowledgePackageContext {
    fn package_line(&self) -> String {
        let version = self.version.as_deref().unwrap_or("unknown");
        let requirement = self
            .requirement
            .as_deref()
            .map(|requirement| format!(", requirement {requirement}"))
            .unwrap_or_default();
        format!(
            "{}:{}@{} ({:?}{})",
            self.ecosystem, self.name, version, self.version_provenance, requirement
        )
    }
}

impl KnowledgeManualContext {
    fn manual_line(&self) -> String {
        let title = self
            .title
            .as_deref()
            .map(|title| format!("{title}: "))
            .unwrap_or_default();
        format!(
            "Prefer this {} guidance: {}{}",
            version_status_label(self.version_status),
            title,
            self.excerpt
        )
    }
}

fn version_status_label(status: ManualVersionStatus) -> &'static str {
    match status {
        ManualVersionStatus::Exact => "exact",
        ManualVersionStatus::CompatibleRange => "compatible",
        ManualVersionStatus::Unversioned => "unversioned",
        ManualVersionStatus::Mismatch => "mismatched",
    }
}

fn group_manuals_by_package(
    manual_evidence: &[ManualEvidence],
) -> BTreeMap<(String, String), Vec<KnowledgeManualContext>> {
    let mut grouped: BTreeMap<(String, String), Vec<ManualEvidence>> = BTreeMap::new();
    for manual in manual_evidence {
        grouped
            .entry((manual.ecosystem.clone(), manual.package.clone()))
            .or_default()
            .push(manual.clone());
    }

    grouped
        .into_iter()
        .map(|(package, manuals)| (package, select_manuals(manuals)))
        .collect()
}

fn select_manuals(manuals: Vec<ManualEvidence>) -> Vec<KnowledgeManualContext> {
    let mut selected = manuals
        .iter()
        .filter(|manual| {
            matches!(
                manual.version_status,
                ManualVersionStatus::Exact | ManualVersionStatus::CompatibleRange
            )
        })
        .cloned()
        .collect::<Vec<_>>();

    if selected.is_empty() {
        selected = manuals
            .into_iter()
            .filter(|manual| manual.version_status == ManualVersionStatus::Unversioned)
            .collect();
    }

    selected.sort_by(|left, right| {
        version_status_rank(right.version_status)
            .cmp(&version_status_rank(left.version_status))
            .then_with(|| right.confidence.total_cmp(&left.confidence))
            .then_with(|| left.locator.cmp(&right.locator))
    });
    selected.truncate(MAX_MANUALS_PER_PACKAGE);
    selected
        .into_iter()
        .map(|manual| KnowledgeManualContext {
            version_status: manual.version_status,
            source: manual.source,
            locator: manual.locator,
            title: manual.title,
            excerpt: manual.excerpt,
            confidence: manual.confidence,
        })
        .collect()
}

fn version_status_rank(status: ManualVersionStatus) -> u8 {
    match status {
        ManualVersionStatus::Exact => 3,
        ManualVersionStatus::CompatibleRange => 2,
        ManualVersionStatus::Unversioned => 1,
        ManualVersionStatus::Mismatch => 0,
    }
}

#[cfg(test)]
mod tests {
    use shunt_core::{
        ApprovalState, ArtifactId, ManualEvidence, ManualVersionStatus, PackageFact,
        PackageVersionProvenance, TaskId, UnderstandingArtifact,
    };
    use time::macros::datetime;

    use super::KnowledgeContext;

    #[test]
    fn renders_version_aware_context() {
        let artifact = artifact(
            vec![PackageFact {
                ecosystem: "cargo".into(),
                name: "ratatui".into(),
                version: Some("0.29.0".into()),
                requirement: Some("0.29".into()),
                version_provenance: PackageVersionProvenance::ExactLock,
                manifest_path: "Cargo.toml".into(),
                evidence: vec![],
                confidence: 0.95,
            }],
            vec![ManualEvidence {
                ecosystem: "cargo".into(),
                package: "ratatui".into(),
                version: Some("0.29.0".into()),
                version_status: ManualVersionStatus::Exact,
                source: "deepwiki".into(),
                locator: "ratatui/layout".into(),
                title: Some("Layout".into()),
                excerpt: "Use the Layout API for splits.".into(),
                relevance_reason: "matched layout".into(),
                confidence: 0.9,
            }],
        );

        let rendered = KnowledgeContext::from_artifact(&artifact).render_agent_context();

        assert!(rendered.contains("KNOWLEDGE CONTEXT"));
        assert!(rendered.contains("cargo:ratatui@0.29.0"));
        assert!(rendered.contains("Prefer this exact guidance"));
        assert!(rendered.contains("Use the Layout API"));
    }

    #[test]
    fn prefers_versioned_manuals_over_unversioned() {
        let context = KnowledgeContext::from_evidence(
            &[PackageFact {
                ecosystem: "npm".into(),
                name: "react".into(),
                version: Some("18.2.0".into()),
                requirement: Some("^18".into()),
                version_provenance: PackageVersionProvenance::ExactLock,
                manifest_path: "package.json".into(),
                evidence: vec![],
                confidence: 0.9,
            }],
            &[
                ManualEvidence {
                    ecosystem: "npm".into(),
                    package: "react".into(),
                    version: None,
                    version_status: ManualVersionStatus::Unversioned,
                    source: "docs".into(),
                    locator: "react/latest".into(),
                    title: None,
                    excerpt: "generic docs".into(),
                    relevance_reason: "fallback".into(),
                    confidence: 1.0,
                },
                ManualEvidence {
                    ecosystem: "npm".into(),
                    package: "react".into(),
                    version: Some("18.2.0".into()),
                    version_status: ManualVersionStatus::Exact,
                    source: "docs".into(),
                    locator: "react/18".into(),
                    title: None,
                    excerpt: "versioned docs".into(),
                    relevance_reason: "exact".into(),
                    confidence: 0.8,
                },
            ],
        );

        let rendered = context.render_agent_context();

        assert!(rendered.contains("versioned docs"));
        assert!(!rendered.contains("generic docs"));
    }

    #[test]
    fn renders_manual_only_packages_without_local_package_facts() {
        let context = KnowledgeContext::from_evidence(
            &[],
            &[ManualEvidence {
                ecosystem: "npm".into(),
                package: "react".into(),
                version: None,
                version_status: ManualVersionStatus::Unversioned,
                source: "public-search".into(),
                locator: "https://react.dev/learn/installation".into(),
                title: Some("Installation".into()),
                excerpt: "Use create-vite or a framework such as Next.js for new React apps."
                    .into(),
                relevance_reason: "package react was named in the request".into(),
                confidence: 0.82,
            }],
        );

        let rendered = context.render_agent_context();

        assert!(rendered.contains("npm:react@unknown"));
        assert!(rendered.contains("create-vite"));
    }

    #[test]
    fn omits_mismatched_manuals_from_rendered_context() {
        let context = KnowledgeContext::from_evidence(
            &[PackageFact {
                ecosystem: "cargo".into(),
                name: "ratatui".into(),
                version: Some("0.29.0".into()),
                requirement: Some("0.29".into()),
                version_provenance: PackageVersionProvenance::ExactLock,
                manifest_path: "Cargo.toml".into(),
                evidence: vec![],
                confidence: 0.9,
            }],
            &[
                ManualEvidence {
                    ecosystem: "cargo".into(),
                    package: "ratatui".into(),
                    version: Some("0.28.0".into()),
                    version_status: ManualVersionStatus::Mismatch,
                    source: "docs".into(),
                    locator: "ratatui/0.28/layout".into(),
                    title: Some("Old Layout".into()),
                    excerpt: "Do not use this old layout pattern.".into(),
                    relevance_reason: "mismatch".into(),
                    confidence: 0.95,
                },
                ManualEvidence {
                    ecosystem: "cargo".into(),
                    package: "ratatui".into(),
                    version: Some("0.29.0".into()),
                    version_status: ManualVersionStatus::Exact,
                    source: "docs".into(),
                    locator: "ratatui/0.29/layout".into(),
                    title: Some("Layout".into()),
                    excerpt: "Use Layout::vertical([Constraint::Length(3), Constraint::Min(0)]).split(area).".into(),
                    relevance_reason: "exact".into(),
                    confidence: 0.9,
                },
            ],
        );

        let rendered = context.render_agent_context();

        assert!(rendered.contains("Layout::vertical"));
        assert!(!rendered.contains("Old Layout"));
    }

    fn artifact(
        package_facts: Vec<PackageFact>,
        manual_evidence: Vec<ManualEvidence>,
    ) -> UnderstandingArtifact {
        UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: TaskId("task-1".into()),
            original_request: "fix layout".into(),
            interpreted_goal: "fix layout".into(),
            success_criteria: vec![],
            constraints: vec![],
            target_scope: vec![],
            work_contract: Default::default(),
            evidence: vec![],
            candidate_files: vec![],
            package_facts,
            manual_evidence,
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
}
