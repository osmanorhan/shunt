use std::time::Duration;

use reqwest::blocking::Client;
use serde::Deserialize;
use shunt_core::{Ambiguity, ManualEvidence, ManualVersionStatus};

use crate::LookupResolution;

pub(crate) trait LookupSource: Send + Sync {
    fn resolve(&self, ambiguity: &Ambiguity) -> Option<LookupResolution>;
}

struct NpmRegistryResolver {
    client: Client,
}

struct CratesIoResolver {
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

impl LookupSource for NpmRegistryResolver {
    fn resolve(&self, ambiguity: &Ambiguity) -> Option<LookupResolution> {
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
        build_lookup_resolution(
            ambiguity,
            &candidates,
            "registry-lookup",
            "Latest npm versions",
            results,
        )
    }
}

impl LookupSource for CratesIoResolver {
    fn resolve(&self, ambiguity: &Ambiguity) -> Option<LookupResolution> {
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
        build_lookup_resolution(
            ambiguity,
            &candidates,
            "registry-lookup",
            "Latest crates.io versions",
            results,
        )
    }
}

pub(crate) fn default_lookup_sources() -> Vec<Box<dyn LookupSource>> {
    vec![
        Box::new(NpmRegistryResolver::default()),
        Box::new(CratesIoResolver::default()),
    ]
}

fn build_lookup_resolution(
    ambiguity: &Ambiguity,
    candidates: &[String],
    source: &str,
    prefix: &str,
    results: Vec<String>,
) -> Option<LookupResolution> {
    if results.is_empty() {
        return None;
    }

    let resolution = format!("{prefix}: {}", results.join(", "));
    let package = candidates
        .first()
        .cloned()
        .unwrap_or_else(|| ambiguity.id.clone());
    Some(LookupResolution {
        ambiguity_id: ambiguity.id.clone(),
        resolution: resolution.clone(),
        evidence: ManualEvidence {
            ecosystem: "registry".into(),
            package,
            version: None,
            version_status: ManualVersionStatus::Unversioned,
            source: source.into(),
            locator: String::new(),
            title: Some("Auto-resolved registry fact".into()),
            excerpt: resolution,
            relevance_reason: format!("resolved lookup ambiguity {}", ambiguity.id),
            confidence: 0.95,
        },
    })
}

fn extract_npm_package_names(ambiguity: &Ambiguity) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let text = format!("{} {}", ambiguity.question, ambiguity.options.join(" "));
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

fn extract_crate_names(ambiguity: &Ambiguity) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
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
