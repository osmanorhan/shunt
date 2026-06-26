use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

pub struct WorkspaceSearch {
    root: PathBuf,
}

/// Don't read files larger than this for content matching (skip blobs/minified).
const MAX_CONTENT_BYTES: u64 = 2_000_000;

const EXCLUDED_DIRS: &[&str] = &[
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    ".eggs",
];

fn is_excluded(path: &Path) -> bool {
    path.components().any(|c| {
        matches!(c.as_os_str().to_str(), Some(name) if EXCLUDED_DIRS.contains(&name))
    })
}
/// Cap returned files so a broad query can't flood the agent's context.
const MAX_HITS: usize = 50;

pub struct TextHit {
    pub path: String,
    pub line: usize,
    pub text: String,
}

impl WorkspaceSearch {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Find files whose path or content contains the query — index-free, so it
    /// is fast and bounded on any repo size (the heavyweight semantic index does
    /// not scale to large repos and is not needed to locate files). Walks the
    /// workspace honouring `.gitignore`, matches path first (cheap) then content.
    /// Empty query returns all workspace files.
    pub fn search_files(&self, query: &str) -> Vec<String> {
        if query.trim().is_empty() {
            return self.list_files();
        }
        let needle = query.trim().to_ascii_lowercase();
        let walker = WalkBuilder::new(&self.root).hidden(true).build();
        let mut hits: Vec<String> = Vec::new();
        for entry in walker.flatten() {
            if hits.len() >= MAX_HITS {
                break;
            }
            let path = entry.path();
            if !path.is_file() || is_excluded(path) {
                continue;
            }
            let Ok(rel) = path.strip_prefix(&self.root) else {
                continue;
            };
            let rel = rel.to_string_lossy().into_owned();
            // Path match is cheap and catches "find file by name".
            if rel.to_ascii_lowercase().contains(&needle) {
                hits.push(rel);
                continue;
            }
            // Content match — skip oversized/binary files.
            if path
                .metadata()
                .map(|m| m.len() > MAX_CONTENT_BYTES)
                .unwrap_or(true)
            {
                continue;
            }
            if let Ok(content) = std::fs::read_to_string(path)
                && content.to_ascii_lowercase().contains(&needle)
            {
                hits.push(rel);
            }
        }
        hits.sort();
        hits.dedup();
        hits
    }

    /// List all files in the workspace.
    pub fn list_files(&self) -> Vec<String> {
        let walker = WalkBuilder::new(&self.root).hidden(true).build();
        let mut files: Vec<String> = walker
            .flatten()
            .filter(|e| e.path().is_file() && !is_excluded(e.path()))
            .filter_map(|e| {
                e.path()
                    .strip_prefix(&self.root)
                    .ok()
                    .map(|p| p.to_string_lossy().into_owned())
            })
            .collect();
        files.sort();
        files
    }

    /// Find lines matching a pattern across all files.
    pub fn search_text(&self, query: &str) -> Vec<TextHit> {
        let query_lower = query.to_ascii_lowercase();
        let walker = WalkBuilder::new(&self.root).hidden(true).build();
        let mut hits = Vec::new();
        for entry in walker.flatten() {
            let path = entry.path();
            if !path.is_file() || is_excluded(path) {
                continue;
            }
            let rel = match path.strip_prefix(&self.root) {
                Ok(p) => p.to_string_lossy().into_owned(),
                Err(_) => continue,
            };
            if let Ok(content) = std::fs::read_to_string(path) {
                for (i, line) in content.lines().enumerate() {
                    if line.to_ascii_lowercase().contains(&query_lower) {
                        hits.push(TextHit {
                            path: rel.clone(),
                            line: i + 1,
                            text: line.to_string(),
                        });
                    }
                }
            }
        }
        hits
    }
}
