use ignore::WalkBuilder;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::SearchError;

/// Maximum source files the crawler will collect before aborting.
/// Protects against accidentally running from a large root like the home directory.
pub const DEFAULT_MAX_FILES: usize = 50_000;

#[derive(Clone, Debug)]
pub struct CrawlConfig {
    pub extensions: HashSet<String>,
    pub hidden: bool,
    /// Abort with `SearchError::TooManyFiles` if more than this many matching
    /// files are found. Defaults to `DEFAULT_MAX_FILES`.
    pub max_files: usize,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            extensions: HashSet::from([
                "rs".to_string(),
                "js".to_string(),
                "jsx".to_string(),
                "mjs".to_string(),
                "cjs".to_string(),
                "ts".to_string(),
                "tsx".to_string(),
                "py".to_string(),
            ]),
            hidden: true,
            max_files: DEFAULT_MAX_FILES,
        }
    }
}

/// Directory names that are always excluded regardless of `.gitignore`.
/// These are build artifact and dependency directories that are never
/// useful to index and can be extremely large.
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

fn is_excluded_path(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component.as_os_str().to_str(),
            Some(name) if EXCLUDED_DIRS.contains(&name)
        )
    })
}

pub fn crawl_files(root: &Path, config: &CrawlConfig) -> Result<Vec<PathBuf>, SearchError> {
    let mut files = Vec::new();
    let mut builder = WalkBuilder::new(root);
    builder.hidden(config.hidden);

    for entry in builder.build().filter_map(Result::ok) {
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }

        let path = entry.into_path();

        if is_excluded_path(&path) {
            continue;
        }

        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            continue;
        };

        if config.extensions.contains(extension) {
            files.push(path);
            if files.len() > config.max_files {
                return Err(SearchError::TooManyFiles {
                    root: root.to_path_buf(),
                    limit: config.max_files,
                });
            }
        }
    }

    files.sort();
    Ok(files)
}
