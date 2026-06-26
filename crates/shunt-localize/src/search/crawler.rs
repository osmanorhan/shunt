use ignore::WalkBuilder;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct CrawlConfig {
    pub extensions: HashSet<String>,
    pub hidden: bool,
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
            hidden: false,
        }
    }
}

pub fn crawl_files(root: &Path, config: &CrawlConfig) -> Vec<PathBuf> {
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
        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            continue;
        };

        if config.extensions.contains(extension) {
            files.push(path);
        }
    }

    files.sort();
    files
}
