use super::ChunkId;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

#[derive(Debug, Eq, PartialEq)]
pub enum SearchError {
    DuplicateChunkId(ChunkId),
    UnknownChunkId(ChunkId),
    InvalidRoot(PathBuf),
    Io { path: PathBuf, message: String },
    ManifestFormat { path: PathBuf, message: String },
    ParserUnavailable(String),
    SnapshotFormat { path: PathBuf, message: String },
    TooManyFiles { root: PathBuf, limit: usize },
}

impl Display for SearchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateChunkId(id) => write!(f, "duplicate chunk id: {id}"),
            Self::UnknownChunkId(id) => write!(f, "unknown chunk id: {id}"),
            Self::InvalidRoot(path) => write!(f, "invalid repository path: {}", path.display()),
            Self::Io { path, message } => write!(f, "{}: {message}", path.display()),
            Self::ManifestFormat { path, message } => write!(f, "{}: {message}", path.display()),
            Self::ParserUnavailable(language) => write!(f, "parser unavailable for {language}"),
            Self::SnapshotFormat { path, message } => write!(f, "{}: {message}", path.display()),
            Self::TooManyFiles { root, limit } => write!(
                f,
                "{} contains more than {limit} source files — run shunt from a project subdirectory, not a large root like your home directory",
                root.display()
            ),
        }
    }
}

impl Error for SearchError {}
