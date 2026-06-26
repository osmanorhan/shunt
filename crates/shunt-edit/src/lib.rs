//! `shunt-edit` — deterministic, position-addressed code editing.
//!
//! A small, reusable engine for applying *position-addressed* edits to source
//! text — the way real editors do it (LSP `TextEdit` ranges, AST nodes) — rather
//! than reproducing exact text to find-and-replace.
//!
//! ## Why
//! Built for **small language models**. Exact-text `str_replace` requires the
//! model to reproduce verbatim code, which small models can't do reliably. Here
//! the model addresses an edit by **line number** — which it can read straight
//! off a numbered listing ([`numbered`]) — and the engine applies it
//! deterministically. Precision lives in the tool, not the model.
//!
//! ## Surface
//! Pure transforms only: `source: &str` + [`Edit`] → `Result<String, EditError>`.
//! No file IO, no agent/CLI concepts — a library + a playground for editing
//! strategies we can benchmark for local models. Add new strategies behind the
//! same [`Edit`]/[`apply`] surface without touching callers.
//!
//! ```
//! use shunt_edit::{apply, Edit};
//! let src = "fn a() {}\nfn b() {}\n";
//! let out = apply(src, &Edit::ReplaceLines { start: 2, end: 2, new_text: "fn c() {}".into() }).unwrap();
//! assert_eq!(out, "fn a() {}\nfn c() {}\n");
//! ```

mod apply;
mod view;

pub use apply::{apply, apply_all};
pub use view::{numbered, numbered_window};

// ── Edit ──────────────────────────────────────────────────────────────────────

/// A position-addressed edit. Line numbers are **1-indexed and inclusive** — the
/// same numbers a human (or model) reads off a numbered listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Edit {
    /// Replace lines `start..=end` with `new_text`. Empty `new_text` deletes the
    /// lines (same as [`Edit::DeleteLines`]). A single trailing newline in
    /// `new_text` is ignored — it's treated as a list of replacement lines.
    ReplaceLines {
        start: usize,
        end: usize,
        new_text: String,
    },
    /// Insert `new_text` immediately after line `after`. `after == 0` inserts
    /// before the first line; `after == line_count` appends at the end.
    InsertAfter { after: usize, new_text: String },
    /// Delete lines `start..=end`.
    DeleteLines { start: usize, end: usize },
}

impl Edit {
    /// The inclusive line span this edit removes/replaces, or `None` for a pure
    /// insert. Used for overlap detection in [`apply_all`].
    pub(crate) fn footprint(&self) -> Option<(usize, usize)> {
        match self {
            Edit::ReplaceLines { start, end, .. } | Edit::DeleteLines { start, end } => {
                Some((*start, *end))
            }
            Edit::InsertAfter { .. } => None,
        }
    }

    /// Sort key for bottom-to-top application (higher first keeps lower line
    /// numbers valid). Inserts after line K order as if at line K+1.
    pub(crate) fn anchor(&self) -> usize {
        match self {
            Edit::ReplaceLines { start, .. } | Edit::DeleteLines { start, .. } => *start,
            Edit::InsertAfter { after, .. } => after + 1,
        }
    }
}

// ── EditError ──────────────────────────────────────────────────────────────────

/// Why an edit could not be applied. Messages are written to be fed back to a
/// model verbatim so it can correct its next attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditError {
    /// A line range fell outside `1..=line_count`.
    OutOfRange {
        start: usize,
        end: usize,
        line_count: usize,
    },
    /// `start > end`.
    InvertedRange { start: usize, end: usize },
    /// An insert anchor exceeded `line_count`.
    InsertOutOfRange { after: usize, line_count: usize },
    /// Two edits in a batch touch overlapping lines.
    Overlap {
        a: (usize, usize),
        b: (usize, usize),
    },
}

impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditError::OutOfRange {
                start,
                end,
                line_count,
            } => write!(
                f,
                "lines {start}-{end} are outside the file (it has {line_count} line(s); valid range is 1-{line_count})"
            ),
            EditError::InvertedRange { start, end } => {
                write!(f, "start line {start} is after end line {end}")
            }
            EditError::InsertOutOfRange { after, line_count } => write!(
                f,
                "cannot insert after line {after}: the file has {line_count} line(s) (use 0..={line_count})"
            ),
            EditError::Overlap { a, b } => write!(
                f,
                "edits overlap: lines {}-{} and {}-{} touch the same region",
                a.0, a.1, b.0, b.1
            ),
        }
    }
}

impl std::error::Error for EditError {}
