//! Line-range edit strategy: splice replacement lines into the source.
//!
//! Line-oriented splicing is the simplest *correct* implementation for
//! line-addressed edits (no char-offset bookkeeping, no rope quirks). A
//! char/column-addressed strategy (e.g. ropey-backed) can be added later behind
//! the same [`crate::Edit`] surface for column-precise edits.

use crate::{Edit, EditError};

/// Apply a single [`Edit`] to `source`, returning the new content.
///
/// The file's trailing-newline state is preserved.
pub fn apply(source: &str, edit: &Edit) -> Result<String, EditError> {
    let (mut lines, trailing) = into_lines(source);
    let n = lines.len();
    match edit {
        Edit::ReplaceLines {
            start,
            end,
            new_text,
        } => {
            check_range(*start, *end, n)?;
            lines.splice((start - 1)..*end, split_new(new_text));
        }
        Edit::DeleteLines { start, end } => {
            check_range(*start, *end, n)?;
            lines.splice((start - 1)..*end, std::iter::empty());
        }
        Edit::InsertAfter { after, new_text } => {
            if *after > n {
                return Err(EditError::InsertOutOfRange {
                    after: *after,
                    line_count: n,
                });
            }
            let at = *after;
            lines.splice(at..at, split_new(new_text));
        }
    }
    Ok(join_lines(lines, trailing))
}

/// Apply several edits, addressed against the **original** line numbers. Edits
/// are validated (and checked for mutual overlap) up front, then applied
/// bottom-to-top so earlier line numbers stay valid. Order of `edits` is
/// irrelevant; overlapping range-edits are an [`EditError::Overlap`].
pub fn apply_all(source: &str, edits: &[Edit]) -> Result<String, EditError> {
    let (lines, _) = into_lines(source);
    let n = lines.len();

    // Validate each edit individually, then check pairwise overlap of footprints.
    for edit in edits {
        validate(edit, n)?;
    }
    for (i, a) in edits.iter().enumerate() {
        for b in &edits[i + 1..] {
            if let (Some(fa), Some(fb)) = (a.footprint(), b.footprint())
                && fa.0 <= fb.1
                && fb.0 <= fa.1
            {
                return Err(EditError::Overlap { a: fa, b: fb });
            }
        }
    }

    // Apply highest-anchored first; lower line numbers are unaffected by edits above.
    let mut ordered: Vec<&Edit> = edits.iter().collect();
    ordered.sort_by_key(|edit| std::cmp::Reverse(edit.anchor()));

    let mut out = source.to_string();
    for edit in ordered {
        out = apply(&out, edit)?;
    }
    Ok(out)
}

// ── helpers ────────────────────────────────────────────────────────────────────

fn validate(edit: &Edit, n: usize) -> Result<(), EditError> {
    match edit {
        Edit::ReplaceLines { start, end, .. } | Edit::DeleteLines { start, end } => {
            check_range(*start, *end, n)
        }
        Edit::InsertAfter { after, .. } => {
            if *after > n {
                Err(EditError::InsertOutOfRange {
                    after: *after,
                    line_count: n,
                })
            } else {
                Ok(())
            }
        }
    }
}

fn check_range(start: usize, end: usize, n: usize) -> Result<(), EditError> {
    if start > end {
        return Err(EditError::InvertedRange { start, end });
    }
    if start < 1 || end > n {
        return Err(EditError::OutOfRange {
            start,
            end,
            line_count: n,
        });
    }
    Ok(())
}

/// Split `source` into its lines plus whether it ended with a newline. An empty
/// source is zero lines.
fn into_lines(source: &str) -> (Vec<String>, bool) {
    if source.is_empty() {
        return (Vec::new(), false);
    }
    let trailing = source.ends_with('\n');
    let body = source.strip_suffix('\n').unwrap_or(source);
    (body.split('\n').map(str::to_string).collect(), trailing)
}

fn join_lines(lines: Vec<String>, trailing: bool) -> String {
    let mut s = lines.join("\n");
    if trailing && !s.is_empty() {
        s.push('\n');
    }
    s
}

/// Interpret `new_text` as a list of replacement lines. Empty → no lines; a
/// single trailing newline is ignored (so "a\nb" and "a\nb\n" both → [a, b]).
fn split_new(new_text: &str) -> Vec<String> {
    if new_text.is_empty() {
        return Vec::new();
    }
    let body = new_text.strip_suffix('\n').unwrap_or(new_text);
    body.split('\n').map(str::to_string).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rep(start: usize, end: usize, t: &str) -> Edit {
        Edit::ReplaceLines {
            start,
            end,
            new_text: t.into(),
        }
    }

    #[test]
    fn replace_middle_line() {
        let out = apply("a\nb\nc\n", &rep(2, 2, "B")).unwrap();
        assert_eq!(out, "a\nB\nc\n");
    }

    #[test]
    fn replace_first_line() {
        let out = apply("a\nb\nc\n", &rep(1, 1, "A")).unwrap();
        assert_eq!(out, "A\nb\nc\n");
    }

    #[test]
    fn replace_last_line_with_trailing_newline() {
        let out = apply("a\nb\nc\n", &rep(3, 3, "C")).unwrap();
        assert_eq!(out, "a\nb\nC\n");
    }

    #[test]
    fn replace_last_line_no_trailing_newline_preserved() {
        let out = apply("a\nb\nc", &rep(3, 3, "C")).unwrap();
        assert_eq!(out, "a\nb\nC"); // no trailing newline preserved
    }

    #[test]
    fn replace_range_with_multiple_lines() {
        let out = apply("a\nb\nc\nd\n", &rep(2, 3, "X\nY\nZ")).unwrap();
        assert_eq!(out, "a\nX\nY\nZ\nd\n");
    }

    #[test]
    fn replace_multiple_lines_with_one() {
        let out = apply("a\nb\nc\nd\n", &rep(2, 3, "B")).unwrap();
        assert_eq!(out, "a\nB\nd\n");
    }

    #[test]
    fn new_text_trailing_newline_ignored() {
        let a = apply("a\nb\n", &rep(1, 1, "X")).unwrap();
        let b = apply("a\nb\n", &rep(1, 1, "X\n")).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, "X\nb\n");
    }

    #[test]
    fn replace_with_empty_deletes() {
        let out = apply("a\nb\nc\n", &rep(2, 2, "")).unwrap();
        assert_eq!(out, "a\nc\n");
    }

    #[test]
    fn delete_lines() {
        let out = apply("a\nb\nc\nd\n", &Edit::DeleteLines { start: 2, end: 3 }).unwrap();
        assert_eq!(out, "a\nd\n");
    }

    #[test]
    fn delete_all_lines() {
        let out = apply("a\nb\n", &Edit::DeleteLines { start: 1, end: 2 }).unwrap();
        assert_eq!(out, "");
    }

    #[test]
    fn insert_after_middle() {
        let out = apply(
            "a\nb\nc\n",
            &Edit::InsertAfter {
                after: 1,
                new_text: "X\nY".into(),
            },
        )
        .unwrap();
        assert_eq!(out, "a\nX\nY\nb\nc\n");
    }

    #[test]
    fn insert_at_top() {
        let out = apply(
            "a\nb\n",
            &Edit::InsertAfter {
                after: 0,
                new_text: "head".into(),
            },
        )
        .unwrap();
        assert_eq!(out, "head\na\nb\n");
    }

    #[test]
    fn insert_at_end() {
        let out = apply(
            "a\nb\n",
            &Edit::InsertAfter {
                after: 2,
                new_text: "tail".into(),
            },
        )
        .unwrap();
        assert_eq!(out, "a\nb\ntail\n");
    }

    #[test]
    fn insert_into_empty_file() {
        let out = apply(
            "",
            &Edit::InsertAfter {
                after: 0,
                new_text: "first".into(),
            },
        )
        .unwrap();
        assert_eq!(out, "first");
    }

    #[test]
    fn out_of_range_errors() {
        assert_eq!(
            apply("a\nb\n", &rep(3, 3, "x")),
            Err(EditError::OutOfRange {
                start: 3,
                end: 3,
                line_count: 2
            })
        );
        assert_eq!(
            apply("a\nb\n", &rep(0, 1, "x")),
            Err(EditError::OutOfRange {
                start: 0,
                end: 1,
                line_count: 2
            })
        );
    }

    #[test]
    fn inverted_range_errors() {
        assert_eq!(
            apply("a\nb\nc\n", &rep(3, 2, "x")),
            Err(EditError::InvertedRange { start: 3, end: 2 })
        );
    }

    #[test]
    fn insert_out_of_range_errors() {
        assert_eq!(
            apply(
                "a\nb\n",
                &Edit::InsertAfter {
                    after: 5,
                    new_text: "x".into()
                }
            ),
            Err(EditError::InsertOutOfRange {
                after: 5,
                line_count: 2
            })
        );
    }

    #[test]
    fn apply_all_multiple_non_overlapping() {
        // Order intentionally not bottom-to-top; engine sorts.
        let edits = vec![rep(1, 1, "A"), rep(3, 3, "C")];
        let out = apply_all("a\nb\nc\n", &edits).unwrap();
        assert_eq!(out, "A\nb\nC\n");
    }

    #[test]
    fn apply_all_mixed_replace_and_insert() {
        let edits = vec![
            rep(1, 1, "A"),
            Edit::InsertAfter {
                after: 2,
                new_text: "mid".into(),
            },
        ];
        let out = apply_all("a\nb\nc\n", &edits).unwrap();
        assert_eq!(out, "A\nb\nmid\nc\n");
    }

    #[test]
    fn apply_all_detects_overlap() {
        let edits = vec![rep(1, 2, "X"), rep(2, 3, "Y")];
        assert_eq!(
            apply_all("a\nb\nc\n", &edits),
            Err(EditError::Overlap {
                a: (1, 2),
                b: (2, 3)
            })
        );
    }

    #[test]
    fn apply_all_validates_against_original_line_count() {
        let edits = vec![rep(5, 5, "x")];
        assert!(matches!(
            apply_all("a\nb\n", &edits),
            Err(EditError::OutOfRange { .. })
        ));
    }

    #[test]
    fn error_messages_are_model_friendly() {
        let msg = apply("a\nb\n", &rep(9, 9, "x")).unwrap_err().to_string();
        assert!(msg.contains("9-9"));
        assert!(msg.contains("2 line"));
        assert!(msg.contains("1-2"));
    }
}
