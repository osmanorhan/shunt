//! The read side: render source with line numbers so a model can address edits
//! by line — the handle it then uses in [`crate::Edit`]. This is what makes
//! position-addressed editing usable by a model: it never has to reproduce code,
//! only point at line numbers it can see.

/// Render the whole `source` with 1-indexed, right-aligned line numbers:
/// `   1: first line`. The gutter width adapts to the line count.
///
/// Uses `": "` (colon-space) rather than `│` so models universally read this as
/// a line annotation and never copy the gutter into replacement content.
pub fn numbered(source: &str) -> String {
    let lines: Vec<&str> = line_slice(source);
    render(&lines, 1, lines.len())
}

/// Render only lines `start..=end` (1-indexed, inclusive, clamped to the file)
/// with line numbers — a window for large files. Out-of-range or inverted
/// windows render nothing.
pub fn numbered_window(source: &str, start: usize, end: usize) -> String {
    let lines = line_slice(source);
    let n = lines.len();
    if n == 0 || start > end || start > n {
        return String::new();
    }
    let lo = start.max(1);
    let hi = end.min(n);
    render(&lines[lo - 1..hi], lo, hi)
}

fn line_slice(source: &str) -> Vec<&str> {
    if source.is_empty() {
        return Vec::new();
    }
    source
        .strip_suffix('\n')
        .unwrap_or(source)
        .split('\n')
        .collect()
}

fn render(lines: &[&str], first_no: usize, last_no: usize) -> String {
    let width = last_no.to_string().len().max(1);
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let no = first_no + i;
        out.push_str(&format!("{no:>width$}: {line}\n", no = no, width = width));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_each_line() {
        let out = numbered("alpha\nbeta\n");
        assert_eq!(out, "1: alpha\n2: beta\n");
    }

    #[test]
    fn gutter_width_adapts() {
        let src = (1..=10)
            .map(|i| format!("L{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out = numbered(&src);
        // 1-digit numbers are right-aligned to width 2.
        assert!(out.starts_with(" 1: L1\n"));
        assert!(out.contains("10: L10"));
    }

    #[test]
    fn handles_no_trailing_newline() {
        let out = numbered("only");
        assert_eq!(out, "1: only\n");
    }

    #[test]
    fn empty_source_is_empty() {
        assert_eq!(numbered(""), "");
        assert_eq!(numbered_window("", 1, 5), "");
    }

    #[test]
    fn window_clamps_to_file() {
        let src = "a\nb\nc\nd\ne\n";
        let out = numbered_window(src, 2, 4);
        assert_eq!(out, "2: b\n3: c\n4: d\n");
    }

    #[test]
    fn window_past_end_clamps() {
        let src = "a\nb\nc\n";
        let out = numbered_window(src, 2, 99);
        assert_eq!(out, "2: b\n3: c\n");
    }

    #[test]
    fn window_start_past_end_is_empty() {
        assert_eq!(numbered_window("a\nb\n", 5, 9), "");
        assert_eq!(numbered_window("a\nb\n", 2, 1), "");
    }

    #[test]
    fn window_numbers_match_file_lines() {
        // The numbers in a window are the file's real line numbers, so an edit
        // addressed from a window targets the right lines.
        let src = "a\nb\nc\nd\n";
        let out = numbered_window(src, 3, 3);
        assert_eq!(out, "3: c\n");
    }
}
