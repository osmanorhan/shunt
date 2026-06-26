//! End-to-end: the read view and the edit engine agree on line numbers. This is
//! the property the whole approach rests on — a line number the model reads off
//! [`shunt_edit::numbered`] addresses exactly that line in [`shunt_edit::apply`].

use shunt_edit::{Edit, apply, numbered, numbered_window};

const SAMPLE: &str = "\
import { x } from 'y';

export function greet(name) {
  return `Hello, ${name}`;
}

export const TIMEOUT = 5000;
";

#[test]
fn line_number_from_view_targets_same_line() {
    // The view shows TIMEOUT on line 7.
    let view = numbered(SAMPLE);
    let timeout_line = view
        .lines()
        .find(|l| l.contains("TIMEOUT"))
        .and_then(|l| l.split(':').next())
        .and_then(|n| n.trim().parse::<usize>().ok())
        .expect("TIMEOUT line number");
    assert_eq!(timeout_line, 7);

    // Address an edit by that exact number.
    let out = apply(
        SAMPLE,
        &Edit::ReplaceLines {
            start: timeout_line,
            end: timeout_line,
            new_text: "export const TIMEOUT = 30000;".into(),
        },
    )
    .unwrap();
    assert!(out.contains("TIMEOUT = 30000;"));
    assert!(!out.contains("5000"));
    // Nothing else moved.
    assert!(out.contains("export function greet(name) {"));
}

#[test]
fn every_line_number_round_trips() {
    // For each line, replacing it via its displayed number changes that line and
    // no other — the invariant that makes line-addressed editing safe.
    let n = SAMPLE.trim_end_matches('\n').split('\n').count();
    for line_no in 1..=n {
        let marker = format!("__EDITED_{line_no}__");
        let out = apply(
            SAMPLE,
            &Edit::ReplaceLines {
                start: line_no,
                end: line_no,
                new_text: marker.clone(),
            },
        )
        .unwrap();
        let out_lines: Vec<&str> = out.trim_end_matches('\n').split('\n').collect();
        assert_eq!(out_lines[line_no - 1], marker, "line {line_no} should be edited");
        assert_eq!(out_lines.len(), n, "line count unchanged for single-line replace");
    }
}

#[test]
fn windowed_read_addresses_correctly() {
    // Read a small window, then edit using a number from that window.
    let window = numbered_window(SAMPLE, 3, 5);
    assert!(window.starts_with("3: "));
    let out = apply(
        SAMPLE,
        &Edit::ReplaceLines {
            start: 4,
            end: 4,
            new_text: "  return `Hi, ${name}`;".into(),
        },
    )
    .unwrap();
    assert!(out.contains("Hi, ${name}"));
    assert!(!out.contains("Hello, ${name}"));
}
