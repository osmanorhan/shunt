//! Edit-strategy benchmark — a research playground for how SMALL models should
//! address edits (F6 / research/directions/edit-engine.md).
//!
//! These tests drive `shunt-edit`'s position-addressed editing with the live
//! model **without touching the real AgentSession** — so we can compare strategies
//! and find ones that work for local models before graduating a winner.
//!
//! Run:
//!   FRAME_TEST_ENDPOINT=http://127.0.0.1:8080 \
//!     cargo test -p shunt-bench edit_strategy -- --ignored --nocapture --test-threads=1
//!
//! Point at a real file instead of the embedded fixture:
//!   FRAME_EDIT_FILE=/path/to/file.ts FRAME_EDIT_TASK="…" FRAME_EDIT_EXPECT="substr" …

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde::Deserialize;
    use serde_json::json;

    use shunt_edit::{Edit, apply, numbered, numbered_window};
    use shunt_infer::{OpenAiCompatProvider, ToolProvider};

    /// Strip a leading/trailing markdown code fence if the model added one.
    fn strip_fences(text: &str) -> String {
        let t = text.trim();
        if let Some(rest) = t.strip_prefix("```") {
            // drop the optional language tag on the first line, and the closing fence
            let body = rest.split_once('\n').map(|(_, body)| body).unwrap_or("");
            return body
                .trim_end()
                .trim_end_matches("```")
                .trim_end()
                .to_string();
        }
        t.to_string()
    }

    /// Just the line range — integers only. CONTENT never goes in the grammar
    /// schema (small models burn their whole thinking budget and emit empty
    /// output — the same reason the AgentSession tool schema excludes content).
    #[derive(Deserialize, Debug)]
    struct LineRange {
        start: usize,
        end: usize,
    }

    /// Provider with a generous timeout. Capabilities are NOT detected — that
    /// forces JsonSchema-grammar mode, which on this gemma build ran 194s and
    /// returned empty for a content-bearing schema. Default mode matches the
    /// working AgentSession.
    fn edit_provider() -> Option<OpenAiCompatProvider> {
        let endpoint = std::env::var("FRAME_TEST_ENDPOINT").ok()?;
        let model = std::env::var("FRAME_TEST_MODEL").unwrap_or_else(|_| "gemma4-12b".into());
        OpenAiCompatProvider::with_timeout(endpoint, model, Duration::from_secs(300)).ok()
    }

    /// A self-contained fixture with one subtle bug among several functions, so
    /// the model must pick the RIGHT line. `clamp` should return `max` when
    /// value > max, but returns `value`. Note there are two `return value;`
    /// lines — only the one inside the `value > max` block is wrong.
    const EMBEDDED: &str = "\
export function add(a: number, b: number): number {
  return a + b;
}

export function clamp(value: number, min: number, max: number): number {
  if (value < min) {
    return min;
  }
  if (value > max) {
    return value;
  }
  return value;
}

export function isEven(n: number): boolean {
  return n % 2 === 0;
}
";

    /// (source, task, expected_substring_after_fix)
    fn fixture() -> (String, String, String) {
        if let Ok(path) = std::env::var("FRAME_EDIT_FILE") {
            let src = std::fs::read_to_string(&path).expect("FRAME_EDIT_FILE unreadable");
            let task = std::env::var("FRAME_EDIT_TASK").expect("set FRAME_EDIT_TASK");
            let expect = std::env::var("FRAME_EDIT_EXPECT").unwrap_or_default();
            return (src, task, expect);
        }
        (
            EMBEDDED.to_string(),
            "The clamp(value, min, max) function should return `max` when value is greater than \
             max, but it incorrectly returns `value`. Fix clamp so values above max are clamped to max."
                .to_string(),
            "max".to_string(),
        )
    }

    /// STRATEGY: line-range replace. The model sees a line-numbered file and emits
    /// {start, end, new_text}; `shunt-edit` applies it deterministically. No
    /// exact-`old_str` reproduction — the alternative to the F6 failure.
    #[test]
    #[ignore = "edit-strategy research probe; needs FRAME_TEST_ENDPOINT"]
    fn live_edit_strategy_line_range() {
        let Some(provider) = edit_provider() else {
            return;
        };
        let (source, task, expect) = fixture();
        let view = numbered(&source);

        // STEP 1 — structure (grammar): which lines? Integers only, no content.
        let range_system = "You locate the lines that must change to fix a bug. You are given a \
            file with line numbers and a task. Reply with JSON: start = first line number to \
            replace, end = last line number to replace (1-indexed, inclusive). Pick the SMALLEST \
            contiguous range. Do not include any code.";
        let range_user = format!("TASK: {task}\n\nFILE (line-numbered):\n{view}");
        let range_schema = json!({
            "type": "object",
            "properties": { "start": { "type": "integer" }, "end": { "type": "integer" } },
            "required": ["start", "end"]
        });
        let range: LineRange = match provider.generate_structured_named(
            "line_range",
            range_system,
            &range_user,
            &range_schema,
        ) {
            Ok(r) => r,
            Err(err) => panic!("model failed to choose a line range: {err}"),
        };
        println!("STEP 1 — model chose lines {}-{}", range.start, range.end);

        // STEP 2 — content (call_text): the replacement for just those lines. This is
        // the gemma-reliable path; content is NEVER asked for via the grammar.
        let current = numbered_window(&source, range.start, range.end);
        let content_system = "You output ONLY replacement source code — no explanation, no \
            markdown fences, no line numbers. Output exactly the lines that should replace the \
            given ones.";
        let content_user = format!(
            "TASK: {task}\n\nReplace these lines (shown with their numbers) with corrected code. \
             Output only the replacement lines, no numbers:\n{current}"
        );
        let new_text = match provider.generate_text(content_system, &content_user) {
            Ok(t) => strip_fences(&t),
            Err(err) => panic!("model failed to produce replacement content: {err}"),
        };
        println!("STEP 2 — replacement:\n---\n{new_text}\n---");

        let applied = apply(
            &source,
            &Edit::ReplaceLines {
                start: range.start,
                end: range.end,
                new_text: new_text.clone(),
            },
        );
        match applied {
            Ok(new_src) => {
                let changed = new_src != source;
                let content_produced = !new_text.trim().is_empty();
                // `expect` is matched against the model's REPLACEMENT (not the whole
                // file, which already contains the token elsewhere). Informational —
                // a valid fix may be phrased differently than the canonical one.
                let on_target = expect.is_empty() || new_text.contains(&expect);
                println!(
                    "scorecard: applied=true changed={changed} content_produced={content_produced} on_target={on_target} (expect {expect:?})"
                );
                println!("--- result (window) ---\n{}", numbered(&new_src));
                // The strategy works when it lands a non-empty edit at the chosen
                // lines. (Content *quality* is the on_target metric we track.)
                assert!(changed, "edit applied but changed nothing");
                assert!(
                    content_produced,
                    "STEP 2 produced empty content (the F7 thinking trap)"
                );
            }
            // An out-of-range / inverted edit is itself a finding: the model
            // pointed at the wrong lines. The engine's error is model-facing.
            Err(e) => panic!("model's line range did not apply: {e}"),
        }
    }
}
