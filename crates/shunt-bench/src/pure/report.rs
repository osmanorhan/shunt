use std::path::Path;

use crate::capability::{CapabilityTask, Outcome};

use super::agent_loop::{LogKind, RunTrace};
use super::metrics::PureRunMetrics;

pub struct EntryScore {
    pub model: String,
    pub engine: String,
    pub quant: Option<String>,
    pub reachable: bool,
    pub task_scores: Vec<PureTaskScore>,
}

pub struct PureTaskScore {
    pub task: String,
    pub difficulty: String,
    pub runs: Vec<PureRunMetrics>,
}

impl PureTaskScore {
    pub fn successes(&self) -> usize {
        self.runs
            .iter()
            .filter(|r| r.outcome == Outcome::Success)
            .count()
    }
    pub fn n(&self) -> usize {
        self.runs.len()
    }
    pub fn success_rate(&self) -> f32 {
        if self.runs.is_empty() {
            0.0
        } else {
            self.successes() as f32 / self.n() as f32
        }
    }
    pub fn avg_turns(&self) -> f32 {
        avg(self.runs.iter().map(|r| r.turns as f32))
    }
    pub fn avg_secs(&self) -> f32 {
        avg(self.runs.iter().map(|r| r.wall_secs))
    }
    pub fn avg_native_tool(&self) -> f32 {
        avg(self.runs.iter().map(|r| r.native_tool_pct))
    }
    pub fn avg_valid_args(&self) -> f32 {
        avg(self.runs.iter().map(|r| r.valid_args_pct))
    }
    pub fn glyphs(&self) -> String {
        self.runs.iter().map(|r| r.outcome.glyph()).collect()
    }
}

impl EntryScore {
    pub fn overall_rate(&self) -> f32 {
        let total: usize = self.task_scores.iter().map(|t| t.n()).sum();
        if total == 0 {
            return 0.0;
        }
        let ok: usize = self.task_scores.iter().map(|t| t.successes()).sum();
        ok as f32 / total as f32
    }
    pub fn avg_native_tool(&self) -> f32 {
        let runs: Vec<_> = self.task_scores.iter().flat_map(|t| &t.runs).collect();
        if runs.is_empty() {
            return 0.0;
        }
        runs.iter().map(|r| r.native_tool_pct).sum::<f32>() / runs.len() as f32
    }
    pub fn avg_valid_args(&self) -> f32 {
        let runs: Vec<_> = self.task_scores.iter().flat_map(|t| &t.runs).collect();
        if runs.is_empty() {
            return 1.0;
        }
        runs.iter().map(|r| r.valid_args_pct).sum::<f32>() / runs.len() as f32
    }
    pub fn avg_turns(&self) -> f32 {
        let runs: Vec<_> = self.task_scores.iter().flat_map(|t| &t.runs).collect();
        if runs.is_empty() {
            return 0.0;
        }
        runs.iter().map(|r| r.turns as f32).sum::<f32>() / runs.len() as f32
    }
    pub fn avg_secs(&self) -> f32 {
        let runs: Vec<_> = self.task_scores.iter().flat_map(|t| &t.runs).collect();
        if runs.is_empty() {
            return 0.0;
        }
        runs.iter().map(|r| r.wall_secs).sum::<f32>() / runs.len() as f32
    }
    pub fn entry_label(&self) -> String {
        match &self.quant {
            Some(q) => format!("{} [{}/{}]", self.model, self.engine, q),
            None => format!("{} [{}]", self.model, self.engine),
        }
    }
}

pub fn render_pure_report(entries: &[EntryScore]) -> String {
    let mut out = String::new();
    out.push_str("# pure-bench: agentic stack leaderboard\n\n");
    out.push_str("Outcome glyphs: ✓ success · ≈ wrong edit · ∅ no edit · ✗ not completed\n\n");

    for entry in entries {
        out.push_str(&format!("## {}\n\n", entry.entry_label()));
        if !entry.reachable {
            out.push_str("_endpoint unreachable — skipped_\n\n");
            continue;
        }
        out.push_str("| task | difficulty | success | runs | native-tool | valid-args | avg turns | avg s |\n");
        out.push_str("|------|-----------|---------|------|-------------|------------|-----------|-------|\n");
        for ts in &entry.task_scores {
            out.push_str(&format!(
                "| {} | {} | {}/{} {} | `{}` | {} | {} | {:.1} | {:.0} |\n",
                ts.task,
                ts.difficulty,
                ts.successes(),
                ts.n(),
                pct(ts.success_rate()),
                ts.glyphs(),
                pct(ts.avg_native_tool()),
                pct(ts.avg_valid_args()),
                ts.avg_turns(),
                ts.avg_secs(),
            ));
        }
        out.push_str(&format!("\n**overall: {}**\n\n", pct(entry.overall_rate())));
    }

    // Leaderboard: sort by success rate desc, then native-tool %, then valid-args %, then fewer turns.
    let mut ranked: Vec<&EntryScore> = entries.iter().filter(|e| e.reachable).collect();
    ranked.sort_by(|a, b| {
        b.overall_rate()
            .total_cmp(&a.overall_rate())
            .then(b.avg_native_tool().total_cmp(&a.avg_native_tool()))
            .then(b.avg_valid_args().total_cmp(&a.avg_valid_args()))
            .then(a.avg_turns().total_cmp(&b.avg_turns()))
    });

    out.push_str("## Leaderboard\n\n");
    out.push_str("| rank | model | engine | quant | success | native-tool | valid-args | avg turns | avg s |\n");
    out.push_str("|------|-------|--------|-------|---------|-------------|------------|-----------|-------|\n");
    for (i, e) in ranked.iter().enumerate() {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.0} |\n",
            i + 1,
            e.model,
            e.engine,
            e.quant.as_deref().unwrap_or("—"),
            pct(e.overall_rate()),
            pct(e.avg_native_tool()),
            pct(e.avg_valid_args()),
            e.avg_turns(),
            e.avg_secs(),
        ));
    }
    out
}

pub fn write_run_log(
    entry: &EntryScore,
    task: &CapabilityTask,
    run_idx: usize,
    metrics: &PureRunMetrics,
    trace: &RunTrace,
    file_snapshots: &[(&str, String, bool)],
) {
    let dir = Path::new("pure-logs").join(safe_name(&entry.entry_label()));
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join(format!("{}-run-{run_idx}.md", safe_name(task.name)));

    let mut log = String::new();
    log.push_str(&format!(
        "# {} / {} / run {}\n\n",
        entry.entry_label(),
        task.name,
        run_idx
    ));
    log.push_str(&format!("- difficulty: {}\n", task.difficulty.label()));
    log.push_str(&format!(
        "- outcome: {:?} `{}`\n",
        metrics.outcome,
        metrics.outcome.glyph()
    ));
    log.push_str(&format!("- stop_reason: {}\n", metrics.stop_reason));
    log.push_str(&format!("- turns: {}\n", metrics.turns));
    log.push_str(&format!(
        "- native_tool: {}\n",
        pct(metrics.native_tool_pct)
    ));
    log.push_str(&format!("- valid_args: {}\n", pct(metrics.valid_args_pct)));
    log.push_str(&format!("- unknown_tool: {}\n", metrics.unknown_tool_count));
    log.push_str(&format!(
        "- schema_mismatch: {}\n",
        metrics.schema_mismatch_count
    ));
    log.push_str(&format!("- thrash: {}\n", metrics.thrash));
    log.push_str(&format!(
        "- read_before_edit: {}\n",
        metrics.read_before_edit
    ));
    log.push_str(&format!("- total_tokens: {}\n", metrics.total_tokens));
    log.push_str(&format!("- wall_secs: {:.1}\n", metrics.wall_secs));

    log.push_str("\n## Request\n\n");
    log.push_str(task.request);

    log.push_str("\n\n## Checked Files\n\n");
    for (rel, content, passed) in file_snapshots {
        log.push_str(&format!(
            "### `{rel}` (check_passed: `{passed}`)\n\n```\n{content}```\n\n"
        ));
    }

    log.push_str("## Tool Call Log\n\n");
    for entry in &trace.log {
        match &entry.kind {
            LogKind::ModelResponse {
                finish_reason,
                content_preview,
            } => {
                log.push_str(&format!(
                    "**turn {}** model response (finish_reason={:?}){}\n\n",
                    entry.turn,
                    finish_reason,
                    content_preview
                        .as_ref()
                        .map(|c| format!(": `{c}`"))
                        .unwrap_or_default(),
                ));
            }
            LogKind::ToolCall {
                name,
                args_preview,
                result_preview,
                is_error,
                valid,
            } => {
                let status = if *is_error { "err" } else { "ok" };
                let validity = if *valid { "" } else { " [INVALID]" };
                log.push_str(&format!(
                    "  → `{name}`{validity} args=`{args_preview}` → [{status}] `{result_preview}`\n\n",
                ));
            }
            LogKind::Stopped { reason } => {
                log.push_str(&format!("**stopped:** {reason}\n\n"));
            }
        }
    }

    let _ = std::fs::write(&path, log);
}

fn avg(iter: impl Iterator<Item = f32>) -> f32 {
    let v: Vec<f32> = iter.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

fn pct(r: f32) -> String {
    format!("{:.0}%", r * 100.0)
}

pub fn safe_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}
