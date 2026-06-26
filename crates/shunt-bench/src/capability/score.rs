//! Metrics + scorecard for the model-capability benchmark.
//!
//! Deliberately model-agnostic: every run reduces to a small, comparable set of
//! numbers + a classified outcome, so any model/engine produces the same shape of
//! result and the scorecard ranks them apples-to-apples.

use std::time::Duration;

/// What happened on a single (model, task) run — a coarse, comparable outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Completed AND the on-disk result matches ground truth.
    Success,
    /// Completed and edited, but the result is wrong.
    WrongEdit,
    /// Completed without changing the target (e.g. "no changes needed").
    NoEdit,
    /// Did not complete: stopped, hit the turn budget, or a call timed out.
    NotCompleted,
}

impl Outcome {
    pub fn is_success(self) -> bool {
        self == Outcome::Success
    }
    pub fn glyph(self) -> char {
        match self {
            Outcome::Success => '✓',
            Outcome::WrongEdit => '≈',
            Outcome::NoEdit => '∅',
            Outcome::NotCompleted => '✗',
        }
    }
}

/// One run of one task.
#[derive(Debug, Clone)]
pub struct RunMetrics {
    pub outcome: Outcome,
    pub tool_calls: usize,
    pub elapsed: Duration,
}

/// All runs of one task for one model.
#[derive(Debug, Clone)]
pub struct TaskScore {
    pub task: String,
    pub difficulty: String,
    pub runs: Vec<RunMetrics>,
}

impl TaskScore {
    pub fn successes(&self) -> usize {
        self.runs.iter().filter(|r| r.outcome.is_success()).count()
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
    pub fn avg_secs(&self) -> f32 {
        if self.runs.is_empty() {
            0.0
        } else {
            self.runs
                .iter()
                .map(|r| r.elapsed.as_secs_f32())
                .sum::<f32>()
                / self.n() as f32
        }
    }
    pub fn avg_tool_calls(&self) -> f32 {
        if self.runs.is_empty() {
            0.0
        } else {
            self.runs.iter().map(|r| r.tool_calls as f32).sum::<f32>() / self.n() as f32
        }
    }
    /// Compact glyph strip, one per run, e.g. `✓✓∅✓`.
    pub fn glyphs(&self) -> String {
        self.runs.iter().map(|r| r.outcome.glyph()).collect()
    }
}

/// Everything for one model across the task suite.
#[derive(Debug, Clone)]
pub struct ModelScorecard {
    pub model: String,
    pub engine: String,
    pub reachable: bool,
    pub tasks: Vec<TaskScore>,
}

impl ModelScorecard {
    /// Overall success rate across all runs of all tasks.
    pub fn overall_rate(&self) -> f32 {
        let total: usize = self.tasks.iter().map(|t| t.n()).sum();
        if total == 0 {
            return 0.0;
        }
        let ok: usize = self.tasks.iter().map(|t| t.successes()).sum();
        ok as f32 / total as f32
    }
}

/// Render all model scorecards as a markdown report, ranked by overall rate.
pub fn render_report(cards: &[ModelScorecard]) -> String {
    let mut out = String::new();
    out.push_str("# Model capability benchmark\n\n");
    out.push_str("Outcome glyphs: ✓ success · ≈ wrong edit · ∅ no edit · ✗ did not complete\n\n");

    // Per-model detail.
    for card in cards {
        out.push_str(&format!("## {} ({})\n\n", card.model, card.engine));
        if !card.reachable {
            out.push_str("_endpoint unreachable — skipped_\n\n");
            continue;
        }
        out.push_str("| task | difficulty | success | runs | avg s | avg turns |\n");
        out.push_str("|------|-----------|---------|------|-------|-----------|\n");
        for t in &card.tasks {
            out.push_str(&format!(
                "| {} | {} | {}/{} {} | `{}` | {:.0} | {:.1} |\n",
                t.task,
                t.difficulty,
                t.successes(),
                t.n(),
                pct(t.success_rate()),
                t.glyphs(),
                t.avg_secs(),
                t.avg_tool_calls(),
            ));
        }
        out.push_str(&format!("\n**overall: {}**\n\n", pct(card.overall_rate())));
    }

    // Leaderboard.
    let mut ranked: Vec<&ModelScorecard> = cards.iter().filter(|c| c.reachable).collect();
    ranked.sort_by(|a, b| b.overall_rate().total_cmp(&a.overall_rate()));
    out.push_str("## Leaderboard\n\n| rank | model | overall |\n|------|-------|---------|\n");
    for (i, c) in ranked.iter().enumerate() {
        out.push_str(&format!(
            "| {} | {} | {} |\n",
            i + 1,
            c.model,
            pct(c.overall_rate())
        ));
    }
    out
}

fn pct(r: f32) -> String {
    format!("{:.0}%", r * 100.0)
}
