use std::path::Path;

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Serialize, de::DeserializeOwned};
use shunt_core::{
    AdaptationPackage, CorrectionPackage, FrontierCase, RecipeRun, TaskRun, UnderstandingArtifact,
    ledger::{LedgerEntry, LedgerEntryId, LedgerEntryRecord},
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type StoreResult<T> = Result<T, StoreError>;

pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Self> {
        let conn = Connection::open(path)?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_in_memory() -> StoreResult<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    pub fn put_task_run(&self, task: &TaskRun) -> StoreResult<()> {
        self.put_record(
            "task_runs",
            &task.id.0,
            Some(&task.id.0),
            task.updated_at.unix_timestamp(),
            task,
        )
    }

    pub fn get_task_run(&self, task_id: &str) -> StoreResult<Option<TaskRun>> {
        self.get_record("task_runs", task_id)
    }

    pub fn put_understanding_artifact(&self, artifact: &UnderstandingArtifact) -> StoreResult<()> {
        self.put_record(
            "understanding_artifacts",
            &artifact.id.0,
            Some(&artifact.task_id.0),
            artifact.updated_at.unix_timestamp(),
            artifact,
        )
    }

    pub fn get_understanding_artifact(
        &self,
        artifact_id: &str,
    ) -> StoreResult<Option<UnderstandingArtifact>> {
        self.get_record("understanding_artifacts", artifact_id)
    }

    pub fn put_recipe_run(&self, recipe_run: &RecipeRun) -> StoreResult<()> {
        self.put_record(
            "recipe_runs",
            &recipe_run.id.0,
            Some(&recipe_run.task_id.0),
            recipe_run.updated_at.unix_timestamp(),
            recipe_run,
        )
    }

    pub fn get_recipe_run(&self, recipe_run_id: &str) -> StoreResult<Option<RecipeRun>> {
        self.get_record("recipe_runs", recipe_run_id)
    }

    pub fn put_frontier_case(&self, frontier_case: &FrontierCase) -> StoreResult<()> {
        self.put_record(
            "frontier_cases",
            &frontier_case.id.0,
            Some(&frontier_case.task_id.0),
            frontier_case.updated_at.unix_timestamp(),
            frontier_case,
        )
    }

    pub fn get_frontier_case(&self, frontier_case_id: &str) -> StoreResult<Option<FrontierCase>> {
        self.get_record("frontier_cases", frontier_case_id)
    }

    pub fn list_frontier_cases_for_task(&self, task_id: &str) -> StoreResult<Vec<FrontierCase>> {
        self.list_records_for_task("frontier_cases", task_id)
    }

    pub fn put_agent_pause<T>(
        &self,
        task_id: &str,
        updated_at: time::OffsetDateTime,
        pause: &T,
    ) -> StoreResult<()>
    where
        T: Serialize,
    {
        self.put_record(
            "agent_pauses",
            task_id,
            Some(task_id),
            updated_at.unix_timestamp(),
            pause,
        )
    }

    pub fn get_agent_pause<T>(&self, task_id: &str) -> StoreResult<Option<T>>
    where
        T: DeserializeOwned,
    {
        self.get_record("agent_pauses", task_id)
    }

    pub fn delete_agent_pause(&self, task_id: &str) -> StoreResult<()> {
        self.conn
            .execute("DELETE FROM agent_pauses WHERE id = ?1", params![task_id])?;
        Ok(())
    }

    pub fn put_correction_package(&self, correction: &CorrectionPackage) -> StoreResult<()> {
        self.put_record(
            "correction_packages",
            &correction.id.0,
            None,
            correction.created_at.unix_timestamp(),
            correction,
        )
    }

    pub fn get_correction_package(
        &self,
        correction_id: &str,
    ) -> StoreResult<Option<CorrectionPackage>> {
        self.get_record("correction_packages", correction_id)
    }

    pub fn put_adaptation_package(&self, adaptation: &AdaptationPackage) -> StoreResult<()> {
        self.put_record(
            "adaptation_packages",
            &adaptation.id.0,
            None,
            adaptation.created_at.unix_timestamp(),
            adaptation,
        )
    }

    pub fn get_adaptation_package(
        &self,
        adaptation_id: &str,
    ) -> StoreResult<Option<AdaptationPackage>> {
        self.get_record("adaptation_packages", adaptation_id)
    }

    // ── Ledger ────────────────────────────────────────────────────────────────

    /// Append a single ledger entry for `task_id`.
    /// Assigns the next sequence number for the task automatically.
    pub fn append_ledger_entry(
        &self,
        task_id: &str,
        entry: LedgerEntry,
    ) -> StoreResult<LedgerEntryRecord> {
        let sequence = self.next_ledger_sequence(task_id)?;
        let id = LedgerEntryId(format!("le-{task_id}-{sequence}"));
        let created_at = time::OffsetDateTime::now_utc();
        let record = LedgerEntryRecord {
            id: id.clone(),
            task_id: task_id.into(),
            sequence,
            entry,
            created_at,
        };
        let body = serde_json::to_string(&record)?;
        self.conn.execute(
            "INSERT INTO ledger_entries (id, task_id, sequence, created_at, body)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                id.0,
                task_id,
                sequence as i64,
                created_at.unix_timestamp(),
                body
            ],
        )?;
        Ok(record)
    }

    /// All ledger entries for `task_id` in ascending sequence order.
    pub fn list_ledger_entries(&self, task_id: &str) -> StoreResult<Vec<LedgerEntryRecord>> {
        let mut stmt = self
            .conn
            .prepare("SELECT body FROM ledger_entries WHERE task_id = ?1 ORDER BY sequence ASC")?;
        let rows = stmt.query_map(rusqlite::params![task_id], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }
        Ok(out)
    }

    fn next_ledger_sequence(&self, task_id: &str) -> StoreResult<u64> {
        let max: Option<i64> = self
            .conn
            .query_row(
                "SELECT MAX(sequence) FROM ledger_entries WHERE task_id = ?1",
                rusqlite::params![task_id],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        Ok(max.map(|n| n as u64 + 1).unwrap_or(0))
    }

    fn migrate(&self) -> StoreResult<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS task_runs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                body TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_task_runs_task_id ON task_runs(task_id);

            CREATE TABLE IF NOT EXISTS understanding_artifacts (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                body TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_understanding_artifacts_task_id
                ON understanding_artifacts(task_id);

            CREATE TABLE IF NOT EXISTS recipe_runs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                body TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_recipe_runs_task_id ON recipe_runs(task_id);

            CREATE TABLE IF NOT EXISTS frontier_cases (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                body TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_frontier_cases_task_id ON frontier_cases(task_id);

            CREATE TABLE IF NOT EXISTS agent_pauses (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                body TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_agent_pauses_task_id ON agent_pauses(task_id);

            CREATE TABLE IF NOT EXISTS correction_packages (
                id TEXT PRIMARY KEY,
                task_id TEXT,
                updated_at INTEGER NOT NULL,
                body TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS adaptation_packages (
                id TEXT PRIMARY KEY,
                task_id TEXT,
                updated_at INTEGER NOT NULL,
                body TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ledger_entries (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                body TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ledger_entries_task_id
                ON ledger_entries(task_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_entries_task_seq
                ON ledger_entries(task_id, sequence);
            "#,
        )?;

        Ok(())
    }

    fn put_record<T>(
        &self,
        table: &str,
        id: &str,
        task_id: Option<&str>,
        updated_at: i64,
        value: &T,
    ) -> StoreResult<()>
    where
        T: Serialize,
    {
        let body = serde_json::to_string(value)?;
        let sql = format!(
            "INSERT INTO {table} (id, task_id, updated_at, body)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(id) DO UPDATE SET
               task_id = excluded.task_id,
               updated_at = excluded.updated_at,
               body = excluded.body"
        );

        self.conn
            .execute(&sql, params![id, task_id, updated_at, body])?;
        Ok(())
    }

    fn get_record<T>(&self, table: &str, id: &str) -> StoreResult<Option<T>>
    where
        T: DeserializeOwned,
    {
        let sql = format!("SELECT body FROM {table} WHERE id = ?1");
        let body: Option<String> = self
            .conn
            .query_row(&sql, params![id], |row| row.get(0))
            .optional()?;

        body.map(|body| serde_json::from_str(&body))
            .transpose()
            .map_err(Into::into)
    }

    fn list_records_for_task<T>(&self, table: &str, task_id: &str) -> StoreResult<Vec<T>>
    where
        T: DeserializeOwned,
    {
        let sql =
            format!("SELECT body FROM {table} WHERE task_id = ?1 ORDER BY updated_at ASC, id ASC");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![task_id], |row| row.get::<_, String>(0))?;

        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row?)?);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use shunt_core::{
        ApprovalState, ApprovalStatus, ArtifactId, EvidenceKind, EvidenceRef, FrontierCase,
        FrontierCaseId, FrontierReason, FrontierStatus, TaskId, TaskPhase, TaskRun,
        UncertaintyEvent, UncertaintyKind, UnderstandingArtifact,
    };
    use time::macros::datetime;

    use super::SqliteStore;

    #[test]
    fn round_trips_task_and_artifact() {
        let store = SqliteStore::open_in_memory().unwrap();
        let now = datetime!(2026-05-01 12:00 UTC);

        let task = TaskRun {
            id: TaskId("task-1".into()),
            workspace_root: "/tmp/workspace".into(),
            phase: TaskPhase::Understand,
            current_artifact: ArtifactId("artifact-1".into()),
            active_recipe_run: None,
            frontier_cases: vec![],
            created_at: now,
            updated_at: now,
        };

        let artifact = UnderstandingArtifact {
            id: ArtifactId("artifact-1".into()),
            task_id: task.id.clone(),
            original_request: "fix the parser".into(),
            interpreted_goal: "repair parser failure in config loading".into(),
            success_criteria: vec!["tests pass".into()],
            constraints: vec!["stay local-first".into()],
            target_scope: vec!["src/config.rs".into()],
            work_contract: Default::default(),
            evidence: vec![EvidenceRef {
                kind: EvidenceKind::File,
                locator: "src/config.rs".into(),
                summary: "parser entrypoint".into(),
            }],
            candidate_files: vec![],
            package_facts: vec![],
            manual_evidence: vec![],
            assumptions: vec![],
            ambiguities: vec![],
            selected_recipe: None,
            risks: vec![],
            confidence: 0.7,
            approval: ApprovalState {
                status: ApprovalStatus::NeedsReview,
                decided_by: None,
                decided_at: None,
                note: None,
            },
            revision: 1,
            workspace_profile: shunt_core::WorkspaceProfile::default(),
            created_at: now,
            updated_at: now,
        };

        store.put_task_run(&task).unwrap();
        store.put_understanding_artifact(&artifact).unwrap();

        assert_eq!(store.get_task_run("task-1").unwrap(), Some(task));
        assert_eq!(
            store.get_understanding_artifact("artifact-1").unwrap(),
            Some(artifact)
        );
    }

    #[test]
    fn lists_frontier_cases_by_task() {
        let store = SqliteStore::open_in_memory().unwrap();
        let now = datetime!(2026-05-01 12:00 UTC);

        let frontier_case = FrontierCase {
            id: FrontierCaseId("frontier-1".into()),
            task_id: TaskId("task-1".into()),
            artifact_id: ArtifactId("artifact-1".into()),
            recipe_run_id: None,
            reason: FrontierReason::LowConfidence,
            status: FrontierStatus::Open,
            summary: "understanding confidence dropped below threshold".into(),
            uncertainty_events: vec![UncertaintyEvent {
                task_id: TaskId("task-1".into()),
                stage: None,
                kind: UncertaintyKind::LowConfidence,
                summary: "too many unresolved ambiguities".into(),
                confidence: Some(0.32),
                created_at: now,
            }],
            created_at: now,
            updated_at: now,
        };

        store.put_frontier_case(&frontier_case).unwrap();

        let cases = store.list_frontier_cases_for_task("task-1").unwrap();
        assert_eq!(cases, vec![frontier_case]);
    }

    #[test]
    fn ledger_append_and_list_in_sequence_order() {
        use shunt_core::ledger::{ActionRecord, LedgerEntry};

        let store = SqliteStore::open_in_memory().unwrap();

        let e1 = store
            .append_ledger_entry(
                "task-1",
                LedgerEntry::ModelAction(ActionRecord {
                    call_id: None,
                    phase: "clarify".into(),
                    tool: "clarify_node".into(),
                    elapsed_ms: 300,
                    outcome: "valid".into(),
                    summary: "goal clarified".into(),
                }),
            )
            .unwrap();
        let e2 = store
            .append_ledger_entry(
                "task-1",
                LedgerEntry::ModelAction(ActionRecord {
                    call_id: None,
                    phase: "understand".into(),
                    tool: "understand_node".into(),
                    elapsed_ms: 500,
                    outcome: "valid".into(),
                    summary: "scope understood".into(),
                }),
            )
            .unwrap();

        let entries = store.list_ledger_entries("task-1").unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].sequence, 0);
        assert_eq!(entries[0].id, e1.id);
        assert_eq!(entries[1].sequence, 1);
        assert_eq!(entries[1].id, e2.id);
        assert!(
            matches!(entries[0].entry, LedgerEntry::ModelAction(ref a) if a.phase == "clarify")
        );
    }

    #[test]
    fn ledger_entries_are_isolated_per_task() {
        use shunt_core::ledger::{ContextSummary, LedgerEntry};

        let store = SqliteStore::open_in_memory().unwrap();

        store
            .append_ledger_entry(
                "task-A",
                LedgerEntry::ContextSummary(ContextSummary {
                    covering_entry_ids: vec![],
                    summary: "summary for A".into(),
                    durable_facts: vec![],
                }),
            )
            .unwrap();
        store
            .append_ledger_entry(
                "task-B",
                LedgerEntry::ContextSummary(ContextSummary {
                    covering_entry_ids: vec![],
                    summary: "summary for B".into(),
                    durable_facts: vec![],
                }),
            )
            .unwrap();

        assert_eq!(store.list_ledger_entries("task-A").unwrap().len(), 1);
        assert_eq!(store.list_ledger_entries("task-B").unwrap().len(), 1);
        assert!(store.list_ledger_entries("task-C").unwrap().is_empty());
    }
}
