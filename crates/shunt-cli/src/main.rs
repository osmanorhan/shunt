use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use serde::{Deserialize, Serialize};
use shunt_core::machine::{
    ArtifactSnapshot, AutonomyPolicy, Command as MachineCommand, Notification, TaskState,
    UserRequest,
};
use shunt_core::{
    AmbiguityId, ArtifactId, ChangeSet, CorrectionPackage, FileOp, FrontierReason, RecipeRef,
    RecipeRun, TaskRun, UncertaintyEvent, UncertaintyKind,
};
use shunt_infer::{OpenAiCompatProvider, ProviderCapabilities, SessionBudgetOverride};
use shunt_runtime::session::{SessionHandle, spawn_session};
use shunt_runtime::{ArtifactUpdate, TaskRuntime};
use shunt_store::{SqliteStore, StoreError};
use thiserror::Error;
use time::OffsetDateTime;
use tokio::sync::broadcast;

#[derive(Debug, Error)]
enum CliError {
    #[error("{0}")]
    Store(#[from] StoreError),
    #[error("{0}")]
    Runtime(#[from] shunt_runtime::RuntimeError),
    #[error("task not found: {0}")]
    TaskNotFound(String),
    #[error("artifact not found: {0}")]
    ArtifactNotFound(String),
    #[error("change content or search/replace is required")]
    MissingChangeContent,
    #[error("search and replacement must be provided together")]
    IncompleteSearchReplace,
    #[error("db parent directory is missing")]
    MissingDbParent,
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("config parse error: {0}")]
    ConfigParse(#[from] toml::de::Error),
    #[error("config write error: {0}")]
    ConfigSerialize(#[from] toml::ser::Error),
    #[error("config file not found: {0}")]
    ConfigMissing(String),
    #[error("model is not configured; set `model` in .shunt/config.toml")]
    MissingModel,
    #[error("terminal error: {0}")]
    Terminal(String),
}

#[derive(Parser)]
#[command(name = "shunt")]
#[command(about = "AI coding assistant")]
struct Cli {
    #[arg(long, global = true)]
    cwd: Option<PathBuf>,
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    trace: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Agent {
        prompt: Option<String>,
        #[arg(long)]
        once: bool,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    Task {
        #[command(subcommand)]
        command: TaskCommand,
    },
    Artifact {
        #[command(subcommand)]
        command: ArtifactCommand,
    },
    Frontier {
        #[command(subcommand)]
        command: FrontierCommand,
    },
    Correction {
        #[command(subcommand)]
        command: CorrectionCommand,
    },
}

#[derive(Subcommand)]
enum ConfigCommand {
    Show,
    Init {
        #[arg(long)]
        force: bool,
    },
}

#[derive(Subcommand)]
enum TaskCommand {
    Start {
        #[arg(long)]
        request: String,
        #[arg(long)]
        task_id: Option<String>,
        #[arg(long)]
        artifact_id: Option<String>,
    },
    Show {
        #[arg(long)]
        task_id: String,
    },
    Clarify {
        #[arg(long)]
        artifact_id: String,
    },
    Understand {
        #[arg(long)]
        artifact_id: String,
        #[arg(long)]
        heuristic_only: bool,
    },
    Localize {
        #[arg(long)]
        artifact_id: String,
    },
    ExecuteStart {
        #[arg(long)]
        task_id: String,
    },
    ExecuteInspect {
        #[arg(long)]
        task_id: String,
    },
    ExecutePropose {
        #[arg(long)]
        task_id: String,
    },
    ExecuteVerify {
        #[arg(long)]
        task_id: String,
    },
    ExecuteGenerateChange {
        #[arg(long)]
        task_id: String,
    },
    ExecuteAddChange {
        #[arg(long)]
        task_id: String,
        #[arg(long)]
        path: String,
        #[arg(long)]
        description: String,
        #[arg(long)]
        search: Option<String>,
        #[arg(long)]
        replacement: Option<String>,
        #[arg(long)]
        content: Option<String>,
        #[arg(long)]
        content_file: Option<PathBuf>,
    },
    ExecuteApply {
        #[arg(long)]
        task_id: String,
    },
    ExecuteValidate {
        #[arg(long)]
        task_id: String,
    },
}

#[derive(Subcommand)]
enum ArtifactCommand {
    Revise {
        #[arg(long)]
        artifact_id: String,
        #[arg(long)]
        goal: Option<String>,
        #[arg(long = "success")]
        success_criteria: Vec<String>,
        #[arg(long = "constraint")]
        constraints: Vec<String>,
        #[arg(long = "scope")]
        target_scope: Vec<String>,
        #[arg(long)]
        confidence: Option<f32>,
    },
    Resolve {
        #[arg(long)]
        artifact_id: String,
        #[arg(long)]
        ambiguity_id: String,
        #[arg(long)]
        resolution: String,
    },
    Approve {
        #[arg(long)]
        artifact_id: String,
        #[arg(long)]
        note: Option<String>,
    },
}

#[derive(Subcommand)]
enum FrontierCommand {
    Add {
        #[arg(long)]
        task_id: String,
        #[arg(long)]
        artifact_id: String,
        #[arg(long)]
        summary: String,
        #[arg(long)]
        reason: FrontierReasonArg,
        #[arg(long)]
        event_kind: UncertaintyKindArg,
        #[arg(long)]
        event_summary: String,
        #[arg(long)]
        event_confidence: Option<f32>,
        #[arg(long)]
        frontier_case_id: Option<String>,
    },
}

#[derive(Subcommand)]
enum CorrectionCommand {
    Patch {
        #[arg(long)]
        frontier_case_id: String,
        #[arg(long)]
        summary: String,
        #[arg(long)]
        path: String,
        #[arg(long)]
        description: String,
        #[arg(long)]
        search: Option<String>,
        #[arg(long)]
        replacement: Option<String>,
        #[arg(long)]
        content: Option<String>,
        #[arg(long)]
        content_file: Option<PathBuf>,
        #[arg(long)]
        correction_id: Option<String>,
    },
    Replay {
        #[arg(long)]
        correction_id: String,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FrontierReasonArg {
    LowConfidence,
    VerifierFailure,
    RepeatedVerifierFailure,
    RecipeInstability,
    ToolChurn,
    MaterialUserCorrection,
    RepeatedPatchFailure,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum UncertaintyKindArg {
    Ambiguity,
    MissingEvidence,
    LowConfidence,
    VerifierFailure,
    VerifierDisagreement,
    RetryExhausted,
    RecipeOscillation,
    ToolThrash,
    UserCorrection,
    PatchRejection,
}

#[derive(Debug, Clone, Serialize)]
struct TaskView {
    task: TaskRun,
    artifact: shunt_core::UnderstandingArtifact,
    active_recipe_run: Option<RecipeRun>,
    frontier_cases: Vec<shunt_core::FrontierCase>,
}

#[derive(Debug, Serialize)]
struct CorrectionView {
    correction: CorrectionPackage,
    frontier_case: shunt_core::FrontierCase,
    recipe_run: RecipeRun,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct AppConfig {
    db: String,
    endpoint: String,
    model: Option<String>,
    timeout_secs: u64,
    recipe_id: String,
    recipe_version: String,
    decided_by: String,
    /// Extra gitignore-style patterns added on top of the built-in defaults.
    #[serde(default)]
    ignore_patterns: Vec<String>,
    /// Optional per-project agent budget overrides.
    /// Set in .shunt/config.toml under [agent]. Unset fields keep the model-derived default.
    #[serde(default)]
    agent: SessionBudgetOverride,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            db: ".shunt/shunt.db".into(),
            endpoint: "http://127.0.0.1:8080".into(),
            model: None,
            timeout_secs: 300,
            recipe_id: "manual.inspect-propose".into(),
            recipe_version: "v1".into(),
            decided_by: "user".into(),
            ignore_patterns: Vec::new(),
            agent: SessionBudgetOverride::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct AppContext {
    workspace_root: PathBuf,
    config_path: PathBuf,
    config: AppConfig,
}

impl AppContext {
    fn db_path(&self) -> PathBuf {
        resolve_path(&self.workspace_root, Path::new(&self.config.db))
    }

    fn provider(&self) -> Result<OpenAiCompatProvider, CliError> {
        let model = self.config.model.clone().ok_or(CliError::MissingModel)?;
        let caps = ProviderCapabilities::detect(&model, &self.config.endpoint);
        OpenAiCompatProvider::with_timeout(
            self.config.endpoint.clone(),
            model,
            Duration::from_secs(self.config.timeout_secs),
        )
        .map(|provider| provider.with_capabilities(caps))
        .map_err(shunt_runtime::RuntimeError::from)
        .map_err(CliError::from)
    }

    fn recipe(&self) -> RecipeRef {
        RecipeRef {
            id: self.config.recipe_id.clone(),
            version: self.config.recipe_version.clone(),
        }
    }

    fn init_config(&self, force: bool) -> Result<(), CliError> {
        if self.config_path.exists() && !force {
            return Err(CliError::Terminal(format!(
                "config already exists at {}; rerun with --force to overwrite",
                self.config_path.display()
            )));
        }
        write_default_config(&self.config_path)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum SessionEntryKind {
    User,
    Frame,
    Progress,
    Question,
    Error,
    /// Unified diff block — body lines prefixed with `+`, `-`, or ` `.
    Diff,
    /// Agent tool invocation — blue arrow prefix.
    ToolCall,
    /// Successful tool result — green check prefix.
    ToolOk,
    /// Failed tool result — red x prefix.
    ToolErr,
    /// Setup command line — yellow dollar prefix.
    SetupCmd,
    /// Task successfully completed — bright green check.
    Done,
}

#[derive(Debug, Clone)]
struct SessionEntry {
    kind: SessionEntryKind,
    body: String,
}

#[derive(Debug, Clone)]
struct PendingQuestion {
    ambiguity_id: String,
}

#[derive(Debug, Clone)]
enum PendingApprovalKind {
    Plan,
    Commands,
}

#[derive(Debug, Clone)]
struct PendingApproval {
    kind: PendingApprovalKind,
}

struct AgentApp {
    input: String,
    /// Byte offset of the cursor in `input`. Invariant: always a valid char boundary.
    cursor_pos: usize,
    entries: Vec<SessionEntry>,
    /// Number of rendered log lines to stay above the live tail.
    log_scroll_from_bottom: usize,
    status: String,
    active_run: bool,
    /// When the current run started — used to show elapsed time in the right pane.
    run_started_at: Option<Instant>,
    machine_state: Option<TaskState>,
    pending_question: Option<PendingQuestion>,
    pending_approval: Option<PendingApproval>,
    /// Latest agent reasoning snapshot — updated after each LLM call.
    reasoning: Option<ReasoningSnapshot>,
    /// Live agent-turn state — shown in the right pane during AgentSession runs.
    agent_live: Option<AgentLiveState>,
    /// Files successfully written in the last completed task. Passed as prior context to the next session.
    last_session_files: Vec<String>,
    /// A git stash was created before the last apply — undo is available via `u`.
    can_undo: bool,
    /// Staged changes are ready to commit — shows `[g] commit` hint.
    can_commit: bool,
    /// Description from the last ChangeProposed notification — used as commit message.
    last_task_description: Option<String>,
    /// Current git branch + dirty marker, e.g. "main" or "main*". None = not a git repo.
    git_branch: Option<String>,
    /// Live session — `None` between tasks.
    session: Option<SessionHandle>,
    /// Notification stream from the active session.
    notif_rx: Option<broadcast::Receiver<Notification>>,
    /// Tokio runtime kept alive for the duration of the process.
    tokio_rt: Arc<tokio::runtime::Runtime>,
    context: AppContext,
    /// Call ID of the in-progress model call (for deduplication across retries).
    streaming_call_id: Option<u64>,
    /// Actual structured output tokens (content / tool-call args) — shown in full.
    streaming_buf: String,
    /// Character count of reasoning_content tokens received so far.
    /// Shown as a compact "[thinking... N chars]" line instead of full text.
    thinking_chars: usize,
}

/// Live state updated on every AgentToolCall/Result — shown in the right pane.
#[derive(Debug, Clone)]
struct AgentLiveState {
    turn: usize,
    max_turns: usize,
    last_tool: String,
    last_summary: String,
    files_read: Vec<String>,
    files_written: Vec<String>,
    pending_write: Option<String>,
    last_ok: bool,
    last_detail: String,
}

/// Rendered form of the agent's current understanding, shown in the sidebar.
#[derive(Debug, Clone)]
struct ReasoningSnapshot {
    goal: String,
    confidence_pct: u8,
    evidence_count: usize,
    candidate_paths: Vec<String>,
    open_ambiguity_count: usize,
    open_risks: Vec<String>,
}

impl AgentApp {
    fn new(context: AppContext, tokio_rt: Arc<tokio::runtime::Runtime>) -> Self {
        Self {
            input: String::new(),
            cursor_pos: 0,
            entries: vec![SessionEntry {
                kind: SessionEntryKind::Frame,
                body: format!(
                    "Frame ready  model={}  endpoint={}",
                    context.config.model.as_deref().unwrap_or("<unset>"),
                    context.config.endpoint
                ),
            }],
            log_scroll_from_bottom: 0,
            status: "idle".into(),
            active_run: false,
            run_started_at: None,
            machine_state: None,
            pending_question: None,
            pending_approval: None,
            reasoning: None,
            agent_live: None,
            last_session_files: Vec::new(),
            can_undo: false,
            can_commit: false,
            last_task_description: None,
            git_branch: None,
            session: None,
            notif_rx: None,
            tokio_rt,
            context,
            streaming_call_id: None,
            streaming_buf: String::new(),
            thinking_chars: 0,
        }
    }

    fn push(&mut self, kind: SessionEntryKind, body: impl Into<String>) {
        self.entries.push(SessionEntry {
            kind: kind.clone(),
            body: body.into(),
        });
        // Scroll to bottom for anything the user must not miss.
        // Progress entries during streaming don't scroll (user may be reading history).
        match kind {
            SessionEntryKind::Question
            | SessionEntryKind::Error
            | SessionEntryKind::Frame
            | SessionEntryKind::Done => {
                self.log_scroll_from_bottom = 0;
            }
            _ => {}
        }
    }

    fn scroll_log_up(&mut self, lines: usize) {
        self.log_scroll_from_bottom = self.log_scroll_from_bottom.saturating_add(lines);
    }

    fn scroll_log_down(&mut self, lines: usize) {
        self.log_scroll_from_bottom = self.log_scroll_from_bottom.saturating_sub(lines);
    }
}

type AppTerminal = Terminal<CrosstermBackend<Stdout>>;

struct TerminalSession {
    terminal: AppTerminal,
}

impl TerminalSession {
    fn enter() -> Result<Self, CliError> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout)).map_err(CliError::Io)?;
        Ok(Self { terminal })
    }

    fn terminal_mut(&mut self) -> &mut AppTerminal {
        &mut self.terminal
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        let _ = self.terminal.show_cursor();
    }
}

/// Initialise file-based tracing.  Writes to `.shunt/debug.log` in the workspace.
/// Every LLM prompt/response, tool call, and agent turn is captured here.
/// To also see output in the terminal: RUST_LOG=debug shunt ...
fn init_debug_log(context: &AppContext) {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let log_path = context.workspace_root.join(".shunt/debug.log");

    // Rotate: rename old log so each run starts fresh but the previous is kept.
    let prev = context.workspace_root.join(".shunt/debug.log.prev");
    if log_path.exists() {
        let _ = std::fs::rename(&log_path, &prev);
    }

    let file = match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
    {
        Ok(f) => f,
        Err(_) => return, // can't open log; continue without it
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"));

    let file_layer = fmt::layer()
        .with_writer(std::sync::Mutex::new(file))
        .with_ansi(false)
        .with_target(true)
        .with_thread_ids(false)
        .with_level(true);

    // Only initialise if no subscriber is already set (allow RUST_LOG override).
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(file_layer)
        .try_init();
}

fn main() -> Result<(), CliError> {
    let cli = Cli::parse();
    let auto_create_config = !matches!(
        &cli.command,
        Some(Command::Config {
            command: ConfigCommand::Init { .. }
        })
    );
    let context = load_context(cli.cwd, cli.config, auto_create_config)?;
    ensure_db_parent(&context.db_path())?;
    init_debug_log(&context);

    match cli.command {
        None => run_agent_session(context, None, false)?,
        Some(Command::Agent { prompt, once }) => run_agent_session(context, prompt, once)?,
        Some(Command::Config { command }) => handle_config(command, &context)?,
        Some(Command::Task { command }) => handle_task(command, &context)?,
        Some(Command::Artifact { command }) => handle_artifact(command, &context)?,
        Some(Command::Frontier { command }) => handle_frontier(command, &context)?,
        Some(Command::Correction { command }) => handle_correction(command, &context)?,
    }

    Ok(())
}

fn is_home_directory(path: &Path) -> bool {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .and_then(|home| home.canonicalize().ok())
        .is_some_and(|home| home == path)
}

fn load_context(
    cwd_override: Option<PathBuf>,
    config_override: Option<PathBuf>,
    auto_create_config: bool,
) -> Result<AppContext, CliError> {
    let workspace_root = canonicalize_or_current(cwd_override)?;

    if is_home_directory(&workspace_root) {
        return Err(CliError::Terminal(
            "refusing to use your home directory as the workspace — \
             cd into a project directory first, or pass --cwd <project-path>"
                .into(),
        ));
    }
    let has_explicit_config = config_override.is_some();
    let config_path = config_override
        .map(|path| resolve_path(&workspace_root, &path))
        .unwrap_or_else(|| workspace_root.join(".shunt/config.toml"));
    let default_config_path = workspace_root.join(".shunt/config.toml");

    if auto_create_config && !has_explicit_config && !default_config_path.exists() {
        write_default_config(&default_config_path)?;
    }

    let config = if config_path.exists() {
        toml::from_str(&std::fs::read_to_string(&config_path)?)?
    } else {
        AppConfig::default()
    };

    Ok(AppContext {
        workspace_root,
        config_path,
        config,
    })
}

/// Read current git branch + dirty state from workspace_root. Returns e.g. "main" or "main*".
fn read_git_branch(workspace_root: &std::path::Path) -> Option<String> {
    let branch = std::process::Command::new("git")
        .args([
            "-C",
            &workspace_root.display().to_string(),
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())?;

    let dirty = std::process::Command::new("git")
        .args([
            "-C",
            &workspace_root.display().to_string(),
            "status",
            "--porcelain",
        ])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    Some(if dirty { format!("{branch}*") } else { branch })
}

/// Probe the configured endpoint and common alternatives; push suggestions into the TUI.
fn probe_and_suggest(app: &mut AgentApp) {
    app.git_branch = read_git_branch(&app.context.workspace_root.clone());
    let configured = app.context.config.endpoint.clone();
    let model_set = app.context.config.model.is_some();

    // Candidate ports to probe (configured port is always first).
    let configured_port = configured
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .split(':')
        .nth(1)
        .and_then(|p| p.split('/').next())
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8080);

    let probe_ports: Vec<u16> = [8080u16, 8081, 11434]
        .iter()
        .copied()
        .filter(|&p| p != configured_port)
        .collect();

    if shunt_infer::probe_endpoint(&configured) {
        // Configured endpoint is up. Auto-set model if not already configured.
        if !model_set {
            match shunt_infer::list_models(&configured) {
                Some(models) if !models.is_empty() => {
                    let model = &models[0];
                    if let Err(e) = patch_model_in_config(&app.context.config_path, model) {
                        app.push(
                            SessionEntryKind::Frame,
                            format!("Detected model {model} but couldn't write config: {e}"),
                        );
                    } else {
                        app.context.config.model = Some(model.clone());
                        app.push(
                            SessionEntryKind::Frame,
                            format!(
                                "Auto-configured model = \"{model}\" (saved to .shunt/config.toml)"
                            ),
                        );
                    }
                }
                _ => {
                    app.push(
                        SessionEntryKind::Frame,
                        "No model configured. Add model = \"<id>\" to .shunt/config.toml"
                            .to_string(),
                    );
                }
            }
        }
    } else {
        // Configured endpoint is down. Probe alternatives.
        let mut found: Vec<(String, Option<Vec<String>>)> = Vec::new();
        for port in probe_ports {
            let alt = format!("http://127.0.0.1:{port}");
            if shunt_infer::probe_endpoint(&alt) {
                let models = shunt_infer::list_models(&alt);
                found.push((alt, models));
            }
        }

        if found.is_empty() {
            app.push(
                SessionEntryKind::Frame,
                format!(
                    "No LLM server reachable at {} or common ports (8080, 8081, 11434). Start llama.cpp/Ollama or update endpoint in .shunt/config.toml.",
                    configured
                ),
            );
        } else {
            app.push(
                SessionEntryKind::Frame,
                format!("Configured endpoint {} is not reachable.", configured),
            );
            for (alt, models) in &found {
                let model_hint = match models {
                    Some(m) if !m.is_empty() => format!(" (models: {})", m.join(", ")),
                    _ => String::new(),
                };
                app.push(
                    SessionEntryKind::Frame,
                    format!(
                        "Found LLM server at {}{} — update .shunt/config.toml: endpoint = \"{}\"",
                        alt, model_hint, alt
                    ),
                );
                if !model_set
                    && let Some(models) = models
                    && let Some(model) = models.first()
                {
                    app.push(
                        SessionEntryKind::Frame,
                        format!("  Set model = \"{model}\" in .shunt/config.toml"),
                    );
                }
            }
        }
    }
}

fn run_agent_session(
    context: AppContext,
    initial_prompt: Option<String>,
    once: bool,
) -> Result<(), CliError> {
    if once {
        if let Some(prompt) = initial_prompt {
            let view = run_handle_request(&context, prompt)?;
            return print_task_view(&view);
        }
        return Err(CliError::Terminal("agent --once requires a prompt".into()));
    }

    let tokio_rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .map_err(|e| CliError::Terminal(format!("tokio runtime: {e}")))?,
    );
    let mut app = AgentApp::new(context, Arc::clone(&tokio_rt));
    probe_and_suggest(&mut app);
    let mut terminal = TerminalSession::enter()?;

    if let Some(prompt) = initial_prompt {
        submit_prompt(&mut app, prompt);
    }

    loop {
        drain_session(&mut app);
        draw_agent(terminal.terminal_mut(), &app)?;
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        match event::read()? {
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => app.scroll_log_up(3),
                MouseEventKind::ScrollDown => app.scroll_log_down(3),
                _ => {}
            },
            Event::Key(key) => {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match (key.code, key.modifiers) {
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                    (KeyCode::Esc, _) => {
                        app.input.clear();
                        app.cursor_pos = 0;
                        app.status = "Input cleared".into();
                    }
                    (KeyCode::Char('q'), KeyModifiers::NONE) if app.input.is_empty() => break,
                    (KeyCode::Char('u'), KeyModifiers::NONE)
                        if app.input.is_empty() && app.can_undo && !app.active_run =>
                    {
                        let workspace = app.context.workspace_root.display().to_string();
                        match shunt_runtime::runner::git_stash_pop(&workspace) {
                            Ok(msg) => {
                                app.can_undo = false;
                                app.can_commit = false;
                                app.last_session_files.clear();
                                let note = if msg.is_empty() {
                                    "Undo applied.".into()
                                } else {
                                    format!("Undo: {msg}")
                                };
                                app.push(SessionEntryKind::Frame, note);
                                app.git_branch =
                                    read_git_branch(&app.context.workspace_root.clone());
                            }
                            Err(e) => {
                                app.push(SessionEntryKind::Error, format!("Undo failed: {e}"));
                            }
                        }
                    }
                    (KeyCode::Char('g'), KeyModifiers::NONE)
                        if app.input.is_empty() && app.can_commit && !app.active_run =>
                    {
                        let workspace = app.context.workspace_root.display().to_string();
                        let msg = app
                            .last_task_description
                            .clone()
                            .unwrap_or_else(|| "shunt: apply changes".to_string());
                        let result = std::process::Command::new("git")
                            .args(["-C", &workspace, "add", "-A"])
                            .output()
                            .and_then(|_| {
                                std::process::Command::new("git")
                                    .args(["-C", &workspace, "commit", "-m", &msg])
                                    .output()
                            });
                        match result {
                            Ok(out) if out.status.success() => {
                                app.can_commit = false;
                                app.can_undo = false;
                                let commit_out =
                                    String::from_utf8_lossy(&out.stdout).trim().to_string();
                                app.push(
                                    SessionEntryKind::Frame,
                                    format!("Committed: {commit_out}"),
                                );
                                app.git_branch =
                                    read_git_branch(&app.context.workspace_root.clone());
                            }
                            Ok(out) => {
                                let stderr =
                                    String::from_utf8_lossy(&out.stderr).trim().to_string();
                                app.push(
                                    SessionEntryKind::Error,
                                    format!("git commit failed: {stderr}"),
                                );
                            }
                            Err(e) => {
                                app.push(SessionEntryKind::Error, format!("git commit error: {e}"));
                            }
                        }
                    }
                    (KeyCode::PageUp, _) => app.scroll_log_up(10),
                    (KeyCode::PageDown, _) => app.scroll_log_down(10),
                    (KeyCode::Home, KeyModifiers::CONTROL) => {
                        app.log_scroll_from_bottom = usize::MAX;
                    }
                    (KeyCode::End, KeyModifiers::CONTROL) => {
                        app.log_scroll_from_bottom = 0;
                    }
                    // Cursor movement within the input box.
                    (KeyCode::Left, _) => {
                        app.cursor_pos = prev_char_boundary(&app.input, app.cursor_pos);
                    }
                    (KeyCode::Right, _) => {
                        app.cursor_pos = next_char_boundary(&app.input, app.cursor_pos);
                    }
                    (KeyCode::Home, KeyModifiers::NONE) => {
                        let before = &app.input[..app.cursor_pos];
                        app.cursor_pos = match before.rfind('\n') {
                            Some(nl) => nl + 1,
                            None => 0,
                        };
                    }
                    (KeyCode::End, KeyModifiers::NONE) => {
                        app.cursor_pos = match app.input[app.cursor_pos..].find('\n') {
                            Some(nl) => app.cursor_pos + nl,
                            None => app.input.len(),
                        };
                    }
                    (KeyCode::Delete, _) if app.cursor_pos < app.input.len() => {
                        let end = next_char_boundary(&app.input, app.cursor_pos);
                        app.input.drain(app.cursor_pos..end);
                    }
                    (KeyCode::Enter, KeyModifiers::ALT) => {
                        app.input.insert(app.cursor_pos, '\n');
                        app.cursor_pos += 1;
                    }
                    (KeyCode::Enter, _) => {
                        let prompt = app.input.trim().to_string();
                        app.input.clear();
                        app.cursor_pos = 0;
                        if !prompt.is_empty() {
                            submit_prompt(&mut app, prompt);
                        }
                    }
                    (KeyCode::Backspace, _) if app.cursor_pos > 0 => {
                        let new_pos = prev_char_boundary(&app.input, app.cursor_pos);
                        app.input.drain(new_pos..app.cursor_pos);
                        app.cursor_pos = new_pos;
                    }
                    (KeyCode::Char(ch), _) => {
                        app.input.insert(app.cursor_pos, ch);
                        app.cursor_pos += ch.len_utf8();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn prev_char_boundary(s: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut p = pos - 1;
    while p > 0 && !s.is_char_boundary(p) {
        p -= 1;
    }
    p
}

fn next_char_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    let mut p = pos + 1;
    while p < s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p
}

/// Returns (col, row) in terminal cells for the given byte offset in a multi-line string.
fn cursor_visual_pos(input: &str, cursor_pos: usize) -> (u16, u16) {
    let before = &input[..cursor_pos.min(input.len())];
    let row = before.chars().filter(|&c| c == '\n').count() as u16;
    let col = match before.rfind('\n') {
        Some(nl) => (before.len() - nl - 1) as u16,
        None => before.len() as u16,
    };
    (col, row)
}

fn submit_prompt(app: &mut AgentApp, prompt: String) {
    // Handle slash commands before routing.
    if let Some(rest) = prompt.strip_prefix('/').or_else(|| {
        if prompt.trim() == "?" {
            Some("help")
        } else {
            None
        }
    }) {
        handle_slash_command(app, rest.trim());
        return;
    }

    let tx = app.session.as_ref().map(|s| s.commands.clone());

    // Route to clarification answer.
    if let Some(question) = app.pending_question.take() {
        app.push(SessionEntryKind::User, prompt.clone());
        if let Some(tx) = tx {
            let cmd = MachineCommand::Answer {
                ambiguity_id: AmbiguityId(question.ambiguity_id.clone()),
                answer: prompt,
            };
            if let Err(e) = tx.blocking_send(cmd) {
                app.push(SessionEntryKind::Error, format!("send error: {e}"));
            }
        } else {
            app.push(SessionEntryKind::Error, String::from("session lost"));
        }
        return;
    }

    // Route to approval/rejection.
    if let Some(approval) = app.pending_approval.take() {
        let lower = prompt.trim().to_lowercase();
        let approved = matches!(lower.as_str(), "y" | "yes" | "approve");
        let rejected = matches!(lower.as_str(), "n" | "no" | "reject");
        if !approved && !rejected {
            app.push(
                SessionEntryKind::Frame,
                String::from("Type 'y' to approve or 'n' to reject."),
            );
            app.pending_approval = Some(approval);
            return;
        }
        app.push(SessionEntryKind::User, prompt);
        if let Some(tx) = tx {
            let cmd = match approval.kind {
                PendingApprovalKind::Plan => {
                    if approved {
                        MachineCommand::Approve
                    } else {
                        MachineCommand::Reject
                    }
                }
                PendingApprovalKind::Commands => {
                    if approved {
                        MachineCommand::ApproveDangerousCommands
                    } else {
                        MachineCommand::RejectDangerousCommands
                    }
                }
            };
            if let Err(e) = tx.blocking_send(cmd) {
                app.push(SessionEntryKind::Error, format!("send error: {e}"));
            }
        } else {
            app.push(SessionEntryKind::Error, String::from("session lost"));
        }
        return;
    }

    if app.active_run {
        app.status = "Run already in progress".into();
        return;
    }

    // Start a brand-new session for this prompt.
    let now = OffsetDateTime::now_utc();
    let task_id = format!("task-{}", now.unix_timestamp_nanos());
    let artifact_id = ArtifactId(format!("artifact-{}", now.unix_timestamp_nanos()));
    let store_path = app.context.db_path().display().to_string();
    let workspace_root = app.context.workspace_root.display().to_string();
    let recipe = app.context.recipe();
    let policy = AutonomyPolicy::agentic();

    let provider = match app.context.provider() {
        Ok(p) => p,
        Err(e) => {
            app.push(SessionEntryKind::Error, format!("provider: {e}"));
            return;
        }
    };

    let prior_context_files = app.last_session_files.clone();
    let ignore_patterns = app.context.config.ignore_patterns.clone();
    let handle = app.tokio_rt.block_on(async {
        spawn_session(
            task_id,
            artifact_id,
            workspace_root,
            recipe,
            store_path,
            provider,
            policy,
            tokio::runtime::Handle::current(),
            prior_context_files,
            ignore_patterns,
        )
    });
    let notif_rx = handle.subscribe();

    // Kick off the task.
    if let Err(e) = handle.commands.blocking_send(MachineCommand::Submit {
        request: prompt.clone(),
    }) {
        app.push(SessionEntryKind::Error, format!("submit error: {e}"));
        return;
    }

    app.push(SessionEntryKind::User, prompt);
    app.active_run = true;
    app.run_started_at = Some(Instant::now());
    app.can_undo = false;
    app.can_commit = false;
    app.machine_state = Some(TaskState::Running);
    app.pending_question = None;
    app.pending_approval = None;
    app.status = String::from("running");
    app.session = Some(handle);
    app.notif_rx = Some(notif_rx);
}

fn handle_slash_command(app: &mut AgentApp, cmd: &str) {
    if app.active_run {
        app.push(
            SessionEntryKind::Frame,
            "Slash commands are not available while a run is in progress.".to_string(),
        );
        return;
    }

    let (name, arg) = cmd
        .split_once(' ')
        .map(|(a, b)| (a, b.trim()))
        .unwrap_or((cmd, ""));

    match name {
        "clear" => {
            app.entries.clear();
            app.agent_live = None;
            app.reasoning = None;
            app.can_undo = false;
            app.last_session_files.clear();
            app.machine_state = None;
            app.push(
                SessionEntryKind::Frame,
                format!(
                    "Session cleared  model={}  endpoint={}",
                    app.context.config.model.as_deref().unwrap_or("<unset>"),
                    app.context.config.endpoint
                ),
            );
        }
        "model" => {
            if arg.is_empty() {
                let current = app.context.config.model.as_deref().unwrap_or("<unset>");
                app.push(
                    SessionEntryKind::Frame,
                    format!("Current model: {current}  (usage: /model <id>)"),
                );
            } else {
                app.context.config.model = Some(arg.to_string());
                app.push(SessionEntryKind::Frame, format!("Model set to: {arg}  (in-session only; update .shunt/config.toml to persist)"));
            }
        }
        "endpoint" => {
            if arg.is_empty() {
                app.push(
                    SessionEntryKind::Frame,
                    format!(
                        "Current endpoint: {}  (usage: /endpoint <url>)",
                        app.context.config.endpoint
                    ),
                );
            } else {
                app.context.config.endpoint = arg.to_string();
                app.push(SessionEntryKind::Frame, format!("Endpoint set to: {arg}  (in-session only; update .shunt/config.toml to persist)"));
            }
        }
        "help" | "?" => {
            app.push(
                SessionEntryKind::Frame,
                "/clear              — clear session history\n\
                 /model <id>         — set model for this session\n\
                 /endpoint <url>     — set LLM endpoint for this session\n\
                 /help or ?          — show this help\n\
                 u (idle)            — undo last applied changes (git stash pop)\n\
                 q (idle)            — quit"
                    .to_string(),
            );
        }
        other => {
            app.push(
                SessionEntryKind::Frame,
                format!("Unknown command: /{other}  (type /help for available commands)"),
            );
        }
    }
}

/// Drain notifications from the live session and update app state.
fn drain_session(app: &mut AgentApp) {
    // Drain notifications first — they may carry question text before state changes.
    let mut notifs: Vec<Notification> = Vec::new();
    let mut notif_closed = false;
    let mut lagged_count = 0u64;
    if let Some(rx) = app.notif_rx.as_mut() {
        loop {
            match rx.try_recv() {
                Ok(notif) => notifs.push(notif),
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(count)) => {
                    lagged_count = lagged_count.saturating_add(count);
                }
                Err(broadcast::error::TryRecvError::Closed) => {
                    notif_closed = true;
                    break;
                }
            }
        }
    }
    if notif_closed {
        app.notif_rx = None;
    }
    if lagged_count > 0 {
        app.push(
            SessionEntryKind::Error,
            format!("notification stream lagged; {lagged_count} event(s) were dropped"),
        );
    }
    for notif in notifs {
        apply_notification(app, notif);
    }

    // Check live TaskState from the watch channel.
    if let Some(session) = &app.session {
        let current = session.current_state();
        let prev = app.machine_state.clone();

        if prev.as_ref() != Some(&current) {
            app.machine_state = Some(current.clone());
            app.status = current.label().to_string();

            match &current {
                TaskState::WaitingForUser {
                    request: UserRequest::Clarification { .. },
                } => {
                    // pending_question is populated by ClarificationNeeded notification.
                }
                TaskState::WaitingForUser {
                    request: UserRequest::Approval { .. },
                } => {
                    app.pending_question = None;
                }
                TaskState::WaitingForUser {
                    request: UserRequest::DangerousCommands { commands, reason },
                } => {
                    app.pending_question = None;
                    if app.pending_approval.is_none() {
                        let display = commands
                            .iter()
                            .map(|c| c.display())
                            .collect::<Vec<_>>()
                            .join("\n  ");
                        let r = reason.clone();
                        app.push(
                            SessionEntryKind::Question,
                            format!("⚠ Dangerous commands ({r}):\n  {display}\nType 'y' to run, 'n' to skip."),
                        );
                        app.pending_approval = Some(PendingApproval {
                            kind: PendingApprovalKind::Commands,
                        });
                    }
                }
                _ => {
                    app.pending_question = None;
                    app.pending_approval = None;
                }
            }

            if current.is_terminal() {
                use shunt_core::machine::StopReason;
                if matches!(current, TaskState::Completed) {
                    app.push(SessionEntryKind::Done, String::from("Task completed."));
                } else if let TaskState::Stopped { reason } = &current {
                    let msg = match reason {
                        StopReason::Failed { reason } => format!("Failed: {reason}"),
                        StopReason::Cancelled => "Cancelled.".into(),
                        StopReason::FrontierRaised { case } => {
                            format!("Frontier raised: {}", case.0)
                        }
                    };
                    app.push(SessionEntryKind::Error, msg);
                }
                app.active_run = false;
                app.run_started_at = None;
                // Save written files for the next session's prior context.
                if let Some(live) = app.agent_live.take() {
                    if !live.files_written.is_empty() {
                        app.last_session_files = live.files_written;
                    }
                } else {
                    app.agent_live = None;
                }
                app.session = None;
                app.notif_rx = None;
                app.pending_question = None;
                app.pending_approval = None;
                // Refresh git branch (changes may have been applied).
                app.git_branch = read_git_branch(&app.context.workspace_root.clone());
            }
        }
    }
}

fn snapshot_to_reasoning(s: &ArtifactSnapshot) -> ReasoningSnapshot {
    ReasoningSnapshot {
        goal: s.interpreted_goal.clone(),
        confidence_pct: (s.confidence * 100.0).round() as u8,
        evidence_count: s.evidence_count,
        candidate_paths: s.candidate_paths.clone(),
        open_ambiguity_count: s.open_ambiguity_count,
        open_risks: s.open_risks.clone(),
    }
}

fn apply_notification(app: &mut AgentApp, notif: Notification) {
    // Accumulate streaming tokens into a live buffer.
    match &notif {
        Notification::InferenceToken {
            call_id,
            text,
            is_thinking,
        } => {
            if app.streaming_call_id != Some(*call_id) {
                // New call — reset buffers.
                app.streaming_buf.clear();
                app.thinking_chars = 0;
                app.streaming_call_id = Some(*call_id);
            }
            if *is_thinking {
                app.thinking_chars += text.len();
            } else {
                app.streaming_buf.push_str(text);
            }
            // Don't touch log_scroll_from_bottom — let the user scroll freely.
            return;
        }
        Notification::InferenceCallFinished {
            call_id,
            tool,
            elapsed_ms,
            outcome,
        } => {
            // In agent mode the raw JSON token stream is noise — AgentToolCall/Result
            // provide the pretty-printed equivalent. Suppress JSON blobs.
            let is_json_blob = app.streaming_buf.trim_start().starts_with('{') && app.active_run;
            if !is_json_blob && !app.streaming_buf.is_empty() {
                app.push(SessionEntryKind::Progress, app.streaming_buf.clone());
            }
            app.streaming_buf.clear();
            app.thinking_chars = 0;
            app.streaming_call_id = None;
            // Show raw timing line only outside agent mode.
            if !app.active_run {
                app.push(
                    SessionEntryKind::Progress,
                    format!(
                        "← {} #{} {} {:.1}s",
                        tool,
                        call_id,
                        outcome,
                        *elapsed_ms as f64 / 1000.0
                    ),
                );
            }
            return;
        }
        _ => {}
    }

    // Update the reasoning pane whenever a snapshot is available.
    match &notif {
        Notification::ModelCallFinished {
            snapshot: Some(s), ..
        }
        | Notification::LocalizeFinished { snapshot: s, .. }
        | Notification::ApprovalNeeded { snapshot: s, .. } => {
            app.reasoning = Some(snapshot_to_reasoning(s));
        }
        _ => {}
    }

    match &notif {
        Notification::ClarificationNeeded {
            ambiguity_id,
            question,
            options,
            confidence,
        } => {
            // Only set if not already showing this ambiguity.
            let already_set = app
                .pending_question
                .as_ref()
                .map(|q| q.ambiguity_id == *ambiguity_id)
                .unwrap_or(false);
            if !already_set {
                let conf_str = if *confidence > 0.0 {
                    format!(" [conf {}%]", (*confidence * 100.0).round() as u8)
                } else {
                    String::new()
                };
                let display = if options.is_empty() {
                    format!("{question}{conf_str}")
                } else {
                    format!("{question}{conf_str}\nOptions: {}", options.join(" / "))
                };
                app.pending_question = Some(PendingQuestion {
                    ambiguity_id: ambiguity_id.clone(),
                });
                app.push(SessionEntryKind::Question, display);
                app.status = "waiting for answer".into();
            }
            return;
        }
        Notification::ApprovalNeeded {
            candidate_count,
            snapshot,
        } => {
            let count = *candidate_count;
            app.pending_approval = Some(PendingApproval {
                kind: PendingApprovalKind::Plan,
            });
            let goal_line = if snapshot.interpreted_goal.is_empty() {
                String::new()
            } else {
                format!("Goal: {}\n", snapshot.interpreted_goal)
            };
            let files_line = if snapshot.candidate_paths.is_empty() {
                String::new()
            } else {
                format!("Files: {}\n", snapshot.candidate_paths.join(", "))
            };
            let msg = format!(
                "{goal_line}{files_line}Plan ready: {count} candidate file(s) found [conf {}%].\nType 'y' to approve and execute, 'n' to reject.",
                (snapshot.confidence * 100.0).round() as u8,
            );
            app.push(SessionEntryKind::Question, msg);
            app.status = "waiting for approval".into();
            return;
        }
        // Show proposed changes as a summary header + per-file diff block.
        Notification::ChangeProposed {
            description,
            ops,
            commands,
            diffs,
        } => {
            use shunt_core::machine::DiffLine;
            app.last_task_description = Some(description.clone());
            let cmds_str = if commands.is_empty() {
                String::new()
            } else {
                format!(" + run: {}", commands.join(", "))
            };
            let ops_str = ops.join(", ");
            app.push(
                SessionEntryKind::Progress,
                format!("Proposing: {description} [{ops_str}]{cmds_str}"),
            );
            for diff in diffs {
                if diff.lines.is_empty() {
                    continue;
                }
                app.push(SessionEntryKind::Progress, format!("── {} ──", diff.path));
                let mut body = String::new();
                for dl in &diff.lines {
                    match dl {
                        DiffLine::Added(l) => {
                            body.push('+');
                            body.push_str(l);
                            body.push('\n');
                        }
                        DiffLine::Removed(l) => {
                            body.push('-');
                            body.push_str(l);
                            body.push('\n');
                        }
                        DiffLine::Context(l) => {
                            body.push(' ');
                            body.push_str(l);
                            body.push('\n');
                        }
                    }
                }
                app.push(SessionEntryKind::Diff, body.trim_end().to_string());
            }
            return;
        }
        // PatchApprovalNeeded removed in M5.3 — patch gate is gone.
        Notification::DangerousCommandsProposed { commands, reason } => {
            let display = commands
                .iter()
                .map(|c| c.display())
                .collect::<Vec<_>>()
                .join("\n  ");
            app.pending_approval = Some(PendingApproval {
                kind: PendingApprovalKind::Commands,
            });
            app.push(
                SessionEntryKind::Question,
                format!(
                    "⚠ Dangerous commands ({reason}):\n  {display}\nType 'y' to run, 'n' to skip."
                ),
            );
            app.status = String::from("waiting for danger approval");
            return;
        }
        Notification::AgentToolCall {
            turn,
            max_turns,
            tool,
            summary,
        } => {
            let prev = app.agent_live.as_ref();
            let mut files_read = prev.map(|a| a.files_read.clone()).unwrap_or_default();
            let files_written = prev.map(|a| a.files_written.clone()).unwrap_or_default();
            if tool == "read_file" && !files_read.contains(summary) {
                files_read.push(summary.clone());
            }
            let pending_write = if tool == "write_file" || tool == "str_replace" {
                Some(summary.clone())
            } else {
                None
            };
            app.agent_live = Some(AgentLiveState {
                turn: *turn,
                max_turns: *max_turns,
                last_tool: tool.clone(),
                last_summary: summary.clone(),
                files_read,
                files_written,
                pending_write,
                last_ok: true,
                last_detail: String::new(),
            });
            app.push(SessionEntryKind::ToolCall, format!("{tool}  {summary}"));
            return;
        }
        Notification::AgentToolResult {
            turn: _,
            ok,
            detail,
        } => {
            if let Some(live) = app.agent_live.as_mut() {
                live.last_ok = *ok;
                live.last_detail = detail[..detail.len().min(160)].to_string();
                // Confirm a pending write succeeded.
                if *ok {
                    if let Some(path) = live.pending_write.take()
                        && !live.files_written.contains(&path)
                    {
                        live.files_written.push(path);
                    }
                } else {
                    live.pending_write = None;
                }
            }
            if !detail.is_empty() {
                let kind = if *ok {
                    SessionEntryKind::ToolOk
                } else {
                    SessionEntryKind::ToolErr
                };
                app.push(kind, detail[..detail.len().min(160)].to_string());
            }
            return;
        }
        Notification::UndoAvailable => {
            app.can_undo = true;
            app.can_commit = true;
            return;
        }
        Notification::SetupCommandStarted { display } => {
            app.push(SessionEntryKind::SetupCmd, display.clone());
            return;
        }
        Notification::ModelCallStarted { .. } => {
            if let Some(text) = format_notification(&notif) {
                app.push(SessionEntryKind::Frame, text);
            }
            return;
        }
        _ => {}
    }

    if let Some(text) = format_notification(&notif) {
        app.push(SessionEntryKind::Progress, text);
    }
}

fn format_notification(notif: &Notification) -> Option<String> {
    match notif {
        Notification::TaskStarted => Some("Task started".into()),
        Notification::ObserveStarted => Some("Observing workspace...".into()),
        Notification::ObserveFinished { summary } => Some(format!("Observed: {summary}")),
        Notification::PhaseEntered { summary, .. } => Some(summary.clone()),
        Notification::ModelCallStarted { phase } => {
            use shunt_core::TaskPhase;
            let label = match phase {
                TaskPhase::Clarify => "Clarifying goal with model…",
                TaskPhase::Understand => "Understanding task scope…",
                TaskPhase::Execute => "Planning changes…",
                _ => "Model thinking…",
            };
            Some(label.into())
        }
        Notification::InferenceCallStarted {
            call_id,
            tool,
            model,
            mode,
        } => Some(format!("→ {tool} #{call_id} model={model} mode={mode}")),
        // InferenceCallFinished and InferenceToken are handled directly in
        // apply_notification to flush the streaming buffer first.
        Notification::InferenceCallFinished { .. } | Notification::InferenceToken { .. } => None,
        Notification::ModelCallFinished {
            phase,
            summary,
            snapshot,
        } => {
            if let Some(s) = snapshot {
                let conf = (s.confidence * 100.0).round() as u8;
                Some(format!(
                    "[{phase:?}] {summary} — {} [conf {conf}%]",
                    s.interpreted_goal
                ))
            } else {
                Some(format!("[{phase:?}] {summary}"))
            }
        }
        Notification::LocalizeStarted => Some("Searching for relevant files...".into()),
        Notification::LocalizeFinished { summary, snapshot } => {
            if snapshot.candidate_paths.is_empty() {
                Some(format!("Located: {summary}"))
            } else {
                Some(format!(
                    "Located: {} — {}",
                    summary,
                    snapshot.candidate_paths.join(", ")
                ))
            }
        }
        Notification::UserInputNeeded { summary } => Some(format!("Input needed: {summary}")),
        Notification::StageChanged { stage, status } => {
            use shunt_core::{ExecutionStageKind, StageStatus};
            let stage_label = match stage {
                ExecutionStageKind::Inspect => "Inspect",
                ExecutionStageKind::Propose => "Plan changes",
                ExecutionStageKind::Apply => "Apply changes",
                ExecutionStageKind::Verify => "Verify",
                ExecutionStageKind::Validate => "Validate",
                ExecutionStageKind::Setup => "Setup",
            };
            let status_label = match status {
                StageStatus::Pending => "pending",
                StageStatus::Running => "starting…",
                StageStatus::Passed => "done",
                StageStatus::Failed => "failed",
                StageStatus::Blocked => "blocked",
            };
            Some(format!("{stage_label}: {status_label}"))
        }
        Notification::FrontierRaised { reason, summary } => {
            // summary now contains failed-verifier detail (e.g. "npm_test: npm test: err…")
            Some(format!("⚠ Blocked [{reason:?}] — {summary}"))
        }
        Notification::Note { text } => Some(text.clone()),
        Notification::RunCompleted { summary } => Some(format!("Done: {summary}")),
        Notification::RunFailed { summary } => Some(format!("Failed: {summary}")),
        Notification::UndoAvailable => None, // handled in apply_notification
        Notification::SetupStarted { count } => {
            Some(format!("Running {count} setup command(s)..."))
        }
        // Handled directly in apply_notification as SetupCmd entry kind.
        Notification::SetupCommandStarted { .. } => None,
        Notification::SetupFinished { summary } => Some(format!("Setup: {summary}")),
        // ChangeProposed renders a summary header + diff block — handled in apply_notification.
        Notification::ChangeProposed { .. } => None,
        // Handled by apply_notification directly.
        Notification::ClarificationNeeded { .. }
        | Notification::ApprovalNeeded { .. }
        | Notification::DangerousCommandsProposed { .. }
        | Notification::AgentToolCall { .. }
        | Notification::AgentToolResult { .. } => None,
    }
}

fn build_reasoning_pane(r: &ReasoningSnapshot) -> Paragraph<'_> {
    let mut lines: Vec<Line> = Vec::new();

    // Confidence bar: filled blocks up to 10 chars
    let filled = (r.confidence_pct as usize * 10 / 100).min(10);
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(10 - filled));
    let conf_color = if r.confidence_pct >= 80 {
        Color::Green
    } else if r.confidence_pct >= 60 {
        Color::Yellow
    } else {
        Color::Red
    };
    lines.push(Line::from(vec![
        Span::styled(format!("{bar} "), Style::default().fg(conf_color)),
        Span::styled(
            format!("{}%", r.confidence_pct),
            Style::default().fg(conf_color),
        ),
    ]));
    lines.push(Line::raw(""));

    // Goal (wrapped manually at pane width ~34 chars)
    lines.push(Line::styled("Goal:", Style::default().fg(Color::Cyan)));
    for chunk in r.goal.chars().collect::<Vec<_>>().chunks(32) {
        lines.push(Line::raw(format!("  {}", chunk.iter().collect::<String>())));
    }
    lines.push(Line::raw(""));

    // Evidence
    lines.push(Line::from(vec![
        Span::styled("Evidence: ", Style::default().fg(Color::DarkGray)),
        Span::raw(format!("{} refs", r.evidence_count)),
    ]));

    // Candidates
    if !r.candidate_paths.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::styled("Files:", Style::default().fg(Color::Cyan)));
        for path in &r.candidate_paths {
            let short = if path.len() > 30 {
                format!("…{}", &path[path.len() - 29..])
            } else {
                path.clone()
            };
            lines.push(Line::raw(format!("  {short}")));
        }
    }

    // Open risks
    if !r.open_risks.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::styled("Risks:", Style::default().fg(Color::Yellow)));
        for risk in &r.open_risks {
            let short = if risk.len() > 30 {
                format!("{}…", &risk[..29])
            } else {
                risk.clone()
            };
            lines.push(Line::raw(format!("  {short}")));
        }
    }

    // Open ambiguities
    if r.open_ambiguity_count > 0 {
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::styled("Ambiguities: ", Style::default().fg(Color::Yellow)),
            Span::raw(format!("{} open", r.open_ambiguity_count)),
        ]));
    }

    Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" reasoning ")
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false })
}

fn build_agent_live_pane(live: &AgentLiveState, elapsed: Option<Duration>) -> Paragraph<'_> {
    let mut lines: Vec<Line> = Vec::new();

    // Turn progress bar + elapsed time
    let filled = ((live.turn + 1) * 10 / live.max_turns.max(1)).min(10);
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(10 - filled));
    let elapsed_str = elapsed
        .map(|d| format!(" {:.0}s", d.as_secs_f64()))
        .unwrap_or_default();
    lines.push(Line::from(vec![
        Span::styled(format!("{bar} "), Style::default().fg(Color::Cyan)),
        Span::styled(
            format!("{}/{}", live.turn + 1, live.max_turns),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(elapsed_str, Style::default().fg(Color::DarkGray)),
    ]));
    lines.push(Line::raw(""));

    // Last tool call
    let tool_color = if live.last_ok {
        Color::Green
    } else {
        Color::Red
    };
    lines.push(Line::from(vec![
        Span::styled("→ ", Style::default().fg(Color::Blue)),
        Span::styled(live.last_tool.to_string(), Style::default().fg(tool_color)),
    ]));
    // Summary: wrap at pane width (~34 chars)
    for chunk in live.last_summary.chars().collect::<Vec<_>>().chunks(32) {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                chunk.iter().collect::<String>(),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    // Last result detail
    if !live.last_detail.is_empty() {
        let result_color = if live.last_ok {
            Color::DarkGray
        } else {
            Color::Red
        };
        let sym = if live.last_ok { "✓" } else { "✗" };
        lines.push(Line::raw(""));
        let mut first = true;
        for chunk in live.last_detail.chars().collect::<Vec<_>>().chunks(32) {
            let chunk_str = chunk.iter().collect::<String>();
            if first {
                lines.push(Line::from(vec![
                    Span::styled(format!("{sym} "), Style::default().fg(tool_color)),
                    Span::styled(chunk_str, Style::default().fg(result_color)),
                ]));
                first = false;
            } else {
                lines.push(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(chunk_str, Style::default().fg(result_color)),
                ]));
            }
        }
    }

    // Files read
    if !live.files_read.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::styled("read:", Style::default().fg(Color::DarkGray)));
        for path in &live.files_read {
            let short = if path.len() > 30 {
                format!("…{}", &path[path.len() - 29..])
            } else {
                path.clone()
            };
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(short, Style::default().fg(Color::Cyan)),
            ]));
        }
    }

    // Files written / modified
    if !live.files_written.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::styled("wrote:", Style::default().fg(Color::DarkGray)));
        for path in &live.files_written {
            let short = if path.len() > 30 {
                format!("…{}", &path[path.len() - 29..])
            } else {
                path.clone()
            };
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(short, Style::default().fg(Color::Green)),
            ]));
        }
    }

    Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" agent ")
                .border_style(Style::default().fg(Color::Cyan)),
        )
        .wrap(Wrap { trim: false })
}

fn build_idle_pane(
    model: &str,
    workspace: &str,
    git_branch: Option<&str>,
    last_files: &[String],
    can_undo: bool,
    can_commit: bool,
) -> Paragraph<'static> {
    let short_ws = if workspace.len() > 32 {
        format!("…{}", &workspace[workspace.len() - 31..])
    } else {
        workspace.to_string()
    };
    let mut lines: Vec<Line<'static>> = vec![
        Line::styled("model", Style::default().fg(Color::DarkGray)),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(model.to_string(), Style::default().fg(Color::Yellow)),
        ]),
        Line::raw(""),
        Line::styled("workspace", Style::default().fg(Color::DarkGray)),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(short_ws, Style::default().fg(Color::DarkGray)),
        ]),
    ];
    if let Some(branch) = git_branch {
        let branch_color = if branch.ends_with('*') {
            Color::Yellow
        } else {
            Color::DarkGray
        };
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("git:{branch}"), Style::default().fg(branch_color)),
        ]));
    }

    // Last session files
    if !last_files.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::styled(
            "last written",
            Style::default().fg(Color::DarkGray),
        ));
        for path in last_files.iter().take(6) {
            let short = if path.len() > 30 {
                format!("…{}", &path[path.len() - 29..])
            } else {
                path.clone()
            };
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(short, Style::default().fg(Color::Green)),
            ]));
        }
    }

    // Available actions
    lines.push(Line::raw(""));
    lines.push(Line::styled("keys", Style::default().fg(Color::DarkGray)));
    lines.push(Line::raw("  Enter    submit"));
    lines.push(Line::raw("  ←/→      cursor"));
    lines.push(Line::raw("  Home/End  line bounds"));
    lines.push(Line::raw("  PgUp/Dn  scroll"));
    if can_undo {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("u         undo", Style::default().fg(Color::Yellow)),
        ]));
    }
    if can_commit {
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled("g         commit", Style::default().fg(Color::Yellow)),
        ]));
    }
    lines.push(Line::raw("  /help    commands"));
    lines.push(Line::raw("  q        quit"));

    Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" shunt ")
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false })
}

fn draw_agent(terminal: &mut AppTerminal, app: &AgentApp) -> Result<(), CliError> {
    terminal.draw(|frame| {
        let area = frame.area();
        // Input box grows with content (min 3 = 1 border + 1 line + 1 border, max 8).
        let input_lines = app.input.lines().count().max(1);
        let input_height = (input_lines as u16 + 2).min(8);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),            // header bar
                Constraint::Min(5),               // main body (log + optional reasoning pane)
                Constraint::Length(input_height), // input
            ])
            .split(area);

        // ── Header bar ─────────────────────────────────────────────────────
        let state_label = app
            .machine_state
            .as_ref()
            .map(|s| s.label())
            .unwrap_or("idle");
        let status_label = if let Some(live) = &app.agent_live {
            if app.thinking_chars > 0 {
                format!(
                    "[{state_label} · turn {}/{} · ⋯ {:.0}k]",
                    live.turn + 1,
                    live.max_turns,
                    app.thinking_chars as f64 / 1000.0
                )
            } else {
                format!(
                    "[{state_label} · turn {}/{}]",
                    live.turn + 1,
                    live.max_turns
                )
            }
        } else if app.thinking_chars > 0 {
            format!(
                "[{state_label} · ⋯ {:.0}k]",
                app.thinking_chars as f64 / 1000.0
            )
        } else {
            format!("[{state_label}]")
        };
        let header = Paragraph::new(Line::from(
            vec![
                Span::styled(" shunt ", Style::default().fg(Color::Cyan)),
                Span::raw("  "),
                Span::styled(
                    app.context.config.model.as_deref().unwrap_or("<unset>"),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw("  "),
                Span::styled(
                    status_label,
                    if app.active_run {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ),
                Span::raw("  "),
                Span::styled(
                    app.context.workspace_root.display().to_string(),
                    Style::default().fg(Color::DarkGray),
                ),
            ]
            .into_iter()
            .chain(
                app.git_branch
                    .as_deref()
                    .map(|b| {
                        vec![
                            Span::raw("  "),
                            Span::styled(
                                format!("git:{b}"),
                                Style::default().fg(if b.ends_with('*') {
                                    Color::Yellow
                                } else {
                                    Color::DarkGray
                                }),
                            ),
                        ]
                    })
                    .unwrap_or_default(),
            )
            .collect::<Vec<_>>(),
        ));
        frame.render_widget(header, chunks[0]);

        // ── Body: log + right pane (always visible when screen is wide enough) ──
        let body_area = chunks[1];
        let show_right = body_area.width > 80;
        let (log_area, right_area) = if show_right {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(40), Constraint::Length(38)])
                .split(body_area);
            (cols[0], Some(cols[1]))
        } else {
            (body_area, None)
        };

        // ── Conversation log ────────────────────────────────────────────────
        let log_lines = build_log_lines(app);
        let scroll = log_scroll(&log_lines, log_area.height, app.log_scroll_from_bottom);
        let log = Paragraph::new(Text::from(log_lines))
            .block(Block::default().borders(Borders::LEFT))
            .scroll((scroll, 0))
            .wrap(Wrap { trim: false });
        frame.render_widget(log, log_area);

        // ── Right pane — always shows something useful ─────────────────────
        if let Some(area) = right_area {
            let elapsed = app.run_started_at.map(|t| t.elapsed());
            if let Some(live) = &app.agent_live {
                frame.render_widget(build_agent_live_pane(live, elapsed), area);
            } else if let Some(r) = &app.reasoning {
                frame.render_widget(build_reasoning_pane(r), area);
            } else {
                let model = app.context.config.model.as_deref().unwrap_or("<unset>");
                let workspace = app.context.workspace_root.display().to_string();
                frame.render_widget(
                    build_idle_pane(
                        model,
                        &workspace,
                        app.git_branch.as_deref(),
                        &app.last_session_files,
                        app.can_undo,
                        app.can_commit,
                    ),
                    area,
                );
            }
        }

        // ── Input ───────────────────────────────────────────────────────────
        let input_block = Block::default()
            .borders(Borders::ALL)
            .title(input_title(app));
        let input = Paragraph::new(app.input.as_str()).block(input_block);
        frame.render_widget(input, chunks[2]);
        // Cursor at actual cursor_pos, not always end of line.
        let (cursor_col, cursor_row) = cursor_visual_pos(&app.input, app.cursor_pos);
        let cursor_row = cursor_row.min(input_height.saturating_sub(3));
        frame.set_cursor_position((chunks[2].x + 1 + cursor_col, chunks[2].y + 1 + cursor_row));
    })?;
    Ok(())
}

fn build_log_lines(app: &AgentApp) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry in &app.entries {
        match entry.kind {
            SessionEntryKind::Diff => {
                // git-diff coloring: +green, -red, context dimmed.
                for line in entry.body.lines() {
                    if let Some(rest) = line.strip_prefix('+') {
                        lines.push(Line::from(vec![
                            Span::styled("  + ", Style::default().fg(Color::Green)),
                            Span::styled(rest.to_string(), Style::default().fg(Color::Green)),
                        ]));
                    } else if let Some(rest) = line.strip_prefix('-') {
                        lines.push(Line::from(vec![
                            Span::styled("  - ", Style::default().fg(Color::Red)),
                            Span::styled(rest.to_string(), Style::default().fg(Color::Red)),
                        ]));
                    } else {
                        let rest = line.strip_prefix(' ').unwrap_or(line);
                        lines.push(Line::from(vec![
                            Span::raw("    "),
                            Span::styled(rest.to_string(), Style::default().fg(Color::DarkGray)),
                        ]));
                    }
                }
            }
            // Agent tool calls: indented blue "→ tool  summary" line.
            SessionEntryKind::ToolCall => {
                let mut first = true;
                for line in entry.body.lines() {
                    if first {
                        lines.push(Line::from(vec![
                            Span::styled("  → ", Style::default().fg(Color::Blue)),
                            Span::styled(line.to_string(), Style::default().fg(Color::Cyan)),
                        ]));
                        first = false;
                    } else {
                        lines.push(Line::from(vec![
                            Span::raw("    "),
                            Span::styled(line.to_string(), Style::default().fg(Color::DarkGray)),
                        ]));
                    }
                }
            }
            // Successful tool result: indented green check.
            SessionEntryKind::ToolOk => {
                let mut first = true;
                for line in entry.body.lines() {
                    if first {
                        lines.push(Line::from(vec![
                            Span::styled("  ✓ ", Style::default().fg(Color::Green)),
                            Span::styled(line.to_string(), Style::default().fg(Color::DarkGray)),
                        ]));
                        first = false;
                    } else {
                        lines.push(Line::from(vec![
                            Span::raw("    "),
                            Span::styled(line.to_string(), Style::default().fg(Color::DarkGray)),
                        ]));
                    }
                }
            }
            // Failed tool result: indented red x.
            SessionEntryKind::ToolErr => {
                let mut first = true;
                for line in entry.body.lines() {
                    if first {
                        lines.push(Line::from(vec![
                            Span::styled("  ✗ ", Style::default().fg(Color::Red)),
                            Span::styled(line.to_string(), Style::default().fg(Color::Red)),
                        ]));
                        first = false;
                    } else {
                        lines.push(Line::from(vec![
                            Span::raw("    "),
                            Span::styled(line.to_string(), Style::default().fg(Color::Red)),
                        ]));
                    }
                }
            }
            // Setup command: indented yellow $.
            SessionEntryKind::SetupCmd => {
                lines.push(Line::from(vec![
                    Span::styled("  $ ", Style::default().fg(Color::Yellow)),
                    Span::styled(entry.body.to_string(), Style::default().fg(Color::Yellow)),
                ]));
            }
            // Task done: full-width bright green.
            SessionEntryKind::Done => {
                lines.push(Line::from(vec![
                    Span::styled(" ✔ ", Style::default().fg(Color::LightGreen)),
                    Span::styled(
                        entry.body.to_string(),
                        Style::default().fg(Color::LightGreen),
                    ),
                ]));
            }
            // Standard entries with (prefix, prefix_color, body_color).
            _ => {
                let (prefix, prefix_color, body_color) = match entry.kind {
                    SessionEntryKind::User => ("›", Color::Green, Color::White),
                    SessionEntryKind::Frame => ("·", Color::Cyan, Color::White),
                    SessionEntryKind::Progress => ("·", Color::DarkGray, Color::Reset),
                    SessionEntryKind::Question => ("?", Color::Magenta, Color::White),
                    SessionEntryKind::Error => ("!", Color::Red, Color::Red),
                    _ => unreachable!(),
                };
                let mut first = true;
                for line in entry.body.lines() {
                    if first {
                        lines.push(Line::from(vec![
                            Span::styled(format!(" {prefix} "), Style::default().fg(prefix_color)),
                            Span::styled(line.to_string(), Style::default().fg(body_color)),
                        ]));
                        first = false;
                    } else {
                        lines.push(Line::from(vec![
                            Span::raw("   "),
                            Span::styled(line.to_string(), Style::default().fg(body_color)),
                        ]));
                    }
                }
            }
        }
    }

    // Live streaming output while a model call is in progress.
    if app.streaming_call_id.is_some() {
        if app.thinking_chars > 0 && app.streaming_buf.is_empty() {
            // Thinking phase: show char count indicator.
            lines.push(Line::from(vec![
                Span::styled("  ⋯ ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("thinking… {} chars", app.thinking_chars),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
        } else if !app.streaming_buf.is_empty() {
            let is_json = app.streaming_buf.trim_start().starts_with('{') && app.active_run;
            if is_json {
                // Agent JSON generation: show compact "generating..." indicator.
                let turn_info = app
                    .agent_live
                    .as_ref()
                    .map(|l| format!(" turn {}/{}", l.turn + 1, l.max_turns))
                    .unwrap_or_default();
                lines.push(Line::from(vec![
                    Span::styled("  ‥ ", Style::default().fg(Color::Blue)),
                    Span::styled(
                        format!("generating{turn_info}…"),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            } else {
                // Non-agent streaming (clarify/understand): show raw content.
                let mut first = true;
                for text_line in app.streaming_buf.lines() {
                    if first {
                        lines.push(Line::from(vec![
                            Span::styled(" ‥ ", Style::default().fg(Color::Blue)),
                            Span::styled(
                                text_line.to_string(),
                                Style::default().fg(Color::DarkGray),
                            ),
                        ]));
                        first = false;
                    } else {
                        lines.push(Line::from(vec![
                            Span::raw("   "),
                            Span::styled(
                                text_line.to_string(),
                                Style::default().fg(Color::DarkGray),
                            ),
                        ]));
                    }
                }
            }
        }
    }

    if lines.is_empty() {
        lines.push(Line::from("  Type a task and press Enter."));
    }
    lines
}

fn log_scroll(lines: &[Line<'static>], area_height: u16, from_bottom: usize) -> u16 {
    let visible = area_height.saturating_sub(2) as usize;
    if visible == 0 || lines.len() <= visible {
        0
    } else {
        let max_scroll = lines.len() - visible;
        max_scroll
            .saturating_sub(from_bottom.min(max_scroll))
            .min(u16::MAX as usize) as u16
    }
}

fn input_title(app: &AgentApp) -> &'static str {
    if app.pending_question.is_some() {
        "Answer  Enter=submit  Esc=clear  Ctrl-C=quit"
    } else if let Some(ref a) = app.pending_approval {
        match a.kind {
            PendingApprovalKind::Commands => "y=run commands  n=skip  Ctrl-C=quit",
            _ => "y=approve  n=reject  Ctrl-C=quit",
        }
    } else if app.active_run {
        "Watching...  PgUp/PgDn=scroll  Ctrl-End=live  Ctrl-C=quit"
    } else if app.can_commit && app.can_undo {
        "Prompt  Enter=submit  g=commit  u=undo  q=quit"
    } else if app.can_commit {
        "Prompt  Enter=submit  g=commit  q=quit"
    } else if app.can_undo {
        "Prompt  Enter=submit  u=undo  q=quit"
    } else {
        "Prompt  Enter=submit  Alt+Enter=newline  PgUp/PgDn=scroll  q=quit"
    }
}

/// Headless ("full agentic") run: drives the same core session machine the TUI
/// uses, but with a fully-autonomous policy + an `AutoResponder` instead of a
/// human. No separate business logic — just a different client of the core.
fn run_handle_request(context: &AppContext, request: String) -> Result<TaskView, CliError> {
    use shunt_runtime::driver::{AutoResponder, drive_session};

    tracing::debug!("handle command (headless agentic)");
    let now = OffsetDateTime::now_utc();
    let task_id = format!("task-{}", now.unix_timestamp_nanos());
    let artifact_id = ArtifactId(format!("artifact-{}", now.unix_timestamp_nanos()));
    let provider = context.provider()?;
    let store_path = context.db_path().display().to_string();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .map_err(|e| CliError::Terminal(format!("tokio runtime: {e}")))?;

    let mut responder = AutoResponder::default();
    let outcome = drive_session(
        task_id.clone(),
        artifact_id,
        context.workspace_root.display().to_string(),
        context.recipe(),
        store_path,
        provider,
        AutonomyPolicy::headless(),
        rt.handle().clone(),
        request,
        &mut responder,
    );
    tracing::debug!(final_state = ?outcome.final_state, "headless run finished");

    // Rebuild the task view from the store the session persisted to.
    let runtime = TaskRuntime::new(SqliteStore::open(context.db_path())?);
    let result = runtime.load_handle_result(&task_id)?;
    Ok(TaskView {
        task: result.task,
        artifact: result.artifact,
        active_recipe_run: result.active_recipe_run,
        frontier_cases: result.frontier_cases,
    })
}

fn print_task_view(view: &TaskView) -> Result<(), CliError> {
    println!("{}", summarize_task_view(view));
    Ok(())
}

fn summarize_task_view(view: &TaskView) -> String {
    let mut lines = vec![format!(
        "task: {}  phase: {:?}",
        view.task.id.0, view.task.phase
    )];

    lines.push(format!("goal: {}", view.artifact.interpreted_goal));

    if !view.artifact.target_scope.is_empty() {
        lines.push("target scope:".into());
        for scope in view.artifact.target_scope.iter().take(5) {
            lines.push(format!("  - {scope}"));
        }
    }

    if !view.artifact.candidate_files.is_empty() {
        lines.push("candidate files:".into());
        for candidate in view.artifact.candidate_files.iter().take(5) {
            lines.push(format!("  - {}", candidate.path));
        }
    }

    if !view.artifact.package_facts.is_empty() {
        lines.push("package facts:".into());
        for fact in view.artifact.package_facts.iter().take(5) {
            lines.push(format!(
                "  - {}:{} @ {}",
                fact.ecosystem,
                fact.name,
                fact.version.as_deref().unwrap_or("unknown")
            ));
        }
    }

    if !view.artifact.manual_evidence.is_empty() {
        lines.push("manual evidence:".into());
        for manual in view.artifact.manual_evidence.iter().take(3) {
            lines.push(format!(
                "  - {} {} {:?} {}",
                manual.package,
                manual.version.as_deref().unwrap_or("<unversioned>"),
                manual.version_status,
                manual.locator
            ));
        }
    }

    if !view.artifact.ambiguities.is_empty() {
        lines.push("ambiguities:".into());
        for ambiguity in &view.artifact.ambiguities {
            lines.push(format!("  - {}", ambiguity.question));
        }
    }

    if let Some(recipe_run) = &view.active_recipe_run {
        lines.push(format!(
            "recipe: {}@{}  stage: {:?}",
            recipe_run.recipe.id, recipe_run.recipe.version, recipe_run.current_stage
        ));
        if !recipe_run.proposed_changes.is_empty() {
            lines.push("proposed changes:".into());
            for change in recipe_run.proposed_changes.iter().take(5) {
                lines.push(format!("  - {} ({})", change.path, change.description));
            }
        }
    }

    if !view.frontier_cases.is_empty() {
        lines.push(format!("frontier cases: {}", view.frontier_cases.len()));
    }

    lines.join("\n")
}

fn handle_config(command: ConfigCommand, context: &AppContext) -> Result<(), CliError> {
    match command {
        ConfigCommand::Show => print_json(&context.config)?,
        ConfigCommand::Init { force } => {
            context.init_config(force)?;
            println!("wrote {}", context.config_path.display());
        }
    }
    Ok(())
}

fn handle_task(command: TaskCommand, context: &AppContext) -> Result<(), CliError> {
    let db = context.db_path();
    match command {
        TaskCommand::Start {
            request,
            task_id,
            artifact_id,
        } => {
            tracing::debug!("task start");
            let now = OffsetDateTime::now_utc();
            let task_id = task_id.unwrap_or_else(|| format!("task-{}", now.unix_timestamp_nanos()));
            let artifact_id =
                artifact_id.unwrap_or_else(|| format!("artifact-{}", now.unix_timestamp_nanos()));

            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let (task, artifact) = runtime.start_task(
                now,
                task_id,
                artifact_id,
                context.workspace_root.display().to_string(),
                request,
            )?;
            print_json(&TaskView {
                task,
                artifact,
                active_recipe_run: None,
                frontier_cases: vec![],
            })?;
        }
        TaskCommand::Show { task_id } => {
            tracing::debug!("task show task={task_id}");
            let store = SqliteStore::open(db)?;
            let task = store
                .get_task_run(&task_id)?
                .ok_or_else(|| CliError::TaskNotFound(task_id.clone()))?;
            let artifact = store
                .get_understanding_artifact(&task.current_artifact.0)?
                .ok_or_else(|| CliError::ArtifactNotFound(task.current_artifact.0.clone()))?;
            let frontier_cases = store.list_frontier_cases_for_task(&task_id)?;
            let active_recipe_run = match &task.active_recipe_run {
                Some(recipe_run_id) => store.get_recipe_run(&recipe_run_id.0)?,
                None => None,
            };
            print_json(&TaskView {
                task,
                artifact,
                active_recipe_run,
                frontier_cases,
            })?;
        }
        TaskCommand::Clarify { artifact_id } => {
            tracing::debug!("task clarify artifact={artifact_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let provider = context.provider()?;
            let artifact = runtime
                .clarify_task(&artifact_id, OffsetDateTime::now_utc(), &provider)?
                .ok_or(CliError::ArtifactNotFound(artifact_id))?;
            print_json(&artifact)?;
        }
        TaskCommand::Understand {
            artifact_id,
            heuristic_only,
        } => {
            tracing::debug!("task understand artifact={artifact_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let artifact = if heuristic_only {
                runtime.understand_task(&artifact_id, OffsetDateTime::now_utc())?
            } else {
                let provider = context.provider()?;
                runtime.understand_task_with_provider(
                    &artifact_id,
                    OffsetDateTime::now_utc(),
                    &provider,
                )?
            }
            .ok_or(CliError::ArtifactNotFound(artifact_id))?;
            print_json(&artifact)?;
        }
        TaskCommand::Localize { artifact_id } => {
            tracing::debug!("task localize artifact={artifact_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let artifact = runtime
                .localize_task(&artifact_id, OffsetDateTime::now_utc())?
                .ok_or(CliError::ArtifactNotFound(artifact_id))?;
            print_json(&artifact)?;
        }
        TaskCommand::ExecuteStart { task_id } => {
            tracing::debug!("task execute-start task={task_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let recipe_run =
                runtime.start_execution(&task_id, OffsetDateTime::now_utc(), context.recipe())?;
            print_json(&recipe_run)?;
        }
        TaskCommand::ExecuteInspect { task_id } => {
            tracing::debug!("task execute-inspect task={task_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            print_json(&runtime.execute_inspect(&task_id, OffsetDateTime::now_utc())?)?;
        }
        TaskCommand::ExecutePropose { task_id } => {
            tracing::debug!("task execute-propose task={task_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            print_json(&runtime.execute_propose(&task_id, OffsetDateTime::now_utc())?)?;
        }
        TaskCommand::ExecuteVerify { task_id } => {
            tracing::debug!("task execute-verify task={task_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            print_json(&runtime.execute_verify(&task_id, OffsetDateTime::now_utc())?)?;
        }
        TaskCommand::ExecuteGenerateChange { task_id } => {
            tracing::debug!("task execute-generate-change task={task_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let provider = context.provider()?;
            print_json(&runtime.generate_proposed_change(
                &task_id,
                OffsetDateTime::now_utc(),
                &provider,
                &[],
                &context.config.ignore_patterns,
            )?)?;
        }
        TaskCommand::ExecuteAddChange {
            task_id,
            path,
            description,
            search,
            replacement,
            content,
            content_file,
        } => {
            tracing::debug!("task execute-add-change task={task_id} path={path}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let change = build_manual_change(
                &context.workspace_root,
                path,
                description,
                search,
                replacement,
                content,
                content_file,
            )?;
            print_json(&runtime.set_change_set(&task_id, OffsetDateTime::now_utc(), change)?)?;
        }
        TaskCommand::ExecuteApply { task_id } => {
            tracing::debug!("task execute-apply task={task_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            print_json(&runtime.execute_apply(&task_id, OffsetDateTime::now_utc())?)?;
        }
        TaskCommand::ExecuteValidate { task_id } => {
            tracing::debug!("task execute-validate task={task_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            print_json(&runtime.execute_validate(&task_id, OffsetDateTime::now_utc())?)?;
        }
    }

    Ok(())
}

fn handle_artifact(command: ArtifactCommand, context: &AppContext) -> Result<(), CliError> {
    let db = context.db_path();
    match command {
        ArtifactCommand::Revise {
            artifact_id,
            goal,
            success_criteria,
            constraints,
            target_scope,
            confidence,
        } => {
            tracing::debug!("artifact revise artifact={artifact_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let artifact = runtime
                .revise_artifact(
                    &artifact_id,
                    OffsetDateTime::now_utc(),
                    ArtifactUpdate {
                        interpreted_goal: goal,
                        success_criteria: some_if_any(success_criteria),
                        constraints: some_if_any(constraints),
                        target_scope: some_if_any(target_scope),
                        confidence,
                        ..Default::default()
                    },
                )?
                .ok_or(CliError::ArtifactNotFound(artifact_id))?;
            print_json(&artifact)?;
        }
        ArtifactCommand::Resolve {
            artifact_id,
            ambiguity_id,
            resolution,
        } => {
            tracing::debug!("artifact resolve artifact={artifact_id} ambiguity={ambiguity_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let artifact = runtime
                .resolve_ambiguity(
                    &artifact_id,
                    &ambiguity_id,
                    resolution,
                    OffsetDateTime::now_utc(),
                )?
                .ok_or(CliError::ArtifactNotFound(artifact_id))?;
            print_json(&artifact)?;
        }
        ArtifactCommand::Approve { artifact_id, note } => {
            tracing::debug!("artifact approve artifact={artifact_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let artifact = runtime
                .approve_artifact(
                    &artifact_id,
                    context.config.decided_by.clone(),
                    note,
                    OffsetDateTime::now_utc(),
                )?
                .ok_or(CliError::ArtifactNotFound(artifact_id))?;
            print_json(&artifact)?;
        }
    }

    Ok(())
}

fn handle_frontier(command: FrontierCommand, context: &AppContext) -> Result<(), CliError> {
    let db = context.db_path();
    match command {
        FrontierCommand::Add {
            task_id,
            artifact_id,
            summary,
            reason,
            event_kind,
            event_summary,
            event_confidence,
            frontier_case_id,
        } => {
            tracing::debug!("frontier add task={task_id} artifact={artifact_id}");
            let now = OffsetDateTime::now_utc();
            let store = SqliteStore::open(db)?;
            let task = store
                .get_task_run(&task_id)?
                .ok_or_else(|| CliError::TaskNotFound(task_id.clone()))?;
            let artifact = store
                .get_understanding_artifact(&artifact_id)?
                .ok_or_else(|| CliError::ArtifactNotFound(artifact_id.clone()))?;
            let runtime = TaskRuntime::new(store);
            let frontier_case = runtime.record_frontier_case(
                now,
                frontier_case_id
                    .unwrap_or_else(|| format!("frontier-{}", now.unix_timestamp_nanos())),
                &task,
                &artifact,
                reason.into(),
                summary,
                vec![UncertaintyEvent {
                    task_id: task.id.clone(),
                    stage: None,
                    kind: event_kind.into(),
                    summary: event_summary,
                    confidence: event_confidence,
                    created_at: now,
                }],
            )?;
            print_json(&frontier_case)?;
        }
    }

    Ok(())
}

fn handle_correction(command: CorrectionCommand, context: &AppContext) -> Result<(), CliError> {
    let db = context.db_path();
    match command {
        CorrectionCommand::Patch {
            frontier_case_id,
            summary,
            path,
            description,
            search,
            replacement,
            content,
            content_file,
            correction_id,
        } => {
            tracing::debug!("correction patch frontier_case={frontier_case_id}");
            let now = OffsetDateTime::now_utc();
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let change = build_manual_change(
                &context.workspace_root,
                path,
                description,
                search,
                replacement,
                content,
                content_file,
            )?;
            let (correction, frontier_case, recipe_run) = runtime.create_patch_correction(
                &frontier_case_id,
                correction_id
                    .unwrap_or_else(|| format!("correction-{}", now.unix_timestamp_nanos())),
                summary,
                change,
                now,
            )?;
            print_json(&CorrectionView {
                correction,
                frontier_case,
                recipe_run,
            })?;
        }
        CorrectionCommand::Replay { correction_id } => {
            tracing::debug!("correction replay correction={correction_id}");
            let runtime = TaskRuntime::new(SqliteStore::open(db)?);
            let (correction, frontier_case, recipe_run) =
                runtime.replay_correction(&correction_id, OffsetDateTime::now_utc())?;
            print_json(&CorrectionView {
                correction,
                frontier_case,
                recipe_run,
            })?;
        }
    }

    Ok(())
}

fn ensure_db_parent(path: &Path) -> Result<(), CliError> {
    let Some(parent) = path.parent() else {
        return Err(CliError::MissingDbParent);
    };
    std::fs::create_dir_all(parent)?;
    Ok(())
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn print_json<T>(value: &T) -> Result<(), CliError>
where
    T: Serialize,
{
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn build_manual_change(
    workspace_root: &Path,
    path: String,
    _description: String,
    search: Option<String>,
    replacement: Option<String>,
    content: Option<String>,
    content_file: Option<PathBuf>,
) -> Result<ChangeSet, CliError> {
    let op = match (search, replacement) {
        (Some(search), Some(replacement)) => FileOp::Edit {
            path,
            search,
            replacement,
        },
        (None, None) => FileOp::Create {
            path,
            contents: load_change_contents(workspace_root, content, content_file)?,
        },
        _ => return Err(CliError::IncompleteSearchReplace),
    };
    Ok(ChangeSet {
        ops: vec![op],
        commands: vec![],
    })
}

fn load_change_contents(
    workspace_root: &Path,
    content: Option<String>,
    content_file: Option<PathBuf>,
) -> Result<String, CliError> {
    if let Some(content) = content {
        return Ok(content);
    }
    if let Some(content_file) = content_file {
        return Ok(std::fs::read_to_string(resolve_path(
            workspace_root,
            &content_file,
        ))?);
    }
    Err(CliError::MissingChangeContent)
}

fn canonicalize_or_current(path: Option<PathBuf>) -> Result<PathBuf, CliError> {
    let path = path.unwrap_or(std::env::current_dir()?);
    Ok(path.canonicalize().unwrap_or(path))
}

/// Update the `model = "..."` line in a config file in-place.
/// Preserves all other content including comments.
/// If no model line exists, appends one.
fn patch_model_in_config(path: &Path, model: &str) -> Result<(), CliError> {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let new_line = format!("model = \"{}\"", model);
    let updated = if content.lines().any(|l| l.trim_start().starts_with("model")) {
        content
            .lines()
            .map(|l| {
                if l.trim_start().starts_with("model") {
                    new_line.as_str()
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    } else {
        format!("{content}{new_line}\n")
    };
    std::fs::write(path, updated)?;
    Ok(())
}

fn write_default_config(path: &Path) -> Result<(), CliError> {
    let Some(parent) = path.parent() else {
        return Err(CliError::ConfigMissing(path.display().to_string()));
    };
    std::fs::create_dir_all(parent)?;
    std::fs::write(path, default_config_template())?;
    Ok(())
}

fn some_if_any<T>(values: Vec<T>) -> Option<Vec<T>> {
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

fn default_config_template() -> &'static str {
    r#"db = ".shunt/shunt.db"
endpoint = "http://127.0.0.1:8080"
model = "gemma4-12b"
timeout_secs = 300
recipe_id = "manual.inspect-propose"
recipe_version = "v1"
decided_by = "user"
"#
}

impl From<FrontierReasonArg> for FrontierReason {
    fn from(value: FrontierReasonArg) -> Self {
        match value {
            FrontierReasonArg::LowConfidence => Self::LowConfidence,
            FrontierReasonArg::VerifierFailure => Self::VerifierFailure,
            FrontierReasonArg::RepeatedVerifierFailure => Self::RepeatedVerifierFailure,
            FrontierReasonArg::RecipeInstability => Self::RecipeInstability,
            FrontierReasonArg::ToolChurn => Self::ToolChurn,
            FrontierReasonArg::MaterialUserCorrection => Self::MaterialUserCorrection,
            FrontierReasonArg::RepeatedPatchFailure => Self::RepeatedPatchFailure,
        }
    }
}

impl From<UncertaintyKindArg> for UncertaintyKind {
    fn from(value: UncertaintyKindArg) -> Self {
        match value {
            UncertaintyKindArg::Ambiguity => Self::Ambiguity,
            UncertaintyKindArg::MissingEvidence => Self::MissingEvidence,
            UncertaintyKindArg::LowConfidence => Self::LowConfidence,
            UncertaintyKindArg::VerifierFailure => Self::VerifierFailure,
            UncertaintyKindArg::VerifierDisagreement => Self::VerifierDisagreement,
            UncertaintyKindArg::RetryExhausted => Self::RetryExhausted,
            UncertaintyKindArg::RecipeOscillation => Self::RecipeOscillation,
            UncertaintyKindArg::ToolThrash => Self::ToolThrash,
            UncertaintyKindArg::UserCorrection => Self::UserCorrection,
            UncertaintyKindArg::PatchRejection => Self::PatchRejection,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn temp_workspace(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "shunt-cli-{name}-{}-{}",
            std::process::id(),
            OffsetDateTime::now_utc().unix_timestamp_nanos()
        ))
    }

    #[test]
    fn default_config_uses_local_db() {
        let config = AppConfig::default();
        assert_eq!(config.db, ".shunt/shunt.db");
        assert_eq!(config.endpoint, "http://127.0.0.1:8080");
        assert!(config.model.is_none());
    }

    #[test]
    fn formats_physical_inference_call_notifications() {
        // Started still goes through format_notification.
        let started = format_notification(&Notification::InferenceCallStarted {
            call_id: 7,
            tool: "output".into(),
            model: "local-model".into(),
            mode: "RequiredString".into(),
        })
        .unwrap();
        assert!(started.contains("→ output #7"));

        // Finished and Token are handled directly in apply_notification (to flush
        // the streaming buffer first) so format_notification returns None for them.
        assert!(
            format_notification(&Notification::InferenceCallFinished {
                call_id: 7,
                tool: "output".into(),
                elapsed_ms: 1250,
                outcome: "tool_call".into(),
            })
            .is_none()
        );
        assert!(
            format_notification(&Notification::InferenceToken {
                call_id: 7,
                text: "hello".into(),
                is_thinking: false,
            })
            .is_none()
        );
    }

    #[test]
    fn log_scroll_follows_tail_and_supports_history() {
        let lines = (0..30)
            .map(|i| Line::from(format!("line {i}")))
            .collect::<Vec<_>>();

        assert_eq!(log_scroll(&lines, 12, 0), 20);
        assert_eq!(log_scroll(&lines, 12, 5), 15);
        assert_eq!(log_scroll(&lines, 12, usize::MAX), 0);
    }

    #[test]
    fn resolve_path_keeps_absolute_paths() {
        let base = Path::new("/workspace");
        let path = Path::new("/tmp/shunt.db");
        assert_eq!(resolve_path(base, path), PathBuf::from("/tmp/shunt.db"));
    }

    #[test]
    fn resolve_path_joins_relative_paths_to_workspace() {
        let base = Path::new("/workspace");
        let path = Path::new("nested/shunt.db");
        assert_eq!(
            resolve_path(base, path),
            PathBuf::from("/workspace/nested/shunt.db")
        );
    }

    #[test]
    fn load_context_creates_default_config_in_cwd_workspace() {
        let workspace = temp_workspace("context");
        let _ = fs::remove_dir_all(&workspace);
        fs::create_dir_all(&workspace).unwrap();

        let context = load_context(Some(workspace.clone()), None, true).unwrap();

        assert_eq!(context.workspace_root, workspace);
        assert_eq!(context.config_path, workspace.join(".shunt/config.toml"));
        assert!(context.config_path.is_file());
        assert_eq!(context.db_path(), workspace.join(".shunt/shunt.db"));
    }

    #[test]
    fn load_change_contents_resolves_relative_file_from_workspace_root() {
        let workspace = temp_workspace("content-file");
        let _ = fs::remove_dir_all(&workspace);
        fs::create_dir_all(workspace.join("notes")).unwrap();
        fs::write(workspace.join("notes/change.txt"), "updated contents\n").unwrap();

        let content =
            load_change_contents(&workspace, None, Some(PathBuf::from("notes/change.txt")))
                .unwrap();

        assert_eq!(content, "updated contents\n");
    }

    #[test]
    fn summarize_task_view_lists_candidates() {
        let task = TaskRun {
            id: shunt_core::TaskId("task-1".into()),
            workspace_root: "/workspace".into(),
            phase: shunt_core::TaskPhase::Agree,
            current_artifact: shunt_core::ArtifactId("artifact-1".into()),
            active_recipe_run: None,
            frontier_cases: vec![],
            created_at: OffsetDateTime::UNIX_EPOCH,
            updated_at: OffsetDateTime::UNIX_EPOCH,
        };
        let artifact = shunt_core::UnderstandingArtifact {
            id: shunt_core::ArtifactId("artifact-1".into()),
            task_id: shunt_core::TaskId("task-1".into()),
            original_request: "fix localizer".into(),
            interpreted_goal: "fix localizer".into(),
            success_criteria: vec![],
            constraints: vec![],
            target_scope: vec!["crates/shunt-localize/src/lib.rs".into()],
            evidence: vec![],
            candidate_files: vec![shunt_core::CandidateFile {
                path: "crates/shunt-localize/src/lib.rs".into(),
                summary: "candidate".into(),
            }],
            package_facts: vec![],
            manual_evidence: vec![],
            assumptions: vec![],
            ambiguities: vec![],
            selected_recipe: None,
            risks: vec![],
            confidence: 0.72,
            approval: shunt_core::ApprovalState::draft(),
            revision: 1,
            workspace_profile: shunt_core::WorkspaceProfile::default(),
            created_at: OffsetDateTime::UNIX_EPOCH,
            updated_at: OffsetDateTime::UNIX_EPOCH,
        };
        let summary = summarize_task_view(&TaskView {
            task,
            artifact,
            active_recipe_run: None,
            frontier_cases: vec![],
        });
        assert!(summary.contains("candidate files:"));
        assert!(summary.contains("crates/shunt-localize/src/lib.rs"));
    }
}
