//! Probe-and-compose scope resolution (Layer X, Bet #1).
//!
//! The `ScopeOrchestrator` replaces the monolithic SemanticLocalizer localize step.
//! It runs a small set of narrow probes in sequence, keeps verified answers, and
//! composes them into `OrchestratorResult` — the new scope for the task.
//!
//! Three probes ship with M3:
//!
//! * `ExistingFilesProbe` — wraps SemanticLocalizer; finds files that already exist
//!   (the old localizer demoted to one probe).
//! * `ManifestProbe` — structurally finds workspace manifests without text matching.
//! * `NewFilePathProbe` — asks for a new path when no existing files were found.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::json;
use shunt_core::{
    CandidateFile, EvidenceKind, EvidenceRef, UnderstandingArtifact, VerifierOutcome,
    VerifierStatus,
};
use shunt_infer::{InferError, InferResult, Probe, ProbeCtx, ProbeResult, ToolProvider, ToolSpec};
use shunt_localize::{Localizer, SemanticLocalizer};

// ── Manifest file names recognised by ManifestProbe ──────────────────────────

pub(crate) const MANIFEST_NAMES: &[&str] = &[
    "cargo.toml",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pipfile",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
    "gemfile",
    "mix.exs",
    "deno.json",
    "deno.jsonc",
    "bunfig.toml",
];

pub(crate) fn is_manifest(name: &str) -> bool {
    MANIFEST_NAMES.contains(&name)
}

// ── ExistingFilesProbe ────────────────────────────────────────────────────────

/// Wraps the SemanticLocalizer and returns whatever existing files it finds.
/// This is the old localize step demoted to one probe in the composition.
#[derive(Default)]
pub struct ExistingFilesProbe {
    localizer: SemanticLocalizer,
}

impl Probe for ExistingFilesProbe {
    fn id(&self) -> &str {
        "existing_files"
    }

    fn run(&self, ctx: &ProbeCtx, _provider: &dyn ToolProvider) -> InferResult<ProbeResult> {
        let workspace = ctx.workspace_root.to_string_lossy();
        let packet = self
            .localizer
            .localize(&workspace, &ctx.artifact)
            .map_err(|e| InferError::InvalidOutput {
                retries: 0,
                reason: e.to_string(),
            })?;

        let paths: Vec<String> = packet
            .primary_candidates
            .iter()
            .chain(packet.supporting_candidates.iter())
            .map(|c| c.file.path.clone())
            .collect();

        let evidence: Vec<EvidenceRef> = paths
            .iter()
            .map(|path| EvidenceRef {
                kind: EvidenceKind::File,
                locator: path.clone(),
                summary: format!("found by semantic search: {path}"),
            })
            .collect();

        let confidence = if paths.is_empty() { 0.0 } else { 0.72 };

        Ok(ProbeResult {
            answer: json!({ "paths": paths }),
            evidence,
            confidence,
        })
    }
}

// ── ManifestProbe ─────────────────────────────────────────────────────────────

/// Purely structural: scans the workspace root for manifest files.
/// Returns matching paths — no text matching, no model call.
/// Covers "add a dependency to …" additive tasks where there are no existing
/// text hits but the target file is structurally obvious.
pub struct ManifestProbe;

impl Probe for ManifestProbe {
    fn id(&self) -> &str {
        "manifest"
    }

    fn run(&self, ctx: &ProbeCtx, _provider: &dyn ToolProvider) -> InferResult<ProbeResult> {
        let mut paths = Vec::new();

        if let Ok(entries) = fs::read_dir(&ctx.workspace_root) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                if entry.path().is_file() && is_manifest(&name) {
                    paths.push(entry.file_name().to_string_lossy().into_owned());
                }
            }
        }
        paths.sort();

        let evidence: Vec<EvidenceRef> = paths
            .iter()
            .map(|path| EvidenceRef {
                kind: EvidenceKind::File,
                locator: path.clone(),
                summary: format!("manifest file detected at workspace root: {path}"),
            })
            .collect();

        let confidence = if paths.is_empty() { 0.0 } else { 0.85 };

        Ok(ProbeResult {
            answer: json!({ "paths": paths }),
            evidence,
            confidence,
        })
    }

    fn verify(&self, result: &ProbeResult, workspace: &Path) -> VerifierOutcome {
        let paths = result
            .answer
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();

        let all_exist = paths.iter().all(|p| workspace.join(p).exists());
        if all_exist {
            VerifierOutcome {
                verifier: "manifest".into(),
                status: VerifierStatus::Passed,
                summary: format!("all {} manifest paths exist on disk", paths.len()),
            }
        } else {
            VerifierOutcome {
                verifier: "manifest".into(),
                status: VerifierStatus::Failed,
                summary: "one or more manifest paths not found on disk".into(),
            }
        }
    }
}

// ── NewFilePathProbe ──────────────────────────────────────────────────────────

/// Bounded model call: asks the LLM for the single new file path to create.
/// Only runs when no existing files and no manifests were found — i.e. scaffold
/// or similar additive tasks where the target path must be invented.
pub struct NewFilePathProbe;

impl Probe for NewFilePathProbe {
    fn id(&self) -> &str {
        "new_file_path"
    }

    fn run(&self, ctx: &ProbeCtx, provider: &dyn ToolProvider) -> InferResult<ProbeResult> {
        let tool = ToolSpec {
            name: "new_file_path".into(),
            description: "Return the relative path of the new file that should be created to fulfil this request. Use the workspace layout as context.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path for the new file, e.g. src/auth.rs"
                    }
                },
                "required": ["path"]
            }),
        };

        let system = include_str!("../../../prompts/scope_new_file.system.txt");
        let user = include_str!("../../../prompts/scope_new_file.user.txt")
            .replace("{original_request}", &ctx.artifact.original_request)
            .replace("{interpreted_goal}", &ctx.artifact.interpreted_goal);

        let tc = provider.call_tool(system, &user, &tool)?;
        let path = tc
            .arguments
            .get("path")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| InferError::InvalidOutput {
                retries: 0,
                reason: "new_file_path: missing 'path' field".into(),
            })?;

        if path.trim().is_empty() || path.contains("..") {
            return Err(InferError::InvalidOutput {
                retries: 0,
                reason: format!("new_file_path: invalid path '{path}'"),
            });
        }

        let evidence = vec![EvidenceRef {
            kind: EvidenceKind::File,
            locator: path.clone(),
            summary: format!("new file path proposed by model: {path}"),
        }];

        Ok(ProbeResult {
            answer: json!({ "path": path }),
            evidence,
            confidence: 0.65,
        })
    }

    fn verify(&self, result: &ProbeResult, _workspace: &Path) -> VerifierOutcome {
        let path = result
            .answer
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        // A new path must be relative, non-empty, and not contain path traversal.
        if !path.is_empty() && !path.starts_with('/') && !path.contains("..") {
            VerifierOutcome {
                verifier: "new_file_path".into(),
                status: VerifierStatus::Passed,
                summary: format!("new file path '{path}' is a valid relative path"),
            }
        } else {
            VerifierOutcome {
                verifier: "new_file_path".into(),
                status: VerifierStatus::Failed,
                summary: format!("new file path '{path}' failed validation"),
            }
        }
    }
}

// ── OrchestratorResult ────────────────────────────────────────────────────────

/// Output of the `ScopeOrchestrator`.
#[derive(Debug, Clone)]
pub struct OrchestratorResult {
    /// Resolved scope — union of verified probe answers.
    /// May include paths that don't exist yet (new-file scaffold paths).
    pub target_scope: Vec<String>,
    /// `CandidateFile` entries for scope members that already exist on disk.
    pub candidate_files: Vec<CandidateFile>,
    /// Evidence collected by all probes.
    pub evidence: Vec<EvidenceRef>,
    /// Per-probe log for eval and frontier capture.
    pub probe_log: Vec<ProbeLogEntry>,
}

/// One entry in the probe execution log.
#[derive(Debug, Clone)]
pub struct ProbeLogEntry {
    pub probe_id: String,
    pub answer: serde_json::Value,
    pub confidence: f32,
    pub verified: bool,
    pub verifier_summary: String,
}

// ── ScopeOrchestrator ─────────────────────────────────────────────────────────

/// Deterministic orchestrator: runs probes, keeps verified answers, composes
/// scope.  Contains no model reasoning — that lives entirely in the probes.
pub struct ScopeOrchestrator {
    existing_files: ExistingFilesProbe,
    manifest: ManifestProbe,
    new_file_path: NewFilePathProbe,
}

impl Default for ScopeOrchestrator {
    fn default() -> Self {
        Self {
            existing_files: ExistingFilesProbe::default(),
            manifest: ManifestProbe,
            new_file_path: NewFilePathProbe,
        }
    }
}

impl ScopeOrchestrator {
    /// Run all probes and compose the scope.
    ///
    /// Strategy:
    /// 1. Run `ExistingFilesProbe` (semantic search).
    /// 2. Run `ManifestProbe` (structural, no model).
    /// 3. If both produced zero paths, run `NewFilePathProbe` (model call).
    ///
    /// Scope = union of verified probe paths, deduplicated.
    pub fn run(
        &self,
        workspace_root: &str,
        artifact: &UnderstandingArtifact,
        provider: &dyn ToolProvider,
    ) -> OrchestratorResult {
        let ws = PathBuf::from(workspace_root);
        let ctx = ProbeCtx {
            workspace_root: ws.clone(),
            artifact: artifact.clone(),
        };

        let mut scope: Vec<String> = Vec::new();
        let mut evidence: Vec<EvidenceRef> = Vec::new();
        let mut probe_log: Vec<ProbeLogEntry> = Vec::new();

        // probe 1: existing files
        run_probe(
            &self.existing_files,
            &ctx,
            provider,
            &ws,
            &mut scope,
            &mut evidence,
            &mut probe_log,
        );

        // probe 2: manifests (always — zero cost, purely structural)
        run_probe(
            &self.manifest,
            &ctx,
            provider,
            &ws,
            &mut scope,
            &mut evidence,
            &mut probe_log,
        );

        // probe 3: new file path only when both probes above found nothing
        if scope.is_empty() {
            run_probe(
                &self.new_file_path,
                &ctx,
                provider,
                &ws,
                &mut scope,
                &mut evidence,
                &mut probe_log,
            );
        }

        let candidate_files = scope
            .iter()
            .filter(|path| ws.join(path).exists())
            .map(|path| {
                let source = probe_log
                    .iter()
                    .find(|l| {
                        l.answer
                            .get("paths")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().any(|v| v.as_str() == Some(path.as_str())))
                            .unwrap_or(false)
                            || l.answer.get("path").and_then(|v| v.as_str()) == Some(path.as_str())
                    })
                    .map(|l| l.probe_id.as_str())
                    .unwrap_or("unknown");
                CandidateFile {
                    path: path.clone(),
                    summary: format!("probe scope: {source}"),
                }
            })
            .collect();

        OrchestratorResult {
            target_scope: scope,
            candidate_files,
            evidence,
            probe_log,
        }
    }
}

/// Run one probe: on success verify it; add paths, evidence, log entry.
/// On failure, log the error but continue (other probes may still succeed).
fn run_probe(
    probe: &dyn Probe,
    ctx: &ProbeCtx,
    provider: &dyn ToolProvider,
    workspace: &Path,
    scope: &mut Vec<String>,
    evidence: &mut Vec<EvidenceRef>,
    log: &mut Vec<ProbeLogEntry>,
) {
    let result = match probe.run(ctx, provider) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(probe = probe.id(), error = %e, "probe failed, skipping");
            log.push(ProbeLogEntry {
                probe_id: probe.id().to_string(),
                answer: json!(null),
                confidence: 0.0,
                verified: false,
                verifier_summary: format!("probe error: {e}"),
            });
            return;
        }
    };

    let outcome = probe.verify(&result, workspace);
    let verified = outcome.status == VerifierStatus::Passed;

    log.push(ProbeLogEntry {
        probe_id: probe.id().to_string(),
        answer: result.answer.clone(),
        confidence: result.confidence,
        verified,
        verifier_summary: outcome.summary.clone(),
    });

    if !verified {
        tracing::debug!(probe = probe.id(), summary = %outcome.summary, "probe verification failed");
        return;
    }

    // Extract paths from the answer and add to scope (deduplicated).
    let new_paths = extract_paths(&result.answer);
    for path in new_paths {
        if !scope.contains(&path) {
            scope.push(path);
        }
    }
    for ev in result.evidence {
        if !evidence.iter().any(|e| e.locator == ev.locator) {
            evidence.push(ev);
        }
    }
}

/// Extract paths from a probe answer — handles both `{ paths: [...] }` and
/// `{ path: "..." }` shapes.
fn extract_paths(answer: &serde_json::Value) -> Vec<String> {
    if let Some(arr) = answer.get("paths").and_then(|v| v.as_array()) {
        return arr
            .iter()
            .filter_map(|v| v.as_str())
            .map(str::to_string)
            .collect();
    }
    if let Some(s) = answer.get("path").and_then(|v| v.as_str()) {
        return vec![s.to_string()];
    }
    Vec::new()
}
