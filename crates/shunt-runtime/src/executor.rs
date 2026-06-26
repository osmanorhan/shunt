//! Command executor — the single place where shell commands actually run.
//!
//! Design rules:
//!   - Only structured `CommandSpec` values are accepted; no shell strings.
//!   - Each command runs in the workspace root as its working directory.
//!   - stdout/stderr are captured and truncated to the last 2 KB.
//!   - Commands run sequentially; a failed command does NOT abort the rest.
//!   - No shell expansion — args are passed directly to `std::process::Command`.
//!   - stdout and stderr are drained on separate threads to prevent pipe-buffer
//!     deadlock when the child produces large output.
//!   - A wall-clock timeout kills the child and reports failure.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use shunt_core::{CommandOutcome, CommandSpec};
use tracing::{debug, warn};

const MAX_OUTPUT_BYTES: usize = 2048;
const DEFAULT_TIMEOUT_SECS: u64 = 120;

pub fn run_commands(
    workspace_root: &str,
    commands: &[CommandSpec],
    mut on_start: impl FnMut(&CommandSpec),
    mut on_finish: impl FnMut(&CommandOutcome),
) -> Vec<CommandOutcome> {
    let cwd = Path::new(workspace_root);
    let timeout = Duration::from_secs(DEFAULT_TIMEOUT_SECS);
    commands
        .iter()
        .map(|spec| {
            on_start(spec);
            let outcome = execute_one(cwd, spec, timeout);
            on_finish(&outcome);
            outcome
        })
        .collect()
}

fn execute_one(cwd: &Path, spec: &CommandSpec, timeout: Duration) -> CommandOutcome {
    debug!(program = %spec.program, args = ?spec.args, "executing command");

    let mut child = match Command::new(&spec.program)
        .args(&spec.args)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(err) => {
            warn!(program = %spec.program, %err, "command could not be spawned");
            return CommandOutcome {
                spec: spec.clone(),
                exit_code: -1,
                stdout: String::new(),
                stderr: format!("spawn error: {err}"),
                success: false,
            };
        }
    };

    // Drain stdout and stderr on separate threads so a child that writes more
    // than the pipe buffer (~64 KB on Linux) cannot deadlock waiting for a reader.
    let mut stdout_pipe = child.stdout.take().expect("piped");
    let mut stderr_pipe = child.stderr.take().expect("piped");

    let (tx_out, rx_out) = mpsc::channel::<Vec<u8>>();
    let (tx_err, rx_err) = mpsc::channel::<Vec<u8>>();

    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        let _ = tx_out.send(buf);
    });
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        let _ = tx_err.send(buf);
    });

    // Wait for the child with a real deadline.
    let deadline = Instant::now() + timeout;
    let exit_status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Ok(s),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait(); // reap the zombie
                    warn!(program = %spec.program, secs = timeout.as_secs(), "command timed out");
                    return CommandOutcome {
                        spec: spec.clone(),
                        exit_code: -1,
                        stdout: String::new(),
                        stderr: format!("timed out after {}s — process killed", timeout.as_secs()),
                        success: false,
                    };
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => break Err(e),
        }
    };

    // Pipes closed when child exited; drain threads finish quickly.
    let stdout_bytes = rx_out.recv().unwrap_or_default();
    let stderr_bytes = rx_err.recv().unwrap_or_default();

    match exit_status {
        Ok(status) => {
            let exit_code = status.code().unwrap_or(-1);
            let success = status.success();
            let stdout = tail_utf8(&stdout_bytes, MAX_OUTPUT_BYTES);
            let stderr = tail_utf8(&stderr_bytes, MAX_OUTPUT_BYTES);
            if !success {
                warn!(program = %spec.program, exit_code, stderr = %stderr, "command failed");
            }
            CommandOutcome {
                spec: spec.clone(),
                exit_code,
                stdout,
                stderr,
                success,
            }
        }
        Err(err) => {
            warn!(program = %spec.program, %err, "command wait failed");
            CommandOutcome {
                spec: spec.clone(),
                exit_code: -1,
                stdout: String::new(),
                stderr: format!("wait error: {err}"),
                success: false,
            }
        }
    }
}

fn tail_utf8(bytes: &[u8], max: usize) -> String {
    let s = String::from_utf8_lossy(bytes);
    if s.len() <= max {
        s.into_owned()
    } else {
        format!("…{}", &s[s.len() - max..])
    }
}
