//! Command safety classification — the single authority for deciding whether a
//! `CommandSpec` is safe to run automatically, needs user approval, or must be
//! blocked outright.
//!
//! Lives in `shunt-core` so every crate that executes commands (`shunt-runtime`
//! executor/runner, `shunt-infer` agent) shares the same rules without duplication.
//!
//! Classification is purely structural: program name + argument patterns.
//! No model calls, no I/O.
//!
//! Shell command strings are never classified as safe. Parsing shell syntax
//! incompletely creates bypasses through pipelines, compound commands, and
//! wrappers, so `shell -c` always requires explicit approval.

use std::path::Path;

use crate::CommandSpec;

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandSafety {
    /// Run automatically — no user gate required.
    Safe,
    /// Pause and ask the user before running.
    Dangerous { reason: String },
    /// Reject immediately — never run regardless of approval.
    Blocked { reason: String },
}

impl CommandSafety {
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::Safe)
    }
    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Blocked { .. })
    }
}

/// Classify a single command spec.
pub fn classify(spec: &CommandSpec) -> CommandSafety {
    let prog = basename(&spec.program);
    let args: Vec<&str> = spec.args.iter().map(String::as_str).collect();
    classify_inner(prog, &args)
}

/// Classify all commands; return the worst classification across the set.
/// Order: Blocked > Dangerous > Safe.
pub fn classify_all(specs: &[CommandSpec]) -> CommandSafety {
    let mut worst = CommandSafety::Safe;
    for spec in specs {
        match classify(spec) {
            CommandSafety::Blocked { reason } => return CommandSafety::Blocked { reason },
            d @ CommandSafety::Dangerous { .. } => {
                if worst.is_safe() {
                    worst = d;
                }
            }
            CommandSafety::Safe => {}
        }
    }
    worst
}

// ── Internal ──────────────────────────────────────────────────────────────────

fn basename(program: &str) -> &str {
    Path::new(program)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(program)
}

/// Shells that accept `-c "command string"`.
const SHELLS: &[&str] = &["sh", "bash", "zsh", "dash", "fish", "ksh", "csh", "tcsh"];

fn classify_inner(prog: &str, args: &[&str]) -> CommandSafety {
    if SHELLS.contains(&prog) && args.contains(&"-c") {
        return CommandSafety::Dangerous {
            reason: format!("`{prog} -c` executes an opaque shell command"),
        };
    }

    // ── Blocked outright ─────────────────────────────────────────────────────
    if is_blocked(prog, args) {
        return CommandSafety::Blocked {
            reason: format!("`{prog}` is not permitted"),
        };
    }

    // ── Dangerous: needs approval ─────────────────────────────────────────────
    if let Some(reason) = dangerous_reason(prog, args) {
        return CommandSafety::Dangerous { reason };
    }

    CommandSafety::Safe
}

fn is_blocked(prog: &str, args: &[&str]) -> bool {
    // Disk / filesystem destruction.
    if matches!(
        prog,
        "mkfs"
            | "mkfs.ext4"
            | "mkfs.btrfs"
            | "mkfs.vfat"
            | "mkfs.ntfs"
            | "fdisk"
            | "parted"
            | "shred"
            | "dd"
            | "wipefs"
            | "wipe"
    ) {
        return true;
    }
    // rm -rf targeting root-level paths.
    if prog == "rm" {
        let has_rf = args.iter().any(|a| is_rm_recursive_force(a));
        let targets_root = args
            .iter()
            .any(|a| matches!(*a, "/" | "/home" | "/usr" | "/etc" | "/var" | "/boot" | "~"));
        if has_rf && targets_root {
            return true;
        }
    }
    false
}

fn is_rm_recursive_force(arg: &str) -> bool {
    let s = arg.trim_start_matches('-');
    (s.contains('r') || s == "recursive") && (s.contains('f') || s == "force")
}

fn dangerous_reason(prog: &str, args: &[&str]) -> Option<String> {
    match prog {
        "rm" => Some(format!(
            "`rm {}` will delete files permanently",
            args.join(" ")
        )),

        "sudo" | "su" | "doas" => Some(format!("`{prog}` requires elevated privileges")),

        "chmod" | "chown" => {
            if args
                .iter()
                .any(|a| matches!(*a, "-R" | "--recursive" | "-r"))
            {
                Some(format!(
                    "`{prog} -R` will change permissions/ownership recursively"
                ))
            } else {
                None
            }
        }

        "git" => {
            if args.first() == Some(&"push") && args.iter().any(|a| matches!(*a, "--force" | "-f"))
            {
                return Some("`git push --force` will rewrite remote history".into());
            }
            if args.first() == Some(&"reset") && args.contains(&"--hard") {
                return Some("`git reset --hard` will discard uncommitted changes".into());
            }
            if args.first() == Some(&"clean") && args.contains(&"-f") {
                return Some("`git clean -f` will delete untracked files".into());
            }
            None
        }

        "mv" => {
            if args.last().map(|a| a.starts_with('/')).unwrap_or(false) {
                Some(format!(
                    "`mv` target is an absolute path: {}",
                    args.last().unwrap_or(&"?")
                ))
            } else {
                None
            }
        }

        "truncate" => Some("`truncate` will destroy file contents".into()),

        "kill" | "pkill" | "killall" => {
            if args
                .iter()
                .any(|a| matches!(*a, "-9" | "-KILL" | "--signal=9"))
            {
                Some(format!("`{prog} -9` sends SIGKILL"))
            } else {
                None
            }
        }

        "curl" | "wget" => {
            if args
                .iter()
                .any(|a| *a == "|" || a.ends_with("| sh") || a.ends_with("| bash"))
            {
                Some("`curl | sh` executes remote code".into())
            } else {
                None
            }
        }

        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CommandSpec;

    fn cmd(program: &str, args: &[&str]) -> CommandSpec {
        CommandSpec::new(program, args.iter().copied())
    }

    #[test]
    fn npm_install_is_safe() {
        assert!(classify(&cmd("npm", &["install"])).is_safe());
    }

    #[test]
    fn cargo_build_is_safe() {
        assert!(classify(&cmd("cargo", &["build"])).is_safe());
    }

    #[test]
    fn rm_is_dangerous() {
        assert!(matches!(
            classify(&cmd("rm", &["src/old.rs"])),
            CommandSafety::Dangerous { .. }
        ));
    }

    #[test]
    fn rm_rf_root_is_blocked() {
        assert!(classify(&cmd("rm", &["-rf", "/"])).is_blocked());
    }

    #[test]
    fn sudo_is_dangerous() {
        assert!(matches!(
            classify(&cmd("sudo", &["apt", "install", "x"])),
            CommandSafety::Dangerous { .. }
        ));
    }

    #[test]
    fn git_push_force_is_dangerous() {
        assert!(matches!(
            classify(&cmd("git", &["push", "--force"])),
            CommandSafety::Dangerous { .. }
        ));
    }

    #[test]
    fn git_commit_is_safe() {
        assert!(classify(&cmd("git", &["commit", "-m", "msg"])).is_safe());
    }

    #[test]
    fn pip_install_is_safe() {
        assert!(classify(&cmd("pip", &["install", "requests"])).is_safe());
    }

    #[test]
    fn mkfs_is_blocked() {
        assert!(classify(&cmd("mkfs.ext4", &["/dev/sda1"])).is_blocked());
    }

    // Shell wrapper unwrapping — the key regression tests for the bypass fix.

    #[test]
    fn sh_c_mkfs_requires_approval() {
        assert!(matches!(
            classify(&cmd("sh", &["-c", "mkfs.ext4 /dev/sda"])),
            CommandSafety::Dangerous { .. }
        ));
    }

    #[test]
    fn bash_c_rm_rf_root_requires_approval() {
        assert!(matches!(
            classify(&cmd("bash", &["-c", "rm -rf /"])),
            CommandSafety::Dangerous { .. }
        ));
    }

    #[test]
    fn sh_c_cargo_build_requires_approval() {
        assert!(matches!(
            classify(&cmd("sh", &["-c", "cargo build"])),
            CommandSafety::Dangerous { .. }
        ));
    }

    #[test]
    fn compound_shell_command_requires_approval() {
        assert!(matches!(
            classify(&cmd("sh", &["-c", "cargo build; rm -rf /"])),
            CommandSafety::Dangerous { .. }
        ));
    }
}
