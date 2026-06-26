//! Programmatic fixture workspaces for scenario testing.
//!
//! Each fixture creates a realistic but minimal project on disk inside a
//! temporary directory.  The temp dir is returned so its lifetime controls
//! cleanup; callers hold it alive for the duration of the scenario.

use std::fs;
use std::path::Path;

use tempfile::TempDir;

/// A temporary workspace for one scenario run.
pub struct Workspace {
    pub dir: TempDir,
}

impl Workspace {
    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    pub fn root_str(&self) -> String {
        self.dir.path().display().to_string()
    }
}

/// Minimal Rust library + CLI fixture.
///
/// Structure:
///   Cargo.toml          (serde + clap deps, my-cli package)
///   src/main.rs         (fn main, clap parsing, calls lib functions)
///   src/lib.rs          (pub fn greet, pub fn parse_item)
///   src/utils.rs        (pub fn format_output, pub fn validate_input,
///                        pub fn divide — intentional divide-by-zero risk)
pub fn rust_cli() -> Workspace {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path();

    write(root, "Cargo.toml", RUST_CLI_CARGO_TOML);

    fs::create_dir_all(root.join("src")).unwrap();
    write(root, "src/main.rs", RUST_CLI_MAIN);
    write(root, "src/lib.rs", RUST_CLI_LIB);
    write(root, "src/utils.rs", RUST_CLI_UTILS);

    Workspace { dir }
}

/// Minimal TypeScript + npm fixture.
///
/// Structure:
///   package.json        (deps: zod, uuid)
///   package-lock.json   (minimal stub for lock resolution)
///   src/index.ts        (main entry, imports utils + config)
///   src/utils.ts        (helper functions)
///   src/config.ts       (config loading from env)
///   src/auth.ts         (stub auth module — used by ask_clarify scenario)
pub fn ts_app() -> Workspace {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path();

    write(root, "package.json", TS_PACKAGE_JSON);
    write(root, "package-lock.json", TS_LOCK);

    fs::create_dir_all(root.join("src")).unwrap();
    write(root, "src/index.ts", TS_INDEX);
    write(root, "src/utils.ts", TS_UTILS);
    write(root, "src/config.ts", TS_CONFIG);
    write(root, "src/auth.ts", TS_AUTH);

    Workspace { dir }
}

fn write(root: &Path, rel: &str, contents: &str) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, contents).unwrap();
}

// ── Rust CLI fixture content ────────────────────────────────────────────────

// No external deps — keeps cargo test fast (no network, no crate downloads)
const RUST_CLI_CARGO_TOML: &str = r#"[package]
name = "my-cli"
version = "0.1.0"
edition = "2021"
"#;

const RUST_CLI_MAIN: &str = r#"use my_cli::{parse_item, greet};
use my_cli::utils::format_output;

struct Args {
    input: String,
    name: Option<String>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    Args {
        input: args.get(1).cloned().unwrap_or_default(),
        name: args.get(2).cloned(),
    }
}

fn main() {
    let args = parse_args();
    if let Some(name) = &args.name {
        println!("{}", greet(name));
    }
    let item = parse_item(&args.input);
    println!("{}", format_output(&item));
}
"#;

const RUST_CLI_LIB: &str = r#"pub mod utils;

/// Greet a user by name.
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

/// Parse an input string into a canonical form.
pub fn parse_item(input: &str) -> String {
    input.trim().to_string()
}

/// Count words in a string.
pub fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greet_formats_correctly() {
        assert_eq!(greet("Alice"), "Hello, Alice!");
    }

    #[test]
    fn parse_item_trims() {
        assert_eq!(parse_item("  hello  "), "hello");
    }
}
"#;

const RUST_CLI_UTILS: &str = r#"/// Format an output string with brackets.
pub fn format_output(s: &str) -> String {
    format!("[{}]", s)
}

/// Validate that an input is non-empty.
pub fn validate_input(s: &str) -> bool {
    !s.is_empty()
}

/// Integer division — panics if divisor is zero.
pub fn divide(a: i32, b: i32) -> i32 {
    a / b
}

/// Compute percentage from numerator and denominator.
pub fn percentage(part: f64, total: f64) -> f64 {
    if total == 0.0 {
        return 0.0;
    }
    (part / total) * 100.0
}
"#;

// ── TypeScript fixture content ───────────────────────────────────────────────

const TS_PACKAGE_JSON: &str = r#"{
  "name": "my-ts-app",
  "version": "0.1.0",
  "description": "A demo TypeScript application",
  "main": "src/index.ts",
  "scripts": {
    "build": "tsc",
    "start": "ts-node src/index.ts",
    "test": "jest"
  },
  "dependencies": {
    "zod": "^3.22.4",
    "uuid": "^9.0.0"
  },
  "devDependencies": {
    "typescript": "^5.3.3",
    "@types/node": "^20.11.0",
    "jest": "^29.7.0"
  }
}
"#;

const TS_LOCK: &str = r#"{
  "name": "my-ts-app",
  "version": "0.1.0",
  "lockfileVersion": 3,
  "packages": {
    "node_modules/zod": { "version": "3.22.4" },
    "node_modules/uuid": { "version": "9.0.0" }
  }
}
"#;

const TS_INDEX: &str = r#"import { loadConfig } from './config';
import { formatResult, validateInput } from './utils';
import { createAuth } from './auth';

async function main(): Promise<void> {
  const config = loadConfig();
  const auth = createAuth(config.authMode);
  await auth.initialize();

  const input = process.argv[2] ?? '';
  if (!validateInput(input)) {
    console.error('Invalid input');
    process.exit(1);
  }
  console.log(formatResult(input));
}

main().catch(console.error);
"#;

const TS_UTILS: &str = r#"/**
 * Format a result string for display.
 */
export function formatResult(s: string): string {
  return `[${s.trim()}]`;
}

/**
 * Validate that an input string is non-empty and printable.
 */
export function validateInput(s: string): boolean {
  return s.trim().length > 0;
}

/**
 * Count words in a string.
 */
export function countWords(s: string): number {
  return s.trim().split(/\s+/).filter(Boolean).length;
}

/**
 * Truncate a string to a maximum length.
 */
export function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max) + '...' : s;
}
"#;

const TS_CONFIG: &str = r#"export interface AppConfig {
  authMode: 'oauth' | 'apikey' | 'none';
  apiEndpoint: string;
  timeoutMs: number;
}

/**
 * Load configuration from environment variables.
 */
export function loadConfig(): AppConfig {
  return {
    authMode: (process.env.AUTH_MODE as AppConfig['authMode']) ?? 'none',
    apiEndpoint: process.env.API_ENDPOINT ?? 'http://localhost:3000',
    timeoutMs: parseInt(process.env.TIMEOUT_MS ?? '5000', 10),
  };
}
"#;

const TS_AUTH: &str = r#"import { AppConfig } from './config';

export interface Auth {
  initialize(): Promise<void>;
  getToken(): string | null;
}

/**
 * Create an auth provider based on the configured mode.
 * Currently only 'none' is implemented; oauth and apikey are stubs.
 */
export function createAuth(mode: AppConfig['authMode']): Auth {
  switch (mode) {
    case 'none':
      return {
        initialize: async () => {},
        getToken: () => null,
      };
    case 'oauth':
      // TODO: implement OAuth flow
      throw new Error('OAuth not yet implemented');
    case 'apikey':
      // TODO: implement API key auth
      throw new Error('API key auth not yet implemented');
  }
}
"#;
