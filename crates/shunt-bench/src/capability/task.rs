//! Capability tasks — graded, self-contained fixtures with on-disk ground truth.
//!
//! Each task probes a capability dimension we care about for small-model editing
//! (positioning, content generation, convergence, multi-site edits) at a known
//! difficulty. **To add a task, append one `CapabilityTask` to `suite()`** — the
//! runner and scorecard pick it up automatically.

use std::fs;

use tempfile::TempDir;

use crate::fixtures::Workspace;

pub type ContentCheck = (&'static str, fn(&str) -> bool);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Difficulty {
    Trivial,
    Easy,
    Medium,
    Hard,
}

impl Difficulty {
    pub fn label(self) -> &'static str {
        match self {
            Difficulty::Trivial => "trivial",
            Difficulty::Easy => "easy",
            Difficulty::Medium => "medium",
            Difficulty::Hard => "hard",
        }
    }
}

/// A single capability probe. Pure data + two function pointers, so the suite is
/// a flat list anyone can extend.
pub struct CapabilityTask {
    pub name: &'static str,
    pub difficulty: Difficulty,
    /// The instruction handed to the agent (issue-style).
    pub request: &'static str,
    /// Files to materialise in the workspace: (relative path, contents).
    pub files: &'static [(&'static str, &'static str)],
    /// Ground truth checks over final on-disk file contents: (relative path, check).
    pub checks: &'static [ContentCheck],
}

impl CapabilityTask {
    /// Build a fresh temp workspace with this task's files.
    pub fn workspace(&self) -> Workspace {
        let dir = TempDir::new().expect("tempdir");
        for (rel, contents) in self.files {
            let path = dir.path().join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("mkdir");
            }
            fs::write(&path, contents).expect("write fixture");
        }
        Workspace { dir }
    }

    pub fn full_request(&self) -> String {
        self.request.to_string()
    }

    /// True if the target file now satisfies ground truth.
    pub fn passed(&self, ws: &Workspace) -> bool {
        self.checks.iter().all(|(rel, check)| {
            let content = fs::read_to_string(ws.root().join(rel)).unwrap_or_default();
            check(&content)
        })
    }
}

// ── The suite ─────────────────────────────────────────────────────────────────

pub fn suite() -> Vec<CapabilityTask> {
    vec![
        // T1: trivial single-token change in a tiny file. Tests the basic
        // read→locate→replace loop. Most models should pass this.
        CapabilityTask {
            name: "change_constant",
            difficulty: Difficulty::Trivial,
            request: "In src/config.ts change the default timeout from 5000 to 30000.",
            files: &[(
                "src/config.ts",
                "export interface Cfg { timeoutMs: number; }\n\n\
                 export function loadConfig(): Cfg {\n  \
                 return { timeoutMs: parseInt(process.env.TIMEOUT_MS ?? '5000', 10) };\n}\n",
            )],
            checks: &[("src/config.ts", |c| {
                c.contains("30000") && !c.contains("5000")
            })],
        },
        // T2: add a self-contained function. Tests content generation.
        CapabilityTask {
            name: "add_function",
            difficulty: Difficulty::Easy,
            request: "Add a public function `farewell` to src/lib.rs that takes \
                      name: &str and returns a String saying goodbye to that name.",
            files: &[(
                "src/lib.rs",
                "/// Greet a user by name.\npub fn greet(name: &str) -> String {\n    \
                 format!(\"Hello, {name}!\")\n}\n",
            )],
            checks: &[("src/lib.rs", |c| {
                c.contains("fn farewell") && c.contains("greet")
            })],
        },
        // T3: fix a bug on a SPECIFIC line among look-alikes in a moderate file.
        // Tests positioning + discrimination (two identical `return value;` lines).
        CapabilityTask {
            name: "fix_clamp",
            difficulty: Difficulty::Medium,
            request: "The clamp(value, min, max) function returns `value` when value \
                      is greater than max, but it should return `max`. Fix it.",
            files: &[(
                "src/util.ts",
                "export function add(a: number, b: number): number {\n  return a + b;\n}\n\n\
                 export function clamp(value: number, min: number, max: number): number {\n  \
                 if (value < min) {\n    return min;\n  }\n  \
                 if (value > max) {\n    return value;\n  }\n  return value;\n}\n\n\
                 export function isEven(n: number): boolean {\n  return n % 2 === 0;\n}\n",
            )],
            // The value>max branch must now return max (not value). Cheap proxy:
            // the file references `max` in a return and still has clamp intact.
            checks: &[("src/util.ts", |c| {
                c.contains("return max") && c.contains("function clamp")
            })],
        },
        // T4: two-site change (rename a symbol used in two files). Tests
        // convergence across multiple edits without thrashing.
        CapabilityTask {
            name: "rename_two_sites",
            difficulty: Difficulty::Hard,
            request: "Rename the function `greet` to `greet_user` in src/lib.rs and \
                      update its caller in src/main.rs.",
            files: &[
                (
                    "src/lib.rs",
                    "pub fn greet(name: &str) -> String {\n    format!(\"Hi, {name}\")\n}\n",
                ),
                (
                    "src/main.rs",
                    "use crate::greet;\nfn main() {\n    println!(\"{}\", greet(\"Al\"));\n}\n",
                ),
            ],
            checks: &[
                ("src/lib.rs", |c| {
                    c.contains("fn greet_user") && !c.contains("fn greet(")
                }),
                ("src/main.rs", |c| {
                    c.contains("greet_user") && !c.contains("greet(")
                }),
            ],
        },
        // T5: add a branch in the right order. Tests small control-flow insertion
        // without disturbing the existing disabled-user behavior.
        CapabilityTask {
            name: "add_locked_branch",
            difficulty: Difficulty::Medium,
            request: "In src/status.ts, make statusForUser return 'locked' for locked users before the active-user check.",
            files: &[(
                "src/status.ts",
                "export interface User { active: boolean; locked: boolean; disabled: boolean; }\n\n\
                 export function statusForUser(user: User): string {\n  \
                 if (user.disabled) {\n    return 'disabled';\n  }\n  \
                 if (user.active) {\n    return 'active';\n  }\n  return 'pending';\n}\n",
            )],
            checks: &[(
                "src/status.ts",
                // prettier may convert single→double quotes; check content not quote style
                |c| {
                    c.contains("user.locked")
                        && c.contains("return")
                        && c.contains("locked")
                        && c.contains("disabled")
                },
            )],
        },
        // T6: remove a legacy branch while preserving the remaining cases.
        CapabilityTask {
            name: "remove_legacy_mode",
            difficulty: Difficulty::Medium,
            request: "In src/mode.ts, stop accepting the 'legacy' mode. Only 'modern' and 'strict' should be valid.",
            files: &[(
                "src/mode.ts",
                "export type Mode = 'modern' | 'strict';\n\n\
                 export function parseMode(input: string): Mode {\n  \
                 if (input === 'modern') {\n    return 'modern';\n  }\n  \
                 if (input === 'legacy') {\n    return 'modern';\n  }\n  \
                 if (input === 'strict') {\n    return 'strict';\n  }\n  \
                 throw new Error(`Unknown mode: ${input}`);\n}\n",
            )],
            checks: &[(
                "src/mode.ts",
                // prettier may convert single→double quotes; check content not quote style
                |c| !c.contains("legacy") && c.contains("modern") && c.contains("strict"),
            )],
        },
        // T7: rename a symbol across export, import, and call sites.
        CapabilityTask {
            name: "rename_export_import_call",
            difficulty: Difficulty::Hard,
            request: "Rename the exported function `sum` to `addNumbers` in src/math.ts and update src/report.ts to import and call the new name.",
            files: &[
                (
                    "src/math.ts",
                    "export function sum(a: number, b: number): number {\n  return a + b;\n}\n",
                ),
                (
                    "src/report.ts",
                    "import { sum } from './math';\n\nexport function total(items: number[]): number {\n  return items.reduce((acc, item) => sum(acc, item), 0);\n}\n",
                ),
            ],
            checks: &[
                ("src/math.ts", |c| {
                    c.contains("function addNumbers") && !c.contains("function sum")
                }),
                ("src/report.ts", |c| {
                    c.contains("addNumbers") && !c.contains("sum(")
                }),
            ],
        },
        // T8: thread a new config field through construction and use sites.
        CapabilityTask {
            name: "thread_config_field",
            difficulty: Difficulty::Hard,
            request: "Add `retry_count: usize` to ClientConfig in src/client.rs, default it to 3, and pass it into connect_with_timeout.",
            files: &[(
                "src/client.rs",
                "pub struct ClientConfig {\n    pub timeout_ms: u64,\n}\n\n\
                 impl Default for ClientConfig {\n    fn default() -> Self {\n        \
                 Self { timeout_ms: 1000 }\n    }\n}\n\n\
                 pub fn build_client(cfg: ClientConfig) -> Client {\n    \
                 connect_with_timeout(cfg.timeout_ms)\n}\n\n\
                 pub struct Client;\n\n\
                 fn connect_with_timeout(_timeout_ms: u64) -> Client {\n    Client\n}\n",
            )],
            checks: &[("src/client.rs", |c| {
                c.contains("retry_count: usize")
                    && c.contains("retry_count: 3")
                    && c.contains("cfg.retry_count")
                    && c.contains("connect_with_timeout")
            })],
        },
        // T9: change production behavior and keep the test expectation in sync.
        CapabilityTask {
            name: "sync_pricing_test",
            difficulty: Difficulty::Hard,
            request: "Change sales tax from 8% to 10% in src/pricing.ts and update the matching test expectation in tests/pricing.test.ts.",
            files: &[
                (
                    "src/pricing.ts",
                    "export function totalWithTax(subtotal: number): number {\n  return subtotal * 1.08;\n}\n",
                ),
                (
                    "tests/pricing.test.ts",
                    "import { totalWithTax } from '../src/pricing';\n\nit('applies sales tax', () => {\n  expect(totalWithTax(100)).toBe(108);\n});\n",
                ),
            ],
            checks: &[
                ("src/pricing.ts", |c| {
                    (c.contains("1.10") || c.contains("1.1")) && !c.contains("1.08")
                }),
                ("tests/pricing.test.ts", |c| {
                    c.contains("110") && !c.contains("108")
                }),
            ],
        },
        // T10: move error handling to the caller instead of swallowing failures in
        // a helper. Tests a multi-block semantic edit in one file.
        CapabilityTask {
            name: "move_error_handling",
            difficulty: Difficulty::Hard,
            request: "In src/users.ts, stop swallowing errors inside fetchJson. Let fetchJson throw, and make loadUser catch failures and return `{ kind: 'unavailable' }`.",
            files: &[(
                "src/users.ts",
                "type UserResult = { kind: 'user'; name: string } | { kind: 'missing' } | { kind: 'unavailable' };\n\n\
                 async function fetchJson(url: string): Promise<any | null> {\n  \
                 try {\n    const res = await fetch(url);\n    return await res.json();\n  } catch {\n    return null;\n  }\n}\n\n\
                 export async function loadUser(id: string): Promise<UserResult> {\n  \
                 const data = await fetchJson(`/users/${id}`);\n  if (!data) {\n    return { kind: 'missing' };\n  }\n  \
                 return { kind: 'user', name: data.name };\n}\n",
            )],
            checks: &[(
                "src/users.ts",
                // prettier may convert single→double quotes; check content not quote style
                |c| {
                    !c.contains("return null")
                        && c.contains("catch")
                        && c.contains("unavailable")
                        && c.contains("missing")
                },
            )],
        },
        // T11: service-style task with a new route plus a smoke script. This is closer
        // to Terminal-Bench tasks because the verifier can legitimately start the
        // server, curl it, and inspect behavior instead of only reading files.
        CapabilityTask {
            name: "node_health_route_smoke",
            difficulty: Difficulty::Hard,
            request: "In this small Node service, add a GET /healthz endpoint that returns JSON with `{ ok: true, version: process.env.APP_VERSION ?? 'dev' }`, keep the existing /users/:id route working, and add a new smoke script at scripts/smoke-health.sh that curls the health endpoint.",
            files: &[
                (
                    "package.json",
                    "{\n  \"name\": \"mini-service\",\n  \"version\": \"1.0.0\",\n  \"type\": \"module\",\n  \"scripts\": {\n    \"start\": \"node src/server.js\"\n  }\n}\n",
                ),
                (
                    "src/server.js",
                    "import http from 'node:http';\n\nfunction json(res, status, body) {\n  res.writeHead(status, { 'content-type': 'application/json' });\n  res.end(JSON.stringify(body));\n}\n\nconst server = http.createServer((req, res) => {\n  if (!req.url) {\n    return json(res, 400, { error: 'missing_url' });\n  }\n\n  if (req.url.startsWith('/users/')) {\n    const id = req.url.slice('/users/'.length);\n    return json(res, 200, { id, role: 'member' });\n  }\n\n  return json(res, 404, { error: 'not_found' });\n});\n\nserver.listen(process.env.PORT ?? 3000);\n",
                ),
            ],
            checks: &[
                ("src/server.js", |c| {
                    c.contains("/healthz")
                        && c.contains("APP_VERSION")
                        && c.contains("ok")
                        && c.contains("/users/")
                }),
                ("scripts/smoke-health.sh", |c| {
                    c.contains("curl") && c.contains("healthz")
                }),
            ],
        },
        // T12: manifest + implementation + tests. The model should update package.json,
        // refactor the code, and keep the test expectation in sync.
        CapabilityTask {
            name: "dependency_refactor_slugify",
            difficulty: Difficulty::Hard,
            request: "Add the npm dependency `slugify`, replace the manual slug building in src/title.js with `slugify(title, { lower: true, trim: true, strict: true })`, and update the test in tests/title.test.js to assert that punctuation is removed.",
            files: &[
                (
                    "package.json",
                    "{\n  \"name\": \"title-tools\",\n  \"version\": \"1.0.0\",\n  \"type\": \"module\",\n  \"scripts\": {\n    \"test\": \"node --test\"\n  },\n  \"dependencies\": {}\n}\n",
                ),
                (
                    "package-lock.json",
                    "{\n  \"name\": \"title-tools\",\n  \"lockfileVersion\": 3,\n  \"packages\": {}\n}\n",
                ),
                (
                    "src/title.js",
                    "export function toTitleId(title) {\n  return title\n    .trim()\n    .toLowerCase()\n    .replace(/[^a-z0-9\\s-]/g, '')\n    .replace(/\\s+/g, '-')\n    .replace(/-+/g, '-');\n}\n",
                ),
                (
                    "tests/title.test.js",
                    "import test from 'node:test';\nimport assert from 'node:assert/strict';\nimport { toTitleId } from '../src/title.js';\n\ntest('builds a title id', () => {\n  assert.equal(toTitleId(' Hello, World! 2026 '), 'hello,-world!-2026');\n});\n",
                ),
            ],
            checks: &[
                ("package.json", |c| c.contains("slugify")),
                ("src/title.js", |c| {
                    c.contains("slugify(") && c.contains("strict: true")
                }),
                ("tests/title.test.js", |c| {
                    c.contains("hello-world-2026") && !c.contains("hello,-world!-2026")
                }),
            ],
        },
        // T13: thread a new flag through config, CLI parsing, and rendering. This
        // forces the model to touch multiple files and keep behavior/docs aligned.
        CapabilityTask {
            name: "thread_json_flag_cli",
            difficulty: Difficulty::Hard,
            request: "Add a `--json` flag to this Rust CLI. Parse it in src/main.rs, thread it through ReportOptions in src/report.rs, make render_report output compact JSON when json is true, and update README.md usage to mention the new flag.",
            files: &[
                (
                    "Cargo.toml",
                    "[package]\nname = \"report-cli\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
                ),
                (
                    "src/main.rs",
                    "mod report;\n\nuse report::{render_report, ReportOptions};\n\nfn main() {\n    let args: Vec<String> = std::env::args().collect();\n    let verbose = args.iter().any(|arg| arg == \"--verbose\");\n    let options = ReportOptions { verbose };\n    println!(\"{}\", render_report(7, &options));\n}\n",
                ),
                (
                    "src/report.rs",
                    "pub struct ReportOptions {\n    pub verbose: bool,\n}\n\npub fn render_report(total: usize, options: &ReportOptions) -> String {\n    if options.verbose {\n        format!(\"processed {total} records\")\n    } else {\n        total.to_string()\n    }\n}\n",
                ),
                (
                    "README.md",
                    "# report-cli\n\nUsage:\n\n`cargo run -- --verbose`\n",
                ),
            ],
            checks: &[
                ("src/main.rs", |c| {
                    c.contains("--json")
                        && (c.contains("json:") || c.contains("ReportOptions { verbose, json }"))
                }),
                ("src/report.rs", |c| {
                    c.contains("pub json: bool")
                        && c.contains("total")
                        && c.contains("options.json")
                }),
                ("README.md", |c| c.contains("--json")),
            ],
        },
    ]
}
