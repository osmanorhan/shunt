use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::{Value, json};

pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: false }
    }
    pub fn err(msg: impl Into<String>) -> Self {
        Self { content: msg.into(), is_error: true }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolInvocation {
    ListFiles { dir: Option<String> },
    ReadFile { path: String },
    Search { query: String },
    EditFile { path: String, mode: EditMode, old: Option<String>, new: Option<String>, content: Option<String> },
    Finish { summary: Option<String> },
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditMode {
    StrReplace,
    Write,
}

/// Parse (name, args) into a typed invocation. Err for unknown tool or schema mismatch.
pub fn parse_invocation(name: &str, args: &Value) -> Result<ToolInvocation, ParseError> {
    match name {
        "list_files" => {
            let dir = args.get("dir").and_then(|v| v.as_str()).map(String::from);
            Ok(ToolInvocation::ListFiles { dir })
        }
        "read_file" => {
            let path = required_str(args, "path")?;
            Ok(ToolInvocation::ReadFile { path })
        }
        "search" => {
            let query = required_str(args, "query")?;
            Ok(ToolInvocation::Search { query })
        }
        "edit_file" => {
            let path = required_str(args, "path")?;
            let mode: EditMode = serde_json::from_value(
                args.get("mode").cloned().unwrap_or(Value::Null),
            )
            .map_err(|e| ParseError::SchemaMismatch(format!("invalid mode: {e}")))?;
            let old = args.get("old").and_then(|v| v.as_str()).map(String::from);
            let new = args.get("new").and_then(|v| v.as_str()).map(String::from);
            let content = args.get("content").and_then(|v| v.as_str()).map(String::from);
            Ok(ToolInvocation::EditFile { path, mode, old, new, content })
        }
        "finish" => {
            let summary = args.get("summary").and_then(|v| v.as_str()).map(String::from);
            Ok(ToolInvocation::Finish { summary })
        }
        other => Err(ParseError::UnknownTool(other.to_string())),
    }
}

#[derive(Debug)]
pub enum ParseError {
    UnknownTool(String),
    SchemaMismatch(String),
}

fn required_str(args: &Value, field: &str) -> Result<String, ParseError> {
    args.get(field)
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| ParseError::SchemaMismatch(format!("missing required field: {field}")))
}

const IGNORE: &[&str] = &[".git", ".shunt", "node_modules", "target"];
const SEARCH_HIT_LIMIT: usize = 50;

pub fn dispatch(inv: &ToolInvocation, workspace: &Path) -> ToolResult {
    match inv {
        ToolInvocation::ListFiles { dir } => {
            let base = dir
                .as_deref()
                .map(|d| workspace.join(d))
                .unwrap_or_else(|| workspace.to_path_buf());
            let mut entries = Vec::new();
            collect_files(&base, workspace, &mut entries);
            if entries.is_empty() {
                ToolResult::ok("(empty)")
            } else {
                ToolResult::ok(entries.join("\n"))
            }
        }
        ToolInvocation::ReadFile { path } => {
            let full = workspace.join(path);
            match fs::read_to_string(&full) {
                Ok(content) => {
                    let numbered: String = content
                        .lines()
                        .enumerate()
                        .map(|(i, l)| format!("{}: {}\n", i + 1, l))
                        .collect();
                    ToolResult::ok(if numbered.is_empty() { "(empty file)".into() } else { numbered })
                }
                Err(e) => ToolResult::err(format!("error reading {path}: {e}")),
            }
        }
        ToolInvocation::Search { query } => {
            let mut hits = Vec::new();
            search_files(workspace, workspace, query, &mut hits);
            if hits.is_empty() {
                ToolResult::ok(format!("no matches for: {query}"))
            } else {
                ToolResult::ok(hits.join("\n"))
            }
        }
        ToolInvocation::EditFile { path, mode, old, new, content } => {
            let full = workspace.join(path);
            match mode {
                EditMode::StrReplace => {
                    let Some(old) = old else {
                        return ToolResult::err("str_replace mode requires 'old'");
                    };
                    let Some(new) = new else {
                        return ToolResult::err("str_replace mode requires 'new'");
                    };
                    match fs::read_to_string(&full) {
                        Ok(existing) => {
                            if !existing.contains(old.as_str()) {
                                return ToolResult::err(format!("'old' string not found in {path}"));
                            }
                            let updated = existing.replacen(old.as_str(), new.as_str(), 1);
                            match fs::write(&full, updated) {
                                Ok(()) => ToolResult::ok(format!("ok: replaced in {path}")),
                                Err(e) => ToolResult::err(format!("error writing {path}: {e}")),
                            }
                        }
                        Err(e) => ToolResult::err(format!("error reading {path}: {e}")),
                    }
                }
                EditMode::Write => {
                    let Some(content) = content else {
                        return ToolResult::err("write mode requires 'content'");
                    };
                    if let Some(parent) = full.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    match fs::write(&full, content) {
                        Ok(()) => ToolResult::ok(format!("ok: wrote {path}")),
                        Err(e) => ToolResult::err(format!("error writing {path}: {e}")),
                    }
                }
            }
        }
        ToolInvocation::Finish { summary } => {
            ToolResult::ok(format!("finish: {}", summary.as_deref().unwrap_or("done")))
        }
    }
}

fn collect_files(dir: &Path, workspace: &Path, out: &mut Vec<String>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut items: Vec<_> = rd.flatten().collect();
    items.sort_by_key(|e| e.file_name());
    for entry in items {
        let name = entry.file_name().to_string_lossy().to_string();
        if IGNORE.contains(&name.as_str()) {
            continue;
        }
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            collect_files(&path, workspace, out);
        } else {
            let rel = path.strip_prefix(workspace).unwrap_or(&path);
            out.push(rel.display().to_string());
        }
    }
}

fn search_files(dir: &Path, workspace: &Path, query: &str, hits: &mut Vec<String>) {
    if hits.len() >= SEARCH_HIT_LIMIT {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut items: Vec<_> = rd.flatten().collect();
    items.sort_by_key(|e| e.file_name());
    for entry in items {
        if hits.len() >= SEARCH_HIT_LIMIT {
            return;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if IGNORE.contains(&name.as_str()) {
            continue;
        }
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            search_files(&path, workspace, query, hits);
        } else if let Ok(content) = fs::read_to_string(&path) {
            let rel = path.strip_prefix(workspace).unwrap_or(&path);
            let rel_str = rel.display().to_string();
            for (i, line) in content.lines().enumerate() {
                if line.contains(query) {
                    hits.push(format!("{}:{}: {}", rel_str, i + 1, line));
                    if hits.len() >= SEARCH_HIT_LIMIT {
                        return;
                    }
                }
            }
        }
    }
}

pub fn schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all files in the workspace (or a subdirectory). Returns relative paths, one per line. Ignores .git, node_modules, and target.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dir": {
                            "type": "string",
                            "description": "Optional subdirectory to list (relative to workspace root). Omit to list all files."
                        }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Return the contents of a file with line numbers. Always read a file before editing it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file from workspace root."
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for a substring across all workspace files. Returns matching lines as path:line_number: content. Returns at most 50 results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Substring to search for."
                        }
                    },
                    "required": ["query"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Edit or create a file. Use mode 'str_replace' to replace the first occurrence of 'old' with 'new'. Use mode 'write' to overwrite or create the file with 'content'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file from workspace root."
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["str_replace", "write"],
                            "description": "'str_replace' replaces the first occurrence of 'old' with 'new'. 'write' overwrites the entire file with 'content'."
                        },
                        "old": {
                            "type": "string",
                            "description": "Required for str_replace: the exact text to find and replace."
                        },
                        "new": {
                            "type": "string",
                            "description": "Required for str_replace: the replacement text."
                        },
                        "content": {
                            "type": "string",
                            "description": "Required for write: the complete new file contents."
                        }
                    },
                    "required": ["path", "mode"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Signal that the task is complete. Call this after making all required changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Optional brief summary of what was done."
                        }
                    }
                }
            }
        }),
    ]
}
