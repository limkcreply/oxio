//! Built-in tools. Read tools (read_file, list_dir, glob, read_many_files, grep)
//! run ungated. Write tools (write_file, …) declare `kind()=Write`, so the
//! `safety::Guarded` wrapper fronts them with a permission prompt - this crate
//! carries no permission logic itself.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use base64::Engine as _;
use oxio_core::{Ctx, OxioError, Result, Tool, ToolKind, ToolOutput, ToolSpec};

pub mod docs;
pub mod sandbox;
use globset::GlobBuilder;
use grep_regex::RegexMatcherBuilder;
use grep_searcher::{sinks::UTF8, Searcher};
use ignore::WalkBuilder;
use serde_json::{json, Value};

/// Files modified within this window are "recent" and sorted newest-first;
/// older ones sort alphabetically.
const RECENCY_WINDOW: Duration = Duration::from_secs(12 * 3600);

/// Cap tool output so a huge file can't blow the context (real spill-to-file
/// lands with the tools/safety step).
const MAX_OUTPUT: usize = 64 * 1024;

/// Hard cap on grep match rows, so a too-broad pattern can't flood the context
/// even before the byte cap bites.
const MAX_MATCHES: usize = 1000;

fn cap(mut s: String) -> String {
    if s.len() > MAX_OUTPUT {
        s.truncate(MAX_OUTPUT);
        s.push_str("\n…[truncated]");
    }
    s
}

fn arg_str<'a>(input: &'a Value, key: &str, tool: &str) -> Result<&'a str> {
    input
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| OxioError::Tool(format!("{tool}: missing string arg '{key}'")))
}

fn arg_u64(input: &Value, key: &str) -> Option<u64> {
    input.get(key).and_then(|v| v.as_u64())
}

/// Default line cap when no `limit` is given (matches vendor defaults).
const DEFAULT_LINE_LIMIT: usize = 2000;

/// Shared file reader with a binary/encoding guard. `Ok(text)` or `Err(reason)`
/// where reason is a short human note. Used by BOTH `read_file` and
/// `read_many_files` so the guard lives in one place.
fn read_guarded(path: &Path) -> std::result::Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.contains(&0) {
        return Err(format!("binary ({} bytes)", bytes.len()));
    }
    String::from_utf8(bytes).map_err(|_| "not valid UTF-8".to_string())
}

fn is_globby(s: &str) -> bool {
    s.contains('*') || s.contains('?') || s.contains('[') || s.contains('{')
}

/// The literal leading directory of a glob pattern (before the first metachar),
/// used as the walk root. `"src/**/*.rs"` -> `"src"`; `"**/*.rs"` -> `"."`.
fn glob_base(pattern: &str) -> PathBuf {
    let mut base = PathBuf::new();
    for comp in Path::new(pattern).components() {
        if is_globby(&comp.as_os_str().to_string_lossy()) {
            break;
        }
        base.push(comp.as_os_str());
    }
    if base.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        base
    }
}

/// A gitignore-honoring recursive walker used by the file-collecting tools.
fn walker(base: &Path) -> WalkBuilder {
    let mut wb = WalkBuilder::new(base);
    wb.git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .parents(true)
        .require_git(false);
    wb
}

/// One-line, length-capped rendering of a string arg for `Tool::summarize` - collapses
/// whitespace/newlines and truncates, so a long command or query stays one tidy line.
fn one_line_arg(input: &Value, field: &str, cap: usize) -> Option<String> {
    let v = input.get(field).and_then(|x| x.as_str())?.trim();
    if v.is_empty() {
        return None;
    }
    let one = v.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() > cap {
        Some(format!("{}…", one.chars().take(cap).collect::<String>()))
    } else {
        Some(one)
    }
}

/// File paths named in a patch envelope (`*** Update/Add/Delete File:`), for summarizing
/// an apply_patch call as the files it touches rather than the raw patch text.
fn patch_files(patch: &str) -> Vec<String> {
    patch
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            ["*** Update File:", "*** Add File:", "*** Delete File:"]
                .iter()
                .find_map(|pfx| l.strip_prefix(pfx).map(|r| r.trim().to_string()))
        })
        .collect()
}

#[cfg(test)]
mod summarize_tests {
    use super::*;
    use oxio_core::Tool;
    use serde_json::json;

    #[test]
    fn tools_render_readable_summaries() {
        assert_eq!(
            ReadFile.summarize(&json!({"path": "src/main.rs"})),
            "Read src/main.rs"
        );
        assert_eq!(
            ReadFile.summarize(&json!({"path": "b.xlsx", "sheet": "Data"})),
            "Read b.xlsx (sheet Data)"
        );
        assert_eq!(ListDir.summarize(&json!({"path": "crates"})), "List crates");
        assert_eq!(
            GlobTool.summarize(&json!({"pattern": "**/*.rs"})),
            "Find **/*.rs"
        );
        assert_eq!(
            WriteFile.summarize(&json!({"path": "x.txt"})),
            "Write x.txt"
        );
        assert_eq!(
            ApplyPatch.summarize(
                &json!({"patch": "*** Begin Patch\n*** Update File: a.rs\n*** End Patch"})
            ),
            "Edit a.rs"
        );
        // The bash/web_search summaries build on one_line_arg - a multi-line command
        // collapses to one capped line.
        assert_eq!(
            one_line_arg(&json!({"command": "echo hi\n  && ls -la"}), "command", 80).as_deref(),
            Some("echo hi && ls -la")
        );
    }

    #[test]
    fn decorators_forward_summarize_to_inner() {
        // The kernel calls summarize on the OUTERMOST wrapper - it must reach the tool.
        let hinted = Hinting::wrap(Arc::new(ReadFile), &BTreeMap::new(), true);
        assert_eq!(hinted.summarize(&json!({"path": "a.rs"})), "Read a.rs");
    }
}

/// Read a UTF-8 text file.
pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: "Read a text file, or a PDF / DOCX / XLSX (text is extracted locally). \
Returns line-numbered content. For large files, use `offset` (1-based start line) and `limit` \
(line count) to read a range. For a spreadsheet, `sheet` reads just one sheet by name (the \
first read lists the sheet names). A scanned image-only PDF has no text to extract."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "file path" },
                    "offset": { "type": "integer", "description": "1-based start line (optional)" },
                    "limit": { "type": "integer", "description": "max lines to read (optional)" },
                    "sheet": { "type": "string", "description": "spreadsheet: read only this sheet, by name or 1-based index (optional)" }
                },
                "required": ["path"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    fn summarize(&self, input: &Value) -> String {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        if let Some(sheet) = input
            .get("sheet")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
        {
            format!("Read {path} (sheet {sheet})")
        } else if let Some(off) = input.get("offset").and_then(|v| v.as_u64()) {
            format!("Read {path} (from line {off})")
        } else {
            format!("Read {path}")
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let path = arg_str(&input, "path", "read_file")?;
        let p = Path::new(path);

        // Documents (pdf/docx/xlsx) → extract their text locally so a text-only model can
        // read them (the plain reader below would reject them as binary). A scanned PDF has
        // no text layer, so say so plainly rather than return nothing.
        let sheet = input
            .get("sheet")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty());
        let text = if let Some(kind) = docs::DocKind::from_path(p) {
            // Targeted single-sheet read (spreadsheets only) once the AI has picked a sheet.
            if let (Some(sheet), true) = (sheet, matches!(kind, docs::DocKind::Xlsx)) {
                match docs::extract_sheet(p, sheet) {
                    Ok(t) => t,
                    Err(e) => return Ok(ToolOutput::error(format!("read_file: {path}: {e}"))),
                }
            } else {
                match docs::extract(p, kind) {
                    Ok(ex) if ex.scanned_pdf => {
                        return Ok(ToolOutput::ok(format!(
                            "[{path}: scanned PDF - no text layer to extract. It needs OCR or a \
vision model; use view_image on a page image, or configure a vision provider.]"
                        )));
                    }
                    Ok(ex) => ex.text,
                    Err(e) => return Ok(ToolOutput::error(format!("read_file: {path}: {e}"))),
                }
            }
        } else {
            // Shared guarded reader (also used by read_many_files) - one binary/UTF-8 guard.
            match read_guarded(p) {
                Ok(t) => t,
                Err(reason) => {
                    return Ok(ToolOutput::error(format!("read_file: {path}: {reason}")))
                }
            }
        };

        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        if total == 0 {
            return Ok(ToolOutput::ok(String::new()));
        }
        let start = arg_u64(&input, "offset")
            .map(|n| (n as usize).max(1))
            .unwrap_or(1);
        let want = arg_u64(&input, "limit")
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LINE_LIMIT);
        if start > total {
            return Ok(ToolOutput::ok(format!(
                "[offset {start} is past end of file - {total} total lines]"
            )));
        }

        let start_idx = start - 1;
        let end_idx = (start_idx + want).min(total);
        let mut body = String::new();
        for (i, line) in lines[start_idx..end_idx].iter().enumerate() {
            body.push_str(&format!("{:>6}\t{}\n", start + i, line));
        }
        // Truncation-status contract.
        if start_idx > 0 || end_idx < total {
            let mut note = format!("[showing lines {}-{} of {}", start, end_idx, total);
            if end_idx < total {
                note.push_str(&format!(" - use offset={} to continue", end_idx + 1));
            }
            note.push(']');
            body.push_str(&note);
        }
        Ok(ToolOutput::ok(cap(body)))
    }
}

/// List the entries of a directory.
pub struct ListDir;

#[async_trait]
impl Tool for ListDir {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".into(),
            description:
                "List entries in a directory. Honors .gitignore by default (hides ignored \
files and dotfiles); set all=true to show everything. Directories are marked with a trailing /."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "any absolute directory path on the machine (not limited to the working directory)" },
                    "all": { "type": "boolean", "description": "show ignored + hidden entries (default false)" }
                },
                "required": ["path"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    fn summarize(&self, input: &Value) -> String {
        match input
            .get("path")
            .and_then(|v| v.as_str())
            .filter(|p| !p.is_empty())
        {
            Some(p) => format!("List {p}"),
            None => "List directory".into(),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let path = arg_str(&input, "path", "list_dir")?;
        if let Err(e) = std::fs::metadata(path) {
            return Ok(ToolOutput::error(format!("list_dir: {e}")));
        }
        let all = input.get("all").and_then(|v| v.as_bool()).unwrap_or(false);

        // ripgrep's `ignore` walker: honor .gitignore/.ignore + skip dotfiles unless `all`.
        // `require_git(false)` so .gitignore is respected even outside a git repo.
        let mut wb = WalkBuilder::new(path);
        wb.max_depth(Some(1))
            .hidden(!all)
            .git_ignore(!all)
            .git_global(!all)
            .git_exclude(!all)
            .ignore(!all)
            .parents(!all)
            .require_git(false);

        let mut names = Vec::new();
        for entry in wb.build() {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if entry.depth() == 0 {
                continue; // the directory itself
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            names.push(if is_dir { format!("{name}/") } else { name });
        }
        names.sort();
        Ok(ToolOutput::ok(cap(names.join("\n"))))
    }
}

/// Find files matching a glob pattern, honoring .gitignore, recent files first.
pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "glob".into(),
            description: "Find files matching a glob pattern (e.g. `**/*.rs`). Honors .gitignore. \
Recently modified files are listed first."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "glob, e.g. **/*.rs" },
                    "path": { "type": "string", "description": "any absolute base directory (default .; widen to a parent or home dir to search beyond the working directory)" },
                    "case_sensitive": { "type": "boolean", "description": "default false" }
                },
                "required": ["pattern"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    fn summarize(&self, input: &Value) -> String {
        match input
            .get("pattern")
            .and_then(|v| v.as_str())
            .filter(|p| !p.is_empty())
        {
            Some(p) => format!("Find {p}"),
            None => "Find files".into(),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let pattern = arg_str(&input, "pattern", "glob")?;
        let base = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let case_sensitive = input
            .get("case_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if let Err(e) = std::fs::metadata(base) {
            return Ok(ToolOutput::error(format!("glob: {e}")));
        }
        let matcher = match GlobBuilder::new(pattern)
            .case_insensitive(!case_sensitive)
            .build()
        {
            Ok(g) => g.compile_matcher(),
            Err(e) => return Ok(ToolOutput::error(format!("glob: bad pattern: {e}"))),
        };

        let wb = walker(Path::new(base));

        let mut hits: Vec<(String, SystemTime)> = Vec::new();
        for entry in wb.build() {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let full = entry.path();
            let rel = full.strip_prefix(base).unwrap_or(full);
            if matcher.is_match(rel) {
                let mtime = entry
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                hits.push((full.display().to_string(), mtime));
            }
        }

        let now = SystemTime::now();
        hits.sort_by(|a, b| {
            let ar = now
                .duration_since(a.1)
                .map(|d| d < RECENCY_WINDOW)
                .unwrap_or(false);
            let br = now
                .duration_since(b.1)
                .map(|d| d < RECENCY_WINDOW)
                .unwrap_or(false);
            match (ar, br) {
                (true, true) => b.1.cmp(&a.1), // both recent → newest first
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => a.0.cmp(&b.0), // alphabetical by path
            }
        });

        if hits.is_empty() {
            return Ok(ToolOutput::ok(format!(
                "[no files matched {pattern}]{}",
                widen_cue(base)
            )));
        }
        let list: Vec<String> = hits.into_iter().map(|(p, _)| p).collect();
        Ok(ToolOutput::ok(cap(list.join("\n"))))
    }
}

/// Read a KNOWN set of files in one call. Deliberate bulk read - NOT for search.
pub struct ReadManyFiles;

#[async_trait]
impl Tool for ReadManyFiles {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_many_files".into(),
            description:
                "Read a KNOWN set of files at once: specific file paths, a directory (walked, \
.gitignore honored), and/or glob patterns (e.g. src/**/*.rs). Use this for deliberate bulk reads. \
To SEARCH for something, prefer `grep`/`glob` then `read_file` the hit - that is far more \
token-efficient than bulk-reading. Binary files are skipped; total output is capped."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "paths": { "type": "array", "items": { "type": "string" }, "description": "file paths, directory paths, or glob patterns" },
                    "exclude": { "type": "array", "items": { "type": "string" }, "description": "glob patterns to exclude (optional)" }
                },
                "required": ["paths"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    fn summarize(&self, input: &Value) -> String {
        let paths = input.get("paths").and_then(|v| v.as_array());
        match paths.map(|a| a.len()).unwrap_or(0) {
            0 => "Read files".into(),
            1 => {
                let one = paths.and_then(|a| a[0].as_str()).unwrap_or("");
                format!("Read {one}")
            }
            n => format!("Read {n} files"),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let paths = input
            .get("paths")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                OxioError::Tool("read_many_files: 'paths' must be an array of strings".into())
            })?;
        if paths.is_empty() {
            return Ok(ToolOutput::error("read_many_files: 'paths' is empty"));
        }

        let mut excludes = Vec::new();
        if let Some(arr) = input.get("exclude").and_then(|v| v.as_array()) {
            for e in arr.iter().filter_map(|v| v.as_str()) {
                match GlobBuilder::new(e).build() {
                    Ok(g) => excludes.push(g.compile_matcher()),
                    Err(err) => {
                        return Ok(ToolOutput::error(format!(
                            "read_many_files: bad exclude '{e}': {err}"
                        )))
                    }
                }
            }
        }

        let mut set: BTreeSet<PathBuf> = BTreeSet::new();
        for entry in paths.iter().filter_map(|v| v.as_str()) {
            if is_globby(entry) {
                let matcher = match GlobBuilder::new(entry).build() {
                    Ok(g) => g.compile_matcher(),
                    Err(err) => {
                        return Ok(ToolOutput::error(format!(
                            "read_many_files: bad pattern '{entry}': {err}"
                        )))
                    }
                };
                for e in walker(&glob_base(entry)).build().flatten() {
                    if !e.file_type().map(|t| t.is_file()).unwrap_or(false) {
                        continue;
                    }
                    let full = e.path();
                    let s = full.to_string_lossy();
                    let cand = s.strip_prefix("./").unwrap_or(&s);
                    if matcher.is_match(cand) {
                        set.insert(full.to_path_buf());
                    }
                }
            } else {
                let p = PathBuf::from(entry);
                if p.is_file() {
                    set.insert(p);
                } else if p.is_dir() {
                    for e in walker(&p).build().flatten() {
                        if e.file_type().map(|t| t.is_file()).unwrap_or(false) {
                            set.insert(e.path().to_path_buf());
                        }
                    }
                }
            }
        }

        let mut files: Vec<PathBuf> = set.into_iter().collect();
        if !excludes.is_empty() {
            files.retain(|f| {
                let s = f.to_string_lossy();
                let cand = s.strip_prefix("./").unwrap_or(&s);
                !excludes.iter().any(|m| m.is_match(cand))
            });
        }
        if files.is_empty() {
            return Ok(ToolOutput::ok("[no files matched]".to_string()));
        }

        let total = files.len();
        let mut body = String::new();
        for (shown, f) in files.iter().enumerate() {
            let content = match read_guarded(f) {
                Ok(t) => cap(t),
                Err(reason) => format!("[{reason}; skipped]"),
            };
            let section = format!("===== {} =====\n{}\n\n", f.display(), content);
            if !body.is_empty() && body.len() + section.len() > MAX_OUTPUT {
                body.push_str(&format!(
                    "[showing {shown} of {total} files - output cap reached; narrow `paths`, add `exclude`, or grep+read_file to search]"
                ));
                return Ok(ToolOutput::ok(body));
            }
            body.push_str(&section);
        }
        Ok(ToolOutput::ok(body))
    }
}

/// Appended to a search tool's ZERO-result output: a neutral cue that the tool is NOT
/// path-bound and can explore wider. Empty is often legitimately empty, so this informs
/// (retry broader if the target may be elsewhere) rather than asserting failure - it
/// targets the model's "nothing here → I can't" misfire by naming the next move.
fn widen_cue(scope: &str) -> String {
    format!(
        "\n[0 results in {scope}. Not limited to here - you can search ANY path on this machine. \
If you EXPECTED a match that exists, it's likely elsewhere: consider retrying a broader path (the parent, \
the repo root, or a home dir like /Users/<you>), up to machine-wide. If 0 is a valid answer here - you \
were checking absence, or the target may simply not exist - ignore this. It's a hint, not a requirement.]"
    )
}

/// Appended to a resource-existence PROBE that came back negative (not found here). A
/// probe checks whether some resource - executable, package, module, service - is
/// available; the resource is a variable the model chose (`manim`, `figma`, anything),
/// never something this layer knows. Neutral by design: one location was checked, the
/// thing may live elsewhere, and if it is genuinely absent that is a valid answer.
/// This closes the exact miss where a single `command -v X` / `import X` probe is taken
/// as proof of absence.
fn probe_cue() -> String {
    "\n[Not found HERE - but this checked ONE location (usually PATH or the system \
environment). Before concluding it is absent, it may live elsewhere: a project-local \
environment (.venv/venv/conda/pyenv/node_modules), a sibling or versioned/dated path, or a \
non-PATH install. Consider searching wider - e.g. `find` from a parent or home directory, or \
inspect the project's environment dirs - then re-probe that path. If it is genuinely not \
installed, that is a valid answer; ignore this cue.]"
        .to_string()
}

/// True when a command's PURPOSE is to check whether a RESOURCE exists / is available - an
/// executable, package, or module. A negative result from such a probe is a CLUE to look
/// wider, not proof of absence. Matches the probe SHAPE only (locator builtin, existence
/// flag, or a package/module query); the resource keyword is whatever the model passed and
/// is deliberately irrelevant here - nothing is hardcoded to a specific tool or package.
fn is_resource_probe(cmd: &str) -> bool {
    // Executable locators: `which X`, `command -v X`, `type X`, `whereis X`, `hash X`.
    const LOCATORS: [&str; 5] = ["which", "command", "type", "whereis", "hash"];
    let toks: Vec<&str> = cmd
        .split(|c: char| c.is_whitespace() || matches!(c, '|' | ';' | '&' | '(' | ')' | '`'))
        .filter(|t| !t.is_empty())
        .collect();
    if toks.is_empty() {
        return false;
    }
    if toks.iter().any(|t| LOCATORS.contains(t)) {
        return true;
    }
    // Existence/availability flags on any program: `X --version`, `X -V`, `X --help`.
    if toks
        .iter()
        .any(|t| matches!(*t, "--version" | "-V" | "--help"))
    {
        return true;
    }
    // Package/module existence queries (command + subcommand shape).
    for pair in toks.windows(2) {
        match (pair[0], pair[1]) {
            ("pip" | "pip3", "show")
            | ("npm", "ls" | "list")
            | ("gem", "which" | "list")
            | ("brew", "list")
            | ("dpkg", "-l")
            | ("cargo", "tree") => return true,
            _ => {}
        }
    }
    // Python import probe: `python -c "import X"`.
    if (toks.contains(&"python") || toks.contains(&"python3")) && cmd.contains("import ") {
        return true;
    }
    false
}

/// Appended to a cloud/remote LOG query that came back with ZERO entries. The whole point:
/// a query returning nothing is a fact about the QUERY, not about the world - turning "0
/// logs" into "no errors happened" is the failure. Names the real traps: a narrow/relative
/// time window that drifts (especially across an approval delay - the query was composed at
/// one time and may run much later), a wrong log group/region, or an over-tight filter.
/// Neutral: if the window and scope were genuinely right, 0 is a valid answer.
fn cloud_log_cue() -> String {
    "\n[0 log entries returned - this is a fact about your QUERY, not proof that nothing \
happened. Do NOT report \"no errors\" from this alone. The usual trap is the time window: a \
narrow/RELATIVE window (\"last N minutes\", a now-anchored start) DRIFTS - if approval lapsed \
between composing and running this, \"recent\" no longer means what it did. A narrow/recent window \
is only right when you are SURE the event is happening NOW - live, real-time debugging you are \
actively driving. Otherwise treat the time as UNKNOWN: do not guess a range - a rare error may be \
a single event years old, so any guessed window misses it. Instead filter by the error \
KEYWORD/pattern over the WIDEST available window \
(up to full retention), sorted NEWEST-FIRST with a limit - e.g. Insights `filter @message like \
/(?i)error/ | sort @timestamp desc | limit 50` - so the latest matching errors surface whenever \
they occurred. Also verify the log group / project / region and loosen the filter. Don't lean on \
the user for a time range - they usually can't say when it happened. Only after a full-retention, \
keyword-filtered, newest-first scan is still empty is absence plausible.]"
        .to_string()
}

/// True when a command queries CLOUD or remote LOGS (where a bad time window/filter silently
/// yields zero results). Matches the command SHAPE across providers; no keyword/service names.
fn is_cloud_log_query(cmd: &str) -> bool {
    let toks: Vec<&str> = cmd
        .split(|c: char| c.is_whitespace() || matches!(c, '|' | ';' | '&' | '(' | ')' | '`'))
        .filter(|t| !t.is_empty())
        .collect();
    if toks.windows(2).any(|w| {
        matches!(
            (w[0], w[1]),
            ("aws", "logs")
                | ("gcloud", "logging")
                | ("kubectl", "logs")
                | ("docker", "logs")
                | ("stern", _)
        )
    }) {
        return true;
    }
    // Azure log queries don't use an adjacent pair.
    cmd.contains("log-analytics") || cmd.contains("activity-log")
}

/// True when a cloud-log-query's OUTPUT indicates zero results even though stdout isn't empty
/// - the CLIs return JSON like `{"events": []}` / `{"results": []}` rather than nothing.
fn looks_like_no_log_results(stdout: &[u8]) -> bool {
    let s = String::from_utf8_lossy(stdout);
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed == "[]" || trimmed == "{}" {
        return true;
    }
    let compact: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    [
        "\"events\":[]",
        "\"results\":[]",
        "\"logEvents\":[]",
        "\"entries\":[]",
        "\"rows\":[]",
    ]
    .iter()
    .any(|marker| compact.contains(marker))
}

/// True when a shell command's PURPOSE is finding/searching - so an empty result plausibly
/// means "look wider" (vs a non-search command that just happens to print nothing). Kept to
/// unambiguous searchers; `ls`/`cat` are excluded (empty is usually the real answer there).
fn is_search_command(cmd: &str) -> bool {
    const SEARCHERS: [&str; 7] = ["find", "grep", "rg", "ag", "locate", "mdfind", "fd"];
    cmd.split(|c: char| c.is_whitespace() || matches!(c, '|' | ';' | '&' | '(' | '`'))
        .any(|tok| SEARCHERS.contains(&tok))
}

/// Regex content search, ripgrep-backed. The primary "find where X is" tool.
pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep".into(),
            description: "Search file CONTENTS for a regex, recursively from a path (.gitignore honored, \
binaries skipped). This is the tool to FIND where something is; then `read_file` the hit. Output is \
`file:line:text`, capped. Use `glob` to filter which files are searched."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "regex to search for (Rust regex syntax)" },
                    "path": { "type": "string", "description": "any absolute path to search (default: current dir; pass a parent, repo root, or a home dir to look beyond it - this is not limited to the working directory)" },
                    "glob": { "type": "string", "description": "only search files matching this glob (e.g. **/*.rs)" },
                    "case_sensitive": { "type": "boolean", "description": "default true; set false for case-insensitive" }
                },
                "required": ["pattern"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    fn summarize(&self, input: &Value) -> String {
        let pat = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
        match input
            .get("path")
            .and_then(|v| v.as_str())
            .filter(|p| !p.is_empty() && *p != ".")
        {
            Some(p) => format!("Search \"{pat}\" in {p}"),
            None => format!("Search \"{pat}\""),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let pattern = arg_str(&input, "pattern", "grep")?;
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let case_sensitive = input
            .get("case_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let matcher = match RegexMatcherBuilder::new()
            .case_insensitive(!case_sensitive)
            .build(pattern)
        {
            Ok(m) => m,
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "grep: bad regex '{pattern}': {e}"
                )))
            }
        };

        let glob_matcher = match input.get("glob").and_then(|v| v.as_str()) {
            Some(g) => match GlobBuilder::new(g).build() {
                Ok(built) => Some(built.compile_matcher()),
                Err(e) => return Ok(ToolOutput::error(format!("grep: bad glob '{g}': {e}"))),
            },
            None => None,
        };

        let base = Path::new(path);
        let mut searcher = Searcher::new();
        let mut out = String::new();
        let mut total = 0usize;
        let mut files = 0usize;
        let mut truncated = false;

        'outer: for dent in walker(base).build().flatten() {
            if !dent.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let fpath = dent.path();
            if let Some(gm) = &glob_matcher {
                let s = fpath.to_string_lossy();
                let cand = s.strip_prefix("./").unwrap_or(&s);
                if !gm.is_match(cand) {
                    continue;
                }
            }

            let mut hits: Vec<(u64, String)> = Vec::new();
            // Per-file errors (unreadable/odd encoding) are non-fatal: skip the file.
            let _ = searcher.search_path(
                &matcher,
                fpath,
                UTF8(|lnum, line| {
                    hits.push((lnum, line.trim_end_matches(['\r', '\n']).to_string()));
                    Ok(true)
                }),
            );
            if hits.is_empty() {
                continue;
            }
            files += 1;
            for (lnum, text) in hits {
                let row = format!("{}:{}:{}\n", fpath.display(), lnum, text);
                if total >= MAX_MATCHES || out.len() + row.len() > MAX_OUTPUT {
                    truncated = true;
                    break 'outer;
                }
                out.push_str(&row);
                total += 1;
            }
        }

        if out.is_empty() {
            return Ok(ToolOutput::ok(format!("[no matches]{}", widen_cue(path))));
        }
        if truncated {
            out.push_str(&format!(
                "[{total} matches across {files} files shown - cap reached; narrow `pattern`, `path`, or `glob`]"
            ));
        }
        Ok(ToolOutput::ok(out))
    }
}

/// Whole-file write. `kind()=Write`, so `safety::Guarded` fronts it with a
/// permission prompt; this tool itself just writes.
pub struct WriteFile;

#[async_trait]
impl Tool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Write CONTENT to a file at PATH, creating parent directories as needed and \
overwriting any existing file. Returns the path and byte count. For a targeted change to an existing \
file prefer `edit`; use this for a new file or a full rewrite."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "file path to write" },
                    "content": { "type": "string", "description": "full file contents" }
                },
                "required": ["path", "content"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Write
    }
    fn summarize(&self, input: &Value) -> String {
        match input
            .get("path")
            .and_then(|v| v.as_str())
            .filter(|p| !p.is_empty())
        {
            Some(p) => format!("Write {p}"),
            None => "Write file".into(),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let path = arg_str(&input, "path", "write_file")?;
        let content = arg_str(&input, "content", "write_file")?;
        let p = Path::new(path);
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return Ok(ToolOutput::error(format!(
                        "write_file: cannot create dir '{}': {e}",
                        parent.display()
                    )));
                }
            }
        }
        match std::fs::write(p, content) {
            Ok(()) => Ok(ToolOutput::ok(format!(
                "wrote {} bytes to {path}",
                content.len()
            ))),
            Err(e) => Ok(ToolOutput::error(format!("write_file: {e}"))),
        }
    }
}

// --- apply_patch (edit): the V4A patch FORMAT ---
// v1 locates Update hunks by EXACT line-context match and fails cleanly when a hunk
// cannot be found. Fuzzy/whitespace-tolerant matching and `@@`-marker disambiguation
// are a deferred enhancement.

#[derive(Debug)]
enum PatchOp {
    Add {
        path: String,
        contents: String,
    },
    Delete {
        path: String,
    },
    Update {
        path: String,
        move_to: Option<String>,
        hunks: Vec<Hunk>,
    },
}

/// One contiguous change: `before` = context + removed lines (what to find);
/// `after` = context + added lines (the replacement).
#[derive(Debug, Default)]
struct Hunk {
    before: Vec<String>,
    after: Vec<String>,
}

fn parse_patch(patch: &str) -> std::result::Result<Vec<PatchOp>, String> {
    let mut lines = patch.lines().peekable();
    match lines.next() {
        Some(l) if l.trim() == "*** Begin Patch" => {}
        _ => return Err("patch must start with '*** Begin Patch'".into()),
    }
    let mut ops = Vec::new();
    while let Some(&line) = lines.peek() {
        if line.trim() == "*** End Patch" {
            return Ok(ops);
        }
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            let path = path.trim().to_string();
            lines.next();
            let mut contents = String::new();
            while let Some(&l) = lines.peek() {
                if l.starts_with("*** ") {
                    break;
                }
                let c = l
                    .strip_prefix('+')
                    .ok_or_else(|| format!("Add File line must start with '+': {l}"))?;
                contents.push_str(c);
                contents.push('\n');
                lines.next();
            }
            ops.push(PatchOp::Add { path, contents });
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            let path = path.trim().to_string();
            lines.next();
            ops.push(PatchOp::Delete { path });
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            let path = path.trim().to_string();
            lines.next();
            let mut move_to = None;
            if let Some(&l) = lines.peek() {
                if let Some(m) = l.strip_prefix("*** Move to: ") {
                    move_to = Some(m.trim().to_string());
                    lines.next();
                }
            }
            let mut hunks = Vec::new();
            let mut cur = Hunk::default();
            while let Some(&l) = lines.peek() {
                if l.starts_with("*** ") {
                    break;
                }
                if l.starts_with("@@") {
                    // hunk boundary: bank the current hunk if it has content.
                    if !cur.before.is_empty() || !cur.after.is_empty() {
                        hunks.push(std::mem::take(&mut cur));
                    }
                    lines.next();
                    continue;
                }
                if let Some(rest) = l.strip_prefix('+') {
                    cur.after.push(rest.to_string());
                } else if let Some(rest) = l.strip_prefix('-') {
                    cur.before.push(rest.to_string());
                } else {
                    let rest = l.strip_prefix(' ').unwrap_or(l);
                    cur.before.push(rest.to_string());
                    cur.after.push(rest.to_string());
                }
                lines.next();
            }
            if !cur.before.is_empty() || !cur.after.is_empty() {
                hunks.push(cur);
            }
            ops.push(PatchOp::Update {
                path,
                move_to,
                hunks,
            });
        } else {
            return Err(format!("unexpected line in patch: {line}"));
        }
    }
    Err("patch missing '*** End Patch'".into())
}

/// Corrective-hint layer - a oxio innovation. Vendors bury a format hint inside
/// each tool's error path; oxio instead makes it a COMPOSABLE, CONFIG-DRIVEN
/// decorator: wrap any tool so that when it returns an ERROR, a simple, direct usage
/// hint (config override first, else a built-in default) is appended to steer the
/// model's retry back to correct usage. Zero-cost on success (only fires on error),
/// never a leniency (the tool's own contract stays strict), and expandable to any
/// tool via `[hints.<tool>]`. Meaningful for capable models (~20B+) that make an
/// occasional slip; a model too weak to follow the hint is out of scope by design.
pub struct Hinting {
    inner: Arc<dyn Tool>,
    name: String,
    /// User `[hints.<tool>]` override (static). When absent, the built-in escalating
    /// hint picks from the retry count + actual error at call time.
    config_hint: Option<String>,
    /// Consecutive error count for this tool (reset on any success). Drives ESCALATION:
    /// the 1st failure gets the COMPREHENSIVE all-upfront hint (fix everything at once -
    /// cheapest when the model just needs the full spec); a 2nd+ consecutive failure
    /// means the comprehensive hint was too heavy to act on, so switch to a GRANULAR,
    /// error-aware hint that targets only the specific remaining mistake. This is the
    /// user's design: broad first, narrow on repeat - and a live diagnostic of whether
    /// one upfront hint is digestible for the model.
    errors: AtomicUsize,
}

impl Hinting {
    /// Wrap `inner` with the corrective-hint layer when enabled and a hint could apply
    /// (config override, or the tool has built-in hints). Otherwise return `inner`
    /// unchanged - no wrapper, no overhead. `enabled=false` disables it (strong model).
    pub fn wrap(
        inner: Arc<dyn Tool>,
        hints: &BTreeMap<String, String>,
        enabled: bool,
    ) -> Arc<dyn Tool> {
        if !enabled {
            return inner;
        }
        let name = inner.spec().name;
        let config_hint = hints.get(&name).cloned().filter(|h| !h.trim().is_empty());
        if config_hint.is_none() && !has_default_hint(&name) {
            return inner;
        }
        Arc::new(Hinting {
            inner,
            name,
            config_hint,
            errors: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl Tool for Hinting {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }
    fn kind(&self) -> ToolKind {
        self.inner.kind()
    }
    fn summarize(&self, input: &Value) -> String {
        self.inner.summarize(input)
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> Result<ToolOutput> {
        let out = self.inner.call(input, ctx).await?;
        if !out.is_error {
            // Success clears the streak - the next failure starts broad again.
            self.errors.store(0, Ordering::Relaxed);
            return Ok(out);
        }
        // ESCALATE by consecutive-error count: 1st failure → comprehensive (fix it all
        // at once); 2nd+ → granular, error-aware (the broad hint didn't land, so aim at
        // the specific remaining mistake). Config override wins over both.
        let n = self.errors.fetch_add(1, Ordering::Relaxed) + 1;
        let hint = self.config_hint.clone().or_else(|| {
            if n <= 1 {
                comprehensive_hint(&self.name).map(str::to_string)
            } else {
                default_hint(&self.name, &out.content).map(str::to_string)
            }
        });
        match hint {
            // Don't double-append if the hint text is already present in the error.
            Some(h) if !out.content.contains(h.trim()) => {
                Ok(ToolOutput::error(format!("{}\n{}", out.content, h)))
            }
            _ => Ok(out),
        }
    }
}

/// Whether a tool has any built-in hint (so `wrap` knows to install the layer).
fn has_default_hint(tool: &str) -> bool {
    matches!(tool, "apply_patch")
}

/// COMPREHENSIVE (all-upfront) hint for a tool's FIRST failure - states every common
/// gotcha at once so a capable model can fix them in a single retry, no sequential
/// discovery. Cheap when the whole spec is short (apply_patch = envelope + context).
/// If the model STILL fails after this, the escalation falls through to the GRANULAR
/// `default_hint`, which targets the one specific remaining mistake.
fn comprehensive_hint(tool: &str) -> Option<&'static str> {
    match tool {
        "apply_patch" => Some(
            "Resend as a full patch, minding BOTH rules:\n\
1) ENVELOPE - start with '*** Begin Patch', end with '*** End Patch', exactly:\n\
*** Begin Patch\n\
*** Update File: path/to/file\n\
@@\n\
 context line (leading space keeps it)\n\
-removed line\n\
+added line\n\
*** End Patch\n\
2) CONTEXT - the space-prefixed context lines must match the file EXACTLY (whitespace \
and punctuation included); re-read the file to copy them if unsure. For a large change, \
use write_file instead.",
        ),
        _ => None,
    }
}

/// Built-in ERROR-AWARE corrective hint: chosen from the tool name AND the actual
/// error text, so the model is coached on the REAL failure, not a generic blob. Used
/// on the 2nd+ consecutive failure (after `comprehensive_hint` didn't land), so it
/// narrows to the one thing still wrong. A user `[hints.<tool>]` entry overrides this.
/// This escalating coaching layer is a oxio original - vendors bury a single static
/// hint inside each tool.
fn default_hint(tool: &str, error: &str) -> Option<&'static str> {
    match tool {
        "apply_patch" => {
            let format_problem = error.contains("Begin Patch")
                || error.contains("End Patch")
                || error.contains("must start")
                || error.contains("Update File")
                || error.contains("no file operations");
            if format_problem {
                // Envelope / structure is wrong.
                Some(
                    "Resend the FULL patch in EXACTLY this format:\n\
*** Begin Patch\n\
*** Update File: path/to/file\n\
@@\n\
 context line (leading space keeps it)\n\
-removed line\n\
+added line\n\
*** End Patch\n\
It MUST start with '*** Begin Patch' and end with '*** End Patch'.",
                )
            } else {
                // Envelope was fine - the hunk context didn't match the file.
                Some(
                    "The patch format is valid, but its context lines were not found in the file, \
so nothing could be located to change. Re-read the file and copy the surrounding context \
lines EXACTLY (whitespace and punctuation included) into the hunk - or use write_file to \
rewrite the whole file if the change is large.",
                )
            }
        }
        _ => None,
    }
}

/// One diff line for the approval preview: kind (`' '` context, `'-'` removed,
/// `'+'` added, `'#'` file header) + the text. The UI colours by kind.
pub type DiffLine = (char, String);

/// Colored-diff preview of a pending write/edit, for the approval prompt - so the
/// user SEES which lines change (red removed / green added) before allowing it.
/// Empty for tools we don't preview. No external diff crate: a compact
/// LCS line diff (capped to stay cheap on huge files).
pub fn diff_preview(tool: &str, input: &Value) -> Vec<DiffLine> {
    match tool {
        "write_file" => {
            let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let old = std::fs::read_to_string(path).unwrap_or_default();
            let verb = if old.is_empty() { "create" } else { "update" };
            let mut out = vec![('#', format!("{verb} {path}"))];
            out.extend(line_diff(&old, content));
            out
        }
        "apply_patch" => {
            let patch = input.get("patch").and_then(|v| v.as_str()).unwrap_or("");
            let ops = match parse_patch(patch) {
                Ok(o) => o,
                // Honest to the USER: name the real reason (model's patch is off-spec),
                // not a vague "unparseable". The model gets its own corrective hint.
                Err(e) => return vec![('#', format!("malformed patch - {e}"))],
            };
            let mut out = Vec::new();
            for op in &ops {
                match op {
                    PatchOp::Add { path, contents } => {
                        out.push(('#', format!("add {path}")));
                        out.extend(contents.lines().map(|l| ('+', l.to_string())));
                    }
                    PatchOp::Delete { path } => out.push(('#', format!("delete {path}"))),
                    PatchOp::Update {
                        path,
                        move_to,
                        hunks,
                    } => {
                        let hdr = match move_to {
                            Some(m) => format!("update {path} → {m}"),
                            None => format!("update {path}"),
                        };
                        out.push(('#', hdr));
                        for h in hunks {
                            out.extend(line_diff(&h.before.join("\n"), &h.after.join("\n")));
                        }
                    }
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Minimal LCS line diff → context/removed/added lines. O(n·m); falls back to a flat
/// remove-then-add for very large inputs so the preview never gets expensive.
fn line_diff(old: &str, new: &str) -> Vec<DiffLine> {
    let a: Vec<&str> = if old.is_empty() {
        Vec::new()
    } else {
        old.lines().collect()
    };
    let b: Vec<&str> = if new.is_empty() {
        Vec::new()
    } else {
        new.lines().collect()
    };
    let (n, m) = (a.len(), b.len());
    if n > 3000 || m > 3000 {
        let mut out: Vec<DiffLine> = a.iter().map(|l| ('-', l.to_string())).collect();
        out.extend(b.iter().map(|l| ('+', l.to_string())));
        return out;
    }
    let mut dp = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if a[i] == b[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < n && j < m {
        if a[i] == b[j] {
            out.push((' ', a[i].to_string()));
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            out.push(('-', a[i].to_string()));
            i += 1;
        } else {
            out.push(('+', b[j].to_string()));
            j += 1;
        }
    }
    while i < n {
        out.push(('-', a[i].to_string()));
        i += 1;
    }
    while j < m {
        out.push(('+', b[j].to_string()));
        j += 1;
    }
    out
}

/// First index where `needle` occurs contiguously in `hay`.
/// Which lenience tier located a hunk's context. Reported so a loose match (the model's
/// context was sloppy) is visible rather than silent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MatchTier {
    Exact,
    TrailingWs,
    Trimmed,
    Normalized,
}

impl MatchTier {
    fn label(self) -> &'static str {
        match self {
            MatchTier::Exact => "exact",
            MatchTier::TrailingWs => "trailing-whitespace",
            MatchTier::Trimmed => "whitespace",
            MatchTier::Normalized => "unicode-normalized",
        }
    }
}

/// Why a hunk's context could not be located to a single place.
#[derive(Debug, PartialEq, Eq)]
enum SeekError {
    /// No position matched at any lenience tier.
    NotFound,
    /// The context matched in more than one place - refusing to guess which.
    Ambiguous(usize),
}

/// A per-tier line normaliser used by the fuzzy matcher (identity, rstrip, trim, unicode).
type Normalizer = fn(&str) -> String;

fn norm_exact(s: &str) -> String {
    s.to_string()
}
fn norm_rstrip(s: &str) -> String {
    s.trim_end().to_string()
}
fn norm_trim(s: &str) -> String {
    s.trim().to_string()
}

/// Normalise common Unicode punctuation to ASCII (dashes, smart quotes, exotic spaces)
/// so an ASCII-authored patch still matches source that uses typographic characters.
fn norm_unicode(s: &str) -> String {
    s.trim()
        .chars()
        .map(|c| match c {
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            '\u{00A0}' | '\u{2002}' | '\u{2003}' | '\u{2004}' | '\u{2005}' | '\u{2006}'
            | '\u{2007}' | '\u{2008}' | '\u{2009}' | '\u{200A}' | '\u{202F}' | '\u{205F}'
            | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

/// Locate `needle` in `hay` with DECREASING strictness - exact → ignore trailing ws →
/// trim both sides → unicode-normalise, so a near-miss context (a reflowed line, stray
/// whitespace, a smart quote) still applies instead of failing and forcing a model retry.
///
/// Two safeguards:
///  1. **Ambiguity guard.** Returning the FIRST match at each tier is unsafe - if the context
///     appears in >1 place it silently patches the first, which can corrupt the wrong
///     occurrence. We instead REFUSE (`SeekError::Ambiguous`) so the caller can ask the
///     model for more context.
///  2. **Tier reporting.** We return WHICH tier matched, so a loose (sloppy-context) match
///     is surfaced rather than applied silently.
fn find_seq(
    hay: &[String],
    needle: &[String],
) -> std::result::Result<(usize, usize, MatchTier), SeekError> {
    let n = needle.len();
    if n == 0 || n > hay.len() {
        return Err(SeekError::NotFound);
    }
    let tiers: [(MatchTier, Normalizer); 4] = [
        (MatchTier::Exact, norm_exact),
        (MatchTier::TrailingWs, norm_rstrip),
        (MatchTier::Trimmed, norm_trim),
        (MatchTier::Normalized, norm_unicode),
    ];
    for (tier, norm) in tiers {
        let hits: Vec<usize> = (0..=hay.len() - n)
            .filter(|&s| (0..n).all(|k| norm(&hay[s + k]) == norm(&needle[k])))
            .collect();
        match hits.len() {
            0 => continue,
            1 => return Ok((hits[0], hits[0] + n, tier)),
            c => return Err(SeekError::Ambiguous(c)),
        }
    }
    Err(SeekError::NotFound)
}

#[cfg(test)]
mod patch_matcher_ab {
    //! A/B evidence for the fuzzy-matcher port: the SAME near-miss patch contexts run
    //! through the pre-port EXACT-ONLY matcher (baseline) and the new FUZZY matcher. It
    //! quantifies the failure reduction with zero model/token noise - the deterministic
    //! Layer-1 of the fuzzy-vs-hint comparison. (Layer-2, live token counts against the
    //! local model, runs separately.)
    use super::{find_seq, MatchTier, SeekError};

    /// The pre-port EXACT-ONLY matcher, kept verbatim as the A/B baseline.
    fn exact_only(hay: &[String], needle: &[String]) -> Option<(usize, usize)> {
        let n = needle.len();
        if n == 0 || n > hay.len() {
            return None;
        }
        (0..=hay.len() - n)
            .find(|&s| hay[s..s + n] == needle[..])
            .map(|s| (s, s + n))
    }

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn ab_fuzzy_accepts_near_misses_that_exact_rejects() {
        // Each case is a realistic near-miss a local model produces: the needle is "close"
        // but not byte-identical to the file. Exact-only must FAIL every one; fuzzy must
        // APPLY every one - that gap is the whole value of the port.
        let cases: Vec<(Vec<String>, Vec<String>)> = vec![
            // 1. file has trailing whitespace the model dropped
            (
                v(&["fn a() {", "    let x = 1;   ", "}"]),
                v(&["    let x = 1;"]),
            ),
            // 2. model dropped leading indentation on the context
            (
                v(&["    if cond {", "        run();", "    }"]),
                v(&["if cond {", "run();", "}"]),
            ),
            // 3. file uses smart double-quotes, patch uses ASCII
            (v(&["let s = \u{201C}hi\u{201D};"]), v(&["let s = \"hi\";"])),
            // 4. file uses an em dash, patch uses ASCII hyphen
            (
                v(&["// step one \u{2014} then two"]),
                v(&["// step one - then two"]),
            ),
            // 5. file has a non-breaking space, patch a normal space
            (v(&["let a\u{00A0}= b;"]), v(&["let a = b;"])),
        ];

        let exact_hits = cases
            .iter()
            .filter(|(h, n)| exact_only(h, n).is_some())
            .count();
        let fuzzy_hits = cases.iter().filter(|(h, n)| find_seq(h, n).is_ok()).count();

        // A/B result: exact-only catches NONE; fuzzy catches ALL. Every one of these would,
        // under exact-only, fail → fire the hint → make the model re-read + retry.
        assert_eq!(exact_hits, 0, "exact-only should reject every near-miss");
        assert_eq!(
            fuzzy_hits,
            cases.len(),
            "fuzzy should accept every near-miss"
        );
    }

    #[test]
    fn fuzzy_prefers_exact_and_reports_tier() {
        let hay = v(&["a", "b", "c"]);
        assert_eq!(
            find_seq(&hay, &v(&["b", "c"])),
            Ok((1, 3, MatchTier::Exact))
        );
        // A trailing-whitespace-only difference is reported at that tier, not Exact.
        let hay = v(&["x   ", "y"]);
        assert_eq!(
            find_seq(&hay, &v(&["x", "y"])),
            Ok((0, 2, MatchTier::TrailingWs))
        );
    }

    #[test]
    fn ambiguous_context_is_refused_not_first_matched() {
        // Silently patching the first "dup" would corrupt the wrong one. We refuse and report
        // the count so the model can add context - never corrupt the wrong occurrence.
        let hay = v(&["x", "dup", "y", "dup", "z"]);
        assert_eq!(find_seq(&hay, &v(&["dup"])), Err(SeekError::Ambiguous(2)));
    }
}

fn apply_ops(ops: Vec<PatchOp>) -> std::result::Result<String, String> {
    let mut summary = Vec::new();
    for op in ops {
        match op {
            PatchOp::Add { path, contents } => {
                let p = Path::new(&path);
                if let Some(parent) = p.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent).map_err(|e| format!("{path}: {e}"))?;
                    }
                }
                std::fs::write(p, &contents).map_err(|e| format!("{path}: {e}"))?;
                summary.push(format!("added {path} ({} bytes)", contents.len()));
            }
            PatchOp::Delete { path } => {
                std::fs::remove_file(&path).map_err(|e| format!("{path}: {e}"))?;
                summary.push(format!("deleted {path}"));
            }
            PatchOp::Update {
                path,
                move_to,
                hunks,
            } => {
                let original =
                    std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
                let had_trailing_nl = original.ends_with('\n');
                let mut lines: Vec<String> = original.lines().map(|s| s.to_string()).collect();
                for (i, h) in hunks.iter().enumerate() {
                    if h.before.is_empty() {
                        return Err(format!("{path}: hunk {} has no context to locate", i + 1));
                    }
                    let (start, end, tier) = find_seq(&lines, &h.before).map_err(|e| match e {
                        SeekError::NotFound => {
                            format!("{path}: could not locate hunk {} in file", i + 1)
                        }
                        SeekError::Ambiguous(c) => format!(
                            "{path}: hunk {} context matches {c} places - add more surrounding \
context lines so it points to exactly one",
                            i + 1
                        ),
                    })?;
                    // Surface a loose match (context wasn't exact) so it isn't applied silently.
                    if tier != MatchTier::Exact {
                        summary.push(format!(
                            "{path}: hunk {} located by {} match",
                            i + 1,
                            tier.label()
                        ));
                    }
                    lines.splice(start..end, h.after.iter().cloned());
                }
                let mut out = lines.join("\n");
                if had_trailing_nl {
                    out.push('\n');
                }
                let target = move_to.clone().unwrap_or_else(|| path.clone());
                std::fs::write(&target, &out).map_err(|e| format!("{target}: {e}"))?;
                match &move_to {
                    Some(mv) if mv != &path => {
                        std::fs::remove_file(&path).ok();
                        summary.push(format!("updated {path} -> {mv}"));
                    }
                    _ => summary.push(format!("updated {path}")),
                }
            }
        }
    }
    Ok(summary.join("; "))
}

/// Edit files with a V4A patch. `kind()=Write`, so `safety::Guarded`
/// fronts it with a permission prompt.
pub struct ApplyPatch;

#[async_trait]
impl Tool for ApplyPatch {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "apply_patch".into(),
            description: "Edit files with a V4A patch envelope:\n*** Begin Patch\n[ file sections ]\n\
*** End Patch\nHeaders: '*** Add File: <path>' (each following line prefixed '+'), \
'*** Delete File: <path>', '*** Update File: <path>' (optionally then '*** Move to: <path>'). In an \
Update, '@@' marks a location, ' ' is a context line, '-' removes, '+' adds. Prefer this over \
write_file for changing an existing file."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": { "patch": { "type": "string", "description": "the full patch envelope" } },
                "required": ["patch"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Write
    }
    fn summarize(&self, input: &Value) -> String {
        let patch = input.get("patch").and_then(|v| v.as_str()).unwrap_or("");
        let files = patch_files(patch);
        match files.len() {
            0 => "Edit files".into(),
            1 => format!("Edit {}", files[0]),
            n => format!("Edit {} (+{} more)", files[0], n - 1),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let patch = arg_str(&input, "patch", "apply_patch")?;
        let ops = match parse_patch(patch) {
            Ok(o) => o,
            Err(e) => return Ok(ToolOutput::error(format!("apply_patch: {e}"))),
        };
        if ops.is_empty() {
            return Ok(ToolOutput::error(
                "apply_patch: no file operations in patch",
            ));
        }
        match apply_ops(ops) {
            Ok(summary) => Ok(ToolOutput::ok(summary)),
            Err(e) => Ok(ToolOutput::error(format!("apply_patch: {e}"))),
        }
    }
}

/// Format combined stdout/stderr + exit code from a finished process. Shared by
/// foreground and background exec.
fn format_exec(code: Option<i32>, stdout: &[u8], stderr: &[u8]) -> String {
    let mut body = String::new();
    match code {
        Some(0) => {}
        Some(c) => body.push_str(&format!("[exit {c}]\n")),
        None => body.push_str("[terminated by signal]\n"),
    }
    body.push_str(&String::from_utf8_lossy(stdout));
    let se = String::from_utf8_lossy(stderr);
    if !se.is_empty() {
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str("[stderr] ");
        body.push_str(&se);
    }
    if body.trim().is_empty() {
        body = "(no output, exit 0)".into();
    }
    body
}

/// Spawn a foreground process, wait, and return its output (piped, capped,
/// killed-on-drop so the kernel's `guard()` cancels it). Shared by bash + worktree.
async fn exec_foreground(mut cmd: tokio::process::Command, who: &str, cue: EmptyCue) -> ToolOutput {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Foreground safety net: a command that never returns (a server, a hung process) would
    // otherwise block the turn forever. Cap the wait; on timeout the child is dropped and
    // kill_on_drop kills it, freeing the agent with a pointer to `background:true`.
    let secs = std::env::var("OXIO_BASH_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(120);
    match cmd.spawn() {
        Err(e) => ToolOutput::error(format!("{who}: failed to start: {e}")),
        Ok(child) => {
            match tokio::time::timeout(Duration::from_secs(secs), child.wait_with_output()).await {
                Err(_) => ToolOutput::error(format!(
                "{who}: no result after {secs}s, so it was stopped. For a long-running process \
                 (dev server, watch, tail -f) run it with background:true and read it with \
                 task_output - don't block the turn waiting on something that won't return."
            )),
                Ok(Err(e)) => ToolOutput::error(format!("{who}: {e}")),
                Ok(Ok(out)) => {
                    let mut body = format_exec(out.status.code(), &out.stdout, &out.stderr);
                    let empty_out = out.stdout.iter().all(u8::is_ascii_whitespace);
                    match cue {
                        // A searcher with empty stdout → the neutral widen-cue. (grep exits 1 on
                        // no-match, find exits 0 - so key on empty STDOUT, not the exit code.)
                        EmptyCue::Search if empty_out => {
                            body.push_str(&widen_cue("the searched path"))
                        }
                        // A resource-existence probe that came back NEGATIVE - non-zero exit, or
                        // truly-empty output - gets the neutral "look wider before concluding
                        // absent" cue. A probe that succeeds (exit 0 with output, incl. version
                        // text on stderr) is a real answer → no cue.
                        EmptyCue::Probe
                            if out.status.code() != Some(0)
                                || (empty_out
                                    && out.stderr.iter().all(u8::is_ascii_whitespace)) =>
                        {
                            body.push_str(&probe_cue());
                        }
                        // A cloud/remote log query with ZERO entries - empty stdout OR a
                        // "no results" JSON marker (aws/gcloud return `{"events": []}`, not
                        // empty output) - gets the neutral "0 logs != 0 events" cue.
                        EmptyCue::CloudLog
                            if empty_out || looks_like_no_log_results(&out.stdout) =>
                        {
                            body.push_str(&cloud_log_cue());
                        }
                        _ => {}
                    }
                    ToolOutput::ok(cap(body))
                }
            }
        }
    }
}

/// What cue (if any) `exec_foreground` appends when a command's result is unproductive.
/// The right cue depends on the command's PURPOSE - an empty search vs a resource probe
/// that found nothing mean different next moves - so the caller classifies and passes it.
#[derive(Clone, Copy, PartialEq)]
enum EmptyCue {
    /// Neither a search nor a probe - no cue (empty output is just empty output).
    Off,
    /// A searcher (grep/find/…): empty STDOUT → widen cue.
    Search,
    /// A resource-existence probe (which/command -v/pip show/import …): a NEGATIVE result
    /// → the neutral "it may live elsewhere; look wider" cue.
    Probe,
    /// A cloud/remote LOG query (aws logs / gcloud logging / kubectl logs …): ZERO log
    /// entries → the neutral "0 logs is not 0 events; widen the window/filter" cue. This
    /// targets the classic stateless failure - a narrow/relative time window ("last N
    /// min") that drifts across an approval delay, returns nothing, and gets reported as
    /// "no errors".
    CloudLog,
}

/// Registry of background `bash` tasks. Shared by the `bash` + `task_*` tools.
#[derive(Default)]
pub struct Tasks {
    next: AtomicU64,
    map: Mutex<HashMap<u64, BgTask>>,
}

struct BgTask {
    cmd: String,
    output: Arc<Mutex<String>>,
    done: Arc<AtomicBool>,
    abort: tokio::task::AbortHandle,
}

impl Tasks {
    pub fn new() -> Arc<Self> {
        Arc::new(Tasks::default())
    }

    /// Spawn a command detached, draining its output into the registry; returns id.
    fn spawn(&self, command: &str, cwd: Option<&str>) -> std::result::Result<u64, String> {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(d) = cwd {
            cmd.current_dir(d);
        }
        let child = cmd.spawn().map_err(|e| e.to_string())?;
        let output = Arc::new(Mutex::new(String::new()));
        let done = Arc::new(AtomicBool::new(false));
        let (o, d) = (output.clone(), done.clone());
        let handle = tokio::spawn(async move {
            if let Ok(out) = child.wait_with_output().await {
                *o.lock().unwrap() = format_exec(out.status.code(), &out.stdout, &out.stderr);
            }
            d.store(true, Ordering::SeqCst);
        });
        let id = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        self.map.lock().unwrap().insert(
            id,
            BgTask {
                cmd: command.to_string(),
                output,
                done,
                abort: handle.abort_handle(),
            },
        );
        Ok(id)
    }
}

/// Run a shell command via `sh -c`. `kind()=Write` (arbitrary exec), so
/// `safety::Guarded` gates it. Foreground: `kill_on_drop`, so the kernel's
/// `guard()` kills it on cancel/deadline (no orphan). `background:true` runs it
/// detached and returns a task id to poll with `task_output`. v1 GATE-ONLY - no
/// OS sandbox (deferred as its own module).
pub struct BashTool {
    tasks: Arc<Tasks>,
}

impl BashTool {
    pub fn new(tasks: Arc<Tasks>) -> Self {
        BashTool { tasks }
    }
}

#[async_trait]
impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command via `sh -c`; returns combined stdout/stderr + exit code. \
Use for builds, tests, git, and anything without a dedicated tool. Set `background:true` for a \
long-running command - it returns a task id you poll with `task_output`. Prefer the dedicated tools \
when they fit.\n\
To LOCATE a file/resource machine-wide: try an index first for a fast hit (`mdfind -name X` on macOS, \
`locate X` on Linux - see the OS in the environment block), then `find / -iname '*X*' 2>/dev/null` is \
the DEFINITIVE check - it's the only thing that proves absence. Start at `~` and widen to `/` if empty. \
Empty in the current directory is NEVER the machine-wide answer."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "the shell command to run" },
                    "background": { "type": "boolean", "description": "run detached; poll with task_output" }
                },
                "required": ["command"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Write
    }
    fn summarize(&self, input: &Value) -> String {
        match one_line_arg(input, "command", 80) {
            Some(c) => format!("Run {c}"),
            None => "Run command".into(),
        }
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> Result<ToolOutput> {
        let command = arg_str(&input, "command", "bash")?;
        if input
            .get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return match self.tasks.spawn(command, ctx.cwd.as_deref()) {
                Ok(id) => Ok(ToolOutput::ok(format!(
                    "started background task {id}; poll with task_output {id}"
                ))),
                Err(e) => Ok(ToolOutput::error(format!("bash: failed to start: {e}"))),
            };
        }
        // Classify the command so exec_foreground can append the right cue on an
        // unproductive result. Probe first: `command -v grep` is a probe, not a search,
        // even though it names a searcher.
        let cue = if is_resource_probe(command) {
            EmptyCue::Probe
        } else if is_cloud_log_query(command) {
            EmptyCue::CloudLog
        } else if is_search_command(command) {
            EmptyCue::Search
        } else {
            EmptyCue::Off
        };
        // Resolve the working directory (the sandbox's writable root, and the child's cwd).
        let cwd: PathBuf = ctx
            .cwd
            .as_ref()
            .map(PathBuf::from)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        // If a sandbox policy is active (OXIO_SANDBOX set - the prompt-off path), run
        // the shell under seatbelt: writes confined to cwd+temp, network off by default.
        // Unset (the interactive, approval-gated path) → run unwrapped, unchanged.
        let base_args = vec!["-c".to_string(), command.to_string()];
        let (program, args) = match sandbox::policy_from_env(&cwd) {
            Some(policy) => {
                sandbox::wrap(&policy, "sh", &base_args).unwrap_or(("sh".to_string(), base_args))
            }
            None => ("sh".to_string(), base_args),
        };
        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&args);
        cmd.current_dir(&cwd);
        Ok(exec_foreground(cmd, "bash", cue).await)
    }
}

/// Manage git worktrees (list/add/remove). Thin, structured wrapper over
/// `git worktree` (the model could also do it via `bash`); kind=Write → gated.
pub struct Worktree;

#[async_trait]
impl Tool for Worktree {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "worktree".into(),
            description:
                "Manage git worktrees. action=list | add | remove. `add` needs `path` (and \
optional `branch`); `remove` needs `path`. Useful to work on an isolated checkout."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "add", "remove"] },
                    "path": { "type": "string" },
                    "branch": { "type": "string" }
                },
                "required": ["action"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Write
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> Result<ToolOutput> {
        let action = arg_str(&input, "action", "worktree")?;
        let mut args: Vec<String> = vec!["worktree".into()];
        match action {
            "list" => args.push("list".into()),
            "add" => {
                args.push("add".into());
                args.push(arg_str(&input, "path", "worktree")?.into());
                if let Some(b) = input.get("branch").and_then(|v| v.as_str()) {
                    args.push(b.into());
                }
            }
            "remove" => {
                args.push("remove".into());
                args.push(arg_str(&input, "path", "worktree")?.into());
            }
            other => {
                return Ok(ToolOutput::error(format!(
                    "worktree: unknown action '{other}' (list|add|remove)"
                )));
            }
        }
        let mut cmd = tokio::process::Command::new("git");
        cmd.args(&args);
        if let Some(dir) = &ctx.cwd {
            cmd.current_dir(dir);
        }
        Ok(exec_foreground(cmd, "worktree", EmptyCue::Off).await)
    }
}

/// List background tasks and their status.
pub struct TaskList {
    tasks: Arc<Tasks>,
}
#[async_trait]
impl Tool for TaskList {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "task_list".into(),
            description: "List background bash tasks and whether each is running or done.".into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        }
    }
    async fn call(&self, _input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let map = self.tasks.map.lock().unwrap();
        if map.is_empty() {
            return Ok(ToolOutput::ok("(no background tasks)".to_string()));
        }
        let mut ids: Vec<&u64> = map.keys().collect();
        ids.sort();
        let mut out = String::new();
        for id in ids {
            let t = &map[id];
            let status = if t.done.load(Ordering::SeqCst) {
                "done"
            } else {
                "running"
            };
            out.push_str(&format!("{id} [{status}] {}\n", t.cmd));
        }
        Ok(ToolOutput::ok(out.trim_end().to_string()))
    }
}

/// Get a background task's output once it has finished.
pub struct TaskOutput {
    tasks: Arc<Tasks>,
}
#[async_trait]
impl Tool for TaskOutput {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "task_output".into(),
            description:
                "Get a background task's output by id. Reports 'still running' until it finishes."
                    .into(),
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer" } }, "required": ["id"] }),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let id = match input.get("id").and_then(|v| v.as_u64()) {
            Some(i) => i,
            None => return Ok(ToolOutput::error("task_output: missing integer 'id'")),
        };
        let map = self.tasks.map.lock().unwrap();
        match map.get(&id) {
            None => Ok(ToolOutput::error(format!("task_output: no task {id}"))),
            Some(t) if !t.done.load(Ordering::SeqCst) => {
                Ok(ToolOutput::ok(format!("task {id} still running")))
            }
            Some(t) => Ok(ToolOutput::ok(cap(t.output.lock().unwrap().clone()))),
        }
    }
}

/// Stop (kill) a running background task by id.
pub struct TaskStop {
    tasks: Arc<Tasks>,
}
#[async_trait]
impl Tool for TaskStop {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "task_stop".into(),
            description: "Stop (kill) a running background task by id.".into(),
            input_schema: json!({ "type": "object", "properties": { "id": { "type": "integer" } }, "required": ["id"] }),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let id = match input.get("id").and_then(|v| v.as_u64()) {
            Some(i) => i,
            None => return Ok(ToolOutput::error("task_stop: missing integer 'id'")),
        };
        let map = self.tasks.map.lock().unwrap();
        match map.get(&id) {
            None => Ok(ToolOutput::error(format!("task_stop: no task {id}"))),
            Some(t) => {
                t.abort.abort(); // drops the drainer future → child dropped → kill_on_drop kills it
                t.done.store(true, Ordering::SeqCst);
                Ok(ToolOutput::ok(format!("stopped task {id}")))
            }
        }
    }
}

/// Crude HTML-to-text: drop script/style blocks, strip tags, decode a few
/// entities, collapse whitespace. A real readability extractor is deferred.
fn html_to_text(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut skip = 0i32; // depth inside <script>/<style>
    let mut tag = String::new();
    for c in html.chars() {
        if in_tag {
            if c == '>' {
                in_tag = false;
                let t = tag.trim().to_ascii_lowercase();
                if t.starts_with("script") || t.starts_with("style") {
                    skip += 1;
                } else if t.starts_with("/script") || t.starts_with("/style") {
                    skip = (skip - 1).max(0);
                }
                if skip == 0 {
                    text.push(' '); // separate block content
                }
                tag.clear();
            } else {
                tag.push(c);
            }
        } else if c == '<' {
            in_tag = true;
        } else if skip == 0 {
            text.push(c);
        }
    }
    let text = text
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    let mut out = String::with_capacity(text.len());
    let mut last_ws = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !last_ws {
                out.push(' ');
                last_ws = true;
            }
        } else {
            out.push(c);
            last_ws = false;
        }
    }
    out.trim().to_string()
}

/// Fetch a URL over HTTP(S) and return its text (HTML stripped), capped.
/// `kind()=Read` (no local mutation) and ungated; network-egress approval is a
/// deferred safety axis, not built here.
pub struct WebFetch;

#[async_trait]
impl Tool for WebFetch {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_fetch".into(),
            description:
                "Fetch a URL over HTTP(S) and return its readable text content (HTML tags \
stripped), capped. Use to read a documentation page or article whose URL you already have."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": { "url": { "type": "string", "description": "the http(s) URL to fetch" } },
                "required": ["url"]
            }),
        }
    }
    fn summarize(&self, input: &Value) -> String {
        match input
            .get("url")
            .and_then(|v| v.as_str())
            .filter(|u| !u.is_empty())
        {
            Some(u) => format!("Fetch {u}"),
            None => "Fetch URL".into(),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let url = arg_str(&input, "url", "web_fetch")?;
        let client = match reqwest::Client::builder()
            .user_agent("oxio/0.1")
            .timeout(Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutput::error(format!("web_fetch: client: {e}"))),
        };
        let resp = match client.get(url).send().await {
            Ok(r) => r,
            Err(e) => return Ok(ToolOutput::error(format!("web_fetch: {e}"))),
        };
        let status = resp.status();
        let ctype = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let body = match resp.text().await {
            Ok(t) => t,
            Err(e) => return Ok(ToolOutput::error(format!("web_fetch: read body: {e}"))),
        };
        if !status.is_success() {
            return Ok(ToolOutput::error(format!(
                "web_fetch: HTTP {} from {url}",
                status.as_u16()
            )));
        }
        let text = if ctype.contains("html") || body.trim_start().starts_with('<') {
            html_to_text(&body)
        } else {
            body
        };
        Ok(ToolOutput::ok(cap(text)))
    }
}

#[derive(Clone)]
struct PlanStep {
    step: String,
    status: String,
}

/// Records a step-by-step task plan. The model passes the FULL plan each call; we store + echo a
/// checklist. Persistent plan state surfaced in the UI is deferred.
pub struct UpdatePlan {
    state: Mutex<Vec<PlanStep>>,
}

impl UpdatePlan {
    pub fn new() -> Self {
        UpdatePlan {
            state: Mutex::new(Vec::new()),
        }
    }
}

impl Default for UpdatePlan {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for UpdatePlan {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "update_plan".into(),
            description: "Record or update a short step-by-step plan. Pass the FULL plan each call as \
`plan`: a list of {step, status}, status one of pending|in_progress|completed. Keep exactly one step \
in_progress. Returns the current checklist."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": { "type": "string" },
                                "status": { "type": "string", "enum": ["pending", "in_progress", "completed"] }
                            },
                            "required": ["step", "status"]
                        }
                    }
                },
                "required": ["plan"]
            }),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let arr = input
            .get("plan")
            .and_then(|v| v.as_array())
            .ok_or_else(|| OxioError::Tool("update_plan: 'plan' must be an array".into()))?;
        let mut steps = Vec::new();
        for item in arr {
            let step = item.get("step").and_then(|v| v.as_str());
            let status = item.get("status").and_then(|v| v.as_str());
            match (step, status) {
                (Some(s), Some(st)) if ["pending", "in_progress", "completed"].contains(&st) => {
                    steps.push(PlanStep {
                        step: s.to_string(),
                        status: st.to_string(),
                    });
                }
                (Some(_), Some(st)) => {
                    return Ok(ToolOutput::error(format!(
                        "update_plan: bad status '{st}' (pending|in_progress|completed)"
                    )));
                }
                _ => {
                    return Ok(ToolOutput::error(
                        "update_plan: each item needs 'step' and 'status'",
                    ))
                }
            }
        }
        *self.state.lock().unwrap() = steps.clone();
        let mut out = String::from("plan:\n");
        for s in &steps {
            let mark = match s.status.as_str() {
                "completed" => "[x]",
                "in_progress" => "[~]",
                _ => "[ ]",
            };
            out.push_str(&format!("{mark} {}\n", s.step));
        }
        Ok(ToolOutput::ok(out.trim_end().to_string()))
    }
}

// --- web_search : a configured search BACKEND, never a bypass ---
// oxio is local-primary, so web_search is its own backend, selected by env like the
// provider API keys. If unconfigured, the tool DEMANDS setup (loud directive), never
// silently no-ops. Brave (API key) and SearXNG (self-host URL) ship first; more backends
// slot behind this seam.

enum SearchBackend {
    Brave { key: String },
    Searxng { url: String },
    Tavily { key: String },
    Google { key: String, cx: String },
}

struct Hit {
    title: String,
    url: String,
    snippet: String,
}

fn search_setup_directive() -> String {
    "web_search has no backend configured; it will not search until you set OXIO_SEARCH_BACKEND to \
one of these (oxio gives options - pick freely; none is a forced downgrade):\n\
- brave   (independent index, the backend Anthropic's web search uses): OXIO_SEARCH_API_KEY=<https://brave.com/search/api/>\n\
- google  (Google's index, what Gemini grounding uses): OXIO_SEARCH_API_KEY + OXIO_SEARCH_CX=<programmable search engine id>\n\
- tavily  (agent-optimized, returns extracted content): OXIO_SEARCH_API_KEY=<https://tavily.com>\n\
- searxng (self-hosted, keyless, private): OXIO_SEARCH_URL=<your instance url>"
        .to_string()
}

fn search_backend_from_env() -> std::result::Result<SearchBackend, String> {
    match std::env::var("OXIO_SEARCH_BACKEND").ok().as_deref() {
        Some("brave") => {
            let key = std::env::var("OXIO_SEARCH_API_KEY")
                .map_err(|_| "brave backend needs OXIO_SEARCH_API_KEY".to_string())?;
            Ok(SearchBackend::Brave { key })
        }
        Some("searxng") => {
            let url = std::env::var("OXIO_SEARCH_URL")
                .map_err(|_| "searxng backend needs OXIO_SEARCH_URL".to_string())?;
            Ok(SearchBackend::Searxng { url })
        }
        Some("tavily") => {
            let key = std::env::var("OXIO_SEARCH_API_KEY")
                .map_err(|_| "tavily backend needs OXIO_SEARCH_API_KEY".to_string())?;
            Ok(SearchBackend::Tavily { key })
        }
        Some("google") => {
            let key = std::env::var("OXIO_SEARCH_API_KEY")
                .map_err(|_| "google backend needs OXIO_SEARCH_API_KEY".to_string())?;
            let cx = std::env::var("OXIO_SEARCH_CX").map_err(|_| {
                "google backend needs OXIO_SEARCH_CX (search engine id)".to_string()
            })?;
            Ok(SearchBackend::Google { key, cx })
        }
        Some(other) => Err(format!(
            "unknown OXIO_SEARCH_BACKEND '{other}' (brave|google|tavily|searxng)"
        )),
        None => Err(search_setup_directive()),
    }
}

/// Map an array of result items into `Hit`s using the given field keys. Every search
/// backend differs only in where its array lives and its field names, so they share this.
fn hits_from_array(arr: &[Value], url_key: &str, title_key: &str, snippet_key: &str) -> Vec<Hit> {
    arr.iter()
        .filter_map(|it| {
            let url = it.get(url_key)?.as_str()?.to_string();
            Some(Hit {
                title: it
                    .get(title_key)
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
                url,
                snippet: it
                    .get(snippet_key)
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            })
        })
        .collect()
}

fn parse_brave(v: &Value) -> Vec<Hit> {
    v.get("web")
        .and_then(|w| w.get("results"))
        .and_then(|r| r.as_array())
        .map(|arr| hits_from_array(arr, "url", "title", "description"))
        .unwrap_or_default()
}

fn parse_searxng(v: &Value) -> Vec<Hit> {
    v.get("results")
        .and_then(|r| r.as_array())
        .map(|arr| hits_from_array(arr, "url", "title", "content"))
        .unwrap_or_default()
}

fn parse_tavily(v: &Value) -> Vec<Hit> {
    v.get("results")
        .and_then(|r| r.as_array())
        .map(|arr| hits_from_array(arr, "url", "title", "content"))
        .unwrap_or_default()
}

fn parse_google(v: &Value) -> Vec<Hit> {
    v.get("items")
        .and_then(|r| r.as_array())
        .map(|arr| hits_from_array(arr, "link", "title", "snippet"))
        .unwrap_or_default()
}

/// Send a request, check status, parse the body as JSON. Shared by all backends.
async fn get_json(req: reqwest::RequestBuilder, who: &str) -> std::result::Result<Value, String> {
    let resp = req.send().await.map_err(|e| e.to_string())?;
    let status = resp.status();
    let text = resp.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("{who} HTTP {}", status.as_u16()));
    }
    serde_json::from_str(&text).map_err(|e| format!("{who}: bad JSON: {e}"))
}

async fn run_search(
    client: &reqwest::Client,
    backend: &SearchBackend,
    query: &str,
) -> std::result::Result<Vec<Hit>, String> {
    match backend {
        SearchBackend::Brave { key } => {
            let req = client
                .get("https://api.search.brave.com/res/v1/web/search")
                .query(&[("q", query), ("count", "8")])
                .header("Accept", "application/json")
                .header("X-Subscription-Token", key);
            Ok(parse_brave(&get_json(req, "brave").await?))
        }
        SearchBackend::Searxng { url } => {
            let endpoint = format!("{}/search", url.trim_end_matches('/'));
            let req = client
                .get(&endpoint)
                .query(&[("q", query), ("format", "json")]);
            Ok(parse_searxng(&get_json(req, "searxng").await?))
        }
        SearchBackend::Tavily { key } => {
            // POST JSON; body built by hand so we need no reqwest "json" feature.
            let body = json!({ "api_key": key, "query": query, "max_results": 8 }).to_string();
            let req = client
                .post("https://api.tavily.com/search")
                .header("Content-Type", "application/json")
                .body(body);
            Ok(parse_tavily(&get_json(req, "tavily").await?))
        }
        SearchBackend::Google { key, cx } => {
            let req = client
                .get("https://www.googleapis.com/customsearch/v1")
                .query(&[
                    ("key", key.as_str()),
                    ("cx", cx.as_str()),
                    ("q", query),
                    ("num", "8"),
                ]);
            Ok(parse_google(&get_json(req, "google").await?))
        }
    }
}

fn format_hits(hits: &[Hit], max: usize) -> String {
    if hits.is_empty() {
        return "[no results]".to_string();
    }
    let mut out = String::new();
    for h in hits.iter().take(max) {
        out.push_str(&format!("{}\n{}\n{}\n\n", h.title, h.url, h.snippet));
    }
    out.trim_end().to_string()
}

/// Web search via a configured backend (Brave or SearXNG). kind=Read (ungated;
/// network-egress approval deferred, same as web_fetch). Unconfigured → a setup
/// directive, never a silent skip.
pub struct WebSearch;

#[async_trait]
impl Tool for WebSearch {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "web_search".into(),
            description:
                "Search the web and return the top results (title, URL, snippet). Requires a \
configured search backend; if none is set the tool returns setup instructions."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": { "query": { "type": "string", "description": "the search query" } },
                "required": ["query"]
            }),
        }
    }
    fn summarize(&self, input: &Value) -> String {
        match one_line_arg(input, "query", 80) {
            Some(q) => format!("Search the web: {q}"),
            None => "Web search".into(),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let query = arg_str(&input, "query", "web_search")?;
        let backend = match search_backend_from_env() {
            Ok(b) => b,
            Err(msg) => return Ok(ToolOutput::error(format!("web_search: {msg}"))),
        };
        let client = match reqwest::Client::builder()
            .user_agent("oxio/0.1")
            .timeout(Duration::from_secs(30))
            .build()
        {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutput::error(format!("web_search: client: {e}"))),
        };
        match run_search(&client, &backend, query).await {
            Ok(hits) => Ok(ToolOutput::ok(cap(format_hits(&hits, 8)))),
            Err(e) => Ok(ToolOutput::error(format!("web_search: {e}"))),
        }
    }
}

// --- memory : minimal file-backed fact store (first slice of the memory module) ---
// Base: Gemini `memoryTool` / Claude `SessionMemory`. The step-6 memory MODULE
// will own richer storage (extraction/consolidation) and may relocate the file;
// this is the durable-fact surface, not a deferral.

fn mem_save(path: &Path, text: &str) -> std::result::Result<(), String> {
    use std::io::Write as _;
    if let Some(p) = path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    writeln!(f, "{}", text.replace('\n', " ")).map_err(|e| e.to_string())?;
    Ok(())
}

fn mem_list(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| {
            s.lines()
                .map(|l| l.to_string())
                .filter(|l| !l.trim().is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Persist and recall durable facts across turns. `kind()=Read`: it writes only
/// to the dedicated oxio memory file, not user code, so it is not gated.
pub struct Memory;

#[async_trait]
impl Tool for Memory {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "memory".into(),
            description: "Persist or recall durable facts across turns. action=save stores `text` at \
`scope` = project (default - facts about THIS repo) or global (applies in EVERY project - the user's \
lasting preferences/decisions). action=list returns facts from both scopes. Use for stable preferences \
and decisions, not transient chatter."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["save", "list"] },
                    "text": { "type": "string", "description": "the fact to save (action=save)" },
                    "scope": { "type": "string", "enum": ["project", "global"], "description": "save target: project (default, this repo) or global (all projects). Use global for facts about the user; project for facts about this codebase." }
                },
                "required": ["action"]
            }),
        }
    }
    fn summarize(&self, input: &Value) -> String {
        let global = input.get("scope").and_then(|v| v.as_str()) == Some("global");
        match input.get("action").and_then(|v| v.as_str()).unwrap_or("") {
            "save" => {
                let path = if global {
                    memory::global_memory_path()
                } else {
                    memory::memory_path()
                };
                let fact: String = input
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .chars()
                    .take(50)
                    .collect();
                let tier = if global { "global" } else { "project" };
                format!("Remember ({tier} · {}): {fact}", path.display())
            }
            "list" => "Recall memory".into(),
            _ => "memory".into(),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let action = arg_str(&input, "action", "memory")?;
        let global = input.get("scope").and_then(|v| v.as_str()) == Some("global");
        match action {
            "save" => {
                let text = arg_str(&input, "text", "memory")?;
                let path = if global {
                    memory::global_memory_path()
                } else {
                    memory::memory_path()
                };
                match mem_save(&path, text) {
                    Ok(()) => Ok(ToolOutput::ok(format!("saved to {}", path.display()))),
                    Err(e) => Ok(ToolOutput::error(format!("memory: {e}"))),
                }
            }
            "list" => {
                let g = mem_list(&memory::global_memory_path());
                let p = mem_list(&memory::memory_path());
                if g.is_empty() && p.is_empty() {
                    return Ok(ToolOutput::ok("(no memories)"));
                }
                let mut out = String::new();
                for f in &g {
                    out.push_str(&format!("- (global) {f}\n"));
                }
                for f in &p {
                    out.push_str(&format!("- (project) {f}\n"));
                }
                Ok(ToolOutput::ok(out.trim_end().to_string()))
            }
            other => Ok(ToolOutput::error(format!(
                "memory: unknown action '{other}' (save|list)"
            ))),
        }
    }
}

/// Keyword search over the built-in tool catalog (dynamic discovery). Today all
/// tools are advertised each turn, so this is a convenience; it becomes
/// load-bearing only with deferred advertisement at scale (many MCP tools). The
/// catalog snapshot covers built-ins; MCP/agent tools (added at kernel assembly)
/// are not included.
pub struct ToolSearch {
    catalog: Vec<ToolSpec>,
}

#[async_trait]
impl Tool for ToolSearch {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "tool_search".into(),
            description: "Find tools by keyword; returns matching tool names and descriptions."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let query = arg_str(&input, "query", "tool_search")?.to_ascii_lowercase();
        let mut out = String::new();
        for spec in &self.catalog {
            if spec.name.to_ascii_lowercase().contains(&query)
                || spec.description.to_ascii_lowercase().contains(&query)
            {
                let one_line = spec.description.lines().next().unwrap_or("");
                out.push_str(&format!("{}: {}\n", spec.name, one_line));
            }
        }
        if out.is_empty() {
            return Ok(ToolOutput::ok(format!("[no tools match '{query}']")));
        }
        Ok(ToolOutput::ok(out.trim_end().to_string()))
    }
}

/// The default tool set. Read tools run ungated; write tools (kind=Write) are
/// fronted by `safety::Guarded` when the kernel is assembled.
/// Read-only git inspection: `status` / `diff` / `log` / `show`. A dedicated,
/// structured tool so a small model queries repo state reliably instead of
/// shelling `git ... | ...`. Writes (commit/add/branch) stay in the gated `bash`.
struct GitTool;

#[async_trait]
impl Tool for GitTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "git".into(),
            description:
                "Inspect the git repo (read-only). op=status|diff|log|show. diff takes optional \
`path` and `staged`; log takes optional `count` (default 20); show takes `rev` (default HEAD). For \
commit/add/branch or any write, use bash."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "op": { "type": "string", "enum": ["status", "diff", "log", "show"] },
                    "path": { "type": "string", "description": "file/dir to diff" },
                    "staged": { "type": "boolean", "description": "diff staged changes" },
                    "count": { "type": "integer", "description": "log entries (default 20)" },
                    "rev": { "type": "string", "description": "revision for show (default HEAD)" }
                },
                "required": ["op"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let op = arg_str(&input, "op", "git")?;
        let mut cmd = tokio::process::Command::new("git");
        match op {
            "status" => {
                cmd.args(["status", "--short", "--branch"]);
            }
            "diff" => {
                cmd.arg("diff");
                if input
                    .get("staged")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    cmd.arg("--staged");
                }
                if let Some(p) = input.get("path").and_then(|v| v.as_str()) {
                    cmd.arg(p);
                }
            }
            "log" => {
                let n = input.get("count").and_then(|v| v.as_u64()).unwrap_or(20);
                cmd.args(["log", "--oneline", "-n"]).arg(n.to_string());
            }
            "show" => {
                let rev = input.get("rev").and_then(|v| v.as_str()).unwrap_or("HEAD");
                cmd.arg("show").arg(rev);
            }
            other => {
                return Ok(ToolOutput::error(format!(
                    "git: unknown op '{other}' (status|diff|log|show)"
                )));
            }
        }
        Ok(exec_foreground(cmd, "git", EmptyCue::Off).await)
    }
}

// --- generate_image : text→image via an OpenAI-compatible endpoint (cloud OR local) ---
// Off the coding path but useful for assets. Endpoint-agnostic: it speaks the OpenAI
// `/images/generations` + `b64_json` contract, so it works with OpenAI cloud AND any local
// server that speaks the same contract (e.g. LocalAI over SDXL/Flux). Local-first, cloud
// optional - the API key is REQUIRED for OpenAI but skipped for a keyless local endpoint.
// (ComfyUI uses a different queue-a-workflow API and would need its own adapter.)

/// Resolve the image API key: oxio-namespaced first, then the standard OpenAI var,
/// then the vision provider's key (same OpenAI account, already configured for vision).
/// `None` is fine for a keyless local endpoint; only OpenAI requires it.
fn image_api_key() -> Option<String> {
    for var in [
        "OXIO_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OXIO_OPENAI_VISION_API_KEY",
    ] {
        if let Ok(k) = std::env::var(var) {
            if !k.trim().is_empty() {
                return Some(k);
            }
        }
    }
    None
}

/// A collision-free `<slug>.png` path in the current directory (where the terminal sits),
/// so a generated image lands next to the user and is visible in an IDE file panel.
fn unique_png_path(prompt: &str) -> PathBuf {
    let mut slug = String::new();
    let mut prev_dash = false;
    for c in prompt.chars().take(48) {
        if c.is_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }
    let base = slug.trim_matches('-');
    let base = if base.is_empty() { "image" } else { base };
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut path = cwd.join(format!("{base}.png"));
    let mut n = 1;
    while path.exists() {
        path = cwd.join(format!("{base}-{n}.png"));
        n += 1;
    }
    path
}

/// Generate an image from a text prompt and save a PNG where the terminal sits. kind=Write
/// (a paid/producing action that writes a file → approval-gated). Quality defaults to the
/// session setting (`/imageQ`, via `OXIO_IMAGE_QUALITY`); endpoint/model are overridable
/// by env for a local server.
pub struct GenerateImage;

#[async_trait]
impl Tool for GenerateImage {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "generate_image".into(),
            description:
                "Generate an image from a text prompt and save it as a PNG in the current \
directory; returns the path. Uses an OpenAI-compatible image endpoint (cloud or local)."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "what to draw" },
                    "size": { "type": "string", "description": "e.g. 1024x1024 (optional)" },
                    "quality": { "type": "string", "description": "low|medium|high (optional; default from /imageQ)" }
                },
                "required": ["prompt"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Write
    }
    fn summarize(&self, input: &Value) -> String {
        match one_line_arg(input, "prompt", 60) {
            Some(p) => format!("Generate image: {p}"),
            None => "Generate image".into(),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let prompt = arg_str(&input, "prompt", "generate_image")?;
        let base = std::env::var("OXIO_IMAGE_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".into());
        let model = std::env::var("OXIO_IMAGE_MODEL").unwrap_or_else(|_| "gpt-image-2".into());
        let size = input
            .get("size")
            .and_then(|v| v.as_str())
            .unwrap_or("1024x1024");
        let quality = input
            .get("quality")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                std::env::var("OXIO_IMAGE_QUALITY")
                    .ok()
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_else(|| "medium".into());

        // OpenAI needs a key; a local (custom base_url) endpoint may be keyless.
        let key = image_api_key();
        let is_openai = base.contains("api.openai.com");
        if is_openai && key.is_none() {
            return Ok(ToolOutput::error(
                "generate_image needs an API key for OpenAI - set OXIO_OPENAI_API_KEY (or \
OPENAI_API_KEY), or point OXIO_IMAGE_BASE_URL at a local image server."
                    .to_string(),
            ));
        }

        let body =
            json!({ "model": model, "prompt": prompt, "size": size, "quality": quality, "n": 1 })
                .to_string();
        let url = format!("{}/images/generations", base.trim_end_matches('/'));
        let mut req = reqwest::Client::new()
            .post(&url)
            .header("Content-Type", "application/json")
            .body(body);
        if let Some(k) = &key {
            req = req.header("Authorization", format!("Bearer {k}"));
        }
        let v = match get_json(req, "generate_image").await {
            Ok(v) => v,
            Err(e) => return Ok(ToolOutput::error(format!("generate_image: {e}"))),
        };
        let b64 = match v
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|d| d.get("b64_json"))
            .and_then(|b| b.as_str())
        {
            Some(b) => b,
            None => {
                let head: String = v.to_string().chars().take(200).collect();
                return Ok(ToolOutput::error(format!(
                    "generate_image: no image in response: {head}"
                )));
            }
        };
        let bytes = match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(b) => b,
            Err(e) => {
                return Ok(ToolOutput::error(format!(
                    "generate_image: bad base64: {e}"
                )))
            }
        };
        let revised = v
            .get("data")
            .and_then(|d| d.get(0))
            .and_then(|d| d.get("revised_prompt"))
            .and_then(|r| r.as_str())
            .filter(|s| !s.is_empty());
        let path = unique_png_path(prompt);
        if let Err(e) = std::fs::write(&path, &bytes) {
            return Ok(ToolOutput::error(format!(
                "generate_image: write {}: {e}",
                path.display()
            )));
        }
        // The result steers the MODEL's reply (the user never sees this text). Guide a
        // natural, trust-preserving response: own the generation, speak FROM the prompt,
        // and don't fabricate visual specifics (colours/style/details) that weren't asked -
        // no denial ("I can't see it"), no imagined pixels.
        let mut msg = format!(
            "Image generated from the prompt \"{prompt}\" and saved to {} ({size}, {model}).",
            path.display()
        );
        if let Some(r) = revised {
            msg.push_str(&format!(
                "\nThe image model expanded the prompt to: \"{r}\"."
            ));
        }
        msg.push_str(
            "\n\nReply briefly: confirm the image is created and give the file path, described in \
terms of THIS prompt. Do not assert specific colours, style, or details that were not in the \
prompt.",
        );
        Ok(ToolOutput::ok(msg))
    }
}

/// `render_page` - the local-first "artifact": render a self-contained HTML page by
/// writing it to a file and serving it on an ephemeral `127.0.0.1` port, then opening
/// the browser. No cloud, no account - the page lives on this machine. For visual
/// output a terminal can't show (dashboards, diagrams, reports). The server thread runs
/// for the life of the session.
struct RenderPage;

/// Write `html` to a temp artifacts dir and return the file path.
fn write_artifact(slug: &str, html: &str) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("oxio-artifacts");
    std::fs::create_dir_all(&dir).map_err(|e| OxioError::Tool(e.to_string()))?;
    let path = dir.join(format!("{slug}.html"));
    std::fs::write(&path, html).map_err(|e| OxioError::Tool(e.to_string()))?;
    Ok(path)
}

/// A running local artifact server: the port it listens on, the CURRENT page content
/// (swappable in place for UPDATE), and a stop flag for clean SHUTDOWN.
struct ArtifactServer {
    port: u16,
    content: Arc<Mutex<String>>,
    stop: Arc<AtomicBool>,
}

/// Registry of live artifact servers, keyed by slug. Re-rendering the same title
/// UPDATES in place (same port); a title (or all) can be shut down cleanly.
fn artifacts() -> &'static Mutex<HashMap<String, ArtifactServer>> {
    static REG: std::sync::OnceLock<Mutex<HashMap<String, ArtifactServer>>> =
        std::sync::OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Slug of the most recently rendered/updated artifact, so the UI can offer a
/// "re-open the last page" action - the local mirror of a reopen-last-artifact keybind
/// (the cloud tools reopen a hosted URL; oxio reopens the local one).
fn last_artifact() -> &'static Mutex<Option<String>> {
    static LAST: std::sync::OnceLock<Mutex<Option<String>>> = std::sync::OnceLock::new();
    LAST.get_or_init(|| Mutex::new(None))
}

/// `(title, url)` of the most recently rendered artifact that is still serving, if any.
/// The UI's `/open` opens this. `None` when nothing has been rendered or it was closed.
pub fn latest_artifact() -> Option<(String, String)> {
    // Copy the slug out and drop the LAST lock BEFORE locking the registry, so the two
    // mutexes are never held nested (no lock-order coupling with the render path).
    let slug = { last_artifact().lock().ok()?.clone()? };
    let reg = artifacts().lock().ok()?;
    let srv = reg.get(&slug)?;
    Some((slug, format!("http://127.0.0.1:{}/", srv.port)))
}

/// Every live artifact as `(title, url)`, for listing when several pages are open.
pub fn artifact_list() -> Vec<(String, String)> {
    let Ok(reg) = artifacts().lock() else {
        return Vec::new();
    };
    let mut v: Vec<(String, String)> = reg
        .iter()
        .map(|(slug, srv)| (slug.clone(), format!("http://127.0.0.1:{}/", srv.port)))
        .collect();
    v.sort();
    v
}

/// Open a URL in the user's default browser - the local mirror of opening a cloud artifact.
pub fn open_url(url: &str) {
    open_in_browser(url);
}

/// Bind a RANDOM port in the IANA dynamic/private range (49152–65535) - avoids common
/// dev ports (3000/5000/8000/8080/…) so it won't clash with the user's other servers.
/// Seeded from the clock, advanced with an LCG; retry on collision, fall back to an
/// OS-chosen ephemeral port only if every roll is taken.
fn bind_random_port() -> Result<(std::net::TcpListener, u16)> {
    use std::net::TcpListener;
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0x9E37) as u64;
    for _ in 0..16 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let port = 49152u32 + ((seed >> 17) as u32 % (65535 - 49152));
        if let Ok(l) = TcpListener::bind(("127.0.0.1", port as u16)) {
            return Ok((l, port as u16));
        }
    }
    let l = TcpListener::bind("127.0.0.1:0").map_err(|e| OxioError::Tool(e.to_string()))?;
    let p = l
        .local_addr()
        .map_err(|e| OxioError::Tool(e.to_string()))?
        .port();
    Ok((l, p))
}

/// Serve the shared `content` on a random local port until `stop` is set. The accept
/// loop is non-blocking and polls `stop`, so SHUTDOWN is prompt and frees the port (the
/// listener drops on exit). UPDATE works by swapping `content` - the next request serves
/// the new page, so a browser refresh shows changes on the SAME url.
fn start_server(content: Arc<Mutex<String>>, stop: Arc<AtomicBool>) -> Result<u16> {
    use std::io::{Read, Write};
    let (listener, port) = bind_random_port()?;
    listener.set_nonblocking(true).ok();
    std::thread::spawn(move || {
        loop {
            if stop.load(Ordering::Relaxed) {
                break; // listener drops here → port freed
            }
            match listener.accept() {
                Ok((mut s, _)) => {
                    s.set_nonblocking(false).ok();
                    let mut buf = [0u8; 1024];
                    let _ = s.read(&mut buf); // drain request; we serve the current page for any GET
                    let body = content.lock().map(|g| g.clone()).unwrap_or_default();
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(), body
                    );
                    let _ = s.write_all(resp.as_bytes());
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) => break,
            }
        }
    });
    Ok(port)
}

fn open_in_browser(url: &str) {
    let cmd = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "start"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(cmd).arg(url).spawn();
}

fn slugify(title: &str) -> String {
    let s: String = title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let s = s.trim_matches('-').to_string();
    if s.is_empty() {
        "artifact".to_string()
    } else {
        s
    }
}

#[async_trait]
impl Tool for RenderPage {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "render_page".into(),
            description: "Render a self-contained HTML page LOCALLY and open it in the browser - a \
                          local-first artifact. Write the full page as `html` (inline CSS/JS; assets as \
                          data: URIs or from a CDN - it is served from localhost, not hosted anywhere). \
                          Use for visual output a terminal cannot show (dashboards, diagrams, reports, \
                          small apps). Returns the local URL and file path."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "html": { "type": "string", "description": "the complete HTML document to render" },
                    "title": { "type": "string", "description": "short name for the page (used for the filename)" }
                },
                "required": ["html"]
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        // Opens a browser + starts a local server → effectful, so it is permission-gated.
        ToolKind::Write
    }
    fn summarize(&self, input: &Value) -> String {
        match one_line_arg(input, "title", 40) {
            Some(t) => format!("Render page: {t}"),
            None => "Render page".to_string(),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let html = arg_str(&input, "html", "render_page")?.to_string();
        let title = input
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("artifact");
        let slug = slugify(title);
        let path = write_artifact(&slug, &html)?;
        // Mark this as the most-recent page so the UI's `/open` can re-open it later.
        // Set before touching the registry lock so the two mutexes are never nested.
        if let Ok(mut last) = last_artifact().lock() {
            *last = Some(slug.clone());
        }
        let mut reg = artifacts()
            .lock()
            .map_err(|_| OxioError::Tool("artifact registry poisoned".into()))?;
        // UPDATE: same title → swap content on the live server, keep the port/URL.
        if let Some(srv) = reg.get(&slug) {
            if let Ok(mut c) = srv.content.lock() {
                *c = html;
            }
            let url = format!("http://127.0.0.1:{}/", srv.port);
            return Ok(ToolOutput::ok(format!(
                "Updated '{title}' at {url} (same local port {}; file: {}). Refresh the browser to see it.",
                srv.port,
                path.display()
            )));
        }
        // CREATE: new server on a random port.
        let content = Arc::new(Mutex::new(html));
        let stop = Arc::new(AtomicBool::new(false));
        let port = start_server(content.clone(), stop.clone())?;
        reg.insert(
            slug,
            ArtifactServer {
                port,
                content,
                stop,
            },
        );
        drop(reg);
        let url = format!("http://127.0.0.1:{port}/");
        open_in_browser(&url);
        // Exact port returned so the model always knows where the page is - no surprise.
        Ok(ToolOutput::ok(format!(
            "Rendered '{title}' locally at {url} (random local port {port}; file: {}). Served from this \
             machine - open the URL. Re-run render_page with the same title to UPDATE; use close_page to shut it down.",
            path.display()
        )))
    }
}

/// `close_page` - cleanly shut down a local artifact server (frees its port).
struct ClosePage;

#[async_trait]
impl Tool for ClosePage {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "close_page".into(),
            description:
                "Cleanly shut down a local page started by render_page, freeing its port. \
                          Pass `title` to close one page, or omit to close all of them."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string", "description": "the page to close (omit to close all)" }
                }
            }),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Write
    }
    fn summarize(&self, input: &Value) -> String {
        match one_line_arg(input, "title", 40) {
            Some(t) => format!("Close page: {t}"),
            None => "Close all pages".to_string(),
        }
    }
    async fn call(&self, input: Value, _ctx: &Ctx) -> Result<ToolOutput> {
        let mut reg = artifacts()
            .lock()
            .map_err(|_| OxioError::Tool("artifact registry poisoned".into()))?;
        match input.get("title").and_then(|v| v.as_str()) {
            Some(t) => {
                let slug = slugify(t);
                match reg.remove(&slug) {
                    Some(srv) => {
                        srv.stop.store(true, Ordering::Relaxed);
                        Ok(ToolOutput::ok(format!(
                            "Closed '{t}' - local port {} freed.",
                            srv.port
                        )))
                    }
                    None => Ok(ToolOutput::ok(format!("No live page titled '{t}'."))),
                }
            }
            None => {
                let n = reg.len();
                for (_, srv) in reg.drain() {
                    srv.stop.store(true, Ordering::Relaxed);
                }
                Ok(ToolOutput::ok(format!("Closed {n} local page(s).")))
            }
        }
    }
}

pub fn builtin() -> Vec<Arc<dyn Tool>> {
    let tasks = Tasks::new();
    let mut v: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFile),
        Arc::new(ListDir),
        Arc::new(GlobTool),
        Arc::new(ReadManyFiles),
        Arc::new(GrepTool),
        Arc::new(WriteFile),
        Arc::new(ApplyPatch),
        Arc::new(BashTool::new(tasks.clone())),
        Arc::new(WebFetch),
        Arc::new(UpdatePlan::new()),
        Arc::new(WebSearch),
        Arc::new(Memory),
        Arc::new(TaskList {
            tasks: tasks.clone(),
        }),
        Arc::new(TaskOutput {
            tasks: tasks.clone(),
        }),
        Arc::new(TaskStop { tasks }),
        Arc::new(Worktree),
        Arc::new(GitTool),
        Arc::new(GenerateImage),
        Arc::new(RenderPage),
        Arc::new(ClosePage),
    ];
    // tool_search sees a snapshot of the other built-ins (not itself).
    let catalog: Vec<ToolSpec> = v.iter().map(|t| t.spec()).collect();
    v.push(Arc::new(ToolSearch { catalog }));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_command_classification() {
        // Real searchers → cue-eligible (empty result may mean "look wider"). Neutral
        // inputs on purpose - the classifier keys on the COMMAND (find/grep/rg/locate),
        // never on any search term.
        assert!(is_search_command("find . -name '*.rs'"));
        assert!(is_search_command("cd /x && grep -r foo ."));
        assert!(is_search_command("rg TODO src | head"));
        assert!(is_search_command("locate config"));
        // Not searches → no cue (empty is the real answer / irrelevant).
        assert!(!is_search_command("ls -la")); // listing, empty dir is valid
        assert!(!is_search_command("cat file.txt"));
        assert!(!is_search_command("mkdir -p a/b"));
        // The cue is a hint, and explicitly stands down when 0 is a valid answer.
        assert!(widen_cue("x").contains("It's a hint, not a requirement"));
        assert!(widen_cue("x").contains("target may simply not exist"));
    }

    #[test]
    fn resource_probe_classification_is_keyword_agnostic() {
        // Probe SHAPE triggers, whatever the resource keyword is (manim, figma, anything).
        assert!(is_resource_probe("command -v manim"));
        assert!(is_resource_probe("which figma"));
        assert!(is_resource_probe("type node"));
        assert!(is_resource_probe("ffmpeg --version"));
        assert!(is_resource_probe("pip show numpy"));
        assert!(is_resource_probe("npm ls react"));
        assert!(is_resource_probe(r#"python3 -c "import manim""#));
        assert!(is_resource_probe(r#"python -c "import figma_sdk""#));
        // A probe that names a searcher is still a probe, not a search (probe wins).
        assert!(is_resource_probe("command -v grep"));
        // Not probes → no probe cue.
        assert!(!is_resource_probe("ls -la"));
        assert!(!is_resource_probe("cat file.txt"));
        assert!(!is_resource_probe("grep -r foo ."));
        // Neutral + instructional: names the next move, but stands down if truly absent.
        assert!(
            probe_cue().contains("may live elsewhere") || probe_cue().contains("live elsewhere")
        );
        assert!(probe_cue().contains("valid answer"));
        assert!(probe_cue().contains("find"));
    }

    #[test]
    fn cloud_log_query_classification_and_zero_result_detection() {
        // Cloud/remote log queries across providers - shape only, no service keywords.
        assert!(is_cloud_log_query(
            "aws logs filter-log-events --log-group-name /x --start-time 0"
        ));
        assert!(is_cloud_log_query(
            "aws logs start-query --query-string 'fields @message'"
        ));
        assert!(is_cloud_log_query("gcloud logging read 'severity>=ERROR'"));
        assert!(is_cloud_log_query("kubectl logs deploy/api --since=3m"));
        assert!(is_cloud_log_query(
            "az monitor log-analytics query -w ws --analytics-query 'X'"
        ));
        // Not cloud-log queries.
        assert!(!is_cloud_log_query("grep ERROR app.log"));
        assert!(!is_cloud_log_query("cat server.log"));
        // Zero-result detection: empty, bare [], and the JSON "no results" markers.
        assert!(looks_like_no_log_results(b""));
        assert!(looks_like_no_log_results(b"[]"));
        assert!(looks_like_no_log_results(br#"{"events": []}"#));
        assert!(looks_like_no_log_results(br#"{ "results" : [ ] }"#));
        // A real hit is not zero.
        assert!(!looks_like_no_log_results(
            br#"{"events": [{"message": "boom"}]}"#
        ));
        // Cue centers on assumption-as-fact, the drift trap, and a better query PATTERN
        // that doesn't need to know when the error happened (wide + newest-first + limit).
        let cue = cloud_log_cue();
        assert!(cue.contains("not proof that nothing happened"));
        assert!(cue.contains("DRIFTS"));
        assert!(cue.contains("NEWEST-FIRST") && cue.contains("sort @timestamp desc"));
        assert!(cue.contains("usually can't say when"));
    }

    #[tokio::test]
    async fn cloud_log_cue_fires_on_zero_results_json() {
        unsafe { std::env::remove_var("OXIO_SANDBOX") };
        let bash = BashTool::new(Tasks::new());
        let ctx = Ctx::default();
        // Simulate an aws-logs query returning zero events (JSON marker, non-empty stdout).
        let out = bash
            .call(json!({"command": r#"echo '{"events": []}' && aws logs filter-log-events --log-group x"#}), &ctx)
            .await
            .unwrap();
        // The command name makes it a cloud-log query; the empty-events JSON triggers the cue.
        assert!(
            out.content.contains("not proof that nothing happened"),
            "zero-log query gets the cue: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn probe_cue_fires_end_to_end_on_absent_resource_only() {
        // SAFETY: ensure the sandbox path is inactive so bash runs unwrapped in the test.
        unsafe { std::env::remove_var("OXIO_SANDBOX") };
        let bash = BashTool::new(Tasks::new());
        let ctx = Ctx::default();

        // Absent resource → negative probe → the neutral look-wider cue is appended.
        let absent = bash
            .call(
                json!({"command": "command -v oxio_definitely_absent_xyz"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(
            absent.content.contains("Not found HERE"),
            "absent probe should get the cue: {}",
            absent.content
        );

        // Present resource (sh exists) → real positive answer → NO cue.
        let present = bash
            .call(json!({"command": "command -v sh"}), &ctx)
            .await
            .unwrap();
        assert!(
            !present.content.contains("Not found HERE"),
            "present probe must not get the cue: {}",
            present.content
        );

        // A non-probe empty command → no probe cue either.
        let plain = bash.call(json!({"command": "true"}), &ctx).await.unwrap();
        assert!(
            !plain.content.contains("Not found HERE"),
            "non-probe must not get the cue: {}",
            plain.content
        );
    }

    fn tmp(name: &str, contents: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("oxio-tool-{}-{}", std::process::id(), name));
        std::fs::write(&p, contents).unwrap();
        p
    }

    #[tokio::test]
    async fn read_file_line_numbered_and_missing_errors() {
        let p = tmp("read.txt", "hello tools");
        let out = ReadFile
            .call(json!({ "path": p.to_string_lossy() }), &Ctx::default())
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(
            out.content.contains("1\thello tools"),
            "line-numbered; got {:?}",
            out.content
        );
        assert!(
            !out.content.contains("showing lines"),
            "single line must not note truncation"
        );

        let miss = ReadFile
            .call(json!({ "path": "/no/such/file/xyz" }), &Ctx::default())
            .await
            .unwrap();
        assert!(miss.is_error);
        let _ = std::fs::remove_file(&p);
    }

    #[tokio::test]
    async fn read_file_range_and_truncation_status() {
        let p = tmp("range.txt", "a\nb\nc\nd\ne");
        let out = ReadFile
            .call(
                json!({ "path": p.to_string_lossy(), "limit": 2 }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(out.content.contains("1\ta") && out.content.contains("2\tb"));
        assert!(!out.content.contains("3\tc"));
        assert!(
            out.content
                .contains("showing lines 1-2 of 5 - use offset=3 to continue"),
            "got {:?}",
            out.content
        );

        let out2 = ReadFile
            .call(
                json!({ "path": p.to_string_lossy(), "offset": 3, "limit": 2 }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(out2.content.contains("3\tc") && out2.content.contains("4\td"));
        assert!(out2
            .content
            .contains("showing lines 3-4 of 5 - use offset=5 to continue"));

        let out3 = ReadFile
            .call(
                json!({ "path": p.to_string_lossy(), "offset": 99 }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(!out3.is_error && out3.content.contains("past end of file"));
        let _ = std::fs::remove_file(&p);
    }

    #[tokio::test]
    async fn read_file_binary_and_non_utf8_guard() {
        let bin = tmp("bin.dat", "");
        std::fs::write(&bin, [0x68u8, 0x00, 0x69]).unwrap(); // contains NUL
        let out = ReadFile
            .call(json!({ "path": bin.to_string_lossy() }), &Ctx::default())
            .await
            .unwrap();
        assert!(
            out.is_error && out.content.contains("binary"),
            "got {:?}",
            out.content
        );
        let _ = std::fs::remove_file(&bin);

        let bad = tmp("bad.dat", "");
        std::fs::write(&bad, [0xffu8, 0xfe, 0x41]).unwrap(); // invalid UTF-8, no NUL
        let out2 = ReadFile
            .call(json!({ "path": bad.to_string_lossy() }), &Ctx::default())
            .await
            .unwrap();
        assert!(
            out2.is_error && out2.content.contains("UTF-8"),
            "got {:?}",
            out2.content
        );
        let _ = std::fs::remove_file(&bad);
    }

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("oxio-lsdir-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn list_dir_lists_and_marks_types() {
        let d = tmp_dir("basic");
        std::fs::write(d.join("a.txt"), "x").unwrap();
        std::fs::write(d.join("b.txt"), "x").unwrap();
        std::fs::create_dir(d.join("sub")).unwrap();
        let out = ListDir
            .call(json!({ "path": d.to_string_lossy() }), &Ctx::default())
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("a.txt") && out.content.contains("b.txt"));
        assert!(
            out.content.contains("sub/"),
            "dirs marked with /; got {:?}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn list_dir_respects_gitignore_and_all_toggle() {
        let d = tmp_dir("gi");
        std::fs::write(d.join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(d.join("kept.txt"), "x").unwrap();
        std::fs::write(d.join("ignored.txt"), "x").unwrap();

        let out = ListDir
            .call(json!({ "path": d.to_string_lossy() }), &Ctx::default())
            .await
            .unwrap();
        assert!(out.content.contains("kept.txt"));
        assert!(
            !out.content.contains("ignored.txt"),
            "gitignore should hide it; got {:?}",
            out.content
        );
        assert!(
            !out.content.contains(".gitignore"),
            "dotfile hidden by default"
        );

        let out2 = ListDir
            .call(
                json!({ "path": d.to_string_lossy(), "all": true }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(
            out2.content.contains("ignored.txt") && out2.content.contains(".gitignore"),
            "all=true shows everything; got {:?}",
            out2.content
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn list_dir_missing_errors() {
        let out = ListDir
            .call(json!({ "path": "/no/such/dir/xyz" }), &Ctx::default())
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn glob_matches_and_respects_gitignore() {
        let d = tmp_dir("glob");
        std::fs::write(d.join(".gitignore"), "ign.rs\n").unwrap();
        std::fs::write(d.join("keep.rs"), "x").unwrap();
        std::fs::write(d.join("ign.rs"), "x").unwrap();
        std::fs::write(d.join("note.txt"), "x").unwrap();
        let out = GlobTool
            .call(
                json!({ "pattern": "**/*.rs", "path": d.to_string_lossy() }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(out.content.contains("keep.rs"));
        assert!(
            !out.content.contains("ign.rs"),
            "gitignore excludes; got {:?}",
            out.content
        );
        assert!(!out.content.contains("note.txt"), "pattern excludes .txt");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn glob_recent_first() {
        let d = tmp_dir("globrec");
        std::fs::write(d.join("older.rs"), "x").unwrap();
        std::thread::sleep(Duration::from_millis(30));
        std::fs::write(d.join("newer.rs"), "x").unwrap();
        let out = GlobTool
            .call(
                json!({ "pattern": "**/*.rs", "path": d.to_string_lossy() }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        let ni = out.content.find("newer.rs").expect("newer present");
        let oi = out.content.find("older.rs").expect("older present");
        assert!(ni < oi, "recent file first; got {:?}", out.content);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn glob_bad_pattern_errors() {
        let out = GlobTool
            .call(json!({ "pattern": "[", "path": "." }), &Ctx::default())
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn read_many_explicit_files_and_dir() {
        let d = tmp_dir("rm");
        std::fs::write(d.join("a.txt"), "AAA").unwrap();
        std::fs::write(d.join("b.txt"), "BBB").unwrap();
        let out = ReadManyFiles
            .call(json!({ "paths": [d.join("a.txt").to_string_lossy(), d.join("b.txt").to_string_lossy()] }), &Ctx::default())
            .await
            .unwrap();
        assert!(out.content.contains("a.txt") && out.content.contains("AAA"));
        assert!(out.content.contains("b.txt") && out.content.contains("BBB"));

        let out2 = ReadManyFiles
            .call(json!({ "paths": [d.to_string_lossy()] }), &Ctx::default())
            .await
            .unwrap();
        assert!(out2.content.contains("AAA") && out2.content.contains("BBB"));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn read_many_glob_gitignore_exclude_binary() {
        let d = tmp_dir("rm2");
        std::fs::write(d.join(".gitignore"), "ign.rs\n").unwrap();
        std::fs::write(d.join("x.rs"), "XRS").unwrap();
        std::fs::write(d.join("ign.rs"), "IGN").unwrap();
        std::fs::write(d.join("y.txt"), "YTXT").unwrap();
        std::fs::write(d.join("bin.rs"), [0u8, 1, 2]).unwrap();

        let pat = format!("{}/**/*.rs", d.to_string_lossy());
        let out = ReadManyFiles
            .call(json!({ "paths": [pat] }), &Ctx::default())
            .await
            .unwrap();
        assert!(out.content.contains("XRS"), "got {:?}", out.content);
        assert!(!out.content.contains("IGN"), "gitignore excludes ign.rs");
        assert!(!out.content.contains("YTXT"), "pattern excludes .txt");
        assert!(
            out.content.contains("binary"),
            "binary skipped-note; got {:?}",
            out.content
        );

        let out2 = ReadManyFiles
            .call(
                json!({ "paths": [d.to_string_lossy()], "exclude": ["**/*.txt"] }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(
            !out2.content.contains("YTXT"),
            "excluded txt; got {:?}",
            out2.content
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn read_many_missing_paths_errors() {
        let e = ReadManyFiles.call(json!({}), &Ctx::default()).await;
        assert!(e.is_err());
    }

    #[tokio::test]
    async fn grep_finds_lines_and_respects_gitignore() {
        let d = tmp_dir("grep");
        std::fs::write(d.join(".gitignore"), "ignored.rs\n").unwrap();
        std::fs::write(d.join("hit.rs"), "fn a() {}\nlet NEEDLE = 1;\n").unwrap();
        std::fs::write(d.join("miss.rs"), "nothing here\n").unwrap();
        std::fs::write(d.join("ignored.rs"), "let NEEDLE = 2;\n").unwrap();

        let out = GrepTool
            .call(
                json!({ "pattern": "NEEDLE", "path": d.to_string_lossy() }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(
            out.content.contains("hit.rs:2:let NEEDLE = 1;"),
            "got {:?}",
            out.content
        );
        assert!(!out.content.contains("miss.rs"), "non-matching file absent");
        assert!(
            !out.content.contains("ignored.rs"),
            "gitignored file skipped; got {:?}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn grep_glob_filter_and_case_insensitive() {
        let d = tmp_dir("grep2");
        std::fs::write(d.join("a.rs"), "let needle = 1;\n").unwrap();
        std::fs::write(d.join("b.txt"), "let needle = 2;\n").unwrap();

        // glob restricts to .rs; case-insensitive matches lowercase against uppercase pattern
        let out = GrepTool
            .call(
                json!({ "pattern": "NEEDLE", "path": d.to_string_lossy(), "glob": "**/*.rs", "case_sensitive": false }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(
            out.content.contains("a.rs:1:"),
            "matched .rs; got {:?}",
            out.content
        );
        assert!(
            !out.content.contains("b.txt"),
            "glob excluded .txt; got {:?}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn write_file_creates_parents_and_overwrites() {
        let d = tmp_dir("wf");
        let nested = d.join("a").join("b").join("f.txt");
        let out = WriteFile
            .call(
                json!({ "path": nested.to_string_lossy(), "content": "hello" }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(!out.is_error, "got {:?}", out.content);
        assert!(out.content.contains("wrote 5 bytes"));
        assert_eq!(std::fs::read_to_string(&nested).unwrap(), "hello");

        // overwrite
        WriteFile
            .call(
                json!({ "path": nested.to_string_lossy(), "content": "bye" }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&nested).unwrap(), "bye");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn write_file_declares_write_kind_and_needs_args() {
        assert_eq!(WriteFile.kind(), ToolKind::Write, "gate depends on this");
        let e = WriteFile
            .call(json!({ "path": "x" }), &Ctx::default())
            .await;
        assert!(e.is_err(), "missing content is a tool error");
    }

    #[tokio::test]
    async fn apply_patch_add_update_move_delete() {
        let d = tmp_dir("ap");
        assert_eq!(ApplyPatch.kind(), ToolKind::Write, "gate depends on this");
        let f = d.join("app.py");
        let f_str = f.to_string_lossy().to_string();

        // Add File
        let add = format!(
            "*** Begin Patch\n*** Add File: {f_str}\n+print(\"Hi\")\n+x = 1\n*** End Patch"
        );
        let out = ApplyPatch
            .call(json!({ "patch": add }), &Ctx::default())
            .await
            .unwrap();
        assert!(!out.is_error, "got {:?}", out.content);
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "print(\"Hi\")\nx = 1\n"
        );

        // Update File: change a line, with a trailing context line to anchor it
        let upd = format!("*** Begin Patch\n*** Update File: {f_str}\n@@\n-print(\"Hi\")\n+print(\"Hello\")\n x = 1\n*** End Patch");
        let out = ApplyPatch
            .call(json!({ "patch": upd }), &Ctx::default())
            .await
            .unwrap();
        assert!(!out.is_error, "got {:?}", out.content);
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "print(\"Hello\")\nx = 1\n"
        );

        // Move to: rename the file
        let g = d.join("main.py");
        let g_str = g.to_string_lossy().to_string();
        let mv = format!("*** Begin Patch\n*** Update File: {f_str}\n*** Move to: {g_str}\n@@\n-x = 1\n+x = 2\n*** End Patch");
        let out = ApplyPatch
            .call(json!({ "patch": mv }), &Ctx::default())
            .await
            .unwrap();
        assert!(!out.is_error, "got {:?}", out.content);
        assert!(!f.exists(), "original removed after move");
        assert_eq!(
            std::fs::read_to_string(&g).unwrap(),
            "print(\"Hello\")\nx = 2\n"
        );

        // Delete File
        let del = format!("*** Begin Patch\n*** Delete File: {g_str}\n*** End Patch");
        let out = ApplyPatch
            .call(json!({ "patch": del }), &Ctx::default())
            .await
            .unwrap();
        assert!(!out.is_error, "got {:?}", out.content);
        assert!(!g.exists(), "deleted");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn memory_save_list_roundtrip() {
        let d = tmp_dir("mem");
        let f = d.join("memory.jsonl");
        mem_save(&f, "user prefers rust").unwrap();
        mem_save(&f, "deploy via cdk only").unwrap();
        let facts = mem_list(&f);
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0], "user prefers rust");
        assert_eq!(facts[1], "deploy via cdk only");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn memory_bad_action_and_missing_action() {
        let bogus = Memory
            .call(json!({ "action": "nuke" }), &Ctx::default())
            .await
            .unwrap();
        assert!(bogus.is_error && bogus.content.contains("unknown action"));
        assert!(
            Memory.call(json!({}), &Ctx::default()).await.is_err(),
            "missing action is a tool error"
        );
    }

    #[test]
    fn web_search_parses_brave_and_searxng() {
        let brave = json!({ "web": { "results": [
            { "title": "Rust", "url": "https://rust-lang.org", "description": "systems lang" }
        ] } });
        let h = parse_brave(&brave);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].url, "https://rust-lang.org");
        assert_eq!(h[0].snippet, "systems lang");

        let sx = json!({ "results": [
            { "title": "Rust", "url": "https://rust-lang.org", "content": "systems lang" }
        ] });
        let h2 = parse_searxng(&sx);
        assert_eq!(h2.len(), 1);
        assert_eq!(h2[0].url, "https://rust-lang.org");

        let tv = json!({ "results": [
            { "title": "Rust", "url": "https://rust-lang.org", "content": "systems lang" }
        ] });
        assert_eq!(parse_tavily(&tv).len(), 1);

        let gg = json!({ "items": [
            { "title": "Rust", "link": "https://rust-lang.org", "snippet": "systems lang" }
        ] });
        let hg = parse_google(&gg);
        assert_eq!(hg.len(), 1);
        assert_eq!(
            hg[0].url, "https://rust-lang.org",
            "google uses 'link' not 'url'"
        );

        assert!(format_hits(&h, 8).contains("rust-lang.org"));
        assert_eq!(format_hits(&[], 8), "[no results]");
    }

    #[test]
    fn web_search_directive_names_all_backends() {
        let d = search_setup_directive();
        for b in ["brave", "google", "tavily", "searxng"] {
            assert!(d.contains(b), "directive must guide {b} setup: {d}");
        }
    }

    #[tokio::test]
    async fn web_search_needs_query() {
        assert!(WebSearch.call(json!({}), &Ctx::default()).await.is_err());
    }

    #[tokio::test]
    async fn update_plan_formats_checklist_and_validates() {
        let p = UpdatePlan::new();
        let out = p
            .call(
                json!({ "plan": [
                    { "step": "read code", "status": "completed" },
                    { "step": "fix bug", "status": "in_progress" },
                    { "step": "add test", "status": "pending" }
                ] }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(!out.is_error, "got {:?}", out.content);
        assert!(out.content.contains("[x] read code"));
        assert!(out.content.contains("[~] fix bug"));
        assert!(out.content.contains("[ ] add test"));

        let bad = p
            .call(
                json!({ "plan": [{ "step": "x", "status": "done" }] }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(bad.is_error && bad.content.contains("bad status"));

        assert!(
            p.call(json!({}), &Ctx::default()).await.is_err(),
            "missing plan is a tool error"
        );
    }

    #[test]
    fn web_fetch_html_to_text_strips_and_decodes() {
        let t = html_to_text("<h1>Hi</h1><script>var x=1</script><p>world &amp; peace</p>");
        assert!(t.contains("Hi"), "got {t:?}");
        assert!(t.contains("world & peace"), "entities decoded; got {t:?}");
        assert!(!t.contains("var x"), "script content dropped; got {t:?}");
    }

    #[tokio::test]
    async fn web_fetch_needs_url() {
        assert!(WebFetch.call(json!({}), &Ctx::default()).await.is_err());
    }

    #[tokio::test]
    async fn bash_runs_and_captures_output() {
        let out = BashTool::new(Tasks::new())
            .call(json!({ "command": "echo hello" }), &Ctx::default())
            .await
            .unwrap();
        assert!(!out.is_error);
        assert!(out.content.contains("hello"), "got {:?}", out.content);
    }

    #[tokio::test]
    async fn bash_reports_nonzero_exit_and_stderr() {
        let out = BashTool::new(Tasks::new())
            .call(
                json!({ "command": "echo oops 1>&2; exit 3" }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(out.content.contains("[exit 3]"), "got {:?}", out.content);
        assert!(
            out.content.contains("oops"),
            "stderr captured; got {:?}",
            out.content
        );
    }

    #[tokio::test]
    async fn bash_is_write_kind_and_needs_command() {
        assert_eq!(
            BashTool::new(Tasks::new()).kind(),
            ToolKind::Write,
            "must be gated"
        );
        assert!(BashTool::new(Tasks::new())
            .call(json!({}), &Ctx::default())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn tool_search_finds_by_keyword() {
        let ts = builtin()
            .into_iter()
            .find(|t| t.spec().name == "tool_search")
            .expect("tool_search registered");
        let hit = ts
            .call(json!({ "query": "regex" }), &Ctx::default())
            .await
            .unwrap();
        assert!(
            hit.content.contains("grep"),
            "grep matches 'regex'; got {:?}",
            hit.content
        );
        let none = ts
            .call(json!({ "query": "zzxqwv" }), &Ctx::default())
            .await
            .unwrap();
        assert!(none.content.contains("no tools match"));
        assert!(
            ts.call(json!({}), &Ctx::default()).await.is_err(),
            "missing query is a tool error"
        );
    }

    #[tokio::test]
    async fn worktree_is_write_kind_and_validates() {
        assert_eq!(Worktree.kind(), ToolKind::Write, "gate depends on this");
        let bad = Worktree
            .call(json!({ "action": "teleport" }), &Ctx::default())
            .await
            .unwrap();
        assert!(bad.is_error && bad.content.contains("unknown action"));
        assert!(
            Worktree
                .call(json!({ "action": "add" }), &Ctx::default())
                .await
                .is_err(),
            "add without path is a tool error"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bash_background_task_runs_polls_and_lists() {
        let tasks = Tasks::new();
        let started = BashTool::new(tasks.clone())
            .call(
                json!({ "command": "echo bg-hi", "background": true }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(
            started.content.contains("started background task"),
            "got {:?}",
            started.content
        );

        let to = TaskOutput {
            tasks: tasks.clone(),
        };
        let mut result = String::new();
        for _ in 0..100 {
            let o = to.call(json!({ "id": 1 }), &Ctx::default()).await.unwrap();
            if !o.content.contains("still running") {
                result = o.content;
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            result.contains("bg-hi"),
            "background output captured; got {result:?}"
        );

        let listed = TaskList { tasks }
            .call(json!({}), &Ctx::default())
            .await
            .unwrap();
        assert!(
            listed.content.contains("[done]"),
            "task_list shows done; got {:?}",
            listed.content
        );
    }

    #[tokio::test]
    async fn bash_honors_cwd() {
        let d = tmp_dir("bash");
        std::fs::write(d.join("marker.txt"), "x").unwrap();
        let ctx = Ctx {
            cwd: Some(d.to_string_lossy().to_string()),
            ..Ctx::default()
        };
        let out = BashTool::new(Tasks::new())
            .call(json!({ "command": "ls" }), &ctx)
            .await
            .unwrap();
        assert!(
            out.content.contains("marker.txt"),
            "cwd honored; got {:?}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn apply_patch_bad_envelope_and_missing_context_error() {
        let d = tmp_dir("ap2");
        // no Begin Patch
        let bad = ApplyPatch
            .call(json!({ "patch": "*** Update File: x\n" }), &Ctx::default())
            .await
            .unwrap();
        assert!(bad.is_error && bad.content.contains("Begin Patch"));

        // context that does not exist in the file
        let f = d.join("f.txt");
        std::fs::write(&f, "alpha\nbeta\n").unwrap();
        let miss = format!(
            "*** Begin Patch\n*** Update File: {}\n@@\n-NOTHERE\n+x\n*** End Patch",
            f.to_string_lossy()
        );
        let out = ApplyPatch
            .call(json!({ "patch": miss }), &Ctx::default())
            .await
            .unwrap();
        assert!(
            out.is_error && out.content.contains("could not locate"),
            "got {:?}",
            out.content
        );
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "alpha\nbeta\n",
            "file untouched on failed patch"
        );
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn grep_no_match_and_bad_regex() {
        let d = tmp_dir("grep3");
        std::fs::write(d.join("x.rs"), "hello\n").unwrap();
        let none = GrepTool
            .call(
                json!({ "pattern": "zzzz", "path": d.to_string_lossy() }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        // Zero matches: still starts with the marker, and carries the neutral widen-cue
        // telling the model it can search any path (targets the "nothing here → I can't" misfire).
        assert!(
            none.content.starts_with("[no matches]"),
            "got: {}",
            none.content
        );
        assert!(
            none.content.contains("search ANY path"),
            "widen-cue missing: {}",
            none.content
        );

        let bad = GrepTool
            .call(
                json!({ "pattern": "(", "path": d.to_string_lossy() }),
                &Ctx::default(),
            )
            .await
            .unwrap();
        assert!(bad.is_error, "unbalanced regex is a tool error");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn missing_arg_is_tool_error() {
        let e = ReadFile.call(json!({}), &Ctx::default()).await;
        assert!(e.is_err());
    }

    #[tokio::test]
    async fn git_status_executes() {
        // The crate lives inside the oxio git repo, so `git status` runs; the
        // tool returns a structured ok result regardless of repo state.
        let out = GitTool
            .call(json!({ "op": "status" }), &Ctx::default())
            .await
            .unwrap();
        assert!(
            !out.is_error,
            "git status returns a structured result: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn git_unknown_op_errors() {
        let out = GitTool
            .call(json!({ "op": "nope" }), &Ctx::default())
            .await
            .unwrap();
        assert!(out.is_error && out.content.contains("unknown op"));
    }
}

#[cfg(test)]
mod render_page_tests {
    use super::{
        artifact_list, artifacts, last_artifact, latest_artifact, slugify, start_server,
        ArtifactServer,
    };
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn fetch(port: u16) -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut resp = String::new();
        s.read_to_string(&mut resp).unwrap();
        resp
    }

    #[test]
    fn slugify_makes_a_safe_filename() {
        assert_eq!(slugify("Dev Connect Rig!"), "dev-connect-rig");
        assert_eq!(slugify("   "), "artifact");
    }

    #[test]
    fn serves_updates_and_shuts_down() {
        let content = Arc::new(Mutex::new("<h1>hi from oxio</h1>".to_string()));
        let stop = Arc::new(AtomicBool::new(false));
        let port = start_server(content.clone(), stop.clone()).unwrap();
        assert!(
            port >= 49152,
            "port {port} should be in the dynamic range, avoiding common ports"
        );

        // initial serve
        let r1 = fetch(port);
        assert!(
            r1.starts_with("HTTP/1.1 200")
                && r1.contains("hi from oxio")
                && r1.contains("text/html")
        );

        // UPDATE in place → same port serves the new page
        *content.lock().unwrap() = "<h1>updated page</h1>".into();
        assert!(
            fetch(port).contains("updated page"),
            "update reflected on same port"
        );

        // SHUTDOWN → port is freed
        stop.store(true, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(150));
        assert!(
            TcpStream::connect(("127.0.0.1", port)).is_err(),
            "port freed after clean shutdown"
        );
    }

    #[test]
    fn latest_artifact_tracks_last_render_and_lists_live_pages() {
        // Register a page the way render_page does, then verify the /open accessors see it.
        let stop = Arc::new(AtomicBool::new(false));
        let port = start_server(Arc::new(Mutex::new("<h1>x</h1>".into())), stop.clone()).unwrap();
        artifacts().lock().unwrap().insert(
            "report".into(),
            ArtifactServer {
                port,
                content: Arc::new(Mutex::new(String::new())),
                stop: stop.clone(),
            },
        );
        *last_artifact().lock().unwrap() = Some("report".into());

        let (title, url) = latest_artifact().expect("a page was rendered");
        assert_eq!(title, "report");
        assert_eq!(url, format!("http://127.0.0.1:{port}/"));
        assert!(
            artifact_list().iter().any(|(t, _)| t == "report"),
            "listed among live pages"
        );

        // Closing it (removing from the registry) means latest no longer resolves.
        stop.store(true, Ordering::Relaxed);
        artifacts().lock().unwrap().remove("report");
        assert!(
            latest_artifact().is_none(),
            "closed page is no longer the latest"
        );
    }
}
