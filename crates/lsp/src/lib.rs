//! Minimal LSP client: spawn a configured language server over stdio, open a
//! file, and collect its diagnostics. The transport is hand-rolled Content-Length
//! JSON-RPC (LSP framing differs from MCP's newline-delimited, and there is no
//! clean client SDK). v1 covers DIAGNOSTICS (push-based, highest value); more
//! queries (definition/hover/references) are staged on the same client.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use oxio_core::{Ctx, Result, Tool, ToolKind, ToolOutput, ToolSpec};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStdin, ChildStdout, Command};

/// A configured language server (stdio) for a file extension.
#[derive(Clone, Debug)]
pub struct LspServerCfg {
    pub command: String,
    pub args: Vec<String>,
}

fn language_id(file: &Path) -> &'static str {
    match file.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "c" => "c",
        "cc" | "cpp" | "cxx" | "hpp" => "cpp",
        "h" => "c",
        "rs" => "rust",
        "py" => "python",
        "ts" => "typescript",
        "js" => "javascript",
        "go" => "go",
        _ => "plaintext",
    }
}

async fn write_msg(stdin: &mut ChildStdin, msg: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(msg)?;
    stdin
        .write_all(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes())
        .await?;
    stdin.write_all(&body).await?;
    stdin.flush().await
}

async fn read_msg(stdout: &mut ChildStdout) -> io::Result<Value> {
    let mut headers = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        stdout.read_exact(&mut byte).await?;
        headers.push(byte[0]);
        if headers.ends_with(b"\r\n\r\n") {
            break;
        }
        if headers.len() > 8192 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LSP header too long",
            ));
        }
    }
    let header_str = String::from_utf8_lossy(&headers);
    let len: usize = header_str
        .lines()
        .find_map(|l| l.strip_prefix("Content-Length:"))
        .and_then(|n| n.trim().parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no Content-Length"))?;
    let mut buf = vec![0u8; len];
    stdout.read_exact(&mut buf).await?;
    serde_json::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn format_diagnostics(params: &Value) -> String {
    let diags = params
        .get("diagnostics")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if diags.is_empty() {
        return "[no diagnostics]".to_string();
    }
    let mut out = String::new();
    for d in &diags {
        let line = d
            .pointer("/range/start/line")
            .and_then(|v| v.as_u64())
            .map(|l| l + 1)
            .unwrap_or(0);
        let sev = match d.get("severity").and_then(|v| v.as_u64()) {
            Some(1) => "error",
            Some(2) => "warning",
            Some(3) => "info",
            Some(4) => "hint",
            _ => "diag",
        };
        let msg = d.get("message").and_then(|v| v.as_str()).unwrap_or("");
        out.push_str(&format!("{sev} L{line}: {msg}\n"));
    }
    out.trim_end().to_string()
}

/// Spawn the server, open `file`, and return its diagnostics (or a bounded-wait note).
pub async fn diagnostics(
    cfg: &LspServerCfg,
    root: &Path,
    file: &Path,
) -> std::result::Result<String, String> {
    let text =
        std::fs::read_to_string(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    let mut child = Command::new(&cfg.command)
        .args(&cfg.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", cfg.command))?;
    let mut stdin = child.stdin.take().ok_or("no stdin")?;
    let mut stdout = child.stdout.take().ok_or("no stdout")?;

    let root_uri = format!("file://{}", root.display());
    let file_uri = format!("file://{}", file.display());

    write_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"processId":null,"rootUri":root_uri,"capabilities":{}}}),
    )
    .await
    .map_err(|e| e.to_string())?;
    // Drain until the initialize response (id: 1).
    loop {
        let m = read_msg(&mut stdout).await.map_err(|e| e.to_string())?;
        if m.get("id").and_then(|v| v.as_i64()) == Some(1) {
            break;
        }
    }
    write_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","method":"initialized","params":{}}),
    )
    .await
    .map_err(|e| e.to_string())?;
    write_msg(
        &mut stdin,
        &json!({"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":file_uri,"languageId":language_id(file),"version":1,"text":text}}}),
    )
    .await
    .map_err(|e| e.to_string())?;

    let want = file_uri.clone();
    let collect = async {
        loop {
            let m = read_msg(&mut stdout).await.map_err(|e| e.to_string())?;
            if m.get("method").and_then(|v| v.as_str()) == Some("textDocument/publishDiagnostics")
                && m.pointer("/params/uri").and_then(|v| v.as_str()) == Some(want.as_str())
            {
                return Ok::<Value, String>(m.get("params").cloned().unwrap_or(Value::Null));
            }
        }
    };
    match tokio::time::timeout(Duration::from_secs(10), collect).await {
        Ok(r) => Ok(format_diagnostics(&r?)),
        Err(_) => Ok("[no diagnostics within 10s]".to_string()),
    }
}

/// The `lsp` tool: diagnostics for a file via a configured per-extension server.
pub struct LspTool {
    /// extension (without dot) -> server config
    pub servers: HashMap<String, LspServerCfg>,
}

#[async_trait]
impl Tool for LspTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "lsp".into(),
            description:
                "Language-server diagnostics for a file: action=diagnostics, `path` = the file. \
Returns errors/warnings from the configured language server. Requires a server configured for the \
file's extension."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["diagnostics"] },
                    "path": { "type": "string" }
                },
                "required": ["action", "path"]
            }),
        }
    }
    async fn call(&self, input: Value, ctx: &Ctx) -> Result<ToolOutput> {
        let action = input
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("diagnostics");
        if action != "diagnostics" {
            return Ok(ToolOutput::error(format!(
                "lsp: unknown action '{action}' (diagnostics)"
            )));
        }
        let path = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return Ok(ToolOutput::error("lsp: missing 'path'")),
        };
        let file = Path::new(path);
        let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
        let cfg = match self.servers.get(ext) {
            Some(c) => c,
            None => {
                return Ok(ToolOutput::error(format!(
                    "lsp: no server configured for '.{ext}' - set [lsp.{ext}] command=... in config"
                )));
            }
        };
        let root = ctx
            .cwd
            .as_deref()
            .map(Path::new)
            .or_else(|| file.parent())
            .unwrap_or_else(|| Path::new("."));
        match diagnostics(cfg, root, file).await {
            Ok(s) => Ok(ToolOutput::ok(s)),
            Err(e) => Ok(ToolOutput::error(format!("lsp: {e}"))),
        }
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
}
