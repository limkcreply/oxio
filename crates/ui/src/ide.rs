//! Editor selection bridge. An editor extension (VS Code, etc.) runs a loopback MCP server and
//! pushes `ide/contextUpdate` (the active selection) as a notification; the mcp client stores the
//! latest params, and this reads them so a submitted prompt can carry "the code I'm looking at".
//! No editor connected (or nothing selected) yields `None` and the prompt is sent unchanged.

/// The active editor selection, extracted from the IDE context.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
}

impl Selection {
    /// Number of selected lines (inclusive of both endpoints).
    fn line_count(&self) -> u32 {
        self.end_line.saturating_sub(self.start_line) + 1
    }

    /// The transcript chip, e.g. `Selected 12 lines from src/main.rs`.
    pub fn chip(&self) -> String {
        let n = self.line_count();
        let unit = if n == 1 { "line" } else { "lines" };
        format!("Selected {n} {unit} from {}", self.file)
    }

    /// The context block prepended to the user's message so the model sees the code.
    pub fn context_block(&self) -> String {
        format!(
            "[selected in editor: {} lines {}-{}]\n{}\n\n",
            self.file, self.start_line, self.end_line, self.text
        )
    }
}

/// The IDE MCP endpoint. Primary discovery is `OXIO_IDE_PORT` (the extension injects it into new
/// integrated terminals); the fallback is the newest port file the extension writes, so a terminal
/// opened BEFORE the extension activated still connects. `None` = no editor.
pub fn ide_mcp_url() -> Option<String> {
    Some(format!("http://127.0.0.1:{}/mcp", ide_port()?))
}

/// The editor server's port: `OXIO_IDE_PORT` if the extension set it in this terminal, else the
/// newest port file it wrote (covers a terminal opened before the extension activated).
fn ide_port() -> Option<u16> {
    std::env::var("OXIO_IDE_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .or_else(port_from_file)
}

/// The newest `oxio-ide-*.json` port file in the temp dir (the extension writes one on activation),
/// for terminals that predate `OXIO_IDE_PORT` being set.
fn port_from_file() -> Option<u16> {
    let mut newest: Option<(std::time::SystemTime, u16)> = None;
    for entry in std::fs::read_dir(std::env::temp_dir()).ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("oxio-ide-") || !name.ends_with(".json") {
            continue;
        }
        let Some(port) = std::fs::read_to_string(entry.path())
            .ok()
            .and_then(|b| serde_json::from_str::<serde_json::Value>(&b).ok())
            .and_then(|v| v.get("port").and_then(|p| p.as_u64()))
            .map(|p| p as u16)
        else {
            continue;
        };
        let mtime = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(std::time::UNIX_EPOCH);
        if newest.map(|(t, _)| mtime > t).unwrap_or(true) {
            newest = Some((mtime, port));
        }
    }
    newest.map(|(_, p)| p)
}

/// The extension id `code --install-extension` and `--list-extensions` use.
const EXTENSION_ID: &str = "oxio.oxio";

/// On a run inside VS Code, offer to install the companion extension (consent, never silent).
/// No-op outside VS Code, without the `code` CLI, once the extension is present, or when no
/// bundled `.vsix` ships next to the binary. Returns true if an install was started.
pub fn offer_extension_install() -> bool {
    if std::env::var("TERM_PROGRAM").ok().as_deref() != Some("vscode") {
        return false;
    }
    if !code_cli_present() || extension_installed() {
        return false;
    }
    let Some(vsix) = bundled_vsix() else {
        return false;
    };
    use std::io::Write as _;
    print!("Install the oxio VS Code companion (editor selection + diff approval)? [y/N]: ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() || !line.trim().eq_ignore_ascii_case("y") {
        return false;
    }
    let ok = std::process::Command::new("code")
        .arg("--install-extension")
        .arg(&vsix)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        println!("installed - reload the VS Code window (Developer: Reload Window) to enable it.");
    } else {
        println!("could not install the extension automatically - see editors/vscode/README.md.");
    }
    ok
}

fn code_cli_present() -> bool {
    std::process::Command::new("code")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn extension_installed() -> bool {
    std::process::Command::new("code")
        .arg("--list-extensions")
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .any(|l| l.trim() == EXTENSION_ID)
        })
        .unwrap_or(false)
}

/// The companion extension, embedded in the binary at build time so a public install carries it
/// with no network fetch. Empty in a dev build where the `.vsix` was not produced (the release
/// pipeline builds it before compiling); an empty payload skips the offer.
const VSIX: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../editors/vscode/oxio.vsix"
));

/// Write the embedded `.vsix` to a temp file for `code --install-extension`, or `None` when no
/// extension is bundled.
fn bundled_vsix() -> Option<std::path::PathBuf> {
    if VSIX.is_empty() {
        return None;
    }
    let path = std::env::temp_dir().join("oxio-companion.vsix");
    std::fs::write(&path, VSIX).ok()?;
    Some(path)
}

/// The editor selection RIGHT NOW: at submit, pull it from the extension's `getSelection` MCP tool
/// (same transport as the diff - pulled at submit, over the plugin's MCP). No push, so no
/// race with a change event or a late connection. `None` = no editor, nothing selected, or the
/// call failed.
pub async fn fetch_selection() -> Option<Selection> {
    #[cfg(feature = "mcp")]
    {
        parse_selection(&mcp::ide_get_selection().await?)
    }
    #[cfg(not(feature = "mcp"))]
    {
        None
    }
}

/// Parse the `getSelection` result's `selection` value: `{file, startLine, endLine, text}` or
/// `null`. An absent/empty `text` means nothing is selected.
#[cfg_attr(not(feature = "mcp"), allow(dead_code))]
fn parse_selection(sel: &serde_json::Value) -> Option<Selection> {
    let text = sel.get("text")?.as_str()?.to_string();
    if text.is_empty() {
        return None;
    }
    Some(Selection {
        file: sel
            .get("file")
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string(),
        start_line: sel.get("startLine").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        end_line: sel.get("endLine").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_a_selection() {
        let sel = json!({"file":"src/main.rs","startLine":10,"endLine":21,"text":"fn main() {}"});
        let s = parse_selection(&sel).expect("valid selection parses");
        assert_eq!(
            s,
            Selection {
                file: "src/main.rs".into(),
                start_line: 10,
                end_line: 21,
                text: "fn main() {}".into(),
            }
        );
    }

    #[test]
    fn empty_or_missing_yields_none() {
        assert!(
            parse_selection(&json!({"text":""})).is_none(),
            "empty text = nothing selected"
        );
        assert!(
            parse_selection(&serde_json::Value::Null).is_none(),
            "null = no selection"
        );
        assert!(parse_selection(&json!({})).is_none(), "no text field");
    }

    #[test]
    fn chip_counts_lines_and_pluralizes() {
        let one = Selection {
            file: "a.rs".into(),
            start_line: 5,
            end_line: 5,
            text: "x".into(),
        };
        assert_eq!(one.chip(), "Selected 1 line from a.rs");
        let many = Selection {
            file: "a.rs".into(),
            start_line: 10,
            end_line: 21,
            text: "x".into(),
        };
        assert_eq!(many.chip(), "Selected 12 lines from a.rs");
    }

    #[test]
    fn context_block_carries_file_range_and_code() {
        let s = Selection {
            file: "a.rs".into(),
            start_line: 1,
            end_line: 2,
            text: "let x = 1;".into(),
        };
        let b = s.context_block();
        assert!(b.contains("a.rs lines 1-2"));
        assert!(b.contains("let x = 1;"));
    }
}
