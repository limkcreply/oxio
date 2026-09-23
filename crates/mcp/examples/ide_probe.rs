//! Scratch probe: exercise the editor bridge exactly as oxio does at runtime - connect to the
//! extension's loopback MCP server, then pull the active selection via the `getSelection` tool.
//! Run: `cargo run -p mcp --example ide_probe -- http://127.0.0.1:<port>/mcp`.

use mcp::{connect_ide, ide_get_selection};
use std::time::Duration;
use tokio::time::timeout;

#[tokio::main]
async fn main() {
    let url = std::env::args().nth(1).expect("usage: ide_probe <url>");
    eprintln!("connect_ide({url}) ...");
    match timeout(Duration::from_secs(6), connect_ide(&url)).await {
        Err(_) => {
            eprintln!("connect_ide TIMEOUT (handshake never completed in 6s)");
            std::process::exit(2);
        }
        Ok(Err(e)) => {
            eprintln!("connect_ide ERR: {e}");
            std::process::exit(1);
        }
        Ok(Ok(v)) => eprintln!("connect_ide OK (version={v:?})"),
    }
    match timeout(Duration::from_secs(6), ide_get_selection()).await {
        Err(_) => eprintln!("ide_get_selection TIMEOUT"),
        Ok(sel) => eprintln!("ide_get_selection = {sel:?}"),
    }
}
