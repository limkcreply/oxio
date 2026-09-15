//! Scratch probe: connect to a remote (streamable-HTTP) MCP server by URL and print
//! its advertised tools (name + full description + input schema) and its server-level
//! `instructions`. Run: `cargo run -p mcp --example probe_url -- http://host:port/mcp`.

use mcp::{load_server, McpServerCfg};

#[tokio::main]
async fn main() {
    let url = std::env::args().nth(1).expect("usage: probe_url <url>");
    let cfg = McpServerCfg {
        url: Some(url.clone()),
        ..Default::default()
    };
    match load_server("probe", &cfg, None, None).await {
        Ok(loaded) => {
            match &loaded.instructions {
                Some(ins) => println!("=== server instructions ===\n{ins}\n"),
                None => println!("=== server instructions ===\n(none)\n"),
            }
            println!("=== {} tools ===", loaded.tools.len());
            for t in &loaded.tools {
                let s = t.spec();
                println!(
                    "\n## {}\n{}\ninput_schema: {}",
                    s.name, s.description, s.input_schema
                );
            }
        }
        Err(e) => {
            eprintln!("failed: {e}");
            std::process::exit(1);
        }
    }
}
