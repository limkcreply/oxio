//! Live probe: connect to the reference filesystem MCP server over stdio and
//! list the tools it advertises. Run: `cargo run -p mcp --example probe`.
//! (npx fetches the server on first run.)

use mcp::{load_server, McpServerCfg};

#[tokio::main]
async fn main() {
    let cfg = McpServerCfg {
        command: "npx".into(),
        args: vec![
            "-y".into(),
            "@modelcontextprotocol/server-filesystem".into(),
            "/tmp".into(),
        ],
        ..Default::default()
    };
    match load_server("fs", &cfg, None, None).await {
        Ok(loaded) => {
            println!("connected - {} tools advertised:", loaded.tools.len());
            for t in &loaded.tools {
                println!("  {}", t.spec().name);
            }
            match loaded.instructions {
                Some(ins) => println!("\nserver instructions:\n{ins}"),
                None => println!("\n(server provided no instructions)"),
            }
        }
        Err(e) => {
            eprintln!("failed: {e}");
            std::process::exit(1);
        }
    }
}
