//! Live probe: run diagnostics on a C file via clangd. `cargo run -p lsp --example probe`.

use std::path::Path;

use lsp::{diagnostics, LspServerCfg};

#[tokio::main]
async fn main() {
    let path = Path::new("/tmp/oxio_lsp_probe.c");
    // `x` is undeclared -> clangd should report an error.
    std::fs::write(path, "int main(void){ return x; }\n").unwrap();
    let cfg = LspServerCfg {
        command: "clangd".into(),
        args: vec![],
    };
    match diagnostics(&cfg, Path::new("/tmp"), path).await {
        Ok(s) => println!("diagnostics:\n{s}"),
        Err(e) => {
            eprintln!("failed: {e}");
            std::process::exit(1);
        }
    }
}
