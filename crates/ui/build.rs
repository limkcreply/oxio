use std::path::Path;

fn main() {
    // `include_bytes!` needs the bundled extension to exist at compile time. In a fresh clone or a
    // plain dev build it is absent, so write an empty placeholder to keep the build working (an
    // empty payload skips the install offer). The release pipeline drops the real oxio.vsix here
    // before compiling, so the shipped binary carries it.
    let vsix = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../editors/vscode/oxio.vsix");
    if !vsix.exists() {
        let _ = std::fs::write(&vsix, b"");
    }
    println!("cargo:rerun-if-changed=../../editors/vscode/oxio.vsix");
}
