//! macOS Seatbelt sandbox for shell execution - first slice of an OS sandbox.
//!
//! The permission gate (`safety::Guarded`) keeps interactive shell safe by asking a
//! human before each command. The sandbox is the complementary mechanism: it is what
//! lets a command run safely WITHOUT that prompt - the precondition for prompt-off
//! subagents and (later) a full-auto mode. The `.sbpl` policies are macOS-specific
//! security work; this module composes them and invokes `/usr/bin/sandbox-exec`.
//!
//! Scope of this slice: macOS only; reads allowed everywhere, writes confined to the
//! workspace + temp, network off by default. Linux (landlock+seccomp) and Windows
//! (WFP) are separate slices; see PROGRESS.md tech-debt.

use std::path::{Path, PathBuf};

const BASE_POLICY: &str = include_str!("seatbelt_base_policy.sbpl");
const NETWORK_POLICY: &str = include_str!("seatbelt_network_policy.sbpl");

/// Only ever the system `sandbox-exec`. Using an absolute path defends against a
/// PATH-injected impostor: if `/usr/bin/sandbox-exec` itself is tampered with, the
/// attacker already has root.
const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// What a sandboxed command may touch. Reads are allowed everywhere (a coding shell
/// needs the toolchain, libraries, and headers); writes are confined to
/// `writable_roots`; network is denied unless `allow_network`.
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    pub writable_roots: Vec<PathBuf>,
    pub allow_network: bool,
}

impl SandboxPolicy {
    /// Workspace-write: writes confined to `cwd` plus the OS temp dirs, network off.
    /// This is the mode a prompt-off subagent should run shell under.
    pub fn workspace_write(cwd: &Path) -> Self {
        let mut writable_roots = vec![cwd.to_path_buf()];
        for p in ["/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp"] {
            writable_roots.push(PathBuf::from(p));
        }
        // The per-user temp dir (TMPDIR, e.g. /var/folders/…) - many tools write here.
        if let Ok(tmp) = std::env::var("TMPDIR") {
            if !tmp.trim().is_empty() {
                writable_roots.push(PathBuf::from(tmp));
            }
        }
        Self {
            writable_roots,
            allow_network: false,
        }
    }
}

/// Whether the macOS Seatbelt sandbox is usable on this machine.
pub fn available() -> bool {
    cfg!(target_os = "macos") && Path::new(SANDBOX_EXEC).exists()
}

/// Read the sandbox policy from the environment, or `None` when disabled. This is the
/// opt-in seam a prompt-off subagent (or a `--sandbox` run) sets; the interactive
/// approval path leaves it unset and shell runs unwrapped as before.
///
/// `OXIO_SANDBOX=workspace-write` (or `1`/`on`) enables; `OXIO_SANDBOX_NET=1`
/// additionally allows network. Anything else (incl. unset/`off`/`0`) → `None`.
pub fn policy_from_env(cwd: &Path) -> Option<SandboxPolicy> {
    match std::env::var("OXIO_SANDBOX").ok()?.trim() {
        "workspace-write" | "1" | "on" => {
            let mut policy = SandboxPolicy::workspace_write(cwd);
            policy.allow_network = matches!(
                std::env::var("OXIO_SANDBOX_NET").as_deref(),
                Ok("1") | Ok("true")
            );
            Some(policy)
        }
        _ => None,
    }
}

/// Compose the full `sandbox-exec` policy string plus the `-D` path parameters.
/// Writable roots are passed as parameters (not inlined) so paths never need quoting
/// or escaping inside the policy body.
fn build_policy(policy: &SandboxPolicy) -> (String, Vec<(String, PathBuf)>) {
    let mut sections = vec![BASE_POLICY.to_string()];
    // Reads: allow everywhere - a coding shell must reach the toolchain and libs.
    sections.push("; allow read-only file operations\n(allow file-read*)".to_string());

    // Writes: confined to the writable roots, each a `-D`-supplied subpath.
    // Canonicalize each root so it matches the path the KERNEL sees: on macOS the temp
    // and cwd paths often traverse a symlink (`/var -> /private/var`, `/tmp ->
    // /private/tmp`), and a literal, un-resolved param would silently fail to match the
    // real write path. Fall back to the given path when it does not yet exist.
    let mut params = Vec::new();
    if !policy.writable_roots.is_empty() {
        let filters: Vec<String> = policy
            .writable_roots
            .iter()
            .enumerate()
            .map(|(i, root)| {
                let key = format!("WRITABLE_ROOT_{i}");
                let resolved = std::fs::canonicalize(root).unwrap_or_else(|_| root.clone());
                params.push((key.clone(), resolved));
                format!("(subpath (param \"{key}\"))")
            })
            .collect();
        sections.push(format!("(allow file-write*\n{}\n)", filters.join(" ")));
    }

    // Network: the base policy's `(deny default)` covers the off case; only widen
    // when explicitly allowed.
    if policy.allow_network {
        sections.push("(allow network-outbound)\n(allow network-inbound)".to_string());
        sections.push(NETWORK_POLICY.to_string());
    }

    (sections.join("\n"), params)
}

/// Wrap `(program, args)` to run under Seatbelt with `policy`, returning the new
/// `(program, args)` to spawn:
/// `sandbox-exec -p <policy> -D<KEY>=<path> … -- <program> <args…>`.
/// Returns `None` when the sandbox isn't available, so the caller runs unwrapped.
pub fn wrap(
    policy: &SandboxPolicy,
    program: &str,
    args: &[String],
) -> Option<(String, Vec<String>)> {
    if !available() {
        return None;
    }
    let (policy_str, params) = build_policy(policy);
    let mut out = vec!["-p".to_string(), policy_str];
    for (key, path) in params {
        out.push(format!("-D{key}={}", path.to_string_lossy()));
    }
    out.push("--".to_string());
    out.push(program.to_string());
    out.extend(args.iter().cloned());
    Some((SANDBOX_EXEC.to_string(), out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_write_confines_writes_to_cwd_and_temp_no_network() {
        let p = SandboxPolicy::workspace_write(Path::new("/work/proj"));
        assert!(p.writable_roots.contains(&PathBuf::from("/work/proj")));
        assert!(p.writable_roots.contains(&PathBuf::from("/private/tmp")));
        assert!(!p.allow_network);
    }

    #[test]
    fn build_policy_denies_default_allows_reads_and_confines_writes() {
        let policy = SandboxPolicy {
            writable_roots: vec![PathBuf::from("/work")],
            allow_network: false,
        };
        let (text, params) = build_policy(&policy);
        assert!(text.contains("(deny default)"), "closed by default");
        assert!(text.contains("(allow file-read*)"), "reads allowed");
        assert!(
            text.contains("(allow file-write*"),
            "writes section present"
        );
        assert!(
            !text.contains("(allow network-outbound)"),
            "no network when disabled"
        );
        assert_eq!(
            params,
            vec![("WRITABLE_ROOT_0".to_string(), PathBuf::from("/work"))]
        );
    }

    #[test]
    fn build_policy_adds_network_when_allowed() {
        let policy = SandboxPolicy {
            writable_roots: vec![],
            allow_network: true,
        };
        let (text, _) = build_policy(&policy);
        assert!(text.contains("(allow network-outbound)"));
        assert!(
            text.contains("com.apple.SecurityServer"),
            "network helper policy appended"
        );
    }

    #[test]
    fn policy_from_env_off_by_default() {
        // SAFETY: single-threaded test; unique var handling.
        unsafe { std::env::remove_var("OXIO_SANDBOX") };
        assert!(policy_from_env(Path::new("/work")).is_none());
    }

    /// The real proof: run a command through the wrapper's `sandbox-exec` invocation
    /// and confirm the boundary actually holds - a write inside the workspace succeeds,
    /// a write outside it is denied. macOS-only; a no-op where the sandbox is absent.
    #[test]
    fn boundary_allows_writes_inside_workspace_and_denies_outside() {
        if !available() {
            return; // not macOS / no sandbox-exec - nothing to enforce here
        }
        use std::process::Command;
        let ws = std::env::temp_dir().join(format!("oxio-sbx-{}", std::process::id()));
        std::fs::create_dir_all(&ws).unwrap();
        let outside =
            std::env::temp_dir().join(format!("oxio-sbx-outside-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&outside);

        // writable_roots = the workspace ONLY (no temp), so a sibling temp path is outside.
        let policy = SandboxPolicy {
            writable_roots: vec![ws.clone()],
            allow_network: false,
        };

        let run = |script: String| -> bool {
            let (prog, args) = wrap(&policy, "sh", &["-c".to_string(), script]).expect("available");
            Command::new(prog)
                .args(args)
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };

        let inside = ws.join("ok.txt");
        assert!(
            run(format!("echo hi > {}", inside.display())),
            "write inside workspace should succeed"
        );
        assert!(inside.exists(), "file inside workspace was written");

        let denied = run(format!("echo bad > {}", outside.display()));
        assert!(
            !denied,
            "write outside the workspace must be denied by the sandbox"
        );
        assert!(
            !outside.exists(),
            "no file should be created outside the workspace"
        );

        let _ = std::fs::remove_dir_all(&ws);
        let _ = std::fs::remove_file(&outside);
    }

    #[test]
    fn wrap_emits_sandbox_exec_invocation_when_available() {
        // Only assert the shape on macOS where sandbox-exec exists; elsewhere wrap → None.
        let policy = SandboxPolicy {
            writable_roots: vec![PathBuf::from("/work")],
            allow_network: false,
        };
        match wrap(&policy, "sh", &["-c".into(), "echo hi".into()]) {
            Some((prog, args)) => {
                assert_eq!(prog, SANDBOX_EXEC);
                assert_eq!(args[0], "-p");
                assert!(args.iter().any(|a| a.starts_with("-DWRITABLE_ROOT_0=")));
                let dd = args
                    .iter()
                    .position(|a| a == "--")
                    .expect("has -- separator");
                assert_eq!(
                    &args[dd + 1..],
                    &["sh".to_string(), "-c".to_string(), "echo hi".to_string()]
                );
            }
            None => assert!(
                !available(),
                "wrap only returns None when the sandbox is unavailable"
            ),
        }
    }
}
