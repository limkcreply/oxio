//! Durable cross-session memory. A file-backed store (shared by convention with
//! the `memory` tool at `OXIO_MEMORY_FILE` / `.oxio/memory.jsonl`) plus a
//! `PreTurn` injector that feeds stored facts back into context - closing the
//! loop so saved/extracted memory actually reaches the model. Per-session
//! extraction (tier 2) and cross-session consolidation (tier 3) append here.

use std::path::PathBuf;

use async_trait::async_trait;
use oxio_core::{Hook, Message, Result, Role, Transformer, TurnState};

/// The PER-PROJECT memory file (relative to cwd → scoped to this project).
pub fn memory_path() -> PathBuf {
    std::env::var("OXIO_MEMORY_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(".oxio/memory.jsonl"))
}

/// The USER-GLOBAL memory file - in the config dir, beside `config.toml` and the global
/// `AGENTS.md`, so it applies across EVERY project (user prefs vs repo-specific facts).
pub fn global_memory_path() -> PathBuf {
    std::env::var("OXIO_GLOBAL_MEMORY_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| match dirs::config_dir() {
            Some(dir) => dir.join("oxio").join("memory.jsonl"),
            None => PathBuf::from(".oxio/global-memory.jsonl"),
        })
}

fn read_lines(path: PathBuf) -> Vec<String> {
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

/// Load stored durable facts for a scope (per-project by default; global = user-wide).
pub fn load_memories() -> Vec<String> {
    read_lines(memory_path())
}

/// Load the user-global durable facts.
pub fn load_global_memories() -> Vec<String> {
    read_lines(global_memory_path())
}

fn append_to(path: PathBuf, fact: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    if let Some(p) = path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p)?;
        }
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", fact.replace('\n', " "))
}

/// Append a durable fact to the PER-PROJECT store.
pub fn append_memory(fact: &str) -> std::io::Result<()> {
    append_to(memory_path(), fact)
}

/// Append a durable fact to the USER-GLOBAL store (applies across every project).
pub fn append_global_memory(fact: &str) -> std::io::Result<()> {
    append_to(global_memory_path(), fact)
}

/// Rewrite the whole store (tier-3 consolidation: dedup/merge/prune).
pub fn replace_memories(facts: &[String]) -> std::io::Result<()> {
    use std::io::Write as _;
    let path = memory_path();
    if let Some(p) = path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p)?;
        }
    }
    let mut f = std::fs::File::create(&path)?;
    for fact in facts {
        writeln!(f, "{}", fact.replace('\n', " "))?;
    }
    Ok(())
}

/// Injects stored durable memories into context at `PreTurn` - once per turn, as
/// a system message after the main prompt - so the model always has them. The
/// block is a system message, so `session::SessionStore` (which captures only
/// non-system messages) never accumulates it.
pub struct MemoryInjector;

#[async_trait]
impl Transformer for MemoryInjector {
    async fn transform(&self, hook: Hook, state: &mut TurnState) -> Result<()> {
        if hook != Hook::PreTurn {
            return Ok(());
        }
        // Both tiers: user-global (applies everywhere) + this project's own. Global lines
        // are tagged so the model (and user) can tell repo-specific facts from user-wide ones.
        let global = load_global_memories();
        let project = load_memories();
        if global.is_empty() && project.is_empty() {
            return Ok(());
        }
        let mut lines: Vec<String> = Vec::new();
        lines.extend(global.iter().map(|m| format!("- (global) {m}")));
        lines.extend(project.iter().map(|m| format!("- {m}")));
        let block = format!(
            "[Durable memory - apply when relevant]\n{}",
            lines.join("\n")
        );
        // Insert after any existing system messages, before the first non-system.
        let pos = state
            .messages
            .iter()
            .position(|m| m.role != Role::System)
            .unwrap_or(state.messages.len());
        state
            .messages
            .insert(pos, Message::text(Role::System, block));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxio_core::TurnState;

    fn state(msgs: Vec<Message>) -> TurnState {
        TurnState {
            input: String::new(),
            model: "m".into(),
            messages: msgs,
            tools: vec![],
            params: Default::default(),
            usage: Default::default(),
            last_error: None,
            notices: Vec::new(),
        }
    }

    // One test (env var is process-global - avoid inter-test races).
    #[tokio::test]
    async fn injector_reads_store_and_appends_roundtrip() {
        let dir = std::env::temp_dir().join(format!("oxio_mem_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("memory.jsonl");
        std::env::set_var("OXIO_MEMORY_FILE", &file);

        // Empty store → no injection.
        let mut empty = state(vec![Message::text(Role::User, "hi")]);
        MemoryInjector
            .transform(Hook::PreTurn, &mut empty)
            .await
            .unwrap();
        assert_eq!(empty.messages.len(), 1, "no memory → no injection");

        // append_memory → load → inject.
        append_memory("user prefers rust").unwrap();
        append_memory("deploy via cdk only").unwrap();
        assert_eq!(load_memories().len(), 2);

        let mut st = state(vec![
            Message::text(Role::System, "sys"),
            Message::text(Role::User, "hi"),
        ]);
        MemoryInjector
            .transform(Hook::PreTurn, &mut st)
            .await
            .unwrap();
        assert_eq!(st.messages[0].as_text(), "sys");
        assert_eq!(
            st.messages[1].role,
            Role::System,
            "memory injected as system, after main prompt"
        );
        assert!(st.messages[1].as_text().contains("user prefers rust"));
        assert_eq!(st.messages[2].as_text(), "hi", "user input follows");

        // tier-3 consolidation rewrites the whole store.
        replace_memories(&["merged: rust + cdk".to_string()]).unwrap();
        assert_eq!(load_memories(), vec!["merged: rust + cdk".to_string()]);

        std::env::remove_var("OXIO_MEMORY_FILE");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
