//! Persistent multi-turn conversation history - the plan's "session state is a
//! module" pillar, as a bolt-on `Transformer` (no kernel change). The kernel
//! builds a fresh message list per `run_turn`; this injects prior history at
//! `PreTurn` and captures the turn back at `PostTurn`, so the REPL remembers the
//! conversation. Prerequisite for manual `/compact` and the memory module.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use oxio_core::{Hook, Message, Result, Role, Transformer, TurnState};
use serde::{Deserialize, Serialize};

/// Head line of a transcript. Lets `--resume` list past sessions with context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub cwd: String,
    pub model: String,
    pub created_ms: u64,
}

/// One transcript line: either the head `meta`, or a timestamped `message`. A
/// plain struct with optional fields (avoids serde internally-tagged-enum edge
/// cases) and stays forward-compatible - unknown fields are ignored.
#[derive(Serialize, Deserialize)]
struct Record {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    meta: Option<SessionMeta>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ts_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    message: Option<Message>,
}

/// A past session, for the `--resume` picker.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub path: PathBuf,
    pub meta: Option<SessionMeta>,
    /// First user prompt, truncated - the human-recognizable preview.
    pub preview: String,
    pub message_count: usize,
}

/// Milliseconds since the Unix epoch (for session/message timestamps).
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Holds the conversation transcript (non-system messages) across turns, and
/// optionally mirrors every new message to a `.jsonl` on disk for `--continue`.
#[derive(Default)]
pub struct SessionStore {
    history: Mutex<Vec<Message>>,
    /// Where to append the verbatim transcript. `None` = in-memory only (tests).
    transcript: Mutex<Option<PathBuf>>,
    /// Head metadata to write once, on the first append of a fresh session.
    meta: Mutex<Option<SessionMeta>>,
    /// Whether the meta head line has been written (or the file pre-exists on resume).
    meta_written: Mutex<bool>,
}

impl SessionStore {
    /// In-memory only (no transcript file). Used by tests and non-persistent callers.
    pub fn new() -> Arc<Self> {
        Arc::new(SessionStore::default())
    }

    /// Auto-log every turn's new messages verbatim to `path` (created on first
    /// append). `meta` is written as the head line before the first message.
    pub fn new_logging(path: PathBuf, meta: SessionMeta) -> Arc<Self> {
        index_session(&path);
        let store = SessionStore::default();
        *store.transcript.lock().unwrap() = Some(path);
        *store.meta.lock().unwrap() = Some(meta);
        Arc::new(store)
    }

    /// Point logging at `path` (used by `--continue` to keep appending to the
    /// resumed session's file).
    pub fn set_transcript(&self, path: PathBuf) {
        *self.transcript.lock().unwrap() = Some(path);
    }

    pub fn transcript_path(&self) -> Option<PathBuf> {
        self.transcript.lock().unwrap().clone()
    }

    /// Load a transcript file into history AND keep appending to it - the
    /// `--continue`/`--resume` action: full verbatim restore, then resume logging.
    /// The file already has its meta head, so we do not rewrite it.
    pub fn continue_from(&self, path: &Path) {
        self.replace(load_transcript(path));
        self.set_transcript(path.to_path_buf());
        *self.meta_written.lock().unwrap() = true;
    }

    /// Append messages to the transcript, one tagged JSON line each. Writes the
    /// meta head first on a fresh session. Best-effort: a persistence failure
    /// never breaks the turn.
    fn append_records(&self, msgs: &[Message]) {
        if msgs.is_empty() {
            return;
        }
        let Some(path) = self.transcript_path() else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) else {
            return;
        };
        // Head meta line, once, on a fresh session.
        if !*self.meta_written.lock().unwrap() {
            if let Some(meta) = self.meta.lock().unwrap().clone() {
                let rec = Record {
                    meta: Some(meta),
                    ts_ms: None,
                    message: None,
                };
                if let Ok(line) = serde_json::to_string(&rec) {
                    let _ = writeln!(f, "{line}");
                }
            }
            *self.meta_written.lock().unwrap() = true;
        }
        let ts_ms = now_ms();
        for m in msgs {
            let rec = Record {
                meta: None,
                ts_ms: Some(ts_ms),
                message: Some(m.clone()),
            };
            if let Ok(line) = serde_json::to_string(&rec) {
                let _ = writeln!(f, "{line}");
            }
        }
    }

    pub fn clear(&self) {
        self.history.lock().unwrap().clear();
    }
    pub fn is_empty(&self) -> bool {
        self.history.lock().unwrap().is_empty()
    }
    pub fn len(&self) -> usize {
        self.history.lock().unwrap().len()
    }
    /// Snapshot of stored history (for manual /compact).
    pub fn snapshot(&self) -> Vec<Message> {
        self.history.lock().unwrap().clone()
    }
    /// Replace stored history (e.g. after manual compaction, or a `--continue` load).
    pub fn replace(&self, msgs: Vec<Message>) {
        *self.history.lock().unwrap() = msgs;
    }
}

/// Directory holding session transcripts. `OXIO_SESSION_DIR` overrides the
/// default `.oxio/sessions`.
pub fn sessions_dir() -> PathBuf {
    std::env::var_os("OXIO_SESSION_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".oxio/sessions"))
}

/// Path for a fresh session transcript, keyed by a millisecond timestamp so files
/// sort chronologically.
pub fn new_session_path() -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    sessions_dir().join(format!("sess-{ts}.jsonl"))
}

/// Machine-level index mapping a session id (the file stem) to its transcript path, so a
/// session can be found by id from any project. `OXIO_SESSION_INDEX` overrides the default.
pub fn session_index_path() -> PathBuf {
    std::env::var_os("OXIO_SESSION_INDEX")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_default();
            home.join(".oxio").join("session-index")
        })
}

/// Record `id -> path` in the machine index (best-effort, append-only). Called on session
/// creation so `--resume <id>` can locate a session created in any project.
pub fn index_session(path: &Path) {
    append_index(&session_index_path(), &absolutize(path));
}

/// Resolve `path` against the current directory when relative. The index is
/// machine-wide and read from any cwd, so a relative `.oxio/sessions/...` would
/// fail `exists()` outside its own project folder. cwd here is the session's
/// project at creation time.
fn absolutize(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|d| d.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

fn append_index(idx: &Path, path: &Path) {
    let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
        return;
    };
    if let Some(dir) = idx.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(idx) {
        let _ = writeln!(f, "{id}\t{}", path.display());
    }
}

/// Resolve a session id to its transcript path via the machine index. Returns the path only
/// if the id is indexed and the file still exists; newest matching entry wins.
pub fn session_path_by_id(id: &str) -> Option<PathBuf> {
    lookup_index(&session_index_path(), id)
}

fn lookup_index(idx: &Path, id: &str) -> Option<PathBuf> {
    let body = std::fs::read_to_string(idx).ok()?;
    body.lines()
        .rev()
        .filter_map(|l| l.split_once('\t'))
        .find(|(k, _)| *k == id)
        .map(|(_, p)| PathBuf::from(p))
        .filter(|p| p.exists())
}

/// The most recently modified session transcript, if any (for `--continue`).
pub fn latest_session() -> Option<PathBuf> {
    let dir = sessions_dir();
    let mut entries: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(&dir)
        .ok()?
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    entries.sort_by_key(|(t, _)| *t);
    entries.pop().map(|(_, p)| p)
}

/// Read a transcript file back into messages, in order. Lossless: every `Msg`
/// record contributes its verbatim `Message`; the meta head and malformed lines
/// are skipped rather than failing the whole load.
pub fn load_transcript(path: &Path) -> Vec<Message> {
    let Ok(f) = File::open(path) else {
        return Vec::new();
    };
    BufReader::new(f)
        .lines()
        .map_while(std::result::Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            serde_json::from_str::<Record>(&l)
                .ok()
                .and_then(|r| r.message)
        })
        .collect()
}

/// Read just the head meta line of a transcript, if present.
pub fn read_session_meta(path: &Path) -> Option<SessionMeta> {
    let f = File::open(path).ok()?;
    for line in BufReader::new(f).lines().map_while(std::result::Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(r) = serde_json::from_str::<Record>(&line) {
            return r.meta;
        }
        // Meta is the first record; stop once we hit a message.
        return None;
    }
    None
}

/// List past sessions (newest first) with meta + a first-prompt preview, for the
/// `--resume` picker and `session list`.
pub fn list_sessions() -> Vec<SessionInfo> {
    let dir = sessions_dir();
    let mut infos: Vec<(SystemTime, SessionInfo)> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "jsonl"))
            .filter_map(|e| {
                let path = e.path();
                let mtime = e.metadata().ok()?.modified().ok()?;
                let msgs = load_transcript(&path);
                let preview = msgs
                    .iter()
                    .find(|m| m.role == Role::User)
                    .map(|m| m.as_text().chars().take(60).collect::<String>())
                    .unwrap_or_default();
                Some((
                    mtime,
                    SessionInfo {
                        meta: read_session_meta(&path),
                        preview,
                        message_count: msgs.len(),
                        path,
                    },
                ))
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    infos.sort_by_key(|(t, _)| *t);
    infos.into_iter().rev().map(|(_, i)| i).collect()
}

/// One matching message from a past session, for the `recall` tool.
#[derive(Debug, Clone)]
pub struct RecallHit {
    pub session_id: String,
    pub role: String,
    /// A trimmed one-line window of the message around the match.
    pub excerpt: String,
}

/// Every transcript path recorded in the machine index, in file order (oldest first).
fn indexed_paths() -> Vec<PathBuf> {
    let Ok(body) = std::fs::read_to_string(session_index_path()) else {
        return Vec::new();
    };
    body.lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(_, p)| PathBuf::from(p))
        .collect()
}

/// Search past session transcripts for `query` (case-insensitive substring). This
/// project's sessions come first (newest first), then any other sessions in the machine
/// index. Returns up to `limit` matching user/assistant messages as excerpts. Read-only.
pub fn search_sessions(query: &str, limit: usize) -> Vec<RecallHit> {
    // This project's sessions first, then machine-wide, deduped by path so a project
    // session is never scanned twice.
    let mut seen = std::collections::HashSet::new();
    let mut paths: Vec<PathBuf> = Vec::new();
    for info in list_sessions() {
        if seen.insert(info.path.clone()) {
            paths.push(info.path);
        }
    }
    for p in indexed_paths().into_iter().rev() {
        if p.exists() && seen.insert(p.clone()) {
            paths.push(p);
        }
    }
    search_paths(&paths, query, limit)
}

/// The core scan over an explicit, ordered list of transcript files. Split out from
/// `search_sessions` so it is testable without touching the machine's session dirs.
fn search_paths(paths: &[PathBuf], query: &str, limit: usize) -> Vec<RecallHit> {
    let q = query.trim().to_lowercase();
    if q.is_empty() || limit == 0 {
        return Vec::new();
    }
    let mut hits = Vec::new();
    for path in paths {
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("?")
            .to_string();
        for m in load_transcript(path) {
            if !matches!(m.role, Role::User | Role::Assistant) {
                continue;
            }
            let text = m.as_text();
            if text.to_lowercase().contains(&q) {
                hits.push(RecallHit {
                    session_id: id.clone(),
                    role: format!("{:?}", m.role).to_lowercase(),
                    excerpt: excerpt(&text, &q),
                });
                if hits.len() >= limit {
                    return hits;
                }
            }
        }
    }
    hits
}

/// A whitespace-collapsed window of `text` around the first case-insensitive match of
/// `q_lower`, with ellipses when trimmed. Operates on chars so it never splits a
/// multi-byte boundary.
fn excerpt(text: &str, q_lower: &str) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = flat.chars().collect();
    let lower = flat.to_lowercase();
    let Some(byte_pos) = lower.find(q_lower) else {
        return chars.iter().take(160).collect();
    };
    let char_pos = lower[..byte_pos].chars().count();
    let start = char_pos.saturating_sub(50);
    let end = (char_pos + q_lower.chars().count() + 110).min(chars.len());
    let mut s = String::new();
    if start > 0 {
        s.push('…');
    }
    s.extend(&chars[start..end]);
    if end < chars.len() {
        s.push('…');
    }
    s
}

#[async_trait]
impl Transformer for SessionStore {
    async fn transform(&self, hook: Hook, state: &mut TurnState) -> Result<()> {
        match hook {
            Hook::PreTurn => {
                let history = self.history.lock().unwrap().clone();
                if history.is_empty() {
                    return Ok(());
                }
                // Rebuild: system message(s) first, then prior history, then this
                // turn's non-system messages (the new user input). Order-independent
                // of the system-prompt transformer.
                let (system, current): (Vec<Message>, Vec<Message>) = state
                    .messages
                    .drain(..)
                    .partition(|m| m.role == Role::System);
                let mut rebuilt = system;
                rebuilt.extend(history);
                rebuilt.extend(current);
                state.messages = rebuilt;
            }
            Hook::PostTurn => {
                // Capture the whole turn's non-system messages as the new history
                // (includes the prior history injected at PreTurn - it accumulates).
                let hist: Vec<Message> = state
                    .messages
                    .iter()
                    .filter(|m| m.role != Role::System)
                    .cloned()
                    .collect();
                // Fresh transcript: persist the whole history (including any resumed or
                // seeded context) so the file is self-contained. Established one: append
                // only this turn's tail (from its last user message), position-based so
                // mid-turn compaction cannot drop records.
                let start = if *self.meta_written.lock().unwrap() {
                    hist.iter().rposition(|m| m.role == Role::User).unwrap_or(0)
                } else {
                    0
                };
                self.append_records(&hist[start..]);
                *self.history.lock().unwrap() = hist;
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test]
    async fn history_carries_across_turns() {
        let store = SessionStore::new();

        // Turn 1: system + user + assistant → captured at PostTurn.
        let mut t1 = state(vec![
            Message::text(Role::System, "sys"),
            Message::text(Role::User, "q1"),
            Message::text(Role::Assistant, "a1"),
        ]);
        store.transform(Hook::PostTurn, &mut t1).await.unwrap();
        assert_eq!(store.len(), 2, "q1 + a1 stored (system excluded)");

        // Turn 2: fresh [system, user q2] → PreTurn injects prior history.
        let mut t2 = state(vec![
            Message::text(Role::System, "sys"),
            Message::text(Role::User, "q2"),
        ]);
        store.transform(Hook::PreTurn, &mut t2).await.unwrap();
        let texts: Vec<String> = t2.messages.iter().map(|m| m.as_text()).collect();
        assert_eq!(
            texts,
            vec!["sys", "q1", "a1", "q2"],
            "system, prior history, then new input"
        );
    }

    #[tokio::test]
    async fn clear_resets_history() {
        let store = SessionStore::new();
        let mut t = state(vec![Message::text(Role::User, "x")]);
        store.transform(Hook::PostTurn, &mut t).await.unwrap();
        assert!(!store.is_empty());
        store.clear();
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn transcript_persists_verbatim_and_reloads_lossless() {
        let dir = std::env::temp_dir().join(format!("oxio-sess-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t.jsonl");
        let _ = std::fs::remove_file(&path);
        // Keep new_logging's index write off the real machine index.
        std::env::set_var("OXIO_SESSION_INDEX", dir.join("idx"));
        let meta = SessionMeta {
            id: "id".into(),
            cwd: ".".into(),
            model: "m".into(),
            created_ms: 0,
        };
        let store = SessionStore::new_logging(path.clone(), meta);

        // Turn 1: user q1 + assistant a1.
        let mut t1 = state(vec![
            Message::text(Role::User, "q1"),
            Message::text(Role::Assistant, "a1"),
        ]);
        store.transform(Hook::PreTurn, &mut t1).await.unwrap();
        store.transform(Hook::PostTurn, &mut t1).await.unwrap();

        // Turn 2: fresh [system, q2] → PreTurn injects prior, kernel-style a2 appended.
        let mut t2 = state(vec![
            Message::text(Role::System, "sys"),
            Message::text(Role::User, "q2"),
        ]);
        store.transform(Hook::PreTurn, &mut t2).await.unwrap();
        t2.messages.push(Message::text(Role::Assistant, "a2"));
        store.transform(Hook::PostTurn, &mut t2).await.unwrap();

        // File holds ALL four messages verbatim, in order, no duplication.
        let texts: Vec<String> = load_transcript(&path).iter().map(|m| m.as_text()).collect();
        assert_eq!(
            texts,
            vec!["q1", "a1", "q2", "a2"],
            "full verbatim transcript, no dupes"
        );

        // --continue: a new store restores every message losslessly.
        let resumed = SessionStore::new();
        resumed.continue_from(&path);
        assert_eq!(resumed.len(), 4, "resume restores the full session");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn session_index_write_then_resolve() {
        let tag = now_ms();
        let sess = std::env::temp_dir().join(format!("sess-{tag}.jsonl"));
        std::fs::write(&sess, "x").unwrap();
        let idx = std::env::temp_dir().join(format!("oxio-idx-{tag}"));
        append_index(&idx, &sess);
        let id = sess.file_stem().unwrap().to_str().unwrap();
        assert_eq!(
            lookup_index(&idx, id),
            Some(sess.clone()),
            "id resolves to its path"
        );
        assert_eq!(lookup_index(&idx, "sess-absent"), None);
        let _ = std::fs::remove_file(&sess);
        let _ = std::fs::remove_file(&idx);
    }

    #[test]
    fn absolutize_makes_relative_paths_absolute() {
        // The production path: sessions_dir() is relative, so an unresolved
        // relative entry would break cross-project resume.
        assert!(absolutize(Path::new(".oxio/sessions/sess-1.jsonl")).is_absolute());
        let abs = std::env::temp_dir().join("sess-2.jsonl");
        assert_eq!(
            absolutize(&abs),
            abs,
            "absolute paths pass through unchanged"
        );
    }

    #[test]
    fn recall_finds_a_message_and_skips_non_matches() {
        let tag = now_ms();
        let dir = std::env::temp_dir().join(format!("oxio-recall-{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sess-recall.jsonl");
        let rec = Record {
            meta: None,
            ts_ms: Some(1),
            message: Some(Message::text(Role::User, "the meaning of pink is apple")),
        };
        std::fs::write(&path, format!("{}\n", serde_json::to_string(&rec).unwrap())).unwrap();

        let hits = search_paths(std::slice::from_ref(&path), "pink", 10);
        assert_eq!(hits.len(), 1, "the matching message is found");
        assert_eq!(hits[0].session_id, "sess-recall");
        assert_eq!(hits[0].role, "user");
        assert!(hits[0].excerpt.to_lowercase().contains("pink"));

        assert!(search_paths(std::slice::from_ref(&path), "banana", 10).is_empty());
        assert!(search_paths(std::slice::from_ref(&path), "", 10).is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn excerpt_windows_around_match_without_panicking() {
        let e = excerpt("café pink lady from the orchard", "pink");
        assert!(e.to_lowercase().contains("pink"), "multibyte text is safe");
        let long = format!("{}pink", "x ".repeat(200));
        let e2 = excerpt(&long, "pink");
        assert!(e2.contains('…'), "long text is ellipsized");
        assert!(e2.to_lowercase().contains("pink"));
    }
}
