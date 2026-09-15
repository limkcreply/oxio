//! Local-endpoint discovery for first-run onboarding. Probes the well-known
//! OpenAI-compatible local servers (Ollama, LM Studio, and others) so oxio can
//! auto-detect a model instead of making the user hand-edit config.

use std::time::Duration;

/// A detected local model endpoint.
pub struct LocalEndpoint {
    /// Human label, e.g. "Ollama".
    pub label: String,
    /// API root to put in `base_url`, e.g. `http://localhost:11434/v1`.
    pub base_url: String,
    /// EVERY model the server advertises. An endpoint like Ollama can serve many
    /// (all downloaded models) with none "active" - it hot-loads per request - so we
    /// list them all and let the user pick, never guessing one.
    pub models: Vec<String>,
}

/// GET `{base_url}/models`; return EVERY advertised model id (empty if the server
/// is unreachable or advertises none). Used by `/connect models` so a user can see
/// all models a single machine serves and switch between them by id.
pub async fn list_models(base_url: &str) -> Vec<String> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_millis(1500))
        .build()
    {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let resp = match client.get(&url).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => return Vec::new(),
    };
    let v: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    v.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|x| x.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// The well-known local OpenAI-compatible servers oxio probes on first run.
/// Convenience ONLY - it covers the famous defaults so a zero-config user is found
/// without typing. It deliberately does NOT try to find custom-port or remote servers:
/// those are reached by registration (`provider add` / `/connect add`), the real
/// code-free mechanism. Extending discovery is NOT how you add an endpoint - you add it.
pub const KNOWN_LOCAL_ENDPOINTS: &[(&str, &str)] = &[
    ("Ollama", "http://localhost:11434/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
    ("llama.cpp", "http://localhost:8080/v1"),
    ("vLLM", "http://localhost:8000/v1"),
    ("Jan", "http://localhost:1337/v1"),
    ("text-gen-webui", "http://localhost:5000/v1"),
];

/// Scan the well-known local endpoints. Returns those that responded (with ALL the
/// models each advertises), in probe order. Empty if nothing famous is running - which
/// is NOT a dead end: the caller always offers manual add (custom endpoint / cloud).
pub async fn scan_local() -> Vec<LocalEndpoint> {
    // Probe every endpoint concurrently so a wall of dead ports doesn't stall first run
    // (6 × 1.5s sequential → one ~1.5s round). Preserve probe order in the result.
    let probes = KNOWN_LOCAL_ENDPOINTS
        .iter()
        .map(|(label, base)| async move { (*label, *base, list_models(base).await) });
    let mut found = Vec::new();
    for (label, base, models) in futures::future::join_all(probes).await {
        if !models.is_empty() {
            found.push(LocalEndpoint {
                label: label.to_string(),
                base_url: base.to_string(),
                models,
            });
        }
    }
    found
}
