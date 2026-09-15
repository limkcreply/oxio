//! MCP OAuth 2.0 for remote servers - interactive login + token store.
//!
//! rmcp does the protocol - discovery, **dynamic client registration**, and token
//! **refresh**; oxio does only the loopback callback, the browser open, and token
//! persistence. Dep-free: std + `url` + rmcp.
//!
//! Flow (login): `OAuthState::new` → `start_authorization` (dynamic registration) →
//! `get_authorization_url` → open browser → loopback catches `?code&state` →
//! `handle_callback` → `get_credentials` → persist. Connect: restore the stored token
//! and `get_access_token()` (auto-refreshes) → `Authorization: Bearer`.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;

use rmcp::transport::auth::{AuthorizationRequest, OAuthState, OAuthTokenResponse};
use serde::{Deserialize, Serialize};

/// Persisted OAuth material for one server: the (possibly dynamically-registered)
/// client id and the token response (access + refresh tokens). Written to
/// `<config>/oauth/<server>.json` with 0600 perms.
#[derive(Serialize, Deserialize)]
struct StoredToken {
    client_id: String,
    token: OAuthTokenResponse,
}

/// Interactive OAuth login for a remote MCP server. Opens the browser, catches the
/// loopback redirect, exchanges the code, and persists the token to `token_path`.
pub async fn login(
    server_name: &str,
    server_url: &str,
    scopes: &[String],
    client_id: Option<&str>,
    token_path: &Path,
) -> Result<(), String> {
    login_with(
        server_url,
        scopes,
        client_id,
        token_path,
        |auth_url, _port| {
            // Always print (so headless/SSH works) and try to open the browser.
            println!(
                "Authorize '{server_name}' by opening this URL in your browser:\n\n{auth_url}\n"
            );
            let _ = open_browser(auth_url);
        },
    )
    .await?;
    println!(
        "Authorized '{server_name}'. Token stored at {}",
        token_path.display()
    );
    Ok(())
}

/// Core login flow with the authorization URL surfaced via `on_auth_url(url, loopback_port)`
/// BEFORE blocking on the callback - so production opens the browser and a test can deliver
/// the redirect to the loopback. Everything else (discovery, registration, exchange, persist)
/// is identical to the public `login`.
async fn login_with(
    server_url: &str,
    scopes: &[String],
    client_id: Option<&str>,
    token_path: &Path,
    on_auth_url: impl FnOnce(&str, u16),
) -> Result<(), String> {
    // 1) loopback callback listener on an ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| format!("bind loopback: {e}"))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    // 2) OAuth state + start authorization (rmcp: discovery + dynamic client registration).
    let mut st = OAuthState::new(server_url, None)
        .await
        .map_err(|e| format!("oauth init: {e}"))?;
    let mut req = AuthorizationRequest::new(redirect_uri).with_client_name("oxio");
    if !scopes.is_empty() {
        req = req.with_scopes(scopes.iter().cloned());
    }
    if let Some(cid) = client_id {
        req = req.with_preregistered_client(cid);
    }
    st.start_authorization(req)
        .await
        .map_err(|e| format!("start authorization: {e}"))?;

    // 3) authorization URL → caller (browser in prod; callback injection in tests).
    let auth_url = st
        .get_authorization_url()
        .await
        .map_err(|e| e.to_string())?;
    on_auth_url(&auth_url, port);

    // 4) wait for the loopback callback.
    let (code, state) = wait_for_callback(listener)?;

    // 5) exchange the code + persist the token.
    st.handle_callback(&code, &state)
        .await
        .map_err(|e| format!("callback exchange: {e}"))?;
    let (cid, token) = st.get_credentials().await.map_err(|e| e.to_string())?;
    let token = token.ok_or_else(|| "provider returned no token".to_string())?;
    persist(
        token_path,
        &StoredToken {
            client_id: cid,
            token,
        },
    )?;
    Ok(())
}

/// Connect-side: restore the stored token and return a fresh bearer string
/// (`get_access_token` refreshes when expired). Errors point the user at `login`.
pub async fn resolve_bearer(server_url: &str, token_path: &Path) -> Result<String, String> {
    let text = std::fs::read_to_string(token_path)
        .map_err(|_| "no stored OAuth token - run `oxio login <server>` first".to_string())?;
    let stored: StoredToken =
        serde_json::from_str(&text).map_err(|e| format!("stored token unreadable: {e}"))?;
    let mut st = OAuthState::new(server_url, None)
        .await
        .map_err(|e| format!("oauth init: {e}"))?;
    st.set_credentials(&stored.client_id, stored.token)
        .await
        .map_err(|e| format!("restore token: {e}"))?;
    st.get_access_token()
        .await
        .map_err(|e| format!("get access token: {e}"))
}

fn wait_for_callback(listener: TcpListener) -> Result<(String, String), String> {
    let (mut stream, _) = listener
        .accept()
        .map_err(|e| format!("callback accept: {e}"))?;
    let mut buf = [0u8; 4096];
    let n = stream.read(&mut buf).map_err(|e| e.to_string())?;
    let req = String::from_utf8_lossy(&buf[..n]);
    // First line: `GET /callback?code=..&state=.. HTTP/1.1`.
    let target = req
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("");
    let result = parse_callback(target);
    let body = "Authentication complete. You may close this window.";
    let _ = stream.write_all(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: text/plain\r\n\r\n{}",
            body.len(),
            body
        )
        .as_bytes(),
    );
    result
}

/// Parse the OAuth redirect target (`/callback?code=..&state=..`) into `(code, state)`,
/// surfacing a provider `error=` as an Err. Pure - the testable core of the callback.
fn parse_callback(target: &str) -> Result<(String, String), String> {
    let url = url::Url::parse(&format!("http://127.0.0.1{target}"))
        .map_err(|e| format!("bad callback: {e}"))?;
    let (mut code, mut state, mut err) = (None, None, None);
    for (k, v) in url.query_pairs() {
        match &*k {
            "code" => code = Some(v.into_owned()),
            "state" => state = Some(v.into_owned()),
            "error" => err = Some(v.into_owned()),
            _ => {}
        }
    }
    if let Some(e) = err {
        return Err(format!("provider returned error: {e}"));
    }
    match (code, state) {
        (Some(c), Some(s)) => Ok((c, s)),
        _ => Err("callback missing code/state".to_string()),
    }
}

fn open_browser(url: &str) -> std::io::Result<()> {
    let cmd = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "start"
    } else {
        "xdg-open"
    };
    std::process::Command::new(cmd).arg(url).spawn().map(|_| ())
}

fn persist(path: &Path, stored: &StoredToken) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(stored).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_callback;

    #[test]
    fn parses_code_and_state() {
        assert_eq!(
            parse_callback("/callback?code=abc&state=xyz").unwrap(),
            ("abc".into(), "xyz".into())
        );
    }

    #[test]
    fn url_decodes_values() {
        // state often carries encoded chars; url crate must decode them.
        assert_eq!(
            parse_callback("/callback?code=a%2Fb&state=x%20y").unwrap(),
            ("a/b".into(), "x y".into())
        );
    }

    #[test]
    fn surfaces_provider_error() {
        assert!(parse_callback("/callback?error=access_denied").is_err());
    }

    #[test]
    fn missing_code_is_error() {
        assert!(parse_callback("/callback?state=xyz").is_err());
    }

    /// A minimal RFC-8414 OAuth authorization server on a loopback port (std only, no
    /// deps): serves well-known metadata, dynamic client registration, and a token
    /// endpoint. Enough for rmcp's `OAuthState` to discover → register → exchange.
    /// Returns the base URL.
    fn spawn_mock_oauth_server() -> String {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let b = base.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = [0u8; 8192];
                let n = s.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let path = req
                    .lines()
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("");
                let json = if path.starts_with("/.well-known/oauth-authorization-server")
                    || path.starts_with("/.well-known/openid-configuration")
                {
                    Some(format!(
                        r#"{{"issuer":"{b}","authorization_endpoint":"{b}/authorize","token_endpoint":"{b}/token","registration_endpoint":"{b}/register","response_types_supported":["code"],"grant_types_supported":["authorization_code","refresh_token"],"code_challenge_methods_supported":["S256"],"token_endpoint_auth_methods_supported":["none"],"scopes_supported":["read"]}}"#
                    ))
                } else if path.starts_with("/register") {
                    // rmcp requires the registration RESPONSE to echo redirect_uris; pull the
                    // requested array out of the POST body (crude slice - array of strings).
                    let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
                    let redirects = body
                        .find("\"redirect_uris\"")
                        .and_then(|i| {
                            let r = &body[i..];
                            let lb = r.find('[')?;
                            let rb = r[lb..].find(']')? + lb;
                            Some(r[lb..=rb].to_string())
                        })
                        .unwrap_or_else(|| "[\"http://127.0.0.1/callback\"]".to_string());
                    Some(format!(
                        r#"{{"client_id":"test-client","redirect_uris":{redirects},"token_endpoint_auth_method":"none"}}"#
                    ))
                } else if path.starts_with("/token") {
                    Some(r#"{"access_token":"test-access-token","token_type":"Bearer","refresh_token":"test-refresh-token","expires_in":3600}"#.to_string())
                } else {
                    None // incl. /.well-known/oauth-protected-resource → 404 → rmcp falls back
                };
                let resp = match json {
                    Some(j) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        j.len(), j
                    ),
                    None => "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string(),
                };
                let _ = s.write_all(resp.as_bytes());
            }
        });
        base
    }

    /// Full login seam against the mock: discovery → dynamic registration → authorization
    /// URL → loopback callback → token exchange → persisted token. Deterministic, no
    /// network, no browser.
    #[tokio::test]
    async fn full_login_flow_against_mock_server() {
        let base = spawn_mock_oauth_server();
        std::thread::sleep(std::time::Duration::from_millis(50));
        let tmp = std::env::temp_dir().join(format!("oxio-oauthflow-{}.json", std::process::id()));
        let check = tmp.clone();

        super::login_with(&base, &[], None, &tmp, |auth_url, port| {
            let state = url::Url::parse(auth_url)
                .ok()
                .and_then(|u| u.query_pairs().find(|(k, _)| k == "state").map(|(_, v)| v.into_owned()))
                .unwrap_or_default();
            std::thread::spawn(move || {
                use std::io::Write;
                std::thread::sleep(std::time::Duration::from_millis(50)); // let wait_for_callback accept
                if let Ok(mut c) = std::net::TcpStream::connect(("127.0.0.1", port)) {
                    let _ = c.write_all(
                        format!("GET /callback?code=test_code&state={state} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes(),
                    );
                }
            });
        })
        .await
        .expect("full oauth login flow");

        let stored = std::fs::read_to_string(&check).expect("token persisted");
        assert!(
            stored.contains("test-access-token"),
            "stored token holds the access token"
        );
        let _ = std::fs::remove_file(&check);
    }

    // End-to-end test of the owned loopback seam: a real TCP client hits the callback
    // listener (as the browser redirect would), and `wait_for_callback` must read the
    // raw request, return the code/state, and send back a 200.
    #[test]
    fn loopback_captures_code_and_state_over_a_real_socket() {
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || super::wait_for_callback(listener));

        let mut c = TcpStream::connect(("127.0.0.1", port)).unwrap();
        c.write_all(b"GET /callback?code=the_code&state=the_state HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut resp = String::new();
        let _ = c.read_to_string(&mut resp);
        assert!(resp.contains("200 OK"), "browser gets a 200: {resp}");

        let (code, state) = server.join().unwrap().unwrap();
        assert_eq!(code, "the_code");
        assert_eq!(state, "the_state");
    }
}
