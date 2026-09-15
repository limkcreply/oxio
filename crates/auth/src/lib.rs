//! Auth is a THIN seam. We do not reimplement vendor auth - the real sign-in is
//! delegated to each vendor's official mechanism (reuse its stored credential,
//! shell to its official `login`, or a proven OAuth crate) when that vendor is
//! built. For now: the universal, verifiable strategies - API key and none.

use async_trait::async_trait;

/// Produces the bearer credential (if any) to attach to a request. Async so an
/// OAuth strategy can refresh transparently later.
#[async_trait]
pub trait Auth: Send + Sync {
    async fn bearer(&self) -> anyhow::Result<Option<String>>;
}

/// No credential (local servers).
pub struct NoAuth;

#[async_trait]
impl Auth for NoAuth {
    async fn bearer(&self) -> anyhow::Result<Option<String>> {
        Ok(None)
    }
}

/// A static API key.
pub struct ApiKeyAuth {
    key: String,
}

impl ApiKeyAuth {
    pub fn new(key: impl Into<String>) -> Self {
        Self { key: key.into() }
    }
}

#[async_trait]
impl Auth for ApiKeyAuth {
    async fn bearer(&self) -> anyhow::Result<Option<String>> {
        Ok(Some(self.key.clone()))
    }
}

/// Read a provider's API key from `OXIO_<PROVIDER>_API_KEY`. First cut; the
/// OS keychain is the fuller store (see plan). Returns None if unset/empty.
pub fn api_key_from_env(provider: &str) -> Option<String> {
    let var = format!("OXIO_{}_API_KEY", provider.to_uppercase().replace('-', "_"));
    std::env::var(var).ok().filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn no_auth_yields_nothing() {
        assert_eq!(NoAuth.bearer().await.unwrap(), None);
    }

    #[tokio::test]
    async fn api_key_yields_key() {
        assert_eq!(
            ApiKeyAuth::new("sk-abc").bearer().await.unwrap(),
            Some("sk-abc".to_string())
        );
    }

    #[test]
    fn env_var_resolution() {
        std::env::set_var("OXIO_TESTVENDOR_API_KEY", "sk-env");
        assert_eq!(api_key_from_env("testvendor").as_deref(), Some("sk-env"));
        assert_eq!(api_key_from_env("nope-unset"), None);
        std::env::remove_var("OXIO_TESTVENDOR_API_KEY");
    }
}
