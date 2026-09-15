//! Provider implementations for oxio. Each is a wire adapter behind the
//! frozen `oxio_core::Provider` trait. `chat_completions` first (local +
//! most vendors); `responses` and `anthropic` land in step 2.

mod anthropic;
mod chain;
mod chat_completions;
mod discovery;
mod responses;
mod swap;
mod thinking;
mod toolcalls;

pub use anthropic::AnthropicAdapter;
pub use chain::ChainProvider;
pub use chat_completions::{ChatCompletionsAdapter, Sampling};
pub use discovery::{list_models, scan_local, LocalEndpoint, KNOWN_LOCAL_ENDPOINTS};
pub use responses::ResponsesAdapter;
pub use swap::SwappableProvider;
