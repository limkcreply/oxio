//! The registry holds every installed extension. Populated at startup, read by
//! the kernel during a turn. Modules never see each other - they only add here.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::kernel::Hook;
use crate::traits::{Module, Observer, Provider, Tool, Transformer};

/// The live tool map, shared behind a lock so a module can update its own tools at
/// runtime (e.g. an MCP server sending `tools/list_changed`) while the kernel reads
/// the current set at the start of each turn. Registration is still startup-time via
/// `add_tool`; runtime swaps go through [`Registry::swap_server_tools`].
pub type SharedTools = Arc<RwLock<HashMap<String, Arc<dyn Tool>>>>;

#[derive(Default)]
pub struct Registry {
    pub(crate) providers: Vec<Arc<dyn Provider>>,
    pub(crate) tools: SharedTools,
    pub(crate) observers: Vec<Arc<dyn Observer>>,
    pub(crate) transformers: Vec<(Hook, Arc<dyn Transformer>)>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Providers form the fallback chain in registration order (first = primary).
    pub fn add_provider(&mut self, p: Arc<dyn Provider>) -> &mut Self {
        self.providers.push(p);
        self
    }

    pub fn add_tool(&mut self, t: Arc<dyn Tool>) -> &mut Self {
        self.tools
            .write()
            .expect("tools lock")
            .insert(t.spec().name, t);
        self
    }

    /// A clonable handle to the live tool map, for a module that updates its tools at
    /// runtime. Holding it lets that module call [`Registry::swap_server_tools`].
    pub fn tools_handle(&self) -> SharedTools {
        Arc::clone(&self.tools)
    }

    /// Replace all tools whose name starts with `prefix` with `tools` (used when an MCP
    /// server re-advertises after `tools/list_changed`). Atomic under the write lock; the
    /// kernel's next turn sees the new set. A no-op-safe operation on any handle.
    pub fn swap_server_tools(map: &SharedTools, prefix: &str, tools: Vec<Arc<dyn Tool>>) {
        let mut g = map.write().expect("tools lock");
        g.retain(|name, _| !name.starts_with(prefix));
        for t in tools {
            g.insert(t.spec().name, t);
        }
    }

    pub fn add_observer(&mut self, o: Arc<dyn Observer>) -> &mut Self {
        self.observers.push(o);
        self
    }

    /// Register a transformer to run at a specific [`Hook`].
    pub fn add_transformer(&mut self, hook: Hook, t: Arc<dyn Transformer>) -> &mut Self {
        self.transformers.push((hook, t));
        self
    }

    /// Install a feature module (which adds its own providers/tools/etc.).
    pub fn install(&mut self, module: Box<dyn Module>) -> &mut Self {
        module.register(self);
        self
    }

    pub(crate) fn tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.read().expect("tools lock").get(name).cloned()
    }
}
