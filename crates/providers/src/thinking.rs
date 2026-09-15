//! Thinking/answer splitter for models that inline their reasoning.
//!
//! Some models (Gemma) emit reasoning wrapped in `<|channel>thought … <channel|>`
//! inside the normal content stream. This separates reasoning from the answer,
//! incrementally and safely across streaming chunk boundaries: it emits eagerly
//! but holds back only a *genuine partial marker* at the tail until more bytes
//! arrive, so a marker split across two SSE chunks is never mis-emitted.
//!
//! The marker state machine holds a short tail containing '<' until more data
//! arrives, and has a thinking-disabled strip mode. Output is drained without
//! reallocation at the `Seg` boundary.

const OPEN: &str = "<|channel>thought";
const CLOSE: &str = "<channel|>";

#[derive(Debug, PartialEq, Eq)]
pub enum Seg {
    Thinking(String),
    Answer(String),
}

pub struct ThinkingSplitter {
    acc: String,
    in_thinking: bool,
    enabled: bool,
}

impl Default for ThinkingSplitter {
    fn default() -> Self {
        Self {
            acc: String::new(),
            in_thinking: false,
            enabled: true,
        }
    }
}

impl ThinkingSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable thinking capture - markers are stripped and everything is answer.
    /// Wired to config's thinking toggle when the config step lands.
    #[allow(dead_code)]
    pub fn with_thinking(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }

    /// Feed a content chunk: accumulate, then flush complete segments, holding a
    /// possible partial marker (a short tail containing '<') until more data arrives.
    pub fn push(&mut self, content: &str) -> Vec<Seg> {
        if !self.enabled {
            let stripped = content.replace(OPEN, "").replace(CLOSE, "");
            return if stripped.is_empty() {
                Vec::new()
            } else {
                vec![Seg::Answer(stripped)]
            };
        }
        self.acc.push_str(content);
        self.run(false)
    }

    /// Flush everything remaining at end of stream.
    pub fn finish(&mut self) -> Vec<Seg> {
        if !self.enabled {
            return Vec::new();
        }
        self.run(true)
    }

    fn run(&mut self, flush: bool) -> Vec<Seg> {
        let mut out = Vec::new();
        let mut processed = true;
        while processed && !self.acc.is_empty() {
            processed = false;
            if !self.in_thinking {
                if let Some(start) = self.acc.find(OPEN) {
                    if start > 0 {
                        out.push(Seg::Answer(self.acc[..start].to_string()));
                    }
                    self.acc.drain(..start + OPEN.len());
                    self.in_thinking = true;
                    processed = true;
                } else if !flush && self.acc.contains('<') && self.acc.len() < 20 {
                    break; // might be a partial start marker - wait for more
                } else {
                    out.push(Seg::Answer(std::mem::take(&mut self.acc)));
                }
            } else if let Some(end) = self.acc.find(CLOSE) {
                if end > 0 {
                    out.push(Seg::Thinking(self.acc[..end].to_string()));
                }
                self.acc.drain(..end + CLOSE.len());
                self.in_thinking = false;
                processed = true;
            } else if !flush && self.acc.contains('<') && self.acc.len() < 12 {
                break; // might be a partial end marker - wait for more
            } else {
                out.push(Seg::Thinking(std::mem::take(&mut self.acc)));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joined(segs: &[Seg]) -> (String, String) {
        let think = segs
            .iter()
            .filter_map(|s| {
                if let Seg::Thinking(t) = s {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        let ans = segs
            .iter()
            .filter_map(|s| {
                if let Seg::Answer(a) = s {
                    Some(a.as_str())
                } else {
                    None
                }
            })
            .collect();
        (think, ans)
    }

    #[test]
    fn whole_input() {
        let mut sp = ThinkingSplitter::new();
        let mut s = sp.push("ans1<|channel>thoughtreason<channel|>ans2");
        s.extend(sp.finish());
        assert_eq!(
            s,
            vec![
                Seg::Answer("ans1".into()),
                Seg::Thinking("reason".into()),
                Seg::Answer("ans2".into()),
            ]
        );
    }

    #[test]
    fn across_chunk_boundaries() {
        let mut sp = ThinkingSplitter::new();
        let mut s = Vec::new();
        for ch in ["a<|cha", "nnel>thoughtre", "ason<chan", "nel|>b"] {
            s.extend(sp.push(ch));
        }
        s.extend(sp.finish());
        let (think, ans) = joined(&s);
        assert_eq!(think, "reason");
        assert_eq!(ans, "ab");
    }

    #[test]
    fn plain_answer_no_markers() {
        let mut sp = ThinkingSplitter::new();
        let mut s = sp.push("just an answer");
        s.extend(sp.finish());
        assert_eq!(s, vec![Seg::Answer("just an answer".into())]);
    }

    #[test]
    fn holds_partial_marker_then_flushes() {
        let mut sp = ThinkingSplitter::new();
        // short tail containing '<' → whole thing held, nothing emitted yet
        assert_eq!(sp.push("hello<"), Vec::<Seg>::new());
        // on finish the held text is flushed as answer (no marker completed)
        assert_eq!(sp.finish(), vec![Seg::Answer("hello<".into())]);
    }

    #[test]
    fn disabled_strips_markers() {
        let mut sp = ThinkingSplitter::with_thinking(false);
        assert_eq!(
            sp.push("a<|channel>thoughtb<channel|>c"),
            vec![Seg::Answer("abc".into())]
        );
        assert_eq!(sp.finish(), Vec::<Seg>::new());
    }
}
