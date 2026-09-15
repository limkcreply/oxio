//! Robust tool-call format adapter layer.
//!
//! Local models emit tool calls in many text formats, and OpenAI-compat servers
//! are inconsistent about converting them to the native `tool_calls` field. This
//! layer parses the common text formats when native calls are absent. Adding
//! support for a new model/format = write one parser fn and add it to `FORMATS`.
//!
//! Each parser returns `(calls, cleaned_text)` - the tool calls it found plus the
//! message text with the call markup removed. Parsers are tried in order; the
//! first that finds a call wins. `parse` is the single entry point.

use oxio_core::ToolCall;
use serde_json::{json, Value};

type FormatParser = fn(&str) -> (Vec<ToolCall>, String);

/// Registered text tool-call formats, tried in order. Extend here as new models
/// appear (Llama `<|python_tag|>`, Mistral `[TOOL_CALLS]`, fenced ```json, …).
const FORMATS: &[FormatParser] = &[parse_xml_function, parse_json_tool_call];

/// Parse tool calls emitted as text, trying each known format. Returns
/// `(calls, cleaned_text)`; empty calls + original text if none match.
pub fn parse(text: &str) -> (Vec<ToolCall>, String) {
    for parser in FORMATS {
        let (calls, cleaned) = parser(text);
        if !calls.is_empty() {
            return (calls, cleaned);
        }
    }
    (Vec::new(), text.to_string())
}

/// Turn an accumulated tool-call argument string into a JSON value WITHOUT hiding
/// malformed input. Empty = a legitimate no-argument call (`{}`). Non-empty but
/// unparseable = the model emitted broken JSON; surface it as `_malformed_arguments`
/// so the tool rejects it with a clear schema error, instead of silently running
/// with `{}` (a bug-hiding fallback). Shared by all wire adapters.
pub(crate) fn parse_tool_args(raw: &str) -> Value {
    let t = raw.trim();
    if t.is_empty() {
        return json!({});
    }
    serde_json::from_str(t).unwrap_or_else(|_| json!({ "_malformed_arguments": raw }))
}

/// A tool-call argument value: JSON when it parses (numbers, arrays, objects,
/// bools), else a plain string (paths, prose, commands).
fn arg_value(val: &str) -> Value {
    let t = val.trim();
    serde_json::from_str::<Value>(t).unwrap_or_else(|_| Value::String(t.to_string()))
}

fn cut_before(text: &str, marker: &str) -> String {
    text.find(marker)
        .map(|i| text[..i].trim().to_string())
        .unwrap_or_default()
}

/// Qwen3-Coder XML: `<function=NAME><parameter=P>V</parameter>…</function>`
/// (optionally wrapped in `<tool_call>…</tool_call>`).
fn parse_xml_function(text: &str) -> (Vec<ToolCall>, String) {
    const F_OPEN: &str = "<function=";
    const F_CLOSE: &str = "</function>";
    const P_OPEN: &str = "<parameter=";
    const P_CLOSE: &str = "</parameter>";

    let mut calls = Vec::new();
    let mut search = text;
    let mut n = 0;
    while let Some(fs) = search.find(F_OPEN) {
        let after = &search[fs + F_OPEN.len()..];
        let ne = match after.find('>') {
            Some(e) => e,
            None => break,
        };
        let name = after[..ne].trim().to_string();
        let body_start = fs + F_OPEN.len() + ne + 1;
        let (body, next) = match search[body_start..].find(F_CLOSE) {
            Some(e) => (
                &search[body_start..body_start + e],
                body_start + e + F_CLOSE.len(),
            ),
            None => (&search[body_start..], search.len()),
        };
        let mut args = serde_json::Map::new();
        let mut pb = body;
        while let Some(ps) = pb.find(P_OPEN) {
            let pa = &pb[ps + P_OPEN.len()..];
            let pne = match pa.find('>') {
                Some(e) => e,
                None => break,
            };
            let pname = pa[..pne].trim().to_string();
            let vs = ps + P_OPEN.len() + pne + 1;
            let ve = match pb[vs..].find(P_CLOSE) {
                Some(e) => e,
                None => break,
            };
            args.insert(pname, arg_value(&pb[vs..vs + ve]));
            pb = &pb[vs + ve + P_CLOSE.len()..];
        }
        if !name.is_empty() {
            n += 1;
            calls.push(ToolCall {
                id: format!("call_{n}"),
                name,
                arguments: Value::Object(args),
            });
        }
        search = &search[next..];
    }
    if calls.is_empty() {
        (calls, text.to_string())
    } else {
        (calls, cut_before(text, F_OPEN))
    }
}

/// Hermes/Qwen JSON: `<tool_call>{ "name":.., "arguments":.. }</tool_call>`.
fn parse_json_tool_call(text: &str) -> (Vec<ToolCall>, String) {
    const T_OPEN: &str = "<tool_call>";
    const T_CLOSE: &str = "</tool_call>";

    let mut calls = Vec::new();
    let mut search = text;
    let mut n = 0;
    while let Some(ts) = search.find(T_OPEN) {
        let after = &search[ts + T_OPEN.len()..];
        let te = match after.find(T_CLOSE) {
            Some(e) => e,
            None => break,
        };
        if let Ok(v) = serde_json::from_str::<Value>(after[..te].trim()) {
            if let Some(name) = v.get("name").and_then(|x| x.as_str()) {
                n += 1;
                calls.push(ToolCall {
                    id: format!("call_{n}"),
                    name: name.to_string(),
                    arguments: v.get("arguments").cloned().unwrap_or_else(|| json!({})),
                });
            }
        }
        search = &after[te + T_CLOSE.len()..];
    }
    if calls.is_empty() {
        (calls, text.to_string())
    } else {
        (calls, cut_before(text, T_OPEN))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen_xml_function() {
        let text = "I'll create it.\n<function=write_file>\n<parameter=path>\n/tmp/x.txt\n</parameter>\n<parameter=content>\nhello-gate\n</parameter>\n</function>\n</tool_call>";
        let (calls, cleaned) = parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].arguments["path"], "/tmp/x.txt");
        assert_eq!(calls[0].arguments["content"], "hello-gate");
        assert_eq!(
            cleaned, "I'll create it.",
            "call markup stripped, pre-text kept"
        );
    }

    #[test]
    fn parses_hermes_json_tool_call() {
        let text =
            "<tool_call>{\"name\":\"grep\",\"arguments\":{\"pattern\":\"fn main\"}}</tool_call>";
        let (calls, _) = parse(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
        assert_eq!(calls[0].arguments["pattern"], "fn main");
    }

    #[test]
    fn typed_args_parse_as_json() {
        let text = "<function=read_file>\n<parameter=path>\nsrc/lib.rs\n</parameter>\n<parameter=limit>\n50\n</parameter>\n</function>";
        let (calls, _) = parse(text);
        assert_eq!(
            calls[0].arguments["path"], "src/lib.rs",
            "path stays a string"
        );
        assert_eq!(
            calls[0].arguments["limit"], 50,
            "numeric arg parses to a number"
        );
    }

    #[test]
    fn plain_text_yields_no_calls() {
        let (calls, cleaned) = parse("just a normal answer, no tools");
        assert!(calls.is_empty());
        assert_eq!(cleaned, "just a normal answer, no tools");
    }

    #[test]
    fn tool_args_empty_is_noargs_malformed_surfaces() {
        assert_eq!(
            parse_tool_args(""),
            json!({}),
            "empty = legitimate no-arg call"
        );
        assert_eq!(
            parse_tool_args("  "),
            json!({}),
            "whitespace-only = no-arg call"
        );
        assert_eq!(
            parse_tool_args(r#"{"path":"a.rs"}"#)["path"],
            "a.rs",
            "valid JSON parses"
        );
        // Broken JSON must NOT silently become {} - it surfaces for a clear error.
        let m = parse_tool_args(r#"{"path": "#);
        assert!(
            m.get("_malformed_arguments").is_some(),
            "malformed args surface, not hidden as {{}}"
        );
    }
}
