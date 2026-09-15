//! Live check: send one request to a real OpenAI-compatible endpoint (default: a
//! local server on `localhost:11434`) and print the reply. Also exercises streaming.
//! Point it anywhere with OXIO_BASE / OXIO_MODEL.
//!
//!   cargo run -p providers --example probe
//!   OXIO_BASE=http://localhost:11434/v1 OXIO_MODEL=llama3.1 cargo run -p providers --example probe

use futures::StreamExt;
use oxio_core::{Ctx, Message, Params, Provider, Request, Role, StreamEvent};
use providers::ChatCompletionsAdapter;

#[tokio::main]
async fn main() {
    let base = std::env::var("OXIO_BASE").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let model = std::env::var("OXIO_MODEL").unwrap_or_else(|_| "llama3.2".into());
    eprintln!("probing {base} (model {model})");

    let p = ChatCompletionsAdapter::new("local", base, model, None);
    let req = Request {
        model: String::new(),
        messages: vec![Message::text(
            Role::User,
            "Reply with exactly one word: pong",
        )],
        tools: vec![],
        params: Params::default(),
    };

    // 1) non-streaming complete()
    match p.complete(req.clone(), &Ctx::default()).await {
        Ok(r) => println!(
            "complete() -> {:?} (usage {:?})",
            r.message.as_text(),
            r.usage
        ),
        Err(e) => {
            eprintln!("complete() ERROR: {e}");
            std::process::exit(1);
        }
    }

    // 2) streaming stream()
    match p.stream(req, &Ctx::default()).await {
        Ok(mut s) => {
            print!("stream() -> ");
            while let Some(ev) = s.next().await {
                match ev {
                    Ok(StreamEvent::ThinkingDelta(t)) => print!("\x1b[2m{t}\x1b[0m"),
                    Ok(StreamEvent::TextDelta(t)) => print!("{t}"),
                    Ok(StreamEvent::Done { usage, .. }) => println!("  [done, usage {usage:?}]"),
                    Ok(StreamEvent::ToolCallDelta { name, .. }) => print!("<tool:{name}>"),
                    Ok(StreamEvent::Notice { text, .. }) => print!("\x1b[2m{text}\x1b[0m"),
                    Err(e) => {
                        eprintln!("stream() ERROR: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("stream() ERROR: {e}");
            std::process::exit(1);
        }
    }
}
