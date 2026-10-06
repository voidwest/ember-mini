//! ember-mini: a minimal GPT-2 inference engine.
//!
//! ```text
//! text --tokenizer--> token ids --model--> logits --argmax--> next token id
//!                                   ^                                |
//!                                   +--------------------------------+
//! ```
//!
//! Usage: `cargo run --release -- "your prompt" [number of new tokens]`

mod kv_cache;
mod loader;
mod model;
mod ops;
mod quant;

use std::io::Write;
use std::time::Instant;
use tokenizers::Tokenizer;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

const MODEL_PATH: &str = "gpt2.Q8_0.gguf";
const TOKENIZER_PATH: &str = "tokenizer.json";
/// GPT-2 uses `<|endoftext|>` both to end a text and to start one.
const END_OF_TEXT: u32 = 50256;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let prompt = args
        .next()
        .unwrap_or_else(|| "The meaning of life is".to_string());
    let n_new: usize = match args.next() {
        Some(s) => s.parse().map_err(|_| format!("not a token count: {s}"))?,
        None => 40,
    };

    // 1. Load the weights and the tokenizer.
    let start = Instant::now();
    let model = model::Gpt2::from_gguf(loader::load(MODEL_PATH)?)?;
    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| format!("cannot load {TOKENIZER_PATH}: {e}"))?;
    let cfg = &model.config;
    eprintln!(
        "loaded GPT-2: {} layers, {} heads, width {}, {} tokens, in {:.2} s",
        cfg.n_layers,
        cfg.n_heads,
        cfg.d_model,
        cfg.n_vocab,
        start.elapsed().as_secs_f32()
    );

    // 2. Change the text into token ids.
    let mut tokens = tokenizer.encode(prompt.as_str(), false)?.get_ids().to_vec();
    if tokens.is_empty() {
        tokens.push(END_OF_TEXT);
    }
    if tokens.len() > cfg.n_ctx {
        return Err(format!("the prompt has more than {} tokens", cfg.n_ctx).into());
    }
    let n_prompt = tokens.len();

    // 3. Prefill: run all prompt tokens except the last one. This only fills
    // the KV cache. The generation loop below runs the last prompt token.
    let mut cache = kv_cache::KvCache::new(cfg.n_layers, cfg.n_ctx, cfg.d_model);
    let start = Instant::now();
    let mut n_forward = 0;
    for (pos, &token) in tokens[..n_prompt - 1].iter().enumerate() {
        model.forward(token, pos, &mut cache);
        n_forward += 1;
    }

    // 4. Generation: run the newest token, take the token with the highest
    // score (greedy decoding), add it to the text, and do it again.
    print!("{prompt}");
    let mut printed = 0;
    for _ in 0..n_new {
        let pos = tokens.len() - 1;
        if pos >= cfg.n_ctx {
            break; // no position embedding is left
        }
        let logits = model.forward(tokens[pos], pos, &mut cache);
        n_forward += 1;
        let next = ops::argmax(&logits) as u32;
        if next == END_OF_TEXT {
            break;
        }
        tokens.push(next);

        // Decode all new tokens again each time: one character (for example
        // an emoji) can be split across two tokens. A partial character
        // decodes as U+FFFD, so wait for the next token in that case.
        let text = tokenizer.decode(&tokens[n_prompt..], false)?;
        if !text.ends_with('\u{FFFD}') {
            print!("{}", &text[printed..]);
            std::io::stdout().flush()?;
            printed = text.len();
        }
    }
    println!();

    eprintln!(
        "{} prompt tokens, {} new tokens, {:.1} forward passes/s",
        n_prompt,
        tokens.len() - n_prompt,
        n_forward as f32 / start.elapsed().as_secs_f32()
    );
    Ok(())
}
