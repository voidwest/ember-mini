//! ember-mini: a minimal GPT-2 inference engine.
//!
//! ```text
//! text --tokenizer--> token ids --model--> logits --argmax--> next token id
//!                                   ^                                |
//!                                   +--------------------------------+
//! ```
//!
//! Usage: `cargo run -- "your prompt" [number of new tokens]`
//!
//! EXERCISE 6. Write `generate`. Then run the program (see the README).

// The exercises start as empty functions, so the compiler warns about unused
// code. Remove this line when you finish all exercises.
#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

mod kv_cache;
mod loader;
mod model;
mod ops;
mod quant;

use kv_cache::KvCache;
use model::Gpt2;
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
    let model = Gpt2::from_gguf(loader::load(MODEL_PATH)?)?;
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

    // 3 and 4. Run the prompt, then generate new tokens.
    print!("{prompt}");
    let start = Instant::now();
    let n_forward = generate(&model, &tokenizer, &mut tokens, n_new)?;
    println!();

    eprintln!(
        "{} prompt tokens, {} new tokens, {:.1} forward passes/s",
        n_prompt,
        tokens.len() - n_prompt,
        n_forward as f32 / start.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Runs the prompt in `tokens` through the model. Then adds up to `n_new`
/// tokens to `tokens`, one at a time, and prints the new text.
/// Returns the number of forward passes.
fn generate(
    model: &Gpt2,
    tokenizer: &Tokenizer,
    tokens: &mut Vec<u32>,
    n_new: usize,
) -> Result<usize> {
    let cfg = &model.config;
    let n_prompt = tokens.len();
    let mut cache = KvCache::new(cfg.n_layers, cfg.n_ctx, cfg.d_model);
    let mut n_forward = 0;
    let mut printed = 0;

    // TODO 1. Prefill. Run the prompt tokens through `model.forward`, at
    // positions 0, 1, 2 and so on. This fills the KV cache.
    // Hint: only the logits of the last prompt token are necessary. You can
    // keep them, or you can run all prompt tokens except the last one here
    // and let the loop below run the last one.

    // TODO 2. Generation. Do these steps up to `n_new` times:
    //   a. Take the token with the highest logit (`ops::argmax`). This is
    //      greedy decoding.
    //   b. If the token is END_OF_TEXT, stop.
    //   c. Push it to `tokens`. Print the new text with
    //      `print_new_text(tokenizer, &tokens[n_prompt..], &mut printed)?`.
    //   d. Run the new token through `model.forward` at its position, to get
    //      the logits for the next step.
    //
    // Hint: GPT-2 has position embeddings for `cfg.n_ctx` positions only.
    // Do not run `model.forward` at position `cfg.n_ctx` or higher.
    // Add 1 to `n_forward` for each call to `model.forward`, and return it.
    todo!("generate")
}

/// Prints the part of the text of `new_tokens` that is not printed yet.
/// `printed` is the number of bytes that are printed already.
///
/// It decodes all new tokens again each time, because one character (for
/// example an emoji) can be split across two tokens. A partial character
/// decodes as U+FFFD, so the function waits for the next token in that case.
fn print_new_text(tokenizer: &Tokenizer, new_tokens: &[u32], printed: &mut usize) -> Result<()> {
    let text = tokenizer.decode(new_tokens, false)?;
    if !text.ends_with('\u{FFFD}') {
        print!("{}", &text[*printed..]);
        std::io::stdout().flush()?;
        *printed = text.len();
    }
    Ok(())
}
