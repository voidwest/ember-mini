# ember-mini

> [!NOTE]
> **This is an educational repository.** It is not for production use.
> On this branch, the important parts of the engine are missing: you write
> them yourself. Each missing part has a `todo!()` and hints in the comments,
> and tests tell you when it works. The complete code is on the
> [`solution`](https://github.com/voidwest/ember-mini/tree/solution) branch.

Build a GPT-2 inference engine in Rust, from the bytes of the model file to
the generated text. The finished engine has about 700 lines. It runs on one
CPU thread and it has no dependencies for the math.

It comes from the
[first version](https://github.com/voidwest/ember/tree/3244f2d3437eea14bc54700b418d35b9b5f219f6)
of [ember](https://github.com/voidwest/ember) (May 2026). ember later became a
large research tool. This project keeps only the core.

## What you build

```text
"Hello"  --tokenizer-->  [15496]  --model-->  50257 scores  --argmax-->  next token
                                    ^                                       |
                                    +---------------------------------------+
```

1. The tokenizer changes the text into token ids. (This part is done: it
   uses the `tokenizers` crate.)
2. The model runs one token at a time. For each token, it gives a score
   (a logit) for each of the 50,257 tokens in the vocabulary.
3. The engine selects the token with the highest score (greedy decoding).
4. The engine gives that token back to the model, and the loop continues.

The KV cache keeps the keys and values of the earlier tokens. Thus, each step
calculates only the newest token.

## Before you start

1. Install Rust 1.88 or later from [rustup.rs](https://rustup.rs).

2. Download the model (178 MB) into the project folder. Exercises 5 and 6
   need it:

   ```bash
   curl -L -o gpt2.Q8_0.gguf https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/7eae6f079f0164bff66b86eea5159f7a368f9381/gpt2.Q8_0.gguf
   ```

3. Make sure that the SHA-256 is
   `0afd77d848df0488b1cd347851d5dde8ffa745826be2b5b3c364721c1b26f5d9`:

   ```bash
   shasum -a 256 gpt2.Q8_0.gguf
   ```

`tokenizer.json` is the GPT-2 tokenizer. It is in the project folder already.

## The exercises

Do the exercises in this order. Each one uses the ones before it.

| # | File | What you write | Check |
|---|---|---|---|
| 1 | `src/ops.rs` | The math: dot product, matrix-vector product, layer norm, GELU, softmax, argmax | `cargo test ops` |
| 2 | `src/quant.rs` | Bytes to f32, and Q8_0 blocks to f32 | `cargo test quant` |
| 3 | `src/kv_cache.rs` | Store and read the key and value vectors | `cargo test kv_cache` |
| 4 | `src/loader.rs` | The GGUF reader: header, metadata, tensor infos, tensor data | `cargo test loader` |
| 5 | `src/model.rs` | The GPT-2 forward pass: embedding, attention, MLP, output | `cargo test model` |
| 6 | `src/main.rs` | The generation loop: prefill, then greedy decoding | `cargo run` (below) |

For exercise 6, run:

```bash
cargo run -- "The capital of France is" 20
```

If all of your code is correct, the output is:

```text
The capital of France is the capital of France, and it is the capital of France. The French people are the most powerful
```

### Tips

- At the start, all tests fail with "not yet implemented". This is
  normal. Each test passes when you write the code that it checks.
- To find all missing parts, search for `todo!(`.
- Read the structs and the finished functions near each `todo!()`. They show
  the data layout. For example, `Gpt2::from_gguf` shows the shape of each
  weight.
- When you finish, remove the `#![allow(...)]` line at the top of
  `src/main.rs`. The compiler then shows the code that you do not use.
- If you are blocked, look at the solution for one file. This shows the
  answer, so use it only when necessary:

  ```bash
  git show origin/solution:src/ops.rs
  ```

## After the exercises

These are good next steps. Each one is in the full ember.

- **Speed.** All math is on one thread, with no SIMD. Almost all of the time
  goes into `ops::dot`. Try 8 separate sums in `dot`, or threads in `matvec`.
- **Quantized math.** The loader changes all weights into f32, which uses 4x
  more memory. Real engines keep Q8_0 and use it directly in the dot product.
- **Batched prefill.** The prompt goes through the model one token at a time.
  A batch of all prompt tokens uses matrix-matrix products, which are faster.
- **Sampling.** Only greedy decoding is available. Add temperature and top-k.
- **Other models.** Only GPT-2 is available. Llama adds RoPE, RMSNorm,
  SwiGLU and grouped-query attention.

## Check against llama.cpp

The solution gives the same text as llama.cpp. To compare, give llama.cpp the
same weights in F32
(`llama-quantize --allow-requantize gpt2.Q8_0.gguf gpt2.f32.gguf F32`) and
use greedy decoding (`--temp 0`). With these settings, the two engines gave
the same 100 tokens for four prompts.

With the Q8_0 file, llama.cpp also changes the activations into 8-bit values.
Its results are thus a little different, and greedy text can diverge after a
few tokens.

## License

The code is under the MIT license (see `LICENSE`). `tokenizer.json` is from
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2). See
that page for its license.
