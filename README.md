# ember-mini

> [!NOTE]
> **This is an educational repository.** It shows how an LLM inference engine
> works, in code that you can read from start to end. It is not for production
> use. Each part is as simple as possible, so it is slow and it runs only
> GPT-2.

A minimal GPT-2 inference engine in about 700 lines of Rust. It has only the
parts that are necessary to run GPT-2 on a CPU.

It comes from the
[first version](https://github.com/voidwest/ember/tree/3244f2d3437eea14bc54700b418d35b9b5f219f6)
of [ember](https://github.com/voidwest/ember) (May 2026). ember later became a
large research tool. This project keeps only the core.

## Run it

1. Download the model (178 MB) into this folder:

   ```bash
   curl -L -o gpt2.Q8_0.gguf https://huggingface.co/QuantFactory/gpt2-GGUF/resolve/7eae6f079f0164bff66b86eea5159f7a368f9381/gpt2.Q8_0.gguf
   ```

2. Make sure that the SHA-256 is
   `0afd77d848df0488b1cd347851d5dde8ffa745826be2b5b3c364721c1b26f5d9`:

   ```bash
   shasum -a 256 gpt2.Q8_0.gguf
   ```

3. Run a prompt. The second argument is the number of new tokens (default 40):

   ```bash
   cargo run --release -- "The capital of France is" 40
   ```

`tokenizer.json` is the GPT-2 tokenizer from
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2). It is
in this folder already.

## How it works

```text
"Hello"  --tokenizer-->  [15496]  --model-->  50257 scores  --argmax-->  next token
                                    ^                                       |
                                    +---------------------------------------+
```

1. The tokenizer changes the text into token ids.
2. The model runs one token at a time. For each token, it gives a score
   (a logit) for each of the 50,257 tokens in the vocabulary.
3. The engine selects the token with the highest score (greedy decoding).
4. The engine gives that token back to the model, and the loop continues.

The KV cache keeps the keys and values of the earlier tokens. Thus, each step
calculates only the newest token.

## Read the code in this order

| File | What it does |
|---|---|
| `src/main.rs` | The full loop: load, tokenize, prefill, generate, print. |
| `src/loader.rs` | Reads the GGUF file: header, metadata, tensor infos, tensor data. |
| `src/quant.rs` | Changes Q8_0 and F16 weights into f32. |
| `src/model.rs` | GPT-2: embeddings, 12 transformer blocks, the output head. |
| `src/kv_cache.rs` | Keeps the key and value vectors of the earlier tokens. |
| `src/ops.rs` | The math: matrix-vector product, layer norm, GELU, softmax. |

## What this version does not have

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

To compare with llama.cpp, give it the same weights in F32
(`llama-quantize --allow-requantize gpt2.Q8_0.gguf gpt2.f32.gguf F32`) and
use greedy decoding (`--temp 0`). With these settings, the two engines
gave the same 100 tokens for four prompts.

With the Q8_0 file, llama.cpp also changes the activations into 8-bit values.
Its results are thus a little different, and greedy text can diverge after a
few tokens.

## License

The code is under the MIT license (see `LICENSE`). `tokenizer.json` is from
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2). See
that page for its license.
