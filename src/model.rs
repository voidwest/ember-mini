//! GPT-2: the transformer that changes one token into a score for each
//! possible next token.
//!
//! One forward pass for one token:
//!
//! ```text
//! x = token_embedding[token] + position_embedding[pos]
//! for each block (12 in GPT-2 small):
//!     x = x + attention(layer_norm(x))     // look at earlier tokens
//!     x = x + mlp(layer_norm(x))           // process this token alone
//! logits = output_head(layer_norm(x))
//! ```

use crate::kv_cache::KvCache;
use crate::loader::Gguf;
use crate::ops;
use crate::Result;

pub struct Config {
    pub n_layers: usize,
    pub n_heads: usize,
    /// The width of the vector for each token (768 for GPT-2 small).
    pub d_model: usize,
    /// The maximum number of tokens that the model can see (1024).
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub eps: f32,
}

/// A dense layer: `y = W x + b`.
struct Linear {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    n_out: usize,
}

impl Linear {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; self.n_out];
        ops::matvec(&self.weight, self.bias.as_deref(), x, &mut out);
        out
    }
}

struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

/// One transformer block.
struct Block {
    ln_1: LayerNorm,
    /// Makes the query, key and value vectors in one matmul.
    attn_qkv: Linear,
    attn_out: Linear,
    ln_2: LayerNorm,
    mlp_up: Linear,
    mlp_down: Linear,
}

pub struct Gpt2 {
    pub config: Config,
    /// Token embeddings, `[n_vocab, d_model]`.
    wte: Vec<f32>,
    /// Position embeddings, `[n_ctx, d_model]`.
    wpe: Vec<f32>,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head: Linear,
}

impl Gpt2 {
    /// Builds the model from the tensors in a GGUF file. The tensor names are
    /// the names that llama.cpp gives to the GPT-2 weights.
    pub fn from_gguf(mut gguf: Gguf) -> Result<Self> {
        if gguf.get_str("general.architecture") != Some("gpt2") {
            return Err("this engine runs only GPT-2 GGUF files".into());
        }
        let d = gguf.get_usize("gpt2.embedding_length")?;
        let n_vocab = match gguf.tensors.get("token_embd.weight") {
            Some(t) => t.shape[0],
            None => return Err("missing tensor token_embd.weight".into()),
        };
        let config = Config {
            n_layers: gguf.get_usize("gpt2.block_count")?,
            n_heads: gguf.get_usize("gpt2.attention.head_count")?,
            d_model: d,
            n_ctx: gguf.get_usize("gpt2.context_length")?,
            n_vocab,
            eps: gguf
                .get_f32("gpt2.attention.layer_norm_epsilon")
                .unwrap_or(1e-5),
        };
        if d % config.n_heads != 0 {
            return Err("the embedding width must divide into whole heads".into());
        }

        let mut blocks = Vec::new();
        for i in 0..config.n_layers {
            let p = format!("blk.{i}");
            blocks.push(Block {
                ln_1: take_norm(&mut gguf, &format!("{p}.attn_norm"), d)?,
                attn_qkv: take_linear(&mut gguf, &format!("{p}.attn_qkv"), d, 3 * d)?,
                attn_out: take_linear(&mut gguf, &format!("{p}.attn_output"), d, d)?,
                ln_2: take_norm(&mut gguf, &format!("{p}.ffn_norm"), d)?,
                mlp_up: take_linear(&mut gguf, &format!("{p}.ffn_up"), d, 4 * d)?,
                mlp_down: take_linear(&mut gguf, &format!("{p}.ffn_down"), 4 * d, d)?,
            });
        }

        let wte = take(&mut gguf, "token_embd.weight", &[n_vocab, d])?;
        // GPT-2 uses the token embedding matrix again as the output layer.
        // Some GGUF files also keep a copy of it as `output.weight`.
        let head_weight = if gguf.tensors.contains_key("output.weight") {
            take(&mut gguf, "output.weight", &[n_vocab, d])?
        } else {
            wte.clone()
        };

        Ok(Self {
            wpe: take(&mut gguf, "position_embd.weight", &[config.n_ctx, d])?,
            ln_f: take_norm(&mut gguf, "output_norm", d)?,
            head: Linear {
                weight: head_weight,
                bias: None,
                n_out: n_vocab,
            },
            wte,
            blocks,
            config,
        })
    }

    /// Runs one token at position `pos` through the model. It adds the key
    /// and value of this token to `cache`, and reads the earlier ones from it.
    /// The result has one score (logit) for each token in the vocabulary.
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let c = &self.config;
        let d = c.d_model;
        let head_dim = d / c.n_heads;
        let tok = token as usize;

        // 1. Embedding: the vector of the token plus the vector of its position.
        let mut x: Vec<f32> = (0..d)
            .map(|i| self.wte[tok * d + i] + self.wpe[pos * d + i])
            .collect();

        let mut h = vec![0.0; d];
        for (layer, block) in self.blocks.iter().enumerate() {
            // 2. Attention. Each head compares the query of this token with
            // the keys of all tokens so far (this token included). The
            // softmax of the scores gives the weight for each value vector.
            // There are no later tokens yet, so no mask is necessary.
            ops::layer_norm(&x, &block.ln_1.weight, &block.ln_1.bias, c.eps, &mut h);
            let qkv = block.attn_qkv.forward(&h);
            let (q, kv) = qkv.split_at(d);
            let (k, v) = kv.split_at(d);
            cache.store(layer, pos, k, v);

            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn = vec![0.0; d];
            for head in 0..c.n_heads {
                let (lo, hi) = (head * head_dim, (head + 1) * head_dim);
                let mut scores: Vec<f32> = (0..=pos)
                    .map(|t| ops::dot(&q[lo..hi], &cache.key(layer, t)[lo..hi]) * scale)
                    .collect();
                ops::softmax(&mut scores);
                for (t, &weight) in scores.iter().enumerate() {
                    let value = &cache.value(layer, t)[lo..hi];
                    for (out, &val) in attn[lo..hi].iter_mut().zip(value) {
                        *out += weight * val;
                    }
                }
            }
            ops::add(&mut x, &block.attn_out.forward(&attn));

            // 3. MLP: widen to 4 * d, apply GELU, narrow back to d.
            ops::layer_norm(&x, &block.ln_2.weight, &block.ln_2.bias, c.eps, &mut h);
            let mut up = block.mlp_up.forward(&h);
            ops::gelu(&mut up);
            ops::add(&mut x, &block.mlp_down.forward(&up));
        }

        // 4. The final norm, then one score for each token in the vocabulary.
        ops::layer_norm(&x, &self.ln_f.weight, &self.ln_f.bias, c.eps, &mut h);
        self.head.forward(&h)
    }
}

/// Removes one tensor from the file and checks its shape.
fn take(gguf: &mut Gguf, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let t = gguf
        .tensors
        .remove(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if t.shape != shape {
        return Err(format!("{name}: shape {:?}, expected {shape:?}", t.shape).into());
    }
    Ok(t.data)
}

fn take_linear(gguf: &mut Gguf, name: &str, n_in: usize, n_out: usize) -> Result<Linear> {
    Ok(Linear {
        weight: take(gguf, &format!("{name}.weight"), &[n_out, n_in])?,
        bias: Some(take(gguf, &format!("{name}.bias"), &[n_out])?),
        n_out,
    })
}

fn take_norm(gguf: &mut Gguf, name: &str, d: usize) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: take(gguf, &format!("{name}.weight"), &[d])?,
        bias: take(gguf, &format!("{name}.bias"), &[d])?,
    })
}
