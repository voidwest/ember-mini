//! The KV cache keeps the key and value vectors of all earlier tokens.
//!
//! Attention compares the newest token with all earlier tokens. Without a
//! cache, the model must calculate the keys and values of the full text again
//! for each new token. With a cache, the model calculates only the key and
//! value of the newest token, and reads the earlier ones from here.

pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    n_ctx: usize,
    d_model: usize,
}

impl KvCache {
    /// Memory layout: `[layer][position][d_model]`. All heads of one position
    /// are next to each other, in the same order as in the model.
    pub fn new(n_layers: usize, n_ctx: usize, d_model: usize) -> Self {
        let len = n_layers * n_ctx * d_model;
        Self {
            k: vec![0.0; len],
            v: vec![0.0; len],
            n_ctx,
            d_model,
        }
    }

    fn start(&self, layer: usize, pos: usize) -> usize {
        (layer * self.n_ctx + pos) * self.d_model
    }

    pub fn store(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let s = self.start(layer, pos);
        self.k[s..s + self.d_model].copy_from_slice(k);
        self.v[s..s + self.d_model].copy_from_slice(v);
    }

    pub fn key(&self, layer: usize, pos: usize) -> &[f32] {
        let s = self.start(layer, pos);
        &self.k[s..s + self.d_model]
    }

    pub fn value(&self, layer: usize, pos: usize) -> &[f32] {
        let s = self.start(layer, pos);
        &self.v[s..s + self.d_model]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_back_what_it_stores() {
        let mut cache = KvCache::new(2, 4, 3);
        cache.store(1, 2, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert_eq!(cache.key(1, 2), &[1.0, 2.0, 3.0]);
        assert_eq!(cache.value(1, 2), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn layers_and_positions_do_not_overlap() {
        let mut cache = KvCache::new(2, 4, 3);
        cache.store(0, 3, &[1.0; 3], &[1.0; 3]);
        cache.store(1, 0, &[2.0; 3], &[2.0; 3]);
        assert_eq!(cache.key(0, 3), &[1.0; 3]);
        assert_eq!(cache.key(1, 0), &[2.0; 3]);
        assert_eq!(cache.key(0, 2), &[0.0; 3]);
        assert_eq!(cache.value(1, 1), &[0.0; 3]);
    }
}
