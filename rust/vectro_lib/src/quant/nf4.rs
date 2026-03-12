//! NF4 (Normal Float 4-bit) quantization.
//!
//! Implements the NF4 encoding from Dettmers et al. 2023 ("QLoRA").
//! Each dimension is mapped to the nearest value in a 16-entry codebook whose
//! levels are the quantiles of the standard normal distribution, scaled per
//! vector by its abs-max.
//!
//! Storage: two 4-bit codes are packed into one u8 (low nibble = even dim,
//! high nibble = odd dim).  This gives exactly `ceil(d/2)` bytes per vector.
//!
//! Algorithm parity target (from PLAN.md Phase 16):
//!   cosine similarity of decode(encode(v)) vs v  ≥ 0.985

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// NF4 codebook — quantiles of N(0, 1), exactly reproducing the Python reference.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.6961928,
    -0.5250730,
    -0.3949003,
    -0.2844677,
    -0.1848745,
    -0.09105004,
    0.0,
    0.07958031,
    0.16093908,
    0.24611496,
    0.33791524,
    0.44070983,
    0.56266755,
    0.72295761,
    1.0,
];

/// Mid-points between adjacent NF4 levels, used for nearest-neighbour search.
const NF4_MIDS: [f32; 15] = {
    let mut m = [0.0f32; 15];
    let mut i = 0;
    while i < 15 {
        m[i] = (NF4_LEVELS[i] + NF4_LEVELS[i + 1]) * 0.5;
        i += 1;
    }
    m
};

/// Find the NF4 level index nearest to `x` via binary search on midpoints.
/// `x` must be in [-1, 1].
#[inline]
fn nearest_nf4(x: f32) -> u8 {
    // binary search: find first mid where x < mid → index before that
    let mut lo = 0usize;
    let mut hi = 15usize;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if x < NF4_MIDS[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo as u8
}

/// One NF4-quantized vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nf4Vector {
    /// Packed 4-bit codes: low nibble = even index, high nibble = odd index.
    pub packed: Vec<u8>,
    /// Per-vector abs-max scale factor.
    pub scale: f32,
    /// Original dimension (needed for decode when d is odd).
    pub dim: usize,
}

impl Nf4Vector {
    /// Encode a single f32 slice to packed NF4.
    pub fn encode(v: &[f32]) -> Self {
        let dim = v.len();
        let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        let inv = 1.0 / scale;

        let bytes_per_vec = (dim + 1) / 2;
        let mut packed = vec![0u8; bytes_per_vec];

        let mut i = 0;
        while i + 1 < dim {
            let lo = nearest_nf4((v[i] * inv).clamp(-1.0, 1.0));
            let hi = nearest_nf4((v[i + 1] * inv).clamp(-1.0, 1.0));
            packed[i / 2] = lo | (hi << 4);
            i += 2;
        }
        if dim % 2 == 1 {
            let lo = nearest_nf4((v[dim - 1] * inv).clamp(-1.0, 1.0));
            packed[bytes_per_vec - 1] = lo;
        }

        Self { packed, scale, dim }
    }

    /// Decode packed NF4 back to approximate f32.
    pub fn decode(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim);
        let mut i = 0;
        while i + 1 < self.dim {
            let byte = self.packed[i / 2];
            out.push(NF4_LEVELS[(byte & 0x0F) as usize] * self.scale);
            out.push(NF4_LEVELS[((byte >> 4) & 0x0F) as usize] * self.scale);
            i += 2;
        }
        if self.dim % 2 == 1 {
            let byte = self.packed[self.packed.len() - 1];
            out.push(NF4_LEVELS[(byte & 0x0F) as usize] * self.scale);
        }
        out
    }
}

/// Encode a batch of f32 vectors to NF4 in parallel.
pub fn encode_batch(vectors: &[Vec<f32>]) -> Vec<Nf4Vector> {
    vectors.par_iter().map(|v| Nf4Vector::encode(v)).collect()
}

/// Decode a batch of NF4 vectors back to f32 in parallel.
pub fn decode_batch(encoded: &[Nf4Vector]) -> Vec<Vec<f32>> {
    encoded.par_iter().map(|e| e.decode()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: cosine similarity between two equal-length slices.
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { return -1.0; }
        dot / (na * nb)
    }

    #[test]
    fn roundtrip_cosine_quality() {
        // d=768, normally-distributed values — matches the primary benchmark vector shape
        let v: Vec<f32> = (0..768).map(|i| ((i as f32 * 0.007) - 2.688).sin() * 0.8).collect();
        let enc = Nf4Vector::encode(&v);
        let dec = enc.decode();
        assert_eq!(dec.len(), 768);
        let cos = cosine(&v, &dec);
        assert!(cos >= 0.985, "cosine {cos} < 0.985 (NF4 parity spec)");
    }

    #[test]
    fn zero_vector() {
        let v = vec![0.0f32; 64];
        let enc = Nf4Vector::encode(&v);
        assert_eq!(enc.scale, 1.0);
        let dec = enc.decode();
        // All decoded values should be NF4_LEVELS[nearest_nf4(0)] * 1.0 = 0.0
        for x in &dec { assert!(x.abs() < 1e-5); }
    }

    #[test]
    fn odd_dimension() {
        let v = vec![0.3f32, -0.7, 0.5, -0.1, 0.9];
        let enc = Nf4Vector::encode(&v);
        let dec = enc.decode();
        assert_eq!(dec.len(), 5);
        let cos = cosine(&v, &dec);
        assert!(cos >= 0.98, "odd-dim cosine {cos}");
    }

    #[test]
    fn packing_round_trip_short() {
        // Two elements → 1 byte.
        let v = vec![1.0f32, -1.0];
        let enc = Nf4Vector::encode(&v);
        assert_eq!(enc.packed.len(), 1);
        let dec = enc.decode();
        // level[0] * scale ≈ -1, level[15] * scale ≈ +1
        assert!((dec[0] - 1.0).abs() < 1e-5, "got {}", dec[0]);
        assert!((dec[1] + 1.0).abs() < 1e-5, "got {}", dec[1]);
    }

    #[test]
    fn batch_encode_decode() {
        let vecs: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..128).map(|j| ((i * j) as f32 * 0.01).sin()).collect())
            .collect();
        let encoded = encode_batch(&vecs);
        let decoded = decode_batch(&encoded);
        for (orig, dec) in vecs.iter().zip(decoded.iter()) {
            let cos = cosine(orig, dec);
            if orig.iter().any(|x| *x != 0.0) {
                assert!(cos >= 0.985, "batch cosine {cos} < 0.985");
            }
        }
    }

    #[test]
    fn binary_search_midpoints() {
        // nearest_nf4(0) should return index 7 (NF4_LEVELS[7] = 0.0)
        assert_eq!(nearest_nf4(0.0), 7);
        // nearest_nf4(-1.0) → index 0
        assert_eq!(nearest_nf4(-1.0), 0);
        // nearest_nf4(1.0) → index 15
        assert_eq!(nearest_nf4(1.0), 15);
    }
}
