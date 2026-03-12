//! INT8 symmetric abs-max quantization.
//!
//! Each vector is independently scaled by its abs-max value so every element
//! maps into [-127, 127].  This is the same scheme used by the Mojo SIMD
//! kernel (`quantizer_simd.mojo`) — full algorithm parity is required.
//!
//! Encoding:  q_i = round(v_i / abs_max * 127)  →  i8
//! Decoding:  v̂_i = q_i / 127.0 * abs_max
//!
//! The `rayon` parallel iterator handles per-vector row independence.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// One INT8-quantized vector, plus the per-vector abs-max scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Int8Vector {
    /// Quantized values in [-127, 127].
    pub codes: Vec<i8>,
    /// Scale factor = abs_max of the original f32 vector.
    pub scale: f32,
}

impl Int8Vector {
    /// Encode a single f32 slice to INT8.
    pub fn encode(v: &[f32]) -> Self {
        let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        let inv = 127.0 / scale;
        let codes: Vec<i8> = v.iter().map(|x| (x * inv).round().clamp(-127.0, 127.0) as i8).collect();
        Self { codes, scale }
    }

    /// Decode back to approximate f32.
    pub fn decode(&self) -> Vec<f32> {
        let factor = self.scale / 127.0;
        self.codes.iter().map(|&q| (q as f32) * factor).collect()
    }

    /// Dot product with an f32 query without full dequantization.
    /// Uses the scale factor to weight the result correctly.
    pub fn dot_query(&self, query_norm: &[f32]) -> f32 {
        let raw: f32 = self.codes.iter().zip(query_norm.iter()).map(|(&q, &qv)| (q as f32) * qv).sum();
        raw * (self.scale / 127.0)
    }
}

/// Encode a batch of f32 vectors to INT8 in parallel.
pub fn encode_batch(vectors: &[Vec<f32>]) -> Vec<Int8Vector> {
    vectors.par_iter().map(|v| Int8Vector::encode(v)).collect()
}

/// Decode a batch of INT8 vectors back to f32.
pub fn decode_batch(encoded: &[Int8Vector]) -> Vec<Vec<f32>> {
    encoded.par_iter().map(|e| e.decode()).collect()
}

/// Cosine similarity between an f32 query and an INT8-encoded vector.
///
/// Avoids full dequantization: uses the raw i8 dot product scaled by the
/// encoded vector's scale, combined with the pre-normed query.
pub fn cosine_int8(query: &[f32], encoded: &Int8Vector) -> f32 {
    if query.len() != encoded.codes.len() {
        return -1.0;
    }
    let q_norm_sq: f32 = query.iter().map(|x| x * x).sum();
    if q_norm_sq == 0.0 {
        return -1.0;
    }
    let q_norm = q_norm_sq.sqrt();

    // Dot of dequantized encoded vector and query
    let factor = encoded.scale / 127.0;
    let dot: f32 = encoded.codes.iter().zip(query.iter()).map(|(&q, &qv)| (q as f32) * factor * qv).sum();

    // Norm of dequantized encoded vector
    let enc_norm_sq: f32 = encoded.codes.iter().map(|&q| { let f = (q as f32) * factor; f * f }).sum();
    let enc_norm = enc_norm_sq.sqrt();

    let denom = enc_norm * q_norm;
    if denom == 0.0 { -1.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_reconstruct_quality() {
        let v: Vec<f32> = (0..768).map(|i| ((i as f32 * 0.01) - 3.84).sin()).collect();
        let enc = Int8Vector::encode(&v);
        let dec = enc.decode();
        assert_eq!(dec.len(), v.len());
        // Cosine similarity of original vs decoded must be >= 0.9999 (Mojo parity spec)
        let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
        let norm_v: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_d: f32 = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos = dot / (norm_v * norm_d);
        assert!(cos >= 0.9999, "cosine similarity {} < 0.9999", cos);
    }

    #[test]
    fn zero_vector() {
        let v = vec![0.0f32; 128];
        let enc = Int8Vector::encode(&v);
        assert_eq!(enc.scale, 1.0);
        assert!(enc.codes.iter().all(|&q| q == 0));
    }

    #[test]
    fn encoding_symmetry() {
        let v = vec![1.0f32, -1.0, 0.5, -0.5];
        let enc = Int8Vector::encode(&v);
        assert_eq!(enc.codes[0], 127);
        assert_eq!(enc.codes[1], -127);
    }

    #[test]
    fn batch_encode_decode_parity() {
        let vecs: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..64).map(|j| ((i + j) as f32 * 0.1).sin()).collect())
            .collect();
        let encoded = encode_batch(&vecs);
        let decoded = decode_batch(&encoded);
        assert_eq!(decoded.len(), 100);
        for (orig, dec) in vecs.iter().zip(decoded.iter()) {
            let dot: f32 = orig.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let n1: f32 = orig.iter().map(|x| x * x).sum::<f32>().sqrt();
            let n2: f32 = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n1 > 0.0 && n2 > 0.0 {
                assert!(dot / (n1 * n2) >= 0.9999);
            }
        }
    }

    #[test]
    fn cosine_int8_matches_float() {
        let v = vec![0.6f32, 0.8, -0.3, 0.1];
        let q = vec![0.5f32, 0.7, -0.2, 0.2];
        let enc = Int8Vector::encode(&v);

        let dot: f32 = v.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nq: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let float_cos = dot / (nv * nq);

        let int8_cos = cosine_int8(&q, &enc);
        // Should be within 1% of true cosine
        assert!((float_cos - int8_cos).abs() < 0.01, "float_cos={float_cos} int8_cos={int8_cos}");
    }
}
