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
    /// Encode a single f32 slice to INT8 (portable scalar path).
    pub fn encode(v: &[f32]) -> Self {
        let abs_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
        let inv = 127.0 / scale;
        let codes: Vec<i8> = v.iter().map(|x| (x * inv).round().clamp(-127.0, 127.0) as i8).collect();
        Self { codes, scale }
    }

    /// SIMD-accelerated encode.
    ///
    /// Dispatch priority:
    ///  1. AArch64 — NEON (compile-time; mandated by ARMv8).
    ///  2. x86-64 + AVX2 — AVX2 path via runtime `is_x86_feature_detected!`.
    ///  3. All other targets — portable scalar `encode`.
    #[inline]
    pub fn encode_fast(v: &[f32]) -> Self {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: AArch64-v8 mandates NEON; no runtime feature detection needed.
        return unsafe { encode_neon(v) };

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 feature detection.
            return unsafe { encode_avx2(v) };
        }

        #[cfg(not(target_arch = "aarch64"))]
        return Self::encode(v);
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

/// AVX2-vectorised INT8 encode for x86-64.
///
/// Two passes over `v`:
///  1. AVX2 abs-max reduction (8-wide float, then horizontal reduce).
///  2. Multiply-round-narrow loop: float32x8 → int32x8 → pack to int16x8
///     → pack to int8 (low 8 bytes), stored with `_mm_storel_epi64`.
///
/// Processes 8 elements per iteration; scalar tail for remainder.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn encode_avx2(v: &[f32]) -> Int8Vector {
    use std::arch::x86_64::*;

    let n = v.len();
    if n == 0 {
        return Int8Vector { codes: vec![], scale: 1.0 };
    }
    let ptr = v.as_ptr();

    // ── Pass 1: abs-max reduction (8 floats per iteration) ──────────────────
    let sign_mask = _mm256_set1_ps(-0.0_f32); // 0x8000_0000 in every lane
    let mut vmax256 = _mm256_setzero_ps();
    let chunks8 = n / 8;
    for i in 0..chunks8 {
        let a = _mm256_loadu_ps(ptr.add(i * 8));
        let abs_a = _mm256_andnot_ps(sign_mask, a); // clear sign bit = abs(a)
        vmax256 = _mm256_max_ps(vmax256, abs_a);
    }
    // Reduce 8 lanes → 4 lanes
    let hi128 = _mm256_extractf128_ps(vmax256, 1);
    let lo128 = _mm256_castps256_ps128(vmax256);
    let max128 = _mm_max_ps(hi128, lo128);
    // Reduce 4 lanes → 1 scalar
    let m2 = _mm_movehl_ps(max128, max128);     // [max128[2], max128[3], …]
    let m3 = _mm_max_ps(max128, m2);             // [max(0,2), max(1,3), …]
    let m4 = _mm_shuffle_ps(m3, m3, 0x55);      // broadcast index-1 element
    let m5 = _mm_max_ps(m3, m4);                 // [max(0,1,2,3), …]
    let mut abs_max = _mm_cvtss_f32(m5);
    // Scalar tail
    for &x in &v[chunks8 * 8..] {
        let ax = x.abs();
        if ax > abs_max {
            abs_max = ax;
        }
    }

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max };
    let inv   = 127.0_f32 / scale;
    let vinv  = _mm256_set1_ps(inv);

    // ── Pass 2: quantise f32 → i8 (8 per iteration) ──────────────────
    let mut codes  = vec![0i8; n];
    let out_ptr = codes.as_mut_ptr();

    for i in 0..chunks8 {
        let base = i * 8;
        let x = _mm256_loadu_ps(ptr.add(base));
        // Round-to-nearest (current MXCSR mode; default = nearest-even).
        let i32s = _mm256_cvtps_epi32(_mm256_mul_ps(x, vinv));
        // Extract low and high 128-bit halves as integer registers (AVX2).
        let lo   = _mm256_castsi256_si128(i32s);        // low  4 × i32
        let hi   = _mm256_extracti128_si256(i32s, 1);   // high 4 × i32
        let i16s = _mm_packs_epi32(lo, hi);              // 8 × i16, saturating
        let i8s  = _mm_packs_epi16(i16s, i16s);          // 16 × i8 (low 8 valid)
        // Store low 8 bytes (= our 8 quantised values) without alignment req.
        _mm_storel_epi64(out_ptr.add(base) as *mut __m128i, i8s);
    }
    // Scalar tail
    for i in chunks8 * 8..n {
        *out_ptr.add(i) = (v[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }

    Int8Vector { codes, scale }
}

/// NEON-vectorised INT8 encode for AArch64.
///
/// Two passes over `v`:
///  1. NEON abs-max reduction (4-wide, then horizontal reduce).
///  2. Multiply-round-narrow loop storing 16 i8 values per iteration via four
///     float32x4_t registers → int32x4_t → int16x8_t → int8x16_t.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn encode_neon(v: &[f32]) -> Int8Vector {
    use std::arch::aarch64::*;

    let n = v.len();
    if n == 0 {
        return Int8Vector { codes: vec![], scale: 1.0 };
    }
    let ptr = v.as_ptr();

    // ── Pass 1: NEON abs-max ────────────────────────────────────────────────
    let mut vmax = vdupq_n_f32(0.0_f32);
    let chunks4 = n / 4;
    for i in 0..chunks4 {
        let a = vld1q_f32(ptr.add(i * 4));
        vmax = vmaxq_f32(vmax, vabsq_f32(a));
    }
    let mut abs_max = vmaxvq_f32(vmax); // horizontal reduce over 4 lanes
    for &x in &v[chunks4 * 4..] {
        let ax = x.abs();
        if ax > abs_max {
            abs_max = ax;
        }
    }

    let scale = if abs_max == 0.0 { 1.0_f32 } else { abs_max };
    let inv = 127.0_f32 / scale;
    let vinv = vdupq_n_f32(inv);

    // ── Pass 2: quantise f32 → i8 ───────────────────────────────────────────
    // 16 elements per iteration: 4 × float32x4_t → int32x4_t → int16x8_t → int8x16_t
    let mut codes = vec![0i8; n];
    let out_ptr = codes.as_mut_ptr();
    let chunks16 = n / 16;

    for i in 0..chunks16 {
        let base = i * 16;
        // multiply then round-to-nearest (exact on already-integer floats after conversion)
        let r0 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base     )), vinv));
        let r1 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  4)), vinv));
        let r2 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base +  8)), vinv));
        let r3 = vrndnq_f32(vmulq_f32(vld1q_f32(ptr.add(base + 12)), vinv));
        // f32x4 → i32x4 (truncation of already-rounded ints is exact)
        let i0 = vcvtq_s32_f32(r0);
        let i1 = vcvtq_s32_f32(r1);
        let i2 = vcvtq_s32_f32(r2);
        let i3 = vcvtq_s32_f32(r3);
        // i32x4 → i16x4: values in [-127, 127] so no overflow
        let s01 = vcombine_s16(vmovn_s32(i0), vmovn_s32(i1));
        let s23 = vcombine_s16(vmovn_s32(i2), vmovn_s32(i3));
        // i16x8 → i8x8 with saturation (defensive; values already in range)
        let b0 = vqmovn_s16(s01);
        let b1 = vqmovn_s16(s23);
        // store 16 bytes
        vst1q_s8(out_ptr.add(base), vcombine_s8(b0, b1));
    }

    // scalar tail for the remainder (< 16 elements)
    for i in chunks16 * 16..n {
        *out_ptr.add(i) = (v[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }

    Int8Vector { codes, scale }
}

/// Encode a batch of f32 vectors to INT8 in parallel, using SIMD where available.
pub fn encode_batch(vectors: &[Vec<f32>]) -> Vec<Int8Vector> {
    vectors.par_iter().map(|v| Int8Vector::encode_fast(v)).collect()
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

/// Batch encode an N×D f32 matrix (flat row-major) to INT8 without any per-row
/// heap allocation.
///
/// # Arguments
/// * `input`      — flat f32 slice, length = `n * d`
/// * `n`, `d`     — number of vectors and dimension
/// * `codes_out`  — caller-allocated i8 slice, length = `n * d` (written in-place)
/// * `scales_out` — caller-allocated f32 slice, length = `n`  
///                  stores `abs_max / 127.0` per row (direct dequant factor)
///
/// Uses rayon for row-parallel execution; inner quantisation loop is
/// auto-vectorised by LLVM (NEON on AArch64, AVX2 on x86-64).
pub fn batch_encode_into(
    input: &[f32],
    _n: usize,
    d: usize,
    codes_out: &mut [i8],
    scales_out: &mut [f32],
) {
    input
        .par_chunks(d)
        .zip(codes_out.par_chunks_mut(d))
        .zip(scales_out.par_iter_mut())
        .for_each(|((row, out_codes), out_scale)| {
            let abs_max = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if abs_max == 0.0 { 1.0 } else { abs_max };
            let inv = 127.0 / scale;
            for (c, &v) in out_codes.iter_mut().zip(row.iter()) {
                *c = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
            *out_scale = scale / 127.0;
        });
}

/// Batch decode INT8 codes back to f32 without any per-row heap allocation.
///
/// # Arguments
/// * `codes`  — flat i8 slice, length = `n * d`
/// * `scales` — per-row scale factors (`abs_max / 127.0`), length = `n`
/// * `d`      — vector dimension
/// * `out`    — caller-allocated f32 slice, length = `n * d` (written in-place)
pub fn batch_decode_into(codes: &[i8], scales: &[f32], d: usize, out: &mut [f32]) {
    codes
        .par_chunks(d)
        .zip(scales.par_iter())
        .zip(out.par_chunks_mut(d))
        .for_each(|((row_codes, &scale), out_row)| {
            for (o, &c) in out_row.iter_mut().zip(row_codes.iter()) {
                *o = c as f32 * scale;
            }
        });
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

    #[test]
    fn encode_fast_matches_scalar() {
        // Verify that the SIMD path produces bit-identical results to the scalar path
        // across a variety of vector lengths (including non-multiples of 16).
        for &len in &[0usize, 1, 3, 7, 15, 16, 17, 64, 128, 256, 768] {
            let v: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.17) - 3.0).sin()).collect();
            let scalar = Int8Vector::encode(&v);
            let fast   = Int8Vector::encode_fast(&v);
            assert_eq!(scalar.scale, fast.scale, "scale mismatch at len={len}");
            assert_eq!(scalar.codes, fast.codes, "codes mismatch at len={len}");
        }
    }

    /// AVX2-specific parity test (only compiled and run on x86-64 with AVX2).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn encode_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return; // skip on CPUs without AVX2
        }
        for &len in &[0usize, 1, 3, 7, 8, 9, 15, 16, 17, 64, 128, 256, 768] {
            let v: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.13) - 2.5).cos()).collect();
            let scalar = Int8Vector::encode(&v);
            // SAFETY: guarded by feature check above.
            let avx2   = unsafe { encode_avx2(&v) };
            assert_eq!(scalar.scale, avx2.scale, "scale mismatch at len={len}");
            assert_eq!(scalar.codes, avx2.codes, "codes mismatch at len={len}");
        }
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn encode_decode_roundtrip(
            v in proptest::collection::vec(proptest::num::f32::NORMAL, 1..512usize)
        ) {
            let enc = Int8Vector::encode(&v);
            let dec = enc.decode();
            let dot: f32 = v.iter().zip(dec.iter()).map(|(a, b)| a * b).sum();
            let n1: f32  = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let n2: f32  = dec.iter().map(|x| x * x).sum::<f32>().sqrt();
            // Skip vectors whose squared-norm overflows f32 (>~ 1e19 per element).
            if n1 > 0.0 && n1.is_finite() && n2 > 0.0 && n2.is_finite() && dot.is_finite() {
                prop_assert!(
                    dot / (n1 * n2) >= 0.999,
                    "cosine {:.6} < 0.999 at len {}",
                    dot / (n1 * n2),
                    v.len()
                );
            }
        }

        #[test]
        fn scale_matches_abs_max(
            v in proptest::collection::vec(proptest::num::f32::NORMAL, 1..256usize)
        ) {
            let enc = Int8Vector::encode(&v);
            let true_max = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            prop_assert!(
                (enc.scale - true_max).abs() < 1e-6,
                "scale {} != abs_max {}",
                enc.scale,
                true_max
            );
        }
    }
}
