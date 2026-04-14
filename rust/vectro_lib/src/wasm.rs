//! WASM bindings for vectro_lib.
//!
//! Exposes INT8 and NF4 single-vector encoding to JavaScript via wasm-bindgen.
//! Build with:
//!   wasm-pack build rust/vectro_lib --target web --release
//!
//! The resulting `pkg/` directory contains the `.wasm` binary, a JS wrapper,
//! and TypeScript type declarations.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

use crate::quant::int8::Int8Vector;
use crate::quant::nf4::Nf4Vector;

// --------------------------------------------------------------------------- //
// INT8 bindings
// --------------------------------------------------------------------------- //

/// Encode a f32 slice to INT8 quantised codes.
///
/// Returns a `Int8Array`-compatible `Vec<i32>` (wasm-bindgen does not support
/// `Vec<i8>` directly — values are sign-extended to i32 for portability).
///
/// Pair with `encode_int8_scale` to obtain the per-vector scale factor needed
/// for dequantisation: `decoded[i] = codes[i] / 127.0 * scale`.
#[wasm_bindgen]
pub fn encode_int8(vec: &[f32]) -> Vec<i32> {
    Int8Vector::encode_fast(vec)
        .codes
        .into_iter()
        .map(|c| c as i32)
        .collect()
}

/// Return the per-vector abs-max scale for an INT8-encoded vector.
///
/// Call this alongside `encode_int8` using the **same** input slice.
/// Because both functions call `encode_fast` independently, this incurs a
/// second pass over the data. Consumers that need both fields should call
/// `encode_int8_full` instead.
#[wasm_bindgen]
pub fn encode_int8_scale(vec: &[f32]) -> f32 {
    Int8Vector::encode_fast(vec).scale
}

/// Encode a f32 slice to INT8 and return `[codes..., scale]` as a single
/// `Float32Array`-friendly `Vec<f32>`.
///
/// Layout: `result[0..dim]` = codes cast to f32, `result[dim]` = scale.
/// This single-pass variant is preferred when both codes and scale are needed.
#[wasm_bindgen]
pub fn encode_int8_full(vec: &[f32]) -> Vec<f32> {
    let q = Int8Vector::encode_fast(vec);
    let mut out: Vec<f32> = q.codes.into_iter().map(|c| c as f32).collect();
    out.push(q.scale);
    out
}

// --------------------------------------------------------------------------- //
// NF4 bindings
// --------------------------------------------------------------------------- //

/// Encode a f32 slice to packed NF4 nibbles (QLoRA-style).
///
/// Returns a `Uint8Array`-compatible `Vec<u8>` of `ceil(dim / 2)` bytes.
/// Each byte packs two consecutive 4-bit NF4 codes (low nibble first).
///
/// Pair with `encode_nf4_scale` and `encode_nf4_dim` to fully reconstruct.
#[wasm_bindgen]
pub fn encode_nf4(vec: &[f32]) -> Vec<u8> {
    Nf4Vector::encode_fast(vec).packed
}

/// Return the per-vector abs-max scale for an NF4-encoded vector.
#[wasm_bindgen]
pub fn encode_nf4_scale(vec: &[f32]) -> f32 {
    Nf4Vector::encode_fast(vec).scale
}

/// Return the original embedding dimension encoded in the NF4 struct.
///
/// Needed to correctly unpack the last byte when `dim` is odd.
#[wasm_bindgen]
pub fn encode_nf4_dim(vec: &[f32]) -> usize {
    vec.len()
}
