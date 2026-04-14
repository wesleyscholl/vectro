/**
 * @vectro/core — TypeScript definitions for the Vectro N-API native addon.
 *
 * Phase 2 surface (ADR-001): full .vqz parser, decompressor, and INT8
 * dequantization with NEON SIMD on arm64 and auto-vectorized scalar on x86-64.
 *
 * See: docs/adr-001-javascript-bindings.md
 */

/**
 * Parsed representation of the 64-byte VQZ file header.
 */
export interface VqzHeader {
  /** Format version (currently 1). */
  version: number;
  /** Compression flag: 0 = none, 1 = zstd, 2 = zlib. */
  compFlag: number;
  /** Total number of vectors in the artifact. */
  n: number;
  /** Dimensionality of each vector. */
  dims: number;
  /** Product-quantization subspace count (0 for non-PQ formats). */
  nSubspaces: number;
  /** Length of the metadata blob in bytes immediately following the header. */
  metadataLen: number;
}

/**
 * Data returned by {@link readVqz}, {@link parseBody}, and {@link VqzReader.read}.
 */
export interface VqzData {
  /** INT8 quantized codes, row-major order, shape (n * dims). */
  quantized: Int8Array;
  /** Per-vector float32 scale factors, length n. */
  scales: Float32Array;
  /** Dimensionality of each reconstructed vector. */
  dims: number;
  /** Number of vectors stored in the artifact. */
  n: number;
}

/**
 * Parse only the 64-byte VQZ header from a buffer.
 *
 * Validates the magic bytes (`VECTRO\x03\x00`) and returns all six header
 * fields. The buffer must be at least 64 bytes long.
 *
 * @param buffer - Buffer containing (at least) the 64-byte VQZ header.
 * @returns Parsed header fields as a {@link VqzHeader}.
 */
export function parseHeader(buffer: Buffer): VqzHeader;

/**
 * Split a decompressed VQZ body into typed arrays.
 *
 * The body layout is `int8[n * dims] || float32[n]` (C row-major, no padding).
 * `quantized` and `scales` in the returned object share a single underlying
 * `ArrayBuffer`; do not resize or detach either array.
 *
 * @param buffer - Decompressed body bytes (header and metadata already stripped).
 * @param n      - Number of vectors.
 * @param dims   - Vector dimensionality.
 * @returns VqzData with typed array views into a shared ArrayBuffer.
 */
export function parseBody(buffer: Buffer, n: number, dims: number): VqzData;

/**
 * Dequantize INT8 quantized codes to float32 using per-vector scales.
 *
 * Equivalent to the NumPy expression:
 *
 * ```python
 * quantized.astype(np.float32) * scales[:, np.newaxis]
 * ```
 *
 * Uses ARM NEON intrinsics on arm64; scalar path with `-O3` auto-vectorization
 * (SSE/AVX) on x86-64.
 *
 * @param quantized - Flat INT8 array of shape (n * dims) in row-major order.
 * @param scales    - Float32 array of length n containing per-vector scale factors.
 * @param dims      - Dimensionality of each vector.
 * @returns Float32 array of shape (n * dims) in row-major order.
 */
export function dequantize(
  quantized: Int8Array,
  scales: Float32Array,
  dims: number,
): Float32Array;

/**
 * Read a Vectro `.vqz` file and return its decoded contents.
 *
 * 1. Opens `path` and reads the 64-byte header.
 * 2. Decompresses the body with libzstd or zlib per `comp_flag`.
 * 3. Splits the body into an `Int8Array` (quantized codes) and `Float32Array`
 *    (per-vector scale factors).
 *
 * @param path - Absolute or relative path to the `.vqz` file.
 * @returns Parsed {@link VqzData} object.
 */
export function readVqz(path: string): VqzData;

/**
 * Object-oriented handle for reading a `.vqz` file.
 *
 * Equivalent to {@link readVqz} but with an explicit open/close lifecycle,
 * useful when reading is deferred or multiple handles are managed.
 */
export declare class VqzReader {
  /**
   * Open a `.vqz` file.
   *
   * @param path - Path to the `.vqz` file.
   */
  constructor(path: string);

  /**
   * Read and decode the file opened by the constructor.
   *
   * @returns Decoded {@link VqzData} object.
   */
  read(): VqzData;

  /**
   * Release any native resources held by this reader.
   */
  close(): void;
}
