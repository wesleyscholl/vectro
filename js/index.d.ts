/**
 * @vectro/core — TypeScript definitions for the Vectro N-API native addon.
 *
 * Phase 1 surface (v1.x): .vqz reader + INT8 dequantization.
 * Phase 2 surface (v2.x): WASM target for browser and edge environments.
 *
 * See: docs/adr-001-javascript-bindings.md
 */

/**
 * Data returned by {@link readVqz}.
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
 * Dequantize INT8 quantized codes to float32 using per-vector scales.
 *
 * Equivalent to the NumPy expression:
 *
 * ```python
 * quantized.astype(np.float32) * scales[:, np.newaxis]
 * ```
 *
 * @param quantized - Flat INT8 array of shape (n * dims) in row-major order.
 * @param scales    - Float32 array of length n containing per-vector scale factors.
 * @param dims      - Dimensionality of each vector.
 * @returns         - Float32 array of shape (n * dims) in row-major order.
 *
 * @throws {Error} Not yet implemented — see ADR-001 Phase 1.
 */
export function dequantize(
  quantized: Int8Array,
  scales: Float32Array,
  dims: number,
): Float32Array;

/**
 * Read a Vectro `.vqz` file and return the compressed data.
 *
 * The function:
 * 1. Opens the file and parses the 64-byte header (magic, version, comp_flag, n, dims).
 * 2. Decompresses the body with libzstd or zlib depending on the header's comp_flag.
 * 3. Splits the decompressed body into the INT8 quantized array and float32 scales.
 *
 * @param path - Absolute or relative path to the `.vqz` file.
 * @returns    - Parsed {@link VqzData} object.
 *
 * @throws {Error} Not yet implemented — see ADR-001 Phase 1.
 */
export function readVqz(path: string): VqzData;

/**
 * Streaming reader for large `.vqz` archives.
 *
 * Allows incremental processing of artifacts too large to hold in memory.
 *
 * @throws {Error} Not yet implemented — see ADR-001 Phase 1.
 */
export declare class VqzReader {
  /**
   * Open a `.vqz` file for streaming.
   *
   * @param path - Path to the `.vqz` file.
   */
  constructor(path: string);

  /**
   * Read the entire file and return its contents.
   */
  read(): VqzData;

  /**
   * Release file handle and native resources.
   */
  close(): void;
}
