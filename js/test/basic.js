'use strict';
/**
 * basic.js — Integration tests for @vectro/core N-API addon.
 *
 * Tests: parseHeader, parseBody, dequantize, readVqz, VqzReader.
 * Run:   node test/basic.js            (from js/ directory)
 *        npm test                       (from js/ directory)
 */

const assert = require('node:assert');
const fs     = require('node:fs');
const os     = require('node:os');
const path   = require('node:path');

const native = require('../index.js');

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓  ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✗  ${name}`);
    console.error(`     ${err.message}`);
    failed++;
  }
}

// ---------------------------------------------------------------------------
// VQZ header builder — produces a minimal 64-byte header + optional body.
// Matches the format defined in python/storage_v3.py.
// ---------------------------------------------------------------------------
const VQZ_MAGIC = Buffer.from([0x56, 0x45, 0x43, 0x54, 0x52, 0x4f, 0x03, 0x00]); // "VECTRO\x03\x00"
const HEADER_SIZE = 64;
const COMP_NONE   = 0;

/**
 * Build a 64-byte VQZ header buffer.
 * @param {object} opts
 * @param {number} opts.version       - uint16
 * @param {number} opts.compFlag      - uint16
 * @param {bigint} opts.nVectors      - uint64
 * @param {number} opts.dims          - uint32
 * @param {number} [opts.nSubspaces]  - uint16
 * @param {number} [opts.metadataLen] - uint32
 */
function buildHeader({ version, compFlag, nVectors, dims, nSubspaces = 0, metadataLen = 0 }) {
  const hdr = Buffer.alloc(HEADER_SIZE, 0);
  VQZ_MAGIC.copy(hdr, 0);
  hdr.writeUInt16LE(version,      8);
  hdr.writeUInt16LE(compFlag,     10);
  hdr.writeBigUInt64LE(nVectors,  12);
  hdr.writeUInt32LE(dims,         20);
  hdr.writeUInt16LE(nSubspaces,   24);
  hdr.writeUInt32LE(metadataLen,  26);
  // bytes [30..63] stay zero (checksum + reserved)
  return hdr;
}

/**
 * Build a minimal uncompressed .vqz file buffer.
 * @param {number}   n     - number of vectors
 * @param {number}   dims  - dimensionality
 * @param {Int8Array} q    - quantized codes (n*dims elements)
 * @param {Float32Array} s - scales (n elements)
 */
function buildVqzBuffer(n, dims, q, s) {
  const hdr  = buildHeader({ version: 1, compFlag: COMP_NONE, nVectors: BigInt(n), dims });
  const body = Buffer.allocUnsafe(q.byteLength + s.byteLength);
  Buffer.from(q.buffer, q.byteOffset, q.byteLength).copy(body, 0);
  Buffer.from(s.buffer, s.byteOffset, s.byteLength).copy(body, q.byteLength);
  return Buffer.concat([hdr, body]);
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------
const N    = 3;
const DIMS = 4;
// Known quantized codes: N × DIMS
const Q_CODES = new Int8Array([
   1,  2,  3,  4,
  -1, -2, -3, -4,
   0, 127, -128, 64,
]);
// Per-vector scale factors
const SCALES  = new Float32Array([1.0, 2.0, 0.5]);

// Expected dequantized output = Q_CODES[i] * SCALES[i]
const EXPECTED = new Float32Array([
   1.0,   2.0,   3.0,   4.0,
  -2.0,  -4.0,  -6.0,  -8.0,
   0.0,  63.5, -64.0,  32.0,
]);
const TOLERANCE = 1e-5;

// ---------------------------------------------------------------------------
// 1 — parseHeader
// ---------------------------------------------------------------------------
console.log('\nparseHeader');
test('parses valid 64-byte header', () => {
  const hdr = buildHeader({ version: 1, compFlag: COMP_NONE, nVectors: BigInt(N), dims: DIMS, nSubspaces: 0, metadataLen: 0 });
  const h = native.parseHeader(hdr);
  assert.strictEqual(h.version,    1,       'version');
  assert.strictEqual(h.compFlag,   COMP_NONE, 'compFlag');
  assert.strictEqual(Number(h.n),  N,       'n');
  assert.strictEqual(h.dims,       DIMS,    'dims');
  assert.strictEqual(h.nSubspaces, 0,       'nSubspaces');
  assert.strictEqual(h.metadataLen, 0,      'metadataLen');
});

test('throws on buffer too small', () => {
  assert.throws(
    () => native.parseHeader(Buffer.alloc(10)),
    /too small/i,
  );
});

test('throws on bad magic', () => {
  const hdr = buildHeader({ version: 1, compFlag: 0, nVectors: 1n, dims: 4 });
  hdr[0] = 0x00; // corrupt magic
  assert.throws(() => native.parseHeader(hdr), /magic/i);
});

test('throws if not a Buffer', () => {
  assert.throws(() => native.parseHeader('not a buffer'), /expected Buffer/i);
});

// ---------------------------------------------------------------------------
// 2 — parseBody
// ---------------------------------------------------------------------------
console.log('\nparseBody');
test('splits body into Int8Array + Float32Array', () => {
  const body = Buffer.allocUnsafe(Q_CODES.byteLength + SCALES.byteLength);
  Buffer.from(Q_CODES.buffer).copy(body, 0);
  Buffer.from(SCALES.buffer).copy(body, Q_CODES.byteLength);

  const d = native.parseBody(body, N, DIMS);
  assert.strictEqual(d.n,    N,    'n');
  assert.strictEqual(d.dims, DIMS, 'dims');
  assert.strictEqual(d.quantized.length, N * DIMS, 'quantized.length');
  assert.strictEqual(d.scales.length,    N,         'scales.length');

  for (let i = 0; i < Q_CODES.length; i++) {
    assert.strictEqual(d.quantized[i], Q_CODES[i], `quantized[${i}]`);
  }
  for (let i = 0; i < SCALES.length; i++) {
    assert.ok(Math.abs(d.scales[i] - SCALES[i]) < TOLERANCE, `scales[${i}]`);
  }
});

test('throws when buffer too small', () => {
  const tiny = Buffer.alloc(4);
  assert.throws(() => native.parseBody(tiny, N, DIMS), /too small/i);
});

// ---------------------------------------------------------------------------
// 3 — dequantize
// ---------------------------------------------------------------------------
console.log('\ndequantize');
test('numeric correctness against known values', () => {
  const out = native.dequantize(Q_CODES, SCALES, DIMS);
  assert.strictEqual(out.length, N * DIMS, 'output length');
  for (let i = 0; i < EXPECTED.length; i++) {
    assert.ok(
      Math.abs(out[i] - EXPECTED[i]) < TOLERANCE,
      `out[${i}]: expected ${EXPECTED[i]}, got ${out[i]}`,
    );
  }
});

test('returns Float32Array', () => {
  const out = native.dequantize(Q_CODES, SCALES, DIMS);
  assert.ok(out instanceof Float32Array, 'instanceof Float32Array');
});

test('throws on mismatched lengths', () => {
  const wrongScales = new Float32Array([1.0]); // too short
  assert.throws(
    () => native.dequantize(Q_CODES, wrongScales, DIMS),
    /must equal/i,
  );
});

test('handles single vector, large dims', () => {
  const LARGE = 768;
  const q = new Int8Array(LARGE).fill(100);
  const s = new Float32Array([0.01]);
  const out = native.dequantize(q, s, LARGE);
  assert.strictEqual(out.length, LARGE);
  for (let i = 0; i < LARGE; i++) {
    assert.ok(Math.abs(out[i] - 1.0) < TOLERANCE, `large[${i}]`);
  }
});

// ---------------------------------------------------------------------------
// 4 — readVqz (full file round-trip)
// ---------------------------------------------------------------------------
console.log('\nreadVqz');

let tmpFile;
test('writes then reads a temporary .vqz file', () => {
  tmpFile = path.join(os.tmpdir(), `vectro_test_${process.pid}.vqz`);
  const buf = buildVqzBuffer(N, DIMS, Q_CODES, SCALES);
  fs.writeFileSync(tmpFile, buf);

  const d = native.readVqz(tmpFile);
  assert.strictEqual(d.n,    N,        'n');
  assert.strictEqual(d.dims, DIMS,     'dims');
  assert.strictEqual(d.quantized.length, N * DIMS, 'quantized.length');
  assert.strictEqual(d.scales.length,    N,         'scales.length');

  for (let i = 0; i < Q_CODES.length; i++) {
    assert.strictEqual(d.quantized[i], Q_CODES[i], `quantized[${i}]`);
  }
  for (let i = 0; i < SCALES.length; i++) {
    assert.ok(Math.abs(d.scales[i] - SCALES[i]) < TOLERANCE, `scales[${i}]`);
  }
});

test('reconstructed values match dequantize output', () => {
  if (!tmpFile) throw new Error('tmpFile not set — previous test failed');
  const d = native.readVqz(tmpFile);
  const out = native.dequantize(d.quantized, d.scales, DIMS);
  for (let i = 0; i < EXPECTED.length; i++) {
    assert.ok(
      Math.abs(out[i] - EXPECTED[i]) < TOLERANCE,
      `reconstructed[${i}]: expected ${EXPECTED[i]}, got ${out[i]}`,
    );
  }
});

test('throws on non-existent file', () => {
  assert.throws(() => native.readVqz('/tmp/__vectro_nonexistent__.vqz'), /cannot open/i);
});

// clean up temp file
if (tmpFile) {
  try { fs.unlinkSync(tmpFile); } catch (_) {}
}

// ---------------------------------------------------------------------------
// 5 — VqzReader (class lifecycle)
// ---------------------------------------------------------------------------
console.log('\nVqzReader');

let tmpFile2;
test('constructor + read + close lifecycle', () => {
  tmpFile2 = path.join(os.tmpdir(), `vectro_reader_${process.pid}.vqz`);
  const buf = buildVqzBuffer(N, DIMS, Q_CODES, SCALES);
  fs.writeFileSync(tmpFile2, buf);

  const reader = new native.VqzReader(tmpFile2);
  const d = reader.read();
  assert.strictEqual(d.n,    N,    'n');
  assert.strictEqual(d.dims, DIMS, 'dims');
  reader.close();
});

test('close then read throws', () => {
  if (!tmpFile2) throw new Error('tmpFile2 not set');
  const reader = new native.VqzReader(tmpFile2);
  reader.close();
  assert.throws(() => reader.read(), /closed|never opened/i);
});

// clean up
if (tmpFile2) {
  try { fs.unlinkSync(tmpFile2); } catch (_) {}
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
console.log(`\n${'─'.repeat(40)}`);
if (failed === 0) {
  console.log(`All ${passed} tests passed.`);
  process.exit(0);
} else {
  console.error(`${failed} of ${passed + failed} tests FAILED.`);
  process.exit(1);
}
