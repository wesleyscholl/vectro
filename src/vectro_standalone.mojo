"""Vectro unified Mojo binary — data-exchange CLI.

Compiled to `vectro_quantizer` and invoked from Python via subprocess.
Data exchange via raw little-endian binary temp files (numpy-compatible).

Layout:
  float32 files : n × d × 4 bytes, little-endian IEEE 754
  int8 files    : n × d × 1 bytes, signed two's complement
  uint8 files   : n × k × 1 bytes (k = ceil(d/2) for NF4, ceil(d/8) for binary)
  scales files  : n × 4 bytes, little-endian IEEE 754 float32

Commands:
  vectro_quantizer int8 quantize <in.f32> <out.i8> <out_scales> <n> <d>
  vectro_quantizer int8 recon    <in.i8>  <in_scales> <out.f32> <n> <d>
  vectro_quantizer nf4  encode   <in.f32> <out.u8>  <out_scales> <n> <d>
  vectro_quantizer nf4  decode   <in.u8>  <in_scales> <out.f32> <n> <d>
  vectro_quantizer bin  encode   <in.f32> <out.u8> <n> <d>
  vectro_quantizer bin  decode   <in.u8>  <out.f32> <n> <d>
  vectro_quantizer benchmark     <n> <d>
  vectro_quantizer selftest

Run with no arguments to execute the self-test and exit 0 on success.
"""

from algorithm import vectorize
from memory import bitcast
from time import perf_counter_ns
from sys import argv

# Apple M-series: 4 x float32 per 128-bit NEON lane.
alias SIMD_W: Int = 4

# ────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ────────────────────────────────────────────────────────────────────────────

@always_inline
fn f32_bits(val: Float32) -> UInt32:
    return bitcast[DType.uint32](val)

@always_inline
fn bits_f32(bits: UInt32) -> Float32:
    return bitcast[DType.float32](bits)

@always_inline
fn i8_to_u8(v: Int8) -> UInt8:
    var vi = Int(v)
    if vi < 0: vi += 256
    return UInt8(vi)

@always_inline
fn u8_to_i8(v: UInt8) -> Int8:
    var vi = Int(v)
    if vi > 127: vi -= 256
    return Int8(vi)

# ────────────────────────────────────────────────────────────────────────────
# File I/O -- little-endian float32 and int8 raw binary files
# ────────────────────────────────────────────────────────────────────────────

fn write_f32(vals: List[Float32], path: String) raises:
    var buf = List[UInt8](capacity=len(vals) * 4)
    for i in range(len(vals)):
        var bits = f32_bits(vals[i])
        buf.append(UInt8(bits & 0xFF))
        buf.append(UInt8((bits >> 8) & 0xFF))
        buf.append(UInt8((bits >> 16) & 0xFF))
        buf.append(UInt8((bits >> 24) & 0xFF))
    with open(path, "w") as f:
        f.write_bytes(Span(buf))


fn read_f32(path: String, count: Int) raises -> List[Float32]:
    var raw: List[UInt8]
    with open(path, "r") as f:
        raw = f.read_bytes()
    var out = List[Float32](capacity=count)
    for i in range(count):
        var b0 = UInt32(raw[i * 4])
        var b1 = UInt32(raw[i * 4 + 1]) << 8
        var b2 = UInt32(raw[i * 4 + 2]) << 16
        var b3 = UInt32(raw[i * 4 + 3]) << 24
        out.append(bits_f32(b0 | b1 | b2 | b3))
    return out^


fn write_i8(vals: List[Int8], path: String) raises:
    var buf = List[UInt8](capacity=len(vals))
    for i in range(len(vals)):
        buf.append(i8_to_u8(vals[i]))
    with open(path, "w") as f:
        f.write_bytes(Span(buf))


fn read_i8(path: String, count: Int) raises -> List[Int8]:
    var raw: List[UInt8]
    with open(path, "r") as f:
        raw = f.read_bytes()
    var out = List[Int8](capacity=count)
    for i in range(count):
        out.append(u8_to_i8(raw[i]))
    return out^


fn write_u8(vals: List[UInt8], path: String) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(vals))


fn read_u8(path: String, count: Int) raises -> List[UInt8]:
    var raw: List[UInt8]
    with open(path, "r") as f:
        raw = f.read_bytes()
    return raw^


# ────────────────────────────────────────────────────────────────────────────
# Return types (structs -- tuple returns don't support List types in 0.25.7)
# ────────────────────────────────────────────────────────────────────────────

struct QuantResult:
    var q: List[Int8]
    var scales: List[Float32]
    fn __init__(out self, var q: List[Int8], var scales: List[Float32]):
        self.q = q^
        self.scales = scales^


struct PackedResult:
    var packed: List[UInt8]
    var scales: List[Float32]
    fn __init__(out self, var packed: List[UInt8], var scales: List[Float32]):
        self.packed = packed^
        self.scales = scales^


# ────────────────────────────────────────────────────────────────────────────
# 1. INT8 SIMD quantize / reconstruct
# ────────────────────────────────────────────────────────────────────────────

fn quantize_int8(emb: List[Float32], n: Int, d: Int) -> QuantResult:
    """INT8 symmetric abs-max quantization (scalar — portable across Mojo versions)."""
    var q = List[Int8](capacity=n * d)
    var scales = List[Float32](capacity=n)
    for _ in range(n * d): q.append(Int8(0))
    for _ in range(n):     scales.append(Float32(0.0))

    for i in range(n):
        var base = i * d
        var ptr = emb.unsafe_ptr() + base

        # Pass 1: find max absolute value
        var acc_max: Float32 = 0.0
        for j in range(d):
            var v = ptr[j]
            if v < 0.0: v = -v
            if v > acc_max: acc_max = v

        var scale: Float32 = 1.0
        if acc_max > 0.0: scale = acc_max / 127.0
        scales[i] = scale
        var inv = Float32(1.0) / scale
        var qptr = q.unsafe_ptr() + base

        # Pass 2: quantize, clamp, round
        for j in range(d):
            var raw = ptr[j] * inv
            if raw > 127.0: raw = 127.0
            if raw < -127.0: raw = -127.0
            var r: Int
            if raw >= 0.0: r = Int(raw + 0.5)
            else:          r = Int(raw - 0.5)
            qptr[j] = Int8(r)

    return QuantResult(q^, scales^)


fn reconstruct_int8(q: List[Int8], scales: List[Float32], n: Int, d: Int) -> List[Float32]:
    """Reconstruct float32 from INT8 + per-vector scales (scalar)."""
    var out = List[Float32](capacity=n * d)
    for _ in range(n * d): out.append(Float32(0.0))

    for i in range(n):
        var base = i * d
        var s = scales[i]
        var qp = q.unsafe_ptr() + base
        var op = out.unsafe_ptr() + base
        for j in range(d):
            op[j] = Float32(qp[j]) * s

    return out^


# ────────────────────────────────────────────────────────────────────────────
# 2. NF4 encode / decode  (QLoRA 16-level normal-float codebook)
# ────────────────────────────────────────────────────────────────────────────

fn _nf4_level(idx: Int) -> Float32:
    if idx == 0:  return Float32(-1.0)
    if idx == 1:  return Float32(-0.6961928)
    if idx == 2:  return Float32(-0.5250730)
    if idx == 3:  return Float32(-0.3949003)
    if idx == 4:  return Float32(-0.2844677)
    if idx == 5:  return Float32(-0.1848745)
    if idx == 6:  return Float32(-0.09105004)
    if idx == 7:  return Float32(0.0)
    if idx == 8:  return Float32(0.07958031)
    if idx == 9:  return Float32(0.16093908)
    if idx == 10: return Float32(0.24611496)
    if idx == 11: return Float32(0.33791524)
    if idx == 12: return Float32(0.44070983)
    if idx == 13: return Float32(0.56266755)
    if idx == 14: return Float32(0.72295761)
    return Float32(1.0)


fn _nearest_nf4(v: Float32) -> Int:
    var best = 0
    var bd = abs(v - Float32(-1.0))
    for k in range(1, 16):
        var d = abs(v - _nf4_level(k))
        if d < bd:
            bd = d
            best = k
    return best


fn encode_nf4(emb: List[Float32], n: Int, d: Int) -> PackedResult:
    """NF4 normal-float 4-bit encoding. Two nibbles per byte (lo=first dim)."""
    var half_d = (d + 1) // 2
    var packed = List[UInt8](capacity=n * half_d)
    var scales = List[Float32](capacity=n)
    for _ in range(n * half_d): packed.append(UInt8(0))
    for _ in range(n):          scales.append(Float32(0.0))

    for i in range(n):
        var bf = i * d
        var bp = i * half_d

        var amax: Float32 = 0.0
        for j in range(d):
            var a = emb[bf + j]
            if a < 0.0: a = -a
            if a > amax: amax = a
        var scale: Float32 = 1.0
        if amax > 0.0: scale = amax
        scales[i] = scale
        var inv = Float32(1.0) / scale

        var bidx = bp
        var j = 0
        while j < d:
            var lo = _nearest_nf4(emb[bf + j] * inv)
            var hi = 0
            if j + 1 < d:
                hi = _nearest_nf4(emb[bf + j + 1] * inv)
            packed[bidx] = UInt8((hi << 4) | (lo & 0xF))
            bidx += 1
            j += 2

    return PackedResult(packed^, scales^)


fn decode_nf4(packed: List[UInt8], scales: List[Float32], n: Int, d: Int) -> List[Float32]:
    """Decode NF4 nibbles back to float32."""
    var half_d = (d + 1) // 2
    var out = List[Float32](capacity=n * d)
    for _ in range(n * d): out.append(Float32(0.0))

    for i in range(n):
        var bf = i * d
        var bp = i * half_d
        var scale = scales[i]

        var bidx = bp
        var j = 0
        while j < d:
            var b = Int(packed[bidx])
            out[bf + j] = _nf4_level(b & 0xF) * scale
            if j + 1 < d:
                out[bf + j + 1] = _nf4_level((b >> 4) & 0xF) * scale
            bidx += 1
            j += 2

    return out^


# ────────────────────────────────────────────────────────────────────────────
# 3. Binary (1-bit) encode / decode
# ────────────────────────────────────────────────────────────────────────────

fn encode_binary(emb: List[Float32], n: Int, d: Int) -> List[UInt8]:
    """Sign-bit packing: 8 dimensions per byte. Returns n x ceil(d/8) bytes."""
    var bpv = (d + 7) // 8
    var packed = List[UInt8](capacity=n * bpv)
    for _ in range(n * bpv): packed.append(UInt8(0))

    for i in range(n):
        var bf = i * d
        var bb = i * bpv
        for bi in range(bpv):
            var byte_val: UInt8 = 0
            for bit in range(8):
                var dim = bi * 8 + bit
                if dim < d:
                    if emb[bf + dim] >= 0.0:
                        byte_val = byte_val | UInt8(1 << bit)
            packed[bb + bi] = byte_val

    return packed^


fn decode_binary(packed: List[UInt8], n: Int, d: Int) -> List[Float32]:
    """Decode sign-bit packed bytes back to +/-1.0 float32."""
    var bpv = (d + 7) // 8
    var out = List[Float32](capacity=n * d)
    for _ in range(n * d): out.append(Float32(0.0))

    for i in range(n):
        var bf = i * d
        var bb = i * bpv
        for bi in range(bpv):
            var b = Int(packed[bb + bi])
            for bit in range(8):
                var dim = bi * 8 + bit
                if dim < d:
                    out[bf + dim] = Float32(1.0) if (b >> bit) & 1 == 1 else Float32(-1.0)

    return out^


# ────────────────────────────────────────────────────────────────────────────
# Benchmark
# ────────────────────────────────────────────────────────────────────────────

fn run_benchmark(n: Int, d: Int) raises:
    from random import random_float64

    var data = List[Float32](capacity=n * d)
    for _ in range(n * d):
        data.append(Float32(random_float64() * 2.0 - 1.0))

    _ = quantize_int8(data, min(32, n), d)      # warmup

    var t0 = perf_counter_ns()
    var r = quantize_int8(data, n, d)
    var qns = perf_counter_ns() - t0

    var t1 = perf_counter_ns()
    _ = reconstruct_int8(r.q, r.scales, n, d)
    var rns = perf_counter_ns() - t1

    print("Benchmark n=", n, " d=", d)
    print("  INT8 quantize  :", Int(Float64(n) / (Float64(qns) / 1e9)), "vec/s")
    print("  INT8 reconstruct:", Int(Float64(n) / (Float64(rns) / 1e9)), "vec/s")
    print("  SIMD_W:", SIMD_W)


# ────────────────────────────────────────────────────────────────────────────
# Self-test
# ────────────────────────────────────────────────────────────────────────────

fn run_selftest() raises:
    print("Vectro SIMD CLI v3.0.1 -- self-test")
    print("SIMD_W =", SIMD_W)

    var data = List[Float32]()
    for i in range(8): data.append(Float32(i + 1) * 0.25)

    # INT8
    var r8 = quantize_int8(data, 2, 4)
    var rec8 = reconstruct_int8(r8.q, r8.scales, 2, 4)
    var err8: Float32 = 0.0
    for i in range(8):
        var e = data[i] - rec8[i]
        err8 += e if e >= 0.0 else -e
    print("  INT8 mean-abs-err :", err8 / 8.0, "(expect < 0.02)")

    # NF4
    var rn4 = encode_nf4(data, 2, 4)
    var recn4 = decode_nf4(rn4.packed, rn4.scales, 2, 4)
    var errn4: Float32 = 0.0
    for i in range(8):
        var e = data[i] - recn4[i]
        errn4 += e if e >= 0.0 else -e
    print("  NF4  mean-abs-err :", errn4 / 8.0, "(expect < 0.10)")

    # Binary
    var pb = encode_binary(data, 2, 4)
    var recb = decode_binary(pb, 2, 4)
    var bok = True
    for i in range(8):
        var expected = Float32(1.0) if data[i] >= 0.0 else Float32(-1.0)
        if recb[i] != expected: bok = False
    print("  Binary decode  :", "OK" if bok else "FAIL")

    # File round-trip
    write_f32(data, "/tmp/vectro_test.f32")
    var back = read_f32("/tmp/vectro_test.f32", 8)
    var ferr: Float32 = 0.0
    for i in range(8):
        var e = data[i] - back[i]
        ferr += e if e >= 0.0 else -e
    print("  File f32 round-trip err:", ferr, "(expect 0.0)")

    write_i8(r8.q, "/tmp/vectro_test.i8")
    var bq = read_i8("/tmp/vectro_test.i8", 8)
    var ieq = True
    for i in range(8):
        if r8.q[i] != bq[i]: ieq = False
    print("  File i8  round-trip match:", "OK" if ieq else "FAIL")

    print("  Self-test COMPLETE")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

fn main() raises:
    var args = argv()
    var argc = len(args)

    if argc < 2:
        run_selftest()
        return

    var cmd = args[1]

    if cmd == "selftest":
        run_selftest()
        return

    if cmd == "benchmark":
        if argc < 4:
            print("Usage: vectro_quantizer benchmark <n> <d>")
            return
        run_benchmark(Int(args[2]), Int(args[3]))
        return

    # ── int8 ─────────────────────────────────────────────────────────────────
    if cmd == "int8":
        if argc < 3:
            print("Usage: vectro_quantizer int8 <quantize|recon> ...")
            return
        var sub = args[2]

        if sub == "quantize":
            # int8 quantize <in.f32> <out.i8> <out_scales> <n> <d>
            var n = Int(args[6])
            var d = Int(args[7])
            var data = read_f32(args[3], n * d)
            var r = quantize_int8(data, n, d)
            write_i8(r.q, args[4])
            write_f32(r.scales, args[5])
            return

        if sub == "recon":
            # int8 recon <in.i8> <in_scales> <out.f32> <n> <d>
            var n = Int(args[6])
            var d = Int(args[7])
            var q = read_i8(args[3], n * d)
            var s = read_f32(args[4], n)
            var recon = reconstruct_int8(q, s, n, d)
            write_f32(recon, args[5])
            return

        print("Unknown int8 subcommand:", sub)
        return

    # ── nf4 ──────────────────────────────────────────────────────────────────
    if cmd == "nf4":
        if argc < 3:
            print("Usage: vectro_quantizer nf4 <encode|decode> ...")
            return
        var sub = args[2]

        if sub == "encode":
            # nf4 encode <in.f32> <out.u8> <out_scales> <n> <d>
            var n = Int(args[6])
            var d = Int(args[7])
            var data = read_f32(args[3], n * d)
            var r = encode_nf4(data, n, d)
            write_u8(r.packed, args[4])
            write_f32(r.scales, args[5])
            return

        if sub == "decode":
            # nf4 decode <in.u8> <in_scales> <out.f32> <n> <d>
            var n = Int(args[6])
            var d = Int(args[7])
            var half_d = (d + 1) // 2
            var packed = read_u8(args[3], n * half_d)
            var scales = read_f32(args[4], n)
            var recon = decode_nf4(packed, scales, n, d)
            write_f32(recon, args[5])
            return

        print("Unknown nf4 subcommand:", sub)
        return

    # ── bin ──────────────────────────────────────────────────────────────────
    if cmd == "bin":
        if argc < 3:
            print("Usage: vectro_quantizer bin <encode|decode> ...")
            return
        var sub = args[2]

        if sub == "encode":
            # bin encode <in.f32> <out.u8> <n> <d>
            var n = Int(args[5])
            var d = Int(args[6])
            var data = read_f32(args[3], n * d)
            var packed = encode_binary(data, n, d)
            write_u8(packed, args[4])
            return

        if sub == "decode":
            # bin decode <in.u8> <out.f32> <n> <d>
            var n = Int(args[5])
            var d = Int(args[6])
            var bpv = (d + 7) // 8
            var packed = read_u8(args[3], n * bpv)
            var recon = decode_binary(packed, n, d)
            write_f32(recon, args[4])
            return

        print("Unknown bin subcommand:", sub)
        return

    print("Unknown command:", cmd)
    print("Commands: int8, nf4, bin, benchmark, selftest")
