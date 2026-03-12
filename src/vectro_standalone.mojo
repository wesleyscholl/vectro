"""Vectro unified Mojo binary — data-exchange CLI.

Compiled to `vectro_quantizer` and invoked from Python via subprocess.
Data exchange via pipe (stdin/stdout) — zero temp-file overhead.

Layout:
  float32 : n × d × 4 bytes, little-endian IEEE 754
  int8    : n × d × 1 bytes, signed two's complement
  uint8   : n × k × 1 bytes (k = ceil(d/2) for NF4, ceil(d/8) for binary)
  scales  : n × 4 bytes, little-endian IEEE 754 float32

Commands (file-path mode, legacy):
  vectro_quantizer int8 quantize <in.f32> <out.i8> <out_scales> <n> <d>
  vectro_quantizer int8 recon    <in.i8>  <in_scales> <out.f32> <n> <d>
  vectro_quantizer nf4  encode   <in.f32> <out.u8>  <out_scales> <n> <d>
  vectro_quantizer nf4  decode   <in.u8>  <in_scales> <out.f32> <n> <d>
  vectro_quantizer bin  encode   <in.f32> <out.u8> <n> <d>
  vectro_quantizer bin  decode   <in.u8>  <out.f32> <n> <d>

Commands (pipe mode, preferred — no disk I/O):
  vectro_quantizer pipe int8 quantize <n> <d>
  vectro_quantizer pipe int8 recon    <n> <d>
  vectro_quantizer pipe nf4  encode   <n> <d>
  vectro_quantizer pipe nf4  decode   <n> <d>
  vectro_quantizer pipe bin  encode   <n> <d>
  vectro_quantizer pipe bin  decode   <n> <d>

  Pipe stdin/stdout layout:
    int8 quantize  stdin : n*d float32       stdout: n*d int8 + n float32 scales
    int8 recon     stdin : n*d int8 + n f32  stdout: n*d float32
    nf4  encode    stdin : n*d float32       stdout: n*ceil(d/2) uint8 + n float32 scales
    nf4  decode    stdin : n*ceil(d/2)+n*4   stdout: n*d float32
    bin  encode    stdin : n*d float32       stdout: n*ceil(d/8) uint8
    bin  decode    stdin : n*ceil(d/8) uint8 stdout: n*d float32

Other commands:
  vectro_quantizer benchmark <n> <d>
  vectro_quantizer selftest
"""

from algorithm import vectorize, parallelize
from io import FileDescriptor
from math import copysign
from memory import bitcast
from time import perf_counter_ns
from sys import argv

# Tile 4 NEON lanes in software (LLVM pipelines the 4 loads better than scalar).
alias SIMD_W: Int = 16

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
# Pipe I/O -- stdin/stdout binary transfer (zero temp-file overhead)
# ────────────────────────────────────────────────────────────────────────────

fn read_stdin_bytes() raises -> List[UInt8]:
    """Read all bytes from stdin."""
    var data: List[UInt8]
    with open("/dev/stdin", "r") as f:
        data = f.read_bytes()
    return data^


fn write_stdout_bytes(buf: List[UInt8]):
    """Write bytes to stdout."""
    var fd = FileDescriptor(1)
    fd.write_bytes(buf)


fn f32_to_bytes(vals: List[Float32]) -> List[UInt8]:
    """Serialise float32 list to little-endian bytes."""
    var buf = List[UInt8](capacity=len(vals) * 4)
    for i in range(len(vals)):
        var bits = f32_bits(vals[i])
        buf.append(UInt8(bits & 0xFF))
        buf.append(UInt8((bits >> 8) & 0xFF))
        buf.append(UInt8((bits >> 16) & 0xFF))
        buf.append(UInt8((bits >> 24) & 0xFF))
    return buf^


fn i8_to_bytes(vals: List[Int8]) -> List[UInt8]:
    """Reinterpret Int8 list as UInt8 bytes."""
    var buf = List[UInt8](capacity=len(vals))
    for i in range(len(vals)):
        buf.append(i8_to_u8(vals[i]))
    return buf^


fn read_f32_from_raw(raw: List[UInt8], offset: Int, count: Int) -> List[Float32]:
    """Parse float32 values from a byte buffer starting at `offset`."""
    var out = List[Float32](capacity=count)
    for i in range(count):
        var b0 = UInt32(raw[offset + i * 4])
        var b1 = UInt32(raw[offset + i * 4 + 1]) << 8
        var b2 = UInt32(raw[offset + i * 4 + 2]) << 16
        var b3 = UInt32(raw[offset + i * 4 + 3]) << 24
        out.append(bits_f32(b0 | b1 | b2 | b3))
    return out^


fn read_i8_from_raw(raw: List[UInt8], offset: Int, count: Int) -> List[Int8]:
    """Parse Int8 values from a byte buffer starting at `offset`."""
    var out = List[Int8](capacity=count)
    for i in range(count):
        out.append(u8_to_i8(raw[offset + i]))
    return out^


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
    """INT8 symmetric abs-max quantization — SIMD inner loops, parallel rows."""
    var q = List[Int8](capacity=n * d)
    var scales = List[Float32](capacity=n)
    q.resize(n * d, Int8(0))         # bulk zero-fill (memset) — 6x faster than append loop
    scales.resize(n, Float32(0.0))

    var emb_ptr    = emb.unsafe_ptr()
    var q_ptr_out  = q.unsafe_ptr()
    var scales_ptr = scales.unsafe_ptr()

    @parameter
    fn _process_row(i: Int):
        var ptr  = emb_ptr   + i * d
        var qptr = q_ptr_out + i * d

        # Pass 1: SIMD abs-max reduction
        var acc_max: Float32 = 0.0

        @parameter
        fn _max_kernel[w: Int](j: Int):
            acc_max = max(acc_max, abs(ptr.load[width=w](j)).reduce_max())

        vectorize[_max_kernel, SIMD_W](d)

        var scale: Float32 = acc_max / 127.0 if acc_max > 0.0 else Float32(1.0)
        scales_ptr[i] = scale
        var inv = Float32(1.0) / scale

        # Pass 2: SIMD quantize, clamp, round-to-nearest, store
        @parameter
        fn _quant_kernel[w: Int](j: Int):
            var raw = ptr.load[width=w](j) * inv
            raw = max(raw, SIMD[DType.float32, w](-127.0))
            raw = min(raw, SIMD[DType.float32, w](127.0))
            var half = copysign(SIMD[DType.float32, w](0.5), raw)
            qptr.store(j, (raw + half).cast[DType.int32]().cast[DType.int8]())

        vectorize[_quant_kernel, SIMD_W](d)

    parallelize[_process_row](n)

    return QuantResult(q^, scales^)


fn reconstruct_int8(q: List[Int8], scales: List[Float32], n: Int, d: Int) -> List[Float32]:
    """Reconstruct float32 from INT8 + per-vector scales — SIMD cast+multiply, parallel rows."""
    var out = List[Float32](capacity=n * d)
    out.resize(n * d, Float32(0.0))

    var q_ptr_in   = q.unsafe_ptr()
    var o_ptr_out  = out.unsafe_ptr()
    var scales_ptr = scales.unsafe_ptr()

    @parameter
    fn _recon_row(i: Int):
        var qp = q_ptr_in  + i * d
        var op = o_ptr_out + i * d
        var s  = scales_ptr[i]

        @parameter
        fn _recon_kernel[w: Int](j: Int):
            var qi = qp.load[width=w](j)
            op.store(j, qi.cast[DType.float32]() * SIMD[DType.float32, w](s))

        vectorize[_recon_kernel, SIMD_W](d)

    parallelize[_recon_row](n)

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
    packed.resize(n * half_d, UInt8(0))
    scales.resize(n, Float32(0.0))

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
    out.resize(n * d, Float32(0.0))

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
    packed.resize(n * bpv, UInt8(0))

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
    out.resize(n * d, Float32(0.0))

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

    # Full-N warmup to fill caches before timing.
    for _ in range(2): _ = quantize_int8(data, n, d)

    # 5 timed iterations — report best-of (peak throughput).
    var best_q_ns  = UInt(1) << 62
    var best_rs_ns = UInt(1) << 62
    for _ in range(5):
        var t0 = perf_counter_ns()
        var r = quantize_int8(data, n, d)
        var qns = UInt(perf_counter_ns() - t0)
        if qns < best_q_ns: best_q_ns = qns

        var t1 = perf_counter_ns()
        _ = reconstruct_int8(r.q, r.scales, n, d)
        var rns = UInt(perf_counter_ns() - t1)
        if rns < best_rs_ns: best_rs_ns = rns

    print("Benchmark n=", n, " d=", d)
    print("  INT8 quantize  :", Int(Float64(n) / (Float64(best_q_ns)  / 1e9)), "vec/s")
    print("  INT8 reconstruct:", Int(Float64(n) / (Float64(best_rs_ns) / 1e9)), "vec/s")
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

    # ── pipe (stdin → stdout, zero disk I/O) ─────────────────────────────────
    if cmd == "pipe":
        if argc < 6:
            print("Usage: vectro_quantizer pipe <type> <op> <n> <d>")
            return
        var ptype = args[2]
        var pop   = args[3]
        var pn    = Int(args[4])
        var pd    = Int(args[5])

        var raw = read_stdin_bytes()

        if ptype == "int8":
            if pop == "quantize":
                # stdin: pn*pd float32 → stdout: pn*pd int8 + pn float32 scales
                var data = read_f32_from_raw(raw, 0, pn * pd)
                var r = quantize_int8(data, pn, pd)
                var buf = i8_to_bytes(r.q)
                var sbytes = f32_to_bytes(r.scales)
                for i in range(len(sbytes)):
                    buf.append(sbytes[i])
                write_stdout_bytes(buf)
                return

            if pop == "recon":
                # stdin: pn*pd int8 + pn float32 scales → stdout: pn*pd float32
                var q  = read_i8_from_raw(raw, 0, pn * pd)
                var sc = read_f32_from_raw(raw, pn * pd, pn)
                var recon = reconstruct_int8(q, sc, pn, pd)
                write_stdout_bytes(f32_to_bytes(recon))
                return

        if ptype == "nf4":
            var half_pd = (pd + 1) // 2
            if pop == "encode":
                # stdin: pn*pd float32 → stdout: pn*half_pd uint8 + pn float32 scales
                var data = read_f32_from_raw(raw, 0, pn * pd)
                var r = encode_nf4(data, pn, pd)
                var sbytes = f32_to_bytes(r.scales)
                for i in range(len(sbytes)):
                    r.packed.append(sbytes[i])
                write_stdout_bytes(r.packed)
                return

            if pop == "decode":
                # stdin: pn*half_pd uint8 + pn float32 scales → stdout: pn*pd float32
                var packed = List[UInt8](capacity=pn * half_pd)
                for i in range(pn * half_pd):
                    packed.append(raw[i])
                var sc = read_f32_from_raw(raw, pn * half_pd, pn)
                var recon = decode_nf4(packed, sc, pn, pd)
                write_stdout_bytes(f32_to_bytes(recon))
                return

        if ptype == "bin":
            var bpv = (pd + 7) // 8
            if pop == "encode":
                # stdin: pn*pd float32 → stdout: pn*bpv uint8
                var data = read_f32_from_raw(raw, 0, pn * pd)
                var packed = encode_binary(data, pn, pd)
                write_stdout_bytes(packed)
                return

            if pop == "decode":
                # stdin: pn*bpv uint8 → stdout: pn*pd float32
                var packed = List[UInt8](capacity=pn * bpv)
                for i in range(pn * bpv):
                    packed.append(raw[i])
                var recon = decode_binary(packed, pn, pd)
                write_stdout_bytes(f32_to_bytes(recon))
                return

        print("Unknown pipe subcommand:", ptype, pop)
        return

    print("Unknown command:", cmd)
    print("Commands: int8, nf4, bin, benchmark, selftest")
