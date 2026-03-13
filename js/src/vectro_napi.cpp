/**
 * vectro_napi.cpp ― Vectro N-API native addon (ADR-001 Phase 2).
 *
 * Implements the full .vqz reader and INT8 dequantize kernel.  Reads a
 * .vqz file (64-byte header + optional metadata + zstd/zlib/raw body),
 * decompresses it, and exposes the int8 quantized matrix and float32 scales
 * to Node.js as zero-copy typed arrays where possible.
 *
 * Public N-API surface:
 *   dequantize(quantized: Int8Array, scales: Float32Array, dims: number) → Float32Array
 *   readVqz(path: string) → { quantized, scales, dims, n }
 *   VqzReader class: constructor(path), read() → VqzData, close()
 *
 * Build: npm run build  (requires node-gyp and a C++17 compiler)
 * Platform targets: darwin-arm64, darwin-x64, linux-x64, linux-arm64, win32-x64
 *
 * VQZ binary layout:
 *   Offset  Length  Field
 *   0       8       magic: "VECTRO\x03\x00"
 *   8       2       version: uint16le
 *   10      2       comp_flags: uint16le  (0=none, 1=zstd, 2=zlib)
 *   12      8       n_vectors: uint64le
 *   20      4       dims: uint32le
 *   24      2       n_subspaces: uint16le
 *   26      4       metadata_len: uint32le
 *   30      8       checksum: (first 8 bytes of blake2b — not validated here)
 *   38      26      reserved
 *   64      meta    metadata (metadata_len bytes)
 *   64+meta ??      body: int8[n*d] ++ float32[n]
 */

#include <napi.h>
#include <zlib.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants matching storage_v3.py
// ---------------------------------------------------------------------------

static const char     VQZ_MAGIC[]   = "VECTRO\x03\x00";   // 8 bytes
static const size_t   HEADER_SIZE   = 64;
static const uint16_t COMP_NONE     = 0;
static const uint16_t COMP_ZLIB     = 2;      // also covers zstd fallback to zlib

// ---------------------------------------------------------------------------
// VQZ header — packed struct matching the Python definition.
// We read each field manually to avoid alignment / padding issues.
// ---------------------------------------------------------------------------

struct VqzHeader {
    char     magic[8];
    uint16_t version;
    uint16_t comp_flags;
    uint64_t n_vectors;
    uint32_t dims;
    uint16_t n_subspaces;
    uint32_t metadata_len;
    /* checksum(8) + reserved(26) — ignored */
};

static VqzHeader parse_header(const uint8_t* buf) {
    VqzHeader h{};
    std::memcpy(h.magic, buf, 8);
    // Little-endian reads
    std::memcpy(&h.version,      buf + 8,  2);
    std::memcpy(&h.comp_flags,   buf + 10, 2);
    std::memcpy(&h.n_vectors,    buf + 12, 8);
    std::memcpy(&h.dims,         buf + 20, 4);
    std::memcpy(&h.n_subspaces,  buf + 24, 2);
    std::memcpy(&h.metadata_len, buf + 26, 4);
    return h;
}

static void validate_magic(const VqzHeader& h) {
    if (std::memcmp(h.magic, VQZ_MAGIC, 8) != 0) {
        throw std::runtime_error("not a .vqz file: invalid magic bytes");
    }
}

// ---------------------------------------------------------------------------
// zlib decompression helper
// ---------------------------------------------------------------------------

static std::vector<uint8_t> zlib_decompress(
        const uint8_t* in, size_t in_size, size_t expected_out_size) {
    std::vector<uint8_t> out(expected_out_size);
    uLongf dest_len = static_cast<uLongf>(expected_out_size);
    int rc = uncompress(out.data(), &dest_len,
                        in,        static_cast<uLong>(in_size));
    if (rc != Z_OK) {
        throw std::runtime_error(
            std::string("zlib decompression failed: code ") + std::to_string(rc));
    }
    out.resize(dest_len);
    return out;
}

// ---------------------------------------------------------------------------
// VQZ file read helper — returns raw uncompressed body bytes
// ---------------------------------------------------------------------------

struct VqzData {
    uint64_t n;          // number of vectors
    uint32_t dims;       // dimension
    std::vector<int8_t>  quantized;   // shape [n * dims]
    std::vector<float>   scales;      // shape [n]
};

static VqzData read_vqz_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        throw std::runtime_error("cannot open file: " + path);
    }

    // Read header
    uint8_t hdr_buf[HEADER_SIZE];
    f.read(reinterpret_cast<char*>(hdr_buf), HEADER_SIZE);
    if (static_cast<size_t>(f.gcount()) < HEADER_SIZE) {
        throw std::runtime_error("file too short to contain a VQZ header");
    }

    VqzHeader h = parse_header(hdr_buf);
    validate_magic(h);

    // Skip optional metadata
    if (h.metadata_len > 0) {
        f.seekg(h.metadata_len, std::ios::cur);
    }

    // Read body into memory
    std::vector<uint8_t> body_raw;
    {
        const auto cur = f.tellg();
        f.seekg(0, std::ios::end);
        const auto end = f.tellg();
        f.seekg(cur);
        const size_t body_size = static_cast<size_t>(end - cur);
        body_raw.resize(body_size);
        f.read(reinterpret_cast<char*>(body_raw.data()), body_size);
    }

    // Expected uncompressed size: n * dims bytes (int8) + n * 4 bytes (float32)
    const size_t n = static_cast<size_t>(h.n_vectors);
    const size_t d = static_cast<size_t>(h.dims);
    const size_t expected = n * d + n * sizeof(float);

    std::vector<uint8_t> body;
    if (h.comp_flags == COMP_NONE) {
        body = std::move(body_raw);
        if (body.size() < expected) {
            throw std::runtime_error("VQZ body too short (raw mode)");
        }
    } else {
        // comp_flags 1 = zstd; fall back to zlib since we don't link libzstd
        // (Phase 1 CI does not ship libzstd; zlib is always available on all
        //  platforms as it is a Node.js dependency).  If the file was written
        //  with zstd, the decompressor will fail with an error — users should
        //  re-save with comp_flags=2 (zlib) or 0 (none) for Node.js use.
        body = zlib_decompress(body_raw.data(), body_raw.size(), expected);
    }

    VqzData out;
    out.n    = h.n_vectors;
    out.dims = h.dims;

    // Split body into quantized + scales
    const size_t quant_bytes  = n * d;
    const size_t scales_bytes = n * sizeof(float);
    if (body.size() < quant_bytes + scales_bytes) {
        throw std::runtime_error("decompressed body too short");
    }

    out.quantized.resize(quant_bytes);
    std::memcpy(out.quantized.data(), body.data(), quant_bytes);

    out.scales.resize(n);
    std::memcpy(out.scales.data(), body.data() + quant_bytes, scales_bytes);

    return out;
}

// ---------------------------------------------------------------------------
// dequantize
//
// Signature: dequantize(quantized: Int8Array, scales: Float32Array, dims: number)
//                       → Float32Array
//
// Reconstructs float32 vectors: out[i*d + j] = quantized[i*d + j] * scales[i]
// ---------------------------------------------------------------------------

Napi::Value Dequantize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    try {
        if (info.Length() < 3) {
            throw std::runtime_error("dequantize requires 3 arguments: (Int8Array, Float32Array, dims)");
        }
        if (!info[0].IsTypedArray() || !info[1].IsTypedArray() || !info[2].IsNumber()) {
            throw std::runtime_error("dequantize: arguments must be (Int8Array, Float32Array, number)");
        }

        auto q_ta  = info[0].As<Napi::TypedArray>();
        auto s_ta  = info[1].As<Napi::TypedArray>();
        size_t dims = static_cast<size_t>(info[2].As<Napi::Number>().Uint32Value());

        if (q_ta.TypedArrayType() != napi_int8_array) {
            throw std::runtime_error("dequantize: first argument must be Int8Array");
        }
        if (s_ta.TypedArrayType() != napi_float32_array) {
            throw std::runtime_error("dequantize: second argument must be Float32Array");
        }
        if (dims == 0) {
            throw std::runtime_error("dequantize: dims must be > 0");
        }

        const int8_t* q = reinterpret_cast<const int8_t*>(q_ta.ArrayBuffer().Data());
        const float * s = reinterpret_cast<const float *>(s_ta.ArrayBuffer().Data());
        const size_t  n = s_ta.ElementLength();

        if (q_ta.ElementLength() != n * dims) {
            throw std::runtime_error("dequantize: quantized length != n * dims");
        }

        // Allocate output Float32Array
        auto result_buf = Napi::ArrayBuffer::New(env, n * dims * sizeof(float));
        float* out = reinterpret_cast<float*>(result_buf.Data());

        // Scalar kernel: out[i*d + j] = q[i*d + j] * scales[i]
        // The compiler auto-vectorises this loop on -O2 with AVX2/NEON.
        for (size_t i = 0; i < n; ++i) {
            const float scale = s[i];
            const size_t base = i * dims;
            for (size_t j = 0; j < dims; ++j) {
                out[base + j] = static_cast<float>(q[base + j]) * scale;
            }
        }

        return Napi::Float32Array::New(env, n * dims, result_buf, 0);

    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

// ---------------------------------------------------------------------------
// readVqz
//
// Signature: readVqz(path: string) → { quantized: Int8Array, scales: Float32Array,
//                                       dims: number, n: number }
// ---------------------------------------------------------------------------

Napi::Value ReadVqz(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    try {
        if (info.Length() < 1 || !info[0].IsString()) {
            throw std::runtime_error("readVqz requires exactly one string argument (path)");
        }

        const std::string path = info[0].As<Napi::String>().Utf8Value();
        VqzData data = read_vqz_file(path);

        const size_t n = static_cast<size_t>(data.n);
        const size_t d = static_cast<size_t>(data.dims);

        // Wrap quantized in an Int8Array (copy into a new ArrayBuffer)
        auto q_buf = Napi::ArrayBuffer::New(env, n * d);
        std::memcpy(q_buf.Data(), data.quantized.data(), n * d);
        auto q_arr = Napi::Int8Array::New(env, n * d, q_buf, 0);

        // Wrap scales in a Float32Array
        auto s_buf = Napi::ArrayBuffer::New(env, n * sizeof(float));
        std::memcpy(s_buf.Data(), data.scales.data(), n * sizeof(float));
        auto s_arr = Napi::Float32Array::New(env, n, s_buf, 0);

        // Build result object
        Napi::Object result = Napi::Object::New(env);
        result.Set("quantized", q_arr);
        result.Set("scales",    s_arr);
        result.Set("dims",      Napi::Number::New(env, static_cast<double>(d)));
        result.Set("n",         Napi::Number::New(env, static_cast<double>(n)));

        return result;

    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

// ---------------------------------------------------------------------------
// VqzReader class
// ---------------------------------------------------------------------------

class VqzReader : public Napi::ObjectWrap<VqzReader> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "VqzReader", {
      InstanceMethod("read",  &VqzReader::Read),
      InstanceMethod("close", &VqzReader::Close),
    });
    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);
    exports.Set("VqzReader", func);
    return exports;
  }

  VqzReader(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<VqzReader>(info), path_(""), loaded_(false) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
      Napi::TypeError::New(env, "VqzReader constructor requires a path string")
          .ThrowAsJavaScriptException();
      return;
    }
    path_ = info[0].As<Napi::String>().Utf8Value();
  }

  Napi::Value Read(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    try {
      if (path_.empty()) {
        throw std::runtime_error("VqzReader: no path set");
      }
      data_   = read_vqz_file(path_);
      loaded_ = true;

      const size_t n = static_cast<size_t>(data_.n);
      const size_t d = static_cast<size_t>(data_.dims);

      auto q_buf = Napi::ArrayBuffer::New(env, n * d);
      std::memcpy(q_buf.Data(), data_.quantized.data(), n * d);
      auto q_arr = Napi::Int8Array::New(env, n * d, q_buf, 0);

      auto s_buf = Napi::ArrayBuffer::New(env, n * sizeof(float));
      std::memcpy(s_buf.Data(), data_.scales.data(), n * sizeof(float));
      auto s_arr = Napi::Float32Array::New(env, n, s_buf, 0);

      Napi::Object result = Napi::Object::New(env);
      result.Set("quantized", q_arr);
      result.Set("scales",    s_arr);
      result.Set("dims",      Napi::Number::New(env, static_cast<double>(d)));
      result.Set("n",         Napi::Number::New(env, static_cast<double>(n)));
      return result;

    } catch (const std::exception& ex) {
      Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
      return env.Undefined();
    }
  }

  Napi::Value Close(const Napi::CallbackInfo& info) {
    // Data is heap-allocated inside std::vector — just clear to release memory.
    data_ = VqzData{};
    loaded_ = false;
    return info.Env().Null();
  }

 private:
  std::string path_;
  VqzData     data_;
  bool        loaded_;
};

// ---------------------------------------------------------------------------
// Module initialisation
// ---------------------------------------------------------------------------

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("dequantize", Napi::Function::New(env, Dequantize));
  exports.Set("readVqz",    Napi::Function::New(env, ReadVqz));
  VqzReader::Init(env, exports);
  return exports;
}

NODE_API_MODULE(vectro_napi, Init)

