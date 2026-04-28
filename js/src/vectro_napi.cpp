/**
 * vectro_napi.cpp — Vectro N-API native addon (ADR-001 Phase 2).
 *
 * Full implementation of the .vqz parser and INT8 dequantization kernel.
 * See docs/adr-001-javascript-bindings.md for architecture decisions.
 *
 * Public N-API surface:
 *   parseHeader(buffer: Buffer)                              → VqzHeader object
 *   parseBody(buffer: Buffer, n: number, dims: number)       → VqzData object
 *   dequantize(q: Int8Array, s: Float32Array, dims: number)  → Float32Array
 *   readVqz(path: string)                                    → VqzData object
 *   VqzReader class: constructor(path), read(), close()
 *
 * Build:  npm run build   (requires node-gyp, C++17, libzstd, zlib)
 *   macOS:  brew install zstd
 *   Linux:  apt-get install libzstd-dev zlib1g-dev
 * Platform targets: darwin-arm64, darwin-x64, linux-x64, linux-arm64, win32-x64
 */

#include <napi.h>
#include <zlib.h>
#include <zstd.h>
#include <zstd_errors.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// VQZ format constants — must match python/storage_v3.py exactly.
// Header layout (64 bytes, all fields little-endian):
//   [0:8]   magic       b"VECTRO\x03\x00"
//   [8:10]  version     uint16  (currently 1)
//   [10:12] comp_flag   uint16  (0=none, 1=zstd, 2=zlib)
//   [12:20] n_vectors   uint64
//   [20:24] dims        uint32
//   [24:26] n_subspaces uint16
//   [26:30] metadata_len uint32
//   [30:38] checksum    8 bytes blake2b (informational, not verified here)
//   [38:64] reserved    zero
// Body (uncompressed): int8[n*dims] || float32[n]
// ---------------------------------------------------------------------------

static const uint8_t VQZ_MAGIC[8] = {'V', 'E', 'C', 'T', 'R', 'O', 0x03, 0x00};
static constexpr size_t   HEADER_SIZE = 64;
static constexpr uint16_t COMP_NONE   = 0;
static constexpr uint16_t COMP_ZSTD   = 1;
static constexpr uint16_t COMP_ZLIB   = 2;

// ---------------------------------------------------------------------------
// Little-endian reader helpers
// ---------------------------------------------------------------------------

static inline uint16_t le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
static inline uint32_t le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}
static inline uint64_t le64(const uint8_t* p) {
    return static_cast<uint64_t>(le32(p)) | (static_cast<uint64_t>(le32(p + 4)) << 32);
}

// ---------------------------------------------------------------------------
// VQZ header struct
// ---------------------------------------------------------------------------

struct VqzHeader {
    uint16_t version;
    uint16_t comp_flag;
    uint64_t n_vectors;
    uint32_t dims;
    uint16_t n_subspaces;
    uint32_t metadata_len;
};

// Throws std::runtime_error on invalid or truncated data.
static VqzHeader parse_header_raw(const uint8_t* raw, size_t len) {
    if (len < HEADER_SIZE) {
        throw std::runtime_error(
            "Buffer too small for VQZ header: need 64 bytes, got " +
            std::to_string(len));
    }
    if (std::memcmp(raw, VQZ_MAGIC, 8) != 0) {
        throw std::runtime_error("Not a valid VQZ file: magic mismatch");
    }
    VqzHeader h;
    h.version      = le16(raw + 8);
    h.comp_flag    = le16(raw + 10);
    h.n_vectors    = le64(raw + 12);
    h.dims         = le32(raw + 20);
    h.n_subspaces  = le16(raw + 24);
    h.metadata_len = le32(raw + 26);
    return h;
}

// ---------------------------------------------------------------------------
// Decompression (zstd + zlib)
// ---------------------------------------------------------------------------

static std::vector<uint8_t> decompress_body(const uint8_t* data, size_t len,
                                             uint16_t comp_flag) {
    if (comp_flag == COMP_NONE) {
        return std::vector<uint8_t>(data, data + len);
    }

    if (comp_flag == COMP_ZSTD) {
        unsigned long long frame_size = ZSTD_getFrameContentSize(data, len);
        if (frame_size == ZSTD_CONTENTSIZE_ERROR) {
            throw std::runtime_error("zstd: compressed frame appears corrupt");
        }
        // ZSTD_CONTENTSIZE_UNKNOWN: content size was not stored; grow as needed.
        size_t out_cap = (frame_size == ZSTD_CONTENTSIZE_UNKNOWN)
                             ? len * 8
                             : static_cast<size_t>(frame_size);
        for (int attempt = 0; attempt < 8; ++attempt) {
            std::vector<uint8_t> out(out_cap);
            size_t result = ZSTD_decompress(out.data(), out_cap, data, len);
            if (!ZSTD_isError(result)) {
                out.resize(result);
                return out;
            }
            if (ZSTD_getErrorCode(result) == ZSTD_error_dstSize_tooSmall) {
                out_cap *= 2;
            } else {
                throw std::runtime_error(
                    std::string("zstd decompress error: ") +
                    ZSTD_getErrorName(result));
            }
        }
        throw std::runtime_error("zstd: output size exceeded 256× compressed size");
    }

    if (comp_flag == COMP_ZLIB) {
        z_stream strm{};
        if (inflateInit(&strm) != Z_OK) {
            throw std::runtime_error("zlib inflateInit failed");
        }
        strm.next_in  = const_cast<Bytef*>(data);
        strm.avail_in = static_cast<uInt>(len);
        std::vector<uint8_t> out;
        uint8_t chunk[65536];
        int ret = Z_OK;
        while (ret != Z_STREAM_END) {
            strm.next_out  = chunk;
            strm.avail_out = sizeof(chunk);
            ret = inflate(&strm, Z_NO_FLUSH);
            if (ret != Z_OK && ret != Z_STREAM_END) {
                inflateEnd(&strm);
                throw std::runtime_error(
                    "zlib inflate failed (code=" + std::to_string(ret) + ")");
            }
            out.insert(out.end(), chunk,
                       chunk + sizeof(chunk) - strm.avail_out);
        }
        inflateEnd(&strm);
        return out;
    }

    throw std::runtime_error(
        "Unknown comp_flag: " + std::to_string(comp_flag) +
        " (expected 0=none, 1=zstd, 2=zlib)");
}

// ---------------------------------------------------------------------------
// Build VqzData JS object from decoded quantized codes + scale factors.
// quantized and scales share a single ArrayBuffer aligned for Float32Array.
// ---------------------------------------------------------------------------

static Napi::Object make_vqz_data(Napi::Env env,
                                   const uint8_t* q_data,
                                   const float*   s_data,
                                   size_t n, size_t dims) {
    size_t qb = n * dims;
    // Float32Array requires 4-byte-aligned offset; pad quantized section if needed.
    size_t qb_padded = (qb + 3) & ~static_cast<size_t>(3);
    size_t total     = qb_padded + n * sizeof(float);

    Napi::ArrayBuffer ab  = Napi::ArrayBuffer::New(env, total);
    auto*             raw = static_cast<uint8_t*>(ab.Data());
    std::memcpy(raw,            q_data, qb);
    std::memcpy(raw + qb_padded, s_data, n * sizeof(float));

    Napi::Int8Array   quantized = Napi::Int8Array::New(env, qb, ab, 0);
    Napi::Float32Array scales   = Napi::Float32Array::New(env, n, ab, qb_padded);

    Napi::Object obj = Napi::Object::New(env);
    obj.Set("quantized", quantized);
    obj.Set("scales",    scales);
    obj.Set("n",         Napi::Number::New(env, static_cast<double>(n)));
    obj.Set("dims",      Napi::Number::New(env, static_cast<double>(dims)));
    return obj;
}

// ---------------------------------------------------------------------------
// parseHeader(buffer: Buffer)
//   → { version, compFlag, n, dims, nSubspaces, metadataLen }
// ---------------------------------------------------------------------------

Napi::Value ParseHeader(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsBuffer()) {
        Napi::TypeError::New(env, "parseHeader: expected Buffer argument")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    try {
        auto      buf = info[0].As<Napi::Buffer<uint8_t>>();
        VqzHeader h   = parse_header_raw(buf.Data(), buf.ByteLength());
        Napi::Object obj = Napi::Object::New(env);
        obj.Set("version",     Napi::Number::New(env, h.version));
        obj.Set("compFlag",    Napi::Number::New(env, h.comp_flag));
        obj.Set("n",           Napi::Number::New(env, static_cast<double>(h.n_vectors)));
        obj.Set("dims",        Napi::Number::New(env, h.dims));
        obj.Set("nSubspaces",  Napi::Number::New(env, h.n_subspaces));
        obj.Set("metadataLen", Napi::Number::New(env, h.metadata_len));
        return obj;
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

// ---------------------------------------------------------------------------
// parseBody(buffer: Buffer, n: number, dims: number) → VqzData
//
// Splits a decompressed VQZ body into Int8Array (codes) + Float32Array (scales).
// Body layout: int8[n*dims] || float32[n]  — no header, no padding, C row-major.
// ---------------------------------------------------------------------------

Napi::Value ParseBody(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[0].IsBuffer() ||
        !info[1].IsNumber() || !info[2].IsNumber()) {
        Napi::TypeError::New(env,
            "parseBody: expected (Buffer, n: number, dims: number)")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    try {
        auto   buf  = info[0].As<Napi::Buffer<uint8_t>>();
        size_t n    = static_cast<size_t>(info[1].As<Napi::Number>().Int64Value());
        size_t dims = static_cast<size_t>(info[2].As<Napi::Number>().Uint32Value());
        size_t qb   = n * dims;
        size_t expected = qb + n * sizeof(float);
        if (buf.ByteLength() < expected) {
            throw std::runtime_error(
                "parseBody: buffer too small — need " + std::to_string(expected) +
                " bytes for " + std::to_string(n) + " vectors × " +
                std::to_string(dims) + " dims, got " +
                std::to_string(buf.ByteLength()));
        }
        const auto* raw = buf.Data();
        return make_vqz_data(env, raw,
                             reinterpret_cast<const float*>(raw + qb),
                             n, dims);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

// ---------------------------------------------------------------------------
// dequantize(quantized: Int8Array, scales: Float32Array, dims: number)
//            → Float32Array
//
// out[i*dims + j] = (float)quantized[i*dims + j] * scales[i]
//
// ARM64: 16-element NEON loop (int8 → int16 → int32 → float32 × scale).
// x86-64 / other: scalar loop; compiler auto-vectorizes at -O3.
// ---------------------------------------------------------------------------

Napi::Value Dequantize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 ||
        !info[0].IsTypedArray() || !info[1].IsTypedArray() || !info[2].IsNumber()) {
        Napi::TypeError::New(env,
            "dequantize: expected (Int8Array, Float32Array, dims: number)")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    try {
        auto   quantized = info[0].As<Napi::Int8Array>();
        auto   scales    = info[1].As<Napi::Float32Array>();
        size_t dims      = static_cast<size_t>(info[2].As<Napi::Number>().Uint32Value());
        size_t n         = scales.ElementLength();

        if (quantized.ElementLength() != n * dims) {
            throw std::runtime_error(
                "dequantize: quantized.length (" +
                std::to_string(quantized.ElementLength()) +
                ") must equal scales.length (" + std::to_string(n) +
                ") * dims (" + std::to_string(dims) + ")");
        }

        auto out = Napi::Float32Array::New(env, n * dims);

        const int8_t* qp = quantized.Data();
        const float*  sp = scales.Data();
        float*        op = out.Data();

        for (size_t i = 0; i < n; ++i) {
            const float   scale = sp[i];
            const int8_t* row_q = qp + i * dims;
            float*        row_o = op + i * dims;
            size_t j = 0;

#ifdef __ARM_NEON
            float32x4_t vscale = vdupq_n_f32(scale);
            // Process 16 int8 elements per iteration.
            for (; j + 16 <= dims; j += 16) {
                int8x16_t vq   = vld1q_s8(row_q + j);
                // Sign-extend int8 → int16 (two 8-lane halves)
                int16x8_t lo16 = vmovl_s8(vget_low_s8(vq));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(vq));
                // Sign-extend int16 → int32 (four 4-lane quarters)
                int32x4_t q0   = vmovl_s16(vget_low_s16(lo16));
                int32x4_t q1   = vmovl_s16(vget_high_s16(lo16));
                int32x4_t q2   = vmovl_s16(vget_low_s16(hi16));
                int32x4_t q3   = vmovl_s16(vget_high_s16(hi16));
                // Convert int32 → float32 and multiply by per-vector scale.
                vst1q_f32(row_o + j,      vmulq_f32(vcvtq_f32_s32(q0), vscale));
                vst1q_f32(row_o + j + 4,  vmulq_f32(vcvtq_f32_s32(q1), vscale));
                vst1q_f32(row_o + j + 8,  vmulq_f32(vcvtq_f32_s32(q2), vscale));
                vst1q_f32(row_o + j + 12, vmulq_f32(vcvtq_f32_s32(q3), vscale));
            }
#endif
            // Scalar remainder — auto-vectorized by Clang/GCC at -O3
            // (generates SSE/AVX on x86-64, remaining NEON lanes on ARM64).
            for (; j < dims; ++j) {
                row_o[j] = static_cast<float>(row_q[j]) * scale;
            }
        }
        return out;
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

// ---------------------------------------------------------------------------
// readVqz(path: string) → VqzData
// ---------------------------------------------------------------------------

Napi::Value ReadVqz(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "readVqz: expected string path")
            .ThrowAsJavaScriptException();
        return env.Null();
    }
    try {
        std::string path = info[0].As<Napi::String>().Utf8Value();

        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            throw std::runtime_error("readVqz: cannot open file: " + path);
        }
        auto file_size = static_cast<size_t>(f.tellg());
        f.seekg(0, std::ios::beg);

        std::vector<uint8_t> file_data(file_size);
        f.read(reinterpret_cast<char*>(file_data.data()),
               static_cast<std::streamsize>(file_size));
        f.close();

        VqzHeader h          = parse_header_raw(file_data.data(), file_size);
        size_t    body_start = HEADER_SIZE + h.metadata_len;

        if (body_start > file_size) {
            throw std::runtime_error(
                "readVqz: file truncated before body (file_size=" +
                std::to_string(file_size) + ", body_start=" +
                std::to_string(body_start) + ")");
        }

        std::vector<uint8_t> body = decompress_body(
            file_data.data() + body_start,
            file_size - body_start,
            h.comp_flag);

        size_t n   = static_cast<size_t>(h.n_vectors);
        size_t dims = h.dims;
        size_t qb  = n * dims;

        if (body.size() < qb + n * sizeof(float)) {
            throw std::runtime_error(
                "readVqz: decompressed body (" + std::to_string(body.size()) +
                " bytes) is smaller than declared " + std::to_string(n) +
                " vectors × " + std::to_string(dims) + " dims");
        }

        return make_vqz_data(env, body.data(),
                             reinterpret_cast<const float*>(body.data() + qb),
                             n, dims);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

// ---------------------------------------------------------------------------
// VqzReader — object-style handle for reading a .vqz file
// ---------------------------------------------------------------------------

class VqzReader : public Napi::ObjectWrap<VqzReader> {
 public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports) {
        Napi::Function func = DefineClass(env, "VqzReader", {
            InstanceMethod("read",  &VqzReader::Read),
            InstanceMethod("close", &VqzReader::Close),
        });
        exports.Set("VqzReader", func);
        return exports;
    }

    explicit VqzReader(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<VqzReader>(info) {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsString()) {
            Napi::TypeError::New(env, "VqzReader: expected string path")
                .ThrowAsJavaScriptException();
            return;
        }
        path_ = info[0].As<Napi::String>().Utf8Value();
    }

    Napi::Value Read(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();
        if (path_.empty()) {
            Napi::Error::New(env,
                "VqzReader.read: reader is closed or was never opened")
                .ThrowAsJavaScriptException();
            return env.Null();
        }
        try {
            std::ifstream f(path_, std::ios::binary | std::ios::ate);
            if (!f.is_open()) {
                throw std::runtime_error(
                    "VqzReader.read: cannot open file: " + path_);
            }
            auto file_size = static_cast<size_t>(f.tellg());
            f.seekg(0, std::ios::beg);
            std::vector<uint8_t> file_data(file_size);
            f.read(reinterpret_cast<char*>(file_data.data()),
                   static_cast<std::streamsize>(file_size));
            f.close();

            VqzHeader h          = parse_header_raw(file_data.data(), file_size);
            size_t    body_start = HEADER_SIZE + h.metadata_len;
            if (body_start > file_size) {
                throw std::runtime_error(
                    "VqzReader.read: file truncated before body");
            }
            std::vector<uint8_t> body = decompress_body(
                file_data.data() + body_start,
                file_size - body_start,
                h.comp_flag);

            size_t n   = static_cast<size_t>(h.n_vectors);
            size_t dims = h.dims;
            size_t qb  = n * dims;
            if (body.size() < qb + n * sizeof(float)) {
                throw std::runtime_error(
                    "VqzReader.read: decompressed body too small");
            }
            return make_vqz_data(env, body.data(),
                                 reinterpret_cast<const float*>(body.data() + qb),
                                 n, dims);
        } catch (const std::exception& ex) {
            Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
            return env.Null();
        }
    }

    Napi::Value Close(const Napi::CallbackInfo& info) {
        path_.clear();
        return info.Env().Undefined();
    }

 private:
    std::string path_;
};

// ---------------------------------------------------------------------------
// Module initialisation
// ---------------------------------------------------------------------------

Napi::Object ModuleInit(Napi::Env env, Napi::Object exports) {
    exports.Set("parseHeader", Napi::Function::New(env, ParseHeader));
    exports.Set("parseBody",   Napi::Function::New(env, ParseBody));
    exports.Set("dequantize",  Napi::Function::New(env, Dequantize));
    exports.Set("readVqz",     Napi::Function::New(env, ReadVqz));
    VqzReader::Init(env, exports);
    return exports;
}

NODE_API_MODULE(vectro_napi, ModuleInit)
