/**
 * vectro_napi.cpp ― Vectro N-API native addon stub (ADR-001 Phase 1).
 *
 * This file establishes the N-API surface and project structure as specified in
 * docs/adr-001-javascript-bindings.md.  The function bodies intentionally throw
 * "not yet implemented" errors — full implementation is tracked as ADR-001 Phase 2.
 *
 * Public N-API surface:
 *   dequantize(quantized: Int8Array, scales: Float32Array, dims: number) → Float32Array
 *   readVqz(path: string) → { quantized, scales, dims, n }
 *   VqzReader class: constructor(path), read() → VqzData, close()
 *
 * Build: npm run build  (requires node-gyp and a C++17 compiler)
 * Platform targets: darwin-arm64, darwin-x64, linux-x64, linux-arm64, win32-x64
 */

#include <napi.h>
#include <string>

// ---------------------------------------------------------------------------
// dequantize
// ---------------------------------------------------------------------------

Napi::Value Dequantize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  // TODO (ADR-001 Phase 2): Parse Int8Array + Float32Array, run SIMD multiply,
  // return Float32Array without copying via ArrayBuffer::NewExternalBuffer.
  Napi::Error::New(env,
    "dequantize is not yet implemented — see docs/adr-001-javascript-bindings.md")
    .ThrowAsJavaScriptException();
  return env.Undefined();
}

// ---------------------------------------------------------------------------
// readVqz
// ---------------------------------------------------------------------------

Napi::Value ReadVqz(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  // TODO (ADR-001 Phase 2): Parse 64-byte VQZ header, decompress body with
  // statically-linked libzstd, split into Int8Array + Float32Array, return object.
  Napi::Error::New(env,
    "readVqz is not yet implemented — see docs/adr-001-javascript-bindings.md")
    .ThrowAsJavaScriptException();
  return env.Undefined();
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
      : Napi::ObjectWrap<VqzReader>(info) {
    // TODO (ADR-001 Phase 2): open file handle, validate magic bytes.
    Napi::Error::New(info.Env(),
      "VqzReader is not yet implemented — see docs/adr-001-javascript-bindings.md")
      .ThrowAsJavaScriptException();
  }

  Napi::Value Read(const Napi::CallbackInfo& info) {
    Napi::Error::New(info.Env(), "VqzReader::read — not yet implemented")
      .ThrowAsJavaScriptException();
    return info.Env().Undefined();
  }

  Napi::Value Close(const Napi::CallbackInfo& info) {
    Napi::Error::New(info.Env(), "VqzReader::close — not yet implemented")
      .ThrowAsJavaScriptException();
    return info.Env().Undefined();
  }
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
