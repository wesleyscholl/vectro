{
  "targets": [
    {
      "target_name": "vectro_napi",
      "sources": ["src/vectro_napi.cpp"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags_cc": ["-O3", "-std=c++17"],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
      "conditions": [
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": "1",
              "Optimization": "3"
            }
          },
          "link_settings": {
            "libraries": ["zlib.lib", "zstd.lib"]
          }
        }],
        ["OS=='mac'", {
          "xcode_settings": {
            "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
            "CLANG_CXX_LIBRARY": "libc++",
            "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
            "MACOSX_DEPLOYMENT_TARGET": "11.0",
            "OTHER_CPLUSPLUSFLAGS": ["-O3"]
          },
          "include_dirs": [
            "<!@(node -p \"require('node-addon-api').include\")",
            "<!(brew --prefix zstd)/include"
          ],
          "link_settings": {
            "libraries": [
              "-lz",
              "<!(brew --prefix zstd)/lib/libzstd.dylib"
            ]
          }
        }],
        ["OS=='linux'", {
          "cflags_cc": ["-O3", "-std=c++17"],
          "link_settings": {
            "libraries": ["-lz", "-lzstd"]
          }
        }]
      ]
    }
  ]
}
