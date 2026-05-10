---
paths: ["**/*_test.rs", "**/tests/**", "**/test_*.py"]
---
# Testing Rules
Every code file needs a corresponding test file.
Rust: `cargo test` must be green. Python: `python -m pytest` must be green.
PyO3 bindings: test both Rust side and Python side independently.
Python-only mode is always the correctness baseline for SIMD/FFI code.
Never commit with known failing tests.
