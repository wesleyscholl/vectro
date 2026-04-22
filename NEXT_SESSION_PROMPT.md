# Vectro — Next Session Prompt

Read this first. It takes under a minute and prevents context drift.

## Current State (as of 2026-04-22)

- Version: Python 4.11.1 / Rust 7.4.0
- Branch: main
- Latest tagged release: v4.11.1
- Test status: 789 passed, 1 skipped, 0 failed

## What was done in the latest wave

1. Binary batch profile correctness fixed in python/batch_api.py:
- profile="binary" now routes to binary_api.quantize_binary() (no INT8 fallback)
- reported compression ratio corrected to ~32x
- reconstruct_vector() no longer touches scales in binary mode (fixed IndexError)

2. Documentation contract drift reduced:
- CLAUDE.md project identity, baseline tests, and roadmap header synced
- AGENTS.md baseline tests and roadmap header synced
- README version/test badges synced to 4.11.1 and 789 tests
- PLAN.md and CHANGELOG.md stale 790 references corrected to 789

## Active invariants to respect

- INT8 throughput floor remains >=10M vec/s on M3 for Mojo SIMD path.
- Quantization quality floors in CLAUDE.md are hard gates.
- No dequantize-then-matmul shortcuts for quantized math paths.
- Documentation must be updated in the same prompt cycle as code changes.

## Known blockers / open risks

- CLAUDE.md and AGENTS.md still contain historical roadmap rows with older test counts by version (intentional historical records). Do not rewrite historical entries unless facts are wrong.
- Pre-existing sklearn C-extension subprocess isolation debt remains in tests/test_rq.py and tests/test_v3_api.py notes.

## Next high-value tasks

1. v5.0/v8.0 architecture ADR drafting
- Define LLM embedding pipeline (<1 ms), WASM target, and Rust CLI direction.

2. Test hygiene hardening
- Audit and isolate any remaining process-state-sensitive tests via subprocess patterns.

3. Benchmark reproducibility pass
- Re-run canonical benchmark commands and ensure benchmark docs remain consistent with measured data.

## Commands

- Full tests:
  python3 -m pytest tests/ -q --timeout=120

- Mojo benchmark:
  pixi run benchmark

- Rust tests:
  cargo test --workspace --locked
