# Vectro — Next Session Prompt

Read this first. It takes under a minute and prevents context drift.

## Current State (as of 2026-04-22)

- Version: Python 4.11.1 / Rust 7.4.0
- Branch: main
- Latest tagged release: v4.11.1
- Test status: 792 passed, 1 skipped, 0 failed

## What was done in the latest wave

1. Binary batch profile correctness fixed in python/batch_api.py:
- profile="binary" now routes to binary_api.quantize_binary() (no INT8 fallback)
- reported compression ratio corrected to ~32x
- reconstruct_vector() no longer touches scales in binary mode (fixed IndexError)

2. Documentation contract drift reduced:
- CLAUDE.md project identity, baseline tests, and roadmap header synced
- AGENTS.md baseline tests and roadmap header synced
- README version/test badges synced to 4.11.1 and 792 tests
- PLAN.md and CHANGELOG.md stale test-count references corrected to 792

3. Roadmap execution guidance corrected:
- CLAUDE.md and AGENTS.md now explicitly mark v5.0/v8.0 ADR gate as complete.
- Removed stale "ADR drafting" as a next task now that `docs/adr-002-v4-architecture.md` is already accepted.

4. Test hygiene hardening increment:
- Added `tests/test_sklearn_subprocess_isolation.py` to execute sklearn-backed
  RQ/v3 paths in fresh subprocesses and guard against in-process C-extension reload risks.

## Active invariants to respect

- INT8 throughput floor remains >=10M vec/s on M3 for Mojo SIMD path.
- Quantization quality floors in CLAUDE.md are hard gates.
- No dequantize-then-matmul shortcuts for quantized math paths.
- Documentation must be updated in the same prompt cycle as code changes.

## Known blockers / open risks

- CLAUDE.md and AGENTS.md still contain historical roadmap rows with older test counts by version (intentional historical records). Do not rewrite historical entries unless facts are wrong.
- Legacy suites still use repo-root `sys.path.insert(...)` patterns; standardizing on a shared
  import helper remains open.

## Next high-value tasks

1. Continue test hygiene hardening
- Audit high-churn suites (`test_arrow_bridge.py`, `test_torch_bridge.py`) for process-state
  mutations and migrate fragile import-behavior checks to subprocess boundaries where appropriate.

2. Benchmark reproducibility pass
- Re-run canonical benchmark commands and ensure benchmark docs remain consistent with measured data.

3. ADR-002 execution audit
- Verify all v4.1 implementation gates referenced by `docs/adr-002-v4-architecture.md`
  are either complete or tracked with explicit blockers (single-shot latency gate,
  wasm-pack CI gate, profile registry test coverage).

## Commands

- Full tests:
  python3 -m pytest tests/ -q --timeout=120

- Mojo benchmark:
  pixi run benchmark

- Rust tests:
  cargo test --workspace --locked
