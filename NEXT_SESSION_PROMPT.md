# Vectro — Next Session Prompt

Read this first. It takes under a minute and prevents context drift.

## Current State (as of 2026-04-23)

- Version: Python 4.11.2 / Rust 7.4.0
- Branch: main
- Latest tagged release: v4.11.1 (tag; code is at 4.11.2 pending tag)
- Test status: 792 passed, 1 skipped, 0 failed

## What was done in the latest wave

1. Test hygiene hardening — shared path-helper migration COMPLETE:
- Migrated all 29 remaining test files from inline `sys.path.insert(...)` to
  `tests/_path_setup.ensure_repo_root_on_path()`.
- Files using `python/`-relative bare imports (test_binary.py, test_nf4.py, test_pq.py,
  test_hnsw.py, test_storage_v3.py) converted to `from python.xxx import` prefix.
- All in-body bare `import storage_v3` calls updated to
  `import python.storage_v3 as storage_v3` for monkeypatch correctness.
- Zero `sys.path` mutations remain in any test file outside `tests/_path_setup.py`.

## Active invariants to respect

- INT8 throughput floor remains >=10M vec/s on M3 for Mojo SIMD path.
- Quantization quality floors in CLAUDE.md are hard gates.
- No dequantize-then-matmul shortcuts for quantized math paths.
- Documentation must be updated in the same prompt cycle as code changes.

## Known blockers / open risks

- CLAUDE.md and AGENTS.md still contain historical roadmap rows with older test counts by version (intentional historical records). Do not rewrite historical entries unless facts are wrong.

## Next high-value tasks

1. Benchmark reproducibility pass
- Re-run canonical benchmark commands and ensure benchmark docs remain consistent with measured data.

2. ADR-002 execution audit
- Verify all v4.1 implementation gates referenced by `docs/adr-002-v4-architecture.md`
  are either complete or tracked with explicit blockers (single-shot latency gate,
  wasm-pack CI gate, profile registry test coverage).

3. Version tag
- Tag v4.11.2 in git once benchmark docs are confirmed accurate.

## Commands

- Full tests:
  python3 -m pytest tests/ -q --timeout=120

- Mojo benchmark:
  pixi run benchmark

- Rust tests:
  cargo test --workspace --locked
