# Vectro — Next Session Prompt

Read this first. It takes under a minute and prevents context drift.

## Current State (as of 2026-04-22)

- Version: Python 4.11.2 / Rust 7.4.0
- Branch: main
- Latest tagged release: v4.11.1 (tag; code is at 4.11.2 pending tag)
- Test status: 792 passed, 1 skipped, 0 failed

## What was done in the latest wave

1. ADR-002 CI gate closure:
- Decision 1: hardened `latency-gate` job in `.github/workflows/ci.yml`.
  It now builds `vectro_py` in release mode and executes
  `tests/test_latency_singleshot.py -k "test_p99"`.
- Decision 2: `.github/workflows/wasm.yml` now runs
  `wasm-pack test --headless --chrome -- --lib` before WASM build/size checks.
- Added 11 wasm browser tests in `rust/vectro_lib/src/wasm.rs` plus
  `wasm-bindgen-test` dev-dependency in `rust/vectro_lib/Cargo.toml`.

2. Test hygiene hardening COMPLETE:
- Migrated all 29 remaining test files from inline `sys.path.insert(...)` to
  `tests/_path_setup.ensure_repo_root_on_path()`.
- Files using `python/`-relative bare imports (test_binary.py, test_nf4.py, test_pq.py,
  test_hnsw.py, test_storage_v3.py) converted to `from python.xxx import` prefix.
- All in-body bare `import storage_v3` calls updated to
  `import python.storage_v3 as storage_v3` for monkeypatch correctness.
- Zero `sys.path` mutations remain in any test file outside `tests/_path_setup.py`.

3. Version consistency fix:
- Bumped lingering `4.11.1` version constants/files to `4.11.2` in all
  release-gated locations.

## Active invariants to respect

- INT8 throughput floor remains >=10M vec/s on M3 for Mojo SIMD path.
- Quantization quality floors in CLAUDE.md are hard gates.
- No dequantize-then-matmul shortcuts for quantized math paths.
- Documentation must be updated in the same prompt cycle as code changes.

## Known blockers / open risks

- CLAUDE.md and AGENTS.md still contain historical roadmap rows with older test counts by version (intentional historical records). Do not rewrite historical entries unless facts are wrong.
- `latency-gate` runs on shared GitHub runners; occasional tail-latency jitter is possible.
  If flakes appear, move this gate to dedicated/self-hosted hardware rather than relaxing p99 contract.

## Next high-value tasks

1. Benchmark reproducibility pass
- Re-run canonical benchmark commands and ensure benchmark docs remain consistent with measured data.

2. CI observation pass
- Watch first CI runs with new WASM and latency gates to confirm no flaky behavior.
  If stable, keep strict gate.

3. Version tag
- Tag v4.11.2 in git once benchmark docs are confirmed accurate and CI is green.

## Commands

- Full tests:
  python3 -m pytest tests/ -q --timeout=120

- Mojo benchmark:
  pixi run benchmark

- Rust tests:
  cargo test --workspace --locked
