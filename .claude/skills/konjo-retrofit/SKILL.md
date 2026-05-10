---
name: konjo-retrofit
description: Retrofit the Konjo Quality Framework onto an existing repo. Use when asked to add konjo quality gates, audit an existing codebase, or run a quality sprint.
user-invocable: true
---

# Konjo Retrofit Protocol

1. Baseline Audit — measure everything, fix nothing yet
2. Triage — P0 critical, P1 debt, P2 style
3. Install at current baseline minus 2%
4. Coverage ratchet: +5% per sprint to 80% floor
5. Complexity ratchet: one function at a time
6. DRY cleanup: highest similarity first
7. Wall 3: soft-fail week 1, blocking week 2

## Hybrid Checklist (vectro)
- [ ] `cargo audit` clean
- [ ] `cargo deny check` clean
- [ ] `cargo llvm-cov` ≥ 80%
- [ ] `clippy -D unwrap_used` passes
- [ ] `ruff check` and `ruff format --check` clean
- [ ] `mypy --strict` clean
- [ ] Python-only mode is always the correctness baseline
- [ ] Rust/Mojo acceleration matches Python numerically
- [ ] PyO3 bindings tested from both sides
