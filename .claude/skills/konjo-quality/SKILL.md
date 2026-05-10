---
name: konjo-quality
description: Konjo Code Quality Framework — Three-Wall framework to prevent AI slop. Auto-load when writing tests, reviewing code quality, refactoring, or on quality gate failures.
user-invocable: true
---

# Konjo Quality Framework — Agent Reference

## The Three Walls

| Wall | When | Blocks |
|------|------|--------|
| Wall 1 | Pre-commit | The commit |
| Wall 2 | CI / GitHub Actions | The merge |
| Wall 3 | Claude Opus review | The merge |

## Thresholds

| Metric | Hard Block | Target |
|--------|-----------|--------|
| Line coverage | ≥ 80% | ≥ 95% |
| Mutation survival | ≤ 10% | 0% |
| Cognitive complexity | ≤ 15 | ≤ 10 |
| File length | ≤ 500L | ≤ 300L |
| DRY violations | 0 | 0 |
| unwrap() non-test | 0 | 0 |

## Running Locally
```bash
cargo clippy --workspace -- -D warnings -D clippy::unwrap_used
cargo llvm-cov nextest --workspace --fail-under-lines 80
python3 .konjo/scripts/dry_check.py --staged-only
git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py
```
