---
paths:
  - "**/*_test.rs"
  - "**/tests/**"
  - "**/test_*.rs"
  - "**/spec/**"
---
# Testing Rules

A sprint is NEVER complete until all tests pass.
100% coverage is the floor — every code file needs a corresponding test file.

**Unit:** deterministic, isolated functions.
**Integration:** crate interactions, DB boundaries, API handoffs.
**E2E:** full agent loop end-to-end — no mocking the git manager, memory store, or Claude runner.
**CLI:** new flags must be tested for expected output and failure modes.

Anti-mocking rule: E2E tests must test reality. Never mock the DB or network in E2E tests.
Never commit with known failing tests. `cargo test` must be green before `git push`.
