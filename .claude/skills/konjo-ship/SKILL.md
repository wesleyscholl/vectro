---
name: konjo-ship
description: Konjo sprint completion checklist. Use when closing out a sprint.
user-invocable: true
---
# Konjo Ship

## Sprint Completion Checklist
```
[ ] All tests pass — `cargo test` and `python -m pytest` green
[ ] `cargo clippy -- -D warnings` clean
[ ] `ruff check` and `ruff format --check` clean
[ ] CHANGELOG.md updated
[ ] PLAN.md updated
[ ] README.md reflects current state
[ ] git add && git commit -m "type(scope): description" && git push
```

## Session Handoff Template
```
SHIPPED      [what was completed]
TESTS        [passing / failing / count]
PUSHED       [commit hash]
NEXT SESSION [exact next task]
DISCOVERIES  [papers, repos, techniques found]
HEALTH       [Green / Yellow / Red]
```
