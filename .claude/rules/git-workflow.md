# Git Workflow

- Conventional Commits: `type(scope): description`
  - Types: feat, fix, refactor, test, docs, chore, perf
- `git add && git commit && git push` after every completed sprint
- `cargo build` and `cargo test` must be green before committing
- Never suppress command output — failures must be visible in real time
- Never commit with known failing tests
