# Git Workflow
- Conventional Commits: `type(scope): description`
- `cargo build` and `cargo test` must be green before committing
- `python -m pytest` must also be green for Python components
- `ruff check` and `cargo clippy -- -D warnings` must be clean
- Never commit with known failing tests
