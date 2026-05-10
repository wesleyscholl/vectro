---
paths:
  - "**/*.rs"
---
# Rust Conventions

- No `unwrap()` or `expect()` outside test code — use `anyhow::Result` and `?`
- No blocking I/O on async paths — `tokio::task::spawn_blocking` for synchronous ops
- No silent failures — `tracing::warn!` when a fallback path swallows an error
- Tokio is the only async runtime — never introduce another executor
- Error types: `thiserror` for library crates, `anyhow` for binary/glue code
- Prefer `Arc<T>` for read-heavy shared state; `dashmap` over `Mutex<HashMap>` for concurrent maps
- Document every `unsafe` block: state the invariant it relies on
- `cargo clippy -- -D warnings` must be clean before shipping
- Memory leaks are bugs — for long-running server paths, assert RSS doesn't grow monotonically under load
