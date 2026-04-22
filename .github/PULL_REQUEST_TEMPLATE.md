## Summary
<!-- What does this PR change and why? -->

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactor / cleanup
- [ ] Python / WASM binding change

## Checklist
- [ ] `cargo test` passes locally
- [ ] `cargo clippy -- -D warnings` reports no errors
- [ ] `cargo fmt --check` passes
- [ ] `pytest python/tests/` passes locally (if Python bindings changed)
- [ ] `ruff check python/` reports no errors (if Python changed)
- [ ] No hardcoded absolute paths (no `/Users/<name>/...`, no `/home/<name>/...`)
- [ ] No embedding files, eval output files, or model weights staged for commit
- [ ] Changes are scoped to one logical concern (not a kitchen sink PR)
- [ ] Performance-sensitive changes include a before/after `vectro bench` run
- [ ] New binary format variants include a format-detection test in `vectro_lib`

## Related issues
<!-- Closes #123 -->
