---
paths:
  - "**/benchmarks/**"
  - "**/bench_*.rs"
  - "**/perf/**"
---
# Benchmarking Rules

- Minimum 5 warmup runs before timing. Discard warmup in reported metrics.
- Report p50, p95, p99, stddev — not just mean.
- Document hardware completely: chip, RAM, OS, driver version, thermal state, process isolation method.
- Close background apps. Disable Spotlight indexing before a run.
- Statistical significance: paired t-test or Wilcoxon signed-rank when comparing implementations.
- Results → `benchmarks/results/<timestamp>_<name>/`. Never overwrite — always a new directory.
- Regression gate: >5% p95 latency or >10% peak memory = hard stop, profile and fix before merging.
