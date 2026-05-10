---
paths: ["**/benchmarks/**", "**/bench_*.rs", "**/bench_*.py", "**/perf/**"]
---
# Benchmarking Rules
- Minimum 5 warmup runs. Report p50/p95/p99/stddev.
- Document hardware completely.
- Statistical significance: paired t-test or Wilcoxon signed-rank.
- Results in `benchmarks/results/<timestamp>_<name>/`. Never overwrite.
- Regression gate: >5% p95 latency = hard stop.
