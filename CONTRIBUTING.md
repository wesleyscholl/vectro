# Contributing to Vectro

Thank you for your interest in contributing!  Vectro is a Mojo-first,
production-grade embedding compression library targeting Apple Silicon.
Contributions that improve compression ratio, throughput, or API coverage
are especially welcome.

Not sure where to start? Browse
[GitHub Discussions](https://github.com/wesleyscholl/vectro/discussions) or
open an issue — we're friendly and responsive.

---

## Good First Issues

Look for issues tagged
[`good first issue`](https://github.com/wesleyscholl/vectro/labels/good%20first%20issue)
on GitHub.  Common entry points:

| Label | Examples |
|---|---|
| `good first issue` | Add a new CLI flag, improve an error message, add a test |
| `docs` | Fix a typo, add a usage example, improve the quickstart |
| `bench` | Run a new embedding model through the benchmark suite |
| `integration` | Add a new vector-DB connector in `python/integrations/` |

If none of the open issues fit, open a
[Discussion](https://github.com/wesleyscholl/vectro/discussions/new?category=ideas)
proposing your idea before you write code.

---

## Setting Up the Development Environment

### Python (required)

```bash
# 1. Clone and enter the repo
git clone https://github.com/wesleyscholl/vectro.git
cd vectro

# 2. Create a virtual environment (Python 3.10+ recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode with dev extras
pip install -e ".[dev]"
```

### Mojo (optional — for SIMD hot-path development)

Mojo SDK and the [pixi](https://prefix.dev/docs/pixi/) package manager are
required to build the SIMD binary (`vectro_binary`).

```bash
# Build the Mojo binary
pixi run build-mojo

# Verify the self-test passes
pixi run selftest
```

If the Mojo binary is absent, Vectro falls back to a pure-Python NumPy
implementation automatically.  All Python-layer tests pass without it.

### Rust extension (optional)

The JS bindings use a Node-API C++ extension built with node-gyp:

```bash
cd js && npm install && npm run build
```

---

## Running Tests

```bash
# Full Python test suite (no model weights required)
pytest tests/ -v --timeout=120

# Skip Mojo-dependent tests when the binary is absent
pytest tests/ -v -k "not mojo"

# Run a focused subset
pytest tests/ -k "binary" -v
```

The test suite uses synthetic tensors — no external model files are needed.

---

## Linting and Type Checking

```bash
# Lint + format check
ruff check python/ tests/
ruff format --check python/ tests/

# Type checking
mypy python/
```

CI will fail if `ruff check` reports any errors.

---

## What Makes a Good PR

- **Focused** — one logical change per PR
- **Tests pass** — `pytest tests/ --timeout=120` green before opening the PR
- **No model weights committed** — keep binary artefacts out of the repo
- **Accuracy contracts hold** — quantization accuracy must not drop below
  per-method minimums (see `AGENTS.md` / `CLAUDE.md` for the table)
- **Performance-sensitive changes** include a before/after `pixi run benchmark`
  run comparing INT8 throughput
- **CHANGELOG updated** — add an entry under the correct `[Unreleased]` heading

---

## Quantization Accuracy Contracts

A PR that drops any method below its minimum cosine similarity is a **hard
stop** and will not be merged until the regression is fixed.

| Method | Min cosine | Compression |
|--------|-----------|-------------|
| INT8   | ≥ 0.9999  | ~4×         |
| INT4   | ≥ 0.9990  | ~8×         |
| NF4    | ≥ 0.9800  | ~8×         |
| Binary | ≥ 0.7500  | ~32×        |
| PQ-96  | ≥ 0.9500  | ~32×        |
| RQ     | ≥ 0.9700  | ~16×        |

---

## Reporting Bugs

Use [GitHub Issues](https://github.com/wesleyscholl/vectro/issues) with the
**Bug report** template.  Please include:

- Python version and OS
- Whether the Mojo binary is installed (`pixi run selftest`)
- A minimal reproducer using synthetic data (no proprietary embeddings)

---

## Architecture Overview

```
python/             Python API layer (thin wrappers, no algorithm logic)
  _mojo_bridge.py   IPC bridge: JSON over stdin/stdout pipe to Mojo binary
  *_api.py          One file per quantization method
  integrations/     Vector-DB connectors (one file per DB)
src/vectro_mojo/    Mojo SIMD hot paths — all quantization inner loops
rust/               Legacy scaffold (do not add features)
js/                 C++ Node-API bindings
tests/              Python test suite (mirrors python/ structure)
```

Algorithm logic lives in Mojo.  Python files are thin wrappers that delegate
to `_mojo_bridge.py`.  Never add quantization math to Python.

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, tagged releases |
| `dev` | Integration branch for multi-part features |
| `research/` | Experimental work with no promotion date |

---

## License

By contributing you agree that your contributions will be licensed under the
[MIT License](LICENSE) that covers this project.
