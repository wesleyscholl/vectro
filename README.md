# Vectro — Embedding Compressor (MVP)

This is an MVP layout for "Vectro", an embedding compressor prototype.

What is included:
- Simple per-vector int8 quantizer implemented in Python as a fallback (`python/interface.py`).
- Unit tests using pytest (`python/tests/test_quantization.py`).
- Mojo stubs in `src/quantizer.mojo` for future high-performance reimplementation.

How to run the tests locally:

Create a virtualenv and install dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r vectro/requirements.txt
pytest -q vectro/python/tests
```

Notes:
- The Mojo files are placeholders documenting the intended API. Once Mojo is available, replace the Python implementation with Mojo bindings for performance.
- The quantizer uses per-vector scaling to maximize fidelity.
