# Vectro — Embedding Compressor (MVP)

Vectro is a small prototype and reference implementation for compressing embedding vectors
used in retrieval / vector search pipelines. The goal is to provide a simple, extensible
stack you can use to reduce embedding storage and transfer costs while maintaining
retrieval quality.

This repo contains:

- A Python reference quantizer (per-vector int8) in `python/interface.py` (easy to run).
- Mojo stubs in `src/quantizer.mojo` (intended for a future high-performance core).
- A CLI (`bin/vectro`) with `compress` and `eval` commands.
- A benchmark harness (`python/bench.py`) for throughput and quality metrics.
- A small sample dataset generator (`data/generate_sample.py`) and example notebook.

Current functionality (MVP)
---------------------------

- Per-vector int8 quantization:
	- For each vector v: scale = max_abs(v) / 127 (scale=1.0 for zero vectors).
	- Quantized bytes: q = round(v / scale) cast to int8, stored as a flat array.
	- Reconstruction: q * scale per-vector.
- Python fallback implementation is the default. If a Mojo backend is compiled and
	exposed as a Python module at `vectro.src.quantizer`, the code will automatically
	use the Mojo implementation for quantize/reconstruct.
- CLI features:
	- `vectro compress --in embeddings.npy --out compressed.npz` — compress a `.npy` file.
	- `vectro eval --orig embeddings.npy --comp compressed.npz` — reconstruct and print bytes + mean cosine.
- Benchmarking:
	- `python python/bench.py` will run Python (and Mojo if available) backends and report throughput, mean cosine, and recall@k.

Quickstart
----------

Create a virtualenv, install the minimal dependencies and run tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q python/tests
```

Compress and evaluate an embeddings file:

```bash
# compress
./bin/vectro compress --in data/sample_embeddings.npy --out data/sample_compressed.npz

# evaluate
./bin/vectro eval --orig data/sample_embeddings.npy --comp data/sample_compressed.npz
```

Benchmarks
----------

Run the lightweight benchmark (default uses the Python fallback):

```bash
python python/bench.py --n 2000 --d 128 --queries 100 --k 10
```

If you build and expose the Mojo backend the harness will also run the Mojo-backed
implementation and compare throughput and quality.

Sample data and notebook
------------------------

Generate a small sample dataset:

```bash
python data/generate_sample.py --n 500 --d 128 --out data/sample_embeddings.npy
./bin/vectro compress --in data/sample_embeddings.npy --out data/sample_compressed.npz
```

Open `notebooks/example_visualization.ipynb` to see a simple PCA visualization of the reconstructed vectors.

Integrations: Qdrant and Weaviate (pre-compress before indexing)
-------------------------------------------------------------

Pre-compressing embeddings before indexing can reduce storage and network costs. Common patterns:

1. Compress on ingest: generate embeddings, compress them with Vectro, store compressed bytes in object store or as metadata.
2. Decompress before indexing: on the ingestion pipeline reconstruct in memory (streamed) and push floats to the vector DB (Qdrant/Weaviate).
3. Decompress at query time: store compressed vectors and decompress the small set of candidates when answering queries.

Qdrant (decompress then index) — pseudo-code:

```python
import numpy as np
from python.interface import reconstruct_embeddings
# load compressed
npz = np.load('compressed.npz')
q = npz['q']
scales = npz['scales']
dims = int(npz['dims'])
recon = reconstruct_embeddings(q, scales, dims)
# Upload recon (as lists) to Qdrant via its client (not shown).
```

Weaviate (similar approach) — pseudo-code included in the earlier examples.

Roadmap
-------

Short-term (MVP → Alpha)
- Implement and test Mojo core with parallel loops / SIMD for quantize & reconstruct.
- Add a small build or packaging story to produce a Python-importable Mojo module (so `vectro.src.quantizer` becomes available).
- Add chunked streaming I/O to `bin/vectro` and `python/bench.py` to handle datasets larger than memory.

Mid-term (Beta)
- Implement product quantization (PQ) and optimized PQ (OPQ) backends.
- Add benchmarks vs FAISS and report recall/throughput trade-offs.
- Provide a compact on-disk compressed format and fast random access API.

Long-term (1.0+)
- Training-aware learned compression (autoencoder or vector quantization).
- Hosted API/service and dashboard showing compression vs retrieval quality tradeoffs.
- Integrations/adapters for Qdrant, Weaviate, Milvus and LangChain/LlamaIndex.

Contribution & License
----------------------

This project is MIT-licensed. Contributions welcome — open a PR with small, focused changes. If you add a Mojo backend, include tests that compare to the Python fallback and verify numeric parity for basic metrics.

Contact / Next steps
--------------------

If you'd like I can:
- Add Mojo build bindings and a simple `make` or `python setup` step to compile the backend.
- Implement streaming/chunked CLI operations for large datasets.
- Prototype PQ in Python and compare it against the per-vector int8 baseline.

Pick the next feature and I’ll implement it.