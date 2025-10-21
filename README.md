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

CLI examples
-----------

Compress embeddings stored in a NumPy `.npy` file:

```bash
# create venv and install deps (see above)
./bin/vectro compress --in embeddings.npy --out compressed.npz
```

Evaluate a compressed file (prints original bytes, compressed bytes, and mean cosine):

```bash
./bin/vectro eval --orig embeddings.npy --comp compressed.npz
```

Benchmarking
------------

There's a small benchmark harness at `python/bench.py` that measures compression time, reconstruction time, mean cosine similarity, recall@k, and a simple bytes comparison. Example:

```bash
python python/bench.py --n 2000 --d 128 --queries 100 --k 10
```

Note: By default the benchmark uses the Python fallback quantizer. If you compile and expose the Mojo backend at `vectro.src.quantizer`, the harness will automatically run Mojo-backed quantization as well and report both results.

Integrations: Qdrant and Weaviate (pre-compress before indexing)
-------------------------------------------------------------

Pre-compressing embeddings before indexing can save storage costs and reduce network transfer. The simplest pattern is:

1. Generate or obtain embeddings as a NumPy array `embeddings.npy` (shape [n, d]).
2. Compress with Vectro: `./bin/vectro compress --in embeddings.npy --out compressed.npz`.
3. When indexing, reconstruct on the client side (or in a pre-processing step) to float vectors and push to the vector DB. Alternatively, store compressed payloads in metadata and decompress at query time.

Qdrant example (pre-compress, decompress then index):

```python
import numpy as np
from vectro.python.interface import reconstruct_embeddings
import qdrant_client

# load compressed
npz = np.load('compressed.npz')
q = npz['q']
scales = npz['scales']
dims = int(npz['dims'])

# reconstruct to floats
recon = reconstruct_embeddings(q, scales, dims)

# index into Qdrant (pseudo-code)
# client = qdrant_client.QdrantClient(url='http://localhost:6333')
# client.upload_collection(collection_name='my_collection', vectors=recon.tolist())
```

Weaviate example (uploading reconstructed vectors):

```python
import numpy as np
from vectro.python.interface import reconstruct_embeddings
import weaviate

# load compressed
npz = np.load('compressed.npz')
q = npz['q']
scales = npz['scales']
dims = int(npz['dims'])

recon = reconstruct_embeddings(q, scales, dims)

# weaviate_client = weaviate.Client('http://localhost:8080')
# for i, vec in enumerate(recon):
#     weaviate_client.data_object.create({'id': str(i)}, 'EmbeddingClass', vector=vec.tolist())
```

Notes:
- For very large datasets, reconstruct vectors in streaming chunks to avoid memory pressure.
- Another pattern: store compressed bytes as metadata in the vector DB and decompress on read for advanced workflows.
