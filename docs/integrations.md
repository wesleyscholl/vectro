# Integrations

Vectro ships first-party adapters for popular vector databases, ML frameworks,
and data formats. Install optional dependencies via:

```bash
pip install "vectro[integrations]"   # Qdrant, Weaviate, PyTorch, HuggingFace
pip install "vectro[data]"           # Apache Arrow, Parquet
```

---

## Vector Databases

### InMemoryVectorDBConnector (built-in)

A simple in-memory connector useful for testing and small-scale RAG demos:

```python
from python.integrations import InMemoryVectorDBConnector
import numpy as np

store = InMemoryVectorDBConnector()
rng = np.random.default_rng(0)
embeddings = rng.standard_normal((100, 384)).astype(np.float32)

batch_id = store.store_batch(embeddings, metadata={"dataset": "test"})
results = store.search(embeddings[0], top_k=5)
for r in results:
    print(r.id, r.score)
```

### Qdrant

```bash
pip install qdrant-client
```

```python
from python.integrations import QdrantConnector

conn = QdrantConnector(
    url="http://localhost:6333",
    collection_name="my_embeddings",
    vector_dim=768,
)

import numpy as np
rng = np.random.default_rng(0)
embeddings = rng.standard_normal((500, 768)).astype(np.float32)

# Compress and upload
conn.upsert(embeddings, ids=list(range(500)))

# Search
results = conn.search(embeddings[0], top_k=10)
```

> **Note** – Vectro compresses the vectors before upload and transparently
> decompresses them on reconstruction. The Qdrant collection stores INT8
> payloads, not float32, saving ~4× memory on the server.

### Weaviate

```bash
pip install weaviate-client
```

```python
from python.integrations import WeaviateConnector

conn = WeaviateConnector(
    url="http://localhost:8080",
    class_name="Article",
    vector_dim=384,
)

conn.upsert(embeddings, ids=[f"doc-{i}" for i in range(len(embeddings))])
results = conn.search(embeddings[0], top_k=5)
```

---

## PyTorch / HuggingFace

```bash
pip install torch transformers
```

### Compress a `torch.Tensor`

```python
from python.integrations import compress_tensor, reconstruct_tensor
import torch

tensor = torch.randn(256, 768)   # batch of embeddings

result = compress_tensor(tensor)
restored = reconstruct_tensor(result)   # torch.Tensor, same shape
```

### `HuggingFaceCompressor`

Drop-in wrapper around any `sentence-transformers`-compatible encoder that
compresses outputs on-the-fly:

```python
from python.integrations import HuggingFaceCompressor

hf = HuggingFaceCompressor(model_name="sentence-transformers/all-MiniLM-L6-v2")

sentences = ["Hello world", "Vectro is fast", "LLM embeddings"]
results = hf.encode_and_compress(sentences)   # list[QuantizationResult]
```

---

## Apache Arrow & Parquet

```bash
pip install "vectro[data]"
```

### Arrow Tables

```python
from python.integrations import result_to_table, table_to_result
from python import Vectro
import numpy as np

rng = np.random.default_rng(0)
embeddings = rng.standard_normal((100, 128)).astype(np.float32)
result = Vectro().compress_batch(embeddings)

# Convert to Arrow
table = result_to_table(result)
print(table.schema)

# Round-trip
result2 = table_to_result(table)
```

### Parquet I/O

```python
from python.integrations import write_parquet, read_parquet

write_parquet(result, "embeddings.parquet")
result2 = read_parquet("embeddings.parquet")
```

### Arrow IPC bytes

```python
from python.integrations import to_arrow_bytes, from_arrow_bytes

payload = to_arrow_bytes(result)    # bytes  — send over network
result2 = from_arrow_bytes(payload) # reconstruct locally
```

---

## Streaming Decompressor

For datasets too large to decompress all at once:

```python
from python import StreamingDecompressor, Vectro

result = Vectro().load_compressed("huge_dataset.npz")

total = 0
for chunk in StreamingDecompressor(result, chunk_size=512, backend="auto"):
    # chunk: float32 array of shape (up to 512, dim)
    total += len(chunk)

print(f"Processed {total} vectors")
```

`backend="auto"` picks the fastest available backend (mojo → rust → python).

---

## Advanced Quantization

### INT2 Quantization (experimental)

```python
from python import quantize_int2, dequantize_int2
import numpy as np

rng = np.random.default_rng(0)
vecs = rng.standard_normal((64, 128)).astype(np.float32)

quantized, scale, zero_point = quantize_int2(vecs)
restored = dequantize_int2(quantized, scale, zero_point)
```

### Adaptive quantization

Chooses INT4 or INT8 automatically based on per-vector variance:

```python
from python import quantize_adaptive

results = quantize_adaptive(vecs, threshold=0.1)
```
