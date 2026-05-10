# Vectro REST API

A thin FastAPI wrapper around [`vectro`](https://github.com/konjoai/vectro)'s
`HNSWIndex` so any language can build, query, and tear down approximate
nearest-neighbour indices over HTTP.

The service is stateless across restarts — indices live in process memory.
For durable storage, run behind a stateful host (Render persistent service,
Fly volume, ECS+EBS) or front it with your own snapshot loop.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or via Docker:

```bash
docker build -t vectro-api .
docker run --rm -p 8000:8000 vectro-api
```

OpenAPI / Swagger UI: `http://localhost:8000/docs`.

## Endpoints

### `POST /index` — create

```bash
curl -X POST http://localhost:8000/index \
  -H 'content-type: application/json' \
  -d '{"name":"demo","dim":4,"metric":"cosine"}'
```

```json
{"name":"demo","dim":4,"metric":"cosine"}
```

`metric` is `"cosine"` (default) or `"l2"`.

### `POST /index/{name}/add` — add vectors

`ids` is optional; when omitted, the service auto-assigns string IDs
(`"0"`, `"1"`, ...) matching insertion order.

```bash
curl -X POST http://localhost:8000/index/demo/add \
  -H 'content-type: application/json' \
  -d '{
    "vectors": [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0]
    ],
    "ids": ["x", "y", "z"]
  }'
```

```json
{"added":3,"total":3}
```

### `POST /index/{name}/search` — kNN search

```bash
curl -X POST http://localhost:8000/index/demo/search \
  -H 'content-type: application/json' \
  -d '{"query":[0.99,0.01,0.0,0.0],"k":2}'
```

```json
{"hits":[{"id":"x","distance":0.00005},{"id":"y","distance":1.0}]}
```

Optional `ef` controls the HNSW search beam width (default `max(k*4, 64)`).

### `GET /index/{name}/stats` — inspect

```bash
curl http://localhost:8000/index/demo/stats
```

```json
{"name":"demo","dim":4,"metric":"cosine","count":3}
```

### `DELETE /index/{name}` — drop

```bash
curl -X DELETE http://localhost:8000/index/demo -i
```

```
HTTP/1.1 204 No Content
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status":"ok","indices":1}
```

## Errors

| Status | When |
|--------|------|
| `400`  | Vector dim mismatch, NaN/Inf in payload, `ids` length ≠ `vectors` length |
| `404`  | Index name does not exist |
| `409`  | `POST /index` with a name that already exists |
| `422`  | Pydantic validation (missing field, wrong type, value out of bounds) |

## Tests

```bash
pytest api/test_api.py -v
```

## Deploy to Render

The repo ships with a `render.yaml` blueprint at the root. From the Render
dashboard: **New → Blueprint → point at the repo**, and Render builds the
Dockerfile and exposes port 8000 automatically.
