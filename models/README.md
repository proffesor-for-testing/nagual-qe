# Models

This directory is where Nagual looks for ONNX embedding models at runtime.
The models themselves are **not** committed to git (they're ~86MB) — you
download them once when you first set up the project.

## Required models (for `onnx-embed` feature)

| File | Size | Source |
|------|------|--------|
| `all-MiniLM-L6-v2.onnx` | ~86MB | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| `tokenizer.json` | ~0.4MB | (same repo, `tokenizer.json`) |

## Download

```bash
cd models
curl -L -o all-MiniLM-L6-v2.onnx \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -L -o tokenizer.json \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
```

Or for system-wide install:

```bash
mkdir -p ~/.nagual/models
cd ~/.nagual/models
# ... same curl commands
```

## Skipping ONNX

If you don't want to deal with ONNX, build without the default feature
set — Nagual falls back to a deterministic hash embedder:

```bash
cargo build --release --no-default-features --features kos
```

Trade-off: hash embeddings are ~4× faster but capture no semantic meaning.
They're fine for exact-match dedup and graph topology work; they're not
useful for "find similar patterns" queries.

## Optional router model

The `fastgrnn_router` is a tiny learned classifier that decides whether an
incoming query should be routed to FTS (full-text search) or the vector
index. It's optional — when the ONNX file isn't present, Nagual falls
back to random Xavier-initialized weights + heuristic routing.

| File | Committed? | Purpose |
|------|-----------|---------|
| `fastgrnn_router.json` | ✅ yes | Router weights metadata (embedded at build via `include_str!`) |
| `fastgrnn_router.onnx` | ❌ no | Compiled model — `.gitignore`d because ONNX binaries bake source-tree paths into traceback metadata |
| `train_fastgrnn.py` | ✅ yes | PyTorch training script |

To build your own:

```bash
cd models
python3 train_fastgrnn.py     # produces fastgrnn_router.onnx + .onnx.data
```

Nagual will pick it up automatically on next startup.
