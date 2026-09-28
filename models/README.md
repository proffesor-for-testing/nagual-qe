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

## Router model (`fastgrnn_router.json`)

`fastgrnn_router.json` holds the weights of the FastGRNN that estimates **query complexity** for
`VendorRouter` (library API, `nagual::router`): below 0.3 → local-small, below 0.5 → local-large,
0.5 and above → cloud. It is embedded at build time (`include_str!`); no ONNX file is needed.

| File | Committed? | Purpose |
|------|-----------|---------|
| `router_queries.jsonl` | ✅ yes | 160 labelled queries (40 per level), each with a fixed `train`/`test` split |
| `fastgrnn_router.json` | ✅ yes | Trained weights + training config + train/test metrics + data hash |
| `train_fastgrnn.py` | ✅ yes | Trainer (pure Python; `--export-onnx` needs torch) |
| `fastgrnn_router.onnx` | ❌ no | Optional ONNX export, `.gitignore`d |

Labelling rubric for `router_queries.jsonl`:

| Level | Target | What belongs here |
|---|---|---|
| `low` | 0.15 | chit-chat, trivial facts, a single command or definition |
| `medium` | 0.40 | one well-scoped task or explanation |
| `high` | 0.60 | multi-step task with context or constraints; real debugging; component-level design |
| `very_high` | 0.85 | architecture, proofs, distributed/concurrent correctness, security threat models, multi-part trade-off analysis |

Retrain (from the repository root) after changing the features or the labelled set:

```bash
cargo run -q --example router_features --no-default-features --features kos -- \
    models/router_queries.jsonl > /tmp/router_features.jsonl
python3 models/train_fastgrnn.py --features /tmp/router_features.jsonl --output models/fastgrnn_router.json
cargo test --test router_tests quality     # held-out bar: level >= 65%, local-vs-cloud >= 85%, max 1 level off
```

The model is selected on training loss only; the `test` split is for measurement. Current held-out
result: level 65.0%, local-vs-cloud 90.0%, worst error 1 level (the previous weights, trained on
random synthetic features: 47.5% / 90% / 2 levels, with every score between 0.497 and 0.518).
