# Local Setup

This guide walks through installing Nagual-QE on your own machine. For
production / cloud deployment see [gcloud-deploy.md](gcloud-deploy.md).

## Prerequisites

- **Rust** 1.75+ (install via [rustup](https://rustup.rs))
- **SQLite** 3.35+ (bundled via `rusqlite` but the CLI is useful)
- **ONNX Runtime** 1.17+ — *optional*, needed only for the `onnx-embed`
  feature. Without it, Nagual falls back to a deterministic hash embedder.
- **Docker** 24+ — *optional*, only needed if you want to run the
  PostgreSQL dual-write backend locally.

### Install ONNX Runtime

| Platform | Command |
|----------|---------|
| macOS    | `brew install onnxruntime` |
| Debian/Ubuntu (arm64) | Download `onnxruntime-linux-aarch64-1.17.0.tgz` from the [releases page](https://github.com/microsoft/onnxruntime/releases), then `sudo cp libonnxruntime.so* /usr/lib && sudo ldconfig` |
| Debian/Ubuntu (x86_64) | Same but use `onnxruntime-linux-x64-1.17.0.tgz` |

Tell Nagual where it lives:

```bash
# macOS (arm64)
export ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib

# Linux
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
```

## Build

```bash
git clone https://github.com/proffesor-for-testing/nagual-qe
cd nagual-qe

# Default: KOS features + ONNX embedder (~1,300 unit tests, ~10s compile)
cargo check

# Run tests
cargo test --lib

# Release binary
cargo build --release
sudo cp target/release/nagual /usr/local/bin/
# macOS — re-sign after moving to avoid dyld issues:
codesign --force --sign - /usr/local/bin/nagual
```

### Feature flags

| Flag | What it enables |
|------|-----------------|
| `kos` (default) | Lineage, witness, delta, tiers, epochs, EWC, routing ladder |
| `onnx-embed` (default) | ONNX-based embeddings via `all-MiniLM-L6-v2` |
| `serve` | HTTP + WebSocket dashboard |
| `tui` | Terminal UI |
| `full` | `serve + tui` |
| `mincut` | Stoer-Wagner graph partitioning |
| `brain-sync` | Optional collective-knowledge sync (see note below) |

Build examples:

```bash
# Hash embedder only (no ONNX dependency)
cargo build --release --no-default-features --features kos

# With the dashboard
cargo build --release --features serve

# Everything
cargo build --release --features full,mincut
```

## Download ONNX model

The default embedder uses `all-MiniLM-L6-v2` (128-dim). The model file is
~86MB and is **not** committed to the repo.

```bash
mkdir -p ~/.nagual/models
cd ~/.nagual/models

# Download from HuggingFace
curl -L -o all-MiniLM-L6-v2.onnx \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -L -o tokenizer.json \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
```

Nagual searches for the model in:

1. `./models/` (project-local)
2. `../models/`
3. `~/.nagual/models/` (recommended for system-wide installs)

## Configuration

```bash
mkdir -p ~/.nagual
cp config/nagual.toml ~/.nagual/config.toml
# Edit as needed — sqlite_path, postgres_url, sync settings
```

Minimum viable config:

```toml
[database]
sqlite_path = "~/.nagual/nagual.db"

[search]
hnsw_m = 24
hnsw_ef_construction = 200
hnsw_ef_search = 200
```

## Verify

```bash
nagual status
nagual knowledge store "My first insight" --solution "What I learned" --domain "demo"
nagual knowledge search "insight"
```

You should see the pattern you just stored appear in search results with a
non-zero relevance score.

## Optional: PostgreSQL dual-write

See [database-setup.md](database-setup.md).

## Optional: Dashboard

```bash
cargo build --release --features serve
nagual serve --port 3333
# open http://localhost:3333
# Create a dashboard user: nagual user create <name>
```

## Note on `brain-sync`

The `brain-sync` feature enables pushing/pulling patterns to/from an
external "Brain" service. Nagual-QE ships with **no default endpoint** —
you must set `BRAIN_URL` to opt in. This is intentional: we do not ship
default third-party endpoints in the public binary.
