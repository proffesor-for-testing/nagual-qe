# Nagual-QE

**A Rust-native self-learning knowledge system for Quality Engineering.**

Nagual-QE stores every pattern, test insight, and decision you encounter,
learns which ones actually work, and surfaces the right one when you need it
again. It is a local-first, privacy-respecting memory layer for engineers and
their AI agents.

> *"The Tonal is the island of the known. The Nagual is the vast unknown
> surrounding the island. The warrior's path is not to destroy the Tonal, but
> to keep it lean so the Nagual can emerge."*

See [NAGUAL_CONSTITUTION.md](NAGUAL_CONSTITUTION.md) for the eight principles
the system operates by.

---

## What it does

- **Stores** problem → solution patterns with provenance, tags, and embeddings
- **Learns** from outcomes: every success or failure updates a Bayesian quality
  score (Beta distribution), not just a scalar reward
- **Retrieves** with hybrid search — full-text (FTS5) + cosine similarity over
  128-dim embeddings + graph-boosted scoring
- **Consolidates** near-duplicates automatically (BLAKE3 + cosine thresholds)
- **Predicts** with Brier-calibrated confidence; self-scores its own accuracy
- **Redacts PII** before any data ever leaves the local machine
- **Serves** a browser dashboard + HTTP/WebSocket API + Unix socket for local
  agent integration

Designed to be the memory layer behind AI-assisted QE workflows — but it works
equally well as a standalone engineer's notebook.

---

## Why `nagual-qe`

This is the **public** edition of Nagual — the same core engine, shipped with
a QE-flavoured seed pattern set (optional) so you get value on day one. Use it
to back an [agentic-qe](https://github.com/proffesor-for-testing/agentic-qe)
fleet, or stand it up on its own.

### Core features

| Capability | Feature flag | Default |
|---|---|---|
| ReasoningBank (pattern storage + retrieval) | always on | ✅ |
| Hash embedder (no external deps) | always on | ✅ |
| ONNX embedder (all-MiniLM-L6-v2) | `onnx-embed` | ✅ |
| KOS (lineage, witness, delta, tiers, epochs) | `kos` | ✅ |
| HTTP + WebSocket dashboard | `serve` | opt-in |
| Terminal UI | `tui` | opt-in |
| MinCut graph clustering | `mincut` | opt-in |
| Cross-domain transfer learning | `domain-expansion` | opt-in |
| Meta-cognitive self-critique | `strange-loop-meta` | opt-in |

---

## Quick start

### macOS

```bash
# Prereqs: Rust, ONNX Runtime
brew install onnxruntime

# Build + install
git clone https://github.com/proffesor-for-testing/nagual-qe
cd nagual-qe
bash scripts/mac-setup.sh

# Verify
nagual status
nagual knowledge store "Flaky test caused by async race" \
  --solution "Add explicit await on the setup fixture" \
  --domain "qe.flaky" --tags "async,race-condition"
nagual knowledge search "flaky async"
```

### Linux (Debian/Ubuntu)

```bash
# Prereqs
sudo apt-get install -y build-essential pkg-config libssl-dev sqlite3
# ONNX Runtime (arm64 example — see docs/setup.md for x86_64)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-aarch64-1.17.0.tgz
tar xf onnxruntime-linux-aarch64-1.17.0.tgz
sudo cp onnxruntime-linux-aarch64-1.17.0/lib/libonnxruntime.so* /usr/lib/ && sudo ldconfig

# Build
cargo build --release
sudo cp target/release/nagual /usr/local/bin/
```

### Without ONNX (hash embedder only)

```bash
cargo build --release --no-default-features --features kos
```

### Run the dashboard

```bash
cargo build --release --features serve
nagual serve --port 3333
# open http://localhost:3333
```

---

## Documentation

- [docs/setup.md](docs/setup.md) — local installation (all platforms)
- [docs/database-setup.md](docs/database-setup.md) — SQLite + optional PostgreSQL
- [docs/gcloud-deploy.md](docs/gcloud-deploy.md) — production deployment on GCE
- [docs/seeding-qe.md](docs/seeding-qe.md) — importing the optional QE seed
- [docs/architecture.md](docs/architecture.md) — module map + data flow
- [NAGUAL_CONSTITUTION.md](NAGUAL_CONSTITUTION.md) — the eight principles

---

## Data storage

- **SQLite** at `./nagual.db` (or `~/.nagual/nagual.db`) — always.
- **PostgreSQL** (optional) via `DualWriteAdapter` — needed for the
  `ruvector-postgres` extension (HNSW + GNN + SONA) and multi-host deployments.

No data ever leaves your machine unless you explicitly enable cloud sync.
When you do, the PII redactor strips 12 classes of sensitive strings
(paths, IPs, emails, API keys, SSH keys, JWTs, phone numbers, etc.) before
any write to PostgreSQL or any outbound HTTP call.

---

## Status

Pre-1.0, actively developed. The public surface is stable enough to build on,
but expect migration notes between minor versions. All changes are documented
in [CHANGELOG.md](CHANGELOG.md).

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the local
development loop, code style, and the test requirements we enforce before
merging.

---

## License

MIT — see [LICENSE](LICENSE).
