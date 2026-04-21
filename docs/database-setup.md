# Database Setup

Nagual-QE uses **SQLite** as its primary store. PostgreSQL (via the
`ruvector-postgres` extension) is **optional** — enable it when you want:

- Native vector indexing (HNSW / pgvector-compatible `ruvector(128)` type)
- Multi-host / shared access to a pattern store
- `pg_notify`-based real-time updates for the dashboard

## SQLite — no setup needed

Nagual creates `./nagual.db` on first run. The schema is applied
automatically by the embedded migration engine. SQLite is WAL-mode by
default, so concurrent reads are safe during writes.

```bash
nagual knowledge store "..." --solution "..." --domain "demo"
# Check health
nagual health --detailed
```

## PostgreSQL — optional dual-write

The dual-write adapter fires every write to SQLite first (source of truth)
and asynchronously to PostgreSQL (read replica with richer indexing).

### Requirements

- PostgreSQL 14+
- The [`ruvector`](https://github.com/ruvnet/ruvector) extension built for
  your architecture

### Quick start with Docker

```bash
# 1. Build the ruvector-postgres image for your architecture (once).
#    See https://github.com/ruvnet/ruvector/tree/main/crates/ruvector-postgres
#    Example for arm64:
#      docker build --platform linux/arm64 \
#        -f crates/ruvector-postgres/Dockerfile \
#        -t ruvector-postgres:arm64 .

# 2. Copy the env template and set a strong password
cp .env.example .env
# Edit .env — change POSTGRES_PASSWORD at minimum

# 3. Start the container
docker compose up -d postgres

# 4. Verify the container is healthy
docker compose ps
# Expected: nagual-ruvector  ...  (healthy)

# 5. Tell nagual to dual-write
export DATABASE_URL=postgres://nagual:<your-password>@localhost:5432/nagual
# or set postgres_url in ~/.nagual/config.toml

# 6. Sync the existing SQLite DB into Postgres (one-time)
nagual knowledge sync
```

### Migrations

Migrations live in `migrations/`. The Docker image applies them on first
boot via `scripts/init-postgres.sql`. To apply them against an existing
Postgres instance:

```bash
nagual migrate --postgres-url "$DATABASE_URL"
```

Migrations are numbered + idempotent — running them twice is a no-op.

### Known gotchas

- **HNSW index creation is currently commented out** for newly seeded
  databases. Sequential scan with `<=>` cosine distance is fast enough up
  to ~10k vectors. See `migrations/002_hnsw_indexes.up.sql` and
  `migrations/012_profdag_hnsw.up.sql`.
- **The `ruvector-postgres` published image is amd64-only** as of writing.
  Build it from source for arm64 (Apple Silicon, Graviton) — see the
  comment block at the top of `docker-compose.yml`.
- **`nagual knowledge sync` is one-directional** (SQLite → PG). For
  bidirectional sync between two nagual instances, use `nagual cloud push`
  and `nagual cloud pull`.

## Backups

Don't skip this. See `deploy/nagual-backup.sh` for a ready-to-install cron
job that snapshots SQLite + PostgreSQL to GCS every 6 hours. Adapt the
destination to S3 / B2 / plain disk if you prefer — the interface is just
`gsutil cp`.

Local-only alternative:

```bash
# One-off snapshot
nagual sync backup --full --source ~/.nagual/nagual.db

# Scheduled via nagual's built-in scheduler
nagual sync schedule --interval 6h --destination ~/.nagual/backups
```

## Schema reference

The high-level tables:

| Table | Purpose |
|-------|---------|
| `reasoning_patterns` | Primary pattern store (problem, solution, reward, tags, embedding) |
| `outcomes` | Every recorded success/failure with MAST classification |
| `strategies` | EGUR strategy cache (successful approaches per problem class) |
| `predictions` | Calibrated predictions + Brier score tracking |
| `context_graph` | ProfDAG knowledge graph edges |
| `lineage` | KOS pattern lineage (ancestry, derivations) |
| `witness_chains` | KOS witness attestation chains |
| `delta_events` | KOS append-only change events |
| `pattern_tiers` | Tier graduation (candidate → verified → reflex) |

See `migrations/*.up.sql` for authoritative column definitions.
