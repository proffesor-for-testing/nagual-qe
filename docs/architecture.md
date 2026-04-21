# Architecture

Nagual-QE is a layered Rust application. The core is a ReasoningBank
(pattern store + retrieval), wrapped in a learning loop that updates reward
signals from outcomes, wrapped in a service layer (CLI, HTTP, WebSocket,
Unix socket).

## Module map

```
src/
├── reasoning_bank/     Core pattern types, storage, retrieval, builder
├── learning/           SONA learner, consolidation, self-improvement, drift
├── ml/                 ONNX embedder, hash embedder, cosine similarity, cache
├── db/                 SQLite, PostgreSQL, DualWriteAdapter, FTS5, conflicts
├── graph/              ProfDAG knowledge graph + GNN pressure propagation
├── profdag/            ProfDAG API, schemas, search
├── prediction/         Brier-calibrated prediction engine
├── security/           SQLCipher encryption, credentials, auth
├── sync/               Backup, restore, incremental sync, PII redactor
├── cloud/              Client/server for remote nagual instances
├── serve/              HTTP + WebSocket + Unix socket dashboard backend
├── cli/                CLI commands (subcommand per domain)
├── api/                Programmatic Rust API (used by serve)
├── mcp/                Model Context Protocol server
├── constitution/       Runtime principle enforcement
├── coherence/          Belief consistency gate
├── dream/              Background consolidation ("dreaming")
├── research/           Research-mode (exploration / novelty)
├── router/             Request routing, tiering ladder
├── lineage/            KOS: pattern ancestry + derivation
├── witness/            KOS: attestation chains
├── delta/              KOS: append-only change events
├── epoch/              KOS: time-partitioned learning windows
├── tiering/            KOS: candidate → verified → reflex graduation
├── agent_views/        KOS: per-agent memory projections
├── migration/          Schema migration runner
├── health/             Liveness + readiness checks
├── introspection/      Self-describing metadata
├── observability/      Tracing + metrics
├── injection/          Prompt injection / context poisoning detection
└── planning/           Goal-oriented action planning
```

## Core data model

```text
Pattern (reasoning_bank::Pattern)
├── id: Uuid
├── problem: String
├── solution: String
├── domain: String
├── tags: Vec<String>
├── embedding: Option<Vec<f32>>  (128-dim)
├── reward: f32                   (legacy linear score)
├── bayesian_score: BetaParams    (α, β — the modern score)
├── surprise: f32                 (novelty vs existing patterns)
├── reuse_count: u32
├── created_at / updated_at
├── embedding_method: "onnx" | "hash"
└── metadata: HashMap<String, Value>
```

## Write path

```
┌─────────────────┐
│ nagual store    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Constitution checks     │  ← NeverDeleteWithoutBackup, AlwaysRecordMAST
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ PII redactor (outbound) │  ← only for PG / cloud / brain
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ DualWriteAdapter        │
│  ├─ SQLite (sync)       │  ← source of truth
│  └─ PostgreSQL (async)  │  ← fire-and-forget via tokio::spawn
└─────────────────────────┘
```

## Read path (hybrid retrieval)

```
┌─────────────────┐
│ nagual search Q │
└────────┬────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
     FTS5 on      Cosine sim      Tag filter     Graph boost
     problem +    on embedding    on tags        from ProfDAG
     solution
         │              │              │              │
         └──────┬───────┴──────┬───────┘              │
                ▼              ▼                      │
           MMR rerank    Hyperbolic                   │
                         (optional)                   │
                      │                               │
                      ▼                               │
           ┌──────────────────────────────────────────┘
           │
           ▼
      Scored results
      (relevance, bayesian_mean, recency)
```

## Learning loop

```
   Outcome recorded (success/failure)
            │
            ▼
   Update bayesian_score (α += success, β += failure)
            │
            ▼
   Record drift metric (distance from domain centroid)
            │
            ▼
   If failure → classify with MAST
            │
            ▼
   Schedule consolidation (dedup + merge near-duplicates)
            │
            ▼
   If pattern crosses tier threshold → auto-promote
```

## The five KOS subsystems

KOS ("Knowledge Operating System") is the `kos` feature flag — an
integrated set of subsystems that give patterns longer-lived provenance
and multi-agent accountability:

- **Lineage**: where did this pattern come from? (ancestor graph)
- **Witness**: who has attested to this pattern's quality? (chain)
- **Delta**: what changed, when, by whom? (append-only events)
- **Epochs**: time-partitioned learning windows (for rollback)
- **Tiering**: candidate → verified → reflex graduation ladder

These are wired through a shared `McpRegistry` bridge so that MCP tool
calls from external agents see consistent views.

## Service layer

```
┌────────────────────────────────────────────────────┐
│ nagual serve (axum)                                │
├────────────────────────────────────────────────────┤
│ HTTP     /api/patterns, /api/search, /api/outcome  │
│ WS       /ws/live  (real-time pattern stream)      │
│ Socket   ~/.nagual/nagual.sock (local agents)      │
│ Listen   pg_notify (for multi-host coherence)      │
└────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│ Dashboard (dashboard.html + login.html)            │
│ Vanilla JS, no build step                          │
└────────────────────────────────────────────────────┘
```

## Further reading

- [NAGUAL_CONSTITUTION.md](../NAGUAL_CONSTITUTION.md) — the principles
- [database-setup.md](database-setup.md) — schema + migrations
- [gcloud-deploy.md](gcloud-deploy.md) — production deployment
- [seeding-qe.md](seeding-qe.md) — importing the QE pattern seed
