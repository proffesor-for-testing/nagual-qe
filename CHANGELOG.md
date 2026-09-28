# Changelog

All notable changes to Nagual-QE will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-09-28

### Fixed
- **Build on `master`** — restored after dependabot major bumps: SHAKE-256 now comes from the
  `shake` crate (sha3 0.12 dropped the XOFs); the four self-owned dynamic SQL sites are wrapped in
  `sqlx::AssertSqlSafe` (sqlx 0.9); `rand_chacha` pinned back to 0.3 to match `rand` 0.8.
- **`knowledge list --domain X --limit N`** returned nothing for small `N`: the limit was applied to
  the DB fetch before the domain filter and the reward/usage/created sorts. Filtering and sorting now
  happen before pagination. Regression tests added.
- **Log output** — JSON tracing lines go to stderr instead of stdout, so `nagual … | grep` sees only
  command output; `RUST_LOG` now fully controls the filter.
- **`nagual serve` panicked on startup** after the axum 0.8 bump: path parameters used the 0.7 `:id`
  syntax, which axum 0.8 rejects when the router is built. Routes now use `{id}`; the router is built in
  `build_router()` and covered by tests (the handler tests never built it, so the suite stayed green).
- **Local-only mode never engaged**: `serve` always opens an API-key store, so a fresh install with no
  token, no keys and no dashboard users answered 401 to its own dashboard. Local-only now means: no master
  token, no dashboard users, zero active keys (checked per request; fails closed on DB errors).
- Dashboard endpoints (`/api/patterns`, `/api/graph/3d`, `/api/pulse`, domain stats, surprise, recent
  events, health) hard-coded a `created_at` column; CLI-created databases use `timestamp`, so they
  returned 500 on every fresh local install. The column is now detected.
- `learn record` printed `Reward: 0.20` (the outcome's target), which read as the pattern's new reward. It
  now prints `Pattern reward: 0.500 -> 0.470 (moved toward 0.20 …)`; JSON adds `pattern_reward_before/after`.
- Builds without `onnx-embed` no longer warn that `ORT_DYLIB_PATH` is missing on every command.
- Router latency no longer truncates sub-microsecond decisions to 0 µs.
- Flaky `test_high_dimensional_embeddings` (failed ~38% of runs on random fixtures) and the
  non-compiling `ml::lora` doctest.

### Security
- **PII redaction on the HTTP read path** — `/api` search, get and list responses run
  problem/solution/context through the 12-pattern redactor, so a secret stored in the local DB is
  not exfiltrated into agent context by an ordinary search. The local DB is still never rewritten.
- `/api/graph/3d` accepts `?limit=` (default 5000, ceiling 20000) instead of a hard 2000.

### Changed
- CI runs the whole suite (unit, integration and doc tests) with `kos serve`, not only `--lib`.
- Dependabot no longer auto-proposes majors for `sqlx`, `sha3`, `axum`, and keeps `rand_chacha` on
  the `rand` 0.8 line.
- README: documented the no-ONNX build path (`--no-default-features --features "kos serve"`).

### Known issues
- `tests/router_tests.rs`: `test_text_complexity_estimation` and
  `test_vendor_selection_high_complexity` fail. Both exercise a reference router defined inside the
  test file, not `src/router`; the in-file model and its assertions disagree. Needs a decision on which side is right.

## [0.1.0] - 2026-04-21

### Added
- Initial public release
- ReasoningBank pattern storage with BLAKE3 dedup
- Hybrid retrieval (FTS5 + cosine + tag + graph boost)
- SONA learning loop with Bayesian quality scoring
- KOS subsystems: lineage, witness, delta, epochs, tiers
- HTTP + WebSocket + Unix socket serve mode
- PII redactor with 12 sensitive-string patterns
- Constitution runtime enforcement (5 operational rules)
- Brier-calibrated prediction engine
- Self-improvement + drift monitoring
- Optional `brain-sync`, `mincut`, `domain-expansion`, `strange-loop-meta`
