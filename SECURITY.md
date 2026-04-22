# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's [security advisories
page](https://github.com/proffesor-for-testing/nagual-qe/security/advisories/new).
Do **not** open a public issue for security problems.

We aim to respond within 72 hours and will coordinate disclosure with you
before any public announcement.

## Dependency history

### `cloud-storage 0.11.1` — **removed** (2026-04)

Previously shipped as a dependency for the GCS backup adapter. The crate
is unmaintained and dragged in two vulnerable transitives:

| Transitive | Version | Advisory |
|------------|---------|----------|
| `jsonwebtoken` | 7.2.0 | [GHSA-h395-gr6q-cpjc](https://github.com/advisories/GHSA-h395-gr6q-cpjc) — type confusion in JWT validators |
| `ring` | 0.16.20 | [GHSA-4p46-pwfr-66x6](https://github.com/advisories/GHSA-4p46-pwfr-66x6) — AES panic with overflow checking |

Audit showed the crate had **zero actual call sites** in our code — the
GCS adapter in `src/sync/gcloud.rs` was stubbed ("simulated upload") and
every `cloud_storage::` reference lived in a comment. Removing the dep
cleared both alerts and eliminated ~100 transitives from the lockfile
with no functional change.

If you want real GCS upload/download calls, wire up the maintained
[`google-cloud-storage`](https://crates.io/crates/google-cloud-storage)
crate (or [`object_store`](https://crates.io/crates/object_store) for a
multi-cloud backend) behind an optional feature flag. The `sync/gcloud.rs`
interface — `GCloudAdapter`, `GCloudConfig`, `EncryptionConfig` — is
already in place; only the I/O leaves need to be filled in.

### `rand 0.9.x` (dev-dependency only)

Pulled in by `proptest` for property-based tests. The 0.9.x advisory
(GHSA-cq8v-f236-94qc) affects `rand::rng()` with custom loggers; our
tests don't install custom loggers, and `rand` is not in the production
binary (direct dep is pinned at 0.8).

## Hardening recommendations

When deploying Nagual beyond a single-user local install, also follow
the checklist in [docs/gcloud-deploy.md](docs/gcloud-deploy.md#security-checklist).
Key items:

- Rotate `NAGUAL_API_TOKEN` and dashboard user passwords at least annually.
- Run behind a reverse proxy or Cloudflare Tunnel; never expose
  `nagual serve` directly to the internet.
- Use the `postgres_url` env var or config file — never commit connection
  strings with passwords into the repo.
- PII redaction is applied to all outbound writes (PostgreSQL, cloud API,
  optional external Brain sync). Local SQLite is NOT redacted — treat it
  as sensitive user data.
