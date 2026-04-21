# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's [security advisories
page](https://github.com/proffesor-for-testing/nagual-qe/security/advisories/new).
Do **not** open a public issue for security problems.

We aim to respond within 72 hours and will coordinate disclosure with you
before any public announcement.

## Known accepted transitive risks

The following Dependabot alerts on transitive dependencies are tracked
but not actionable without replacing the offending intermediate
dependency. Each has been assessed against our actual usage.

### `cloud-storage 0.11.1` chain

Nagual uses [`cloud-storage`](https://crates.io/crates/cloud-storage) as
an optional client for Google Cloud Storage backups. The crate is
effectively unmaintained and pulls in older versions of two audited
dependencies:

| Transitive | Version | Advisory | Our assessment |
|------------|---------|----------|----------------|
| `jsonwebtoken` | 7.2.0 | [GHSA — type confusion → auth bypass](https://github.com/advisories) in JWT **validators** | **Not exploitable.** Our only JWT usage is signing short-lived access tokens for GCS upload auth. We never validate JWTs; the advisory affects validators only. |
| `ring` | 0.16.20 | AES functions may panic when overflow checking is enabled | **Not exploitable.** Release builds disable overflow checking. The `ring` 0.16 line received no further patch releases (final version). |

**Mitigation**: disable cloud sync (set `sync.enabled = false` in
`~/.nagual/config.toml`, or omit the `[sync.gcloud]` section). Nothing
else in Nagual depends on the `cloud-storage` chain.

**Long-term fix tracked in [issue #TBD]** — replace `cloud-storage` with
a maintained GCS client (candidates: `google-cloud-storage`,
`object_store`). Contributions welcome.

### `rand 0.9.4` (dev-dependency only)

Pulled in by `proptest` for property-based tests. The 0.9.x advisory
affecting `rand::rng()` with custom loggers does not apply — our tests
don't install custom loggers, and `rand` is not in the production
binary (direct dep is pinned at 0.8).

### `lru 0.16.4`

Past the advisory fix version (`0.16.3`). No action required.

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
