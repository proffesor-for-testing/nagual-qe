# Importing the QE Seed

Nagual-QE ships with an **optional** seed set of Quality Engineering
patterns: flaky-test heuristics, test-design patterns, bug-pattern
recognizers, and so on. Importing the seed gives you a working knowledge
base on day one — without it, you start with an empty database and the
system learns from scratch as you use it.

## Where the seed lives

The seed is distributed as a versioned JSONL file in the
[`seeds/`](../seeds/) directory (or as a release asset, depending on the
release). Each line is a self-contained pattern record:

```json
{"problem": "...", "solution": "...", "domain": "qe.flaky", "tags": ["..."], "reward": 0.85}
```

The seed is curated and PII-scrubbed — no client names, no credentials, no
absolute paths. If you notice anything that shouldn't be public, please
open an issue.

## Import

```bash
# Preview without writing (dry run)
nagual knowledge import --seed seeds/qe-seed-v1.jsonl --dry-run

# Actually import
nagual knowledge import --seed seeds/qe-seed-v1.jsonl

# Import into a specific domain prefix (useful for namespacing)
nagual knowledge import --seed seeds/qe-seed-v1.jsonl --domain-prefix "seed."
```

Imported patterns are marked with `metadata.source = "seed"` so you can
distinguish them from your own later.

## Re-import (idempotent)

The importer dedups by content hash (BLAKE3 over `problem + solution`).
Re-running import with the same seed is a no-op. Re-running with a newer
seed version only adds the new patterns.

## Removing the seed

If you decide the seed isn't for you:

```bash
nagual knowledge delete --where 'metadata.source = "seed"' --backup
```

The Constitution rule `NeverDeleteWithoutBackup` forces the `--backup`
flag — the deleted patterns are archived to `~/.nagual/backups/` and can
be restored.

## Contributing patterns to the seed

See the "Scope of pattern contributions" section in
[CONTRIBUTING.md](../CONTRIBUTING.md). Briefly:

- No PII, no client-specific details, no credentials
- Actually useful — reproducible problem + concrete solution
- Appropriate domain tag (see `seeds/DOMAINS.md` when it lands)

Pattern submissions go through a PR review focused on the above.
