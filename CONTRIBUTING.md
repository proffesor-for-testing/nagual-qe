# Contributing to Nagual-QE

Thanks for your interest in contributing. This document covers the minimum
you need to make a change land cleanly.

## Before you start

- Open an issue first for anything larger than a one-file bug fix. It is
  much cheaper to agree on a direction in a comment thread than to rebase a
  rejected PR.
- Run `cargo check && cargo test --lib` locally — CI will run the full
  suite, but the short feedback loop saves both of us time.

## Setup

See [docs/setup.md](docs/setup.md) for the full local installation. The
10-second version:

```bash
git clone https://github.com/proffesor-for-testing/nagual-qe
cd nagual-qe
cargo check                                    # default features (kos + onnx-embed)
cargo test --lib                               # ~1,300 unit tests
cargo test --test profdag_e2e_test             # integration tests
```

## Code style

- Rust 2021 edition, `rustfmt` defaults (run `cargo fmt`).
- `clippy` warnings should not increase — run `cargo clippy --all-targets`.
- Keep public APIs small. Internals can move freely; public re-exports cannot.
- Tests colocated with the module they test; integration tests in `tests/`.

## What we ask of PRs

1. **Tests pass.** Unit + integration + the benches that compile. No flakes.
2. **No new warnings.** Use `#[allow(dead_code)]` sparingly and only with a
   comment explaining *why* the item is kept.
3. **No `unwrap()` / `expect()` in library code** — return `Result` or
   document the invariant. Tests may unwrap freely.
4. **No PII in commits.** Run the local PII sweep before pushing:
   ```bash
   grep -rnE '(/Users/[a-z]+|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|@gmail\.com)' \
     --include='*.rs' --include='*.md' --include='*.sh' --include='*.toml' .
   ```
5. **Constitution compliance.** If your change touches pattern
   storage/deletion, confirm the five [Operational Rules]
   (NAGUAL_CONSTITUTION.md#operational-rules-runtime-enforced) are still
   enforceable.

## Commit messages

Conventional-ish — prefixes we recognize:

```
feat:    new capability
fix:     bug fix
perf:    performance improvement (include before/after numbers)
refactor: no behavior change
docs:    documentation only
test:    test-only change
chore:   tooling / build / CI
```

One subject line, wrap the body at 72 chars, reference issues with
`Fixes #123`.

## Scope of pattern contributions

We welcome QE pattern contributions (bug patterns, test-design patterns,
flaky-test heuristics). The seed export lives in `seeds/` — open a PR
adding a `.jsonl` entry. Each entry is reviewed for:

- No PII, no client-identifying data, no credentials
- Actually useful (reproducible problem + concrete solution)
- Appropriate domain tag (see `seeds/DOMAINS.md`)

## Code of Conduct

Be kind, assume good faith, and critique ideas rather than people. We
follow the principles in [NAGUAL_CONSTITUTION.md](NAGUAL_CONSTITUTION.md) —
especially Principle 0 (Seek Truth) and Principle 4 (Epistemic Humility).
If you disagree with a decision, say so clearly and bring evidence.

## License

By contributing you agree that your contributions will be licensed under
the [MIT License](LICENSE).
