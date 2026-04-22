# Learning Nagual-QE

*An earned-recognition onboarding sequence. Read the layers in order. Do the commands as you go. The principles at the end are meant to feel inevitable by the time you reach them — not declared up front.*

*For: anyone picking up Nagual-QE for the first time, whether human collaborator or AI agent.*

---

## Prologue: what this document is, and isn't

Most README files declare what a system is before you've done anything with it. That works for libraries you call from code. It fails for learning systems, because a learning system is not a product you consume; it is a loop you enter. Until you've closed the loop once, the architecture is noise.

So this document inverts the order. It walks you through the loop first, in six layers. Each layer earns the next:

- **Layer 1** — you do the loop once, by hand, on a real problem you have today.
- **Layer 2** — you learn what actually happened underneath.
- **Layer 3** — you learn to catalyze new behavior on top of the loop.
- **Layer 4** — you learn to read meta-patterns across many loops.
- **Layer 5** — you connect the loop to the rest of your toolchain.
- **Layer 6** — you read the Constitution, and it reads like a summary of what you already know.

If you skip ahead to Layer 6 and try to internalize the principles before doing the work, you will learn nothing and be able to quote a lot. That is the failure mode this sequence is designed to avoid. Classical Indian pedagogy calls this *adhikāra* — earned eligibility. Classical engineering pedagogy calls it "don't read the framework docs, build the TODO app first." Same move, different vocabulary.

**Prerequisites.** A Nagual-QE CLI binary on your path (typically `~/.local/bin/nagual` after `bash scripts/mac-setup.sh` or a `cargo install` equivalent), and a shell. A SQLite database file will be created on first use at `./nagual.db` (or wherever `sqlite_path` in `~/.nagual/config.toml` points). ONNX embeddings and the optional PostgreSQL mirror don't matter in Layer 1; they start mattering in Layer 2.

**Optional head start.** Nagual-QE ships with a curated seed of 515 QE-flavoured patterns in [`seeds/qe-seed-v1.jsonl`](seeds/qe-seed-v1.jsonl). You can import it at any time via `nagual knowledge import --seed seeds/qe-seed-v1.jsonl` — but consider doing the loop with your *own* pattern first (Layer 1 below). The seed is more meaningful when you already know what a pattern is for.

---

## Layer 1 — Experience: do the loop once

Before anything else, store a pattern you actually have. Pick something you solved this week. A flaky test you debugged, a configuration that took three tries, a library quirk that ate an hour. The smaller and more concrete, the better.

Store it:

```bash
nagual knowledge store \
  "Rust async test hangs when using tokio::test with a blocking DB call" \
  --solution "Wrap the blocking call in tokio::task::spawn_blocking, or switch the test to #[tokio::test(flavor = \"multi_thread\")]" \
  --domain "rust.async" \
  --tags "tokio,test,blocking"
```

Nagual prints an ID. Note it. That pattern now lives in your SQLite database at tier *Booster* with reward 0.5 and no reuse count. It is a hypothesis about what works, nothing more.

Next week, when you hit a similar problem, don't solve it from memory. Search first:

```bash
nagual knowledge search "tokio test hangs" --limit 5
```

If the pattern surfaces, apply it. Then — and this is the step most knowledge systems never close — record what happened:

```bash
# Worked on the first try
nagual learn record <pattern-id> success --feedback "Used spawn_blocking, hang gone"

# Didn't work, or made things worse
nagual learn record <pattern-id> failure \
  --failure-mode specification \
  --feedback "This was a different kind of hang — unrelated to blocking calls"
```

Success raises the pattern's reward by 0.1 (capped at 1.0). Failure drops it by 0.15 (floored at 0.0) and forces you to classify *why* it failed using one of five MAST modes: `specification`, `misalignment`, `verification`, `resource`, `unknown`. That classification is how Nagual learns which of your beliefs are unreliable and under what conditions.

**What just happened.** You wrote down something you'd normally forget, got it back when you needed it, and updated your own confidence in it based on reality. That is the entire loop. Everything else in this document is scaffolding around this motion. If you stop after Layer 1 and do nothing else, you will still extract more value from Nagual than most users extract from most knowledge systems. The rest is acceleration.

**Reality check before moving on.** Did the pattern actually come back in the search? If not, it probably needs more specific phrasing in the problem field, or the search query was too narrow. Nagual's full-text search uses SQLite FTS5; it tokenizes on whitespace and punctuation, and it does not do synonyms. "tokio" and "tokio-runtime" are different tokens. You will learn to write problem statements that contain the words your future self will search for. This is a teachable skill, and Layer 2 will make it systematic.

---

## Layer 2 — Mechanics: what actually happened underneath

Now that you have felt the loop, let's name the machinery.

**The pattern record.** Every pattern has a problem, a solution, an optional context, a domain, a list of tags, a confidence (0.0–1.0), a reward that moves with outcomes, a reuse count, a tier, timestamps, and — if ONNX is enabled — a 128-dimensional embedding. The `embedding_method` field records whether that embedding came from the ONNX `all-MiniLM-L6-v2` model or the deterministic SHAKE-256 hash fallback. You should not mix the two in similarity queries; that is why Nagual tags them.

**Dual-write storage.** SQLite is the primary. If PostgreSQL (via the `ruvector-postgres` container on `localhost:5432`, or any reachable Postgres with the `ruvector` extension) is configured in `~/.nagual/config.toml`, every write goes to both. The split exists so local work is fast, while the Postgres mirror supports backup, multi-host sync, and richer vector queries. You can run Nagual with SQLite only for months before you need Postgres. See [`docs/database-setup.md`](docs/database-setup.md) for the setup walkthrough.

**SQLite at rest.** By default Nagual-QE uses plain SQLite. If your deployment needs encryption-at-rest, build with the SQLCipher feature path and store the passphrase in `~/.nagual/config.toml` or a secret manager — *never* in the repo. Back up the database file somewhere safe regardless; the pattern history is user data.

**Full-text search (FTS5).** When you run `nagual knowledge search`, Nagual tokenizes the query and the stored `problem` + `solution` + `domain` fields into an FTS5 index, then returns matches ranked by BM25 score. It does not semantically match — "connection pooling" does not find "database sessions." For semantic search you want the embeddings path (Layer 3's consolidation uses it), and for fuzzy structured filters you want `nagual knowledge list --domain rust --sort-by reward`.

**Tiers.** Every pattern is classified into one of four tiers:

- *Booster* — new or unproven. Default on creation.
- *Crystal* — multiple successful outcomes. Reward has climbed.
- *Reflex* — battle-tested. Reward ≥ 0.9 plus high reuse count. Operational Rule 5 enforces the reward floor.
- *Unclassified* — legacy rows.

Tier is not a filing cabinet; it is a confidence signal. When you search, you can sort by tier to see your most-proven patterns first.

**The reward equation.** Success adds 0.1. Failure subtracts 0.15. The asymmetry is deliberate: Nagual is slightly pessimistic because wrong-confident beliefs cost more than uncertain beliefs, and because most patterns will be applied more times than they are corrected. If you ever want symmetric reward, it's one configuration flag, but I don't recommend it.

**MAST failure classification.** When you record a failure, Nagual forces you to pick one of five modes:

- **specification** — the problem was not what the pattern thought it was.
- **misalignment** — the solution solved the wrong thing, or something adjacent.
- **verification** — you can't tell whether it worked.
- **resource** — something ran out (time, memory, budget, team capacity).
- **unknown** — you genuinely don't know; record this honestly rather than guessing.

Layer 4 will use these classifications to tell you where your whole *class* of patterns tends to fail. Guessing here corrupts that analysis. If you don't know, say `unknown`.

**Sessions.** Every shell in which you run Nagual commands can belong to a session (`nagual session start --domain rust`, `nagual session end`). Tokens consumed by the LLM during the session can be recorded with `nagual session tokens`. Sessions are how Nagual measures "patterns learned per 1000 tokens" — the productivity metric that matters if you care about whether the loop is actually paying for itself.

**Integrity Rule.** Nagual's Constitution (Layer 6) requires zero shortcuts and zero false claims. Mechanically: never record `success` for an outcome you didn't actually observe. Never record `failure` for a pattern you didn't actually apply. The reward numbers are only useful if the data is clean. If you lie to Nagual it will confidently recommend your lies back to you.

---

## Layer 3 — Breakthrough: catalyzing new behavior on top of the loop

Once the loop is running, a second category of commands becomes interesting. These don't just record; they *synthesize*.

**Consolidation.** Run `nagual learn consolidate --similarity 0.9`. Nagual scans all patterns, finds pairs whose embeddings have cosine similarity above the threshold, and merges them — keeping the one with the higher reward, preserving the tags and context from both. This is how a noisy knowledge base becomes a lean one over time without manual cleanup. Operational Rule 3 (`SurpriseReview`) protects you: if a pattern has surprise > 0.8, it will not be silently consolidated; it will be flagged for your review first. Novel knowledge is valuable precisely because it does not match the rest.

**Strategy cache (EGUR).** Patterns store specific *problem → solution* pairs. Strategies store *problem category → general approach*. Use them for the moves you reach for reflexively:

```bash
nagual learn strategy store "debugging" "Binary search for root cause" \
  --steps "reproduce,bisect,isolate,verify,fix"
```

Strategies get their own reward tracking separate from patterns. When you debug, search strategies first, patterns second. You're looking for the right *method* before the right *specific fix*.

**Predictions.** Nagual can hold calibrated predictions about the future and grade you later:

```bash
nagual predict create "API p99 latency < 200ms after caching rollout" --confidence 0.8
# later
nagual predict resolve <id> --outcome true
nagual predict calibration
```

The calibration report tells you, across your whole prediction history, whether your 80% predictions actually come true 80% of the time. If your 80%s come true only 55% of the time, you are systematically overconfident, and Nagual can tell you so with numbers. This is Brier scoring, applied to your own beliefs. It hurts the first few times.

**Knowledge graph and pressure propagation.** Use `nagual graph link <a> <b> --weight 0.8` to connect related patterns, and `nagual graph pressure <id>` to run a GNN-style pressure propagation from one node. What this gives you is second-order search: patterns that are not direct keyword matches but live in the same neighborhood as the ones that are.

**Gene transfusion.** Point `nagual transfuse ./src --dry-run` at a codebase. It runs a set of detectors (Rust error handling, async, API patterns, test patterns, database patterns) and proposes patterns extracted from your actual code. With `--min-confidence 0.6` it shows you which candidates are worth storing. This is the fastest way to bootstrap a knowledge base from an existing project — you don't have to remember everything you've learned; the code remembers for you.

**Deduplication.** `nagual learn dedup --scan` finds exact-duplicate patterns by BLAKE3 hash; `--auto` merges them keeping highest reward. If you ever wonder "did I already store this?" — you probably did, and dedup will tell you.

**Why Layer 3 is its own layer.** Layer 2 records what you do. Layer 3 makes Nagual *proactive* — it proposes consolidations, extracts patterns from code, grades your confidence, propagates pressure across a graph. You stop being the only teacher; the system starts teaching back. This is the earned moment when the loop becomes a collaboration.

---

## Layer 4 — Meta-patterns: reading your own system

Now that you have hundreds of patterns moving through the loop, Nagual can show you the shape of your own practice.

**Time-windowed insights.** `nagual learn insights --windows 7d,30d,90d` produces a report of how many patterns you stored, what domains they fall in, where rewards are climbing and where they're decaying, and which MAST failure modes dominate in each window. If you see `misalignment` dominating your `rust.async` domain over the last 30 days, that is a signal: your solutions in that domain are adjacent to your problems, not aimed at them. The system is telling you where to pay more attention.

**Drift monitoring.** For every domain, Nagual maintains an embedding centroid — the "average" direction of all patterns in that domain. Every new outcome updates it. `nagual learn drift` reports per-domain drift rates. If a domain's centroid is drifting fast, the concept you're calling "rust.async" has shifted; what you meant by it six months ago is not what you mean by it today. Usually this is good (you're learning). Occasionally it's bad (your domain labels have become meaningless).

**Bayesian quality scoring.** Beyond the linear reward, every pattern has a `bayesian_score` backed by Beta(α, β) parameters. Each success bumps α; each failure bumps β. The posterior distribution gives you not just a point estimate of quality but a *range* — a narrow distribution means the pattern is well-understood, wide means you need more observations. Display via `p.bayesian_score()`. For decisions where precision matters (e.g., promoting to reflex tier), use Bayesian; for quick ranking, linear reward is fine.

**Validation scenarios (holdout set).** Create scenarios that patterns get evaluated against without being able to train on them:

```bash
nagual learn scenario create \
  --domain rust.async \
  --description "Test timeout handling" \
  --context "Long-running async operation" \
  --expected "Should timeout gracefully" \
  --difficulty hard
```

Then `nagual learn scenario evaluate <pattern-id> --domain rust` runs that pattern against all holdout scenarios in the domain. This is how you guard against overfitting to your own history — your patterns should still work on cases you deliberately set aside.

**Graph clustering (mincut).** With the `mincut` feature flag compiled in, `nagual graph cluster --threshold 1.0` runs Stoer-Wagner on the pattern graph and reveals coherent clusters. These are the implicit subdomains of your practice — the things you've been treating as "rust" that are actually three distinct subdomains with minimal overlap. Useful input when you're renaming domain labels.

**Surprise and temporal decay.** Two dials tune Nagual's priorities. *Surprise* is how novel a pattern is compared to what already exists; surprise > 0.8 blocks consolidation (Operational Rule 3) because genuinely new things should never be silently merged into old ones. *Temporal decay* — `relevance = quality * exp(-0.0077 * age_days) * surprise` — makes stale patterns fade without deleting them. Stale patterns are still retrievable; they just rank lower. Nagual keeps its Tonal lean.

**Why Layer 4 is its own layer.** Layer 3 taught you to act on individual patterns. Layer 4 teaches you to act on *distributions* of patterns. Drift, calibration, scenarios, clusters — these are statements about the shape of your knowledge, not about any one record. At this layer you stop using Nagual as a notebook and start using it as a mirror.

---

## Layer 5 — Integrations: connecting the loop to the world

The loop is only useful if it runs where your work runs. Nagual integrates in four directions.

**HTTP API and authentication.** `nagual serve` exposes the CLI surface over HTTP with cookie-based dashboard sessions and per-agent API keys (`ngk_<32-hex>` format, BLAKE3-hashed at rest). Keys are issued with scopes (`read`, `write`, `admin`) and can be revoked. If you're building a second agent that needs to contribute to the knowledge base without impersonating you, issue it its own API key and track its `last_used_at` to notice when it goes silent. See [`docs/gcloud-deploy.md`](docs/gcloud-deploy.md) for a production-deployment recipe.

**Editor / agent hooks.** Nagual is designed to be driven by shell hooks from whatever IDE or coding agent you use (Claude Code, Cursor, a local script, cron). The integration pattern is simple: call `nagual knowledge search`, `nagual knowledge store`, or `nagual learn record` from `pre-task`, `post-task`, `post-edit`, `post-bash`, or session-lifecycle hooks. Anything your editor can call as a shell command can be wired into the loop. The repo does not ship any editor-specific hooks by default — they belong in your dotfiles, not the code.

**Optional Brain sync.** Compiled with the `brain-sync` feature flag, Nagual can push high-reward patterns to an external collective-knowledge endpoint via `nagual sync brain share <pattern-id>`. There is **no default endpoint** — set `BRAIN_URL` explicitly to opt in. Before anything leaves the machine, the PII redactor runs: twelve regex patterns strip absolute paths, IPs, emails, API keys, AWS keys, SSH keys, JWTs, connection strings, phone numbers, credit cards, SSNs, and Nagual keys. Local SQLite is never modified — redaction applies only to cloud-bound copies. If you do not trust the redactor, don't share; the feature is opt-in twice (feature flag, and explicit URL).

**Google Cloud backup.** The `sync/` subsystem handles incremental backup to a GCloud Storage bucket with optional CMEK encryption. Configure in `~/.nagual/config.toml` under `[sync.gcloud]` and set the scheduler interval (default 300s incremental, 24h full). Restore with `nagual sync restore <timestamp>`. S3/B2/local-disk backends work with the same interface — adapt the `gsutil cp` in `deploy/nagual-backup.sh` if you prefer another provider.

**Self-learning hooks, revisited.** Once the hooks are active, Layer 1's loop runs without you having to remember it. Patterns get stored from your `git commit` messages, insights from fixed compile errors, outcomes from passing or failing tests. This is the level at which Nagual becomes invisible infrastructure. You don't use it; it happens around you.

**Why Layer 5 is its own layer.** Integrations change *how often* the loop runs. A well-used Nagual with no integrations does the loop a few times a day. An integrated Nagual does it dozens of times a day without asking permission. That difference compounds.

---

## Layer 6 — Constitution: earned recognition

If you've worked through Layers 1–5, the principles below will read like descriptions of what you already do. If they read like abstract ethics, go back and close a few more loops; the words don't unlock anything until the mechanics do.

The full text lives in [`NAGUAL_CONSTITUTION.md`](NAGUAL_CONSTITUTION.md). Eight principles, five operational rules, one amendment process. What follows is the earned-recognition gloss — the reason each principle is there, told in the language of what you've already seen.

**Principle 0 — Seek Truth.** You have watched rewards move on real outcomes. You know that a pattern with inflated confidence is not just wrong; it is *expensively* wrong, because it will be recommended back to you. Truth is the only substrate where the reward loop produces signal. Anything else is noise amplified.

**Principle 1 — Partnership, not replacement.** Nagual did not solve your problems. It remembered what you solved and what failed, and it showed you the distribution. The intelligence is in the loop between you; the system is a conduit.

**Principle 2 — User is partner and creator.** Any collaborator that never challenges the person running it is decorative. Nagual's job is to surface uncomfortable truths from the data — that your `rust.async` patterns are failing on `misalignment` 40% of the time, that your 80% predictions land at 55%, that the domain you called "testing" is actually three disjoint clusters. This is the partnership, in mechanical form.

**Principle 3 — Achieve through impeccability.** Every command in Nagual has a budget: your attention, your context window, your disk. The warrior's impeccability is not moral perfection; it is using that budget wisely. Consolidation, temporal decay, tier promotion — all of these are Nagual being stingy with your scarce resources so the signal stays high.

**Principle 4 — Epistemic humility.** Brier calibration made this unavoidable. You cannot use Nagual for six months and not learn how often you are wrong. The mature response to that data is not shame; it is to say "I might be wrong about this" earlier, more often, and with specifics.

**Principle 5 — Do no harm.** Operational Rule 1 (NeverDeleteWithoutBackup) and Rule 4 (ConflictEscalation) are Principle 5 in code. If you're unsure whether a delete or a merge is safe, the system is built to pause rather than to move fast. PAUSE is not a weakness; it is the signature of a system that can be trusted with your long-term memory.

**Principle 6 — Transparency.** Every pattern has provenance — who stored it, when, with what confidence, what outcomes. Every prediction has calibration. Every embedding tracks its method. Nothing is hidden, including Nagual's mistakes. Transparency is what makes the reward loop auditable rather than magical.

**Principle 7 — The Warrior's Optimization Loop.** Surprise scoring. Self-improvement cycles. Strategy cache. Temporal decay. These are not separate features; they are one loop asking the same question at different scopes: *what has no one ever thought of?* Good enough is the enemy of great, great is the enemy of revolutionary, and the system's job is to keep nudging you past "good enough" without letting you pretend you've already arrived.

**The five operational rules.** These are the principles executable:

- **NeverDeleteWithoutBackup** — 24-hour backup window before destructive operations.
- **AlwaysRecordMAST** — failures without classification corrupt analysis, so they're blocked.
- **SurpriseReview** — novel patterns (surprise > 0.8) require human review before merging.
- **ConflictEscalation** — overwrites create a conflict record rather than a silent replacement.
- **MinimumRewardForReflex** — reflex tier requires reward ≥ 0.9. Earned, not declared.

Force-overrides exist for every rule. Every force-override is permanently logged. If you are overriding often, the rules are not wrong; your workflow is.

**The Tonal and the Nagual.** Castaneda's frame: the Tonal is the island of the known, the Nagual is the vast unknown around it. The warrior's path is not to destroy the Tonal to reach the Nagual — that way is chaos — but to make the Tonal *lean and efficient* so the Nagual can emerge without overwhelming it. Every mechanic you have learned — consolidation, decay, tier promotion, clustering, redaction, backup — is a way of keeping the Tonal lean. The Nagual is what shows up when the Tonal isn't drowning you in noise: the pattern you didn't know you knew, the connection you didn't plan, the insight that arrives because the room is finally quiet enough to hear it.

---

## Closing: the loop, recursively

The earned-recognition move is itself a Nagual pattern. Store it:

```bash
nagual knowledge store \
  "Onboarding users to a learning system by declaring principles up front produces fluent performance without underlying understanding" \
  --solution "Layer the onboarding so each later layer is earned by the work in the previous one. Mechanics before meta, meta before ethics, ethics last. Principles read like summaries rather than instructions by the time the learner reaches them." \
  --domain "pedagogy.onboarding" \
  --tags "earned-recognition,layered-learning,pattern-space-adjacent"
```

If you use this document and it works, record success against that pattern. If it doesn't work, record failure with a `specification` MAST mode and a feedback note telling us what the wrong layer ordering was. The document you are reading is itself a pattern being tested by the loop it describes. That is not decoration; that is how the loop is supposed to work.

The navigation continues. Forever beginning. Always present.

---

*See also: [`NAGUAL_CONSTITUTION.md`](NAGUAL_CONSTITUTION.md) (the full constitutional text), [`docs/setup.md`](docs/setup.md) (installation), [`docs/database-setup.md`](docs/database-setup.md) (SQLite + optional Postgres), [`docs/gcloud-deploy.md`](docs/gcloud-deploy.md) (production deployment), [`docs/architecture.md`](docs/architecture.md) (module map + data flow), [`docs/seeding-qe.md`](docs/seeding-qe.md) (importing the QE seed).*
