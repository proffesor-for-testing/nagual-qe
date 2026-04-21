#!/usr/bin/env bash
# =================================================================
# Nagual-QE demo — golden path through the core workflow
#
# Run this after `bash scripts/mac-setup.sh` (or equivalent on Linux).
# It stores three patterns in a fresh temp DB, searches them, records
# an outcome, imports the bundled QE seed, and shows the final stats.
#
# Usage:
#   bash scripts/demo.sh
#
# To record as an asciinema cast:
#   brew install asciinema                 # (or: pipx install asciinema)
#   asciinema rec demo.cast -c 'bash scripts/demo.sh'
#   # Convert to SVG (optional):
#   npm i -g svg-term-cli
#   svg-term --in demo.cast --out docs/demo.svg --window --no-cursor
# =================================================================
set -euo pipefail

DEMO_DB="$(mktemp -d)/nagual-demo.db"
NAGUAL="${NAGUAL_BIN:-nagual}"

hr() { printf '\n\033[90m─────────────────────────────────────────────────────\033[0m\n'; }
say() { printf '\n\033[36m$ %s\033[0m\n' "$*"; }

hr
echo "Nagual-QE demo — temp DB at $DEMO_DB"
hr

# ── Store three QE patterns ─────────────────────────────────────────
say "$NAGUAL knowledge store 'Flaky async test caused by race in setup fixture' ..."
$NAGUAL knowledge store \
  "Flaky async test caused by race in setup fixture" \
  --solution "Add explicit await on the shared fixture. The fixture runs in a tokio::spawn and the test starts before it completes ~5% of the time. Using an Arc<Notify> fixes this." \
  --domain "qe.flaky" \
  --tags "async,race-condition,tokio" \
  --db-path "$DEMO_DB"

say "$NAGUAL knowledge store 'Test coverage dropped 4% after refactor' ..."
$NAGUAL knowledge store \
  "Test coverage dropped 4% after refactor" \
  --solution "The refactor introduced three new enum variants but no tests exercised them. Run the coverage-drop-investigator to trace which files lost coverage, then add property tests covering each new variant." \
  --domain "qe.coverage" \
  --tags "coverage,refactor,property-testing" \
  --db-path "$DEMO_DB"

say "$NAGUAL knowledge store 'Agents overclaim completion when tests are stubbed' ..."
$NAGUAL knowledge store \
  "Agents overclaim completion when tests are stubbed" \
  --solution "Run /audit after swarm completion — verify cargo check + cargo test + grep for stub bodies. Compare claimed vs actual. Never trust agent-reported status without independent verification." \
  --domain "agentic-qe" \
  --tags "audit,verification,agents" \
  --db-path "$DEMO_DB"

# ── Search ──────────────────────────────────────────────────────────
hr
say "$NAGUAL knowledge search 'flaky async'"
$NAGUAL knowledge search "flaky async" --limit 3 --db-path "$DEMO_DB"

hr
say "$NAGUAL knowledge search 'agents trust verification'"
$NAGUAL knowledge search "agents trust verification" --limit 3 --db-path "$DEMO_DB"

# ── Record an outcome (reward goes up) ──────────────────────────────
hr
say "$NAGUAL knowledge list --domain qe.flaky"
PATTERN_ID=$($NAGUAL knowledge list --domain "qe.flaky" --db-path "$DEMO_DB" --json 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["id"])' 2>/dev/null || echo "")

if [ -n "$PATTERN_ID" ]; then
  say "$NAGUAL learn record $PATTERN_ID success --feedback 'Worked on the CI pipeline too'"
  $NAGUAL learn record "$PATTERN_ID" success \
    --feedback "Worked on the CI pipeline too" \
    --db-path "$DEMO_DB" || true
fi

# ── Status ──────────────────────────────────────────────────────────
hr
say "$NAGUAL status"
$NAGUAL status --db-path "$DEMO_DB"

hr
echo
echo "Demo DB: $DEMO_DB  (delete when done)"
