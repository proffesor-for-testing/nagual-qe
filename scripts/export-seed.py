#!/usr/bin/env python3
"""
Export a QE-flavoured seed bundle from a Nagual instance.

Fetches all patterns from a running `nagual serve` instance (via the cloud
API), filters to QE-relevant domains, applies PII redaction, and writes a
JSONL seed file suitable for `nagual knowledge import --seed`.

Usage:
    NAGUAL_CLOUD_URL=https://your-nagual/api \
    NAGUAL_CLOUD_KEY=ngk_... \
    NAGUAL_SEED_NAMES="Your Name,Your Alias" \
    NAGUAL_SEED_GH_HANDLE="your-gh-handle" \
    NAGUAL_SEED_GCP_PROJECTS="your-project-id" \
    python3 scripts/export-seed.py --output seeds/qe-seed-v1.jsonl

Privacy: every problem/solution/context string passes through a redactor
that strips absolute paths, emails, IPs, API keys, and internal-project
URLs. Additional first-party names/handles/projects can be supplied via
the NAGUAL_SEED_* env vars above. ALWAYS review the output manually
before publishing a seed.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Domain filters
# ──────────────────────────────────────────────────────────────────────

# If a pattern's domain matches any of these prefixes/regexes, INCLUDE it.
INCLUDE_DOMAIN_PATTERNS = [
    r'^agentic-qe(\.|$)',
    r'^agentic-quality-engineering(\.|$)',
    r'^agentic-testing',
    r'^agentic\.(qe|quality|testing)',
    r'^agents\.testing',
    r'^quality-engineering(\.|$)',
    r'^quality(\.|-|$)',
    r'^qe(\.|-|$)',
    r'^testing(\.|$)',
    r'^test-?automation',
    r'^test-?maturity',
    r'^test-?design',
    r'^test-?success',
    r'^test-?patterns?',
    r'^tdd(\.|$)',
    r'^bdd(\.|$)',
    r'^holistic-testing',
    r'^risk-based-testing',
    r'^context-driven-testing',
    r'^shift-left-testing',
    r'^shift-right-testing',
    r'^integration-testing',
    r'^security-testing',
    r'^mutation-testing',
    r'^chaos-engineering',
    r'^exploratory-testing',
    r'^regression-testing',
    r'^compatibility-testing',
    r'^accessibility-testing',
    r'^compliance-testing',
    r'^performance-testing',
    r'^database-testing',
    r'^mobile-testing',
    r'^visual-testing',
    r'^localization-testing',
    r'^api-testing',
    r'^skills\.(qe|test|tdd|bdd|mutation|chaos|holistic|exploratory|shift|contract|regression|database|api-|visual-|performance-|security-|compatibility-|accessibility-|compliance-|localization-|mobile-|verification|code-review|n8n-|testability-|risk-based|context-driven|refactoring-|bug-reporting|pair-programming|skill-|technical-|debug-|xp-)',
    r'^methodology\.(testing|holistic)',
    r'^rst\.',
    r'^htsm\.',
    r'^quality-forge',
    r'^quality-metrics',
    r'^quality-philosophy',
    r'^safety\.sensing\.testing',
    r'^presentation\.agentic-qe',
    r'^flow-nexus-testing',
    r'^claude-skills-zip\.(risk-based|tdd-|holistic-|exploratory-|quality-|performance-|security-|context-driven|api-|code-review|test-automation)',
    r'^qe&cf-claude-skills',
    r'^ai-quality-strategy',
    r'^agentic-engineering\.(agent-quality-gates|patterns\.spec-tdd)',
    r'^holistic-testing',
    r'^test-feed$',
    r'^test-success$',
    r'^quality-assessment',
]

# If a pattern's domain matches any of these, EXCLUDE it even if previously
# included. Applied second. Extend via NAGUAL_SEED_EXCLUDE_DOMAINS env var.
EXCLUDE_DOMAIN_PATTERNS = [
    r'^clients$',                   # client rosters
    r'^context-compact$',           # session context dumps
    r'^compaction-flush$',          # session context dumps
    r'^test\.cloud-sync$',          # internal nagual tests
    r'^test\.dual_write$',          # internal nagual tests
    r'^test\.config_fallback$',     # internal nagual tests
    r'^test$',                      # too generic
    r'\.changelog$',                # release notes
    r'\.releases$',                 # release notes
    r'\.invoices',                  # billing
]
# Optional user-supplied exclusion regexes (comma-separated regex strings).
_extra_ex = os.environ.get("NAGUAL_SEED_EXCLUDE_DOMAINS", "")
EXCLUDE_DOMAIN_PATTERNS.extend(x.strip() for x in _extra_ex.split(",") if x.strip())

INCLUDE_RX = re.compile("|".join(INCLUDE_DOMAIN_PATTERNS), re.I)
EXCLUDE_RX = re.compile("|".join(EXCLUDE_DOMAIN_PATTERNS), re.I)

# ──────────────────────────────────────────────────────────────────────
# PII redactor
# ──────────────────────────────────────────────────────────────────────

REDACTIONS: list[tuple[re.Pattern, str]] = [
    # Absolute home paths (macOS + Linux)
    (re.compile(r"/Users/[A-Za-z0-9_-]+"), "/Users/REDACTED"),
    (re.compile(r"/home/[A-Za-z0-9_-]+"), "/home/REDACTED"),
    # Email
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL_REDACTED]"),
    # Public IPv4 (leave private ranges alone — they're less identifying
    # and often appear in infra docs). We redact anything that looks like
    # a full IP address since it's rarely load-bearing in QE patterns.
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_REDACTED]"),
    # Nagual API keys
    (re.compile(r"\bngk_[A-Fa-f0-9]{8,}\b"), "[NAGUAL_KEY_REDACTED]"),
    # OpenAI / Anthropic / generic API keys
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[API_KEY_REDACTED]"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "[API_KEY_REDACTED]"),
    # AWS
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[AWS_KEY_REDACTED]"),
    # GitHub tokens
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), "[GITHUB_TOKEN_REDACTED]"),
    # JWT (3 base64url segments)
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
     "[JWT_REDACTED]"),
    # Personal names — configure via NAGUAL_SEED_NAMES env var (comma-
    # separated list of names/aliases/handles to redact to [USER]).
    # Keep public citations (e.g. known conference speakers) intact by
    # listing only first-party names.
    # Personal / internal domains — configure via NAGUAL_SEED_DOMAINS env var
    # (comma-separated list of hostnames like "mybrand.com,mytool.example.net").
    # Personal GitHub handle — configure via NAGUAL_SEED_GH_HANDLE env var.
    # (Default catches only the explicit placeholder form.)
    # GCP project IDs — catch the generic "ferrous-griffin-NNNN-xxxx" form
    # (GCP's auto-generated default project ids). Configure specific
    # project names via NAGUAL_SEED_GCP_PROJECTS env var.
    (re.compile(r"\b[a-z]+-[a-z]+-\d{4,}-[a-z0-9]{2,}\b"), "YOUR_GCP_PROJECT"),
    # GCS buckets — configure your bucket via NAGUAL_SEED_GCS_BUCKETS env var.
    # Generic Cloudflare tunnel ID pattern (UUID in cloudflared-style URLs).
    # We leave plain UUIDs alone because pattern IDs are also UUIDs; only
    # redact when they appear in tunnel/creds contexts via the env var.
    # Third-party endpoints — configure via NAGUAL_SEED_THIRD_PARTY_URLS
    # env var if you use a specific collective-knowledge service.
    # Client / engagement names — configure via NAGUAL_SEED_CLIENTS env var
    # (comma-separated list of company/project names to redact to [CLIENT]).
    # Internal-tool names — via NAGUAL_SEED_TOOLS (redacted to [TOOL]).
    # Internal-org names  — via NAGUAL_SEED_ORGS (redacted to [ORG]).
    # Internal project dir names — via NAGUAL_SEED_PROJECT_DIRS (redacted
    # to the string "nagual-qe").
    # SSH / git URLs — generic form (any github.com repo reference).
    # Leave public-domain repos alone by listing specific internal ones
    # via NAGUAL_SEED_GIT_REPOS env var.
]

# Optional user-supplied extras — loaded from env vars so the public
# redactor stays generic. Each env var is a comma-separated list; every
# entry becomes a case-insensitive whole-word match to the given label.
def _extras_from_env() -> list[tuple[re.Pattern, str]]:
    ENV_TO_LABEL = [
        ("NAGUAL_SEED_NAMES",             "[USER]"),
        ("NAGUAL_SEED_CLIENTS",           "[CLIENT]"),
        ("NAGUAL_SEED_TOOLS",             "[TOOL]"),
        ("NAGUAL_SEED_ORGS",              "[ORG]"),
        ("NAGUAL_SEED_GCP_PROJECTS",      "YOUR_GCP_PROJECT"),
        ("NAGUAL_SEED_GCS_BUCKETS",       "gs://YOUR_BUCKET"),
        ("NAGUAL_SEED_TUNNEL_IDS",        "[TUNNEL_ID]"),
        ("NAGUAL_SEED_PROJECT_DIRS",      "nagual-qe"),
        ("NAGUAL_SEED_DOMAINS",           "example.com"),
        ("NAGUAL_SEED_THIRD_PARTY_URLS",  "[BRAIN_URL]"),
        ("NAGUAL_SEED_GIT_REPOS",         "git@github.com:YOUR_ORG/nagual-qe.git"),
    ]
    extras: list[tuple[re.Pattern, str]] = []
    for env, label in ENV_TO_LABEL:
        for v in (x.strip() for x in os.environ.get(env, "").split(",") if x.strip()):
            # GCS buckets: user supplies the bucket name; match with gs:// prefix.
            if env == "NAGUAL_SEED_GCS_BUCKETS":
                extras.append((re.compile(r"gs://" + re.escape(v) + r"\S*"), label))
            # Tunnel IDs: literal UUID match.
            elif env == "NAGUAL_SEED_TUNNEL_IDS":
                extras.append((re.compile(re.escape(v)), label))
            # Handles (no \b because hyphens don't match \w).
            elif env == "NAGUAL_SEED_GH_HANDLE":
                extras.append((re.compile(re.escape(v)), label))
            else:
                extras.append((re.compile(r"\b" + re.escape(v) + r"\b", re.I), label))
    # GitHub handle is its own env var for backwards compat.
    gh = os.environ.get("NAGUAL_SEED_GH_HANDLE", "")
    for h in (x.strip() for x in gh.split(",") if x.strip()):
        extras.append((re.compile(re.escape(h)), "YOUR_ORG"))
    return extras

REDACTIONS = REDACTIONS + _extras_from_env()

CATEGORIES_SEEN: Counter = Counter()

def redact(text: str | None) -> tuple[str, list[str]]:
    """Return (redacted_text, list_of_categories_hit)."""
    if not text:
        return text or "", []
    categories: list[str] = []
    out = text
    for rx, repl in REDACTIONS:
        new_out, n = rx.subn(repl, out)
        if n:
            cat = repl.strip("[]").lower()
            categories.append(cat)
            CATEGORIES_SEEN[cat] += n
        out = new_out
    return out, categories

# ──────────────────────────────────────────────────────────────────────
# Fetch + filter
# ──────────────────────────────────────────────────────────────────────

def load_key() -> str:
    key = os.environ.get("NAGUAL_CLOUD_KEY")
    if key:
        return key
    env_file = Path.home() / ".nagual" / "cloud-hooks.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            m = re.match(r'(?:export\s+)?NAGUAL_CLOUD_KEY\s*=\s*"?([^"\n]+)"?', line.strip())
            if m:
                return m.group(1)
    sys.exit("No NAGUAL_CLOUD_KEY in env or ~/.nagual/cloud-hooks.env")

def fetch_all(base: str, key: str, limit: int = 200) -> list[dict]:
    hdr = {"Authorization": f"Bearer {key}", "User-Agent": "nagual-qe-seed-export/0.1"}
    out: list[dict] = []
    offset = 0
    while True:
        req = urllib.request.Request(f"{base}/patterns?limit={limit}&offset={offset}", headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            sys.exit(f"HTTP {e.code} at offset={offset}: {e.reason}")
        patterns = data.get("patterns", [])
        if not patterns:
            break
        out.extend(patterns)
        total = data.get("total", 0)
        print(f"  fetched {len(out)}/{total}", end="\r", file=sys.stderr)
        offset += limit
        if offset >= total:
            break
        time.sleep(0.05)
    print(file=sys.stderr)
    return out

def is_qe(domain: str) -> bool:
    if not domain:
        return False
    if EXCLUDE_RX.search(domain):
        return False
    return bool(INCLUDE_RX.search(domain))

def dedup_key(p: dict) -> str:
    # Stable dedup on problem prefix + domain — catches exact duplicates
    # while allowing distinct-in-spirit patterns in the same domain.
    return (p.get("problem", "")[:200].strip().lower() + "|" + p.get("domain", ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("NAGUAL_CLOUD_URL", "http://localhost:3333/api"))
    ap.add_argument("--output", default="seeds/qe-seed-v1.jsonl")
    ap.add_argument("--input", help="JSON dump (skip API fetch if provided)")
    ap.add_argument("--min-solution-length", type=int, default=80,
                    help="Drop patterns whose solution is shorter than this")
    ap.add_argument("--limit-per-domain", type=int, default=None,
                    help="Cap patterns per domain (useful for oversized domains like rust.testing)")
    args = ap.parse_args()

    if args.input:
        patterns = json.loads(Path(args.input).read_text())
        print(f"Loaded {len(patterns)} patterns from {args.input}", file=sys.stderr)
    else:
        key = load_key()
        print(f"Fetching from {args.base} ...", file=sys.stderr)
        patterns = fetch_all(args.base, key)
        print(f"Fetched {len(patterns)} patterns", file=sys.stderr)

    # 1. Domain filter
    kept = [p for p in patterns if is_qe(p.get("domain", ""))]
    print(f"After domain filter: {len(kept)}", file=sys.stderr)

    # 2. Minimum solution length
    kept = [p for p in kept if len((p.get("solution") or "").strip()) >= args.min_solution_length]
    print(f"After length filter (>={args.min_solution_length} chars): {len(kept)}", file=sys.stderr)

    # 3. Dedup by (problem prefix + domain)
    seen: set[str] = set()
    unique: list[dict] = []
    for p in kept:
        k = dedup_key(p)
        if k in seen:
            continue
        seen.add(k)
        unique.append(p)
    print(f"After dedup: {len(unique)}", file=sys.stderr)

    # 4. Optional per-domain cap
    if args.limit_per_domain:
        by_dom: dict[str, list[dict]] = {}
        for p in sorted(unique, key=lambda x: -(x.get("reward") or 0.5)):
            by_dom.setdefault(p["domain"], []).append(p)
        unique = []
        for dom, ps in by_dom.items():
            unique.extend(ps[: args.limit_per_domain])
        print(f"After per-domain cap ({args.limit_per_domain}): {len(unique)}", file=sys.stderr)

    # 5. Redact + write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    domain_hist: Counter = Counter()
    # Domain-name-level scrubbing — neutralize client/company/personal
    # markers that appear as segments in pattern domain names. The list
    # of markers comes from NAGUAL_SEED_DOMAIN_MARKERS env var
    # (comma-separated; case-insensitive; "." and "-" word boundaries).
    markers_raw = os.environ.get("NAGUAL_SEED_DOMAIN_MARKERS", "")
    markers = [re.escape(m.strip()) for m in markers_raw.split(",") if m.strip()]
    domain_scrub: list[tuple[re.Pattern, str]] = []
    if markers:
        alt = "|".join(markers)
        domain_scrub = [
            (re.compile(rf"\.(?:{alt})(?=\.|$|-|_)", re.I), ".generic"),
            (re.compile(rf"^(?:{alt})(?=\.|-|_)", re.I), "generic"),
        ]
    with out_path.open("w") as f:
        for p in unique:
            problem, _ = redact(p.get("problem", ""))
            solution, _ = redact(p.get("solution", ""))
            context, _ = redact(p.get("context") or "")
            domain = p.get("domain", "unknown")
            for rx, repl in domain_scrub:
                domain = rx.sub(repl, domain)
            domain_hist[domain] += 1
            record = {
                "problem": problem,
                "solution": solution,
                "domain": domain,
                "reward": p.get("reward", 0.5),
                "tier": p.get("tier", "booster"),
            }
            if context:
                record["context"] = context
            # Keep empty tags list for schema stability
            record["tags"] = []
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(unique)} patterns to {out_path}", file=sys.stderr)
    print(f"Total size: {out_path.stat().st_size // 1024} KB", file=sys.stderr)
    print(f"\nRedactions applied (category: count):", file=sys.stderr)
    for cat, n in CATEGORIES_SEEN.most_common():
        print(f"  {n:5d}  {cat}", file=sys.stderr)
    print(f"\nTop 20 domains in seed:", file=sys.stderr)
    for d, n in domain_hist.most_common(20):
        print(f"  {n:5d}  {d}", file=sys.stderr)

if __name__ == "__main__":
    main()
