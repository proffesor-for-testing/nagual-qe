#!/bin/bash
# =================================================================
# Nagual-QE macOS Setup Script
# Builds nagual, installs it to PATH, and creates default config.
#
# GCloud setup is OPTIONAL — configure via ~/.nagual/config.toml after
# running this script, or follow docs/gcloud-deploy.md.
#
# Env overrides:
#   NAGUAL_HOME       — config/data dir (default: $HOME/.nagual)
#   INSTALL_DIR       — where to place the binary (default: $HOME/.local/bin)
#   ORT_DYLIB_PATH    — ONNX Runtime dylib (default: /opt/homebrew/lib/libonnxruntime.dylib)
# =================================================================
set -euo pipefail

NAGUAL_HOME="${NAGUAL_HOME:-$HOME/.nagual}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
ORT_DYLIB_PATH_DEFAULT="${ORT_DYLIB_PATH:-/opt/homebrew/lib/libonnxruntime.dylib}"

echo "=== Nagual-QE macOS Setup ==="
echo ""

# ---------------------------------------------------------------
# 1. Check prerequisites
# ---------------------------------------------------------------
echo "[1/4] Checking prerequisites..."

if ! command -v rustc &> /dev/null; then
  echo "  Rust not found. Installing via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
echo "  Rust: $(rustc --version)"

if [ ! -f "$ORT_DYLIB_PATH_DEFAULT" ]; then
  echo "  WARN: ONNX Runtime not found at $ORT_DYLIB_PATH_DEFAULT"
  echo "        Install with: brew install onnxruntime"
  echo "        Or build with --no-default-features --features kos to skip ONNX"
fi

# ---------------------------------------------------------------
# 2. Build nagual
# ---------------------------------------------------------------
echo ""
echo "[2/4] Building nagual..."

if [ ! -f "Cargo.toml" ] || ! grep -q 'name = "nagual"' Cargo.toml; then
  echo "  ERROR: Run this script from the nagual-qe repo root."
  exit 1
fi

export ORT_DYLIB_PATH="$ORT_DYLIB_PATH_DEFAULT"
cargo build --release
NAGUAL_BIN="./target/release/nagual"
echo "  Built: $NAGUAL_BIN"

# ---------------------------------------------------------------
# 3. Install binary to PATH
# ---------------------------------------------------------------
echo ""
echo "[3/4] Installing nagual to $INSTALL_DIR..."

mkdir -p "$INSTALL_DIR"
cp "$NAGUAL_BIN" "$INSTALL_DIR/nagual"
chmod +x "$INSTALL_DIR/nagual"

# macOS: re-sign the copied binary (adhoc) — otherwise dyld may hang
if command -v codesign &> /dev/null; then
  codesign --force --sign - "$INSTALL_DIR/nagual" 2>/dev/null || true
fi

# Add to PATH if not already there
if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
  SHELL_RC="$HOME/.zshrc"
  [ -f "$HOME/.bashrc" ] && SHELL_RC="$HOME/.bashrc"
  echo "export PATH=\"$INSTALL_DIR:\$PATH\"" >> "$SHELL_RC"
  echo "  Added $INSTALL_DIR to PATH in $SHELL_RC — restart your shell to pick it up"
  export PATH="$INSTALL_DIR:$PATH"
fi

echo "  Installed: $(command -v nagual || echo "$INSTALL_DIR/nagual")"

# ---------------------------------------------------------------
# 4. Create config directory and default config (if missing)
# ---------------------------------------------------------------
echo ""
echo "[4/4] Creating config at $NAGUAL_HOME/config.toml..."

mkdir -p "$NAGUAL_HOME"

if [ ! -f "$NAGUAL_HOME/config.toml" ]; then
  cat > "$NAGUAL_HOME/config.toml" <<'TOML'
# Nagual-QE Configuration (local defaults)
# See docs/setup.md for all options.

[database]
sqlite_path = "~/.nagual/nagual.db"
# Uncomment to enable dual-write to PostgreSQL:
# postgres_url = "postgresql://nagual:change_me@localhost:5432/nagual"

[sync]
enabled = false          # set to true after configuring sync.gcloud below
mode = "local"           # "local" | "gcloud" | "disabled"
interval_minutes = 30
auto_backup = true

# ---- Optional: Google Cloud Storage backup ----
# See docs/gcloud-deploy.md for bucket + IAM setup.
# [sync.gcloud]
# bucket = "your-bucket-name"
# project = "your-gcp-project-id"
# prefix = "nagual-sync"

[sync.backup]
directory = "~/.nagual/backups"
compression_level = 6
max_backups = 10

[learning]
consolidation_interval_hours = 6
high_reward_threshold = 0.8
low_reward_threshold = 0.4

[search]
hnsw_m = 24
hnsw_ef_construction = 200
hnsw_ef_search = 200
TOML
  echo "  Created $NAGUAL_HOME/config.toml"
else
  echo "  Config already exists at $NAGUAL_HOME/config.toml — leaving unchanged"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Verify:       nagual status"
echo "  2. Store a note: nagual knowledge store 'My first insight' --solution 'What I learned'"
echo "  3. Search:       nagual knowledge search 'insight'"
echo ""
echo "Optional:"
echo "  * Import QE seed: see docs/seeding-qe.md"
echo "  * Cloud deploy:   see docs/gcloud-deploy.md"
echo "  * Dashboard:      nagual serve  (then open http://localhost:3333)"
