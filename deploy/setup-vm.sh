#!/usr/bin/env bash
# =================================================================
# Nagual-QE — GCE VM setup (arm64)
# Provisions a fresh Debian 12 arm64 VM, installs Docker, ONNX
# Runtime, cloudflared, Rust, and a dedicated `nagual` service user.
#
# This script is fully parameterized — set these env vars before running:
#
#   NAGUAL_GCP_PROJECT   (required) — your GCP project ID
#   NAGUAL_GCP_ZONE      (default: us-central1-a)
#   NAGUAL_VM_NAME       (default: nagual-vm)
#   NAGUAL_DISK_NAME     (default: nagual-data)
#   NAGUAL_DISK_SIZE     (default: 50GB)
#   NAGUAL_MACHINE_TYPE  (default: t2a-standard-1  [arm64])
#
# Prereqs: gcloud CLI authenticated and Compute Engine API enabled.
# =================================================================
set -euo pipefail

PROJECT="${NAGUAL_GCP_PROJECT:?set NAGUAL_GCP_PROJECT to your GCP project id}"
ZONE="${NAGUAL_GCP_ZONE:-us-central1-a}"
VM_NAME="${NAGUAL_VM_NAME:-nagual-vm}"
DISK_NAME="${NAGUAL_DISK_NAME:-nagual-data}"
DISK_SIZE="${NAGUAL_DISK_SIZE:-50GB}"
MACHINE_TYPE="${NAGUAL_MACHINE_TYPE:-t2a-standard-1}"

echo "=== Nagual-QE GCE Setup ==="
echo "  Project:      $PROJECT"
echo "  Zone:         $ZONE"
echo "  VM name:      $VM_NAME"
echo "  Machine type: $MACHINE_TYPE"
echo ""

# ──────────────────────────────────────────────
# Phase 1: Enable APIs + Create Infrastructure
# ──────────────────────────────────────────────

echo "[1/5] Enabling Compute Engine API..."
gcloud services enable compute.googleapis.com --project "$PROJECT" 2>/dev/null || true

echo "[2/5] Creating persistent SSD disk ($DISK_SIZE)..."
if gcloud compute disks describe "$DISK_NAME" --zone "$ZONE" --project "$PROJECT" &>/dev/null; then
    echo "  Disk '$DISK_NAME' already exists — skipping."
else
    gcloud compute disks create "$DISK_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --size "$DISK_SIZE" \
        --type pd-ssd
fi

echo "[3/5] Creating VM ($MACHINE_TYPE)..."
if gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --project "$PROJECT" &>/dev/null; then
    echo "  VM '$VM_NAME' already exists — skipping."
else
    gcloud compute instances create "$VM_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --machine-type "$MACHINE_TYPE" \
        --image-family debian-12-arm64 \
        --image-project debian-cloud \
        --disk "name=$DISK_NAME,mode=rw,auto-delete=no" \
        --boot-disk-size 20GB \
        --tags nagual-server \
        --metadata startup-script='#!/bin/bash
            mkdir -p /data
            if ! mountpoint -q /data; then
                if ! blkid /dev/sdb | grep -q ext4; then
                    mkfs.ext4 -F /dev/sdb
                fi
                mount /dev/sdb /data
                grep -q "/dev/sdb" /etc/fstab || echo "/dev/sdb /data ext4 defaults 0 2" >> /etc/fstab
            fi'
fi

echo "[4/5] Waiting for VM to be reachable..."
sleep 10

echo "[5/5] Running remote provisioning..."
gcloud compute ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command '
set -euo pipefail

echo "--- Installing Docker + system packages ---"
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io docker-compose-plugin sqlite3 curl wget git build-essential pkg-config libssl-dev

sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

echo "--- Installing ONNX Runtime (arm64) ---"
if [ ! -f /usr/lib/libonnxruntime.so ]; then
    cd /tmp
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-aarch64-1.17.0.tgz
    tar xf onnxruntime-linux-aarch64-1.17.0.tgz
    sudo cp onnxruntime-linux-aarch64-1.17.0/lib/libonnxruntime.so* /usr/lib/
    sudo ldconfig
    rm -rf onnxruntime-linux-aarch64-1.17.0*
    echo "  ONNX Runtime installed."
fi

echo "--- Installing cloudflared (optional — for Cloudflare Tunnel) ---"
if ! command -v cloudflared &>/dev/null; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
    sudo dpkg -i cloudflared-linux-arm64.deb
    rm cloudflared-linux-arm64.deb
fi

echo "--- Installing Rust toolchain ---"
if ! command -v cargo &>/dev/null; then
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

echo "--- Creating nagual service user ---"
if ! id nagual &>/dev/null; then
    sudo useradd -r -s /bin/false -d /data nagual
fi

echo "--- VM setup complete ---"
'

echo ""
echo "=== Next Steps ==="
echo "  1. SSH in:          gcloud compute ssh $VM_NAME --zone $ZONE --project $PROJECT"
echo "  2. Clone the repo:  git clone https://github.com/proffesor-for-testing/nagual-qe /data/nagual-qe"
echo "  3. Build:           cd /data/nagual-qe && source ~/.cargo/env && cargo build --release --features serve"
echo "  4. Install binary:  sudo cp target/release/nagual /usr/local/bin/"
echo "  5. Start Postgres:  cd /data/nagual-qe && sudo docker compose up -d postgres"
echo "  6. Install systemd: sudo cp deploy/nagual.service /etc/systemd/system/"
echo "                      (edit paths/ports in the unit file first)"
echo "  7. Enable service:  sudo systemctl daemon-reload && sudo systemctl enable --now nagual"
echo ""
echo "  Optional: configure Cloudflare Tunnel — see deploy/cloudflared-config.example.yml"
echo ""
