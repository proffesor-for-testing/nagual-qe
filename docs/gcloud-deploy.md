# Deploying Nagual-QE on Google Cloud

This is the setup we use for single-host deployments on a GCE VM. It uses
Cloudflare Tunnel for ingress (no public IP, no inbound firewall rules) and
GCS for backups. Adapt freely — nothing here is cloud-specific except the
`setup-vm.sh` script which assumes `gcloud`.

## What you get

- 1× GCE VM running `nagual serve` + PostgreSQL (Docker) + `cloudflared`
- HTTPS dashboard at `nagual.YOURDOMAIN.com` (routed through Cloudflare)
- Scheduled SQLite + PostgreSQL backups to a GCS bucket
- Locked-down firewall (outbound only — no inbound rules needed)

Expected cost on `t2a-standard-1` (arm64, 1 vCPU, 4GB): ~$10–15/month with
no sustained-use discount, plus a few cents for GCS storage.

## 1. Provision the VM

```bash
export NAGUAL_GCP_PROJECT=your-project-id
export NAGUAL_GCP_ZONE=us-central1-a      # optional
export NAGUAL_VM_NAME=nagual-vm            # optional

bash deploy/setup-vm.sh
```

This creates:
- A 50GB persistent SSD attached as `/data`
- The VM with Docker, ONNX Runtime, cloudflared, Rust pre-installed
- A `nagual` system user for the service

## 2. Clone, build, install

```bash
gcloud compute ssh $NAGUAL_VM_NAME --zone $NAGUAL_GCP_ZONE --project $NAGUAL_GCP_PROJECT

# On the VM:
cd /data
git clone https://github.com/proffesor-for-testing/nagual-qe
cd nagual-qe
source ~/.cargo/env
cargo build --release --features serve
sudo cp target/release/nagual /usr/local/bin/
```

## 3. Start PostgreSQL

```bash
# On the VM
cd /data/nagual-qe
cp .env.example .env
# Edit .env — set a strong POSTGRES_PASSWORD

sudo docker compose up -d postgres

# Verify
sudo docker ps
sudo docker compose logs postgres | tail
```

## 4. Configure secrets

```bash
# On the VM
sudo mkdir -p /etc/nagual
sudo tee /etc/nagual/env >/dev/null <<EOF
NAGUAL_API_TOKEN=$(openssl rand -hex 32)
NAGUAL_SESSION_SECRET=$(openssl rand -hex 32)
DATABASE_URL=postgres://nagual:<your-password>@localhost:5432/nagual
EOF
sudo chmod 600 /etc/nagual/env
```

## 5. Install the systemd service

```bash
# On the VM
sudo cp /data/nagual-qe/deploy/nagual.service /etc/systemd/system/
# Edit paths if they differ from /data/nagual-qe:
sudo $EDITOR /etc/systemd/system/nagual.service

sudo systemctl daemon-reload
sudo systemctl enable --now nagual

# Verify
sudo systemctl status nagual
sudo journalctl -u nagual -f
```

The service listens on `localhost:3333` by default. Nothing is exposed to
the internet yet — the next step adds Cloudflare Tunnel.

## 6. (Optional) Cloudflare Tunnel

Prerequisites: a Cloudflare account + a domain you own on Cloudflare DNS.

```bash
# On the VM — one-time browser login
cloudflared tunnel login

# Create the tunnel (records TUNNEL_ID)
cloudflared tunnel create nagual
# Prints: Created tunnel nagual with id XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX

# Copy the credentials file to a system location
sudo mkdir -p /etc/cloudflared
sudo cp ~/.cloudflared/<TUNNEL_ID>.json /etc/cloudflared/credentials.json
sudo chmod 600 /etc/cloudflared/credentials.json

# Install the ingress config
sudo cp /data/nagual-qe/deploy/cloudflared-config.example.yml \
       /etc/cloudflared/config.yml
sudo $EDITOR /etc/cloudflared/config.yml
#   - replace REPLACE_WITH_YOUR_TUNNEL_ID with the UUID from `cloudflared tunnel create`
#   - replace nagual.YOURDOMAIN.com with your actual hostname

# Add the DNS route
cloudflared tunnel route dns nagual nagual.YOURDOMAIN.com

# Install + start
sudo cloudflared service install
sudo systemctl enable --now cloudflared

# Verify
curl -I https://nagual.YOURDOMAIN.com
```

You should get a 401 (auth required) — that's expected. Log in to the
dashboard via the username/password you create next.

## 7. Create a dashboard user

```bash
# On the VM
sudo -u nagual nagual user create admin --db-path /data/nagual-qe/nagual.db
# Prompts for password; stores bcrypt hash in the DB
```

## 8. (Optional) Scheduled backups

```bash
# On the VM
sudo cp /data/nagual-qe/deploy/nagual-backup.sh /usr/local/bin/nagual-backup
sudo chmod +x /usr/local/bin/nagual-backup

# Create the GCS bucket
BUCKET=your-project-nagual-backups
gsutil mb -l us-central1 gs://$BUCKET
gsutil lifecycle set <(cat <<'JSON'
{"rule":[{"action":{"type":"Delete"},"condition":{"age":90,"isLive":false}}]}
JSON
) gs://$BUCKET

# Tell the backup script where to go
sudo tee /etc/nagual/backup.env >/dev/null <<EOF
NAGUAL_BACKUP_BUCKET=gs://$BUCKET/backups
EOF

# Install the cron job (runs every 6 hours)
echo "0 */6 * * * nagual /usr/local/bin/nagual-backup 2>&1 | logger -t nagual-backup" \
  | sudo tee /etc/cron.d/nagual-backup

# Test it manually
sudo -u nagual /usr/local/bin/nagual-backup
```

## 9. Monitoring

- `sudo journalctl -u nagual -f` — live logs
- `nagual health --detailed` — component-level health from the CLI
- Dashboard at `https://nagual.YOURDOMAIN.com/#/health` — live metrics

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nagual` service won't start | Missing `/etc/nagual/env` | `sudo systemctl status nagual` and check journal |
| Dashboard returns 502 | `cloudflared` not running or wrong local port | `sudo systemctl status cloudflared` |
| `cargo build` hangs on VM | Out of memory on `t2a-standard-1` | Build locally and `scp` the binary, OR upgrade to `t2a-standard-2` for the build only |
| Docker PG container unhealthy | Arm64/amd64 mismatch on `ruvector-postgres` | Build the image from source for your arch (see `database-setup.md`) |

## Security checklist

- [ ] `/etc/nagual/env` mode 0600, owned by root
- [ ] `/etc/cloudflared/credentials.json` mode 0600
- [ ] `POSTGRES_PASSWORD` in `.env` is strong (32+ chars random)
- [ ] `NAGUAL_API_TOKEN` rotated at least annually
- [ ] Firewall allows only outbound traffic (cloudflared punches out)
- [ ] VM instance tag `nagual-server` scoped in GCP firewall rules
- [ ] Backups tested — restore at least once
