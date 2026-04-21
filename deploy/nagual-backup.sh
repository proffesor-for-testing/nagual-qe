#!/usr/bin/env bash
# =================================================================
# Nagual-QE automated backup script (optional).
# Snapshots SQLite + PostgreSQL and uploads to a GCS bucket.
#
# Install as a cron job on the deployment host:
#   sudo cp deploy/nagual-backup.sh /usr/local/bin/nagual-backup
#   sudo chmod +x /usr/local/bin/nagual-backup
#   echo "0 */6 * * * nagual /usr/local/bin/nagual-backup 2>&1 | logger -t nagual-backup" \
#     | sudo tee /etc/cron.d/nagual-backup
#
# Customize via env (or /etc/nagual/backup.env, sourced if present):
#   NAGUAL_DB_PATH         (default: /data/nagual-qe/nagual.db)
#   NAGUAL_BACKUP_BUCKET   (required: gs://your-bucket/backups)
#   NAGUAL_PG_CONTAINER    (default: nagual-ruvector)
#   NAGUAL_PG_USER         (default: nagual)
#   NAGUAL_PG_DB           (default: nagual)
#   NAGUAL_LOCAL_BACKUP    (default: /data/backups)
#   NAGUAL_RETENTION_DAYS  (default: 7)
# =================================================================
set -euo pipefail

[ -f /etc/nagual/backup.env ] && . /etc/nagual/backup.env

DB_PATH="${NAGUAL_DB_PATH:-/data/nagual-qe/nagual.db}"
BUCKET="${NAGUAL_BACKUP_BUCKET:?set NAGUAL_BACKUP_BUCKET=gs://your-bucket/backups}"
PG_CONTAINER="${NAGUAL_PG_CONTAINER:-nagual-ruvector}"
PG_USER="${NAGUAL_PG_USER:-nagual}"
PG_DB="${NAGUAL_PG_DB:-nagual}"
LOCAL_BACKUP_DIR="${NAGUAL_LOCAL_BACKUP:-/data/backups}"
RETENTION_DAYS="${NAGUAL_RETENTION_DAYS:-7}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOCAL_BACKUP_DIR"

# ── SQLite backup ──
echo "[backup] Starting SQLite backup..."
SQLITE_BACKUP="$LOCAL_BACKUP_DIR/nagual-$TIMESTAMP.db.gz"
sqlite3 "$DB_PATH" ".backup '$LOCAL_BACKUP_DIR/nagual-$TIMESTAMP.db'"
gzip "$LOCAL_BACKUP_DIR/nagual-$TIMESTAMP.db"
gsutil -q cp "$SQLITE_BACKUP" "$BUCKET/sqlite/nagual-$TIMESTAMP.db.gz"
echo "[backup] SQLite backup uploaded: $BUCKET/sqlite/nagual-$TIMESTAMP.db.gz"

# ── PostgreSQL backup (only if container is running) ──
if docker ps --format '{{.Names}}' | grep -q "^${PG_CONTAINER}$"; then
    echo "[backup] Starting PostgreSQL backup..."
    PG_BACKUP="$LOCAL_BACKUP_DIR/postgres-$TIMESTAMP.sql.gz"
    docker exec "$PG_CONTAINER" pg_dump -U "$PG_USER" "$PG_DB" | gzip > "$PG_BACKUP"
    gsutil -q cp "$PG_BACKUP" "$BUCKET/postgres/postgres-$TIMESTAMP.sql.gz"
    echo "[backup] PostgreSQL backup uploaded: $BUCKET/postgres/postgres-$TIMESTAMP.sql.gz"
else
    echo "[backup] Postgres container '$PG_CONTAINER' not running — skipped PG backup."
fi

# ── Local rotation ──
find "$LOCAL_BACKUP_DIR" -name "*.gz" -mtime +"$RETENTION_DAYS" -delete

echo "[backup] Backup complete at $TIMESTAMP"
