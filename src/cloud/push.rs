//! Push logic: send local patterns to the cloud.

use chrono::{DateTime, Utc};
use tracing::{debug, info};

use crate::db::SqliteDb;
use crate::error::Result;
use crate::sync::pii::global_redactor;

use super::client::CloudClient;
use super::sync_state;
use super::types::SyncPatternData;

/// Batch size for push requests.
const PUSH_BATCH_SIZE: usize = 100;

/// Push local patterns to a remote nagual server.
///
/// Only pushes patterns modified since the last successful push
/// (unless `full` is true).
pub async fn cloud_push(
    db: &SqliteDb,
    remote_url: &str,
    api_token: &str,
    full: bool,
) -> Result<PushSummary> {
    // Ensure sync state table exists
    sync_state::init_sync_state_table(db).await?;

    // Get last push timestamp
    let since = if full {
        None
    } else {
        sync_state::get_sync_state(db, remote_url)
            .await?
            .and_then(|s| s.last_push_at)
    };

    // Query patterns modified since last push
    let patterns = query_patterns_since(db, since).await?;

    // Strip PII from all text fields before pushing to remote server.
    // Local SQLite data is NEVER modified.
    //
    // Note: SyncPatternData intentionally omits `title` and `summary` fields
    // (see cloud/types.rs), so they do not need redaction here. Only the
    // free-text fields that are present — problem, solution, context, and
    // critique — are scrubbed.
    let patterns: Vec<SyncPatternData> = patterns
        .into_iter()
        .map(|mut p| {
            let redactor = global_redactor();
            p.problem = redactor.strip_pii(&p.problem).text;
            p.solution = redactor.strip_pii(&p.solution).text;
            p.context = redactor.strip_pii(&p.context).text;
            if let Some(ref critique) = p.critique {
                p.critique = Some(redactor.strip_pii(critique).text);
            }
            p
        })
        .collect();

    if patterns.is_empty() {
        return Ok(PushSummary {
            total: 0,
            created: 0,
            updated: 0,
            skipped: 0,
        });
    }

    let client = CloudClient::new(remote_url, api_token);
    let total = patterns.len();
    let mut created = 0usize;
    let mut updated = 0usize;
    let mut skipped = 0usize;

    // Push in batches
    for chunk in patterns.chunks(PUSH_BATCH_SIZE) {
        let response = client.push_patterns(chunk).await?;
        created += response.created;
        updated += response.updated;
        skipped += response.skipped;
        debug!(
            batch_size = chunk.len(),
            created = response.created,
            updated = response.updated,
            skipped = response.skipped,
            "Push batch completed"
        );
    }

    // Update sync state with max updated_at from pushed patterns
    if let Some(max_ts) = patterns.iter().filter_map(|p| {
        DateTime::parse_from_rfc3339(&p.updated_at)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    }).max() {
        sync_state::update_push_state(db, remote_url, max_ts, total as i64).await?;
    }

    info!(
        total = total,
        created = created,
        updated = updated,
        skipped = skipped,
        "Cloud push completed"
    );

    Ok(PushSummary {
        total,
        created,
        updated,
        skipped,
    })
}

/// Query patterns modified since a timestamp from local SQLite.
async fn query_patterns_since(
    db: &SqliteDb,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<SyncPatternData>> {
    let since_str = since.map(|dt| dt.to_rfc3339());

    let patterns = db.with_connection(move |conn| {
        let (sql, params): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(ref ts) = since_str {
            (
                "SELECT id, problem, solution, context, category, tags, reward, confidence, \
                 success, reuse_count, effectiveness, timestamp, updated_at, agent_id, \
                 session_id, content_hash, critique \
                 FROM reasoning_patterns WHERE updated_at > ? ORDER BY updated_at ASC",
                vec![Box::new(ts.clone()) as Box<dyn rusqlite::types::ToSql>],
            )
        } else {
            (
                "SELECT id, problem, solution, context, category, tags, reward, confidence, \
                 success, reuse_count, effectiveness, timestamp, updated_at, agent_id, \
                 session_id, content_hash, critique \
                 FROM reasoning_patterns ORDER BY updated_at ASC",
                vec![],
            )
        };

        let mut stmt = conn.prepare(sql).map_err(crate::error::DatabaseError::from)?;

        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                let id: String = row.get(0)?;
                let problem: String = row.get(1)?;
                let solution: String = row.get(2)?;
                let context: String = row.get::<_, Option<String>>(3)?.unwrap_or_default();
                let domain: String = row.get(4)?;
                let tags_json: String = row.get::<_, Option<String>>(5)?.unwrap_or_else(|| "[]".to_string());
                let reward: f64 = row.get(6)?;
                let confidence: f64 = row.get(7)?;
                let success: bool = row.get::<_, i32>(8)? != 0;
                let reuse_count: i32 = row.get(9)?;
                let effectiveness: f64 = row.get(10)?;
                let created_at: String = row.get(11)?;
                let updated_at: String = row.get(12)?;
                let agent_id: Option<String> = row.get(13)?;
                let session_id: Option<String> = row.get(14)?;
                let content_hash: Option<String> = row.get(15)?;
                let critique: Option<String> = row.get(16)?;

                let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

                Ok(SyncPatternData {
                    id,
                    problem,
                    solution,
                    context,
                    domain,
                    tags,
                    reward: reward as f32,
                    confidence: confidence as f32,
                    success,
                    reuse_count: reuse_count as u32,
                    effectiveness: effectiveness as f32,
                    created_at,
                    updated_at,
                    agent_id,
                    session_id,
                    content_hash,
                    critique: critique.filter(|c| !c.is_empty()),
                })
            })
            .map_err(crate::error::DatabaseError::from)?;

        let mut patterns = Vec::new();
        for row in rows {
            patterns.push(row.map_err(crate::error::DatabaseError::from)?);
        }
        Ok(patterns)
    }).await?;

    Ok(patterns)
}

/// Summary of a push operation.
#[derive(Debug)]
pub struct PushSummary {
    pub total: usize,
    pub created: usize,
    pub updated: usize,
    pub skipped: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_summary_default() {
        let summary = PushSummary {
            total: 0,
            created: 0,
            updated: 0,
            skipped: 0,
        };
        assert_eq!(summary.total, 0);
    }
}
