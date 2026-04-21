//! Pull logic: fetch patterns from the cloud into local SQLite.

use chrono::{DateTime, Utc};
use tracing::{debug, info};

use crate::db::SqliteDb;
use crate::error::Result;
use crate::reasoning_bank::storage::PatternStorage;

use super::client::CloudClient;
use super::sync_state;

/// Page size for pull requests.
const PULL_PAGE_SIZE: usize = 100;

/// Pull patterns from a remote nagual server into local storage.
///
/// Only pulls patterns modified since the last successful pull
/// (unless `full` is true).
pub async fn cloud_pull(
    storage: &PatternStorage,
    db: &SqliteDb,
    remote_url: &str,
    api_token: &str,
    full: bool,
) -> Result<PullSummary> {
    // Ensure sync state table exists
    sync_state::init_sync_state_table(db).await?;

    // Get last pull timestamp
    let since = if full {
        None
    } else {
        sync_state::get_sync_state(db, remote_url)
            .await?
            .and_then(|s| s.last_pull_at)
    };

    let client = CloudClient::new(remote_url, api_token);
    let mut total = 0usize;
    let mut offset = 0usize;
    let mut max_updated_at: Option<DateTime<Utc>> = None;

    loop {
        let response = client
            .pull_patterns(since, PULL_PAGE_SIZE, offset)
            .await?;

        let page_count = response.patterns.len();
        if page_count == 0 {
            break;
        }

        // Upsert each pattern into local storage
        for sync_data in &response.patterns {
            let pattern = sync_data.to_pattern();
            storage.store_pattern(&pattern).await?;

            // Track max updated_at
            if let Ok(dt) = DateTime::parse_from_rfc3339(&sync_data.updated_at) {
                let dt = dt.with_timezone(&Utc);
                max_updated_at = Some(match max_updated_at {
                    Some(current) if dt > current => dt,
                    Some(current) => current,
                    None => dt,
                });
            }
        }

        total += page_count;
        debug!(
            page_count = page_count,
            total = total,
            has_more = response.has_more,
            "Pull page completed"
        );

        if !response.has_more {
            break;
        }

        offset += page_count;
    }

    // Update sync state
    if let Some(max_ts) = max_updated_at {
        sync_state::update_pull_state(db, remote_url, max_ts, total as i64).await?;
    }

    info!(total = total, "Cloud pull completed");

    Ok(PullSummary { total })
}

/// Summary of a pull operation.
#[derive(Debug)]
pub struct PullSummary {
    pub total: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pull_summary() {
        let summary = PullSummary { total: 42 };
        assert_eq!(summary.total, 42);
    }
}
