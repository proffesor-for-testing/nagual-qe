//! `nagual activity ingest` — bulk ingest from Screenpipe into Nagual knowledge base.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Timelike, Utc};

use crate::error::Result;
use crate::reasoning_bank::pattern::{Pattern, PatternCategory, PatternMetadata};

use super::client::{ScreenpipeClient, SearchParams};
use super::helpers::*;
use super::ospipe::{EmbeddingDim, OSpipeConfig, OSpipePipeline};
use super::types::{
    DedupStatsOutput, IngestOutput, OSpipeIngestOutput, RawOcrRow,
};
use super::IngestArgs;

pub(super) async fn run_ingest(args: &IngestArgs) -> Result<()> {
    let sp_url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&sp_url);

    if client.health().await.is_none() {
        eprintln!("Error: Screenpipe is not reachable at {}", sp_url);
        eprintln!("Start Screenpipe first: screenpipe");
        return Ok(());
    }

    let duration = parse_duration(&args.since);
    let end = Utc::now();
    let start = end - duration;
    let start_str = start.format("%Y-%m-%dT%H:%M:%SZ").to_string();

    if !args.json {
        println!("\nIngesting Screenpipe activity");
        println!("{:=<60}", "");
        println!("  Time range:   {} → {}", start.format("%H:%M"), end.format("%H:%M"));
        println!("  Content:      {}", args.content_type);
        println!("  Min length:   {}", args.min_length);
        println!("  Focused only: {}", args.focused_only);
        if let Some(ref app) = args.app_name {
            println!("  App filter:   {}", app);
        }
        println!(
            "  Embeddings:   {}",
            if args.embed { "yes (768-dim via Screenpipe)" } else { "no" }
        );
        println!("{:-<60}", "");
    }

    // ---- Try bulk ingest via /raw_sql (faster than paginated REST) ----
    let use_raw_sql = args.content_type == "all" || args.content_type == "ocr";

    let rows: Vec<RawOcrRow> = if use_raw_sql {
        let mut sql = format!(
            "SELECT f.id as frame_id, f.timestamp, f.app_name, \
             COALESCE(f.window_name, '') as window_name, \
             o.text, f.focused, f.browser_url \
             FROM frames f \
             JOIN ocr_text o ON f.id = o.frame_id \
             WHERE f.timestamp >= '{}' \
             AND length(o.text) >= {}",
            start_str, args.min_length,
        );
        if args.focused_only {
            sql.push_str(" AND f.focused = 1");
        }
        if let Some(ref app) = args.app_name {
            sql.push_str(&format!(" AND f.app_name = '{}'", app.replace('\'', "''")));
        }
        sql.push_str(" ORDER BY f.timestamp DESC LIMIT 2000");

        match client.raw_sql(&sql).await {
            Ok(raw_rows) => raw_rows
                .into_iter()
                .filter_map(|v| serde_json::from_value(v).ok())
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "raw_sql failed, falling back to search API");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    let source;

    // Also pull audio transcriptions if requested
    let audio_rows: Vec<RawOcrRow> =
        if args.content_type == "all" || args.content_type == "audio" {
            let sql = format!(
                "SELECT a.id as frame_id, a.timestamp, a.device as app_name, \
                 '' as window_name, a.transcription as text, \
                 NULL as focused, NULL as browser_url \
                 FROM audio_transcriptions a \
                 WHERE a.timestamp >= '{}' \
                 AND length(a.transcription) >= {} \
                 ORDER BY a.timestamp DESC LIMIT 500",
                start_str, args.min_length,
            );
            client
                .raw_sql(&sql)
                .await
                .ok()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|v| serde_json::from_value(v).ok())
                .collect()
        } else {
            Vec::new()
        };

    // Also pull UI events if content_type is "all" or "input"
    let ui_event_rows: Vec<RawOcrRow> =
        if args.content_type == "all" || args.content_type == "input" {
            let sql = format!(
                "SELECT m.id as frame_id, m.timestamp, \
                 COALESCE(m.app, 'unknown') as app_name, \
                 COALESCE(m.window, '') as window_name, \
                 COALESCE(m.text_output, m.type) as text, \
                 NULL as focused, NULL as browser_url \
                 FROM ui_monitoring m \
                 WHERE m.timestamp >= '{}' \
                 AND length(COALESCE(m.text_output, m.type)) >= 5 \
                 ORDER BY m.timestamp DESC LIMIT 500",
                start_str,
            );
            client
                .raw_sql(&sql)
                .await
                .ok()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|v| serde_json::from_value(v).ok())
                .collect()
        } else {
            Vec::new()
        };

    let all_rows_count;

    // Choose data source
    let all_rows: Vec<RawOcrRow> =
        if !rows.is_empty() || !audio_rows.is_empty() || !ui_event_rows.is_empty() {
            source = "raw_sql";
            let mut combined = rows;
            combined.extend(audio_rows);
            combined.extend(ui_event_rows);
            all_rows_count = combined.len();
            combined
        } else {
            // Fallback to paginated search API
            source = "search_api";
            let mut items: Vec<RawOcrRow> = Vec::new();
            let mut offset = 0;
            loop {
                let params = SearchParams {
                    query: "",
                    content_type: &args.content_type,
                    start: Some(&start),
                    end: Some(&end),
                    app_name: args.app_name.as_deref(),
                    focused: if args.focused_only { Some(true) } else { None },
                    min_length: Some(args.min_length),
                    limit: 100,
                    offset,
                    ..Default::default()
                };
                match client.search(&params).await {
                    Ok(resp) => {
                        let count = resp.data.len();
                        for item in resp.data {
                            let text = if !item.content.text.is_empty() {
                                item.content.text
                            } else {
                                item.content.transcription.unwrap_or_default()
                            };
                            items.push(RawOcrRow {
                                frame_id: 0,
                                timestamp: item.content.timestamp.unwrap_or_default(),
                                app_name: item.content.app_name.unwrap_or_default(),
                                window_name: item.content.window_name.unwrap_or_default(),
                                text,
                                focused: item.content.focused,
                                browser_url: item.content.browser_url,
                            });
                        }
                        if count < 100 {
                            break;
                        }
                        offset += 100;
                        if offset > 1000 {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            all_rows_count = items.len();
            items
        };

    if all_rows.is_empty() {
        if args.json {
            let out = IngestOutput {
                ingested: 0,
                skipped: 0,
                embedded: 0,
                time_range_start: start.to_rfc3339(),
                time_range_end: end.to_rfc3339(),
                source: source.to_string(),
                patterns_created: vec![],
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        } else {
            println!("  No activity found in the given time range.");
        }
        return Ok(());
    }

    // ========== OSpipe Pipeline Branch ==========
    if args.ospipe {
        return run_ospipe_ingest(
            args,
            &all_rows,
            all_rows_count,
            source,
            start,
            end,
        )
        .await;
    }

    // ========== Legacy Ingest Path ==========
    // Group by app + 5-minute buckets
    let mut buckets: HashMap<String, Vec<&RawOcrRow>> = HashMap::new();
    for row in &all_rows {
        let bucket_ts = if let Ok(dt) = DateTime::parse_from_rfc3339(&row.timestamp) {
            let minutes = dt.minute() / 5 * 5;
            format!("{}-{:02}", dt.format("%Y-%m-%d %H"), minutes)
        } else if row.timestamp.len() >= 16 {
            row.timestamp[..16].to_string()
        } else {
            "unknown-time".to_string()
        };
        let key = format!("{}|{}", row.app_name, bucket_ts);
        buckets.entry(key).or_default().push(row);
    }

    let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;
    let mut ingested = 0usize;
    let mut embedded = 0usize;
    let mut pattern_ids: Vec<String> = Vec::new();

    for (key, items) in &buckets {
        let parts: Vec<&str> = key.splitn(2, '|').collect();
        let app_name = parts.first().copied().unwrap_or("unknown");
        let domain = categorize_app(app_name);

        let combined_text: String = items
            .iter()
            .map(|r| r.text.as_str())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" | ");
        let combined_text = truncate(&combined_text, 2000);

        if combined_text.trim().is_empty() {
            continue;
        }

        let windows: Vec<String> = items
            .iter()
            .map(|r| r.window_name.clone())
            .filter(|w| !w.is_empty())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(5)
            .collect();

        let urls: Vec<String> = items
            .iter()
            .filter_map(|r| r.browser_url.clone())
            .filter(|u| !u.is_empty())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(3)
            .collect();

        let focused_count = items.iter().filter(|r| r.focused == Some(true)).count();

        let problem = format!(
            "Used {}: {}",
            app_name,
            if windows.is_empty() {
                format!("{} items captured", items.len())
            } else {
                windows.join(", ")
            }
        );

        let mut tags = vec![
            app_name.to_string(),
            domain.to_string(),
            "screenpipe".to_string(),
        ];
        if focused_count > 0 {
            tags.push("focused".to_string());
        }

        let mut metadata = PatternMetadata::new()
            .with_source("screenpipe")
            .with_extra("app_name", serde_json::json!(app_name))
            .with_extra("item_count", serde_json::json!(items.len()))
            .with_extra("focused_count", serde_json::json!(focused_count))
            .with_extra("windows", serde_json::json!(windows));
        if !urls.is_empty() {
            metadata = metadata.with_extra("browser_urls", serde_json::json!(urls));
        }

        let mut builder = Pattern::builder()
            .problem(&problem)
            .solution(&combined_text)
            .category(PatternCategory::Custom(format!("activity.{}", domain)))
            .context(format!("Screenpipe activity capture from {}", app_name))
            .effectiveness(0.5)
            .confidence(0.5)
            .tags(tags)
            .metadata(metadata);

        if args.embed {
            let embed_text = truncate(&format!("{} {}", problem, combined_text), 500);
            if let Some(embedding) = client.embed(&embed_text).await {
                builder = builder.embedding(embedding);
                embedded += 1;
            }
        }

        let pattern = builder.build();
        let id = pattern.id().to_string();

        if let Err(e) = storage.store_pattern(&pattern).await {
            tracing::warn!(error = %e, pattern_id = %id, "Failed to store activity pattern");
            continue;
        }

        pattern_ids.push(id);
        ingested += 1;
    }

    if resolve_postgres_url(args.postgres_url.as_deref()).is_some() {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    let skipped = buckets.len() - ingested;

    if args.json {
        let out = IngestOutput {
            ingested,
            skipped,
            embedded,
            time_range_start: start.to_rfc3339(),
            time_range_end: end.to_rfc3339(),
            source: source.to_string(),
            patterns_created: pattern_ids,
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("\n  Source:              {}", source);
        println!("  Screenpipe rows:     {}", all_rows_count);
        println!("  Activity buckets:    {}", buckets.len());
        println!("  Patterns created:    {}", ingested);
        if embedded > 0 {
            println!("  Embeddings (768d):   {}", embedded);
        }
        if skipped > 0 {
            println!("  Skipped (empty):     {}", skipped);
        }
        println!("  Database:            {}", args.db_path.display());
        println!("{:=<60}\n", "");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// OSpipe Pipeline Ingest
// ---------------------------------------------------------------------------

/// Build OSpipeConfig from CLI arguments.
fn build_ospipe_config(args: &IngestArgs) -> OSpipeConfig {
    let embedding_dim = EmbeddingDim::from_str(&args.embedding_dim)
        .unwrap_or(EmbeddingDim::Dim384);

    let dedup_window = parse_dedup_window(&args.dedup_window);

    OSpipeConfig::default()
        .with_pii_policy(&args.pii_policy)
        .with_dedup_threshold(args.dedup_threshold)
        .with_dedup_window(dedup_window)
        .with_embedding_dim(embedding_dim)
        .with_generate_embeddings(args.embed)
}

/// Parse dedup window string (e.g., "5m", "10m") into Duration.
fn parse_dedup_window(s: &str) -> Duration {
    let s = s.trim().to_lowercase();
    let (num_str, unit) = s.split_at(s.len().saturating_sub(1));
    let num: u64 = num_str.parse().unwrap_or(5);
    match unit {
        "s" => Duration::from_secs(num),
        "m" => Duration::from_secs(num * 60),
        "h" => Duration::from_secs(num * 3600),
        _ => Duration::from_secs(5 * 60), // default 5 minutes
    }
}

/// Run ingestion through the OSpipe pipeline with PII protection and deduplication.
async fn run_ospipe_ingest(
    args: &IngestArgs,
    all_rows: &[RawOcrRow],
    all_rows_count: usize,
    source: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<()> {
    use super::ospipe::IngestResult;

    let config = build_ospipe_config(args);
    let storage = Arc::new(init_storage(&args.db_path, args.postgres_url.as_deref()).await?);

    if !args.json {
        println!("\n  OSpipe Pipeline Mode");
        println!("  PII policy:       {}", config.pii_policy);
        println!("  Dedup threshold:  {:.2}", config.dedup_threshold);
        println!("  Dedup window:     {}s", config.dedup_window.as_secs());
        println!("  Embedding dim:    {}", config.embedding_dim);
        println!("{:-<60}", "");
    }

    let mut pipeline = OSpipePipeline::new(Arc::clone(&storage), config);

    // Initialize embedder if embeddings are requested
    if args.embed {
        if let Err(e) = pipeline.init_embedder() {
            if !args.json {
                println!(
                    "  Warning: Could not initialize embedder: {}",
                    e
                );
                println!("  Proceeding without local embeddings.");
            }
        }
    }

    let mut result = IngestResult::new();
    result.time_range_start = Some(start);
    result.time_range_end = Some(end);

    // Process each row through the pipeline
    for row in all_rows {
        if row.text.trim().is_empty() {
            continue;
        }

        let timestamp = DateTime::parse_from_rfc3339(&row.timestamp)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        // Build metadata for this item
        let metadata = PatternMetadata::new()
            .with_source("ospipe")
            .with_extra("app_name", serde_json::json!(&row.app_name))
            .with_extra("window_name", serde_json::json!(&row.window_name))
            .with_extra("focused", serde_json::json!(row.focused));

        let item_result = pipeline
            .process_item(&row.text, &row.app_name, timestamp, Some(metadata))
            .await;

        result.add(&item_result);
    }

    // Sync PostgreSQL if configured
    if resolve_postgres_url(args.postgres_url.as_deref()).is_some() {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    // Get dedup statistics
    let dedup_stats = pipeline.dedup_stats();

    if args.json {
        let out = OSpipeIngestOutput {
            ingested: result.stored_count,
            rejected: result.rejected_count,
            redacted: result.redacted_count,
            duplicates: result.duplicate_count,
            errors: result.error_count,
            embedded: result.embeddings_generated,
            time_range_start: start.to_rfc3339(),
            time_range_end: end.to_rfc3339(),
            source: source.to_string(),
            patterns_created: result.pattern_ids,
            dedup_stats: DedupStatsOutput {
                total_checked: dedup_stats.total_checked as usize,
                duplicates_found: dedup_stats.duplicates_found as usize,
                unique_items: dedup_stats.unique_items as usize,
            },
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("\n  Source:              {}", source);
        println!("  Screenpipe rows:     {}", all_rows_count);
        println!("  Patterns stored:     {}", result.stored_count);
        if result.rejected_count > 0 {
            println!("  Rejected (PII):      {}", result.rejected_count);
        }
        if result.redacted_count > 0 {
            println!("  Redacted (PII):      {}", result.redacted_count);
        }
        if result.duplicate_count > 0 {
            println!("  Deduplicated:        {}", result.duplicate_count);
        }
        if result.error_count > 0 {
            println!("  Errors:              {}", result.error_count);
        }
        if result.embeddings_generated > 0 {
            println!("  Embeddings:          {}", result.embeddings_generated);
        }
        println!("  Database:            {}", args.db_path.display());
        println!("{:=<60}\n", "");
    }

    Ok(())
}
