//! `nagual activity search` — search across activity history.
//!
//! Supports three modes:
//! - Default full-text search via Screenpipe `/search`
//! - `--semantic` — embedding-based similarity via `/semantic-search`
//! - `--keyword` — keyword search with text positions via `/search/keyword`

use chrono::{Duration, Utc};

use crate::error::Result;

use super::client::{ScreenpipeClient, SearchParams};
use super::helpers::*;
use super::types::*;
use super::SearchArgs;

pub(super) async fn run_search(args: &SearchArgs) -> Result<()> {
    let sp_url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&sp_url);
    let sp_available = client.health().await.is_some();

    let end = Utc::now();
    let start = args
        .since
        .as_ref()
        .map(|s| end - parse_duration(s))
        .unwrap_or_else(|| end - Duration::days(7));

    let mut results: Vec<SearchResultItem> = Vec::new();

    if sp_available {
        if args.semantic {
            // Semantic (embedding-based) search
            let threshold = args.threshold.unwrap_or(0.3);
            match client
                .semantic_search(&args.query, args.limit, threshold, None)
                .await
            {
                Ok(resp) => {
                    for item in resp.data {
                        let text = if !item.content.text.is_empty() {
                            item.content.text
                        } else {
                            item.content.transcription.unwrap_or_default()
                        };
                        let score_str = item
                            .score
                            .map(|s| format!(" (score: {:.3})", s))
                            .unwrap_or_default();
                        results.push(SearchResultItem {
                            content_type: format!("{}{}", item.content_type, score_str),
                            app_name: item.content.app_name.unwrap_or_default(),
                            window_name: item.content.window_name.unwrap_or_default(),
                            text: truncate(&text, 200),
                            timestamp: item.content.timestamp.unwrap_or_default(),
                            browser_url: item.content.browser_url,
                            focused: item.content.focused,
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Semantic search failed");
                }
            }
        } else if args.keyword {
            // Keyword search with text positions
            match client
                .keyword_search(&args.query, args.limit, None, Some(&start), Some(&end))
                .await
            {
                Ok(resp) => {
                    for item in resp.data {
                        let text = if !item.content.text.is_empty() {
                            item.content.text
                        } else {
                            item.content.transcription.unwrap_or_default()
                        };
                        results.push(SearchResultItem {
                            content_type: item.content_type,
                            app_name: item.content.app_name.unwrap_or_default(),
                            window_name: item.content.window_name.unwrap_or_default(),
                            text: truncate(&text, 200),
                            timestamp: item.content.timestamp.unwrap_or_default(),
                            browser_url: item.content.browser_url,
                            focused: item.content.focused,
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Keyword search failed");
                }
            }
        } else {
            // Default full-text search
            let params = SearchParams {
                query: &args.query,
                content_type: "all",
                start: Some(&start),
                end: Some(&end),
                app_name: args.app_name.as_deref(),
                window_name: args.window_name.as_deref(),
                focused: if args.focused_only { Some(true) } else { None },
                min_length: args.min_length,
                limit: args.limit,
                offset: 0,
            };
            if let Ok(resp) = client.search(&params).await {
                for item in resp.data {
                    let text = if !item.content.text.is_empty() {
                        item.content.text
                    } else {
                        item.content.transcription.unwrap_or_default()
                    };
                    results.push(SearchResultItem {
                        content_type: item.content_type,
                        app_name: item.content.app_name.unwrap_or_default(),
                        window_name: item.content.window_name.unwrap_or_default(),
                        text: truncate(&text, 200),
                        timestamp: item.content.timestamp.unwrap_or_default(),
                        browser_url: item.content.browser_url,
                        focused: item.content.focused,
                    });
                }
            }
        }
    }

    // Also search stored activity patterns (for all modes)
    let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;
    let local_patterns = storage.get_recent(500).await?;
    let query_lower = args.query.to_lowercase();
    for p in local_patterns {
        if results.len() >= args.limit {
            break;
        }
        if !p.category().to_string().starts_with("activity.") {
            continue;
        }
        if p.timestamp() < start {
            continue;
        }
        if p.problem().to_lowercase().contains(&query_lower)
            || p.solution().to_lowercase().contains(&query_lower)
        {
            results.push(SearchResultItem {
                content_type: "nagual_pattern".to_string(),
                app_name: p
                    .metadata()
                    .extra
                    .get("app_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                window_name: String::new(),
                text: truncate(p.problem(), 200),
                timestamp: p.timestamp().to_rfc3339(),
                browser_url: None,
                focused: None,
            });
        }
    }

    results.truncate(args.limit);

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        let mode = if args.semantic {
            "semantic"
        } else if args.keyword {
            "keyword"
        } else {
            "full-text"
        };
        println!("\nActivity Search ({}): \"{}\"", mode, args.query);
        println!("{:=<70}", "");

        if results.is_empty() {
            println!("  No results found.");
            if !sp_available {
                println!("  (Screenpipe is not running — only searching stored patterns)");
            }
        } else {
            for (i, r) in results.iter().enumerate() {
                let focused_tag = match r.focused {
                    Some(true) => " *",
                    _ => "",
                };
                println!(
                    "\n  {}. [{}]{} {} — {}",
                    i + 1,
                    r.content_type,
                    focused_tag,
                    r.app_name,
                    truncate(&r.window_name, 50)
                );
                println!("     {}", r.text);
                if let Some(ref url) = r.browser_url {
                    if !url.is_empty() {
                        println!("     URL: {}", url);
                    }
                }
                if !r.timestamp.is_empty() {
                    println!("     @ {}", r.timestamp);
                }
            }
        }

        println!("\n{:=<70}", "");
        println!("  {} results found\n", results.len());
    }

    Ok(())
}
