//! `nagual activity summary` — activity summary by time period.

use std::collections::HashMap;

use chrono::Utc;

use crate::error::Result;

use super::client::ScreenpipeClient;
use super::helpers::*;
use super::types::*;
use super::SummaryArgs;

pub(super) async fn run_summary(args: &SummaryArgs) -> Result<()> {
    let sp_url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&sp_url);

    let duration = parse_duration(&args.period);
    let end = Utc::now();
    let start = end - duration;
    let start_str = start.format("%Y-%m-%dT%H:%M:%SZ").to_string();

    let sp_available = client.health().await.is_some();
    let mut app_counts: HashMap<String, usize> = HashMap::new();
    let mut domain_counts: HashMap<String, usize> = HashMap::new();
    let mut total_items = 0usize;
    let mut audio_count = 0usize;

    if sp_available {
        let sql = format!(
            "SELECT app_name, COUNT(*) as cnt \
             FROM frames WHERE timestamp >= '{}' \
             GROUP BY app_name ORDER BY cnt DESC",
            start_str,
        );
        if let Ok(rows) = client.raw_sql(&sql).await {
            for r in &rows {
                let app = r["app_name"].as_str().unwrap_or("unknown").to_string();
                let cnt = r["cnt"].as_u64().unwrap_or(0) as usize;
                let domain = categorize_app(&app);
                *app_counts.entry(app).or_default() += cnt;
                *domain_counts.entry(domain.to_string()).or_default() += cnt;
                total_items += cnt;
            }
        }

        let audio_sql = format!(
            "SELECT COUNT(*) as cnt FROM audio_transcriptions WHERE timestamp >= '{}'",
            start_str,
        );
        if let Ok(rows) = client.raw_sql(&audio_sql).await {
            if let Some(r) = rows.first() {
                audio_count = r["cnt"].as_u64().unwrap_or(0) as usize;
                if audio_count > 0 {
                    *domain_counts.entry("audio".to_string()).or_default() += audio_count;
                    total_items += audio_count;
                }
            }
        }
    }

    let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;
    let local_patterns = storage.get_recent(500).await?;
    let activity_patterns: Vec<_> = local_patterns
        .iter()
        .filter(|p| p.category().to_string().starts_with("activity."))
        .filter(|p| p.timestamp() >= start)
        .filter(|p| {
            if args.domain == "all" {
                true
            } else {
                p.category().to_string().contains(&args.domain)
            }
        })
        .collect();

    let mut sorted_apps: Vec<AppUsageItem> = app_counts
        .into_iter()
        .map(|(app, count)| AppUsageItem {
            domain: categorize_app(&app).to_string(),
            app_name: app,
            count,
        })
        .collect();
    sorted_apps.sort_by(|a, b| b.count.cmp(&a.count));

    if args.json {
        let out = SummaryOutput {
            period: args.period.clone(),
            total_items,
            apps: sorted_apps,
            domains: domain_counts,
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        println!("\nActivity Summary ({})", args.period);
        println!("{:=<60}", "");

        if sp_available {
            println!("  Screenpipe items:   {}", total_items);
            if audio_count > 0 {
                println!("  Audio segments:     {}", audio_count);
            }
        } else {
            println!("  Screenpipe:         not running");
        }
        println!("  Nagual patterns:    {}", activity_patterns.len());

        if !sorted_apps.is_empty() {
            println!("\n  Top Apps:");
            for (i, app) in sorted_apps.iter().take(10).enumerate() {
                println!(
                    "    {}. {:<25} {:>5} items  [{}]",
                    i + 1,
                    app.app_name,
                    app.count,
                    app.domain
                );
            }
        }

        if !domain_counts.is_empty() {
            println!("\n  Domains:");
            let mut sorted_domains: Vec<_> = domain_counts.iter().collect();
            sorted_domains.sort_by(|a, b| b.1.cmp(a.1));
            for (domain, count) in sorted_domains {
                println!("    {:<20} {:>5} items", domain, count);
            }
        }

        if activity_patterns.is_empty() && !sp_available {
            println!("\n  No activity data available.");
            println!("  Install: curl -fsSL get.screenpi.pe/cli | sh");
            println!("  Or ingest first: nagual activity ingest --since 1h");
        }

        println!("{:=<60}\n", "");
    }

    Ok(())
}
