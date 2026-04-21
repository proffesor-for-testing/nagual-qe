//! `nagual activity apps` — app usage breakdown.

use chrono::Utc;

use crate::error::Result;

use super::client::ScreenpipeClient;
use super::helpers::*;
use super::types::*;
use super::AppsArgs;

pub(super) async fn run_apps(args: &AppsArgs) -> Result<()> {
    let sp_url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&sp_url);

    if client.health().await.is_none() {
        if args.json {
            println!(
                "{}",
                serde_json::json!({"error": "Screenpipe is not reachable", "url": sp_url})
            );
        } else {
            eprintln!("Screenpipe is not reachable at {}", sp_url);
            eprintln!("Install: curl -fsSL get.screenpi.pe/cli | sh");
        }
        return Ok(());
    }

    let duration = parse_duration(&args.period);
    let end = Utc::now();
    let start = end - duration;
    let start_str = start.format("%Y-%m-%dT%H:%M:%SZ").to_string();

    let sql = format!(
        "SELECT COALESCE(app_name, 'unknown') as app_name, COUNT(*) as cnt \
         FROM frames WHERE timestamp >= '{}' \
         GROUP BY app_name ORDER BY cnt DESC",
        start_str,
    );

    let mut sorted: Vec<AppUsageItem> = match client.raw_sql(&sql).await {
        Ok(rows) => rows
            .iter()
            .map(|r| {
                let app = r["app_name"].as_str().unwrap_or("unknown").to_string();
                let count = r["cnt"].as_u64().unwrap_or(0) as usize;
                AppUsageItem {
                    domain: categorize_app(&app).to_string(),
                    app_name: app,
                    count,
                }
            })
            .collect(),
        Err(_) => vec![],
    };
    sorted.sort_by(|a, b| b.count.cmp(&a.count));

    if args.json {
        println!("{}", serde_json::to_string_pretty(&sorted)?);
    } else {
        println!("\nApp Usage ({})", args.period);
        println!("{:=<60}", "");
        println!(
            "  {:<25} {:>8} {:>12}",
            "App", "Items", "Domain"
        );
        println!("  {:-<55}", "");

        let total: usize = sorted.iter().map(|a| a.count).sum();
        for app in &sorted {
            let pct = if total > 0 {
                (app.count as f64 / total as f64 * 100.0) as u32
            } else {
                0
            };
            println!(
                "  {:<25} {:>8} {:>10}  {}%",
                truncate(&app.app_name, 25),
                app.count,
                app.domain,
                pct
            );
        }

        println!("  {:-<55}", "");
        println!("  Total items: {}", total);
        println!("{:=<60}\n", "");
    }

    Ok(())
}
