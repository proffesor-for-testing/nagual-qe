//! `nagual activity events` — UI event search and statistics.
//!
//! Queries Screenpipe's UI event tracking: clicks, keystrokes, clipboard,
//! app switches, scroll events, and more.

use chrono::Utc;
use clap::{Args, Subcommand};

use crate::error::Result;

use super::client::ScreenpipeClient;
use super::helpers::*;
use super::types::*;

/// Search and analyze UI events captured by Screenpipe.
#[derive(Args, Debug)]
pub struct EventsCommand {
    #[command(subcommand)]
    pub subcommand: EventsSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum EventsSubcommand {
    /// Search UI events.
    Search(EventsSearchArgs),

    /// Show UI event statistics by type and app.
    Stats(EventsStatsArgs),
}

#[derive(Args, Debug)]
pub struct EventsSearchArgs {
    /// Filter by event type: click, text, scroll, key, app_switch, clipboard.
    #[arg(long)]
    pub event_type: Option<String>,

    /// Filter by application name.
    #[arg(long)]
    pub app_name: Option<String>,

    /// Filter by window name (substring match).
    #[arg(long)]
    pub window_name: Option<String>,

    /// How far back to search (e.g. "1h", "1d").
    #[arg(long, default_value = "1h")]
    pub since: String,

    /// Maximum results.
    #[arg(long, default_value = "50")]
    pub limit: usize,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct EventsStatsArgs {
    /// How far back to analyze (e.g. "1h", "1d", "today").
    #[arg(long, default_value = "today")]
    pub since: String,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

impl EventsCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            EventsSubcommand::Search(args) => run_events_search(args).await,
            EventsSubcommand::Stats(args) => run_events_stats(args).await,
        }
    }
}

async fn run_events_search(args: &EventsSearchArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    let duration = parse_duration(&args.since);
    let end = Utc::now();
    let start = end - duration;
    let start_str = start.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let end_str = end.format("%Y-%m-%dT%H:%M:%SZ").to_string();

    let params = UiEventsParams {
        start_time: Some(&start_str),
        end_time: Some(&end_str),
        event_type: args.event_type.as_deref(),
        app_name: args.app_name.as_deref(),
        window_name: args.window_name.as_deref(),
        limit: args.limit,
        offset: 0,
    };

    let resp = client.ui_events(&params).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&resp.data)?);
    } else {
        println!("\nUI Events (last {})", args.since);
        println!("{:=<70}", "");

        if resp.data.is_empty() {
            println!("  No UI events found.");
        } else {
            for (i, ev) in resp.data.iter().enumerate() {
                let app = ev.app_name.as_deref().unwrap_or("unknown");
                let text = ev
                    .text_content
                    .as_deref()
                    .map(|t| truncate(t, 60))
                    .unwrap_or_default();
                println!(
                    "  {}. [{}] {} — {}{}",
                    i + 1,
                    ev.event_type,
                    app,
                    ev.timestamp,
                    if text.is_empty() {
                        String::new()
                    } else {
                        format!("\n     {}", text)
                    }
                );
            }
        }

        let total = resp
            .pagination
            .as_ref()
            .map(|p| p.total)
            .unwrap_or(resp.data.len());
        println!("\n{:=<70}", "");
        println!("  {} events shown (total: {})\n", resp.data.len(), total);
    }

    Ok(())
}

async fn run_events_stats(args: &EventsStatsArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    let duration = parse_duration(&args.since);
    let end = Utc::now();
    let start = end - duration;
    let start_str = start.format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let end_str = end.format("%Y-%m-%dT%H:%M:%SZ").to_string();

    let stats = client.ui_events_stats(&start_str, &end_str).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("\nUI Event Statistics (last {})", args.since);
        println!("{:=<60}", "");

        if stats.is_empty() {
            println!("  No UI event statistics available.");
        } else {
            println!(
                "  {:<20} {:>8} {}",
                "Event Type", "Count", "App"
            );
            println!("  {:-<55}", "");
            for s in &stats {
                let app = s.app_name.as_deref().unwrap_or("all");
                println!("  {:<20} {:>8} {}", s.event_type, s.count, app);
            }
        }

        println!("{:=<60}\n", "");
    }

    Ok(())
}
