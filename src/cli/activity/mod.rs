//! Activity tracking CLI commands via Screenpipe integration.
//!
//! Pulls screen capture, OCR, audio transcription, and UI event data from
//! Screenpipe's REST API and stores it as patterns in Nagual's knowledge base.
//!
//! ## Subcommands
//!
//! - `status`   — Health check, device listing, DB row counts
//! - `ingest`   — Bulk ingest activity into the knowledge base
//! - `summary`  — Activity summary by time period
//! - `search`   — Full-text, semantic, or keyword search across activity
//! - `apps`     — App usage breakdown
//! - `tags`     — Add/remove tags on Screenpipe content
//! - `speakers` — Manage identified speakers from audio
//! - `events`   — Search and analyze UI events (clicks, keys, clipboard)
//! - `pipes`    — List and manage Screenpipe plugins

mod apps;
mod client;
mod helpers;
mod ingest;
pub mod ospipe;
mod pipes;
mod search;
mod speakers;
mod status;
mod summary;
mod tags;
mod types;
mod ui_events;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use clap::{Args, Subcommand};

use crate::error::Result;

// Re-export for use in cli/mod.rs
pub use self::pipes::PipesCommand;
pub use self::speakers::SpeakersCommand;
pub use self::tags::TagsCommand;
pub use self::ui_events::EventsCommand;

// ---------------------------------------------------------------------------
// CLI argument types
// ---------------------------------------------------------------------------

/// Activity tracking commands via Screenpipe integration.
///
/// Track, ingest, and analyze screen activity data from Screenpipe.
#[derive(Args, Debug)]
pub struct ActivityCommand {
    #[command(subcommand)]
    pub subcommand: ActivitySubcommand,
}

#[derive(Subcommand, Debug)]
pub enum ActivitySubcommand {
    /// Check if Screenpipe is running and show connection stats.
    Status(StatusArgs),

    /// Ingest recent activity from Screenpipe into the knowledge base.
    Ingest(IngestArgs),

    /// Generate an activity summary for a time period.
    Summary(SummaryArgs),

    /// Search across activity history.
    Search(SearchArgs),

    /// Show app usage breakdown.
    Apps(AppsArgs),

    /// Add or remove tags on Screenpipe content.
    Tags(TagsCommand),

    /// Manage identified audio speakers.
    Speakers(SpeakersCommand),

    /// Search and analyze UI events (clicks, keystrokes, clipboard).
    Events(EventsCommand),

    /// List and manage Screenpipe pipes (plugins).
    Pipes(PipesCommand),
}

#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct IngestArgs {
    /// How far back to ingest (e.g. "1h", "4h", "1d").
    #[arg(long, default_value = "1h")]
    pub since: String,

    /// Content type filter: ocr, audio, input, all.
    #[arg(long, default_value = "all")]
    pub content_type: String,

    /// Filter by application name.
    #[arg(long)]
    pub app_name: Option<String>,

    /// Only ingest focused (active) window content.
    #[arg(long)]
    pub focused_only: bool,

    /// Minimum OCR text length to include (skip tiny fragments).
    #[arg(long, default_value = "50")]
    pub min_length: usize,

    /// Generate 768-dim embeddings via Screenpipe's local model.
    #[arg(long)]
    pub embed: bool,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,

    // ======================= OSpipe Options =======================

    /// Enable OSpipe pipeline (PII protection + deduplication + embeddings).
    #[arg(long)]
    pub ospipe: bool,

    /// PII policy: reject, redact, warn, allow. Default: redact.
    #[arg(long, default_value = "redact")]
    pub pii_policy: String,

    /// Cosine similarity threshold for deduplication (0.0-1.0). Default: 0.9.
    #[arg(long, default_value = "0.9")]
    pub dedup_threshold: f32,

    /// Time window for deduplication (e.g. "5m", "10m"). Default: 5m.
    #[arg(long, default_value = "5m")]
    pub dedup_window: String,

    /// Embedding dimension: 128 or 384. Default: 128 (nagual native).
    #[arg(long, default_value = "128")]
    pub embedding_dim: String,
}

#[derive(Args, Debug)]
pub struct SummaryArgs {
    /// Time period: "today", "1d", "7d".
    #[arg(long, default_value = "today")]
    pub period: String,

    /// Domain filter: coding, browsing, meetings, all.
    #[arg(long, default_value = "all")]
    pub domain: String,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Search query text.
    #[arg(long)]
    pub query: String,

    /// Maximum number of results.
    #[arg(long, default_value = "10")]
    pub limit: usize,

    /// How far back to search (e.g. "1h", "1d").
    #[arg(long)]
    pub since: Option<String>,

    /// Filter by application name.
    #[arg(long)]
    pub app_name: Option<String>,

    /// Filter by window title (substring match).
    #[arg(long)]
    pub window_name: Option<String>,

    /// Only show focused (active) window results.
    #[arg(long)]
    pub focused_only: bool,

    /// Minimum text length.
    #[arg(long)]
    pub min_length: Option<usize>,

    /// Use semantic (embedding-based) similarity search.
    #[arg(long)]
    pub semantic: bool,

    /// Use keyword search with text positions.
    #[arg(long)]
    pub keyword: bool,

    /// Similarity threshold for semantic search (0.0–1.0).
    #[arg(long)]
    pub threshold: Option<f32>,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct AppsArgs {
    /// Time period: "today", "1d", "7d".
    #[arg(long, default_value = "today")]
    pub period: String,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

impl ActivityCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            ActivitySubcommand::Status(args) => status::run_status(args).await,
            ActivitySubcommand::Ingest(args) => ingest::run_ingest(args).await,
            ActivitySubcommand::Summary(args) => summary::run_summary(args).await,
            ActivitySubcommand::Search(args) => search::run_search(args).await,
            ActivitySubcommand::Apps(args) => apps::run_apps(args).await,
            ActivitySubcommand::Tags(cmd) => cmd.run().await,
            ActivitySubcommand::Speakers(cmd) => cmd.run().await,
            ActivitySubcommand::Events(cmd) => cmd.run().await,
            ActivitySubcommand::Pipes(cmd) => cmd.run().await,
        }
    }
}
