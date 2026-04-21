//! `nagual activity tags` — add/remove tags on Screenpipe content.

use clap::{Args, Subcommand};

use crate::error::Result;

use super::client::ScreenpipeClient;
use super::helpers::resolve_screenpipe_url;

/// Manage tags on Screenpipe content items.
#[derive(Args, Debug)]
pub struct TagsCommand {
    #[command(subcommand)]
    pub subcommand: TagsSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum TagsSubcommand {
    /// Add tags to a content item.
    Add(TagsAddArgs),

    /// Remove tags from a content item.
    Remove(TagsRemoveArgs),
}

#[derive(Args, Debug)]
pub struct TagsAddArgs {
    /// Content type: vision, audio, or ui.
    pub content_type: String,

    /// Content item ID.
    pub id: i64,

    /// Tags to add.
    pub tags: Vec<String>,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct TagsRemoveArgs {
    /// Content type: vision, audio, or ui.
    pub content_type: String,

    /// Content item ID.
    pub id: i64,

    /// Tags to remove.
    pub tags: Vec<String>,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

impl TagsCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            TagsSubcommand::Add(args) => run_tags_add(args).await,
            TagsSubcommand::Remove(args) => run_tags_remove(args).await,
        }
    }
}

async fn run_tags_add(args: &TagsAddArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    client
        .add_tags(&args.content_type, args.id, args.tags.clone())
        .await?;

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "action": "add",
                "content_type": args.content_type,
                "id": args.id,
                "tags": args.tags,
            })
        );
    } else {
        println!(
            "Added {} tag(s) to {} #{}",
            args.tags.len(),
            args.content_type,
            args.id
        );
        for tag in &args.tags {
            println!("  + {}", tag);
        }
    }

    Ok(())
}

async fn run_tags_remove(args: &TagsRemoveArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    client
        .remove_tags(&args.content_type, args.id, args.tags.clone())
        .await?;

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "action": "remove",
                "content_type": args.content_type,
                "id": args.id,
                "tags": args.tags,
            })
        );
    } else {
        println!(
            "Removed {} tag(s) from {} #{}",
            args.tags.len(),
            args.content_type,
            args.id
        );
        for tag in &args.tags {
            println!("  - {}", tag);
        }
    }

    Ok(())
}
