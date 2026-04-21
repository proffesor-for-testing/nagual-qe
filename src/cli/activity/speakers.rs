//! `nagual activity speakers` — manage Screenpipe speaker identification.

use clap::{Args, Subcommand};

use crate::error::Result;

use super::client::ScreenpipeClient;
use super::helpers::resolve_screenpipe_url;

/// Manage audio speakers identified by Screenpipe.
#[derive(Args, Debug)]
pub struct SpeakersCommand {
    #[command(subcommand)]
    pub subcommand: SpeakersSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum SpeakersSubcommand {
    /// List or search speakers.
    List(SpeakersListArgs),

    /// Rename a speaker.
    Update(SpeakersUpdateArgs),

    /// Delete a speaker.
    Delete(SpeakersDeleteArgs),

    /// Merge two speakers (absorb one into another).
    Merge(SpeakersMergeArgs),

    /// Find speakers similar to a given speaker.
    Similar(SpeakersSimilarArgs),
}

#[derive(Args, Debug)]
pub struct SpeakersListArgs {
    /// Filter by speaker name (substring match).
    #[arg(long)]
    pub name: Option<String>,

    /// Show only unnamed speakers.
    #[arg(long)]
    pub unnamed: bool,

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
pub struct SpeakersUpdateArgs {
    /// Speaker ID.
    pub id: i64,

    /// New name for the speaker.
    pub name: String,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,
}

#[derive(Args, Debug)]
pub struct SpeakersDeleteArgs {
    /// Speaker ID to delete.
    pub id: i64,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,
}

#[derive(Args, Debug)]
pub struct SpeakersMergeArgs {
    /// Speaker ID to keep.
    pub keep_id: i64,

    /// Speaker ID to merge into the kept speaker.
    pub merge_id: i64,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,
}

#[derive(Args, Debug)]
pub struct SpeakersSimilarArgs {
    /// Speaker ID to find similar speakers for.
    pub id: i64,

    /// Maximum results.
    #[arg(long, default_value = "10")]
    pub limit: usize,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

impl SpeakersCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            SpeakersSubcommand::List(args) => run_speakers_list(args).await,
            SpeakersSubcommand::Update(args) => run_speakers_update(args).await,
            SpeakersSubcommand::Delete(args) => run_speakers_delete(args).await,
            SpeakersSubcommand::Merge(args) => run_speakers_merge(args).await,
            SpeakersSubcommand::Similar(args) => run_speakers_similar(args).await,
        }
    }
}

async fn run_speakers_list(args: &SpeakersListArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    let speakers = if args.unnamed {
        client.speakers_unnamed(args.limit, 0).await?
    } else {
        client.speakers_search(args.name.as_deref()).await?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&speakers)?);
    } else {
        println!("\nSpeakers");
        println!("{:=<50}", "");
        if speakers.is_empty() {
            println!("  No speakers found.");
        } else {
            println!("  {:<8} {}", "ID", "Name");
            println!("  {:-<45}", "");
            for s in &speakers {
                let name = if s.name.is_empty() {
                    "(unnamed)".to_string()
                } else {
                    s.name.clone()
                };
                println!("  {:<8} {}", s.id, name);
            }
        }
        println!("{:=<50}\n", "");
    }

    Ok(())
}

async fn run_speakers_update(args: &SpeakersUpdateArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    client.speakers_update(args.id, &args.name).await?;
    println!("Speaker #{} renamed to \"{}\"", args.id, args.name);

    Ok(())
}

async fn run_speakers_delete(args: &SpeakersDeleteArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    client.speakers_delete(args.id).await?;
    println!("Speaker #{} deleted", args.id);

    Ok(())
}

async fn run_speakers_merge(args: &SpeakersMergeArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    client.speakers_merge(args.keep_id, args.merge_id).await?;
    println!(
        "Merged speaker #{} into #{}",
        args.merge_id, args.keep_id
    );

    Ok(())
}

async fn run_speakers_similar(args: &SpeakersSimilarArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    let speakers = client.speakers_similar(args.id, args.limit).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&speakers)?);
    } else {
        println!("\nSpeakers similar to #{}", args.id);
        println!("{:=<50}", "");
        if speakers.is_empty() {
            println!("  No similar speakers found.");
        } else {
            println!("  {:<8} {}", "ID", "Name");
            println!("  {:-<45}", "");
            for s in &speakers {
                let name = if s.name.is_empty() {
                    "(unnamed)".to_string()
                } else {
                    s.name.clone()
                };
                println!("  {:<8} {}", s.id, name);
            }
        }
        println!("{:=<50}\n", "");
    }

    Ok(())
}
