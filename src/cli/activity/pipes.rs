//! `nagual activity pipes` — manage Screenpipe plugins (pipes).

use clap::{Args, Subcommand};

use crate::error::Result;

use super::client::ScreenpipeClient;
use super::helpers::resolve_screenpipe_url;

/// Manage Screenpipe pipes (plugins).
#[derive(Args, Debug)]
pub struct PipesCommand {
    #[command(subcommand)]
    pub subcommand: PipesSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum PipesSubcommand {
    /// List all installed pipes.
    List(PipesListArgs),

    /// Show details for a specific pipe.
    Info(PipesInfoArgs),

    /// Enable (start) a pipe.
    Enable(PipesToggleArgs),

    /// Disable (stop) a pipe.
    Disable(PipesToggleArgs),
}

#[derive(Args, Debug)]
pub struct PipesListArgs {
    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct PipesInfoArgs {
    /// Pipe ID to inspect.
    pub pipe_id: String,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,

    /// Output as JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct PipesToggleArgs {
    /// Pipe ID to enable/disable.
    pub pipe_id: String,

    /// Screenpipe API URL.
    #[arg(long, env = "SCREENPIPE_URL")]
    pub screenpipe_url: Option<String>,
}

impl PipesCommand {
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            PipesSubcommand::List(args) => run_pipes_list(args).await,
            PipesSubcommand::Info(args) => run_pipes_info(args).await,
            PipesSubcommand::Enable(args) => run_pipes_enable(args).await,
            PipesSubcommand::Disable(args) => run_pipes_disable(args).await,
        }
    }
}

async fn run_pipes_list(args: &PipesListArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    let pipes = client.pipes_list().await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&pipes)?);
    } else {
        println!("\nScreenpipe Pipes");
        println!("{:=<60}", "");

        if pipes.is_empty() {
            println!("  No pipes installed.");
        } else {
            println!(
                "  {:<30} {:<10} {}",
                "ID", "Status", "Source"
            );
            println!("  {:-<55}", "");
            for p in &pipes {
                let status = if p.enabled { "enabled" } else { "disabled" };
                let source = p.source.as_deref().unwrap_or("-");
                println!("  {:<30} {:<10} {}", p.id, status, source);
            }
        }

        println!("{:=<60}\n", "");
    }

    Ok(())
}

async fn run_pipes_info(args: &PipesInfoArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    let pipe = client.pipes_info(&args.pipe_id).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&pipe)?);
    } else {
        println!("\nPipe: {}", pipe.id);
        println!("{:=<50}", "");
        if let Some(ref name) = pipe.name {
            println!("  Name:    {}", name);
        }
        println!(
            "  Status:  {}",
            if pipe.enabled { "enabled" } else { "disabled" }
        );
        if let Some(ref source) = pipe.source {
            println!("  Source:  {}", source);
        }
        if let Some(port) = pipe.port {
            println!("  Port:    {}", port);
        }
        println!("{:=<50}\n", "");
    }

    Ok(())
}

async fn run_pipes_enable(args: &PipesToggleArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    client.pipes_enable(&args.pipe_id).await?;
    println!("Pipe \"{}\" enabled", args.pipe_id);

    Ok(())
}

async fn run_pipes_disable(args: &PipesToggleArgs) -> Result<()> {
    let url = resolve_screenpipe_url(args.screenpipe_url.as_deref());
    let client = ScreenpipeClient::new(&url);

    if client.health().await.is_none() {
        eprintln!("Screenpipe is not reachable at {}", url);
        return Ok(());
    }

    client.pipes_disable(&args.pipe_id).await?;
    println!("Pipe \"{}\" disabled", args.pipe_id);

    Ok(())
}
