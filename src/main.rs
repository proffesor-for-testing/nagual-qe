//! Nagual CLI - Self-Learning Agentic System
//!
//! Main entry point for the Nagual command-line interface.
//!
//! # Commands
//!
//! - `nagual migrate` - Run database migrations
//! - `nagual health` - Check system health
//! - `nagual conflicts` - Manage sync conflicts
//! - `nagual graph` - Graph operations (pressure, link, query, stats)
//! - `nagual learn` - Learning and self-improvement
//! - `nagual predict` - Prediction management
//! - `nagual knowledge` - Knowledge store/search/get/delete
//! - `nagual sync` - Backup, restore, and sync status
//! - `nagual patterns` - Pattern store/search/stats/consolidate
//! - `nagual session` - Session management and analytics
//! - `nagual status` - System status dashboard
//! - `nagual transfuse` - Gene Transfusion (extract patterns from codebases)
//! - `nagual plan` - Goal-Oriented Action Planning (GOAP)
//! - `nagual research` - Research Swarm for autonomous knowledge acquisition
//! - `nagual introspect` - Strange Loop self-analysis and recommendations
//! - `nagual coherence` - Coherence Gate for belief consistency verification
//! - `nagual constitution` - Display principles and operational rules
//! - `nagual dream` - Dream Cycle for background maintenance and consolidation

use clap::Parser;
use nagual::cli::{Cli, Commands};
use nagual::error::Result;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Check ONNX runtime environment before any ort code is touched.
/// This prevents hangs when ORT_DYLIB_PATH is missing.
fn check_onnx_environment() {
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        // Check common locations
        let common_paths = [
            "/opt/homebrew/lib/libonnxruntime.dylib",  // macOS ARM
            "/usr/local/lib/libonnxruntime.dylib",     // macOS Intel
            "/usr/lib/libonnxruntime.so",              // Linux
            "/usr/local/lib/libonnxruntime.so",        // Linux alt
        ];

        for path in common_paths {
            if std::path::Path::new(path).exists() {
                // Auto-set if found
                std::env::set_var("ORT_DYLIB_PATH", path);
                return;
            }
        }

        // Warn but don't fail - ML features will error when used
        eprintln!(
            "Warning: ORT_DYLIB_PATH not set and ONNX runtime not found in common locations.\n\
             ML features (embeddings) will not work. Set ORT_DYLIB_PATH to fix."
        );
    }
}

/// Parse CLI arguments before starting async runtime.
/// This allows --version and --help to exit cleanly without
/// initializing tokio or tracing infrastructure.
fn main() {
    // Check ONNX environment early to prevent hangs
    check_onnx_environment();

    // Parse CLI first - this handles --version and --help early,
    // exiting before we initialize any heavy infrastructure
    let cli = Cli::parse();

    // Now run the async main with the parsed CLI
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime")
        .block_on(async_main(cli))
        .unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        });
}

async fn async_main(cli: Cli) -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer().json())
        .with(EnvFilter::from_default_env().add_directive("nagual=info".parse().unwrap()))
        .init();

    match cli.command {
        Commands::Activity(cmd) => cmd.run().await,
        Commands::Apikey(cmd) => cmd.run().await,
        Commands::Cloud(cmd) => cmd.run().await,
        Commands::Coherence(cmd) => cmd.execute(cli.json).await,
        Commands::Constitution(cmd) => cmd.run().await,
        Commands::Dream(cmd) => cmd.run().await,
        Commands::Migrate(cmd) => cmd.run().await,
        Commands::Health(cmd) => cmd.run().await,
        Commands::Conflicts(cmd) => cmd.run().await,
        Commands::Graph(cmd) => cmd.run().await,
        Commands::Learn(cmd) => cmd.run().await,
        Commands::Predict(cmd) => cmd.run().await,
        Commands::Knowledge(cmd) => cmd.run().await,
        Commands::Sync(cmd) => cmd.run().await,
        Commands::Patterns(cmd) => cmd.run().await,
        Commands::Pulse(cmd) => cmd.run().await,
        Commands::Session(cmd) => cmd.run().await,
        Commands::Status(cmd) => cmd.run().await,
        Commands::Transfuse(cmd) => cmd.run().await,
        Commands::Plan(cmd) => cmd.execute(cli.json).await,
        Commands::Research(cmd) => nagual::cli::research::run(cmd).await,
        Commands::Introspect(cmd) => cmd.execute(cli.json).await,
        #[cfg(feature = "serve")]
        Commands::Serve(cmd) => cmd.run().await,
        Commands::User(cmd) => cmd.run().await,
        Commands::Kos(cmd) => cmd.run().await,
    }
}
