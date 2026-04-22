//! Knowledge management CLI commands.
//!
//! Provides commands for storing, searching, retrieving, and deleting
//! knowledge items in the ReasoningBank with persistent SQLite storage.
//!
//! # Usage Examples
//!
//! ```bash
//! # Store new knowledge
//! nagual knowledge store "How to handle async errors" --domain rust.async --tags "error-handling,async"
//!
//! # Search knowledge
//! nagual knowledge search "async error handling" --limit 10 --domain rust
//!
//! # Get specific item
//! nagual knowledge get abc123-def456
//!
//! # Delete item
//! nagual knowledge delete abc123-def456
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Subcommand};
use serde::Serialize;

use super::common::{init_storage, resolve_postgres_url};
use crate::constitution::{Constitution, Operation, OperationContext};
use crate::db::{DualWriteAdapter, DualWriteConfig, PostgresDb, SqliteDb};
use crate::error::{NagualError, Result};
use crate::events::{EventBus, NagualEvent};
use crate::ml::to_array1;
#[cfg(feature = "onnx-embed")]
use crate::ml::{Embedder, EmbedderConfig};
#[cfg(not(feature = "onnx-embed"))]
use crate::ml::HashEmbedder;
use crate::ml::LoraStorage;
use crate::reasoning_bank::pattern::{Pattern, PatternCategory, PatternId, PatternMetadata};
use crate::reasoning_bank::storage::{PatternStorage, StorageConfig};
use crate::reasoning_bank::{
    self as rb, retrieve_patterns_hyperbolic, staged_retrieve_patterns, HyperbolicRetrievalConfig,
    PatternQuery, RetrievalConfig, RetrievalStaging,
};

/// Parse a floating point value and validate it's in the unit interval [0.0, 1.0].
fn parse_unit_interval(s: &str) -> std::result::Result<f32, String> {
    let value: f32 = s.parse().map_err(|e| format!("Invalid number: {}", e))?;
    if !(0.0..=1.0).contains(&value) {
        return Err(format!(
            "Value {} out of range. Must be between 0.0 and 1.0 inclusive.",
            value
        ));
    }
    Ok(value)
}

/// Knowledge management commands.
///
/// Store, search, retrieve, and delete knowledge items from the ReasoningBank.
#[derive(Args, Debug)]
pub struct KnowledgeCommand {
    #[command(subcommand)]
    pub subcommand: KnowledgeSubcommand,
}

/// Knowledge subcommands.
#[derive(Subcommand, Debug)]
pub enum KnowledgeSubcommand {
    /// Store new knowledge in the ReasoningBank.
    ///
    /// Creates a new knowledge item with the provided content, domain,
    /// and optional tags. Returns the generated ID.
    Store(StoreArgs),

    /// Search knowledge by text query or embedding.
    ///
    /// Performs similarity search using text queries, optionally
    /// filtering by domain and tags.
    Search(SearchArgs),

    /// Get a specific knowledge item by ID.
    ///
    /// Retrieves the full details of a knowledge item including
    /// all metadata and statistics.
    Get(GetArgs),

    /// Delete a knowledge item by ID.
    ///
    /// Permanently removes a knowledge item from the database.
    Delete(DeleteArgs),

    /// List knowledge items with optional filtering.
    ///
    /// Shows a paginated list of knowledge items with optional
    /// domain and tag filters.
    List(ListArgs),

    /// Sync all patterns from SQLite to PostgreSQL.
    ///
    /// Reads all patterns from the local SQLite database and upserts
    /// them into PostgreSQL for dual-write consistency.
    Sync(SyncArgs),

    /// Import patterns from a JSONL seed file.
    ///
    /// Reads a JSONL file (one pattern per line) and stores each record
    /// in the ReasoningBank. Patterns are marked with
    /// `metadata.source = "seed"` for later filtering. Import is
    /// idempotent on content hash — re-running with the same seed
    /// skips records that already exist.
    Import(ImportArgs),
}

/// Arguments for the store subcommand.
#[derive(Args, Debug)]
pub struct StoreArgs {
    /// The content/problem description to store.
    #[arg(value_name = "CONTENT", allow_hyphen_values = true)]
    pub content: String,

    /// Optional solution for the knowledge item.
    #[arg(short, long, allow_hyphen_values = true)]
    pub solution: Option<String>,

    /// Domain/category for the knowledge (e.g., "rust.async", "database").
    #[arg(short, long, default_value = "general")]
    pub domain: String,

    /// Tags for the knowledge item (comma-separated).
    #[arg(short, long, value_delimiter = ',')]
    pub tags: Vec<String>,

    /// Context or additional information.
    #[arg(long, allow_hyphen_values = true)]
    pub context: Option<String>,

    /// Initial effectiveness score (0.0-1.0).
    #[arg(long, default_value = "0.5", value_parser = parse_unit_interval)]
    pub effectiveness: f32,

    /// Initial confidence score (0.0-1.0).
    #[arg(long, default_value = "0.5", value_parser = parse_unit_interval)]
    pub confidence: f32,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL for dual-write.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Use demo mode (in-memory database).
    #[arg(long)]
    pub demo: bool,
}

/// Arguments for the search subcommand.
#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Search query text.
    #[arg(value_name = "QUERY")]
    pub query: String,

    /// Filter by domain (supports hierarchy, e.g., "rust" matches "rust.async").
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Filter by tags (comma-separated, matches any).
    #[arg(short, long, value_delimiter = ',')]
    pub tags: Vec<String>,

    /// Minimum reward threshold (0.0-1.0).
    #[arg(long)]
    pub min_reward: Option<f32>,

    /// Maximum number of results to return.
    #[arg(short, long, default_value = "10")]
    pub limit: usize,

    /// Offset for pagination.
    #[arg(long, default_value = "0")]
    pub offset: usize,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL for dual-write.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show verbose output including embeddings.
    #[arg(short, long)]
    pub verbose: bool,

    /// Use hyperbolic (Poincare ball) distance for hierarchy-aware retrieval.
    ///
    /// Re-ranks results using hyperbolic geometry, favoring patterns
    /// in the same domain hierarchy over those at similar Euclidean distance.
    #[arg(long)]
    pub hyperbolic: bool,

    /// Use demo mode with sample data.
    #[arg(long)]
    pub demo: bool,

    /// FTS keyword weight in hybrid search (0.0-1.0, default 0.3).
    #[arg(long, default_value = "0.3")]
    pub fts_weight: f32,

    /// Vector similarity weight in hybrid search (0.0-1.0, default 0.7).
    #[arg(long, default_value = "0.7")]
    pub vector_weight: f32,
}

/// Arguments for the get subcommand.
#[derive(Args, Debug)]
pub struct GetArgs {
    /// The knowledge item ID to retrieve.
    #[arg(value_name = "ID")]
    pub id: String,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL for dual-write.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show verbose output including embeddings.
    #[arg(short, long)]
    pub verbose: bool,

    /// Use demo mode with sample data.
    #[arg(long)]
    pub demo: bool,
}

/// Arguments for the delete subcommand.
#[derive(Args, Debug)]
pub struct DeleteArgs {
    /// The knowledge item ID to delete.
    #[arg(value_name = "ID")]
    pub id: String,

    /// Force deletion without confirmation.
    #[arg(short, long)]
    pub force: bool,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL for dual-write.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the list subcommand.
#[derive(Args, Debug)]
pub struct ListArgs {
    /// Filter by domain.
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Filter by tags (comma-separated).
    #[arg(short, long, value_delimiter = ',')]
    pub tags: Vec<String>,

    /// Sort by field: created, updated, reward, effectiveness, usage.
    #[arg(long, default_value = "updated")]
    pub sort: String,

    /// Sort order: asc or desc.
    #[arg(long, default_value = "desc")]
    pub order: String,

    /// Maximum number of results.
    #[arg(short, long, default_value = "20")]
    pub limit: usize,

    /// Offset for pagination.
    #[arg(long, default_value = "0")]
    pub offset: usize,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL for dual-write.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Use demo mode with sample data.
    #[arg(long)]
    pub demo: bool,
}

/// Arguments for the sync subcommand.
#[derive(Args, Debug)]
pub struct SyncArgs {
    /// Path to SQLite database (source).
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL (destination).
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Batch size for sync operations.
    #[arg(long, default_value = "100")]
    pub batch_size: usize,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the import subcommand.
#[derive(Args, Debug)]
pub struct ImportArgs {
    /// Path to the JSONL seed file (one pattern record per line).
    #[arg(long, value_name = "FILE")]
    pub seed: PathBuf,

    /// Parse and validate the seed file without writing anything.
    #[arg(long)]
    pub dry_run: bool,

    /// Prepend this string to every pattern's domain (e.g. "seed.").
    #[arg(long, value_name = "PREFIX")]
    pub domain_prefix: Option<String>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// PostgreSQL connection URL for dual-write.
    #[arg(long, env = "DATABASE_URL")]
    pub postgres_url: Option<String>,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

impl KnowledgeCommand {
    /// Execute the knowledge command.
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            KnowledgeSubcommand::Store(args) => run_store(args).await,
            KnowledgeSubcommand::Search(args) => run_search(args).await,
            KnowledgeSubcommand::Get(args) => run_get(args).await,
            KnowledgeSubcommand::Delete(args) => run_delete(args).await,
            KnowledgeSubcommand::List(args) => run_list(args).await,
            KnowledgeSubcommand::Sync(args) => run_sync(args).await,
            KnowledgeSubcommand::Import(args) => run_import(args).await,
        }
    }
}

/// Run the store command.
async fn run_store(args: &StoreArgs) -> Result<()> {
    tracing::info!("Storing new knowledge item");

    // Build the pattern
    let pattern = Pattern::builder()
        .problem(&args.content)
        .solution(args.solution.as_deref().unwrap_or(""))
        .category(PatternCategory::from(args.domain.as_str()))
        .context(args.context.as_deref().unwrap_or(""))
        .effectiveness(args.effectiveness)
        .confidence(args.confidence)
        .tags(args.tags.clone())
        .build();

    let id = pattern.id().to_string();

    if args.demo {
        // Demo mode - just show what would be stored
        if args.json {
            let output = StoreOutput {
                id: id.clone(),
                domain: args.domain.clone(),
                tags: args.tags.clone(),
                success: true,
                message: "Knowledge stored successfully (demo mode)".to_string(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("\nKnowledge Stored Successfully (demo mode)");
            println!("{:-<60}", "");
            println!("ID: {}", id);
            println!("Domain: {}", args.domain);
            println!("Tags: {}", args.tags.join(", "));
            println!("Content: {}", truncate(&args.content, 100));
            if let Some(ref solution) = args.solution {
                println!("Solution: {}", truncate(solution, 100));
            }
            println!("{:-<60}\n", "");
        }
    } else {
        // Initialize storage with dual-write support
        let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;
        let stored_id = storage.store_pattern(&pattern).await?;

        // Emit PatternStored event (F16)
        let event_bus = EventBus::new();
        event_bus.publish_sync(NagualEvent::pattern_stored(
            &stored_id.to_string(),
            &args.domain,
        ));

        // Allow background PostgreSQL write to complete before CLI exits
        if resolve_postgres_url(args.postgres_url.as_deref()).is_some() {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }

        tracing::info!(
            pattern_id = %stored_id,
            db_path = %args.db_path.display(),
            "Pattern stored"
        );

        if args.json {
            let output = StoreOutput {
                id: stored_id.to_string(),
                domain: args.domain.clone(),
                tags: args.tags.clone(),
                success: true,
                message: format!("Knowledge stored to {}", args.db_path.display()),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("\nKnowledge Stored Successfully");
            println!("{:-<60}", "");
            println!("ID: {}", stored_id);
            println!("Domain: {}", args.domain);
            println!("Tags: {}", args.tags.join(", "));
            println!("Database: {}", args.db_path.display());
            println!("{:-<60}\n", "");
        }
    }

    Ok(())
}

/// Run the search command.
async fn run_search(args: &SearchArgs) -> Result<()> {
    tracing::info!("Searching knowledge: {}", args.query);

    // Initialize storage outside the block so it's available for FTS5 search later
    let storage = if !args.demo {
        Some(init_storage(&args.db_path, args.postgres_url.as_deref()).await?)
    } else {
        None
    };

    let patterns = if args.demo {
        create_demo_patterns()
    } else {
        let storage = storage.as_ref().unwrap();

        // Get recent patterns from database (needed for hyperbolic retrieval)
        let all_patterns = storage.get_recent(args.limit * 10).await?;

        if all_patterns.is_empty() {
            tracing::info!("No patterns in database, showing demo data hint");
            if !args.json {
                println!("\nNo patterns found in database at: {}", args.db_path.display());
                println!("Use 'nagual knowledge store' to add patterns, or --demo for sample data.\n");
            }
        }

        all_patterns
    };

    // Hyperbolic retrieval (F09) or text matching
    #[allow(unused_mut)]
    let mut results: Vec<_> = if args.hyperbolic && !args.demo {
        #[cfg(feature = "onnx-embed")]
        {
        // Try embedding-based hyperbolic retrieval
        let model_path = "models/all-MiniLM-L6-v2.onnx";
        let tokenizer_path = "models/tokenizer.json";
        if std::path::Path::new(model_path).exists() {
            let config = EmbedderConfig::dim_128(model_path, tokenizer_path);
            match Embedder::new(&config) {
                Ok(embedder) => match embedder.embed(&args.query) {
                    Ok(embed_result) => {
                        let base_embedding = to_array1(&embed_result.embedding);

                        // Apply LoRA domain adapter if available (F11)
                        let query_embedding = if let Some(ref domain) = args.domain {
                            let lora_storage = LoraStorage::new("./models/lora");
                            match lora_storage.load(domain) {
                                Ok(adapter) => {
                                    match adapter.transform(&base_embedding.view()) {
                                        Ok(transformed) => {
                                            tracing::info!("Applied LoRA adapter for domain '{}'", domain);
                                            transformed
                                        }
                                        Err(_) => base_embedding,
                                    }
                                }
                                Err(_) => base_embedding,
                            }
                        } else {
                            base_embedding
                        };

                        let mut pq = PatternQuery::new(&args.query);
                        if let Some(ref domain) = args.domain {
                            pq = pq.with_domains(vec![domain.as_str()]);
                        }
                        if let Some(min_reward) = args.min_reward {
                            pq = pq.with_min_reward(min_reward);
                        }
                        pq = pq.with_limit(args.limit);

                        let retrieval_config = RetrievalConfig::default();
                        let hyper_config = HyperbolicRetrievalConfig::default();

                        // Convert CLI patterns to retrieval format
                        let rb_patterns: Vec<rb::Pattern> =
                            patterns.iter().map(rb::Pattern::from).collect();

                        // Use staged retrieval (F06) feeding into hyperbolic re-ranking (F09)
                        let mut staging = RetrievalStaging::new();
                        let _ = staged_retrieve_patterns(
                            &mut staging,
                            &rb_patterns,
                            &query_embedding.view(),
                            &pq,
                            &retrieval_config,
                        ); // Populate staging cache for future calls

                        match retrieve_patterns_hyperbolic(
                            &rb_patterns,
                            &query_embedding.view(),
                            &pq,
                            &retrieval_config,
                            &hyper_config,
                        ) {
                            Ok(result) => {
                                if !args.json {
                                    println!("  (hyperbolic retrieval: {} candidates scored)", result.total_candidates);
                                }
                                result
                                    .patterns
                                    .into_iter()
                                    .map(|sp| {
                                        Pattern::builder()
                                            .id(sp.pattern.id.as_str())
                                            .problem(&sp.pattern.problem)
                                            .solution(&sp.pattern.solution)
                                            .category(PatternCategory::from(
                                                sp.pattern.domain.as_str(),
                                            ))
                                            .context(
                                                sp.pattern
                                                    .context
                                                    .as_deref()
                                                    .unwrap_or(""),
                                            )
                                            .confidence(sp.pattern.confidence)
                                            .reward(sp.pattern.reward)
                                            .reuse_count(sp.pattern.usage_count)
                                            .tags(sp.pattern.tags)
                                            .build()
                                    })
                                    .collect()
                            }
                            Err(e) => {
                                tracing::warn!("Hyperbolic retrieval failed, falling back to text search: {}", e);
                                Vec::new() // fall through to text search below
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Could not embed query: {}. Falling back to text search.", e);
                        Vec::new()
                    }
                },
                Err(e) => {
                    eprintln!("Warning: Could not load embedding model: {}. Falling back to text search.", e);
                    Vec::new()
                }
            }
        } else {
            eprintln!("Warning: ONNX model not found at {}. Run 'nagual learn embed' first.", model_path);
            eprintln!("Falling back to text search.\n");
            Vec::new()
        }
        } // end #[cfg(feature = "onnx-embed")] block
        #[cfg(not(feature = "onnx-embed"))]
        {
            // Use HashEmbedder as a lightweight embedding fallback
            let hash_embedder = HashEmbedder::new();
            match hash_embedder.embed(&args.query) {
                Ok(embed_result) => {
                    let query_embedding = to_array1(&embed_result.embedding);
                    if !args.json {
                        println!("  (using hash embedder for similarity search)");
                    }
                    // Score all patterns by cosine similarity to the hash embedding
                    let mut scored: Vec<(f32, Pattern)> = patterns
                        .iter()
                        .filter_map(|p| {
                            p.embedding().map(|emb| {
                                let emb_arr = to_array1(emb);
                                let sim = crate::ml::cosine_similarity(
                                    &query_embedding.view(),
                                    &emb_arr.view(),
                                );
                                (sim, p.clone())
                            })
                        })
                        .filter(|(sim, _)| *sim > 0.0)
                        .collect();
                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    scored.into_iter().take(args.limit).map(|(_, p)| p).collect()
                }
                Err(_) => Vec::new(),
            }
        }
    } else {
        Vec::new() // empty signals: use text matching fallback
    };

    // FTS5 search fallback (when hyperbolic unavailable or not requested)
    if results.is_empty() {
        // Try FTS5 first — proper full-text search with BM25 ranking
        if let Some(ref storage) = storage {
            match storage.fts_search(&args.query, args.limit + args.offset).await {
                Ok(fts_results) if !fts_results.is_empty() => {
                    tracing::info!("FTS5 search returned {} results", fts_results.len());
                    results = fts_results
                        .into_iter()
                        .filter(|p| {
                            if let Some(ref domain) = args.domain {
                                p.category().to_string().starts_with(domain)
                            } else {
                                true
                            }
                        })
                        .filter(|p| {
                            if let Some(min_reward) = args.min_reward {
                                p.reward() >= min_reward
                            } else {
                                true
                            }
                        })
                        .skip(args.offset)
                        .take(args.limit)
                        .collect();
                }
                Ok(_) => {
                    tracing::info!("FTS5 returned no results, falling back to substring match");
                }
                Err(e) => {
                    tracing::warn!("FTS5 search failed ({}), falling back to substring match", e);
                }
            }
        }

        // Final fallback: naive substring matching on preloaded patterns
        if results.is_empty() {
            let query_lower = args.query.to_lowercase();
            let query_words: Vec<&str> = query_lower.split_whitespace().collect();
            results = patterns
                .into_iter()
                .filter(|p| {
                    let problem = p.problem().to_lowercase();
                    let solution = p.solution().to_lowercase();
                    // Match if ALL words appear in problem or solution (not as one substring)
                    query_words.iter().all(|word| {
                        problem.contains(word) || solution.contains(word)
                    })
                })
                .filter(|p| {
                    if let Some(ref domain) = args.domain {
                        p.category().to_string().starts_with(domain)
                    } else {
                        true
                    }
                })
                .filter(|p| {
                    if let Some(min_reward) = args.min_reward {
                        p.reward() >= min_reward
                    } else {
                        true
                    }
                })
                .skip(args.offset)
                .take(args.limit)
                .collect();
        }

        // Sort by relevance (simplified: by reward)
        results.sort_by(|a, b| b.reward().partial_cmp(&a.reward()).unwrap_or(std::cmp::Ordering::Equal));
    }

    if args.json {
        let output: Vec<SearchResultItem> = results
            .iter()
            .map(|p| SearchResultItem {
                id: p.id().to_string(),
                problem: p.problem().to_string(),
                solution: p.solution().to_string(),
                domain: p.category().to_string(),
                reward: p.reward(),
                effectiveness: p.effectiveness(),
                tags: p.tags().to_vec(),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nSearch Results for: \"{}\"", args.query);
        println!("{:=<70}", "");

        if results.is_empty() {
            println!("\nNo results found.");
        } else {
            for (i, pattern) in results.iter().enumerate() {
                println!(
                    "\n{}. {} (reward: {:.2}, effectiveness: {:.2})",
                    i + 1,
                    truncate(pattern.problem(), 60),
                    pattern.reward(),
                    pattern.effectiveness()
                );
                println!("   ID: {}", pattern.id());
                println!("   Domain: {}", pattern.category());
                if !pattern.tags().is_empty() {
                    println!("   Tags: {}", pattern.tags().join(", "));
                }
                if args.verbose {
                    println!("   Solution: {}", truncate(pattern.solution(), 80));
                }
            }
        }

        println!("\n{:=<70}", "");
        println!("Found {} results\n", results.len());
    }

    // Record usage for returned patterns (feeds auto-promotion engine)
    if !results.is_empty() {
        if let Some(ref storage) = storage {
            // Try to get active session ID for context tracking
            let session_id: Option<String> = {
                let session_mgr = crate::db::SessionManager::new(storage.adapter().sqlite().clone());
                match session_mgr.get_active_session().await {
                    Ok(Some(session)) => Some(session.id),
                    _ => None,
                }
            };

            for pattern in &results {
                let id_str = pattern.id().to_string();
                let _ = storage
                    .record_pattern_usage(
                        &id_str,
                        session_id.as_deref(),
                        None, // no task_id in CLI context
                        "retrieval",
                    )
                    .await;
            }

            tracing::debug!(
                count = results.len(),
                session_id = ?session_id,
                "Recorded pattern usage for search results"
            );
        }
    }

    Ok(())
}

/// Run the get command.
async fn run_get(args: &GetArgs) -> Result<()> {
    tracing::info!("Getting knowledge item: {}", args.id);

    let pattern = if args.demo {
        let patterns = create_demo_patterns();
        patterns.into_iter().find(|p| p.id().to_string() == args.id)
    } else {
        // Retrieve from database
        let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;
        let pattern_id = PatternId::from(args.id.as_str());
        storage.get_pattern(&pattern_id).await?
    };

    match pattern {
        Some(p) => {
            if args.json {
                let output = KnowledgeItem {
                    id: p.id().to_string(),
                    problem: p.problem().to_string(),
                    solution: p.solution().to_string(),
                    context: p.context().to_string(),
                    domain: p.category().to_string(),
                    effectiveness: p.effectiveness(),
                    reward: p.reward(),
                    confidence: p.confidence(),
                    reuse_count: p.reuse_count(),
                    tags: p.tags().to_vec(),
                    created_at: p.timestamp().to_rfc3339(),
                    updated_at: p.updated_at().to_rfc3339(),
                };
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("\nKnowledge Item Details");
                println!("{:=<60}", "");
                println!("ID: {}", p.id());
                println!("Domain: {}", p.category());
                println!("Tags: {}", p.tags().join(", "));
                println!("{:-<60}", "");
                println!("Problem:");
                println!("  {}", p.problem());
                println!("\nSolution:");
                println!("  {}", p.solution());
                if !p.context().is_empty() {
                    println!("\nContext:");
                    println!("  {}", p.context());
                }
                println!("{:-<60}", "");
                println!("Effectiveness: {:.3}", p.effectiveness());
                println!("Reward: {:.3}", p.reward());
                println!("Quality: {:.3}", p.bayesian_score().mean());
                println!("Confidence: {:.3}", p.confidence());
                println!("Reuse Count: {}", p.reuse_count());
                println!("Created: {}", p.timestamp().format("%Y-%m-%d %H:%M:%S UTC"));
                println!("Updated: {}", p.updated_at().format("%Y-%m-%d %H:%M:%S UTC"));
                if args.verbose && p.has_embedding() {
                    println!("Embedding: {} dimensions", p.embedding().map(|e| e.len()).unwrap_or(0));
                }
                println!("{:=<60}\n", "");
            }
        }
        None => {
            if args.json {
                let output = ErrorOutput {
                    error: format!("Knowledge item not found: {}", args.id),
                    error_type: "not_found".to_string(),
                };
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                eprintln!("Error: Knowledge item not found: {}", args.id);
            }
        }
    }

    Ok(())
}

/// Run the delete command.
async fn run_delete(args: &DeleteArgs) -> Result<()> {
    tracing::info!("Deleting knowledge item: {}", args.id);

    if !args.force {
        println!("Warning: This will permanently delete the knowledge item.");
        println!("Use --force to confirm deletion.");
        return Ok(());
    }

    // Constitution check before deletion (F08)
    let constitution = Constitution::new();
    let ctx = OperationContext {
        operation: Operation::Delete,
        pattern_id: Some(args.id.clone()),
        reward: None,
        tier: None,
        surprise_score: None,
        has_recent_backup: false,
        failure_mode: None,
        domain: None,
    };
    if !constitution.is_allowed(&ctx) {
        let msg = format!("Constitution blocked deletion of pattern: {}", args.id);
        if args.json {
            let output = DeleteOutput {
                id: args.id.clone(),
                success: false,
                message: msg,
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            eprintln!("{}", msg);
        }
        return Ok(());
    }

    // Delete from database
    let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;
    let pattern_id = PatternId::from(args.id.as_str());
    storage.delete_pattern(&pattern_id).await?;

    // Emit PatternDeleted event (F16)
    let event_bus = EventBus::new();
    event_bus.publish_sync(NagualEvent::pattern_deleted(&args.id));

    // Allow background PostgreSQL write to complete before CLI exits
    if resolve_postgres_url(args.postgres_url.as_deref()).is_some() {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    tracing::info!(
        pattern_id = %args.id,
        db_path = %args.db_path.display(),
        "Pattern deleted"
    );

    if args.json {
        let output = DeleteOutput {
            id: args.id.clone(),
            success: true,
            message: format!("Knowledge item deleted from {}", args.db_path.display()),
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nKnowledge item deleted: {}", args.id);
        println!("Database: {}\n", args.db_path.display());
    }

    Ok(())
}

/// Run the list command.
async fn run_list(args: &ListArgs) -> Result<()> {
    tracing::info!("Listing knowledge items");

    let patterns = if args.demo {
        create_demo_patterns()
    } else {
        // List from database
        let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;

        // Get patterns based on sort criteria
        let all_patterns = match args.sort.as_str() {
            "effectiveness" => storage.get_top_effective(args.limit + args.offset).await?,
            _ => storage.get_recent(args.limit + args.offset).await?,
        };

        if all_patterns.is_empty() && !args.json {
            println!("\nNo patterns found in database at: {}", args.db_path.display());
            println!("Use 'nagual knowledge store' to add patterns, or --demo for sample data.\n");
        }

        all_patterns
    };

    // Filter by domain
    let mut results: Vec<_> = patterns
        .into_iter()
        .filter(|p| {
            if let Some(ref domain) = args.domain {
                p.category().to_string().starts_with(domain)
            } else {
                true
            }
        })
        .collect();

    // Sort
    match args.sort.as_str() {
        "created" => results.sort_by(|a, b| {
            if args.order == "asc" {
                a.timestamp().cmp(&b.timestamp())
            } else {
                b.timestamp().cmp(&a.timestamp())
            }
        }),
        "reward" => results.sort_by(|a, b| {
            if args.order == "asc" {
                a.reward().partial_cmp(&b.reward()).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.reward().partial_cmp(&a.reward()).unwrap_or(std::cmp::Ordering::Equal)
            }
        }),
        "effectiveness" => results.sort_by(|a, b| {
            if args.order == "asc" {
                a.effectiveness().partial_cmp(&b.effectiveness()).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.effectiveness().partial_cmp(&a.effectiveness()).unwrap_or(std::cmp::Ordering::Equal)
            }
        }),
        "usage" => results.sort_by(|a, b| {
            if args.order == "asc" {
                a.reuse_count().cmp(&b.reuse_count())
            } else {
                b.reuse_count().cmp(&a.reuse_count())
            }
        }),
        _ => results.sort_by(|a, b| {
            if args.order == "asc" {
                a.updated_at().cmp(&b.updated_at())
            } else {
                b.updated_at().cmp(&a.updated_at())
            }
        }),
    }

    // Paginate
    let results: Vec<_> = results.into_iter().skip(args.offset).take(args.limit).collect();

    if args.json {
        let output: Vec<KnowledgeListItem> = results
            .iter()
            .map(|p| KnowledgeListItem {
                id: p.id().to_string(),
                problem: truncate(p.problem(), 80),
                domain: p.category().to_string(),
                reward: p.reward(),
                updated_at: p.updated_at().to_rfc3339(),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nKnowledge Items");
        println!("{:=<90}", "");
        println!(
            "{:<36}  {:<30}  {:>8}  {:>10}",
            "ID", "Problem", "Reward", "Domain"
        );
        println!("{:-<90}", "");

        for pattern in &results {
            println!(
                "{:<36}  {:<30}  {:>8.3}  {:>10}",
                truncate(&pattern.id().to_string(), 36),
                truncate(pattern.problem(), 30),
                pattern.reward(),
                truncate(&pattern.category().to_string(), 10)
            );
        }

        println!("{:-<90}", "");
        println!("Showing {} items\n", results.len());
    }

    Ok(())
}

/// Run the sync command - copy all patterns from SQLite to PostgreSQL.
async fn run_sync(args: &SyncArgs) -> Result<()> {
    tracing::info!("Starting SQLite → PostgreSQL sync");

    // Resolve PostgreSQL URL
    let pg_url = resolve_postgres_url(args.postgres_url.as_deref());
    let pg_url = match pg_url {
        Some(url) => url,
        None => {
            eprintln!("Error: No PostgreSQL URL provided.");
            eprintln!("Use --postgres-url, DATABASE_URL env, or set postgres_url in ~/.nagual/config.toml");
            return Ok(());
        }
    };

    println!("\nSQLite → PostgreSQL Sync");
    println!("{:=<60}", "");
    println!("Source: {}", args.db_path.display());
    println!("Target: {}",
        if let Some(at_pos) = pg_url.find('@') {
            if let Some(colon_pos) = pg_url[..at_pos].rfind(':') {
                format!("{}****{}", &pg_url[..colon_pos + 1], &pg_url[at_pos..])
            } else { pg_url.clone() }
        } else { pg_url.clone() }
    );
    println!("{:-<60}", "");

    // Open SQLite (read-only source)
    let sqlite = Arc::new(SqliteDb::open(&args.db_path)?);
    let sqlite_config = DualWriteConfig {
        dlq_path: args.db_path.with_extension("dlq.db").to_string_lossy().to_string(),
        ..Default::default()
    };
    let sqlite_adapter = Arc::new(DualWriteAdapter::new(sqlite, None, sqlite_config)?);
    let sqlite_storage = PatternStorage::new(sqlite_adapter, StorageConfig::default()).await?;

    // Read all patterns from SQLite
    let patterns = sqlite_storage.get_recent(100_000).await?;
    let total = patterns.len();
    println!("Found {} patterns in SQLite\n", total);

    if total == 0 {
        println!("Nothing to sync.");
        return Ok(());
    }

    // Connect to PostgreSQL
    let pg = match PostgresDb::connect(&pg_url, 5).await {
        Ok(pg) => pg,
        Err(e) => {
            eprintln!("Error: Failed to connect to PostgreSQL: {}", e);
            return Ok(());
        }
    };
    let pool = pg.pool();

    let mut synced = 0usize;
    let mut errors = 0usize;

    for (i, pattern) in patterns.iter().enumerate() {
        let id = pattern.id().to_string();
        let timestamp = pattern.timestamp();
        let updated_at = pattern.updated_at();
        let category = pattern.category().to_string();
        let problem = pattern.problem();
        let solution = pattern.solution();
        let context = pattern.context();
        let effectiveness = pattern.effectiveness() as f64;
        let reuse_count = pattern.reuse_count() as i32;
        let reward = pattern.reward() as f64;
        let success = pattern.success();
        let critique = pattern.critique();
        let confidence = pattern.confidence() as f64;
        let embedding: Option<Vec<f32>> = pattern.embedding().map(|e| e.to_vec());
        let tags = serde_json::json!(pattern.tags());
        let related: serde_json::Value = serde_json::json!([]);
        let metadata: serde_json::Value = serde_json::json!({});

        let result = sqlx::query(
            r#"INSERT INTO reasoning_patterns (
                id, timestamp, updated_at, category, problem, solution, context,
                effectiveness, reuse_count, reward, success, critique,
                agent_id, session_id, confidence, embedding, tags,
                related_patterns, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            ON CONFLICT (id) DO UPDATE SET
                updated_at = EXCLUDED.updated_at,
                category = EXCLUDED.category,
                problem = EXCLUDED.problem,
                solution = EXCLUDED.solution,
                context = EXCLUDED.context,
                effectiveness = EXCLUDED.effectiveness,
                reuse_count = EXCLUDED.reuse_count,
                reward = EXCLUDED.reward,
                success = EXCLUDED.success,
                critique = EXCLUDED.critique,
                confidence = EXCLUDED.confidence,
                embedding = EXCLUDED.embedding,
                tags = EXCLUDED.tags,
                related_patterns = EXCLUDED.related_patterns,
                metadata = EXCLUDED.metadata"#,
        )
        .bind(&id)
        .bind(timestamp)
        .bind(updated_at)
        .bind(&category)
        .bind(problem)
        .bind(solution)
        .bind(context)
        .bind(effectiveness)
        .bind(reuse_count)
        .bind(reward)
        .bind(success)
        .bind(critique)
        .bind(None::<String>) // agent_id
        .bind(None::<String>) // session_id
        .bind(confidence)
        .bind(&embedding)
        .bind(&tags)
        .bind(&related)
        .bind(&metadata)
        .execute(pool)
        .await;

        match result {
            Ok(_) => {
                synced += 1;
            }
            Err(e) => {
                errors += 1;
                if errors <= 5 {
                    eprintln!("  ERR [{}]: {}", id, e);
                }
            }
        }

        // Progress every batch_size
        if (i + 1) % args.batch_size == 0 {
            println!("  Progress: {}/{} ({} errors)", i + 1, total, errors);
        }
    }

    println!("\n{:=<60}", "");
    println!("Sync Complete");
    println!("  Total:  {}", total);
    println!("  Synced: {}", synced);
    println!("  Errors: {}", errors);
    println!("{:=<60}\n", "");

    Ok(())
}

// ─── import ────────────────────────────────────────────────────────

/// One record in a seed JSONL file. Matches the schema produced by
/// `scripts/export-seed.py`.
#[derive(serde::Deserialize, Debug)]
struct SeedRecord {
    problem: String,
    solution: String,
    domain: String,
    #[serde(default)]
    context: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "default_seed_reward")]
    reward: f32,
    // Accepted but currently ignored — tier is derived from reward + reuse count
    // at promotion time. Kept here for forward-compat with future exporters.
    #[serde(default, rename = "tier")]
    _tier: Option<String>,
}

fn default_seed_reward() -> f32 {
    0.5
}

/// Run the import command.
async fn run_import(args: &ImportArgs) -> Result<()> {
    use std::io::BufRead;

    // ── Parse JSONL ──────────────────────────────────────────────
    let file = std::fs::File::open(&args.seed).map_err(|e| NagualError::Config {
        message: format!("Cannot open seed file {}: {}", args.seed.display(), e),
    })?;
    let reader = std::io::BufReader::new(file);

    let mut records: Vec<(usize, SeedRecord)> = Vec::new();
    let mut parse_errors: Vec<(usize, String)> = Vec::new();

    for (i, line_res) in reader.lines().enumerate() {
        let line_no = i + 1;
        let line = line_res.map_err(|e| NagualError::Config {
            message: format!("Error reading line {}: {}", line_no, e),
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        match serde_json::from_str::<SeedRecord>(trimmed) {
            Ok(r) => records.push((line_no, r)),
            Err(e) => parse_errors.push((line_no, e.to_string())),
        }
    }

    let total_records = records.len();
    let parse_error_count = parse_errors.len();

    if !args.json {
        println!("\nParsed {} records from {}", total_records, args.seed.display());
        if parse_error_count > 0 {
            eprintln!("\nParse errors ({}):", parse_error_count);
            for (line, err) in parse_errors.iter().take(10) {
                eprintln!("  line {}: {}", line, err);
            }
            if parse_error_count > 10 {
                eprintln!("  ... and {} more", parse_error_count - 10);
            }
        }
    }

    // ── Dry run: stop here ───────────────────────────────────────
    if args.dry_run {
        if args.json {
            let output = serde_json::json!({
                "dry_run": true,
                "total_records": total_records,
                "parse_errors": parse_error_count,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("\nDry run — no changes written.");
            println!("  Would import: {} patterns", total_records);
            if let Some(ref p) = args.domain_prefix {
                println!("  Domain prefix: {:?}", p);
            }
        }
        return Ok(());
    }

    // ── Load existing content hashes for idempotent import ──────
    let storage = init_storage(&args.db_path, args.postgres_url.as_deref()).await?;
    let existing_hashes: std::collections::HashSet<String> = {
        let sql = "SELECT content_hash FROM reasoning_patterns \
                   WHERE content_hash IS NOT NULL AND content_hash != ''";
        let rows: Vec<String> = storage
            .adapter()
            .sqlite()
            .query(sql, &[], |row| row.get::<_, String>(0))
            .await
            .unwrap_or_default();
        rows.into_iter().collect()
    };

    // ── Import ───────────────────────────────────────────────────
    let mut imported = 0;
    let mut skipped_duplicate = 0;
    let mut failed = 0;
    let mut seen_this_run: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (line_no, record) in records {
        let domain = match &args.domain_prefix {
            Some(p) => format!("{}{}", p, record.domain),
            None => record.domain.clone(),
        };

        let mut pattern = Pattern::builder()
            .problem(record.problem.clone())
            .solution(record.solution.clone())
            .category(PatternCategory::from(domain.as_str()))
            .context(record.context.clone())
            .reward(record.reward)
            .tags(record.tags.clone())
            .metadata(PatternMetadata::new().with_source("seed"))
            .build();

        pattern.compute_content_hash();

        if let Some(hash) = pattern.content_hash() {
            if existing_hashes.contains(hash) || !seen_this_run.insert(hash.to_string()) {
                skipped_duplicate += 1;
                continue;
            }
        }

        match storage.store_pattern(&pattern).await {
            Ok(_) => imported += 1,
            Err(e) => {
                tracing::warn!(line = line_no, error = %e, "Failed to import pattern");
                if !args.json {
                    eprintln!("  line {}: store error — {}", line_no, e);
                }
                failed += 1;
            }
        }
    }

    // Allow background PostgreSQL writes to complete before CLI exits
    if resolve_postgres_url(args.postgres_url.as_deref()).is_some() {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    // ── Report ───────────────────────────────────────────────────
    if args.json {
        let output = serde_json::json!({
            "total_records": total_records,
            "imported": imported,
            "skipped_duplicate": skipped_duplicate,
            "failed": failed,
            "parse_errors": parse_error_count,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nImport Summary");
        println!("{:-<60}", "");
        println!("  Total records:    {}", total_records);
        println!("  Imported:         {}", imported);
        println!("  Skipped (dup):    {}", skipped_duplicate);
        if failed > 0 {
            println!("  Failed:           {}", failed);
        }
        if parse_error_count > 0 {
            println!("  Parse errors:     {}", parse_error_count);
        }
        println!("  Database:         {}", args.db_path.display());
        println!("{:-<60}\n", "");
    }

    Ok(())
}

// Output structures for JSON

#[derive(Serialize)]
struct StoreOutput {
    id: String,
    domain: String,
    tags: Vec<String>,
    success: bool,
    message: String,
}

#[derive(Serialize)]
struct SearchResultItem {
    id: String,
    problem: String,
    solution: String,
    domain: String,
    reward: f32,
    effectiveness: f32,
    tags: Vec<String>,
}

#[derive(Serialize)]
struct KnowledgeItem {
    id: String,
    problem: String,
    solution: String,
    context: String,
    domain: String,
    effectiveness: f32,
    reward: f32,
    confidence: f32,
    reuse_count: u32,
    tags: Vec<String>,
    created_at: String,
    updated_at: String,
}

#[derive(Serialize)]
struct KnowledgeListItem {
    id: String,
    problem: String,
    domain: String,
    reward: f32,
    updated_at: String,
}

#[derive(Serialize)]
struct DeleteOutput {
    id: String,
    success: bool,
    message: String,
}

#[derive(Serialize)]
struct ErrorOutput {
    error: String,
    error_type: String,
}

/// Truncate a string to a maximum length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Create demo patterns for testing.
fn create_demo_patterns() -> Vec<Pattern> {
    vec![
        Pattern::builder()
            .id("demo-001")
            .problem("How to handle async errors in Rust")
            .solution("Use Result type with async/await and proper error propagation using ? operator")
            .category(PatternCategory::Resilience)
            .effectiveness(0.92)
            .reward(0.88)
            .confidence(0.90)
            .reuse_count(15)
            .tag("rust")
            .tag("async")
            .tag("error-handling")
            .build(),
        Pattern::builder()
            .id("demo-002")
            .problem("Database connection pooling best practices")
            .solution("Use sqlx pool with proper configuration, health checks, and connection limits")
            .category(PatternCategory::Performance)
            .effectiveness(0.85)
            .reward(0.82)
            .confidence(0.88)
            .reuse_count(12)
            .tag("database")
            .tag("pooling")
            .tag("performance")
            .build(),
        Pattern::builder()
            .id("demo-003")
            .problem("API rate limiting implementation")
            .solution("Implement token bucket algorithm with Redis backend for distributed rate limiting")
            .category(PatternCategory::ApiDesign)
            .effectiveness(0.78)
            .reward(0.75)
            .confidence(0.80)
            .reuse_count(8)
            .tag("api")
            .tag("rate-limiting")
            .tag("redis")
            .build(),
        Pattern::builder()
            .id("demo-004")
            .problem("Memory leak detection in long-running services")
            .solution("Use valgrind, memory profiling with periodic snapshots, and heap analysis")
            .category(PatternCategory::Performance)
            .effectiveness(0.65)
            .reward(0.60)
            .confidence(0.70)
            .reuse_count(5)
            .tag("memory")
            .tag("debugging")
            .tag("profiling")
            .build(),
        Pattern::builder()
            .id("demo-005")
            .problem("Testing async code patterns")
            .solution("Use tokio-test with proper timeout handling and mock async runtimes")
            .category(PatternCategory::Testing)
            .effectiveness(0.88)
            .reward(0.85)
            .confidence(0.90)
            .reuse_count(20)
            .tag("testing")
            .tag("async")
            .tag("tokio")
            .build(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // Helper struct for testing CLI parsing
    #[derive(Parser, Debug)]
    struct TestCli {
        #[command(subcommand)]
        command: TestCommand,
    }

    #[derive(Subcommand, Debug)]
    enum TestCommand {
        Knowledge(KnowledgeCommand),
    }

    #[test]
    fn test_cli_parse_store() {
        let args = vec![
            "test",
            "knowledge",
            "store",
            "Test content",
            "--domain",
            "rust.async",
            "--tags",
            "tag1,tag2",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_search() {
        let args = vec![
            "test",
            "knowledge",
            "search",
            "async error",
            "--domain",
            "rust",
            "--limit",
            "5",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_get() {
        let args = vec!["test", "knowledge", "get", "abc123", "--json"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_delete() {
        let args = vec!["test", "knowledge", "delete", "abc123", "--force"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_list() {
        let args = vec![
            "test",
            "knowledge",
            "list",
            "--domain",
            "rust",
            "--sort",
            "reward",
            "--limit",
            "10",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("Hello", 10), "Hello");
        assert_eq!(truncate("Hello World", 8), "Hello...");
    }

    #[test]
    fn test_demo_patterns() {
        let patterns = create_demo_patterns();
        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.id().as_str() == "demo-001"));
    }

    #[test]
    fn test_parse_unit_interval_valid() {
        assert_eq!(parse_unit_interval("0.0").unwrap(), 0.0);
        assert_eq!(parse_unit_interval("0.5").unwrap(), 0.5);
        assert_eq!(parse_unit_interval("1.0").unwrap(), 1.0);
        assert_eq!(parse_unit_interval("0.75").unwrap(), 0.75);
    }

    #[test]
    fn test_parse_unit_interval_invalid() {
        // Out of range
        assert!(parse_unit_interval("-0.1").is_err());
        assert!(parse_unit_interval("1.1").is_err());
        assert!(parse_unit_interval("2.0").is_err());
        assert!(parse_unit_interval("-1.0").is_err());

        // Invalid format
        assert!(parse_unit_interval("abc").is_err());
        assert!(parse_unit_interval("").is_err());
    }

    #[test]
    fn test_cli_parse_store_with_effectiveness_validation() {
        // Valid effectiveness value
        let args_valid = vec![
            "test",
            "knowledge",
            "store",
            "Test content",
            "--effectiveness",
            "0.8",
        ];
        let cli_valid = TestCli::try_parse_from(args_valid);
        assert!(cli_valid.is_ok());

        // Invalid effectiveness value (out of range)
        let args_invalid = vec![
            "test",
            "knowledge",
            "store",
            "Test content",
            "--effectiveness",
            "1.5",
        ];
        let cli_invalid = TestCli::try_parse_from(args_invalid);
        assert!(cli_invalid.is_err());
    }
}
