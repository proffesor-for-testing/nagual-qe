//! Pattern management CLI commands.
//!
//! Provides commands for storing, searching, viewing statistics,
//! and consolidating patterns in the ReasoningBank with persistent SQLite storage.
//!
//! # Usage Examples
//!
//! ```bash
//! # Store a pattern (interactive or JSON input)
//! nagual patterns store --interactive
//! nagual patterns store --json '{"problem": "...", "solution": "..."}'
//!
//! # Search patterns
//! nagual patterns search "async error handling" --domain rust
//!
//! # Show statistics
//! nagual patterns stats --domain rust --detailed
//!
//! # Run consolidation
//! nagual patterns consolidate --similarity 0.9 --dry-run
//! ```

use std::path::PathBuf;

use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};

use super::common::init_storage_sqlite_only;
use crate::constitution::{Constitution, Operation, OperationContext};
use crate::error::Result;
use crate::events::{EventBus, NagualEvent};
use crate::learning::{consolidate_patterns, PatternConsolidationConfig};
use crate::ml::to_array1;
#[cfg(feature = "onnx-embed")]
use crate::ml::{Embedder, EmbedderConfig};
#[cfg(not(feature = "onnx-embed"))]
use crate::ml::HashEmbedder;
use crate::ml::LoraStorage;
use crate::reasoning_bank::dna::PatternDNA;
use crate::reasoning_bank::pattern::{Pattern, PatternCategory};
use crate::reasoning_bank::pyramid;
use crate::reasoning_bank::{
    self as rb, retrieve_patterns_hyperbolic, staged_retrieve_patterns, HyperbolicRetrievalConfig,
    PatternQuery, RetrievalConfig, RetrievalStaging,
};
use crate::reasoning_bank::PatternTier;

/// Pattern management commands.
///
/// Store, search, analyze, and consolidate patterns in the ReasoningBank.
#[derive(Args, Debug)]
pub struct PatternsCommand {
    #[command(subcommand)]
    pub subcommand: PatternsSubcommand,
}

/// Patterns subcommands.
#[derive(Subcommand, Debug)]
pub enum PatternsSubcommand {
    /// Store a new pattern.
    ///
    /// Creates a new pattern in the ReasoningBank. Can be used
    /// interactively or with JSON input.
    Store(StorePatternArgs),

    /// Search patterns by query.
    ///
    /// Performs similarity search across patterns, supporting
    /// domain filtering and relevance scoring.
    Search(SearchPatternArgs),

    /// Show pattern statistics.
    ///
    /// Displays comprehensive statistics about stored patterns
    /// including domain breakdown, reward distribution, and trends.
    Stats(StatsArgs),

    /// Run pattern consolidation.
    ///
    /// Merges similar patterns, archives low performers, and
    /// cleans up stale entries.
    Consolidate(ConsolidatePatternArgs),

    /// Analyze pattern quality.
    ///
    /// Evaluates pattern quality and generates improvement recommendations.
    Analyze(AnalyzeArgs),

    /// Export patterns to a JSON file.
    ///
    /// Exports patterns from the database to a portable JSON format,
    /// with optional filtering by domain, reward, and tier.
    Export(ExportPatternArgs),

    /// Import patterns from a JSON file.
    ///
    /// Imports patterns from a previously exported JSON file,
    /// with deduplication to avoid storing duplicates.
    Import(ImportPatternArgs),

    /// Manage pyramid summaries (title + summary).
    ///
    /// Generate or view statistics for pyramid summaries, which provide
    /// a hierarchical view: title (10 words) -> summary (50 words) -> full content.
    Pyramid(PyramidArgs),
}

/// Arguments for the store subcommand.
#[derive(Args, Debug)]
pub struct StorePatternArgs {
    /// Pattern as JSON string.
    #[arg(long, conflicts_with = "interactive")]
    pub json_input: Option<String>,

    /// Interactive mode for entering pattern details.
    #[arg(short, long)]
    pub interactive: bool,

    /// Problem description (for non-interactive mode).
    #[arg(long)]
    pub problem: Option<String>,

    /// Solution description (for non-interactive mode).
    #[arg(long)]
    pub solution: Option<String>,

    /// Pattern category/domain.
    #[arg(long)]
    pub category: Option<String>,

    /// Context information.
    #[arg(long)]
    pub context: Option<String>,

    /// Tags (comma-separated).
    #[arg(long, value_delimiter = ',')]
    pub tags: Vec<String>,

    /// Initial effectiveness score (0.0-1.0).
    #[arg(long, default_value = "0.5")]
    pub effectiveness: f32,

    /// Initial confidence score (0.0-1.0).
    #[arg(long, default_value = "0.5")]
    pub confidence: f32,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub output_json: bool,

    /// Use demo mode.
    #[arg(long)]
    pub demo: bool,
}

/// Arguments for the search subcommand.
#[derive(Args, Debug)]
pub struct SearchPatternArgs {
    /// Search query.
    #[arg(value_name = "QUERY")]
    pub query: String,

    /// Filter by domain.
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Filter by category.
    #[arg(long)]
    pub category: Option<String>,

    /// Filter by tags (comma-separated).
    #[arg(short, long, value_delimiter = ',')]
    pub tags: Vec<String>,

    /// Minimum reward threshold.
    #[arg(long)]
    pub min_reward: Option<f32>,

    /// Minimum effectiveness threshold.
    #[arg(long)]
    pub min_effectiveness: Option<f32>,

    /// Maximum number of results.
    #[arg(short, long, default_value = "10")]
    pub limit: usize,

    /// Use MMR (Maximal Marginal Relevance) for diverse results.
    #[arg(long)]
    pub mmr: bool,

    /// MMR diversity weight (0.0-1.0).
    #[arg(long, default_value = "0.3")]
    pub mmr_lambda: f32,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show verbose output.
    #[arg(short, long)]
    pub verbose: bool,

    /// Show DNA barcode visualization next to each result.
    #[arg(long)]
    pub dna: bool,

    /// Use hyperbolic (Poincare ball) distance for hierarchy-aware retrieval.
    #[arg(long)]
    pub hyperbolic: bool,

    /// Use demo mode.
    #[arg(long)]
    pub demo: bool,
}

/// Arguments for the stats subcommand.
#[derive(Args, Debug)]
pub struct StatsArgs {
    /// Filter by domain.
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Show detailed statistics.
    #[arg(short = 'D', long)]
    pub detailed: bool,

    /// Include domain hierarchy breakdown.
    #[arg(long)]
    pub hierarchy: bool,

    /// Number of top patterns to show.
    #[arg(long, default_value = "10")]
    pub top: usize,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Use demo mode.
    #[arg(long)]
    pub demo: bool,
}

/// Arguments for the consolidate subcommand.
#[derive(Args, Debug)]
pub struct ConsolidatePatternArgs {
    /// Similarity threshold for merging (0.0-1.0).
    #[arg(short, long, default_value = "0.9")]
    pub similarity: f32,

    /// Auto-archive patterns below this reward.
    #[arg(long)]
    pub archive_threshold: Option<f32>,

    /// Minimum age in days for archiving.
    #[arg(long, default_value = "30")]
    pub min_age_days: i64,

    /// Dry-run mode (show what would happen).
    #[arg(long)]
    pub dry_run: bool,

    /// Target domain for consolidation.
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Show verbose output.
    #[arg(short, long)]
    pub verbose: bool,
}

/// Arguments for the analyze subcommand.
#[derive(Args, Debug)]
pub struct AnalyzeArgs {
    /// Target domain for analysis.
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Include improvement recommendations.
    #[arg(long)]
    pub recommendations: bool,

    /// Maximum recommendations to generate.
    #[arg(long, default_value = "10")]
    pub max_recommendations: usize,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,

    /// Use demo mode.
    #[arg(long)]
    pub demo: bool,
}

/// Arguments for the export subcommand.
#[derive(Args, Debug)]
pub struct ExportPatternArgs {
    /// Output file path.
    #[arg(short, long, default_value = "patterns-export.json")]
    pub output: PathBuf,

    /// Filter by domain.
    #[arg(short, long)]
    pub domain: Option<String>,

    /// Minimum reward threshold.
    #[arg(long)]
    pub min_reward: Option<f32>,

    /// Filter by tier.
    #[arg(long)]
    pub tier: Option<String>,

    /// Maximum patterns to export.
    #[arg(short, long)]
    pub limit: Option<usize>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the import subcommand.
#[derive(Args, Debug)]
pub struct ImportPatternArgs {
    /// Input file path.
    #[arg(value_name = "FILE")]
    pub input: PathBuf,

    /// Dry-run mode (show what would be imported).
    #[arg(long)]
    pub dry_run: bool,

    /// Skip duplicate detection.
    #[arg(long)]
    pub force: bool,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the pyramid subcommand.
#[derive(Args, Debug)]
pub struct PyramidArgs {
    /// Generate pyramids for patterns missing them.
    #[arg(long)]
    pub generate: bool,

    /// Preview without making changes.
    #[arg(long)]
    pub dry_run: bool,

    /// Show statistics only.
    #[arg(long)]
    pub stats: bool,

    /// Limit number of patterns to process.
    #[arg(long)]
    pub limit: Option<usize>,

    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON.
    #[arg(long)]
    pub json: bool,
}

impl PatternsCommand {
    /// Execute the patterns command.
    pub async fn run(&self) -> Result<()> {
        match &self.subcommand {
            PatternsSubcommand::Store(args) => run_store_pattern(args).await,
            PatternsSubcommand::Search(args) => run_search_pattern(args).await,
            PatternsSubcommand::Stats(args) => run_stats(args).await,
            PatternsSubcommand::Consolidate(args) => run_consolidate_pattern(args).await,
            PatternsSubcommand::Analyze(args) => run_analyze(args).await,
            PatternsSubcommand::Export(args) => run_export_patterns(args).await,
            PatternsSubcommand::Import(args) => run_import_patterns(args).await,
            PatternsSubcommand::Pyramid(args) => run_pyramid(args).await,
        }
    }
}

/// Run the store pattern command.
async fn run_store_pattern(args: &StorePatternArgs) -> Result<()> {
    tracing::info!("Storing new pattern");

    // Build pattern from arguments
    let pattern = if let Some(ref json_str) = args.json_input {
        // Parse from JSON
        match serde_json::from_str::<PatternInput>(json_str) {
            Ok(input) => Pattern::builder()
                .problem(input.problem)
                .solution(input.solution)
                .category(PatternCategory::from(input.category.unwrap_or_else(|| "general".to_string()).as_str()))
                .context(input.context.unwrap_or_default())
                .effectiveness(input.effectiveness.unwrap_or(0.5))
                .confidence(input.confidence.unwrap_or(0.5))
                .tags(input.tags.unwrap_or_default())
                .build(),
            Err(e) => {
                if args.output_json {
                    let output = ErrorOutput {
                        error: format!("Invalid JSON input: {}", e),
                        error_type: "parse_error".to_string(),
                    };
                    println!("{}", serde_json::to_string_pretty(&output)?);
                } else {
                    eprintln!("Error: Invalid JSON input: {}", e);
                }
                return Ok(());
            }
        }
    } else if args.interactive {
        // Interactive mode - show prompt
        println!("Interactive pattern entry not yet implemented.");
        println!("Please use --json-input or --problem/--solution flags.");
        return Ok(());
    } else if let (Some(problem), Some(solution)) = (&args.problem, &args.solution) {
        // Build from individual arguments
        Pattern::builder()
            .problem(problem)
            .solution(solution)
            .category(PatternCategory::from(args.category.as_deref().unwrap_or("general")))
            .context(args.context.as_deref().unwrap_or(""))
            .effectiveness(args.effectiveness)
            .confidence(args.confidence)
            .tags(args.tags.clone())
            .build()
    } else {
        let msg = "Either --json-input, --interactive, or both --problem and --solution are required";
        if args.output_json {
            let output = ErrorOutput {
                error: msg.to_string(),
                error_type: "missing_input".to_string(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            eprintln!("Error: {}", msg);
        }
        return Ok(());
    };

    if args.demo {
        let id = pattern.id().to_string();
        if args.output_json {
            let output = StoreOutput {
                id: id.clone(),
                category: pattern.category().to_string(),
                success: true,
                message: "Pattern stored successfully (demo mode)".to_string(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("\nPattern Stored (demo mode)");
            println!("{:-<60}", "");
            println!("ID: {}", id);
            println!("Category: {}", pattern.category());
            println!("Problem: {}", truncate(pattern.problem(), 60));
            println!("Solution: {}", truncate(pattern.solution(), 60));
            println!("{:-<60}\n", "");
        }
    } else {
        let storage = init_storage_sqlite_only(&args.db_path).await?;
        let stored_id = storage.store_pattern(&pattern).await?;

        // Emit PatternStored event (F16)
        let event_bus = EventBus::new();
        event_bus.publish_sync(NagualEvent::pattern_stored(
            &stored_id.to_string(),
            &pattern.category().to_string(),
        ));

        if args.output_json {
            let output = StoreOutput {
                id: stored_id.to_string(),
                category: pattern.category().to_string(),
                success: true,
                message: format!("Pattern stored to {}", args.db_path.display()),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("\nPattern Stored Successfully");
            println!("{:-<60}", "");
            println!("ID: {}", stored_id);
            println!("Category: {}", pattern.category());
            println!("Database: {}", args.db_path.display());
            println!("Problem: {}", truncate(pattern.problem(), 60));
            println!("Solution: {}", truncate(pattern.solution(), 60));
            println!("{:-<60}\n", "");
        }
    }

    Ok(())
}

/// Run the search pattern command.
async fn run_search_pattern(args: &SearchPatternArgs) -> Result<()> {
    tracing::info!("Searching patterns: {}", args.query);

    let patterns = if args.demo {
        create_demo_patterns()
    } else {
        // Search in SQLite database
        let storage = init_storage_sqlite_only(&args.db_path).await?;
        let all_patterns = storage.get_recent(args.limit * 10).await?;

        if all_patterns.is_empty() && !args.json {
            println!("\nNo patterns found in database at: {}", args.db_path.display());
            println!("Use 'nagual patterns store' to add patterns, or --demo for sample data.\n");
        }

        all_patterns
    };

    // Hyperbolic retrieval (F09) or text matching
    #[allow(unused_mut)]
    let mut results: Vec<_> = if args.hyperbolic && !args.demo {
        #[cfg(feature = "onnx-embed")]
        {
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
                        );

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
                                Vec::new()
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
        Vec::new()
    };

    // Text matching fallback
    if results.is_empty() {
        let query_lower = args.query.to_lowercase();
        results = patterns
            .into_iter()
            .filter(|p| {
                p.problem().to_lowercase().contains(&query_lower)
                    || p.solution().to_lowercase().contains(&query_lower)
                    || p.context().to_lowercase().contains(&query_lower)
            })
            .filter(|p| {
                if let Some(ref domain) = args.domain {
                    p.category().to_string().contains(domain)
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
            .filter(|p| {
                if let Some(min_eff) = args.min_effectiveness {
                    p.effectiveness() >= min_eff
                } else {
                    true
                }
            })
            .take(args.limit)
            .collect();

        // Sort by quality score
        results.sort_by(|a, b| {
            b.quality_score()
                .partial_cmp(&a.quality_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    if args.json {
        let output: Vec<SearchResult> = results
            .iter()
            .enumerate()
            .map(|(i, p)| SearchResult {
                rank: i + 1,
                id: p.id().to_string(),
                problem: p.problem().to_string(),
                solution: p.solution().to_string(),
                category: p.category().to_string(),
                quality_score: p.quality_score(),
                effectiveness: p.effectiveness(),
                reward: p.reward(),
                reuse_count: p.reuse_count(),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        println!("\nPattern Search Results: \"{}\"", args.query);
        println!("{:=<80}", "");

        if results.is_empty() {
            println!("\nNo patterns found matching your query.");
        } else {
            for (i, pattern) in results.iter().enumerate() {
                let dna_str = if args.dna {
                    let age_days = pattern.age_seconds() / 86400;
                    let domain_str = pattern.category().to_string();
                    let compact = PatternDNA::to_compact(
                        &domain_str,
                        pattern.reward(),
                        age_days,
                        pattern.reuse_count(),
                        pattern.surprise_score(),
                        PatternTier::Booster,
                    );
                    format!(" {}", compact)
                } else {
                    String::new()
                };
                println!(
                    "\n{}. [Score: {:.3}]{} {}",
                    i + 1,
                    pattern.quality_score(),
                    dna_str,
                    truncate(pattern.problem(), 60)
                );
                println!("   ID: {}", pattern.id());
                println!("   Category: {} | Effectiveness: {:.2} | Reward: {:.2} | Quality: {:.3}",
                    pattern.category(),
                    pattern.effectiveness(),
                    pattern.reward(),
                    pattern.bayesian_score().mean()
                );
                if args.verbose {
                    println!("   Solution: {}", truncate(pattern.solution(), 70));
                    if !pattern.context().is_empty() {
                        println!("   Context: {}", truncate(pattern.context(), 70));
                    }
                }
            }
        }

        println!("\n{:=<80}", "");
        println!("Found {} patterns\n", results.len());
    }

    Ok(())
}

/// Run the stats command.
async fn run_stats(args: &StatsArgs) -> Result<()> {
    tracing::info!("Generating pattern statistics");

    let patterns = if args.demo {
        create_demo_patterns()
    } else {
        // Retrieve from SQLite database
        let storage = init_storage_sqlite_only(&args.db_path).await?;
        let all_patterns = storage.get_recent(10000).await?; // Get all patterns for stats

        if all_patterns.is_empty() && !args.json {
            println!("\nNo patterns found in database at: {}", args.db_path.display());
            println!("Use 'nagual patterns store' to add patterns, or --demo for sample data.\n");
        }

        all_patterns
    };

    // Calculate stats
    let total = patterns.len();
    let avg_reward: f32 = if total > 0 {
        patterns.iter().map(|p| p.reward()).sum::<f32>() / total as f32
    } else {
        0.0
    };
    let avg_effectiveness: f32 = if total > 0 {
        patterns.iter().map(|p| p.effectiveness()).sum::<f32>() / total as f32
    } else {
        0.0
    };
    let total_reuse: u32 = patterns.iter().map(|p| p.reuse_count()).sum();
    let with_embeddings = patterns.iter().filter(|p| p.has_embedding()).count();

    // Category breakdown
    let mut category_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for p in &patterns {
        *category_counts.entry(p.category().to_string()).or_default() += 1;
    }

    let stats = PatternStatistics {
        total_patterns: total,
        average_reward: avg_reward,
        average_effectiveness: avg_effectiveness,
        total_reuse_count: total_reuse,
        patterns_with_embeddings: with_embeddings,
        categories: category_counts.into_iter().map(|(k, v)| CategoryStats {
            name: k,
            count: v,
        }).collect(),
        top_patterns: patterns.iter()
            .take(args.top)
            .map(|p| TopPatternInfo {
                id: p.id().to_string(),
                problem: truncate(p.problem(), 50),
                quality_score: p.quality_score(),
            })
            .collect(),
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        println!("\nPattern Statistics");
        println!("{:=<60}", "");

        if let Some(ref domain) = args.domain {
            println!("Domain Filter: {}", domain);
            println!("{:-<60}", "");
        }

        println!("\nOverview:");
        println!("  Total Patterns: {}", stats.total_patterns);
        println!("  Average Reward: {:.3}", stats.average_reward);
        println!("  Average Effectiveness: {:.3}", stats.average_effectiveness);
        println!("  Total Reuse Count: {}", stats.total_reuse_count);
        println!("  With Embeddings: {}", stats.patterns_with_embeddings);

        if args.detailed || args.hierarchy {
            println!("\nCategories:");
            for cat in &stats.categories {
                println!("  {}: {} patterns", cat.name, cat.count);
            }
        }

        if args.detailed {
            println!("\nTop Patterns by Quality:");
            for (i, p) in stats.top_patterns.iter().enumerate() {
                println!("  {}. [{}] {} (score: {:.3})",
                    i + 1,
                    truncate(&p.id, 8),
                    p.problem,
                    p.quality_score
                );
            }
        }

        println!("{:=<60}\n", "");
    }

    Ok(())
}

/// Run the consolidate pattern command.
async fn run_consolidate_pattern(args: &ConsolidatePatternArgs) -> Result<()> {
    tracing::info!("Running pattern consolidation");

    let storage = init_storage_sqlite_only(&args.db_path).await?;

    // Constitution check before consolidation (F08)
    let constitution = Constitution::new();
    let ctx = OperationContext {
        operation: Operation::Consolidate,
        pattern_id: None,
        reward: None,
        tier: None,
        surprise_score: None,
        has_recent_backup: false,
        failure_mode: None,
        domain: None,
    };
    if !constitution.is_allowed(&ctx) {
        eprintln!("Constitution blocked consolidation. Ensure a recent backup exists or disable enforcement.");
        return Ok(());
    }

    // Create consolidation config and run real consolidation
    let consolidation_config = PatternConsolidationConfig {
        similarity_threshold: args.similarity,
        dry_run: args.dry_run,
        max_patterns_to_process: 10000,
        ..Default::default()
    };

    println!("\nRunning pattern consolidation...");
    println!("  Similarity threshold: {:.2}", args.similarity);
    println!("  Dry run: {}", args.dry_run);
    println!();

    let real_result = consolidate_patterns(&storage, &consolidation_config).await?;

    // Emit ConsolidationCompleted event (F16)
    if !real_result.dry_run && real_result.patterns_consolidated > 0 {
        let event_bus = EventBus::new();
        let merged_ids: Vec<String> = real_result
            .groups
            .iter()
            .flat_map(|g| g.merged_ids.iter().map(|id| id.to_string()))
            .collect();
        event_bus.publish_sync(NagualEvent::consolidation_completed(
            real_result.patterns_consolidated,
            0,
            merged_ids,
        ));
    }

    // Map real result to output structures
    let result = ConsolidationResult {
        patterns_analyzed: real_result.patterns_processed,
        groups_found: real_result.groups_formed,
        patterns_merged: real_result.patterns_consolidated,
        patterns_archived: 0,
        similarity_threshold: args.similarity,
        archive_threshold: args.archive_threshold,
        dry_run: args.dry_run,
        duration_ms: real_result.duration_ms,
        details: if args.verbose && !real_result.groups.is_empty() {
            Some(
                real_result
                    .groups
                    .iter()
                    .map(|g| ConsolidationDetail {
                        action: "merge".to_string(),
                        pattern_ids: std::iter::once(g.primary_id.to_string())
                            .chain(g.merged_ids.iter().map(|id| id.to_string()))
                            .collect(),
                        reason: format!("Similarity: {:.2}", g.average_similarity),
                    })
                    .collect(),
            )
        } else {
            None
        },
    };

    if args.json {
        let output = ConsolidationOutput {
            success: true,
            result,
            message: if args.dry_run {
                "Consolidation analysis completed (dry run)".to_string()
            } else {
                "Consolidation completed successfully".to_string()
            },
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        if args.dry_run {
            println!("\nConsolidation Analysis (Dry Run)");
        } else {
            println!("\nConsolidation Completed");
        }
        println!("{:=<60}", "");
        println!("Patterns analyzed: {}", result.patterns_analyzed);
        println!("Similarity groups found: {}", result.groups_found);
        println!("Patterns merged: {}", result.patterns_merged);
        println!("Patterns archived: {}", result.patterns_archived);
        println!("Duration: {}ms", result.duration_ms);
        println!("{:-<60}", "");
        println!("Similarity threshold: {:.2}", result.similarity_threshold);
        if let Some(threshold) = result.archive_threshold {
            println!("Archive threshold: {:.2}", threshold);
        }

        if let Some(ref details) = result.details {
            println!("\nDetails:");
            for d in details {
                println!("  [{}] {} - {}", d.action.to_uppercase(), d.pattern_ids.join(", "), d.reason);
            }
        }

        println!("{:=<60}\n", "");
    }

    Ok(())
}

/// Run the analyze command.
async fn run_analyze(args: &AnalyzeArgs) -> Result<()> {
    tracing::info!("Analyzing pattern quality");

    let patterns = if args.demo {
        create_demo_patterns()
    } else {
        tracing::warn!("Database retrieval not yet implemented, using demo data");
        create_demo_patterns()
    };

    // Calculate analysis
    let high_quality: Vec<_> = patterns.iter().filter(|p| p.quality_score() > 0.7).collect();
    let low_quality: Vec<_> = patterns.iter().filter(|p| p.quality_score() < 0.4).collect();
    let stale: Vec<_> = patterns.iter().filter(|p| p.reuse_count() == 0 && p.age_seconds() > 30 * 24 * 60 * 60).collect();

    let analysis = AnalysisResult {
        total_patterns: patterns.len(),
        high_quality_count: high_quality.len(),
        low_quality_count: low_quality.len(),
        stale_count: stale.len(),
        average_quality: patterns.iter().map(|p| p.quality_score()).sum::<f32>() / patterns.len().max(1) as f32,
        recommendations: if args.recommendations {
            let mut recs = Vec::new();
            if !low_quality.is_empty() {
                recs.push(Recommendation {
                    priority: "high".to_string(),
                    action: "review".to_string(),
                    description: format!("{} low-quality patterns should be reviewed or archived", low_quality.len()),
                    pattern_ids: low_quality.iter().take(5).map(|p| p.id().to_string()).collect(),
                });
            }
            if !stale.is_empty() {
                recs.push(Recommendation {
                    priority: "medium".to_string(),
                    action: "archive".to_string(),
                    description: format!("{} stale patterns (no reuse in 30+ days) could be archived", stale.len()),
                    pattern_ids: stale.iter().take(5).map(|p| p.id().to_string()).collect(),
                });
            }
            Some(recs)
        } else {
            None
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&analysis)?);
    } else {
        println!("\nPattern Quality Analysis");
        println!("{:=<60}", "");

        if let Some(ref domain) = args.domain {
            println!("Domain: {}", domain);
            println!("{:-<60}", "");
        }

        println!("\nQuality Summary:");
        println!("  Total Patterns: {}", analysis.total_patterns);
        println!("  Average Quality Score: {:.3}", analysis.average_quality);
        println!("  High Quality (>0.7): {}", analysis.high_quality_count);
        println!("  Low Quality (<0.4): {}", analysis.low_quality_count);
        println!("  Stale (no reuse 30d): {}", analysis.stale_count);

        if let Some(ref recs) = analysis.recommendations {
            println!("\nRecommendations:");
            for (i, rec) in recs.iter().enumerate() {
                println!("  {}. [{}] {}", i + 1, rec.priority.to_uppercase(), rec.description);
                println!("     Action: {}", rec.action);
                if !rec.pattern_ids.is_empty() {
                    println!("     Affected: {}", rec.pattern_ids.join(", "));
                }
            }
        }

        println!("{:=<60}\n", "");
    }

    Ok(())
}

/// Run pattern export.
async fn run_export_patterns(args: &ExportPatternArgs) -> Result<()> {
    use crate::reasoning_bank::export::*;

    tracing::info!("Exporting patterns");

    let storage = init_storage_sqlite_only(&args.db_path).await?;
    let all_patterns = storage.get_recent(100000).await?;

    // Apply filters
    let filtered: Vec<_> = all_patterns
        .into_iter()
        .filter(|p| {
            if let Some(ref domain) = args.domain {
                p.category().to_string().contains(domain)
            } else {
                true
            }
        })
        .filter(|p| {
            if let Some(min_r) = args.min_reward {
                p.reward() >= min_r
            } else {
                true
            }
        })
        .take(args.limit.unwrap_or(usize::MAX))
        .collect();

    let exported: Vec<ExportedPattern> = filtered
        .iter()
        .map(|p| ExportedPattern {
            problem: p.problem().to_string(),
            solution: p.solution().to_string(),
            domain: p.category().to_string(),
            context: if p.context().is_empty() {
                None
            } else {
                Some(p.context().to_string())
            },
            tags: p.tags().to_vec(),
            reward: p.reward(),
            effectiveness: p.effectiveness(),
            confidence: p.confidence(),
            success_rate: if p.success() { 1.0 } else { 0.0 },
            reuse_count: p.reuse_count(),
            tier: "booster".to_string(),
        })
        .collect();

    let export = PatternExport {
        version: EXPORT_VERSION.to_string(),
        exported_at: chrono::Utc::now(),
        source: format!("nagual v{}", env!("CARGO_PKG_VERSION")),
        pattern_count: exported.len(),
        patterns: exported,
    };

    write_export(&export, &args.output)?;

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "success": true,
                "file": args.output.display().to_string(),
                "pattern_count": export.pattern_count,
            }))?
        );
    } else {
        println!("\nExport Complete");
        println!("{:-<60}", "");
        println!("File: {}", args.output.display());
        println!("Patterns exported: {}", export.pattern_count);
        println!("{:-<60}\n", "");
    }

    Ok(())
}

/// Run pattern import.
async fn run_import_patterns(args: &ImportPatternArgs) -> Result<()> {
    use crate::reasoning_bank::export::*;

    tracing::info!("Importing patterns from {}", args.input.display());

    let export = read_export(&args.input).map_err(|e| {
        crate::error::NagualError::internal(format!("Failed to read export file: {}", e))
    })?;

    println!(
        "\nImport {} (v{})",
        if args.dry_run { "Preview" } else { "Starting" },
        export.version
    );
    println!("{:-<60}", "");
    println!("Source: {}", export.source);
    println!("Exported at: {}", export.exported_at);
    println!("Patterns in file: {}", export.pattern_count);

    if args.dry_run {
        println!("\n--- DRY RUN (no changes made) ---\n");
        for (i, p) in export.patterns.iter().enumerate().take(20) {
            println!(
                "  {}. [{}] {} (reward: {:.2})",
                i + 1,
                p.domain,
                truncate(&p.problem, 50),
                p.reward
            );
        }
        if export.pattern_count > 20 {
            println!("  ... and {} more", export.pattern_count - 20);
        }
    } else {
        // Actual import - store each pattern
        let storage = init_storage_sqlite_only(&args.db_path).await?;
        let existing = storage.get_recent(100000).await?;
        let existing_keys: std::collections::HashSet<String> = existing
            .iter()
            .map(|p| dedup_key(p.problem(), &p.category().to_string()))
            .collect();

        let mut imported = 0;
        let mut skipped = 0;

        for p in &export.patterns {
            let key = dedup_key(&p.problem, &p.domain);
            if !args.force && existing_keys.contains(&key) {
                skipped += 1;
                continue;
            }

            // Store pattern via the Pattern builder and storage
            let pattern = Pattern::builder()
                .problem(&p.problem)
                .solution(&p.solution)
                .category(PatternCategory::from(p.domain.as_str()))
                .context(p.context.as_deref().unwrap_or(""))
                .effectiveness(p.effectiveness)
                .reward(p.reward)
                .confidence(p.confidence)
                .tags(p.tags.clone())
                .build();

            match storage.store_pattern(&pattern).await {
                Ok(_) => imported += 1,
                Err(e) => {
                    tracing::warn!("Failed to import pattern: {}", e);
                    skipped += 1;
                }
            }
        }

        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "success": true,
                    "imported": imported,
                    "skipped": skipped,
                    "total_in_file": export.pattern_count,
                }))?
            );
        } else {
            println!("\nImported: {}", imported);
            println!("Skipped (duplicates): {}", skipped);
        }
    }

    println!("{:-<60}\n", "");
    Ok(())
}

/// Run pyramid summary generation/stats.
async fn run_pyramid(args: &PyramidArgs) -> Result<()> {
    tracing::info!("Managing pyramid summaries");

    let adapter = pyramid::init_adapter(&args.db_path).await?;

    if args.stats || (!args.generate && !args.dry_run) {
        // Show statistics
        let stats = pyramid::get_pyramid_stats(&adapter).await?;

        if args.json {
            let output = PyramidStatsOutput {
                total_patterns: stats.total_patterns,
                with_pyramid: stats.with_pyramid,
                without_pyramid: stats.without_pyramid,
                with_title_only: stats.with_title_only,
                with_summary_only: stats.with_summary_only,
                coverage_percent: stats.coverage_percent(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            println!("\nPyramid Summary Statistics");
            println!("{:=<60}", "");
            println!("Total patterns:       {}", stats.total_patterns);
            println!(
                "With pyramid:         {} ({:.1}%)",
                stats.with_pyramid,
                stats.coverage_percent()
            );
            println!("Without pyramid:      {}", stats.without_pyramid);
            println!("  - Title only:       {}", stats.with_title_only);
            println!("  - Summary only:     {}", stats.with_summary_only);
            println!("{:=<60}\n", "");
        }
    } else if args.generate || args.dry_run {
        // Generate pyramids
        let dry_run = args.dry_run;
        let stats = pyramid::generate_missing_pyramids(&adapter, dry_run, args.limit).await?;

        if args.json {
            let output = PyramidGenerateOutput {
                success: true,
                dry_run,
                patterns_processed: stats.generated,
                total_patterns: stats.total_patterns,
                with_pyramid: stats.with_pyramid,
                coverage_percent: stats.coverage_percent(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            if dry_run {
                println!("\nPyramid Generation (Dry Run)");
                println!("{:=<60}", "");
                println!("Would generate pyramids for {} patterns", stats.generated);
            } else {
                println!("\nPyramid Generation Complete");
                println!("{:=<60}", "");
                println!("Generated pyramids for {} patterns", stats.generated);
            }
            println!(
                "Coverage: {} / {} ({:.1}%)",
                stats.with_pyramid,
                stats.total_patterns,
                stats.coverage_percent()
            );
            println!("{:=<60}\n", "");
        }
    }

    Ok(())
}

// Input/Output structures

#[derive(Deserialize)]
struct PatternInput {
    problem: String,
    solution: String,
    category: Option<String>,
    context: Option<String>,
    tags: Option<Vec<String>>,
    effectiveness: Option<f32>,
    confidence: Option<f32>,
}

#[derive(Serialize)]
struct StoreOutput {
    id: String,
    category: String,
    success: bool,
    message: String,
}

#[derive(Serialize)]
struct SearchResult {
    rank: usize,
    id: String,
    problem: String,
    solution: String,
    category: String,
    quality_score: f32,
    effectiveness: f32,
    reward: f32,
    reuse_count: u32,
}

#[derive(Serialize)]
struct PatternStatistics {
    total_patterns: usize,
    average_reward: f32,
    average_effectiveness: f32,
    total_reuse_count: u32,
    patterns_with_embeddings: usize,
    categories: Vec<CategoryStats>,
    top_patterns: Vec<TopPatternInfo>,
}

#[derive(Serialize)]
struct CategoryStats {
    name: String,
    count: usize,
}

#[derive(Serialize)]
struct TopPatternInfo {
    id: String,
    problem: String,
    quality_score: f32,
}

#[derive(Serialize)]
struct ConsolidationResult {
    patterns_analyzed: usize,
    groups_found: usize,
    patterns_merged: usize,
    patterns_archived: usize,
    similarity_threshold: f32,
    archive_threshold: Option<f32>,
    dry_run: bool,
    duration_ms: u64,
    details: Option<Vec<ConsolidationDetail>>,
}

#[derive(Serialize)]
struct ConsolidationDetail {
    action: String,
    pattern_ids: Vec<String>,
    reason: String,
}

#[derive(Serialize)]
struct ConsolidationOutput {
    success: bool,
    result: ConsolidationResult,
    message: String,
}

#[derive(Serialize)]
struct AnalysisResult {
    total_patterns: usize,
    high_quality_count: usize,
    low_quality_count: usize,
    stale_count: usize,
    average_quality: f32,
    recommendations: Option<Vec<Recommendation>>,
}

#[derive(Serialize)]
struct Recommendation {
    priority: String,
    action: String,
    description: String,
    pattern_ids: Vec<String>,
}

#[derive(Serialize)]
struct ErrorOutput {
    error: String,
    error_type: String,
}

#[derive(Serialize)]
struct PyramidStatsOutput {
    total_patterns: usize,
    with_pyramid: usize,
    without_pyramid: usize,
    with_title_only: usize,
    with_summary_only: usize,
    coverage_percent: f32,
}

#[derive(Serialize)]
struct PyramidGenerateOutput {
    success: bool,
    dry_run: bool,
    patterns_processed: usize,
    total_patterns: usize,
    with_pyramid: usize,
    coverage_percent: f32,
}

/// Truncate string to max length.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Create demo patterns.
fn create_demo_patterns() -> Vec<Pattern> {
    vec![
        Pattern::builder()
            .id("pat-001")
            .problem("How to implement retry with exponential backoff")
            .solution("Use tokio::time::sleep with doubling delay, add jitter, cap at max_delay")
            .category(PatternCategory::Resilience)
            .context("Network operations, API calls, database connections")
            .effectiveness(0.92)
            .reward(0.88)
            .confidence(0.90)
            .reuse_count(25)
            .tag("retry")
            .tag("resilience")
            .tag("async")
            .build(),
        Pattern::builder()
            .id("pat-002")
            .problem("Efficient batch processing for large datasets")
            .solution("Use chunked iterators with tokio::spawn for parallel processing, bounded channels for backpressure")
            .category(PatternCategory::Performance)
            .context("Data pipelines, ETL processes")
            .effectiveness(0.85)
            .reward(0.82)
            .confidence(0.88)
            .reuse_count(18)
            .tag("batch")
            .tag("performance")
            .tag("parallel")
            .build(),
        Pattern::builder()
            .id("pat-003")
            .problem("Secure credential management in applications")
            .solution("Use environment variables for secrets, never hardcode, use secret managers in production")
            .category(PatternCategory::Security)
            .context("Application configuration, deployment")
            .effectiveness(0.95)
            .reward(0.90)
            .confidence(0.95)
            .reuse_count(30)
            .tag("security")
            .tag("credentials")
            .tag("secrets")
            .build(),
        Pattern::builder()
            .id("pat-004")
            .problem("Testing async code with proper timeout handling")
            .solution("Use tokio::time::timeout wrapper, implement test-specific mock implementations")
            .category(PatternCategory::Testing)
            .context("Unit tests, integration tests")
            .effectiveness(0.78)
            .reward(0.75)
            .confidence(0.80)
            .reuse_count(15)
            .tag("testing")
            .tag("async")
            .tag("timeout")
            .build(),
        Pattern::builder()
            .id("pat-005")
            .problem("Outdated logging pattern")
            .solution("Use println! for all logging")
            .category(PatternCategory::Custom("logging".to_string()))
            .context("Legacy code")
            .effectiveness(0.25)
            .reward(0.20)
            .confidence(0.30)
            .reuse_count(0)
            .tag("logging")
            .tag("outdated")
            .build(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct TestCli {
        #[command(subcommand)]
        command: TestCommand,
    }

    #[derive(Subcommand, Debug)]
    enum TestCommand {
        Patterns(PatternsCommand),
    }

    #[test]
    fn test_cli_parse_store_json() {
        let args = vec![
            "test",
            "patterns",
            "store",
            "--json-input",
            r#"{"problem": "test", "solution": "solution"}"#,
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_store_args() {
        let args = vec![
            "test",
            "patterns",
            "store",
            "--problem",
            "How to cache",
            "--solution",
            "Use Redis",
            "--category",
            "performance",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_search() {
        let args = vec![
            "test",
            "patterns",
            "search",
            "retry backoff",
            "--domain",
            "resilience",
            "--limit",
            "5",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_stats() {
        let args = vec!["test", "patterns", "stats", "--detailed", "--json"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_consolidate() {
        let args = vec![
            "test",
            "patterns",
            "consolidate",
            "--similarity",
            "0.85",
            "--dry-run",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_analyze() {
        let args = vec![
            "test",
            "patterns",
            "analyze",
            "--recommendations",
            "--demo",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_export() {
        let args = vec![
            "test",
            "patterns",
            "export",
            "--output",
            "my-export.json",
            "--domain",
            "performance",
            "--min-reward",
            "0.7",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_export_defaults() {
        let args = vec!["test", "patterns", "export"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_import() {
        let args = vec![
            "test",
            "patterns",
            "import",
            "patterns-export.json",
            "--dry-run",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_import_force() {
        let args = vec![
            "test",
            "patterns",
            "import",
            "patterns-export.json",
            "--force",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_pyramid_stats() {
        let args = vec!["test", "patterns", "pyramid", "--stats"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_pyramid_generate() {
        let args = vec!["test", "patterns", "pyramid", "--generate"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_pyramid_dry_run() {
        let args = vec![
            "test",
            "patterns",
            "pyramid",
            "--generate",
            "--dry-run",
            "--limit",
            "100",
        ];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_cli_parse_pyramid_json() {
        let args = vec!["test", "patterns", "pyramid", "--stats", "--json"];
        let cli = TestCli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_demo_patterns() {
        let patterns = create_demo_patterns();
        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.category() == &PatternCategory::Resilience));
    }
}
