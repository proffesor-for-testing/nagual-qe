//! Graph CLI command implementation
//!
//! Provides commands for graph operations including pressure propagation
//! visualization, edge management, and neighbor queries.
//!
//! Usage:
//! - `nagual graph pressure <node_id>` - Show influence scores from a node
//! - `nagual graph pressure <node_id> --depth 5` - Set iteration depth
//! - `nagual graph pressure <node_id> --damping 0.9` - Set damping factor
//! - `nagual graph pressure <node_id> --top 10` - Show top N influenced nodes
//! - `nagual graph pressure <node_id> --json` - Output as JSON
//! - `nagual graph link <source> <target>` - Create edge between nodes
//! - `nagual graph query <node>` - Query node neighbors

use std::path::PathBuf;

use clap::{Args, Subcommand};
use serde::Serialize;

use crate::error::Result;
use crate::graph::{
    propagate_pressure, EdgeType, GraphProvider, GraphStorage, GraphStorageConfig, InMemoryGraph,
    PressureConfig, PressureError, PressureResult,
};

/// Database-backed graph provider that loads edges from SQLite.
///
/// Implements the `GraphProvider` trait by querying the context_graph table
/// for edges and neighbors. Caches data in memory for efficient pressure propagation.
pub struct SqliteGraphProvider {
    /// In-memory graph loaded from database (public for move semantics)
    pub graph: InMemoryGraph,
}

impl SqliteGraphProvider {
    /// Load graph from SQLite database.
    ///
    /// Reads all edges from the context_graph table and builds an in-memory
    /// graph structure for efficient neighbor lookups during pressure propagation.
    pub fn load(db_path: &PathBuf) -> std::result::Result<Self, String> {
        use rusqlite::Connection;

        let conn = Connection::open(db_path).map_err(|e| format!("Failed to open database: {}", e))?;

        // Check if context_graph table exists
        let table_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='context_graph'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map(|count| count > 0)
            .unwrap_or(false);

        if !table_exists {
            return Err("context_graph table not found in database".to_string());
        }

        let mut graph = InMemoryGraph::new();

        // Load all edges from context_graph table
        let mut stmt = conn
            .prepare("SELECT source_id, target_id, strength FROM context_graph")
            .map_err(|e| format!("Failed to prepare query: {}", e))?;

        let edges = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, f64>(2)?,
                ))
            })
            .map_err(|e| format!("Failed to query edges: {}", e))?;

        let mut edge_count = 0;
        for edge_result in edges {
            if let Ok((source, target, strength)) = edge_result {
                graph.add_edge(source, target, strength);
                edge_count += 1;
            }
        }

        tracing::info!(
            "Loaded {} edges from database at {:?}",
            edge_count,
            db_path
        );

        Ok(Self { graph })
    }

    /// Get the underlying in-memory graph.
    pub fn inner(&self) -> &InMemoryGraph {
        &self.graph
    }
}

impl GraphProvider for SqliteGraphProvider {
    fn get_neighbors(&self, node_id: &str) -> Vec<(String, f64)> {
        self.graph.get_neighbors(node_id)
    }

    fn node_exists(&self, node_id: &str) -> bool {
        self.graph.node_exists(node_id)
    }
}

/// Extended database statistics with more graph metrics.
#[derive(Debug, Clone, Serialize)]
pub struct ExtendedGraphStats {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Average degree (edges per node)
    pub avg_degree: f64,
    /// Number of connected components (approximation)
    pub connected_components: usize,
    /// Edge counts by type
    pub edges_by_type: Vec<(String, usize)>,
    /// Database path
    pub db_path: String,
}

impl ExtendedGraphStats {
    /// Load statistics from database.
    pub async fn load(db_path: &PathBuf) -> std::result::Result<Self, String> {
        let config = GraphStorageConfig::new(db_path.to_string_lossy().to_string());
        let storage = GraphStorage::with_config(config)
            .map_err(|e| format!("Failed to open graph storage: {}", e))?;

        let stats = storage
            .stats()
            .await
            .map_err(|e| format!("Failed to get stats: {}", e))?;

        let avg_degree = if stats.node_count > 0 {
            (stats.edge_count as f64 * 2.0) / stats.node_count as f64
        } else {
            0.0
        };

        // Approximate connected components (for now, use a simple heuristic)
        // A full implementation would require graph traversal
        let connected_components = if stats.node_count == 0 {
            0
        } else if stats.edge_count == 0 {
            stats.node_count
        } else {
            // Heuristic: assume mostly connected for graphs with reasonable edge density
            1
        };

        Ok(Self {
            node_count: stats.node_count,
            edge_count: stats.edge_count,
            avg_degree,
            connected_components,
            edges_by_type: stats.edges_by_type,
            db_path: db_path.to_string_lossy().to_string(),
        })
    }
}

/// Graph operations command
///
/// Provides graph-based analysis tools including GNN-style pressure propagation
/// for influence measurement, edge management, and neighbor queries.
#[derive(Args, Debug)]
pub struct GraphCommand {
    #[command(subcommand)]
    pub command: GraphSubcommands,
}

/// Graph subcommands
#[derive(Subcommand, Debug)]
pub enum GraphSubcommands {
    /// Run pressure propagation from a node
    ///
    /// Calculates influence scores showing how "pressure" flows from the
    /// specified node through the graph based on edge weights.
    Pressure(PressureCommand),

    /// Show graph statistics
    ///
    /// Displays summary statistics about the knowledge graph including
    /// node count, edge count, and connectivity metrics.
    Stats(StatsCommand),

    /// Create an edge between two nodes
    ///
    /// Links a source node to a target node with an optional weight.
    /// Creates nodes if they don't exist.
    Link(LinkCommand),

    /// Query neighbors of a node
    ///
    /// Returns direct neighbors (incoming and outgoing) of a node
    /// along with edge weights and relationship information.
    Query(QueryCommand),

    /// Discover coherent knowledge clusters via minimum cut
    ///
    /// Uses the Stoer-Wagner algorithm to partition the knowledge graph
    /// into coherent clusters. Clusters with internal edge weight above
    /// the threshold are considered cohesive.
    #[cfg(feature = "mincut")]
    Cluster(ClusterCommand),
}

/// Pressure propagation command
#[derive(Args, Debug)]
pub struct PressureCommand {
    /// Starting node ID for pressure propagation
    #[arg(value_name = "NODE_ID")]
    pub node_id: String,

    /// Iteration depth (max propagation steps)
    #[arg(short, long, default_value = "3", value_name = "DEPTH")]
    pub depth: usize,

    /// Damping factor (0.0-1.0, controls pressure transfer ratio)
    #[arg(long, default_value = "0.85", value_name = "FACTOR")]
    pub damping: f64,

    /// Convergence epsilon (stop early if change is below this)
    #[arg(long, default_value = "0.000001", value_name = "EPSILON")]
    pub epsilon: f64,

    /// Show only top N influenced nodes
    #[arg(short, long, default_value = "20", value_name = "N")]
    pub top: usize,

    /// Normalize output scores to sum to 1.0
    #[arg(short, long)]
    pub normalize: bool,

    /// Path to SQLite database containing the graph
    #[arg(long, value_name = "PATH", default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,

    /// Use mock/demo graph instead of database
    #[arg(long)]
    pub demo: bool,

    /// Include detailed execution statistics
    #[arg(long)]
    pub stats: bool,
}

/// Graph statistics command
#[derive(Args, Debug)]
pub struct StatsCommand {
    /// Path to SQLite database containing the graph
    #[arg(long, value_name = "PATH", default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,
}

/// Create edge between nodes command
#[derive(Args, Debug)]
pub struct LinkCommand {
    /// Source node ID
    #[arg(value_name = "SOURCE")]
    pub source: String,

    /// Target node ID
    #[arg(value_name = "TARGET")]
    pub target: String,

    /// Edge weight (0.0-1.0, default 1.0)
    #[arg(short, long, default_value = "1.0")]
    pub weight: f64,

    /// Edge label/type
    #[arg(short, long)]
    pub label: Option<String>,

    /// Create bidirectional edge
    #[arg(short, long)]
    pub bidirectional: bool,

    /// Path to SQLite database
    #[arg(long, value_name = "PATH", default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,

    /// Use demo mode (in-memory graph)
    #[arg(long)]
    pub demo: bool,
}

/// Query node neighbors command
#[derive(Args, Debug)]
pub struct QueryCommand {
    /// Node ID to query
    #[arg(value_name = "NODE")]
    pub node: String,

    /// Direction: incoming, outgoing, or both
    #[arg(short, long, default_value = "both")]
    pub direction: String,

    /// Maximum depth for traversal (default 1 for direct neighbors)
    #[arg(long, default_value = "1")]
    pub depth: usize,

    /// Minimum edge weight to include
    #[arg(long)]
    pub min_weight: Option<f64>,

    /// Maximum number of neighbors to return
    #[arg(short, long, default_value = "50")]
    pub limit: usize,

    /// Path to SQLite database
    #[arg(long, value_name = "PATH", default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Output results as JSON
    #[arg(long)]
    pub json: bool,

    /// Show verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Use demo mode (in-memory graph)
    #[arg(long)]
    pub demo: bool,
}

/// Discover coherent knowledge clusters via minimum cut
#[cfg(feature = "mincut")]
#[derive(Args, Debug)]
pub struct ClusterCommand {
    /// Minimum cut threshold (clusters with cut weight above this stay together)
    #[arg(short, long, default_value = "1.0")]
    pub threshold: f64,

    /// Database path
    #[arg(long, value_name = "PATH", default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Show detailed node lists per cluster
    #[arg(short, long)]
    pub verbose: bool,
}

impl GraphCommand {
    /// Execute the graph command
    pub async fn run(&self) -> Result<()> {
        match &self.command {
            GraphSubcommands::Pressure(cmd) => cmd.run().await,
            GraphSubcommands::Stats(cmd) => cmd.run().await,
            GraphSubcommands::Link(cmd) => cmd.run().await,
            GraphSubcommands::Query(cmd) => cmd.run().await,
            #[cfg(feature = "mincut")]
            GraphSubcommands::Cluster(cmd) => cmd.run().await,
        }
    }
}

impl PressureCommand {
    /// Execute the pressure propagation command
    pub async fn run(&self) -> Result<()> {
        tracing::info!(
            "Running pressure propagation from node '{}' with depth={}, damping={}",
            self.node_id,
            self.depth,
            self.damping
        );

        // Build configuration
        let config = PressureConfig {
            damping_factor: self.damping.clamp(0.0, 1.0),
            max_iterations: self.depth.max(1),
            epsilon: self.epsilon.max(0.0),
            normalize: self.normalize,
            ..Default::default()
        };

        // Get graph provider
        let graph: Box<dyn GraphProvider> = if self.demo {
            Box::new(create_demo_graph())
        } else {
            // Load from database
            if !self.db_path.exists() {
                tracing::warn!(
                    "Database not found at {:?}, using demo graph",
                    self.db_path
                );
                Box::new(create_demo_graph())
            } else {
                match SqliteGraphProvider::load(&self.db_path) {
                    Ok(provider) => {
                        if provider.inner().node_count() == 0 {
                            tracing::warn!("Database graph is empty, using demo graph");
                            Box::new(create_demo_graph())
                        } else {
                            Box::new(provider)
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load graph from database: {}, using demo graph", e);
                        Box::new(create_demo_graph())
                    }
                }
            }
        };

        // Run propagation
        let result = match propagate_pressure(graph.as_ref(), &self.node_id, &config) {
            Ok(r) => r,
            Err(e) => {
                return self.handle_error(e);
            }
        };

        // Output results
        if self.json {
            self.print_json(&result)?;
        } else {
            self.display_table(&result);
        }

        Ok(())
    }

    /// Handle pressure propagation errors
    fn handle_error(&self, error: PressureError) -> Result<()> {
        if self.json {
            let output = ErrorOutput {
                error: error.to_string(),
                error_type: match &error {
                    PressureError::NodeNotFound { .. } => "node_not_found",
                    PressureError::InvalidConfig { .. } => "invalid_config",
                    PressureError::LimitExceeded { .. } => "limit_exceeded",
                }
                .to_string(),
            };
            println!("{}", serde_json::to_string_pretty(&output)?);
        } else {
            eprintln!("Error: {}", error);
        }
        Ok(())
    }

    /// Print results as JSON
    fn print_json(&self, result: &PressureResult) -> Result<()> {
        let output = PressureJsonOutput {
            source_node: result.source_node.clone(),
            scores: result
                .top_n(self.top)
                .into_iter()
                .map(|(k, v)| NodeScore {
                    node_id: k.clone(),
                    pressure: *v,
                })
                .collect(),
            stats: if self.stats {
                Some(StatsOutput {
                    iterations_used: result.stats.iterations_used,
                    converged: result.stats.converged,
                    final_delta: result.stats.final_delta,
                    nodes_reached: result.stats.nodes_reached,
                    total_pressure: result.stats.total_pressure,
                    execution_time_us: result.stats.execution_time_us,
                })
            } else {
                None
            },
            config: ConfigOutput {
                damping_factor: result.config.damping_factor,
                max_iterations: result.config.max_iterations,
                epsilon: result.config.epsilon,
                normalize: result.config.normalize,
            },
        };

        println!("{}", serde_json::to_string_pretty(&output)?);
        Ok(())
    }

    /// Display results as a formatted table
    fn display_table(&self, result: &PressureResult) {
        println!("\nPressure Propagation Results");
        println!("{:=<60}", "");
        println!("Source Node: {}", result.source_node);
        println!(
            "Config: damping={:.2}, iterations={}, epsilon={:.0e}",
            result.config.damping_factor, result.config.max_iterations, result.config.epsilon
        );
        println!("{:-<60}", "");

        // Display top N nodes
        let top_nodes = result.top_n(self.top);
        let max_node_len = top_nodes
            .iter()
            .map(|(id, _)| id.len())
            .max()
            .unwrap_or(10)
            .max(10);

        println!(
            "{:<width$}  {:>12}  {:>10}",
            "Node",
            "Pressure",
            "Influence",
            width = max_node_len
        );
        println!("{:-<width$}  {:->12}  {:->10}", "", "", "", width = max_node_len);

        let max_pressure = top_nodes.first().map(|(_, p)| **p).unwrap_or(1.0);

        for (node_id, pressure) in &top_nodes {
            // Calculate influence bar
            let influence = *pressure / max_pressure;
            let bar_width = (influence * 10.0).round() as usize;
            let bar = "#".repeat(bar_width);

            println!(
                "{:<width$}  {:>12.6}  {:<10}",
                node_id,
                pressure,
                bar,
                width = max_node_len
            );
        }

        // Show statistics if requested
        if self.stats {
            println!("{:-<60}", "");
            println!("Statistics:");
            println!("  Iterations used: {}", result.stats.iterations_used);
            println!(
                "  Converged: {}",
                if result.stats.converged { "yes" } else { "no" }
            );
            println!("  Final delta: {:.2e}", result.stats.final_delta);
            println!("  Nodes reached: {}", result.stats.nodes_reached);
            println!("  Total pressure: {:.6}", result.stats.total_pressure);
            println!("  Execution time: {}us", result.stats.execution_time_us);
        }

        println!("{:=<60}\n", "");
    }
}

impl StatsCommand {
    /// Execute the stats command
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Fetching graph statistics from {:?}", self.db_path);

        // Load statistics from database
        let stats = if self.db_path.exists() {
            match ExtendedGraphStats::load(&self.db_path).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("Failed to load stats from database: {}", e);
                    ExtendedGraphStats {
                        node_count: 0,
                        edge_count: 0,
                        avg_degree: 0.0,
                        connected_components: 0,
                        edges_by_type: Vec::new(),
                        db_path: self.db_path.to_string_lossy().to_string(),
                    }
                }
            }
        } else {
            tracing::warn!("Database not found at {:?}", self.db_path);
            ExtendedGraphStats {
                node_count: 0,
                edge_count: 0,
                avg_degree: 0.0,
                connected_components: 0,
                edges_by_type: Vec::new(),
                db_path: self.db_path.to_string_lossy().to_string(),
            }
        };

        if self.json {
            println!("{}", serde_json::to_string_pretty(&stats)?);
        } else {
            println!("\nGraph Statistics");
            println!("{:=<50}", "");
            println!("Database: {}", stats.db_path);
            println!("Node count: {}", stats.node_count);
            println!("Edge count: {}", stats.edge_count);
            println!("Average degree: {:.2}", stats.avg_degree);
            println!("Connected components: {}", stats.connected_components);

            if !stats.edges_by_type.is_empty() {
                println!("\nEdges by Type:");
                for (edge_type, count) in &stats.edges_by_type {
                    println!("  {}: {}", edge_type, count);
                }
            }

            println!("{:=<50}\n", "");

            if stats.node_count == 0 {
                println!("Note: Database appears empty or not found.");
                println!("Use --demo flag with pressure command to test with sample data.");
                println!("Use 'nagual graph link <source> <target>' to create edges.");
            }
        }

        Ok(())
    }
}

impl LinkCommand {
    /// Execute the link command
    pub async fn run(&self) -> Result<()> {
        tracing::info!(
            "Creating edge: {} -> {} (weight: {})",
            self.source,
            self.target,
            self.weight
        );

        // Validate weight
        let weight = self.weight.clamp(0.0, 1.0);

        if self.demo {
            // Demo mode - just show what would happen
            let output = LinkOutput {
                source: self.source.clone(),
                target: self.target.clone(),
                weight,
                label: self.label.clone(),
                bidirectional: self.bidirectional,
                success: true,
                message: "Edge created successfully (demo mode)".to_string(),
            };

            if self.json {
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                println!("\nEdge Created (demo mode)");
                println!("{:-<50}", "");
                println!("Source: {}", output.source);
                println!("Target: {}", output.target);
                println!("Weight: {:.3}", output.weight);
                if let Some(ref label) = output.label {
                    println!("Label: {}", label);
                }
                if output.bidirectional {
                    println!("Direction: Bidirectional");
                }
                println!("{:-<50}\n", "");
            }
        } else {
            // Persist edge to database
            let config = GraphStorageConfig::new(self.db_path.to_string_lossy().to_string());
            let storage = match GraphStorage::with_config(config) {
                Ok(s) => s,
                Err(e) => {
                    let output = LinkOutput {
                        source: self.source.clone(),
                        target: self.target.clone(),
                        weight,
                        label: self.label.clone(),
                        bidirectional: self.bidirectional,
                        success: false,
                        message: format!("Failed to open database: {}", e),
                    };
                    if self.json {
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    } else {
                        eprintln!("Error: {}", output.message);
                    }
                    return Ok(());
                }
            };

            // Determine edge type from label or use default
            let edge_type = self.label
                .as_ref()
                .and_then(|l| EdgeType::from_str(l))
                .unwrap_or(EdgeType::RelatedTo);

            // Create the edge
            let result = storage
                .create_edge(&self.source, &self.target, edge_type, weight, None)
                .await;

            match result {
                Ok(create_result) => {
                    let message = if create_result.created {
                        "Edge created successfully".to_string()
                    } else {
                        format!(
                            "Edge updated (previous weight: {:.3})",
                            create_result.previous_strength.unwrap_or(0.0)
                        )
                    };

                    // Create reverse edge if bidirectional
                    if self.bidirectional {
                        let _ = storage
                            .create_edge(&self.target, &self.source, edge_type, weight, None)
                            .await;
                    }

                    let output = LinkOutput {
                        source: self.source.clone(),
                        target: self.target.clone(),
                        weight,
                        label: self.label.clone(),
                        bidirectional: self.bidirectional,
                        success: true,
                        message,
                    };

                    if self.json {
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    } else {
                        println!("\nEdge Created");
                        println!("{:-<50}", "");
                        println!("Source: {}", output.source);
                        println!("Target: {}", output.target);
                        println!("Weight: {:.3}", output.weight);
                        println!("Type: {}", edge_type);
                        if self.bidirectional {
                            println!("Direction: Bidirectional");
                        }
                        println!("Status: {}", output.message);
                        println!("{:-<50}\n", "");
                    }
                }
                Err(e) => {
                    let output = LinkOutput {
                        source: self.source.clone(),
                        target: self.target.clone(),
                        weight,
                        label: self.label.clone(),
                        bidirectional: self.bidirectional,
                        success: false,
                        message: format!("Failed to create edge: {}", e),
                    };

                    if self.json {
                        println!("{}", serde_json::to_string_pretty(&output)?);
                    } else {
                        eprintln!("Error: {}", output.message);
                    }
                }
            }
        }

        Ok(())
    }
}

impl QueryCommand {
    /// Execute the query command
    pub async fn run(&self) -> Result<()> {
        tracing::info!("Querying neighbors of: {}", self.node);

        let graph: InMemoryGraph = if self.demo {
            create_demo_graph()
        } else {
            // Load from database
            if !self.db_path.exists() {
                if self.verbose {
                    tracing::warn!("Database not found at {:?}, using demo graph", self.db_path);
                }
                create_demo_graph()
            } else {
                match SqliteGraphProvider::load(&self.db_path) {
                    Ok(provider) => {
                        if provider.inner().node_count() == 0 {
                            if self.verbose {
                                tracing::warn!("Database graph is empty, using demo graph");
                            }
                            create_demo_graph()
                        } else {
                            // Clone the inner graph from the provider
                            provider.graph
                        }
                    }
                    Err(e) => {
                        if self.verbose {
                            tracing::warn!("Failed to load graph from database: {}, using demo graph", e);
                        }
                        create_demo_graph()
                    }
                }
            }
        };

        // Check if node exists
        if !graph.node_exists(&self.node) {
            let output = QueryErrorOutput {
                error: format!("Node not found: {}", self.node),
                error_type: "node_not_found".to_string(),
            };

            if self.json {
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                eprintln!("Error: Node '{}' not found in graph", self.node);
            }
            return Ok(());
        }

        // Get neighbors
        let outgoing = graph.get_neighbors(&self.node);
        let incoming = graph.get_reverse_neighbors(&self.node);

        // Build result based on direction
        let mut neighbors: Vec<NeighborInfo> = Vec::new();

        match self.direction.as_str() {
            "outgoing" => {
                for (target, weight) in outgoing {
                    if self.min_weight.map(|m| weight >= m).unwrap_or(true) {
                        neighbors.push(NeighborInfo {
                            node_id: target.clone(),
                            direction: "outgoing".to_string(),
                            weight,
                            depth: 1,
                        });
                    }
                }
            }
            "incoming" => {
                for (source, weight) in incoming {
                    if self.min_weight.map(|m| weight >= m).unwrap_or(true) {
                        neighbors.push(NeighborInfo {
                            node_id: source.clone(),
                            direction: "incoming".to_string(),
                            weight,
                            depth: 1,
                        });
                    }
                }
            }
            _ => {
                // Both directions
                for (target, weight) in outgoing {
                    if self.min_weight.map(|m| weight >= m).unwrap_or(true) {
                        neighbors.push(NeighborInfo {
                            node_id: target.clone(),
                            direction: "outgoing".to_string(),
                            weight,
                            depth: 1,
                        });
                    }
                }
                for (source, weight) in incoming {
                    if self.min_weight.map(|m| weight >= m).unwrap_or(true) {
                        // Avoid duplicates for bidirectional edges
                        if !neighbors.iter().any(|n| n.node_id == source) {
                            neighbors.push(NeighborInfo {
                                node_id: source.clone(),
                                direction: "incoming".to_string(),
                                weight,
                                depth: 1,
                            });
                        }
                    }
                }
            }
        }

        // Sort by weight descending and limit
        neighbors.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(self.limit);

        let result = QueryResult {
            node: self.node.clone(),
            direction: self.direction.clone(),
            neighbor_count: neighbors.len(),
            neighbors,
        };

        if self.json {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("\nNeighbors of: {}", result.node);
            println!("Direction: {}", result.direction);
            println!("{:=<60}", "");

            if result.neighbors.is_empty() {
                println!("\nNo neighbors found.");
            } else {
                println!(
                    "\n{:<30}  {:>10}  {:>10}",
                    "Node", "Direction", "Weight"
                );
                println!("{:-<60}", "");

                for neighbor in &result.neighbors {
                    println!(
                        "{:<30}  {:>10}  {:>10.4}",
                        &neighbor.node_id,
                        &neighbor.direction,
                        neighbor.weight
                    );
                }
            }

            println!("{:-<60}", "");
            println!("Total neighbors: {}\n", result.neighbor_count);
        }

        Ok(())
    }
}

#[cfg(feature = "mincut")]
impl ClusterCommand {
    /// Execute the cluster discovery command
    pub async fn run(&self) -> Result<()> {
        use crate::graph::mincut::from_in_memory_graph;

        tracing::info!(
            "Discovering clusters with threshold={:.2} from {:?}",
            self.threshold,
            self.db_path
        );

        // Load graph from database (same pattern as QueryCommand)
        let graph: InMemoryGraph = if !self.db_path.exists() {
            tracing::warn!("Database not found at {:?}, using demo graph", self.db_path);
            create_demo_graph()
        } else {
            match SqliteGraphProvider::load(&self.db_path) {
                Ok(provider) => {
                    if provider.inner().node_count() == 0 {
                        tracing::warn!("Database graph is empty, using demo graph");
                        create_demo_graph()
                    } else {
                        provider.graph
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load graph from database: {}, using demo graph",
                        e
                    );
                    create_demo_graph()
                }
            }
        };

        // Convert to MinCutGraph and discover clusters
        let mcg = from_in_memory_graph(&graph);

        if mcg.node_count() == 0 {
            println!("No nodes in graph. Use 'nagual graph link' to create edges first.");
            return Ok(());
        }

        let clusters = mcg.discover_clusters(self.threshold);

        if clusters.is_empty() {
            println!("No clusters discovered (graph has no edges).");
            println!("Use `nagual graph link <source> <target>` to add edges first.");
            return Ok(());
        }

        println!("\nFound {} clusters:\n", clusters.len());

        for cluster in &clusters {
            println!(
                "Cluster #{} ({} patterns, weight: {:.1})",
                cluster.id + 1,
                cluster.node_ids.len(),
                cluster.internal_weight
            );

            if self.verbose {
                for node_id in &cluster.node_ids {
                    println!("  - {}", node_id);
                }
            } else {
                // Show up to 5 nodes, then summarize
                let show_count = cluster.node_ids.len().min(5);
                for node_id in cluster.node_ids.iter().take(show_count) {
                    println!("  - {}", node_id);
                }
                if cluster.node_ids.len() > 5 {
                    println!("  ... and {} more", cluster.node_ids.len() - 5);
                }
            }
            println!();
        }

        Ok(())
    }
}

/// Create a demo graph for testing
fn create_demo_graph() -> InMemoryGraph {
    let mut graph = InMemoryGraph::new();

    // Create a sample knowledge graph structure
    // Main concepts
    graph.add_edge("rust", "memory-safety", 0.95);
    graph.add_edge("rust", "ownership", 0.90);
    graph.add_edge("rust", "concurrency", 0.85);
    graph.add_edge("rust", "performance", 0.80);
    graph.add_edge("rust", "cargo", 0.75);

    // Memory safety subtopics
    graph.add_edge("memory-safety", "borrowing", 0.90);
    graph.add_edge("memory-safety", "lifetimes", 0.85);
    graph.add_edge("memory-safety", "no-gc", 0.70);

    // Ownership subtopics
    graph.add_edge("ownership", "borrowing", 0.85);
    graph.add_edge("ownership", "move-semantics", 0.80);
    graph.add_edge("ownership", "raii", 0.75);

    // Concurrency subtopics
    graph.add_edge("concurrency", "threads", 0.85);
    graph.add_edge("concurrency", "async-await", 0.80);
    graph.add_edge("concurrency", "channels", 0.75);
    graph.add_edge("concurrency", "mutex", 0.70);

    // Cross-connections
    graph.add_edge("borrowing", "lifetimes", 0.80);
    graph.add_edge("async-await", "tokio", 0.90);
    graph.add_edge("cargo", "dependencies", 0.85);
    graph.add_edge("cargo", "build-system", 0.80);

    graph
}

// JSON output structures
#[derive(Serialize)]
struct PressureJsonOutput {
    source_node: String,
    scores: Vec<NodeScore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stats: Option<StatsOutput>,
    config: ConfigOutput,
}

#[derive(Serialize)]
struct NodeScore {
    node_id: String,
    pressure: f64,
}

#[derive(Serialize)]
struct StatsOutput {
    iterations_used: usize,
    converged: bool,
    final_delta: f64,
    nodes_reached: usize,
    total_pressure: f64,
    execution_time_us: u64,
}

#[derive(Serialize)]
struct ConfigOutput {
    damping_factor: f64,
    max_iterations: usize,
    epsilon: f64,
    normalize: bool,
}

#[derive(Serialize)]
struct ErrorOutput {
    error: String,
    error_type: String,
}

#[derive(Serialize)]
struct LinkOutput {
    source: String,
    target: String,
    weight: f64,
    label: Option<String>,
    bidirectional: bool,
    success: bool,
    message: String,
}

#[derive(Serialize)]
struct QueryResult {
    node: String,
    direction: String,
    neighbor_count: usize,
    neighbors: Vec<NeighborInfo>,
}

#[derive(Serialize)]
struct NeighborInfo {
    node_id: String,
    direction: String,
    weight: f64,
    depth: usize,
}

#[derive(Serialize)]
struct QueryErrorOutput {
    error: String,
    error_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_graph_creation() {
        let graph = create_demo_graph();
        assert!(graph.node_exists("rust"));
        assert!(graph.node_exists("memory-safety"));
        assert!(graph.node_exists("ownership"));
    }

    #[test]
    fn test_demo_graph_pressure() {
        let graph = create_demo_graph();
        let config = PressureConfig::default();

        let result = propagate_pressure(&graph, "rust", &config).unwrap();

        // Rust should be the source with highest initial pressure
        assert!(result.pressure_scores.contains_key("rust"));

        // Direct neighbors should have pressure
        assert!(result.pressure_scores.contains_key("memory-safety"));
        assert!(result.pressure_scores.contains_key("ownership"));
        assert!(result.pressure_scores.contains_key("concurrency"));
    }

    #[test]
    fn test_pressure_command_config() {
        let cmd = PressureCommand {
            node_id: "test".to_string(),
            depth: 5,
            damping: 0.9,
            epsilon: 1e-8,
            top: 10,
            normalize: true,
            db_path: PathBuf::from("./test.db"),
            json: false,
            demo: true,
            stats: true,
        };

        assert_eq!(cmd.depth, 5);
        assert_eq!(cmd.damping, 0.9);
        assert_eq!(cmd.top, 10);
        assert!(cmd.normalize);
        assert!(cmd.demo);
        assert!(cmd.stats);
    }

    #[tokio::test]
    async fn test_pressure_command_demo_mode() {
        let cmd = PressureCommand {
            node_id: "rust".to_string(),
            depth: 3,
            damping: 0.85,
            epsilon: 1e-6,
            top: 5,
            normalize: false,
            db_path: PathBuf::from("./test.db"),
            json: false,
            demo: true,
            stats: false,
        };

        // Should not error with demo mode
        let result = cmd.run().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_pressure_command_invalid_node() {
        let cmd = PressureCommand {
            node_id: "nonexistent-node-xyz".to_string(),
            depth: 3,
            damping: 0.85,
            epsilon: 1e-6,
            top: 5,
            normalize: false,
            db_path: PathBuf::from("./test.db"),
            json: false,
            demo: true,
            stats: false,
        };

        // Should handle error gracefully
        let result = cmd.run().await;
        assert!(result.is_ok()); // Errors are handled internally
    }
}
