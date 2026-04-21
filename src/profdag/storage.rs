//! ProfDAG Storage layer with dual-write support.
//!
//! Provides CRUD operations for ProfDAG nodes and edges with support for
//! both SQLite and PostgreSQL backends following the dual-write pattern.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use super::profiler::{OperationType, ProfDAGProfiler};
use super::{
    EdgeType, NodeType, ProfDAGEdge, ProfDAGError, ProfDAGNode, ProfDAGResult,
    TemporalDirection, SQLITE_PROFDAG_EDGES_TABLE, SQLITE_PROFDAG_NODES_TABLE,
};
use crate::db::DualWriteAdapter;

/// Configuration for ProfDAG storage.
#[derive(Debug, Clone)]
pub struct ProfDAGStorageConfig {
    /// Expected embedding dimension (default: 128).
    pub embedding_dim: usize,

    /// Whether to enforce DAG constraints (no cycles).
    pub enforce_dag: bool,

    /// Minimum similarity score for automatic similar_to edges.
    pub similarity_threshold: f64,

    /// Maximum number of similar_to edges per node.
    pub max_similar_edges: usize,
}

impl Default for ProfDAGStorageConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 128,
            enforce_dag: true,
            similarity_threshold: 0.7,
            max_similar_edges: 10,
        }
    }
}

/// Query parameters for neighbor lookups.
#[derive(Debug, Clone, Default)]
pub struct NeighborQuery {
    /// Filter by edge types (empty = all types).
    pub edge_types: Vec<EdgeType>,

    /// Minimum edge weight.
    pub min_weight: Option<f64>,

    /// Maximum number of results.
    pub limit: Option<usize>,

    /// Include incoming edges (target = this node).
    pub include_incoming: bool,

    /// Include outgoing edges (source = this node).
    pub include_outgoing: bool,
}

impl NeighborQuery {
    /// Create a new query for outgoing edges.
    pub fn outgoing() -> Self {
        Self {
            include_outgoing: true,
            include_incoming: false,
            ..Default::default()
        }
    }

    /// Create a new query for incoming edges.
    pub fn incoming() -> Self {
        Self {
            include_outgoing: false,
            include_incoming: true,
            ..Default::default()
        }
    }

    /// Create a new query for both directions.
    pub fn both() -> Self {
        Self {
            include_outgoing: true,
            include_incoming: true,
            ..Default::default()
        }
    }

    /// Filter by edge type.
    pub fn with_edge_type(mut self, edge_type: EdgeType) -> Self {
        self.edge_types.push(edge_type);
        self
    }

    /// Set minimum weight.
    pub fn with_min_weight(mut self, min_weight: f64) -> Self {
        self.min_weight = Some(min_weight);
        self
    }

    /// Set limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Result of a neighbor query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborResult {
    /// The neighboring node.
    pub node: ProfDAGNode,

    /// The edge connecting to this neighbor.
    pub edge: ProfDAGEdge,

    /// Whether this is an incoming edge (true) or outgoing (false).
    pub is_incoming: bool,
}

/// Result of a similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarNode {
    /// The similar node.
    pub node: ProfDAGNode,

    /// Cosine similarity score.
    pub similarity: f64,
}

/// Statistics about the ProfDAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfDAGStats {
    /// Total number of nodes.
    pub node_count: usize,

    /// Number of nodes by type.
    pub nodes_by_type: Vec<(String, usize)>,

    /// Total number of edges.
    pub edge_count: usize,

    /// Number of edges by type.
    pub edges_by_type: Vec<(String, usize)>,

    /// Average edges per node.
    pub avg_edges_per_node: f64,

    /// Number of nodes with embeddings.
    pub nodes_with_embeddings: usize,
}

/// ProfDAG storage with dual-write capability.
pub struct ProfDAGStorage {
    /// The dual-write adapter for synchronized persistence.
    adapter: Arc<DualWriteAdapter>,

    /// Storage configuration.
    config: ProfDAGStorageConfig,

    /// Optional profiler for recording operation timings.
    profiler: Option<Arc<ProfDAGProfiler>>,
}

impl ProfDAGStorage {
    /// Create a new ProfDAGStorage.
    pub async fn new(
        adapter: Arc<DualWriteAdapter>,
        config: ProfDAGStorageConfig,
    ) -> ProfDAGResult<Self> {
        let storage = Self {
            adapter,
            config,
            profiler: None,
        };

        // Initialize the database schema
        storage.init_schema().await?;

        Ok(storage)
    }

    /// Create with default configuration.
    pub async fn with_defaults(adapter: Arc<DualWriteAdapter>) -> ProfDAGResult<Self> {
        Self::new(adapter, ProfDAGStorageConfig::default()).await
    }

    /// Attach an optional profiler for recording operation timings.
    /// Note: This consumes and returns self because ProfDAGStorage is typically
    /// wrapped in Arc after construction.
    pub fn with_profiler(mut self, profiler: Arc<ProfDAGProfiler>) -> Self {
        self.profiler = Some(profiler);
        self
    }

    /// Initialize the database schema.
    async fn init_schema(&self) -> ProfDAGResult<()> {
        self.adapter
            .sqlite()
            .execute_batch(SQLITE_PROFDAG_NODES_TABLE)
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        self.adapter
            .sqlite()
            .execute_batch(SQLITE_PROFDAG_EDGES_TABLE)
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        info!("ProfDAG schema initialized");
        Ok(())
    }

    // ========================================================================
    // Node Operations
    // ========================================================================

    /// Insert a new node.
    pub async fn insert_node(&self, node: &ProfDAGNode) -> ProfDAGResult<String> {
        let _guard = self.profiler.as_ref().map(|p| p.start_operation(OperationType::StorageWrite));
        self.validate_node(node)?;

        let embedding_json = match &node.embedding {
            Some(e) => Some(serde_json::to_string(e)?),
            None => None,
        };

        let metadata_json = serde_json::to_string(&node.metadata)?;

        let sql = r#"
            INSERT INTO profdag_nodes (
                id, node_type, content, embedding, metadata,
                source_id, source_type, confidence, importance,
                agent_id, session_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        self.adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &node.id,
                    &node.node_type.as_str(),
                    &node.content,
                    &embedding_json as &dyn rusqlite::ToSql,
                    &metadata_json,
                    &node.source_id as &dyn rusqlite::ToSql,
                    &node.source_type as &dyn rusqlite::ToSql,
                    &(node.confidence as f64),
                    &(node.importance as f64),
                    &node.agent_id as &dyn rusqlite::ToSql,
                    &node.session_id as &dyn rusqlite::ToSql,
                    &node.created_at.to_rfc3339(),
                    &node.updated_at.map(|t| t.to_rfc3339()) as &dyn rusqlite::ToSql,
                ],
            )
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        debug!(node_id = %node.id, node_type = %node.node_type, "Node inserted");
        Ok(node.id.clone())
    }

    /// Get a node by ID.
    pub async fn get_node(&self, id: &str) -> ProfDAGResult<Option<ProfDAGNode>> {
        let _guard = self.profiler.as_ref().map(|p| p.start_operation(OperationType::StorageRead));
        let sql = "SELECT * FROM profdag_nodes WHERE id = ?";

        let node = self
            .adapter
            .sqlite()
            .query_one(sql, &[&id], |row| Self::node_from_row(row))
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        Ok(node)
    }

    /// Update an existing node.
    pub async fn update_node(&self, node: &ProfDAGNode) -> ProfDAGResult<()> {
        self.validate_node(node)?;

        let embedding_json = match &node.embedding {
            Some(e) => Some(serde_json::to_string(e)?),
            None => None,
        };

        let metadata_json = serde_json::to_string(&node.metadata)?;
        let updated_at = Utc::now().to_rfc3339();

        let sql = r#"
            UPDATE profdag_nodes SET
                node_type = ?,
                content = ?,
                embedding = ?,
                metadata = ?,
                source_id = ?,
                source_type = ?,
                confidence = ?,
                importance = ?,
                agent_id = ?,
                session_id = ?,
                updated_at = ?
            WHERE id = ?
        "#;

        let rows = self
            .adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &node.node_type.as_str(),
                    &node.content,
                    &embedding_json as &dyn rusqlite::ToSql,
                    &metadata_json,
                    &node.source_id as &dyn rusqlite::ToSql,
                    &node.source_type as &dyn rusqlite::ToSql,
                    &(node.confidence as f64),
                    &(node.importance as f64),
                    &node.agent_id as &dyn rusqlite::ToSql,
                    &node.session_id as &dyn rusqlite::ToSql,
                    &updated_at,
                    &node.id,
                ],
            )
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        if rows == 0 {
            return Err(ProfDAGError::NodeNotFound { id: node.id.clone() });
        }

        debug!(node_id = %node.id, "Node updated");
        Ok(())
    }

    /// Delete a node by ID.
    ///
    /// This also deletes all edges connected to this node (via CASCADE).
    pub async fn delete_node(&self, id: &str) -> ProfDAGResult<bool> {
        let sql = "DELETE FROM profdag_nodes WHERE id = ?";

        let rows = self
            .adapter
            .sqlite()
            .execute(sql, &[&id])
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        if rows > 0 {
            info!(node_id = %id, "Node deleted");
        }

        Ok(rows > 0)
    }

    /// Get nodes by type.
    pub async fn get_nodes_by_type(
        &self,
        node_type: NodeType,
        limit: usize,
    ) -> ProfDAGResult<Vec<ProfDAGNode>> {
        let sql = r#"
            SELECT * FROM profdag_nodes
            WHERE node_type = ?
            ORDER BY created_at DESC
            LIMIT ?
        "#;

        let nodes = self
            .adapter
            .sqlite()
            .query(sql, &[&node_type.as_str(), &(limit as i64)], |row| {
                Self::node_from_row(row)
            })
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        Ok(nodes)
    }

    /// Get nodes by source reference.
    pub async fn get_nodes_by_source(
        &self,
        source_type: &str,
        source_id: &str,
    ) -> ProfDAGResult<Vec<ProfDAGNode>> {
        let sql = r#"
            SELECT * FROM profdag_nodes
            WHERE source_type = ? AND source_id = ?
            ORDER BY created_at DESC
        "#;

        let nodes = self
            .adapter
            .sqlite()
            .query(sql, &[&source_type, &source_id], |row| {
                Self::node_from_row(row)
            })
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        Ok(nodes)
    }

    // ========================================================================
    // Edge Operations
    // ========================================================================

    /// Insert a new edge.
    pub async fn insert_edge(&self, edge: &ProfDAGEdge) -> ProfDAGResult<String> {
        let _guard = self.profiler.as_ref().map(|p| p.start_operation(OperationType::StorageWrite));
        self.validate_edge(edge)?;

        let metadata_json = serde_json::to_string(&edge.metadata)?;

        let sql = r#"
            INSERT INTO profdag_edges (
                id, source_id, target_id, edge_type, weight, metadata,
                temporal_distance_hours, temporal_direction,
                similarity_score, wormhole_strength, wormhole_reason,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#;

        self.adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &edge.id,
                    &edge.source_id,
                    &edge.target_id,
                    &edge.edge_type.as_str(),
                    &edge.weight,
                    &metadata_json,
                    &edge.temporal_distance_hours as &dyn rusqlite::ToSql,
                    &edge.temporal_direction.map(|d| d.as_str()) as &dyn rusqlite::ToSql,
                    &edge.similarity_score as &dyn rusqlite::ToSql,
                    &edge.wormhole_strength as &dyn rusqlite::ToSql,
                    &edge.wormhole_reason as &dyn rusqlite::ToSql,
                    &edge.created_at.to_rfc3339(),
                    &edge.updated_at.map(|t| t.to_rfc3339()) as &dyn rusqlite::ToSql,
                ],
            )
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        debug!(
            edge_id = %edge.id,
            source = %edge.source_id,
            target = %edge.target_id,
            edge_type = %edge.edge_type,
            "Edge inserted"
        );
        Ok(edge.id.clone())
    }

    /// Get an edge by ID.
    pub async fn get_edge(&self, id: &str) -> ProfDAGResult<Option<ProfDAGEdge>> {
        let sql = "SELECT * FROM profdag_edges WHERE id = ?";

        let edge = self
            .adapter
            .sqlite()
            .query_one(sql, &[&id], |row| Self::edge_from_row(row))
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        Ok(edge)
    }

    /// Update an existing edge.
    pub async fn update_edge(&self, edge: &ProfDAGEdge) -> ProfDAGResult<()> {
        self.validate_edge(edge)?;

        let metadata_json = serde_json::to_string(&edge.metadata)?;
        let updated_at = Utc::now().to_rfc3339();

        let sql = r#"
            UPDATE profdag_edges SET
                weight = ?,
                metadata = ?,
                temporal_distance_hours = ?,
                temporal_direction = ?,
                similarity_score = ?,
                wormhole_strength = ?,
                wormhole_reason = ?,
                updated_at = ?
            WHERE id = ?
        "#;

        let rows = self
            .adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &edge.weight,
                    &metadata_json,
                    &edge.temporal_distance_hours as &dyn rusqlite::ToSql,
                    &edge.temporal_direction.map(|d| d.as_str()) as &dyn rusqlite::ToSql,
                    &edge.similarity_score as &dyn rusqlite::ToSql,
                    &edge.wormhole_strength as &dyn rusqlite::ToSql,
                    &edge.wormhole_reason as &dyn rusqlite::ToSql,
                    &updated_at,
                    &edge.id,
                ],
            )
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        if rows == 0 {
            return Err(ProfDAGError::EdgeNotFound { id: edge.id.clone() });
        }

        debug!(edge_id = %edge.id, "Edge updated");
        Ok(())
    }

    /// Delete an edge by ID.
    pub async fn delete_edge(&self, id: &str) -> ProfDAGResult<bool> {
        let sql = "DELETE FROM profdag_edges WHERE id = ?";

        let rows = self
            .adapter
            .sqlite()
            .execute(sql, &[&id])
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        if rows > 0 {
            info!(edge_id = %id, "Edge deleted");
        }

        Ok(rows > 0)
    }

    /// Get or create an edge (upsert).
    pub async fn upsert_edge(&self, edge: &ProfDAGEdge) -> ProfDAGResult<String> {
        self.validate_edge(edge)?;

        let metadata_json = serde_json::to_string(&edge.metadata)?;

        let sql = r#"
            INSERT INTO profdag_edges (
                id, source_id, target_id, edge_type, weight, metadata,
                temporal_distance_hours, temporal_direction,
                similarity_score, wormhole_strength, wormhole_reason,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                weight = excluded.weight,
                metadata = excluded.metadata,
                temporal_distance_hours = excluded.temporal_distance_hours,
                temporal_direction = excluded.temporal_direction,
                similarity_score = excluded.similarity_score,
                wormhole_strength = excluded.wormhole_strength,
                wormhole_reason = excluded.wormhole_reason,
                updated_at = excluded.updated_at
        "#;

        self.adapter
            .sqlite()
            .execute(
                sql,
                &[
                    &edge.id,
                    &edge.source_id,
                    &edge.target_id,
                    &edge.edge_type.as_str(),
                    &edge.weight,
                    &metadata_json,
                    &edge.temporal_distance_hours as &dyn rusqlite::ToSql,
                    &edge.temporal_direction.map(|d| d.as_str()) as &dyn rusqlite::ToSql,
                    &edge.similarity_score as &dyn rusqlite::ToSql,
                    &edge.wormhole_strength as &dyn rusqlite::ToSql,
                    &edge.wormhole_reason as &dyn rusqlite::ToSql,
                    &edge.created_at.to_rfc3339(),
                    &edge.updated_at.map(|t| t.to_rfc3339()).unwrap_or_else(|| Utc::now().to_rfc3339()),
                ],
            )
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        debug!(
            edge_id = %edge.id,
            source = %edge.source_id,
            target = %edge.target_id,
            "Edge upserted"
        );
        Ok(edge.id.clone())
    }

    // ========================================================================
    // Graph Queries
    // ========================================================================

    /// Get neighbors of a node.
    pub async fn get_neighbors(
        &self,
        node_id: &str,
        query: &NeighborQuery,
    ) -> ProfDAGResult<Vec<NeighborResult>> {
        let _guard = self.profiler.as_ref().map(|p| p.start_operation(OperationType::StorageRead));
        let mut results = Vec::new();

        // Outgoing edges
        if query.include_outgoing {
            let outgoing = self.get_outgoing_neighbors(node_id, query).await?;
            results.extend(outgoing);
        }

        // Incoming edges
        if query.include_incoming {
            let incoming = self.get_incoming_neighbors(node_id, query).await?;
            results.extend(incoming);
        }

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Get outgoing neighbors (where node_id is the source).
    async fn get_outgoing_neighbors(
        &self,
        node_id: &str,
        query: &NeighborQuery,
    ) -> ProfDAGResult<Vec<NeighborResult>> {
        let mut sql = String::from(
            r#"
            SELECT e.*, n.*
            FROM profdag_edges e
            JOIN profdag_nodes n ON e.target_id = n.id
            WHERE e.source_id = ?
        "#,
        );

        let mut params: Vec<Box<dyn rusqlite::ToSql + Send + Sync>> =
            vec![Box::new(node_id.to_string())];

        if !query.edge_types.is_empty() {
            let type_placeholders: Vec<&str> = query.edge_types.iter().map(|t| t.as_str()).collect();
            let placeholders = vec!["?"; type_placeholders.len()].join(", ");
            sql.push_str(&format!(" AND e.edge_type IN ({})", placeholders));
            for t in type_placeholders {
                params.push(Box::new(t.to_string()));
            }
        }

        if let Some(min_weight) = query.min_weight {
            sql.push_str(" AND e.weight >= ?");
            params.push(Box::new(min_weight));
        }

        sql.push_str(" ORDER BY e.weight DESC");

        if let Some(limit) = query.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        let param_refs: Vec<&dyn rusqlite::ToSql> =
            params.iter().map(|p| p.as_ref() as &dyn rusqlite::ToSql).collect();

        let results = self
            .adapter
            .sqlite()
            .query(&sql, &param_refs, |row| {
                let edge = Self::edge_from_row(row)?;
                let node = Self::node_from_row_offset(row, 13)?; // 13 edge columns
                Ok(NeighborResult {
                    node,
                    edge,
                    is_incoming: false,
                })
            })
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        Ok(results)
    }

    /// Get incoming neighbors (where node_id is the target).
    async fn get_incoming_neighbors(
        &self,
        node_id: &str,
        query: &NeighborQuery,
    ) -> ProfDAGResult<Vec<NeighborResult>> {
        let mut sql = String::from(
            r#"
            SELECT e.*, n.*
            FROM profdag_edges e
            JOIN profdag_nodes n ON e.source_id = n.id
            WHERE e.target_id = ?
        "#,
        );

        let mut params: Vec<Box<dyn rusqlite::ToSql + Send + Sync>> =
            vec![Box::new(node_id.to_string())];

        if !query.edge_types.is_empty() {
            let type_placeholders: Vec<&str> = query.edge_types.iter().map(|t| t.as_str()).collect();
            let placeholders = vec!["?"; type_placeholders.len()].join(", ");
            sql.push_str(&format!(" AND e.edge_type IN ({})", placeholders));
            for t in type_placeholders {
                params.push(Box::new(t.to_string()));
            }
        }

        if let Some(min_weight) = query.min_weight {
            sql.push_str(" AND e.weight >= ?");
            params.push(Box::new(min_weight));
        }

        sql.push_str(" ORDER BY e.weight DESC");

        if let Some(limit) = query.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        let param_refs: Vec<&dyn rusqlite::ToSql> =
            params.iter().map(|p| p.as_ref() as &dyn rusqlite::ToSql).collect();

        let results = self
            .adapter
            .sqlite()
            .query(&sql, &param_refs, |row| {
                let edge = Self::edge_from_row(row)?;
                let node = Self::node_from_row_offset(row, 13)?;
                Ok(NeighborResult {
                    node,
                    edge,
                    is_incoming: true,
                })
            })
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        Ok(results)
    }

    /// Get statistics about the ProfDAG.
    pub async fn stats(&self) -> ProfDAGResult<ProfDAGStats> {
        let node_count: usize = self
            .adapter
            .sqlite()
            .query_one("SELECT COUNT(*) FROM profdag_nodes", &[], |row| row.get(0))
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?
            .unwrap_or(0);

        let edge_count: usize = self
            .adapter
            .sqlite()
            .query_one("SELECT COUNT(*) FROM profdag_edges", &[], |row| row.get(0))
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?
            .unwrap_or(0);

        let nodes_by_type = self
            .adapter
            .sqlite()
            .query(
                "SELECT node_type, COUNT(*) FROM profdag_nodes GROUP BY node_type",
                &[],
                |row| {
                    let node_type: String = row.get(0)?;
                    let count: i64 = row.get(1)?;
                    Ok((node_type, count as usize))
                },
            )
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        let edges_by_type = self
            .adapter
            .sqlite()
            .query(
                "SELECT edge_type, COUNT(*) FROM profdag_edges GROUP BY edge_type",
                &[],
                |row| {
                    let edge_type: String = row.get(0)?;
                    let count: i64 = row.get(1)?;
                    Ok((edge_type, count as usize))
                },
            )
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?;

        let nodes_with_embeddings: usize = self
            .adapter
            .sqlite()
            .query_one(
                "SELECT COUNT(*) FROM profdag_nodes WHERE embedding IS NOT NULL",
                &[],
                |row| row.get(0),
            )
            .await
            .map_err(|e| ProfDAGError::Database(e.to_string()))?
            .unwrap_or(0);

        let avg_edges_per_node = if node_count > 0 {
            edge_count as f64 / node_count as f64
        } else {
            0.0
        };

        Ok(ProfDAGStats {
            node_count,
            nodes_by_type,
            edge_count,
            edges_by_type,
            avg_edges_per_node,
            nodes_with_embeddings,
        })
    }

    // ========================================================================
    // Validation
    // ========================================================================

    /// Validate a node before storage.
    fn validate_node(&self, node: &ProfDAGNode) -> ProfDAGResult<()> {
        if node.content.is_empty() {
            return Err(ProfDAGError::Database(
                "Node content cannot be empty".to_string(),
            ));
        }

        if let Some(ref embedding) = node.embedding {
            if embedding.len() != self.config.embedding_dim {
                return Err(ProfDAGError::DimensionMismatch {
                    expected: self.config.embedding_dim,
                    actual: embedding.len(),
                });
            }
        }

        Ok(())
    }

    /// Validate an edge before storage.
    fn validate_edge(&self, edge: &ProfDAGEdge) -> ProfDAGResult<()> {
        if edge.source_id == edge.target_id {
            return Err(ProfDAGError::SelfLoop(edge.source_id.clone()));
        }

        if !(0.0..=1.0).contains(&edge.weight) {
            return Err(ProfDAGError::InvalidWeight(edge.weight));
        }

        Ok(())
    }

    // ========================================================================
    // Row Mapping
    // ========================================================================

    /// Convert a database row to a ProfDAGNode.
    fn node_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProfDAGNode> {
        Self::node_from_row_offset(row, 0)
    }

    /// Convert a database row to a ProfDAGNode with column offset.
    fn node_from_row_offset(row: &rusqlite::Row<'_>, offset: usize) -> rusqlite::Result<ProfDAGNode> {
        let id: String = row.get(offset)?;
        let node_type_str: String = row.get(offset + 1)?;
        let content: String = row.get(offset + 2)?;
        let embedding_json: Option<String> = row.get(offset + 3)?;
        let metadata_json: String = row.get::<_, Option<String>>(offset + 4)?.unwrap_or_else(|| "{}".to_string());
        let source_id: Option<String> = row.get(offset + 5)?;
        let source_type: Option<String> = row.get(offset + 6)?;
        let confidence: f64 = row.get(offset + 7)?;
        let importance: f64 = row.get(offset + 8)?;
        let agent_id: Option<String> = row.get(offset + 9)?;
        let session_id: Option<String> = row.get(offset + 10)?;
        let created_at_str: String = row.get(offset + 11)?;
        let updated_at_str: Option<String> = row.get(offset + 12)?;

        let node_type = NodeType::from_str(&node_type_str)
            .ok_or_else(|| rusqlite::Error::FromSqlConversionFailure(
                offset + 1,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("Invalid node type: {}", node_type_str))),
            ))?;

        let embedding: Option<Vec<f32>> = embedding_json.and_then(|json| {
            match serde_json::from_str(&json) {
                Ok(e) => Some(e),
                Err(e) => {
                    warn!(node_id = %id, "Corrupted embedding JSON, treating as None: {}", e);
                    None
                }
            }
        });

        let metadata: serde_json::Value = match serde_json::from_str(&metadata_json) {
            Ok(m) => m,
            Err(e) => {
                warn!(node_id = %id, "Corrupted metadata JSON, using empty: {}", e);
                serde_json::json!({})
            }
        };

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let updated_at = updated_at_str.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .ok()
        });

        Ok(ProfDAGNode {
            id,
            node_type,
            content,
            embedding,
            metadata,
            source_id,
            source_type,
            confidence: confidence as f32,
            importance: importance as f32,
            agent_id,
            session_id,
            created_at,
            updated_at,
        })
    }

    /// Convert a database row to a ProfDAGEdge.
    fn edge_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProfDAGEdge> {
        let id: String = row.get(0)?;
        let source_id: String = row.get(1)?;
        let target_id: String = row.get(2)?;
        let edge_type_str: String = row.get(3)?;
        let weight: f64 = row.get(4)?;
        let metadata_json: String = row.get::<_, Option<String>>(5)?.unwrap_or_else(|| "{}".to_string());
        let temporal_distance_hours: Option<i32> = row.get(6)?;
        let temporal_direction_str: Option<String> = row.get(7)?;
        let similarity_score: Option<f64> = row.get(8)?;
        let wormhole_strength: Option<f64> = row.get(9)?;
        let wormhole_reason: Option<String> = row.get(10)?;
        let created_at_str: String = row.get(11)?;
        let updated_at_str: Option<String> = row.get(12)?;

        let edge_type = EdgeType::from_str(&edge_type_str)
            .ok_or_else(|| rusqlite::Error::FromSqlConversionFailure(
                3,
                rusqlite::types::Type::Text,
                Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("Invalid edge type: {}", edge_type_str))),
            ))?;

        let metadata: serde_json::Value = match serde_json::from_str(&metadata_json) {
            Ok(m) => m,
            Err(e) => {
                warn!(edge_id = %id, "Corrupted edge metadata JSON, using empty: {}", e);
                serde_json::json!({})
            }
        };

        let temporal_direction =
            temporal_direction_str.and_then(|s| TemporalDirection::from_str(&s));

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        let updated_at = updated_at_str.and_then(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .ok()
        });

        Ok(ProfDAGEdge {
            id,
            source_id,
            target_id,
            edge_type,
            weight,
            metadata,
            temporal_distance_hours,
            temporal_direction,
            similarity_score,
            wormhole_strength,
            wormhole_reason,
            created_at,
            updated_at,
        })
    }

    /// Get the storage configuration.
    pub fn config(&self) -> &ProfDAGStorageConfig {
        &self.config
    }

    /// Get the underlying adapter.
    pub fn adapter(&self) -> &Arc<DualWriteAdapter> {
        &self.adapter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbor_query_builder() {
        let query = NeighborQuery::outgoing()
            .with_edge_type(EdgeType::LeadsTo)
            .with_min_weight(0.5)
            .with_limit(10);

        assert!(query.include_outgoing);
        assert!(!query.include_incoming);
        assert_eq!(query.edge_types.len(), 1);
        assert_eq!(query.min_weight, Some(0.5));
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_storage_config_default() {
        let config = ProfDAGStorageConfig::default();
        assert_eq!(config.embedding_dim, 128);
        assert!(config.enforce_dag);
        assert_eq!(config.similarity_threshold, 0.7);
        assert_eq!(config.max_similar_edges, 10);
    }
}
