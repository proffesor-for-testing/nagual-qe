//! Edge creation and management for the context graph.
//!
//! Provides the `GraphStorage` struct for managing edges in SQLite,
//! with support for upsert operations and edge querying.

use chrono::{DateTime, Utc};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{EdgeType, GraphEdge, GraphError};

/// Configuration for graph storage.
#[derive(Debug, Clone)]
pub struct GraphStorageConfig {
    /// Path to the SQLite database file.
    pub path: String,
    /// Whether to enable WAL mode.
    pub wal_mode: bool,
    /// Busy timeout in milliseconds.
    pub busy_timeout_ms: u32,
}

impl Default for GraphStorageConfig {
    fn default() -> Self {
        Self {
            path: "nagual.db".to_string(),
            wal_mode: true,
            busy_timeout_ms: 5000,
        }
    }
}

impl GraphStorageConfig {
    /// Create a new config with the given path.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }

    /// Create config for in-memory database (testing).
    pub fn in_memory() -> Self {
        Self {
            path: ":memory:".to_string(),
            wal_mode: false,
            busy_timeout_ms: 5000,
        }
    }
}

/// Result of edge creation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCreateResult {
    /// Whether the edge was created (true) or updated (false).
    pub created: bool,
    /// The edge ID.
    pub edge_id: String,
    /// Previous strength if edge was updated.
    pub previous_strength: Option<f64>,
}

/// Statistics about the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of edges.
    pub edge_count: usize,
    /// Number of unique nodes.
    pub node_count: usize,
    /// Edge counts by type.
    pub edges_by_type: Vec<(String, usize)>,
}

/// Graph storage for managing edges in SQLite.
pub struct GraphStorage {
    conn: Arc<RwLock<Connection>>,
    config: GraphStorageConfig,
}

impl GraphStorage {
    /// Open or create a graph storage at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GraphError> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let config = GraphStorageConfig::new(&path_str);
        Self::with_config(config)
    }

    /// Create a graph storage with the given configuration.
    pub fn with_config(config: GraphStorageConfig) -> Result<Self, GraphError> {
        let conn = if config.path == ":memory:" {
            Connection::open_in_memory()?
        } else {
            Connection::open(&config.path)?
        };

        // Configure SQLite
        let mut pragmas = vec![format!(
            "PRAGMA busy_timeout = {};",
            config.busy_timeout_ms
        )];

        if config.wal_mode && config.path != ":memory:" {
            pragmas.push("PRAGMA journal_mode = WAL;".to_string());
            pragmas.push("PRAGMA synchronous = NORMAL;".to_string());
        }

        conn.execute_batch(&pragmas.join("\n"))?;

        // Create tables
        conn.execute_batch(super::SQLITE_CONTEXT_GRAPH_TABLE)?;

        Ok(Self {
            conn: Arc::new(RwLock::new(conn)),
            config,
        })
    }

    /// Create a graph storage for testing (in-memory).
    pub fn for_testing() -> Result<Self, GraphError> {
        Self::with_config(GraphStorageConfig::in_memory())
    }

    /// Get the database path.
    pub fn path(&self) -> &str {
        &self.config.path
    }

    /// Create or update an edge.
    pub async fn create_edge(
        &self,
        source_id: &str,
        target_id: &str,
        edge_type: EdgeType,
        strength: f64,
        metadata: Option<serde_json::Value>,
    ) -> Result<EdgeCreateResult, GraphError> {
        if source_id == target_id {
            return Err(GraphError::SelfLoop(source_id.to_string()));
        }

        if !(0.0..=1.0).contains(&strength) {
            return Err(GraphError::InvalidStrength(strength));
        }

        let edge = GraphEdge::with_metadata(
            source_id,
            target_id,
            edge_type,
            strength,
            metadata.unwrap_or(serde_json::Value::Null),
        );

        let conn = self.conn.write().await;

        // Check if edge exists
        let existing: Option<(String, f64)> = conn
            .query_row(
                "SELECT id, strength FROM context_graph WHERE source_id = ? AND target_id = ? AND edge_type = ?",
                [source_id, target_id, edge_type.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .ok();

        if let Some((existing_id, prev_strength)) = existing {
            // Update existing edge
            conn.execute(
                "UPDATE context_graph SET strength = ?, metadata = ?, updated_at = ? WHERE id = ?",
                (
                    strength,
                    edge.metadata.as_ref().map(|m| m.to_string()),
                    Utc::now().to_rfc3339(),
                    &existing_id,
                ),
            )?;

            Ok(EdgeCreateResult {
                created: false,
                edge_id: existing_id,
                previous_strength: Some(prev_strength),
            })
        } else {
            // Insert new edge
            conn.execute(
                "INSERT INTO context_graph (id, source_id, target_id, edge_type, strength, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    &edge.id,
                    source_id,
                    target_id,
                    edge_type.as_str(),
                    strength,
                    edge.metadata.as_ref().map(|m| m.to_string()),
                    edge.created_at.to_rfc3339(),
                ),
            )?;

            Ok(EdgeCreateResult {
                created: true,
                edge_id: edge.id,
                previous_strength: None,
            })
        }
    }

    /// Get an edge by ID.
    pub async fn get_edge(&self, edge_id: &str) -> Result<Option<GraphEdge>, GraphError> {
        let conn = self.conn.read().await;

        let result: Option<GraphEdge> = conn
            .query_row(
                "SELECT id, source_id, target_id, edge_type, strength, metadata, created_at, updated_at FROM context_graph WHERE id = ?",
                [edge_id],
                |row| {
                    let edge_type_str: String = row.get(3)?;
                    let metadata_str: Option<String> = row.get(5)?;
                    let created_at_str: String = row.get(6)?;
                    let updated_at_str: Option<String> = row.get(7)?;

                    Ok(GraphEdge {
                        id: row.get(0)?,
                        source_id: row.get(1)?,
                        target_id: row.get(2)?,
                        edge_type: EdgeType::from_str(&edge_type_str).unwrap_or(EdgeType::RelatedTo),
                        strength: row.get(4)?,
                        metadata: metadata_str.and_then(|s| serde_json::from_str(&s).ok()),
                        created_at: DateTime::parse_from_rfc3339(&created_at_str)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                        updated_at: updated_at_str.and_then(|s| {
                            DateTime::parse_from_rfc3339(&s)
                                .map(|dt| dt.with_timezone(&Utc))
                                .ok()
                        }),
                    })
                },
            )
            .ok();

        Ok(result)
    }

    /// Delete an edge by ID.
    pub async fn delete_edge(&self, edge_id: &str) -> Result<bool, GraphError> {
        let conn = self.conn.write().await;
        let rows = conn.execute("DELETE FROM context_graph WHERE id = ?", [edge_id])?;
        Ok(rows > 0)
    }

    /// Get graph statistics.
    pub async fn stats(&self) -> Result<GraphStats, GraphError> {
        let conn = self.conn.read().await;

        let edge_count: usize = conn.query_row(
            "SELECT COUNT(*) FROM context_graph",
            [],
            |row| row.get(0),
        )?;

        let node_count: usize = conn.query_row(
            "SELECT COUNT(DISTINCT source_id) + COUNT(DISTINCT target_id) FROM context_graph",
            [],
            |row| row.get(0),
        )?;

        let mut stmt = conn.prepare(
            "SELECT edge_type, COUNT(*) FROM context_graph GROUP BY edge_type",
        )?;
        let edges_by_type: Vec<(String, usize)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(GraphStats {
            edge_count,
            node_count,
            edges_by_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_edge() {
        let storage = GraphStorage::for_testing().unwrap();
        let result = storage
            .create_edge("a", "b", EdgeType::RelatedTo, 0.8, None)
            .await
            .unwrap();

        assert!(result.created);
        assert!(result.previous_strength.is_none());
    }

    #[tokio::test]
    async fn test_update_edge() {
        let storage = GraphStorage::for_testing().unwrap();

        // Create edge
        storage
            .create_edge("a", "b", EdgeType::RelatedTo, 0.5, None)
            .await
            .unwrap();

        // Update edge
        let result = storage
            .create_edge("a", "b", EdgeType::RelatedTo, 0.9, None)
            .await
            .unwrap();

        assert!(!result.created);
        assert_eq!(result.previous_strength, Some(0.5));
    }

    #[tokio::test]
    async fn test_self_loop_rejected() {
        let storage = GraphStorage::for_testing().unwrap();
        let result = storage
            .create_edge("a", "a", EdgeType::RelatedTo, 0.8, None)
            .await;

        assert!(matches!(result, Err(GraphError::SelfLoop(_))));
    }

    #[tokio::test]
    async fn test_invalid_strength_rejected() {
        let storage = GraphStorage::for_testing().unwrap();
        let result = storage
            .create_edge("a", "b", EdgeType::RelatedTo, 1.5, None)
            .await;

        assert!(matches!(result, Err(GraphError::InvalidStrength(_))));
    }

    #[tokio::test]
    async fn test_get_edge() {
        let storage = GraphStorage::for_testing().unwrap();
        let create_result = storage
            .create_edge("a", "b", EdgeType::SimilarTo, 0.85, None)
            .await
            .unwrap();

        let edge = storage.get_edge(&create_result.edge_id).await.unwrap();
        assert!(edge.is_some());

        let edge = edge.unwrap();
        assert_eq!(edge.source_id, "a");
        assert_eq!(edge.target_id, "b");
        assert_eq!(edge.edge_type, EdgeType::SimilarTo);
    }

    #[tokio::test]
    async fn test_delete_edge() {
        let storage = GraphStorage::for_testing().unwrap();
        let create_result = storage
            .create_edge("a", "b", EdgeType::RelatedTo, 0.5, None)
            .await
            .unwrap();

        let deleted = storage.delete_edge(&create_result.edge_id).await.unwrap();
        assert!(deleted);

        let edge = storage.get_edge(&create_result.edge_id).await.unwrap();
        assert!(edge.is_none());
    }

    #[tokio::test]
    async fn test_stats() {
        let storage = GraphStorage::for_testing().unwrap();

        storage
            .create_edge("a", "b", EdgeType::RelatedTo, 0.5, None)
            .await
            .unwrap();
        storage
            .create_edge("b", "c", EdgeType::SimilarTo, 0.8, None)
            .await
            .unwrap();

        let stats = storage.stats().await.unwrap();
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.edges_by_type.len(), 2);
    }
}
