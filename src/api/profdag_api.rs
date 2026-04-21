//! ProfDAG namespace API for graph-based knowledge operations.
//!
//! The ProfDAG API provides methods for working with the Probabilistic
//! Forecasting DAG, including node/edge management, HNSW-powered search,
//! and trajectory recording.
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::profdag::{ProfDAGNode, ProfDAGEdge, NodeType, EdgeType};
//!
//! // Insert a node
//! let node = ProfDAGNode::new(NodeType::Pattern, "How to handle timeouts")
//!     .with_confidence(0.85);
//! let node_id = nagual.profdag.insert_node(&node).await?;
//!
//! // Get a node
//! let node = nagual.profdag.get_node(&node_id).await?;
//!
//! // Get ProfDAG statistics
//! let stats = nagual.profdag.stats().await?;
//!
//! // Access the trajectory recorder
//! let recorder = nagual.profdag.recorder();
//! ```

use std::sync::Arc;

use tracing::{debug, info, instrument};

use super::NagualState;
use crate::error::{NagualError, Result};
use crate::profdag::{
    ProfDAGEdge, ProfDAGNode, ProfDAGSearch, ProfDAGStats, ProfDAGStorage,
    RecorderConfig, SearchConfig, TrajectoryRecorder,
};

/// API for ProfDAG knowledge graph operations.
///
/// This API provides access to the Probabilistic Forecasting DAG,
/// including node/edge management, HNSW-powered vector search, and
/// trajectory recording for reasoning path capture.
pub struct ProfDAGApi {
    storage: Arc<ProfDAGStorage>,
    search: ProfDAGSearch,
    recorder: Arc<TrajectoryRecorder>,
}

impl ProfDAGApi {
    /// Create a new ProfDAGApi from shared Nagual state.
    pub(crate) async fn new(state: NagualState) -> Result<Self> {
        let storage = Arc::new(
            ProfDAGStorage::with_defaults(state.adapter.clone())
                .await
                .map_err(|e| NagualError::internal(e.to_string()))?,
        );

        let recorder = Arc::new(TrajectoryRecorder::with_storage(
            RecorderConfig::default(),
            storage.clone(),
        ));

        let search = ProfDAGSearch::new(storage.clone(), SearchConfig::default());

        Ok(Self {
            storage,
            search,
            recorder,
        })
    }

    /// Get the ProfDAG storage for direct operations.
    pub fn storage(&self) -> &Arc<ProfDAGStorage> {
        &self.storage
    }

    /// Get the HNSW search engine.
    pub fn search(&self) -> &ProfDAGSearch {
        &self.search
    }

    /// Get the trajectory recorder.
    pub fn recorder(&self) -> &Arc<TrajectoryRecorder> {
        &self.recorder
    }

    /// Insert a node into the ProfDAG.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to insert
    ///
    /// # Returns
    ///
    /// The ID of the inserted node.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nagual::profdag::{ProfDAGNode, NodeType};
    ///
    /// let node = ProfDAGNode::new(NodeType::Pattern, "Error handling pattern")
    ///     .with_confidence(0.9);
    /// let id = nagual.profdag.insert_node(&node).await?;
    /// ```
    #[instrument(skip(self, node))]
    pub async fn insert_node(&self, node: &ProfDAGNode) -> Result<String> {
        let id = self
            .storage
            .insert_node(node)
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?;

        debug!(id = %id, "ProfDAG node inserted");
        Ok(id)
    }

    /// Get a node by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The node ID to look up
    ///
    /// # Returns
    ///
    /// The node if found, or None.
    #[instrument(skip(self))]
    pub async fn get_node(&self, id: &str) -> Result<Option<ProfDAGNode>> {
        self.storage
            .get_node(id)
            .await
            .map_err(|e| NagualError::internal(e.to_string()))
    }

    /// Insert an edge into the ProfDAG.
    ///
    /// # Arguments
    ///
    /// * `edge` - The edge to insert
    ///
    /// # Returns
    ///
    /// The ID of the inserted edge.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nagual::profdag::{ProfDAGEdge, EdgeType};
    ///
    /// let edge = ProfDAGEdge::new(&source_id, &target_id, EdgeType::LeadsTo, 0.8);
    /// let id = nagual.profdag.insert_edge(&edge).await?;
    /// ```
    #[instrument(skip(self, edge))]
    pub async fn insert_edge(&self, edge: &ProfDAGEdge) -> Result<String> {
        let id = self
            .storage
            .insert_edge(edge)
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?;

        debug!(id = %id, "ProfDAG edge inserted");
        Ok(id)
    }

    /// Get ProfDAG statistics.
    ///
    /// # Returns
    ///
    /// Statistics including node count, edge count, and type distributions.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let stats = nagual.profdag.stats().await?;
    /// println!("Nodes: {}, Edges: {}", stats.node_count, stats.edge_count);
    /// ```
    #[instrument(skip(self))]
    pub async fn stats(&self) -> Result<ProfDAGStats> {
        let stats = self
            .storage
            .stats()
            .await
            .map_err(|e| NagualError::internal(e.to_string()))?;

        info!(
            nodes = stats.node_count,
            edges = stats.edge_count,
            "ProfDAG stats retrieved"
        );
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profdag::SearchConfig;

    #[test]
    fn test_search_config_defaults() {
        let config = SearchConfig::default();
        // Tuned HNSW defaults (post-ProfDAG)
        assert_eq!(config.hnsw_m, 24);
        assert_eq!(config.hnsw_ef_construction, 200);
        assert_eq!(config.hnsw_ef_search, 200);
    }
}
