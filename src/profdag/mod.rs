//! ProfDAG - Probabilistic Forecasting Directed Acyclic Graph
//!
//! A graph-based knowledge representation system for tracking patterns,
//! trajectories, predictions, and decisions with temporal and semantic
//! relationships.
//!
//! # Overview
//!
//! ProfDAG provides a sophisticated DAG structure for:
//!
//! - **Pattern Evolution**: Track how patterns evolve over time
//! - **Trajectory Reasoning**: Connect sequences of actions/decisions
//! - **Decision Support**: Leverage wormhole connections for insights
//! - **Temporal Forecasting**: Use temporal links for prediction
//! - **Semantic Retrieval**: Find similar nodes via embedding search
//! - **Light Cone Reasoning**: Temporal causality via history/future cones
//!
//! # Node Types
//!
//! - `pattern`: A learned problem-solution pair from ReasoningBank
//! - `trajectory`: A sequence of actions/decisions in a session
//! - `prediction`: A probabilistic forecast from the prediction engine
//! - `decision`: A choice point with tracked outcomes
//!
//! # Edge Types
//!
//! - `leads_to`: Causal/temporal progression between nodes
//! - `similar_to`: Semantic similarity based on embeddings
//! - `derived_from`: One node was derived/generated from another
//! - `wormhole`: Non-local connection bridging distant graph regions
//! - `temporal_link`: Time-based relationship with direction
//!
//! # Light Cone Model
//!
//! The light cone model provides temporal reasoning capabilities:
//!
//! ```text
//!                    Future Cone
//!                   /           \
//!                  /  Predicted  \
//!                 /    Outcomes   \
//!                /        |        \
//!               +---------+---------+
//!               |     NOW (center)  |
//!               +---------+---------+
//!                \        |        /
//!                 \  Causal Past  /
//!                  \   Events    /
//!                   \           /
//!                    History Cone
//! ```
//!
//! - **History Cone**: Trace causal past ("What led to X?")
//! - **Future Cone**: Probabilistic predictions ("What might follow?")
//! - **Cognitive Core**: Active working set ("What's relevant now?")
//!
//! # Example
//!
//! ```rust,ignore
//! use nagual::profdag::{ProfDAGStorage, ProfDAGNode, ProfDAGEdge, NodeType, EdgeType};
//! use nagual::profdag::light_cone::{LightCone, LightConeConfig};
//!
//! // Create storage
//! let storage = ProfDAGStorage::new(adapter).await?;
//!
//! // Create a pattern node
//! let node = ProfDAGNode::new(NodeType::Pattern, "How to handle timeouts")
//!     .with_embedding(embedding)
//!     .with_confidence(0.85);
//!
//! let node_id = storage.insert_node(&node).await?;
//!
//! // Create an edge
//! let edge = ProfDAGEdge::new(source_id, target_id, EdgeType::LeadsTo, 0.9);
//! storage.insert_edge(&edge).await?;
//!
//! // Create a light cone for temporal reasoning
//! let mut light_cone = LightCone::new(node_id, Arc::new(storage));
//! light_cone.build().await?;
//!
//! // Query the past
//! let causes = light_cone.what_led_to(&node_id, 5).await?;
//!
//! // Query the future
//! let predictions = light_cone.what_might_follow(&node_id).await?;
//!
//! // Query current context
//! let active = light_cone.whats_relevant();
//! ```

pub mod cognitive_core;
pub mod edge;
pub mod future_cone;
pub mod history_cone;
pub mod light_cone;
pub mod node;
pub mod optimizer;
pub mod profiler;
pub mod search;
pub mod storage;
pub mod trajectory_recorder;
pub mod wormhole;
pub mod wormhole_detector;

pub use edge::{EdgeType, ProfDAGEdge, TemporalDirection};
pub use node::{NodeType, ProfDAGNode};
pub use storage::{
    NeighborQuery, NeighborResult, ProfDAGStats, ProfDAGStorage, ProfDAGStorageConfig,
    SimilarNode,
};

// HNSW-powered vector similarity search
pub use search::{ProfDAGSearch, SearchConfig, SearchMetrics, SearchStats};

// Trajectory recorder types for full reasoning path capture
pub use trajectory_recorder::{
    CompleteResult, RecorderConfig, RecordingSession, ReplayResult, StepSummary,
    TrajectoryRecorder,
};

// Wormhole neural shortcuts for fast pattern access
pub use wormhole::{
    CoAccessRecord, Wormhole, WormholeConfig, WormholeCreationReason,
    WormholeMaintenanceResult, WormholeManager, WormholeStats,
};

// Wormhole detection for automatic shortcut creation
pub use wormhole_detector::{
    DetectionResult, DetectorConfig, DetectorStats, WormholeCandidate, WormholeDetector,
};

// Performance profiler for ProfDAG operations
pub use profiler::{
    OperationType, ProfDAGProfiler, ProfileSnapshot, ProfilerConfig,
};

// Performance optimizer with recommendation engine
pub use optimizer::{
    Bottleneck, OptimizerConfig, ProfDAGOptimizer, Recommendation,
};

// Light Cone temporal reasoning model
pub use cognitive_core::{ActivePattern, AttentionStats, CognitiveCore, CognitiveCoreConfig};
pub use future_cone::{
    FutureCone, FutureConeConfig, PredictedOutcome, PredictionCluster, ProbabilitySummary,
};
pub use history_cone::{CausalChain, HistoryCone, HistoryConeConfig, TemporalNode};
pub use light_cone::{LightCone, LightConeConfig, LightConeStats, NodeId, PatternId};

use thiserror::Error;

/// Errors specific to ProfDAG operations.
#[derive(Error, Debug)]
pub enum ProfDAGError {
    /// Database error
    #[error("Database error: {0}")]
    Database(String),

    /// Node not found
    #[error("Node not found: {id}")]
    NodeNotFound { id: String },

    /// Edge not found
    #[error("Edge not found: {id}")]
    EdgeNotFound { id: String },

    /// Invalid node type
    #[error("Invalid node type: {0}")]
    InvalidNodeType(String),

    /// Invalid edge type
    #[error("Invalid edge type: {0}")]
    InvalidEdgeType(String),

    /// Self-loop not allowed
    #[error("Self-loops are not allowed: {0}")]
    SelfLoop(String),

    /// Invalid weight
    #[error("Invalid weight {0}: must be between 0.0 and 1.0")]
    InvalidWeight(f64),

    /// Cycle detected (DAG violation)
    #[error("Cycle detected: adding edge would create a cycle")]
    CycleDetected,

    /// Embedding dimension mismatch
    #[error("Embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl From<rusqlite::Error> for ProfDAGError {
    fn from(err: rusqlite::Error) -> Self {
        ProfDAGError::Database(err.to_string())
    }
}

impl From<sqlx::Error> for ProfDAGError {
    fn from(err: sqlx::Error) -> Self {
        ProfDAGError::Database(err.to_string())
    }
}

/// Result type for ProfDAG operations.
pub type ProfDAGResult<T> = std::result::Result<T, ProfDAGError>;

/// SQL for creating the profdag_nodes table in SQLite.
pub const SQLITE_PROFDAG_NODES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS profdag_nodes (
    -- Primary key: UUID as text
    id TEXT PRIMARY KEY,

    -- Node type classification
    node_type TEXT NOT NULL CHECK (node_type IN (
        'pattern', 'trajectory', 'prediction', 'decision'
    )),

    -- Content: The actual data/description for this node
    content TEXT NOT NULL,

    -- Vector embedding (stored as JSON array)
    embedding TEXT,

    -- Flexible metadata storage (JSON)
    metadata TEXT DEFAULT '{}',

    -- Reference to source entity
    source_id TEXT,
    source_type TEXT,

    -- Quality metrics
    confidence REAL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    importance REAL DEFAULT 0.5 CHECK (importance >= 0.0 AND importance <= 1.0),

    -- Session tracking
    agent_id TEXT,
    session_id TEXT,

    -- Timestamps
    created_at TEXT NOT NULL,
    updated_at TEXT
);

-- Indexes for profdag_nodes
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_type ON profdag_nodes (node_type);
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_source ON profdag_nodes (source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_agent_session ON profdag_nodes (agent_id, session_id);
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_importance ON profdag_nodes (importance DESC);
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_confidence ON profdag_nodes (confidence DESC);
CREATE INDEX IF NOT EXISTS idx_profdag_nodes_created_at ON profdag_nodes (created_at DESC);
"#;

/// SQL for creating the profdag_edges table in SQLite.
pub const SQLITE_PROFDAG_EDGES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS profdag_edges (
    -- Primary key: UUID as text
    id TEXT PRIMARY KEY,

    -- Source and target node references
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,

    -- Edge type classification
    edge_type TEXT NOT NULL CHECK (edge_type IN (
        'leads_to', 'similar_to', 'derived_from', 'wormhole', 'temporal_link'
    )),

    -- Edge weight/strength
    weight REAL NOT NULL DEFAULT 0.5 CHECK (weight >= 0.0 AND weight <= 1.0),

    -- Flexible metadata storage (JSON)
    metadata TEXT DEFAULT '{}',

    -- Temporal metadata
    temporal_distance_hours INTEGER,
    temporal_direction TEXT CHECK (temporal_direction IS NULL OR temporal_direction IN ('forward', 'backward', 'concurrent')),

    -- Similarity metadata
    similarity_score REAL CHECK (similarity_score IS NULL OR (similarity_score >= 0.0 AND similarity_score <= 1.0)),

    -- Wormhole metadata
    wormhole_strength REAL CHECK (wormhole_strength IS NULL OR (wormhole_strength >= 0.0 AND wormhole_strength <= 1.0)),
    wormhole_reason TEXT,

    -- Timestamps
    created_at TEXT NOT NULL,
    updated_at TEXT,

    -- Constraints
    UNIQUE (source_id, target_id, edge_type),
    CHECK (source_id != target_id),
    FOREIGN KEY (source_id) REFERENCES profdag_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES profdag_nodes(id) ON DELETE CASCADE
);

-- Indexes for profdag_edges
CREATE INDEX IF NOT EXISTS idx_profdag_edges_source ON profdag_edges (source_id);
CREATE INDEX IF NOT EXISTS idx_profdag_edges_target ON profdag_edges (target_id);
CREATE INDEX IF NOT EXISTS idx_profdag_edges_source_type ON profdag_edges (source_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_profdag_edges_target_type ON profdag_edges (target_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_profdag_edges_edge_type ON profdag_edges (edge_type);
CREATE INDEX IF NOT EXISTS idx_profdag_edges_weight ON profdag_edges (weight DESC);
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profdag_error_display() {
        let err = ProfDAGError::NodeNotFound {
            id: "test-123".to_string(),
        };
        assert!(err.to_string().contains("test-123"));

        let err = ProfDAGError::SelfLoop("node-1".to_string());
        assert!(err.to_string().contains("Self-loops"));

        let err = ProfDAGError::InvalidWeight(1.5);
        assert!(err.to_string().contains("1.5"));
    }

    #[test]
    fn test_profdag_error_from_rusqlite() {
        let sqlite_err = rusqlite::Error::InvalidQuery;
        let profdag_err: ProfDAGError = sqlite_err.into();
        assert!(matches!(profdag_err, ProfDAGError::Database(_)));
    }
}
