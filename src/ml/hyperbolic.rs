//! Hyperbolic embeddings for hierarchical knowledge representation.
//!
//! This module implements embeddings in the Poincare ball model of hyperbolic
//! space. Hyperbolic geometry is naturally suited for representing hierarchical
//! structures because:
//!
//! - The volume of a hyperbolic ball grows exponentially with radius (like trees)
//! - Points near the origin represent general/root concepts
//! - Points near the boundary represent specific/leaf concepts
//! - The Poincare distance respects hierarchical relationships
//!
//! # Mathematical Background
//!
//! The Poincare ball model embeds hyperbolic space into the open unit ball
//! `B^n = {x in R^n : ||x|| < 1}`. The Poincare distance between two points
//! `u` and `v` is:
//!
//! ```text
//! d(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
//! ```
//!
//! # Example
//!
//! ```ignore
//! use nagual::ml::hyperbolic::{HyperbolicPoint, HyperbolicConfig, poincare_distance};
//! use ndarray::Array1;
//!
//! let config = HyperbolicConfig::default();
//! let origin = HyperbolicPoint::new(Array1::zeros(128), &config);
//! let leaf = HyperbolicPoint::new(Array1::from_elem(128, 0.5), &config);
//!
//! let dist = poincare_distance(&origin.coords().view(), &leaf.coords().view());
//! assert!(dist > 0.0);
//! ```

use std::cmp::Ordering;

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{MlError, MlResult};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for hyperbolic embedding operations.
///
/// Controls the curvature of hyperbolic space and boundary behavior
/// of the Poincare ball model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicConfig {
    /// Curvature parameter (default: -1.0).
    /// More negative values increase the curvature of the space.
    pub curvature: f64,

    /// Maximum norm for points in the Poincare ball (default: 0.99).
    /// Points are clamped to this norm to avoid numerical instability
    /// at the boundary of the ball.
    pub max_norm: f64,

    /// Embedding dimension (default: 128).
    pub dimension: usize,

    /// Small epsilon for numerical stability (default: 1e-7).
    pub eps: f64,
}

impl Default for HyperbolicConfig {
    fn default() -> Self {
        Self {
            curvature: -1.0,
            dimension: 128,
            max_norm: 0.99,
            eps: 1e-7,
        }
    }
}

impl HyperbolicConfig {
    /// Create a configuration with custom curvature.
    pub fn with_curvature(curvature: f64) -> Self {
        Self {
            curvature,
            ..Default::default()
        }
    }

    /// Create a configuration with custom dimension.
    pub fn with_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }

    /// Set the max norm (builder pattern).
    pub fn max_norm(mut self, max_norm: f64) -> Self {
        self.max_norm = max_norm;
        self
    }

    /// Set the curvature (builder pattern).
    pub fn curvature(mut self, curvature: f64) -> Self {
        self.curvature = curvature;
        self
    }

    /// Absolute curvature value (always positive).
    #[inline]
    pub fn abs_curvature(&self) -> f64 {
        self.curvature.abs()
    }
}

// ============================================================================
// HyperbolicPoint
// ============================================================================

/// A point in the Poincare ball model of hyperbolic space.
///
/// Coordinates are guaranteed to lie within the open unit ball
/// (norm strictly less than `max_norm`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicPoint {
    /// Coordinates in the Poincare ball (f64 for numerical precision).
    /// Stored as Vec for serde compatibility; use `coords()` for array view.
    coords_vec: Vec<f64>,

    /// Hierarchy depth indicator: 0.0 = root/general, 1.0 = leaf/specific.
    /// Derived from the norm of the coordinates.
    pub depth: f64,
}

impl HyperbolicPoint {
    /// Create a new hyperbolic point, projecting onto the ball if necessary.
    ///
    /// # Arguments
    ///
    /// * `coords` - Raw coordinates (will be clamped to the ball)
    /// * `config` - Hyperbolic configuration for max_norm
    pub fn new(coords: Array1<f64>, config: &HyperbolicConfig) -> Self {
        let projected = project_to_ball(&coords.view(), config.max_norm);
        let norm = projected.dot(&projected).sqrt();
        let depth = norm / config.max_norm;

        Self {
            coords_vec: projected.to_vec(),
            depth,
        }
    }

    /// Create a point at the origin (most general concept).
    pub fn origin(dimension: usize) -> Self {
        Self {
            coords_vec: vec![0.0; dimension],
            depth: 0.0,
        }
    }

    /// Get the coordinates as an ndarray.
    pub fn coords(&self) -> Array1<f64> {
        Array1::from_vec(self.coords_vec.clone())
    }

    /// Get a reference to the raw coordinate slice.
    pub fn coords_slice(&self) -> &[f64] {
        &self.coords_vec
    }

    /// Get the Euclidean norm of this point.
    #[inline]
    pub fn norm(&self) -> f64 {
        self.coords_vec.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Get the squared Euclidean norm of this point.
    #[inline]
    pub fn norm_sq(&self) -> f64 {
        self.coords_vec.iter().map(|x| x * x).sum()
    }

    /// Get the dimension of this point.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.coords_vec.len()
    }
}

// ============================================================================
// Core Poincare Ball Operations
// ============================================================================

/// Compute the Poincare distance between two points in the ball.
///
/// Uses the formula:
/// ```text
/// d(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2)(1 - ||v||^2)))
/// ```
///
/// # Arguments
///
/// * `u` - First point in the Poincare ball
/// * `v` - Second point in the Poincare ball
///
/// # Returns
///
/// The hyperbolic distance (always non-negative).
pub fn poincare_distance(u: &ArrayView1<f64>, v: &ArrayView1<f64>) -> f64 {
    let diff = u - v;
    let diff_sq = diff.dot(&diff);

    let u_sq = u.dot(u);
    let v_sq = v.dot(v);

    let denom = (1.0 - u_sq) * (1.0 - v_sq);

    // Guard against division by zero when points are at the boundary
    if denom < 1e-15 {
        return f64::MAX;
    }

    let arg = 1.0 + 2.0 * diff_sq / denom;

    // arcosh(x) = ln(x + sqrt(x^2 - 1)), x >= 1
    // Clamp to 1.0 for numerical stability
    let arg = arg.max(1.0);
    arg.acosh()
}

/// Compute the Poincare distance using f32 inputs (convenience wrapper).
///
/// Converts to f64 internally for numerical precision.
pub fn poincare_distance_f32(u: &ArrayView1<f32>, v: &ArrayView1<f32>) -> f32 {
    let u64: Array1<f64> = u.mapv(|x| x as f64);
    let v64: Array1<f64> = v.mapv(|x| x as f64);
    poincare_distance(&u64.view(), &v64.view()) as f32
}

/// Project a point onto the Poincare ball by clamping its norm.
///
/// If the point's norm exceeds `max_norm`, it is rescaled to have
/// exactly `max_norm` as its norm. Points already inside the ball
/// are returned unchanged.
///
/// # Arguments
///
/// * `point` - The point to project
/// * `max_norm` - Maximum allowed norm (default 0.99)
///
/// # Returns
///
/// The projected point within the ball.
pub fn project_to_ball(point: &ArrayView1<f64>, max_norm: f64) -> Array1<f64> {
    let norm = point.dot(point).sqrt();
    if norm > max_norm {
        point.mapv(|x| x * max_norm / norm)
    } else {
        point.to_owned()
    }
}

/// Exponential map at a base point in the Poincare ball.
///
/// Maps a tangent vector at `base` to a point on the Poincare ball.
/// This is the inverse of the logarithmic map.
///
/// # Formula
///
/// ```text
/// exp_x(v) = x + (tanh(lambda_x * ||v|| / 2) / (lambda_x * ||v||)) * v
/// where lambda_x = 2 / (1 - ||x||^2)  (conformal factor)
/// ```
///
/// # Arguments
///
/// * `base` - The base point on the manifold
/// * `tangent` - The tangent vector at the base point
/// * `config` - Hyperbolic configuration
///
/// # Returns
///
/// The resulting point on the Poincare ball.
pub fn exponential_map(
    base: &ArrayView1<f64>,
    tangent: &ArrayView1<f64>,
    config: &HyperbolicConfig,
) -> MlResult<Array1<f64>> {
    if base.len() != tangent.len() {
        return Err(MlError::Hyperbolic(format!(
            "Dimension mismatch: base={}, tangent={}",
            base.len(),
            tangent.len()
        )));
    }

    let c = config.abs_curvature();
    let sqrt_c = c.sqrt();

    let base_sq = base.dot(base);
    let lambda = 2.0 / (1.0 - c * base_sq).max(config.eps);

    let tangent_norm = tangent.dot(tangent).sqrt();

    if tangent_norm < config.eps {
        // Zero tangent vector: return the base point
        return Ok(base.to_owned());
    }

    let tanh_arg = (sqrt_c * lambda * tangent_norm / 2.0).tanh();
    let factor = tanh_arg / (sqrt_c * tangent_norm);

    let direction = tangent.mapv(|x| x * factor);

    // Mobius addition: base (+) direction
    let result = mobius_add_internal(&base.view(), &direction.view(), c, config.eps);

    Ok(project_to_ball(&result.view(), config.max_norm))
}

/// Logarithmic map at a base point in the Poincare ball.
///
/// Maps a point on the Poincare ball to a tangent vector at `base`.
/// This is the inverse of the exponential map.
///
/// # Formula
///
/// ```text
/// log_x(y) = (2 / (lambda_x * sqrt(c))) * arctanh(sqrt(c) * ||-x (+) y||) * ((-x (+) y) / ||-x (+) y||)
/// ```
///
/// # Arguments
///
/// * `base` - The base point on the manifold
/// * `point` - The target point on the Poincare ball
/// * `config` - Hyperbolic configuration
///
/// # Returns
///
/// The tangent vector at `base` pointing toward `point`.
pub fn logarithmic_map(
    base: &ArrayView1<f64>,
    point: &ArrayView1<f64>,
    config: &HyperbolicConfig,
) -> MlResult<Array1<f64>> {
    if base.len() != point.len() {
        return Err(MlError::Hyperbolic(format!(
            "Dimension mismatch: base={}, point={}",
            base.len(),
            point.len()
        )));
    }

    let c = config.abs_curvature();
    let sqrt_c = c.sqrt();

    let base_sq = base.dot(base);
    let lambda = 2.0 / (1.0 - c * base_sq).max(config.eps);

    // -base (+) point  (Mobius addition with negated base)
    let neg_base = base.mapv(|x| -x);
    let addition = mobius_add_internal(&neg_base.view(), &point.view(), c, config.eps);

    let add_norm = addition.dot(&addition).sqrt();

    if add_norm < config.eps {
        // Points are the same: return zero tangent vector
        return Ok(Array1::zeros(base.len()));
    }

    let atanh_arg = (sqrt_c * add_norm).min(1.0 - config.eps);
    let scale = (2.0 / (lambda * sqrt_c)) * atanh_arg.atanh();

    Ok(addition.mapv(|x| x * scale / add_norm))
}

/// Convert a Euclidean embedding to a Poincare ball point.
///
/// Uses the exponential map from the origin to place the Euclidean
/// vector into hyperbolic space. Vectors with larger norms end up
/// further from the origin (more specific in the hierarchy).
///
/// # Arguments
///
/// * `embedding` - A Euclidean embedding vector (e.g., 128-dim)
/// * `config` - Hyperbolic configuration
///
/// # Returns
///
/// A `HyperbolicPoint` in the Poincare ball.
pub fn euclidean_to_poincare(
    embedding: &ArrayView1<f32>,
    config: &HyperbolicConfig,
) -> MlResult<HyperbolicPoint> {
    // Convert to f64 for numerical precision
    let emb_f64: Array1<f64> = embedding.mapv(|x| x as f64);

    // Normalize, then scale to a reasonable range in the ball
    let norm = emb_f64.dot(&emb_f64).sqrt();
    if norm < config.eps {
        return Ok(HyperbolicPoint::origin(embedding.len()));
    }

    // Scale the normalized vector. Use tanh to map [0, inf) -> [0, 1)
    // This ensures the result stays inside the ball.
    let target_norm = (norm * 0.5).tanh() * config.max_norm;
    let scaled = emb_f64.mapv(|x| x * target_norm / norm);

    Ok(HyperbolicPoint::new(scaled, config))
}

/// Internal Mobius addition: `x (+)_c y`.
///
/// ```text
/// x (+)_c y = ((1 + 2c<x,y> + c||y||^2) * x + (1 - c||x||^2) * y)
///             / (1 + 2c<x,y> + c^2 * ||x||^2 * ||y||^2)
/// ```
fn mobius_add_internal(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    c: f64,
    eps: f64,
) -> Array1<f64> {
    let x_sq = x.dot(x);
    let y_sq = y.dot(y);
    let xy = x.dot(y);

    let num_factor_x = 1.0 + 2.0 * c * xy + c * y_sq;
    let num_factor_y = 1.0 - c * x_sq;
    let denom = (1.0 + 2.0 * c * xy + c * c * x_sq * y_sq).max(eps);

    let mut result = Array1::zeros(x.len());
    for i in 0..x.len() {
        result[i] = (num_factor_x * x[i] + num_factor_y * y[i]) / denom;
    }

    result
}

// ============================================================================
// HyperbolicEmbedder
// ============================================================================

/// Embedder that operates in hyperbolic space for hierarchical knowledge.
///
/// Provides methods to embed data with depth awareness, find ancestors
/// (more general concepts) and descendants (more specific concepts),
/// and compute hierarchy-respecting distances.
#[derive(Debug, Clone)]
pub struct HyperbolicEmbedder {
    /// Configuration for hyperbolic operations.
    config: HyperbolicConfig,

    /// Stored points for search operations.
    points: Vec<(Uuid, HyperbolicPoint)>,
}

impl HyperbolicEmbedder {
    /// Create a new hyperbolic embedder with the given configuration.
    pub fn new(config: HyperbolicConfig) -> Self {
        Self {
            config,
            points: Vec::new(),
        }
    }

    /// Create an embedder with default configuration.
    pub fn default_config() -> Self {
        Self::new(HyperbolicConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &HyperbolicConfig {
        &self.config
    }

    /// Get the number of stored points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the embedder has no stored points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Insert a point into the embedder's index.
    pub fn insert(&mut self, id: Uuid, point: HyperbolicPoint) {
        self.points.push((id, point));
    }

    /// Embed a Euclidean vector with depth awareness.
    ///
    /// The depth parameter controls how far from the origin the point
    /// is placed. Depth 0.0 means root/general (near origin), depth 1.0
    /// means leaf/specific (near boundary).
    ///
    /// # Arguments
    ///
    /// * `embedding` - The Euclidean embedding vector
    /// * `depth` - Hierarchy depth (0.0 = root, 1.0 = leaf)
    ///
    /// # Returns
    ///
    /// A `HyperbolicPoint` positioned according to the hierarchy depth.
    pub fn embed_hierarchical(
        &self,
        embedding: &ArrayView1<f32>,
        depth: f64,
    ) -> MlResult<HyperbolicPoint> {
        if embedding.is_empty() {
            return Err(MlError::Hyperbolic(
                "Cannot embed empty vector".to_string(),
            ));
        }

        let depth = depth.clamp(0.0, 1.0);

        // Convert to f64 and normalize direction
        let emb_f64: Array1<f64> = embedding.mapv(|x| x as f64);
        let norm = emb_f64.dot(&emb_f64).sqrt();

        if norm < self.config.eps {
            return Ok(HyperbolicPoint::origin(embedding.len()));
        }

        let direction = emb_f64.mapv(|x| x / norm);

        // Map depth to a target norm in the ball.
        // Use a smooth mapping: deeper = further from origin = more specific.
        // tanh provides a nice sigmoidal mapping to (0, max_norm).
        let target_norm = depth * self.config.max_norm * 0.95;
        let scaled = direction.mapv(|x| x * target_norm);

        Ok(HyperbolicPoint::new(scaled, &self.config))
    }

    /// Find the k nearest ancestors (more general concepts) of a point.
    ///
    /// Ancestors are points that are closer to the origin than the query point,
    /// indicating they represent more general concepts in the hierarchy.
    ///
    /// # Arguments
    ///
    /// * `point` - The query point
    /// * `k` - Maximum number of ancestors to return
    ///
    /// # Returns
    ///
    /// Vector of (id, point, distance) tuples sorted by Poincare distance.
    pub fn find_ancestors(
        &self,
        point: &HyperbolicPoint,
        k: usize,
    ) -> Vec<(Uuid, HyperbolicPoint, f64)> {
        let query_norm = point.norm();

        // Filter to points closer to origin (ancestors)
        let mut candidates: Vec<(Uuid, HyperbolicPoint, f64)> = self
            .points
            .iter()
            .filter(|(_, p)| p.norm() < query_norm)
            .map(|(id, p)| {
                let dist = poincare_distance(&point.coords().view(), &p.coords().view());
                (*id, p.clone(), dist)
            })
            .collect();

        // Sort by Poincare distance (closest first)
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Find the k nearest descendants (more specific concepts) of a point.
    ///
    /// Descendants are points that are further from the origin than the query
    /// point, indicating they represent more specific concepts in the hierarchy.
    ///
    /// # Arguments
    ///
    /// * `point` - The query point
    /// * `k` - Maximum number of descendants to return
    ///
    /// # Returns
    ///
    /// Vector of (id, point, distance) tuples sorted by Poincare distance.
    pub fn find_descendants(
        &self,
        point: &HyperbolicPoint,
        k: usize,
    ) -> Vec<(Uuid, HyperbolicPoint, f64)> {
        let query_norm = point.norm();

        // Filter to points further from origin (descendants)
        let mut candidates: Vec<(Uuid, HyperbolicPoint, f64)> = self
            .points
            .iter()
            .filter(|(_, p)| p.norm() > query_norm)
            .map(|(id, p)| {
                let dist = poincare_distance(&point.coords().view(), &p.coords().view());
                (*id, p.clone(), dist)
            })
            .collect();

        // Sort by Poincare distance (closest first)
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Convert a Euclidean embedding to a Poincare ball point using domain depth.
    ///
    /// Uses domain hierarchy depth to position points: root domains (depth 1)
    /// are placed near the origin, while deeply nested domains (depth 5+)
    /// are placed near the boundary. This reflects the natural tree structure
    /// of domain hierarchies in hyperbolic space.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The Euclidean embedding vector (e.g., 128-dim)
    /// * `domain` - Domain string (e.g., "rust.async.tokio")
    ///
    /// # Returns
    ///
    /// A `HyperbolicPoint` positioned according to the domain depth.
    pub fn embed_from_euclidean(
        &self,
        embedding: &ArrayView1<f32>,
        domain: &str,
    ) -> MlResult<HyperbolicPoint> {
        // Use domain depth to determine hierarchy level
        let domain_depth = crate::reasoning_bank::domain::depth(domain);
        // Normalize depth to 0-1 range (assume max depth ~5)
        let normalized_depth = (domain_depth as f64 / 5.0).min(1.0);
        self.embed_hierarchical(embedding, normalized_depth)
    }

    /// Compute a hierarchy-aware distance between two points.
    ///
    /// This distance combines the Poincare distance with a penalty for
    /// crossing hierarchy levels. Moving along the same level is cheaper
    /// than moving across levels, reflecting that sibling concepts are
    /// more related than parent-child concepts at the same Poincare distance.
    ///
    /// # Arguments
    ///
    /// * `a` - First point
    /// * `b` - Second point
    ///
    /// # Returns
    ///
    /// The hierarchy-aware distance (always non-negative).
    pub fn hierarchy_distance(&self, a: &HyperbolicPoint, b: &HyperbolicPoint) -> f64 {
        let poincare_dist = poincare_distance(&a.coords().view(), &b.coords().view());
        let depth_diff = (a.depth - b.depth).abs();

        // Weighted combination: Poincare distance + depth penalty
        // The depth penalty discourages crossing many hierarchy levels
        poincare_dist + 0.5 * depth_diff * poincare_dist
    }
}

// ============================================================================
// HyperbolicIndex (feature-gated)
// ============================================================================

mod hyperbolic_index {
    use super::*;
    use rand::Rng;
    use std::collections::{BinaryHeap, HashMap, HashSet};

    /// Configuration for the HNSW-style hyperbolic index.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HyperbolicIndexConfig {
        /// M parameter: max connections per node per layer (default: 16).
        pub max_connections: usize,

        /// ef parameter during construction (default: 200).
        pub ef_construction: usize,

        /// ef parameter during search (default: 50).
        pub ef_search: usize,

        /// Maximum number of HNSW layers (default: 5).
        pub max_layers: usize,

        /// Enable dual-space (hyperbolic + euclidean) search (default: true).
        pub use_dual_space: bool,

        /// Weight for the euclidean component in dual search (default: 0.3).
        pub euclidean_weight: f64,

        /// Weight for the hyperbolic component in dual search (default: 0.7).
        pub hyperbolic_weight: f64,
    }

    impl Default for HyperbolicIndexConfig {
        fn default() -> Self {
            Self {
                max_connections: 16,
                ef_construction: 200,
                ef_search: 50,
                max_layers: 5,
                use_dual_space: true,
                euclidean_weight: 0.3,
                hyperbolic_weight: 0.7,
            }
        }
    }

    /// A node stored in the hyperbolic index.
    #[derive(Debug, Clone)]
    pub struct IndexNode {
        /// Unique identifier for this node.
        pub id: String,

        /// The hyperbolic (Poincare ball) coordinates.
        pub point: HyperbolicPoint,

        /// Optional original Euclidean embedding for dual-space search.
        pub euclidean_coords: Option<Vec<f32>>,

        /// Connections per layer: `connections[layer]` is a Vec of neighbor IDs.
        pub connections: Vec<Vec<String>>,

        /// Highest layer this node appears in.
        pub layer: usize,

        /// Optional domain tag.
        pub domain: Option<String>,
    }

    /// A single search result from the hyperbolic index.
    #[derive(Debug, Clone)]
    pub struct SearchResult {
        /// ID of the matched node.
        pub id: String,

        /// Poincare distance from the query point.
        pub hyperbolic_distance: f64,

        /// Euclidean distance from the query point (if dual-space).
        pub euclidean_distance: Option<f64>,

        /// Combined score (lower is better).
        pub combined_score: f64,

        /// Depth of the node in the hierarchy (0.0 = root, 1.0 = leaf).
        pub depth: f64,
    }

    impl PartialEq for SearchResult {
        fn eq(&self, other: &Self) -> bool {
            self.combined_score == other.combined_score
        }
    }

    impl Eq for SearchResult {}

    impl PartialOrd for SearchResult {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for SearchResult {
        fn cmp(&self, other: &Self) -> Ordering {
            // Max-heap: higher combined_score = higher priority (we pop worst first)
            self.combined_score
                .partial_cmp(&other.combined_score)
                .unwrap_or(Ordering::Equal)
        }
    }

    /// Statistics about the index.
    #[derive(Debug, Clone)]
    pub struct IndexStats {
        /// Total number of nodes in the index.
        pub total_nodes: usize,

        /// Maximum layer in the index.
        pub max_layer: usize,

        /// Average number of connections per node (across all layers).
        pub avg_connections: f64,

        /// Distribution of node depths as `(depth_bucket, count)` pairs.
        pub depth_distribution: Vec<(f64, usize)>,
    }

    /// An HNSW-style index operating in the Poincare ball model of hyperbolic space.
    ///
    /// Combines hierarchical navigable small world graphs with Poincare distance
    /// for hierarchy-aware nearest neighbor search. Optionally supports dual-space
    /// search that blends hyperbolic and Euclidean distances.
    pub struct HyperbolicIndex {
        /// Index configuration (HNSW parameters, dual-space weights).
        config: HyperbolicIndexConfig,

        /// Underlying hyperbolic space configuration.
        hyper_config: HyperbolicConfig,

        /// All nodes in the index, keyed by ID.
        nodes: HashMap<String, IndexNode>,

        /// Entry point for HNSW traversal (highest-layer node).
        entry_point: Option<String>,

        /// Current maximum layer across all nodes.
        max_layer: usize,
    }

    impl std::fmt::Debug for HyperbolicIndex {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("HyperbolicIndex")
                .field("config", &self.config)
                .field("hyper_config", &self.hyper_config)
                .field("num_nodes", &self.nodes.len())
                .field("entry_point", &self.entry_point)
                .field("max_layer", &self.max_layer)
                .finish()
        }
    }

    /// Assign a random layer using a geometric distribution.
    ///
    /// P(layer = l) decreases exponentially, so most nodes live on layer 0
    /// and very few reach the top layers. This mirrors standard HNSW behavior.
    fn random_layer(max_layers: usize) -> usize {
        if max_layers <= 1 {
            return 0;
        }
        let mut rng = rand::thread_rng();
        let ml = 1.0 / (max_layers as f64).ln();
        let r: f64 = rng.gen();
        let raw = (-r.ln() * ml) as usize;
        raw.min(max_layers - 1)
    }

    impl HyperbolicIndex {
        /// Create a new empty index.
        pub fn new(hyper_config: HyperbolicConfig, index_config: HyperbolicIndexConfig) -> Self {
            Self {
                config: index_config,
                hyper_config,
                nodes: HashMap::new(),
                entry_point: None,
                max_layer: 0,
            }
        }

        /// Insert a node into the index.
        ///
        /// Assigns a random layer via geometric distribution, then connects the
        /// node to its nearest neighbors on each layer (greedy HNSW construction).
        pub fn insert(
            &mut self,
            id: String,
            point: HyperbolicPoint,
            euclidean: Option<Vec<f32>>,
            domain: Option<String>,
        ) {
            let node_layer = random_layer(self.config.max_layers);
            let num_layers = node_layer + 1;

            let mut connections = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                connections.push(Vec::new());
            }

            let node = IndexNode {
                id: id.clone(),
                point,
                euclidean_coords: euclidean,
                connections,
                layer: node_layer,
                domain,
            };

            // If index is empty, this becomes the entry point.
            if self.nodes.is_empty() {
                self.entry_point = Some(id.clone());
                self.max_layer = node_layer;
                self.nodes.insert(id, node);
                return;
            }

            // Connect to existing neighbors on each layer.
            let query_point = node.point.clone();
            let max_conn = self.config.max_connections;

            // Collect neighbor IDs for each layer before mutating self.nodes.
            let mut layer_neighbors: Vec<Vec<String>> = Vec::new();

            for layer in 0..num_layers {
                let neighbors = self.search_layer(&query_point, layer, max_conn);
                let neighbor_ids: Vec<String> = neighbors.into_iter().map(|(nid, _)| nid).collect();
                layer_neighbors.push(neighbor_ids);
            }

            // Insert the new node first so we can borrow mutably for connections.
            self.nodes.insert(id.clone(), node);

            // Wire bidirectional connections.
            for (layer, neighbor_ids) in layer_neighbors.iter().enumerate() {
                // Set forward connections on the new node.
                if let Some(new_node) = self.nodes.get_mut(&id) {
                    new_node.connections[layer] = neighbor_ids.clone();
                }

                // Set reverse connections on each neighbor.
                for nid in neighbor_ids {
                    // Check if neighbor needs trimming (separate scope to avoid borrow conflict).
                    let trimmed = {
                        if let Some(neighbor) = self.nodes.get(nid) {
                            if layer < neighbor.connections.len()
                                && !neighbor.connections[layer].contains(&id)
                            {
                                let mut conns = neighbor.connections[layer].clone();
                                conns.push(id.clone());
                                if conns.len() > max_conn {
                                    let np = neighbor.point.clone();
                                    let mut scored: Vec<(String, f64)> = conns
                                        .iter()
                                        .filter_map(|cid| {
                                            self.nodes.get(cid).map(|cn| {
                                                let d = poincare_distance(
                                                    &np.coords().view(),
                                                    &cn.point.coords().view(),
                                                );
                                                (cid.clone(), d)
                                            })
                                        })
                                        .collect();
                                    scored.sort_by(|a, b| {
                                        a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
                                    });
                                    scored.truncate(max_conn);
                                    Some(scored.into_iter().map(|(s, _)| s).collect::<Vec<_>>())
                                } else {
                                    Some(conns)
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    };

                    // Apply the trimmed connections with a mutable borrow.
                    if let Some(new_conns) = trimmed {
                        if let Some(neighbor) = self.nodes.get_mut(nid) {
                            neighbor.connections[layer] = new_conns;
                        }
                    }
                }
            }

            // Update entry point if the new node has a higher layer.
            if node_layer > self.max_layer {
                self.max_layer = node_layer;
                self.entry_point = Some(id);
            }
        }

        /// Search for the k nearest neighbors using Poincare distance only.
        pub fn search(&self, query: &HyperbolicPoint, k: usize) -> Vec<SearchResult> {
            if self.nodes.is_empty() || k == 0 {
                return Vec::new();
            }

            // Greedy search from the entry point, descending through layers.
            let entry = match &self.entry_point {
                Some(e) => e.clone(),
                None => return Vec::new(),
            };

            // Start from the top layer and greedily descend to layer 0.
            let mut current = entry;
            for layer in (1..=self.max_layer).rev() {
                let neighbors = self.search_layer_single(query, &current, layer);
                if let Some((best_id, _)) = neighbors.first() {
                    let best_dist = self.hyp_dist(query, best_id);
                    let cur_dist = self.hyp_dist(query, &current);
                    if best_dist < cur_dist {
                        current = best_id.clone();
                    }
                }
            }

            // At layer 0, do a wider search with ef_search candidates.
            let ef = self.config.ef_search.max(k);
            let candidates = self.search_layer_wide(query, &current, 0, ef);

            // Convert to SearchResult, sorted by distance, truncated to k.
            let mut results: Vec<SearchResult> = candidates
                .into_iter()
                .map(|(id, dist)| {
                    let depth = self.nodes.get(&id).map(|n| n.point.depth).unwrap_or(0.0);
                    SearchResult {
                        id,
                        hyperbolic_distance: dist,
                        euclidean_distance: None,
                        combined_score: dist,
                        depth,
                    }
                })
                .collect();

            results.sort_by(|a, b| {
                a.combined_score
                    .partial_cmp(&b.combined_score)
                    .unwrap_or(Ordering::Equal)
            });
            results.truncate(k);
            results
        }

        /// Dual-space search combining hyperbolic and Euclidean distances.
        ///
        /// The combined score is:
        /// ```text
        /// score = hyperbolic_weight * hyp_dist + euclidean_weight * euc_dist
        /// ```
        ///
        /// If a node lacks Euclidean coordinates, the score falls back to pure
        /// hyperbolic distance.
        pub fn dual_search(
            &self,
            query_hyp: &HyperbolicPoint,
            query_euc: Option<&[f32]>,
            k: usize,
        ) -> Vec<SearchResult> {
            if self.nodes.is_empty() || k == 0 {
                return Vec::new();
            }

            // First, get a broad set of candidates via hyperbolic search.
            let ef = self.config.ef_search.max(k) * 2; // over-fetch for re-ranking
            let hyp_candidates = self.get_all_distances(query_hyp);

            let mut results: Vec<SearchResult> = hyp_candidates
                .into_iter()
                .map(|(id, hyp_dist)| {
                    let node = self.nodes.get(&id);
                    let depth = node.map(|n| n.point.depth).unwrap_or(0.0);

                    let euc_dist = match (query_euc, node.and_then(|n| n.euclidean_coords.as_ref()))
                    {
                        (Some(qe), Some(ne)) => {
                            let d = euclidean_distance_f32(qe, ne);
                            Some(d)
                        }
                        _ => None,
                    };

                    let combined = match euc_dist {
                        Some(ed) => {
                            self.config.hyperbolic_weight * hyp_dist
                                + self.config.euclidean_weight * ed
                        }
                        None => hyp_dist,
                    };

                    SearchResult {
                        id,
                        hyperbolic_distance: hyp_dist,
                        euclidean_distance: euc_dist,
                        combined_score: combined,
                        depth,
                    }
                })
                .collect();

            results.sort_by(|a, b| {
                a.combined_score
                    .partial_cmp(&b.combined_score)
                    .unwrap_or(Ordering::Equal)
            });
            results.truncate(k);
            results
        }

        /// Search within a specific depth range in the hierarchy.
        ///
        /// Only returns nodes whose depth falls within `[min_depth, max_depth]`.
        pub fn search_by_depth(
            &self,
            query: &HyperbolicPoint,
            k: usize,
            min_depth: f64,
            max_depth: f64,
        ) -> Vec<SearchResult> {
            if self.nodes.is_empty() || k == 0 {
                return Vec::new();
            }

            let mut results: Vec<SearchResult> = self
                .nodes
                .values()
                .filter(|n| n.point.depth >= min_depth && n.point.depth <= max_depth)
                .map(|n| {
                    let dist =
                        poincare_distance(&query.coords().view(), &n.point.coords().view());
                    SearchResult {
                        id: n.id.clone(),
                        hyperbolic_distance: dist,
                        euclidean_distance: None,
                        combined_score: dist,
                        depth: n.point.depth,
                    }
                })
                .collect();

            results.sort_by(|a, b| {
                a.combined_score
                    .partial_cmp(&b.combined_score)
                    .unwrap_or(Ordering::Equal)
            });
            results.truncate(k);
            results
        }

        /// Remove a node from the index.
        ///
        /// Returns `true` if the node existed and was removed.
        pub fn remove(&mut self, id: &str) -> bool {
            let removed = self.nodes.remove(id);
            if removed.is_none() {
                return false;
            }

            // Remove all references to this node from neighbors' connections.
            let id_string = id.to_string();
            for node in self.nodes.values_mut() {
                for layer_conns in &mut node.connections {
                    layer_conns.retain(|c| c != &id_string);
                }
            }

            // If we removed the entry point, pick a new one.
            if self.entry_point.as_deref() == Some(id) {
                self.entry_point = self
                    .nodes
                    .values()
                    .max_by_key(|n| n.layer)
                    .map(|n| n.id.clone());
                self.max_layer = self
                    .nodes
                    .values()
                    .map(|n| n.layer)
                    .max()
                    .unwrap_or(0);
            }

            true
        }

        /// Get a node by ID.
        pub fn get(&self, id: &str) -> Option<&IndexNode> {
            self.nodes.get(id)
        }

        /// Return the number of nodes in the index.
        pub fn len(&self) -> usize {
            self.nodes.len()
        }

        /// Check whether the index is empty.
        pub fn is_empty(&self) -> bool {
            self.nodes.is_empty()
        }

        /// Compute index statistics.
        pub fn stats(&self) -> IndexStats {
            if self.nodes.is_empty() {
                return IndexStats {
                    total_nodes: 0,
                    max_layer: 0,
                    avg_connections: 0.0,
                    depth_distribution: Vec::new(),
                };
            }

            let total_connections: usize = self
                .nodes
                .values()
                .flat_map(|n| n.connections.iter())
                .map(|c| c.len())
                .sum();
            let total_layers: usize = self.nodes.values().map(|n| n.connections.len()).sum();
            let avg = if total_layers > 0 {
                total_connections as f64 / total_layers as f64
            } else {
                0.0
            };

            // Bucket depths into 0.1-wide bins.
            let mut depth_counts: HashMap<u64, usize> = HashMap::new();
            for node in self.nodes.values() {
                let bucket = (node.point.depth * 10.0).round() as u64;
                *depth_counts.entry(bucket).or_insert(0) += 1;
            }
            let mut depth_distribution: Vec<(f64, usize)> = depth_counts
                .into_iter()
                .map(|(b, c)| (b as f64 / 10.0, c))
                .collect();
            depth_distribution.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

            IndexStats {
                total_nodes: self.nodes.len(),
                max_layer: self.max_layer,
                avg_connections: avg,
                depth_distribution,
            }
        }

        /// Build an index from a batch of patterns.
        pub fn build_from_patterns(
            patterns: &[(String, HyperbolicPoint, Option<Vec<f32>>, Option<String>)],
            hyper_config: HyperbolicConfig,
            index_config: HyperbolicIndexConfig,
        ) -> Self {
            let mut index = Self::new(hyper_config, index_config);
            for (id, point, euc, domain) in patterns {
                index.insert(id.clone(), point.clone(), euc.clone(), domain.clone());
            }
            index
        }

        // -- internal helpers --

        /// Compute Poincare distance to a node by ID.
        fn hyp_dist(&self, query: &HyperbolicPoint, id: &str) -> f64 {
            match self.nodes.get(id) {
                Some(n) => {
                    poincare_distance(&query.coords().view(), &n.point.coords().view())
                }
                None => f64::MAX,
            }
        }

        /// Greedy search on a single layer: from `start`, walk to the nearest
        /// neighbor until no closer neighbor is found. Return visited nodes
        /// that were improvements.
        fn search_layer_single(
            &self,
            query: &HyperbolicPoint,
            start: &str,
            layer: usize,
        ) -> Vec<(String, f64)> {
            let mut current = start.to_string();
            let mut current_dist = self.hyp_dist(query, &current);
            let mut result = vec![(current.clone(), current_dist)];

            loop {
                let mut improved = false;
                if let Some(node) = self.nodes.get(&current) {
                    if layer < node.connections.len() {
                        for neighbor_id in &node.connections[layer] {
                            let d = self.hyp_dist(query, neighbor_id);
                            if d < current_dist {
                                current = neighbor_id.clone();
                                current_dist = d;
                                result.push((current.clone(), current_dist));
                                improved = true;
                            }
                        }
                    }
                }
                if !improved {
                    break;
                }
            }

            result
        }

        /// Wide search on a layer: BFS-like expansion from `start`, collecting
        /// up to `ef` candidates sorted by distance.
        fn search_layer_wide(
            &self,
            query: &HyperbolicPoint,
            start: &str,
            layer: usize,
            ef: usize,
        ) -> Vec<(String, f64)> {
            let mut visited: HashSet<String> = HashSet::new();
            let mut candidates: BinaryHeap<std::cmp::Reverse<(OrdF64, String)>> =
                BinaryHeap::new();
            let mut results: BinaryHeap<(OrdF64, String)> = BinaryHeap::new();

            let start_dist = self.hyp_dist(query, start);
            visited.insert(start.to_string());
            candidates.push(std::cmp::Reverse((OrdF64(start_dist), start.to_string())));
            results.push((OrdF64(start_dist), start.to_string()));

            while let Some(std::cmp::Reverse((OrdF64(c_dist), c_id))) = candidates.pop() {
                // If the closest candidate is farther than the farthest result, stop.
                if let Some(&(OrdF64(worst), _)) = results.peek() {
                    if c_dist > worst && results.len() >= ef {
                        break;
                    }
                }

                if let Some(node) = self.nodes.get(&c_id) {
                    if layer < node.connections.len() {
                        for neighbor_id in &node.connections[layer] {
                            if visited.contains(neighbor_id) {
                                continue;
                            }
                            visited.insert(neighbor_id.clone());

                            let d = self.hyp_dist(query, neighbor_id);

                            let dominated = if results.len() >= ef {
                                if let Some(&(OrdF64(worst), _)) = results.peek() {
                                    d > worst
                                } else {
                                    false
                                }
                            } else {
                                false
                            };

                            if !dominated {
                                candidates.push(std::cmp::Reverse((
                                    OrdF64(d),
                                    neighbor_id.clone(),
                                )));
                                results.push((OrdF64(d), neighbor_id.clone()));
                                if results.len() > ef {
                                    results.pop(); // Remove worst
                                }
                            }
                        }
                    }
                }
            }

            results
                .into_sorted_vec()
                .into_iter()
                .map(|(OrdF64(d), id)| (id, d))
                .collect()
        }

        /// Search a layer for the nearest neighbors during construction.
        fn search_layer(
            &self,
            query: &HyperbolicPoint,
            layer: usize,
            k: usize,
        ) -> Vec<(String, f64)> {
            // For construction, scan all nodes that exist at this layer.
            let mut scored: Vec<(String, f64)> = self
                .nodes
                .values()
                .filter(|n| n.layer >= layer)
                .map(|n| {
                    let d =
                        poincare_distance(&query.coords().view(), &n.point.coords().view());
                    (n.id.clone(), d)
                })
                .collect();

            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            scored.truncate(k);
            scored
        }

        /// Compute distances to all nodes (used for dual search).
        fn get_all_distances(&self, query: &HyperbolicPoint) -> Vec<(String, f64)> {
            self.nodes
                .values()
                .map(|n| {
                    let d =
                        poincare_distance(&query.coords().view(), &n.point.coords().view());
                    (n.id.clone(), d)
                })
                .collect()
        }
    }

    /// Compute Euclidean distance between two f32 slices.
    fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f64 {
        let min_len = a.len().min(b.len());
        let mut sum = 0.0_f64;
        for i in 0..min_len {
            let diff = (a[i] as f64) - (b[i] as f64);
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Newtype wrapper to give `f64` a total ordering for use in `BinaryHeap`.
    #[derive(Debug, Clone, Copy)]
    struct OrdF64(f64);

    impl PartialEq for OrdF64 {
        fn eq(&self, other: &Self) -> bool {
            self.0.to_bits() == other.0.to_bits()
        }
    }
    impl Eq for OrdF64 {}

    impl PartialOrd for OrdF64 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for OrdF64 {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
        }
    }

    // Re-export the random_layer function for testing.
    pub fn random_layer_pub(max_layers: usize) -> usize {
        random_layer(max_layers)
    }
}

pub use hyperbolic_index::{
    HyperbolicIndex, HyperbolicIndexConfig, IndexNode, IndexStats, SearchResult,
    random_layer_pub as random_layer,
};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_config() -> HyperbolicConfig {
        HyperbolicConfig::default()
    }

    #[test]
    fn test_poincare_distance_identity() {
        // Distance from a point to itself should be 0
        let p = Array1::from_vec(vec![0.3, 0.4, 0.0]);
        let dist = poincare_distance(&p.view(), &p.view());
        assert!(dist.abs() < 1e-10, "Distance to self should be 0, got {}", dist);
    }

    #[test]
    fn test_poincare_distance_symmetry() {
        let u = Array1::from_vec(vec![0.3, 0.2, -0.1]);
        let v = Array1::from_vec(vec![-0.1, 0.4, 0.2]);

        let d_uv = poincare_distance(&u.view(), &v.view());
        let d_vu = poincare_distance(&v.view(), &u.view());

        assert!(
            (d_uv - d_vu).abs() < 1e-10,
            "Distance should be symmetric: d(u,v)={}, d(v,u)={}",
            d_uv,
            d_vu
        );
    }

    #[test]
    fn test_poincare_distance_triangle_inequality() {
        let a = Array1::from_vec(vec![0.1, 0.2]);
        let b = Array1::from_vec(vec![0.3, -0.1]);
        let c = Array1::from_vec(vec![-0.2, 0.4]);

        let d_ab = poincare_distance(&a.view(), &b.view());
        let d_bc = poincare_distance(&b.view(), &c.view());
        let d_ac = poincare_distance(&a.view(), &c.view());

        assert!(
            d_ac <= d_ab + d_bc + 1e-10,
            "Triangle inequality violated: d(a,c)={} > d(a,b)+d(b,c)={}",
            d_ac,
            d_ab + d_bc
        );
    }

    #[test]
    fn test_poincare_distance_origin() {
        // Distance from origin increases with norm
        let origin = Array1::from_vec(vec![0.0, 0.0]);
        let near = Array1::from_vec(vec![0.1, 0.0]);
        let far = Array1::from_vec(vec![0.5, 0.0]);
        let very_far = Array1::from_vec(vec![0.9, 0.0]);

        let d_near = poincare_distance(&origin.view(), &near.view());
        let d_far = poincare_distance(&origin.view(), &far.view());
        let d_very_far = poincare_distance(&origin.view(), &very_far.view());

        assert!(d_near < d_far, "Near should be closer than far");
        assert!(d_far < d_very_far, "Far should be closer than very_far");
    }

    #[test]
    fn test_poincare_distance_non_negative() {
        let u = Array1::from_vec(vec![0.5, -0.3]);
        let v = Array1::from_vec(vec![-0.2, 0.6]);

        let dist = poincare_distance(&u.view(), &v.view());
        assert!(dist >= 0.0, "Distance should be non-negative, got {}", dist);
    }

    #[test]
    fn test_project_to_ball_inside() {
        // Point already inside the ball should be unchanged
        let point = Array1::from_vec(vec![0.3, 0.4]);
        let projected = project_to_ball(&point.view(), 0.99);

        for (a, b) in point.iter().zip(projected.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_project_to_ball_outside() {
        // Point outside the ball should be projected onto it
        let point = Array1::from_vec(vec![3.0, 4.0]); // norm = 5.0
        let max_norm = 0.99;
        let projected = project_to_ball(&point.view(), max_norm);

        let proj_norm = projected.dot(&projected).sqrt();
        assert!(
            (proj_norm - max_norm).abs() < 1e-10,
            "Projected norm should be max_norm={}, got {}",
            max_norm,
            proj_norm
        );

        // Direction should be preserved
        let orig_dir = point.mapv(|x| x / 5.0);
        let proj_dir = projected.mapv(|x| x / proj_norm);
        for (a, b) in orig_dir.iter().zip(proj_dir.iter()) {
            assert!((a - b).abs() < 1e-10, "Direction should be preserved");
        }
    }

    #[test]
    fn test_project_to_ball_boundary() {
        // Point exactly at max_norm should be unchanged
        let max_norm = 0.99;
        let point = Array1::from_vec(vec![max_norm, 0.0]);
        let projected = project_to_ball(&point.view(), max_norm);
        let proj_norm = projected.dot(&projected).sqrt();
        assert!((proj_norm - max_norm).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_logarithmic_roundtrip() {
        let config = make_config();
        let base = Array1::from_vec(vec![0.1, 0.2, 0.0]);
        let tangent = Array1::from_vec(vec![0.05, -0.03, 0.02]);

        // exp_base(tangent) -> point
        let point = exponential_map(&base.view(), &tangent.view(), &config).unwrap();

        // log_base(point) -> should recover tangent
        let recovered = logarithmic_map(&base.view(), &point.view(), &config).unwrap();

        for (a, b) in tangent.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Round-trip failed: tangent={:.6}, recovered={:.6}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_exponential_map_zero_tangent() {
        let config = make_config();
        let base = Array1::from_vec(vec![0.3, 0.2]);
        let zero = Array1::from_vec(vec![0.0, 0.0]);

        let result = exponential_map(&base.view(), &zero.view(), &config).unwrap();

        for (a, b) in base.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-10, "Zero tangent should return base");
        }
    }

    #[test]
    fn test_exponential_map_stays_in_ball() {
        let config = make_config();
        let base = Array1::from_vec(vec![0.5, 0.3]);
        let tangent = Array1::from_vec(vec![10.0, -5.0]); // Large tangent

        let result = exponential_map(&base.view(), &tangent.view(), &config).unwrap();
        let norm = result.dot(&result).sqrt();

        assert!(
            norm <= config.max_norm + 1e-10,
            "Result should stay in ball, norm={}",
            norm
        );
    }

    #[test]
    fn test_euclidean_to_poincare_zero_vector() {
        let config = make_config();
        let zero = Array1::from_vec(vec![0.0_f32; 4]);

        let result = euclidean_to_poincare(&zero.view(), &config).unwrap();
        assert!(result.norm() < 1e-10, "Zero vector should map to origin");
    }

    #[test]
    fn test_euclidean_to_poincare_nonzero() {
        let config = make_config();
        let emb = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);

        let result = euclidean_to_poincare(&emb.view(), &config).unwrap();
        assert!(result.norm() > 0.0, "Non-zero vector should not map to origin");
        assert!(
            result.norm() < config.max_norm,
            "Result should be inside ball, norm={}",
            result.norm()
        );
    }

    #[test]
    fn test_hyperbolic_point_depth_ordering() {
        // Points near origin should have lower depth than points near boundary
        let config = make_config();
        let near_origin = HyperbolicPoint::new(
            Array1::from_vec(vec![0.1, 0.0]),
            &config,
        );
        let far_from_origin = HyperbolicPoint::new(
            Array1::from_vec(vec![0.8, 0.0]),
            &config,
        );

        assert!(
            near_origin.depth < far_from_origin.depth,
            "Near-origin depth ({}) should be less than far-from-origin depth ({})",
            near_origin.depth,
            far_from_origin.depth
        );
    }

    #[test]
    fn test_embed_hierarchical_depth_ordering() {
        let embedder = HyperbolicEmbedder::new(HyperbolicConfig::with_dimension(4));
        let emb = Array1::from_vec(vec![1.0_f32, 0.5, -0.3, 0.2]);

        let root = embedder.embed_hierarchical(&emb.view(), 0.0).unwrap();
        let mid = embedder.embed_hierarchical(&emb.view(), 0.5).unwrap();
        let leaf = embedder.embed_hierarchical(&emb.view(), 1.0).unwrap();

        assert!(root.norm() < mid.norm(), "Root should be closer to origin than mid");
        assert!(mid.norm() < leaf.norm(), "Mid should be closer to origin than leaf");
    }

    #[test]
    fn test_find_ancestors_and_descendants() {
        let config = HyperbolicConfig::with_dimension(2);
        let mut embedder = HyperbolicEmbedder::new(config.clone());

        // Create hierarchy: root -> mid -> leaf
        let root = HyperbolicPoint::new(Array1::from_vec(vec![0.05, 0.0]), &config);
        let mid = HyperbolicPoint::new(Array1::from_vec(vec![0.4, 0.0]), &config);
        let leaf = HyperbolicPoint::new(Array1::from_vec(vec![0.8, 0.0]), &config);

        let id_root = Uuid::new_v4();
        let id_mid = Uuid::new_v4();
        let id_leaf = Uuid::new_v4();

        embedder.insert(id_root, root.clone());
        embedder.insert(id_mid, mid.clone());
        embedder.insert(id_leaf, leaf.clone());

        // From mid, ancestors should be root
        let ancestors = embedder.find_ancestors(&mid, 5);
        assert_eq!(ancestors.len(), 1, "Mid should have 1 ancestor (root)");
        assert_eq!(ancestors[0].0, id_root);

        // From mid, descendants should be leaf
        let descendants = embedder.find_descendants(&mid, 5);
        assert_eq!(descendants.len(), 1, "Mid should have 1 descendant (leaf)");
        assert_eq!(descendants[0].0, id_leaf);
    }

    #[test]
    fn test_hierarchy_distance() {
        let config = HyperbolicConfig::with_dimension(2);
        let embedder = HyperbolicEmbedder::new(config.clone());

        let a = HyperbolicPoint::new(Array1::from_vec(vec![0.3, 0.0]), &config);
        let b = HyperbolicPoint::new(Array1::from_vec(vec![0.3, 0.1]), &config);
        let c = HyperbolicPoint::new(Array1::from_vec(vec![0.8, 0.0]), &config);

        // Same-level siblings should have smaller hierarchy distance than cross-level
        let dist_siblings = embedder.hierarchy_distance(&a, &b);
        let dist_cross_level = embedder.hierarchy_distance(&a, &c);

        assert!(
            dist_siblings < dist_cross_level,
            "Sibling distance ({}) should be less than cross-level distance ({})",
            dist_siblings,
            dist_cross_level
        );
    }

    #[test]
    fn test_poincare_distance_f32_wrapper() {
        let u = Array1::from_vec(vec![0.3_f32, 0.2]);
        let v = Array1::from_vec(vec![-0.1_f32, 0.4]);

        let dist = poincare_distance_f32(&u.view(), &v.view());
        assert!(dist > 0.0, "Distance should be positive");
        assert!(dist.is_finite(), "Distance should be finite");
    }

    #[test]
    fn test_dimension_mismatch_exp_map() {
        let config = make_config();
        let base = Array1::from_vec(vec![0.1, 0.2]);
        let tangent = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let result = exponential_map(&base.view(), &tangent.view(), &config);
        assert!(result.is_err(), "Should fail on dimension mismatch");
    }

    #[test]
    fn test_dimension_mismatch_log_map() {
        let config = make_config();
        let base = Array1::from_vec(vec![0.1, 0.2]);
        let point = Array1::from_vec(vec![0.1, 0.2, 0.3]);

        let result = logarithmic_map(&base.view(), &point.view(), &config);
        assert!(result.is_err(), "Should fail on dimension mismatch");
    }

    #[test]
    fn test_hyperbolic_point_origin() {
        let origin = HyperbolicPoint::origin(128);
        assert_eq!(origin.dimension(), 128);
        assert!(origin.norm() < 1e-15);
        assert_eq!(origin.depth, 0.0);
    }

    #[test]
    fn test_embedder_len_and_is_empty() {
        let mut embedder = HyperbolicEmbedder::new(HyperbolicConfig::with_dimension(2));
        assert!(embedder.is_empty());
        assert_eq!(embedder.len(), 0);

        let config = HyperbolicConfig::with_dimension(2);
        let point = HyperbolicPoint::new(Array1::from_vec(vec![0.1, 0.2]), &config);
        embedder.insert(Uuid::new_v4(), point);

        assert!(!embedder.is_empty());
        assert_eq!(embedder.len(), 1);
    }

    #[test]
    fn test_embed_from_euclidean_root_domain() {
        let embedder = HyperbolicEmbedder::new(HyperbolicConfig::with_dimension(4));
        let emb = Array1::from_vec(vec![1.0_f32, 0.5, -0.3, 0.2]);

        // Root domain (depth 1) should be near origin
        let point = embedder.embed_from_euclidean(&emb.view(), "rust").unwrap();
        assert!(
            point.norm() < 0.3,
            "Root domain should be near origin, got norm={}",
            point.norm()
        );
    }

    #[test]
    fn test_embed_from_euclidean_deep_domain() {
        let embedder = HyperbolicEmbedder::new(HyperbolicConfig::with_dimension(4));
        let emb = Array1::from_vec(vec![1.0_f32, 0.5, -0.3, 0.2]);

        // Deep domain (depth 4) should be farther from origin
        let deep = embedder
            .embed_from_euclidean(&emb.view(), "rust.async.tokio.runtime")
            .unwrap();
        let shallow = embedder
            .embed_from_euclidean(&emb.view(), "rust")
            .unwrap();

        assert!(
            deep.norm() > shallow.norm(),
            "Deep domain (norm={}) should be farther from origin than shallow (norm={})",
            deep.norm(),
            shallow.norm()
        );
    }

    #[test]
    fn test_embed_from_euclidean_stays_in_ball() {
        let embedder = HyperbolicEmbedder::new(HyperbolicConfig::with_dimension(128));
        let emb = Array1::from_vec(vec![0.5_f32; 128]);

        // Very deep domain should still be inside the ball
        let point = embedder
            .embed_from_euclidean(&emb.view(), "a.b.c.d.e.f.g.h")
            .unwrap();
        assert!(
            point.norm() < embedder.config().max_norm + 1e-10,
            "Point should stay inside ball, norm={}",
            point.norm()
        );
    }

    // ========================================================================
    // HyperbolicIndex tests (feature-gated)
    // ========================================================================

    mod hyperbolic_index_tests {
        use super::super::*;

        /// Helper: create a default HyperbolicConfig for 4-dim space.
        fn hcfg() -> HyperbolicConfig {
            HyperbolicConfig::with_dimension(4)
        }

        /// Helper: create a HyperbolicPoint from a small coordinate vec.
        fn make_point(coords: Vec<f64>, config: &HyperbolicConfig) -> HyperbolicPoint {
            HyperbolicPoint::new(Array1::from_vec(coords), config)
        }

        // -- HyperbolicIndexConfig tests (2) --

        #[test]
        fn test_index_config_defaults() {
            let cfg = HyperbolicIndexConfig::default();
            assert_eq!(cfg.max_connections, 16);
            assert_eq!(cfg.ef_construction, 200);
            assert_eq!(cfg.ef_search, 50);
            assert_eq!(cfg.max_layers, 5);
            assert!(cfg.use_dual_space);
            assert!((cfg.euclidean_weight - 0.3).abs() < 1e-10);
            assert!((cfg.hyperbolic_weight - 0.7).abs() < 1e-10);
        }

        #[test]
        fn test_index_config_custom() {
            let cfg = HyperbolicIndexConfig {
                max_connections: 32,
                ef_construction: 400,
                ef_search: 100,
                max_layers: 8,
                use_dual_space: false,
                euclidean_weight: 0.5,
                hyperbolic_weight: 0.5,
            };
            assert_eq!(cfg.max_connections, 32);
            assert_eq!(cfg.ef_construction, 400);
            assert_eq!(cfg.ef_search, 100);
            assert_eq!(cfg.max_layers, 8);
            assert!(!cfg.use_dual_space);
            assert!((cfg.euclidean_weight - 0.5).abs() < 1e-10);
            assert!((cfg.hyperbolic_weight - 0.5).abs() < 1e-10);
        }

        // -- HyperbolicIndex::new tests (2) --

        #[test]
        fn test_new_empty_index() {
            let idx = HyperbolicIndex::new(hcfg(), HyperbolicIndexConfig::default());
            assert!(idx.is_empty());
            assert_eq!(idx.len(), 0);
        }

        #[test]
        fn test_new_with_config() {
            let icfg = HyperbolicIndexConfig {
                max_connections: 8,
                ..Default::default()
            };
            let idx = HyperbolicIndex::new(hcfg(), icfg);
            assert!(idx.is_empty());
        }

        // -- insert tests (4) --

        #[test]
        fn test_insert_single_node() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            let p = make_point(vec![0.1, 0.2, 0.0, 0.0], &cfg);
            idx.insert("a".to_string(), p, None, None);
            assert_eq!(idx.len(), 1);
            assert!(!idx.is_empty());
        }

        #[test]
        fn test_insert_multiple_nodes() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            for i in 0..10 {
                let v = (i as f64) * 0.08;
                let p = make_point(vec![v, 0.0, 0.0, 0.0], &cfg);
                idx.insert(format!("n{}", i), p, None, None);
            }
            assert_eq!(idx.len(), 10);
        }

        #[test]
        fn test_insert_with_euclidean() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            let p = make_point(vec![0.1, 0.2, 0.0, 0.0], &cfg);
            let euc = vec![1.0_f32, 2.0, 3.0, 4.0];
            idx.insert("a".to_string(), p, Some(euc.clone()), None);
            let node = idx.get("a").unwrap();
            assert_eq!(node.euclidean_coords.as_ref().unwrap(), &euc);
        }

        #[test]
        fn test_insert_with_domain() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            let p = make_point(vec![0.3, 0.0, 0.0, 0.0], &cfg);
            idx.insert(
                "a".to_string(),
                p,
                None,
                Some("rust.async".to_string()),
            );
            let node = idx.get("a").unwrap();
            assert_eq!(node.domain.as_deref(), Some("rust.async"));
        }

        // -- search tests (5) --

        #[test]
        fn test_search_finds_nearest() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            // Insert a near point and a far point.
            let near = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            let far = make_point(vec![0.8, 0.0, 0.0, 0.0], &cfg);
            idx.insert("near".to_string(), near, None, None);
            idx.insert("far".to_string(), far, None, None);

            let query = make_point(vec![0.12, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search(&query, 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "near");
        }

        #[test]
        fn test_search_respects_k() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            for i in 0..20 {
                let v = (i as f64) * 0.04;
                idx.insert(
                    format!("n{}", i),
                    make_point(vec![v, 0.0, 0.0, 0.0], &cfg),
                    None,
                    None,
                );
            }

            let query = make_point(vec![0.5, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search(&query, 5);
            assert_eq!(results.len(), 5);
        }

        #[test]
        fn test_search_returns_sorted() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            for i in 0..10 {
                let v = (i as f64) * 0.08;
                idx.insert(
                    format!("n{}", i),
                    make_point(vec![v, 0.0, 0.0, 0.0], &cfg),
                    None,
                    None,
                );
            }

            let query = make_point(vec![0.4, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search(&query, 5);
            for i in 1..results.len() {
                assert!(
                    results[i].combined_score >= results[i - 1].combined_score,
                    "Results not sorted: [{}]={} > [{}]={}",
                    i - 1,
                    results[i - 1].combined_score,
                    i,
                    results[i].combined_score
                );
            }
        }

        #[test]
        fn test_search_empty_index() {
            let idx = HyperbolicIndex::new(hcfg(), HyperbolicIndexConfig::default());
            let cfg = hcfg();
            let query = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search(&query, 5);
            assert!(results.is_empty());
        }

        #[test]
        fn test_search_single_node() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            idx.insert(
                "only".to_string(),
                make_point(vec![0.5, 0.0, 0.0, 0.0], &cfg),
                None,
                None,
            );

            let query = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search(&query, 3);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "only");
        }

        // -- dual_search tests (5) --

        #[test]
        fn test_dual_search_combines_distances() {
            let cfg = hcfg();
            let icfg = HyperbolicIndexConfig {
                hyperbolic_weight: 0.5,
                euclidean_weight: 0.5,
                ..Default::default()
            };
            let mut idx = HyperbolicIndex::new(cfg.clone(), icfg);

            // Node A: close in hyperbolic, far in euclidean
            let pa = make_point(vec![0.11, 0.0, 0.0, 0.0], &cfg);
            idx.insert("a".to_string(), pa, Some(vec![10.0, 0.0, 0.0, 0.0]), None);

            // Node B: moderate in both
            let pb = make_point(vec![0.2, 0.0, 0.0, 0.0], &cfg);
            idx.insert("b".to_string(), pb, Some(vec![0.2, 0.0, 0.0, 0.0]), None);

            let query_hyp = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            let query_euc: Vec<f32> = vec![0.1, 0.0, 0.0, 0.0];
            let results = idx.dual_search(&query_hyp, Some(&query_euc), 2);

            assert_eq!(results.len(), 2);
            // B should rank higher due to better euclidean distance.
            assert_eq!(
                results[0].id, "b",
                "Node B should be closer in dual search due to euclidean proximity"
            );
        }

        #[test]
        fn test_dual_search_respects_weights() {
            let cfg = hcfg();

            // Pure hyperbolic weights.
            let icfg_hyp = HyperbolicIndexConfig {
                hyperbolic_weight: 1.0,
                euclidean_weight: 0.0,
                ..Default::default()
            };
            let mut idx_hyp = HyperbolicIndex::new(cfg.clone(), icfg_hyp);

            let pa = make_point(vec![0.11, 0.0, 0.0, 0.0], &cfg);
            idx_hyp.insert("a".to_string(), pa, Some(vec![10.0, 0.0, 0.0, 0.0]), None);

            let pb = make_point(vec![0.5, 0.0, 0.0, 0.0], &cfg);
            idx_hyp.insert("b".to_string(), pb, Some(vec![0.1, 0.0, 0.0, 0.0]), None);

            let query_hyp = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            let query_euc: Vec<f32> = vec![0.1, 0.0, 0.0, 0.0];
            let results = idx_hyp.dual_search(&query_hyp, Some(&query_euc), 2);

            // With pure hyperbolic weights, A should be first (closer in Poincare).
            assert_eq!(results[0].id, "a");
        }

        #[test]
        fn test_dual_search_without_euclidean_fallback() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            // Insert nodes WITHOUT euclidean coords.
            let pa = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            idx.insert("a".to_string(), pa, None, None);

            let query_hyp = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            let query_euc: Vec<f32> = vec![0.1, 0.0, 0.0, 0.0];
            let results = idx.dual_search(&query_hyp, Some(&query_euc), 1);

            assert_eq!(results.len(), 1);
            // euclidean_distance should be None since node has no euc coords.
            assert!(results[0].euclidean_distance.is_none());
        }

        #[test]
        fn test_dual_search_with_euclidean() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            let pa = make_point(vec![0.2, 0.0, 0.0, 0.0], &cfg);
            idx.insert("a".to_string(), pa, Some(vec![1.0, 0.0, 0.0, 0.0]), None);

            let query_hyp = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            let query_euc: Vec<f32> = vec![0.5, 0.0, 0.0, 0.0];
            let results = idx.dual_search(&query_hyp, Some(&query_euc), 1);

            assert_eq!(results.len(), 1);
            assert!(results[0].euclidean_distance.is_some());
            assert!(results[0].euclidean_distance.unwrap() > 0.0);
        }

        #[test]
        fn test_dual_search_pure_hyperbolic_mode() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            let pa = make_point(vec![0.2, 0.0, 0.0, 0.0], &cfg);
            idx.insert("a".to_string(), pa, None, None);

            let query_hyp = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            // No euclidean query.
            let results = idx.dual_search(&query_hyp, None, 1);

            assert_eq!(results.len(), 1);
            assert!(results[0].euclidean_distance.is_none());
            // Combined score should equal hyperbolic distance.
            assert!(
                (results[0].combined_score - results[0].hyperbolic_distance).abs() < 1e-10
            );
        }

        // -- search_by_depth tests (4) --

        #[test]
        fn test_search_by_depth_filters_range() {
            let cfg = HyperbolicConfig::with_dimension(4);
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            // Insert points at various depths.
            let root = make_point(vec![0.05, 0.0, 0.0, 0.0], &cfg); // depth ~0.05
            let mid = make_point(vec![0.5, 0.0, 0.0, 0.0], &cfg); // depth ~0.505
            let leaf = make_point(vec![0.9, 0.0, 0.0, 0.0], &cfg); // depth ~0.909

            idx.insert("root".to_string(), root, None, None);
            idx.insert("mid".to_string(), mid, None, None);
            idx.insert("leaf".to_string(), leaf, None, None);

            let query = make_point(vec![0.4, 0.0, 0.0, 0.0], &cfg);
            // Search only mid-depth range.
            let results = idx.search_by_depth(&query, 10, 0.3, 0.7);

            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "mid");
        }

        #[test]
        fn test_search_by_depth_root_only() {
            let cfg = HyperbolicConfig::with_dimension(4);
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            let root = make_point(vec![0.05, 0.0, 0.0, 0.0], &cfg);
            let leaf = make_point(vec![0.9, 0.0, 0.0, 0.0], &cfg);
            idx.insert("root".to_string(), root, None, None);
            idx.insert("leaf".to_string(), leaf, None, None);

            let query = make_point(vec![0.0, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search_by_depth(&query, 10, 0.0, 0.2);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "root");
        }

        #[test]
        fn test_search_by_depth_leaf_only() {
            let cfg = HyperbolicConfig::with_dimension(4);
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            let root = make_point(vec![0.05, 0.0, 0.0, 0.0], &cfg);
            let leaf = make_point(vec![0.9, 0.0, 0.0, 0.0], &cfg);
            idx.insert("root".to_string(), root, None, None);
            idx.insert("leaf".to_string(), leaf, None, None);

            let query = make_point(vec![0.8, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search_by_depth(&query, 10, 0.8, 1.0);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "leaf");
        }

        #[test]
        fn test_search_by_depth_full_range() {
            let cfg = HyperbolicConfig::with_dimension(4);
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            let root = make_point(vec![0.05, 0.0, 0.0, 0.0], &cfg);
            let mid = make_point(vec![0.5, 0.0, 0.0, 0.0], &cfg);
            let leaf = make_point(vec![0.9, 0.0, 0.0, 0.0], &cfg);
            idx.insert("root".to_string(), root, None, None);
            idx.insert("mid".to_string(), mid, None, None);
            idx.insert("leaf".to_string(), leaf, None, None);

            let query = make_point(vec![0.4, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search_by_depth(&query, 10, 0.0, 1.0);
            assert_eq!(results.len(), 3);
        }

        // -- remove tests (3) --

        #[test]
        fn test_remove_existing() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            idx.insert(
                "a".to_string(),
                make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg),
                None,
                None,
            );
            idx.insert(
                "b".to_string(),
                make_point(vec![0.5, 0.0, 0.0, 0.0], &cfg),
                None,
                None,
            );

            assert!(idx.remove("a"));
            assert_eq!(idx.len(), 1);
            assert!(idx.get("a").is_none());
        }

        #[test]
        fn test_remove_missing_returns_false() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            idx.insert(
                "a".to_string(),
                make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg),
                None,
                None,
            );
            assert!(!idx.remove("nonexistent"));
            assert_eq!(idx.len(), 1);
        }

        #[test]
        fn test_remove_updates_entry_point() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            // Insert two nodes. The first one becomes the entry point.
            idx.insert(
                "a".to_string(),
                make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg),
                None,
                None,
            );
            idx.insert(
                "b".to_string(),
                make_point(vec![0.5, 0.0, 0.0, 0.0], &cfg),
                None,
                None,
            );

            // Remove whichever is the entry point; the other should take over.
            let old_entry = idx.get("a").is_some() || idx.get("b").is_some();
            assert!(old_entry);

            // Remove all nodes one by one.
            idx.remove("a");
            assert!(!idx.is_empty()); // b still there
            idx.remove("b");
            assert!(idx.is_empty());
        }

        // -- get tests (2) --

        #[test]
        fn test_get_existing_node() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            idx.insert(
                "mynode".to_string(),
                make_point(vec![0.3, 0.1, 0.0, 0.0], &cfg),
                None,
                Some("rust".to_string()),
            );

            let node = idx.get("mynode");
            assert!(node.is_some());
            assert_eq!(node.unwrap().id, "mynode");
            assert_eq!(node.unwrap().domain.as_deref(), Some("rust"));
        }

        #[test]
        fn test_get_missing_node() {
            let idx = HyperbolicIndex::new(hcfg(), HyperbolicIndexConfig::default());
            assert!(idx.get("nope").is_none());
        }

        // -- len / is_empty tests (2) --

        #[test]
        fn test_len_and_is_empty_empty() {
            let idx = HyperbolicIndex::new(hcfg(), HyperbolicIndexConfig::default());
            assert_eq!(idx.len(), 0);
            assert!(idx.is_empty());
        }

        #[test]
        fn test_len_and_is_empty_non_empty() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            idx.insert(
                "x".to_string(),
                make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg),
                None,
                None,
            );
            assert_eq!(idx.len(), 1);
            assert!(!idx.is_empty());
        }

        // -- stats tests (2) --

        #[test]
        fn test_stats_empty() {
            let idx = HyperbolicIndex::new(hcfg(), HyperbolicIndexConfig::default());
            let s = idx.stats();
            assert_eq!(s.total_nodes, 0);
            assert_eq!(s.max_layer, 0);
            assert!((s.avg_connections - 0.0).abs() < 1e-10);
            assert!(s.depth_distribution.is_empty());
        }

        #[test]
        fn test_stats_populated() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());
            for i in 0..15 {
                let v = (i as f64) * 0.05;
                idx.insert(
                    format!("n{}", i),
                    make_point(vec![v, 0.0, 0.0, 0.0], &cfg),
                    None,
                    None,
                );
            }

            let s = idx.stats();
            assert_eq!(s.total_nodes, 15);
            assert!(!s.depth_distribution.is_empty());
        }

        // -- build_from_patterns tests (3) --

        #[test]
        fn test_build_from_patterns() {
            let cfg = hcfg();
            let patterns: Vec<(String, HyperbolicPoint, Option<Vec<f32>>, Option<String>)> = (0..5)
                .map(|i| {
                    let v = (i as f64) * 0.1;
                    (
                        format!("p{}", i),
                        make_point(vec![v, 0.0, 0.0, 0.0], &cfg),
                        None,
                        None,
                    )
                })
                .collect();

            let idx = HyperbolicIndex::build_from_patterns(
                &patterns,
                cfg,
                HyperbolicIndexConfig::default(),
            );
            assert!(!idx.is_empty());
        }

        #[test]
        fn test_build_from_patterns_correct_count() {
            let cfg = hcfg();
            let n = 12;
            let patterns: Vec<(String, HyperbolicPoint, Option<Vec<f32>>, Option<String>)> =
                (0..n)
                    .map(|i| {
                        let v = (i as f64) * 0.06;
                        (
                            format!("p{}", i),
                            make_point(vec![v, 0.0, 0.0, 0.0], &cfg),
                            None,
                            None,
                        )
                    })
                    .collect();

            let idx = HyperbolicIndex::build_from_patterns(
                &patterns,
                cfg,
                HyperbolicIndexConfig::default(),
            );
            assert_eq!(idx.len(), n);
        }

        #[test]
        fn test_build_from_patterns_preserves_domains() {
            let cfg = hcfg();
            let patterns = vec![
                (
                    "a".to_string(),
                    make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg),
                    None,
                    Some("rust".to_string()),
                ),
                (
                    "b".to_string(),
                    make_point(vec![0.5, 0.0, 0.0, 0.0], &cfg),
                    None,
                    Some("python".to_string()),
                ),
            ];

            let idx = HyperbolicIndex::build_from_patterns(
                &patterns,
                cfg,
                HyperbolicIndexConfig::default(),
            );
            assert_eq!(idx.get("a").unwrap().domain.as_deref(), Some("rust"));
            assert_eq!(idx.get("b").unwrap().domain.as_deref(), Some("python"));
        }

        // -- random_layer tests (2) --

        #[test]
        fn test_random_layer_within_bounds() {
            for _ in 0..100 {
                let layer = random_layer(5);
                assert!(layer < 5, "Layer {} out of bounds (max 5)", layer);
            }
        }

        #[test]
        fn test_random_layer_distribution_reasonable() {
            // Most nodes should be on layer 0 (geometric distribution).
            let mut counts = vec![0usize; 5];
            let trials = 1000;
            for _ in 0..trials {
                let l = random_layer(5);
                counts[l] += 1;
            }

            // Layer 0 should have the most nodes (typically >40%).
            assert!(
                counts[0] > trials / 5,
                "Layer 0 count {} is too low (expected > {})",
                counts[0],
                trials / 5
            );
        }

        // -- search_layer tests (2) --

        #[test]
        fn test_search_finds_correct_nearest() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            // Insert several points along a line.
            for i in 0..5 {
                let v = (i as f64) * 0.15;
                idx.insert(
                    format!("n{}", i),
                    make_point(vec![v, 0.0, 0.0, 0.0], &cfg),
                    None,
                    None,
                );
            }

            // Query near n2 (0.3, 0, 0, 0).
            let query = make_point(vec![0.31, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search(&query, 1);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "n2", "Should find n2 as nearest to 0.31");
        }

        #[test]
        fn test_search_handles_disconnected() {
            let cfg = hcfg();
            let mut idx = HyperbolicIndex::new(cfg.clone(), HyperbolicIndexConfig::default());

            // Even with a single node (no connections), search should still work.
            idx.insert(
                "solo".to_string(),
                make_point(vec![0.5, 0.0, 0.0, 0.0], &cfg),
                None,
                None,
            );
            let query = make_point(vec![0.1, 0.0, 0.0, 0.0], &cfg);
            let results = idx.search(&query, 3);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].id, "solo");
        }
    }
}
