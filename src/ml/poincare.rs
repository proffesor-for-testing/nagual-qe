//! Poincare ball model operations and k-nearest-neighbor search.
//!
//! Provides a complete implementation of the Poincare ball model including
//! Mobius gyrovector space operations, parallel transport, midpoint computation,
//! and a k-nearest-neighbor search index using Poincare distance.
//!
//! # Mobius Gyrovector Space
//!
//! The Poincare ball with Mobius addition forms a gyrovector space, which is
//! the hyperbolic analogue of a vector space. Key operations:
//!
//! - **Mobius addition**: `x (+) y` - the hyperbolic analogue of vector addition
//! - **Mobius scalar multiplication**: `r (*) x` - scaling in hyperbolic space
//! - **Parallel transport**: Moving vectors between tangent spaces
//!
//! # Example
//!
//! ```ignore
//! use nagual::ml::poincare::{PoincareModel, PoincareBall, PoincareKNN};
//! use ndarray::Array1;
//! use uuid::Uuid;
//!
//! let ball = PoincareBall::new(128, -1.0);
//! let model = PoincareModel::new(ball);
//!
//! let x = Array1::from_elem(128, 0.01);
//! let y = Array1::from_elem(128, 0.02);
//! let sum = model.mobius_add(&x.view(), &y.view());
//! ```

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::hyperbolic::{poincare_distance, project_to_ball, HyperbolicConfig};
use super::{MlError, MlResult};

// ============================================================================
// PoincareBall
// ============================================================================

/// Definition of a Poincare ball with dimension and curvature.
///
/// The Poincare ball `B_c^n` is the open ball of radius `1/sqrt(c)` in `R^n`
/// equipped with the Riemannian metric tensor:
///
/// ```text
/// g_x = (lambda_x)^2 * g_E
/// where lambda_x = 2 / (1 - c * ||x||^2)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoincareBall {
    /// Dimension of the ball.
    pub dimension: usize,

    /// Curvature parameter (negative, default -1.0).
    pub curvature: f64,

    /// Maximum norm for numerical stability.
    pub max_norm: f64,

    /// Epsilon for numerical stability.
    pub eps: f64,
}

impl PoincareBall {
    /// Create a new Poincare ball with the given dimension and curvature.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Dimension of the embedding space
    /// * `curvature` - Curvature parameter (should be negative)
    pub fn new(dimension: usize, curvature: f64) -> Self {
        Self {
            dimension,
            curvature,
            max_norm: 0.99,
            eps: 1e-7,
        }
    }

    /// Create a Poincare ball with default curvature (-1.0).
    pub fn with_dimension(dimension: usize) -> Self {
        Self::new(dimension, -1.0)
    }

    /// Get the absolute curvature value.
    #[inline]
    pub fn abs_curvature(&self) -> f64 {
        self.curvature.abs()
    }

    /// Compute the conformal factor (lambda) at a point.
    ///
    /// ```text
    /// lambda_x = 2 / (1 - c * ||x||^2)
    /// ```
    #[inline]
    pub fn conformal_factor(&self, x: &ArrayView1<f64>) -> f64 {
        let c = self.abs_curvature();
        let x_sq = x.dot(x);
        2.0 / (1.0 - c * x_sq).max(self.eps)
    }

    /// Check if a point is inside the ball.
    pub fn contains(&self, x: &ArrayView1<f64>) -> bool {
        let c = self.abs_curvature();
        let norm_sq = x.dot(x);
        norm_sq < 1.0 / c
    }

    /// Convert to HyperbolicConfig for interop.
    pub fn to_config(&self) -> HyperbolicConfig {
        HyperbolicConfig {
            curvature: self.curvature,
            dimension: self.dimension,
            max_norm: self.max_norm,
            eps: self.eps,
        }
    }
}

impl Default for PoincareBall {
    fn default() -> Self {
        Self::with_dimension(128)
    }
}

// ============================================================================
// PoincareModel
// ============================================================================

/// High-level model for operations in the Poincare ball.
///
/// Wraps a `PoincareBall` and provides methods for Mobius algebra,
/// parallel transport, and midpoint computation.
#[derive(Debug, Clone)]
pub struct PoincareModel {
    /// The underlying Poincare ball.
    pub ball: PoincareBall,
}

impl PoincareModel {
    /// Create a new Poincare model with the given ball.
    pub fn new(ball: PoincareBall) -> Self {
        Self { ball }
    }

    /// Create a model with default 128-dimensional ball.
    pub fn default_model() -> Self {
        Self::new(PoincareBall::default())
    }

    /// Mobius addition in the Poincare ball.
    ///
    /// ```text
    /// x (+)_c y = ((1 + 2c<x,y> + c||y||^2) * x + (1 - c||x||^2) * y)
    ///             / (1 + 2c<x,y> + c^2 * ||x||^2 * ||y||^2)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - First operand in the ball
    /// * `y` - Second operand in the ball
    ///
    /// # Returns
    ///
    /// The Mobius sum, projected back into the ball.
    pub fn mobius_add(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let c = self.ball.abs_curvature();
        let eps = self.ball.eps;

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

        project_to_ball(&result.view(), self.ball.max_norm)
    }

    /// Mobius scalar multiplication.
    ///
    /// ```text
    /// r (*)_c x = (1/sqrt(c)) * tanh(r * arctanh(sqrt(c) * ||x||)) * (x / ||x||)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `r` - Scalar multiplier
    /// * `x` - Point in the ball
    ///
    /// # Returns
    ///
    /// The scaled point, projected back into the ball.
    pub fn mobius_scalar_mul(&self, r: f64, x: &ArrayView1<f64>) -> Array1<f64> {
        let c = self.ball.abs_curvature();
        let sqrt_c = c.sqrt();
        let eps = self.ball.eps;

        let x_norm = x.dot(x).sqrt();

        if x_norm < eps {
            return x.to_owned();
        }

        let atanh_arg = (sqrt_c * x_norm).min(1.0 - eps);
        let target_norm = (r * atanh_arg.atanh()).tanh() / sqrt_c;

        let direction = x.mapv(|v| v / x_norm);
        let result = direction.mapv(|v| v * target_norm);

        project_to_ball(&result.view(), self.ball.max_norm)
    }

    /// Parallel transport of a tangent vector from one point to another.
    ///
    /// Transports tangent vector `v` from the tangent space at `x` to
    /// the tangent space at `y`.
    ///
    /// ```text
    /// P_{x->y}(v) = (lambda_x / lambda_y) * gyration(y, -x, v)
    /// ```
    ///
    /// For simplicity, we use the conformal factor ratio approach:
    ///
    /// ```text
    /// P_{x->y}(v) = (lambda_x / lambda_y) * v
    /// ```
    ///
    /// This is an approximation that works well when x and y are close.
    ///
    /// # Arguments
    ///
    /// * `x` - Source point
    /// * `y` - Target point
    /// * `v` - Tangent vector at x
    ///
    /// # Returns
    ///
    /// The transported tangent vector at y.
    pub fn parallel_transport(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        v: &ArrayView1<f64>,
    ) -> MlResult<Array1<f64>> {
        if x.len() != y.len() || x.len() != v.len() {
            return Err(MlError::Hyperbolic(format!(
                "Dimension mismatch in parallel transport: x={}, y={}, v={}",
                x.len(),
                y.len(),
                v.len()
            )));
        }

        let lambda_x = self.ball.conformal_factor(x);
        let lambda_y = self.ball.conformal_factor(y);

        let scale = lambda_x / lambda_y;
        Ok(v.mapv(|val| val * scale))
    }

    /// Compute the Einstein midpoint of multiple points.
    ///
    /// The Einstein midpoint generalizes the notion of centroid to hyperbolic
    /// space. It is computed using the Klein model as an intermediate:
    ///
    /// ```text
    /// midpoint = sum(gamma_i * x_i) / sum(gamma_i)
    /// where gamma_i = 1 / sqrt(1 - c * ||x_i||^2)  (Lorentz factor)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `points` - Slice of points in the Poincare ball
    ///
    /// # Returns
    ///
    /// The Einstein midpoint, or error if points is empty.
    pub fn midpoint(&self, points: &[ArrayView1<f64>]) -> MlResult<Array1<f64>> {
        if points.is_empty() {
            return Err(MlError::Hyperbolic(
                "Cannot compute midpoint of empty set".to_string(),
            ));
        }

        if points.len() == 1 {
            return Ok(points[0].to_owned());
        }

        let c = self.ball.abs_curvature();
        let dim = points[0].len();

        // Compute Lorentz-weighted sum
        let mut weighted_sum: Array1<f64> = Array1::zeros(dim);
        let mut weight_total = 0.0;

        for p in points {
            let norm_sq = p.dot(p);
            let gamma = 1.0 / (1.0 - c * norm_sq).max(self.ball.eps).sqrt();
            weight_total += gamma;
            for i in 0..dim {
                weighted_sum[i] += gamma * p[i];
            }
        }

        if weight_total < self.ball.eps {
            return Ok(Array1::zeros(dim));
        }

        let result = weighted_sum.mapv(|x| x / weight_total);
        Ok(project_to_ball(&result.view(), self.ball.max_norm))
    }

    /// Compute the Poincare distance between two points.
    ///
    /// Convenience wrapper around [`poincare_distance`].
    pub fn distance(&self, u: &ArrayView1<f64>, v: &ArrayView1<f64>) -> f64 {
        poincare_distance(u, v)
    }
}

// ============================================================================
// PoincareKNN
// ============================================================================

/// Entry in the KNN priority queue.
#[derive(Debug, Clone)]
struct KnnEntry {
    id: Uuid,
    distance: f64,
    #[allow(dead_code)]
    norm: f64,
}

impl PartialEq for KnnEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for KnnEntry {}

impl PartialOrd for KnnEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KnnEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: reverse order so smallest distance is popped last
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// K-nearest-neighbor search index using Poincare distance.
///
/// Provides brute-force KNN search in hyperbolic space with support
/// for filtered searches (ancestors, descendants).
///
/// For large-scale deployments, this should be replaced with an
/// approximate NN index (e.g., VP-tree for metric spaces), but
/// the brute-force approach is correct and sufficient for moderate
/// dataset sizes (< 100K points).
#[derive(Debug, Clone)]
pub struct PoincareKNN {
    /// Stored points indexed by UUID.
    points: Vec<(Uuid, Array1<f64>)>,

    /// Maximum norm for projection.
    max_norm: f64,
}

impl PoincareKNN {
    /// Create a new empty KNN index.
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            max_norm: 0.99,
        }
    }

    /// Create a KNN index with a custom max_norm.
    pub fn with_max_norm(max_norm: f64) -> Self {
        Self {
            points: Vec::new(),
            max_norm,
        }
    }

    /// Insert a point into the index.
    ///
    /// The point is projected onto the ball if necessary.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the point
    /// * `point` - Coordinates in the Poincare ball
    pub fn insert(&mut self, id: Uuid, point: Array1<f64>) {
        let projected = project_to_ball(&point.view(), self.max_norm);
        self.points.push((id, projected));
    }

    /// Get the number of points in the index.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Search for the k nearest neighbors by Poincare distance.
    ///
    /// # Arguments
    ///
    /// * `query` - The query point
    /// * `k` - Maximum number of neighbors to return
    ///
    /// # Returns
    ///
    /// Vector of (id, distance) pairs sorted by ascending distance.
    pub fn search(&self, query: &ArrayView1<f64>, k: usize) -> Vec<(Uuid, f64)> {
        if self.points.is_empty() || k == 0 {
            return Vec::new();
        }

        // Use a max-heap of size k to track closest points
        let mut heap: BinaryHeap<KnnEntry> = BinaryHeap::new();

        for (id, point) in &self.points {
            let dist = poincare_distance(query, &point.view());

            if heap.len() < k {
                heap.push(KnnEntry {
                    id: *id,
                    distance: dist,
                    norm: point.dot(point).sqrt(),
                });
            } else if let Some(top) = heap.peek() {
                // top is the FARTHEST point in our k-best (max-heap)
                if dist < top.distance {
                    heap.pop();
                    heap.push(KnnEntry {
                        id: *id,
                        distance: dist,
                        norm: point.dot(point).sqrt(),
                    });
                }
            }
        }

        // Convert to sorted Vec
        let mut results: Vec<(Uuid, f64)> = heap.into_iter().map(|e| (e.id, e.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Search for the k nearest ancestors (points closer to origin).
    ///
    /// An ancestor is a point whose norm is strictly less than the query
    /// point's norm, indicating it represents a more general concept.
    ///
    /// # Arguments
    ///
    /// * `query` - The query point
    /// * `k` - Maximum number of ancestors to return
    ///
    /// # Returns
    ///
    /// Vector of (id, distance) pairs sorted by ascending distance.
    pub fn search_ancestors(&self, query: &ArrayView1<f64>, k: usize) -> Vec<(Uuid, f64)> {
        if self.points.is_empty() || k == 0 {
            return Vec::new();
        }

        let query_norm = query.dot(query).sqrt();

        let mut heap: BinaryHeap<KnnEntry> = BinaryHeap::new();

        for (id, point) in &self.points {
            let point_norm = point.dot(point).sqrt();

            // Only consider points closer to origin (ancestors)
            if point_norm >= query_norm {
                continue;
            }

            let dist = poincare_distance(query, &point.view());

            if heap.len() < k {
                heap.push(KnnEntry {
                    id: *id,
                    distance: dist,
                    norm: point_norm,
                });
            } else if let Some(top) = heap.peek() {
                if dist < top.distance {
                    heap.pop();
                    heap.push(KnnEntry {
                        id: *id,
                        distance: dist,
                        norm: point_norm,
                    });
                }
            }
        }

        let mut results: Vec<(Uuid, f64)> = heap.into_iter().map(|e| (e.id, e.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Search for the k nearest descendants (points further from origin).
    ///
    /// A descendant is a point whose norm is strictly greater than the
    /// query point's norm, indicating it represents a more specific concept.
    ///
    /// # Arguments
    ///
    /// * `query` - The query point
    /// * `k` - Maximum number of descendants to return
    ///
    /// # Returns
    ///
    /// Vector of (id, distance) pairs sorted by ascending distance.
    pub fn search_descendants(&self, query: &ArrayView1<f64>, k: usize) -> Vec<(Uuid, f64)> {
        if self.points.is_empty() || k == 0 {
            return Vec::new();
        }

        let query_norm = query.dot(query).sqrt();

        let mut heap: BinaryHeap<KnnEntry> = BinaryHeap::new();

        for (id, point) in &self.points {
            let point_norm = point.dot(point).sqrt();

            // Only consider points further from origin (descendants)
            if point_norm <= query_norm {
                continue;
            }

            let dist = poincare_distance(query, &point.view());

            if heap.len() < k {
                heap.push(KnnEntry {
                    id: *id,
                    distance: dist,
                    norm: point_norm,
                });
            } else if let Some(top) = heap.peek() {
                if dist < top.distance {
                    heap.pop();
                    heap.push(KnnEntry {
                        id: *id,
                        distance: dist,
                        norm: point_norm,
                    });
                }
            }
        }

        let mut results: Vec<(Uuid, f64)> = heap.into_iter().map(|e| (e.id, e.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Remove a point by ID.
    ///
    /// # Returns
    ///
    /// `true` if the point was found and removed.
    pub fn remove(&mut self, id: &Uuid) -> bool {
        let len_before = self.points.len();
        self.points.retain(|(pid, _)| pid != id);
        self.points.len() < len_before
    }

    /// Clear all points from the index.
    pub fn clear(&mut self) {
        self.points.clear();
    }
}

impl Default for PoincareKNN {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_ball() -> PoincareBall {
        PoincareBall::new(4, -1.0)
    }

    fn make_model() -> PoincareModel {
        PoincareModel::new(make_ball())
    }

    #[test]
    fn test_poincare_ball_creation() {
        let ball = PoincareBall::new(128, -1.0);
        assert_eq!(ball.dimension, 128);
        assert_eq!(ball.curvature, -1.0);
        assert_eq!(ball.abs_curvature(), 1.0);
    }

    #[test]
    fn test_poincare_ball_contains() {
        let ball = PoincareBall::new(2, -1.0);

        let inside = Array1::from_vec(vec![0.3, 0.4]);
        assert!(ball.contains(&inside.view()));

        // Point outside the ball (norm > 1/sqrt(c) = 1.0)
        let outside = Array1::from_vec(vec![0.8, 0.8]); // norm ~= 1.13
        assert!(!ball.contains(&outside.view()));
    }

    #[test]
    fn test_conformal_factor() {
        let ball = PoincareBall::new(2, -1.0);

        // At origin, lambda = 2 / (1 - 0) = 2
        let origin = Array1::from_vec(vec![0.0, 0.0]);
        let lambda = ball.conformal_factor(&origin.view());
        assert!((lambda - 2.0).abs() < 1e-10);

        // Closer to boundary, lambda increases
        let near_boundary = Array1::from_vec(vec![0.9, 0.0]);
        let lambda_near = ball.conformal_factor(&near_boundary.view());
        assert!(lambda_near > lambda, "Conformal factor should increase near boundary");
    }

    #[test]
    fn test_mobius_add_identity() {
        let model = make_model();

        // x (+) 0 = x
        let x = Array1::from_vec(vec![0.3, 0.2, -0.1, 0.0]);
        let zero = Array1::zeros(4);

        let result = model.mobius_add(&x.view(), &zero.view());

        for (a, b) in x.iter().zip(result.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "x (+) 0 should equal x: {:.8} vs {:.8}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_mobius_add_stays_in_ball() {
        let model = make_model();

        let x = Array1::from_vec(vec![0.5, 0.3, -0.2, 0.1]);
        let y = Array1::from_vec(vec![-0.3, 0.4, 0.2, -0.1]);

        let result = model.mobius_add(&x.view(), &y.view());
        let norm = result.dot(&result).sqrt();

        assert!(
            norm < 1.0,
            "Mobius addition result should be inside ball, norm={}",
            norm
        );
    }

    #[test]
    fn test_mobius_scalar_mul_zero() {
        let model = make_model();

        // 0 * x = 0
        let x = Array1::from_vec(vec![0.3, 0.2, -0.1, 0.4]);
        let result = model.mobius_scalar_mul(0.0, &x.view());

        let norm = result.dot(&result).sqrt();
        assert!(norm < 1e-10, "0 * x should be near zero, norm={}", norm);
    }

    #[test]
    fn test_mobius_scalar_mul_one() {
        let model = make_model();

        // 1 * x = x
        let x = Array1::from_vec(vec![0.3, 0.2, -0.1, 0.0]);
        let result = model.mobius_scalar_mul(1.0, &x.view());

        for (a, b) in x.iter().zip(result.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "1 * x should equal x: {:.8} vs {:.8}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_mobius_scalar_mul_scaling() {
        let model = make_model();

        let x = Array1::from_vec(vec![0.1, 0.0, 0.0, 0.0]);

        // 2 * x should be further from origin than x
        let scaled = model.mobius_scalar_mul(2.0, &x.view());
        let norm_x = x.dot(&x).sqrt();
        let norm_scaled = scaled.dot(&scaled).sqrt();

        assert!(
            norm_scaled > norm_x,
            "2 * x should be further from origin: {:.6} vs {:.6}",
            norm_scaled,
            norm_x
        );
    }

    #[test]
    fn test_parallel_transport_dimension_mismatch() {
        let model = make_model();

        let x = Array1::from_vec(vec![0.1, 0.2]);
        let y = Array1::from_vec(vec![0.3, 0.4, 0.5]);
        let v = Array1::from_vec(vec![0.01, 0.02]);

        let result = model.parallel_transport(&x.view(), &y.view(), &v.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_transport_same_point() {
        let model = make_model();

        // Transport from x to x should be identity
        let x = Array1::from_vec(vec![0.1, 0.2, 0.0, 0.0]);
        let v = Array1::from_vec(vec![0.05, -0.03, 0.01, 0.0]);

        let result = model.parallel_transport(&x.view(), &x.view(), &v.view()).unwrap();

        for (a, b) in v.iter().zip(result.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Transport to same point should be identity"
            );
        }
    }

    #[test]
    fn test_midpoint_single_point() {
        let model = make_model();

        let p = Array1::from_vec(vec![0.3, 0.2, -0.1, 0.0]);
        let result = model.midpoint(&[p.view()]).unwrap();

        for (a, b) in p.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-10, "Midpoint of single point should be itself");
        }
    }

    #[test]
    fn test_midpoint_symmetric() {
        let model = make_model();

        let p1 = Array1::from_vec(vec![0.2, 0.0, 0.0, 0.0]);
        let p2 = Array1::from_vec(vec![-0.2, 0.0, 0.0, 0.0]);

        let mid = model.midpoint(&[p1.view(), p2.view()]).unwrap();

        // Midpoint of symmetric points should be near origin
        let mid_norm = mid.dot(&mid).sqrt();
        assert!(
            mid_norm < 0.05,
            "Midpoint of symmetric points should be near origin, norm={}",
            mid_norm
        );
    }

    #[test]
    fn test_midpoint_empty() {
        let model = make_model();
        let result = model.midpoint(&[]);
        assert!(result.is_err(), "Midpoint of empty set should fail");
    }

    #[test]
    fn test_knn_search_basic() {
        let mut knn = PoincareKNN::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        knn.insert(id1, Array1::from_vec(vec![0.1, 0.0]));
        knn.insert(id2, Array1::from_vec(vec![0.5, 0.0]));
        knn.insert(id3, Array1::from_vec(vec![0.9, 0.0]));

        assert_eq!(knn.len(), 3);

        // Search near the first point
        let query = Array1::from_vec(vec![0.12, 0.0]);
        let results = knn.search(&query.view(), 2);

        assert_eq!(results.len(), 2);
        // Closest should be id1
        assert_eq!(results[0].0, id1, "Nearest neighbor should be id1");
    }

    #[test]
    fn test_knn_search_empty() {
        let knn = PoincareKNN::new();
        let query = Array1::from_vec(vec![0.1, 0.0]);
        let results = knn.search(&query.view(), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_knn_search_k_greater_than_n() {
        let mut knn = PoincareKNN::new();
        knn.insert(Uuid::new_v4(), Array1::from_vec(vec![0.1, 0.0]));
        knn.insert(Uuid::new_v4(), Array1::from_vec(vec![0.5, 0.0]));

        let query = Array1::from_vec(vec![0.3, 0.0]);
        let results = knn.search(&query.view(), 10);

        assert_eq!(results.len(), 2, "Should return all points when k > n");
    }

    #[test]
    fn test_knn_search_ancestors() {
        let mut knn = PoincareKNN::new();

        let id_root = Uuid::new_v4();
        let id_mid = Uuid::new_v4();
        let id_leaf = Uuid::new_v4();

        // Root (norm ~0.1), Mid (norm ~0.5), Leaf (norm ~0.9)
        knn.insert(id_root, Array1::from_vec(vec![0.1, 0.0]));
        knn.insert(id_mid, Array1::from_vec(vec![0.5, 0.0]));
        knn.insert(id_leaf, Array1::from_vec(vec![0.9, 0.0]));

        // Query from near-leaf: ancestors should be root and mid (norm < 0.85)
        let query = Array1::from_vec(vec![0.85, 0.0]);
        let ancestors = knn.search_ancestors(&query.view(), 5);

        assert_eq!(ancestors.len(), 2, "Should find root (0.1) and mid (0.5) as ancestors");
        // Check that they are sorted by distance
        for i in 1..ancestors.len() {
            assert!(ancestors[i - 1].1 <= ancestors[i].1, "Should be sorted by distance");
        }
    }

    #[test]
    fn test_knn_search_descendants() {
        let mut knn = PoincareKNN::new();

        let id_root = Uuid::new_v4();
        let id_mid = Uuid::new_v4();
        let id_leaf = Uuid::new_v4();

        knn.insert(id_root, Array1::from_vec(vec![0.1, 0.0]));
        knn.insert(id_mid, Array1::from_vec(vec![0.5, 0.0]));
        knn.insert(id_leaf, Array1::from_vec(vec![0.9, 0.0]));

        // Query from root: descendants should be mid and leaf
        let query = Array1::from_vec(vec![0.05, 0.0]);
        let descendants = knn.search_descendants(&query.view(), 5);

        assert_eq!(descendants.len(), 3); // All three are further from origin
    }

    #[test]
    fn test_knn_remove() {
        let mut knn = PoincareKNN::new();

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        knn.insert(id1, Array1::from_vec(vec![0.1, 0.0]));
        knn.insert(id2, Array1::from_vec(vec![0.5, 0.0]));

        assert_eq!(knn.len(), 2);
        assert!(knn.remove(&id1));
        assert_eq!(knn.len(), 1);
        assert!(!knn.remove(&id1)); // Already removed
    }

    #[test]
    fn test_knn_clear() {
        let mut knn = PoincareKNN::new();
        knn.insert(Uuid::new_v4(), Array1::from_vec(vec![0.1, 0.0]));
        knn.insert(Uuid::new_v4(), Array1::from_vec(vec![0.5, 0.0]));

        assert_eq!(knn.len(), 2);
        knn.clear();
        assert!(knn.is_empty());
    }

    #[test]
    fn test_knn_results_sorted_by_distance() {
        let mut knn = PoincareKNN::new();

        // Insert points at various locations
        for i in 0..20 {
            let x = (i as f64) * 0.04;
            knn.insert(Uuid::new_v4(), Array1::from_vec(vec![x, 0.01 * (i as f64)]));
        }

        let query = Array1::from_vec(vec![0.3, 0.1]);
        let results = knn.search(&query.view(), 10);

        // Verify results are sorted by distance
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1 + 1e-12,
                "Results should be sorted: {} > {}",
                results[i - 1].1,
                results[i].1
            );
        }
    }

    #[test]
    fn test_poincare_ball_to_config() {
        let ball = PoincareBall::new(64, -2.0);
        let config = ball.to_config();

        assert_eq!(config.dimension, 64);
        assert_eq!(config.curvature, -2.0);
        assert_eq!(config.max_norm, ball.max_norm);
    }

    #[test]
    fn test_model_distance() {
        let model = make_model();

        let u = Array1::from_vec(vec![0.1, 0.2, 0.0, 0.0]);
        let v = Array1::from_vec(vec![0.3, -0.1, 0.0, 0.0]);

        let dist = model.distance(&u.view(), &v.view());
        assert!(dist > 0.0, "Distance should be positive");
        assert!(dist.is_finite(), "Distance should be finite");
    }
}
