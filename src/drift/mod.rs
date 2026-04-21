//! Drift monitoring for pattern embedding spaces.
//!
//! Tracks per-domain embedding centroids over time and computes the
//! coefficient of variation (CV) of consecutive centroid distances
//! to detect drift or stagnation in knowledge domains.
//!
//! # How It Works
//!
//! 1. Embeddings are recorded per domain via [`DriftMonitor::record`].
//! 2. The monitor maintains a sliding window of embedding history.
//! 3. [`DriftMonitor::compute_drift`] computes pairwise L2 distances
//!    between consecutive embeddings, then calculates the CV
//!    (standard deviation / mean) of those distances.
//! 4. A high CV indicates drift (embeddings are changing erratically).
//!    A very low CV indicates stagnation (no meaningful change).
//!
//! # Example
//!
//! ```rust
//! use nagual::drift::{DriftMonitor, DriftTrend};
//!
//! let mut monitor = DriftMonitor::new();
//!
//! // Record embeddings over time
//! monitor.record("rust", vec![1.0, 0.0, 0.0]);
//! monitor.record("rust", vec![1.0, 0.1, 0.0]);
//! monitor.record("rust", vec![1.0, 0.2, 0.0]);
//!
//! if let Some(report) = monitor.compute_drift("rust") {
//!     println!("CV: {:.3}", report.coefficient_of_variation);
//!     println!("Trend: {}", report.trend);
//!     println!("Action: {}", report.suggested_action);
//! }
//! ```

use std::collections::HashMap;

/// Drift report for a single domain.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// The domain this report covers.
    pub domain: String,
    /// Coefficient of variation (std_dev / mean) of consecutive centroid distances.
    pub coefficient_of_variation: f64,
    /// Whether the domain is considered to be drifting (CV > threshold).
    pub is_drifting: bool,
    /// Detected trend direction based on recent vs overall distances.
    pub trend: DriftTrend,
    /// Human-readable suggestion for what to do.
    pub suggested_action: String,
    /// Number of embeddings in the current window.
    pub window_size: usize,
    /// Number of centroid snapshots used in computation.
    pub centroid_count: usize,
}

/// Direction of embedding drift over time.
#[derive(Debug, Clone, PartialEq)]
pub enum DriftTrend {
    /// Distances are consistent -- no significant change.
    Stable,
    /// Recent distances are larger than historical average.
    Increasing,
    /// Recent distances are smaller than historical average.
    Decreasing,
    /// Not enough data to determine trend.
    Insufficient,
}

impl std::fmt::Display for DriftTrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "stable"),
            Self::Increasing => write!(f, "increasing"),
            Self::Decreasing => write!(f, "decreasing"),
            Self::Insufficient => write!(f, "insufficient data"),
        }
    }
}

/// Monitors embedding drift per domain.
///
/// Records embeddings over time and computes statistical measures
/// to detect when a domain's embedding distribution is shifting
/// (drift) or has stopped changing (stagnation).
pub struct DriftMonitor {
    /// Per-domain embedding history.
    centroids: HashMap<String, Vec<Vec<f32>>>,
    /// Maximum number of embeddings to retain per domain.
    max_window: usize,
    /// CV threshold above which drift is flagged.
    cv_threshold: f64,
}

impl DriftMonitor {
    /// Create a new drift monitor with default settings.
    ///
    /// Defaults: max_window = 50, cv_threshold = 0.5.
    pub fn new() -> Self {
        Self {
            centroids: HashMap::new(),
            max_window: 50,
            cv_threshold: 0.5,
        }
    }

    /// Create a drift monitor with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `max_window` - Maximum number of embeddings to retain per domain.
    /// * `cv_threshold` - CV value above which drift is detected.
    pub fn with_config(max_window: usize, cv_threshold: f64) -> Self {
        Self {
            centroids: HashMap::new(),
            max_window: max_window.max(2),
            cv_threshold: cv_threshold.max(0.0),
        }
    }

    /// Record a new embedding for a domain.
    ///
    /// The embedding is appended to the domain's history. If the history
    /// exceeds `2 * max_window`, the oldest entries are pruned.
    pub fn record(&mut self, domain: &str, embedding: Vec<f32>) {
        let history = self.centroids.entry(domain.to_string()).or_default();
        history.push(embedding);
        // Prune if history grows too large (keep max_window entries)
        if history.len() > self.max_window * 2 {
            history.drain(..self.max_window);
        }
    }

    /// Compute a drift report for a specific domain.
    ///
    /// Returns `None` if the domain has fewer than 2 recorded embeddings
    /// (need at least 2 to compute a distance).
    pub fn compute_drift(&self, domain: &str) -> Option<DriftReport> {
        let history = self.centroids.get(domain)?;
        if history.len() < 2 {
            return Some(DriftReport {
                domain: domain.to_string(),
                coefficient_of_variation: 0.0,
                is_drifting: false,
                trend: DriftTrend::Insufficient,
                suggested_action: "Collect more data before drift analysis is meaningful"
                    .to_string(),
                window_size: history.len(),
                centroid_count: history.len(),
            });
        }

        // Compute L2 distances between consecutive embeddings
        let distances: Vec<f64> = history
            .windows(2)
            .map(|pair| l2_distance(&pair[0], &pair[1]))
            .collect();

        let n = distances.len() as f64;
        let mean_dist: f64 = distances.iter().sum::<f64>() / n;

        let variance: f64 = distances
            .iter()
            .map(|d| (d - mean_dist).powi(2))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        let cv = if mean_dist > 1e-10 {
            std_dev / mean_dist
        } else {
            0.0
        };

        let is_drifting = cv > self.cv_threshold;

        // Determine trend by comparing recent distances to overall mean
        let trend = self.compute_trend(&distances, mean_dist);

        let suggested_action = if is_drifting {
            "Investigating potential embedding drift -- check for data quality issues".to_string()
        } else if cv < 0.1 && mean_dist < 1e-6 {
            "Domain appears stagnant -- consider encouraging new contributions".to_string()
        } else {
            "Normal operation".to_string()
        };

        Some(DriftReport {
            domain: domain.to_string(),
            coefficient_of_variation: cv,
            is_drifting,
            trend,
            suggested_action,
            window_size: history.len(),
            centroid_count: history.len(),
        })
    }

    /// Get drift reports for all tracked domains.
    pub fn all_reports(&self) -> Vec<DriftReport> {
        self.centroids
            .keys()
            .filter_map(|domain| self.compute_drift(domain))
            .collect()
    }

    /// Return the list of tracked domains.
    pub fn domains(&self) -> Vec<&str> {
        self.centroids.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of recorded embeddings for a domain.
    pub fn count(&self, domain: &str) -> usize {
        self.centroids
            .get(domain)
            .map(|h| h.len())
            .unwrap_or(0)
    }

    /// Determine the trend direction from recent vs historical distances.
    fn compute_trend(&self, distances: &[f64], mean_dist: f64) -> DriftTrend {
        if distances.len() < 5 {
            return DriftTrend::Insufficient;
        }

        let recent_count = 5.min(distances.len());
        let recent_avg: f64 =
            distances[distances.len() - recent_count..].iter().sum::<f64>()
                / recent_count as f64;

        if recent_avg > mean_dist * 1.3 {
            DriftTrend::Increasing
        } else if recent_avg < mean_dist * 0.7 {
            DriftTrend::Decreasing
        } else {
            DriftTrend::Stable
        }
    }
}

impl Default for DriftMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the L2 (Euclidean) distance between two vectors.
///
/// If the vectors have different lengths, only the overlapping
/// prefix is compared.
fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    let len = a.len().min(b.len());
    let sum: f64 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(&x, &y)| {
            let diff = (x - y) as f64;
            diff * diff
        })
        .sum();
    sum.sqrt()
}

// ---------------------------------------------------------------------------
// SQLite persistence for DriftReport
// ---------------------------------------------------------------------------

const DRIFT_TABLE_DDL: &str =
    "CREATE TABLE IF NOT EXISTS drift_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT NOT NULL,
        coefficient_of_variation REAL NOT NULL,
        is_drifting INTEGER NOT NULL,
        trend TEXT NOT NULL,
        suggested_action TEXT NOT NULL,
        window_size INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );";

/// Persist a drift report to SQLite.
pub fn persist_drift_report(db_path: &str, report: &DriftReport) -> std::result::Result<(), String> {
    use rusqlite::Connection;
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch(DRIFT_TABLE_DDL).map_err(|e| e.to_string())?;
    conn.execute(
        "INSERT INTO drift_log (domain, coefficient_of_variation, is_drifting, trend, suggested_action, window_size)
         VALUES (?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            report.domain,
            report.coefficient_of_variation,
            report.is_drifting as i32,
            report.trend.to_string(),
            report.suggested_action,
            report.window_size as i32,
        ],
    )
    .map_err(|e| e.to_string())?;
    // Keep only last 500 entries
    conn.execute_batch(
        "DELETE FROM drift_log WHERE id NOT IN (SELECT id FROM drift_log ORDER BY id DESC LIMIT 500);"
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Load drift reports from SQLite (most recent first).
pub fn load_drift_reports(db_path: &str) -> std::result::Result<Vec<DriftReport>, String> {
    use rusqlite::Connection;
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch(DRIFT_TABLE_DDL).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT domain, coefficient_of_variation, is_drifting, trend, suggested_action, window_size
             FROM drift_log ORDER BY created_at DESC",
        )
        .map_err(|e| e.to_string())?;
    let reports = stmt
        .query_map([], |row| {
            let trend_str: String = row.get(3)?;
            let trend = match trend_str.as_str() {
                "increasing" => DriftTrend::Increasing,
                "decreasing" => DriftTrend::Decreasing,
                "stable" => DriftTrend::Stable,
                _ => DriftTrend::Insufficient,
            };
            Ok(DriftReport {
                domain: row.get(0)?,
                coefficient_of_variation: row.get(1)?,
                is_drifting: row.get::<_, i32>(2)? != 0,
                trend,
                suggested_action: row.get(4)?,
                window_size: row.get::<_, i32>(5)? as usize,
                centroid_count: 0,
            })
        })
        .map_err(|e| e.to_string())?;
    reports
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

/// Load drift reports for a specific domain from SQLite.
pub fn load_drift_reports_for_domain(
    db_path: &str,
    domain: &str,
) -> std::result::Result<Vec<DriftReport>, String> {
    use rusqlite::Connection;
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.execute_batch(DRIFT_TABLE_DDL).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT domain, coefficient_of_variation, is_drifting, trend, suggested_action, window_size
             FROM drift_log WHERE domain = ? ORDER BY created_at DESC LIMIT 1",
        )
        .map_err(|e| e.to_string())?;
    let reports = stmt
        .query_map(rusqlite::params![domain], |row| {
            let trend_str: String = row.get(3)?;
            let trend = match trend_str.as_str() {
                "increasing" => DriftTrend::Increasing,
                "decreasing" => DriftTrend::Decreasing,
                "stable" => DriftTrend::Stable,
                _ => DriftTrend::Insufficient,
            };
            Ok(DriftReport {
                domain: row.get(0)?,
                coefficient_of_variation: row.get(1)?,
                is_drifting: row.get::<_, i32>(2)? != 0,
                trend,
                suggested_action: row.get(4)?,
                window_size: row.get::<_, i32>(5)? as usize,
                centroid_count: 0,
            })
        })
        .map_err(|e| e.to_string())?;
    reports
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_updates_history() {
        let mut monitor = DriftMonitor::new();
        assert_eq!(monitor.count("test"), 0);

        monitor.record("test", vec![1.0, 0.0, 0.0]);
        assert_eq!(monitor.count("test"), 1);

        monitor.record("test", vec![0.0, 1.0, 0.0]);
        assert_eq!(monitor.count("test"), 2);
    }

    #[test]
    fn test_drift_detection_high_cv() {
        let mut monitor = DriftMonitor::with_config(100, 0.3);

        // Record embeddings with wildly varying distances (high CV)
        let embeddings = vec![
            vec![0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],  // distance = 10
            vec![10.1, 0.0, 0.0],  // distance = 0.1
            vec![20.0, 0.0, 0.0],  // distance = 9.9
            vec![20.05, 0.0, 0.0], // distance = 0.05
            vec![30.0, 0.0, 0.0],  // distance = 9.95
            vec![30.02, 0.0, 0.0], // distance = 0.02
        ];

        for emb in embeddings {
            monitor.record("unstable", emb);
        }

        let report = monitor.compute_drift("unstable").unwrap();
        assert!(
            report.is_drifting,
            "High CV ({:.3}) should indicate drift",
            report.coefficient_of_variation
        );
        assert!(
            report.coefficient_of_variation > 0.3,
            "CV should be > 0.3, got {:.3}",
            report.coefficient_of_variation
        );
    }

    #[test]
    fn test_stagnation_detection_low_cv() {
        let mut monitor = DriftMonitor::new();

        // Record nearly identical embeddings (very low CV, near-zero distances)
        for _ in 0..10 {
            monitor.record("stagnant", vec![1.0, 0.0, 0.0]);
        }

        let report = monitor.compute_drift("stagnant").unwrap();
        assert!(
            !report.is_drifting,
            "Identical embeddings should not be flagged as drifting"
        );
        assert!(
            report.coefficient_of_variation < 0.1,
            "CV for identical embeddings should be near 0, got {:.3}",
            report.coefficient_of_variation
        );
        assert!(
            report.suggested_action.contains("stagnant"),
            "Should suggest stagnation action, got: {}",
            report.suggested_action
        );
    }

    #[test]
    fn test_insufficient_data() {
        let mut monitor = DriftMonitor::new();
        monitor.record("sparse", vec![1.0, 0.0]);

        let report = monitor.compute_drift("sparse").unwrap();
        assert_eq!(report.trend, DriftTrend::Insufficient);
        assert!(!report.is_drifting);
    }

    #[test]
    fn test_nonexistent_domain() {
        let monitor = DriftMonitor::new();
        assert!(monitor.compute_drift("nope").is_none());
    }

    #[test]
    fn test_multiple_domains_independent() {
        let mut monitor = DriftMonitor::new();

        // Domain A: stable
        for _ in 0..5 {
            monitor.record("stable", vec![1.0, 0.0]);
        }

        // Domain B: drifting (use with_config for lower threshold)
        let mut monitor2 = DriftMonitor::with_config(100, 0.2);
        for _ in 0..5 {
            monitor2.record("stable", vec![1.0, 0.0]);
        }
        let drifting_embeddings = vec![
            vec![0.0, 0.0],
            vec![5.0, 0.0],
            vec![5.01, 0.0],
            vec![10.0, 0.0],
            vec![10.01, 0.0],
            vec![15.0, 0.0],
        ];
        for emb in drifting_embeddings {
            monitor2.record("drifting", emb);
        }

        let stable_report = monitor2.compute_drift("stable").unwrap();
        let drift_report = monitor2.compute_drift("drifting").unwrap();

        assert!(!stable_report.is_drifting);
        assert!(
            drift_report.is_drifting,
            "Drifting domain should be flagged, CV = {:.3}",
            drift_report.coefficient_of_variation
        );
    }

    #[test]
    fn test_all_reports() {
        let mut monitor = DriftMonitor::new();

        monitor.record("domain_a", vec![1.0, 0.0]);
        monitor.record("domain_a", vec![1.1, 0.0]);
        monitor.record("domain_b", vec![0.0, 1.0]);
        monitor.record("domain_b", vec![0.0, 1.1]);

        let reports = monitor.all_reports();
        assert_eq!(reports.len(), 2);

        let domains: Vec<&str> = reports.iter().map(|r| r.domain.as_str()).collect();
        assert!(domains.contains(&"domain_a"));
        assert!(domains.contains(&"domain_b"));
    }

    #[test]
    fn test_window_pruning() {
        let mut monitor = DriftMonitor::with_config(5, 0.5);

        // Record more than 2*max_window embeddings
        for i in 0..15 {
            monitor.record("pruned", vec![i as f32, 0.0]);
        }

        // After pruning, should have at most max_window entries
        assert!(
            monitor.count("pruned") <= 10,
            "History should be pruned, got {} entries",
            monitor.count("pruned")
        );
    }

    #[test]
    fn test_trend_increasing() {
        let mut monitor = DriftMonitor::with_config(100, 2.0); // high threshold so not flagged as drifting

        // Start with small consistent distances, then increase
        let embeddings = vec![
            vec![0.0],
            vec![0.1],  // dist 0.1
            vec![0.2],  // dist 0.1
            vec![0.3],  // dist 0.1
            vec![0.4],  // dist 0.1
            vec![0.5],  // dist 0.1
            vec![0.6],  // dist 0.1
            // Now jump to large distances
            vec![5.6],  // dist 5.0
            vec![10.6], // dist 5.0
            vec![15.6], // dist 5.0
            vec![20.6], // dist 5.0
            vec![25.6], // dist 5.0
        ];

        for emb in embeddings {
            monitor.record("trend", emb);
        }

        let report = monitor.compute_drift("trend").unwrap();
        assert_eq!(
            report.trend,
            DriftTrend::Increasing,
            "Recent large distances should indicate increasing trend"
        );
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![3.0f32, 0.0];
        let b = vec![0.0f32, 4.0];
        let dist = l2_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6, "3-4-5 triangle, got {dist}");
    }

    #[test]
    fn test_drift_trend_display() {
        assert_eq!(format!("{}", DriftTrend::Stable), "stable");
        assert_eq!(format!("{}", DriftTrend::Increasing), "increasing");
        assert_eq!(format!("{}", DriftTrend::Decreasing), "decreasing");
        assert_eq!(format!("{}", DriftTrend::Insufficient), "insufficient data");
    }

    #[test]
    fn test_default_trait() {
        let monitor = DriftMonitor::default();
        assert_eq!(monitor.max_window, 50);
        assert!((monitor.cv_threshold - 0.5).abs() < f64::EPSILON);
    }

    // ----- SQLite persistence tests -----

    #[test]
    fn test_persist_and_load_drift_report() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_drift.db");
        let db_str = db_path.to_str().unwrap();

        let report = DriftReport {
            domain: "rust.async".to_string(),
            coefficient_of_variation: 0.42,
            is_drifting: false,
            trend: DriftTrend::Stable,
            suggested_action: "Normal operation".to_string(),
            window_size: 10,
            centroid_count: 10,
        };
        persist_drift_report(db_str, &report).unwrap();

        let loaded = load_drift_reports(db_str).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].domain, "rust.async");
        assert!((loaded[0].coefficient_of_variation - 0.42).abs() < 0.001);
        assert_eq!(loaded[0].trend, DriftTrend::Stable);
    }

    #[test]
    fn test_load_drift_reports_empty() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_empty_drift.db");
        let db_str = db_path.to_str().unwrap();

        let loaded = load_drift_reports(db_str).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_load_drift_reports_for_domain() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_domain_drift.db");
        let db_str = db_path.to_str().unwrap();

        let report_a = DriftReport {
            domain: "rust".to_string(),
            coefficient_of_variation: 0.3,
            is_drifting: false,
            trend: DriftTrend::Stable,
            suggested_action: "OK".to_string(),
            window_size: 5,
            centroid_count: 5,
        };
        let report_b = DriftReport {
            domain: "python".to_string(),
            coefficient_of_variation: 0.8,
            is_drifting: true,
            trend: DriftTrend::Increasing,
            suggested_action: "Investigate".to_string(),
            window_size: 8,
            centroid_count: 8,
        };
        persist_drift_report(db_str, &report_a).unwrap();
        persist_drift_report(db_str, &report_b).unwrap();

        let rust_reports = load_drift_reports_for_domain(db_str, "rust").unwrap();
        assert_eq!(rust_reports.len(), 1);
        assert_eq!(rust_reports[0].domain, "rust");

        let missing = load_drift_reports_for_domain(db_str, "go").unwrap();
        assert!(missing.is_empty());
    }
}
