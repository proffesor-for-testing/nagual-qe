//! System metrics collection and storage.
//!
//! Provides a `SystemMetric` struct for recording metrics with tags,
//! a `MetricsCollector` for aggregating metrics, and SQLite storage
//! with automatic 30-day retention.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn, instrument};

use crate::error::{DatabaseError, Result};

/// A single metric measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetric {
    /// Unique identifier for this metric record.
    pub id: Option<i64>,
    /// Metric name (e.g., "request_latency", "memory_usage").
    pub name: String,
    /// Metric value (always stored as f64 for uniformity).
    pub value: f64,
    /// Optional tags for categorization and filtering.
    pub tags: HashMap<String, String>,
    /// Timestamp when the metric was recorded.
    pub timestamp: DateTime<Utc>,
}

impl SystemMetric {
    /// Create a new metric with the current timestamp.
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            id: None,
            name: name.into(),
            value,
            tags: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    /// Create a metric with a specific timestamp.
    pub fn with_timestamp(name: impl Into<String>, value: f64, timestamp: DateTime<Utc>) -> Self {
        Self {
            id: None,
            name: name.into(),
            value,
            tags: HashMap::new(),
            timestamp,
        }
    }

    /// Add a tag to the metric.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Add multiple tags to the metric.
    pub fn with_tags(mut self, tags: HashMap<String, String>) -> Self {
        self.tags.extend(tags);
        self
    }

    /// Convert tags to JSON string for storage.
    pub fn tags_json(&self) -> String {
        serde_json::to_string(&self.tags).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse tags from JSON string.
    pub fn parse_tags_json(json: &str) -> HashMap<String, String> {
        serde_json::from_str(json).unwrap_or_default()
    }
}

/// Metric type for different kinds of measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    /// A counter that only increases.
    Counter,
    /// A gauge that can go up or down.
    Gauge,
    /// A histogram for distribution analysis.
    Histogram,
    /// A timer for latency measurements.
    Timer,
}

impl MetricType {
    /// Get the string representation for storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Timer => "timer",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "counter" => Some(MetricType::Counter),
            "gauge" => Some(MetricType::Gauge),
            "histogram" => Some(MetricType::Histogram),
            "timer" => Some(MetricType::Timer),
            _ => None,
        }
    }
}

/// Configuration for the metrics collector.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Retention period for metrics (default: 30 days).
    pub retention_days: u32,
    /// Maximum metrics to keep in memory before flushing.
    pub buffer_size: usize,
    /// Flush interval in seconds.
    pub flush_interval_secs: u64,
    /// Enable automatic cleanup of old metrics.
    pub auto_cleanup: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            retention_days: 30,
            buffer_size: 1000,
            flush_interval_secs: 60,
            auto_cleanup: true,
        }
    }
}

/// Metrics collector with SQLite persistence.
pub struct MetricsCollector {
    /// In-memory buffer for metrics before flushing.
    buffer: RwLock<Vec<SystemMetric>>,
    /// Configuration.
    config: MetricsConfig,
    /// Database connection for persistence.
    conn: Option<Arc<RwLock<Connection>>>,
    /// In-memory counters for fast access.
    counters: RwLock<HashMap<String, f64>>,
    /// In-memory gauges for fast access.
    gauges: RwLock<HashMap<String, f64>>,
}

impl MetricsCollector {
    /// Create a new metrics collector without persistence.
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            buffer: RwLock::new(Vec::with_capacity(config.buffer_size)),
            config,
            conn: None,
            counters: RwLock::new(HashMap::new()),
            gauges: RwLock::new(HashMap::new()),
        }
    }

    /// Create a metrics collector with SQLite persistence.
    #[instrument(skip(config))]
    pub fn with_sqlite(conn: Connection, config: MetricsConfig) -> Result<Self> {
        // Initialize the schema
        init_metrics_schema(&conn)?;

        let collector = Self {
            buffer: RwLock::new(Vec::with_capacity(config.buffer_size)),
            config,
            conn: Some(Arc::new(RwLock::new(conn))),
            counters: RwLock::new(HashMap::new()),
            gauges: RwLock::new(HashMap::new()),
        };

        info!("Initialized metrics collector with SQLite persistence");
        Ok(collector)
    }

    /// Record a metric.
    #[instrument(skip(self, metric), fields(metric_name = %metric.name))]
    pub fn record(&self, metric: SystemMetric) {
        debug!(name = %metric.name, value = metric.value, "Recording metric");

        let mut buffer = self.buffer.write();
        buffer.push(metric);

        // Flush if buffer is full
        if buffer.len() >= self.config.buffer_size {
            drop(buffer);
            if let Err(e) = self.flush() {
                warn!(error = %e, "Failed to flush metrics buffer");
            }
        }
    }

    /// Record a metric with name and value (convenience method).
    pub fn record_metric(&self, name: impl Into<String>, value: f64) {
        self.record(SystemMetric::new(name, value));
    }

    /// Record a metric with tags.
    pub fn record_metric_with_tags(
        &self,
        name: impl Into<String>,
        value: f64,
        tags: HashMap<String, String>,
    ) {
        self.record(SystemMetric::new(name, value).with_tags(tags));
    }

    /// Increment a counter by 1.
    pub fn increment(&self, name: &str) {
        self.increment_by(name, 1.0);
    }

    /// Increment a counter by a specific amount.
    pub fn increment_by(&self, name: &str, amount: f64) {
        let mut counters = self.counters.write();
        let value = counters.entry(name.to_string()).or_insert(0.0);
        *value += amount;

        // Also record as a metric
        self.record(
            SystemMetric::new(name, *value)
                .with_tag("type", "counter"),
        );
    }

    /// Set a gauge value.
    pub fn gauge(&self, name: &str, value: f64) {
        {
            let mut gauges = self.gauges.write();
            gauges.insert(name.to_string(), value);
        }

        self.record(
            SystemMetric::new(name, value)
                .with_tag("type", "gauge"),
        );
    }

    /// Record a timer measurement (in milliseconds).
    pub fn timer(&self, name: &str, duration_ms: f64) {
        self.record(
            SystemMetric::new(name, duration_ms)
                .with_tag("type", "timer")
                .with_tag("unit", "ms"),
        );
    }

    /// Record a timer measurement from a Duration.
    pub fn timer_duration(&self, name: &str, duration: Duration) {
        self.timer(name, duration.as_secs_f64() * 1000.0);
    }

    /// Get the current value of a counter.
    pub fn get_counter(&self, name: &str) -> f64 {
        self.counters.read().get(name).copied().unwrap_or(0.0)
    }

    /// Get the current value of a gauge.
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.read().get(name).copied()
    }

    /// Flush buffered metrics to storage.
    #[instrument(skip(self))]
    pub fn flush(&self) -> Result<usize> {
        let metrics: Vec<SystemMetric> = {
            let mut buffer = self.buffer.write();
            std::mem::take(&mut *buffer)
        };

        if metrics.is_empty() {
            return Ok(0);
        }

        let count = metrics.len();

        if let Some(ref conn) = self.conn {
            let conn = conn.write();
            let tx = conn.unchecked_transaction().map_err(DatabaseError::from)?;

            for metric in &metrics {
                tx.execute(
                    "INSERT INTO system_metrics (name, value, tags_json, timestamp) VALUES (?1, ?2, ?3, ?4)",
                    params![
                        metric.name,
                        metric.value,
                        metric.tags_json(),
                        metric.timestamp.to_rfc3339(),
                    ],
                ).map_err(DatabaseError::from)?;
            }

            tx.commit().map_err(DatabaseError::from)?;
            debug!(count = count, "Flushed metrics to SQLite");
        }

        Ok(count)
    }

    /// Clean up old metrics based on retention policy.
    #[instrument(skip(self))]
    pub fn cleanup_old_metrics(&self) -> Result<usize> {
        if let Some(ref conn) = self.conn {
            let conn = conn.write();
            let cutoff = Utc::now() - chrono::Duration::days(self.config.retention_days as i64);
            let cutoff_str = cutoff.to_rfc3339();

            let deleted = conn
                .execute(
                    "DELETE FROM system_metrics WHERE timestamp < ?1",
                    params![cutoff_str],
                )
                .map_err(DatabaseError::from)?;

            if deleted > 0 {
                info!(
                    deleted = deleted,
                    retention_days = self.config.retention_days,
                    "Cleaned up old metrics"
                );
            }

            Ok(deleted)
        } else {
            Ok(0)
        }
    }

    /// Query metrics by name within a time range.
    #[instrument(skip(self))]
    pub fn query_metrics(
        &self,
        name: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<Vec<SystemMetric>> {
        if let Some(ref conn) = self.conn {
            let conn = conn.read();
            let mut stmt = conn
                .prepare(
                    "SELECT id, name, value, tags_json, timestamp
                     FROM system_metrics
                     WHERE name = ?1 AND timestamp >= ?2 AND timestamp <= ?3
                     ORDER BY timestamp ASC",
                )
                .map_err(DatabaseError::from)?;

            let metrics = stmt
                .query_map(
                    params![name, from.to_rfc3339(), to.to_rfc3339()],
                    |row| {
                        let tags_json: String = row.get(3)?;
                        let timestamp_str: String = row.get(4)?;
                        Ok(SystemMetric {
                            id: Some(row.get(0)?),
                            name: row.get(1)?,
                            value: row.get(2)?,
                            tags: SystemMetric::parse_tags_json(&tags_json),
                            timestamp: DateTime::parse_from_rfc3339(&timestamp_str)
                                .map(|dt| dt.with_timezone(&Utc))
                                .unwrap_or_else(|_| Utc::now()),
                        })
                    },
                )
                .map_err(DatabaseError::from)?;

            let mut results = Vec::new();
            for metric in metrics {
                results.push(metric.map_err(DatabaseError::from)?);
            }

            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get aggregated statistics for a metric.
    #[instrument(skip(self))]
    pub fn get_metric_stats(
        &self,
        name: &str,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Result<MetricStats> {
        if let Some(ref conn) = self.conn {
            let conn = conn.read();
            let mut stmt = conn
                .prepare(
                    "SELECT
                        COUNT(*) as count,
                        MIN(value) as min_val,
                        MAX(value) as max_val,
                        AVG(value) as avg_val,
                        SUM(value) as sum_val
                     FROM system_metrics
                     WHERE name = ?1 AND timestamp >= ?2 AND timestamp <= ?3",
                )
                .map_err(DatabaseError::from)?;

            let stats = stmt
                .query_row(params![name, from.to_rfc3339(), to.to_rfc3339()], |row| {
                    Ok(MetricStats {
                        name: name.to_string(),
                        count: row.get(0)?,
                        min: row.get::<_, Option<f64>>(1)?.unwrap_or(0.0),
                        max: row.get::<_, Option<f64>>(2)?.unwrap_or(0.0),
                        avg: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
                        sum: row.get::<_, Option<f64>>(4)?.unwrap_or(0.0),
                        from,
                        to,
                    })
                })
                .map_err(DatabaseError::from)?;

            Ok(stats)
        } else {
            Ok(MetricStats {
                name: name.to_string(),
                count: 0,
                min: 0.0,
                max: 0.0,
                avg: 0.0,
                sum: 0.0,
                from,
                to,
            })
        }
    }

    /// List all unique metric names.
    pub fn list_metric_names(&self) -> Result<Vec<String>> {
        if let Some(ref conn) = self.conn {
            let conn = conn.read();
            let mut stmt = conn
                .prepare("SELECT DISTINCT name FROM system_metrics ORDER BY name")
                .map_err(DatabaseError::from)?;

            let names = stmt
                .query_map([], |row| row.get(0))
                .map_err(DatabaseError::from)?;

            let mut results = Vec::new();
            for name in names {
                results.push(name.map_err(DatabaseError::from)?);
            }

            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get total metrics count.
    pub fn total_count(&self) -> Result<usize> {
        if let Some(ref conn) = self.conn {
            let conn = conn.read();
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM system_metrics", [], |row| row.get(0))
                .map_err(DatabaseError::from)?;
            Ok(count as usize)
        } else {
            Ok(self.buffer.read().len())
        }
    }
}

/// Aggregated statistics for a metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    /// Metric name.
    pub name: String,
    /// Number of data points.
    pub count: i64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Average value.
    pub avg: f64,
    /// Sum of all values.
    pub sum: f64,
    /// Start of time range.
    pub from: DateTime<Utc>,
    /// End of time range.
    pub to: DateTime<Utc>,
}

impl MetricStats {
    /// Calculate the range (max - min).
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Check if there's any data.
    pub fn has_data(&self) -> bool {
        self.count > 0
    }
}

/// Initialize the metrics schema.
fn init_metrics_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            tags_json TEXT DEFAULT '{}',
            timestamp TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(name);
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON system_metrics(name, timestamp);",
    )
    .map_err(DatabaseError::from)?;

    Ok(())
}

/// Helper macro for recording metrics with automatic timing.
#[macro_export]
macro_rules! timed_metric {
    ($collector:expr, $name:expr, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        $collector.timer_duration($name, start.elapsed());
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_collector() -> MetricsCollector {
        MetricsCollector::new(MetricsConfig::default())
    }

    fn test_collector_with_db() -> (MetricsCollector, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("metrics.db");
        let conn = Connection::open(&db_path).unwrap();
        let collector = MetricsCollector::with_sqlite(conn, MetricsConfig::default()).unwrap();
        (collector, temp_dir)
    }

    #[test]
    fn test_metric_creation() {
        let metric = SystemMetric::new("test_metric", 42.0)
            .with_tag("env", "test")
            .with_tag("host", "localhost");

        assert_eq!(metric.name, "test_metric");
        assert_eq!(metric.value, 42.0);
        assert_eq!(metric.tags.get("env"), Some(&"test".to_string()));
        assert_eq!(metric.tags.get("host"), Some(&"localhost".to_string()));
    }

    #[test]
    fn test_metric_tags_json() {
        let metric = SystemMetric::new("test", 1.0)
            .with_tag("key", "value");

        let json = metric.tags_json();
        assert!(json.contains("key"));
        assert!(json.contains("value"));

        let parsed = SystemMetric::parse_tags_json(&json);
        assert_eq!(parsed.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_counter_operations() {
        let collector = test_collector();

        collector.increment("requests");
        assert_eq!(collector.get_counter("requests"), 1.0);

        collector.increment_by("requests", 5.0);
        assert_eq!(collector.get_counter("requests"), 6.0);
    }

    #[test]
    fn test_gauge_operations() {
        let collector = test_collector();

        collector.gauge("memory_usage", 1024.0);
        assert_eq!(collector.get_gauge("memory_usage"), Some(1024.0));

        collector.gauge("memory_usage", 2048.0);
        assert_eq!(collector.get_gauge("memory_usage"), Some(2048.0));

        assert_eq!(collector.get_gauge("nonexistent"), None);
    }

    #[test]
    fn test_timer() {
        let collector = test_collector();

        collector.timer("request_latency", 150.5);
        collector.timer_duration("operation", Duration::from_millis(250));

        // Just verify no panics - actual values are in the buffer
    }

    #[test]
    fn test_flush_with_db() {
        let (collector, _temp) = test_collector_with_db();

        collector.record_metric("test1", 1.0);
        collector.record_metric("test2", 2.0);
        collector.record_metric("test3", 3.0);

        let flushed = collector.flush().unwrap();
        assert_eq!(flushed, 3);

        let count = collector.total_count().unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_query_metrics() {
        let (collector, _temp) = test_collector_with_db();

        let now = Utc::now();
        collector.record(SystemMetric::with_timestamp("query_test", 1.0, now));
        collector.record(SystemMetric::with_timestamp("query_test", 2.0, now));
        collector.record(SystemMetric::with_timestamp("other", 3.0, now));

        collector.flush().unwrap();

        let from = now - chrono::Duration::hours(1);
        let to = now + chrono::Duration::hours(1);

        let metrics = collector.query_metrics("query_test", from, to).unwrap();
        assert_eq!(metrics.len(), 2);
    }

    #[test]
    fn test_metric_stats() {
        let (collector, _temp) = test_collector_with_db();

        let now = Utc::now();
        collector.record(SystemMetric::with_timestamp("stats_test", 10.0, now));
        collector.record(SystemMetric::with_timestamp("stats_test", 20.0, now));
        collector.record(SystemMetric::with_timestamp("stats_test", 30.0, now));

        collector.flush().unwrap();

        let from = now - chrono::Duration::hours(1);
        let to = now + chrono::Duration::hours(1);

        let stats = collector.get_metric_stats("stats_test", from, to).unwrap();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 30.0);
        assert_eq!(stats.sum, 60.0);
        assert_eq!(stats.avg, 20.0);
    }

    #[test]
    fn test_list_metric_names() {
        let (collector, _temp) = test_collector_with_db();

        collector.record_metric("alpha", 1.0);
        collector.record_metric("beta", 2.0);
        collector.record_metric("gamma", 3.0);
        collector.record_metric("alpha", 4.0); // Duplicate name

        collector.flush().unwrap();

        let names = collector.list_metric_names().unwrap();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"alpha".to_string()));
        assert!(names.contains(&"beta".to_string()));
        assert!(names.contains(&"gamma".to_string()));
    }

    #[test]
    fn test_cleanup_old_metrics() {
        let (collector, _temp) = test_collector_with_db();

        // Record a metric with an old timestamp
        let old_time = Utc::now() - chrono::Duration::days(60);
        collector.record(SystemMetric::with_timestamp("old_metric", 1.0, old_time));

        // Record a recent metric
        collector.record(SystemMetric::new("recent_metric", 2.0));

        collector.flush().unwrap();

        let deleted = collector.cleanup_old_metrics().unwrap();
        assert_eq!(deleted, 1);

        let count = collector.total_count().unwrap();
        assert_eq!(count, 1);
    }
}
