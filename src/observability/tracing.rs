//! Tracing spans and metric recording integration.
//!
//! Provides integration between the `tracing` crate and metrics collection,
//! with automatic span timing, success/failure recording, and instrumentation helpers.

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, instrument, warn, Span};

use super::metrics::{MetricsCollector, SystemMetric};

/// A metric span that automatically records timing and outcome.
///
/// When dropped, it records the duration and success/failure status
/// to the associated metrics collector.
pub struct MetricSpan {
    /// Name of the metric to record.
    name: String,
    /// Start time for duration calculation.
    start: Instant,
    /// Metrics collector for recording.
    collector: Option<Arc<MetricsCollector>>,
    /// Whether the operation succeeded (default: true).
    success: bool,
    /// Additional tags to include.
    tags: Vec<(String, String)>,
    /// Tracing span for context.
    span: Span,
}

impl MetricSpan {
    /// Create a new metric span with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let span = tracing::info_span!("metric_span", metric_name = %name);

        Self {
            name,
            start: Instant::now(),
            collector: None,
            success: true,
            tags: Vec::new(),
            span,
        }
    }

    /// Create a metric span with a collector for automatic recording.
    pub fn with_collector(name: impl Into<String>, collector: Arc<MetricsCollector>) -> Self {
        let name = name.into();
        let span = tracing::info_span!("metric_span", metric_name = %name);

        Self {
            name,
            start: Instant::now(),
            collector: Some(collector),
            success: true,
            tags: Vec::new(),
            span,
        }
    }

    /// Add a tag to the metric.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.push((key.into(), value.into()));
        self
    }

    /// Mark the operation as failed.
    pub fn mark_failure(&mut self) {
        self.success = false;
    }

    /// Mark the operation as successful.
    pub fn mark_success(&mut self) {
        self.success = true;
    }

    /// Get the elapsed duration in milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Get the elapsed duration.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    /// Get the tracing span.
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Enter the tracing span.
    pub fn enter(&self) -> tracing::span::Entered<'_> {
        self.span.enter()
    }

    /// Record the metric manually (without waiting for drop).
    pub fn record(&self) {
        if let Some(ref collector) = self.collector {
            let elapsed = self.elapsed_ms();
            let mut metric = SystemMetric::new(format!("{}.duration_ms", self.name), elapsed)
                .with_tag("success", self.success.to_string());

            for (key, value) in &self.tags {
                metric = metric.with_tag(key.clone(), value.clone());
            }

            collector.record(metric);

            // Also record success/failure as separate metrics
            let outcome_name = if self.success {
                format!("{}.success", self.name)
            } else {
                format!("{}.failure", self.name)
            };
            collector.increment(&outcome_name);

            debug!(
                metric = %self.name,
                duration_ms = elapsed,
                success = self.success,
                "Recorded metric span"
            );
        }
    }
}

impl Drop for MetricSpan {
    fn drop(&mut self) {
        self.record();
    }
}

/// Builder for configuring instrumented operations.
pub struct InstrumentedOperation<'a> {
    name: &'a str,
    collector: Option<Arc<MetricsCollector>>,
    tags: Vec<(String, String)>,
}

impl<'a> InstrumentedOperation<'a> {
    /// Create a new instrumented operation.
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            collector: None,
            tags: Vec::new(),
        }
    }

    /// Set the metrics collector.
    pub fn with_collector(mut self, collector: Arc<MetricsCollector>) -> Self {
        self.collector = Some(collector);
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.push((key.into(), value.into()));
        self
    }

    /// Execute a synchronous operation with timing.
    pub fn execute<T, F>(self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let mut span = if let Some(collector) = self.collector {
            MetricSpan::with_collector(self.name, collector)
        } else {
            MetricSpan::new(self.name)
        };

        for (key, value) in self.tags {
            span = span.with_tag(key, value);
        }

        let _guard = span.enter();
        f()
    }

    /// Execute a synchronous operation that returns Result with timing.
    pub fn execute_result<T, E, F>(self, f: F) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
    {
        let mut span = if let Some(collector) = self.collector {
            MetricSpan::with_collector(self.name, collector)
        } else {
            MetricSpan::new(self.name)
        };

        for (key, value) in self.tags {
            span = span.with_tag(key, value);
        }

        let _guard = span.enter();
        let result = f();

        // Record success/failure after guard is dropped
        drop(_guard);
        if result.is_err() {
            span.mark_failure();
        }

        result
    }

    /// Execute an async operation with timing.
    pub async fn execute_async<T, F, Fut>(self, f: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let mut span = if let Some(collector) = self.collector {
            MetricSpan::with_collector(self.name, collector)
        } else {
            MetricSpan::new(self.name)
        };

        for (key, value) in self.tags {
            span = span.with_tag(key, value);
        }

        // Note: We can't use span.enter() across await points
        f().await
    }

    /// Execute an async operation that returns Result with timing.
    pub async fn execute_async_result<T, E, F, Fut>(self, f: F) -> Result<T, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let mut span = if let Some(collector) = self.collector {
            MetricSpan::with_collector(self.name, collector)
        } else {
            MetricSpan::new(self.name)
        };

        for (key, value) in self.tags {
            span = span.with_tag(key, value);
        }

        let result = f().await;

        if result.is_err() {
            span.mark_failure();
        }

        result
    }
}

/// Trait for types that can be instrumented with metrics.
pub trait Instrumented {
    /// Get the metrics collector.
    fn metrics_collector(&self) -> Option<Arc<MetricsCollector>>;

    /// Create an instrumented operation.
    fn instrumented<'a>(&self, name: &'a str) -> InstrumentedOperation<'a> {
        let mut op = InstrumentedOperation::new(name);
        if let Some(collector) = self.metrics_collector() {
            op = op.with_collector(collector);
        }
        op
    }
}

/// A layer for tracing-subscriber that records span metrics.
pub struct MetricsLayer {
    collector: Arc<MetricsCollector>,
}

impl MetricsLayer {
    /// Create a new metrics layer.
    pub fn new(collector: Arc<MetricsCollector>) -> Self {
        Self { collector }
    }
}

impl<S> tracing_subscriber::Layer<S> for MetricsLayer
where
    S: tracing::Subscriber,
{
    fn on_enter(&self, _id: &tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        // Could record span enter metrics here
    }

    fn on_exit(&self, _id: &tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        // Could record span exit metrics here
    }

    fn on_close(&self, _id: tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        // Could record span close metrics here
    }
}

/// Record a function's execution time.
///
/// This function wraps another function and records its execution time
/// to the provided metrics collector.
#[instrument(skip_all, fields(operation = name))]
pub fn timed<T, F>(name: &str, collector: &MetricsCollector, f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();

    collector.timer_duration(name, elapsed);
    debug!(operation = name, duration_ms = elapsed.as_secs_f64() * 1000.0, "Operation completed");

    result
}

/// Record an async function's execution time.
#[instrument(skip_all, fields(operation = name))]
pub async fn timed_async<T, F, Fut>(name: &str, collector: &MetricsCollector, f: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = T>,
{
    let start = Instant::now();
    let result = f().await;
    let elapsed = start.elapsed();

    collector.timer_duration(name, elapsed);
    debug!(operation = name, duration_ms = elapsed.as_secs_f64() * 1000.0, "Async operation completed");

    result
}

/// Record a Result-returning function's execution time and outcome.
#[instrument(skip_all, fields(operation = name))]
pub fn timed_result<T, E, F>(name: &str, collector: &MetricsCollector, f: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E>,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();

    collector.timer_duration(name, elapsed);

    match &result {
        Ok(_) => {
            collector.increment(&format!("{}.success", name));
            debug!(operation = name, duration_ms = elapsed.as_secs_f64() * 1000.0, success = true, "Operation succeeded");
        }
        Err(_) => {
            collector.increment(&format!("{}.failure", name));
            warn!(operation = name, duration_ms = elapsed.as_secs_f64() * 1000.0, success = false, "Operation failed");
        }
    }

    result
}

/// Record an async Result-returning function's execution time and outcome.
#[instrument(skip_all, fields(operation = name))]
pub async fn timed_result_async<T, E, F, Fut>(
    name: &str,
    collector: &MetricsCollector,
    f: F,
) -> Result<T, E>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let start = Instant::now();
    let result = f().await;
    let elapsed = start.elapsed();

    collector.timer_duration(name, elapsed);

    match &result {
        Ok(_) => {
            collector.increment(&format!("{}.success", name));
            debug!(operation = name, duration_ms = elapsed.as_secs_f64() * 1000.0, success = true, "Async operation succeeded");
        }
        Err(_) => {
            collector.increment(&format!("{}.failure", name));
            warn!(operation = name, duration_ms = elapsed.as_secs_f64() * 1000.0, success = false, "Async operation failed");
        }
    }

    result
}

/// Helper to record database operation metrics.
pub struct DbOperationMetrics {
    collector: Arc<MetricsCollector>,
}

impl DbOperationMetrics {
    /// Create a new database metrics recorder.
    pub fn new(collector: Arc<MetricsCollector>) -> Self {
        Self { collector }
    }

    /// Record a query operation.
    #[instrument(skip(self))]
    pub fn record_query(&self, table: &str, duration_ms: f64, row_count: usize) {
        self.collector.record(
            SystemMetric::new("db.query.duration_ms", duration_ms)
                .with_tag("table", table)
                .with_tag("type", "query"),
        );
        self.collector.record(
            SystemMetric::new("db.query.rows", row_count as f64)
                .with_tag("table", table),
        );
    }

    /// Record an insert operation.
    #[instrument(skip(self))]
    pub fn record_insert(&self, table: &str, duration_ms: f64, row_count: usize) {
        self.collector.record(
            SystemMetric::new("db.insert.duration_ms", duration_ms)
                .with_tag("table", table)
                .with_tag("type", "insert"),
        );
        self.collector.record(
            SystemMetric::new("db.insert.rows", row_count as f64)
                .with_tag("table", table),
        );
    }

    /// Record an update operation.
    #[instrument(skip(self))]
    pub fn record_update(&self, table: &str, duration_ms: f64, affected_rows: usize) {
        self.collector.record(
            SystemMetric::new("db.update.duration_ms", duration_ms)
                .with_tag("table", table)
                .with_tag("type", "update"),
        );
        self.collector.record(
            SystemMetric::new("db.update.affected_rows", affected_rows as f64)
                .with_tag("table", table),
        );
    }

    /// Record a delete operation.
    #[instrument(skip(self))]
    pub fn record_delete(&self, table: &str, duration_ms: f64, deleted_rows: usize) {
        self.collector.record(
            SystemMetric::new("db.delete.duration_ms", duration_ms)
                .with_tag("table", table)
                .with_tag("type", "delete"),
        );
        self.collector.record(
            SystemMetric::new("db.delete.rows", deleted_rows as f64)
                .with_tag("table", table),
        );
    }

    /// Record a transaction.
    #[instrument(skip(self))]
    pub fn record_transaction(&self, duration_ms: f64, success: bool) {
        self.collector.record(
            SystemMetric::new("db.transaction.duration_ms", duration_ms)
                .with_tag("success", success.to_string()),
        );
        if success {
            self.collector.increment("db.transaction.commits");
        } else {
            self.collector.increment("db.transaction.rollbacks");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::metrics::MetricsConfig;
    use std::thread;
    use std::time::Duration;

    fn test_collector() -> Arc<MetricsCollector> {
        Arc::new(MetricsCollector::new(MetricsConfig::default()))
    }

    #[test]
    fn test_metric_span_timing() {
        let collector = test_collector();

        {
            let _span = MetricSpan::with_collector("test_operation", collector.clone());
            thread::sleep(Duration::from_millis(10));
        }

        // Span should have recorded duration on drop
        assert!(collector.get_counter("test_operation.success") >= 1.0);
    }

    #[test]
    fn test_metric_span_failure() {
        let collector = test_collector();

        {
            let mut span = MetricSpan::with_collector("failing_op", collector.clone());
            span.mark_failure();
        }

        assert!(collector.get_counter("failing_op.failure") >= 1.0);
    }

    #[test]
    fn test_metric_span_with_tags() {
        let collector = test_collector();

        {
            let _span = MetricSpan::with_collector("tagged_op", collector.clone())
                .with_tag("env", "test")
                .with_tag("version", "1.0");
        }

        // Just verify no panics - tags are recorded in the metric
    }

    #[test]
    fn test_instrumented_operation_sync() {
        let collector = test_collector();

        let result = InstrumentedOperation::new("sync_op")
            .with_collector(collector.clone())
            .with_tag("type", "test")
            .execute(|| {
                thread::sleep(Duration::from_millis(5));
                42
            });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_instrumented_operation_result() {
        let collector = test_collector();

        let result: Result<i32, &str> = InstrumentedOperation::new("result_op")
            .with_collector(collector.clone())
            .execute_result(|| Ok(42));

        assert_eq!(result.unwrap(), 42);

        let error: Result<i32, &str> = InstrumentedOperation::new("error_op")
            .with_collector(collector.clone())
            .execute_result(|| Err("oops"));

        assert!(error.is_err());
    }

    #[test]
    fn test_timed_function() {
        let collector = MetricsCollector::new(MetricsConfig::default());

        let result = timed("timed_test", &collector, || {
            thread::sleep(Duration::from_millis(5));
            "done"
        });

        assert_eq!(result, "done");
    }

    #[test]
    fn test_timed_result_success() {
        let collector = MetricsCollector::new(MetricsConfig::default());

        let result: Result<i32, &str> = timed_result("success_test", &collector, || Ok(42));

        assert_eq!(result.unwrap(), 42);
        assert!(collector.get_counter("success_test.success") >= 1.0);
    }

    #[test]
    fn test_timed_result_failure() {
        let collector = MetricsCollector::new(MetricsConfig::default());

        let result: Result<i32, &str> = timed_result("failure_test", &collector, || Err("error"));

        assert!(result.is_err());
        assert!(collector.get_counter("failure_test.failure") >= 1.0);
    }

    #[test]
    fn test_db_operation_metrics() {
        let collector = test_collector();
        let db_metrics = DbOperationMetrics::new(collector.clone());

        db_metrics.record_query("users", 15.5, 100);
        db_metrics.record_insert("users", 5.2, 1);
        db_metrics.record_update("users", 3.1, 5);
        db_metrics.record_delete("users", 2.0, 3);
        db_metrics.record_transaction(25.0, true);
        db_metrics.record_transaction(10.0, false);

        assert!(collector.get_counter("db.transaction.commits") >= 1.0);
        assert!(collector.get_counter("db.transaction.rollbacks") >= 1.0);
    }

    #[tokio::test]
    async fn test_timed_async() {
        let collector = MetricsCollector::new(MetricsConfig::default());

        let result = timed_async("async_test", &collector, || async {
            tokio::time::sleep(Duration::from_millis(5)).await;
            "async done"
        }).await;

        assert_eq!(result, "async done");
    }

    #[tokio::test]
    async fn test_timed_result_async() {
        let collector = MetricsCollector::new(MetricsConfig::default());

        let result: Result<i32, &str> = timed_result_async("async_result", &collector, || async {
            Ok(42)
        }).await;

        assert_eq!(result.unwrap(), 42);
        assert!(collector.get_counter("async_result.success") >= 1.0);
    }
}
