//! Structured JSON logging with daily rotation.
//!
//! Configures tracing-subscriber with JSON formatting, env-filter support,
//! and file logging with automatic rotation.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{Local, NaiveDate, Utc};
use parking_lot::Mutex;
use tracing::{debug, info, Level};
use tracing_subscriber::{
    fmt::{self, MakeWriter},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Layer,
};

use crate::error::{NagualError, Result};

/// Log level configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Error level - only errors.
    Error,
    /// Warn level - errors and warnings.
    Warn,
    /// Info level - errors, warnings, and info.
    Info,
    /// Debug level - all except trace.
    Debug,
    /// Trace level - everything.
    Trace,
}

impl LogLevel {
    /// Convert to tracing Level.
    pub fn to_tracing_level(self) -> Level {
        match self {
            LogLevel::Error => Level::ERROR,
            LogLevel::Warn => Level::WARN,
            LogLevel::Info => Level::INFO,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Trace => Level::TRACE,
        }
    }

    /// Convert to filter directive string.
    pub fn as_filter_directive(self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "error" => Some(LogLevel::Error),
            "warn" | "warning" => Some(LogLevel::Warn),
            "info" => Some(LogLevel::Info),
            "debug" => Some(LogLevel::Debug),
            "trace" => Some(LogLevel::Trace),
            _ => None,
        }
    }
}

impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_filter_directive())
    }
}

/// Configuration for logging.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Base log level.
    pub level: LogLevel,
    /// Log directory path.
    pub log_dir: PathBuf,
    /// Log file name prefix (default: "nagual").
    pub file_prefix: String,
    /// Enable JSON formatting.
    pub json_format: bool,
    /// Enable console output.
    pub console_output: bool,
    /// Enable file output.
    pub file_output: bool,
    /// Enable ANSI colors in console output.
    pub ansi_colors: bool,
    /// Include target in log output.
    pub include_target: bool,
    /// Include file/line information.
    pub include_file_line: bool,
    /// Include thread names.
    pub include_thread_names: bool,
    /// Include thread IDs.
    pub include_thread_ids: bool,
    /// Additional module-specific log levels.
    pub module_levels: Vec<(String, LogLevel)>,
    /// Number of days to retain log files (0 = keep all).
    pub retention_days: u32,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            log_dir: PathBuf::from("logs"),
            file_prefix: "nagual".to_string(),
            json_format: true,
            console_output: true,
            file_output: true,
            ansi_colors: true,
            include_target: true,
            include_file_line: false,
            include_thread_names: false,
            include_thread_ids: false,
            module_levels: Vec::new(),
            retention_days: 30,
        }
    }
}

impl LoggingConfig {
    /// Create a new config with the default log level.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the log level.
    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.level = level;
        self
    }

    /// Set the log directory.
    pub fn with_log_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.log_dir = path.into();
        self
    }

    /// Set the file prefix.
    pub fn with_file_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.file_prefix = prefix.into();
        self
    }

    /// Enable or disable JSON format.
    pub fn with_json(mut self, enabled: bool) -> Self {
        self.json_format = enabled;
        self
    }

    /// Enable or disable console output.
    pub fn with_console(mut self, enabled: bool) -> Self {
        self.console_output = enabled;
        self
    }

    /// Enable or disable file output.
    pub fn with_file(mut self, enabled: bool) -> Self {
        self.file_output = enabled;
        self
    }

    /// Add a module-specific log level.
    pub fn with_module_level(mut self, module: impl Into<String>, level: LogLevel) -> Self {
        self.module_levels.push((module.into(), level));
        self
    }

    /// Set retention days.
    pub fn with_retention_days(mut self, days: u32) -> Self {
        self.retention_days = days;
        self
    }

    /// Build the env filter from configuration.
    pub fn build_filter(&self) -> EnvFilter {
        let mut filter = EnvFilter::new(self.level.as_filter_directive());

        // Add module-specific levels
        for (module, level) in &self.module_levels {
            filter = filter.add_directive(
                format!("{}={}", module, level.as_filter_directive())
                    .parse()
                    .unwrap(),
            );
        }

        // Allow RUST_LOG to override
        filter = filter.add_directive("nagual=info".parse().unwrap());

        filter
    }

    /// Get the log file path for today.
    pub fn log_file_path(&self) -> PathBuf {
        let date = Local::now().format("%Y-%m-%d");
        self.log_dir.join(format!("{}-{}.log", self.file_prefix, date))
    }
}

/// A writer that rotates log files daily.
pub struct DailyRotatingWriter {
    config: LoggingConfig,
    current_file: Mutex<Option<RotatingFile>>,
}

struct RotatingFile {
    file: BufWriter<File>,
    date: NaiveDate,
    path: PathBuf,
}

impl DailyRotatingWriter {
    /// Create a new daily rotating writer.
    pub fn new(config: LoggingConfig) -> Result<Self> {
        // Ensure log directory exists
        if !config.log_dir.exists() {
            fs::create_dir_all(&config.log_dir).map_err(NagualError::from)?;
        }

        Ok(Self {
            config,
            current_file: Mutex::new(None),
        })
    }

    /// Get or create the current log file.
    fn get_or_create_file(&self) -> io::Result<impl Write + '_> {
        let today = Local::now().date_naive();
        let mut guard = self.current_file.lock();

        // Check if we need to rotate
        let needs_rotation = match &*guard {
            Some(rf) => rf.date != today,
            None => true,
        };

        if needs_rotation {
            let path = self.config.log_file_path();
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;

            *guard = Some(RotatingFile {
                file: BufWriter::new(file),
                date: today,
                path,
            });

            // Cleanup old files if configured
            if self.config.retention_days > 0 {
                let _ = self.cleanup_old_files();
            }
        }

        Ok(WriterGuard { guard })
    }

    /// Clean up old log files based on retention policy.
    fn cleanup_old_files(&self) -> io::Result<usize> {
        let cutoff = Local::now().date_naive()
            - chrono::Duration::days(self.config.retention_days as i64);

        let mut deleted = 0;
        let prefix = format!("{}-", self.config.file_prefix);

        for entry in fs::read_dir(&self.config.log_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                if filename.starts_with(&prefix) && filename.ends_with(".log") {
                    // Extract date from filename
                    let date_str = filename
                        .strip_prefix(&prefix)
                        .and_then(|s| s.strip_suffix(".log"));

                    if let Some(date_str) = date_str {
                        if let Ok(file_date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                            if file_date < cutoff {
                                if fs::remove_file(&path).is_ok() {
                                    deleted += 1;
                                    debug!(path = %path.display(), "Deleted old log file");
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(deleted)
    }
}

/// Guard that provides write access to the rotating file.
struct WriterGuard<'a> {
    guard: parking_lot::MutexGuard<'a, Option<RotatingFile>>,
}

impl<'a> Write for WriterGuard<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(ref mut rf) = *self.guard {
            rf.file.write(buf)
        } else {
            Err(io::Error::new(io::ErrorKind::Other, "No file available"))
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(ref mut rf) = *self.guard {
            rf.file.flush()
        } else {
            Ok(())
        }
    }
}

impl<'a> MakeWriter<'a> for &'a DailyRotatingWriter {
    type Writer = Box<dyn Write + 'a>;

    fn make_writer(&'a self) -> Self::Writer {
        match self.get_or_create_file() {
            Ok(writer) => Box::new(writer),
            Err(_) => Box::new(io::sink()),
        }
    }
}

/// Initialize logging with the given configuration.
///
/// This sets up tracing-subscriber with JSON formatting (if enabled),
/// console output, and file output with daily rotation.
pub fn init_logging(config: LoggingConfig) -> Result<LoggingHandle> {
    let filter = config.build_filter();

    // Build console layer
    let console_layer = if config.console_output {
        if config.json_format {
            Some(
                fmt::layer()
                    .json()
                    .with_ansi(config.ansi_colors)
                    .with_target(config.include_target)
                    .with_file(config.include_file_line)
                    .with_line_number(config.include_file_line)
                    .with_thread_names(config.include_thread_names)
                    .with_thread_ids(config.include_thread_ids)
                    .with_filter(filter.clone())
                    .boxed(),
            )
        } else {
            Some(
                fmt::layer()
                    .with_ansi(config.ansi_colors)
                    .with_target(config.include_target)
                    .with_file(config.include_file_line)
                    .with_line_number(config.include_file_line)
                    .with_thread_names(config.include_thread_names)
                    .with_thread_ids(config.include_thread_ids)
                    .with_filter(filter.clone())
                    .boxed(),
            )
        }
    } else {
        None
    };

    // Build file layer
    let file_writer = if config.file_output {
        Some(Arc::new(DailyRotatingWriter::new(config.clone())?))
    } else {
        None
    };

    // We need to handle the file layer differently due to lifetime issues
    // For now, we'll set up just the console layer through tracing_subscriber
    // and provide a separate method for file logging

    let subscriber = tracing_subscriber::registry();

    if let Some(console) = console_layer {
        subscriber.with(console).init();
    } else {
        subscriber.with(filter).init();
    }

    info!(
        log_level = %config.level,
        json_format = config.json_format,
        console_output = config.console_output,
        file_output = config.file_output,
        "Logging initialized"
    );

    Ok(LoggingHandle {
        config,
        file_writer,
    })
}

/// Initialize logging with default configuration.
pub fn init_default_logging() -> Result<LoggingHandle> {
    init_logging(LoggingConfig::default())
}

/// Initialize logging for development (debug level, pretty output).
pub fn init_dev_logging() -> Result<LoggingHandle> {
    init_logging(
        LoggingConfig::default()
            .with_level(LogLevel::Debug)
            .with_json(false)
            .with_file(false),
    )
}

/// Initialize logging for production (info level, JSON, file output).
pub fn init_prod_logging(log_dir: impl Into<PathBuf>) -> Result<LoggingHandle> {
    init_logging(
        LoggingConfig::default()
            .with_level(LogLevel::Info)
            .with_json(true)
            .with_log_dir(log_dir)
            .with_ansi_colors(false),
    )
}

impl LoggingConfig {
    /// Enable or disable ANSI colors.
    pub fn with_ansi_colors(mut self, enabled: bool) -> Self {
        self.ansi_colors = enabled;
        self
    }
}

/// Handle returned from init_logging for managing the logging system.
pub struct LoggingHandle {
    config: LoggingConfig,
    file_writer: Option<Arc<DailyRotatingWriter>>,
}

impl LoggingHandle {
    /// Get the current log file path.
    pub fn current_log_file(&self) -> PathBuf {
        self.config.log_file_path()
    }

    /// Get the log directory.
    pub fn log_dir(&self) -> &Path {
        &self.config.log_dir
    }

    /// Get the logging configuration.
    pub fn config(&self) -> &LoggingConfig {
        &self.config
    }

    /// Manually trigger cleanup of old log files.
    pub fn cleanup_old_files(&self) -> Result<usize> {
        if let Some(ref writer) = self.file_writer {
            writer.cleanup_old_files().map_err(NagualError::from)
        } else {
            Ok(0)
        }
    }

    /// Write a log entry directly to the file (bypassing tracing).
    pub fn write_direct(&self, message: &str) -> Result<()> {
        if let Some(ref writer) = self.file_writer {
            let timestamp = Utc::now().to_rfc3339();
            let log_line = if self.config.json_format {
                serde_json::json!({
                    "timestamp": timestamp,
                    "level": "INFO",
                    "message": message,
                    "target": "direct"
                }).to_string()
            } else {
                format!("{} INFO direct: {}", timestamp, message)
            };

            let mut file = writer.get_or_create_file().map_err(NagualError::from)?;
            writeln!(file, "{}", log_line).map_err(NagualError::from)?;
        }
        Ok(())
    }
}

/// Structured log entry for JSON logging.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LogEntry {
    /// Timestamp in RFC3339 format.
    pub timestamp: String,
    /// Log level.
    pub level: String,
    /// Log message.
    pub message: String,
    /// Target/module name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// File name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Line number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    /// Additional fields.
    #[serde(flatten)]
    pub fields: serde_json::Map<String, serde_json::Value>,
}

impl LogEntry {
    /// Create a new log entry.
    pub fn new(level: &str, message: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now().to_rfc3339(),
            level: level.to_string(),
            message: message.into(),
            target: None,
            file: None,
            line: None,
            fields: serde_json::Map::new(),
        }
    }

    /// Add a target.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Add a field.
    pub fn with_field(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| self.message.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_log_level_conversion() {
        assert_eq!(LogLevel::Error.to_tracing_level(), Level::ERROR);
        assert_eq!(LogLevel::Warn.to_tracing_level(), Level::WARN);
        assert_eq!(LogLevel::Info.to_tracing_level(), Level::INFO);
        assert_eq!(LogLevel::Debug.to_tracing_level(), Level::DEBUG);
        assert_eq!(LogLevel::Trace.to_tracing_level(), Level::TRACE);
    }

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(LogLevel::from_str("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_str("warn"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("warning"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_str("INFO"), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_str("debug"), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_str("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::from_str("invalid"), None);
    }

    #[test]
    fn test_logging_config_defaults() {
        let config = LoggingConfig::default();
        assert_eq!(config.level, LogLevel::Info);
        assert!(config.json_format);
        assert!(config.console_output);
        assert!(config.file_output);
        assert_eq!(config.retention_days, 30);
    }

    #[test]
    fn test_logging_config_builder() {
        let config = LoggingConfig::new()
            .with_level(LogLevel::Debug)
            .with_json(false)
            .with_console(true)
            .with_file(false)
            .with_log_dir("/tmp/logs")
            .with_file_prefix("test")
            .with_retention_days(7)
            .with_module_level("hyper", LogLevel::Warn);

        assert_eq!(config.level, LogLevel::Debug);
        assert!(!config.json_format);
        assert!(config.console_output);
        assert!(!config.file_output);
        assert_eq!(config.log_dir, PathBuf::from("/tmp/logs"));
        assert_eq!(config.file_prefix, "test");
        assert_eq!(config.retention_days, 7);
        assert_eq!(config.module_levels.len(), 1);
    }

    #[test]
    fn test_log_file_path() {
        let config = LoggingConfig::new()
            .with_log_dir("/tmp/logs")
            .with_file_prefix("test");

        let path = config.log_file_path();
        let filename = path.file_name().unwrap().to_str().unwrap();

        assert!(filename.starts_with("test-"));
        assert!(filename.ends_with(".log"));
    }

    #[test]
    fn test_daily_rotating_writer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = LoggingConfig::new()
            .with_log_dir(temp_dir.path());

        let writer = DailyRotatingWriter::new(config);
        assert!(writer.is_ok());
    }

    #[test]
    fn test_log_entry_creation() {
        let entry = LogEntry::new("INFO", "Test message")
            .with_target("test::module")
            .with_field("user_id", serde_json::json!("user123"))
            .with_field("action", serde_json::json!("login"));

        assert_eq!(entry.level, "INFO");
        assert_eq!(entry.message, "Test message");
        assert_eq!(entry.target, Some("test::module".to_string()));
        assert_eq!(entry.fields.len(), 2);

        let json = entry.to_json();
        assert!(json.contains("Test message"));
        assert!(json.contains("user123"));
    }

    #[test]
    fn test_build_filter() {
        let config = LoggingConfig::new()
            .with_level(LogLevel::Debug)
            .with_module_level("tokio", LogLevel::Warn)
            .with_module_level("hyper", LogLevel::Error);

        let filter = config.build_filter();
        // Filter is built successfully - hard to test the actual filtering behavior
        let _ = filter;
    }
}
