//! Monthly restore drill automation.
//!
//! Provides automated restore drills to verify backup integrity and
//! practice recovery procedures. Drills are scheduled monthly (first Sunday)
//! and perform a test restore with data integrity verification.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Datelike, Duration, Utc, Weekday};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use super::backup::{BackupConfig, BackupManager, BackupMetadata, BackupType};
use super::restore::{RestoreConfig, RestoreManager, RestoreResult};
use crate::error::{NagualError, Result};

/// Configuration for restore drills.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreDrillConfig {
    /// Directory to perform test restores
    pub drill_directory: PathBuf,
    /// Path to the production database
    pub production_db_path: PathBuf,
    /// Path to backups
    pub backup_dir: PathBuf,
    /// Whether to perform full integrity check
    pub verify_integrity: bool,
    /// Whether to compare record counts
    pub compare_record_counts: bool,
    /// Maximum age of backup to use for drill (hours)
    pub max_backup_age_hours: u64,
    /// Whether to clean up drill files after completion
    pub cleanup_after_drill: bool,
    /// Notification email for drill reports (optional)
    pub notification_email: Option<String>,
}

impl Default for RestoreDrillConfig {
    fn default() -> Self {
        Self {
            drill_directory: PathBuf::from("./drills"),
            production_db_path: PathBuf::from("./nagual.db"),
            backup_dir: PathBuf::from("./backups"),
            verify_integrity: true,
            compare_record_counts: true,
            max_backup_age_hours: 24,
            cleanup_after_drill: true,
            notification_email: None,
        }
    }
}

impl RestoreDrillConfig {
    /// Create a new drill configuration.
    pub fn new(
        production_db: impl Into<PathBuf>,
        backup_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            production_db_path: production_db.into(),
            backup_dir: backup_dir.into(),
            ..Default::default()
        }
    }

    /// Set the drill directory.
    pub fn with_drill_directory(mut self, dir: impl Into<PathBuf>) -> Self {
        self.drill_directory = dir.into();
        self
    }

    /// Set whether to verify integrity.
    pub fn with_integrity_check(mut self, verify: bool) -> Self {
        self.verify_integrity = verify;
        self
    }

    /// Set whether to compare record counts.
    pub fn with_record_count_comparison(mut self, compare: bool) -> Self {
        self.compare_record_counts = compare;
        self
    }

    /// Set notification email.
    pub fn with_notification_email(mut self, email: impl Into<String>) -> Self {
        self.notification_email = Some(email.into());
        self
    }
}

/// Result of a restore drill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrillResult {
    /// Drill completed successfully
    Success,
    /// Drill completed with warnings
    Warning,
    /// Drill failed
    Failed,
}

impl std::fmt::Display for DrillResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DrillResult::Success => write!(f, "success"),
            DrillResult::Warning => write!(f, "warning"),
            DrillResult::Failed => write!(f, "failed"),
        }
    }
}

/// Detailed report of a restore drill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrillReport {
    /// Unique drill identifier
    pub drill_id: String,
    /// When the drill was started
    pub started_at: DateTime<Utc>,
    /// When the drill completed
    pub completed_at: DateTime<Utc>,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Overall result
    pub result: DrillResult,
    /// Backup used for the drill
    pub backup_used: BackupMetadata,
    /// Restore result
    pub restore_result: Option<RestoreResult>,
    /// Whether integrity check passed
    pub integrity_check_passed: bool,
    /// Production database record count
    pub production_record_count: u64,
    /// Restored database record count
    pub restored_record_count: u64,
    /// Record count difference
    pub record_count_difference: i64,
    /// List of issues found
    pub issues: Vec<DrillIssue>,
    /// List of warnings
    pub warnings: Vec<String>,
    /// Path to drill database (if not cleaned up)
    pub drill_db_path: Option<String>,
}

impl DrillReport {
    /// Create a new drill report.
    fn new(backup: BackupMetadata) -> Self {
        Self {
            drill_id: uuid::Uuid::new_v4().to_string(),
            started_at: Utc::now(),
            completed_at: Utc::now(),
            duration_ms: 0,
            result: DrillResult::Success,
            backup_used: backup,
            restore_result: None,
            integrity_check_passed: true,
            production_record_count: 0,
            restored_record_count: 0,
            record_count_difference: 0,
            issues: Vec::new(),
            warnings: Vec::new(),
            drill_db_path: None,
        }
    }

    /// Add an issue to the report.
    fn add_issue(&mut self, issue: DrillIssue) {
        let severity = issue.severity;
        self.issues.push(issue);
        if matches!(severity, IssueSeverity::Critical | IssueSeverity::High) {
            self.result = DrillResult::Failed;
        } else if self.result != DrillResult::Failed {
            self.result = DrillResult::Warning;
        }
    }

    /// Add a warning to the report.
    fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
        if self.result == DrillResult::Success {
            self.result = DrillResult::Warning;
        }
    }

    /// Finalize the report.
    fn finalize(&mut self, start_time: std::time::Instant) {
        self.completed_at = Utc::now();
        self.duration_ms = start_time.elapsed().as_millis() as u64;
    }
}

/// An issue found during the drill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrillIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Issue message
    pub message: String,
    /// Additional details
    pub details: Option<String>,
}

/// Severity of a drill issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueSeverity {
    /// Informational only
    Info,
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical - drill failed
    Critical,
}

/// Category of a drill issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueCategory {
    /// Backup-related issue
    Backup,
    /// Restore-related issue
    Restore,
    /// Integrity-related issue
    Integrity,
    /// Data-related issue
    Data,
    /// Configuration issue
    Configuration,
}

impl std::fmt::Display for IssueCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueCategory::Backup => write!(f, "backup"),
            IssueCategory::Restore => write!(f, "restore"),
            IssueCategory::Integrity => write!(f, "integrity"),
            IssueCategory::Data => write!(f, "data"),
            IssueCategory::Configuration => write!(f, "configuration"),
        }
    }
}

/// Restore drill manager.
pub struct RestoreDrill {
    config: RestoreDrillConfig,
}

impl RestoreDrill {
    /// Create a new restore drill manager.
    pub fn new(config: RestoreDrillConfig) -> Result<Self> {
        // Ensure drill directory exists
        if !config.drill_directory.exists() {
            fs::create_dir_all(&config.drill_directory).map_err(NagualError::from)?;
        }

        Ok(Self { config })
    }

    /// Get the configuration.
    pub fn config(&self) -> &RestoreDrillConfig {
        &self.config
    }

    /// Run a restore drill.
    pub async fn run_drill(&self) -> Result<DrillReport> {
        let start_time = std::time::Instant::now();

        info!("Starting restore drill");

        // Find the latest backup to use
        let backup = self.find_suitable_backup()?;
        let mut report = DrillReport::new(backup.clone());

        debug!(backup_id = %backup.id, "Using backup for drill");

        // Create drill database path
        let drill_db_path = self.config.drill_directory.join(format!(
            "drill-{}.db",
            Utc::now().format("%Y%m%d-%H%M%S")
        ));

        // Perform the restore
        match self.perform_restore(&backup, &drill_db_path).await {
            Ok(restore_result) => {
                report.restore_result = Some(restore_result.clone());

                // Check for restore warnings
                for warning in &restore_result.warnings {
                    report.add_warning(warning.clone());
                }
            }
            Err(e) => {
                report.add_issue(DrillIssue {
                    severity: IssueSeverity::Critical,
                    category: IssueCategory::Restore,
                    message: "Failed to restore backup".to_string(),
                    details: Some(e.to_string()),
                });
                report.finalize(start_time);
                return Ok(report);
            }
        }

        // Verify integrity
        if self.config.verify_integrity {
            match self.verify_integrity(&drill_db_path) {
                Ok(true) => {
                    report.integrity_check_passed = true;
                    debug!("Integrity check passed");
                }
                Ok(false) => {
                    report.integrity_check_passed = false;
                    report.add_issue(DrillIssue {
                        severity: IssueSeverity::Critical,
                        category: IssueCategory::Integrity,
                        message: "Integrity check failed".to_string(),
                        details: None,
                    });
                }
                Err(e) => {
                    report.integrity_check_passed = false;
                    report.add_issue(DrillIssue {
                        severity: IssueSeverity::High,
                        category: IssueCategory::Integrity,
                        message: "Failed to run integrity check".to_string(),
                        details: Some(e.to_string()),
                    });
                }
            }
        }

        // Compare record counts
        if self.config.compare_record_counts && self.config.production_db_path.exists() {
            match self.compare_record_counts(&drill_db_path).await {
                Ok((prod_count, drill_count)) => {
                    report.production_record_count = prod_count;
                    report.restored_record_count = drill_count;
                    report.record_count_difference = drill_count as i64 - prod_count as i64;

                    if report.record_count_difference.abs() > 0 {
                        let diff_percent = (report.record_count_difference.abs() as f64
                            / prod_count.max(1) as f64)
                            * 100.0;

                        if diff_percent > 5.0 {
                            report.add_issue(DrillIssue {
                                severity: IssueSeverity::High,
                                category: IssueCategory::Data,
                                message: format!(
                                    "Record count difference exceeds 5%: {} vs {} ({:.1}%)",
                                    drill_count, prod_count, diff_percent
                                ),
                                details: None,
                            });
                        } else if diff_percent > 1.0 {
                            report.add_warning(format!(
                                "Record count difference: {} vs {} ({:.1}%)",
                                drill_count, prod_count, diff_percent
                            ));
                        }
                    }
                }
                Err(e) => {
                    report.add_warning(format!("Failed to compare record counts: {}", e));
                }
            }
        }

        // Cleanup if configured
        if self.config.cleanup_after_drill {
            if let Err(e) = fs::remove_file(&drill_db_path) {
                report.add_warning(format!("Failed to cleanup drill database: {}", e));
            }
            // Also cleanup WAL and SHM files
            let _ = fs::remove_file(drill_db_path.with_extension("db-wal"));
            let _ = fs::remove_file(drill_db_path.with_extension("db-shm"));
        } else {
            report.drill_db_path = Some(drill_db_path.to_string_lossy().to_string());
        }

        report.finalize(start_time);

        // Save report
        self.save_report(&report)?;

        info!(
            drill_id = %report.drill_id,
            result = %report.result,
            duration_ms = report.duration_ms,
            "Restore drill completed"
        );

        Ok(report)
    }

    /// Check if it's time for a monthly drill (first Sunday of the month).
    pub fn is_drill_time(&self) -> bool {
        let now = Utc::now();
        let day = now.day();
        let weekday = now.weekday();

        // First Sunday is between day 1-7 and is a Sunday
        day <= 7 && weekday == Weekday::Sun
    }

    /// Get the next scheduled drill date.
    pub fn next_drill_date(&self) -> DateTime<Utc> {
        use chrono::NaiveDate;

        let now = Utc::now();
        let today = now.date_naive();

        // Helper to find first Sunday of a given month/year
        let first_sunday_of_month = |year: i32, month: u32| -> NaiveDate {
            let first_day = NaiveDate::from_ymd_opt(year, month, 1)
                .unwrap_or_else(|| NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
            let weekday = first_day.weekday();
            let days_to_sunday = match weekday {
                Weekday::Sun => 0,
                Weekday::Mon => 6,
                Weekday::Tue => 5,
                Weekday::Wed => 4,
                Weekday::Thu => 3,
                Weekday::Fri => 2,
                Weekday::Sat => 1,
            };
            first_day + Duration::days(days_to_sunday)
        };

        // Start with current month
        let mut year = today.year();
        let mut month = today.month();

        // Find next first Sunday that's in the future
        loop {
            let first_sunday = first_sunday_of_month(year, month);

            if first_sunday > today {
                // Found a future first Sunday
                return first_sunday
                    .and_hms_opt(3, 0, 0)
                    .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                    .unwrap_or(now);
            }

            // Move to next month
            if month == 12 {
                year += 1;
                month = 1;
            } else {
                month += 1;
            }

            // Safety: don't loop forever (max 12 months ahead)
            if month == today.month() && year == today.year() + 1 {
                break;
            }
        }

        // Fallback: return first Sunday of next month
        let (next_year, next_month) = if today.month() == 12 {
            (today.year() + 1, 1)
        } else {
            (today.year(), today.month() + 1)
        };

        first_sunday_of_month(next_year, next_month)
            .and_hms_opt(3, 0, 0)
            .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
            .unwrap_or(now)
    }

    /// List previous drill reports.
    pub fn list_reports(&self) -> Result<Vec<DrillReport>> {
        let reports_dir = self.config.drill_directory.join("reports");
        if !reports_dir.exists() {
            return Ok(Vec::new());
        }

        let mut reports = Vec::new();

        for entry in fs::read_dir(&reports_dir).map_err(NagualError::from)? {
            let entry = entry.map_err(NagualError::from)?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                let content = fs::read_to_string(&path).map_err(NagualError::from)?;
                if let Ok(report) = serde_json::from_str::<DrillReport>(&content) {
                    reports.push(report);
                }
            }
        }

        // Sort by date, newest first
        reports.sort_by(|a, b| b.started_at.cmp(&a.started_at));

        Ok(reports)
    }

    // Private helper methods

    fn find_suitable_backup(&self) -> Result<BackupMetadata> {
        let backup_config = BackupConfig::new(&self.config.production_db_path, &self.config.backup_dir);
        let backup_manager = BackupManager::new(backup_config)?;

        let backups = backup_manager.list_backups()?;
        let cutoff = Utc::now() - Duration::hours(self.config.max_backup_age_hours as i64);

        // Prefer full backups within the time window
        let suitable: Vec<_> = backups
            .iter()
            .filter(|b| b.created_at >= cutoff)
            .collect();

        if suitable.is_empty() {
            return Err(NagualError::config(format!(
                "No backups found within the last {} hours",
                self.config.max_backup_age_hours
            )));
        }

        // Prefer full backups
        if let Some(full) = suitable.iter().find(|b| b.backup_type == BackupType::Full) {
            return Ok((*full).clone());
        }

        Ok(suitable[0].clone())
    }

    async fn perform_restore(&self, backup: &BackupMetadata, target: &Path) -> Result<RestoreResult> {
        let restore_config = RestoreConfig::new(target, &self.config.backup_dir)
            .with_backup_before_restore(false);

        let restore_manager = RestoreManager::with_config(restore_config)?;
        restore_manager.restore_from_backup(&backup.path).await
    }

    fn verify_integrity(&self, db_path: &Path) -> Result<bool> {
        let conn = rusqlite::Connection::open(db_path).map_err(|e| {
            NagualError::config(format!("Failed to open drill database: {}", e))
        })?;

        let result: String = conn
            .query_row("PRAGMA integrity_check", [], |row| row.get(0))
            .map_err(|e| NagualError::config(format!("Integrity check failed: {}", e)))?;

        Ok(result == "ok")
    }

    async fn compare_record_counts(&self, drill_db: &Path) -> Result<(u64, u64)> {
        // Count records in production database
        let prod_count = self.count_records(&self.config.production_db_path)?;

        // Count records in drill database
        let drill_count = self.count_records(drill_db)?;

        Ok((prod_count, drill_count))
    }

    fn count_records(&self, db_path: &Path) -> Result<u64> {
        let conn = rusqlite::Connection::open(db_path).map_err(|e| {
            NagualError::config(format!("Failed to open database: {}", e))
        })?;

        // Get list of tables
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            .map_err(|e| NagualError::config(format!("Failed to list tables: {}", e)))?;

        let tables: Vec<String> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| NagualError::config(format!("Failed to query tables: {}", e)))?
            .filter_map(|r| r.ok())
            .collect();

        let mut total = 0u64;

        for table in tables {
            let count: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM \"{}\"", table), [], |row| {
                    row.get(0)
                })
                .unwrap_or(0);
            total += count as u64;
        }

        Ok(total)
    }

    fn save_report(&self, report: &DrillReport) -> Result<()> {
        let reports_dir = self.config.drill_directory.join("reports");
        if !reports_dir.exists() {
            fs::create_dir_all(&reports_dir).map_err(NagualError::from)?;
        }

        let path = reports_dir.join(format!("{}.json", report.drill_id));
        let json = serde_json::to_string_pretty(report)?;
        fs::write(&path, json).map_err(NagualError::from)?;

        debug!(report_id = %report.drill_id, "Saved drill report");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_db(dir: &TempDir, name: &str) -> PathBuf {
        let db_path = dir.path().join(name);
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);
             INSERT INTO test VALUES (1, 'test');
             INSERT INTO test VALUES (2, 'test2');",
        )
        .unwrap();
        db_path
    }

    #[test]
    fn test_drill_config_default() {
        let config = RestoreDrillConfig::default();
        assert!(config.verify_integrity);
        assert!(config.compare_record_counts);
        assert!(config.cleanup_after_drill);
    }

    #[test]
    fn test_drill_result_display() {
        assert_eq!(DrillResult::Success.to_string(), "success");
        assert_eq!(DrillResult::Warning.to_string(), "warning");
        assert_eq!(DrillResult::Failed.to_string(), "failed");
    }

    #[test]
    fn test_is_drill_time() {
        let config = RestoreDrillConfig::default();
        let drill = RestoreDrill::new(config).unwrap();

        // This test is time-dependent, just ensure it doesn't panic
        let _ = drill.is_drill_time();
    }

    #[test]
    fn test_next_drill_date() {
        let config = RestoreDrillConfig::default();
        let drill = RestoreDrill::new(config).unwrap();

        let next = drill.next_drill_date();
        assert!(next > Utc::now());
        assert_eq!(next.weekday(), Weekday::Sun);
        assert!(next.day() <= 7);
    }

    #[tokio::test]
    async fn test_count_records() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = create_test_db(&temp_dir, "test.db");

        let config = RestoreDrillConfig::new(&db_path, temp_dir.path().join("backups"));
        let drill = RestoreDrill::new(config).unwrap();

        let count = drill.count_records(&db_path).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_verify_integrity() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = create_test_db(&temp_dir, "test.db");

        let config = RestoreDrillConfig::new(&db_path, temp_dir.path().join("backups"));
        let drill = RestoreDrill::new(config).unwrap();

        assert!(drill.verify_integrity(&db_path).unwrap());
    }

    #[test]
    fn test_drill_report_add_issue() {
        let backup = BackupMetadata::new(BackupType::Full, "src", "dst");
        let mut report = DrillReport::new(backup);

        assert_eq!(report.result, DrillResult::Success);

        report.add_issue(DrillIssue {
            severity: IssueSeverity::Low,
            category: IssueCategory::Data,
            message: "Minor issue".to_string(),
            details: None,
        });

        assert_eq!(report.result, DrillResult::Warning);

        report.add_issue(DrillIssue {
            severity: IssueSeverity::Critical,
            category: IssueCategory::Integrity,
            message: "Critical issue".to_string(),
            details: None,
        });

        assert_eq!(report.result, DrillResult::Failed);
    }
}
