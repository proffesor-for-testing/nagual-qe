//! Pulse command - heartbeat visualization for pattern creation activity.
//!
//! Renders a GitHub-style contribution heatmap in the terminal showing
//! pattern creation frequency over the past N weeks. Uses Unicode block
//! characters with varying intensity to represent daily counts.

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{Datelike, Duration, Local, NaiveDate};
use clap::Args;
use rusqlite::Connection;

use crate::error::Result;

/// Unicode block characters for intensity levels.
const BLOCK_EMPTY: char = ' ';
const BLOCK_LOW: char = '\u{2591}'; // ░
const BLOCK_MED: char = '\u{2592}'; // ▒
const BLOCK_HIGH: char = '\u{2593}'; // ▓
const BLOCK_FULL: char = '\u{2588}'; // █

/// Pulse command for heartbeat visualization.
///
/// Queries pattern creation dates from the SQLite database and renders
/// a 52-week x 7-day grid in the terminal using Unicode block characters.
/// Color intensity represents the daily pattern count.
#[derive(Args, Debug)]
pub struct PulseCommand {
    /// Path to SQLite database.
    #[arg(long, default_value = "./nagual.db")]
    pub db_path: PathBuf,

    /// Number of weeks to display (1-104).
    #[arg(long, default_value = "52")]
    pub weeks: u32,
}

/// Map a daily pattern count to a Unicode block character.
fn count_to_block(count: u32) -> char {
    match count {
        0 => BLOCK_EMPTY,
        1..=2 => BLOCK_LOW,
        3..=5 => BLOCK_MED,
        6..=10 => BLOCK_HIGH,
        _ => BLOCK_FULL,
    }
}

/// Map a daily pattern count to an ANSI color code.
/// Uses green shades similar to GitHub's contribution graph.
fn count_to_color(count: u32) -> &'static str {
    match count {
        0 => "\x1b[90m",       // dark gray
        1..=2 => "\x1b[32m",   // green
        3..=5 => "\x1b[92m",   // bright green
        6..=10 => "\x1b[33m",  // yellow
        _ => "\x1b[93m",       // bright yellow
    }
}

/// Query daily pattern counts from the SQLite database.
fn query_daily_counts(db_path: &PathBuf) -> std::result::Result<HashMap<NaiveDate, u32>, String> {
    let conn = Connection::open(db_path).map_err(|e| format!("Cannot open database: {}", e))?;

    let mut stmt = conn
        .prepare(
            "SELECT DATE(created_at) as day, COUNT(*) as cnt \
             FROM reasoning_patterns \
             GROUP BY DATE(created_at)",
        )
        .map_err(|e| format!("Query prepare failed: {}", e))?;

    let rows = stmt
        .query_map([], |row| {
            let day_str: String = row.get(0)?;
            let cnt: i64 = row.get(1)?;
            Ok((day_str, cnt as u32))
        })
        .map_err(|e| format!("Query execution failed: {}", e))?;

    let mut counts = HashMap::new();
    for row_result in rows {
        if let Ok((day_str, cnt)) = row_result {
            if let Ok(date) = NaiveDate::parse_from_str(&day_str, "%Y-%m-%d") {
                counts.insert(date, cnt);
            }
        }
    }

    Ok(counts)
}

/// Compute the starting date (a Monday) for the grid given a number of weeks.
fn grid_start_date(weeks: u32) -> NaiveDate {
    let today = Local::now().date_naive();
    let days_since_monday = today.weekday().num_days_from_monday();
    // End of the grid is the coming Sunday (end of current week)
    let end_of_week = today + Duration::days(6 - days_since_monday as i64);
    // Start is `weeks` weeks before end_of_week's Monday
    end_of_week - Duration::weeks(weeks as i64) + Duration::days(1)
}

/// Build month labels aligned to the week columns.
fn build_month_labels(start: NaiveDate, weeks: u32) -> String {
    let mut labels = String::new();
    let mut last_month = 0u32;

    for w in 0..weeks {
        let week_start = start + Duration::weeks(w as i64);
        let month = week_start.month();
        if month != last_month {
            let name = month_abbrev(month);
            labels.push_str(name);
            last_month = month;
        } else {
            labels.push(' ');
        }
    }

    labels
}

/// Get a 3-letter month abbreviation from a 1-based month number.
fn month_abbrev(month: u32) -> &'static str {
    match month {
        1 => "Jan",
        2 => "Feb",
        3 => "Mar",
        4 => "Apr",
        5 => "May",
        6 => "Jun",
        7 => "Jul",
        8 => "Aug",
        9 => "Sep",
        10 => "Oct",
        11 => "Nov",
        12 => "Dec",
        _ => "???",
    }
}

/// Weekday label for a given row (0=Mon, 1=Tue, ..., 6=Sun).
fn weekday_label(row: u32) -> &'static str {
    match row {
        0 => "Mon",
        1 => "    ",
        2 => "Wed",
        3 => "    ",
        4 => "Fri",
        5 => "    ",
        6 => "Sun",
        _ => "    ",
    }
}

/// Render the pulse grid to a String.
///
/// The grid has 7 rows (Mon-Sun) and `weeks` columns.
/// Each cell shows a Unicode block character whose intensity
/// reflects the number of patterns created on that day.
fn render_grid(counts: &HashMap<NaiveDate, u32>, weeks: u32) -> String {
    let reset = "\x1b[0m";
    let start = grid_start_date(weeks);
    let mut output = String::new();

    // Header
    output.push_str("\n  Nagual Pulse - Pattern Creation Heartbeat\n");
    output.push_str("  ==========================================\n\n");

    // Month labels row
    output.push_str("      "); // indent for weekday labels
    let month_labels = build_month_labels(start, weeks);
    output.push_str(&month_labels);
    output.push('\n');

    // Grid rows (Mon=0 through Sun=6)
    for row in 0..7u32 {
        output.push_str(weekday_label(row));
        output.push(' ');

        for w in 0..weeks {
            let date = start + Duration::weeks(w as i64) + Duration::days(row as i64);
            let count = counts.get(&date).copied().unwrap_or(0);
            let block = count_to_block(count);
            let color = count_to_color(count);
            output.push_str(color);
            output.push(block);
            output.push(reset.chars().next().unwrap_or(' '));
            // Full reset sequence per cell
            output.push_str(&reset[1..]);
        }
        output.push('\n');
    }

    // Legend
    output.push_str("\n  Legend: ");
    output.push_str(&format!("\x1b[90m {reset} none  "));
    output.push_str(&format!("\x1b[32m{BLOCK_LOW}{reset} 1-2  "));
    output.push_str(&format!("\x1b[92m{BLOCK_MED}{reset} 3-5  "));
    output.push_str(&format!("\x1b[33m{BLOCK_HIGH}{reset} 6-10  "));
    output.push_str(&format!("\x1b[93m{BLOCK_FULL}{reset} >10"));
    output.push('\n');

    // Summary stats
    let total: u32 = counts.values().sum();
    let active_days = counts.len();
    let max_day = counts.values().max().copied().unwrap_or(0);
    output.push_str(&format!(
        "\n  Total: {} patterns across {} active days (max {} in a day)\n",
        total, active_days, max_day
    ));

    output
}

impl PulseCommand {
    /// Execute the pulse command.
    pub async fn run(&self) -> Result<()> {
        let weeks = self.weeks.clamp(1, 104);

        let counts = match query_daily_counts(&self.db_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: Could not query database: {}", e);
                eprintln!("Showing empty pulse grid.\n");
                HashMap::new()
            }
        };

        let grid = render_grid(&counts, weeks);
        print!("{}", grid);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_to_block_intensity_levels() {
        assert_eq!(count_to_block(0), BLOCK_EMPTY, "count=0 should render space");
        assert_eq!(count_to_block(1), BLOCK_LOW, "count=1 should render low block");
        assert_eq!(count_to_block(2), BLOCK_LOW, "count=2 should render low block");
        assert_eq!(count_to_block(3), BLOCK_MED, "count=3 should render medium block");
        assert_eq!(count_to_block(5), BLOCK_MED, "count=5 should render medium block");
        assert_eq!(count_to_block(6), BLOCK_HIGH, "count=6 should render high block");
        assert_eq!(count_to_block(10), BLOCK_HIGH, "count=10 should render high block");
        assert_eq!(count_to_block(11), BLOCK_FULL, "count=11 should render full block");
        assert_eq!(count_to_block(100), BLOCK_FULL, "count=100 should render full block");
    }

    #[test]
    fn test_date_bucketing_and_grid_start() {
        use chrono::Weekday;
        let start = grid_start_date(52);
        // Start should be a Monday
        assert_eq!(
            start.weekday(),
            Weekday::Mon,
            "Grid should start on a Monday"
        );

        // The grid should span roughly 52 weeks (364 days)
        let today = Local::now().date_naive();
        let diff = today - start;
        // Should be between 357 and 371 days (52 weeks give or take weekday alignment)
        assert!(
            diff.num_days() >= 357 && diff.num_days() <= 371,
            "Grid span should be approximately 52 weeks, got {} days",
            diff.num_days()
        );
    }

    #[test]
    fn test_render_grid_empty_data() {
        let counts: HashMap<NaiveDate, u32> = HashMap::new();
        let grid = render_grid(&counts, 52);

        // Grid should contain the header
        assert!(grid.contains("Nagual Pulse"), "Grid should contain header");
        // Grid should contain weekday labels
        assert!(grid.contains("Mon"), "Grid should contain Mon label");
        assert!(grid.contains("Wed"), "Grid should contain Wed label");
        assert!(grid.contains("Fri"), "Grid should contain Fri label");
        // Grid should contain the legend
        assert!(grid.contains("Legend"), "Grid should contain legend");
        // Total should be 0
        assert!(
            grid.contains("Total: 0 patterns across 0 active days"),
            "Empty grid should show 0 patterns"
        );
    }

    #[test]
    fn test_render_grid_with_known_data() {
        let mut counts = HashMap::new();
        let today = Local::now().date_naive();
        counts.insert(today, 7);
        counts.insert(today - Duration::days(1), 2);
        counts.insert(today - Duration::days(7), 15);

        let grid = render_grid(&counts, 4);

        // Should show 3 active days
        assert!(
            grid.contains("3 active days"),
            "Grid should report 3 active days, got: {}",
            grid
        );
        // Total should be 7+2+15 = 24
        assert!(
            grid.contains("Total: 24 patterns"),
            "Grid should report 24 total patterns"
        );
        // Max in a day should be 15
        assert!(
            grid.contains("max 15 in a day"),
            "Grid should report max 15"
        );
    }

    #[test]
    fn test_cli_args_parse_pulse_defaults() {
        use clap::Parser;

        #[derive(Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: TestCmd,
        }

        #[derive(clap::Subcommand, Debug)]
        enum TestCmd {
            Pulse(PulseCommand),
        }

        let args = vec!["test", "pulse"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::Pulse(cmd) => {
                assert_eq!(cmd.weeks, 52, "Default weeks should be 52");
                assert_eq!(
                    cmd.db_path,
                    PathBuf::from("./nagual.db"),
                    "Default db_path should be ./nagual.db"
                );
            }
        }
    }

    #[test]
    fn test_cli_args_parse_pulse_custom() {
        use clap::Parser;

        #[derive(Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: TestCmd,
        }

        #[derive(clap::Subcommand, Debug)]
        enum TestCmd {
            Pulse(PulseCommand),
        }

        let args = vec!["test", "pulse", "--weeks", "26", "--db-path", "/tmp/test.db"];
        let cli = TestCli::try_parse_from(args).unwrap();
        match cli.cmd {
            TestCmd::Pulse(cmd) => {
                assert_eq!(cmd.weeks, 26, "Weeks should be 26");
                assert_eq!(
                    cmd.db_path,
                    PathBuf::from("/tmp/test.db"),
                    "db_path should be /tmp/test.db"
                );
            }
        }
    }

    #[test]
    fn test_month_abbrev() {
        assert_eq!(month_abbrev(1), "Jan");
        assert_eq!(month_abbrev(6), "Jun");
        assert_eq!(month_abbrev(12), "Dec");
        assert_eq!(month_abbrev(0), "???");
        assert_eq!(month_abbrev(13), "???");
    }

    #[test]
    fn test_weekday_labels() {
        assert_eq!(weekday_label(0), "Mon");
        assert_eq!(weekday_label(2), "Wed");
        assert_eq!(weekday_label(4), "Fri");
        assert_eq!(weekday_label(6), "Sun");
        // Even rows (Tue, Thu, Sat) should be blank-ish
        assert_eq!(weekday_label(1), "    ");
        assert_eq!(weekday_label(3), "    ");
        assert_eq!(weekday_label(5), "    ");
    }

    #[test]
    fn test_count_to_color_returns_ansi() {
        // All color codes should start with ESC [
        assert!(count_to_color(0).starts_with("\x1b["));
        assert!(count_to_color(1).starts_with("\x1b["));
        assert!(count_to_color(4).starts_with("\x1b["));
        assert!(count_to_color(8).starts_with("\x1b["));
        assert!(count_to_color(20).starts_with("\x1b["));
    }

    #[tokio::test]
    async fn test_pulse_run_with_empty_db() {
        let tmp_path = std::env::temp_dir().join("nagual_test_pulse_empty.db");
        let _ = std::fs::remove_file(&tmp_path);

        // Create a DB with the reasoning_patterns table but no rows
        {
            let conn = Connection::open(&tmp_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE reasoning_patterns (
                    id TEXT PRIMARY KEY,
                    problem TEXT,
                    solution TEXT,
                    domain TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );",
            )
            .unwrap();
        }

        let cmd = PulseCommand {
            db_path: tmp_path.clone(),
            weeks: 4,
        };

        // Should succeed without errors
        let result = cmd.run().await;
        assert!(result.is_ok(), "Pulse with empty DB should succeed");
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[tokio::test]
    async fn test_pulse_run_with_data() {
        let tmp_path = std::env::temp_dir().join("nagual_test_pulse_data.db");
        let _ = std::fs::remove_file(&tmp_path);

        {
            let conn = Connection::open(&tmp_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE reasoning_patterns (
                    id TEXT PRIMARY KEY,
                    problem TEXT,
                    solution TEXT,
                    domain TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                INSERT INTO reasoning_patterns (id, problem, solution, domain, created_at)
                VALUES ('p1', 'test', 'sol', 'rust', datetime('now'));
                INSERT INTO reasoning_patterns (id, problem, solution, domain, created_at)
                VALUES ('p2', 'test2', 'sol2', 'rust', datetime('now'));",
            )
            .unwrap();
        }

        let cmd = PulseCommand {
            db_path: tmp_path.clone(),
            weeks: 4,
        };

        let result = cmd.run().await;
        assert!(result.is_ok(), "Pulse with data should succeed");
        let _ = std::fs::remove_file(&tmp_path);
    }
}
