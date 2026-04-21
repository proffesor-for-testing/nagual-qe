//! Shared utility functions for the activity module.

use chrono::{DateTime, Duration, Utc};

// Re-export storage initialization from common module
pub(super) use crate::cli::common::{init_storage, resolve_postgres_url};

pub(super) const DEFAULT_SCREENPIPE_URL: &str = "http://localhost:3030";

/// Minimal percent-encoding for query parameters.
pub(super) fn urlencoding(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace('&', "%26")
        .replace('=', "%3D")
        .replace('+', "%2B")
        .replace('#', "%23")
}

/// Resolve the Screenpipe API URL from explicit arg, config file, or default.
pub(super) fn resolve_screenpipe_url(explicit: Option<&str>) -> String {
    if let Some(url) = explicit {
        if !url.is_empty() {
            return url.to_string();
        }
    }
    if let Some(home) = std::env::var("HOME").ok().map(std::path::PathBuf::from) {
        let config_path = home.join(".nagual").join("config.toml");
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("screenpipe_url") {
                    if let Some(value) = trimmed.split('=').nth(1) {
                        let url = value.trim().trim_matches('"').trim_matches('\'');
                        if !url.is_empty() {
                            return url.to_string();
                        }
                    }
                }
            }
        }
    }
    DEFAULT_SCREENPIPE_URL.to_string()
}

/// Parse a human-friendly duration string into a chrono Duration.
///
/// Supports: "today" (since midnight), "30m", "4h", "7d", "2w".
pub(super) fn parse_duration(s: &str) -> Duration {
    let s = s.trim().to_lowercase();
    if s == "today" {
        let now = Utc::now();
        let midnight = now.date_naive().and_hms_opt(0, 0, 0).unwrap();
        let midnight_utc: DateTime<Utc> = DateTime::from_naive_utc_and_offset(midnight, Utc);
        return now - midnight_utc;
    }
    let (num_str, unit) = s.split_at(s.len().saturating_sub(1));
    let num: i64 = num_str.parse().unwrap_or(1);
    match unit {
        "m" => Duration::minutes(num),
        "h" => Duration::hours(num),
        "d" => Duration::days(num),
        "w" => Duration::weeks(num),
        _ => Duration::hours(1),
    }
}

/// Categorize an application name into a domain.
pub(super) fn categorize_app(app_name: &str) -> &'static str {
    let s = app_name.to_lowercase();
    if s.contains("code") || s.contains("vim") || s.contains("neovim")
        || s.contains("intellij") || s.contains("xcode") || s.contains("cursor")
    {
        "coding"
    } else if s.contains("chrome") || s.contains("firefox") || s.contains("safari")
        || s.contains("arc") || s.contains("brave") || s.contains("edge")
    {
        "browsing"
    } else if s.contains("slack") || s.contains("discord") || s.contains("teams")
        || s.contains("messages") || s.contains("telegram")
    {
        "communication"
    } else if s.contains("zoom") || s.contains("meet") || s.contains("facetime") {
        "meetings"
    } else if s.contains("terminal") || s.contains("iterm") || s.contains("warp")
        || s.contains("alacritty") || s.contains("kitty")
    {
        "terminal"
    } else if s.contains("finder") || s.contains("preview") || s.contains("photos") {
        "files"
    } else if s.contains("notion") || s.contains("obsidian") || s.contains("bear")
        || s.contains("notes")
    {
        "notes"
    } else if s.contains("figma") || s.contains("sketch") || s.contains("canva") {
        "design"
    } else if s.contains("spotify") || s.contains("music") || s.contains("youtube") {
        "media"
    } else {
        "other"
    }
}

/// Truncate a string to a maximum length, adding "..." if truncated.
pub(super) fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
