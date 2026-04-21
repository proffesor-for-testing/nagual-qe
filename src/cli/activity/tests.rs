//! Tests for the activity module.

use super::*;
use clap::Parser;

#[derive(Parser, Debug)]
struct TestCli {
    #[command(subcommand)]
    command: TestCommand,
}

#[derive(clap::Subcommand, Debug)]
enum TestCommand {
    Activity(ActivityCommand),
}

// --- Existing command parsing tests ---

#[test]
fn test_parse_status() {
    let args = vec!["test", "activity", "status"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_status_json() {
    let args = vec!["test", "activity", "status", "--json"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_ingest() {
    let args = vec!["test", "activity", "ingest", "--since", "4h"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_ingest_with_filters() {
    let args = vec![
        "test", "activity", "ingest",
        "--since", "1d",
        "--content-type", "ocr",
        "--app-name", "VS Code",
        "--focused-only",
        "--min-length", "100",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_ingest_input_content_type() {
    let args = vec![
        "test", "activity", "ingest",
        "--since", "1h",
        "--content-type", "input",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_ingest_with_embed() {
    let args = vec![
        "test", "activity", "ingest",
        "--since", "1h",
        "--embed",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_summary() {
    let args = vec!["test", "activity", "summary", "--period", "today"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_summary_with_domain() {
    let args = vec![
        "test", "activity", "summary",
        "--period", "7d",
        "--domain", "coding",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_search() {
    let args = vec![
        "test", "activity", "search",
        "--query", "rust async",
        "--limit", "5",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_search_with_filters() {
    let args = vec![
        "test", "activity", "search",
        "--query", "debugging",
        "--since", "1d",
        "--app-name", "Code",
        "--window-name", "nagual",
        "--focused-only",
        "--min-length", "20",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_search_semantic() {
    let args = vec![
        "test", "activity", "search",
        "--query", "debugging memory leak",
        "--semantic",
        "--threshold", "0.3",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_search_keyword() {
    let args = vec![
        "test", "activity", "search",
        "--query", "fn main",
        "--keyword",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_apps() {
    let args = vec!["test", "activity", "apps", "--period", "today"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_apps_json() {
    let args = vec!["test", "activity", "apps", "--period", "7d", "--json"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

// --- Duration / helper tests ---

use super::helpers::*;

#[test]
fn test_parse_duration_hours() {
    assert_eq!(parse_duration("4h").num_hours(), 4);
}

#[test]
fn test_parse_duration_days() {
    assert_eq!(parse_duration("7d").num_days(), 7);
}

#[test]
fn test_parse_duration_minutes() {
    assert_eq!(parse_duration("30m").num_minutes(), 30);
}

#[test]
fn test_parse_duration_default() {
    assert_eq!(parse_duration("???").num_hours(), 1);
}

#[test]
fn test_parse_duration_today() {
    let d = parse_duration("today");
    assert!(d.num_hours() >= 0 && d.num_hours() <= 24);
}

#[test]
fn test_categorize_app_coding() {
    assert_eq!(categorize_app("Visual Studio Code"), "coding");
    assert_eq!(categorize_app("Cursor"), "coding");
    assert_eq!(categorize_app("Xcode"), "coding");
}

#[test]
fn test_categorize_app_browsing() {
    assert_eq!(categorize_app("Google Chrome"), "browsing");
    assert_eq!(categorize_app("Firefox"), "browsing");
    assert_eq!(categorize_app("Arc"), "browsing");
}

#[test]
fn test_categorize_app_communication() {
    assert_eq!(categorize_app("Slack"), "communication");
    assert_eq!(categorize_app("Discord"), "communication");
}

#[test]
fn test_categorize_app_meetings() {
    assert_eq!(categorize_app("zoom.us"), "meetings");
    assert_eq!(categorize_app("Google Meet"), "meetings");
}

#[test]
fn test_categorize_app_terminal() {
    assert_eq!(categorize_app("Terminal"), "terminal");
    assert_eq!(categorize_app("iTerm2"), "terminal");
    assert_eq!(categorize_app("Warp"), "terminal");
}

#[test]
fn test_categorize_app_notes() {
    assert_eq!(categorize_app("Notion"), "notes");
    assert_eq!(categorize_app("Obsidian"), "notes");
}

#[test]
fn test_categorize_app_other() {
    assert_eq!(categorize_app("RandomApp"), "other");
}

#[test]
fn test_urlencoding() {
    assert_eq!(urlencoding("hello world"), "hello%20world");
    assert_eq!(urlencoding("a&b=c"), "a%26b%3Dc");
    assert_eq!(urlencoding("100%"), "100%25");
}

#[test]
fn test_truncate() {
    assert_eq!(truncate("Hello", 10), "Hello");
    assert_eq!(truncate("Hello World!", 8), "Hello...");
}

#[test]
fn test_resolve_screenpipe_url_default() {
    assert_eq!(resolve_screenpipe_url(None), DEFAULT_SCREENPIPE_URL);
}

#[test]
fn test_resolve_screenpipe_url_explicit() {
    assert_eq!(
        resolve_screenpipe_url(Some("http://custom:8080")),
        "http://custom:8080"
    );
}

// --- New command parsing tests ---

#[test]
fn test_parse_tags_add() {
    let args = vec![
        "test", "activity", "tags", "add",
        "vision", "123", "important", "review",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_tags_remove() {
    let args = vec![
        "test", "activity", "tags", "remove",
        "audio", "456", "draft",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_speakers_list() {
    let args = vec!["test", "activity", "speakers", "list"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_speakers_list_unnamed() {
    let args = vec!["test", "activity", "speakers", "list", "--unnamed"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_speakers_list_by_name() {
    let args = vec![
        "test", "activity", "speakers", "list",
        "--name", "Alice",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_speakers_update() {
    let args = vec!["test", "activity", "speakers", "update", "1", "Alice"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_speakers_delete() {
    let args = vec!["test", "activity", "speakers", "delete", "5"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_speakers_merge() {
    let args = vec!["test", "activity", "speakers", "merge", "1", "3"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_speakers_similar() {
    let args = vec!["test", "activity", "speakers", "similar", "2", "--limit", "5"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_events_search() {
    let args = vec![
        "test", "activity", "events", "search",
        "--event-type", "click",
        "--since", "1h",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_events_search_by_app() {
    let args = vec![
        "test", "activity", "events", "search",
        "--app-name", "Code",
        "--since", "1d",
        "--limit", "20",
    ];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_events_stats() {
    let args = vec!["test", "activity", "events", "stats", "--since", "today"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_pipes_list() {
    let args = vec!["test", "activity", "pipes", "list"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_pipes_info() {
    let args = vec!["test", "activity", "pipes", "info", "my-pipe"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_pipes_enable() {
    let args = vec!["test", "activity", "pipes", "enable", "my-pipe"];
    assert!(TestCli::try_parse_from(args).is_ok());
}

#[test]
fn test_parse_pipes_disable() {
    let args = vec!["test", "activity", "pipes", "disable", "my-pipe"];
    assert!(TestCli::try_parse_from(args).is_ok());
}
