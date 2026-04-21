//! Type definitions for Screenpipe API responses and CLI output.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Screenpipe API response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(super) struct HealthResponse {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub frame_status: Option<String>,
    #[serde(default)]
    pub audio_status: Option<String>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub last_frame_timestamp: Option<String>,
    #[serde(default)]
    pub last_audio_timestamp: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub(super) struct AudioDevice {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub is_default: bool,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub(super) struct VisionMonitor {
    #[serde(default)]
    pub id: u32,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    #[serde(default)]
    pub is_default: bool,
}

#[derive(Debug, Deserialize)]
pub(super) struct ScreenpipeSearchResponse {
    #[serde(default)]
    pub data: Vec<ScreenpipeItem>,
    #[serde(default)]
    pub pagination: Option<PaginationInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct PaginationInfo {
    #[serde(default)]
    pub total: usize,
}

#[derive(Debug, Deserialize)]
pub(super) struct ScreenpipeItem {
    #[serde(rename = "type")]
    pub content_type: String,
    pub content: ScreenpipeContent,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct ScreenpipeContent {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub app_name: Option<String>,
    #[serde(default)]
    pub window_name: Option<String>,
    #[serde(default)]
    pub browser_url: Option<String>,
    #[serde(default)]
    pub focused: Option<bool>,
    #[serde(default)]
    pub transcription: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
}

// ---------------------------------------------------------------------------
// CLI output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub(super) struct StatusOutput {
    pub connected: bool,
    pub url: String,
    pub status: String,
    pub frame_status: Option<String>,
    pub audio_status: Option<String>,
    pub monitors: Vec<VisionMonitor>,
    pub audio_devices: Vec<AudioDevice>,
    pub db_rows: DbRowCounts,
}

#[derive(Serialize, Default)]
pub(super) struct DbRowCounts {
    pub frames: usize,
    pub ocr: usize,
    pub audio: usize,
    pub ui_monitoring: usize,
    pub accessibility: usize,
}

#[derive(Serialize)]
pub(super) struct IngestOutput {
    pub ingested: usize,
    pub skipped: usize,
    pub embedded: usize,
    pub time_range_start: String,
    pub time_range_end: String,
    pub source: String,
    pub patterns_created: Vec<String>,
}

/// JSON output for OSpipe pipeline ingestion.
#[derive(Serialize)]
pub(super) struct OSpipeIngestOutput {
    pub ingested: usize,
    pub rejected: usize,
    pub redacted: usize,
    pub duplicates: usize,
    pub errors: usize,
    pub embedded: usize,
    pub time_range_start: String,
    pub time_range_end: String,
    pub source: String,
    pub patterns_created: Vec<String>,
    pub dedup_stats: DedupStatsOutput,
}

/// Deduplication statistics for JSON output.
#[derive(Serialize)]
pub(super) struct DedupStatsOutput {
    pub total_checked: usize,
    pub duplicates_found: usize,
    pub unique_items: usize,
}

#[derive(Serialize)]
pub(super) struct AppUsageItem {
    pub app_name: String,
    pub domain: String,
    pub count: usize,
}

#[derive(Serialize)]
pub(super) struct SummaryOutput {
    pub period: String,
    pub total_items: usize,
    pub apps: Vec<AppUsageItem>,
    pub domains: HashMap<String, usize>,
}

#[derive(Serialize)]
pub(super) struct SearchResultItem {
    pub content_type: String,
    pub app_name: String,
    pub window_name: String,
    pub text: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub browser_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focused: Option<bool>,
}

/// Row returned by the raw_sql bulk query for ingestion.
#[derive(Debug, Deserialize)]
pub(super) struct RawOcrRow {
    #[serde(default)]
    pub frame_id: i64,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub app_name: String,
    #[serde(default)]
    pub window_name: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub focused: Option<bool>,
    #[serde(default)]
    pub browser_url: Option<String>,
}

// ---------------------------------------------------------------------------
// New Screenpipe API types: tags, speakers, UI events, pipes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct Speaker {
    #[serde(default)]
    pub id: i64,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub metadata: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct UiEvent {
    #[serde(default)]
    pub id: i64,
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub event_type: String,
    #[serde(default)]
    pub app_name: Option<String>,
    #[serde(default)]
    pub window_name: Option<String>,
    #[serde(default)]
    pub text_content: Option<String>,
    #[serde(default)]
    pub initial_traversal_at: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct UiEventsResponse {
    #[serde(default)]
    pub data: Vec<UiEvent>,
    #[serde(default)]
    pub pagination: Option<PaginationInfo>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct UiEventStat {
    pub event_type: String,
    pub count: usize,
    #[serde(default)]
    pub app_name: Option<String>,
}

#[derive(Default)]
pub(super) struct UiEventsParams<'a> {
    pub start_time: Option<&'a str>,
    pub end_time: Option<&'a str>,
    pub event_type: Option<&'a str>,
    pub app_name: Option<&'a str>,
    pub window_name: Option<&'a str>,
    pub limit: usize,
    pub offset: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct PipeInfo {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct SemanticResult {
    #[serde(default)]
    pub content_type: String,
    pub content: ScreenpipeContent,
    #[serde(default)]
    pub score: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub(super) struct SemanticSearchResponse {
    #[serde(default)]
    pub data: Vec<SemanticResult>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct VisionStatus {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub is_running: bool,
}
