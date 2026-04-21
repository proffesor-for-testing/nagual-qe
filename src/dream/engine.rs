//! Dream Cycle Engine
//!
//! Background maintenance system that runs during idle periods.
//! Performs pattern consolidation, refresh, prediction calibration,
//! and spreading activation.
//!
//! ## Integration with Other Components
//!
//! The Dream Cycle can optionally integrate with:
//! - `ResearchCoordinator`: For triggering research on stale patterns
//! - `CalibrationAdjuster`: For updating prediction calibration
//! - `GraphStorage`: For creating/strengthening edges in the context graph

use std::sync::Arc;
use std::time::Instant;
use chrono::{Duration, Utc};
use tracing::{debug, info, warn, instrument};

use super::types::*;
use crate::db::SqliteDb;
use crate::error::NagualError;
use crate::graph::{GraphStorage, EdgeType};
use crate::research::{ResearchCoordinator, ResearchRequest, ResearchDepth};

/// Local calibration bucket for dream cycle calibration phase
/// (Minimal version of prediction::calibration::CalibrationBucket)
#[derive(Debug, Clone)]
struct DreamCalibrationBucket {
    pub bucket_id: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub prediction_count: u32,
    pub actual_positive_count: u32,
    pub total_brier_score: f64,
    pub domain: String,
    pub updated_at: chrono::DateTime<Utc>,
}

impl DreamCalibrationBucket {
    fn new(lower_bound: f64, upper_bound: f64) -> Self {
        let bucket_index = (lower_bound * 10.0).round() as u32;
        Self {
            bucket_id: format!("general_{}", bucket_index),
            lower_bound,
            upper_bound,
            prediction_count: 0,
            actual_positive_count: 0,
            total_brier_score: 0.0,
            domain: "general".to_string(),
            updated_at: Utc::now(),
        }
    }

    /// Update the bucket with a resolved prediction
    fn update(&mut self, actual_outcome: bool, brier_score: f64) {
        self.prediction_count += 1;
        if actual_outcome {
            self.actual_positive_count += 1;
        }
        self.total_brier_score += brier_score;
        self.updated_at = Utc::now();
    }
}

/// Dream cycle engine with optional component integrations
pub struct DreamCycle {
    db: Arc<SqliteDb>,
    config: DreamConfig,
    last_cycle: Option<DreamResult>,
    last_activity: Instant,
    total_cycles: usize,
    total_items_processed: usize,
    /// Optional research coordinator for refresh phase
    research: Option<Arc<ResearchCoordinator>>,
    /// Optional graph storage for activate phase
    graph: Option<Arc<GraphStorage>>,
}

impl DreamCycle {
    /// Create a new dream cycle engine
    pub fn new(db: Arc<SqliteDb>, config: DreamConfig) -> Self {
        Self {
            db,
            config,
            last_cycle: None,
            last_activity: Instant::now(),
            total_cycles: 0,
            total_items_processed: 0,
            research: None,
            graph: None,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(db: Arc<SqliteDb>) -> Self {
        Self::new(db, DreamConfig::default())
    }

    /// Create a fully-integrated dream cycle with all components
    pub fn with_integrations(
        db: Arc<SqliteDb>,
        config: DreamConfig,
        research: Option<Arc<ResearchCoordinator>>,
        graph: Option<Arc<GraphStorage>>,
    ) -> Self {
        Self {
            db,
            config,
            last_cycle: None,
            last_activity: Instant::now(),
            total_cycles: 0,
            total_items_processed: 0,
            research,
            graph,
        }
    }

    /// Set the research coordinator for refresh phase
    pub fn set_research(&mut self, research: Arc<ResearchCoordinator>) {
        self.research = Some(research);
    }

    /// Set the graph storage for activate phase
    pub fn set_graph(&mut self, graph: Arc<GraphStorage>) {
        self.graph = Some(graph);
    }

    /// Check if research integration is enabled
    pub fn has_research_integration(&self) -> bool {
        self.research.is_some()
    }

    /// Check if graph integration is enabled
    pub fn has_graph_integration(&self) -> bool {
        self.graph.is_some()
    }

    /// Update configuration
    pub fn set_config(&mut self, config: DreamConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &DreamConfig {
        &self.config
    }

    /// Record activity (resets idle timer)
    pub fn record_activity(&mut self) {
        self.last_activity = Instant::now();
    }

    /// Check if system is idle enough for a dream cycle
    pub fn is_idle(&self) -> bool {
        self.last_activity.elapsed().as_secs() >= self.config.idle_threshold_seconds
    }

    /// Get current status
    pub fn status(&self) -> DreamStatus {
        let state = if !self.config.enabled {
            DreamState::Disabled
        } else {
            DreamState::Idle
        };

        let next_cycle_in_seconds = if self.config.enabled {
            let elapsed = self.last_activity.elapsed().as_secs();
            if elapsed < self.config.idle_threshold_seconds {
                Some(self.config.idle_threshold_seconds - elapsed)
            } else {
                Some(0) // Ready to run
            }
        } else {
            None
        };

        DreamStatus {
            enabled: self.config.enabled,
            state,
            last_cycle: self.last_cycle.clone(),
            next_cycle_in_seconds,
            total_cycles: self.total_cycles,
            total_items_processed: self.total_items_processed,
        }
    }

    /// Run a complete dream cycle
    #[instrument(skip(self))]
    pub async fn run_cycle(&mut self) -> Result<DreamResult, NagualError> {
        let cycle_id = uuid::Uuid::new_v4().to_string();
        let started_at = Utc::now();
        let mut phases_completed = Vec::new();
        let mut tokens_used = 0;

        info!(cycle_id = %cycle_id, "Starting dream cycle");

        let deadline = started_at + Duration::seconds(self.config.max_duration_seconds as i64);

        // Phase 1: Consolidate
        if self.config.phases.consolidate && Utc::now() < deadline {
            debug!("Running consolidate phase");
            match self.run_consolidate_phase().await {
                Ok(result) => phases_completed.push(result),
                Err(e) => {
                    warn!("Consolidate phase failed: {}", e);
                    phases_completed.push(PhaseResult {
                        phase: DreamPhase::Consolidate,
                        success: false,
                        items_processed: 0,
                        duration_ms: 0,
                        details: PhaseDetails::Consolidate {
                            patterns_merged: 0,
                            patterns_archived: 0,
                            duplicates_removed: 0,
                        },
                    });
                }
            }
        }

        // Phase 2: Refresh
        if self.config.phases.refresh && Utc::now() < deadline {
            debug!("Running refresh phase");
            match self.run_refresh_phase().await {
                Ok((result, toks)) => {
                    phases_completed.push(result);
                    tokens_used += toks;
                }
                Err(e) => {
                    warn!("Refresh phase failed: {}", e);
                    phases_completed.push(PhaseResult {
                        phase: DreamPhase::Refresh,
                        success: false,
                        items_processed: 0,
                        duration_ms: 0,
                        details: PhaseDetails::Refresh {
                            patterns_refreshed: 0,
                            research_triggered: 0,
                            patterns_updated: 0,
                        },
                    });
                }
            }
        }

        // Phase 3: Calibrate
        if self.config.phases.calibrate && Utc::now() < deadline {
            debug!("Running calibrate phase");
            match self.run_calibrate_phase().await {
                Ok(result) => phases_completed.push(result),
                Err(e) => {
                    warn!("Calibrate phase failed: {}", e);
                    phases_completed.push(PhaseResult {
                        phase: DreamPhase::Calibrate,
                        success: false,
                        items_processed: 0,
                        duration_ms: 0,
                        details: PhaseDetails::Calibrate {
                            predictions_reviewed: 0,
                            brier_score_before: 0.0,
                            brier_score_after: 0.0,
                        },
                    });
                }
            }
        }

        // Phase 4: Activate
        if self.config.phases.activate && Utc::now() < deadline {
            debug!("Running activate phase");
            match self.run_activate_phase().await {
                Ok(result) => phases_completed.push(result),
                Err(e) => {
                    warn!("Activate phase failed: {}", e);
                    phases_completed.push(PhaseResult {
                        phase: DreamPhase::Activate,
                        success: false,
                        items_processed: 0,
                        duration_ms: 0,
                        details: PhaseDetails::Activate {
                            connections_strengthened: 0,
                            new_connections: 0,
                            activation_spread: 0.0,
                        },
                    });
                }
            }
        }

        let completed_at = Utc::now();
        let total_duration_ms = (completed_at - started_at).num_milliseconds().max(0) as u64;

        let result = DreamResult {
            cycle_id: cycle_id.clone(),
            started_at,
            completed_at,
            phases_completed,
            total_duration_ms,
            tokens_used,
        };

        // Update statistics
        self.total_cycles += 1;
        self.total_items_processed += result.total_items_processed();
        self.last_cycle = Some(result.clone());
        self.record_activity(); // Reset idle timer after cycle

        info!(
            cycle_id = %cycle_id,
            phases = result.phases_completed.len(),
            items = result.total_items_processed(),
            duration_ms = total_duration_ms,
            "Dream cycle completed"
        );

        Ok(result)
    }

    /// Run the consolidation phase
    async fn run_consolidate_phase(&self) -> Result<PhaseResult, NagualError> {
        let start = Instant::now();
        let mut merged = 0;
        let mut archived = 0;
        let mut deduped = 0;
        let mut processed = 0;

        // Get patterns for consolidation
        let patterns: Vec<ConsolidationCandidate> = self.get_consolidation_candidates().await?;

        // Find and handle similar patterns
        let budget = self.config.budget.max_patterns_consolidated;

        for i in 0..patterns.len().min(budget) {
            if processed >= budget {
                break;
            }

            for j in (i + 1)..patterns.len().min(i + 10) {
                let similarity = self.calculate_jaccard_similarity(
                    &patterns[i].problem,
                    &patterns[j].problem,
                );

                if similarity > 0.95 {
                    // Exact duplicate - archive lower reward one
                    let archive_id = if patterns[i].reward >= patterns[j].reward {
                        &patterns[j].id
                    } else {
                        &patterns[i].id
                    };
                    self.archive_pattern(archive_id).await?;
                    deduped += 1;
                } else if similarity > 0.85 {
                    // Very similar - merge into higher reward pattern
                    let (keep_id, merge_id) = if patterns[i].reward >= patterns[j].reward {
                        (&patterns[i].id, &patterns[j].id)
                    } else {
                        (&patterns[j].id, &patterns[i].id)
                    };
                    self.merge_patterns(keep_id, merge_id).await?;
                    merged += 1;
                }
            }

            processed += 1;
        }

        // Archive low-quality patterns (reward < 0.3, not accessed in 30+ days)
        let low_quality = self.find_low_quality_patterns(0.3, 30).await?;
        for pattern in low_quality.iter().take(10) {
            self.archive_pattern(&pattern.id).await?;
            archived += 1;
        }

        Ok(PhaseResult {
            phase: DreamPhase::Consolidate,
            success: true,
            items_processed: processed,
            duration_ms: start.elapsed().as_millis() as u64,
            details: PhaseDetails::Consolidate {
                patterns_merged: merged,
                patterns_archived: archived,
                duplicates_removed: deduped,
            },
        })
    }

    /// Run the refresh phase
    ///
    /// When ResearchCoordinator is integrated, this phase will:
    /// 1. Find stale patterns that haven't been updated recently
    /// 2. Trigger research for patterns that need updating
    /// 3. Update patterns with new information
    async fn run_refresh_phase(&self) -> Result<(PhaseResult, usize), NagualError> {
        let start = Instant::now();
        let mut refreshed = 0;
        let mut researched = 0;
        let mut updated = 0;
        let mut tokens = 0;

        // Find stale patterns (not updated in 30+ days)
        let stale = self.find_stale_patterns(30).await?;
        let budget = self.config.budget.max_patterns_refreshed;

        for pattern in stale.iter().take(budget) {
            // Check if pattern topic is still relevant by checking recent usage
            let relevance = self.calculate_pattern_relevance(&pattern.id).await?;

            if relevance < 0.2 {
                // Topic no longer relevant - archive it
                self.archive_pattern(&pattern.id).await?;
            } else {
                // Mark as refreshed (bump updated_at)
                self.touch_pattern(&pattern.id).await?;
                refreshed += 1;

                // Trigger research if ResearchCoordinator is integrated
                if relevance < 0.5 {
                    if let Some(ref research_coord) = self.research {
                        match self.research_for_pattern_update(research_coord, &pattern.problem, &pattern.domain).await {
                            Ok(research_tokens) => {
                                tokens += research_tokens;
                                researched += 1;
                                info!(pattern_id = %pattern.id, tokens = research_tokens, "Research triggered for stale pattern");
                            }
                            Err(e) => {
                                warn!(pattern_id = %pattern.id, error = %e, "Research failed for stale pattern");
                            }
                        }
                    } else {
                        // No research coordinator - just count as would-research
                        researched += 1;
                        debug!(pattern_id = %pattern.id, "Would trigger research (no coordinator)");
                    }
                }
            }

            updated += 1;
        }

        Ok((
            PhaseResult {
                phase: DreamPhase::Refresh,
                success: true,
                items_processed: refreshed,
                duration_ms: start.elapsed().as_millis() as u64,
                details: PhaseDetails::Refresh {
                    patterns_refreshed: refreshed,
                    research_triggered: researched,
                    patterns_updated: updated,
                },
            },
            tokens,
        ))
    }

    /// Trigger research to update a stale pattern
    async fn research_for_pattern_update(
        &self,
        coordinator: &ResearchCoordinator,
        topic: &str,
        domain: &str,
    ) -> Result<usize, NagualError> {
        let request = ResearchRequest::new(format!("Update knowledge: {}", topic))
            .with_depth(ResearchDepth::Quick)
            .with_domain(domain);

        match coordinator.research(request).await {
            Ok(result) => {
                info!(
                    topic = %topic,
                    patterns_created = result.patterns_created.len(),
                    tokens = result.total_tokens,
                    "Research completed for pattern update"
                );
                Ok(result.total_tokens)
            }
            Err(e) => {
                warn!(topic = %topic, error = %e, "Research failed");
                Err(e)
            }
        }
    }

    /// Run the calibration phase
    ///
    /// This phase:
    /// 1. Loads resolved predictions
    /// 2. Updates calibration buckets with actual outcomes
    /// 3. Calculates and reports Brier score changes
    async fn run_calibrate_phase(&self) -> Result<PhaseResult, NagualError> {
        let start = Instant::now();

        // Get resolved predictions
        let predictions = self.get_resolved_predictions().await?;
        let budget = self.config.budget.max_predictions_calibrated;
        let predictions_to_process: Vec<_> = predictions.into_iter().take(budget).collect();

        // Calculate Brier score before calibration
        let brier_before = self.calculate_brier_score(&predictions_to_process);

        // Load or create calibration buckets
        let mut buckets = self.load_calibration_buckets().await?;

        // Update calibration buckets for each prediction
        let mut updated_count = 0;
        for (id, confidence, outcome) in &predictions_to_process {
            // Find the appropriate bucket
            let bucket_idx = (confidence * 10.0).floor() as usize;
            let bucket_idx = bucket_idx.min(9); // Clamp to 9 for probability == 1.0

            if bucket_idx < buckets.len() {
                // Calculate Brier score for this prediction
                let brier = if *outcome {
                    (confidence - 1.0).powi(2)
                } else {
                    confidence.powi(2)
                };

                // Update the bucket
                buckets[bucket_idx].update(*outcome, brier);
                updated_count += 1;

                debug!(
                    prediction_id = %id,
                    confidence = %confidence,
                    outcome = %outcome,
                    bucket = bucket_idx,
                    "Updated calibration bucket"
                );
            }

            // Also update the prediction's calibration flag
            self.mark_prediction_calibrated(id).await?;
        }

        // Save updated buckets
        self.save_calibration_buckets(&buckets).await?;

        // Calculate Brier score after calibration updates
        let brier_after = if updated_count > 0 {
            // Recalculate overall Brier from updated buckets
            self.calculate_bucket_brier(&buckets)
        } else {
            brier_before
        };

        info!(
            predictions = predictions_to_process.len(),
            brier_before = %brier_before,
            brier_after = %brier_after,
            buckets_updated = updated_count,
            "Calibration phase completed"
        );

        Ok(PhaseResult {
            phase: DreamPhase::Calibrate,
            success: true,
            items_processed: predictions_to_process.len(),
            duration_ms: start.elapsed().as_millis() as u64,
            details: PhaseDetails::Calibrate {
                predictions_reviewed: predictions_to_process.len(),
                brier_score_before: brier_before,
                brier_score_after: brier_after,
            },
        })
    }

    /// Load calibration buckets from database or create defaults
    async fn load_calibration_buckets(&self) -> Result<Vec<DreamCalibrationBucket>, NagualError> {
        // Try to load from database
        let existing: Vec<DreamCalibrationBucket> = self.db.query(
            r#"
            SELECT bucket_id, lower_bound, upper_bound, prediction_count,
                   actual_positive_count, total_brier_score, domain, updated_at
            FROM calibration_buckets
            WHERE domain = 'general'
            ORDER BY lower_bound ASC
            "#,
            &[],
            |row| {
                let bucket_id: String = row.get(0)?;
                let lower_bound_str: String = row.get(1)?;
                let upper_bound_str: String = row.get(2)?;
                let prediction_count: i32 = row.get(3)?;
                let actual_positive_count: i32 = row.get(4)?;
                let total_brier_str: String = row.get(5)?;
                let domain: String = row.get(6)?;
                let updated_at_str: String = row.get(7)?;

                let updated_at = chrono::DateTime::parse_from_rfc3339(&updated_at_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                Ok(DreamCalibrationBucket {
                    bucket_id,
                    lower_bound: lower_bound_str.parse().unwrap_or(0.0),
                    upper_bound: upper_bound_str.parse().unwrap_or(0.1),
                    prediction_count: prediction_count as u32,
                    actual_positive_count: actual_positive_count as u32,
                    total_brier_score: total_brier_str.parse().unwrap_or(0.0),
                    domain,
                    updated_at,
                })
            },
        ).await.unwrap_or_default();

        if existing.len() == 10 {
            return Ok(existing);
        }

        // Create default buckets
        let buckets: Vec<DreamCalibrationBucket> = (0..10)
            .map(|i| DreamCalibrationBucket::new(i as f64 * 0.1, (i + 1) as f64 * 0.1))
            .collect();

        Ok(buckets)
    }

    /// Save calibration buckets to database
    async fn save_calibration_buckets(&self, buckets: &[DreamCalibrationBucket]) -> Result<(), NagualError> {
        // Ensure table exists
        self.db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS calibration_buckets (
                bucket_id TEXT PRIMARY KEY,
                lower_bound REAL NOT NULL,
                upper_bound REAL NOT NULL,
                prediction_count INTEGER NOT NULL DEFAULT 0,
                actual_positive_count INTEGER NOT NULL DEFAULT 0,
                total_brier_score REAL NOT NULL DEFAULT 0.0,
                domain TEXT DEFAULT 'general',
                updated_at TEXT NOT NULL
            )
            "#,
            &[],
        ).await?;

        for bucket in buckets {
            let now = Utc::now().to_rfc3339();
            self.db.execute(
                r#"
                INSERT OR REPLACE INTO calibration_buckets
                (bucket_id, lower_bound, upper_bound, prediction_count, actual_positive_count, total_brier_score, domain, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                "#,
                &[
                    &bucket.bucket_id,
                    &bucket.lower_bound.to_string(),
                    &bucket.upper_bound.to_string(),
                    &(bucket.prediction_count as i64).to_string(),
                    &(bucket.actual_positive_count as i64).to_string(),
                    &bucket.total_brier_score.to_string(),
                    &bucket.domain,
                    &now,
                ],
            ).await?;
        }

        Ok(())
    }

    /// Calculate overall Brier score from buckets
    fn calculate_bucket_brier(&self, buckets: &[DreamCalibrationBucket]) -> f64 {
        let total_predictions: u32 = buckets.iter().map(|b| b.prediction_count).sum();
        if total_predictions == 0 {
            return 0.0;
        }

        let total_brier: f64 = buckets.iter().map(|b| b.total_brier_score).sum();
        total_brier / total_predictions as f64
    }

    /// Mark a prediction as having been calibrated
    async fn mark_prediction_calibrated(&self, id: &str) -> Result<(), NagualError> {
        let now = Utc::now().to_rfc3339();
        let id_owned = id.to_string();
        self.db.execute(
            "UPDATE predictions SET calibrated_at = ? WHERE id = ?",
            &[&now, &id_owned],
        ).await?;
        Ok(())
    }

    /// Run the spreading activation phase
    ///
    /// This phase uses spreading activation to:
    /// 1. Find recently active patterns
    /// 2. Discover related patterns by domain and content similarity
    /// 3. Create/strengthen edges in the context graph
    ///
    /// When GraphStorage is integrated, uses the proper context_graph table.
    async fn run_activate_phase(&self) -> Result<PhaseResult, NagualError> {
        let start = Instant::now();
        let mut strengthened = 0;
        let mut new_connections = 0;

        // Get recently used patterns
        let active_patterns = self.get_recently_used_patterns(100).await?;

        for pattern_id in &active_patterns {
            // Find related patterns by domain and similarity
            let related = self.find_related_patterns(pattern_id, 10).await?;

            for rel in related {
                if let Some(ref graph) = self.graph {
                    // Use proper GraphStorage integration
                    match self.process_activation_with_graph(graph, pattern_id, &rel).await {
                        Ok(created) => {
                            if created {
                                new_connections += 1;
                            } else {
                                strengthened += 1;
                            }
                        }
                        Err(e) => {
                            debug!(error = %e, "Failed to process activation edge");
                        }
                    }
                } else {
                    // Fallback to simple edges table
                    let edge_exists = self.edge_exists(pattern_id, &rel.id).await?;

                    if edge_exists {
                        self.strengthen_edge(pattern_id, &rel.id, 0.1).await?;
                        strengthened += 1;
                    } else if rel.similarity > 0.7 {
                        self.create_edge(pattern_id, &rel.id, rel.similarity).await?;
                        new_connections += 1;
                    }
                }
            }
        }

        let activation_spread = if active_patterns.is_empty() {
            0.0
        } else {
            (strengthened + new_connections) as f64 / active_patterns.len() as f64
        };

        info!(
            active_patterns = active_patterns.len(),
            strengthened = strengthened,
            new_connections = new_connections,
            activation_spread = %activation_spread,
            graph_integrated = self.graph.is_some(),
            "Activate phase completed"
        );

        Ok(PhaseResult {
            phase: DreamPhase::Activate,
            success: true,
            items_processed: active_patterns.len(),
            duration_ms: start.elapsed().as_millis() as u64,
            details: PhaseDetails::Activate {
                connections_strengthened: strengthened,
                new_connections,
                activation_spread,
            },
        })
    }

    /// Process an activation edge using proper GraphStorage
    async fn process_activation_with_graph(
        &self,
        graph: &GraphStorage,
        source_id: &str,
        related: &RelatedPattern,
    ) -> Result<bool, NagualError> {
        // Try to create/update edge
        let result = graph.create_edge(
            source_id,
            &related.id,
            EdgeType::SimilarTo,
            related.similarity,
            Some(serde_json::json!({
                "source": "dream_cycle_activation",
                "created_at": Utc::now().to_rfc3339()
            })),
        ).await.map_err(|e| NagualError::internal(e.to_string()))?;

        if result.created {
            debug!(
                source = %source_id,
                target = %related.id,
                similarity = %related.similarity,
                "Created new SimilarTo edge"
            );
            Ok(true) // New edge created
        } else {
            // Edge existed - we updated strength
            debug!(
                source = %source_id,
                target = %related.id,
                previous = result.previous_strength,
                current = related.similarity,
                "Strengthened existing edge"
            );
            Ok(false) // Updated existing edge
        }
    }

    // ================== Helper Methods ==================

    async fn get_consolidation_candidates(&self) -> Result<Vec<ConsolidationCandidate>, NagualError> {
        let candidates: Vec<ConsolidationCandidate> = self.db.query(
            r#"
            SELECT id, problem, solution, category, CAST(reward AS TEXT) as reward, timestamp
            FROM reasoning_patterns
            WHERE archived IS NULL OR archived = 0
            ORDER BY reward DESC
            LIMIT 200
            "#,
            &[],
            |row| {
                let reward_str: String = row.get(4)?;
                let timestamp_str: String = row.get(5)?;
                Ok(ConsolidationCandidate {
                    id: row.get(0)?,
                    problem: row.get(1)?,
                    solution: row.get(2)?,
                    domain: row.get(3)?,
                    reward: reward_str.parse().unwrap_or(0.5),
                    created_at: chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                })
            },
        ).await?;

        Ok(candidates)
    }

    fn calculate_jaccard_similarity(&self, a: &str, b: &str) -> f64 {
        let a_words: std::collections::HashSet<_> = a.split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        let b_words: std::collections::HashSet<_> = b.split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        if a_words.is_empty() && b_words.is_empty() {
            return 1.0;
        }

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    async fn archive_pattern(&self, id: &str) -> Result<(), NagualError> {
        let now = Utc::now().to_rfc3339();
        let id_owned = id.to_string();
        self.db.execute(
            "UPDATE reasoning_patterns SET archived = 1, updated_at = ? WHERE id = ?",
            &[&now, &id_owned],
        ).await?;
        Ok(())
    }

    async fn merge_patterns(&self, keep_id: &str, merge_id: &str) -> Result<(), NagualError> {
        // Get the pattern to merge
        let merge_id_str = merge_id.to_string();
        let merge_data: Option<(String, String)> = self.db.query_one(
            "SELECT solution, tags FROM reasoning_patterns WHERE id = ?",
            &[&merge_id_str],
            |row| Ok((row.get(0)?, row.get::<_, Option<String>>(1)?.unwrap_or_default())),
        ).await?;

        if let Some((merge_solution, merge_tags)) = merge_data {
            let now = Utc::now().to_rfc3339();
            let keep_id_owned = keep_id.to_string();

            // Append solution to keep pattern
            self.db.execute(
                r#"
                UPDATE reasoning_patterns
                SET solution = solution || '\n\n[Merged from similar pattern]\n' || ?,
                    tags = CASE WHEN tags IS NULL OR tags = '' THEN ? ELSE tags || ',' || ? END,
                    updated_at = ?
                WHERE id = ?
                "#,
                &[
                    &merge_solution,
                    &merge_tags,
                    &merge_tags,
                    &now,
                    &keep_id_owned,
                ],
            ).await?;

            // Archive the merged pattern
            self.archive_pattern(merge_id).await?;
        }

        Ok(())
    }

    async fn find_low_quality_patterns(&self, min_reward: f64, days_inactive: i64) -> Result<Vec<StalePattern>, NagualError> {
        let cutoff = (Utc::now() - Duration::days(days_inactive)).to_rfc3339();
        let min_reward_str = min_reward.to_string();

        let patterns: Vec<StalePattern> = self.db.query(
            r#"
            SELECT id, problem, category
            FROM reasoning_patterns
            WHERE (archived IS NULL OR archived = 0)
              AND CAST(reward AS REAL) < ?
              AND updated_at < ?
            ORDER BY reward ASC
            LIMIT 20
            "#,
            &[&min_reward_str, &cutoff],
            |row| Ok(StalePattern {
                id: row.get(0)?,
                problem: row.get(1)?,
                domain: row.get(2)?,
                days_since_update: days_inactive,
                relevance_score: 0.0,
            }),
        ).await?;

        Ok(patterns)
    }

    async fn find_stale_patterns(&self, days_old: i64) -> Result<Vec<StalePattern>, NagualError> {
        let cutoff = (Utc::now() - Duration::days(days_old)).to_rfc3339();

        let patterns: Vec<StalePattern> = self.db.query(
            r#"
            SELECT id, problem, category, updated_at
            FROM reasoning_patterns
            WHERE (archived IS NULL OR archived = 0)
              AND updated_at < ?
            ORDER BY updated_at ASC
            LIMIT 50
            "#,
            &[&cutoff],
            |row| {
                let updated_at: String = row.get(3)?;
                let days = chrono::DateTime::parse_from_rfc3339(&updated_at)
                    .map(|dt| (Utc::now() - dt.with_timezone(&Utc)).num_days())
                    .unwrap_or(days_old);

                Ok(StalePattern {
                    id: row.get(0)?,
                    problem: row.get(1)?,
                    domain: row.get(2)?,
                    days_since_update: days,
                    relevance_score: 0.5,
                })
            },
        ).await?;

        Ok(patterns)
    }

    async fn calculate_pattern_relevance(&self, id: &str) -> Result<f64, NagualError> {
        // Check if pattern has been accessed recently
        let id_owned = id.to_string();
        let data: Option<(String, String)> = self.db.query_one(
            r#"
            SELECT CAST(reward AS TEXT), updated_at
            FROM reasoning_patterns
            WHERE id = ?
            "#,
            &[&id_owned],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).await?;

        if let Some((reward_str, updated_at)) = data {
            let reward: f64 = reward_str.parse().unwrap_or(0.5);

            let days_old = chrono::DateTime::parse_from_rfc3339(&updated_at)
                .map(|dt| (Utc::now() - dt.with_timezone(&Utc)).num_days())
                .unwrap_or(30);

            // Relevance decays with time, boosted by reward
            let decay = (-0.01 * days_old as f64).exp();
            Ok(reward * decay)
        } else {
            Ok(0.0)
        }
    }

    async fn touch_pattern(&self, id: &str) -> Result<(), NagualError> {
        let now = Utc::now().to_rfc3339();
        let id_owned = id.to_string();
        self.db.execute(
            "UPDATE reasoning_patterns SET updated_at = ? WHERE id = ?",
            &[&now, &id_owned],
        ).await?;
        Ok(())
    }

    async fn get_resolved_predictions(&self) -> Result<Vec<(String, f64, bool)>, NagualError> {
        let predictions: Vec<(String, f64, bool)> = self.db.query(
            r#"
            SELECT id, confidence, outcome
            FROM predictions
            WHERE resolved = 1
            ORDER BY resolved_at DESC
            LIMIT 100
            "#,
            &[],
            |row| {
                let id: String = row.get(0)?;
                let confidence_str: String = row.get(1)?;
                let outcome_str: String = row.get(2)?;
                let confidence: f64 = confidence_str.parse().unwrap_or(0.5);
                let outcome: bool = outcome_str == "1" || outcome_str.to_lowercase() == "true";
                Ok((id, confidence, outcome))
            },
        ).await.unwrap_or_default();

        Ok(predictions)
    }

    fn calculate_brier_score(&self, predictions: &[(String, f64, bool)]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        let sum: f64 = predictions.iter()
            .map(|(_, conf, outcome)| {
                let actual = if *outcome { 1.0 } else { 0.0 };
                (conf - actual).powi(2)
            })
            .sum();

        sum / predictions.len() as f64
    }


    async fn get_recently_used_patterns(&self, limit: usize) -> Result<Vec<String>, NagualError> {
        let limit_str = limit.to_string();
        let ids: Vec<String> = self.db.query(
            r#"
            SELECT id FROM reasoning_patterns
            WHERE (archived IS NULL OR archived = 0)
            ORDER BY updated_at DESC
            LIMIT ?
            "#,
            &[&limit_str],
            |row| row.get(0),
        ).await?;

        Ok(ids)
    }

    async fn find_related_patterns(&self, pattern_id: &str, limit: usize) -> Result<Vec<RelatedPattern>, NagualError> {
        let pattern_id_owned = pattern_id.to_string();
        let limit_str = limit.to_string();

        // Find patterns in the same domain
        let related: Vec<RelatedPattern> = self.db.query(
            r#"
            SELECT p2.id
            FROM reasoning_patterns p1
            JOIN reasoning_patterns p2 ON p1.category = p2.category AND p1.id != p2.id
            WHERE p1.id = ? AND (p2.archived IS NULL OR p2.archived = 0)
            LIMIT ?
            "#,
            &[&pattern_id_owned, &limit_str],
            |row| Ok(RelatedPattern {
                id: row.get(0)?,
                similarity: 0.75, // Domain match gives base similarity
            }),
        ).await?;

        Ok(related)
    }

    async fn edge_exists(&self, from_id: &str, to_id: &str) -> Result<bool, NagualError> {
        let from_owned = from_id.to_string();
        let to_owned = to_id.to_string();

        let count: Option<i64> = self.db.query_one(
            "SELECT COUNT(*) FROM edges WHERE source_id = ? AND target_id = ?",
            &[&from_owned, &to_owned],
            |row| row.get(0),
        ).await?;

        Ok(count.unwrap_or(0) > 0)
    }

    async fn strengthen_edge(&self, from_id: &str, to_id: &str, delta: f64) -> Result<(), NagualError> {
        let delta_str = delta.to_string();
        let now = Utc::now().to_rfc3339();
        let from_owned = from_id.to_string();
        let to_owned = to_id.to_string();

        self.db.execute(
            r#"
            UPDATE edges
            SET weight = MIN(1.0, weight + ?),
                updated_at = ?
            WHERE source_id = ? AND target_id = ?
            "#,
            &[&delta_str, &now, &from_owned, &to_owned],
        ).await?;
        Ok(())
    }

    async fn create_edge(&self, from_id: &str, to_id: &str, weight: f64) -> Result<(), NagualError> {
        let edge_id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let weight_str = weight.to_string();
        let from_owned = from_id.to_string();
        let to_owned = to_id.to_string();

        self.db.execute(
            r#"
            INSERT OR IGNORE INTO edges (id, source_id, target_id, weight, edge_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'spreading_activation', ?, ?)
            "#,
            &[&edge_id, &from_owned, &to_owned, &weight_str, &now, &now],
        ).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn setup_dream_cycle() -> DreamCycle {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());

        // Create tables
        db.execute(
            r#"CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                problem TEXT,
                solution TEXT,
                category TEXT,
                reward REAL DEFAULT 0.5,
                tags TEXT,
                archived INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"#,
            &[],
        ).await.unwrap();

        db.execute(
            r#"CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                confidence REAL,
                outcome TEXT,
                resolved INTEGER DEFAULT 0,
                resolved_at TEXT
            )"#,
            &[],
        ).await.unwrap();

        db.execute(
            r#"CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT,
                target_id TEXT,
                weight REAL,
                edge_type TEXT,
                created_at TEXT,
                updated_at TEXT
            )"#,
            &[],
        ).await.unwrap();

        // Add test patterns
        let now = Utc::now().to_rfc3339();
        for i in 0..10 {
            let id = format!("pattern-{}", i);
            let problem = format!("Test problem {}", i);
            let solution = format!("Test solution {}", i);
            let category = "test.domain".to_string();
            let reward = "0.6".to_string();
            db.execute(
                r#"INSERT INTO reasoning_patterns (id, problem, solution, category, reward, timestamp, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)"#,
                &[&id, &problem, &solution, &category, &reward, &now, &now],
            ).await.unwrap();
        }

        DreamCycle::new(db, DreamConfig::default())
    }

    #[tokio::test]
    async fn test_dream_cycle_creation() {
        let cycle = setup_dream_cycle().await;
        assert!(cycle.config.enabled);
        assert_eq!(cycle.total_cycles, 0);
    }

    #[tokio::test]
    async fn test_dream_cycle_status() {
        let cycle = setup_dream_cycle().await;
        let status = cycle.status();

        assert!(status.enabled);
        assert_eq!(status.state, DreamState::Idle);
        assert_eq!(status.total_cycles, 0);
        assert!(status.last_cycle.is_none());
    }

    #[tokio::test]
    async fn test_run_dream_cycle() {
        let mut cycle = setup_dream_cycle().await;
        let result = cycle.run_cycle().await.unwrap();

        assert!(!result.cycle_id.is_empty());
        assert!(!result.phases_completed.is_empty());
        assert_eq!(cycle.total_cycles, 1);
    }

    #[tokio::test]
    async fn test_consolidate_phase() {
        let cycle = setup_dream_cycle().await;
        let result = cycle.run_consolidate_phase().await.unwrap();

        assert_eq!(result.phase, DreamPhase::Consolidate);
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_refresh_phase() {
        let cycle = setup_dream_cycle().await;
        let (result, tokens) = cycle.run_refresh_phase().await.unwrap();

        assert_eq!(result.phase, DreamPhase::Refresh);
        assert!(result.success);
        assert_eq!(tokens, 0); // No actual research
    }

    #[tokio::test]
    async fn test_calibrate_phase() {
        let cycle = setup_dream_cycle().await;
        let result = cycle.run_calibrate_phase().await.unwrap();

        assert_eq!(result.phase, DreamPhase::Calibrate);
        assert!(result.success);
    }

    #[tokio::test]
    async fn test_activate_phase() {
        let cycle = setup_dream_cycle().await;
        let result = cycle.run_activate_phase().await.unwrap();

        assert_eq!(result.phase, DreamPhase::Activate);
        assert!(result.success);
    }

    #[test]
    fn test_jaccard_similarity() {
        let cycle = DreamCycle {
            db: Arc::new(SqliteDb::open_in_memory().unwrap()),
            config: DreamConfig::default(),
            last_cycle: None,
            last_activity: Instant::now(),
            total_cycles: 0,
            total_items_processed: 0,
            research: None,
            graph: None,
        };

        assert!((cycle.calculate_jaccard_similarity("hello world", "hello world") - 1.0).abs() < 0.001);
        assert!((cycle.calculate_jaccard_similarity("hello world", "hello") - 0.5).abs() < 0.001);
        assert!((cycle.calculate_jaccard_similarity("a b c", "d e f") - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_brier_score_calculation() {
        let cycle = DreamCycle {
            db: Arc::new(SqliteDb::open_in_memory().unwrap()),
            config: DreamConfig::default(),
            last_cycle: None,
            last_activity: Instant::now(),
            total_cycles: 0,
            total_items_processed: 0,
            research: None,
            graph: None,
        };

        // Perfect predictions
        let perfect = vec![
            ("1".to_string(), 1.0, true),
            ("2".to_string(), 0.0, false),
        ];
        assert!((cycle.calculate_brier_score(&perfect) - 0.0).abs() < 0.001);

        // Worst predictions
        let worst = vec![
            ("1".to_string(), 0.0, true),
            ("2".to_string(), 1.0, false),
        ];
        assert!((cycle.calculate_brier_score(&worst) - 1.0).abs() < 0.001);
    }
}
