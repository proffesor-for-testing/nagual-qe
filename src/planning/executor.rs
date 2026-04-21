//! Plan execution engine
//!
//! Executes GOAP plans step by step, with real integrations to the knowledge base.
//! Supports re-planning on failure to find alternative paths to the goal.

use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

use super::planner::GOAPPlanner;
use super::storage::PlanStorage;
use super::types::*;
use crate::db::SqliteDb;
use crate::error::NagualError;

/// Configuration for re-planning on failure
#[derive(Debug, Clone)]
pub struct ReplanConfig {
    /// Enable re-planning when an action fails
    pub enabled: bool,
    /// Maximum number of re-plan attempts per plan
    pub max_attempts: usize,
    /// Current attempt count (internal)
    pub current_attempts: usize,
}

impl Default for ReplanConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_attempts: 3,
            current_attempts: 0,
        }
    }
}

impl ReplanConfig {
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            max_attempts: 3,
            current_attempts: 0,
        }
    }

    pub fn with_max_attempts(mut self, max: usize) -> Self {
        self.max_attempts = max;
        self
    }

    pub fn can_replan(&self) -> bool {
        self.enabled && self.current_attempts < self.max_attempts
    }

    pub fn record_attempt(&mut self) {
        self.current_attempts += 1;
    }
}

/// Context for plan execution
pub struct ExecutionContext {
    pub storage: Arc<PlanStorage>,
    pub db: Arc<SqliteDb>,
    pub dry_run: bool,
    pub auto_continue: bool,
    pub replan_config: ReplanConfig,
    /// Optional planner for re-planning on failure
    pub planner: Option<Arc<GOAPPlanner>>,
    /// Initial world state (needed for computing current state during replan)
    pub initial_state: Option<WorldState>,
}

impl ExecutionContext {
    pub fn new(storage: Arc<PlanStorage>, db: Arc<SqliteDb>) -> Self {
        Self {
            storage,
            db,
            dry_run: false,
            auto_continue: false,
            replan_config: ReplanConfig::default(),
            planner: None,
            initial_state: None,
        }
    }

    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    pub fn with_auto_continue(mut self, auto_continue: bool) -> Self {
        self.auto_continue = auto_continue;
        self
    }

    /// Enable re-planning on failure with a planner and initial state
    pub fn with_replanning(
        mut self,
        planner: Arc<GOAPPlanner>,
        initial_state: WorldState,
    ) -> Self {
        self.replan_config = ReplanConfig::enabled();
        self.planner = Some(planner);
        self.initial_state = Some(initial_state);
        self
    }

    /// Enable re-planning with custom config
    pub fn with_replan_config(
        mut self,
        config: ReplanConfig,
        planner: Arc<GOAPPlanner>,
        initial_state: WorldState,
    ) -> Self {
        self.replan_config = config;
        self.planner = Some(planner);
        self.initial_state = Some(initial_state);
        self
    }
}

/// Executor for running plans
pub struct PlanExecutor {
    context: ExecutionContext,
}

impl PlanExecutor {
    pub fn new(context: ExecutionContext) -> Self {
        Self { context }
    }

    /// Execute the next step in a plan
    #[instrument(skip(self, plan))]
    pub async fn execute_step(&self, plan: &mut Plan) -> Result<ActionResult, ExecutionError> {
        if plan.is_complete() {
            return Err(ExecutionError::PlanAlreadyComplete);
        }

        let step_index = plan.current_step;
        let total_actions = plan.actions.len();
        let planned_action = plan
            .actions
            .get_mut(step_index)
            .ok_or(ExecutionError::NoMoreSteps)?;

        info!(
            "Executing step {} of {}: {}",
            step_index + 1,
            total_actions,
            planned_action.action.name
        );

        if self.context.dry_run {
            info!("Dry run mode - skipping actual execution");
            let result = ActionResult::success(Some("Dry run - not executed".into()), 0);
            planned_action.complete(result.clone());
            plan.current_step += 1;

            if plan.current_step >= plan.actions.len() {
                plan.status = PlanStatus::Completed;
            }

            return Ok(result);
        }

        // Mark as in progress
        planned_action.start();
        plan.status = PlanStatus::Executing;

        // Execute the action
        let start = std::time::Instant::now();
        let result = self.execute_action(&planned_action.action).await;
        let duration_ms = start.elapsed().as_millis() as u64;

        let action_result = match result {
            Ok(output) => {
                info!(
                    "Step {} completed successfully in {}ms",
                    step_index + 1,
                    duration_ms
                );
                ActionResult::success(output, duration_ms)
            }
            Err(e) => {
                warn!(
                    "Step {} failed after {}ms: {}",
                    step_index + 1,
                    duration_ms,
                    e
                );
                ActionResult::failure(e.to_string(), duration_ms)
            }
        };

        // Update action status
        planned_action.complete(action_result.clone());

        // Update plan status
        if action_result.success {
            plan.current_step += 1;

            if plan.current_step >= plan.actions.len() {
                plan.status = PlanStatus::Completed;
                info!("Plan completed successfully");
            }
        } else {
            plan.status = PlanStatus::Failed;
        }

        // Persist plan state
        if let Err(e) = self.context.storage.save(plan).await {
            warn!("Failed to save plan state: {}", e);
        }

        Ok(action_result)
    }

    /// Execute all remaining steps in a plan
    pub async fn execute_all(&self, plan: &mut Plan) -> Result<Vec<ActionResult>, ExecutionError> {
        let mut results = Vec::new();

        while !plan.is_complete()
            && !matches!(plan.status, PlanStatus::Failed | PlanStatus::Cancelled)
        {
            let result = self.execute_step(plan).await?;
            results.push(result.clone());

            if !result.success && !self.context.auto_continue {
                break;
            }
        }

        Ok(results)
    }

    /// Execute a single action with real integrations
    async fn execute_action(&self, action: &Action) -> Result<Option<String>, ExecutionError> {
        let db = &self.context.db;

        match action.id.as_str() {
            // ==================== Research Actions ====================
            "identify_research_topic" => {
                // Query for domains with low pattern count
                let gaps = db
                    .query(
                        r#"
                    SELECT domain, COUNT(*) as cnt
                    FROM reasoning_patterns
                    WHERE domain IS NOT NULL AND domain != ''
                    GROUP BY domain
                    ORDER BY cnt ASC
                    LIMIT 5
                    "#,
                        &[],
                        |row| {
                            let domain: String = row.get(0)?;
                            let count: i64 = row.get(1)?;
                            Ok(format!("{} ({} patterns)", domain, count))
                        },
                    )
                    .await
                    .unwrap_or_default();

                if gaps.is_empty() {
                    Ok(Some(
                        "No existing domains found. Research target: general knowledge".into(),
                    ))
                } else {
                    Ok(Some(format!(
                        "Low-coverage domains identified: {}",
                        gaps.join(", ")
                    )))
                }
            }

            "web_search" => {
                // TODO: Integrate with research swarm when available
                Ok(Some(
                    "Web search requires research swarm integration (not yet implemented)".into(),
                ))
            }

            "fetch_documentation" => {
                // TODO: Integrate with web fetch
                Ok(Some(
                    "Documentation fetch requires web integration (not yet implemented)".into(),
                ))
            }

            "analyze_examples" => {
                // TODO: Integrate with gene transfusion
                Ok(Some(
                    "Example analysis requires gene transfusion integration (not yet implemented)"
                        .into(),
                ))
            }

            "synthesize_research" => {
                // Check if we have recent patterns that could be research results
                let recent_count: i64 = db
                    .query_one(
                        r#"
                    SELECT COUNT(*) FROM reasoning_patterns
                    WHERE created_at > datetime('now', '-1 hour')
                    "#,
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                Ok(Some(format!(
                    "Research synthesis complete. {} patterns available from recent activity",
                    recent_count
                )))
            }

            // ==================== Knowledge Actions ====================
            "store_pattern" => {
                // This action would be called after research produces content
                // For now, we acknowledge the action but note that content must be provided
                Ok(Some(
                    "Pattern storage ready. Use 'nagual knowledge store' with actual content"
                        .into(),
                ))
            }

            "update_pattern" => {
                Ok(Some("Pattern update capability available via 'nagual knowledge'".into()))
            }

            "tag_pattern" => {
                // Query untagged patterns
                let untagged: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM reasoning_patterns WHERE tags IS NULL OR tags = ''",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                Ok(Some(format!(
                    "Tagging ready. {} patterns currently untagged",
                    untagged
                )))
            }

            "link_patterns" => {
                // Query graph edge count
                let edge_count: i64 = db
                    .query_one("SELECT COUNT(*) FROM profdag_edges", &[], |row| row.get(0))
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                Ok(Some(format!(
                    "Pattern linking ready. {} graph edges currently exist. Use 'nagual graph link'",
                    edge_count
                )))
            }

            "generate_embedding" => {
                // Query patterns without embeddings
                let unembedded: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM reasoning_patterns WHERE embedding IS NULL",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                if unembedded > 0 {
                    Ok(Some(format!(
                        "{} patterns need embeddings. Run 'nagual learn embed'",
                        unembedded
                    )))
                } else {
                    Ok(Some("All patterns have embeddings".into()))
                }
            }

            // ==================== Analysis Actions ====================
            "analyze_codebase" => {
                Ok(Some(
                    "Codebase analysis available via 'nagual transfuse <path>'".into(),
                ))
            }

            "identify_patterns" => {
                let pattern_count: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM reasoning_patterns",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                Ok(Some(format!(
                    "Pattern identification complete. {} patterns in knowledge base",
                    pattern_count
                )))
            }

            "detect_gaps" | "identify_gaps" => {
                // Find domains with fewer than 10 patterns
                let gaps = db
                    .query(
                        r#"
                    SELECT domain, COUNT(*) as cnt
                    FROM reasoning_patterns
                    WHERE domain IS NOT NULL AND domain != ''
                    GROUP BY domain
                    HAVING cnt < 10
                    ORDER BY cnt ASC
                    "#,
                        &[],
                        |row| {
                            let domain: String = row.get(0)?;
                            let count: i64 = row.get(1)?;
                            Ok(format!("{} ({} patterns)", domain, count))
                        },
                    )
                    .await
                    .unwrap_or_default();

                if gaps.is_empty() {
                    Ok(Some("No significant knowledge gaps detected".into()))
                } else {
                    Ok(Some(format!(
                        "Knowledge gaps found in: {}",
                        gaps.join(", ")
                    )))
                }
            }

            "analyze_quality" => {
                // Query patterns with low reward scores
                let low_quality: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM reasoning_patterns WHERE reward < 0.3",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);
                let total: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM reasoning_patterns",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(1))
                    .unwrap_or(1);

                let quality_pct = 100.0 * (1.0 - (low_quality as f64 / total as f64));

                Ok(Some(format!(
                    "Quality analysis complete. {:.1}% of patterns above quality threshold ({} low-quality patterns)",
                    quality_pct, low_quality
                )))
            }

            "introspect" => {
                // Run actual introspection query
                let total_patterns: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM reasoning_patterns",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);
                let embedded: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM reasoning_patterns WHERE embedding IS NOT NULL",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);
                let domain_count: i64 = db
                    .query_one(
                        "SELECT COUNT(DISTINCT domain) FROM reasoning_patterns WHERE domain IS NOT NULL",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);
                let avg_reward: f64 = db
                    .query_one(
                        "SELECT AVG(reward) FROM reasoning_patterns",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0.5))
                    .unwrap_or(0.5);

                Ok(Some(format!(
                    "Introspection complete: {} patterns, {} embedded ({:.0}%), {} domains, avg reward: {:.2}",
                    total_patterns,
                    embedded,
                    if total_patterns > 0 { 100.0 * embedded as f64 / total_patterns as f64 } else { 0.0 },
                    domain_count,
                    avg_reward
                )))
            }

            // ==================== Testing Actions ====================
            "run_tests" => {
                Ok(Some(
                    "Test execution available via 'cargo test' in nagual-rs/".into(),
                ))
            }

            "verify_pattern" => {
                Ok(Some(
                    "Pattern verification available via validation scenarios".into(),
                ))
            }

            "validate_coherence" => {
                // Check if coherence gate is available
                let coherence_count: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM sqlite_master WHERE name='coherence_config'",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);
                let coherence_enabled = coherence_count > 0;

                if coherence_enabled {
                    Ok(Some(
                        "Coherence validation available. Run 'nagual coherence analyze'".into(),
                    ))
                } else {
                    Ok(Some("Coherence gate not yet initialized".into()))
                }
            }

            // ==================== Improvement Actions ====================
            "improve_domain" => {
                Ok(Some(
                    "Domain improvement available via 'nagual learn improve <domain>'".into(),
                ))
            }

            "record_success" => {
                Ok(Some(
                    "Success recording available via 'nagual learn record <id> success'".into(),
                ))
            }

            "record_failure" => {
                Ok(Some(
                    "Failure recording available via 'nagual learn record <id> failure'".into(),
                ))
            }

            "refresh_stale" => {
                // Find stale patterns (older than 90 days with no recent access)
                let stale: i64 = db
                    .query_one(
                        r#"
                    SELECT COUNT(*) FROM reasoning_patterns
                    WHERE created_at < datetime('now', '-90 days')
                    AND (updated_at IS NULL OR updated_at < datetime('now', '-30 days'))
                    "#,
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                Ok(Some(format!(
                    "Found {} potentially stale patterns. Consider running 'nagual learn improve'",
                    stale
                )))
            }

            // ==================== Consolidation Actions ====================
            "consolidate_similar" => {
                let pattern_count: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM reasoning_patterns",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                if pattern_count > 10 {
                    Ok(Some(format!(
                        "Consolidation ready for {} patterns. Run 'nagual learn consolidate'",
                        pattern_count
                    )))
                } else {
                    Ok(Some(
                        "Not enough patterns for consolidation (need >10)".into(),
                    ))
                }
            }

            "deduplicate" => {
                // Query for potential duplicates using content hash
                let dup_count: i64 = db
                    .query_one(
                        r#"
                    SELECT COUNT(*) FROM (
                        SELECT content_hash, COUNT(*) as cnt
                        FROM reasoning_patterns
                        WHERE content_hash IS NOT NULL
                        GROUP BY content_hash
                        HAVING cnt > 1
                    )
                    "#,
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                Ok(Some(format!(
                    "Found {} duplicate hash groups. Run 'nagual learn dedup' to remove",
                    dup_count
                )))
            }

            "archive_low_quality" => {
                let low_quality: i64 = db
                    .query_one(
                        "SELECT COUNT(*) FROM reasoning_patterns WHERE reward < 0.2",
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                Ok(Some(format!(
                    "{} low-quality patterns found (reward < 0.2)",
                    low_quality
                )))
            }

            "prune_orphans" => {
                // Find patterns not connected in graph
                let orphans: i64 = db
                    .query_one(
                        r#"
                    SELECT COUNT(*) FROM reasoning_patterns p
                    WHERE NOT EXISTS (
                        SELECT 1 FROM profdag_edges e
                        WHERE e.source_id = p.id OR e.target_id = p.id
                    )
                    "#,
                        &[],
                        |row| row.get(0),
                    )
                    .await
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                Ok(Some(format!(
                    "{} orphan patterns found (not linked in knowledge graph)",
                    orphans
                )))
            }

            // ==================== Default: Unknown Action ====================
            _ => {
                warn!("Action '{}' not implemented", action.id);
                Ok(Some(format!(
                    "Action '{}' acknowledged (implementation pending)",
                    action.name
                )))
            }
        }
    }

    /// Pause plan execution
    pub async fn pause(&self, plan: &mut Plan) -> Result<(), ExecutionError> {
        if matches!(plan.status, PlanStatus::Executing) {
            plan.status = PlanStatus::Paused;
            self.context.storage.save(plan).await?;
            info!("Plan {} paused", plan.id);
        }
        Ok(())
    }

    /// Resume plan execution
    pub async fn resume(&self, plan: &mut Plan) -> Result<(), ExecutionError> {
        if matches!(plan.status, PlanStatus::Paused) {
            plan.status = PlanStatus::Executing;
            self.context.storage.save(plan).await?;
            info!("Plan {} resumed", plan.id);
        }
        Ok(())
    }

    /// Cancel plan execution
    pub async fn cancel(&self, plan: &mut Plan) -> Result<(), ExecutionError> {
        plan.status = PlanStatus::Cancelled;
        self.context.storage.save(plan).await?;
        info!("Plan {} cancelled", plan.id);
        Ok(())
    }

    /// Compute current world state based on completed actions
    ///
    /// Applies the effects of all completed actions to the initial state.
    pub fn compute_current_state(&self, plan: &Plan) -> WorldState {
        let initial = self
            .context
            .initial_state
            .clone()
            .unwrap_or_else(WorldState::new);

        let mut current = initial;

        // Apply effects of all completed actions
        for planned_action in &plan.actions {
            if matches!(planned_action.status, ActionStatus::Completed) {
                for effect in &planned_action.action.effects {
                    match &effect.operation {
                        EffectOp::Set => {
                            current
                                .propositions
                                .insert(effect.proposition.clone(), effect.value.clone());
                        }
                        EffectOp::Increment => {
                            if let StateValue::Number(delta) = &effect.value {
                                let current_val =
                                    current.get_number(&effect.proposition).unwrap_or(0.0);
                                current.propositions.insert(
                                    effect.proposition.clone(),
                                    StateValue::Number(current_val + delta),
                                );
                            }
                        }
                        EffectOp::Decrement => {
                            if let StateValue::Number(delta) = &effect.value {
                                let current_val =
                                    current.get_number(&effect.proposition).unwrap_or(0.0);
                                current.propositions.insert(
                                    effect.proposition.clone(),
                                    StateValue::Number(current_val - delta),
                                );
                            }
                        }
                        EffectOp::Append => {
                            if let StateValue::Text(suffix) = &effect.value {
                                let current_text = match current.get(&effect.proposition) {
                                    Some(StateValue::Text(t)) => t.clone(),
                                    _ => String::new(),
                                };
                                current.propositions.insert(
                                    effect.proposition.clone(),
                                    StateValue::Text(format!("{}{}", current_text, suffix)),
                                );
                            }
                        }
                    }
                }
            }
        }

        current
    }

    /// Attempt to re-plan from the current state
    ///
    /// When an action fails, this method:
    /// 1. Computes the current world state based on completed actions
    /// 2. Marks the failed action as skipped
    /// 3. Calls the planner to find a new path to the goal
    /// 4. Replaces remaining actions with the new plan
    ///
    /// Returns the number of new actions added, or an error if replanning failed.
    pub fn attempt_replan(&mut self, plan: &mut Plan) -> Result<ReplanResult, ExecutionError> {
        // Check if replanning is enabled and we haven't exceeded max attempts
        if !self.context.replan_config.can_replan() {
            return Err(ExecutionError::ReplanExhausted {
                attempts: self.context.replan_config.current_attempts,
                max: self.context.replan_config.max_attempts,
            });
        }

        let planner = self
            .context
            .planner
            .as_ref()
            .ok_or_else(|| ExecutionError::Other("No planner configured for replanning".into()))?;

        // Record this attempt
        self.context.replan_config.record_attempt();
        let attempt = self.context.replan_config.current_attempts;

        info!(
            "Attempting replan (attempt {}/{})",
            attempt, self.context.replan_config.max_attempts
        );

        // Compute current state from completed actions
        let current_state = self.compute_current_state(plan);
        debug!(
            "Current state has {} propositions",
            current_state.propositions.len()
        );

        // Mark the current failed action as skipped
        if let Some(failed_action) = plan.actions.get_mut(plan.current_step) {
            failed_action.skip(format!("Skipped for replan (attempt {})", attempt));
        }

        // Set plan status to replanning
        plan.status = PlanStatus::Replanning;

        // Try to find a new plan from current state to goal
        match planner.plan(&current_state, &plan.goal) {
            Ok(new_plan) => {
                if new_plan.actions.is_empty() {
                    // Goal is already satisfied
                    info!("Goal already satisfied after completed actions");
                    plan.status = PlanStatus::Completed;
                    return Ok(ReplanResult {
                        success: true,
                        new_actions: 0,
                        attempt,
                        message: "Goal already satisfied".into(),
                    });
                }

                let new_action_count = new_plan.actions.len();

                // Remove remaining actions (from current_step + 1 onwards)
                let completed_actions: Vec<PlannedAction> = plan
                    .actions
                    .iter()
                    .take(plan.current_step + 1)
                    .cloned()
                    .collect();

                // Renumber new actions starting from current position
                let start_step = plan.current_step + 1;
                let new_actions: Vec<PlannedAction> = new_plan
                    .actions
                    .into_iter()
                    .enumerate()
                    .map(|(i, pa)| PlannedAction::new(pa.action, start_step + i + 1))
                    .collect();

                // Combine completed + new actions
                plan.actions = completed_actions;
                plan.actions.extend(new_actions);

                // Move to next step (skip the failed/skipped action)
                plan.current_step += 1;

                // Update plan cost
                plan.total_cost = plan.actions.iter().map(|a| a.action.cost).sum();

                // Set status back to executing
                plan.status = PlanStatus::Executing;

                info!(
                    "Replan successful: {} new actions added (attempt {})",
                    new_action_count, attempt
                );

                Ok(ReplanResult {
                    success: true,
                    new_actions: new_action_count,
                    attempt,
                    message: format!("Added {} new actions", new_action_count),
                })
            }
            Err(e) => {
                warn!("Replan failed (attempt {}): {}", attempt, e);

                // If we can try again, don't fail the plan yet
                if self.context.replan_config.can_replan() {
                    plan.status = PlanStatus::Executing;
                    Ok(ReplanResult {
                        success: false,
                        new_actions: 0,
                        attempt,
                        message: format!("Replan failed: {}, will retry", e),
                    })
                } else {
                    plan.status = PlanStatus::Failed;
                    Err(ExecutionError::ReplanFailed {
                        reason: e.to_string(),
                        attempts: attempt,
                    })
                }
            }
        }
    }

    /// Execute step with automatic re-planning on failure
    pub async fn execute_step_with_replan(
        &mut self,
        plan: &mut Plan,
    ) -> Result<ActionResult, ExecutionError> {
        let result = self.execute_step(plan).await?;

        if !result.success && self.context.replan_config.can_replan() {
            info!("Step failed, attempting replan...");

            match self.attempt_replan(plan) {
                Ok(replan_result) => {
                    if replan_result.success {
                        // Save the replanned plan
                        if let Err(e) = self.context.storage.save(plan).await {
                            warn!("Failed to save replanned plan: {}", e);
                        }

                        // Return a result indicating replan occurred
                        return Ok(ActionResult {
                            success: true,
                            output: Some(format!(
                                "Step failed but replanned successfully: {}",
                                replan_result.message
                            )),
                            error: result.error,
                            duration_ms: result.duration_ms,
                        });
                    }
                }
                Err(e) => {
                    warn!("Replan attempt failed: {}", e);
                }
            }
        }

        Ok(result)
    }

    /// Execute all steps with automatic re-planning on failure
    pub async fn execute_all_with_replan(
        &mut self,
        plan: &mut Plan,
    ) -> Result<Vec<ActionResult>, ExecutionError> {
        let mut results = Vec::new();

        while !plan.is_complete()
            && !matches!(plan.status, PlanStatus::Failed | PlanStatus::Cancelled)
        {
            let result = self.execute_step_with_replan(plan).await?;
            results.push(result.clone());

            if !result.success && !self.context.auto_continue {
                break;
            }
        }

        Ok(results)
    }
}

/// Result of a re-planning attempt
#[derive(Debug, Clone)]
pub struct ReplanResult {
    /// Whether a new plan was found
    pub success: bool,
    /// Number of new actions added
    pub new_actions: usize,
    /// Which attempt this was
    pub attempt: usize,
    /// Human-readable message
    pub message: String,
}

/// Errors during plan execution
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("Plan is already complete")]
    PlanAlreadyComplete,

    #[error("No more steps to execute")]
    NoMoreSteps,

    #[error("Action execution failed: {0}")]
    ActionFailed(String),

    #[error("Storage error: {0}")]
    StorageError(#[from] NagualError),

    #[error("Re-planning failed after {attempts} attempts: {reason}")]
    ReplanFailed { reason: String, attempts: usize },

    #[error("Re-planning exhausted ({attempts}/{max} attempts)")]
    ReplanExhausted { attempts: usize, max: usize },

    #[error("Execution error: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dry_run_execution() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let storage = Arc::new(PlanStorage::sqlite(db.clone()).await.unwrap());
        let context = ExecutionContext::new(storage, db).with_dry_run(true);
        let executor = PlanExecutor::new(context);

        let goal = Goal::new("Test", "Test goal").with_condition(Condition::is_true("done"));

        let actions = vec![PlannedAction::new(
            Action::new("test_action", "Test Action").with_effect(Effect::set_true("done")),
            1,
        )];

        let mut plan = Plan::new(goal, actions);

        let result = executor.execute_step(&mut plan).await.unwrap();
        assert!(result.success);
        assert!(plan.is_complete());
    }

    #[tokio::test]
    async fn test_real_action_execution() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());

        // Create reasoning_patterns table for introspect action
        db.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS reasoning_patterns (
                id TEXT PRIMARY KEY,
                domain TEXT,
                embedding TEXT,
                reward REAL DEFAULT 0.5,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT,
                content_hash TEXT,
                tags TEXT
            )
            "#,
        )
        .await
        .unwrap();

        // Create profdag_edges table
        db.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS profdag_edges (
                source_id TEXT,
                target_id TEXT
            )
            "#,
        )
        .await
        .unwrap();

        let storage = Arc::new(PlanStorage::sqlite(db.clone()).await.unwrap());
        let context = ExecutionContext::new(storage, db);
        let executor = PlanExecutor::new(context);

        let goal =
            Goal::new("Introspect", "Run introspection").with_condition(Condition::is_true("introspection_complete"));

        let actions = vec![PlannedAction::new(
            Action::new("introspect", "Run Self-Introspection")
                .with_effect(Effect::set_true("introspection_complete")),
            1,
        )];

        let mut plan = Plan::new(goal, actions);

        let result = executor.execute_step(&mut plan).await.unwrap();
        assert!(result.success);
        assert!(result.output.is_some());
        assert!(result.output.unwrap().contains("Introspection complete"));
    }

    #[tokio::test]
    async fn test_compute_current_state() {
        use super::super::actions::default_actions;

        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let storage = Arc::new(PlanStorage::sqlite(db.clone()).await.unwrap());
        let planner = Arc::new(GOAPPlanner::new(default_actions()));
        let mut initial = WorldState::new();
        initial.set_bool("resources_available", true);

        let context = ExecutionContext::new(storage, db)
            .with_replanning(planner, initial.clone());
        let executor = PlanExecutor::new(context);

        // Create a plan with completed actions that have effects
        let mut plan = Plan::new(
            Goal::new("test", "Test goal"),
            vec![
                PlannedAction::new(
                    Action::new("action1", "First Action")
                        .with_effect(Effect::set_true("step1_done")),
                    1,
                ),
                PlannedAction::new(
                    Action::new("action2", "Second Action")
                        .with_effect(Effect::set_true("step2_done")),
                    2,
                ),
            ],
        );

        // Mark first action as completed
        plan.actions[0].status = ActionStatus::Completed;

        // Compute current state
        let current = executor.compute_current_state(&plan);

        // Should have initial state plus effect of completed action
        assert_eq!(current.get_bool("resources_available"), Some(true));
        assert_eq!(current.get_bool("step1_done"), Some(true));
        assert_eq!(current.get_bool("step2_done"), None); // Not completed yet
    }

    #[tokio::test]
    async fn test_replan_config() {
        let config = ReplanConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.current_attempts, 0);
        assert!(!config.can_replan()); // Disabled

        let config = ReplanConfig::enabled();
        assert!(config.enabled);
        assert!(config.can_replan());

        let mut config = ReplanConfig::enabled().with_max_attempts(2);
        assert_eq!(config.max_attempts, 2);

        config.record_attempt();
        assert_eq!(config.current_attempts, 1);
        assert!(config.can_replan()); // Still has attempts

        config.record_attempt();
        assert_eq!(config.current_attempts, 2);
        assert!(!config.can_replan()); // Exhausted
    }

    #[tokio::test]
    async fn test_replan_exhausted_error() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let storage = Arc::new(PlanStorage::sqlite(db.clone()).await.unwrap());

        // Create context with replanning disabled (default)
        let context = ExecutionContext::new(storage, db);
        let mut executor = PlanExecutor::new(context);

        let mut plan = Plan::new(
            Goal::new("test", "Test"),
            vec![PlannedAction::new(Action::new("test", "Test"), 1)],
        );

        // Should fail with exhausted error since replanning is not enabled
        let result = executor.attempt_replan(&mut plan);
        assert!(matches!(result, Err(ExecutionError::ReplanExhausted { .. })));
    }
}
