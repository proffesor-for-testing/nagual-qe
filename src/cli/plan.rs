//! Plan command for GOAP planning
//!
//! Goal-Oriented Action Planning with SQLite persistence and knowledge-aware world state.

use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info};

use crate::db::SqliteDb;
use crate::planning::{
    default_actions, workflow::WorkflowDefinition, ExecutionContext, GOAPPlanner, GoalParser, Plan,
    PlanExecutor, PlanStatus, PlanStorage, WorldState,
};

/// Plan command for goal-oriented action planning
#[derive(Args, Debug)]
pub struct PlanCommand {
    #[command(subcommand)]
    pub subcommand: Option<PlanSubcommand>,

    /// Natural language goal to plan for (shortcut for 'plan create')
    #[arg(index = 1)]
    pub goal: Option<String>,

    /// Priority level (1-10)
    #[arg(long, short = 'p', default_value = "5")]
    pub priority: u8,

    /// Show plan without executing (dry run)
    #[arg(long)]
    pub dry_run: bool,

    /// Path to SQLite database
    #[arg(long, default_value = "nagual.db")]
    pub db_path: PathBuf,
}

#[derive(Subcommand, Debug)]
pub enum PlanSubcommand {
    /// Create a new plan for a goal
    Create {
        /// The goal description in natural language
        goal: String,

        /// Priority level (1-10, higher = more important)
        #[arg(long, short = 'p', default_value = "5")]
        priority: u8,

        /// Execute immediately after planning
        #[arg(long, short = 'x')]
        execute: bool,
    },

    /// Show current/recent plan status
    Status {
        /// Plan ID (optional, shows current if not specified)
        plan_id: Option<String>,
    },

    /// Execute the next step in a plan
    Step {
        /// Plan ID
        plan_id: String,

        /// Attempt to re-plan if the step fails
        #[arg(long, short = 'r')]
        replan: bool,
    },

    /// Execute all remaining steps in a plan
    Run {
        /// Plan ID
        plan_id: String,

        /// Continue even if a step fails
        #[arg(long)]
        continue_on_error: bool,

        /// Attempt to re-plan if a step fails
        #[arg(long, short = 'r')]
        replan: bool,

        /// Maximum re-plan attempts (default: 3)
        #[arg(long, default_value = "3")]
        max_replan_attempts: usize,
    },

    /// List all plans
    List {
        /// Filter by status (ready, executing, completed, failed)
        #[arg(long, short = 's')]
        status: Option<String>,

        /// Maximum number of plans to show
        #[arg(long, short = 'l', default_value = "10")]
        limit: usize,
    },

    /// Show available actions for planning
    Actions {
        /// Filter by category
        #[arg(long, short = 'c')]
        category: Option<String>,
    },

    /// Cancel a plan
    Cancel {
        /// Plan ID
        plan_id: String,
    },

    /// Delete a plan
    Delete {
        /// Plan ID
        plan_id: String,

        /// Skip confirmation
        #[arg(long)]
        force: bool,
    },

    /// Show current world state (knowledge base status)
    World,

    /// Validate a YAML workflow file
    Validate {
        /// Path to the YAML workflow file
        path: PathBuf,
    },

    /// Show a YAML workflow file (steps, dependencies, failure policies)
    Show {
        /// Path to the YAML workflow file
        path: PathBuf,
    },
}

impl PlanCommand {
    pub async fn execute(&self, json_output: bool) -> Result<(), crate::error::NagualError> {
        // Workflow subcommands don't need the DB
        if let Some(sub) = &self.subcommand {
            match sub {
                PlanSubcommand::Validate { path } => {
                    return Self::validate_workflow(path, json_output).await;
                }
                PlanSubcommand::Show { path } => {
                    return Self::show_workflow(path, json_output).await;
                }
                _ => {}
            }
        }

        // Open SQLite database for persistent storage
        let db = Arc::new(SqliteDb::open(&self.db_path)?);
        let storage = Arc::new(PlanStorage::sqlite(db.clone()).await?);

        match &self.subcommand {
            Some(sub) => match sub {
                PlanSubcommand::Create {
                    goal,
                    priority,
                    execute,
                } => {
                    self.create_plan(goal, *priority, *execute, json_output, storage, db)
                        .await
                }
                PlanSubcommand::Status { plan_id } => {
                    self.show_status(plan_id.as_deref(), json_output, storage)
                        .await
                }
                PlanSubcommand::Step { plan_id, replan } => {
                    self.execute_step(plan_id, *replan, json_output, storage, db)
                        .await
                }
                PlanSubcommand::Run {
                    plan_id,
                    continue_on_error,
                    replan,
                    max_replan_attempts,
                } => {
                    self.run_plan(
                        plan_id,
                        *continue_on_error,
                        *replan,
                        *max_replan_attempts,
                        json_output,
                        storage,
                        db,
                    )
                    .await
                }
                PlanSubcommand::List { status, limit } => {
                    self.list_plans(status.as_deref(), *limit, json_output, storage)
                        .await
                }
                PlanSubcommand::Actions { category } => {
                    self.show_actions(category.as_deref(), json_output).await
                }
                PlanSubcommand::Cancel { plan_id } => self.cancel_plan(plan_id, storage).await,
                PlanSubcommand::Delete { plan_id, force } => {
                    self.delete_plan(plan_id, *force, storage).await
                }
                PlanSubcommand::World => self.show_world_state(json_output, db).await,
                // Validate and Show are handled before DB initialization (early return above)
                PlanSubcommand::Validate { .. } | PlanSubcommand::Show { .. } => unreachable!(),
            },
            None => {
                // If goal is provided directly, create and show plan
                if let Some(goal) = &self.goal {
                    self.create_plan(goal, self.priority, false, json_output, storage, db)
                        .await
                } else {
                    // Show help
                    println!("Usage: nagual plan <GOAL> or nagual plan <SUBCOMMAND>");
                    println!();
                    println!("Examples:");
                    println!("  nagual plan \"improve test coverage\"");
                    println!("  nagual plan create \"research async patterns\" --execute");
                    println!("  nagual plan status");
                    println!("  nagual plan list --status ready");
                    println!("  nagual plan actions");
                    println!("  nagual plan world  # Show current world state");
                    Ok(())
                }
            }
        }
    }

    /// Build world state from the knowledge base
    async fn build_world_state(&self, db: &SqliteDb) -> WorldState {
        let mut state = WorldState::new();

        // Query pattern count
        if let Ok(Some(count)) = db
            .query_one(
                "SELECT COUNT(*) FROM reasoning_patterns",
                &[],
                |row| row.get::<_, i64>(0),
            )
            .await
        {
            state.set_number("pattern_count", count as f64);
            debug!("World state: pattern_count = {}", count);
        }

        // Check if domains have been analyzed (have patterns)
        if let Ok(domains) = db
            .query(
                "SELECT DISTINCT domain FROM reasoning_patterns WHERE domain IS NOT NULL AND domain != ''",
                &[],
                |row| row.get::<_, String>(0),
            )
            .await
        {
            for domain in &domains {
                let key = format!("{}_analyzed", domain.replace('.', "_"));
                state.set_bool(&key, true);
            }
            debug!("World state: {} domains analyzed", domains.len());
        }

        // Check for recent introspection (within last 24 hours)
        // For now, just check if introspect module exists
        state.set_bool("introspection_available", true);

        // Check for coherence gate availability
        if let Ok(Some(config_exists)) = db
            .query_one(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='coherence_config'",
                &[],
                |row| row.get::<_, i64>(0),
            )
            .await
        {
            state.set_bool("coherence_available", config_exists > 0);
        }

        // Check for embeddings availability
        if let Ok(Some(embedded_count)) = db
            .query_one(
                "SELECT COUNT(*) FROM reasoning_patterns WHERE embedding IS NOT NULL",
                &[],
                |row| row.get::<_, i64>(0),
            )
            .await
        {
            state.set_number("embedded_patterns", embedded_count as f64);
            state.set_bool("embeddings_available", embedded_count > 0);
        }

        // Check for test availability
        state.set_bool("tests_exist", true); // Assume tests exist if this is nagual

        // Check for codebase path (for gene transfusion)
        state.set_bool("codebase_path_known", true);

        state
    }

    async fn create_plan(
        &self,
        goal_desc: &str,
        priority: u8,
        execute: bool,
        json_output: bool,
        storage: Arc<PlanStorage>,
        db: Arc<SqliteDb>,
    ) -> Result<(), crate::error::NagualError> {
        info!("Creating plan for goal: {}", goal_desc);

        // Parse goal from natural language
        let mut goal = GoalParser::parse(goal_desc).map_err(|e| {
            crate::error::NagualError::Internal {
                message: format!("Failed to parse goal: {}", e),
            }
        })?;
        goal.priority = priority;

        // Create planner with default actions
        let planner = GOAPPlanner::new(default_actions());

        // Build world state from knowledge base
        let current = self.build_world_state(&db).await;
        debug!(
            "Built world state with {} propositions",
            current.propositions.len()
        );

        // Plan
        let plan = planner.plan(&current, &goal).map_err(|e| {
            crate::error::NagualError::Internal {
                message: format!("Planning failed: {}", e),
            }
        })?;

        // Save plan to SQLite
        storage.save(&plan).await?;
        info!("Plan {} saved to database", plan.id);

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan).unwrap_or_default()
            );
        } else {
            self.print_plan(&plan);
        }

        // Execute if requested
        if execute && !plan.actions.is_empty() {
            println!();
            println!("Executing plan...");
            let context = ExecutionContext::new(storage.clone(), db.clone())
                .with_dry_run(self.dry_run);
            let executor = PlanExecutor::new(context);
            let mut plan = plan;
            let results = executor.execute_all(&mut plan).await.map_err(|e| {
                crate::error::NagualError::Internal {
                    message: format!("Execution failed: {}", e),
                }
            })?;

            println!();
            println!("Execution complete: {} steps", results.len());
            for (i, result) in results.iter().enumerate() {
                let status = if result.success { "✓" } else { "✗" };
                println!(
                    "  {} Step {}: {}",
                    status,
                    i + 1,
                    result.output.as_deref().unwrap_or("completed")
                );
            }
        }

        Ok(())
    }

    async fn show_status(
        &self,
        plan_id: Option<&str>,
        json_output: bool,
        storage: Arc<PlanStorage>,
    ) -> Result<(), crate::error::NagualError> {
        let plan = if let Some(id) = plan_id {
            storage.load(id).await?
        } else {
            storage.get_current().await?
        };

        match plan {
            Some(plan) => {
                if json_output {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&plan).unwrap_or_default()
                    );
                } else {
                    self.print_plan(&plan);
                }
            }
            None => {
                println!("No active plan found.");
                println!("Create one with: nagual plan \"your goal here\"");
            }
        }

        Ok(())
    }

    async fn execute_step(
        &self,
        plan_id: &str,
        replan: bool,
        json_output: bool,
        storage: Arc<PlanStorage>,
        db: Arc<SqliteDb>,
    ) -> Result<(), crate::error::NagualError> {
        let mut plan = storage.load(plan_id).await?.ok_or_else(|| {
            crate::error::NagualError::Internal {
                message: format!("Plan not found: {}", plan_id),
            }
        })?;

        let mut context =
            ExecutionContext::new(storage.clone(), db.clone()).with_dry_run(self.dry_run);

        // Configure replanning if enabled
        if replan {
            let planner = Arc::new(GOAPPlanner::new(default_actions()));
            let initial_state = self.build_world_state(&db).await;
            context = context.with_replanning(planner, initial_state);
        }

        let mut executor = PlanExecutor::new(context);

        let result = if replan {
            executor
                .execute_step_with_replan(&mut plan)
                .await
                .map_err(|e| crate::error::NagualError::Internal {
                    message: format!("Step execution failed: {}", e),
                })?
        } else {
            executor.execute_step(&mut plan).await.map_err(|e| {
                crate::error::NagualError::Internal {
                    message: format!("Step execution failed: {}", e),
                }
            })?
        };

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&result).unwrap_or_default()
            );
        } else {
            let status = if result.success { "✓" } else { "✗" };
            println!("{} Step completed", status);
            if let Some(output) = &result.output {
                println!("  Output: {}", output);
            }
            if let Some(error) = &result.error {
                println!("  Error: {}", error);
            }
            println!("  Duration: {}ms", result.duration_ms);
            println!();
            println!(
                "Progress: {:.0}% ({}/{})",
                plan.progress_percentage(),
                plan.completed_steps(),
                plan.step_count()
            );
        }

        Ok(())
    }

    async fn run_plan(
        &self,
        plan_id: &str,
        continue_on_error: bool,
        replan: bool,
        max_replan_attempts: usize,
        json_output: bool,
        storage: Arc<PlanStorage>,
        db: Arc<SqliteDb>,
    ) -> Result<(), crate::error::NagualError> {
        let mut plan = storage.load(plan_id).await?.ok_or_else(|| {
            crate::error::NagualError::Internal {
                message: format!("Plan not found: {}", plan_id),
            }
        })?;

        let mut context = ExecutionContext::new(storage.clone(), db.clone())
            .with_dry_run(self.dry_run)
            .with_auto_continue(continue_on_error);

        // Configure replanning if enabled
        if replan {
            let planner = Arc::new(GOAPPlanner::new(default_actions()));
            let initial_state = self.build_world_state(&db).await;
            let replan_config =
                crate::planning::ReplanConfig::enabled().with_max_attempts(max_replan_attempts);
            context = context.with_replan_config(replan_config, planner, initial_state);
        }

        let mut executor = PlanExecutor::new(context);

        let results = if replan {
            executor
                .execute_all_with_replan(&mut plan)
                .await
                .map_err(|e| crate::error::NagualError::Internal {
                    message: format!("Execution failed: {}", e),
                })?
        } else {
            executor.execute_all(&mut plan).await.map_err(|e| {
                crate::error::NagualError::Internal {
                    message: format!("Execution failed: {}", e),
                }
            })?
        };

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&results).unwrap_or_default()
            );
        } else {
            println!("Plan execution complete");
            println!();
            for (i, result) in results.iter().enumerate() {
                let status = if result.success { "✓" } else { "✗" };
                println!(
                    "  {} Step {}: {}",
                    status,
                    i + 1,
                    result.output.as_deref().unwrap_or("completed")
                );
            }
            println!();
            println!("Final status: {}", plan.status);
        }

        Ok(())
    }

    async fn list_plans(
        &self,
        status_filter: Option<&str>,
        limit: usize,
        json_output: bool,
        storage: Arc<PlanStorage>,
    ) -> Result<(), crate::error::NagualError> {
        let plans = if let Some(status_str) = status_filter {
            let status = match status_str.to_lowercase().as_str() {
                "ready" => PlanStatus::Ready,
                "executing" => PlanStatus::Executing,
                "completed" => PlanStatus::Completed,
                "failed" => PlanStatus::Failed,
                "paused" => PlanStatus::Paused,
                "cancelled" => PlanStatus::Cancelled,
                _ => {
                    return Err(crate::error::NagualError::Internal {
                        message: format!("Unknown status: {}", status_str),
                    })
                }
            };
            storage.list_by_status(status).await?
        } else {
            storage.list().await?
        };

        let plans: Vec<_> = plans.into_iter().take(limit).collect();

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&plans).unwrap_or_default()
            );
        } else if plans.is_empty() {
            println!("No plans found.");
        } else {
            println!("Plans ({}):", plans.len());
            println!();
            for plan in &plans {
                let status_icon = match plan.status {
                    PlanStatus::Ready => "○",
                    PlanStatus::Executing => "◐",
                    PlanStatus::Completed => "●",
                    PlanStatus::Failed => "✗",
                    PlanStatus::Paused => "⏸",
                    PlanStatus::Cancelled => "⊘",
                    _ => "?",
                };
                println!(
                    "  {} [{}] {} ({} steps, {:.0}%)",
                    status_icon,
                    &plan.id[..8.min(plan.id.len())],
                    plan.goal.name,
                    plan.step_count(),
                    plan.progress_percentage()
                );
            }
        }

        Ok(())
    }

    async fn show_actions(
        &self,
        category_filter: Option<&str>,
        json_output: bool,
    ) -> Result<(), crate::error::NagualError> {
        let actions = default_actions();

        let filtered: Vec<_> = if let Some(cat) = category_filter {
            let cat_lower = cat.to_lowercase();
            actions
                .into_iter()
                .filter(|a| format!("{:?}", a.category).to_lowercase().contains(&cat_lower))
                .collect()
        } else {
            actions
        };

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&filtered).unwrap_or_default()
            );
        } else {
            println!("Available Actions ({}):", filtered.len());
            println!();

            let mut current_category = None;
            for action in &filtered {
                let category = format!("{:?}", action.category);
                if current_category.as_ref() != Some(&category) {
                    println!();
                    println!("{}:", category);
                    current_category = Some(category);
                }
                println!("  • {} (cost: {:.1})", action.name, action.cost);
                if !action.description.is_empty() {
                    println!("    {}", action.description);
                }
            }
        }

        Ok(())
    }

    async fn cancel_plan(
        &self,
        plan_id: &str,
        storage: Arc<PlanStorage>,
    ) -> Result<(), crate::error::NagualError> {
        let mut plan = storage.load(plan_id).await?.ok_or_else(|| {
            crate::error::NagualError::Internal {
                message: format!("Plan not found: {}", plan_id),
            }
        })?;

        plan.status = PlanStatus::Cancelled;
        storage.save(&plan).await?;

        println!("Plan {} cancelled", plan_id);
        Ok(())
    }

    async fn delete_plan(
        &self,
        plan_id: &str,
        _force: bool,
        storage: Arc<PlanStorage>,
    ) -> Result<(), crate::error::NagualError> {
        // TODO: Add confirmation if not --force
        let deleted = storage.delete(plan_id).await?;

        if deleted {
            println!("Plan {} deleted", plan_id);
        } else {
            println!("Plan {} not found", plan_id);
        }

        Ok(())
    }

    async fn show_world_state(
        &self,
        json_output: bool,
        db: Arc<SqliteDb>,
    ) -> Result<(), crate::error::NagualError> {
        let state = self.build_world_state(&db).await;

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&state).unwrap_or_default()
            );
        } else {
            println!("Current World State");
            println!("==================");
            println!();

            let mut keys: Vec<_> = state.propositions.keys().collect();
            keys.sort();

            for key in keys {
                if let Some(value) = state.propositions.get(key) {
                    println!("  {}: {}", key, value);
                }
            }
        }

        Ok(())
    }

    async fn validate_workflow(
        path: &Path,
        json_output: bool,
    ) -> Result<(), crate::error::NagualError> {
        let workflow = WorkflowDefinition::from_yaml_file(path).map_err(|e| {
            crate::error::NagualError::Internal {
                message: format!("Failed to parse workflow '{}': {}", path.display(), e),
            }
        })?;

        match workflow.validate() {
            Ok(()) => {
                if json_output {
                    println!(
                        "{}",
                        serde_json::json!({
                            "valid": true,
                            "name": workflow.name,
                            "steps": workflow.steps.len(),
                        })
                    );
                } else {
                    println!(
                        "Workflow '{}' is valid ({} steps)",
                        workflow.name,
                        workflow.steps.len()
                    );
                }
            }
            Err(errors) => {
                if json_output {
                    println!(
                        "{}",
                        serde_json::json!({
                            "valid": false,
                            "name": workflow.name,
                            "errors": errors,
                        })
                    );
                } else {
                    println!("Workflow '{}' has {} error(s):", workflow.name, errors.len());
                    for err in &errors {
                        println!("  - {}", err);
                    }
                }
            }
        }

        Ok(())
    }

    async fn show_workflow(
        path: &Path,
        json_output: bool,
    ) -> Result<(), crate::error::NagualError> {
        let workflow = WorkflowDefinition::from_yaml_file(path).map_err(|e| {
            crate::error::NagualError::Internal {
                message: format!("Failed to parse workflow '{}': {}", path.display(), e),
            }
        })?;

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&workflow).unwrap_or_default()
            );
            return Ok(());
        }

        println!("Workflow: {}", workflow.name);
        println!("Description: {}", workflow.description);
        println!("Domain: {}", workflow.domain);
        println!("Timeout: {}s", workflow.timeout_secs);

        if !workflow.variables.is_empty() {
            println!();
            println!("Variables:");
            for (key, value) in &workflow.variables {
                println!("  {} = {}", key, value);
            }
        }

        println!();
        println!("Steps ({}):", workflow.steps.len());
        for (i, step) in workflow.steps.iter().enumerate() {
            println!();
            println!("  {}. [{}] (agent: {})", i + 1, step.id, step.agent_type);
            println!("     Prompt: {}", step.prompt);

            if !step.depends_on.is_empty() {
                println!("     Depends on: {}", step.depends_on.join(", "));
            }

            match &step.on_failure {
                crate::planning::workflow::FailurePolicy::Abort => {
                    println!("     On failure: abort");
                }
                crate::planning::workflow::FailurePolicy::Skip => {
                    println!("     On failure: skip");
                }
                crate::planning::workflow::FailurePolicy::Retry { max_retries } => {
                    println!("     On failure: retry (max {})", max_retries);
                }
            }

            if step.approval_gate {
                println!("     Approval gate: yes");
            }

            if let Some(timeout) = step.timeout_secs {
                println!("     Timeout: {}s", timeout);
            }

            if step.background {
                println!("     Background: yes");
            }
        }

        // Show validation status
        println!();
        match workflow.validate() {
            Ok(()) => println!("Validation: passed"),
            Err(errors) => {
                println!("Validation: {} error(s)", errors.len());
                for err in &errors {
                    println!("  - {}", err);
                }
            }
        }

        Ok(())
    }

    fn print_plan(&self, plan: &Plan) {
        println!("Plan: {}", plan.goal.name);
        println!("ID: {}", plan.id);
        println!("Status: {}", plan.status);
        println!("Priority: {}", plan.goal.priority);
        println!("Created: {}", plan.created_at.format("%Y-%m-%d %H:%M:%S"));

        if let Some(duration) = plan.estimated_duration_seconds {
            println!("Estimated Duration: {}s", duration);
        }

        println!();
        println!("Steps ({}):", plan.step_count());

        if plan.actions.is_empty() {
            println!("  (goal already satisfied - no actions needed)");
        } else {
            for action in &plan.actions {
                let status_icon = match action.status {
                    crate::planning::ActionStatus::Pending => "○",
                    crate::planning::ActionStatus::InProgress => "◐",
                    crate::planning::ActionStatus::Completed => "●",
                    crate::planning::ActionStatus::Failed => "✗",
                    crate::planning::ActionStatus::Skipped => "⊘",
                };
                println!(
                    "  {} {}. {} (cost: {:.1})",
                    status_icon, action.step, action.action.name, action.action.cost
                );
                if !action.action.description.is_empty() {
                    println!("       {}", action.action.description);
                }
            }
        }

        println!();
        println!("Total Cost: {:.1}", plan.total_cost);
        println!("Progress: {:.0}%", plan.progress_percentage());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_command_parse() {
        use clap::Parser;

        #[derive(Parser)]
        struct TestCli {
            #[command(subcommand)]
            cmd: TestCmd,
        }

        #[derive(clap::Subcommand)]
        enum TestCmd {
            Plan(PlanCommand),
        }

        let args = vec!["test", "plan", "improve test coverage"];
        let result = TestCli::try_parse_from(args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_plan_command_with_db_path() {
        use clap::Parser;

        #[derive(Parser)]
        struct TestCli {
            #[command(subcommand)]
            cmd: TestCmd,
        }

        #[derive(clap::Subcommand)]
        enum TestCmd {
            Plan(PlanCommand),
        }

        let args = vec![
            "test",
            "plan",
            "improve test coverage",
            "--db-path",
            "custom.db",
        ];
        let result = TestCli::try_parse_from(args);
        assert!(result.is_ok());
    }
}
