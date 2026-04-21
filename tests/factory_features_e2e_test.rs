//! End-to-end tests for Software Factory features (Weeks 1-3)
//!
//! This test suite validates the integration of various Nagual features:
//! - Satisfaction metrics flow
//! - Trajectory chain analysis
//! - Deduplication workflow
//! - Pyramid summary generation
//! - Session lifecycle management
//! - Scenario evaluation for holdout validation
//! - Gene transfusion for pattern extraction

use std::sync::Arc;

use nagual::db::{DualWriteAdapter, SqliteDb};
use nagual::error::Result;
use nagual::learning::{
    scenario::{Difficulty, Scenario, ScenarioEvaluator, ScenarioStorage},
    trajectory::{Trajectory, TrajectoryBuilder, TrajectoryStep},
    Outcome,
};
use nagual::reasoning_bank::{
    dedup::{scan_duplicates, DedupConfig},
    pattern::{Pattern, PatternCategory, PatternId},
    pyramid::{generate_summary, generate_title, get_pyramid_stats, generate_missing_pyramids},
    storage::{PatternStorage, StorageConfig},
};

use tempfile::TempDir;

mod common;
use common::{normalized_embedding, TestFixture};

// ============================================================================
// Test Utilities
// ============================================================================

/// Create a test DualWriteAdapter backed by in-memory SQLite.
async fn create_test_adapter() -> Result<Arc<DualWriteAdapter>> {
    Ok(Arc::new(DualWriteAdapter::new_for_testing()?))
}

/// Create a PatternStorage for testing.
async fn create_test_storage() -> Result<Arc<PatternStorage>> {
    let adapter = create_test_adapter().await?;
    Ok(Arc::new(
        PatternStorage::new(adapter, StorageConfig::default()).await?,
    ))
}

/// Create a test session table in SQLite for session lifecycle tests.
async fn setup_session_table(db: &SqliteDb) -> Result<()> {
    db.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at TEXT,
            tokens_used INTEGER DEFAULT 0,
            patterns_learned INTEGER DEFAULT 0,
            patterns_retrieved INTEGER DEFAULT 0,
            domain TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_domain ON sessions(domain);
        "#,
    )
    .await
}

// ============================================================================
// Test 1: Satisfaction Metrics Flow
// ============================================================================

#[tokio::test]
async fn test_satisfaction_metrics_flow() {
    let storage = create_test_storage().await.expect("create storage");

    // Create and store a pattern
    let pattern = Pattern::builder()
        .problem("How to handle database connection timeouts")
        .solution("Implement retry with exponential backoff and circuit breaker pattern")
        .category(PatternCategory::Resilience)
        .build();

    let pattern_id = pattern.id().clone();
    storage.store_pattern(&pattern).await.expect("store pattern");

    // Retrieve the pattern
    let retrieved = storage
        .get_pattern(&pattern_id)
        .await
        .expect("get pattern")
        .expect("pattern should exist");

    // Verify initial satisfaction score defaults
    assert!(
        (retrieved.satisfaction_score() - 0.5).abs() < 0.01,
        "Initial satisfaction score should be ~0.5, got {}",
        retrieved.satisfaction_score()
    );
    assert_eq!(
        retrieved.satisfaction_trials(),
        0,
        "Initial satisfaction trials should be 0"
    );

    // Record positive satisfaction outcomes
    // Note: record_satisfaction is a method on Pattern, not storage
    // So we need to get the pattern, mutate it, and update it
    let mut pat = storage
        .get_pattern(&pattern_id)
        .await
        .expect("get pattern")
        .expect("pattern exists");
    pat.record_satisfaction(true);
    storage.update_pattern(&pat).await.expect("update pattern 1");

    let mut pat = storage
        .get_pattern(&pattern_id)
        .await
        .expect("get pattern")
        .expect("pattern exists");
    pat.record_satisfaction(true);
    storage.update_pattern(&pat).await.expect("update pattern 2");

    // Record one negative outcome
    let mut pat = storage
        .get_pattern(&pattern_id)
        .await
        .expect("get pattern")
        .expect("pattern exists");
    pat.record_satisfaction(false);
    storage.update_pattern(&pat).await.expect("update pattern 3");

    // Verify updated satisfaction metrics
    let updated = storage
        .get_pattern(&pattern_id)
        .await
        .expect("get updated pattern")
        .expect("pattern should exist");

    // Satisfaction should be ~0.67 (2/3 success)
    // The exact formula may include Bayesian smoothing, so check approximate value
    let expected_min = 0.5; // At minimum, should be above random
    let expected_max = 0.9; // Shouldn't be perfect with 1 failure
    assert!(
        updated.satisfaction_score() >= expected_min && updated.satisfaction_score() <= expected_max,
        "Satisfaction score should be in [{}, {}], got {}",
        expected_min,
        expected_max,
        updated.satisfaction_score()
    );
    assert_eq!(
        updated.satisfaction_trials(),
        3,
        "Should have 3 satisfaction trials"
    );

    println!("=== Satisfaction Metrics Test ===");
    println!("  Initial score: 0.5");
    println!(
        "  After 2 success + 1 failure: {:.2}",
        updated.satisfaction_score()
    );
    println!("  Trials: {}", updated.satisfaction_trials());
}

// ============================================================================
// Test 2: Trajectory Chain Analysis
// ============================================================================

#[tokio::test]
async fn test_trajectory_chain_analysis() {
    // Create a trajectory with multiple steps
    let trajectory = TrajectoryBuilder::new()
        .session_id("test-session-chain")
        .query("How to optimize database queries?")
        .add_step(TrajectoryStep::pattern_retrieval(
            vec![PatternId::from_string("pat_1"), PatternId::from_string("pat_2")],
            "database optimization",
            0.85,
        ))
        .add_step(TrajectoryStep::decision(
            vec![PatternId::from_string("pat_1")],
            "selected pattern with indexing solution",
            0.9,
        ))
        .add_step(TrajectoryStep::pattern_application(
            PatternId::from_string("pat_1"),
            "Applied indexing strategy successfully",
            0.95,
        ))
        .outcome(Outcome::Success, 0.92)
        .build();

    // Verify trajectory structure
    assert_eq!(trajectory.step_count(), 3, "Should have 3 steps");
    assert!(trajectory.is_complete(), "Trajectory should be complete");
    assert!(trajectory.success, "Trajectory should be marked successful");

    // Verify pattern chain
    let pattern_ids = trajectory.all_pattern_ids();
    assert!(
        pattern_ids.len() >= 1,
        "Should have at least one pattern ID"
    );

    // Verify step types
    let retrieval_steps = trajectory.steps_by_type(nagual::learning::trajectory::StepType::PatternRetrieval);
    assert_eq!(
        retrieval_steps.len(),
        1,
        "Should have 1 pattern retrieval step"
    );

    let decision_steps = trajectory.steps_by_type(nagual::learning::trajectory::StepType::Decision);
    assert_eq!(decision_steps.len(), 1, "Should have 1 decision step");

    let application_steps =
        trajectory.steps_by_type(nagual::learning::trajectory::StepType::PatternApplication);
    assert_eq!(
        application_steps.len(),
        1,
        "Should have 1 application step"
    );

    // Verify average confidence
    let avg_confidence = trajectory.average_confidence();
    assert!(
        avg_confidence > 0.8,
        "Average confidence should be > 0.8, got {}",
        avg_confidence
    );

    println!("=== Trajectory Chain Analysis Test ===");
    println!("  Steps: {}", trajectory.step_count());
    println!("  Patterns: {:?}", pattern_ids.len());
    println!("  Avg confidence: {:.2}", avg_confidence);
    println!("  Reward: {:.2}", trajectory.total_reward);
}

// ============================================================================
// Test 3: Dedup Workflow
// ============================================================================

#[tokio::test]
async fn test_dedup_workflow() {
    let storage = create_test_storage().await.expect("create storage");

    // Create exact duplicate patterns (same problem + solution)
    let mut p1 = Pattern::builder()
        .problem("How to handle null pointer exceptions")
        .solution("Use Option/Optional types and proper null checking")
        .category(PatternCategory::Resilience)
        .reward(0.9)
        .build();
    p1.compute_content_hash();

    let mut p2 = Pattern::builder()
        .problem("How to handle null pointer exceptions")
        .solution("Use Option/Optional types and proper null checking")
        .category(PatternCategory::Resilience)
        .reward(0.5)
        .build();
    p2.compute_content_hash();

    // Create a distinct pattern
    let mut p3 = Pattern::builder()
        .problem("How to implement caching")
        .solution("Use Redis with TTL-based expiration")
        .category(PatternCategory::Performance)
        .build();
    p3.compute_content_hash();

    // Store all patterns
    storage.store_pattern(&p1).await.expect("store p1");
    storage.store_pattern(&p2).await.expect("store p2");
    storage.store_pattern(&p3).await.expect("store p3");

    // Verify count
    let initial_count = storage.count().await.expect("count");
    assert_eq!(initial_count, 3, "Should have 3 patterns initially");

    // Run dedup scan (dry run)
    let config = DedupConfig::default();
    let result = scan_duplicates(&storage, &config).await.expect("scan");

    // Verify dedup detection
    assert_eq!(result.total_patterns, 3, "Should scan 3 patterns");
    assert!(result.dry_run, "Should be a dry run");

    // Should find 1 exact duplicate group
    assert!(
        !result.exact_duplicates.is_empty() || result.duplicate_count >= 1,
        "Should find at least 1 duplicate (exact group or near-duplicate)"
    );

    println!("=== Dedup Workflow Test ===");
    println!("  Total patterns: {}", result.total_patterns);
    println!("  Exact duplicate groups: {}", result.exact_duplicates.len());
    println!("  Near-duplicate groups: {}", result.near_duplicates.len());
    println!(
        "  Total duplicate patterns: {}",
        result.duplicate_count
    );
    println!("  Duration: {}ms", result.duration_ms);
}

// ============================================================================
// Test 4: Pyramid Generation
// ============================================================================

#[test]
fn test_pyramid_generation() {
    // Test title generation
    let problem = "How to implement a rate limiter for API endpoints using the token bucket algorithm to prevent abuse";
    let title = generate_title(problem);

    // Should be first 10 words
    let word_count = title.split_whitespace().count();
    assert!(
        word_count <= 10,
        "Title should have <= 10 words, got {}",
        word_count
    );
    assert!(
        title.starts_with("How to implement"),
        "Title should preserve beginning"
    );

    // Test summary generation
    let solution = "Implement a token bucket with these components:\n\n1. A bucket that holds tokens\n2. A refill rate (tokens per second)\n3. A maximum capacity\n\nWhen a request comes in, check if bucket has tokens. If yes, decrement and allow. If no, reject with 429 status.";
    let summary = generate_summary(solution);

    // Summary should be first paragraph or ~50 words
    let summary_words = summary.split_whitespace().count();
    assert!(
        summary_words <= 55,
        "Summary should have <= ~50 words, got {}",
        summary_words
    );

    // Test empty solution handling
    let empty_summary = generate_summary("");
    assert_eq!(
        empty_summary, "(No solution provided)",
        "Empty solution should return placeholder"
    );

    // Test long solution truncation
    let long_solution = "word ".repeat(100);
    let truncated = generate_summary(&long_solution);
    assert!(
        truncated.ends_with("..."),
        "Long solution should be truncated with ellipsis"
    );
    let truncated_words = truncated.trim_end_matches("...").split_whitespace().count();
    assert_eq!(truncated_words, 50, "Should truncate to 50 words");

    println!("=== Pyramid Generation Test ===");
    println!("  Title ({} words): {}", word_count, title);
    println!("  Summary ({} words): {}", summary_words, &summary[..50.min(summary.len())]);
}

#[tokio::test]
async fn test_pyramid_stats_and_generation() {
    let adapter = create_test_adapter().await.expect("create adapter");

    // Initialize the database with patterns
    adapter.sqlite().execute_batch(r#"
        CREATE TABLE IF NOT EXISTS reasoning_patterns (
            id TEXT PRIMARY KEY,
            problem TEXT NOT NULL,
            solution TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            title TEXT,
            summary TEXT
        );

        INSERT INTO reasoning_patterns (id, problem, solution)
        VALUES
            ('pat_1', 'Problem one with a long description that needs summarizing', 'Solution one with detailed steps'),
            ('pat_2', 'Problem two about async handling', 'Use tokio runtime and async/await'),
            ('pat_3', 'Problem three', 'Solution three');
    "#).await.expect("init db");

    // Get initial stats
    let stats = get_pyramid_stats(&adapter).await.expect("get stats");
    assert_eq!(stats.total_patterns, 3, "Should have 3 patterns");
    assert_eq!(
        stats.without_pyramid, 3,
        "All patterns should be without pyramid initially"
    );

    // Generate pyramids (dry run)
    let dry_result = generate_missing_pyramids(&adapter, true, None)
        .await
        .expect("dry run");
    assert_eq!(
        dry_result.generated, 3,
        "Dry run should identify 3 patterns to generate"
    );

    // Generate pyramids (for real)
    let result = generate_missing_pyramids(&adapter, false, None)
        .await
        .expect("generate");
    assert_eq!(result.generated, 3, "Should generate 3 pyramids");

    // Verify all have pyramids now
    let final_stats = get_pyramid_stats(&adapter).await.expect("final stats");
    assert_eq!(
        final_stats.with_pyramid, 3,
        "All patterns should have pyramids"
    );
    assert_eq!(
        final_stats.without_pyramid, 0,
        "No patterns should be without pyramid"
    );

    println!("=== Pyramid Stats & Generation Test ===");
    println!("  Patterns: {}", stats.total_patterns);
    println!("  Generated: {}", result.generated);
    println!(
        "  Coverage: {:.1}%",
        final_stats.coverage_percent()
    );
}

// ============================================================================
// Test 5: Session Lifecycle
// ============================================================================

#[tokio::test]
async fn test_session_lifecycle() {
    use nagual::db::sessions::SessionManager;

    // Create test database
    let db = Arc::new(SqliteDb::open_in_memory().expect("open in-memory db"));

    // Setup session table
    setup_session_table(&db).await.expect("setup session table");

    let manager = SessionManager::new(db);

    // Start a session
    let session = manager
        .start_session(Some("rust"))
        .await
        .expect("start session");

    assert!(!session.id.is_empty(), "Session should have an ID");
    assert!(session.is_active(), "Session should be active");
    assert_eq!(session.tokens_used, 0, "Initial tokens should be 0");
    assert_eq!(
        session.patterns_learned, 0,
        "Initial patterns_learned should be 0"
    );
    assert_eq!(
        session.domain,
        Some("rust".to_string()),
        "Domain should be 'rust'"
    );

    // Record activity
    manager
        .record_tokens(&session.id, 5000)
        .await
        .expect("record tokens");
    manager
        .record_pattern_learned(&session.id)
        .await
        .expect("record pattern 1");
    manager
        .record_pattern_learned(&session.id)
        .await
        .expect("record pattern 2");
    manager
        .record_pattern_retrieved(&session.id)
        .await
        .expect("record retrieval");

    // End session
    manager
        .end_session(&session.id)
        .await
        .expect("end session");

    // Verify session state
    let ended = manager
        .get_session(&session.id)
        .await
        .expect("get session")
        .expect("session should exist");

    assert!(!ended.is_active(), "Session should no longer be active");
    assert_eq!(ended.tokens_used, 5000, "Should have 5000 tokens");
    assert_eq!(ended.patterns_learned, 2, "Should have 2 patterns learned");
    assert_eq!(
        ended.patterns_retrieved, 1,
        "Should have 1 pattern retrieved"
    );

    // Check efficiency
    let efficiency = ended.efficiency();
    assert!(
        efficiency > 0.0,
        "Efficiency should be positive, got {}",
        efficiency
    );
    // 2 patterns / 5K tokens = 0.4 patterns/K tokens
    assert!(
        (efficiency - 0.4).abs() < 0.01,
        "Efficiency should be ~0.4, got {}",
        efficiency
    );

    // Get stats
    let stats = manager.get_stats().await.expect("get stats");
    assert_eq!(stats.total_sessions, 1, "Should have 1 session");
    assert_eq!(stats.total_tokens, 5000, "Total tokens should be 5000");
    assert_eq!(
        stats.total_patterns_learned, 2,
        "Total patterns learned should be 2"
    );
    assert!(stats.efficiency > 0.0, "Stats efficiency should be positive");

    println!("=== Session Lifecycle Test ===");
    println!("  Session ID: {}", session.id);
    println!("  Tokens used: {}", ended.tokens_used);
    println!("  Patterns learned: {}", ended.patterns_learned);
    println!("  Efficiency: {:.2} patterns/K tokens", efficiency);
}

// ============================================================================
// Test 6: Scenario Evaluation
// ============================================================================

#[tokio::test]
async fn test_scenario_evaluation() {
    let adapter = create_test_adapter().await.expect("create adapter");
    let scenario_storage = ScenarioStorage::new(&adapter);

    // Initialize schema
    scenario_storage
        .init_schema()
        .await
        .expect("init scenario schema");

    // Create a holdout scenario
    let scenario = Scenario::new("rust.error_handling")
        .with_description("Handle async errors gracefully")
        .with_input_context("Async function that can fail with multiple error types")
        .with_expected_behavior(
            "Should use Result type, handle errors with descriptive messages, avoid panics",
        )
        .with_difficulty(Difficulty::Medium)
        .as_holdout(true)
        .with_tag("async")
        .with_tag("error")
        .build();

    scenario_storage
        .create_scenario(&scenario)
        .await
        .expect("create scenario");

    // Create a pattern to evaluate
    let pattern = Pattern::builder()
        .problem("How to handle async errors in Rust")
        .solution(
            "Use Result<T, E> for fallible operations. Handle errors with match or the ? operator. \
             Provide descriptive error messages using thiserror or anyhow crates. \
             Never panic in library code - return errors instead.",
        )
        .category(PatternCategory::Custom("rust.error_handling".to_string()))
        .build();

    // Evaluate pattern against scenario
    let evaluator = ScenarioEvaluator::new();
    let eval = evaluator.evaluate(&pattern, &scenario);

    // Verify evaluation
    assert!(
        eval.score > 0.3,
        "Score should be above 0.3 due to keyword matches, got {}",
        eval.score
    );
    assert!(
        !eval.pattern_id.is_empty(),
        "Evaluation should have pattern ID"
    );
    assert_eq!(
        eval.scenario_id.as_str(),
        scenario.id.as_str(),
        "Scenario ID should match"
    );

    // Record the evaluation
    scenario_storage
        .record_evaluation(&eval)
        .await
        .expect("record evaluation");

    // Retrieve updated scenario to check counts
    let updated = scenario_storage
        .get_scenario(&scenario.id)
        .await
        .expect("get updated scenario")
        .expect("scenario should exist");

    // Verify pass/fail counts updated
    let expected_pass_count = if eval.passed { 1 } else { 0 };
    let expected_fail_count = if eval.passed { 0 } else { 1 };
    assert_eq!(
        updated.pass_count, expected_pass_count,
        "Pass count should match evaluation"
    );
    assert_eq!(
        updated.fail_count, expected_fail_count,
        "Fail count should match evaluation"
    );

    // Get holdout stats
    let stats = scenario_storage.get_stats().await.expect("get scenario stats");
    assert_eq!(
        stats.holdout_scenarios, 1,
        "Should have 1 holdout scenario"
    );
    assert_eq!(
        stats.total_evaluations, 1,
        "Should have 1 total evaluation"
    );

    println!("=== Scenario Evaluation Test ===");
    println!("  Scenario: {}", scenario.description);
    println!("  Difficulty: {}", scenario.difficulty);
    println!("  Evaluation score: {:.2}", eval.score);
    println!("  Passed: {}", eval.passed);
    println!("  Feedback: {:?}", eval.feedback);
}

// ============================================================================
// Test 7: Gene Transfusion
// ============================================================================

#[test]
fn test_gene_transfusion() {
    use nagual::reasoning_bank::transfusion::{
        RustAsyncDetector, RustErrorHandlingDetector, TransfusionConfig, Transfuser,
    };
    use std::fs;

    let dir = TempDir::new().expect("create temp dir");

    // Create a subdirectory with a non-dot name to avoid hidden directory skip
    let src_dir = dir.path().join("src");
    fs::create_dir(&src_dir).expect("create src dir");

    // Create test Rust file with recognizable patterns
    let test_file = src_dir.join("test_patterns.rs");
    fs::write(
        &test_file,
        r#"
use thiserror::Error;
use tokio::sync::mpsc;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Database error: {0}")]
    Database(String),
}

async fn process_data() -> Result<(), MyError> {
    tokio::spawn(async move {
        do_async_work().await;
    });

    let (tx, mut rx) = mpsc::channel(100);

    tokio::select! {
        result = operation_a() => handle_a(result),
        result = operation_b() => handle_b(result),
    }

    Ok(())
}

#[tokio::test]
async fn test_async_process() {
    let result = process_data().await;
    assert!(result.is_ok());
}
"#,
    )
    .expect("write test file");

    // Configure transfuser
    let config = TransfusionConfig {
        min_confidence: 0.7,
        dry_run: true,
        max_files: 10,
        ..Default::default()
    };

    // Debug: verify file was created
    assert!(test_file.exists(), "Test file should exist at {:?}", test_file);
    let content = fs::read_to_string(&test_file).expect("read file");
    assert!(!content.is_empty(), "Test file should have content");

    // Transfuse from the src subdirectory (not the hidden temp root)
    let transfuser = Transfuser::new(config);
    let result = transfuser.transfuse(&src_dir).expect("run transfusion");

    // Verify extraction
    assert!(
        result.files_scanned >= 1,
        "Should scan at least 1 file, got {}",
        result.files_scanned
    );
    assert!(
        result.patterns_extracted >= 1,
        "Should extract at least 1 pattern, got {}",
        result.patterns_extracted
    );

    // Check categories detected
    let has_error_handling = result.by_category.contains_key("rust.error_handling");
    let has_async = result.by_category.contains_key("rust.async");
    let has_testing = result.by_category.contains_key("rust.testing");

    println!("=== Gene Transfusion Test ===");
    println!("  Files scanned: {}", result.files_scanned);
    println!("  Patterns extracted: {}", result.patterns_extracted);
    println!("  By category: {:?}", result.by_category);
    println!("  By detector: {:?}", result.by_detector);
    println!(
        "  Detections: error_handling={}, async={}, testing={}",
        has_error_handling, has_async, has_testing
    );

    // At least one category should be detected
    assert!(
        has_error_handling || has_async || has_testing,
        "Should detect at least one pattern category"
    );
}

// ============================================================================
// Test: Detector-specific pattern extraction
// ============================================================================

#[test]
fn test_rust_error_handling_detector() {
    use nagual::reasoning_bank::transfusion::RustErrorHandlingDetector;
    use nagual::reasoning_bank::transfusion::PatternDetector;

    let detector = RustErrorHandlingDetector::new();
    let content = r#"
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Query failed: {0}")]
    QueryFailed(String),
}

impl std::error::Error for CustomError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
"#;

    let patterns = detector.detect(content, "test.rs");

    assert!(
        !patterns.is_empty(),
        "Should detect error handling patterns"
    );

    // Check that thiserror was detected
    let has_thiserror = patterns
        .iter()
        .any(|p| p.tags.contains(&"thiserror".to_string()));
    assert!(has_thiserror, "Should detect thiserror pattern");

    // Verify domain
    for pattern in &patterns {
        assert_eq!(
            pattern.domain, "rust.error_handling",
            "Domain should be rust.error_handling"
        );
    }

    println!("=== Error Handling Detector Test ===");
    println!("  Patterns found: {}", patterns.len());
    for p in &patterns {
        println!("    - {} (conf: {:.2})", p.problem, p.confidence);
    }
}

#[test]
fn test_rust_async_detector() {
    use nagual::reasoning_bank::transfusion::PatternDetector;
    use nagual::reasoning_bank::transfusion::RustAsyncDetector;

    let detector = RustAsyncDetector::new();
    let content = r#"
async fn main() {
    let handle = tokio::spawn(async {
        process().await
    });

    tokio::select! {
        v = async_op_1() => handle_1(v),
        v = async_op_2() => handle_2(v),
    }

    let (tx, rx) = mpsc::channel(100);
}
"#;

    let patterns = detector.detect(content, "test.rs");

    assert!(
        !patterns.is_empty(),
        "Should detect async patterns"
    );

    // Check for spawn pattern
    let has_spawn = patterns.iter().any(|p| p.tags.contains(&"spawn".to_string()));
    // Check for select pattern
    let has_select = patterns.iter().any(|p| p.tags.contains(&"select".to_string()));
    // Check for channel pattern
    let has_channel = patterns.iter().any(|p| p.tags.contains(&"channel".to_string()));

    println!("=== Async Detector Test ===");
    println!("  Patterns found: {}", patterns.len());
    println!(
        "  spawn={}, select={}, channel={}",
        has_spawn, has_select, has_channel
    );

    assert!(
        has_spawn || has_select || has_channel,
        "Should detect at least one async pattern type"
    );
}

// ============================================================================
// Integration Test: Full Learning Workflow
// ============================================================================

#[tokio::test]
async fn test_full_learning_workflow() {
    let storage = create_test_storage().await.expect("create storage");

    // 1. Store a pattern
    let pattern = Pattern::builder()
        .problem("How to implement connection pooling for PostgreSQL")
        .solution(
            "Use SQLx's PgPool with proper configuration. Set max_connections based on expected load. \
             Implement health checks and automatic reconnection. Use timeouts to prevent hanging connections.",
        )
        .category(PatternCategory::Performance)
        .confidence(0.8)
        .build();

    let pattern_id = pattern.id().clone();
    storage.store_pattern(&pattern).await.expect("store pattern");

    // 2. Record successful reuse
    let mut updated_pattern = storage
        .get_pattern(&pattern_id)
        .await
        .expect("get")
        .expect("exists");
    updated_pattern.increment_reuse_count();
    updated_pattern.set_reward(0.9);
    storage
        .update_pattern(&updated_pattern)
        .await
        .expect("update");

    // 3. Record satisfaction
    let mut pat = storage
        .get_pattern(&pattern_id)
        .await
        .expect("get pattern")
        .expect("pattern exists");
    pat.record_satisfaction(true);
    storage.update_pattern(&pat).await.expect("update satisfaction");

    // 4. Verify final state
    let final_pattern = storage
        .get_pattern(&pattern_id)
        .await
        .expect("get final")
        .expect("exists");

    assert!(
        final_pattern.reward() > 0.5,
        "Reward should be elevated"
    );
    assert_eq!(
        final_pattern.reuse_count(),
        1,
        "Reuse count should be 1"
    );
    assert_eq!(
        final_pattern.satisfaction_trials(),
        1,
        "Should have 1 satisfaction trial"
    );

    // 5. Test search (fts_search requires query and limit)
    let results = storage
        .fts_search("connection pooling", 10)
        .await
        .expect("fts_search");
    assert!(
        !results.is_empty(),
        "Search should find the pattern"
    );

    println!("=== Full Learning Workflow Test ===");
    println!("  Pattern ID: {}", pattern_id);
    println!("  Final reward: {:.2}", final_pattern.reward());
    println!("  Reuse count: {}", final_pattern.reuse_count());
    println!("  Search results: {}", results.len());
}
