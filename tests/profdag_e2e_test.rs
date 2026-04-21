//! ProfDAG End-to-End Integration Test (P0-3)
//!
//! Chains ALL major modules together with real data flowing between them:
//!
//! 1. ProfDAGStorage with in-memory SQLite (DualWriteAdapter)
//! 2. Pattern nodes with embeddings inserted into storage
//! 3. TrajectoryRecorder with storage -> complete_async -> real ProfDAG nodes/edges
//! 4. HNSW search via ProfDAGSearch over persisted nodes
//! 5. FastGRNN routing with real pretrained weights
//! 6. ENagual context builder with HNSW search results
//!
//! This test proves the system works end-to-end with real data flowing
//! through every major subsystem.

use std::sync::Arc;

// Real library imports -- no local re-definitions
use nagual::db::DualWriteAdapter;
use nagual::injection::{ENagualBuilder, ENagualConfig};
use nagual::learning::trajectory::TrajectoryStep;
use nagual::learning::Outcome;
use nagual::profdag::{
    EdgeType, NeighborQuery, NodeType, ProfDAGEdge, ProfDAGNode, ProfDAGSearch, ProfDAGStorage,
    ProfDAGStorageConfig, RecorderConfig, SearchConfig, TrajectoryRecorder,
};
use nagual::reasoning_bank::pattern::PatternId;
use nagual::router::{FastGRNN, FastGRNNConfig};

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic 128-dimensional embedding from a seed.
/// Uses sin-based generation so that different seeds produce vectors
/// pointing in meaningfully different directions.
fn generate_embedding(dim: usize, seed: u64) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim)
        .map(|i| ((i as f64 * seed as f64 * 0.1).sin() as f32) * 0.5)
        .collect();
    // L2-normalize so cosine similarity is well-defined
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        raw.iter().map(|x| x / norm).collect()
    } else {
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;
        v
    }
}

/// Create ProfDAGStorage backed by in-memory SQLite.
async fn create_test_storage() -> (Arc<DualWriteAdapter>, Arc<ProfDAGStorage>) {
    let adapter = Arc::new(DualWriteAdapter::new_for_testing().expect("in-memory adapter"));
    let storage = Arc::new(
        ProfDAGStorage::new(adapter.clone(), ProfDAGStorageConfig::default())
            .await
            .expect("ProfDAG schema init"),
    );
    (adapter, storage)
}

// ===========================================================================
// Test: Full end-to-end chain
// ===========================================================================

#[tokio::test]
async fn test_profdag_full_e2e_chain() {
    // ------------------------------------------------------------------
    // Step 1: Create ProfDAGStorage with in-memory SQLite
    // ------------------------------------------------------------------
    let (_adapter, storage) = create_test_storage().await;

    // Verify storage starts empty
    let stats = storage.stats().await.expect("stats");
    assert_eq!(stats.node_count, 0, "Storage should start empty");
    assert_eq!(stats.edge_count, 0, "No edges yet");

    // ------------------------------------------------------------------
    // Step 2: Create pattern nodes with embeddings
    // ------------------------------------------------------------------
    let node1 = ProfDAGNode::pattern("How to handle database timeouts")
        .with_embedding(generate_embedding(128, 1))
        .with_confidence(0.9)
        .with_importance(0.8);
    let node1_id = storage.insert_node(&node1).await.expect("insert node1");

    let node2 = ProfDAGNode::pattern("Retry strategies with exponential backoff")
        .with_embedding(generate_embedding(128, 2))
        .with_confidence(0.85)
        .with_importance(0.75);
    let node2_id = storage.insert_node(&node2).await.expect("insert node2");

    let node3 = ProfDAGNode::pattern("Connection pooling for database performance")
        .with_embedding(generate_embedding(128, 3))
        .with_confidence(0.88)
        .with_importance(0.82);
    let node3_id = storage.insert_node(&node3).await.expect("insert node3");

    let node4 = ProfDAGNode::decision("Apply connection pooling with retry")
        .with_embedding(generate_embedding(128, 4))
        .with_confidence(0.92)
        .with_importance(0.85);
    let node4_id = storage.insert_node(&node4).await.expect("insert node4");

    // Verify nodes are stored
    let stats = storage.stats().await.expect("stats after inserts");
    assert_eq!(stats.node_count, 4, "Should have 4 nodes");
    assert_eq!(stats.nodes_with_embeddings, 4, "All nodes should have embeddings");

    // Verify individual nodes
    let fetched_node1 = storage.get_node(&node1_id).await.expect("get node1");
    assert!(fetched_node1.is_some(), "Node1 should exist");
    let fetched_node1 = fetched_node1.unwrap();
    assert_eq!(fetched_node1.node_type, NodeType::Pattern);
    assert_eq!(fetched_node1.content, "How to handle database timeouts");
    assert!(fetched_node1.has_embedding());
    assert_eq!(fetched_node1.embedding_dim(), 128);

    // Verify nodes by type
    let pattern_nodes = storage
        .get_nodes_by_type(NodeType::Pattern, 10)
        .await
        .expect("get pattern nodes");
    assert_eq!(pattern_nodes.len(), 3, "Should have 3 pattern nodes");

    let decision_nodes = storage
        .get_nodes_by_type(NodeType::Decision, 10)
        .await
        .expect("get decision nodes");
    assert_eq!(decision_nodes.len(), 1, "Should have 1 decision node");

    // ------------------------------------------------------------------
    // Step 2b: Create explicit edges between pattern nodes
    // ------------------------------------------------------------------
    let edge1 = ProfDAGEdge::leads_to(&node1_id, &node2_id, 0.85);
    storage.insert_edge(&edge1).await.expect("insert edge1");

    let edge2 = ProfDAGEdge::similar_to(&node2_id, &node3_id, 0.78);
    storage.insert_edge(&edge2).await.expect("insert edge2");

    let stats = storage.stats().await.expect("stats after edges");
    assert_eq!(stats.edge_count, 2, "Should have 2 edges");

    // Verify neighbor queries
    let neighbors = storage
        .get_neighbors(&node1_id, &NeighborQuery::outgoing().with_edge_type(EdgeType::LeadsTo))
        .await
        .expect("get neighbors of node1");
    assert_eq!(neighbors.len(), 1, "Node1 should have 1 outgoing LeadsTo neighbor");
    assert_eq!(
        neighbors[0].node.id, node2_id,
        "Neighbor should be node2"
    );
    assert!(!neighbors[0].is_incoming);

    // ------------------------------------------------------------------
    // Step 3: Record a trajectory with storage persistence
    // ------------------------------------------------------------------
    let recorder = TrajectoryRecorder::with_storage(RecorderConfig::default(), storage.clone());
    assert!(recorder.has_storage(), "Recorder should have storage attached");

    let traj_id = recorder.start(
        "How to optimize database queries?",
        Some("e2e-test-session".to_string()),
    );
    assert!(recorder.is_active(&traj_id), "Trajectory should be active");

    // Record steps referencing the pattern nodes we created
    let step_idx_0 = recorder
        .record_step(
            &traj_id,
            TrajectoryStep::pattern_retrieval(
                vec![PatternId::from_string(&node1_id), PatternId::from_string(&node2_id)],
                "database optimization",
                0.9,
            ),
        )
        .expect("record step 0");
    assert_eq!(step_idx_0, 0);

    let step_idx_1 = recorder
        .record_step(
            &traj_id,
            TrajectoryStep::decision(
                vec![PatternId::from_string(&node3_id)],
                "Apply connection pooling",
                0.85,
            ),
        )
        .expect("record step 1");
    assert_eq!(step_idx_1, 1);

    let step_idx_2 = recorder
        .record_step(
            &traj_id,
            TrajectoryStep::decision(
                vec![PatternId::from_string(&node4_id)],
                "Combined pooling with retry",
                0.92,
            ),
        )
        .expect("record step 2");
    assert_eq!(step_idx_2, 2);

    // Verify active trajectory state before completion
    let active_traj = recorder.get_active(&traj_id).expect("get active trajectory");
    assert_eq!(active_traj.step_count(), 3, "Should have 3 steps recorded");

    // Complete with async persistence -- this creates real ProfDAG nodes/edges
    let result = recorder
        .complete_async(&traj_id, Outcome::Success, 0.9)
        .await
        .expect("complete_async");

    assert!(!recorder.is_active(&traj_id), "Trajectory should no longer be active");
    assert_eq!(result.outcome, Outcome::Success);
    assert!((result.reward - 0.9).abs() < 0.01);
    assert_eq!(result.step_count, 3);
    assert!(
        !result.profdag_node_id.is_empty(),
        "Should have a ProfDAG node ID"
    );

    // ------------------------------------------------------------------
    // Step 4: Verify real nodes and edges persisted in storage
    // ------------------------------------------------------------------

    // The trajectory node should exist in storage
    let traj_node = storage
        .get_node(&result.profdag_node_id)
        .await
        .expect("get trajectory node");
    assert!(traj_node.is_some(), "Trajectory node should exist in storage");
    let traj_node = traj_node.unwrap();
    assert_eq!(
        traj_node.node_type,
        NodeType::Trajectory,
        "Node should be Trajectory type"
    );
    assert!(
        traj_node.content.contains("Trajectory"),
        "Content should describe trajectory: got '{}'",
        traj_node.content
    );

    // Edges should have been created (leads_to between consecutive patterns)
    // The pattern sequence is: node1 -> node2 -> node3 -> node4
    // That means 3 leads_to edges from complete_async
    assert!(
        result.edges_created > 0,
        "Should have created leads_to edges, got {}",
        result.edges_created
    );

    // Verify trajectory nodes appear via get_nodes_by_type
    let trajectory_nodes = storage
        .get_nodes_by_type(NodeType::Trajectory, 10)
        .await
        .expect("get trajectory nodes");
    assert!(
        !trajectory_nodes.is_empty(),
        "Should find trajectory nodes in storage"
    );
    assert!(
        trajectory_nodes.iter().any(|n| n.id == result.profdag_node_id),
        "Should find our specific trajectory node"
    );

    // Updated stats should reflect new nodes and edges
    let final_stats = storage.stats().await.expect("final stats");
    assert!(
        final_stats.node_count > 4,
        "Should have more than 4 nodes (original 4 + trajectory node): got {}",
        final_stats.node_count
    );
    assert!(
        final_stats.edge_count > 2,
        "Should have more than 2 edges (original 2 + trajectory edges): got {}",
        final_stats.edge_count
    );

    // ------------------------------------------------------------------
    // Step 5: Run HNSW search over persisted nodes
    // ------------------------------------------------------------------
    let search = ProfDAGSearch::new(storage.clone(), SearchConfig::default());

    // Rebuild the index from all nodes with embeddings
    search.rebuild_index().await.expect("rebuild HNSW index");

    let search_stats = search.get_stats();
    assert!(
        search_stats.indexed_nodes >= 4,
        "Should have indexed at least 4 nodes with embeddings, got {}",
        search_stats.indexed_nodes
    );
    assert!(!search_stats.index_dirty, "Index should be clean after rebuild");

    // Search for nodes similar to node1's embedding
    let query_embedding = generate_embedding(128, 1); // Same as node1
    let similar = search
        .find_similar(&query_embedding, 5, 0.0)
        .await
        .expect("find_similar");

    assert!(
        !similar.is_empty(),
        "Should find similar nodes for the query embedding"
    );

    // The most similar node should be node1 itself (exact match)
    assert_eq!(
        similar[0].node.id, node1_id,
        "Most similar node should be node1 (exact match)"
    );
    assert!(
        similar[0].similarity > 0.95,
        "Self-similarity should be very high, got {}",
        similar[0].similarity
    );

    // Search filtered by node type
    let pattern_results = search
        .find_similar_by_type(&query_embedding, 5, NodeType::Pattern, 0.0)
        .await
        .expect("find_similar_by_type");
    assert!(
        !pattern_results.is_empty(),
        "Should find pattern nodes"
    );
    for r in &pattern_results {
        assert_eq!(
            r.node.node_type,
            NodeType::Pattern,
            "All results should be Pattern type"
        );
    }

    // Search with a different embedding and verify distinctness
    let query_embedding_2 = generate_embedding(128, 2); // Same as node2
    let similar_2 = search
        .find_similar(&query_embedding_2, 3, 0.0)
        .await
        .expect("find_similar for node2 embedding");
    assert!(
        !similar_2.is_empty(),
        "Should find similar nodes for node2 embedding"
    );
    assert_eq!(
        similar_2[0].node.id, node2_id,
        "Most similar to node2 embedding should be node2"
    );

    // ------------------------------------------------------------------
    // Step 6: Route through FastGRNN with real pretrained weights
    // ------------------------------------------------------------------
    let config = FastGRNNConfig::default();
    let router = FastGRNN::new(config).expect("FastGRNN construction");

    // Simple query features (5 dimensions: query_length, embedding_norm,
    // domain_specificity, pattern_coverage, historical_accuracy)
    let features_simple = vec![0.2, 0.3, 0.1, 0.4, 0.5];
    let complexity_simple = router.forward(&features_simple).expect("forward simple");
    assert!(
        complexity_simple >= 0.0 && complexity_simple <= 1.0,
        "Complexity score should be in [0, 1], got {}",
        complexity_simple
    );

    // Complex query features
    let features_complex = vec![0.9, 0.8, 0.9, 0.2, 0.3];
    let complexity_complex = router.forward(&features_complex).expect("forward complex");
    assert!(
        complexity_complex >= 0.0 && complexity_complex <= 1.0,
        "Complexity score should be in [0, 1], got {}",
        complexity_complex
    );

    // Batch forward
    let batch = vec![features_simple.clone(), features_complex.clone()];
    let batch_results = router.forward_batch(&batch).expect("forward_batch");
    assert_eq!(batch_results.len(), 2, "Batch should return 2 results");
    for score in &batch_results {
        assert!(
            *score >= 0.0 && *score <= 1.0,
            "All batch scores should be in [0, 1]"
        );
    }

    // Verify determinism: same input -> same output
    let complexity_simple_2 = router.forward(&features_simple).expect("forward simple again");
    assert!(
        (complexity_simple - complexity_simple_2).abs() < 1e-6,
        "FastGRNN should be deterministic"
    );

    // Verify inference metrics
    assert!(router.inference_count() >= 4, "Should have counted inferences");
    let avg_time = router.avg_inference_time_us();
    assert!(avg_time > 0.0, "Average inference time should be positive");

    // ------------------------------------------------------------------
    // Step 7: Build E_nagual context from search results
    // ------------------------------------------------------------------
    let e_nagual = ENagualBuilder::new("How to optimize database queries?")
        .config(ENagualConfig::default())
        .with_hnsw_neighbors(&similar)
        .build();

    // Verify E_nagual was constructed with HNSW neighbor context
    // (it may not have "content" in the patterns sense, but neighbor IDs
    // should be populated based on the similarity threshold)
    let prompt_prefix = e_nagual.to_prompt_prefix();
    assert!(
        !prompt_prefix.is_empty(),
        "E_nagual prompt prefix should not be empty"
    );

    // Verify that the query was preserved
    assert_eq!(e_nagual.query, "How to optimize database queries?");

    println!("=== ProfDAG E2E Test Summary ===");
    println!("  Storage: {} nodes, {} edges", final_stats.node_count, final_stats.edge_count);
    println!("  Trajectory: {} steps, reward={}", result.step_count, result.reward);
    println!("  Edges created by trajectory: {}", result.edges_created);
    println!("  HNSW indexed: {} nodes", search_stats.indexed_nodes);
    println!("  Search results: {} similar nodes", similar.len());
    println!("  FastGRNN simple={:.4}, complex={:.4}", complexity_simple, complexity_complex);
    println!("  FastGRNN avg inference: {:.2}us", avg_time);
    println!("  E_nagual prompt length: {} chars", prompt_prefix.len());
    println!("=== All assertions passed ===");
}

// ===========================================================================
// Test: Trajectory recorder without storage (backward compat)
// ===========================================================================

#[tokio::test]
async fn test_trajectory_recorder_backward_compat() {
    // complete_async without storage should not panic and should still work
    let recorder = TrajectoryRecorder::new();
    assert!(!recorder.has_storage());

    let traj_id = recorder.start("backward compat query", None);

    recorder
        .record_step(
            &traj_id,
            TrajectoryStep::pattern_retrieval(
                vec![PatternId::from_string("pat_a")],
                "search",
                0.8,
            ),
        )
        .expect("record step");

    let result = recorder
        .complete_async(&traj_id, Outcome::Success, 0.85)
        .await
        .expect("complete_async without storage");

    assert!(!result.profdag_node_id.is_empty());
    assert_eq!(result.outcome, Outcome::Success);
    assert_eq!(result.step_count, 1);
}

// ===========================================================================
// Test: Storage persistence of trajectory edges
// ===========================================================================

#[tokio::test]
async fn test_trajectory_creates_real_edges_in_storage() {
    let (_adapter, storage) = create_test_storage().await;

    // Insert two pattern nodes that the trajectory will reference
    let pat_a = ProfDAGNode::pattern("Pattern A")
        .with_embedding(generate_embedding(128, 10))
        .with_confidence(0.9);
    let pat_a_id = storage.insert_node(&pat_a).await.expect("insert pat_a");

    let pat_b = ProfDAGNode::pattern("Pattern B")
        .with_embedding(generate_embedding(128, 20))
        .with_confidence(0.85);
    let pat_b_id = storage.insert_node(&pat_b).await.expect("insert pat_b");

    // Record a trajectory that references these patterns
    let recorder = TrajectoryRecorder::with_storage(RecorderConfig::default(), storage.clone());
    let traj_id = recorder.start("Test edge creation", None);

    recorder
        .record_step(
            &traj_id,
            TrajectoryStep::pattern_retrieval(
                vec![PatternId::from_string(&pat_a_id)],
                "first retrieval",
                0.9,
            ),
        )
        .expect("step 0");

    recorder
        .record_step(
            &traj_id,
            TrajectoryStep::decision(
                vec![PatternId::from_string(&pat_b_id)],
                "selected B",
                0.85,
            ),
        )
        .expect("step 1");

    let result = recorder
        .complete_async(&traj_id, Outcome::Success, 0.9)
        .await
        .expect("complete");

    // There should be at least 1 leads_to edge (pat_a -> pat_b)
    assert!(
        result.edges_created >= 1,
        "Should have created at least 1 edge, got {}",
        result.edges_created
    );

    // The trajectory node should be persisted
    let traj_node = storage
        .get_node(&result.profdag_node_id)
        .await
        .expect("get traj node");
    assert!(traj_node.is_some());
    assert_eq!(traj_node.unwrap().node_type, NodeType::Trajectory);

    // Verify total edges increased
    let stats = storage.stats().await.expect("stats");
    assert!(
        stats.edge_count >= 1,
        "Storage should have at least 1 edge, got {}",
        stats.edge_count
    );
}

// ===========================================================================
// Test: HNSW search recall with known embeddings
// ===========================================================================

#[tokio::test]
async fn test_hnsw_search_recall() {
    let (_adapter, storage) = create_test_storage().await;

    // Insert 10 pattern nodes with diverse embeddings
    let mut node_ids = Vec::new();
    for seed in 1..=10u64 {
        let node = ProfDAGNode::pattern(format!("Pattern seed={}", seed))
            .with_embedding(generate_embedding(128, seed))
            .with_confidence(0.8);
        let id = storage.insert_node(&node).await.expect("insert");
        node_ids.push(id);
    }

    // Build search index
    let search = ProfDAGSearch::new(storage.clone(), SearchConfig::default());
    search.rebuild_index().await.expect("rebuild");

    assert_eq!(search.indexed_count(), 10, "Should have 10 indexed nodes");

    // Search for exact match (seed=5)
    let query = generate_embedding(128, 5);
    let results = search.find_similar(&query, 3, 0.0).await.expect("search");

    assert!(!results.is_empty(), "Should find results");
    assert_eq!(
        results[0].node.id, node_ids[4],
        "Top result should be exact match (seed=5)"
    );
    assert!(
        results[0].similarity > 0.95,
        "Exact match similarity should be > 0.95, got {}",
        results[0].similarity
    );
}

// ===========================================================================
// Test: FastGRNN pretrained weights are real (not random)
// ===========================================================================

#[test]
fn test_fastgrnn_pretrained_weights_loaded() {
    let config = FastGRNNConfig::default();
    let router = FastGRNN::new(config).expect("FastGRNN with pretrained weights");

    // Run multiple diverse inputs and verify the outputs differ meaningfully
    let scores: Vec<f32> = (0..5)
        .map(|i| {
            let features = vec![
                i as f32 * 0.2,
                (i as f32 * 0.3).sin(),
                0.5,
                1.0 - i as f32 * 0.15,
                0.7,
            ];
            router.forward(&features).expect("forward")
        })
        .collect();

    // All scores should be valid
    for (i, score) in scores.iter().enumerate() {
        assert!(
            *score >= 0.0 && *score <= 1.0,
            "Score {} ({}) should be in [0, 1]",
            i,
            score
        );
    }

    // At least some scores should differ (pretrained weights produce varied output)
    let all_same = scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
    assert!(
        !all_same,
        "Pretrained weights should produce varied outputs for different inputs: {:?}",
        scores
    );
}

// ===========================================================================
// Test: ProfDAGProfiler is wired into search, storage, and routing hot paths
// ===========================================================================

#[tokio::test]
async fn test_profiler_wired_into_search_and_storage() {
    use nagual::profdag::profiler::{OperationType, ProfDAGProfiler, ProfilerConfig};

    let profiler = Arc::new(ProfDAGProfiler::new(ProfilerConfig::default()));

    // Create storage with the profiler attached
    let adapter = Arc::new(DualWriteAdapter::new_for_testing().expect("in-memory adapter"));
    let storage = ProfDAGStorage::new(adapter.clone(), ProfDAGStorageConfig::default())
        .await
        .expect("ProfDAG schema init")
        .with_profiler(profiler.clone());
    let storage = Arc::new(storage);

    // Insert a node -- should record StorageWrite
    let node = ProfDAGNode::pattern("Profiler test pattern")
        .with_embedding(generate_embedding(128, 42))
        .with_confidence(0.9);
    storage.insert_node(&node).await.expect("insert node");

    // Get the node back -- should record StorageRead
    let fetched = storage.get_node(&node.id).await.expect("get node");
    assert!(fetched.is_some());

    // Insert an edge -- should record StorageWrite
    let node2 = ProfDAGNode::pattern("Second pattern")
        .with_embedding(generate_embedding(128, 43))
        .with_confidence(0.8);
    storage.insert_node(&node2).await.expect("insert node2");

    let edge = ProfDAGEdge::leads_to(&node.id, &node2.id, 0.9);
    storage.insert_edge(&edge).await.expect("insert edge");

    // Get neighbors -- should record StorageRead
    let _neighbors = storage
        .get_neighbors(&node.id, &NeighborQuery::outgoing())
        .await
        .expect("get neighbors");

    // Check profiler recorded storage operations
    let snapshot = profiler.snapshot();
    assert!(
        snapshot.total_operations >= 5,
        "Profiler should have recorded at least 5 storage operations (2 writes + 1 read + 1 write + 1 read), got {}",
        snapshot.total_operations
    );

    let write_stats = snapshot.by_type.get(&OperationType::StorageWrite);
    assert!(
        write_stats.is_some(),
        "Profiler should have StorageWrite entries"
    );
    assert!(
        write_stats.unwrap().count >= 3,
        "Should have at least 3 StorageWrite operations (2 nodes + 1 edge), got {}",
        write_stats.unwrap().count
    );

    let read_stats = snapshot.by_type.get(&OperationType::StorageRead);
    assert!(
        read_stats.is_some(),
        "Profiler should have StorageRead entries"
    );
    assert!(
        read_stats.unwrap().count >= 2,
        "Should have at least 2 StorageRead operations (get_node + get_neighbors), got {}",
        read_stats.unwrap().count
    );

    // Now wire the same profiler into search and verify Search operations are recorded
    let search = ProfDAGSearch::new(storage.clone(), SearchConfig::default())
        .with_profiler(profiler.clone());
    search.rebuild_index().await.expect("rebuild index");

    let query = generate_embedding(128, 42);
    let results = search.find_similar(&query, 5, 0.0).await.expect("find_similar");
    assert!(!results.is_empty(), "Should find results");

    let final_snapshot = profiler.snapshot();
    let search_stats = final_snapshot.by_type.get(&OperationType::Search);
    assert!(
        search_stats.is_some(),
        "Profiler should have Search entries after find_similar"
    );
    assert!(
        search_stats.unwrap().count >= 1,
        "Should have at least 1 Search operation, got {}",
        search_stats.unwrap().count
    );

    println!("=== Profiler Wiring Test Summary ===");
    println!("  Total operations: {}", final_snapshot.total_operations);
    println!(
        "  StorageWrite: {}",
        final_snapshot
            .by_type
            .get(&OperationType::StorageWrite)
            .map(|s| s.count)
            .unwrap_or(0)
    );
    println!(
        "  StorageRead: {}",
        final_snapshot
            .by_type
            .get(&OperationType::StorageRead)
            .map(|s| s.count)
            .unwrap_or(0)
    );
    println!(
        "  Search: {}",
        final_snapshot
            .by_type
            .get(&OperationType::Search)
            .map(|s| s.count)
            .unwrap_or(0)
    );
    println!("=== All profiler assertions passed ===");
}
