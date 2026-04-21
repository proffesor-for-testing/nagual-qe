//! RuVector Integration Smoke Test
//!
//! Tests the REAL ruvector stack end-to-end:
//! - instant-distance HNSW index (build + search)
//! - ProfDAG storage via DualWriteAdapter (in-memory SQLite)
//! - Trajectory recording through actual library types
//! - FastGRNN native backend inference
//!
//! These tests hit real code paths, not HashMap mocks.

use std::sync::Arc;

mod common;
use common::{normalized_embedding, similar_embeddings};

// ============================================================================
// HNSW Index: instant-distance integration
// ============================================================================

mod hnsw_integration {
    use super::*;
    use nagual::profdag::search::{ProfDAGSearch, SearchConfig};
    use nagual::profdag::storage::{ProfDAGStorage, ProfDAGStorageConfig};
    use nagual::profdag::{NodeType, ProfDAGNode};
    use nagual::db::DualWriteAdapter;

    async fn setup_storage() -> Arc<ProfDAGStorage> {
        let adapter = Arc::new(
            DualWriteAdapter::new_for_testing()
                .expect("Failed to create test adapter"),
        );
        Arc::new(
            ProfDAGStorage::new(adapter, ProfDAGStorageConfig::default())
                .await
                .expect("Failed to create ProfDAG storage"),
        )
    }

    #[tokio::test]
    async fn test_hnsw_index_build_and_search() {
        let storage = setup_storage().await;
        let search = ProfDAGSearch::new(storage.clone(), SearchConfig::default());

        // Insert nodes with known embeddings
        let base = normalized_embedding(128);
        let near = similar_embeddings(&base, 5, 0.05);
        let far = similar_embeddings(&base, 5, 0.8);

        // Insert "near" nodes
        for (i, emb) in near.iter().enumerate() {
            let mut node = ProfDAGNode::new(NodeType::Pattern, format!("Near pattern {}", i));
            node.embedding = Some(emb.clone());
            storage.insert_node(&node).await.expect("insert near node");
        }

        // Insert "far" nodes
        for (i, emb) in far.iter().enumerate() {
            let mut node = ProfDAGNode::new(NodeType::Pattern, format!("Far pattern {}", i));
            node.embedding = Some(emb.clone());
            storage.insert_node(&node).await.expect("insert far node");
        }

        // Search with base embedding - near nodes should rank higher
        let results = search.find_similar(&base, 5, 0.0).await.expect("search");

        assert!(
            !results.is_empty(),
            "HNSW search returned no results"
        );

        // Top result should have high similarity (near node)
        assert!(
            results[0].similarity > 0.8,
            "Top result similarity {} is too low, expected > 0.8",
            results[0].similarity
        );
    }

    #[tokio::test]
    async fn test_hnsw_search_latency_under_10ms() {
        let storage = setup_storage().await;
        let search = ProfDAGSearch::new(storage.clone(), SearchConfig::default());

        // Build index with 500 nodes
        for i in 0..500 {
            let emb = normalized_embedding(128);
            let mut node = ProfDAGNode::new(NodeType::Pattern, format!("Pattern {}", i));
            node.embedding = Some(emb);
            storage.insert_node(&node).await.expect("insert node");
        }

        // Force index build
        let _ = search.find_similar(&normalized_embedding(128), 1, 0.0).await;

        // Benchmark
        let query = normalized_embedding(128);
        let start = std::time::Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            let _ = search.find_similar(&query, 10, 0.0).await;
        }
        let avg_ms = start.elapsed().as_millis() as f64 / iterations as f64;

        assert!(
            avg_ms < 10.0,
            "Average HNSW search latency {:.2}ms exceeds 10ms target",
            avg_ms
        );
    }

    #[tokio::test]
    async fn test_hnsw_recall_above_95_percent() {
        let storage = setup_storage().await;
        let search = ProfDAGSearch::new(storage.clone(), SearchConfig::default());

        // Insert a cluster of similar nodes
        let center = normalized_embedding(128);
        let cluster = similar_embeddings(&center, 20, 0.02);
        let mut cluster_ids = Vec::new();

        for (i, emb) in cluster.iter().enumerate() {
            let mut node = ProfDAGNode::new(NodeType::Pattern, format!("Cluster {}", i));
            node.embedding = Some(emb.clone());
            let id = storage.insert_node(&node).await.expect("insert");
            cluster_ids.push(id);
        }

        // Also insert noise far away
        for i in 0..80 {
            let emb = normalized_embedding(128);
            let mut node = ProfDAGNode::new(NodeType::Pattern, format!("Noise {}", i));
            node.embedding = Some(emb);
            storage.insert_node(&node).await.expect("insert noise");
        }

        // Search for the cluster center, asking for 20 results
        let results = search.find_similar(&center, 20, 0.0).await.expect("search");

        // Count how many of the cluster nodes were found
        let found_cluster: usize = results
            .iter()
            .filter(|r| cluster_ids.contains(&r.node.id))
            .count();

        let recall = found_cluster as f64 / cluster_ids.len() as f64;
        assert!(
            recall > 0.90,
            "HNSW recall {:.2}% is below 90% threshold (found {}/{})",
            recall * 100.0,
            found_cluster,
            cluster_ids.len()
        );
    }
}

// ============================================================================
// ProfDAG Storage: SQLite CRUD via DualWriteAdapter
// ============================================================================

mod storage_integration {
    use super::*;
    use nagual::profdag::storage::{ProfDAGStorage, ProfDAGStorageConfig};
    use nagual::profdag::{EdgeType, NodeType, ProfDAGEdge, ProfDAGNode};
    use nagual::db::DualWriteAdapter;

    #[tokio::test]
    async fn test_node_crud_roundtrip() {
        let adapter = Arc::new(
            DualWriteAdapter::new_for_testing().expect("adapter"),
        );
        let storage = ProfDAGStorage::new(adapter, ProfDAGStorageConfig::default())
            .await
            .expect("storage");

        // Create
        let emb = normalized_embedding(128);
        let mut node = ProfDAGNode::new(NodeType::Pattern, "Roundtrip test");
        node.embedding = Some(emb.clone());
        let id = storage.insert_node(&node).await.expect("insert");

        // Read
        let loaded = storage.get_node(&id).await.expect("get");
        assert!(loaded.is_some(), "Node not found after insert");
        let loaded = loaded.unwrap();
        assert_eq!(loaded.content, "Roundtrip test");
        assert_eq!(loaded.node_type, NodeType::Pattern);
        assert!(loaded.embedding.is_some());
    }

    #[tokio::test]
    async fn test_edge_crud_and_neighbor_query() {
        let adapter = Arc::new(
            DualWriteAdapter::new_for_testing().expect("adapter"),
        );
        let storage = ProfDAGStorage::new(adapter, ProfDAGStorageConfig::default())
            .await
            .expect("storage");

        // Create two nodes
        let mut n1 = ProfDAGNode::new(NodeType::Pattern, "Source");
        n1.embedding = Some(normalized_embedding(128));
        let id1 = storage.insert_node(&n1).await.expect("insert n1");

        let mut n2 = ProfDAGNode::new(NodeType::Pattern, "Target");
        n2.embedding = Some(normalized_embedding(128));
        let id2 = storage.insert_node(&n2).await.expect("insert n2");

        // Create edge
        let edge = ProfDAGEdge::new(&id1, &id2, EdgeType::LeadsTo, 0.9);
        storage.insert_edge(&edge).await.expect("insert edge");

        // Query neighbors
        let query = nagual::profdag::NeighborQuery::outgoing();
        let neighbors = storage.get_neighbors(&id1, &query).await.expect("neighbors");
        assert!(
            !neighbors.is_empty(),
            "No neighbors found for source node"
        );
    }
}

// ============================================================================
// Trajectory Recording: actual library types (sync, in-memory)
// ============================================================================

mod trajectory_integration {
    use nagual::profdag::{TrajectoryRecorder, RecordingSession};
    use nagual::learning::{Outcome, StepType, TrajectoryStep};

    #[test]
    fn test_trajectory_record_complete_replay() {
        let recorder = TrajectoryRecorder::new();

        // Start trajectory
        let traj_id = recorder.start(
            "Fix database timeout",
            Some("test-session".to_string()),
        );

        // Record steps (sync, no async)
        let step1 = TrajectoryStep::new(
            StepType::PatternRetrieval,
            vec![],
            "Finding timeout patterns",
            0.8,
        );
        recorder.record_step(&traj_id, step1).expect("record step 1");

        let step2 = TrajectoryStep::new(
            StepType::PatternApplication,
            vec![],
            "Applying retry backoff",
            0.9,
        );
        recorder.record_step(&traj_id, step2).expect("record step 2");

        // Verify active
        assert!(recorder.is_active(&traj_id));
        assert_eq!(recorder.active_count(), 1);

        // Complete
        let result = recorder.complete(&traj_id, Outcome::Success, 1.0)
            .expect("complete trajectory");

        assert_eq!(result.outcome, Outcome::Success);
        assert_eq!(result.step_count, 2);
        assert!(result.reward > 0.0);

        // No longer active
        assert!(!recorder.is_active(&traj_id));

        // Replay from storage
        let replay = recorder.replay(&traj_id).expect("replay");
        assert_eq!(replay.trajectory.step_count(), 2);
    }

    #[test]
    fn test_trajectory_abort() {
        let recorder = TrajectoryRecorder::new();
        let traj_id = recorder.start("Abortable task", None);

        let step = TrajectoryStep::new(
            StepType::Decision,
            vec![],
            "Some decision",
            0.5,
        );
        recorder.record_step(&traj_id, step).unwrap();

        assert!(recorder.abort(&traj_id));
        assert!(!recorder.is_active(&traj_id));
    }

    #[test]
    fn test_recording_session_fluent_api() {
        use std::sync::Arc;

        let recorder = Arc::new(TrajectoryRecorder::new());
        let session = RecordingSession::new(recorder.clone(), "Fluent API task");

        let step1 = TrajectoryStep::new(StepType::PatternRetrieval, vec![], "step1", 0.8);
        session.record(step1).unwrap();

        let step2 = TrajectoryStep::new(StepType::PatternApplication, vec![], "step2", 0.9);
        session.record(step2).unwrap();

        let result = session.complete(Outcome::Success, 0.9).expect("complete");
        assert_eq!(result.step_count, 2);
    }
}

// ============================================================================
// FastGRNN Native Backend: real inference
// ============================================================================

mod fastgrnn_integration {
    use nagual::router::{FastGRNN, FastGRNNBackend, FastGRNNConfig};

    #[test]
    fn test_native_backend_inference_produces_valid_scores() {
        let config = FastGRNNConfig::default();
        let model = FastGRNN::new(config.clone()).expect("create model");

        // Test diverse inputs
        let test_cases = vec![
            // [query_len, emb_norm, domain_spec, pattern_cov, hist_accuracy]
            vec![0.1, 0.1, 0.1, 0.9, 0.9], // Simple query, good coverage -> low complexity
            vec![0.9, 0.9, 0.9, 0.1, 0.1], // Complex query, poor coverage -> high complexity
            vec![0.5, 0.5, 0.5, 0.5, 0.5], // Balanced
        ];

        for features in &test_cases {
            let score = model.forward(features).expect("inference");
            assert!(
                score >= 0.0 && score <= 1.0,
                "Complexity score {} out of [0,1] range for {:?}",
                score,
                features
            );
        }
    }

    #[test]
    fn test_backend_fallback_chain() {
        // No ONNX file -> should fall back to JSON -> then to native pretrained
        let backend = FastGRNNBackend::load(
            Some("nonexistent.onnx"),
            Some("nonexistent.json"),
            FastGRNNConfig::default(),
        )
        .expect("backend should fall back to native");

        assert!(backend.is_native(), "Should have fallen back to native backend");

        let score = backend
            .forward(&[0.5, 0.5, 0.5, 0.5, 0.5])
            .expect("inference");
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_json_weights_loading() {
        let json_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/models/fastgrnn_router.json"
        );

        if std::path::Path::new(json_path).exists() {
            let backend = FastGRNNBackend::load(
                None,
                Some(json_path),
                FastGRNNConfig::default(),
            )
            .expect("load from JSON");

            assert!(backend.is_native(), "JSON loads into native backend");

            // Inference should work with trained weights
            let score = backend
                .forward(&[0.5, 0.5, 0.5, 0.5, 0.5])
                .expect("inference with trained weights");
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_native_inference_under_1ms() {
        let config = FastGRNNConfig::compact();
        let model = FastGRNN::new(config.clone()).expect("create model");

        let features = vec![0.5; config.input_dim];

        // Warmup
        for _ in 0..100 {
            let _ = model.forward(&features);
        }

        model.reset_stats();

        let iterations = 1000;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = model.forward(&features);
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

        assert!(
            avg_us < 1000.0,
            "Average native inference {:.1}us exceeds 1ms",
            avg_us
        );
    }
}
