//! Router tests — exercised against the production router in `nagual::router`.
//!
//! This file used to define its own ~530-line mock `Router` and test that mock, so none of its
//! 54 tests could fail because of a change in `src/router`. It now covers the same areas —
//! FastGRNN inference, complexity estimation, vendor selection, fallback chains, latency,
//! statistics, properties and edge cases — against the real `VendorRouter`, `VendorSelector`,
//! `ComplexityEstimator` and `FastGRNN`.
//!
//! Estimator quality is asserted against `models/router_queries.jsonl` (labelled queries, rubric in
//! models/README.md): the FastGRNN weights are trained on its `train` split only, and
//! `quality_tests` checks the held-out `test` split against a bar fixed before training.

use std::collections::HashSet;

use nagual::router::{
    ComplexityEstimator, ComplexityLevel, EstimatorConfig, FallbackChain, FastGRNN, FastGRNNConfig,
    RouterConfig, Vendor, VendorConfig, VendorRouter, VendorSelector,
};
use proptest::prelude::*;

mod common;
use common::normalized_embedding;

const DIM: usize = 128;

fn router() -> VendorRouter {
    VendorRouter::new(RouterConfig::default()).expect("default router config is valid")
}

fn selector() -> VendorSelector {
    VendorSelector::new(VendorConfig::default())
}

/// Deterministic unit-norm embedding (proptest-independent).
fn fixed_embedding() -> Vec<f32> {
    let v: Vec<f32> = (0..DIM)
        .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
        .collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / n).collect()
}

fn all_vendors() -> [Vendor; 4] {
    [
        Vendor::LocalSmall,
        Vendor::LocalLarge,
        Vendor::Claude,
        Vendor::GPT,
    ]
}

// ─── FastGRNN inference ─────────────────────────────────────────────────────────────────

mod fastgrnn_tests {
    use super::*;

    #[test]
    fn test_output_is_a_probability() {
        let model = FastGRNN::new(FastGRNNConfig::default()).unwrap();
        for features in [
            vec![0.0; 5],
            vec![1.0; 5],
            vec![0.5; 5],
            vec![0.006, 0.67, 0.2, 0.92, 0.5],
        ] {
            let y = model.forward(&features).unwrap();
            assert!((0.0..=1.0).contains(&y), "forward({features:?}) = {y}");
        }
    }

    #[test]
    fn test_inference_is_deterministic() {
        let model = FastGRNN::new(FastGRNNConfig::default()).unwrap();
        let f = vec![0.1, 0.6, 0.4, 0.8, 0.5];
        assert_eq!(model.forward(&f).unwrap(), model.forward(&f).unwrap());

        let other = FastGRNN::new(FastGRNNConfig::default()).unwrap();
        assert_eq!(
            model.forward(&f).unwrap(),
            other.forward(&f).unwrap(),
            "pretrained weights must be fixed"
        );
    }

    #[test]
    fn test_batch_matches_single_inference() {
        let model = FastGRNN::new(FastGRNNConfig::default()).unwrap();
        let batch = vec![vec![0.0; 5], vec![0.3, 0.6, 0.2, 0.9, 0.5], vec![1.0; 5]];
        let batched = model.forward_batch(&batch).unwrap();
        for (features, got) in batch.iter().zip(batched) {
            assert_eq!(got, model.forward(features).unwrap());
        }
    }

    #[test]
    fn test_rejects_wrong_input_dimension() {
        let model = FastGRNN::new(FastGRNNConfig::default()).unwrap();
        assert!(model.forward(&[0.5; 3]).is_err());
        assert!(model.forward(&[0.5; 6]).is_err());
    }

    #[test]
    fn test_counts_inferences() {
        let model = FastGRNN::new(FastGRNNConfig::default()).unwrap();
        for _ in 0..7 {
            model.forward(&[0.5; 5]).unwrap();
        }
        assert_eq!(model.inference_count(), 7);
        model.reset_stats();
        assert_eq!(model.inference_count(), 0);
    }

    #[test]
    fn test_model_is_edge_sized() {
        let model = FastGRNN::new(FastGRNNConfig::default()).unwrap();
        assert!(
            model.model_size_bytes() < 16 * 1024,
            "{} bytes",
            model.model_size_bytes()
        );
    }
}

// ─── Complexity estimation ──────────────────────────────────────────────────────────────

mod complexity_tests {
    use super::*;

    #[test]
    fn test_level_boundaries() {
        assert_eq!(ComplexityLevel::from_score(0.0), ComplexityLevel::Low);
        assert_eq!(ComplexityLevel::from_score(0.299), ComplexityLevel::Low);
        assert_eq!(ComplexityLevel::from_score(0.3), ComplexityLevel::Medium);
        assert_eq!(ComplexityLevel::from_score(0.499), ComplexityLevel::Medium);
        assert_eq!(ComplexityLevel::from_score(0.5), ComplexityLevel::High);
        assert_eq!(ComplexityLevel::from_score(0.699), ComplexityLevel::High);
        assert_eq!(ComplexityLevel::from_score(0.7), ComplexityLevel::VeryHigh);
        assert_eq!(ComplexityLevel::from_score(1.0), ComplexityLevel::VeryHigh);
    }

    #[test]
    fn test_features_are_normalised() {
        let est = ComplexityEstimator::new(EstimatorConfig::default());
        let f = est
            .extract_features(
                "Explain this code: ```rust fn main() {} ``` and fix it",
                &fixed_embedding(),
            )
            .unwrap();
        for (name, v) in ["length", "norm", "domain", "coverage", "accuracy"]
            .iter()
            .zip(f.to_vector())
        {
            assert!((0.0..=1.0).contains(&v), "{name} = {v}");
        }
    }

    #[test]
    fn test_longer_query_has_larger_length_feature() {
        let est = ComplexityEstimator::new(EstimatorConfig::default());
        let e = fixed_embedding();
        let short = est.extract_features("What is 2+2?", &e).unwrap();
        let long = est.extract_features(&"why ".repeat(300), &e).unwrap();
        assert!(long.query_length > short.query_length);
        let huge = est.extract_features(&"x".repeat(10_000), &e).unwrap();
        assert_eq!(
            huge.query_length, 1.0,
            "length feature saturates at max_query_length"
        );
    }

    #[test]
    fn test_technical_terms_raise_domain_specificity() {
        let est = ComplexityEstimator::new(EstimatorConfig::default());
        let e = fixed_embedding();
        let general = est.extract_features("how can you help", &e).unwrap();
        let technical = est
            .extract_features("concurrent async cache implementation", &e)
            .unwrap();
        assert!(
            technical.domain_specificity > general.domain_specificity,
            "{} !> {}",
            technical.domain_specificity,
            general.domain_specificity
        );
    }

    #[test]
    fn test_recorded_accuracy_is_used_for_the_same_query() {
        let est = ComplexityEstimator::new(EstimatorConfig::default());
        let e = fixed_embedding();
        let q = "cache invalidation strategy";
        assert_eq!(
            est.extract_features(q, &e).unwrap().historical_accuracy,
            0.5,
            "neutral default"
        );
        est.record_accuracy(q, 0.9);
        assert!((est.extract_features(q, &e).unwrap().historical_accuracy - 0.9).abs() < 0.2);
        est.clear_cache();
        assert_eq!(
            est.extract_features(q, &e).unwrap().historical_accuracy,
            0.5
        );
    }

    #[test]
    fn test_estimate_is_a_probability_and_level_matches_score() {
        let r = router();
        for q in [
            "What is 2+2?",
            "hello",
            "Design a lock-free concurrent hash map",
        ] {
            let score = r.estimate_complexity(q, &fixed_embedding()).unwrap();
            assert!((0.0..=1.0).contains(&score.score), "{q}: {}", score.score);
            let d = r.route(q, &fixed_embedding()).unwrap();
            assert_eq!(d.level, ComplexityLevel::from_score(d.complexity));
        }
    }
}

// ─── Vendor selection ───────────────────────────────────────────────────────────────────

mod vendor_selection_tests {
    use super::*;

    #[test]
    fn test_thresholds_map_to_tiers() {
        let s = selector();
        assert_eq!(s.select(0.0, 1.0).vendor, Vendor::LocalSmall);
        assert_eq!(s.select(0.29, 1.0).vendor, Vendor::LocalSmall);
        assert_eq!(s.select(0.3, 1.0).vendor, Vendor::LocalLarge);
        assert_eq!(s.select(0.49, 1.0).vendor, Vendor::LocalLarge);
        assert_eq!(s.select(0.5, 1.0).vendor, Vendor::Claude);
        assert_eq!(s.select(0.95, 1.0).vendor, Vendor::Claude);
    }

    #[test]
    fn test_high_complexity_routes_to_a_cloud_vendor() {
        let d = selector().select(0.9, 0.9);
        assert!(d.vendor.is_cloud(), "{:?}", d.vendor);
        assert!(!d.is_fallback);
    }

    #[test]
    fn test_low_complexity_stays_local_and_cheap() {
        let d = selector().select(0.1, 0.9);
        assert!(d.vendor.is_local());
        assert_eq!(d.vendor.relative_cost(), 1);
    }

    #[test]
    fn test_quality_profile_escalates_earlier_than_latency_profile() {
        let quality = VendorSelector::new(VendorConfig::high_quality());
        let latency = VendorSelector::new(VendorConfig::low_latency());
        // high_quality leaves local at 0.4, low_latency at 0.6. (`cloud_threshold` is not used by
        // `select`: everything at or above `local_large_threshold` goes to Claude.)
        let x = 0.5;
        assert!(quality.select(x, 1.0).vendor.is_cloud());
        assert!(latency.select(x, 1.0).vendor.is_local());
    }

    #[test]
    fn test_decision_carries_its_reason_and_chain() {
        let d = selector().select(0.4, 0.7);
        assert_eq!(d.vendor, Vendor::LocalLarge);
        assert!(d.reason.contains("local-large"), "{}", d.reason);
        assert_eq!(d.fallback_chain.vendors.first(), Some(&Vendor::LocalLarge));
        assert!((d.confidence - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vendor_names_round_trip() {
        for v in all_vendors() {
            assert_eq!(Vendor::from_str(v.as_str()), Some(v));
            assert_ne!(v.is_local(), v.is_cloud());
        }
        assert_eq!(Vendor::from_str("anthropic"), Some(Vendor::Claude));
        assert_eq!(Vendor::from_str("openai"), Some(Vendor::GPT));
        assert_eq!(Vendor::from_str("mystery"), None);
    }
}

// ─── Fallback chains and vendor health ──────────────────────────────────────────────────

mod fallback_chain_tests {
    use super::*;

    #[test]
    fn test_chain_escalates_in_cost_order() {
        let chain = FallbackChain::default();
        let costs: Vec<u32> = chain.vendors.iter().map(|v| v.relative_cost()).collect();
        assert!(costs.windows(2).all(|w| w[0] <= w[1]), "{costs:?}");
        assert_eq!(
            FallbackChain::starting_from(Vendor::Claude).vendors,
            vec![Vendor::Claude, Vendor::GPT]
        );
        assert_eq!(
            FallbackChain::starting_from(Vendor::Claude).next_after(Vendor::Claude),
            Some(Vendor::GPT)
        );
        assert_eq!(
            FallbackChain::starting_from(Vendor::GPT).next_after(Vendor::GPT),
            None
        );
        assert!(FallbackChain::local_only()
            .vendors
            .iter()
            .all(|v| v.is_local()));
        assert!(FallbackChain::cloud_only()
            .vendors
            .iter()
            .all(|v| v.is_cloud()));
    }

    #[test]
    fn test_three_consecutive_failures_make_a_vendor_unavailable() {
        let s = selector();
        s.record_failure(Vendor::LocalSmall, "timeout".into());
        s.record_failure(Vendor::LocalSmall, "timeout".into());
        assert!(
            s.is_vendor_available(Vendor::LocalSmall),
            "two failures are tolerated"
        );
        s.record_failure(Vendor::LocalSmall, "timeout".into());
        assert!(!s.is_vendor_available(Vendor::LocalSmall));
    }

    #[test]
    fn test_success_resets_the_failure_streak() {
        let s = selector();
        s.record_failure(Vendor::Claude, "429".into());
        s.record_failure(Vendor::Claude, "429".into());
        s.record_success(Vendor::Claude, 1200);
        s.record_failure(Vendor::Claude, "429".into());
        assert!(s.is_vendor_available(Vendor::Claude));
        assert_eq!(
            s.get_status(Vendor::Claude)
                .unwrap()
                .consecutive_failure_count(),
            1
        );
    }

    #[test]
    fn test_unavailable_primary_falls_back_up_the_chain() {
        let s = selector();
        s.mark_unavailable(Vendor::LocalSmall);
        let d = s.select(0.1, 0.9);
        assert_eq!(d.vendor, Vendor::LocalLarge);
        assert!(d.is_fallback);
        assert!(d.reason.contains("local-small unavailable"), "{}", d.reason);
    }

    #[test]
    fn test_get_fallback_skips_unavailable_vendors() {
        let s = selector();
        s.mark_unavailable(Vendor::LocalLarge);
        assert_eq!(s.get_fallback(Vendor::LocalSmall), Some(Vendor::Claude));
        s.mark_unavailable(Vendor::GPT);
        assert_eq!(s.get_fallback(Vendor::Claude), None);
    }

    #[test]
    fn test_marking_available_again_restores_primary() {
        let s = selector();
        s.mark_unavailable(Vendor::Claude);
        assert_eq!(s.select(0.6, 1.0).vendor, Vendor::GPT);
        s.mark_available(Vendor::Claude);
        let d = s.select(0.6, 1.0);
        assert_eq!(d.vendor, Vendor::Claude);
        assert!(!d.is_fallback);
    }

    #[test]
    fn test_router_outcomes_feed_vendor_health() {
        let r = router();
        for _ in 0..3 {
            r.record_outcome("q", Vendor::LocalLarge, false, 0);
        }
        assert!(!r.vendor_status(Vendor::LocalLarge).unwrap().is_available());
        assert_eq!(r.get_fallback(Vendor::LocalSmall), Some(Vendor::Claude));
    }
}

// ─── Latency ────────────────────────────────────────────────────────────────────────────

mod performance_tests {
    use super::*;
    use std::time::Instant;

    /// The router config promises a routing decision within `max_latency_ms` (5 ms).
    #[test]
    fn test_routing_stays_within_its_latency_budget() {
        let r = router();
        let e = fixed_embedding();
        let budget_us = r.config().max_latency_ms * 1000;
        let start = Instant::now();
        let n = 200;
        for i in 0..n {
            let d = r
                .route(&format!("query number {i} about async caches"), &e)
                .unwrap();
            assert!(
                d.routing_latency_us <= budget_us,
                "{} µs > {} µs",
                d.routing_latency_us,
                budget_us
            );
        }
        let avg_us = start.elapsed().as_micros() as u64 / n;
        assert!(avg_us <= budget_us, "average {avg_us} µs");
    }

    #[test]
    fn test_latency_is_recorded_not_zero() {
        let r = router();
        r.route("hello", &fixed_embedding()).unwrap();
        assert!(r.metrics().avg_latency_us() > 0.0);
        let (count, avg_us) = r.model_stats();
        assert_eq!(count, 1);
        assert!(avg_us >= 0.0);
    }
}

// ─── Statistics ─────────────────────────────────────────────────────────────────────────

mod stats_tests {
    use super::*;

    #[test]
    fn test_vendor_status_rates() {
        let s = selector();
        let st = s.get_status(Vendor::GPT).unwrap();
        assert_eq!(st.success_rate(), 1.0, "no data yet: optimistic");
        s.record_success(Vendor::GPT, 1000);
        s.record_success(Vendor::GPT, 3000);
        s.record_failure(Vendor::GPT, "500".into());
        assert!((st.success_rate() - 2.0 / 3.0).abs() < 1e-9);
        assert!(
            (st.avg_latency_us() - 2000.0).abs() < 1e-9,
            "latency averages over successes"
        );
    }

    #[test]
    fn test_metrics_distribution_and_fallback_rate() {
        let s = selector();
        s.select(0.1, 1.0);
        s.select(0.1, 1.0);
        s.select(0.6, 1.0);
        s.mark_unavailable(Vendor::LocalSmall);
        s.select(0.1, 1.0); // fallback to LocalLarge

        let m = s.metrics();
        let dist = m.vendor_distribution();
        let total: f64 = dist.values().sum();
        assert!((total - 1.0).abs() < 1e-9, "{dist:?}");
        assert!((dist[&Vendor::LocalSmall] - 0.5).abs() < 1e-9);
        assert!((m.fallback_rate() - 0.25).abs() < 1e-9);

        m.reset();
        assert_eq!(m.fallback_rate(), 0.0);
    }

    #[test]
    fn test_reset_status_restores_all_vendors() {
        let s = selector();
        for v in all_vendors() {
            s.mark_unavailable(v);
        }
        s.reset_status();
        assert!(all_vendors().iter().all(|&v| s.is_vendor_available(v)));
    }
}

// ─── Confidence ─────────────────────────────────────────────────────────────────────────

mod confidence_tests {
    use super::*;

    #[test]
    fn test_confidence_is_bounded() {
        let r = router();
        for q in [
            "",
            "hello",
            "Explain this code: ```rust fn main() {} ```",
            &"why ".repeat(500),
        ] {
            let d = r.route(q, &fixed_embedding()).unwrap();
            assert!(
                (0.0..=1.0).contains(&d.confidence),
                "{q:?}: {}",
                d.confidence
            );
            let s = r.route_simple(q, &fixed_embedding()).unwrap();
            assert!(
                (0.0..=1.0).contains(&s.confidence),
                "{q:?}: {}",
                s.confidence
            );
        }
    }

    #[test]
    fn test_known_history_raises_confidence() {
        let est = ComplexityEstimator::new(EstimatorConfig::default());
        let e = fixed_embedding();
        let q = "database migration rollback";
        let before = est.estimate_simple(q, &e).unwrap().confidence;
        est.record_accuracy(q, 0.95);
        let after = est.estimate_simple(q, &e).unwrap().confidence;
        assert!(after > before, "{after} !> {before}");
    }
}

// ─── Properties ─────────────────────────────────────────────────────────────────────────

fn finite_embedding() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..1.0, 1..256)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn prop_route_never_panics_and_stays_in_range(query in ".{0,400}", emb in finite_embedding()) {
        let d = router().route(&query, &emb).unwrap();
        prop_assert!((0.0..=1.0).contains(&d.complexity));
        prop_assert!((0.0..=1.0).contains(&d.confidence));
        prop_assert_eq!(d.level, ComplexityLevel::from_score(d.complexity));
    }

    #[test]
    fn prop_route_is_deterministic(query in ".{0,120}", emb in finite_embedding()) {
        let r = router();
        let a = r.route(&query, &emb).unwrap();
        let b = r.route(&query, &emb).unwrap();
        prop_assert_eq!(a.vendor, b.vendor);
        prop_assert_eq!(a.complexity, b.complexity);
    }

    /// More complexity never routes to a cheaper vendor (all vendors healthy).
    #[test]
    fn prop_selection_is_monotonic_in_cost(a in 0.0f32..=1.0, b in 0.0f32..=1.0) {
        let s = selector();
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        prop_assert!(s.select(lo, 1.0).vendor.relative_cost() <= s.select(hi, 1.0).vendor.relative_cost());
    }

    /// With some vendors down, selection still returns an available vendor whenever one exists
    /// at or above the primary tier.
    #[test]
    fn prop_fallback_picks_an_available_vendor(x in 0.0f32..=1.0, down in prop::collection::vec(0usize..4, 0..3)) {
        let s = selector();
        let down: HashSet<usize> = down.into_iter().collect();
        for &i in &down {
            s.mark_unavailable(all_vendors()[i]);
        }
        let d = s.select(x, 1.0);
        let primary = selector().select(x, 1.0).vendor;
        let any_up = FallbackChain::starting_from(primary).vendors.iter().any(|&v| s.is_vendor_available(v));
        if any_up {
            prop_assert!(s.is_vendor_available(d.vendor), "{:?} is down", d.vendor);
        }
        prop_assert_eq!(d.is_fallback, d.vendor != primary);
    }
}

// ─── Edge cases ─────────────────────────────────────────────────────────────────────────

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_embedding_is_rejected() {
        assert!(router().route("anything", &[]).is_err());
        assert!(router().route_simple("anything", &[]).is_err());
    }

    /// A NaN component used to produce complexity NaN, which fell through every threshold to the
    /// most expensive tier. It must be rejected instead.
    #[test]
    fn test_non_finite_embedding_is_rejected() {
        let mut e = normalized_embedding(DIM);
        e[3] = f32::NAN;
        assert!(router().route("q", &e).is_err());
        e[3] = f32::INFINITY;
        assert!(router().route("q", &e).is_err());
        assert!(router().estimate_complexity("q", &e).is_err());
    }

    #[test]
    fn test_empty_and_whitespace_queries_route() {
        for q in ["", "   ", "\n\t"] {
            let d = router().route(q, &fixed_embedding()).unwrap();
            assert!((0.0..=1.0).contains(&d.complexity));
        }
    }

    #[test]
    fn test_unicode_and_huge_queries_route() {
        let r = router();
        r.route(
            "Kako da testiram async kod? 非同期テスト 🦀",
            &fixed_embedding(),
        )
        .unwrap();
        r.route(&"ő".repeat(100_000), &fixed_embedding()).unwrap();
    }

    #[test]
    fn test_any_embedding_dimension_is_accepted() {
        let r = router();
        for dim in [1, 3, 128, 384, 1536] {
            r.route("q", &normalized_embedding(dim)).unwrap();
        }
    }
}

// ─── Estimator quality on held-out labelled queries ────────────────────────────────────

mod quality_tests {
    use super::*;

    const LEVELS: [ComplexityLevel; 4] = [
        ComplexityLevel::Low,
        ComplexityLevel::Medium,
        ComplexityLevel::High,
        ComplexityLevel::VeryHigh,
    ];

    fn label(s: &str) -> ComplexityLevel {
        match s {
            "low" => ComplexityLevel::Low,
            "medium" => ComplexityLevel::Medium,
            "high" => ComplexityLevel::High,
            "very_high" => ComplexityLevel::VeryHigh,
            other => panic!("unknown level {other}"),
        }
    }

    fn rank(l: ComplexityLevel) -> i32 {
        LEVELS.iter().position(|&x| x == l).unwrap() as i32
    }

    /// (query, labelled level) for one split of models/router_queries.jsonl.
    fn split(name: &str) -> Vec<(String, ComplexityLevel)> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/models/router_queries.jsonl");
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .filter(|r| r["split"] == name)
            .map(|r| {
                (
                    r["query"].as_str().unwrap().to_string(),
                    label(r["level"].as_str().unwrap()),
                )
            })
            .collect()
    }

    /// The bar was fixed before training: exact level >= 65%, local-vs-cloud >= 85%, and no
    /// prediction more than one level off. Measured at training time: 65.0% / 90.0% / 1.
    #[test]
    fn test_held_out_queries_meet_the_quality_bar() {
        let r = router();
        let test = split("test");
        assert_eq!(test.len(), 40);

        let (mut exact, mut tier, mut worst) = (0usize, 0usize, 0i32);
        let mut misses = Vec::new();
        for (q, want) in &test {
            let d = r.route(q, &fixed_embedding()).unwrap();
            if d.level == *want {
                exact += 1;
            } else {
                misses.push(format!(
                    "{:?} -> {:?} ({:.2}): {}",
                    want, d.level, d.complexity, q
                ));
            }
            let want_cloud = matches!(want, ComplexityLevel::High | ComplexityLevel::VeryHigh);
            if d.vendor.is_cloud() == want_cloud {
                tier += 1;
            }
            worst = worst.max((rank(d.level) - rank(*want)).abs());
        }
        let n = test.len() as f64;
        let (exact_acc, tier_acc) = (exact as f64 / n, tier as f64 / n);
        let report = misses.join("\n  ");
        assert!(
            exact_acc >= 0.65,
            "level accuracy {exact_acc:.3} < 0.65\n  {report}"
        );
        assert!(
            tier_acc >= 0.85,
            "local-vs-cloud accuracy {tier_acc:.3} < 0.85\n  {report}"
        );
        assert!(
            worst <= 1,
            "a prediction was {worst} levels off\n  {report}"
        );
    }

    /// The router must discriminate: the old weights scored everything 0.47–0.53.
    #[test]
    fn test_scores_spread_across_levels() {
        let r = router();
        let mean = |lvl: ComplexityLevel| {
            let xs: Vec<f32> = split("test")
                .iter()
                .filter(|(_, l)| *l == lvl)
                .map(|(q, _)| r.route(q, &fixed_embedding()).unwrap().complexity)
                .collect();
            xs.iter().sum::<f32>() / xs.len() as f32
        };
        let means: Vec<f32> = LEVELS.iter().map(|&l| mean(l)).collect();
        assert!(
            means.windows(2).all(|w| w[0] < w[1]),
            "per-level means not increasing: {means:?}"
        );
        assert!(means[3] - means[0] > 0.4, "spread too small: {means:?}");
    }

    /// Anchors from the original spec: everyday questions stay on the cheapest tier, hard
    /// design/proof work goes to the cloud.
    #[test]
    fn test_anchor_queries() {
        let r = router();
        for q in ["hello", "What is 2+2?"] {
            let d = r.route(q, &fixed_embedding()).unwrap();
            assert_eq!(d.vendor, Vendor::LocalSmall, "{q}: {:.3}", d.complexity);
        }
        let hard =
            "Design a lock-free concurrent hash map in Rust with epoch-based memory reclamation, \
                    prove linearizability, and analyse ABA hazards under contention";
        let d = r.route(hard, &fixed_embedding()).unwrap();
        assert!(
            d.vendor.is_cloud() && d.level == ComplexityLevel::VeryHigh,
            "{:.3} {:?}",
            d.complexity,
            d.level
        );
    }

    /// The Rust forward pass must reproduce what the training script measured on the test split
    /// (guards against drift between `router_features`, the trainer and `FastGRNN::forward`).
    #[test]
    fn test_rust_inference_matches_recorded_training_metrics() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/models/fastgrnn_router.json");
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let recorded = doc["metrics"]["test"]["level_accuracy"].as_f64().unwrap();

        let r = router();
        let test = split("test");
        let exact = test
            .iter()
            .filter(|(q, want)| r.route(q, &fixed_embedding()).unwrap().level == *want)
            .count() as f64
            / test.len() as f64;
        assert!(
            (exact - recorded).abs() < 1e-9,
            "rust {exact} vs trainer {recorded}"
        );
    }
}
