//! Attention Surgery Demo
//!
//! Demonstrates how to use E_nagual attention bias injection for open-weight
//! models. This example uses mock data and does not require loading an actual
//! model.
//!
//! Run with: `cargo run --example attention_surgery_demo`

use nagual::injection::attention_surgery::{
    AttentionSurgery, AttentionSurgeryConfig, BiasMethod, ModelConfig, RiskLevel,
};
use nagual::injection::model_hooks::{AttentionState, ENagualHook, HookRegistry, LoggingHook};
use nagual::injection::{ENagual, ENagualConfig};
use nagual::reasoning_bank::{FactorScores, Pattern, ScoredPattern};

fn main() {
    println!("=================================================================");
    println!("  Attention Surgery Demo - E_nagual for Open-Weight Models");
    println!("=================================================================\n");

    // ---------------------------------------------------------------
    // 1. Create mock E_nagual from patterns
    // ---------------------------------------------------------------
    println!("--- Step 1: Creating E_nagual from patterns ---\n");

    let patterns: Vec<ScoredPattern> = vec![
        make_pattern(
            "How to handle database timeouts?",
            "Use connection pooling with retry logic and exponential backoff",
            0.92,
            0.88,
        ),
        make_pattern(
            "How to optimize SQL queries?",
            "Add indexes, use EXPLAIN ANALYZE, prefer batch operations",
            0.85,
            0.82,
        ),
        make_pattern(
            "How to handle concurrent writes?",
            "Use optimistic locking with version columns",
            0.78,
            0.75,
        ),
    ];

    let e_nagual = ENagual::new("How should I handle database performance issues?")
        .with_patterns(patterns)
        .with_config(ENagualConfig::default());

    println!(
        "  E_nagual computed: {} patterns, overall confidence {:.0}%",
        e_nagual.pattern_count(),
        e_nagual.overall_confidence() * 100.0
    );

    // ---------------------------------------------------------------
    // 2. Configure attention surgery
    // ---------------------------------------------------------------
    println!("\n--- Step 2: Configuring attention surgery ---\n");

    let config = AttentionSurgeryConfig::builder()
        .with_bias_scale(0.1)
        .with_bias_method(BiasMethod::Additive)
        .with_max_bias_norm(2.0)
        .with_warmup_tokens(10)
        .with_decay_rate(0.9)
        .build();

    println!("  Bias method:     {}", config.bias_method);
    println!("  Bias scale:      {}", config.bias_scale);
    println!("  Max bias norm:   {}", config.max_bias_norm);
    println!("  Warmup tokens:   {}", config.warmup_tokens);
    println!("  Decay rate:      {}", config.decay_rate);

    let surgery = AttentionSurgery::new(config);
    let model_config = ModelConfig::llama_7b();

    println!("\n  Model: {} ({} layers, {} heads, head_dim {})",
        model_config.model_type, model_config.num_layers,
        model_config.num_heads, model_config.head_dim
    );

    // ---------------------------------------------------------------
    // 3. Prepare per-layer biases
    // ---------------------------------------------------------------
    println!("\n--- Step 3: Preparing per-layer biases ---\n");

    let seq_len = 64;
    let biases = surgery.prepare_biases_for_seq_len(&e_nagual, &model_config, seq_len);

    for bias in &biases {
        println!(
            "  Layer {:2}: bias norm = {:.4}, heads = {}, seq_len = {}",
            bias.target_layer,
            bias.norm(),
            bias.num_heads(),
            bias.seq_len(),
        );
    }

    // ---------------------------------------------------------------
    // 4. Estimate impact
    // ---------------------------------------------------------------
    println!("\n--- Step 4: Estimating impact ---\n");

    let impact = surgery.estimate_impact(&biases);

    println!("  Total bias norm:        {:.4}", impact.total_bias_norm);
    println!("  Affected layers:        {}", impact.affected_layers);
    println!("  Affected heads:         {}", impact.affected_heads);
    println!("  Est. KL divergence:     {:.6}", impact.estimated_kl_divergence);
    println!("  Risk level:             {}", impact.risk_level);

    if impact.risk_level <= RiskLevel::Medium {
        println!("\n  SAFE to apply biases.");
    } else {
        println!("\n  WARNING: High risk - consider reducing bias_scale.");
    }

    // ---------------------------------------------------------------
    // 5. Register hooks and simulate inference
    // ---------------------------------------------------------------
    println!("\n--- Step 5: Simulating inference with hooks ---\n");

    let e_nagual_hook = ENagualHook::new(
        AttentionSurgery::new(AttentionSurgeryConfig::builder()
            .with_warmup_tokens(0) // no warmup for demo
            .build()),
        biases,
    );

    let mut registry = HookRegistry::new();
    registry.register(Box::new(e_nagual_hook));
    registry.register(Box::new(LoggingHook::new("demo_logger")));

    println!("  Registered hooks: {:?}", registry.hooks());

    // Simulate a few inference steps.
    let num_layers = 4; // just the targeted layers
    let target_start = model_config.num_layers - num_layers;

    for token_step in 0..3 {
        for layer_offset in 0..num_layers {
            let layer = target_start + layer_offset;
            let mut state = AttentionState::new(
                vec![0.0; model_config.head_dim],
                vec![0.0; model_config.head_dim],
                vec![0.0; model_config.head_dim],
                layer,
                0,
            )
            .with_num_heads(model_config.num_heads)
            .with_seq_len(seq_len)
            .with_attention_scores(vec![0.0; model_config.num_heads * seq_len]);

            registry.execute_pre_attention(layer, &mut state).unwrap();
        }

        // Simulate token sampling.
        let mock_logits = vec![0.0; 32000];
        registry.execute_on_token(token_step, &mock_logits);
    }

    println!("  Simulated 3 token steps across {} layers.", num_layers);

    // ---------------------------------------------------------------
    // 6. Pattern attention scores
    // ---------------------------------------------------------------
    println!("\n--- Step 6: Computing pattern attention ---\n");

    let pattern_embeddings = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.7, 0.7, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let query_embedding = vec![0.9, 0.1, 0.0, 0.0];

    let attention_weights = surgery.compute_pattern_attention(&pattern_embeddings, &query_embedding);

    for (i, w) in attention_weights.iter().enumerate() {
        println!("  Pattern {}: attention weight = {:.4}", i, w);
    }

    println!("\n=================================================================");
    println!("  Demo complete.");
    println!("=================================================================");
}

fn make_pattern(problem: &str, solution: &str, similarity: f32, confidence: f32) -> ScoredPattern {
    let pattern = Pattern::new(problem, solution, "database.performance")
        .with_context("Production workload")
        .with_confidence(confidence)
        .with_reward(0.85);

    ScoredPattern {
        pattern,
        similarity,
        final_score: (similarity + confidence) / 2.0,
        factor_scores: FactorScores::default(),
    }
}
