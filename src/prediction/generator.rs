//! Prediction Generator - Creates predictions from patterns and context.
//!
//! This module provides functionality for generating predictions based on
//! historical patterns and current context. It analyzes similar patterns,
//! extracts outcome likelihood, and produces calibrated predictions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    calculate_probability, estimate_timeline, link_evidence, EvidenceLink, Prediction,
    PredictionBuilder, PredictionError, PredictionId, PredictionResult, ProbabilityConfig,
    ProbabilityResult, TimelineConfig, TimelineEstimate, WeightedPattern,
};

/// Context for generating a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationContext {
    /// The question or statement being predicted
    pub query: String,
    /// Domain for the prediction
    pub domain: String,
    /// Additional context information
    pub context: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// Agent ID
    pub agent_id: Option<String>,
    /// Minimum confidence threshold for including patterns
    pub min_confidence: f64,
    /// Minimum similarity threshold for patterns
    pub min_similarity: f64,
    /// Maximum number of patterns to consider
    pub max_patterns: usize,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GenerationContext {
    /// Create a new generation context with a query.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            domain: String::new(),
            context: String::new(),
            tags: Vec::new(),
            session_id: None,
            agent_id: None,
            min_confidence: 0.3,
            min_similarity: 0.5,
            max_patterns: 20,
            metadata: HashMap::new(),
        }
    }

    /// Set the domain.
    pub fn with_domain(mut self, domain: impl Into<String>) -> Self {
        self.domain = domain.into();
        self
    }

    /// Set additional context.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = context.into();
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set session ID.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set agent ID.
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Set minimum confidence threshold.
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set minimum similarity threshold.
    pub fn with_min_similarity(mut self, similarity: f64) -> Self {
        self.min_similarity = similarity.clamp(0.0, 1.0);
        self
    }

    /// Set maximum patterns to consider.
    pub fn with_max_patterns(mut self, max: usize) -> Self {
        self.max_patterns = max;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

impl Default for GenerationContext {
    fn default() -> Self {
        Self::new("")
    }
}

/// Analysis of a single pattern for prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysis {
    /// Pattern ID
    pub pattern_id: String,
    /// Similarity to the query (0.0-1.0)
    pub similarity: f64,
    /// Pattern's success rate
    pub success_rate: f64,
    /// Pattern's confidence
    pub confidence: f64,
    /// Pattern's effectiveness
    pub effectiveness: f64,
    /// Weight contribution to the prediction
    pub weight: f64,
    /// Time since pattern was created (in days)
    pub age_days: i64,
    /// Whether the pattern is considered relevant
    pub is_relevant: bool,
}

impl PatternAnalysis {
    /// Calculate a combined quality score for the pattern.
    pub fn quality_score(&self) -> f64 {
        // Weighted combination of metrics
        let base_score = self.success_rate * 0.4
            + self.confidence * 0.3
            + self.effectiveness * 0.2
            + self.similarity * 0.1;

        // Apply recency decay (patterns older than 30 days start decaying)
        let recency_factor = if self.age_days <= 30 {
            1.0
        } else {
            (-((self.age_days - 30) as f64) / 180.0).exp()
        };

        base_score * recency_factor
    }
}

/// Result of prediction generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// The generated prediction
    pub prediction: Prediction,
    /// Probability calculation details
    pub probability_result: ProbabilityResult,
    /// Timeline estimation details
    pub timeline_estimate: TimelineEstimate,
    /// Evidence links
    pub evidence_links: Vec<EvidenceLink>,
    /// Analysis of patterns used
    pub pattern_analyses: Vec<PatternAnalysis>,
    /// Generation metadata
    pub metadata: GenerationMetadata,
}

/// Metadata about the prediction generation process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationMetadata {
    /// Number of patterns considered
    pub patterns_considered: usize,
    /// Number of patterns used (after filtering)
    pub patterns_used: usize,
    /// Average pattern similarity
    pub avg_similarity: f64,
    /// Average pattern quality
    pub avg_quality: f64,
    /// Generation duration in milliseconds
    pub duration_ms: u64,
    /// Timestamp of generation
    pub generated_at: DateTime<Utc>,
    /// Any warnings during generation
    pub warnings: Vec<String>,
}

/// A generated prediction with all supporting data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedPrediction {
    /// The prediction
    pub prediction: Prediction,
    /// Evidence pattern IDs with relevance scores
    pub evidence: Vec<EvidenceLink>,
    /// Pattern analyses
    pub analyses: Vec<PatternAnalysis>,
    /// Generation timestamp
    pub generated_at: DateTime<Utc>,
}

/// Input pattern data for prediction generation.
/// This is a simplified view of patterns used during generation.
#[derive(Debug, Clone)]
pub struct InputPattern {
    /// Pattern ID
    pub id: String,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Confidence (0.0-1.0)
    pub confidence: f64,
    /// Effectiveness (0.0-1.0)
    pub effectiveness: f64,
    /// Similarity to current context (0.0-1.0)
    pub similarity: f64,
    /// Pattern creation timestamp
    pub created_at: DateTime<Utc>,
    /// Optional resolution time in days (for timeline estimation)
    pub resolution_time_days: Option<u32>,
}

impl InputPattern {
    /// Create a new input pattern.
    pub fn new(id: impl Into<String>, similarity: f64) -> Self {
        Self {
            id: id.into(),
            success_rate: 0.5,
            confidence: 0.5,
            effectiveness: 0.5,
            similarity,
            created_at: Utc::now(),
            resolution_time_days: None,
        }
    }

    /// Set success rate.
    pub fn with_success_rate(mut self, rate: f64) -> Self {
        self.success_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set effectiveness.
    pub fn with_effectiveness(mut self, effectiveness: f64) -> Self {
        self.effectiveness = effectiveness.clamp(0.0, 1.0);
        self
    }

    /// Set created_at timestamp.
    pub fn with_created_at(mut self, created_at: DateTime<Utc>) -> Self {
        self.created_at = created_at;
        self
    }

    /// Set resolution time.
    pub fn with_resolution_time(mut self, days: u32) -> Self {
        self.resolution_time_days = Some(days);
        self
    }

    /// Get age in days.
    pub fn age_days(&self) -> i64 {
        (Utc::now() - self.created_at).num_days()
    }
}

/// Prediction generator that creates predictions from patterns.
#[derive(Debug, Clone)]
pub struct PredictionGenerator {
    /// Probability calculation configuration
    pub probability_config: ProbabilityConfig,
    /// Timeline estimation configuration
    pub timeline_config: TimelineConfig,
    /// Minimum patterns required for prediction
    pub min_patterns: usize,
}

impl PredictionGenerator {
    /// Create a new prediction generator with default configuration.
    pub fn new() -> Self {
        Self {
            probability_config: ProbabilityConfig::default(),
            timeline_config: TimelineConfig::default(),
            min_patterns: 1,
        }
    }

    /// Set probability configuration.
    pub fn with_probability_config(mut self, config: ProbabilityConfig) -> Self {
        self.probability_config = config;
        self
    }

    /// Set timeline configuration.
    pub fn with_timeline_config(mut self, config: TimelineConfig) -> Self {
        self.timeline_config = config;
        self
    }

    /// Set minimum patterns required.
    pub fn with_min_patterns(mut self, min: usize) -> Self {
        self.min_patterns = min;
        self
    }

    /// Generate a prediction from patterns and context.
    pub fn generate_prediction(
        &self,
        patterns: &[InputPattern],
        context: &GenerationContext,
    ) -> PredictionResult<GenerationResult> {
        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // Filter patterns based on context thresholds
        let filtered_patterns: Vec<&InputPattern> = patterns
            .iter()
            .filter(|p| {
                p.similarity >= context.min_similarity && p.confidence >= context.min_confidence
            })
            .take(context.max_patterns)
            .collect();

        // Check minimum patterns
        if filtered_patterns.len() < self.min_patterns {
            return Err(PredictionError::InsufficientPatterns {
                required: self.min_patterns,
                found: filtered_patterns.len(),
            });
        }

        // Analyze patterns
        let pattern_analyses: Vec<PatternAnalysis> = filtered_patterns
            .iter()
            .map(|p| PatternAnalysis {
                pattern_id: p.id.clone(),
                similarity: p.similarity,
                success_rate: p.success_rate,
                confidence: p.confidence,
                effectiveness: p.effectiveness,
                weight: 0.0, // Will be calculated during probability computation
                age_days: p.age_days(),
                is_relevant: true,
            })
            .collect();

        // Convert to weighted patterns for probability calculation
        let weighted_patterns: Vec<WeightedPattern> = filtered_patterns
            .iter()
            .map(|p| WeightedPattern {
                success_rate: p.success_rate,
                confidence: p.confidence,
                similarity: p.similarity,
                recency_weight: calculate_recency_weight(p.age_days()),
            })
            .collect();

        // Calculate probability
        let probability_result =
            calculate_probability(&weighted_patterns, None, &self.probability_config)?;

        // Estimate timeline
        let resolution_times: Vec<u32> = filtered_patterns
            .iter()
            .filter_map(|p| p.resolution_time_days)
            .collect();

        let timeline_estimate = if resolution_times.is_empty() {
            warnings.push("No resolution time data available, using defaults".to_string());
            TimelineEstimate::default()
        } else {
            estimate_timeline(&resolution_times, &self.timeline_config)?
        };

        // Create evidence links
        let evidence_links: Vec<EvidenceLink> = filtered_patterns
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let analysis = &pattern_analyses[i];
                link_evidence(&p.id, analysis.quality_score())
            })
            .collect();

        // Build the prediction
        let mut builder = PredictionBuilder::new()
            .id(PredictionId::new())
            .description(&context.query)
            .probability(probability_result.probability)
            .confidence(probability_result.confidence)
            .timeline(timeline_estimate.min_days, timeline_estimate.max_days)
            .domain(&context.domain)
            .context(&context.context)
            .tags(context.tags.clone());

        // Add evidence pattern IDs
        for link in &evidence_links {
            builder = builder.evidence_pattern(&link.pattern_id);
        }

        // Add optional fields
        if let Some(ref session_id) = context.session_id {
            builder = builder.session_id(session_id);
        }
        if let Some(ref agent_id) = context.agent_id {
            builder = builder.agent_id(agent_id);
        }

        // Add metadata
        for (key, value) in &context.metadata {
            builder = builder.meta(key, value.clone());
        }

        let prediction = builder.build()?;

        // Calculate generation metadata
        let duration_ms = start_time.elapsed().as_millis() as u64;
        let avg_similarity = if pattern_analyses.is_empty() {
            0.0
        } else {
            pattern_analyses.iter().map(|a| a.similarity).sum::<f64>()
                / pattern_analyses.len() as f64
        };
        let avg_quality = if pattern_analyses.is_empty() {
            0.0
        } else {
            pattern_analyses.iter().map(|a| a.quality_score()).sum::<f64>()
                / pattern_analyses.len() as f64
        };

        let metadata = GenerationMetadata {
            patterns_considered: patterns.len(),
            patterns_used: filtered_patterns.len(),
            avg_similarity,
            avg_quality,
            duration_ms,
            generated_at: Utc::now(),
            warnings,
        };

        Ok(GenerationResult {
            prediction,
            probability_result,
            timeline_estimate,
            evidence_links,
            pattern_analyses,
            metadata,
        })
    }
}

impl Default for PredictionGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate recency weight for a pattern based on its age.
fn calculate_recency_weight(age_days: i64) -> f64 {
    // Full weight for patterns less than 7 days old
    // Exponential decay with half-life of 30 days
    if age_days <= 7 {
        1.0
    } else {
        let decay_days = (age_days - 7) as f64;
        let half_life = 30.0;
        (-(decay_days / half_life) * std::f64::consts::LN_2).exp()
    }
}

/// Convenience function to generate a prediction.
pub fn generate_prediction(
    patterns: &[InputPattern],
    context: &GenerationContext,
) -> PredictionResult<GenerationResult> {
    PredictionGenerator::new().generate_prediction(patterns, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn create_test_patterns() -> Vec<InputPattern> {
        vec![
            InputPattern::new("p1", 0.9)
                .with_success_rate(0.8)
                .with_confidence(0.85)
                .with_effectiveness(0.9)
                .with_resolution_time(7),
            InputPattern::new("p2", 0.7)
                .with_success_rate(0.7)
                .with_confidence(0.75)
                .with_effectiveness(0.8)
                .with_resolution_time(14),
            InputPattern::new("p3", 0.6)
                .with_success_rate(0.6)
                .with_confidence(0.65)
                .with_effectiveness(0.7)
                .with_resolution_time(10),
        ]
    }

    #[test]
    fn test_generation_context() {
        let context = GenerationContext::new("Will the deployment succeed?")
            .with_domain("devops.deployment")
            .with_context("Production environment")
            .with_tag("critical")
            .with_min_confidence(0.5)
            .with_min_similarity(0.6);

        assert_eq!(context.query, "Will the deployment succeed?");
        assert_eq!(context.domain, "devops.deployment");
        assert_eq!(context.min_confidence, 0.5);
        assert_eq!(context.min_similarity, 0.6);
    }

    #[test]
    fn test_pattern_analysis_quality_score() {
        let analysis = PatternAnalysis {
            pattern_id: "p1".to_string(),
            similarity: 0.9,
            success_rate: 0.8,
            confidence: 0.9,
            effectiveness: 0.85,
            weight: 0.5,
            age_days: 5,
            is_relevant: true,
        };

        let score = analysis.quality_score();
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_pattern_analysis_recency_decay() {
        let mut recent = PatternAnalysis {
            pattern_id: "recent".to_string(),
            similarity: 0.8,
            success_rate: 0.7,
            confidence: 0.8,
            effectiveness: 0.75,
            weight: 0.5,
            age_days: 1,
            is_relevant: true,
        };

        let mut old = recent.clone();
        old.pattern_id = "old".to_string();
        old.age_days = 90;

        assert!(recent.quality_score() > old.quality_score());
    }

    #[test]
    fn test_input_pattern() {
        let pattern = InputPattern::new("test-pattern", 0.85)
            .with_success_rate(0.75)
            .with_confidence(0.8)
            .with_effectiveness(0.9)
            .with_resolution_time(14);

        assert_eq!(pattern.id, "test-pattern");
        assert!((pattern.similarity - 0.85).abs() < 0.001);
        assert!((pattern.success_rate - 0.75).abs() < 0.001);
        assert_eq!(pattern.resolution_time_days, Some(14));
    }

    #[test]
    fn test_prediction_generator() {
        let generator = PredictionGenerator::new().with_min_patterns(2);

        let patterns = create_test_patterns();
        let context = GenerationContext::new("Test prediction")
            .with_domain("testing")
            .with_min_similarity(0.5);

        let result = generator.generate_prediction(&patterns, &context).unwrap();

        assert!(!result.prediction.id().as_str().is_empty());
        assert!(result.prediction.probability() > 0.0);
        assert!(result.prediction.probability() <= 1.0);
        assert!(result.prediction.confidence() > 0.0);
        assert!(result.pattern_analyses.len() >= 2);
    }

    #[test]
    fn test_generate_prediction_insufficient_patterns() {
        let generator = PredictionGenerator::new().with_min_patterns(5);

        let patterns = vec![InputPattern::new("p1", 0.9).with_success_rate(0.8)];
        let context = GenerationContext::new("Test");

        let result = generator.generate_prediction(&patterns, &context);
        assert!(matches!(
            result,
            Err(PredictionError::InsufficientPatterns { .. })
        ));
    }

    #[test]
    fn test_generate_prediction_filters_by_similarity() {
        let generator = PredictionGenerator::new().with_min_patterns(1);

        let patterns = vec![
            InputPattern::new("low_sim", 0.3).with_success_rate(0.9),
            InputPattern::new("high_sim", 0.8).with_success_rate(0.7),
        ];

        let context = GenerationContext::new("Test").with_min_similarity(0.5);

        let result = generator.generate_prediction(&patterns, &context).unwrap();

        // Only high_sim pattern should be used
        assert_eq!(result.metadata.patterns_used, 1);
        assert!(result
            .pattern_analyses
            .iter()
            .all(|a| a.similarity >= 0.5));
    }

    #[test]
    fn test_recency_weight() {
        // Recent patterns should have higher weight
        let recent_weight = calculate_recency_weight(1);
        let week_old_weight = calculate_recency_weight(7);
        let month_old_weight = calculate_recency_weight(37); // 30 days after the 7-day grace period
        let old_weight = calculate_recency_weight(100);

        assert!((recent_weight - 1.0).abs() < 0.001);
        assert!((week_old_weight - 1.0).abs() < 0.001);
        assert!(month_old_weight < week_old_weight);
        assert!(old_weight < month_old_weight);
    }

    #[test]
    fn test_generation_metadata() {
        let patterns = create_test_patterns();
        let context = GenerationContext::new("Test").with_min_similarity(0.5);

        let result = generate_prediction(&patterns, &context).unwrap();

        assert!(result.metadata.patterns_considered >= result.metadata.patterns_used);
        assert!(result.metadata.duration_ms < 1000); // Should be fast
        assert!(result.metadata.avg_similarity > 0.0);
    }

    #[test]
    fn test_evidence_links_created() {
        let patterns = create_test_patterns();
        let context = GenerationContext::new("Test").with_min_similarity(0.5);

        let result = generate_prediction(&patterns, &context).unwrap();

        assert!(!result.evidence_links.is_empty());
        assert_eq!(
            result.evidence_links.len(),
            result.prediction.evidence_pattern_ids().len()
        );
    }
}
