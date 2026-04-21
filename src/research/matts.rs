//! MaTTS: Memory-aware Test-Time Scaling for research quality
//!
//! Implements parallel trajectory execution and attention-weighted consensus
//! aggregation for high-quality research results.

use std::collections::HashMap;
use tracing::{debug, info};

use super::types::*;

/// MaTTS configuration
#[derive(Debug, Clone)]
pub struct MaTTSConfig {
    /// Number of parallel trajectories (k)
    pub k_trajectories: usize,
    /// Quality threshold for early stopping
    pub early_stop_threshold: f64,
    /// Minimum agreement between trajectories
    pub min_agreement: f64,
    /// Temperature for attention weighting
    pub attention_temperature: f64,
}

impl Default for MaTTSConfig {
    fn default() -> Self {
        Self {
            k_trajectories: 3,
            early_stop_threshold: 0.85,
            min_agreement: 0.6,
            attention_temperature: 1.0,
        }
    }
}

impl MaTTSConfig {
    pub fn quick() -> Self {
        Self {
            k_trajectories: 1,
            early_stop_threshold: 0.7,
            min_agreement: 0.5,
            attention_temperature: 1.0,
        }
    }

    pub fn deep() -> Self {
        Self {
            k_trajectories: 5,
            early_stop_threshold: 0.9,
            min_agreement: 0.7,
            attention_temperature: 0.8,
        }
    }
}

/// MaTTS consensus engine
pub struct MaTTS {
    config: MaTTSConfig,
}

impl MaTTS {
    pub fn new(config: MaTTSConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(MaTTSConfig::default())
    }

    /// Aggregate multiple trajectories into consensus using attention-weighted voting
    pub fn aggregate(&self, trajectories: &[ResearchTrajectory]) -> ConsensusResult {
        if trajectories.is_empty() {
            return ConsensusResult {
                summary: "No research results available".into(),
                confidence: 0.0,
                key_findings: vec![],
                sources: vec![],
                agreement_score: 0.0,
            };
        }

        info!(
            "Aggregating {} trajectories with MaTTS",
            trajectories.len()
        );

        // Calculate attention weights based on quality scores
        let weights = self.calculate_attention_weights(trajectories);
        debug!("Attention weights: {:?}", weights);

        // Collect all findings with their weighted scores
        let mut finding_scores: HashMap<String, f64> = HashMap::new();
        let mut all_sources: Vec<String> = Vec::new();

        for (traj, &weight) in trajectories.iter().zip(weights.iter()) {
            for finding in &traj.findings {
                let normalized_content = finding.content.trim().to_lowercase();
                *finding_scores.entry(normalized_content).or_insert(0.0) +=
                    weight * finding.confidence;

                if !finding.source.is_empty() && !all_sources.contains(&finding.source) {
                    all_sources.push(finding.source.clone());
                }
            }
        }

        // Sort findings by weighted score
        let mut sorted_findings: Vec<_> = finding_scores.into_iter().collect();
        sorted_findings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top findings
        let key_findings: Vec<String> = sorted_findings
            .iter()
            .take(5)
            .map(|(content, _)| content.clone())
            .collect();

        // Calculate agreement score (how much trajectories agree)
        let agreement_score = self.calculate_agreement(trajectories);

        // Calculate overall confidence
        let avg_quality: f64 =
            trajectories.iter().map(|t| t.quality_score).sum::<f64>() / trajectories.len() as f64;
        let confidence = (avg_quality * 0.6 + agreement_score * 0.4).clamp(0.0, 1.0);

        // Generate summary from top findings
        let summary = if key_findings.is_empty() {
            "No significant findings from research".to_string()
        } else {
            key_findings.join("\n\n")
        };

        info!(
            "MaTTS consensus: confidence={:.2}, agreement={:.2}, findings={}",
            confidence,
            agreement_score,
            key_findings.len()
        );

        ConsensusResult {
            summary,
            confidence,
            key_findings,
            sources: all_sources,
            agreement_score,
        }
    }

    /// Calculate attention weights using softmax with temperature
    fn calculate_attention_weights(&self, trajectories: &[ResearchTrajectory]) -> Vec<f64> {
        if trajectories.is_empty() {
            return vec![];
        }

        if trajectories.len() == 1 {
            return vec![1.0];
        }

        // Apply temperature scaling to quality scores
        let scaled_scores: Vec<f64> = trajectories
            .iter()
            .map(|t| t.quality_score / self.config.attention_temperature)
            .collect();

        // Softmax normalization
        let max_score = scaled_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scaled_scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        exp_scores.iter().map(|e| e / sum_exp).collect()
    }

    /// Calculate agreement between trajectories using Jaccard similarity
    fn calculate_agreement(&self, trajectories: &[ResearchTrajectory]) -> f64 {
        if trajectories.len() < 2 {
            return 1.0; // Single trajectory has perfect self-agreement
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        // Compare each pair of trajectories
        for i in 0..trajectories.len() {
            for j in (i + 1)..trajectories.len() {
                let sim = self.trajectory_similarity(&trajectories[i], &trajectories[j]);
                total_similarity += sim;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate similarity between two trajectories using Jaccard on findings
    fn trajectory_similarity(&self, a: &ResearchTrajectory, b: &ResearchTrajectory) -> f64 {
        let a_findings: std::collections::HashSet<_> = a
            .findings
            .iter()
            .map(|f| f.content.trim().to_lowercase())
            .collect();

        let b_findings: std::collections::HashSet<_> = b
            .findings
            .iter()
            .map(|f| f.content.trim().to_lowercase())
            .collect();

        if a_findings.is_empty() && b_findings.is_empty() {
            return 1.0;
        }

        let intersection = a_findings.intersection(&b_findings).count();
        let union = a_findings.union(&b_findings).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    /// Check if early stopping conditions are met
    pub fn should_early_stop(&self, trajectories: &[ResearchTrajectory]) -> bool {
        if trajectories.is_empty() {
            return false;
        }

        // Check if any trajectory exceeds quality threshold
        let has_high_quality = trajectories
            .iter()
            .any(|t| t.quality_score >= self.config.early_stop_threshold);

        if has_high_quality && trajectories.len() >= 2 {
            // Also check agreement
            let agreement = self.calculate_agreement(trajectories);
            if agreement >= self.config.min_agreement {
                debug!(
                    "Early stopping: high quality with agreement {:.2}",
                    agreement
                );
                return true;
            }
        }

        false
    }

    /// Self-contrast: compare finding with its negation to assess confidence
    pub fn self_contrast_score(&self, finding: &str) -> f64 {
        // Simple heuristic: findings with specific details score higher
        let specificity_markers = [
            "specifically",
            "example",
            "such as",
            "e.g.",
            "i.e.",
            "because",
            "when",
            "where",
            "how",
        ];

        let has_specificity = specificity_markers
            .iter()
            .any(|m| finding.to_lowercase().contains(m));

        let has_code = finding.contains("```") || finding.contains("`");
        let has_numbers = finding.chars().any(|c| c.is_ascii_digit());

        let mut score: f64 = 0.5;
        if has_specificity {
            score += 0.2;
        }
        if has_code {
            score += 0.2;
        }
        if has_numbers {
            score += 0.1;
        }

        score.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(agent_type: AgentType, findings: Vec<&str>, quality: f64) -> ResearchTrajectory {
        let mut traj = ResearchTrajectory::new(agent_type);
        for f in findings {
            traj.add_finding(ResearchFinding::new(f, "test").with_confidence(0.8));
        }
        traj.complete(quality);
        traj
    }

    #[test]
    fn test_single_trajectory_aggregation() {
        let matts = MaTTS::with_defaults();
        let traj = make_trajectory(AgentType::WebSearch, vec!["Finding 1", "Finding 2"], 0.8);

        let result = matts.aggregate(&[traj]);

        assert!(result.confidence > 0.0);
        assert!(!result.key_findings.is_empty());
        assert_eq!(result.agreement_score, 1.0); // Single trajectory
    }

    #[test]
    fn test_multiple_trajectory_consensus() {
        let matts = MaTTS::with_defaults();

        let traj1 = make_trajectory(
            AgentType::WebSearch,
            vec!["Common finding", "Unique to 1"],
            0.8,
        );
        let traj2 = make_trajectory(
            AgentType::DocFetch,
            vec!["Common finding", "Unique to 2"],
            0.7,
        );
        let traj3 = make_trajectory(
            AgentType::KnowledgeBase,
            vec!["Common finding", "Unique to 3"],
            0.6,
        );

        let result = matts.aggregate(&[traj1, traj2, traj3]);

        // Common finding should be ranked highest
        assert!(result.key_findings[0].contains("common finding"));
        assert!(result.agreement_score > 0.0);
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let matts = MaTTS::with_defaults();

        let trajectories = vec![
            make_trajectory(AgentType::WebSearch, vec!["A"], 0.9),
            make_trajectory(AgentType::DocFetch, vec!["B"], 0.5),
            make_trajectory(AgentType::KnowledgeBase, vec!["C"], 0.3),
        ];

        let weights = matts.calculate_attention_weights(&trajectories);
        let sum: f64 = weights.iter().sum();

        assert!((sum - 1.0).abs() < 0.001);
        // Higher quality should have higher weight
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
    }

    #[test]
    fn test_early_stopping() {
        let matts = MaTTS::new(MaTTSConfig {
            early_stop_threshold: 0.8,
            min_agreement: 0.5,
            ..Default::default()
        });

        let traj1 = make_trajectory(AgentType::WebSearch, vec!["Shared"], 0.9);
        let traj2 = make_trajectory(AgentType::DocFetch, vec!["Shared"], 0.85);

        assert!(matts.should_early_stop(&[traj1, traj2]));
    }

    #[test]
    fn test_self_contrast_scoring() {
        let matts = MaTTS::with_defaults();

        let generic = "This is good";
        let specific = "For example, use `tokio::spawn` when you need concurrent execution";

        assert!(matts.self_contrast_score(specific) > matts.self_contrast_score(generic));
    }
}
