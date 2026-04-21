//! Built-in actions for knowledge work

use super::types::*;

/// Get the default set of actions for nagual
pub fn default_actions() -> Vec<Action> {
    let mut actions = Vec::new();

    // Research actions
    actions.extend(research_actions());

    // Knowledge actions
    actions.extend(knowledge_actions());

    // Analysis actions
    actions.extend(analysis_actions());

    // Testing actions
    actions.extend(testing_actions());

    // Improvement actions
    actions.extend(improvement_actions());

    // Consolidation actions
    actions.extend(consolidation_actions());

    actions
}

/// Research-related actions
pub fn research_actions() -> Vec<Action> {
    vec![
        Action::new("identify_research_topic", "Identify Research Topic")
            .with_description("Determine what topic needs to be researched")
            .with_effect(Effect::set_true("research_target_defined"))
            .with_cost(0.5)
            .with_duration(10)
            .with_category(ActionCategory::Research),

        Action::new("web_search", "Web Search")
            .with_description("Search the web for information on a topic")
            .with_precondition(Condition::is_true("research_target_defined"))
            .with_effect(Effect::set_true("web_results_available"))
            .with_cost(2.0)
            .with_duration(30)
            .with_category(ActionCategory::Research),

        Action::new("fetch_documentation", "Fetch Documentation")
            .with_description("Fetch official documentation for a topic")
            .with_precondition(Condition::is_true("research_target_defined"))
            .with_effect(Effect::set_true("documentation_fetched"))
            .with_cost(1.5)
            .with_duration(20)
            .with_category(ActionCategory::Research),

        Action::new("analyze_examples", "Analyze Code Examples")
            .with_description("Find and analyze code examples")
            .with_precondition(Condition::is_true("research_target_defined"))
            .with_effect(Effect::set_true("examples_analyzed"))
            .with_cost(2.0)
            .with_duration(45)
            .with_category(ActionCategory::Research),

        Action::new("synthesize_research", "Synthesize Research Results")
            .with_description("Combine research findings into actionable knowledge")
            .with_precondition(Condition::is_true("web_results_available"))
            .with_effect(Effect::set_true("research_synthesized"))
            .with_effect(Effect::set_true("has_research_results"))
            .with_cost(1.0)
            .with_duration(15)
            .with_category(ActionCategory::Research),
    ]
}

/// Knowledge management actions
pub fn knowledge_actions() -> Vec<Action> {
    vec![
        Action::new("store_pattern", "Store Pattern")
            .with_description("Store learned knowledge as a pattern")
            .with_precondition(Condition::is_true("has_research_results"))
            .with_effect(Effect::set_true("pattern_stored"))
            .with_effect(Effect::increment("pattern_count", 1.0))
            .with_cost(1.0)
            .with_duration(5)
            .with_category(ActionCategory::Knowledge),

        Action::new("update_pattern", "Update Existing Pattern")
            .with_description("Update an existing pattern with new information")
            .with_precondition(Condition::is_true("pattern_identified"))
            .with_precondition(Condition::is_true("has_research_results"))
            .with_effect(Effect::set_true("pattern_updated"))
            .with_cost(1.0)
            .with_duration(5)
            .with_category(ActionCategory::Knowledge),

        Action::new("tag_pattern", "Add Tags to Pattern")
            .with_description("Add categorization tags to a pattern")
            .with_precondition(Condition::is_true("pattern_stored"))
            .with_effect(Effect::set_true("pattern_tagged"))
            .with_cost(0.5)
            .with_duration(3)
            .with_category(ActionCategory::Knowledge),

        Action::new("link_patterns", "Link Related Patterns")
            .with_description("Create graph edges between related patterns")
            .with_precondition(Condition::is_true("pattern_stored"))
            .with_effect(Effect::set_true("patterns_linked"))
            .with_cost(0.5)
            .with_duration(5)
            .with_category(ActionCategory::Knowledge),

        Action::new("generate_embedding", "Generate Embedding")
            .with_description("Generate vector embedding for semantic search")
            .with_precondition(Condition::is_true("pattern_stored"))
            .with_effect(Effect::set_true("embedding_generated"))
            .with_cost(0.5)
            .with_duration(2)
            .with_category(ActionCategory::Knowledge),
    ]
}

/// Analysis actions
pub fn analysis_actions() -> Vec<Action> {
    vec![
        Action::new("analyze_codebase", "Analyze Codebase")
            .with_description("Analyze code structure and patterns")
            .with_precondition(Condition::is_true("codebase_path_known"))
            .with_effect(Effect::set_true("codebase_analyzed"))
            .with_cost(3.0)
            .with_duration(120)
            .with_category(ActionCategory::Analysis),

        Action::new("identify_patterns", "Identify Code Patterns")
            .with_description("Find recurring patterns in codebase")
            .with_precondition(Condition::is_true("codebase_analyzed"))
            .with_effect(Effect::set_true("code_patterns_identified"))
            .with_cost(2.0)
            .with_duration(60)
            .with_category(ActionCategory::Analysis),

        Action::new("detect_gaps", "Detect Knowledge Gaps")
            .with_description("Find areas with low pattern coverage")
            .with_effect(Effect::set_true("gaps_identified"))
            .with_effect(Effect::set_true("research_target_defined"))
            .with_cost(1.5)
            .with_duration(30)
            .with_category(ActionCategory::Analysis),

        Action::new("analyze_quality", "Analyze Pattern Quality")
            .with_description("Assess quality and effectiveness of patterns")
            .with_effect(Effect::set_true("quality_analyzed"))
            .with_cost(1.0)
            .with_duration(20)
            .with_category(ActionCategory::Analysis),

        Action::new("introspect", "Run Self-Introspection")
            .with_description("Analyze system health and recommendations")
            .with_effect(Effect::set_true("introspection_complete"))
            .with_effect(Effect::set_true("recommendations_available"))
            .with_cost(1.5)
            .with_duration(30)
            .with_category(ActionCategory::Analysis),
    ]
}

/// Testing actions
pub fn testing_actions() -> Vec<Action> {
    vec![
        Action::new("run_tests", "Run Tests")
            .with_description("Execute test suite")
            .with_precondition(Condition::is_true("tests_exist"))
            .with_effect(Effect::set_true("tests_executed"))
            .with_cost(2.0)
            .with_duration(60)
            .with_category(ActionCategory::Testing),

        Action::new("verify_pattern", "Verify Pattern Effectiveness")
            .with_description("Test if a pattern works as expected")
            .with_precondition(Condition::is_true("pattern_stored"))
            .with_effect(Effect::set_true("pattern_verified"))
            .with_cost(1.5)
            .with_duration(30)
            .with_category(ActionCategory::Testing),

        Action::new("validate_coherence", "Validate Knowledge Coherence")
            .with_description("Check for contradictions in knowledge base")
            .with_effect(Effect::set_true("coherence_validated"))
            .with_cost(1.0)
            .with_duration(15)
            .with_category(ActionCategory::Testing),
    ]
}

/// Improvement actions
pub fn improvement_actions() -> Vec<Action> {
    vec![
        Action::new("improve_domain", "Improve Domain Knowledge")
            .with_description("Run self-improvement for a knowledge domain")
            .with_precondition(Condition::is_true("domain_identified"))
            .with_effect(Effect::set_true("domain_improved"))
            .with_cost(2.5)
            .with_duration(90)
            .with_category(ActionCategory::Improvement),

        Action::new("record_success", "Record Success Outcome")
            .with_description("Record successful use of a pattern")
            .with_precondition(Condition::is_true("pattern_identified"))
            .with_effect(Effect::set_true("outcome_recorded"))
            .with_effect(Effect::increment("reward", 0.1))
            .with_cost(0.5)
            .with_duration(3)
            .with_category(ActionCategory::Improvement),

        Action::new("record_failure", "Record Failure Outcome")
            .with_description("Record failed use with MAST classification")
            .with_precondition(Condition::is_true("pattern_identified"))
            .with_effect(Effect::set_true("outcome_recorded"))
            .with_effect(Effect::set_true("failure_classified"))
            .with_cost(0.5)
            .with_duration(5)
            .with_category(ActionCategory::Improvement),

        Action::new("refresh_stale", "Refresh Stale Patterns")
            .with_description("Update patterns that have become outdated")
            .with_precondition(Condition::is_true("stale_patterns_identified"))
            .with_effect(Effect::set_true("patterns_refreshed"))
            .with_cost(3.0)
            .with_duration(120)
            .with_category(ActionCategory::Improvement),
    ]
}

/// Consolidation actions
pub fn consolidation_actions() -> Vec<Action> {
    vec![
        Action::new("consolidate_similar", "Consolidate Similar Patterns")
            .with_description("Merge similar patterns to reduce duplication")
            .with_precondition(Condition::greater_than("pattern_count", 10.0))
            .with_effect(Effect::set_true("patterns_consolidated"))
            .with_cost(2.0)
            .with_duration(60)
            .with_category(ActionCategory::Consolidation),

        Action::new("deduplicate", "Deduplicate Patterns")
            .with_description("Remove exact duplicate patterns")
            .with_effect(Effect::set_true("duplicates_removed"))
            .with_cost(1.0)
            .with_duration(30)
            .with_category(ActionCategory::Consolidation),

        Action::new("archive_low_quality", "Archive Low-Quality Patterns")
            .with_description("Archive patterns below quality threshold")
            .with_precondition(Condition::is_true("quality_analyzed"))
            .with_effect(Effect::set_true("low_quality_archived"))
            .with_cost(1.0)
            .with_duration(15)
            .with_category(ActionCategory::Consolidation),

        Action::new("prune_orphans", "Prune Orphan Patterns")
            .with_description("Remove patterns with no graph connections")
            .with_effect(Effect::set_true("orphans_pruned"))
            .with_cost(0.5)
            .with_duration(10)
            .with_category(ActionCategory::Consolidation),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_actions_not_empty() {
        let actions = default_actions();
        assert!(!actions.is_empty());
    }

    #[test]
    fn test_all_actions_have_ids() {
        let actions = default_actions();
        for action in actions {
            assert!(!action.id.is_empty(), "Action should have an ID");
            assert!(!action.name.is_empty(), "Action should have a name");
        }
    }

    #[test]
    fn test_action_categories() {
        let actions = default_actions();
        let categories: std::collections::HashSet<_> = actions
            .iter()
            .map(|a| a.category)
            .collect();

        // Should have multiple categories
        assert!(categories.len() >= 4);
    }
}
