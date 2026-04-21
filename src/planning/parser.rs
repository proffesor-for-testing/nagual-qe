//! Natural language goal parsing

use tracing::debug;

use super::types::*;
use super::planner::PlanningError;

/// Parser for natural language goals
pub struct GoalParser;

impl GoalParser {
    /// Parse a natural language goal description into a Goal struct
    pub fn parse(description: &str) -> Result<Goal, PlanningError> {
        if description.trim().is_empty() {
            return Err(PlanningError::GoalParseError("Empty goal description".into()));
        }

        let description_lower = description.to_lowercase();

        // Detect goal type and generate appropriate conditions
        let conditions = Self::infer_conditions(&description_lower);

        let mut goal = Goal::from_description(description);
        goal.conditions = conditions;

        // Infer priority from urgency words
        goal.priority = Self::infer_priority(&description_lower);

        debug!(
            "Parsed goal '{}' with {} conditions, priority {}",
            description,
            goal.conditions.len(),
            goal.priority
        );

        Ok(goal)
    }

    /// Infer conditions from goal description
    fn infer_conditions(description: &str) -> Vec<Condition> {
        let mut conditions = Vec::new();

        // Research-related goals
        if description.contains("research") || description.contains("learn about") ||
           description.contains("understand") || description.contains("find out") {
            conditions.push(Condition::is_true("has_research_results"));
        }

        // Knowledge storage goals
        if description.contains("store") || description.contains("save") ||
           description.contains("remember") || description.contains("document") {
            conditions.push(Condition::is_true("pattern_stored"));
        }

        // Coverage/improvement goals
        if description.contains("coverage") || description.contains("improve") ||
           description.contains("increase") || description.contains("better") {
            conditions.push(Condition::is_true("domain_improved"));
        }

        // Gap detection goals
        if description.contains("gap") || description.contains("missing") ||
           description.contains("find what") || description.contains("identify") {
            conditions.push(Condition::is_true("gaps_identified"));
        }

        // Consolidation goals
        if description.contains("consolidate") || description.contains("merge") ||
           description.contains("deduplicate") || description.contains("clean") {
            conditions.push(Condition::is_true("patterns_consolidated"));
        }

        // Quality goals
        if description.contains("quality") || description.contains("review") ||
           description.contains("assess") || description.contains("evaluate") {
            conditions.push(Condition::is_true("quality_analyzed"));
        }

        // Testing goals
        if description.contains("test") || description.contains("verify") ||
           description.contains("validate") || description.contains("check") {
            conditions.push(Condition::is_true("tests_executed"));
        }

        // Analysis goals
        if description.contains("analyze") || description.contains("examine") ||
           description.contains("inspect") || description.contains("investigate") {
            conditions.push(Condition::is_true("codebase_analyzed"));
        }

        // Introspection goals
        if description.contains("health") || description.contains("status") ||
           description.contains("introspect") || description.contains("self") {
            conditions.push(Condition::is_true("introspection_complete"));
        }

        // If no conditions inferred, default to research + store
        if conditions.is_empty() {
            conditions.push(Condition::is_true("has_research_results"));
            conditions.push(Condition::is_true("pattern_stored"));
        }

        conditions
    }

    /// Infer priority from urgency words
    fn infer_priority(description: &str) -> u8 {
        if description.contains("urgent") || description.contains("critical") ||
           description.contains("asap") || description.contains("immediately") {
            return 10;
        }

        if description.contains("important") || description.contains("high priority") ||
           description.contains("soon") {
            return 8;
        }

        if description.contains("when possible") || description.contains("eventually") ||
           description.contains("low priority") {
            return 3;
        }

        // Default priority
        5
    }

    /// Extract domain from goal description
    pub fn extract_domain(description: &str) -> Option<String> {
        let description_lower = description.to_lowercase();

        // Common programming domains
        let domains = [
            ("rust", "rust"),
            ("python", "python"),
            ("javascript", "javascript"),
            ("typescript", "typescript"),
            ("go ", "go"),
            ("golang", "go"),
            ("java ", "java"),
            ("async", "async"),
            ("database", "database"),
            ("api", "api"),
            ("testing", "testing"),
            ("security", "security"),
            ("performance", "performance"),
            ("error handling", "error-handling"),
            ("logging", "logging"),
            ("authentication", "auth"),
            ("caching", "caching"),
        ];

        for (keyword, domain) in domains {
            if description_lower.contains(keyword) {
                return Some(domain.to_string());
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_research_goal() {
        let goal = GoalParser::parse("Research best practices for error handling").unwrap();

        assert!(!goal.conditions.is_empty());
        assert!(goal.conditions.iter().any(|c| c.proposition == "has_research_results"));
    }

    #[test]
    fn test_parse_urgent_goal() {
        let goal = GoalParser::parse("Urgently fix the security vulnerability").unwrap();

        assert_eq!(goal.priority, 10);
    }

    #[test]
    fn test_parse_low_priority_goal() {
        let goal = GoalParser::parse("When possible, clean up the documentation").unwrap();

        assert_eq!(goal.priority, 3);
    }

    #[test]
    fn test_extract_domain() {
        assert_eq!(
            GoalParser::extract_domain("Learn about Rust async patterns"),
            Some("rust".to_string())
        );

        assert_eq!(
            GoalParser::extract_domain("Improve database query performance"),
            Some("database".to_string())
        );

        assert_eq!(
            GoalParser::extract_domain("Generic improvement"),
            None
        );
    }

    #[test]
    fn test_empty_description() {
        let result = GoalParser::parse("");
        assert!(matches!(result, Err(PlanningError::GoalParseError(_))));
    }
}
