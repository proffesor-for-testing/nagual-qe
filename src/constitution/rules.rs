//! Individual constitution rule implementations.

use super::*;
use chrono::Utc;

/// Rule 1: Never delete a pattern without a recent backup.
pub struct NeverDeleteWithoutBackup;

impl ConstitutionRule for NeverDeleteWithoutBackup {
    fn name(&self) -> &str { "NeverDeleteWithoutBackup" }

    fn check(&self, context: &OperationContext) -> CheckResult {
        let allowed = context.has_recent_backup;
        CheckResult {
            allowed,
            rule: self.name().to_string(),
            message: if allowed {
                "Recent backup exists, deletion permitted".to_string()
            } else {
                "No backup within 24 hours. Run 'nagual sync backup' first.".to_string()
            },
            severity: Severity::Error,
            checked_at: Utc::now(),
        }
    }

    fn applies_to(&self) -> Vec<Operation> {
        vec![Operation::Delete]
    }
}

/// Rule 2: Failure outcomes must include MAST classification.
pub struct AlwaysRecordMAST;

impl ConstitutionRule for AlwaysRecordMAST {
    fn name(&self) -> &str { "AlwaysRecordMAST" }

    fn check(&self, context: &OperationContext) -> CheckResult {
        let has_mast = context.failure_mode.is_some();
        let valid_modes = ["specification", "misalignment", "verification", "resource", "unknown"];
        let is_valid = context.failure_mode.as_ref()
            .map(|m| valid_modes.contains(&m.as_str()))
            .unwrap_or(false);

        let allowed = has_mast && is_valid;
        CheckResult {
            allowed,
            rule: self.name().to_string(),
            message: if allowed {
                format!("MAST classification: {}", context.failure_mode.as_deref().unwrap_or(""))
            } else if !has_mast {
                "Failure outcome must include MAST classification (specification, misalignment, verification, resource, unknown)".to_string()
            } else {
                format!("Invalid MAST mode: {:?}. Must be one of: specification, misalignment, verification, resource, unknown", context.failure_mode)
            },
            severity: Severity::Warning,
            checked_at: Utc::now(),
        }
    }

    fn applies_to(&self) -> Vec<Operation> {
        vec![Operation::RecordFailure]
    }
}

/// Rule 3: High-surprise patterns require review before consolidation.
pub struct SurpriseReview;

impl ConstitutionRule for SurpriseReview {
    fn name(&self) -> &str { "SurpriseReview" }

    fn check(&self, context: &OperationContext) -> CheckResult {
        let surprise = context.surprise_score.unwrap_or(0.0);
        let allowed = surprise <= 0.8;
        CheckResult {
            allowed,
            rule: self.name().to_string(),
            message: if allowed {
                format!("Surprise score {:.2} is within normal range", surprise)
            } else {
                format!("Surprise score {:.2} > 0.8 — pattern is highly novel. Flag for human review before auto-consolidation.", surprise)
            },
            severity: Severity::Warning,
            checked_at: Utc::now(),
        }
    }

    fn applies_to(&self) -> Vec<Operation> {
        vec![Operation::Consolidate]
    }
}

/// Rule 4: Conflicting patterns should escalate, not silently overwrite.
pub struct ConflictEscalation;

impl ConstitutionRule for ConflictEscalation {
    fn name(&self) -> &str { "ConflictEscalation" }

    fn check(&self, context: &OperationContext) -> CheckResult {
        // For overwrite operations, always warn (actual conflict detection
        // would require comparing old and new solutions)
        CheckResult {
            allowed: true, // Allow but warn
            rule: self.name().to_string(),
            message: format!(
                "Pattern {} overwrite — ensure conflicting solutions are resolved, not replaced",
                context.pattern_id.as_deref().unwrap_or("unknown")
            ),
            severity: Severity::Info,
            checked_at: Utc::now(),
        }
    }

    fn applies_to(&self) -> Vec<Operation> {
        vec![Operation::Overwrite, Operation::Consolidate]
    }
}

/// Rule 5: Only patterns with reward >= 0.9 can be in reflex tier.
pub struct MinimumRewardForReflex;

impl ConstitutionRule for MinimumRewardForReflex {
    fn name(&self) -> &str { "MinimumRewardForReflex" }

    fn check(&self, context: &OperationContext) -> CheckResult {
        let is_reflex = context.tier.as_deref() == Some("reflex");
        let reward = context.reward.unwrap_or(0.0);

        if !is_reflex {
            return CheckResult {
                allowed: true,
                rule: self.name().to_string(),
                message: "Not a reflex-tier promotion, rule does not apply".to_string(),
                severity: Severity::Info,
                checked_at: Utc::now(),
            };
        }

        let allowed = reward >= 0.9;
        CheckResult {
            allowed,
            rule: self.name().to_string(),
            message: if allowed {
                format!("Reward {:.2} meets reflex threshold (>= 0.9)", reward)
            } else {
                format!("Reward {:.2} below reflex threshold (0.9). Pattern cannot be promoted to reflex.", reward)
            },
            severity: Severity::Error,
            checked_at: Utc::now(),
        }
    }

    fn applies_to(&self) -> Vec<Operation> {
        vec![Operation::Promote]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context(op: Operation) -> OperationContext {
        OperationContext {
            operation: op,
            pattern_id: Some("test-id".to_string()),
            reward: None,
            tier: None,
            surprise_score: None,
            has_recent_backup: false,
            failure_mode: None,
            domain: None,
        }
    }

    #[test]
    fn test_never_delete_without_backup() {
        let rule = NeverDeleteWithoutBackup;

        let mut ctx = make_context(Operation::Delete);
        assert!(!rule.check(&ctx).allowed);

        ctx.has_recent_backup = true;
        assert!(rule.check(&ctx).allowed);
    }

    #[test]
    fn test_always_record_mast() {
        let rule = AlwaysRecordMAST;

        let mut ctx = make_context(Operation::RecordFailure);
        assert!(!rule.check(&ctx).allowed);

        ctx.failure_mode = Some("specification".to_string());
        assert!(rule.check(&ctx).allowed);

        ctx.failure_mode = Some("invalid".to_string());
        assert!(!rule.check(&ctx).allowed);
    }

    #[test]
    fn test_surprise_review() {
        let rule = SurpriseReview;

        let mut ctx = make_context(Operation::Consolidate);
        ctx.surprise_score = Some(0.5);
        assert!(rule.check(&ctx).allowed);

        ctx.surprise_score = Some(0.85);
        assert!(!rule.check(&ctx).allowed);
    }

    #[test]
    fn test_minimum_reward_for_reflex() {
        let rule = MinimumRewardForReflex;

        let mut ctx = make_context(Operation::Promote);
        ctx.tier = Some("reflex".to_string());
        ctx.reward = Some(0.95);
        assert!(rule.check(&ctx).allowed);

        ctx.reward = Some(0.85);
        assert!(!rule.check(&ctx).allowed);

        // Non-reflex should pass
        ctx.tier = Some("crystal".to_string());
        ctx.reward = Some(0.5);
        assert!(rule.check(&ctx).allowed);
    }
}
