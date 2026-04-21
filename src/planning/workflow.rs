//! YAML-based workflow definitions for deterministic swarm orchestration.
//!
//! Workflows define multi-step pipelines where each step specifies:
//! - An agent type to spawn
//! - A prompt/task description
//! - Preconditions (which prior steps must succeed)
//! - Approval gates (optional human review points)
//! - Failure handling (retry, skip, abort)

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

/// A complete workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    /// Workflow name (used as strategy cache key).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Domain for strategy cache storage.
    pub domain: String,
    /// Ordered list of steps.
    pub steps: Vec<WorkflowStep>,
    /// Global timeout in seconds (default: 3600).
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Variables available to all steps.
    #[serde(default)]
    pub variables: HashMap<String, String>,
}

fn default_timeout() -> u64 {
    3600
}

/// A single step in a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Step identifier (unique within workflow).
    pub id: String,
    /// Agent type to use (e.g., "researcher", "coder", "tester").
    pub agent_type: String,
    /// Task prompt for the agent.
    pub prompt: String,
    /// Steps that must complete successfully before this one runs.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Whether this step requires human approval before proceeding.
    #[serde(default)]
    pub approval_gate: bool,
    /// Failure handling policy.
    #[serde(default)]
    pub on_failure: FailurePolicy,
    /// Step-specific timeout override (seconds).
    pub timeout_secs: Option<u64>,
    /// Whether this step runs in background.
    #[serde(default)]
    pub background: bool,
}

/// What to do when a step fails.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum FailurePolicy {
    /// Abort the entire workflow.
    #[default]
    Abort,
    /// Retry the step (up to max_retries).
    Retry { max_retries: u32 },
    /// Skip this step and continue.
    Skip,
}

/// Helper struct for deserializing the `retry` variant from YAML maps.
#[derive(Deserialize)]
struct RetryConfig {
    max_retries: u32,
}

impl Serialize for FailurePolicy {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            FailurePolicy::Abort => serializer.serialize_str("abort"),
            FailurePolicy::Skip => serializer.serialize_str("skip"),
            FailurePolicy::Retry { max_retries } => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                let mut inner = HashMap::new();
                inner.insert("max_retries", *max_retries);
                map.serialize_entry("retry", &inner)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for FailurePolicy {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de;

        struct FailurePolicyVisitor;

        impl<'de> de::Visitor<'de> for FailurePolicyVisitor {
            type Value = FailurePolicy;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("\"abort\", \"skip\", or a map with key \"retry\"")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v {
                    "abort" => Ok(FailurePolicy::Abort),
                    "skip" => Ok(FailurePolicy::Skip),
                    other => Err(E::custom(format!("unknown failure policy: {other}"))),
                }
            }

            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let key: String = map
                    .next_key()?
                    .ok_or_else(|| de::Error::custom("expected key \"retry\""))?;
                if key == "retry" {
                    let config: RetryConfig = map.next_value()?;
                    Ok(FailurePolicy::Retry {
                        max_retries: config.max_retries,
                    })
                } else {
                    Err(de::Error::custom(format!(
                        "unknown failure policy key: {key}"
                    )))
                }
            }
        }

        deserializer.deserialize_any(FailurePolicyVisitor)
    }
}

/// Runtime state of a workflow execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowExecution {
    pub workflow_name: String,
    pub started_at: String,
    pub status: WorkflowStatus,
    pub step_results: HashMap<String, StepResult>,
}

/// Overall status of a workflow execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStatus {
    Running,
    Completed,
    Failed,
    WaitingApproval { step_id: String },
}

/// Result of a single step execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub step_id: String,
    pub status: StepStatus,
    pub output: Option<String>,
    pub duration_secs: u64,
}

/// Status of a single step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

impl WorkflowDefinition {
    /// Load a workflow from a YAML file.
    pub fn from_yaml_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }

    /// Parse a workflow from a YAML string.
    pub fn from_yaml(yaml: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let workflow: WorkflowDefinition = serde_yaml::from_str(yaml)?;
        Ok(workflow)
    }

    /// Validate the workflow definition.
    ///
    /// Checks for:
    /// - Dependencies referencing unknown step IDs
    /// - Dependency cycles (via topological sort)
    /// - Duplicate step IDs
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        let step_ids: HashSet<_> = self.steps.iter().map(|s| s.id.as_str()).collect();

        // Check for duplicate step IDs
        if step_ids.len() != self.steps.len() {
            errors.push("Duplicate step IDs found".to_string());
        }

        // Check for missing dependencies
        for step in &self.steps {
            for dep in &step.depends_on {
                if !step_ids.contains(dep.as_str()) {
                    errors.push(format!(
                        "Step '{}' depends on unknown step '{}'",
                        step.id, dep
                    ));
                }
            }
        }

        // Check for cycles using Kahn's algorithm (topological sort)
        if errors.is_empty() {
            let mut in_degree: HashMap<&str, usize> = HashMap::new();
            let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

            for step in &self.steps {
                in_degree.entry(step.id.as_str()).or_insert(0);
                adj.entry(step.id.as_str()).or_default();
                for dep in &step.depends_on {
                    adj.entry(dep.as_str()).or_default().push(step.id.as_str());
                    *in_degree.entry(step.id.as_str()).or_insert(0) += 1;
                }
            }

            let mut queue: VecDeque<&str> = in_degree
                .iter()
                .filter(|(_, &deg)| deg == 0)
                .map(|(&id, _)| id)
                .collect();

            let mut visited = 0usize;
            while let Some(node) = queue.pop_front() {
                visited += 1;
                if let Some(neighbors) = adj.get(node) {
                    for &neighbor in neighbors {
                        if let Some(deg) = in_degree.get_mut(neighbor) {
                            *deg -= 1;
                            if *deg == 0 {
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }
            }

            if visited != self.steps.len() {
                errors.push("Workflow contains a dependency cycle".to_string());
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get steps that can run now (all dependencies satisfied).
    pub fn ready_steps(&self, completed: &HashSet<String>) -> Vec<&WorkflowStep> {
        self.steps
            .iter()
            .filter(|s| !completed.contains(&s.id))
            .filter(|s| s.depends_on.iter().all(|dep| completed.contains(dep)))
            .collect()
    }

    /// Convert to strategy cache format for EGUR storage.
    pub fn to_strategy_steps(&self) -> String {
        self.steps
            .iter()
            .map(|s| format!("{}:{}", s.agent_type, s.id))
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_yaml() -> &'static str {
        r#"
name: release-pipeline
description: "Standard release pipeline: test, audit, build, verify"
domain: release

steps:
  - id: run-tests
    agent_type: tester
    prompt: "Run cargo test --lib and report results"
    on_failure:
      retry:
        max_retries: 2

  - id: security-audit
    agent_type: security-scanner
    prompt: "Run cargo audit and report vulnerabilities"
    depends_on: [run-tests]
    on_failure: skip

  - id: build-release
    agent_type: coder
    prompt: "Build release binary with cargo build --release"
    depends_on: [run-tests]

  - id: integration-tests
    agent_type: tester
    prompt: "Run integration tests against the release binary"
    depends_on: [build-release]
    approval_gate: true

  - id: deploy-staging
    agent_type: coder
    prompt: "Deploy to staging environment"
    depends_on: [integration-tests, security-audit]
"#
    }

    fn make_step(id: &str, agent_type: &str, deps: Vec<&str>) -> WorkflowStep {
        WorkflowStep {
            id: id.to_string(),
            agent_type: agent_type.to_string(),
            prompt: format!("Do {}", id),
            depends_on: deps.into_iter().map(String::from).collect(),
            approval_gate: false,
            on_failure: FailurePolicy::default(),
            timeout_secs: None,
            background: false,
        }
    }

    fn make_workflow(steps: Vec<WorkflowStep>) -> WorkflowDefinition {
        WorkflowDefinition {
            name: "test-workflow".to_string(),
            description: "Test".to_string(),
            domain: "test".to_string(),
            steps,
            timeout_secs: 3600,
            variables: HashMap::new(),
        }
    }

    #[test]
    fn test_workflow_parse_yaml() {
        let wf = WorkflowDefinition::from_yaml(sample_yaml()).expect("should parse");
        assert_eq!(wf.name, "release-pipeline");
        assert_eq!(wf.domain, "release");
        assert_eq!(wf.steps.len(), 5);
        assert_eq!(wf.timeout_secs, 3600); // default
        assert_eq!(wf.steps[0].id, "run-tests");
        assert_eq!(wf.steps[0].agent_type, "tester");
        assert_eq!(
            wf.steps[0].on_failure,
            FailurePolicy::Retry { max_retries: 2 }
        );
        assert_eq!(wf.steps[1].on_failure, FailurePolicy::Skip);
        assert_eq!(wf.steps[2].on_failure, FailurePolicy::Abort); // default
        assert!(wf.steps[3].approval_gate);
        assert_eq!(wf.steps[4].depends_on, vec!["integration-tests", "security-audit"]);
    }

    #[test]
    fn test_workflow_validate_missing_dep() {
        let wf = make_workflow(vec![make_step("b", "coder", vec!["a"])]);
        let err = wf.validate().unwrap_err();
        assert!(err[0].contains("unknown step 'a'"));
    }

    #[test]
    fn test_workflow_validate_valid() {
        let wf = make_workflow(vec![
            make_step("a", "tester", vec![]),
            make_step("b", "coder", vec!["a"]),
            make_step("c", "tester", vec!["a", "b"]),
        ]);
        assert!(wf.validate().is_ok());
    }

    #[test]
    fn test_workflow_ready_steps_initial() {
        let wf = WorkflowDefinition::from_yaml(sample_yaml()).unwrap();
        let completed = HashSet::new();
        let ready = wf.ready_steps(&completed);
        // Only run-tests has no dependencies
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "run-tests");
    }

    #[test]
    fn test_workflow_ready_steps_after_completion() {
        let wf = WorkflowDefinition::from_yaml(sample_yaml()).unwrap();
        let mut completed = HashSet::new();
        completed.insert("run-tests".to_string());
        let ready = wf.ready_steps(&completed);
        // security-audit and build-release both depend only on run-tests
        let ids: HashSet<&str> = ready.iter().map(|s| s.id.as_str()).collect();
        assert!(ids.contains("security-audit"));
        assert!(ids.contains("build-release"));
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn test_workflow_failure_policy_default() {
        assert_eq!(FailurePolicy::default(), FailurePolicy::Abort);
    }

    #[test]
    fn test_workflow_to_strategy_steps() {
        let wf = make_workflow(vec![
            make_step("a", "tester", vec![]),
            make_step("b", "coder", vec!["a"]),
        ]);
        assert_eq!(wf.to_strategy_steps(), "tester:a,coder:b");
    }

    #[test]
    fn test_workflow_execution_status_serialization() {
        let exec = WorkflowExecution {
            workflow_name: "test".to_string(),
            started_at: "2026-01-01T00:00:00Z".to_string(),
            status: WorkflowStatus::WaitingApproval {
                step_id: "deploy".to_string(),
            },
            step_results: HashMap::new(),
        };
        let json = serde_json::to_string(&exec).unwrap();
        assert!(json.contains("waiting_approval"));
        assert!(json.contains("deploy"));
        let deser: WorkflowExecution = serde_json::from_str(&json).unwrap();
        assert_eq!(
            deser.status,
            WorkflowStatus::WaitingApproval {
                step_id: "deploy".to_string()
            }
        );
    }

    #[test]
    fn test_workflow_step_status_serialization() {
        let result = StepResult {
            step_id: "build".to_string(),
            status: StepStatus::Completed,
            output: Some("success".to_string()),
            duration_secs: 42,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"completed\""));
        let deser: StepResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.status, StepStatus::Completed);
        assert_eq!(deser.duration_secs, 42);
    }

    #[test]
    fn test_workflow_from_yaml_with_variables() {
        let yaml = r#"
name: param-workflow
description: "Workflow with variables"
domain: test
timeout_secs: 1800
variables:
  target_env: staging
  version: "1.2.3"
steps:
  - id: deploy
    agent_type: coder
    prompt: "Deploy version ${version} to ${target_env}"
"#;
        let wf = WorkflowDefinition::from_yaml(yaml).unwrap();
        assert_eq!(wf.timeout_secs, 1800);
        assert_eq!(wf.variables.get("target_env").unwrap(), "staging");
        assert_eq!(wf.variables.get("version").unwrap(), "1.2.3");
        assert_eq!(wf.variables.len(), 2);
    }
}
