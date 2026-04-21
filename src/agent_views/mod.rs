//! Per-Agent Views for the Knowledge Operating System (KOS P8).
//!
//! Provides multi-agent isolation with per-agent pattern views, bitmap membership
//! filters, and grant/revoke access control. Each agent can be configured with
//! a view mode that determines which patterns it can see:
//!
//! - `Include` mode: agent sees only explicitly granted patterns
//! - `Exclude` mode: agent sees all patterns except explicitly excluded ones
//! - `All` mode: agent sees everything (default)
//!
//! Domain-level isolation is also supported: agents can be restricted to specific
//! knowledge domains, preventing cross-domain information leakage in multi-agent
//! orchestration scenarios.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;

use crate::db::SqliteDb;
use crate::error::{NagualError, Result};

// ---------------------------------------------------------------------------
// ViewMode
// ---------------------------------------------------------------------------

/// Determines how an agent's pattern visibility is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewMode {
    /// Agent sees only explicitly granted patterns.
    Include,
    /// Agent sees all patterns except explicitly excluded ones.
    Exclude,
    /// Agent sees everything (default).
    All,
}

impl ViewMode {
    /// Return the string representation of this mode.
    pub fn as_str(&self) -> &'static str {
        match self {
            ViewMode::Include => "include",
            ViewMode::Exclude => "exclude",
            ViewMode::All => "all",
        }
    }
}

impl From<&str> for ViewMode {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "include" => ViewMode::Include,
            "exclude" => ViewMode::Exclude,
            _ => ViewMode::All,
        }
    }
}

impl std::fmt::Display for ViewMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// AgentView
// ---------------------------------------------------------------------------

/// A registered agent's view configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentView {
    /// Unique identifier for the agent.
    pub agent_id: String,
    /// How visibility is computed for this agent.
    pub view_mode: ViewMode,
    /// Domains this agent is allowed to access (empty = all domains when mode is All).
    pub domain_filters: Vec<String>,
    /// Explicitly granted pattern IDs (used in Include mode).
    pub pattern_grants: Vec<String>,
    /// Explicitly excluded pattern IDs (used in Exclude mode).
    pub pattern_excludes: Vec<String>,
    /// When this agent view was created.
    pub created_at: DateTime<Utc>,
    /// When this agent view was last updated.
    pub updated_at: DateTime<Utc>,
    /// Optional JSON metadata.
    pub metadata: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// ViewConfig
// ---------------------------------------------------------------------------

/// Global configuration for the agent views subsystem.
#[derive(Debug, Clone)]
pub struct ViewConfig {
    /// Default view mode for newly registered agents.
    pub default_mode: ViewMode,
    /// Whether domain isolation is enforced (if false, domain_filters are ignored).
    pub enable_domain_isolation: bool,
    /// Maximum number of pattern grants per agent.
    pub max_grants_per_agent: usize,
    /// Maximum number of registered agents.
    pub max_agents: usize,
}

impl Default for ViewConfig {
    fn default() -> Self {
        Self {
            default_mode: ViewMode::All,
            enable_domain_isolation: false,
            max_grants_per_agent: 10_000,
            max_agents: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// AccessDecision
// ---------------------------------------------------------------------------

/// The result of an access check for a single pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessDecision {
    /// Whether access is allowed.
    pub allowed: bool,
    /// Human-readable reason for the decision.
    pub reason: String,
    /// The agent that was checked.
    pub agent_id: String,
    /// The pattern that was checked.
    pub pattern_id: String,
}

// ---------------------------------------------------------------------------
// ViewStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about registered agent views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewStats {
    /// Total number of registered agents.
    pub total_agents: u64,
    /// Count of agents per view mode (e.g. {"all": 5, "include": 2}).
    pub agents_by_mode: HashMap<String, u64>,
    /// Total number of pattern grants across all agents.
    pub total_grants: u64,
    /// Total number of pattern excludes across all agents.
    pub total_excludes: u64,
}

// ---------------------------------------------------------------------------
// BitmapFilter
// ---------------------------------------------------------------------------

/// A set-based membership filter for pattern IDs.
///
/// Uses `HashSet` internally -- the "bitmap" name reflects the conceptual
/// operation (membership testing) rather than a literal bitmap data structure.
#[derive(Debug, Clone)]
pub struct BitmapFilter {
    inner: HashSet<String>,
}

impl BitmapFilter {
    /// Create an empty filter.
    pub fn new() -> Self {
        Self {
            inner: HashSet::new(),
        }
    }

    /// Add a pattern ID to the filter.
    pub fn add(&mut self, id: &str) {
        self.inner.insert(id.to_string());
    }

    /// Remove a pattern ID from the filter.
    pub fn remove(&mut self, id: &str) {
        self.inner.remove(id);
    }

    /// Check whether the filter contains a given pattern ID.
    pub fn contains(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    /// Return the number of entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check whether the filter is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over the contained pattern IDs.
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.inner.iter()
    }
}

impl Default for BitmapFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ViewManager
// ---------------------------------------------------------------------------

/// Manages per-agent views, grants, exclusions, and access decisions.
pub struct ViewManager {
    db: Arc<SqliteDb>,
    config: ViewConfig,
    cache: RwLock<HashMap<String, AgentView>>,
}

impl ViewManager {
    // -- Schema -----------------------------------------------------------------

    const CREATE_TABLES_SQL: &'static str = r#"
        CREATE TABLE IF NOT EXISTS agent_views (
            agent_id TEXT PRIMARY KEY,
            view_mode TEXT NOT NULL DEFAULT 'all',
            domain_filters TEXT DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_pattern_grants (
            agent_id TEXT NOT NULL,
            pattern_id TEXT NOT NULL,
            grant_type TEXT NOT NULL DEFAULT 'include',
            granted_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (agent_id, pattern_id, grant_type)
        );

        CREATE INDEX IF NOT EXISTS idx_grants_agent ON agent_pattern_grants(agent_id);
        CREATE INDEX IF NOT EXISTS idx_grants_pattern ON agent_pattern_grants(pattern_id);
        CREATE INDEX IF NOT EXISTS idx_grants_type ON agent_pattern_grants(grant_type);
    "#;

    // -- Construction -----------------------------------------------------------

    /// Create a new `ViewManager`, initialise the schema, and load existing views
    /// into the in-memory cache.
    pub async fn new(db: Arc<SqliteDb>, config: ViewConfig) -> Result<Self> {
        db.execute_batch(Self::CREATE_TABLES_SQL).await?;

        let manager = Self {
            db,
            config,
            cache: RwLock::new(HashMap::new()),
        };

        manager.reload_cache().await?;
        Ok(manager)
    }

    // -- Internal helpers -------------------------------------------------------

    /// Reload every agent view (with grants/excludes) from the database into the
    /// in-memory cache.
    async fn reload_cache(&self) -> Result<()> {
        let views = self.load_all_views().await?;
        let mut cache = self.cache.write();
        cache.clear();
        for view in views {
            cache.insert(view.agent_id.clone(), view);
        }
        Ok(())
    }

    /// Load all `AgentView` rows from the database, including their grants and
    /// excludes.
    async fn load_all_views(&self) -> Result<Vec<AgentView>> {
        let rows: Vec<(String, String, String, String, String, Option<String>)> = self
            .db
            .query(
                "SELECT agent_id, view_mode, domain_filters, created_at, updated_at, metadata FROM agent_views",
                &[],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .await?;

        let mut views = Vec::with_capacity(rows.len());
        for (agent_id, mode_str, domains_json, created_str, updated_str, meta_str) in rows {
            let grants = self.load_grants(&agent_id, "include").await?;
            let excludes = self.load_grants(&agent_id, "exclude").await?;
            let domains: Vec<String> =
                serde_json::from_str(&domains_json).unwrap_or_default();
            let created_at = parse_datetime(&created_str);
            let updated_at = parse_datetime(&updated_str);
            let metadata = meta_str.and_then(|s| serde_json::from_str(&s).ok());

            views.push(AgentView {
                agent_id,
                view_mode: ViewMode::from(mode_str.as_str()),
                domain_filters: domains,
                pattern_grants: grants,
                pattern_excludes: excludes,
                created_at,
                updated_at,
                metadata,
            });
        }

        Ok(views)
    }

    /// Load grant/exclude pattern IDs for a specific agent and type.
    async fn load_grants(&self, agent_id: &str, grant_type: &str) -> Result<Vec<String>> {
        let ids: Vec<String> = self
            .db
            .query(
                "SELECT pattern_id FROM agent_pattern_grants WHERE agent_id = ? AND grant_type = ?",
                &[&agent_id as &dyn rusqlite::ToSql, &grant_type],
                |row| row.get(0),
            )
            .await?;
        Ok(ids)
    }

    /// Touch the `updated_at` timestamp for an agent.
    async fn touch_updated(&self, agent_id: &str) -> Result<()> {
        self.db
            .execute(
                "UPDATE agent_views SET updated_at = datetime('now') WHERE agent_id = ?",
                &[&agent_id as &dyn rusqlite::ToSql],
            )
            .await?;
        Ok(())
    }

    // -- Public API -------------------------------------------------------------

    /// Register a new agent with the given view mode.
    ///
    /// Returns the newly created `AgentView`. If the agent already exists an
    /// error is returned.
    pub async fn register_agent(
        &self,
        agent_id: &str,
        mode: ViewMode,
    ) -> Result<AgentView> {
        // Check max agents limit.
        {
            let cache = self.cache.read();
            if cache.len() >= self.config.max_agents {
                return Err(NagualError::internal(format!(
                    "Maximum number of agents ({}) reached",
                    self.config.max_agents
                )));
            }
            if cache.contains_key(agent_id) {
                return Err(NagualError::internal(format!(
                    "Agent '{}' is already registered",
                    agent_id
                )));
            }
        }

        let now = Utc::now();
        let now_str = now.format("%Y-%m-%d %H:%M:%S").to_string();

        self.db
            .execute(
                "INSERT INTO agent_views (agent_id, view_mode, domain_filters, created_at, updated_at) VALUES (?, ?, '[]', ?, ?)",
                &[
                    &agent_id as &dyn rusqlite::ToSql,
                    &mode.as_str(),
                    &now_str,
                    &now_str,
                ],
            )
            .await?;

        let view = AgentView {
            agent_id: agent_id.to_string(),
            view_mode: mode,
            domain_filters: Vec::new(),
            pattern_grants: Vec::new(),
            pattern_excludes: Vec::new(),
            created_at: now,
            updated_at: now,
            metadata: None,
        };

        self.cache.write().insert(agent_id.to_string(), view.clone());
        Ok(view)
    }

    /// Retrieve an agent's current view configuration.
    pub async fn get_view(&self, agent_id: &str) -> Result<Option<AgentView>> {
        let cache = self.cache.read();
        Ok(cache.get(agent_id).cloned())
    }

    /// Grant an agent access to a specific pattern (used in Include mode).
    pub async fn grant_access(
        &self,
        agent_id: &str,
        pattern_id: &str,
    ) -> Result<()> {
        self.ensure_agent_exists(agent_id)?;
        self.check_grants_limit(agent_id)?;

        // INSERT OR IGNORE for idempotency.
        self.db
            .execute(
                "INSERT OR IGNORE INTO agent_pattern_grants (agent_id, pattern_id, grant_type) VALUES (?, ?, 'include')",
                &[&agent_id as &dyn rusqlite::ToSql, &pattern_id],
            )
            .await?;

        self.touch_updated(agent_id).await?;

        // Update cache.
        {
            let mut cache = self.cache.write();
            if let Some(view) = cache.get_mut(agent_id) {
                if !view.pattern_grants.contains(&pattern_id.to_string()) {
                    view.pattern_grants.push(pattern_id.to_string());
                }
                view.updated_at = Utc::now();
            }
        }

        Ok(())
    }

    /// Revoke a previously granted pattern from an agent.
    pub async fn revoke_access(
        &self,
        agent_id: &str,
        pattern_id: &str,
    ) -> Result<()> {
        self.ensure_agent_exists(agent_id)?;

        self.db
            .execute(
                "DELETE FROM agent_pattern_grants WHERE agent_id = ? AND pattern_id = ? AND grant_type = 'include'",
                &[&agent_id as &dyn rusqlite::ToSql, &pattern_id],
            )
            .await?;

        self.touch_updated(agent_id).await?;

        {
            let mut cache = self.cache.write();
            if let Some(view) = cache.get_mut(agent_id) {
                view.pattern_grants.retain(|id| id != pattern_id);
                view.updated_at = Utc::now();
            }
        }

        Ok(())
    }

    /// Add a pattern to an agent's exclusion list (used in Exclude mode).
    pub async fn exclude_pattern(
        &self,
        agent_id: &str,
        pattern_id: &str,
    ) -> Result<()> {
        self.ensure_agent_exists(agent_id)?;

        self.db
            .execute(
                "INSERT OR IGNORE INTO agent_pattern_grants (agent_id, pattern_id, grant_type) VALUES (?, ?, 'exclude')",
                &[&agent_id as &dyn rusqlite::ToSql, &pattern_id],
            )
            .await?;

        self.touch_updated(agent_id).await?;

        {
            let mut cache = self.cache.write();
            if let Some(view) = cache.get_mut(agent_id) {
                if !view.pattern_excludes.contains(&pattern_id.to_string()) {
                    view.pattern_excludes.push(pattern_id.to_string());
                }
                view.updated_at = Utc::now();
            }
        }

        Ok(())
    }

    /// Remove a pattern from an agent's exclusion list.
    pub async fn remove_exclude(
        &self,
        agent_id: &str,
        pattern_id: &str,
    ) -> Result<()> {
        self.ensure_agent_exists(agent_id)?;

        self.db
            .execute(
                "DELETE FROM agent_pattern_grants WHERE agent_id = ? AND pattern_id = ? AND grant_type = 'exclude'",
                &[&agent_id as &dyn rusqlite::ToSql, &pattern_id],
            )
            .await?;

        self.touch_updated(agent_id).await?;

        {
            let mut cache = self.cache.write();
            if let Some(view) = cache.get_mut(agent_id) {
                view.pattern_excludes.retain(|id| id != pattern_id);
                view.updated_at = Utc::now();
            }
        }

        Ok(())
    }

    /// Set the domain filter list for an agent.
    pub async fn set_domain_filter(
        &self,
        agent_id: &str,
        domains: Vec<String>,
    ) -> Result<()> {
        self.ensure_agent_exists(agent_id)?;

        let domains_json = serde_json::to_string(&domains)
            .map_err(|e| NagualError::internal(format!("Failed to serialize domains: {e}")))?;

        self.db
            .execute(
                "UPDATE agent_views SET domain_filters = ?, updated_at = datetime('now') WHERE agent_id = ?",
                &[&domains_json as &dyn rusqlite::ToSql, &agent_id],
            )
            .await?;

        {
            let mut cache = self.cache.write();
            if let Some(view) = cache.get_mut(agent_id) {
                view.domain_filters = domains;
                view.updated_at = Utc::now();
            }
        }

        Ok(())
    }

    /// Check whether `agent_id` is allowed to access `pattern_id`.
    ///
    /// If the agent is not registered, access is denied.
    pub async fn check_access(
        &self,
        agent_id: &str,
        pattern_id: &str,
        pattern_domain: Option<&str>,
    ) -> Result<AccessDecision> {
        let cache = self.cache.read();
        let view = match cache.get(agent_id) {
            Some(v) => v,
            None => {
                return Ok(AccessDecision {
                    allowed: false,
                    reason: format!("Agent '{}' is not registered", agent_id),
                    agent_id: agent_id.to_string(),
                    pattern_id: pattern_id.to_string(),
                });
            }
        };

        // Domain isolation check.
        if self.config.enable_domain_isolation && !view.domain_filters.is_empty() {
            if let Some(domain) = pattern_domain {
                let domain_allowed = view.domain_filters.iter().any(|d| {
                    domain == d.as_str() || domain.starts_with(&format!("{}.", d))
                });
                if !domain_allowed {
                    return Ok(AccessDecision {
                        allowed: false,
                        reason: format!(
                            "Pattern domain '{}' is not in agent's allowed domains",
                            domain
                        ),
                        agent_id: agent_id.to_string(),
                        pattern_id: pattern_id.to_string(),
                    });
                }
            }
        }

        // Mode-based check.
        match view.view_mode {
            ViewMode::All => Ok(AccessDecision {
                allowed: true,
                reason: "Agent has All mode - full access".to_string(),
                agent_id: agent_id.to_string(),
                pattern_id: pattern_id.to_string(),
            }),
            ViewMode::Include => {
                let granted = view.pattern_grants.contains(&pattern_id.to_string());
                Ok(AccessDecision {
                    allowed: granted,
                    reason: if granted {
                        "Pattern is explicitly granted".to_string()
                    } else {
                        "Pattern is not in agent's grant list (Include mode)".to_string()
                    },
                    agent_id: agent_id.to_string(),
                    pattern_id: pattern_id.to_string(),
                })
            }
            ViewMode::Exclude => {
                let excluded = view.pattern_excludes.contains(&pattern_id.to_string());
                Ok(AccessDecision {
                    allowed: !excluded,
                    reason: if excluded {
                        "Pattern is explicitly excluded".to_string()
                    } else {
                        "Pattern is not in agent's exclude list (Exclude mode)".to_string()
                    },
                    agent_id: agent_id.to_string(),
                    pattern_id: pattern_id.to_string(),
                })
            }
        }
    }

    /// Filter a list of pattern IDs based on an agent's view configuration.
    ///
    /// Returns only those pattern IDs that the agent is allowed to see.
    /// The `domains` slice must be parallel to `pattern_ids` -- `domains[i]` is
    /// the domain of `pattern_ids[i]`. If a domain is unknown, pass an empty string.
    pub async fn filter_patterns(
        &self,
        agent_id: &str,
        pattern_ids: &[String],
        domains: &[String],
    ) -> Result<Vec<String>> {
        let cache = self.cache.read();
        let view = match cache.get(agent_id) {
            Some(v) => v,
            None => return Ok(Vec::new()),
        };

        // Build fast lookup sets.
        let grants_set: HashSet<&str> =
            view.pattern_grants.iter().map(|s| s.as_str()).collect();
        let excludes_set: HashSet<&str> =
            view.pattern_excludes.iter().map(|s| s.as_str()).collect();
        let domain_set: HashSet<&str> =
            view.domain_filters.iter().map(|s| s.as_str()).collect();

        let mut result = Vec::new();

        for (i, pid) in pattern_ids.iter().enumerate() {
            // Domain check.
            if self.config.enable_domain_isolation && !domain_set.is_empty() {
                let d = domains.get(i).map(|s| s.as_str()).unwrap_or("");
                if !d.is_empty() {
                    let domain_ok = domain_set.iter().any(|&allowed| {
                        d == allowed || d.starts_with(&format!("{}.", allowed))
                    });
                    if !domain_ok {
                        continue;
                    }
                }
            }

            // Mode check.
            let allowed = match view.view_mode {
                ViewMode::All => true,
                ViewMode::Include => grants_set.contains(pid.as_str()),
                ViewMode::Exclude => !excludes_set.contains(pid.as_str()),
            };

            if allowed {
                result.push(pid.clone());
            }
        }

        Ok(result)
    }

    /// Change an agent's view mode.
    pub async fn set_view_mode(
        &self,
        agent_id: &str,
        mode: ViewMode,
    ) -> Result<()> {
        self.ensure_agent_exists(agent_id)?;

        self.db
            .execute(
                "UPDATE agent_views SET view_mode = ?, updated_at = datetime('now') WHERE agent_id = ?",
                &[&mode.as_str() as &dyn rusqlite::ToSql, &agent_id],
            )
            .await?;

        {
            let mut cache = self.cache.write();
            if let Some(view) = cache.get_mut(agent_id) {
                view.view_mode = mode;
                view.updated_at = Utc::now();
            }
        }

        Ok(())
    }

    /// Delete an agent and all associated grants/excludes.
    pub async fn delete_agent(&self, agent_id: &str) -> Result<()> {
        self.db
            .execute(
                "DELETE FROM agent_pattern_grants WHERE agent_id = ?",
                &[&agent_id as &dyn rusqlite::ToSql],
            )
            .await?;

        self.db
            .execute(
                "DELETE FROM agent_views WHERE agent_id = ?",
                &[&agent_id as &dyn rusqlite::ToSql],
            )
            .await?;

        self.cache.write().remove(agent_id);
        Ok(())
    }

    /// List all registered agent views.
    pub async fn list_agents(&self) -> Result<Vec<AgentView>> {
        let cache = self.cache.read();
        Ok(cache.values().cloned().collect())
    }

    /// Aggregate statistics about registered agents.
    pub async fn stats(&self) -> Result<ViewStats> {
        let cache = self.cache.read();
        let mut agents_by_mode: HashMap<String, u64> = HashMap::new();
        let mut total_grants: u64 = 0;
        let mut total_excludes: u64 = 0;

        for view in cache.values() {
            *agents_by_mode
                .entry(view.view_mode.as_str().to_string())
                .or_insert(0) += 1;
            total_grants += view.pattern_grants.len() as u64;
            total_excludes += view.pattern_excludes.len() as u64;
        }

        Ok(ViewStats {
            total_agents: cache.len() as u64,
            agents_by_mode,
            total_grants,
            total_excludes,
        })
    }

    // -- Private helpers --------------------------------------------------------

    fn ensure_agent_exists(&self, agent_id: &str) -> Result<()> {
        let cache = self.cache.read();
        if !cache.contains_key(agent_id) {
            return Err(NagualError::internal(format!(
                "Agent '{}' is not registered",
                agent_id
            )));
        }
        Ok(())
    }

    fn check_grants_limit(&self, agent_id: &str) -> Result<()> {
        let cache = self.cache.read();
        if let Some(view) = cache.get(agent_id) {
            let total = view.pattern_grants.len() + view.pattern_excludes.len();
            if total >= self.config.max_grants_per_agent {
                return Err(NagualError::internal(format!(
                    "Agent '{}' has reached the maximum number of grants ({})",
                    agent_id, self.config.max_grants_per_agent
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: parse a datetime string from SQLite into `DateTime<Utc>`.
// ---------------------------------------------------------------------------

fn parse_datetime(s: &str) -> DateTime<Utc> {
    chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
        .map(|naive| naive.and_utc())
        .unwrap_or_else(|_| Utc::now())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    async fn setup_test_db() -> Arc<SqliteDb> {
        let db = SqliteDb::open_in_memory().unwrap();
        Arc::new(db)
    }

    // -----------------------------------------------------------------------
    // ViewMode tests (3)
    // -----------------------------------------------------------------------

    #[test]
    fn test_view_mode_as_str() {
        assert_eq!(ViewMode::Include.as_str(), "include");
        assert_eq!(ViewMode::Exclude.as_str(), "exclude");
        assert_eq!(ViewMode::All.as_str(), "all");
    }

    #[test]
    fn test_view_mode_from_str() {
        assert_eq!(ViewMode::from("include"), ViewMode::Include);
        assert_eq!(ViewMode::from("exclude"), ViewMode::Exclude);
        assert_eq!(ViewMode::from("all"), ViewMode::All);
        assert_eq!(ViewMode::from("INCLUDE"), ViewMode::Include);
        assert_eq!(ViewMode::from("Exclude"), ViewMode::Exclude);
        // Unknown string defaults to All.
        assert_eq!(ViewMode::from("unknown"), ViewMode::All);
    }

    #[test]
    fn test_view_mode_equality() {
        let a = ViewMode::Include;
        let b = ViewMode::Include;
        let c = ViewMode::Exclude;
        assert_eq!(a, b);
        assert_ne!(a, c);

        // Copy semantics.
        let d = a;
        assert_eq!(a, d);
    }

    // -----------------------------------------------------------------------
    // BitmapFilter tests (4)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bitmap_filter_add_remove() {
        let mut f = BitmapFilter::new();
        f.add("p1");
        f.add("p2");
        assert_eq!(f.len(), 2);

        f.remove("p1");
        assert_eq!(f.len(), 1);
        assert!(!f.contains("p1"));
        assert!(f.contains("p2"));
    }

    #[test]
    fn test_bitmap_filter_contains() {
        let mut f = BitmapFilter::new();
        assert!(!f.contains("p1"));
        f.add("p1");
        assert!(f.contains("p1"));
        assert!(!f.contains("p2"));
    }

    #[test]
    fn test_bitmap_filter_len_empty() {
        let f = BitmapFilter::new();
        assert_eq!(f.len(), 0);
        assert!(f.is_empty());

        let mut f2 = BitmapFilter::new();
        f2.add("x");
        assert_eq!(f2.len(), 1);
        assert!(!f2.is_empty());
    }

    #[test]
    fn test_bitmap_filter_iter() {
        let mut f = BitmapFilter::new();
        f.add("a");
        f.add("b");
        f.add("c");

        let mut items: Vec<String> = f.iter().cloned().collect();
        items.sort();
        assert_eq!(items, vec!["a", "b", "c"]);
    }

    // -----------------------------------------------------------------------
    // ViewConfig tests (2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_view_config_defaults() {
        let config = ViewConfig::default();
        assert_eq!(config.default_mode, ViewMode::All);
        assert!(!config.enable_domain_isolation);
        assert_eq!(config.max_grants_per_agent, 10_000);
        assert_eq!(config.max_agents, 100);
    }

    #[test]
    fn test_view_config_custom() {
        let config = ViewConfig {
            default_mode: ViewMode::Include,
            enable_domain_isolation: true,
            max_grants_per_agent: 500,
            max_agents: 10,
        };
        assert_eq!(config.default_mode, ViewMode::Include);
        assert!(config.enable_domain_isolation);
        assert_eq!(config.max_grants_per_agent, 500);
        assert_eq!(config.max_agents, 10);
    }

    // -----------------------------------------------------------------------
    // register_agent tests (3)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_register_agent_new() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        let view = mgr.register_agent("agent-1", ViewMode::Include).await.unwrap();
        assert_eq!(view.agent_id, "agent-1");
        assert_eq!(view.view_mode, ViewMode::Include);
        assert!(view.domain_filters.is_empty());
        assert!(view.pattern_grants.is_empty());
        assert!(view.pattern_excludes.is_empty());
    }

    #[tokio::test]
    async fn test_register_agent_duplicate() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::All).await.unwrap();
        let result = mgr.register_agent("agent-1", ViewMode::All).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("already registered"));
    }

    #[tokio::test]
    async fn test_register_agent_max_agents() {
        let db = setup_test_db().await;
        let config = ViewConfig {
            max_agents: 2,
            ..Default::default()
        };
        let mgr = ViewManager::new(db, config).await.unwrap();

        mgr.register_agent("a1", ViewMode::All).await.unwrap();
        mgr.register_agent("a2", ViewMode::All).await.unwrap();
        let result = mgr.register_agent("a3", ViewMode::All).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Maximum number of agents"));
    }

    // -----------------------------------------------------------------------
    // grant/revoke tests (3)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_grant_access() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Include).await.unwrap();
        mgr.grant_access("agent-1", "pat-100").await.unwrap();

        let view = mgr.get_view("agent-1").await.unwrap().unwrap();
        assert!(view.pattern_grants.contains(&"pat-100".to_string()));
    }

    #[tokio::test]
    async fn test_revoke_access() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Include).await.unwrap();
        mgr.grant_access("agent-1", "pat-100").await.unwrap();
        mgr.revoke_access("agent-1", "pat-100").await.unwrap();

        let view = mgr.get_view("agent-1").await.unwrap().unwrap();
        assert!(!view.pattern_grants.contains(&"pat-100".to_string()));
    }

    #[tokio::test]
    async fn test_grant_access_idempotent() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Include).await.unwrap();
        mgr.grant_access("agent-1", "pat-100").await.unwrap();
        mgr.grant_access("agent-1", "pat-100").await.unwrap();

        let view = mgr.get_view("agent-1").await.unwrap().unwrap();
        // Should only appear once.
        assert_eq!(
            view.pattern_grants.iter().filter(|p| *p == "pat-100").count(),
            1
        );
    }

    // -----------------------------------------------------------------------
    // exclude tests (2)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_exclude_pattern() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Exclude).await.unwrap();
        mgr.exclude_pattern("agent-1", "pat-secret").await.unwrap();

        let view = mgr.get_view("agent-1").await.unwrap().unwrap();
        assert!(view.pattern_excludes.contains(&"pat-secret".to_string()));
    }

    #[tokio::test]
    async fn test_remove_exclude() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Exclude).await.unwrap();
        mgr.exclude_pattern("agent-1", "pat-secret").await.unwrap();
        mgr.remove_exclude("agent-1", "pat-secret").await.unwrap();

        let view = mgr.get_view("agent-1").await.unwrap().unwrap();
        assert!(!view.pattern_excludes.contains(&"pat-secret".to_string()));
    }

    // -----------------------------------------------------------------------
    // check_access Include mode tests (3)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_check_access_include_granted() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Include).await.unwrap();
        mgr.grant_access("agent-1", "pat-1").await.unwrap();

        let decision = mgr.check_access("agent-1", "pat-1", None).await.unwrap();
        assert!(decision.allowed);
        assert!(decision.reason.contains("granted"));
    }

    #[tokio::test]
    async fn test_check_access_include_denied() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Include).await.unwrap();

        let decision = mgr.check_access("agent-1", "pat-999", None).await.unwrap();
        assert!(!decision.allowed);
        assert!(decision.reason.contains("not in agent's grant list"));
    }

    #[tokio::test]
    async fn test_check_access_include_domain_filter() {
        let db = setup_test_db().await;
        let config = ViewConfig {
            enable_domain_isolation: true,
            ..Default::default()
        };
        let mgr = ViewManager::new(db, config).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Include).await.unwrap();
        mgr.grant_access("agent-1", "pat-1").await.unwrap();
        mgr.set_domain_filter("agent-1", vec!["rust".to_string()])
            .await
            .unwrap();

        // Pattern in allowed domain -- access depends on grant.
        let decision = mgr.check_access("agent-1", "pat-1", Some("rust")).await.unwrap();
        assert!(decision.allowed);

        // Pattern in disallowed domain -- blocked by domain filter.
        let decision = mgr
            .check_access("agent-1", "pat-1", Some("python"))
            .await
            .unwrap();
        assert!(!decision.allowed);
        assert!(decision.reason.contains("domain"));
    }

    // -----------------------------------------------------------------------
    // check_access Exclude mode tests (2)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_check_access_exclude_denied() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Exclude).await.unwrap();
        mgr.exclude_pattern("agent-1", "pat-secret").await.unwrap();

        let decision = mgr
            .check_access("agent-1", "pat-secret", None)
            .await
            .unwrap();
        assert!(!decision.allowed);
        assert!(decision.reason.contains("excluded"));
    }

    #[tokio::test]
    async fn test_check_access_exclude_allowed() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Exclude).await.unwrap();
        mgr.exclude_pattern("agent-1", "pat-secret").await.unwrap();

        let decision = mgr
            .check_access("agent-1", "pat-public", None)
            .await
            .unwrap();
        assert!(decision.allowed);
        assert!(decision.reason.contains("not in agent's exclude list"));
    }

    // -----------------------------------------------------------------------
    // check_access All mode tests (1)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_check_access_all_mode() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::All).await.unwrap();

        let decision = mgr
            .check_access("agent-1", "any-pattern", None)
            .await
            .unwrap();
        assert!(decision.allowed);
        assert!(decision.reason.contains("All mode"));
    }

    // -----------------------------------------------------------------------
    // filter_patterns tests (2)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_filter_patterns_include_mode() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Include).await.unwrap();
        mgr.grant_access("agent-1", "p1").await.unwrap();
        mgr.grant_access("agent-1", "p3").await.unwrap();

        let ids = vec![
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
            "p4".to_string(),
        ];
        let domains = vec![String::new(); 4];

        let filtered = mgr.filter_patterns("agent-1", &ids, &domains).await.unwrap();
        assert_eq!(filtered, vec!["p1".to_string(), "p3".to_string()]);
    }

    #[tokio::test]
    async fn test_filter_patterns_exclude_mode() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Exclude).await.unwrap();
        mgr.exclude_pattern("agent-1", "p2").await.unwrap();

        let ids = vec![
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
        ];
        let domains = vec![String::new(); 3];

        let filtered = mgr.filter_patterns("agent-1", &ids, &domains).await.unwrap();
        assert_eq!(filtered, vec!["p1".to_string(), "p3".to_string()]);
    }

    // -----------------------------------------------------------------------
    // set_domain_filter tests (2)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_set_domain_filter_isolation() {
        let db = setup_test_db().await;
        let config = ViewConfig {
            enable_domain_isolation: true,
            ..Default::default()
        };
        let mgr = ViewManager::new(db, config).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::All).await.unwrap();
        mgr.set_domain_filter("agent-1", vec!["rust".to_string()])
            .await
            .unwrap();

        // Allowed domain.
        let d = mgr.check_access("agent-1", "p1", Some("rust")).await.unwrap();
        assert!(d.allowed);

        // Subdomain also allowed.
        let d = mgr
            .check_access("agent-1", "p1", Some("rust.async"))
            .await
            .unwrap();
        assert!(d.allowed);

        // Disallowed domain.
        let d = mgr
            .check_access("agent-1", "p1", Some("python"))
            .await
            .unwrap();
        assert!(!d.allowed);
    }

    #[tokio::test]
    async fn test_set_domain_filter_multi_domain() {
        let db = setup_test_db().await;
        let config = ViewConfig {
            enable_domain_isolation: true,
            ..Default::default()
        };
        let mgr = ViewManager::new(db, config).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::All).await.unwrap();
        mgr.set_domain_filter(
            "agent-1",
            vec!["rust".to_string(), "database".to_string()],
        )
        .await
        .unwrap();

        let d1 = mgr.check_access("agent-1", "p1", Some("rust")).await.unwrap();
        assert!(d1.allowed);

        let d2 = mgr
            .check_access("agent-1", "p1", Some("database"))
            .await
            .unwrap();
        assert!(d2.allowed);

        let d3 = mgr
            .check_access("agent-1", "p1", Some("python"))
            .await
            .unwrap();
        assert!(!d3.allowed);
    }

    // -----------------------------------------------------------------------
    // delete_agent tests (1)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_delete_agent() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("agent-1", ViewMode::Include).await.unwrap();
        mgr.grant_access("agent-1", "p1").await.unwrap();
        mgr.exclude_pattern("agent-1", "p2").await.unwrap();

        mgr.delete_agent("agent-1").await.unwrap();

        let view = mgr.get_view("agent-1").await.unwrap();
        assert!(view.is_none());

        // Stats should reflect removal.
        let s = mgr.stats().await.unwrap();
        assert_eq!(s.total_agents, 0);
    }

    // -----------------------------------------------------------------------
    // list_agents and stats tests (2)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_list_agents() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("a1", ViewMode::All).await.unwrap();
        mgr.register_agent("a2", ViewMode::Include).await.unwrap();
        mgr.register_agent("a3", ViewMode::Exclude).await.unwrap();

        let mut agents = mgr.list_agents().await.unwrap();
        agents.sort_by(|a, b| a.agent_id.cmp(&b.agent_id));
        assert_eq!(agents.len(), 3);
        assert_eq!(agents[0].agent_id, "a1");
        assert_eq!(agents[1].agent_id, "a2");
        assert_eq!(agents[2].agent_id, "a3");
    }

    #[tokio::test]
    async fn test_stats() {
        let db = setup_test_db().await;
        let mgr = ViewManager::new(db, ViewConfig::default()).await.unwrap();

        mgr.register_agent("a1", ViewMode::All).await.unwrap();
        mgr.register_agent("a2", ViewMode::Include).await.unwrap();
        mgr.register_agent("a3", ViewMode::Include).await.unwrap();

        mgr.grant_access("a2", "p1").await.unwrap();
        mgr.grant_access("a2", "p2").await.unwrap();
        mgr.grant_access("a3", "p3").await.unwrap();
        mgr.exclude_pattern("a1", "px").await.unwrap();

        let s = mgr.stats().await.unwrap();
        assert_eq!(s.total_agents, 3);
        assert_eq!(*s.agents_by_mode.get("all").unwrap_or(&0), 1);
        assert_eq!(*s.agents_by_mode.get("include").unwrap_or(&0), 2);
        assert_eq!(s.total_grants, 3);
        assert_eq!(s.total_excludes, 1);
    }
}
