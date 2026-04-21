//! Coherence Gate engine for belief consistency verification
//!
//! Uses embedding-based cosine similarity for semantic contradiction detection,
//! with BeliefGraph persistence and PatternStorage integration.

use std::sync::Arc;

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

use super::types::*;
use crate::db::SqliteDb;
use crate::error::NagualError;
use crate::ml::cosine_similarity;
#[cfg(feature = "onnx-embed")]
use crate::ml::{Embedder, EmbedderConfig};

/// Coherence Gate for verifying belief consistency
pub struct CoherenceGate {
    db: Arc<SqliteDb>,
    config: CoherenceConfig,
    /// Optional embedder for semantic similarity
    #[cfg(feature = "onnx-embed")]
    embedder: Option<Arc<Embedder>>,
    /// Cached belief graph (loaded from DB)
    belief_graph: BeliefGraph,
}

impl CoherenceGate {
    /// Create a new coherence gate without embedder
    pub fn new(db: Arc<SqliteDb>, config: CoherenceConfig) -> Self {
        Self {
            db,
            config,
            #[cfg(feature = "onnx-embed")]
            embedder: None,
            belief_graph: BeliefGraph::new(),
        }
    }

    /// Create with an embedder for semantic similarity
    #[cfg(feature = "onnx-embed")]
    pub fn with_embedder(db: Arc<SqliteDb>, config: CoherenceConfig, embedder: Arc<Embedder>) -> Self {
        Self {
            db,
            config,
            embedder: Some(embedder),
            belief_graph: BeliefGraph::new(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(db: Arc<SqliteDb>) -> Self {
        Self::new(db, CoherenceConfig::default())
    }

    /// Create with configuration and belief graph loaded from database.
    /// Automatically loads embedder if model file exists.
    pub async fn with_persisted_config(db: Arc<SqliteDb>) -> Result<Self, NagualError> {
        Self::init_schema(&db).await?;
        let config = Self::load_config(&db).await?.unwrap_or_default();
        let belief_graph = Self::load_belief_graph(&db).await?;

        #[cfg(feature = "onnx-embed")]
        let embedder = Self::try_load_embedder();
        #[cfg(feature = "onnx-embed")]
        if embedder.is_some() {
            info!("Coherence gate initialized with embedder for semantic similarity");
        } else {
            debug!("Coherence gate using text-based similarity (no embedder model found)");
        }
        #[cfg(not(feature = "onnx-embed"))]
        debug!("Coherence gate using text-based similarity (onnx-embed feature disabled)");

        Ok(Self {
            db,
            config,
            #[cfg(feature = "onnx-embed")]
            embedder,
            belief_graph,
        })
    }

    /// Try to load embedder from default model paths
    #[cfg(feature = "onnx-embed")]
    fn try_load_embedder() -> Option<Arc<Embedder>> {
        // Model and tokenizer path pairs to try
        let paths = [
            ("models/all-MiniLM-L6-v2.onnx", "models/tokenizer.json"),
            ("../models/all-MiniLM-L6-v2.onnx", "../models/tokenizer.json"),
            ("nagual-rs/models/all-MiniLM-L6-v2.onnx", "nagual-rs/models/tokenizer.json"),
        ];

        for (model_path, tokenizer_path) in &paths {
            if std::path::Path::new(model_path).exists() {
                let config = EmbedderConfig::dim_128(*model_path, *tokenizer_path);
                // Wrap in catch_unwind because ort can panic if ORT_DYLIB_PATH isn't set
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Embedder::new(&config)
                }));

                match result {
                    Ok(Ok(embedder)) => {
                        info!("Loaded embedder from {}", model_path);
                        return Some(Arc::new(embedder));
                    }
                    Ok(Err(e)) => {
                        warn!("Failed to load embedder from {}: {}", model_path, e);
                    }
                    Err(_) => {
                        warn!("Embedder loading panicked (ORT runtime likely not configured)");
                    }
                }
            }
        }
        None
    }

    /// Create with everything: persisted config, embedder, and belief graph
    #[cfg(feature = "onnx-embed")]
    pub async fn with_full_config(
        db: Arc<SqliteDb>,
        embedder: Arc<Embedder>,
    ) -> Result<Self, NagualError> {
        Self::init_schema(&db).await?;
        let config = Self::load_config(&db).await?.unwrap_or_default();
        let belief_graph = Self::load_belief_graph(&db).await?;
        Ok(Self {
            db,
            config,
            embedder: Some(embedder),
            belief_graph,
        })
    }

    /// Set the embedder
    #[cfg(feature = "onnx-embed")]
    pub fn set_embedder(&mut self, embedder: Arc<Embedder>) {
        self.embedder = Some(embedder);
    }

    /// Check if embedder is available
    pub fn has_embedder(&self) -> bool {
        #[cfg(feature = "onnx-embed")]
        { self.embedder.is_some() }
        #[cfg(not(feature = "onnx-embed"))]
        { false }
    }

    /// Get the current configuration
    pub fn config(&self) -> &CoherenceConfig {
        &self.config
    }

    /// Get the belief graph
    pub fn belief_graph(&self) -> &BeliefGraph {
        &self.belief_graph
    }

    /// Initialize database schema for beliefs and config
    async fn init_schema(db: &SqliteDb) -> Result<(), NagualError> {
        // Config table
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS coherence_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            "#,
            &[],
        ).await?;

        // Beliefs table (no foreign key to allow independent operation)
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS beliefs (
                id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL,
                statement TEXT NOT NULL,
                domain TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                embedding TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            "#,
            &[],
        ).await?;

        // Create index on pattern_id for efficient lookups
        let _ = db.execute(
            "CREATE INDEX IF NOT EXISTS idx_beliefs_pattern_id ON beliefs(pattern_id)",
            &[],
        ).await;

        // Create index on domain for efficient filtering
        let _ = db.execute(
            "CREATE INDEX IF NOT EXISTS idx_beliefs_domain ON beliefs(domain)",
            &[],
        ).await;

        // Belief edges table (no foreign keys to allow independent operation)
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS belief_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_belief_id TEXT NOT NULL,
                to_belief_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(from_belief_id, to_belief_id, relation)
            )
            "#,
            &[],
        ).await?;

        // Create index on belief edges
        let _ = db.execute(
            "CREATE INDEX IF NOT EXISTS idx_belief_edges_from ON belief_edges(from_belief_id)",
            &[],
        ).await;

        let _ = db.execute(
            "CREATE INDEX IF NOT EXISTS idx_belief_edges_to ON belief_edges(to_belief_id)",
            &[],
        ).await;

        debug!("Coherence schema initialized");
        Ok(())
    }

    /// Load configuration from database
    async fn load_config(db: &SqliteDb) -> Result<Option<CoherenceConfig>, NagualError> {
        let sql = "SELECT key, value FROM coherence_config";
        let rows: Vec<(String, String)> = db.query(sql, &[], |row| {
            Ok((row.get(0)?, row.get(1)?))
        }).await?;

        if rows.is_empty() {
            return Ok(None);
        }

        let mut config = CoherenceConfig::default();
        for (key, value) in rows {
            match key.as_str() {
                "energy_threshold" => {
                    if let Ok(v) = value.parse() {
                        config.energy_threshold = v;
                    }
                }
                "similarity_threshold" => {
                    if let Ok(v) = value.parse() {
                        config.similarity_threshold = v;
                    }
                }
                "max_conflicts" => {
                    if let Ok(v) = value.parse() {
                        config.max_conflicts = v;
                    }
                }
                "check_enabled" => {
                    config.check_enabled = value == "true";
                }
                _ => {}
            }
        }

        Ok(Some(config))
    }

    /// Load belief graph from database
    async fn load_belief_graph(db: &SqliteDb) -> Result<BeliefGraph, NagualError> {
        let mut graph = BeliefGraph::new();

        // Load beliefs
        let beliefs_sql = r#"
            SELECT id, pattern_id, statement, domain, confidence, embedding
            FROM beliefs
        "#;

        let beliefs: Vec<(String, String, String, String, f64, Option<String>)> = db.query(
            beliefs_sql,
            &[],
            |row| Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
            )),
        ).await?;

        for (id, pattern_id, statement, domain, confidence, embedding_json) in beliefs {
            let belief = Belief {
                id: id.clone(),
                pattern_id,
                statement,
                domain,
                confidence,
                dependencies: Vec::new(),
                contradicts: Vec::new(),
            };

            // Parse embedding if present
            if let Some(ref _emb_json) = embedding_json {
                // Embedding stored but we don't need it in memory for basic belief
            }

            graph.add_belief(belief);
        }

        // Load edges
        let edges_sql = r#"
            SELECT from_belief_id, to_belief_id, relation, weight
            FROM belief_edges
        "#;

        let edges: Vec<(String, String, String, f64)> = db.query(
            edges_sql,
            &[],
            |row| Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
            )),
        ).await?;

        for (from_id, to_id, relation_str, weight) in edges {
            let relation = match relation_str.as_str() {
                "supports" => BeliefRelation::Supports,
                "contradicts" => BeliefRelation::Contradicts,
                "depends_on" => BeliefRelation::DependsOn,
                "refines" => BeliefRelation::Refines,
                _ => continue,
            };

            graph.add_edge(BeliefEdge {
                from: from_id,
                to: to_id,
                relation,
                weight,
            });
        }

        debug!("Loaded belief graph: {} beliefs, {} edges", graph.beliefs.len(), graph.edges.len());
        Ok(graph)
    }

    /// Save configuration to database
    pub async fn save_config(&self) -> Result<(), NagualError> {
        Self::init_schema(&self.db).await?;

        let config_items = [
            ("energy_threshold", self.config.energy_threshold.to_string()),
            ("similarity_threshold", self.config.similarity_threshold.to_string()),
            ("max_conflicts", self.config.max_conflicts.to_string()),
            ("check_enabled", self.config.check_enabled.to_string()),
        ];

        for (key, value) in &config_items {
            self.db.execute(
                "INSERT OR REPLACE INTO coherence_config (key, value) VALUES (?, ?)",
                &[key, value],
            ).await?;
        }

        info!("Coherence configuration saved to database");
        Ok(())
    }

    /// Update configuration and persist it
    pub async fn update_config(&mut self, updates: CoherenceConfigUpdate) -> Result<(), NagualError> {
        if let Some(e) = updates.energy_threshold {
            self.config.energy_threshold = e.clamp(0.0, 1.0);
        }
        if let Some(s) = updates.similarity_threshold {
            self.config.similarity_threshold = s.clamp(0.0, 1.0);
        }
        if let Some(m) = updates.max_conflicts {
            self.config.max_conflicts = m;
        }
        if let Some(e) = updates.check_enabled {
            self.config.check_enabled = e;
        }

        self.save_config().await
    }

    /// Store a belief in the database
    pub async fn store_belief(&mut self, belief: &Belief, embedding: Option<&[f32]>) -> Result<(), NagualError> {
        Self::init_schema(&self.db).await?;

        let embedding_json = embedding.map(|e| serde_json::to_string(e).unwrap_or_default());

        self.db.execute(
            r#"
            INSERT OR REPLACE INTO beliefs (id, pattern_id, statement, domain, confidence, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
            &[
                &belief.id,
                &belief.pattern_id,
                &belief.statement,
                &belief.domain,
                &belief.confidence.to_string(),
                &embedding_json.unwrap_or_default(),
            ],
        ).await?;

        // Add to in-memory graph
        self.belief_graph.add_belief(belief.clone());

        debug!("Stored belief {} for pattern {}", belief.id, belief.pattern_id);
        Ok(())
    }

    /// Store a belief edge (relationship) in the database
    pub async fn store_edge(&mut self, edge: &BeliefEdge) -> Result<(), NagualError> {
        Self::init_schema(&self.db).await?;

        let relation_str = edge.relation.to_string();

        self.db.execute(
            r#"
            INSERT OR REPLACE INTO belief_edges (from_belief_id, to_belief_id, relation, weight)
            VALUES (?, ?, ?, ?)
            "#,
            &[
                &edge.from,
                &edge.to,
                &relation_str,
                &edge.weight.to_string(),
            ],
        ).await?;

        // Add to in-memory graph
        self.belief_graph.add_edge(edge.clone());

        debug!("Stored edge: {} --[{}]--> {}", edge.from, relation_str, edge.to);
        Ok(())
    }

    /// Delete beliefs for a pattern
    pub async fn delete_beliefs_for_pattern(&mut self, pattern_id: &str) -> Result<(), NagualError> {
        // Delete edges first (foreign key constraint)
        self.db.execute(
            r#"
            DELETE FROM belief_edges
            WHERE from_belief_id IN (SELECT id FROM beliefs WHERE pattern_id = ?)
               OR to_belief_id IN (SELECT id FROM beliefs WHERE pattern_id = ?)
            "#,
            &[&pattern_id, &pattern_id],
        ).await?;

        // Delete beliefs
        self.db.execute(
            "DELETE FROM beliefs WHERE pattern_id = ?",
            &[&pattern_id],
        ).await?;

        // Remove from in-memory graph
        let belief_ids: Vec<String> = self.belief_graph.beliefs.values()
            .filter(|b| b.pattern_id == pattern_id)
            .map(|b| b.id.clone())
            .collect();

        for id in &belief_ids {
            self.belief_graph.beliefs.remove(id);
        }
        self.belief_graph.edges.retain(|e| !belief_ids.contains(&e.from) && !belief_ids.contains(&e.to));

        debug!("Deleted beliefs for pattern {}", pattern_id);
        Ok(())
    }

    /// Generate embedding for text using the embedder
    fn generate_embedding(&self, text: &str) -> Option<Vec<f32>> {
        #[cfg(feature = "onnx-embed")]
        {
            self.embedder.as_ref().and_then(|embedder| {
                match embedder.embed(text) {
                    Ok(result) => Some(result.embedding),
                    Err(e) => {
                        warn!("Failed to generate embedding: {}", e);
                        None
                    }
                }
            })
        }
        #[cfg(not(feature = "onnx-embed"))]
        {
            let _ = text;
            None
        }
    }

    /// Calculate cosine similarity between two embeddings
    fn embedding_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }
        let arr_a = Array1::from_vec(a.to_vec());
        let arr_b = Array1::from_vec(b.to_vec());
        cosine_similarity(&arr_a.view(), &arr_b.view()) as f64
    }

    /// Check if a new pattern is coherent with existing knowledge
    #[instrument(skip(self, problem, solution))]
    pub async fn check(
        &self,
        problem: &str,
        solution: &str,
        domain: &str,
    ) -> Result<CoherenceResult, NagualError> {
        if !self.config.check_enabled {
            return Ok(CoherenceResult::coherent(1.0, self.config.energy_threshold, 0));
        }

        info!("Checking coherence for new pattern in domain '{}' (embedder: {})",
              domain,
              if self.has_embedder() { "enabled" } else { "disabled" });

        // 1. Generate embedding for new content if embedder available
        let new_text = format!("{} {}", problem, solution);
        let new_embedding = self.generate_embedding(&new_text);

        // 2. Extract beliefs from the new pattern
        let new_beliefs = self.extract_beliefs("new", problem, solution, domain);
        debug!("Extracted {} beliefs from new pattern", new_beliefs.len());

        // 3. Find potentially conflicting patterns
        let candidates = if new_embedding.is_some() {
            self.find_similar_patterns_by_embedding(domain, problem, solution, new_embedding.as_deref()).await?
        } else {
            self.find_similar_patterns_by_keywords(domain, problem, solution).await?
        };
        debug!("Found {} candidate patterns", candidates.len());

        // 4. Check for contradictions using embeddings when available
        let conflicts = self.detect_conflicts(&new_beliefs, &candidates, new_embedding.as_deref())?;
        debug!("Detected {} potential conflicts", conflicts.len());

        // 5. Count supporting patterns
        let supporting = self.count_supporting_patterns(&new_beliefs, &candidates, new_embedding.as_deref())?;
        debug!("Found {} supporting patterns", supporting);

        // 6. Calculate coherence energy
        let energy = self.calculate_energy(&conflicts, supporting);
        debug!("Calculated coherence energy: {:.3}", energy);

        // 7. Determine action
        let is_coherent = energy >= self.config.energy_threshold;
        let recommendation = self.recommend_action(&conflicts, energy);

        info!(
            "Coherence check complete: {} (energy={:.3}, conflicts={}, supporting={})",
            if is_coherent { "COHERENT" } else { "INCOHERENT" },
            energy,
            conflicts.len(),
            supporting
        );

        Ok(CoherenceResult {
            is_coherent,
            energy,
            threshold: self.config.energy_threshold,
            conflicts,
            supporting_patterns: supporting,
            recommendation,
        })
    }

    /// Check coherence and return extracted beliefs for persistence.
    ///
    /// This method returns both the coherence result and the extracted beliefs,
    /// allowing the caller to persist the beliefs after successful pattern storage.
    #[instrument(skip(self, problem, solution))]
    pub async fn check_with_beliefs(
        &self,
        pattern_id: &str,
        problem: &str,
        solution: &str,
        domain: &str,
    ) -> Result<(CoherenceResult, Vec<Belief>), NagualError> {
        if !self.config.check_enabled {
            let beliefs = self.extract_beliefs(pattern_id, problem, solution, domain);
            return Ok((CoherenceResult::coherent(1.0, self.config.energy_threshold, 0), beliefs));
        }

        info!("Checking coherence for pattern '{}' in domain '{}' (embedder: {})",
              pattern_id, domain,
              if self.has_embedder() { "enabled" } else { "disabled" });

        // 1. Generate embedding for new content if embedder available
        let new_text = format!("{} {}", problem, solution);
        let new_embedding = self.generate_embedding(&new_text);

        // 2. Extract beliefs from the new pattern (with actual pattern_id)
        let new_beliefs = self.extract_beliefs(pattern_id, problem, solution, domain);
        debug!("Extracted {} beliefs from pattern {}", new_beliefs.len(), pattern_id);

        // 3. Find potentially conflicting patterns
        let candidates = if new_embedding.is_some() {
            self.find_similar_patterns_by_embedding(domain, problem, solution, new_embedding.as_deref()).await?
        } else {
            self.find_similar_patterns_by_keywords(domain, problem, solution).await?
        };
        debug!("Found {} candidate patterns", candidates.len());

        // 4. Check for contradictions using embeddings when available
        let conflicts = self.detect_conflicts(&new_beliefs, &candidates, new_embedding.as_deref())?;
        debug!("Detected {} potential conflicts", conflicts.len());

        // 5. Count supporting patterns
        let supporting = self.count_supporting_patterns(&new_beliefs, &candidates, new_embedding.as_deref())?;
        debug!("Found {} supporting patterns", supporting);

        // 6. Calculate coherence energy
        let energy = self.calculate_energy(&conflicts, supporting);
        debug!("Calculated coherence energy: {:.3}", energy);

        // 7. Determine action
        let is_coherent = energy >= self.config.energy_threshold;
        let recommendation = self.recommend_action(&conflicts, energy);

        info!(
            "Coherence check complete for {}: {} (energy={:.3}, conflicts={}, supporting={})",
            pattern_id,
            if is_coherent { "COHERENT" } else { "INCOHERENT" },
            energy,
            conflicts.len(),
            supporting
        );

        let result = CoherenceResult {
            is_coherent,
            energy,
            threshold: self.config.energy_threshold,
            conflicts,
            supporting_patterns: supporting,
            recommendation,
        };

        Ok((result, new_beliefs))
    }

    /// Persist beliefs for a successfully stored pattern.
    ///
    /// This should be called after a pattern passes coherence check and is stored.
    /// It persists the extracted beliefs and detects relationships with existing beliefs.
    #[instrument(skip(self, beliefs))]
    pub async fn persist_beliefs_for_pattern(
        &mut self,
        pattern_id: &str,
        beliefs: Vec<Belief>,
    ) -> Result<usize, NagualError> {
        info!("Persisting {} beliefs for pattern {}", beliefs.len(), pattern_id);

        let mut persisted_count = 0;

        for belief in &beliefs {
            // Generate embedding for the belief if embedder is available
            let embedding = self.generate_embedding(&belief.statement);

            // Store the belief
            self.store_belief(belief, embedding.as_deref()).await?;
            persisted_count += 1;

            // Detect and store relationships with existing beliefs
            self.detect_and_store_relationships(belief, embedding.as_deref()).await?;
        }

        info!("Persisted {} beliefs for pattern {}", persisted_count, pattern_id);
        Ok(persisted_count)
    }

    /// Detect and store relationships between a new belief and existing beliefs.
    async fn detect_and_store_relationships(
        &mut self,
        new_belief: &Belief,
        new_embedding: Option<&[f32]>,
    ) -> Result<(), NagualError> {
        // Clone domain beliefs to avoid borrow checker issues
        // (we need to call &mut self methods inside the loop)
        let domain_beliefs: Vec<Belief> = self.belief_graph
            .beliefs_in_domain(&new_belief.domain)
            .into_iter()
            .cloned()
            .collect();

        // Collect edges to store (compute all relationships first)
        let mut edges_to_store: Vec<(BeliefEdge, &'static str, f64)> = Vec::new();

        for existing in &domain_beliefs {
            // Skip if same belief
            if existing.id == new_belief.id {
                continue;
            }

            // Calculate similarity
            let similarity = if let Some(new_emb) = new_embedding {
                // Try to get existing belief's embedding from DB
                if let Some(existing_emb) = self.get_belief_embedding(&existing.id).await? {
                    self.embedding_similarity(new_emb, &existing_emb)
                } else {
                    self.calculate_text_similarity(
                        &new_belief.statement.to_lowercase(),
                        &existing.statement.to_lowercase(),
                    )
                }
            } else {
                self.calculate_text_similarity(
                    &new_belief.statement.to_lowercase(),
                    &existing.statement.to_lowercase(),
                )
            };

            // Determine relationship type based on similarity and content
            if similarity > 0.85 {
                // Very high similarity = likely supports/reinforces
                let edge = BeliefEdge::supports(&new_belief.id, &existing.id, similarity);
                edges_to_store.push((edge, "support", similarity));
            } else if self.beliefs_may_contradict(new_belief, existing) && similarity > 0.5 {
                // Check for contradiction markers
                let edge = BeliefEdge::contradicts(&new_belief.id, &existing.id, similarity);
                edges_to_store.push((edge, "contradiction", similarity));
            } else if similarity > 0.7 {
                // Moderate similarity = may refine
                let edge = BeliefEdge::refines(&new_belief.id, &existing.id);
                edges_to_store.push((edge, "refines", similarity));
            }
        }

        // Now store all the edges
        for (edge, relation_type, similarity) in edges_to_store {
            self.store_edge(&edge).await?;
            debug!("Stored {} edge: {} -> {} (sim: {:.3})",
                   relation_type, edge.from, edge.to, similarity);
        }

        Ok(())
    }

    /// Check if two beliefs may contradict each other based on keywords
    fn beliefs_may_contradict(&self, a: &Belief, b: &Belief) -> bool {
        let a_lower = a.statement.to_lowercase();
        let b_lower = b.statement.to_lowercase();

        let contradiction_pairs = [
            ("should", "should not"),
            ("always", "never"),
            ("use", "avoid"),
            ("enable", "disable"),
            ("prefer", "avoid"),
            ("sync", "async"),
            ("blocking", "non-blocking"),
            ("mutable", "immutable"),
            ("safe", "unsafe"),
        ];

        for (pos, neg) in &contradiction_pairs {
            let a_has_pos = a_lower.contains(pos);
            let a_has_neg = a_lower.contains(neg);
            let b_has_pos = b_lower.contains(pos);
            let b_has_neg = b_lower.contains(neg);

            if (a_has_pos && b_has_neg) || (a_has_neg && b_has_pos) {
                return true;
            }
        }

        false
    }

    /// Get the embedding for an existing belief from the database
    async fn get_belief_embedding(&self, belief_id: &str) -> Result<Option<Vec<f32>>, NagualError> {
        let sql = "SELECT embedding FROM beliefs WHERE id = ?";
        let result: Option<Option<String>> = self.db.query_one(
            sql,
            &[&belief_id],
            |row| row.get(0),
        ).await?;

        if let Some(Some(embedding_json)) = result {
            if !embedding_json.is_empty() {
                if let Ok(embedding) = serde_json::from_str::<Vec<f32>>(&embedding_json) {
                    return Ok(Some(embedding));
                }
            }
        }

        Ok(None)
    }

    /// Check coherence for an existing pattern by ID
    pub async fn check_pattern(&self, pattern_id: &str) -> Result<CoherenceResult, NagualError> {
        let sql = r#"
            SELECT problem, solution, category
            FROM reasoning_patterns
            WHERE id = ?
        "#;

        let result = self.db.query_one(
            sql,
            &[&pattern_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        ).await?;

        match result {
            Some((problem, solution, domain)) => {
                self.check(&problem, &solution, &domain).await
            }
            None => Err(NagualError::Internal {
                message: format!("Pattern not found: {}", pattern_id),
            }),
        }
    }

    /// Extract beliefs from pattern content
    fn extract_beliefs(
        &self,
        pattern_id: &str,
        problem: &str,
        solution: &str,
        domain: &str,
    ) -> Vec<Belief> {
        let mut beliefs = Vec::new();

        // Extract problem belief
        let problem_belief = Belief::new(pattern_id, problem, domain)
            .with_confidence(0.7);
        beliefs.push(problem_belief);

        // Extract solution beliefs (split on sentences)
        for (i, sentence) in solution.split('.').enumerate() {
            let sentence = sentence.trim();
            if sentence.len() > 10 {
                let belief = Belief::new(
                    pattern_id,
                    sentence,
                    domain,
                ).with_confidence(0.8);
                beliefs.push(belief);
            }

            // Limit to first 5 beliefs
            if i >= 4 {
                break;
            }
        }

        beliefs
    }

    /// Find similar patterns using embedding-based cosine similarity
    async fn find_similar_patterns_by_embedding(
        &self,
        domain: &str,
        problem: &str,
        solution: &str,
        new_embedding: Option<&[f32]>,
    ) -> Result<Vec<CandidatePattern>, NagualError> {
        let domain_pattern = format!("{}%", domain.split('.').next().unwrap_or(domain));

        let sql = r#"
            SELECT id, problem, solution, category, reward, embedding
            FROM reasoning_patterns
            WHERE category LIKE ?
              AND embedding IS NOT NULL
              AND embedding != ''
            ORDER BY reward DESC
            LIMIT 100
        "#;

        let patterns: Vec<CandidatePattern> = self.db.query(
            sql,
            &[&domain_pattern],
            |row| {
                Ok(CandidatePattern {
                    id: row.get(0)?,
                    problem: row.get(1)?,
                    solution: row.get(2)?,
                    domain: row.get(3)?,
                    reward: row.get(4)?,
                    embedding: row.get::<_, Option<String>>(5)?,
                })
            },
        ).await?;

        if patterns.is_empty() {
            return self.find_similar_patterns_by_keywords(domain, problem, solution).await;
        }

        let combined_text = format!("{} {}", problem, solution);

        // Filter by similarity
        let filtered: Vec<CandidatePattern> = patterns.into_iter()
            .filter(|p| {
                // Use embedding similarity if we have both embeddings
                if let (Some(new_emb), Some(ref emb_str)) = (new_embedding, &p.embedding) {
                    if let Ok(stored_emb) = serde_json::from_str::<Vec<f32>>(emb_str) {
                        let sim = self.embedding_similarity(new_emb, &stored_emb);
                        return sim > 0.3; // Cosine similarity threshold for candidates
                    }
                }

                // Fall back to text similarity
                let text_sim = self.calculate_text_similarity(
                    &combined_text.to_lowercase(),
                    &format!("{} {}", p.problem, p.solution).to_lowercase(),
                );
                text_sim > 0.2
            })
            .collect();

        Ok(filtered)
    }

    /// Fallback: Find similar patterns using keyword matching
    async fn find_similar_patterns_by_keywords(
        &self,
        domain: &str,
        problem: &str,
        solution: &str,
    ) -> Result<Vec<CandidatePattern>, NagualError> {
        let domain_pattern = format!("{}%", domain.split('.').next().unwrap_or(domain));

        let sql = r#"
            SELECT id, problem, solution, category, reward
            FROM reasoning_patterns
            WHERE category LIKE ?
            ORDER BY reward DESC
            LIMIT 50
        "#;

        let patterns: Vec<CandidatePattern> = self.db.query(
            sql,
            &[&domain_pattern],
            |row| {
                Ok(CandidatePattern {
                    id: row.get(0)?,
                    problem: row.get(1)?,
                    solution: row.get(2)?,
                    domain: row.get(3)?,
                    reward: row.get(4)?,
                    embedding: None,
                })
            },
        ).await?;

        let filtered: Vec<_> = patterns.into_iter()
            .filter(|p| self.has_keyword_overlap(problem, solution, &p.problem, &p.solution))
            .collect();

        Ok(filtered)
    }

    /// Check if two pattern pairs have sufficient keyword overlap
    fn has_keyword_overlap(&self, prob1: &str, sol1: &str, prob2: &str, sol2: &str) -> bool {
        let p1_lower = prob1.to_lowercase();
        let p1_words: std::collections::HashSet<&str> = p1_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let s1_lower = sol1.to_lowercase();
        let s1_words: std::collections::HashSet<&str> = s1_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let p2_lower = prob2.to_lowercase();
        let p2_words: std::collections::HashSet<&str> = p2_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let s2_lower = sol2.to_lowercase();
        let s2_words: std::collections::HashSet<&str> = s2_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let problem_overlap = p1_words.iter().filter(|w| p2_words.contains(*w)).count();
        let solution_overlap = s1_words.iter().filter(|w| s2_words.contains(*w)).count();

        problem_overlap >= 2 || solution_overlap >= 2
    }

    /// Detect conflicts between new beliefs and existing patterns
    fn detect_conflicts(
        &self,
        new_beliefs: &[Belief],
        candidates: &[CandidatePattern],
        new_embedding: Option<&[f32]>,
    ) -> Result<Vec<Conflict>, NagualError> {
        let mut conflicts = Vec::new();

        for new_belief in new_beliefs {
            for candidate in candidates {
                let candidate_beliefs = self.extract_beliefs(
                    &candidate.id,
                    &candidate.problem,
                    &candidate.solution,
                    &candidate.domain,
                );

                // Parse candidate embedding once
                let candidate_emb: Option<Vec<f32>> = candidate.embedding.as_ref()
                    .and_then(|s| serde_json::from_str(s).ok());

                for existing in &candidate_beliefs {
                    if let Some(conflict) = self.check_contradiction(
                        new_belief,
                        existing,
                        candidate,
                        new_embedding,
                        candidate_emb.as_deref(),
                    ) {
                        conflicts.push(conflict);
                    }
                }
            }
        }

        // Deduplicate by pattern pair
        conflicts.sort_by(|a, b| a.pattern_b_id.cmp(&b.pattern_b_id));
        conflicts.dedup_by(|a, b| a.pattern_b_id == b.pattern_b_id);

        // Limit conflicts
        conflicts.truncate(10);

        Ok(conflicts)
    }

    /// Check if two beliefs contradict each other
    /// Uses embedding similarity when available, falls back to keyword analysis
    fn check_contradiction(
        &self,
        a: &Belief,
        b: &Belief,
        _candidate: &CandidatePattern,
        new_embedding: Option<&[f32]>,
        candidate_embedding: Option<&[f32]>,
    ) -> Option<Conflict> {
        // Must be in same/related domain
        let domain_match = a.domain == b.domain
            || a.domain.starts_with(&format!("{}.", b.domain))
            || b.domain.starts_with(&format!("{}.", a.domain));

        if !domain_match {
            return None;
        }

        let a_lower = a.statement.to_lowercase();
        let b_lower = b.statement.to_lowercase();

        // Calculate similarity - prefer embedding similarity if available
        let similarity = if let (Some(new_emb), Some(cand_emb)) = (new_embedding, candidate_embedding) {
            self.embedding_similarity(new_emb, cand_emb)
        } else {
            self.calculate_text_similarity(&a_lower, &b_lower)
        };

        // Check for explicit contradiction markers
        let contradiction_pairs = [
            ("should", "should not"),
            ("always", "never"),
            ("use", "avoid"),
            ("enable", "disable"),
            ("prefer", "avoid"),
            ("recommended", "not recommended"),
            ("best practice", "anti-pattern"),
            ("do", "don't"),
            ("sync", "async"),
            ("blocking", "non-blocking"),
            ("mutable", "immutable"),
            ("safe", "unsafe"),
        ];

        for (pos, neg) in &contradiction_pairs {
            let a_has_pos = a_lower.contains(pos);
            let a_has_neg = a_lower.contains(neg);
            let b_has_pos = b_lower.contains(pos);
            let b_has_neg = b_lower.contains(neg);

            if (a_has_pos && b_has_neg) || (a_has_neg && b_has_pos) {
                let severity = if similarity > 0.7 {
                    ConflictSeverity::Major
                } else if similarity > 0.5 {
                    ConflictSeverity::Moderate
                } else {
                    ConflictSeverity::Minor
                };

                return Some(Conflict::new(
                    a,
                    b,
                    severity,
                    &format!(
                        "Contradiction detected: '{}' vs '{}'",
                        &a.statement[..a.statement.len().min(50)],
                        &b.statement[..b.statement.len().min(50)]
                    ),
                    similarity,
                ));
            }
        }

        // Check for semantic conflict with high similarity but different recommendations
        if similarity > self.config.similarity_threshold && similarity < 0.95 {
            let topic_words = ["use", "prefer", "choose", "implement", "apply", "recommend"];
            let a_has_topic = topic_words.iter().any(|w| a_lower.contains(w));
            let b_has_topic = topic_words.iter().any(|w| b_lower.contains(w));

            if a_has_topic && b_has_topic {
                let a_subjects = self.extract_subjects(&a_lower);
                let b_subjects = self.extract_subjects(&b_lower);

                if !a_subjects.is_empty() && !b_subjects.is_empty() {
                    let overlap: Vec<_> = a_subjects.intersection(&b_subjects).collect();
                    if overlap.is_empty() {
                        return Some(Conflict::new(
                            a,
                            b,
                            ConflictSeverity::Minor,
                            "Different recommendations for similar context",
                            similarity,
                        ));
                    }
                }
            }
        }

        None
    }

    /// Extract subject words from a statement
    fn extract_subjects(&self, text: &str) -> std::collections::HashSet<String> {
        let stop_words = ["the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "for",
            "and", "but", "or", "nor", "so", "yet", "to", "of", "in", "on", "at",
            "by", "with", "from", "use", "prefer", "choose", "implement", "apply"];

        text.split_whitespace()
            .filter(|w| w.len() > 3)
            .filter(|w| !stop_words.contains(w))
            .map(|w| w.to_string())
            .collect()
    }

    /// Calculate text similarity (Jaccard index)
    fn calculate_text_similarity(&self, a: &str, b: &str) -> f64 {
        let a_words: std::collections::HashSet<_> = a.split_whitespace().collect();
        let b_words: std::collections::HashSet<_> = b.split_whitespace().collect();

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Count patterns that support the new beliefs
    fn count_supporting_patterns(
        &self,
        new_beliefs: &[Belief],
        candidates: &[CandidatePattern],
        new_embedding: Option<&[f32]>,
    ) -> Result<usize, NagualError> {
        let mut supporting = 0;

        for candidate in candidates {
            let candidate_beliefs = self.extract_beliefs(
                &candidate.id,
                &candidate.problem,
                &candidate.solution,
                &candidate.domain,
            );

            // Parse candidate embedding once
            let candidate_emb: Option<Vec<f32>> = candidate.embedding.as_ref()
                .and_then(|s| serde_json::from_str(s).ok());

            // Check if any beliefs are reinforcing (similar + high reward)
            for new_belief in new_beliefs {
                for existing in &candidate_beliefs {
                    // Use embedding similarity if available
                    let similarity = if let (Some(new_emb), Some(ref cand_emb)) = (new_embedding, &candidate_emb) {
                        self.embedding_similarity(new_emb, cand_emb)
                    } else {
                        self.calculate_text_similarity(
                            &new_belief.statement.to_lowercase(),
                            &existing.statement.to_lowercase(),
                        )
                    };

                    if similarity > 0.7 && candidate.reward > 0.6 {
                        supporting += 1;
                        break;
                    }
                }
            }
        }

        Ok(supporting)
    }

    /// Calculate coherence energy using belief relationships
    fn calculate_energy(&self, conflicts: &[Conflict], supporting: usize) -> f64 {
        // Energy formula: E = (1 - conflict_penalty) * (1 + support_bonus)
        let conflict_penalty: f64 = conflicts.iter()
            .map(|c| match c.severity {
                ConflictSeverity::Minor => 0.1,
                ConflictSeverity::Moderate => 0.25,
                ConflictSeverity::Major => 0.5,
            })
            .sum();

        let support_bonus = (supporting as f64 * 0.1).min(0.3);

        let energy = (1.0 - conflict_penalty.min(1.0)) * (1.0 + support_bonus);
        energy.clamp(0.0, 1.0)
    }

    /// Recommend action based on conflicts and energy
    fn recommend_action(&self, conflicts: &[Conflict], energy: f64) -> CoherenceAction {
        if conflicts.is_empty() {
            return CoherenceAction::Accept;
        }

        let major_conflicts = conflicts.iter()
            .filter(|c| matches!(c.severity, ConflictSeverity::Major))
            .count();

        if major_conflicts > 0 {
            return CoherenceAction::Reject {
                reason: format!("{} major contradiction(s) detected", major_conflicts),
            };
        }

        if conflicts.len() > self.config.max_conflicts {
            return CoherenceAction::RequireReview {
                conflicts: conflicts.iter().map(|c| c.description.clone()).collect(),
            };
        }

        if energy >= self.config.energy_threshold {
            CoherenceAction::AcceptWithWarning {
                warnings: conflicts.iter().map(|c| c.description.clone()).collect(),
            }
        } else {
            CoherenceAction::RequireReview {
                conflicts: conflicts.iter().map(|c| c.description.clone()).collect(),
            }
        }
    }

    /// Analyze coherence across the entire knowledge base
    #[instrument(skip(self))]
    pub async fn analyze_global_coherence(&self) -> Result<GlobalCoherenceReport, NagualError> {
        info!("Analyzing global coherence (embedder: {})",
              if self.has_embedder() { "enabled" } else { "disabled" });

        // Get pattern count per domain
        let sql = r#"
            SELECT category, COUNT(*) as count
            FROM reasoning_patterns
            GROUP BY category
            ORDER BY count DESC
            LIMIT 20
        "#;

        let domain_counts: Vec<(String, i64)> = self.db.query(
            sql,
            &[],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).await?;

        // Get total pattern count
        let total_sql = "SELECT COUNT(*) FROM reasoning_patterns";
        let total_patterns: i64 = self.db.query_one(
            total_sql,
            &[],
            |row| row.get(0),
        ).await?.unwrap_or(0);

        // Sample patterns for conflict analysis
        let sample_sql = r#"
            SELECT id, problem, solution, category, embedding
            FROM reasoning_patterns
            WHERE reward > 0.5
            ORDER BY RANDOM()
            LIMIT 100
        "#;

        let samples: Vec<CandidatePattern> = self.db.query(
            sample_sql,
            &[],
            |row| {
                Ok(CandidatePattern {
                    id: row.get(0)?,
                    problem: row.get(1)?,
                    solution: row.get(2)?,
                    domain: row.get(3)?,
                    reward: 0.5,
                    embedding: row.get::<_, Option<String>>(4)?,
                })
            },
        ).await?;

        let sampled_count = samples.len();

        // Check for conflicts within samples
        let mut all_conflicts = Vec::new();
        let mut comparisons_made = 0;

        for (i, sample) in samples.iter().enumerate() {
            let beliefs = self.extract_beliefs(
                &sample.id,
                &sample.problem,
                &sample.solution,
                &sample.domain,
            );

            // Parse sample embedding once
            let sample_emb: Option<Vec<f32>> = sample.embedding.as_ref()
                .and_then(|s| serde_json::from_str(s).ok());

            for other in samples.iter().skip(i + 1) {
                if sample.domain == other.domain {
                    comparisons_made += 1;
                    let other_beliefs = self.extract_beliefs(
                        &other.id,
                        &other.problem,
                        &other.solution,
                        &other.domain,
                    );

                    // Parse other embedding
                    let other_emb: Option<Vec<f32>> = other.embedding.as_ref()
                        .and_then(|s| serde_json::from_str(s).ok());

                    for a in &beliefs {
                        for b in &other_beliefs {
                            if let Some(conflict) = self.check_contradiction(
                                a, b, other,
                                sample_emb.as_deref(),
                                other_emb.as_deref(),
                            ) {
                                all_conflicts.push(conflict);
                            }
                        }
                    }
                }
            }
        }

        // Calculate coherence
        let max_expected_conflicts = (comparisons_made as f64 * 0.1).max(1.0);
        let conflict_ratio = all_conflicts.len() as f64 / max_expected_conflicts;
        let overall_coherence = (1.0 - conflict_ratio.min(1.0)).max(0.0);

        info!(
            "Global coherence analysis: {:.1}% coherent ({} conflicts in {} comparisons)",
            overall_coherence * 100.0,
            all_conflicts.len(),
            comparisons_made
        );

        Ok(GlobalCoherenceReport {
            total_patterns: total_patterns as usize,
            sampled_patterns: sampled_count,
            comparisons_made,
            conflicts_detected: all_conflicts.len(),
            overall_coherence,
            top_domains: domain_counts.into_iter()
                .take(10)
                .map(|(d, c)| (d, c as usize))
                .collect(),
            sample_conflicts: all_conflicts.into_iter().take(10).collect(),
        })
    }

    /// Get belief graph statistics
    pub fn graph_stats(&self) -> (usize, usize) {
        (self.belief_graph.beliefs.len(), self.belief_graph.edges.len())
    }
}

/// Updates for coherence configuration
#[derive(Debug, Clone, Default)]
pub struct CoherenceConfigUpdate {
    pub energy_threshold: Option<f64>,
    pub similarity_threshold: Option<f64>,
    pub max_conflicts: Option<usize>,
    pub check_enabled: Option<bool>,
}

/// Candidate pattern for comparison
#[derive(Debug, Clone)]
pub struct CandidatePattern {
    pub id: String,
    pub problem: String,
    pub solution: String,
    pub domain: String,
    pub reward: f64,
    pub embedding: Option<String>,
}

/// Global coherence analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCoherenceReport {
    pub total_patterns: usize,
    pub sampled_patterns: usize,
    pub comparisons_made: usize,
    pub conflicts_detected: usize,
    pub overall_coherence: f64,
    pub top_domains: Vec<(String, usize)>,
    pub sample_conflicts: Vec<Conflict>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_similarity() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let gate = CoherenceGate::with_defaults(db);

        let sim = gate.calculate_text_similarity(
            "use tokio for async runtime",
            "use tokio for async operations",
        );
        assert!(sim > 0.5);

        let sim_low = gate.calculate_text_similarity(
            "use tokio",
            "prefer blocking io",
        );
        assert!(sim_low < 0.3);
    }

    #[test]
    fn test_energy_calculation() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let gate = CoherenceGate::with_defaults(db);

        // No conflicts, no support
        let energy = gate.calculate_energy(&[], 0);
        assert_eq!(energy, 1.0);

        // No conflicts, some support
        let energy = gate.calculate_energy(&[], 3);
        assert!(energy > 1.0 - 0.01);

        // One minor conflict
        let conflict = Conflict {
            belief_a: "a".into(),
            belief_b: "b".into(),
            pattern_a_id: "p1".into(),
            pattern_b_id: "p2".into(),
            severity: ConflictSeverity::Minor,
            description: "test".into(),
            similarity: 0.5,
        };
        let energy = gate.calculate_energy(&[conflict], 0);
        assert!(energy < 1.0);
        assert!(energy > 0.8);
    }

    #[test]
    fn test_recommend_action() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let gate = CoherenceGate::with_defaults(db);

        // No conflicts
        let action = gate.recommend_action(&[], 1.0);
        assert!(matches!(action, CoherenceAction::Accept));

        // Major conflict
        let major_conflict = Conflict {
            belief_a: "a".into(),
            belief_b: "b".into(),
            pattern_a_id: "p1".into(),
            pattern_b_id: "p2".into(),
            severity: ConflictSeverity::Major,
            description: "major".into(),
            similarity: 0.9,
        };
        let action = gate.recommend_action(&[major_conflict], 0.5);
        assert!(matches!(action, CoherenceAction::Reject { .. }));
    }

    #[test]
    fn test_extract_beliefs() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let gate = CoherenceGate::with_defaults(db);

        let beliefs = gate.extract_beliefs(
            "p1",
            "How to handle async errors",
            "Use Result type. Propagate with ?. Log before returning.",
            "rust.async",
        );

        assert!(!beliefs.is_empty());
        assert!(beliefs.len() <= 5);
    }

    #[test]
    fn test_embedding_similarity() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let gate = CoherenceGate::with_defaults(db);

        // Same vector = 1.0
        let v1 = vec![1.0, 0.0, 0.0];
        let sim = gate.embedding_similarity(&v1, &v1);
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal = 0.0
        let v2 = vec![0.0, 1.0, 0.0];
        let sim = gate.embedding_similarity(&v1, &v2);
        assert!(sim.abs() < 0.001);

        // Opposite = -1.0
        let v3 = vec![-1.0, 0.0, 0.0];
        let sim = gate.embedding_similarity(&v1, &v3);
        assert!((sim + 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_config_persistence() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let mut gate = CoherenceGate::with_defaults(db.clone());

        // Update config
        gate.update_config(CoherenceConfigUpdate {
            energy_threshold: Some(0.6),
            similarity_threshold: Some(0.9),
            max_conflicts: Some(5),
            check_enabled: Some(true),
        }).await.unwrap();

        // Create new gate and load config
        let gate2 = CoherenceGate::with_persisted_config(db).await.unwrap();

        assert!((gate2.config.energy_threshold - 0.6).abs() < 0.001);
        assert!((gate2.config.similarity_threshold - 0.9).abs() < 0.001);
        assert_eq!(gate2.config.max_conflicts, 5);
    }

    #[tokio::test]
    async fn test_belief_persistence() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let mut gate = CoherenceGate::with_persisted_config(db.clone()).await.unwrap();

        // Store a belief
        let belief = Belief::new("pattern-1", "Use async for I/O", "rust.async")
            .with_confidence(0.9);

        gate.store_belief(&belief, None).await.unwrap();

        // Store an edge
        let belief2 = Belief::new("pattern-2", "Async improves throughput", "rust.async")
            .with_confidence(0.8);
        gate.store_belief(&belief2, None).await.unwrap();

        let edge = BeliefEdge::supports(&belief.id, &belief2.id, 0.9);
        gate.store_edge(&edge).await.unwrap();

        // Reload and verify
        let gate2 = CoherenceGate::with_persisted_config(db).await.unwrap();

        let (beliefs_count, edges_count) = gate2.graph_stats();
        assert_eq!(beliefs_count, 2);
        assert_eq!(edges_count, 1);
    }

    #[tokio::test]
    async fn test_delete_beliefs_for_pattern() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let mut gate = CoherenceGate::with_persisted_config(db.clone()).await.unwrap();

        // Store beliefs for two patterns
        let b1 = Belief::new("pattern-1", "Statement 1", "rust");
        let b2 = Belief::new("pattern-1", "Statement 2", "rust");
        let b3 = Belief::new("pattern-2", "Statement 3", "rust");

        gate.store_belief(&b1, None).await.unwrap();
        gate.store_belief(&b2, None).await.unwrap();
        gate.store_belief(&b3, None).await.unwrap();

        let (count, _) = gate.graph_stats();
        assert_eq!(count, 3);

        // Delete beliefs for pattern-1
        gate.delete_beliefs_for_pattern("pattern-1").await.unwrap();

        let (count_after, _) = gate.graph_stats();
        assert_eq!(count_after, 1);
    }

    #[test]
    fn test_has_embedder() {
        let db = Arc::new(SqliteDb::open_in_memory().unwrap());
        let gate = CoherenceGate::with_defaults(db);
        assert!(!gate.has_embedder());
    }
}
