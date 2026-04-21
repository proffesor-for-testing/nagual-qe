//! KOS (Knowledge Operating System) API
//!
//! Provides programmatic access to all KOS features through the `Nagual` API.
//! Each field is feature-gated to match the corresponding KOS module.

use std::sync::Arc;

use crate::db::SqliteDb;
use crate::error::Result;

/// KOS API -- unified access to Knowledge Operating System features.
///
/// Each field is feature-gated to match the corresponding KOS module.
/// Construct via `KosApi::new()` with an `Arc<SqliteDb>`.
pub struct KosApi {
    pub lineage: Arc<crate::lineage::LineageQuery>,

    pub witness: Arc<crate::witness::WitnessChain>,

    pub delta: Arc<crate::delta::DeltaStore>,

    pub coherence: Arc<crate::coherence::scoring::CoherenceScorer>,

    pub transfer: Arc<crate::learning::transfer::DomainTransferEngine>,

    pub epochs: Arc<crate::epoch::EpochManager>,

    pub tiering: Arc<crate::tiering::TieringManager>,

    pub views: Arc<crate::agent_views::ViewManager>,

    pub ewc: Arc<crate::learning::ewc::EwcManager>,

    pub ladder: Arc<crate::router::ladder::RoutingLadder>,

    pub hnsw: Arc<crate::ml::hyperbolic::HyperbolicIndex>,
}

impl KosApi {
    /// Create a new KosApi, initializing all enabled KOS modules.
    ///
    /// Modules that require async initialization (TieringManager, ViewManager,
    /// EwcManager, RoutingLadder) are constructed here. Modules that need
    /// explicit `init_schema()` calls have them invoked during construction.
    pub async fn new(db: Arc<SqliteDb>) -> Result<Self> {
        // Construct all feature-gated modules

        let lineage = Arc::new(crate::lineage::LineageQuery::new(db.clone()));

        let witness = {
            let w = crate::witness::WitnessChain::new(db.clone());
            w.init_schema().await?;
            Arc::new(w)
        };

        let delta = Arc::new(crate::delta::DeltaStore::new(db.clone(), 100));

        let coherence = {
            let scorer = crate::coherence::scoring::CoherenceScorer::new(
                db.clone(),
                crate::coherence::scoring::ScoringConfig::default(),
            );
            scorer.init_schema().await?;
            Arc::new(scorer)
        };

        let transfer = {
            let engine = crate::learning::transfer::DomainTransferEngine::new(
                db.clone(),
                crate::learning::transfer::TransferConfig::default(),
            );
            engine.init_schema().await?;
            Arc::new(engine)
        };

        let epochs = {
            let mgr = crate::epoch::EpochManager::new(db.clone());
            mgr.init_schema().await?;
            Arc::new(mgr)
        };

        let tiering = Arc::new(
            crate::tiering::TieringManager::new(
                db.clone(),
                crate::tiering::TieringConfig::default(),
            )
            .await?,
        );

        let views = Arc::new(
            crate::agent_views::ViewManager::new(
                db.clone(),
                crate::agent_views::ViewConfig::default(),
            )
            .await?,
        );

        let ewc = Arc::new(
            crate::learning::ewc::EwcManager::new(
                db.clone(),
                crate::learning::ewc::EwcEngineConfig::default(),
            )
            .await?,
        );

        let ladder = Arc::new(
            crate::router::ladder::RoutingLadder::new(
                db.clone(),
                crate::router::ladder::LadderConfig::default(),
            )
            .await?,
        );

        let hnsw = Arc::new(crate::ml::hyperbolic::HyperbolicIndex::new(
            crate::ml::hyperbolic::HyperbolicConfig::default(),
            crate::ml::hyperbolic::HyperbolicIndexConfig::default(),
        ));

        Ok(Self {
            lineage,
            witness,
            delta,
            coherence,
            transfer,
            epochs,
            tiering,
            views,
            ewc,
            ladder,
            hnsw,
        })
    }
}
