//! LoRA (Low-Rank Adaptation) fine-tuning for per-domain specialist embedding models.
//!
//! Trains lightweight adapters that improve retrieval accuracy for specific domains
//! using contrastive learning on pattern pairs.
//!
//! # Architecture
//!
//! A LoRA adapter applies a low-rank transformation to base embeddings:
//! `output = input + alpha * B @ A @ input`
//! where A is (rank x base_dim) and B is (base_dim x rank).
//!
//! For rank=4, dim=128, this adds only ~4KB of parameters per domain.
//!
//! # Usage
//!
//! ```rust,no_run
//! use nagual::ml::lora::{LoraAdapter, LoraConfig, LoraTrainer, TrainingConfig};
//!
//! let config = LoraConfig::default();
//! let mut adapter = LoraAdapter::new("rust", config);
//!
//! // Train on contrastive pairs
//! let trainer = LoraTrainer::new(TrainingConfig::default());
//! let result = trainer.train(&mut adapter, &pairs);
//! ```

pub mod adapter;
pub mod storage;
pub mod trainer;

pub use adapter::{LoraAdapter, LoraConfig};
pub use storage::{LoraStorage, StoredAdapter};
pub use trainer::{LoraTrainer, TrainingConfig, TrainingPair, TrainingResult};
