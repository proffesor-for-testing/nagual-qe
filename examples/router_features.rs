//! Dump router features for the labelled query set, for `models/train_fastgrnn.py`.
//!
//! ```bash
//! cargo run --example router_features --no-default-features --features kos -- \
//!     models/router_queries.jsonl > /tmp/router_features.jsonl
//! ```
//!
//! Features are computed by the production `ComplexityEstimator`, so the trained weights see
//! exactly what `VendorRouter` sees at runtime.

use std::io::{BufRead, BufReader, Write};

use nagual::router::{ComplexityEstimator, EstimatorConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/router_queries.jsonl".into());
    let estimator = ComplexityEstimator::new(EstimatorConfig::default());
    // The estimator validates but does not score the embedding.
    let embedding = vec![0.1f32; 128];

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in BufReader::new(std::fs::File::open(&path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut row: serde_json::Value = serde_json::from_str(&line)?;
        let query = row["query"]
            .as_str()
            .ok_or("row without query")?
            .to_string();
        let features = estimator.extract_features(&query, &embedding)?.to_vector();
        row["features"] = serde_json::json!(features);
        writeln!(out, "{}", row)?;
    }
    Ok(())
}
