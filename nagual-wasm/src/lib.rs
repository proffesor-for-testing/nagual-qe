//! # nagual-wasm
//!
//! WASM runtime for ProfDAG pattern search in browser/edge environments.
//!
//! This crate provides:
//! - In-memory pattern index with HNSW-like search
//! - IndexedDB persistence for patterns
//! - JSON-based pattern import/export
//! - Vector similarity search optimized for browser
//!
//! ## Performance Targets
//!
//! - Bundle size: < 2MB (without ONNX)
//! - Search latency: < 10ms for 10K patterns
//! - Memory efficient in-browser storage

mod bindings;
mod search;
mod storage;

pub use bindings::*;
pub use search::*;
pub use storage::*;

use wasm_bindgen::prelude::*;

/// Initialize the WASM module.
///
/// This sets up panic hooks for better error messages and initializes logging.
/// Should be called once at startup.
#[wasm_bindgen(start)]
pub fn init() {
    // Set up panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    // Initialize WASM logger
    wasm_logger::init(wasm_logger::Config::default());

    log::info!("nagual-wasm initialized");
}

/// Get the version of the WASM module.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get build information.
#[wasm_bindgen]
pub fn build_info() -> JsValue {
    let info = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "name": env!("CARGO_PKG_NAME"),
        "target": "wasm32-unknown-unknown",
        "features": {
            "indexeddb": true,
            "vector_search": true,
            "json_export": true
        }
    });

    serde_wasm_bindgen::to_value(&info).unwrap_or(JsValue::NULL)
}
