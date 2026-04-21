//! IndexedDB persistence for WASM patterns.
//!
//! Provides asynchronous storage using the browser's IndexedDB API.
//! Patterns are stored in an object store and can be loaded/saved
//! incrementally.

use js_sys::Array;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use web_sys::{IdbDatabase, IdbOpenDbRequest, IdbRequest, IdbTransaction, IdbTransactionMode};

/// Database configuration.
const DB_NAME: &str = "nagual_profdag";
const DB_VERSION: u32 = 1;
const STORE_NAME: &str = "patterns";

/// Error type for storage operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageError {
    pub code: String,
    pub message: String,
}

impl StorageError {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn from_js(value: JsValue) -> Self {
        let message = if let Some(s) = value.as_string() {
            s
        } else if let Some(err) = value.dyn_ref::<js_sys::Error>() {
            err.message().into()
        } else {
            format!("{:?}", value)
        };

        Self::new("JS_ERROR", message)
    }
}

impl From<StorageError> for JsValue {
    fn from(err: StorageError) -> Self {
        serde_wasm_bindgen::to_value(&err).unwrap_or(JsValue::from_str(&err.message))
    }
}

/// Check if an object store exists in the database.
fn store_exists(db: &IdbDatabase, name: &str) -> bool {
    let names = db.object_store_names();
    for i in 0..names.length() {
        if let Some(n) = names.get(i) {
            if n == name {
                return true;
            }
        }
    }
    false
}

/// IndexedDB storage for patterns.
#[wasm_bindgen]
pub struct IndexedDBStorage {
    db: Option<IdbDatabase>,
}

#[wasm_bindgen]
impl IndexedDBStorage {
    /// Create a new storage instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { db: None }
    }

    /// Check if storage is available.
    #[wasm_bindgen]
    pub fn is_available() -> bool {
        if let Some(window) = web_sys::window() {
            window.indexed_db().ok().flatten().is_some()
        } else {
            false
        }
    }

    /// Open the database connection.
    #[wasm_bindgen]
    pub async fn open(&mut self) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;

        let idb = window
            .indexed_db()
            .map_err(|e| StorageError::from_js(e))?
            .ok_or_else(|| JsValue::from_str("IndexedDB not available"))?;

        let request = idb
            .open_with_u32(DB_NAME, DB_VERSION)
            .map_err(|e| StorageError::from_js(e))?;

        // Set up upgrade handler
        let on_upgrade = Closure::once(Box::new(move |event: web_sys::IdbVersionChangeEvent| {
            let target = event.target().unwrap();
            let request: IdbOpenDbRequest = target.dyn_into().unwrap();
            let db = request.result().unwrap().dyn_into::<IdbDatabase>().unwrap();

            // Create object store if it doesn't exist
            if !store_exists(&db, STORE_NAME) {
                let params = web_sys::IdbObjectStoreParameters::new();
                params.set_key_path(&JsValue::from_str("id"));

                let store = db
                    .create_object_store_with_optional_parameters(STORE_NAME, &params)
                    .unwrap();

                // Create indexes
                let _ = store.create_index_with_str("by_type", "pattern_type");
                let _ = store.create_index_with_str("by_created", "created_at");
            }
        }) as Box<dyn FnOnce(_)>);

        request.set_onupgradeneeded(Some(on_upgrade.as_ref().unchecked_ref()));
        on_upgrade.forget();

        // Wait for success
        let db = Self::await_request(&request).await?;
        self.db = Some(db.dyn_into::<IdbDatabase>()?);

        log::info!("IndexedDB opened successfully");
        Ok(())
    }

    /// Close the database connection.
    #[wasm_bindgen]
    pub fn close(&mut self) {
        if let Some(db) = self.db.take() {
            db.close();
            log::info!("IndexedDB closed");
        }
    }

    /// Save a single pattern.
    #[wasm_bindgen]
    pub async fn save_pattern(&self, pattern_js: JsValue) -> Result<(), JsValue> {
        let db = self.ensure_db()?;

        let transaction = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::from_js(e))?;

        let store = transaction
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::from_js(e))?;

        let request = store
            .put(&pattern_js)
            .map_err(|e| StorageError::from_js(e))?;

        Self::await_request(&request).await?;
        Ok(())
    }

    /// Save multiple patterns.
    #[wasm_bindgen]
    pub async fn save_patterns(&self, patterns_js: JsValue) -> Result<u32, JsValue> {
        let db = self.ensure_db()?;
        let patterns: Array = patterns_js.dyn_into()?;

        let transaction = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::from_js(e))?;

        let store = transaction
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::from_js(e))?;

        let mut count = 0u32;
        for i in 0..patterns.length() {
            let pattern = patterns.get(i);
            store.put(&pattern).map_err(|e| StorageError::from_js(e))?;
            count += 1;
        }

        // Wait for transaction to complete
        Self::await_transaction(&transaction).await?;

        log::info!("Saved {} patterns to IndexedDB", count);
        Ok(count)
    }

    /// Load all patterns.
    #[wasm_bindgen]
    pub async fn load_all_patterns(&self) -> Result<JsValue, JsValue> {
        let db = self.ensure_db()?;

        let transaction = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readonly)
            .map_err(|e| StorageError::from_js(e))?;

        let store = transaction
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::from_js(e))?;

        let request = store.get_all().map_err(|e| StorageError::from_js(e))?;

        let result = Self::await_request(&request).await?;
        Ok(result)
    }

    /// Load a single pattern by ID.
    #[wasm_bindgen]
    pub async fn load_pattern(&self, id: &str) -> Result<JsValue, JsValue> {
        let db = self.ensure_db()?;

        let transaction = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readonly)
            .map_err(|e| StorageError::from_js(e))?;

        let store = transaction
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::from_js(e))?;

        let request = store
            .get(&JsValue::from_str(id))
            .map_err(|e| StorageError::from_js(e))?;

        let result = Self::await_request(&request).await?;
        Ok(result)
    }

    /// Delete a pattern by ID.
    #[wasm_bindgen]
    pub async fn delete_pattern(&self, id: &str) -> Result<(), JsValue> {
        let db = self.ensure_db()?;

        let transaction = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::from_js(e))?;

        let store = transaction
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::from_js(e))?;

        let request = store
            .delete(&JsValue::from_str(id))
            .map_err(|e| StorageError::from_js(e))?;

        Self::await_request(&request).await?;
        Ok(())
    }

    /// Clear all patterns.
    #[wasm_bindgen]
    pub async fn clear(&self) -> Result<(), JsValue> {
        let db = self.ensure_db()?;

        let transaction = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readwrite)
            .map_err(|e| StorageError::from_js(e))?;

        let store = transaction
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::from_js(e))?;

        let request = store.clear().map_err(|e| StorageError::from_js(e))?;

        Self::await_request(&request).await?;
        log::info!("Cleared all patterns from IndexedDB");
        Ok(())
    }

    /// Get the count of stored patterns.
    #[wasm_bindgen]
    pub async fn count(&self) -> Result<u32, JsValue> {
        let db = self.ensure_db()?;

        let transaction = db
            .transaction_with_str_and_mode(STORE_NAME, IdbTransactionMode::Readonly)
            .map_err(|e| StorageError::from_js(e))?;

        let store = transaction
            .object_store(STORE_NAME)
            .map_err(|e| StorageError::from_js(e))?;

        let request = store.count().map_err(|e| StorageError::from_js(e))?;

        let result = Self::await_request(&request).await?;
        Ok(result.as_f64().unwrap_or(0.0) as u32)
    }

    /// Check if the database is connected.
    #[wasm_bindgen]
    pub fn is_connected(&self) -> bool {
        self.db.is_some()
    }

    // Internal helpers

    fn ensure_db(&self) -> Result<&IdbDatabase, JsValue> {
        self.db
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Database not opened. Call open() first."))
    }

    async fn await_request(request: &IdbRequest) -> Result<JsValue, JsValue> {
        let (tx, rx) = futures::channel::oneshot::channel();
        let tx = Rc::new(RefCell::new(Some(tx)));

        let tx_success = tx.clone();
        let on_success = Closure::once(Box::new(move |_event: web_sys::Event| {
            if let Some(tx) = tx_success.borrow_mut().take() {
                let _ = tx.send(Ok(()));
            }
        }) as Box<dyn FnOnce(_)>);

        let tx_error = tx;
        let on_error = Closure::once(Box::new(move |_event: web_sys::Event| {
            if let Some(tx) = tx_error.borrow_mut().take() {
                let _ = tx.send(Err(()));
            }
        }) as Box<dyn FnOnce(_)>);

        request.set_onsuccess(Some(on_success.as_ref().unchecked_ref()));
        request.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        on_success.forget();
        on_error.forget();

        rx.await
            .map_err(|_| JsValue::from_str("Request cancelled"))?
            .map_err(|_| {
                request
                    .error()
                    .ok()
                    .flatten()
                    .map(|e: web_sys::DomException| JsValue::from(e.message()))
                    .unwrap_or_else(|| JsValue::from_str("Unknown error"))
            })?;

        request
            .result()
            .map_err(|e| StorageError::from_js(e).into())
    }

    async fn await_transaction(transaction: &IdbTransaction) -> Result<(), JsValue> {
        let (tx, rx) = futures::channel::oneshot::channel();
        let tx = Rc::new(RefCell::new(Some(tx)));

        let tx_complete = tx.clone();
        let on_complete = Closure::once(Box::new(move |_event: web_sys::Event| {
            if let Some(tx) = tx_complete.borrow_mut().take() {
                let _ = tx.send(Ok(()));
            }
        }) as Box<dyn FnOnce(_)>);

        let tx_error = tx;
        let on_error = Closure::once(Box::new(move |_event: web_sys::Event| {
            if let Some(tx) = tx_error.borrow_mut().take() {
                let _ = tx.send(Err(()));
            }
        }) as Box<dyn FnOnce(_)>);

        transaction.set_oncomplete(Some(on_complete.as_ref().unchecked_ref()));
        transaction.set_onerror(Some(on_error.as_ref().unchecked_ref()));

        on_complete.forget();
        on_error.forget();

        rx.await
            .map_err(|_| JsValue::from_str("Transaction cancelled"))?
            .map_err(|_| {
                if let Some(e) = transaction.error() {
                    JsValue::from_str(&e.message())
                } else {
                    JsValue::from_str("Transaction failed")
                }
            })
    }
}

impl Default for IndexedDBStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct StorageStats {
    /// Number of patterns stored
    pattern_count: u32,

    /// Database name
    db_name: String,

    /// Database version
    db_version: u32,

    /// Whether connected
    connected: bool,
}

#[wasm_bindgen]
impl StorageStats {
    #[wasm_bindgen(getter)]
    pub fn pattern_count(&self) -> u32 {
        self.pattern_count
    }

    #[wasm_bindgen(getter)]
    pub fn db_name(&self) -> String {
        self.db_name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn db_version(&self) -> u32 {
        self.db_version
    }

    #[wasm_bindgen(getter)]
    pub fn connected(&self) -> bool {
        self.connected
    }
}

impl StorageStats {
    pub fn new(pattern_count: u32, connected: bool) -> Self {
        Self {
            pattern_count,
            db_name: DB_NAME.to_string(),
            db_version: DB_VERSION,
            connected,
        }
    }
}
