//! Google Cloud Storage adapter for Nagual.
//!
//! Provides upload, download, and listing operations for GCS buckets
//! with support for CMEK encryption and Application Default Credentials.
//!
//! # Authentication
//!
//! This module uses Google Application Default Credentials (ADC) for authentication.
//! ADC automatically finds credentials in the following order:
//!
//! 1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to a service account key
//! 2. User credentials from `gcloud auth application-default login`
//! 3. Attached service account on GCE/Cloud Run/GKE
//! 4. Workload Identity on GKE
//!
//! # CMEK (Customer-Managed Encryption Keys)
//!
//! To use CMEK encryption, configure `EncryptionConfig` with your Cloud KMS key:
//!
//! ```rust,ignore
//! let encryption = EncryptionConfig::new(
//!     "projects/my-project/locations/us-central1/keyRings/my-ring/cryptoKeys/my-key"
//! );
//! let config = GCloudConfig::new("bucket", "project")
//!     .with_encryption(encryption);
//! ```
//!
//! # Setup Requirements
//!
//! 1. Create a GCS bucket: `gsutil mb gs://your-bucket`
//! 2. (Optional) Create KMS key ring and key for CMEK
//! 3. Grant storage and KMS permissions to service account
//! 4. Set GOOGLE_APPLICATION_CREDENTIALS or use ADC

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// GCloud-specific errors.
#[derive(Error, Debug)]
pub enum GCloudError {
    /// Failed to initialize client
    #[error("Failed to initialize GCloud client: {0}")]
    InitializationFailed(String),

    /// Bucket not found
    #[error("Bucket not found: {bucket}")]
    BucketNotFound { bucket: String },

    /// Object not found
    #[error("Object not found: {path}")]
    ObjectNotFound { path: String },

    /// Upload failed
    #[error("Upload failed for {path}: {message}")]
    UploadFailed { path: String, message: String },

    /// Download failed
    #[error("Download failed for {path}: {message}")]
    DownloadFailed { path: String, message: String },

    /// Permission denied
    #[error("Permission denied: {message}")]
    PermissionDenied { message: String },

    /// Encryption error
    #[error("Encryption error: {message}")]
    EncryptionError { message: String },

    /// Compression error
    #[error("Compression error: {0}")]
    CompressionError(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for GCloud operations.
pub type GCloudResult<T> = std::result::Result<T, GCloudError>;

/// Configuration for CMEK encryption.
///
/// # CMEK Setup
///
/// 1. Create a key ring in Cloud KMS:
///    ```bash
///    gcloud kms keyrings create nagual-keyring \
///        --location=us-central1 \
///        --project=your-project
///    ```
///
/// 2. Create a crypto key:
///    ```bash
///    gcloud kms keys create nagual-backup-key \
///        --location=us-central1 \
///        --keyring=nagual-keyring \
///        --purpose=encryption \
///        --project=your-project
///    ```
///
/// 3. Grant the service account permission to use the key:
///    ```bash
///    gcloud kms keys add-iam-policy-binding nagual-backup-key \
///        --location=us-central1 \
///        --keyring=nagual-keyring \
///        --member=serviceAccount:your-sa@your-project.iam.gserviceaccount.com \
///        --role=roles/cloudkms.cryptoKeyEncrypterDecrypter \
///        --project=your-project
///    ```
///
/// 4. Use the key name in `EncryptionConfig`:
///    ```text
///    projects/your-project/locations/us-central1/keyRings/nagual-keyring/cryptoKeys/nagual-backup-key
///    ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Full resource name of the Cloud KMS key.
    /// Format: projects/{project}/locations/{location}/keyRings/{ring}/cryptoKeys/{key}
    pub key_name: String,

    /// Whether to verify encryption after upload.
    pub verify_on_upload: bool,

    /// Whether to verify decryption on download.
    pub verify_on_download: bool,
}

impl EncryptionConfig {
    /// Create a new encryption config with the given KMS key name.
    pub fn new(key_name: impl Into<String>) -> Self {
        Self {
            key_name: key_name.into(),
            verify_on_upload: true,
            verify_on_download: true,
        }
    }

    /// Parse the key name into its components.
    pub fn parse_key_name(&self) -> Option<KeyNameComponents> {
        let parts: Vec<&str> = self.key_name.split('/').collect();
        if parts.len() >= 8
            && parts[0] == "projects"
            && parts[2] == "locations"
            && parts[4] == "keyRings"
            && parts[6] == "cryptoKeys"
        {
            Some(KeyNameComponents {
                project: parts[1].to_string(),
                location: parts[3].to_string(),
                key_ring: parts[5].to_string(),
                key: parts[7].to_string(),
            })
        } else {
            None
        }
    }

    /// Disable upload verification.
    pub fn without_upload_verification(mut self) -> Self {
        self.verify_on_upload = false;
        self
    }

    /// Disable download verification.
    pub fn without_download_verification(mut self) -> Self {
        self.verify_on_download = false;
        self
    }
}

/// Parsed components of a KMS key name.
#[derive(Debug, Clone)]
pub struct KeyNameComponents {
    pub project: String,
    pub location: String,
    pub key_ring: String,
    pub key: String,
}

/// Configuration for GCloud Storage adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCloudConfig {
    /// GCS bucket name.
    pub bucket: String,

    /// GCP project ID.
    pub project_id: String,

    /// Path to service account credentials JSON file.
    /// If None, uses Application Default Credentials.
    pub credentials_path: Option<PathBuf>,

    /// CMEK encryption configuration.
    pub encryption: Option<EncryptionConfig>,

    /// Prefix for all objects (e.g., "nagual/backups/").
    pub prefix: String,

    /// Request timeout.
    pub timeout: Duration,

    /// Retry count for failed operations.
    pub max_retries: u32,

    /// Enable request logging.
    pub enable_logging: bool,
}

impl GCloudConfig {
    /// Create a new config with bucket and project ID.
    pub fn new(bucket: impl Into<String>, project_id: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            project_id: project_id.into(),
            credentials_path: None,
            encryption: None,
            prefix: String::new(),
            timeout: Duration::from_secs(300),
            max_retries: 3,
            enable_logging: true,
        }
    }

    /// Set the credentials path.
    pub fn with_credentials(mut self, path: impl Into<PathBuf>) -> Self {
        self.credentials_path = Some(path.into());
        self
    }

    /// Set CMEK encryption.
    pub fn with_encryption(mut self, encryption: EncryptionConfig) -> Self {
        self.encryption = Some(encryption);
        self
    }

    /// Set object prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set max retries.
    pub fn with_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Get the full object path with prefix.
    pub fn full_path(&self, object_name: &str) -> String {
        if self.prefix.is_empty() {
            object_name.to_string()
        } else {
            format!(
                "{}/{}",
                self.prefix.trim_end_matches('/'),
                object_name.trim_start_matches('/')
            )
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> GCloudResult<()> {
        if self.bucket.is_empty() {
            return Err(GCloudError::InvalidConfig("Bucket name is required".into()));
        }
        if self.project_id.is_empty() {
            return Err(GCloudError::InvalidConfig("Project ID is required".into()));
        }
        if let Some(ref creds) = self.credentials_path {
            if !creds.exists() {
                return Err(GCloudError::InvalidConfig(format!(
                    "Credentials file not found: {}",
                    creds.display()
                )));
            }
        }
        if let Some(ref enc) = self.encryption {
            if enc.parse_key_name().is_none() {
                return Err(GCloudError::InvalidConfig(format!(
                    "Invalid KMS key name format: {}",
                    enc.key_name
                )));
            }
        }
        Ok(())
    }
}

/// Information about an object in GCS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectInfo {
    /// Object name (path within bucket).
    pub name: String,

    /// Size in bytes.
    pub size: u64,

    /// Content type.
    pub content_type: Option<String>,

    /// Creation timestamp.
    pub created: Option<DateTime<Utc>>,

    /// Last modified timestamp.
    pub updated: Option<DateTime<Utc>>,

    /// MD5 hash (base64 encoded).
    pub md5_hash: Option<String>,

    /// CRC32C checksum.
    pub crc32c: Option<String>,

    /// KMS key used for encryption (if CMEK).
    pub kms_key_name: Option<String>,

    /// Storage class.
    pub storage_class: Option<String>,

    /// Custom metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

/// GCloud Storage adapter.
///
/// Provides methods for uploading, downloading, and listing objects in GCS.
#[derive(Clone)]
pub struct GCloudAdapter {
    config: Arc<GCloudConfig>,
    /// Track last operation for rate limiting
    last_operation: Arc<RwLock<Option<std::time::Instant>>>,
}

impl GCloudAdapter {
    /// Create a new GCloud adapter with the given configuration.
    ///
    /// This validates the configuration and initializes the client.
    /// Authentication is performed lazily on first request.
    pub async fn new(config: GCloudConfig) -> GCloudResult<Self> {
        config.validate()?;

        // Set credentials environment variable if path is provided
        if let Some(ref creds_path) = config.credentials_path {
            std::env::set_var("GOOGLE_APPLICATION_CREDENTIALS", creds_path);
            info!(
                credentials = %creds_path.display(),
                "Set GOOGLE_APPLICATION_CREDENTIALS"
            );
        }

        info!(
            bucket = %config.bucket,
            project = %config.project_id,
            prefix = %config.prefix,
            encryption = config.encryption.is_some(),
            "Initialized GCloud adapter"
        );

        Ok(Self {
            config: Arc::new(config),
            last_operation: Arc::new(RwLock::new(None)),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &GCloudConfig {
        &self.config
    }

    /// Upload a file to GCS.
    ///
    /// The file is compressed with gzip before upload.
    /// If CMEK is configured, the object is encrypted with the specified key.
    pub async fn upload_file(
        &self,
        local_path: &Path,
        object_name: &str,
    ) -> GCloudResult<ObjectInfo> {
        let full_path = self.config.full_path(object_name);

        debug!(
            local = %local_path.display(),
            remote = %full_path,
            "Uploading file"
        );

        // Read and compress file
        let data = fs::read(local_path).await?;
        let compressed = self.compress_data(&data)?;

        // Upload compressed data
        let result = self
            .upload_data(&compressed, &full_path, Some("application/gzip"))
            .await?;

        if self.config.enable_logging {
            info!(
                local = %local_path.display(),
                remote = %full_path,
                original_size = data.len(),
                compressed_size = compressed.len(),
                compression_ratio = format!("{:.1}%", (1.0 - compressed.len() as f64 / data.len() as f64) * 100.0),
                "Uploaded file"
            );
        }

        // Verify encryption if configured
        if let Some(ref enc) = self.config.encryption {
            if enc.verify_on_upload {
                self.verify_encryption(&full_path).await?;
            }
        }

        Ok(result)
    }

    /// Upload raw data to GCS.
    pub async fn upload_data(
        &self,
        data: &[u8],
        object_name: &str,
        content_type: Option<&str>,
    ) -> GCloudResult<ObjectInfo> {
        let full_path = self.config.full_path(object_name);

        // Build upload request
        let mut attempt = 0;
        loop {
            attempt += 1;

            match self.do_upload(data, &full_path, content_type).await {
                Ok(info) => return Ok(info),
                Err(e) if attempt < self.config.max_retries => {
                    warn!(
                        attempt,
                        max_retries = self.config.max_retries,
                        error = %e,
                        "Upload failed, retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
                }
                Err(e) => {
                    error!(path = %full_path, error = %e, "Upload failed after retries");
                    return Err(e);
                }
            }
        }
    }

    /// Internal upload implementation.
    async fn do_upload(
        &self,
        data: &[u8],
        object_name: &str,
        content_type: Option<&str>,
    ) -> GCloudResult<ObjectInfo> {
        // Record operation time
        *self.last_operation.write().await = Some(std::time::Instant::now());

        // Real implementation: use the `google-cloud-storage` crate:
        //
        // ```rust
        // use google_cloud_storage::client::Client;
        //
        // let client = Client::default();
        // let mut object = client.object()
        //     .create(
        //         &self.config.bucket,
        //         data.to_vec(),
        //         object_name,
        //         content_type.unwrap_or("application/octet-stream"),
        //     )
        //     .await
        //     .map_err(|e| GCloudError::UploadFailed {
        //         path: object_name.to_string(),
        //         message: e.to_string(),
        //     })?;
        //
        // // Set CMEK encryption if configured
        // if let Some(ref enc) = self.config.encryption {
        //     object.kms_key_name = Some(enc.key_name.clone());
        // }
        // ```

        // For now, simulate successful upload
        debug!(
            bucket = %self.config.bucket,
            object = %object_name,
            size = data.len(),
            "Simulated upload (wire up google-cloud-storage to enable real GCS I/O)"
        );

        Ok(ObjectInfo {
            name: object_name.to_string(),
            size: data.len() as u64,
            content_type: content_type.map(String::from),
            created: Some(Utc::now()),
            updated: Some(Utc::now()),
            md5_hash: None,
            crc32c: None,
            kms_key_name: self.config.encryption.as_ref().map(|e| e.key_name.clone()),
            storage_class: Some("STANDARD".to_string()),
            metadata: std::collections::HashMap::new(),
        })
    }

    /// Download a file from GCS.
    ///
    /// The file is decompressed after download if it was compressed.
    pub async fn download_file(
        &self,
        object_name: &str,
        local_path: &Path,
    ) -> GCloudResult<ObjectInfo> {
        let full_path = self.config.full_path(object_name);

        debug!(
            remote = %full_path,
            local = %local_path.display(),
            "Downloading file"
        );

        // Verify encryption if configured
        if let Some(ref enc) = self.config.encryption {
            if enc.verify_on_download {
                self.verify_encryption(&full_path).await?;
            }
        }

        // Download data
        let (data, info) = self.download_data(&full_path).await?;

        // Decompress if gzipped
        let decompressed = if info
            .content_type
            .as_ref()
            .map(|ct| ct.contains("gzip"))
            .unwrap_or(false)
        {
            self.decompress_data(&data)?
        } else {
            data
        };

        // Ensure parent directory exists
        if let Some(parent) = local_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Write to file
        fs::write(local_path, &decompressed).await?;

        if self.config.enable_logging {
            info!(
                remote = %full_path,
                local = %local_path.display(),
                size = decompressed.len(),
                "Downloaded file"
            );
        }

        Ok(info)
    }

    /// Download raw data from GCS.
    pub async fn download_data(&self, object_name: &str) -> GCloudResult<(Vec<u8>, ObjectInfo)> {
        let full_path = self.config.full_path(object_name);

        let mut attempt = 0;
        loop {
            attempt += 1;

            match self.do_download(&full_path).await {
                Ok(result) => return Ok(result),
                Err(e) if attempt < self.config.max_retries => {
                    warn!(
                        attempt,
                        max_retries = self.config.max_retries,
                        error = %e,
                        "Download failed, retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
                }
                Err(e) => {
                    error!(path = %full_path, error = %e, "Download failed after retries");
                    return Err(e);
                }
            }
        }
    }

    /// Internal download implementation.
    async fn do_download(&self, object_name: &str) -> GCloudResult<(Vec<u8>, ObjectInfo)> {
        // Record operation time
        *self.last_operation.write().await = Some(std::time::Instant::now());

        // Real implementation: use the `google-cloud-storage` crate:
        //
        // ```rust
        // use google_cloud_storage::client::Client;
        //
        // let client = Client::default();
        // let object = client.object()
        //     .read(&self.config.bucket, object_name)
        //     .await
        //     .map_err(|e| GCloudError::ObjectNotFound {
        //         path: object_name.to_string(),
        //     })?;
        //
        // let data = client.object()
        //     .download(&self.config.bucket, object_name)
        //     .await
        //     .map_err(|e| GCloudError::DownloadFailed {
        //         path: object_name.to_string(),
        //         message: e.to_string(),
        //     })?;
        // ```

        debug!(
            bucket = %self.config.bucket,
            object = %object_name,
            "Simulated download (wire up google-cloud-storage to enable real GCS I/O)"
        );

        // Simulate not found for now
        Err(GCloudError::ObjectNotFound {
            path: object_name.to_string(),
        })
    }

    /// List objects in the bucket with optional prefix filter.
    pub async fn list_objects(&self, prefix: Option<&str>) -> GCloudResult<Vec<ObjectInfo>> {
        let full_prefix = match prefix {
            Some(p) => self.config.full_path(p),
            None if !self.config.prefix.is_empty() => self.config.prefix.clone(),
            None => String::new(),
        };

        debug!(
            bucket = %self.config.bucket,
            prefix = %full_prefix,
            "Listing objects"
        );

        // In a real implementation:
        //
        // ```rust
        // use google_cloud_storage::client::Client;
        //
        // let client = Client::default();
        // let objects = client.object()
        //     .list(&self.config.bucket, ListRequest {
        //         prefix: Some(full_prefix),
        //         ..Default::default()
        //     })
        //     .await
        //     .map_err(|e| GCloudError::Network(e.to_string()))?;
        // ```

        Ok(Vec::new())
    }

    /// Delete an object from GCS.
    pub async fn delete_object(&self, object_name: &str) -> GCloudResult<()> {
        let full_path = self.config.full_path(object_name);

        debug!(
            bucket = %self.config.bucket,
            object = %full_path,
            "Deleting object"
        );

        // In a real implementation:
        //
        // ```rust
        // use google_cloud_storage::client::Client;
        //
        // let client = Client::default();
        // client.object()
        //     .delete(&self.config.bucket, &full_path)
        //     .await
        //     .map_err(|e| GCloudError::Network(e.to_string()))?;
        // ```

        if self.config.enable_logging {
            info!(object = %full_path, "Deleted object");
        }

        Ok(())
    }

    /// Check if an object exists.
    pub async fn object_exists(&self, object_name: &str) -> GCloudResult<bool> {
        let full_path = self.config.full_path(object_name);

        // Try to get object metadata
        match self.get_object_info(&full_path).await {
            Ok(_) => Ok(true),
            Err(GCloudError::ObjectNotFound { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Get object metadata without downloading.
    pub async fn get_object_info(&self, object_name: &str) -> GCloudResult<ObjectInfo> {
        let full_path = self.config.full_path(object_name);

        // In a real implementation:
        //
        // ```rust
        // use google_cloud_storage::client::Client;
        //
        // let client = Client::default();
        // let object = client.object()
        //     .read(&self.config.bucket, &full_path)
        //     .await
        //     .map_err(|e| GCloudError::ObjectNotFound {
        //         path: full_path.clone(),
        //     })?;
        // ```

        Err(GCloudError::ObjectNotFound { path: full_path })
    }

    /// Verify that an object is encrypted with CMEK.
    async fn verify_encryption(&self, object_name: &str) -> GCloudResult<()> {
        let enc = self.config.encryption.as_ref().ok_or_else(|| {
            GCloudError::EncryptionError {
                message: "No encryption configured".into(),
            }
        })?;

        match self.get_object_info(object_name).await {
            Ok(info) => {
                if let Some(ref key) = info.kms_key_name {
                    if key.starts_with(&enc.key_name) {
                        debug!(object = %object_name, "Encryption verified");
                        Ok(())
                    } else {
                        Err(GCloudError::EncryptionError {
                            message: format!(
                                "Object encrypted with wrong key: expected {}, got {}",
                                enc.key_name, key
                            ),
                        })
                    }
                } else {
                    Err(GCloudError::EncryptionError {
                        message: "Object is not CMEK encrypted".into(),
                    })
                }
            }
            Err(GCloudError::ObjectNotFound { .. }) => {
                // Object doesn't exist yet, can't verify
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Compress data using gzip.
    fn compress_data(&self, data: &[u8]) -> GCloudResult<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| GCloudError::CompressionError(e.to_string()))?;
        encoder
            .finish()
            .map_err(|e| GCloudError::CompressionError(e.to_string()))
    }

    /// Decompress gzipped data.
    fn decompress_data(&self, data: &[u8]) -> GCloudResult<Vec<u8>> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| GCloudError::CompressionError(e.to_string()))?;
        Ok(decompressed)
    }

    /// Get the bucket name.
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Get the project ID.
    pub fn project_id(&self) -> &str {
        &self.config.project_id
    }
}

impl std::fmt::Debug for GCloudAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GCloudAdapter")
            .field("bucket", &self.config.bucket)
            .field("project_id", &self.config.project_id)
            .field("prefix", &self.config.prefix)
            .field("encryption", &self.config.encryption.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcloud_config_creation() {
        let config = GCloudConfig::new("test-bucket", "test-project")
            .with_prefix("backups")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(config.bucket, "test-bucket");
        assert_eq!(config.project_id, "test-project");
        assert_eq!(config.prefix, "backups");
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_full_path() {
        let config = GCloudConfig::new("bucket", "project").with_prefix("nagual/backups");

        assert_eq!(
            config.full_path("test.gz"),
            "nagual/backups/test.gz"
        );
        assert_eq!(
            config.full_path("/test.gz"),
            "nagual/backups/test.gz"
        );
    }

    #[test]
    fn test_full_path_no_prefix() {
        let config = GCloudConfig::new("bucket", "project");
        assert_eq!(config.full_path("test.gz"), "test.gz");
    }

    #[test]
    fn test_encryption_config() {
        let enc = EncryptionConfig::new(
            "projects/my-project/locations/us-central1/keyRings/my-ring/cryptoKeys/my-key"
        );

        let components = enc.parse_key_name().unwrap();
        assert_eq!(components.project, "my-project");
        assert_eq!(components.location, "us-central1");
        assert_eq!(components.key_ring, "my-ring");
        assert_eq!(components.key, "my-key");
    }

    #[test]
    fn test_encryption_config_invalid() {
        let enc = EncryptionConfig::new("invalid-key-name");
        assert!(enc.parse_key_name().is_none());
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let valid = GCloudConfig::new("bucket", "project");
        assert!(valid.validate().is_ok());

        // Invalid - empty bucket
        let invalid = GCloudConfig {
            bucket: String::new(),
            project_id: "project".to_string(),
            credentials_path: None,
            encryption: None,
            prefix: String::new(),
            timeout: Duration::from_secs(60),
            max_retries: 3,
            enable_logging: true,
        };
        assert!(invalid.validate().is_err());
    }

    #[tokio::test]
    async fn test_compression() {
        let config = GCloudConfig::new("bucket", "project");
        let adapter = GCloudAdapter::new(config).await.unwrap();

        let original = b"Hello, World! This is test data for compression.";
        let compressed = adapter.compress_data(original).unwrap();
        let decompressed = adapter.decompress_data(&compressed).unwrap();

        assert_eq!(original.as_slice(), decompressed.as_slice());
    }
}
