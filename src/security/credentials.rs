//! Credential management and rotation.
//!
//! Provides secure credential storage, rotation scheduling, and
//! encryption key management for the Nagual system.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rand::{distributions::Alphanumeric, rngs::OsRng, Rng, RngCore};
use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::error::{NagualError, Result};
use crate::security::audit::{AuditEventType, AuditLogger, AuditOutcome};

/// Type of credential being managed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CredentialType {
    /// Database password
    DatabasePassword,
    /// Encryption key for data at rest
    EncryptionKey,
    /// API key for external services
    ApiKey,
    /// JWT signing key
    JwtSigningKey,
    /// Service account credentials
    ServiceAccount,
    /// Backup encryption key
    BackupKey,
    /// SQLite encryption key
    SqliteKey,
    /// Master key (encrypts other keys)
    MasterKey,
}

impl std::fmt::Display for CredentialType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CredentialType::DatabasePassword => write!(f, "database_password"),
            CredentialType::EncryptionKey => write!(f, "encryption_key"),
            CredentialType::ApiKey => write!(f, "api_key"),
            CredentialType::JwtSigningKey => write!(f, "jwt_signing_key"),
            CredentialType::ServiceAccount => write!(f, "service_account"),
            CredentialType::BackupKey => write!(f, "backup_key"),
            CredentialType::SqliteKey => write!(f, "sqlite_key"),
            CredentialType::MasterKey => write!(f, "master_key"),
        }
    }
}

/// Status of a credential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredentialStatus {
    /// Credential is active and valid
    Active,
    /// Credential is scheduled for rotation
    PendingRotation,
    /// Credential is being rotated
    Rotating,
    /// Credential has been rotated (old version)
    Rotated,
    /// Credential has been revoked
    Revoked,
    /// Credential has expired
    Expired,
}

/// Metadata about a managed credential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialMetadata {
    /// Type of credential
    pub credential_type: CredentialType,
    /// Unique identifier
    pub id: String,
    /// Current status
    pub status: CredentialStatus,
    /// When the credential was created
    pub created_at: DateTime<Utc>,
    /// When the credential was last rotated
    pub last_rotated: Option<DateTime<Utc>>,
    /// When the credential expires (if applicable)
    pub expires_at: Option<DateTime<Utc>>,
    /// Rotation interval
    pub rotation_interval: Option<Duration>,
    /// Version number (increments with each rotation)
    pub version: u32,
    /// Description
    pub description: Option<String>,
}

/// Result of a rotation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationResult {
    /// ID of the credential that was rotated
    pub credential_id: String,
    /// New version number
    pub new_version: u32,
    /// When the rotation occurred
    pub rotated_at: DateTime<Utc>,
    /// Whether the rotation was successful
    pub success: bool,
    /// Error message if rotation failed
    pub error: Option<String>,
    /// Time taken to rotate
    pub duration_ms: u64,
}

/// Rotation policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationPolicy {
    /// Minimum rotation interval
    pub min_interval: Duration,
    /// Maximum age before forced rotation
    pub max_age: Duration,
    /// Whether to allow manual rotation
    pub allow_manual: bool,
    /// Number of old versions to retain
    pub retain_versions: usize,
    /// Whether to require approval for rotation
    pub require_approval: bool,
}

impl Default for RotationPolicy {
    fn default() -> Self {
        Self {
            min_interval: Duration::from_secs(24 * 60 * 60), // 1 day minimum
            max_age: Duration::from_secs(90 * 24 * 60 * 60), // 90 days max
            allow_manual: true,
            retain_versions: 2,
            require_approval: false,
        }
    }
}

impl RotationPolicy {
    /// Create a strict rotation policy.
    pub fn strict() -> Self {
        Self {
            min_interval: Duration::from_secs(7 * 24 * 60 * 60), // 7 days minimum
            max_age: Duration::from_secs(30 * 24 * 60 * 60),     // 30 days max
            allow_manual: true,
            retain_versions: 1,
            require_approval: true,
        }
    }

    /// Create a relaxed rotation policy.
    pub fn relaxed() -> Self {
        Self {
            min_interval: Duration::from_secs(60 * 60), // 1 hour minimum
            max_age: Duration::from_secs(365 * 24 * 60 * 60), // 1 year max
            allow_manual: true,
            retain_versions: 3,
            require_approval: false,
        }
    }
}

/// Encrypted credential storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EncryptedCredential {
    /// Encrypted value
    ciphertext: Vec<u8>,
    /// Nonce used for encryption
    nonce: Vec<u8>,
    /// Key version used for encryption
    key_version: u32,
}

/// Credential manager for secure storage and rotation.
pub struct CredentialManager {
    /// Master key for encrypting credentials (zeroized on drop for security)
    master_key: Zeroizing<[u8; 32]>,
    /// Stored credentials (encrypted)
    credentials: RwLock<HashMap<String, EncryptedCredential>>,
    /// Credential metadata
    metadata: RwLock<HashMap<String, CredentialMetadata>>,
    /// Rotation policies by credential type
    policies: RwLock<HashMap<CredentialType, RotationPolicy>>,
    /// Scheduled rotations
    scheduled_rotations: RwLock<HashMap<String, Instant>>,
    /// Audit logger (optional)
    audit_logger: Option<Arc<AuditLogger>>,
    /// Rotation callbacks
    rotation_callbacks: RwLock<HashMap<CredentialType, Box<dyn Fn(&str, &[u8]) -> Result<()> + Send + Sync>>>,
}

impl CredentialManager {
    /// Create a new credential manager with the given master key.
    ///
    /// # Security
    /// The master key should be derived from a secure source (e.g., KMS, HSM, or
    /// key derivation from a master password with Argon2).
    /// The key is wrapped in `Zeroizing` to ensure it is securely erased from memory on drop.
    pub fn new(master_key: [u8; 32]) -> Self {
        Self {
            master_key: Zeroizing::new(master_key),
            credentials: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            policies: RwLock::new(HashMap::new()),
            scheduled_rotations: RwLock::new(HashMap::new()),
            audit_logger: None,
            rotation_callbacks: RwLock::new(HashMap::new()),
        }
    }

    /// Create a credential manager with a derived master key.
    ///
    /// Uses Argon2 to derive a key from the password and salt.
    pub fn from_password(password: &str, salt: &[u8]) -> Result<Self> {
        let master_key = derive_key(password, salt)?;
        Ok(Self::new(master_key))
    }

    /// Set the audit logger for credential operations.
    pub fn with_audit_logger(mut self, logger: Arc<AuditLogger>) -> Self {
        self.audit_logger = Some(logger);
        self
    }

    /// Set a rotation policy for a credential type.
    pub fn set_policy(&self, credential_type: CredentialType, policy: RotationPolicy) {
        self.policies.write().insert(credential_type, policy);
    }

    /// Register a callback to be called after credential rotation.
    pub fn on_rotation<F>(&self, credential_type: CredentialType, callback: F)
    where
        F: Fn(&str, &[u8]) -> Result<()> + Send + Sync + 'static,
    {
        self.rotation_callbacks
            .write()
            .insert(credential_type, Box::new(callback));
    }

    /// Store a new credential.
    pub async fn store(
        &self,
        credential_type: CredentialType,
        id: &str,
        value: &[u8],
        description: Option<&str>,
    ) -> Result<()> {
        // Encrypt the credential
        let encrypted = self.encrypt(value)?;

        // Create metadata
        let metadata = CredentialMetadata {
            credential_type,
            id: id.to_string(),
            status: CredentialStatus::Active,
            created_at: Utc::now(),
            last_rotated: None,
            expires_at: None,
            rotation_interval: self
                .policies
                .read()
                .get(&credential_type)
                .map(|p| p.max_age),
            version: 1,
            description: description.map(|s| s.to_string()),
        };

        // Store
        self.credentials.write().insert(id.to_string(), encrypted);
        self.metadata.write().insert(id.to_string(), metadata);

        // Audit log
        if let Some(ref logger) = self.audit_logger {
            let _ = logger
                .log(
                    logger
                        .builder(AuditEventType::DataCreate, "system", "store_credential")
                        .resource("credential", id)
                        .metadata("credential_type", serde_json::json!(credential_type.to_string()))
                        .build(),
                )
                .await;
        }

        Ok(())
    }

    /// Retrieve a credential.
    pub async fn get(&self, id: &str) -> Result<Vec<u8>> {
        let encrypted = self
            .credentials
            .read()
            .get(id)
            .cloned()
            .ok_or_else(|| NagualError::internal(format!("Credential not found: {}", id)))?;

        // Check if credential is valid
        if let Some(meta) = self.metadata.read().get(id) {
            if meta.status == CredentialStatus::Revoked {
                return Err(NagualError::internal("Credential has been revoked"));
            }
            if meta.status == CredentialStatus::Expired {
                return Err(NagualError::internal("Credential has expired"));
            }
            if let Some(expires_at) = meta.expires_at {
                if expires_at < Utc::now() {
                    return Err(NagualError::internal("Credential has expired"));
                }
            }
        }

        let value = self.decrypt(&encrypted)?;

        // Audit log
        if let Some(ref logger) = self.audit_logger {
            let _ = logger.log_access("system", "credential", id).await;
        }

        Ok(value)
    }

    /// Rotate a credential.
    pub async fn rotate(&self, id: &str) -> Result<RotationResult> {
        let start = Instant::now();
        let rotated_at = Utc::now();

        // Get current metadata
        let mut meta = self
            .metadata
            .read()
            .get(id)
            .cloned()
            .ok_or_else(|| NagualError::internal(format!("Credential not found: {}", id)))?;

        // Check rotation policy
        if let Some(policy) = self.policies.read().get(&meta.credential_type) {
            if let Some(last_rotated) = meta.last_rotated {
                let since_last = (rotated_at - last_rotated).to_std().unwrap_or(Duration::ZERO);
                if since_last < policy.min_interval && !policy.allow_manual {
                    return Err(NagualError::internal(
                        "Rotation interval not met and manual rotation not allowed",
                    ));
                }
            }
        }

        // Generate new credential value
        let new_value = match meta.credential_type {
            CredentialType::DatabasePassword => generate_password(32),
            CredentialType::EncryptionKey
            | CredentialType::BackupKey
            | CredentialType::SqliteKey
            | CredentialType::MasterKey => generate_key(32),
            CredentialType::ApiKey => generate_api_key(),
            CredentialType::JwtSigningKey => generate_key(64),
            CredentialType::ServiceAccount => generate_password(48),
        };

        // Encrypt new value
        let encrypted = self.encrypt(&new_value)?;

        // Update status during rotation
        meta.status = CredentialStatus::Rotating;
        self.metadata.write().insert(id.to_string(), meta.clone());

        // Execute rotation callback if registered
        let callback_result = if let Some(callback) =
            self.rotation_callbacks.read().get(&meta.credential_type)
        {
            callback(id, &new_value)
        } else {
            Ok(())
        };

        // Handle callback result
        let (success, error) = match callback_result {
            Ok(()) => {
                // Update credential
                self.credentials.write().insert(id.to_string(), encrypted);

                // Update metadata
                meta.status = CredentialStatus::Active;
                meta.last_rotated = Some(rotated_at);
                meta.version += 1;

                if let Some(policy) = self.policies.read().get(&meta.credential_type) {
                    meta.expires_at = Some(rotated_at + chrono::Duration::from_std(policy.max_age).unwrap());
                }

                self.metadata.write().insert(id.to_string(), meta.clone());

                (true, None)
            }
            Err(e) => {
                // Revert status
                meta.status = CredentialStatus::Active;
                self.metadata.write().insert(id.to_string(), meta.clone());

                (false, Some(e.to_string()))
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        // Audit log
        if let Some(ref logger) = self.audit_logger {
            let _ = logger
                .log_credential_rotation("system", meta.credential_type.to_string())
                .await;
        }

        Ok(RotationResult {
            credential_id: id.to_string(),
            new_version: meta.version,
            rotated_at,
            success,
            error,
            duration_ms,
        })
    }

    /// Revoke a credential.
    pub async fn revoke(&self, id: &str) -> Result<()> {
        let mut meta = self
            .metadata
            .write();

        if let Some(m) = meta.get_mut(id) {
            m.status = CredentialStatus::Revoked;

            // Audit log
            if let Some(ref logger) = self.audit_logger {
                let _ = logger
                    .log(
                        logger
                            .builder(AuditEventType::DataDelete, "system", "revoke_credential")
                            .resource("credential", id)
                            .outcome(AuditOutcome::Success)
                            .build(),
                    )
                    .await;
            }

            Ok(())
        } else {
            Err(NagualError::internal(format!("Credential not found: {}", id)))
        }
    }

    /// Check which credentials need rotation.
    pub fn check_rotation_needed(&self) -> Vec<(String, CredentialMetadata)> {
        let now = Utc::now();
        let metadata = self.metadata.read();
        let policies = self.policies.read();

        metadata
            .iter()
            .filter(|(_, meta)| {
                if meta.status != CredentialStatus::Active {
                    return false;
                }

                // Check expiration
                if let Some(expires_at) = meta.expires_at {
                    if expires_at <= now {
                        return true;
                    }
                }

                // Check rotation interval
                if let Some(policy) = policies.get(&meta.credential_type) {
                    if let Some(last_rotated) = meta.last_rotated {
                        let since_last = (now - last_rotated).to_std().unwrap_or(Duration::ZERO);
                        if since_last >= policy.max_age {
                            return true;
                        }
                    } else {
                        // Never rotated, check creation time
                        let since_creation = (now - meta.created_at).to_std().unwrap_or(Duration::ZERO);
                        if since_creation >= policy.max_age {
                            return true;
                        }
                    }
                }

                false
            })
            .map(|(id, meta)| (id.clone(), meta.clone()))
            .collect()
    }

    /// Schedule automatic rotation for a credential.
    pub fn schedule_rotation(&self, id: &str, at: Instant) {
        self.scheduled_rotations.write().insert(id.to_string(), at);
    }

    /// Get all scheduled rotations.
    pub fn get_scheduled_rotations(&self) -> Vec<(String, Instant)> {
        self.scheduled_rotations
            .read()
            .iter()
            .map(|(id, time)| (id.clone(), *time))
            .collect()
    }

    /// Process due rotations.
    pub async fn process_due_rotations(&self) -> Vec<RotationResult> {
        let now = Instant::now();
        let due: Vec<String> = self
            .scheduled_rotations
            .read()
            .iter()
            .filter(|(_, time)| **time <= now)
            .map(|(id, _)| id.clone())
            .collect();

        let mut results = Vec::new();
        for id in due {
            self.scheduled_rotations.write().remove(&id);
            match self.rotate(&id).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    results.push(RotationResult {
                        credential_id: id,
                        new_version: 0,
                        rotated_at: Utc::now(),
                        success: false,
                        error: Some(e.to_string()),
                        duration_ms: 0,
                    });
                }
            }
        }

        results
    }

    /// Get credential metadata.
    pub fn get_metadata(&self, id: &str) -> Option<CredentialMetadata> {
        self.metadata.read().get(id).cloned()
    }

    /// List all credential metadata.
    pub fn list_credentials(&self) -> Vec<CredentialMetadata> {
        self.metadata.read().values().cloned().collect()
    }

    /// Encrypt data with the master key.
    fn encrypt(&self, plaintext: &[u8]) -> Result<EncryptedCredential> {
        let cipher = Aes256Gcm::new_from_slice(self.master_key.as_ref())
            .map_err(|e| NagualError::internal(format!("Invalid key: {}", e)))?;

        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| NagualError::internal(format!("Encryption failed: {}", e)))?;

        Ok(EncryptedCredential {
            ciphertext,
            nonce: nonce_bytes.to_vec(),
            key_version: 1,
        })
    }

    /// Decrypt data with the master key.
    fn decrypt(&self, encrypted: &EncryptedCredential) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new_from_slice(self.master_key.as_ref())
            .map_err(|e| NagualError::internal(format!("Invalid key: {}", e)))?;

        let nonce = Nonce::from_slice(&encrypted.nonce);

        cipher
            .decrypt(nonce, encrypted.ciphertext.as_ref())
            .map_err(|e| NagualError::internal(format!("Decryption failed: {}", e)))
    }
}

/// Derive a key from a password using Argon2.
pub fn derive_key(password: &str, salt: &[u8]) -> Result<[u8; 32]> {
    use argon2::{
        password_hash::{PasswordHasher, SaltString},
        Argon2,
    };

    // Ensure salt is valid base64
    let salt_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD_NO_PAD, salt);
    let salt_string = SaltString::from_b64(&salt_b64)
        .map_err(|e| NagualError::internal(format!("Invalid salt: {}", e)))?;

    let argon2 = Argon2::default();
    let hash = argon2
        .hash_password(password.as_bytes(), &salt_string)
        .map_err(|e| NagualError::internal(format!("Key derivation failed: {}", e)))?;

    let hash_bytes = hash
        .hash
        .ok_or_else(|| NagualError::internal("No hash output"))?;

    let mut key = [0u8; 32];
    let output = hash_bytes.as_bytes();
    let len = std::cmp::min(output.len(), 32);
    key[..len].copy_from_slice(&output[..len]);

    Ok(key)
}

/// Generate a secure random password.
pub fn generate_password(length: usize) -> Vec<u8> {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(length)
        .map(|c| c as u8)
        .collect()
}

/// Generate a secure random key.
pub fn generate_key(length: usize) -> Vec<u8> {
    let mut key = vec![0u8; length];
    OsRng.fill_bytes(&mut key);
    key
}

/// Generate an API key with a prefix.
pub fn generate_api_key() -> Vec<u8> {
    let random: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    format!("nag_{}", random).into_bytes()
}

/// Generate a secure salt.
pub fn generate_salt() -> [u8; 16] {
    let mut salt = [0u8; 16];
    OsRng.fill_bytes(&mut salt);
    salt
}

/// Hash a password for verification.
pub fn hash_password(password: &str) -> Result<String> {
    use argon2::{
        password_hash::{PasswordHasher, SaltString},
        Argon2,
    };

    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();

    let hash = argon2
        .hash_password(password.as_bytes(), &salt)
        .map_err(|e| NagualError::internal(format!("Password hashing failed: {}", e)))?;

    Ok(hash.to_string())
}

/// Verify a password against a hash.
pub fn verify_password(password: &str, hash: &str) -> bool {
    use argon2::{password_hash::PasswordVerifier, Argon2};

    let parsed = match argon2::password_hash::PasswordHash::new(hash) {
        Ok(h) => h,
        Err(_) => return false,
    };

    Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

/// Compute SHA-256 hash of data.
pub fn sha256_hash(data: &[u8]) -> String {
    let hash = digest(&SHA256, data);
    hex::encode(hash.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_password() {
        let password = generate_password(32);
        assert_eq!(password.len(), 32);

        // Should be alphanumeric
        assert!(password.iter().all(|&c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn test_generate_key() {
        let key = generate_key(32);
        assert_eq!(key.len(), 32);

        // Should have high entropy (unlikely to be all zeros)
        assert!(key.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_generate_api_key() {
        let key = generate_api_key();
        let key_str = String::from_utf8(key).unwrap();

        assert!(key_str.starts_with("nag_"));
        assert!(key_str.len() > 4);
    }

    #[test]
    fn test_password_hash_verify() {
        let password = "my_secure_password";
        let hash = hash_password(password).unwrap();

        assert!(verify_password(password, &hash));
        assert!(!verify_password("wrong_password", &hash));
    }

    #[test]
    fn test_sha256_hash() {
        let data = b"hello world";
        let hash = sha256_hash(data);

        // SHA-256 produces 64 hex characters
        assert_eq!(hash.len(), 64);

        // Same input should produce same output
        assert_eq!(sha256_hash(data), hash);

        // Different input should produce different output
        assert_ne!(sha256_hash(b"different"), hash);
    }

    #[tokio::test]
    async fn test_credential_manager_store_get() {
        let master_key = generate_key(32).try_into().unwrap();
        let manager = CredentialManager::new(master_key);

        let credential_value = b"my_secret_password";
        manager
            .store(
                CredentialType::DatabasePassword,
                "db_password",
                credential_value,
                Some("PostgreSQL password"),
            )
            .await
            .unwrap();

        let retrieved = manager.get("db_password").await.unwrap();
        assert_eq!(retrieved, credential_value);
    }

    #[tokio::test]
    async fn test_credential_rotation() {
        let master_key = generate_key(32).try_into().unwrap();
        let manager = CredentialManager::new(master_key);

        // Set a relaxed policy for testing
        manager.set_policy(CredentialType::DatabasePassword, RotationPolicy::relaxed());

        // Store initial credential
        manager
            .store(CredentialType::DatabasePassword, "test_cred", b"initial", None)
            .await
            .unwrap();

        // Rotate
        let result = manager.rotate("test_cred").await.unwrap();
        assert!(result.success);
        assert_eq!(result.new_version, 2);

        // New value should be different
        let new_value = manager.get("test_cred").await.unwrap();
        assert_ne!(new_value, b"initial");
    }

    #[tokio::test]
    async fn test_credential_revocation() {
        let master_key = generate_key(32).try_into().unwrap();
        let manager = CredentialManager::new(master_key);

        manager
            .store(CredentialType::ApiKey, "api_key_1", b"secret", None)
            .await
            .unwrap();

        // Revoke
        manager.revoke("api_key_1").await.unwrap();

        // Should fail to retrieve
        let result = manager.get("api_key_1").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_credential_type_display() {
        assert_eq!(CredentialType::DatabasePassword.to_string(), "database_password");
        assert_eq!(CredentialType::EncryptionKey.to_string(), "encryption_key");
        assert_eq!(CredentialType::ApiKey.to_string(), "api_key");
    }

    #[test]
    fn test_check_rotation_needed() {
        let master_key = generate_key(32).try_into().unwrap();
        let manager = CredentialManager::new(master_key);

        // Set a policy with very short max_age for testing
        manager.set_policy(
            CredentialType::ApiKey,
            RotationPolicy {
                min_interval: Duration::from_secs(0),
                max_age: Duration::from_secs(0), // Immediate expiration
                allow_manual: true,
                retain_versions: 1,
                require_approval: false,
            },
        );

        // Store a credential
        let metadata = CredentialMetadata {
            credential_type: CredentialType::ApiKey,
            id: "test_api_key".to_string(),
            status: CredentialStatus::Active,
            created_at: Utc::now() - chrono::Duration::hours(1), // Created 1 hour ago
            last_rotated: None,
            expires_at: None,
            rotation_interval: Some(Duration::from_secs(0)),
            version: 1,
            description: None,
        };

        manager
            .metadata
            .write()
            .insert("test_api_key".to_string(), metadata);

        let needs_rotation = manager.check_rotation_needed();
        assert!(!needs_rotation.is_empty());
        assert!(needs_rotation.iter().any(|(id, _)| id == "test_api_key"));
    }
}
