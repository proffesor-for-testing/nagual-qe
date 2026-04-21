//! Cryptographic utilities for database encryption.
//!
//! Provides key derivation using Argon2id, salt management,
//! and secure key handling for SQLCipher encryption.

use argon2::{
    password_hash::SaltString,
    Argon2, Params, Version,
};
use rand::rngs::OsRng;
use std::fmt;
use thiserror::Error;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Errors related to cryptographic operations.
#[derive(Error, Debug)]
pub enum CryptoError {
    /// Key derivation failed
    #[error("Key derivation failed: {0}")]
    KeyDerivation(String),

    /// Invalid salt
    #[error("Invalid salt: {0}")]
    InvalidSalt(String),

    /// Salt generation failed
    #[error("Salt generation failed: {0}")]
    SaltGeneration(String),

    /// Invalid key length
    #[error("Invalid key length: expected {expected}, got {actual}")]
    InvalidKeyLength { expected: usize, actual: usize },

    /// Password too short
    #[error("Password too short: minimum {min_length} characters required")]
    PasswordTooShort { min_length: usize },

    /// Invalid parameters
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
}

/// Result type for crypto operations.
pub type CryptoResult<T> = std::result::Result<T, CryptoError>;

/// Argon2id parameters for key derivation.
///
/// These follow OWASP recommendations:
/// - m (memory): 65536 KiB (64 MiB)
/// - t (iterations): 3
/// - p (parallelism): 4
#[derive(Debug, Clone)]
pub struct Argon2Params {
    /// Memory cost in KiB
    pub memory_cost: u32,
    /// Time cost (iterations)
    pub time_cost: u32,
    /// Parallelism factor
    pub parallelism: u32,
    /// Output key length in bytes
    pub output_length: usize,
}

impl Default for Argon2Params {
    fn default() -> Self {
        Self {
            memory_cost: 65536, // 64 MiB
            time_cost: 3,
            parallelism: 4,
            output_length: 32, // 256-bit key
        }
    }
}

impl Argon2Params {
    /// Create params for testing (faster but less secure).
    pub fn for_testing() -> Self {
        Self {
            memory_cost: 1024,   // 1 MiB
            time_cost: 1,
            parallelism: 1,
            output_length: 32,
        }
    }

    /// Create high-security params (slower but more secure).
    pub fn high_security() -> Self {
        Self {
            memory_cost: 131072, // 128 MiB
            time_cost: 4,
            parallelism: 4,
            output_length: 32,
        }
    }

    /// Convert to argon2 crate Params.
    fn to_argon2_params(&self) -> CryptoResult<Params> {
        Params::new(
            self.memory_cost,
            self.time_cost,
            self.parallelism,
            Some(self.output_length),
        )
        .map_err(|e| CryptoError::InvalidParams(e.to_string()))
    }
}

/// A securely managed encryption key that is zeroed on drop.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct DerivedKey {
    bytes: Vec<u8>,
}

impl DerivedKey {
    /// Create a new derived key from raw bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Get the key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get the key as a hex string (for SQLCipher PRAGMA key).
    pub fn as_hex(&self) -> String {
        hex_encode(&self.bytes)
    }

    /// Get the key length in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Check if the key is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl fmt::Debug for DerivedKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DerivedKey([REDACTED {} bytes])", self.bytes.len())
    }
}

/// A salt for key derivation.
#[derive(Clone)]
pub struct Salt {
    bytes: [u8; 16],
}

impl Salt {
    /// Generate a new random salt.
    pub fn generate() -> CryptoResult<Self> {
        let salt_string = SaltString::generate(&mut OsRng);
        let salt_str = salt_string.as_str();

        // Take the first 16 bytes (128 bits) of the encoded salt
        let mut bytes = [0u8; 16];
        let salt_bytes = salt_str.as_bytes();
        let len = salt_bytes.len().min(16);
        bytes[..len].copy_from_slice(&salt_bytes[..len]);

        Ok(Self { bytes })
    }

    /// Create a salt from raw bytes.
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Self { bytes }
    }

    /// Create a salt from a hex string.
    pub fn from_hex(hex: &str) -> CryptoResult<Self> {
        let decoded = hex_decode(hex)?;
        if decoded.len() != 16 {
            return Err(CryptoError::InvalidSalt(format!(
                "expected 16 bytes, got {}",
                decoded.len()
            )));
        }
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&decoded);
        Ok(Self { bytes })
    }

    /// Get the salt as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get the salt as a hex string.
    pub fn as_hex(&self) -> String {
        hex_encode(&self.bytes)
    }
}

impl fmt::Debug for Salt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Salt({})", self.as_hex())
    }
}

/// Key derivation service using Argon2id.
pub struct KeyDerivation {
    params: Argon2Params,
}

impl KeyDerivation {
    /// Create a new key derivation service with default parameters.
    pub fn new() -> Self {
        Self {
            params: Argon2Params::default(),
        }
    }

    /// Create a key derivation service with custom parameters.
    pub fn with_params(params: Argon2Params) -> Self {
        Self { params }
    }

    /// Derive an encryption key from a password and salt.
    ///
    /// Uses Argon2id which is resistant to both side-channel attacks
    /// and GPU-based cracking.
    pub fn derive_key(&self, password: &str, salt: &Salt) -> CryptoResult<DerivedKey> {
        self.validate_password(password)?;

        let argon2_params = self.params.to_argon2_params()?;
        let argon2 = Argon2::new(argon2::Algorithm::Argon2id, Version::V0x13, argon2_params);

        let mut output = vec![0u8; self.params.output_length];

        argon2
            .hash_password_into(password.as_bytes(), salt.as_bytes(), &mut output)
            .map_err(|e| CryptoError::KeyDerivation(e.to_string()))?;

        Ok(DerivedKey::new(output))
    }

    /// Derive a key and generate a new random salt.
    ///
    /// Returns both the derived key and the salt that should be stored.
    pub fn derive_key_with_new_salt(&self, password: &str) -> CryptoResult<(DerivedKey, Salt)> {
        let salt = Salt::generate()?;
        let key = self.derive_key(password, &salt)?;
        Ok((key, salt))
    }

    /// Validate password meets minimum requirements.
    fn validate_password(&self, password: &str) -> CryptoResult<()> {
        const MIN_PASSWORD_LENGTH: usize = 8;
        if password.len() < MIN_PASSWORD_LENGTH {
            return Err(CryptoError::PasswordTooShort {
                min_length: MIN_PASSWORD_LENGTH,
            });
        }
        Ok(())
    }

    /// Get the current parameters.
    pub fn params(&self) -> &Argon2Params {
        &self.params
    }
}

impl Default for KeyDerivation {
    fn default() -> Self {
        Self::new()
    }
}

/// Derive an encryption key from a password and salt using default parameters.
///
/// This is a convenience function for simple use cases.
pub fn derive_key_from_password(password: &str, salt: &Salt) -> CryptoResult<DerivedKey> {
    KeyDerivation::new().derive_key(password, salt)
}

/// Derive a key with a new random salt using default parameters.
pub fn derive_key_with_salt(password: &str) -> CryptoResult<(DerivedKey, Salt)> {
    KeyDerivation::new().derive_key_with_new_salt(password)
}

/// Encode bytes to hex string.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Decode hex string to bytes.
fn hex_decode(hex: &str) -> CryptoResult<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return Err(CryptoError::InvalidSalt(
            "hex string has odd length".to_string(),
        ));
    }

    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|e| CryptoError::InvalidSalt(e.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_salt_generation() {
        let salt1 = Salt::generate().unwrap();
        let salt2 = Salt::generate().unwrap();

        // Salts should be different
        assert_ne!(salt1.as_bytes(), salt2.as_bytes());

        // Salt should be 16 bytes
        assert_eq!(salt1.as_bytes().len(), 16);
    }

    #[test]
    fn test_salt_hex_roundtrip() {
        let salt = Salt::generate().unwrap();
        let hex = salt.as_hex();
        let restored = Salt::from_hex(&hex).unwrap();

        assert_eq!(salt.as_bytes(), restored.as_bytes());
    }

    #[test]
    fn test_key_derivation() {
        let kdf = KeyDerivation::with_params(Argon2Params::for_testing());
        let salt = Salt::generate().unwrap();

        let key1 = kdf.derive_key("test_password_123", &salt).unwrap();
        let key2 = kdf.derive_key("test_password_123", &salt).unwrap();

        // Same password + salt should produce same key
        assert_eq!(key1.as_bytes(), key2.as_bytes());

        // Key should be 32 bytes
        assert_eq!(key1.len(), 32);
    }

    #[test]
    fn test_different_passwords_different_keys() {
        let kdf = KeyDerivation::with_params(Argon2Params::for_testing());
        let salt = Salt::generate().unwrap();

        let key1 = kdf.derive_key("password_one_123", &salt).unwrap();
        let key2 = kdf.derive_key("password_two_456", &salt).unwrap();

        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_different_salts_different_keys() {
        let kdf = KeyDerivation::with_params(Argon2Params::for_testing());
        let salt1 = Salt::generate().unwrap();
        let salt2 = Salt::generate().unwrap();

        let key1 = kdf.derive_key("same_password_here", &salt1).unwrap();
        let key2 = kdf.derive_key("same_password_here", &salt2).unwrap();

        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_password_too_short() {
        let kdf = KeyDerivation::new();
        let salt = Salt::generate().unwrap();

        let result = kdf.derive_key("short", &salt);
        assert!(matches!(result, Err(CryptoError::PasswordTooShort { .. })));
    }

    #[test]
    fn test_derive_key_with_new_salt() {
        let kdf = KeyDerivation::with_params(Argon2Params::for_testing());

        let (key1, salt1) = kdf.derive_key_with_new_salt("my_secure_password").unwrap();
        let (key2, salt2) = kdf.derive_key_with_new_salt("my_secure_password").unwrap();

        // Different salts should be generated
        assert_ne!(salt1.as_bytes(), salt2.as_bytes());

        // Different salts lead to different keys
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }

    #[test]
    fn test_key_hex_format() {
        let kdf = KeyDerivation::with_params(Argon2Params::for_testing());
        let salt = Salt::generate().unwrap();
        let key = kdf.derive_key("password123!", &salt).unwrap();

        let hex = key.as_hex();

        // 32 bytes = 64 hex characters
        assert_eq!(hex.len(), 64);

        // Should only contain valid hex chars
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_key_debug_redacted() {
        let kdf = KeyDerivation::with_params(Argon2Params::for_testing());
        let salt = Salt::generate().unwrap();
        let key = kdf.derive_key("password123!", &salt).unwrap();

        let debug = format!("{:?}", key);

        // Debug output should not contain actual key bytes
        assert!(debug.contains("REDACTED"));
        assert!(debug.contains("32 bytes"));
    }
}
