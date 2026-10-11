//! KawPoW, the reference this miner checks every GPU solution against and
//! generates its kernels from: ProgPoW with Ravencoin's Keccak-f800
//! padding, the ethash light cache and cDag, and light (CPU) evaluation.
//!
//! Ported from rs-quai's `rsq-pow`
//! (`common.rs`, `epoch_cache.rs`, `kawpow.rs`, at dbd388d), itself a port
//! of go-quai v0.56.0's `consensus/kawpow` and `consensus/progpow`, and
//! verified there against go-quai across the mainnet chain. Only the
//! KawPoW part is here (no Quai ProgPoW), with one dependency (`sha3`), so
//! the miner builds on its own. The Ravencoin block vectors of go-quai's
//! tests come with it (`tests/`).
//!
//! GPL-3.0-only, like rs-quai.

// The port is kept as it is in rsq-pow, so the two can be diffed line for
// line; these are style lints the rs-quai workspace allows.
#![allow(
    clippy::manual_is_multiple_of,
    clippy::should_implement_trait,
    clippy::needless_range_loop
)]

pub mod common;
mod epoch_cache;
pub mod kawpow;

pub use epoch_cache::{EpochCache, EpochCaches};

/// PoW verification errors (`consensus.ErrInvalidMixHash`, header guards).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowError {
    /// The computed mix hash differs from the header's.
    InvalidMixHash,
    /// A header guard (epoch/cache bounds) failed.
    Invalid(String),
}

impl core::fmt::Display for PowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PowError::InvalidMixHash => f.write_str("invalid mixHash"),
            PowError::Invalid(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for PowError {}
