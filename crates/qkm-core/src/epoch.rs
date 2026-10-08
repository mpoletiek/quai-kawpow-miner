//! KawPoW epochs and CPU verification (rs-quai's `rsq-pow`).
//!
//! Heights are Ravencoin donor heights: 7,500 blocks per epoch, a new
//! ProgPoW program every 3 blocks.

use std::sync::Arc;

use rsq_pow::common::PROGPOW_MIX_BYTES;
use rsq_pow::kawpow::{PARAMS, PERIOD_LENGTH, kawpow_light};
use rsq_pow::{EpochCache, EpochCaches};

/// Blocks per epoch.
pub const EPOCH_LENGTH: u64 = 7500;

/// The last epoch this miner accepts work for: its DAG is about 33 GiB,
/// decades away (epoch 609 in October 2026). A height past it comes from a
/// broken or hostile pool, and computing its seed or light cache would take
/// minutes to years.
pub const MAX_EPOCH: u64 = 4096;

/// Whether a donor height is one this miner can mine.
pub fn plausible_height(height: u64) -> bool {
    height / EPOCH_LENGTH <= MAX_EPOCH
}

/// The ProgPoW period of a height.
pub fn period(height: u64) -> u64 {
    height / PERIOD_LENGTH
}

/// Sizes of one epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Epoch {
    /// Epoch number (`height / 7500`).
    pub number: u64,
    /// Full dataset (DAG) bytes.
    pub dataset_bytes: u64,
    /// Light cache bytes.
    pub cache_bytes: u64,
}

impl Epoch {
    /// The epoch of a donor height.
    pub fn for_height(height: u64) -> Epoch {
        Epoch {
            number: height / EPOCH_LENGTH,
            dataset_bytes: PARAMS.dataset_size(height),
            cache_bytes: PARAMS.cache_size(height),
        }
    }

    /// A height inside this epoch (its first block).
    pub fn first_height(&self) -> u64 {
        self.number * EPOCH_LENGTH
    }

    /// 64-byte dataset items in the DAG.
    pub fn dag_items(&self) -> u64 {
        self.dataset_bytes / 64
    }

    /// The modulus of the per-loop DAG offset, in 256-byte entries
    /// (`64 * (size / 256) / (LANES * DAG_LOADS)` in u32 arithmetic, as in
    /// `rsq_pow::common::prog_loop`).
    pub fn dag_elements(&self) -> u32 {
        let ds = (self.dataset_bytes / PROGPOW_MIX_BYTES) as u32;
        64u32.wrapping_mul(ds) / 64
    }
}

/// Light caches (cache + cDag) for CPU verification, a few epochs kept.
pub struct Verifier {
    caches: EpochCaches,
}

impl Default for Verifier {
    fn default() -> Self {
        Verifier {
            caches: EpochCaches::new(PARAMS, 3),
        }
    }
}

/// A KawPoW evaluation: the mix digest and the final hash, both as the
/// little-endian words `rsq-pow` produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hashes {
    /// The 256-bit mix digest.
    pub mix: [u8; 32],
    /// The final hash; read as a big-endian number it is compared with the
    /// target.
    pub hash: [u8; 32],
}

impl Verifier {
    /// The light cache of the epoch containing `height` (built on first use).
    pub fn epoch_cache(&self, height: u64) -> Arc<EpochCache> {
        self.caches.get(height)
    }

    /// KawPoW of `header` (the header hash words as the GPU kernel takes
    /// them: the stratum job's header bytes, little-endian words) and
    /// `nonce` at `height`.
    pub fn hash(&self, height: u64, header: &[u8; 32], nonce: u64) -> Hashes {
        let ec = self.caches.get(height);
        let (mix, hash) = kawpow_light(&ec, PARAMS.dataset_size(height), header, nonce, height);
        Hashes { mix, hash }
    }
}

/// The 64-bit boundary the GPU compares with: the top 64 bits of a 256-bit
/// big-endian target. A hash whose top 64 bits are `<=` this may meet the
/// target; the CPU check decides.
pub fn boundary64(target_be: &[u8; 32]) -> u64 {
    u64::from_be_bytes([
        target_be[0],
        target_be[1],
        target_be[2],
        target_be[3],
        target_be[4],
        target_be[5],
        target_be[6],
        target_be[7],
    ])
}

/// Whether a final hash (big-endian number) is at most `target_be`.
pub fn meets(hash: &[u8; 32], target_be: &[u8; 32]) -> bool {
    hash <= target_be
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_608_sizes() {
        let e = Epoch::for_height(4_565_235);
        assert_eq!(e.number, 608);
        // 2^30 + 608 * 2^23 rounded down to a prime number of 128-byte rows.
        assert!(e.dataset_bytes <= (1 << 30) + 608 * (1 << 23));
        assert!(e.dataset_bytes > (1 << 30) + 607 * (1 << 23));
        assert_eq!(u64::from(e.dag_elements()), e.dataset_bytes / 256);
    }

    #[test]
    fn boundary_is_top_64_bits() {
        let mut t = [0u8; 32];
        t[4] = 0xff;
        assert_eq!(boundary64(&t), 0x0000_0000_ff00_0000);
        assert!(meets(&[0u8; 32], &t));
        assert!(!meets(&[0xff; 32], &t));
    }
}
