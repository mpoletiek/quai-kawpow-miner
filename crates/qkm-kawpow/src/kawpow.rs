//! KawPoW (`consensus/kawpow`), evaluated on the Ravencoin donor header in
//! AuxPow: ProgPoW with period 3 and Ravencoin's Keccak-f800 initial/final
//! state (the "RAVENCOINKAWPOW" padding words).

use crate::PowError;
use crate::common::{SizeParams, keccak_f800_round, main_loop};
use crate::epoch_cache::{EpochCache, EpochCaches};

/// KawPoW period length.
pub const PERIOD_LENGTH: u64 = 3;

/// ethash sizes with a 7,500-block epoch.
pub const PARAMS: SizeParams = SizeParams {
    dataset_init: 1 << 30,
    dataset_growth: 1 << 23,
    cache_init: 1 << 24,
    cache_growth: 1 << 17,
    epoch_length: 7500,
};

/// `maxKawpowCacheBytes`.
pub const MAX_CACHE_BYTES: u64 = 256 * 1024 * 1024;
/// `maxKawpowDonorHeight`.
pub const MAX_DONOR_HEIGHT: u64 = 14_000_000;

/// `ravencoinKawpow`: "RAVENCOINKAWPOW" as u32 words.
pub const RAVENCOIN_KAWPOW: [u32; 15] = [
    0x72, 0x41, 0x56, 0x45, 0x4E, 0x43, 0x4F, 0x49, 0x4E, 0x4B, 0x41, 0x57, 0x50, 0x4F, 0x57,
];

fn keccak_f800(st: &mut [u32; 25]) {
    for r in 0..22 {
        keccak_f800_round(st, r);
    }
}

/// `kawpow(hash, nonce, size, blockNumber, cDag, lookup)` → `(mixHash, powHash)`.
/// `hash` is the little-endian (byte-reversed) KAWPOW header hash.
pub fn kawpow(
    hash: &[u8; 32],
    nonce: u64,
    size: u64,
    block_number: u64,
    cdag: &[u32],
    lookup: &mut dyn FnMut(u32) -> [u8; 256],
) -> ([u8; 32], [u8; 32]) {
    let mut temp = [0u32; 25];
    for i in 0..8 {
        temp[i] = u32::from_le_bytes([
            hash[i * 4],
            hash[i * 4 + 1],
            hash[i * 4 + 2],
            hash[i * 4 + 3],
        ]);
    }
    temp[8] = nonce as u32;
    temp[9] = (nonce >> 32) as u32;
    temp[10..25].copy_from_slice(&RAVENCOIN_KAWPOW);
    keccak_f800(&mut temp);
    let seed_head = u64::from(temp[0]) + (u64::from(temp[1]) << 32);
    let result = main_loop(seed_head, block_number, PERIOD_LENGTH, size, cdag, lookup);
    let mut mix = [0u8; 32];
    for i in 0..8 {
        mix[i * 4..i * 4 + 4].copy_from_slice(&result[i].to_le_bytes());
    }
    let mut state = [0u32; 25];
    state[..8].copy_from_slice(&temp[..8]);
    state[8..16].copy_from_slice(&result);
    state[16..25].copy_from_slice(&RAVENCOIN_KAWPOW[..9]);
    keccak_f800(&mut state);
    let mut digest = [0u8; 32];
    for i in 0..8 {
        digest[i * 4..i * 4 + 4].copy_from_slice(&state[i].to_le_bytes());
    }
    (mix, digest)
}

/// `kawpowLight` using an epoch cache.
pub fn kawpow_light(
    ec: &EpochCache,
    size: u64,
    hash: &[u8; 32],
    nonce: u64,
    block_number: u64,
) -> ([u8; 32], [u8; 32]) {
    let mut lookup = |base: u32| crate::common::light_entry(&ec.cache, base);
    kawpow(hash, nonce, size, block_number, &ec.cdag, &mut lookup)
}

/// `validateKawpowBlockNumber` (donor height bounds).
pub fn validate_block_number(height: u64) -> Result<(), PowError> {
    if height > MAX_DONOR_HEIGHT {
        return Err(PowError::Invalid(format!(
            "KAWPOW donor height {height} exceeds maximum {MAX_DONOR_HEIGHT}"
        )));
    }
    let cache = PARAMS.cache_size(height);
    if cache > MAX_CACHE_BYTES {
        return Err(PowError::Invalid(format!(
            "KAWPOW donor height {height} requires cache size {cache}"
        )));
    }
    Ok(())
}

/// KawPoW engine with its epoch caches.
pub struct Kawpow {
    caches: EpochCaches,
}

impl Default for Kawpow {
    fn default() -> Self {
        Kawpow {
            caches: EpochCaches::new(PARAMS, 3),
        }
    }
}

impl Kawpow {
    /// `ComputePowLight` for a Ravencoin donor header: returns
    /// `(mixHash (as stored in the header: byte-reversed), powHash)`.
    ///
    /// `kawpow_header_hash` is `RavencoinHeader::kawpow_header_hash()`
    /// (natural sha256d order); it is reversed before hashing, and the
    /// resulting mix is reversed back, exactly as go-quai does.
    pub fn compute_light(
        &self,
        kawpow_header_hash: &[u8; 32],
        nonce64: u64,
        height: u64,
    ) -> Result<([u8; 32], [u8; 32]), PowError> {
        validate_block_number(height)?;
        let ec = self.caches.get(height);
        let size = PARAMS.dataset_size(height);
        let mut h = *kawpow_header_hash;
        h.reverse();
        let (mut mix, pow) = kawpow_light(&ec, size, &h, nonce64, height);
        mix.reverse();
        Ok((mix, pow))
    }
}
