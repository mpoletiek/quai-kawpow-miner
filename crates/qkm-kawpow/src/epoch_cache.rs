//! Verification caches per (algorithm, epoch), generated once and shared.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::common::{SizeParams, generate_cache, generate_cdag};

/// Light-verification data for one epoch.
#[derive(Debug)]
pub struct EpochCache {
    /// Epoch number.
    pub epoch: u64,
    /// Verification cache (u32 words).
    pub cache: Vec<u32>,
    /// 16 KiB cDag.
    pub cdag: Vec<u32>,
}

/// A small keep-recent map of epoch caches; generation happens once per
/// epoch even under concurrent requests.
pub struct EpochCaches {
    params: SizeParams,
    max: usize,
    inner: Mutex<HashMap<u64, Arc<OnceLock<Arc<EpochCache>>>>>,
    order: Mutex<Vec<u64>>,
}

impl EpochCaches {
    /// New cache set keeping at most `max` epochs.
    pub fn new(params: SizeParams, max: usize) -> Self {
        EpochCaches {
            params,
            max,
            inner: Mutex::new(HashMap::new()),
            order: Mutex::new(Vec::new()),
        }
    }

    /// Size parameters.
    pub fn params(&self) -> &SizeParams {
        &self.params
    }

    /// Returns (generating if needed) the cache for `block`'s epoch.
    pub fn get(&self, block: u64) -> Arc<EpochCache> {
        let epoch = block / self.params.epoch_length;
        let cell = {
            let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let cell = map
                .entry(epoch)
                .or_insert_with(|| Arc::new(OnceLock::new()))
                .clone();
            let mut order = self.order.lock().unwrap_or_else(|e| e.into_inner());
            order.retain(|e| *e != epoch);
            order.push(epoch);
            while order.len() > self.max {
                let old = order.remove(0);
                map.remove(&old);
            }
            cell
        };
        cell.get_or_init(|| {
            // go-quai: size/seed for `epoch*epochLength + 1`.
            let b = epoch * self.params.epoch_length + 1;
            let size = self.params.cache_size(b);
            let seed = self.params.seed_hash(b);
            let cache = generate_cache(size, &seed);
            let cdag = generate_cdag(&cache);
            Arc::new(EpochCache { epoch, cache, cdag })
        })
        .clone()
    }
}
