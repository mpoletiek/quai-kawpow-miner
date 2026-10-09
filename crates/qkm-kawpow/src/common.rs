//! Shared ethash/ProgPoW building blocks (`consensus/{progpow,kawpow}/algorithm.go`).

use sha3::{Digest, Keccak256, Keccak512};

/// Width of mix.
pub const MIX_BYTES: u64 = 128;
/// Hash length in bytes.
pub const HASH_BYTES: u64 = 64;
/// 32-bit words per hash.
pub const HASH_WORDS: usize = 16;
/// Parents per dataset item.
pub const DATASET_PARENTS: u32 = 512;
/// Cache production rounds.
pub const CACHE_ROUNDS: usize = 3;
/// ProgPoW/KawPoW cDag size in 32-bit words (16 KiB).
pub const CACHE_WORDS: usize = 16 * 1024 / 4;

/// Size parameters of an ethash-family algorithm.
#[derive(Clone, Copy, Debug)]
pub struct SizeParams {
    /// Dataset bytes at epoch 0.
    pub dataset_init: u64,
    /// Dataset growth per epoch.
    pub dataset_growth: u64,
    /// Cache bytes at epoch 0.
    pub cache_init: u64,
    /// Cache growth per epoch.
    pub cache_growth: u64,
    /// Blocks per epoch.
    pub epoch_length: u64,
}

/// Deterministic Miller–Rabin for 64-bit `n` (exact, matching Go's
/// `ProbablyPrime(1)`, which is Baillie-PSW and exact below 2^64).
pub fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % p == 0 {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut s = 0;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    let mulmod = |a: u64, b: u64| ((u128::from(a) * u128::from(b)) % u128::from(n)) as u64;
    let powmod = |mut b: u64, mut e: u64| {
        let mut r = 1u64;
        b %= n;
        while e > 0 {
            if e & 1 == 1 {
                r = mulmod(r, b);
            }
            b = mulmod(b, b);
            e >>= 1;
        }
        r
    };
    'outer: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        let mut x = powmod(a, d);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mulmod(x, x);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

impl SizeParams {
    /// `calcCacheSize(epoch)`.
    pub fn cache_size_for_epoch(&self, epoch: u64) -> u64 {
        let mut size = self.cache_init + self.cache_growth * epoch - HASH_BYTES;
        while !is_prime_u64(size / HASH_BYTES) {
            size -= 2 * HASH_BYTES;
        }
        size
    }

    /// `calcDatasetSize(epoch)`.
    pub fn dataset_size_for_epoch(&self, epoch: u64) -> u64 {
        let mut size = self.dataset_init + self.dataset_growth * epoch - MIX_BYTES;
        while !is_prime_u64(size / MIX_BYTES) {
            size -= 2 * MIX_BYTES;
        }
        size
    }

    /// `cacheSize(block)`.
    pub fn cache_size(&self, block: u64) -> u64 {
        self.cache_size_for_epoch(block / self.epoch_length)
    }

    /// `datasetSize(block)`.
    pub fn dataset_size(&self, block: u64) -> u64 {
        self.dataset_size_for_epoch(block / self.epoch_length)
    }

    /// `seedHash(block)`: keccak256 applied `block/epochLength` times to 32 zero bytes.
    pub fn seed_hash(&self, block: u64) -> [u8; 32] {
        let mut seed = [0u8; 32];
        for _ in 0..block / self.epoch_length {
            seed = Keccak256::digest(seed).into();
        }
        seed
    }
}

fn keccak512(data: &[u8]) -> [u8; 64] {
    Keccak512::digest(data).into()
}

/// `generateCache`: sequential keccak512 fill + 3 rounds of RandMemoHash.
/// Returns the cache as little-endian u32 words.
pub fn generate_cache(size: u64, seed: &[u8; 32]) -> Vec<u32> {
    let size = size as usize;
    let hb = HASH_BYTES as usize;
    let rows = size / hb;
    let mut cache = vec![0u8; size];
    cache[..hb].copy_from_slice(&keccak512(seed));
    let mut off = hb;
    while off < size {
        let prev: [u8; 64] = cache[off - hb..off].try_into().unwrap_or([0u8; 64]);
        cache[off..off + hb].copy_from_slice(&keccak512(&prev));
        off += hb;
    }
    let mut temp = [0u8; 64];
    for _ in 0..CACHE_ROUNDS {
        for j in 0..rows {
            let src = ((j + rows - 1) % rows) * hb;
            let dst = j * hb;
            let first =
                u32::from_le_bytes([cache[dst], cache[dst + 1], cache[dst + 2], cache[dst + 3]]);
            let xor = (first % rows as u32) as usize * hb;
            for k in 0..hb {
                temp[k] = cache[src + k] ^ cache[xor + k];
            }
            cache[dst..dst + hb].copy_from_slice(&keccak512(&temp));
        }
    }
    cache
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `fnv(a, b)`.
#[inline]
pub fn fnv(a: u32, b: u32) -> u32 {
    a.wrapping_mul(0x0100_0193) ^ b
}

/// `generateDatasetItem(cache, index)`: 64 bytes.
pub fn dataset_item(cache: &[u32], index: u32) -> [u8; 64] {
    let rows = (cache.len() / HASH_WORDS) as u32;
    let base = ((index % rows) as usize) * HASH_WORDS;
    let mut mix = [0u8; 64];
    for i in 0..HASH_WORDS {
        let mut w = cache[base + i];
        if i == 0 {
            w ^= index;
        }
        mix[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    let mix = keccak512(&mix);
    let mut int_mix = [0u32; HASH_WORDS];
    for (i, w) in int_mix.iter_mut().enumerate() {
        *w = u32::from_le_bytes([mix[i * 4], mix[i * 4 + 1], mix[i * 4 + 2], mix[i * 4 + 3]]);
    }
    for i in 0..DATASET_PARENTS {
        let parent = (fnv(index ^ i, int_mix[(i % 16) as usize]) % rows) as usize * HASH_WORDS;
        for k in 0..HASH_WORDS {
            int_mix[k] = int_mix[k].wrapping_mul(0x0100_0193) ^ cache[parent + k];
        }
    }
    let mut out = [0u8; 64];
    for (i, w) in int_mix.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    keccak512(&out)
}

/// Four dataset items computed together: the parent reads of the four
/// independent chains are interleaved so their cache misses overlap (each
/// chain alone is a sequence of dependent random reads). Item `j` equals
/// `dataset_item(cache, idx[j])`.
pub fn dataset_items4(cache: &[u32], idx: [u32; 4]) -> [[u8; 64]; 4] {
    let rows = (cache.len() / HASH_WORDS) as u32;
    let mut mixes = [[0u32; HASH_WORDS]; 4];
    for (j, m) in mixes.iter_mut().enumerate() {
        let base = ((idx[j] % rows) as usize) * HASH_WORDS;
        let mut bytes = [0u8; 64];
        for i in 0..HASH_WORDS {
            let mut w = cache[base + i];
            if i == 0 {
                w ^= idx[j];
            }
            bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        let h = keccak512(&bytes);
        for (i, w) in m.iter_mut().enumerate() {
            *w = u32::from_le_bytes([h[i * 4], h[i * 4 + 1], h[i * 4 + 2], h[i * 4 + 3]]);
        }
    }
    for i in 0..DATASET_PARENTS {
        let mut parents = [0usize; 4];
        for j in 0..4 {
            parents[j] =
                (fnv(idx[j] ^ i, mixes[j][(i % 16) as usize]) % rows) as usize * HASH_WORDS;
        }
        for j in 0..4 {
            let row = &cache[parents[j]..parents[j] + HASH_WORDS];
            for k in 0..HASH_WORDS {
                mixes[j][k] = mixes[j][k].wrapping_mul(0x0100_0193) ^ row[k];
            }
        }
    }
    let mut out = [[0u8; 64]; 4];
    for (j, m) in mixes.iter().enumerate() {
        let mut bytes = [0u8; 64];
        for (i, w) in m.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        out[j] = keccak512(&bytes);
    }
    out
}

/// The 256-byte DAG entry at `base` (a multiple of 64 words) from the
/// light cache: items `base/16 .. base/16 + 4`.
pub fn light_entry(cache: &[u32], base: u32) -> [u8; 256] {
    let first = base / 16;
    let items = dataset_items4(
        cache,
        [
            first,
            first.wrapping_add(1),
            first.wrapping_add(2),
            first.wrapping_add(3),
        ],
    );
    let mut out = [0u8; 256];
    for (k, it) in items.iter().enumerate() {
        out[k * 64..k * 64 + 64].copy_from_slice(it);
    }
    out
}

/// `generateCDag`: the first 16 KiB of the dataset as u32 words.
pub fn generate_cdag(cache: &[u32]) -> Vec<u32> {
    let mut cdag = vec![0u32; CACHE_WORDS];
    for i in 0..(CACHE_WORDS / 16) as u32 {
        let item = dataset_item(cache, i);
        for j in 0..16 {
            let o = j * 4;
            cdag[i as usize * 16 + j] =
                u32::from_le_bytes([item[o], item[o + 1], item[o + 2], item[o + 3]]);
        }
    }
    cdag
}

/// `keccakfRNDC`.
const RNDC: [u32; 24] = [
    0x0000_0001,
    0x0000_8082,
    0x0000_808a,
    0x8000_8000,
    0x0000_808b,
    0x8000_0001,
    0x8000_8081,
    0x0000_8009,
    0x0000_008a,
    0x0000_0088,
    0x8000_8009,
    0x8000_000a,
    0x8000_808b,
    0x0000_008b,
    0x0000_8089,
    0x0000_8003,
    0x0000_8002,
    0x0000_0080,
    0x0000_800a,
    0x8000_000a,
    0x8000_8081,
    0x0000_8080,
    0x8000_0001,
    0x8000_8008,
];
const ROTC: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];
const PILN: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

/// Go `rotl32` (`n % 32`; `n = 0` shifts by 32, which Go defines as 0).
#[inline]
pub fn rotl32(x: u32, n: u32) -> u32 {
    x.rotate_left(n % 32)
}

/// Go `rotr32`.
#[inline]
pub fn rotr32(x: u32, n: u32) -> u32 {
    x.rotate_right(n % 32)
}

/// One Keccak-f[800] round (`keccakF800Round`).
pub fn keccak_f800_round(st: &mut [u32; 25], r: usize) {
    let mut bc = [0u32; 5];
    for i in 0..5 {
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
    }
    for i in 0..5 {
        let t = bc[(i + 4) % 5] ^ rotl32(bc[(i + 1) % 5], 1);
        for j in (0..25).step_by(5) {
            st[j + i] ^= t;
        }
    }
    let mut t = st[1];
    for i in 0..24 {
        let j = PILN[i];
        bc[0] = st[j];
        st[j] = rotl32(t, ROTC[i]);
        t = bc[0];
    }
    for j in (0..25).step_by(5) {
        let b = [st[j], st[j + 1], st[j + 2], st[j + 3], st[j + 4]];
        st[j] ^= !b[1] & b[2];
        st[j + 1] ^= !b[2] & b[3];
        st[j + 2] ^= !b[3] & b[4];
        st[j + 3] ^= !b[4] & b[0];
        st[j + 4] ^= !b[0] & b[1];
    }
    st[0] ^= RNDC[r];
}

/// `fnv1a`.
#[inline]
pub fn fnv1a(h: &mut u32, d: u32) -> u32 {
    *h = (*h ^ d).wrapping_mul(0x0100_0193);
    *h
}

/// `kiss99State`.
#[derive(Clone, Copy, Default)]
pub struct Kiss99 {
    z: u32,
    w: u32,
    jsr: u32,
    jcong: u32,
}

impl Kiss99 {
    /// `kiss99(st)`.
    #[inline]
    pub fn next(&mut self) -> u32 {
        self.z = 36969u32
            .wrapping_mul(self.z & 65535)
            .wrapping_add(self.z >> 16);
        self.w = 18000u32
            .wrapping_mul(self.w & 65535)
            .wrapping_add(self.w >> 16);
        let mwc = (self.z << 16).wrapping_add(self.w);
        self.jsr ^= self.jsr << 17;
        self.jsr ^= self.jsr >> 13;
        self.jsr ^= self.jsr << 5;
        self.jcong = 69069u32.wrapping_mul(self.jcong).wrapping_add(1_234_567);
        (mwc ^ self.jcong).wrapping_add(self.jsr)
    }
}

/// Registers per lane.
pub const REGS: usize = 32;
/// Lanes.
pub const LANES: usize = 16;
/// DAG loads per lane.
pub const DAG_LOADS: usize = 4;
/// Cache accesses per round.
pub const CNT_CACHE: u32 = 11;
/// Math ops per round.
pub const CNT_MATH: u32 = 18;
/// DAG accesses (loop count).
pub const CNT_DAG: u32 = 64;
/// Mix bytes for the dataset-size divisor.
pub const PROGPOW_MIX_BYTES: u64 = 256;

/// `fillMix(seed, laneId)`.
pub fn fill_mix(seed: u64, lane: u32) -> [u32; REGS] {
    let mut h = 0x811c_9dc5u32;
    let st = Kiss99 {
        z: fnv1a(&mut h, seed as u32),
        w: fnv1a(&mut h, (seed >> 32) as u32),
        jsr: fnv1a(&mut h, lane),
        jcong: fnv1a(&mut h, lane),
    };
    let mut st = st;
    let mut mix = [0u32; REGS];
    for m in &mut mix {
        *m = st.next();
    }
    mix
}

/// `merge(a, b, r)`.
#[inline]
pub fn merge(a: &mut u32, b: u32, r: u32) {
    *a = match r % 4 {
        0 => a.wrapping_mul(33).wrapping_add(b),
        1 => (*a ^ b).wrapping_mul(33),
        2 => rotl32(*a, ((r >> 16) % 31) + 1) ^ b,
        _ => rotr32(*a, ((r >> 16) % 31) + 1) ^ b,
    };
}

/// `progpowInit(seed)`.
pub fn init(seed: u64) -> (Kiss99, [u32; 32], [u32; 32]) {
    let mut h = 0x811c_9dc5u32;
    let mut st = Kiss99 {
        z: fnv1a(&mut h, seed as u32),
        w: fnv1a(&mut h, (seed >> 32) as u32),
        jsr: fnv1a(&mut h, seed as u32),
        jcong: fnv1a(&mut h, (seed >> 32) as u32),
    };
    let mut dst: [u32; 32] = core::array::from_fn(|i| i as u32);
    let mut src: [u32; 32] = core::array::from_fn(|i| i as u32);
    for i in (1..REGS as u32).rev() {
        let j = st.next() % (i + 1);
        dst.swap(i as usize, j as usize);
        let j = st.next() % (i + 1);
        src.swap(i as usize, j as usize);
    }
    (st, dst, src)
}

/// `progpowMath(a, b, r)`.
#[inline]
pub fn math(a: u32, b: u32, r: u32) -> u32 {
    match r % 11 {
        0 => a.wrapping_add(b),
        1 => a.wrapping_mul(b),
        2 => ((u64::from(a) * u64::from(b)) >> 32) as u32,
        3 => a.min(b),
        4 => rotl32(a, b),
        5 => rotr32(a, b),
        6 => a & b,
        7 => a | b,
        8 => a ^ b,
        9 => a.leading_zeros() + b.leading_zeros(),
        _ => a.count_ones() + b.count_ones(),
    }
}

/// `progpowLoop`/`kawpowLoop` (identical bodies).
pub fn prog_loop(
    period: u64,
    loop_: u32,
    mix: &mut [[u32; REGS]; LANES],
    lookup: &mut dyn FnMut(u32) -> [u8; 256],
    cdag: &[u32],
    dataset_size: u32,
) {
    let g_offset = mix[(loop_ % LANES as u32) as usize][0]
        % (64u32.wrapping_mul(dataset_size) / (LANES as u32 * DAG_LOADS as u32));
    let base = g_offset
        .wrapping_mul(LANES as u32)
        .wrapping_mul(DAG_LOADS as u32);
    // The four 64-byte items at base, base+16, base+32, base+48 words.
    let dag_item = lookup(base);
    for l in 0..LANES as u32 {
        let (mut rs, dst_seq, src_seq) = init(period);
        let mut src_c = 0usize;
        let mut dst_c = 0usize;
        let lane = &mut mix[l as usize];
        for i in 0..CNT_MATH {
            if i < CNT_CACHE {
                let src = src_seq[src_c % REGS] as usize;
                src_c += 1;
                let offset = (lane[src] % CACHE_WORDS as u32) as usize;
                let data32 = cdag[offset];
                let dst = dst_seq[dst_c % REGS] as usize;
                dst_c += 1;
                let r = rs.next();
                merge(&mut lane[dst], data32, r);
            }
            let src_rnd = rs.next() % (REGS as u32 * (REGS as u32 - 1));
            let src1 = src_rnd % REGS as u32;
            let mut src2 = src_rnd / REGS as u32;
            if src2 >= src1 {
                src2 += 1;
            }
            let r = rs.next();
            let data32 = math(lane[src1 as usize], lane[src2 as usize], r);
            let dst = dst_seq[dst_c % REGS] as usize;
            dst_c += 1;
            let r = rs.next();
            merge(&mut lane[dst], data32, r);
        }
        let index = (((l ^ loop_) % LANES as u32) * DAG_LOADS as u32) as usize;
        let rd = |k: usize| {
            let o = 4 * (index + k);
            u32::from_le_bytes([
                dag_item[o],
                dag_item[o + 1],
                dag_item[o + 2],
                dag_item[o + 3],
            ])
        };
        let r = rs.next();
        merge(&mut lane[0], rd(0), r);
        for k in 1..DAG_LOADS {
            let dst = dst_seq[dst_c % REGS] as usize;
            dst_c += 1;
            let r = rs.next();
            merge(&mut lane[dst], rd(k), r);
        }
    }
}

/// Runs the 64 DAG loops and reduces to the 8-word result.
pub fn main_loop(
    seed: u64,
    block_number: u64,
    period_length: u64,
    size: u64,
    cdag: &[u32],
    lookup: &mut dyn FnMut(u32) -> [u8; 256],
) -> [u32; 8] {
    let mut mix = [[0u32; REGS]; LANES];
    for (lane, m) in mix.iter_mut().enumerate() {
        *m = fill_mix(seed, lane as u32);
    }
    let period = block_number / period_length;
    let ds = (size / PROGPOW_MIX_BYTES) as u32;
    for l in 0..CNT_DAG {
        prog_loop(period, l, &mut mix, lookup, cdag, ds);
    }
    let mut lane_results = [0u32; LANES];
    for (lane, lr) in lane_results.iter_mut().enumerate() {
        *lr = 0x811c_9dc5;
        for i in 0..REGS {
            fnv1a(lr, mix[lane][i]);
        }
    }
    let mut result = [0x811c_9dc5u32; 8];
    for (lane, lr) in lane_results.iter().enumerate() {
        fnv1a(&mut result[lane % 8], *lr);
    }
    result
}

/// The full dataset (`generateDataset`): `size / 64` items of 64 bytes,
/// generated on `threads` threads. Item `i` equals `dataset_item(cache, i)`.
pub fn generate_dataset(cache: &[u32], size: u64, threads: usize) -> Vec<u8> {
    let items = (size / 64) as usize;
    let mut data = vec![0u8; items * 64];
    let threads = threads.max(1);
    let per = items.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, chunk) in data.chunks_mut(per * 64).enumerate() {
            s.spawn(move || {
                let first = t * per;
                let mut quads = chunk.chunks_exact_mut(256);
                let mut j = 0usize;
                for quad in quads.by_ref() {
                    let i = (first + j) as u32;
                    let items = dataset_items4(cache, [i, i + 1, i + 2, i + 3]);
                    for (k, it) in items.iter().enumerate() {
                        quad[k * 64..k * 64 + 64].copy_from_slice(it);
                    }
                    j += 4;
                }
                for item in quads.into_remainder().chunks_mut(64) {
                    item.copy_from_slice(&dataset_item(cache, (first + j) as u32));
                    j += 1;
                }
            });
        }
    });
    data
}

#[cfg(test)]
mod items_tests {
    use super::*;

    #[test]
    fn interleaved_items_match_single_items() {
        let cache: Vec<u32> = (0..16 * 4099u32)
            .map(|i| i.wrapping_mul(0x9e37_79b9).rotate_left(i % 31))
            .collect();
        for first in [0u32, 7, 4095, 123_456, u32::MAX - 3] {
            let idx = [
                first,
                first.wrapping_add(1),
                first.wrapping_add(2),
                first.wrapping_add(3),
            ];
            let four = dataset_items4(&cache, idx);
            for j in 0..4 {
                assert_eq!(four[j], dataset_item(&cache, idx[j]));
            }
        }
        // Thread chunks that are not multiples of four items.
        let ds = generate_dataset(&cache, 64 * 37, 3);
        for (i, item) in ds.chunks(64).enumerate() {
            assert_eq!(item, dataset_item(&cache, i as u32));
        }
    }
}
