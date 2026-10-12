//! CUDA C sources: the per-period KawPoW search kernel (generated from a
//! [`Program`]) and the DAG generation kernel.
//!
//! The search kernel follows kawpowminer's structure (GPL-3.0): a group of
//! 16 threads (the ProgPoW lanes) computes the hashes of its 16 nonces one
//! after another, each thread holding one lane's 32 registers, with the
//! first 16 KiB of the DAG in shared memory.
//!
//! Where it stands (RTX 3060 Ti GDDR6X, epoch 610, 2026-10-11): memory runs
//! at 9251 MHz under CUDA (P2), about 592 GB/s, and every hash reads 16 KiB
//! of DAG. At 225 W the kernel moves 93-94% of that (34 MH/s); at 200 W,
//! 90% (32 MH/s, 161 kH/W, the most efficient point), held there by the
//! power cap. Nsight Compute: 60% of stalls wait on the DAG load; 78-80
//! registers in most periods, 3 blocks (24 warps) per SM, which shared
//! memory (16 KiB a block) limits as much as registers. T-Rex 0.26.8's
//! benchmark on the same card ties at 200 W and reports 1.4% more at 225 W
//! (its own figure: it refuses the profiler). Measured and no faster, each
//! within noise or 0.5-6% slower, at 200 W and 225 W:
//! - without the fence (64 registers, 5 fewer instructions an iteration),
//!   and unrolled 2 to 16 (down to 126 instructions an iteration): the DAG
//!   load then issues 16-65 instructions before its use instead of 121;
//! - more warps: a 64-register cap, blocks of 512 or 1024;
//! - each hash's first DAG load issued before the previous digest and
//!   `fill_mix` (exact, 72 registers, 1% slower);
//! - grids of 228 to 3648 blocks, 2 or 3 streams; `ldg`, `cg`, `cs` and
//!   plain loads;
//! - memory clocks locked at 9501 MHz (P2 keeps 9251; 6% slower at 225 W)
//!   or +500 MHz offset (9501 MHz, no faster: the memory takes the power).

use std::fmt::Write;

use crate::program::{Op, Program};
use qkm_kawpow::kawpow::RAVENCOIN_KAWPOW;

/// Most solutions one launch reports (`out[0]` is the count, then `gid`
/// and 8 mix words per solution).
pub const MAX_RESULTS: u32 = 4;
/// Words in the search results buffer.
pub const RESULT_WORDS: usize = 1 + MAX_RESULTS as usize * 9;

/// Kernel names.
pub const SEARCH_FN: &str = "kawpow_search";
/// Debug kernel: writes every nonce's mix digest and final hash.
pub const HASH_FN: &str = "kawpow_hash";
/// DAG generation kernels, run in this order over each chunk of items.
pub const DAG_FNS: [&str; 3] = ["dag_init", "dag_mix", "dag_final"];

fn merge(out: &mut String, a: &str, b: &str, r: u32) {
    let k = ((r >> 16) % 31) + 1;
    let _ = match r % 4 {
        0 => writeln!(out, "{a} = ({a} * 33) + {b};"),
        1 => writeln!(out, "{a} = ({a} ^ {b}) * 33;"),
        2 => writeln!(out, "{a} = ROTL32({a}, {k}) ^ {b};"),
        _ => writeln!(out, "{a} = ROTR32({a}, {k}) ^ {b};"),
    };
}

fn math(out: &mut String, d: &str, a: &str, b: &str, r: u32) {
    let _ = match r % 11 {
        0 => writeln!(out, "{d} = {a} + {b};"),
        1 => writeln!(out, "{d} = {a} * {b};"),
        2 => writeln!(out, "{d} = __umulhi({a}, {b});"),
        3 => writeln!(out, "{d} = ({a} < {b}) ? {a} : {b};"),
        4 => writeln!(out, "{d} = ROTL32({a}, {b});"),
        5 => writeln!(out, "{d} = ROTR32({a}, {b});"),
        6 => writeln!(out, "{d} = {a} & {b};"),
        7 => writeln!(out, "{d} = {a} | {b};"),
        8 => writeln!(out, "{d} = {a} ^ {b};"),
        9 => writeln!(out, "{d} = __clz({a}) + __clz({b});"),
        _ => writeln!(out, "{d} = __popc({a}) + __popc({b});"),
    };
}

/// The `progPowLoop` device function for one program.
pub fn prog_loop_source(p: &Program, fence: bool) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "// Inner loop for period {}", p.period);
    s.push_str(
        "__device__ __forceinline__ void progPowLoop(const uint32_t loop, uint32_t mix[PROGPOW_REGS],\n\
         \x20   const dag_t* __restrict__ g_dag, const uint32_t c_dag[PROGPOW_CACHE_WORDS], const bool hack_false)\n{\n\
         uint4 data_dag;\nuint32_t offset, data;\n\
         const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);\n\
         offset = SHFL(mix[0], loop % PROGPOW_LANES, PROGPOW_LANES);\n\
         offset %= PROGPOW_DAG_ELEMENTS;\n\
         offset = offset * PROGPOW_LANES + (lane_id ^ loop) % PROGPOW_LANES;\n\
         data_dag = DAG_LOAD(((const uint4*)g_dag) + offset);\n",
    );
    // A never-taken fence keeps the compiler from moving the DAG load's use
    // up (the load should overlap the whole loop).
    let fence_line = if fence {
        "if (hack_false) __threadfence_block();\n"
    } else {
        ""
    };
    s.push_str(fence_line);
    for op in &p.ops {
        match *op {
            Op::Cache { src, dst, merge: r } => {
                let _ = writeln!(s, "offset = mix[{src}] % PROGPOW_CACHE_WORDS;");
                s.push_str("data = c_dag[offset];\n");
                merge(&mut s, &format!("mix[{dst}]"), "data", r);
            }
            Op::Math {
                src1,
                src2,
                math: m,
                dst,
                merge: r,
            } => {
                math(
                    &mut s,
                    "data",
                    &format!("mix[{src1}]"),
                    &format!("mix[{src2}]"),
                    m,
                );
                merge(&mut s, &format!("mix[{dst}]"), "data", r);
            }
        }
    }
    s.push_str(fence_line);
    for (k, &(dst, r)) in p.dag.iter().enumerate() {
        let lane = ["x", "y", "z", "w"][k];
        merge(
            &mut s,
            &format!("mix[{dst}]"),
            &format!("data_dag.{lane}"),
            r,
        );
    }
    s.push_str("}\n");
    s
}

const PRELUDE: &str = r#"typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#define ROTL32(x, n) __funnelshift_l((x), (x), (n))
#define ROTR32(x, n) __funnelshift_r((x), (x), (n))
#define SHFL(x, y, z) __shfl_sync(0xFFFFFFFFu, (x), (y), (z))
#define PROGPOW_LANES 16
#define PROGPOW_REGS 32
#define PROGPOW_DAG_LOADS 4
#define PROGPOW_CACHE_WORDS 4096
#define PROGPOW_CNT_DAG 64
#define FNV_PRIME 0x01000193u
#define FNV_OFFSET 0x811c9dc5u
#define fnv1a(h, d) (h = ((h) ^ (d)) * FNV_PRIME)
typedef struct __align__(16) { uint32_t s[PROGPOW_DAG_LOADS]; } dag_t;
typedef struct { uint32_t w[8]; } header_t;
"#;

const SEARCH_BODY: &str = r#"
__constant__ const uint32_t keccakf_rndc[24] = {
    0x00000001, 0x00008082, 0x0000808a, 0x80008000, 0x0000808b, 0x80000001,
    0x80008081, 0x00008009, 0x0000008a, 0x00000088, 0x80008009, 0x8000000a,
    0x8000808b, 0x0000008b, 0x00008089, 0x00008003, 0x00008002, 0x00000080,
    0x0000800a, 0x8000000a, 0x80008081, 0x00008080, 0x80000001, 0x80008008};

__device__ __forceinline__ void keccak_f800_round(uint32_t st[25], const int r)
{
    const uint32_t rotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
                               27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
    const uint32_t piln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
                               15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
    uint32_t t, bc[5];
    for (int i = 0; i < 5; i++)
        bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
    for (int i = 0; i < 5; i++) {
        t = bc[(i + 4) % 5] ^ ROTL32(bc[(i + 1) % 5], 1);
        for (int j = 0; j < 25; j += 5)
            st[j + i] ^= t;
    }
    t = st[1];
    for (int i = 0; i < 24; i++) {
        const uint32_t j = piln[i];
        bc[0] = st[j];
        st[j] = ROTL32(t, rotc[i]);
        t = bc[0];
    }
    for (int j = 0; j < 25; j += 5) {
        for (int i = 0; i < 5; i++)
            bc[i] = st[j + i];
        for (int i = 0; i < 5; i++)
            st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }
    st[0] ^= keccakf_rndc[r];
}

__device__ __forceinline__ void keccak_f800(uint32_t st[25])
{
    for (int r = 0; r < 22; r++)
        keccak_f800_round(st, r);
}

typedef struct { uint32_t z, w, jsr, jcong; } kiss99_t;

__device__ __forceinline__ uint32_t kiss99(kiss99_t &st)
{
    st.z = 36969 * (st.z & 65535) + (st.z >> 16);
    st.w = 18000 * (st.w & 65535) + (st.w >> 16);
    const uint32_t mwc = (st.z << 16) + st.w;
    st.jsr ^= (st.jsr << 17);
    st.jsr ^= (st.jsr >> 13);
    st.jsr ^= (st.jsr << 5);
    st.jcong = 69069 * st.jcong + 1234567;
    return (mwc ^ st.jcong) + st.jsr;
}

__device__ __forceinline__ void fill_mix(const uint32_t seed0, const uint32_t seed1, const uint32_t lane_id,
                                         uint32_t mix[PROGPOW_REGS])
{
    uint32_t h = FNV_OFFSET;
    kiss99_t st;
    st.z = fnv1a(h, seed0);
    st.w = fnv1a(h, seed1);
    st.jsr = fnv1a(h, lane_id);
    st.jcong = fnv1a(h, lane_id);
#pragma unroll
    for (int i = 0; i < PROGPOW_REGS; i++)
        mix[i] = kiss99(st);
}

__device__ __forceinline__ uint32_t swab32(const uint32_t x) { return __byte_perm(x, x, 0x0123); }

// This thread's nonce: its mix digest and the final Keccak state words 0..7.
__device__ __forceinline__ void kawpow_hash_one(const uint64_t nonce, const header_t &header,
                                                const dag_t* __restrict__ g_dag, const uint32_t* c_dag,
                                                const bool hack_false, uint32_t digest[8], uint32_t fin[8])
{
    const uint32_t lane_id = threadIdx.x & (PROGPOW_LANES - 1);
    uint32_t seed[8];
    {
        uint32_t st[25];
        for (int i = 0; i < 8; i++)
            st[i] = header.w[i];
        st[8] = (uint32_t)nonce;
        st[9] = (uint32_t)(nonce >> 32);
        for (int i = 10; i < 25; i++)
            st[i] = RAVENCOIN_KAWPOW_(i - 10);
        keccak_f800(st);
        for (int i = 0; i < 8; i++)
            seed[i] = st[i];
    }
#pragma unroll 1
    for (uint32_t h = 0; h < PROGPOW_LANES; h++) {
        uint32_t mix[PROGPOW_REGS];
        const uint32_t s0 = SHFL(seed[0], h, PROGPOW_LANES);
        const uint32_t s1 = SHFL(seed[1], h, PROGPOW_LANES);
        fill_mix(s0, s1, lane_id, mix);
#pragma unroll QKM_UNROLL
        for (uint32_t l = 0; l < PROGPOW_CNT_DAG; l++)
            progPowLoop(l, mix, g_dag, c_dag, hack_false);
        uint32_t lane_digest = FNV_OFFSET;
#pragma unroll
        for (int i = 0; i < PROGPOW_REGS; i++)
            fnv1a(lane_digest, mix[i]);
#if QKM_LEAN_DIGEST
        // digest[j] = fnv(fnv(OFFSET, lane j's), lane j + 8's): lane j < 8
        // folds its own with lane j + 8's (one shuffle), then lane h
        // gathers the 8 (eight shuffles) instead of every lane computing
        // all 8 from 16 shuffles.
        uint32_t dj = FNV_OFFSET;
        fnv1a(dj, lane_digest);
        fnv1a(dj, SHFL(lane_digest, lane_id + 8, PROGPOW_LANES));
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t v = SHFL(dj, i, PROGPOW_LANES);
            if (h == lane_id)
                digest[i] = v;
        }
#else
        uint32_t d[8];
#pragma unroll
        for (int i = 0; i < 8; i++)
            d[i] = FNV_OFFSET;
        for (int i = 0; i < PROGPOW_LANES; i += 8)
#pragma unroll
            for (int j = 0; j < 8; j++)
                fnv1a(d[j], SHFL(lane_digest, i + j, PROGPOW_LANES));
        if (h == lane_id)
            for (int i = 0; i < 8; i++)
                digest[i] = d[i];
#endif
    }
    {
        uint32_t st[25];
        for (int i = 0; i < 8; i++)
            st[i] = seed[i];
        for (int i = 8; i < 16; i++)
            st[i] = digest[i - 8];
        for (int i = 16; i < 25; i++)
            st[i] = RAVENCOIN_KAWPOW_(i - 16);
        keccak_f800(st);
        for (int i = 0; i < 8; i++)
            fin[i] = st[i];
    }
}

__device__ __forceinline__ void load_c_dag(uint32_t* c_dag, const dag_t* __restrict__ g_dag)
{
    for (uint32_t word = threadIdx.x * PROGPOW_DAG_LOADS; word < PROGPOW_CACHE_WORDS;
         word += blockDim.x * PROGPOW_DAG_LOADS) {
        const dag_t load = g_dag[word / PROGPOW_DAG_LOADS];
        for (int i = 0; i < PROGPOW_DAG_LOADS; i++)
            c_dag[word + i] = load.s[i];
    }
    __syncthreads();
}

// Solutions whose final hash's top 64 bits are <= target: out[0] counts
// them, then per solution the gid and the 8 mix words.
extern "C" __global__ void __launch_bounds__(QKM_BLOCK QKM_MIN_BLOCKS) kawpow_search(
    const uint64_t start_nonce, const header_t header, const uint64_t target,
    const dag_t* __restrict__ g_dag, uint32_t* g_out, const uint32_t hack_false)
{
    __shared__ uint32_t c_dag[PROGPOW_CACHE_WORDS];
    load_c_dag(c_dag, g_dag);
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t digest[8], fin[8];
    kawpow_hash_one(start_nonce + gid, header, g_dag, c_dag, hack_false != 0, digest, fin);
    const uint64_t result = ((uint64_t)swab32(fin[0]) << 32) | swab32(fin[1]);
    if (result > target)
        return;
    const uint32_t index = atomicInc(&g_out[0], 0xffffffffu);
    if (index >= QKM_MAX_RESULTS)
        return;
    g_out[1 + index * 9] = gid;
    for (int i = 0; i < 8; i++)
        g_out[2 + index * 9 + i] = digest[i];
}

// Every nonce's mix digest and final hash words (16 words per nonce).
extern "C" __global__ void __launch_bounds__(QKM_BLOCK QKM_MIN_BLOCKS) kawpow_hash(
    const uint64_t start_nonce, const header_t header, const dag_t* __restrict__ g_dag, uint32_t* g_out)
{
    __shared__ uint32_t c_dag[PROGPOW_CACHE_WORDS];
    load_c_dag(c_dag, g_dag);
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t digest[8], fin[8];
    kawpow_hash_one(start_nonce + gid, header, g_dag, c_dag, false, digest, fin);
    for (int i = 0; i < 8; i++) {
        g_out[gid * 16 + i] = digest[i];
        g_out[gid * 16 + 8 + i] = fin[i];
    }
}
"#;

/// How the kernel reads the DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DagLoad {
    /// Read-only data cache (`ld.global.nc`, what `const __restrict__` gives).
    Ldg,
    /// Cache in L2 only (`ld.global.cg`).
    Cg,
    /// Streaming, evict first (`ld.global.cs`).
    Cs,
    /// Plain load (`ld.global`).
    Plain,
}

impl DagLoad {
    /// The `DAG_LOAD(p)` expression for a `const uint4*`.
    fn expr(self) -> &'static str {
        match self {
            DagLoad::Ldg => "__ldg(p)",
            DagLoad::Cg => "__ldcg(p)",
            DagLoad::Cs => "__ldcs(p)",
            DagLoad::Plain => "(*(p))",
        }
    }

    /// Parses `ldg`, `cg`, `cs` or `plain`.
    pub fn parse(s: &str) -> Option<DagLoad> {
        Some(match s {
            "ldg" => DagLoad::Ldg,
            "cg" => DagLoad::Cg,
            "cs" => DagLoad::Cs,
            "plain" => DagLoad::Plain,
            _ => return None,
        })
    }
}

/// Tunables baked into a search module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SearchParams {
    /// Threads per block (a multiple of 16; the launch must use it).
    pub block: u32,
    /// `__launch_bounds__` minimum resident blocks per SM (0: let ptxas
    /// choose); more blocks means fewer registers per thread.
    pub min_blocks: u32,
    /// How the DAG is read.
    pub dag_load: DagLoad,
    /// Keep the compiler fence around the DAG load (kawpowminer's
    /// `hack_false`).
    pub fence: bool,
    /// Unroll factor of the 64-iteration DAG loop.
    pub unroll: u32,
    /// Reduce the lane digests with 9 shuffles instead of 16 (kawpowminer's
    /// way when off). On: +0.7% kH/W on an RTX 3060 Ti at 225 W (6
    /// interleaved pairs).
    pub lean_digest: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            block: 256,
            min_blocks: 0,
            dag_load: DagLoad::Ldg,
            fence: true,
            unroll: 1,
            lean_digest: true,
        }
    }
}

/// The whole search module for one period: prelude, `progPowLoop`, and the
/// `kawpow_search` / `kawpow_hash` kernels.
pub fn search_source(p: &Program, dag_elements: u32, params: SearchParams) -> String {
    let mut s = String::from(PRELUDE);
    let _ = writeln!(s, "#define PROGPOW_DAG_ELEMENTS {dag_elements}u");
    let _ = writeln!(s, "#define QKM_BLOCK {}", params.block);
    if params.min_blocks > 0 {
        let _ = writeln!(s, "#define QKM_MIN_BLOCKS , {}", params.min_blocks);
    } else {
        s.push_str("#define QKM_MIN_BLOCKS\n");
    }
    let _ = writeln!(s, "#define DAG_LOAD(p) {}", params.dag_load.expr());
    let _ = writeln!(s, "#define QKM_MAX_RESULTS {MAX_RESULTS}u");
    let _ = writeln!(
        s,
        "#define QKM_LEAN_DIGEST {}",
        u8::from(params.lean_digest)
    );
    let words: Vec<String> = RAVENCOIN_KAWPOW
        .iter()
        .map(|w| format!("0x{w:02x}u"))
        .collect();
    let _ = writeln!(
        s,
        "__constant__ const uint32_t ravencoin_kawpow[15] = {{{}}};\n#define RAVENCOIN_KAWPOW_(i) ravencoin_kawpow[i]",
        words.join(", ")
    );
    s.push_str(&prog_loop_source(p, params.fence));
    // Pragma arguments are not macro-expanded: write the factor in.
    s.push_str(&SEARCH_BODY.replace(
        "#pragma unroll QKM_UNROLL",
        &format!("#pragma unroll {}", params.unroll.max(1)),
    ));
    s
}

/// The DAG generation module (ethash `calcDatasetItem` for items
/// `start..end`, split into [`DAG_FNS`]).
pub const DAG_SOURCE: &str = r#"typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#define FNV_PRIME 0x01000193u

__constant__ const uint64_t keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

__device__ __forceinline__ uint64_t rotl64(const uint64_t x, const int n) { return (x << n) | (x >> (64 - n)); }

__device__ void keccak_f1600(uint64_t st[25])
{
    const int rotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
                          27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
    const int piln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
                          15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
    uint64_t t, bc[5];
#pragma unroll 1
    for (int r = 0; r < 24; r++) {
#pragma unroll
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
#pragma unroll
        for (int i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
#pragma unroll
            for (int j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }
        t = st[1];
#pragma unroll
        for (int i = 0; i < 24; i++) {
            const int j = piln[i];
            bc[0] = st[j];
            st[j] = rotl64(t, rotc[i]);
            t = bc[0];
        }
#pragma unroll
        for (int j = 0; j < 25; j += 5) {
#pragma unroll
            for (int i = 0; i < 5; i++)
                bc[i] = st[j + i];
#pragma unroll
            for (int i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= keccak_rc[r];
    }
}

// Legacy Keccak-512 of 64 bytes (16 little-endian words), in place.
__device__ __forceinline__ void keccak512_64(uint32_t w[16])
{
    uint64_t st[25];
#pragma unroll
    for (int j = 0; j < 8; j++)
        st[j] = (uint64_t)w[2 * j] | ((uint64_t)w[2 * j + 1] << 32);
    st[8] = 0x8000000000000001ULL;
#pragma unroll
    for (int j = 9; j < 25; j++)
        st[j] = 0;
    keccak_f1600(st);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        w[2 * j] = (uint32_t)st[j];
        w[2 * j + 1] = (uint32_t)(st[j] >> 32);
    }
}

// The DAG in three passes over items start..end, so the long parent loop
// runs without Keccak's registers (higher occupancy hides its latency):
// dag_init writes keccak512(light[i % rows] ^ i), dag_mix applies the 512
// parents in place, dag_final writes keccak512 of the result.
extern "C" __global__ void dag_init(const uint32_t start, const uint32_t end, const uint4* __restrict__ light,
                                    const uint32_t light_rows, uint4* dag)
{
    const uint32_t idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end)
        return;
    uint32_t mix[16];
    const uint4* row = light + (uint64_t)(idx % light_rows) * 4;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint4 v = row[k];
        mix[4 * k] = v.x;
        mix[4 * k + 1] = v.y;
        mix[4 * k + 2] = v.z;
        mix[4 * k + 3] = v.w;
    }
    mix[0] ^= idx;
    keccak512_64(mix);
    uint4* out = dag + (uint64_t)idx * 4;
#pragma unroll
    for (int k = 0; k < 4; k++)
        out[k] = make_uint4(mix[4 * k], mix[4 * k + 1], mix[4 * k + 2], mix[4 * k + 3]);
}

// Four threads per item, each holding one uint4 of its mix: the owner of
// word i % 16 broadcasts the parent index and each thread loads its 16
// bytes of the parent row, so a warp reads 8 whole 64-byte rows per load.
extern "C" __global__ void dag_mix(const uint32_t start, const uint32_t end, const uint4* __restrict__ light,
                                   const uint32_t light_rows, uint4* dag)
{
    const uint32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t idx = start + (gtid >> 2);
    const uint32_t t = threadIdx.x & 3;
    const bool live = idx < end;
    uint4* item = dag + (uint64_t)(live ? idx : start) * 4;
    uint4 m = item[t];
#pragma unroll 4
    for (uint32_t i = 0; i < 512; i++) {
        const uint32_t c = i & 3;
        const uint32_t mine = c == 0 ? m.x : c == 1 ? m.y : c == 2 ? m.z : m.w;
        const uint32_t w = __shfl_sync(0xFFFFFFFFu, mine, (i & 15) >> 2, 4);
        const uint32_t parent = (((idx ^ i) * FNV_PRIME) ^ w) % light_rows;
        const uint4 v = __ldg(light + (uint64_t)parent * 4 + t);
        m.x = m.x * FNV_PRIME ^ v.x;
        m.y = m.y * FNV_PRIME ^ v.y;
        m.z = m.z * FNV_PRIME ^ v.z;
        m.w = m.w * FNV_PRIME ^ v.w;
    }
    if (live)
        item[t] = m;
}

extern "C" __global__ void dag_final(const uint32_t start, const uint32_t end, uint4* dag)
{
    const uint32_t idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= end)
        return;
    uint4* item = dag + (uint64_t)idx * 4;
    uint32_t mix[16];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const uint4 v = item[k];
        mix[4 * k] = v.x;
        mix[4 * k + 1] = v.y;
        mix[4 * k + 2] = v.z;
        mix[4 * k + 3] = v.w;
    }
    keccak512_64(mix);
#pragma unroll
    for (int k = 0; k < 4; k++)
        item[k] = make_uint4(mix[4 * k], mix[4 * k + 1], mix[4 * k + 2], mix[4 * k + 3]);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_has_every_op_and_period() {
        let p = Program::new(1_521_745);
        let src = search_source(&p, 24_117_247, SearchParams::default());
        assert!(src.contains("// Inner loop for period 1521745"));
        assert!(src.contains("#define PROGPOW_DAG_ELEMENTS 24117247u"));
        assert_eq!(src.matches("data = c_dag[offset];").count(), 11);
        // 11 cache merges + 18 math merges + 4 DAG merges.
        let merges = src
            .lines()
            .filter(|l| l.starts_with("mix[") && l.contains(" = "))
            .count();
        assert_eq!(merges, 33);
    }
}
