//! The miner: stratum events in, GPU work out, every solution checked on
//! the CPU before it is submitted.
//!
//! Threads:
//! - the coordinator (the caller of [`run`]): turns stratum events into the
//!   current work, decides epoch switches, prints status, reads telemetry;
//! - one thread per GPU: owns that GPU's DAG and kernels, and is the only
//!   thread that builds or drops its DAG, so an epoch switch can never race
//!   a build. The GPUs share the work, the light caches and the compiled
//!   kernels, and each searches its own part of the nonce space;
//! - the submit thread: checks each GPU solution with qkm-kawpow and submits
//!   the ones that meet the target;
//! - the stratum client's threads, and the API thread when enabled.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{
    Receiver, RecvTimeoutError, SyncSender, TrySendError, channel, sync_channel,
};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use qkm_core::kernel::SearchParams;
use qkm_core::qkm_kawpow::kawpow::PARAMS;
use qkm_core::{EPOCH_LENGTH, Epoch, Verifier, boundary64, meets};
use qkm_cuda::{Cubins, Dag, Geometry, Gpu, Header, Kernels, Searcher};
use qkm_stratum::{
    Client, Config, Event, Job, Pool, Share, Submitter, difficulty_of_target, redact,
};

use crate::hooks::{Hooks, HooksConfig, Kind, Note, Watch};
use crate::stats::{GpuState, Stats};
use crate::telemetry::Telemetry;
use crate::tune::TuneConfig;

/// Miner settings.
#[derive(Clone, Debug)]
pub struct MinerConfig {
    /// Pools in order of preference.
    pub pools: Vec<Pool>,
    /// CUDA device indexes (one stratum connection and worker for all).
    pub devices: Vec<usize>,
    /// Seconds between status lines.
    pub status_every: Duration,
    /// Kernel tunables.
    pub search: SearchParams,
    /// Launch geometry.
    pub geometry: Geometry,
    /// Build the next epoch's DAG ahead when VRAM allows both.
    pub dag_prefetch: bool,
    /// A job whose epoch is more than one away from the current one is only
    /// followed when a second job confirms it at least this much later.
    pub far_epoch_confirm: Duration,
    /// GPU settings applied at start (power limit, clocks, fans).
    pub tune: TuneConfig,
    /// The n-th GPU waits n times this before each DAG build, so the GPUs
    /// don't change load all at once.
    pub start_stagger: Duration,
    /// A GPU at this temperature (Celsius) pauses...
    pub temp_limit: Option<u32>,
    /// ...until it cools to this (default: 15 below the limit).
    pub temp_resume: Option<u32>,
    /// Report the hashrate to the pool (`eth_submitHashrate`) every minute.
    pub report_hashrate: bool,
    /// A GPU that mines but whose hash count does not move for this long is
    /// stuck (a hung kernel or driver): the miner exits, for a supervisor
    /// to restart it.
    pub stall_timeout: Duration,
    /// Event hooks (off unless a program or webhook is set).
    pub hooks: HooksConfig,
}

impl MinerConfig {
    /// Defaults for `pools`.
    pub fn new(pools: Vec<Pool>) -> MinerConfig {
        MinerConfig {
            pools,
            devices: vec![0],
            status_every: Duration::from_secs(30),
            search: SearchParams::default(),
            geometry: Geometry {
                grid: 256,
                streams: 2,
            },
            dag_prefetch: true,
            far_epoch_confirm: Duration::from_secs(2),
            tune: TuneConfig::default(),
            start_stagger: Duration::ZERO,
            temp_limit: None,
            temp_resume: None,
            report_hashrate: false,
            stall_timeout: Duration::from_secs(120),
            hooks: HooksConfig::default(),
        }
    }

    /// Where a GPU paused at [`MinerConfig::temp_limit`] starts again.
    pub fn temp_resume(&self) -> u32 {
        self.temp_resume
            .unwrap_or_else(|| self.temp_limit.unwrap_or(0).saturating_sub(15))
    }
}

/// UTC wall time `HH:MM:SS` for log lines.
pub fn now_hms() -> String {
    let s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}", s / 3600 % 24, s / 60 % 60, s % 60)
}

macro_rules! log {
    ($($a:tt)*) => { crate::output::event(format!("{} {}", now_hms(), format!($($a)*))) };
}

/// One unit of GPU work.
#[derive(Clone, Debug)]
struct Work {
    generation: u64,
    session: u64,
    job: String,
    header: [u8; 32],
    height: u64,
    target: [u8; 32],
    /// Nonce bits fixed by the pool's extranonce (already shifted).
    prefix: u64,
    /// Mask of the nonce bits the miner chooses.
    free_mask: u64,
}

/// The current work, with a generation number that changes on any update.
#[derive(Default)]
struct Slot {
    work: Mutex<Option<Work>>,
    generation: AtomicU64,
    changed: Condvar,
}

impl Slot {
    fn set(&self, w: Option<Work>) {
        let mut g = self.work.lock().unwrap_or_else(|e| e.into_inner());
        let n = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        *g = w.map(|mut w| {
            w.generation = n;
            w
        });
        self.changed.notify_all();
    }

    fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// The work, waiting up to `d` when there is none.
    fn get(&self, d: Duration) -> Option<Work> {
        let g = self.work.lock().unwrap_or_else(|e| e.into_inner());
        if g.is_some() {
            return g.clone();
        }
        let (g, _) = self
            .changed
            .wait_timeout(g, d)
            .unwrap_or_else(|e| e.into_inner());
        g.clone()
    }
}

/// A GPU solution waiting for the CPU check.
struct Candidate {
    /// The GPU that found it (an index into `Stats::gpus`).
    gpu: usize,
    work: Work,
    nonce: u64,
    mix: [u32; 8],
}

fn words_le(b: &[u8; 32]) -> [u32; 8] {
    Header::from_bytes(b).0
}

/// xorshift64*, seeded from the clock.
struct Rng(u64);
impl Rng {
    fn new() -> Rng {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(1, |d| d.as_nanos() as u64);
        Rng(t | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// GPU `i` of `n`'s part of the nonce bits the pool leaves free
/// (`free_mask`, the low bits): the top free bits select the GPU. Returns
/// the GPU's prefix and the mask of the bits it counts in.
fn nonce_range(prefix: u64, free_mask: u64, i: usize, n: usize) -> (u64, u64) {
    let k = n.max(1).next_power_of_two().trailing_zeros();
    if k == 0 {
        return (prefix, free_mask);
    }
    let free_bits = free_mask.count_ones();
    (prefix | ((i as u64) << (free_bits - k)), free_mask >> k)
}

struct GpuCtx {
    /// Index into `Stats::gpus`, and the number of GPUs.
    index: usize,
    count: usize,
    /// `None` while the device could not be reopened after a reset.
    gpu: Option<Arc<Gpu>>,
    cubins: Arc<Cubins>,
    slot: Arc<Slot>,
    stats: Arc<Stats>,
    verifier: Arc<Verifier>,
    candidates: SyncSender<Candidate>,
    stop: Arc<AtomicBool>,
    cfg: MinerConfig,
    hooks: Hooks,
}

fn gib(b: u64) -> f64 {
    b as f64 / f64::from(1u32 << 30)
}

/// Failures in a row (with no hashing in between) before the device's
/// context is reset: a sticky CUDA error (an illegal address or ECC fault,
/// often from too much overclock) ruins a context for good.
const RESET_AFTER: u32 = 3;
/// Failures in a row before the GPU is given up (the miner exits when
/// every GPU has).
const GIVE_UP_AFTER: u32 = 8;

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

/// The GPU thread. Errors and panics are logged and retried after a pause;
/// see [`RESET_AFTER`] and [`GIVE_UP_AFTER`].
fn gpu_thread(mut c: GpuCtx) {
    let stats = c.stats.clone();
    let st = &stats.gpus()[c.index];
    let dev = st.device;
    let mut failures = 0u32;
    while !c.stop.load(Ordering::SeqCst) {
        let mined_before = st.hashes.load(Ordering::Relaxed);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu_loop(&c)))
            .unwrap_or_else(|p| Err(format!("panicked: {}", panic_text(p.as_ref()))));
        let e = match r {
            Ok(()) => return,
            Err(e) => e,
        };
        if st.hashes.load(Ordering::Relaxed) > mined_before {
            failures = 0;
        }
        failures += 1;
        st.set_state(GpuState::Error);
        if failures >= GIVE_UP_AFTER {
            log!("gpu{dev}: GPU error ({failures} in a row): {e}; giving this GPU up");
            c.hooks.fire(Note::gpu(
                Kind::GpuFailed,
                dev,
                format!("given up after {failures} errors in a row: {e}"),
            ));
            st.set_state(GpuState::Failed);
            return;
        }
        log!("gpu{dev}: GPU error ({failures}): {e}; retrying");
        c.hooks
            .fire(Note::gpu(Kind::GpuError, dev, format!("error {failures}: {e}")));
        if failures >= RESET_AFTER {
            // Everything on the old context was dropped with gpu_loop's
            // state (the DAG built ahead is joined on drop); only this
            // handle is left.
            c.gpu = None;
            match qkm_cuda::reset_device(dev).and_then(|()| Gpu::new(dev)) {
                Ok(g) => {
                    log!("gpu{dev}: device reset");
                    c.hooks.fire(Note::gpu(
                        Kind::GpuReset,
                        dev,
                        format!("device reset after {failures} errors in a row"),
                    ));
                    c.gpu = Some(Arc::new(g));
                }
                Err(err) => {
                    log!("gpu{dev}: device reset failed: {err}");
                    c.hooks.fire(Note::gpu(
                        Kind::GpuError,
                        dev,
                        format!("device reset failed: {err}"),
                    ));
                }
            }
        }
        let wait = Duration::from_secs(u64::from(failures.min(6)) * 5);
        let until = Instant::now() + wait;
        while Instant::now() < until && !c.stop.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

/// A next-epoch DAG being built on another thread (no build when the card
/// can't hold two DAGs and the switch rebuilds instead). Dropping it waits
/// for the build, so a device reset never races one.
struct NextDag {
    epoch: u64,
    build: Option<std::thread::JoinHandle<Result<Dag, String>>>,
}

impl Drop for NextDag {
    fn drop(&mut self) {
        if let Some(h) = self.build.take() {
            let _ = h.join();
        }
    }
}

#[allow(clippy::too_many_lines)]
fn gpu_loop(c: &GpuCtx) -> Result<(), String> {
    let Some(gpu) = &c.gpu else {
        return Err("the device could not be reopened".into());
    };
    let st = &c.stats.gpus()[c.index];
    let dev = st.device;
    let e = |e: qkm_cuda::Error| e.to_string();
    let mut kernels = Kernels::shared(gpu, c.cubins.clone());
    let mut searcher = Searcher::new(gpu, c.cfg.geometry).map_err(e)?;
    let mut dag: Option<Dag> = None;
    let mut next_dag: Option<NextDag> = None;
    let mut light_prefetched: Option<u64> = None;
    let mut rng = Rng::new();
    let mut current: Option<(String, u64)> = None;
    let mut nonce = 0u64;
    let mut overflow_logged = 0u64;
    loop {
        if c.stop.load(Ordering::SeqCst) {
            return Ok(());
        }
        if st.paused.load(Ordering::Relaxed) {
            st.set_state(GpuState::TooHot);
            std::thread::sleep(Duration::from_millis(250));
            continue;
        }
        let Some(w) = c.slot.get(Duration::from_millis(250)) else {
            st.set_state(GpuState::Idle);
            continue;
        };
        let epoch = Epoch::for_height(w.height);
        if dag.as_ref().map(|d| d.epoch) != Some(epoch) {
            // The DAG built ahead: wait for it if this is its epoch; any
            // other is dropped (joined first, so two builds never overlap).
            let mut ready = None;
            if let Some(mut nd) = next_dag.take()
                && let Some(h) = nd.build.take()
            {
                match h.join() {
                    Ok(Ok(d)) if nd.epoch == epoch.number => ready = Some(d),
                    Ok(Ok(_)) => {}
                    Ok(Err(err)) => {
                        log!("gpu{dev}: epoch {}: DAG built ahead failed: {err}", nd.epoch);
                    }
                    Err(_) => log!("gpu{dev}: epoch {}: the DAG prefetch thread panicked", nd.epoch),
                }
            }
            // Free the old DAG before allocating the next one.
            drop(dag.take());
            if let Some(d) = ready {
                log!("gpu{dev}: epoch {}: prefetched DAG ready", epoch.number);
                dag = Some(d);
                continue;
            }
            gpu.release_memory().map_err(e)?;
            st.set_state(GpuState::BuildingDag(epoch.number));
            let (free, _) = gpu.memory().map_err(e)?;
            if (free as u64) < epoch.dataset_bytes {
                return Err(format!(
                    "epoch {} needs a {:.2} GiB DAG but only {:.2} GiB of GPU memory is free",
                    epoch.number,
                    gib(epoch.dataset_bytes),
                    gib(free as u64)
                ));
            }
            // The light cache is shared: every GPU waits for the same build.
            let t = Instant::now();
            let ec = c.verifier.epoch_cache(w.height);
            let cache_s = t.elapsed().as_secs_f64();
            // Then stagger the GPUs from that moment (the first goes at once).
            let until = Instant::now() + c.cfg.start_stagger.saturating_mul(c.index as u32);
            while Instant::now() < until {
                if c.stop.load(Ordering::SeqCst) {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let t = Instant::now();
            dag = Some(gpu.generate_dag(epoch, &ec.cache).map_err(e)?);
            log!(
                "gpu{dev}: epoch {}: DAG {:.2} GiB built in {:.1} s (light cache {:.1} s)",
                epoch.number,
                gib(epoch.dataset_bytes),
                t.elapsed().as_secs_f64(),
                cache_s
            );
            continue;
        }
        let Some(d) = dag.as_ref() else { continue };
        // Near the end of an epoch: the next light cache (CPU), and the next
        // DAG when the card can hold both.
        let into = w.height % EPOCH_LENGTH;
        let next_first = (epoch.number + 1) * EPOCH_LENGTH;
        if into + 100 >= EPOCH_LENGTH && light_prefetched != Some(epoch.number + 1) {
            light_prefetched = Some(epoch.number + 1);
            let v = c.verifier.clone();
            std::thread::spawn(move || {
                let _ = v.epoch_cache(next_first);
            });
            kernels.prefetch(
                qkm_core::period(next_first),
                Epoch::for_height(next_first).dag_elements(),
            );
        }
        if c.cfg.dag_prefetch && into + 30 >= EPOCH_LENGTH && next_dag.is_none() {
            let next = Epoch::for_height(next_first);
            let (free, _) = gpu.memory().map_err(e)?;
            let build = (free as u64 > next.dataset_bytes + (512 << 20)).then(|| {
                let (g, v) = (gpu.clone(), c.verifier.clone());
                log!("gpu{dev}: epoch {}: building the next DAG ahead", next.number);
                std::thread::spawn(move || {
                    let ec = v.epoch_cache(next_first);
                    g.generate_dag(next, &ec.cache).map_err(|e| e.to_string())
                })
            });
            next_dag = Some(NextDag {
                epoch: next.number,
                build,
            });
        }
        let k = kernels.get(gpu, w.height, d.epoch).map_err(e)?;
        st.set_state(GpuState::Mining);
        let (prefix, mask) = nonce_range(w.prefix, w.free_mask, c.index, c.count);
        // Nonces one round of launches covers past those already counted.
        let inflight = c.cfg.geometry.batch(k.params.block) * c.cfg.geometry.streams.max(1) as u64;
        let key = (w.job.clone(), w.session);
        if current.as_ref() != Some(&key) {
            current = Some(key);
            // A random start in this GPU's free bits, leaving room to count up.
            nonce = prefix | (rng.next() & (mask >> 1));
        }
        // Stay inside this GPU's slice: start over at its bottom when the
        // top is near, and stop each run before reaching it.
        if mask - (nonce & mask) < 4 * inflight {
            nonce = prefix;
        }
        let room = mask - (nonce & mask);
        let t0 = Instant::now();
        let mut counted = 0u64;
        let generation = w.generation;
        let (done, found) = searcher
            .run(
                &k,
                d,
                Header::from_bytes(&w.header),
                boundary64(&w.target),
                nonce,
                &mut |done, found| {
                    st.hashes.fetch_add(done - counted, Ordering::Relaxed);
                    counted = done;
                    !c.stop.load(Ordering::SeqCst)
                        && !st.paused.load(Ordering::Relaxed)
                        && c.slot.generation() == generation
                        && found.is_empty()
                        && done + 2 * inflight < room
                        && t0.elapsed() < Duration::from_secs(2)
                },
            )
            .map_err(e)?;
        nonce = prefix | (nonce.wrapping_add(done) & mask);
        if searcher.overflow > overflow_logged {
            log!(
                "gpu{dev}: {} solutions lost (more than a launch can report): the share difficulty is far too low",
                searcher.overflow - overflow_logged
            );
            overflow_logged = searcher.overflow;
        }
        for f in found {
            let cand = Candidate {
                gpu: c.index,
                work: w.clone(),
                nonce: f.nonce,
                mix: f.mix,
            };
            if let Err(TrySendError::Full(_)) = c.candidates.try_send(cand) {
                c.stats.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Solutions waiting for the CPU check (about 150 a second on one core);
/// beyond this many they are dropped and counted.
const CANDIDATE_QUEUE: usize = 256;

/// Final hashes of recent shares remembered to never send one twice.
const SENT_REMEMBERED: usize = 4096;

/// The submit thread: qkm-kawpow checks every solution; mismatches are GPU
/// bugs and are never submitted, and a share already sent is not sent
/// again. (Quai nodes count a share by its final hash, header and nonce, and
/// re-send the same work under a new job id after every accepted share: a
/// miner that searches it again from the same nonce re-finds its own
/// shares, which the node refuses as duplicates.)
fn submit_thread(
    rx: Receiver<Candidate>,
    verifier: Arc<Verifier>,
    submit: Submitter,
    stats: Arc<Stats>,
    seq: Arc<AtomicU64>,
) {
    let mut sent: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
    let mut order: std::collections::VecDeque<[u8; 32]> = std::collections::VecDeque::new();
    for c in rx {
        let h = verifier.hash(c.work.height, &c.work.header, c.nonce);
        if words_le(&h.mix) != c.mix {
            stats.gpu_errors.fetch_add(1, Ordering::Relaxed);
            log!(
                "BUG: gpu{}'s mix for nonce {:#018x} at height {} differs from the CPU reference's; not submitted",
                stats.gpus().get(c.gpu).map_or(c.gpu, |g| g.device),
                c.nonce,
                c.work.height
            );
            continue;
        }
        // The GPU compares the top 64 bits only.
        if !meets(&h.hash, &c.work.target) {
            continue;
        }
        if !sent.insert(h.hash) {
            stats.duplicates.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        order.push_back(h.hash);
        if order.len() > SENT_REMEMBERED
            && let Some(old) = order.pop_front()
        {
            sent.remove(&old);
        }
        let n = seq.fetch_add(1, Ordering::SeqCst) + 1;
        stats.share_sent(n, c.gpu);
        submit.submit(Share {
            seq: n,
            job: c.work.job.clone(),
            session: c.work.session,
            nonce: c.nonce,
            header: c.work.header,
            mix: h.mix,
        });
    }
}

/// The coordinator's view of the pool.
struct PoolState {
    session: u64,
    prefix: u64,
    free_mask: u64,
    target: Option<[u8; 32]>,
    authorized: bool,
    job: Option<Job>,
    /// The epoch the miner follows.
    epoch: Option<u64>,
    /// A far-epoch job waiting for confirmation, and when it first came.
    suspect: Option<(u64, Instant, Job)>,
}

impl PoolState {
    fn new() -> PoolState {
        PoolState {
            session: 0,
            prefix: 0,
            free_mask: u64::MAX,
            target: None,
            authorized: false,
            job: None,
            epoch: None,
            suspect: None,
        }
    }

    fn work(&self) -> Option<Work> {
        let (Some(job), Some(target)) = (&self.job, self.target) else {
            return None;
        };
        if !self.authorized {
            return None;
        }
        Some(Work {
            generation: 0,
            session: self.session,
            job: job.id.clone(),
            header: job.header,
            height: job.height,
            target,
            prefix: self.prefix,
            free_mask: self.free_mask,
        })
    }
}

/// Decides whether to follow `job`: the same or the next epoch at once;
/// any other (an earlier one: a stale template; a far one) only when a
/// second job from it confirms it `confirm` later. Following a stray job
/// would cost every GPU two DAG builds.
fn follow(ps: &mut PoolState, job: Job, confirm: Duration) -> Option<Job> {
    let e = job.height / EPOCH_LENGTH;
    let Some(cur) = ps.epoch else {
        ps.epoch = Some(e);
        ps.suspect = None;
        return Some(job);
    };
    if e == cur || e == cur + 1 {
        ps.epoch = Some(e);
        ps.suspect = None;
        return Some(job);
    }
    match &ps.suspect {
        Some((se, first, _)) if *se == e && first.elapsed() >= confirm => {
            log!("following the pool to epoch {e} (confirmed by a second job)");
            ps.epoch = Some(e);
            ps.suspect = None;
            Some(job)
        }
        Some((se, ..)) if *se == e => None,
        _ => {
            log!(
                "job {} at height {} is in epoch {e}, not epoch {cur} or the next; waiting for a second job before switching",
                job.id,
                job.height
            );
            ps.suspect = Some((e, Instant::now(), job));
            None
        }
    }
}

/// Why mining stopped short.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// A restart will not help (configuration, every pool refused us):
    /// exit code 2, else 1.
    pub permanent: bool,
    /// What happened.
    pub reason: String,
}

impl Failure {
    fn permanent(reason: impl Into<String>) -> Failure {
        Failure {
            permanent: true,
            reason: reason.into(),
        }
    }

    fn transient(reason: impl Into<String>) -> Failure {
        Failure {
            permanent: false,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

/// Hashes per KawPoW stratum difficulty unit (difficulty 1 = target
/// `0x00000000ff…`).
const HASHES_PER_DIFF: f64 = 1_099_511_627_776.0 / 255.0;

/// Joins `h` within `deadline`; a thread stuck (in a hung CUDA call) is
/// left behind rather than blocking the exit.
fn join_by<T>(h: std::thread::JoinHandle<T>, deadline: Instant) -> bool {
    while !h.is_finished() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = h.join();
    true
}

/// Mines until `stop` is set (`Ok`), or until mining cannot go on (`Err`:
/// every pool refused authorization, every GPU failed, a GPU stopped
/// hashing, or the stratum client died). The stats are shared with the
/// caller (API, TUI).
#[allow(clippy::too_many_lines)]
pub fn run(cfg: MinerConfig, stop: Arc<AtomicBool>, stats: Arc<Stats>) -> Result<(), Failure> {
    let listed: Vec<usize> = stats.gpus().iter().map(|g| g.device).collect();
    if listed != cfg.devices {
        return Err(Failure::permanent(format!(
            "the stats are for devices {listed:?} but the miner is for {:?}",
            cfg.devices
        )));
    }
    // Settings first, so no GPU hashes before its clocks are set.
    if !cfg.tune.is_empty() {
        apply_tuning(&cfg.devices, &cfg.tune);
    }
    // Open every GPU; one that fails is left out (and shows as idle).
    let mut opened = Vec::new();
    let mut gpus = Vec::new();
    for (i, &d) in cfg.devices.iter().enumerate() {
        match Gpu::new(d) {
            Ok(gpu) => {
                if let Ok(bus) = qkm_cuda::pci_bus_id(d) {
                    stats.gpus()[i].set_bus(&bus);
                }
                log!(
                    "gpu{d}: {} (sm_{}{}), {:.2} GiB",
                    gpu.name,
                    gpu.arch.0,
                    gpu.arch.1,
                    gib(gpu.memory().map_or(0, |m| m.1) as u64)
                );
                let telemetry = qkm_cuda::pci_bus_id(d)
                    .ok()
                    .and_then(|id| Telemetry::open(Some(&id), d as u32));
                if telemetry.is_none() {
                    log!("gpu{d}: NVML unavailable: no temperature or power readings");
                }
                gpus.push((i, gpu.name.clone(), telemetry));
                opened.push((i, Arc::new(gpu)));
            }
            Err(e) => log!("gpu{d}: cannot open: {e}; mining without it"),
        }
    }
    if opened.is_empty() {
        return Err(Failure::transient(format!(
            "no usable GPU among devices {:?}",
            cfg.devices
        )));
    }
    if let Some(p) = cfg.pools.first() {
        stats.set_worker(&p.user);
        if let Ok(pay) = crate::quai::parse_payout(&p.user) {
            let lock = p
                .pass
                .split([',', '_'])
                .find_map(|o| o.strip_prefix("lock="))
                .unwrap_or("0");
            stats.set_payout(&format!("{} · {} · lock {lock}", pay.ledger, pay.zone()));
        }
    }
    for (i, p) in cfg.pools.iter().enumerate() {
        log!(
            "pool {i}: {} as {} ({})",
            p.host_port(),
            p.user,
            redact(&p.pass)
        );
    }
    if let Some(l) = cfg.temp_limit {
        log!(
            "temperature limit {l}C: a GPU that reaches it pauses until it cools to {}C",
            cfg.temp_resume()
        );
    }
    let user = cfg.pools.first().map_or("", |p| p.user.as_str());
    let (hooks, hook_runner) = Hooks::start(cfg.hooks.clone(), user, stats.clone());
    if hook_runner.is_some() {
        log!("{}", cfg.hooks.describe());
    }
    hooks.fire(Note::new(
        Kind::MinerStarted,
        format!(
            "started on {} of {} GPUs",
            opened.len(),
            cfg.devices.len()
        ),
    ));
    let verifier = Arc::new(Verifier::default());
    let slot = Arc::new(Slot::default());
    let (ev_tx, ev_rx) = channel();
    let client = Client::start(Config::new(cfg.pools.clone()), ev_tx);
    let (cand_tx, cand_rx) = sync_channel(CANDIDATE_QUEUE);
    let seq = Arc::new(AtomicU64::new(0));
    let submitter = {
        let (v, s, st, q) = (
            verifier.clone(),
            client.submitter(),
            stats.clone(),
            seq.clone(),
        );
        std::thread::Builder::new()
            .name("submit".into())
            .spawn(move || submit_thread(cand_rx, v, s, st, q))
            .map_err(|e| Failure::transient(e.to_string()))?
    };
    let hashrate_reporter = client.submitter();
    let cubins = Arc::new(Cubins::new(cfg.search));
    let mut gpu_handles = Vec::new();
    for (i, gpu) in opened {
        let ctx = GpuCtx {
            index: i,
            count: cfg.devices.len(),
            gpu: Some(gpu),
            cubins: cubins.clone(),
            slot: slot.clone(),
            stats: stats.clone(),
            verifier: verifier.clone(),
            candidates: cand_tx.clone(),
            stop: stop.clone(),
            cfg: cfg.clone(),
            hooks: hooks.clone(),
        };
        gpu_handles.push(
            std::thread::Builder::new()
                .name(format!("gpu{}", cfg.devices[i]))
                .spawn(move || gpu_thread(ctx))
                .map_err(|e| Failure::transient(e.to_string()))?,
        );
    }
    // The submit thread ends when the last GPU thread drops its sender.
    drop(cand_tx);
    let mut ps = PoolState::new();
    let mut last_status = Instant::now();
    let mut last_sample = Instant::now() - Duration::from_secs(10);
    let mut last_report = Instant::now();
    let mut dropped_logged = 0u64;
    // Per GPU: the hash count last seen and since when it has not moved.
    let mut progress: Vec<(u64, Instant)> = vec![(0, Instant::now()); stats.gpus().len()];
    // The epoch whose seed hash was last computed, and the seed.
    let mut seed: Option<(u64, [u8; 32])> = None;
    // The difficulty the share-rate warnings were last checked for.
    let mut warned_for: Option<f64> = None;
    let mining_since = Instant::now();
    let mut watch = Watch::new(&cfg.hooks, mining_since);
    let mut fail: Option<Failure> = None;
    while !stop.load(Ordering::SeqCst) {
        if last_sample.elapsed() >= Duration::from_secs(5) {
            last_sample = Instant::now();
            stats.sample();
            for (i, name, t) in &gpus {
                let g = &stats.gpus()[*i];
                let r = t.as_ref().map(Telemetry::read).unwrap_or_default();
                // Thermal limit, with hysteresis.
                if let (Some(limit), Some(temp)) = (cfg.temp_limit, r.temperature_c) {
                    let paused = g.paused.load(Ordering::Relaxed);
                    if !paused && temp >= limit {
                        g.paused.store(true, Ordering::Relaxed);
                        log!(
                            "gpu{}: {temp}C reached the {limit}C limit; paused until {}C",
                            g.device,
                            cfg.temp_resume()
                        );
                        hooks.fire(Note::gpu(
                            Kind::GpuTooHot,
                            g.device,
                            format!(
                                "{temp}C reached the {limit}C limit; paused until {}C",
                                cfg.temp_resume()
                            ),
                        ));
                    } else if paused && temp <= cfg.temp_resume() {
                        g.paused.store(false, Ordering::Relaxed);
                        log!("gpu{}: cooled to {temp}C; mining again", g.device);
                        hooks.fire(Note::gpu(
                            Kind::GpuCooled,
                            g.device,
                            format!("cooled to {temp}C; mining again"),
                        ));
                    }
                }
                g.set_reading(name, r);
            }
            // A GPU that says it mines but whose count does not move is
            // stuck in a kernel or the driver; only a new process helps.
            for (k, g) in stats.gpus().iter().enumerate() {
                let h = g.hashes.load(Ordering::Relaxed);
                if h != progress[k].0 || g.state() != GpuState::Mining {
                    progress[k] = (h, Instant::now());
                } else if progress[k].1.elapsed() >= cfg.stall_timeout {
                    fail = Some(Failure::transient(format!(
                        "gpu{} stopped hashing {} s ago (a hung kernel or driver)",
                        g.device,
                        progress[k].1.elapsed().as_secs()
                    )));
                }
            }
            if gpu_handles.iter().all(std::thread::JoinHandle::is_finished) {
                fail = Some(Failure::transient("every GPU failed"));
            }
            if fail.is_some() {
                break;
            }
            // States that last too long: no pool, a low hashrate.
            let now = Instant::now();
            if let Some(n) = watch.pool_usable(ps.authorized, now) {
                log!("{}", n.detail);
                hooks.fire(n);
            }
            if let Some(n) = watch.hashrate(stats.hashrate(Duration::from_secs(60)), now) {
                log!("{}", n.detail);
                hooks.fire(n);
            }
            let dropped = stats.dropped.load(Ordering::Relaxed);
            if dropped > dropped_logged {
                log!(
                    "{} GPU solutions dropped: the CPU check cannot keep up; raise the share difficulty (d=)",
                    dropped - dropped_logged
                );
                dropped_logged = dropped;
            }
            // Share-rate sanity, once the hashrate is known.
            let rate = stats.hashrate(Duration::from_secs(60));
            let diff = ps.target.map(|t| difficulty_of_target(&t));
            if let Some(d) = diff
                && mining_since.elapsed() >= Duration::from_secs(120)
                && rate > 0.0
                && warned_for != Some(d)
            {
                warned_for = Some(d);
                let per_share = d * HASHES_PER_DIFF / rate;
                if per_share > 300.0 {
                    log!(
                        "difficulty {d}: one share every {:.0} min at {:.1} MH/s; Quai nodes close a connection after 10 min without a share. Lower d=",
                        per_share / 60.0,
                        rate / 1e6
                    );
                } else if per_share < 0.05 {
                    log!(
                        "difficulty {d}: {:.0} shares a second at {:.1} MH/s; the CPU check may fall behind. Raise d=",
                        1.0 / per_share,
                        rate / 1e6
                    );
                }
            }
        }
        if cfg.report_hashrate
            && last_report.elapsed() >= Duration::from_secs(60)
            && ps.authorized
        {
            last_report = Instant::now();
            hashrate_reporter.report_hashrate(stats.hashrate(Duration::from_secs(60)));
        }
        if last_status.elapsed() >= cfg.status_every {
            last_status = Instant::now();
            print_status(&stats);
        }
        // A suspect job nobody contradicted for long enough.
        if ps
            .suspect
            .as_ref()
            .is_some_and(|(_, t, _)| t.elapsed() >= cfg.far_epoch_confirm * 5)
            && let Some((e, _, job)) = ps.suspect.take()
        {
            log!("following the pool to epoch {e} (no other job since)");
            ps.epoch = Some(e);
            stats.set_job(&job.id, job.height);
            ps.job = Some(job);
            slot.set(ps.work());
        }
        let ev = match ev_rx.recv_timeout(Duration::from_millis(500)) {
            Ok(ev) => ev,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                fail = Some(Failure::transient("the stratum client stopped unexpectedly"));
                break;
            }
        };
        match ev {
            Event::Connected {
                pool,
                url,
                session,
                tls,
            } => {
                log!("connected to {url}{}", if tls { " (TLS)" } else { "" });
                for n in watch.connected(pool, &url) {
                    hooks.fire(n);
                }
                stats.set_pool(&url);
                ps.session = session;
                ps.authorized = false;
                ps.job = None;
                // Each pool sends its own target.
                ps.target = None;
                ps.prefix = 0;
                ps.free_mask = u64::MAX;
                slot.set(None);
            }
            Event::Extranonce(x) => {
                // Up to 6 bytes (16 free bits are plenty); Quai's 1-byte
                // session id is harmless as a prefix.
                if x.len() > 6 {
                    log!(
                        "the pool's extranonce is {} bytes; at most 6 are supported, so it is ignored (shares may be refused)",
                        x.len()
                    );
                }
                let n = if x.len() > 6 { 0 } else { x.len() };
                let mut p = 0u64;
                for b in &x[..n] {
                    p = (p << 8) | u64::from(*b);
                }
                let bits = 8 * n as u32;
                ps.prefix = if bits == 0 { 0 } else { p << (64 - bits) };
                ps.free_mask = if bits == 0 {
                    u64::MAX
                } else {
                    u64::MAX >> bits
                };
                slot.set(ps.work());
            }
            Event::Authorized => {
                log!("authorized");
                ps.authorized = true;
                stats.connected.store(true, Ordering::Relaxed);
                slot.set(ps.work());
            }
            Event::AuthFailed(why) => {
                log!("authorization refused: {why}");
                hooks.fire(Note::new(
                    Kind::AuthRefused,
                    format!("authorization refused: {why}"),
                ));
            }
            Event::Target(t) => {
                let d = difficulty_of_target(&t);
                if ps.target != Some(t) {
                    log!("share target: difficulty {d:.4}");
                }
                stats.set_difficulty(d);
                ps.target = Some(t);
                slot.set(ps.work());
            }
            Event::Job(job) => {
                // The parser caps heights, so this stays cheap; computed once
                // per epoch.
                let e = job.height / EPOCH_LENGTH;
                if seed.is_none_or(|(se, _)| se != e) {
                    seed = Some((e, PARAMS.seed_hash(e * EPOCH_LENGTH + 1)));
                }
                if seed.is_some_and(|(_, want)| job.seed != want) {
                    log!("job {}: the seed hash does not match epoch {e}", job.id);
                }
                if let Some(job) = follow(&mut ps, job, cfg.far_epoch_confirm) {
                    stats.set_job(&job.id, job.height);
                    ps.job = Some(job);
                    slot.set(ps.work());
                }
            }
            Event::ShareResult {
                seq,
                accepted,
                error,
                latency,
            } => {
                let by = stats
                    .share_result(seq, accepted, error.as_deref(), latency)
                    .map_or_else(String::new, |d| format!(" from gpu{d}"));
                if accepted {
                    log!(
                        "share {seq}{by} accepted ({} ms)  A{}",
                        latency.as_millis(),
                        stats.accepted.load(Ordering::Relaxed)
                    );
                } else {
                    let why = error.unwrap_or_else(|| "rejected".into());
                    log!("share {seq}{by} not accepted: {why}");
                    // Stale and lost shares are the pool connection's
                    // story (pool_* events), not a refusal.
                    if crate::stats::refusal(&why) == crate::stats::Refusal::Rejected {
                        hooks.fire(Note::new(
                            Kind::ShareRejected,
                            format!("share {seq}{by} rejected: {why}"),
                        ));
                    }
                }
            }
            Event::Notice(n) => log!("{n}"),
            Event::Disconnected { reason } => {
                log!("disconnected: {}", redact(&reason));
                hooks.fire(Note::new(
                    Kind::PoolDisconnected,
                    format!("disconnected: {}", redact(&reason)),
                ));
                stats.connected.store(false, Ordering::Relaxed);
                ps.authorized = false;
                ps.job = None;
                slot.set(None);
            }
            Event::Fatal(why) => {
                fail = Some(Failure::permanent(why));
                break;
            }
        }
    }
    match &fail {
        Some(why) => log!("stopping: {why}"),
        None => log!("stopping"),
    }
    hooks.fire(Note::new(
        Kind::MinerStopping,
        fail.as_ref()
            .map_or_else(|| "stopping (on request)".into(), |f| format!("stopping: {f}")),
    ));
    stop.store(true, Ordering::SeqCst);
    client.stop();
    slot.set(None);
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut stuck = 0;
    for h in gpu_handles {
        if !join_by(h, deadline) {
            stuck += 1;
        }
    }
    if stuck > 0 {
        log!("{stuck} GPU thread(s) did not stop (hung in the driver); exiting anyway");
    }
    let _ = join_by(submitter, deadline);
    print_status(&stats);
    if let Some(r) = hook_runner {
        r.finish(cfg.hooks.timeout + Duration::from_secs(2));
    }
    fail.map_or(Ok(()), Err)
}

/// The status line, then one line per GPU when there are several (stdout
/// in plain mode, the log file either way).
fn print_status(stats: &Stats) {
    let mut out = format!("{} {}", now_hms(), stats.line());
    for l in stats.gpu_lines() {
        out.push('\n');
        out.push_str(&l);
    }
    crate::output::status(&out);
}

/// Applies the GPU settings, one line per GPU; failures are logged and
/// mining goes on. Without root, every setting is refused: that is one line
/// (the settings are then expected from `sudo quai-kawpow-miner tune`).
pub fn apply_tuning(devices: &[usize], tune: &TuneConfig) {
    let nvml = match nvml_wrapper::Nvml::init() {
        Ok(n) => n,
        Err(e) => {
            log!("tuning: NVML unavailable ({e}); GPU settings not applied");
            return;
        }
    };
    let mut lines = Vec::new();
    let (mut set, mut failed, mut denied) = (0, 0, 0);
    for &d in devices {
        let t = tune.for_device(d);
        if t.is_empty() {
            continue;
        }
        let a = match qkm_cuda::pci_bus_id(d) {
            Ok(id) => crate::tune::apply(&nvml, &id, &t),
            Err(e) => crate::tune::Applied {
                text: e.to_string(),
                failed: 1,
                denied: 0,
            },
        };
        set += 1;
        failed += a.failed;
        denied += a.denied;
        let flag = if a.failed > 0 { " (FAILED)" } else { "" };
        lines.push(format!("gpu{d}: tuning: {}{flag}", a.text));
    }
    if set > 0 && failed > 0 && failed == denied {
        // Not root: the settings come from `sudo quai-kawpow-miner tune`.
        // Power limits can be read without root, so check those.
        let off: Vec<String> = devices
            .iter()
            .filter_map(|&d| {
                let want = tune.for_device(d).power_limit?;
                let id = qkm_cuda::pci_bus_id(d).ok()?;
                let have = nvml.device_by_pci_bus_id(id).ok()?.power_management_limit().ok()? / 1000;
                (have != want).then(|| format!("gpu{d} {have} W (want {want} W)"))
            })
            .collect();
        if off.is_empty() {
            log!("tuning: not root; using the GPUs' current settings (power limits are as configured)");
        } else {
            log!(
                "tuning: not root and the GPUs are not as configured: {}; run `sudo quai-kawpow-miner tune` with the same settings",
                off.join(", ")
            );
        }
        return;
    }
    for l in lines {
        log!("{l}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(h: u64) -> Job {
        Job {
            id: format!("j{h}"),
            header: [0; 32],
            seed: [0; 32],
            target: None,
            clean: true,
            height: h,
            bits: String::new(),
        }
    }

    #[test]
    fn gpus_get_disjoint_nonce_ranges() {
        // One GPU: the whole free range.
        assert_eq!(nonce_range(0, u64::MAX, 0, 1), (0, u64::MAX));
        // Seven GPUs under a 2-byte extranonce: 3 bits pick the GPU.
        let prefix = 0xabcd_u64 << 48;
        let free = u64::MAX >> 16;
        let mut seen = Vec::new();
        for i in 0..7 {
            let (p, m) = nonce_range(prefix, free, i, 7);
            assert_eq!(m, free >> 3);
            assert_eq!(p & !free, prefix, "the pool's bits are kept");
            assert_eq!(p & m, 0);
            // The lowest and highest nonce this GPU can reach.
            let (lo, hi) = (p, p | m);
            for &(l, h) in &seen {
                assert!(hi < l || lo > h, "GPU {i} overlaps another");
            }
            seen.push((lo, hi));
        }
        // Two GPUs, no extranonce: the top bit.
        assert_eq!(nonce_range(0, u64::MAX, 1, 2), (1 << 63, u64::MAX >> 1));
    }

    #[test]
    fn far_epoch_needs_a_confirming_second_job() {
        let mut ps = PoolState::new();
        let c = Duration::from_millis(50);
        assert!(follow(&mut ps, job(4_565_000), c).is_some()); // epoch 608
        assert!(follow(&mut ps, job(4_567_600), c).is_some()); // 609: adjacent
        // A stale template's epoch: ignored, then ignored again too soon.
        assert!(follow(&mut ps, job(4_206_442), c).is_none());
        assert!(follow(&mut ps, job(4_206_443), c).is_none());
        // Back to normal clears it.
        assert!(follow(&mut ps, job(4_567_601), c).is_some());
        // One stale job from the previous epoch is ignored too.
        assert!(follow(&mut ps, job(4_567_000), c).is_none());
        assert!(follow(&mut ps, job(4_567_602), c).is_some());
        assert!(ps.suspect.is_none());
        // Confirmed after the delay.
        assert!(follow(&mut ps, job(100), c).is_none());
        std::thread::sleep(Duration::from_millis(60));
        assert!(follow(&mut ps, job(101), c).is_some());
        assert_eq!(ps.epoch, Some(0));
    }
}
