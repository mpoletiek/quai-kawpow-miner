//! The miner: stratum events in, GPU work out, every solution checked on
//! the CPU before it is submitted.
//!
//! Threads:
//! - the coordinator (the caller of [`run`]): turns stratum events into the
//!   current work, decides epoch switches, prints status, reads telemetry;
//! - the GPU thread: owns the DAG and the kernels, and is the only thread
//!   that builds or drops a DAG, so an epoch switch can never race a build;
//! - the submit thread: checks each GPU solution with rsq-pow and submits
//!   the ones that meet the target;
//! - the stratum client's threads, and the API thread when enabled.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use qkm_core::kernel::SearchParams;
use qkm_core::rsq_pow::kawpow::PARAMS;
use qkm_core::{EPOCH_LENGTH, Epoch, Verifier, boundary64, meets};
use qkm_cuda::{Dag, Geometry, Gpu, Header, Kernels, Searcher};
use qkm_stratum::{
    Client, Config, Event, Job, Pool, Share, Submitter, difficulty_of_target, redact,
};

use crate::stats::{GpuState, Stats};
use crate::telemetry::Telemetry;

/// Miner settings.
#[derive(Clone, Debug)]
pub struct MinerConfig {
    /// Pools in order of preference.
    pub pools: Vec<Pool>,
    /// CUDA device index.
    pub device: usize,
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
}

impl MinerConfig {
    /// Defaults for `pools`.
    pub fn new(pools: Vec<Pool>) -> MinerConfig {
        MinerConfig {
            pools,
            device: 0,
            status_every: Duration::from_secs(30),
            search: SearchParams::default(),
            geometry: Geometry {
                grid: 256,
                streams: 2,
            },
            dag_prefetch: true,
            far_epoch_confirm: Duration::from_secs(2),
        }
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
    ($($a:tt)*) => { println!("{} {}", now_hms(), format!($($a)*)) };
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

struct GpuCtx {
    gpu: Arc<Gpu>,
    slot: Arc<Slot>,
    stats: Arc<Stats>,
    verifier: Arc<Verifier>,
    candidates: Sender<Candidate>,
    stop: Arc<AtomicBool>,
    cfg: MinerConfig,
}

fn gib(b: u64) -> f64 {
    b as f64 / f64::from(1u32 << 30)
}

/// The GPU thread. CUDA errors are logged and retried; they never end the
/// miner.
fn gpu_thread(c: GpuCtx) {
    let mut failures = 0u32;
    while !c.stop.load(Ordering::SeqCst) {
        match gpu_loop(&c) {
            Ok(()) => return,
            Err(e) => {
                failures += 1;
                log!("GPU error ({failures}): {e}; retrying");
                c.stats.set_gpu(GpuState::Idle);
                let wait = Duration::from_secs(u64::from(failures.min(6)) * 5);
                let until = Instant::now() + wait;
                while Instant::now() < until && !c.stop.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        }
    }
}

/// A next-epoch DAG being built on another thread (`None` when the card
/// can't hold two DAGs and the switch rebuilds instead).
type NextDag = Option<std::thread::JoinHandle<Result<Dag, String>>>;

fn gpu_loop(c: &GpuCtx) -> Result<(), String> {
    let gpu = &c.gpu;
    let e = |e: qkm_cuda::Error| e.to_string();
    let mut kernels = Kernels::new(gpu, c.cfg.search);
    let mut searcher = Searcher::new(gpu, c.cfg.geometry).map_err(e)?;
    let mut dag: Option<Dag> = None;
    let mut next_dag: Option<(u64, NextDag)> = None;
    let mut light_prefetched: Option<u64> = None;
    let mut rng = Rng::new();
    let mut current: Option<(String, u64)> = None;
    let mut nonce = 0u64;
    loop {
        if c.stop.load(Ordering::SeqCst) {
            return Ok(());
        }
        let Some(w) = c.slot.get(Duration::from_millis(250)) else {
            c.stats.set_gpu(GpuState::Idle);
            continue;
        };
        let epoch = Epoch::for_height(w.height);
        if dag.as_ref().map(|d| d.epoch) != Some(epoch) {
            // The DAG built ahead: wait for it if this is its epoch; any
            // other is dropped (joined first, so two builds never overlap).
            let mut ready = None;
            if let Some((n, Some(h))) = next_dag.take() {
                match h.join() {
                    Ok(Ok(d)) if n == epoch.number => ready = Some(d),
                    Ok(Ok(_)) => {}
                    Ok(Err(err)) => log!("epoch {n}: DAG built ahead failed: {err}"),
                    Err(_) => log!("epoch {n}: the DAG prefetch thread panicked"),
                }
            }
            // Free the old DAG before allocating the next one.
            drop(dag.take());
            if let Some(d) = ready {
                log!("epoch {}: prefetched DAG ready", epoch.number);
                dag = Some(d);
                continue;
            }
            gpu.release_memory().map_err(e)?;
            c.stats.set_gpu(GpuState::BuildingDag(epoch.number));
            let (free, _) = gpu.memory().map_err(e)?;
            if (free as u64) < epoch.dataset_bytes {
                return Err(format!(
                    "epoch {} needs a {:.2} GiB DAG but only {:.2} GiB of GPU memory is free",
                    epoch.number,
                    gib(epoch.dataset_bytes),
                    gib(free as u64)
                ));
            }
            let t = Instant::now();
            let ec = c.verifier.epoch_cache(w.height);
            let cache_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            dag = Some(gpu.generate_dag(epoch, &ec.cache).map_err(e)?);
            log!(
                "epoch {}: DAG {:.2} GiB built in {:.1} s (light cache {:.1} s)",
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
            if free as u64 > next.dataset_bytes + (512 << 20) {
                let (g, v) = (gpu.clone(), c.verifier.clone());
                log!("epoch {}: building the next DAG ahead", next.number);
                let h = std::thread::spawn(move || {
                    let ec = v.epoch_cache(next_first);
                    g.generate_dag(next, &ec.cache).map_err(|e| e.to_string())
                });
                next_dag = Some((next.number, Some(h)));
            } else {
                // Not enough memory for two DAGs: rebuild at the switch.
                next_dag = Some((next.number, None));
            }
        }
        let k = kernels.get(gpu, w.height, d.epoch).map_err(e)?;
        c.stats.set_gpu(GpuState::Mining);
        let key = (w.job.clone(), w.session);
        if current.as_ref() != Some(&key) {
            current = Some(key);
            // A random start in the free bits, leaving room to count up.
            nonce = w.prefix | (rng.next() & (w.free_mask >> 1));
        }
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
                    c.stats.hashes.fetch_add(done - counted, Ordering::Relaxed);
                    counted = done;
                    !c.stop.load(Ordering::SeqCst)
                        && c.slot.generation() == generation
                        && found.is_empty()
                        && t0.elapsed() < Duration::from_secs(2)
                },
            )
            .map_err(e)?;
        nonce = w.prefix | (nonce.wrapping_add(done) & w.free_mask);
        for f in found {
            let _ = c.candidates.send(Candidate {
                work: w.clone(),
                nonce: f.nonce,
                mix: f.mix,
            });
        }
    }
}

/// The submit thread: rsq-pow checks every solution; mismatches are GPU
/// bugs and are never submitted.
fn submit_thread(
    rx: Receiver<Candidate>,
    verifier: Arc<Verifier>,
    submit: Submitter,
    stats: Arc<Stats>,
    seq: Arc<AtomicU64>,
) {
    for c in rx {
        let h = verifier.hash(c.work.height, &c.work.header, c.nonce);
        if words_le(&h.mix) != c.mix {
            stats.gpu_errors.fetch_add(1, Ordering::Relaxed);
            log!(
                "BUG: the GPU's mix for nonce {:#018x} at height {} differs from rsq-pow's; not submitted",
                c.nonce,
                c.work.height
            );
            continue;
        }
        // The GPU compares the top 64 bits only.
        if !meets(&h.hash, &c.work.target) {
            continue;
        }
        let n = seq.fetch_add(1, Ordering::SeqCst) + 1;
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

/// Decides whether to follow `job`: same or adjacent epoch at once; a far
/// epoch only when a second far job confirms it `confirm` later.
fn follow(ps: &mut PoolState, job: Job, confirm: Duration) -> Option<Job> {
    let e = job.height / EPOCH_LENGTH;
    let Some(cur) = ps.epoch else {
        ps.epoch = Some(e);
        ps.suspect = None;
        return Some(job);
    };
    if e.abs_diff(cur) <= 1 {
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
                "job {} at height {} is in epoch {e}, far from epoch {cur}; waiting for a second job before switching",
                job.id,
                job.height
            );
            ps.suspect = Some((e, Instant::now(), job));
            None
        }
    }
}

/// Mines until `stop` is set. The stats are shared with the caller (API).
pub fn run(cfg: MinerConfig, stop: Arc<AtomicBool>, stats: Arc<Stats>) -> Result<(), String> {
    let gpu = Arc::new(Gpu::new(cfg.device).map_err(|e| e.to_string())?);
    log!(
        "{} (sm_{}{}), {:.2} GiB",
        gpu.name,
        gpu.arch.0,
        gpu.arch.1,
        gib(gpu.memory().map_err(|e| e.to_string())?.1 as u64)
    );
    for (i, p) in cfg.pools.iter().enumerate() {
        log!(
            "pool {i}: {} as {} ({})",
            p.host_port(),
            p.user,
            redact(&p.pass)
        );
    }
    let telemetry = {
        let bus = gpu
            .ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)
            .ok();
        let dev = gpu
            .ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
            .ok();
        let dom = gpu
            .ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)
            .ok();
        let id = match (dom, bus, dev) {
            (Some(a), Some(b), Some(d)) => Some(format!("{a:08x}:{b:02x}:{d:02x}.0")),
            _ => None,
        };
        Telemetry::open(id.as_deref(), cfg.device as u32)
    };
    if telemetry.is_none() {
        log!("NVML unavailable: no temperature or power readings");
    }
    let verifier = Arc::new(Verifier::default());
    let slot = Arc::new(Slot::default());
    let (ev_tx, ev_rx) = channel();
    let client = Client::start(Config::new(cfg.pools.clone()), ev_tx);
    let (cand_tx, cand_rx) = channel();
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
            .map_err(|e| e.to_string())?
    };
    let gpu_handle = {
        let ctx = GpuCtx {
            gpu: gpu.clone(),
            slot: slot.clone(),
            stats: stats.clone(),
            verifier: verifier.clone(),
            candidates: cand_tx,
            stop: stop.clone(),
            cfg: cfg.clone(),
        };
        std::thread::Builder::new()
            .name("gpu".into())
            .spawn(move || gpu_thread(ctx))
            .map_err(|e| e.to_string())?
    };
    let mut ps = PoolState::new();
    let mut last_status = Instant::now();
    let mut last_sample = Instant::now() - Duration::from_secs(10);
    let name = gpu.name.clone();
    while !stop.load(Ordering::SeqCst) {
        if last_sample.elapsed() >= Duration::from_secs(5) {
            last_sample = Instant::now();
            stats.sample();
            if let Some(t) = &telemetry {
                stats.set_device(&name, t.read());
            } else {
                stats.set_device(&name, Default::default());
            }
        }
        if last_status.elapsed() >= cfg.status_every {
            last_status = Instant::now();
            println!("{} {}", now_hms(), stats.line());
        }
        // A far-epoch job nobody contradicted for long enough.
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
            Err(RecvTimeoutError::Disconnected) => break,
        };
        match ev {
            Event::Connected { url, session, .. } => {
                log!("connected to {url}");
                stats.set_pool(&url);
                ps.session = session;
                ps.authorized = false;
                ps.job = None;
                slot.set(None);
            }
            Event::Extranonce(x) => {
                let n = x.len().min(4);
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
            }
            Event::Authorized => {
                log!("authorized");
                ps.authorized = true;
                stats.connected.store(true, Ordering::Relaxed);
                slot.set(ps.work());
            }
            Event::AuthFailed(why) => log!("authorization refused: {why}"),
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
                let e = job.height / EPOCH_LENGTH;
                let want = PARAMS.seed_hash(e * EPOCH_LENGTH + 1);
                if job.seed != want {
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
                stats.share_result(accepted, error.as_deref(), latency);
                if accepted {
                    log!(
                        "share {seq} accepted ({} ms)  A{}",
                        latency.as_millis(),
                        stats.accepted.load(Ordering::Relaxed)
                    );
                } else {
                    log!(
                        "share {seq} not accepted: {}",
                        error.unwrap_or_else(|| "rejected".into())
                    );
                }
            }
            Event::Disconnected { reason } => {
                log!("disconnected: {}", redact(&reason));
                stats.connected.store(false, Ordering::Relaxed);
                ps.authorized = false;
                ps.job = None;
                slot.set(None);
            }
        }
    }
    log!("stopping");
    client.stop();
    slot.set(None);
    let _ = gpu_handle.join();
    let _ = submitter.join();
    println!("{} {}", now_hms(), stats.line());
    Ok(())
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
        assert!(ps.suspect.is_none());
        // Confirmed after the delay.
        assert!(follow(&mut ps, job(100), c).is_none());
        std::thread::sleep(Duration::from_millis(60));
        assert!(follow(&mut ps, job(101), c).is_some());
        assert_eq!(ps.epoch, Some(0));
    }
}
