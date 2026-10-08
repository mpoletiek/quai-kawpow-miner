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
use qkm_cuda::{Cubins, Dag, Geometry, Gpu, Header, Kernels, Searcher};
use qkm_stratum::{
    Client, Config, Event, Job, Pool, Share, Submitter, difficulty_of_target, redact,
};

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
    gpu: Arc<Gpu>,
    cubins: Arc<Cubins>,
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
    let dev = c.stats.gpus()[c.index].device;
    while !c.stop.load(Ordering::SeqCst) {
        match gpu_loop(&c) {
            Ok(()) => return,
            Err(e) => {
                failures += 1;
                log!("gpu{dev}: GPU error ({failures}): {e}; retrying");
                c.stats.gpus()[c.index].set_state(GpuState::Idle);
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
    let st = &c.stats.gpus()[c.index];
    let dev = st.device;
    let e = |e: qkm_cuda::Error| e.to_string();
    let mut kernels = Kernels::shared(gpu, c.cubins.clone());
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
            st.set_state(GpuState::Idle);
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
                    Ok(Err(err)) => log!("gpu{dev}: epoch {n}: DAG built ahead failed: {err}"),
                    Err(_) => log!("gpu{dev}: epoch {n}: the DAG prefetch thread panicked"),
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
            let until = Instant::now() + c.cfg.start_stagger * c.index as u32;
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
            if free as u64 > next.dataset_bytes + (512 << 20) {
                let (g, v) = (gpu.clone(), c.verifier.clone());
                log!("gpu{dev}: epoch {}: building the next DAG ahead", next.number);
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
        st.set_state(GpuState::Mining);
        let (prefix, mask) = nonce_range(w.prefix, w.free_mask, c.index, c.count);
        let key = (w.job.clone(), w.session);
        if current.as_ref() != Some(&key) {
            current = Some(key);
            // A random start in this GPU's free bits, leaving room to count up.
            nonce = prefix | (rng.next() & (mask >> 1));
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
                    st.hashes.fetch_add(done - counted, Ordering::Relaxed);
                    counted = done;
                    !c.stop.load(Ordering::SeqCst)
                        && c.slot.generation() == generation
                        && found.is_empty()
                        && t0.elapsed() < Duration::from_secs(2)
                },
            )
            .map_err(e)?;
        nonce = prefix | (nonce.wrapping_add(done) & mask);
        for f in found {
            let _ = c.candidates.send(Candidate {
                gpu: c.index,
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
                "BUG: gpu{}'s mix for nonce {:#018x} at height {} differs from rsq-pow's; not submitted",
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
    let listed: Vec<usize> = stats.gpus().iter().map(|g| g.device).collect();
    if listed != cfg.devices {
        return Err(format!(
            "the stats are for devices {listed:?} but the miner is for {:?}",
            cfg.devices
        ));
    }
    // Settings first, so no GPU hashes before its clocks are set.
    if !cfg.tune.is_empty() {
        apply_tuning(&cfg.devices, &cfg.tune);
    }
    // Open every GPU; one that fails is left out (and shows as idle).
    let mut gpus = Vec::new();
    for (i, &d) in cfg.devices.iter().enumerate() {
        match Gpu::new(d) {
            Ok(gpu) => {
                let gpu = Arc::new(gpu);
                log!(
                    "gpu{d}: {} (sm_{}{}), {:.2} GiB",
                    gpu.name,
                    gpu.arch.0,
                    gpu.arch.1,
                    gib(gpu.memory().map_err(|e| e.to_string())?.1 as u64)
                );
                let telemetry = qkm_cuda::pci_bus_id(d)
                    .ok()
                    .and_then(|id| Telemetry::open(Some(&id), d as u32));
                if telemetry.is_none() {
                    log!("gpu{d}: NVML unavailable: no temperature or power readings");
                }
                gpus.push((i, gpu, telemetry));
            }
            Err(e) => log!("gpu{d}: cannot open: {e}; mining without it"),
        }
    }
    if gpus.is_empty() {
        return Err(format!("no usable GPU among devices {:?}", cfg.devices));
    }
    for (i, p) in cfg.pools.iter().enumerate() {
        log!(
            "pool {i}: {} as {} ({})",
            p.host_port(),
            p.user,
            redact(&p.pass)
        );
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
    let cubins = Arc::new(Cubins::new(cfg.search));
    let mut gpu_handles = Vec::new();
    for (i, gpu, _) in &gpus {
        let ctx = GpuCtx {
            index: *i,
            count: cfg.devices.len(),
            gpu: gpu.clone(),
            cubins: cubins.clone(),
            slot: slot.clone(),
            stats: stats.clone(),
            verifier: verifier.clone(),
            candidates: cand_tx.clone(),
            stop: stop.clone(),
            cfg: cfg.clone(),
        };
        gpu_handles.push(
            std::thread::Builder::new()
                .name(format!("gpu{}", cfg.devices[*i]))
                .spawn(move || gpu_thread(ctx))
                .map_err(|e| e.to_string())?,
        );
    }
    // The submit thread ends when the last GPU thread drops its sender.
    drop(cand_tx);
    let mut ps = PoolState::new();
    let mut last_status = Instant::now();
    let mut last_sample = Instant::now() - Duration::from_secs(10);
    while !stop.load(Ordering::SeqCst) {
        if last_sample.elapsed() >= Duration::from_secs(5) {
            last_sample = Instant::now();
            stats.sample();
            for (i, gpu, t) in &gpus {
                let r = t.as_ref().map(Telemetry::read).unwrap_or_default();
                stats.gpus()[*i].set_reading(&gpu.name, r);
            }
        }
        if last_status.elapsed() >= cfg.status_every {
            last_status = Instant::now();
            print_status(&stats);
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
                stats.share_result(seq, accepted, error.as_deref(), latency);
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
    for h in gpu_handles {
        let _ = h.join();
    }
    let _ = submitter.join();
    print_status(&stats);
    Ok(())
}

/// The status line, then one line per GPU when there are several.
fn print_status(stats: &Stats) {
    let mut out = format!("{} {}", now_hms(), stats.line());
    for l in stats.gpu_lines() {
        out.push('\n');
        out.push_str(&l);
    }
    println!("{out}");
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
        log!(
            "tuning: not root, GPU settings left as they are (apply them with `sudo quai-kawpow-miner tune`)"
        );
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
        assert!(ps.suspect.is_none());
        // Confirmed after the delay.
        assert!(follow(&mut ps, job(100), c).is_none());
        std::thread::sleep(Duration::from_millis(60));
        assert!(follow(&mut ps, job(101), c).is_some());
        assert_eq!(ps.epoch, Some(0));
    }
}
