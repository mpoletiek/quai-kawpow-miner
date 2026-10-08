//! Counters shared by the miner's threads, the status line and the API.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What the GPU is doing.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum GpuState {
    /// Waiting for work.
    #[default]
    Idle,
    /// Building the DAG of an epoch.
    BuildingDag(u64),
    /// Hashing.
    Mining,
}

/// Telemetry of one device (NVML).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceReading {
    /// Celsius.
    pub temperature_c: Option<u32>,
    /// Watts.
    pub power_w: Option<f64>,
    /// Power limit in watts.
    pub power_limit_w: Option<f64>,
    /// Fan percent.
    pub fan_pct: Option<u32>,
    /// Memory clock, MHz.
    pub mem_clock_mhz: Option<u32>,
    /// SM clock, MHz.
    pub sm_clock_mhz: Option<u32>,
}

/// Hashes per second over the last `window` of `samples`.
fn rate(samples: &VecDeque<(Instant, u64)>, window: Duration) -> f64 {
    let Some(&(t1, h1)) = samples.back() else {
        return 0.0;
    };
    let Some(&(t0, h0)) = samples
        .iter()
        .find(|(t, _)| t1.duration_since(*t) <= window)
        .filter(|(t, _)| *t < t1)
    else {
        return 0.0;
    };
    (h1 - h0) as f64 / t1.duration_since(t0).as_secs_f64()
}

/// Appends a sample, keeping ten minutes.
fn push_sample(samples: &mut VecDeque<(Instant, u64)>, now: Instant, hashes: u64) {
    samples.push_back((now, hashes));
    while samples
        .front()
        .is_some_and(|(t, _)| now.duration_since(*t) > Duration::from_secs(600))
    {
        samples.pop_front();
    }
}

fn state_name(g: &GpuState) -> String {
    match g {
        GpuState::Idle => "idle".to_string(),
        GpuState::BuildingDag(e) => format!("building DAG {e}"),
        GpuState::Mining => "mining".to_string(),
    }
}

fn state_json(g: &GpuState) -> Value {
    match g {
        GpuState::Idle => json!({"state": "idle"}),
        GpuState::BuildingDag(e) => json!({"state": "building_dag", "epoch": e}),
        GpuState::Mining => json!({"state": "mining"}),
    }
}

/// " 64C 150W fan 33%" (the fields the card reports).
fn reading_text(r: &DeviceReading) -> String {
    let mut s = String::new();
    if let Some(t) = r.temperature_c {
        s.push_str(&format!(" {t}C"));
    }
    if let Some(p) = r.power_w {
        s.push_str(&format!(" {p:.0}W"));
    }
    if let Some(f) = r.fan_pct {
        s.push_str(&format!(" fan {f}%"));
    }
    s
}

/// One GPU's counters.
pub struct GpuStats {
    /// CUDA device index.
    pub device: usize,
    /// Nonces hashed since start.
    pub hashes: AtomicU64,
    /// Its shares accepted.
    pub accepted: AtomicU64,
    /// Its shares not accepted (refused, stale or lost).
    pub refused: AtomicU64,
    samples: Mutex<VecDeque<(Instant, u64)>>,
    state: Mutex<GpuState>,
    reading: Mutex<(String, DeviceReading)>,
}

impl GpuStats {
    fn new(device: usize) -> GpuStats {
        GpuStats {
            device,
            hashes: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            samples: Mutex::new(VecDeque::new()),
            state: Mutex::new(GpuState::Idle),
            reading: Mutex::new((String::new(), DeviceReading::default())),
        }
    }

    /// Hashes per second over the last `window`.
    pub fn hashrate(&self, window: Duration) -> f64 {
        rate(&lock(&self.samples), window)
    }

    /// What the GPU is doing.
    pub fn set_state(&self, g: GpuState) {
        *lock(&self.state) = g;
    }

    /// The device name and its latest reading.
    pub fn set_reading(&self, name: &str, r: DeviceReading) {
        *lock(&self.reading) = (name.to_string(), r);
    }
}

/// Shared counters.
pub struct Stats {
    started: Instant,
    /// Shares accepted.
    pub accepted: AtomicU64,
    /// Shares refused (not stale).
    pub rejected: AtomicU64,
    /// Shares refused as stale (unknown job) or dropped after a reconnect.
    pub stale: AtomicU64,
    /// Shares with no answer in time.
    pub lost: AtomicU64,
    /// GPU solutions the CPU check refused (bugs; never submitted).
    pub gpu_errors: AtomicU64,
    /// Connected and authorized.
    pub connected: AtomicBool,
    gpus: Vec<GpuStats>,
    samples: Mutex<VecDeque<(Instant, u64)>>,
    /// Which GPU found each share in flight (by share sequence number).
    owners: Mutex<HashMap<u64, usize>>,
    pool: Mutex<String>,
    job: Mutex<Option<(String, u64)>>,
    difficulty: Mutex<f64>,
    last_accept: Mutex<Option<Instant>>,
    last_reject: Mutex<Option<String>>,
    latency_ms: Mutex<VecDeque<u64>>,
}

impl Default for Stats {
    fn default() -> Self {
        Stats::new(&[0])
    }
}

impl Stats {
    /// Counters for the GPUs with these CUDA device indexes.
    pub fn new(devices: &[usize]) -> Stats {
        Stats {
            started: Instant::now(),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            stale: AtomicU64::new(0),
            lost: AtomicU64::new(0),
            gpu_errors: AtomicU64::new(0),
            connected: AtomicBool::new(false),
            gpus: devices.iter().map(|&d| GpuStats::new(d)).collect(),
            samples: Mutex::new(VecDeque::new()),
            owners: Mutex::new(HashMap::new()),
            pool: Mutex::new(String::new()),
            job: Mutex::new(None),
            difficulty: Mutex::new(0.0),
            last_accept: Mutex::new(None),
            last_reject: Mutex::new(None),
            latency_ms: Mutex::new(VecDeque::new()),
        }
    }

    /// The GPUs' counters, in the order given to [`Stats::new`].
    pub fn gpus(&self) -> &[GpuStats] {
        &self.gpus
    }

    /// Nonces hashed since start, all GPUs.
    pub fn hashes(&self) -> u64 {
        self.gpus
            .iter()
            .map(|g| g.hashes.load(Ordering::Relaxed))
            .sum()
    }

    /// Records a hashrate sample (call every few seconds).
    pub fn sample(&self) {
        let now = Instant::now();
        for g in &self.gpus {
            push_sample(&mut lock(&g.samples), now, g.hashes.load(Ordering::Relaxed));
        }
        push_sample(&mut lock(&self.samples), now, self.hashes());
    }

    /// Hashes per second over the last `window`, all GPUs.
    pub fn hashrate(&self, window: Duration) -> f64 {
        rate(&lock(&self.samples), window)
    }

    /// Seconds since start.
    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    /// The pool in use (`host:port`, empty when none).
    pub fn set_pool(&self, p: &str) {
        *lock(&self.pool) = p.to_string();
    }

    /// The current job.
    pub fn set_job(&self, id: &str, height: u64) {
        *lock(&self.job) = Some((id.to_string(), height));
    }

    /// The current stratum difficulty.
    pub fn set_difficulty(&self, d: f64) {
        *lock(&self.difficulty) = d;
    }

    /// Share `seq` was found by GPU `gpu` (an index into [`Stats::gpus`]).
    pub fn share_sent(&self, seq: u64, gpu: usize) {
        lock(&self.owners).insert(seq, gpu);
    }

    /// The answer to share `seq`.
    pub fn share_result(&self, seq: u64, accepted: bool, error: Option<&str>, latency: Duration) {
        if let Some(g) = lock(&self.owners)
            .remove(&seq)
            .and_then(|i| self.gpus.get(i))
        {
            let c = if accepted { &g.accepted } else { &g.refused };
            c.fetch_add(1, Ordering::Relaxed);
        }
        if accepted {
            self.accepted.fetch_add(1, Ordering::Relaxed);
            *lock(&self.last_accept) = Some(Instant::now());
            let mut l = lock(&self.latency_ms);
            l.push_back(latency.as_millis() as u64);
            if l.len() > 100 {
                l.pop_front();
            }
            return;
        }
        let e = error.unwrap_or("rejected");
        let counter = match e {
            "nojob" | "stale" | "stale session" | "not connected" | "not authorized yet" => {
                &self.stale
            }
            "no response" | "connection closed before an answer" => &self.lost,
            _ => &self.rejected,
        };
        counter.fetch_add(1, Ordering::Relaxed);
        *lock(&self.last_reject) = Some(e.to_string());
    }

    /// The GPUs' states: "mining" when all mine, else e.g. "mining 6/7,
    /// building DAG 609".
    fn states(&self) -> String {
        let states: Vec<GpuState> = self.gpus.iter().map(|g| lock(&g.state).clone()).collect();
        let mining = states.iter().filter(|s| **s == GpuState::Mining).count();
        if mining == states.len() {
            return "mining".to_string();
        }
        let mut s = if states.len() > 1 {
            format!("mining {mining}/{}", states.len())
        } else {
            String::new()
        };
        let mut others: Vec<String> = states
            .iter()
            .filter(|s| **s != GpuState::Mining)
            .map(state_name)
            .collect();
        others.dedup();
        for o in others {
            if !s.is_empty() {
                s.push_str(", ");
            }
            s.push_str(&o);
        }
        s
    }

    /// The status line (with the card's readings when there is one GPU).
    pub fn line(&self) -> String {
        let up = self.uptime().as_secs();
        let job = lock(&self.job)
            .clone()
            .map_or_else(|| "-".to_string(), |(_, h)| format!("h{h}"));
        let mut s = format!(
            "{:>3}h{:02}m  {:6.2} MH/s (10 min {:.2})  A{} R{} S{} L{}  diff {:.3}  {}  {}  {}",
            up / 3600,
            up / 60 % 60,
            self.hashrate(Duration::from_secs(60)) / 1e6,
            self.hashrate(Duration::from_secs(600)) / 1e6,
            self.accepted.load(Ordering::Relaxed),
            self.rejected.load(Ordering::Relaxed),
            self.stale.load(Ordering::Relaxed),
            self.lost.load(Ordering::Relaxed),
            *lock(&self.difficulty),
            job,
            self.states(),
            if self.connected.load(Ordering::Relaxed) {
                lock(&self.pool).clone()
            } else {
                "no pool".to_string()
            },
        );
        if let [g] = self.gpus.as_slice() {
            s.push(' ');
            s.push_str(&reading_text(&lock(&g.reading).1));
        }
        let ge = self.gpu_errors.load(Ordering::Relaxed);
        if ge > 0 {
            s.push_str(&format!("  GPU-ERRORS {ge}"));
        }
        s
    }

    /// One line per GPU (none with a single GPU: [`Stats::line`] has it).
    pub fn gpu_lines(&self) -> Vec<String> {
        if self.gpus.len() < 2 {
            return Vec::new();
        }
        self.gpus
            .iter()
            .map(|g| {
                let st = lock(&g.state).clone();
                format!(
                    "  gpu{:<2} {:6.2} MH/s (10 min {:.2})  A{} R{} {}{}",
                    g.device,
                    g.hashrate(Duration::from_secs(60)) / 1e6,
                    g.hashrate(Duration::from_secs(600)) / 1e6,
                    g.accepted.load(Ordering::Relaxed),
                    g.refused.load(Ordering::Relaxed),
                    if st == GpuState::Mining {
                        String::new()
                    } else {
                        format!("{} ", state_name(&st))
                    },
                    reading_text(&lock(&g.reading).1),
                )
            })
            .collect()
    }

    /// Everything, for the JSON API.
    pub fn json(&self) -> Value {
        let lat = lock(&self.latency_ms).clone();
        let avg_lat = if lat.is_empty() {
            Value::Null
        } else {
            json!(lat.iter().sum::<u64>() / lat.len() as u64)
        };
        let gpus: Vec<Value> = self
            .gpus
            .iter()
            .map(|g| {
                let (name, r) = lock(&g.reading).clone();
                json!({
                    "device": g.device,
                    "name": name,
                    "state": state_json(&lock(&g.state)),
                    "hashrate": {
                        "1m": g.hashrate(Duration::from_secs(60)),
                        "10m": g.hashrate(Duration::from_secs(600)),
                        "total_hashes": g.hashes.load(Ordering::Relaxed),
                    },
                    "shares": {
                        "accepted": g.accepted.load(Ordering::Relaxed),
                        "refused": g.refused.load(Ordering::Relaxed),
                    },
                    "temperature_c": r.temperature_c,
                    "power_w": r.power_w,
                    "power_limit_w": r.power_limit_w,
                    "fan_pct": r.fan_pct,
                    "mem_clock_mhz": r.mem_clock_mhz,
                    "sm_clock_mhz": r.sm_clock_mhz,
                })
            })
            .collect();
        json!({
            "miner": format!("quai-kawpow-miner/{}", env!("CARGO_PKG_VERSION")),
            "uptime_s": self.uptime().as_secs(),
            "connected": self.connected.load(Ordering::Relaxed),
            "pool": *lock(&self.pool),
            "hashrate": {
                "1m": self.hashrate(Duration::from_secs(60)),
                "10m": self.hashrate(Duration::from_secs(600)),
                "total_hashes": self.hashes(),
            },
            "shares": {
                "accepted": self.accepted.load(Ordering::Relaxed),
                "rejected": self.rejected.load(Ordering::Relaxed),
                "stale": self.stale.load(Ordering::Relaxed),
                "lost": self.lost.load(Ordering::Relaxed),
                "gpu_errors": self.gpu_errors.load(Ordering::Relaxed),
                "last_accept_s": lock(&self.last_accept).map(|t| t.elapsed().as_secs()),
                "last_reject": *lock(&self.last_reject),
                "avg_latency_ms": avg_lat,
            },
            "difficulty": *lock(&self.difficulty),
            "job": lock(&self.job).clone().map(|(id, h)| json!({"id": id, "height": h, "epoch": h / 7500})),
            "gpus": gpus,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_share_answers() {
        let s = Stats::new(&[0, 3]);
        s.share_sent(1, 1);
        s.share_sent(2, 1);
        s.share_sent(3, 0);
        s.share_result(1, true, None, Duration::from_millis(15));
        s.share_result(2, false, Some("nojob"), Duration::ZERO);
        s.share_result(3, false, Some("no response"), Duration::ZERO);
        s.share_result(4, false, Some("share difficulty below target"), Duration::ZERO);
        assert_eq!(s.accepted.load(Ordering::Relaxed), 1);
        assert_eq!(s.stale.load(Ordering::Relaxed), 1);
        assert_eq!(s.lost.load(Ordering::Relaxed), 1);
        assert_eq!(s.rejected.load(Ordering::Relaxed), 1);
        assert_eq!(s.json()["shares"]["avg_latency_ms"], 15);
        // Per GPU: the second GPU (device 3) found shares 1 and 2.
        let g = &s.json()["gpus"][1];
        assert_eq!(g["device"], 3);
        assert_eq!(g["shares"]["accepted"], 1);
        assert_eq!(g["shares"]["refused"], 1);
        assert_eq!(s.gpus()[0].refused.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn summarizes_gpu_states() {
        let s = Stats::new(&[0, 1, 2]);
        for g in s.gpus() {
            g.set_state(GpuState::Mining);
        }
        assert_eq!(s.states(), "mining");
        s.gpus()[2].set_state(GpuState::BuildingDag(609));
        assert_eq!(s.states(), "mining 2/3, building DAG 609");
        assert_eq!(s.gpu_lines().len(), 3);
        assert!(Stats::new(&[0]).gpu_lines().is_empty());
    }
}
