//! Counters shared by the miner's threads, the status line and the API.

use std::collections::VecDeque;
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

/// Shared counters.
pub struct Stats {
    started: Instant,
    /// Nonces hashed since start.
    pub hashes: AtomicU64,
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
    samples: Mutex<VecDeque<(Instant, u64)>>,
    pool: Mutex<String>,
    job: Mutex<Option<(String, u64)>>,
    difficulty: Mutex<f64>,
    gpu: Mutex<GpuState>,
    last_accept: Mutex<Option<Instant>>,
    last_reject: Mutex<Option<String>>,
    device: Mutex<(String, DeviceReading)>,
    latency_ms: Mutex<VecDeque<u64>>,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            started: Instant::now(),
            hashes: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            stale: AtomicU64::new(0),
            lost: AtomicU64::new(0),
            gpu_errors: AtomicU64::new(0),
            connected: AtomicBool::new(false),
            samples: Mutex::new(VecDeque::new()),
            pool: Mutex::new(String::new()),
            job: Mutex::new(None),
            difficulty: Mutex::new(0.0),
            gpu: Mutex::new(GpuState::Idle),
            last_accept: Mutex::new(None),
            last_reject: Mutex::new(None),
            device: Mutex::new((String::new(), DeviceReading::default())),
            latency_ms: Mutex::new(VecDeque::new()),
        }
    }
}

impl Stats {
    /// Records a hashrate sample (call every few seconds).
    pub fn sample(&self) {
        let mut s = lock(&self.samples);
        let now = Instant::now();
        s.push_back((now, self.hashes.load(Ordering::Relaxed)));
        while s
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > Duration::from_secs(600))
        {
            s.pop_front();
        }
    }

    /// Hashes per second over the last `window`.
    pub fn hashrate(&self, window: Duration) -> f64 {
        let s = lock(&self.samples);
        let Some(&(t1, h1)) = s.back() else {
            return 0.0;
        };
        let Some(&(t0, h0)) = s
            .iter()
            .find(|(t, _)| t1.duration_since(*t) <= window)
            .filter(|(t, _)| *t < t1)
        else {
            return 0.0;
        };
        (h1 - h0) as f64 / t1.duration_since(t0).as_secs_f64()
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

    /// What the GPU is doing.
    pub fn set_gpu(&self, g: GpuState) {
        *lock(&self.gpu) = g;
    }

    /// The device name and its latest reading.
    pub fn set_device(&self, name: &str, r: DeviceReading) {
        *lock(&self.device) = (name.to_string(), r);
    }

    /// A share answer.
    pub fn share_result(&self, accepted: bool, error: Option<&str>, latency: Duration) {
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

    /// One status line.
    pub fn line(&self) -> String {
        let (name, r) = lock(&self.device).clone();
        let _ = name;
        let up = self.uptime().as_secs();
        let gpu = match &*lock(&self.gpu) {
            GpuState::Idle => "idle".to_string(),
            GpuState::BuildingDag(e) => format!("building DAG {e}"),
            GpuState::Mining => "mining".to_string(),
        };
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
            gpu,
            if self.connected.load(Ordering::Relaxed) {
                lock(&self.pool).clone()
            } else {
                "no pool".to_string()
            },
        );
        if let Some(t) = r.temperature_c {
            s.push_str(&format!("  {t}C"));
        }
        if let Some(p) = r.power_w {
            s.push_str(&format!(" {p:.0}W"));
        }
        if let Some(f) = r.fan_pct {
            s.push_str(&format!(" fan {f}%"));
        }
        let ge = self.gpu_errors.load(Ordering::Relaxed);
        if ge > 0 {
            s.push_str(&format!("  GPU-ERRORS {ge}"));
        }
        s
    }

    /// Everything, for the JSON API.
    pub fn json(&self) -> Value {
        let (name, r) = lock(&self.device).clone();
        let lat = lock(&self.latency_ms).clone();
        let avg_lat = if lat.is_empty() {
            Value::Null
        } else {
            json!(lat.iter().sum::<u64>() / lat.len() as u64)
        };
        let gpu = match &*lock(&self.gpu) {
            GpuState::Idle => json!({"state": "idle"}),
            GpuState::BuildingDag(e) => json!({"state": "building_dag", "epoch": e}),
            GpuState::Mining => json!({"state": "mining"}),
        };
        json!({
            "miner": format!("quai-kawpow-miner/{}", env!("CARGO_PKG_VERSION")),
            "uptime_s": self.uptime().as_secs(),
            "connected": self.connected.load(Ordering::Relaxed),
            "pool": *lock(&self.pool),
            "hashrate": {
                "1m": self.hashrate(Duration::from_secs(60)),
                "10m": self.hashrate(Duration::from_secs(600)),
                "total_hashes": self.hashes.load(Ordering::Relaxed),
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
            "gpu": gpu,
            "device": {
                "name": name,
                "temperature_c": r.temperature_c,
                "power_w": r.power_w,
                "power_limit_w": r.power_limit_w,
                "fan_pct": r.fan_pct,
                "mem_clock_mhz": r.mem_clock_mhz,
                "sm_clock_mhz": r.sm_clock_mhz,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_share_answers() {
        let s = Stats::default();
        s.share_result(true, None, Duration::from_millis(15));
        s.share_result(false, Some("nojob"), Duration::ZERO);
        s.share_result(false, Some("no response"), Duration::ZERO);
        s.share_result(false, Some("share difficulty below target"), Duration::ZERO);
        assert_eq!(s.accepted.load(Ordering::Relaxed), 1);
        assert_eq!(s.stale.load(Ordering::Relaxed), 1);
        assert_eq!(s.lost.load(Ordering::Relaxed), 1);
        assert_eq!(s.rejected.load(Ordering::Relaxed), 1);
        assert_eq!(s.json()["shares"]["avg_latency_ms"], 15);
    }
}
