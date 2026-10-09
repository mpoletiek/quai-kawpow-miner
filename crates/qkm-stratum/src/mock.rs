//! A mock Quai KawPoW stratum pool for tests: it speaks the same lines as
//! rs-quai's `rsq-stratum` and checks every share the way that server's
//! `submit_kawpow_share` does (header reversed into "natural" order and back,
//! mix compared byte-reversed, final hash read big-endian against the
//! target, duplicates and unknown jobs refused).

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use qkm_core::Verifier;
use qkm_core::qkm_kawpow::kawpow::PARAMS;
use serde_json::{Value, json};

use crate::proto::hex0x;

/// A job the mock handed out.
#[derive(Clone, Debug)]
pub struct MockJob {
    /// Its id.
    pub id: String,
    /// Header hash.
    pub header: [u8; 32],
    /// Donor height.
    pub height: u64,
}

/// What the mock saw.
#[derive(Clone, Debug, Default)]
pub struct Tally {
    /// Shares accepted.
    pub accepted: u64,
    /// Shares refused, by reason.
    pub rejected: HashMap<String, u64>,
    /// Connections opened.
    pub connections: u64,
    /// Workers that authorized (`user` as sent).
    pub workers: Vec<String>,
}

struct Shared {
    target: [u8; 32],
    token: Option<String>,
    jobs: HashMap<String, MockJob>,
    latest: Option<MockJob>,
    seen: HashSet<[u8; 32]>,
    tally: Tally,
    sessions: Vec<Arc<Mutex<TcpStream>>>,
    /// Delay before answering a submit (to exercise client timeouts).
    answer_delay: Duration,
}

/// A running mock pool.
pub struct MockPool {
    addr: SocketAddr,
    shared: Arc<Mutex<Shared>>,
    verifier: Arc<Verifier>,
    next_job: AtomicU64,
    stop: Arc<AtomicBool>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn send_line(s: &Mutex<TcpStream>, line: &str) -> bool {
    let mut g = lock(s);
    g.write_all(line.as_bytes())
        .and_then(|()| g.write_all(b"\n"))
        .is_ok()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    crate::proto::unhex(s).ok()
}

/// rs-quai's checks at `mining.authorize`: the payout address must be in
/// the node's zone (cyprus1 here: first byte 0x00), `lock` must be 0..3,
/// and the token must match when one is required. Options are separated
/// by `,` or `_`.
fn authorize_error(user: &str, pass: &str, token: Option<&str>) -> Option<&'static str> {
    let address = user.split('.').next().unwrap_or("");
    let zone_ok = unhex(address).is_some_and(|b| b.first() == Some(&0x00));
    if !zone_ok {
        return Some("address is not internal to this zone");
    }
    let opts: Vec<&str> = pass.split([',', '_']).collect();
    for o in &opts {
        if let Some(l) = o.strip_prefix("lock=")
            && !l.parse::<u8>().is_ok_and(|v| v <= 3)
        {
            return Some("invalid lock value");
        }
    }
    if let Some(t) = token
        && !opts.iter().any(|o| o.strip_prefix("auth=") == Some(t))
    {
        return Some("unauthorized");
    }
    None
}

impl MockPool {
    /// Listens on 127.0.0.1 (any port). `token`: require `auth=<token>`.
    pub fn start(target: [u8; 32], token: Option<&str>) -> MockPool {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let shared = Arc::new(Mutex::new(Shared {
            target,
            token: token.map(str::to_string),
            jobs: HashMap::new(),
            latest: None,
            seen: HashSet::new(),
            tally: Tally::default(),
            sessions: Vec::new(),
            answer_delay: Duration::ZERO,
        }));
        let verifier = Arc::new(Verifier::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (sh, v, st) = (shared.clone(), verifier.clone(), stop.clone());
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                if st.load(Ordering::SeqCst) {
                    return;
                }
                let Ok(conn) = conn else { continue };
                let (sh, v) = (sh.clone(), v.clone());
                std::thread::spawn(move || serve(conn, &sh, &v));
            }
        });
        MockPool {
            addr,
            shared,
            verifier,
            next_job: AtomicU64::new(1),
            stop,
        }
    }

    /// `host:port`.
    pub fn addr(&self) -> String {
        self.addr.to_string()
    }

    /// The verifier (its light caches), shared with tests that mine on the CPU.
    pub fn verifier(&self) -> Arc<Verifier> {
        self.verifier.clone()
    }

    /// Hands out a new job at `height` to every session (and to later ones).
    pub fn push_job(&self, height: u64, clean: bool) -> MockJob {
        let n = self.next_job.fetch_add(1, Ordering::SeqCst);
        let mut header = [0u8; 32];
        let mut x = n.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ height;
        for b in &mut header {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = x as u8;
        }
        let job = MockJob {
            id: format!("job{n:04x}"),
            header,
            height,
        };
        let mut g = lock(&self.shared);
        g.jobs.insert(job.id.clone(), job.clone());
        g.latest = Some(job.clone());
        let line = notify_line(&job, &g.target, clean);
        g.sessions.retain(|s| send_line(s, &line));
        job
    }

    /// Changes the target and tells every session.
    pub fn set_target(&self, target: [u8; 32]) {
        let mut g = lock(&self.shared);
        g.target = target;
        let line =
            json!({"id": null, "method": "mining.set_target", "params": [hex0x(&target)[2..]]})
                .to_string();
        g.sessions.retain(|s| send_line(s, &line));
    }

    /// Sends `mining.set_difficulty` to every session.
    pub fn set_difficulty(&self, d: f64) {
        let mut g = lock(&self.shared);
        let line =
            json!({"id": null, "method": "mining.set_difficulty", "params": [d]}).to_string();
        g.sessions.retain(|s| send_line(s, &line));
    }

    /// Answers submits only after `d` (0 = at once).
    pub fn set_answer_delay(&self, d: Duration) {
        lock(&self.shared).answer_delay = d;
    }

    /// Closes every connection.
    pub fn drop_connections(&self) {
        let mut g = lock(&self.shared);
        for s in g.sessions.drain(..) {
            let _ = lock(&s).shutdown(Shutdown::Both);
        }
    }

    /// What happened so far.
    pub fn tally(&self) -> Tally {
        lock(&self.shared).tally.clone()
    }

    /// Stops accepting connections and closes the open ones.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.drop_connections();
        let _ = TcpStream::connect(self.addr);
    }
}

impl Drop for MockPool {
    fn drop(&mut self) {
        self.stop();
    }
}

fn notify_line(job: &MockJob, target: &[u8; 32], clean: bool) -> String {
    let epoch = job.height / 7500;
    let seed = PARAMS.seed_hash(epoch * 7500 + 1);
    json!({"id": null, "method": "mining.notify", "params": [
        job.id, hex0x(&job.header)[2..], hex0x(&seed)[2..], hex0x(target)[2..], clean, job.height, "1b068045"
    ]})
    .to_string()
}

fn respond(s: &Mutex<TcpStream>, id: &Value, result: Value, error: Value) -> bool {
    send_line(
        s,
        &format!("{{\"id\":{id},\"result\":{result},\"error\":{error}}}"),
    )
}

/// rs-quai's check, step by step (`Server::submit_kawpow_share`).
fn check(
    sh: &Mutex<Shared>,
    v: &Verifier,
    job_id: &str,
    nonce_hex: &str,
    mix_hex: &str,
) -> Result<(), String> {
    let (job, target) = {
        let g = lock(sh);
        (g.jobs.get(job_id).cloned(), g.target)
    };
    let Some(job) = job else {
        return Err("nojob".into());
    };
    let nonce = u64::from_str_radix(nonce_hex.trim_start_matches("0x"), 16)
        .map_err(|e| format!("invalid nonce: {e}"))?;
    let mix = unhex(mix_hex).unwrap_or_default();
    if mix.len() != 32 {
        return Err(format!("invalid mix hash: length={}", mix.len()));
    }
    // natural = reverse(job header); compute_light reverses it again.
    let mut natural = job.header;
    natural.reverse();
    let mut h = natural;
    h.reverse();
    let hashes = v.hash(job.height, &h, nonce);
    let mut calc_mix = hashes.mix;
    calc_mix.reverse();
    let mut mix_le = [0u8; 32];
    mix_le.copy_from_slice(&mix);
    mix_le.reverse();
    if calc_mix != mix_le {
        return Err("mixhash mismatch".into());
    }
    if hashes.hash > target {
        return Err("share difficulty below target".into());
    }
    let mut g = lock(sh);
    if !g.seen.insert(hashes.hash) {
        return Err("duplicate share".into());
    }
    Ok(())
}

fn serve(conn: TcpStream, sh: &Arc<Mutex<Shared>>, v: &Verifier) {
    let Ok(read) = conn.try_clone() else { return };
    let out = Arc::new(Mutex::new(conn));
    lock(sh).tally.connections += 1;
    let mut r = BufReader::new(read);
    let mut line = String::new();
    loop {
        line.clear();
        match r.read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let Ok(req) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let params = req
            .get("params")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let p = |i: usize| {
            params
                .get(i)
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        match req.get("method").and_then(Value::as_str).unwrap_or("") {
            "mining.subscribe" => {
                respond(&out, &id, json!([null, "c2"]), Value::Null);
            }
            "mining.authorize" => {
                let token = lock(sh).token.clone();
                if let Some(e) = authorize_error(&p(0), &p(1), token.as_deref()) {
                    respond(&out, &id, Value::Bool(false), json!(e));
                    if e == "unauthorized" {
                        return;
                    }
                    continue;
                }
                respond(&out, &id, Value::Bool(true), Value::Null);
                let mut g = lock(sh);
                g.tally.workers.push(p(0));
                let set = json!({"id": null, "method": "mining.set_target", "params": [hex0x(&g.target)[2..]]}).to_string();
                send_line(&out, &set);
                if let Some(j) = g.latest.clone() {
                    send_line(&out, &notify_line(&j, &g.target, true));
                }
                g.sessions.push(out.clone());
            }
            "mining.submit" => {
                let delay = lock(sh).answer_delay;
                if !delay.is_zero() {
                    std::thread::sleep(delay);
                }
                match check(sh, v, &p(1), &p(2), &p(4)) {
                    Ok(()) => {
                        lock(sh).tally.accepted += 1;
                        respond(&out, &id, Value::Bool(true), Value::Null);
                    }
                    Err(e) => {
                        *lock(sh).tally.rejected.entry(e.clone()).or_default() += 1;
                        respond(&out, &id, Value::Bool(false), json!(e));
                    }
                }
            }
            _ => {
                respond(&out, &id, Value::Null, Value::Null);
            }
        }
    }
}
