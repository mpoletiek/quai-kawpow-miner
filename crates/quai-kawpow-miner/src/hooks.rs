//! Event hooks: tell the operator when something happens, by running a
//! program and/or posting to a webhook (Discord, Slack, ntfy, your own).
//!
//! Hooks never slow mining. Events go through a bounded queue to one
//! delivery thread, and each delivery has a deadline. A burst of one kind
//! of event (across the rig: seven GPUs pausing for heat is one message,
//! not seven) is folded: the first goes out at once, and the latest of the
//! rest goes out when [`HooksConfig::interval`] has passed, saying how many
//! it stands for and on which GPUs. So the last word on anything is never
//! lost, and what was held back goes out before `miner_stopping`.
//!
//! Nothing secret leaves: the auth token is not passed on, pool text
//! arrives redacted and is stripped of control characters, and the webhook
//! URL (which often holds a secret) is logged as scheme and host only. The
//! program runs directly, never through a shell.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::json;

use crate::stats::Stats;

/// What happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// The miner started (GPUs open, connecting).
    MinerStarted,
    /// The miner is stopping (on request, or why it must).
    MinerStopping,
    /// Connected to a pool.
    PoolConnected,
    /// A pool connection ended.
    PoolDisconnected,
    /// Moved to a backup pool.
    PoolFailover,
    /// Back on the first pool after a backup.
    PoolPrimaryBack,
    /// No usable pool (connected and authorized) for a while.
    PoolDown,
    /// A pool is usable again after [`Kind::PoolDown`].
    PoolUp,
    /// A pool refused the login (address, lock or token).
    AuthRefused,
    /// A share was not accepted.
    ShareRejected,
    /// A GPU error (retried).
    GpuError,
    /// A GPU's context was reset after repeated errors.
    GpuReset,
    /// A GPU was given up.
    GpuFailed,
    /// A GPU reached the temperature limit and paused.
    GpuTooHot,
    /// A paused GPU cooled down and mines again.
    GpuCooled,
    /// The hashrate stayed below the threshold.
    HashrateLow,
    /// The hashrate is back above the threshold.
    HashrateOk,
}

impl Kind {
    /// Every kind.
    pub const ALL: [Kind; 17] = [
        Kind::MinerStarted,
        Kind::MinerStopping,
        Kind::PoolConnected,
        Kind::PoolDisconnected,
        Kind::PoolFailover,
        Kind::PoolPrimaryBack,
        Kind::PoolDown,
        Kind::PoolUp,
        Kind::AuthRefused,
        Kind::ShareRejected,
        Kind::GpuError,
        Kind::GpuReset,
        Kind::GpuFailed,
        Kind::GpuTooHot,
        Kind::GpuCooled,
        Kind::HashrateLow,
        Kind::HashrateOk,
    ];

    /// The name used in the config, the payload and `QKM_EVENT`.
    pub fn name(self) -> &'static str {
        match self {
            Kind::MinerStarted => "miner_started",
            Kind::MinerStopping => "miner_stopping",
            Kind::PoolConnected => "pool_connected",
            Kind::PoolDisconnected => "pool_disconnected",
            Kind::PoolFailover => "pool_failover",
            Kind::PoolPrimaryBack => "pool_primary_back",
            Kind::PoolDown => "pool_down",
            Kind::PoolUp => "pool_up",
            Kind::AuthRefused => "auth_refused",
            Kind::ShareRejected => "share_rejected",
            Kind::GpuError => "gpu_error",
            Kind::GpuReset => "gpu_reset",
            Kind::GpuFailed => "gpu_failed",
            Kind::GpuTooHot => "gpu_too_hot",
            Kind::GpuCooled => "gpu_cooled",
            Kind::HashrateLow => "hashrate_low",
            Kind::HashrateOk => "hashrate_ok",
        }
    }

    /// The kind with this name.
    pub fn parse(name: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.name() == name)
    }

    /// Sent when no `events` list is given: the problems (and their end),
    /// not the routine.
    pub fn default_on(self) -> bool {
        !matches!(
            self,
            Kind::MinerStarted
                | Kind::PoolConnected
                | Kind::PoolDisconnected
                | Kind::ShareRejected
                | Kind::GpuError
                | Kind::GpuCooled
        )
    }
}

/// Reads an `events` list: names, or `all`, or `default` (plus more).
pub fn parse_events(names: &[String]) -> Result<Vec<Kind>, String> {
    let mut out: Vec<Kind> = Vec::new();
    for n in names {
        let n = n.trim();
        let add: Vec<Kind> = match n {
            "all" => Kind::ALL.to_vec(),
            "default" => Kind::ALL.into_iter().filter(|k| k.default_on()).collect(),
            _ => vec![Kind::parse(n).ok_or_else(|| {
                let all: Vec<&str> = Kind::ALL.iter().map(|k| k.name()).collect();
                format!(
                    "unknown hook event {n:?}; the events are all, default, {}",
                    all.join(", ")
                )
            })?],
        };
        for k in add {
            if !out.contains(&k) {
                out.push(k);
            }
        }
    }
    Ok(out)
}

/// The payload's format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Format {
    /// A JSON object with the details, plus `text` (Slack) and `content`
    /// (Discord) holding the message.
    #[default]
    Json,
    /// The message as plain text (ntfy).
    Text,
}

/// A webhook address. Its path often holds a secret, so it is never shown.
#[derive(Clone, PartialEq, Eq)]
pub struct Webhook {
    tls: bool,
    host: String,
    port: u16,
    path: String,
}

impl std::fmt::Debug for Webhook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.shown())
    }
}

impl Webhook {
    /// Reads `http://host[:port]/path` or `https://...`.
    pub fn parse(url: &str) -> Result<Webhook, String> {
        let url = url.trim();
        let bad = || "the webhook must be an http:// or https:// URL".to_string();
        let (scheme, rest) = url.split_once("://").ok_or_else(bad)?;
        let tls = match scheme.to_ascii_lowercase().as_str() {
            "https" => true,
            "http" => false,
            _ => return Err(bad()),
        };
        let (authority, path) = match rest.find(['/', '?']) {
            Some(i) if rest.as_bytes()[i] == b'?' => (&rest[..i], format!("/{}", &rest[i..])),
            Some(i) => (&rest[..i], rest[i..].to_string()),
            None => (rest, "/".to_string()),
        };
        if authority.contains('@') {
            return Err("the webhook URL must not hold a user name or password".into());
        }
        let default_port = if tls { 443 } else { 80 };
        let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
            let (h, after) = v6.split_once(']').ok_or_else(bad)?;
            let port = match after.strip_prefix(':') {
                Some(p) => p.parse().map_err(|_| bad())?,
                None if after.is_empty() => default_port,
                None => return Err(bad()),
            };
            (h.to_string(), port)
        } else {
            match authority.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), p.parse().map_err(|_| bad())?),
                None => (authority.to_string(), default_port),
            }
        };
        if host.is_empty() || path.chars().any(|c| c.is_control() || c == ' ') {
            return Err(bad());
        }
        Ok(Webhook {
            tls,
            host,
            port,
            path,
        })
    }

    /// `https://host/…`: enough to recognise it, without the secret.
    pub fn shown(&self) -> String {
        format!(
            "{}://{}/…",
            if self.tls { "https" } else { "http" },
            self.host
        )
    }

    fn authority(&self) -> String {
        let h = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        if self.port == if self.tls { 443 } else { 80 } {
            h
        } else {
            format!("{h}:{}", self.port)
        }
    }
}

/// When the hashrate counts as low.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowRate {
    /// Hashes per second.
    pub below: f64,
    /// For this long.
    pub after: Duration,
}

/// Hook settings.
#[derive(Clone, Debug)]
pub struct HooksConfig {
    /// A program to run per event (`$1` and `QKM_EVENT` name the event).
    pub command: Option<PathBuf>,
    /// An address to POST each event to.
    pub webhook: Option<Webhook>,
    /// The webhook payload's format.
    pub format: Format,
    /// The events sent (`None`: [`Kind::default_on`]).
    pub events: Option<Vec<Kind>>,
    /// [`Kind::HashrateLow`] and [`Kind::HashrateOk`] (off when `None`).
    pub hashrate_low: Option<LowRate>,
    /// [`Kind::PoolDown`] after this long without a usable pool.
    pub pool_down_after: Duration,
    /// Events of one kind closer than this are folded.
    pub interval: Duration,
    /// Each delivery's deadline (the program is killed after it).
    pub timeout: Duration,
}

impl Default for HooksConfig {
    fn default() -> HooksConfig {
        HooksConfig {
            command: None,
            webhook: None,
            format: Format::Json,
            events: None,
            hashrate_low: None,
            pool_down_after: Duration::from_secs(120),
            interval: Duration::from_secs(60),
            timeout: Duration::from_secs(10),
        }
    }
}

impl HooksConfig {
    /// No program and no webhook: nothing to deliver to.
    pub fn is_off(&self) -> bool {
        self.command.is_none() && self.webhook.is_none()
    }

    /// Whether `kind` is sent.
    pub fn wants(&self, kind: Kind) -> bool {
        match &self.events {
            Some(list) => list.contains(&kind),
            None => kind.default_on(),
        }
    }

    /// One line for the log: what goes where.
    pub fn describe(&self) -> String {
        let mut to = Vec::new();
        if let Some(c) = &self.command {
            to.push(format!("run {}", c.display()));
        }
        if let Some(w) = &self.webhook {
            to.push(format!("post to {}", w.shown()));
        }
        let events: Vec<&str> = Kind::ALL
            .into_iter()
            .filter(|k| self.wants(*k))
            .map(Kind::name)
            .collect();
        format!("hooks: {} on {}", to.join(" and "), events.join(", "))
    }
}

/// One event.
#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    /// What happened.
    pub kind: Kind,
    /// The CUDA device it concerns, if one.
    pub gpu: Option<usize>,
    /// The details, as a phrase (`"91C reached the 90C limit; paused"`).
    pub detail: String,
}

impl Note {
    /// A rig-wide event.
    pub fn new(kind: Kind, detail: impl Into<String>) -> Note {
        Note {
            kind,
            gpu: None,
            detail: detail.into(),
        }
    }

    /// An event about one GPU.
    pub fn gpu(kind: Kind, device: usize, detail: impl Into<String>) -> Note {
        Note {
            kind,
            gpu: Some(device),
            detail: detail.into(),
        }
    }
}

/// Printable text only, at most `max` characters.
fn clean(s: &str, max: usize) -> String {
    s.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .take(max)
        .collect()
}

/// The worker name in `0xADDRESS.worker` (or the whole thing).
fn worker_name(user: &str) -> &str {
    user.split_once('.').map_or(user, |(_, w)| w)
}

/// What one delivery carries.
#[derive(Clone, Debug)]
struct Delivery {
    note: Note,
    /// Earlier events of the same kind folded into this one.
    folded: u32,
    /// Every GPU the folded events concern, this one's included.
    gpus: Vec<usize>,
    hashrate_mhs: f64,
    time: u64,
}

impl Delivery {
    /// A delivery of `note` alone.
    fn single(note: Note, hashrate_mhs: f64) -> Delivery {
        Delivery {
            gpus: note.gpu.into_iter().collect(),
            note,
            folded: 0,
            hashrate_mhs,
            time: unix_now(),
        }
    }

    fn gpu_list(&self) -> String {
        self.gpus
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn message(&self, worker: &str) -> String {
        let gpu = self.note.gpu.map_or_else(String::new, |g| format!("gpu{g}: "));
        let on = if self.gpus.len() > 1 {
            let names: Vec<String> = self.gpus.iter().map(|g| format!("gpu{g}")).collect();
            format!(", on {}", names.join(" "))
        } else {
            String::new()
        };
        let more = match self.folded {
            0 => String::new(),
            1 => format!(" (and 1 more like it{on})"),
            n => format!(" (and {n} more like it{on})"),
        };
        clean(
            &format!("{worker}: {gpu}{}{more}", self.note.detail),
            1800,
        )
    }

    fn json(&self, worker: &str) -> Vec<u8> {
        let msg = self.message(worker);
        json!({
            "event": self.note.kind.name(),
            "message": msg,
            "detail": clean(&self.note.detail, 1800),
            "gpu": self.note.gpu,
            "gpus": self.gpus,
            "worker": worker,
            "hashrate_mhs": (self.hashrate_mhs * 100.0).round() / 100.0,
            "folded": self.folded,
            "time": self.time,
            "text": msg,
            "content": msg,
        })
        .to_string()
        .into_bytes()
    }

    fn env(&self, worker: &str) -> Vec<(&'static str, String)> {
        vec![
            ("QKM_EVENT", self.note.kind.name().to_string()),
            ("QKM_MESSAGE", self.message(worker)),
            ("QKM_DETAIL", clean(&self.note.detail, 1800)),
            (
                "QKM_GPU",
                self.note.gpu.map_or_else(String::new, |g| g.to_string()),
            ),
            ("QKM_GPUS", self.gpu_list()),
            ("QKM_WORKER", worker.to_string()),
            ("QKM_HASHRATE_MHS", format!("{:.2}", self.hashrate_mhs)),
            ("QKM_FOLDED", self.folded.to_string()),
            ("QKM_TIME", self.time.to_string()),
        ]
    }
}

enum Msg {
    Note(Note),
    /// Deliver what is held back, then stop.
    Finish,
}

/// Where events go: cheap to clone, safe from any thread. A no-op when
/// hooks are off.
#[derive(Clone)]
pub struct Hooks {
    tx: Option<SyncSender<Msg>>,
    cfg: Arc<HooksConfig>,
    dropped: Arc<AtomicU64>,
}

/// The delivery thread.
pub struct Runner {
    tx: SyncSender<Msg>,
    handle: JoinHandle<()>,
}

/// Events waiting for delivery; more are dropped (and counted).
const QUEUE: usize = 256;

fn log(text: &str) {
    crate::output::event(format!("{} {text}", crate::miner::now_hms()));
}

impl Hooks {
    /// Hooks that do nothing.
    pub fn off() -> Hooks {
        Hooks {
            tx: None,
            cfg: Arc::new(HooksConfig::default()),
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Starts delivering per `cfg` for `user` (`0xADDRESS.worker`); the
    /// hashrate comes from `stats`. `None` runner when hooks are off.
    pub fn start(cfg: HooksConfig, user: &str, stats: Arc<Stats>) -> (Hooks, Option<Runner>) {
        if cfg.is_off() {
            return (Hooks::off(), None);
        }
        let tls = match tls_config(webpki_roots()) {
            Ok(t) => t,
            Err(e) => {
                log(&format!("hooks: {e}; hooks are off"));
                return (Hooks::off(), None);
            }
        };
        let cfg = Arc::new(cfg);
        let (tx, rx) = sync_channel(QUEUE);
        let worker = clean(worker_name(user), 64);
        let c = cfg.clone();
        let handle = std::thread::Builder::new()
            .name("hooks".into())
            .spawn(move || deliver_loop(&rx, &c, &worker, &stats, &tls));
        match handle {
            Ok(handle) => (
                Hooks {
                    tx: Some(tx.clone()),
                    cfg,
                    dropped: Arc::new(AtomicU64::new(0)),
                },
                Some(Runner { tx, handle }),
            ),
            Err(e) => {
                log(&format!("hooks: cannot start: {e}; hooks are off"));
                (Hooks::off(), None)
            }
        }
    }

    /// Queues `note` if its kind is wanted (never blocks).
    pub fn fire(&self, note: Note) {
        let Some(tx) = &self.tx else { return };
        if !self.cfg.wants(note.kind) {
            return;
        }
        match tx.try_send(Msg::Note(note)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                if self.dropped.fetch_add(1, Ordering::Relaxed) == 0 {
                    log("hooks: the delivery queue is full; dropping events");
                }
            }
            Err(TrySendError::Disconnected(_)) => {}
        }
    }

    /// The settings.
    pub fn config(&self) -> &HooksConfig {
        &self.cfg
    }
}

impl Runner {
    /// Delivers what is queued and held back, waiting at most `wait`.
    pub fn finish(self, wait: Duration) {
        let deadline = Instant::now() + wait;
        // The queue may be full; keep trying until the deadline.
        let mut msg = Msg::Finish;
        loop {
            match self.tx.try_send(msg) {
                Ok(()) => break,
                Err(TrySendError::Full(m)) if Instant::now() < deadline => {
                    msg = m;
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => return,
            }
        }
        while !self.handle.is_finished() {
            if Instant::now() >= deadline {
                log("hooks: delivery did not finish in time; going on without it");
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.handle.join();
    }
}

/// An event held back by the folding: the latest, and how many before it.
struct Held {
    d: Delivery,
    /// When it arrived (delivery order at a flush).
    at: Instant,
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn deliver_loop(
    rx: &Receiver<Msg>,
    cfg: &HooksConfig,
    worker: &str,
    stats: &Stats,
    tls: &Arc<rustls::ClientConfig>,
) {
    let mut last: HashMap<Kind, Instant> = HashMap::new();
    let mut held: HashMap<Kind, Held> = HashMap::new();
    let send = |d: &Delivery| deliver(cfg, worker, d, tls);
    // Everything held back, oldest first.
    let flush = |held: &mut HashMap<Kind, Held>, last: &mut HashMap<Kind, Instant>| {
        let mut rest: Vec<(Kind, Held)> = held.drain().collect();
        rest.sort_by_key(|(_, h)| h.at);
        for (k, h) in rest {
            last.insert(k, Instant::now());
            send(&h.d);
        }
    };
    loop {
        // Held events whose interval has passed, oldest first.
        let now = Instant::now();
        let mut due: Vec<(Kind, Instant)> = held
            .iter()
            .filter(|(k, _)| last.get(k).is_none_or(|t| *t + cfg.interval <= now))
            .map(|(k, h)| (*k, h.at))
            .collect();
        due.sort_by_key(|(_, at)| *at);
        for (k, _) in due {
            if let Some(h) = held.remove(&k) {
                last.insert(k, Instant::now());
                send(&h.d);
            }
        }
        let msg = match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(m) => m,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => Msg::Finish,
        };
        let note = match msg {
            Msg::Note(n) => n,
            Msg::Finish => {
                flush(&mut held, &mut last);
                return;
            }
        };
        let rate = stats.hashrate(Duration::from_secs(60)) / 1e6;
        let kind = note.kind;
        if kind == Kind::MinerStopping {
            // The last word: what was held back goes first.
            flush(&mut held, &mut last);
            send(&Delivery::single(note, rate));
            continue;
        }
        let recent = last
            .get(&kind)
            .is_some_and(|t| t.elapsed() < cfg.interval);
        if !recent {
            last.insert(kind, Instant::now());
            send(&Delivery::single(note, rate));
            continue;
        }
        let (folded, mut gpus) = match held.remove(&kind) {
            Some(h) => (h.d.folded + 1, h.d.gpus),
            None => (0, Vec::new()),
        };
        if let Some(g) = note.gpu
            && !gpus.contains(&g)
        {
            gpus.push(g);
            gpus.sort_unstable();
        }
        held.insert(
            kind,
            Held {
                d: Delivery {
                    note,
                    folded,
                    gpus,
                    hashrate_mhs: rate,
                    time: unix_now(),
                },
                at: Instant::now(),
            },
        );
    }
}

fn deliver(cfg: &HooksConfig, worker: &str, d: &Delivery, tls: &Arc<rustls::ClientConfig>) {
    if let Some(cmd) = &cfg.command
        && let Err(e) = run_command(cmd, d, worker, cfg.timeout)
    {
        log(&format!(
            "hooks: {} for {}: {e}",
            cmd.display(),
            d.note.kind.name()
        ));
    }
    if let Some(w) = &cfg.webhook {
        let (body, ctype) = match cfg.format {
            Format::Json => (d.json(worker), "application/json"),
            Format::Text => (d.message(worker).into_bytes(), "text/plain; charset=utf-8"),
        };
        let title = format!("quai-kawpow-miner {worker}");
        match post(w, ctype, &title, &body, cfg.timeout, tls) {
            Ok(code) if (200..300).contains(&code) => {}
            Ok(code) => log(&format!(
                "hooks: {} answered HTTP {code} for {}",
                w.shown(),
                d.note.kind.name()
            )),
            Err(e) => log(&format!(
                "hooks: {} for {}: {e}",
                w.shown(),
                d.note.kind.name()
            )),
        }
    }
}

/// Runs the program with the event in its environment; it gets `timeout`.
fn run_command(
    cmd: &std::path::Path,
    d: &Delivery,
    worker: &str,
    timeout: Duration,
) -> Result<(), String> {
    let mut child = Command::new(cmd)
        .arg(d.note.kind.name())
        .envs(d.env(worker))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot run: {e}"))?;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(status)) => return Err(format!("exited with {status}")),
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "still running after {} s; killed",
                    timeout.as_secs()
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(e) => return Err(format!("wait: {e}")),
        }
    }
}

fn webpki_roots() -> rustls::RootCertStore {
    rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    }
}

fn tls_config(roots: rustls::RootCertStore) -> Result<Arc<rustls::ClientConfig>, String> {
    let c = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| format!("TLS setup: {e}"))?
    .with_root_certificates(roots)
    .with_no_client_auth();
    Ok(Arc::new(c))
}

/// Longest response head read (only the status line matters).
const MAX_RESPONSE_HEAD: usize = 8 * 1024;

/// POSTs `body` and returns the HTTP status.
fn post(
    w: &Webhook,
    content_type: &str,
    title: &str,
    body: &[u8],
    timeout: Duration,
    tls: &Arc<rustls::ClientConfig>,
) -> Result<u16, String> {
    let deadline = Instant::now() + timeout;
    let addrs: Vec<_> = (w.host.as_str(), w.port)
        .to_socket_addrs()
        .map_err(|e| format!("resolve {}: {e}", w.host))?
        .collect();
    let mut sock = None;
    let mut last = format!("no address for {}", w.host);
    for a in addrs {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        match TcpStream::connect_timeout(&a, left) {
            Ok(s) => {
                sock = Some(s);
                break;
            }
            Err(e) => last = format!("connect {a}: {e}"),
        }
    }
    let sock = sock.ok_or(last)?;
    let left = deadline
        .saturating_duration_since(Instant::now())
        .max(Duration::from_millis(1));
    let _ = sock.set_read_timeout(Some(left));
    let _ = sock.set_write_timeout(Some(left));
    let head = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: quai-kawpow-miner/{}\r\nContent-Type: {content_type}\r\nTitle: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        w.path,
        w.authority(),
        env!("CARGO_PKG_VERSION"),
        clean(title, 100),
        body.len()
    );
    let mut request = head.into_bytes();
    request.extend_from_slice(body);
    if w.tls {
        let name = rustls::pki_types::ServerName::try_from(w.host.clone())
            .map_err(|e| format!("{:?} is not a host name: {e}", w.host))?;
        let conn = rustls::ClientConnection::new(tls.clone(), name)
            .map_err(|e| format!("TLS setup: {e}"))?;
        let mut s = rustls::StreamOwned::new(conn, sock);
        s.write_all(&request).map_err(|e| format!("send: {e}"))?;
        status_of(&mut s)
    } else {
        let mut s = sock;
        s.write_all(&request).map_err(|e| format!("send: {e}"))?;
        status_of(&mut s)
    }
}

/// The status code from `HTTP/1.x NNN ...`.
fn status_of(r: &mut impl Read) -> Result<u16, String> {
    let mut head = Vec::new();
    let mut buf = [0u8; 512];
    while !head.contains(&b'\n') {
        let n = r.read(&mut buf).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        head.extend_from_slice(&buf[..n]);
        if head.len() > MAX_RESPONSE_HEAD {
            break;
        }
    }
    let line = String::from_utf8_lossy(&head);
    let line = line.lines().next().unwrap_or("");
    let mut parts = line.split_whitespace();
    match (parts.next(), parts.next().and_then(|c| c.parse().ok())) {
        (Some(v), Some(code)) if v.starts_with("HTTP/") => Ok(code),
        _ => Err(format!("not an HTTP answer: {:?}", clean(line, 80))),
    }
}

/// Watches for the events that are a state lasting too long: no usable
/// pool ([`Kind::PoolDown`]) and a low hashrate ([`Kind::HashrateLow`]),
/// and tells failover from return by the pool index.
pub struct Watch {
    down_after: Duration,
    low: Option<LowRate>,
    /// Since when no pool has been usable (`None` while one is).
    down_since: Option<Instant>,
    down_fired: bool,
    low_since: Option<Instant>,
    low_fired: bool,
    pool: Option<usize>,
}

impl Watch {
    /// Starting at `now`, with no pool yet.
    pub fn new(cfg: &HooksConfig, now: Instant) -> Watch {
        Watch {
            down_after: cfg.pool_down_after,
            low: cfg.hashrate_low,
            down_since: Some(now),
            down_fired: false,
            low_since: Some(now),
            low_fired: false,
            pool: None,
        }
    }

    /// Connected to `pools[pool]` at `url`.
    pub fn connected(&mut self, pool: usize, url: &str) -> Vec<Note> {
        let mut out = vec![Note::new(Kind::PoolConnected, format!("connected to {url}"))];
        match (self.pool, pool) {
            (Some(p), 0) if p > 0 => out.push(Note::new(
                Kind::PoolPrimaryBack,
                format!("back on the first pool, {url}"),
            )),
            (prev, p) if p > 0 && prev != Some(p) => out.push(Note::new(
                Kind::PoolFailover,
                format!("failed over to pool {p}, {url}"),
            )),
            _ => {}
        }
        self.pool = Some(pool);
        out
    }

    /// Whether a pool is usable (connected and authorized) at `now`.
    pub fn pool_usable(&mut self, usable: bool, now: Instant) -> Option<Note> {
        if usable {
            let since = self.down_since.take();
            if std::mem::take(&mut self.down_fired) {
                let mins = since.map_or(0, |t| now.duration_since(t).as_secs() / 60);
                return Some(Note::new(
                    Kind::PoolUp,
                    format!("a pool is usable again after {mins} min"),
                ));
            }
            return None;
        }
        let since = *self.down_since.get_or_insert(now);
        if !self.down_fired && now.duration_since(since) >= self.down_after {
            self.down_fired = true;
            return Some(Note::new(
                Kind::PoolDown,
                format!(
                    "no usable pool for {} min",
                    now.duration_since(since).as_secs() / 60
                ),
            ));
        }
        None
    }

    /// The hashrate (hashes per second, over the last minute) at `now`.
    pub fn hashrate(&mut self, rate: f64, now: Instant) -> Option<Note> {
        let low = self.low?;
        if rate >= low.below {
            self.low_since = None;
            if std::mem::take(&mut self.low_fired) {
                return Some(Note::new(
                    Kind::HashrateOk,
                    format!(
                        "hashrate back to {:.1} MH/s (threshold {:.1})",
                        rate / 1e6,
                        low.below / 1e6
                    ),
                ));
            }
            return None;
        }
        let since = *self.low_since.get_or_insert(now);
        if !self.low_fired && now.duration_since(since) >= low.after {
            self.low_fired = true;
            return Some(Note::new(
                Kind::HashrateLow,
                format!(
                    "hashrate {:.1} MH/s, below {:.1} MH/s for {} min",
                    rate / 1e6,
                    low.below / 1e6,
                    now.duration_since(since).as_secs() / 60
                ),
            ));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn event_names_round_trip_and_lists_parse() {
        for k in Kind::ALL {
            assert_eq!(Kind::parse(k.name()), Some(k));
        }
        let all = parse_events(&["all".into()]).unwrap();
        assert_eq!(all.len(), Kind::ALL.len());
        let d = parse_events(&["default".into(), "gpu_error".into()]).unwrap();
        assert!(d.contains(&Kind::GpuFailed) && d.contains(&Kind::GpuError));
        assert!(!d.contains(&Kind::ShareRejected));
        let e = parse_events(&["gpu_fail".into()]).unwrap_err();
        assert!(e.contains("gpu_failed"), "{e}");
    }

    #[test]
    fn webhook_urls() {
        let w = Webhook::parse("https://discord.com/api/webhooks/1/SECRET").unwrap();
        assert_eq!((w.tls, w.port, w.path.as_str()), (true, 443, "/api/webhooks/1/SECRET"));
        assert_eq!(w.authority(), "discord.com");
        assert!(!format!("{w:?}").contains("SECRET"));
        assert!(!w.shown().contains("SECRET"));
        let w = Webhook::parse("http://10.0.0.11:8080").unwrap();
        assert_eq!((w.tls, w.port, w.path.as_str()), (false, 8080, "/"));
        assert_eq!(w.authority(), "10.0.0.11:8080");
        let w = Webhook::parse("http://[::1]:9/x?y=1").unwrap();
        assert_eq!((w.host.as_str(), w.port, w.path.as_str()), ("::1", 9, "/x?y=1"));
        assert_eq!(w.authority(), "[::1]:9");
        assert_eq!(Webhook::parse("http://h?a=1").unwrap().path, "/?a=1");
        for bad in ["ftp://h/", "h/x", "http://", "http://u:p@h/", "http://h:x/", "http://h/a b"] {
            assert!(Webhook::parse(bad).is_err(), "{bad}");
        }
    }

    fn delivery(kind: Kind, gpu: Option<usize>, detail: &str) -> Delivery {
        Delivery {
            note: Note {
                kind,
                gpu,
                detail: detail.into(),
            },
            folded: 2,
            gpus: gpu.into_iter().collect(),
            hashrate_mhs: 186.04,
            time: 1_791_000_000,
        }
    }

    #[test]
    fn payloads() {
        let d = delivery(Kind::GpuTooHot, Some(3), "91C reached the 90C limit;\npaused");
        let msg = d.message("octo0");
        assert_eq!(
            msg,
            "octo0: gpu3: 91C reached the 90C limit; paused (and 2 more like it)"
        );
        let v: serde_json::Value = serde_json::from_slice(&d.json("octo0")).unwrap();
        assert_eq!(v["event"], "gpu_too_hot");
        assert_eq!(v["gpu"], 3);
        assert_eq!(v["content"], msg.as_str());
        assert_eq!(v["text"], msg.as_str());
        assert_eq!(v["hashrate_mhs"], 186.04);
        let env: HashMap<_, _> = d.env("octo0").into_iter().collect();
        assert_eq!(env["QKM_EVENT"], "gpu_too_hot");
        assert_eq!(env["QKM_GPU"], "3");
        assert_eq!(env["QKM_FOLDED"], "2");
        assert_eq!(env["QKM_GPUS"], "3");
        let mut rig = delivery(Kind::GpuTooHot, Some(5), "paused");
        rig.gpus = vec![0, 2, 5];
        assert_eq!(
            rig.message("octo0"),
            "octo0: gpu5: paused (and 2 more like it, on gpu0 gpu2 gpu5)"
        );
        let v: serde_json::Value = serde_json::from_slice(&rig.json("octo0")).unwrap();
        assert_eq!(v["gpus"], serde_json::json!([0, 2, 5]));
        let rig = delivery(Kind::PoolDown, None, "x".repeat(5000).as_str());
        assert!(rig.message("w").chars().count() <= 1800);
        assert_eq!(rig.env("w")[3], ("QKM_GPU", String::new()));
        assert_eq!(worker_name("0xab.rig1"), "rig1");
    }

    #[test]
    fn watch_pool_down_and_up() {
        let cfg = HooksConfig::default();
        let t0 = Instant::now();
        let mut w = Watch::new(&cfg, t0);
        let s = |n| t0 + Duration::from_secs(n);
        assert_eq!(w.pool_usable(false, s(60)), None);
        let down = w.pool_usable(false, s(121)).unwrap();
        assert_eq!(down.kind, Kind::PoolDown);
        assert_eq!(w.pool_usable(false, s(500)), None, "once");
        assert_eq!(w.pool_usable(true, s(600)).unwrap().kind, Kind::PoolUp);
        assert_eq!(w.pool_usable(true, s(601)), None);
        // A short drop says nothing.
        assert_eq!(w.pool_usable(false, s(700)), None);
        assert_eq!(w.pool_usable(true, s(760)), None);
    }

    #[test]
    fn watch_failover_and_return() {
        let cfg = HooksConfig::default();
        let mut w = Watch::new(&cfg, Instant::now());
        let kinds = |v: Vec<Note>| v.into_iter().map(|n| n.kind).collect::<Vec<_>>();
        assert_eq!(kinds(w.connected(0, "a")), [Kind::PoolConnected]);
        assert_eq!(kinds(w.connected(0, "a")), [Kind::PoolConnected]);
        assert_eq!(kinds(w.connected(1, "b")), [Kind::PoolConnected, Kind::PoolFailover]);
        assert_eq!(kinds(w.connected(1, "b")), [Kind::PoolConnected]);
        assert_eq!(kinds(w.connected(2, "c")), [Kind::PoolConnected, Kind::PoolFailover]);
        assert_eq!(kinds(w.connected(0, "a")), [Kind::PoolConnected, Kind::PoolPrimaryBack]);
        // Starting on a backup (the first pool never answered) is a failover.
        let mut w = Watch::new(&cfg, Instant::now());
        assert_eq!(kinds(w.connected(1, "b")), [Kind::PoolConnected, Kind::PoolFailover]);
    }

    #[test]
    fn watch_hashrate() {
        let cfg = HooksConfig {
            hashrate_low: Some(LowRate {
                below: 100e6,
                after: Duration::from_secs(300),
            }),
            ..HooksConfig::default()
        };
        let t0 = Instant::now();
        let s = |n| t0 + Duration::from_secs(n);
        let mut w = Watch::new(&cfg, t0);
        // Warming up (DAG build) counts, but is short.
        assert_eq!(w.hashrate(0.0, s(30)), None);
        assert_eq!(w.hashrate(150e6, s(60)), None);
        assert_eq!(w.hashrate(90e6, s(100)), None);
        assert_eq!(w.hashrate(90e6, s(399)), None);
        let low = w.hashrate(90e6, s(400)).unwrap();
        assert_eq!(low.kind, Kind::HashrateLow);
        assert!(low.detail.contains("90.0 MH/s, below 100.0"), "{}", low.detail);
        assert_eq!(w.hashrate(80e6, s(900)), None, "once");
        assert_eq!(w.hashrate(120e6, s(905)).unwrap().kind, Kind::HashrateOk);
        assert_eq!(w.hashrate(120e6, s(910)), None);
        // Off without a threshold.
        let mut w = Watch::new(&HooksConfig::default(), t0);
        assert_eq!(w.hashrate(0.0, s(10_000)), None);
    }

    /// A one-shot HTTP server: answers `status` and returns the request.
    fn http_server(status: u16) -> (u16, std::thread::JoinHandle<String>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            let req = read_request(&mut c);
            let _ = c.write_all(format!("HTTP/1.1 {status} X\r\nContent-Length: 0\r\n\r\n").as_bytes());
            req
        });
        (port, h)
    }

    /// Reads a request head and its Content-Length body.
    fn read_request(c: &mut impl Read) -> String {
        let mut got = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = c.read(&mut buf).unwrap();
            got.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&got).to_string();
            if let Some(i) = text.find("\r\n\r\n") {
                let len: usize = text
                    .lines()
                    .find_map(|l| l.strip_prefix("Content-Length: "))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                if got.len() >= i + 4 + len {
                    return text;
                }
            }
            if n == 0 {
                return text;
            }
        }
    }

    #[test]
    fn posts_over_http() {
        let tls = tls_config(webpki_roots()).unwrap();
        let (port, h) = http_server(204);
        let w = Webhook::parse(&format!("http://127.0.0.1:{port}/hook/SECRET?x=1")).unwrap();
        let d = delivery(Kind::GpuFailed, Some(1), "given up");
        let code = post(&w, "application/json", "t", &d.json("octo0"), Duration::from_secs(5), &tls).unwrap();
        assert_eq!(code, 204);
        let req = h.join().unwrap();
        assert!(req.starts_with("POST /hook/SECRET?x=1 HTTP/1.1\r\n"), "{req}");
        assert!(req.contains(&format!("Host: 127.0.0.1:{port}\r\n")), "{req}");
        assert!(req.contains("Content-Type: application/json\r\n"), "{req}");
        let body = req.split("\r\n\r\n").nth(1).unwrap();
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(v["event"], "gpu_failed");

        let (port, h) = http_server(500);
        let w = Webhook::parse(&format!("http://127.0.0.1:{port}/")).unwrap();
        assert_eq!(post(&w, "text/plain", "t", b"hi", Duration::from_secs(5), &tls), Ok(500));
        h.join().unwrap();
    }

    #[test]
    fn posts_over_https_and_checks_the_certificate() {
        let ck = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let cert = ck.cert.der().clone();
        let key = rustls::pki_types::PrivateKeyDer::Pkcs8(ck.signing_key.serialize_der().into());
        let server = Arc::new(
            rustls::ServerConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert.clone()], key)
            .unwrap(),
        );
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            let mut bodies = Vec::new();
            for _ in 0..2 {
                let (c, _) = l.accept().unwrap();
                let conn = rustls::ServerConnection::new(server.clone()).unwrap();
                let mut s = rustls::StreamOwned::new(conn, c);
                let _ = s.sock.set_read_timeout(Some(Duration::from_secs(5)));
                let req = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    read_request(&mut s)
                }));
                if let Ok(req) = req {
                    let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                    let _ = s.flush();
                    bodies.push(req);
                }
            }
            bodies
        });
        let w = Webhook::parse(&format!("https://localhost:{port}/x")).unwrap();
        // Mozilla's roots: the self-signed certificate is refused.
        let public = tls_config(webpki_roots()).unwrap();
        let e = post(&w, "text/plain", "t", b"no", Duration::from_secs(5), &public).unwrap_err();
        assert!(e.contains("certificate") || e.contains("UnknownIssuer"), "{e}");
        // Trusting it: delivered.
        let mut roots = rustls::RootCertStore::empty();
        roots.add(cert).unwrap();
        let own = tls_config(roots).unwrap();
        let code = post(&w, "text/plain", "t", b"hello", Duration::from_secs(5), &own).unwrap();
        assert_eq!(code, 200);
        let got = h.join().unwrap();
        assert!(got.iter().any(|r| r.ends_with("\r\n\r\nhello")), "{got:?}");
    }

    #[test]
    fn a_dead_webhook_fails_within_the_timeout() {
        // A listener that never answers.
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let tls = tls_config(webpki_roots()).unwrap();
        let w = Webhook::parse(&format!("http://127.0.0.1:{port}/")).unwrap();
        let t = Instant::now();
        let e = post(&w, "text/plain", "t", b"x", Duration::from_millis(500), &tls).unwrap_err();
        assert!(t.elapsed() < Duration::from_secs(3), "{e}");
        drop(l);
    }

    fn script(dir: &std::path::Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let p = dir.join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    #[test]
    fn runs_the_command_with_the_event() {
        let dir = std::env::temp_dir().join(format!("qkm-hooks-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("out");
        let ok = script(
            &dir,
            "ok.sh",
            &format!(
                "echo \"$1|$QKM_EVENT|$QKM_GPU|$QKM_WORKER|$QKM_MESSAGE\" >> {}",
                out.display()
            ),
        );
        let d = delivery(Kind::GpuReset, Some(4), "device reset");
        run_command(&ok, &d, "octo0", Duration::from_secs(5)).unwrap();
        let got = std::fs::read_to_string(&out).unwrap();
        assert_eq!(
            got.trim(),
            "gpu_reset|gpu_reset|4|octo0|octo0: gpu4: device reset (and 2 more like it)"
        );
        let fails = script(&dir, "fails.sh", "exit 3");
        let e = run_command(&fails, &d, "w", Duration::from_secs(5)).unwrap_err();
        assert!(e.contains("exit status: 3"), "{e}");
        let slow = script(&dir, "slow.sh", "sleep 30");
        let t = Instant::now();
        let e = run_command(&slow, &d, "w", Duration::from_millis(300)).unwrap_err();
        assert!(e.contains("killed") && t.elapsed() < Duration::from_secs(3), "{e}");
        let e = run_command(&dir.join("missing"), &d, "w", Duration::from_secs(1)).unwrap_err();
        assert!(e.contains("cannot run"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The whole path: events through the queue, folding, and the last
    /// held one delivered at finish.
    #[test]
    fn delivers_folds_and_flushes() {
        let dir = std::env::temp_dir().join(format!("qkm-hooks-run-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("out");
        let cmd = script(
            &dir,
            "log.sh",
            &format!(
                "echo \"$QKM_EVENT $QKM_GPU $QKM_FOLDED $QKM_GPUS $QKM_DETAIL\" >> {}",
                out.display()
            ),
        );
        let cfg = HooksConfig {
            command: Some(cmd),
            events: Some(parse_events(&["all".into()]).unwrap()),
            interval: Duration::from_secs(60),
            ..HooksConfig::default()
        };
        let stats = Arc::new(Stats::new(&[0, 1]));
        let (hooks, runner) = Hooks::start(cfg, "0xab.rig9", stats);
        for i in 0..4 {
            hooks.fire(Note::gpu(Kind::GpuError, 0, format!("error {i}")));
        }
        hooks.fire(Note::gpu(Kind::GpuError, 1, "other gpu"));
        hooks.fire(Note::gpu(Kind::GpuTooHot, 2, "hot"));
        hooks.fire(Note::new(Kind::MinerStopping, "stopping"));
        // Held back after the last word: delivered at finish.
        hooks.fire(Note::gpu(Kind::GpuTooHot, 3, "hot too"));
        runner.unwrap().finish(Duration::from_secs(10));
        let got = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = got.lines().collect();
        // One message per kind at once; the rest of the GPU errors (on
        // GPUs 0 and 1) folded into one, sent before miner_stopping.
        assert_eq!(
            lines,
            [
                "gpu_error 0 0 0 error 0",
                "gpu_too_hot 2 0 2 hot",
                "gpu_error 1 3 0,1 other gpu",
                "miner_stopping  0  stopping",
                "gpu_too_hot 3 0 3 hot too",
            ],
            "{got}"
        );
        // Off: nothing happens, nothing blocks.
        Hooks::off().fire(Note::new(Kind::MinerStopping, "x"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
