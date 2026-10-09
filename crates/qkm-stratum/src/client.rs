//! The stratum client: one connection at a time, reconnects with backoff,
//! fails over across pools, and reports everything as [`Event`]s.
//!
//! Nothing the server does can crash it: bad lines are logged and skipped,
//! over-long lines end the connection, a closed or silent connection (or an
//! unanswered subscribe or authorize) is replaced, and a pool that keeps
//! failing hands over to the next one. After [`Config::primary_retry`] on a
//! backup, the first pool is probed every minute and taken back once it
//! answers. When every pool has refused authorization (a wrong address,
//! lock or token: nothing a retry fixes), [`Event::Fatal`] ends the client.
//!
//! A pool with [`Pool::tls`] is reached over TLS ([`crate::tls`]); a
//! refused certificate counts as a failed connection, like a refused port.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::proto::{
    Job, MAX_LOGGED, Message, error_text, parse_line, redact, request, sanitize, submit_params,
    subscribe_extranonce, target_for_difficulty,
};
use crate::tls::{Reader, Tls, Writer};

/// Longest line accepted from a pool (Quai's are well under 1 KiB); a
/// longer one ends the connection.
pub const MAX_LINE: usize = 64 * 1024;

/// One pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pool {
    /// `host:port` (a `stratum+tcp://` or `stratum+ssl://` prefix is
    /// accepted; [`Pool::tls`], not the prefix, decides TLS).
    pub url: String,
    /// `0xADDRESS.worker`.
    pub user: String,
    /// Password, e.g. `auth=<token>,d=0.2`.
    pub pass: String,
    /// Connect over TLS with these settings ([`crate::tls::wants_tls`]
    /// tells from the URL whether a pool wants it).
    pub tls: Option<Tls>,
}

impl Pool {
    /// The address to connect to.
    pub fn host_port(&self) -> &str {
        let u = self.url.trim();
        let u = u.split_once("://").map_or(u, |(_, rest)| rest);
        let u = u.rsplit_once('@').map_or(u, |(_, hp)| hp);
        u.trim_end_matches('/')
    }

    /// The host name or IP address, without the port or IPv6 brackets.
    pub fn host(&self) -> &str {
        let hp = self.host_port();
        let h = hp.rsplit_once(':').map_or(hp, |(h, _)| h);
        h.trim_start_matches('[').trim_end_matches(']')
    }

    /// The worker name sent with each share (the whole username).
    pub fn worker(&self) -> &str {
        &self.user
    }
}

/// Client settings.
#[derive(Clone, Debug)]
pub struct Config {
    /// Pools in order of preference.
    pub pools: Vec<Pool>,
    /// The agent string sent with `mining.subscribe`.
    pub agent: String,
    /// TCP connect timeout.
    pub connect_timeout: Duration,
    /// How long a request may wait for its response.
    pub response_timeout: Duration,
    /// Reconnect when no job arrives for this long.
    pub work_timeout: Duration,
    /// Consecutive failures on one pool before trying the next.
    pub failures_before_failover: u32,
    /// After this long on a backup pool, the first pool is probed (every
    /// [`Config::probe_every`]) and taken back once it accepts a connection.
    pub primary_retry: Duration,
    /// How often a backup session probes the first pool.
    pub probe_every: Duration,
    /// Longest wait between reconnect attempts.
    pub max_backoff: Duration,
}

impl Config {
    /// Defaults for `pools`.
    pub fn new(pools: Vec<Pool>) -> Config {
        Config {
            pools,
            agent: format!("quai-kawpow-miner/{}", env!("CARGO_PKG_VERSION")),
            connect_timeout: Duration::from_secs(5),
            response_timeout: Duration::from_secs(10),
            work_timeout: Duration::from_secs(180),
            failures_before_failover: 2,
            primary_retry: Duration::from_secs(600),
            probe_every: Duration::from_secs(60),
            max_backoff: Duration::from_secs(30),
        }
    }
}

/// A solution to submit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Share {
    /// Caller's sequence number, echoed in [`Event::ShareResult`].
    pub seq: u64,
    /// The job it solves.
    pub job: String,
    /// The connection the job came from ([`Event::Connected::session`]).
    pub session: u64,
    /// The full 64-bit nonce.
    pub nonce: u64,
    /// The job's header hash.
    pub header: [u8; 32],
    /// `kawpow_light`'s mix digest, byte for byte.
    pub mix: [u8; 32],
}

/// What happened, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// Connected to `pools[pool]`; `session` numbers connections.
    Connected {
        /// Index into [`Config::pools`].
        pool: usize,
        /// `host:port`.
        url: String,
        /// Connection number.
        session: u64,
        /// Over TLS.
        tls: bool,
    },
    /// The nonce prefix for this connection.
    Extranonce(Vec<u8>),
    /// `mining.authorize` accepted.
    Authorized,
    /// `mining.authorize` refused (the reason, as sent).
    AuthFailed(String),
    /// A new share target (from `set_target` or `set_difficulty`).
    Target([u8; 32]),
    /// A new job.
    Job(Job),
    /// The answer to a submitted share.
    ShareResult {
        /// [`Share::seq`].
        seq: u64,
        /// Accepted by the pool.
        accepted: bool,
        /// The pool's reason, or the client's ("no response", "not
        /// connected", "stale session").
        error: Option<String>,
        /// Time from submit to answer.
        latency: Duration,
    },
    /// The connection ended; the client reconnects on its own.
    Disconnected {
        /// Why.
        reason: String,
    },
    /// Something worth logging (e.g. a line from the pool that was skipped).
    Notice(String),
    /// Every pool refused authorization; the client has stopped.
    Fatal(String),
}

enum Cmd {
    Submit(Share),
    Hashrate(f64),
    Line(u64, String),
    Closed(u64, String),
    PrimaryUp(u64),
    Stop,
}

enum Pending {
    Subscribe { sent: Instant },
    Authorize { sent: Instant },
    Submit { seq: u64, sent: Instant },
    /// A request whose answer does not matter (`eth_submitHashrate`).
    Other,
}

/// Submits shares to a running [`Client`] from another thread.
#[derive(Clone)]
pub struct Submitter {
    tx: Sender<Cmd>,
}

impl Submitter {
    /// Queues a share for submission.
    pub fn submit(&self, share: Share) {
        let _ = self.tx.send(Cmd::Submit(share));
    }

    /// Reports the miner's hashrate (hashes per second) to the pool with
    /// `eth_submitHashrate` (pools display it; Quai nodes only log it).
    pub fn report_hashrate(&self, hashes_per_second: f64) {
        let _ = self.tx.send(Cmd::Hashrate(hashes_per_second));
    }
}

/// A running client.
pub struct Client {
    tx: Sender<Cmd>,
    handle: Option<JoinHandle<()>>,
}

impl Client {
    /// Starts connecting; events go to `events`.
    pub fn start(cfg: Config, events: Sender<Event>) -> Client {
        let (tx, rx) = channel();
        let tx2 = tx.clone();
        let handle = std::thread::Builder::new()
            .name("stratum".into())
            .spawn(move || run(cfg, rx, tx2, events))
            .ok();
        Client { tx, handle }
    }

    /// Queues a share for submission.
    pub fn submit(&self, share: Share) {
        let _ = self.tx.send(Cmd::Submit(share));
    }

    /// A handle other threads can submit through.
    pub fn submitter(&self) -> Submitter {
        Submitter {
            tx: self.tx.clone(),
        }
    }

    /// Stops the client and closes its connection.
    pub fn stop(mut self) {
        let _ = self.tx.send(Cmd::Stop);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Stop);
    }
}

/// The sending side of a connection.
enum Out {
    Plain(TcpStream),
    Tls(Writer),
}

impl Out {
    /// Closes the connection (both ways, which also ends the reader).
    fn close(&mut self) {
        match self {
            Out::Plain(s) => {
                let _ = s.shutdown(Shutdown::Both);
            }
            Out::Tls(w) => w.close(),
        }
    }
}

impl Write for Out {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Out::Plain(s) => s.write(buf),
            Out::Tls(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Out::Plain(s) => s.flush(),
            Out::Tls(w) => w.flush(),
        }
    }
}

/// A connected pool: the socket, and the TLS session over it if any.
struct Link {
    sock: TcpStream,
    tls: Option<crate::tls::Session>,
}

/// Connects to `pool`, with the TLS handshake if it wants TLS.
fn open(pool: &Pool, timeout: Duration) -> Result<Link, String> {
    let sock = connect(pool.host_port(), timeout)?;
    match &pool.tls {
        None => Ok(Link { sock, tls: None }),
        Some(t) => {
            let s = t.connect(pool.host(), sock, timeout)?;
            let sock = s.socket().try_clone().map_err(|e| format!("socket: {e}"))?;
            Ok(Link { sock, tls: Some(s) })
        }
    }
}

fn send(
    out: &mut Out,
    next_id: &mut u64,
    pending: &mut HashMap<u64, Pending>,
    method: &str,
    params: Value,
    kind: Pending,
) -> Result<(), String> {
    let id = *next_id;
    *next_id += 1;
    let mut line = request(id, method, params);
    line.push('\n');
    out.write_all(line.as_bytes())
        .and_then(|()| out.flush())
        .map_err(|e| format!("write: {e}"))?;
    pending.insert(id, kind);
    Ok(())
}

fn connect(url: &str, timeout: Duration) -> Result<TcpStream, String> {
    let addrs = url
        .to_socket_addrs()
        .map_err(|e| format!("resolve {url}: {e}"))?;
    let mut last = format!("no address for {url}");
    for a in addrs {
        match TcpStream::connect_timeout(&a, timeout) {
            Ok(s) => return Ok(s),
            Err(e) => last = format!("connect {a}: {e}"),
        }
    }
    Err(last)
}

/// Waits `d` unless a stop arrives; `true` when stopping. Shares queued in
/// the meantime are answered as not sent.
fn wait(rx: &Receiver<Cmd>, d: Duration, events: &Sender<Event>) -> bool {
    let until = Instant::now() + d;
    loop {
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return false;
        }
        match rx.recv_timeout(left) {
            Ok(Cmd::Stop) | Err(RecvTimeoutError::Disconnected) => return true,
            Ok(Cmd::Submit(s)) => {
                let _ = events.send(Event::ShareResult {
                    seq: s.seq,
                    accepted: false,
                    error: Some("not connected".into()),
                    latency: Duration::ZERO,
                });
            }
            Ok(_) | Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

enum End {
    Stop,
    Reconnect(String),
    AuthFailed(String),
    /// The first pool answers again: go back to it.
    Primary,
}

/// What [`session`] needs to know about failing back to the first pool.
struct Probe {
    /// The first pool's `host:port`.
    url: String,
    /// When to start probing it.
    from: Instant,
}

/// Reads lines of at most [`MAX_LINE`] bytes from the pool into `tx`.
fn read_lines(read_half: impl Read, session_id: u64, tx: &Sender<Cmd>) {
    let mut r = BufReader::new(read_half);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match (&mut r).take(MAX_LINE as u64 + 1).read_until(b'\n', &mut buf) {
            Ok(0) => {
                let _ = tx.send(Cmd::Closed(session_id, "closed by the pool".into()));
                return;
            }
            Ok(_) if buf.len() > MAX_LINE && buf.last() != Some(&b'\n') => {
                let _ = tx.send(Cmd::Closed(
                    session_id,
                    format!("the pool sent a line longer than {} KiB", MAX_LINE / 1024),
                ));
                return;
            }
            Ok(_) => {
                let line = String::from_utf8_lossy(&buf).trim().to_string();
                if tx.send(Cmd::Line(session_id, line)).is_err() {
                    return;
                }
            }
            Err(e) => {
                let _ = tx.send(Cmd::Closed(session_id, format!("read: {e}")));
                return;
            }
        }
    }
}

/// `eth_submitHashrate`'s client id: 32 bytes, fixed for the process.
fn hashrate_id() -> String {
    use std::hash::{BuildHasher, RandomState};
    let s = RandomState::new();
    let words: Vec<String> = (0..4u8)
        .map(|i| format!("{:016x}", s.hash_one((i, std::process::id()))))
        .collect();
    format!("0x{}", words.concat())
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn session(
    cfg: &Config,
    pool: &Pool,
    link: Link,
    session_id: u64,
    probe: Option<Probe>,
    rx: &Receiver<Cmd>,
    tx: &Sender<Cmd>,
    events: &Sender<Event>,
) -> End {
    let Link { sock, tls } = link;
    let _ = sock.set_nodelay(true);
    let _ = sock.set_write_timeout(Some(cfg.response_timeout));
    let Ok(read_half) = sock.try_clone() else {
        return End::Reconnect("cannot clone the socket".into());
    };
    let _ = read_half.set_read_timeout(Some(cfg.work_timeout + Duration::from_secs(30)));
    let (reader, mut out): (Box<dyn Read + Send>, Out) = match tls {
        None => (Box::new(read_half), Out::Plain(sock)),
        Some(s) => {
            let (r, w): (Reader, Writer) = s.split(read_half);
            (Box::new(r), Out::Tls(w))
        }
    };
    let tx_r = tx.clone();
    let _ = std::thread::Builder::new()
        .name("stratum-read".into())
        .spawn(move || read_lines(reader, session_id, &tx_r));
    let mut next_id = 1u64;
    let mut pending: HashMap<u64, Pending> = HashMap::new();
    if let Err(e) = send(
        &mut out,
        &mut next_id,
        &mut pending,
        "mining.subscribe",
        json!([cfg.agent]),
        Pending::Subscribe {
            sent: Instant::now(),
        },
    ) {
        return End::Reconnect(e);
    }
    if let Err(e) = send(
        &mut out,
        &mut next_id,
        &mut pending,
        "mining.authorize",
        json!([pool.user, pool.pass]),
        Pending::Authorize {
            sent: Instant::now(),
        },
    ) {
        return End::Reconnect(e);
    }
    let mut authorized = false;
    let mut last_work = Instant::now();
    let mut no_response = 0u32;
    let mut last_probe: Option<Instant> = None;
    let hr_id = hashrate_id();
    let end = loop {
        // Unanswered requests.
        let now = Instant::now();
        let handshake_late = pending.values().find_map(|p| match p {
            Pending::Subscribe { sent } if now.duration_since(*sent) > cfg.response_timeout => {
                Some("mining.subscribe")
            }
            Pending::Authorize { sent } if now.duration_since(*sent) > cfg.response_timeout => {
                Some("mining.authorize")
            }
            _ => None,
        });
        if let Some(m) = handshake_late {
            break End::Reconnect(format!("no answer to {m}"));
        }
        // On a backup: probe the first pool now and then.
        if let Some(p) = &probe
            && now >= p.from
            && last_probe.is_none_or(|t| now.duration_since(t) >= cfg.probe_every)
        {
            last_probe = Some(now);
            let (url, timeout, tx_p) = (p.url.clone(), cfg.connect_timeout, tx.clone());
            let _ = std::thread::Builder::new()
                .name("stratum-probe".into())
                .spawn(move || {
                    if let Ok(s) = connect(&url, timeout) {
                        let _ = s.shutdown(Shutdown::Both);
                        let _ = tx_p.send(Cmd::PrimaryUp(session_id));
                    }
                });
        }
        let late: Vec<u64> = pending
            .iter()
            .filter_map(|(id, p)| match p {
                Pending::Submit { sent, .. }
                    if now.duration_since(*sent) > cfg.response_timeout =>
                {
                    Some(*id)
                }
                _ => None,
            })
            .collect();
        for id in late {
            if let Some(Pending::Submit { seq, sent }) = pending.remove(&id) {
                no_response += 1;
                let _ = events.send(Event::ShareResult {
                    seq,
                    accepted: false,
                    error: Some("no response".into()),
                    latency: now.duration_since(sent),
                });
            }
        }
        if no_response >= 3 {
            break End::Reconnect("the pool stopped answering shares".into());
        }
        if now.duration_since(last_work) > cfg.work_timeout {
            break End::Reconnect(format!("no job for {} s", cfg.work_timeout.as_secs()));
        }
        let cmd = match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(c) => c,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break End::Stop,
        };
        match cmd {
            Cmd::Stop => break End::Stop,
            Cmd::PrimaryUp(id) if id == session_id => break End::Primary,
            Cmd::PrimaryUp(_) => {}
            Cmd::Hashrate(h) => {
                if authorized && h.is_finite() && h >= 0.0 {
                    let rate = format!("0x{:x}", h as u64);
                    if let Err(e) = send(
                        &mut out,
                        &mut next_id,
                        &mut pending,
                        "eth_submitHashrate",
                        json!([rate, hr_id]),
                        Pending::Other,
                    ) {
                        break End::Reconnect(e);
                    }
                }
            }
            Cmd::Closed(id, reason) if id == session_id => break End::Reconnect(reason),
            Cmd::Closed(..) => {}
            Cmd::Line(id, _) if id != session_id => {}
            Cmd::Line(_, line) => {
                if line.is_empty() {
                    continue;
                }
                let msg = match parse_line(&line) {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = events.send(Event::Notice(format!(
                            "stratum: skipping a line ({e}): {}",
                            sanitize(&redact(&line), MAX_LOGGED)
                        )));
                        continue;
                    }
                };
                match msg {
                    Message::Response { id, result, error } => match pending.remove(&id) {
                        Some(Pending::Subscribe { .. }) => {
                            let x = subscribe_extranonce(&result);
                            let _ = events.send(Event::Extranonce(x));
                        }
                        Some(Pending::Authorize { .. }) => {
                            if result == Value::Bool(true) {
                                authorized = true;
                                let _ = events.send(Event::Authorized);
                            } else {
                                let why =
                                    error_text(&error).unwrap_or_else(|| "refused".to_string());
                                let _ = events.send(Event::AuthFailed(why.clone()));
                                break End::AuthFailed(why);
                            }
                        }
                        Some(Pending::Submit { seq, sent }) => {
                            no_response = 0;
                            let accepted = result == Value::Bool(true) && error.is_null();
                            let _ = events.send(Event::ShareResult {
                                seq,
                                accepted,
                                error: if accepted {
                                    None
                                } else {
                                    Some(error_text(&error).unwrap_or_else(|| "rejected".into()))
                                },
                                latency: sent.elapsed(),
                            });
                        }
                        Some(Pending::Other) | None => {}
                    },
                    Message::Notify(job) => {
                        last_work = Instant::now();
                        if let Some(t) = job.target {
                            let _ = events.send(Event::Target(t));
                        }
                        let _ = events.send(Event::Job(job));
                    }
                    Message::SetTarget(t) => {
                        let _ = events.send(Event::Target(t));
                    }
                    Message::SetDifficulty(d) => {
                        let _ = events.send(Event::Target(target_for_difficulty(d)));
                    }
                    Message::SetExtranonce(x) => {
                        let _ = events.send(Event::Extranonce(x));
                    }
                    Message::Reconnect => {
                        break End::Reconnect("the pool asked to reconnect".into());
                    }
                    Message::Other(_) => {}
                }
            }
            Cmd::Submit(s) => {
                let error = if !authorized {
                    Some("not authorized yet")
                } else if s.session != session_id {
                    Some("stale session")
                } else {
                    None
                };
                if let Some(e) = error {
                    let _ = events.send(Event::ShareResult {
                        seq: s.seq,
                        accepted: false,
                        error: Some(e.into()),
                        latency: Duration::ZERO,
                    });
                    continue;
                }
                let params = submit_params(pool.worker(), &s.job, s.nonce, &s.header, &s.mix);
                if let Err(e) = send(
                    &mut out,
                    &mut next_id,
                    &mut pending,
                    "mining.submit",
                    params,
                    Pending::Submit {
                        seq: s.seq,
                        sent: Instant::now(),
                    },
                ) {
                    let _ = events.send(Event::ShareResult {
                        seq: s.seq,
                        accepted: false,
                        error: Some(e.clone()),
                        latency: Duration::ZERO,
                    });
                    break End::Reconnect(e);
                }
            }
        }
    };
    // Shares still waiting get an answer before the connection goes.
    for (_, p) in pending.drain() {
        if let Pending::Submit { seq, sent } = p {
            let _ = events.send(Event::ShareResult {
                seq,
                accepted: false,
                error: Some("connection closed before an answer".into()),
                latency: sent.elapsed(),
            });
        }
    }
    out.close();
    end
}

fn run(cfg: Config, rx: Receiver<Cmd>, tx: Sender<Cmd>, events: Sender<Event>) {
    if cfg.pools.is_empty() {
        let _ = events.send(Event::Disconnected {
            reason: "no pool configured".into(),
        });
        return;
    }
    let mut idx = 0usize;
    let mut failures = 0u32;
    let mut backoff = Duration::from_secs(1);
    let mut session_id = 0u64;
    let mut on_backup_since: Option<Instant> = None;
    // Pools that refused authorization (and why).
    let mut refused: Vec<Option<String>> = vec![None; cfg.pools.len()];
    loop {
        let pool = &cfg.pools[idx];
        let url = pool.host_port().to_string();
        let outcome = match open(pool, cfg.connect_timeout) {
            Err(e) => Err(e),
            Ok(link) => {
                session_id += 1;
                let _ = events.send(Event::Connected {
                    pool: idx,
                    url: url.clone(),
                    session: session_id,
                    tls: link.tls.is_some(),
                });
                let started = Instant::now();
                let probe = on_backup_since.map(|t| Probe {
                    url: cfg.pools[0].host_port().to_string(),
                    from: t + cfg.primary_retry,
                });
                let end = session(&cfg, pool, link, session_id, probe, &rx, &tx, &events);
                Ok((end, started.elapsed()))
            }
        };
        let (reason, healthy, auth_failed) = match outcome {
            Err(e) => (e, false, false),
            Ok((End::Stop, _)) => {
                let _ = events.send(Event::Disconnected {
                    reason: "stopped".into(),
                });
                return;
            }
            Ok((End::Primary, _)) => {
                let _ = events.send(Event::Disconnected {
                    reason: format!("{url}: the first pool answers again; going back to it"),
                });
                idx = 0;
                on_backup_since = None;
                failures = 0;
                backoff = Duration::from_secs(1);
                continue;
            }
            Ok((End::Reconnect(r), lasted)) => {
                // A session that got going clears an old refusal.
                refused[idx] = None;
                (r, lasted > Duration::from_secs(60), false)
            }
            Ok((End::AuthFailed(why), _)) => {
                refused[idx] = Some(why.clone());
                (format!("authorization refused: {why}"), false, true)
            }
        };
        let _ = events.send(Event::Disconnected {
            reason: format!("{url}: {reason}"),
        });
        if refused.iter().all(Option::is_some) {
            let why: Vec<String> = cfg
                .pools
                .iter()
                .zip(&refused)
                .map(|(p, r)| format!("{}: {}", p.host_port(), r.as_deref().unwrap_or("")))
                .collect();
            let _ = events.send(Event::Fatal(format!(
                "every pool refused authorization ({}); check the address, lock and auth token",
                why.join("; ")
            )));
            return;
        }
        if healthy {
            failures = 0;
            backoff = Duration::from_secs(1);
        } else {
            failures += 1;
        }
        if auth_failed || failures >= cfg.failures_before_failover {
            if cfg.pools.len() > 1 {
                idx = (idx + 1) % cfg.pools.len();
                on_backup_since = (idx != 0).then(Instant::now);
                failures = 0;
            }
            if auth_failed {
                backoff = cfg.max_backoff;
            }
        }
        if wait(&rx, backoff, &events) {
            return;
        }
        if !healthy {
            backoff = (backoff * 2).min(cfg.max_backoff);
        }
    }
}
