//! The stratum client: one connection at a time, reconnects with backoff,
//! fails over across pools, and reports everything as [`Event`]s.
//!
//! Nothing the server does can crash it: bad lines are logged and skipped,
//! a closed or silent connection is replaced, and a pool that keeps failing
//! hands over to the next one.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::proto::{
    Job, Message, error_text, parse_line, redact, request, submit_params, subscribe_extranonce,
    target_for_difficulty,
};

/// One pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pool {
    /// `host:port` (a `stratum+tcp://` prefix is accepted).
    pub url: String,
    /// `0xADDRESS.worker`.
    pub user: String,
    /// Password, e.g. `auth=<token>,d=0.2`.
    pub pass: String,
}

impl Pool {
    /// The address to connect to.
    pub fn host_port(&self) -> &str {
        let u = self.url.trim();
        let u = u.split_once("://").map_or(u, |(_, rest)| rest);
        let u = u.rsplit_once('@').map_or(u, |(_, hp)| hp);
        u.trim_end_matches('/')
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
    /// After this long on a backup pool, the next reconnect tries the
    /// first pool again.
    pub primary_retry: Duration,
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
}

enum Cmd {
    Submit(Share),
    Line(u64, String),
    Closed(u64, String),
    Stop,
}

enum Pending {
    Subscribe,
    Authorize,
    Submit { seq: u64, sent: Instant },
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

fn send(
    out: &mut TcpStream,
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
    AuthFailed,
}

#[allow(clippy::too_many_lines)]
fn session(
    cfg: &Config,
    pool: &Pool,
    stream: TcpStream,
    session_id: u64,
    rx: &Receiver<Cmd>,
    tx: &Sender<Cmd>,
    events: &Sender<Event>,
) -> End {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(cfg.response_timeout));
    let Ok(read_half) = stream.try_clone() else {
        return End::Reconnect("cannot clone the socket".into());
    };
    let _ = read_half.set_read_timeout(Some(cfg.work_timeout + Duration::from_secs(30)));
    let tx_r = tx.clone();
    let _ = std::thread::Builder::new()
        .name("stratum-read".into())
        .spawn(move || {
            let mut r = BufReader::new(read_half);
            let mut line = String::new();
            loop {
                line.clear();
                match r.read_line(&mut line) {
                    Ok(0) => {
                        let _ = tx_r.send(Cmd::Closed(session_id, "closed by the pool".into()));
                        return;
                    }
                    Ok(_) => {
                        if tx_r
                            .send(Cmd::Line(session_id, line.trim().to_string()))
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx_r.send(Cmd::Closed(session_id, format!("read: {e}")));
                        return;
                    }
                }
            }
        });
    let mut out = stream;
    let mut next_id = 1u64;
    let mut pending: HashMap<u64, Pending> = HashMap::new();
    if let Err(e) = send(
        &mut out,
        &mut next_id,
        &mut pending,
        "mining.subscribe",
        json!([cfg.agent]),
        Pending::Subscribe,
    ) {
        return End::Reconnect(e);
    }
    if let Err(e) = send(
        &mut out,
        &mut next_id,
        &mut pending,
        "mining.authorize",
        json!([pool.user, pool.pass]),
        Pending::Authorize,
    ) {
        return End::Reconnect(e);
    }
    let mut authorized = false;
    let mut last_work = Instant::now();
    let mut no_response = 0u32;
    let end = loop {
        // Unanswered requests.
        let now = Instant::now();
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
                            redact(&line)
                        )));
                        continue;
                    }
                };
                match msg {
                    Message::Response { id, result, error } => match pending.remove(&id) {
                        Some(Pending::Subscribe) => {
                            let x = subscribe_extranonce(&result);
                            let _ = events.send(Event::Extranonce(x));
                        }
                        Some(Pending::Authorize) => {
                            if result == Value::Bool(true) {
                                authorized = true;
                                let _ = events.send(Event::Authorized);
                            } else {
                                let why =
                                    error_text(&error).unwrap_or_else(|| "refused".to_string());
                                let _ = events.send(Event::AuthFailed(why));
                                break End::AuthFailed;
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
                        None => {}
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
    let _ = out.shutdown(Shutdown::Both);
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
    loop {
        if idx != 0 && on_backup_since.is_some_and(|t| t.elapsed() > cfg.primary_retry) {
            idx = 0;
            on_backup_since = None;
        }
        let pool = &cfg.pools[idx];
        let url = pool.host_port().to_string();
        let outcome = match connect(&url, cfg.connect_timeout) {
            Err(e) => Err(e),
            Ok(stream) => {
                session_id += 1;
                let _ = events.send(Event::Connected {
                    pool: idx,
                    url: url.clone(),
                    session: session_id,
                });
                let started = Instant::now();
                let end = session(&cfg, pool, stream, session_id, &rx, &tx, &events);
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
            Ok((End::Reconnect(r), lasted)) => (r, lasted > Duration::from_secs(60), false),
            Ok((End::AuthFailed, _)) => ("authorization refused".into(), false, true),
        };
        let _ = events.send(Event::Disconnected {
            reason: format!("{url}: {reason}"),
        });
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
