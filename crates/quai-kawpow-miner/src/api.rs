//! A tiny read-only HTTP API: `GET /metrics` returns Prometheus metrics,
//! `GET /health` returns `ok`, `GET /view` what the full screen shows (the
//! [`View`] and the recent log lines, for `watch`), any other path the stats
//! as JSON.
//!
//! It never trusts the client: the request head is capped at
//! [`MAX_REQUEST`] and each connection gets [`DEADLINE`] in all, so a slow
//! or endless client cannot hold it or fill memory. No CORS header: web
//! pages the operator visits cannot read the rig's details.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::stats::{Stats, View};

/// What `/view` returns.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Screen {
    /// The stats.
    pub view: View,
    /// The recent log lines, oldest first.
    pub log: Vec<String>,
}

/// Longest request head read.
pub const MAX_REQUEST: usize = 8 * 1024;
/// Time one connection may take, from accept to answer.
pub const DEADLINE: Duration = Duration::from_secs(2);

/// Reads the request head (up to the blank line) within the limits; `None`
/// when the client is too slow, too long or gone.
fn read_head(conn: &mut TcpStream) -> Option<String> {
    let start = Instant::now();
    let mut head = Vec::with_capacity(512);
    let mut buf = [0u8; 1024];
    while !head.windows(4).any(|w| w == b"\r\n\r\n") && !head.windows(2).any(|w| w == b"\n\n") {
        let left = DEADLINE.checked_sub(start.elapsed())?;
        conn.set_read_timeout(Some(left.max(Duration::from_millis(1))))
            .ok()?;
        let n = conn.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        head.extend_from_slice(&buf[..n]);
        if head.len() > MAX_REQUEST {
            return None;
        }
    }
    Some(String::from_utf8_lossy(&head).into_owned())
}

/// The path of a request head's first line (`GET /metrics HTTP/1.1`).
fn path_of(head: &str) -> &str {
    head.lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .map_or("/", |p| p.split('?').next().unwrap_or(p))
}

fn answer(conn: &mut TcpStream, stats: &Stats) {
    let Some(head) = read_head(conn) else {
        return;
    };
    let (ctype, body) = match path_of(&head) {
        "/metrics" => ("text/plain; version=0.0.4", stats.prometheus()),
        "/health" => ("text/plain", "ok\n".to_string()),
        "/view" => (
            "application/json",
            serde_json::to_string(&Screen {
                view: stats.view(),
                log: crate::output::recent(),
            })
            .unwrap_or_default(),
        ),
        _ => ("application/json", stats.json().to_string()),
    };
    let _ = conn.set_write_timeout(Some(DEADLINE));
    let _ = write!(
        conn,
        "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

/// Longest `/view` answer `fetch_screen` reads.
const MAX_SCREEN: u64 = 16 << 20;

/// `GET /view` from the API at `addr` (`host:port`), as `watch` shows it.
pub fn fetch_screen(addr: &str, timeout: Duration) -> Result<Screen, String> {
    use std::net::ToSocketAddrs;
    let mut last = format!("no address for {addr}");
    let mut conn = None;
    for a in addr.to_socket_addrs().map_err(|e| format!("{addr}: {e}"))? {
        match TcpStream::connect_timeout(&a, timeout) {
            Ok(c) => {
                conn = Some(c);
                break;
            }
            Err(e) => last = format!("{a}: {e}"),
        }
    }
    let mut conn = conn.ok_or(last)?;
    let _ = conn.set_read_timeout(Some(timeout));
    let _ = conn.set_write_timeout(Some(timeout));
    conn.write_all(format!("GET /view HTTP/1.0\r\nHost: {addr}\r\n\r\n").as_bytes())
        .map_err(|e| format!("{addr}: {e}"))?;
    let mut raw = Vec::new();
    conn.take(MAX_SCREEN)
        .read_to_end(&mut raw)
        .map_err(|e| format!("{addr}: {e}"))?;
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| format!("{addr}: not an HTTP answer"))?;
    let status = String::from_utf8_lossy(&raw[..split]);
    if !status.lines().next().is_some_and(|l| l.contains(" 200")) {
        return Err(format!(
            "{addr}: {}",
            status.lines().next().unwrap_or("no status")
        ));
    }
    serde_json::from_slice(&raw[split + 4..]).map_err(|e| {
        format!("{addr}: not a quai-kawpow-miner API with /view (an older miner?): {e}")
    })
}

/// Serves the stats on `addr` in a background thread.
pub fn serve(addr: SocketAddr, stats: Arc<Stats>) -> std::io::Result<SocketAddr> {
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    std::thread::Builder::new()
        .name("api".into())
        .spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { continue };
                answer(&mut conn, &stats);
            }
        })?;
    Ok(bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(addr: SocketAddr, req: &[u8]) -> String {
        let mut c = TcpStream::connect(addr).unwrap();
        c.write_all(req).unwrap();
        let mut out = String::new();
        let _ = c.read_to_string(&mut out);
        out
    }

    #[test]
    fn routes_and_limits() {
        let stats = Arc::new(Stats::new(&[0, 1]));
        let addr = serve("127.0.0.1:0".parse().unwrap(), stats).unwrap();
        let json = get(addr, b"GET / HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(json.contains("application/json") && json.contains("\"gpus\""));
        assert!(!json.contains("Access-Control-Allow-Origin"));
        let m = get(addr, b"GET /metrics?x=1 HTTP/1.1\r\n\r\n");
        assert!(m.contains("qkm_hashrate_hashes_per_second"), "{m}");
        assert!(m.contains("qkm_gpu_hashes_total{device=\"1\""), "{m}");
        assert!(get(addr, b"GET /health HTTP/1.0\r\n\r\n").ends_with("ok\n"));
        let v = get(addr, b"GET /view HTTP/1.1\r\n\r\n");
        let body = v.split("\r\n\r\n").nth(1).unwrap();
        let s: Screen = serde_json::from_str(body).unwrap();
        assert_eq!(s.view.gpus.len(), 2);
        let s = fetch_screen(&addr.to_string(), Duration::from_secs(2)).unwrap();
        assert_eq!(s.view.gpus[1].device, 1);
        let health_only = TcpListener::bind("127.0.0.1:0").unwrap();
        let dead = health_only.local_addr().unwrap().to_string();
        drop(health_only);
        assert!(fetch_screen(&dead, Duration::from_secs(1)).is_err());
        // An endless head is cut off, and the API still answers after.
        let mut c = TcpStream::connect(addr).unwrap();
        let _ = c.write_all(&vec![b'a'; 64 * 1024]);
        drop(c);
        // A client that never finishes its head is dropped at the deadline.
        let slow = TcpStream::connect(addr).unwrap();
        let t = Instant::now();
        assert!(get(addr, b"GET / HTTP/1.1\r\n\r\n").contains("\"gpus\""));
        assert!(t.elapsed() < DEADLINE + Duration::from_secs(2));
        drop(slow);
    }
}
